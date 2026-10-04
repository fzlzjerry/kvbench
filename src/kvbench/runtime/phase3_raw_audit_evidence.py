"""Strict indexing and no-follow ingestion for Phase 3 raw audit evidence.

This module is deliberately CPU-only.  It does not collect, interpret, copy,
mutate, or finalize evidence.  It validates one complete per-process operation
index and returns checksum-verified immutable source bytes for a later
coordinator-owned append-only copy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from types import MappingProxyType
from typing import ClassVar

from kvbench.errors import SchemaValidationError
from kvbench.runtime.phase3_audit_operation import (
    Phase3AuditOperationKey,
    validate_phase3_audit_operation_set,
)
from kvbench.schema.base import (
    StrictModel,
    canonical_json_bytes,
    require_identifier,
    require_relative_path,
    require_run_id,
    require_schema,
    require_sha256,
)
from kvbench.schema.phase3 import FROZEN_PHASE3_POINT_IDS


PHASE3_RAW_AUDIT_FILE_SCHEMA_VERSION = (
    "kvbench-phase3-raw-audit-file-1.0.0"
)
PHASE3_RAW_AUDIT_OPERATION_SCHEMA_VERSION = (
    "kvbench-phase3-raw-audit-operation-3.0.0"
)
PHASE3_RAW_AUDIT_RUN_INDEX_SCHEMA_VERSION = (
    "kvbench-phase3-raw-audit-run-index-3.0.0"
)

RAW_AUDIT_STATUS_COMPLETED = "completed"
RAW_AUDIT_STATUS_FAILED = "failed"
RAW_AUDIT_STATUS_NOT_ATTEMPTED = "not_attempted_after_failure"
RAW_AUDIT_OPERATION_STATUSES = (
    RAW_AUDIT_STATUS_COMPLETED,
    RAW_AUDIT_STATUS_FAILED,
    RAW_AUDIT_STATUS_NOT_ATTEMPTED,
)
PHASE3_RAW_AUDIT_SESSION_PROVENANCE_KIND = "phase3_session_provenance"

# Completed operations have an exact, closed file-kind set.  Failed operations
# may declare any canonical subset plus additional machine-readable partial
# kinds, because a failure can occur between any two append-only writes.
REQUIRED_COMPLETED_RAW_AUDIT_FILE_KINDS = tuple(
    sorted(
        {
            "b011_audit",
            "b011_gqa_chrome_trace",
            "b011_mha_chrome_trace",
            "b012_allocation_audit",
            "b012_allocator_snapshot",
            "b012_allocator_trace",
        }
    )
)

MAX_RAW_AUDIT_FILE_SIZE_BYTES = 256 * 1024 * 1024
# Decision 0014 binds the run envelope to the frozen maximum of sixteen
# operations and the observed worst-case per-operation bundle, rounded up to
# 72 MiB.  The independent per-file and file-count limits remain unchanged.
MAX_RAW_AUDIT_RUN_SIZE_BYTES = 16 * 72 * 1024 * 1024
MAX_RAW_AUDIT_RUN_INDEX_SIZE_BYTES = 16 * 1024 * 1024
MAX_RAW_AUDIT_FILES_PER_RUN = 512
PHASE3_RAW_AUDIT_EXPECTED_RUN_COUNT = 20
PHASE3_RAW_AUDIT_EXPECTED_FIXED_L_RUN_COUNT = 16
PHASE3_RAW_AUDIT_EXPECTED_GROWING_RUN_COUNT = 4
PHASE3_RAW_AUDIT_EXPECTED_OPERATION_COUNT = 80
_READ_CHUNK_BYTES = 1024 * 1024
_KIND_RE = re.compile(r"\A[a-z][a-z0-9_]{0,127}\Z")
_FAILURE_REASON_RE = re.compile(r"\A[a-z][a-z0-9_.:-]{0,255}\Z")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")


class Phase3RawAuditEvidenceError(RuntimeError):
    """Raw audit evidence is absent, unsafe, changed, or inconsistent."""


def _require_safe_evidence_path(value: str) -> None:
    if type(value) is not str:
        raise ValueError("raw audit evidence path must be a string")
    require_relative_path(value, field_name="raw audit evidence path")
    path = PurePosixPath(value)
    try:
        encoded_length = len(value.encode("utf-8"))
        part_lengths = tuple(len(part.encode("utf-8")) for part in path.parts)
    except UnicodeEncodeError as error:
        raise ValueError("raw audit evidence path must be valid UTF-8") from error
    if (
        encoded_length > 1024
        or any(length > 255 for length in part_lengths)
        or _CONTROL_CHAR_RE.search(value) is not None
    ):
        raise ValueError("raw audit evidence path is unsafe or too long")


def _require_file_kind(value: str) -> None:
    if type(value) is not str or _KIND_RE.fullmatch(value) is None:
        raise ValueError("raw audit evidence kind is malformed")


def _require_failure_reason(value: str | None) -> None:
    if type(value) is not str or _FAILURE_REASON_RE.fullmatch(value) is None:
        raise ValueError("raw audit failure reason must be machine-readable")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase3RawAuditFile(StrictModel):
    """One exact immutable source file declared by path, role, size, and hash."""

    schema_version: str
    path: str
    kind: str
    sha256: str
    size_bytes: int

    SCHEMA_VERSION: ClassVar[str] = PHASE3_RAW_AUDIT_FILE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        _require_safe_evidence_path(self.path)
        _require_file_kind(self.kind)
        require_sha256(self.sha256, field_name="raw audit file SHA-256")
        if (
            type(self.size_bytes) is not int
            or self.size_bytes < 0
            or self.size_bytes > MAX_RAW_AUDIT_FILE_SIZE_BYTES
        ):
            raise ValueError("raw audit file size is outside the hard limit")

    @classmethod
    def from_bytes(
        cls,
        *,
        path: str,
        kind: str,
        payload: bytes,
    ) -> Phase3RawAuditFile:
        """Create a declaration from bytes already held by the producer."""

        if type(payload) is not bytes:
            raise TypeError("raw audit payload must be immutable bytes")
        return cls(
            schema_version=cls.SCHEMA_VERSION,
            path=path,
            kind=kind,
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Phase3RawAuditOperationRecord(StrictModel):
    """Raw-file lifecycle record for one exact Phase 3 decode operation."""

    schema_version: str
    operation: Phase3AuditOperationKey
    status: str
    failure_reason: str | None
    files: tuple[Phase3RawAuditFile, ...]

    SCHEMA_VERSION: ClassVar[str] = PHASE3_RAW_AUDIT_OPERATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        if type(self.operation) is not Phase3AuditOperationKey:
            raise ValueError("operation record requires an exact operation key")
        if type(self.status) is not str or self.status not in (
            RAW_AUDIT_OPERATION_STATUSES
        ):
            raise ValueError("raw audit operation status is invalid")
        if type(self.files) is not tuple or any(
            type(item) is not Phase3RawAuditFile for item in self.files
        ):
            raise ValueError("raw audit files must be an exact tuple")

        paths = tuple(item.path for item in self.files)
        kinds = tuple(item.kind for item in self.files)
        if len(set(paths)) != len(paths):
            raise ValueError("raw audit operation declares duplicate paths")
        if len(set(kinds)) != len(kinds):
            raise ValueError("raw audit operation declares duplicate kinds")
        if tuple(sorted(self.files, key=lambda item: (item.kind, item.path))) != (
            self.files
        ):
            raise ValueError("raw audit file declarations are not canonical")

        if self.status == RAW_AUDIT_STATUS_COMPLETED:
            if self.failure_reason is not None:
                raise ValueError("completed raw audit operation has a failure reason")
            expected_kinds = REQUIRED_COMPLETED_RAW_AUDIT_FILE_KINDS
            expected_with_provenance = tuple(
                sorted(
                    (*expected_kinds, PHASE3_RAW_AUDIT_SESSION_PROVENANCE_KIND)
                )
            )
            allowed_kinds = {expected_kinds}
            if self.operation.decode_step == 0:
                allowed_kinds.add(expected_with_provenance)
            if kinds not in allowed_kinds:
                raise ValueError(
                    "completed raw audit operation file set is incomplete"
                )
            if any(item.size_bytes == 0 for item in self.files):
                raise ValueError("completed raw audit operation has an empty file")
        elif self.status == RAW_AUDIT_STATUS_FAILED:
            _require_failure_reason(self.failure_reason)
        else:
            _require_failure_reason(self.failure_reason)
            if self.files:
                raise ValueError("unattempted raw audit operation cannot declare files")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase3RawAuditRunIndex(StrictModel):
    """Complete ordered raw-audit operation index for exactly one process run."""

    schema_version: str
    run_id: str
    point_id: str
    records: tuple[Phase3RawAuditOperationRecord, ...]

    SCHEMA_VERSION: ClassVar[str] = PHASE3_RAW_AUDIT_RUN_INDEX_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        require_run_id(self.run_id)
        require_identifier(self.point_id, field_name="point_id")
        if type(self.records) is not tuple or not self.records or any(
            type(item) is not Phase3RawAuditOperationRecord
            for item in self.records
        ):
            raise ValueError("raw audit run records must be a nonempty exact tuple")

        operations = validate_phase3_audit_operation_set(
            tuple(record.operation for record in self.records)
        )
        if any(
            operation.run_id != self.run_id or operation.point_id != self.point_id
            for operation in operations
        ):
            raise ValueError("raw audit run identity differs from operation keys")

        failure_seen = False
        for record in self.records:
            if not failure_seen:
                if record.status == RAW_AUDIT_STATUS_FAILED:
                    failure_seen = True
                elif record.status != RAW_AUDIT_STATUS_COMPLETED:
                    raise ValueError(
                        "raw audit operation was unattempted before any failure"
                    )
            elif record.status != RAW_AUDIT_STATUS_NOT_ATTEMPTED:
                raise ValueError(
                    "every operation after a raw audit failure must be unattempted"
                )

        provenance_records = tuple(
            record
            for record in self.records
            if any(
                item.kind == PHASE3_RAW_AUDIT_SESSION_PROVENANCE_KIND
                for item in record.files
            )
        )
        all_completed = all(
            record.status == RAW_AUDIT_STATUS_COMPLETED
            for record in self.records
        )
        if all_completed:
            if (
                len(provenance_records) != 1
                or provenance_records[0] is not self.records[0]
            ):
                raise ValueError(
                    "completed raw audit run requires one step-zero provenance file"
                )
        elif provenance_records:
            raise ValueError(
                "failed raw audit run cannot claim complete session provenance"
            )

        declared = tuple(
            item
            for record in self.records
            for item in record.files
        )
        if len(declared) > MAX_RAW_AUDIT_FILES_PER_RUN:
            raise ValueError("raw audit run file count exceeds the hard limit")
        paths = tuple(item.path for item in declared)
        if len(set(paths)) != len(paths):
            raise ValueError("raw audit run declares a path more than once")
        path_set = set(paths)
        for value in paths:
            for parent in PurePosixPath(value).parents:
                parent_text = parent.as_posix()
                if parent_text != "." and parent_text in path_set:
                    raise ValueError(
                        "raw audit file path conflicts with a declared directory"
                    )
        if sum(item.size_bytes for item in declared) > (
            MAX_RAW_AUDIT_RUN_SIZE_BYTES
        ):
            raise ValueError("raw audit run size exceeds the hard limit")

    @classmethod
    def create(
        cls,
        records: Sequence[Phase3RawAuditOperationRecord],
    ) -> Phase3RawAuditRunIndex:
        """Build an index while deriving its run and point identity."""

        if isinstance(records, (str, bytes)):
            raise TypeError("raw audit records must be a sequence")
        frozen = tuple(records)
        if not frozen:
            raise ValueError("raw audit run records must be nonempty")
        if any(type(item) is not Phase3RawAuditOperationRecord for item in frozen):
            raise TypeError("raw audit records have the wrong type")
        return cls(
            schema_version=cls.SCHEMA_VERSION,
            run_id=frozen[0].operation.run_id,
            point_id=frozen[0].operation.point_id,
            records=frozen,
        )


def parse_phase3_raw_audit_run_index_bytes(
    payload: bytes,
) -> Phase3RawAuditRunIndex:
    """Parse one canonical, duplicate-free raw-audit index byte string."""

    if type(payload) is not bytes:
        raise TypeError("raw audit run-index payload must be immutable bytes")
    if not payload:
        raise Phase3RawAuditEvidenceError("raw audit run index is absent")
    if len(payload) > MAX_RAW_AUDIT_RUN_INDEX_SIZE_BYTES:
        raise Phase3RawAuditEvidenceError(
            "raw audit run index exceeds the hard byte limit"
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise Phase3RawAuditEvidenceError(
            "raw audit run index is not valid UTF-8"
        ) from error

    def reject_constant(_value: str) -> None:
        raise Phase3RawAuditEvidenceError(
            "raw audit run index contains a non-finite number"
        )

    def parse_finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise Phase3RawAuditEvidenceError(
                "raw audit run index contains a non-finite number"
            )
        return parsed

    def reject_duplicate_keys(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise Phase3RawAuditEvidenceError(
                    "raw audit run index contains a duplicate object key"
                )
            result[key] = value
        return result

    try:
        parsed = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
            parse_float=parse_finite_float,
        )
    except Phase3RawAuditEvidenceError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as error:
        raise Phase3RawAuditEvidenceError(
            "raw audit run index is malformed JSON"
        ) from error
    try:
        canonical = canonical_json_bytes(parsed)
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise Phase3RawAuditEvidenceError(
            "raw audit run index cannot be canonically serialized"
        ) from error
    if canonical != payload:
        raise Phase3RawAuditEvidenceError(
            "raw audit run index is not canonical JSON"
        )
    try:
        return Phase3RawAuditRunIndex.from_dict(parsed)
    except SchemaValidationError as error:
        raise Phase3RawAuditEvidenceError(
            "raw audit run index fails its strict schema"
        ) from error


_CAMPAIGN_COMMON_IDENTITY_FIELDS = (
    "execution_git_sha",
    "hardware_identity_sha256",
    "software_identity_sha256",
    "model_identity_sha256",
    "backend_identity_sha256",
    "source_identity_sha256",
)


def validate_phase3_raw_audit_campaign_indices(
    indices: Sequence[Phase3RawAuditRunIndex],
    *,
    campaign_membership: Mapping[str, str],
    fixed_l_campaign_id: str,
    growing_context_campaign_id: str,
) -> tuple[Phase3RawAuditRunIndex, ...]:
    """Validate exact fresh-campaign membership, coverage, and identities."""

    if isinstance(indices, (str, bytes)) or not isinstance(indices, Sequence):
        raise TypeError("raw audit campaign indices must be a sequence")
    frozen = tuple(indices)
    if len(frozen) != PHASE3_RAW_AUDIT_EXPECTED_RUN_COUNT:
        raise ValueError("raw audit campaign must contain exactly 20 run indices")
    if any(type(index) is not Phase3RawAuditRunIndex for index in frozen):
        raise TypeError("raw audit campaign contains a non-index value")
    for campaign_id in (fixed_l_campaign_id, growing_context_campaign_id):
        if type(campaign_id) is not str:
            raise TypeError("raw audit campaign IDs must be strings")
        require_run_id(campaign_id)
    if fixed_l_campaign_id == growing_context_campaign_id:
        raise ValueError("raw audit campaign IDs must be distinct")
    if not isinstance(campaign_membership, Mapping):
        raise TypeError("raw audit campaign membership must be a mapping")
    if any(
        type(run_id) is not str or type(campaign_id) is not str
        for run_id, campaign_id in campaign_membership.items()
    ):
        raise TypeError("raw audit campaign membership must map strings")

    point_ids = tuple(index.point_id for index in frozen)
    if point_ids != FROZEN_PHASE3_POINT_IDS:
        raise ValueError(
            "raw audit campaign is not the exact ordered frozen 20-point grid"
        )
    run_ids = tuple(index.run_id for index in frozen)
    if len(set(run_ids)) != PHASE3_RAW_AUDIT_EXPECTED_RUN_COUNT:
        raise ValueError("raw audit campaign run IDs must be unique")
    if set(campaign_membership) != set(run_ids):
        raise ValueError(
            "raw audit campaign membership must cover exactly the run IDs"
        )

    fixed = tuple(
        index
        for index in frozen
        if index.records[0].operation.runner_kind.value == "fixed_l"
    )
    growing = tuple(
        index
        for index in frozen
        if index.records[0].operation.runner_kind.value == "growing_context"
    )
    if len(fixed) != PHASE3_RAW_AUDIT_EXPECTED_FIXED_L_RUN_COUNT:
        raise ValueError("raw audit campaign must contain 16 fixed-L runs")
    if len(growing) != PHASE3_RAW_AUDIT_EXPECTED_GROWING_RUN_COUNT:
        raise ValueError("raw audit campaign must contain four growing runs")
    if any(len(index.records) != 1 for index in fixed):
        raise ValueError("each fixed-L run must contain one audit operation")
    if any(len(index.records) != 16 for index in growing):
        raise ValueError("each growing run must contain 16 audit operations")
    operations = tuple(
        record.operation for index in frozen for record in index.records
    )
    if len(operations) != PHASE3_RAW_AUDIT_EXPECTED_OPERATION_COUNT:
        raise ValueError("raw audit campaign must contain exactly 80 operations")
    fingerprints = tuple(
        operation.operation_fingerprint_sha256 for operation in operations
    )
    if len(set(fingerprints)) != PHASE3_RAW_AUDIT_EXPECTED_OPERATION_COUNT:
        raise ValueError("raw audit campaign operation fingerprints must be unique")

    for index in fixed:
        if campaign_membership[index.run_id] != fixed_l_campaign_id:
            raise ValueError("fixed-L run has incorrect campaign membership")
    for index in growing:
        if campaign_membership[index.run_id] != growing_context_campaign_id:
            raise ValueError("growing run has incorrect campaign membership")
    for field in _CAMPAIGN_COMMON_IDENTITY_FIELDS:
        if len({getattr(operation, field) for operation in operations}) != 1:
            raise ValueError(
                f"raw audit campaign mixes operation identity field {field}"
            )
    return frozen


def _metadata_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_mode & 0o7777,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _open_flags(*, directory: bool, nonblocking: bool = False) -> int:
    required = ("O_CLOEXEC", "O_NOFOLLOW")
    if directory:
        required += ("O_DIRECTORY",)
    if any(not hasattr(os, name) for name in required):
        raise Phase3RawAuditEvidenceError(
            "platform lacks required no-follow descriptor flags"
        )
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    if directory:
        flags |= os.O_DIRECTORY
    if nonblocking and hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    return flags


def _expected_directories(paths: Sequence[str]) -> frozenset[str]:
    directories: set[str] = set()
    for value in paths:
        parts = PurePosixPath(value).parts
        for stop in range(1, len(parts)):
            directories.add(PurePosixPath(*parts[:stop]).as_posix())
    return frozenset(directories)


def _read_regular_file(
    directory_fd: int,
    name: str,
    relative: str,
    declaration: Phase3RawAuditFile,
    before: os.stat_result,
    *,
    owner_uid: int,
    max_file_size_bytes: int,
) -> bytes:
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != owner_uid
    ):
        raise Phase3RawAuditEvidenceError(
            f"raw audit evidence file is unsafe: {relative}"
        )
    if before.st_size != declaration.size_bytes:
        raise Phase3RawAuditEvidenceError(
            f"raw audit evidence size changed: {relative}"
        )
    if before.st_size > max_file_size_bytes:
        raise Phase3RawAuditEvidenceError(
            f"raw audit evidence exceeds the file-size limit: {relative}"
        )

    try:
        descriptor = os.open(
            name,
            _open_flags(directory=False, nonblocking=True),
            dir_fd=directory_fd,
        )
    except OSError as error:
        raise Phase3RawAuditEvidenceError(
            f"raw audit evidence file cannot be opened safely: {relative}"
        ) from error
    try:
        opened = os.fstat(descriptor)
        if _metadata_identity(opened) != _metadata_identity(before):
            raise Phase3RawAuditEvidenceError(
                f"raw audit evidence changed while opening: {relative}"
            )
        chunks: list[bytes] = []
        remaining = declaration.size_bytes
        while remaining:
            chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, remaining))
            if not chunk:
                raise Phase3RawAuditEvidenceError(
                    f"raw audit evidence was truncated while reading: {relative}"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise Phase3RawAuditEvidenceError(
                f"raw audit evidence grew while reading: {relative}"
            )
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        if _metadata_identity(after) != _metadata_identity(opened):
            raise Phase3RawAuditEvidenceError(
                f"raw audit evidence changed while reading: {relative}"
            )
    finally:
        os.close(descriptor)

    if hashlib.sha256(payload).hexdigest() != declaration.sha256:
        raise Phase3RawAuditEvidenceError(
            f"raw audit evidence digest changed: {relative}"
        )
    return payload


def _walk_evidence_directory(
    directory_fd: int,
    prefix: str,
    *,
    declarations: Mapping[str, Phase3RawAuditFile],
    expected_directories: frozenset[str],
    owner_uid: int,
    max_file_size_bytes: int,
    retained: dict[str, bytes],
) -> None:
    opened_directory = os.fstat(directory_fd)
    if not stat.S_ISDIR(opened_directory.st_mode):
        raise Phase3RawAuditEvidenceError("raw audit evidence directory is invalid")
    try:
        names: list[str] = []
        maximum_entries = len(declarations) + len(expected_directories)
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                names.append(entry.name)
                if len(names) > maximum_entries:
                    raise Phase3RawAuditEvidenceError(
                        "raw audit evidence tree exceeds its declared entry limit"
                    )
    except OSError as error:
        raise Phase3RawAuditEvidenceError(
            "raw audit evidence directory cannot be enumerated"
        ) from error
    names.sort()

    for name in names:
        if (
            not name
            or "/" in name
            or "\\" in name
            or _CONTROL_CHAR_RE.search(name) is not None
        ):
            raise Phase3RawAuditEvidenceError(
                "raw audit evidence directory contains an unsafe name"
            )
        relative = name if not prefix else f"{prefix}/{name}"
        try:
            before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as error:
            raise Phase3RawAuditEvidenceError(
                f"raw audit evidence entry vanished: {relative}"
            ) from error
        if stat.S_ISLNK(before.st_mode):
            raise Phase3RawAuditEvidenceError(
                f"raw audit evidence contains a symlink: {relative}"
            )
        if stat.S_ISDIR(before.st_mode):
            if relative not in expected_directories:
                raise Phase3RawAuditEvidenceError(
                    f"raw audit evidence contains an undeclared directory: {relative}"
                )
            if before.st_uid != owner_uid:
                raise Phase3RawAuditEvidenceError(
                    f"raw audit evidence directory owner differs: {relative}"
                )
            try:
                child_fd = os.open(
                    name,
                    _open_flags(directory=True),
                    dir_fd=directory_fd,
                )
            except OSError as error:
                raise Phase3RawAuditEvidenceError(
                    f"raw audit evidence directory cannot be opened safely: {relative}"
                ) from error
            try:
                child_opened = os.fstat(child_fd)
                if _metadata_identity(child_opened) != _metadata_identity(before):
                    raise Phase3RawAuditEvidenceError(
                        "raw audit evidence directory changed while opening: "
                        f"{relative}"
                    )
                _walk_evidence_directory(
                    child_fd,
                    relative,
                    declarations=declarations,
                    expected_directories=expected_directories,
                    owner_uid=owner_uid,
                    max_file_size_bytes=max_file_size_bytes,
                    retained=retained,
                )
                child_after = os.fstat(child_fd)
                if _metadata_identity(child_after) != _metadata_identity(
                    child_opened
                ):
                    raise Phase3RawAuditEvidenceError(
                        "raw audit evidence directory changed while reading: "
                        f"{relative}"
                    )
            finally:
                os.close(child_fd)
            continue
        declaration = declarations.get(relative)
        if declaration is None:
            raise Phase3RawAuditEvidenceError(
                f"raw audit evidence contains an undeclared file: {relative}"
            )
        retained[relative] = _read_regular_file(
            directory_fd,
            name,
            relative,
            declaration,
            before,
            owner_uid=owner_uid,
            max_file_size_bytes=max_file_size_bytes,
        )

    directory_after = os.fstat(directory_fd)
    if _metadata_identity(directory_after) != _metadata_identity(opened_directory):
        raise Phase3RawAuditEvidenceError(
            "raw audit evidence directory changed during enumeration"
        )


def _validate_ingestion_request(
    index: Phase3RawAuditRunIndex,
    *,
    max_file_size_bytes: int,
    max_total_size_bytes: int,
    max_file_count: int,
) -> tuple[dict[str, Phase3RawAuditFile], tuple[str, ...]]:
    if type(index) is not Phase3RawAuditRunIndex:
        raise TypeError("index must be a Phase3RawAuditRunIndex")
    for name, value, hard_limit in (
        (
            "max_file_size_bytes",
            max_file_size_bytes,
            MAX_RAW_AUDIT_FILE_SIZE_BYTES,
        ),
        (
            "max_total_size_bytes",
            max_total_size_bytes,
            MAX_RAW_AUDIT_RUN_SIZE_BYTES,
        ),
        (
            "max_file_count",
            max_file_count,
            MAX_RAW_AUDIT_FILES_PER_RUN,
        ),
    ):
        if type(value) is not int or value <= 0 or value > hard_limit:
            raise ValueError(f"{name} must be positive and within the hard limit")

    declared_files = tuple(
        item for record in index.records for item in record.files
    )
    if len(declared_files) > max_file_count:
        raise Phase3RawAuditEvidenceError(
            "raw audit evidence exceeds the file-count limit"
        )
    total_size = sum(item.size_bytes for item in declared_files)
    if total_size > max_total_size_bytes:
        raise Phase3RawAuditEvidenceError(
            "raw audit evidence exceeds the total-size limit"
        )
    if any(item.size_bytes > max_file_size_bytes for item in declared_files):
        raise Phase3RawAuditEvidenceError(
            "raw audit evidence exceeds the file-size limit"
        )
    declarations = {item.path: item for item in declared_files}
    return declarations, tuple(declarations)


def _open_directory_path_no_follow(root: str | Path) -> int:
    """Open every root component relative to a pinned parent descriptor."""

    try:
        supplied = os.fspath(root)
    except (OSError, TypeError, ValueError) as error:
        raise Phase3RawAuditEvidenceError(
            "raw audit evidence root must be an existing real directory"
        ) from error
    if type(supplied) is not str or not supplied:
        raise Phase3RawAuditEvidenceError(
            "raw audit evidence root must be a nonempty text path"
        )
    try:
        absolute = os.path.abspath(supplied)
        descriptor = os.open(os.path.sep, _open_flags(directory=True))
    except (OSError, TypeError, ValueError) as error:
        raise Phase3RawAuditEvidenceError(
            "raw audit evidence root cannot be opened safely"
        ) from error
    try:
        components = tuple(part for part in absolute.split(os.path.sep) if part)
        for component in components:
            try:
                child = os.open(
                    component,
                    _open_flags(directory=True),
                    dir_fd=descriptor,
                )
            except OSError as error:
                raise Phase3RawAuditEvidenceError(
                    "raw audit evidence root contains an unsafe component"
                ) from error
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _ingest_from_open_root(
    root_fd: int,
    declarations: Mapping[str, Phase3RawAuditFile],
    paths: Sequence[str],
    *,
    expected_owner_uid: int,
    max_file_size_bytes: int,
) -> Mapping[str, bytes]:
    try:
        opened_root = os.fstat(root_fd)
    except OSError as error:
        raise Phase3RawAuditEvidenceError(
            "raw audit evidence root descriptor is invalid"
        ) from error
    if (
        not stat.S_ISDIR(opened_root.st_mode)
        or opened_root.st_uid != expected_owner_uid
        or opened_root.st_mode & 0o077
    ):
        raise Phase3RawAuditEvidenceError(
            "raw audit evidence root must be private and expected-owner-owned"
        )

    retained: dict[str, bytes] = {}
    _walk_evidence_directory(
        root_fd,
        "",
        declarations=declarations,
        expected_directories=_expected_directories(paths),
        owner_uid=opened_root.st_uid,
        max_file_size_bytes=max_file_size_bytes,
        retained=retained,
    )
    root_after = os.fstat(root_fd)
    if _metadata_identity(root_after) != _metadata_identity(opened_root):
        raise Phase3RawAuditEvidenceError(
            "raw audit evidence root changed while reading"
        )

    if set(retained) != set(declarations):
        missing = sorted(set(declarations) - set(retained))
        detail = missing[0] if missing else "unknown"
        raise Phase3RawAuditEvidenceError(
            f"raw audit evidence is missing a declared file: {detail}"
        )
    return MappingProxyType(
        {path: retained[path] for path in sorted(retained)}
    )


def ingest_phase3_raw_audit_evidence(
    root: str | Path,
    index: Phase3RawAuditRunIndex,
    *,
    max_file_size_bytes: int = MAX_RAW_AUDIT_FILE_SIZE_BYTES,
    max_total_size_bytes: int = MAX_RAW_AUDIT_RUN_SIZE_BYTES,
    max_file_count: int = MAX_RAW_AUDIT_FILES_PER_RUN,
) -> Mapping[str, bytes]:
    """Read exactly one declared tree through componentwise no-follow opens.

    The returned mapping is read-only and every value is immutable ``bytes``.
    The source root and its contents are never opened for writing.
    """

    declarations, paths = _validate_ingestion_request(
        index,
        max_file_size_bytes=max_file_size_bytes,
        max_total_size_bytes=max_total_size_bytes,
        max_file_count=max_file_count,
    )
    root_fd = _open_directory_path_no_follow(root)
    try:
        return _ingest_from_open_root(
            root_fd,
            declarations,
            paths,
            expected_owner_uid=os.geteuid(),
            max_file_size_bytes=max_file_size_bytes,
        )
    finally:
        os.close(root_fd)


def ingest_phase3_raw_audit_evidence_fd(
    root_fd: int,
    index: Phase3RawAuditRunIndex,
    *,
    expected_owner_uid: int | None = None,
    max_file_size_bytes: int = MAX_RAW_AUDIT_FILE_SIZE_BYTES,
    max_total_size_bytes: int = MAX_RAW_AUDIT_RUN_SIZE_BYTES,
    max_file_count: int = MAX_RAW_AUDIT_FILES_PER_RUN,
) -> Mapping[str, bytes]:
    """Read evidence through a freshly pinned, caller-owned root descriptor."""

    if type(root_fd) is not int or root_fd < 0:
        raise TypeError("raw audit evidence root_fd must be a nonnegative integer")
    if expected_owner_uid is None:
        owner_uid = os.geteuid()
    elif type(expected_owner_uid) is int and expected_owner_uid >= 0:
        owner_uid = expected_owner_uid
    else:
        raise ValueError("expected_owner_uid must be a nonnegative integer")
    declarations, paths = _validate_ingestion_request(
        index,
        max_file_size_bytes=max_file_size_bytes,
        max_total_size_bytes=max_total_size_bytes,
        max_file_count=max_file_count,
    )
    try:
        pinned_root_fd = os.open(
            ".",
            _open_flags(directory=True),
            dir_fd=root_fd,
        )
    except OSError as error:
        raise Phase3RawAuditEvidenceError(
            "raw audit evidence root descriptor cannot be pinned"
        ) from error
    try:
        return _ingest_from_open_root(
            pinned_root_fd,
            declarations,
            paths,
            expected_owner_uid=owner_uid,
            max_file_size_bytes=max_file_size_bytes,
        )
    finally:
        os.close(pinned_root_fd)
