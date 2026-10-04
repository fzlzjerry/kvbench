"""Transport-only boundaries for Phase 3 worker audit evidence.

The primary channel retains the complete worker evidence payload.  The raw-index
channel contains only declarations and digests; raw audit bytes remain in the
coordinator-pinned out-of-band directory.  Producer completion admits timing
transport only and never substitutes for coordinator-owned semantic replay.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, TypeAlias

from kvbench.runtime.phase3_audit_operation import (
    Phase3AuditOperationKey,
    validate_phase3_audit_operation_set,
)
from kvbench.runtime.phase3_raw_audit_evidence import (
    PHASE3_RAW_AUDIT_OPERATION_SCHEMA_VERSION,
    RAW_AUDIT_STATUS_COMPLETED,
    RAW_AUDIT_STATUS_FAILED,
    RAW_AUDIT_STATUS_NOT_ATTEMPTED,
    Phase3RawAuditOperationRecord,
    Phase3RawAuditRunIndex,
)

from kvbench.schema.base import (
    canonical_json_bytes,
    require_identifier,
    require_run_id,
    sha256_hex,
)


PHASE3_WORKER_CHANNEL_COMMITMENT_SCHEMA_VERSION = (
    "kvbench-phase3-worker-channel-commitment-1.0.0"
)
PRIMARY_WORKER_EVIDENCE_CHANNEL = "primary_worker_evidence"
RAW_AUDIT_INDEX_CHANNEL = "raw_audit_index"
PHASE3_RAW_AUDIT_OPERATION_PLAN_ENV = (
    "KVBENCH_PHASE3_RAW_AUDIT_OPERATION_PLAN"
)
PHASE3_RAW_AUDIT_OPERATION_PLAN_SCHEMA_VERSION = (
    "kvbench-phase3-raw-audit-operation-plan-1.0.0"
)


class Phase3RawAuditProducerError(RuntimeError):
    """A raw-audit producer plan is incomplete, unsafe, or inconsistent."""


RawAuditOperationProducer: TypeAlias = Callable[
    [Phase3AuditOperationKey, Path], Phase3RawAuditOperationRecord
]


def build_phase3_worker_channel_commitment(
    *,
    run_id: str,
    point_id: str,
    primary_evidence_bytes: bytes,
    raw_audit_index_bytes: bytes,
) -> dict[str, Any]:
    """Bind the roles, exact bytes, and process identity of both channels."""

    require_run_id(run_id)
    require_identifier(point_id, field_name="point_id")
    if type(primary_evidence_bytes) is not bytes or not primary_evidence_bytes:
        raise ValueError("primary worker evidence bytes are absent")
    if type(raw_audit_index_bytes) is not bytes or not raw_audit_index_bytes:
        raise ValueError("raw-audit index channel bytes are absent")
    return {
        "schema_version": PHASE3_WORKER_CHANNEL_COMMITMENT_SCHEMA_VERSION,
        "run_id": run_id,
        "point_id": point_id,
        "channels": {
            PRIMARY_WORKER_EVIDENCE_CHANNEL: {
                "sha256": sha256_hex(primary_evidence_bytes),
                "size_bytes": len(primary_evidence_bytes),
            },
            RAW_AUDIT_INDEX_CHANNEL: {
                "sha256": sha256_hex(raw_audit_index_bytes),
                "size_bytes": len(raw_audit_index_bytes),
            },
        },
    }


def phase3_worker_channel_commitment_sha256(
    *,
    run_id: str,
    point_id: str,
    primary_evidence_bytes: bytes,
    raw_audit_index_bytes: bytes,
) -> str:
    """Return the handshake digest for the exact two-channel commitment."""

    payload = build_phase3_worker_channel_commitment(
        run_id=run_id,
        point_id=point_id,
        primary_evidence_bytes=primary_evidence_bytes,
        raw_audit_index_bytes=raw_audit_index_bytes,
    )
    return sha256_hex(canonical_json_bytes(payload))


def build_phase3_raw_audit_operation_plan_bytes(
    operations: Sequence[Phase3AuditOperationKey],
) -> bytes:
    """Serialize the exact coordinator-owned operation plan canonically."""

    frozen = validate_phase3_audit_operation_set(operations)
    return canonical_json_bytes(
        {
            "schema_version": PHASE3_RAW_AUDIT_OPERATION_PLAN_SCHEMA_VERSION,
            "operations": [operation.to_dict() for operation in frozen],
        }
    )


def parse_phase3_raw_audit_operation_plan_bytes(
    payload: bytes,
) -> tuple[Phase3AuditOperationKey, ...]:
    """Parse a canonical, duplicate-free raw-audit operation plan."""

    if type(payload) is not bytes or not payload:
        raise Phase3RawAuditProducerError("raw-audit operation plan is absent")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise Phase3RawAuditProducerError(
                    "raw-audit operation plan contains a duplicate key"
                )
            result[key] = value
        return result

    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except Phase3RawAuditProducerError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase3RawAuditProducerError(
            "raw-audit operation plan is not valid UTF-8 JSON"
        ) from error
    if type(decoded) is not dict or set(decoded) != {
        "schema_version",
        "operations",
    }:
        raise Phase3RawAuditProducerError(
            "raw-audit operation plan has the wrong fields"
        )
    if (
        decoded["schema_version"]
        != PHASE3_RAW_AUDIT_OPERATION_PLAN_SCHEMA_VERSION
    ):
        raise Phase3RawAuditProducerError(
            "raw-audit operation plan schema differs"
        )
    if type(decoded["operations"]) is not list:
        raise Phase3RawAuditProducerError(
            "raw-audit operation plan operations must be a list"
        )
    try:
        if canonical_json_bytes(decoded) != payload:
            raise Phase3RawAuditProducerError(
                "raw-audit operation plan is not canonical"
            )
        operations = tuple(
            Phase3AuditOperationKey.from_dict(item)
            for item in decoded["operations"]
        )
        return validate_phase3_audit_operation_set(operations)
    except Phase3RawAuditProducerError:
        raise
    except (TypeError, ValueError) as error:
        raise Phase3RawAuditProducerError(
            "raw-audit operation plan is invalid"
        ) from error


class Phase3RawAuditProducerRegistry:
    """One-shot registry binding every planned operation before collection."""

    def __init__(
        self,
        expected_operations: Sequence[Phase3AuditOperationKey],
    ) -> None:
        self._expected_operations = validate_phase3_audit_operation_set(
            expected_operations
        )
        self._expected_by_fingerprint = {
            operation.operation_fingerprint_sha256: operation
            for operation in self._expected_operations
        }
        self._producers: dict[str, RawAuditOperationProducer] = {}
        self._collection_started = False

    @property
    def expected_operations(self) -> tuple[Phase3AuditOperationKey, ...]:
        return self._expected_operations

    def register(
        self,
        operation: Phase3AuditOperationKey,
        producer: RawAuditOperationProducer,
    ) -> None:
        """Register exactly one callable for one planned operation."""

        if self._collection_started:
            raise Phase3RawAuditProducerError(
                "raw-audit producer registration is sealed"
            )
        if type(operation) is not Phase3AuditOperationKey:
            raise TypeError("raw-audit producer operation has the wrong type")
        if not callable(producer):
            raise TypeError("raw-audit producer must be callable")
        fingerprint = operation.operation_fingerprint_sha256
        if self._expected_by_fingerprint.get(fingerprint) != operation:
            raise Phase3RawAuditProducerError(
                "raw-audit producer operation is outside the trusted plan"
            )
        if fingerprint in self._producers:
            raise Phase3RawAuditProducerError(
                "raw-audit producer is already registered"
            )
        self._producers[fingerprint] = producer

    def require_complete_registration(
        self,
    ) -> tuple[RawAuditOperationProducer, ...]:
        """Return producers in plan order only with exact coverage."""

        missing = tuple(
            operation.operation_fingerprint_sha256
            for operation in self._expected_operations
            if operation.operation_fingerprint_sha256 not in self._producers
        )
        if missing:
            raise Phase3RawAuditProducerError(
                "raw-audit producers are incomplete before measurement"
            )
        if len(self._producers) != len(self._expected_operations):
            raise Phase3RawAuditProducerError(
                "raw-audit producer registry contains extra entries"
            )
        return tuple(
            self._producers[operation.operation_fingerprint_sha256]
            for operation in self._expected_operations
        )

    @staticmethod
    def _validate_empty_private_root(raw_root: Path) -> os.stat_result:
        if not isinstance(raw_root, Path) or not raw_root.is_absolute():
            raise Phase3RawAuditProducerError(
                "raw-audit producer root must be an absolute Path"
            )
        try:
            metadata = raw_root.lstat()
        except OSError as error:
            raise Phase3RawAuditProducerError(
                "raw-audit producer root is absent"
            ) from error
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or any(raw_root.iterdir())
        ):
            raise Phase3RawAuditProducerError(
                "raw-audit producer root must be empty, private, and owned"
            )
        return metadata

    @staticmethod
    def _validate_record(
        operation: Phase3AuditOperationKey,
        record: Phase3RawAuditOperationRecord,
    ) -> None:
        if type(record) is not Phase3RawAuditOperationRecord:
            raise Phase3RawAuditProducerError(
                "raw-audit producer returned the wrong record type"
            )
        if record.operation != operation:
            raise Phase3RawAuditProducerError(
                "raw-audit producer returned a different operation"
            )
        if record.status == RAW_AUDIT_STATUS_NOT_ATTEMPTED:
            raise Phase3RawAuditProducerError(
                "a called raw-audit producer cannot return unattempted"
            )
        expected_directory = f"step-{operation.decode_step:04d}"
        if any(
            not declaration.path
            or PurePosixPath(declaration.path).parts[0] != expected_directory
            for declaration in record.files
        ):
            raise Phase3RawAuditProducerError(
                "raw-audit producer declared a path outside its operation"
            )

    def collect(self, raw_root: Path) -> Phase3RawAuditRunIndex:
        """Invoke each registered producer once, preserving a failure tail."""

        if self._collection_started:
            raise Phase3RawAuditProducerError(
                "raw-audit producer collection cannot be retried"
            )
        producers = self.require_complete_registration()
        root_metadata = self._validate_empty_private_root(raw_root)
        self._collection_started = True
        records: list[Phase3RawAuditOperationRecord] = []
        failure_seen = False
        for operation, producer in zip(
            self._expected_operations,
            producers,
            strict=True,
        ):
            if failure_seen:
                records.append(
                    Phase3RawAuditOperationRecord(
                        schema_version=(
                            PHASE3_RAW_AUDIT_OPERATION_SCHEMA_VERSION
                        ),
                        operation=operation,
                        status=RAW_AUDIT_STATUS_NOT_ATTEMPTED,
                        failure_reason="prior_operation_failed",
                        files=(),
                    )
                )
                continue
            record = producer(operation, raw_root)
            self._validate_record(operation, record)
            records.append(record)
            failure_seen = record.status == RAW_AUDIT_STATUS_FAILED
        try:
            current_root_metadata = raw_root.lstat()
        except OSError as error:
            raise Phase3RawAuditProducerError(
                "raw-audit producer root vanished during collection"
            ) from error
        if (
            current_root_metadata.st_dev != root_metadata.st_dev
            or current_root_metadata.st_ino != root_metadata.st_ino
            or current_root_metadata.st_uid != root_metadata.st_uid
            or current_root_metadata.st_mode != root_metadata.st_mode
        ):
            raise Phase3RawAuditProducerError(
                "raw-audit producer root changed during collection"
            )
        return Phase3RawAuditRunIndex.create(tuple(records))


def require_phase3_raw_audit_measurement_admission(
    index: Phase3RawAuditRunIndex,
    expected_operations: Sequence[Phase3AuditOperationKey],
) -> None:
    """Admit timing transport only after every producer completes collection.

    This does not evaluate GQA or allocator semantics; the coordinator's
    independent replay remains authoritative.
    """

    if type(index) is not Phase3RawAuditRunIndex:
        raise TypeError("raw-audit measurement admission requires a run index")
    expected = validate_phase3_audit_operation_set(expected_operations)
    observed = tuple(record.operation for record in index.records)
    if observed != expected:
        raise Phase3RawAuditProducerError(
            "raw-audit run index differs from the trusted operation plan"
        )
    failed = next(
        (
            record
            for record in index.records
            if record.status != RAW_AUDIT_STATUS_COMPLETED
        ),
        None,
    )
    if failed is not None:
        raise Phase3RawAuditProducerError(
            "raw-audit collection failed before measurement at "
            f"decode step {failed.operation.decode_step}: "
            f"{failed.failure_reason}"
        )
