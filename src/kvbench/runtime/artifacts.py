"""Append-only, no-replace run artifact lifecycle for Phase 2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
from typing import Any

from kvbench.errors import (
    ArtifactConflictError,
    ArtifactSafetyError,
    ArtifactStateError,
    ChecksumMismatchError,
    ProvenanceError,
    SchemaValidationError,
)
from kvbench.schema import (
    ArtifactInventory,
    CompletionMarker,
    LifecycleRecord,
    RunManifest,
    RunStatus,
)
from kvbench.schema.base import require_run_id


_STAGING_DIRECTORY = ".kvbench-staging"
_RESERVATION_DIRECTORY = ".kvbench-reservations"
_CONTROL_PATHS = {
    "manifest.initial.json",
    "manifest.json",
    "artifact_inventory.json",
    "checksums.sha256",
    "COMPLETE",
}
_TERMINAL_STATUSES = frozenset(
    status.value for status in RunStatus if status.is_terminal
)
_MUTABLE_MANIFEST_FIELDS = {
    "status",
    "started_at_utc",
    "finished_at_utc",
    "inventory_path",
    "failure_reason",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _as_payload(model: object) -> dict[str, Any]:
    if isinstance(model, Mapping):
        return dict(model)
    to_dict = getattr(model, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return dict(result)
    raise TypeError("schema value must be a mapping or expose to_dict()")


def validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str):
        raise ArtifactSafetyError("run_id must be a string")
    try:
        require_run_id(run_id)
    except ValueError as error:
        raise ArtifactSafetyError(
            "run_id must be 1-128 lowercase safe ASCII characters"
        ) from error
    return run_id


def _path_overlaps(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def validate_artifact_root(
    root: Path,
    *,
    formal_evidence_roots: Sequence[Path],
) -> Path:
    if root.is_symlink():
        raise ArtifactSafetyError("artifact root may not be a symlink")
    resolved = root.expanduser().resolve(strict=False)
    if resolved == Path(resolved.anchor):
        raise ArtifactSafetyError("artifact root may not be a filesystem root")
    for formal in formal_evidence_roots:
        formal_resolved = formal.expanduser().resolve(strict=False)
        if _path_overlaps(resolved, formal_resolved):
            raise ArtifactSafetyError(
                "artifact root overlaps a formal immutable evidence boundary",
            )
    return resolved


def _safe_relative_path(value: str) -> PurePosixPath:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ArtifactSafetyError("artifact path must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ArtifactSafetyError("artifact path must not be absolute or traverse")
    return path


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_real_directory(path: Path, *, mode: int) -> None:
    created = False
    try:
        path.mkdir(mode=mode)
        created = True
    except FileExistsError:
        pass
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ArtifactSafetyError("artifact control directory is unsafe") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ArtifactSafetyError("artifact control path is not a real directory")
    if created:
        _fsync_directory(path.parent)


def _mkdir_parents_no_symlink(root: Path, relative_parent: PurePosixPath) -> None:
    current = root
    for part in relative_parent.parts:
        current = current / part
        try:
            current.mkdir(mode=0o700)
            _fsync_directory(current.parent)
        except FileExistsError:
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise ArtifactSafetyError(
                    "artifact parent contains a symlink or non-directory component",
                ) from None


def _write_exclusive(root: Path, relative: str, data: bytes) -> Path:
    safe = _safe_relative_path(relative)
    _mkdir_parents_no_symlink(root, safe.parent)
    target = root.joinpath(*safe.parts)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(target, flags, 0o600)
    except FileExistsError as error:
        raise ArtifactConflictError("artifact path already exists") from error
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(target.parent)
    return target


def _rename_noreplace(source: Path, target: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = library.renameat2
    except AttributeError as error:
        raise ArtifactStateError(
            "renameat2 is required for no-replace atomic finalization",
        ) from error
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise ArtifactConflictError("final run ID already exists")
    raise OSError(error_number, os.strerror(error_number), str(target))


def _file_role(relative: str) -> str:
    if relative == "manifest.json" or relative == "manifest.initial.json":
        return "manifest"
    if relative.startswith("lifecycle/"):
        return "lifecycle"
    if relative.startswith("logs/"):
        return "log"
    if relative.startswith("raw/"):
        return "raw"
    if relative.startswith("config/"):
        return "configuration"
    if relative.startswith("environment/"):
        return "environment"
    if relative.startswith("telemetry/"):
        return "telemetry"
    if relative.startswith("allocation/"):
        return "allocation"
    if relative.startswith("gqa/"):
        return "gqa"
    if relative.startswith("numerical/"):
        return "numerical"
    if relative.startswith("validation/"):
        return "validation"
    return "artifact"


def _payload_files(stage: Path, exclusions: set[str]) -> list[Path]:
    files: list[Path] = []
    for path in sorted(
        stage.rglob("*"),
        key=lambda candidate: candidate.relative_to(stage).as_posix(),
    ):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ArtifactSafetyError("symlinks are forbidden in run artifacts")
        if not stat.S_ISREG(metadata.st_mode):
            continue
        if metadata.st_nlink != 1:
            raise ArtifactSafetyError("hard-linked files are forbidden in run artifacts")
        relative = path.relative_to(stage).as_posix()
        if relative not in exclusions:
            files.append(path)
    return files


def _validate_manifest(payload: Mapping[str, Any]) -> object:
    try:
        from kvbench.schema import Phase6RunManifest, parse_run_manifest
        from kvbench.schema.phase8 import Phase8RunManifest
        from kvbench.schema.phase9 import Phase9CalibrationManifest
        from kvbench.schema.phase11 import (
            Phase11RQ23RunManifest,
            Phase11RunManifest,
        )
        from kvbench.schema.phase13b import Phase13BBatchAdmissionManifest

        if payload.get("schema_version") == Phase13BBatchAdmissionManifest.SCHEMA_VERSION:
            return Phase13BBatchAdmissionManifest.from_dict(dict(payload))

        if payload.get("schema_version") == Phase11RQ23RunManifest.SCHEMA_VERSION:
            return Phase11RQ23RunManifest.from_dict(dict(payload))
        if payload.get("schema_version") == Phase11RunManifest.SCHEMA_VERSION:
            return Phase11RunManifest.from_dict(dict(payload))
        if payload.get("schema_version") == Phase9CalibrationManifest.SCHEMA_VERSION:
            return Phase9CalibrationManifest.from_dict(dict(payload))
        if payload.get("schema_version") == Phase8RunManifest.SCHEMA_VERSION:
            return Phase8RunManifest.from_dict(dict(payload))
        if payload.get("schema_version") == Phase6RunManifest.SCHEMA_VERSION:
            return Phase6RunManifest.from_dict(dict(payload))
        return parse_run_manifest(dict(payload))
    except SchemaValidationError:
        raise
    except (TypeError, ValueError) as error:
        raise SchemaValidationError("run manifest validation failed") from error


def _status_text(payload: Mapping[str, Any]) -> str:
    status = payload.get("status")
    if isinstance(status, RunStatus):
        return status.value
    if isinstance(status, str):
        return status
    raise SchemaValidationError("manifest.status must be a valid RunStatus")


def _manifest_intent(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in _MUTABLE_MANIFEST_FIELDS
    }


@dataclass(frozen=True)
class RunValidationResult:
    run_dir: Path
    valid: bool
    complete: bool
    status: str | None
    errors: tuple[str, ...]
    artifact_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "valid": self.valid,
            "complete": self.complete,
            "status": self.status,
            "errors": list(self.errors),
            "artifact_count": self.artifact_count,
        }


class AppendOnlyArtifactStore:
    """Factory for uniquely reserved append-only runs."""

    def __init__(
        self,
        root: str | Path,
        *,
        formal_evidence_roots: Sequence[str | Path] | None = None,
    ) -> None:
        repository_root = Path(__file__).resolve().parents[3]
        roots = [repository_root / "docs" / "evidence"]
        if formal_evidence_roots is not None:
            roots.extend(Path(item) for item in formal_evidence_roots)
        self.root = validate_artifact_root(Path(root), formal_evidence_roots=roots)

    def create(self, run_id: str, initial_manifest: object) -> "ArtifactRun":
        safe_id = validate_run_id(run_id)
        payload = _as_payload(initial_manifest)
        parsed = _validate_manifest(payload)
        normalized = _as_payload(parsed)
        if normalized.get("run_id") != safe_id or _status_text(normalized) != "created":
            raise SchemaValidationError(
                "initial manifest must match the run ID and have created status",
            )
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise ArtifactSafetyError("artifact root may not be a symlink")
        root_metadata = self.root.lstat()
        if not stat.S_ISDIR(root_metadata.st_mode):
            raise ArtifactSafetyError("artifact root must be a real directory")
        staging_root = self.root / _STAGING_DIRECTORY
        reservations_root = self.root / _RESERVATION_DIRECTORY
        _ensure_real_directory(staging_root, mode=0o700)
        _ensure_real_directory(reservations_root, mode=0o700)
        final = self.root / safe_id
        if final.exists() or final.is_symlink():
            raise ArtifactConflictError("final run ID already exists")
        if any(staging_root.glob(f"{safe_id}.*.staging")):
            raise ArtifactConflictError(
                "run ID has an existing incomplete staging directory"
            )
        reservation = reservations_root / safe_id
        try:
            reservation.mkdir(mode=0o500)
        except FileExistsError as error:
            raise ArtifactConflictError(
                "run ID is already reserved by a finalized or incomplete run",
            ) from error
        token = secrets.token_hex(12)
        stage = staging_root / f"{safe_id}.{token}.staging"
        try:
            stage.mkdir(mode=0o700)
        except BaseException:
            # The durable reservation deliberately remains as evidence of the
            # attempted ID; callers must choose a new ID rather than reuse it.
            raise
        run = ArtifactRun(
            root=self.root,
            run_id=safe_id,
            stage=stage,
            final=final,
            reservation=reservation,
        )
        run._initialize(normalized)
        return run


class ArtifactRun:
    """A single mutable staging run that can be finalized exactly once."""

    def __init__(
        self,
        *,
        root: Path,
        run_id: str,
        stage: Path,
        final: Path,
        reservation: Path,
    ) -> None:
        self.root = root
        self.run_id = run_id
        self.stage = stage
        self.final = final
        self.reservation = reservation
        self._state = "created"
        self._finalized = False
        self._initial_manifest: dict[str, Any] | None = None

    @property
    def state(self) -> str:
        return self._state

    def _require_state(self, *allowed: str) -> None:
        if self._finalized or self._state not in allowed:
            raise ArtifactStateError(
                "operation is not allowed in the current run state"
            )

    def _initialize(self, initial_manifest: object) -> None:
        payload = _as_payload(initial_manifest)
        parsed = _validate_manifest(payload)
        normalized = _as_payload(parsed)
        if normalized.get("run_id") != self.run_id or _status_text(normalized) != "created":
            raise SchemaValidationError(
                "initial manifest must match the run ID and have created status",
            )
        self._initial_manifest = normalized
        _write_exclusive(self.stage, "manifest.initial.json", _json_bytes(normalized))
        self._write_lifecycle(1, "created")

    def _write_lifecycle(self, sequence: int, state: str) -> None:
        _write_exclusive(
            self.stage,
            f"lifecycle/{sequence:04d}-{state}.json",
            _json_bytes(
                {
                    "schema_version": "kvbench-lifecycle-1.0.0",
                    "run_id": self.run_id,
                    "sequence": sequence,
                    "state": state,
                }
            ),
        )

    def start(self) -> None:
        self._require_state("created")
        self._write_lifecycle(2, "running")
        self._state = "running"

    def write_bytes(self, relative_path: str, data: bytes) -> Path:
        self._require_state("running")
        safe = _safe_relative_path(relative_path).as_posix()
        if safe in _CONTROL_PATHS or safe.startswith("lifecycle/"):
            raise ArtifactSafetyError(
                "artifact path is reserved for lifecycle control"
            )
        if not isinstance(data, bytes):
            raise TypeError("artifact data must be bytes")
        return _write_exclusive(self.stage, safe, data)

    def write_json(self, relative_path: str, payload: Mapping[str, Any]) -> Path:
        return self.write_bytes(relative_path, _json_bytes(payload))

    def finalize(self, final_manifest: object) -> Path:
        self._require_state("running")
        payload = _as_payload(final_manifest)
        parsed = _validate_manifest(payload)
        normalized = _as_payload(parsed)
        status = _status_text(normalized)
        if normalized.get("run_id") != self.run_id or status not in _TERMINAL_STATUSES:
            raise SchemaValidationError(
                "final manifest must match the run ID and use a terminal status",
            )
        if self._initial_manifest is None or _manifest_intent(
            normalized
        ) != _manifest_intent(self._initial_manifest):
            raise ProvenanceError(
                "final manifest changes immutable initial command or provenance",
            )

        from kvbench.schema import Phase3RunManifest, Phase6RunManifest
        from kvbench.schema.phase8 import Phase8RunManifest
        from kvbench.schema.phase9 import Phase9CalibrationManifest
        from kvbench.schema.phase11 import Phase11RunManifest
        from kvbench.schema.phase13b import Phase13BBatchAdmissionManifest

        if isinstance(parsed, Phase3RunManifest):
            required = {
                "config/plan.json",
                "config/referenced_fingerprints.json",
                "environment/live_hardware.json",
                "environment/worker_environment.json",
                "logs/worker.stderr.txt",
                "logs/worker.stdout.txt",
                "validation/point.json",
                "validation/process_audit_outcome.json",
                "validation/worker_result.json",
            }
            if parsed.status is RunStatus.COMPLETED:
                required.update(
                    {
                        "allocation/audit.json",
                        "environment/process.after.json",
                        "environment/process.before.json",
                        "environment/process.during.json",
                        "environment/process.ready.json",
                        "environment/process.release_audit.json",
                        "gqa/audit.json",
                        "numerical/agreement.json",
                        "raw/timing.json",
                        "raw/worker_evidence.json",
                        "telemetry/snapshots.json",
                        "validation/model_identity.json",
                    }
                )
            missing = sorted(
                relative
                for relative in required
                if not (self.stage / relative).is_file()
            )
            if missing:
                raise ArtifactStateError(
                    "Phase 3 finalization lacks required evidence payloads"
                )

        if isinstance(parsed, Phase6RunManifest):
            required = {
                "config/method.json",
                "environment/container_identity.json",
                "raw/runner.json",
                "validation/point.json",
            }
            missing = sorted(
                relative
                for relative in required
                if not (self.stage / relative).is_file()
            )
            if missing:
                raise ArtifactStateError(
                    "Phase 6 finalization lacks required evidence payloads"
                )

        if isinstance(parsed, Phase8RunManifest):
            required = {
                "config/method.json",
                "environment/container_identity.json",
                "raw/runner.json",
                "validation/point.json",
            }
            missing = sorted(
                relative
                for relative in required
                if not (self.stage / relative).is_file()
            )
            if missing:
                raise ArtifactStateError(
                    "Phase 8 finalization lacks required evidence payloads"
                )

        if (
            isinstance(parsed, Phase11RunManifest)
            and parsed.status is RunStatus.COMPLETED
        ):
            required = {
                "accounting/contexts.json",
                "allocation/audit.json",
                "config/authority.json",
                "environment/container_identity.json",
                "execution-path/audit.json",
                "gqa/audit.json",
                "numerical/fixture-conformance.json",
                "validation/admission-candidate.json",
                "validation/bounded-grid.json",
                "validation/cuda-graph.json",
                "validation/sanitizer.json",
            }
            missing = sorted(
                relative
                for relative in required
                if not (self.stage / relative).is_file()
            )
            if missing:
                raise ArtifactStateError(
                    "Phase 11 finalization lacks required evidence payloads"
                )

        if (
            isinstance(parsed, Phase13BBatchAdmissionManifest)
            and parsed.status is RunStatus.COMPLETED
        ):
            required = {
                "admission/kivi-method-admission.json",
                "admission/kvquant-method-admission.json",
                "admission/turboquant-method-admission.json",
                "allocation/audit.json",
                "config/authority.json",
                "environment/container_identity.json",
                "gqa/audit.json",
                "numerical/batch-control.json",
                "validation/cuda-graph.json",
                "validation/historical-preservation.json",
                "validation/matrix.json",
                "validation/non-default-stream.json",
                "validation/sanitizer.json",
            }
            missing = sorted(
                relative
                for relative in required
                if not (self.stage / relative).is_file()
            )
            if missing:
                raise ArtifactStateError(
                    "Phase 13B finalization lacks required evidence payloads: "
                    f"{missing!r}"
                )

        if (
            isinstance(parsed, Phase9CalibrationManifest)
            and parsed.status is RunStatus.COMPLETED
        ):
            required = {
                "authority_manifest.json",
                "calibration_config.json",
                "dataset_manifest.json",
                "environment.json",
                "fisher/fisher.safetensors",
                "fisher_manifest.json",
                "inventory.json",
                "layer_stats.parquet",
                "model_manifest.json",
                "outlier_policy.json",
                "quantizers/kvq2.safetensors",
                "quantizers/kvq3.safetensors",
                "quantizers/kvq4.safetensors",
                "tokenizer_manifest.json",
                "tokens/input_ids.safetensors",
            }
            missing = sorted(
                relative
                for relative in required
                if not (self.stage / relative).is_file()
            )
            if missing:
                raise ArtifactStateError(
                    "Phase 9 finalization lacks required calibration payloads: "
                    f"{missing!r}"
                )

        self._write_lifecycle(3, "finalizing")
        self._state = "finalizing"
        self._write_lifecycle(4, status)
        _write_exclusive(self.stage, "manifest.json", _json_bytes(normalized))

        inventory_items = []
        inventory_exclusions = {
            "artifact_inventory.json",
            "checksums.sha256",
            "COMPLETE",
        }
        for path in _payload_files(self.stage, inventory_exclusions):
            relative = path.relative_to(self.stage).as_posix()
            inventory_items.append(
                {
                    "path": relative,
                    "role": _file_role(relative),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        inventory_payload = {
            "schema_version": "kvbench-artifact-inventory-1.0.0",
            "run_id": self.run_id,
            "files": inventory_items,
            "excluded_control_files": [
                "artifact_inventory.json",
                "checksums.sha256",
                "COMPLETE",
            ],
        }
        inventory = ArtifactInventory.from_dict(inventory_payload)
        _write_exclusive(
            self.stage,
            "artifact_inventory.json",
            _json_bytes(_as_payload(inventory)),
        )

        ledger_entries: list[tuple[str, str]] = []
        for path in _payload_files(self.stage, {"checksums.sha256", "COMPLETE"}):
            relative = path.relative_to(self.stage).as_posix()
            ledger_entries.append((sha256_file(path), relative))
        ledger_data = "".join(
            f"{digest}  {relative}\n" for digest, relative in ledger_entries
        ).encode("utf-8")
        _write_exclusive(self.stage, "checksums.sha256", ledger_data)
        ledger_sha256 = sha256_bytes(ledger_data)
        completion_payload = {
            "schema_version": "kvbench-completion-1.0.0",
            "run_id": self.run_id,
            "status": status,
            "manifest_sha256": sha256_file(self.stage / "manifest.json"),
            "artifact_inventory_sha256": sha256_file(
                self.stage / "artifact_inventory.json"
            ),
            "checksum_ledger_sha256": ledger_sha256,
            "written_last": True,
        }
        _write_exclusive(self.stage, "COMPLETE", _json_bytes(completion_payload))

        validation = validate_run_directory(self.stage, expect_final_name=False)
        if not validation.valid or not validation.complete:
            raise ChecksumMismatchError(
                "staged run failed final integrity validation",
            )

        for path in sorted(self.stage.rglob("*"), reverse=True):
            if path.is_file():
                path.chmod(0o444)
            elif path.is_dir():
                path.chmod(0o555)
        self.stage.chmod(0o555)
        _fsync_directory(self.stage)
        _rename_noreplace(self.stage, self.final)
        _fsync_directory(self.root)
        self._state = status
        self._finalized = True
        return self.final


def _parse_ledger(path: Path) -> tuple[dict[str, str], list[str]]:
    entries: dict[str, str] = {}
    errors: list[str] = []
    try:
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            return {}, ["checksum ledger is unsafe"]
        raw = path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            errors.append("checksum ledger lacks its canonical trailing newline")
        lines = raw.decode("utf-8").splitlines()
    except (OSError, UnicodeError):
        return {}, ["checksum ledger is unreadable"]
    for index, line in enumerate(lines, start=1):
        parts = line.split("  ", 1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            errors.append(f"checksum ledger line {index} is malformed")
            continue
        digest, relative = parts
        try:
            safe = _safe_relative_path(relative).as_posix()
        except ArtifactSafetyError:
            errors.append(f"checksum ledger line {index} has an unsafe path")
            continue
        if safe in entries:
            errors.append(f"checksum ledger path is duplicated: {safe}")
            continue
        entries[safe] = digest
    if list(entries) != sorted(entries):
        errors.append("checksum ledger paths are not in canonical lexical order")
    return entries, errors


def _load_json_object(path: Path, label: str, errors: list[str]) -> dict[str, Any] | None:
    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(_: str) -> None:
        raise ValueError("non-finite JSON number")

    try:
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            errors.append(f"{label} is missing or unsafe")
            return None
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, ValueError):
        errors.append(f"{label} is missing or invalid JSON")
        return None
    if not isinstance(payload, dict):
        errors.append(f"{label} must be a JSON object")
        return None
    return payload


def _lifecycle_status(
    directory: Path,
    *,
    run_id: str | None,
    terminal_status: str | None,
    complete: bool,
    errors: list[str],
) -> str | None:
    lifecycle_root = directory / "lifecycle"
    try:
        metadata = lifecycle_root.lstat()
    except OSError:
        errors.append("lifecycle directory is missing")
        return None
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        errors.append("lifecycle directory is unsafe")
        return None
    records: list[LifecycleRecord] = []
    try:
        paths = sorted(lifecycle_root.iterdir())
    except OSError:
        errors.append("lifecycle directory is unreadable")
        return None
    for path in paths:
        payload = _load_json_object(path, "lifecycle record", errors)
        if payload is None:
            continue
        try:
            record = LifecycleRecord.from_dict(payload)
        except SchemaValidationError:
            errors.append("lifecycle record failed strict schema validation")
            continue
        expected_name = f"{record.sequence:04d}-{record.state}.json"
        if path.name != expected_name:
            errors.append("lifecycle record filename does not match its content")
        if run_id is not None and record.run_id != run_id:
            errors.append("lifecycle record run_id mismatch")
        records.append(record)
    sequences = [record.sequence for record in records]
    if sequences != list(range(1, len(records) + 1)):
        errors.append("lifecycle records are not a contiguous ordered prefix")
    if complete:
        if len(records) != 4:
            errors.append("completed run must contain exactly four lifecycle records")
        elif records[-1].state != terminal_status:
            errors.append("terminal lifecycle state does not match manifest status")
    return records[-1].state if records else None


def _validate_run_directory_inner(
    run_dir: str | Path,
    *,
    expect_final_name: bool = True,
) -> RunValidationResult:
    lexical = Path(run_dir).absolute()
    errors: list[str] = []
    try:
        lexical_metadata = lexical.lstat()
    except OSError:
        return RunValidationResult(
            lexical,
            False,
            False,
            None,
            ("run directory is missing or unsafe",),
            0,
        )
    if stat.S_ISLNK(lexical_metadata.st_mode) or not stat.S_ISDIR(
        lexical_metadata.st_mode
    ):
        return RunValidationResult(
            lexical,
            False,
            False,
            None,
            ("run directory is missing or unsafe",),
            0,
        )
    directory = lexical.resolve(strict=True)

    initial_payload = _load_json_object(
        directory / "manifest.initial.json", "initial manifest", errors
    )
    initial_manifest: RunManifest | None = None
    initial_run_id: str | None = None
    if initial_payload is not None:
        try:
            initial_manifest = _validate_manifest(initial_payload)
            initial_run_id = initial_manifest.run_id
            if initial_manifest.status is not RunStatus.CREATED:
                errors.append("initial manifest status must be created")
        except SchemaValidationError:
            errors.append("initial manifest failed independent schema validation")

    complete_path = directory / "COMPLETE"
    if not complete_path.is_file():
        status = _lifecycle_status(
            directory,
            run_id=initial_run_id,
            terminal_status=None,
            complete=False,
            errors=errors,
        )
        return RunValidationResult(
            directory,
            False,
            False,
            status,
            tuple(errors or ["run is incomplete: COMPLETE is absent"]),
            len(_payload_files(directory, set())),
        )

    completion = _load_json_object(complete_path, "completion marker", errors)
    manifest = _load_json_object(directory / "manifest.json", "manifest", errors)
    inventory = _load_json_object(
        directory / "artifact_inventory.json", "artifact inventory", errors
    )
    ledger_path = directory / "checksums.sha256"
    ledger, ledger_errors = _parse_ledger(ledger_path)
    errors.extend(ledger_errors)

    actual_ledger_files = {
        path.relative_to(directory).as_posix()
        for path in _payload_files(directory, {"checksums.sha256", "COMPLETE"})
    }
    if set(ledger) != actual_ledger_files:
        errors.append("checksum ledger does not exactly cover run payload files")
    for relative, expected in ledger.items():
        target = directory / relative
        if not target.is_file() or target.is_symlink():
            errors.append(f"checksummed artifact is missing or unsafe: {relative}")
        elif sha256_file(target) != expected:
            errors.append(f"checksum mismatch: {relative}")

    status: str | None = None
    run_id: str | None = None
    parsed_manifest: RunManifest | None = None
    if manifest is not None:
        try:
            parsed_manifest = _validate_manifest(manifest)
            normalized = _as_payload(parsed_manifest)
            status = _status_text(normalized)
            run_id_value = normalized.get("run_id")
            run_id = run_id_value if isinstance(run_id_value, str) else None
            if status not in _TERMINAL_STATUSES:
                errors.append("final manifest status is not terminal")
        except (SchemaValidationError, TypeError):
            errors.append("manifest failed independent schema validation")

    if initial_manifest is not None and parsed_manifest is not None:
        if _manifest_intent(_as_payload(initial_manifest)) != _manifest_intent(
            _as_payload(parsed_manifest)
        ):
            errors.append("final manifest changes immutable initial provenance")
        if initial_manifest.run_id != parsed_manifest.run_id:
            errors.append("initial and final manifest run IDs differ")

    _lifecycle_status(
        directory,
        run_id=run_id or initial_run_id,
        terminal_status=status,
        complete=True,
        errors=errors,
    )

    if expect_final_name and run_id is not None and directory.name != run_id:
        errors.append("final directory name does not match manifest run_id")
    if expect_final_name:
        write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        for path in (directory, *sorted(directory.rglob("*"))):
            metadata = path.lstat()
            if metadata.st_mode & write_bits:
                errors.append("finalized run contains writable content")
                break

    inventory_count = 0
    if inventory is not None:
        try:
            normalized_inventory = _as_payload(ArtifactInventory.from_dict(inventory))
            files = normalized_inventory.get("files")
            inventory_count = len(files) if isinstance(files, list) else 0
            declared = {
                item["path"]
                for item in files or []
                if isinstance(item, Mapping) and isinstance(item.get("path"), str)
            }
            actual_inventory_files = {
                path.relative_to(directory).as_posix()
                for path in _payload_files(
                    directory,
                    {"artifact_inventory.json", "checksums.sha256", "COMPLETE"},
                )
            }
            if declared != actual_inventory_files:
                errors.append("artifact inventory does not exactly cover payload files")
            for item in files or []:
                if not isinstance(item, Mapping):
                    continue
                relative = item.get("path")
                if not isinstance(relative, str):
                    continue
                target = directory / relative
                if target.is_file():
                    if item.get("sha256") != sha256_file(target):
                        errors.append(f"artifact inventory hash mismatch: {relative}")
                    if item.get("size_bytes") != target.stat().st_size:
                        errors.append(f"artifact inventory size mismatch: {relative}")
        except (SchemaValidationError, TypeError, KeyError):
            errors.append("artifact inventory failed independent schema validation")

    completion_marker: CompletionMarker | None = None
    if completion is not None:
        try:
            completion_marker = CompletionMarker.from_dict(completion)
        except SchemaValidationError:
            errors.append("completion marker failed independent schema validation")
    if completion_marker is not None:
        if run_id is not None and completion_marker.run_id != run_id:
            errors.append("completion marker run_id mismatch")
        if status is not None and completion_marker.status.value != status:
            errors.append("completion marker status mismatch")
        if (directory / "manifest.json").is_file() and (
            completion_marker.manifest_sha256
            != sha256_file(directory / "manifest.json")
        ):
            errors.append("completion marker manifest hash mismatch")
        if (directory / "artifact_inventory.json").is_file() and (
            completion_marker.artifact_inventory_sha256
            != sha256_file(directory / "artifact_inventory.json")
        ):
            errors.append("completion marker inventory hash mismatch")
        if ledger_path.is_file() and (
            completion_marker.checksum_ledger_sha256 != sha256_file(ledger_path)
        ):
            errors.append("completion marker ledger hash mismatch")

    return RunValidationResult(
        directory,
        not errors,
        True,
        status,
        tuple(errors),
        inventory_count,
    )


def validate_run_directory(
    run_dir: str | Path,
    *,
    expect_final_name: bool = True,
) -> RunValidationResult:
    """Validate a run without letting unsafe filesystem content escape."""

    try:
        return _validate_run_directory_inner(
            run_dir,
            expect_final_name=expect_final_name,
        )
    except (
        OSError,
        TypeError,
        ValueError,
        ArtifactSafetyError,
        SchemaValidationError,
    ) as error:
        return RunValidationResult(
            Path(run_dir).absolute(),
            False,
            False,
            None,
            (f"run validation encountered unsafe or inconsistent content: {type(error).__name__}",),
            0,
        )


def phase3_artifact_store(
    repository_root: str | Path | None = None,
) -> AppendOnlyArtifactStore:
    """Return the exact local Phase 3 engineering-evidence store."""

    root = (
        Path(repository_root).resolve(strict=True)
        if repository_root is not None
        else Path(__file__).resolve().parents[3]
    )
    return AppendOnlyArtifactStore(
        root / "artifacts" / "phase3",
        formal_evidence_roots=(
            root / "docs" / "evidence",
            root / "artifacts" / "quality",
            root / "artifacts" / "profiler",
            root / "paper-results",
            root / "paper_results",
            root / "results",
        ),
    )


def phase6_artifact_store(
    repository_root: str | Path | None = None,
) -> AppendOnlyArtifactStore:
    """Return the exact local Phase 6 container-admission store."""

    root = (
        Path(repository_root).resolve(strict=True)
        if repository_root is not None
        else Path(__file__).resolve().parents[3]
    )
    return AppendOnlyArtifactStore(
        root / "artifacts" / "phase6",
        formal_evidence_roots=(
            root / "docs" / "evidence",
            root / "artifacts" / "quality",
            root / "artifacts" / "profiler",
            root / "paper-results",
            root / "paper_results",
            root / "results",
        ),
    )


def phase8_artifact_store(
    repository_root: str | Path | None = None,
) -> AppendOnlyArtifactStore:
    """Return the exact local Phase 8 container-admission store."""

    root = (
        Path(repository_root).resolve(strict=True)
        if repository_root is not None
        else Path(__file__).resolve().parents[3]
    )
    return AppendOnlyArtifactStore(
        root / "artifacts" / "phase8",
        formal_evidence_roots=(
            root / "docs" / "evidence",
            root / "artifacts" / "quality",
            root / "artifacts" / "profiler",
            root / "paper-results",
            root / "paper_results",
            root / "results",
        ),
    )


def phase9_calibration_artifact_store(
    repository_root: str | Path | None = None,
) -> AppendOnlyArtifactStore:
    """Return the exact local Phase 9 KVQuant calibration store."""

    root = (
        Path(repository_root).resolve(strict=True)
        if repository_root is not None
        else Path(__file__).resolve().parents[3]
    )
    return AppendOnlyArtifactStore(
        root / "calibration" / "kvquant",
        formal_evidence_roots=(
            root / "docs" / "evidence",
            root / "artifacts" / "quality",
            root / "artifacts" / "profiler",
            root / "paper-results",
            root / "paper_results",
            root / "results",
        ),
    )


def summarize_run_directory(run_dir: str | Path) -> dict[str, Any]:
    """Return structural facts only; this function makes no scientific claims."""

    directory = Path(run_dir).resolve(strict=False)
    validation = validate_run_directory(directory)
    manifest_path = (
        directory / "manifest.json"
        if (directory / "manifest.json").is_file()
        else directory / "manifest.initial.json"
    )
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                manifest = loaded
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
    return {
        "schema_version": "kvbench-summary-output-1.0.0",
        "run_id": manifest.get("run_id", directory.name),
        "status": validation.status,
        "complete": validation.complete,
        "valid": validation.valid,
        "run_kind": manifest.get("run_kind"),
        "claim_class": manifest.get("claim_class"),
        "performance_claim_eligible": manifest.get(
            "performance_claim_eligible"
        ),
        "measurement_scope": manifest.get("measurement_scope"),
        "quality_status": (
            manifest.get("quality", {}).get("quality_status")
            if isinstance(manifest.get("quality"), dict)
            else None
        ),
        "claim_eligibility": (
            manifest.get("quality", {}).get("claim_eligibility")
            if isinstance(manifest.get("quality"), dict)
            else None
        ),
        "artifact_count": validation.artifact_count,
        "validation_errors": list(validation.errors),
        "scientific_conclusions_generated": False,
    }
