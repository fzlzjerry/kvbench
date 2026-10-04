"""Fail-closed coordinator for the two frozen Phase 3 BF16 plans."""

from __future__ import annotations

from collections import Counter
import dataclasses
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import select
import signal
import stat
import subprocess
import tempfile
import time
from typing import Any, Mapping
from kvbench.adapters import (
    MethodRuntimeContext,
    build_method_adapter,
)


from kvbench.config import REPOSITORY_ROOT, ExperimentBundle, load_phase3_admission_bundle
from kvbench.errors import KVBenchError, SchemaValidationError
from kvbench.runtime.artifacts import (
    phase3_artifact_store,
    sha256_file,
    validate_run_directory,
)
from kvbench.runtime.gqa_device_dispatch import (
    REQUIRED_SUT_SOURCES,
    phase3_source_identity_sha256,
)
from kvbench.runtime.phase3_audit_operation import Phase3AuditOperationKey
from kvbench.runtime.phase3_campaign import Phase3CampaignRecorder
from kvbench.runtime.phase3_raw_audit_evidence import (
    RAW_AUDIT_STATUS_COMPLETED,
    Phase3RawAuditEvidenceError,
    Phase3RawAuditRunIndex,
    ingest_phase3_raw_audit_evidence_fd,
    parse_phase3_raw_audit_run_index_bytes,
)
from kvbench.runtime.phase3_worker_channels import (
    PHASE3_RAW_AUDIT_OPERATION_PLAN_ENV,
    build_phase3_raw_audit_operation_plan_bytes,
    build_phase3_worker_channel_commitment,
    phase3_worker_channel_commitment_sha256,
)
from kvbench.runtime.process_supervision import (
    DeviceProcessObservation,
    HandshakeStage,
    OwnershipDisposition,
    ProcessIdentity,
    ProcessIdentityUnavailable,
    ProcessSupervisionError,
    RunOwnedProcessRegistry,
    SnapshotDisposition,
    command_fingerprint,
    publish_bytes_no_replace,
    read_published_bytes,
    read_process_identity,
    write_handshake_event,
)
from kvbench.schema import (
    BF16BackendIdentity,
    BF16CacheIdentity,
    ClaimClass,
    ConfigSourceKind,
    GateDisposition,
    GQAVerdict,
    MeasurementScope,
    MethodConfigFingerprint,
    Phase3CommandSpec,
    Phase3RunManifest,
    Phase3WorkerResult,
    RunKind,
    RunStatus,
    canonical_json_bytes,
    derive_cache_layout_fingerprint,
    derive_phase3_point_fingerprint,
    expand_phase3_process_points,
    sha256_hex,
)
from kvbench.schema.phase3 import (
    PHASE3_CONTRACT_FINGERPRINT,
    PHASE3_DRIVER_VERSION,
    PHASE3_E00_MANIFEST_SHA256,
    PHASE3_E00_RUN_ID,
    PHASE3_FIXED_PLAN_PATH,
    PHASE3_GPU_FULL_NAME,
    PHASE3_GPU_UUID,
    PHASE3_GROWING_PLAN_PATH,
    PHASE3_HARDWARE_FINGERPRINT,
    PHASE3_HARDWARE_ID,
    PHASE3_MEASUREMENT_PROTOCOL_FINGERPRINT,
    PHASE3_MODEL_FINGERPRINT,
    PHASE3_PCI_BUS_ID,
    PHASE3_PCI_DEVICE_ID,
    PHASE3_PLAN_FINGERPRINTS,
    PHASE3_PYTHON_EXECUTABLE,
    PHASE3_REPOSITORY_ROOT,
    PHASE3_SOFTWARE_ENVIRONMENT_ID,
    PHASE3_SOFTWARE_FINGERPRINT,
)


PYTHON_EXECUTABLE = Path(PHASE3_PYTHON_EXECUTABLE)
PROCESS_QUERY = REPOSITORY_ROOT / "preflight" / "process_query.py"
STATIC_CACHE_SOURCE = REPOSITORY_ROOT / "src/kvbench/runtime/static_cache.py"
TORCH_PACKAGE_ROOT = (
    REPOSITORY_ROOT / ".venv/lib/python3.12/site-packages/torch"
)
E00_MANIFEST = (
    REPOSITORY_ROOT
    / "docs/evidence/e00"
    / PHASE3_E00_RUN_ID
    / "manifest.json"
)
FAILED_E00_MANIFEST = (
    REPOSITORY_ROOT
    / "docs/evidence/e00/e00-20260722T041628.190813Z-980eff7b6f59-0dd71f2d/manifest.json"
)
FAILED_E00_LEDGER = FAILED_E00_MANIFEST.with_name("checksums.sha256")
SUCCESSFUL_E00_LEDGER = E00_MANIFEST.with_name("checksums.sha256")
PHASE3_DEPENDENCY_LOCK = REPOSITORY_ROOT / "preflight/requirements-phase3.txt"
PHASE2_FINAL_SHA = "7c36d130565acef0883acb638c6b6c731b3f32ad"
PERFORMANCE_FREEZE_MARKER = REPOSITORY_ROOT / "PERFORMANCE_DATA_FROZEN"
MAX_STDOUT_BYTES = 1024 * 1024
MAX_IPC_BYTES = 64 * 1024 * 1024
WORKER_TIMEOUT_SECONDS = 3600
READY_TIMEOUT_SECONDS = 120
SENSITIVE_ENV_FRAGMENTS = (
    "token",
    "secret",
    "password",
    "credential",
    "api_key",
    "proxy",
)
SENSITIVE_ENV_KEY_EXEMPTIONS = frozenset({"TOKENIZERS_PARALLELISM"})
HANDSHAKE_DIRECTORY_ENV = "KVBENCH_PHASE3_HANDSHAKE_DIR"
COMMAND_FINGERPRINT_ENV = "KVBENCH_PHASE3_COMMAND_FINGERPRINT"
RAW_AUDIT_ROOT_ENV = "KVBENCH_PHASE3_RAW_AUDIT_ROOT"
RAW_AUDIT_INDEX_IPC_ENV = "KVBENCH_PHASE3_RAW_AUDIT_INDEX_IPC_PATH"
WORKER_EVIDENCE_V1 = "kvbench-phase3-worker-evidence-1.0.0"
WORKER_EVIDENCE_V2 = "kvbench-phase3-worker-evidence-2.0.0"
RAW_AUDIT_INGESTION_OUTCOME_SCHEMA_VERSION = (
    "kvbench-phase3-raw-audit-ingestion-outcome-2.0.0"
)
PHASE3_WORKER_HANDSHAKE_EVIDENCE_SCHEMA_VERSION = (
    "kvbench-phase3-worker-handshake-3.0.0"
)
PHASE3_PROCESS_AUDIT_SCHEMA_VERSION = "kvbench-phase3-process-audit-3.0.0"
PHASE3_EXECUTION_SOURCE_PIN_SCHEMA_VERSION = (
    "kvbench-phase3-execution-source-pin-2.0.0"
)
PHASE3_EXECUTION_SOURCE_PATHS = (
    "src/kvbench/__init__.py",
    "src/kvbench/__main__.py",
    "src/kvbench/cli.py",
    "src/kvbench/config.py",
    "src/kvbench/errors.py",
    "src/kvbench/runtime/__init__.py",
    "src/kvbench/runtime/allocation.py",
    "src/kvbench/runtime/allocation_attribution.py",
    "src/kvbench/runtime/artifacts.py",
    "src/kvbench/runtime/backend.py",
    "src/kvbench/runtime/bf16_endpoint.py",
    "src/kvbench/runtime/command.py",
    "src/kvbench/runtime/cuda_graph.py",
    "src/kvbench/runtime/fixed_l_runner.py",
    "src/kvbench/runtime/gqa_audit.py",
    "src/kvbench/runtime/gqa_device_dispatch.py",
    "src/kvbench/runtime/gqa_taxonomy.py",
    "src/kvbench/runtime/growing_context_runner.py",
    "src/kvbench/runtime/model_loader.py",
    "src/kvbench/runtime/numerical.py",
    "src/kvbench/runtime/phase3_allocator_controls.py",
    "src/kvbench/runtime/phase3_audit_operation.py",
    "src/kvbench/runtime/phase3_campaign.py",
    "src/kvbench/runtime/phase3_coordinator.py",
    "src/kvbench/runtime/phase3_endpoint_audit.py",
    "src/kvbench/runtime/phase3_raw_audit_evidence.py",
    "src/kvbench/runtime/phase3_report.py",
    "src/kvbench/runtime/phase3_report_publication.py",
    "src/kvbench/runtime/phase3_worker.py",
    "src/kvbench/runtime/phase3_worker_channels.py",
    "src/kvbench/runtime/process_supervision.py",
    "src/kvbench/runtime/static_cache.py",
    "src/kvbench/runtime/telemetry.py",
    "src/kvbench/runtime/timing.py",
    "src/kvbench/schema/__init__.py",
    "src/kvbench/schema/base.py",
    "src/kvbench/schema/config.py",
    "src/kvbench/schema/phase3.py",
    "src/kvbench/schema/result.py",
    "src/kvbench/validation.py",
)
PHASE3_WORKER_TERMINATION_SCHEMA_VERSION = (
    "kvbench-phase3-worker-termination-1.0.0"
)
TRANSPORT_PRIMARY_CHANNEL_ARTIFACT = (
    "raw/transport/primary_worker_evidence.v1.jsonl"
)
TRANSPORT_SIDECAR_CHANNEL_ARTIFACT = (
    "raw/transport/raw_audit_index_sidecar.v2.jsonl"
)
TRANSPORT_COMMITMENT_ARTIFACT = "raw/transport/channel_commitment.json"
TRANSPORT_COMMITMENT_DIGEST_ARTIFACT = (
    "raw/transport/channel_commitment.sha256"
)
READY_NOT_OBSERVED_V2 = {
    "schema_version": "kvbench-phase3-worker-ready-2.0.0",
    "readiness_observed": False,
    "pid": None,
    "process_start_time_ticks": None,
    "cuda_imported": None,
}


class Phase3CoordinatorError(RuntimeError):
    """Campaign coordination failed before a trustworthy worker result."""


class Phase3WorkerTerminationUnresolved(Phase3CoordinatorError):
    """A failed run was preserved, but its spawned worker was not resolved."""


def _phase3_execution_source_identity_sha256(
    source_sha256_by_path: Mapping[str, str],
) -> str:
    """Hash the exact ordered package source manifest used by Phase 3."""

    if tuple(source_sha256_by_path) != PHASE3_EXECUTION_SOURCE_PATHS:
        raise Phase3CoordinatorError(
            "execution source identity requires the exact package source set"
        )
    sources: list[dict[str, str]] = []
    for relative in PHASE3_EXECUTION_SOURCE_PATHS:
        digest = source_sha256_by_path[relative]
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise Phase3CoordinatorError(
                "execution source identity contains an invalid digest"
            )
        sources.append({"relative_path": relative, "sha256": digest})
    return sha256_hex(
        canonical_json_bytes(
            {
                "schema_version": "kvbench-phase3-execution-source-identity-1.0.0",
                "sources": sources,
            }
        )
    )


@dataclasses.dataclass(frozen=True)
class Phase3ExecutionSourcePin:
    """Exact execution-commit bytes retained before the worker is spawned."""

    execution_git_sha: str
    source_bytes_by_path: tuple[tuple[str, bytes], ...]
    source_identity_sha256: str
    execution_source_identity_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.execution_git_sha) is not str
            or len(self.execution_git_sha) != 40
            or any(
                character not in "0123456789abcdef"
                for character in self.execution_git_sha
            )
        ):
            raise Phase3CoordinatorError("execution source pin Git SHA is invalid")
        if (
            tuple(
                relative for relative, _ in self.source_bytes_by_path
            )
            != PHASE3_EXECUTION_SOURCE_PATHS
            or any(
                type(payload) is not bytes or not payload
                for _, payload in self.source_bytes_by_path
            )
        ):
            raise Phase3CoordinatorError(
                "execution source pin does not contain the exact package source set"
            )
        if self.source_identity_sha256 != phase3_source_identity_sha256(
            self.sut_source_sha256_by_path
        ):
            raise Phase3CoordinatorError("execution SUT source identity is invalid")
        if self.execution_source_identity_sha256 != (
            _phase3_execution_source_identity_sha256(self.source_sha256_by_path)
        ):
            raise Phase3CoordinatorError(
                "execution package source identity is invalid"
            )

    @property
    def source_sha256_by_path(self) -> dict[str, str]:
        return {
            relative: sha256_hex(payload)
            for relative, payload in self.source_bytes_by_path
        }

    @property
    def sut_source_sha256_by_path(self) -> dict[str, str]:
        source_digests = self.source_sha256_by_path
        return {
            relative: source_digests[relative]
            for relative in REQUIRED_SUT_SOURCES
        }

    def to_dict(self, *, verification_stage: str) -> dict[str, Any]:
        if verification_stage not in {"before_spawn", "after_worker_exit"}:
            raise ValueError("execution source verification stage is invalid")
        return {
            "schema_version": PHASE3_EXECUTION_SOURCE_PIN_SCHEMA_VERSION,
            "execution_git_sha": self.execution_git_sha,
            "verification_stage": verification_stage,
            "source_identity_sha256": self.source_identity_sha256,
            "execution_source_identity_sha256": (
                self.execution_source_identity_sha256
            ),
            "sources": [
                {
                    "relative_path": relative,
                    "sha256": sha256_hex(payload),
                    "size_bytes": len(payload),
                }
                for relative, payload in self.source_bytes_by_path
            ],
            "live_bytes_equal_declared_commit": True,
        }


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _run_checked(argv: tuple[str, ...]) -> str:
    result = subprocess.run(
        argv,
        cwd=REPOSITORY_ROOT,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        env={
            "PATH": "/usr/bin:/bin",
            "LC_ALL": "C",
            "LANG": "C",
            "TZ": "UTC",
        },
    )
    if result.returncode != 0:
        raise Phase3CoordinatorError(f"command failed: {argv[0]}")
    return result.stdout.strip()


def _git_identity() -> tuple[str, bool]:
    sha = _run_checked(("/usr/bin/git", "rev-parse", "HEAD"))
    dirty = bool(_run_checked(("/usr/bin/git", "status", "--porcelain=v1")))
    if dirty:
        raise Phase3CoordinatorError("Phase 3 execution requires a clean Git tree")
    ancestor = subprocess.run(
        ("/usr/bin/git", "merge-base", "--is-ancestor", PHASE2_FINAL_SHA, sha),
        cwd=REPOSITORY_ROOT,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
    )
    if ancestor.returncode != 0:
        raise Phase3CoordinatorError("accepted Phase 2 final commit is not an ancestor")
    return sha, dirty


def _validate_entry_evidence() -> None:
    expected_files = {
        FAILED_E00_MANIFEST: "0720734d29c90f609e51cf4c5e4f0b1fadce220e23e146e566f860bb962c0035",
        FAILED_E00_LEDGER: "8716fc317747e7e9b5c06017cb8e5339df610c5a89d0d7fbee82ad07fbc68b52",
        E00_MANIFEST: PHASE3_E00_MANIFEST_SHA256,
        SUCCESSFUL_E00_LEDGER: "5a610162163979aca97beb2b7b0b480befb85d0b4e63b77c26ec46c36864eca8",
        PHASE3_DEPENDENCY_LOCK: "cebe254a3e03a48e3e67100ce11d5623fc0dc722dc43e2f482152beb644a08e9",
    }
    if any(
        not path.is_file() or sha256_file(path) != expected
        for path, expected in expected_files.items()
    ):
        raise Phase3CoordinatorError("certified native-host E00 manifest changed")
    if PERFORMANCE_FREEZE_MARKER.exists() or PERFORMANCE_FREEZE_MARKER.is_symlink():
        raise Phase3CoordinatorError("quality freeze marker must remain absent")
    for forbidden in (
        REPOSITORY_ROOT / "paper-results",
        REPOSITORY_ROOT / "paper_results",
        REPOSITORY_ROOT / "artifacts/quality",
        REPOSITORY_ROOT / "artifacts/profiler",
    ):
        if forbidden.exists() or forbidden.is_symlink():
            raise Phase3CoordinatorError("forbidden result root exists")


def _live_hardware() -> dict[str, Any]:
    argv = (
        "/usr/bin/nvidia-smi",
        "--query-gpu=name,uuid,pci.bus_id,pci.device_id,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    )
    rows = [line.strip() for line in _run_checked(argv).splitlines() if line.strip()]
    if len(rows) != 1:
        raise Phase3CoordinatorError("Phase 3 requires exactly one visible GPU")
    values = [item.strip() for item in rows[0].split(",")]
    if len(values) != 6:
        raise Phase3CoordinatorError("live GPU identity field count differs")
    observed = tuple(values)
    expected = (
        PHASE3_GPU_FULL_NAME,
        PHASE3_GPU_UUID,
        PHASE3_PCI_BUS_ID,
        PHASE3_PCI_DEVICE_ID,
        PHASE3_DRIVER_VERSION,
        "12.0",
    )
    if observed != expected:
        raise Phase3CoordinatorError("live GPU differs from certified native-host G0")
    return {
        "schema_version": "kvbench-phase3-live-hardware-1.0.0",
        "gpu_name": values[0],
        "gpu_uuid": values[1],
        "pci_bus_id": values[2],
        "pci_device_id": values[3],
        "driver_version": values[4],
        "compute_capability": values[5],
        "native_g0_status": "PASS",
        "container_parity_status": "not_evaluated",
        "blocker_b010": "OPEN",
    }


def _worker_environment(
    temp_root: Path,
    *,
    raw_audit_operations: tuple[Phase3AuditOperationKey, ...] | None = None,
) -> dict[str, str]:
    environment = {
        "PATH": "/usr/local/cuda/bin:/usr/bin:/bin",
        "PYTHONPATH": (
            f"{REPOSITORY_ROOT / '.phase3/site-packages'}:"
            f"{REPOSITORY_ROOT / 'src'}"
        ),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUTF8": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "0",
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
        "TZ": "UTC",
        "KVBENCH_PHASE3_IPC_PATH": str(temp_root / "worker-evidence.json"),
        "KVBENCH_PHASE3_AUDIT_READY": str(temp_root / "worker-ready.json"),
        "KVBENCH_PHASE3_AUDIT_RELEASE": str(temp_root / "worker-release"),
        HANDSHAKE_DIRECTORY_ENV: str(temp_root / "worker-handshake"),
        RAW_AUDIT_ROOT_ENV: str(temp_root / "raw-audits"),
        RAW_AUDIT_INDEX_IPC_ENV: str(temp_root / "raw-audit-index.json"),
        "KVBENCH_PHASE3_HANDSHAKE_TIMEOUT_SECONDS": str(READY_TIMEOUT_SECONDS),
    }
    if raw_audit_operations is not None:
        environment[PHASE3_RAW_AUDIT_OPERATION_PLAN_ENV] = (
            build_phase3_raw_audit_operation_plan_bytes(
                raw_audit_operations
            ).decode("utf-8")
        )
    if any(
        fragment in key.lower()
        for key in environment
        if key not in SENSITIVE_ENV_KEY_EXEMPTIONS
        for fragment in SENSITIVE_ENV_FRAGMENTS
    ):
        raise Phase3CoordinatorError("worker environment contains a forbidden key")
    return environment


def _parse_canonical_json(raw: bytes, *, maximum_bytes: int, label: str) -> dict[str, Any]:
    if not raw or len(raw) > maximum_bytes or not raw.endswith(b"\n"):
        raise Phase3CoordinatorError(f"{label} is absent, oversized, or unterminated")
    body = raw[:-1]
    if b"\n" in body or b"\r" in body:
        raise Phase3CoordinatorError(f"{label} must be one canonical JSON line")

    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    def reject_constant(_: str) -> None:
        raise ValueError("non-finite JSON")

    try:
        value = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=reject_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise Phase3CoordinatorError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != body:
        raise Phase3CoordinatorError(f"{label} is not a canonical JSON object")
    return value


def _open_private_raw_audit_root(path: Path) -> tuple[int, int]:
    """Pin the exact empty coordinator-created raw root before worker spawn."""

    required_flags = ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required_flags):
        raise Phase3CoordinatorError(
            "platform lacks required raw-audit descriptor flags"
        )
    try:
        before = path.lstat()
    except OSError as error:
        raise Phase3CoordinatorError("raw-audit root is absent") from error
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISDIR(before.st_mode)
        or before.st_uid != os.geteuid()
        or stat.S_IMODE(before.st_mode) != 0o700
    ):
        raise Phase3CoordinatorError(
            "raw-audit root must be private and coordinator-owned"
        )
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise Phase3CoordinatorError(
            "raw-audit root cannot be pinned safely"
        ) from error
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or opened.st_uid != before.st_uid
            or stat.S_IMODE(opened.st_mode) != 0o700
        ):
            raise Phase3CoordinatorError(
                "raw-audit root changed while it was pinned"
            )
        with os.scandir(descriptor) as entries:
            if next(entries, None) is not None:
                raise Phase3CoordinatorError(
                    "raw-audit root was not empty before worker spawn"
                )
        return descriptor, opened.st_uid
    except BaseException:
        os.close(descriptor)
        raise


def _ipc_metadata_identity(metadata: os.stat_result) -> tuple[int, ...]:
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


def _open_private_ipc_parent(path: Path) -> tuple[int, int]:
    """Pin the coordinator-created private IPC parent before worker spawn."""

    required_flags = ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required_flags):
        raise Phase3CoordinatorError("platform lacks required IPC descriptor flags")
    try:
        before = path.lstat()
    except OSError as error:
        raise Phase3CoordinatorError("worker IPC parent is absent") from error
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISDIR(before.st_mode)
        or before.st_uid != os.geteuid()
        or stat.S_IMODE(before.st_mode) != 0o700
    ):
        raise Phase3CoordinatorError(
            "worker IPC parent must be private and coordinator-owned"
        )
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
    except OSError as error:
        raise Phase3CoordinatorError("worker IPC parent cannot be pinned") from error
    opened = os.fstat(descriptor)
    if (
        (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        or opened.st_uid != before.st_uid
        or stat.S_IMODE(opened.st_mode) != 0o700
    ):
        os.close(descriptor)
        raise Phase3CoordinatorError("worker IPC parent changed while opening")
    return descriptor, opened.st_uid


def _read_pinned_ipc_file(
    parent_fd: int,
    name: str,
    *,
    expected_owner_uid: int,
    maximum_bytes: int,
    label: str,
) -> bytes:
    """Read one worker IPC file with no path reopening or symlink following."""

    if (
        type(parent_fd) is not int
        or parent_fd < 0
        or type(name) is not str
        or not name
        or "/" in name
        or "\\" in name
    ):
        raise Phase3CoordinatorError(f"{label} request is unsafe")
    if type(expected_owner_uid) is not int or expected_owner_uid < 0:
        raise Phase3CoordinatorError(f"{label} owner is invalid")
    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise Phase3CoordinatorError(f"{label} size limit is invalid")
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as error:
        raise Phase3CoordinatorError(f"{label} is absent") from error
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != expected_owner_uid
        or stat.S_IMODE(before.st_mode) != 0o600
        or before.st_size <= 0
        or before.st_size > maximum_bytes
    ):
        raise Phase3CoordinatorError(f"{label} is unsafe or oversized")
    required_flags = ("O_CLOEXEC", "O_NOFOLLOW")
    if any(not hasattr(os, item) for item in required_flags):
        raise Phase3CoordinatorError("platform lacks required IPC file flags")
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
    except OSError as error:
        raise Phase3CoordinatorError(f"{label} cannot be opened safely") from error
    try:
        opened = os.fstat(descriptor)
        if _ipc_metadata_identity(opened) != _ipc_metadata_identity(before):
            raise Phase3CoordinatorError(f"{label} changed while opening")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise Phase3CoordinatorError(f"{label} was truncated while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise Phase3CoordinatorError(f"{label} grew while reading")
        after = os.fstat(descriptor)
        if _ipc_metadata_identity(after) != _ipc_metadata_identity(opened):
            raise Phase3CoordinatorError(f"{label} changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_live_sut_source_bytes(relative_path: str) -> bytes:
    """Read one pinned package source without following links or races."""

    if relative_path not in PHASE3_EXECUTION_SOURCE_PATHS:
        raise Phase3CoordinatorError(
            "requested source is outside the exact execution source set"
        )
    path = REPOSITORY_ROOT / relative_path
    try:
        before = path.lstat()
    except OSError as error:
        raise Phase3CoordinatorError("required live SUT source is absent") from error
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size <= 0
        or before.st_size > 16 * 1024 * 1024
    ):
        raise Phase3CoordinatorError("required live SUT source is unsafe")
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_CLOEXEC"):
        raise Phase3CoordinatorError("platform lacks required source file flags")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise Phase3CoordinatorError("required live SUT source cannot be opened") from error
    try:
        opened = os.fstat(descriptor)
        if _ipc_metadata_identity(opened) != _ipc_metadata_identity(before):
            raise Phase3CoordinatorError("required live SUT source changed while opening")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise Phase3CoordinatorError("required live SUT source was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise Phase3CoordinatorError("required live SUT source grew while reading")
        after = os.fstat(descriptor)
        if _ipc_metadata_identity(after) != _ipc_metadata_identity(opened):
            raise Phase3CoordinatorError("required live SUT source changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_declared_commit_source_bytes(
    execution_git_sha: str,
    relative_path: str,
) -> bytes:
    """Read the exact source blob declared by the execution commit."""

    if (
        type(execution_git_sha) is not str
        or len(execution_git_sha) != 40
        or any(character not in "0123456789abcdef" for character in execution_git_sha)
    ):
        raise Phase3CoordinatorError("declared execution Git SHA is invalid")
    if relative_path not in PHASE3_EXECUTION_SOURCE_PATHS:
        raise Phase3CoordinatorError(
            "requested commit source is outside the execution source set"
        )
    result = subprocess.run(
        (
            "/usr/bin/git",
            "cat-file",
            "blob",
            f"{execution_git_sha}:{relative_path}",
        ),
        cwd=REPOSITORY_ROOT,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
    )
    if result.returncode != 0 or not result.stdout:
        raise Phase3CoordinatorError("declared execution commit source is unavailable")
    return result.stdout


def _pin_phase3_execution_sources(
    execution_git_sha: str,
) -> Phase3ExecutionSourcePin:
    """Bind all current Phase 3 package bytes before worker spawn."""

    if _run_checked(("/usr/bin/git", "rev-parse", "HEAD")) != execution_git_sha:
        raise Phase3CoordinatorError("live Git HEAD differs from execution Git SHA")
    pinned: list[tuple[str, bytes]] = []
    for relative_path in PHASE3_EXECUTION_SOURCE_PATHS:
        declared = _read_declared_commit_source_bytes(
            execution_git_sha,
            relative_path,
        )
        live = _read_live_sut_source_bytes(relative_path)
        if live != declared:
            raise Phase3CoordinatorError(
                "live execution source differs from the declared execution commit"
            )
        pinned.append((relative_path, declared))
    source_digests = {
        relative: sha256_hex(payload) for relative, payload in pinned
    }
    sut_source_digests = {
        relative: source_digests[relative] for relative in REQUIRED_SUT_SOURCES
    }
    return Phase3ExecutionSourcePin(
        execution_git_sha=execution_git_sha,
        source_bytes_by_path=tuple(pinned),
        source_identity_sha256=(
            phase3_source_identity_sha256(sut_source_digests)
        ),
        execution_source_identity_sha256=(
            _phase3_execution_source_identity_sha256(source_digests)
        ),
    )


def _revalidate_phase3_execution_sources(pin: Phase3ExecutionSourcePin) -> None:
    """Fail if HEAD, a commit blob, or live SUT bytes changed during execution."""

    if type(pin) is not Phase3ExecutionSourcePin:
        raise TypeError("execution source pin has the wrong type")
    if _run_checked(("/usr/bin/git", "rev-parse", "HEAD")) != pin.execution_git_sha:
        raise Phase3CoordinatorError("execution Git HEAD changed during worker lifetime")
    if (
        tuple(relative for relative, _ in pin.source_bytes_by_path)
        != PHASE3_EXECUTION_SOURCE_PATHS
    ):
        raise Phase3CoordinatorError("execution source pin has the wrong source set")
    for relative_path, pinned_bytes in pin.source_bytes_by_path:
        declared = _read_declared_commit_source_bytes(
            pin.execution_git_sha,
            relative_path,
        )
        live = _read_live_sut_source_bytes(relative_path)
        if declared != pinned_bytes or live != pinned_bytes:
            raise Phase3CoordinatorError(
                "execution commit or live SUT source changed during worker lifetime"
            )
    if (
        phase3_source_identity_sha256(pin.sut_source_sha256_by_path)
        != pin.source_identity_sha256
    ):
        raise Phase3CoordinatorError("execution SUT identity changed after pinning")
    if (
        _phase3_execution_source_identity_sha256(pin.source_sha256_by_path)
        != pin.execution_source_identity_sha256
    ):
        raise Phase3CoordinatorError(
            "execution package identity changed after pinning"
        )


def _expected_phase3_raw_audit_operations(
    *,
    point: Any,
    run_id: str,
    git_sha: str,
    cache: BF16CacheIdentity,
    backend: BF16BackendIdentity,
    source_sha256_by_path: Mapping[str, str],
) -> tuple[Phase3AuditOperationKey, ...]:
    """Derive keys only from coordinator state pinned before worker spawn."""

    expected_plan_path = (
        PHASE3_FIXED_PLAN_PATH
        if point.runner_kind.value == "fixed_l"
        else PHASE3_GROWING_PLAN_PATH
    )
    plan_fingerprint = PHASE3_PLAN_FINGERPRINTS[expected_plan_path]
    source_identity = phase3_source_identity_sha256(source_sha256_by_path)
    decode_steps = (
        (0,)
        if point.runner_kind.value == "fixed_l"
        else tuple(range(point.output_steps))
    )
    return tuple(
        Phase3AuditOperationKey.from_point(
            run_id=run_id,
            point=point,
            decode_step=decode_step,
            cache_layout_fingerprint=cache.layout_fingerprint,
            execution_git_sha=git_sha,
            plan_fingerprint=plan_fingerprint,
            hardware_identity_sha256=PHASE3_HARDWARE_FINGERPRINT,
            software_identity_sha256=PHASE3_SOFTWARE_FINGERPRINT,
            model_identity_sha256=PHASE3_MODEL_FINGERPRINT,
            backend_identity_sha256=backend.fingerprint(),
            source_identity_sha256=source_identity,
        )
        for decode_step in decode_steps
    )


def _raw_audit_ingestion_outcome() -> dict[str, Any]:
    return {
        "schema_version": RAW_AUDIT_INGESTION_OUTCOME_SCHEMA_VERSION,
        "worker_evidence_schema_version": None,
        "raw_index_schema_version": None,
        "required": True,
        "attempted": False,
        "sidecar_observed": False,
        "commitment_validation_attempted": False,
        "commitment_validation_passed": False,
        "channel_artifacts_preserved": False,
        "collection_validation_attempted": False,
        "collection_validation_passed": False,
        "declaration_completion_observed": False,
        "semantic_validation_pending": False,
        "passed": False,
        "ingestion_passed": False,
        "collection_completion_passed": False,
        "semantic_validation_attempted": False,
        "semantic_validation_passed": False,
        "semantic_operations": [],
        "scientific_completion_passed": False,
        "terminal_eligible": False,
        "status": "not_attempted",
        "process_audit_passed": False,
        "ipc_digest_validated": False,
        "channel_commitment_validated": False,
        "execution_source_pinned_before_spawn": False,
        "execution_source_revalidated_after_worker_exit": False,
        "expected_operation_count_precomputed_before_spawn": 0,
        "source_root_pinned_before_spawn": False,
        "source_root_path_reopened_after_spawn": False,
        "primary_channel_artifact": None,
        "sidecar_channel_artifact": None,
        "commitment_payload_artifact": None,
        "commitment_digest_artifact": None,
        "commitment_sha256": None,
        "index_run_id": None,
        "index_point_id": None,
        "index_sha256": None,
        "declared_file_count": 0,
        "declared_size_bytes": 0,
        "ingested_file_count": 0,
        "ingested_size_bytes": 0,
        "artifact_file_count": 0,
        "artifact_size_bytes": 0,
        "raw_bytes_embedded_in_ipc": None,
        "failure_reason": None,
    }


def _preserve_phase3_worker_channel_artifacts(
    *,
    run: Any,
    run_id: str,
    point_id: str,
    primary_evidence_bytes: bytes,
    raw_audit_index_bytes: bytes,
) -> tuple[dict[str, Any], str]:
    """Append the exact channels plus their canonical role-bound commitment."""

    if type(primary_evidence_bytes) is not bytes or not primary_evidence_bytes:
        raise Phase3CoordinatorError("primary worker evidence channel is absent")
    if type(raw_audit_index_bytes) is not bytes or not raw_audit_index_bytes:
        raise Phase3CoordinatorError("raw-audit sidecar channel is absent")
    try:
        commitment = build_phase3_worker_channel_commitment(
            run_id=run_id,
            point_id=point_id,
            primary_evidence_bytes=primary_evidence_bytes,
            raw_audit_index_bytes=raw_audit_index_bytes,
        )
        commitment_bytes = canonical_json_bytes(commitment)
        commitment_sha256 = sha256_hex(commitment_bytes)
        independently_derived = phase3_worker_channel_commitment_sha256(
            run_id=run_id,
            point_id=point_id,
            primary_evidence_bytes=primary_evidence_bytes,
            raw_audit_index_bytes=raw_audit_index_bytes,
        )
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise Phase3CoordinatorError(
            "worker channel commitment cannot be constructed"
        ) from error
    if independently_derived != commitment_sha256:
        raise Phase3CoordinatorError("worker channel commitment derivations differ")
    run.write_bytes(TRANSPORT_PRIMARY_CHANNEL_ARTIFACT, primary_evidence_bytes)
    run.write_bytes(TRANSPORT_SIDECAR_CHANNEL_ARTIFACT, raw_audit_index_bytes)
    run.write_bytes(TRANSPORT_COMMITMENT_ARTIFACT, commitment_bytes)
    run.write_bytes(
        TRANSPORT_COMMITMENT_DIGEST_ARTIFACT,
        commitment_sha256.encode("ascii") + b"\n",
    )
    return commitment, commitment_sha256


def _validate_worker_evidence_v1(
    *,
    evidence: Mapping[str, Any],
    expected_run_id: str,
    expected_point_id: str,
    result: Phase3WorkerResult,
    cache_layout_fingerprint: str,
) -> None:
    """Retain the immutable legacy worker-evidence validation semantics."""

    if (
        evidence.get("run_id") != expected_run_id
        or evidence.get("point_id") != expected_point_id
        or evidence.get("worker_result") != result.to_dict()
    ):
        raise Phase3CoordinatorError("worker evidence identity join failed")
    runtime = evidence.get("runtime")
    if isinstance(runtime, Mapping):
        if runtime.get("cache_layout_fingerprint") != cache_layout_fingerprint:
            raise Phase3CoordinatorError("runtime cache fingerprint differs")



def _replay_phase3_raw_audit_semantics(
    *,
    index: Phase3RawAuditRunIndex,
    retained: Mapping[str, bytes],
    execution_source_pin: Phase3ExecutionSourcePin,
    backend_identity: BF16BackendIdentity,
    outcome: dict[str, Any],
) -> None:
    """Independently derive every B-011/B-012 verdict from retained bytes."""

    from kvbench.runtime.allocation_attribution import (
        PHASE3_BACKEND_IDENTITY,
        SplitKCompositeRawInputs,
        build_phase3_production_allocation_binding,
    )
    from kvbench.runtime.gqa_device_dispatch import (
        Phase3AllocationJoinFacts,
        Phase3AllocationRawEvidence,
        combine_phase3_geometry_bound_gqa_allocation_verdict,
        revalidate_phase3_geometry_bound_dispatch_evidence_from_raw,
    )
    from kvbench.runtime.phase3_allocator_controls import (
        verify_phase3_paired_allocator_controls,
    )

    if type(execution_source_pin) is not Phase3ExecutionSourcePin:
        raise Phase3CoordinatorError(
            "semantic replay requires the pinned execution sources"
        )
    if type(backend_identity) is not BF16BackendIdentity:
        raise Phase3CoordinatorError(
            "semantic replay requires the frozen backend identity"
        )
    outcome["collection_validation_passed"] = True
    completed = all(
        record.status == RAW_AUDIT_STATUS_COMPLETED
        for record in index.records
    )
    outcome["collection_completion_passed"] = completed
    if not completed:
        outcome.update(
            {
                "passed": False,
                "semantic_validation_pending": False,
                "scientific_completion_passed": False,
                "terminal_eligible": False,
                "status": "ingested_failed_evidence",
            }
        )
        return

    source_bytes = dict(execution_source_pin.source_bytes_by_path)
    sut_sources = {
        relative: source_bytes[relative]
        for relative in REQUIRED_SUT_SOURCES
    }
    backend_raw = canonical_json_bytes(backend_identity.to_dict())
    if backend_raw.decode("utf-8") != PHASE3_BACKEND_IDENTITY:
        raise Phase3CoordinatorError(
            "semantic replay backend identities differ"
        )

    outcome["semantic_validation_attempted"] = True
    semantic_operations: list[dict[str, Any]] = []
    dispatch_digests: list[str] = []
    allocation_digests: list[str] = []
    output_digests: list[str] = []
    output_finite: list[bool] = []
    historical_predecessors: list[str] = []
    destination_digests: list[str] = []
    provenance_raw: bytes | None = None

    required_bundle_keys = {
        "snapshot",
        "memory_stats_before",
        "memory_stats_after",
        "memory_accounting_before",
        "memory_accounting_after",
        "operation_witness",
        "gqa_allocator_control",
        "mha_allocator_control",
        "audit_sha256_ledger",
    }
    for record in index.records:
        by_kind = {
            declaration.kind: retained[declaration.path]
            for declaration in record.files
        }
        try:
            b011_raw = by_kind["b011_audit"]
            gqa_raw = by_kind["b011_gqa_chrome_trace"]
            mha_raw = by_kind["b011_mha_chrome_trace"]
            allocation_audit_raw = by_kind["b012_allocation_audit"]
            allocation_bundle_raw = by_kind["b012_allocator_snapshot"]
            allocation_trace_raw = by_kind["b012_allocator_trace"]
        except KeyError as error:
            raise Phase3CoordinatorError(
                "completed raw audit operation lacks a required file kind"
            ) from error

        bundle = _parse_canonical_json(
            allocation_bundle_raw + b"\n",
            maximum_bytes=len(allocation_bundle_raw) + 1,
            label="reduced allocator evidence bundle",
        )
        if set(bundle) != required_bundle_keys:
            raise Phase3CoordinatorError(
                "reduced allocator evidence bundle has the wrong fields"
            )
        ledger = bundle["audit_sha256_ledger"]
        witness = bundle["operation_witness"]
        gqa_allocator_raw = canonical_json_bytes(
            bundle["gqa_allocator_control"]
        )
        mha_allocator_raw = canonical_json_bytes(
            bundle["mha_allocator_control"]
        )
        if type(ledger) is not str or not isinstance(witness, Mapping):
            raise Phase3CoordinatorError(
                "reduced allocator evidence bundle is malformed"
            )
        try:
            allocation_raw = Phase3AllocationRawEvidence(
                snapshot_raw=canonical_json_bytes(bundle["snapshot"]),
                trace_raw=allocation_trace_raw,
                memory_stats_before_raw=canonical_json_bytes(
                    bundle["memory_stats_before"]
                ),
                memory_stats_after_raw=canonical_json_bytes(
                    bundle["memory_stats_after"]
                ),
                memory_accounting_before_raw=canonical_json_bytes(
                    bundle["memory_accounting_before"]
                ),
                memory_accounting_after_raw=canonical_json_bytes(
                    bundle["memory_accounting_after"]
                ),
                operation_witness_raw=canonical_json_bytes(witness),
                audit_raw=allocation_audit_raw,
                audit_sha256_ledger_raw=ledger.encode("ascii"),
            )
        except (TypeError, UnicodeEncodeError, ValueError) as error:
            raise Phase3CoordinatorError(
                "reduced allocator evidence cannot reconstruct raw bytes"
            ) from error

        operation_key = record.operation
        dispatch = (
            revalidate_phase3_geometry_bound_dispatch_evidence_from_raw(
                b011_audit_raw=b011_raw,
                operation_key=operation_key,
                gqa_raw=gqa_raw,
                mha_raw=mha_raw,
                backend_identity_raw=backend_raw,
                source_bytes_by_path=sut_sources,
            )
        )
        assert dispatch.gqa.raw_trace is not None
        assert dispatch.mha.raw_trace is not None
        paired = verify_phase3_paired_allocator_controls(
            gqa_raw=gqa_allocator_raw,
            mha_control_raw=mha_allocator_raw,
            operation_key=operation_key,
            gqa_dispatch_trace_raw=gqa_raw,
            mha_dispatch_trace_raw=mha_raw,
        )
        if not paired.passed:
            raise Phase3CoordinatorError(
                "raw paired allocator controls did not verify"
            )
        split_k_inputs = SplitKCompositeRawInputs.from_raw_bytes(
            gqa_dispatch_trace=gqa_raw,
            mha_dispatch_trace=mha_raw,
            gqa_allocator_control=gqa_allocator_raw,
            mha_allocator_control=mha_allocator_raw,
            split_k_pair_multiplicity=(
                paired.split_k_pair_multiplicity
            ),
        )
        production_binding = build_phase3_production_allocation_binding(
            operation_key=operation_key,
            backend_identity=PHASE3_BACKEND_IDENTITY,
            split_k_raw_inputs=split_k_inputs,
        )
        allocation_facts = Phase3AllocationJoinFacts.from_raw_evidence(
            operation_key=operation_key,
            production_binding=production_binding,
            raw_evidence=allocation_raw,
            gqa_dispatch_trace_sha256=dispatch.gqa.raw_trace.sha256,
            mha_dispatch_trace_sha256=dispatch.mha.raw_trace.sha256,
            dispatch_trace_validation_sha256=(
                dispatch.trace_validation.evidence_sha256
            ),
        )
        combined = combine_phase3_geometry_bound_gqa_allocation_verdict(
            dispatch_audit=dispatch,
            allocation_facts=allocation_facts,
        )

        measured_output = witness.get("measured_output")
        measured_after = witness.get("measured_after")
        measured_before = witness.get("measured_before")
        if (
            not isinstance(measured_output, Mapping)
            or not isinstance(measured_after, Mapping)
            or not isinstance(measured_before, Mapping)
            or type(measured_output.get("sha256")) is not str
            or type(measured_output.get("finite")) is not bool
            or type(
                measured_after.get("destination_slot_sha256")
            ) is not str
            or type(
                measured_before.get("historical_prefix_sha256")
            ) is not str
        ):
            raise Phase3CoordinatorError(
                "allocator witness lacks session-chain observations"
            )
        dispatch_digest = sha256_hex(b011_raw)
        allocation_digest = sha256_hex(allocation_audit_raw)
        dispatch_digests.append(dispatch_digest)
        allocation_digests.append(allocation_digest)
        output_digests.append(measured_output["sha256"])
        output_finite.append(measured_output["finite"])
        historical_predecessors.append(
            measured_before["historical_prefix_sha256"]
        )
        destination_digests.append(
            measured_after["destination_slot_sha256"]
        )
        semantic_operations.append(
            {
                "operation_fingerprint_sha256": (
                    operation_key.operation_fingerprint_sha256
                ),
                "dispatch_audit_sha256": dispatch_digest,
                "allocation_audit_sha256": allocation_digest,
                "gqa_verdict": combined.verdict.value,
                "gqa_reasons": list(combined.reasons),
                "device_kernel_families": {
                    "gqa": dispatch.gqa_kernel_sequence.family,
                    "mha_control": dispatch.mha_kernel_sequence.family,
                },
                "allocation_criterion_id": (
                    allocation_facts.criterion_id
                ),
                "allocation_event_count": (
                    allocation_facts.criterion_allocation_event_count
                ),
                "allocation_class_counts": dict(
                    allocation_facts.criterion_class_counts
                ),
                "allocation_failure_reasons": list(
                    allocation_facts.criterion_failure_reasons
                ),
                "allocation_join_sha256": (
                    allocation_facts.evidence_sha256
                ),
                "paired_allocator_control_sha256": {
                    "gqa": sha256_hex(gqa_allocator_raw),
                    "mha_control": sha256_hex(mha_allocator_raw),
                },
                "split_k_pair_multiplicity": [
                    {
                        "num_splits": splits,
                        "pair_count": count,
                    }
                    for splits, count in (
                        paired.split_k_pair_multiplicity
                    )
                ],
            }
        )
        if (
            combined.verdict
            is not GQAVerdict.NONMATERIALIZATION_VERIFIED
        ):
            outcome["semantic_operations"] = semantic_operations
            raise Phase3CoordinatorError(
                "raw-derived GQA non-materialization did not verify"
            )
        if record.operation.decode_step == 0:
            provenance_raw = by_kind.get(
                "phase3_session_provenance"
            )

    if provenance_raw is None:
        raise Phase3CoordinatorError(
            "completed raw audit run lacks session provenance"
        )
    provenance = _parse_canonical_json(
        provenance_raw + b"\n",
        maximum_bytes=len(provenance_raw) + 1,
        label="Phase 3 session provenance",
    )
    expected_provenance_keys = {
        "schema_version",
        "receipt_sha256",
        "cache_pointers",
        "cache_layout_fingerprint",
        "operation_fingerprints",
        "dispatch_audit_sha256",
        "allocation_audit_sha256",
        "audit_output_sha256",
        "audit_output_finite",
        "graph_retained",
        "prefix_sha256",
        "history_chain_sha256",
    }
    receipt_sha256 = provenance.get("receipt_sha256")
    cache_pointers = provenance.get("cache_pointers")
    prefix_sha256 = provenance.get("prefix_sha256")
    if (
        set(provenance) != expected_provenance_keys
        or provenance.get("schema_version")
        != "kvbench-phase3-endpoint-session-1.0.0"
        or type(receipt_sha256) is not str
        or len(receipt_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in receipt_sha256
        )
        or not isinstance(cache_pointers, Mapping)
        or set(cache_pointers)
        != {
            "keys_data_ptr",
            "values_data_ptr",
            "keys_storage_ptr",
            "values_storage_ptr",
        }
        or any(
            type(value) is not int or value <= 0
            for value in cache_pointers.values()
        )
        or provenance.get("cache_layout_fingerprint")
        != index.records[0].operation.cache_layout_fingerprint
        or provenance.get("operation_fingerprints")
        != [
            record.operation.operation_fingerprint_sha256
            for record in index.records
        ]
        or provenance.get("dispatch_audit_sha256")
        != dispatch_digests
        or provenance.get("allocation_audit_sha256")
        != allocation_digests
        or provenance.get("audit_output_sha256") != output_digests
        or provenance.get("audit_output_finite") != output_finite
        or output_finite != [True] * len(index.records)
        or provenance.get("graph_retained")
        is not (
            index.records[0].operation.graph_mode.value
            == "cuda_graph"
        )
        or type(prefix_sha256) is not str
        or len(prefix_sha256) != 64
    ):
        raise Phase3CoordinatorError(
            "session provenance differs from raw-derived operation evidence"
        )

    chain = prefix_sha256
    for predecessor, destination in zip(
        historical_predecessors,
        destination_digests,
        strict=True,
    ):
        if predecessor != chain:
            raise Phase3CoordinatorError(
                "allocator witness history chain is discontinuous"
            )
        chain = hashlib.sha256(
            f"{chain}:{destination}".encode("ascii")
        ).hexdigest()
    if provenance.get("history_chain_sha256") != chain:
        raise Phase3CoordinatorError(
            "session provenance history chain differs"
        )

    outcome["semantic_operations"] = semantic_operations
    terminal_eligible = bool(
        outcome.get("process_audit_passed") is True
        and outcome.get("commitment_validation_passed") is True
        and outcome.get(
            "execution_source_revalidated_after_worker_exit"
        )
        is True
    )
    outcome.update(
        {
            "passed": terminal_eligible,
            "semantic_validation_passed": True,
            "semantic_validation_pending": False,
            "scientific_completion_passed": True,
            "terminal_eligible": terminal_eligible,
            "status": (
                "validated"
                if terminal_eligible
                else "semantic_validated_transport_incomplete"
            ),
            "failure_reason": None,
        }
    )

def _ingest_worker_evidence_v2(
    *,
    evidence: Mapping[str, Any],
    expected_run_id: str,
    expected_point_id: str,
    raw_audit_root_fd: int,
    raw_audit_owner_uid: int,
    expected_operations: tuple[Phase3AuditOperationKey, ...],
    run: Any,
    outcome: dict[str, Any],
    execution_source_pin: Phase3ExecutionSourcePin | None = None,
    backend_identity: BF16BackendIdentity | None = None,
) -> Phase3RawAuditRunIndex:
    """Validate the minimal v2 envelope and append its pinned raw bytes once."""

    expected_keys = {
        "schema_version",
        "raw_audit_run_index",
        "raw_audit_run_index_sha256",
    }
    if set(evidence) != expected_keys:
        raise Phase3CoordinatorError(
            "worker evidence v2 contains fields outside the raw-index envelope"
        )
    index_payload = evidence.get("raw_audit_run_index")
    declared_digest = evidence.get("raw_audit_run_index_sha256")
    if type(index_payload) is not dict or type(declared_digest) is not str:
        raise Phase3CoordinatorError("worker evidence v2 envelope is malformed")
    try:
        index_bytes = canonical_json_bytes(index_payload)
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise Phase3CoordinatorError(
            "worker evidence v2 index cannot be reconstructed canonically"
        ) from error
    observed_digest = sha256_hex(index_bytes)
    outcome["index_sha256"] = observed_digest
    if declared_digest != observed_digest:
        raise Phase3CoordinatorError("raw-audit run-index SHA-256 differs")
    try:
        index = parse_phase3_raw_audit_run_index_bytes(index_bytes)
    except Phase3RawAuditEvidenceError as error:
        raise Phase3CoordinatorError("raw-audit run index is invalid") from error
    outcome["index_run_id"] = index.run_id
    outcome["index_point_id"] = index.point_id
    if index.run_id != expected_run_id or index.point_id != expected_point_id:
        raise Phase3CoordinatorError("raw-audit run-index identity join failed")
    observed_operations = tuple(record.operation for record in index.records)
    if observed_operations != expected_operations:
        raise Phase3CoordinatorError(
            "raw-audit operation provenance differs from coordinator-owned state"
        )
    declared_completion = all(
        record.status == RAW_AUDIT_STATUS_COMPLETED for record in index.records
    )

    declarations = tuple(
        item for record in index.records for item in record.files
    )
    outcome["declared_file_count"] = len(declarations)
    outcome["declared_size_bytes"] = sum(
        item.size_bytes for item in declarations
    )
    try:
        retained = ingest_phase3_raw_audit_evidence_fd(
            raw_audit_root_fd,
            index,
            expected_owner_uid=raw_audit_owner_uid,
        )
    except Phase3RawAuditEvidenceError as error:
        raise Phase3CoordinatorError(
            "raw-audit source evidence failed secure ingestion"
        ) from error
    expected_paths = {item.path for item in declarations}
    if set(retained) != expected_paths or any(
        type(payload) is not bytes for payload in retained.values()
    ):
        raise Phase3CoordinatorError(
            "raw-audit ingestion returned an unexpected byte set"
        )
    outcome["ingested_file_count"] = len(retained)
    outcome["ingested_size_bytes"] = sum(len(value) for value in retained.values())
    for relative in sorted(retained):
        payload = retained[relative]
        run.write_bytes(f"raw/audits/files/{relative}", payload)
        outcome["artifact_file_count"] += 1
        outcome["artifact_size_bytes"] += len(payload)
    run.write_bytes("raw/audits/index.json", index_bytes)
    outcome.update(
        {
            "passed": False,
            "ingestion_passed": True,
            "collection_validation_passed": False,
            "collection_completion_passed": False,
            "declaration_completion_observed": declared_completion,
            "semantic_validation_attempted": False,
            "semantic_validation_passed": False,
            "semantic_validation_pending": declared_completion,
            "scientific_completion_passed": False,
            "terminal_eligible": False,
            "status": (
                "ingested_declared_complete_pending_semantic_validation"
                if declared_completion
                else "ingested_failed_evidence"
            ),
            "raw_bytes_embedded_in_ipc": False,
            "failure_reason": None,
        }
    )
    if (execution_source_pin is None) != (backend_identity is None):
        raise Phase3CoordinatorError(
            "semantic replay inputs are incomplete"
        )
    if execution_source_pin is not None and backend_identity is not None:
        try:
            _replay_phase3_raw_audit_semantics(
                index=index,
                retained=retained,
                execution_source_pin=execution_source_pin,
                backend_identity=backend_identity,
                outcome=outcome,
            )
        except BaseException as error:
            failure_reason = (
                f"{type(error).__name__}: "
                f"{' '.join(str(error).split())}"
            )[:1000]
            outcome.update(
                {
                    "passed": False,
                    "semantic_validation_pending": False,
                    "semantic_validation_passed": False,
                    "scientific_completion_passed": False,
                    "terminal_eligible": False,
                    "status": "semantic_validation_failed",
                    "failure_reason": failure_reason,
                }
            )
            if isinstance(error, Phase3CoordinatorError):
                raise
            raise Phase3CoordinatorError(
                "raw-audit semantic replay failed"
            ) from error
    return index


def _process_snapshot(*, pid: int | None = None, start_ticks: int | None = None) -> dict[str, Any]:
    argv = [str(PYTHON_EXECUTABLE), str(PROCESS_QUERY)]
    if pid is not None or start_ticks is not None:
        if pid is None or start_ticks is None:
            raise Phase3CoordinatorError("supervised process identity is incomplete")
        argv.extend(
            [
                "--supervised-root-pid",
                str(pid),
                "--supervised-root-start-ticks",
                str(start_ticks),
            ]
        )
    result = subprocess.run(
        argv,
        cwd=REPOSITORY_ROOT,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        env={
            "PATH": "/usr/bin:/bin",
            "LC_ALL": "C",
            "LANG": "C",
            "TZ": "UTC",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    payload = _parse_canonical_json(
        result.stdout,
        maximum_bytes=16 * 1024 * 1024,
        label="GPU process snapshot",
    )
    if result.returncode != payload.get("query_exit_code"):
        raise Phase3CoordinatorError("GPU process snapshot exit-code mismatch")
    return payload


def _snapshot_clean(snapshot: Mapping[str, Any], *, allow_supervised: bool) -> bool:
    allowed = snapshot.get("allowed_compute_processes")
    return bool(
        snapshot.get("query_exit_code") == 0
        and snapshot.get("errors") == []
        and snapshot.get("foreign_compute_processes") == []
        and snapshot.get("unknown_processes") == []
        and isinstance(allowed, list)
        and (allow_supervised or allowed == [])
    )


def _exclusive_release(path: Path) -> None:
    try:
        publish_bytes_no_replace(path, b"release\n")
    except ProcessSupervisionError as error:
        raise Phase3CoordinatorError(
            "worker release publication failed"
        ) from error


def _pidfd_open(pid: int) -> tuple[bool, int | None]:
    opener = getattr(os, "pidfd_open", None)
    if opener is None:
        return False, None
    try:
        return True, opener(pid, 0)
    except OSError:
        return True, None


def _nonreaping_exit_observed(registry: RunOwnedProcessRegistry) -> bool:
    """Observe exit through pidfd or waitid WNOWAIT without reaping."""

    if registry.exit_observed:
        return True
    pidfd = registry.pidfd
    if pidfd is not None:
        poller = select.poll()
        poller.register(pidfd, select.POLLIN | select.POLLHUP | select.POLLERR)
        if poller.poll(0):
            registry.note_exit_observed()
            return True
        return False
    try:
        observation = os.waitid(
            os.P_PID,
            registry.identity.process.pid,
            os.WEXITED | os.WNOHANG | os.WNOWAIT,
        )
    except ChildProcessError as error:
        raise Phase3CoordinatorError(
            "registered worker was reaped outside the supervisor"
        ) from error
    if observation is not None:
        registry.note_exit_observed()
        return True
    return False


def _wait_for_ready(
    registry: RunOwnedProcessRegistry,
    ready_path: Path,
    handshake_directory: Path,
) -> dict[str, Any]:
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            registry.refresh_handshake_directory(handshake_directory)
        except ProcessSupervisionError as error:
            raise Phase3CoordinatorError("worker_started event is invalid") from error
        ready_raw: bytes | None = None
        if HandshakeStage.WORKER_STARTED in registry.observed_worker_stages:
            try:
                ready_raw = read_published_bytes(
                    ready_path,
                    maximum_bytes=16 * 1024,
                )
            except FileNotFoundError:
                pass
            except ProcessSupervisionError as error:
                raise Phase3CoordinatorError(
                    "worker readiness path is unsafe"
                ) from error
        if ready_raw is not None:
            payload = _parse_canonical_json(
                ready_raw,
                maximum_bytes=16 * 1024,
                label="worker readiness",
            )
            if (
                payload.get("pid") != registry.identity.process.pid
                or not isinstance(payload.get("process_start_time_ticks"), int)
                or payload.get("process_start_time_ticks", -1) < 0
                or payload.get("process_start_time_ticks")
                != registry.identity.process.start_time_ticks
                or payload.get("cuda_imported") is not False
            ):
                raise Phase3CoordinatorError("worker readiness identity differs")
            return payload
        if _nonreaping_exit_observed(registry):
            try:
                registry.refresh_handshake_directory(handshake_directory)
            except ProcessSupervisionError as error:
                raise Phase3CoordinatorError("worker handshake is invalid") from error
            raise Phase3CoordinatorError(
                "registered worker exited before worker_started readiness completed"
            )
        time.sleep(0.05)
    raise Phase3CoordinatorError("worker readiness timed out")


def _wait_for_registered_exit_observation(
    registry: RunOwnedProcessRegistry,
    *,
    timeout_seconds: float,
) -> bool:
    """Wait for pidfd/waitid WNOWAIT readiness without reaping the child."""

    if not isinstance(timeout_seconds, (int, float)) or isinstance(
        timeout_seconds, bool
    ) or timeout_seconds <= 0:
        raise ValueError("registered-worker exit timeout is invalid")
    deadline = time.monotonic() + float(timeout_seconds)
    while True:
        if _nonreaping_exit_observed(registry):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.05, remaining))


def _send_owned_process_signal(
    process: subprocess.Popen[bytes],
    requested_signal: int,
    *,
    pidfd: int | None,
    expected_identity: ProcessIdentity | None,
    ownership_label: str,
) -> bool:
    """Signal a still-owned process without trusting a reusable PID alone."""

    signal_name = signal.Signals(requested_signal).name
    pidfd_sender = getattr(signal, "pidfd_send_signal", None)
    if pidfd is not None and callable(pidfd_sender):
        try:
            pidfd_sender(pidfd, requested_signal)
        except ProcessLookupError:
            return False
        except OSError as error:
            raise Phase3CoordinatorError(
                f"{ownership_label} pidfd {signal_name} failed"
            ) from error
        return True
    if expected_identity is None:
        raise Phase3CoordinatorError(
            f"{ownership_label} lacks a stable identity before {signal_name}"
        )
    if process.pid != expected_identity.pid:
        raise Phase3CoordinatorError(
            f"{ownership_label} process handle PID differs before {signal_name}"
        )
    try:
        current_identity = read_process_identity(process.pid)
    except ProcessIdentityUnavailable:
        return False
    if current_identity != expected_identity:
        raise Phase3CoordinatorError(
            f"{ownership_label} identity changed before {signal_name}"
        )
    try:
        os.killpg(process.pid, requested_signal)
    except ProcessLookupError:
        return False
    except OSError as error:
        raise Phase3CoordinatorError(
            f"{ownership_label} process-group {signal_name} failed"
        ) from error
    return True


def _signal_registered_worker(
    process: subprocess.Popen[bytes],
    registry: RunOwnedProcessRegistry,
    requested_signal: int,
) -> bool:
    """Signal the registered identity, recording disappearance without reuse."""

    sent = _send_owned_process_signal(
        process,
        requested_signal,
        pidfd=registry.pidfd,
        expected_identity=registry.identity.process,
        ownership_label="registered worker",
    )
    if not sent:
        registry.note_exit_observed()
    return sent


def _reap_registered_worker(
    process: subprocess.Popen[bytes],
    registry: RunOwnedProcessRegistry,
    *,
    timeout_seconds: float,
) -> tuple[int, Any]:
    """Perform the sole process.wait only after a non-reaping exit observation."""

    if not registry.exit_observed:
        raise Phase3CoordinatorError(
            "registered worker cannot be waited before non-reaping exit observation"
        )
    if registry.reaped:
        raise Phase3CoordinatorError("registered worker was already reaped")
    try:
        returncode = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        raise Phase3CoordinatorError(
            "registered worker wait timed out after exit observation"
        ) from error
    try:
        event = registry.record_supervisor_reaped(
            returncode,
            recorded_at_utc=_utc_now(),
        )
    except ProcessSupervisionError as error:
        raise Phase3CoordinatorError(
            "registered worker reap evidence could not be recorded"
        ) from error
    return returncode, event


def _terminate_registered_worker(
    process: subprocess.Popen[bytes],
    registry: RunOwnedProcessRegistry,
    *,
    handshake_directory: Path | None = None,
) -> tuple[int, Any]:
    """Request TERM/KILL, observe without reaping, then reap exactly once."""

    if registry.reaped:
        raise Phase3CoordinatorError("registered worker was already reaped")
    _signal_registered_worker(process, registry, signal.SIGTERM)
    if not _wait_for_registered_exit_observation(
        registry,
        timeout_seconds=10.0,
    ):
        _signal_registered_worker(process, registry, signal.SIGKILL)
        if not _wait_for_registered_exit_observation(
            registry,
            timeout_seconds=10.0,
        ):
            raise Phase3CoordinatorError(
                "registered worker exit was not observed after SIGKILL"
            )
    if handshake_directory is not None:
        try:
            registry.refresh_handshake_directory(handshake_directory)
        except ProcessSupervisionError:
            # Ownership reaping must still complete; malformed stages remain
            # absent and therefore produce an owned-worker failure verdict.
            pass
    return _reap_registered_worker(
        process,
        registry,
        timeout_seconds=10.0,
    )


def _terminate_unregistered_worker(
    process: subprocess.Popen[bytes],
    *,
    pidfd: int | None = None,
    expected_identity: ProcessIdentity | None = None,
) -> int:
    """Clean up a pre-registration child without trusting its PID alone."""

    if process.returncode is not None:
        return process.returncode
    _send_owned_process_signal(
        process,
        signal.SIGTERM,
        pidfd=pidfd,
        expected_identity=expected_identity,
        ownership_label="unregistered worker",
    )
    try:
        return process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        _send_owned_process_signal(
            process,
            signal.SIGKILL,
            pidfd=pidfd,
            expected_identity=expected_identity,
            ownership_label="unregistered worker",
        )
        return process.wait(timeout=10)


def _device_observation(value: object) -> DeviceProcessObservation:
    if not isinstance(value, Mapping):
        raise Phase3CoordinatorError("GPU process record is not an object")
    gpu_uuid = value.get("gpu_uuid")
    pid = value.get("pid")
    start_ticks = value.get("process_start_time_ticks")
    if (
        not isinstance(gpu_uuid, str)
        or not isinstance(pid, int)
        or isinstance(pid, bool)
        or not isinstance(start_ticks, int)
        or isinstance(start_ticks, bool)
    ):
        raise Phase3CoordinatorError("GPU process record identity is malformed")
    return DeviceProcessObservation(
        gpu_uuid=gpu_uuid,
        pid=pid,
        process_start_time_ticks=None if start_ticks == 0 else start_ticks,
    )


def _registry_snapshot_verdict(
    snapshot: Mapping[str, Any],
    registry: RunOwnedProcessRegistry,
    *,
    terminal_resolution_allowed: bool,
) -> dict[str, Any]:
    """Join raw device records to the registry without weakening foreign checks."""

    errors = snapshot.get("errors")
    if not isinstance(errors, list) or any(not isinstance(item, str) for item in errors):
        registry.note_unverified_device_evidence(
            "GPU process snapshot errors are malformed"
        )
        raise Phase3CoordinatorError("GPU process snapshot errors are malformed")
    observations: list[DeviceProcessObservation] = []
    unknown_observations: list[DeviceProcessObservation] = []
    for key in (
        "allowed_compute_processes",
        "foreign_compute_processes",
        "unknown_processes",
    ):
        records = snapshot.get(key)
        if not isinstance(records, list):
            registry.note_unverified_device_evidence(
                "GPU process snapshot list is malformed"
            )
            raise Phase3CoordinatorError("GPU process snapshot list is malformed")
        try:
            parsed = [_device_observation(item) for item in records]
        except Phase3CoordinatorError:
            registry.note_unverified_device_evidence(
                "GPU process record identity is malformed"
            )
            raise
        observations.extend(parsed)
        if key == "unknown_processes":
            unknown_observations.extend(parsed)
    registered = registry.identity
    if terminal_resolution_allowed and any(
        item.pid == registered.process.pid
        and item.gpu_uuid == registered.gpu_uuid
        and item.process_start_time_ticks is None
        for item in unknown_observations
    ):
        registry.observe_proc_start_time(None)
    try:
        verdict = registry.classify_device_snapshot(tuple(observations))
    except ProcessSupervisionError as error:
        registry.note_unverified_device_evidence(
            "GPU process ownership join failed"
        )
        raise Phase3CoordinatorError("GPU process ownership join failed") from error
    registered_pmon_gap_error = (
        f"compute_apps GPU {registered.gpu_uuid} "
        f"PID {registered.process.pid} has no pmon process type"
    )
    registered_proc_unavailable_prefix = (
        f"cannot read /proc/{registered.process.pid}/stat"
    )
    registered_query_race_owned = bool(
        errors
        and snapshot.get("query_exit_code") == 2
        and verdict.disposition is SnapshotDisposition.OWNED_ONLY
        and registered_pmon_gap_error in errors
        and all(
            error == registered_pmon_gap_error
            or (
                terminal_resolution_allowed
                and error.startswith(registered_proc_unavailable_prefix)
            )
            for error in errors
        )
    )
    terminal_resolution_used = bool(
        registered_query_race_owned
        and terminal_resolution_allowed
        and any(
            error.startswith(registered_proc_unavailable_prefix)
            for error in errors
        )
    )
    query_exit_code = snapshot.get("query_exit_code")
    clean_query = query_exit_code == 0 and errors == []
    query_evidence_hard_failure = (
        not clean_query and not registered_query_race_owned
    )
    if query_evidence_hard_failure:
        registry.note_unverified_device_evidence(
            "GPU process query failed outside exact registered worker resolution"
        )
    passed = bool(
        not verdict.hard_failure
        and (clean_query or registered_query_race_owned)
    )
    return {
        "passed": passed,
        "terminal_registered_process_resolution": terminal_resolution_used,
        "query_evidence_hard_failure": query_evidence_hard_failure,
        "registry_verdict": verdict.to_dict(),
        "raw_query_exit_code": query_exit_code,
        "raw_errors": errors,
    }


def _cache_identity(
    point: Any,
    *,
    implementation_sha256: str,
) -> BF16CacheIdentity:
    if (
        type(implementation_sha256) is not str
        or len(implementation_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in implementation_sha256
        )
    ):
        raise Phase3CoordinatorError("pinned cache implementation digest is invalid")
    capacity = point.context_length + point.output_steps
    workspace_bytes = 32 * point.batch_size * (32 + 8) * 1 * 64 * 2
    payload = {
        "schema_version": BF16CacheIdentity.SCHEMA_VERSION,
        "layout_name": "layers_batch_kv_heads_context_head_dim",
        "dtype": "bfloat16",
        "num_layers": 32,
        "batch_size": point.batch_size,
        "num_kv_heads": 8,
        "capacity": capacity,
        "head_dim": 128,
        "tensor_storage_bytes": (
            2 * 32 * point.batch_size * 8 * capacity * 128 * 2
        ),
        "padding_bytes": 0,
        "workspace_bytes": workspace_bytes,
        "device": "cuda:0",
        "implementation_sha256": implementation_sha256,
        "layout_fingerprint": derive_cache_layout_fingerprint(
            num_layers=32,
            batch_size=point.batch_size,
            num_kv_heads=8,
            capacity=capacity,
            head_dim=128,
            device="cuda:0",
            workspace_bytes=workspace_bytes,
            implementation_sha256=implementation_sha256,
        ),
    }
    return BF16CacheIdentity.from_dict(payload)


def _validate_cache_source_join(
    cache: BF16CacheIdentity,
    source_pin: Phase3ExecutionSourcePin,
) -> None:
    """Require cache layout identity to use the exact pinned implementation."""

    if type(cache) is not BF16CacheIdentity:
        raise TypeError("cache source join has the wrong cache type")
    if type(source_pin) is not Phase3ExecutionSourcePin:
        raise TypeError("cache source join has the wrong source-pin type")
    static_cache_relative = STATIC_CACHE_SOURCE.relative_to(
        REPOSITORY_ROOT
    ).as_posix()
    expected = source_pin.source_sha256_by_path.get(static_cache_relative)
    if expected is None or cache.implementation_sha256 != expected:
        raise Phase3CoordinatorError(
            "cache identity differs from pinned static-cache source"
        )


def _backend_identity_stdlib() -> BF16BackendIdentity:
    """Verify backend bytes without importing PyTorch in the coordinator."""

    expected_sources = {
        "include/ATen/native/transformers/cuda/flash_attn/flash_api.h": (
            "1474aa79d8aa6ce39984dbc3c0aad9dba283ab819f034370e5cfb70980524ee7"
        ),
        "lib/libtorch_cuda.so": (
            "b248fb7e9935440965e4736eea48868b315ba41012734b7ce058fc0a2d0b1984"
        ),
        "nn/attention/__init__.py": (
            "56e10b6f965cc050db782dd4dc472097c9b02ec5b5fe3ab2c8b04055c0b0bbe0"
        ),
        "nn/attention/varlen.py": (
            "2f5384e0bc8ce371d00a1c09d38ad019517009798e7cb3434f56cf4b9fa351ea"
        ),
        "nn/functional.py": (
            "27493186ee22f811b553e31d9c804d4d46716d1be62d034d731537f66f27ef19"
        ),
    }
    for relative, expected in expected_sources.items():
        path = TORCH_PACKAGE_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase3CoordinatorError("frozen backend source bytes differ")
    return BF16BackendIdentity.from_dict(
        {
            "schema_version": BF16BackendIdentity.SCHEMA_VERSION,
            "backend_id": "torch_sdpa_flash_gqa",
            "torch_version": "2.12.1+cu130",
            "torch_git_sha": "7269437d655783a26cba32aa88195b741ff496aa",
            "cuda_runtime_version": "13.0",
            "cudnn_version": "9.20.0",
            "triton_version": "3.7.1",
            "flash_generation": "FA2",
            "flash_version": "2.5.7",
            "dispatch_api": "torch.nn.functional.scaled_dot_product_attention",
            "selected_backend": "flash_attention",
            "enable_gqa": True,
            "compile_mode": "disabled",
            "source_artifacts": [
                {"path": path, "sha256": digest}
                for path, digest in sorted(expected_sources.items())
            ],
        }
    )


def _worker_argv(plan_path: str, point: Any, run_id: str) -> tuple[str, ...]:
    return (
        PHASE3_PYTHON_EXECUTABLE,
        "-m",
        "kvbench",
        "phase3-worker",
        "--plan",
        plan_path,
        "--point-id",
        point.point_id,
        "--replicate",
        str(point.process_replicate),
        "--run-id",
        run_id,
    )


def _initial_manifest(
    *,
    bundle: ExperimentBundle,
    plan_path: str,
    point: Any,
    run_id: str,
    created_at: str,
    git_sha: str,
    environment_sha256: str,
    backend: BF16BackendIdentity,
    cache: BF16CacheIdentity,
) -> Phase3RunManifest:
    method_fingerprint = MethodConfigFingerprint.from_config(
        bundle.methods[0],
        "bf16",
    )
    adapter = build_method_adapter(
        bundle.methods[0],
        MethodRuntimeContext(
            model_id=bundle.model.model_id,
            model_revision=bundle.model.revision,
            backend_id=backend.backend_id,
            backend_fingerprint=backend.fingerprint(),
            num_layers=bundle.model.geometry.num_hidden_layers,
            num_query_heads=bundle.model.geometry.num_query_heads,
            num_kv_heads=bundle.model.geometry.num_kv_heads,
            head_dim=bundle.model.geometry.head_dim,
        ),
    )
    adapter_config_fingerprint = adapter.config_fingerprint(
        cache.layout_fingerprint
    )

    command = Phase3CommandSpec(
        schema_version=Phase3CommandSpec.SCHEMA_VERSION,
        argv=_worker_argv(plan_path, point, run_id),
        working_directory=PHASE3_REPOSITORY_ROOT,
        environment_sha256=environment_sha256,
        dry_run=False,
    )
    payload = {
        "schema_version": Phase3RunManifest.SCHEMA_VERSION,
        "artifact_schema_version": Phase3RunManifest.ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "created",
        "created_at_utc": created_at,
        "started_at_utc": None,
        "finished_at_utc": None,
        "run_kind": RunKind.PHASE3_ADMISSION.value,
        "runner_kind": point.runner_kind.value,
        "graph_mode": point.graph_mode.value,
        "claim_class": ClaimClass.NONE.value,
        "measurement_scope": MeasurementScope.NATIVE_HOST_ADMISSION.value,
        "performance_claim_eligible": False,
        "plan_source": {
            "kind": ConfigSourceKind.PATH.value,
            "path": plan_path,
            "canonical_inline_json": None,
            "sha256": bundle.plan.fingerprint(),
        },
        "plan_fingerprint": bundle.plan.fingerprint(),
        "point_id": point.point_id,
        "point_fingerprint": derive_phase3_point_fingerprint(point.point_id),
        "git_sha": git_sha,
        "git_dirty": False,
        "container_digest": None,
        "hardware_id": PHASE3_HARDWARE_ID,
        "hardware_fingerprint": PHASE3_HARDWARE_FINGERPRINT,
        "native_g0_status": GateDisposition.PASS.value,
        "e00_run_id": PHASE3_E00_RUN_ID,
        "e00_manifest_sha256": PHASE3_E00_MANIFEST_SHA256,
        "blocker_b010": "OPEN",
        "gpu_uuid": PHASE3_GPU_UUID,
        "gpu_full_name": PHASE3_GPU_FULL_NAME,
        "pci_bus_id": PHASE3_PCI_BUS_ID,
        "pci_device_id": PHASE3_PCI_DEVICE_ID,
        "driver_version": PHASE3_DRIVER_VERSION,
        "software_environment_id": PHASE3_SOFTWARE_ENVIRONMENT_ID,
        "software_fingerprint": PHASE3_SOFTWARE_FINGERPRINT,
        "model_identity": bundle.model.to_dict(),
        "model_fingerprint": bundle.model.fingerprint(),
        "method": "bf16",
        "method_config_id": "bf16",
        "method_config_fingerprint": method_fingerprint.to_dict(),
        "adapter_config_fingerprint": adapter_config_fingerprint,
        "contract_fingerprint": PHASE3_CONTRACT_FINGERPRINT,
        "measurement_protocol_fingerprint": (
            PHASE3_MEASUREMENT_PROTOCOL_FINGERPRINT
        ),
        "backend_identity": backend.to_dict(),
        "backend_fingerprint": backend.fingerprint(),
        "cache_identity": cache.to_dict(),
        "batch_size": point.batch_size,
        "context_length": point.context_length,
        "output_steps": point.output_steps,
        "warmup_count": bundle.plan.measurement.warmup_count,
        "measured_count": bundle.plan.measurement.measured_count,
        "measured_batches": bundle.plan.measurement.measured_batches,
        "count_unit": bundle.plan.measurement.count_unit.value,
        "random_seed": bundle.plan.measurement.seed,
        "process_replicate": point.process_replicate,
        "quality": bundle.plan.quality.to_dict(),
        "command": command.to_dict(),
        "inventory_path": None,
        "failure_reason": None,
    }
    return Phase3RunManifest.from_dict(payload)


def _failed_result(
    *,
    bundle: ExperimentBundle,
    point: Any,
    run_id: str,
    reason: str,
) -> Phase3WorkerResult:
    expected = (
        bundle.plan.measurement.measured_count
        * bundle.plan.measurement.measured_batches
        if point.runner_kind.value == "fixed_l"
        else point.output_steps
        * bundle.plan.measurement.measured_count
        * bundle.plan.measurement.measured_batches
    )
    return Phase3WorkerResult(
        schema_version=Phase3WorkerResult.SCHEMA_VERSION,
        run_id=run_id,
        point_id=point.point_id,
        runner_kind=point.runner_kind,
        count_unit=bundle.plan.measurement.count_unit,
        status=RunStatus.ABORTED,
        expected_operations=expected,
        completed_operations=0,
        failed_operations=0,
        output_checksum=None,
        failure_reason=reason[:1000],
    )


def _write_runtime_artifacts(run: Any, evidence: Mapping[str, Any]) -> None:
    run.write_json("raw/worker_evidence.json", evidence)
    runtime = evidence.get("runtime")
    numerical = evidence.get("numerical")
    if isinstance(numerical, Mapping):
        run.write_json("numerical/agreement.json", numerical)
    if not isinstance(runtime, Mapping):
        return
    timing = runtime.get("timing")
    if isinstance(timing, Mapping):
        run.write_json(
            "raw/timing.json",
            {
                **timing,
                "quality_status": "unvalidated",
                "claim_eligibility": "performance_only",
                "performance_claim_eligible": False,
                "measurement_scope": "native_host_admission",
                "profiler_instrumented": False,
            },
        )
    run.write_json(
        "allocation/audit.json",
        {
            "allocation": runtime.get("allocation"),
            "memory_evidence": runtime.get("memory_evidence"),
            "cache_accounting": runtime.get("cache_accounting"),
            "instrumented_duration_reported_as_timing": False,
        },
    )
    run.write_json(
        "gqa/audit.json",
        {
            "source": runtime.get("gqa_source"),
            "cache_geometry": runtime.get("gqa_cache_geometry"),
            "operator": runtime.get("gqa_operator"),
            "operators": runtime.get("gqa_operators"),
            "mha_control": runtime.get("mha_control"),
            "prefill_backend": runtime.get("prefill_backend"),
            "decode_backend": runtime.get("backend"),
        },
    )
    run.write_json(
        "telemetry/snapshots.json",
        {
            "before": runtime.get("telemetry_before"),
            "after": runtime.get("telemetry_after"),
            "sampling_interval_seconds": runtime.get(
                "telemetry_sampling_interval_seconds"
            ),
            "queried_inside_decode_hot_path": False,
            "stability_inference": False,
        },
    )


def _resolve_phase3_terminal_status(
    *,
    result: Phase3WorkerResult,
    process_audit_passed: bool,
    worker_evidence_valid: bool,
    raw_audit_outcome: Mapping[str, Any],
    failure_reason: str | None,
) -> tuple[RunStatus, str | None]:
    """Keep safely ingested failed audits terminal but never completed."""

    run_evidence_accepted = process_audit_passed and worker_evidence_valid
    completed_raw_audit_eligible = bool(
        raw_audit_outcome.get("ingestion_passed") is True
        and raw_audit_outcome.get("scientific_completion_passed") is True
        and raw_audit_outcome.get("terminal_eligible") is True
    )
    if (
        run_evidence_accepted
        and result.status is RunStatus.COMPLETED
        and not completed_raw_audit_eligible
    ):
        return (
            RunStatus.ABORTED,
            "worker reported completed without complete scientific raw-audit evidence",
        )
    if run_evidence_accepted:
        return result.status, result.failure_reason
    return RunStatus.ABORTED, failure_reason or "run evidence validation failed"


def _resolved_phase3_worker_result(
    result: Phase3WorkerResult,
    *,
    final_status: RunStatus,
    final_reason: str | None,
) -> Phase3WorkerResult:
    """Make the coordinator-owned terminal result match the run manifest."""

    if type(result) is not Phase3WorkerResult:
        raise TypeError("terminal worker result has the wrong type")
    if type(final_status) is not RunStatus or not final_status.is_terminal:
        raise TypeError("terminal worker status is invalid")
    if final_status is RunStatus.COMPLETED:
        if final_reason is not None:
            raise Phase3CoordinatorError("completed terminal result has a reason")
    elif type(final_reason) is not str or not final_reason:
        raise Phase3CoordinatorError("failed terminal result lacks a reason")
    payload = result.to_dict()
    payload.update(
        {
            "status": final_status.value,
            "failure_reason": final_reason,
        }
    )
    return Phase3WorkerResult.from_dict(payload)


def _terminal_manifest(
    initial: Phase3RunManifest,
    *,
    started_at: str,
    status: RunStatus,
    failure_reason: str | None,
) -> Phase3RunManifest:
    payload = initial.to_dict()
    payload.update(
        {
            "status": status.value,
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "inventory_path": "artifact_inventory.json",
            "failure_reason": None if status is RunStatus.COMPLETED else failure_reason,
        }
    )
    return Phase3RunManifest.from_dict(payload)


def _run_point(
    *,
    bundle: ExperimentBundle,
    plan_path: str,
    point: Any,
    run_id: str,
    git_sha: str,
    backend: BF16BackendIdentity,
    live_hardware: Mapping[str, Any],
) -> dict[str, Any]:
    store = phase3_artifact_store(REPOSITORY_ROOT)
    created_at = _utc_now()
    with tempfile.TemporaryDirectory(prefix=f"kvbench-{run_id}-", dir="/tmp") as raw_temp:
        temp_root = Path(raw_temp).resolve(strict=True)
        handshake_directory = temp_root / "worker-handshake"
        handshake_directory.mkdir(mode=0o700)
        raw_audit_root = temp_root / "raw-audits"
        raw_audit_root.mkdir(mode=0o700)
        execution_source_pin = _pin_phase3_execution_sources(git_sha)
        static_cache_relative = STATIC_CACHE_SOURCE.relative_to(
            REPOSITORY_ROOT
        ).as_posix()
        cache = _cache_identity(
            point,
            implementation_sha256=(
                execution_source_pin.source_sha256_by_path[
                    static_cache_relative
                ]
            ),
        )
        _validate_cache_source_join(cache, execution_source_pin)
        expected_raw_audit_operations = _expected_phase3_raw_audit_operations(
            point=point,
            run_id=run_id,
            git_sha=git_sha,
            cache=cache,
            backend=backend,
            source_sha256_by_path=(
                execution_source_pin.sut_source_sha256_by_path
            ),
        )
        environment = _worker_environment(
            temp_root,
            raw_audit_operations=expected_raw_audit_operations,
        )
        environment_sha256 = sha256_hex(canonical_json_bytes(environment))
        worker_argv = _worker_argv(plan_path, point, run_id)
        expected_command_fingerprint = command_fingerprint(
            worker_argv,
            working_directory=str(REPOSITORY_ROOT),
            environment_sha256=environment_sha256,
        )
        environment[COMMAND_FINGERPRINT_ENV] = expected_command_fingerprint
        initial = _initial_manifest(
            bundle=bundle,
            plan_path=plan_path,
            point=point,
            run_id=run_id,
            created_at=created_at,
            git_sha=git_sha,
            environment_sha256=environment_sha256,
            backend=backend,
            cache=cache,
        )
        run = store.create(run_id, initial)
        run.start()
        started_at = _utc_now()
        run.write_json("config/plan.json", bundle.plan.to_dict())
        run.write_json(
            "config/referenced_fingerprints.json",
            {
                "schema_version": "kvbench-phase3-references-1.0.0",
                "fingerprints": [
                    {"path": path, "sha256": digest}
                    for path, digest in bundle.canonical_fingerprints
                ],
                "formal_blockers_retained": list(bundle.all_blockers),
            },
        )
        run.write_json(
            "validation/point.json",
            {
                **point.to_dict(),
                "point_fingerprint": derive_phase3_point_fingerprint(
                    point.point_id
                ),
            },
        )
        run.write_json("environment/worker_environment.json", environment)
        run.write_json("environment/live_hardware.json", live_hardware)
        process_snapshots: dict[str, Any] = {
            "ready": dict(READY_NOT_OBSERVED_V2),
        }
        result: Phase3WorkerResult | None = None
        evidence: dict[str, Any] | None = None
        process: subprocess.Popen[bytes] | None = None
        spawned_process_identity: ProcessIdentity | None = None
        registry: RunOwnedProcessRegistry | None = None
        pidfd_supported = hasattr(os, "pidfd_open")
        opened_pidfd: int | None = None
        pidfd_was_opened = False
        raw_audit_root_fd: int | None = None
        raw_audit_owner_uid: int | None = None
        ipc_parent_fd: int | None = None
        ipc_parent_owner_uid: int | None = None
        raw_audit_outcome = _raw_audit_ingestion_outcome()
        ownership_outcome: dict[str, Any] | None = None
        handshake_evidence: dict[str, Any] | None = None
        stdout_path = temp_root / "worker.stdout"
        stderr_path = temp_root / "worker.stderr"
        failure_reason: str | None = None
        process_audit_passed = False
        worker_evidence_valid = False
        execution_sources_revalidated = False
        worker_termination: dict[str, Any] = {
            "schema_version": PHASE3_WORKER_TERMINATION_SCHEMA_VERSION,
            "run_id": run_id,
            "required": False,
            "registered": None,
            "resolved": True,
            "disposition": "not_required",
            "returncode": None,
            "failure_reason": None,
            "source_revalidation_attempted_after_resolution": False,
            "source_revalidated_after_resolution": False,
            "pidfd_closed_after_resolution": None,
            "pidfd_closed_after_source_revalidation_attempt": None,
        }
        try:
            raw_audit_outcome.update(
                {
                    "execution_source_pinned_before_spawn": True,
                    "expected_operation_count_precomputed_before_spawn": len(
                        expected_raw_audit_operations
                    ),
                }
            )
            run.write_json(
                "validation/execution_source_pin.before_spawn.json",
                execution_source_pin.to_dict(verification_stage="before_spawn"),
            )
            before = _process_snapshot()
            process_snapshots["before"] = before
            if not _snapshot_clean(before, allow_supervised=False):
                raise Phase3CoordinatorError("foreign or unknown compute before worker")
            ipc_parent_fd, ipc_parent_owner_uid = _open_private_ipc_parent(
                temp_root
            )
            raw_audit_root_fd, raw_audit_owner_uid = (
                _open_private_raw_audit_root(raw_audit_root)
            )
            raw_audit_outcome["source_root_pinned_before_spawn"] = True
            with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
                process = subprocess.Popen(
                    worker_argv,
                    cwd=REPOSITORY_ROOT,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    start_new_session=True,
                )
                worker_termination.update(
                    {
                        "required": True,
                        "registered": False,
                        "resolved": False,
                        "disposition": "spawned_registration_pending",
                    }
                )
                pidfd_supported, opened_pidfd = _pidfd_open(process.pid)
                pidfd_was_opened = opened_pidfd is not None
                try:
                    spawned_process_identity = read_process_identity(process.pid)
                    registry = RunOwnedProcessRegistry.register_spawn(
                        process_identity=spawned_process_identity,
                        expected_supervisor_pid=os.getpid(),
                        process_handle=process,
                        pidfd_supported=pidfd_supported,
                        pidfd=opened_pidfd,
                        run_id=run_id,
                        gpu_uuid=PHASE3_GPU_UUID,
                        spawned_at_utc=_utc_now(),
                        expected_command_fingerprint=expected_command_fingerprint,
                    )
                    worker_termination.update(
                        {
                            "registered": True,
                            "disposition": "registered_running",
                        }
                    )
                except ProcessSupervisionError as error:
                    raise Phase3CoordinatorError(
                        "worker process registration failed"
                    ) from error
                ready = _wait_for_ready(
                    registry,
                    Path(environment["KVBENCH_PHASE3_AUDIT_READY"]),
                    handshake_directory,
                )
                process_snapshots["ready"] = ready
                registry.refresh_handshake_directory(handshake_directory)
                release_audit = _process_snapshot()
                registry.refresh_handshake_directory(handshake_directory)
                process_snapshots["release_audit"] = release_audit
                release_verdict = _registry_snapshot_verdict(
                    release_audit,
                    registry,
                    terminal_resolution_allowed=False,
                )
                process_snapshots["release_registry_verdict"] = release_verdict
                if release_verdict["passed"] is not True:
                    raise Phase3CoordinatorError("worker release audit failed closed")
                _exclusive_release(Path(environment["KVBENCH_PHASE3_AUDIT_RELEASE"]))
                during_samples: list[dict[str, Any]] = []
                during_verdicts: list[dict[str, Any]] = []
                saw_registered_compute = False
                worker_deadline = time.monotonic() + WORKER_TIMEOUT_SECONDS
                while True:
                    try:
                        registry.refresh_handshake_directory(handshake_directory)
                    except ProcessSupervisionError as error:
                        raise Phase3CoordinatorError(
                            "worker handshake refresh failed"
                        ) from error
                    if _nonreaping_exit_observed(registry):
                        registry.refresh_handshake_directory(handshake_directory)
                        break
                    if time.monotonic() >= worker_deadline:
                        raise Phase3CoordinatorError("worker execution timed out")
                    candidate = _process_snapshot()
                    registry.refresh_handshake_directory(handshake_directory)
                    exited_during_snapshot = _nonreaping_exit_observed(registry)
                    registry.refresh_handshake_directory(handshake_directory)
                    terminal_resolution_allowed = bool(
                        exited_during_snapshot
                        or HandshakeStage.EVIDENCE_FLUSHED
                        in registry.observed_worker_stages
                    )
                    candidate_verdict = _registry_snapshot_verdict(
                        candidate,
                        registry,
                        terminal_resolution_allowed=terminal_resolution_allowed,
                    )
                    during_samples.append(candidate)
                    during_verdicts.append(candidate_verdict)
                    registry_verdict = candidate_verdict.get("registry_verdict")
                    if isinstance(registry_verdict, Mapping) and registry_verdict.get(
                        "owned"
                    ):
                        saw_registered_compute = True
                    if candidate_verdict["passed"] is not True:
                        process_snapshots["during"] = {
                            "schema_version": "kvbench-phase3-process-monitor-2.0.0",
                            "sampling_target_seconds": 2.0,
                            "samples": during_samples,
                            "sample_registry_verdicts": during_verdicts,
                        }
                        raise Phase3CoordinatorError(
                            "worker process audit detected foreign or unverified compute"
                        )
                    if exited_during_snapshot:
                        break
                    time.sleep(2.0)
                process_snapshots["during"] = {
                    "schema_version": "kvbench-phase3-process-monitor-2.0.0",
                    "sampling_target_seconds": 2.0,
                    "samples": during_samples,
                    "sample_registry_verdicts": during_verdicts,
                    "saw_registered_compute": saw_registered_compute,
                    "fast_exit_before_first_telemetry_poll": not during_samples,
                    "monitoring_stopped_before_worker_exit": False,
                }
                registry.refresh_handshake_directory(handshake_directory)
                returncode, reaped_event = _reap_registered_worker(
                    process,
                    registry,
                    timeout_seconds=WORKER_TIMEOUT_SECONDS,
                )
                worker_termination.update(
                    {
                        "resolved": True,
                        "disposition": "registered_reaped",
                        "returncode": returncode,
                        "failure_reason": None,
                    }
                )
                write_handshake_event(handshake_directory, reaped_event)
                worker_termination[
                    "source_revalidation_attempted_after_resolution"
                ] = True
                _revalidate_phase3_execution_sources(execution_source_pin)
                _validate_cache_source_join(cache, execution_source_pin)
                execution_sources_revalidated = True
                worker_termination[
                    "source_revalidated_after_resolution"
                ] = True
                raw_audit_outcome[
                    "execution_source_revalidated_after_worker_exit"
                ] = True
                run.write_json(
                    "validation/execution_source_pin.after_worker_exit.json",
                    execution_source_pin.to_dict(
                        verification_stage="after_worker_exit"
                    ),
                )
                ownership = registry.terminal_outcome()
                ownership_outcome = ownership.to_dict()
                if ownership.disposition is not OwnershipDisposition.OWNED_COMPLETED:
                    raise Phase3CoordinatorError(
                        f"worker ownership failed: {ownership.disposition.value}"
                    )
            registry.refresh_handshake_directory(handshake_directory)
            after = _process_snapshot()
            registry.refresh_handshake_directory(handshake_directory)
            process_snapshots["after"] = after
            after_verdict = _registry_snapshot_verdict(
                after,
                registry,
                terminal_resolution_allowed=True,
            )
            process_snapshots["after_registry_verdict"] = after_verdict
            if after_verdict["passed"] is not True:
                raise Phase3CoordinatorError(
                    "post-reap process audit detected foreign or unverified compute"
                )
            process_audit_passed = True
            raw_audit_outcome["process_audit_passed"] = True
            stdout = stdout_path.read_bytes()
            parsed_result = _parse_canonical_json(
                stdout,
                maximum_bytes=MAX_STDOUT_BYTES,
                label="worker stdout",
            )
            result = Phase3WorkerResult.from_dict(parsed_result)
            if (
                result.run_id != run_id
                or result.point_id != point.point_id
                or result.runner_kind is not point.runner_kind
                or result.count_unit is not bundle.plan.measurement.count_unit
            ):
                raise Phase3CoordinatorError("worker result differs from requested point")
            if process.returncode != 0:
                raise Phase3CoordinatorError("worker returned nonzero with a result")
            if ipc_parent_fd is None or ipc_parent_owner_uid is None:
                raise Phase3CoordinatorError(
                    "worker IPC parent was not pinned before spawn"
                )
            ipc_bytes = _read_pinned_ipc_file(
                ipc_parent_fd,
                "worker-evidence.json",
                expected_owner_uid=ipc_parent_owner_uid,
                maximum_bytes=MAX_IPC_BYTES,
                label="primary worker evidence",
            )
            evidence = _parse_canonical_json(
                ipc_bytes,
                maximum_bytes=MAX_IPC_BYTES,
                label="primary worker evidence",
            )
            evidence_schema_version = evidence.get("schema_version")
            raw_audit_outcome["worker_evidence_schema_version"] = (
                evidence_schema_version
            )
            if evidence_schema_version != WORKER_EVIDENCE_V1:
                raise Phase3CoordinatorError(
                    "primary worker evidence must retain the complete v1 schema"
                )
            raw_audit_outcome.update(
                {
                    "required": True,
                    "attempted": True,
                    "status": "sidecar_expected",
                }
            )
            raw_index_ipc_bytes = _read_pinned_ipc_file(
                ipc_parent_fd,
                "raw-audit-index.json",
                expected_owner_uid=ipc_parent_owner_uid,
                maximum_bytes=MAX_IPC_BYTES,
                label="raw-audit index sidecar",
            )
            raw_audit_outcome["sidecar_observed"] = True
            raw_index_evidence = _parse_canonical_json(
                raw_index_ipc_bytes,
                maximum_bytes=MAX_IPC_BYTES,
                label="raw-audit index sidecar",
            )
            raw_index_schema_version = raw_index_evidence.get("schema_version")
            raw_audit_outcome["raw_index_schema_version"] = (
                raw_index_schema_version
            )
            if raw_index_schema_version != WORKER_EVIDENCE_V2:
                raise Phase3CoordinatorError(
                    "raw-audit index sidecar schema version is unsupported"
                )
            assert registry is not None
            registry_evidence = registry.to_evidence()
            events = registry_evidence.get("handshake_events")
            if not isinstance(events, list):
                raise Phase3CoordinatorError(
                    "registered worker handshake evidence is malformed"
                )
            evidence_events = [
                event
                for event in events
                if isinstance(event, Mapping)
                and event.get("stage") == HandshakeStage.EVIDENCE_FLUSHED.value
            ]
            raw_audit_outcome["commitment_validation_attempted"] = True
            commitment_sha256 = phase3_worker_channel_commitment_sha256(
                run_id=run_id,
                point_id=point.point_id,
                primary_evidence_bytes=ipc_bytes,
                raw_audit_index_bytes=raw_index_ipc_bytes,
            )
            if (
                len(evidence_events) != 1
                or evidence_events[0].get("evidence_sha256")
                != commitment_sha256
            ):
                raise Phase3CoordinatorError(
                    "evidence_flushed digest differs from two-channel commitment"
                )
            raw_audit_outcome["ipc_digest_validated"] = True
            raw_audit_outcome["channel_commitment_validated"] = True
            raw_audit_outcome["commitment_validation_passed"] = True
            _, preserved_commitment_sha256 = (
                _preserve_phase3_worker_channel_artifacts(
                    run=run,
                    run_id=run_id,
                    point_id=point.point_id,
                    primary_evidence_bytes=ipc_bytes,
                    raw_audit_index_bytes=raw_index_ipc_bytes,
                )
            )
            if preserved_commitment_sha256 != commitment_sha256:
                raise Phase3CoordinatorError(
                    "preserved worker channel commitment digest differs"
                )
            raw_audit_outcome.update(
                {
                    "channel_artifacts_preserved": True,
                    "primary_channel_artifact": TRANSPORT_PRIMARY_CHANNEL_ARTIFACT,
                    "sidecar_channel_artifact": TRANSPORT_SIDECAR_CHANNEL_ARTIFACT,
                    "commitment_payload_artifact": TRANSPORT_COMMITMENT_ARTIFACT,
                    "commitment_digest_artifact": TRANSPORT_COMMITMENT_DIGEST_ARTIFACT,
                    "commitment_sha256": commitment_sha256,
                }
            )
            _validate_worker_evidence_v1(
                evidence=evidence,
                expected_run_id=run_id,
                expected_point_id=point.point_id,
                result=result,
                cache_layout_fingerprint=cache.layout_fingerprint,
            )
            raw_audit_outcome["status"] = "validating_transport"
            if raw_audit_root_fd is None or raw_audit_owner_uid is None:
                raise Phase3CoordinatorError(
                    "raw-audit root was not pinned before worker spawn"
                )
            if expected_raw_audit_operations is None:
                raise Phase3CoordinatorError(
                    "raw-audit operations were not pinned before worker spawn"
                )
            raw_audit_outcome["collection_validation_attempted"] = True
            _ingest_worker_evidence_v2(
                evidence=raw_index_evidence,
                expected_run_id=run_id,
                expected_point_id=point.point_id,
                raw_audit_root_fd=raw_audit_root_fd,
                raw_audit_owner_uid=raw_audit_owner_uid,
                expected_operations=expected_raw_audit_operations,
                run=run,
                outcome=raw_audit_outcome,
                execution_source_pin=execution_source_pin,
                backend_identity=backend,
            )
            worker_evidence_valid = True
        except BaseException as error:
            failure_reason = f"{type(error).__name__}: {' '.join(str(error).split())}"[:1000]
            if raw_audit_outcome["ingestion_passed"] is not True:
                raw_audit_outcome.update(
                    {
                        "passed": False,
                        "status": (
                            "failed"
                            if raw_audit_outcome["attempted"]
                            else "not_attempted"
                        ),
                        "failure_reason": failure_reason,
                    }
                )
            if process is not None and process.returncode is None:
                if registry is not None:
                    try:
                        registry.refresh_handshake_directory(handshake_directory)
                    except ProcessSupervisionError:
                        pass
                    try:
                        _, reaped_event = _terminate_registered_worker(
                            process,
                            registry,
                            handshake_directory=handshake_directory,
                        )
                    except BaseException as termination_error:
                        worker_termination.update(
                            {
                                "registered": True,
                                "resolved": False,
                                "disposition": (
                                    "registered_termination_unresolved"
                                ),
                                "returncode": process.returncode,
                                "failure_reason": (
                                    f"{type(termination_error).__name__}: "
                                    f"{' '.join(str(termination_error).split())}"
                                )[:1000],
                            }
                        )
                        failure_reason = (
                            f"{type(termination_error).__name__}: "
                            f"{' '.join(str(termination_error).split())}"
                        )[:1000]
                        raw_audit_outcome.update(
                            {
                                "passed": False,
                                "status": "failed",
                                "failure_reason": failure_reason,
                            }
                        )
                    else:
                        worker_termination.update(
                            {
                                "registered": True,
                                "resolved": True,
                                "disposition": "registered_terminated_reaped",
                                "returncode": process.returncode,
                                "failure_reason": None,
                            }
                        )
                        try:
                            registry.refresh_handshake_directory(
                                handshake_directory
                            )
                        except ProcessSupervisionError:
                            pass
                        try:
                            write_handshake_event(
                                handshake_directory, reaped_event
                            )
                        except ProcessSupervisionError:
                            pass
                else:
                    try:
                        unregistered_returncode = _terminate_unregistered_worker(
                            process,
                            pidfd=opened_pidfd,
                            expected_identity=spawned_process_identity,
                        )
                    except BaseException as termination_error:
                        worker_termination.update(
                            {
                                "registered": False,
                                "resolved": False,
                                "disposition": (
                                    "unregistered_termination_unresolved"
                                ),
                                "returncode": process.returncode,
                                "failure_reason": (
                                    f"{type(termination_error).__name__}: "
                                    f"{' '.join(str(termination_error).split())}"
                                )[:1000],
                            }
                        )
                        failure_reason = (
                            f"{type(termination_error).__name__}: "
                            f"{' '.join(str(termination_error).split())}"
                        )[:1000]
                        raw_audit_outcome.update(
                            {
                                "passed": False,
                                "status": "failed",
                                "failure_reason": failure_reason,
                            }
                        )
                    else:
                        worker_termination.update(
                            {
                                "registered": False,
                                "resolved": True,
                                "disposition": "unregistered_terminated_reaped",
                                "returncode": unregistered_returncode,
                                "failure_reason": None,
                            }
                        )
            elif (
                process is not None
                and registry is not None
                and process.returncode is not None
                and not registry.reaped
            ):
                failure_reason = (
                    "Phase3CoordinatorError: registered worker process handle "
                    "was reaped before non-reaping exit ownership was recorded"
                )
                worker_termination.update(
                    {
                        "registered": True,
                        "resolved": False,
                        "disposition": "registered_reap_unverified",
                        "returncode": process.returncode,
                        "failure_reason": failure_reason,
                    }
                )
            worker_exit_confirmed = bool(
                process is not None
                and worker_termination["resolved"] is True
                and (
                    process.returncode is not None
                    or registry is not None
                    and registry.reaped
                )
            )
            if (
                worker_exit_confirmed
                and execution_source_pin is not None
                and not execution_sources_revalidated
            ):
                try:
                    worker_termination[
                        "source_revalidation_attempted_after_resolution"
                    ] = True
                    _revalidate_phase3_execution_sources(execution_source_pin)
                    _validate_cache_source_join(cache, execution_source_pin)
                    execution_sources_revalidated = True
                    worker_termination[
                        "source_revalidated_after_resolution"
                    ] = True
                    raw_audit_outcome[
                        "execution_source_revalidated_after_worker_exit"
                    ] = True
                    run.write_json(
                        "validation/execution_source_pin.after_worker_exit.json",
                        execution_source_pin.to_dict(
                            verification_stage="after_worker_exit"
                        ),
                    )
                except BaseException as source_error:
                    failure_reason = (
                        f"{type(source_error).__name__}: "
                        f"{' '.join(str(source_error).split())}"
                    )[:1000]
                    raw_audit_outcome.update(
                        {
                            "passed": False,
                            "status": "failed",
                            "failure_reason": failure_reason,
                        }
                    )
            if registry is not None and registry.reaped:
                ownership_outcome = registry.terminal_outcome().to_dict()
            try:
                if "after" not in process_snapshots:
                    process_snapshots["after"] = _process_snapshot()
            except BaseException as after_error:
                process_snapshots["after_error"] = {
                    "type": type(after_error).__name__,
                    "message": "post-worker process snapshot failed",
                }
            result = _failed_result(
                bundle=bundle,
                point=point,
                run_id=run_id,
                reason=failure_reason,
            )
        finally:
            if opened_pidfd is not None:
                worker_termination["pidfd_closed_after_resolution"] = bool(
                    worker_termination["resolved"] is True
                )
                worker_termination[
                    "pidfd_closed_after_source_revalidation_attempt"
                ] = bool(
                    worker_termination[
                        "source_revalidation_attempted_after_resolution"
                    ]
                )
                os.close(opened_pidfd)
                opened_pidfd = None
            if raw_audit_root_fd is not None:
                os.close(raw_audit_root_fd)
                raw_audit_root_fd = None
            if ipc_parent_fd is not None:
                os.close(ipc_parent_fd)
                ipc_parent_fd = None
        stdout = stdout_path.read_bytes() if stdout_path.is_file() else b""
        stderr = stderr_path.read_bytes() if stderr_path.is_file() else b""
        run.write_bytes("logs/worker.stdout.txt", stdout)
        run.write_bytes("logs/worker.stderr.txt", stderr)
        for name, snapshot in process_snapshots.items():
            run.write_json(f"environment/process.{name}.json", snapshot)
        run.write_json(
            "validation/worker_termination.json",
            worker_termination,
        )
        if registry is not None:
            registry_evidence = registry.to_evidence()
            registry_evidence["pidfd_closed_by_supervisor"] = pidfd_was_opened
            registry_evidence["process_handle_reaped_by_supervisor"] = registry.reaped
            handshake_evidence = {
                "schema_version": PHASE3_WORKER_HANDSHAKE_EVIDENCE_SCHEMA_VERSION,
                "run_id": run_id,
                "events": registry_evidence["handshake_events"],
                "terminal_outcome": ownership_outcome,
                "evidence_flushed_required_for_owned_completion": True,
                "zero_returncode_required_for_owned_completion": True,
                "worker_exiting_required_for_owned_completion": False,
                "rapid_zero_exit_after_evidence_flushed_owned_completion_allowed": True,
            }
        else:
            registry_evidence = {
                "schema_version": RunOwnedProcessRegistry.SCHEMA_VERSION,
                "registry_created": False,
                "run_id": run_id,
                "pidfd_supported": pidfd_supported,
                "pidfd_closed_by_supervisor": pidfd_was_opened,
                "owned_completion_policy": (
                    RunOwnedProcessRegistry.OWNED_COMPLETION_POLICY
                ),
                "evidence_flushed_required_for_owned_completion": True,
                "zero_returncode_required_for_owned_completion": True,
                "worker_exiting_required_for_owned_completion": False,
                "rapid_zero_exit_after_evidence_flushed_owned_completion_allowed": True,
            }
            handshake_evidence = {
                "schema_version": PHASE3_WORKER_HANDSHAKE_EVIDENCE_SCHEMA_VERSION,
                "run_id": run_id,
                "events": [],
                "terminal_outcome": None,
                "evidence_flushed_required_for_owned_completion": True,
                "zero_returncode_required_for_owned_completion": True,
                "worker_exiting_required_for_owned_completion": False,
                "rapid_zero_exit_after_evidence_flushed_owned_completion_allowed": True,
            }
        run.write_json("environment/process.registry.json", registry_evidence)
        run.write_json("environment/process.handshake.json", handshake_evidence)
        ownership_disposition = (
            None
            if ownership_outcome is None
            else ownership_outcome.get("disposition")
        )
        exclusivity_passed = bool(
            ownership_outcome is not None
            and ownership_outcome.get("exclusivity_passed") is True
        )
        run.write_json(
            "validation/process_audit_outcome.json",
            {
                "schema_version": PHASE3_PROCESS_AUDIT_SCHEMA_VERSION,
                "passed": process_audit_passed,
                "certified_helper": "preflight/process_query.py",
                "registry_created": registry is not None,
                "ownership_verdict": ownership_disposition,
                "owned_completion_basis": (
                    None
                    if ownership_outcome is None
                    else ownership_outcome.get("owned_completion_basis")
                ),
                "owned_completion_policy": RunOwnedProcessRegistry.OWNED_COMPLETION_POLICY,
                "worker_exiting_required_for_owned_completion": False,
                "rapid_zero_exit_after_evidence_flushed_owned_completion_allowed": True,
                "exclusivity_passed": exclusivity_passed,
                "evidence_flushed": bool(
                    ownership_outcome is not None
                    and ownership_outcome.get("evidence_flushed") is True
                ),
                "worker_exiting_observed": bool(
                    ownership_outcome is not None
                    and ownership_outcome.get("worker_exiting_observed") is True
                ),
                "pid_start_time_protected": registry is not None,
                "pidfd_supported": pidfd_supported,
                "pidfd_opened": pidfd_was_opened,
                "pidfd_closed": pidfd_was_opened,
                "worker_termination_resolved": worker_termination["resolved"],
                "worker_termination_disposition": worker_termination["disposition"],
                "failure_reason": (
                    None if process_audit_passed else failure_reason
                ),
                "foreign_compute_allowed": False,
                "unknown_compute_allowed": False,
            },
        )
        run.write_json(
            "validation/worker_evidence_outcome.json",
            {
                "schema_version": (
                    "kvbench-phase3-worker-evidence-outcome-1.0.0"
                ),
                "passed": worker_evidence_valid,
                "process_audit_passed": process_audit_passed,
                "failure_reason": (
                    None if worker_evidence_valid else failure_reason
                ),
            },
        )
        run.write_json(
            "validation/raw_audit_ingestion_outcome.json",
            raw_audit_outcome,
        )
        final_status, final_reason = _resolve_phase3_terminal_status(
            result=result,
            process_audit_passed=process_audit_passed,
            worker_evidence_valid=worker_evidence_valid,
            raw_audit_outcome=raw_audit_outcome,
            failure_reason=failure_reason,
        )
        source_result_bytes = canonical_json_bytes(result.to_dict())
        resolved_result = _resolved_phase3_worker_result(
            result,
            final_status=final_status,
            final_reason=final_reason,
        )
        resolved_result_bytes = canonical_json_bytes(resolved_result.to_dict())
        run.write_json(
            "validation/worker_terminal_resolution.json",
            {
                "schema_version": (
                    "kvbench-phase3-worker-terminal-resolution-1.0.0"
                ),
                "source_worker_status": result.status.value,
                "resolved_terminal_status": resolved_result.status.value,
                "status_overridden": resolved_result.status is not result.status,
                "resolution_reason": final_reason,
                "source_worker_result_sha256": sha256_hex(source_result_bytes),
                "resolved_worker_result_sha256": sha256_hex(
                    resolved_result_bytes
                ),
                "source_primary_channel_preserved": evidence is not None,
            },
        )
        run.write_json(
            "validation/worker_result.json", resolved_result.to_dict()
        )
        if evidence is not None:
            _write_runtime_artifacts(run, evidence)
            model_identity = evidence.get("model_identity")
            if isinstance(model_identity, Mapping):
                run.write_json("validation/model_identity.json", model_identity)
        final = _terminal_manifest(
            initial,
            started_at=started_at,
            status=final_status,
            failure_reason=final_reason,
        )
        final_path = run.finalize(final)
    validation = validate_run_directory(final_path)
    if not validation.valid or not validation.complete:
        raise Phase3CoordinatorError("final Phase 3 run failed checksum validation")
    return {
        "run_id": run_id,
        "point_id": point.point_id,
        "status": final_status.value,
        "run_dir": str(final_path.relative_to(REPOSITORY_ROOT)),
        "checksum_valid": True,
        "worker_termination_resolved": worker_termination["resolved"],
        "timing_collected": bool(
            evidence is not None
            and isinstance(evidence.get("runtime"), Mapping)
            and evidence["runtime"].get("timing") is not None
        ),
    }


def run_phase3_campaign(plan_path: str | Path) -> dict[str, Any]:
    """Run every frozen point once, preserving failures and never retrying."""

    _validate_entry_evidence()
    git_sha, _ = _git_identity()
    live_hardware = _live_hardware()
    bundle = load_phase3_admission_bundle(plan_path)
    if not bundle.execution_ready:
        raise Phase3CoordinatorError("Phase 3 bundle is not narrowly execution-authorized")
    relative_plan = bundle.plan_path.relative_to(REPOSITORY_ROOT).as_posix()
    points = expand_phase3_process_points(bundle.plan)
    backend = _backend_identity_stdlib()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S%fZ").lower()
    campaign_id = f"phase3-{timestamp}-{git_sha[:8]}-{secrets.token_hex(3)}"
    planned_run_ids = tuple(
        f"{campaign_id}-{point.point_id}" for point in points
    )
    recorder = Phase3CampaignRecorder.create(
        repository_root=REPOSITORY_ROOT,
        campaign_id=campaign_id,
        created_at_utc=_utc_now(),
        git_sha=git_sha,
        plan_path=relative_plan,
        plan_fingerprint=bundle.plan.fingerprint(),
        point_ids=tuple(point.point_id for point in points),
        run_ids=planned_run_ids,
    )
    runs: list[dict[str, Any]] = []
    unexpected_error: BaseException | None = None
    try:
        for point, run_id in zip(points, planned_run_ids):
            run_result = _run_point(
                bundle=bundle,
                plan_path=relative_plan,
                point=point,
                run_id=run_id,
                git_sha=git_sha,
                backend=backend,
                live_hardware=live_hardware,
            )
            runs.append(run_result)
            if run_result.get("worker_termination_resolved") is False:
                raise Phase3WorkerTerminationUnresolved(
                    "campaign aborted after preserving run with unresolved "
                    f"worker termination: {run_id}"
                )
    except BaseException as error:
        unexpected_error = error
    counts = Counter(item["status"] for item in runs)
    result = {
        "schema_version": "kvbench-phase3-campaign-result-1.0.0",
        "ok": unexpected_error is None
        and len(runs) == len(points)
        and all(item["status"] == RunStatus.COMPLETED.value for item in runs),
        "campaign_id": campaign_id,
        "git_sha": git_sha,
        "plan": relative_plan,
        "plan_fingerprint": bundle.plan.fingerprint(),
        "expected_process_count": bundle.plan.expected_process_count,
        "attempted_process_count": len(runs),
        "unattempted_point_ids": [
            point.point_id for point in points[len(runs) :]
        ],
        "status_counts": dict(sorted(counts.items())),
        "runs": runs,
        "execution_attempted": True,
        "timing_collected": any(item["timing_collected"] for item in runs),
        "profiler_executed": False,
        "quality_executed": False,
        "performance_claim_eligible": False,
        "measurement_scope": "native_host_admission",
        "selective_rerun_performed": False,
        "preregistered_before_execution": True,
        "unexpected_campaign_abort": unexpected_error is not None,
        "unexpected_failure": (
            None
            if unexpected_error is None
            else f"{type(unexpected_error).__name__}: "
            f"{' '.join(str(unexpected_error).split())}"[:1000]
        ),
        "finished_at_utc": _utc_now(),
    }
    campaign_path = recorder.finalize(result)
    result["campaign_record"] = campaign_path.relative_to(
        REPOSITORY_ROOT
    ).as_posix()
    if unexpected_error is not None:
        raise unexpected_error
    return result
