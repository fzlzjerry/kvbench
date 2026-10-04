"""Strict run, sample, exclusion, and artifact schemas."""

from __future__ import annotations

import dataclasses
from datetime import datetime
from enum import StrEnum
import json
import math
from typing import ClassVar

from kvbench.schema.base import (
    ClaimClass,
    ClaimEligibility,
    GraphMode,
    QualityStatus,
    QualityValidationState,
    RunKind,
    RunnerKind,
    RunStatus,
    StrictModel,
    canonical_json_bytes,
    require_git_sha,
    require_identifier,
    require_oci_digest,
    require_relative_path,
    require_run_id,
    require_schema,
    require_sha256,
    require_utc_timestamp,
    sha256_hex,
)
from kvbench.schema.config import MethodConfigFingerprint, MethodName


class ConfigSourceKind(StrEnum):
    PATH = "path"
    INLINE = "inline"


class ExclusionReason(StrEnum):
    CAPACITY_INFEASIBLE = "capacity_infeasible"
    OOM = "oom"
    UNSTABLE = "unstable"
    BACKEND_FALLBACK = "backend_fallback"
    UNSUPPORTED_GEOMETRY = "unsupported_geometry"
    PRECONDITION_FAILED = "precondition_failed"
    PHASE_NOT_IMPLEMENTED = "phase_not_implemented"
    ABORTED = "aborted"


@dataclasses.dataclass(frozen=True, slots=True)
class ConfigSource(StrictModel):
    kind: ConfigSourceKind
    path: str | None
    canonical_inline_json: str | None
    sha256: str

    def __post_init__(self) -> None:
        require_sha256(self.sha256)
        if self.kind is ConfigSourceKind.PATH:
            if self.path is None or self.canonical_inline_json is not None:
                raise ValueError("path config source requires path and null inline config")
            require_relative_path(self.path, field_name="config source path")
        else:
            if self.path is not None or self.canonical_inline_json is None:
                raise ValueError("inline config source requires canonical JSON and null path")
            def reject_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
                result: dict[str, object] = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError("inline config contains a duplicate key")
                    result[key] = value
                return result

            def reject_constant(_: str) -> None:
                raise ValueError("inline config contains a non-finite number")

            try:
                parsed = json.loads(
                    self.canonical_inline_json,
                    object_pairs_hook=reject_pairs,
                    parse_constant=reject_constant,
                )
            except json.JSONDecodeError as error:
                raise ValueError("canonical_inline_json is invalid") from error
            if canonical_json_bytes(parsed).decode("utf-8") != self.canonical_inline_json:
                raise ValueError("inline config is not canonical JSON")
            if sha256_hex(self.canonical_inline_json.encode("utf-8")) != self.sha256:
                raise ValueError("inline config SHA-256 does not match canonical bytes")


@dataclasses.dataclass(frozen=True, slots=True)
class CommandSpec(StrictModel):
    schema_version: str
    argv: tuple[str, ...]
    dry_run: bool

    SCHEMA_VERSION: ClassVar[str] = "kvbench-command-1.0.0"

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        if not self.argv or any(not item or "\x00" in item for item in self.argv):
            raise ValueError("command argv must contain non-empty safe strings")
        if not self.dry_run:
            raise ValueError("Phase 2 command specifications must be dry runs")


@dataclasses.dataclass(frozen=True, slots=True)
class ArtifactEntry(StrictModel):
    path: str
    role: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        require_relative_path(self.path, field_name="artifact path")
        if not self.role.strip():
            raise ValueError("artifact role must be non-empty")
        if self.size_bytes < 0:
            raise ValueError("artifact size must be nonnegative")
        require_sha256(self.sha256)


@dataclasses.dataclass(frozen=True, slots=True)
class ArtifactInventory(StrictModel):
    schema_version: str
    run_id: str
    files: tuple[ArtifactEntry, ...]
    excluded_control_files: tuple[str, ...]

    SCHEMA_VERSION: ClassVar[str] = "kvbench-artifact-inventory-1.0.0"
    CONTROL_FILES: ClassVar[tuple[str, ...]] = (
        "artifact_inventory.json",
        "checksums.sha256",
        "COMPLETE",
    )

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        require_run_id(self.run_id)
        paths = [entry.path for entry in self.files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("artifact entries must be unique and lexically sorted")
        if any(path in self.CONTROL_FILES for path in paths):
            raise ValueError("inventory entries cannot include excluded control files")
        if self.excluded_control_files != self.CONTROL_FILES:
            raise ValueError("excluded control files do not match the artifact contract")

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def total_bytes(self) -> int:
        return sum(entry.size_bytes for entry in self.files)


@dataclasses.dataclass(frozen=True, slots=True)
class LifecycleRecord(StrictModel):
    schema_version: str
    run_id: str
    sequence: int
    state: str

    SCHEMA_VERSION: ClassVar[str] = "kvbench-lifecycle-1.0.0"

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        require_run_id(self.run_id)
        fixed = {1: "created", 2: "running", 3: "finalizing"}
        if self.sequence in fixed:
            if self.state != fixed[self.sequence]:
                raise ValueError("lifecycle sequence and state are incompatible")
            return
        if self.sequence != 4:
            raise ValueError("lifecycle sequence must be between 1 and 4")
        try:
            status = RunStatus(self.state)
        except ValueError as error:
            raise ValueError("terminal lifecycle state is invalid") from error
        if not status.is_terminal:
            raise ValueError("fourth lifecycle state must be terminal")


@dataclasses.dataclass(frozen=True, slots=True)
class CompletionMarker(StrictModel):
    schema_version: str
    run_id: str
    status: RunStatus
    manifest_sha256: str
    artifact_inventory_sha256: str
    checksum_ledger_sha256: str
    written_last: bool

    SCHEMA_VERSION: ClassVar[str] = "kvbench-completion-1.0.0"

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        require_run_id(self.run_id)
        if not self.status.is_terminal:
            raise ValueError("completion status must be terminal")
        for digest in (
            self.manifest_sha256,
            self.artifact_inventory_sha256,
            self.checksum_ledger_sha256,
        ):
            require_sha256(digest)
        if not self.written_last:
            raise ValueError("completion marker must declare written_last")


@dataclasses.dataclass(frozen=True, slots=True)
class RunManifest(StrictModel):
    schema_version: str
    artifact_schema_version: str
    run_id: str
    status: RunStatus
    created_at_utc: str
    started_at_utc: str | None
    finished_at_utc: str | None
    run_kind: RunKind
    runner_kind: RunnerKind
    graph_mode: GraphMode
    claim_class: ClaimClass
    plan_source: ConfigSource
    git_sha: str
    git_dirty: bool
    container_digest: str | None
    hardware_id: str
    hardware_fingerprint: str
    software_environment_id: str
    software_fingerprint: str
    model_id: str
    model_revision: str
    model_fingerprint: str
    method: MethodName
    method_config_id: str
    method_config_fingerprint: MethodConfigFingerprint
    contract_fingerprint: str
    attention_backend: str | None
    cache_layout: str | None
    random_seed: int
    process_replicate: int
    quality: QualityStatus
    command: CommandSpec
    inventory_path: str | None
    failure_reason: str | None

    SCHEMA_VERSION: ClassVar[str] = "kvbench-run-manifest-1.0.0"
    ARTIFACT_SCHEMA_VERSION: ClassVar[str] = "kvbench-artifacts-1.0.0"

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        require_schema(self.artifact_schema_version, self.ARTIFACT_SCHEMA_VERSION)
        phase3_only_statuses = {
            RunStatus.MODEL_IDENTITY_UNRESOLVED,
            RunStatus.MODEL_ACCESS_BLOCKED,
            RunStatus.BACKEND_UNSUPPORTED,
            RunStatus.ALLOCATION_FAILED,
            RunStatus.STATE_DRIFT_DETECTED,
            RunStatus.GQA_MATERIALIZATION_DETECTED,
            RunStatus.GRAPH_REPLAY_FAILED,
        }
        if (
            self.run_kind is RunKind.PHASE3_ADMISSION
            or self.status in phase3_only_statuses
        ):
            raise ValueError(
                "Phase 3 run kinds and failure states require the Phase 3 manifest schema"
            )
        require_run_id(self.run_id)
        require_utc_timestamp(self.created_at_utc, field_name="created_at_utc")
        require_git_sha(self.git_sha)
        for value, name in (
            (self.hardware_fingerprint, "hardware_fingerprint"),
            (self.software_fingerprint, "software_fingerprint"),
            (self.model_fingerprint, "model_fingerprint"),
            (self.contract_fingerprint, "contract_fingerprint"),
        ):
            require_sha256(value, field_name=name)
        for value, name in (
            (self.hardware_id, "hardware_id"),
            (self.software_environment_id, "software_environment_id"),
            (self.method_config_id, "method_config_id"),
        ):
            require_identifier(value, field_name=name)
        if not self.model_id or not self.model_revision:
            raise ValueError("model identity must be retained")
        if self.random_seed < 0 or self.process_replicate <= 0:
            raise ValueError("seed must be nonnegative and replicate positive")
        if self.method_config_fingerprint.method is not self.method:
            raise ValueError("method fingerprint does not match manifest method")

        if self.status is RunStatus.CREATED:
            if any(
                value is not None
                for value in (
                    self.started_at_utc,
                    self.finished_at_utc,
                    self.inventory_path,
                    self.failure_reason,
                )
            ):
                raise ValueError("created manifest cannot contain later lifecycle fields")
        elif self.status in {RunStatus.RUNNING, RunStatus.FINALIZING}:
            if self.started_at_utc is None:
                raise ValueError("running/finalizing manifest requires started_at_utc")
            require_utc_timestamp(self.started_at_utc, field_name="started_at_utc")
            if any(
                value is not None
                for value in (
                    self.finished_at_utc,
                    self.inventory_path,
                    self.failure_reason,
                )
            ):
                raise ValueError("nonterminal manifest has terminal-only fields")
        else:
            if self.started_at_utc is None or self.finished_at_utc is None:
                raise ValueError("terminal manifest requires start and finish timestamps")
            require_utc_timestamp(self.started_at_utc, field_name="started_at_utc")
            require_utc_timestamp(self.finished_at_utc, field_name="finished_at_utc")
            if self.inventory_path != "artifact_inventory.json":
                raise ValueError("terminal manifest must reference artifact_inventory.json")
            if self.status is RunStatus.COMPLETED and self.failure_reason is not None:
                raise ValueError("completed manifest cannot carry failure_reason")
            if self.status.is_failure and not self.failure_reason:
                raise ValueError("terminal failure requires a reason")
        timestamps = [
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            for value in (
                self.created_at_utc,
                self.started_at_utc,
                self.finished_at_utc,
            )
            if value is not None
        ]
        if timestamps != sorted(timestamps):
            raise ValueError("manifest lifecycle timestamps are out of order")

        if self.plan_source.kind is not ConfigSourceKind.PATH:
            raise ValueError("saved command requires a path config source")
        source_path = self.plan_source.path
        if source_path is None:
            raise ValueError("path config source is missing its path")
        expected = (
            "kvbench",
            "run",
            "--plan",
            source_path,
            "--dry-run",
        )
        if (
            self.command.argv != expected
            or not self.command.dry_run
        ):
            raise ValueError("command does not match the saved plan source")

        if self.run_kind in {RunKind.TIMING, RunKind.NSYS, RunKind.NCU}:
            if self.git_dirty or self.container_digest is None:
                raise ValueError("formal performance/profiler run requires clean Git and container")
            require_oci_digest(self.container_digest)
            require_git_sha(self.model_revision)
            if not self.attention_backend or not self.cache_layout:
                raise ValueError("formal run requires explicit backend and cache layout")
            if not self.method_config_fingerprint.execution_ready:
                raise ValueError("formal run requires an execution-ready method fingerprint")
            if (
                self.quality.quality_status is not QualityValidationState.UNVALIDATED
                or self.quality.claim_eligibility is not ClaimEligibility.PERFORMANCE_ONLY
            ):
                raise ValueError("formal runs require unvalidated/performance_only quality metadata")
        if self.run_kind is RunKind.SYNTHETIC:
            if self.claim_class is not ClaimClass.NONE:
                raise ValueError("synthetic fixtures cannot carry a claim class")
            if (
                self.quality.quality_status is not QualityValidationState.NOT_APPLICABLE
                or self.quality.claim_eligibility is not ClaimEligibility.NONE
            ):
                raise ValueError("synthetic fixtures require not-applicable quality metadata")
        if self.run_kind in {RunKind.NSYS, RunKind.NCU}:
            if self.claim_class is not ClaimClass.MECHANISM_ONLY:
                raise ValueError("profiler runs are mechanism_only")
        if self.run_kind is RunKind.TIMING and self.claim_class not in {
            ClaimClass.SAME_WORK_LATENCY,
            ClaimClass.CAPACITY_AMPLIFICATION,
            ClaimClass.MECHANISM_ONLY,
        }:
            raise ValueError("timing run requires an explicit performance claim class")
        if self.run_kind in {RunKind.HARDWARE_PREFLIGHT, RunKind.CORRECTNESS}:
            if self.claim_class is not ClaimClass.NONE:
                raise ValueError("preflight/correctness runs cannot carry claims")


@dataclasses.dataclass(frozen=True, slots=True)
class HbmEvidence(StrictModel):
    schema_version: str
    source_run_id: str
    tool: RunKind
    metric_names: tuple[str, ...]
    bf16_hbm_bytes: int
    method_hbm_bytes: int
    directly_measured: bool
    estimated: bool
    evidence_sha256: str

    SCHEMA_VERSION: ClassVar[str] = "kvbench-hbm-evidence-1.0.0"

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        require_run_id(self.source_run_id)
        if self.tool is not RunKind.NCU:
            raise ValueError("HBM evidence must come from an ncu run")
        if (
            not self.metric_names
            or any(not name.strip() for name in self.metric_names)
            or len(set(self.metric_names)) != len(self.metric_names)
        ):
            raise ValueError("HBM evidence metrics must be non-empty and unique")
        if self.bf16_hbm_bytes <= 0 or self.method_hbm_bytes <= 0:
            raise ValueError("direct HBM byte observations must be positive")
        if not self.directly_measured or self.estimated:
            raise ValueError("estimated traffic cannot be HBM evidence")
        require_sha256(self.evidence_sha256)


@dataclasses.dataclass(frozen=True, slots=True)
class SampleRecord(StrictModel):
    schema_version: str
    run_id: str
    timestamp_utc: str
    git_sha: str
    git_dirty: bool
    container_digest: str | None
    gpu_uuid: str
    gpu_full_name: str
    pci_device_id: str
    driver_version: str
    cuda_runtime_version: str
    cuda_toolkit_version: str
    torch_version: str
    triton_version: str
    hardware_fingerprint: str
    software_fingerprint: str
    contract_fingerprint: str
    method: MethodName
    method_config_id: str
    method_config_fingerprint: MethodConfigFingerprint
    model_id: str
    model_revision: str
    model_fingerprint: str
    weight_dtype: str
    batch_size: int
    context_length: int
    decode_step: int
    runner_kind: RunnerKind
    graph_mode: GraphMode
    attention_backend: str
    cache_layout: str
    r_nominal: float | None
    r_alloc: float | None
    r_hbm: float | None
    hbm_evidence: HbmEvidence | None
    logical_bf16_bytes: int | None
    cache_allocated_bytes: int | None
    cache_data_bytes: int | None
    metadata_bytes: int | None
    scale_zero_bytes: int | None
    norm_bytes: int | None
    residual_bytes: int | None
    sink_bytes: int | None
    outlier_value_bytes: int | None
    outlier_index_bytes: int | None
    padding_bytes: int | None
    workspace_bytes: int | None
    peak_memory_bytes: int | None
    wall_time_ms: float | None
    gpu_event_ms: float | None
    kernel_count: int | None
    sm_clock_mhz: int | None
    memory_clock_mhz: int | None
    power_w: float | None
    temperature_c: float | None
    replicate: int
    step_index: int
    random_seed: int
    run_kind: RunKind
    status: RunStatus
    failure_reason: str | None
    claim_class: ClaimClass
    quality: QualityStatus

    SCHEMA_VERSION: ClassVar[str] = "kvbench-sample-1.0.0"

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        if self.run_kind is RunKind.PHASE3_ADMISSION:
            raise ValueError("Phase 3 samples require the Phase 3 evidence schema")
        require_run_id(self.run_id)
        require_utc_timestamp(self.timestamp_utc)
        require_git_sha(self.git_sha)
        if not self.model_id or not self.attention_backend or not self.cache_layout:
            raise ValueError("sample must retain model, backend, and cache-layout identity")
        if not self.weight_dtype or any(
            not value
            for value in (
                self.gpu_uuid,
                self.gpu_full_name,
                self.pci_device_id,
                self.driver_version,
                self.cuda_runtime_version,
                self.cuda_toolkit_version,
                self.torch_version,
                self.triton_version,
            )
        ):
            raise ValueError("sample identity strings must be non-empty")
        require_identifier(self.method_config_id, field_name="method_config_id")
        if any(value <= 0 for value in (self.batch_size, self.context_length, self.decode_step, self.replicate)):
            raise ValueError("batch, context, decode step, and replicate must be positive")
        if self.step_index < 0 or self.random_seed < 0:
            raise ValueError("step_index and random_seed must be nonnegative")
        for digest in (
            self.hardware_fingerprint,
            self.software_fingerprint,
            self.contract_fingerprint,
            self.model_fingerprint,
        ):
            require_sha256(digest)
        if self.method_config_fingerprint.method is not self.method:
            raise ValueError("sample method fingerprint does not match method")
        bytes_values = (
            self.logical_bf16_bytes,
            self.cache_allocated_bytes,
            self.cache_data_bytes,
            self.metadata_bytes,
            self.scale_zero_bytes,
            self.norm_bytes,
            self.residual_bytes,
            self.sink_bytes,
            self.outlier_value_bytes,
            self.outlier_index_bytes,
            self.padding_bytes,
            self.workspace_bytes,
            self.peak_memory_bytes,
        )
        if any(value is not None and value < 0 for value in bytes_values):
            raise ValueError("byte metrics must be nonnegative")
        for value in (
            self.r_nominal,
            self.r_alloc,
            self.r_hbm,
            self.wall_time_ms,
            self.gpu_event_ms,
            self.power_w,
        ):
            if value is not None and value <= 0.0:
                raise ValueError("ratio/time/power metrics must be positive")
        for value in (self.kernel_count, self.sm_clock_mhz, self.memory_clock_mhz):
            if value is not None and value < 0:
                raise ValueError("count/clock metrics must be nonnegative")
        components = (
            self.cache_data_bytes,
            self.metadata_bytes,
            self.scale_zero_bytes,
            self.norm_bytes,
            self.residual_bytes,
            self.sink_bytes,
            self.outlier_value_bytes,
            self.outlier_index_bytes,
            self.padding_bytes,
        )
        allocation_evidence = (
            self.logical_bf16_bytes,
            self.cache_allocated_bytes,
            *components,
        )
        if self.r_alloc is not None or any(value is not None for value in allocation_evidence):
            if any(value is None for value in allocation_evidence):
                raise ValueError("allocation byte evidence must be complete or entirely null")
            if self.cache_allocated_bytes != sum(value or 0 for value in components):
                raise ValueError("cache byte breakdown does not sum to allocated bytes")
        if self.peak_memory_bytes is not None and self.cache_allocated_bytes is not None:
            workspace = self.workspace_bytes or 0
            if self.peak_memory_bytes < self.cache_allocated_bytes + workspace:
                raise ValueError("peak memory is below cache plus workspace")
        if self.r_alloc is not None:
            if self.logical_bf16_bytes is None or not self.cache_allocated_bytes:
                raise ValueError("r_alloc requires logical and allocated byte evidence")
            expected = self.logical_bf16_bytes / self.cache_allocated_bytes
            if not math.isclose(self.r_alloc, expected, rel_tol=1e-9, abs_tol=0.0):
                raise ValueError("r_alloc does not match byte evidence")
        if self.r_hbm is None:
            if self.hbm_evidence is not None:
                raise ValueError("HBM evidence requires an r_hbm value")
        else:
            if self.run_kind is not RunKind.NCU or self.hbm_evidence is None:
                raise ValueError("r_hbm requires direct ncu evidence")
            expected_hbm = (
                self.hbm_evidence.bf16_hbm_bytes
                / self.hbm_evidence.method_hbm_bytes
            )
            if not math.isclose(self.r_hbm, expected_hbm, rel_tol=1e-9):
                raise ValueError("r_hbm does not match direct evidence")
            if self.hbm_evidence.source_run_id != self.run_id:
                raise ValueError("HBM evidence source_run_id must match the sample run")
        if self.run_kind is RunKind.TIMING:
            if self.status is RunStatus.COMPLETED and (
                self.wall_time_ms is None or self.gpu_event_ms is None
            ):
                raise ValueError("completed timing sample requires both timing metrics")
            if self.r_hbm is not None or self.hbm_evidence is not None:
                raise ValueError("timing samples cannot contain profiler HBM evidence")
        elif self.run_kind in {RunKind.NSYS, RunKind.NCU}:
            if self.wall_time_ms is not None or self.gpu_event_ms is not None:
                raise ValueError("profiler-instrumented timing must remain null")
        if self.run_kind in {RunKind.TIMING, RunKind.NSYS, RunKind.NCU}:
            if self.git_dirty or self.container_digest is None:
                raise ValueError("formal sample requires clean Git and container digest")
            require_oci_digest(self.container_digest)
            require_git_sha(self.model_revision)
            if self.weight_dtype != "bfloat16":
                raise ValueError("formal Phase 2 study records require bfloat16 weights")
            if not self.method_config_fingerprint.execution_ready:
                raise ValueError("formal sample requires an execution-ready fingerprint")
            if (
                self.quality.quality_status is not QualityValidationState.UNVALIDATED
                or self.quality.claim_eligibility is not ClaimEligibility.PERFORMANCE_ONLY
            ):
                raise ValueError("formal samples require unvalidated/performance_only quality metadata")
        if self.run_kind is RunKind.TIMING and self.claim_class not in {
            ClaimClass.SAME_WORK_LATENCY,
            ClaimClass.CAPACITY_AMPLIFICATION,
            ClaimClass.MECHANISM_ONLY,
        }:
            raise ValueError("timing sample requires an explicit performance claim class")
        if self.run_kind in {RunKind.NSYS, RunKind.NCU} and (
            self.claim_class is not ClaimClass.MECHANISM_ONLY
        ):
            raise ValueError("profiler samples must be mechanism_only")
        if self.run_kind in {RunKind.HARDWARE_PREFLIGHT, RunKind.CORRECTNESS} and (
            self.claim_class is not ClaimClass.NONE
        ):
            raise ValueError("preflight/correctness samples cannot carry claims")
        if self.run_kind is RunKind.SYNTHETIC:
            if self.claim_class is not ClaimClass.NONE:
                raise ValueError("synthetic samples cannot carry claims")
            if (
                self.quality.quality_status is not QualityValidationState.NOT_APPLICABLE
                or self.quality.claim_eligibility is not ClaimEligibility.NONE
            ):
                raise ValueError("synthetic samples require not-applicable quality metadata")
        if not self.status.is_terminal:
            raise ValueError("sample status must be terminal")
        if self.status is RunStatus.COMPLETED and self.failure_reason is not None:
            raise ValueError("completed sample cannot carry failure_reason")
        if self.status.is_failure and not self.failure_reason:
            raise ValueError("failed sample requires failure_reason")


@dataclasses.dataclass(frozen=True, slots=True)
class MetricSummary(StrictModel):
    count: int
    minimum: float
    median: float
    maximum: float

    def __post_init__(self) -> None:
        if self.count <= 0 or not all(
            math.isfinite(value) for value in (self.minimum, self.median, self.maximum)
        ):
            raise ValueError("metric summary values must be finite with positive count")
        if not self.minimum <= self.median <= self.maximum:
            raise ValueError("metric summary bounds are not ordered")


@dataclasses.dataclass(frozen=True, slots=True)
class RunSummary(StrictModel):
    schema_version: str
    run_id: str
    status: RunStatus
    run_kind: RunKind
    generated_at_utc: str
    source_manifest_sha256: str
    sample_count: int
    exclusion_count: int
    completed_sample_count: int
    failed_sample_count: int
    wall_time_ms: MetricSummary | None
    quality: QualityStatus
    claim_class: ClaimClass
    scientific_conclusions_generated: bool

    SCHEMA_VERSION: ClassVar[str] = "kvbench-run-summary-1.0.0"

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        if self.run_kind is RunKind.PHASE3_ADMISSION:
            raise ValueError("Phase 3 summaries require the Phase 3 evidence schema")
        require_run_id(self.run_id)
        require_utc_timestamp(self.generated_at_utc)
        require_sha256(self.source_manifest_sha256)
        counts = (
            self.sample_count,
            self.exclusion_count,
            self.completed_sample_count,
            self.failed_sample_count,
        )
        if any(value < 0 for value in counts):
            raise ValueError("summary counts must be nonnegative")
        if self.completed_sample_count + self.failed_sample_count > self.sample_count:
            raise ValueError("sample outcome counts exceed sample_count")
        if not self.status.is_terminal:
            raise ValueError("run summary status must be terminal")
        if self.run_kind in {RunKind.NSYS, RunKind.NCU} and self.wall_time_ms is not None:
            raise ValueError("profiler summaries cannot contain timing metrics")
        if self.wall_time_ms is not None and self.wall_time_ms.count != self.completed_sample_count:
            raise ValueError("timing summary count must match completed samples")
        if self.run_kind is RunKind.TIMING:
            if self.claim_class not in {
                ClaimClass.SAME_WORK_LATENCY,
                ClaimClass.CAPACITY_AMPLIFICATION,
                ClaimClass.MECHANISM_ONLY,
            }:
                raise ValueError("timing summary requires an explicit claim class")
            if (
                self.quality.quality_status is not QualityValidationState.UNVALIDATED
                or self.quality.claim_eligibility is not ClaimEligibility.PERFORMANCE_ONLY
            ):
                raise ValueError("timing summary requires unvalidated/performance_only quality metadata")
        if self.run_kind in {RunKind.NSYS, RunKind.NCU} and (
            self.quality.quality_status is not QualityValidationState.UNVALIDATED
            or self.quality.claim_eligibility is not ClaimEligibility.PERFORMANCE_ONLY
        ):
            raise ValueError("profiler summary requires unvalidated/performance_only quality metadata")
        if self.run_kind in {RunKind.NSYS, RunKind.NCU} and (
            self.claim_class is not ClaimClass.MECHANISM_ONLY
        ):
            raise ValueError("profiler summaries must be mechanism_only")
        if self.run_kind is RunKind.SYNTHETIC:
            if self.claim_class is not ClaimClass.NONE:
                raise ValueError("synthetic summaries cannot carry claims")
            if (
                self.quality.quality_status is not QualityValidationState.NOT_APPLICABLE
                or self.quality.claim_eligibility is not ClaimEligibility.NONE
            ):
                raise ValueError("synthetic summaries require not-applicable quality metadata")
        if self.run_kind in {RunKind.HARDWARE_PREFLIGHT, RunKind.CORRECTNESS} and (
            self.claim_class is not ClaimClass.NONE
        ):
            raise ValueError("preflight/correctness summaries cannot carry claims")
        if self.scientific_conclusions_generated:
            raise ValueError("Phase 2 summaries cannot generate scientific conclusions")


@dataclasses.dataclass(frozen=True, slots=True)
class ExclusionRecord(StrictModel):
    schema_version: str
    run_id: str
    timestamp_utc: str
    status: RunStatus
    reason_code: ExclusionReason
    reason: str
    batch_size: int | None
    context_length: int | None
    method_config_fingerprint: MethodConfigFingerprint | None
    graph_mode: GraphMode | None
    run_kind: RunKind
    predicted_memory_bytes: int | None
    memory_limit_bytes: int | None
    expected_backend: str | None
    observed_backend: str | None
    retry_eligible: bool

    SCHEMA_VERSION: ClassVar[str] = "kvbench-exclusion-1.0.0"

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        require_run_id(self.run_id)
        require_utc_timestamp(self.timestamp_utc)
        if not self.status.is_failure or not self.reason.strip():
            raise ValueError("exclusion requires terminal failure status and reason")
        for name, value in (
            ("batch_size", self.batch_size),
            ("context_length", self.context_length),
            ("predicted_memory_bytes", self.predicted_memory_bytes),
            ("memory_limit_bytes", self.memory_limit_bytes),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.reason_code in {
            ExclusionReason.CAPACITY_INFEASIBLE,
            ExclusionReason.OOM,
        }:
            if self.predicted_memory_bytes is None or self.memory_limit_bytes is None:
                raise ValueError("memory exclusion requires predicted and limit bytes")
        if self.reason_code is ExclusionReason.CAPACITY_INFEASIBLE:
            if self.status is not RunStatus.CAPACITY_INFEASIBLE:
                raise ValueError("capacity reason requires capacity_infeasible status")
            if (self.predicted_memory_bytes or 0) <= (self.memory_limit_bytes or 0):
                raise ValueError("predicted memory must exceed the configured limit")
        if self.reason_code is ExclusionReason.BACKEND_FALLBACK:
            if (
                not self.expected_backend
                or not self.observed_backend
                or self.expected_backend == self.observed_backend
            ):
                raise ValueError("backend fallback requires distinct expected/observed values")
        exact_status = {
            ExclusionReason.CAPACITY_INFEASIBLE: RunStatus.CAPACITY_INFEASIBLE,
            ExclusionReason.UNSTABLE: RunStatus.UNSTABLE,
            ExclusionReason.BACKEND_FALLBACK: RunStatus.BACKEND_FALLBACK,
            ExclusionReason.UNSUPPORTED_GEOMETRY: RunStatus.UNSUPPORTED_GEOMETRY,
            ExclusionReason.PHASE_NOT_IMPLEMENTED: RunStatus.ABORTED,
            ExclusionReason.ABORTED: RunStatus.ABORTED,
        }
        required_status = exact_status.get(self.reason_code)
        if required_status is not None and self.status is not required_status:
            raise ValueError("exclusion reason and terminal status are incompatible")
        if self.reason_code is ExclusionReason.OOM and self.status not in {
            RunStatus.RUNTIME_FAILED,
            RunStatus.CAPACITY_INFEASIBLE,
        }:
            raise ValueError("OOM requires runtime_failed or capacity_infeasible status")
