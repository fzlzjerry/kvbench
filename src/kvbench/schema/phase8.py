"""Strict, narrow records for Phase 8 KIVI admission."""

from __future__ import annotations

import dataclasses
import math
from typing import ClassVar, Literal

from kvbench.schema.base import (
    ClaimEligibility,
    GraphMode,
    MeasurementScope,
    QualityExecutionState,
    QualityValidationState,
    RunnerKind,
    RunStatus,
    StrictModel,
    require_git_sha,
    require_identifier,
    require_oci_digest,
    require_run_id,
    require_schema,
    require_sha256,
    require_utc_timestamp,
)
from kvbench.schema.phase3 import GateDisposition
from kvbench.schema.phase6 import _validate_lifecycle
from kvbench.schema.method_admission import MethodAdmissionEvidenceReference


RATIO_REL_TOLERANCE = 1e-9
RECIPROCAL_ABS_TOLERANCE = 1e-9
PHASE8_AUTHORIZED_CONTAINER_DIGEST = (
    "sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e"
)
PHASE8_OFFICIAL_COMMIT = "876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6"
PHASE8_BASE_TREE = "c94c31b2cfd44eeb9a18cff9dcdf03adff4ac49b"
PHASE8_PATCHED_TREE = "b617493dea5aff1a754cd27ad6be12ac512b2aee"
PHASE8_DECISION_0018_PATCH_SHA256 = (
    "c9c2dd52d4c81b844d1d1d7218ad2cd60a5b31574a387f716d466cb01310423d"
)
PHASE8_EXTENSION_SHA256 = (
    "45d29ec1a3cecc4b253d1d1dd6139ef4f91cff88993db61a9d73685314851aa9"
)
PHASE8_FIXTURE_ROOT_DIGEST = (
    "abd164da0adf9e0c1404e8fba1f6a6e42e57944481cdf060b91e8cef175ed302"
)
PHASE8_MANDATORY_CONFIGS = ("k4v4", "k2v4", "k2v2")
PHASE8_HELD_OUT_CONFIG = "k4v2"


def _require_positive_finite(value: float, *, field_name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{field_name} must be positive and finite")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase8AllocationRatios:
    """Canonical ratios derived from physical method and BF16 capacity."""

    allocated_bytes: int
    bf16_allocated_bytes: int
    rho_alloc: float
    r_alloc: float

    def __post_init__(self) -> None:
        if (
            type(self.allocated_bytes) is not int
            or type(self.bf16_allocated_bytes) is not int
            or self.allocated_bytes <= 0
            or self.bf16_allocated_bytes <= 0
        ):
            raise ValueError("allocation byte counts must be positive integers")
        _require_positive_finite(self.rho_alloc, field_name="rho_alloc")
        _require_positive_finite(self.r_alloc, field_name="r_alloc")
        expected_rho = self.allocated_bytes / self.bf16_allocated_bytes
        expected_r = self.bf16_allocated_bytes / self.allocated_bytes
        if not math.isclose(
            self.rho_alloc,
            expected_rho,
            rel_tol=RATIO_REL_TOLERANCE,
            abs_tol=0.0,
        ):
            raise ValueError("rho_alloc does not equal allocated/BF16")
        if not math.isclose(
            self.r_alloc,
            expected_r,
            rel_tol=RATIO_REL_TOLERANCE,
            abs_tol=0.0,
        ):
            raise ValueError("r_alloc does not equal BF16/allocated")
        if abs(self.r_alloc * self.rho_alloc - 1.0) > RECIPROCAL_ABS_TOLERANCE:
            raise ValueError("r_alloc and rho_alloc are not reciprocal")

    @classmethod
    def from_bytes(
        cls, *, allocated_bytes: int, bf16_allocated_bytes: int
    ) -> "Phase8AllocationRatios":
        return cls(
            allocated_bytes=allocated_bytes,
            bf16_allocated_bytes=bf16_allocated_bytes,
            rho_alloc=allocated_bytes / bf16_allocated_bytes,
            r_alloc=bf16_allocated_bytes / allocated_bytes,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Phase7LegacyAllocationRatio:
    """Explicitly renamed interpretation of the immutable Phase 7 field."""

    rho_alloc_legacy: float

    def __post_init__(self) -> None:
        _require_positive_finite(
            self.rho_alloc_legacy, field_name="rho_alloc_legacy"
        )

    @classmethod
    def from_phase7_r_alloc(cls, legacy_value: float) -> "Phase7LegacyAllocationRatio":
        return cls(rho_alloc_legacy=legacy_value)

    @property
    def canonical_r_alloc(self) -> float:
        return 1.0 / self.rho_alloc_legacy


@dataclasses.dataclass(frozen=True, slots=True)
class Phase8ByteBreakdown(StrictModel):
    """Every adapter-owned persistent storage category, exactly once."""

    quantized_historical_k_payload: int
    quantized_historical_v_payload: int
    k_scales: int
    k_zeros: int
    v_scales: int
    v_zeros: int
    other_metadata: int
    residual_k: int
    residual_v: int
    fp16_staging: int
    quantization_staging: int
    padding_alignment: int
    persistent_workspace: int
    value_rollover_shift_scratch: int
    block_group_rounding: int

    def __post_init__(self) -> None:
        if any(
            type(getattr(self, field.name)) is not int
            or getattr(self, field.name) < 0
            for field in dataclasses.fields(self)
        ):
            raise ValueError("Phase 8 byte categories must be nonnegative integers")

    @property
    def total(self) -> int:
        return sum(
            int(getattr(self, field.name))
            for field in dataclasses.fields(self)
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Phase8ByteAccounting(StrictModel):
    """Capacity allocation and active source storage remain distinct."""

    capacity: int
    active_context: int
    allocated_bytes: int
    predicted_allocated_bytes: int
    active_storage_bytes: int
    logical_bf16_allocated_bytes: int
    logical_bf16_active_bytes: int
    rho_alloc: float
    r_alloc: float
    predicted_relative_error: float
    temporary_peak_bytes: int
    breakdown: Phase8ByteBreakdown
    r_hbm: None

    def __post_init__(self) -> None:
        integer_fields = (
            self.capacity,
            self.active_context,
            self.allocated_bytes,
            self.predicted_allocated_bytes,
            self.active_storage_bytes,
            self.logical_bf16_allocated_bytes,
            self.logical_bf16_active_bytes,
            self.temporary_peak_bytes,
        )
        if any(type(value) is not int or value < 0 for value in integer_fields):
            raise ValueError("Phase 8 accounting integers must be nonnegative")
        if (
            self.capacity <= 0
            or self.active_context > self.capacity
            or self.allocated_bytes <= 0
            or self.predicted_allocated_bytes <= 0
            or self.logical_bf16_allocated_bytes <= 0
            or self.breakdown.total != self.allocated_bytes
        ):
            raise ValueError("Phase 8 accounting capacity or sum is invalid")
        ratios = Phase8AllocationRatios(
            allocated_bytes=self.allocated_bytes,
            bf16_allocated_bytes=self.logical_bf16_allocated_bytes,
            rho_alloc=self.rho_alloc,
            r_alloc=self.r_alloc,
        )
        del ratios
        expected_error = abs(
            self.predicted_allocated_bytes - self.allocated_bytes
        ) / self.allocated_bytes
        if (
            not math.isfinite(self.predicted_relative_error)
            or self.predicted_relative_error < 0.0
            or not math.isclose(
                self.predicted_relative_error,
                expected_error,
                rel_tol=RATIO_REL_TOLERANCE,
                abs_tol=RATIO_REL_TOLERANCE,
            )
            or self.predicted_relative_error >= 0.01
            or self.r_hbm is not None
        ):
            raise ValueError("Phase 8 predicted accounting or r_hbm is invalid")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase8RunManifest(StrictModel):
    """One point from only the preregistered bounded KIVI grid."""

    schema_version: str
    artifact_schema_version: str
    run_id: str
    status: RunStatus
    git_sha: str
    git_dirty: bool
    created_at_utc: str
    started_at_utc: str | None
    finished_at_utc: str | None
    runner_kind: RunnerKind
    graph_mode: GraphMode
    method_configuration: str
    k_bits: int
    v_bits: int
    method_config_fingerprint: str
    method_fingerprint: str
    adapter_version: str
    adapter_source_sha256: str
    official_base_commit: str
    official_base_tree: str
    patched_tree: str
    decision_0018_patch_sha256: str
    extension_sha256: str
    fixture_root_digest: str
    group_size: int
    residual_length: int
    dtype_boundary: Literal["bf16_to_fp16_official_kivi_to_bf16"]
    cache_layout_fingerprint: str
    authorized_container_digest: str
    batch_size: int
    context_length: int
    output_steps: int
    capacity: int
    accounting: Phase8ByteAccounting
    quality_status: QualityValidationState
    claim_eligibility: ClaimEligibility
    performance_claim_eligible: bool
    measurement_scope: MeasurementScope
    quality_execution: QualityExecutionState
    quality_benchmark_executed: bool
    performance_data_frozen: bool
    speedup_calculated: bool
    full_scan_state: Literal["CLOSED"]
    inventory_path: str | None
    failure_reason: str | None

    SCHEMA_VERSION: ClassVar[str] = "kvbench-phase8-kivi-run-manifest-1.0.0"
    ARTIFACT_SCHEMA_VERSION: ClassVar[str] = "kvbench-artifacts-1.0.0"

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        require_schema(
            self.artifact_schema_version,
            self.ARTIFACT_SCHEMA_VERSION,
        )
        require_run_id(self.run_id)
        require_git_sha(self.git_sha)
        if self.git_dirty:
            raise ValueError("Phase 8 admission requires a clean Git tree")
        require_identifier(
            self.method_configuration, field_name="method_configuration"
        )
        expected_bits = {
            "k4v4": (4, 4),
            "k2v4": (2, 4),
            "k2v2": (2, 2),
            "k4v2": (4, 2),
        }
        if (
            self.method_configuration not in expected_bits
            or (self.k_bits, self.v_bits)
            != expected_bits[self.method_configuration]
        ):
            raise ValueError("Phase 8 KIVI configuration differs")
        for value, name in (
            (self.method_config_fingerprint, "method_config_fingerprint"),
            (self.method_fingerprint, "method_fingerprint"),
            (self.adapter_source_sha256, "adapter_source_sha256"),
            (self.decision_0018_patch_sha256, "decision_0018_patch_sha256"),
            (self.extension_sha256, "extension_sha256"),
            (self.fixture_root_digest, "fixture_root_digest"),
            (self.cache_layout_fingerprint, "cache_layout_fingerprint"),
        ):
            require_sha256(value, field_name=name)
        require_identifier(self.adapter_version, field_name="adapter_version")
        require_git_sha(self.official_base_commit)
        require_git_sha(self.official_base_tree)
        require_git_sha(self.patched_tree)
        require_oci_digest(self.authorized_container_digest)
        if (
            self.official_base_commit,
            self.official_base_tree,
            self.patched_tree,
            self.decision_0018_patch_sha256,
            self.extension_sha256,
            self.fixture_root_digest,
            self.authorized_container_digest,
            self.group_size,
            self.residual_length,
        ) != (
            PHASE8_OFFICIAL_COMMIT,
            PHASE8_BASE_TREE,
            PHASE8_PATCHED_TREE,
            PHASE8_DECISION_0018_PATCH_SHA256,
            PHASE8_EXTENSION_SHA256,
            PHASE8_FIXTURE_ROOT_DIGEST,
            PHASE8_AUTHORIZED_CONTAINER_DIGEST,
            32,
            32,
        ):
            raise ValueError("Phase 8 source, fixture, or container authority differs")
        point = (
            self.method_configuration,
            self.runner_kind,
            self.graph_mode,
            self.context_length,
            self.output_steps,
        )
        allowed = {
            *(
                (config, RunnerKind.FIXED_L, graph, 128, 1)
                for config in PHASE8_MANDATORY_CONFIGS
                for graph in (GraphMode.EAGER, GraphMode.CUDA_GRAPH)
            ),
            ("k4v4", RunnerKind.FIXED_L, GraphMode.EAGER, 4096, 1),
            ("k4v4", RunnerKind.FIXED_L, GraphMode.CUDA_GRAPH, 4096, 1),
            ("k4v4", RunnerKind.GROWING_CONTEXT, GraphMode.EAGER, 31, 4),
            ("k4v2", RunnerKind.FIXED_L, GraphMode.EAGER, 128, 1),
        }
        if (
            point not in allowed
            or self.batch_size != 1
            or self.capacity < self.context_length + self.output_steps
            or self.accounting.capacity != self.capacity
            or self.accounting.active_context < self.context_length
        ):
            raise ValueError("run is outside the bounded Phase 8 grid")
        if (
            self.quality_status is not QualityValidationState.UNVALIDATED
            or self.claim_eligibility is not ClaimEligibility.PERFORMANCE_ONLY
            or self.performance_claim_eligible
            or self.measurement_scope
            is not MeasurementScope.MEASUREMENT_CONTAINER_ADMISSION
            or self.quality_execution is not QualityExecutionState.LOCKED
            or self.quality_benchmark_executed
            or self.performance_data_frozen
            or self.speedup_calculated
            or self.full_scan_state != "CLOSED"
        ):
            raise ValueError("Phase 8 non-claim governance differs")
        _validate_lifecycle(
            status=self.status,
            created_at_utc=self.created_at_utc,
            started_at_utc=self.started_at_utc,
            finished_at_utc=self.finished_at_utc,
            inventory_path=self.inventory_path,
            failure_reason=self.failure_reason,
        )


PHASE8_ADMISSION_CHECK_IDS = (
    "fixture_conformance",
    "byte_accounting",
    "residual_rollover",
    "token_integrity",
    "static_cache",
    "no_measured_torch_cat",
    "direct_compressed_decode",
    "native_gqa",
    "no_unknown_allocation",
    "graph_capture_replay",
    "graph_zero_replay_allocation",
    "no_backend_fallback",
    "compute_sanitizer",
    "bounded_admission_grid",
    "immutable_checksums",
    "durable_publication",
    "clean_retrieval",
)


@dataclasses.dataclass(frozen=True, slots=True)
class Phase8AdmissionCheck(StrictModel):
    check_id: str
    status: GateDisposition
    summary: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        require_identifier(self.check_id, field_name="check_id")
        if self.check_id not in PHASE8_ADMISSION_CHECK_IDS:
            raise ValueError("unknown Phase 8 admission check")
        if not self.summary.strip() or not self.evidence_ids:
            raise ValueError("Phase 8 admission check requires summary and evidence")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("Phase 8 evidence IDs must be unique")
        for evidence_id in self.evidence_ids:
            require_identifier(evidence_id, field_name="evidence_id")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase8AdmissionGates(StrictModel):
    g0: GateDisposition
    g1: GateDisposition
    g2_tq: GateDisposition
    g2_kivi: GateDisposition
    global_g2: GateDisposition
    g3: GateDisposition
    g4: GateDisposition
    g5: GateDisposition
    full_scan_state: Literal["CLOSED"]

    def __post_init__(self) -> None:
        if (
            self.g0,
            self.g1,
            self.g2_tq,
            self.global_g2,
            self.g3,
            self.g4,
            self.g5,
        ) != (
            GateDisposition.PASS,
            GateDisposition.PASS,
            GateDisposition.PASS,
            GateDisposition.NOT_EVALUATED,
            GateDisposition.NOT_EVALUATED,
            GateDisposition.NOT_EVALUATED,
            GateDisposition.NOT_EVALUATED,
        ):
            raise ValueError("Phase 8 must preserve global gates")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase8MethodAdmissionReport(StrictModel):
    """One G2-KIVI report joining mandatory and held-out evidence."""

    schema_version: str
    created_at_utc: str
    status: GateDisposition
    mandatory_configurations: tuple[str, ...]
    held_out_configuration: str
    admitted_configurations: tuple[str, ...]
    method_fingerprints: dict[str, str]
    cache_layout_fingerprints: dict[str, str]
    adapter_version: str
    adapter_source_sha256: str
    official_base_commit: str
    official_base_tree: str
    patched_tree: str
    decision_0018_patch_sha256: str
    extension_sha256: str
    fixture_root_digest: str
    authorized_container_digest: str
    checks: tuple[Phase8AdmissionCheck, ...]
    evidence_references: tuple[MethodAdmissionEvidenceReference, ...]
    gates: Phase8AdmissionGates
    blockers: tuple[str, ...]
    local_root_digest: str
    r2_uri: str
    bucket_lock_identity: str
    clean_retrieval: bool
    claim_eligibility: ClaimEligibility
    quality_status: QualityValidationState
    quality_execution: QualityExecutionState
    performance_claim_eligible: bool
    performance_data_frozen: bool
    quality_benchmark_executed: bool
    speedup_calculated: bool
    r_hbm: None
    measurement_scope: MeasurementScope
    creation_git_sha: str

    SCHEMA_VERSION: ClassVar[str] = "kvbench-phase8-kivi-admission-report-1.0.0"

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        require_utc_timestamp(self.created_at_utc)
        require_git_sha(self.creation_git_sha)
        for value, name in (
            (self.adapter_source_sha256, "adapter_source_sha256"),
            (self.decision_0018_patch_sha256, "decision_0018_patch_sha256"),
            (self.extension_sha256, "extension_sha256"),
            (self.fixture_root_digest, "fixture_root_digest"),
            (self.local_root_digest, "local_root_digest"),
        ):
            require_sha256(value, field_name=name)
        for value in self.method_fingerprints.values():
            require_sha256(value, field_name="method_fingerprint")
        for value in self.cache_layout_fingerprints.values():
            require_sha256(value, field_name="cache_layout_fingerprint")
        require_git_sha(self.official_base_commit)
        require_git_sha(self.official_base_tree)
        require_git_sha(self.patched_tree)
        require_oci_digest(self.authorized_container_digest)
        require_identifier(self.adapter_version, field_name="adapter_version")
        if (
            self.mandatory_configurations != PHASE8_MANDATORY_CONFIGS
            or self.held_out_configuration != PHASE8_HELD_OUT_CONFIG
            or set(self.method_fingerprints)
            != set((*PHASE8_MANDATORY_CONFIGS, PHASE8_HELD_OUT_CONFIG))
            or set(self.cache_layout_fingerprints)
            != set((*PHASE8_MANDATORY_CONFIGS, PHASE8_HELD_OUT_CONFIG))
        ):
            raise ValueError("Phase 8 report configuration set differs")
        if (
            self.official_base_commit,
            self.official_base_tree,
            self.patched_tree,
            self.decision_0018_patch_sha256,
            self.extension_sha256,
            self.fixture_root_digest,
            self.authorized_container_digest,
        ) != (
            PHASE8_OFFICIAL_COMMIT,
            PHASE8_BASE_TREE,
            PHASE8_PATCHED_TREE,
            PHASE8_DECISION_0018_PATCH_SHA256,
            PHASE8_EXTENSION_SHA256,
            PHASE8_FIXTURE_ROOT_DIGEST,
            PHASE8_AUTHORIZED_CONTAINER_DIGEST,
        ):
            raise ValueError("Phase 8 report authority differs")
        if tuple(check.check_id for check in self.checks) != PHASE8_ADMISSION_CHECK_IDS:
            raise ValueError("Phase 8 admission checks are missing or reordered")
        references = {
            reference.evidence_id: reference
            for reference in self.evidence_references
        }
        if len(references) != len(self.evidence_references):
            raise ValueError("Phase 8 evidence reference IDs must be unique")
        referenced = {
            evidence_id
            for check in self.checks
            for evidence_id in check.evidence_ids
        }
        if referenced != set(references):
            raise ValueError("Phase 8 checks and evidence must join exactly")
        if self.status not in {
            GateDisposition.PASS,
            GateDisposition.PARTIAL,
            GateDisposition.BLOCKED,
            GateDisposition.FAIL,
        }:
            raise ValueError("G2-KIVI status must be terminal")
        if self.status is GateDisposition.PASS:
            if (
                self.admitted_configurations != PHASE8_MANDATORY_CONFIGS
                or any(
                    check.status is not GateDisposition.PASS
                    for check in self.checks
                )
                or self.gates.g2_kivi is not GateDisposition.PASS
                or self.blockers
                or not self.clean_retrieval
                or not self.r2_uri.startswith(
                    "r2://kvbench-artifacts/kvbench/sha256/"
                )
                or not self.bucket_lock_identity.strip()
            ):
                raise ValueError("G2-KIVI PASS requires every local and durable gate")
        elif (
            not self.blockers
            or self.gates.g2_kivi is not self.status
            or self.clean_retrieval
        ):
            raise ValueError("non-PASS G2-KIVI must carry exact blockers")
        if (
            self.claim_eligibility is not ClaimEligibility.PERFORMANCE_ONLY
            or self.quality_status is not QualityValidationState.UNVALIDATED
            or self.quality_execution is not QualityExecutionState.LOCKED
            or self.performance_claim_eligible
            or self.performance_data_frozen
            or self.quality_benchmark_executed
            or self.speedup_calculated
            or self.r_hbm is not None
            or self.measurement_scope
            is not MeasurementScope.MEASUREMENT_CONTAINER_ADMISSION
        ):
            raise ValueError("Phase 8 report must remain a non-claim")
