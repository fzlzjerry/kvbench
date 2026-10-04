"""Strict compact admission report for one cache-method adapter."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

from kvbench.schema.base import (
    ClaimEligibility,
    MeasurementScope,
    QualityExecutionState,
    QualityValidationState,
    StrictModel,
    require_git_sha,
    require_identifier,
    require_relative_path,
    require_schema,
    require_sha256,
    require_utc_timestamp,
)
from kvbench.schema.config import MethodName
from kvbench.schema.phase3 import GateDisposition
from kvbench.schema.phase6 import (
    AUTHORIZED_CONTAINER_DIGEST,
    FIXTURE_SET_SHA256,
    MANDATORY_CONFIG_SLOT_SIZES,
    PINNED_SOURCE_COMMIT,
    PINNED_SOURCE_TREE,
)


@dataclasses.dataclass(frozen=True, slots=True)
class MethodAdmissionEvidenceReference(StrictModel):
    evidence_id: str
    path: str
    sha256: str

    def __post_init__(self) -> None:
        require_identifier(self.evidence_id, field_name="evidence_id")
        require_relative_path(self.path, field_name="evidence path")
        require_sha256(self.sha256)


@dataclasses.dataclass(frozen=True, slots=True)
class MethodAdmissionCheckResult(StrictModel):
    status: GateDisposition
    summary: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.summary.strip():
            raise ValueError("admission check summary must be non-empty")
        if not self.evidence_ids:
            raise ValueError("admission check requires evidence references")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("admission check evidence IDs must be unique")
        for evidence_id in self.evidence_ids:
            require_identifier(evidence_id, field_name="evidence_id")


@dataclasses.dataclass(frozen=True, slots=True)
class MethodAdmissionModelIdentity(StrictModel):
    model_id: str
    revision: str
    fingerprint: str

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        require_git_sha(self.revision)
        require_sha256(self.fingerprint, field_name="model fingerprint")


@dataclasses.dataclass(frozen=True, slots=True)
class MethodAdmissionBackendIdentity(StrictModel):
    backend_id: str
    fingerprint: str

    def __post_init__(self) -> None:
        require_identifier(self.backend_id, field_name="backend_id")
        require_sha256(self.fingerprint, field_name="backend fingerprint")


@dataclasses.dataclass(frozen=True, slots=True)
class MethodAdmissionGates(StrictModel):
    g0: GateDisposition
    g1: GateDisposition
    g2: GateDisposition
    g3: GateDisposition
    g4: GateDisposition
    g5: GateDisposition
    full_scan_state: str

    def __post_init__(self) -> None:
        if self.full_scan_state != "closed":
            raise ValueError("Full Scan must remain closed")


@dataclasses.dataclass(frozen=True, slots=True)
class MethodAdmissionReport(StrictModel):
    """References evidence; raw traces and samples remain external."""

    schema_version: str
    created_at_utc: str
    status: GateDisposition
    method_name: MethodName
    method_config_id: str
    method_config_fingerprint: str
    adapter_version: str
    adapter_config_fingerprint: str
    model_identity: MethodAdmissionModelIdentity
    backend_identity: MethodAdmissionBackendIdentity
    cache_layout_fingerprint: str
    correctness: MethodAdmissionCheckResult
    byte_accounting: MethodAdmissionCheckResult
    execution_path: MethodAdmissionCheckResult
    graph: MethodAdmissionCheckResult
    reproducibility_status: GateDisposition
    evidence_references: tuple[MethodAdmissionEvidenceReference, ...]
    gates: MethodAdmissionGates
    blockers: tuple[str, ...]
    claim_eligibility: ClaimEligibility
    quality_status: QualityValidationState
    quality_execution: QualityExecutionState
    performance_claim_eligible: bool
    performance_data_frozen: bool
    quality_benchmark_executed: bool
    measurement_scope: MeasurementScope
    creation_git_sha: str

    SCHEMA_VERSION: ClassVar[str] = "kvbench-method-admission-report-1.0.0"

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        require_utc_timestamp(self.created_at_utc)
        require_identifier(
            self.method_config_id,
            field_name="method_config_id",
        )
        require_sha256(
            self.method_config_fingerprint,
            field_name="method_config_fingerprint",
        )
        require_identifier(self.adapter_version, field_name="adapter_version")
        require_sha256(
            self.adapter_config_fingerprint,
            field_name="adapter_config_fingerprint",
        )
        require_sha256(
            self.cache_layout_fingerprint,
            field_name="cache_layout_fingerprint",
        )
        require_git_sha(self.creation_git_sha)
        if self.status not in {
            GateDisposition.PASS,
            GateDisposition.PARTIAL,
            GateDisposition.BLOCKED,
            GateDisposition.FAIL,
        }:
            raise ValueError("method admission status must be terminal")
        check_results = (
            self.correctness,
            self.byte_accounting,
            self.execution_path,
            self.graph,
        )
        if self.status is GateDisposition.PASS and (
            any(item.status is not GateDisposition.PASS for item in check_results)
            or self.reproducibility_status is not GateDisposition.PASS
        ):
            raise ValueError("PASS requires every method admission check to pass")
        references = {
            item.evidence_id: item for item in self.evidence_references
        }
        if len(references) != len(self.evidence_references):
            raise ValueError("evidence reference IDs must be unique")
        referenced_ids = {
            evidence_id
            for result in check_results
            for evidence_id in result.evidence_ids
        }
        if referenced_ids != set(references):
            raise ValueError("check results and evidence references must join exactly")
        if (
            not self.blockers
            or len(set(self.blockers)) != len(self.blockers)
            or any(not blocker.strip() for blocker in self.blockers)
        ):
            raise ValueError("blockers must be non-empty and unique")
        if (
            self.claim_eligibility is not ClaimEligibility.PERFORMANCE_ONLY
            or self.quality_status is not QualityValidationState.UNVALIDATED
            or self.quality_execution is not QualityExecutionState.LOCKED
            or self.performance_claim_eligible
            or self.performance_data_frozen
            or self.quality_benchmark_executed
            or self.measurement_scope is not MeasurementScope.NATIVE_HOST_ADMISSION
        ):
            raise ValueError("method admission must retain native-host non-claim state")
        if self.method_name is MethodName.BF16:
            expected_gates = (
                GateDisposition.PASS,
                GateDisposition.PASS,
                GateDisposition.NOT_EVALUATED,
                GateDisposition.NOT_EVALUATED,
                GateDisposition.NOT_EVALUATED,
                GateDisposition.NOT_EVALUATED,
            )
            observed_gates = (
                self.gates.g0,
                self.gates.g1,
                self.gates.g2,
                self.gates.g3,
                self.gates.g4,
                self.gates.g5,
            )
            if observed_gates != expected_gates:
                raise ValueError("BF16 report must preserve G0/G1 and leave G2-G5 open")
            if self.blockers != ("B-009", "B-010"):
                raise ValueError("BF16 report must retain B-009 and B-010")

PHASE6_METHOD_ADMISSION_CHECK_IDS = (
    "fixture_conformance",
    "byte_accounting",
    "static_cache_skip_policy",
    "store_append_correctness",
    "decode_tolerance",
    "finite_output",
    "no_full_prefix_dequantization",
    "no_gqa_replication",
    "no_cache_growth",
    "no_unknown_allocation",
    "graph_capture_replay",
    "graph_zero_replay_allocation",
    "no_backend_fallback",
    "compute_sanitizer",
    "bounded_admission_grid",
    "immutable_checksums",
    "durable_publication",
)


@dataclasses.dataclass(frozen=True, slots=True)
class MethodAdmissionCheckV2(StrictModel):
    """One independently evidenced Phase 6 gate."""

    check_id: str
    status: GateDisposition
    summary: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        require_identifier(self.check_id, field_name="check_id")
        if self.check_id not in PHASE6_METHOD_ADMISSION_CHECK_IDS:
            raise ValueError("unknown Phase 6 admission check")
        if not self.summary.strip():
            raise ValueError("admission check summary must be non-empty")
        if not self.evidence_ids:
            raise ValueError("admission check requires evidence")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("admission check evidence IDs must be unique")
        for evidence_id in self.evidence_ids:
            require_identifier(evidence_id, field_name="evidence_id")


@dataclasses.dataclass(frozen=True, slots=True)
class MethodAdmissionGatesV2(StrictModel):
    g0: GateDisposition
    g1: GateDisposition
    g2_tq: GateDisposition
    global_g2: GateDisposition
    g3: GateDisposition
    g4: GateDisposition
    g5: GateDisposition
    full_scan_state: str

    def __post_init__(self) -> None:
        if (
            self.g0,
            self.g1,
            self.global_g2,
            self.g3,
            self.g4,
            self.g5,
            self.full_scan_state,
        ) != (
            GateDisposition.PASS,
            GateDisposition.PASS,
            GateDisposition.NOT_EVALUATED,
            GateDisposition.NOT_EVALUATED,
            GateDisposition.NOT_EVALUATED,
            GateDisposition.NOT_EVALUATED,
            "CLOSED",
        ):
            raise ValueError("Phase 6 must preserve global admission gates")


@dataclasses.dataclass(frozen=True, slots=True)
class MethodAdmissionReportV2(StrictModel):
    """One compact G2-TQ report joining all mandatory configuration evidence."""

    schema_version: str
    created_at_utc: str
    status: GateDisposition
    method_name: MethodName
    mandatory_config_ids: tuple[str, ...]
    admitted_config_ids: tuple[str, ...]
    method_config_fingerprints: dict[str, str]
    adapter_version: str
    adapter_source_sha256: str
    adapter_config_fingerprints: dict[str, str]
    source_commit: str
    source_tree: str
    fixture_set_sha256: str
    container_digest: str
    model_identity: MethodAdmissionModelIdentity
    backend_identity: MethodAdmissionBackendIdentity
    cache_layout_fingerprints: dict[str, str]
    checks: tuple[MethodAdmissionCheckV2, ...]
    reproducibility_status: GateDisposition
    evidence_references: tuple[MethodAdmissionEvidenceReference, ...]
    gates: MethodAdmissionGatesV2
    blockers: tuple[str, ...]
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

    SCHEMA_VERSION: ClassVar[str] = "kvbench-method-admission-report-2.0.0"

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        require_utc_timestamp(self.created_at_utc)
        require_git_sha(self.creation_git_sha)
        require_git_sha(self.source_commit)
        require_git_sha(self.source_tree)
        if self.method_name is not MethodName.TURBOQUANT:
            raise ValueError("Phase 6 admission method must be TurboQuant")
        mandatory = tuple(MANDATORY_CONFIG_SLOT_SIZES)
        if self.mandatory_config_ids != mandatory:
            raise ValueError("mandatory TurboQuant configuration set differs")
        if len(set(self.admitted_config_ids)) != len(self.admitted_config_ids):
            raise ValueError("admitted configuration IDs must be unique")
        if any(item not in mandatory for item in self.admitted_config_ids):
            raise ValueError("admitted configuration is not mandatory")
        for mapping, name in (
            (self.method_config_fingerprints, "method config"),
            (self.adapter_config_fingerprints, "adapter config"),
            (self.cache_layout_fingerprints, "cache layout"),
        ):
            if set(mapping) != set(mandatory):
                raise ValueError(f"{name} fingerprints do not cover mandatory set")
            for digest in mapping.values():
                require_sha256(digest, field_name=f"{name} fingerprint")
        require_identifier(self.adapter_version, field_name="adapter_version")
        require_sha256(
            self.adapter_source_sha256,
            field_name="adapter_source_sha256",
        )
        require_sha256(self.fixture_set_sha256, field_name="fixture_set_sha256")
        if (
            self.source_commit,
            self.source_tree,
            self.fixture_set_sha256,
            self.container_digest,
        ) != (
            PINNED_SOURCE_COMMIT,
            PINNED_SOURCE_TREE,
            FIXTURE_SET_SHA256,
            AUTHORIZED_CONTAINER_DIGEST,
        ):
            raise ValueError("Phase 6 report authority differs")
        observed_check_ids = tuple(item.check_id for item in self.checks)
        if observed_check_ids != PHASE6_METHOD_ADMISSION_CHECK_IDS:
            raise ValueError("Phase 6 admission checks are missing or reordered")
        references = {
            item.evidence_id: item for item in self.evidence_references
        }
        if len(references) != len(self.evidence_references):
            raise ValueError("evidence reference IDs must be unique")
        referenced = {
            evidence_id
            for check in self.checks
            for evidence_id in check.evidence_ids
        }
        if referenced != set(references):
            raise ValueError("checks and evidence references must join exactly")
        if self.status not in {
            GateDisposition.PASS,
            GateDisposition.PARTIAL,
            GateDisposition.BLOCKED,
            GateDisposition.FAIL,
        }:
            raise ValueError("G2-TQ status must be terminal")
        if self.status is GateDisposition.PASS:
            if (
                any(
                    check.status is not GateDisposition.PASS
                    for check in self.checks
                )
                or self.reproducibility_status is not GateDisposition.PASS
                or self.admitted_config_ids != mandatory
                or self.gates.g2_tq is not GateDisposition.PASS
                or self.blockers
            ):
                raise ValueError("G2-TQ PASS requires every mandatory gate")
        else:
            if (
                not self.blockers
                or len(set(self.blockers)) != len(self.blockers)
                or any(not item.strip() for item in self.blockers)
                or self.gates.g2_tq is not self.status
            ):
                raise ValueError("non-PASS G2-TQ requires exact blockers/status")
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
            raise ValueError("Phase 6 report must retain non-claim governance")
