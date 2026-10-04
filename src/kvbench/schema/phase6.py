"""Strict non-claim-bearing Phase 6 TurboQuant admission manifest."""

from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import Any, ClassVar

from kvbench.schema.base import (
    ClaimClass,
    ClaimEligibility,
    GraphMode,
    MeasurementScope,
    QualityExecutionState,
    QualityValidationState,
    RunKind,
    RunnerKind,
    RunStatus,
    StrictModel,
    require_git_sha,
    require_identifier,
    require_oci_digest,
    require_relative_path,
    require_run_id,
    require_schema,
    require_sha256,
    require_utc_timestamp,
)
from kvbench.schema.config import MethodName


AUTHORIZED_CONTAINER_DIGEST = (
    "sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e"
)
PINNED_SOURCE_COMMIT = "752a3a504485790a2e8491cacbb35c137339ad34"
PINNED_SOURCE_TREE = "3ec7a4eb00f9bc8fec399bea6cf7de27a7936372"
FIXTURE_SET_SHA256 = (
    "774ec946a8839d4de012bc6fba0ee5a933ab1488ecc43354d8573b4481b12f76"
)
FIXTURE_ROOT_LEDGER_SHA256 = (
    "d4dbf7933c417a956c3789af404b38aac146f705ec7f8e2a03cad999fc294b38"
)
MANDATORY_CONFIG_SLOT_SIZES = {
    "turboquant_4bit_nc": 134,
    "turboquant_k3v4_nc": 118,
    "turboquant_3bit_nc": 102,
}
BF16_LAYERS = (0, 1, 30, 31)
COMPRESSED_LAYERS = tuple(range(2, 30))
STORE_SOURCE_SHA256 = (
    "4c71a35db7264d6a8cae66e9eac5757b1f7f6dcf8fdcdb3913c6e4015ddd9679"
)
DECODE_SOURCE_SHA256 = (
    "99845ff4e71f12cbc9571763bd259cf8d64987b7d2e591e49b98f1dc9fb25f4c"
)
STAGE2_SOURCE_SHA256 = (
    "920b2aca82e62ea894bf2010c9871ec9f59db422c3f8210d62794992013ef352"
)


@dataclasses.dataclass(frozen=True, slots=True)
class Phase6BackendIdentity(StrictModel):
    """Exact compressed-cache kernel family and carried-source identity."""

    schema_version: str
    backend_id: str
    backend_fingerprint: str
    store_kernel_family: str
    decode_kernel_families: tuple[str, ...]
    store_source_sha256: str
    decode_source_sha256: str
    stage2_source_sha256: str

    SCHEMA_VERSION: ClassVar[str] = "kvbench-phase6-backend-identity-1.0.0"

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        require_identifier(self.backend_id, field_name="backend_id")
        for value, name in (
            (self.backend_fingerprint, "backend_fingerprint"),
            (self.store_source_sha256, "store_source_sha256"),
            (self.decode_source_sha256, "decode_source_sha256"),
            (self.stage2_source_sha256, "stage2_source_sha256"),
        ):
            require_sha256(value, field_name=name)
        if self.store_kernel_family != "_tq_fused_store_mse":
            raise ValueError("Phase 6 store kernel family is not pinned")
        if self.decode_kernel_families != (
            "_tq_decode_stage1",
            "_fwd_kernel_stage2",
        ):
            raise ValueError("Phase 6 decode kernel families are not pinned")
        if (
            self.store_source_sha256,
            self.decode_source_sha256,
            self.stage2_source_sha256,
        ) != (
            STORE_SOURCE_SHA256,
            DECODE_SOURCE_SHA256,
            STAGE2_SOURCE_SHA256,
        ):
            raise ValueError("Phase 6 carried kernel source identity differs")


def _validate_lifecycle(
    *,
    status: RunStatus,
    created_at_utc: str,
    started_at_utc: str | None,
    finished_at_utc: str | None,
    inventory_path: str | None,
    failure_reason: str | None,
) -> None:
    require_utc_timestamp(created_at_utc, field_name="created_at_utc")
    if status is RunStatus.CREATED:
        if any(
            value is not None
            for value in (
                started_at_utc,
                finished_at_utc,
                inventory_path,
                failure_reason,
            )
        ):
            raise ValueError("created manifest has later lifecycle fields")
    elif status in {RunStatus.RUNNING, RunStatus.FINALIZING}:
        if started_at_utc is None:
            raise ValueError("nonterminal manifest requires started_at_utc")
        require_utc_timestamp(started_at_utc, field_name="started_at_utc")
        if any(
            value is not None
            for value in (finished_at_utc, inventory_path, failure_reason)
        ):
            raise ValueError("nonterminal manifest has terminal-only fields")
    else:
        if started_at_utc is None or finished_at_utc is None:
            raise ValueError("terminal manifest requires lifecycle timestamps")
        require_utc_timestamp(started_at_utc, field_name="started_at_utc")
        require_utc_timestamp(finished_at_utc, field_name="finished_at_utc")
        if inventory_path != "artifact_inventory.json":
            raise ValueError("terminal manifest has the wrong inventory path")
        if status is RunStatus.COMPLETED and failure_reason is not None:
            raise ValueError("completed manifest cannot carry failure_reason")
        if status.is_failure and not failure_reason:
            raise ValueError("terminal failure requires a reason")
    timestamps = [
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        for value in (created_at_utc, started_at_utc, finished_at_utc)
        if value is not None
    ]
    if timestamps != sorted(timestamps):
        raise ValueError("manifest lifecycle timestamps are out of order")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase6RunManifest(StrictModel):
    """One exact point from the bounded Measurement Container grid."""

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
    measurement_scope: MeasurementScope
    performance_claim_eligible: bool
    git_sha: str
    git_dirty: bool
    container_digest: str
    method: MethodName
    method_config_id: str
    method_config_fingerprint: str
    adapter_version: str
    adapter_source_sha256: str
    adapter_config_fingerprint: str
    pinned_source_commit: str
    pinned_source_tree: str
    fixture_set_sha256: str
    fixture_root_ledger_sha256: str
    cache_layout_fingerprint: str
    slot_size_bytes: int
    compressed_layers: tuple[int, ...]
    bf16_layers: tuple[int, ...]
    backend_identity: Phase6BackendIdentity
    batch_size: int
    context_length: int
    output_steps: int
    quality_status: QualityValidationState
    claim_eligibility: ClaimEligibility
    quality_execution: QualityExecutionState
    performance_data_frozen: bool
    quality_benchmark_executed: bool
    speedup_calculated: bool
    r_hbm: None
    full_scan_state: str
    inventory_path: str | None
    failure_reason: str | None

    SCHEMA_VERSION: ClassVar[str] = "kvbench-phase6-run-manifest-1.0.0"
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
            raise ValueError("Phase 6 admission requires a clean Git tree")
        require_oci_digest(self.container_digest)
        if self.container_digest != AUTHORIZED_CONTAINER_DIGEST:
            raise ValueError("Phase 6 requires the authorized image digest")
        if self.run_kind is not RunKind.PHASE6_ADMISSION:
            raise ValueError("Phase 6 run kind must be phase6_admission")
        if self.claim_class is not ClaimClass.NONE:
            raise ValueError("Phase 6 admission cannot carry a claim")
        if (
            self.measurement_scope
            is not MeasurementScope.MEASUREMENT_CONTAINER_ADMISSION
            or self.performance_claim_eligible
        ):
            raise ValueError("Phase 6 is container admission only")
        if self.method is not MethodName.TURBOQUANT:
            raise ValueError("Phase 6 manifest method must be TurboQuant")
        if self.method_config_id not in MANDATORY_CONFIG_SLOT_SIZES:
            raise ValueError("Phase 6 manifest configuration is not mandatory")
        require_identifier(
            self.method_config_id,
            field_name="method_config_id",
        )
        for value, name in (
            (self.method_config_fingerprint, "method_config_fingerprint"),
            (self.adapter_source_sha256, "adapter_source_sha256"),
            (self.adapter_config_fingerprint, "adapter_config_fingerprint"),
            (self.fixture_set_sha256, "fixture_set_sha256"),
            (self.fixture_root_ledger_sha256, "fixture_root_ledger_sha256"),
            (self.cache_layout_fingerprint, "cache_layout_fingerprint"),
        ):
            require_sha256(value, field_name=name)
        if not self.adapter_version.strip():
            raise ValueError("adapter_version must be non-empty")
        require_git_sha(self.pinned_source_commit)
        require_git_sha(self.pinned_source_tree)
        if (
            self.pinned_source_commit,
            self.pinned_source_tree,
            self.fixture_set_sha256,
            self.fixture_root_ledger_sha256,
        ) != (
            PINNED_SOURCE_COMMIT,
            PINNED_SOURCE_TREE,
            FIXTURE_SET_SHA256,
            FIXTURE_ROOT_LEDGER_SHA256,
        ):
            raise ValueError("Phase 6 source or fixture authority differs")
        if self.slot_size_bytes != MANDATORY_CONFIG_SLOT_SIZES[
            self.method_config_id
        ]:
            raise ValueError("Phase 6 slot size differs from the pinned source")
        if (
            self.compressed_layers != COMPRESSED_LAYERS
            or self.bf16_layers != BF16_LAYERS
        ):
            raise ValueError("Phase 6 full-model layer policy differs")
        if type(self.batch_size) is not int or self.batch_size != 1:
            raise ValueError("Phase 6 bounded admission requires B=1")
        point = (
            self.runner_kind,
            self.graph_mode,
            self.context_length,
            self.output_steps,
        )
        if self.runner_kind is RunnerKind.FIXED_L:
            if self.output_steps != 1:
                raise ValueError("fixed-L admission has one scratch-slot step")
            allowed = {
                (RunnerKind.FIXED_L, GraphMode.EAGER, 128, 1),
                (RunnerKind.FIXED_L, GraphMode.CUDA_GRAPH, 128, 1),
            }
            if self.method_config_id == "turboquant_4bit_nc":
                allowed |= {
                    (RunnerKind.FIXED_L, GraphMode.EAGER, 4096, 1),
                    (RunnerKind.FIXED_L, GraphMode.CUDA_GRAPH, 4096, 1),
                }
            if point not in allowed:
                raise ValueError("fixed-L point is outside the bounded grid")
        elif point != (
            RunnerKind.GROWING_CONTEXT,
            GraphMode.EAGER,
            128,
            4,
        ) or self.method_config_id != "turboquant_4bit_nc":
            raise ValueError("growing point is outside the bounded grid")
        if (
            self.quality_status is not QualityValidationState.UNVALIDATED
            or self.claim_eligibility is not ClaimEligibility.PERFORMANCE_ONLY
            or self.quality_execution is not QualityExecutionState.LOCKED
            or self.performance_data_frozen
            or self.quality_benchmark_executed
            or self.speedup_calculated
            or self.r_hbm is not None
            or self.full_scan_state != "CLOSED"
        ):
            raise ValueError("Phase 6 non-claim governance differs")
        if self.inventory_path is not None:
            require_relative_path(
                self.inventory_path,
                field_name="inventory_path",
            )
        _validate_lifecycle(
            status=self.status,
            created_at_utc=self.created_at_utc,
            started_at_utc=self.started_at_utc,
            finished_at_utc=self.finished_at_utc,
            inventory_path=self.inventory_path,
            failure_reason=self.failure_reason,
        )


def parse_phase6_run_manifest(payload: dict[str, Any]) -> Phase6RunManifest:
    """Parse only the separate strict Phase 6 manifest schema."""

    return Phase6RunManifest.from_dict(payload)
