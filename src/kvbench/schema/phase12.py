"""Strict records for the narrow Phase 12 unified-admission campaign."""

from __future__ import annotations

import dataclasses
import math
import random
import statistics
from enum import StrEnum
from typing import ClassVar, Literal

from kvbench.schema.base import (
    QualityExecutionState,
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
from kvbench.schema.phase3 import GateDisposition


PHASE12_AUTHORIZED_CONTAINER_DIGEST = (
    "sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e"
)
PHASE12_MAIN_CONFIGURATIONS = (
    "bf16",
    "tq_4bit_nc",
    "tq_k3v4_nc",
    "tq_3bit_nc",
    "k4v4",
    "k2v4",
    "k2v2",
    "kvq4",
    "kvq3",
    "kvq2",
)
PHASE12_HELD_OUT_CONFIGURATIONS = ("turboquant_k8v4", "k4v2")
PHASE12_CONFIG_FINGERPRINTS = {
    "bf16": "81ca6a0d74727a9c8a54f14d6d222dee502ee5e9a9ab6e8c9a03b4fed98f371b",
    "tq_4bit_nc": (
        "5b56167c81ef2042be5fa45ed4e7f8ddc60670d5611283e19e848307076c27eb"
    ),
    "tq_k3v4_nc": (
        "ff92cd334059888584564bb84999353a698baf3b991a602a827a53aab726908e"
    ),
    "tq_3bit_nc": (
        "0d17950c2502fe7399cf5a896efa429861eb53194060ab95a7369287889dc49a"
    ),
    "k4v4": "97289ed9c875e27013ddcf7659fc6e849b3d438c58d0d86bd3dcac5d82eefb09",
    "k2v4": "568493e09cad122088716533c954beb6b25a01209fa28016f761b2ede4930a3f",
    "k2v2": "667395cefa882efc7c54f9088e3706dcdc3ba33c8734bdf8de9e0dd8ae1124b8",
    "kvq4": "8f3ea4f49056a5c4ada715a853ec506de4b6bcab262cfd88dc5796bacc032fa0",
    "kvq3": "2f0d1a99db2e6884745b6cd54c50eedfa17744b89a3e8b2ffd840986126bd802",
    "kvq2": "eb75d6cbf8ff27365cd2799c4e0232649c94d6f094cb4d041bbe8c3ac1cda5ee",
}
PHASE12_RANDOMIZATION_SEEDS = (20260730, 20260731, 20260732)
PHASE12_RANDOMIZED_ORDERS = (
    (
        "k2v4",
        "k2v2",
        "kvq3",
        "tq_3bit_nc",
        "bf16",
        "tq_4bit_nc",
        "kvq4",
        "k4v4",
        "tq_k3v4_nc",
        "kvq2",
    ),
    (
        "tq_k3v4_nc",
        "k2v4",
        "k2v2",
        "kvq4",
        "kvq2",
        "tq_4bit_nc",
        "kvq3",
        "tq_3bit_nc",
        "k4v4",
        "bf16",
    ),
    (
        "tq_k3v4_nc",
        "tq_4bit_nc",
        "k4v4",
        "bf16",
        "k2v4",
        "k2v2",
        "tq_3bit_nc",
        "kvq2",
        "kvq3",
        "kvq4",
    ),
)
PHASE12_REPLICATES = 3
PHASE12_CV_THRESHOLD = 0.03
PHASE12_PRIOR_GATES = ("G1", "G2", "G3", "G4")
PHASE12_RUNNER_KIND = "fixed_l"
PHASE12_GRAPH_MODE = "cuda_graph"
PHASE12_BATCH_SIZE = 1
PHASE12_CONTEXT_LENGTH = 4096
PHASE12_WARMUP_STEPS = 64
PHASE12_MEASURED_STEPS = 128
PHASE12_MEASURED_BATCHES = 5
PHASE12_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
PHASE12_MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
PHASE12_TOKENIZER_ID = PHASE12_MODEL_ID
PHASE12_TOKENIZER_REVISION = PHASE12_MODEL_REVISION


def derive_phase12_randomized_order(seed: int) -> tuple[str, ...]:
    if type(seed) is not int or seed not in PHASE12_RANDOMIZATION_SEEDS:
        raise ValueError("Phase 12 randomization seed is not frozen")
    order = list(PHASE12_MAIN_CONFIGURATIONS)
    random.Random(seed).shuffle(order)
    return tuple(order)


if tuple(
    derive_phase12_randomized_order(seed)
    for seed in PHASE12_RANDOMIZATION_SEEDS
) != PHASE12_RANDOMIZED_ORDERS:
    raise RuntimeError("frozen Phase 12 randomized orders do not derive")


class Phase12G5Disposition(StrEnum):
    PASS = "PASS"
    UNSTABLE = "unstable"


class Phase12PublicationState(StrEnum):
    PENDING = "PENDING"
    PASS = "PASS"


@dataclasses.dataclass(frozen=True, slots=True)
class Phase12EvidenceReference(StrictModel):
    evidence_id: str
    path: str
    sha256: str

    def __post_init__(self) -> None:
        require_identifier(self.evidence_id, field_name="evidence_id")
        require_relative_path(self.path, field_name="evidence path")
        require_sha256(self.sha256)


@dataclasses.dataclass(frozen=True, slots=True)
class Phase12PriorGateEvidence(StrictModel):
    gate: Literal["G1", "G2", "G3", "G4"]
    status: GateDisposition
    criteria_satisfied: bool
    evidence: tuple[Phase12EvidenceReference, ...]

    def __post_init__(self) -> None:
        if self.gate not in PHASE12_PRIOR_GATES:
            raise ValueError("unknown Phase 12 prior gate")
        if not self.evidence:
            raise ValueError("prior-gate evidence must not be empty")
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("prior-gate evidence IDs must be unique")
        if (self.status is GateDisposition.PASS) != self.criteria_satisfied:
            raise ValueError("prior gate PASS must exactly match its criteria")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase12ByteAccounting(StrictModel):
    data_payload_bytes: int
    metadata_bytes: int
    sparse_bytes: int
    sink_residual_bytes: int
    padding_bytes: int
    workspace_bytes: int
    predicted_allocated_bytes: int
    allocated_bytes: int
    logical_bf16_bytes: int
    rho_alloc: float
    r_alloc: float
    predicted_relative_error: float
    r_hbm: None

    def __post_init__(self) -> None:
        category_values = (
            self.data_payload_bytes,
            self.metadata_bytes,
            self.sparse_bytes,
            self.sink_residual_bytes,
            self.padding_bytes,
            self.workspace_bytes,
        )
        if any(type(value) is not int or value < 0 for value in category_values):
            raise ValueError("Phase 12 byte categories must be nonnegative integers")
        if (
            type(self.predicted_allocated_bytes) is not int
            or type(self.allocated_bytes) is not int
            or type(self.logical_bf16_bytes) is not int
            or self.predicted_allocated_bytes <= 0
            or self.allocated_bytes <= 0
            or self.logical_bf16_bytes <= 0
        ):
            raise ValueError("Phase 12 allocation totals must be positive integers")
        if sum(category_values) != self.allocated_bytes:
            raise ValueError("Phase 12 byte categories must sum to owned bytes")
        expected_error = (
            abs(self.predicted_allocated_bytes - self.allocated_bytes)
            / self.allocated_bytes
        )
        expected_rho = self.allocated_bytes / self.logical_bf16_bytes
        expected_r = self.logical_bf16_bytes / self.allocated_bytes
        observed_and_expected = (
            (self.predicted_relative_error, expected_error),
            (self.rho_alloc, expected_rho),
            (self.r_alloc, expected_r),
        )
        if any(
            not math.isfinite(observed)
            or not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12)
            for observed, expected in observed_and_expected
        ):
            raise ValueError("Phase 12 allocation ratios do not derive from bytes")
        if (
            self.predicted_relative_error >= 0.01
            or abs(self.rho_alloc * self.r_alloc - 1.0) > 1e-9
            or self.r_hbm is not None
        ):
            raise ValueError("Phase 12 memory gate or r_hbm is invalid")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase12ConfigurationAdmission(StrictModel):
    method_config_id: str
    method_config_fingerprint: str
    prior_gates: tuple[Phase12PriorGateEvidence, ...]
    byte_accounting: Phase12ByteAccounting
    no_fallback: bool
    speedup_calculated: bool

    def __post_init__(self) -> None:
        if self.method_config_id not in PHASE12_MAIN_CONFIGURATIONS:
            raise ValueError("configuration is not in the Phase 12 main set")
        if (
            self.method_config_fingerprint
            != PHASE12_CONFIG_FINGERPRINTS[self.method_config_id]
        ):
            raise ValueError("Phase 12 configuration fingerprint differs")
        if tuple(item.gate for item in self.prior_gates) != PHASE12_PRIOR_GATES:
            raise ValueError("configuration must carry exactly ordered G1-G4 evidence")
        if not self.no_fallback or self.speedup_calculated:
            raise ValueError("Phase 12 configuration must remain fallback-free non-claim")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase12ExcludedConfiguration(StrictModel):
    method_config_id: str
    reason: Literal["validation_only_control"]

    def __post_init__(self) -> None:
        if self.method_config_id not in PHASE12_HELD_OUT_CONFIGURATIONS:
            raise ValueError("only frozen held-out controls may be excluded")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase12RandomizedOrder(StrictModel):
    replicate_index: int
    seed: int
    configurations: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.replicate_index) is not int or not (
            0 <= self.replicate_index < PHASE12_REPLICATES
        ):
            raise ValueError("Phase 12 replicate index is invalid")
        if (
            self.seed != PHASE12_RANDOMIZATION_SEEDS[self.replicate_index]
            or self.configurations
            != PHASE12_RANDOMIZED_ORDERS[self.replicate_index]
            or self.configurations
            != derive_phase12_randomized_order(self.seed)
        ):
            raise ValueError("Phase 12 randomized order differs from the frozen plan")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase12G5Run(StrictModel):
    run_id: str
    method_config_id: str
    replicate_index: int
    seed: int
    order_index: int
    manifest_path: str
    manifest_sha256: str
    process_median_ms: float
    output_checksum: str
    kernel_path_fingerprint: str
    allocation_fingerprint: str
    temperature_min_c: float
    temperature_max_c: float
    sm_clock_min_mhz: int
    sm_clock_max_mhz: int
    memory_clock_min_mhz: int
    memory_clock_max_mhz: int
    power_min_w: float
    power_max_w: float
    finite_output: bool
    no_backend_fallback: bool
    allocation_stable: bool
    kernel_path_stable: bool
    gpu_exclusive: bool
    speedup_calculated: bool

    def __post_init__(self) -> None:
        require_run_id(self.run_id)
        require_relative_path(self.manifest_path, field_name="manifest_path")
        for value, field_name in (
            (self.manifest_sha256, "manifest_sha256"),
            (self.output_checksum, "output_checksum"),
            (self.kernel_path_fingerprint, "kernel_path_fingerprint"),
            (self.allocation_fingerprint, "allocation_fingerprint"),
        ):
            require_sha256(value, field_name=field_name)
        if self.method_config_id not in PHASE12_MAIN_CONFIGURATIONS:
            raise ValueError("G5 run configuration is not in the main set")
        if type(self.replicate_index) is not int or not (
            0 <= self.replicate_index < PHASE12_REPLICATES
        ):
            raise ValueError("G5 replicate index is invalid")
        expected_order = derive_phase12_randomized_order(self.seed)
        if (
            self.seed != PHASE12_RANDOMIZATION_SEEDS[self.replicate_index]
            or type(self.order_index) is not int
            or not (0 <= self.order_index < len(expected_order))
            or expected_order[self.order_index] != self.method_config_id
        ):
            raise ValueError("G5 run does not match its frozen randomized order")
        floating_values = (
            self.process_median_ms,
            self.temperature_min_c,
            self.temperature_max_c,
            self.power_min_w,
            self.power_max_w,
        )
        if (
            any(type(value) is not float or not math.isfinite(value) for value in floating_values)
            or self.process_median_ms <= 0.0
            or self.temperature_min_c > self.temperature_max_c
            or self.power_min_w < 0.0
            or self.power_min_w > self.power_max_w
        ):
            raise ValueError("G5 timing or telemetry values are invalid")
        clock_values = (
            self.sm_clock_min_mhz,
            self.sm_clock_max_mhz,
            self.memory_clock_min_mhz,
            self.memory_clock_max_mhz,
        )
        if (
            any(type(value) is not int or value <= 0 for value in clock_values)
            or self.sm_clock_min_mhz > self.sm_clock_max_mhz
            or self.memory_clock_min_mhz > self.memory_clock_max_mhz
        ):
            raise ValueError("G5 clock ranges are invalid")
        if (
            not self.finite_output
            or not self.no_backend_fallback
            or not self.allocation_stable
            or not self.kernel_path_stable
            or not self.gpu_exclusive
            or self.speedup_calculated
        ):
            raise ValueError("G5 run violates the frozen stability or non-claim contract")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase12G5Statistics(StrictModel):
    method_config_id: str
    run_ids: tuple[str, ...]
    process_medians_ms: tuple[float, ...]
    median_ms: float
    minimum_ms: float
    maximum_ms: float
    mean_ms: float
    standard_deviation_ms: float
    coefficient_of_variation: float
    temperature_min_c: float
    temperature_max_c: float
    sm_clock_min_mhz: int
    sm_clock_max_mhz: int
    memory_clock_min_mhz: int
    memory_clock_max_mhz: int
    power_min_w: float
    power_max_w: float
    output_checksum_agreement: bool
    kernel_path_agreement: bool
    allocation_agreement: bool
    disposition: Phase12G5Disposition

    def __post_init__(self) -> None:
        if self.method_config_id not in PHASE12_MAIN_CONFIGURATIONS:
            raise ValueError("G5 statistics configuration is not in the main set")
        if len(self.run_ids) != PHASE12_REPLICATES or len(set(self.run_ids)) != 3:
            raise ValueError("G5 statistics require exactly three unique runs")
        for run_id in self.run_ids:
            require_run_id(run_id)
        if len(self.process_medians_ms) != PHASE12_REPLICATES:
            raise ValueError("G5 statistics require exactly three process medians")
        numeric_values = (
            *self.process_medians_ms,
            self.median_ms,
            self.minimum_ms,
            self.maximum_ms,
            self.mean_ms,
            self.standard_deviation_ms,
            self.coefficient_of_variation,
            self.temperature_min_c,
            self.temperature_max_c,
            self.power_min_w,
            self.power_max_w,
        )
        if any(type(value) is not float or not math.isfinite(value) for value in numeric_values):
            raise ValueError("G5 statistics must be finite JSON floats")
        if (
            any(value <= 0.0 for value in self.process_medians_ms)
            or self.minimum_ms <= 0.0
            or self.standard_deviation_ms < 0.0
            or self.coefficient_of_variation < 0.0
        ):
            raise ValueError("G5 timing statistics are invalid")
        expected_values = (
            (self.median_ms, float(statistics.median(self.process_medians_ms))),
            (self.minimum_ms, min(self.process_medians_ms)),
            (self.maximum_ms, max(self.process_medians_ms)),
            (self.mean_ms, statistics.mean(self.process_medians_ms)),
            (
                self.standard_deviation_ms,
                statistics.stdev(self.process_medians_ms),
            ),
            (
                self.coefficient_of_variation,
                statistics.stdev(self.process_medians_ms)
                / statistics.mean(self.process_medians_ms),
            ),
        )
        if any(
            not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12)
            for observed, expected in expected_values
        ):
            raise ValueError("G5 statistics must derive from the process medians")
        if (
            self.temperature_min_c > self.temperature_max_c
            or self.sm_clock_min_mhz <= 0
            or self.sm_clock_min_mhz > self.sm_clock_max_mhz
            or self.memory_clock_min_mhz <= 0
            or self.memory_clock_min_mhz > self.memory_clock_max_mhz
            or self.power_min_w < 0.0
            or self.power_min_w > self.power_max_w
        ):
            raise ValueError("G5 aggregate telemetry ranges are invalid")
        agreements = (
            self.output_checksum_agreement,
            self.kernel_path_agreement,
            self.allocation_agreement,
        )
        expected_disposition = (
            Phase12G5Disposition.PASS
            if self.coefficient_of_variation <= PHASE12_CV_THRESHOLD
            and all(agreements)
            else Phase12G5Disposition.UNSTABLE
        )
        if self.disposition is not expected_disposition:
            raise ValueError("G5 disposition does not follow the frozen threshold")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase12GlobalGates(StrictModel):
    g0: GateDisposition
    g1: GateDisposition
    g2: GateDisposition
    g3: GateDisposition
    g4: GateDisposition
    g5: GateDisposition
    pilot_state: Literal["READY", "NOT_READY"]
    full_scan_state: Literal["CLOSED"]
    quality_execution: QualityExecutionState
    performance_data_frozen: bool

    def __post_init__(self) -> None:
        all_pass = all(
            gate is GateDisposition.PASS
            for gate in (self.g0, self.g1, self.g2, self.g3, self.g4, self.g5)
        )
        if (
            self.g0 is not GateDisposition.PASS
            or (self.pilot_state == "READY") != all_pass
            or self.full_scan_state != "CLOSED"
            or self.quality_execution is not QualityExecutionState.LOCKED
            or self.performance_data_frozen
        ):
            raise ValueError("Phase 12 governance state is invalid")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase12UnifiedAdmissionReport(StrictModel):
    schema_version: str
    created_at_utc: str
    campaign_id: str
    execution_git_sha: str
    authorized_container_digest: str
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    runner_kind: Literal["fixed_l"]
    graph_mode: Literal["cuda_graph"]
    batch_size: int
    context_length: int
    warmup_steps: int
    measured_steps: int
    measured_batches: int
    independent_process_replicates: int
    cv_threshold: float
    configurations: tuple[Phase12ConfigurationAdmission, ...]
    excluded_configurations: tuple[Phase12ExcludedConfiguration, ...]
    randomized_orders: tuple[Phase12RandomizedOrder, ...]
    runs: tuple[Phase12G5Run, ...]
    g5_statistics: tuple[Phase12G5Statistics, ...]
    publication_state: Phase12PublicationState
    publication_receipt: Phase12EvidenceReference | None
    published_root_sha256: str | None
    r2_uri: str | None
    object_count: int | None
    complete_last: bool
    clean_retrieval: bool
    gates: Phase12GlobalGates
    speedup_calculated: bool

    SCHEMA_VERSION: ClassVar[str] = "kvbench-phase12-unified-admission-1.0.0"

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        require_utc_timestamp(self.created_at_utc)
        require_identifier(self.campaign_id, field_name="campaign_id")
        require_git_sha(self.execution_git_sha)
        require_oci_digest(self.authorized_container_digest)
        if (
            self.authorized_container_digest
            != PHASE12_AUTHORIZED_CONTAINER_DIGEST
            or self.model_id != PHASE12_MODEL_ID
            or self.model_revision != PHASE12_MODEL_REVISION
            or self.tokenizer_id != PHASE12_TOKENIZER_ID
            or self.tokenizer_revision != PHASE12_TOKENIZER_REVISION
            or self.runner_kind != PHASE12_RUNNER_KIND
            or self.graph_mode != PHASE12_GRAPH_MODE
            or self.batch_size != PHASE12_BATCH_SIZE
            or self.context_length != PHASE12_CONTEXT_LENGTH
            or self.warmup_steps != PHASE12_WARMUP_STEPS
            or self.measured_steps != PHASE12_MEASURED_STEPS
            or self.measured_batches != PHASE12_MEASURED_BATCHES
            or self.independent_process_replicates != PHASE12_REPLICATES
            or type(self.cv_threshold) is not float
            or self.cv_threshold != PHASE12_CV_THRESHOLD
            or self.speedup_calculated
        ):
            raise ValueError("Phase 12 common point or non-claim state differs")
        configuration_ids = tuple(
            item.method_config_id for item in self.configurations
        )
        if configuration_ids != PHASE12_MAIN_CONFIGURATIONS:
            raise ValueError("Phase 12 requires the exact ordered 10-config main set")
        excluded_ids = tuple(
            item.method_config_id for item in self.excluded_configurations
        )
        if excluded_ids != PHASE12_HELD_OUT_CONFIGURATIONS:
            raise ValueError("Phase 12 held-out controls differ")
        observed_orders = tuple(
            (item.replicate_index, item.seed, item.configurations)
            for item in self.randomized_orders
        )
        expected_orders = tuple(
            (index, seed, PHASE12_RANDOMIZED_ORDERS[index])
            for index, seed in enumerate(PHASE12_RANDOMIZATION_SEEDS)
        )
        if observed_orders != expected_orders:
            raise ValueError("Phase 12 requires exactly three frozen process orders")
        expected_run_sequence = tuple(
            (replicate_index, seed, order_index, method_config_id)
            for replicate_index, (seed, order) in enumerate(
                zip(
                    PHASE12_RANDOMIZATION_SEEDS,
                    PHASE12_RANDOMIZED_ORDERS,
                    strict=True,
                )
            )
            for order_index, method_config_id in enumerate(order)
        )
        observed_run_sequence = tuple(
            (
                item.replicate_index,
                item.seed,
                item.order_index,
                item.method_config_id,
            )
            for item in self.runs
        )
        if (
            observed_run_sequence != expected_run_sequence
            or len({item.run_id for item in self.runs}) != len(self.runs)
        ):
            raise ValueError("Phase 12 requires exactly the ordered 30-run matrix")
        statistics_ids = tuple(
            item.method_config_id for item in self.g5_statistics
        )
        if statistics_ids != PHASE12_MAIN_CONFIGURATIONS:
            raise ValueError("Phase 12 requires one G5 summary per main configuration")
        for summary in self.g5_statistics:
            matching_runs = tuple(
                item for item in self.runs
                if item.method_config_id == summary.method_config_id
            )
            if summary.run_ids != tuple(item.run_id for item in matching_runs):
                raise ValueError("G5 statistics do not join to their exact runs")
            if summary.process_medians_ms != tuple(
                item.process_median_ms for item in matching_runs
            ):
                raise ValueError("G5 statistics do not join to process medians")
            if (
                summary.output_checksum_agreement
                != (len({item.output_checksum for item in matching_runs}) == 1)
                or summary.kernel_path_agreement
                != (
                    len(
                        {
                            item.kernel_path_fingerprint
                            for item in matching_runs
                        }
                    )
                    == 1
                )
                or summary.allocation_agreement
                != (
                    len(
                        {
                            item.allocation_fingerprint
                            for item in matching_runs
                        }
                    )
                    == 1
                )
            ):
                raise ValueError("G5 agreement flags do not derive from exact runs")
        prior_gate_statuses = {
            gate: tuple(
                next(item for item in record.prior_gates if item.gate == gate).status
                for record in self.configurations
            )
            for gate in PHASE12_PRIOR_GATES
        }
        for gate_name, global_status in (
            ("G1", self.gates.g1),
            ("G2", self.gates.g2),
            ("G3", self.gates.g3),
            ("G4", self.gates.g4),
        ):
            all_configurations_pass = all(
                status is GateDisposition.PASS
                for status in prior_gate_statuses[gate_name]
            )
            if (global_status is GateDisposition.PASS) != all_configurations_pass:
                raise ValueError(
                    f"global {gate_name} must require every configuration; "
                    "majority voting is forbidden"
                )
        all_g5_pass = all(
            item.disposition is Phase12G5Disposition.PASS
            for item in self.g5_statistics
        )
        if self.publication_state is Phase12PublicationState.PENDING:
            if (
                self.publication_receipt is not None
                or self.published_root_sha256 is not None
                or self.r2_uri is not None
                or self.object_count is not None
                or self.complete_last
                or self.clean_retrieval
            ):
                raise ValueError("pending Phase 12 publication must carry no result")
            expected_g5 = (
                GateDisposition.NOT_EVALUATED
                if all_g5_pass
                else GateDisposition.FAIL
            )
        else:
            if (
                self.publication_receipt is None
                or self.published_root_sha256 is None
                or self.r2_uri
                != (
                    "r2://kvbench-artifacts/kvbench/sha256/"
                    f"{self.published_root_sha256}/"
                )
                or type(self.object_count) is not int
                or self.object_count <= 0
                or not self.complete_last
                or not self.clean_retrieval
            ):
                raise ValueError("published Phase 12 evidence is incomplete")
            require_sha256(
                self.published_root_sha256,
                field_name="published_root_sha256",
            )
            expected_g5 = (
                GateDisposition.PASS
                if all_g5_pass
                else GateDisposition.FAIL
            )
        if self.gates.g5 is not expected_g5:
            raise ValueError(
                "global G5 requires every configuration and durable publication"
            )
