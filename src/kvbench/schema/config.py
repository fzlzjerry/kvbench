"""Strict configuration and identity schemas for Phase 2."""

from __future__ import annotations

import dataclasses
from enum import StrEnum
from typing import Any, ClassVar, Literal

from kvbench.schema.base import (
    CANONICALIZATION_VERSION,
    ClaimClass,
    ClaimEligibility,
    GraphMode,
    QualityStatus,
    QualityValidationState,
    Resolution,
    ResolutionState,
    RunKind,
    RunnerKind,
    StrictModel,
    canonical_json_bytes,
    require_git_sha,
    require_identifier,
    require_oci_digest,
    require_relative_path,
    require_schema,
    require_sha256,
    sha256_hex,
)
from kvbench.schema.phase9_config import KVQuantCalibrationReference


class DocumentType(StrEnum):
    HARDWARE = "hardware"
    MODEL = "model"
    METHOD = "method"
    EXPERIMENT = "experiment"


class MethodName(StrEnum):
    BF16 = "bf16"
    TURBOQUANT = "turboquant"
    KIVI = "kivi"
    KVQUANT = "kvquant"


class PlanKind(StrEnum):
    SMOKE = "smoke"
    PILOT = "pilot"
    GRAPH_AB = "graph_ab"
    PROFILER_SUBSET = "profiler_subset"
    FULL_SCAN = "full_scan"


class VariantRole(StrEnum):
    BASELINE = "baseline"
    MAIN = "main"
    HELD_OUT = "held_out"


class ExecutionKind(StrEnum):
    NATIVE_HOST = "native_host"
    CONTAINER = "container"


class GraphSupport(StrEnum):
    NOT_EVALUATED = "not_evaluated"
    EAGER_ONLY = "eager_only"
    CUDA_GRAPH = "cuda_graph"


@dataclasses.dataclass(frozen=True, slots=True)
class ConfigReference(StrictModel):
    path: str
    sha256: str | None

    def __post_init__(self) -> None:
        require_relative_path(self.path)
        if self.sha256 is not None:
            require_sha256(self.sha256)


@dataclasses.dataclass(frozen=True, slots=True)
class HardwareManifest(StrictModel):
    schema_version: str
    document_type: DocumentType
    hardware_id: str
    resolution: Resolution
    expected_gpu_family: str
    expected_gpu_count: int
    tensor_parallel_size: int
    exclusive_gpu: bool
    max_memory_fraction: float
    require_stable_clocks: bool
    record_full_sku: bool
    e00_run_id: str
    e00_manifest_path: str
    e00_manifest_sha256: str
    g0_status: str
    container_parity_status: str
    container_parity_blocker: str | None

    SCHEMA_VERSION: ClassVar[str] = "kvbench.hardware.v1"

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        if self.document_type is not DocumentType.HARDWARE:
            raise ValueError("document_type must be hardware")
        require_identifier(self.hardware_id, field_name="hardware_id")
        require_identifier(self.e00_run_id, field_name="e00_run_id")
        require_relative_path(self.e00_manifest_path, field_name="e00_manifest_path")
        require_sha256(self.e00_manifest_sha256, field_name="e00_manifest_sha256")
        if self.expected_gpu_count != 1 or self.tensor_parallel_size != 1:
            raise ValueError("Phase 2 hardware contract requires one GPU and TP=1")
        if not 0.0 < self.max_memory_fraction <= 1.0:
            raise ValueError("max_memory_fraction must be in (0, 1]")
        if not self.expected_gpu_family.strip():
            raise ValueError("expected_gpu_family must be non-empty")
        if self.resolution.status is not ResolutionState.RESOLVED:
            raise ValueError("certified hardware identity must be resolved")
        if self.g0_status != "PASS":
            raise ValueError("native-host g0_status must be PASS")
        if self.container_parity_status != "not_evaluated":
            raise ValueError("container parity must remain not_evaluated")
        if self.container_parity_blocker != "B-010":
            raise ValueError("container parity must retain blocker B-010")


@dataclasses.dataclass(frozen=True, slots=True)
class ModelGeometry(StrictModel):
    num_hidden_layers: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    max_context_length: int

    def __post_init__(self) -> None:
        values = (
            self.num_hidden_layers,
            self.num_query_heads,
            self.num_kv_heads,
            self.head_dim,
            self.max_context_length,
        )
        if any(value <= 0 for value in values):
            raise ValueError("all model geometry values must be positive")
        if self.num_query_heads <= self.num_kv_heads:
            raise ValueError("primary geometry must be GQA")
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")


@dataclasses.dataclass(frozen=True, slots=True)
class ModelIdentity(StrictModel):
    schema_version: str
    document_type: DocumentType
    model_config_id: str
    resolution: Resolution
    model_id: str | None
    revision: str | None
    config_sha256: str | None
    tokenizer_id: str | None
    tokenizer_revision: str | None
    tokenizer_config_sha256: str | None
    decoder_only: bool
    full_attention: bool
    gqa_required: bool
    min_parameters_billion: float
    max_parameters_billion: float
    target_context_length: int
    geometry: ModelGeometry | None

    SCHEMA_VERSION: ClassVar[str] = "kvbench.model.v1"

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        if self.document_type is not DocumentType.MODEL:
            raise ValueError("document_type must be model")
        require_identifier(self.model_config_id, field_name="model_config_id")
        if not 0.0 < self.min_parameters_billion <= self.max_parameters_billion:
            raise ValueError("parameter range must be positive and ordered")
        if self.target_context_length <= 0:
            raise ValueError("target_context_length must be positive")
        identities = (
            self.model_id,
            self.revision,
            self.config_sha256,
            self.tokenizer_id,
            self.tokenizer_revision,
            self.tokenizer_config_sha256,
        )
        if self.resolution.status is ResolutionState.RESOLVED:
            if any(value is None or not value for value in identities) or self.geometry is None:
                raise ValueError("resolved model identity requires all identity fields")
            require_git_sha(self.revision or "")
            require_git_sha(self.tokenizer_revision or "")
            require_sha256(self.config_sha256 or "")
            require_sha256(self.tokenizer_config_sha256 or "")
            if not (self.decoder_only and self.full_attention and self.gqa_required):
                raise ValueError("resolved primary model must satisfy architectural constraints")
        elif any(value is not None for value in identities) or self.geometry is not None:
            raise ValueError("unresolved model identity fields must be explicit nulls")


@dataclasses.dataclass(frozen=True, slots=True)
class SoftwareEnvironment(StrictModel):
    schema_version: str
    environment_id: str
    resolution: Resolution
    execution_kind: ExecutionKind
    container_image: str | None
    container_digest: str | None
    python_version: str | None
    cuda_runtime_version: str | None
    cuda_toolkit_version: str | None
    torch_version: str | None
    triton_version: str | None
    dependency_lock_sha256: str | None

    SCHEMA_VERSION: ClassVar[str] = "kvbench.software-environment.v1"

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        require_identifier(self.environment_id, field_name="environment_id")
        if self.dependency_lock_sha256 is not None:
            require_sha256(self.dependency_lock_sha256)
        if self.container_digest is not None:
            require_oci_digest(self.container_digest)
        if self.execution_kind is ExecutionKind.NATIVE_HOST and (
            self.container_image is not None or self.container_digest is not None
        ):
            raise ValueError("native-host environment cannot claim a container identity")
        if (self.container_image is None) != (self.container_digest is None):
            raise ValueError("container image and digest must be supplied together")
        versions = (
            self.python_version,
            self.cuda_runtime_version,
            self.cuda_toolkit_version,
            self.torch_version,
            self.triton_version,
            self.dependency_lock_sha256,
        )
        if self.resolution.status is ResolutionState.RESOLVED:
            if any(value is None or not value for value in versions):
                raise ValueError("resolved software environment requires every version and lock")
            if self.execution_kind is ExecutionKind.CONTAINER:
                if not self.container_image or not self.container_digest:
                    raise ValueError("resolved container requires image and digest")


@dataclasses.dataclass(frozen=True, slots=True)
class MethodImplementation(StrictModel):
    resolution: Resolution
    attention_backend: str | None
    cache_layout: str | None
    adapter_source_sha256: str | None
    kernel_binary_sha256: str | None
    graph_support: GraphSupport

    def __post_init__(self) -> None:
        if self.resolution.status is ResolutionState.RESOLVED:
            if not self.attention_backend or not self.cache_layout:
                raise ValueError("resolved method implementation requires backend and layout")
            if self.adapter_source_sha256 is None:
                raise ValueError("resolved method implementation requires adapter source hash")
            require_sha256(self.adapter_source_sha256)
            if self.kernel_binary_sha256 is not None:
                require_sha256(self.kernel_binary_sha256)
            if self.graph_support is GraphSupport.NOT_EVALUATED:
                raise ValueError("resolved implementation must evaluate graph support")
        else:
            for value in (self.adapter_source_sha256, self.kernel_binary_sha256):
                if value is not None:
                    require_sha256(value)


@dataclasses.dataclass(frozen=True, slots=True)
class BF16Parameters(StrictModel):
    parameter_type: Literal["bf16"]
    cache_dtype: Literal["bfloat16"]


@dataclasses.dataclass(frozen=True, slots=True)
class TurboQuantParameters(StrictModel):
    parameter_type: Literal["turboquant"]
    cache_dtype_name: str | None
    key_bits: int
    value_bits: int
    key_path: str
    norm_correction: bool
    skipped_layers: tuple[int, ...] | None
    block_size: int | None
    decode_split_count: int | None

    def __post_init__(self) -> None:
        if self.key_bits not in {3, 4, 8} or self.value_bits not in {3, 4}:
            raise ValueError("TurboQuant bitwidth is outside the preregistered set")
        if self.key_path not in {"mse", "fp8"}:
            raise ValueError("TurboQuant key_path is invalid")
        if self.cache_dtype_name is not None and not self.cache_dtype_name.strip():
            raise ValueError("cache_dtype_name must be null or non-empty")
        if self.skipped_layers is not None:
            if any(layer < 0 for layer in self.skipped_layers):
                raise ValueError("skipped_layers must be nonnegative")
            if len(set(self.skipped_layers)) != len(self.skipped_layers):
                raise ValueError("skipped_layers must be unique")
        if self.block_size is not None and self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.decode_split_count is not None and self.decode_split_count <= 0:
            raise ValueError("decode_split_count must be positive")


@dataclasses.dataclass(frozen=True, slots=True)
class KiviParameters(StrictModel):
    parameter_type: Literal["kivi"]
    k_bits: int
    v_bits: int
    group_size: int
    residual_length: int

    def __post_init__(self) -> None:
        if self.k_bits not in {2, 4} or self.v_bits not in {2, 4}:
            raise ValueError("KIVI bitwidth must be 2 or 4")
        if self.group_size != 32 or self.residual_length != 32:
            raise ValueError("KIVI group_size and residual_length must remain 32")


@dataclasses.dataclass(frozen=True, slots=True)
class KVQuantParameters(StrictModel):
    parameter_type: Literal["kvquant"]
    bits: int
    sink_tokens: int
    outlier_cap: int | None
    calibration_artifact_sha256: str | None
    sparse_index_dtype: str | None
    lut_scale_dtype: str | None

    def __post_init__(self) -> None:
        if self.bits not in {2, 3, 4}:
            raise ValueError("KVQuant bits must be 2, 3, or 4")
        if self.sink_tokens != 5:
            raise ValueError("KVQuant sink_tokens must match the preregistered value 5")
        if self.outlier_cap is not None and self.outlier_cap <= 0:
            raise ValueError("outlier_cap must be positive")
        if self.calibration_artifact_sha256 is not None:
            require_sha256(self.calibration_artifact_sha256)


MethodParameters = (
    BF16Parameters | TurboQuantParameters | KiviParameters | KVQuantParameters
)


@dataclasses.dataclass(frozen=True, slots=True)
class MethodVariant(StrictModel):
    variant_id: str
    role: VariantRole
    resolution: Resolution
    parameters: MethodParameters

    def __post_init__(self) -> None:
        require_identifier(self.variant_id, field_name="variant_id")
        if self.resolution.status is ResolutionState.RESOLVED:
            if isinstance(self.parameters, TurboQuantParameters):
                required = (
                    self.parameters.cache_dtype_name,
                    self.parameters.skipped_layers,
                    self.parameters.block_size,
                    self.parameters.decode_split_count,
                )
                if any(value is None for value in required):
                    raise ValueError(
                        "resolved TurboQuant variant requires every runtime parameter"
                    )
            if isinstance(self.parameters, KVQuantParameters):
                if (
                    self.parameters.outlier_cap is None
                    or self.parameters.calibration_artifact_sha256 is None
                    or not self.parameters.sparse_index_dtype
                    or not self.parameters.lut_scale_dtype
                ):
                    raise ValueError(
                        "resolved KVQuant variant requires calibration and dtype parameters"
                    )


@dataclasses.dataclass(frozen=True, slots=True)
class MethodConfig(StrictModel):
    schema_version: str
    document_type: DocumentType
    method_config_id: str
    method: MethodName
    resolution: Resolution
    source_lock_id: str | None
    source_revision: str | None
    authority_status: str
    implementation: MethodImplementation
    variants: tuple[MethodVariant, ...]
    calibration: KVQuantCalibrationReference | None = None

    SCHEMA_VERSION: ClassVar[str] = "kvbench.method.v1"

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        if self.document_type is not DocumentType.METHOD:
            raise ValueError("document_type must be method")
        require_identifier(self.method_config_id, field_name="method_config_id")
        if not self.variants:
            raise ValueError("method config requires at least one variant")
        variant_ids = [variant.variant_id for variant in self.variants]
        if len(set(variant_ids)) != len(variant_ids):
            raise ValueError("variant IDs must be unique")
        if not self.authority_status.strip():
            raise ValueError("authority_status must be non-empty")
        if (
            self.resolution.status is ResolutionState.RESOLVED
            and "unresolved" in self.authority_status.lower()
        ):
            raise ValueError("resolved method cannot retain unresolved authority status")
        parameter_types = {
            MethodName.BF16: BF16Parameters,
            MethodName.TURBOQUANT: TurboQuantParameters,
            MethodName.KIVI: KiviParameters,
            MethodName.KVQUANT: KVQuantParameters,
        }
        if any(not isinstance(variant.parameters, parameter_types[self.method]) for variant in self.variants):
            raise ValueError("method and variant parameter types are incompatible")
        if self.method is MethodName.BF16:
            if self.source_lock_id is not None or self.source_revision is not None:
                raise ValueError("BF16 placeholder must not invent a source revision")
            if len(self.variants) != 1 or self.variants[0].variant_id != "bf16":
                raise ValueError("BF16 requires exactly the bf16 baseline variant")
            if self.variants[0].role is not VariantRole.BASELINE:
                raise ValueError("BF16 variant role must be baseline")
        else:
            if not self.source_lock_id or not self.source_revision:
                raise ValueError("quantized methods require the Phase 0 source lock identity")
            require_identifier(self.source_lock_id, field_name="source_lock_id")
            require_git_sha(self.source_revision)
        self._validate_registered_variants()
        self._validate_kvquant_calibration()

    def _validate_kvquant_calibration(self) -> None:
        if self.method is not MethodName.KVQUANT:
            if self.calibration is not None:
                raise ValueError(
                    "only KVQuant may carry a calibration reference"
                )
            return
        if self.calibration is None:
            if any(
                variant.resolution.status is ResolutionState.RESOLVED
                for variant in self.variants
            ):
                raise ValueError(
                    "resolved KVQuant calibration parameters require a reference"
                )
            return
        if self.source_revision != self.calibration.source_base_commit:
            raise ValueError("KVQuant source and calibration authority differ")
        checksums = {
            "kvq4": self.calibration.kvq4_sha256,
            "kvq3": self.calibration.kvq3_sha256,
            "kvq2": self.calibration.kvq2_sha256,
        }
        for variant in self.variants:
            parameters = variant.parameters
            if not isinstance(parameters, KVQuantParameters):
                raise ValueError("KVQuant variant parameters are invalid")
            if (
                parameters.sink_tokens != self.calibration.sink_tokens
                or parameters.outlier_cap != self.calibration.key_outlier_cap
                or parameters.calibration_artifact_sha256
                != checksums[variant.variant_id]
                or parameters.sparse_index_dtype
                != self.calibration.outlier_index_dtype
                or parameters.lut_scale_dtype
                != self.calibration.metadata_dtype
            ):
                raise ValueError("KVQuant variant calibration linkage drifted")

    def _validate_registered_variants(self) -> None:
        expected: dict[MethodName, dict[str, tuple[Any, ...]]] = {
            MethodName.TURBOQUANT: {
                "turboquant_4bit_nc": (4, 4, "mse", True, VariantRole.MAIN),
                "turboquant_k3v4_nc": (3, 4, "mse", True, VariantRole.MAIN),
                "turboquant_3bit_nc": (3, 3, "mse", True, VariantRole.MAIN),
                "turboquant_k8v4": (8, 4, "fp8", False, VariantRole.HELD_OUT),
            },
            MethodName.KIVI: {
                "k4v4": (4, 4, VariantRole.MAIN),
                "k2v4": (2, 4, VariantRole.MAIN),
                "k2v2": (2, 2, VariantRole.MAIN),
                "k4v2": (4, 2, VariantRole.HELD_OUT),
            },
            MethodName.KVQUANT: {
                "kvq4": (4, VariantRole.MAIN),
                "kvq3": (3, VariantRole.MAIN),
                "kvq2": (2, VariantRole.MAIN),
            },
        }
        if self.method is MethodName.BF16:
            return
        registered = expected[self.method]
        if set(variant.variant_id for variant in self.variants) != set(registered):
            raise ValueError("method variants must match the preregistered set")
        for variant in self.variants:
            wanted = registered[variant.variant_id]
            parameters = variant.parameters
            if isinstance(parameters, TurboQuantParameters):
                observed = (
                    parameters.key_bits,
                    parameters.value_bits,
                    parameters.key_path,
                    parameters.norm_correction,
                    variant.role,
                )
            elif isinstance(parameters, KiviParameters):
                observed = (parameters.k_bits, parameters.v_bits, variant.role)
            elif isinstance(parameters, KVQuantParameters):
                observed = (parameters.bits, variant.role)
            else:
                raise ValueError("unexpected method parameter type")
            if observed != wanted:
                raise ValueError("variant semantics do not match preregistration")

    def to_dict(self) -> dict[str, Any]:
        payload = StrictModel.to_dict(self)
        if self.calibration is None:
            payload.pop("calibration")
        return payload


@dataclasses.dataclass(frozen=True, slots=True)
class MethodConfigFingerprint(StrictModel):
    schema_version: str
    method: MethodName
    variant_id: str
    canonicalization: str
    algorithm: str
    sha256: str
    execution_ready: bool

    SCHEMA_VERSION: ClassVar[str] = "kvbench.method-fingerprint.v1"

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        require_identifier(self.variant_id, field_name="variant_id")
        if self.canonicalization != CANONICALIZATION_VERSION:
            raise ValueError("canonicalization version is unsupported")
        if self.algorithm != "sha256":
            raise ValueError("fingerprint algorithm must be sha256")
        require_sha256(self.sha256)

    @classmethod
    def from_config(
        cls, config: MethodConfig, variant_id: str
    ) -> "MethodConfigFingerprint":
        try:
            variant = next(item for item in config.variants if item.variant_id == variant_id)
        except StopIteration as error:
            raise ValueError("variant_id does not exist in method config") from error
        payload = {
            "method": config.method,
            "variant": variant,
            "implementation": config.implementation,
            "source_revision": config.source_revision,
        }
        if config.calibration is not None:
            payload["calibration"] = config.calibration
        ready = all(
            resolution.status is ResolutionState.RESOLVED
            for resolution in (
                config.resolution,
                variant.resolution,
                config.implementation.resolution,
            )
        )
        return cls(
            schema_version=cls.SCHEMA_VERSION,
            method=config.method,
            variant_id=variant_id,
            canonicalization=CANONICALIZATION_VERSION,
            algorithm="sha256",
            sha256=sha256_hex(canonical_json_bytes(payload)),
            execution_ready=ready,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class MethodSelection(StrictModel):
    config: ConfigReference
    variants: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.variants or len(set(self.variants)) != len(self.variants):
            raise ValueError("method selection variants must be non-empty and unique")
        for variant in self.variants:
            require_identifier(variant, field_name="variant")


@dataclasses.dataclass(frozen=True, slots=True)
class GridConfig(StrictModel):
    resolution: Resolution
    batch_sizes: tuple[int, ...]
    context_lengths: tuple[int, ...]
    output_tokens_for_request_validation: int | None

    def __post_init__(self) -> None:
        for name, values in (
            ("batch_sizes", self.batch_sizes),
            ("context_lengths", self.context_lengths),
        ):
            if any(value <= 0 for value in values) or len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique positive integers")
        if self.resolution.status is ResolutionState.RESOLVED and (
            not self.batch_sizes or not self.context_lengths
        ):
            raise ValueError("resolved grid axes must be non-empty")
        if (
            self.output_tokens_for_request_validation is not None
            and self.output_tokens_for_request_validation <= 0
        ):
            raise ValueError("request-validation output tokens must be positive")


@dataclasses.dataclass(frozen=True, slots=True)
class MeasurementConfig(StrictModel):
    resolution: Resolution
    warmup_steps: int | None
    measured_steps: int | None
    process_replicates: int | None
    randomize_within_method_block: bool | None
    rotate_method_block_order: bool | None
    seed: int | None

    def __post_init__(self) -> None:
        if self.warmup_steps is not None and self.warmup_steps < 0:
            raise ValueError("warmup_steps must be nonnegative")
        for name, value in (
            ("measured_steps", self.measured_steps),
            ("process_replicates", self.process_replicates),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if self.resolution.status is ResolutionState.RESOLVED and any(
            value is None
            for value in (
                self.warmup_steps,
                self.measured_steps,
                self.process_replicates,
                self.randomize_within_method_block,
                self.rotate_method_block_order,
                self.seed,
            )
        ):
            raise ValueError("resolved measurement protocol requires every field")


@dataclasses.dataclass(frozen=True, slots=True)
class AdmissionConfig(StrictModel):
    required_gates: tuple[str, ...]
    require_container_parity_g0: bool
    require_admission_pass: bool
    full_scan_state: str
    blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.required_gates or any(not gate.strip() for gate in self.required_gates):
            raise ValueError("required_gates must be non-empty")
        if len(set(self.required_gates)) != len(self.required_gates):
            raise ValueError("required_gates must be unique")
        if len(set(self.blockers)) != len(self.blockers):
            raise ValueError("admission blockers must be unique")
        if self.required_gates != ("G0", "G1", "G2", "G3", "G4", "G5"):
            raise ValueError("Phase 2 plans must retain ordered G0-G5 admission gates")
        if not self.require_container_parity_g0 or not self.require_admission_pass:
            raise ValueError("container parity and unified admission must be required")
        if self.full_scan_state != "closed":
            raise ValueError("full scan must remain closed in Phase 2")


@dataclasses.dataclass(frozen=True, slots=True)
class OutputConfig(StrictModel):
    format: str
    raw_samples: bool
    environment_manifest: bool
    telemetry: bool
    checksum: str
    overwrite: bool

    def __post_init__(self) -> None:
        if self.format != "parquet" or self.checksum != "sha256":
            raise ValueError("outputs require parquet and sha256")
        if not self.raw_samples or not self.environment_manifest or self.overwrite:
            raise ValueError("outputs require raw/environment data and forbid overwrite")


@dataclasses.dataclass(frozen=True, slots=True)
class ExperimentConfig(StrictModel):
    schema_version: str
    document_type: DocumentType
    plan_id: str
    plan_kind: PlanKind
    resolution: Resolution
    description: str
    hardware: ConfigReference
    model: ConfigReference
    methods: tuple[MethodSelection, ...]
    experiment_contract: ConfigReference
    measurement_protocol: ConfigReference
    software_environment: SoftwareEnvironment
    run_kinds: tuple[RunKind, ...]
    runner_kind: RunnerKind
    graph_modes: tuple[GraphMode, ...]
    claim_class: ClaimClass
    grid: GridConfig
    measurement: MeasurementConfig
    admission: AdmissionConfig
    outputs: OutputConfig
    quality: QualityStatus

    SCHEMA_VERSION: ClassVar[str] = "kvbench.experiment.v1"

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        if self.document_type is not DocumentType.EXPERIMENT:
            raise ValueError("document_type must be experiment")
        require_identifier(self.plan_id, field_name="plan_id")
        if not self.description.strip() or not self.methods:
            raise ValueError("plan description and method selections are required")
        paths = [selection.config.path for selection in self.methods]
        if len(set(paths)) != len(paths):
            raise ValueError("method config paths must be unique")
        if not self.run_kinds or len(set(self.run_kinds)) != len(self.run_kinds):
            raise ValueError("run_kinds must be non-empty and unique")
        if not self.graph_modes or len(set(self.graph_modes)) != len(self.graph_modes):
            raise ValueError("graph_modes must be non-empty and unique")
        if self.runner_kind is not RunnerKind.FIXED_L:
            raise ValueError("Phase 2 plan templates must retain fixed_l runner semantics")
        if (
            self.quality.quality_status is not QualityValidationState.UNVALIDATED
            or self.quality.claim_eligibility is not ClaimEligibility.PERFORMANCE_ONLY
        ):
            raise ValueError(
                "performance plans require unvalidated/performance_only quality metadata"
            )
        blockers = set(self.admission.blockers) | set(self.resolution.blockers)
        if not {"B-009", "B-010"}.issubset(blockers):
            raise ValueError("Phase 2 plans must retain blockers B-009 and B-010")
        if self.plan_kind is PlanKind.PROFILER_SUBSET:
            if set(self.run_kinds) != {RunKind.NSYS, RunKind.NCU}:
                raise ValueError("profiler subset must contain nsys and ncu run kinds")
            if self.claim_class is not ClaimClass.MECHANISM_ONLY:
                raise ValueError("profiler subset is mechanism_only")
            if self.graph_modes != (GraphMode.CUDA_GRAPH,):
                raise ValueError("profiler subset template must retain cuda_graph mode")
        elif any(kind in {RunKind.NSYS, RunKind.NCU} for kind in self.run_kinds):
            raise ValueError("profiler run kinds require profiler_subset plan")
        if self.plan_kind is PlanKind.GRAPH_AB:
            if set(self.graph_modes) != {GraphMode.EAGER, GraphMode.CUDA_GRAPH}:
                raise ValueError("graph A/B requires eager and cuda_graph")
            if self.run_kinds != (RunKind.TIMING,):
                raise ValueError("graph A/B run kind must be timing")
            if self.claim_class is not ClaimClass.MECHANISM_ONLY:
                raise ValueError("graph A/B is a mechanism-only experiment")
        if self.plan_kind in {PlanKind.SMOKE, PlanKind.PILOT, PlanKind.FULL_SCAN}:
            if self.run_kinds != (RunKind.TIMING,):
                raise ValueError("smoke/pilot/full-scan plans require timing run kind")
            if self.claim_class is not ClaimClass.SAME_WORK_LATENCY:
                raise ValueError("smoke/pilot/full-scan plans are same_work_latency")
        full_contexts = (
            4096,
            8192,
            16384,
            24576,
            32768,
            49152,
            65536,
            98304,
            131072,
        )
        if self.plan_kind is PlanKind.SMOKE:
            if self.graph_modes != (GraphMode.EAGER,):
                raise ValueError("smoke template must retain eager mode")
            if self.grid.batch_sizes != (1,) or self.grid.context_lengths != (4096,):
                raise ValueError("smoke plan must retain its validation-only singleton grid")
        if self.plan_kind is not PlanKind.PROFILER_SUBSET:
            if self.grid.output_tokens_for_request_validation != 256:
                raise ValueError("request-validation output length must remain 256")
        if self.plan_kind is PlanKind.PILOT:
            if self.graph_modes != (GraphMode.CUDA_GRAPH,):
                raise ValueError("pilot template must retain cuda_graph mode")
            if self.grid.batch_sizes != (1, 4, 8) or self.grid.context_lengths != full_contexts:
                raise ValueError("pilot grid does not match preregistration")
            values = (
                self.measurement.warmup_steps,
                self.measurement.measured_steps,
                self.measurement.process_replicates,
            )
            if values != (64, 128, 3):
                raise ValueError("pilot measurement counts do not match preregistration")
            if not (
                self.measurement.randomize_within_method_block
                and self.measurement.rotate_method_block_order
            ):
                raise ValueError("pilot randomization policy must remain enabled")
        if self.plan_kind is PlanKind.GRAPH_AB:
            graph_contexts = (4096, 16384, 24576, 32768, 65536, 131072)
            if self.grid.batch_sizes != (1, 4) or self.grid.context_lengths != graph_contexts:
                raise ValueError("graph A/B subset does not match preregistration")
        if self.plan_kind is PlanKind.FULL_SCAN:
            if self.graph_modes != (GraphMode.CUDA_GRAPH,):
                raise ValueError("full-scan template must retain cuda_graph mode")
            if self.resolution.status is not ResolutionState.BLOCKED:
                raise ValueError("full scan must remain blocked in Phase 2")
            if self.admission.full_scan_state != "closed":
                raise ValueError("full scan admission must remain closed")
            if self.grid.batch_sizes != (1, 2, 4, 8, 16) or self.grid.context_lengths != full_contexts:
                raise ValueError("full-scan grid does not match preregistration")
            values = (
                self.measurement.warmup_steps,
                self.measurement.measured_steps,
                self.measurement.process_replicates,
            )
            if values != (64, 256, 5):
                raise ValueError("full-scan measurement counts do not match preregistration")
            if not (
                self.measurement.randomize_within_method_block
                and self.measurement.rotate_method_block_order
            ):
                raise ValueError("full-scan randomization policy must remain enabled")
