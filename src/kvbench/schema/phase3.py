"""Strict Phase 3 model, plan, admission, and run schemas."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from datetime import datetime
from enum import StrEnum
import math
from pathlib import PurePosixPath
import statistics
from typing import ClassVar, Literal, TypeAlias

from kvbench.errors import SchemaValidationError
from kvbench.schema.base import (
    ClaimClass,
    ClaimEligibility,
    GraphMode,
    MeasurementScope,
    QualityExecutionState,
    QualityStatus,
    QualityValidationState,
    Resolution,
    ResolutionState,
    RunKind,
    RunnerKind,
    RunStatus,
    StrictModel,
    canonical_json_bytes,
    require_git_sha,
    require_identifier,
    require_relative_path,
    require_run_id,
    require_schema,
    require_sha256,
    require_utc_timestamp,
    sha256_hex,
)
from kvbench.schema.config import (
    ConfigReference,
    DocumentType,
    ExecutionKind,
    HardwareManifest,
    MethodConfig,
    MethodConfigFingerprint,
    MethodName,
    MethodSelection,
    ModelGeometry,
    SoftwareEnvironment,
)
from kvbench.schema.result import ConfigSource, ConfigSourceKind, RunManifest


PRIMARY_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
PRIMARY_MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
PRIMARY_MODEL_SNAPSHOT = (
    "/root/.cache/huggingface/hub/"
    "models--meta-llama--Llama-3.1-8B-Instruct/snapshots/"
    "0e9e39f249a16976918f6564b8830bc894c89659"
)
PHASE3_RANDOM_SEED = 20260722
PHASE3_REPOSITORY_ROOT = "/home/rockrock/cmu_paper"
PHASE3_PYTHON_EXECUTABLE = f"{PHASE3_REPOSITORY_ROOT}/.venv/bin/python"
PHASE3_FIXED_PLAN_PATH = "configs/plans/phase3_bf16_fixed_l.yaml"
PHASE3_GROWING_PLAN_PATH = "configs/plans/phase3_bf16_growing.yaml"
PHASE3_PLAN_FINGERPRINTS: dict[str, str] = {
    PHASE3_FIXED_PLAN_PATH: (
        "d8f2b4e61f6569d5b8cb75b84bbf36a3b60927b0575d514c2d9bd0aac7da6a2d"
    ),
    PHASE3_GROWING_PLAN_PATH: (
        "4598647f5ba04deff187d11346c0695b857464d06729b96b46b838080d80cd63"
    ),
}
PHASE3_HARDWARE_ID = "rtx_pro_6000_blackwell_96gb"
PHASE3_HARDWARE_FINGERPRINT = (
    "b4531c256a28c0110766f8f725b90d1954fa3647d5f89b8f721a91cf5141c4e6"
)
PHASE3_MODEL_FINGERPRINT = (
    "9f37a5b8acc1a19d390f000280a1f24e18c26f56c0415acb09ba40b87f59143f"
)
PHASE3_BF16_CONFIG_FINGERPRINT = (
    "49f69c7b59463f25fb994042c08a7b170519ef8fe2a3b22292e55a6ace7649ae"
)
PHASE3_BF16_VARIANT_FINGERPRINT = (
    "81ca6a0d74727a9c8a54f14d6d222dee502ee5e9a9ab6e8c9a03b4fed98f371b"
)
PHASE3_CONTRACT_FINGERPRINT = (
    "eac6737a637cbc3d6b4b67bc7455e950b50bcb4ee922621b8dcba3051938d7fd"
)
PHASE3_MEASUREMENT_PROTOCOL_FINGERPRINT = (
    "fc4208706327655fc503ff09898229b3a3a2e2bbdc0e833961d7a64e2ba03f7a"
)
PHASE3_SOFTWARE_ENVIRONMENT_ID = "phase3_native_host"
PHASE3_SOFTWARE_FINGERPRINT = (
    "bc9c80780d3a68b163a5869de7bbc5862915a883858f9b9d5666a4e305f8c728"
)
PHASE3_E00_RUN_ID = "e00-20260722T050632.375718Z-6442ba1f7554-02d5bd32"
PHASE3_E00_MANIFEST_SHA256 = (
    "d054df714bb5eea1f114bf10a03a2879f56ec8d17d3b07e24fe6efcaba6b7aca"
)
PHASE3_GPU_UUID = "GPU-75bd273e-6b20-0d22-1b0b-5fbb6fb0025b"
PHASE3_GPU_FULL_NAME = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
PHASE3_PCI_BUS_ID = "00000000:05:00.0"
PHASE3_PCI_DEVICE_ID = "0x2BB110DE"
PHASE3_DRIVER_VERSION = "595.71.05"


class ModelArtifactRole(StrEnum):
    LICENSE = "license"
    MODEL_CONFIG = "model_config"
    MODEL_WEIGHTS = "model_weights"
    TOKENIZER = "tokenizer"


class MeasurementCountUnit(StrEnum):
    DECODE_OPERATIONS = "decode_operations"
    TRAJECTORIES = "trajectories"


class GateDisposition(StrEnum):
    PASS = "PASS"
    PARTIAL = "PARTIAL"
    BLOCKED = "BLOCKED"
    FAIL = "FAIL"
    NOT_EVALUATED = "NOT_EVALUATED"


@dataclasses.dataclass(frozen=True, slots=True)
class ArtifactDigest(StrictModel):
    path: str
    role: ModelArtifactRole
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        require_relative_path(self.path, field_name="artifact path")
        if self.size_bytes <= 0:
            raise ValueError("artifact size must be positive")
        require_sha256(self.sha256)


@dataclasses.dataclass(frozen=True, slots=True)
class SourceDigest(StrictModel):
    path: str
    sha256: str

    def __post_init__(self) -> None:
        require_relative_path(self.path, field_name="source path")
        require_sha256(self.sha256)


@dataclasses.dataclass(frozen=True, slots=True)
class RopeIdentity(StrictModel):
    rope_type: str
    factor: float
    high_frequency_factor: float
    low_frequency_factor: float
    original_max_position_embeddings: int
    theta: float

    def __post_init__(self) -> None:
        expected = ("llama3", 8.0, 4.0, 1.0, 8192, 500000.0)
        observed = (
            self.rope_type,
            self.factor,
            self.high_frequency_factor,
            self.low_frequency_factor,
            self.original_max_position_embeddings,
            self.theta,
        )
        if observed != expected:
            raise ValueError("RoPE identity does not match the frozen checkpoint")


_EXPECTED_MODEL_ARTIFACTS: dict[
    str, tuple[ModelArtifactRole, int, str]
] = {
    "LICENSE": (
        ModelArtifactRole.LICENSE,
        7627,
        "64e1b2889b7892e6bbe7a7ed5bfe6ff793c61f9d584345f8f41cf9f5cb30a369",
    ),
    "config.json": (
        ModelArtifactRole.MODEL_CONFIG,
        855,
        "29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e",
    ),
    "generation_config.json": (
        ModelArtifactRole.MODEL_CONFIG,
        184,
        "189fb0c0d7fd8a527db217c0a60a0e013f0394cd8800f9697a666a9e75e5f7fd",
    ),
    "model-00001-of-00004.safetensors": (
        ModelArtifactRole.MODEL_WEIGHTS,
        4_976_698_672,
        "2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668",
    ),
    "model-00002-of-00004.safetensors": (
        ModelArtifactRole.MODEL_WEIGHTS,
        4_999_802_720,
        "09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15",
    ),
    "model-00003-of-00004.safetensors": (
        ModelArtifactRole.MODEL_WEIGHTS,
        4_915_916_176,
        "fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa",
    ),
    "model-00004-of-00004.safetensors": (
        ModelArtifactRole.MODEL_WEIGHTS,
        1_168_138_808,
        "92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b",
    ),
    "model.safetensors.index.json": (
        ModelArtifactRole.MODEL_CONFIG,
        23_950,
        "146776fce3f6db1103aa6f249e65ee5544c5923ce6f971b092eee79aa6e5d37b",
    ),
    "special_tokens_map.json": (
        ModelArtifactRole.TOKENIZER,
        296,
        "6f38c73729248f6c127296386e3cdde96e254636cc58b4169d3fd32328d9a8ec",
    ),
    "tokenizer.json": (
        ModelArtifactRole.TOKENIZER,
        9_085_657,
        "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4",
    ),
    "tokenizer_config.json": (
        ModelArtifactRole.TOKENIZER,
        55_351,
        "177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424",
    ),
}


@dataclasses.dataclass(frozen=True, slots=True)
class ModelIdentityV2(StrictModel):
    schema_version: str
    document_type: DocumentType
    model_config_id: str
    resolution: Resolution
    model_id: str
    revision: str
    config_sha256: str
    tokenizer_id: str
    tokenizer_revision: str
    tokenizer_config_sha256: str
    decoder_only: bool
    full_attention: bool
    gqa_required: bool
    min_parameters_billion: float
    max_parameters_billion: float
    parameter_count: int
    architecture: str
    hidden_size: int
    vocabulary_size: int
    target_context_length: int
    geometry: ModelGeometry
    weight_dtype: str
    tokenizer_class: str
    rope: RopeIdentity
    local_snapshot_path: str
    network_policy: str
    artifacts: tuple[ArtifactDigest, ...]
    license_name: str
    access_state: str
    same_checkpoint_reserved_for_quality_validation: bool

    SCHEMA_VERSION: ClassVar[str] = "kvbench.model.v2"

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        if self.document_type is not DocumentType.MODEL:
            raise ValueError("document_type must be model")
        require_identifier(self.model_config_id, field_name="model_config_id")
        if self.resolution.status is not ResolutionState.RESOLVED:
            raise ValueError("Phase 3 model identity must be resolved")
        if self.model_id != PRIMARY_MODEL_ID or self.tokenizer_id != PRIMARY_MODEL_ID:
            raise ValueError("model and tokenizer IDs must match the frozen checkpoint")
        if (
            self.revision != PRIMARY_MODEL_REVISION
            or self.tokenizer_revision != PRIMARY_MODEL_REVISION
        ):
            raise ValueError("model and tokenizer revisions must match the frozen revision")
        require_git_sha(self.revision)
        require_git_sha(self.tokenizer_revision)
        require_sha256(self.config_sha256, field_name="config_sha256")
        require_sha256(
            self.tokenizer_config_sha256,
            field_name="tokenizer_config_sha256",
        )
        if self.config_sha256 != _EXPECTED_MODEL_ARTIFACTS["config.json"][2]:
            raise ValueError("config hash does not match the frozen artifact")
        if (
            self.tokenizer_config_sha256
            != _EXPECTED_MODEL_ARTIFACTS["tokenizer_config.json"][2]
        ):
            raise ValueError("tokenizer config hash does not match the frozen artifact")
        if not (self.decoder_only and self.full_attention and self.gqa_required):
            raise ValueError("primary model must be decoder-only full-attention GQA")
        if (self.min_parameters_billion, self.max_parameters_billion) != (7.0, 9.0):
            raise ValueError("parameter range must remain 7B-9B")
        if self.target_context_length <= 0:
            raise ValueError("target_context_length must be positive")
        expected_scalar_identity = (
            8_030_261_248,
            "LlamaForCausalLM",
            4096,
            128256,
            131072,
            "bfloat16",
            "PreTrainedTokenizerFast",
        )
        observed_scalar_identity = (
            self.parameter_count,
            self.architecture,
            self.hidden_size,
            self.vocabulary_size,
            self.target_context_length,
            self.weight_dtype,
            self.tokenizer_class,
        )
        if observed_scalar_identity != expected_scalar_identity:
            raise ValueError("model scalar identity does not match the frozen checkpoint")
        expected_geometry = (32, 32, 8, 128, 131072)
        observed_geometry = (
            self.geometry.num_hidden_layers,
            self.geometry.num_query_heads,
            self.geometry.num_kv_heads,
            self.geometry.head_dim,
            self.geometry.max_context_length,
        )
        if observed_geometry != expected_geometry:
            raise ValueError("model geometry does not match the frozen checkpoint")
        snapshot = PurePosixPath(self.local_snapshot_path)
        if not snapshot.is_absolute() or ".." in snapshot.parts:
            raise ValueError("local snapshot path must be an absolute safe POSIX path")
        if self.local_snapshot_path != PRIMARY_MODEL_SNAPSHOT:
            raise ValueError("local snapshot path does not match the frozen cache identity")
        if self.network_policy != "offline_local_files_only":
            raise ValueError("Phase 3 model execution must be offline and local-only")
        paths = tuple(artifact.path for artifact in self.artifacts)
        if paths != tuple(sorted(_EXPECTED_MODEL_ARTIFACTS)):
            raise ValueError("model artifacts must be complete, unique, and sorted")
        for artifact in self.artifacts:
            if (
                artifact.role,
                artifact.size_bytes,
                artifact.sha256,
            ) != _EXPECTED_MODEL_ARTIFACTS[artifact.path]:
                raise ValueError("model artifact identity does not match frozen evidence")
        if self.license_name != "Llama 3.1 Community License":
            raise ValueError("license identity does not match the frozen checkpoint")
        if self.access_state != "manually_gated_access_verified":
            raise ValueError("model access state must be explicitly verified")
        if not self.same_checkpoint_reserved_for_quality_validation:
            raise ValueError("the exact checkpoint must be reserved for later quality validation")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase3GridConfig(StrictModel):
    batch_sizes: tuple[int, ...]
    context_lengths: tuple[int, ...]
    output_steps: int

    def __post_init__(self) -> None:
        if (
            not self.batch_sizes
            or not self.context_lengths
            or self.output_steps <= 0
        ):
            raise ValueError("Phase 3 grid axes and output steps must be positive")
        if (
            len(set(self.batch_sizes)) != len(self.batch_sizes)
            or len(set(self.context_lengths)) != len(self.context_lengths)
            or any(value <= 0 for value in (*self.batch_sizes, *self.context_lengths))
        ):
            raise ValueError("Phase 3 grid axes must contain unique positive integers")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase3MeasurementConfig(StrictModel):
    warmup_count: int
    measured_count: int
    measured_batches: int
    count_unit: MeasurementCountUnit
    ordinary_process_replicates: int
    seed: int

    def __post_init__(self) -> None:
        if min(
            self.warmup_count,
            self.measured_count,
            self.measured_batches,
            self.ordinary_process_replicates,
        ) <= 0:
            raise ValueError("Phase 3 measurement counts must be positive")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase3StabilityConfig(StrictModel):
    batch_size: int
    context_length: int
    graph_modes: tuple[GraphMode, ...]
    total_process_replicates: int
    maximum_cv_percent: float

    def __post_init__(self) -> None:
        expected = (
            1,
            4096,
            (GraphMode.EAGER, GraphMode.CUDA_GRAPH),
            3,
            3.0,
        )
        observed = (
            self.batch_size,
            self.context_length,
            self.graph_modes,
            self.total_process_replicates,
            self.maximum_cv_percent,
        )
        if observed != expected:
            raise ValueError("stability subset does not match Phase 3 preregistration")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase3AdmissionConfig(StrictModel):
    required_gates: tuple[str, ...]
    evaluated_gate: str
    native_g0_status: GateDisposition
    later_gates_status: GateDisposition
    full_scan_state: str
    retained_open_blockers: tuple[str, ...]
    require_native_g0: bool
    require_container_parity_g0: bool
    require_durable_artifact_store: bool

    def __post_init__(self) -> None:
        if self.required_gates != ("G0", "G1") or self.evaluated_gate != "G1":
            raise ValueError("Phase 3 evaluates only BF16 Baseline G1 while retaining G0")
        if (
            self.native_g0_status is not GateDisposition.PASS
            or self.later_gates_status is not GateDisposition.NOT_EVALUATED
        ):
            raise ValueError("Phase 3 gate dispositions are invalid")
        if self.full_scan_state != "closed":
            raise ValueError("Full Scan must remain closed")
        if self.retained_open_blockers != ("B-009", "B-010"):
            raise ValueError("Phase 3 must retain open blockers B-009 and B-010")
        if not self.require_native_g0:
            raise ValueError("certified native-host G0 is required")
        if self.require_container_parity_g0 or self.require_durable_artifact_store:
            raise ValueError("container parity and durable storage remain unresolved")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase3OutputConfig(StrictModel):
    artifact_root: str
    format: str
    raw_samples: bool
    environment_manifest: bool
    telemetry: bool
    checksum: str
    overwrite: bool

    def __post_init__(self) -> None:
        require_relative_path(self.artifact_root, field_name="artifact_root")
        if self.artifact_root != "artifacts/phase3" or self.format != "json":
            raise ValueError("Phase 3 admission output must be JSON under artifacts/phase3")
        if self.checksum != "sha256":
            raise ValueError("Phase 3 artifacts require SHA-256 checksums")
        if not self.raw_samples or not self.environment_manifest or not self.telemetry:
            raise ValueError("Phase 3 outputs require raw samples, environment, and telemetry")
        if self.overwrite:
            raise ValueError("Phase 3 artifacts are append-only")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase3AdmissionPlan(StrictModel):
    schema_version: str
    document_type: DocumentType
    plan_id: str
    plan_kind: Literal["phase3_admission"]
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
    measurement_scope: MeasurementScope
    performance_claim_eligible: bool
    context_length_convention: Literal["historical_prefix_tokens"]
    grid: Phase3GridConfig
    measurement: Phase3MeasurementConfig
    stability: Phase3StabilityConfig | None
    admission: Phase3AdmissionConfig
    outputs: Phase3OutputConfig
    quality: QualityStatus
    expected_process_count: int

    SCHEMA_VERSION: ClassVar[str] = "kvbench.phase3-admission-plan.v1"

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        if self.document_type is not DocumentType.EXPERIMENT:
            raise ValueError("document_type must be experiment")
        require_identifier(self.plan_id, field_name="plan_id")
        if self.resolution.status is not ResolutionState.RESOLVED:
            raise ValueError("Phase 3 admission semantics must be resolved")
        if not self.description.strip():
            raise ValueError("Phase 3 plan description must be non-empty")
        expected_references = (
            (
                self.hardware,
                "configs/hardware/rtx_pro_6000.yaml",
                PHASE3_HARDWARE_FINGERPRINT,
            ),
            (
                self.model,
                "configs/models/primary_gqa_model.yaml",
                PHASE3_MODEL_FINGERPRINT,
            ),
            (
                self.experiment_contract,
                "docs/experiment_contract.md",
                PHASE3_CONTRACT_FINGERPRINT,
            ),
            (
                self.measurement_protocol,
                "docs/measurement_protocol.md",
                PHASE3_MEASUREMENT_PROTOCOL_FINGERPRINT,
            ),
        )
        for reference, expected_path, expected_sha256 in expected_references:
            if (
                reference.path != expected_path
                or reference.sha256 != expected_sha256
            ):
                raise ValueError("Phase 3 references must use frozen paths and hashes")
        if len(self.methods) != 1:
            raise ValueError("Phase 3 admits exactly one BF16 method selection")
        selection = self.methods[0]
        if (
            selection.config.path != "configs/methods/bf16.yaml"
            or selection.config.sha256 != PHASE3_BF16_CONFIG_FINGERPRINT
            or selection.variants != ("bf16",)
        ):
            raise ValueError("Phase 3 admits only the exact BF16 baseline config")
        expected_software = (
            PHASE3_SOFTWARE_ENVIRONMENT_ID,
            ResolutionState.RESOLVED,
            ExecutionKind.NATIVE_HOST,
            None,
            None,
            "3.12.3",
            "13.0",
            "13.0.88",
            "2.12.1+cu130",
            "3.7.1",
            "cebe254a3e03a48e3e67100ce11d5623fc0dc722dc43e2f482152beb644a08e9",
        )
        observed_software = (
            self.software_environment.environment_id,
            self.software_environment.resolution.status,
            self.software_environment.execution_kind,
            self.software_environment.container_image,
            self.software_environment.container_digest,
            self.software_environment.python_version,
            self.software_environment.cuda_runtime_version,
            self.software_environment.cuda_toolkit_version,
            self.software_environment.torch_version,
            self.software_environment.triton_version,
            self.software_environment.dependency_lock_sha256,
        )
        if observed_software != expected_software:
            raise ValueError("Phase 3 software identity does not match the native host lock")
        if self.software_environment.fingerprint() != PHASE3_SOFTWARE_FINGERPRINT:
            raise ValueError("Phase 3 software fingerprint does not match the frozen lock")
        if self.measurement.seed != PHASE3_RANDOM_SEED:
            raise ValueError("Phase 3 random seed must remain 20260722")
        if self.run_kinds != (RunKind.PHASE3_ADMISSION,):
            raise ValueError("Phase 3 plan run kind must be phase3_admission")
        if self.claim_class is not ClaimClass.NONE:
            raise ValueError("Phase 3 admission evidence carries no claim class")
        if self.measurement_scope is not MeasurementScope.NATIVE_HOST_ADMISSION:
            raise ValueError("Phase 3 measurement scope must be native_host_admission")
        if self.performance_claim_eligible:
            raise ValueError("Phase 3 evidence cannot support a performance claim")
        if (
            self.quality.quality_status is not QualityValidationState.UNVALIDATED
            or self.quality.claim_eligibility is not ClaimEligibility.PERFORMANCE_ONLY
            or self.quality.quality_execution is not QualityExecutionState.LOCKED
            or self.quality.performance_data_frozen
        ):
            raise ValueError("Phase 3 quality governance must remain locked and unvalidated")

        if self.runner_kind is RunnerKind.FIXED_L:
            expected = (
                "phase3_bf16_fixed_l",
                (GraphMode.EAGER, GraphMode.CUDA_GRAPH),
                (1, 4),
                (128, 4096, 16384),
                1,
                16,
                32,
                5,
                MeasurementCountUnit.DECODE_OPERATIONS,
                1,
                16,
            )
            observed = (
                self.plan_id,
                self.graph_modes,
                self.grid.batch_sizes,
                self.grid.context_lengths,
                self.grid.output_steps,
                self.measurement.warmup_count,
                self.measurement.measured_count,
                self.measurement.measured_batches,
                self.measurement.count_unit,
                self.measurement.ordinary_process_replicates,
                self.expected_process_count,
            )
            if observed != expected or self.stability is None:
                raise ValueError("fixed-L admission grid does not match preregistration")
        elif self.runner_kind is RunnerKind.GROWING_CONTEXT:
            expected = (
                "phase3_bf16_growing",
                (GraphMode.EAGER,),
                (1, 4),
                (128, 4096),
                16,
                1,
                1,
                1,
                MeasurementCountUnit.TRAJECTORIES,
                1,
                4,
            )
            observed = (
                self.plan_id,
                self.graph_modes,
                self.grid.batch_sizes,
                self.grid.context_lengths,
                self.grid.output_steps,
                self.measurement.warmup_count,
                self.measurement.measured_count,
                self.measurement.measured_batches,
                self.measurement.count_unit,
                self.measurement.ordinary_process_replicates,
                self.expected_process_count,
            )
            if observed != expected or self.stability is not None:
                raise ValueError("growing-context admission grid does not match preregistration")
        else:
            raise ValueError("unsupported Phase 3 runner kind")


@dataclasses.dataclass(frozen=True, slots=True)
class _FrozenPhase3Point:
    plan_path: str
    plan_fingerprint: str
    runner_kind: RunnerKind
    graph_mode: GraphMode
    batch_size: int
    context_length: int
    output_steps: int
    warmup_count: int
    measured_count: int
    measured_batches: int
    count_unit: MeasurementCountUnit
    random_seed: int
    process_replicate: int
    capacity: int
    stability_member: bool

    @property
    def point_id(self) -> str:
        return (
            f"{self.runner_kind.value}-b{self.batch_size}-l{self.context_length}-"
            f"{self.graph_mode.value}-r{self.process_replicate}"
        )

    @property
    def fingerprint(self) -> str:
        return sha256_hex(
            canonical_json_bytes(
                {
                    "schema": "kvbench-phase3-process-point-1.0.0",
                    "point_id": self.point_id,
                    "plan_path": self.plan_path,
                    "plan_fingerprint": self.plan_fingerprint,
                    "runner_kind": self.runner_kind.value,
                    "graph_mode": self.graph_mode.value,
                    "batch_size": self.batch_size,
                    "context_length": self.context_length,
                    "output_steps": self.output_steps,
                    "warmup_count": self.warmup_count,
                    "measured_count": self.measured_count,
                    "measured_batches": self.measured_batches,
                    "count_unit": self.count_unit.value,
                    "random_seed": self.random_seed,
                    "process_replicate": self.process_replicate,
                    "capacity": self.capacity,
                }
            )
        )


def _build_frozen_phase3_points() -> tuple[_FrozenPhase3Point, ...]:
    points: list[_FrozenPhase3Point] = []
    for batch_size in (1, 4):
        for context_length in (128, 4096, 16384):
            for graph_mode in (GraphMode.EAGER, GraphMode.CUDA_GRAPH):
                points.append(
                    _FrozenPhase3Point(
                        plan_path=PHASE3_FIXED_PLAN_PATH,
                        plan_fingerprint=PHASE3_PLAN_FINGERPRINTS[
                            PHASE3_FIXED_PLAN_PATH
                        ],
                        runner_kind=RunnerKind.FIXED_L,
                        graph_mode=graph_mode,
                        batch_size=batch_size,
                        context_length=context_length,
                        output_steps=1,
                        warmup_count=16,
                        measured_count=32,
                        measured_batches=5,
                        count_unit=MeasurementCountUnit.DECODE_OPERATIONS,
                        random_seed=PHASE3_RANDOM_SEED,
                        process_replicate=1,
                        capacity=context_length + 1,
                        stability_member=batch_size == 1
                        and context_length == 4096,
                    )
                )
    for graph_mode in (GraphMode.EAGER, GraphMode.CUDA_GRAPH):
        for process_replicate in (2, 3):
            points.append(
                _FrozenPhase3Point(
                    plan_path=PHASE3_FIXED_PLAN_PATH,
                    plan_fingerprint=PHASE3_PLAN_FINGERPRINTS[
                        PHASE3_FIXED_PLAN_PATH
                    ],
                    runner_kind=RunnerKind.FIXED_L,
                    graph_mode=graph_mode,
                    batch_size=1,
                    context_length=4096,
                    output_steps=1,
                    warmup_count=16,
                    measured_count=32,
                    measured_batches=5,
                    count_unit=MeasurementCountUnit.DECODE_OPERATIONS,
                    random_seed=PHASE3_RANDOM_SEED,
                    process_replicate=process_replicate,
                    capacity=4097,
                    stability_member=True,
                )
            )
    for batch_size in (1, 4):
        for context_length in (128, 4096):
            points.append(
                _FrozenPhase3Point(
                    plan_path=PHASE3_GROWING_PLAN_PATH,
                    plan_fingerprint=PHASE3_PLAN_FINGERPRINTS[
                        PHASE3_GROWING_PLAN_PATH
                    ],
                    runner_kind=RunnerKind.GROWING_CONTEXT,
                    graph_mode=GraphMode.EAGER,
                    batch_size=batch_size,
                    context_length=context_length,
                    output_steps=16,
                    warmup_count=1,
                    measured_count=1,
                    measured_batches=1,
                    count_unit=MeasurementCountUnit.TRAJECTORIES,
                    random_seed=PHASE3_RANDOM_SEED,
                    process_replicate=1,
                    capacity=context_length + 16,
                    stability_member=False,
                )
            )
    return tuple(points)


_FROZEN_PHASE3_POINTS = _build_frozen_phase3_points()
_FROZEN_PHASE3_POINTS_BY_ID = {
    point.point_id: point for point in _FROZEN_PHASE3_POINTS
}
FROZEN_PHASE3_POINT_IDS: tuple[str, ...] = tuple(
    point.point_id for point in _FROZEN_PHASE3_POINTS
)
FROZEN_PHASE3_STABILITY_POINT_IDS: tuple[str, ...] = tuple(
    point.point_id for point in _FROZEN_PHASE3_POINTS if point.stability_member
)


def derive_phase3_point_fingerprint(point_id: str) -> str:
    """Return the exact preregistered tuple fingerprint for one Phase 3 point."""

    try:
        return _FROZEN_PHASE3_POINTS_BY_ID[point_id].fingerprint
    except KeyError as error:
        raise ValueError("point_id is outside the frozen Phase 3 grid") from error


@dataclasses.dataclass(frozen=True, slots=True)
class Phase3ProcessPoint(StrictModel):
    point_id: str
    runner_kind: RunnerKind
    graph_mode: GraphMode
    batch_size: int
    context_length: int
    output_steps: int
    process_replicate: int
    stability_member: bool

    def __post_init__(self) -> None:
        require_identifier(self.point_id, field_name="point_id")
        if min(
            self.batch_size,
            self.context_length,
            self.output_steps,
            self.process_replicate,
        ) <= 0:
            raise ValueError("point dimensions and replicate must be positive")
        if (
            self.runner_kind is RunnerKind.GROWING_CONTEXT
            and self.graph_mode is not GraphMode.EAGER
        ):
            raise ValueError("growing-context Phase 3 points are eager-only")
        expected = _FROZEN_PHASE3_POINTS_BY_ID.get(self.point_id)
        observed = (
            self.runner_kind,
            self.graph_mode,
            self.batch_size,
            self.context_length,
            self.output_steps,
            self.process_replicate,
            self.stability_member,
        )
        if expected is None or observed != (
            expected.runner_kind,
            expected.graph_mode,
            expected.batch_size,
            expected.context_length,
            expected.output_steps,
            expected.process_replicate,
            expected.stability_member,
        ):
            raise ValueError("process point does not match the frozen Phase 3 grid")


def expand_phase3_process_points(
    plan: Phase3AdmissionPlan,
) -> tuple[Phase3ProcessPoint, ...]:
    """Expand the preregistered plan without consulting runtime state."""

    points: list[Phase3ProcessPoint] = []
    for batch_size in plan.grid.batch_sizes:
        for context_length in plan.grid.context_lengths:
            for graph_mode in plan.graph_modes:
                stability_member = bool(
                    plan.stability is not None
                    and batch_size == plan.stability.batch_size
                    and context_length == plan.stability.context_length
                    and graph_mode in plan.stability.graph_modes
                )
                point_id = (
                    f"{plan.runner_kind.value}-b{batch_size}-l{context_length}-"
                    f"{graph_mode.value}-r1"
                )
                points.append(
                    Phase3ProcessPoint(
                        point_id=point_id,
                        runner_kind=plan.runner_kind,
                        graph_mode=graph_mode,
                        batch_size=batch_size,
                        context_length=context_length,
                        output_steps=plan.grid.output_steps,
                        process_replicate=1,
                        stability_member=stability_member,
                    )
                )
    if plan.stability is not None:
        for graph_mode in plan.stability.graph_modes:
            for replicate in range(
                plan.measurement.ordinary_process_replicates + 1,
                plan.stability.total_process_replicates + 1,
            ):
                point_id = (
                    f"fixed_l-b{plan.stability.batch_size}-"
                    f"l{plan.stability.context_length}-{graph_mode.value}-r{replicate}"
                )
                points.append(
                    Phase3ProcessPoint(
                        point_id=point_id,
                        runner_kind=RunnerKind.FIXED_L,
                        graph_mode=graph_mode,
                        batch_size=plan.stability.batch_size,
                        context_length=plan.stability.context_length,
                        output_steps=1,
                        process_replicate=replicate,
                        stability_member=True,
                    )
                )
    if len(points) != plan.expected_process_count:
        raise ValueError("expanded process count does not match the frozen plan")
    if len({point.point_id for point in points}) != len(points):
        raise ValueError("expanded point IDs must be unique")
    return tuple(points)


_EXPECTED_BACKEND_SOURCES: dict[str, str] = {
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


@dataclasses.dataclass(frozen=True, slots=True)
class BF16BackendIdentity(StrictModel):
    schema_version: str
    backend_id: str
    torch_version: str
    torch_git_sha: str
    cuda_runtime_version: str
    cudnn_version: str
    triton_version: str
    flash_generation: str
    flash_version: str
    dispatch_api: str
    selected_backend: str
    enable_gqa: bool
    compile_mode: str
    source_artifacts: tuple[SourceDigest, ...]

    SCHEMA_VERSION: ClassVar[str] = "kvbench.phase3-bf16-backend.v1"

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        require_identifier(self.backend_id, field_name="backend_id")
        require_git_sha(self.torch_git_sha)
        expected_identity = (
            "torch_sdpa_flash_gqa",
            "2.12.1+cu130",
            "7269437d655783a26cba32aa88195b741ff496aa",
            "13.0",
            "9.20.0",
            "3.7.1",
            "FA2",
            "2.5.7",
            "torch.nn.functional.scaled_dot_product_attention",
            "flash_attention",
            True,
            "disabled",
        )
        observed_identity = (
            self.backend_id,
            self.torch_version,
            self.torch_git_sha,
            self.cuda_runtime_version,
            self.cudnn_version,
            self.triton_version,
            self.flash_generation,
            self.flash_version,
            self.dispatch_api,
            self.selected_backend,
            self.enable_gqa,
            self.compile_mode,
        )
        if observed_identity != expected_identity:
            raise ValueError("BF16 backend identity does not match Decision 0007")
        paths = tuple(item.path for item in self.source_artifacts)
        if paths != tuple(sorted(_EXPECTED_BACKEND_SOURCES)):
            raise ValueError("backend source artifacts must be complete and sorted")
        for item in self.source_artifacts:
            if item.sha256 != _EXPECTED_BACKEND_SOURCES[item.path]:
                raise ValueError("backend source hash does not match Decision 0007")


def derive_cache_layout_fingerprint(
    *,
    num_layers: int,
    batch_size: int,
    num_kv_heads: int,
    capacity: int,
    head_dim: int,
    device: str,
    workspace_bytes: int,
    implementation_sha256: str,
) -> str:
    """Derive the runtime-compatible cache-layout fingerprint from declarations."""

    require_sha256(implementation_sha256, field_name="implementation_sha256")
    if min(num_layers, batch_size, num_kv_heads, capacity, head_dim) <= 0:
        raise ValueError("cache fingerprint geometry must be positive")
    if workspace_bytes < 0:
        raise ValueError("cache fingerprint workspace must be nonnegative")
    shape = (num_layers, batch_size, num_kv_heads, capacity, head_dim)
    strides = (
        batch_size * num_kv_heads * capacity * head_dim,
        num_kv_heads * capacity * head_dim,
        capacity * head_dim,
        head_dim,
        1,
    )
    return sha256_hex(
        canonical_json_bytes(
            {
                "schema": "kvbench-bf16-static-cache-layout-1.0.0",
                "shape": list(shape),
                "strides": list(strides),
                "dtype": "torch.bfloat16",
                "element_size": 2,
                "device": device,
                "padding_bytes": 0,
                "workspace_bytes": workspace_bytes,
                "implementation_sha256": implementation_sha256,
            }
        )
    )


@dataclasses.dataclass(frozen=True, slots=True)
class BF16CacheIdentity(StrictModel):
    schema_version: str
    layout_name: str
    dtype: str
    num_layers: int
    batch_size: int
    num_kv_heads: int
    capacity: int
    head_dim: int
    tensor_storage_bytes: int
    padding_bytes: int
    workspace_bytes: int
    device: str
    implementation_sha256: str
    layout_fingerprint: str

    SCHEMA_VERSION: ClassVar[str] = "kvbench.phase3-bf16-cache.v1"

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        if self.layout_name != "layers_batch_kv_heads_context_head_dim":
            raise ValueError("cache layout name is not the frozen BF16 layout")
        if self.dtype != "bfloat16":
            raise ValueError("Phase 3 cache dtype must be bfloat16")
        if min(
            self.num_layers,
            self.batch_size,
            self.num_kv_heads,
            self.capacity,
            self.head_dim,
        ) <= 0:
            raise ValueError("cache geometry must be positive")
        if (self.num_layers, self.num_kv_heads, self.head_dim) != (32, 8, 128):
            raise ValueError("cache must use the frozen KV-head geometry")
        expected_bytes = (
            2
            * self.num_layers
            * self.batch_size
            * self.num_kv_heads
            * self.capacity
            * self.head_dim
            * 2
        )
        if self.tensor_storage_bytes != expected_bytes:
            raise ValueError("cache storage bytes do not match the exact formula")
        if self.padding_bytes != 0 or self.workspace_bytes < 0:
            raise ValueError("cache padding must be zero and workspace nonnegative")
        if self.device != "cuda:0":
            raise ValueError("Phase 3 cache device identity must be cuda:0")
        require_sha256(self.implementation_sha256)
        require_sha256(self.layout_fingerprint)
        expected_fingerprint = derive_cache_layout_fingerprint(
            num_layers=self.num_layers,
            batch_size=self.batch_size,
            num_kv_heads=self.num_kv_heads,
            capacity=self.capacity,
            head_dim=self.head_dim,
            device=self.device,
            workspace_bytes=self.workspace_bytes,
            implementation_sha256=self.implementation_sha256,
        )
        if self.layout_fingerprint != expected_fingerprint:
            raise ValueError("cache layout fingerprint does not match its declaration")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase3CommandSpec(StrictModel):
    schema_version: str
    argv: tuple[str, ...]
    working_directory: str
    environment_sha256: str
    dry_run: bool

    SCHEMA_VERSION: ClassVar[str] = "kvbench-phase3-command-1.0.0"

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        if not self.argv or any(not item or "\x00" in item for item in self.argv):
            raise ValueError("command argv must contain non-empty safe strings")
        working = PurePosixPath(self.working_directory)
        if not working.is_absolute() or ".." in working.parts:
            raise ValueError("working_directory must be an absolute safe POSIX path")
        require_sha256(self.environment_sha256)
        if self.dry_run:
            raise ValueError("Phase 3 execution command cannot be a dry run")


def _validate_manifest_lifecycle(
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
            raise ValueError("created manifest cannot contain later lifecycle fields")
    elif status in {RunStatus.RUNNING, RunStatus.FINALIZING}:
        if started_at_utc is None:
            raise ValueError("running/finalizing manifest requires started_at_utc")
        require_utc_timestamp(started_at_utc, field_name="started_at_utc")
        if any(
            value is not None
            for value in (finished_at_utc, inventory_path, failure_reason)
        ):
            raise ValueError("nonterminal manifest has terminal-only fields")
    else:
        if started_at_utc is None or finished_at_utc is None:
            raise ValueError("terminal manifest requires start and finish timestamps")
        require_utc_timestamp(started_at_utc, field_name="started_at_utc")
        require_utc_timestamp(finished_at_utc, field_name="finished_at_utc")
        if inventory_path != "artifact_inventory.json":
            raise ValueError("terminal manifest must reference artifact_inventory.json")
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
class Phase3RunManifest(StrictModel):
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
    plan_source: ConfigSource
    plan_fingerprint: str
    point_id: str
    point_fingerprint: str
    git_sha: str
    git_dirty: bool
    container_digest: None
    hardware_id: str
    hardware_fingerprint: str
    native_g0_status: GateDisposition
    e00_run_id: str
    e00_manifest_sha256: str
    blocker_b010: str
    gpu_uuid: str
    gpu_full_name: str
    pci_bus_id: str
    pci_device_id: str
    driver_version: str
    software_environment_id: str
    software_fingerprint: str
    model_identity: ModelIdentityV2
    model_fingerprint: str
    method: MethodName
    method_config_id: str
    method_config_fingerprint: MethodConfigFingerprint
    contract_fingerprint: str
    measurement_protocol_fingerprint: str
    backend_identity: BF16BackendIdentity
    backend_fingerprint: str
    cache_identity: BF16CacheIdentity
    batch_size: int
    context_length: int
    output_steps: int
    warmup_count: int
    measured_count: int
    measured_batches: int
    count_unit: MeasurementCountUnit
    random_seed: int
    process_replicate: int
    quality: QualityStatus
    command: Phase3CommandSpec
    inventory_path: str | None
    failure_reason: str | None
    adapter_config_fingerprint: str | None = None

    SCHEMA_VERSION: ClassVar[str] = "kvbench-phase3-run-manifest-1.0.0"
    ARTIFACT_SCHEMA_VERSION: ClassVar[str] = "kvbench-artifacts-1.0.0"

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        require_schema(self.artifact_schema_version, self.ARTIFACT_SCHEMA_VERSION)
        require_run_id(self.run_id)
        require_identifier(self.point_id, field_name="point_id")
        require_git_sha(self.git_sha)
        if self.git_dirty:
            raise ValueError("Phase 3 admission requires a clean Git tree")
        if self.container_digest is not None:
            raise ValueError("Phase 3 native-host evidence cannot claim a container")
        for value, name in (
            (self.plan_fingerprint, "plan_fingerprint"),
            (self.point_fingerprint, "point_fingerprint"),
            (self.hardware_fingerprint, "hardware_fingerprint"),
            (self.e00_manifest_sha256, "e00_manifest_sha256"),
            (self.software_fingerprint, "software_fingerprint"),
            (self.model_fingerprint, "model_fingerprint"),
            (self.contract_fingerprint, "contract_fingerprint"),
            (
                self.measurement_protocol_fingerprint,
                "measurement_protocol_fingerprint",
            ),
            (self.backend_fingerprint, "backend_fingerprint"),
        ):
            require_sha256(value, field_name=name)
        if self.adapter_config_fingerprint is not None:
            require_sha256(
                self.adapter_config_fingerprint,
                field_name="adapter_config_fingerprint",
            )
        for value, name in (
            (self.hardware_id, "hardware_id"),
            (self.e00_run_id, "e00_run_id"),
            (self.software_environment_id, "software_environment_id"),
            (self.method_config_id, "method_config_id"),
        ):
            require_identifier(value, field_name=name)
        if any(
            not value.strip()
            for value in (
                self.gpu_uuid,
                self.gpu_full_name,
                self.pci_bus_id,
                self.pci_device_id,
                self.driver_version,
            )
        ):
            raise ValueError("live hardware identity strings must be non-empty")
        if self.run_kind is not RunKind.PHASE3_ADMISSION:
            raise ValueError("Phase 3 manifest run kind must be phase3_admission")
        expected_native_identity = (
            PHASE3_HARDWARE_ID,
            PHASE3_HARDWARE_FINGERPRINT,
            GateDisposition.PASS,
            PHASE3_E00_RUN_ID,
            PHASE3_E00_MANIFEST_SHA256,
            "OPEN",
            PHASE3_GPU_UUID,
            PHASE3_GPU_FULL_NAME,
            PHASE3_PCI_BUS_ID,
            PHASE3_PCI_DEVICE_ID,
            PHASE3_DRIVER_VERSION,
            PHASE3_SOFTWARE_ENVIRONMENT_ID,
            PHASE3_SOFTWARE_FINGERPRINT,
        )
        observed_native_identity = (
            self.hardware_id,
            self.hardware_fingerprint,
            self.native_g0_status,
            self.e00_run_id,
            self.e00_manifest_sha256,
            self.blocker_b010,
            self.gpu_uuid,
            self.gpu_full_name,
            self.pci_bus_id,
            self.pci_device_id,
            self.driver_version,
            self.software_environment_id,
            self.software_fingerprint,
        )
        if observed_native_identity != expected_native_identity:
            raise ValueError("manifest does not match the certified native-host G0")
        if self.claim_class is not ClaimClass.NONE:
            raise ValueError("Phase 3 admission manifests cannot carry claims")
        if self.measurement_scope is not MeasurementScope.NATIVE_HOST_ADMISSION:
            raise ValueError("Phase 3 manifest scope must be native_host_admission")
        if self.performance_claim_eligible:
            raise ValueError("Phase 3 admission evidence is not performance-claim eligible")
        if self.method is not MethodName.BF16:
            raise ValueError("Phase 3 admits only BF16")
        expected_method_fingerprint = (
            MethodName.BF16,
            "bf16",
            PHASE3_BF16_VARIANT_FINGERPRINT,
            False,
        )
        observed_method_fingerprint = (
            self.method_config_fingerprint.method,
            self.method_config_fingerprint.variant_id,
            self.method_config_fingerprint.sha256,
            self.method_config_fingerprint.execution_ready,
        )
        if (
            self.method_config_id != "bf16"
            or observed_method_fingerprint != expected_method_fingerprint
        ):
            raise ValueError(
                "Phase 3 requires the exact formally blocked BF16 fingerprint"
            )
        if self.model_identity.fingerprint() != self.model_fingerprint:
            raise ValueError("model fingerprint does not match embedded identity")
        if self.model_fingerprint != PHASE3_MODEL_FINGERPRINT:
            raise ValueError("manifest model is not the frozen Phase 3 checkpoint")
        if self.backend_identity.fingerprint() != self.backend_fingerprint:
            raise ValueError("backend fingerprint does not match embedded identity")
        if (
            self.quality.quality_status is not QualityValidationState.UNVALIDATED
            or self.quality.claim_eligibility is not ClaimEligibility.PERFORMANCE_ONLY
            or self.quality.quality_execution is not QualityExecutionState.LOCKED
            or self.quality.performance_data_frozen
        ):
            raise ValueError("Phase 3 quality governance must remain locked and unvalidated")
        if self.plan_source.kind is not ConfigSourceKind.PATH:
            raise ValueError("Phase 3 command requires a path plan source")
        plan_path = self.plan_source.path
        if plan_path is None:
            raise ValueError("Phase 3 path plan source is missing its path")
        expected_point = _FROZEN_PHASE3_POINTS_BY_ID.get(self.point_id)
        if expected_point is None:
            raise ValueError("manifest point_id is outside the frozen Phase 3 grid")
        expected_tuple = (
            expected_point.plan_path,
            expected_point.plan_fingerprint,
            expected_point.fingerprint,
            expected_point.runner_kind,
            expected_point.graph_mode,
            expected_point.batch_size,
            expected_point.context_length,
            expected_point.output_steps,
            expected_point.warmup_count,
            expected_point.measured_count,
            expected_point.measured_batches,
            expected_point.count_unit,
            expected_point.random_seed,
            expected_point.process_replicate,
        )
        observed_tuple = (
            plan_path,
            self.plan_fingerprint,
            self.point_fingerprint,
            self.runner_kind,
            self.graph_mode,
            self.batch_size,
            self.context_length,
            self.output_steps,
            self.warmup_count,
            self.measured_count,
            self.measured_batches,
            self.count_unit,
            self.random_seed,
            self.process_replicate,
        )
        if observed_tuple != expected_tuple:
            raise ValueError("manifest fields do not match the frozen process point")
        if (
            self.plan_source.sha256 != expected_point.plan_fingerprint
            or self.plan_fingerprint != expected_point.plan_fingerprint
        ):
            raise ValueError("manifest plan fingerprint join is inconsistent")
        if self.cache_identity.batch_size != self.batch_size:
            raise ValueError("cache and manifest batch sizes differ")
        if self.cache_identity.capacity != expected_point.capacity:
            raise ValueError("cache capacity does not match runner semantics")
        expected_argv = (
            PHASE3_PYTHON_EXECUTABLE,
            "-m",
            "kvbench",
            "phase3-worker",
            "--plan",
            expected_point.plan_path,
            "--point-id",
            expected_point.point_id,
            "--replicate",
            str(expected_point.process_replicate),
            "--run-id",
            self.run_id,
        )
        if (
            self.command.argv != expected_argv
            or self.command.working_directory != PHASE3_REPOSITORY_ROOT
        ):
            raise ValueError("execution command does not exactly reconstruct the worker")
        _validate_manifest_lifecycle(
            status=self.status,
            created_at_utc=self.created_at_utc,
            started_at_utc=self.started_at_utc,
            finished_at_utc=self.finished_at_utc,
            inventory_path=self.inventory_path,
            failure_reason=self.failure_reason,
        )


RunManifestType: TypeAlias = RunManifest | Phase3RunManifest


def parse_run_manifest(value: Mapping[str, object]) -> RunManifestType:
    """Dispatch a strict run manifest without weakening the Phase 2 v1 model."""

    schema_version = value.get("schema_version")
    models: dict[str, type[RunManifest] | type[Phase3RunManifest]] = {
        RunManifest.SCHEMA_VERSION: RunManifest,
        Phase3RunManifest.SCHEMA_VERSION: Phase3RunManifest,
    }
    if not isinstance(schema_version, str) or schema_version not in models:
        raise SchemaValidationError(
            "missing or unsupported run manifest schema_version",
            path=("schema_version",),
        )
    return models[schema_version].from_dict(dict(value))


@dataclasses.dataclass(frozen=True, slots=True)
class Phase3WorkerResult(StrictModel):
    schema_version: str
    run_id: str
    point_id: str
    runner_kind: RunnerKind
    count_unit: MeasurementCountUnit
    status: RunStatus
    expected_operations: int
    completed_operations: int
    failed_operations: int
    output_checksum: str | None
    failure_reason: str | None

    SCHEMA_VERSION: ClassVar[str] = "kvbench-phase3-worker-result-1.0.0"

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        require_run_id(self.run_id)
        require_identifier(self.point_id, field_name="point_id")
        point = _FROZEN_PHASE3_POINTS_BY_ID.get(self.point_id)
        if point is None:
            raise ValueError("worker point_id is outside the frozen Phase 3 grid")
        expected_operations = (
            point.measured_count * point.measured_batches
            if point.runner_kind is RunnerKind.FIXED_L
            else point.output_steps * point.measured_count * point.measured_batches
        )
        if (
            self.runner_kind is not point.runner_kind
            or self.count_unit is not point.count_unit
            or self.expected_operations != expected_operations
        ):
            raise ValueError("worker operation semantics do not match its frozen point")
        if not self.status.is_terminal:
            raise ValueError("worker result status must be terminal")
        if (
            self.expected_operations <= 0
            or self.completed_operations < 0
            or self.failed_operations < 0
            or self.completed_operations + self.failed_operations
            > self.expected_operations
        ):
            raise ValueError("worker operation counts must be nonnegative")
        if self.output_checksum is not None:
            require_sha256(self.output_checksum, field_name="output_checksum")
        if self.status is RunStatus.COMPLETED:
            if (
                self.completed_operations != self.expected_operations
                or self.failed_operations != 0
                or self.output_checksum is None
                or self.failure_reason is not None
            ):
                raise ValueError("completed worker result is incomplete")
        elif not self.failure_reason:
            raise ValueError("failed worker result requires a reason")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase3RunEvidence(StrictModel):
    run_id: str
    point_id: str
    point_fingerprint: str
    plan_path: str
    plan_fingerprint: str
    status: RunStatus
    manifest_sha256: str
    artifact_inventory_sha256: str
    checksum_ledger_sha256: str
    checksum_valid: bool

    def __post_init__(self) -> None:
        require_run_id(self.run_id)
        require_identifier(self.point_id, field_name="point_id")
        point = _FROZEN_PHASE3_POINTS_BY_ID.get(self.point_id)
        if point is None:
            raise ValueError("G1 evidence point is outside the frozen Phase 3 grid")
        if (
            self.point_fingerprint != point.fingerprint
            or self.plan_path != point.plan_path
            or self.plan_fingerprint != point.plan_fingerprint
        ):
            raise ValueError("G1 evidence does not retain its exact point/plan join")
        require_sha256(self.point_fingerprint, field_name="point_fingerprint")
        require_relative_path(self.plan_path, field_name="plan_path")
        require_sha256(self.plan_fingerprint, field_name="plan_fingerprint")
        if not self.status.is_terminal:
            raise ValueError("G1 run evidence must be terminal")
        for digest in (
            self.manifest_sha256,
            self.artifact_inventory_sha256,
            self.checksum_ledger_sha256,
        ):
            require_sha256(digest)


G1_CRITERIA: tuple[str, ...] = (
    "exact_model_and_tokenizer_identity",
    "exact_bf16_backend_identity",
    "numerical_reference_match",
    "no_torch_cat_growth",
    "no_unexplained_measured_region_allocation",
    "kv_head_cache_geometry",
    "gqa_not_materialized",
    "fixed_l_runner",
    "growing_context_runner",
    "eager_lane",
    "cuda_graph_capture_and_replay",
    "eager_graph_numerical_agreement",
    "graph_replay_no_allocation",
    "stable_output_checksums",
    "independent_process_replicates",
    "stability_threshold",
    "no_backend_fallback",
    "no_model_substitution",
    "no_formal_paper_claim",
    "immutable_checksum_valid_artifacts",
)
if len(set(G1_CRITERIA)) != len(G1_CRITERIA):
    raise RuntimeError("G1 criteria must be unique")


def _g1_expected_point_ids(criterion: str) -> tuple[str, ...]:
    fixed_ids = tuple(
        point.point_id
        for point in _FROZEN_PHASE3_POINTS
        if point.runner_kind is RunnerKind.FIXED_L
    )
    growing_ids = tuple(
        point.point_id
        for point in _FROZEN_PHASE3_POINTS
        if point.runner_kind is RunnerKind.GROWING_CONTEXT
    )
    eager_ids = tuple(
        point.point_id
        for point in _FROZEN_PHASE3_POINTS
        if point.graph_mode is GraphMode.EAGER
    )
    graph_ids = tuple(
        point.point_id
        for point in _FROZEN_PHASE3_POINTS
        if point.graph_mode is GraphMode.CUDA_GRAPH
    )
    if criterion == "fixed_l_runner":
        return fixed_ids
    if criterion == "growing_context_runner":
        return growing_ids
    if criterion == "eager_lane":
        return eager_ids
    if criterion in {
        "cuda_graph_capture_and_replay",
        "eager_graph_numerical_agreement",
        "graph_replay_no_allocation",
    }:
        return graph_ids
    if criterion in {
        "independent_process_replicates",
        "stability_threshold",
    }:
        return FROZEN_PHASE3_STABILITY_POINT_IDS
    return FROZEN_PHASE3_POINT_IDS


def g1_expected_point_ids(criterion: str) -> tuple[str, ...]:
    """Return the frozen scientific lane for one declared G1 criterion."""

    if criterion not in G1_CRITERIA:
        raise ValueError("unknown G1 criterion")
    return _g1_expected_point_ids(criterion)


@dataclasses.dataclass(frozen=True, slots=True)
class Phase3G1Criterion(StrictModel):
    criterion: str
    disposition: GateDisposition
    evidence_run_ids: tuple[str, ...]
    reason: str | None

    def __post_init__(self) -> None:
        require_identifier(self.criterion, field_name="criterion")
        if self.criterion not in G1_CRITERIA:
            raise ValueError("unknown G1 criterion")
        if len(set(self.evidence_run_ids)) != len(self.evidence_run_ids):
            raise ValueError("criterion evidence run IDs must be unique")
        for run_id in self.evidence_run_ids:
            require_run_id(run_id)
        if self.disposition is GateDisposition.PASS:
            if not self.evidence_run_ids or self.reason is not None:
                raise ValueError("passing criterion requires evidence and no failure reason")
        elif not self.reason:
            raise ValueError("non-passing criterion requires a reason")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase3StabilitySummary(StrictModel):
    graph_mode: GraphMode
    point_ids: tuple[str, ...]
    evidence_run_ids: tuple[str, ...]
    process_replicates: int
    process_median_host_wall_ms: tuple[float, ...]
    median_host_wall_ms: float
    minimum_host_wall_ms: float
    maximum_host_wall_ms: float
    coefficient_of_variation_percent: float
    temperature_min_c: float
    temperature_max_c: float
    sm_clock_min_mhz: int
    sm_clock_max_mhz: int
    power_min_w: float
    power_max_w: float
    summary_artifact_path: str
    summary_artifact_sha256: str

    def __post_init__(self) -> None:
        expected_points = tuple(
            point.point_id
            for point in _FROZEN_PHASE3_POINTS
            if point.stability_member and point.graph_mode is self.graph_mode
        )
        if (
            self.graph_mode not in {GraphMode.EAGER, GraphMode.CUDA_GRAPH}
            or self.point_ids != expected_points
            or len(self.evidence_run_ids) != 3
            or len(set(self.evidence_run_ids)) != 3
            or self.process_replicates != 3
            or len(self.process_median_host_wall_ms) != 3
        ):
            raise ValueError("stability summary does not match the frozen replicate set")
        for run_id in self.evidence_run_ids:
            require_run_id(run_id)
        numeric_values = (
            *self.process_median_host_wall_ms,
            self.median_host_wall_ms,
            self.minimum_host_wall_ms,
            self.maximum_host_wall_ms,
            self.coefficient_of_variation_percent,
            self.temperature_min_c,
            self.temperature_max_c,
            self.power_min_w,
            self.power_max_w,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("stability summary values must be finite")
        if (
            any(value <= 0.0 for value in self.process_median_host_wall_ms)
            or self.median_host_wall_ms <= 0.0
            or self.minimum_host_wall_ms <= 0.0
            or self.minimum_host_wall_ms > self.maximum_host_wall_ms
        ):
            raise ValueError("stability host-wall values must be positive and ordered")
        if self.coefficient_of_variation_percent < 0.0:
            raise ValueError("stability coefficient of variation must be nonnegative")
        derived_median = float(statistics.median(self.process_median_host_wall_ms))
        derived_minimum = min(self.process_median_host_wall_ms)
        derived_maximum = max(self.process_median_host_wall_ms)
        derived_cv = (
            statistics.stdev(self.process_median_host_wall_ms)
            / statistics.mean(self.process_median_host_wall_ms)
            * 100.0
        )
        derived_values = (
            (self.median_host_wall_ms, derived_median),
            (self.minimum_host_wall_ms, derived_minimum),
            (self.maximum_host_wall_ms, derived_maximum),
            (self.coefficient_of_variation_percent, derived_cv),
        )
        if any(
            not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12)
            for observed, expected in derived_values
        ):
            raise ValueError("stability statistics must derive from raw process medians")
        if (
            self.temperature_min_c > self.temperature_max_c
            or self.sm_clock_min_mhz <= 0
            or self.sm_clock_min_mhz > self.sm_clock_max_mhz
            or self.power_min_w < 0.0
            or self.power_min_w > self.power_max_w
        ):
            raise ValueError("stability telemetry ranges are invalid")
        require_sha256(
            self.summary_artifact_sha256,
            field_name="summary_artifact_sha256",
        )
        expected_summary_path = f"stability/{self.graph_mode.value}.json"
        require_relative_path(
            self.summary_artifact_path,
            field_name="summary_artifact_path",
        )
        if self.summary_artifact_path != expected_summary_path:
            raise ValueError("stability summary path does not match graph mode")


@dataclasses.dataclass(frozen=True, slots=True)
class Phase3G1AdmissionReport(StrictModel):
    schema_version: str
    generated_at_utc: str
    git_sha: str
    status: GateDisposition
    g0: GateDisposition
    g1: GateDisposition
    g2: GateDisposition
    g3: GateDisposition
    g4: GateDisposition
    g5: GateDisposition
    full_scan_state: str
    quality: QualityStatus
    quality_benchmark_executed: bool
    quality_only_dependencies_installed: bool
    measurement_scope: MeasurementScope
    performance_claim_eligible: bool
    performance_data_frozen: bool
    blocker_b009: str
    blocker_b010: str
    expected_process_count: int
    plan_sources: tuple[SourceDigest, ...]
    run_evidence: tuple[Phase3RunEvidence, ...]
    stability_summaries: tuple[Phase3StabilitySummary, ...]
    criteria: tuple[Phase3G1Criterion, ...]
    all_artifacts_checksum_valid: bool
    formal_paper_claim_generated: bool

    SCHEMA_VERSION: ClassVar[str] = "kvbench-phase3-g1-admission-report-1.0.0"

    def __post_init__(self) -> None:
        require_schema(self.schema_version, self.SCHEMA_VERSION)
        require_utc_timestamp(self.generated_at_utc)
        require_git_sha(self.git_sha)
        if self.status not in {
            GateDisposition.PASS,
            GateDisposition.PARTIAL,
            GateDisposition.BLOCKED,
            GateDisposition.FAIL,
        }:
            raise ValueError("Phase 3 report status must be terminal")
        if self.g0 is not GateDisposition.PASS or self.g1 is not self.status:
            raise ValueError("G0 remains PASS and G1 must match the report status")
        if any(
            gate is not GateDisposition.NOT_EVALUATED
            for gate in (self.g2, self.g3, self.g4, self.g5)
        ):
            raise ValueError("G2-G5 must remain NOT_EVALUATED")
        if self.full_scan_state != "closed":
            raise ValueError("Full Scan must remain closed")
        if (
            self.quality.quality_status is not QualityValidationState.UNVALIDATED
            or self.quality.claim_eligibility is not ClaimEligibility.PERFORMANCE_ONLY
            or self.quality.quality_execution is not QualityExecutionState.LOCKED
            or self.quality.performance_data_frozen
            or self.quality_benchmark_executed
            or self.quality_only_dependencies_installed
        ):
            raise ValueError("quality execution must remain locked and absent")
        if (
            self.measurement_scope is not MeasurementScope.NATIVE_HOST_ADMISSION
            or self.performance_claim_eligible
            or self.performance_data_frozen
        ):
            raise ValueError("Phase 3 report must retain non-claim native-host scope")
        if self.blocker_b009 != "OPEN" or self.blocker_b010 != "OPEN":
            raise ValueError("B-009 and B-010 must remain OPEN")
        if self.expected_process_count != 20:
            raise ValueError("Phase 3 preregistration contains exactly 20 processes")
        expected_plan_sources = tuple(
            SourceDigest(path=path, sha256=fingerprint)
            for path, fingerprint in PHASE3_PLAN_FINGERPRINTS.items()
        )
        if self.plan_sources != expected_plan_sources:
            raise ValueError("G1 report plan fingerprints do not match the frozen plans")
        run_ids = tuple(item.run_id for item in self.run_evidence)
        point_ids = tuple(item.point_id for item in self.run_evidence)
        if len(set(run_ids)) != len(run_ids) or len(set(point_ids)) != len(point_ids):
            raise ValueError("G1 run and point evidence identities must be unique")
        criterion_names = tuple(item.criterion for item in self.criteria)
        if criterion_names != G1_CRITERIA:
            raise ValueError("G1 criteria must be complete and retain their frozen order")
        dispositions = {item.disposition for item in self.criteria}
        expected_status = (
            GateDisposition.FAIL
            if GateDisposition.FAIL in dispositions
            else GateDisposition.BLOCKED
            if GateDisposition.BLOCKED in dispositions
            else GateDisposition.PARTIAL
            if GateDisposition.PARTIAL in dispositions
            else GateDisposition.PASS
        )
        if self.status is not expected_status:
            raise ValueError("G1 status must equal the worst criterion disposition")
        known_runs = set(run_ids)
        if any(
            run_id not in known_runs
            for criterion in self.criteria
            for run_id in criterion.evidence_run_ids
        ):
            raise ValueError("criterion references unknown run evidence")
        evidence_by_run_id = {
            item.run_id: item for item in self.run_evidence
        }
        for summary in self.stability_summaries:
            if any(
                run_id not in known_runs for run_id in summary.evidence_run_ids
            ):
                raise ValueError("stability summary references unknown run evidence")
            joined_points = tuple(
                evidence_by_run_id[run_id].point_id
                for run_id in summary.evidence_run_ids
            )
            if joined_points != summary.point_ids:
                raise ValueError("stability summary run/point join is inconsistent")
        if self.formal_paper_claim_generated:
            raise ValueError("Phase 3 cannot generate a formal paper claim")
        if self.status is GateDisposition.PASS:
            if point_ids != FROZEN_PHASE3_POINT_IDS:
                raise ValueError("G1 PASS requires the exact ordered 20-point grid")
            if any(
                item.status is not RunStatus.COMPLETED or not item.checksum_valid
                for item in self.run_evidence
            ):
                raise ValueError("G1 PASS requires completed checksum-valid runs")
            if any(
                item.disposition is not GateDisposition.PASS
                for item in self.criteria
            ):
                raise ValueError("G1 PASS requires every criterion to pass")
            if not self.all_artifacts_checksum_valid:
                raise ValueError("G1 PASS requires checksum-valid artifacts")
            run_id_by_point = {
                item.point_id: item.run_id for item in self.run_evidence
            }
            for criterion in self.criteria:
                expected_evidence = tuple(
                    run_id_by_point[point_id]
                    for point_id in _g1_expected_point_ids(criterion.criterion)
                )
                if criterion.evidence_run_ids != expected_evidence:
                    raise ValueError(
                        "G1 criterion evidence does not cover its exact scientific lane"
                    )
            if tuple(
                summary.graph_mode for summary in self.stability_summaries
            ) != (GraphMode.EAGER, GraphMode.CUDA_GRAPH):
                raise ValueError("G1 PASS requires eager and graph stability summaries")
            for summary in self.stability_summaries:
                expected_run_ids = tuple(
                    run_id_by_point[point_id] for point_id in summary.point_ids
                )
                if summary.evidence_run_ids != expected_run_ids:
                    raise ValueError(
                        "stability summary is not joined to its exact run evidence"
                    )
                if summary.coefficient_of_variation_percent > 3.0:
                    raise ValueError("G1 PASS requires stability CV <= 3 percent")


Phase3ConfigDocument: TypeAlias = ModelIdentityV2 | Phase3AdmissionPlan
Phase3ReferencedModel: TypeAlias = ModelIdentityV2
Phase3ReferencedMethod: TypeAlias = MethodConfig
Phase3ReferencedHardware: TypeAlias = HardwareManifest
