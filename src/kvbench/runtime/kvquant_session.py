"""Narrow common-runner session bridge for Phase 11 KVQuant admission."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import ctypes
from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
import struct
from typing import Any

from kvbench.adapters import MethodRuntimeContext, build_method_adapter
from kvbench.adapters.kvquant import (
    KVQUANT_AGGREGATE_PATCH_SHA256,
    KVQUANT_AUTHORIZED_CONTAINER_DIGEST,
    KVQUANT_CORRECTED_COMMIT,
    KVQUANT_CORRECTED_CUDA_SHA256,
    KVQUANT_CORRECTED_TREE,
    KVQUANT_DECISIONS,
    KVQUANT_DECISION_0021_PATCH_SHA256,
    KVQUANT_Q4_DECISIONS,
    KVQUANT_DETERMINISTIC_VALUE_DECODE_APIS,
    KVQUANT_EXECUTION_SOURCE_IDENTIFIER,
    KVQUANT_EXTENSION_SHA256,
    KVQUANT_METHOD_IDENTIFIER,
    KVQUANT_QUANTIZER_SHA256,
    KVQUANT_UPSTREAM_BASE_COMMIT,
    KVQUANT_UPSTREAM_BASE_TREE,
    KVQuantMethodAdapter,
)
from kvbench.config import load_config
from kvbench.runtime.allocation import MemorySnapshot, capture_cuda_memory_snapshot
from kvbench.runtime.backend import BACKEND_IDENTITY
from kvbench.runtime.bf16_endpoint import BF16DecodeEndpoint
from kvbench.runtime.cuda_graph import CapturedFixedGraph, capture_fixed_graph
from kvbench.runtime.kvquant_cache import (
    KVQUANT_CONFIG_BITS,
    KVQUANT_Q4_VALUE_DECODE_WORKSPACE_FORMULA_VERSION,
    KVQuantStaticCache,
)
from kvbench.runtime.model_loader import (
    LoadedFrozenModel,
    MODEL_ID,
    MODEL_REVISION,
    validate_loaded_frozen_model_receipt,
)
from kvbench.runtime.numerical import (
    NumericalComparison,
    compare_tensors_untimed,
    tensor_sha256_untimed,
)
from kvbench.runtime.turboquant_session import (
    EndpointSessionError,
    TurboQuantEndpointSession,
)
from kvbench.schema import (
    GraphMode,
    MeasurementScope,
    MethodConfig,
    MethodName,
    RunnerKind,
    canonical_json_bytes,
    sha256_hex,
)
from kvbench.schema.config import KVQuantParameters, VariantRole
from kvbench.schema.phase11 import (
    PHASE11_BOUNDED_POINT_SIGNATURES,
    PHASE11_CALIBRATION_ID,
    PHASE11_CALIBRATION_ROOT,
    PHASE11_DECODE_ATOL,
    PHASE11_DECODE_RTOL,
    PHASE11_FIXTURE_ID,
    PHASE11_FIXTURE_ROOT,
    PHASE11_HISTORICAL_FIXTURE_ID,
    PHASE11_HISTORICAL_FIXTURE_ROOT,
)


PHASE11_ENDPOINT_WARMUP_OPERATIONS = 3
_LAYERS = 32
_QUERY_HEADS = 32
_KV_HEADS = 8
KVQUANT_LEGACY_BACKEND_FINGERPRINT = (
    "d18b73d981115453aaeaa3100b2e33b4403cdc48a1b2c4e518708d95810d4396"
)
_HEAD_DIM = 128
_METHOD_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "methods" / "kvquant.yaml"
)
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_FIXED_GRID = frozenset(
    (configuration, graph_mode, context)
    for (
        configuration,
        runner_kind,
        graph_mode,
        context,
        output_steps,
    ) in PHASE11_BOUNDED_POINT_SIGNATURES
    if runner_kind is RunnerKind.FIXED_L and output_steps == 1
)
_GROWING_GRID = tuple(
    signature
    for signature in PHASE11_BOUNDED_POINT_SIGNATURES
    if signature[1] is RunnerKind.GROWING_CONTEXT
)
if len(_GROWING_GRID) != 1:  # pragma: no cover - import-time authority guard
    raise RuntimeError("Phase 11 must contain exactly one growing configuration")
_GROWING_SIGNATURE = _GROWING_GRID[0]


def _torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as error:
        raise EndpointSessionError("PyTorch is unavailable") from error


def _require_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_CHARACTERS for character in value)
    ):
        raise EndpointSessionError(f"{label} is not a SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class KVQuantOperationKey:
    """One operation from only the nine-point Phase 11 admission grid."""

    configuration: str
    runner_kind: RunnerKind
    graph_mode: GraphMode
    historical_context: int
    attended_context: int
    batch_size: int
    capacity: int
    decode_step: int
    operation_fingerprint_sha256: str

    def __post_init__(self) -> None:
        if self.configuration not in KVQUANT_CONFIG_BITS:
            raise ValueError("operation configuration is not frozen")
        if (
            self.batch_size != 1
            or type(self.historical_context) is not int
            or self.historical_context <= 0
            or self.attended_context != self.historical_context + 1
            or type(self.capacity) is not int
            or self.capacity < self.attended_context
            or type(self.decode_step) is not int
            or self.decode_step < 0
        ):
            raise ValueError("Phase 11 operation geometry is invalid")
        if self.runner_kind is RunnerKind.FIXED_L:
            if (
                self.decode_step != 0
                or self.capacity != self.historical_context + 1
                or (
                    self.configuration,
                    self.graph_mode,
                    self.historical_context,
                )
                not in _FIXED_GRID
            ):
                raise ValueError("fixed operation is outside the Phase 11 grid")
        else:
            growing_configuration = _GROWING_SIGNATURE[0]
            growing_graph = _GROWING_SIGNATURE[2]
            growing_start = _GROWING_SIGNATURE[3]
            growing_steps = _GROWING_SIGNATURE[4]
            if (
                self.runner_kind is not RunnerKind.GROWING_CONTEXT
                or self.configuration != growing_configuration
                or self.graph_mode is not growing_graph
                or self.capacity != growing_start + growing_steps
                or self.decode_step >= growing_steps
                or self.historical_context != growing_start + self.decode_step
            ):
                raise ValueError(
                    "growing operation is outside the Phase 11 grid"
                )
        _require_sha256(
            self.operation_fingerprint_sha256,
            "operation fingerprint",
        )

    @classmethod
    def create(
        cls,
        *,
        configuration: str,
        runner_kind: RunnerKind,
        graph_mode: GraphMode,
        starting_context: int,
        capacity: int,
        decode_step: int,
    ) -> "KVQuantOperationKey":
        historical = starting_context + (
            decode_step
            if runner_kind is RunnerKind.GROWING_CONTEXT
            else 0
        )
        payload = {
            "schema_version": "kvbench-phase11-kvquant-operation-key-1.0.0",
            "configuration": configuration,
            "runner_kind": runner_kind.value,
            "graph_mode": graph_mode.value,
            "historical_context": historical,
            "attended_context": historical + 1,
            "batch_size": 1,
            "capacity": capacity,
            "decode_step": decode_step,
        }
        return cls(
            configuration=configuration,
            runner_kind=runner_kind,
            graph_mode=graph_mode,
            historical_context=historical,
            attended_context=historical + 1,
            batch_size=1,
            capacity=capacity,
            decode_step=decode_step,
            operation_fingerprint_sha256=sha256_hex(
                canonical_json_bytes(payload)
            ),
        )


def build_kvquant_operation_keys(
    *,
    configuration: str,
    runner_kind: RunnerKind,
    graph_mode: GraphMode,
    starting_context: int,
    output_steps: int,
) -> tuple[KVQuantOperationKey, ...]:
    """Build exactly one of the nine bounded Phase 11 admission points."""

    signature = (
        configuration,
        runner_kind,
        graph_mode,
        starting_context,
        output_steps,
    )
    if (
        configuration not in KVQUANT_CONFIG_BITS
        or not isinstance(runner_kind, RunnerKind)
        or not isinstance(graph_mode, GraphMode)
        or type(starting_context) is not int
        or type(output_steps) is not int
        or signature not in PHASE11_BOUNDED_POINT_SIGNATURES
    ):
        raise ValueError("operation set is outside the bounded Phase 11 grid")
    if runner_kind is RunnerKind.FIXED_L:
        capacity = starting_context + 1
        count = 1
    else:
        capacity = starting_context + output_steps
        count = output_steps
    return tuple(
        KVQuantOperationKey.create(
            configuration=configuration,
            runner_kind=runner_kind,
            graph_mode=graph_mode,
            starting_context=starting_context,
            capacity=capacity,
            decode_step=step,
        )
        for step in range(count)
    )


def phase11_kvquant_backend_fingerprint(configuration: str = "kvq4") -> str:
    """Bind the common endpoint to the corrected execution-source authority."""

    if configuration not in KVQUANT_CONFIG_BITS:
        raise EndpointSessionError("KVQuant backend configuration differs")
    if configuration != "kvq4":
        return KVQUANT_LEGACY_BACKEND_FINGERPRINT
    source_root = Path(__file__).resolve().parents[1]
    payload = {
        "schema_version": "kvbench-phase13r-kvquant-q4-backend-1.0.0",
        "prefill_backend": BACKEND_IDENTITY,
        "decode_backend": {
            "implementation": "corrected_kvquant_direct_compressed_decode",
            "method_identifier": KVQUANT_METHOD_IDENTIFIER,
            "execution_source_identifier": (
                KVQUANT_EXECUTION_SOURCE_IDENTIFIER
            ),
            "upstream_base_commit": KVQUANT_UPSTREAM_BASE_COMMIT,
            "upstream_base_tree": KVQUANT_UPSTREAM_BASE_TREE,
            "decision_0021_patch_sha256": (
                KVQUANT_DECISION_0021_PATCH_SHA256
            ),
            "aggregate_patch_sha256": KVQUANT_AGGREGATE_PATCH_SHA256,
            "corrected_commit": KVQUANT_CORRECTED_COMMIT,
            "corrected_tree": KVQUANT_CORRECTED_TREE,
            "corrected_cuda_sha256": KVQUANT_CORRECTED_CUDA_SHA256,
            "extension_sha256": KVQUANT_EXTENSION_SHA256,
            "decisions": list(KVQUANT_Q4_DECISIONS),
            "fixture_id": PHASE11_FIXTURE_ID,
            "fixture_root": PHASE11_FIXTURE_ROOT,
            "authorized_container_digest": (
                KVQUANT_AUTHORIZED_CONTAINER_DIGEST
            ),
            "deterministic_value_decode": {
                "apis": {
                    str(bits): api
                    for bits, api in (
                        KVQUANT_DETERMINISTIC_VALUE_DECODE_APIS.items()
                    )
                },
                "caller_owned_workspace": True,
                "q4_workspace_shape": "[batch,32,capacity_tiles,128]",
                "workspace_dtype": "float32",
                "q4_workspace_formula_version": (
                    KVQUANT_Q4_VALUE_DECODE_WORKSPACE_FORMULA_VERSION
                ),
                "q4_workspace_capacity_source": "declared_cache_capacity",
                "q23_workspace_alias": "decode_logits",
                "q23_additional_allocated_bytes": 0,
                "reduction_order": (
                    "token_ascending_slot_ascending_then_tile_ascending"
                ),
                "current_pytorch_cuda_stream": True,
            },
        },
        "local_sources": {
            relative: hashlib.sha256(
                (source_root / relative).read_bytes()
            ).hexdigest()
            for relative in (
                "adapters/kvquant.py",
                "runtime/bf16_endpoint.py",
                "runtime/kvquant_cache.py",
                "runtime/kvquant_session.py",
            )
        },
    }
    return sha256_hex(canonical_json_bytes(payload))


def kvquant_runtime_context(configuration: str = "kvq4") -> MethodRuntimeContext:
    """Return the sole frozen full-model context for all KVQuant variants."""

    return MethodRuntimeContext(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        backend_id="pytorch_flash_kvquant_longctx_deterministic_q23_v4",
        backend_fingerprint=phase11_kvquant_backend_fingerprint(configuration),
        num_layers=_LAYERS,
        num_query_heads=_QUERY_HEADS,
        num_kv_heads=_KV_HEADS,
        head_dim=_HEAD_DIM,
    )


def load_frozen_kvquant_method_config() -> MethodConfig:
    """Load only the exact Phase 9 technical and calibration authority."""

    document = load_config(_METHOD_CONFIG_PATH)
    calibration = (
        document.calibration if isinstance(document, MethodConfig) else None
    )
    if (
        type(document) is not MethodConfig
        or document.method is not MethodName.KVQUANT
        or document.method_config_id != "kvquant"
        or document.source_lock_id != "kvquant"
        or document.source_revision != KVQUANT_UPSTREAM_BASE_COMMIT
        or calibration is None
        or calibration.method_identifier != KVQUANT_METHOD_IDENTIFIER
        or calibration.calibration_id != PHASE11_CALIBRATION_ID
        or calibration.calibration_root_digest != PHASE11_CALIBRATION_ROOT
        or calibration.source_base_commit != KVQUANT_UPSTREAM_BASE_COMMIT
        or calibration.source_base_tree != KVQUANT_UPSTREAM_BASE_TREE
        or calibration.patch_sha256 != KVQUANT_DECISION_0021_PATCH_SHA256
        or tuple(variant.variant_id for variant in document.variants)
        != tuple(KVQUANT_CONFIG_BITS)
    ):
        raise EndpointSessionError(
            "canonical KVQuant method authority differs"
        )
    for variant in document.variants:
        parameters = variant.parameters
        if (
            variant.role is not VariantRole.MAIN
            or not isinstance(parameters, KVQuantParameters)
            or parameters.bits != KVQUANT_CONFIG_BITS[variant.variant_id]
            or parameters.sink_tokens != 5
            or parameters.outlier_cap != 12
            or parameters.calibration_artifact_sha256
            != KVQUANT_QUANTIZER_SHA256[variant.variant_id]
            or parameters.sparse_index_dtype != "int32"
            or parameters.lut_scale_dtype != "float32"
        ):
            raise EndpointSessionError(
                "KVQuant variant calibration binding differs"
            )
    return document


def _tensor_sequence_digest(
    tensors: Sequence[tuple[str, Any]],
    *,
    historical_context: int,
    configuration: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        canonical_json_bytes(
            {
                "schema_version": (
                    "kvbench-phase11-kvquant-historical-prefix-1.0.0"
                ),
                "configuration": configuration,
                "historical_context": historical_context,
            }
        )
    )
    for label, tensor in tensors:
        digest.update(label.encode("ascii"))
        digest.update(tensor_sha256_untimed(tensor).encode("ascii"))
    return digest.hexdigest()


def _historical_prefix_sha256(
    cache: KVQuantStaticCache,
    historical_context: int,
) -> str:
    """Hash source-faithful history while excluding the fixed scratch slot."""

    if (
        type(historical_context) is not int
        or historical_context < cache.sink_tokens
        or historical_context > cache.active_context
    ):
        raise EndpointSessionError("historical prefix length is invalid")
    quantized = historical_context - cache.sink_tokens
    tensors = (
        ("packed_key", cache.packed_key_cache[..., :quantized]),
        ("packed_value", cache.packed_value_cache[..., :quantized]),
        ("key_codebook", cache.key_codebook),
        ("key_lookup", cache.key_lookup_table),
        ("key_lower", cache.key_lower_threshold),
        ("key_upper", cache.key_upper_threshold),
        ("key_zero", cache.key_zero_point),
        ("rope_inv_freq", cache.rope_inv_freq),
        ("value_codebook", cache.value_codebook),
        ("value_lookup", cache.value_lookup_cache[:, :, :quantized]),
        ("key_sparse_values", cache.key_sparse_values[:, :, :quantized]),
        ("key_sparse_indices", cache.key_sparse_indices[:, :, :quantized]),
        ("value_sparse_values", cache.value_sparse_values[:, :, :quantized]),
        ("value_sparse_indices", cache.value_sparse_indices[:, :, :quantized]),
        ("key_counts", cache.key_active_counts[:, :, :quantized]),
        ("value_counts", cache.value_active_counts[:, :, :quantized]),
        ("sink_key", cache.sink_key),
        ("sink_value", cache.sink_value),
    )
    return _tensor_sequence_digest(
        tensors,
        historical_context=historical_context,
        configuration=cache.config_name,
    )


def _key_active_entries_untimed(
    cache: KVQuantStaticCache,
    active_context: int,
) -> int:
    """Read exact logical occupancy only outside measured execution."""

    torch = _torch()
    quantized = max(0, active_context - cache.sink_tokens)
    selected = (
        cache.key_active_counts[:, :, :quantized]
        .detach()
        .contiguous()
        .view(torch.uint8)
        .to(device="cpu", copy=True)
    )
    byte_count = int(selected.numel())
    raw_array = (ctypes.c_ubyte * byte_count).from_address(
        int(selected.data_ptr())
    )
    raw = memoryview(raw_array).cast("B")
    if len(raw) != byte_count or byte_count % 4:
        raise EndpointSessionError("Key active-count bytes are incomplete")
    return sum(
        value[0]
        for value in struct.iter_unpack("<i", raw)
    )


def _endpoint_rope_scratch_state(
    endpoint: BF16DecodeEndpoint,
    cache: KVQuantStaticCache,
) -> tuple[int, int, dict[str, int]]:
    """Return exact/predicted bytes and pointers for persistent RoPE scratch."""

    torch = _torch()
    tensors = (
        ("endpoint_query_rope_scratch", endpoint.query_rope_scratch),
        ("endpoint_key_rope_scratch", endpoint.key_rope_scratch),
    )
    expected_shapes = (
        (
            cache.num_layers,
            cache.batch_size,
            cache.num_query_heads,
            1,
            cache.head_dim // 2,
        ),
        (
            cache.num_layers,
            cache.batch_size,
            cache.num_kv_heads,
            1,
            cache.head_dim // 2,
        ),
    )
    if any(
        tuple(int(item) for item in tensor.shape) != expected_shape
        or tensor.dtype != torch.bfloat16
        or tensor.device != cache.device
        or not tensor.is_contiguous()
        for (_, tensor), expected_shape in zip(
            tensors,
            expected_shapes,
            strict=True,
        )
    ):
        raise EndpointSessionError(
            "endpoint RoPE scratch differs from frozen persistent layout"
        )
    actual_bytes = sum(
        int(tensor.untyped_storage().nbytes()) for _, tensor in tensors
    )
    predicted_bytes = (
        cache.num_layers
        * cache.batch_size
        * (cache.num_query_heads + cache.num_kv_heads)
        * (cache.head_dim // 2)
        * 2
    )
    if (
        actual_bytes != predicted_bytes
        or endpoint.workspace_bytes != actual_bytes
    ):
        raise EndpointSessionError(
            "endpoint RoPE scratch byte identity differs"
        )
    pointers = {
        f"{name}_data_ptr": int(tensor.data_ptr())
        for name, tensor in tensors
    }
    if len(set(pointers.values())) != len(pointers):
        raise EndpointSessionError("endpoint RoPE scratch storage aliases")
    return actual_bytes, predicted_bytes, pointers


def _composite_cache_pointers(
    cache: KVQuantStaticCache,
    endpoint: BF16DecodeEndpoint,
) -> dict[str, int]:
    """Bind static-cache and endpoint scratch identities as one custody set."""

    _, _, endpoint_pointers = _endpoint_rope_scratch_state(endpoint, cache)
    pointers = cache.pointers()
    if set(pointers).intersection(endpoint_pointers):
        raise EndpointSessionError("composite pointer labels overlap")
    return {**pointers, **endpoint_pointers}


class KVQuantEndpointSession(TurboQuantEndpointSession):
    """One warmed KVQuant endpoint using the common runner session contract."""

    measurement_scope = MeasurementScope.MEASUREMENT_CONTAINER_ADMISSION

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._pointers = self.current_cache_pointers()

    def current_cache_pointers(self) -> dict[str, int]:
        return _composite_cache_pointers(self.cache, self.endpoint)

    def current_historical_prefix_sha256(self) -> str:
        return _historical_prefix_sha256(
            self.cache,
            self.operation_keys[0].historical_context,
        )

    def method_cache_accounting(self) -> dict[str, object]:
        accounting = self.cache.accounting().to_dict()
        if self.method.allocated_bytes(self.cache) != accounting["allocated_bytes"]:
            raise EndpointSessionError("method and cache bytes differ")
        endpoint_bytes, endpoint_predicted, _ = _endpoint_rope_scratch_state(
            self.endpoint,
            self.cache,
        )
        accounting["predicted_tensor_bytes"] = (
            int(accounting["predicted_tensor_bytes"]) + endpoint_predicted
        )
        accounting["measured_tensor_bytes"] = (
            int(accounting["measured_tensor_bytes"]) + endpoint_bytes
        )
        accounting["allocated_bytes"] = (
            int(accounting["allocated_bytes"]) + endpoint_bytes
        )
        accounting["staging_bytes"] = (
            int(accounting["staging_bytes"]) + endpoint_bytes
        )
        accounting["endpoint_rope_scratch_bytes"] = endpoint_bytes
        accounting["relative_error"] = (
            abs(
                int(accounting["predicted_tensor_bytes"])
                - int(accounting["allocated_bytes"])
            )
            / int(accounting["allocated_bytes"])
        )
        if float(accounting["relative_error"]) >= 0.01:
            raise EndpointSessionError(
                "composite KVQuant allocation differs by at least 1%"
            )
        key_entries = _key_active_entries_untimed(
            self.cache,
            self.cache.active_context,
        )
        accounting.update(
            {
                "active_storage_bytes": self.cache.active_storage_bytes(
                    key_active_entries=key_entries
                ),
                "logical_bf16_active_bytes": (
                    self.cache.active_logical_bf16_bytes()
                ),
                "logical_bf16_allocated_bytes": (
                    self.cache.logical_bf16_storage_bytes
                ),
                "key_active_entries": key_entries,
            }
        )

        if self.cache.config_name == "kvq4":
            geometry = self.cache.q4_value_decode_workspace_geometry()
            accounting.update(
                {
                    "q4_workspace_formula_version": geometry[
                        "formula_version"
                    ],
                    "q4_workspace_tile_size": geometry["tile_size"],
                    "q4_workspace_tile_capacity": geometry[
                        "tile_capacity"
                    ],
                    "q4_workspace_total_attended_capacity": geometry[
                        "total_attended_capacity"
                    ],
                    "q4_workspace_quantized_value_capacity": geometry[
                        "quantized_value_capacity"
                    ],
                    "q4_workspace_shape": geometry["workspace_shape"],
                    "q4_workspace_bytes": geometry["workspace_bytes"],
                }
            )
        return accounting

    def method_byte_breakdown(self) -> dict[str, int]:
        breakdown = dict(self.method.byte_breakdown(self.cache))
        endpoint_bytes, _, _ = _endpoint_rope_scratch_state(
            self.endpoint,
            self.cache,
        )
        try:
            breakdown["staging"] += endpoint_bytes
        except KeyError as error:
            raise EndpointSessionError(
                "KVQuant byte breakdown lacks staging"
            ) from error
        if sum(breakdown.values()) != int(
            self.method_cache_accounting()["allocated_bytes"]
        ):
            raise EndpointSessionError(
                "composite KVQuant byte breakdown is not exact"
            )
        return dict(sorted(breakdown.items()))

    def method_allocation_ratios(self) -> dict[str, float | None]:
        allocated = int(self.method_cache_accounting()["allocated_bytes"])
        logical = self.cache.logical_bf16_storage_bytes
        rho_alloc = allocated / logical
        r_alloc = logical / allocated
        return {
            "rho_alloc": rho_alloc,
            "r_alloc": r_alloc,
            "reciprocal_product": rho_alloc * r_alloc,
            "reciprocal_error": abs(rho_alloc * r_alloc - 1.0),
            "r_hbm": None,
        }

    def admit(
        self,
        *,
        observed_outputs: Sequence[tuple[str, bool]],
        execution_path_passed: bool,
        allocation_passed: bool,
        graph_passed: bool,
    ) -> None:
        """Use the common state machine with KVQuant physical-prefix custody."""

        verdicts = (
            execution_path_passed,
            allocation_passed,
            graph_passed,
        )
        observed = tuple(observed_outputs)
        if (
            self._state != "warmed"
            or any(type(value) is not bool for value in verdicts)
            or len(observed) != len(self.operation_keys)
        ):
            raise EndpointSessionError("session admission evidence is invalid")
        for index, (digest, finite) in enumerate(observed):
            _require_sha256(digest, "audit output")
            if (
                type(finite) is not bool
                or (digest, finite) != self._warmed_outputs[index]
            ):
                raise EndpointSessionError("post-warmup output is unstable")
        if (
            not all(verdicts)
            or not all(finite for _, finite in observed)
            or (
                self.eager_graph_comparison is not None
                and not self.eager_graph_comparison.passed
            )
        ):
            self._state = "failed"
            raise EndpointSessionError("session audits did not pass")
        validate_loaded_frozen_model_receipt(self.loaded)
        if (
            self.loaded.receipt.receipt_sha256 != self._receipt_sha256
            or self.current_cache_pointers() != self._pointers
        ):
            raise EndpointSessionError("session identity changed during audit")
        first = self.operation_keys[0]
        if first.runner_kind is RunnerKind.GROWING_CONTEXT:
            if self._reset_growing is None:
                raise EndpointSessionError("growing reset is unavailable")
            self._reset_growing()
        if self.current_historical_prefix_sha256() != self._prefix_sha256:
            raise EndpointSessionError("historical prefix changed during audit")
        self._audit_outputs = dict(enumerate(observed))
        self._state = "ready"


def build_kvquant_endpoint_session(
    *,
    loaded: LoadedFrozenModel,
    operation_keys: tuple[KVQuantOperationKey, ...],
    prefix_input_ids: Any,
    decode_input_ids: Any,
) -> KVQuantEndpointSession:
    """Allocate, bind authority, prefill, warm, and retain one KVQuant session."""

    validate_loaded_frozen_model_receipt(loaded)
    if not operation_keys:
        raise EndpointSessionError("operation set is empty")
    first = operation_keys[0]
    expected_steps = (
        1
        if first.runner_kind is RunnerKind.FIXED_L
        else len(operation_keys)
    )
    if operation_keys != build_kvquant_operation_keys(
        configuration=first.configuration,
        runner_kind=first.runner_kind,
        graph_mode=first.graph_mode,
        starting_context=first.historical_context,
        output_steps=expected_steps,
    ):
        raise EndpointSessionError("operation set differs from bounded form")
    if (
        tuple(int(item) for item in prefix_input_ids.shape)
        != (1, first.historical_context)
        or tuple(int(item) for item in decode_input_ids.shape)
        != (1, expected_steps)
        or prefix_input_ids.device != decode_input_ids.device
        or prefix_input_ids.device.type != "cuda"
    ):
        raise EndpointSessionError("endpoint inputs differ from operation set")

    method_config = load_frozen_kvquant_method_config()
    method = build_method_adapter(
        method_config,
        kvquant_runtime_context(first.configuration),
        variant_id=first.configuration,
    )
    if type(method) is not KVQuantMethodAdapter:
        raise EndpointSessionError("factory did not return KVQuant adapter")
    method.prepare_runtime()
    model_memory = capture_cuda_memory_snapshot(
        "model_baseline",
        device=prefix_input_ids.device,
    )
    cache = method.allocate(
        batch_size=1,
        capacity=first.capacity,
        device=prefix_input_ids.device,
        workspace_bytes=0,
    )
    if type(cache) is not KVQuantStaticCache:
        raise EndpointSessionError("cache is not KVQuant static state")
    method.initialize_cache_untimed(cache)
    endpoint = BF16DecodeEndpoint(loaded.model, cache, method)
    cache_memory = capture_cuda_memory_snapshot(
        "post_cache_allocation",
        device=prefix_input_ids.device,
    )
    adapter_fingerprint = method.config_fingerprint(
        cache.layout_fingerprint()
    )
    torch = _torch()
    positions = tuple(
        torch.tensor(
            [first.historical_context + step],
            dtype=torch.long,
            device=prefix_input_ids.device,
        )
        for step in range(expected_steps)
    )
    if first.runner_kind is RunnerKind.FIXED_L:
        cache.bind_fixed_position_tensor_untimed(
            positions[0],
            logical_position=first.historical_context,
        )
    else:
        cache.bind_growing_position_tensors_untimed(
            positions,
            starting_position=first.historical_context,
        )
    rope = tuple(
        endpoint.prepare_position_embeddings(position.unsqueeze(0))
        for position in positions
    )
    tokens = tuple(
        decode_input_ids[:, step : step + 1]
        for step in range(expected_steps)
    )

    def reset_growing() -> None:
        method.reset_cache_untimed(cache)
        endpoint.prefill(prefix_input_ids)
        cache.prepare_growing(first.historical_context, expected_steps)

    fixed_operation: Callable[[], Any] | None = None
    growing_operations: tuple[Callable[[], Any], ...] = ()
    graph: CapturedFixedGraph | None = None
    graph_evidence: Mapping[str, object] | None = None
    graph_comparison: NumericalComparison | None = None
    warmed_outputs: list[tuple[str, bool]] = []
    endpoint.prefill(prefix_input_ids)
    initial_prefix = _historical_prefix_sha256(
        cache,
        first.historical_context,
    )
    if first.runner_kind is RunnerKind.FIXED_L:
        cache.prepare_fixed(first.historical_context)

        def fixed_step() -> Any:
            return endpoint.decode(tokens[0], positions[0], rope[0])

        fixed_operation = fixed_step
        eager_output: Any | None = None
        for _ in range(PHASE11_ENDPOINT_WARMUP_OPERATIONS):
            eager_output = fixed_step()
        if eager_output is None:
            raise EndpointSessionError("fixed warmup produced no output")
        if (
            _historical_prefix_sha256(cache, first.historical_context)
            != initial_prefix
        ):
            raise EndpointSessionError("fixed warmup changed history")
        observed = eager_output.detach().to(device="cpu", copy=True)
        if first.graph_mode is GraphMode.CUDA_GRAPH:
            graph = capture_fixed_graph(
                fixed_step,
                warmup_steps=0,
                device=cache.device,
            )
            pointers_before = _composite_cache_pointers(cache, endpoint)
            first_replay = (
                graph.replay().detach().to(device="cpu", copy=True).clone()
            )
            second_replay = (
                graph.replay().detach().to(device="cpu", copy=True).clone()
            )
            torch.cuda.synchronize(device=cache.device)
            graph_comparison = compare_tensors_untimed(
                first_replay,
                observed,
                atol=PHASE11_DECODE_ATOL,
                rtol=PHASE11_DECODE_RTOL,
            )
            history_stable = (
                _historical_prefix_sha256(cache, first.historical_context)
                == initial_prefix
            )
            pointers_stable = (
                _composite_cache_pointers(cache, endpoint) == pointers_before
            )
            graph_evidence = {
                **graph.to_dict(),
                "consecutive_replay_outputs_exact": bool(
                    torch.equal(first_replay, second_replay)
                ),
                "first_replay_checksum": tensor_sha256_untimed(first_replay),
                "second_replay_checksum": tensor_sha256_untimed(second_replay),
                "cache_pointers_stable": pointers_stable,
                "historical_prefix_unchanged": history_stable,
                "replay_allocation_audited_separately": True,
            }
            if not history_stable or not pointers_stable:
                raise EndpointSessionError(
                    "graph replay changed KVQuant cache identity"
                )
            observed = second_replay
        warmed_outputs.append(
            (
                tensor_sha256_untimed(observed),
                bool(torch.isfinite(observed).all()),
            )
        )
    else:
        cache.prepare_growing(first.historical_context, expected_steps)
        operations: list[Callable[[], Any]] = []
        for step in range(expected_steps):

            def growing_step(step: int = step) -> Any:
                cache.select_growing_step(step)
                return endpoint.decode(tokens[step], positions[step], rope[step])

            operations.append(growing_step)
        growing_operations = tuple(operations)
        for operation in growing_operations:
            observed = operation().detach().to(device="cpu", copy=True)
            warmed_outputs.append(
                (
                    tensor_sha256_untimed(observed),
                    bool(torch.isfinite(observed).all()),
                )
            )
        reset_growing()
        if (
            _historical_prefix_sha256(cache, first.historical_context)
            != initial_prefix
        ):
            raise EndpointSessionError("growing reset changed prefix")
    prefix_sha256 = _historical_prefix_sha256(
        cache,
        first.historical_context,
    )
    return KVQuantEndpointSession(
        loaded=loaded,
        operation_keys=operation_keys,
        endpoint=endpoint,
        cache=cache,
        method=method,
        adapter_config_fingerprint=adapter_fingerprint,
        model_memory=model_memory,
        cache_memory=cache_memory,
        fixed_operation=fixed_operation,
        graph=graph,
        graph_evidence=graph_evidence,
        eager_graph_comparison=graph_comparison,
        growing_operations=growing_operations,
        reset_growing=(
            reset_growing
            if first.runner_kind is RunnerKind.GROWING_CONTEXT
            else None
        ),
        warmed_outputs=tuple(warmed_outputs),
        prefix_sha256=prefix_sha256,
    )
