"""Static native-GQA cache storage for the Phase 11 KVQuant adapter.

This module owns storage and lifecycle bookkeeping only.  Quantization,
packing, sparse selection, and decode remain calls into the checksum-bound
KVQuant CUDA extension.  All CUDA-visible tensors are allocated in the
constructor so store, append, and decode can operate only on caller-owned
buffers.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
from typing import Any, Final

from kvbench.runtime.static_cache import CacheBoundsError, CacheStateError


KVQUANT_CONFIG_BITS: Final[dict[str, int]] = {
    "kvq4": 4,
    "kvq3": 3,
    "kvq2": 2,
}
KVQUANT_NUM_LAYERS: Final[int] = 32
KVQUANT_BATCH_SIZE: Final[int] = 1
KVQUANT_SUPPORTED_BATCH_SIZES: Final[tuple[int, ...]] = (1, 2, 4, 8, 16)
KVQUANT_NUM_QUERY_HEADS: Final[int] = 32
KVQUANT_NUM_KV_HEADS: Final[int] = 8
KVQUANT_HEAD_DIM: Final[int] = 128
KVQUANT_SINK_TOKENS: Final[int] = 5
KVQUANT_KEY_CAP: Final[int] = 12
KVQUANT_VALUE_CAP: Final[int] = 12
KVQUANT_ROPE_DIM: Final[int] = 64
KVQUANT_Q4_VALUE_DECODE_TILE_WIDTH: Final[int] = 128
KVQUANT_Q4_VALUE_DECODE_WORKSPACE_FORMULA_VERSION: Final[str] = (
    "kvbench-kvquant-q4-value-workspace-capacity-v1"
)
KVQUANT_LEGACY_CACHE_IMPLEMENTATION_SHA256: Final[str] = (
    "24e0152347edbfca46ae8d911b424b2d0a1da9237d886c8075b0c071e605ee96"
)
KVQUANT_Q23_VALUE_DECODE_TILE_WIDTH: Final[int] = 128
_TORCH: Any | None = None


def _torch() -> Any:
    global _TORCH
    if _TORCH is None:
        try:
            _TORCH = importlib.import_module("torch")
        except ModuleNotFoundError as error:  # pragma: no cover - environment
            raise CacheStateError(
                "PyTorch is required for KVQuantStaticCache"
            ) from error
    return _TORCH


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _storage_bytes(tensor: Any) -> int:
    return int(tensor.untyped_storage().nbytes())


def kvquant_q4_value_decode_quantized_capacity(
    total_attended_capacity: int,
) -> int:
    """Return the q4 non-sink Value capacity for one static cache."""

    capacity = _positive_int(
        total_attended_capacity,
        "total_attended_capacity",
    )
    return max(0, capacity - KVQUANT_SINK_TOKENS)


def kvquant_q4_value_decode_tile_capacity(
    total_attended_capacity: int,
) -> int:
    """Return the exact 128-token q4 tile capacity without active-state input."""

    quantized = kvquant_q4_value_decode_quantized_capacity(
        total_attended_capacity
    )
    return (
        quantized + KVQUANT_Q4_VALUE_DECODE_TILE_WIDTH - 1
    ) // KVQUANT_Q4_VALUE_DECODE_TILE_WIDTH


def kvquant_q4_value_decode_workspace_shape(
    *,
    batch_size: int,
    total_attended_capacity: int,
) -> tuple[int, int, int, int]:
    """Return the sole Decision 0036 caller-owned q4 workspace shape."""

    batch = _positive_int(batch_size, "batch_size")
    return (
        batch,
        KVQUANT_NUM_QUERY_HEADS,
        kvquant_q4_value_decode_tile_capacity(total_attended_capacity),
        KVQUANT_HEAD_DIM,
    )


def kvquant_q4_value_decode_workspace_bytes(
    *,
    batch_size: int,
    total_attended_capacity: int,
) -> int:
    """Return exact FP32 bytes for the Decision 0036 workspace."""

    shape = kvquant_q4_value_decode_workspace_shape(
        batch_size=batch_size,
        total_attended_capacity=total_attended_capacity,
    )
    return math.prod(shape) * 4


@dataclass(frozen=True, slots=True)
class KVQuantStaticCacheAccounting:
    """Exact physical accounting for one preallocated KVQuant cache."""

    predicted_tensor_bytes: int
    measured_tensor_bytes: int
    allocated_bytes: int
    padding_bytes: int
    staging_bytes: int
    workspace_bytes: int
    temporary_peak_bytes: int
    capacity: int
    active_context: int

    @property
    def relative_error(self) -> float:
        if self.allocated_bytes == 0:
            return 0.0
        return (
            abs(self.predicted_tensor_bytes - self.allocated_bytes)
            / self.allocated_bytes
        )

    def to_dict(self) -> dict[str, int | float]:
        return {
            "predicted_tensor_bytes": self.predicted_tensor_bytes,
            "measured_tensor_bytes": self.measured_tensor_bytes,
            "allocated_bytes": self.allocated_bytes,
            "padding_bytes": self.padding_bytes,
            "staging_bytes": self.staging_bytes,
            "workspace_bytes": self.workspace_bytes,
            "temporary_peak_bytes": self.temporary_peak_bytes,
            "capacity": self.capacity,
            "active_context": self.active_context,
            "relative_error": self.relative_error,
        }


@dataclass(frozen=True, slots=True)
class KVQuantAllocationRatios:
    """Canonical allocated-storage ratios for the frozen BF16 comparison."""

    allocated_bytes: int
    bf16_allocated_bytes: int
    rho_alloc: float
    r_alloc: float
    reciprocal_error: float
    r_hbm: None = None

    @classmethod
    def from_bytes(
        cls,
        *,
        allocated_bytes: int,
        bf16_allocated_bytes: int,
    ) -> "KVQuantAllocationRatios":
        allocated = _positive_int(allocated_bytes, "allocated_bytes")
        logical = _positive_int(bf16_allocated_bytes, "bf16_allocated_bytes")
        rho = allocated / logical
        reciprocal = logical / allocated
        return cls(
            allocated_bytes=allocated,
            bf16_allocated_bytes=logical,
            rho_alloc=rho,
            r_alloc=reciprocal,
            reciprocal_error=abs(rho * reciprocal - 1.0),
        )

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "allocated_bytes": self.allocated_bytes,
            "bf16_allocated_bytes": self.bf16_allocated_bytes,
            "rho_alloc": self.rho_alloc,
            "r_alloc": self.r_alloc,
            "reciprocal_error": self.reciprocal_error,
            "r_hbm": self.r_hbm,
        }


class KVQuantStaticCache:
    """One fixed-shape KVQuant cache for the frozen Llama-3.1 GQA geometry."""

    def __init__(
        self,
        *,
        config_name: str,
        num_layers: int,
        batch_size: int,
        num_query_heads: int,
        num_kv_heads: int,
        capacity: int,
        head_dim: int,
        device: Any,
        workspace_bytes: int = 0,
    ) -> None:
        torch = _torch()
        if config_name not in KVQUANT_CONFIG_BITS:
            raise ValueError("unsupported KVQuant configuration")
        geometry = (
            _positive_int(num_layers, "num_layers"),
            _positive_int(batch_size, "batch_size"),
            _positive_int(num_query_heads, "num_query_heads"),
            _positive_int(num_kv_heads, "num_kv_heads"),
            _positive_int(capacity, "capacity"),
            _positive_int(head_dim, "head_dim"),
        )
        if (
            geometry[0] != KVQUANT_NUM_LAYERS
            or geometry[1] not in KVQUANT_SUPPORTED_BATCH_SIZES
            or geometry[2:4]
            != (KVQUANT_NUM_QUERY_HEADS, KVQUANT_NUM_KV_HEADS)
            or geometry[5] != KVQUANT_HEAD_DIM
        ):
            raise ValueError(
                "KVQuant cache requires admitted layers=32 B in {1,2,4,8,16} "
                "H_Q=32 H_KV=8 D=128 geometry"
            )
        if geometry[4] < KVQUANT_SINK_TOKENS:
            raise ValueError("KVQuant capacity must include all five sink tokens")

        self.external_workspace_bytes = _nonnegative_int(
            workspace_bytes, "workspace_bytes"
        )
        self.config_name = config_name
        self.bits = KVQUANT_CONFIG_BITS[config_name]
        self.levels = 1 << self.bits
        self.packed_rows = self.bits * KVQUANT_HEAD_DIM // 32
        self.num_layers = geometry[0]
        self.batch_size = geometry[1]
        self.num_query_heads = geometry[2]
        self.num_kv_heads = geometry[3]
        self.capacity = geometry[4]
        self.head_dim = geometry[5]
        self.sink_tokens = KVQUANT_SINK_TOKENS
        self.key_cap = KVQUANT_KEY_CAP
        self.value_cap = KVQUANT_VALUE_CAP
        self.device = torch.device(device)
        self.interface_dtype = torch.bfloat16
        self.sink_dtype = torch.float16

        # Source-compatible dense storage.  The CUDA ABI keeps the full
        # declared width; sink positions remain unused here and live only in
        # the separate full-precision sink buffers.
        dense_shape = (
            self.num_layers,
            self.batch_size,
            self.num_kv_heads,
            self.packed_rows,
            self.capacity,
        )
        self.packed_key_cache = torch.zeros(
            dense_shape, dtype=torch.int32, device=self.device
        )
        self.packed_value_cache = torch.zeros(
            dense_shape, dtype=torch.int32, device=self.device
        )

        # Frozen Key metadata is copied into these tensors during setup.
        self.key_codebook = torch.empty(
            (self.num_layers, self.levels),
            dtype=torch.float32,
            device=self.device,
        )
        self.key_lookup_table = torch.empty(
            (
                self.num_layers,
                self.num_kv_heads,
                self.head_dim,
                self.levels,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        key_channel_shape = (
            self.num_layers,
            self.num_kv_heads,
            self.head_dim,
        )
        self.key_lower_threshold = torch.empty(
            key_channel_shape, dtype=torch.float32, device=self.device
        )
        self.key_upper_threshold = torch.empty_like(self.key_lower_threshold)
        self.key_zero_point = torch.empty_like(self.key_lower_threshold)
        self.rope_inv_freq = torch.empty(
            (KVQUANT_ROPE_DIM,), dtype=torch.float32, device=self.device
        )

        # Value lookup rows are per-token metadata written in place.
        self.value_codebook = torch.empty(
            (self.num_layers, self.levels),
            dtype=torch.float32,
            device=self.device,
        )
        self.value_lookup_cache = torch.zeros(
            (
                self.num_layers,
                self.batch_size,
                self.capacity,
                self.levels,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        sparse_shape = (
            self.num_layers,
            self.batch_size,
            self.capacity,
            self.key_cap,
        )
        self.key_sparse_values = torch.zeros(
            sparse_shape, dtype=torch.float32, device=self.device
        )
        self.key_sparse_indices = torch.zeros(
            sparse_shape, dtype=torch.int32, device=self.device
        )
        self.value_sparse_values = torch.zeros(
            sparse_shape, dtype=torch.float32, device=self.device
        )
        self.value_sparse_indices = torch.zeros(
            sparse_shape, dtype=torch.int32, device=self.device
        )
        count_shape = (
            self.num_layers,
            self.batch_size,
            self.capacity,
        )
        self.key_active_counts = torch.zeros(
            count_shape, dtype=torch.int32, device=self.device
        )
        self.value_active_counts = torch.zeros(
            count_shape, dtype=torch.int32, device=self.device
        )
        self.sink_position_mask = torch.zeros(
            (self.capacity,), dtype=torch.bool, device=self.device
        )
        self.sink_position_mask[: self.sink_tokens].fill_(True)

        self.sink_key = torch.zeros(
            (
                self.num_layers,
                self.batch_size,
                self.num_kv_heads,
                self.head_dim,
                self.sink_tokens,
            ),
            dtype=self.sink_dtype,
            device=self.device,
        )
        self.sink_value = torch.zeros(
            (
                self.num_layers,
                self.batch_size,
                self.num_kv_heads,
                self.sink_tokens,
                self.head_dim,
            ),
            dtype=self.sink_dtype,
            device=self.device,
        )

        # One-token conversion and selector staging shared across sequential
        # layers.  The extension always sees native flattened KV width 1024.
        kv_token_shape = (
            self.batch_size,
            self.num_kv_heads,
            1,
            self.head_dim,
        )
        self.key_pre_rope_bf16_staging = torch.empty(
            kv_token_shape, dtype=self.interface_dtype, device=self.device
        )
        self.key_attention_bf16_staging = torch.empty_like(
            self.key_pre_rope_bf16_staging
        )
        self.value_bf16_staging = torch.empty_like(
            self.key_pre_rope_bf16_staging
        )
        self.key_float_staging = torch.empty(
            (1, self.num_kv_heads * self.head_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.value_float_staging = torch.empty_like(self.key_float_staging)
        self.key_normalized_staging = torch.empty_like(self.key_float_staging)
        self.key_rescaled_staging = torch.empty_like(self.key_float_staging)

        query_shape = (
            self.batch_size,
            self.num_query_heads,
            self.head_dim,
        )
        self.query_bf16_staging = torch.empty(
            query_shape, dtype=self.interface_dtype, device=self.device
        )
        self.query_float_staging = torch.empty(
            query_shape, dtype=torch.float32, device=self.device
        )
        self.output_bf16_staging = torch.empty(
            query_shape, dtype=self.interface_dtype, device=self.device
        )

        self.selector_values = torch.zeros(
            (1, self.key_cap), dtype=torch.float32, device=self.device
        )
        self.selector_indices = torch.zeros(
            (1, self.key_cap), dtype=torch.int32, device=self.device
        )
        self.selector_count = torch.zeros(
            (1,), dtype=torch.int32, device=self.device
        )
        self.selector_dense_lower = torch.zeros(
            (1,), dtype=torch.float32, device=self.device
        )
        self.selector_dense_upper = torch.zeros_like(self.selector_dense_lower)
        self.selector_sink_mask = torch.zeros(
            (1,), dtype=torch.bool, device=self.device
        )
        self.dummy_thresholds = torch.zeros(
            (self.num_kv_heads * self.head_dim,),
            dtype=torch.float32,
            device=self.device,
        )
        self.key_selector_lower = torch.full(
            (self.num_kv_heads * self.head_dim,),
            -1.0,
            dtype=torch.float32,
            device=self.device,
        )
        self.key_selector_upper = torch.ones(
            (self.num_kv_heads * self.head_dim,),
            dtype=torch.float32,
            device=self.device,
        )

        self.value_metadata_row_staging = torch.empty(
            (self.levels,), dtype=torch.float32, device=self.device
        )
        self.value_scale_staging = torch.empty(
            (1,), dtype=torch.float32, device=self.device
        )
        self.value_offset_staging = torch.empty_like(self.value_scale_staging)
        self.value_zero_point_staging = torch.empty_like(
            self.value_scale_staging
        )
        self.value_lower_bound_staging = torch.empty_like(
            self.value_scale_staging
        )
        self.value_upper_bound_staging = torch.empty_like(
            self.value_scale_staging
        )
        # Store-time bounds remain caller-owned so the exact upstream
        # parallel Value pack can consume the full prefix without allocating
        # threshold outputs.  Only the declared prefix slice is live.
        self.value_store_lower_bounds = torch.zeros(
            (self.batch_size, self.capacity),
            dtype=torch.float32,
            device=self.device,
        )
        self.value_store_upper_bounds = torch.zeros_like(
            self.value_store_lower_bounds
        )
        self.fixed_negative_one = torch.full(
            (1,), -1.0, dtype=torch.float32, device=self.device
        )
        self.fixed_positive_one = torch.ones(
            (1,), dtype=torch.float32, device=self.device
        )
        self.fixed_zero = torch.zeros(
            (1,), dtype=torch.float32, device=self.device
        )
        self.position_id_staging = torch.zeros(
            (1,), dtype=torch.int64, device=self.device
        )

        # Direct compressed-cache decode workspaces.  Query-head dimensions
        # are confined to legitimate Q/logit/output tensors, never K/V state.
        logits_shape = (
            self.batch_size,
            self.num_query_heads,
            self.capacity,
        )
        self.decode_logits = torch.zeros(
            logits_shape, dtype=torch.float32, device=self.device
        )
        self.decode_logits_bf16 = torch.zeros(
            logits_shape, dtype=torch.bfloat16, device=self.device
        )
        self.decode_softmax = torch.empty_like(self.decode_logits)
        self.sink_logits_fp16 = torch.zeros(
            (
                self.batch_size,
                self.num_query_heads,
                self.sink_tokens,
            ),
            dtype=torch.float16,
            device=self.device,
        )
        self.decode_merge = torch.zeros(
            query_shape, dtype=torch.float32, device=self.device
        )
        self.decode_sparse_correction = torch.zeros_like(self.decode_merge)
        self.decode_sink_contribution = torch.zeros_like(self.decode_merge)
        self.decode_quantized_output = torch.zeros_like(self.decode_merge)
        self.sink_output_fp16 = torch.zeros(
            query_shape, dtype=torch.float16, device=self.device
        )
        self.q4_value_decode_quantized_capacity = (
            kvquant_q4_value_decode_quantized_capacity(self.capacity)
            if self.config_name == "kvq4"
            else None
        )
        self.q4_value_decode_tile_capacity = (
            kvquant_q4_value_decode_tile_capacity(self.capacity)
            if self.config_name == "kvq4"
            else None
        )
        self.q4_value_decode_workspace_shape = (
            kvquant_q4_value_decode_workspace_shape(
                batch_size=self.batch_size,
                total_attended_capacity=self.capacity,
            )
            if self.config_name == "kvq4"
            else None
        )
        self.q4_value_decode_workspace_bytes = (
            kvquant_q4_value_decode_workspace_bytes(
                batch_size=self.batch_size,
                total_attended_capacity=self.capacity,
            )
            if self.config_name == "kvq4"
            else 0
        )
        self.q4_value_decode_workspace = (
            torch.empty(
                self.q4_value_decode_workspace_shape,
                dtype=torch.float32,
                device=self.device,
            )
            if self.q4_value_decode_workspace_shape is not None
            else None
        )
        q23_max_quantized = max(self.capacity - self.sink_tokens, 0)
        q23_max_tiles = (
            q23_max_quantized + KVQUANT_Q23_VALUE_DECODE_TILE_WIDTH - 1
        ) // KVQUANT_Q23_VALUE_DECODE_TILE_WIDTH
        q23_stored_tiles = max(q23_max_tiles - 1, 0)
        q23_workspace_elements = (
            q23_stored_tiles * KVQUANT_Q23_VALUE_DECODE_TILE_WIDTH
        )
        self.q23_value_decode_workspace = (
            self.decode_logits[:, :, :q23_workspace_elements].view(
                self.batch_size,
                self.num_query_heads,
                q23_stored_tiles,
                self.head_dim,
            )
            if self.config_name in {"kvq3", "kvq2"}
            else None
        )
        self.reserved_workspace = torch.empty(
            (self.external_workspace_bytes,),
            dtype=torch.uint8,
            device=self.device,
        )

        self._active_context = 0
        self._mode = "idle"
        self._declared_prefill_length = 0
        self._prefix_length = 0
        self._output_steps = 0
        self._growing_step = -1
        self._fixed_position_binding: tuple[int, int] | None = None
        self._growing_position_bindings: tuple[tuple[int, int], ...] = ()
        self._expected_decode_position_binding: tuple[int, int] | None = None
        self._known_key_active_entries: int | None = 0

    @property
    def active_context(self) -> int:
        return self._active_context

    @property
    def maximum_context(self) -> int:
        return self.capacity

    @property
    def quantized_active_context(self) -> int:
        return max(0, self.active_context - self.sink_tokens)

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def logical_bf16_storage_bytes(self) -> int:
        return self._logical_bf16_bytes(self.capacity)

    @property
    def r_hbm(self) -> None:
        """Phase 11 does not infer physical HBM traffic."""

        return None

    def q4_value_decode_workspace_geometry(self) -> dict[str, object]:
        """Return the exact Decision 0036 geometry for manifests and audits."""

        if self.config_name != "kvq4":
            raise CacheStateError("q4 workspace geometry requires kvq4")
        return {
            "formula_version": (
                KVQUANT_Q4_VALUE_DECODE_WORKSPACE_FORMULA_VERSION
            ),
            "tile_size": KVQUANT_Q4_VALUE_DECODE_TILE_WIDTH,
            "tile_capacity": self.q4_value_decode_tile_capacity,
            "total_attended_capacity": self.capacity,
            "quantized_value_capacity": (
                self.q4_value_decode_quantized_capacity
            ),
            "workspace_shape": list(self.q4_value_decode_workspace_shape or ()),
            "workspace_bytes": self.q4_value_decode_workspace_bytes,
            "dtype": "float32",
            "allocation_time": "cache_state_construction",
            "resize_during_decode": False,
            "counted_as_cache_payload": False,
        }

    def require_q4_value_decode_workspace(
        self,
        *,
        quantized_length: int,
    ) -> Any:
        """Validate and return the preallocated q4 workspace without resizing."""

        torch = _torch()
        quantized = _nonnegative_int(quantized_length, "quantized_length")
        workspace = self.q4_value_decode_workspace
        expected_shape = self.q4_value_decode_workspace_shape
        if (
            self.config_name != "kvq4"
            or workspace is None
            or expected_shape is None
            or quantized > int(self.q4_value_decode_quantized_capacity or 0)
            or kvquant_q4_value_decode_tile_capacity(
                quantized + self.sink_tokens
            )
            > int(self.q4_value_decode_tile_capacity or 0)
            or tuple(int(item) for item in workspace.shape) != expected_shape
            or workspace.dtype != torch.float32
            or workspace.device != self.device
            or not workspace.is_contiguous()
            or _storage_bytes(workspace)
            != self.q4_value_decode_workspace_bytes
        ):
            raise CacheStateError(
                "KVQuant q4 deterministic workspace capacity differs"
            )
        return workspace

    def _check_length(self, length: int, *, allow_zero: bool = False) -> int:
        minimum = 0 if allow_zero else 1
        if (
            isinstance(length, bool)
            or not isinstance(length, int)
            or length < minimum
            or length > self.capacity
        ):
            raise CacheBoundsError(
                f"length must be between {minimum} and capacity {self.capacity}"
            )
        return length

    def _check_layer(self, layer_idx: int) -> int:
        if isinstance(layer_idx, bool) or not isinstance(layer_idx, int):
            raise CacheBoundsError("layer index must be an integer")
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise CacheBoundsError("layer index is outside the static cache")
        return layer_idx

    def _payload_slot(self, position: int) -> int:
        if (
            isinstance(position, bool)
            or not isinstance(position, int)
            or position < self.sink_tokens
            or position >= self.capacity
        ):
            raise CacheBoundsError(
                "quantized position is outside the non-sink cache capacity"
            )
        return position - self.sink_tokens

    def _position_tensor_pointer(self, cache_position: Any) -> int:
        torch = _torch()
        if (
            tuple(int(item) for item in cache_position.shape) != (1,)
            or cache_position.dtype != torch.int64
            or cache_position.device != self.device
            or not cache_position.is_contiguous()
        ):
            raise CacheStateError(
                "cache position tensor differs from frozen scalar layout"
            )
        pointer = cache_position.data_ptr()
        if type(pointer) is not int or pointer <= 0:
            raise CacheStateError("cache position tensor has no stable pointer")
        return pointer

    def bind_fixed_position_tensor_untimed(
        self,
        cache_position: Any,
        *,
        logical_position: int,
    ) -> None:
        """Bind one precreated fixed-L position tensor before measured use."""

        self._payload_slot(logical_position)
        binding = (
            self._position_tensor_pointer(cache_position),
            logical_position,
        )
        if (
            self._fixed_position_binding is not None
            and self._fixed_position_binding != binding
        ):
            raise CacheStateError("fixed cache position binding changed")
        self._fixed_position_binding = binding

    def bind_growing_position_tensors_untimed(
        self,
        cache_positions: tuple[Any, ...],
        *,
        starting_position: int,
    ) -> None:
        """Bind the complete precreated growing trajectory before measurement."""

        if not cache_positions:
            raise CacheStateError("growing cache position binding is empty")
        bindings = tuple(
            (
                self._position_tensor_pointer(cache_position),
                starting_position + step,
            )
            for step, cache_position in enumerate(cache_positions)
        )
        for _, logical_position in bindings:
            self._payload_slot(logical_position)
        if len({pointer for pointer, _ in bindings}) != len(bindings):
            raise CacheStateError("growing cache position pointers overlap")
        if (
            self._growing_position_bindings
            and self._growing_position_bindings != bindings
        ):
            raise CacheStateError("growing cache position binding changed")
        self._growing_position_bindings = bindings

    def validate_decode_position_binding(
        self,
        cache_position: Any,
        *,
        payload_slot: int,
    ) -> None:
        """Fail closed on pointer/slot drift without reading a CUDA scalar."""

        binding = self._expected_decode_position_binding
        if binding is None:
            raise CacheStateError("decode cache position is not statically bound")
        pointer, logical_position = binding
        if (
            self._position_tensor_pointer(cache_position) != pointer
            or self._payload_slot(logical_position) != payload_slot
        ):
            raise CacheStateError(
                "decode cache position differs from the physical slot"
            )

    def payload_slot_for_position(self, position: int) -> int:
        """Map one absolute non-sink position to the source cache column."""

        return self._payload_slot(position)

    def is_sink_position(self, position: int) -> bool:
        if (
            isinstance(position, bool)
            or not isinstance(position, int)
            or position < 0
            or position >= self.capacity
        ):
            raise CacheBoundsError("position is outside the static cache")
        return position < self.sink_tokens

    def _logical_bf16_bytes(self, context: int) -> int:
        checked = self._check_length(context, allow_zero=True)
        return (
            2
            * self.num_layers
            * self.batch_size
            * self.num_kv_heads
            * checked
            * self.head_dim
            * 2
        )

    def _owned_named_tensors(self) -> tuple[tuple[str, Any], ...]:
        tensors = (
            ("packed_key_cache", self.packed_key_cache),
            ("packed_value_cache", self.packed_value_cache),
            ("key_codebook", self.key_codebook),
            ("key_lookup_table", self.key_lookup_table),
            ("key_lower_threshold", self.key_lower_threshold),
            ("key_upper_threshold", self.key_upper_threshold),
            ("key_zero_point", self.key_zero_point),
            ("rope_inv_freq", self.rope_inv_freq),
            ("value_codebook", self.value_codebook),
            ("value_lookup_cache", self.value_lookup_cache),
            ("key_sparse_values", self.key_sparse_values),
            ("key_sparse_indices", self.key_sparse_indices),
            ("value_sparse_values", self.value_sparse_values),
            ("value_sparse_indices", self.value_sparse_indices),
            ("key_active_counts", self.key_active_counts),
            ("value_active_counts", self.value_active_counts),
            ("sink_position_mask", self.sink_position_mask),
            ("sink_key", self.sink_key),
            ("sink_value", self.sink_value),
            ("key_pre_rope_bf16_staging", self.key_pre_rope_bf16_staging),
            ("key_attention_bf16_staging", self.key_attention_bf16_staging),
            ("value_bf16_staging", self.value_bf16_staging),
            ("key_float_staging", self.key_float_staging),
            ("value_float_staging", self.value_float_staging),
            ("key_normalized_staging", self.key_normalized_staging),
            ("key_rescaled_staging", self.key_rescaled_staging),
            ("query_bf16_staging", self.query_bf16_staging),
            ("query_float_staging", self.query_float_staging),
            ("output_bf16_staging", self.output_bf16_staging),
            ("selector_values", self.selector_values),
            ("selector_indices", self.selector_indices),
            ("selector_count", self.selector_count),
            ("selector_dense_lower", self.selector_dense_lower),
            ("selector_dense_upper", self.selector_dense_upper),
            ("selector_sink_mask", self.selector_sink_mask),
            ("dummy_thresholds", self.dummy_thresholds),
            ("key_selector_lower", self.key_selector_lower),
            ("key_selector_upper", self.key_selector_upper),
            ("value_metadata_row_staging", self.value_metadata_row_staging),
            ("value_scale_staging", self.value_scale_staging),
            ("value_offset_staging", self.value_offset_staging),
            ("value_zero_point_staging", self.value_zero_point_staging),
            ("value_lower_bound_staging", self.value_lower_bound_staging),
            ("value_upper_bound_staging", self.value_upper_bound_staging),
            ("value_store_lower_bounds", self.value_store_lower_bounds),
            ("value_store_upper_bounds", self.value_store_upper_bounds),
            ("fixed_negative_one", self.fixed_negative_one),
            ("fixed_positive_one", self.fixed_positive_one),
            ("fixed_zero", self.fixed_zero),
            ("position_id_staging", self.position_id_staging),
            ("decode_logits", self.decode_logits),
            ("decode_logits_bf16", self.decode_logits_bf16),
            ("decode_softmax", self.decode_softmax),
            ("sink_logits_fp16", self.sink_logits_fp16),
            ("decode_merge", self.decode_merge),
            ("decode_sparse_correction", self.decode_sparse_correction),
            ("decode_sink_contribution", self.decode_sink_contribution),
            ("decode_quantized_output", self.decode_quantized_output),
            ("sink_output_fp16", self.sink_output_fp16),
        )
        if self.q4_value_decode_workspace is not None:
            tensors += (
                (
                    "q4_value_decode_workspace",
                    self.q4_value_decode_workspace,
                ),
            )
        return (*tensors, ("reserved_workspace", self.reserved_workspace))

    def _owned_tensors(self) -> tuple[Any, ...]:
        return tuple(tensor for _, tensor in self._owned_named_tensors())

    def byte_breakdown(self) -> dict[str, int]:
        """Return an exact, mutually exclusive physical byte breakdown."""

        key_metadata = sum(
            _storage_bytes(tensor)
            for tensor in (
                self.key_codebook,
                self.key_lookup_table,
                self.key_lower_threshold,
                self.key_upper_threshold,
                self.key_zero_point,
                self.rope_inv_freq,
            )
        )
        value_metadata = sum(
            _storage_bytes(tensor)
            for tensor in (self.value_codebook, self.value_lookup_cache)
        )
        count_mask = sum(
            _storage_bytes(tensor)
            for tensor in (
                self.key_active_counts,
                self.value_active_counts,
                self.sink_position_mask,
            )
        )
        staging = sum(
            _storage_bytes(tensor)
            for tensor in (
                self.key_pre_rope_bf16_staging,
                self.key_attention_bf16_staging,
                self.value_bf16_staging,
                self.key_float_staging,
                self.value_float_staging,
                self.key_normalized_staging,
                self.key_rescaled_staging,
                self.query_bf16_staging,
                self.query_float_staging,
                self.output_bf16_staging,
                self.selector_values,
                self.selector_indices,
                self.selector_count,
                self.selector_dense_lower,
                self.selector_dense_upper,
                self.selector_sink_mask,
                self.dummy_thresholds,
                self.key_selector_lower,
                self.key_selector_upper,
                self.value_metadata_row_staging,
                self.value_scale_staging,
                self.value_offset_staging,
                self.value_zero_point_staging,
                self.value_lower_bound_staging,
                self.value_upper_bound_staging,
                self.value_store_lower_bounds,
                self.value_store_upper_bounds,
                self.fixed_negative_one,
                self.fixed_positive_one,
                self.fixed_zero,
                self.position_id_staging,
            )
        )
        workspace = sum(
            _storage_bytes(tensor)
            for tensor in (
                self.decode_logits,
                self.decode_logits_bf16,
                self.decode_softmax,
                self.sink_logits_fp16,
                self.decode_merge,
                self.decode_sparse_correction,
                self.decode_sink_contribution,
                self.decode_quantized_output,
                self.sink_output_fp16,
                self.reserved_workspace,
            )
        )
        breakdown = {
            "dense_k_payload": _storage_bytes(self.packed_key_cache),
            "dense_v_payload": _storage_bytes(self.packed_value_cache),
            "key_metadata": key_metadata,
            "value_metadata": value_metadata,
            "key_sparse_values": _storage_bytes(self.key_sparse_values),
            "key_sparse_indices": _storage_bytes(self.key_sparse_indices),
            "value_sparse_values": _storage_bytes(self.value_sparse_values),
            "value_sparse_indices": _storage_bytes(self.value_sparse_indices),
            "active_count_mask": count_mask,
            "sink_k": _storage_bytes(self.sink_key),
            "sink_v": _storage_bytes(self.sink_value),
            "staging": staging,
            "padding_alignment": 0,
            "persistent_workspace": workspace,
        }
        if self.q4_value_decode_workspace is not None:
            breakdown["q4_value_decode_workspace"] = _storage_bytes(
                self.q4_value_decode_workspace
            )
        owned = sum(_storage_bytes(tensor) for tensor in self._owned_tensors())
        if sum(breakdown.values()) != owned:
            raise CacheStateError("KVQuant persistent byte breakdown is not exact")
        return breakdown

    def predicted_byte_breakdown(self) -> dict[str, int]:
        """Independent shape/dtype formula for the physical allocation."""

        layers = self.num_layers
        heads = self.num_kv_heads
        dimension = self.head_dim
        capacity = self.capacity
        levels = self.levels
        query_elements = self.batch_size * self.num_query_heads * dimension
        kv_elements = self.batch_size * heads * dimension
        dense_bytes = (
            layers
            * self.batch_size
            * heads
            * self.packed_rows
            * capacity
            * 4
        )
        key_metadata = (
            layers * levels * 4
            + layers * heads * dimension * levels * 4
            + 3 * layers * heads * dimension * 4
            + KVQUANT_ROPE_DIM * 4
        )
        value_metadata = (
            layers * levels * 4
            + layers * self.batch_size * capacity * levels * 4
        )
        sparse_bytes = (
            layers * self.batch_size * capacity * self.key_cap * 4
        )
        count_mask = (
            2 * layers * self.batch_size * capacity * 4 + capacity
        )
        sink_bytes = (
            layers
            * self.batch_size
            * heads
            * dimension
            * self.sink_tokens
            * 2
        )
        staging = (
            3 * kv_elements * 2
            + 4 * heads * dimension * 4
            + query_elements * 2
            + query_elements * 4
            + query_elements * 2
            + self.key_cap * 4
            + self.key_cap * 4
            + 3 * 4
            + 1
            + heads * dimension * 4
            + 2 * heads * dimension * 4
            + levels * 4
            + 5 * 4
            + 2 * self.batch_size * capacity * 4
            + 3 * 4
            + 8
        )
        workspace = (
            2 * self.batch_size * self.num_query_heads * capacity * 4
            + self.batch_size * self.num_query_heads * capacity * 2
            + self.batch_size
            * self.num_query_heads
            * self.sink_tokens
            * 2
            + 4 * query_elements * 4
            + query_elements * 2
            + self.external_workspace_bytes
        )
        breakdown = {
            "dense_k_payload": dense_bytes,
            "dense_v_payload": dense_bytes,
            "key_metadata": key_metadata,
            "value_metadata": value_metadata,
            "key_sparse_values": sparse_bytes,
            "key_sparse_indices": sparse_bytes,
            "value_sparse_values": sparse_bytes,
            "value_sparse_indices": sparse_bytes,
            "active_count_mask": count_mask,
            "sink_k": sink_bytes,
            "sink_v": sink_bytes,
            "staging": staging,
            "padding_alignment": 0,
            "persistent_workspace": workspace,
        }
        if self.config_name == "kvq4":
            breakdown["q4_value_decode_workspace"] = (
                self.q4_value_decode_workspace_bytes
            )
        return breakdown

    def accounting(self) -> KVQuantStaticCacheAccounting:
        breakdown = self.byte_breakdown()
        predicted = sum(self.predicted_byte_breakdown().values())
        allocated = sum(breakdown.values())
        accounting = KVQuantStaticCacheAccounting(
            predicted_tensor_bytes=predicted,
            measured_tensor_bytes=allocated,
            allocated_bytes=allocated,
            padding_bytes=breakdown["padding_alignment"],
            staging_bytes=breakdown["staging"],
            workspace_bytes=(
                breakdown["persistent_workspace"]
                + breakdown.get("q4_value_decode_workspace", 0)
            ),
            temporary_peak_bytes=0,
            capacity=self.capacity,
            active_context=self.active_context,
        )
        if accounting.relative_error >= 0.01:
            raise CacheStateError(
                "KVQuant predicted allocation differs by at least 1%"
            )
        return accounting

    def ratios(self) -> KVQuantAllocationRatios:
        return KVQuantAllocationRatios.from_bytes(
            allocated_bytes=self.accounting().allocated_bytes,
            bf16_allocated_bytes=self.logical_bf16_storage_bytes,
        )

    def record_key_active_entries(self, total_entries: int) -> None:
        """Record a post-measurement count for exact logical accounting.

        The caller obtains this value outside the measured region.  This
        method never reads a CUDA scalar and is not part of graph replay.
        """

        nonsink_rows = (
            self.num_layers
            * self.batch_size
            * max(0, self.active_context - self.sink_tokens)
        )
        total = _nonnegative_int(total_entries, "total_entries")
        if total > nonsink_rows * self.key_cap:
            raise CacheBoundsError("Key active entries exceed fixed sparse capacity")
        self._known_key_active_entries = total

    def active_byte_breakdown(
        self,
        active_context: int | None = None,
        *,
        key_active_entries: int | None = None,
    ) -> dict[str, int]:
        """Return exact logically occupied cache bytes at one context."""

        context = (
            self.active_context
            if active_context is None
            else self._check_length(active_context, allow_zero=True)
        )
        sink = min(context, self.sink_tokens)
        nonsink = max(0, context - self.sink_tokens)
        rows = self.num_layers * self.batch_size * nonsink
        if key_active_entries is None:
            if context == self.active_context:
                key_active_entries = self._known_key_active_entries
            elif nonsink == 0:
                key_active_entries = 0
        if key_active_entries is None:
            raise CacheStateError(
                "exact active Key sparse count is required for logical accounting"
            )
        key_entries = _nonnegative_int(
            key_active_entries, "key_active_entries"
        )
        if key_entries > rows * self.key_cap:
            raise CacheBoundsError("Key active entries exceed fixed sparse capacity")
        dense = (
            self.num_layers
            * self.batch_size
            * self.num_kv_heads
            * self.head_dim
            * nonsink
            * self.bits
            // 8
        )
        key_metadata = self.predicted_byte_breakdown()["key_metadata"]
        value_metadata = (
            self.num_layers * self.levels * 4
            + rows * self.levels * 4
        )
        sink_bytes = (
            self.num_layers
            * self.batch_size
            * self.num_kv_heads
            * self.head_dim
            * sink
            * 2
        )
        return {
            "dense_k_payload": dense,
            "dense_v_payload": dense,
            "key_metadata": key_metadata,
            "value_metadata": value_metadata,
            "key_sparse_values": key_entries * 4,
            "key_sparse_indices": key_entries * 4,
            "value_sparse_values": rows * self.value_cap * 4,
            "value_sparse_indices": rows * self.value_cap * 4,
            "active_count_mask": rows * 2 * 4 + context,
            "sink_k": sink_bytes,
            "sink_v": sink_bytes,
            "padding_alignment": 0,
        }

    def active_storage_bytes(
        self,
        active_context: int | None = None,
        *,
        key_active_entries: int | None = None,
    ) -> int:
        return sum(
            self.active_byte_breakdown(
                active_context,
                key_active_entries=key_active_entries,
            ).values()
        )

    def active_logical_bf16_bytes(
        self, active_context: int | None = None
    ) -> int:
        context = self.active_context if active_context is None else active_context
        return self._logical_bf16_bytes(context)

    def pointers(self) -> dict[str, int]:
        """Return audit-only pointer identities outside measured execution."""

        pointers = {
            f"{name}_data_ptr": int(tensor.data_ptr())
            for name, tensor in self._owned_named_tensors()
        }
        if self.q23_value_decode_workspace is not None:
            pointers["q23_value_decode_workspace_data_ptr"] = int(
                self.q23_value_decode_workspace.data_ptr()
            )
        return pointers

    def layout_fingerprint(self) -> str:
        payload = {
            "schema": (
                "kvbench-kvquant-static-cache-layout-1.4.0"
                if self.config_name == "kvq4"
                else "kvbench-kvquant-static-cache-layout-1.3.0"
            ),
            "configuration": self.config_name,
            "bits": self.bits,
            "levels": self.levels,
            "num_layers": self.num_layers,
            "batch_size": self.batch_size,
            "num_query_heads": self.num_query_heads,
            "num_kv_heads": self.num_kv_heads,
            "gqa_groups": self.num_query_heads // self.num_kv_heads,
            "head_dim": self.head_dim,
            "capacity": self.capacity,
            "sink_tokens": self.sink_tokens,
            "key_cap": self.key_cap,
            "value_cap": self.value_cap,
            "packed_rows": self.packed_rows,
            "dense_shape": tuple(self.packed_key_cache.shape),
            "sparse_shape": tuple(self.key_sparse_values.shape),
            "sink_key_shape": tuple(self.sink_key.shape),
            "sink_value_shape": tuple(self.sink_value.shape),
            "value_store_bounds_shape": tuple(
                self.value_store_lower_bounds.shape
            ),
            "decode_logits_bf16_shape": tuple(self.decode_logits_bf16.shape),
            "sink_logits_fp16_shape": tuple(self.sink_logits_fp16.shape),
            "sink_output_fp16_shape": tuple(self.sink_output_fp16.shape),
            "q4_value_decode_workspace_shape": (
                None
                if self.q4_value_decode_workspace is None
                else tuple(self.q4_value_decode_workspace.shape)
            ),
            "q4_value_decode_workspace_dtype": (
                None
                if self.q4_value_decode_workspace is None
                else str(self.q4_value_decode_workspace.dtype)
            ),
            "q4_value_decode_tile_width": (
                KVQUANT_Q4_VALUE_DECODE_TILE_WIDTH
                if self.q4_value_decode_workspace is not None
                else None
            ),
            "q4_value_decode_max_tiles": (
                self.q4_value_decode_tile_capacity
                if self.q4_value_decode_workspace is not None
                else None
            ),
            "q23_value_decode_workspace_shape": (
                None
                if self.q23_value_decode_workspace is None
                else tuple(self.q23_value_decode_workspace.shape)
            ),
            "q23_value_decode_workspace_dtype": (
                None
                if self.q23_value_decode_workspace is None
                else str(self.q23_value_decode_workspace.dtype)
            ),
            "q23_value_decode_workspace_alias": (
                None
                if self.q23_value_decode_workspace is None
                else "decode_logits"
            ),
            "q23_value_decode_additional_allocated_bytes": 0,
            "q23_value_decode_tile_width": (
                KVQUANT_Q23_VALUE_DECODE_TILE_WIDTH
                if self.q23_value_decode_workspace is not None
                else None
            ),
            "dense_dtype": str(self.packed_key_cache.dtype),
            "sparse_value_dtype": str(self.key_sparse_values.dtype),
            "sparse_index_dtype": str(self.key_sparse_indices.dtype),
            "interface_dtype": str(self.interface_dtype),
            "sink_dtype": str(self.sink_dtype),
            "device": str(self.device),
            "persistent_breakdown": self.byte_breakdown(),
            "implementation_sha256": (
                hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                if self.config_name == "kvq4"
                else KVQUANT_LEGACY_CACHE_IMPLEMENTATION_SHA256
            ),
        }

        if self.config_name == "kvq4":
            payload.update(
                {
                    "q4_value_decode_tile_capacity": (
                        self.q4_value_decode_tile_capacity
                    ),
                    "q4_value_decode_total_attended_capacity": self.capacity,
                    "q4_value_decode_quantized_capacity": (
                        self.q4_value_decode_quantized_capacity
                    ),
                    "q4_value_decode_workspace_bytes": (
                        self.q4_value_decode_workspace_bytes
                    ),
                    "q4_value_decode_workspace_formula_version": (
                        KVQUANT_Q4_VALUE_DECODE_WORKSPACE_FORMULA_VERSION
                    ),
                }
            )
        return hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def gqa_geometry(self) -> dict[str, int | bool]:
        return {
            "num_query_heads": self.num_query_heads,
            "num_kv_heads": self.num_kv_heads,
            "gqa_group_size": self.num_query_heads // self.num_kv_heads,
            "native_kv_head_storage": True,
            "query_head_sized_kv_cache": False,
        }

    def storage_geometry(self) -> dict[str, tuple[int, ...]]:
        """Expose exact native-HKV shapes for manifests and path audits."""

        geometry = {
            "dense_k": tuple(self.packed_key_cache.shape),
            "dense_v": tuple(self.packed_value_cache.shape),
            "key_metadata": tuple(self.key_lookup_table.shape),
            "value_metadata": tuple(self.value_lookup_cache.shape),
            "value_store_bounds": tuple(
                self.value_store_lower_bounds.shape
            ),
            "key_sparse_values": tuple(self.key_sparse_values.shape),
            "key_sparse_indices": tuple(self.key_sparse_indices.shape),
            "value_sparse_values": tuple(self.value_sparse_values.shape),
            "value_sparse_indices": tuple(self.value_sparse_indices.shape),
            "sink_k": tuple(self.sink_key.shape),
            "sink_v": tuple(self.sink_value.shape),
            "decode_logits_bf16": tuple(self.decode_logits_bf16.shape),
            "sink_logits_fp16": tuple(self.sink_logits_fp16.shape),
            "sink_output_fp16": tuple(self.sink_output_fp16.shape),
        }
        if self.q4_value_decode_workspace is not None:
            geometry["q4_value_decode_workspace"] = tuple(
                self.q4_value_decode_workspace.shape
            )
        if self.q23_value_decode_workspace is not None:
            geometry["q23_value_decode_workspace_alias"] = tuple(
                self.q23_value_decode_workspace.shape
            )
        return geometry

    def zero_payload_slot(self, *, layer_idx: int, payload_slot: int) -> None:
        """Zero one caller-owned append destination before source ``atomicOr``."""

        layer = self._check_layer(layer_idx)
        if (
            isinstance(payload_slot, bool)
            or not isinstance(payload_slot, int)
            or payload_slot < 0
            or payload_slot >= self.capacity
        ):
            raise CacheBoundsError("payload slot is outside the static cache")
        self.packed_key_cache[layer, :, :, :, payload_slot].zero_()
        self.packed_value_cache[layer, :, :, :, payload_slot].zero_()
        self.value_lookup_cache[layer, :, payload_slot].zero_()
        self.key_sparse_values[layer, :, payload_slot].zero_()
        self.key_sparse_indices[layer, :, payload_slot].zero_()
        self.value_sparse_values[layer, :, payload_slot].zero_()
        self.value_sparse_indices[layer, :, payload_slot].zero_()
        self.key_active_counts[layer, :, payload_slot].zero_()
        self.value_active_counts[layer, :, payload_slot].zero_()

    def begin_prefill(self) -> None:
        if self._mode not in {"idle", "ready"}:
            raise CacheStateError("prefill cannot begin in the current mode")
        self._mode = "prefill"
        self._declared_prefill_length = 0
        self._expected_decode_position_binding = None
        self._known_key_active_entries = None

    def finish_prefill(
        self,
        length: int,
        *,
        key_active_entries: int | None = None,
    ) -> None:
        if self._mode != "prefill":
            raise CacheStateError("finish_prefill requires prefill mode")
        checked = self._check_length(length)
        self._active_context = checked
        self._mode = "ready"
        if key_active_entries is None:
            self._known_key_active_entries = (
                0 if checked <= self.sink_tokens else None
            )
        else:
            self.record_key_active_entries(key_active_entries)

    def prepare_prefill(self, prefix_length: int) -> None:
        self._declared_prefill_length = self._check_length(prefix_length)
        self.begin_prefill()
        self._declared_prefill_length = prefix_length

    def complete_prefill(self) -> None:
        if self._declared_prefill_length <= 0:
            raise CacheStateError("prefill length was not declared")
        self.finish_prefill(self._declared_prefill_length)

    def begin_fixed(self, prefix_length: int) -> None:
        length = self._check_length(prefix_length)
        if length < self.sink_tokens:
            raise CacheBoundsError("fixed-L requires the complete sink prefix")
        if length >= self.capacity:
            raise CacheBoundsError("fixed-L requires a reserved scratch position")
        if self._active_context != length:
            raise CacheStateError(
                "fixed-L prefix length does not match active state"
            )
        if (
            self._fixed_position_binding is None
            or self._fixed_position_binding[1] != length
        ):
            raise CacheStateError(
                "fixed-L cache position is not bound to the prefix"
            )
        self._prefix_length = length
        self._expected_decode_position_binding = self._fixed_position_binding
        self._mode = "fixed"

    def prepare_fixed(self, prefix_length: int) -> None:
        self.begin_fixed(prefix_length)

    def fixed_slot(self, layer_idx: int) -> int:
        self._check_layer(layer_idx)
        if self._mode != "fixed":
            raise CacheStateError("fixed slot requires fixed mode")
        return self._payload_slot(self._prefix_length)

    def fixed_scratch_overwrite(self, *, layer_idx: int) -> int:
        slot = self.fixed_slot(layer_idx)
        self.zero_payload_slot(layer_idx=layer_idx, payload_slot=slot)
        return slot

    def begin_growing(self, prefix_length: int, output_steps: int) -> None:
        length = self._check_length(prefix_length)
        steps = _positive_int(output_steps, "output_steps")
        if length < self.sink_tokens:
            raise CacheBoundsError("growing mode requires the complete sink prefix")
        if length + steps > self.capacity:
            raise CacheBoundsError("growing trajectory exceeds allocated capacity")
        if self._active_context != length:
            raise CacheStateError(
                "growing prefix length does not match active state"
            )
        if (
            len(self._growing_position_bindings) != steps
            or any(
                logical_position != length + step
                for step, (_, logical_position) in enumerate(
                    self._growing_position_bindings
                )
            )
        ):
            raise CacheStateError(
                "growing cache positions are not bound to the trajectory"
            )
        self._prefix_length = length
        self._output_steps = steps
        self._growing_step = -1
        self._expected_decode_position_binding = None
        self._mode = "growing_ready"

    def prepare_growing(self, prefix_length: int, output_steps: int) -> None:
        self.begin_growing(prefix_length, output_steps)

    def select_growing_step(self, step: int) -> None:
        if self._mode not in {"growing_ready", "growing_step"}:
            raise CacheStateError("growing step selection requires growing mode")
        if isinstance(step, bool) or not isinstance(step, int):
            raise CacheBoundsError("growing step must be an integer")
        if step < 0 or step >= self._output_steps:
            raise CacheBoundsError("growing step is outside the declared trajectory")
        if self._active_context != self._prefix_length + step:
            raise CacheStateError("growing active context does not match selected step")
        self._growing_step = step
        self._expected_decode_position_binding = (
            self._growing_position_bindings[step]
        )
        self._mode = "growing_step"

    def growing_slot(self, layer_idx: int) -> int:
        self._check_layer(layer_idx)
        if self._mode != "growing_step" or self._growing_step < 0:
            raise CacheStateError("growing slot requires an active step")
        position = self._prefix_length + self._growing_step
        return self._payload_slot(position)

    def growing_scratch_overwrite(self, *, layer_idx: int) -> int:
        slot = self.growing_slot(layer_idx)
        self.zero_payload_slot(layer_idx=layer_idx, payload_slot=slot)
        return slot

    def commit_growing(
        self,
        *,
        key_active_entries_added: int | None = None,
    ) -> None:
        if self._mode != "growing_step" or self._growing_step < 0:
            raise CacheStateError("no growing step is active")
        self._active_context = self._prefix_length + self._growing_step + 1
        if (
            self._known_key_active_entries is not None
            and key_active_entries_added is not None
        ):
            added = _nonnegative_int(
                key_active_entries_added, "key_active_entries_added"
            )
            if added > self.num_layers * self.key_cap:
                raise CacheBoundsError(
                    "growing Key active entries exceed one-token capacity"
                )
            self._known_key_active_entries += added
        elif key_active_entries_added is None:
            self._known_key_active_entries = None
        self._expected_decode_position_binding = None
        self._mode = "growing_ready"

    def finish_growing_step(self) -> None:
        self.commit_growing()

    def reset_growing(self) -> None:
        if self._mode != "growing_ready":
            raise CacheStateError("growing reset requires no active step")
        if self._active_context != self._prefix_length + self._output_steps:
            raise CacheStateError("growing trajectory is incomplete")
        self._mode = "ready"
        self._prefix_length = 0
        self._output_steps = 0
        self._growing_step = -1
        self._expected_decode_position_binding = None

    def reset_active_length(
        self,
        length: int = 0,
        *,
        key_active_entries: int | None = None,
    ) -> None:
        checked = self._check_length(length, allow_zero=True)
        self._active_context = checked
        self._mode = "idle" if checked == 0 else "ready"
        self._declared_prefill_length = 0
        self._prefix_length = 0
        self._output_steps = 0
        self._growing_step = -1
        self._expected_decode_position_binding = None
        if key_active_entries is None:
            self._known_key_active_entries = (
                0 if checked <= self.sink_tokens else None
            )
        else:
            self.record_key_active_entries(key_active_entries)
