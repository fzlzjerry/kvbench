"""Static cache storage for the Phase 8 patched-official KIVI adapter.

This module intentionally owns storage and rollover indexing only.  It does
not implement quantization or attention: those remain a thin adapter wrapper
around the checksum-bound official CUDA extension.  Keeping the static layout
here makes the adapter's measured path a sequence of in-place copies and
kernel calls rather than cache growth or concatenation.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import struct
from typing import Any, Final

from kvbench.runtime.static_cache import CacheBoundsError, CacheStateError
from kvbench.schema.phase8 import Phase8AllocationRatios


KIVI_GROUP_SIZE: Final[int] = 32
KIVI_RESIDUAL_LENGTH: Final[int] = 32
KIVI_CONFIG_BITS: Final[dict[str, tuple[int, int]]] = {
    "k4v4": (4, 4),
    "k2v4": (2, 4),
    "k2v2": (2, 2),
    "k4v2": (4, 2),
}
KIVI_MANDATORY_CONFIGS: Final[tuple[str, ...]] = ("k4v4", "k2v4", "k2v2")
KIVI_HELD_OUT_CONFIG: Final[str] = "k4v2"
_TORCH: Any | None = None


def _torch() -> Any:
    global _TORCH
    if _TORCH is None:
        try:
            _TORCH = importlib.import_module("torch")
        except ModuleNotFoundError as error:  # pragma: no cover - environment
            raise CacheStateError("PyTorch is required for KIVIStaticCache") from error
    return _TORCH


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _storage_bytes(tensor: Any) -> int:
    return int(tensor.untyped_storage().nbytes())


def _raw_tensor_bytes_untimed(tensor: Any) -> memoryview:
    """Expose exact contiguous CPU tensor bytes without Python-byte iteration."""

    byte_count = int(tensor.numel()) * int(tensor.element_size())
    raw_array = (ctypes.c_ubyte * byte_count).from_address(int(tensor.data_ptr()))
    raw_view = memoryview(raw_array).cast("B")
    if len(raw_view) != byte_count:
        raise CacheStateError("KIVI physical checksum storage is incomplete")
    return raw_view


@dataclass(frozen=True, slots=True)
class KIVIStaticCacheAccounting:
    """Complete physical accounting for the preallocated cache."""

    predicted_tensor_bytes: int
    measured_tensor_bytes: int
    allocated_bytes: int
    padding_bytes: int
    workspace_bytes: int
    capacity: int
    active_context: int
    temporary_peak_bytes: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "predicted_tensor_bytes": self.predicted_tensor_bytes,
            "measured_tensor_bytes": self.measured_tensor_bytes,
            "allocated_bytes": self.allocated_bytes,
            "padding_bytes": self.padding_bytes,
            "workspace_bytes": self.workspace_bytes,
            "capacity": self.capacity,
            "active_context": self.active_context,
            "temporary_peak_bytes": self.temporary_peak_bytes,
        }


@dataclass(frozen=True, slots=True)
class KIVIRollover:
    """Token movement declared by one static append bookkeeping operation."""

    key_group_ready: bool
    key_history_start: int | None
    value_token_evicted: int | None


class KIVIStaticCache:
    """One KIVI-specific pointer-stable cache for the frozen GQA geometry.

    Persistent K/V storage stays native to the eight KV heads.  The CPU token
    index ledgers are deliberately separate from CUDA storage: they are test
    and audit bookkeeping, not an input to the measured CUDA decode path.
    """

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
        if config_name not in KIVI_CONFIG_BITS:
            raise ValueError("unsupported KIVI configuration")
        geometry = (
            _positive_int(num_layers, "num_layers"),
            _positive_int(batch_size, "batch_size"),
            _positive_int(num_query_heads, "num_query_heads"),
            _positive_int(num_kv_heads, "num_kv_heads"),
            _positive_int(capacity, "capacity"),
            _positive_int(head_dim, "head_dim"),
        )
        if (
            geometry[1] not in {1, 2, 4, 8, 16}
            or geometry[2:4] != (32, 8)
            or geometry[5] != 128
        ):
            raise ValueError(
                "KIVI cache requires admitted B in {1,2,4,8,16} "
                "H_Q=32 H_KV=8 D=128 geometry"
            )
        if isinstance(workspace_bytes, bool) or not isinstance(workspace_bytes, int):
            raise ValueError("workspace_bytes must be a nonnegative integer")
        if workspace_bytes < 0:
            raise ValueError("workspace_bytes must be a nonnegative integer")

        self.config_name = config_name
        self.k_bits, self.v_bits = KIVI_CONFIG_BITS[config_name]
        self.num_layers = geometry[0]
        self.batch_size = geometry[1]
        self.num_query_heads = geometry[2]
        self.num_kv_heads = geometry[3]
        self.capacity = geometry[4]
        self.head_dim = geometry[5]
        self.group_size = KIVI_GROUP_SIZE
        self.residual_length = KIVI_RESIDUAL_LENGTH
        self.device = torch.device(device)
        self.dtype = torch.float16
        self.external_workspace_bytes = workspace_bytes

        # K only enters history in complete 32-token groups.  The residual
        # buffer is separately preallocated even when capacity is a multiple
        # of 32; this is the static alternative to source-path concatenation.
        self.key_history_capacity = (self.capacity // self.group_size) * self.group_size
        self.value_history_capacity = max(0, self.capacity - self.residual_length)
        self.key_history_groups = self.key_history_capacity // self.group_size
        self.value_head_groups = self.head_dim // self.group_size
        self.key_packed_words = self.key_history_capacity * self.k_bits // 32
        self.value_packed_words = self.head_dim * self.v_bits // 32

        self.packed_key_history = torch.empty(
            (
                self.num_layers,
                self.batch_size,
                self.num_kv_heads,
                self.key_packed_words,
                self.head_dim,
            ),
            dtype=torch.int32,
            device=self.device,
        )
        self.packed_value_history = torch.empty(
            (
                self.num_layers,
                self.batch_size,
                self.num_kv_heads,
                self.value_packed_words,
                self.value_history_capacity,
            ),
            dtype=torch.int32,
            device=self.device,
        )
        self.key_scales = torch.empty(
            (
                self.num_layers,
                self.batch_size,
                self.num_kv_heads,
                self.key_history_groups,
                self.head_dim,
            ),
            dtype=torch.float16,
            device=self.device,
        )
        self.key_minimums = torch.empty_like(self.key_scales)
        self.value_scales = torch.empty(
            (
                self.num_layers,
                self.batch_size,
                self.num_kv_heads,
                self.value_head_groups,
                self.value_history_capacity,
            ),
            dtype=torch.float16,
            device=self.device,
        )
        self.value_minimums = torch.empty_like(self.value_scales)

        self.key_residual = torch.empty(
            (
                self.num_layers,
                self.batch_size,
                self.num_kv_heads,
                self.residual_length,
                self.head_dim,
            ),
            dtype=torch.float16,
            device=self.device,
        )
        self.value_residual_ring = torch.empty_like(self.key_residual)
        # A fixed, contiguous presentation buffer replaces source-path
        # slice/contiguous allocation when the V residual ring wraps.
        self.value_residual_ordered_staging = torch.empty_like(self.key_residual)

        # Half-only KIVI ABI staging and fixed decode workspaces.
        self.query_fp16_staging = torch.empty(
            (self.batch_size, self.num_query_heads, self.head_dim),
            dtype=torch.float16,
            device=self.device,
        )
        self.key_fp16_staging = torch.empty(
            (self.batch_size, self.num_kv_heads, 1, self.head_dim),
            dtype=torch.float16,
            device=self.device,
        )
        self.value_fp16_staging = torch.empty_like(self.key_fp16_staging)
        self.key_scale_fp16_staging = torch.empty(
            (self.batch_size, self.num_kv_heads, self.head_dim, 1),
            dtype=torch.float16,
            device=self.device,
        )
        self.key_minimum_fp16_staging = torch.empty_like(self.key_scale_fp16_staging)
        self.value_scale_fp16_staging = torch.empty(
            (self.batch_size, self.num_kv_heads, 1, self.value_head_groups),
            dtype=torch.float16,
            device=self.device,
        )
        self.value_minimum_fp16_staging = torch.empty_like(self.value_scale_fp16_staging)
        self.quantization_fp16_staging = torch.empty(
            (
                self.batch_size,
                self.num_kv_heads,
                self.head_dim,
                self.group_size,
            ),
            dtype=torch.float16,
            device=self.device,
        )
        self.quantization_int_staging = torch.empty(
            self.quantization_fp16_staging.shape,
            dtype=torch.int32,
            device=self.device,
        )
        self.quantization_packed_staging = torch.empty(
            (self.batch_size, self.num_kv_heads, self.head_dim, 8),
            dtype=torch.int32,
            device=self.device,
        )
        self.decode_logits = torch.empty(
            (self.batch_size, self.num_query_heads, self.capacity),
            dtype=torch.float16,
            device=self.device,
        )
        # The frozen pybind entry point allocates its return tensor.  Phase 8
        # launches the same bound CUDA kernels directly into this contiguous
        # static buffer so the eager path has no allocator event.
        self.key_kernel_output_fp16 = torch.empty_like(self.decode_logits)
        self.decode_softmax = torch.empty(
            (self.batch_size, self.num_query_heads, self.capacity),
            dtype=torch.float32,
            device=self.device,
        )
        self.decode_merge = torch.empty(
            (self.batch_size, self.num_query_heads, self.head_dim),
            dtype=torch.float16,
            device=self.device,
        )
        self.decode_output_fp16 = torch.empty_like(self.decode_merge)
        self.output_buffer = torch.empty(
            (self.batch_size, self.num_query_heads, self.head_dim),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.reserved_workspace = torch.empty(
            (workspace_bytes,), dtype=torch.uint8, device=self.device
        )

        # These ledgers deliberately remain CPU-resident and are never used by
        # decode.  They permit exact rollover and no-loss/no-duplication tests
        # without tensor-to-host conversion in a measured CUDA method.
        self.key_history_token_indices = torch.full(
            (self.num_layers, self.key_history_capacity), -1, dtype=torch.int64
        )
        self.key_residual_token_indices = torch.full(
            (self.num_layers, self.residual_length), -1, dtype=torch.int64
        )
        self.value_history_token_indices = torch.full(
            (self.num_layers, self.value_history_capacity), -1, dtype=torch.int64
        )
        self.value_residual_token_indices = torch.full(
            (self.num_layers, self.residual_length), -1, dtype=torch.int64
        )
        self._key_history_counts = [0] * self.num_layers
        self._key_residual_counts = [0] * self.num_layers
        self._value_history_counts = [0] * self.num_layers
        self._value_residual_counts = [0] * self.num_layers
        self._value_residual_heads = [0] * self.num_layers
        self._fixed_scratch_tokens = [-1] * self.num_layers
        self._active_context = 0
        self._mode = "idle"
        self._prefix_length = 0
        self._output_steps = 0
        self._growing_step = -1

    @property
    def active_context(self) -> int:
        return self._active_context

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def maximum_context(self) -> int:
        return self.capacity

    @property
    def logical_bf16_storage_bytes(self) -> int:
        return self._logical_bf16_bytes(self.capacity)

    @property
    def r_hbm(self) -> None:
        """KIVI Phase 8 does not estimate or report HBM traffic."""

        return None

    def _logical_bf16_bytes(self, context: int, *, layers: int | None = None) -> int:
        checked = self._check_length(context, allow_zero=True)
        layer_count = self.num_layers if layers is None else layers
        return (
            layer_count
            * self.batch_size
            * self.num_kv_heads
            * checked
            * self.head_dim
            * 2
            * 2
        )

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

    def _owned_tensors(self) -> tuple[Any, ...]:
        return (
            self.packed_key_history,
            self.packed_value_history,
            self.key_scales,
            self.key_minimums,
            self.value_scales,
            self.value_minimums,
            self.key_residual,
            self.value_residual_ring,
            self.value_residual_ordered_staging,
            self.query_fp16_staging,
            self.key_fp16_staging,
            self.value_fp16_staging,
            self.key_scale_fp16_staging,
            self.key_minimum_fp16_staging,
            self.value_scale_fp16_staging,
            self.value_minimum_fp16_staging,
            self.quantization_fp16_staging,
            self.quantization_int_staging,
            self.quantization_packed_staging,
            self.decode_logits,
            self.key_kernel_output_fp16,
            self.decode_softmax,
            self.decode_merge,
            self.decode_output_fp16,
            self.output_buffer,
            self.reserved_workspace,
            self.key_history_token_indices,
            self.key_residual_token_indices,
            self.value_history_token_indices,
            self.value_residual_token_indices,
        )

    def byte_breakdown(self) -> dict[str, int]:
        """Exact physical bytes owned by the static adapter cache."""

        historical = {
            "quantized_k_payload": _storage_bytes(self.packed_key_history),
            "quantized_v_payload": _storage_bytes(self.packed_value_history),
            "key_scales": _storage_bytes(self.key_scales),
            "key_zero_points": _storage_bytes(self.key_minimums),
            "value_scales": _storage_bytes(self.value_scales),
            "value_zero_points": _storage_bytes(self.value_minimums),
            "residual_k": _storage_bytes(self.key_residual),
            "residual_v": _storage_bytes(self.value_residual_ring),
        }
        fp16_staging = sum(
            _storage_bytes(tensor)
            for tensor in (
                self.value_residual_ordered_staging,
                self.query_fp16_staging,
                self.key_fp16_staging,
                self.value_fp16_staging,
                self.key_scale_fp16_staging,
                self.key_minimum_fp16_staging,
                self.value_scale_fp16_staging,
                self.value_minimum_fp16_staging,
                self.decode_output_fp16,
            )
        )
        quantization_staging = sum(
            _storage_bytes(tensor)
            for tensor in (
                self.quantization_fp16_staging,
                self.quantization_int_staging,
                self.quantization_packed_staging,
            )
        )
        persistent_workspace = sum(
            _storage_bytes(tensor)
            for tensor in (
                self.decode_logits,
                self.key_kernel_output_fp16,
                self.decode_softmax,
                self.decode_merge,
                self.output_buffer,
                self.reserved_workspace,
            )
        )
        breakdown = {
            **historical,
            "other_metadata": sum(
                _storage_bytes(tensor)
                for tensor in (
                    self.key_history_token_indices,
                    self.key_residual_token_indices,
                    self.value_history_token_indices,
                    self.value_residual_token_indices,
                )
            ),
            "fp16_staging": fp16_staging,
            "quantization_staging": quantization_staging,
            "persistent_workspace": persistent_workspace,
            "value_rollover_shift_scratch": 0,
            "padding_alignment": 0,
            "block_group_rounding_bytes": 0,
        }
        owned = sum(_storage_bytes(tensor) for tensor in self._owned_tensors())
        if sum(breakdown.values()) != owned:
            raise CacheStateError("KIVI persistent byte breakdown is not exact")
        return breakdown

    def predicted_byte_breakdown(self) -> dict[str, int]:
        """Independent geometry formula for the static physical allocation."""

        layers = self.num_layers
        batch = self.batch_size
        kv_heads = self.num_kv_heads
        query_heads = self.num_query_heads
        dimension = self.head_dim
        residual = self.residual_length
        capacity = self.capacity
        key_residual_bytes = (
            layers * batch * kv_heads * residual * dimension * 2
        )
        value_residual_bytes = key_residual_bytes
        ordered_value_bytes = value_residual_bytes
        fp16_staging = sum(
            (
                ordered_value_bytes,
                batch * query_heads * dimension * 2,
                2 * batch * kv_heads * dimension * 2,
                2 * batch * kv_heads * dimension * 2,
                2 * batch * kv_heads * self.value_head_groups * 2,
                batch * query_heads * dimension * 2,
            )
        )
        quantization_elements = batch * kv_heads * dimension * residual
        quantization_staging = (
            quantization_elements * 2
            + quantization_elements * 4
            + batch * kv_heads * dimension * 8 * 4
        )
        persistent_workspace = (
            2 * batch * query_heads * capacity * 2
            + batch * query_heads * capacity * 4
            + batch * query_heads * dimension * 2
            + batch * query_heads * dimension * 2
            + self.external_workspace_bytes
        )
        return {
            "quantized_k_payload": (
                layers
                * batch
                * kv_heads
                * self.key_packed_words
                * dimension
                * 4
            ),
            "quantized_v_payload": (
                layers
                * batch
                * kv_heads
                * self.value_packed_words
                * self.value_history_capacity
                * 4
            ),
            "key_scales": (
                layers
                * batch
                * kv_heads
                * self.key_history_groups
                * dimension
                * 2
            ),
            "key_zero_points": (
                layers
                * batch
                * kv_heads
                * self.key_history_groups
                * dimension
                * 2
            ),
            "value_scales": (
                layers
                * batch
                * kv_heads
                * self.value_head_groups
                * self.value_history_capacity
                * 2
            ),
            "value_zero_points": (
                layers
                * batch
                * kv_heads
                * self.value_head_groups
                * self.value_history_capacity
                * 2
            ),
            "residual_k": key_residual_bytes,
            "residual_v": value_residual_bytes,
            "other_metadata": (
                layers
                * (
                    self.key_history_capacity
                    + self.residual_length
                    + self.value_history_capacity
                    + self.residual_length
                )
                * 8
            ),
            "fp16_staging": fp16_staging,
            "quantization_staging": quantization_staging,
            "persistent_workspace": persistent_workspace,
            "value_rollover_shift_scratch": 0,
            "padding_alignment": 0,
            "block_group_rounding_bytes": 0,
        }

    def _reference_active_byte_breakdown(self, context: int) -> dict[str, int]:
        """One-layer source-faithful active KIVI storage, excluding staging."""

        length = self._check_length(context, allow_zero=True)
        key_history = (length // self.group_size) * self.group_size
        key_residual = length - key_history
        value_history = max(0, length - self.residual_length)
        value_residual = min(length, self.residual_length)
        categories = {
            "quantized_k_payload": (
                self.batch_size
                * self.num_kv_heads
                * self.head_dim
                * key_history
                * self.k_bits
                // 8
            ),
            "quantized_v_payload": (
                self.batch_size
                * self.num_kv_heads
                * value_history
                * self.head_dim
                * self.v_bits
                // 8
            ),
            "key_scales": (
                self.batch_size
                * self.num_kv_heads
                * self.head_dim
                * (key_history // self.group_size)
                * 2
            ),
            "key_zero_points": (
                self.batch_size
                * self.num_kv_heads
                * self.head_dim
                * (key_history // self.group_size)
                * 2
            ),
            "value_scales": (
                self.batch_size
                * self.num_kv_heads
                * value_history
                * self.value_head_groups
                * 2
            ),
            "value_zero_points": (
                self.batch_size
                * self.num_kv_heads
                * value_history
                * self.value_head_groups
                * 2
            ),
            "other_metadata": 0,
            "residual_k": (
                self.batch_size
                * self.num_kv_heads
                * key_residual
                * self.head_dim
                * 2
            ),
            "residual_v": (
                self.batch_size
                * self.num_kv_heads
                * value_residual
                * self.head_dim
                * 2
            ),
            "padding_alignment": 0,
            "persistent_workspace": 0,
        }
        return categories

    def reference_active_byte_breakdown(self, context: int | None = None) -> dict[str, int]:
        """Fixture-comparable, one-layer active storage accounting."""

        active = self.active_context if context is None else context
        return self._reference_active_byte_breakdown(active)

    def active_byte_breakdown(self, active_context: int | None = None) -> dict[str, int]:
        """Full-model active source storage; staging is intentionally separate."""

        active = self.active_context if active_context is None else active_context
        one_layer = self._reference_active_byte_breakdown(active)
        return {name: value * self.num_layers for name, value in one_layer.items()}

    def active_storage_bytes(self, active_context: int | None = None) -> int:
        return sum(self.active_byte_breakdown(active_context).values())

    def active_logical_bf16_bytes(self, active_context: int | None = None) -> int:
        active = self.active_context if active_context is None else active_context
        return self._logical_bf16_bytes(active)

    def accounting(self) -> KIVIStaticCacheAccounting:
        breakdown = self.byte_breakdown()
        allocated = sum(breakdown.values())
        predicted = sum(self.predicted_byte_breakdown().values())
        if abs(predicted - allocated) / allocated >= 0.01:
            raise CacheStateError("KIVI predicted allocation differs by at least 1%")
        workspace = (
            breakdown["fp16_staging"]
            + breakdown["quantization_staging"]
            + breakdown["persistent_workspace"]
        )
        workspace += breakdown["value_rollover_shift_scratch"]
        return KIVIStaticCacheAccounting(
            predicted_tensor_bytes=predicted,
            measured_tensor_bytes=allocated,
            allocated_bytes=allocated,
            padding_bytes=breakdown["padding_alignment"],
            workspace_bytes=workspace,
            capacity=self.capacity,
            active_context=self.active_context,
        )

    def ratios(self) -> Phase8AllocationRatios:
        return Phase8AllocationRatios.from_bytes(
            allocated_bytes=self.accounting().allocated_bytes,
            bf16_allocated_bytes=self.logical_bf16_storage_bytes,
        )

    def pointers(self) -> dict[str, int]:
        names = (
            "packed_key_history",
            "packed_value_history",
            "key_scales",
            "key_minimums",
            "value_scales",
            "value_minimums",
            "key_residual",
            "value_residual_ring",
            "value_residual_ordered_staging",
            "query_fp16_staging",
            "key_fp16_staging",
            "value_fp16_staging",
            "key_scale_fp16_staging",
            "key_minimum_fp16_staging",
            "value_scale_fp16_staging",
            "value_minimum_fp16_staging",
            "quantization_fp16_staging",
            "quantization_int_staging",
            "quantization_packed_staging",
            "decode_logits",
            "key_kernel_output_fp16",
            "decode_softmax",
            "decode_merge",
            "decode_output_fp16",
            "output_buffer",
            "reserved_workspace",
            "key_history_token_indices",
            "key_residual_token_indices",
            "value_history_token_indices",
            "value_residual_token_indices",
        )
        return {
            f"{name}_data_ptr": int(tensor.data_ptr())
            for name, tensor in zip(names, self._owned_tensors(), strict=True)
        }

    def layout_fingerprint(self) -> str:
        payload = {
            "schema": "kvbench-kivi-static-cache-layout-1.1.0",
            "configuration": self.config_name,
            "bits": {"key": self.k_bits, "value": self.v_bits},
            "num_layers": self.num_layers,
            "batch_size": self.batch_size,
            "num_query_heads": self.num_query_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "capacity": self.capacity,
            "key_history_capacity": self.key_history_capacity,
            "value_history_capacity": self.value_history_capacity,
            "key_packed_words": self.key_packed_words,
            "value_packed_words": self.value_packed_words,
            "group_size": self.group_size,
            "residual_length": self.residual_length,
            "native_kv_head_storage": True,
            "gqa_mapping": "query_head // 4",
            "packed_key_shape": list(self.packed_key_history.shape),
            "packed_value_shape": list(self.packed_value_history.shape),
            "residual_shape": list(self.key_residual.shape),
            "persistent_breakdown": self.byte_breakdown(),
            "device": str(self.device),
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def gqa_geometry(self) -> dict[str, Any]:
        return {
            "num_query_heads": self.num_query_heads,
            "num_kv_heads": self.num_kv_heads,
            "gqa_group_size": self.num_query_heads // self.num_kv_heads,
            "kv_head_mapping": "query_head // 4",
            "native_kv_head_storage": True,
            "gqa_materialized": False,
            "expanded_kv_heads": 0,
        }

    def initialize_deterministic(self) -> None:
        for tensor in self._owned_tensors():
            tensor.zero_()
        self._clear_token_ledgers()
        self.reset_active_length(0)

    def release_owned_cuda_resources_for_sanitizer(self) -> None:
        """Irreversibly release probe-owned storage before context teardown."""

        if self._mode == "released":
            return
        from kvbench.runtime.turboquant_cache import (
            _release_tensor_storages_for_sanitizer,
        )

        _release_tensor_storages_for_sanitizer(self._owned_tensors())
        for handle in getattr(self, "_kivi_handles", {}).values():
            handle.prefill_key_states = None
            handle.prefill_value_states = None
            handle.pending_key = None
            handle.pending_value = None
            handle.commit_after_decode = False
        getattr(self, "_kivi_handles", {}).clear()
        self._mode = "released"

    def _clear_token_ledgers(self) -> None:
        for ledger in (
            self.key_history_token_indices,
            self.key_residual_token_indices,
            self.value_history_token_indices,
            self.value_residual_token_indices,
        ):
            ledger.fill_(-1)
        self._key_history_counts[:] = [0] * self.num_layers
        self._key_residual_counts[:] = [0] * self.num_layers
        self._value_history_counts[:] = [0] * self.num_layers
        self._value_residual_counts[:] = [0] * self.num_layers
        self._value_residual_heads[:] = [0] * self.num_layers
        self._fixed_scratch_tokens[:] = [-1] * self.num_layers

    def reset_active_length(self, length: int = 0) -> None:
        checked = self._check_length(length, allow_zero=True)
        if checked == 0:
            self._clear_token_ledgers()
        self._active_context = checked
        self._mode = "idle"
        self._prefix_length = 0
        self._output_steps = 0
        self._growing_step = -1
        for handle in getattr(self, "_kivi_handles", {}).values():
            handle.prefill = False
            handle.prefill_key_states = None
            handle.prefill_value_states = None
            handle.pending_key = None
            handle.pending_value = None
            handle.commit_after_decode = False

    def prepare_prefill(self, prefix_length: int) -> None:
        self._prefix_length = self._check_length(prefix_length)
        self._mode = "prefill"

    def complete_prefill(self) -> None:
        if self._mode != "prefill":
            raise CacheStateError("complete_prefill requires prefill mode")
        self._active_context = self._prefix_length
        self._mode = "ready"

    def prepare_fixed(self, prefix_length: int) -> None:
        length = self._check_length(prefix_length)
        if length >= self.capacity:
            raise CacheBoundsError("fixed-L requires a reserved scratch position")
        if self._active_context != length:
            raise CacheStateError("fixed-L prefix length does not match active state")
        self._prefix_length = length
        self._mode = "fixed"

    def prepare_growing(self, prefix_length: int, output_steps: int) -> None:
        length = self._check_length(prefix_length)
        steps = _positive_int(output_steps, "output_steps")
        if length + steps > self.capacity:
            raise CacheBoundsError("growing trajectory exceeds allocated capacity")
        if self._active_context != length:
            raise CacheStateError("growing prefix length does not match active state")
        self._prefix_length = length
        self._output_steps = steps
        self._growing_step = -1
        self._mode = "growing_ready"

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
        self._mode = "growing_step"

    def update(self, *, layer_idx: int, token_index: int) -> KIVIRollover:
        """Record one append without allocation, cat, or source-path slicing.

        The adapter performs the corresponding in-place FP16 copies and
        official quantizer calls.  This small ledger operation exposes exactly
        which K group or V token must move at a source-faithful boundary.
        """

        layer = self._check_layer(layer_idx)
        if isinstance(token_index, bool) or not isinstance(token_index, int):
            raise CacheBoundsError("token index must be an integer")
        if token_index < 0 or token_index >= self.capacity:
            raise CacheBoundsError("token index exceeds static capacity")
        expected_token = (
            self._key_history_counts[layer]
            + self._key_residual_counts[layer]
        )
        if token_index != expected_token:
            raise CacheStateError(
                "KIVI token sequence is missing, duplicated, or reordered"
            )
        key_count = self._key_residual_counts[layer]
        if key_count >= self.residual_length:
            raise CacheStateError("K residual ledger is inconsistent")
        self.key_residual_token_indices[layer, key_count] = token_index
        key_count += 1
        key_group_ready = key_count == self.residual_length
        key_history_start: int | None = None
        if key_group_ready:
            history_count = self._key_history_counts[layer]
            if history_count + self.residual_length > self.key_history_capacity:
                raise CacheBoundsError("K history exceeds static capacity")
            self.key_history_token_indices[layer, history_count : history_count + self.residual_length].copy_(
                self.key_residual_token_indices[layer]
            )
            self.key_residual_token_indices[layer].fill_(-1)
            self._key_history_counts[layer] = history_count + self.residual_length
            self._key_residual_counts[layer] = 0
            key_history_start = history_count
        else:
            self._key_residual_counts[layer] = key_count

        value_count = self._value_residual_counts[layer]
        value_head = self._value_residual_heads[layer]
        value_token_evicted: int | None = None
        if value_count < self.residual_length:
            slot = (value_head + value_count) % self.residual_length
            self.value_residual_token_indices[layer, slot] = token_index
            self._value_residual_counts[layer] = value_count + 1
        else:
            history_count = self._value_history_counts[layer]
            if history_count >= self.value_history_capacity:
                raise CacheBoundsError("V history exceeds static capacity")
            evicted = self.value_residual_token_indices[layer, value_head]
            value_token_evicted = token_index - self.residual_length
            self.value_history_token_indices[layer, history_count] = evicted
            self.value_residual_token_indices[layer, value_head] = token_index
            self._value_history_counts[layer] = history_count + 1
            self._value_residual_heads[layer] = (value_head + 1) % self.residual_length

        # The same cache state is filled layer-by-layer; active context tracks
        # the longest admitted per-layer history, never a H_Q-expanded tensor.
        self._active_context = max(self._active_context, token_index + 1)
        if self._active_context > self.capacity:
            raise CacheBoundsError("append exceeds static capacity")
        return KIVIRollover(
            key_group_ready=key_group_ready,
            key_history_start=key_history_start,
            value_token_evicted=value_token_evicted,
        )

    def ordered_value_residual_token_indices(self, layer_idx: int) -> Any:
        """Return a fixed-size CPU view in logical order for audit code."""

        layer = self._check_layer(layer_idx)
        count = self._value_residual_counts[layer]
        head = self._value_residual_heads[layer]
        result = _torch().full((self.residual_length,), -1, dtype=_torch().int64)
        first = min(count, self.residual_length - head)
        if first:
            result[:first].copy_(
                self.value_residual_token_indices[layer, head : head + first]
            )
        if count > first:
            result[first:count].copy_(
                self.value_residual_token_indices[layer, : count - first]
            )
        return result[:count]

    def token_index_state(self, layer_idx: int) -> dict[str, Any]:
        """Expose fixed ledgers for fixture/audit code outside hot execution."""

        layer = self._check_layer(layer_idx)
        return {
            "quantized_key_tokens": self.key_history_token_indices[
                layer, : self._key_history_counts[layer]
            ],
            "residual_key_tokens": self.key_residual_token_indices[
                layer, : self._key_residual_counts[layer]
            ],
            "quantized_value_tokens": self.value_history_token_indices[
                layer, : self._value_history_counts[layer]
            ],
            "residual_value_tokens": self.ordered_value_residual_token_indices(layer),
        }

    def fixed_scratch_overwrite(self, *, layer_idx: int, token_index: int) -> None:
        """Record fixed-L scratch use without mutating any historical ledger."""

        layer = self._check_layer(layer_idx)
        if self._mode != "fixed":
            raise CacheStateError("fixed scratch overwrite requires fixed mode")
        if isinstance(token_index, bool) or not isinstance(token_index, int):
            raise CacheBoundsError("token index must be an integer")
        if token_index != self._prefix_length or token_index >= self.capacity:
            raise CacheBoundsError("fixed scratch token is outside its reserved slot")
        self._fixed_scratch_tokens[layer] = token_index

    def history_checksum(self, layer_idx: int) -> str:
        """Audit-only logical and physical checksum excluding fixed scratch."""

        state = self.token_index_state(layer_idx)
        digest = hashlib.sha256()
        for name in (
            "quantized_key_tokens",
            "residual_key_tokens",
            "quantized_value_tokens",
            "residual_value_tokens",
        ):
            ledger = state[name]
            digest.update(name.encode("ascii"))
            digest.update(struct.pack("<Q", int(ledger.numel())))
            for index in range(int(ledger.numel())):
                # Audit-only CPU-ledger serialization. `update` never reads
                # a CUDA tensor back to the host and this helper is outside
                # every measured decode boundary.
                digest.update(struct.pack("<q", int(ledger[index])))
        digest.update(self.physical_history_checksum(layer_idx).encode("ascii"))
        return digest.hexdigest()

    def fixed_scratch_history_checksum(self, layer_idx: int) -> str:
        return self.history_checksum(layer_idx)

    def physical_history_checksum(self, layer_idx: int) -> str:
        """Hash exact persistent K/V history and residual bytes, untimed."""

        layer = self._check_layer(layer_idx)
        digest = hashlib.sha256()
        tensors = (
            ("packed_key_history", self.packed_key_history[layer]),
            ("packed_value_history", self.packed_value_history[layer]),
            ("key_scales", self.key_scales[layer]),
            ("key_minimums", self.key_minimums[layer]),
            ("value_scales", self.value_scales[layer]),
            ("value_minimums", self.value_minimums[layer]),
            ("key_residual", self.key_residual[layer]),
            ("value_residual_ring", self.value_residual_ring[layer]),
        )
        for name, tensor in tensors:
            copied = tensor.detach().to(device="cpu", copy=True).contiguous()
            byte_count = int(copied.numel()) * int(copied.element_size())
            raw = _raw_tensor_bytes_untimed(copied)
            digest.update(name.encode("ascii"))
            digest.update(str(tuple(int(item) for item in tensor.shape)).encode("ascii"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(raw)
        return digest.hexdigest()
