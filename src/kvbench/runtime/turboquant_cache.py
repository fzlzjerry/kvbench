"""Static full-model cache state for the Phase 6 TurboQuant adapter."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
from typing import Any, Mapping
import weakref

from kvbench.runtime.static_cache import (
    BF16StaticCache,
    CacheBoundsError,
    CacheStateError,
)


TURBOQUANT_BLOCK_SIZE = 16
TURBOQUANT_MAX_KV_SPLITS = 4
TURBOQUANT_COMPRESSED_LAYERS = tuple(range(2, 30))
TURBOQUANT_BF16_LAYERS = (0, 1, 30, 31)
TURBOQUANT_MANDATORY_CONFIGS = (
    "turboquant_4bit_nc",
    "turboquant_k3v4_nc",
    "turboquant_3bit_nc",
)
TURBOQUANT_SLOT_SIZES: Mapping[str, int] = {
    "turboquant_4bit_nc": 134,
    "turboquant_k3v4_nc": 118,
    "turboquant_3bit_nc": 102,
}
_SLOT_COMPONENTS: Mapping[str, Mapping[str, int]] = {
    "turboquant_4bit_nc": {
        "compressed_key_payload_bytes": 64,
        "key_norm_metadata_bytes": 2,
        "compressed_value_payload_bytes": 64,
        "value_scale_metadata_bytes": 2,
        "value_zero_point_metadata_bytes": 2,
    },
    "turboquant_k3v4_nc": {
        "compressed_key_payload_bytes": 48,
        "key_norm_metadata_bytes": 2,
        "compressed_value_payload_bytes": 64,
        "value_scale_metadata_bytes": 2,
        "value_zero_point_metadata_bytes": 2,
    },
    "turboquant_3bit_nc": {
        "compressed_key_payload_bytes": 48,
        "key_norm_metadata_bytes": 2,
        "compressed_value_payload_bytes": 48,
        "value_scale_metadata_bytes": 2,
        "value_zero_point_metadata_bytes": 2,
    },
}
_TORCH: Any | None = None


def _torch() -> Any:
    global _TORCH
    if _TORCH is None:
        try:
            _TORCH = importlib.import_module("torch")
        except ModuleNotFoundError as error:
            raise CacheStateError(
                "PyTorch is required for TurboQuantStaticCache"
            ) from error
    return _TORCH


def _release_tensor_storages_for_sanitizer(
    tensors: tuple[Any, ...],
) -> None:
    """Irreversibly release unique tensor storages in an isolated probe."""

    storages: dict[int, Any] = {}
    for tensor in tensors:
        if tensor is None:
            continue
        storage = tensor.untyped_storage()
        if int(storage.nbytes()) == 0:
            continue
        storages.setdefault(int(storage._cdata), storage)
    for storage in storages.values():
        storage.resize_(0)


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _tensor_storage_bytes(tensor: Any) -> int:
    return int(tensor.untyped_storage().nbytes())


def _tensor_bytes_untimed(tensor: Any) -> bytes:
    contiguous = tensor.detach().contiguous().cpu()
    return contiguous.view(_torch().uint8).numpy().tobytes()


@dataclass(frozen=True, slots=True)
class TurboQuantCacheAccounting:
    """Exact persistent-storage accounting for one TurboQuant cache."""

    predicted_tensor_bytes: int
    measured_tensor_bytes: int
    allocated_bytes: int
    padding_bytes: int
    workspace_bytes: int
    capacity: int
    rounded_capacity: int
    active_context: int
    temporary_peak_bytes: int

    def to_dict(self) -> dict[str, int]:
        return {
            "predicted_tensor_bytes": self.predicted_tensor_bytes,
            "measured_tensor_bytes": self.measured_tensor_bytes,
            "allocated_bytes": self.allocated_bytes,
            "padding_bytes": self.padding_bytes,
            "workspace_bytes": self.workspace_bytes,
            "capacity": self.capacity,
            "rounded_capacity": self.rounded_capacity,
            "active_context": self.active_context,
            "temporary_peak_bytes": self.temporary_peak_bytes,
        }


@dataclass(slots=True)
class TurboQuantAttentionHandle:
    """Precreated method-owned attended-state handle used by the endpoint."""

    cache: "TurboQuantStaticCache"
    layer_idx: int
    key_states: Any | None = None
    value_states: Any | None = None
    prefill: bool = False


class TurboQuantStaticCache:
    """One pointer-stable packed cache plus four unchanged BF16 layer slots."""

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
        if config_name not in TURBOQUANT_MANDATORY_CONFIGS:
            raise ValueError("unsupported TurboQuant configuration")
        geometry = (
            _positive_int(num_layers, "num_layers"),
            _positive_int(batch_size, "batch_size"),
            _positive_int(num_query_heads, "num_query_heads"),
            _positive_int(num_kv_heads, "num_kv_heads"),
            _positive_int(capacity, "capacity"),
            _positive_int(head_dim, "head_dim"),
        )
        if (
            geometry[0] != 32
            or geometry[1] not in {1, 2, 4, 8, 16}
            or geometry[2:4] != (32, 8)
            or geometry[5] != 128
        ):
            raise ValueError(
                "TurboQuant cache requires admitted B in {1,2,4,8,16} "
                "Llama GQA geometry"
            )
        if (
            isinstance(workspace_bytes, bool)
            or not isinstance(workspace_bytes, int)
            or workspace_bytes < 0
        ):
            raise ValueError("workspace_bytes must be a nonnegative integer")

        self.config_name = config_name
        self.num_layers = geometry[0]
        self.batch_size = geometry[1]
        self.num_query_heads = geometry[2]
        self.num_kv_heads = geometry[3]
        self.capacity = geometry[4]
        self.allocated_capacity = geometry[4]
        self.head_dim = geometry[5]
        self.block_size = TURBOQUANT_BLOCK_SIZE
        self.block_count = math.ceil(self.capacity / self.block_size)
        self.rounded_capacity = self.block_count * self.block_size
        self.compressed_layers = TURBOQUANT_COMPRESSED_LAYERS
        self.bf16_layers = TURBOQUANT_BF16_LAYERS
        self.slot_size = TURBOQUANT_SLOT_SIZES[config_name]
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        self.external_workspace_bytes = workspace_bytes

        try:
            from kvbench.third_party.vllm_turboquant import (
                TurboQuantConfig,
                _fwd_kernel_stage2,
                _tq_decode_stage1,
                _tq_fused_store_mse,
                build_hadamard,
                solve_lloyd_max,
            )
        except (ImportError, ModuleNotFoundError) as error:
            raise CacheStateError(
                "pinned TurboQuant compatibility package is unavailable"
            ) from error

        self._store_kernel = _tq_fused_store_mse
        self._decode_stage1_kernel = _tq_decode_stage1
        self._decode_stage2_kernel = _fwd_kernel_stage2
        self.tq_config = TurboQuantConfig.from_cache_dtype(
            config_name,
            self.head_dim,
        )
        if (
            bool(self.tq_config.key_fp8)
            or int(self.tq_config.slot_size_aligned) != self.slot_size
        ):
            raise CacheStateError(
                "pinned TurboQuant config differs from the frozen layout"
            )

        self.packed_cache = torch.zeros(
            (
                len(self.compressed_layers),
                self.batch_size * self.block_count,
                self.block_size,
                self.num_kv_heads,
                self.slot_size,
            ),
            dtype=torch.uint8,
            device=self.device,
        )
        self.bf16_cache = BF16StaticCache(
            num_layers=len(self.bf16_layers),
            batch_size=self.batch_size,
            num_kv_heads=self.num_kv_heads,
            capacity=self.rounded_capacity,
            head_dim=self.head_dim,
            device=self.device,
        )

        self.block_table = torch.arange(
            self.batch_size * self.block_count,
            dtype=torch.int32,
            device=self.device,
        ).reshape(self.batch_size, self.block_count)
        positions = torch.arange(
            self.rounded_capacity, dtype=torch.int32, device=self.device
        ).reshape(self.rounded_capacity, 1)
        batch_offsets = (
            torch.arange(
                self.batch_size, dtype=torch.int32, device=self.device
            )
            * self.rounded_capacity
        ).reshape(1, self.batch_size)
        # Position-major ordering matches flattened [T,B,H_KV,D] staging.
        # B=1 retains the historical mapping byte-for-byte.
        self.slot_mapping = (positions + batch_offsets).reshape(-1)
        self._seq_lens = torch.arange(
            1,
            self.rounded_capacity + 1,
            dtype=torch.int32,
            device=self.device,
        ).reshape(self.rounded_capacity, 1).expand(
            self.rounded_capacity, self.batch_size
        ).clone()
        self._single_slot_mappings = tuple(
            self.slot_mapping[
                position * self.batch_size : (position + 1) * self.batch_size
            ]
            for position in range(self.rounded_capacity)
        )
        self._single_seq_lens = tuple(
            self._seq_lens[position]
            for position in range(self.rounded_capacity)
        )

        hadamard = build_hadamard(self.head_dim, str(self.device))
        self._sanitizer_hadamard_source = weakref.ref(hadamard)
        self.Pi = hadamard.detach().clone()
        self.PiT = self.Pi.T.contiguous()
        centroids, midpoints = solve_lloyd_max(
            self.head_dim,
            int(self.tq_config.centroid_bits),
        )
        self.centroids = centroids.to(self.device).clone()
        self.midpoints = midpoints.to(self.device).clone()

        store_shape = (
            self.capacity,
            self.batch_size,
            self.num_kv_heads,
            self.head_dim,
        )
        self.store_key_float = torch.empty(
            store_shape,
            dtype=torch.float32,
            device=self.device,
        )
        self.store_value_float = torch.empty_like(self.store_key_float)
        self.store_rotated_key = torch.empty_like(self.store_key_float)
        self.store_norms = torch.empty(
            (self.capacity, self.batch_size, self.num_kv_heads, 1),
            dtype=torch.float32,
            device=self.device,
        )
        self.store_norm_denominator = torch.empty_like(self.store_norms)
        self.decode_query_float = torch.empty(
            (self.batch_size, self.num_query_heads, self.head_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.decode_rotated_query = torch.empty_like(
            self.decode_query_float
        )
        self.decode_mid_o = torch.empty(
            (
                self.batch_size,
                self.num_query_heads,
                TURBOQUANT_MAX_KV_SPLITS,
                self.head_dim + 1,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        self.decode_output = torch.empty(
            (self.batch_size, self.num_query_heads, self.head_dim),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.decode_lse = torch.empty(
            (self.batch_size, self.num_query_heads),
            dtype=torch.float32,
            device=self.device,
        )
        self.reserved_workspace = torch.empty(
            (workspace_bytes,),
            dtype=torch.uint8,
            device=self.device,
        )

        self._compressed_layer_slots = {
            layer: slot for slot, layer in enumerate(self.compressed_layers)
        }
        self._bf16_layer_slots = {
            layer: slot for slot, layer in enumerate(self.bf16_layers)
        }
        cache_proxy = weakref.proxy(self)
        self._handles = {
            layer: TurboQuantAttentionHandle(cache_proxy, layer)
            for layer in self.compressed_layers
        }
        self._active_context = 0
        self._mode = "idle"
        self._prefix_length = 0
        self._output_steps = 0
        self._growing_step = -1
        self._current_slot_mapping = self.slot_mapping[:0]
        self._current_seq_lens = self._single_seq_lens[0]

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
    def current_slot_mapping(self) -> Any:
        return self._current_slot_mapping

    @property
    def current_seq_lens(self) -> Any:
        return self._current_seq_lens

    @property
    def logical_bf16_storage_bytes(self) -> int:
        return (
            2
            * self.num_layers
            * self.batch_size
            * self.num_kv_heads
            * self.capacity
            * self.head_dim
            * 2
        )

    @property
    def r_nominal(self) -> float:
        compressed_bits = int(self.tq_config.key_quant_bits) + int(
            self.tq_config.value_quant_bits
        )
        nominal_compressed = (
            len(self.compressed_layers)
            * self.batch_size
            * self.capacity
            * self.num_kv_heads
            * self.head_dim
            * compressed_bits
            // 8
        )
        skipped = (
            2
            * len(self.bf16_layers)
            * self.batch_size
            * self.capacity
            * self.num_kv_heads
            * self.head_dim
            * 2
        )
        return self.logical_bf16_storage_bytes / (
            nominal_compressed + skipped
        )

    @property
    def r_alloc(self) -> float:
        return self.logical_bf16_storage_bytes / self.accounting().allocated_bytes

    @property
    def r_hbm(self) -> None:
        return None

    def _owned_tensors(self) -> tuple[Any, ...]:
        return (
            self.packed_cache,
            self.bf16_cache.keys,
            self.bf16_cache.values,
            self.block_table,
            self.slot_mapping,
            self._seq_lens,
            self.Pi,
            self.PiT,
            self.centroids,
            self.midpoints,
            self.store_key_float,
            self.store_value_float,
            self.store_rotated_key,
            self.store_norms,
            self.store_norm_denominator,
            self.decode_query_float,
            self.decode_rotated_query,
            self.decode_mid_o,
            self.decode_output,
            self.decode_lse,
            self.reserved_workspace,
        )

    def release_owned_cuda_resources_for_sanitizer(self) -> None:
        """Irreversibly release probe-owned tensors before context teardown."""

        if self._mode == "released":
            return
        hadamard_source = self._sanitizer_hadamard_source()
        owned_tensors = self._owned_tensors()
        if hadamard_source is not None:
            owned_tensors = (*owned_tensors, hadamard_source)
        _release_tensor_storages_for_sanitizer(owned_tensors)
        from kvbench.third_party.vllm_turboquant.compat import (
            _build_hadamard_cached,
        )

        _build_hadamard_cached.cache_clear()
        self._sanitizer_hadamard_source = None
        for handle in self._handles.values():
            handle.key_states = None
            handle.value_states = None
            handle.prefill = False
        self._handles.clear()
        self._current_slot_mapping = None
        self._current_seq_lens = None
        self._single_slot_mappings = ()
        self._single_seq_lens = ()

        bf16_cache = self.bf16_cache
        if bf16_cache is not None:
            bf16_cache.reset_active_length(0)
            bf16_cache.keys = None
            bf16_cache.values = None
            self.bf16_cache = None

        for name in (
            "packed_cache",
            "block_table",
            "slot_mapping",
            "_seq_lens",
            "Pi",
            "PiT",
            "centroids",
            "midpoints",
            "store_key_float",
            "store_value_float",
            "store_rotated_key",
            "store_norms",
            "store_norm_denominator",
            "decode_query_float",
            "decode_rotated_query",
            "decode_mid_o",
            "decode_output",
            "decode_lse",
            "reserved_workspace",
        ):
            setattr(self, name, None)
        self._active_context = 0
        self._mode = "released"

    def byte_breakdown(self) -> dict[str, int]:
        components = _SLOT_COMPONENTS[self.config_name]
        requested_slots = (
            len(self.compressed_layers)
            * self.batch_size
            * self.capacity
            * self.num_kv_heads
        )
        breakdown = {
            name: requested_slots * bytes_per_slot
            for name, bytes_per_slot in components.items()
        }
        semantic_slot_size = sum(components.values())
        breakdown["slot_padding_alignment_bytes"] = (
            requested_slots * (self.slot_size - semantic_slot_size)
        )
        skipped_component = (
            len(self.bf16_layers)
            * self.batch_size
            * self.num_kv_heads
            * self.capacity
            * self.head_dim
            * 2
        )
        breakdown["skipped_layer_bf16_key_bytes"] = skipped_component
        breakdown["skipped_layer_bf16_value_bytes"] = skipped_component
        extra_positions = self.rounded_capacity - self.capacity
        breakdown["block_rounding_overhead_bytes"] = (
            len(self.compressed_layers)
            * self.batch_size
            * extra_positions
            * self.num_kv_heads
            * self.slot_size
            + 2
            * len(self.bf16_layers)
            * self.batch_size
            * self.num_kv_heads
            * extra_positions
            * self.head_dim
            * 2
        )
        breakdown["mapping_metadata_bytes"] = sum(
            _tensor_storage_bytes(tensor)
            for tensor in (
                self.block_table,
                self.slot_mapping,
                self._seq_lens,
            )
        )
        cache_and_mapping = (
            _tensor_storage_bytes(self.packed_cache)
            + _tensor_storage_bytes(self.bf16_cache.keys)
            + _tensor_storage_bytes(self.bf16_cache.values)
            + breakdown["mapping_metadata_bytes"]
        )
        total_owned = sum(
            _tensor_storage_bytes(tensor) for tensor in self._owned_tensors()
        )
        breakdown["persistent_workspace_bytes"] = total_owned - cache_and_mapping
        if sum(breakdown.values()) != total_owned:
            raise CacheStateError(
                "TurboQuant persistent byte breakdown is not exact"
            )
        return breakdown

    def accounting(self) -> TurboQuantCacheAccounting:
        breakdown = self.byte_breakdown()
        allocated = sum(breakdown.values())
        workspace = (
            breakdown["mapping_metadata_bytes"]
            + breakdown["persistent_workspace_bytes"]
        )
        return TurboQuantCacheAccounting(
            predicted_tensor_bytes=allocated,
            measured_tensor_bytes=allocated,
            allocated_bytes=allocated,
            padding_bytes=(
                breakdown["slot_padding_alignment_bytes"]
                + breakdown["block_rounding_overhead_bytes"]
            ),
            workspace_bytes=workspace,
            capacity=self.capacity,
            rounded_capacity=self.rounded_capacity,
            active_context=self.active_context,
            temporary_peak_bytes=0,
        )

    def pointers(self) -> dict[str, int]:
        names = (
            "packed_cache",
            "bf16_keys",
            "bf16_values",
            "block_table",
            "slot_mapping",
            "seq_lens",
            "Pi",
            "PiT",
            "centroids",
            "midpoints",
            "store_key_float",
            "store_value_float",
            "store_rotated_key",
            "store_norms",
            "store_norm_denominator",
            "decode_query_float",
            "decode_rotated_query",
            "decode_mid_o",
            "decode_output",
            "decode_lse",
            "reserved_workspace",
        )
        return {
            f"{name}_data_ptr": int(tensor.data_ptr())
            for name, tensor in zip(names, self._owned_tensors(), strict=True)
        }

    def layout_fingerprint(self) -> str:
        payload = {
            "schema": "kvbench-turboquant-static-cache-layout-1.1.0",
            "configuration": self.config_name,
            "num_layers": self.num_layers,
            "batch_size": self.batch_size,
            "num_query_heads": self.num_query_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "allocated_capacity": self.allocated_capacity,
            "rounded_capacity": self.rounded_capacity,
            "block_size": self.block_size,
            "block_count": self.block_count,
            "block_table": self.block_table.detach().cpu().tolist(),
            "slot_mapping": "deterministic_position_major_batch_banks",
            "slot_size": self.slot_size,
            "compressed_layers": list(self.compressed_layers),
            "bf16_layers": list(self.bf16_layers),
            "packed_shape": list(self.packed_cache.shape),
            "bf16_shape": list(self.bf16_cache.keys.shape),
            "packed_dtype": str(self.packed_cache.dtype),
            "bf16_dtype": str(self.bf16_cache.dtype),
            "device": str(self.device),
            "max_num_kv_splits": TURBOQUANT_MAX_KV_SPLITS,
            "persistent_breakdown": self.byte_breakdown(),
            "implementation_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def gqa_geometry(self) -> dict[str, Any]:
        return {
            "num_query_heads": self.num_query_heads,
            "num_kv_heads": self.num_kv_heads,
            "gqa_group_size": self.num_query_heads // self.num_kv_heads,
            "native_kv_head_storage": True,
            "gqa_materialized": False,
            "packed_cache_shape": list(self.packed_cache.shape),
            "bf16_cache_shape": list(self.bf16_cache.keys.shape),
        }

    def initialize_deterministic(self) -> None:
        self.packed_cache.zero_()
        self.bf16_cache.initialize_deterministic()
        self._active_context = 0
        self._clear_lifecycle()

    def _clear_lifecycle(self) -> None:
        self._mode = "idle"
        self._prefix_length = 0
        self._output_steps = 0
        self._growing_step = -1
        self._current_slot_mapping = self.slot_mapping[:0]
        self._current_seq_lens = self._single_seq_lens[0]
        for handle in self._handles.values():
            handle.key_states = None
            handle.value_states = None
            handle.prefill = False

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

    def reset_active_length(self, length: int = 0) -> None:
        checked = self._check_length(length, allow_zero=True)
        self.bf16_cache.reset_active_length(checked)
        self._active_context = checked
        self._clear_lifecycle()

    def prepare_prefill(self, prefix_length: int) -> None:
        length = self._check_length(prefix_length)
        self.bf16_cache.prepare_prefill(length)
        self._mode = "prefill"
        self._prefix_length = length
        self._current_slot_mapping = self.slot_mapping[
            : length * self.batch_size
        ]
        self._current_seq_lens = self._single_seq_lens[length - 1]

    def complete_prefill(self) -> None:
        if self._mode != "prefill":
            raise CacheStateError("complete_prefill requires prefill mode")
        self.bf16_cache.complete_prefill()
        self._active_context = self._prefix_length
        self._mode = "ready"

    def prepare_fixed(self, prefix_length: int) -> None:
        length = self._check_length(prefix_length)
        if length >= self.capacity:
            raise CacheBoundsError("fixed-L requires a reserved scratch position")
        if self._active_context != length:
            raise CacheStateError(
                "fixed-L prefix length does not match active state"
            )
        self.bf16_cache.prepare_fixed(length)
        self._mode = "fixed"
        self._prefix_length = length
        self._current_slot_mapping = self._single_slot_mappings[length]
        self._current_seq_lens = self._single_seq_lens[length]

    def prepare_growing(self, prefix_length: int, output_steps: int) -> None:
        length = self._check_length(prefix_length)
        steps = _positive_int(output_steps, "output_steps")
        if length + steps > self.capacity:
            raise CacheBoundsError(
                "growing trajectory exceeds allocated capacity"
            )
        if self._active_context != length:
            raise CacheStateError(
                "growing prefix length does not match active state"
            )
        self.bf16_cache.prepare_growing(length, steps)
        self._mode = "growing_ready"
        self._prefix_length = length
        self._output_steps = steps
        self._growing_step = -1

    def select_growing_step(self, step: int) -> None:
        if self._mode not in {"growing_ready", "growing_step"}:
            raise CacheStateError(
                "growing step selection requires growing mode"
            )
        if isinstance(step, bool) or not isinstance(step, int):
            raise CacheBoundsError("growing step must be an integer")
        if step < 0 or step >= self._output_steps:
            raise CacheBoundsError(
                "growing step is outside the declared trajectory"
            )
        if self._active_context != self._prefix_length + step:
            raise CacheStateError(
                "growing step would drift from active context"
            )
        self.bf16_cache.select_growing_step(step)
        position = self._prefix_length + step
        self._current_slot_mapping = self._single_slot_mappings[position]
        self._current_seq_lens = self._single_seq_lens[position]
        self._growing_step = step
        self._mode = "growing_step"

    def finish_growing_step(self) -> None:
        if self._mode != "growing_step" or self._growing_step < 0:
            raise CacheStateError("no growing step is active")
        self.bf16_cache.finish_growing_step()
        self._active_context = self._prefix_length + self._growing_step + 1
        self._mode = "growing_ready"

    def compressed_layer_cache(self, layer_idx: int) -> Any:
        try:
            slot = self._compressed_layer_slots[layer_idx]
        except KeyError as error:
            raise CacheBoundsError(
                "layer is not a TurboQuant-compressed layer"
            ) from error
        return self.packed_cache[slot]

    def update_bf16(
        self,
        key_states: Any,
        value_states: Any,
        layer_idx: int,
        cache_position: Any,
    ) -> tuple[Any, Any]:
        try:
            slot = self._bf16_layer_slots[layer_idx]
        except KeyError as error:
            raise CacheBoundsError(
                "layer is not a BF16 boundary layer"
            ) from error
        return self.bf16_cache.update(
            key_states,
            value_states,
            slot,
            {"cache_position": cache_position},
        )

    def attended_handle(
        self,
        layer_idx: int,
        *,
        key_states: Any | None,
        value_states: Any | None,
        prefill: bool,
    ) -> TurboQuantAttentionHandle:
        try:
            handle = self._handles[layer_idx]
        except KeyError as error:
            raise CacheBoundsError(
                "layer is not a TurboQuant-compressed layer"
            ) from error
        handle.key_states = key_states
        handle.value_states = value_states
        handle.prefill = prefill
        return handle

    def history_sha256(self, historical_length: int) -> str:
        """Hash the untimed packed and BF16 prefix in deterministic order."""

        length = self._check_length(historical_length)
        if self.batch_size == 1:
            packed = self.packed_cache.reshape(
                len(self.compressed_layers),
                self.rounded_capacity,
                self.num_kv_heads,
                self.slot_size,
            )[:, :length]
        else:
            packed = self.packed_cache.reshape(
                len(self.compressed_layers),
                self.batch_size,
                self.rounded_capacity,
                self.num_kv_heads,
                self.slot_size,
            )[:, :, :length]
        bf16_keys, bf16_values = self.bf16_cache.historical_tensors(length)
        header = json.dumps(
            {
                "schema": "kvbench-turboquant-history-1.0.0",
                "configuration": self.config_name,
                "length": length,
                "packed_shape": list(packed.shape),
                "bf16_shape": list(bf16_keys.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256()
        digest.update(header)
        for tensor in (packed, bf16_keys, bf16_values):
            digest.update(b"\0")
            digest.update(_tensor_bytes_untimed(tensor))
        return digest.hexdigest()

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise CacheBoundsError("layer_idx is outside the cache allocation")
        return self.active_context

    def get_mask_sizes(
        self,
        cache_position: Any,
        layer_idx: int,
    ) -> tuple[int, int]:
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise CacheBoundsError("layer_idx is outside the cache allocation")
        query_length = int(cache_position.shape[0])
        return self.active_context + query_length, 0

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise CacheBoundsError("layer_idx is outside the cache allocation")
        return self.capacity
