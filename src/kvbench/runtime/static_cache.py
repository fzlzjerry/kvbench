"""Preallocated BF16 KV-head cache with explicit Phase 3 lifecycle semantics."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Mapping


_TORCH: Any | None = None


class CacheBoundsError(ValueError):
    """A cache position or requested capacity is outside the allocation."""


class CacheStateError(RuntimeError):
    """A cache lifecycle operation is invalid in the current state."""


@dataclass(frozen=True, slots=True)
class CacheAccounting:
    """Exact logical and storage-byte accounting for one static cache."""

    predicted_tensor_bytes: int
    measured_tensor_bytes: int
    allocated_bytes: int
    padding_bytes: int
    workspace_bytes: int
    capacity: int
    active_context: int

    def to_dict(self) -> dict[str, int]:
        return {
            "predicted_tensor_bytes": self.predicted_tensor_bytes,
            "measured_tensor_bytes": self.measured_tensor_bytes,
            "allocated_bytes": self.allocated_bytes,
            "padding_bytes": self.padding_bytes,
            "workspace_bytes": self.workspace_bytes,
            "capacity": self.capacity,
            "active_context": self.active_context,
        }


def _torch() -> Any:
    global _TORCH
    if _TORCH is None:
        try:
            _TORCH = importlib.import_module("torch")
        except ModuleNotFoundError as error:
            raise CacheStateError("PyTorch is required for BF16StaticCache") from error
    return _TORCH


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def cache_accounting_for_geometry(
    *,
    num_layers: int,
    batch_size: int,
    num_kv_heads: int,
    capacity: int,
    head_dim: int,
    workspace_bytes: int = 0,
) -> dict[str, int]:
    """Compute exact cache bytes without importing PyTorch or allocating CUDA."""

    layers = _positive_int(num_layers, "num_layers")
    batch = _positive_int(batch_size, "batch_size")
    kv_heads = _positive_int(num_kv_heads, "num_kv_heads")
    positions = _positive_int(capacity, "capacity")
    dimension = _positive_int(head_dim, "head_dim")
    if isinstance(workspace_bytes, bool) or workspace_bytes < 0:
        raise ValueError("workspace_bytes must be a nonnegative integer")
    tensor_bytes = 2 * layers * batch * kv_heads * positions * dimension * 2
    return {
        "predicted_tensor_bytes": tensor_bytes,
        "padding_bytes": 0,
        "workspace_bytes": int(workspace_bytes),
        "allocated_bytes": tensor_bytes + int(workspace_bytes),
    }


def layout_fingerprint_for_geometry(
    *,
    num_layers: int,
    batch_size: int,
    num_kv_heads: int,
    capacity: int,
    head_dim: int,
    device: str = "cuda:0",
    workspace_bytes: int = 0,
) -> str:
    """Create the manifest fingerprint before worker/device initialization."""

    accounting = cache_accounting_for_geometry(
        num_layers=num_layers,
        batch_size=batch_size,
        num_kv_heads=num_kv_heads,
        capacity=capacity,
        head_dim=head_dim,
        workspace_bytes=workspace_bytes,
    )
    shape = (
        num_layers,
        batch_size,
        num_kv_heads,
        capacity,
        head_dim,
    )
    strides = (
        batch_size * num_kv_heads * capacity * head_dim,
        num_kv_heads * capacity * head_dim,
        capacity * head_dim,
        head_dim,
        1,
    )
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    payload = {
        "schema": "kvbench-bf16-static-cache-layout-1.0.0",
        "shape": list(shape),
        "strides": list(strides),
        "dtype": "torch.bfloat16",
        "element_size": 2,
        "device": device,
        "padding_bytes": accounting["padding_bytes"],
        "workspace_bytes": int(workspace_bytes),
        "implementation_sha256": source_hash,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class BF16StaticCache:
    """Two pointer-stable tensors storing K/V with native KV-head geometry."""

    def __init__(
        self,
        *,
        num_layers: int,
        batch_size: int,
        num_kv_heads: int,
        capacity: int,
        head_dim: int,
        device: Any,
        dtype: Any | None = None,
        workspace_bytes: int = 0,
    ) -> None:
        torch = _torch()
        self.num_layers = _positive_int(num_layers, "num_layers")
        self.batch_size = _positive_int(batch_size, "batch_size")
        self.num_kv_heads = _positive_int(num_kv_heads, "num_kv_heads")
        self.capacity = _positive_int(capacity, "capacity")
        self.head_dim = _positive_int(head_dim, "head_dim")
        if isinstance(workspace_bytes, bool) or workspace_bytes < 0:
            raise ValueError("workspace_bytes must be a nonnegative integer")
        self.workspace_bytes = int(workspace_bytes)
        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if dtype is None else dtype
        if self.dtype != torch.bfloat16:
            raise ValueError("BF16StaticCache accepts only torch.bfloat16")
        shape = (
            self.num_layers,
            self.batch_size,
            self.num_kv_heads,
            self.capacity,
            self.head_dim,
        )
        self.keys = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.values = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self._active_context = 0
        self._mode = "idle"
        self._prefix_length = 0
        self._output_steps = 0
        self._growing_step = -1
        self._destinations: tuple[tuple[Any, Any], ...] = ()
        self._attended: tuple[tuple[Any, Any], ...] = ()
        self._growing_destinations: tuple[
            tuple[tuple[Any, Any], ...], ...
        ] = ()
        self._growing_attended: tuple[
            tuple[tuple[Any, Any], ...], ...
        ] = ()

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
    def tensor_storage_bytes(self) -> int:
        return int(self.keys.untyped_storage().nbytes()) + int(
            self.values.untyped_storage().nbytes()
        )

    @property
    def predicted_tensor_bytes(self) -> int:
        return (
            2
            * self.num_layers
            * self.batch_size
            * self.num_kv_heads
            * self.capacity
            * self.head_dim
            * 2
        )

    def accounting(self) -> CacheAccounting:
        measured = self.tensor_storage_bytes
        return CacheAccounting(
            predicted_tensor_bytes=self.predicted_tensor_bytes,
            measured_tensor_bytes=measured,
            allocated_bytes=measured + self.workspace_bytes,
            padding_bytes=measured - self.predicted_tensor_bytes,
            workspace_bytes=self.workspace_bytes,
            capacity=self.capacity,
            active_context=self.active_context,
        )

    def pointers(self) -> dict[str, int]:
        """Return pointer metadata outside any measured decode boundary."""

        return {
            "keys_data_ptr": int(self.keys.data_ptr()),
            "values_data_ptr": int(self.values.data_ptr()),
            "keys_storage_ptr": int(self.keys.untyped_storage().data_ptr()),
            "values_storage_ptr": int(self.values.untyped_storage().data_ptr()),
        }

    def layout_fingerprint(self) -> str:
        return layout_fingerprint_for_geometry(
            num_layers=self.num_layers,
            batch_size=self.batch_size,
            num_kv_heads=self.num_kv_heads,
            capacity=self.capacity,
            head_dim=self.head_dim,
            device=str(self.device),
            workspace_bytes=self.workspace_bytes,
        )

    def initialize_deterministic(self) -> None:
        """Zero both backing tensors outside measured execution."""

        self.keys.zero_()
        self.values.zero_()
        self._active_context = 0
        self._clear_views()

    def _clear_views(self) -> None:
        self._mode = "idle"
        self._prefix_length = 0
        self._output_steps = 0
        self._growing_step = -1
        self._destinations = ()
        self._attended = ()
        self._growing_destinations = ()
        self._growing_attended = ()

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
        """Reset metadata only; stale tail storage is never exposed implicitly."""

        self._active_context = self._check_length(length, allow_zero=True)
        self._clear_views()

    def prepare_prefill(self, prefix_length: int) -> None:
        length = self._check_length(prefix_length)
        self._mode = "prefill"
        self._prefix_length = length
        self._growing_step = -1
        self._destinations = tuple(
            (
                self.keys[layer, :, :, :length, :],
                self.values[layer, :, :, :length, :],
            )
            for layer in range(self.num_layers)
        )
        self._attended = self._destinations

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
        self._mode = "fixed"
        self._prefix_length = length
        self._destinations = tuple(
            (
                self.keys[layer, :, :, length : length + 1, :],
                self.values[layer, :, :, length : length + 1, :],
            )
            for layer in range(self.num_layers)
        )
        self._attended = tuple(
            (
                self.keys[layer, :, :, : length + 1, :],
                self.values[layer, :, :, : length + 1, :],
            )
            for layer in range(self.num_layers)
        )

    def prepare_growing(self, prefix_length: int, output_steps: int) -> None:
        length = self._check_length(prefix_length)
        steps = _positive_int(output_steps, "output_steps")
        if length + steps > self.capacity:
            raise CacheBoundsError("growing trajectory exceeds allocated capacity")
        if self._active_context != length:
            raise CacheStateError("growing prefix length does not match active state")
        self._mode = "growing_ready"
        self._prefix_length = length
        self._output_steps = steps
        self._growing_step = -1
        self._growing_destinations = tuple(
            tuple(
                (
                    self.keys[
                        layer,
                        :,
                        :,
                        length + step : length + step + 1,
                        :,
                    ],
                    self.values[
                        layer,
                        :,
                        :,
                        length + step : length + step + 1,
                        :,
                    ],
                )
                for layer in range(self.num_layers)
            )
            for step in range(steps)
        )
        self._growing_attended = tuple(
            tuple(
                (
                    self.keys[layer, :, :, : length + step + 1, :],
                    self.values[layer, :, :, : length + step + 1, :],
                )
                for layer in range(self.num_layers)
            )
            for step in range(steps)
        )

    def select_growing_step(self, step: int) -> None:
        if self._mode not in {"growing_ready", "growing_step"}:
            raise CacheStateError("growing step selection requires growing mode")
        if isinstance(step, bool) or not isinstance(step, int):
            raise CacheBoundsError("growing step must be an integer")
        if step < 0 or step >= self._output_steps:
            raise CacheBoundsError("growing step is outside the declared trajectory")
        if self._active_context != self._prefix_length + step:
            raise CacheStateError("growing step would drift from active context")
        self._growing_step = step
        self._mode = "growing_step"

    def finish_growing_step(self) -> None:
        if self._mode != "growing_step" or self._growing_step < 0:
            raise CacheStateError("no growing step is active")
        self._active_context = self._prefix_length + self._growing_step + 1
        self._mode = "growing_ready"

    def update(
        self,
        key_states: Any,
        value_states: Any,
        layer_idx: int,
        cache_kwargs: Mapping[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        """Write only the declared slot and return its precreated attended view."""

        del cache_kwargs
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise CacheBoundsError("layer_idx is outside the cache allocation")
        if self._mode in {"prefill", "fixed"}:
            destination = self._destinations[layer_idx]
            attended = self._attended[layer_idx]
        elif self._mode == "growing_step" and self._growing_step >= 0:
            destination = self._growing_destinations[self._growing_step][layer_idx]
            attended = self._growing_attended[self._growing_step][layer_idx]
        else:
            raise CacheStateError("cache update is not enabled in the current mode")
        expected_shape = tuple(int(item) for item in destination[0].shape)
        if tuple(int(item) for item in key_states.shape) != expected_shape:
            raise CacheStateError("key update shape differs from the declared slot")
        if tuple(int(item) for item in value_states.shape) != expected_shape:
            raise CacheStateError("value update shape differs from the declared slot")
        if key_states.dtype != self.dtype or value_states.dtype != self.dtype:
            raise CacheStateError("cache updates must remain BF16")
        if key_states.device != self.device or value_states.device != self.device:
            raise CacheStateError("cache update device differs from cache storage")
        destination[0].copy_(key_states)
        destination[1].copy_(value_states)
        return attended

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise CacheBoundsError("layer_idx is outside the cache allocation")
        return self.active_context

    def get_mask_sizes(self, cache_position: Any, layer_idx: int) -> tuple[int, int]:
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise CacheBoundsError("layer_idx is outside the cache allocation")
        query_length = int(cache_position.shape[0])
        return self.active_context + query_length, 0

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise CacheBoundsError("layer_idx is outside the cache allocation")
        return self.capacity

    def historical_tensors(self, length: int) -> tuple[Any, Any]:
        """Expose a read-only-intended view for untimed validation helpers."""

        checked = self._check_length(length)
        return self.keys[:, :, :, :checked, :], self.values[:, :, :, :checked, :]
