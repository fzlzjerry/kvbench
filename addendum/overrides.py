"""Container-side implementation variants for the addendum (no frozen file is edited).

Each installer patches the imported frozen classes in this process only and
returns a state dict that the worker records next to its result.
"""

from __future__ import annotations

import hashlib
import inspect
from typing import Any

AS_PORTED_SPLITS = 4


def _source_sha256(function: Any) -> str:
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


def install_tq_split_override(splits: int) -> dict[str, Any]:
    """Launch TurboQuant stage 1/2 with `splits` KV splits.

    The FP32 split scratch (B, Hq, splits, D + 1) is allocated once, when the
    cache is constructed (before prefill, warmup and graph capture), as vLLM
    preallocates its split buffers for CUDA Graphs.  Inside each decode call
    the scratch and the split constant are swapped in and restored on return,
    so the cache's owned tensors, byte accounting, fingerprints and pointer
    checks are those of the as-ported configuration; the scratch bytes are
    reported separately.
    """
    import torch
    import kvbench.adapters.turboquant as adapter_module
    import kvbench.runtime.turboquant_cache as cache_module

    if adapter_module.TURBOQUANT_MAX_KV_SPLITS != AS_PORTED_SPLITS:
        raise RuntimeError("as-ported TurboQuant split count differs from 4")
    if cache_module.TURBOQUANT_MAX_KV_SPLITS != AS_PORTED_SPLITS:
        raise RuntimeError("cache TurboQuant split count differs from 4")
    state: dict[str, Any] = {
        "variant": "tq_splits",
        "splits": int(splits),
        "as_ported_splits": AS_PORTED_SPLITS,
        "scratch": {},
        "scratch_records": [],
        "decode_calls": 0,
        "decode_calls_during_capture": 0,
    }
    cache_class = cache_module.TurboQuantStaticCache
    original_init = cache_class.__init__

    def init_with_split_scratch(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("TurboQuant cache constructed during graph capture")
        scratch = torch.empty(
            (self.batch_size, self.num_query_heads, splits, self.head_dim + 1),
            dtype=torch.float32,
            device=self.device,
        )
        # Keyed by the as-ported scratch pointer: decode handles reach the
        # cache through a weakref proxy, whose id() differs from the cache's.
        state["scratch"][self.decode_mid_o.data_ptr()] = scratch
        state["scratch_records"].append({
            "allocated_at": "cache_construction",
            "shape": list(scratch.shape),
            "bytes": scratch.numel() * scratch.element_size(),
            "data_ptr": scratch.data_ptr(),
        })

    adapter = adapter_module.TurboQuantMethodAdapter
    raw = adapter.__dict__["_decode_compressed"]
    original = raw.__func__ if isinstance(raw, staticmethod) else raw

    def decode_with_splits(handle: Any, query_states: Any, scaling: float) -> Any:
        cache = handle.cache
        scratch = state["scratch"].get(cache.decode_mid_o.data_ptr())
        if scratch is None:
            raise RuntimeError(
                "split scratch was not allocated at cache construction: "
                f"cache key {cache.decode_mid_o.data_ptr()} type {type(cache).__module__}.{type(cache).__qualname__} "
                f"init {getattr(type(cache).__init__, '__qualname__', '?')} "
                f"class is patched class {type(cache) is cache_class} "
                f"recorded ids {list(state['scratch'])} records {len(state['scratch_records'])}")
        saved_scratch = cache.decode_mid_o
        saved_splits = adapter_module.TURBOQUANT_MAX_KV_SPLITS
        cache.decode_mid_o = scratch
        adapter_module.TURBOQUANT_MAX_KV_SPLITS = splits
        state["decode_calls"] += 1
        state["decode_calls_during_capture"] += int(torch.cuda.is_current_stream_capturing())
        try:
            return original(handle, query_states, scaling)
        finally:
            cache.decode_mid_o = saved_scratch
            adapter_module.TURBOQUANT_MAX_KV_SPLITS = saved_splits

    cache_class.__init__ = init_with_split_scratch
    adapter._decode_compressed = staticmethod(decode_with_splits)
    state["override_source_sha256"] = {
        "init_with_split_scratch": _source_sha256(init_with_split_scratch),
        "decode_with_splits": _source_sha256(decode_with_splits),
    }
    return state


def install_kivi_grouped_residual() -> dict[str, Any]:
    """Replace KIVI's per-query-head residual bmm/add loops with per-KV-head batches.

    Query head qh = kv * G + g (G = 4) attends KV head kv, the mapping of the
    existing adapter (kv_head = query_head // KIVI_GQA_GROUP_SIZE).  Viewing the
    [B, 32, ...] query/score/output buffers as [B * 8, 4, ...] and the residual
    K/V as [B * 8, R, 128] expresses that mapping without copies, so each
    residual product is one bmm over B * 8 batches of 4 query rows, as in the
    KV-head-grouped source patch (third_party/patches/kivi/0001, kivi_gqa.py).
    Everything else in _decode_compressed is the existing code verbatim:
    quantized-history kernels, scaling, fixed-capacity softmax, ring unrolling,
    accumulation order (history, residual, pending), commit and rollover.
    """
    import kvbench.adapters.kivi as kivi_module

    adapter = kivi_module.KIVIMethodAdapter
    original = adapter.__dict__["_decode_compressed"]
    group = kivi_module.KIVI_GQA_GROUP_SIZE
    _torch = kivi_module._torch
    CacheStateError = kivi_module.CacheStateError
    KIVI_GROUP_SIZE = kivi_module.KIVI_GROUP_SIZE
    state: dict[str, Any] = {"variant": "kivi_grouped_residual", "gqa_group_size": group,
                             "decode_calls": 0}

    def _decode_compressed(self: Any, handle: Any, query_states: Any, scaling: float) -> Any:
        cache = handle.cache
        if cache.device.type != "cuda":
            raise CacheStateError("KIVI compressed kernels require CUDA execution")
        if tuple(int(x) for x in query_states.shape) != (
            cache.batch_size,
            cache.num_query_heads,
            1,
            cache.head_dim,
        ):
            raise CacheStateError("KIVI decode query has unsupported geometry")
        if query_states.dtype != _torch().bfloat16 or query_states.device != cache.device:
            raise CacheStateError("KIVI decode query differs from BF16 cache device")
        if cache.num_query_heads != cache.num_kv_heads * group:
            raise CacheStateError("KIVI grouped residual requires 32 query heads over 8 KV heads")
        state["decode_calls"] += 1
        _, _, launcher = self._runtime()
        cache.query_fp16_staging.copy_(query_states[:, :, 0, :])
        historical = cache._key_history_counts[handle.layer_idx]
        residual = cache._key_residual_counts[handle.layer_idx]
        total = historical + residual + 1
        if total > cache.capacity or handle.pending_key is None or handle.pending_value is None:
            raise CacheStateError("KIVI decode state is incomplete")

        # KV-head-grouped views: row (b * 8 + kv, g) is query head kv * 4 + g of batch b.
        grouped_batches = cache.batch_size * cache.num_kv_heads
        grouped_query = cache.query_fp16_staging.view(grouped_batches, group, cache.head_dim)
        grouped_logits = cache.decode_logits.view(grouped_batches, group, cache.decode_logits.shape[-1])

        logits = cache.decode_logits[:, :, :total]
        if historical:
            kernel_history = cache.key_history_capacity
            packed = cache.packed_key_history[handle.layer_idx]
            scales = cache.key_scales[handle.layer_idx]
            minimums = cache.key_minimums[handle.layer_idx]
            kernel_output = cache.key_kernel_output_fp16.view(-1)[
                : cache.batch_size * cache.num_query_heads * kernel_history
            ].view(
                cache.batch_size * cache.num_query_heads,
                1,
                kernel_history,
            )
            launcher.launch_into(
                input_tensor=cache.query_fp16_staging.view(
                    cache.batch_size * cache.num_query_heads,
                    1,
                    cache.head_dim,
                ),
                packed=packed.view(
                    cache.batch_size * cache.num_kv_heads,
                    -1,
                    cache.head_dim,
                ),
                scales=scales.view(
                    cache.batch_size * cache.num_kv_heads,
                    -1,
                    cache.head_dim,
                ),
                minimums=minimums.view(
                    cache.batch_size * cache.num_kv_heads,
                    -1,
                    cache.head_dim,
                ),
                output=kernel_output,
                bits=self.k_bits,
                group_size=KIVI_GROUP_SIZE,
                num_query_heads=32,
                num_kv_heads=8,
            )
            logits[:, :, :historical].copy_(
                kernel_output[:, :, :historical].view(
                    cache.batch_size,
                    cache.num_query_heads,
                    historical,
                )
            )
        if residual:
            _torch().bmm(
                grouped_query,
                cache.key_residual[handle.layer_idx]
                .view(grouped_batches, cache.residual_length, cache.head_dim)[:, :residual, :]
                .transpose(-1, -2),
                out=grouped_logits[:, :, historical : historical + residual],
            )
        _torch().bmm(
            grouped_query,
            handle.pending_key.view(grouped_batches, 1, cache.head_dim).transpose(-1, -2),
            out=grouped_logits[:, :, historical + residual : total],
        )
        # The frozen reference applies the scale while scores are still FP16,
        # then requests an FP32 softmax accumulator.
        logits.mul_(float(scaling))
        # Fixed-capacity softmax with an inactive -inf tail (existing adapter).
        cache.decode_softmax.fill_(float("-inf"))
        cache.decode_softmax[:, :, :total].copy_(logits)
        _torch().softmax(
            cache.decode_softmax,
            dim=-1,
            out=cache.decode_softmax,
        )
        logits.copy_(cache.decode_softmax[:, :, :total])

        value_history = cache._value_history_counts[handle.layer_idx]
        cache.decode_output_fp16.zero_()
        if value_history:
            kernel_history = cache.value_history_capacity
            value_weights = cache.key_kernel_output_fp16.view(-1)[
                : cache.batch_size * cache.num_query_heads * kernel_history
            ].view(
                cache.batch_size * cache.num_query_heads,
                1,
                kernel_history,
            )
            value_weights.zero_()
            value_weights[:, :, :value_history].copy_(
                logits[:, :, :value_history].view(
                    cache.batch_size * cache.num_query_heads,
                    1,
                    value_history,
                )
            )
            launcher.launch_into(
                input_tensor=value_weights,
                packed=cache.packed_value_history[
                    handle.layer_idx
                ].view(
                    cache.batch_size * cache.num_kv_heads,
                    -1,
                    kernel_history,
                ),
                scales=cache.value_scales[
                    handle.layer_idx
                ].view(
                    cache.batch_size * cache.num_kv_heads,
                    -1,
                    kernel_history,
                ),
                minimums=cache.value_minimums[
                    handle.layer_idx
                ].view(
                    cache.batch_size * cache.num_kv_heads,
                    -1,
                    kernel_history,
                ),
                output=cache.decode_output_fp16.view(
                    cache.batch_size * cache.num_query_heads,
                    1,
                    cache.head_dim,
                ),
                bits=self.v_bits,
                group_size=KIVI_GROUP_SIZE,
                num_query_heads=32,
                num_kv_heads=8,
            )
        value_residual = cache._value_residual_counts[handle.layer_idx]
        residual_values = cache.value_residual_ring[handle.layer_idx]
        if value_residual:
            head = cache._value_residual_heads[handle.layer_idx]
            ordered = cache.value_residual_ordered_staging[handle.layer_idx]
            first = min(value_residual, cache.residual_length - head)
            ordered[:, :, :first, :].copy_(
                residual_values[:, :, head : head + first, :]
            )
            if value_residual > first:
                ordered[:, :, first:value_residual, :].copy_(
                    residual_values[:, :, : value_residual - first, :]
                )
            residual_values = ordered
        grouped_merge = cache.decode_merge.view(grouped_batches, group, cache.head_dim)
        grouped_output = cache.decode_output_fp16.view(grouped_batches, group, cache.head_dim)
        if value_residual:
            _torch().bmm(
                grouped_logits[:, :, value_history : value_history + value_residual],
                residual_values.view(grouped_batches, cache.residual_length, cache.head_dim)[
                    :, :value_residual, :
                ],
                out=grouped_merge,
            )
            grouped_output.add_(grouped_merge)
        _torch().bmm(
            grouped_logits[:, :, value_history + value_residual : total],
            handle.pending_value.view(grouped_batches, 1, cache.head_dim),
            out=grouped_merge,
        )
        grouped_output.add_(grouped_merge)
        cache.output_buffer.copy_(cache.decode_output_fp16)
        if handle.commit_after_decode:
            self._commit_token(
                cache,
                handle.layer_idx,
                handle.pending_key,
                handle.pending_value,
                self._layer_context(cache, handle.layer_idx),
            )
            handle.commit_after_decode = False
        return cache.output_buffer.unsqueeze(2)

    adapter._decode_compressed = _decode_compressed
    state["replaced_source_sha256"] = _source_sha256(original)
    state["override_source_sha256"] = _source_sha256(_decode_compressed)
    return state


def describe(state: dict[str, Any] | None) -> dict[str, Any] | None:
    """JSON-safe summary of an installer state."""
    if state is None:
        return None
    return {key: value for key, value in state.items() if key != "scratch"}
