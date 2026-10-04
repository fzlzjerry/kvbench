"""Thin Phase 6 adapter over the pinned upstream TurboQuant kernels."""

from __future__ import annotations

import hashlib
import importlib
import math
from pathlib import Path
from typing import Any, Mapping

from kvbench.adapters.base import MethodRuntimeContext
from kvbench.runtime.backend import flash_attention_forward
from kvbench.runtime.static_cache import CacheStateError
from kvbench.runtime.turboquant_cache import (
    TURBOQUANT_BF16_LAYERS,
    TURBOQUANT_COMPRESSED_LAYERS,
    TURBOQUANT_MANDATORY_CONFIGS,
    TURBOQUANT_MAX_KV_SPLITS,
    TurboQuantAttentionHandle,
    TurboQuantStaticCache,
)
from kvbench.schema import canonical_json_bytes, sha256_hex
from kvbench.schema.base import require_sha256


TURBOQUANT_ADAPTER_VERSION = "kvbench-turboquant-method-adapter-1.1.0"
TURBOQUANT_ADAPTER_FINGERPRINT_SCHEMA_VERSION = (
    "kvbench-turboquant-method-adapter-config-1.1.0"
)
TURBOQUANT_SOURCE_COMMIT = "752a3a504485790a2e8491cacbb35c137339ad34"
TURBOQUANT_SOURCE_TREE = "3ec7a4eb00f9bc8fec399bea6cf7de27a7936372"
TURBOQUANT_FIXTURE_SET_SHA256 = (
    "774ec946a8839d4de012bc6fba0ee5a933ab1488ecc43354d8573b4481b12f76"
)
_TORCH: Any | None = None


def _torch() -> Any:
    global _TORCH
    if _TORCH is None:
        try:
            _TORCH = importlib.import_module("torch")
        except ModuleNotFoundError as error:
            raise CacheStateError(
                "PyTorch is required for the TurboQuant adapter"
            ) from error
    return _TORCH


class TurboQuantMethodAdapter:
    """One explicit adapter for the mandatory MSE+NC TurboQuant family."""

    name = "turboquant"
    adapter_version = TURBOQUANT_ADAPTER_VERSION

    def __init__(
        self,
        runtime_context: MethodRuntimeContext,
        config_name: str,
    ) -> None:
        if type(runtime_context) is not MethodRuntimeContext:
            raise TypeError(
                "TurboQuant adapter requires MethodRuntimeContext"
            )
        if (
            runtime_context.num_layers,
            runtime_context.num_query_heads,
            runtime_context.num_kv_heads,
            runtime_context.head_dim,
        ) != (32, 32, 8, 128):
            raise ValueError(
                "TurboQuant adapter requires frozen Llama GQA geometry"
            )
        if config_name not in TURBOQUANT_MANDATORY_CONFIGS:
            raise ValueError("unsupported TurboQuant configuration")
        self.runtime_context = runtime_context
        self.config_name = config_name

    def allocate(
        self,
        *,
        batch_size: int,
        capacity: int,
        device: Any,
        workspace_bytes: int = 0,
    ) -> TurboQuantStaticCache:
        return TurboQuantStaticCache(
            config_name=self.config_name,
            num_layers=self.runtime_context.num_layers,
            batch_size=batch_size,
            num_query_heads=self.runtime_context.num_query_heads,
            num_kv_heads=self.runtime_context.num_kv_heads,
            capacity=capacity,
            head_dim=self.runtime_context.head_dim,
            device=device,
            workspace_bytes=workspace_bytes,
        )

    def _require_cache(self, cache_state: Any) -> TurboQuantStaticCache:
        if type(cache_state) is not TurboQuantStaticCache:
            raise TypeError(
                "TurboQuant adapter cache state must be TurboQuantStaticCache"
            )
        if cache_state.config_name != self.config_name:
            raise CacheStateError(
                "TurboQuant adapter and cache configurations differ"
            )
        return cache_state

    @staticmethod
    def _validate_layer_and_inputs(
        cache: TurboQuantStaticCache,
        key_states: Any,
        value_states: Any,
        layer_idx: int,
        cache_position: Any,
    ) -> int:
        if (
            isinstance(layer_idx, bool)
            or not isinstance(layer_idx, int)
            or layer_idx < 0
            or layer_idx >= cache.num_layers
        ):
            raise CacheStateError(
                "layer_idx is outside the TurboQuant cache allocation"
            )
        expected_prefix = (
            cache.batch_size,
            cache.num_kv_heads,
        )
        if (
            int(key_states.ndim) != 4
            or int(value_states.ndim) != 4
            or tuple(int(item) for item in key_states.shape)
            != tuple(int(item) for item in value_states.shape)
            or tuple(int(item) for item in key_states.shape[:2])
            != expected_prefix
            or int(key_states.shape[3]) != cache.head_dim
        ):
            raise CacheStateError(
                "TurboQuant cache update has unsupported tensor geometry"
            )
        if (
            key_states.dtype != cache.dtype
            or value_states.dtype != cache.dtype
        ):
            raise CacheStateError("TurboQuant cache updates must remain BF16")
        if (
            key_states.device != cache.device
            or value_states.device != cache.device
            or cache_position.device != cache.device
        ):
            raise CacheStateError(
                "TurboQuant update device differs from cache storage"
            )
        tokens = int(key_states.shape[2])
        if (
            int(cache_position.ndim) != 1
            or int(cache_position.shape[0]) != tokens
            or int(cache.current_slot_mapping.shape[0])
            != tokens * cache.batch_size
        ):
            raise CacheStateError(
                "TurboQuant slot mapping differs from the declared update"
            )
        return tokens

    @staticmethod
    def _store_compressed(
        cache: TurboQuantStaticCache,
        key_states: Any,
        value_states: Any,
        layer_idx: int,
        tokens: int,
    ) -> None:
        if cache.device.type != "cuda":
            raise CacheStateError(
                "TurboQuant compressed kernels require CUDA execution"
            )
        torch = _torch()
        key_float = cache.store_key_float[:tokens]
        value_float = cache.store_value_float[:tokens]
        rotated = cache.store_rotated_key[:tokens]
        norms = cache.store_norms[:tokens]
        denominator = cache.store_norm_denominator[:tokens]
        key_float.copy_(key_states.permute(2, 0, 1, 3))
        value_float.copy_(value_states.permute(2, 0, 1, 3))

        flat_key = key_float.view(
            tokens * cache.batch_size * cache.num_kv_heads,
            cache.head_dim,
        )
        flat_value = value_float.view(
            tokens * cache.batch_size * cache.num_kv_heads,
            cache.head_dim,
        )
        flat_rotated = rotated.view(
            tokens * cache.batch_size * cache.num_kv_heads,
            cache.head_dim,
        )
        flat_norms = norms.view(
            tokens * cache.batch_size * cache.num_kv_heads, 1
        )
        flat_denominator = denominator.view(
            tokens * cache.batch_size * cache.num_kv_heads,
            1,
        )
        torch.linalg.vector_norm(
            flat_key,
            dim=1,
            keepdim=True,
            out=flat_norms,
        )
        torch.add(flat_norms, 1e-8, out=flat_denominator)
        torch.div(flat_key, flat_denominator, out=flat_key)
        torch.mm(flat_key, cache.PiT, out=flat_rotated)

        layer_cache = cache.compressed_layer_cache(layer_idx)
        tq_config = cache.tq_config
        mse_bits = int(tq_config.key_mse_bits)
        value_bits = int(tq_config.value_quant_bits)
        mse_bytes = math.ceil(cache.head_dim * mse_bits / 8)
        value_data_bytes = math.ceil(cache.head_dim * value_bits / 8)
        block_value = 1 << (value_data_bytes - 1).bit_length()
        grid = (tokens * cache.batch_size * cache.num_kv_heads,)
        cache._store_kernel[grid](
            flat_rotated,
            flat_norms,
            flat_value,
            cache.midpoints,
            layer_cache,
            cache.current_slot_mapping,
            stride_cache_block=layer_cache.stride(0),
            stride_cache_pos=layer_cache.stride(1),
            stride_cache_head=layer_cache.stride(2),
            D=cache.head_dim,
            H=cache.num_kv_heads,
            BLOCK_SIZE=cache.block_size,
            BLOCK_D=cache.head_dim,
            MSE_BYTES=mse_bytes,
            KPS=int(tq_config.key_packed_size),
            VQB=value_bits,
            VAL_DATA_BYTES=value_data_bytes,
            BLOCK_VAL=block_value,
            MSE_BITS=mse_bits,
            N_CENTROIDS=int(tq_config.n_centroids),
            BLOCK_GRP=16,
            num_warps=4,
            num_stages=1,
        )

    def store_prefill(
        self,
        cache_state: Any,
        key_states: Any,
        value_states: Any,
        layer_idx: int,
        cache_position: Any,
    ) -> tuple[Any, Any]:
        cache = self._require_cache(cache_state)
        if cache.mode != "prefill":
            raise CacheStateError(
                "TurboQuant prefill store requires prefill mode"
            )
        tokens = self._validate_layer_and_inputs(
            cache,
            key_states,
            value_states,
            layer_idx,
            cache_position,
        )
        if layer_idx in TURBOQUANT_BF16_LAYERS:
            return cache.update_bf16(
                key_states,
                value_states,
                layer_idx,
                cache_position,
            )
        if layer_idx not in TURBOQUANT_COMPRESSED_LAYERS:
            raise CacheStateError("TurboQuant layer policy is incomplete")
        self._store_compressed(
            cache,
            key_states,
            value_states,
            layer_idx,
            tokens,
        )
        handle = cache.attended_handle(
            layer_idx,
            key_states=key_states,
            value_states=value_states,
            prefill=True,
        )
        return handle, handle

    def append_decode(
        self,
        cache_state: Any,
        key_states: Any,
        value_states: Any,
        layer_idx: int,
        cache_position: Any,
    ) -> tuple[Any, Any]:
        cache = self._require_cache(cache_state)
        if cache.mode not in {"fixed", "growing_step"}:
            raise CacheStateError(
                "TurboQuant append requires fixed or growing mode"
            )
        tokens = self._validate_layer_and_inputs(
            cache,
            key_states,
            value_states,
            layer_idx,
            cache_position,
        )
        if tokens != 1:
            raise CacheStateError(
                "TurboQuant append requires exactly one token"
            )
        if layer_idx in TURBOQUANT_BF16_LAYERS:
            return cache.update_bf16(
                key_states,
                value_states,
                layer_idx,
                cache_position,
            )
        if layer_idx not in TURBOQUANT_COMPRESSED_LAYERS:
            raise CacheStateError("TurboQuant layer policy is incomplete")
        self._store_compressed(
            cache,
            key_states,
            value_states,
            layer_idx,
            tokens,
        )
        handle = cache.attended_handle(
            layer_idx,
            key_states=None,
            value_states=None,
            prefill=False,
        )
        return handle, handle

    @staticmethod
    def _decode_compressed(
        handle: TurboQuantAttentionHandle,
        query_states: Any,
        scaling: float,
    ) -> Any:
        cache = handle.cache
        if cache.device.type != "cuda":
            raise CacheStateError(
                "TurboQuant compressed kernels require CUDA execution"
            )
        torch = _torch()
        expected = (
            cache.batch_size,
            cache.num_query_heads,
            1,
            cache.head_dim,
        )
        if tuple(int(item) for item in query_states.shape) != expected:
            raise CacheStateError(
                "TurboQuant decode query has unsupported geometry"
            )
        if (
            query_states.dtype != cache.dtype
            or query_states.device != cache.device
        ):
            raise CacheStateError(
                "TurboQuant decode query differs from cache dtype/device"
            )
        cache.decode_query_float.copy_(query_states[:, :, 0, :])
        torch.mm(
            cache.decode_query_float.view(
                cache.batch_size * cache.num_query_heads,
                cache.head_dim,
            ),
            cache.PiT,
            out=cache.decode_rotated_query.view(
                cache.batch_size * cache.num_query_heads,
                cache.head_dim,
            ),
        )

        layer_cache = cache.compressed_layer_cache(handle.layer_idx)
        tq_config = cache.tq_config
        mse_bits = int(tq_config.key_mse_bits)
        value_bits = int(tq_config.value_quant_bits)
        mse_bytes = math.ceil(cache.head_dim * mse_bits / 8)
        value_data_bytes = math.ceil(cache.head_dim * value_bits / 8)
        cache._decode_stage1_kernel[
            (
                cache.batch_size,
                cache.num_query_heads,
                TURBOQUANT_MAX_KV_SPLITS,
            )
        ](
            cache.decode_rotated_query,
            layer_cache,
            cache.block_table,
            cache.current_seq_lens,
            cache.centroids,
            cache.decode_mid_o,
            cache.decode_rotated_query.stride(0),
            cache.decode_rotated_query.stride(1),
            layer_cache.stride(0),
            layer_cache.stride(1),
            layer_cache.stride(2),
            cache.block_table.stride(0),
            cache.decode_mid_o.stride(0),
            cache.decode_mid_o.stride(1),
            cache.decode_mid_o.stride(2),
            NUM_KV_HEADS=cache.num_kv_heads,
            HEAD_DIM=cache.head_dim,
            BLOCK_SIZE=cache.block_size,
            NUM_KV_SPLITS=TURBOQUANT_MAX_KV_SPLITS,
            KV_GROUP_SIZE=(
                cache.num_query_heads // cache.num_kv_heads
            ),
            MSE_BITS=mse_bits,
            MSE_BYTES=mse_bytes,
            KPS=int(tq_config.key_packed_size),
            VQB=value_bits,
            VAL_DATA_BYTES=value_data_bytes,
            ATTN_SCALE=float(scaling),
            BLOCK_D=cache.head_dim,
            BLOCK_KV=4,
            KEY_FP8=0,
            NORM_CORRECTION=1,
            FP8_E4B15=0,
            num_warps=1,
            num_stages=1,
        )
        cache._decode_stage2_kernel[
            (cache.batch_size, cache.num_query_heads)
        ](
            cache.decode_mid_o,
            cache.decode_output,
            cache.decode_lse,
            cache.current_seq_lens,
            cache.decode_mid_o.stride(0),
            cache.decode_mid_o.stride(1),
            cache.decode_mid_o.stride(2),
            cache.decode_output.stride(0),
            cache.decode_output.stride(1),
            cache.decode_lse.stride(0),
            NUM_KV_SPLITS=TURBOQUANT_MAX_KV_SPLITS,
            BLOCK_DV=cache.head_dim,
            Lv=cache.head_dim,
            OUTPUT_FP16=0,
            num_warps=4,
            num_stages=2,
        )
        return cache.decode_output

    def decode_attention(
        self,
        attention: Any,
        query_states: Any,
        key_states: Any,
        value_states: Any,
        *,
        scaling: float,
    ) -> Any:
        if isinstance(key_states, TurboQuantAttentionHandle):
            if key_states is not value_states:
                raise CacheStateError(
                    "TurboQuant attended handles must be identical"
                )
            handle = key_states
            if (
                handle.cache.config_name != self.config_name
                or int(attention.layer_idx) != handle.layer_idx
            ):
                raise CacheStateError(
                    "TurboQuant attended handle identity mismatch"
                )
            if handle.prefill:
                try:
                    if (
                        handle.key_states is None
                        or handle.value_states is None
                    ):
                        raise CacheStateError(
                            "TurboQuant prefill handle lost raw K/V"
                        )
                    output, _ = flash_attention_forward(
                        attention,
                        query_states,
                        handle.key_states,
                        handle.value_states,
                        None,
                        scaling,
                        dropout=0.0,
                    )
                    return output
                finally:
                    handle.key_states = None
                    handle.value_states = None
                    handle.prefill = False
            return self._decode_compressed(
                handle,
                query_states,
                scaling,
            )
        output, _ = flash_attention_forward(
            attention,
            query_states,
            key_states,
            value_states,
            None,
            scaling,
            dropout=0.0,
        )
        return output

    def allocated_bytes(self, cache_state: Any) -> int:
        return self._require_cache(cache_state).accounting().allocated_bytes

    def byte_breakdown(self, cache_state: Any) -> Mapping[str, int]:
        return self._require_cache(cache_state).byte_breakdown()

    def logical_bf16_bytes(self, cache_state: Any) -> int:
        return self._require_cache(cache_state).logical_bf16_storage_bytes

    def config_fingerprint(self, cache_layout_fingerprint: str) -> str:
        require_sha256(
            cache_layout_fingerprint,
            field_name="cache_layout_fingerprint",
        )
        context = self.runtime_context
        payload = {
            "schema_version": (
                TURBOQUANT_ADAPTER_FINGERPRINT_SCHEMA_VERSION
            ),
            "adapter_version": self.adapter_version,
            "method_name": self.name,
            "configuration": self.config_name,
            "cache_layout_fingerprint": cache_layout_fingerprint,
            "model_id": context.model_id,
            "model_revision": context.model_revision,
            "backend_id": context.backend_id,
            "backend_fingerprint": context.backend_fingerprint,
            "num_layers": context.num_layers,
            "num_query_heads": context.num_query_heads,
            "num_kv_heads": context.num_kv_heads,
            "head_dim": context.head_dim,
            "compressed_layers": list(TURBOQUANT_COMPRESSED_LAYERS),
            "bf16_layers": list(TURBOQUANT_BF16_LAYERS),
            "pinned_source_commit": TURBOQUANT_SOURCE_COMMIT,
            "pinned_source_tree": TURBOQUANT_SOURCE_TREE,
            "fixture_set_sha256": TURBOQUANT_FIXTURE_SET_SHA256,
            "supports_cuda_graph": self.supports_cuda_graph(),
            "implementation_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        }
        return sha256_hex(canonical_json_bytes(payload))

    def supports_cuda_graph(self) -> bool:
        return True
