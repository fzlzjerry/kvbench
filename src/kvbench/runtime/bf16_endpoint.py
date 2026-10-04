"""Cat-free manual Llama endpoint for the Phase 3 BF16 decode baseline."""

from __future__ import annotations

import importlib
from typing import Any

from kvbench.adapters import KVCacheMethod
from kvbench.adapters.base import method_requires_pre_rope_key
from kvbench.runtime.static_cache import BF16StaticCache, CacheStateError


_TORCH: Any | None = None


class EndpointGeometryError(RuntimeError):
    """Loaded model/cache geometry differs from the frozen endpoint."""


def _torch() -> Any:
    global _TORCH
    if _TORCH is None:
        try:
            _TORCH = importlib.import_module("torch")
        except ModuleNotFoundError as error:
            raise EndpointGeometryError("PyTorch is unavailable") from error
    return _TORCH


def rotate_half_in_place(
    states: Any,
    cos: Any,
    sin: Any,
    first_half_scratch: Any,
) -> None:
    """Apply Llama half-split RoPE using only preallocated scratch storage."""

    if states.shape[-1] % 2 != 0:
        raise EndpointGeometryError("RoPE head dimension must be even")
    half = int(states.shape[-1]) // 2
    first = states[..., :half]
    second = states[..., half:]
    if tuple(first_half_scratch.shape) != tuple(first.shape):
        raise EndpointGeometryError("RoPE scratch shape differs from projected state")
    cos_view = cos.unsqueeze(1)
    sin_view = sin.unsqueeze(1)
    first_half_scratch.copy_(first)
    first.mul_(cos_view[..., :half])
    first.addcmul_(second, sin_view[..., :half], value=-1.0)
    second.mul_(cos_view[..., half:])
    second.addcmul_(first_half_scratch, sin_view[..., half:])


def preserve_pre_rope_key_in_query_scratch(
    query_rope_scratch: Any,
    key_states: Any,
) -> Any:
    """Copy pre-RoPE Key into caller-owned query RoPE scratch storage."""

    if query_rope_scratch.dtype != key_states.dtype:
        raise EndpointGeometryError(
            "query RoPE scratch and pre-RoPE Key dtypes differ"
        )
    if query_rope_scratch.device != key_states.device:
        raise EndpointGeometryError(
            "query RoPE scratch and pre-RoPE Key devices differ"
        )
    if not query_rope_scratch.is_contiguous():
        raise EndpointGeometryError(
            "query RoPE scratch must be contiguous for pre-RoPE Key reuse"
        )
    required = int(key_states.numel())
    if int(query_rope_scratch.numel()) < required:
        raise EndpointGeometryError(
            "query RoPE scratch is too small for native-KV-head pre-RoPE Key"
        )
    preserved = (
        query_rope_scratch.view(-1)
        .narrow(0, 0, required)
        .view(tuple(key_states.shape))
    )
    preserved.copy_(key_states)
    return preserved


class BF16DecodeEndpoint:
    """Embedding-through-LM-head endpoint with static cache and prepared RoPE."""

    def __init__(
        self,
        model: Any,
        cache: BF16StaticCache,
        method: KVCacheMethod,
    ) -> None:
        torch = _torch()
        self.model = model
        self.cache = cache
        self.method = method
        self.method_requires_pre_rope_key = method_requires_pre_rope_key(method)
        config = model.config
        self.num_layers = int(config.num_hidden_layers)
        self.num_query_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.hidden_size = int(config.hidden_size)
        self.head_dim = int(
            getattr(config, "head_dim", self.hidden_size // self.num_query_heads)
        )
        expected = (
            self.num_layers,
            cache.batch_size,
            self.num_kv_heads,
            self.head_dim,
        )
        observed = (
            cache.num_layers,
            cache.batch_size,
            cache.num_kv_heads,
            cache.head_dim,
        )
        if observed != expected:
            raise EndpointGeometryError("model and cache geometry differ")
        if (
            self.num_layers != 32
            or self.num_query_heads != 32
            or self.num_kv_heads != 8
            or self.head_dim != 128
        ):
            raise EndpointGeometryError("model differs from frozen Llama geometry")
        device = cache.device
        batch = cache.batch_size
        half = self.head_dim // 2
        self.query_rope_scratch = torch.empty(
            (self.num_layers, batch, self.num_query_heads, 1, half),
            dtype=torch.bfloat16,
            device=device,
        )
        self.key_rope_scratch = torch.empty(
            (self.num_layers, batch, self.num_kv_heads, 1, half),
            dtype=torch.bfloat16,
            device=device,
        )

    @property
    def workspace_bytes(self) -> int:
        return int(self.query_rope_scratch.numel() * 2) + int(
            self.key_rope_scratch.numel() * 2
        )

    def prepare_position_embeddings(self, position_ids: Any) -> tuple[Any, Any]:
        """Create RoPE cos/sin outside measured execution."""

        torch = _torch()
        marker = torch.empty((), dtype=torch.bfloat16, device=self.cache.device)
        return self.model.model.rotary_emb(marker, position_ids)

    def _attention(
        self,
        attention: Any,
        hidden_states: Any,
        position_embeddings: tuple[Any, Any],
        cache_position: Any,
        *,
        measured_decode: bool,
    ) -> Any:
        torch = _torch()
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query = attention.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key = attention.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value = attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        if measured_decode:
            query_scratch = self.query_rope_scratch[attention.layer_idx]
            key_scratch = self.key_rope_scratch[attention.layer_idx]
        else:
            query_scratch = torch.empty(
                tuple(query[..., : self.head_dim // 2].shape),
                dtype=query.dtype,
                device=query.device,
            )
            key_scratch = torch.empty_like(key[..., : self.head_dim // 2])
        rotate_half_in_place(query, cos, sin, query_scratch)
        key_pre_rope = None
        if self.method_requires_pre_rope_key:
            if int(key.shape[1]) != self.num_kv_heads:
                raise EndpointGeometryError(
                    "pre-RoPE Key does not use native KV-head geometry"
                )
            key_pre_rope = preserve_pre_rope_key_in_query_scratch(
                query_scratch,
                key,
            )
        rotate_half_in_place(key, cos, sin, key_scratch)
        if measured_decode:
            if self.method_requires_pre_rope_key:
                cached_key, cached_value = self.method.append_decode(
                    self.cache,
                    key,
                    value,
                    int(attention.layer_idx),
                    cache_position,
                    key_pre_rope_states=key_pre_rope,
                )
            else:
                cached_key, cached_value = self.method.append_decode(
                    self.cache,
                    key,
                    value,
                    int(attention.layer_idx),
                    cache_position,
                )
        else:
            if self.method_requires_pre_rope_key:
                cached_key, cached_value = self.method.store_prefill(
                    self.cache,
                    key,
                    value,
                    int(attention.layer_idx),
                    cache_position,
                    key_pre_rope_states=key_pre_rope,
                )
            else:
                cached_key, cached_value = self.method.store_prefill(
                    self.cache,
                    key,
                    value,
                    int(attention.layer_idx),
                    cache_position,
                )
        output = self.method.decode_attention(
            attention,
            query,
            cached_key,
            cached_value,
            scaling=float(attention.scaling),
        )
        output = output.reshape(*input_shape, -1).contiguous()
        return attention.o_proj(output)

    def _base_forward(
        self,
        input_ids: Any,
        position_embeddings: tuple[Any, Any],
        cache_position: Any,
        *,
        measured_decode: bool,
    ) -> Any:
        hidden_states = self.model.model.embed_tokens(input_ids)
        for layer in self.model.model.layers:
            residual = hidden_states
            normalized = layer.input_layernorm(hidden_states)
            attention_output = self._attention(
                layer.self_attn,
                normalized,
                position_embeddings,
                cache_position,
                measured_decode=measured_decode,
            )
            hidden_states = residual + attention_output
            residual = hidden_states
            normalized = layer.post_attention_layernorm(hidden_states)
            hidden_states = residual + layer.mlp(normalized)
        return self.model.model.norm(hidden_states)

    def prefill(self, prefix_input_ids: Any) -> Any:
        """Store an exact prefix outside timing without producing vocabulary logits."""

        torch = _torch()
        if prefix_input_ids.ndim != 2:
            raise EndpointGeometryError("prefix input IDs must have shape [B,L]")
        if int(prefix_input_ids.shape[0]) != self.cache.batch_size:
            raise EndpointGeometryError("prefix batch differs from cache batch")
        length = int(prefix_input_ids.shape[1])
        self.cache.prepare_prefill(length)
        cache_position = torch.arange(
            length,
            dtype=torch.long,
            device=self.cache.device,
        )
        position_ids = cache_position.unsqueeze(0)
        positions = self.prepare_position_embeddings(position_ids)
        hidden = self._base_forward(
            prefix_input_ids,
            positions,
            cache_position,
            measured_decode=False,
        )
        self.cache.complete_prefill()
        return hidden

    def decode(
        self,
        input_ids: Any,
        cache_position: Any,
        position_embeddings: tuple[Any, Any],
    ) -> Any:
        """Run exactly one token through embedding, all layers, norm, and LM head."""

        if input_ids.ndim != 2 or tuple(input_ids.shape) != (
            self.cache.batch_size,
            1,
        ):
            raise EndpointGeometryError("decode input IDs must have shape [B,1]")
        if cache_position.ndim != 1 or int(cache_position.shape[0]) != 1:
            raise EndpointGeometryError("decode cache_position must have shape [1]")
        if self.cache.mode not in {"fixed", "growing_step"}:
            raise CacheStateError("decode cache mode is not fixed or growing_step")
        hidden = self._base_forward(
            input_ids,
            position_embeddings,
            cache_position,
            measured_decode=True,
        )
        return self.model.lm_head(hidden)
