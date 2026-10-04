"""Thin adapter over the validated Phase 3 BF16 cache and Flash endpoint."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from kvbench.adapters.base import MethodRuntimeContext
from kvbench.runtime.backend import flash_attention_forward
from kvbench.runtime.static_cache import BF16StaticCache
from kvbench.schema import canonical_json_bytes, sha256_hex
from kvbench.schema.base import require_sha256


BF16_ADAPTER_VERSION = "kvbench-bf16-method-adapter-1.0.0"
BF16_ADAPTER_FINGERPRINT_SCHEMA_VERSION = (
    "kvbench-bf16-method-adapter-config-1.0.0"
)


def declared_bf16_runtime_context(model: Any) -> MethodRuntimeContext:
    """Build the frozen identity context used by standalone validation paths."""

    from kvbench.runtime.backend import BACKEND_IDENTITY
    from kvbench.runtime.model_loader import MODEL_ID, MODEL_REVISION

    config = model.config
    num_query_heads = int(config.num_attention_heads)
    hidden_size = int(config.hidden_size)
    return MethodRuntimeContext(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        backend_id=str(BACKEND_IDENTITY["backend_id"]),
        backend_fingerprint=sha256_hex(
            canonical_json_bytes(BACKEND_IDENTITY)
        ),
        num_layers=int(config.num_hidden_layers),
        num_query_heads=num_query_heads,
        num_kv_heads=int(config.num_key_value_heads),
        head_dim=int(
            getattr(config, "head_dim", hidden_size // num_query_heads)
        ),
    )


class BF16MethodAdapter:
    """Delegate BF16 storage and attention to the validated Phase 3 code."""

    name = "bf16"
    adapter_version = BF16_ADAPTER_VERSION

    def __init__(self, runtime_context: MethodRuntimeContext) -> None:
        if type(runtime_context) is not MethodRuntimeContext:
            raise TypeError("BF16 adapter requires MethodRuntimeContext")
        if (
            runtime_context.num_layers,
            runtime_context.num_query_heads,
            runtime_context.num_kv_heads,
            runtime_context.head_dim,
        ) != (32, 32, 8, 128):
            raise ValueError("BF16 adapter requires frozen Llama GQA geometry")
        self.runtime_context = runtime_context

    def allocate(
        self,
        *,
        batch_size: int,
        capacity: int,
        device: Any,
        workspace_bytes: int = 0,
    ) -> BF16StaticCache:
        return BF16StaticCache(
            num_layers=self.runtime_context.num_layers,
            batch_size=batch_size,
            num_kv_heads=self.runtime_context.num_kv_heads,
            capacity=capacity,
            head_dim=self.runtime_context.head_dim,
            device=device,
            workspace_bytes=workspace_bytes,
        )

    @staticmethod
    def _require_cache(cache_state: Any) -> BF16StaticCache:
        if type(cache_state) is not BF16StaticCache:
            raise TypeError("BF16 adapter cache state must be BF16StaticCache")
        return cache_state

    def store_prefill(
        self,
        cache_state: Any,
        key_states: Any,
        value_states: Any,
        layer_idx: int,
        cache_position: Any,
    ) -> tuple[Any, Any]:
        cache = self._require_cache(cache_state)
        return cache.update(
            key_states,
            value_states,
            layer_idx,
            {"cache_position": cache_position},
        )

    def append_decode(
        self,
        cache_state: Any,
        key_states: Any,
        value_states: Any,
        layer_idx: int,
        cache_position: Any,
    ) -> tuple[Any, Any]:
        cache = self._require_cache(cache_state)
        return cache.update(
            key_states,
            value_states,
            layer_idx,
            {"cache_position": cache_position},
        )

    def decode_attention(
        self,
        attention: Any,
        query_states: Any,
        key_states: Any,
        value_states: Any,
        *,
        scaling: float,
    ) -> Any:
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
        accounting = self._require_cache(cache_state).accounting()
        return {
            "data_bytes": accounting.predicted_tensor_bytes,
            "workspace_bytes": accounting.workspace_bytes,
            "padding_bytes": accounting.padding_bytes,
            "scale_bytes": 0,
            "zero_point_bytes": 0,
            "metadata_bytes": 0,
        }

    def logical_bf16_bytes(self, cache_state: Any) -> int:
        return self._require_cache(cache_state).accounting().predicted_tensor_bytes

    def config_fingerprint(self, cache_layout_fingerprint: str) -> str:
        require_sha256(
            cache_layout_fingerprint,
            field_name="cache_layout_fingerprint",
        )
        context = self.runtime_context
        payload = {
            "schema_version": BF16_ADAPTER_FINGERPRINT_SCHEMA_VERSION,
            "adapter_version": self.adapter_version,
            "method_name": self.name,
            "cache_dtype": "bfloat16",
            "cache_layout_fingerprint": cache_layout_fingerprint,
            "model_id": context.model_id,
            "model_revision": context.model_revision,
            "backend_id": context.backend_id,
            "backend_fingerprint": context.backend_fingerprint,
            "num_layers": context.num_layers,
            "num_query_heads": context.num_query_heads,
            "num_kv_heads": context.num_kv_heads,
            "head_dim": context.head_dim,
            "supports_cuda_graph": self.supports_cuda_graph(),
            "implementation_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        }
        return sha256_hex(canonical_json_bytes(payload))

    def supports_cuda_graph(self) -> bool:
        return True
