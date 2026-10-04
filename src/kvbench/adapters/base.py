"""Small method boundary shared by decode runners and admission harnesses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from kvbench.schema.base import require_sha256


def method_requires_pre_rope_key(method: Any) -> bool:
    """Return one adapter's static pre-RoPE Key capability.

    The capability is deliberately optional so existing adapters keep their
    original boundary unchanged.  An adapter that needs the additional Key
    representation must declare ``requires_pre_rope_key = True`` on its class.
    """

    capability = getattr(type(method), "requires_pre_rope_key", False)
    if type(capability) is not bool:
        raise TypeError("requires_pre_rope_key must be a class-level bool")
    return capability


@dataclass(frozen=True, slots=True)
class MethodRuntimeContext:
    """Identity and frozen geometry needed by a cache-method adapter."""

    model_id: str
    model_revision: str
    backend_id: str
    backend_fingerprint: str
    num_layers: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int

    def __post_init__(self) -> None:
        if not self.model_id or not self.model_revision or not self.backend_id:
            raise ValueError("method runtime identity fields must be non-empty")
        require_sha256(
            self.backend_fingerprint,
            field_name="backend_fingerprint",
        )
        if min(
            self.num_layers,
            self.num_query_heads,
            self.num_kv_heads,
            self.head_dim,
        ) <= 0:
            raise ValueError("method runtime geometry must be positive")
        if self.num_query_heads % self.num_kv_heads != 0:
            raise ValueError("query heads must be divisible by KV heads")


@runtime_checkable
class KVCacheMethod(Protocol):
    """Only the operations runners and common admission checks require."""

    name: str
    adapter_version: str

    def allocate(
        self,
        *,
        batch_size: int,
        capacity: int,
        device: Any,
        workspace_bytes: int = 0,
    ) -> Any:
        """Allocate method-owned cache state outside measured execution."""

    def store_prefill(
        self,
        cache_state: Any,
        key_states: Any,
        value_states: Any,
        layer_idx: int,
        cache_position: Any,
        *,
        key_pre_rope_states: Any | None = None,
    ) -> tuple[Any, Any]:
        """Store prefill K/V and return the attended cache views.

        ``key_states`` remains the attention-ready, post-RoPE Key.  The
        optional pre-RoPE Key is provided only to adapters that statically
        declare ``requires_pre_rope_key = True``.
        """

    def append_decode(
        self,
        cache_state: Any,
        key_states: Any,
        value_states: Any,
        layer_idx: int,
        cache_position: Any,
        *,
        key_pre_rope_states: Any | None = None,
    ) -> tuple[Any, Any]:
        """Append one decode K/V slot and return attended cache views.

        ``key_states`` remains the attention-ready, post-RoPE Key.  The
        optional pre-RoPE Key follows the same capability rule as prefill.
        """

    def decode_attention(
        self,
        attention: Any,
        query_states: Any,
        key_states: Any,
        value_states: Any,
        *,
        scaling: float,
    ) -> Any:
        """Execute the method-owned attention work."""

    def allocated_bytes(self, cache_state: Any) -> int:
        """Return actual method-owned cache and workspace storage bytes."""

    def byte_breakdown(self, cache_state: Any) -> Mapping[str, int]:
        """Return a deterministic complete owned-storage breakdown."""

    def logical_bf16_bytes(self, cache_state: Any) -> int:
        """Return logical BF16 K/V bytes for the same cache geometry."""

    def config_fingerprint(self, cache_layout_fingerprint: str) -> str:
        """Bind method, runtime identity, and one cache layout."""

    def supports_cuda_graph(self) -> bool:
        """Declare whether the adapter supports capture/replay."""
