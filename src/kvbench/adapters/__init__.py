"""Stable Phase 4 cache-method adapter surface."""

from kvbench.adapters.base import KVCacheMethod, MethodRuntimeContext
from kvbench.adapters.bf16 import (
    BF16_ADAPTER_FINGERPRINT_SCHEMA_VERSION,
    BF16_ADAPTER_VERSION,
    BF16MethodAdapter,
    declared_bf16_runtime_context,
)
from kvbench.adapters.factory import build_method_adapter
from kvbench.adapters.kivi import (
    KIVI_ADAPTER_VERSION,
    KIVIMethodAdapter,
)
from kvbench.adapters.kvquant import (
    KVQUANT_ADAPTER_VERSION,
    KVQUANT_METHOD_IDENTIFIER,
    KVQuantMethodAdapter,
)
from kvbench.adapters.turboquant import (
    TURBOQUANT_ADAPTER_VERSION,
    TurboQuantMethodAdapter,
)

__all__ = [
    "BF16_ADAPTER_FINGERPRINT_SCHEMA_VERSION",
    "BF16_ADAPTER_VERSION",
    "BF16MethodAdapter",
    "KVCacheMethod",
    "KIVI_ADAPTER_VERSION",
    "KIVIMethodAdapter",
    "KVQUANT_ADAPTER_VERSION",
    "KVQUANT_METHOD_IDENTIFIER",
    "KVQuantMethodAdapter",
    "MethodRuntimeContext",
    "TURBOQUANT_ADAPTER_VERSION",
    "TurboQuantMethodAdapter",
    "build_method_adapter",
    "declared_bf16_runtime_context",
]
