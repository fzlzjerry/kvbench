"""Strict interpretation of existing trace/allocation evidence for TurboQuant."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from typing import Any


STORE_FAMILY = "_tq_fused_store_mse"
DECODE_FAMILIES = ("_tq_decode_stage1", "_fwd_kernel_stage2")
FORBIDDEN_KERNEL_TOKENS = (
    "_tq_full_dequant_kv",
    "repeat_interleave",
    "repeat_kv",
)
HOST_SYNC_TOKENS = (
    "cudaDeviceSynchronize",
    "cudaStreamSynchronize",
    "cudaEventSynchronize",
)


def _names(value: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{label} kernel names must be a sequence")
    names = tuple(value)
    if not names or any(type(name) is not str or not name for name in names):
        raise ValueError(f"{label} kernel names must be non-empty strings")
    return names


def _contains(names: Sequence[str], token: str) -> bool:
    return any(token in name for name in names)


def _first(names: Sequence[str], token: str) -> int:
    for index, name in enumerate(names):
        if token in name:
            return index
    return -1


@dataclass(frozen=True, slots=True)
class TurboQuantExecutionPathAudit:
    """One fail-closed summary derived from raw post-warmup evidence."""

    passed: bool
    store_kernel_family: str
    decode_kernel_families: tuple[str, ...]
    store_verified: bool
    append_verified: bool
    decode_verified: bool
    operation_order_verified: bool
    source_identity_verified: bool
    native_gqa_indexing_verified: bool
    full_prefix_dequantization_detected: bool
    gqa_materialization_detected: bool
    query_head_sized_kv_temporary_detected: bool
    host_synchronization_detected: bool
    cache_growth_detected: bool
    backend_fallback_detected: bool
    stable_post_warmup_path: bool
    reasons: tuple[str, ...]
    kernel_sequence_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "store_kernel_family": self.store_kernel_family,
            "decode_kernel_families": list(self.decode_kernel_families),
            "store_verified": self.store_verified,
            "append_verified": self.append_verified,
            "decode_verified": self.decode_verified,
            "operation_order_verified": self.operation_order_verified,
            "source_identity_verified": self.source_identity_verified,
            "native_gqa_indexing_verified": self.native_gqa_indexing_verified,
            "full_prefix_dequantization_detected": (
                self.full_prefix_dequantization_detected
            ),
            "gqa_materialization_detected": self.gqa_materialization_detected,
            "query_head_sized_kv_temporary_detected": (
                self.query_head_sized_kv_temporary_detected
            ),
            "host_synchronization_detected": self.host_synchronization_detected,
            "cache_growth_detected": self.cache_growth_detected,
            "backend_fallback_detected": self.backend_fallback_detected,
            "stable_post_warmup_path": self.stable_post_warmup_path,
            "reasons": list(self.reasons),
            "kernel_sequence_sha256": self.kernel_sequence_sha256,
            "timings_retained": False,
            "performance_claim_eligible": False,
        }


def audit_turboquant_execution_path(
    *,
    store_kernel_names: Sequence[str],
    append_kernel_names: Sequence[str],
    decode_kernel_names: Sequence[str],
    repeated_decode_kernel_names: Sequence[str],
    runtime_event_names: Sequence[str],
    temporary_shapes: Mapping[str, Sequence[int]],
    adapter_hot_path_source: str,
    decode_kernel_source: bytes,
    decode_kernel_sha256: str,
) -> TurboQuantExecutionPathAudit:
    """Apply the frozen path criteria without collecting another profile."""

    store = _names(store_kernel_names, "store")
    append = _names(append_kernel_names, "append")
    decode = _names(decode_kernel_names, "decode")
    repeated = _names(repeated_decode_kernel_names, "repeated decode")
    runtime = tuple(runtime_event_names)
    if any(type(name) is not str or not name for name in runtime):
        raise ValueError("runtime event names must be non-empty strings")
    if type(adapter_hot_path_source) is not str or not adapter_hot_path_source:
        raise ValueError("adapter hot-path source must be non-empty")
    if type(decode_kernel_source) is not bytes or not decode_kernel_source:
        raise ValueError("decode kernel source must be non-empty bytes")
    if (
        type(decode_kernel_sha256) is not str
        or len(decode_kernel_sha256) != 64
        or any(character not in "0123456789abcdef" for character in decode_kernel_sha256)
    ):
        raise ValueError("decode kernel SHA-256 is invalid")

    store_verified = _contains(store, STORE_FAMILY)
    append_verified = _contains(append, STORE_FAMILY)
    stage1 = _first(decode, DECODE_FAMILIES[0])
    stage2 = _first(decode, DECODE_FAMILIES[1])
    decode_verified = stage1 >= 0 and stage2 >= 0
    operation_order_verified = decode_verified and stage1 < stage2
    source_identity_verified = (
        hashlib.sha256(decode_kernel_source).hexdigest()
        == decode_kernel_sha256
    )
    native_gqa_indexing_verified = (
        b"kv_head = hid // KV_GROUP_SIZE" in decode_kernel_source
    )

    all_kernels = (*store, *append, *decode, *repeated)
    full_prefix = _contains(all_kernels, FORBIDDEN_KERNEL_TOKENS[0])
    gqa_materialization = any(
        _contains(all_kernels, token)
        for token in FORBIDDEN_KERNEL_TOKENS[1:]
    )
    query_head_kv = False
    for role, shape_value in temporary_shapes.items():
        if type(role) is not str or not role:
            raise ValueError("temporary role must be a non-empty string")
        shape = tuple(shape_value)
        if any(type(item) is not int or item <= 0 for item in shape):
            raise ValueError("temporary shapes must contain positive integers")
        lowered = role.casefold()
        is_kv_role = "key" in lowered or "value" in lowered or lowered.startswith("kv")
        if is_kv_role and role != "packed_kv_cache" and 32 in shape[:-1]:
            query_head_kv = True

    hot_path_tokens = adapter_hot_path_source.casefold()
    host_sync = any(token in name for name in runtime for token in HOST_SYNC_TOKENS)
    host_sync = host_sync or any(
        token in hot_path_tokens
        for token in (".cpu(", ".item(", ".tolist(", "synchronize(")
    )
    cache_growth = "torch.cat(" in hot_path_tokens or ".cat(" in hot_path_tokens
    stable = decode == repeated
    fallback = not (store_verified and append_verified and decode_verified)

    reasons: list[str] = []
    checks = {
        "store_kernel_absent": store_verified,
        "append_kernel_absent": append_verified,
        "decode_kernel_absent": decode_verified,
        "decode_kernel_order_invalid": operation_order_verified,
        "decode_source_identity_mismatch": source_identity_verified,
        "native_gqa_indexing_unverified": native_gqa_indexing_verified,
        "full_prefix_dequantization_detected": not full_prefix,
        "gqa_materialization_detected": not gqa_materialization,
        "query_head_sized_kv_temporary_detected": not query_head_kv,
        "host_synchronization_detected": not host_sync,
        "cache_growth_detected": not cache_growth,
        "backend_fallback_detected": not fallback,
        "post_warmup_path_unstable": stable,
    }
    reasons.extend(reason for reason, passed in checks.items() if not passed)
    sequence_bytes = "\n".join(all_kernels).encode("utf-8")
    return TurboQuantExecutionPathAudit(
        passed=not reasons,
        store_kernel_family=STORE_FAMILY,
        decode_kernel_families=DECODE_FAMILIES,
        store_verified=store_verified,
        append_verified=append_verified,
        decode_verified=decode_verified,
        operation_order_verified=operation_order_verified,
        source_identity_verified=source_identity_verified,
        native_gqa_indexing_verified=native_gqa_indexing_verified,
        full_prefix_dequantization_detected=full_prefix,
        gqa_materialization_detected=gqa_materialization,
        query_head_sized_kv_temporary_detected=query_head_kv,
        host_synchronization_detected=host_sync,
        cache_growth_detected=cache_growth,
        backend_fallback_detected=fallback,
        stable_post_warmup_path=stable,
        reasons=tuple(reasons),
        kernel_sequence_sha256=hashlib.sha256(sequence_bytes).hexdigest(),
    )
