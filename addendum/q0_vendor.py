"""Verbatim copy of five helpers from scripts/q0_cache_sensitive_correctness.py (main @ 121c24d).

The Full Scan execution repository (ec534d99) predates the Q0 script, so the
addendum correctness worker carries these helpers itself.  Copied functions:
family, _build_method, _allocate_state, _prefill, _finish_step.  SHA-256 of the copied text: 3bb25763bdbad85cab11dd8a80224b03b1a293fb34e5244c21b751af333cf6fe.
"""

from __future__ import annotations

from typing import Any


class Q0Error(RuntimeError):
    pass


def family(configuration: str) -> str:
    if configuration == "bf16":
        return "bf16"
    if configuration.startswith("tq_"):
        return "turboquant"
    if configuration.startswith("kvq"):
        return "kvquant"
    if configuration.startswith("k"):
        return "kivi"
    raise Q0Error(f"unknown Q0 configuration: {configuration}")


def _build_method(loaded: Any, configuration: str) -> tuple[Any, int]:
    from kvbench.adapters import build_method_adapter, declared_bf16_runtime_context
    from kvbench.runtime.kivi_session import kivi_runtime_context, load_frozen_kivi_method_config
    from kvbench.runtime.kvquant_session import kvquant_runtime_context, load_frozen_kvquant_method_config
    from kvbench.runtime.turboquant_session import turboquant_runtime_context
    from scripts import phase12_unified_admission as phase12

    selected = family(configuration)
    if selected == "bf16":
        method = build_method_adapter("bf16", declared_bf16_runtime_context(loaded.model))
    elif selected == "turboquant":
        native = {
            "tq_4bit_nc": "turboquant_4bit_nc",
            "tq_k3v4_nc": "turboquant_k3v4_nc",
            "tq_3bit_nc": "turboquant_3bit_nc",
        }[configuration]
        method = build_method_adapter(native, turboquant_runtime_context())
    elif selected == "kivi":
        method = build_method_adapter(
            load_frozen_kivi_method_config(),
            kivi_runtime_context(),
            variant_id=configuration,
        )
        method.prepare_runtime()
    else:
        method = build_method_adapter(
            load_frozen_kvquant_method_config(),
            kvquant_runtime_context(configuration),
            variant_id=configuration,
        )
        method.prepare_runtime()
    phase12._validate_live_method_identity(method, configuration)
    return method, selected


def _allocate_state(
    loaded: Any,
    configuration: str,
    *,
    batch: int,
    capacity: int,
    mode: str,
    prefix_length: int,
    output_steps: int,
) -> tuple[Any, Any, Any, tuple[Any, ...], tuple[Any, ...]]:
    import torch
    from kvbench.runtime.bf16_endpoint import BF16DecodeEndpoint

    method, selected = _build_method(loaded, configuration)
    workspace = 32 * batch * (32 + 8) * 64 * 2 if selected == "bf16" else 0
    cache = method.allocate(
        batch_size=batch,
        capacity=capacity,
        device=torch.device("cuda:0"),
        workspace_bytes=workspace,
    )
    if selected == "kvquant":
        method.initialize_cache_untimed(cache)
    else:
        cache.initialize_deterministic()
    endpoint = BF16DecodeEndpoint(loaded.model, cache, method)
    positions = tuple(
        torch.tensor([prefix_length + index], dtype=torch.long, device=cache.device)
        for index in range(output_steps)
    )
    if selected == "kvquant":
        if mode == "fixed":
            cache.bind_fixed_position_tensor_untimed(positions[0], logical_position=prefix_length)
        else:
            cache.bind_growing_position_tensors_untimed(positions, starting_position=prefix_length)
    rope = tuple(endpoint.prepare_position_embeddings(position.unsqueeze(0)) for position in positions)
    return method, cache, endpoint, positions, rope


def _prefill(endpoint: Any, prefix: Any, selected: str) -> None:
    if selected == "kvquant":
        from scripts.phase13_pilot import _chunked_kvquant_prefix_store

        with _chunked_kvquant_prefix_store(endpoint):
            endpoint.prefill(prefix)
    else:
        endpoint.prefill(prefix)


def _finish_step(cache: Any, selected: str) -> None:
    # KIVI and KVQuant commit growing state inside their final-layer decode.
    # BF16 and TurboQuant retain caller-owned commit semantics.
    if selected in {"bf16", "turboquant"}:
        cache.finish_growing_step()
