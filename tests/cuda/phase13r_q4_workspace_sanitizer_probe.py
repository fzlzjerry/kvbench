#!/usr/bin/env python3
"""Focused B=8/L=16384 q4 workspace probe for Compute Sanitizer."""

from __future__ import annotations

import json
import math
import os
import sys
from types import SimpleNamespace
from typing import Any

import torch

from kvbench.adapters.kvquant import KVQuantMethodAdapter
from kvbench.runtime.cuda_graph import capture_fixed_graph
from kvbench.runtime.kvquant_fixture import (
    load_fixture_tensor_file_untimed,
    load_kvquant_fixture,
)
from tests.cuda.phase11_kvquant_sanitizer_probe import (
    _require_exact_authority,
    _require_exact_environment,
    _reset_cuda_for_memcheck,
    _runtime_context,
)


BATCH = 8
HISTORICAL_CONTEXT = 16_384
CAPACITY = HISTORICAL_CONTEXT + 1
SOURCE_PREFIX = 17
SINK_TOKENS = 5
EXPECTED_SHAPE = (8, 32, 128, 128)


def _run(device: torch.device) -> dict[str, Any]:
    fixture = load_kvquant_fixture("kvq4", "key_cap_value_fixed12")
    inputs = load_fixture_tensor_file_untimed(fixture, "inputs.safetensors")
    sink = load_fixture_tensor_file_untimed(fixture, "sink.safetensors")
    decode = load_fixture_tensor_file_untimed(
        fixture,
        "decode_output.safetensors",
    )
    key_pre_rope = inputs["key_pre_rope"].repeat(BATCH, 1, 1, 1).to(device)
    value = inputs["value_after_v_proj"].repeat(BATCH, 1, 1, 1).to(device)
    key_attention = key_pre_rope.clone()
    key_attention[:, :, :SINK_TOKENS].copy_(
        sink["sink_key_attention_fp16"]
        .transpose(2, 3)
        .repeat(BATCH, 1, 1, 1)
        .to(device=device, dtype=torch.bfloat16)
    )
    positions = inputs["position_ids"].reshape(-1).to(device)
    query = decode["query_attention_ready"].repeat(BATCH, 1, 1, 1).to(device)
    append_position = torch.tensor(
        [HISTORICAL_CONTEXT],
        dtype=torch.int64,
        device=device,
    )

    method = KVQuantMethodAdapter(_runtime_context(), "kvq4")
    method.prepare_runtime()
    cache = method.allocate(
        batch_size=BATCH,
        capacity=CAPACITY,
        device=device,
    )
    method.initialize_cache_untimed(cache)
    pointers_before = cache.pointers()
    geometry = cache.q4_value_decode_workspace_geometry()
    if (
        tuple(cache.q4_value_decode_workspace.shape) != EXPECTED_SHAPE
        or geometry["tile_capacity"] != 128
        or geometry["quantized_value_capacity"] != 16_380
        or geometry["workspace_bytes"] != 16_777_216
    ):
        raise RuntimeError("Phase 13R q4 workspace geometry differs")

    cache.prepare_prefill(SOURCE_PREFIX)
    method.store_prefill(
        cache,
        key_attention[:, :, :SOURCE_PREFIX],
        value[:, :, :SOURCE_PREFIX],
        0,
        positions[:SOURCE_PREFIX],
        key_pre_rope_states=key_pre_rope[:, :, :SOURCE_PREFIX],
    )
    cache.complete_prefill()
    cache.reset_active_length(HISTORICAL_CONTEXT)
    cache.bind_fixed_position_tensor_untimed(
        append_position,
        logical_position=HISTORICAL_CONTEXT,
    )
    cache.prepare_fixed(HISTORICAL_CONTEXT)

    def operation() -> Any:
        handles = method.append_decode(
            cache,
            key_attention[:, :, SOURCE_PREFIX : SOURCE_PREFIX + 1],
            value[:, :, SOURCE_PREFIX : SOURCE_PREFIX + 1],
            0,
            append_position,
            key_pre_rope_states=key_pre_rope[
                :, :, SOURCE_PREFIX : SOURCE_PREFIX + 1
            ],
        )
        return method.decode_attention(
            SimpleNamespace(layer_idx=0),
            query,
            handles[0],
            handles[1],
            scaling=1.0 / math.sqrt(128),
        )

    first = operation()
    graph = capture_fixed_graph(operation, warmup_steps=1, device=device)
    try:
        replay_one = graph.replay()
        torch.cuda.synchronize(device=device)
        control = replay_one.detach().cpu().clone()
        replay_two = graph.replay()
        torch.cuda.synchronize(device=device)
        if (
            not torch.equal(control, replay_two.detach().cpu())
            or not bool(torch.isfinite(first).all())
            or not bool(torch.isfinite(replay_two).all())
            or graph.to_dict()["fallback"]
            or cache.pointers() != pointers_before
        ):
            raise RuntimeError("Phase 13R q4 graph or pointer result differs")
        result = {
            "status": "PASS",
            "configuration": "kvq4",
            "batch_size": BATCH,
            "historical_context": HISTORICAL_CONTEXT,
            "workspace_shape": list(EXPECTED_SHAPE),
            "workspace_bytes": 16_777_216,
            "tile_capacity": 128,
            "store_append_decode": "PASS",
            "graph_capture_replay": "PASS",
            "pointer_stability": "PASS",
            "finite_output": True,
            "timing_collected": False,
        }
    finally:
        graph.graph.reset()
    del (
        append_position,
        cache,
        control,
        decode,
        first,
        fixture,
        graph,
        inputs,
        key_attention,
        key_pre_rope,
        method,
        positions,
        query,
        replay_one,
        replay_two,
        sink,
        value,
    )
    return result


def main() -> int:
    failures: list[dict[str, str]] = []
    result: dict[str, Any] | None = None
    environment: dict[str, Any] | None = None
    try:
        environment = _require_exact_environment(
            os.environ["KVBENCH_AUTHORIZED_IMAGE_DIGEST"]
        )
        _require_exact_authority()
        result = _run(torch.device("cuda:0"))
    except Exception as error:
        failures.append({"type": type(error).__name__, "message": str(error)})
        error.__traceback__ = None
        del error
    finally:
        if environment is not None:
            try:
                _reset_cuda_for_memcheck()
            except Exception as error:
                failures.append(
                    {"type": type(error).__name__, "message": str(error)}
                )
                error.__traceback__ = None
                del error
    payload = {
        "status": "FAIL" if failures else "PASS",
        "result": result,
        "failures": failures,
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 2 if failures else 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
