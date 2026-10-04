#!/usr/bin/env python3
"""Minimal exact-container KVQuant probe for Compute Sanitizer."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any

import torch

from kvbench.adapters.base import MethodRuntimeContext
from kvbench.adapters.kvquant import (
    KVQUANT_AGGREGATE_PATCH_SHA256,
    KVQUANT_AUTHORIZED_CONTAINER_DIGEST,
    KVQUANT_CORRECTED_COMMIT,
    KVQUANT_CORRECTED_TREE,
    KVQUANT_EXTENSION_SHA256,
    KVQuantMethodAdapter,
)
from kvbench.runtime.cuda_graph import capture_fixed_graph
from kvbench.runtime.kvquant_fixture import (
    KVQUANT_FIXTURE_ID,
    KVQUANT_FIXTURE_ROOT_SHA256,
    compare_decode_output_untimed,
    compare_exact_fixture_tensor_untimed,
    load_fixture_tensor_file_untimed,
    load_kvquant_fixture,
)
from kvbench.runtime.turboquant_admission import (
    PHASE6_CONTAINER_ENVIRONMENT_VALUE,
    PHASE6_CONTAINER_ENVIRONMENT_VARIABLE,
    PHASE6_IMAGE_ENVIRONMENT_VARIABLE,
    require_authorized_cuda_environment,
)


LAYER = 0
PREFIX_LENGTH = 17
CAPACITY = 18
SINK_TOKENS = 5
SCALING = 1.0 / math.sqrt(128)
MODE_CASES = {
    "kvq4-cap": ("kvq4", "key_cap_value_fixed12", False),
    "kvq3-distinct": ("kvq3", "key_few_value_fixed12", False),
    "kvq2": ("kvq2", "key_zero_value_fixed12", False),
    "sink-gqa-fixed": ("kvq4", "key_zero_value_fixed12", False),
    "graph-replay": ("kvq4", "key_cap_value_fixed12", True),
}
_AUTHORITY = {
    "aggregate_patch_sha256": (
        "7b9d3cc6773e8ef37697601c885f2c5ec581dffd57cf59424d03e68f147bd55a"
    ),
    "corrected_commit": "34b0bdfa83082e1f30387d9ac5cca369006e089c",
    "corrected_tree": "1f85af65fe03061583ffe8bd91e47d7ecffdd312",
    "extension_sha256": (
        "b3c33badb8e55b19d6b2ce535182e964ce51e5102d8413b29701dd3d817ad73d"
    ),
    "fixture_id": "kvqref-2e0a0e9022c50cbc6fb497d88cae973e",
    "fixture_root_sha256": (
        "c28682d58706b58812dc1db69ba5eb4982339ba13f39bf67f751794cdaabfdec"
    ),
}


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-config-digest",
        required=True,
        help="Decision 0016 Measurement Container config digest",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=(*MODE_CASES, "all"),
        help="one distinct non-performance kernel/path case",
    )
    return parser.parse_args(argv)


def _require_exact_environment(declared_digest: str) -> dict[str, Any]:
    extension = os.environ.get("KVBENCH_KVQUANT_EXTENSION")
    if (
        not Path("/.dockerenv").is_file()
        or os.environ.get(PHASE6_IMAGE_ENVIRONMENT_VARIABLE)
        != KVQUANT_AUTHORIZED_CONTAINER_DIGEST
        or os.environ.get(PHASE6_CONTAINER_ENVIRONMENT_VARIABLE)
        != PHASE6_CONTAINER_ENVIRONMENT_VALUE
        or extension is None
        or not Path(extension).is_file()
    ):
        raise RuntimeError(
            "Phase 11 KVQuant sanitizer execution is forbidden outside the "
            "exact Measurement Container with the bound extension"
        )
    return require_authorized_cuda_environment(declared_digest)


def _require_exact_authority() -> None:
    observed = {
        "aggregate_patch_sha256": KVQUANT_AGGREGATE_PATCH_SHA256,
        "corrected_commit": KVQUANT_CORRECTED_COMMIT,
        "corrected_tree": KVQUANT_CORRECTED_TREE,
        "extension_sha256": KVQUANT_EXTENSION_SHA256,
        "fixture_id": KVQUANT_FIXTURE_ID,
        "fixture_root_sha256": KVQUANT_FIXTURE_ROOT_SHA256,
    }
    if observed != _AUTHORITY:
        raise RuntimeError("Phase 11 KVQuant sanitizer authority differs")


def _runtime_context() -> MethodRuntimeContext:
    return MethodRuntimeContext(
        model_id="phase11-kvquant-sanitizer",
        model_revision="0e9e39f249a16976918f6564b8830bc894c89659",
        backend_id="kvquant-gqa-longctx-deterministic-q23-v4",
        backend_fingerprint=hashlib.sha256(
            b"phase11-kvquant-sanitizer"
        ).hexdigest(),
        num_layers=32,
        num_query_heads=32,
        num_kv_heads=8,
        head_dim=128,
    )


def _assert_exact(
    fixture: Any,
    file_name: str,
    tensor_name: str,
    observed: Any,
) -> None:
    comparison = compare_exact_fixture_tensor_untimed(
        fixture,
        file_name,
        tensor_name,
        observed,
    )
    if not comparison.passed:
        raise RuntimeError(
            f"sanitizer fixture differs: {file_name}:{tensor_name}"
        )


def _release_tracked_cuda_storages() -> None:
    """Irreversibly release tensors owned by this isolated probe process."""

    storages: dict[int, object] = {}
    tracked_objects = gc.get_objects()
    for value in tracked_objects:
        if type(value) is not torch.Tensor or not value.is_cuda:
            continue
        try:
            storage = value.untyped_storage()
        except RuntimeError:
            continue
        if int(storage.nbytes()) != 0:
            storages.setdefault(int(storage._cdata), storage)
    del tracked_objects
    for storage in storages.values():
        storage.resize_(0)
    storages.clear()
    gc.collect()


def _reset_cuda_for_memcheck(*, max_passes: int = 8) -> None:
    """Drain allocator state and destroy this isolated CUDA context."""

    torch.cuda.synchronize()
    blas_handle = int(torch.cuda.current_blas_handle())
    _release_tracked_cuda_storages()
    last_state = (-1, -1)
    for _ in range(max_passes):
        gc.collect()
        torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch._C._host_emptyCache()
        torch.cuda.synchronize()
        last_state = (
            int(torch.cuda.memory_allocated()),
            int(torch.cuda.memory_reserved()),
        )
        if last_state == (0, 0):
            break
    else:
        raise RuntimeError(
            "CUDA allocator did not drain before reset: "
            f"allocated={last_state[0]} reserved={last_state[1]}"
        )

    cublas = ctypes.CDLL("libcublas.so.13")
    destroy = cublas.cublasDestroy_v2
    destroy.argtypes = [ctypes.c_void_p]
    destroy.restype = ctypes.c_int
    status = int(destroy(ctypes.c_void_p(blas_handle)))
    if status != 0:
        raise RuntimeError(f"cublasDestroy_v2 failed with status {status}")

    cudart = ctypes.CDLL("libcudart.so.13")
    reset = cudart.cudaDeviceReset
    reset.argtypes = []
    reset.restype = ctypes.c_int
    status = int(reset())
    if status != 0:
        raise RuntimeError(f"cudaDeviceReset failed with status {status}")


def _run_case(
    mode: str,
    *,
    device: torch.device,
) -> dict[str, Any]:
    family, case_name, graph_required = MODE_CASES[mode]
    fixture = load_kvquant_fixture(family, case_name)
    inputs = load_fixture_tensor_file_untimed(
        fixture,
        "inputs.safetensors",
    )
    sink = load_fixture_tensor_file_untimed(
        fixture,
        "sink.safetensors",
    )
    decode = load_fixture_tensor_file_untimed(
        fixture,
        "decode_output.safetensors",
    )
    key_pre_rope = inputs["key_pre_rope"].to(device=device)
    value = inputs["value_after_v_proj"].to(device=device)
    key_attention = key_pre_rope.clone()
    key_attention[:, :, :SINK_TOKENS, :].copy_(
        sink["sink_key_attention_fp16"]
        .transpose(2, 3)
        .to(device=device, dtype=torch.bfloat16)
    )
    positions = inputs["position_ids"].reshape(-1).to(device=device)
    query = decode["query_attention_ready"].to(device=device)

    method = KVQuantMethodAdapter(_runtime_context(), family)
    method.prepare_runtime()
    cache = method.allocate(
        batch_size=1,
        capacity=CAPACITY,
        device=device,
    )
    method.initialize_cache_untimed(cache)
    pointers_before = cache.pointers()
    if family == "kvq4":
        if (
            cache.q4_value_decode_workspace is None
            or tuple(cache.q4_value_decode_workspace.shape)
            != cache.q4_value_decode_workspace_shape
        ):
            raise RuntimeError(
                "KVQuant q4 deterministic workspace differs"
            )
    elif cache.q4_value_decode_workspace is not None:
        raise RuntimeError("non-q4 cache owns a q4 decode workspace")
    elif cache.q23_value_decode_workspace is None:
        raise RuntimeError("q3/q2 deterministic workspace alias differs")
    cache.prepare_prefill(PREFIX_LENGTH)
    method.store_prefill(
        cache,
        key_attention[:, :, :PREFIX_LENGTH, :],
        value[:, :, :PREFIX_LENGTH, :],
        LAYER,
        positions[:PREFIX_LENGTH],
        key_pre_rope_states=key_pre_rope[:, :, :PREFIX_LENGTH, :],
    )
    cache.complete_prefill()
    quantized_prefix = PREFIX_LENGTH - SINK_TOKENS
    history = (
        cache.packed_key_cache[
            LAYER, 0, :, :, :quantized_prefix
        ].detach().cpu().clone(),
        cache.packed_value_cache[
            LAYER, 0, :, :, :quantized_prefix
        ].detach().cpu().clone(),
    )
    append_position = positions[PREFIX_LENGTH:CAPACITY]
    cache.bind_fixed_position_tensor_untimed(
        append_position,
        logical_position=PREFIX_LENGTH,
    )
    cache.prepare_fixed(PREFIX_LENGTH)

    def operation() -> Any:
        handles = method.append_decode(
            cache,
            key_attention[:, :, PREFIX_LENGTH:CAPACITY, :],
            value[:, :, PREFIX_LENGTH:CAPACITY, :],
            LAYER,
            append_position,
            key_pre_rope_states=key_pre_rope[
                :, :, PREFIX_LENGTH:CAPACITY, :
            ],
        )
        return method.decode_attention(
            SimpleNamespace(layer_idx=LAYER),
            query,
            handles[0],
            handles[1],
            scaling=SCALING,
        )

    for _ in range(2):
        output = operation()
    graph_result: dict[str, Any] = {
        "captured": False,
        "replays": 0,
    }
    if graph_required:
        graph = capture_fixed_graph(
            operation,
            warmup_steps=1,
            device=device,
        )
        try:
            first = graph.replay()
            torch.cuda.synchronize(device=device)
            first_copy = first.detach().cpu().clone()
            second = graph.replay()
            torch.cuda.synchronize(device=device)
            if not torch.equal(first_copy, second.detach().cpu()):
                raise RuntimeError("KVQuant graph replays differ")
            output = second
            graph_result = {
                "captured": True,
                "replays": 2,
                "fallback": graph.to_dict()["fallback"],
            }
        finally:
            graph.graph.reset()
        del first, first_copy, graph, second
    torch.cuda.synchronize(device=device)

    for tensor_name, observed in (
        ("k_dense_allocated", cache.packed_key_cache[LAYER, 0]),
        ("v_dense_allocated", cache.packed_value_cache[LAYER, 0]),
        ("v_lookup_allocated", cache.value_lookup_cache[LAYER, 0]),
        ("k_sparse_values_allocated", cache.key_sparse_values[LAYER, 0]),
        ("k_sparse_indices_allocated", cache.key_sparse_indices[LAYER, 0]),
        ("v_sparse_values_allocated", cache.value_sparse_values[LAYER, 0]),
        ("v_sparse_indices_allocated", cache.value_sparse_indices[LAYER, 0]),
        ("sink_k", cache.sink_key[LAYER]),
        ("sink_v", cache.sink_value[LAYER]),
    ):
        _assert_exact(
            fixture,
            "append_state.safetensors",
            tensor_name,
            observed,
        )
    comparison = compare_decode_output_untimed(fixture, output)
    if not comparison.passed or not bool(torch.isfinite(output).all()):
        raise RuntimeError("KVQuant sanitizer decode differs or is nonfinite")
    quantized_total = CAPACITY - SINK_TOKENS
    key_counts = (
        cache.key_active_counts[LAYER, 0, :quantized_total]
        .unique()
        .detach()
        .cpu()
        .tolist()
    )
    value_counts = (
        cache.value_active_counts[LAYER, 0, :quantized_total]
        .unique()
        .detach()
        .cpu()
        .tolist()
    )
    if key_counts != [fixture.key_active_count] or value_counts != [12]:
        raise RuntimeError("KVQuant sanitizer sparse occupancy differs")
    if cache.pointers() != pointers_before:
        raise RuntimeError("KVQuant caller-owned pointers changed")
    if (
        not torch.equal(
            cache.packed_key_cache[
                LAYER, 0, :, :, :quantized_prefix
            ].detach().cpu(),
            history[0],
        )
        or not torch.equal(
            cache.packed_value_cache[
                LAYER, 0, :, :, :quantized_prefix
            ].detach().cpu(),
            history[1],
        )
    ):
        raise RuntimeError("KVQuant fixed overwrite changed historical cache")
    geometry = cache.gqa_geometry()
    if (
        geometry["num_query_heads"] != 32
        or geometry["num_kv_heads"] != 8
        or geometry["gqa_group_size"] != 4
        or not geometry["native_kv_head_storage"]
        or geometry["query_head_sized_kv_cache"]
    ):
        raise RuntimeError("KVQuant sanitizer GQA geometry differs")
    result = {
        "case": case_name,
        "configuration": family,
        "decode_finite": True,
        "fixture_conformance": "PASS",
        "fixed_history_unchanged": True,
        "gqa": "32Q/8KV/native",
        "graph": graph_result,
        "key_active_count": fixture.key_active_count,
        "mode": mode,
        "pointer_stability": "PASS",
        "timing_collected": False,
        "value_active_count_non_sink": 12,
        "value_active_count_sink": 0,
    }
    del (
        cache,
        decode,
        fixture,
        history,
        inputs,
        key_attention,
        key_pre_rope,
        method,
        output,
        positions,
        query,
        sink,
        value,
    )
    gc.collect()
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = _parse_args(sys.argv[1:] if argv is None else argv)
    failures: list[dict[str, str]] = []
    results: list[dict[str, Any]] = []
    environment: dict[str, Any] | None = None
    try:
        environment = _require_exact_environment(
            arguments.image_config_digest
        )
        _require_exact_authority()
        device = torch.device("cuda:0")
        modes = tuple(MODE_CASES) if arguments.mode == "all" else (arguments.mode,)
        for mode in modes:
            results.append(_run_case(mode, device=device))
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

    if failures:
        print(
            json.dumps(
                {"failures": failures, "status": "FAIL"},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    if environment is None:
        raise AssertionError("authorized environment result is absent")
    print(
        json.dumps(
            {
                "authority": _AUTHORITY,
                "container_digest": environment["container_digest"],
                "results": results,
                "status": "PASS",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
