#!/usr/bin/env python3
"""Minimal exact-container KIVI probe for Compute Sanitizer memcheck."""

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
from kvbench.adapters.kivi import (
    KIVI_DECISION_0018_PATCH_SHA256,
    KIVI_EXTENSION_SHA256,
    KIVI_FIXTURE_ROOT_SHA256,
    KIVI_NEW_PACK_SHA256,
    KIVI_OFFICIAL_BASE_TREE,
    KIVI_OFFICIAL_COMMIT,
    KIVI_PATCHED_TREE,
    KIVIMethodAdapter,
)
from kvbench.runtime.turboquant_admission import (
    PHASE6_CONTAINER_ENVIRONMENT_VALUE,
    PHASE6_CONTAINER_ENVIRONMENT_VARIABLE,
    PHASE6_IMAGE_ENVIRONMENT_VARIABLE,
    require_authorized_cuda_environment,
)
from kvbench.runtime.turboquant_cache import (
    _release_tensor_storages_for_sanitizer,
)
from kvbench.schema.phase6 import AUTHORIZED_CONTAINER_DIGEST


_AUTHORITY = {
    "official_commit": "876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6",
    "official_base_tree": "c94c31b2cfd44eeb9a18cff9dcdf03adff4ac49b",
    "patched_tree": "b617493dea5aff1a754cd27ad6be12ac512b2aee",
    "decision_0018_patch_sha256": (
        "c9c2dd52d4c81b844d1d1d7218ad2cd60a5b31574a387f716d466cb01310423d"
    ),
    "extension_sha256": (
        "45d29ec1a3cecc4b253d1d1dd6139ef4f91cff88993db61a9d73685314851aa9"
    ),
    "new_pack_sha256": (
        "3678af0e34a0ba18e5d80a4128acf11d4070667c800a15540a16d07253a4f75e"
    ),
    "fixture_root_sha256": (
        "abd164da0adf9e0c1404e8fba1f6a6e42e57944481cdf060b91e8cef175ed302"
    ),
}


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-config-digest",
        required=True,
        help="Decision 0016 Measurement Container digest",
    )
    return parser.parse_args(argv)


def _require_exact_container(declared_digest: str) -> dict[str, Any]:
    if (
        not Path("/.dockerenv").is_file()
        or os.environ.get(PHASE6_IMAGE_ENVIRONMENT_VARIABLE)
        != AUTHORIZED_CONTAINER_DIGEST
        or os.environ.get(PHASE6_CONTAINER_ENVIRONMENT_VARIABLE)
        != PHASE6_CONTAINER_ENVIRONMENT_VALUE
    ):
        raise RuntimeError(
            "Phase 8 KIVI sanitizer execution is forbidden outside the exact "
            "Measurement Container"
        )
    return require_authorized_cuda_environment(declared_digest)


def _require_exact_kivi_authority() -> None:
    observed = {
        "official_commit": KIVI_OFFICIAL_COMMIT,
        "official_base_tree": KIVI_OFFICIAL_BASE_TREE,
        "patched_tree": KIVI_PATCHED_TREE,
        "decision_0018_patch_sha256": KIVI_DECISION_0018_PATCH_SHA256,
        "extension_sha256": KIVI_EXTENSION_SHA256,
        "new_pack_sha256": KIVI_NEW_PACK_SHA256,
        "fixture_root_sha256": KIVI_FIXTURE_ROOT_SHA256,
    }
    if observed != _AUTHORITY:
        raise RuntimeError("KIVI sanitizer source authority differs")


def _runtime_context() -> MethodRuntimeContext:
    return MethodRuntimeContext(
        model_id="phase8-kivi-sanitizer",
        model_revision=KIVI_OFFICIAL_COMMIT,
        backend_id="patched-official-kivi-sanitizer",
        backend_fingerprint=hashlib.sha256(
            b"phase8-patched-official-kivi-sanitizer"
        ).hexdigest(),
        num_layers=32,
        num_query_heads=32,
        num_kv_heads=8,
        head_dim=128,
    )


def _inputs(device: torch.device) -> tuple[Any, Any, Any, Any]:
    generator = torch.Generator(device=device)
    generator.manual_seed(8018)
    key = torch.randn(
        (1, 8, 34, 128),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    value = torch.randn(
        (1, 8, 34, 128),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    query = torch.randn(
        (1, 32, 3, 128),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    positions = torch.arange(34, dtype=torch.int64, device=device)
    return key, value, query, positions


def _prefill(
    method: KIVIMethodAdapter,
    cache: Any,
    key: Any,
    value: Any,
    positions: Any,
    length: int,
) -> None:
    cache.prepare_prefill(length)
    method.store_prefill(
        cache,
        key[:, :, :length, :],
        value[:, :, :length, :],
        0,
        positions[:length],
    )
    cache.complete_prefill()


def _assert_token_movement(cache: Any, expected_context: int) -> None:
    state = cache.token_index_state(0)
    key_tokens = (
        state["quantized_key_tokens"].tolist()
        + state["residual_key_tokens"].tolist()
    )
    value_tokens = (
        state["quantized_value_tokens"].tolist()
        + state["residual_value_tokens"].tolist()
    )
    expected = list(range(expected_context))
    if key_tokens != expected or value_tokens != expected:
        raise RuntimeError("KIVI rollover has a missing, duplicate, or reordered token")

    historical_key_count = (expected_context // 32) * 32
    historical_value_count = max(0, expected_context - 32)
    if (
        state["quantized_key_tokens"].tolist()
        != expected[:historical_key_count]
        or state["residual_key_tokens"].tolist()
        != expected[historical_key_count:]
        or state["quantized_value_tokens"].tolist()
        != expected[:historical_value_count]
        or state["residual_value_tokens"].tolist()
        != expected[historical_value_count:]
    ):
        raise RuntimeError("KIVI rollover region ownership differs")


def _decode(
    method: KIVIMethodAdapter,
    cache: Any,
    attention: Any,
    key: Any,
    value: Any,
    query: Any,
    positions: Any,
    token: int,
    query_index: int,
) -> Any:
    handles = method.append_decode(
        cache,
        key[:, :, token : token + 1, :],
        value[:, :, token : token + 1, :],
        0,
        positions[token : token + 1],
    )
    output = method.decode_attention(
        attention,
        query[:, :, query_index : query_index + 1, :],
        handles[0],
        handles[1],
        scaling=1.0 / math.sqrt(128),
    )
    torch.cuda.synchronize(cache.device)
    if not bool(torch.isfinite(output).all().item()):
        raise RuntimeError("KIVI sanitizer decode produced NaN or Inf")
    return output


def _run_k4v4_rollover(device: torch.device) -> dict[str, Any]:
    method = KIVIMethodAdapter(_runtime_context(), "k4v4")
    method.prepare_runtime()
    cache = method.allocate(batch_size=1, capacity=34, device=device)
    cache.initialize_deterministic()
    key, value, query, positions = _inputs(device)
    attention = SimpleNamespace(layer_idx=0)

    _prefill(method, cache, key, value, positions, 31)
    _assert_token_movement(cache, 31)
    cache.prepare_growing(31, 2)
    for step, token in enumerate((31, 32)):
        cache.select_growing_step(step)
        _decode(
            method,
            cache,
            attention,
            key,
            value,
            query,
            positions,
            token,
            step,
        )
        _assert_token_movement(cache, token + 1)

    # L31->L33 has now crossed both official rollover boundaries.  Decode one
    # fixed scratch token at L33 so historical V invokes bgemv4 as well; fixed
    # execution must not mutate the committed historical state.
    checksum_before = cache.history_checksum(0)
    cache.prepare_fixed(33)
    _decode(
        method,
        cache,
        attention,
        key,
        value,
        query,
        positions,
        33,
        2,
    )
    if cache.history_checksum(0) != checksum_before:
        raise RuntimeError("KIVI fixed scratch decode mutated historical state")
    _assert_token_movement(cache, 33)
    fingerprint = method.config_fingerprint(cache.layout_fingerprint())
    result = {
        "configuration": "k4v4",
        "gemv_bits": [4],
        "rollover": "L31_to_L33",
        "active_context": cache.active_context,
        "key_history_tokens": cache._key_history_counts[0],
        "key_residual_tokens": cache._key_residual_counts[0],
        "value_history_tokens": cache._value_history_counts[0],
        "value_residual_tokens": cache._value_residual_counts[0],
        "method_fingerprint": fingerprint,
        "finite": True,
        "token_movement": "exact",
    }
    cache.release_owned_cuda_resources_for_sanitizer()
    _release_tensor_storages_for_sanitizer((key, value, query, positions))
    del attention, cache, key, method, positions, query, value
    return result


def _run_k2v2(device: torch.device) -> dict[str, Any]:
    method = KIVIMethodAdapter(_runtime_context(), "k2v2")
    method.prepare_runtime()
    cache = method.allocate(batch_size=1, capacity=34, device=device)
    cache.initialize_deterministic()
    key, value, query, positions = _inputs(device)
    attention = SimpleNamespace(layer_idx=0)

    _prefill(method, cache, key, value, positions, 33)
    _assert_token_movement(cache, 33)
    checksum_before = cache.history_checksum(0)
    cache.prepare_fixed(33)
    _decode(
        method,
        cache,
        attention,
        key,
        value,
        query,
        positions,
        33,
        0,
    )
    if cache.history_checksum(0) != checksum_before:
        raise RuntimeError("KIVI 2-bit fixed decode mutated historical state")
    _assert_token_movement(cache, 33)
    fingerprint = method.config_fingerprint(cache.layout_fingerprint())
    result = {
        "configuration": "k2v2",
        "gemv_bits": [2],
        "active_context": cache.active_context,
        "method_fingerprint": fingerprint,
        "finite": True,
        "token_movement": "exact",
    }
    cache.release_owned_cuda_resources_for_sanitizer()
    _release_tensor_storages_for_sanitizer((key, value, query, positions))
    del attention, cache, key, method, positions, query, value
    return result


def _reset_cuda_for_memcheck() -> None:
    """Release the isolated CUDA context before memcheck leak reporting."""

    torch.cuda.synchronize()
    blas_handle = int(torch.cuda.current_blas_handle())
    gc.collect()
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    torch._C._host_emptyCache()
    torch.cuda.synchronize()

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


def main(argv: list[str] | None = None) -> int:
    arguments = _parse_args(sys.argv[1:] if argv is None else argv)
    failures: list[dict[str, str]] = []
    results: list[dict[str, Any]] = []
    environment: dict[str, Any] | None = None
    try:
        environment = _require_exact_container(arguments.image_config_digest)
        _require_exact_kivi_authority()
        device = torch.device("cuda:0")
        results.append(_run_k4v4_rollover(device))
        results.append(_run_k2v2(device))
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
                {"failures": failures, "status": "fail"},
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
                "configurations": results,
                "container_digest": environment["container_digest"],
                "kernel_families": [
                    "bgemv2_kernel_outer_dim",
                    "bgemv4_kernel_outer_dim",
                ],
                "status": "pass",
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
