#!/usr/bin/env python3
"""Minimal store, append, and decode probe for Compute Sanitizer."""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import sys

import torch

from kvbench.runtime.turboquant_admission import (
    evaluate_fixture_configuration,
    release_fixture_cuda_resources_for_sanitizer,
    require_authorized_cuda_environment,
)
from kvbench.runtime.turboquant_cache import TURBOQUANT_MANDATORY_CONFIGS
from kvbench.third_party.vllm_turboquant.compat import (
    _build_hadamard_cached,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configuration",
        required=True,
        choices=TURBOQUANT_MANDATORY_CONFIGS,
    )
    parser.add_argument("--image-config-digest", required=True)
    return parser.parse_args(argv)


def _drain_sanitizer_allocator(*, max_passes: int = 4) -> None:
    """Require the PyTorch allocator to reach an observable zero state."""

    last_state = (-1, -1)
    for _ in range(max_passes):
        gc.collect()
        torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch._C._host_emptyCache()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        last_state = (
            int(torch.cuda.memory_allocated()),
            int(torch.cuda.memory_reserved()),
        )
        if last_state == (0, 0):
            return
    raise RuntimeError(
        "CUDA allocator did not drain before reset: "
        f"allocated={last_state[0]} reserved={last_state[1]}"
    )


def _release_sanitizer_cuda_state() -> None:
    """Release the isolated probe context before memcheck leak reporting."""

    torch.cuda.synchronize()
    blas_handle = int(torch.cuda.current_blas_handle())
    _build_hadamard_cached.cache_clear()
    _drain_sanitizer_allocator()

    cublas = ctypes.CDLL("libcublas.so.13")
    destroy = cublas.cublasDestroy_v2
    destroy.argtypes = [ctypes.c_void_p]
    destroy.restype = ctypes.c_int
    destroy_result = int(destroy(ctypes.c_void_p(blas_handle)))
    if destroy_result != 0:
        raise RuntimeError(
            f"cublasDestroy_v2 failed with status {destroy_result}"
        )

    cudart = ctypes.CDLL("libcudart.so.13")
    reset = cudart.cudaDeviceReset
    reset.argtypes = []
    reset.restype = ctypes.c_int
    reset_result = int(reset())
    if reset_result != 0:
        raise RuntimeError(
            f"cudaDeviceReset failed with status {reset_result}"
        )


def main(argv: list[str] | None = None) -> int:
    arguments = _parse_args(sys.argv[1:] if argv is None else argv)
    environment = require_authorized_cuda_environment(
        arguments.image_config_digest
    )
    failures: list[tuple[str, str]] = []
    resources: dict[str, object] = {}
    result: dict[str, object] | None = None
    try:
        result = evaluate_fixture_configuration(
            arguments.configuration,
            release_cuda_resources_for_sanitizer=True,
            sanitizer_resources=resources,
        )
        if result.get("passed") is not True:
            failures.append(
                (
                    "RuntimeError",
                    "TurboQuant sanitizer probe did not conform",
                )
            )
    except Exception as error:
        failures.append((type(error).__name__, str(error)))
        error.__traceback__ = None
        del error
    finally:
        result = None
        try:
            release_fixture_cuda_resources_for_sanitizer(resources)
        except Exception as error:
            failures.append((type(error).__name__, str(error)))
            error.__traceback__ = None
            del error
        try:
            _release_sanitizer_cuda_state()
        except Exception as error:
            failures.append((type(error).__name__, str(error)))
            error.__traceback__ = None
            del error
    if failures:
        print(
            json.dumps(
                {
                    "configuration": arguments.configuration,
                    "failures": [
                        {"type": name, "message": message}
                        for name, message in failures
                    ],
                    "status": "fail",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "configuration": arguments.configuration,
                "container_digest": environment["container_digest"],
                "operation": "store_append_decode",
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
