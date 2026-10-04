#!/usr/bin/env python3
"""Run only the frozen Phase 10 KVQuant sanitizer-path matrix."""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import sys

from reference.kvquant import generate_fixtures


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--calibration-root", required=True)
    parser.add_argument("--patch-manifest", required=True)
    return parser.parse_args()


def _release_tracked_cuda_storages() -> None:
    """Irreversibly release all tensors owned by this isolated probe."""

    torch = __import__("torch")
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
    """Release the isolated CUDA context before memcheck leak reporting."""

    torch = __import__("torch")
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


def main() -> int:
    arguments = _parse_args()
    output: dict[str, object] | None = None
    failure: tuple[str, str] | None = None
    try:
        output = generate_fixtures.sanitizer_probe(arguments)
    except Exception as error:
        failure = (type(error).__name__, str(error))
        error.__traceback__ = None
        del error
    finally:
        try:
            _reset_cuda_for_memcheck()
        except Exception as error:
            failure = (type(error).__name__, str(error))
            error.__traceback__ = None
            del error
    if failure is not None:
        print(
            json.dumps(
                {
                    "error_type": failure[0],
                    "reason": failure[1],
                    "status": "FAIL",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    if output is None:
        raise RuntimeError("sanitizer probe produced no result")
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
