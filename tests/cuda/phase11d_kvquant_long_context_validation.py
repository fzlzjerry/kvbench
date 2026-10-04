#!/usr/bin/env python3
"""Narrow correctness checks for the Phase 11D deterministic q4 Value API."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from kvbench.runtime.allocation import audit_cuda_allocations


AUTHORIZED_IMAGE = (
    "sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e"
)
API_NAME = (
    "vecquant4matmul_nuq_perchannel_transposed_"
    "mha_batched_fused_opt2_deterministic_out"
)
WIDTH = 4092
CAPACITY = WIDTH + 5
QUERY_HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
SPARSE_CAP = 12
TILE_WIDTH = 128
MAX_TILES = (WIDTH + TILE_WIDTH - 1) // TILE_WIDTH
ATOL = 0.01
RTOL = 0.01


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "determinism",
            "fixtures",
            "stream",
            "graph",
            "sanitizer",
        ),
    )
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--extension-sha256", required=True)
    parser.add_argument("--fixture-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().contiguous().view(torch.uint8).cpu()
    payload = bytes(contiguous.untyped_storage())[: int(contiguous.numel())]
    return hashlib.sha256(payload).hexdigest()


def _load_extension(path: Path, expected_sha256: str) -> ModuleType:
    resolved = path.resolve(strict=True)
    if _sha256(resolved) != expected_sha256:
        raise RuntimeError("Phase 11D extension SHA-256 differs")
    specification = importlib.util.spec_from_file_location("quant_cuda", resolved)
    if specification is None or specification.loader is None:
        raise RuntimeError("Phase 11D extension loader is unavailable")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    if not callable(getattr(module, API_NAME, None)):
        raise RuntimeError("Phase 11D deterministic out-API is absent")
    return module


def _require_environment() -> None:
    if (
        not Path("/.dockerenv").is_file()
        or os.environ.get("KVBENCH_AUTHORIZED_IMAGE_DIGEST") != AUTHORIZED_IMAGE
        or os.environ.get("KVBENCH_EXECUTION_ENVIRONMENT")
        != "measurement_container"
    ):
        raise RuntimeError(
            "Phase 11D CUDA validation requires the authorized container"
        )


def _synthetic_inputs(*, sparse: bool) -> dict[str, torch.Tensor]:
    device = torch.device("cuda:0")
    vector = torch.linspace(
        -0.75,
        0.875,
        steps=QUERY_HEADS * WIDTH,
        dtype=torch.float32,
        device=device,
    ).view(1, QUERY_HEADS, WIDTH)
    packed = torch.full(
        (KV_HEADS, 16, CAPACITY),
        0x76543210,
        dtype=torch.int32,
        device=device,
    )
    lookup = torch.linspace(
        -1.25,
        1.5,
        steps=CAPACITY * 16,
        dtype=torch.float32,
        device=device,
    ).view(CAPACITY, 16)
    if sparse:
        outliers = (
            torch.arange(
                CAPACITY * SPARSE_CAP,
                dtype=torch.float32,
                device=device,
            )
            .view(CAPACITY, SPARSE_CAP)
            .remainder_(97)
            .sub_(48)
            .mul_(0.0005)
        )
    else:
        outliers = torch.zeros(
            (CAPACITY, SPARSE_CAP),
            dtype=torch.float32,
            device=device,
        )
    indices = (
        torch.arange(SPARSE_CAP, dtype=torch.int32, device=device)
        .view(1, SPARSE_CAP)
        .expand(CAPACITY, SPARSE_CAP)
        .contiguous()
    )
    return {
        "vector": vector,
        "packed": packed,
        "lookup": lookup,
        "outliers": outliers,
        "indices": indices,
        "output": torch.empty(
            (1, QUERY_HEADS, HEAD_DIM),
            dtype=torch.float32,
            device=device,
        ),
        "workspace": torch.empty(
            (1, QUERY_HEADS, MAX_TILES, HEAD_DIM),
            dtype=torch.float32,
            device=device,
        ),
    }


def _call(runtime: ModuleType, inputs: dict[str, torch.Tensor]) -> None:
    getattr(runtime, API_NAME)(
        inputs["vector"],
        inputs["packed"],
        inputs["output"],
        inputs["lookup"],
        WIDTH,
        inputs["outliers"],
        inputs["indices"],
        inputs["workspace"],
    )


def _pointers(inputs: dict[str, torch.Tensor]) -> dict[str, int]:
    return {
        name: int(tensor.data_ptr())
        for name, tensor in sorted(inputs.items())
    }


def _independent_control(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    vector = inputs["vector"].detach().cpu()
    packed = inputs["packed"].detach().cpu()
    lookup = inputs["lookup"].detach().cpu()
    outliers = inputs["outliers"].detach().cpu()
    indices = inputs["indices"].detach().cpu().to(torch.int64)

    dense = torch.empty((KV_HEADS, HEAD_DIM, WIDTH), dtype=torch.float32)
    tokens = torch.arange(WIDTH, dtype=torch.int64)
    for kv_head in range(KV_HEADS):
        for channel in range(HEAD_DIM):
            words = packed[kv_head, channel // 8, :WIDTH].to(torch.int64)
            codes = torch.bitwise_and(
                torch.bitwise_right_shift(words, 4 * (channel % 8)),
                0xF,
            )
            dense[kv_head, channel] = lookup[tokens, codes]

    sparse_rows = torch.zeros(
        (WIDTH, KV_HEADS * HEAD_DIM),
        dtype=torch.float32,
    )
    sparse_rows.scatter_add_(1, indices[:WIDTH], outliers[:WIDTH])
    reconstructed = dense + sparse_rows.transpose(0, 1).reshape(
        KV_HEADS,
        HEAD_DIM,
        WIDTH,
    )
    control = torch.empty((1, QUERY_HEADS, HEAD_DIM), dtype=torch.float64)
    for query_head in range(QUERY_HEADS):
        kv_head = query_head // 4
        control[0, query_head] = torch.sum(
            reconstructed[kv_head].to(torch.float64)
            * vector[0, query_head].to(torch.float64),
            dim=1,
        )
    return control.to(torch.float32)


def _control_comparison(
    observed: torch.Tensor,
    control: torch.Tensor,
) -> dict[str, Any]:
    observed_cpu = observed.detach().cpu()
    difference = (observed_cpu - control).abs()
    relative = difference / torch.maximum(
        control.abs(),
        torch.full_like(control, 1.0e-12),
    )
    passed = bool(torch.allclose(observed_cpu, control, atol=ATOL, rtol=RTOL))
    if not passed:
        raise AssertionError("Phase 11D independent numerical control differs")
    return {
        "atol": ATOL,
        "rtol": RTOL,
        "control_sha256": _tensor_sha256(control),
        "maximum_absolute_difference": float(difference.max()),
        "maximum_relative_difference": float(relative.max()),
        "passed": passed,
    }


def _run_repetitions(
    runtime: ModuleType,
    inputs: dict[str, torch.Tensor],
    *,
    repetitions: int,
) -> dict[str, Any]:
    pointers_before = _pointers(inputs)
    hashes: list[str] = []
    for _ in range(repetitions):
        _call(runtime, inputs)
        torch.cuda.synchronize()
        hashes.append(_tensor_sha256(inputs["output"]))
    if len(set(hashes)) != 1:
        raise AssertionError("Phase 11D output remains history-dependent")
    if not bool(torch.isfinite(inputs["output"]).all()):
        raise AssertionError("Phase 11D output contains NaN or Inf")
    if pointers_before != _pointers(inputs):
        raise AssertionError("Phase 11D caller-owned pointer changed")
    return {
        "repetitions": repetitions,
        "output_sha256": hashes,
        "unique_output_sha256_count": len(set(hashes)),
        "all_finite": True,
        "caller_owned_pointers_stable": True,
    }


def _run_determinism(runtime: ModuleType) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for sparse in (False, True):
        inputs = _synthetic_inputs(sparse=sparse)
        result = _run_repetitions(runtime, inputs, repetitions=100)
        control = _independent_control(inputs)
        result["sparse_enabled"] = sparse
        result["independent_control"] = _control_comparison(
            inputs["output"],
            control,
        )
        cases.append(result)
    return {
        "mode": "determinism",
        "quantized_context": WIDTH,
        "workspace_shape": [1, QUERY_HEADS, MAX_TILES, HEAD_DIM],
        "workspace_bytes": (
            QUERY_HEADS * MAX_TILES * HEAD_DIM * torch.finfo(torch.float32).bits
            // 8
        ),
        "cases": cases,
        "passed": True,
    }


class _FixtureRuntime:
    def __init__(self, runtime: ModuleType) -> None:
        self._runtime = runtime
        self._workspace = torch.empty(
            (1, QUERY_HEADS, MAX_TILES, HEAD_DIM),
            dtype=torch.float32,
            device="cuda",
        )
        self._workspace_pointer = int(self._workspace.data_ptr())

    def __getattr__(self, name: str) -> Any:
        legacy_name = (
            "vecquant4matmul_nuq_perchannel_transposed_"
            "mha_batched_fused_opt2"
        )
        if name == legacy_name:
            return self._q4_deterministic
        return getattr(self._runtime, name)

    def _q4_deterministic(
        self,
        vector: torch.Tensor,
        packed: torch.Tensor,
        output: torch.Tensor,
        lookup: torch.Tensor,
        width: int,
        outliers: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        getattr(self._runtime, API_NAME)(
            vector,
            packed,
            output,
            lookup,
            width,
            outliers,
            indices,
            self._workspace,
        )
        if int(self._workspace.data_ptr()) != self._workspace_pointer:
            raise AssertionError("fixture workspace pointer changed")


def _load_source_fixture_module(source_root: Path) -> ModuleType:
    path = (
        source_root.resolve(strict=True)
        / "tests_phase11p"
        / "phase11p_cuda_validation.py"
    )
    specification = importlib.util.spec_from_file_location(
        "phase11p_cuda_validation",
        path,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Phase 11P fixture validator loader is unavailable")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _run_fixtures(
    runtime: ModuleType,
    fixture_root: Path,
    source_root: Path,
) -> dict[str, Any]:
    module = _load_source_fixture_module(source_root)
    proxy = _FixtureRuntime(runtime)
    module.run_all_fixtures(proxy, fixture_root.resolve(strict=True))
    return {
        "mode": "fixtures",
        "cases": 9,
        "q4_decode_api": API_NAME,
        "q3_q2_paths": "unchanged",
        "passed": True,
    }


def _run_stream(runtime: ModuleType) -> dict[str, Any]:
    canonical = _synthetic_inputs(sparse=True)
    _call(runtime, canonical)
    torch.cuda.synchronize()
    canonical_sha256 = _tensor_sha256(canonical["output"])

    inputs = _synthetic_inputs(sparse=True)
    expected_vector = inputs["vector"].clone()
    inputs["vector"].zero_()
    side = torch.cuda.Stream()
    complete = torch.cuda.Event()
    with torch.cuda.stream(side):
        torch.cuda._sleep(50_000_000)
        inputs["vector"].copy_(expected_vector)
        _call(runtime, inputs)
        complete.record(side)
    torch.cuda.current_stream().wait_event(complete)
    torch.cuda.synchronize()
    observed_sha256 = _tensor_sha256(inputs["output"])
    if observed_sha256 != canonical_sha256:
        raise AssertionError("Phase 11D non-default stream ordering failed")
    return {
        "mode": "stream",
        "side_stream": int(side.cuda_stream),
        "default_stream": int(torch.cuda.default_stream().cuda_stream),
        "output_sha256": observed_sha256,
        "passed": True,
    }


def _run_graph(runtime: ModuleType) -> dict[str, Any]:
    inputs = _synthetic_inputs(sparse=True)
    for _ in range(3):
        _call(runtime, inputs)
    torch.cuda.synchronize()
    eager_sha256 = _tensor_sha256(inputs["output"])
    pointers_before = _pointers(inputs)
    eager_allocation = audit_cuda_allocations(
        lambda: _call(runtime, inputs),
        device=torch.device("cuda:0"),
    )
    if not eager_allocation.passed:
        raise AssertionError("Phase 11D eager operation allocated")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _call(runtime, inputs)
    graph.replay()
    torch.cuda.synchronize()
    graph_sha256 = _tensor_sha256(inputs["output"])
    if graph_sha256 != eager_sha256:
        raise AssertionError("Phase 11D eager/graph output differs")
    graph_allocation = audit_cuda_allocations(
        graph.replay,
        device=torch.device("cuda:0"),
    )
    if not graph_allocation.passed:
        raise AssertionError("Phase 11D graph replay allocated")

    replay_hashes: list[str] = []
    for _ in range(10):
        graph.replay()
        torch.cuda.synchronize()
        replay_hashes.append(_tensor_sha256(inputs["output"]))
    if len(set(replay_hashes)) != 1 or replay_hashes[0] != eager_sha256:
        raise AssertionError("Phase 11D graph replay is not exact")
    if pointers_before != _pointers(inputs):
        raise AssertionError("Phase 11D graph changed caller-owned pointers")
    graph.reset()
    return {
        "mode": "graph",
        "capture": "PASS",
        "replay": "PASS",
        "eager_output_sha256": eager_sha256,
        "replay_output_sha256": replay_hashes,
        "caller_owned_pointers_stable": True,
        "eager_allocation": eager_allocation.to_dict(),
        "replay_allocation": graph_allocation.to_dict(),
        "passed": True,
    }


def _reset_cuda_for_sanitizer() -> None:
    torch.cuda.synchronize()
    gc.collect()
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    torch._C._host_emptyCache()
    torch.cuda.synchronize()
    cudart = ctypes.CDLL("libcudart.so.13")
    reset = cudart.cudaDeviceReset
    reset.argtypes = []
    reset.restype = ctypes.c_int
    status = int(reset())
    if status != 0:
        raise RuntimeError(f"cudaDeviceReset failed with status {status}")


def _run_sanitizer(runtime: ModuleType) -> dict[str, Any]:
    inputs = _synthetic_inputs(sparse=True)
    _call(runtime, inputs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _call(runtime, inputs)
    graph.replay()
    torch.cuda.synchronize()
    output_sha256 = _tensor_sha256(inputs["output"])
    graph.reset()
    del graph
    del inputs
    _reset_cuda_for_sanitizer()
    return {
        "mode": "sanitizer",
        "eager": "PASS",
        "graph_replay": "PASS",
        "output_sha256": output_sha256,
        "passed": True,
    }


def main() -> None:
    arguments = _parse_args()
    _require_environment()
    runtime = _load_extension(
        arguments.extension,
        arguments.extension_sha256,
    )
    if arguments.mode == "determinism":
        result = _run_determinism(runtime)
    elif arguments.mode == "fixtures":
        if arguments.fixture_root is None or arguments.source_root is None:
            raise RuntimeError("fixtures mode requires fixture and source roots")
        result = _run_fixtures(
            runtime,
            arguments.fixture_root,
            arguments.source_root,
        )
    elif arguments.mode == "stream":
        result = _run_stream(runtime)
    elif arguments.mode == "graph":
        result = _run_graph(runtime)
    else:
        result = _run_sanitizer(runtime)
    result.update(
        {
            "schema_version": (
                "kvbench-phase11d-long-context-cuda-validation-1.0.0"
            ),
            "authorized_image_digest": AUTHORIZED_IMAGE,
            "extension_sha256": arguments.extension_sha256,
            "performance_timing": False,
        }
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
