#!/usr/bin/env python3
"""Validate the corrected kvq3 Value pack with one immutable fixture."""

from __future__ import annotations

import argparse
import ctypes
import gc
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
from types import ModuleType
from typing import Any

import torch


BITS = 3
LEVELS = 8
ZERO_CODE = 3
KV_HEADS = 8
HEAD_DIM = 128
PACKED_ROWS = 12
SINK_TOKENS = 5
STORE_CONTEXT = 17
TOTAL_CONTEXT = 18
STORE_QUANTIZED = STORE_CONTEXT - SINK_TOKENS
QUANTIZED_CONTEXT = TOTAL_CONTEXT - SINK_TOKENS
SPARSE_CAPACITY = 12

DTYPES = {
    "BOOL": torch.bool,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "U8": torch.uint8,
}


class ValidationError(RuntimeError):
    """Raised when the deterministic kvq3 fixture contract does not hold."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        required=True,
        help="Path to one corrected kvq3 fixture case directory",
    )
    parser.add_argument(
        "--extension",
        required=True,
        help="Path to the exact quant_cuda extension under test",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Repeated executions used to reject history-dependent output",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Also capture and replay the corrected store+append path",
    )
    arguments = parser.parse_args()
    if arguments.repeats < 2:
        parser.error("--repeats must be at least 2")
    return arguments


def _load_extension(path: Path) -> ModuleType:
    extension = path.resolve(strict=True)
    spec = importlib.util.spec_from_file_location("quant_cuda", extension)
    if spec is None or spec.loader is None:
        raise ValidationError(f"cannot load CUDA extension: {extension}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loaded = Path(module.__file__).resolve(strict=True)
    if loaded != extension:
        raise ValidationError(
            f"extension identity mismatch: expected={extension} loaded={loaded}"
        )
    return module


def _load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    payload = path.read_bytes()
    if len(payload) < 8:
        raise ValidationError(f"truncated safetensors file: {path}")
    header_size = struct.unpack("<Q", payload[:8])[0]
    header_stop = 8 + header_size
    if header_stop > len(payload):
        raise ValidationError(f"invalid safetensors header size: {path}")
    header = json.loads(payload[8:header_stop])
    if not isinstance(header, dict):
        raise ValidationError(f"invalid safetensors header: {path}")

    tensors: dict[str, torch.Tensor] = {}
    for name, record in header.items():
        if name == "__metadata__":
            continue
        if (
            not isinstance(record, dict)
            or record.get("dtype") not in DTYPES
            or not isinstance(record.get("shape"), list)
            or not isinstance(record.get("data_offsets"), list)
            or len(record["data_offsets"]) != 2
        ):
            raise ValidationError(f"invalid tensor record {name!r}: {path}")
        start, stop = record["data_offsets"]
        if (
            not isinstance(start, int)
            or not isinstance(stop, int)
            or start < 0
            or stop < start
            or header_stop + stop > len(payload)
        ):
            raise ValidationError(f"invalid tensor bounds {name!r}: {path}")
        storage = bytearray(payload[header_stop + start : header_stop + stop])
        tensor = torch.frombuffer(storage, dtype=DTYPES[record["dtype"]]).clone()
        try:
            tensors[name] = tensor.reshape(record["shape"])
        except RuntimeError as error:
            raise ValidationError(
                f"invalid tensor shape {name!r}: {path}"
            ) from error
    return tensors


def _require_tensor(
    tensors: dict[str, torch.Tensor],
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if name not in tensors:
        raise ValidationError(f"fixture tensor is missing: {name}")
    tensor = tensors[name]
    if tuple(tensor.shape) != shape or tensor.dtype != dtype:
        raise ValidationError(
            f"invalid {name}: shape={tuple(tensor.shape)} dtype={tensor.dtype}"
        )
    return tensor


def _assert_exact(
    actual: torch.Tensor,
    expected: torch.Tensor,
    label: str,
) -> None:
    if torch.equal(actual, expected):
        return
    mismatch = int(torch.count_nonzero(actual != expected).item())
    raise ValidationError(f"{label} differs from scalar oracle ({mismatch} words)")


def _fixture_inputs(case: Path) -> dict[str, torch.Tensor]:
    manifest = json.loads((case / "fixture_manifest.json").read_text())
    if (
        manifest.get("bit_width") != BITS
        or manifest.get("schema_version")
        != "kvbench-phase11pr-kvquant-fixture-1.0.0"
        or manifest.get("source", {}).get("decision") != "0025"
        or manifest.get("numerical_control", {}).get("kvq3_scalar_code_pack")
        != "PASS"
    ):
        raise ValidationError("fixture is not a Decision 0025 corrected kvq3 case")

    inputs = _load_safetensors(case / "inputs.safetensors")
    metadata = _load_safetensors(case / "metadata.safetensors")
    dense = _load_safetensors(case / "dense_payload.safetensors")
    sparse_values = _load_safetensors(case / "sparse_values.safetensors")
    sparse_indices = _load_safetensors(case / "sparse_indices.safetensors")
    return {
        "value": _require_tensor(
            inputs,
            "value_after_v_proj",
            shape=(1, KV_HEADS, TOTAL_CONTEXT, HEAD_DIM),
            dtype=torch.bfloat16,
        ),
        "lookup_store": _require_tensor(
            metadata,
            "value_lookup_after_store",
            shape=(TOTAL_CONTEXT, LEVELS),
            dtype=torch.float32,
        ),
        "lookup_append": _require_tensor(
            metadata,
            "value_lookup_after_append",
            shape=(TOTAL_CONTEXT, LEVELS),
            dtype=torch.float32,
        ),
        "lower": _require_tensor(
            metadata,
            "value_dense_lower_bound",
            shape=(TOTAL_CONTEXT,),
            dtype=torch.float32,
        ),
        "upper": _require_tensor(
            metadata,
            "value_dense_upper_bound",
            shape=(TOTAL_CONTEXT,),
            dtype=torch.float32,
        ),
        "scalar_store": _require_tensor(
            dense,
            "value_scalar_control_after_store",
            shape=(KV_HEADS, PACKED_ROWS, STORE_QUANTIZED),
            dtype=torch.int32,
        ),
        "scalar_append": _require_tensor(
            dense,
            "value_scalar_control_after_append",
            shape=(KV_HEADS, PACKED_ROWS, QUANTIZED_CONTEXT),
            dtype=torch.int32,
        ),
        "selected_values": _require_tensor(
            sparse_values,
            "value_selection_by_position",
            shape=(TOTAL_CONTEXT, SPARSE_CAPACITY),
            dtype=torch.float32,
        ),
        "selected_indices": _require_tensor(
            sparse_indices,
            "value_selection_by_position",
            shape=(TOTAL_CONTEXT, SPARSE_CAPACITY),
            dtype=torch.int32,
        ),
        "selected_counts": _require_tensor(
            sparse_indices,
            "value_active_count_by_position",
            shape=(TOTAL_CONTEXT,),
            dtype=torch.int32,
        ),
    }


def _validate(
    quant_cuda: ModuleType,
    case: Path,
    repeats: int,
    graph_test: bool,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise ValidationError("CUDA is unavailable")
    fixture = _fixture_inputs(case)
    device = torch.device("cuda")

    value = fixture["value"].to(device=device).contiguous()
    lookup_store = fixture["lookup_store"].to(device=device).contiguous()
    lookup_append = fixture["lookup_append"].to(device=device).contiguous()
    lower = fixture["lower"].to(device=device).contiguous()
    upper = fixture["upper"].to(device=device).contiguous()
    scalar_store = fixture["scalar_store"].to(device=device).contiguous()
    scalar_append = fixture["scalar_append"].to(device=device).contiguous()
    selected_values = fixture["selected_values"].to(device=device).contiguous()
    selected_indices = fixture["selected_indices"].to(device=device).contiguous()
    selected_counts = fixture["selected_counts"].to(device=device).contiguous()

    store_value = (
        value[0, :, SINK_TOKENS:STORE_CONTEXT, :]
        .transpose(1, 2)
        .float()
        .contiguous()
    )
    append_value = value[0, :, STORE_CONTEXT, :].float().reshape(-1).contiguous()
    store_lower = lower[SINK_TOKENS:STORE_CONTEXT].contiguous()
    store_upper = upper[SINK_TOKENS:STORE_CONTEXT].contiguous()
    append_lower = lower[STORE_CONTEXT : STORE_CONTEXT + 1].contiguous()
    append_upper = upper[STORE_CONTEXT : STORE_CONTEXT + 1].contiguous()
    append_metadata = lookup_append[STORE_QUANTIZED].contiguous()
    append_zero = append_metadata[ZERO_CODE : ZERO_CODE + 1].contiguous()
    append_selected_values = selected_values[STORE_CONTEXT].contiguous()
    append_selected_indices = selected_indices[STORE_CONTEXT].contiguous()
    append_selected_count = selected_counts[
        STORE_CONTEXT : STORE_CONTEXT + 1
    ].contiguous()

    packed = torch.zeros(
        (KV_HEADS, PACKED_ROWS, TOTAL_CONTEXT),
        dtype=torch.int32,
        device=device,
    )
    lookup = torch.empty_like(lookup_store)
    sparse_values = torch.zeros(
        (TOTAL_CONTEXT, SPARSE_CAPACITY),
        dtype=torch.float32,
        device=device,
    )
    sparse_indices = torch.zeros(
        (TOTAL_CONTEXT, SPARSE_CAPACITY),
        dtype=torch.int32,
        device=device,
    )
    sparse_counts = torch.zeros(
        TOTAL_CONTEXT,
        dtype=torch.int32,
        device=device,
    )
    observed_store = torch.empty_like(scalar_store)
    observed_append = torch.empty_like(scalar_append)
    owned = (
        packed,
        lookup,
        sparse_values,
        sparse_indices,
        sparse_counts,
        observed_store,
        observed_append,
    )
    pointers = tuple(tensor.data_ptr() for tensor in owned)

    def operation() -> None:
        packed.zero_()
        lookup.copy_(lookup_store)
        sparse_values.zero_()
        sparse_indices.zero_()
        sparse_counts.zero_()
        quant_cuda.vecquant3appendvecVsparseParallel(
            packed,
            lookup,
            store_value,
            store_lower,
            store_upper,
        )
        observed_store.copy_(packed[:, :, :STORE_QUANTIZED])
        quant_cuda.append_value_sparse_1024_cap12_out(
            packed,
            lookup,
            append_value,
            append_metadata,
            append_lower,
            append_upper,
            append_zero,
            append_selected_values,
            append_selected_indices,
            append_selected_count,
            sparse_values,
            sparse_indices,
            sparse_counts,
            STORE_QUANTIZED,
            BITS,
        )
        observed_append.copy_(packed[:, :, :QUANTIZED_CONTEXT])

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    for iteration in range(repeats):
        with torch.cuda.stream(side):
            operation()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        _assert_exact(
            observed_store,
            scalar_store,
            f"repeat {iteration} kvq3 parallel store",
        )
        _assert_exact(
            observed_append,
            scalar_append,
            f"repeat {iteration} kvq3 store+append",
        )
        if pointers != tuple(tensor.data_ptr() for tensor in owned):
            raise ValidationError("caller-owned output pointer changed")

    graph_result = "NOT_RUN"
    graph: torch.cuda.CUDAGraph | None = None
    if graph_test:
        for _ in range(3):
            operation()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            operation()
        graph.replay()
        torch.cuda.synchronize()
        _assert_exact(observed_store, scalar_store, "graph kvq3 parallel store")
        _assert_exact(observed_append, scalar_append, "graph kvq3 store+append")
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        if torch.cuda.memory_allocated() != allocated:
            raise ValidationError("graph replay changed allocated bytes")
        if torch.cuda.memory_reserved() != reserved:
            raise ValidationError("graph replay changed reserved bytes")
        if pointers != tuple(tensor.data_ptr() for tensor in owned):
            raise ValidationError("graph replay changed caller-owned pointers")
        graph_result = "PASS_ZERO_REPLAY_ALLOCATION"

    result = {
        "status": "PASS",
        "case": case.name,
        "repeats": repeats,
        "parallel_store_scalar_control": "PASS",
        "append_scalar_control": "PASS",
        "non_default_stream": "PASS",
        "pointer_stability": "PASS",
        "cuda_graph": graph_result,
    }
    if graph is not None:
        graph.reset()
    del (
        fixture,
        value,
        lookup_store,
        lookup_append,
        lower,
        upper,
        scalar_store,
        scalar_append,
        selected_values,
        selected_indices,
        selected_counts,
        store_value,
        append_value,
        store_lower,
        store_upper,
        append_lower,
        append_upper,
        append_metadata,
        append_zero,
        append_selected_values,
        append_selected_indices,
        append_selected_count,
        packed,
        lookup,
        sparse_values,
        sparse_indices,
        sparse_counts,
        observed_store,
        observed_append,
        owned,
        side,
        operation,
        graph,
    )
    return result


def _release_tracked_cuda_storages() -> None:
    storages: dict[int, object] = {}
    tracked = gc.get_objects()
    for value in tracked:
        if type(value) is not torch.Tensor or not value.is_cuda:
            continue
        try:
            storage = value.untyped_storage()
        except RuntimeError:
            continue
        if int(storage.nbytes()) != 0:
            storages.setdefault(int(storage._cdata), storage)
    del tracked
    for storage in storages.values():
        storage.resize_(0)
    storages.clear()
    gc.collect()


def _reset_cuda_for_sanitizer() -> None:
    torch.cuda.synchronize()
    blas_handle = int(torch.cuda.current_blas_handle())
    _release_tracked_cuda_storages()
    state = (-1, -1)
    for _ in range(8):
        gc.collect()
        torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch._C._host_emptyCache()
        torch.cuda.synchronize()
        state = (
            int(torch.cuda.memory_allocated()),
            int(torch.cuda.memory_reserved()),
        )
        if state == (0, 0):
            break
    if state != (0, 0):
        raise ValidationError(
            "CUDA allocator did not drain: "
            f"allocated={state[0]} reserved={state[1]}"
        )

    cublas = ctypes.CDLL("libcublas.so.13")
    destroy = cublas.cublasDestroy_v2
    destroy.argtypes = [ctypes.c_void_p]
    destroy.restype = ctypes.c_int
    status = int(destroy(ctypes.c_void_p(blas_handle)))
    if status != 0:
        raise ValidationError(f"cublasDestroy_v2 failed with status {status}")

    cudart = ctypes.CDLL("libcudart.so.13")
    reset = cudart.cudaDeviceReset
    reset.argtypes = []
    reset.restype = ctypes.c_int
    status = int(reset())
    if status != 0:
        raise ValidationError(f"cudaDeviceReset failed with status {status}")


def main() -> int:
    arguments = _parse_args()
    result: dict[str, Any] | None = None
    failure: tuple[str, str] | None = None
    cuda_started = False
    try:
        fixture = Path(arguments.fixture).resolve(strict=True)
        quant_cuda = _load_extension(Path(arguments.extension))
        cuda_started = True
        result = _validate(
            quant_cuda,
            fixture,
            arguments.repeats,
            arguments.graph,
        )
        del quant_cuda
    except Exception as error:
        failure = (type(error).__name__, str(error))
        error.__traceback__ = None
        del error
    finally:
        if cuda_started:
            try:
                _reset_cuda_for_sanitizer()
            except Exception as error:
                failure = (type(error).__name__, str(error))
                error.__traceback__ = None
                del error

    if failure is not None:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error_type": failure[0],
                    "reason": failure[1],
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    if result is None:
        raise ValidationError("validation produced no result")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
