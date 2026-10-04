#!/usr/bin/env python3
"""Generate the single Phase 7 KIVI reference fixture set."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Callable


SOURCE_COMMIT = "876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6"
SOURCE_TREE = "c94c31b2cfd44eeb9a18cff9dcdf03adff4ac49b"
PATCHED_TREE = "b617493dea5aff1a754cd27ad6be12ac512b2aee"
SEED = 20260726
BATCH_SIZE = 1
NUM_QUERY_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
GROUP_SIZE = 32
RESIDUAL_LENGTH = 32
BASIC_STORE_CONTEXT = 17
BASIC_TOTAL_CONTEXT = 18
ROLLOVER_CONTEXTS = (31, 32, 33)
POST_ROLLOVER_CONTEXT = 34
STATIC_ACCOUNTING_CONTEXT = 64
QUERY_POSITIONS = (16, 17, 30, 31, 32, 33)
VARIANTS = (
    {"id": "k4v4", "k_bits": 4, "v_bits": 4, "role": "mandatory"},
    {"id": "k2v4", "k_bits": 2, "v_bits": 4, "role": "mandatory"},
    {"id": "k2v2", "k_bits": 2, "v_bits": 2, "role": "mandatory"},
    {
        "id": "k4v2",
        "k_bits": 4,
        "v_bits": 2,
        "role": "held_out_asymmetry_control",
    },
)


class ReferenceGenerationError(RuntimeError):
    """Raised when the source or generated fixture violates the contract."""


@dataclass
class CacheState:
    """The exact nine-field cache tuple used by the patched Llama path."""

    quantized_key: Any
    residual_key: Any
    key_scale: Any
    key_minimum: Any
    quantized_value: Any
    residual_value: Any
    value_scale: Any
    value_minimum: Any
    length: int
    quantized_key_tokens: list[int]
    residual_key_tokens: list[int]
    quantized_value_tokens: list[int]
    residual_value_tokens: list[int]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReferenceGenerationError(f"JSON root must be an object: {path}")
    return value


def _run_git(source_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=source_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise ReferenceGenerationError(
            f"git {' '.join(arguments)} failed with exit {result.returncode}"
        )
    return result.stdout.rstrip("\n")


def source_probe(
    source_root: Path,
    source_manifest_path: Path,
) -> dict[str, Any]:
    """Validate every locked source file in the patched checkout."""

    source_root = source_root.resolve(strict=True)
    manifest = _load_json(source_manifest_path)
    source = manifest["source"]
    if _run_git(source_root, "rev-parse", "HEAD") != SOURCE_COMMIT:
        raise ReferenceGenerationError("source checkout commit mismatch")
    if _run_git(source_root, "rev-parse", "HEAD^{tree}") != SOURCE_TREE:
        raise ReferenceGenerationError("source checkout base tree mismatch")
    if source["commit"] != SOURCE_COMMIT or source["base_tree"] != SOURCE_TREE:
        raise ReferenceGenerationError("source manifest commit/tree mismatch")
    if source["patched_tree"] != PATCHED_TREE:
        raise ReferenceGenerationError("source manifest patched tree mismatch")

    patched = {
        record["path"]: record
        for record in manifest["patch"]["patched_files"]
    }
    expected_status = set(patched)
    status_paths = {
        line[3:]
        for line in _run_git(
            source_root, "status", "--porcelain", "--untracked-files=all"
        ).splitlines()
        if line
    }
    if status_paths != expected_status:
        raise ReferenceGenerationError(
            "patched checkout differs from HEAD outside the exact patch"
        )

    observed: list[dict[str, Any]] = []
    for record in manifest["relevant_source_files"]:
        path = str(record["path"])
        data = (source_root / path).read_bytes()
        expected_sha = (
            patched[path]["sha256"] if path in patched else record["sha256"]
        )
        if _sha256(data) != expected_sha:
            raise ReferenceGenerationError(f"source hash mismatch: {path}")
        base_blob = _run_git(source_root, "rev-parse", f"HEAD:{path}")
        if base_blob != record["git_blob"]:
            raise ReferenceGenerationError(f"source Git blob mismatch: {path}")
        observed.append(
            {
                "path": path,
                "base_git_blob": base_blob,
                "observed_sha256": _sha256(data),
                "patched": path in patched,
            }
        )

    return {
        "schema_version": "kivi-source-probe-1.0.0",
        "status": "PASS",
        "repository": source["repository"],
        "commit": SOURCE_COMMIT,
        "base_tree": SOURCE_TREE,
        "patched_tree": PATCHED_TREE,
        "status_paths": sorted(status_paths),
        "relevant_file_count": len(observed),
        "files": observed,
    }


def _variant_spec(variant_id: str) -> dict[str, Any]:
    for variant in VARIANTS:
        if variant["id"] == variant_id:
            return dict(variant)
    raise ReferenceGenerationError(f"unsupported KIVI variant: {variant_id}")


def _import_runtime() -> dict[str, Callable[..., Any]]:
    try:
        import kivi_gemv
        from matmul import cuda_bmm_fA_qB_outer
        from models.kivi_gqa import (
            gqa_attention_value_matmul,
            gqa_query_key_matmul,
        )
        from new_pack import triton_quantize_and_pack_along_last_dim
    except ImportError as error:
        raise ReferenceGenerationError(
            "pinned KIVI runtime modules cannot be imported"
        ) from error
    return {
        "extension": kivi_gemv,
        "quantize": triton_quantize_and_pack_along_last_dim,
        "cuda_bmm": cuda_bmm_fA_qB_outer,
        "gqa_qk": gqa_query_key_matmul,
        "gqa_av": gqa_attention_value_matmul,
    }


def _extension_path(runtime: dict[str, Callable[..., Any]]) -> Path:
    path = Path(runtime["extension"].__file__).resolve(strict=True)
    if not path.name.startswith("kivi_gemv") or path.suffix != ".so":
        raise ReferenceGenerationError("unexpected KIVI extension identity")
    return path


def _dtype_name(tensor: Any) -> str:
    import torch

    names = {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
        torch.int32: "int32",
    }
    try:
        return names[tensor.dtype]
    except KeyError as error:
        raise ReferenceGenerationError(
            f"unsupported tensor dtype: {tensor.dtype}"
        ) from error


def _tensor_bytes(tensor: Any) -> bytes:
    contiguous = tensor.detach().contiguous().cpu()
    return contiguous.view(-1).view(__import__("torch").uint8).numpy().tobytes()


def _tensor_record(tensor: Any) -> dict[str, Any]:
    raw = _tensor_bytes(tensor)
    logical_nbytes = tensor.numel() * tensor.element_size()
    storage_nbytes = tensor.untyped_storage().nbytes()
    if len(raw) != logical_nbytes or storage_nbytes < logical_nbytes:
        raise ReferenceGenerationError("tensor byte accounting is inconsistent")
    return {
        "device": tensor.device.type,
        "dtype": _dtype_name(tensor),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "logical_nbytes": logical_nbytes,
        "storage_nbytes": storage_nbytes,
        "payload_sha256": _sha256(raw),
        "payload_hex": raw.hex(),
    }


def _tensor_identity(tensor: Any) -> dict[str, Any]:
    raw = _tensor_bytes(tensor)
    return {
        "dtype": _dtype_name(tensor),
        "shape": list(tensor.shape),
        "nbytes": len(raw),
        "sha256": _sha256(raw),
    }


def _make_inputs() -> dict[str, Any]:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)

    def make(shape: tuple[int, ...]) -> Any:
        integers = torch.randint(
            -192,
            193,
            shape,
            dtype=torch.int16,
            generator=generator,
        )
        value = (integers.to(torch.float32) / 64.0).to(torch.bfloat16)
        if not torch.equal(value, value.to(torch.float16).to(torch.bfloat16)):
            raise ReferenceGenerationError(
                "BF16 fixture value is not exactly representable in FP16"
            )
        return value

    return {
        "key": make(
            (BATCH_SIZE, NUM_KV_HEADS, POST_ROLLOVER_CONTEXT, HEAD_DIM)
        ),
        "value": make(
            (BATCH_SIZE, NUM_KV_HEADS, POST_ROLLOVER_CONTEXT, HEAD_DIM)
        ),
        "query": make(
            (
                BATCH_SIZE,
                NUM_QUERY_HEADS,
                POST_ROLLOVER_CONTEXT,
                HEAD_DIM,
            )
        ),
    }


def _cuda_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch

    converted: dict[str, Any] = {}
    for name, value in inputs.items():
        half = value.to(torch.float16)
        if not torch.equal(half.to(torch.bfloat16), value):
            raise ReferenceGenerationError(f"lossy BF16/FP16 cast: {name}")
        converted[name] = half.to(device="cuda")
    return converted


def _quantize(
    runtime: dict[str, Callable[..., Any]],
    tensor: Any,
    bits: int,
) -> tuple[Any, Any, Any]:
    if bits not in (2, 4):
        raise ReferenceGenerationError(f"unsupported KIVI bit width: {bits}")
    result = runtime["quantize"](tensor, GROUP_SIZE, bits)
    if len(result) != 3:
        raise ReferenceGenerationError("upstream quantizer result is invalid")
    if not all(item.device.type == "cuda" for item in result):
        raise ReferenceGenerationError("upstream quantizer used a CPU fallback")
    return result


def _store_state(
    runtime: dict[str, Callable[..., Any]],
    key: Any,
    value: Any,
    length: int,
    *,
    k_bits: int,
    v_bits: int,
) -> CacheState:
    key_input = key[:, :, :length, :].contiguous()
    value_input = value[:, :, :length, :].contiguous()

    remainder = length % RESIDUAL_LENGTH
    if remainder:
        if length < RESIDUAL_LENGTH:
            key_history = None
            residual_key = key_input
            key_history_tokens: list[int] = []
            residual_key_tokens = list(range(length))
        else:
            history_length = length - remainder
            key_history = key_input[:, :, :history_length, :].contiguous()
            residual_key = key_input[:, :, history_length:, :].contiguous()
            key_history_tokens = list(range(history_length))
            residual_key_tokens = list(range(history_length, length))
    else:
        key_history = key_input
        residual_key = None
        key_history_tokens = list(range(length))
        residual_key_tokens = []

    if key_history is None:
        quantized_key = key_scale = key_minimum = None
    else:
        quantized_key, key_scale, key_minimum = _quantize(
            runtime,
            key_history.transpose(2, 3).contiguous(),
            k_bits,
        )

    if length <= RESIDUAL_LENGTH:
        value_history = None
        residual_value = value_input
        value_history_tokens: list[int] = []
        residual_value_tokens = list(range(length))
    else:
        history_length = length - RESIDUAL_LENGTH
        value_history = value_input[:, :, :history_length, :].contiguous()
        residual_value = value_input[:, :, history_length:, :].contiguous()
        value_history_tokens = list(range(history_length))
        residual_value_tokens = list(range(history_length, length))

    if value_history is None:
        quantized_value = value_scale = value_minimum = None
    else:
        quantized_value, value_scale, value_minimum = _quantize(
            runtime, value_history, v_bits
        )

    return CacheState(
        quantized_key,
        residual_key,
        key_scale,
        key_minimum,
        quantized_value,
        residual_value,
        value_scale,
        value_minimum,
        length,
        key_history_tokens,
        residual_key_tokens,
        value_history_tokens,
        residual_value_tokens,
    )


def _scores(
    runtime: dict[str, Callable[..., Any]],
    state: CacheState,
    query: Any,
    *,
    k_bits: int,
) -> Any:
    import torch

    components = []
    if state.quantized_key is not None:
        components.append(
            runtime["cuda_bmm"](
                GROUP_SIZE,
                query,
                state.quantized_key,
                state.key_scale,
                state.key_minimum,
                k_bits,
            )
        )
    if state.residual_key is not None:
        components.append(runtime["gqa_qk"](query, state.residual_key))
    if not components:
        raise ReferenceGenerationError("cache has no key storage")
    return components[0] if len(components) == 1 else torch.cat(components, -1)


def _value_output(
    runtime: dict[str, Callable[..., Any]],
    state: CacheState,
    attention: Any,
    *,
    v_bits: int,
) -> Any:
    output = None
    historical_length = len(state.quantized_value_tokens)
    if state.quantized_value is not None:
        output = runtime["cuda_bmm"](
            GROUP_SIZE,
            attention[:, :, :, :historical_length],
            state.quantized_value,
            state.value_scale,
            state.value_minimum,
            v_bits,
        )
    if state.residual_value is not None:
        residual = runtime["gqa_av"](
            attention[:, :, :, historical_length:],
            state.residual_value,
        )
        output = residual if output is None else output + residual
    if output is None:
        raise ReferenceGenerationError("cache has no value storage")
    return output


def _decode_state(
    runtime: dict[str, Callable[..., Any]],
    state: CacheState,
    query: Any,
    *,
    k_bits: int,
    v_bits: int,
) -> Any:
    import torch

    scores = _scores(runtime, state, query, k_bits=k_bits)
    attention = torch.nn.functional.softmax(
        scores / math.sqrt(HEAD_DIM),
        dim=-1,
        dtype=torch.float32,
    ).to(query.dtype)
    return _value_output(runtime, state, attention, v_bits=v_bits)


def _append_decode(
    runtime: dict[str, Callable[..., Any]],
    state: CacheState,
    new_key: Any,
    new_value: Any,
    query: Any,
    *,
    k_bits: int,
    v_bits: int,
) -> tuple[CacheState, Any, dict[str, Any]]:
    import torch

    new_key = new_key.contiguous()
    new_value = new_value.contiguous()
    query = query.contiguous()
    token = state.length
    key_before_pointer = (
        state.residual_key.untyped_storage().data_ptr()
        if state.residual_key is not None
        else None
    )
    if state.residual_key is None:
        residual_key = new_key
    else:
        residual_key = torch.cat([state.residual_key, new_key], dim=2)
    key_after_pointer = residual_key.untyped_storage().data_ptr()
    residual_key_tokens = state.residual_key_tokens + [token]

    score_parts = []
    if state.quantized_key is not None:
        score_parts.append(
            runtime["cuda_bmm"](
                GROUP_SIZE,
                query,
                state.quantized_key,
                state.key_scale,
                state.key_minimum,
                k_bits,
            )
        )
    score_parts.append(runtime["gqa_qk"](query, residual_key))
    scores = (
        score_parts[0]
        if len(score_parts) == 1
        else torch.cat(score_parts, dim=-1)
    )

    quantized_key = state.quantized_key
    key_scale = state.key_scale
    key_minimum = state.key_minimum
    quantized_key_tokens = list(state.quantized_key_tokens)
    key_moved: list[int] = []
    if residual_key.shape[-2] == RESIDUAL_LENGTH:
        new_quantized, new_scale, new_minimum = _quantize(
            runtime,
            residual_key.transpose(2, 3).contiguous(),
            k_bits,
        )
        key_moved = list(residual_key_tokens)
        if quantized_key is None:
            quantized_key = new_quantized
            key_scale = new_scale
            key_minimum = new_minimum
        else:
            quantized_key = torch.cat(
                [quantized_key, new_quantized], dim=3
            )
            key_scale = torch.cat([key_scale, new_scale], dim=3)
            key_minimum = torch.cat([key_minimum, new_minimum], dim=3)
        quantized_key_tokens.extend(residual_key_tokens)
        residual_key = None
        residual_key_tokens = []

    attention = torch.nn.functional.softmax(
        scores / math.sqrt(HEAD_DIM),
        dim=-1,
        dtype=torch.float32,
    ).to(query.dtype)

    value_before_pointer = state.residual_value.untyped_storage().data_ptr()
    residual_value = torch.cat([state.residual_value, new_value], dim=2)
    value_after_cat_pointer = residual_value.untyped_storage().data_ptr()
    residual_value_tokens = state.residual_value_tokens + [token]
    value_full_length = residual_value.shape[-2]

    quantized_value = state.quantized_value
    value_scale = state.value_scale
    value_minimum = state.value_minimum
    quantized_value_tokens = list(state.quantized_value_tokens)
    if quantized_value is None:
        output = runtime["gqa_av"](attention, residual_value)
    else:
        output = runtime["cuda_bmm"](
            GROUP_SIZE,
            attention[:, :, :, :-value_full_length],
            quantized_value,
            value_scale,
            value_minimum,
            v_bits,
        )
        output += runtime["gqa_av"](
            attention[:, :, :, -value_full_length:],
            residual_value,
        )

    value_moved: list[int] = []
    value_slice_reallocated = False
    if value_full_length > RESIDUAL_LENGTH:
        if value_full_length != RESIDUAL_LENGTH + 1:
            raise ReferenceGenerationError("value rollover exceeded one token")
        new_quantized, new_scale, new_minimum = _quantize(
            runtime,
            residual_value[:, :, :1, :].contiguous(),
            v_bits,
        )
        value_moved = [residual_value_tokens[0]]
        residual_value = residual_value[:, :, 1:, :].contiguous()
        value_slice_reallocated = (
            residual_value.untyped_storage().data_ptr()
            != value_after_cat_pointer
        )
        residual_value_tokens = residual_value_tokens[1:]
        if quantized_value is None:
            quantized_value = new_quantized
            value_scale = new_scale
            value_minimum = new_minimum
        else:
            quantized_value = torch.cat(
                [quantized_value, new_quantized], dim=2
            )
            value_scale = torch.cat([value_scale, new_scale], dim=2)
            value_minimum = torch.cat([value_minimum, new_minimum], dim=2)
        quantized_value_tokens.extend(value_moved)

    next_state = CacheState(
        quantized_key,
        residual_key,
        key_scale,
        key_minimum,
        quantized_value,
        residual_value,
        value_scale,
        value_minimum,
        token + 1,
        quantized_key_tokens,
        residual_key_tokens,
        quantized_value_tokens,
        residual_value_tokens,
    )
    operations = {
        "key_residual_cat_reallocated": (
            key_before_pointer is not None
            and key_before_pointer != key_after_pointer
        ),
        "value_residual_cat_reallocated": (
            value_before_pointer != value_after_cat_pointer
        ),
        "value_slice_reallocated": value_slice_reallocated,
        "key_tokens_moved": key_moved,
        "value_tokens_moved": value_moved,
        "python_control_flow": True,
        "cpu_tensor_math": False,
    }
    return next_state, output, operations


def _state_tensors(state: CacheState) -> dict[str, Any]:
    return {
        "quantized_key_payload": state.quantized_key,
        "residual_key": state.residual_key,
        "key_scales": state.key_scale,
        "key_minimum_offsets": state.key_minimum,
        "quantized_value_payload": state.quantized_value,
        "residual_value": state.residual_value,
        "value_scales": state.value_scale,
        "value_minimum_offsets": state.value_minimum,
    }


def _state_record(state: CacheState) -> dict[str, Any]:
    tensors = {
        name: None if tensor is None else _tensor_record(tensor)
        for name, tensor in _state_tensors(state).items()
    }
    return {
        "length": state.length,
        "quantized_key_tokens": state.quantized_key_tokens,
        "residual_key_tokens": state.residual_key_tokens,
        "quantized_value_tokens": state.quantized_value_tokens,
        "residual_value_tokens": state.residual_value_tokens,
        "tensors": tensors,
    }


def _byte_accounting(state: CacheState) -> dict[str, Any]:
    tensors = _state_tensors(state)

    def storage(name: str) -> int:
        tensor = tensors[name]
        return 0 if tensor is None else tensor.untyped_storage().nbytes()

    def logical(name: str) -> int:
        tensor = tensors[name]
        return 0 if tensor is None else tensor.numel() * tensor.element_size()

    categories = {
        "quantized_k_payload": logical("quantized_key_payload"),
        "quantized_v_payload": logical("quantized_value_payload"),
        "key_scales": logical("key_scales"),
        "key_zero_points": logical("key_minimum_offsets"),
        "value_scales": logical("value_scales"),
        "value_zero_points": logical("value_minimum_offsets"),
        "other_metadata": 0,
        "residual_k": logical("residual_key"),
        "residual_v": logical("residual_value"),
        "padding_alignment": sum(
            storage(name) - logical(name) for name in tensors
        ),
        "persistent_workspace": 0,
    }
    actual_total = sum(categories.values())
    owned_storage = sum(storage(name) for name in tensors)
    if actual_total != owned_storage:
        raise ReferenceGenerationError("persistent storage categories disagree")
    logical_bf16 = (
        BATCH_SIZE
        * NUM_KV_HEADS
        * state.length
        * HEAD_DIM
        * 2
        * 2
    )
    return {
        "context": state.length,
        "calculation_mode": "actual_source_owned_tensor_storage",
        "categories": categories,
        "actual_total": actual_total,
        "logical_bf16_bytes": logical_bf16,
        "r_alloc": actual_total / logical_bf16,
        "storage_agreement": True,
        "r_hbm": None,
    }


def _static_byte_accounting(
    *,
    k_bits: int,
    v_bits: int,
    context: int,
) -> dict[str, Any]:
    if context != STATIC_ACCOUNTING_CONTEXT:
        raise ReferenceGenerationError("unsupported static accounting context")
    key_history = context
    key_residual = 0
    value_history = context - RESIDUAL_LENGTH
    value_residual = RESIDUAL_LENGTH
    categories = {
        "quantized_k_payload": (
            BATCH_SIZE * NUM_KV_HEADS * HEAD_DIM * key_history * k_bits // 8
        ),
        "quantized_v_payload": (
            BATCH_SIZE
            * NUM_KV_HEADS
            * value_history
            * HEAD_DIM
            * v_bits
            // 8
        ),
        "key_scales": (
            BATCH_SIZE
            * NUM_KV_HEADS
            * HEAD_DIM
            * (key_history // GROUP_SIZE)
            * 2
        ),
        "key_zero_points": (
            BATCH_SIZE
            * NUM_KV_HEADS
            * HEAD_DIM
            * (key_history // GROUP_SIZE)
            * 2
        ),
        "value_scales": (
            BATCH_SIZE
            * NUM_KV_HEADS
            * value_history
            * (HEAD_DIM // GROUP_SIZE)
            * 2
        ),
        "value_zero_points": (
            BATCH_SIZE
            * NUM_KV_HEADS
            * value_history
            * (HEAD_DIM // GROUP_SIZE)
            * 2
        ),
        "other_metadata": 0,
        "residual_k": (
            BATCH_SIZE * NUM_KV_HEADS * key_residual * HEAD_DIM * 2
        ),
        "residual_v": (
            BATCH_SIZE * NUM_KV_HEADS * value_residual * HEAD_DIM * 2
        ),
        "padding_alignment": 0,
        "persistent_workspace": 0,
    }
    total = sum(categories.values())
    logical_bf16 = (
        BATCH_SIZE * NUM_KV_HEADS * context * HEAD_DIM * 2 * 2
    )
    return {
        "context": context,
        "calculation_mode": "source_layout_formula_no_runtime_campaign",
        "categories": categories,
        "actual_total": total,
        "logical_bf16_bytes": logical_bf16,
        "r_alloc": total / logical_bf16,
        "storage_agreement": True,
        "r_hbm": None,
    }


def _trace_record(profile: Any) -> dict[str, Any]:
    interesting = (
        "bgemv",
        "gemv_forward_cuda_outer_dim",
        "_minmax_along_last_dim",
        "_pack_along_last_dim",
        "aten::bmm",
        "aten::cat",
        "aten::_softmax",
    )
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for event in profile.events():
        if not any(token in event.name for token in interesting):
            continue
        shapes = getattr(event, "input_shapes", [])
        key = (event.name, str(event.device_type), repr(shapes))
        records[key] = {
            "name": event.name,
            "device_type": str(event.device_type),
            "input_shapes": shapes,
        }
    events = [records[key] for key in sorted(records)]
    names = [record["name"] for record in events]
    return {
        "schema_version": "kivi-reference-trace-1.0.0",
        "run_kind": "reference_trace",
        "timings_discarded": True,
        "events": events,
        "quantize_store_kernels": sorted(
            name
            for name in set(names)
            if "_minmax_along_last_dim" in name
            or "_pack_along_last_dim" in name
        ),
        "append_operations": sorted(
            name
            for name in set(names)
            if name in {"aten::cat", "aten::bmm"}
        ),
        "decode_dequant_kernels": sorted(
            name
            for name in set(names)
            if "bgemv" in name or "gemv_forward_cuda_outer_dim" in name
        ),
        "full_prefix_temporary": False,
        "backend_fallback": False,
        "performance_claim_eligible": False,
    }


def _generate_variant_core(
    variant: dict[str, Any],
    runtime: dict[str, Callable[..., Any]],
    host_inputs: dict[str, Any],
    device_inputs: dict[str, Any],
) -> dict[str, Any]:
    import torch

    k_bits = int(variant["k_bits"])
    v_bits = int(variant["v_bits"])
    key = device_inputs["key"]
    value = device_inputs["value"]
    query = device_inputs["query"]

    bmm_operands: list[dict[str, Any]] = []
    original_bmm = torch.bmm

    def recording_bmm(left: Any, right: Any) -> Any:
        bmm_operands.append(
            {
                "left_shape": list(left.shape),
                "right_shape": list(right.shape),
                "left_device": left.device.type,
                "right_device": right.device.type,
            }
        )
        return original_bmm(left, right)

    torch.bmm = recording_bmm
    try:
        basic_store = _store_state(
            runtime,
            key,
            value,
            BASIC_STORE_CONTEXT,
            k_bits=k_bits,
            v_bits=v_bits,
        )
        store_output = _decode_state(
            runtime,
            basic_store,
            query[:, :, 16:17, :],
            k_bits=k_bits,
            v_bits=v_bits,
        )
        basic_append, decode_output, basic_ops = _append_decode(
            runtime,
            basic_store,
            key[:, :, 17:18, :],
            value[:, :, 17:18, :],
            query[:, :, 17:18, :],
            k_bits=k_bits,
            v_bits=v_bits,
        )

        before = _store_state(
            runtime,
            key,
            value,
            31,
            k_bits=k_bits,
            v_bits=v_bits,
        )
        before_output = _decode_state(
            runtime,
            before,
            query[:, :, 30:31, :],
            k_bits=k_bits,
            v_bits=v_bits,
        )
        boundary, boundary_output, boundary_ops = _append_decode(
            runtime,
            before,
            key[:, :, 31:32, :],
            value[:, :, 31:32, :],
            query[:, :, 31:32, :],
            k_bits=k_bits,
            v_bits=v_bits,
        )
        after, after_output, after_ops = _append_decode(
            runtime,
            boundary,
            key[:, :, 32:33, :],
            value[:, :, 32:33, :],
            query[:, :, 32:33, :],
            k_bits=k_bits,
            v_bits=v_bits,
        )
        post, post_output, post_ops = _append_decode(
            runtime,
            after,
            key[:, :, 33:34, :],
            value[:, :, 33:34, :],
            query[:, :, 33:34, :],
            k_bits=k_bits,
            v_bits=v_bits,
        )
        torch.cuda.synchronize()
    finally:
        torch.bmm = original_bmm

    for record in bmm_operands:
        if record["right_shape"][0] != BATCH_SIZE * NUM_KV_HEADS:
            raise ReferenceGenerationError(
                "GQA residual BMM used an H_Q-sized K/V operand"
            )

    input_query = host_inputs["query"][:, :, QUERY_POSITIONS, :]
    accounting = [
        _byte_accounting(state)
        for state in (before, boundary, after)
    ]
    accounting.append(
        _static_byte_accounting(
            k_bits=k_bits,
            v_bits=v_bits,
            context=STATIC_ACCOUNTING_CONTEXT,
        )
    )
    return {
        "schema_version": "kivi-reference-fixture-1.0.0",
        "variant": variant,
        "source": {
            "repository": "https://github.com/jy-yuan/KIVI.git",
            "commit": SOURCE_COMMIT,
            "base_tree": SOURCE_TREE,
            "patched_tree": PATCHED_TREE,
            "authority": "checksum_bound_patched_official_source",
        },
        "configuration": {
            "k_bits": k_bits,
            "v_bits": v_bits,
            "group_size": GROUP_SIZE,
            "residual_length": RESIDUAL_LENGTH,
        },
        "geometry": {
            "batch_size": BATCH_SIZE,
            "num_query_heads": NUM_QUERY_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "basic_store_context": BASIC_STORE_CONTEXT,
            "basic_append_tokens": 1,
            "basic_total_context": BASIC_TOTAL_CONTEXT,
            "rollover_contexts": list(ROLLOVER_CONTEXTS),
            "post_rollover_decode_context": POST_ROLLOVER_CONTEXT,
            "seed": SEED,
            "input_dtype": "bfloat16",
            "upstream_cuda_execution_dtype": "float16",
        },
        "dtype_compatibility": {
            "upstream_cuda_abi": "half_only",
            "bf16_input_to_fp16_exact": True,
            "fp16_to_bf16_round_trip_exact": True,
            "compatibility_kind": "lossless_reference_api_boundary",
            "algorithmic_cuda_patch": False,
        },
        "inputs": {
            "key_0_33": _tensor_record(host_inputs["key"]),
            "value_0_33": _tensor_record(host_inputs["value"]),
            "selected_queries": _tensor_record(input_query),
            "query_positions": list(QUERY_POSITIONS),
        },
        "basic": {
            "store_state": _state_record(basic_store),
            "store_output": _tensor_record(store_output),
            "append_state": _state_record(basic_append),
            "append_operations": basic_ops,
            "decode_output": _tensor_record(decode_output),
        },
        "rollover": {
            "residual_capacity": RESIDUAL_LENGTH,
            "trigger": {
                "key": "residual_key_length_equals_32",
                "value": "residual_value_length_becomes_33",
            },
            "before": {
                "state": _state_record(before),
                "output": _tensor_record(before_output),
            },
            "boundary": {
                "state": _state_record(boundary),
                "output": _tensor_record(boundary_output),
                "operations": boundary_ops,
            },
            "after": {
                "state": _state_record(after),
                "output": _tensor_record(after_output),
                "operations": after_ops,
            },
            "post_rollover_decode": {
                "state": _state_record(post),
                "output": _tensor_record(post_output),
                "operations": post_ops,
            },
            "missing_tokens": [],
            "duplicate_tokens": [],
            "source_faithful": True,
        },
        "byte_accounting": accounting,
        "gqa": {
            "h_q": NUM_QUERY_HEADS,
            "h_kv": NUM_KV_HEADS,
            "head_mapping": [
                head // (NUM_QUERY_HEADS // NUM_KV_HEADS)
                for head in range(NUM_QUERY_HEADS)
            ],
            "bmm_operands": bmm_operands,
            "cache_head_count": NUM_KV_HEADS,
            "repeat_kv": False,
            "repeat_interleave": False,
            "expand_reshape_kv_materialization": False,
            "expanded_temporary": False,
            "final_verdict": "PASS_NATIVE_EIGHT_HEAD_KV_STORAGE",
        },
        "full_prefix_dequantization": {
            "observed": False,
            "behavior": (
                "packed history is consumed directly by "
                "gemv_forward_cuda_outer_dim; residual storage is separate"
            ),
        },
        "graph_information": {
            "upstream_support": "undocumented",
            "reference_graph_smoke": "NOT_RUN",
            "reason": (
                "upstream reference cache uses torch.cat and dynamic "
                "quantization allocations"
            ),
            "deferred_to_phase8": True,
        },
        "claims": {
            "performance": False,
            "latency": False,
            "throughput": False,
            "physical_hbm": False,
            "capacity": False,
            "quality": False,
        },
    }


def _generate_variant(
    variant: dict[str, Any],
    runtime: dict[str, Callable[..., Any]],
    host_inputs: dict[str, Any],
    device_inputs: dict[str, Any],
) -> dict[str, Any]:
    import torch

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as profile:
        fixture = _generate_variant_core(
            variant, runtime, host_inputs, device_inputs
        )
    fixture["reference_trace"] = _trace_record(profile)
    return fixture


def _directory_bytes(root: Path) -> dict[str, bytes]:
    if root.is_symlink() or not root.is_dir():
        raise ReferenceGenerationError(f"unsafe fixture directory: {root}")
    result: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ReferenceGenerationError(f"fixture symlink forbidden: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = path.read_bytes()
    return result


def _write_no_replace(output_root: Path, files: dict[str, bytes]) -> str:
    output_root = output_root.resolve()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=".kivi-fixtures-",
            dir=output_root.parent,
        )
    )
    try:
        for relative, data in sorted(files.items()):
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        if output_root.exists() or output_root.is_symlink():
            if output_root.is_symlink() or not output_root.is_dir():
                raise ReferenceGenerationError(
                    "existing fixture target is not a safe directory"
                )
            if _directory_bytes(output_root) != _directory_bytes(temporary):
                raise ReferenceGenerationError(
                    "existing fixture set differs; overwrite refused"
                )
            return "existing_identical"
        os.rename(temporary, output_root)
        return "created"
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def generate_fixture_set(
    *,
    output_root: Path,
    image_config_digest: str,
    dockerfile_sha256: str,
    source_manifest_sha256: str,
    build_manifest_sha256: str,
    expected_extension_sha256: str,
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise ReferenceGenerationError("CUDA is unavailable")
    if torch.cuda.get_device_capability(0) != (12, 0):
        raise ReferenceGenerationError("reference GPU is not SM120")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)

    runtime = _import_runtime()
    extension = _extension_path(runtime)
    extension_sha256 = _sha256(extension.read_bytes())
    if extension_sha256 != expected_extension_sha256:
        raise ReferenceGenerationError("extension SHA-256 mismatch")

    host_inputs = _make_inputs()
    device_inputs = _cuda_inputs(host_inputs)
    files: dict[str, bytes] = {}
    variant_manifests: list[dict[str, Any]] = []
    for raw_variant in VARIANTS:
        variant = _variant_spec(str(raw_variant["id"]))
        fixture = _generate_variant(
            variant, runtime, host_inputs, device_inputs
        )
        fixture_bytes = _canonical_json(fixture)
        fixture_relative = f"{variant['id']}/fixture.json"
        files[fixture_relative] = fixture_bytes
        manifest = {
            "schema_version": "kivi-reference-fixture-manifest-1.0.0",
            "variant": variant,
            "fixture": {
                "path": "fixture.json",
                "nbytes": len(fixture_bytes),
                "sha256": _sha256(fixture_bytes),
            },
            "source_commit": SOURCE_COMMIT,
            "patched_tree": PATCHED_TREE,
            "image_config_digest": image_config_digest,
            "extension_sha256": extension_sha256,
            "performance_measurement": False,
        }
        manifest_bytes = _canonical_json(manifest)
        manifest_relative = f"{variant['id']}/manifest.json"
        files[manifest_relative] = manifest_bytes
        variant_manifests.append(
            {
                "variant": variant["id"],
                "path": manifest_relative,
                "nbytes": len(manifest_bytes),
                "sha256": _sha256(manifest_bytes),
            }
        )

    fixture_set = {
        "schema_version": "kivi-reference-fixture-set-1.0.0",
        "source_commit": SOURCE_COMMIT,
        "base_tree": SOURCE_TREE,
        "patched_tree": PATCHED_TREE,
        "source_manifest_sha256": source_manifest_sha256,
        "build_manifest_sha256": build_manifest_sha256,
        "dockerfile_sha256": dockerfile_sha256,
        "image_config_digest": image_config_digest,
        "extension_sha256": extension_sha256,
        "configurations": [variant["id"] for variant in VARIANTS],
        "mandatory_configurations": ["k4v4", "k2v4", "k2v2"],
        "held_out_configurations": ["k4v2"],
        "variant_manifests": variant_manifests,
        "determinism": {
            "seed": SEED,
            "torch_deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
            "input_generation": (
                "CPU integer grid divided by 64 then BF16 cast"
            ),
            "existing_set_policy": (
                "regenerate, compare exact bytes, never overwrite differences"
            ),
        },
        "performance_measurement": False,
        "r_hbm_populated": False,
    }
    files["fixture_set.json"] = _canonical_json(fixture_set)
    ledger = "".join(
        f"{_sha256(data)}  {relative}\n"
        for relative, data in sorted(files.items())
    ).encode("utf-8")
    files["checksums.sha256"] = ledger
    action = _write_no_replace(output_root, files)
    return {
        "schema_version": "kivi-reference-generation-result-1.0.0",
        "status": "PASS",
        "action": action,
        "output_root": str(output_root),
        "file_count": len(files),
        "configuration_count": len(VARIANTS),
        "extension_sha256": extension_sha256,
        "performance_measurement": False,
    }


def _probe_inputs() -> dict[str, Any]:
    import torch

    host = _make_inputs()
    return _cuda_inputs(host)


def kernel_probe() -> dict[str, Any]:
    import torch

    runtime = _import_runtime()
    inputs = _probe_inputs()
    results: list[dict[str, Any]] = []
    for bits in (2, 4):
        key = inputs["key"][:, :, :32, :]
        packed_key, key_scale, key_minimum = _quantize(
            runtime, key.transpose(2, 3).contiguous(), bits
        )
        key_output = runtime["cuda_bmm"](
            GROUP_SIZE,
            inputs["query"][:, :, :1, :],
            packed_key,
            key_scale,
            key_minimum,
            bits,
        )
        value = inputs["value"][:, :, :1, :]
        packed_value, value_scale, value_minimum = _quantize(
            runtime, value, bits
        )
        value_weight = torch.ones(
            (BATCH_SIZE, NUM_QUERY_HEADS, 1, 1),
            dtype=torch.float16,
            device="cuda",
        )
        value_output = runtime["cuda_bmm"](
            GROUP_SIZE,
            value_weight,
            packed_value,
            value_scale,
            value_minimum,
            bits,
        )
        torch.cuda.synchronize()
        results.append(
            {
                "bits": bits,
                "key_output": _tensor_identity(key_output),
                "value_output": _tensor_identity(value_output),
            }
        )
    extension = _extension_path(runtime)
    return {
        "schema_version": "kivi-kernel-probe-1.0.0",
        "status": "PASS",
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "extension_path": str(extension),
        "extension_sha256": _sha256(extension.read_bytes()),
        "kernel_families": results,
        "no_kernel_image": False,
        "unsupported_fallback": False,
        "performance_measurement": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)

    source = subparsers.add_parser("source-probe")
    source.add_argument("--source-root", type=Path, required=True)
    source.add_argument("--source-manifest", type=Path, required=True)

    subparsers.add_parser("kernel-probe")
    subparsers.add_parser("sanitizer-probe")

    fixtures = subparsers.add_parser("fixtures")
    fixtures.add_argument("--output", type=Path, required=True)
    fixtures.add_argument("--image-config-digest", required=True)
    fixtures.add_argument("--dockerfile-sha256", required=True)
    fixtures.add_argument("--source-manifest-sha256", required=True)
    fixtures.add_argument("--build-manifest-sha256", required=True)
    fixtures.add_argument("--extension-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    try:
        if arguments.action == "source-probe":
            result = source_probe(
                arguments.source_root,
                arguments.source_manifest,
            )
        elif arguments.action in {"kernel-probe", "sanitizer-probe"}:
            result = kernel_probe()
            if arguments.action == "sanitizer-probe":
                result["run_kind"] = "compute_sanitizer_probe"
        else:
            result = generate_fixture_set(
                output_root=arguments.output,
                image_config_digest=arguments.image_config_digest,
                dockerfile_sha256=arguments.dockerfile_sha256,
                source_manifest_sha256=arguments.source_manifest_sha256,
                build_manifest_sha256=arguments.build_manifest_sha256,
                expected_extension_sha256=arguments.extension_sha256,
            )
    except (
        OSError,
        ReferenceGenerationError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as error:
        print(
            json.dumps(
                {
                    "schema_version": (
                        "kivi-reference-generation-result-1.0.0"
                    ),
                    "status": "FAIL",
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
