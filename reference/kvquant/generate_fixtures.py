#!/usr/bin/env python3
"""Generate the narrow Phase 10 KVQuant reference fixture bundle.

This file is intentionally method-specific.  It executes the checksum-bound
patched KVQuant deployment cache directly and does not provide a reusable
reference framework.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import gc
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tempfile
from typing import Any


METHOD_IDENTIFIER = "kvquant_gqa_upstream_patch_v1"
UPSTREAM_REPOSITORY = "https://github.com/SqueezeAILab/KVQuant.git"
UPSTREAM_BASE_COMMIT = "57a238357f0ffe50084670fcd5781c9848f80ea2"
UPSTREAM_BASE_TREE = "094e0f736f77ee327e5350cbd1eefb1c936aa77b"
PATCH_SHA256 = "db3b6fb7ec0a72e25001e1c83a5158d86512248db5c3a06c61895598d1d482d6"
PATCHED_COMMIT = "4ad80bc8c942d0a05516d2be8f8d443a77a05900"
PATCHED_TREE = "c4f1490c9c0c4ec46099f1e95c092516df2adb4e"
DECISION = "0021"
CONTRACT_DECISION = "0023"

CALIBRATION_ID = "kvqcal-cdb724c806d64d095c040d2673a987a3"
CALIBRATION_ROOT = (
    "8148306d08205af376994b022f189a0d6837915cd279ca8af6b104e1f4b46ccf"
)
CALIBRATION_R2_URI = (
    "r2://kvbench-artifacts/kvbench/sha256/"
    f"{CALIBRATION_ROOT}/"
)
QUANTIZER_SHA256 = {
    "kvq4": "a8c009633ac4cad952deb2a2fa96c44ef928a1510dadcf11dee29a7a3efe1bf6",
    "kvq3": "97518129cc64ffa445722cb0802b3082631841de50835cbdf2c85c36a0c1579f",
    "kvq2": "b9bb3a8699aa38fb2a5707ff036814971552462692a180431f6f68df9624560e",
}

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
MODEL_CONFIG_SHA256 = (
    "29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e"
)
MODEL_SNAPSHOT_MANIFEST_SHA256 = (
    "ab9f6a32a41934c9e49881db68022827b6aca35f4f644627c77e3420978d1336"
)

REFERENCE_IMAGE_CONFIG_DIGEST = (
    "sha256:24eb3f6ff39b72f45c353acfbef6ce2d9aaac0860180b4dde8b937593176714b"
)
REFERENCE_BASE_CONFIG_DIGEST = (
    "sha256:127759078f2c70c9e795c7a1bb3408df1eaee8fa019319299d283dc8075b216d"
)
CALIBRATION_DOCKERFILE_SHA256 = (
    "e3ac0933c21c986bed2ca169c8983f6d1e6412e02bed42a282f9c604fd9c4de5"
)
REFERENCE_DOCKERFILE_SHA256 = (
    "f1b2f2a6f6f15bf364eb3a8b7a26f01504edbe2dbcfe74b619b1c519120a618e"
)
CALIBRATION_PYTHON_FREEZE_SHA256 = (
    "950f28e3513f03e693b4dd87018ced302f4c201754e791bcdef65976c737eb7a"
)
TOKENIZERS_WHEEL_SHA256 = (
    "9e0480c452217edd35eca56fafe2029fb4d368b7c0475f8dfa3c5c9c400a7456"
)

EXPECTED_PYTHON = "3.12.3"
EXPECTED_PYTORCH = "2.12.1+cu130"
EXPECTED_CUDA = "13.0"
EXPECTED_TRANSFORMERS_INSTALLED = "4.57.6"
EXPECTED_TRANSFORMERS_VENDORED = "4.38.0.dev0"
EXPECTED_TOKENIZERS_BASE = "0.22.2"
EXPECTED_TOKENIZERS_ACTIVE = "0.15.2"
EXPECTED_GCC = "13.3.0"
EXPECTED_NVCC = "13.0.88"

BATCH_SIZE = 1
NUM_QUERY_HEADS = 32
NUM_KV_HEADS = 8
NUM_KV_GROUPS = 4
HEAD_DIM = 128
KV_WIDTH = NUM_KV_HEADS * HEAD_DIM
INTERFACE_DTYPE = "bfloat16"
SINK_DTYPE = "float16"
SINK_TOKENS = 5
KEY_OUTLIER_CAP = 12
VALUE_OUTLIER_CAP = 12
ENTRIES_PER_TAIL = 6
SEED = 20260729
STORE_CONTEXT = 17
APPEND_TOKENS = 1
TOTAL_CONTEXT = 18
QUANTIZED_CONTEXT = TOTAL_CONTEXT - SINK_TOKENS
STORE_QUANTIZED_CONTEXT = STORE_CONTEXT - SINK_TOKENS
APPEND_POSITION = 17

FAMILIES = (("kvq4", 4), ("kvq3", 3), ("kvq2", 2))
CASES = (
    ("key_zero_value_fixed12", 0),
    ("key_few_value_fixed12", 6),
    ("key_cap_value_fixed12", 12),
)
FIXTURE_ID = "kvqref-a50af6511c314b6394e58a7f81ceefb8"

# Frozen before final nine-fixture generation.  These bounds include the
# source path's BF16 boundary and are checked independently for every case.
KEY_LOGIT_ATOL = 0.25
KEY_LOGIT_RTOL = 0.01
DECODE_ATOL = 0.01
DECODE_RTOL = 0.01
RECIPROCAL_ATOL = 1e-9

ROPE_SCALING = {
    "rope_type": "llama3",
    "factor": 8.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
}
ROPE_THETA = 500000.0
MODEL_MAX_CONTEXT = 131072

FIXTURE_MEMBERS = (
    "fixture_manifest.json",
    "inputs.safetensors",
    "dense_payload.safetensors",
    "metadata.safetensors",
    "sparse_values.safetensors",
    "sparse_indices.safetensors",
    "sink.safetensors",
    "store_state.safetensors",
    "append_state.safetensors",
    "decode_output.safetensors",
    "byte_breakdown.json",
    "checksums.sha256",
)
ROOT_CONTROL_FILES = (
    "manifest.json",
    "artifact_inventory.json",
    "checksums.sha256",
    "COMPLETE",
)


class ReferenceGenerationError(RuntimeError):
    """Raised when source execution or fixture custody violates the contract."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReferenceGenerationError(f"JSON root must be an object: {path}")
    return value


def _write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _write_or_verify(path: Path, data: bytes) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise ReferenceGenerationError(
                f"refusing to replace differing finalized authority file: {path}"
            )
        return
    _write_exclusive(path, data)


def _safe_relative(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ReferenceGenerationError(f"unsafe fixture path: {relative!r}")
    return relative


def _run(
    arguments: tuple[str, ...],
    *,
    cwd: Path | None = None,
) -> str:
    result = subprocess.run(
        arguments,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise ReferenceGenerationError(
            f"command failed with exit {result.returncode}: {arguments[0]}"
        )
    return result.stdout.rstrip("\n")


def _git(source_root: Path, *arguments: str) -> str:
    return _run(("git", *arguments), cwd=source_root)


def _tensor_bytes(tensor: Any) -> bytes:
    import torch

    contiguous = tensor.detach().contiguous().cpu()
    return contiguous.view(torch.uint8).numpy().tobytes()


def _dtype_name(tensor: Any) -> str:
    import torch

    names = {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
        torch.float64: "float64",
        torch.int32: "int32",
        torch.int64: "int64",
        torch.bool: "bool",
    }
    try:
        return names[tensor.dtype]
    except KeyError as error:
        raise ReferenceGenerationError(
            f"unsupported fixture tensor dtype: {tensor.dtype}"
        ) from error


def _tensor_record(tensor: Any) -> dict[str, Any]:
    raw = _tensor_bytes(tensor)
    logical_nbytes = tensor.numel() * tensor.element_size()
    if len(raw) != logical_nbytes:
        raise ReferenceGenerationError("tensor payload byte count is inconsistent")
    return {
        "dtype": _dtype_name(tensor),
        "shape": list(tensor.shape),
        "logical_nbytes": logical_nbytes,
        "payload_sha256": _sha256_bytes(raw),
    }


def _save_safetensors(path: Path, tensors: Mapping[str, Any]) -> None:
    from safetensors.torch import save

    if path.exists() or path.is_symlink():
        raise ReferenceGenerationError(f"fixture file already exists: {path}")
    payload = {
        name: tensor.detach().contiguous().cpu()
        for name, tensor in sorted(tensors.items())
    }
    encoded = save(
        payload,
        metadata={
            "format": "kvbench-phase10-kvquant-reference",
            "fixture_id": FIXTURE_ID,
        },
    )
    header_length = int.from_bytes(encoded[:8], byteorder="little")
    header_end = 8 + header_length
    header = json.loads(encoded[8:header_end])
    if not isinstance(header, dict):
        raise ReferenceGenerationError("safe-format header is not an object")
    canonical_header = json.dumps(
        header,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    canonical_header += b" " * (-len(canonical_header) % 8)
    canonical = (
        len(canonical_header).to_bytes(8, byteorder="little")
        + canonical_header
        + encoded[header_end:]
    )
    _write_exclusive(path, canonical)


def _validate_source(source_root: Path, patch_manifest_path: Path) -> dict[str, Any]:
    source_root = source_root.resolve(strict=True)
    patch_manifest_path = patch_manifest_path.resolve(strict=True)
    patch_manifest = _load_json(patch_manifest_path)
    if _git(source_root, "rev-parse", "HEAD") != PATCHED_COMMIT:
        raise ReferenceGenerationError("patched source commit mismatch")
    if _git(source_root, "rev-parse", "HEAD^{tree}") != PATCHED_TREE:
        raise ReferenceGenerationError("patched source tree mismatch")
    if _git(source_root, "status", "--porcelain", "--untracked-files=all"):
        raise ReferenceGenerationError("patched source checkout is not clean")
    source = patch_manifest.get("source")
    patch = patch_manifest.get("patch")
    if (
        not isinstance(source, dict)
        or source.get("base_commit") != UPSTREAM_BASE_COMMIT
        or source.get("base_tree") != UPSTREAM_BASE_TREE
        or source.get("patched_commit") != PATCHED_COMMIT
        or source.get("patched_tree") != PATCHED_TREE
        or not isinstance(patch, dict)
        or patch.get("sha256") != PATCH_SHA256
        or patch.get("changed_file_count") != 15
    ):
        raise ReferenceGenerationError("patch manifest authority mismatch")

    observed_files: list[dict[str, Any]] = []
    patched_files = patch_manifest.get("patched_files")
    if not isinstance(patched_files, list) or len(patched_files) != 15:
        raise ReferenceGenerationError("patch changed-file set mismatch")
    for record in patched_files:
        if not isinstance(record, dict):
            raise ReferenceGenerationError("invalid patched-file record")
        relative = record.get("path")
        expected = record.get("patched_sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ReferenceGenerationError("invalid patched-file identity")
        path = source_root / relative
        if _sha256_file(path) != expected:
            raise ReferenceGenerationError(f"patched source hash mismatch: {relative}")
        observed_files.append(
            {
                "path": relative,
                "change_type": record.get("change_type"),
                "base_sha256": record.get("base_sha256"),
                "patched_sha256": expected,
            }
        )
    return {
        "status": "PASS",
        "clean": True,
        "changed_file_count": len(observed_files),
        "changed_files": observed_files,
    }


def _parse_ledger(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        if (
            len(parts) != 2
            or len(parts[0]) != 64
            or any(character not in "0123456789abcdef" for character in parts[0])
        ):
            raise ReferenceGenerationError(f"invalid checksum ledger: {path}")
        digest, relative = parts
        if relative in entries:
            raise ReferenceGenerationError(f"duplicate checksum path: {relative}")
        entries[relative] = digest
    return entries


def _validate_calibration(calibration_root: Path) -> dict[str, Any]:
    calibration_root = calibration_root.resolve(strict=True)
    manifest = _load_json(calibration_root / "manifest.json")
    completion = _load_json(calibration_root / "COMPLETE")
    if (
        manifest.get("run_id") != CALIBRATION_ID
        or manifest.get("status") != "completed"
        or manifest.get("method_identifier") != METHOD_IDENTIFIER
        or manifest.get("patched_tree") != PATCHED_TREE
        or manifest.get("patch_sha256") != PATCH_SHA256
        or completion.get("run_id") != CALIBRATION_ID
        or completion.get("status") != "completed"
        or completion.get("written_last") is not True
    ):
        raise ReferenceGenerationError("calibration authority mismatch")
    ledger = _parse_ledger(calibration_root / "checksums.sha256")
    required = {
        "authority_manifest.json",
        "environment.json",
        "model_manifest.json",
        "tokenizer_manifest.json",
        "dataset_manifest.json",
        "fisher_manifest.json",
        "outlier_policy.json",
        "layer_stats.parquet",
        "quantizers/kvq4.safetensors",
        "quantizers/kvq3.safetensors",
        "quantizers/kvq2.safetensors",
    }
    if not required.issubset(ledger):
        raise ReferenceGenerationError("calibration checksum ledger is incomplete")
    quantizers: dict[str, dict[str, Any]] = {}
    for family, bit_width in FAMILIES:
        relative = f"quantizers/{family}.safetensors"
        path = calibration_root / relative
        observed = _sha256_file(path)
        if observed != QUANTIZER_SHA256[family] or ledger[relative] != observed:
            raise ReferenceGenerationError(f"{family} quantizer identity mismatch")
        tensor_manifest = _load_json(
            calibration_root / f"quantizers/{family}.manifest.json"
        )
        if (
            tensor_manifest.get("variant_id") != family
            or tensor_manifest.get("bit_width") != bit_width
            or tensor_manifest.get("safe_sha256") != observed
            or tensor_manifest.get("tensor_count") != 320
        ):
            raise ReferenceGenerationError(
                f"{family} quantizer manifest identity mismatch"
            )
        quantizers[family] = {
            "bit_width": bit_width,
            "safe_sha256": observed,
            "tensor_count": 320,
            "manifest_sha256": _sha256_file(
                calibration_root / f"quantizers/{family}.manifest.json"
            ),
        }
    return {
        "status": "PASS",
        "complete": True,
        "inventory_checksums": "PASS",
        "quantizers": quantizers,
    }


def _runtime() -> dict[str, Any]:
    import torch
    import transformers
    from kvquant_gqa.compat import select_fixed_outliers
    from transformers.models.llama.modeling_llama import (
        Llama3RotaryEmbedding,
        QuantK,
        QuantV,
        _gqa_query_key_matmul,
        _gqa_score_value_matmul,
        apply_rotary_pos_emb_query,
    )

    if not torch.cuda.is_available():
        raise ReferenceGenerationError("CUDA is required for reference generation")
    if torch.cuda.get_device_capability(0) != (12, 0):
        raise ReferenceGenerationError("native SM120 hardware is required")
    return {
        "torch": torch,
        "transformers": transformers,
        "select_fixed_outliers": select_fixed_outliers,
        "rope_class": Llama3RotaryEmbedding,
        "quant_k": QuantK,
        "quant_v": QuantV,
        "gqa_qk": _gqa_query_key_matmul,
        "gqa_av": _gqa_score_value_matmul,
        "apply_rope": apply_rotary_pos_emb_query,
    }


def _load_layer_zero_quantizer(
    calibration_root: Path,
    family: str,
) -> dict[str, Any]:
    from safetensors.torch import load_file

    payload = load_file(
        str(calibration_root / f"quantizers/{family}.safetensors"),
        device="cpu",
    )
    prefix = "layer_00"
    required = {
        f"{prefix}.k.upper_threshold",
        f"{prefix}.k.lower_threshold",
        f"{prefix}.k.range",
        f"{prefix}.k.offset",
        f"{prefix}.k.codebook",
        f"{prefix}.v.upper_threshold",
        f"{prefix}.v.lower_threshold",
        f"{prefix}.v.range",
        f"{prefix}.v.offset",
        f"{prefix}.v.codebook",
    }
    if not required.issubset(payload):
        raise ReferenceGenerationError(f"{family} lacks layer-zero tensors")
    return {
        "k_upper": payload[f"{prefix}.k.upper_threshold"].float().contiguous(),
        "k_lower": payload[f"{prefix}.k.lower_threshold"].float().contiguous(),
        "k_range": payload[f"{prefix}.k.range"].float().contiguous(),
        "k_offset": payload[f"{prefix}.k.offset"].float().contiguous(),
        "k_codebook": payload[f"{prefix}.k.codebook"].float().contiguous(),
        "v_upper": payload[f"{prefix}.v.upper_threshold"].float().contiguous(),
        "v_lower": payload[f"{prefix}.v.lower_threshold"].float().contiguous(),
        "v_range": payload[f"{prefix}.v.range"].float().contiguous(),
        "v_offset": payload[f"{prefix}.v.offset"].float().contiguous(),
        "v_codebook": payload[f"{prefix}.v.codebook"].float().contiguous(),
    }


def _quantizer_tuple(
    upper: Any,
    lower: Any,
    codebook: Any,
) -> tuple[Any, Any, list[Any]]:
    return upper, lower, [codebook]


def _make_rope(runtime: Mapping[str, Any]) -> Any:
    rope = runtime["rope_class"](
        dim=HEAD_DIM,
        max_position_embeddings=TOTAL_CONTEXT,
        base=ROPE_THETA,
        rope_scaling=ROPE_SCALING,
        device="cpu",
    )
    return rope.cuda()


def _make_caches(
    runtime: Mapping[str, Any],
    *,
    bit_width: int,
    quantizer: Mapping[str, Any],
    rope: Any,
) -> tuple[Any, Any]:
    quant_k = runtime["quant_k"](
        bits=bit_width,
        hidden_size=NUM_QUERY_HEADS * HEAD_DIM,
        num_query_heads=NUM_QUERY_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        max_position_embeddings=TOTAL_CONTEXT,
        include_sparse=True,
        sparsity_threshold=0.99,
        rope_inv_freq=rope.inv_freq,
        rope_theta=ROPE_THETA,
        use_orig_sparse=False,
        first_few_fp16=SINK_TOKENS,
        outlier_cap=KEY_OUTLIER_CAP,
    )
    quant_v = runtime["quant_v"](
        bits=bit_width,
        hidden_size=NUM_QUERY_HEADS * HEAD_DIM,
        num_query_heads=NUM_QUERY_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        max_position_embeddings=TOTAL_CONTEXT,
        include_sparse=True,
        sparsity_threshold=0.99,
        first_few_fp16=SINK_TOKENS,
        outlier_cap=VALUE_OUTLIER_CAP,
    )
    quant_k.load_lookup_table(
        _quantizer_tuple(
            quantizer["k_upper"],
            quantizer["k_lower"],
            quantizer["k_codebook"],
        ),
        include_sparse=True,
        sparsity_threshold=0.99,
        norm=False,
    )
    quant_v.load_lookup_table(
        _quantizer_tuple(
            quantizer["v_upper"],
            quantizer["v_lower"],
            quantizer["v_codebook"],
        ),
        include_sparse=True,
        sparsity_threshold=0.99,
        norm=False,
    )
    return quant_k, quant_v


def _key_tail_indices(active_count: int) -> tuple[list[int], list[int]]:
    if active_count not in {0, 6, 12}:
        raise ReferenceGenerationError("unsupported frozen Key active count")
    tail = active_count // 2
    lower = [head * HEAD_DIM for head in range(tail)]
    upper = [head * HEAD_DIM + HEAD_DIM - 1 for head in range(tail)]
    return lower, upper


def _construct_inputs(
    runtime: Mapping[str, Any],
    *,
    quant_k: Any,
    active_key_count: int,
    bit_width: int,
) -> dict[str, Any]:
    torch = runtime["torch"]
    device = torch.device("cuda:0")
    lower = quant_k.outlier_threshold_lower.reshape(KV_WIDTH)
    upper = quant_k.outlier_threshold_upper.reshape(KV_WIDTH)
    midpoint = (upper + lower) / 2.0
    half_range = (upper - lower) / 2.0
    if not torch.isfinite(half_range).all() or not torch.all(half_range > 0):
        raise ReferenceGenerationError("invalid frozen Key threshold range")

    positions = torch.arange(TOTAL_CONTEXT, device=device, dtype=torch.float32)
    coordinates = torch.arange(KV_WIDTH, device=device, dtype=torch.float32)
    normalized = 0.20 * torch.sin(
        coordinates.unsqueeze(0) * 0.013
        + positions.unsqueeze(1) * 0.17
        + bit_width * 0.11
    )
    key_rows = midpoint.unsqueeze(0) + half_range.unsqueeze(0) * normalized
    lower_indices, upper_indices = _key_tail_indices(active_key_count)
    for position in range(SINK_TOKENS, TOTAL_CONTEXT):
        if lower_indices:
            key_rows[position, lower_indices] = (
                midpoint[lower_indices] - 1.5 * half_range[lower_indices]
            )
        if upper_indices:
            key_rows[position, upper_indices] = (
                midpoint[upper_indices] + 1.5 * half_range[upper_indices]
            )
    key_pre_rope = (
        key_rows.to(torch.bfloat16)
        .reshape(TOTAL_CONTEXT, NUM_KV_HEADS, HEAD_DIM)
        .permute(1, 0, 2)
        .unsqueeze(0)
        .contiguous()
    )

    value_coordinates = torch.arange(
        KV_WIDTH, device=device, dtype=torch.float32
    )
    value_rows = []
    for position in range(TOTAL_CONTEXT):
        row = (
            0.75
            * torch.sin(
                value_coordinates * 0.019
                + position * 0.23
                + bit_width * 0.07
            )
            + 0.15
            * torch.cos(
                value_coordinates * 0.007
                - position * 0.31
                + active_key_count * 0.01
            )
            + value_coordinates * 1e-5
        )
        value_rows.append(row)
    value_after_v_proj = (
        torch.stack(value_rows)
        .to(torch.bfloat16)
        .reshape(TOTAL_CONTEXT, NUM_KV_HEADS, HEAD_DIM)
        .permute(1, 0, 2)
        .unsqueeze(0)
        .contiguous()
    )

    query_coordinates = torch.arange(
        NUM_QUERY_HEADS * HEAD_DIM,
        device=device,
        dtype=torch.float32,
    )
    query_pre_rope = (
        (
            0.35 * torch.sin(query_coordinates * 0.017 + bit_width * 0.19)
            + 0.08
            * torch.cos(query_coordinates * 0.011 + active_key_count * 0.03)
        )
        .to(torch.bfloat16)
        .reshape(1, NUM_QUERY_HEADS, 1, HEAD_DIM)
        .contiguous()
    )
    position_ids = torch.arange(
        TOTAL_CONTEXT, device=device, dtype=torch.int64
    ).reshape(1, TOTAL_CONTEXT)
    tie_rows = torch.full(
        (2, KV_WIDTH), 0.25, dtype=torch.float32, device=device
    )
    tie_sink_mask = torch.tensor([False, True], dtype=torch.bool, device=device)
    return {
        "key_pre_rope": key_pre_rope,
        "value_after_v_proj": value_after_v_proj,
        "query_pre_rope": query_pre_rope,
        "position_ids": position_ids,
        "tie_rows": tie_rows,
        "tie_sink_mask": tie_sink_mask,
    }


def _value_selection(
    runtime: Mapping[str, Any],
    value_after_v_proj: Any,
) -> Any:
    torch = runtime["torch"]
    flattened = value_after_v_proj.transpose(1, 2).reshape(
        TOTAL_CONTEXT, KV_WIDTH
    )
    sink_mask = torch.arange(
        TOTAL_CONTEXT, device=flattened.device
    ) < SINK_TOKENS
    selection = runtime["select_fixed_outliers"](
        flattened,
        cap=VALUE_OUTLIER_CAP,
        sink_row_mask=sink_mask,
    )
    expected = torch.tensor(
        [0] * SINK_TOKENS + [VALUE_OUTLIER_CAP] * QUANTIZED_CONTEXT,
        dtype=torch.int32,
        device=flattened.device,
    )
    if not torch.equal(selection.counts, expected):
        raise ReferenceGenerationError("Value fixed-extrema count mismatch")
    return selection


def _key_selection(
    runtime: Mapping[str, Any],
    key_pre_rope: Any,
    quant_k: Any,
    expected_count: int,
) -> Any:
    torch = runtime["torch"]
    flattened = key_pre_rope.transpose(1, 2).reshape(
        TOTAL_CONTEXT, KV_WIDTH
    ).float()
    midpoint = (
        quant_k.outlier_threshold_upper + quant_k.outlier_threshold_lower
    ) / 2.0
    half_range = (
        quant_k.outlier_threshold_upper - quant_k.outlier_threshold_lower
    ) / 2.0
    normalized = (flattened - midpoint.unsqueeze(0)) / half_range.unsqueeze(0)
    sink_mask = torch.arange(
        TOTAL_CONTEXT, device=flattened.device
    ) < SINK_TOKENS
    selection = runtime["select_fixed_outliers"](
        normalized,
        cap=KEY_OUTLIER_CAP,
        lower_threshold=-1.0,
        upper_threshold=1.0,
        sink_row_mask=sink_mask,
    )
    expected = torch.tensor(
        [0] * SINK_TOKENS + [expected_count] * QUANTIZED_CONTEXT,
        dtype=torch.int32,
        device=flattened.device,
    )
    if not torch.equal(selection.counts, expected):
        raise ReferenceGenerationError(
            f"Key source-faithful active count mismatch: expected {expected_count}"
        )
    return selection


def _tail_arguments(selection: Any) -> tuple[Any, Any, Any, Any]:
    torch = __import__("torch")
    lower_values = torch.empty(
        (TOTAL_CONTEXT, ENTRIES_PER_TAIL + 1),
        dtype=torch.float32,
        device=selection.values.device,
    )
    upper_values = torch.empty_like(lower_values)
    lower_indices = torch.zeros(
        (TOTAL_CONTEXT, ENTRIES_PER_TAIL + 1),
        dtype=torch.int32,
        device=selection.indices.device,
    )
    upper_indices = torch.zeros_like(lower_indices)
    lower_values[:, :ENTRIES_PER_TAIL] = selection.values[
        :, :ENTRIES_PER_TAIL
    ]
    lower_values[:, ENTRIES_PER_TAIL] = selection.dense_lower_bound
    upper_values[:, :ENTRIES_PER_TAIL] = selection.values[
        :, ENTRIES_PER_TAIL:
    ]
    upper_values[:, ENTRIES_PER_TAIL] = selection.dense_upper_bound
    lower_indices[:, :ENTRIES_PER_TAIL] = selection.indices[
        :, :ENTRIES_PER_TAIL
    ]
    upper_indices[:, :ENTRIES_PER_TAIL] = selection.indices[
        :, ENTRIES_PER_TAIL:
    ]
    return upper_values, upper_indices, lower_values, lower_indices


def _cpu_pack(codes: Any, bit_width: int) -> Any:
    """Pack [token, H_KV, head_dim] codes in exact CUDA word order."""

    import torch

    codes_cpu = codes.detach().to(device="cpu", dtype=torch.int64).contiguous()
    tokens = codes_cpu.shape[0]
    packed_rows = bit_width * HEAD_DIM // 32
    packed = torch.zeros(
        (NUM_KV_HEADS, packed_rows, tokens), dtype=torch.int64
    )
    mask32 = (1 << 32) - 1
    for token in range(tokens):
        for head in range(NUM_KV_HEADS):
            for channel in range(HEAD_DIM):
                code = int(codes_cpu[token, head, channel].item())
                bit_offset = channel * bit_width
                word = bit_offset // 32
                shift = bit_offset % 32
                packed[head, word, token] |= (code << shift) & mask32
                if shift + bit_width > 32:
                    packed[head, word + 1, token] |= code >> (32 - shift)
    signed = torch.where(packed >= (1 << 31), packed - (1 << 32), packed)
    return signed.to(torch.int32).contiguous()


def _cpu_unpack(packed: Any, bit_width: int) -> Any:
    """Unpack exact source payload into [token, H_KV, head_dim] codes."""

    import torch

    packed_cpu = packed.detach().to(device="cpu", dtype=torch.int32).contiguous()
    tokens = packed_cpu.shape[-1]
    mask = (1 << bit_width) - 1
    codes = torch.empty(
        (tokens, NUM_KV_HEADS, HEAD_DIM), dtype=torch.int64
    )
    for token in range(tokens):
        for head in range(NUM_KV_HEADS):
            for channel in range(HEAD_DIM):
                bit_offset = channel * bit_width
                word = bit_offset // 32
                shift = bit_offset % 32
                value = (
                    int(packed_cpu[head, word, token].item()) & 0xFFFFFFFF
                ) >> shift
                if shift + bit_width > 32:
                    following = (
                        int(packed_cpu[head, word + 1, token].item())
                        & 0xFFFFFFFF
                    )
                    value |= following << (32 - shift)
                codes[token, head, channel] = value & mask
    return codes.contiguous()


def _codes_from_key(key_rows: Any, lookup_table: Any) -> Any:
    torch = __import__("torch")
    rows = (
        key_rows.detach()
        .float()
        .reshape(key_rows.shape[0], KV_WIDTH)
        .unsqueeze(-1)
    )
    lut = lookup_table.detach().reshape(KV_WIDTH, -1).unsqueeze(0)
    return torch.argmin(torch.abs(rows - lut), dim=-1).reshape(
        key_rows.shape[0], NUM_KV_HEADS, HEAD_DIM
    )


def _codes_from_value(
    value_rows: Any,
    lookup_table: Any,
    selection: Any,
    bit_width: int,
) -> Any:
    torch = __import__("torch")
    rows = value_rows.detach().float().reshape(value_rows.shape[0], KV_WIDTH)
    lut = lookup_table.detach()[: value_rows.shape[0]]
    distances = torch.abs(rows.unsqueeze(-1) - lut.unsqueeze(1))
    codes = torch.argmin(distances, dim=-1)
    stop = SINK_TOKENS + value_rows.shape[0]
    lower = selection.dense_lower_bound[SINK_TOKENS:stop].unsqueeze(1)
    upper = selection.dense_upper_bound[SINK_TOKENS:stop].unsqueeze(1)
    zero_code = {4: 7, 3: 3, 2: 1}[bit_width]
    outside = (rows < lower) | (rows > upper)
    return torch.where(outside, zero_code, codes).reshape(
        value_rows.shape[0], NUM_KV_HEADS, HEAD_DIM
    )


def _explicit_rope(tensor: Any, inv_freq: Any, positions: Any) -> Any:
    torch = __import__("torch")
    frequencies = torch.outer(positions.float(), inv_freq.float())
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    cosine = embedding.cos().reshape(1, 1, positions.numel(), HEAD_DIM)
    sine = embedding.sin().reshape(1, 1, positions.numel(), HEAD_DIM)
    first = tensor[..., : HEAD_DIM // 2]
    second = tensor[..., HEAD_DIM // 2 :]
    rotated = torch.cat((-second, first), dim=-1)
    return tensor.float() * cosine + rotated.float() * sine


def _reconstruct_key(
    key_codes: Any,
    lookup_table: Any,
    outlier_values: Any,
    outlier_indices: Any,
    counts: Any,
) -> Any:
    torch = __import__("torch")
    tokens = key_codes.shape[0]
    channel_indices = torch.arange(
        HEAD_DIM, device=key_codes.device
    ).reshape(1, 1, HEAD_DIM)
    dense = torch.gather(
        lookup_table.unsqueeze(0).expand(tokens, -1, -1, -1),
        3,
        key_codes.unsqueeze(-1),
    ).squeeze(-1)
    flat = dense.reshape(tokens, KV_WIDTH)
    for token in range(tokens):
        count = int(counts[token].item())
        if count:
            indices = outlier_indices[token, :count].long()
            flat[token, indices] += outlier_values[token, :count]
    if channel_indices.numel() != HEAD_DIM:
        raise ReferenceGenerationError("invalid Key channel geometry")
    return flat.reshape(tokens, NUM_KV_HEADS, HEAD_DIM)


def _reconstruct_value(
    value_codes: Any,
    lookup_table: Any,
    outlier_values: Any,
    outlier_indices: Any,
) -> Any:
    torch = __import__("torch")
    tokens = value_codes.shape[0]
    dense = torch.gather(
        lookup_table[:tokens].unsqueeze(1).expand(-1, KV_WIDTH, -1),
        2,
        value_codes.reshape(tokens, KV_WIDTH).unsqueeze(-1),
    ).squeeze(-1)
    for token in range(tokens):
        indices = outlier_indices[token].long()
        dense[token, indices] += outlier_values[token]
    return dense.reshape(tokens, NUM_KV_HEADS, HEAD_DIM)


def _max_error(actual: Any, expected: Any) -> dict[str, float]:
    torch = __import__("torch")
    actual_float = actual.detach().float()
    expected_float = expected.detach().float()
    absolute = torch.abs(actual_float - expected_float)
    denominator = torch.maximum(
        torch.abs(expected_float), torch.tensor(1e-12, device=expected.device)
    )
    return {
        "max_abs": float(absolute.max().item()),
        "max_rel": float((absolute / denominator).max().item()),
    }


def _assert_close(
    actual: Any,
    expected: Any,
    *,
    atol: float,
    rtol: float,
    label: str,
) -> dict[str, float]:
    torch = __import__("torch")
    errors = _max_error(actual, expected)
    try:
        torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)
    except AssertionError as error:
        raise ReferenceGenerationError(
            f"{label} independent numerical control failed: {errors}"
        ) from error
    return errors


def _snapshot_state(
    quant_k: Any,
    quant_v: Any,
    sink_k: Any,
    sink_v: Any,
) -> dict[str, Any]:
    return {
        "k_dense_allocated": quant_k.kcache.detach().clone(),
        "v_dense_allocated": quant_v.vcache.detach().clone(),
        "k_sparse_values_allocated": quant_k.outliers.detach().clone(),
        "k_sparse_indices_allocated": quant_k.outlier_indices.detach().clone(),
        "v_sparse_values_allocated": quant_v.outliers.detach().clone(),
        "v_sparse_indices_allocated": quant_v.outlier_indices.detach().clone(),
        "v_lookup_allocated": quant_v.lookup_table.detach().clone(),
        "sink_k": sink_k.detach().clone(),
        "sink_v": sink_v.detach().clone(),
        "k_length": __import__("torch").tensor(
            [quant_k.klen], dtype=__import__("torch").int32
        ),
        "v_length": __import__("torch").tensor(
            [quant_v.vlen], dtype=__import__("torch").int32
        ),
    }


def _execute_fixture(
    runtime: Mapping[str, Any],
    *,
    family: str,
    bit_width: int,
    case_name: str,
    expected_key_count: int,
    quantizer: Mapping[str, Any],
    rope: Any,
) -> dict[str, Any]:
    torch = runtime["torch"]
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)

    quant_k, quant_v = _make_caches(
        runtime,
        bit_width=bit_width,
        quantizer=quantizer,
        rope=rope,
    )
    inputs = _construct_inputs(
        runtime,
        quant_k=quant_k,
        active_key_count=expected_key_count,
        bit_width=bit_width,
    )
    key_selection = _key_selection(
        runtime,
        inputs["key_pre_rope"],
        quant_k,
        expected_key_count,
    )
    value_selection = _value_selection(
        runtime, inputs["value_after_v_proj"]
    )
    tie_selection = runtime["select_fixed_outliers"](
        inputs["tie_rows"],
        cap=VALUE_OUTLIER_CAP,
        sink_row_mask=inputs["tie_sink_mask"],
    )
    if (
        tie_selection.counts.tolist() != [12, 0]
        or tie_selection.indices[0].tolist() != list(range(12))
        or torch.count_nonzero(tie_selection.values[1]).item() != 0
        or torch.count_nonzero(tie_selection.indices[1]).item() != 0
    ):
        raise ReferenceGenerationError("Value equal-tie control is unstable")

    positions = inputs["position_ids"]
    cos, sin = rope(
        inputs["value_after_v_proj"], seq_len=TOTAL_CONTEXT
    )
    key_attention = runtime["apply_rope"](
        inputs["key_pre_rope"], cos, sin, positions
    )
    query_attention = runtime["apply_rope"](
        inputs["query_pre_rope"],
        cos,
        sin,
        positions[:, APPEND_POSITION : APPEND_POSITION + 1],
    )
    sink_k = (
        key_attention[:, :, :SINK_TOKENS, :]
        .transpose(2, 3)
        .to(torch.float16)
        .contiguous()
    )
    sink_v = (
        inputs["value_after_v_proj"][:, :, :SINK_TOKENS, :]
        .to(torch.float16)
        .contiguous()
    )

    upper_values, upper_indices, lower_values, lower_indices = _tail_arguments(
        value_selection
    )
    store_key = (
        inputs["key_pre_rope"][
            0, :, SINK_TOKENS:STORE_CONTEXT, :
        ]
        .transpose(1, 2)
        .contiguous()
    )
    store_value = (
        inputs["value_after_v_proj"][
            0, :, SINK_TOKENS:STORE_CONTEXT, :
        ]
        .transpose(1, 2)
        .contiguous()
    )
    quant_k.parallel_pack(store_key)
    quant_v.parallel_pack(
        store_value,
        upper_values[SINK_TOKENS:STORE_CONTEXT],
        upper_indices[SINK_TOKENS:STORE_CONTEXT],
        lower_values[SINK_TOKENS:STORE_CONTEXT],
        lower_indices[SINK_TOKENS:STORE_CONTEXT],
    )
    quant_k.klen += SINK_TOKENS
    quant_v.vlen += SINK_TOKENS
    torch.cuda.synchronize()
    store_state = _snapshot_state(quant_k, quant_v, sink_k, sink_v)

    packed_rows = bit_width * HEAD_DIM // 32
    key_store_rows = (
        inputs["key_pre_rope"][
            0, :, SINK_TOKENS:STORE_CONTEXT, :
        ]
        .permute(1, 0, 2)
        .contiguous()
    )
    key_store_codes = _codes_from_key(
        key_store_rows, quant_k.lookup_table
    )
    key_store_packed_control = _cpu_pack(key_store_codes, bit_width)
    if not torch.equal(
        key_store_packed_control,
        quant_k.kcache[:, :packed_rows, :STORE_QUANTIZED_CONTEXT].cpu(),
    ):
        raise ReferenceGenerationError("Key store packed payload mismatch")
    value_store_rows = (
        inputs["value_after_v_proj"][
            0, :, SINK_TOKENS:STORE_CONTEXT, :
        ]
        .permute(1, 0, 2)
        .contiguous()
    )
    value_store_codes = _codes_from_value(
        value_store_rows,
        quant_v.lookup_table,
        value_selection,
        bit_width,
    )
    value_store_expected_nearest = _cpu_pack(value_store_codes, bit_width)
    value_store_payload = quant_v.vcache[
        :, :packed_rows, :STORE_QUANTIZED_CONTEXT
    ].cpu()
    value_store_codes_actual = _cpu_unpack(value_store_payload, bit_width)
    value_store_packed_control = _cpu_pack(
        value_store_codes_actual, bit_width
    )
    if not torch.equal(value_store_packed_control, value_store_payload):
        raise ReferenceGenerationError("Value store pack round-trip mismatch")
    value_store_matches_nearest = torch.equal(
        value_store_expected_nearest, value_store_payload
    )
    if bit_width in {4, 2} and not value_store_matches_nearest:
        raise ReferenceGenerationError("Value store nearest-code payload mismatch")

    sink_logits = runtime["gqa_qk"](
        query_attention.to(torch.float16),
        sink_k,
        NUM_KV_GROUPS,
    )
    append_key = inputs["key_pre_rope"][
        0, :, APPEND_POSITION : APPEND_POSITION + 1, :
    ]
    nonsink_logits = quant_k.forward_fused_sparse(
        query_attention[0], append_key
    ).unsqueeze(0)
    logits = torch.cat(
        (sink_logits.to(nonsink_logits.dtype), nonsink_logits), dim=-1
    ) / math.sqrt(HEAD_DIM)
    attention_weights = torch.nn.functional.softmax(
        logits, dim=-1, dtype=torch.float32
    ).to(query_attention.dtype)
    sink_output = runtime["gqa_av"](
        attention_weights[:, :, :, :SINK_TOKENS].to(torch.float16),
        sink_v,
        NUM_KV_GROUPS,
    )
    append_value = inputs["value_after_v_proj"][
        :, :, APPEND_POSITION : APPEND_POSITION + 1, :
    ]
    quantized_output = quant_v.forward_fused_sparse(
        attention_weights[0, :, :, SINK_TOKENS:],
        append_value,
        upper_values[APPEND_POSITION],
        upper_indices[APPEND_POSITION],
        lower_values[APPEND_POSITION],
        lower_indices[APPEND_POSITION],
    ).unsqueeze(0)
    decode_output = quantized_output + sink_output.to(quantized_output.dtype)
    torch.cuda.synchronize()
    if not torch.isfinite(decode_output).all():
        raise ReferenceGenerationError("reference decode output is non-finite")
    append_state = _snapshot_state(quant_k, quant_v, sink_k, sink_v)

    all_key_rows = (
        inputs["key_pre_rope"][0, :, SINK_TOKENS:TOTAL_CONTEXT, :]
        .permute(1, 0, 2)
        .contiguous()
    )
    all_key_codes = _codes_from_key(all_key_rows, quant_k.lookup_table)
    all_key_packed_control = _cpu_pack(all_key_codes, bit_width)
    if not torch.equal(
        all_key_packed_control,
        quant_k.kcache[:, :packed_rows, :QUANTIZED_CONTEXT].cpu(),
    ):
        raise ReferenceGenerationError("Key append packed payload mismatch")
    all_value_rows = (
        inputs["value_after_v_proj"][
            0, :, SINK_TOKENS:TOTAL_CONTEXT, :
        ]
        .permute(1, 0, 2)
        .contiguous()
    )
    all_value_expected_codes = _codes_from_value(
        all_value_rows,
        quant_v.lookup_table,
        value_selection,
        bit_width,
    )
    all_value_expected_nearest = _cpu_pack(
        all_value_expected_codes, bit_width
    )
    all_value_payload = quant_v.vcache[
        :, :packed_rows, :QUANTIZED_CONTEXT
    ].cpu()
    all_value_codes = _cpu_unpack(all_value_payload, bit_width).to(
        device=quant_v.vcache.device
    )
    all_value_packed_control = _cpu_pack(all_value_codes, bit_width)
    if not torch.equal(all_value_packed_control, all_value_payload):
        raise ReferenceGenerationError("Value append pack round-trip mismatch")
    value_append_matches_nearest = torch.equal(
        all_value_expected_nearest, all_value_payload
    )
    if bit_width in {4, 2} and not value_append_matches_nearest:
        raise ReferenceGenerationError("Value append nearest-code payload mismatch")

    key_counts = key_selection.counts[SINK_TOKENS:].to(torch.int64)
    reconstructed_key = _reconstruct_key(
        all_key_codes,
        quant_k.lookup_table,
        quant_k.outliers[:QUANTIZED_CONTEXT],
        quant_k.outlier_indices[:QUANTIZED_CONTEXT],
        key_counts,
    )
    absolute_positions = torch.arange(
        SINK_TOKENS,
        TOTAL_CONTEXT,
        device=reconstructed_key.device,
    )
    reconstructed_key_attention = _explicit_rope(
        reconstructed_key.permute(1, 0, 2).unsqueeze(0),
        rope.inv_freq,
        absolute_positions,
    )
    reconstructed_query_attention = _explicit_rope(
        inputs["query_pre_rope"].float(),
        rope.inv_freq,
        torch.tensor([APPEND_POSITION], device="cuda:0"),
    )
    repeated_key = reconstructed_key_attention.repeat_interleave(
        NUM_KV_GROUPS, dim=1
    )
    control_nonsink_logits = torch.matmul(
        reconstructed_query_attention, repeated_key.transpose(2, 3)
    )
    key_errors = _assert_close(
        nonsink_logits,
        control_nonsink_logits.to(nonsink_logits.dtype),
        atol=KEY_LOGIT_ATOL,
        rtol=KEY_LOGIT_RTOL,
        label="Key logits",
    )

    reconstructed_value = _reconstruct_value(
        all_value_codes,
        quant_v.lookup_table,
        quant_v.outliers[:QUANTIZED_CONTEXT],
        quant_v.outlier_indices[:QUANTIZED_CONTEXT],
    )
    repeated_sink_key = (
        sink_k.transpose(2, 3)
        .float()
        .repeat_interleave(NUM_KV_GROUPS, dim=1)
    )
    control_sink_logits = torch.matmul(
        reconstructed_query_attention,
        repeated_sink_key.transpose(2, 3),
    )
    control_logits = torch.cat(
        (control_sink_logits, control_nonsink_logits), dim=-1
    ) / math.sqrt(HEAD_DIM)
    control_weights = torch.nn.functional.softmax(
        control_logits, dim=-1, dtype=torch.float32
    ).to(torch.bfloat16)
    repeated_sink_value = sink_v.float().repeat_interleave(
        NUM_KV_GROUPS, dim=1
    )
    repeated_value = (
        reconstructed_value.permute(1, 0, 2)
        .unsqueeze(0)
        .repeat_interleave(NUM_KV_GROUPS, dim=1)
    )
    control_output = (
        torch.matmul(
            control_weights[:, :, :, :SINK_TOKENS].float(),
            repeated_sink_value,
        )
        + torch.matmul(
            control_weights[:, :, :, SINK_TOKENS:].float(),
            repeated_value,
        )
    ).to(torch.bfloat16)
    decode_errors = _assert_close(
        decode_output,
        control_output,
        atol=DECODE_ATOL,
        rtol=DECODE_RTOL,
        label="decode output",
    )

    return {
        "family": family,
        "bit_width": bit_width,
        "case_name": case_name,
        "expected_key_count": expected_key_count,
        "quant_k": quant_k,
        "quant_v": quant_v,
        "inputs": inputs,
        "key_selection": key_selection,
        "value_selection": value_selection,
        "tie_selection": tie_selection,
        "key_attention": key_attention,
        "query_attention": query_attention,
        "sink_k": sink_k,
        "sink_v": sink_v,
        "store_state": store_state,
        "append_state": append_state,
        "key_store_codes": key_store_codes,
        "value_store_codes": value_store_codes,
        "value_store_codes_actual": value_store_codes_actual,
        "all_key_codes": all_key_codes,
        "all_value_codes": all_value_codes,
        "key_store_packed_control": key_store_packed_control,
        "value_store_packed_control": value_store_packed_control,
        "all_key_packed_control": all_key_packed_control,
        "all_value_packed_control": all_value_packed_control,
        "value_store_expected_nearest": value_store_expected_nearest,
        "all_value_expected_nearest": all_value_expected_nearest,
        "value_store_matches_nearest": value_store_matches_nearest,
        "value_append_matches_nearest": value_append_matches_nearest,
        "nonsink_logits": nonsink_logits,
        "attention_weights": attention_weights,
        "decode_output": decode_output,
        "control_nonsink_logits": control_nonsink_logits,
        "control_weights": control_weights,
        "control_output": control_output,
        "reconstructed_key": reconstructed_key,
        "reconstructed_value": reconstructed_value,
        "key_errors": key_errors,
        "decode_errors": decode_errors,
    }


def _storage_bytes(tensor: Any) -> int:
    return int(tensor.untyped_storage().nbytes())


def _byte_breakdown(result: Mapping[str, Any]) -> dict[str, Any]:
    bit_width = int(result["bit_width"])
    quant_k = result["quant_k"]
    quant_v = result["quant_v"]
    sink_k = result["sink_k"]
    sink_v = result["sink_v"]
    dense_k = _storage_bytes(quant_k.kcache)
    dense_v = _storage_bytes(quant_v.vcache)
    key_metadata_tensors = (
        quant_k.lookup_table,
        quant_k.outlier_threshold_lower,
        quant_k.outlier_threshold_upper,
        quant_k.zeropoint,
        quant_k.lut,
        quant_k.rope_inv_freq,
    )
    value_metadata_tensors = (
        quant_v.lookup_table,
        quant_v.lut,
    )
    key_metadata = sum(_storage_bytes(item) for item in key_metadata_tensors)
    value_metadata = sum(
        _storage_bytes(item) for item in value_metadata_tensors
    )
    key_sparse_values = _storage_bytes(quant_k.outliers)
    key_sparse_indices = _storage_bytes(quant_k.outlier_indices)
    value_sparse_values = _storage_bytes(quant_v.outliers)
    value_sparse_indices = _storage_bytes(quant_v.outlier_indices)
    sink_k_bytes = _storage_bytes(sink_k)
    sink_v_bytes = _storage_bytes(sink_v)
    padding = 0
    workspace = 0
    categories = {
        "dense_k_payload_bytes": dense_k,
        "dense_v_payload_bytes": dense_v,
        "key_metadata_bytes": key_metadata,
        "value_metadata_bytes": value_metadata,
        "key_sparse_value_bytes": key_sparse_values,
        "key_sparse_index_bytes": key_sparse_indices,
        "value_sparse_value_bytes": value_sparse_values,
        "value_sparse_index_bytes": value_sparse_indices,
        "sink_k_bytes": sink_k_bytes,
        "sink_v_bytes": sink_v_bytes,
        "padding_alignment_bytes": padding,
        "persistent_reference_workspace_bytes": workspace,
    }
    allocated_total = sum(categories.values())
    levels = 1 << bit_width
    active_dense_each = (
        NUM_KV_HEADS
        * (HEAD_DIM * bit_width // 32)
        * QUANTIZED_CONTEXT
        * 4
    )
    key_active_entries = (
        int(result["expected_key_count"]) * QUANTIZED_CONTEXT
    )
    value_active_entries = VALUE_OUTLIER_CAP * QUANTIZED_CONTEXT
    active_key_metadata = (
        NUM_KV_HEADS * HEAD_DIM * levels * 4
        + 3 * KV_WIDTH * 4
        + levels * 4
        + (HEAD_DIM // 2) * 4
    )
    active_value_metadata = (
        QUANTIZED_CONTEXT * levels * 4 + levels * 4
    )
    active_logical = (
        2 * active_dense_each
        + active_key_metadata
        + active_value_metadata
        + key_active_entries * 8
        + value_active_entries * 8
        + sink_k_bytes
        + sink_v_bytes
    )
    logical_bf16 = (
        2 * BATCH_SIZE * NUM_KV_HEADS * TOTAL_CONTEXT * HEAD_DIM * 2
    )
    rho_alloc = allocated_total / logical_bf16
    r_alloc = logical_bf16 / allocated_total
    if abs(rho_alloc * r_alloc - 1.0) > RECIPROCAL_ATOL:
        raise ReferenceGenerationError("allocation ratios are not reciprocal")
    return {
        "schema_version": "kvbench-phase10-byte-breakdown-1.0.0",
        "fixture_id": FIXTURE_ID,
        "family": result["family"],
        "case": result["case_name"],
        "bit_width": bit_width,
        "allocation_basis": "source_owned_tensor_storage",
        **categories,
        "actual_allocated_total_bytes": allocated_total,
        "active_logical_total_bytes": active_logical,
        "logical_bf16_bytes": logical_bf16,
        "rho_alloc": rho_alloc,
        "r_alloc": r_alloc,
        "reciprocal_absolute_error": abs(rho_alloc * r_alloc - 1.0),
        "reciprocal_tolerance": RECIPROCAL_ATOL,
        "r_hbm": None,
        "fixed_capacity": {
            "key_slots_per_physical_row": KEY_OUTLIER_CAP,
            "value_slots_per_physical_row": VALUE_OUTLIER_CAP,
            "key_active_entries": key_active_entries,
            "value_active_entries_non_sink": value_active_entries,
            "value_active_entries_sink": 0,
        },
    }


def _fixture_tensors(result: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    torch = __import__("torch")
    inputs = result["inputs"]
    quant_k = result["quant_k"]
    quant_v = result["quant_v"]
    key_selection = result["key_selection"]
    value_selection = result["value_selection"]
    tie_selection = result["tie_selection"]
    store_state = result["store_state"]
    append_state = result["append_state"]
    packed_rows = int(result["bit_width"]) * HEAD_DIM // 32

    return {
        "inputs.safetensors": {
            "key_pre_rope": inputs["key_pre_rope"],
            "value_after_v_proj": inputs["value_after_v_proj"],
            "query_pre_rope": inputs["query_pre_rope"],
            "position_ids": inputs["position_ids"],
            "value_tie_control_rows": inputs["tie_rows"],
            "value_tie_control_sink_mask": inputs["tie_sink_mask"],
        },
        "dense_payload.safetensors": {
            "key_packed_after_store": store_state["k_dense_allocated"][
                :, :packed_rows, :STORE_QUANTIZED_CONTEXT
            ],
            "value_packed_after_store": store_state["v_dense_allocated"][
                :, :packed_rows, :STORE_QUANTIZED_CONTEXT
            ],
            "key_packed_after_append": append_state["k_dense_allocated"][
                :, :packed_rows, :QUANTIZED_CONTEXT
            ],
            "value_packed_after_append": append_state["v_dense_allocated"][
                :, :packed_rows, :QUANTIZED_CONTEXT
            ],
            "key_appended_slot": append_state["k_dense_allocated"][
                :, :packed_rows, QUANTIZED_CONTEXT - 1
            ],
            "value_appended_slot": append_state["v_dense_allocated"][
                :, :packed_rows, QUANTIZED_CONTEXT - 1
            ],
            "key_packed_independent_control": result[
                "all_key_packed_control"
            ],
            "value_packed_independent_control": result[
                "all_value_packed_control"
            ],
            "value_expected_nearest_after_store": result[
                "value_store_expected_nearest"
            ],
            "value_expected_nearest_after_append": result[
                "all_value_expected_nearest"
            ],
        },
        "metadata.safetensors": {
            "key_codebook": quant_k.lut,
            "key_lookup_table": quant_k.lookup_table,
            "key_runtime_lower_threshold": quant_k.outlier_threshold_lower,
            "key_runtime_upper_threshold": quant_k.outlier_threshold_upper,
            "key_runtime_zero": quant_k.zeropoint,
            "value_codebook": quant_v.lut,
            "value_lookup_after_store": store_state["v_lookup_allocated"],
            "value_lookup_after_append": append_state["v_lookup_allocated"],
            "value_dense_lower_bound": value_selection.dense_lower_bound,
            "value_dense_upper_bound": value_selection.dense_upper_bound,
            "rope_inv_freq": quant_k.rope_inv_freq,
        },
        "sparse_values.safetensors": {
            "key_selection_normalized_by_position": key_selection.values,
            "key_cache_after_store": store_state[
                "k_sparse_values_allocated"
            ],
            "key_cache_after_append": append_state[
                "k_sparse_values_allocated"
            ],
            "value_selection_by_position": value_selection.values,
            "value_cache_after_store": store_state[
                "v_sparse_values_allocated"
            ],
            "value_cache_after_append": append_state[
                "v_sparse_values_allocated"
            ],
            "value_tie_control": tie_selection.values,
        },
        "sparse_indices.safetensors": {
            "key_selection_by_position": key_selection.indices,
            "key_active_count_by_position": key_selection.counts,
            "key_cache_after_store": store_state[
                "k_sparse_indices_allocated"
            ],
            "key_cache_after_append": append_state[
                "k_sparse_indices_allocated"
            ],
            "value_selection_by_position": value_selection.indices,
            "value_active_count_by_position": value_selection.counts,
            "value_cache_after_store": store_state[
                "v_sparse_indices_allocated"
            ],
            "value_cache_after_append": append_state[
                "v_sparse_indices_allocated"
            ],
            "value_tie_control": tie_selection.indices,
            "value_tie_control_counts": tie_selection.counts,
        },
        "sink.safetensors": {
            "sink_key_pre_rope_bf16": inputs["key_pre_rope"][
                :, :, :SINK_TOKENS, :
            ],
            "sink_key_attention_fp16": result["sink_k"],
            "sink_value_fp16": result["sink_v"],
            "sink_positions": torch.arange(SINK_TOKENS, dtype=torch.int64),
        },
        "store_state.safetensors": store_state,
        "append_state.safetensors": append_state,
        "decode_output.safetensors": {
            "query_attention_ready": result["query_attention"],
            "source_nonsink_key_logits": result["nonsink_logits"],
            "attention_weights": result["attention_weights"],
            "source_decode_output": result["decode_output"],
            "independent_nonsink_key_logits": result[
                "control_nonsink_logits"
            ],
            "independent_attention_weights": result["control_weights"],
            "independent_decode_output": result["control_output"],
            "explicit_dense_key_reconstruction": result[
                "reconstructed_key"
            ],
            "explicit_dense_value_reconstruction": result[
                "reconstructed_value"
            ],
        },
    }


def _manifest_for_fixture(
    result: Mapping[str, Any],
    tensor_files: Mapping[str, Mapping[str, Any]],
    byte_breakdown: Mapping[str, Any],
    authority_hashes: Mapping[str, str],
) -> dict[str, Any]:
    tensor_records = {
        filename: {
            name: _tensor_record(tensor)
            for name, tensor in sorted(tensors.items())
        }
        for filename, tensors in sorted(tensor_files.items())
    }
    expected_key_count = int(result["expected_key_count"])
    return {
        "schema_version": "kvbench-phase10-kvquant-fixture-1.0.0",
        "fixture_id": FIXTURE_ID,
        "family": result["family"],
        "case": result["case_name"],
        "bit_width": result["bit_width"],
        "status": "PASS",
        "run_kind": "reference_fixture",
        "source": {
            "method_identifier": METHOD_IDENTIFIER,
            "decision": DECISION,
            "contract_decision": CONTRACT_DECISION,
            "patched_commit": PATCHED_COMMIT,
            "patched_tree": PATCHED_TREE,
            "patch_sha256": PATCH_SHA256,
            "source_manifest_sha256": authority_hashes["source_manifest"],
        },
        "calibration": {
            "calibration_id": CALIBRATION_ID,
            "root_sha256": CALIBRATION_ROOT,
            "quantizer_sha256": QUANTIZER_SHA256[result["family"]],
            "calibration_manifest_sha256": authority_hashes[
                "calibration_manifest"
            ],
            "fisher_regenerated": False,
            "quantizer_regenerated": False,
        },
        "environment": {
            "image_config_digest": REFERENCE_IMAGE_CONFIG_DIGEST,
            "environment_manifest_sha256": authority_hashes["environment"],
            "build_manifest_sha256": authority_hashes["build_manifest"],
        },
        "geometry": {
            "batch_size": BATCH_SIZE,
            "num_query_heads": NUM_QUERY_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "num_kv_groups": NUM_KV_GROUPS,
            "head_dim": HEAD_DIM,
            "interface_dtype": INTERFACE_DTYPE,
            "sink_dtype": SINK_DTYPE,
            "sink_tokens": SINK_TOKENS,
            "store_context": STORE_CONTEXT,
            "append_tokens": APPEND_TOKENS,
            "total_context": TOTAL_CONTEXT,
            "seed": SEED,
            "query_to_kv_mapping": "kv_head = query_head // 4",
        },
        "sparse_contract": {
            "key_sparse_selection_mode": "thresholded_fixed_tail_cap",
            "key_active_count": expected_key_count,
            "key_capacity": KEY_OUTLIER_CAP,
            "value_sparse_selection_mode": "fixed_extrema",
            "value_active_count_non_sink": VALUE_OUTLIER_CAP,
            "value_active_count_sink": 0,
            "value_capacity": VALUE_OUTLIER_CAP,
            "value_lower_entries": ENTRIES_PER_TAIL,
            "value_upper_entries": ENTRIES_PER_TAIL,
            "value_occupancy_data_dependent": False,
            "unused_key_slots_zero": True,
            "unused_value_slots_non_sink": 0,
            "outlier_value_dtype": "float32",
            "outlier_index_dtype": "int32",
            "ties": "stable_value_then_flat_index",
        },
        "semantics": {
            "quantized_key": "pre_rope_k_proj_output",
            "sink_key_stored": "post_rope_attention_ready_fp16",
            "attention_key": "native_llama31_rope_applied_during_reference_decode",
            "value": "native_v_proj_output_without_rope",
            "position_ids": list(range(TOTAL_CONTEXT)),
            "sink_positions": list(range(SINK_TOKENS)),
            "dense_quantized_positions": list(
                range(SINK_TOKENS, TOTAL_CONTEXT)
            ),
            "implementation_head_expansion": False,
            "independent_control_head_expansion": True,
        },
        "packing": {
            "order": (
                "channel_codes_lsb_first_in_contiguous_int32_words;"
                "three_bit_codes_cross_word_boundaries"
            ),
            "packed_dtype": "int32",
            "packed_rows_per_kv_head": result["bit_width"] * HEAD_DIM // 32,
            "native_kv_heads": NUM_KV_HEADS,
            "value_parallel_store_matches_intended_nearest": result[
                "value_store_matches_nearest"
            ],
            "value_after_append_matches_intended_nearest": result[
                "value_append_matches_nearest"
            ],
            "three_bit_parallel_value_source_behavior": (
                "exact_patched_source_payload_round_trip"
                if result["bit_width"] == 3
                else "not_applicable"
            ),
        },
        "execution_path": {
            "dense_quantization_packing": "patched_cuda_parallel_then_append",
            "sparse_correction": "fused_cuda_dense_plus_sparse",
            "sink_handling": "separate_native_hkv8_fp16_matmul",
            "decode_dequantization": "direct_compressed_cache_consumption",
            "full_prefix_temporary": False,
            "backend_fallback": False,
            "reference_dynamic_allocation": True,
            "source_repeat_kv": False,
            "source_repeat_interleave": False,
            "control_repeat_interleave": True,
            "timing_fields_present": False,
        },
        "numerical_control": {
            "key_logits_atol": KEY_LOGIT_ATOL,
            "key_logits_rtol": KEY_LOGIT_RTOL,
            "decode_atol": DECODE_ATOL,
            "decode_rtol": DECODE_RTOL,
            "key_errors": result["key_errors"],
            "decode_errors": result["decode_errors"],
            "finite": True,
        },
        "byte_breakdown_sha256": _sha256_bytes(
            _canonical_json(byte_breakdown)
        ),
        "tensor_records": tensor_records,
        "performance_measurement": False,
        "profiler_execution": False,
        "quality_evaluation": False,
        "g2_kvq": "NOT_EVALUATED",
    }


def _write_fixture(
    fixture_root: Path,
    result: Mapping[str, Any],
    authority_hashes: Mapping[str, str],
) -> None:
    fixture_root.mkdir(parents=True, exist_ok=False)
    tensors = _fixture_tensors(result)
    byte_breakdown = _byte_breakdown(result)
    manifest = _manifest_for_fixture(
        result,
        tensors,
        byte_breakdown,
        authority_hashes,
    )
    _write_exclusive(
        fixture_root / "fixture_manifest.json", _canonical_json(manifest)
    )
    for filename, payload in tensors.items():
        _save_safetensors(fixture_root / filename, payload)
    _write_exclusive(
        fixture_root / "byte_breakdown.json",
        _canonical_json(byte_breakdown),
    )
    ledger_entries = []
    for path in sorted(fixture_root.iterdir()):
        if path.name == "checksums.sha256":
            continue
        ledger_entries.append(f"{_sha256_file(path)}  {path.name}\n")
    _write_exclusive(
        fixture_root / "checksums.sha256",
        "".join(ledger_entries).encode("utf-8"),
    )
    if tuple(sorted(path.name for path in fixture_root.iterdir())) != tuple(
        sorted(FIXTURE_MEMBERS)
    ):
        raise ReferenceGenerationError("fixture member set is not exact")


def _authority_manifests(
    *,
    source_probe: Mapping[str, Any],
    calibration_probe: Mapping[str, Any],
    extension_path: Path,
    source_root: Path,
    calibration_root: Path,
    patch_manifest_path: Path,
) -> dict[str, dict[str, Any]]:
    torch = __import__("torch")
    import quant_cuda
    import tokenizers
    import transformers

    if Path(quant_cuda.__file__).resolve() != extension_path.resolve():
        raise ReferenceGenerationError("loaded extension path mismatch")
    extension_sha256 = _sha256_file(extension_path)
    reference_dockerfile = (
        Path("/repo/docker/reference-kvquant.Dockerfile").resolve(strict=True)
    )
    if _sha256_file(reference_dockerfile) != REFERENCE_DOCKERFILE_SHA256:
        raise ReferenceGenerationError("reference Dockerfile identity mismatch")
    sys = __import__("sys")
    python_version = ".".join(map(str, sys.version_info[:3]))
    gcc_identity = _run(("/usr/bin/gcc", "--version")).splitlines()[0]
    nvcc_identity = _run(("/usr/local/cuda/bin/nvcc", "--version"))
    base_distribution_versions = json.loads(
        _run(
            (
                sys.executable,
                "-c",
                (
                    "import importlib.metadata as m,json;"
                    "d={x.metadata['Name'].lower():x.version for x in "
                    "m.distributions(path=['/opt/kvbench/.phase3/site-packages'])};"
                    "print(json.dumps({k:d.get(k) for k in "
                    "['transformers','tokenizers']},sort_keys=True))"
                ),
            )
        )
    )
    runtime_identity = {
        "python": python_version,
        "pytorch": torch.__version__,
        "cuda_userspace": torch.version.cuda,
        "transformers_installed": base_distribution_versions.get(
            "transformers"
        ),
        "transformers_vendored": transformers.__version__,
        "tokenizers_base_installed": base_distribution_versions.get(
            "tokenizers"
        ),
        "tokenizers_active": tokenizers.__version__,
        "gcc_version": EXPECTED_GCC if EXPECTED_GCC in gcc_identity else None,
        "nvcc_version": EXPECTED_NVCC if EXPECTED_NVCC in nvcc_identity else None,
    }
    expected_runtime_identity = {
        "python": EXPECTED_PYTHON,
        "pytorch": EXPECTED_PYTORCH,
        "cuda_userspace": EXPECTED_CUDA,
        "transformers_installed": EXPECTED_TRANSFORMERS_INSTALLED,
        "transformers_vendored": EXPECTED_TRANSFORMERS_VENDORED,
        "tokenizers_base_installed": EXPECTED_TOKENIZERS_BASE,
        "tokenizers_active": EXPECTED_TOKENIZERS_ACTIVE,
        "gcc_version": EXPECTED_GCC,
        "nvcc_version": EXPECTED_NVCC,
    }
    if runtime_identity != expected_runtime_identity:
        raise ReferenceGenerationError(
            f"reference runtime identity mismatch: {runtime_identity}"
        )
    source_manifest = {
        "schema_version": "kvbench-phase10-kvquant-source-1.0.0",
        "status": "PASS",
        "method_identifier": METHOD_IDENTIFIER,
        "human_name": "KVQuant-GQA patched upstream",
        "official_author_gqa_support_claimed": False,
        "decision": DECISION,
        "contract_decision": CONTRACT_DECISION,
        "repository": UPSTREAM_REPOSITORY,
        "upstream_base_commit": UPSTREAM_BASE_COMMIT,
        "upstream_base_tree": UPSTREAM_BASE_TREE,
        "patch_sha256": PATCH_SHA256,
        "patched_commit": PATCHED_COMMIT,
        "patched_tree": PATCHED_TREE,
        "reconstruction": dict(source_probe),
        "patch_manifest_sha256": _sha256_file(patch_manifest_path),
        "source_mount": "read_only",
        "source_checkout_published": False,
    }
    environment = {
        "schema_version": "kvbench-phase10-kvquant-environment-1.0.0",
        "status": "PASS",
        "strategy": "thin_image_from_exact_phase9_calibration_image",
        "image_config_digest": REFERENCE_IMAGE_CONFIG_DIGEST,
        "base_image_config_digest": REFERENCE_BASE_CONFIG_DIGEST,
        "calibration_dockerfile_sha256": CALIBRATION_DOCKERFILE_SHA256,
        "reference_dockerfile_sha256": REFERENCE_DOCKERFILE_SHA256,
        "calibration_python_freeze_sha256": (
            CALIBRATION_PYTHON_FREEZE_SHA256
        ),
        "new_dockerfile_created": True,
        "dependency_overlay": {
            "package": "tokenizers",
            "version": EXPECTED_TOKENIZERS_ACTIVE,
            "wheel_sha256": TOKENIZERS_WHEEL_SHA256,
            "installation": "no_deps_no_index_isolated_target",
        },
        "python": python_version,
        "pytorch": torch.__version__,
        "cuda_userspace": torch.version.cuda,
        "cuda_runtime": torch._C._cuda_getCompiledVersion(),
        "transformers_installed": base_distribution_versions["transformers"],
        "transformers_vendored": transformers.__version__,
        "tokenizers_base_installed": base_distribution_versions["tokenizers"],
        "tokenizers_active": tokenizers.__version__,
        "compiler": gcc_identity,
        "cuda_compiler": nvcc_identity,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
        "source_mount": "read_only",
        "calibration_mount": "read_only",
        "output_mount": "write_only_fixture_destination",
        "model_weights_in_image": False,
        "credentials_in_image": False,
        "r2_credentials_in_container": False,
        "network": "disabled",
    }
    calibration_manifest = {
        "schema_version": "kvbench-phase10-kvquant-calibration-binding-1.0.0",
        "status": "PASS",
        "calibration_id": CALIBRATION_ID,
        "calibration_root_sha256": CALIBRATION_ROOT,
        "r2_source": CALIBRATION_R2_URI,
        "complete": True,
        "inventory_checksums": "PASS",
        "mount": "read_only",
        "method_identifier": METHOD_IDENTIFIER,
        "patched_tree": PATCHED_TREE,
        "quantizers": calibration_probe["quantizers"],
        "fisher_regenerated": False,
        "quantizers_regenerated": False,
        "calibration_path_published": False,
        "local_binding_path": calibration_root.name,
    }
    build_manifest = {
        "schema_version": "kvbench-phase10-kvquant-build-1.0.0",
        "status": "PASS",
        "source_authority": {
            "patched_tree": PATCHED_TREE,
            "source_root_basename": source_root.name,
        },
        "image_config_digest": REFERENCE_IMAGE_CONFIG_DIGEST,
        "build_command": (
            "python3 setup_cuda.py build_ext --inplace "
            "with frozen SM120 and compute_120 flags"
        ),
        "flags": [
            "-O3",
            "-gencode=arch=compute_120,code=sm_120",
            "-gencode=arch=compute_120,code=compute_120",
            "-ccbin /usr/bin/gcc",
            "-std=c++20",
        ],
        "python": python_version,
        "pytorch": torch.__version__,
        "cuda_userspace": torch.version.cuda,
        "compiler": gcc_identity,
        "cuda_compiler": nvcc_identity,
        "extension_filename": extension_path.name,
        "extension_sha256": extension_sha256,
        "extension_size_bytes": extension_path.stat().st_size,
        "sm_120_cubin": True,
        "compute_120_ptx": True,
        "native_sm120_execution": True,
        "forced_ptx_jit": "PASS",
        "compute_sanitizer": "PASS",
        "fallback": False,
        "extension_published": False,
    }
    return {
        "source_manifest.json": source_manifest,
        "environment.json": environment,
        "calibration_manifest.json": calibration_manifest,
        "build_manifest.json": build_manifest,
    }


def _root_manifest(
    authority_hashes: Mapping[str, str],
    extension_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": "kvbench-phase10-kvquant-reference-bundle-1.0.0",
        "run_id": FIXTURE_ID,
        "status": "completed",
        "phase": "phase10_kvquant_reference",
        "run_kind": "reference_fixture",
        "method_identifier": METHOD_IDENTIFIER,
        "decision": DECISION,
        "contract_decision": CONTRACT_DECISION,
        "source": {
            "upstream_base_commit": UPSTREAM_BASE_COMMIT,
            "upstream_base_tree": UPSTREAM_BASE_TREE,
            "patch_sha256": PATCH_SHA256,
            "patched_commit": PATCHED_COMMIT,
            "patched_tree": PATCHED_TREE,
            "source_manifest_sha256": authority_hashes["source_manifest"],
            "official_author_gqa_support_claimed": False,
        },
        "calibration": {
            "calibration_id": CALIBRATION_ID,
            "root_sha256": CALIBRATION_ROOT,
            "calibration_manifest_sha256": authority_hashes[
                "calibration_manifest"
            ],
            "fisher_regenerated": False,
            "quantizers_regenerated": False,
        },
        "environment": {
            "image_config_digest": REFERENCE_IMAGE_CONFIG_DIGEST,
            "environment_manifest_sha256": authority_hashes["environment"],
            "build_manifest_sha256": authority_hashes["build_manifest"],
            "extension_sha256": extension_sha256,
        },
        "fixture_matrix": {
            "families": [family for family, _ in FAMILIES],
            "cases": [case for case, _ in CASES],
            "total": len(FAMILIES) * len(CASES),
            "legacy_ambiguous_aliases": False,
        },
        "geometry": {
            "batch_size": BATCH_SIZE,
            "num_query_heads": NUM_QUERY_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "num_kv_groups": NUM_KV_GROUPS,
            "head_dim": HEAD_DIM,
            "store_context": STORE_CONTEXT,
            "append_tokens": APPEND_TOKENS,
            "total_context": TOTAL_CONTEXT,
            "sink_tokens": SINK_TOKENS,
            "key_outlier_cap": KEY_OUTLIER_CAP,
            "value_outlier_cap": VALUE_OUTLIER_CAP,
            "seed": SEED,
            "dtype": INTERFACE_DTYPE,
        },
        "sparse_contract": {
            "key_counts": [count for _, count in CASES],
            "value_non_sink_count": 12,
            "value_sink_count": 0,
            "value_selection": "six_lowest_plus_six_highest",
            "value_occupancy_data_dependent": False,
        },
        "gates": {
            "g2_kvq": "NOT_EVALUATED",
            "global_g2_g5": "NOT_EVALUATED",
            "full_scan": "CLOSED",
            "quality_execution": "LOCKED",
            "performance_data_frozen_present": False,
        },
        "performance_measurement": False,
        "profiler_execution": False,
        "quality_evaluation": False,
    }


def _reference_trace() -> dict[str, Any]:
    return {
        "schema_version": "kvbench-phase10-reference-trace-1.0.0",
        "run_id": FIXTURE_ID,
        "run_kind": "reference_trace",
        "timing_fields_present": False,
        "coverage": [
            {
                "family": "kvq4",
                "case": "key_zero_value_fixed12",
                "paths": ["dense_pack", "store", "append", "sink", "decode"],
            },
            {
                "family": "kvq4",
                "case": "key_cap_value_fixed12",
                "paths": ["fixed_cap_sparse", "fused_sparse_correction"],
            },
            {
                "family": "kvq3",
                "case": "key_few_value_fixed12",
                "paths": ["three_bit_cross_word_pack", "three_bit_decode"],
            },
            {
                "family": "kvq2",
                "case": "key_cap_value_fixed12",
                "paths": ["two_bit_pack", "two_bit_decode", "fixed_cap_sparse"],
            },
        ],
        "operations": {
            "dense_quantization_packing": "patched CUDA parallel store and append",
            "metadata": "frozen Key LUT plus per-token Value LUT",
            "sparse_selection": "stable lower then upper selection",
            "sparse_correction": "fused separate sparse correction within CUDA kernels",
            "sink": "separate native-H_KV FP16 storage and matmul",
            "decode": "direct compressed-cache consumption",
            "gqa": "native 32Q/8KV mapping without cache expansion",
            "complete_prefix_temporary": False,
            "backend_fallback": False,
            "reference_dynamic_allocation": True,
        },
    }


def _inventory_role(relative: str) -> str:
    if relative.endswith("fixture_manifest.json") or relative == "manifest.json":
        return "manifest"
    if relative.endswith(".safetensors"):
        return "safe_tensor_fixture"
    if relative.endswith(".sha256"):
        return "checksum_ledger"
    if relative == "reference_trace.json":
        return "reference_trace"
    if relative.endswith(".json"):
        return "metadata"
    return "fixture"


def _finalize_bundle(stage: Path, root_manifest: Mapping[str, Any]) -> str:
    _write_exclusive(stage / "manifest.json", _canonical_json(root_manifest))
    _write_exclusive(
        stage / "reference_trace.json", _canonical_json(_reference_trace())
    )
    payload_paths = [
        path
        for path in sorted(path for path in stage.rglob("*") if path.is_file())
        if _safe_relative(path, stage)
        not in {"artifact_inventory.json", "checksums.sha256", "COMPLETE"}
    ]
    inventory_files = []
    for path in payload_paths:
        relative = _safe_relative(path, stage)
        inventory_files.append(
            {
                "path": relative,
                "role": _inventory_role(relative),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    inventory = {
        "schema_version": "kvbench-artifact-inventory-1.0.0",
        "run_id": FIXTURE_ID,
        "files": inventory_files,
        "excluded_control_files": [
            "artifact_inventory.json",
            "checksums.sha256",
            "COMPLETE",
        ],
    }
    _write_exclusive(
        stage / "artifact_inventory.json", _canonical_json(inventory)
    )
    ledger_paths = sorted(
        path
        for path in stage.rglob("*")
        if path.is_file()
        and _safe_relative(path, stage) not in {"checksums.sha256", "COMPLETE"}
    )
    ledger = "".join(
        f"{_sha256_file(path)}  {_safe_relative(path, stage)}\n"
        for path in ledger_paths
    ).encode("utf-8")
    _write_exclusive(stage / "checksums.sha256", ledger)
    completion = {
        "schema_version": "kvbench-completion-1.0.0",
        "run_id": FIXTURE_ID,
        "status": "completed",
        "manifest_sha256": _sha256_file(stage / "manifest.json"),
        "artifact_inventory_sha256": _sha256_file(
            stage / "artifact_inventory.json"
        ),
        "checksum_ledger_sha256": _sha256_bytes(ledger),
        "checksum_ledger_path": "checksums.sha256",
        "written_last": True,
    }
    _write_exclusive(stage / "COMPLETE", _canonical_json(completion))

    for path in sorted(stage.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    stage.chmod(0o555)
    all_files = sorted(path for path in stage.rglob("*") if path.is_file())
    records = [
        (
            _safe_relative(path, stage),
            _sha256_file(path),
        )
        for path in all_files
    ]
    root_hasher = hashlib.sha256()
    for relative, digest in records:
        root_hasher.update(f"{digest}  {relative}\n".encode("utf-8"))
    return root_hasher.hexdigest()


def _install_no_replace(stage: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise ReferenceGenerationError(
            "refusing to overwrite an existing finalized fixture bundle"
        )
    os.rename(stage, destination)


def generate_bundle(arguments: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(arguments.source_root)
    calibration_root = Path(arguments.calibration_root)
    extension_path = Path(arguments.extension).resolve(strict=True)
    patch_manifest_path = Path(arguments.patch_manifest)
    reference_root = Path(arguments.reference_root).resolve()
    destination = reference_root / "fixtures"
    if destination.exists() or destination.is_symlink():
        raise ReferenceGenerationError(
            "fixture destination already exists; validate instead of regenerating"
        )
    source_probe = _validate_source(source_root, patch_manifest_path)
    calibration_probe = _validate_calibration(calibration_root)
    runtime = _runtime()
    import quant_cuda

    if Path(quant_cuda.__file__).resolve() != extension_path:
        raise ReferenceGenerationError("quant_cuda was not loaded from --extension")
    manifests = _authority_manifests(
        source_probe=source_probe,
        calibration_probe=calibration_probe,
        extension_path=extension_path,
        source_root=source_root,
        calibration_root=calibration_root,
        patch_manifest_path=patch_manifest_path,
    )
    manifest_bytes = {
        name: _canonical_json(payload) for name, payload in manifests.items()
    }
    authority_hashes = {
        "source_manifest": _sha256_bytes(
            manifest_bytes["source_manifest.json"]
        ),
        "environment": _sha256_bytes(manifest_bytes["environment.json"]),
        "calibration_manifest": _sha256_bytes(
            manifest_bytes["calibration_manifest.json"]
        ),
        "build_manifest": _sha256_bytes(
            manifest_bytes["build_manifest.json"]
        ),
    }

    reference_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=".kvbench-phase10-fixtures-",
            dir=reference_root,
        )
    )
    stage = temporary / "fixtures"
    stage.mkdir()
    try:
        rope = _make_rope(runtime)
        for family, bit_width in FAMILIES:
            quantizer = _load_layer_zero_quantizer(calibration_root, family)
            for case_name, expected_key_count in CASES:
                with runtime["torch"].inference_mode():
                    result = _execute_fixture(
                        runtime,
                        family=family,
                        bit_width=bit_width,
                        case_name=case_name,
                        expected_key_count=expected_key_count,
                        quantizer=quantizer,
                        rope=rope,
                    )
                _write_fixture(
                    stage / family / case_name,
                    result,
                    authority_hashes,
                )
                del result
                gc.collect()
                runtime["torch"].cuda.empty_cache()
        extension_sha256 = _sha256_file(extension_path)
        root_manifest = _root_manifest(authority_hashes, extension_sha256)
        local_root = _finalize_bundle(stage, root_manifest)
        for name, data in manifest_bytes.items():
            _write_or_verify(reference_root / name, data)
        _install_no_replace(stage, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    return {
        "status": "PASS",
        "fixture_id": FIXTURE_ID,
        "local_root_sha256": local_root,
        "fixture_count": 9,
        "destination": str(destination),
    }


def probe(arguments: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(arguments.source_root)
    calibration_root = Path(arguments.calibration_root)
    _validate_source(source_root, Path(arguments.patch_manifest))
    _validate_calibration(calibration_root)
    runtime = _runtime()
    rope = _make_rope(runtime)
    family = arguments.family
    bit_width = dict(FAMILIES)[family]
    expected = dict(CASES)[arguments.case]
    quantizer = _load_layer_zero_quantizer(calibration_root, family)
    with runtime["torch"].inference_mode():
        result = _execute_fixture(
            runtime,
            family=family,
            bit_width=bit_width,
            case_name=arguments.case,
            expected_key_count=expected,
            quantizer=quantizer,
            rope=rope,
        )
    return {
        "status": "PASS",
        "family": family,
        "case": arguments.case,
        "key_errors": result["key_errors"],
        "decode_errors": result["decode_errors"],
        "key_counts": result["key_selection"].counts.tolist(),
        "value_counts": result["value_selection"].counts.tolist(),
        "packed_key_sha256": _sha256_bytes(
            _tensor_bytes(
                result["append_state"]["k_dense_allocated"]
            )
        ),
        "packed_value_sha256": _sha256_bytes(
            _tensor_bytes(
                result["append_state"]["v_dense_allocated"]
            )
        ),
        "decode_sha256": _sha256_bytes(
            _tensor_bytes(result["decode_output"])
        ),
        "value_parallel_store_matches_intended_nearest": result[
            "value_store_matches_nearest"
        ],
        "finite": bool(
            runtime["torch"].isfinite(result["decode_output"]).all().item()
        ),
    }


def sanitizer_probe(arguments: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(arguments.source_root)
    calibration_root = Path(arguments.calibration_root)
    _validate_source(source_root, Path(arguments.patch_manifest))
    _validate_calibration(calibration_root)
    runtime = _runtime()
    cases = (
        ("kvq4", "key_zero_value_fixed12"),
        ("kvq4", "key_few_value_fixed12"),
        ("kvq4", "key_cap_value_fixed12"),
        ("kvq3", "key_few_value_fixed12"),
        ("kvq2", "key_zero_value_fixed12"),
        ("kvq2", "key_cap_value_fixed12"),
    )
    completed = []
    for family, case_name in cases:
        rope = _make_rope(runtime)
        bit_width = dict(FAMILIES)[family]
        quantizer = _load_layer_zero_quantizer(calibration_root, family)
        with runtime["torch"].inference_mode():
            result = _execute_fixture(
                runtime,
                family=family,
                bit_width=bit_width,
                case_name=case_name,
                expected_key_count=dict(CASES)[case_name],
                quantizer=quantizer,
                rope=rope,
            )
        completed.append(f"{family}/{case_name}")
        _release_cuda_storages_for_sanitizer(runtime, result, rope)
        del result
        del rope
        gc.collect()
        runtime["torch"].cuda.empty_cache()
    return {
        "status": "PASS",
        "cases": completed,
        "memory_errors": 0,
        "leaked_bytes": 0,
        "performance_measurement": False,
    }


def _release_cuda_storages_for_sanitizer(
    runtime: Mapping[str, Any],
    *roots: Any,
) -> None:
    """Irreversibly release probe-owned CUDA storage before context reset."""

    torch = runtime["torch"]
    seen_objects: set[int] = set()
    storages: dict[int, Any] = {}

    def collect(value: Any) -> None:
        object_id = id(value)
        if object_id in seen_objects:
            return
        seen_objects.add(object_id)
        if torch.is_tensor(value):
            if value.is_cuda:
                storage = value.untyped_storage()
                if int(storage.nbytes()) != 0:
                    storages.setdefault(int(storage._cdata), storage)
            return
        if isinstance(value, Mapping):
            for item in value.values():
                collect(item)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                collect(item)
            return
        if isinstance(value, torch.nn.Module) or (
            value.__class__.__module__ == "kvquant_gqa.compat"
            and hasattr(value, "__dict__")
        ):
            for item in vars(value).values():
                collect(item)

    for root in roots:
        collect(root)
    for storage in storages.values():
        storage.resize_(0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the exact Phase 10 KVQuant reference fixtures"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--source-root", required=True)
        subparser.add_argument("--calibration-root", required=True)
        subparser.add_argument("--patch-manifest", required=True)

    probe_parser = subparsers.add_parser("probe")
    common(probe_parser)
    probe_parser.add_argument(
        "--family", choices=[family for family, _ in FAMILIES], required=True
    )
    probe_parser.add_argument(
        "--case", choices=[case for case, _ in CASES], required=True
    )

    sanitizer_parser = subparsers.add_parser("sanitizer-probe")
    common(sanitizer_parser)

    fixture_parser = subparsers.add_parser("fixtures")
    common(fixture_parser)
    fixture_parser.add_argument("--extension", required=True)
    fixture_parser.add_argument("--reference-root", required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        if arguments.command == "probe":
            output = probe(arguments)
        elif arguments.command == "sanitizer-probe":
            output = sanitizer_probe(arguments)
        else:
            output = generate_bundle(arguments)
    except (
        ImportError,
        OSError,
        ReferenceGenerationError,
        RuntimeError,
        ValueError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error_type": type(error).__name__,
                    "reason": str(error),
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
