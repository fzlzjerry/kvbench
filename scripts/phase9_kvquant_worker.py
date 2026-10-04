#!/usr/bin/env python3
"""Container-only worker for the single frozen Phase 9 KVQuant calibration."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections.abc import Mapping, Sequence
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import runpy
import subprocess
import sys
import tempfile
from typing import Any


BASE_SEED = 20260721
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
MODEL_SNAPSHOT_MANIFEST_SHA256 = (
    "ab9f6a32a41934c9e49881db68022827b6aca35f4f644627c77e3420978d1336"
)
DATASET_REPOSITORY = "Salesforce/wikitext"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
DATASET_CONVERSION_REVISION = "3f68cd45302c7b4b532d933e71d9e6e54b1c7d5e"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_CONTENT_SHA256 = (
    "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7"
)
DATASET_SIZE_BYTES = 6_357_543
PATCH_SHA256 = "db3b6fb7ec0a72e25001e1c83a5158d86512248db5c3a06c61895598d1d482d6"
PATCHED_COMMIT = "4ad80bc8c942d0a05516d2be8f8d443a77a05900"
PATCHED_TREE = "c4f1490c9c0c4ec46099f1e95c092516df2adb4e"
METHOD_IDENTIFIER = "kvquant_gqa_upstream_patch_v1"
NUM_EXAMPLES = 16
SEQUENCE_LENGTH = 2048
LAYERS = 32
KV_WIDTH = 1024
SINK_TOKENS = 5
OUTLIER_CAP = 12
ENTRIES_PER_TAIL = 6
REPLAY_RTOL = 1e-5
REPLAY_ATOL = 1e-8


class Phase9WorkerError(RuntimeError):
    """Fail-closed calibration worker error."""


def canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_exclusive(path: str | Path, payload: Mapping[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(canonical_json_bytes(payload))


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Phase9WorkerError(f"{path} must contain a JSON object")
    return payload


def _tensor_sha256(tensor: Any) -> str:
    import torch

    contiguous = tensor.detach().cpu().contiguous().reshape(-1)
    digest = hashlib.sha256()
    step = 1 << 20
    for start in range(0, contiguous.numel(), step):
        chunk = contiguous[start : start + step].numpy()
        digest.update(memoryview(chunk).cast("B"))
    return digest.hexdigest()


def _finite_counts(tensor: Any) -> tuple[int, int]:
    import torch

    flat = tensor.detach().reshape(-1)
    nan_count = 0
    inf_count = 0
    step = 1 << 20
    for start in range(0, flat.numel(), step):
        chunk = flat[start : start + step]
        nan_count += int(torch.isnan(chunk).sum().item())
        inf_count += int(torch.isinf(chunk).sum().item())
    return nan_count, inf_count


def freeze_determinism() -> dict[str, object]:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise Phase9WorkerError("CUBLAS_WORKSPACE_CONFIG is not frozen")
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        if os.environ.get(name) != "1":
            raise Phase9WorkerError(f"{name} must equal 1")

    import numpy as np
    import torch

    random.seed(BASE_SEED)
    np.random.seed(BASE_SEED)
    torch.manual_seed(BASE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(BASE_SEED)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {
        "base_seed": BASE_SEED,
        "python_seed": BASE_SEED,
        "numpy_seed": BASE_SEED,
        "torch_cpu_seed": BASE_SEED,
        "torch_cuda_seed": BASE_SEED,
        "clustering_random_state": BASE_SEED,
        "secondary_seed_derivation": "all consumers use the exact base seed",
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "torch_deterministic_algorithms": True,
        "cublas_workspace_config": ":4096:8",
        "tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "thread_count": 1,
        "model_state": "eval",
        "bitwise_determinism_claimed": False,
        "numerical_replay": {
            "rtol": REPLAY_RTOL,
            "atol": REPLAY_ATOL,
            "frozen_before_full_calibration": True,
        },
    }


def _model_authority(arguments: argparse.Namespace) -> dict[str, object]:
    if sha256_file(arguments.model_artifact_manifest) != MODEL_SNAPSHOT_MANIFEST_SHA256:
        raise Phase9WorkerError("model snapshot manifest checksum drifted")
    return {
        "model_artifact_manifest": arguments.model_artifact_manifest,
        "model_artifact_manifest_sha256": MODEL_SNAPSHOT_MANIFEST_SHA256,
        "cache_dir": arguments.cache_dir,
    }


def _default_special_token_ids(tokenizer: Any) -> list[int]:
    encoded = tokenizer(
        "",
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )
    token_ids = encoded.get("input_ids")
    expected = [tokenizer.bos_token_id]
    if token_ids != expected:
        raise Phase9WorkerError("tokenizer special-token behavior drifted")
    return token_ids


def _snapshot_and_tokenizer_manifests(
    arguments: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, object], Any, Path]:
    from kvquant_gqa.compat import (
        load_frozen_config,
        load_frozen_tokenizer,
        resolve_frozen_snapshot,
        validate_frozen_config,
    )

    authority = _model_authority(arguments)
    snapshot_manifest = _load_json(arguments.model_artifact_manifest)
    config = load_frozen_config(**authority)
    geometry = validate_frozen_config(config)
    tokenizer = load_frozen_tokenizer(**authority)
    snapshot = resolve_frozen_snapshot(**authority)
    config_path = snapshot / "config.json"
    raw_config = _load_json(config_path)
    expected_config_sha = snapshot_manifest["files"]["config.json"]
    if sha256_file(config_path) != expected_config_sha:
        raise Phase9WorkerError("model config checksum drifted")

    model_manifest = {
        "schema_version": "kvbench-phase9-model-manifest-1.0.0",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "snapshot_manifest_sha256": MODEL_SNAPSHOT_MANIFEST_SHA256,
        "offline_local_snapshot": True,
        "config_sha256": expected_config_sha,
        "snapshot_files": snapshot_manifest["files"],
        "architecture": raw_config.get("architectures"),
        "model_type": config.model_type,
        "model_dtype": "bfloat16",
        "attention_implementation": "eager",
        "rope": {
            "rope_theta": config.rope_theta,
            "max_position_embeddings": config.max_position_embeddings,
            "rope_scaling": config.rope_scaling,
        },
        "geometry": {
            "layers": LAYERS,
            "query_heads": geometry.num_query_heads,
            "kv_heads": geometry.num_kv_heads,
            "gqa_group_size": geometry.num_kv_groups,
            "head_dimension": geometry.head_dim,
            "hidden_size": geometry.hidden_size,
            "native_kv_width": geometry.kv_width,
        },
        "key_capture": "immediately_after_k_proj_before_rope",
        "value_capture": "immediately_after_v_proj",
        "query_head_expansion": False,
    }
    tokenizer_files = {
        name: digest
        for name, digest in snapshot_manifest["files"].items()
        if name
        in {
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        }
    }
    empty_special = _default_special_token_ids(tokenizer)
    tokenizer_manifest = {
        "schema_version": "kvbench-phase9-tokenizer-manifest-1.0.0",
        "tokenizer_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "class": type(tokenizer).__name__,
        "fast": bool(tokenizer.is_fast),
        "artifact_hashes": tokenizer_files,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "default_special_token_behavior": {
            "add_special_tokens": True,
            "prepended_bos": True,
            "appended_eos": False,
            "empty_input_ids": empty_special,
        },
        "padding": False,
        "truncation": False,
    }
    return model_manifest, tokenizer_manifest, tokenizer, snapshot


def _row_character_spans(rows: list[str]) -> tuple[list[int], list[tuple[int, int]]]:
    starts: list[int] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for index, text in enumerate(rows):
        starts.append(cursor)
        spans.append((cursor, cursor + len(text)))
        cursor += len(text)
        if index + 1 < len(rows):
            cursor += 2
    return starts, spans


def _overlapping_rows(
    offsets: list[tuple[int, int]],
    row_starts: list[int],
    row_spans: list[tuple[int, int]],
) -> list[int]:
    meaningful = [(start, end) for start, end in offsets if end > start]
    if not meaningful:
        return []
    window_start = min(start for start, _ in meaningful)
    window_end = max(end for _, end in meaningful)
    first = max(0, bisect_right(row_starts, window_start) - 1)
    result: list[int] = []
    for index in range(first, len(row_spans)):
        start, end = row_spans[index]
        if start >= window_end:
            break
        if end > window_start:
            result.append(index)
    return result


def _build_frozen_tokens(
    arguments: argparse.Namespace,
) -> tuple[Any, dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    import pyarrow.parquet as pq
    import torch

    dataset_path = Path(arguments.dataset_parquet).resolve(strict=True)
    if dataset_path.stat().st_size != DATASET_SIZE_BYTES:
        raise Phase9WorkerError("WikiText-2 train parquet size drifted")
    if sha256_file(dataset_path) != DATASET_CONTENT_SHA256:
        raise Phase9WorkerError("WikiText-2 train parquet checksum drifted")

    model_manifest, tokenizer_manifest, tokenizer, _snapshot = (
        _snapshot_and_tokenizer_manifests(arguments)
    )
    table = pq.read_table(dataset_path, columns=["text"])
    rows = table.column("text").combine_chunks().to_pylist()
    if not rows or any(not isinstance(text, str) for text in rows):
        raise Phase9WorkerError("WikiText-2 train text column is invalid")
    joined = "\n\n".join(rows)
    encoded = tokenizer(
        joined,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_attention_mask=False,
        return_offsets_mapping=True,
    )
    token_ids = encoded["input_ids"]
    offsets = [tuple(item) for item in encoded["offset_mapping"]]
    if len(token_ids) != len(offsets) or token_ids[0] != tokenizer.bos_token_id:
        raise Phase9WorkerError("tokenizer output identity drifted")

    rng = random.Random(BASE_SEED)
    upper = len(token_ids) - SEQUENCE_LENGTH - 1
    if upper < 0:
        raise Phase9WorkerError("WikiText-2 token stream is too short")
    starts = [rng.randint(0, upper) for _ in range(NUM_EXAMPLES)]
    tensor = torch.tensor(
        [token_ids[start : start + SEQUENCE_LENGTH] for start in starts],
        dtype=torch.int64,
    ).contiguous()
    if tuple(tensor.shape) != (NUM_EXAMPLES, SEQUENCE_LENGTH):
        raise Phase9WorkerError("frozen token tensor shape drifted")

    row_starts, row_spans = _row_character_spans(rows)
    sample_records: list[dict[str, object]] = []
    sequence_hashes: list[str] = []
    for index, start in enumerate(starts):
        sequence = tensor[index]
        digest = _tensor_sha256(sequence)
        sequence_hashes.append(digest)
        sample_records.append(
            {
                "sample_id": f"sample-{index:02d}",
                "selection_order": index,
                "global_token_start": start,
                "global_token_stop_exclusive": start + SEQUENCE_LENGTH,
                "source_train_row_indices": _overlapping_rows(
                    offsets[start : start + SEQUENCE_LENGTH],
                    row_starts,
                    row_spans,
                ),
                "token_count": SEQUENCE_LENGTH,
                "token_ids_sha256": digest,
            }
        )
    token_content_root = sha256_bytes(
        canonical_json_bytes(
            {
                "dtype": "int64",
                "shape": [NUM_EXAMPLES, SEQUENCE_LENGTH],
                "sequence_sha256": sequence_hashes,
            }
        )
    )
    identity = {
        "source": {
            "provider": "Hugging Face Hub",
            "repository": DATASET_REPOSITORY,
            "dataset_revision": DATASET_REVISION,
            "conversion_revision": DATASET_CONVERSION_REVISION,
            "config": DATASET_CONFIG,
            "split": "train",
            "input_object": {
                "size_bytes": DATASET_SIZE_BYTES,
                "sha256": DATASET_CONTENT_SHA256,
            },
        },
        "selection": {
            "algorithm": (
                'join train rows in source order with "\\n\\n"; tokenize once; '
                "random.Random(seed).randint(0, token_count - 2048 - 1) "
                "for 16 ordered windows"
            ),
            "seed": BASE_SEED,
            "number_of_examples": NUM_EXAMPLES,
            "sequence_length": SEQUENCE_LENGTH,
            "concatenation": "\\n\\n",
            "bos": "tokenizer_default_single_bos",
            "eos": "none_appended",
            "padding": False,
            "truncation": False,
            "token_stream_length": len(token_ids),
            "train_row_count": len(rows),
            "samples": sample_records,
        },
        "tokenizer": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "artifact_hashes": tokenizer_manifest["artifact_hashes"],
        },
        "token_tensor": {
            "format": "safetensors",
            "tensor_name": "input_ids",
            "dtype": "int64",
            "shape": [NUM_EXAMPLES, SEQUENCE_LENGTH],
            "content_root_sha256": token_content_root,
            "sequence_sha256": sequence_hashes,
        },
        "test_split_loaded": False,
        "validation_split_loaded": False,
    }
    dataset_root = sha256_bytes(canonical_json_bytes(identity))
    dataset_manifest = {
        "schema_version": "kvbench-phase9-dataset-manifest-1.0.0",
        "dataset_root_sha256": dataset_root,
        **identity,
    }
    return (
        tensor,
        dataset_manifest,
        model_manifest,
        tokenizer_manifest,
        {
            "dataset_root_sha256": dataset_root,
            "token_content_root_sha256": token_content_root,
        },
    )


def _runtime_environment(
    arguments: argparse.Namespace,
    deterministic: Mapping[str, object],
) -> dict[str, object]:
    from importlib.metadata import version as package_version
    import accelerate
    import numpy
    import pandas
    import pyarrow
    import safetensors
    import scipy
    import sklearn
    import torch
    import transformers

    image_manifest = _load_json(arguments.image_manifest)
    if (
        image_manifest.get("image_config_digest")
        != "sha256:127759078f2c70c9e795c7a1bb3408df1eaee8fa019319299d283dc8075b216d"
    ):
        raise Phase9WorkerError("calibration image identity drifted")
    gpu_lines = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()
    if len(gpu_lines) != 1:
        raise Phase9WorkerError("calibration requires exactly one visible GPU")
    uuid, name, memory_mib, driver = [
        part.strip() for part in gpu_lines[0].split(",", maxsplit=3)
    ]
    return {
        "schema_version": "kvbench-phase9-environment-1.0.0",
        "calibration_container": image_manifest,
        "python": sys.version.split()[0],
        "packages": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__,
            "datasets": package_version("datasets"),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "pyarrow": pyarrow.__version__,
            "pandas": pandas.__version__,
            "safetensors": safetensors.__version__,
        },
        "gpu": {
            "uuid": uuid,
            "name": name,
            "memory_total_mib": int(memory_mib),
            "driver_version": driver,
            "cuda_visible": bool(torch.cuda.is_available()),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        },
        "precision": {
            "model_forward": "bfloat16",
            "fisher_accumulation": "float32",
            "quantizer_fitting_activations": "float16",
            "codebook_threshold_computation": "float32",
        },
        "determinism": dict(deterministic),
        "credentials_in_container": False,
        "r2_credentials_in_container": False,
        "model_weights_in_image": False,
        "patched_source_in_image": False,
    }


def command_freeze_dataset(arguments: argparse.Namespace) -> None:
    deterministic = freeze_determinism()
    tensor, dataset, model, tokenizer, roots = _build_frozen_tokens(arguments)
    from safetensors.torch import save_file

    output = Path(arguments.output_root)
    output.mkdir(parents=True, exist_ok=True)
    token_path = output / "tokens" / "input_ids.safetensors"
    token_path.parent.mkdir(parents=True, exist_ok=False)
    save_file({"input_ids": tensor}, str(token_path))
    token_sha = sha256_file(token_path)
    dataset["token_tensor"]["file_sha256"] = token_sha
    dataset["token_tensor"]["path"] = "tokens/input_ids.safetensors"
    write_json_exclusive(output / "dataset_manifest.json", dataset)
    write_json_exclusive(output / "model_manifest.json", model)
    write_json_exclusive(output / "tokenizer_manifest.json", tokenizer)
    write_json_exclusive(
        output / "environment.json",
        _runtime_environment(arguments, deterministic),
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                **roots,
                "token_tensor_sha256": token_sha,
            },
            sort_keys=True,
        )
    )


def _run_upstream(path: Path, argv: list[str], source_python_path: Path) -> None:
    sys.path.insert(0, str(source_python_path))
    original = sys.argv
    try:
        sys.argv = [str(path), *argv]
        runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv = original


def command_run_fisher(arguments: argparse.Namespace) -> None:
    freeze_determinism()
    source = Path(arguments.source_root).resolve(strict=True)
    script = source / "gradients" / "run-fisher.py"
    if not script.is_file():
        raise Phase9WorkerError("patched Fisher entry point is absent")
    argv = [
        "--model-artifact-manifest",
        arguments.model_artifact_manifest,
        "--model-artifact-manifest-sha256",
        MODEL_SNAPSHOT_MANIFEST_SHA256,
        "--model-name-or-path",
        MODEL_ID,
        "--model-revision",
        MODEL_REVISION,
        "--tokenizer-name-or-path",
        MODEL_ID,
        "--tokenizer-revision",
        MODEL_REVISION,
        "--cache-dir",
        arguments.cache_dir,
        "--dataset",
        "wikitext2",
        "--num-examples",
        str(NUM_EXAMPLES),
        "--seqlen",
        str(SEQUENCE_LENGTH),
        "--maxseqlen",
        str(SEQUENCE_LENGTH),
        "--selection-seed",
        str(BASE_SEED),
        "--token-tensor-path",
        arguments.token_tensor,
        "--model-max-length",
        str(SEQUENCE_LENGTH),
        "--output-dir",
        arguments.output_dir,
        "--report-to",
        "none",
        "--disable-tqdm",
        "false",
    ]
    _run_upstream(script, argv, source / "gradients")


def _expected_fisher_names() -> list[str]:
    return [
        f"model.layers.{layer}.self_attn.{projection}.weight"
        for layer in range(LAYERS)
        for projection in ("k_proj", "v_proj")
    ]


def command_fisher_manifest(arguments: argparse.Namespace) -> None:
    freeze_determinism()
    from safetensors import safe_open
    import torch

    fisher_path = Path(arguments.fisher).resolve(strict=True)
    tensors: list[dict[str, object]] = []
    with safe_open(fisher_path, framework="pt", device="cpu") as handle:
        names = sorted(handle.keys())
        expected = sorted(_expected_fisher_names())
        if names != expected:
            raise Phase9WorkerError("Fisher K/V layer coverage is incomplete")
        for name in names:
            tensor = handle.get_tensor(name)
            if tensor.dtype != torch.float32:
                raise Phase9WorkerError(f"{name} is not FP32")
            if tuple(tensor.shape) != (1, NUM_EXAMPLES * SEQUENCE_LENGTH, KV_WIDTH):
                raise Phase9WorkerError(f"{name} has an invalid native-GQA shape")
            nan_count, inf_count = _finite_counts(tensor)
            if nan_count or inf_count:
                raise Phase9WorkerError(f"{name} contains NaN or Inf")
            layer = int(name.split(".")[2])
            role = "K" if ".k_proj." in name else "V"
            tensors.append(
                {
                    "name": name,
                    "layer_index": layer,
                    "tensor_role": role,
                    "capture": (
                        "pre_rope_key" if role == "K" else "post_v_proj_native_hkv"
                    ),
                    "shape": list(tensor.shape),
                    "dtype": "float32",
                    "sha256": _tensor_sha256(tensor),
                    "nan_count": nan_count,
                    "inf_count": inf_count,
                }
            )
            del tensor
    file_digest = sha256_file(fisher_path)
    write_json_exclusive(
        arguments.output,
        {
            "schema_version": "kvbench-phase9-fisher-manifest-1.0.0",
            "path": "fisher/fisher.safetensors",
            "file_sha256": file_digest,
            "layer_count": LAYERS,
            "k_coverage": LAYERS,
            "v_coverage": LAYERS,
            "tensor_count": len(tensors),
            "model_forward_dtype": "bfloat16",
            "gradient_accumulation_dtype": "float32",
            "native_kv_heads": 8,
            "native_kv_width": KV_WIDTH,
            "query_head_expansion": False,
            "model_weight_mutation": False,
            "tensors": tensors,
        },
    )
    print(json.dumps({"status": "PASS", "fisher_root": file_digest}))


def _trusted_convert_quantizer(
    native_path: Path,
    safe_path: Path,
    manifest_path: Path,
    bit_width: int,
) -> None:
    import numpy as np
    from safetensors.torch import save_file
    import torch

    native_sha = sha256_file(native_path)
    native_size = native_path.stat().st_size
    with native_path.open("rb") as handle:
        payload = pickle.load(handle)
    if (
        not isinstance(payload, dict)
        or set(payload) != {"format", "provenance", "quantizers"}
        or payload["format"] != "kvquant_gqa_quantizer_v1"
    ):
        raise Phase9WorkerError("trusted source-native quantizer payload is invalid")
    provenance = payload["provenance"]
    if (
        not isinstance(provenance, dict)
        or provenance.get("bit_width") != bit_width
        or provenance.get("method_identifier") != METHOD_IDENTIFIER
        or provenance.get("patch_digest") != PATCH_SHA256
        or provenance.get("patched_commit") != PATCHED_COMMIT
        or provenance.get("patched_tree") != PATCHED_TREE
        or provenance.get("nuq") is not True
        or provenance.get("dense_and_sparse") is not True
        or provenance.get("key_outlier_cap") != OUTLIER_CAP
        or provenance.get("value_outlier_cap") != OUTLIER_CAP
        or provenance.get("sink_tokens") != SINK_TOKENS
    ):
        raise Phase9WorkerError("source-native quantizer provenance drifted")
    quantizers = payload["quantizers"]
    expected = {
        f"model.layers.{layer}.self_attn.{projection}"
        for layer in range(LAYERS)
        for projection in ("k_proj", "v_proj")
    }
    if not isinstance(quantizers, dict) or set(quantizers) != expected:
        raise Phase9WorkerError("source-native quantizer coverage is incomplete")

    safe_tensors: dict[str, Any] = {}
    tensor_records: list[dict[str, object]] = []
    for name in sorted(expected):
        value = quantizers[name]
        if not isinstance(value, tuple) or len(value) != 3:
            raise Phase9WorkerError(f"quantizer {name} has an invalid tuple")
        layer = int(name.split(".")[2])
        role = "k" if name.endswith("k_proj") else "v"
        upper = torch.as_tensor(value[0], dtype=torch.float32).cpu().contiguous()
        lower = torch.as_tensor(value[1], dtype=torch.float32).cpu().contiguous()
        centroids = value[2]
        if (
            not isinstance(centroids, list)
            or len(centroids) != 1
            or not isinstance(centroids[0], np.ndarray)
        ):
            raise Phase9WorkerError(f"quantizer {name} has an invalid NUQ codebook")
        codebook = torch.as_tensor(
            centroids[0], dtype=torch.float32
        ).cpu().contiguous()
        threshold_shape = (1, KV_WIDTH) if role == "k" else (
            NUM_EXAMPLES * SEQUENCE_LENGTH,
            1,
        )
        if tuple(upper.shape) != threshold_shape or tuple(lower.shape) != threshold_shape:
            raise Phase9WorkerError(f"quantizer {name} threshold shape drifted")
        if tuple(codebook.shape) != (2**bit_width, 1):
            raise Phase9WorkerError(f"quantizer {name} codebook shape drifted")
        range_tensor = ((upper - lower) / 2).contiguous()
        offset_tensor = ((upper + lower) / 2).contiguous()
        values = {
            "upper_threshold": upper,
            "lower_threshold": lower,
            "range": range_tensor,
            "offset": offset_tensor,
            "codebook": codebook,
        }
        for component, tensor in values.items():
            nan_count, inf_count = _finite_counts(tensor)
            if nan_count or inf_count:
                raise Phase9WorkerError(
                    f"quantizer {name} {component} contains NaN or Inf"
                )
            tensor_name = f"layer_{layer:02d}.{role}.{component}"
            safe_tensors[tensor_name] = tensor
            tensor_records.append(
                {
                    "tensor_name": tensor_name,
                    "source_name": name,
                    "layer_index": layer,
                    "tensor_role": role.upper(),
                    "component": component,
                    "shape": list(tensor.shape),
                    "dtype": "float32",
                    "sha256": _tensor_sha256(tensor),
                }
            )
    safe_path.parent.mkdir(parents=True, exist_ok=True)
    if safe_path.exists():
        raise FileExistsError(safe_path)
    metadata = {
        "format": "kvbench-kvquant-safe-v1",
        "method_identifier": METHOD_IDENTIFIER,
        "bit_width": str(bit_width),
        "fisher_root": str(provenance["fisher_root"]),
        "dataset_root": str(provenance["dataset_root"]),
        "token_tensor_sha256": str(provenance["token_tensor_sha256"]),
        "patch_sha256": PATCH_SHA256,
        "patched_tree": PATCHED_TREE,
    }
    save_file(
        {name: safe_tensors[name] for name in sorted(safe_tensors)},
        str(safe_path),
        metadata=metadata,
    )
    safe_sha = sha256_file(safe_path)
    write_json_exclusive(
        manifest_path,
        {
            "schema_version": "kvbench-phase9-quantizer-manifest-1.0.0",
            "variant_id": f"kvq{bit_width}",
            "bit_width": bit_width,
            "safe_format": "safetensors",
            "safe_path": f"quantizers/kvq{bit_width}.safetensors",
            "safe_sha256": safe_sha,
            "source_native_format": "trusted_container_pickle",
            "source_native_pickle_sha256": native_sha,
            "source_native_pickle_size_bytes": native_size,
            "source_native_pickle_published": False,
            "source_native_pickle_retained": False,
            "provenance": provenance,
            "tensor_count": len(tensor_records),
            "tensors": tensor_records,
        },
    )
    native_path.unlink()


def command_run_quantizer(arguments: argparse.Namespace) -> None:
    freeze_determinism()
    source = Path(arguments.source_root).resolve(strict=True)
    script = source / "quant" / "llama_simquant.py"
    if not script.is_file():
        raise Phase9WorkerError("patched quantizer entry point is absent")
    bit_width = arguments.bit_width
    with tempfile.TemporaryDirectory(prefix=f"kvq{bit_width}-native-") as temporary:
        native = Path(temporary) / f"kvq{bit_width}.pickle"
        argv = [
            MODEL_ID,
            "--model-revision",
            MODEL_REVISION,
            "--tokenizer",
            MODEL_ID,
            "--tokenizer-revision",
            MODEL_REVISION,
            "--cache-dir",
            arguments.cache_dir,
            "--model-artifact-manifest",
            arguments.model_artifact_manifest,
            "--model-artifact-manifest-sha256",
            MODEL_SNAPSHOT_MANIFEST_SHA256,
            "--token-tensor-path",
            arguments.token_tensor,
            "--token-tensor-sha256",
            arguments.token_tensor_sha256,
            "--dataset-root",
            arguments.dataset_root,
            "--fisher-root",
            arguments.fisher_root,
            "--patched-commit",
            PATCHED_COMMIT,
            "--patched-tree",
            PATCHED_TREE,
            "--patch-digest",
            PATCH_SHA256,
            "--seed",
            str(BASE_SEED),
            "--nsamples",
            str(NUM_EXAMPLES),
            "--quantize",
            "--abits",
            str(bit_width),
            "--nuq",
            "--perchannel",
            '["k_proj"]',
            "--pertoken",
            '["v_proj"]',
            "--include_sparse",
            "--sparsity-threshold",
            "0.99",
            "--quantizer-path",
            str(native),
            "--fisher",
            arguments.fisher,
            "--seqlen",
            str(SEQUENCE_LENGTH),
            "--maxseqlen",
            str(SEQUENCE_LENGTH),
            "--dataset",
            "wikitext2",
            "--cap_outliers",
            str(OUTLIER_CAP),
            "--first_few_fp16",
            str(SINK_TOKENS),
        ]
        _run_upstream(script, argv, source / "quant")
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass
        _trusted_convert_quantizer(
            native,
            Path(arguments.output_safe),
            Path(arguments.output_manifest),
            bit_width,
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "variant_id": f"kvq{bit_width}",
                "safe_sha256": sha256_file(arguments.output_safe),
            },
            sort_keys=True,
        )
    )


def _load_quantizer_thresholds(
    paths: Mapping[int, Path],
) -> tuple[dict[tuple[int, int, str], tuple[Any, Any]], dict[tuple[int, int, str], str]]:
    from safetensors.torch import load_file

    thresholds: dict[tuple[int, int, str], tuple[Any, Any]] = {}
    codebook_hashes: dict[tuple[int, int, str], str] = {}
    for bit_width, path in paths.items():
        payload = load_file(str(path), device="cpu")
        for layer in range(LAYERS):
            for role in ("k", "v"):
                prefix = f"layer_{layer:02d}.{role}"
                lower = payload[f"{prefix}.lower_threshold"]
                upper = payload[f"{prefix}.upper_threshold"]
                thresholds[(bit_width, layer, role)] = (lower, upper)
                codebook_hashes[(bit_width, layer, role)] = _tensor_sha256(
                    payload[f"{prefix}.codebook"]
                )
    return thresholds, codebook_hashes


def command_layer_stats(arguments: argparse.Namespace) -> None:
    freeze_determinism()
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from kvquant_gqa.compat import (
        load_frozen_model,
        load_frozen_token_tensors,
    )

    authority = _model_authority(arguments)
    model = load_frozen_model(**authority)
    model.config.use_cache = False
    model.eval()
    model.cuda()
    token_ids = load_frozen_token_tensors(
        arguments.token_tensor,
        num_examples=NUM_EXAMPLES,
        sequence_length=SEQUENCE_LENGTH,
    )
    quantizer_paths = {
        4: Path(arguments.kvq4).resolve(strict=True),
        3: Path(arguments.kvq3).resolve(strict=True),
        2: Path(arguments.kvq2).resolve(strict=True),
    }
    cpu_thresholds, codebook_hashes = _load_quantizer_thresholds(quantizer_paths)
    thresholds = {
        key: (lower.cuda(), upper.cuda())
        for key, (lower, upper) in cpu_thresholds.items()
    }
    accumulators: dict[tuple[int, int, str], dict[str, object]] = {}
    for bit_width in (4, 3, 2):
        for layer in range(LAYERS):
            for role in ("k", "v"):
                accumulators[(bit_width, layer, role)] = {
                    "histogram": [0] * (OUTLIER_CAP + 1),
                    "row_count": 0,
                    "eligible_row_count": 0,
                    "sink_row_count": 0,
                    "cap_hit_count": 0,
                    "selected_outlier_count": 0,
                    "clipping_saturation_count": 0,
                    "nan_count": 0,
                    "inf_count": 0,
                }

    current_sample = 0
    handles: list[Any] = []

    def capture(layer: int, role: str):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            values = output.detach().reshape(-1, KV_WIDTH).float()
            rows = values.shape[0]
            if rows != SEQUENCE_LENGTH:
                raise Phase9WorkerError("layer-stat activation row count drifted")
            sink = torch.zeros(rows, dtype=torch.bool, device=values.device)
            sink[:SINK_TOKENS] = True
            for bit_width in (4, 3, 2):
                lower, upper = thresholds[(bit_width, layer, role)]
                if role == "v":
                    start = current_sample * SEQUENCE_LENGTH
                    lower_used = lower[start : start + SEQUENCE_LENGTH]
                    upper_used = upper[start : start + SEQUENCE_LENGTH]
                else:
                    lower_used = lower
                    upper_used = upper
                lower_count = (values < lower_used).sum(dim=1)
                upper_count = (values > upper_used).sum(dim=1)
                selected = lower_count.clamp(max=ENTRIES_PER_TAIL)
                selected = selected + upper_count.clamp(max=ENTRIES_PER_TAIL)
                selected[sink] = 0
                overflow = (lower_count - ENTRIES_PER_TAIL).clamp(min=0)
                overflow = overflow + (upper_count - ENTRIES_PER_TAIL).clamp(min=0)
                overflow[sink] = 0
                histogram = torch.bincount(
                    selected.to(dtype=torch.int64),
                    minlength=OUTLIER_CAP + 1,
                )[: OUTLIER_CAP + 1]
                record = accumulators[(bit_width, layer, role)]
                record["histogram"] = [
                    left + int(right)
                    for left, right in zip(
                        record["histogram"],
                        histogram.cpu().tolist(),
                        strict=True,
                    )
                ]
                record["row_count"] += rows
                record["eligible_row_count"] += rows - SINK_TOKENS
                record["sink_row_count"] += SINK_TOKENS
                record["cap_hit_count"] += int(
                    ((selected == OUTLIER_CAP) & ~sink).sum().item()
                )
                record["selected_outlier_count"] += int(selected.sum().item())
                record["clipping_saturation_count"] += int(overflow.sum().item())
                record["nan_count"] += int(torch.isnan(values).sum().item())
                record["inf_count"] += int(torch.isinf(values).sum().item())

        return hook

    for layer, module in enumerate(model.model.layers):
        handles.append(module.self_attn.k_proj.register_forward_hook(capture(layer, "k")))
        handles.append(module.self_attn.v_proj.register_forward_hook(capture(layer, "v")))
    try:
        with torch.inference_mode():
            for sample_index, sample in enumerate(token_ids):
                current_sample = sample_index
                model.model(
                    input_ids=sample.unsqueeze(0).cuda(non_blocking=False),
                    use_cache=False,
                    return_dict=False,
                )
    finally:
        for handle in handles:
            handle.remove()

    fisher_manifest = _load_json(arguments.fisher_manifest)
    fisher_by_pair = {
        (record["layer_index"], record["tensor_role"]): record["sha256"]
        for record in fisher_manifest["tensors"]
    }
    rows: list[dict[str, object]] = []
    for layer in range(LAYERS):
        for role in ("k", "v"):
            for bit_width in (4, 3, 2):
                record = accumulators[(bit_width, layer, role)]
                if record["nan_count"] or record["inf_count"]:
                    raise Phase9WorkerError("layer-stat activations contain NaN or Inf")
                if record["row_count"] != NUM_EXAMPLES * SEQUENCE_LENGTH:
                    raise Phase9WorkerError("layer-stat coverage is incomplete")
                lower, upper = cpu_thresholds[(bit_width, layer, role)]
                threshold_summary = {
                    "lower_min": float(lower.min().item()),
                    "lower_max": float(lower.max().item()),
                    "lower_mean": float(lower.mean().item()),
                    "upper_min": float(upper.min().item()),
                    "upper_max": float(upper.max().item()),
                    "upper_mean": float(upper.mean().item()),
                }
                eligible = int(record["eligible_row_count"])
                role_upper = role.upper()
                rows.append(
                    {
                        "layer_index": layer,
                        "tensor_role": role_upper,
                        "bit_width": bit_width,
                        "tensor_shape": json.dumps(
                            [NUM_EXAMPLES * SEQUENCE_LENGTH, KV_WIDTH],
                            separators=(",", ":"),
                        ),
                        "fisher_checksum": fisher_by_pair[(layer, role_upper)],
                        "codebook_checksum": codebook_hashes[
                            (bit_width, layer, role)
                        ],
                        "threshold_summary": json.dumps(
                            threshold_summary,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "observed_outlier_count_distribution": json.dumps(
                            record["histogram"],
                            separators=(",", ":"),
                        ),
                        "cap": OUTLIER_CAP,
                        "cap_hit_count": int(record["cap_hit_count"]),
                        "cap_hit_rate": (
                            float(record["cap_hit_count"]) / eligible
                            if eligible
                            else 0.0
                        ),
                        "clipping_saturation_count": int(
                            record["clipping_saturation_count"]
                        ),
                        "sink_token_policy": (
                            "first_5_tokens_per_sequence_fp16_excluded_from_dense_and_sparse"
                        ),
                        "metadata_dtype": "float32",
                        "value_dtype": "float32",
                        "index_dtype": "int32",
                        "nan_count": int(record["nan_count"]),
                        "inf_count": int(record["inf_count"]),
                    }
                )
    if len(rows) != 192:
        raise Phase9WorkerError("layer-stat row coverage is incomplete")
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    table = pa.Table.from_pylist(rows)
    pq.write_table(
        table,
        output,
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "rows": len(rows),
                "sha256": sha256_file(output),
            },
            sort_keys=True,
        )
    )


def command_reconstruct_tokens(arguments: argparse.Namespace) -> None:
    freeze_determinism()
    from safetensors.torch import load_file, save_file
    import torch

    tensor, reconstructed, _model, _tokenizer, _roots = _build_frozen_tokens(arguments)
    expected_manifest = _load_json(arguments.expected_dataset_manifest)
    expected_tensor = load_file(arguments.expected_token_tensor, device="cpu")[
        "input_ids"
    ]
    manifest_equal = reconstructed == {
        key: value
        for key, value in expected_manifest.items()
        if key != "token_tensor"
    } | {
        "token_tensor": {
            key: value
            for key, value in expected_manifest["token_tensor"].items()
            if key not in {"file_sha256", "path"}
        }
    }
    tensor_equal = torch.equal(tensor, expected_tensor)
    with tempfile.TemporaryDirectory(prefix="phase9-token-reconstruction-") as temporary:
        rebuilt = Path(temporary) / "input_ids.safetensors"
        save_file({"input_ids": tensor}, str(rebuilt))
        file_equal = sha256_file(rebuilt) == sha256_file(arguments.expected_token_tensor)
    result = {
        "schema_version": "kvbench-phase9-token-reconstruction-1.0.0",
        "status": (
            "PASS" if manifest_equal and tensor_equal and file_equal else "FAIL"
        ),
        "manifest_identity_equal": manifest_equal,
        "tensor_bitwise_equal": tensor_equal,
        "safe_serialization_byte_equal": file_equal,
        "expected_token_tensor_sha256": sha256_file(
            arguments.expected_token_tensor
        ),
        "reconstructed_content_root_sha256": reconstructed["token_tensor"][
            "content_root_sha256"
        ],
        "test_split_loaded": False,
    }
    write_json_exclusive(arguments.output, result)
    if result["status"] != "PASS":
        raise Phase9WorkerError("frozen token reconstruction failed")
    print(json.dumps(result, sort_keys=True))


def command_replay_fisher(arguments: argparse.Namespace) -> None:
    freeze_determinism()
    import torch
    from safetensors.torch import load_file
    from kvquant_gqa.compat import (
        NativeKVHookCapture,
        assert_model_weights_unchanged,
        freeze_and_snapshot_model_weights,
        load_frozen_config,
        load_frozen_model,
        load_frozen_token_tensors,
        validate_frozen_config,
    )

    authority = _model_authority(arguments)
    config = load_frozen_config(**authority)
    geometry = validate_frozen_config(config)
    model = load_frozen_model(**authority)
    model.config.use_cache = False
    model.eval()
    model.cuda()
    weight_snapshot = freeze_and_snapshot_model_weights(model)
    token_ids = load_frozen_token_tensors(
        arguments.token_tensor,
        num_examples=NUM_EXAMPLES,
        sequence_length=SEQUENCE_LENGTH,
    )
    batch = token_ids[0].unsqueeze(0).cuda(non_blocking=False)
    replay: dict[str, Any] = {}
    with NativeKVHookCapture(
        [model.model.layers[0]],
        geometry,
        require_layer_count=None,
    ) as capture:
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            inputs_embeds = model.get_input_embeddings()(batch)
        inputs_embeds = inputs_embeds.detach().requires_grad_(True)
        outputs = model(
            inputs_embeds=inputs_embeds,
            labels=batch,
            use_cache=False,
        )
        outputs.loss.backward()
        fisher = load_file(arguments.fisher, device="cpu")
        for role, projection in (("K", "k_proj"), ("V", "v_proj")):
            capture_role = "key_pre_rope" if role == "K" else "value"
            actual = capture.gradient_square(0, capture_role).detach().cpu()
            name = f"model.layers.0.self_attn.{projection}.weight"
            expected = fisher[name][:, :SEQUENCE_LENGTH, :]
            exact = torch.equal(actual, expected)
            equivalent = torch.allclose(
                actual,
                expected,
                rtol=REPLAY_RTOL,
                atol=REPLAY_ATOL,
            )
            difference = (actual - expected).abs()
            denominator = expected.abs().clamp_min(REPLAY_ATOL)
            replay[role] = {
                "layer_index": 0,
                "capture": (
                    "pre_rope_key" if role == "K" else "post_v_proj_native_hkv"
                ),
                "shape": list(actual.shape),
                "dtype": "float32",
                "exact": exact,
                "numerically_equivalent": equivalent,
                "rtol": REPLAY_RTOL,
                "atol": REPLAY_ATOL,
                "max_absolute_difference": float(difference.max().item()),
                "max_relative_difference": float(
                    (difference / denominator).max().item()
                ),
                "replay_sha256": _tensor_sha256(actual),
                "reference_sha256": _tensor_sha256(expected),
            }
            if not equivalent:
                raise Phase9WorkerError(f"representative Fisher replay failed for {role}")
    assert_model_weights_unchanged(model, weight_snapshot)
    result = {
        "schema_version": "kvbench-phase9-fisher-replay-1.0.0",
        "status": "PASS",
        "representative_layers": replay,
        "model_weight_mutation": False,
        "complete_fisher_rerun": False,
    }
    write_json_exclusive(arguments.output, result)
    print(json.dumps(result, sort_keys=True))


def command_policy_check(arguments: argparse.Namespace) -> None:
    freeze_determinism()
    import torch
    from kvquant_gqa.compat import select_fixed_outliers

    values = torch.zeros((3, KV_WIDTH), dtype=torch.float32)
    lower_indices = [9, 3, 7, 1, 5, 11]
    upper_indices = [20, 18, 16, 14, 12, 10]
    values[0, lower_indices] = -1.0
    values[0, upper_indices] = 1.0
    values[1, [4, 6]] = -1.0
    values[1, [8, 10]] = 1.0
    values[2, :] = 2.0
    sink = torch.tensor([False, False, True])
    first = select_fixed_outliers(
        values,
        cap=OUTLIER_CAP,
        lower_threshold=-0.5,
        upper_threshold=0.5,
        sink_row_mask=sink,
    )
    second = select_fixed_outliers(
        values,
        cap=OUTLIER_CAP,
        lower_threshold=-0.5,
        upper_threshold=0.5,
        sink_row_mask=sink,
    )
    expected_first = [1, 3, 5, 7, 9, 11, 10, 12, 14, 16, 18, 20]
    checks = {
        "repeat_values_equal": torch.equal(first.values, second.values),
        "repeat_indices_equal": torch.equal(first.indices, second.indices),
        "repeat_counts_equal": torch.equal(first.counts, second.counts),
        "lexicographic_equal_ties": (
            first.indices[0].tolist() == expected_first
        ),
        "six_lower_six_upper": int(first.counts[0].item()) == OUTLIER_CAP,
        "no_duplicate_or_overlap": (
            len(set(first.indices[0].tolist())) == OUTLIER_CAP
        ),
        "unused_values_zero": bool(torch.equal(first.values[1, 4:], torch.zeros(8))),
        "unused_indices_zero": bool(
            torch.equal(first.indices[1, 4:], torch.zeros(8, dtype=torch.int32))
        ),
        "sink_excluded": int(first.counts[2].item()) == 0,
        "value_dtype_float32": first.values.dtype == torch.float32,
        "index_dtype_int32": first.indices.dtype == torch.int32,
        "capacity_12": first.values.shape[-1] == OUTLIER_CAP,
        "no_row_exceeds_capacity": bool((first.counts <= OUTLIER_CAP).all()),
    }
    if not all(checks.values()):
        raise Phase9WorkerError("fixed sparse policy check failed")
    result = {
        "schema_version": "kvbench-phase9-outlier-policy-check-1.0.0",
        "status": "PASS",
        "checks": checks,
    }
    write_json_exclusive(arguments.output, result)
    print(json.dumps(result, sort_keys=True))


def _validate_safe_quantizer(path: Path, bit_width: int) -> None:
    from safetensors import safe_open
    import torch

    expected: set[str] = set()
    for layer in range(LAYERS):
        for role in ("k", "v"):
            for component in (
                "upper_threshold",
                "lower_threshold",
                "range",
                "offset",
                "codebook",
            ):
                expected.add(f"layer_{layer:02d}.{role}.{component}")
    with safe_open(path, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != expected:
            raise Phase9WorkerError(f"kvq{bit_width} safe tensor coverage drifted")
        metadata = handle.metadata()
        if (
            metadata.get("format") != "kvbench-kvquant-safe-v1"
            or metadata.get("bit_width") != str(bit_width)
        ):
            raise Phase9WorkerError(f"kvq{bit_width} safe metadata drifted")
        for name in sorted(expected):
            tensor = handle.get_tensor(name)
            if tensor.dtype != torch.float32:
                raise Phase9WorkerError(f"{name} is not float32")
            if name.endswith(".codebook"):
                shape = (2**bit_width, 1)
            elif ".k." in name:
                shape = (1, KV_WIDTH)
            else:
                shape = (NUM_EXAMPLES * SEQUENCE_LENGTH, 1)
            if tuple(tensor.shape) != shape:
                raise Phase9WorkerError(f"{name} shape drifted")
            nan_count, inf_count = _finite_counts(tensor)
            if nan_count or inf_count:
                raise Phase9WorkerError(f"{name} contains NaN or Inf")


def _compare_safe_quantizers(
    original_path: Path,
    regenerated_path: Path,
) -> dict[str, object]:
    from safetensors import safe_open
    import torch

    original_sha256 = sha256_file(original_path)
    regenerated_sha256 = sha256_file(regenerated_path)
    byte_equal = (
        original_path.stat().st_size == regenerated_path.stat().st_size
        and original_sha256 == regenerated_sha256
    )
    tensor_records: list[dict[str, object]] = []
    shape_dtype_equal = True
    exact_count = 0
    equivalent_count = 0
    max_absolute_difference = 0.0
    max_relative_difference = 0.0
    worst_absolute_tensor: str | None = None
    worst_relative_tensor: str | None = None
    with (
        safe_open(original_path, framework="pt", device="cpu") as original,
        safe_open(regenerated_path, framework="pt", device="cpu") as regenerated,
    ):
        original_keys = sorted(original.keys())
        regenerated_keys = sorted(regenerated.keys())
        key_set_equal = original_keys == regenerated_keys
        metadata_equal = original.metadata() == regenerated.metadata()
        if key_set_equal:
            for name in original_keys:
                reference = original.get_tensor(name)
                candidate = regenerated.get_tensor(name)
                same_shape = tuple(reference.shape) == tuple(candidate.shape)
                same_dtype = reference.dtype == candidate.dtype
                exact = False
                equivalent = False
                tensor_max_absolute = 0.0
                tensor_max_relative = 0.0
                if same_shape and same_dtype:
                    exact = torch.equal(reference, candidate)
                    equivalent = exact or torch.allclose(
                        reference,
                        candidate,
                        rtol=REPLAY_RTOL,
                        atol=REPLAY_ATOL,
                    )
                    difference = (reference - candidate).abs()
                    denominator = reference.abs().clamp_min(REPLAY_ATOL)
                    tensor_max_absolute = float(difference.max().item())
                    tensor_max_relative = float(
                        (difference / denominator).max().item()
                    )
                    exact_count += int(exact)
                    equivalent_count += int(equivalent)
                    if tensor_max_absolute > max_absolute_difference:
                        max_absolute_difference = tensor_max_absolute
                        worst_absolute_tensor = name
                    if tensor_max_relative > max_relative_difference:
                        max_relative_difference = tensor_max_relative
                        worst_relative_tensor = name
                else:
                    shape_dtype_equal = False
                tensor_records.append(
                    {
                        "name": name,
                        "shape": list(reference.shape),
                        "dtype": str(reference.dtype).removeprefix("torch."),
                        "shape_equal": same_shape,
                        "dtype_equal": same_dtype,
                        "exact": exact,
                        "numerically_equivalent": equivalent,
                        "max_absolute_difference": tensor_max_absolute,
                        "max_relative_difference": tensor_max_relative,
                    }
                )
        else:
            shape_dtype_equal = False
    tensor_count = len(tensor_records)
    all_exact = key_set_equal and exact_count == tensor_count
    all_equivalent = (
        key_set_equal
        and shape_dtype_equal
        and equivalent_count == tensor_count
    )
    status = (
        "PASS"
        if metadata_equal and all_equivalent
        else "FAIL"
    )
    return {
        "status": status,
        "file_byte_equal": byte_equal,
        "original_file_sha256": original_sha256,
        "regenerated_file_sha256": regenerated_sha256,
        "key_set_equal": key_set_equal,
        "metadata_equal": metadata_equal,
        "shape_dtype_equal": shape_dtype_equal,
        "tensor_count": tensor_count,
        "exact_tensor_count": exact_count,
        "numerically_equivalent_tensor_count": equivalent_count,
        "all_tensors_exact": all_exact,
        "all_tensors_numerically_equivalent": all_equivalent,
        "rtol": REPLAY_RTOL,
        "atol": REPLAY_ATOL,
        "max_absolute_difference": max_absolute_difference,
        "max_relative_difference": max_relative_difference,
        "worst_absolute_tensor": worst_absolute_tensor,
        "worst_relative_tensor": worst_relative_tensor,
        "tensors": tensor_records,
    }


def command_compare_quantizers(arguments: argparse.Namespace) -> None:
    freeze_determinism()
    original = Path(arguments.original).resolve(strict=True)
    regenerated = Path(arguments.regenerated).resolve(strict=True)
    _validate_safe_quantizer(original, arguments.bit_width)
    _validate_safe_quantizer(regenerated, arguments.bit_width)
    result = _compare_safe_quantizers(original, regenerated)
    result["schema_version"] = "kvbench-phase9-quantizer-comparison-1.0.0"
    result["variant_id"] = f"kvq{arguments.bit_width}"
    write_json_exclusive(arguments.output, result)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PASS":
        raise Phase9WorkerError(
            f"kvq{arguments.bit_width} regenerated safe tensors are not equivalent"
        )


def command_validate_payloads(arguments: argparse.Namespace) -> None:
    freeze_determinism()
    import pyarrow.parquet as pq
    from safetensors import safe_open
    import torch

    root = Path(arguments.bundle).resolve(strict=True)
    required = {
        "authority_manifest.json",
        "calibration_config.json",
        "dataset_manifest.json",
        "environment.json",
        "fisher/fisher.safetensors",
        "fisher_manifest.json",
        "layer_stats.parquet",
        "model_manifest.json",
        "outlier_policy.json",
        "quantizers/kvq2.safetensors",
        "quantizers/kvq3.safetensors",
        "quantizers/kvq4.safetensors",
        "reproducibility/fisher_replay.json",
        "reproducibility/outlier_policy.json",
        "reproducibility/quantizer_regeneration.json",
        "reproducibility/token_reconstruction.json",
        "tokenizer_manifest.json",
        "tokens/input_ids.safetensors",
    }
    missing = sorted(relative for relative in required if not (root / relative).is_file())
    if missing:
        raise Phase9WorkerError(f"calibration payloads are incomplete: {missing}")
    dataset = _load_json(root / "dataset_manifest.json")
    if (
        dataset.get("test_split_loaded") is not False
        or dataset.get("validation_split_loaded") is not False
        or dataset["source"].get("split") != "train"
    ):
        raise Phase9WorkerError("dataset split contract drifted")
    with safe_open(
        root / "tokens/input_ids.safetensors",
        framework="pt",
        device="cpu",
    ) as handle:
        tokens = handle.get_tensor("input_ids")
        if tokens.dtype != torch.int64 or tuple(tokens.shape) != (
            NUM_EXAMPLES,
            SEQUENCE_LENGTH,
        ):
            raise Phase9WorkerError("token tensor contract drifted")
    fisher_manifest = _load_json(root / "fisher_manifest.json")
    if (
        fisher_manifest.get("tensor_count") != 64
        or fisher_manifest.get("k_coverage") != LAYERS
        or fisher_manifest.get("v_coverage") != LAYERS
        or fisher_manifest.get("file_sha256")
        != sha256_file(root / "fisher/fisher.safetensors")
        or any(
            record.get("nan_count") or record.get("inf_count")
            for record in fisher_manifest.get("tensors", [])
        )
    ):
        raise Phase9WorkerError("Fisher manifest contract drifted")
    for bit_width in (4, 3, 2):
        _validate_safe_quantizer(
            root / f"quantizers/kvq{bit_width}.safetensors",
            bit_width,
        )
    stats = pq.read_table(root / "layer_stats.parquet").to_pylist()
    coverage = {
        (row["layer_index"], row["tensor_role"], row["bit_width"])
        for row in stats
    }
    expected_coverage = {
        (layer, role, bit_width)
        for layer in range(LAYERS)
        for role in ("K", "V")
        for bit_width in (4, 3, 2)
    }
    if len(stats) != 192 or coverage != expected_coverage:
        raise Phase9WorkerError("layer statistics coverage drifted")
    if any(row["nan_count"] or row["inf_count"] for row in stats):
        raise Phase9WorkerError("layer statistics contain invalid values")
    for name in (
        "fisher_replay",
        "outlier_policy",
        "quantizer_regeneration",
        "token_reconstruction",
    ):
        if _load_json(root / f"reproducibility/{name}.json").get("status") != "PASS":
            raise Phase9WorkerError(f"reproducibility check {name} failed")
    result = {
        "schema_version": "kvbench-phase9-calibration-validation-1.0.0",
        "status": "PASS",
        "token_shape": [NUM_EXAMPLES, SEQUENCE_LENGTH],
        "fisher_tensors": 64,
        "quantizer_families": ["kvq4", "kvq3", "kvq2"],
        "layer_stat_rows": 192,
        "performance_measurement": False,
        "quality_evaluation": False,
    }
    if arguments.output is not None:
        write_json_exclusive(arguments.output, result)
    print(json.dumps(result, sort_keys=True))


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-artifact-manifest", required=True)
    parser.add_argument("--cache-dir", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze-dataset")
    freeze.add_argument("--dataset-parquet", required=True)
    freeze.add_argument("--image-manifest", required=True)
    freeze.add_argument("--output-root", required=True)
    _add_model_arguments(freeze)
    freeze.set_defaults(function=command_freeze_dataset)

    fisher = commands.add_parser("run-fisher")
    fisher.add_argument("--source-root", required=True)
    fisher.add_argument("--token-tensor", required=True)
    fisher.add_argument("--output-dir", required=True)
    _add_model_arguments(fisher)
    fisher.set_defaults(function=command_run_fisher)

    fisher_manifest = commands.add_parser("fisher-manifest")
    fisher_manifest.add_argument("--fisher", required=True)
    fisher_manifest.add_argument("--output", required=True)
    fisher_manifest.set_defaults(function=command_fisher_manifest)

    quantizer = commands.add_parser("run-quantizer")
    quantizer.add_argument("--bit-width", type=int, choices=(2, 3, 4), required=True)
    quantizer.add_argument("--source-root", required=True)
    quantizer.add_argument("--token-tensor", required=True)
    quantizer.add_argument("--token-tensor-sha256", required=True)
    quantizer.add_argument("--dataset-root", required=True)
    quantizer.add_argument("--fisher", required=True)
    quantizer.add_argument("--fisher-root", required=True)
    quantizer.add_argument("--output-safe", required=True)
    quantizer.add_argument("--output-manifest", required=True)
    _add_model_arguments(quantizer)
    quantizer.set_defaults(function=command_run_quantizer)

    compare = commands.add_parser("compare-quantizers")
    compare.add_argument("--bit-width", type=int, choices=(2, 3, 4), required=True)
    compare.add_argument("--original", required=True)
    compare.add_argument("--regenerated", required=True)
    compare.add_argument("--output", required=True)
    compare.set_defaults(function=command_compare_quantizers)

    stats = commands.add_parser("layer-stats")
    stats.add_argument("--token-tensor", required=True)
    stats.add_argument("--fisher-manifest", required=True)
    stats.add_argument("--kvq4", required=True)
    stats.add_argument("--kvq3", required=True)
    stats.add_argument("--kvq2", required=True)
    stats.add_argument("--output", required=True)
    _add_model_arguments(stats)
    stats.set_defaults(function=command_layer_stats)

    reconstruct = commands.add_parser("reconstruct-tokens")
    reconstruct.add_argument("--dataset-parquet", required=True)
    reconstruct.add_argument("--expected-dataset-manifest", required=True)
    reconstruct.add_argument("--expected-token-tensor", required=True)
    reconstruct.add_argument("--output", required=True)
    _add_model_arguments(reconstruct)
    reconstruct.set_defaults(function=command_reconstruct_tokens)

    replay = commands.add_parser("replay-fisher")
    replay.add_argument("--token-tensor", required=True)
    replay.add_argument("--fisher", required=True)
    replay.add_argument("--output", required=True)
    _add_model_arguments(replay)
    replay.set_defaults(function=command_replay_fisher)

    policy = commands.add_parser("policy-check")
    policy.add_argument("--output", required=True)
    policy.set_defaults(function=command_policy_check)

    validate = commands.add_parser("validate-payloads")
    validate.add_argument("--bundle", required=True)
    validate.add_argument("--output")
    validate.set_defaults(function=command_validate_payloads)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        arguments.function(arguments)
        return 0
    except (
        FileExistsError,
        OSError,
        Phase9WorkerError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
