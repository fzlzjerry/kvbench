#!/usr/bin/env python3
"""Validate the frozen Phase 7 KIVI reference fixtures without regeneration."""

from __future__ import annotations

import argparse
from functools import reduce
import hashlib
import json
import math
import operator
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPOSITORY_ROOT / "reference" / "kivi"
DEFAULT_FIXTURE_ROOT = REFERENCE_ROOT / "fixtures"
SOURCE_MANIFEST_PATH = REFERENCE_ROOT / "source_manifest.json"
ENVIRONMENT_PATH = REFERENCE_ROOT / "environment.json"
BUILD_MANIFEST_PATH = REFERENCE_ROOT / "build_manifest.json"
DOCKERFILE_PATH = REPOSITORY_ROOT / "docker/reference-kivi.Dockerfile"
GENERATOR_PATH = REFERENCE_ROOT / "generate_fixtures.py"
PYTHON_FREEZE_PATH = REFERENCE_ROOT / "python-freeze.txt"
SOURCE_COMMIT = "876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6"
SOURCE_TREE = "c94c31b2cfd44eeb9a18cff9dcdf03adff4ac49b"
PATCHED_TREE = "b617493dea5aff1a754cd27ad6be12ac512b2aee"
DIGEST = re.compile(r"^[0-9a-f]{64}$")
OCI_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
EXPECTED_VARIANTS = {
    "k4v4": (4, 4, "mandatory"),
    "k2v4": (2, 4, "mandatory"),
    "k2v2": (2, 2, "mandatory"),
    "k4v2": (4, 2, "held_out_asymmetry_control"),
}
DTYPE_BYTES = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "int32": 4,
}
FORBIDDEN_TIMING_KEYS = {
    "latency",
    "latency_ms",
    "duration",
    "duration_ms",
    "elapsed",
    "elapsed_ms",
    "cpu_time",
    "cuda_time",
    "self_cpu_time",
    "self_cuda_time",
    "throughput",
    "tokens_per_second",
}


class FixtureValidationError(RuntimeError):
    """Raised when the KIVI fixture set violates its frozen contract."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FixtureValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise FixtureValidationError(f"non-finite JSON value: {value}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise FixtureValidationError(f"missing JSON file: {path}") from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise FixtureValidationError(f"unsafe JSON file: {path}")
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_pairs,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise FixtureValidationError(f"JSON root must be an object: {path}")
    return value


def _safe_file(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if (
        not relative
        or pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise FixtureValidationError(f"unsafe relative path: {relative!r}")
    path = root.joinpath(*pure.parts)
    try:
        metadata = path.lstat()
    except OSError as error:
        raise FixtureValidationError(
            f"missing fixture file: {relative}"
        ) from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise FixtureValidationError(f"unsafe fixture file: {relative}")
    return path


def _walk_no_timing(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = key.lower()
            if (
                lowered in FORBIDDEN_TIMING_KEYS
                or lowered.endswith("_latency")
                or lowered.endswith("_throughput")
            ) and child is not False:
                raise FixtureValidationError(
                    f"timing field is populated at {path}.{key}"
                )
            _walk_no_timing(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_no_timing(child, f"{path}[{index}]")


def _shape_elements(shape: Any) -> int:
    if (
        not isinstance(shape, list)
        or not shape
        or any(type(item) is not int or item <= 0 for item in shape)
    ):
        raise FixtureValidationError(f"invalid tensor shape: {shape!r}")
    return reduce(operator.mul, shape, 1)


def _validate_tensor(record: dict[str, Any], label: str) -> None:
    required = {
        "device",
        "dtype",
        "shape",
        "stride",
        "logical_nbytes",
        "storage_nbytes",
        "payload_sha256",
        "payload_hex",
    }
    if set(record) != required:
        raise FixtureValidationError(f"invalid tensor keys: {label}")
    dtype = record["dtype"]
    if dtype not in DTYPE_BYTES:
        raise FixtureValidationError(f"invalid tensor dtype: {label}")
    expected = _shape_elements(record["shape"]) * DTYPE_BYTES[dtype]
    if record["logical_nbytes"] != expected:
        raise FixtureValidationError(f"logical tensor bytes mismatch: {label}")
    if (
        type(record["storage_nbytes"]) is not int
        or record["storage_nbytes"] < expected
    ):
        raise FixtureValidationError(f"storage tensor bytes mismatch: {label}")
    try:
        payload = bytes.fromhex(record["payload_hex"])
    except (TypeError, ValueError) as error:
        raise FixtureValidationError(
            f"invalid tensor payload hex: {label}"
        ) from error
    if len(payload) != expected:
        raise FixtureValidationError(f"tensor payload size mismatch: {label}")
    if (
        not isinstance(record["payload_sha256"], str)
        or DIGEST.fullmatch(record["payload_sha256"]) is None
        or _sha256(payload) != record["payload_sha256"]
    ):
        raise FixtureValidationError(f"tensor payload hash mismatch: {label}")


def _state_tokens(state: dict[str, Any], label: str) -> None:
    length = state["length"]
    groups = (
        state["quantized_key_tokens"],
        state["residual_key_tokens"],
        state["quantized_value_tokens"],
        state["residual_value_tokens"],
    )
    for group in groups:
        if (
            not isinstance(group, list)
            or any(type(token) is not int for token in group)
            or len(group) != len(set(group))
        ):
            raise FixtureValidationError(f"invalid token region: {label}")
    for side in ((groups[0], groups[1]), (groups[2], groups[3])):
        flattened = side[0] + side[1]
        if sorted(flattened) != list(range(length)):
            raise FixtureValidationError(
                f"missing or duplicated cache token: {label}"
            )
        if set(side[0]) & set(side[1]):
            raise FixtureValidationError(f"overlapping cache region: {label}")

    tensor_names = {
        "quantized_key_payload",
        "residual_key",
        "key_scales",
        "key_minimum_offsets",
        "quantized_value_payload",
        "residual_value",
        "value_scales",
        "value_minimum_offsets",
    }
    if set(state["tensors"]) != tensor_names:
        raise FixtureValidationError(f"state tensor set mismatch: {label}")
    for name, tensor in state["tensors"].items():
        if tensor is None:
            continue
        _validate_tensor(tensor, f"{label}.{name}")
        if tensor["device"] != "cuda" or tensor["shape"][1] != 8:
            raise FixtureValidationError(
                f"cache tensor is not native eight-head CUDA storage: "
                f"{label}.{name}"
            )


def _expected_state_tokens(context: int) -> dict[str, list[int]]:
    expected = {
        17: {
            "quantized_key_tokens": [],
            "residual_key_tokens": list(range(17)),
            "quantized_value_tokens": [],
            "residual_value_tokens": list(range(17)),
        },
        18: {
            "quantized_key_tokens": [],
            "residual_key_tokens": list(range(18)),
            "quantized_value_tokens": [],
            "residual_value_tokens": list(range(18)),
        },
        31: {
            "quantized_key_tokens": [],
            "residual_key_tokens": list(range(31)),
            "quantized_value_tokens": [],
            "residual_value_tokens": list(range(31)),
        },
        32: {
            "quantized_key_tokens": list(range(32)),
            "residual_key_tokens": [],
            "quantized_value_tokens": [],
            "residual_value_tokens": list(range(32)),
        },
        33: {
            "quantized_key_tokens": list(range(32)),
            "residual_key_tokens": [32],
            "quantized_value_tokens": [0],
            "residual_value_tokens": list(range(1, 33)),
        },
        34: {
            "quantized_key_tokens": list(range(32)),
            "residual_key_tokens": [32, 33],
            "quantized_value_tokens": [0, 1],
            "residual_value_tokens": list(range(2, 34)),
        },
    }
    return expected[context]


def _validate_state(
    state: dict[str, Any],
    context: int,
    label: str,
) -> None:
    if state["length"] != context:
        raise FixtureValidationError(f"state length mismatch: {label}")
    _state_tokens(state, label)
    expected = _expected_state_tokens(context)
    for key, value in expected.items():
        if state[key] != value:
            raise FixtureValidationError(
                f"incorrect rollover boundary at {label}.{key}"
            )


def _validate_accounting(
    fixture: dict[str, Any],
    variant_id: str,
) -> None:
    records = fixture["byte_accounting"]
    if [record["context"] for record in records] != [31, 32, 33, 64]:
        raise FixtureValidationError("byte accounting contexts changed")
    state_by_context = {
        31: fixture["rollover"]["before"]["state"],
        32: fixture["rollover"]["boundary"]["state"],
        33: fixture["rollover"]["after"]["state"],
    }
    ratios: list[float] = []
    for record in records:
        context = record["context"]
        categories = record["categories"]
        expected_categories = {
            "quantized_k_payload",
            "quantized_v_payload",
            "key_scales",
            "key_zero_points",
            "value_scales",
            "value_zero_points",
            "other_metadata",
            "residual_k",
            "residual_v",
            "padding_alignment",
            "persistent_workspace",
        }
        if set(categories) != expected_categories:
            raise FixtureValidationError("byte category set mismatch")
        if any(type(value) is not int or value < 0 for value in categories.values()):
            raise FixtureValidationError("invalid byte category")
        total = sum(categories.values())
        logical = 1 * 8 * context * 128 * 2 * 2
        if (
            total != record["actual_total"]
            or record["logical_bf16_bytes"] != logical
            or not math.isclose(record["r_alloc"], total / logical)
            or record["storage_agreement"] is not True
            or record["r_hbm"] is not None
        ):
            raise FixtureValidationError("byte accounting identity mismatch")
        if context in state_by_context:
            storage = sum(
                0 if tensor is None else tensor["storage_nbytes"]
                for tensor in state_by_context[context]["tensors"].values()
            )
            if storage != total:
                raise FixtureValidationError(
                    "actual tensor storage disagrees with byte accounting"
                )
            if record["calculation_mode"] != (
                "actual_source_owned_tensor_storage"
            ):
                raise FixtureValidationError("runtime accounting mode mismatch")
        elif record["calculation_mode"] != (
            "source_layout_formula_no_runtime_campaign"
        ):
            raise FixtureValidationError("static accounting mode mismatch")
        ratios.append(record["r_alloc"])
    if len(set(ratios)) < 3:
        raise FixtureValidationError(
            f"r_alloc does not vary with context: {variant_id}"
        )


def _validate_gqa(fixture: dict[str, Any]) -> None:
    gqa = fixture["gqa"]
    if (
        (gqa["h_q"], gqa["h_kv"]) != (32, 8)
        or gqa["cache_head_count"] != 8
        or gqa["head_mapping"] != [head // 4 for head in range(32)]
        or gqa["repeat_kv"]
        or gqa["repeat_interleave"]
        or gqa["expand_reshape_kv_materialization"]
        or gqa["expanded_temporary"]
        or gqa["final_verdict"] != "PASS_NATIVE_EIGHT_HEAD_KV_STORAGE"
    ):
        raise FixtureValidationError("GQA contract mismatch")
    operands = gqa["bmm_operands"]
    if not operands:
        raise FixtureValidationError("GQA runtime audit is empty")
    for record in operands:
        if (
            record["left_device"] != "cuda"
            or record["right_device"] != "cuda"
            or record["right_shape"][0] != 8
            or record["left_shape"][0] != 8
        ):
            raise FixtureValidationError("H_Q-sized GQA operand observed")


def _validate_trace(fixture: dict[str, Any], k_bits: int, v_bits: int) -> None:
    trace = fixture["reference_trace"]
    if (
        trace["run_kind"] != "reference_trace"
        or not trace["timings_discarded"]
        or trace["full_prefix_temporary"]
        or trace["backend_fallback"]
        or trace["performance_claim_eligible"]
    ):
        raise FixtureValidationError("reference trace contract mismatch")
    if not trace["quantize_store_kernels"]:
        raise FixtureValidationError("quantize/store kernels absent")
    for name in ("_minmax_along_last_dim", "_pack_along_last_dim"):
        if name not in trace["quantize_store_kernels"]:
            raise FixtureValidationError(f"trace kernel absent: {name}")
    if not {"aten::cat", "aten::bmm"}.issubset(
        set(trace["append_operations"])
    ):
        raise FixtureValidationError("append trace operations absent")
    decode_names = " ".join(trace["decode_dequant_kernels"])
    for bits in {k_bits, v_bits}:
        if f"bgemv{bits}_kernel_outer_dim" not in decode_names:
            raise FixtureValidationError(
                f"decode kernel family absent for {bits} bits"
            )


def _validate_fixture(
    fixture: dict[str, Any],
    variant_id: str,
) -> None:
    _walk_no_timing(fixture)
    if fixture["schema_version"] != "kivi-reference-fixture-1.0.0":
        raise FixtureValidationError("fixture schema mismatch")
    k_bits, v_bits, role = EXPECTED_VARIANTS[variant_id]
    variant = fixture["variant"]
    if variant != {
        "id": variant_id,
        "k_bits": k_bits,
        "v_bits": v_bits,
        "role": role,
    }:
        raise FixtureValidationError(f"variant mismatch: {variant_id}")
    if fixture["source"] != {
        "repository": "https://github.com/jy-yuan/KIVI.git",
        "commit": SOURCE_COMMIT,
        "base_tree": SOURCE_TREE,
        "patched_tree": PATCHED_TREE,
        "authority": "checksum_bound_patched_official_source",
    }:
        raise FixtureValidationError("fixture source authority mismatch")
    if fixture["configuration"] != {
        "k_bits": k_bits,
        "v_bits": v_bits,
        "group_size": 32,
        "residual_length": 32,
    }:
        raise FixtureValidationError("fixture configuration mismatch")
    geometry = fixture["geometry"]
    expected_geometry = {
        "batch_size": 1,
        "num_query_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "basic_store_context": 17,
        "basic_append_tokens": 1,
        "basic_total_context": 18,
        "rollover_contexts": [31, 32, 33],
        "post_rollover_decode_context": 34,
        "seed": 20260726,
        "input_dtype": "bfloat16",
        "upstream_cuda_execution_dtype": "float16",
    }
    if geometry != expected_geometry:
        raise FixtureValidationError("fixture geometry mismatch")
    compatibility = fixture["dtype_compatibility"]
    if (
        compatibility["upstream_cuda_abi"] != "half_only"
        or not compatibility["bf16_input_to_fp16_exact"]
        or not compatibility["fp16_to_bf16_round_trip_exact"]
        or compatibility["algorithmic_cuda_patch"]
    ):
        raise FixtureValidationError("dtype compatibility boundary mismatch")

    for name, tensor in fixture["inputs"].items():
        if name == "query_positions":
            if tensor != [16, 17, 30, 31, 32, 33]:
                raise FixtureValidationError("query positions changed")
        else:
            _validate_tensor(tensor, f"inputs.{name}")
            if tensor["dtype"] != "bfloat16" or tensor["device"] != "cpu":
                raise FixtureValidationError("fixture input dtype/device mismatch")

    _validate_state(fixture["basic"]["store_state"], 17, "basic.store")
    _validate_state(fixture["basic"]["append_state"], 18, "basic.append")
    _validate_tensor(fixture["basic"]["store_output"], "basic.store_output")
    _validate_tensor(fixture["basic"]["decode_output"], "basic.decode_output")

    rollover = fixture["rollover"]
    if (
        rollover["residual_capacity"] != 32
        or rollover["missing_tokens"]
        or rollover["duplicate_tokens"]
        or not rollover["source_faithful"]
    ):
        raise FixtureValidationError("rollover summary mismatch")
    stages = (
        ("before", 31),
        ("boundary", 32),
        ("after", 33),
        ("post_rollover_decode", 34),
    )
    for stage, context in stages:
        _validate_state(
            rollover[stage]["state"],
            context,
            f"rollover.{stage}",
        )
        _validate_tensor(
            rollover[stage]["output"],
            f"rollover.{stage}.output",
        )
    if (
        rollover["boundary"]["operations"]["key_tokens_moved"]
        != list(range(32))
        or rollover["boundary"]["operations"]["value_tokens_moved"]
        or rollover["after"]["operations"]["key_tokens_moved"]
        or rollover["after"]["operations"]["value_tokens_moved"] != [0]
        or rollover["post_rollover_decode"]["operations"][
            "value_tokens_moved"
        ]
        != [1]
    ):
        raise FixtureValidationError("rollover token movement mismatch")

    _validate_accounting(fixture, variant_id)
    _validate_gqa(fixture)
    _validate_trace(fixture, k_bits, v_bits)
    if fixture["full_prefix_dequantization"]["observed"]:
        raise FixtureValidationError("full-prefix dequantization was hidden")
    if fixture["graph_information"]["reference_graph_smoke"] != "NOT_RUN":
        raise FixtureValidationError("unexpected graph campaign")
    if any(fixture["claims"].values()):
        raise FixtureValidationError("claim-bearing fixture field populated")


def _validate_ledger(root: Path) -> int:
    ledger = _safe_file(root, "checksums.sha256")
    entries: dict[str, str] = {}
    for number, line in enumerate(
        ledger.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        parts = line.split("  ", 1)
        if len(parts) != 2 or DIGEST.fullmatch(parts[0]) is None:
            raise FixtureValidationError(
                f"invalid checksum ledger line {number}"
            )
        digest, relative = parts
        if relative in entries:
            raise FixtureValidationError(
                f"duplicate checksum path: {relative}"
            )
        entries[relative] = digest
    if list(entries) != sorted(entries) or not entries:
        raise FixtureValidationError("checksum ledger is empty or unsorted")
    for relative, digest in entries.items():
        if _sha256(_safe_file(root, relative).read_bytes()) != digest:
            raise FixtureValidationError(f"checksum mismatch: {relative}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != ledger
    }
    if actual != set(entries):
        raise FixtureValidationError("checksum inventory mismatch")
    return len(entries)


def _validate_manifests(
    fixture_root: Path,
    fixture_set: dict[str, Any],
) -> None:
    source_sha = _sha256(SOURCE_MANIFEST_PATH.read_bytes())
    build_sha = _sha256(BUILD_MANIFEST_PATH.read_bytes())
    dockerfile_sha = _sha256(DOCKERFILE_PATH.read_bytes())
    if (
        fixture_set["source_commit"] != SOURCE_COMMIT
        or fixture_set["base_tree"] != SOURCE_TREE
        or fixture_set["patched_tree"] != PATCHED_TREE
        or fixture_set["source_manifest_sha256"] != source_sha
        or fixture_set["build_manifest_sha256"] != build_sha
        or fixture_set["dockerfile_sha256"] != dockerfile_sha
        or fixture_set["configurations"] != list(EXPECTED_VARIANTS)
        or fixture_set["mandatory_configurations"]
        != ["k4v4", "k2v4", "k2v2"]
        or fixture_set["held_out_configurations"] != ["k4v2"]
        or fixture_set["performance_measurement"]
        or fixture_set["r_hbm_populated"]
    ):
        raise FixtureValidationError("fixture-set manifest mismatch")
    records = fixture_set["variant_manifests"]
    if [record["variant"] for record in records] != list(EXPECTED_VARIANTS):
        raise FixtureValidationError("variant manifest order mismatch")
    for record in records:
        variant_id = record["variant"]
        manifest_path = _safe_file(fixture_root, record["path"])
        raw = manifest_path.read_bytes()
        if len(raw) != record["nbytes"] or _sha256(raw) != record["sha256"]:
            raise FixtureValidationError("variant manifest identity mismatch")
        manifest = _load_json(manifest_path)
        fixture_path = _safe_file(
            manifest_path.parent,
            manifest["fixture"]["path"],
        )
        fixture_raw = fixture_path.read_bytes()
        if (
            manifest["variant"]["id"] != variant_id
            or len(fixture_raw) != manifest["fixture"]["nbytes"]
            or _sha256(fixture_raw) != manifest["fixture"]["sha256"]
            or manifest["source_commit"] != SOURCE_COMMIT
            or manifest["patched_tree"] != PATCHED_TREE
            or manifest["image_config_digest"]
            != fixture_set["image_config_digest"]
            or manifest["extension_sha256"]
            != fixture_set["extension_sha256"]
            or manifest["performance_measurement"]
        ):
            raise FixtureValidationError("per-variant manifest mismatch")
        _validate_fixture(_load_json(fixture_path), variant_id)


def _validate_static_environment() -> tuple[dict[str, Any], dict[str, Any]]:
    source = _load_json(SOURCE_MANIFEST_PATH)
    environment = _load_json(ENVIRONMENT_PATH)
    build = _load_json(BUILD_MANIFEST_PATH)
    _walk_no_timing(environment)
    _walk_no_timing(build)
    if (
        source["source"]["commit"] != SOURCE_COMMIT
        or source["source"]["base_tree"] != SOURCE_TREE
        or source["source"]["patched_tree"] != PATCHED_TREE
        or source["source"]["floating_branch"]
        or source["source"]["unofficial_fork"]
        or len(source["relevant_source_files"]) != 15
    ):
        raise FixtureValidationError("source manifest contract mismatch")
    if (
        _sha256(DOCKERFILE_PATH.read_bytes())
        != environment["dockerfile"]["sha256"]
        or _sha256(GENERATOR_PATH.read_bytes())
        != environment["image"]["generator_sha256"]
        or _sha256(PYTHON_FREEZE_PATH.read_bytes())
        != environment["python_environment"]["freeze_sha256"]
        or _sha256(SOURCE_MANIFEST_PATH.read_bytes())
        != build["source"]["source_manifest_sha256"]
        or build["image"]["platform_manifest_digest"]
        != environment["image"]["platform_manifest_digest"]
        or build["image"]["config_digest"]
        != environment["image"]["config_digest"]
    ):
        raise FixtureValidationError("environment file identity mismatch")
    if environment["measurement_container_modified"]:
        raise FixtureValidationError("Measurement Container modification set")
    required_build_results = (
        "source_probe",
        "native_sm120",
        "sm120_cubin",
        "compute120_ptx",
        "forced_ptx_jit",
        "compute_sanitizer",
        "no_kernel_image",
        "unsupported_fallback",
    )
    for key in required_build_results:
        expected = False if key in {"no_kernel_image", "unsupported_fallback"} else True
        if build["results"][key] is not expected:
            raise FixtureValidationError(f"build result mismatch: {key}")
    if (
        build["source"]["commit"] != SOURCE_COMMIT
        or build["source"]["patched_tree"] != PATCHED_TREE
        or build["performance_measurement"]
        or build["r_hbm_populated"]
        or DIGEST.fullmatch(build["extension"]["sha256"]) is None
        or build["forced_ptx_jit"]["status"] != "PASS"
        or build["compute_sanitizer"]["status"] != "PASS"
        or build["compute_sanitizer"]["error_summary"] != "0 errors"
        or build["compute_sanitizer"]["extension_sha256"]
        != build["extension"]["sha256"]
    ):
        raise FixtureValidationError("build manifest contract mismatch")
    return environment, build


def _validate_image(
    image: str,
    environment: dict[str, Any],
    build: dict[str, Any],
) -> None:
    inspect = subprocess.run(
        ("docker", "image", "inspect", image, "--format", "{{.Id}}"),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if (
        inspect.returncode != 0
        or inspect.stdout.strip() != environment["image"]["local_id"]
    ):
        raise FixtureValidationError("reference image identity mismatch")
    parent = subprocess.run(
        (
            "docker",
            "image",
            "inspect",
            "kvbench-measurement:phase6a",
            "--format",
            "{{.Id}}",
        ),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if (
        parent.returncode != 0
        or parent.stdout.strip()
        != environment["base_image"]["config_digest"]
    ):
        raise FixtureValidationError("reference parent image drift")
    extension = subprocess.run(
        (
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "/bin/bash",
            image,
            "-c",
            (
                "set -eu; extension=$(find /opt/kivi-source/quant "
                "-maxdepth 1 -type f -name 'kivi_gemv*.so'); "
                "sha256sum \"$extension\""
            ),
        ),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    observed = extension.stdout.split()[0] if extension.stdout.split() else ""
    if extension.returncode != 0 or observed != build["extension"]["sha256"]:
        raise FixtureValidationError("live extension identity mismatch")


def validate_all(
    fixture_root: Path = DEFAULT_FIXTURE_ROOT,
    *,
    image: str = "kvbench-reference-kivi:phase7",
    check_image: bool = True,
) -> dict[str, Any]:
    fixture_root = fixture_root.resolve(strict=True)
    if fixture_root.is_symlink() or not fixture_root.is_dir():
        raise FixtureValidationError("unsafe fixture root")
    environment, build = _validate_static_environment()
    fixture_set = _load_json(fixture_root / "fixture_set.json")
    _walk_no_timing(fixture_set)
    if (
        fixture_set["image_config_digest"]
        != environment["image"]["config_digest"]
        or fixture_set["extension_sha256"] != build["extension"]["sha256"]
    ):
        raise FixtureValidationError("fixture environment identity mismatch")
    ledger_entries = _validate_ledger(fixture_root)
    _validate_manifests(fixture_root, fixture_set)
    if check_image:
        _validate_image(image, environment, build)
    return {
        "schema_version": "kivi-reference-validation-result-1.0.0",
        "status": "PASS",
        "fixture_root": str(fixture_root),
        "configuration_count": len(EXPECTED_VARIANTS),
        "mandatory_count": 3,
        "held_out_count": 1,
        "ledger_entries": ledger_entries,
        "image_checked": check_image,
        "performance_measurement": False,
        "r_hbm_populated": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=DEFAULT_FIXTURE_ROOT,
    )
    parser.add_argument(
        "--image",
        default="kvbench-reference-kivi:phase7",
    )
    parser.add_argument("--skip-image", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    try:
        result = validate_all(
            arguments.fixture_root,
            image=arguments.image,
            check_image=not arguments.skip_image,
        )
    except (
        FixtureValidationError,
        OSError,
        ValueError,
        subprocess.SubprocessError,
    ) as error:
        print(
            json.dumps(
                {
                    "schema_version": (
                        "kivi-reference-validation-result-1.0.0"
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
