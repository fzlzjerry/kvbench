#!/usr/bin/env python3
"""Validate the frozen Phase 5 TurboQuant reference fixtures."""

from __future__ import annotations

import argparse
from functools import reduce
import json
import math
import operator
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Any

from kvbench.runtime.artifacts import sha256_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPOSITORY_ROOT / "reference" / "turboquant"
DEFAULT_FIXTURE_ROOT = REFERENCE_ROOT / "fixtures"
SOURCE_MANIFEST_PATH = REFERENCE_ROOT / "source_manifest.json"
ENVIRONMENT_PATH = REFERENCE_ROOT / "environment.json"
DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
FORBIDDEN_TIMING_KEYS = frozenset(
    {
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
)


class FixtureValidationError(RuntimeError):
    """Raised when a reference fixture violates its frozen contract."""


def _reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FixtureValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise FixtureValidationError(f"non-finite JSON value: {value}")


def load_json(path: Path) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise FixtureValidationError(f"missing JSON file: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise FixtureValidationError(f"unsafe JSON file: {path}")
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_pairs,
        parse_constant=_reject_constant,
    )
    if not isinstance(payload, dict):
        raise FixtureValidationError(f"JSON root must be an object: {path}")
    return payload


def _safe_path(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if (
        not relative
        or pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise FixtureValidationError(f"unsafe fixture path: {relative!r}")
    target = root.joinpath(*pure.parts)
    try:
        metadata = target.lstat()
    except OSError as error:
        raise FixtureValidationError(f"missing fixture file: {relative}") from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise FixtureValidationError(f"unsafe fixture file: {relative}")
    return target


def _validate_ledger(root: Path, ledger_relative: str) -> int:
    ledger = _safe_path(root, ledger_relative)
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        ledger.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = line.split("  ", 1)
        if len(parts) != 2 or DIGEST_PATTERN.fullmatch(parts[0]) is None:
            raise FixtureValidationError(
                f"malformed checksum line {line_number}: {ledger_relative}"
            )
        digest, relative = parts
        if relative in entries:
            raise FixtureValidationError(f"duplicate checksum path: {relative}")
        entries[relative] = digest
    if not entries or list(entries) != sorted(entries):
        raise FixtureValidationError(f"checksum ledger is empty or unsorted: {ledger}")
    ledger_parent = ledger.parent
    for relative, digest in entries.items():
        target = _safe_path(ledger_parent, relative)
        if sha256_file(target) != digest:
            raise FixtureValidationError(f"checksum mismatch: {target}")
    actual = {
        path.relative_to(ledger_parent).as_posix()
        for path in ledger_parent.rglob("*")
        if path.is_file()
        and path != ledger
        and (ledger_relative != "checksums.sha256" or path.name != ".generation.lock")
    }
    if set(entries) != actual:
        missing = sorted(actual - set(entries))
        extra = sorted(set(entries) - actual)
        raise FixtureValidationError(
            f"checksum inventory mismatch in {ledger}: missing={missing}, extra={extra}"
        )
    return len(entries)


def _walk_no_timing(value: Any, *, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = key.lower()
            is_timing_key = (
                lowered in FORBIDDEN_TIMING_KEYS or lowered.endswith("_latency")
            )
            if is_timing_key and child is not False:
                raise FixtureValidationError(f"timing field forbidden at {path}.{key}")
            _walk_no_timing(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_no_timing(child, path=f"{path}[{index}]")


def _require_keys(payload: dict[str, Any], required: set[str], label: str) -> None:
    missing = sorted(required - set(payload))
    if missing:
        raise FixtureValidationError(f"{label} missing keys: {missing}")


def _shape_elements(shape: list[int]) -> int:
    if not shape or any(type(item) is not int or item <= 0 for item in shape):
        raise FixtureValidationError(f"invalid tensor shape: {shape!r}")
    return reduce(operator.mul, shape, 1)


def _validate_tensor_record(root: Path, record: dict[str, Any]) -> Path:
    _require_keys(record, {"path", "dtype", "shape", "nbytes", "sha256"}, "tensor")
    target = _safe_path(root, str(record["path"]))
    if DIGEST_PATTERN.fullmatch(str(record["sha256"])) is None:
        raise FixtureValidationError(f"invalid tensor digest: {target}")
    if sha256_file(target) != record["sha256"]:
        raise FixtureValidationError(f"tensor checksum mismatch: {target}")
    dtype_bytes = {"bfloat16": 2, "uint8": 1}.get(record["dtype"])
    if dtype_bytes is None:
        raise FixtureValidationError(f"unsupported fixture dtype: {record['dtype']}")
    expected_nbytes = _shape_elements(record["shape"]) * dtype_bytes
    if record["nbytes"] != expected_nbytes or target.stat().st_size != expected_nbytes:
        raise FixtureValidationError(f"tensor size mismatch: {target}")
    return target


def _configuration_map(source_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    configurations = source_manifest.get("configurations")
    if not isinstance(configurations, list):
        raise FixtureValidationError("source manifest lacks configurations")
    result: dict[str, dict[str, Any]] = {}
    for item in configurations:
        if not isinstance(item, dict) or not isinstance(item.get("cache_dtype"), str):
            raise FixtureValidationError("invalid source configuration record")
        if item["cache_dtype"] in result:
            raise FixtureValidationError("duplicate source configuration")
        result[item["cache_dtype"]] = item
    return result


def _expected_layout(config: dict[str, Any], head_dim: int) -> dict[str, int]:
    key_bits = int(config["key_bits"])
    value_bits = int(config["value_bits"])
    key_fp8 = config["key_quantization"] == "fp8_e4m3"
    packed_keys = head_dim if key_fp8 else math.ceil(head_dim * key_bits / 8)
    key_norm = 0 if key_fp8 else 2
    packed_values = math.ceil(head_dim * value_bits / 8)
    breakdown = {
        "packed_keys": packed_keys,
        "key_norm": key_norm,
        "packed_values": packed_values,
        "value_scale": 2,
        "value_zero_point": 2,
        "alignment_padding": 0,
    }
    slot = sum(breakdown.values())
    breakdown["alignment_padding"] = slot % 2
    return breakdown


def _validate_trace(config_root: Path, trace_record: dict[str, Any], key_fp8: bool) -> None:
    trace_path = _validate_tensorless_file(config_root, trace_record)
    trace = load_json(trace_path)
    _walk_no_timing(trace)
    _require_keys(
        trace,
        {
            "schema_version",
            "run_kind",
            "timings_discarded",
            "store_kernels",
            "append_kernels",
            "decode_kernels",
            "full_prefix_dequantization_observed",
            "gqa_materialization_observed",
            "backend_fallback",
            "performance_claim_eligible",
            "trace_limitation",
        },
        "trace",
    )
    if trace["run_kind"] != "reference_trace" or not trace["timings_discarded"]:
        raise FixtureValidationError("trace kind or timing-discard policy mismatch")
    if trace["performance_claim_eligible"] or trace["backend_fallback"]:
        raise FixtureValidationError("trace cannot authorize claims or fallback")
    expected_store = "_tq_fused_store_fp8" if key_fp8 else "_tq_fused_store_mse"
    for operation in ("store_kernels", "append_kernels"):
        names = trace[operation]
        if not isinstance(names, list) or expected_store not in names:
            raise FixtureValidationError(f"official store kernel absent from {operation}")
    decode_names = trace["decode_kernels"]
    for expected in ("_tq_decode_stage1", "_fwd_kernel_stage2"):
        if expected not in decode_names:
            raise FixtureValidationError(f"official decode kernel absent: {expected}")
    if trace["full_prefix_dequantization_observed"]:
        raise FixtureValidationError("minimal decode unexpectedly used full-prefix dequant")
    if trace["gqa_materialization_observed"]:
        raise FixtureValidationError("minimal decode unexpectedly materialized GQA")


def _validate_tensorless_file(root: Path, record: dict[str, Any]) -> Path:
    _require_keys(record, {"path", "nbytes", "sha256"}, "file")
    target = _safe_path(root, str(record["path"]))
    if target.stat().st_size != record["nbytes"] or sha256_file(target) != record["sha256"]:
        raise FixtureValidationError(f"file identity mismatch: {target}")
    return target


def _validate_configuration(
    fixture_root: Path,
    config_id: str,
    source_config: dict[str, Any],
    fixture_set: dict[str, Any],
    source_sha: str,
    environment_sha: str,
) -> dict[str, Any]:
    config_root = fixture_root / config_id
    if not config_root.is_dir() or config_root.is_symlink():
        raise FixtureValidationError(f"missing fixture directory: {config_id}")
    manifest = load_json(config_root / "manifest.json")
    _walk_no_timing(manifest)
    _require_keys(
        manifest,
        {
            "schema_version",
            "fixture_id",
            "configuration",
            "source",
            "environment",
            "geometry",
            "inputs",
            "layout",
            "outputs",
            "operations",
            "graph",
            "claims",
        },
        "fixture manifest",
    )
    if manifest["schema_version"] != "turboquant-reference-fixture-1.0.0":
        raise FixtureValidationError("fixture schema version mismatch")
    if manifest["configuration"] != source_config:
        raise FixtureValidationError(f"configuration differs from source: {config_id}")
    if manifest["source"]["commit"] != source_sha:
        raise FixtureValidationError("fixture source SHA mismatch")
    if manifest["source"]["manifest_sha256"] != sha256_file(SOURCE_MANIFEST_PATH):
        raise FixtureValidationError("fixture source manifest checksum mismatch")
    if manifest["environment"]["manifest_sha256"] != environment_sha:
        raise FixtureValidationError("fixture environment checksum mismatch")
    geometry = manifest["geometry"]
    expected_geometry = fixture_set["geometry"]
    if geometry != expected_geometry:
        raise FixtureValidationError("fixture geometry differs from frozen set")
    for name, expected in fixture_set["inputs"].items():
        actual = manifest["inputs"].get(name)
        if actual != expected:
            raise FixtureValidationError(f"input identity mismatch: {config_id}/{name}")
        _validate_tensor_record(fixture_root, actual)

    layout = manifest["layout"]
    _require_keys(
        layout,
        {
            "byte_breakdown_per_head_token",
            "key_packed_size",
            "value_packed_size",
            "slot_size",
            "slot_size_aligned",
            "page_bytes",
            "allocated_cache_bytes",
            "owned_cache_bytes_after_store",
            "owned_cache_bytes_after_append",
            "storage_shape",
            "offsets",
        },
        "layout",
    )
    expected_breakdown = _expected_layout(source_config, geometry["head_dim"])
    if layout["byte_breakdown_per_head_token"] != expected_breakdown:
        raise FixtureValidationError(f"byte breakdown differs from source formula: {config_id}")
    if sum(expected_breakdown.values()) != layout["slot_size_aligned"]:
        raise FixtureValidationError(f"byte breakdown sum mismatch: {config_id}")
    if layout["slot_size"] + expected_breakdown["alignment_padding"] != layout[
        "slot_size_aligned"
    ]:
        raise FixtureValidationError(f"slot alignment mismatch: {config_id}")
    if layout["key_packed_size"] != expected_breakdown["packed_keys"] + expected_breakdown[
        "key_norm"
    ]:
        raise FixtureValidationError(f"key packed size mismatch: {config_id}")
    if layout["value_packed_size"] != expected_breakdown["packed_values"] + 4:
        raise FixtureValidationError(f"value packed size mismatch: {config_id}")
    shape = layout["storage_shape"]
    allocated = _shape_elements(shape)
    if allocated != layout["allocated_cache_bytes"]:
        raise FixtureValidationError(f"allocated storage formula mismatch: {config_id}")
    expected_page = geometry["block_size"] * geometry["num_kv_heads"] * layout[
        "slot_size_aligned"
    ]
    if layout["page_bytes"] != expected_page:
        raise FixtureValidationError(f"page byte mismatch: {config_id}")
    if layout["owned_cache_bytes_after_store"] != (
        geometry["initial_context"] * geometry["num_kv_heads"] * layout["slot_size_aligned"]
    ):
        raise FixtureValidationError(f"store-owned byte mismatch: {config_id}")
    if layout["owned_cache_bytes_after_append"] != (
        geometry["total_context"] * geometry["num_kv_heads"] * layout["slot_size_aligned"]
    ):
        raise FixtureValidationError(f"append-owned byte mismatch: {config_id}")

    outputs = manifest["outputs"]
    for name in ("cache_after_store", "cache_after_append"):
        path = _validate_tensor_record(config_root, outputs[name])
        if path.stat().st_size != allocated:
            raise FixtureValidationError(f"actual cache allocation mismatch: {config_id}/{name}")
    append_path = _validate_tensor_record(config_root, outputs["append_slot"])
    if append_path.stat().st_size != geometry["num_kv_heads"] * layout[
        "slot_size_aligned"
    ]:
        raise FixtureValidationError(f"append slot size mismatch: {config_id}")
    decode = outputs["decode_output"]
    _validate_tensor_record(config_root, decode)
    if decode["shape"] != [
        geometry["batch_size"],
        geometry["num_query_heads"],
        geometry["head_dim"],
    ]:
        raise FixtureValidationError(f"decode shape mismatch: {config_id}")
    _validate_trace(config_root, outputs["kernel_trace"], source_config["key_bits"] == 8)
    _validate_ledger(config_root, "checksums.sha256")
    if manifest["operations"] != {
        "store": "vllm.v1.attention.ops.triton_turboquant_store.triton_turboquant_store",
        "append": "vllm.v1.attention.ops.triton_turboquant_store.triton_turboquant_store",
        "decode": "vllm.v1.attention.ops.triton_turboquant_decode.triton_turboquant_decode_attention",
        "local_algorithm_reimplementation": False,
    }:
        raise FixtureValidationError("fixture does not identify the official operation path")
    if manifest["claims"] != {
        "comparative_latency": False,
        "performance": False,
        "quality": False,
    }:
        raise FixtureValidationError("fixture claim boundary mismatch")
    return {
        "configuration": config_id,
        "slot_size": layout["slot_size_aligned"],
        "allocated_cache_bytes": allocated,
        "decode_sha256": decode["sha256"],
    }


def validate_reference(fixture_root: Path = DEFAULT_FIXTURE_ROOT) -> dict[str, Any]:
    if not fixture_root.is_dir() or fixture_root.is_symlink():
        raise FixtureValidationError(f"missing or unsafe fixture root: {fixture_root}")
    source_manifest = load_json(SOURCE_MANIFEST_PATH)
    environment = load_json(ENVIRONMENT_PATH)
    _walk_no_timing(source_manifest)
    _walk_no_timing(environment)
    fixture_set = load_json(fixture_root / "fixture_set.json")
    _walk_no_timing(fixture_set)
    _require_keys(
        fixture_set,
        {
            "schema_version",
            "source_commit",
            "source_manifest_sha256",
            "environment_manifest_sha256",
            "geometry",
            "inputs",
            "configurations",
            "mandatory_configurations",
            "optional_configurations",
            "determinism",
            "claims",
        },
        "fixture set",
    )
    if fixture_set["schema_version"] != "turboquant-reference-fixture-set-1.0.0":
        raise FixtureValidationError("fixture-set schema version mismatch")
    source = source_manifest["source"]
    if fixture_set["source_commit"] != source["commit"]:
        raise FixtureValidationError("fixture-set source commit mismatch")
    source_sha = sha256_file(SOURCE_MANIFEST_PATH)
    environment_sha = sha256_file(ENVIRONMENT_PATH)
    if fixture_set["source_manifest_sha256"] != source_sha:
        raise FixtureValidationError("fixture-set source manifest checksum mismatch")
    if fixture_set["environment_manifest_sha256"] != environment_sha:
        raise FixtureValidationError("fixture-set environment checksum mismatch")
    configurations = _configuration_map(source_manifest)
    ordered = fixture_set["configurations"]
    if ordered != [item["cache_dtype"] for item in source_manifest["configurations"]]:
        raise FixtureValidationError("fixture configuration order differs from source lock")
    mandatory = [name for name, item in configurations.items() if item["phase5_role"] == "mandatory"]
    optional = [name for name, item in configurations.items() if item["phase5_role"] == "optional"]
    if fixture_set["mandatory_configurations"] != mandatory:
        raise FixtureValidationError("mandatory fixture list mismatch")
    if fixture_set["optional_configurations"] != optional:
        raise FixtureValidationError("optional fixture list mismatch")
    for record in fixture_set["inputs"].values():
        _validate_tensor_record(fixture_root, record)
    results = [
        _validate_configuration(
            fixture_root,
            config_id,
            configurations[config_id],
            fixture_set,
            source["commit"],
            environment_sha,
        )
        for config_id in ordered
    ]
    root_count = _validate_ledger(fixture_root, "checksums.sha256")
    if fixture_set["claims"] != {
        "comparative_latency": False,
        "performance": False,
        "quality": False,
    }:
        raise FixtureValidationError("fixture-set claim boundary mismatch")
    return {
        "schema_version": "turboquant-reference-validation-result-1.0.0",
        "status": "pass",
        "fixture_root": str(fixture_root),
        "mandatory_fixture_count": len(mandatory),
        "optional_fixture_count": len(optional),
        "checksum_entry_count": root_count,
        "configurations": results,
        "timing_data_present": False,
        "performance_claim_eligible": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-root", type=Path, default=DEFAULT_FIXTURE_ROOT)
    args = parser.parse_args()
    try:
        result = validate_reference(args.fixture_root)
    except (OSError, ValueError, FixtureValidationError) as error:
        print(
            json.dumps(
                {"status": "fail", "error": type(error).__name__, "message": str(error)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
