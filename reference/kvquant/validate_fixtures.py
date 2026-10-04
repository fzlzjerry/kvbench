#!/usr/bin/env python3
"""Validate the frozen Phase 10 KVQuant fixtures without regeneration."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from functools import reduce
import hashlib
import json
import math
import operator
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPOSITORY_ROOT / "reference" / "kvquant"
DEFAULT_FIXTURE_ROOT = REFERENCE_ROOT / "fixtures"
FIXTURE_ID = "kvqref-a50af6511c314b6394e58a7f81ceefb8"

METHOD_IDENTIFIER = "kvquant_gqa_upstream_patch_v1"
UPSTREAM_BASE_COMMIT = "57a238357f0ffe50084670fcd5781c9848f80ea2"
UPSTREAM_BASE_TREE = "094e0f736f77ee327e5350cbd1eefb1c936aa77b"
PATCH_SHA256 = "db3b6fb7ec0a72e25001e1c83a5158d86512248db5c3a06c61895598d1d482d6"
PATCHED_COMMIT = "4ad80bc8c942d0a05516d2be8f8d443a77a05900"
PATCHED_TREE = "c4f1490c9c0c4ec46099f1e95c092516df2adb4e"
CALIBRATION_ID = "kvqcal-cdb724c806d64d095c040d2673a987a3"
CALIBRATION_ROOT = (
    "8148306d08205af376994b022f189a0d6837915cd279ca8af6b104e1f4b46ccf"
)
REFERENCE_IMAGE_CONFIG_DIGEST = (
    "sha256:24eb3f6ff39b72f45c353acfbef6ce2d9aaac0860180b4dde8b937593176714b"
)
REFERENCE_BASE_CONFIG_DIGEST = (
    "sha256:127759078f2c70c9e795c7a1bb3408df1eaee8fa019319299d283dc8075b216d"
)
REFERENCE_DOCKERFILE_SHA256 = (
    "f1b2f2a6f6f15bf364eb3a8b7a26f01504edbe2dbcfe74b619b1c519120a618e"
)
CALIBRATION_PYTHON_FREEZE_SHA256 = (
    "950f28e3513f03e693b4dd87018ced302f4c201754e791bcdef65976c737eb7a"
)
QUANTIZER_SHA256 = {
    "kvq4": "a8c009633ac4cad952deb2a2fa96c44ef928a1510dadcf11dee29a7a3efe1bf6",
    "kvq3": "97518129cc64ffa445722cb0802b3082631841de50835cbdf2c85c36a0c1579f",
    "kvq2": "b9bb3a8699aa38fb2a5707ff036814971552462692a180431f6f68df9624560e",
}

FAMILIES = {"kvq4": 4, "kvq3": 3, "kvq2": 2}
CASES = {
    "key_zero_value_fixed12": 0,
    "key_few_value_fixed12": 6,
    "key_cap_value_fixed12": 12,
}
FIXTURE_MEMBERS = {
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
}
REFERENCE_MEMBERS = {
    "README.md",
    "generate_fixtures.py",
    "validate_fixtures.py",
    "source_manifest.json",
    "environment.json",
    "calibration_manifest.json",
    "build_manifest.json",
    "fixtures",
}

BATCH_SIZE = 1
NUM_QUERY_HEADS = 32
NUM_KV_HEADS = 8
NUM_KV_GROUPS = 4
HEAD_DIM = 128
KV_WIDTH = NUM_KV_HEADS * HEAD_DIM
SINK_TOKENS = 5
STORE_CONTEXT = 17
TOTAL_CONTEXT = 18
STORE_QUANTIZED_CONTEXT = 12
QUANTIZED_CONTEXT = 13
CAPACITY = 12
ENTRIES_PER_TAIL = 6
SEED = 20260729
KEY_LOGIT_ATOL = 0.25
KEY_LOGIT_RTOL = 0.01
DECODE_ATOL = 0.01
DECODE_RTOL = 0.01
RECIPROCAL_ATOL = 1e-9
ROPE_THETA = 500000.0

DIGEST = re.compile(r"[0-9a-f]{64}")
OCI_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
FORBIDDEN_TIMING_KEYS = {
    "latency",
    "latency_ms",
    "duration",
    "duration_ms",
    "elapsed",
    "elapsed_ms",
    "cpu_time",
    "cuda_time",
    "throughput",
    "tokens_per_second",
}
PROHIBITED_KEYS = {
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "cloudflare_api_token",
    "hf_token",
    "hugging_face_hub_token",
    "authorization",
}
DTYPE_BYTES = {
    "bool": 1,
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "float64": 8,
    "int32": 4,
    "int64": 8,
}


class FixtureValidationError(RuntimeError):
    """Raised when a fixture violates the frozen source-faithful contract."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    _require_regular(path)
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise FixtureValidationError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise FixtureValidationError(f"JSON root must be an object: {path}")
    _walk_governance(value)
    return value


def _require_regular(path: Path, *, immutable: bool = False) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise FixtureValidationError(f"missing file: {path}") from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise FixtureValidationError(f"unsafe file: {path}")
    if immutable and metadata.st_mode & (
        stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    ):
        raise FixtureValidationError(f"finalized file is writable: {path}")


def _safe_relative(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or pure.as_posix() != relative
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise FixtureValidationError(f"unsafe relative path: {relative!r}")
    return relative


def _walk_governance(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = key.lower()
            if lowered in PROHIBITED_KEYS:
                raise FixtureValidationError(
                    f"prohibited credential key at {path}.{key}"
                )
            if (
                lowered in FORBIDDEN_TIMING_KEYS
                or lowered.endswith("_latency")
                or lowered.endswith("_throughput")
            ) and child is not False:
                raise FixtureValidationError(
                    f"timing field is populated at {path}.{key}"
                )
            _walk_governance(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_governance(child, f"{path}[{index}]")


def _parse_ledger(path: Path) -> dict[str, str]:
    _require_regular(path)
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise FixtureValidationError(f"non-canonical checksum ledger: {path}")
    entries: dict[str, str] = {}
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeError as error:
        raise FixtureValidationError(f"invalid checksum ledger: {path}") from error
    for line in lines:
        parts = line.split("  ", 1)
        if len(parts) != 2 or DIGEST.fullmatch(parts[0]) is None:
            raise FixtureValidationError(f"invalid checksum ledger line: {path}")
        digest, relative = parts
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or pure.as_posix() != relative
            or any(part in {"", ".", ".."} for part in pure.parts)
            or relative in entries
        ):
            raise FixtureValidationError(f"unsafe checksum path: {relative!r}")
        entries[relative] = digest
    return entries


def _tensor_bytes(tensor: Any) -> bytes:
    import torch

    return (
        tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    )


def _dtype_name(tensor: Any) -> str:
    import torch

    names = {
        torch.bool: "bool",
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
        torch.float64: "float64",
        torch.int32: "int32",
        torch.int64: "int64",
    }
    try:
        return names[tensor.dtype]
    except KeyError as error:
        raise FixtureValidationError(
            f"unsupported tensor dtype: {tensor.dtype}"
        ) from error


def _load_tensors(path: Path) -> dict[str, Any]:
    from safetensors import safe_open

    _require_regular(path)
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            if (
                not isinstance(metadata, dict)
                or metadata.get("format")
                != "kvbench-phase10-kvquant-reference"
                or metadata.get("fixture_id") != FIXTURE_ID
            ):
                raise FixtureValidationError(
                    f"safetensors metadata mismatch: {path}"
                )
            tensors = {
                name: handle.get_tensor(name).contiguous()
                for name in sorted(handle.keys())
            }
    except FixtureValidationError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise FixtureValidationError(f"unsafe tensor file: {path}") from error
    if not tensors:
        raise FixtureValidationError(f"empty tensor file: {path}")
    return tensors


def _shape_elements(shape: Any) -> int:
    if (
        not isinstance(shape, list)
        or not shape
        or any(type(item) is not int or item <= 0 for item in shape)
    ):
        raise FixtureValidationError(f"invalid tensor shape: {shape!r}")
    return reduce(operator.mul, shape, 1)


def _validate_tensor_records(
    records: Any,
    files: Mapping[str, Mapping[str, Any]],
) -> None:
    if not isinstance(records, dict) or set(records) != set(files):
        raise FixtureValidationError("tensor-record file set mismatch")
    for filename, tensors in files.items():
        file_records = records.get(filename)
        if not isinstance(file_records, dict) or set(file_records) != set(
            tensors
        ):
            raise FixtureValidationError(
                f"tensor-record name set mismatch: {filename}"
            )
        for name, tensor in tensors.items():
            record = file_records[name]
            if (
                not isinstance(record, dict)
                or set(record)
                != {"dtype", "shape", "logical_nbytes", "payload_sha256"}
            ):
                raise FixtureValidationError(
                    f"invalid tensor record: {filename}:{name}"
                )
            dtype = _dtype_name(tensor)
            expected_bytes = tensor.numel() * tensor.element_size()
            if (
                record["dtype"] != dtype
                or record["shape"] != list(tensor.shape)
                or record["logical_nbytes"] != expected_bytes
                or _shape_elements(record["shape"]) * DTYPE_BYTES[dtype]
                != expected_bytes
                or record["payload_sha256"]
                != _sha256_bytes(_tensor_bytes(tensor))
            ):
                raise FixtureValidationError(
                    f"tensor record mismatch: {filename}:{name}"
                )


def _expected_rope_inv_freq() -> Any:
    import torch

    base = 1.0 / (
        ROPE_THETA
        ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )
    wavelength = 2 * torch.pi / base
    low_wavelength = 8192.0
    high_wavelength = 2048.0
    scaled = torch.where(wavelength > low_wavelength, base / 8.0, base)
    smooth = (8192.0 / wavelength - 1.0) / 3.0
    smoothed = (1.0 - smooth) * base / 8.0 + smooth * base
    medium = (wavelength <= low_wavelength) & (
        wavelength >= high_wavelength
    )
    return torch.where(medium, smoothed, scaled).contiguous()


def _explicit_rope(tensor: Any, inv_freq: Any, positions: Any) -> Any:
    import torch

    frequencies = torch.outer(positions.float(), inv_freq.float())
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    cosine = (
        embedding.cos()
        .reshape(1, 1, positions.numel(), HEAD_DIM)
        .to(tensor.dtype)
    )
    sine = (
        embedding.sin()
        .reshape(1, 1, positions.numel(), HEAD_DIM)
        .to(tensor.dtype)
    )
    first = tensor[..., : HEAD_DIM // 2]
    second = tensor[..., HEAD_DIM // 2 :]
    rotated = torch.cat((-second, first), dim=-1)
    return tensor * cosine + rotated * sine


def _assert_close(
    actual: Any,
    expected: Any,
    *,
    atol: float,
    rtol: float,
    label: str,
) -> None:
    import torch

    try:
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=atol, rtol=rtol
        )
    except AssertionError as error:
        raise FixtureValidationError(
            f"numerical comparison failed: {label}"
        ) from error


def _validate_authority() -> dict[str, str]:
    source = _load_json(REFERENCE_ROOT / "source_manifest.json")
    environment = _load_json(REFERENCE_ROOT / "environment.json")
    calibration = _load_json(REFERENCE_ROOT / "calibration_manifest.json")
    build = _load_json(REFERENCE_ROOT / "build_manifest.json")
    if (
        source.get("status") != "PASS"
        or source.get("method_identifier") != METHOD_IDENTIFIER
        or source.get("decision") != "0021"
        or source.get("contract_decision") != "0023"
        or source.get("upstream_base_commit") != UPSTREAM_BASE_COMMIT
        or source.get("upstream_base_tree") != UPSTREAM_BASE_TREE
        or source.get("patch_sha256") != PATCH_SHA256
        or source.get("patched_commit") != PATCHED_COMMIT
        or source.get("patched_tree") != PATCHED_TREE
        or source.get("official_author_gqa_support_claimed") is not False
        or source.get("source_checkout_published") is not False
        or source.get("reconstruction", {}).get("status") != "PASS"
        or source.get("reconstruction", {}).get("changed_file_count") != 15
    ):
        raise FixtureValidationError("source authority mismatch")
    if (
        environment.get("status") != "PASS"
        or environment.get("strategy")
        != "thin_image_from_exact_phase9_calibration_image"
        or environment.get("image_config_digest")
        != REFERENCE_IMAGE_CONFIG_DIGEST
        or environment.get("base_image_config_digest")
        != REFERENCE_BASE_CONFIG_DIGEST
        or environment.get("reference_dockerfile_sha256")
        != REFERENCE_DOCKERFILE_SHA256
        or environment.get("calibration_python_freeze_sha256")
        != CALIBRATION_PYTHON_FREEZE_SHA256
        or environment.get("new_dockerfile_created") is not True
        or environment.get("dependency_overlay")
        != {
            "package": "tokenizers",
            "version": "0.15.2",
            "wheel_sha256": (
                "9e0480c452217edd35eca56fafe2029fb4d368b7c0475f8dfa3c5c9c400a7456"
            ),
            "installation": "no_deps_no_index_isolated_target",
        }
        or environment.get("python") != "3.12.3"
        or environment.get("pytorch") != "2.12.1+cu130"
        or environment.get("cuda_userspace") != "13.0"
        or environment.get("cuda_runtime") != 13000
        or environment.get("transformers_installed") != "4.57.6"
        or environment.get("transformers_vendored") != "4.38.0.dev0"
        or environment.get("tokenizers_base_installed") != "0.22.2"
        or environment.get("tokenizers_active") != "0.15.2"
        or "13.3.0" not in str(environment.get("compiler"))
        or "13.0.88" not in str(environment.get("cuda_compiler"))
        or environment.get("gpu_compute_capability") != [12, 0]
        or environment.get("source_mount") != "read_only"
        or environment.get("calibration_mount") != "read_only"
        or environment.get("model_weights_in_image") is not False
        or environment.get("credentials_in_image") is not False
        or environment.get("r2_credentials_in_container") is not False
        or environment.get("network") != "disabled"
    ):
        raise FixtureValidationError("reference environment mismatch")
    quantizers = calibration.get("quantizers")
    if (
        calibration.get("status") != "PASS"
        or calibration.get("calibration_id") != CALIBRATION_ID
        or calibration.get("calibration_root_sha256") != CALIBRATION_ROOT
        or calibration.get("complete") is not True
        or calibration.get("inventory_checksums") != "PASS"
        or calibration.get("mount") != "read_only"
        or calibration.get("fisher_regenerated") is not False
        or calibration.get("quantizers_regenerated") is not False
        or not isinstance(quantizers, dict)
        or set(quantizers) != set(FAMILIES)
    ):
        raise FixtureValidationError("calibration authority mismatch")
    for family, bit_width in FAMILIES.items():
        record = quantizers[family]
        if (
            not isinstance(record, dict)
            or record.get("bit_width") != bit_width
            or record.get("safe_sha256") != QUANTIZER_SHA256[family]
            or record.get("tensor_count") != 320
        ):
            raise FixtureValidationError(
                f"quantizer authority mismatch: {family}"
            )
    if (
        build.get("status") != "PASS"
        or build.get("source_authority", {}).get("patched_tree")
        != PATCHED_TREE
        or build.get("image_config_digest") != REFERENCE_IMAGE_CONFIG_DIGEST
        or build.get("python") != "3.12.3"
        or build.get("pytorch") != "2.12.1+cu130"
        or build.get("cuda_userspace") != "13.0"
        or "13.3.0" not in str(build.get("compiler"))
        or "13.0.88" not in str(build.get("cuda_compiler"))
        or DIGEST.fullmatch(str(build.get("extension_sha256"))) is None
        or build.get("sm_120_cubin") is not True
        or build.get("compute_120_ptx") is not True
        or build.get("native_sm120_execution") is not True
        or build.get("forced_ptx_jit") != "PASS"
        or build.get("compute_sanitizer") != "PASS"
        or build.get("fallback") is not False
        or build.get("extension_published") is not False
    ):
        raise FixtureValidationError("extension build authority mismatch")
    return {
        "source_manifest": _sha256_file(
            REFERENCE_ROOT / "source_manifest.json"
        ),
        "environment": _sha256_file(REFERENCE_ROOT / "environment.json"),
        "calibration_manifest": _sha256_file(
            REFERENCE_ROOT / "calibration_manifest.json"
        ),
        "build_manifest": _sha256_file(
            REFERENCE_ROOT / "build_manifest.json"
        ),
        "extension": build["extension_sha256"],
    }


def _validate_root_controls(
    fixture_root: Path,
    authority_hashes: Mapping[str, str],
) -> tuple[str, dict[str, Any]]:
    root_files = sorted(
        path for path in fixture_root.rglob("*") if path.is_file()
    )
    for path in root_files:
        _require_regular(path, immutable=True)
    relatives = {_safe_relative(path, fixture_root) for path in root_files}
    required = {
        "manifest.json",
        "artifact_inventory.json",
        "checksums.sha256",
        "COMPLETE",
        "reference_trace.json",
    }
    if not required.issubset(relatives):
        raise FixtureValidationError("fixture root lacks finalization controls")
    manifest = _load_json(fixture_root / "manifest.json")
    inventory = _load_json(fixture_root / "artifact_inventory.json")
    complete = _load_json(fixture_root / "COMPLETE")
    trace = _load_json(fixture_root / "reference_trace.json")
    if (
        manifest.get("run_id") != FIXTURE_ID
        or manifest.get("status") != "completed"
        or manifest.get("method_identifier") != METHOD_IDENTIFIER
        or manifest.get("source", {}).get("patched_tree") != PATCHED_TREE
        or manifest.get("source", {}).get("patch_sha256") != PATCH_SHA256
        or manifest.get("source", {}).get("source_manifest_sha256")
        != authority_hashes["source_manifest"]
        or manifest.get("calibration", {}).get("calibration_id")
        != CALIBRATION_ID
        or manifest.get("calibration", {}).get("root_sha256")
        != CALIBRATION_ROOT
        or manifest.get("calibration", {}).get(
            "calibration_manifest_sha256"
        )
        != authority_hashes["calibration_manifest"]
        or manifest.get("environment", {}).get("image_config_digest")
        != REFERENCE_IMAGE_CONFIG_DIGEST
        or manifest.get("environment", {}).get("environment_manifest_sha256")
        != authority_hashes["environment"]
        or manifest.get("environment", {}).get("build_manifest_sha256")
        != authority_hashes["build_manifest"]
        or manifest.get("environment", {}).get("extension_sha256")
        != authority_hashes["extension"]
    ):
        raise FixtureValidationError("root manifest authority mismatch")
    matrix = manifest.get("fixture_matrix")
    if (
        not isinstance(matrix, dict)
        or matrix.get("families") != list(FAMILIES)
        or matrix.get("cases") != list(CASES)
        or matrix.get("total") != 9
        or matrix.get("legacy_ambiguous_aliases") is not False
    ):
        raise FixtureValidationError("root fixture matrix mismatch")
    sparse = manifest.get("sparse_contract")
    if (
        not isinstance(sparse, dict)
        or sparse.get("key_counts") != [0, 6, 12]
        or sparse.get("value_non_sink_count") != 12
        or sparse.get("value_sink_count") != 0
        or sparse.get("value_selection") != "six_lowest_plus_six_highest"
        or sparse.get("value_occupancy_data_dependent") is not False
    ):
        raise FixtureValidationError("root sparse contract mismatch")
    gates = manifest.get("gates")
    if (
        not isinstance(gates, dict)
        or gates.get("g2_kvq") != "NOT_EVALUATED"
        or gates.get("global_g2_g5") != "NOT_EVALUATED"
        or gates.get("full_scan") != "CLOSED"
        or gates.get("quality_execution") != "LOCKED"
        or gates.get("performance_data_frozen_present") is not False
        or manifest.get("performance_measurement") is not False
        or manifest.get("profiler_execution") is not False
        or manifest.get("quality_evaluation") is not False
    ):
        raise FixtureValidationError("root governance mismatch")

    if (
        inventory.get("schema_version")
        != "kvbench-artifact-inventory-1.0.0"
        or inventory.get("run_id") != FIXTURE_ID
        or inventory.get("excluded_control_files")
        != ["artifact_inventory.json", "checksums.sha256", "COMPLETE"]
        or not isinstance(inventory.get("files"), list)
    ):
        raise FixtureValidationError("root inventory mismatch")
    declared: dict[str, tuple[str, int]] = {}
    for record in inventory["files"]:
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "role", "size_bytes", "sha256"}
            or not isinstance(record["path"], str)
            or not isinstance(record["role"], str)
            or not record["role"]
            or type(record["size_bytes"]) is not int
            or record["size_bytes"] < 0
            or not isinstance(record["sha256"], str)
            or DIGEST.fullmatch(record["sha256"]) is None
            or record["path"] in declared
        ):
            raise FixtureValidationError("invalid root inventory record")
        declared[record["path"]] = (
            record["sha256"],
            record["size_bytes"],
        )
    inventory_expected = relatives - {
        "artifact_inventory.json",
        "checksums.sha256",
        "COMPLETE",
    }
    if set(declared) != inventory_expected or list(declared) != sorted(declared):
        raise FixtureValidationError("root inventory coverage mismatch")
    for relative, (digest, size) in declared.items():
        target = fixture_root / relative
        if target.stat().st_size != size or _sha256_file(target) != digest:
            raise FixtureValidationError("root inventory payload mismatch")

    ledger = _parse_ledger(fixture_root / "checksums.sha256")
    if set(ledger) != relatives - {"checksums.sha256", "COMPLETE"}:
        raise FixtureValidationError("root checksum coverage mismatch")
    for relative, digest in ledger.items():
        if _sha256_file(fixture_root / relative) != digest:
            raise FixtureValidationError("root checksum payload mismatch")
    if (
        complete.get("run_id") != FIXTURE_ID
        or complete.get("status") != "completed"
        or complete.get("written_last") is not True
        or complete.get("checksum_ledger_path") != "checksums.sha256"
        or complete.get("manifest_sha256")
        != _sha256_file(fixture_root / "manifest.json")
        or complete.get("artifact_inventory_sha256")
        != _sha256_file(fixture_root / "artifact_inventory.json")
        or complete.get("checksum_ledger_sha256")
        != _sha256_file(fixture_root / "checksums.sha256")
    ):
        raise FixtureValidationError("COMPLETE marker mismatch")
    if (
        trace.get("run_id") != FIXTURE_ID
        or trace.get("run_kind") != "reference_trace"
        or trace.get("timing_fields_present") is not False
        or trace.get("operations", {}).get("complete_prefix_temporary")
        is not False
        or trace.get("operations", {}).get("backend_fallback") is not False
    ):
        raise FixtureValidationError("reference trace mismatch")

    canonical = "".join(
        f"{_sha256_file(path)}  {_safe_relative(path, fixture_root)}\n"
        for path in root_files
    ).encode("utf-8")
    return _sha256_bytes(canonical), manifest


def _validate_fixture_ledger(fixture_root: Path) -> None:
    members = {path.name for path in fixture_root.iterdir()}
    if members != FIXTURE_MEMBERS:
        raise FixtureValidationError(
            f"unexpected fixture members: {fixture_root}"
        )
    ledger = _parse_ledger(fixture_root / "checksums.sha256")
    if set(ledger) != FIXTURE_MEMBERS - {"checksums.sha256"}:
        raise FixtureValidationError("fixture checksum coverage mismatch")
    for relative, digest in ledger.items():
        if _sha256_file(fixture_root / relative) != digest:
            raise FixtureValidationError("fixture checksum mismatch")


def _validate_geometry(
    manifest: Mapping[str, Any],
    family: str,
    case_name: str,
    bit_width: int,
    key_count: int,
    authority_hashes: Mapping[str, str],
) -> None:
    if (
        manifest.get("schema_version")
        != "kvbench-phase10-kvquant-fixture-1.0.0"
        or manifest.get("fixture_id") != FIXTURE_ID
        or manifest.get("family") != family
        or manifest.get("case") != case_name
        or manifest.get("bit_width") != bit_width
        or manifest.get("status") != "PASS"
        or manifest.get("run_kind") != "reference_fixture"
    ):
        raise FixtureValidationError("fixture identity mismatch")
    source = manifest.get("source")
    calibration = manifest.get("calibration")
    environment = manifest.get("environment")
    if (
        not isinstance(source, dict)
        or source.get("method_identifier") != METHOD_IDENTIFIER
        or source.get("decision") != "0021"
        or source.get("contract_decision") != "0023"
        or source.get("patched_commit") != PATCHED_COMMIT
        or source.get("patched_tree") != PATCHED_TREE
        or source.get("patch_sha256") != PATCH_SHA256
        or source.get("source_manifest_sha256")
        != authority_hashes["source_manifest"]
        or not isinstance(calibration, dict)
        or calibration.get("calibration_id") != CALIBRATION_ID
        or calibration.get("root_sha256") != CALIBRATION_ROOT
        or calibration.get("quantizer_sha256") != QUANTIZER_SHA256[family]
        or calibration.get("calibration_manifest_sha256")
        != authority_hashes["calibration_manifest"]
        or calibration.get("fisher_regenerated") is not False
        or calibration.get("quantizer_regenerated") is not False
        or not isinstance(environment, dict)
        or environment.get("image_config_digest")
        != REFERENCE_IMAGE_CONFIG_DIGEST
        or environment.get("environment_manifest_sha256")
        != authority_hashes["environment"]
        or environment.get("build_manifest_sha256")
        != authority_hashes["build_manifest"]
    ):
        raise FixtureValidationError("fixture authority mismatch")
    geometry = manifest.get("geometry")
    expected_geometry = {
        "batch_size": BATCH_SIZE,
        "num_query_heads": NUM_QUERY_HEADS,
        "num_kv_heads": NUM_KV_HEADS,
        "num_kv_groups": NUM_KV_GROUPS,
        "head_dim": HEAD_DIM,
        "interface_dtype": "bfloat16",
        "sink_dtype": "float16",
        "sink_tokens": SINK_TOKENS,
        "store_context": STORE_CONTEXT,
        "append_tokens": 1,
        "total_context": TOTAL_CONTEXT,
        "seed": SEED,
        "query_to_kv_mapping": "kv_head = query_head // 4",
    }
    if geometry != expected_geometry:
        raise FixtureValidationError("fixture geometry mismatch")
    sparse = manifest.get("sparse_contract")
    if (
        not isinstance(sparse, dict)
        or sparse.get("key_sparse_selection_mode")
        != "thresholded_fixed_tail_cap"
        or sparse.get("key_active_count") != key_count
        or sparse.get("key_capacity") != CAPACITY
        or sparse.get("value_sparse_selection_mode") != "fixed_extrema"
        or sparse.get("value_active_count_non_sink") != CAPACITY
        or sparse.get("value_active_count_sink") != 0
        or sparse.get("value_capacity") != CAPACITY
        or sparse.get("value_lower_entries") != ENTRIES_PER_TAIL
        or sparse.get("value_upper_entries") != ENTRIES_PER_TAIL
        or sparse.get("value_occupancy_data_dependent") is not False
        or sparse.get("outlier_value_dtype") != "float32"
        or sparse.get("outlier_index_dtype") != "int32"
    ):
        raise FixtureValidationError("fixture sparse contract mismatch")
    semantics = manifest.get("semantics")
    if (
        not isinstance(semantics, dict)
        or semantics.get("quantized_key") != "pre_rope_k_proj_output"
        or semantics.get("sink_key_stored")
        != "post_rope_attention_ready_fp16"
        or semantics.get("attention_key")
        != "native_llama31_rope_applied_during_reference_decode"
        or semantics.get("value") != "native_v_proj_output_without_rope"
        or semantics.get("position_ids") != list(range(TOTAL_CONTEXT))
        or semantics.get("sink_positions") != list(range(SINK_TOKENS))
        or semantics.get("implementation_head_expansion") is not False
        or semantics.get("independent_control_head_expansion") is not True
    ):
        raise FixtureValidationError("pre-/post-RoPE semantics mismatch")
    execution = manifest.get("execution_path")
    if (
        not isinstance(execution, dict)
        or execution.get("full_prefix_temporary") is not False
        or execution.get("backend_fallback") is not False
        or execution.get("source_repeat_kv") is not False
        or execution.get("source_repeat_interleave") is not False
        or execution.get("control_repeat_interleave") is not True
        or execution.get("timing_fields_present") is not False
    ):
        raise FixtureValidationError("execution-path contract mismatch")
    packing = manifest.get("packing")
    if (
        not isinstance(packing, dict)
        or packing.get("packed_dtype") != "int32"
        or packing.get("packed_rows_per_kv_head")
        != bit_width * HEAD_DIM // 32
        or packing.get("native_kv_heads") != NUM_KV_HEADS
    ):
        raise FixtureValidationError("packing contract mismatch")
    nearest_store = packing.get(
        "value_parallel_store_matches_intended_nearest"
    )
    nearest_append = packing.get(
        "value_after_append_matches_intended_nearest"
    )
    if bit_width == 3:
        if nearest_store is not False or nearest_append is not False:
            raise FixtureValidationError(
                "3-bit parallel Value source behavior was not preserved"
            )
    elif nearest_store is not True or nearest_append is not True:
        raise FixtureValidationError("Value nearest-code control mismatch")
    numerical = manifest.get("numerical_control")
    if (
        not isinstance(numerical, dict)
        or numerical.get("key_logits_atol") != KEY_LOGIT_ATOL
        or numerical.get("key_logits_rtol") != KEY_LOGIT_RTOL
        or numerical.get("decode_atol") != DECODE_ATOL
        or numerical.get("decode_rtol") != DECODE_RTOL
        or numerical.get("finite") is not True
        or manifest.get("performance_measurement") is not False
        or manifest.get("profiler_execution") is not False
        or manifest.get("quality_evaluation") is not False
        or manifest.get("g2_kvq") != "NOT_EVALUATED"
    ):
        raise FixtureValidationError("fixture numerical governance mismatch")


def _validate_sparse(
    inputs: Mapping[str, Any],
    metadata: Mapping[str, Any],
    sparse_values: Mapping[str, Any],
    sparse_indices: Mapping[str, Any],
    bit_width: int,
    key_count: int,
) -> None:
    import torch

    key_counts = sparse_indices["key_active_count_by_position"]
    value_counts = sparse_indices["value_active_count_by_position"]
    expected_key = torch.tensor(
        [0] * SINK_TOKENS + [key_count] * QUANTIZED_CONTEXT,
        dtype=torch.int32,
    )
    expected_value = torch.tensor(
        [0] * SINK_TOKENS + [CAPACITY] * QUANTIZED_CONTEXT,
        dtype=torch.int32,
    )
    if (
        key_counts.dtype != torch.int32
        or value_counts.dtype != torch.int32
        or not torch.equal(key_counts, expected_key)
        or not torch.equal(value_counts, expected_value)
    ):
        raise FixtureValidationError("sparse active counts mismatch")
    key_values = sparse_values["key_selection_normalized_by_position"]
    key_indices = sparse_indices["key_selection_by_position"]
    value_values = sparse_values["value_selection_by_position"]
    value_indices = sparse_indices["value_selection_by_position"]
    for values, indices, label in (
        (key_values, key_indices, "Key"),
        (value_values, value_indices, "Value"),
    ):
        if (
            values.shape != (TOTAL_CONTEXT, CAPACITY)
            or values.dtype != torch.float32
            or indices.shape != (TOTAL_CONTEXT, CAPACITY)
            or indices.dtype != torch.int32
        ):
            raise FixtureValidationError(f"{label} sparse physical layout mismatch")
    for position in range(TOTAL_CONTEXT):
        count = int(key_counts[position].item())
        active = key_indices[position, :count].tolist()
        if len(active) != len(set(active)):
            raise FixtureValidationError("duplicate Key sparse index")
        tail = count // 2
        if set(active[:tail]) & set(active[tail:]):
            raise FixtureValidationError("Key lower/upper sparse overlap")
        if count and (
            not torch.all(key_values[position, :tail] < -1.0)
            or not torch.all(key_values[position, tail:count] > 1.0)
        ):
            raise FixtureValidationError("Key tail ordering mismatch")
        if count < CAPACITY and (
            torch.count_nonzero(key_values[position, count:]).item()
            or torch.count_nonzero(key_indices[position, count:]).item()
        ):
            raise FixtureValidationError("unused Key sparse slots are not zero")
        value_count = int(value_counts[position].item())
        value_active = value_indices[position, :value_count].tolist()
        if len(value_active) != len(set(value_active)):
            raise FixtureValidationError("duplicate Value sparse index")
        if position < SINK_TOKENS and (
            torch.count_nonzero(value_values[position]).item()
            or torch.count_nonzero(value_indices[position]).item()
        ):
            raise FixtureValidationError("sink Value sparse row is not zero")
    flattened_value = (
        inputs["value_after_v_proj"]
        .transpose(1, 2)
        .reshape(TOTAL_CONTEXT, KV_WIDTH)
        .float()
    )
    for position in range(SINK_TOKENS, TOTAL_CONTEXT):
        row = flattened_value[position]
        lower_expected = torch.argsort(
            row, descending=False, stable=True
        )[:ENTRIES_PER_TAIL]
        excluded = torch.zeros(KV_WIDTH, dtype=torch.bool)
        excluded[lower_expected] = True
        upper_candidates = torch.nonzero(~excluded, as_tuple=False).flatten()
        upper_order = torch.argsort(
            row[upper_candidates], descending=True, stable=True
        )
        upper_expected = upper_candidates[upper_order[:ENTRIES_PER_TAIL]]
        observed = value_indices[position].long()
        if not torch.equal(
            observed,
            torch.cat((lower_expected, upper_expected)),
        ):
            raise FixtureValidationError("Value six-low/six-high selection mismatch")
        if set(observed[:6].tolist()) & set(observed[6:].tolist()):
            raise FixtureValidationError("Value lower/upper sparse overlap")
        if not torch.equal(value_values[position], row[observed]):
            raise FixtureValidationError("Value sparse values mismatch")
    if (
        sparse_indices["value_tie_control_counts"].tolist() != [12, 0]
        or sparse_indices["value_tie_control"][0].tolist() != list(range(12))
        or not torch.equal(
            sparse_values["value_tie_control"][0],
            torch.full((12,), 0.25, dtype=torch.float32),
        )
        or torch.count_nonzero(
            sparse_indices["value_tie_control"][1]
        ).item()
        or torch.count_nonzero(sparse_values["value_tie_control"][1]).item()
    ):
        raise FixtureValidationError("Value equal-tie control mismatch")

    for prefix, expected_count in (("key", key_count), ("value", 12)):
        store_values = sparse_values[f"{prefix}_cache_after_store"]
        store_indices = sparse_indices[f"{prefix}_cache_after_store"]
        cache_values = sparse_values[f"{prefix}_cache_after_append"]
        cache_indices = sparse_indices[f"{prefix}_cache_after_append"]
        if (
            cache_values.shape != (TOTAL_CONTEXT, CAPACITY)
            or cache_indices.shape != (TOTAL_CONTEXT, CAPACITY)
            or cache_values.dtype != torch.float32
            or cache_indices.dtype != torch.int32
            or store_values.shape != (TOTAL_CONTEXT, CAPACITY)
            or store_indices.shape != (TOTAL_CONTEXT, CAPACITY)
            or store_values.dtype != torch.float32
            or store_indices.dtype != torch.int32
        ):
            raise FixtureValidationError(
                f"{prefix} cache sparse allocation mismatch"
            )
        if (
            not torch.equal(
                store_values[:STORE_QUANTIZED_CONTEXT],
                cache_values[:STORE_QUANTIZED_CONTEXT],
            )
            or not torch.equal(
                store_indices[:STORE_QUANTIZED_CONTEXT],
                cache_indices[:STORE_QUANTIZED_CONTEXT],
            )
            or torch.count_nonzero(
                store_values[STORE_QUANTIZED_CONTEXT:]
            ).item()
            or torch.count_nonzero(
                store_indices[STORE_QUANTIZED_CONTEXT:]
            ).item()
            or torch.count_nonzero(
                cache_values[QUANTIZED_CONTEXT:]
            ).item()
            or torch.count_nonzero(
                cache_indices[QUANTIZED_CONTEXT:]
            ).item()
        ):
            raise FixtureValidationError(
                f"{prefix} store/append sparse history mismatch"
            )
        for row_index in range(QUANTIZED_CONTEXT):
            if expected_count < CAPACITY and (
                torch.count_nonzero(
                    cache_values[row_index, expected_count:]
                ).item()
                or torch.count_nonzero(
                    cache_indices[row_index, expected_count:]
                ).item()
            ):
                raise FixtureValidationError(
                    f"{prefix} cache unused slots are not zero"
                )
            active = cache_indices[row_index, :expected_count].tolist()
            if len(active) != len(set(active)):
                raise FixtureValidationError(
                    f"duplicate {prefix} cache sparse index"
                )
            source_position = row_index + SINK_TOKENS
            expected_indices = (
                key_indices
                if prefix == "key"
                else value_indices
            )[source_position, :expected_count]
            if not torch.equal(
                cache_indices[row_index, :expected_count],
                expected_indices,
            ):
                raise FixtureValidationError(
                    f"{prefix} cache/selection index mismatch"
                )

    flattened_key = (
        inputs["key_pre_rope"]
        .transpose(1, 2)
        .reshape(TOTAL_CONTEXT, KV_WIDTH)
        .float()
    )
    key_lut = metadata["key_lookup_table"].reshape(KV_WIDTH, -1)
    key_cache_values = sparse_values["key_cache_after_append"]
    for row_index in range(QUANTIZED_CONTEXT):
        source_position = row_index + SINK_TOKENS
        indices = key_indices[source_position, :key_count].long()
        if key_count:
            selected_scores = key_values[
                source_position, :key_count
            ]
            nearest = torch.where(
                selected_scores < 0,
                key_lut[indices, 0],
                key_lut[indices, -1],
            )
            expected_values = (
                flattened_key[source_position, indices] - nearest
            )
            if not torch.equal(
                key_cache_values[row_index, :key_count],
                expected_values,
            ):
                raise FixtureValidationError(
                    "Key sparse correction value mismatch"
                )

    value_zero_code = {4: 7, 3: 3, 2: 1}[bit_width]
    value_lookup = metadata["value_lookup_after_append"]
    value_cache_values = sparse_values["value_cache_after_append"]
    for row_index in range(QUANTIZED_CONTEXT):
        source_position = row_index + SINK_TOKENS
        indices = value_indices[source_position].long()
        expected_values = (
            flattened_value[source_position, indices]
            - value_lookup[row_index, value_zero_code]
        )
        if not torch.equal(value_cache_values[row_index], expected_values):
            raise FixtureValidationError(
                "Value sparse correction value mismatch"
            )


def _validate_sink_and_rope(
    inputs: Mapping[str, Any],
    metadata: Mapping[str, Any],
    sink: Mapping[str, Any],
    store_state: Mapping[str, Any],
    append_state: Mapping[str, Any],
) -> None:
    import torch

    if (
        sink["sink_key_pre_rope_bf16"].shape
        != (1, NUM_KV_HEADS, SINK_TOKENS, HEAD_DIM)
        or sink["sink_key_pre_rope_bf16"].dtype != torch.bfloat16
        or sink["sink_key_attention_fp16"].shape
        != (1, NUM_KV_HEADS, HEAD_DIM, SINK_TOKENS)
        or sink["sink_key_attention_fp16"].dtype != torch.float16
        or sink["sink_value_fp16"].shape
        != (1, NUM_KV_HEADS, SINK_TOKENS, HEAD_DIM)
        or sink["sink_value_fp16"].dtype != torch.float16
        or sink["sink_positions"].tolist() != list(range(SINK_TOKENS))
    ):
        raise FixtureValidationError("sink layout mismatch")
    if not torch.equal(
        sink["sink_key_pre_rope_bf16"],
        inputs["key_pre_rope"][:, :, :SINK_TOKENS, :],
    ) or not torch.equal(
        sink["sink_value_fp16"],
        inputs["value_after_v_proj"][:, :, :SINK_TOKENS, :].to(
            torch.float16
        ),
    ):
        raise FixtureValidationError("sink input identity mismatch")
    inv_freq = metadata["rope_inv_freq"]
    expected_inv_freq = _expected_rope_inv_freq()
    if not torch.equal(inv_freq, expected_inv_freq):
        raise FixtureValidationError("Llama-3.1 RoPE fingerprint mismatch")
    expected_sink_key = (
        _explicit_rope(
            sink["sink_key_pre_rope_bf16"],
            inv_freq,
            torch.arange(SINK_TOKENS),
        )
        .transpose(2, 3)
        .to(torch.float16)
    )
    if not torch.equal(sink["sink_key_attention_fp16"], expected_sink_key):
        raise FixtureValidationError("sink post-RoPE Key mismatch")
    for state in (store_state, append_state):
        if (
            not torch.equal(state["sink_k"], sink["sink_key_attention_fp16"])
            or not torch.equal(state["sink_v"], sink["sink_value_fp16"])
            or state["sink_k"].shape[1] != NUM_KV_HEADS
            or state["sink_v"].shape[1] != NUM_KV_HEADS
        ):
            raise FixtureValidationError("sink changed across append")


def _validate_dense_store_append(
    dense: Mapping[str, Any],
    store_state: Mapping[str, Any],
    append_state: Mapping[str, Any],
    bit_width: int,
) -> None:
    import torch

    packed_rows = bit_width * HEAD_DIM // 32
    expected_full_shape = (NUM_KV_HEADS, packed_rows, TOTAL_CONTEXT)
    for state, length in ((store_state, 17), (append_state, 18)):
        if (
            state["k_dense_allocated"].shape != expected_full_shape
            or state["v_dense_allocated"].shape != expected_full_shape
            or state["k_dense_allocated"].dtype != torch.int32
            or state["v_dense_allocated"].dtype != torch.int32
            or state["k_dense_allocated"].shape[0] != NUM_KV_HEADS
            or state["v_dense_allocated"].shape[0] != NUM_KV_HEADS
            or state["k_length"].tolist() != [length]
            or state["v_length"].tolist() != [length]
        ):
            raise FixtureValidationError("store/append dense state mismatch")
    if not torch.equal(
        dense["key_packed_after_store"],
        store_state["k_dense_allocated"][:, :, :STORE_QUANTIZED_CONTEXT],
    ) or not torch.equal(
        dense["value_packed_after_store"],
        store_state["v_dense_allocated"][:, :, :STORE_QUANTIZED_CONTEXT],
    ):
        raise FixtureValidationError("dense store payload mismatch")
    if not torch.equal(
        dense["key_packed_after_append"],
        append_state["k_dense_allocated"][:, :, :QUANTIZED_CONTEXT],
    ) or not torch.equal(
        dense["value_packed_after_append"],
        append_state["v_dense_allocated"][:, :, :QUANTIZED_CONTEXT],
    ):
        raise FixtureValidationError("dense append payload mismatch")
    if not torch.equal(
        dense["key_appended_slot"],
        dense["key_packed_after_append"][:, :, -1],
    ) or not torch.equal(
        dense["value_appended_slot"],
        dense["value_packed_after_append"][:, :, -1],
    ):
        raise FixtureValidationError("appended dense slot mismatch")
    if not torch.equal(
        dense["key_packed_independent_control"],
        dense["key_packed_after_append"],
    ) or not torch.equal(
        dense["value_packed_independent_control"],
        dense["value_packed_after_append"],
    ):
        raise FixtureValidationError("dense pack round-trip control mismatch")
    if not torch.equal(
        store_state["k_dense_allocated"][:, :, :STORE_QUANTIZED_CONTEXT],
        append_state["k_dense_allocated"][:, :, :STORE_QUANTIZED_CONTEXT],
    ) or not torch.equal(
        store_state["v_dense_allocated"][:, :, :STORE_QUANTIZED_CONTEXT],
        append_state["v_dense_allocated"][:, :, :STORE_QUANTIZED_CONTEXT],
    ):
        raise FixtureValidationError("append changed stored dense history")
    if bit_width == 3:
        if torch.equal(
            dense["value_expected_nearest_after_store"],
            dense["value_packed_after_store"],
        ) or torch.equal(
            dense["value_expected_nearest_after_append"],
            dense["value_packed_after_append"],
        ):
            raise FixtureValidationError(
                "3-bit Value parallel source payload was not preserved"
            )
    elif not torch.equal(
        dense["value_expected_nearest_after_store"],
        dense["value_packed_after_store"],
    ) or not torch.equal(
        dense["value_expected_nearest_after_append"],
        dense["value_packed_after_append"],
    ):
        raise FixtureValidationError("Value nearest-code control mismatch")


def _validate_decode(decode: Mapping[str, Any]) -> None:
    import torch

    required_shapes = {
        "query_attention_ready": (1, NUM_QUERY_HEADS, 1, HEAD_DIM),
        "source_nonsink_key_logits": (
            1,
            NUM_QUERY_HEADS,
            1,
            QUANTIZED_CONTEXT,
        ),
        "attention_weights": (1, NUM_QUERY_HEADS, 1, TOTAL_CONTEXT),
        "source_decode_output": (1, NUM_QUERY_HEADS, 1, HEAD_DIM),
        "independent_nonsink_key_logits": (
            1,
            NUM_QUERY_HEADS,
            1,
            QUANTIZED_CONTEXT,
        ),
        "independent_attention_weights": (
            1,
            NUM_QUERY_HEADS,
            1,
            TOTAL_CONTEXT,
        ),
        "independent_decode_output": (1, NUM_QUERY_HEADS, 1, HEAD_DIM),
        "explicit_dense_key_reconstruction": (
            QUANTIZED_CONTEXT,
            NUM_KV_HEADS,
            HEAD_DIM,
        ),
        "explicit_dense_value_reconstruction": (
            QUANTIZED_CONTEXT,
            NUM_KV_HEADS,
            HEAD_DIM,
        ),
    }
    if set(decode) != set(required_shapes):
        raise FixtureValidationError("decode tensor set mismatch")
    for name, shape in required_shapes.items():
        tensor = decode[name]
        if tuple(tensor.shape) != shape or not torch.isfinite(tensor).all():
            raise FixtureValidationError(f"invalid decode tensor: {name}")
    if (
        decode["query_attention_ready"].dtype != torch.bfloat16
        or decode["source_decode_output"].dtype != torch.bfloat16
        or decode["independent_decode_output"].dtype != torch.bfloat16
        or decode["explicit_dense_key_reconstruction"].shape[1]
        != NUM_KV_HEADS
        or decode["explicit_dense_value_reconstruction"].shape[1]
        != NUM_KV_HEADS
    ):
        raise FixtureValidationError("decode dtype/GQA mismatch")
    _assert_close(
        decode["source_nonsink_key_logits"],
        decode["independent_nonsink_key_logits"],
        atol=KEY_LOGIT_ATOL,
        rtol=KEY_LOGIT_RTOL,
        label="Key logits",
    )
    _assert_close(
        decode["source_decode_output"],
        decode["independent_decode_output"],
        atol=DECODE_ATOL,
        rtol=DECODE_RTOL,
        label="decode output",
    )
    sums = decode["attention_weights"].float().sum(dim=-1)
    _assert_close(
        sums,
        torch.ones_like(sums),
        atol=2e-3,
        rtol=0.0,
        label="attention probability sum",
    )


def _expected_byte_breakdown(
    bit_width: int,
    key_count: int,
) -> dict[str, int]:
    levels = 1 << bit_width
    packed_rows = bit_width * HEAD_DIM // 32
    dense_each = NUM_KV_HEADS * packed_rows * TOTAL_CONTEXT * 4
    key_metadata = (
        NUM_KV_HEADS * HEAD_DIM * levels * 4
        + 3 * KV_WIDTH * 4
        + levels * 4
        + (HEAD_DIM // 2) * 4
    )
    value_metadata = TOTAL_CONTEXT * levels * 4 + levels * 4
    sparse_each = TOTAL_CONTEXT * CAPACITY * 4
    sink_each = NUM_KV_HEADS * SINK_TOKENS * HEAD_DIM * 2
    active_dense_each = (
        NUM_KV_HEADS * packed_rows * QUANTIZED_CONTEXT * 4
    )
    active_key_metadata = key_metadata
    active_value_metadata = QUANTIZED_CONTEXT * levels * 4 + levels * 4
    active = (
        2 * active_dense_each
        + active_key_metadata
        + active_value_metadata
        + key_count * QUANTIZED_CONTEXT * 8
        + CAPACITY * QUANTIZED_CONTEXT * 8
        + 2 * sink_each
    )
    return {
        "dense_k_payload_bytes": dense_each,
        "dense_v_payload_bytes": dense_each,
        "key_metadata_bytes": key_metadata,
        "value_metadata_bytes": value_metadata,
        "key_sparse_value_bytes": sparse_each,
        "key_sparse_index_bytes": sparse_each,
        "value_sparse_value_bytes": sparse_each,
        "value_sparse_index_bytes": sparse_each,
        "sink_k_bytes": sink_each,
        "sink_v_bytes": sink_each,
        "padding_alignment_bytes": 0,
        "persistent_reference_workspace_bytes": 0,
        "active_logical_total_bytes": active,
        "logical_bf16_bytes": (
            2 * BATCH_SIZE * NUM_KV_HEADS * TOTAL_CONTEXT * HEAD_DIM * 2
        ),
    }


def _validate_bytes(
    payload: Mapping[str, Any],
    family: str,
    case_name: str,
    bit_width: int,
    key_count: int,
) -> None:
    if (
        payload.get("schema_version")
        != "kvbench-phase10-byte-breakdown-1.0.0"
        or payload.get("fixture_id") != FIXTURE_ID
        or payload.get("family") != family
        or payload.get("case") != case_name
        or payload.get("bit_width") != bit_width
        or payload.get("allocation_basis") != "source_owned_tensor_storage"
        or payload.get("r_hbm") is not None
    ):
        raise FixtureValidationError("byte-breakdown identity mismatch")
    expected = _expected_byte_breakdown(bit_width, key_count)
    for name, value in expected.items():
        if payload.get(name) != value:
            raise FixtureValidationError(f"byte-breakdown mismatch: {name}")
    allocated_names = (
        "dense_k_payload_bytes",
        "dense_v_payload_bytes",
        "key_metadata_bytes",
        "value_metadata_bytes",
        "key_sparse_value_bytes",
        "key_sparse_index_bytes",
        "value_sparse_value_bytes",
        "value_sparse_index_bytes",
        "sink_k_bytes",
        "sink_v_bytes",
        "padding_alignment_bytes",
        "persistent_reference_workspace_bytes",
    )
    allocated = sum(int(payload[name]) for name in allocated_names)
    if payload.get("actual_allocated_total_bytes") != allocated:
        raise FixtureValidationError("allocated byte sum mismatch")
    logical_bf16 = expected["logical_bf16_bytes"]
    rho = allocated / logical_bf16
    reciprocal = logical_bf16 / allocated
    if (
        not math.isclose(payload.get("rho_alloc"), rho, rel_tol=0.0, abs_tol=0.0)
        or not math.isclose(
            payload.get("r_alloc"), reciprocal, rel_tol=0.0, abs_tol=0.0
        )
        or abs(payload["rho_alloc"] * payload["r_alloc"] - 1.0)
        > RECIPROCAL_ATOL
        or payload.get("reciprocal_tolerance") != RECIPROCAL_ATOL
    ):
        raise FixtureValidationError("allocation ratio mismatch")
    fixed = payload.get("fixed_capacity")
    if (
        not isinstance(fixed, dict)
        or fixed.get("key_slots_per_physical_row") != CAPACITY
        or fixed.get("value_slots_per_physical_row") != CAPACITY
        or fixed.get("key_active_entries")
        != key_count * QUANTIZED_CONTEXT
        or fixed.get("value_active_entries_non_sink")
        != CAPACITY * QUANTIZED_CONTEXT
        or fixed.get("value_active_entries_sink") != 0
    ):
        raise FixtureValidationError("fixed-cap byte accounting mismatch")


def _validate_one_fixture(
    fixture_root: Path,
    family: str,
    case_name: str,
    authority_hashes: Mapping[str, str],
) -> dict[str, Any]:
    _validate_fixture_ledger(fixture_root)
    bit_width = FAMILIES[family]
    key_count = CASES[case_name]
    manifest = _load_json(fixture_root / "fixture_manifest.json")
    _validate_geometry(
        manifest,
        family,
        case_name,
        bit_width,
        key_count,
        authority_hashes,
    )
    tensor_files = {
        filename: _load_tensors(fixture_root / filename)
        for filename in sorted(FIXTURE_MEMBERS)
        if filename.endswith(".safetensors")
    }
    _validate_tensor_records(manifest.get("tensor_records"), tensor_files)
    byte_breakdown = _load_json(fixture_root / "byte_breakdown.json")
    if manifest.get("byte_breakdown_sha256") != _sha256_bytes(
        (
            json.dumps(
                byte_breakdown,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    ):
        raise FixtureValidationError("byte-breakdown manifest hash mismatch")
    _validate_bytes(
        byte_breakdown, family, case_name, bit_width, key_count
    )
    inputs = tensor_files["inputs.safetensors"]
    dense = tensor_files["dense_payload.safetensors"]
    metadata = tensor_files["metadata.safetensors"]
    sparse_values = tensor_files["sparse_values.safetensors"]
    sparse_indices = tensor_files["sparse_indices.safetensors"]
    sink = tensor_files["sink.safetensors"]
    store_state = tensor_files["store_state.safetensors"]
    append_state = tensor_files["append_state.safetensors"]
    decode = tensor_files["decode_output.safetensors"]
    if (
        inputs["key_pre_rope"].shape
        != (1, NUM_KV_HEADS, TOTAL_CONTEXT, HEAD_DIM)
        or inputs["value_after_v_proj"].shape
        != (1, NUM_KV_HEADS, TOTAL_CONTEXT, HEAD_DIM)
        or inputs["query_pre_rope"].shape
        != (1, NUM_QUERY_HEADS, 1, HEAD_DIM)
        or inputs["position_ids"].tolist() != [list(range(TOTAL_CONTEXT))]
    ):
        raise FixtureValidationError("input geometry mismatch")
    _validate_sparse(
        inputs,
        metadata,
        sparse_values,
        sparse_indices,
        bit_width,
        key_count,
    )
    _validate_sink_and_rope(
        inputs, metadata, sink, store_state, append_state
    )
    _validate_dense_store_append(
        dense, store_state, append_state, bit_width
    )
    _validate_decode(decode)
    return {
        "family": family,
        "case": case_name,
        "bit_width": bit_width,
        "key_active_count": key_count,
        "value_active_count_non_sink": 12,
        "value_active_count_sink": 0,
        "allocated_bytes": byte_breakdown["actual_allocated_total_bytes"],
        "active_logical_bytes": byte_breakdown[
            "active_logical_total_bytes"
        ],
        "rho_alloc": byte_breakdown["rho_alloc"],
        "r_alloc": byte_breakdown["r_alloc"],
        "decode_sha256": _sha256_bytes(
            _tensor_bytes(decode["source_decode_output"])
        ),
    }


def validate_fixture_bundle(fixture_root: Path) -> dict[str, Any]:
    fixture_root = fixture_root.resolve(strict=True)
    if fixture_root != DEFAULT_FIXTURE_ROOT.resolve(strict=True):
        # Retrieval validation may target an alternate clean directory, but
        # parent authority manifests remain the tracked local authority.
        pass
    if fixture_root.parent == REFERENCE_ROOT:
        members = {path.name for path in REFERENCE_ROOT.iterdir()}
        if members != REFERENCE_MEMBERS:
            raise FixtureValidationError("unexpected reference-lane file")
    authority_hashes = _validate_authority()
    local_root, root_manifest = _validate_root_controls(
        fixture_root, authority_hashes
    )
    top_level = {path.name for path in fixture_root.iterdir()}
    expected_top_level = {
        *FAMILIES,
        "manifest.json",
        "artifact_inventory.json",
        "checksums.sha256",
        "COMPLETE",
        "reference_trace.json",
    }
    if top_level != expected_top_level:
        raise FixtureValidationError("unexpected fixture-root entry")
    legacy = {"no_outlier", "few_outliers", "cap_reached"}
    summaries = []
    for family in FAMILIES:
        family_root = fixture_root / family
        if family_root.is_symlink() or not family_root.is_dir():
            raise FixtureValidationError(f"invalid fixture family: {family}")
        cases = {path.name for path in family_root.iterdir()}
        if cases != set(CASES) or cases & legacy:
            raise FixtureValidationError(
                f"fixture case matrix mismatch: {family}"
            )
        for case_name in CASES:
            summaries.append(
                _validate_one_fixture(
                    family_root / case_name,
                    family,
                    case_name,
                    authority_hashes,
                )
            )
    if len(summaries) != 9:
        raise FixtureValidationError("fixture count mismatch")
    if root_manifest.get("fixture_matrix", {}).get("total") != len(summaries):
        raise FixtureValidationError("root fixture count mismatch")
    return {
        "status": "PASS",
        "fixture_id": FIXTURE_ID,
        "local_root_sha256": local_root,
        "fixture_count": len(summaries),
        "fixtures": summaries,
        "source": "PASS",
        "calibration": "PASS",
        "environment": "PASS",
        "build": "PASS",
        "g2_kvq": "NOT_EVALUATED",
        "full_scan": "CLOSED",
        "quality_execution": "LOCKED",
        "performance_data": False,
        "profiler_data": False,
        "quality_data": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Phase 10 KVQuant fixtures without regeneration"
    )
    parser.add_argument(
        "--fixtures",
        default=str(DEFAULT_FIXTURE_ROOT),
        help="finalized fixture root",
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        output = validate_fixture_bundle(Path(arguments.fixtures))
    except (
        FixtureValidationError,
        OSError,
        RuntimeError,
        TypeError,
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
