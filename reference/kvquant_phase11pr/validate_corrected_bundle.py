#!/usr/bin/env python3
"""Validate the narrow mixed-provenance KVQuant kvq3 correction bundle."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from functools import reduce
import hashlib
import json
import math
import operator
from pathlib import Path, PurePosixPath
import stat
import struct
from typing import Any

from reference.kvquant import validate_fixtures as legacy
from scripts.r2_artifact import (
    ArtifactValidationError,
    validate_local_artifact,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE_ROOT = (
    REPOSITORY_ROOT / "reference" / "kvquant_phase11pr" / "fixtures"
)
DEFAULT_OLD_FIXTURE_ROOT = (
    REPOSITORY_ROOT / "reference" / "kvquant" / "fixtures"
)
OLD_CALIBRATION_MANIFEST = (
    REPOSITORY_ROOT / "reference" / "kvquant" / "calibration_manifest.json"
)
PATCH_MANIFEST = (
    REPOSITORY_ROOT
    / "third_party"
    / "patches"
    / "kvquant"
    / "graphsafe-kvq3-manifest.json"
)

FIXTURE_ID = "kvqref-2e0a0e9022c50cbc6fb497d88cae973e"
OLD_FIXTURE_ID = "kvqref-a50af6511c314b6394e58a7f81ceefb8"
OLD_ROOT_SHA256 = (
    "32cdf465a361dd6695b66ccbea0a462bddc075fd9778d0aa8cdaa3f94e6f63ab"
)
PATCH_SHA256 = (
    "23a15db86790299392412c3ce2da7d971f4f073cfaf6839d82d3746c8b56b551"
)
PATCHED_COMMIT = "0d9df350bd1788284e1ce76a8bf6e886beca5efa"
PATCHED_TREE = "a85cf7bf093982a4bf89c33d4e6794d9a85f846d"
EXTENSION_SHA256 = (
    "46c41aad8f56d58608d4c1273bd3a72fd36c8f69f9ca2c5a046f0c811631bf51"
)
MEASUREMENT_CONTAINER = (
    "sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e"
)
REFERENCE_IMAGE = legacy.REFERENCE_IMAGE_CONFIG_DIGEST
OLD_FAMILY_SHA256 = {
    "kvq4": "52bfbdf2c29f546b391b6079497aaf5c3d17a3a125dd0d096c748cc0fae2e0a8",
    "kvq2": "7625600b6b0a5341d542a40edf13693354a9e6a11847a82440349eaed2d927ac",
}
FAMILIES = ("kvq4", "kvq3", "kvq2")
CASES = dict(legacy.CASES)
AUTHORITY_MEMBERS = {
    "source_manifest.json",
    "environment.json",
    "calibration_manifest.json",
    "build_manifest.json",
}
ROOT_MEMBERS = {
    "manifest.json",
    "artifact_inventory.json",
    "checksums.sha256",
    "COMPLETE",
    "reference_trace.json",
    "reuse_proof.json",
    "authority",
    *FAMILIES,
}

TENSOR_MEMBERS = tuple(
    sorted(name for name in legacy.FIXTURE_MEMBERS if name.endswith(".safetensors"))
)
EXPECTED_TENSOR_NAMES = {
    "inputs.safetensors": {
        "key_pre_rope",
        "position_ids",
        "query_pre_rope",
        "value_after_v_proj",
        "value_tie_control_rows",
        "value_tie_control_sink_mask",
    },
    "dense_payload.safetensors": {
        "key_appended_slot",
        "key_packed_after_append",
        "key_packed_after_store",
        "key_packed_independent_control",
        "value_appended_slot",
        "value_expected_nearest_after_append",
        "value_expected_nearest_after_store",
        "value_packed_after_append",
        "value_packed_after_store",
        "value_packed_independent_control",
        "value_scalar_control_after_append",
        "value_scalar_control_after_store",
    },
    "metadata.safetensors": {
        "key_codebook",
        "key_lookup_table",
        "key_runtime_lower_threshold",
        "key_runtime_upper_threshold",
        "key_runtime_zero",
        "rope_inv_freq",
        "value_codebook",
        "value_dense_lower_bound",
        "value_dense_upper_bound",
        "value_lookup_after_append",
        "value_lookup_after_store",
    },
    "sparse_values.safetensors": {
        "key_cache_after_append",
        "key_cache_after_store",
        "key_selection_normalized_by_position",
        "value_cache_after_append",
        "value_cache_after_store",
        "value_selection_by_position",
        "value_tie_control",
    },
    "sparse_indices.safetensors": {
        "key_active_count_by_position",
        "key_cache_after_append",
        "key_cache_after_store",
        "key_selection_by_position",
        "value_active_count_by_position",
        "value_cache_after_append",
        "value_cache_after_store",
        "value_selection_by_position",
        "value_tie_control",
        "value_tie_control_counts",
    },
    "sink.safetensors": {
        "sink_key_attention_fp16",
        "sink_key_pre_rope_bf16",
        "sink_positions",
        "sink_value_fp16",
    },
    "store_state.safetensors": {
        "k_dense_allocated",
        "k_length",
        "k_sparse_indices_allocated",
        "k_sparse_values_allocated",
        "sink_k",
        "sink_v",
        "v_dense_allocated",
        "v_length",
        "v_lookup_allocated",
        "v_sparse_indices_allocated",
        "v_sparse_values_allocated",
    },
    "append_state.safetensors": {
        "k_dense_allocated",
        "k_length",
        "k_sparse_indices_allocated",
        "k_sparse_values_allocated",
        "sink_k",
        "sink_v",
        "v_dense_allocated",
        "v_length",
        "v_lookup_allocated",
        "v_sparse_indices_allocated",
        "v_sparse_values_allocated",
    },
    "decode_output.safetensors": {
        "attention_weights",
        "explicit_dense_key_reconstruction",
        "explicit_dense_value_reconstruction",
        "independent_attention_weights",
        "independent_decode_output",
        "independent_nonsink_key_logits",
        "query_attention_ready",
        "source_decode_output",
        "source_nonsink_key_logits",
    },
}


class CorrectedBundleValidationError(RuntimeError):
    """The corrected fixture bundle violates its frozen narrow contract."""


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


def _tensor_bytes(tensor: Any) -> bytes:
    import torch

    values = (
        tensor.detach()
        .contiguous()
        .cpu()
        .view(torch.uint8)
        .reshape(-1)
        .tolist()
    )
    return bytes(values)


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
        raise CorrectedBundleValidationError(
            f"unsupported tensor dtype: {tensor.dtype}"
        ) from error


def _validate_tensor_records(
    records: Any,
    files: Mapping[str, Mapping[str, Any]],
) -> None:
    if not isinstance(records, dict) or set(records) != set(files):
        raise CorrectedBundleValidationError("tensor-record file set mismatch")
    for filename, tensors in files.items():
        declared = records.get(filename)
        if not isinstance(declared, dict) or set(declared) != set(tensors):
            raise CorrectedBundleValidationError(
                f"tensor-record name set mismatch: {filename}"
            )
        for name, tensor in tensors.items():
            record = declared[name]
            dtype = _dtype_name(tensor)
            expected_bytes = tensor.numel() * tensor.element_size()
            if (
                not isinstance(record, dict)
                or set(record)
                != {"dtype", "shape", "logical_nbytes", "payload_sha256"}
                or record["dtype"] != dtype
                or record["shape"] != list(tensor.shape)
                or record["logical_nbytes"] != expected_bytes
                or record["payload_sha256"]
                != hashlib.sha256(_tensor_bytes(tensor)).hexdigest()
            ):
                raise CorrectedBundleValidationError(
                    f"tensor record mismatch: {filename}:{name}"
                )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return legacy._load_json(path)
    except legacy.FixtureValidationError as error:
        raise CorrectedBundleValidationError(str(error)) from error


def _tree_digest(root: Path) -> tuple[str, int]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(f"{_sha256_file(path)}  {relative}\n".encode("utf-8"))
    return digest.hexdigest(), len(files)


def _shape_elements(shape: Any) -> int:
    if (
        not isinstance(shape, list)
        or any(type(item) is not int or item < 0 for item in shape)
    ):
        raise CorrectedBundleValidationError("invalid safetensors shape")
    return reduce(operator.mul, shape, 1)


def _reject_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CorrectedBundleValidationError(
                f"duplicate safetensors JSON key: {key}"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise CorrectedBundleValidationError(
        f"non-finite safetensors JSON value: {value}"
    )


def _read_safetensors(path: Path, expected_fixture_id: str) -> dict[str, Any]:
    """Read the small frozen tensor subset without importing safetensors."""

    import torch

    try:
        legacy._require_regular(path, immutable=True)
        raw = path.read_bytes()
    except (OSError, legacy.FixtureValidationError) as error:
        raise CorrectedBundleValidationError(
            f"unsafe safetensors file: {path}"
        ) from error
    if len(raw) < 10:
        raise CorrectedBundleValidationError(f"truncated safetensors file: {path}")
    header_size = struct.unpack("<Q", raw[:8])[0]
    if header_size < 2 or header_size > len(raw) - 8:
        raise CorrectedBundleValidationError(
            f"invalid safetensors header length: {path}"
        )
    try:
        header = json.loads(
            raw[8 : 8 + header_size].decode("utf-8"),
            object_pairs_hook=_reject_json_pairs,
            parse_constant=_reject_json_constant,
        )
    except CorrectedBundleValidationError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CorrectedBundleValidationError(
            f"invalid safetensors header: {path}"
        ) from error
    if not isinstance(header, dict):
        raise CorrectedBundleValidationError(
            f"safetensors header is not an object: {path}"
        )
    metadata = header.pop("__metadata__", None)
    if metadata != {
        "format": "kvbench-phase10-kvquant-reference",
        "fixture_id": expected_fixture_id,
    }:
        raise CorrectedBundleValidationError(
            f"safetensors metadata mismatch: {path}"
        )
    dtype_map = {
        "BOOL": (torch.bool, 1),
        "BF16": (torch.bfloat16, 2),
        "F16": (torch.float16, 2),
        "F32": (torch.float32, 4),
        "F64": (torch.float64, 8),
        "I32": (torch.int32, 4),
        "I64": (torch.int64, 8),
    }
    payload = raw[8 + header_size :]
    spans: list[tuple[int, int, str]] = []
    records: list[tuple[str, Any, list[int], int, int]] = []
    for name, record in header.items():
        if (
            not isinstance(name, str)
            or not name
            or "/" in name
            or "\\" in name
            or not isinstance(record, dict)
            or set(record) != {"dtype", "shape", "data_offsets"}
        ):
            raise CorrectedBundleValidationError(
                f"invalid safetensors tensor record: {path}"
            )
        dtype_name = record["dtype"]
        if dtype_name not in dtype_map:
            raise CorrectedBundleValidationError(
                f"unsupported safetensors dtype {dtype_name!r}: {path}"
            )
        shape = record["shape"]
        elements = _shape_elements(shape)
        offsets = record["data_offsets"]
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(item) is not int for item in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or offsets[1] > len(payload)
        ):
            raise CorrectedBundleValidationError(
                f"invalid safetensors offsets: {path}:{name}"
            )
        dtype, item_size = dtype_map[dtype_name]
        if offsets[1] - offsets[0] != elements * item_size:
            raise CorrectedBundleValidationError(
                f"safetensors byte length mismatch: {path}:{name}"
            )
        spans.append((offsets[0], offsets[1], name))
        records.append((name, dtype, shape, offsets[0], offsets[1]))
    if not records:
        raise CorrectedBundleValidationError(f"empty safetensors file: {path}")
    cursor = 0
    for start, end, _ in sorted(spans):
        if start != cursor:
            raise CorrectedBundleValidationError(
                f"non-contiguous safetensors payload: {path}"
            )
        cursor = end
    if cursor != len(payload):
        raise CorrectedBundleValidationError(
            f"unclaimed safetensors payload bytes: {path}"
        )
    tensors: dict[str, Any] = {}
    for name, dtype, shape, start, end in records:
        storage = bytearray(payload[start:end])
        tensor = torch.frombuffer(storage, dtype=dtype).clone()
        tensors[name] = tensor.reshape(shape).contiguous()
    return {name: tensors[name] for name in sorted(tensors)}


def _expected_paths(root: Path) -> tuple[set[Path], set[Path]]:
    files = {
        root / "manifest.json",
        root / "artifact_inventory.json",
        root / "checksums.sha256",
        root / "COMPLETE",
        root / "reference_trace.json",
        root / "reuse_proof.json",
        *(root / "authority" / name for name in AUTHORITY_MEMBERS),
    }
    directories = {root, root / "authority"}
    for family in FAMILIES:
        family_root = root / family
        directories.add(family_root)
        for case_name in CASES:
            fixture_root = family_root / case_name
            directories.add(fixture_root)
            files.update(
                fixture_root / member for member in legacy.FIXTURE_MEMBERS
            )
    return files, directories


def _validate_exact_layout(root: Path) -> int:
    try:
        metadata = root.lstat()
    except OSError as error:
        raise CorrectedBundleValidationError(
            "corrected fixture root is missing"
        ) from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise CorrectedBundleValidationError("corrected fixture root is unsafe")
    if {path.name for path in root.iterdir()} != ROOT_MEMBERS:
        raise CorrectedBundleValidationError(
            "unexpected corrected fixture root entry"
        )
    expected_files, expected_directories = _expected_paths(root)
    observed_files: set[Path] = set()
    observed_directories = {root}
    for path in root.rglob("*"):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise CorrectedBundleValidationError("symlinks are forbidden")
        if stat.S_ISDIR(metadata.st_mode):
            observed_directories.add(path)
        elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
            observed_files.add(path)
        else:
            raise CorrectedBundleValidationError(
                f"unsafe corrected fixture entry: {path}"
            )
    if observed_files != expected_files:
        raise CorrectedBundleValidationError(
            "corrected bundle file set is not exact"
        )
    if observed_directories != expected_directories:
        raise CorrectedBundleValidationError(
            "corrected bundle directory set is not exact"
        )
    for family in FAMILIES:
        family_root = root / family
        if {path.name for path in family_root.iterdir()} != set(CASES):
            raise CorrectedBundleValidationError(
                f"fixture case set mismatch: {family}"
            )
        for case_name in CASES:
            members = {
                path.name for path in (family_root / case_name).iterdir()
            }
            if members != set(legacy.FIXTURE_MEMBERS):
                raise CorrectedBundleValidationError(
                    f"fixture member set mismatch: {family}/{case_name}"
                )
    return len(observed_files)


def _validate_reused_family(
    corrected_root: Path,
    old_root: Path,
    family: str,
) -> dict[str, Any]:
    corrected = corrected_root / family
    old = old_root / family
    corrected_digest, corrected_count = _tree_digest(corrected)
    old_digest, old_count = _tree_digest(old)
    if (
        old_digest != OLD_FAMILY_SHA256[family]
        or corrected_digest != old_digest
        or corrected_count != old_count
        or corrected_count != len(CASES) * len(legacy.FIXTURE_MEMBERS)
    ):
        raise CorrectedBundleValidationError(
            f"{family} was not reused byte-identically"
        )
    corrected_files = {
        path.relative_to(corrected).as_posix(): path
        for path in corrected.rglob("*")
        if path.is_file()
    }
    old_files = {
        path.relative_to(old).as_posix(): path
        for path in old.rglob("*")
        if path.is_file()
    }
    if set(corrected_files) != set(old_files):
        raise CorrectedBundleValidationError(
            f"{family} reused path set mismatch"
        )
    for relative in sorted(old_files):
        source = old_files[relative]
        copied = corrected_files[relative]
        if (
            _sha256_file(source) != _sha256_file(copied)
            or source.read_bytes() != copied.read_bytes()
        ):
            raise CorrectedBundleValidationError(
                f"{family} reused bytes differ: {relative}"
            )
        source_stat = source.stat()
        copied_stat = copied.stat()
        if (
            source_stat.st_dev == copied_stat.st_dev
            and source_stat.st_ino == copied_stat.st_ino
        ):
            raise CorrectedBundleValidationError(
                f"{family} reused file is not an independent copy"
            )
    return {
        "source_fixture_id": OLD_FIXTURE_ID,
        "source_root_sha256": OLD_ROOT_SHA256,
        "family_tree_sha256": corrected_digest,
        "file_count": corrected_count,
        "copy_mode": "ordinary_files_byte_identical",
    }


def _validate_authority(root: Path) -> dict[str, str]:
    authority = root / "authority"
    if {path.name for path in authority.iterdir()} != AUTHORITY_MEMBERS:
        raise CorrectedBundleValidationError("authority member set mismatch")
    source = _load_json(authority / "source_manifest.json")
    environment = _load_json(authority / "environment.json")
    calibration = _load_json(authority / "calibration_manifest.json")
    build = _load_json(authority / "build_manifest.json")
    if (
        source.get("schema_version") != "kvbench-phase11pr-source-1.0.0"
        or source.get("status") != "PASS"
        or source.get("decision") != "0025"
        or source.get("upstream_base_commit") != legacy.UPSTREAM_BASE_COMMIT
        or source.get("upstream_base_tree") != legacy.UPSTREAM_BASE_TREE
        or source.get("aggregate_patch_sha256") != PATCH_SHA256
        or source.get("patched_commit") != PATCHED_COMMIT
        or source.get("patched_tree") != PATCHED_TREE
        or source.get("patch_manifest_sha256")
        != _sha256_file(PATCH_MANIFEST)
        or source.get("source_mount") != "read_only"
        or source.get("official_author_gqa_support_claimed") is not False
        or source.get("reconstruction")
        != {
            "applied_patch_sha256": PATCH_SHA256,
            "base_commit": legacy.UPSTREAM_BASE_COMMIT,
            "base_tree": legacy.UPSTREAM_BASE_TREE,
            "changed_file_count": 18,
            "patched_tree": PATCHED_TREE,
        }
    ):
        raise CorrectedBundleValidationError("corrected source authority mismatch")
    if environment != {
        "schema_version": "kvbench-phase11pr-environment-1.0.0",
        "status": "PASS",
        "fixture_generation_image": REFERENCE_IMAGE,
        "cuda_validation_container": MEASUREMENT_CONTAINER,
        "network": "disabled",
        "source_mount": "read_only",
        "calibration_mount": "read_only",
        "credentials_in_container": False,
    }:
        raise CorrectedBundleValidationError(
            "corrected environment authority mismatch"
        )
    if (authority / "calibration_manifest.json").read_bytes() != (
        OLD_CALIBRATION_MANIFEST.read_bytes()
    ):
        raise CorrectedBundleValidationError(
            "calibration authority was not reused byte-identically"
        )
    if (
        calibration.get("calibration_id") != legacy.CALIBRATION_ID
        or calibration.get("calibration_root_sha256")
        != legacy.CALIBRATION_ROOT
        or calibration.get("fisher_regenerated") is not False
        or calibration.get("quantizers_regenerated") is not False
    ):
        raise CorrectedBundleValidationError(
            "corrected calibration authority mismatch"
        )
    if build != {
        "schema_version": "kvbench-phase11pr-build-1.0.0",
        "status": "PASS",
        "authorized_measurement_container": MEASUREMENT_CONTAINER,
        "patched_tree": PATCHED_TREE,
        "extension_sha256": EXTENSION_SHA256,
        "sm_120_cubin": True,
        "compute_120_ptx": True,
        "extension_published": False,
    }:
        raise CorrectedBundleValidationError("corrected build authority mismatch")
    return {
        "source_manifest": _sha256_file(authority / "source_manifest.json"),
        "environment": _sha256_file(authority / "environment.json"),
        "calibration_manifest": _sha256_file(
            authority / "calibration_manifest.json"
        ),
        "build_manifest": _sha256_file(authority / "build_manifest.json"),
    }


def _validate_root_manifest(
    root: Path,
    authority_hashes: Mapping[str, str],
    reuse: Mapping[str, Mapping[str, Any]],
) -> None:
    manifest = _load_json(root / "manifest.json")
    expected_keys = {
        "schema_version",
        "run_id",
        "status",
        "phase",
        "run_kind",
        "decision",
        "method_identifier",
        "source",
        "calibration",
        "authority_manifest_sha256",
        "fixture_matrix",
        "family_provenance",
        "sparse_contract",
        "gates",
        "performance_measurement",
        "profiler_execution",
        "quality_evaluation",
    }
    if set(manifest) != expected_keys:
        raise CorrectedBundleValidationError(
            "corrected root manifest field set mismatch"
        )
    if (
        manifest["schema_version"]
        != "kvbench-phase11pr-kvquant-reference-bundle-1.0.0"
        or manifest["run_id"] != FIXTURE_ID
        or manifest["status"] != "completed"
        or manifest["phase"] != "phase11p_r_kvq3_value_pack_correction"
        or manifest["run_kind"] != "reference_fixture_correction"
        or manifest["decision"] != "0025"
        or manifest["method_identifier"]
        != "kvquant_gqa_graphsafe_kvq3_v2"
        or manifest["source"]
        != {
            "upstream_base_commit": legacy.UPSTREAM_BASE_COMMIT,
            "upstream_base_tree": legacy.UPSTREAM_BASE_TREE,
            "aggregate_patch_sha256": PATCH_SHA256,
            "patched_commit": PATCHED_COMMIT,
            "patched_tree": PATCHED_TREE,
            "extension_sha256": EXTENSION_SHA256,
        }
        or manifest["calibration"]
        != {
            "calibration_id": legacy.CALIBRATION_ID,
            "root_sha256": legacy.CALIBRATION_ROOT,
            "fisher_regenerated": False,
            "quantizers_regenerated": False,
        }
        or manifest["authority_manifest_sha256"] != dict(authority_hashes)
        or manifest["fixture_matrix"]
        != {
            "families": list(FAMILIES),
            "cases": list(CASES),
            "total": 9,
            "legacy_ambiguous_aliases": False,
        }
        or manifest["sparse_contract"]
        != {
            "key_counts": [0, 6, 12],
            "value_non_sink_count": 12,
            "value_sink_count": 0,
            "value_selection": "six_lowest_plus_six_highest",
        }
        or manifest["gates"]
        != {
            "g2_kvq": "NOT_EVALUATED",
            "global_g2_g5": "NOT_EVALUATED",
            "full_scan": "CLOSED",
            "quality_execution": "LOCKED",
            "performance_data_frozen_present": False,
        }
        or manifest["performance_measurement"] is not False
        or manifest["profiler_execution"] is not False
        or manifest["quality_evaluation"] is not False
    ):
        raise CorrectedBundleValidationError(
            "corrected root manifest authority mismatch"
        )
    expected_provenance = {
        "kvq4": dict(reuse["kvq4"]),
        "kvq3": {
            "fixture_id": FIXTURE_ID,
            "mode": "regenerated_deterministic_kvq3_only",
            "source_decision": "0025",
            "scalar_control": "PASS",
        },
        "kvq2": dict(reuse["kvq2"]),
    }
    if manifest["family_provenance"] != expected_provenance:
        raise CorrectedBundleValidationError(
            "mixed family provenance mismatch"
        )

    proof = _load_json(root / "reuse_proof.json")
    if proof != {
        "schema_version": "kvbench-phase11pr-reuse-proof-1.0.0",
        "old_fixture_id": OLD_FIXTURE_ID,
        "old_root_sha256": OLD_ROOT_SHA256,
        "families": {
            "kvq4": dict(reuse["kvq4"]),
            "kvq2": dict(reuse["kvq2"]),
        },
        "kvq3_regenerated_cases": list(CASES),
    }:
        raise CorrectedBundleValidationError("family reuse proof mismatch")
    trace = _load_json(root / "reference_trace.json")
    if trace != {
        "schema_version": "kvbench-phase11pr-reference-trace-1.0.0",
        "run_id": FIXTURE_ID,
        "run_kind": "reference_trace",
        "timing_fields_present": False,
        "operations": {
            "kvq4": "byte_reused",
            "kvq2": "byte_reused",
            "kvq3": "deterministic_per_token_per_channel_pack",
            "backend_fallback": False,
            "complete_prefix_temporary": False,
        },
    }:
        raise CorrectedBundleValidationError(
            "corrected reference trace mismatch"
        )


def _validate_kvq3_manifest(
    manifest: Mapping[str, Any],
    case_name: str,
    key_count: int,
    authority_hashes: Mapping[str, str],
) -> None:
    expected_top_level = {
        "schema_version",
        "fixture_id",
        "family",
        "case",
        "bit_width",
        "status",
        "run_kind",
        "source",
        "calibration",
        "environment",
        "geometry",
        "sparse_contract",
        "semantics",
        "packing",
        "execution_path",
        "numerical_control",
        "byte_breakdown_sha256",
        "tensor_records",
        "performance_measurement",
        "profiler_execution",
        "quality_evaluation",
        "g2_kvq",
    }
    if (
        set(manifest) != expected_top_level
        or manifest.get("schema_version")
        != "kvbench-phase11pr-kvquant-fixture-1.0.0"
        or manifest.get("fixture_id") != FIXTURE_ID
        or manifest.get("family") != "kvq3"
        or manifest.get("case") != case_name
        or manifest.get("bit_width") != 3
        or manifest.get("status") != "PASS"
        or manifest.get("run_kind") != "reference_fixture"
    ):
        raise CorrectedBundleValidationError(
            f"kvq3 fixture identity mismatch: {case_name}"
        )
    if manifest.get("source") != {
        "method_identifier": legacy.METHOD_IDENTIFIER,
        "decision": "0025",
        "contract_decision": "0023",
        "patched_commit": PATCHED_COMMIT,
        "patched_tree": PATCHED_TREE,
        "patch_sha256": PATCH_SHA256,
        "source_manifest_sha256": authority_hashes["source_manifest"],
    }:
        raise CorrectedBundleValidationError("kvq3 source binding mismatch")
    if manifest.get("calibration") != {
        "calibration_id": legacy.CALIBRATION_ID,
        "root_sha256": legacy.CALIBRATION_ROOT,
        "quantizer_sha256": legacy.QUANTIZER_SHA256["kvq3"],
        "calibration_manifest_sha256": authority_hashes[
            "calibration_manifest"
        ],
        "fisher_regenerated": False,
        "quantizer_regenerated": False,
    }:
        raise CorrectedBundleValidationError("kvq3 calibration binding mismatch")
    if manifest.get("environment") != {
        "image_config_digest": REFERENCE_IMAGE,
        "environment_manifest_sha256": authority_hashes["environment"],
        "build_manifest_sha256": authority_hashes["build_manifest"],
    }:
        raise CorrectedBundleValidationError("kvq3 environment binding mismatch")
    if manifest.get("geometry") != {
        "batch_size": 1,
        "num_query_heads": 32,
        "num_kv_heads": 8,
        "num_kv_groups": 4,
        "head_dim": 128,
        "interface_dtype": "bfloat16",
        "sink_dtype": "float16",
        "sink_tokens": 5,
        "store_context": 17,
        "append_tokens": 1,
        "total_context": 18,
        "seed": 20260729,
        "query_to_kv_mapping": "kv_head = query_head // 4",
    }:
        raise CorrectedBundleValidationError("kvq3 geometry mismatch")
    if manifest.get("sparse_contract") != {
        "key_sparse_selection_mode": "thresholded_fixed_tail_cap",
        "key_active_count": key_count,
        "key_capacity": 12,
        "value_sparse_selection_mode": "fixed_extrema",
        "value_active_count_non_sink": 12,
        "value_active_count_sink": 0,
        "value_capacity": 12,
        "value_lower_entries": 6,
        "value_upper_entries": 6,
        "value_occupancy_data_dependent": False,
        "unused_key_slots_zero": True,
        "unused_value_slots_non_sink": 0,
        "outlier_value_dtype": "float32",
        "outlier_index_dtype": "int32",
        "ties": "stable_value_then_flat_index",
    }:
        raise CorrectedBundleValidationError("kvq3 sparse contract mismatch")
    if manifest.get("semantics") != {
        "quantized_key": "pre_rope_k_proj_output",
        "sink_key_stored": "post_rope_attention_ready_fp16",
        "attention_key": "native_llama31_rope_applied_during_reference_decode",
        "value": "native_v_proj_output_without_rope",
        "position_ids": list(range(18)),
        "sink_positions": list(range(5)),
        "dense_quantized_positions": list(range(5, 18)),
        "implementation_head_expansion": False,
        "independent_control_head_expansion": True,
    }:
        raise CorrectedBundleValidationError(
            "kvq3 pre-/post-RoPE semantics mismatch"
        )
    if manifest.get("packing") != {
        "order": (
            "channel_codes_lsb_first_in_contiguous_int32_words;"
            "three_bit_codes_cross_word_boundaries"
        ),
        "packed_dtype": "int32",
        "packed_rows_per_kv_head": 12,
        "native_kv_heads": 8,
        "value_parallel_store_matches_intended_nearest": True,
        "value_after_append_matches_intended_nearest": True,
        "three_bit_parallel_value_source_behavior": (
            "deterministic_per_token_per_channel_nearest"
        ),
    }:
        raise CorrectedBundleValidationError("kvq3 packing contract mismatch")
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
        raise CorrectedBundleValidationError("kvq3 execution path mismatch")
    numerical = manifest.get("numerical_control")
    if (
        not isinstance(numerical, dict)
        or set(numerical)
        != {
            "key_logits_atol",
            "key_logits_rtol",
            "decode_atol",
            "decode_rtol",
            "key_errors",
            "decode_errors",
            "finite",
            "kvq3_scalar_code_pack",
        }
        or numerical.get("key_logits_atol") != legacy.KEY_LOGIT_ATOL
        or numerical.get("key_logits_rtol") != legacy.KEY_LOGIT_RTOL
        or numerical.get("decode_atol") != legacy.DECODE_ATOL
        or numerical.get("decode_rtol") != legacy.DECODE_RTOL
        or numerical.get("finite") is not True
        or numerical.get("kvq3_scalar_code_pack") != "PASS"
    ):
        raise CorrectedBundleValidationError("kvq3 numerical contract mismatch")
    for label in ("key_errors", "decode_errors"):
        errors = numerical[label]
        if (
            not isinstance(errors, dict)
            or set(errors) != {"max_abs", "max_rel"}
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
                for value in errors.values()
            )
        ):
            raise CorrectedBundleValidationError(
                f"invalid kvq3 {label}"
            )
    if (
        manifest.get("performance_measurement") is not False
        or manifest.get("profiler_execution") is not False
        or manifest.get("quality_evaluation") is not False
        or manifest.get("g2_kvq") != "NOT_EVALUATED"
    ):
        raise CorrectedBundleValidationError("kvq3 governance mismatch")


def _validate_metadata(metadata: Mapping[str, Any]) -> None:
    import torch

    expected = {
        "key_codebook": ((8,), torch.float32),
        "key_lookup_table": ((8, 128, 8), torch.float32),
        "key_runtime_lower_threshold": ((1024,), torch.float32),
        "key_runtime_upper_threshold": ((1024,), torch.float32),
        "key_runtime_zero": ((1024,), torch.float32),
        "rope_inv_freq": ((64,), torch.float32),
        "value_codebook": ((8,), torch.float32),
        "value_dense_lower_bound": ((18,), torch.float32),
        "value_dense_upper_bound": ((18,), torch.float32),
        "value_lookup_after_append": ((18, 8), torch.float32),
        "value_lookup_after_store": ((18, 8), torch.float32),
    }
    if set(metadata) != set(expected):
        raise CorrectedBundleValidationError("kvq3 metadata tensor set mismatch")
    for name, (shape, dtype) in expected.items():
        if tuple(metadata[name].shape) != shape or metadata[name].dtype != dtype:
            raise CorrectedBundleValidationError(
                f"kvq3 metadata tensor mismatch: {name}"
            )


def _float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _scalar_value_pack(
    inputs: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Any:
    import torch

    values = (
        inputs["value_after_v_proj"][0, :, 5:18, :]
        .permute(1, 0, 2)
        .float()
        .contiguous()
    )
    lookup = metadata["value_lookup_after_append"][:13].float().contiguous()
    lower = metadata["value_dense_lower_bound"][5:18].float().contiguous()
    upper = metadata["value_dense_upper_bound"][5:18].float().contiguous()
    codes = torch.empty((13, 8, 128), dtype=torch.int64)
    for token in range(13):
        lower_value = float(lower[token].item())
        upper_value = float(upper[token].item())
        for head in range(8):
            for channel in range(128):
                value = float(values[token, head, channel].item())
                if value < lower_value or value > upper_value:
                    code = 3
                else:
                    code = 0
                    best = abs(
                        _float32(value - float(lookup[token, 0].item()))
                    )
                    for candidate in range(1, 8):
                        distance = abs(
                            _float32(
                                value
                                - float(lookup[token, candidate].item())
                            )
                        )
                        if distance < best:
                            best = distance
                            code = candidate
                codes[token, head, channel] = code
    packed = torch.zeros((8, 12, 13), dtype=torch.int64)
    mask32 = (1 << 32) - 1
    for token in range(13):
        for head in range(8):
            for channel in range(128):
                code = int(codes[token, head, channel].item())
                bit_offset = channel * 3
                word = bit_offset // 32
                shift = bit_offset % 32
                packed[head, word, token] |= (code << shift) & mask32
                if shift + 3 > 32:
                    packed[head, word + 1, token] |= code >> (32 - shift)
    packed[packed >= (1 << 31)] -= 1 << 32
    return packed.to(torch.int32).contiguous()


def _validate_dense_and_states(
    inputs: Mapping[str, Any],
    dense: Mapping[str, Any],
    metadata: Mapping[str, Any],
    sparse_values: Mapping[str, Any],
    sparse_indices: Mapping[str, Any],
    sink: Mapping[str, Any],
    store_state: Mapping[str, Any],
    append_state: Mapping[str, Any],
) -> None:
    import torch

    if set(dense) != EXPECTED_TENSOR_NAMES["dense_payload.safetensors"]:
        raise CorrectedBundleValidationError("kvq3 dense tensor set mismatch")
    expected_state_names = EXPECTED_TENSOR_NAMES["store_state.safetensors"]
    if set(store_state) != expected_state_names or set(append_state) != expected_state_names:
        raise CorrectedBundleValidationError("kvq3 cache-state tensor set mismatch")
    for state, length in ((store_state, 17), (append_state, 18)):
        if (
            state["k_dense_allocated"].shape != (8, 12, 18)
            or state["v_dense_allocated"].shape != (8, 12, 18)
            or state["k_dense_allocated"].dtype != torch.int32
            or state["v_dense_allocated"].dtype != torch.int32
            or state["k_length"].dtype != torch.int32
            or state["v_length"].dtype != torch.int32
            or state["k_length"].tolist() != [length]
            or state["v_length"].tolist() != [length]
            or state["v_lookup_allocated"].shape != (18, 8)
            or state["v_lookup_allocated"].dtype != torch.float32
        ):
            raise CorrectedBundleValidationError("kvq3 cache-state layout mismatch")
    scalar = _scalar_value_pack(inputs, metadata)
    exact_store = scalar[:, :, :12]
    exact_append = scalar
    store_candidates = (
        dense["value_scalar_control_after_store"],
        dense["value_expected_nearest_after_store"],
        dense["value_packed_after_store"],
        store_state["v_dense_allocated"][:, :, :12],
    )
    append_candidates = (
        dense["value_scalar_control_after_append"],
        dense["value_expected_nearest_after_append"],
        dense["value_packed_after_append"],
        dense["value_packed_independent_control"],
        append_state["v_dense_allocated"][:, :, :13],
    )
    if any(not torch.equal(value, exact_store) for value in store_candidates):
        raise CorrectedBundleValidationError(
            "kvq3 store does not exactly match scalar code/pack control"
        )
    if any(not torch.equal(value, exact_append) for value in append_candidates):
        raise CorrectedBundleValidationError(
            "kvq3 append does not exactly match scalar code/pack control"
        )
    if (
        not torch.equal(
            dense["key_packed_after_store"],
            store_state["k_dense_allocated"][:, :, :12],
        )
        or not torch.equal(
            dense["key_packed_after_append"],
            append_state["k_dense_allocated"][:, :, :13],
        )
        or not torch.equal(
            dense["key_packed_independent_control"],
            dense["key_packed_after_append"],
        )
        or not torch.equal(
            dense["key_appended_slot"],
            dense["key_packed_after_append"][:, :, -1],
        )
        or not torch.equal(
            dense["value_appended_slot"],
            dense["value_packed_after_append"][:, :, -1],
        )
        or not torch.equal(
            store_state["k_dense_allocated"][:, :, :12],
            append_state["k_dense_allocated"][:, :, :12],
        )
        or not torch.equal(
            store_state["v_dense_allocated"][:, :, :12],
            append_state["v_dense_allocated"][:, :, :12],
        )
    ):
        raise CorrectedBundleValidationError("kvq3 dense store/append mismatch")
    if (
        not torch.equal(
            metadata["value_lookup_after_store"],
            store_state["v_lookup_allocated"],
        )
        or not torch.equal(
            metadata["value_lookup_after_append"],
            append_state["v_lookup_allocated"],
        )
        or not torch.equal(sink["sink_key_attention_fp16"], store_state["sink_k"])
        or not torch.equal(sink["sink_key_attention_fp16"], append_state["sink_k"])
        or not torch.equal(sink["sink_value_fp16"], store_state["sink_v"])
        or not torch.equal(sink["sink_value_fp16"], append_state["sink_v"])
    ):
        raise CorrectedBundleValidationError(
            "kvq3 metadata/sink cache-state mismatch"
        )
    cross_file = (
        (
            sparse_values["key_cache_after_store"],
            store_state["k_sparse_values_allocated"],
        ),
        (
            sparse_values["key_cache_after_append"],
            append_state["k_sparse_values_allocated"],
        ),
        (
            sparse_values["value_cache_after_store"],
            store_state["v_sparse_values_allocated"],
        ),
        (
            sparse_values["value_cache_after_append"],
            append_state["v_sparse_values_allocated"],
        ),
        (
            sparse_indices["key_cache_after_store"],
            store_state["k_sparse_indices_allocated"],
        ),
        (
            sparse_indices["key_cache_after_append"],
            append_state["k_sparse_indices_allocated"],
        ),
        (
            sparse_indices["value_cache_after_store"],
            store_state["v_sparse_indices_allocated"],
        ),
        (
            sparse_indices["value_cache_after_append"],
            append_state["v_sparse_indices_allocated"],
        ),
    )
    if any(not torch.equal(left, right) for left, right in cross_file):
        raise CorrectedBundleValidationError(
            "kvq3 sparse cache-state projection mismatch"
        )


def _validate_byte_breakdown(
    payload: Mapping[str, Any],
    case_name: str,
    key_count: int,
) -> None:
    if (
        payload.get("schema_version")
        != "kvbench-phase10-byte-breakdown-1.0.0"
        or payload.get("fixture_id") != FIXTURE_ID
        or payload.get("family") != "kvq3"
        or payload.get("case") != case_name
        or payload.get("bit_width") != 3
        or payload.get("allocation_basis") != "source_owned_tensor_storage"
        or payload.get("r_hbm") is not None
    ):
        raise CorrectedBundleValidationError(
            "kvq3 byte-breakdown identity mismatch"
        )
    expected = legacy._expected_byte_breakdown(3, key_count)
    if any(payload.get(name) != value for name, value in expected.items()):
        raise CorrectedBundleValidationError(
            "kvq3 byte-breakdown formula mismatch"
        )
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
    logical = expected["logical_bf16_bytes"]
    if (
        payload.get("actual_allocated_total_bytes") != allocated
        or not math.isclose(
            payload.get("rho_alloc"), allocated / logical, rel_tol=0.0
        )
        or not math.isclose(
            payload.get("r_alloc"), logical / allocated, rel_tol=0.0
        )
        or abs(payload["rho_alloc"] * payload["r_alloc"] - 1.0)
        > legacy.RECIPROCAL_ATOL
        or payload.get("reciprocal_tolerance") != legacy.RECIPROCAL_ATOL
        or payload.get("fixed_capacity")
        != {
            "key_slots_per_physical_row": 12,
            "value_slots_per_physical_row": 12,
            "key_active_entries": key_count * 13,
            "value_active_entries_non_sink": 12 * 13,
            "value_active_entries_sink": 0,
        }
    ):
        raise CorrectedBundleValidationError(
            "kvq3 byte-breakdown accounting mismatch"
        )


def _validate_kvq3_fixture(
    root: Path,
    case_name: str,
    key_count: int,
    authority_hashes: Mapping[str, str],
) -> dict[str, Any]:
    import torch

    try:
        legacy._validate_fixture_ledger(root)
    except legacy.FixtureValidationError as error:
        raise CorrectedBundleValidationError(str(error)) from error
    manifest = _load_json(root / "fixture_manifest.json")
    _validate_kvq3_manifest(
        manifest,
        case_name,
        key_count,
        authority_hashes,
    )
    tensor_files = {
        filename: _read_safetensors(root / filename, FIXTURE_ID)
        for filename in TENSOR_MEMBERS
    }
    if set(tensor_files) != set(EXPECTED_TENSOR_NAMES):
        raise CorrectedBundleValidationError("kvq3 tensor file set mismatch")
    for filename, expected_names in EXPECTED_TENSOR_NAMES.items():
        if set(tensor_files[filename]) != expected_names:
            raise CorrectedBundleValidationError(
                f"kvq3 tensor name set mismatch: {filename}"
            )
    _validate_tensor_records(manifest.get("tensor_records"), tensor_files)
    byte_breakdown = _load_json(root / "byte_breakdown.json")
    if manifest.get("byte_breakdown_sha256") != hashlib.sha256(
        _canonical_json(byte_breakdown)
    ).hexdigest():
        raise CorrectedBundleValidationError(
            "kvq3 byte-breakdown manifest hash mismatch"
        )
    _validate_byte_breakdown(byte_breakdown, case_name, key_count)

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
        set(inputs) != EXPECTED_TENSOR_NAMES["inputs.safetensors"]
        or inputs["key_pre_rope"].shape != (1, 8, 18, 128)
        or inputs["value_after_v_proj"].shape != (1, 8, 18, 128)
        or inputs["query_pre_rope"].shape != (1, 32, 1, 128)
        or inputs["key_pre_rope"].dtype != torch.bfloat16
        or inputs["value_after_v_proj"].dtype != torch.bfloat16
        or inputs["query_pre_rope"].dtype != torch.bfloat16
        or inputs["position_ids"].dtype != torch.int64
        or inputs["position_ids"].tolist() != [list(range(18))]
    ):
        raise CorrectedBundleValidationError("kvq3 input tensor mismatch")
    _validate_metadata(metadata)
    try:
        legacy._validate_sparse(
            inputs,
            metadata,
            sparse_values,
            sparse_indices,
            3,
            key_count,
        )
        legacy._validate_sink_and_rope(
            inputs,
            metadata,
            sink,
            store_state,
            append_state,
        )
        legacy._validate_decode(decode)
    except legacy.FixtureValidationError as error:
        raise CorrectedBundleValidationError(str(error)) from error
    _validate_dense_and_states(
        inputs,
        dense,
        metadata,
        sparse_values,
        sparse_indices,
        sink,
        store_state,
        append_state,
    )
    return {
        "family": "kvq3",
        "case": case_name,
        "key_active_count": key_count,
        "value_active_count_non_sink": 12,
        "value_active_count_sink": 0,
        "scalar_store": "PASS",
        "scalar_append": "PASS",
        "decode_sha256": hashlib.sha256(
            _tensor_bytes(decode["source_decode_output"])
        ).hexdigest(),
        "allocated_bytes": byte_breakdown["actual_allocated_total_bytes"],
    }


def validate_corrected_bundle(
    fixture_root: Path,
    old_fixture_root: Path,
) -> dict[str, Any]:
    fixture_root = fixture_root.resolve(strict=True)
    old_fixture_root = old_fixture_root.resolve(strict=True)
    file_count = _validate_exact_layout(fixture_root)
    try:
        artifact = validate_local_artifact(fixture_root)
        old_artifact = validate_local_artifact(old_fixture_root)
    except ArtifactValidationError as error:
        raise CorrectedBundleValidationError(str(error)) from error
    if old_artifact.root_sha256 != OLD_ROOT_SHA256:
        raise CorrectedBundleValidationError(
            "old Phase 10 fixture root digest mismatch"
        )
    reuse = {
        family: _validate_reused_family(
            fixture_root,
            old_fixture_root,
            family,
        )
        for family in ("kvq4", "kvq2")
    }
    authority_hashes = _validate_authority(fixture_root)
    _validate_root_manifest(fixture_root, authority_hashes, reuse)
    kvq3 = [
        _validate_kvq3_fixture(
            fixture_root / "kvq3" / case_name,
            case_name,
            key_count,
            authority_hashes,
        )
        for case_name, key_count in CASES.items()
    ]
    if file_count != len(artifact.files):
        raise CorrectedBundleValidationError(
            "artifact lifecycle file count mismatch"
        )
    return {
        "status": "PASS",
        "fixture_id": FIXTURE_ID,
        "local_root_sha256": artifact.root_sha256,
        "object_count": len(artifact.files),
        "fixture_count": 9,
        "old_root_sha256": old_artifact.root_sha256,
        "reused_families": reuse,
        "regenerated_family": "kvq3",
        "kvq3": kvq3,
        "scalar_control": "PASS",
        "artifact_lifecycle": "PASS",
        "unexpected_files": False,
        "calibration_changed": False,
        "g2_kvq": "NOT_EVALUATED",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=DEFAULT_FIXTURE_ROOT,
        help="finalized corrected fixture root",
    )
    parser.add_argument(
        "--old-fixtures",
        type=Path,
        default=DEFAULT_OLD_FIXTURE_ROOT,
        help="immutable Phase 10 fixture root",
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        result = validate_corrected_bundle(
            arguments.fixtures,
            arguments.old_fixtures,
        )
    except (
        ArtifactValidationError,
        CorrectedBundleValidationError,
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
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
