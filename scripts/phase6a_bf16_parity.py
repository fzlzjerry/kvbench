#!/usr/bin/env python3
"""Exactly two untimed Phase 6A BF16 measurement-container parity lanes."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
from typing import Any

from kvbench.config import REPOSITORY_ROOT
from kvbench.errors import SchemaValidationError
from kvbench.schema import canonical_json_bytes, sha256_hex
from kvbench.schema.phase3 import (
    PHASE3_BF16_VARIANT_FINGERPRINT,
    PHASE3_GPU_UUID,
)
from preflight.run_preflight import (
    enumerate_evidence_files,
    finalize_stage,
    json_bytes,
    write_exclusive,
)
from scripts.r2_artifact import (
    sha256_file,
    validate_local_artifact,
)

SCHEMA_VERSION = "kvbench-phase6a-bf16-container-parity-1.0.0"
INDEX_SCHEMA_VERSION = "kvbench-phase6a-bf16-container-parity-index-1.0.0"
OUTPUT_ROOT = REPOSITORY_ROOT / "artifacts/phase6a/bf16_parity"
EXPECTED_GPU_NAME = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
EXPECTED_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
EXPECTED_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
LANES = ("eager", "cuda_graph")
_IMAGE_DIGEST = re.compile(r"\Asha256:[0-9a-f]{64}\Z")
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_GIT_SHA = re.compile(r"\A[0-9a-f]{40}\Z")
_RUN_ID = re.compile(r"\A[a-z0-9][a-z0-9._-]{0,127}\Z")
_E00_RUN_ID = re.compile(
    r"\Ae00-[0-9]{8}T[0-9]{6}\.[0-9]{6}Z-[0-9a-f]{12}-[0-9a-f]{8}\Z"
)
_UTC_TIMESTAMP = re.compile(
    r"\A[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z\Z"
)
_LEGACY_PHASE3_PROVENANCE_KEYS = frozenset(
    {
        "schema_version",
        "receipt_sha256",
        "cache_pointers",
        "cache_layout_fingerprint",
        "operation_fingerprints",
        "dispatch_audit_sha256",
        "allocation_audit_sha256",
        "audit_output_sha256",
        "audit_output_finite",
        "graph_retained",
        "prefix_sha256",
        "history_chain_sha256",
    }
)


class Phase6AParityError(RuntimeError):
    """A parity precondition, execution control, or publication failed."""


@dataclass(frozen=True, slots=True)
class PublishedLane:
    """One immutable terminal parity artifact."""

    run_id: str
    graph_mode: str
    status: str
    directory: Path
    root_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "graph_mode": self.graph_mode,
            "status": self.status,
            "directory": self.directory.as_posix(),
            "root_sha256": self.root_sha256,
        }


LaneExecutor = Callable[[str, str, Path], Mapping[str, Any]]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(value)

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise Phase6AParityError(f"{label} is invalid") from error
    if not isinstance(value, dict):
        raise Phase6AParityError(f"{label} is invalid")
    return value


def _image_digest(value: object) -> str:
    if type(value) is not str or _IMAGE_DIGEST.fullmatch(value) is None:
        raise Phase6AParityError("image config digest is invalid")
    return value


def _git_sha(value: object) -> str:
    if type(value) is not str or _GIT_SHA.fullmatch(value) is None:
        raise Phase6AParityError("execution Git SHA is invalid")
    return value


def _sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise Phase6AParityError(f"{label} SHA-256 is invalid")
    return value


def _run_identifier(value: object) -> str:
    if type(value) is not str or _RUN_ID.fullmatch(value) is None:
        raise Phase6AParityError("parity run ID is invalid")
    return value


def _failure(error: Exception) -> dict[str, str]:
    reason = str(error)
    for name in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "CLOUDFLARE_API_TOKEN",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "R2_ACCOUNT_ID",
    ):
        value = os.environ.get(name)
        if value:
            reason = reason.replace(value, f"<redacted:{name}>")
    reason = "".join(
        character if 32 <= ord(character) < 127 else " "
        for character in reason
    )
    return {
        "error_type": type(error).__name__[:128],
        "reason": reason[:1024] or "parity_lane_failed",
    }


def _phase3_provenance_for_legacy_replay(
    provenance_raw: bytes,
    *,
    expected_adapter_config_fingerprint: str,
) -> tuple[dict[str, Any], bytes]:
    """Validate extended adapter provenance and return the legacy projection."""

    expected_fingerprint = _sha256(
        expected_adapter_config_fingerprint,
        "expected adapter runtime fingerprint",
    )

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        for key, value in pairs:
            if key in parsed:
                raise Phase6AParityError(
                    "raw-audit provenance has a duplicate key"
                )
            parsed[key] = value
        return parsed

    def reject_constant(value: str) -> None:
        raise ValueError(value)

    try:
        provenance = json.loads(
            provenance_raw,
            object_pairs_hook=unique,
            parse_constant=reject_constant,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise Phase6AParityError(
            "raw-audit provenance is invalid"
        ) from error
    extended_keys = _LEGACY_PHASE3_PROVENANCE_KEYS | {
        "method_name",
        "adapter_version",
        "adapter_config_fingerprint",
    }
    if (
        not isinstance(provenance, dict)
        or set(provenance) != extended_keys
        or canonical_json_bytes(provenance) != provenance_raw
        or provenance.get("method_name") != "bf16"
        or provenance.get("adapter_version")
        != "kvbench-bf16-method-adapter-1.0.0"
        or provenance.get("adapter_config_fingerprint")
        != expected_fingerprint
    ):
        raise Phase6AParityError(
            "raw-audit adapter provenance differs"
        )
    projection = {
        key: provenance[key]
        for key in sorted(_LEGACY_PHASE3_PROVENANCE_KEYS)
    }
    return provenance, canonical_json_bytes(projection)


def validate_container_g0_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_image_config_digest: str,
) -> dict[str, str]:
    """Require an exact completed PASS container G0."""

    digest = _image_digest(expected_image_config_digest)
    if manifest.get("schema_version") != "e00-manifest-1.1.0":
        raise Phase6AParityError("container G0 schema is not authoritative")
    run, gate = manifest.get("run"), manifest.get("gate")
    environment = manifest.get("execution_environment")
    gpu = manifest.get("gpu")
    if not all(
        isinstance(item, Mapping) for item in (run, gate, environment, gpu)
    ):
        raise Phase6AParityError("container G0 lacks required sections")
    assert isinstance(run, Mapping) and isinstance(gate, Mapping)
    assert isinstance(environment, Mapping) and isinstance(gpu, Mapping)
    checks = gate.get("checks")
    if (
        run.get("gate"),
        run.get("status"),
        run.get("completed"),
        run.get("benchmark_timing_collected"),
        gate.get("aggregate_status"),
    ) != ("G0", "PASS", True, False, "PASS"):
        raise Phase6AParityError("container G0 did not complete with PASS")
    if (
        not isinstance(checks, list)
        or not checks
        or any(
            not isinstance(item, Mapping) or item.get("status") != "PASS"
            for item in checks
        )
    ):
        raise Phase6AParityError("container G0 contains a non-passing check")
    container = environment.get("container")
    if (
        environment.get("kind") != "measurement_container"
        or environment.get("verification_status") != "PASS"
        or environment.get("performance_claim_eligible") is not False
        or not isinstance(container, Mapping)
        or container.get("runtime") != "docker"
        or container.get("image_config_digest") != digest
        or container.get("digest_status")
        != "verified_against_sanitized_image_inspect"
    ):
        raise Phase6AParityError("container G0 image/environment differs")
    capability, uuid = gpu.get("compute_capability"), gpu.get("uuid")
    if (
        gpu.get("collection_status") != "PASS"
        or gpu.get("full_name") != EXPECTED_GPU_NAME
        or uuid != PHASE3_GPU_UUID
        or not isinstance(capability, Mapping)
        or (
            capability.get("major"),
            capability.get("minor"),
            capability.get("text"),
        )
        != (12, 0, "12.0")
    ):
        raise Phase6AParityError("container G0 GPU identity differs")
    run_id = run.get("id")
    if type(run_id) is not str or not run_id:
        raise Phase6AParityError("container G0 run ID is invalid")
    return {
        "run_id": run_id,
        "gpu_uuid": PHASE3_GPU_UUID,
        "gpu_name": EXPECTED_GPU_NAME,
        "compute_capability": "12.0",
        "image_config_digest": digest,
    }


def load_container_g0_artifact(
    path: Path,
    *,
    expected_image_config_digest: str,
) -> dict[str, str]:
    """Validate a finalized local G0 artifact and return its safe identity."""

    artifact = validate_local_artifact(path)
    manifest_path = artifact.directory / "manifest.json"
    identity = validate_container_g0_manifest(
        _load_json(manifest_path, "container G0 manifest"),
        expected_image_config_digest=expected_image_config_digest,
    )
    return {
        **identity,
        "artifact_root_sha256": artifact.root_sha256,
        "manifest_sha256": sha256_file(manifest_path),
        "artifact_directory": artifact.directory.as_posix(),
    }


def _closed_mapping(
    value: object,
    fields: set[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise Phase6AParityError(f"{label} field closure differs")
    return value


def _validate_comparison(value: object, label: str) -> None:
    value = _closed_mapping(
        value,
        {
            "passed",
            "finite",
            "max_absolute_error",
            "max_relative_error",
            "atol",
            "rtol",
        },
        f"{label} comparison",
    )
    if value.get("passed") is not True or value.get("finite") is not True:
        raise Phase6AParityError(f"{label} comparison did not pass")
    for name in ("max_absolute_error", "max_relative_error", "atol", "rtol"):
        observed = value.get(name)
        if (
            type(observed) is not float
            or not math.isfinite(float(observed))
            or not float(observed) >= 0.0
        ):
            raise Phase6AParityError(f"{label} comparison is invalid")


def _validate_full_model_numerical(value: object) -> str:
    value = _closed_mapping(
        value,
        {
            "passed",
            "reference_implementation",
            "reference_cache_type",
            "reference_implementation_restored",
            "tolerance_atol",
            "tolerance_rtol",
            "fixed_repeat_exact",
            "fixed_historical_cache_unchanged",
            "fixed_steps",
            "growing_steps",
            "timing_collected",
            "performance_claim_eligible",
        },
        "full-model numerical control",
    )
    if (
        value.get("passed") is not True
        or value.get("reference_implementation")
        != "transformers_eager_dynamic_cache"
        or value.get("reference_cache_type") != "DynamicCache"
        or value.get("reference_implementation_restored") is not True
        or value.get("fixed_repeat_exact") is not True
        or value.get("fixed_historical_cache_unchanged") is not True
        or value.get("timing_collected") is not False
        or value.get("performance_claim_eligible") is not False
    ):
        raise Phase6AParityError("full-model numerical control did not pass")
    if (
        value.get("tolerance_atol") != 0.125
        or value.get("tolerance_rtol") != 0.02
    ):
        raise Phase6AParityError("full-model numerical tolerance differs")
    fixed, growing = value.get("fixed_steps"), value.get("growing_steps")
    if (
        not isinstance(fixed, list)
        or len(fixed) != 3
        or not isinstance(growing, list)
        or len(growing) != 3
    ):
        raise Phase6AParityError("full-model numerical steps differ")
    for collection, mode in ((fixed, "fixed_l"), (growing, "growing_context")):
        for index, step in enumerate(collection):
            expected_position = 128 if mode == "fixed_l" else 128 + index
            step = _closed_mapping(
                step,
                {
                    "mode",
                    "step",
                    "position",
                    "reference_checksum",
                    "observed_checksum",
                    "comparison",
                },
                "full-model numerical step",
            )
            if (
                type(step.get("step")) is not int
                or type(step.get("position")) is not int
                or (
                    step.get("mode"),
                    step.get("step"),
                    step.get("position"),
                )
                != (mode, index, expected_position)
            ):
                raise Phase6AParityError("full-model numerical step differs")
            _sha256(step.get("reference_checksum"), "reference output")
            _sha256(step.get("observed_checksum"), "observed output")
            _validate_comparison(step.get("comparison"), "full-model")
            comparison = step["comparison"]
            assert isinstance(comparison, Mapping)
            if (
                comparison.get("atol") != 0.125
                or comparison.get("rtol") != 0.02
            ):
                raise Phase6AParityError(
                    "full-model numerical step tolerance differs"
                )
    observed = _sha256(fixed[0].get("observed_checksum"), "L=128 output")
    if any(step.get("observed_checksum") != observed for step in fixed):
        raise Phase6AParityError("fixed-L repeat output differs")
    return observed


def _validate_graph_result(value: object, label: str) -> tuple[str, str]:
    value = _closed_mapping(
        value,
        {
            "passed",
            "prefix_length",
            "graph",
            "eager_replay_comparison",
            "replay_outputs_exact",
            "replay_copies_independent",
            "eager_checksum",
            "first_replay_checksum",
            "second_replay_checksum",
            "cache_pointers_stable",
            "historical_cache_unchanged",
            "replay_allocation",
            "timing_collected",
            "performance_claim_eligible",
        },
        f"{label} graph evidence",
    )
    graph = _closed_mapping(
        value.get("graph"),
        {
            "captured",
            "output_data_ptr",
            "capture_stream_id",
            "fallback",
        },
        f"{label} captured graph",
    )
    comparison = value.get("eager_replay_comparison")
    allocation = _closed_mapping(
        value.get("replay_allocation"),
        {
            "audit_available",
            "passed",
            "allocation_event_count",
            "allocation_event_bytes",
            "event_counts",
            "allocated_before",
            "allocated_after",
            "allocated_delta",
            "reserved_before",
            "reserved_after",
            "reserved_delta",
            "peak_allocated",
            "peak_reserved",
            "failure_reason",
            "instrumented_duration_reported_as_timing",
        },
        f"{label} replay allocation",
    )
    integer_fields = (
        "allocation_event_count",
        "allocation_event_bytes",
        "allocated_before",
        "allocated_after",
        "allocated_delta",
        "reserved_before",
        "reserved_after",
        "reserved_delta",
        "peak_allocated",
        "peak_reserved",
    )
    if (
        value.get("passed") is not True
        or type(value.get("prefix_length")) is not int
        or value.get("prefix_length") != 128
        or graph.get("captured") is not True
        or graph.get("fallback") is not False
        or type(graph.get("output_data_ptr")) is not int
        or graph.get("output_data_ptr") <= 0
        or type(graph.get("capture_stream_id")) is not int
        or graph.get("capture_stream_id") <= 0
        or value.get("replay_outputs_exact") is not True
        or value.get("replay_copies_independent") is not True
        or value.get("cache_pointers_stable") is not True
        or value.get("historical_cache_unchanged") is not True
        or value.get("timing_collected") is not False
        or value.get("performance_claim_eligible") is not False
        or allocation.get("audit_available") is not True
        or allocation.get("passed") is not True
        or any(type(allocation.get(name)) is not int for name in integer_fields)
        or allocation.get("allocation_event_count") != 0
        or allocation.get("allocation_event_bytes") != 0
        or allocation.get("event_counts") != {}
        or allocation.get("allocated_before") < 0
        or allocation.get("allocated_after") < 0
        or allocation.get("allocated_delta")
        != allocation.get("allocated_after") - allocation.get("allocated_before")
        or allocation.get("allocated_delta") > 0
        or allocation.get("reserved_before") < 0
        or allocation.get("reserved_after") < 0
        or allocation.get("reserved_delta")
        != allocation.get("reserved_after") - allocation.get("reserved_before")
        or allocation.get("reserved_delta") > 0
        or allocation.get("peak_allocated") < 0
        or allocation.get("peak_reserved") < 0
        or allocation.get("failure_reason") is not None
        or allocation.get("instrumented_duration_reported_as_timing")
        is not False
    ):
        raise Phase6AParityError(f"{label} graph control did not pass")
    _validate_comparison(comparison, f"{label} eager/replay")
    assert isinstance(comparison, Mapping)
    if comparison.get("atol") != 0.02 or comparison.get("rtol") != 0.02:
        raise Phase6AParityError(f"{label} graph tolerance differs")
    eager = _sha256(value.get("eager_checksum"), f"{label} eager")
    first = _sha256(
        value.get("first_replay_checksum"),
        f"{label} first replay",
    )
    second = _sha256(
        value.get("second_replay_checksum"),
        f"{label} second replay",
    )
    if first != second:
        raise Phase6AParityError(f"{label} replay output differs")
    return eager, first


def _validate_numerical(graph_mode: str, value: object) -> tuple[str, str | None]:
    value = _closed_mapping(
        value,
        {
            "passed",
            "small_tensor",
            "full_model",
            "full_model_graph",
            "timing_collected",
        },
        "numerical controls",
    )
    if (
        value.get("passed") is not True
        or value.get("timing_collected") is not False
    ):
        raise Phase6AParityError("numerical controls did not pass")
    small = _closed_mapping(
        value.get("small_tensor"),
        {
            "passed",
            "reference",
            "atol",
            "rtol",
            "records",
            "timing_collected",
        },
        "small-tensor numerical control",
    )
    if (
        small.get("passed") is not True
        or small.get("reference") != "explicit_fp32_gqa_attention"
        or small.get("atol") != 0.02
        or small.get("rtol") != 0.02
        or small.get("timing_collected") is not False
        or not isinstance(small.get("records"), list)
        or len(small["records"]) != 12
    ):
        raise Phase6AParityError("small-tensor numerical control differs")
    expected_records = [
        (batch, length, mode)
        for batch in (1, 2)
        for length in (7, 17)
        for mode in ("causal_gqa", "decode_gqa", "causal_mha")
    ]
    for record, expected in zip(small["records"], expected_records, strict=True):
        record = _closed_mapping(
            record,
            {
                "batch_size",
                "context_length",
                "mode",
                "boundary_first_finite",
                "boundary_last_finite",
                "comparison",
            },
            "small-tensor numerical record",
        )
        if (
            type(record.get("batch_size")) is not int
            or type(record.get("context_length")) is not int
            or (
                record.get("batch_size"),
                record.get("context_length"),
                record.get("mode"),
            )
            != expected
            or record.get("boundary_first_finite") is not True
            or record.get("boundary_last_finite") is not True
        ):
            raise Phase6AParityError("small-tensor numerical record differs")
        _validate_comparison(record.get("comparison"), "small-tensor")
        comparison = record["comparison"]
        assert isinstance(comparison, Mapping)
        if (
            comparison.get("atol") != 0.02
            or comparison.get("rtol") != 0.02
        ):
            raise Phase6AParityError(
                "small-tensor numerical tolerance differs"
            )
    observed = _validate_full_model_numerical(value.get("full_model"))
    graph = value.get("full_model_graph")
    if graph_mode == "eager":
        if graph is not None:
            raise Phase6AParityError("eager lane contains graph numerical data")
        return observed, None
    graph_eager, graph_replay = _validate_graph_result(
        graph,
        "numerical",
    )
    if graph_eager != observed:
        raise Phase6AParityError("graph numerical eager checksum differs")
    return observed, graph_replay


def validate_parity_result(
    graph_mode: str,
    result: Mapping[str, Any],
) -> None:
    """Fail closed on the exact preregistered parity result projection."""

    if graph_mode not in LANES or not isinstance(result, Mapping):
        raise Phase6AParityError("graph mode is outside the two-lane plan")
    common_fields = {
        "passed",
        "runner",
        "graph_mode",
        "batch_size",
        "context_length",
        "output_steps",
        "backend_fallback",
        "adapter_config_fingerprint",
        "cache_layout_fingerprint",
        "allocated_cache_bytes",
        "logical_bf16_bytes",
        "byte_breakdown",
        "byte_breakdown_sums_to_allocated",
        "numerical",
        "cache_geometry",
        "raw_audit_semantics",
        "output_checksum_join",
        "eager_allocation_criterion_passed",
    }
    mode_fields = (
        {
            "output_finite",
            "output_sha256",
            "cache_pointers_stable",
            "historical_cache_unchanged",
        }
        if graph_mode == "eager"
        else {"graph_validation"}
    )
    if set(result) != common_fields | mode_fields:
        raise Phase6AParityError("parity result field closure differs")
    numerical = result.get("numerical")
    trusted_observed, numerical_graph_replay = _validate_numerical(
        graph_mode,
        numerical,
    )
    breakdown = _closed_mapping(
        result.get("byte_breakdown"),
        {
            "data_bytes",
            "workspace_bytes",
            "padding_bytes",
            "scale_bytes",
            "zero_point_bytes",
            "metadata_bytes",
        },
        "BF16 byte breakdown",
    )
    expected_breakdown = {
        "data_bytes": 16_908_288,
        "workspace_bytes": 163_840,
        "padding_bytes": 0,
        "scale_bytes": 0,
        "zero_point_bytes": 0,
        "metadata_bytes": 0,
    }
    geometry = _closed_mapping(
        result.get("cache_geometry"),
        {
            "cache_shape",
            "num_query_heads",
            "num_kv_heads",
            "uses_kv_head_geometry",
            "measured_storage_bytes",
            "predicted_kv_head_bytes",
            "forbidden_query_head_bytes",
            "query_head_storage_detected",
        },
        "BF16 cache geometry",
    )
    semantic = _closed_mapping(
        result.get("raw_audit_semantics"),
        {
            "semantic_validation_passed",
            "scientific_completion_passed",
            "transport_terminal_eligible",
            "semantic_operations",
            "raw_audit_index_sha256",
            "adapter_runtime_fingerprint",
            "legacy_replay_projection_applied",
            "source_revalidated_after_execution",
        },
        "raw-audit semantics",
    )
    checksum_fields = {
        "passed",
        "trusted_reference_observed_sha256",
        "raw_audit_output_sha256",
    } | (
        {"scenario_output_sha256"}
        if graph_mode == "eager"
        else {
            "scenario_eager_sha256",
            "scenario_first_replay_sha256",
            "scenario_second_replay_sha256",
        }
    )
    checksum_join = _closed_mapping(
        result.get("output_checksum_join"),
        checksum_fields,
        "output checksum join",
    )
    top_integer_fields = (
        "batch_size",
        "context_length",
        "output_steps",
        "allocated_cache_bytes",
        "logical_bf16_bytes",
    )
    geometry_integer_fields = (
        "num_query_heads",
        "num_kv_heads",
        "measured_storage_bytes",
        "predicted_kv_head_bytes",
        "forbidden_query_head_bytes",
    )
    if (
        result.get("passed") is not True
        or result.get("runner") != "fixed_l"
        or result.get("graph_mode") != graph_mode
        or any(type(result.get(name)) is not int for name in top_integer_fields)
        or (
            result.get("batch_size"),
            result.get("context_length"),
            result.get("output_steps"),
        )
        != (1, 128, 1)
        or result.get("backend_fallback") is not False
        or result.get("allocated_cache_bytes") != 17_072_128
        or result.get("logical_bf16_bytes") != 16_908_288
        or result.get("byte_breakdown_sums_to_allocated") is not True
        or any(type(breakdown.get(name)) is not int for name in breakdown)
        or dict(breakdown) != expected_breakdown
        or sum(breakdown.values()) != result.get("allocated_cache_bytes")
        or type(geometry.get("cache_shape")) is not list
        or len(geometry.get("cache_shape")) != 5
        or any(
            type(item) is not int for item in geometry.get("cache_shape")
        )
        or geometry.get("cache_shape") != [32, 1, 8, 129, 128]
        or any(
            type(geometry.get(name)) is not int
            for name in geometry_integer_fields
        )
        or (
            geometry.get("num_query_heads"),
            geometry.get("num_kv_heads"),
        )
        != (32, 8)
        or geometry.get("uses_kv_head_geometry") is not True
        or geometry.get("measured_storage_bytes") != 16_908_288
        or geometry.get("predicted_kv_head_bytes") != 16_908_288
        or geometry.get("forbidden_query_head_bytes") != 67_633_152
        or geometry.get("query_head_storage_detected") is not False
        or semantic.get("semantic_validation_passed") is not True
        or semantic.get("scientific_completion_passed") is not True
        or semantic.get("transport_terminal_eligible") is not False
        or semantic.get("source_revalidated_after_execution") is not True
        or semantic.get("legacy_replay_projection_applied") is not True
        or checksum_join.get("passed") is not True
    ):
        raise Phase6AParityError(
            "parity result violates fixed-lane requirements"
        )
    operations = semantic.get("semantic_operations")
    if not isinstance(operations, list) or len(operations) != 1:
        raise Phase6AParityError("parity raw-audit operation set differs")
    operation = _closed_mapping(
        operations[0],
        {
            "operation_fingerprint_sha256",
            "dispatch_audit_sha256",
            "allocation_audit_sha256",
            "gqa_verdict",
            "gqa_reasons",
            "device_kernel_families",
            "allocation_criterion_id",
            "allocation_event_count",
            "allocation_class_counts",
            "allocation_failure_reasons",
            "allocation_join_sha256",
            "paired_allocator_control_sha256",
            "split_k_pair_multiplicity",
            "operation_output_sha256",
            "operation_output_finite",
        },
        "raw-audit semantic operation",
    )
    families = _closed_mapping(
        operation.get("device_kernel_families"),
        {"gqa", "mha_control"},
        "raw-audit kernel families",
    )
    paired = _closed_mapping(
        operation.get("paired_allocator_control_sha256"),
        {"gqa", "mha_control"},
        "paired allocator control",
    )
    family_values = {
        "pytorch_flash::flash_fwd_kernel",
        "pytorch_flash::flash_fwd_splitkv",
    }
    class_counts = operation.get("allocation_class_counts")
    multiplicities = operation.get("split_k_pair_multiplicity")
    adapter_fingerprint = _sha256(
        result.get("adapter_config_fingerprint"),
        "scenario adapter runtime fingerprint",
    )
    if (
        semantic.get("adapter_runtime_fingerprint")
        != adapter_fingerprint
    ):
        raise Phase6AParityError("raw and scenario adapter identities differ")
    _sha256(semantic.get("raw_audit_index_sha256"), "raw-audit index")
    _sha256(result.get("cache_layout_fingerprint"), "cache layout")
    for field in (
        "operation_fingerprint_sha256",
        "dispatch_audit_sha256",
        "allocation_audit_sha256",
        "allocation_join_sha256",
        "operation_output_sha256",
    ):
        _sha256(operation.get(field), field)
    for name in ("gqa", "mha_control"):
        _sha256(paired.get(name), f"{name} allocator control")
    if (
        not isinstance(class_counts, Mapping)
        or any(
            type(name) is not str
            or not name
            or type(count) is not int
            or count <= 0
            for name, count in class_counts.items()
        )
        or type(operation.get("allocation_event_count")) is not int
        or operation.get("allocation_event_count") < 0
        or sum(class_counts.values()) != operation.get("allocation_event_count")
        or not isinstance(multiplicities, list)
    ):
        raise Phase6AParityError("raw-audit allocation evidence is invalid")
    normalized_multiplicities: list[tuple[int, int]] = []
    for item in multiplicities:
        item = _closed_mapping(
            item,
            {"num_splits", "pair_count"},
            "split-K multiplicity",
        )
        if (
            type(item.get("num_splits")) is not int
            or type(item.get("pair_count")) is not int
            or item.get("num_splits") <= 0
            or item.get("pair_count") <= 0
        ):
            raise Phase6AParityError("split-K multiplicity is invalid")
        normalized_multiplicities.append(
            (item["num_splits"], item["pair_count"])
        )
    if normalized_multiplicities != sorted(set(normalized_multiplicities)):
        raise Phase6AParityError("split-K multiplicity is not canonical")
    criterion = (
        "phase3_eager_attributed_ephemeral_v1"
        if graph_mode == "eager"
        else "phase3_graph_zero_allocation_v1"
    )
    if (
        operation.get("gqa_verdict")
        != "gqa_nonmaterialization_verified"
        or operation.get("gqa_reasons") != []
        or any(
            type(families.get(name)) is not str
            or families.get(name) not in family_values
            for name in families
        )
        or operation.get("allocation_criterion_id") != criterion
        or operation.get("allocation_failure_reasons") != []
        or operation.get("operation_output_finite") is not True
    ):
        raise Phase6AParityError(
            "GQA, allocation, or output audit did not pass"
        )
    raw_output = _sha256(
        operation.get("operation_output_sha256"),
        "raw-audit output",
    )
    for field in checksum_fields - {"passed"}:
        _sha256(checksum_join.get(field), field)
    if (
        checksum_join.get("trusted_reference_observed_sha256")
        != trusted_observed
        or checksum_join.get("raw_audit_output_sha256") != raw_output
    ):
        raise Phase6AParityError("trusted/raw output checksum join differs")
    if graph_mode == "eager":
        if (
            operation.get("allocation_event_count") != 1066
            or class_counts
            != {
                "context_scaled_workspace": 64,
                "fixed_output": 1,
                "fixed_shared_activation": 937,
                "framework_bookkeeping": 64,
            }
            or normalized_multiplicities != [(2, 1)]
        ):
            raise Phase6AParityError(
                "eager allocation summary differs from the frozen lane"
            )
        scenario_output = _sha256(
            result.get("output_sha256"),
            "eager scenario output",
        )
        if (
            result.get("eager_allocation_criterion_passed") is not True
            or result.get("output_finite") is not True
            or result.get("cache_pointers_stable") is not True
            or result.get("historical_cache_unchanged") is not True
            or scenario_output != trusted_observed
            or scenario_output != raw_output
            or checksum_join.get("scenario_output_sha256")
            != scenario_output
        ):
            raise Phase6AParityError("eager allocation criterion did not pass")
        return
    graph = result.get("graph_validation")
    graph_eager, graph_replay = _validate_graph_result(graph, "scenario")
    if (
        graph_eager != trusted_observed
        or graph_replay != numerical_graph_replay
        or graph_replay != raw_output
        or checksum_join.get("scenario_eager_sha256") != graph_eager
        or checksum_join.get("scenario_first_replay_sha256")
        != graph_replay
        or checksum_join.get("scenario_second_replay_sha256")
        != graph_replay
        or result.get("eager_allocation_criterion_passed") is not None
        or operation.get("allocation_event_count") != 0
        or class_counts != {}
        or normalized_multiplicities != []
    ):
        raise Phase6AParityError(
            "CUDA Graph capture/replay parity did not pass"
        )


def _validate_context(context: Mapping[str, Any]) -> None:
    from kvbench.schema.phase3 import BF16BackendIdentity

    required = {
        "execution_git_sha",
        "image_reference",
        "image_config_digest",
        "container_g0",
        "runtime_gpu",
        "model_identity",
        "backend_identity",
        "method_identity",
        "source_identity",
    }
    if set(context) != required:
        raise Phase6AParityError("parity context has the wrong fields")
    _git_sha(context["execution_git_sha"])
    _image_digest(context["image_config_digest"])
    if (
        type(context["image_reference"]) is not str
        or not context["image_reference"]
        or len(context["image_reference"]) > 512
        or any(
            ord(character) < 32 or ord(character) == 127
            for character in context["image_reference"]
        )
    ):
        raise Phase6AParityError("image reference is invalid")
    g0 = _closed_mapping(
        context["container_g0"],
        {
            "run_id",
            "gpu_uuid",
            "gpu_name",
            "compute_capability",
            "image_config_digest",
            "artifact_root_sha256",
            "manifest_sha256",
            "artifact_directory",
        },
        "container G0 identity",
    )
    gpu = _closed_mapping(
        context["runtime_gpu"],
        {"uuid", "name", "compute_capability"},
        "runtime GPU identity",
    )
    model = _closed_mapping(
        context["model_identity"],
        {
            "model_id",
            "model_revision",
            "tokenizer_revision",
            "frozen_identity_sha256",
            "snapshot_file_ledger_sha256",
            "load_receipt_sha256",
            "tokenizer_runtime_sha256",
            "parameter_runtime_sha256",
            "num_query_heads",
            "num_kv_heads",
            "head_dim",
            "weight_dtype",
        },
        "model identity",
    )
    backend = _closed_mapping(
        context["backend_identity"],
        {
            "backend",
            "fingerprint",
            "attention_implementation",
            "fallback_permitted",
        },
        "backend identity",
    )
    method = _closed_mapping(
        context["method_identity"],
        {
            "method",
            "method_config_id",
            "method_config_fingerprint",
            "adapter_version",
            "adapter_implementation_path",
            "adapter_implementation_sha256",
        },
        "method identity",
    )
    source = _closed_mapping(
        context["source_identity"],
        {"source_identity_sha256", "execution_source_identity_sha256"},
        "source identity",
    )
    artifact_directory = g0.get("artifact_directory")
    if (
        g0.get("image_config_digest") != context["image_config_digest"]
        or type(g0.get("run_id")) is not str
        or _E00_RUN_ID.fullmatch(str(g0.get("run_id"))) is None
        or g0.get("gpu_uuid") != PHASE3_GPU_UUID
        or g0.get("gpu_name") != EXPECTED_GPU_NAME
        or g0.get("compute_capability") != "12.0"
        or type(artifact_directory) is not str
        or not artifact_directory
        or not Path(artifact_directory).is_absolute()
        or gpu.get("uuid") != g0.get("gpu_uuid")
        or gpu.get("name") != EXPECTED_GPU_NAME
        or gpu.get("compute_capability") != "12.0"
    ):
        raise Phase6AParityError("runtime GPU/image differs from container G0")
    if (
        model.get("model_id") != EXPECTED_MODEL_ID
        or model.get("model_revision") != EXPECTED_REVISION
        or model.get("tokenizer_revision") != EXPECTED_REVISION
        or (
            model.get("num_query_heads"),
            model.get("num_kv_heads"),
            model.get("head_dim"),
        )
        != (32, 8, 128)
        or any(
            type(model.get(name)) is not int
            for name in ("num_query_heads", "num_kv_heads", "head_dim")
        )
        or model.get("weight_dtype") != "bfloat16"
    ):
        raise Phase6AParityError("frozen model/tokenizer identity differs")
    for field in (
        "artifact_root_sha256",
        "manifest_sha256",
    ):
        _sha256(g0.get(field), f"container G0 {field}")
    for field in (
        "frozen_identity_sha256",
        "snapshot_file_ledger_sha256",
        "load_receipt_sha256",
        "tokenizer_runtime_sha256",
        "parameter_runtime_sha256",
    ):
        _sha256(model.get(field), f"model {field}")
    backend_payload = backend.get("backend")
    try:
        parsed_backend = BF16BackendIdentity.from_dict(backend_payload)
    except (SchemaValidationError, TypeError, ValueError) as error:
        raise Phase6AParityError(
            "forced Flash backend identity is invalid"
        ) from error
    if (
        not isinstance(backend_payload, Mapping)
        or backend_payload.get("backend_id") != "torch_sdpa_flash_gqa"
        or backend_payload.get("selected_backend") != "flash_attention"
        or backend.get("fingerprint") != parsed_backend.fingerprint()
        or backend.get("attention_implementation") != "kvbench_bf16_flash"
        or backend.get("fallback_permitted") is not False
    ):
        raise Phase6AParityError("forced Flash backend identity differs")
    if (
        method.get("method") != "bf16"
        or method.get("method_config_id") != "bf16"
        or method.get("method_config_fingerprint")
        != PHASE3_BF16_VARIANT_FINGERPRINT
        or method.get("adapter_version")
        != "kvbench-bf16-method-adapter-1.0.0"
        or method.get("adapter_implementation_path")
        != "src/kvbench/adapters/bf16.py"
    ):
        raise Phase6AParityError("BF16 adapter identity differs")
    _sha256(
        method.get("adapter_implementation_sha256"),
        "adapter implementation",
    )
    _sha256(source.get("source_identity_sha256"), "source identity")
    _sha256(
        source.get("execution_source_identity_sha256"),
        "execution source identity",
    )


def _container_raw_audit_identities(
    *,
    runtime_gpu: Mapping[str, Any],
    container_g0: Mapping[str, Any],
    image_config_digest: str,
) -> dict[str, str]:
    """Bind raw operations to this exact GPU and certified container G0."""

    digest = _image_digest(image_config_digest)
    hardware = {
        "schema_version": "kvbench-phase6a-container-hardware-identity-1.0.0",
        "gpu_uuid": runtime_gpu.get("uuid"),
        "gpu_name": runtime_gpu.get("name"),
        "compute_capability": runtime_gpu.get("compute_capability"),
    }
    if (
        hardware["gpu_uuid"] != PHASE3_GPU_UUID
        or hardware["gpu_name"] != EXPECTED_GPU_NAME
        or hardware["compute_capability"] != "12.0"
    ):
        raise Phase6AParityError("container hardware identity differs")
    software = {
        "schema_version": "kvbench-phase6a-container-software-identity-1.0.0",
        "image_config_digest": digest,
        "container_g0_root_sha256": _sha256(
            container_g0.get("artifact_root_sha256"),
            "container G0 root",
        ),
        "container_g0_manifest_sha256": _sha256(
            container_g0.get("manifest_sha256"),
            "container G0 manifest",
        ),
    }
    return {
        "hardware_identity_sha256": sha256_hex(
            canonical_json_bytes(hardware)
        ),
        "software_identity_sha256": sha256_hex(
            canonical_json_bytes(software)
        ),
    }


def _new_run_id(graph_mode: str, git_sha: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    ).lower()
    return (
        f"phase6a-bf16-parity-{graph_mode}-{timestamp}-"
        f"{git_sha[:8]}-{secrets.token_hex(3)}"
    )


def _normalized_plan(graph_mode: str) -> dict[str, Any]:
    return {
        "schema_version": (
            "kvbench-phase6a-bf16-container-parity-plan-1.0.0"
        ),
        "run_kind": "correctness",
        "runner": "fixed_l",
        "graph_mode": graph_mode,
        "batch_size": 1,
        "context_length": 128,
        "output_steps": 1,
        "seed": 20260722,
        "process_replicate": 1,
    }


def _parity_command(
    *,
    graph_mode: str,
    image_reference: str,
    image_config_digest: str,
    container_g0_artifact: str,
) -> list[str]:
    return [
        "/opt/kvbench/.venv/bin/python",
        "scripts/phase6a_bf16_parity.py",
        "--graph-mode",
        graph_mode,
        "--image-reference",
        image_reference,
        "--image-config-digest",
        image_config_digest,
        "--container-g0-artifact",
        container_g0_artifact,
    ]


def publish_parity_lane(
    *,
    graph_mode: str,
    context: Mapping[str, Any],
    executor: LaneExecutor,
    output_root: Path = OUTPUT_ROOT,
    run_id: str | None = None,
    validation_environ: Mapping[str, str] | None = None,
) -> PublishedLane:
    """Run and immutably finalize one lane, including terminal failures."""

    if graph_mode not in LANES:
        raise Phase6AParityError("parity graph mode is invalid")
    _validate_context(context)
    identifier = (
        _new_run_id(graph_mode, str(context["execution_git_sha"]))
        if run_id is None
        else _run_identifier(run_id)
    )
    output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if output_root.is_symlink() or not output_root.is_dir():
        raise Phase6AParityError("parity artifact root is unsafe")
    stage = output_root / f".{identifier}.tmp"
    final = output_root / identifier
    if final.exists() or final.is_symlink():
        raise Phase6AParityError("parity run already exists")
    try:
        os.mkdir(stage, 0o700)
    except OSError as error:
        raise Phase6AParityError(
            "parity staging directory is unavailable"
        ) from error
    started, status, failure = _now(), "FAIL", None
    try:
        result = executor(graph_mode, identifier, stage)
        if not isinstance(result, Mapping):
            raise Phase6AParityError("executor returned an invalid result")
        validate_parity_result(graph_mode, result)
        write_exclusive(stage / "result.json", json_bytes(result))
        status = "PASS"
    except Exception as error:
        failure = _failure(error)
        write_exclusive(stage / "failure.json", json_bytes(failure))
    finished = _now()
    retained_names = {
        path.name for path in stage.rglob("*") if path.is_file()
    }
    admission_trace_collected = {
        "gqa.geometry.chrome.json",
        "mha.geometry.chrome.json",
    }.issubset(retained_names)
    command = _parity_command(
        graph_mode=graph_mode,
        image_reference=str(context["image_reference"]),
        image_config_digest=str(context["image_config_digest"]),
        container_g0_artifact=str(
            context["container_g0"]["artifact_directory"]
        ),
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": identifier,
        "status": status,
        "started_at_utc": started,
        "finished_at_utc": finished,
        "run": {
            "id": identifier,
            "status": status,
            "finished_at_utc": finished,
        },
        "execution_git_sha": context["execution_git_sha"],
        "source_worktree_clean": True,
        "setup_completed": True,
        "run_kind": "correctness",
        "runner": "fixed_l",
        "graph_mode": graph_mode,
        "batch_size": 1,
        "context_length": 128,
        "output_steps": 1,
        "seed": 20260722,
        "process_replicate": 1,
        "normalized_plan": _normalized_plan(graph_mode),
        "command": command,
        "container": {
            "image_reference": context["image_reference"],
            "image_config_digest": context["image_config_digest"],
            "digest_authoritative": True,
            "floating_tag_authoritative": False,
            "g0": context["container_g0"],
        },
        "runtime_gpu": context["runtime_gpu"],
        "model_identity": context["model_identity"],
        "backend_identity": context["backend_identity"],
        "method_identity": context["method_identity"],
        "source_identity": context["source_identity"],
        "result_path": "result.json" if status == "PASS" else None,
        "failure_path": "failure.json" if status == "FAIL" else None,
        "failure": failure,
        "functional_evidence_only": True,
        "timing_collected": False,
        "formal_timing_claim_created": False,
        "formal_performance_data_created": False,
        "independent_process_replicates_collected": False,
        "nsight_executed": False,
        "performance_profiling_executed": False,
        "untimed_admission_trace_collected": admission_trace_collected,
        "quality_benchmark_executed": False,
        "quality_status": "unvalidated",
        "quality_execution": "locked",
        "claim_eligibility": "performance_only",
        "performance_claim_eligible": False,
        "performance_data_frozen": False,
        "measurement_scope": "measurement_container_parity",
        "full_scan_state": "closed",
        "gates": {
            "g0": "PASS",
            "g1": "PASS",
            "g2_tq": "BLOCKED",
            "g2": "NOT_EVALUATED",
            "g3": "NOT_EVALUATED",
            "g4": "NOT_EVALUATED",
            "g5": "NOT_EVALUATED",
        },
        "evidence": {
            "commands": [],
            "files": enumerate_evidence_files(stage),
        },
    }
    finalize_stage(stage=stage, final=final, manifest=manifest)
    root_sha256 = validate_local_artifact(
        final,
        environ=validation_environ,
    ).root_sha256
    return PublishedLane(
        identifier,
        graph_mode,
        status,
        final.resolve(strict=True),
        root_sha256,
    )


def publish_parity_setup_failure(
    *,
    graph_mode: str,
    execution_git_sha: str,
    image_reference: str,
    image_config_digest: str,
    container_g0_artifact: Path,
    error: Exception,
    output_root: Path = OUTPUT_ROOT,
    run_id: str | None = None,
    validation_environ: Mapping[str, str] | None = None,
) -> PublishedLane:
    """Immutably retain a post-launch failure before runtime setup completes."""

    if graph_mode not in LANES:
        raise Phase6AParityError("parity graph mode is invalid")
    git_sha = _git_sha(execution_git_sha)
    digest = _image_digest(image_config_digest)
    identifier = (
        _new_run_id(graph_mode, git_sha)
        if run_id is None
        else _run_identifier(run_id)
    )
    output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if output_root.is_symlink() or not output_root.is_dir():
        raise Phase6AParityError("parity artifact root is unsafe")
    stage = output_root / f".{identifier}.tmp"
    final = output_root / identifier
    if final.exists() or final.is_symlink():
        raise Phase6AParityError("parity run already exists")
    try:
        os.mkdir(stage, 0o700)
    except OSError as reserve_error:
        raise Phase6AParityError(
            "parity staging directory is unavailable"
        ) from reserve_error
    started = _now()
    failure = _failure(error)
    write_exclusive(stage / "failure.json", json_bytes(failure))
    command = _parity_command(
        graph_mode=graph_mode,
        image_reference=image_reference,
        image_config_digest=digest,
        container_g0_artifact=container_g0_artifact.as_posix(),
    )
    finished = _now()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": identifier,
        "status": "FAIL",
        "started_at_utc": started,
        "finished_at_utc": finished,
        "run": {
            "id": identifier,
            "status": "FAIL",
            "finished_at_utc": finished,
        },
        "execution_git_sha": git_sha,
        "source_worktree_clean": True,
        "setup_completed": False,
        "setup_failure_stage": "container_g0_or_runtime_setup",
        "run_kind": "correctness",
        "runner": "fixed_l",
        "graph_mode": graph_mode,
        "batch_size": 1,
        "context_length": 128,
        "output_steps": 1,
        "seed": 20260722,
        "process_replicate": 1,
        "normalized_plan": _normalized_plan(graph_mode),
        "command": command,
        "container": {
            "image_reference": image_reference,
            "image_config_digest": digest,
            "digest_authoritative": True,
            "floating_tag_authoritative": False,
            "g0": None,
            "requested_g0_artifact": container_g0_artifact.as_posix(),
        },
        "runtime_gpu": None,
        "model_identity": None,
        "backend_identity": None,
        "method_identity": None,
        "source_identity": None,
        "result_path": None,
        "failure_path": "failure.json",
        "failure": failure,
        "functional_evidence_only": True,
        "timing_collected": False,
        "formal_timing_claim_created": False,
        "formal_performance_data_created": False,
        "independent_process_replicates_collected": False,
        "nsight_executed": False,
        "performance_profiling_executed": False,
        "untimed_admission_trace_collected": False,
        "quality_benchmark_executed": False,
        "quality_status": "unvalidated",
        "quality_execution": "locked",
        "claim_eligibility": "performance_only",
        "performance_claim_eligible": False,
        "performance_data_frozen": False,
        "measurement_scope": "measurement_container_parity",
        "full_scan_state": "closed",
        "gates": {
            "g0": "NOT_VERIFIED",
            "g1": "PASS",
            "g2_tq": "BLOCKED",
            "g2": "NOT_EVALUATED",
            "g3": "NOT_EVALUATED",
            "g4": "NOT_EVALUATED",
            "g5": "NOT_EVALUATED",
        },
        "evidence": {
            "commands": [],
            "files": enumerate_evidence_files(stage),
        },
    }
    finalize_stage(stage=stage, final=final, manifest=manifest)
    root_sha256 = validate_local_artifact(
        final,
        environ=validation_environ,
    ).root_sha256
    return PublishedLane(
        identifier,
        graph_mode,
        "FAIL",
        final.resolve(strict=True),
        root_sha256,
    )


def validate_finalized_parity_artifact(
    root: Path,
    manifest: Mapping[str, Any],
    inventory_files: Sequence[Mapping[str, object]],
    relatives: set[str],
) -> None:
    """Independently revalidate one finalized or cleanly retrieved lane."""

    from kvbench.runtime.phase3_raw_audit_evidence import (
        RAW_AUDIT_STATUS_COMPLETED,
        Phase3RawAuditRunIndex,
    )
    from preflight import run_preflight
    def fail(message: str) -> None:
        raise Phase6AParityError(message)

    def exact_mapping(
        value: object,
        keys: set[str],
        label: str,
    ) -> Mapping[str, Any]:
        if not isinstance(value, Mapping) or set(value) != keys:
            fail(f"{label} field closure differs")
        assert isinstance(value, Mapping)
        return value

    common_keys = {
        "schema_version",
        "run_id",
        "status",
        "started_at_utc",
        "finished_at_utc",
        "run",
        "execution_git_sha",
        "source_worktree_clean",
        "setup_completed",
        "run_kind",
        "runner",
        "graph_mode",
        "batch_size",
        "context_length",
        "output_steps",
        "seed",
        "process_replicate",
        "normalized_plan",
        "command",
        "container",
        "runtime_gpu",
        "model_identity",
        "backend_identity",
        "method_identity",
        "source_identity",
        "result_path",
        "failure_path",
        "failure",
        "functional_evidence_only",
        "timing_collected",
        "formal_timing_claim_created",
        "formal_performance_data_created",
        "independent_process_replicates_collected",
        "nsight_executed",
        "performance_profiling_executed",
        "untimed_admission_trace_collected",
        "quality_benchmark_executed",
        "quality_status",
        "quality_execution",
        "claim_eligibility",
        "performance_claim_eligible",
        "performance_data_frozen",
        "measurement_scope",
        "full_scan_state",
        "gates",
        "evidence",
    }
    setup_completed = manifest.get("setup_completed")
    expected_keys = set(common_keys)
    if setup_completed is False:
        expected_keys.add("setup_failure_stage")
    exact_mapping(manifest, expected_keys, "parity manifest")
    run_id = manifest.get("run_id")
    status = manifest.get("status")
    graph_mode = manifest.get("graph_mode")
    started = manifest.get("started_at_utc")
    finished = manifest.get("finished_at_utc")
    run = exact_mapping(
        manifest.get("run"),
        {"id", "status", "finished_at_utc"},
        "parity run",
    )
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or type(run_id) is not str
        or _RUN_ID.fullmatch(run_id) is None
        or type(status) is not str
        or status not in {"PASS", "FAIL"}
        or type(graph_mode) is not str
        or graph_mode not in LANES
        or run
        != {
            "id": run_id,
            "status": status,
            "finished_at_utc": manifest.get("finished_at_utc"),
        }
        or type(started) is not str
        or _UTC_TIMESTAMP.fullmatch(started) is None
        or type(finished) is not str
        or _UTC_TIMESTAMP.fullmatch(finished) is None
        or type(setup_completed) is not bool
        or manifest.get("source_worktree_clean") is not True
        or any(
            type(manifest.get(name)) is not int
            for name in (
                "batch_size",
                "context_length",
                "output_steps",
                "seed",
                "process_replicate",
            )
        )
        or (
            manifest.get("run_kind"),
            manifest.get("runner"),
            manifest.get("batch_size"),
            manifest.get("context_length"),
            manifest.get("output_steps"),
            manifest.get("seed"),
            manifest.get("process_replicate"),
        )
        != ("correctness", "fixed_l", 1, 128, 1, 20260722, 1)
        or manifest.get("normalized_plan")
        != _normalized_plan(str(graph_mode))
    ):
        fail("parity manifest identity differs")
    plan = exact_mapping(
        manifest.get("normalized_plan"),
        {
            "schema_version",
            "run_kind",
            "runner",
            "graph_mode",
            "batch_size",
            "context_length",
            "output_steps",
            "seed",
            "process_replicate",
        },
        "normalized parity plan",
    )
    if any(
        type(plan.get(name)) is not int
        for name in (
            "batch_size",
            "context_length",
            "output_steps",
            "seed",
            "process_replicate",
        )
    ):
        fail("normalized parity plan numeric type differs")
    git_sha = _git_sha(manifest.get("execution_git_sha"))
    container = manifest.get("container")
    if not isinstance(container, Mapping):
        fail("parity container identity is absent")
    image_reference = container.get("image_reference")
    image_digest = _image_digest(container.get("image_config_digest"))
    if (
        type(image_reference) is not str
        or not image_reference
        or len(image_reference) > 512
        or any(
            ord(character) < 32 or ord(character) == 127
            for character in image_reference
        )
        or container.get("digest_authoritative") is not True
        or container.get("floating_tag_authoritative") is not False
    ):
        fail("parity container digest authority differs")
    if setup_completed is True:
        exact_mapping(
            container,
            {
                "image_reference",
                "image_config_digest",
                "digest_authoritative",
                "floating_tag_authoritative",
                "g0",
            },
            "parity container",
        )
        context = {
            "execution_git_sha": git_sha,
            "image_reference": image_reference,
            "image_config_digest": image_digest,
            "container_g0": container.get("g0"),
            "runtime_gpu": manifest.get("runtime_gpu"),
            "model_identity": manifest.get("model_identity"),
            "backend_identity": manifest.get("backend_identity"),
            "method_identity": manifest.get("method_identity"),
            "source_identity": manifest.get("source_identity"),
        }
        try:
            _validate_context(context)
        except Phase6AParityError as error:
            raise Phase6AParityError(
                "parity runtime identity differs"
            ) from error
        g0 = context["container_g0"]
        assert isinstance(g0, Mapping)
        g0_path = str(g0["artifact_directory"])
        expected_gates = {
            "g0": "PASS",
            "g1": "PASS",
            "g2_tq": "BLOCKED",
            "g2": "NOT_EVALUATED",
            "g3": "NOT_EVALUATED",
            "g4": "NOT_EVALUATED",
            "g5": "NOT_EVALUATED",
        }
    else:
        exact_mapping(
            container,
            {
                "image_reference",
                "image_config_digest",
                "digest_authoritative",
                "floating_tag_authoritative",
                "g0",
                "requested_g0_artifact",
            },
            "parity setup-failure container",
        )
        if (
            status != "FAIL"
            or manifest.get("setup_failure_stage")
            != "container_g0_or_runtime_setup"
            or container.get("g0") is not None
            or any(
                manifest.get(name) is not None
                for name in (
                    "runtime_gpu",
                    "model_identity",
                    "backend_identity",
                    "method_identity",
                    "source_identity",
                )
            )
            or type(container.get("requested_g0_artifact")) is not str
            or not container.get("requested_g0_artifact")
        ):
            fail("parity setup-failure identity differs")
        g0_path = str(container["requested_g0_artifact"])
        expected_gates = {
            "g0": "NOT_VERIFIED",
            "g1": "PASS",
            "g2_tq": "BLOCKED",
            "g2": "NOT_EVALUATED",
            "g3": "NOT_EVALUATED",
            "g4": "NOT_EVALUATED",
            "g5": "NOT_EVALUATED",
        }
    command = _parity_command(
        graph_mode=str(graph_mode),
        image_reference=str(image_reference),
        image_config_digest=image_digest,
        container_g0_artifact=g0_path,
    )
    exact_claims = {
        "functional_evidence_only": True,
        "timing_collected": False,
        "formal_timing_claim_created": False,
        "formal_performance_data_created": False,
        "independent_process_replicates_collected": False,
        "nsight_executed": False,
        "performance_profiling_executed": False,
        "quality_benchmark_executed": False,
        "quality_status": "unvalidated",
        "quality_execution": "locked",
        "claim_eligibility": "performance_only",
        "performance_claim_eligible": False,
        "performance_data_frozen": False,
        "measurement_scope": "measurement_container_parity",
        "full_scan_state": "closed",
    }
    boolean_claims = {
        "functional_evidence_only",
        "timing_collected",
        "formal_timing_claim_created",
        "formal_performance_data_created",
        "independent_process_replicates_collected",
        "nsight_executed",
        "performance_profiling_executed",
        "quality_benchmark_executed",
        "performance_claim_eligible",
        "performance_data_frozen",
    }
    gates = exact_mapping(
        manifest.get("gates"),
        {"g0", "g1", "g2_tq", "g2", "g3", "g4", "g5"},
        "parity gates",
    )
    if (
        manifest.get("command") != command
        or gates != expected_gates
        or any(type(manifest.get(name)) is not bool for name in boolean_claims)
        or any(
            manifest.get(name) != value
            for name, value in exact_claims.items()
        )
        or type(manifest.get("untimed_admission_trace_collected"))
        is not bool
    ):
        fail("parity plan or claim boundary differs")
    evidence = exact_mapping(
        manifest.get("evidence"),
        {"commands", "files"},
        "parity evidence",
    )
    if evidence.get("commands") != [] or not isinstance(
        evidence.get("files"), list
    ):
        fail("parity evidence projection differs")
    for item in evidence["files"]:
        item = exact_mapping(
            item,
            {"id", "path", "role", "sha256", "size_bytes"},
            "parity evidence file",
        )
        relative = item.get("path")
        if (
            type(item.get("id")) is not str
            or not item.get("id")
            or type(relative) is not str
            or not relative
            or type(item.get("role")) is not str
            or not item.get("role")
            or type(item.get("sha256")) is not str
            or _SHA256.fullmatch(item.get("sha256")) is None
            or type(item.get("size_bytes")) is not int
            or item.get("size_bytes") < 0
        ):
            fail("parity evidence file is invalid")
    for item in inventory_files:
        relative = item.get("path")
        if type(relative) is not str:
            fail("parity inventory path is invalid")
        expected_role = (
            "manifest"
            if relative == "manifest.json"
            else run_preflight.file_role(str(relative))
        )
        if item.get("role") != expected_role:
            fail("parity inventory role differs")
    actual_evidence = run_preflight.enumerate_evidence_files(root)
    generated_inventory = [
        item
        for item in actual_evidence
        if item.get("path") == "artifact_inventory.json"
    ]
    if len(generated_inventory) != 1:
        fail("parity generated inventory evidence is absent")
    projected_manifest = dict(manifest)
    projected_evidence = dict(evidence)
    projected_evidence["files"] = sorted(
        [*evidence["files"], generated_inventory[0]],
        key=lambda item: item["path"],
    )
    projected_manifest["evidence"] = projected_evidence
    if run_preflight.evidence_reference_errors(root, projected_manifest):
        fail("parity evidence references differ")
    if status == "FAIL":
        if (
            manifest.get("result_path") is not None
            or manifest.get("failure_path") != "failure.json"
            or "result.json" in relatives
            or "failure.json" not in relatives
        ):
            fail("parity failure closure differs")
        failure = exact_mapping(
            manifest.get("failure"),
            {"error_type", "reason"},
            "parity failure",
        )
        if any(
            type(failure.get(name)) is not str or not failure.get(name)
            for name in failure
        ):
            fail("parity failure record is invalid")
        if _load_json(root / "failure.json", "parity failure") != failure:
            fail("parity failure record differs from manifest")
        return
    if (
        setup_completed is not True
        or manifest.get("result_path") != "result.json"
        or manifest.get("failure_path") is not None
        or manifest.get("failure") is not None
        or manifest.get("untimed_admission_trace_collected") is not True
        or "result.json" not in relatives
        or "failure.json" in relatives
        or "raw-audit-index.json" not in relatives
    ):
        fail("parity PASS closure differs")
    index_path = root / "raw-audit-index.json"
    index_payload = _load_json(index_path, "raw-audit index")
    if index_path.read_bytes() != json_bytes(index_payload):
        fail("raw-audit index is not canonical")
    try:
        index = Phase3RawAuditRunIndex.from_dict(index_payload)
    except (SchemaValidationError, TypeError, ValueError) as error:
        raise Phase6AParityError("raw-audit index is invalid") from error
    expected_point = f"fixed_l-b1-l128-{graph_mode}-r1"
    if (
        index.run_id != run_id
        or index.point_id != expected_point
        or len(index.records) != 1
        or index.records[0].status != RAW_AUDIT_STATUS_COMPLETED
    ):
        fail("raw-audit index identity differs")
    record = index.records[0]
    operation = record.operation
    assert isinstance(context, dict)
    model_identity = context["model_identity"]
    backend_identity = context["backend_identity"]
    source_identity = context["source_identity"]
    assert isinstance(model_identity, Mapping)
    assert isinstance(backend_identity, Mapping)
    assert isinstance(source_identity, Mapping)
    if (
        operation.point_id != expected_point
        or operation.execution_git_sha != git_sha
        or operation.hardware_identity_sha256
        != _container_raw_audit_identities(
            runtime_gpu=context["runtime_gpu"],
            container_g0=context["container_g0"],
            image_config_digest=image_digest,
        )["hardware_identity_sha256"]
        or operation.software_identity_sha256
        != _container_raw_audit_identities(
            runtime_gpu=context["runtime_gpu"],
            container_g0=context["container_g0"],
            image_config_digest=image_digest,
        )["software_identity_sha256"]
        or operation.model_identity_sha256
        != model_identity.get("frozen_identity_sha256")
        or operation.backend_identity_sha256
        != backend_identity.get("fingerprint")
        or operation.source_identity_sha256
        != source_identity.get("source_identity_sha256")
    ):
        fail("raw-audit operation identity differs")
    declared: set[str] = set()
    trace_names: set[str] = set()
    for declaration in record.files:
        relative = f"raw/audits/{declaration.path}"
        declared.add(relative)
        target = root / relative
        if (
            not target.is_file()
            or target.stat().st_size != declaration.size_bytes
            or sha256_file(target) != declaration.sha256
        ):
            fail("raw-audit declared file differs")
        if declaration.kind in {
            "b011_gqa_chrome_trace",
            "b011_mha_chrome_trace",
        }:
            trace_names.add(target.name)
    actual_raw = {
        relative
        for relative in relatives
        if relative.startswith("raw/audits/")
    }
    if declared != actual_raw or trace_names != {
        "gqa.geometry.chrome.json",
        "mha.geometry.chrome.json",
    }:
        fail("raw-audit file closure differs")
    result = _load_json(root / "result.json", "parity result")
    try:
        validate_parity_result(str(graph_mode), result)
    except Phase6AParityError as error:
        raise Phase6AParityError("parity result is invalid") from error
    semantics = result.get("raw_audit_semantics")
    operations = (
        semantics.get("semantic_operations")
        if isinstance(semantics, Mapping)
        else None
    )
    semantic_operation = (
        operations[0]
        if isinstance(operations, list) and len(operations) == 1
        else None
    )
    if (
        not isinstance(semantics, Mapping)
        or semantics.get("raw_audit_index_sha256")
        != sha256_file(index_path)
        or not isinstance(semantic_operation, Mapping)
        or semantic_operation.get("operation_fingerprint_sha256")
        != operation.operation_fingerprint_sha256
        or result.get("cache_layout_fingerprint")
        != operation.cache_layout_fingerprint
        or result.get("adapter_config_fingerprint")
        != semantics.get("adapter_runtime_fingerprint")
    ):
        fail("parity result/raw-index join differs")


def _git_context() -> str:
    environment = {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"}
    head = subprocess.run(
        ("/usr/bin/git", "rev-parse", "HEAD"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    ).stdout.strip()
    _git_sha(head)
    status = subprocess.run(
        (
            "/usr/bin/git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    ).stdout
    if status:
        raise Phase6AParityError("container parity requires a clean worktree")
    return head


def _runtime_gpu(expected: Mapping[str, str]) -> dict[str, str]:
    result = subprocess.run(
        (
            "/usr/bin/nvidia-smi",
            "--query-gpu=uuid,name,compute_cap",
            "--format=csv,noheader,nounits",
        ),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"},
    )
    rows = [
        tuple(part.strip() for part in line.split(","))
        for line in result.stdout.splitlines()
        if line.strip()
    ]
    if len(rows) != 1 or rows[0] != (
        expected.get("gpu_uuid"),
        EXPECTED_GPU_NAME,
        "12.0",
    ):
        raise Phase6AParityError("runtime GPU differs from container G0")
    return {
        "uuid": rows[0][0],
        "name": rows[0][1],
        "compute_capability": rows[0][2],
    }


@dataclass(slots=True)
class _LiveRuntime:
    torch: Any
    loaded: Any
    backend: Any
    adapter: Any
    method_fingerprint: Any
    source_pin: Any


def _verify_phase3_source_alias() -> None:
    """Require the frozen Phase 3 root to contain the exact live SUT bytes."""

    from kvbench.runtime.gqa_device_dispatch import REQUIRED_SUT_SOURCES
    from kvbench.schema.phase3 import PHASE3_REPOSITORY_ROOT

    frozen_root = Path(PHASE3_REPOSITORY_ROOT)
    for relative in REQUIRED_SUT_SOURCES:
        try:
            frozen = (frozen_root / relative).read_bytes()
            live = (REPOSITORY_ROOT / relative).read_bytes()
        except OSError as error:
            raise Phase6AParityError(
                "frozen Phase 3 source alias is unavailable"
            ) from error
        if frozen != live:
            raise Phase6AParityError("frozen Phase 3 source alias differs")


def _prepare_runtime(git_sha: str) -> tuple[_LiveRuntime, dict[str, Any]]:
    from kvbench.runtime.backend import (
        ATTENTION_IMPLEMENTATION,
        backend_identity,
    )
    from kvbench.runtime.model_loader import load_frozen_model
    from kvbench.runtime.phase3_coordinator import (
        _pin_phase3_execution_sources,
    )
    from kvbench.runtime.phase4_smoke import _build_adapter
    from kvbench.schema.phase3 import BF16BackendIdentity

    _verify_phase3_source_alias()
    torch = __import__("torch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise Phase6AParityError("exactly one CUDA device must be visible")
    if tuple(torch.cuda.get_device_capability(0)) != (12, 0):
        raise Phase6AParityError("CUDA compute capability differs from 12.0")
    loaded = load_frozen_model(device="cuda:0")
    if (
        getattr(loaded.model.config, "_attn_implementation", None)
        != ATTENTION_IMPLEMENTATION
    ):
        raise Phase6AParityError("loaded model did not retain forced Flash")
    backend_payload = backend_identity()
    backend = BF16BackendIdentity.from_dict(backend_payload)
    adapter, method_fingerprint = _build_adapter(loaded, backend_payload)
    if method_fingerprint.sha256 != PHASE3_BF16_VARIANT_FINGERPRINT:
        raise Phase6AParityError("frozen BF16 variant fingerprint differs")
    adapter_source = REPOSITORY_ROOT / "src/kvbench/adapters/bf16.py"
    if not adapter_source.is_file() or adapter_source.is_symlink():
        raise Phase6AParityError("BF16 adapter source is unavailable")
    source_pin = _pin_phase3_execution_sources(git_sha)
    identity = loaded.identity
    if (
        identity.model_id,
        identity.revision,
        identity.num_attention_heads,
        identity.num_key_value_heads,
        identity.weight_dtype,
    ) != (EXPECTED_MODEL_ID, EXPECTED_REVISION, 32, 8, "bfloat16"):
        raise Phase6AParityError("loaded model identity differs")
    runtime = _LiveRuntime(
        torch,
        loaded,
        backend,
        adapter,
        method_fingerprint,
        source_pin,
    )
    identities = {
        "model_identity": {
            "model_id": identity.model_id,
            "model_revision": identity.revision,
            "tokenizer_revision": identity.revision,
            "frozen_identity_sha256": loaded.receipt.frozen_identity_sha256,
            "snapshot_file_ledger_sha256": (
                loaded.receipt.snapshot_file_ledger_sha256
            ),
            "load_receipt_sha256": loaded.receipt.receipt_sha256,
            "tokenizer_runtime_sha256": (
                loaded.receipt.tokenizer_runtime_sha256
            ),
            "parameter_runtime_sha256": (
                loaded.receipt.parameter_runtime_sha256
            ),
            "num_query_heads": identity.num_attention_heads,
            "num_kv_heads": identity.num_key_value_heads,
            "head_dim": identity.head_dim,
            "weight_dtype": identity.weight_dtype,
        },
        "backend_identity": {
            "backend": backend_payload,
            "fingerprint": backend.fingerprint(),
            "attention_implementation": ATTENTION_IMPLEMENTATION,
            "fallback_permitted": False,
        },
        "method_identity": {
            "method": adapter.name,
            "method_config_id": "bf16",
            "method_config_fingerprint": method_fingerprint.sha256,
            "adapter_version": adapter.adapter_version,
            "adapter_implementation_path": (
                "src/kvbench/adapters/bf16.py"
            ),
            "adapter_implementation_sha256": sha256_file(adapter_source),
        },
        "source_identity": {
            "source_identity_sha256": source_pin.source_identity_sha256,
            "execution_source_identity_sha256": (
                source_pin.execution_source_identity_sha256
            ),
        },
    }
    return runtime, identities


def _retained(root: Path, record: Any) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for declaration in record.files:
        path = root / declaration.path
        metadata = path.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise Phase6AParityError("raw-audit evidence path is unsafe")
        payload = path.read_bytes()
        if (
            len(payload) != declaration.size_bytes
            or hashlib.sha256(payload).hexdigest() != declaration.sha256
        ):
            raise Phase6AParityError("raw-audit evidence digest differs")
        result[declaration.path] = payload
    return result


def _raw_audit(
    runtime: _LiveRuntime,
    *,
    graph_mode: str,
    run_id: str,
    stage: Path,
    expected_adapter_config_fingerprint: str,
    raw_audit_identities: Mapping[str, str],
) -> dict[str, Any]:
    from kvbench.runtime.backend import forced_flash_execution
    from kvbench.runtime.gqa_device_dispatch import REQUIRED_SUT_SOURCES
    from kvbench.runtime.phase3_audit_operation import Phase3AuditOperationKey
    from kvbench.runtime.phase3_coordinator import (
        _cache_identity,
        _replay_phase3_raw_audit_semantics,
        _revalidate_phase3_execution_sources,
    )
    from kvbench.runtime.phase3_raw_audit_evidence import (
        RAW_AUDIT_STATUS_COMPLETED,
    )
    from kvbench.runtime.phase3_worker import (
        _deterministic_ids,
        _phase3_raw_audit_producer_bindings,
    )
    from kvbench.runtime.phase3_worker_channels import (
        Phase3RawAuditProducerRegistry,
        require_phase3_raw_audit_measurement_admission,
    )
    from kvbench.schema import GraphMode, RunnerKind
    from kvbench.schema.phase3 import (
        PHASE3_FIXED_PLAN_PATH,
        PHASE3_PLAN_FINGERPRINTS,
        Phase3ProcessPoint,
    )

    point = Phase3ProcessPoint(
        point_id=f"fixed_l-b1-l128-{graph_mode}-r1",
        runner_kind=RunnerKind.FIXED_L,
        graph_mode=GraphMode(graph_mode),
        batch_size=1,
        context_length=128,
        output_steps=1,
        process_replicate=1,
        stability_member=False,
    )
    hashes = runtime.source_pin.sut_source_sha256_by_path
    if set(hashes) != set(REQUIRED_SUT_SOURCES):
        raise Phase6AParityError("pinned raw-audit source set differs")
    cache = _cache_identity(
        point,
        implementation_sha256=hashes[
            "src/kvbench/runtime/static_cache.py"
        ],
    )
    operations = (
        Phase3AuditOperationKey.from_point(
            run_id=run_id,
            point=point,
            decode_step=0,
            cache_layout_fingerprint=cache.layout_fingerprint,
            execution_git_sha=runtime.source_pin.execution_git_sha,
            plan_fingerprint=PHASE3_PLAN_FINGERPRINTS[
                PHASE3_FIXED_PLAN_PATH
            ],
            hardware_identity_sha256=_sha256(
                raw_audit_identities.get("hardware_identity_sha256"),
                "container raw-audit hardware identity",
            ),
            software_identity_sha256=_sha256(
                raw_audit_identities.get("software_identity_sha256"),
                "container raw-audit software identity",
            ),
            model_identity_sha256=runtime.loaded.receipt.frozen_identity_sha256,
            backend_identity_sha256=runtime.backend.fingerprint(),
            source_identity_sha256=runtime.source_pin.source_identity_sha256,
        ),
    )
    device = runtime.torch.device("cuda:0")
    prefix = _deterministic_ids(
        runtime.torch,
        batch_size=1,
        length=128,
        offset=10_000,
        device=device,
    )
    decode = _deterministic_ids(
        runtime.torch,
        batch_size=1,
        length=1,
        offset=40_000,
        device=device,
    )
    root = stage / "raw/audits"
    root.mkdir(mode=0o700, parents=True)
    with runtime.torch.inference_mode(), forced_flash_execution():
        _session, bindings = _phase3_raw_audit_producer_bindings(
            expected_operations=operations,
            torch=runtime.torch,
            device=device,
            loaded=runtime.loaded,
            point=point,
            prefix_input_ids=prefix,
            decode_input_ids=decode,
        )
        registry = Phase3RawAuditProducerRegistry(operations)
        for operation, producer in bindings:
            registry.register(operation, producer)
        index = registry.collect(root)
    index_payload = index.to_dict()
    index_raw = json_bytes(index_payload)
    write_exclusive(
        stage / "raw-audit-index.json",
        index_raw,
    )
    require_phase3_raw_audit_measurement_admission(index, operations)
    record = index.records[0]
    if record.status != RAW_AUDIT_STATUS_COMPLETED:
        raise Phase6AParityError("exact endpoint raw audit did not complete")
    retained = _retained(root, record)
    provenance_path = next(
        item.path
        for item in record.files
        if item.kind == "phase3_session_provenance"
    )
    provenance_raw = retained[provenance_path]
    provenance, legacy_provenance_raw = (
        _phase3_provenance_for_legacy_replay(
            provenance_raw,
            expected_adapter_config_fingerprint=(
                expected_adapter_config_fingerprint
            ),
        )
    )
    replay_retained = dict(retained)
    replay_retained[provenance_path] = legacy_provenance_raw
    _revalidate_phase3_execution_sources(runtime.source_pin)
    outcome: dict[str, Any] = {
        "process_audit_passed": False,
        "commitment_validation_passed": False,
        "execution_source_revalidated_after_worker_exit": True,
    }
    _replay_phase3_raw_audit_semantics(
        index=index,
        retained=replay_retained,
        execution_source_pin=runtime.source_pin,
        backend_identity=runtime.backend,
        outcome=outcome,
    )
    if (
        outcome.get("semantic_validation_passed") is not True
        or outcome.get("scientific_completion_passed") is not True
    ):
        raise Phase6AParityError("raw-audit semantic replay did not pass")
    semantic_operations = outcome.get("semantic_operations")
    if (
        not isinstance(semantic_operations, list)
        or len(semantic_operations) != 1
    ):
        raise Phase6AParityError("raw-audit semantic operation count differs")
    semantic_operation = dict(semantic_operations[0])
    semantic_operation["operation_output_sha256"] = (
        provenance["audit_output_sha256"][0]
    )
    semantic_operation["operation_output_finite"] = (
        provenance["audit_output_finite"][0]
    )
    return {
        "semantic_validation_passed": True,
        "scientific_completion_passed": True,
        "transport_terminal_eligible": False,
        "semantic_operations": [semantic_operation],
        "raw_audit_index_sha256": sha256_hex(index_raw),
        "adapter_runtime_fingerprint": (
            expected_adapter_config_fingerprint
        ),
        "legacy_replay_projection_applied": True,
        "source_revalidated_after_execution": True,
    }


def _numerical(runtime: _LiveRuntime, graph_mode: str) -> dict[str, Any]:
    from kvbench.runtime.cuda_graph import validate_full_model_fixed_graph
    from kvbench.runtime.numerical import validate_full_model_reference
    from kvbench.runtime.phase3_worker import _small_attention_controls
    from kvbench.runtime.phase4_smoke import _deterministic_ids

    device = runtime.torch.device("cuda:0")
    small = _small_attention_controls(runtime.torch, device=device)
    prefix = _deterministic_ids(
        runtime.torch,
        length=128,
        offset=10_000,
    )
    decode = _deterministic_ids(
        runtime.torch,
        length=3,
        offset=40_000,
    )
    full = validate_full_model_reference(
        runtime.loaded.model,
        prefix,
        decode,
        method=runtime.adapter,
    ).to_dict()
    graph = None
    if graph_mode == "cuda_graph":
        graph = validate_full_model_fixed_graph(
            runtime.loaded.model,
            prefix,
            decode[:, :1],
            method=runtime.adapter,
        ).to_dict()
    passed = (
        small.get("passed") is True
        and full.get("passed") is True
        and (graph is None or graph.get("passed") is True)
    )
    return {
        "passed": passed,
        "small_tensor": small,
        "full_model": full,
        "full_model_graph": graph,
        "timing_collected": False,
    }


def _executor(
    runtime: _LiveRuntime,
    raw_audit_identities: Mapping[str, str],
) -> LaneExecutor:
    from kvbench.runtime.phase4_smoke import _fixed_eager, _fixed_graph

    scenarios = {"eager": _fixed_eager, "cuda_graph": _fixed_graph}

    def execute(
        graph_mode: str,
        run_id: str,
        stage: Path,
    ) -> Mapping[str, Any]:
        try:
            numerical = _numerical(runtime, graph_mode)
            scenario = scenarios[graph_mode](
                runtime.adapter,
                runtime.loaded,
                runtime.torch,
            )
            adapter_runtime_fingerprint = _sha256(
                scenario.get("adapter_config_fingerprint"),
                "scenario adapter runtime fingerprint",
            )
            semantics = _raw_audit(
                runtime,
                graph_mode=graph_mode,
                run_id=run_id,
                stage=stage,
                expected_adapter_config_fingerprint=(
                    adapter_runtime_fingerprint
                ),
                raw_audit_identities=raw_audit_identities,
            )
            operation = semantics["semantic_operations"][0]
            full = numerical.get("full_model")
            fixed_steps = (
                full.get("fixed_steps")
                if isinstance(full, Mapping)
                else None
            )
            if not isinstance(fixed_steps, list) or not fixed_steps:
                raise Phase6AParityError(
                    "L=128 trusted-reference evidence is absent"
                )
            trusted_observed = _sha256(
                fixed_steps[0].get("observed_checksum")
                if isinstance(fixed_steps[0], Mapping)
                else None,
                "L=128 trusted-reference observed output",
            )
            raw_output = _sha256(
                operation.get("operation_output_sha256"),
                "raw-audit output",
            )
            checksum_join: dict[str, Any] = {
                "passed": False,
                "trusted_reference_observed_sha256": trusted_observed,
                "raw_audit_output_sha256": raw_output,
            }
            if graph_mode == "eager":
                scenario_output = _sha256(
                    scenario.get("output_sha256"),
                    "eager scenario output",
                )
                if not (
                    scenario_output == trusted_observed == raw_output
                ):
                    raise Phase6AParityError(
                        "eager L=128 output checksum join failed"
                    )
                checksum_join.update(
                    {
                        "passed": True,
                        "scenario_output_sha256": scenario_output,
                    }
                )
            else:
                scenario_graph = scenario.get("graph_validation")
                numerical_graph = numerical.get("full_model_graph")
                if not isinstance(scenario_graph, Mapping) or not isinstance(
                    numerical_graph, Mapping
                ):
                    raise Phase6AParityError(
                        "graph checksum evidence is absent"
                    )
                scenario_eager = _sha256(
                    scenario_graph.get("eager_checksum"),
                    "graph scenario eager output",
                )
                scenario_first = _sha256(
                    scenario_graph.get("first_replay_checksum"),
                    "graph scenario first replay",
                )
                scenario_second = _sha256(
                    scenario_graph.get("second_replay_checksum"),
                    "graph scenario second replay",
                )
                numerical_eager = _sha256(
                    numerical_graph.get("eager_checksum"),
                    "graph numerical eager output",
                )
                numerical_first = _sha256(
                    numerical_graph.get("first_replay_checksum"),
                    "graph numerical first replay",
                )
                numerical_second = _sha256(
                    numerical_graph.get("second_replay_checksum"),
                    "graph numerical second replay",
                )
                if not (
                    scenario_eager
                    == numerical_eager
                    == trusted_observed
                    and scenario_first
                    == scenario_second
                    == numerical_first
                    == numerical_second
                    == raw_output
                ):
                    raise Phase6AParityError(
                        "graph L=128 output checksum join failed"
                    )
                checksum_join.update(
                    {
                        "passed": True,
                        "scenario_eager_sha256": scenario_eager,
                        "scenario_first_replay_sha256": scenario_first,
                        "scenario_second_replay_sha256": scenario_second,
                    }
                )
            result: dict[str, Any] = {
                **scenario,
                "passed": (
                    scenario.get("passed") is True
                    and numerical.get("passed") is True
                    and semantics.get("semantic_validation_passed") is True
                ),
                "numerical": numerical,
                "raw_audit_semantics": semantics,
                "output_checksum_join": checksum_join,
                "backend_fallback": False,
                "eager_allocation_criterion_passed": (
                    operation.get("allocation_criterion_id")
                    == "phase3_eager_attributed_ephemeral_v1"
                    and operation.get("allocation_failure_reasons") == []
                )
                if graph_mode == "eager"
                else None,
            }
            if graph_mode == "cuda_graph":
                result["graph_validation"] = scenario.get("graph_validation")
            return result
        finally:
            gc.collect()
            runtime.torch.cuda.empty_cache()

    return execute


def run_live(
    *,
    graph_mode: str,
    image_reference: str,
    image_config_digest: str,
    container_g0_artifact: Path,
) -> dict[str, Any]:
    """Run exactly the two parity lanes after validating container G0."""

    digest = _image_digest(image_config_digest)
    git_sha = _git_context()
    run_id = _new_run_id(graph_mode, git_sha)
    try:
        g0 = load_container_g0_artifact(
            container_g0_artifact,
            expected_image_config_digest=digest,
        )
        runtime_gpu = _runtime_gpu(g0)
        runtime, identities = _prepare_runtime(git_sha)
        raw_audit_identities = _container_raw_audit_identities(
            runtime_gpu=runtime_gpu,
            container_g0=g0,
            image_config_digest=digest,
        )
        context = {
            "execution_git_sha": git_sha,
            "image_reference": image_reference,
            "image_config_digest": digest,
            "container_g0": g0,
            "runtime_gpu": runtime_gpu,
            **identities,
        }
        _validate_context(context)
    except Exception as error:
        published = publish_parity_setup_failure(
            graph_mode=graph_mode,
            execution_git_sha=git_sha,
            image_reference=image_reference,
            image_config_digest=digest,
            container_g0_artifact=container_g0_artifact,
            error=error,
            run_id=run_id,
        )
        return {
            "schema_version": INDEX_SCHEMA_VERSION,
            "status": "FAIL",
            "execution_git_sha": git_sha,
            "image_config_digest": digest,
            "timing_collected": False,
            "performance_claim_eligible": False,
            "measurement_scope": "measurement_container_parity",
            "runs": [published.to_dict()],
        }
    published = publish_parity_lane(
        graph_mode=graph_mode,
        context=context,
        executor=_executor(runtime, raw_audit_identities),
        run_id=run_id,
    )
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "status": published.status,
        "execution_git_sha": git_sha,
        "image_config_digest": digest,
        "timing_collected": False,
        "performance_claim_eligible": False,
        "measurement_scope": "measurement_container_parity",
        "runs": [published.to_dict()],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-reference", required=True)
    parser.add_argument("--image-config-digest", required=True)
    parser.add_argument("--graph-mode", required=True, choices=LANES)
    parser.add_argument(
        "--container-g0-artifact",
        required=True,
        type=Path,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_live(
            graph_mode=args.graph_mode,
            image_reference=args.image_reference,
            image_config_digest=args.image_config_digest,
            container_g0_artifact=args.container_g0_artifact,
        )
    except Exception as error:
        result = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "status": "FAIL",
            "graph_mode": args.graph_mode,
            "failure": _failure(error),
            "timing_collected": False,
            "performance_claim_eligible": False,
            "measurement_scope": "measurement_container_parity",
        }
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
