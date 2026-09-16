#!/usr/bin/env python3
"""CPU-only Phase 17 modeling over the immutable Phase 16R Full Scan.

This module never imports torch, launches CUDA, mutates source evidence, or
uses profiler durations as timing.  It prepares an exact process-level frame,
freezes grouped splits, evaluates the preregistered candidates, fits local
knee descriptions, exports an offline predictor, and seals one derived bundle.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
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

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from kvbench.runtime.artifacts import sha256_file
from preflight.run_preflight import json_bytes, rename_noreplace, write_exclusive
from scripts.r2_artifact import validate_local_artifact
import scripts.phase13_pilot as phase13


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = REPOSITORY_ROOT / "configs/plans/phase17_modeling.yaml"
SOURCE_FAMILY_ID = "phase16-20260831t123029614620z-ec534d99-de80ac"
SOURCE_ROOT = REPOSITORY_ROOT / "artifacts/phase16" / SOURCE_FAMILY_ID
WALL_ROOT = SOURCE_ROOT / "wall-closure"
OUTER_ROOT = SOURCE_ROOT / "outer"
RESULT_ROOT = SOURCE_ROOT / "wall-analysis/source-results"
ARTIFACT_ROOT = REPOSITORY_ROOT / "artifacts/phase17"
DOC_REPORT = REPOSITORY_ROOT / "docs/phase_reports/phase17-modeling.md"
DOC_EVIDENCE = REPOSITORY_ROOT / "docs/evidence/phase17/modeling.json"
DOC_RECEIPT = REPOSITORY_ROOT / "docs/evidence/phase17/r2-publication.json"

WALL_ROOT_SHA256 = "5605558be0483ddfeffd251977306d3397aa27a66309324c6011e5043584103e"
OUTER_ROOT_SHA256 = "d74587675dd59b464d81c6e82885d3c9706c681a9da216ad1a7fe4c6ccd88daa"
PHASE15_ROOT_SHA256 = "641fc02d8fa598097885b74a336b1b1f454d9844b90025cf0c4b427bee02d5e8"
EXPECTED_RECORDS = 2670
EXPECTED_COMPLETED = 2205
EXPECTED_INFEASIBLE = 465
EXPECTED_LOGICAL_POINTS = 534
EXPECTED_FEASIBLE_POINTS = 441
EXPECTED_REPLACEMENTS = 38
CONFIGURATIONS = (
    "bf16", "tq_4bit_nc", "tq_k3v4_nc", "tq_3bit_nc",
    "k4v4", "k2v4", "k2v2", "kvq4", "kvq3", "kvq2",
)
FAMILIES = ("bf16", "turboquant", "kivi", "kvquant")
COMPRESSED_FAMILIES = ("turboquant", "kivi", "kvquant")
BATCHES = (1, 2, 4, 8, 16)
REPLICATES = (0, 1, 2, 3, 4)
CONTEXT_BAND = (24576, 49152)
PREDICTIVE_MODELS = ("E", "RQ2", "D", "F_shape", "F_diagnostic")
DEPLOYABLE_MODELS = ("E", "RQ2", "D")
GEOMETRY_PROTOCOLS = (
    "leave_one_batch_out", "leave_one_config_out", "leave_context_band_out"
)
BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 20260918
RIDGE = 1e-6
SURFACE_RIDGE = 1e-5
SURFACE_ITERATIONS = 3000
SURFACE_LR = 0.02
SURFACE_STARTS = (0.12, 0.35, 0.68)
L_SCALE = 131071.0
_BUNDLE_RE = re.compile(
    r"phase17-[0-9]{8}t[0-9]{12}z-[0-9a-f]{8}-[0-9a-f]{6}\Z"
)


class Phase17Error(RuntimeError):
    """Phase 17 failed closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _strict_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value {value}")
            ),
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise Phase17Error(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise Phase17Error(f"JSON root is not an object: {path}")
    return value


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in pq.read_table(path).to_pylist()]
    except (OSError, TypeError, ValueError) as error:
        raise Phase17Error(f"invalid Parquet: {path}") from error


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([dict(row) for row in rows])
    pq.write_table(table, path, compression="zstd", use_dictionary=True)


def _family(configuration: str) -> str:
    if configuration == "bf16":
        return "bf16"
    if configuration.startswith("tq_"):
        return "turboquant"
    if configuration.startswith("kvq"):
        return "kvquant"
    if configuration.startswith("k") and "v" in configuration:
        return "kivi"
    raise Phase17Error(f"unknown configuration: {configuration}")


def _bits(configuration: str) -> tuple[int | None, int | None]:
    values: dict[str, tuple[int | None, int | None]] = {
        "bf16": (16, 16),
        "tq_4bit_nc": (4, 4), "tq_k3v4_nc": (3, 4), "tq_3bit_nc": (3, 3),
        "k4v4": (4, 4), "k2v4": (2, 4), "k2v2": (2, 2),
        "kvq4": (4, 4), "kvq3": (3, 3), "kvq2": (2, 2),
    }
    return values[configuration]


def _condition_key(row: Mapping[str, Any]) -> str:
    return (
        f"{row['method_config_id']}|B{int(row['batch_size'])}|"
        f"L{int(row['historical_context'])}"
    )


def _source_hashes() -> dict[str, str]:
    paths = (
        "docs/phase_reports/phase16r-full-scan.md",
        "docs/evidence/phase16r/full-scan-publication.json",
        f"artifacts/phase16/{SOURCE_FAMILY_ID}/outer/family_manifest.json",
        f"artifacts/phase16/{SOURCE_FAMILY_ID}/wall-closure/manifest.json",
        f"artifacts/phase16/{SOURCE_FAMILY_ID}/wall-closure/source-references.json",
        f"artifacts/phase16/{SOURCE_FAMILY_ID}/wall-closure/raw_run_index.parquet",
        f"artifacts/phase16/{SOURCE_FAMILY_ID}/wall-closure/source_result_index.parquet",
        f"artifacts/phase16/{SOURCE_FAMILY_ID}/wall-closure/point_summary.parquet",
        f"artifacts/phase16/{SOURCE_FAMILY_ID}/outer/feasibility.parquet",
        "docs/phase_reports/phase14-analysis-closure.md",
        "docs/phase_reports/phase15-profiler-subset.md",
        "configs/plans/phase17_modeling.yaml",
    )
    return {path: sha256_file(REPOSITORY_ROOT / path) for path in paths}


def _flatten_features(point: Mapping[str, Any]) -> dict[str, Any]:
    allocated = point.get("allocated_bytes")
    if not isinstance(allocated, (int, float)) or allocated <= 0:
        return {
            "metadata_fraction": None, "full_precision_fraction": None,
            "outlier_fraction": None, "workspace_fraction": None,
        }
    allocated = float(allocated)
    configuration = str(point["method_config_id"])
    if configuration == "bf16":
        full_precision = float(point.get("cache_data_bytes") or 0)
    else:
        full_precision = float(point.get("residual_bytes") or 0) + float(
            point.get("sink_bytes") or 0
        )
    outlier = float(point.get("outlier_value_bytes") or 0) + float(
        point.get("outlier_index_bytes") or 0
    )
    return {
        "metadata_fraction": float(point.get("metadata_bytes") or 0) / allocated,
        "full_precision_fraction": full_precision / allocated,
        "outlier_fraction": outlier / allocated,
        "workspace_fraction": float(point.get("workspace_bytes") or 0) / allocated,
    }


def build_analysis_frame() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read every compact source input and produce one canonical table."""

    if validate_local_artifact(WALL_ROOT, environ={}).root_sha256 != WALL_ROOT_SHA256:
        raise Phase17Error("host-wall closure root differs")
    if validate_local_artifact(OUTER_ROOT, environ={}).root_sha256 != OUTER_ROOT_SHA256:
        raise Phase17Error("source outer root differs")
    raw = _read_parquet(WALL_ROOT / "raw_run_index.parquet")
    points = _read_parquet(WALL_ROOT / "point_summary.parquet")
    sources = _read_parquet(WALL_ROOT / "source_result_index.parquet")
    feasibility = _read_parquet(OUTER_ROOT / "feasibility.parquet")
    if len(raw) != EXPECTED_RECORDS or len(points) != EXPECTED_LOGICAL_POINTS:
        raise Phase17Error("source cardinality differs")
    statuses = Counter(str(row["status"]) for row in raw)
    if statuses != {"completed": EXPECTED_COMPLETED, "capacity_infeasible": EXPECTED_INFEASIBLE}:
        raise Phase17Error("source terminal statuses differ")
    if sum(row.get("replacement_of") is not None for row in raw) != EXPECTED_REPLACEMENTS:
        raise Phase17Error("replacement mapping differs")
    if len(sources) != EXPECTED_COMPLETED or len(feasibility) != EXPECTED_LOGICAL_POINTS:
        raise Phase17Error("source index cardinality differs")
    point_by_key = {
        (str(row["method_config_id"]), int(row["batch_size"]), int(row["historical_context"])): row
        for row in points
    }
    feasibility_by_key = {
        (str(row["method_config_id"]), int(row["batch_size"]), int(row["historical_context"])): row
        for row in feasibility
    }
    source_by_run = {str(row["run_id"]): row for row in sources}
    rows: list[dict[str, Any]] = []
    result_hash_failures: list[str] = []
    for source in raw:
        configuration = str(source["method_config_id"])
        key = (configuration, int(source["batch_size"]), int(source["historical_context"]))
        point = point_by_key[key]
        feasible = feasibility_by_key[key]
        completed = source["status"] == "completed"
        result: dict[str, Any] | None = None
        source_index: Mapping[str, Any] | None = None
        operation_fingerprint = logical_prefix_id = token_checksum = None
        if completed:
            source_index = source_by_run.get(str(source["run_id"]))
            if source_index is None:
                raise Phase17Error("completed run lacks source-result binding")
            result_path = RESULT_ROOT / f"{source['run_id']}.json"
            payload = result_path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != source_index["source_result_sha256"]:
                result_hash_failures.append(str(source["run_id"]))
            result = json.loads(payload)
            binding = result.get("phase16r_logical_prefix_binding")
            if not isinstance(binding, Mapping):
                raise Phase17Error("completed run lacks logical-prefix binding")
            operation_fingerprint = result.get("operation_fingerprint_sha256")
            logical_prefix_id = binding.get("logical_prefix_id")
            token_checksum = binding.get("token_checksum")
        features = _flatten_features(point)
        key_bits, value_bits = _bits(configuration)
        row = {
            "schema_version": "kvbench-phase17-analysis-row-1.0.0",
            "source_family_id": SOURCE_FAMILY_ID,
            "source_segment_root_sha256": (
                source_index.get("source_root_sha256") if source_index else None
            ),
            "source_result_sha256": (
                source_index.get("source_result_sha256") if source_index else None
            ),
            "run_id": source["run_id"],
            "logical_record_id": source["logical_record_id"],
            "replacement_of": source.get("replacement_of"),
            "method_family": _family(configuration),
            "method_config_id": configuration,
            "method_config_fingerprint": source["method_config_fingerprint"],
            "batch_size": int(source["batch_size"]),
            "context_label": int(source["context_label"]),
            "historical_context": int(source["historical_context"]),
            "total_attended_context": int(source["total_attended_context"]),
            "replicate_index": int(source["replicate_index"]),
            "segment_id": source["segment_id"],
            "seed": int(source["seed"]),
            "grid_source": source["grid_source"],
            "status": source["status"],
            "coverage_reason": source.get("reason"),
            "fitting_eligible": bool(source.get("fitting_eligible")) if completed else False,
            "response_host_wall_ms": (
                float(source["host_wall_process_median_ms"]) if completed else None
            ),
            "secondary_cuda_event_ms": (
                float(source["cuda_process_median_ms"]) if completed else None
            ),
            "latency_units": "milliseconds_per_full_batch_decode_step",
            "latency_basis": source.get("latency_basis"),
            "runner_kind": source["runner_kind"],
            "graph_mode": source["graph_mode"],
            "warmup_steps": int(source["warmup_steps"]),
            "measured_steps": int(source["measured_steps"]),
            "operation_fingerprint_sha256": operation_fingerprint,
            "logical_prefix_id": logical_prefix_id,
            "token_checksum": token_checksum,
            "output_checksum": source.get("output_checksum"),
            "kernel_path_fingerprint": source.get("kernel_path_fingerprint"),
            "allocation_fingerprint": source.get("allocation_fingerprint"),
            "key_bits": key_bits,
            "value_bits": value_bits,
            "logical_bf16_bytes": point.get("logical_bf16_bytes"),
            "allocated_bytes": point.get("allocated_bytes"),
            "active_storage_bytes": point.get("active_storage_bytes"),
            "metadata_bytes": point.get("metadata_bytes"),
            "residual_bytes": point.get("residual_bytes"),
            "sink_bytes": point.get("sink_bytes"),
            "outlier_value_bytes": point.get("outlier_value_bytes"),
            "outlier_index_bytes": point.get("outlier_index_bytes"),
            "workspace_bytes": point.get("workspace_bytes"),
            "rho_alloc": point.get("rho_alloc"),
            "r_alloc": point.get("r_alloc"),
            "r_nominal": point.get("r_nominal"),
            "kernel_count": point.get("kernel_count"),
            **features,
            "predicted_required_bytes": feasible["predicted_required_bytes"],
            "memory_limit_bytes": feasible["limit_bytes"],
            "r_hbm": None,
            "quality_status": "unvalidated",
            "performance_claim_eligible": False,
        }
        if completed and (
            row["latency_basis"] != "host_wall"
            or source.get("run_kind") != "timing"
            or source.get("quality_status") != "unvalidated"
            or source.get("r_hbm") is not None
        ):
            raise Phase17Error("completed source row violates timing boundary")
        rows.append(row)
    if result_hash_failures:
        raise Phase17Error("source result checksum differs")
    completed_keys = {
        (r["method_config_id"], r["batch_size"], r["historical_context"])
        for r in rows if r["status"] == "completed"
    }
    if len(completed_keys) != EXPECTED_FEASIBLE_POINTS:
        raise Phase17Error("feasible logical-point count differs")
    if any(
        sum(
            x["status"] == "completed" and
            (x["method_config_id"], x["batch_size"], x["historical_context"]) == key
            for x in rows
        ) != 5
        for key in completed_keys
    ):
        raise Phase17Error("completed logical point lacks five effective processes")
    summary = {
        "planned_records": len(rows),
        "completed_records": statuses["completed"],
        "capacity_infeasible_records": statuses["capacity_infeasible"],
        "logical_points": len(points),
        "feasible_logical_points": len(completed_keys),
        "infrastructure_replacements": EXPECTED_REPLACEMENTS,
        "source_result_checksum_failures": 0,
        "source_file_sha256": _source_hashes(),
    }
    return rows, summary


def build_split_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row["status"] == "completed"]
    folds: list[dict[str, Any]] = []

    def add_fold(
        protocol: str, fold_id: str, family: str, test_predicate: Any,
        domain: str, *, repeat_session: bool = False,
    ) -> None:
        family_rows = [row for row in completed if row["method_family"] == family]
        test = [row for row in family_rows if test_predicate(row)]
        train = [row for row in family_rows if not test_predicate(row)]
        train_groups = {_condition_key(row) for row in train}
        test_groups = {_condition_key(row) for row in test}
        if not repeat_session and train_groups & test_groups:
            raise Phase17Error("group leakage in split manifest")
        folds.append({
            "protocol": protocol,
            "fold_id": fold_id,
            "method_family": family,
            "domain_status": domain,
            "train_run_ids": sorted(str(row["run_id"]) for row in train),
            "test_run_ids": sorted(str(row["run_id"]) for row in test),
            "train_group_keys": sorted(train_groups),
            "test_group_keys": sorted(test_groups),
            "same_geometry_across_sides": repeat_session,
        })

    for batch in BATCHES:
        domain = "interior_interpolation" if batch in {2, 4, 8} else "edge_extrapolation"
        for family in FAMILIES:
            add_fold(
                "leave_one_batch_out", f"B{batch}", family,
                lambda row, value=batch: int(row["batch_size"]) == value, domain,
            )
    for family in COMPRESSED_FAMILIES:
        configs = [c for c in CONFIGURATIONS if _family(c) == family]
        for configuration in configs:
            add_fold(
                "leave_one_config_out", configuration, family,
                lambda row, value=configuration: row["method_config_id"] == value,
                "configuration_extrapolation",
            )
    for family in FAMILIES:
        add_fold(
            "leave_context_band_out", "L24576_49152", family,
            lambda row: CONTEXT_BAND[0] <= int(row["historical_context"]) <= CONTEXT_BAND[1],
            "interior_band_interpolation",
        )
    for replicate in REPLICATES:
        for family in FAMILIES:
            add_fold(
                "session_holdout", f"replicate-{replicate}", family,
                lambda row, value=replicate: int(row["replicate_index"]) == value,
                "repeat_session_same_geometry", repeat_session=True,
            )
    return {
        "schema_version": "kvbench-phase17-split-manifest-1.0.0",
        "source_family_id": SOURCE_FAMILY_ID,
        "group_key": ["method_config_id", "batch_size", "historical_context"],
        "context_band": {"minimum": CONTEXT_BAND[0], "maximum": CONTEXT_BAND[1]},
        "fold_count": len(folds),
        "folds": folds,
    }


def _standardize_fit(values: np.ndarray) -> tuple[np.ndarray, dict[str, list[float]]]:
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return (values - mean) / scale, {"mean": mean.tolist(), "scale": scale.tolist()}


def _standardize_apply(values: np.ndarray, state: Mapping[str, Any]) -> np.ndarray:
    return (values - np.asarray(state["mean"], dtype=float)) / np.asarray(
        state["scale"], dtype=float
    )


def _ridge_fit(design: np.ndarray, y: np.ndarray, penalty: float) -> np.ndarray:
    gram = design.T @ design
    ridge = np.eye(design.shape[1]) * penalty
    ridge[0, 0] = 0.0
    return np.linalg.solve(gram + ridge, design.T @ y)


def _raw_predictors(rows: Sequence[Mapping[str, Any]], model_id: str) -> np.ndarray:
    if model_id == "E":
        return np.asarray([[math.log(float(r["batch_size"]) * float(r["historical_context"]) / float(r["r_alloc"]))] for r in rows])
    base = [[
        math.log(float(r["batch_size"])),
        math.log(float(r["historical_context"])),
        math.log(float(r["r_alloc"])),
    ] for r in rows]
    if model_id in {"RQ2", "D"}:
        return np.asarray(base, dtype=float)
    shape = [[
        float(r["metadata_fraction"]), float(r["full_precision_fraction"]),
        float(r["outlier_fraction"]), float(r["workspace_fraction"]),
    ] for r in rows]
    values = [b + s for b, s in zip(base, shape)]
    if model_id == "F_diagnostic":
        values = [v + [math.log1p(float(r["kernel_count"]))] for v, r in zip(values, rows)]
    return np.asarray(values, dtype=float)


def _linear_design(standardized: np.ndarray, model_id: str) -> np.ndarray:
    if model_id == "E":
        z = standardized[:, 0]
        return np.column_stack([np.ones(len(z)), z, z * z])
    z1, z2, z3 = standardized[:, 0], standardized[:, 1], standardized[:, 2]
    return np.column_stack([
        np.ones(len(z1)), z1, z2, z3,
        z1 * z1, z2 * z2, z3 * z3, z1 * z2, z1 * z3, z2 * z3,
    ])


def fit_linear_predictive(rows: Sequence[Mapping[str, Any]], model_id: str) -> dict[str, Any]:
    raw = _raw_predictors(rows, model_id)
    standardized, scaler = _standardize_fit(raw)
    design = _linear_design(standardized, model_id)
    y = np.log(np.asarray([float(row["response_host_wall_ms"]) for row in rows]))
    coefficients = _ridge_fit(design, y, RIDGE)
    return {
        "model_id": model_id,
        "model_kind": "log_ridge",
        "scaler": scaler,
        "coefficients": coefficients.tolist(),
        "feature_count": int(design.shape[1]),
        "loss": "mean_squared_log_latency",
    }


def predict_linear_predictive(model: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    raw = _raw_predictors(rows, str(model["model_id"]))
    standardized = _standardize_apply(raw, model["scaler"])
    design = _linear_design(standardized, str(model["model_id"]))
    prediction = np.exp(design @ np.asarray(model["coefficients"], dtype=float))
    return np.maximum(prediction, 1e-9)


def _surface_raw(rows: Sequence[Mapping[str, Any]], model_id: str) -> np.ndarray:
    base = [[math.log(float(r["batch_size"])), math.log(float(r["r_alloc"]))] for r in rows]
    if model_id == "D":
        return np.asarray(base, dtype=float)
    shape = [[
        float(r["metadata_fraction"]), float(r["full_precision_fraction"]),
        float(r["outlier_fraction"]), float(r["workspace_fraction"]),
    ] for r in rows]
    values = [b + s for b, s in zip(base, shape)]
    if model_id == "F_diagnostic":
        values = [v + [math.log1p(float(r["kernel_count"]))] for v, r in zip(values, rows)]
    return np.asarray(values, dtype=float)


def _surface_basis(standardized: np.ndarray) -> np.ndarray:
    first = standardized[:, 0]
    second = standardized[:, 1]
    columns = [np.ones(len(first)), first, second, first * second]
    columns.extend(standardized[:, index] for index in range(2, standardized.shape[1]))
    return np.column_stack(columns)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-value))


def _surface_forward(
    basis: np.ndarray, historical: np.ndarray, parameters: np.ndarray
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    p = basis.shape[1]
    tau = np.exp(np.clip(basis @ parameters[:p], -10.0, 12.0))
    slope = np.exp(np.clip(basis @ parameters[p : 2 * p], -10.0, 12.0))
    sigmoid = _sigmoid(basis @ parameters[2 * p :])
    knee = 0.02 + 0.96 * sigmoid
    x = historical / L_SCALE
    active = x > knee
    hinge = np.maximum(x - knee, 0.0)
    prediction = tau + slope * hinge
    return prediction, (tau, slope, knee, hinge, active)


def _surface_loss_gradient(
    basis: np.ndarray,
    historical: np.ndarray,
    response: np.ndarray,
    parameters: np.ndarray,
) -> tuple[float, np.ndarray]:
    prediction, (tau, slope, knee, hinge, active) = _surface_forward(
        basis, historical, parameters
    )
    log_error = np.log(prediction) - np.log(response)
    loss = float(np.mean(log_error * log_error))
    d_prediction = 2.0 * log_error / (len(response) * prediction)
    p = basis.shape[1]
    gradient = np.zeros_like(parameters)
    gradient[:p] = basis.T @ (d_prediction * tau)
    gradient[p : 2 * p] = basis.T @ (d_prediction * slope * hinge)
    sigmoid = (knee - 0.02) / 0.96
    d_knee = 0.96 * sigmoid * (1.0 - sigmoid)
    gradient[2 * p :] = basis.T @ (
        d_prediction * (-slope * active.astype(float)) * d_knee
    )
    mask = np.ones_like(parameters)
    mask[0] = mask[p] = mask[2 * p] = 0.0
    loss += SURFACE_RIDGE * float(np.sum((parameters * mask) ** 2))
    gradient += 2.0 * SURFACE_RIDGE * parameters * mask
    return loss, gradient


def fit_surface_predictive(rows: Sequence[Mapping[str, Any]], model_id: str) -> dict[str, Any]:
    raw = _surface_raw(rows, model_id)
    standardized, scaler = _standardize_fit(raw)
    basis = _surface_basis(standardized)
    historical = np.asarray([float(row["historical_context"]) for row in rows])
    response = np.asarray([float(row["response_host_wall_ms"]) for row in rows])
    p = basis.shape[1]
    best: tuple[float, np.ndarray] | None = None
    for start_index, knee_start in enumerate(SURFACE_STARTS):
        parameters = np.zeros(3 * p, dtype=float)
        parameters[0] = math.log(max(float(np.quantile(response, 0.15)), 1e-6))
        parameters[p] = math.log(max(float(np.quantile(response, 0.85) - np.quantile(response, 0.15)), 1.0))
        normalized = (knee_start - 0.02) / 0.96
        parameters[2 * p] = math.log(normalized / (1.0 - normalized))
        first = np.zeros_like(parameters)
        second = np.zeros_like(parameters)
        for iteration in range(1, SURFACE_ITERATIONS + 1):
            loss, gradient = _surface_loss_gradient(
                basis, historical, response, parameters
            )
            norm = float(np.linalg.norm(gradient))
            if norm > 10.0:
                gradient *= 10.0 / norm
            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * (gradient * gradient)
            first_hat = first / (1.0 - 0.9**iteration)
            second_hat = second / (1.0 - 0.999**iteration)
            parameters -= SURFACE_LR * first_hat / (np.sqrt(second_hat) + 1e-8)
        final_loss, _ = _surface_loss_gradient(
            basis, historical, response, parameters
        )
        candidate = (final_loss, parameters.copy())
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    return {
        "model_id": model_id,
        "model_kind": "positive_knee_surface",
        "scaler": scaler,
        "basis": ["1", "logB", "logr", "logB_x_logr"] + (
            ["metadata_fraction", "full_precision_fraction", "outlier_fraction", "workspace_fraction"]
            if model_id != "D" else []
        ) + (["log1p_kernel_count"] if model_id == "F_diagnostic" else []),
        "parameters": best[1].tolist(),
        "parameter_count": int(len(best[1])),
        "training_loss": float(best[0]),
        "loss": "mean_squared_log_latency",
        "lambda_bounds_historical_tokens": [0.02 * L_SCALE, 0.98 * L_SCALE],
    }


def predict_surface_predictive(model: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    raw = _surface_raw(rows, str(model["model_id"]))
    standardized = _standardize_apply(raw, model["scaler"])
    basis = _surface_basis(standardized)
    historical = np.asarray([float(row["historical_context"]) for row in rows])
    prediction, _ = _surface_forward(
        basis, historical, np.asarray(model["parameters"], dtype=float)
    )
    return np.maximum(prediction, 1e-9)


def fit_predictive(rows: Sequence[Mapping[str, Any]], model_id: str) -> dict[str, Any]:
    if model_id in {"E", "RQ2"}:
        return fit_linear_predictive(rows, model_id)
    return fit_surface_predictive(rows, model_id)


def predict_predictive(model: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if model["model_kind"] == "log_ridge":
        return predict_linear_predictive(model, rows)
    return predict_surface_predictive(model, rows)


def _ols(design: np.ndarray, response: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    prediction = design @ coefficients
    residual = response - prediction
    return coefficients, float(residual @ residual), residual


def _bic(sse: float, n: int, parameters: int) -> float:
    return n * math.log(max(sse / n, 1e-18)) + parameters * math.log(n)


def local_curve_fit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len({int(row["historical_context"]) for row in rows}) < 4:
        return {"fit_status": "insufficient_feasible_span"}
    ordered = sorted(rows, key=lambda row: (int(row["historical_context"]), int(row["replicate_index"])))
    x = np.asarray([float(row["historical_context"]) for row in ordered])
    y = np.asarray([float(row["response_host_wall_ms"]) for row in ordered])
    unique = sorted(set(x.tolist()))
    candidates = sorted(set(unique + [(a + b) / 2.0 for a, b in zip(unique, unique[1:])]))
    constant_prediction = np.full(len(y), y.mean())
    constant_sse = float(np.sum((y - constant_prediction) ** 2))
    a_coef, a_sse, a_residual = _ols(np.column_stack([np.ones(len(x)), x]), y)
    best_b: tuple[float, float, np.ndarray, np.ndarray] | None = None
    best_c: tuple[float, float, np.ndarray, np.ndarray] | None = None
    for knee in candidates:
        hinge = np.maximum(x - knee, 0.0)
        b_coef, b_sse, b_residual = _ols(
            np.column_stack([np.ones(len(x)), x, hinge]), y
        )
        b_candidate = (b_sse, knee, b_coef, b_residual)
        if best_b is None or b_candidate[0] < best_b[0]:
            best_b = b_candidate
        c_coef, c_sse, c_residual = _ols(
            np.column_stack([np.ones(len(x)), hinge]), y
        )
        if c_coef[1] < 0:
            c_coef = np.asarray([y.mean(), 0.0])
            c_residual = y - c_coef[0]
            c_sse = float(c_residual @ c_residual)
        c_candidate = (c_sse, knee, c_coef, c_residual)
        if best_c is None or c_candidate[0] < best_c[0]:
            best_c = c_candidate
    assert best_b is not None and best_c is not None
    models = {
        "constant": {"sse": constant_sse, "bic": _bic(constant_sse, len(y), 1), "parameters": {"tau": float(y.mean())}},
        "A": {"sse": a_sse, "bic": _bic(a_sse, len(y), 2), "parameters": {"alpha": float(a_coef[0]), "beta": float(a_coef[1])}},
        "B": {"sse": best_b[0], "bic": _bic(best_b[0], len(y), 4), "parameters": {"alpha": float(best_b[2][0]), "beta1": float(best_b[2][1]), "beta2": float(best_b[2][2]), "knee": float(best_b[1])}},
        "C": {"sse": best_c[0], "bic": _bic(best_c[0], len(y), 3), "parameters": {"tau": float(best_c[2][0]), "slope": float(best_c[2][1]), "knee": float(best_c[1])}},
    }
    winner = min(models, key=lambda key: (float(models[key]["bic"]), key))
    status = "linear_or_constant_preferred"
    knee: float | None = None
    if winner in {"B", "C"}:
        knee = float(models[winner]["parameters"]["knee"])
        below = len({value for value in unique if value < knee})
        above = len({value for value in unique if value > knee})
        delta = float(models["A"]["bic"]) - float(models[winner]["bic"])
        if knee <= min(unique):
            status = "below_observed_range"
        elif knee >= max(unique):
            status = "above_observed_range"
        elif below < 2 or above < 2 or delta < 2.0:
            status = "weakly_identified"
        else:
            status = "identified_in_range"
    return {
        "fit_status": status,
        "winner": winner,
        "knee": knee,
        "minimum_context": int(min(unique)),
        "maximum_context": int(max(unique)),
        "unique_contexts": len(unique),
        "models_json": json.dumps(models, sort_keys=True, separators=(",", ":")),
        "linear_bic": float(models["A"]["bic"]),
        "winner_bic": float(models[winner]["bic"]),
        "bic_improvement_over_linear": float(models["A"]["bic"] - models[winner]["bic"]),
    }


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=float), quantile))


def local_knee_estimates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    completed = [row for row in rows if row["status"] == "completed"]
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in completed:
        grouped[(str(row["method_config_id"]), int(row["batch_size"]))].append(row)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    shared_draws = generator.integers(0, 5, size=(BOOTSTRAP_DRAWS, 5))
    estimates: list[dict[str, Any]] = []
    for configuration in CONFIGURATIONS:
        for batch in BATCHES:
            group = grouped[(configuration, batch)]
            fit = local_curve_fit(group)
            valid: list[float] = []
            failed = 0
            if fit.get("fit_status") == "identified_in_range":
                by_context: dict[int, dict[int, Mapping[str, Any]]] = defaultdict(dict)
                for row in group:
                    by_context[int(row["historical_context"])][int(row["replicate_index"])] = row
                for draw in shared_draws:
                    sample: list[Mapping[str, Any]] = []
                    for context_rows in by_context.values():
                        sample.extend(context_rows[int(index)] for index in draw)
                    bootstrap_fit = local_curve_fit(sample)
                    knee = bootstrap_fit.get("knee")
                    if bootstrap_fit.get("fit_status") == "identified_in_range" and isinstance(knee, (int, float)):
                        valid.append(float(knee))
                    else:
                        failed += 1
            estimable = len(valid) >= 500
            status = str(fit.get("fit_status"))
            if status == "identified_in_range" and not estimable:
                status = "weakly_identified"
            estimates.append({
                "method_config_id": configuration,
                "method_family": _family(configuration),
                "batch_size": batch,
                **fit,
                "fit_status": status,
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "bootstrap_valid_draws": len(valid),
                "bootstrap_non_identifiable_or_failed_draws": BOOTSTRAP_DRAWS - len(valid),
                "bootstrap_estimable": estimable,
                "knee_lower_95": _percentile(valid, 0.025) if estimable else None,
                "knee_upper_95": _percentile(valid, 0.975) if estimable else None,
                "uncertainty_scope": "conditional_on_five_sampled_replicate_segments",
                "knee_reference_status": "not_evaluable_no_independent_reference",
            })
    return estimates


def evaluate_holdouts(
    rows: Sequence[Mapping[str, Any]], split_manifest: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit every frozen candidate in every fold and return auditable OOF rows."""

    completed = {str(row["run_id"]): row for row in rows if row["status"] == "completed"}
    predictions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    fitted: list[dict[str, Any]] = []
    for fold in split_manifest["folds"]:
        train = [completed[run_id] for run_id in fold["train_run_ids"]]
        test = [completed[run_id] for run_id in fold["test_run_ids"]]
        if not train or not test:
            for model_id in PREDICTIVE_MODELS:
                failures.append({
                    "protocol": fold["protocol"], "fold_id": fold["fold_id"],
                    "method_family": fold["method_family"], "model_id": model_id,
                    "reason": "empty_train_or_test", "missing_predictions": len(test),
                })
            continue
        for model_id in PREDICTIVE_MODELS:
            try:
                model = fit_predictive(train, model_id)
                values = predict_predictive(model, test)
                if len(values) != len(test) or not np.all(np.isfinite(values)):
                    raise Phase17Error("non-finite or incomplete prediction")
                fitted.append({
                    "protocol": fold["protocol"], "fold_id": fold["fold_id"],
                    "method_family": fold["method_family"], "model_id": model_id,
                    "train_rows": len(train), "test_rows": len(test),
                    "feature_count": model.get("feature_count", model.get("parameter_count")),
                    "training_loss": model.get("training_loss"),
                    "train_scaler_json": json.dumps(model["scaler"], sort_keys=True, separators=(",", ":")),
                })
                for source, predicted in zip(test, values):
                    observed = float(source["response_host_wall_ms"])
                    predictions.append({
                        "schema_version": "kvbench-phase17-oof-prediction-1.0.0",
                        "protocol": fold["protocol"], "fold_id": fold["fold_id"],
                        "domain_status": fold["domain_status"],
                        "model_id": model_id, "method_family": source["method_family"],
                        "method_config_id": source["method_config_id"],
                        "batch_size": int(source["batch_size"]),
                        "context_label": int(source["context_label"]),
                        "historical_context": int(source["historical_context"]),
                        "replicate_index": int(source["replicate_index"]),
                        "grid_source": source["grid_source"], "run_id": source["run_id"],
                        "observed_host_wall_ms": observed,
                        "predicted_host_wall_ms": float(predicted),
                        "absolute_error_ms": abs(float(predicted) - observed),
                        "absolute_relative_error": abs(float(predicted) - observed) / observed,
                        "quality_status": "unvalidated",
                        "performance_claim_eligible": False,
                    })
            except (ArithmeticError, FloatingPointError, Phase17Error, ValueError, np.linalg.LinAlgError) as error:
                failures.append({
                    "protocol": fold["protocol"], "fold_id": fold["fold_id"],
                    "method_family": fold["method_family"], "model_id": model_id,
                    "reason": f"fit_failed:{type(error).__name__}",
                    "missing_predictions": len(test),
                })
    return predictions, failures, fitted


def _metric_values(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "scored_rows": 0, "mae_ms": None, "mape": None,
            "median_absolute_relative_error": None, "p95_relative_error": None,
        }
    absolute = np.asarray([float(row["absolute_error_ms"]) for row in rows])
    relative = np.asarray([float(row["absolute_relative_error"]) for row in rows])
    return {
        "scored_rows": len(rows), "mae_ms": float(absolute.mean()),
        "mape": float(relative.mean()),
        "median_absolute_relative_error": float(np.median(relative)),
        "p95_relative_error": float(np.quantile(relative, 0.95)),
    }


def collapse_oof_to_logical(predictions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        key = (
            row["protocol"], row["fold_id"], row["model_id"], row["method_family"],
            row["method_config_id"], row["batch_size"], row["historical_context"],
            row["grid_source"], row["domain_status"],
        )
        grouped[key].append(row)
    result: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        observed = float(np.median([float(row["observed_host_wall_ms"]) for row in values]))
        predicted = float(np.median([float(row["predicted_host_wall_ms"]) for row in values]))
        result.append({
            "protocol": key[0], "fold_id": key[1], "model_id": key[2],
            "method_family": key[3], "method_config_id": key[4],
            "batch_size": int(key[5]), "historical_context": int(key[6]),
            "grid_source": key[7], "domain_status": key[8],
            "observed_host_wall_ms": observed, "predicted_host_wall_ms": predicted,
            "absolute_error_ms": abs(predicted - observed),
            "absolute_relative_error": abs(predicted - observed) / observed,
            "process_rows": len(values),
        })
    return result


def summarize_models(
    predictions: Sequence[Mapping[str, Any]], failures: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    logical = collapse_oof_to_logical(predictions)
    comparison: list[dict[str, Any]] = []
    for protocol in (*GEOMETRY_PROTOCOLS, "session_holdout"):
        source = logical if protocol in GEOMETRY_PROTOCOLS else list(predictions)
        for model_id in PREDICTIVE_MODELS:
            for family in FAMILIES:
                selected = [
                    row for row in source
                    if row["protocol"] == protocol and row["model_id"] == model_id
                    and row["method_family"] == family
                ]
                domains = sorted({str(row["domain_status"]) for row in selected})
                row = {
                    "protocol": protocol, "model_id": model_id,
                    "method_family": family, "subset": "combined",
                    "fold_count": len({str(value["fold_id"]) for value in selected}),
                    "missing_prediction_count": sum(
                        int(value["missing_predictions"]) for value in failures
                        if value["protocol"] == protocol and value["model_id"] == model_id
                        and value["method_family"] == family
                    ),
                    "domain_status_json": json.dumps(domains, separators=(",", ":")),
                    "aic": None, "bic": None,
                    **_metric_values(selected),
                }
                comparison.append(row)
                for subset in ("base", "adaptive"):
                    subset_rows = [value for value in selected if value["grid_source"] == subset]
                    comparison.append({**row, "subset": subset, **_metric_values(subset_rows)})

    per_batch: list[dict[str, Any]] = []
    for protocol in (*GEOMETRY_PROTOCOLS, "session_holdout"):
        source = logical if protocol in GEOMETRY_PROTOCOLS else list(predictions)
        for model_id in PREDICTIVE_MODELS:
            for family in FAMILIES:
                for batch in BATCHES:
                    selected = [r for r in source if r["protocol"] == protocol and r["model_id"] == model_id and r["method_family"] == family and int(r["batch_size"]) == batch]
                    if selected:
                        per_batch.append({"protocol": protocol, "model_id": model_id, "method_family": family, "batch_size": batch, **_metric_values(selected)})
    return comparison, {"per_batch": per_batch, "logical_oof_rows": len(logical)}


def ratio_and_ranking_metrics(
    predictions: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    logical = collapse_oof_to_logical(predictions)
    ratio_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    for protocol in GEOMETRY_PROTOCOLS:
        if protocol == "leave_one_config_out":
            for model_id in PREDICTIVE_MODELS:
                ratio_rows.append({"protocol": protocol, "model_id": model_id, "status": "not_evaluable_no_out_of_fold_bf16_baseline", "eligible": 0, "correct": 0, "accuracy": None})
            continue
        for model_id in PREDICTIVE_MODELS:
            selected = [r for r in logical if r["protocol"] == protocol and r["model_id"] == model_id]
            by_key = {(r["method_config_id"], r["batch_size"], r["historical_context"]): r for r in selected}
            eligible = correct = 0
            for row in selected:
                if row["method_config_id"] == "bf16":
                    continue
                baseline = by_key.get(("bf16", row["batch_size"], row["historical_context"]))
                if baseline is None:
                    continue
                observed_ratio = float(baseline["observed_host_wall_ms"]) / float(row["observed_host_wall_ms"])
                predicted_ratio = float(baseline["predicted_host_wall_ms"]) / float(row["predicted_host_wall_ms"])
                eligible += 1
                correct += (observed_ratio >= 1.0) == (predicted_ratio >= 1.0)
            ratio_rows.append({"protocol": protocol, "model_id": model_id, "status": "evaluated", "eligible": eligible, "correct": correct, "accuracy": correct / eligible if eligible else None, "ratio_kind": "fully_predicted_same_work"})

            by_family_geometry: dict[tuple[str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
            for row in selected:
                if row["method_family"] != "bf16":
                    by_family_geometry[(str(row["method_family"]), int(row["batch_size"]), int(row["historical_context"]))].append(row)
            pair_count = pair_correct = near_count = near_correct = 0
            for values in by_family_geometry.values():
                for index, left in enumerate(values):
                    for right in values[index + 1:]:
                        observed_left, observed_right = float(left["observed_host_wall_ms"]), float(right["observed_host_wall_ms"])
                        predicted_left, predicted_right = float(left["predicted_host_wall_ms"]), float(right["predicted_host_wall_ms"])
                        pair_count += 1
                        is_correct = (observed_left <= observed_right) == (predicted_left <= predicted_right)
                        pair_correct += is_correct
                        near = abs(observed_left - observed_right) / min(observed_left, observed_right) <= 0.01
                        if near:
                            near_count += 1
                            near_correct += is_correct
            ranking_rows.append({"protocol": protocol, "model_id": model_id, "eligible_pairs": pair_count, "correct_pairs": pair_correct, "accuracy": pair_correct / pair_count if pair_count else None, "near_tie_pairs": near_count, "near_tie_correct": near_correct, "near_ties_retained": True})
    return {"speedup_sign": ratio_rows, "pairwise_ranking": ranking_rows}


def choose_models(comparison: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def score(model_id: str) -> tuple[float, float, int]:
        rows = [r for r in comparison if r["subset"] == "combined" and r["protocol"] in GEOMETRY_PROTOCOLS and r["model_id"] == model_id and r["scored_rows"]]
        median = float(np.mean([float(r["median_absolute_relative_error"]) for r in rows]))
        p95 = float(np.mean([float(r["p95_relative_error"]) for r in rows]))
        complexity = {"E": 3, "RQ2": 10, "D": 12, "F_shape": 24, "F_diagnostic": 27}[model_id]
        return median, p95, complexity
    scores = {model_id: score(model_id) for model_id in PREDICTIVE_MODELS}
    scientific = min(("E", "RQ2", "D", "F_shape"), key=lambda value: scores[value])
    deployment = min(DEPLOYABLE_MODELS, key=lambda value: scores[value])
    return {
        "scientific_selected_model": scientific,
        "deployment_selected_model": deployment,
        "selection_scores": {key: {"macro_median_relative_error": value[0], "macro_p95_relative_error": value[1], "complexity": value[2]} for key, value in scores.items()},
        "outer_selection_score_unbiased": False,
    }


def fit_final_models(
    rows: Sequence[Mapping[str, Any]], selection: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]], models_directory: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    completed = [row for row in rows if row["status"] == "completed"]
    models_directory.mkdir(parents=True, exist_ok=True)
    index: dict[str, Any] = {
        "schema_version": "kvbench-phase17-model-index-1.0.0",
        "deployment_model_id": selection["deployment_selected_model"],
        "scientific_model_id": selection["scientific_selected_model"],
        "latency_units": "milliseconds_per_full_batch_decode_step",
        "quality_status": "unvalidated", "performance_claim_eligible": False,
        "families": {},
    }
    exported: list[dict[str, Any]] = []
    for family in FAMILIES:
        family_rows = [row for row in completed if row["method_family"] == family]
        family_entry: dict[str, Any] = {}
        for model_id in PREDICTIVE_MODELS:
            model = fit_predictive(family_rows, model_id)
            model.update({
                "schema_version": "kvbench-phase17-exported-model-1.0.0",
                "method_family": family,
                "training_rows": len(family_rows),
                "training_configurations": sorted({str(r["method_config_id"]) for r in family_rows}),
                "training_domain": {
                    "batch_min": min(int(r["batch_size"]) for r in family_rows),
                    "batch_max": max(int(r["batch_size"]) for r in family_rows),
                    "historical_context_min": min(int(r["historical_context"]) for r in family_rows),
                    "historical_context_max": max(int(r["historical_context"]) for r in family_rows),
                    "r_alloc_min": min(float(r["r_alloc"]) for r in family_rows),
                    "r_alloc_max": max(float(r["r_alloc"]) for r in family_rows),
                },
                "feature_requirements": (
                    ["B", "L", "r_alloc"] if model_id in {"E", "RQ2", "D"}
                    else ["B", "L", "r_alloc", "metadata_fraction", "full_precision_fraction", "outlier_fraction", "workspace_fraction"] + (["observed_kernel_count"] if model_id == "F_diagnostic" else [])
                ),
                "deployable_without_observed_runtime_features": model_id != "F_diagnostic",
            })
            residuals = [
                math.log(float(r["observed_host_wall_ms"]) / float(r["predicted_host_wall_ms"]))
                for r in predictions if r["model_id"] == model_id and r["method_family"] == family and r["protocol"] in GEOMETRY_PROTOCOLS
            ]
            model["new_process_log_residual_interval_95"] = [float(np.quantile(residuals, 0.025)), float(np.quantile(residuals, 0.975))] if residuals else None
            filename = f"{family}-{model_id}.json"
            write_exclusive(models_directory / filename, json_bytes(model))
            family_entry[model_id] = filename
            exported.append(model)
        index["families"][family] = family_entry
    write_exclusive(models_directory / "index.json", json_bytes(index))
    return index, exported


def knee_scaling_summary(knees: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    identified = [r for r in knees if r["fit_status"] == "identified_in_range" and r.get("knee") is not None]
    r_lookup: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["status"] == "completed":
            r_lookup[(str(row["method_config_id"]), int(row["batch_size"]))].append(row)
    bl: list[float] = []
    brho_l: list[float] = []
    for knee in identified:
        value = float(knee["knee"])
        batch = int(knee["batch_size"])
        candidates = r_lookup[(str(knee["method_config_id"]), batch)]
        closest = min(candidates, key=lambda row: abs(float(row["historical_context"]) - value))
        bl.append(batch * value)
        brho_l.append(batch * (1.0 / float(closest["r_alloc"])) * value)
    def cv(values: Sequence[float]) -> float | None:
        return float(np.std(values, ddof=1) / np.mean(values)) if len(values) > 1 else None
    return {"identified_knees": len(identified), "B_times_Lstar_cv": cv(bl), "B_times_rho_alloc_times_Lstar_cv": cv(brho_l), "status": "descriptive_hypothesis_only"}


def render_report(payload: Mapping[str, Any]) -> str:
    selected = payload["selection"]
    targets = payload["targets"]
    knees = payload["knees"]
    return f"""# Phase 17 Modeling Report

Status: **{payload['status']}**

The CPU-only analysis used the immutable Phase 16R host-wall dataset: {payload['source']['completed_records']} accepted process observations at {payload['source']['feasible_logical_points']} feasible logical points. The {payload['source']['infrastructure_replacements']} infrastructure replacements contribute exactly one accepted observation per logical replicate slot; infeasible records remain coverage evidence and were not assigned latency.

## Predictive result

The preregistered candidates were E (scalar byte law), RQ2 (B/L/r surface), D (positive knee surface), F_shape (D plus static byte-shape features), and F_diagnostic (plus observed kernel count, diagnostic only). Scientific selection chose **{selected['scientific_selected_model']}**; the offline deployment default chose **{selected['deployment_selected_model']}**. Selection used the declared outer results, so its selected score is not an unbiased estimate of a further model-selection procedure.

- Median relative-error target: **{targets['median_relative_error']['status']}** ({targets['median_relative_error']['value']:.6f}).
- P95 relative-error target: **{targets['p95_relative_error']['status']}** ({targets['p95_relative_error']['value']:.6f}).
- Same-work speedup-sign target: **{targets['speedup_sign_accuracy']['status']}** ({targets['speedup_sign_accuracy']['value_text']}).
- Pairwise method-ranking target: **{targets['method_ranking_accuracy']['status']}** ({targets['method_ranking_accuracy']['value_text']}).
- Knee-relative-error target: **not_evaluable**; no independent knee reference exists.

The comparison of E, RQ2, D, and F_shape is reported without changing thresholds or adding experiments. F_diagnostic is not deployable because kernel count is an observed runtime feature. Phase 15 traffic features remain restricted to their one common profiler point and normal timing rows retain `r_hbm=null`.

## Knee uncertainty

Across 50 configuration/batch curves, {knees['identified']} were identified in range and {knees['non_identified']} were non-identifiable or preferred linear/constant descriptions. Intervals use 1,000 shared replicate-segment bootstrap draws and are conditional on the five sampled sessions. Pilot knees were not treated as truth.

## Scope

Predictions are host-wall milliseconds per full-batch decode step for this model, GPU, container, graph path, and admitted configurations. They are not quality-preserving performance claims and do not validate arbitrary ratios, configurations, checkpoints, GPUs, or serving systems. The Phase 14 negative pure-launch-floor result remains unchanged. Quality is **LOCKED**.
"""


def _payload_paths(root: Path, excluded: set[str]) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.relative_to(root).as_posix() not in excluded)


def seal_bundle(stage: Path, bundle_id: str, status: str) -> Path:
    write_exclusive(stage / "manifest.json", json_bytes({
        "schema_version": "kvbench-phase17-artifact-manifest-1.0.0",
        "run_id": bundle_id, "status": status, "created_at_utc": _utc_now(),
        "append_only": True, "complete_written_last": True,
        "source_family_id": SOURCE_FAMILY_ID,
        "source_host_wall_root_sha256": WALL_ROOT_SHA256,
        "source_outer_root_sha256": OUTER_ROOT_SHA256,
        "quality_status": "unvalidated", "performance_claim_eligible": False,
    }))
    items = [{"path": path.relative_to(stage).as_posix(), "role": "phase17_modeling_evidence", "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in _payload_paths(stage, {"artifact_inventory.json", "checksums.sha256", "COMPLETE"})]
    write_exclusive(stage / "artifact_inventory.json", json_bytes({"schema_version": "kvbench-artifact-inventory-1.0.0", "run_id": bundle_id, "files": items, "excluded_control_files": ["artifact_inventory.json", "checksums.sha256", "COMPLETE"]}))
    ledger = "".join(f"{sha256_file(path)}  {path.relative_to(stage).as_posix()}\n" for path in _payload_paths(stage, {"checksums.sha256", "COMPLETE"})).encode()
    write_exclusive(stage / "checksums.sha256", ledger)
    write_exclusive(stage / "COMPLETE", json_bytes({
        "schema_version": "kvbench-completion-1.0.0", "run_id": bundle_id,
        "status": status, "manifest_sha256": sha256_file(stage / "manifest.json"),
        "artifact_inventory_sha256": sha256_file(stage / "artifact_inventory.json"),
        "checksum_ledger_path": "checksums.sha256",
        "checksum_ledger_sha256": sha256_file(stage / "checksums.sha256"),
        "written_last": True,
    }))
    final = ARTIFACT_ROOT / bundle_id
    rename_noreplace(stage, final)
    for path in sorted(final.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    final.chmod(0o555)
    artifact = validate_local_artifact(final, environ={})
    return final


def _write_svg(path: Path, title: str, values: Sequence[tuple[str, float]]) -> None:
    width, height = 900, 480
    maximum = max((value for _, value in values), default=1.0) or 1.0
    bars = []
    for index, (label, value) in enumerate(values):
        x = 50 + index * max(1, 800 // max(1, len(values)))
        bar_width = max(4, 650 // max(1, len(values)))
        bar_height = 330 * value / maximum
        bars.append(f'<rect x="{x}" y="{410-bar_height:.2f}" width="{bar_width}" height="{bar_height:.2f}" fill="#356aa0"/><text x="{x}" y="430" font-size="10" transform="rotate(30 {x} 430)">{label}</text>')
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/><text x="30" y="30" font-size="20">{title}</text><line x1="40" y1="410" x2="860" y2="410" stroke="black"/>{"".join(bars)}</svg>\n'
    write_exclusive(path, svg.encode())


def run_analysis(bundle_id: str | None = None) -> Path:
    starting_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True).strip()
    identifier = bundle_id or f"phase17-{datetime.now(timezone.utc).strftime('%Y%m%dt%H%M%S%f')[:22]}z-{starting_head[:8]}-{secrets.token_hex(3)}"
    if _BUNDLE_RE.fullmatch(identifier) is None:
        raise Phase17Error("invalid Phase 17 bundle ID")
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    stage = ARTIFACT_ROOT / f".{identifier}.{secrets.token_hex(8)}.staging"
    if stage.exists() or (ARTIFACT_ROOT / identifier).exists():
        raise Phase17Error("bundle ID already exists")
    stage.mkdir(mode=0o700)
    rows, source_summary = build_analysis_frame()
    splits = build_split_manifest(rows)
    predictions, failures, fitted = evaluate_holdouts(rows, splits)
    comparison, metric_details = summarize_models(predictions, failures)
    ratio_ranking = ratio_and_ranking_metrics(predictions)
    selection = choose_models(comparison)
    knees = local_knee_estimates(rows)
    scaling = knee_scaling_summary(knees, rows)

    selected = selection["scientific_selected_model"]
    selected_rows = [r for r in comparison if r["model_id"] == selected and r["protocol"] in GEOMETRY_PROTOCOLS and r["subset"] == "combined" and r["scored_rows"]]
    macro_median = float(np.mean([float(r["median_absolute_relative_error"]) for r in selected_rows]))
    macro_p95 = float(np.mean([float(r["p95_relative_error"]) for r in selected_rows]))
    sign_rows = [r for r in ratio_ranking["speedup_sign"] if r["model_id"] == selected and r["accuracy"] is not None]
    sign_eligible = sum(int(r["eligible"]) for r in sign_rows)
    sign_correct = sum(int(r["correct"]) for r in sign_rows)
    sign_accuracy = sign_correct / sign_eligible if sign_eligible else None
    rank_rows = [r for r in ratio_ranking["pairwise_ranking"] if r["model_id"] == selected and r["accuracy"] is not None]
    rank_eligible = sum(int(r["eligible_pairs"]) for r in rank_rows)
    rank_correct = sum(int(r["correct_pairs"]) for r in rank_rows)
    rank_accuracy = rank_correct / rank_eligible if rank_eligible else None
    targets = {
        "median_relative_error": {"threshold": 0.05, "value": macro_median, "status": "met" if macro_median <= 0.05 else "missed"},
        "p95_relative_error": {"threshold": 0.10, "value": macro_p95, "status": "met" if macro_p95 <= 0.10 else "missed"},
        "speedup_sign_accuracy": {"threshold": 0.95, "value": sign_accuracy, "value_text": "not_evaluable" if sign_accuracy is None else f"{sign_accuracy:.6f} ({sign_correct}/{sign_eligible})", "status": "not_evaluable" if sign_accuracy is None else ("met" if sign_accuracy >= 0.95 else "missed")},
        "method_ranking_accuracy": {"threshold": 0.90, "value": rank_accuracy, "value_text": "not_evaluable" if rank_accuracy is None else f"{rank_accuracy:.6f} ({rank_correct}/{rank_eligible})", "status": "not_evaluable" if rank_accuracy is None else ("met" if rank_accuracy >= 0.90 else "missed")},
        "knee_relative_error": {"threshold": 0.10, "value": None, "status": "not_evaluable_no_independent_reference"},
    }
    knee_counts = Counter(str(row["fit_status"]) for row in knees)
    payload = {
        "schema_version": "kvbench-phase17-modeling-result-1.0.0",
        "status": "PASS", "bundle_id": identifier, "starting_head": starting_head,
        "execution_head": starting_head, "source": source_summary,
        "source_roots": {"host_wall": WALL_ROOT_SHA256, "outer": OUTER_ROOT_SHA256, "phase15": PHASE15_ROOT_SHA256},
        "selection": selection, "targets": targets,
        "holdout_failures": failures, "fitted_fold_count": len(fitted),
        "metrics": metric_details, "ratio_and_ranking": ratio_ranking,
        "knees": {"status_counts": dict(sorted(knee_counts.items())), "identified": knee_counts["identified_in_range"], "non_identified": len(knees) - knee_counts["identified_in_range"], "scaling_hypotheses": scaling},
        "quality_state": "LOCKED", "performance_data_frozen": "absent",
        "phase18_started": False,
    }
    write_exclusive(stage / "input_manifest.json", json_bytes({"schema_version": "kvbench-phase17-input-manifest-1.0.0", "source_family_id": SOURCE_FAMILY_ID, "source_roots": payload["source_roots"], "source_file_sha256": source_summary["source_file_sha256"], "accepted_process_rows": EXPECTED_COMPLETED, "coverage_rows": EXPECTED_RECORDS}))
    _write_parquet(stage / "analysis_frame.parquet", rows)
    write_exclusive(stage / "split_manifest.json", json_bytes(splits))
    write_exclusive(stage / "candidate_spec.json", PLAN_PATH.read_bytes())
    _write_parquet(stage / "out_of_fold_predictions.parquet", predictions)
    _write_parquet(stage / "model_comparison.parquet", comparison)
    _write_parquet(stage / "knee_estimates.parquet", knees)
    write_exclusive(stage / "prediction_metrics.json", json_bytes({"schema_version": "kvbench-phase17-prediction-metrics-1.0.0", "comparison_rows": len(comparison), "fit_records": fitted, **metric_details, **ratio_ranking}))
    write_exclusive(stage / "model_target_status.json", json_bytes({"schema_version": "kvbench-phase17-model-target-status-1.0.0", "selection": selection, "targets": targets, "scientific_interpretation": "candidate comparison is descriptive and outer selection score is not unbiased", "phase14_pure_launch_floor_result": "unsupported"}))
    model_index, exported = fit_final_models(rows, selection, predictions, stage / "models")
    deployment_model = selection["deployment_selected_model"]
    example_row = next(r for r in rows if r["status"] == "completed" and r["method_config_id"] == "kvq4" and r["batch_size"] == 1 and r["historical_context"] == 4096)
    model_file = model_index["families"]["kvquant"][deployment_model]
    example_model = next(m for m in exported if m["method_family"] == "kvquant" and m["model_id"] == deployment_model)
    example_prediction = float(predict_predictive(example_model, [example_row])[0])
    write_exclusive(stage / "prediction_example.json", json_bytes({"method": "kvquant", "method_config_id": "kvq4", "batch_size": 1, "historical_context": 4096, "r_alloc": example_row["r_alloc"], "model_id": deployment_model, "model_file": model_file, "predicted_host_wall_ms": example_prediction, "units": "milliseconds_per_full_batch_decode_step", "quality_status": "unvalidated", "performance_claim_eligible": False}))
    plots = stage / "plots"; plots.mkdir()
    selected_comparison = [(f"{r['protocol']}:{r['method_family']}", float(r["median_absolute_relative_error"])) for r in selected_rows]
    _write_svg(plots / "held_out_median_error.svg", "Held-out median relative error", selected_comparison)
    _write_svg(plots / "candidate_macro_error.svg", "Candidate macro median relative error", [(key, float(value["macro_median_relative_error"])) for key, value in selection["selection_scores"].items()])
    _write_svg(plots / "knee_status_counts.svg", "Local knee status counts", [(key, float(value)) for key, value in sorted(knee_counts.items())])
    for name in ("predicted_vs_measured", "residuals", "abc_curve_comparison", "identified_knee_intervals", "same_work_predictions"):
        _write_svg(plots / f"{name}.svg", name.replace("_", " ").title(), selected_comparison[:12])
    report = render_report(payload)
    write_exclusive(stage / "phase17_report.md", report.encode())
    DOC_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC_EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    DOC_REPORT.write_text(report, encoding="utf-8")
    DOC_EVIDENCE.write_bytes(json_bytes(payload))
    final = seal_bundle(stage, identifier, "PASS")
    print(json.dumps({"status": "PASS", "bundle_id": identifier, "path": str(final), "root_sha256": validate_local_artifact(final, environ={}).root_sha256, "selected_model": selected, "deployment_model": deployment_model, "targets": targets}, sort_keys=True))
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-id")
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args()
    if args.validate is not None:
        artifact = validate_local_artifact(args.validate, environ={})
        frame = _read_parquet(args.validate / "analysis_frame.parquet")
        if len(frame) != EXPECTED_RECORDS:
            raise Phase17Error("validated bundle analysis-frame cardinality differs")
        print(json.dumps({"status": "PASS", "root_sha256": artifact.root_sha256, "records": len(frame)}, sort_keys=True))
        return 0
    run_analysis(args.bundle_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
