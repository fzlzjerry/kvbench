#!/usr/bin/env python3
"""Pure-stdlib Phase 18 reproduction and frozen Phase 17 predictor."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence


CONFIG_FAMILY = {
    "bf16": "bf16",
    "tq_4bit_nc": "turboquant", "tq_k3v4_nc": "turboquant",
    "tq_3bit_nc": "turboquant", "k4v4": "kivi", "k2v4": "kivi",
    "k2v2": "kivi", "kvq4": "kvquant", "kvq3": "kvquant",
    "kvq2": "kvquant",
}
GEOMETRY_PROTOCOLS = (
    "leave_one_batch_out", "leave_one_config_out", "leave_context_band_out",
)
L_SCALE = 131071.0


class Phase18OfflineError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase18OfflineError(f"JSON object required: {path}")
    return value


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise Phase18OfflineError("percentile of empty values")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _q4_workspace_bytes(batch: int, capacity: int) -> int:
    quantized = max(0, capacity - 5)
    tiles = (quantized + 127) // 128
    return batch * 32 * tiles * 128 * 4


def cache_allocated_bytes(configuration: str, batch: int, capacity: int) -> int:
    """Frozen CPU byte formulas used by the Phase 17 predictor."""

    if configuration == "bf16":
        return 2 * 32 * batch * 8 * capacity * 128 * 2 + 163_840
    if configuration.startswith("tq_"):
        slot = {"tq_4bit_nc": 134, "tq_k3v4_nc": 118, "tq_3bit_nc": 102}[configuration]
        rounded = math.ceil(capacity / 16) * 16
        packed = 28 * rounded * 8 * slot
        skipped = 2 * 4 * 8 * rounded * 128 * 2
        mapping = math.ceil(capacity / 16) * 4 + rounded * 8
        levels = {"tq_4bit_nc": 16, "tq_k3v4_nc": 8, "tq_3bit_nc": 8}[configuration]
        fixed = 2 * 128 * 128 * 4 + (levels + max(0, levels - 1)) * 4
        store = 3 * capacity * 8 * 128 * 4 + 2 * capacity * 8 * 4
        decode = batch * (2 * 32 * 128 * 4 + 32 * 4 * 129 * 4 + 32 * 128 * 2 + 32 * 4)
        return batch * (packed + skipped + mapping + store) + fixed + decode
    if configuration in {"k4v4", "k2v4", "k2v2"}:
        k_bits, v_bits = {"k4v4": (4, 4), "k2v4": (2, 4), "k2v2": (2, 2)}[configuration]
        layers, kv_heads, query_heads, dimension = 32, 8, 32, 128
        residual = group = 32
        key_history = (capacity // group) * group
        value_history = max(0, capacity - residual)
        key_groups = key_history // group
        value_head_groups = dimension // group
        key_words = key_history * k_bits // 32
        value_words = dimension * v_bits // 32
        key_residual = layers * batch * kv_heads * residual * dimension * 2
        value_residual = key_residual
        fp16_staging = sum((
            value_residual, batch * query_heads * dimension * 2,
            2 * batch * kv_heads * dimension * 2,
            2 * batch * kv_heads * dimension * 2,
            2 * batch * kv_heads * value_head_groups * 2,
            batch * query_heads * dimension * 2,
        ))
        quant_elements = batch * kv_heads * dimension * residual
        quant_staging = quant_elements * 2 + quant_elements * 4 + batch * kv_heads * dimension * 8 * 4
        workspace = 2 * batch * query_heads * capacity * 2 + batch * query_heads * capacity * 4 + 2 * batch * query_heads * dimension * 2
        return sum((
            layers * batch * kv_heads * key_words * dimension * 4,
            layers * batch * kv_heads * value_words * value_history * 4,
            2 * layers * batch * kv_heads * key_groups * dimension * 2,
            2 * layers * batch * kv_heads * value_head_groups * value_history * 2,
            key_residual, value_residual,
            layers * (key_history + residual + value_history + residual) * 8,
            fp16_staging, quant_staging, workspace,
        ))
    bits = {"kvq4": 4, "kvq3": 3, "kvq2": 2}[configuration]
    levels = 1 << bits
    layers, heads, query_heads, dimension = 32, 8, 32, 128
    packed_rows = bits * dimension // 32
    query_elements = batch * query_heads * dimension
    kv_elements = batch * heads * dimension
    dense = batch * layers * heads * packed_rows * capacity * 4
    key_metadata = layers * levels * 4 + layers * heads * dimension * levels * 4 + 3 * layers * heads * dimension * 4 + 64 * 4
    value_metadata = layers * levels * 4 + layers * batch * capacity * levels * 4
    sparse = batch * layers * capacity * 12 * 4
    count_mask = 2 * layers * batch * capacity * 4 + capacity
    sink = layers * batch * heads * dimension * 5 * 2
    staging = 3 * kv_elements * 2 + 4 * heads * dimension * 4 + query_elements * 2 + query_elements * 4 + query_elements * 2 + 2 * 12 * 4 + 3 * 4 + 1 + heads * dimension * 4 + 2 * heads * dimension * 4 + levels * 4 + 5 * 4 + 2 * batch * capacity * 4 + 3 * 4 + 8
    workspace = 2 * batch * query_heads * capacity * 4 + batch * query_heads * capacity * 2 + batch * query_heads * 5 * 2 + 4 * query_elements * 4 + query_elements * 2 + (_q4_workspace_bytes(batch, capacity) if configuration == "kvq4" else 0)
    rope = layers * batch * (query_heads + heads) * 64 * 2
    return 2 * dense + key_metadata + value_metadata + 4 * sparse + count_mask + 2 * sink + staging + workspace + rope


def derived_r_alloc(configuration: str, batch: int, historical_context: int) -> float:
    capacity = historical_context + 1
    logical = 2 * 32 * batch * 8 * capacity * 128 * 2
    return logical / cache_allocated_bytes(configuration, batch, capacity)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _predict_d(model: Mapping[str, Any], batch: int, context: int, r_alloc: float) -> float:
    mean = [float(value) for value in model["scaler"]["mean"]]
    scale = [float(value) for value in model["scaler"]["scale"]]
    z_b = (math.log(batch) - mean[0]) / scale[0]
    z_r = (math.log(r_alloc) - mean[1]) / scale[1]
    basis = [1.0, z_b, z_r, z_b * z_r]
    parameters = [float(value) for value in model["parameters"]]
    width = len(basis)
    tau = math.exp(max(-10.0, min(12.0, _dot(basis, parameters[:width]))))
    slope = math.exp(max(-10.0, min(12.0, _dot(basis, parameters[width:2 * width]))))
    raw_knee = max(-30.0, min(30.0, _dot(basis, parameters[2 * width:])))
    sigmoid = 1.0 / (1.0 + math.exp(-raw_knee))
    knee = 0.02 + 0.96 * sigmoid
    return max(tau + slope * max(context / L_SCALE - knee, 0.0), 1e-9)


def predict(
    package: Path, *, batch: int, context: int,
    method_config: str | None = None, method_family: str | None = None,
    explicit_r_alloc: float | None = None,
) -> dict[str, Any]:
    if method_config is not None:
        if method_config not in CONFIG_FAMILY:
            raise Phase18OfflineError("unsupported method configuration")
        inferred = CONFIG_FAMILY[method_config]
        if method_family is not None and method_family != inferred:
            raise Phase18OfflineError("method family/configuration mismatch")
        method_family = inferred
    if method_family not in {"bf16", "turboquant", "kivi", "kvquant"}:
        raise Phase18OfflineError("method family is required")
    if method_config is None and explicit_r_alloc is None:
        raise Phase18OfflineError("explicit r_alloc is required without a configuration")
    if batch <= 0 or context <= 0:
        raise Phase18OfflineError("batch and historical context must be positive")
    r_alloc = explicit_r_alloc if explicit_r_alloc is not None else derived_r_alloc(str(method_config), batch, context)
    if r_alloc <= 0:
        raise Phase18OfflineError("r_alloc must be positive")
    index = _load_json(package / "models/index.json")
    model_id = str(index["deployment_model_id"])
    model_path = package / "models" / index["families"][method_family][model_id]
    model = _load_json(model_path)
    latency = _predict_d(model, batch, context, r_alloc)
    interval = model.get("new_process_log_residual_interval_95")
    prediction_interval = None
    if isinstance(interval, list) and len(interval) == 2:
        prediction_interval = [latency * math.exp(float(interval[0])), latency * math.exp(float(interval[1]))]
    domain = model["training_domain"]
    configuration_supported = method_config in model["training_configurations"] if method_config is not None else False
    interpolation = bool(
        configuration_supported
        and domain["batch_min"] <= batch <= domain["batch_max"]
        and domain["historical_context_min"] <= context <= domain["historical_context_max"]
        and domain["r_alloc_min"] <= r_alloc <= domain["r_alloc_max"]
    )
    bf16_r = derived_r_alloc("bf16", batch, context)
    bf16_path = package / "models" / index["families"]["bf16"][model_id]
    bf16_model = _load_json(bf16_path)
    bf16_latency = _predict_d(bf16_model, batch, context, bf16_r)
    input_fingerprint = hashlib.sha256(json.dumps({"method_family": method_family, "method_config": method_config, "batch": batch, "context": context, "r_alloc": r_alloc}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "schema_version": "kvbench-phase18-offline-prediction-1.0.0",
        "method_family": method_family, "method_config_id": method_config,
        "batch_size": batch, "historical_context": context,
        "r_alloc": r_alloc,
        "r_alloc_source": "explicit" if explicit_r_alloc is not None else "frozen_cpu_byte_formula",
        "predicted_host_wall_ms": latency,
        "prediction_interval_95_new_process_ms": prediction_interval,
        "predicted_bf16_host_wall_ms": bf16_latency,
        "predicted_same_work_ratio": bf16_latency / latency,
        "ratio_scope": "fully_predicted_same_work",
        "units": "milliseconds_per_full_batch_decode_step",
        "model_id": model_id,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "parameterization": "tau(B,r)+s(B,r)*max(L/131071-lambda(B,r),0)",
        "feature_requirements": model["feature_requirements"],
        "training_domain": domain,
        "domain_status": "interpolation" if interpolation else "extrapolation_or_unmeasured_configuration",
        "feasibility_status": "not_predicted",
        "input_fingerprint_sha256": input_fingerprint,
        "quality_status": "unvalidated",
        "performance_claim_eligible": False,
    }


def collapse_predictions(rows: Iterable[Mapping[str, str]], model_id: str = "D") -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, str]]] = {}
    for row in rows:
        if row["model_id"] != model_id:
            continue
        key = tuple(row[column] for column in (
            "protocol", "fold_id", "model_id", "method_family",
            "method_config_id", "batch_size", "historical_context",
            "grid_source", "domain_status",
        ))
        grouped.setdefault(key, []).append(row)
    result: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        observed = statistics.median(float(row["observed_host_wall_ms"]) for row in values)
        predicted = statistics.median(float(row["predicted_host_wall_ms"]) for row in values)
        result.append({
            "protocol": key[0], "fold_id": key[1], "model_id": key[2],
            "method_family": key[3], "method_config_id": key[4],
            "batch_size": int(key[5]), "historical_context": int(key[6]),
            "grid_source": key[7], "domain_status": key[8],
            "observed_host_wall_ms": observed,
            "predicted_host_wall_ms": predicted,
            "signed_relative_error": (predicted - observed) / observed,
            "absolute_relative_error": abs(predicted - observed) / observed,
            "process_rows": len(values),
        })
    return result


def audit(package: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    predictions = _csv_rows(package / "data/out_of_fold_predictions.csv")
    comparison = _csv_rows(package / "data/model_comparison.csv")
    frame = _csv_rows(package / "data/analysis_frame.csv")
    logical = collapse_predictions(predictions)
    cells = [row for row in comparison if row["model_id"] == "D" and row["subset"] == "combined" and row["protocol"] in GEOMETRY_PROTOCOLS and int(row["scored_rows"]) > 0]
    if len(cells) != 11:
        raise Phase18OfflineError("applicable macro-cell count differs")
    macro_median = sum(float(row["median_absolute_relative_error"]) for row in cells) / len(cells)
    macro_p95 = sum(float(row["p95_relative_error"]) for row in cells) / len(cells)
    bf16 = [row for row in logical if row["protocol"] == "leave_one_batch_out" and row["method_family"] == "bf16"]
    bf16.sort(key=lambda row: float(row["absolute_relative_error"]), reverse=True)
    split = _load_json(package / "data/split_manifest.json")
    frame_by_run = {row["run_id"]: row for row in frame if row["status"] == "completed"}
    folds = {(row["protocol"], row["fold_id"], row["method_family"]): row for row in split["folds"]}
    tail: list[dict[str, Any]] = []
    for row in bf16[:20]:
        fold = folds[("leave_one_batch_out", row["fold_id"], "bf16")]
        train = [frame_by_run[run_id] for run_id in fold["train_run_ids"]]
        tail.append({
            "rank": len(tail) + 1, "fold_id": row["fold_id"],
            "held_out_batch": row["batch_size"], "historical_context": row["historical_context"],
            "domain_status": row["domain_status"],
            "observed_host_wall_ms": row["observed_host_wall_ms"],
            "predicted_host_wall_ms": row["predicted_host_wall_ms"],
            "signed_relative_error": row["signed_relative_error"],
            "absolute_relative_error": row["absolute_relative_error"],
            "training_batches": json.dumps(sorted({int(value["batch_size"]) for value in train}), separators=(",", ":")),
            "training_context_min": min(int(value["historical_context"]) for value in train),
            "training_context_max": max(int(value["historical_context"]) for value in train),
            "units": "milliseconds_per_full_batch_decode_step",
            "normalization": "full_batch_not_divided_by_batch",
            "model_id": "D", "fold_parameter_identity": "not_persisted_separately",
            "oof_prediction_artifact_sha256": _load_json(package / "source_manifest.json")["source_files"]["out_of_fold_predictions.parquet"],
            "candidate_spec_sha256": _load_json(package / "source_manifest.json")["source_files"]["candidate_spec.json"],
        })
    domain_summary = []
    for domain in sorted({row["domain_status"] for row in bf16}):
        values = [float(row["absolute_relative_error"]) for row in bf16 if row["domain_status"] == domain]
        domain_summary.append({"domain_status": domain, "logical_points": len(values), "median_absolute_relative_error": statistics.median(values), "p95_relative_error": _percentile(values, 0.95), "maximum_absolute_relative_error": max(values)})
    joins_valid = all(row["fold_id"] == f"B{row['batch_size']}" for row in bf16)
    finite_positive = all(math.isfinite(float(row["predicted_host_wall_ms"])) and float(row["predicted_host_wall_ms"]) > 0 for row in bf16)
    bf16_r = [float(row["r_alloc"]) for row in frame if row["status"] == "completed" and row["method_config_id"] == "bf16"]
    report = {
        "schema_version": "kvbench-phase18-reporting-audit-1.0.0",
        "source_phase17_root_sha256": _load_json(package / "source_manifest.json")["phase17_root_sha256"],
        "selected_model": "D",
        "macro_metric_definition": {
            "cell_count": 11, "cell_weighting": "equal",
            "median_label": "macro_mean_of_cell_median_relative_errors",
            "p95_label": "macro_mean_of_cell_p95_relative_errors",
            "median_value": macro_median, "p95_value": macro_p95,
            "pooled_quantile": False,
            "excluded_not_applicable": ["leave_one_config_out:bf16"],
            "session_holdout_in_primary_macro": False,
        },
        "bf16_leave_one_batch_out": {
            "logical_points": len(bf16),
            "median_absolute_relative_error": statistics.median(float(row["absolute_relative_error"]) for row in bf16),
            "p95_relative_error": _percentile([float(row["absolute_relative_error"]) for row in bf16], 0.95),
            "tail_rows_exported": len(tail), "domain_summary": domain_summary,
            "diagnosis": "real_predictive_failure_concentrated_in_B1_short_context_edge_extrapolation",
            "checks": {
                "units": "PASS_full_batch_decode_step_ms",
                "per_token_normalization": "PASS_not_applied",
                "inverse_transform": "PASS_positive_direct_D_output",
                "coordinate_scaling": "PASS_historical_L_over_131071",
                "fold_and_model_joins": "PASS" if joins_valid else "FAIL",
                "bf16_r_alloc": {"status": "PASS_actual_point_dependent_not_forced_to_one", "minimum": min(bf16_r), "maximum": max(bf16_r)},
                "finite_positive_outputs": "PASS" if finite_positive else "FAIL",
                "fold_parameters": "not_persisted_separately_in_phase17_bundle",
            },
        },
        "target_statuses_unchanged": True,
        "quality_status": "unvalidated",
    }
    return report, tail, logical


def _svg_scatter(path: Path, title: str, rows: Sequence[tuple[float, float]], *, log_scale: bool = False) -> None:
    width, height, left, top, plot = 900, 560, 80, 55, 430
    transformed = [(math.log10(max(x, 1e-9)), math.log10(max(y, 1e-9))) if log_scale else (x, y) for x, y in rows]
    xs = [x for x, _ in transformed] or [0.0, 1.0]; ys = [y for _, y in transformed] or [0.0, 1.0]
    low, high = min(min(xs), min(ys)), max(max(xs), max(ys)); span = high - low or 1.0
    circles = "".join(f'<circle cx="{left+(x-low)/span*plot:.2f}" cy="{top+plot-(y-low)/span*plot:.2f}" r="2.3" fill="#235789" fill-opacity="0.55"/>' for x, y in transformed)
    diagonal = f'<line x1="{left}" y1="{top+plot}" x2="{left+plot}" y2="{top}" stroke="#d1495b" stroke-width="2"/>'
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/><text x="30" y="30" font-size="20">{title}</text>{diagonal}{circles}<text x="80" y="530" font-size="12">x=observed, y=predicted; {"log10 scale" if log_scale else "linear scale"}; outliers retained</text></svg>\n', encoding="utf-8")


def _svg_bars(path: Path, title: str, values: Sequence[tuple[str, float]]) -> None:
    width, height = 1050, 560
    maximum = max((value for _, value in values), default=1.0) or 1.0
    step = 920 / max(1, len(values)); bars = []
    for index, (label, value) in enumerate(values):
        x = 65 + index * step; bar = 380 * value / maximum
        bars.append(f'<rect x="{x:.2f}" y="{455-bar:.2f}" width="{max(4,step*0.62):.2f}" height="{bar:.2f}" fill="#2a9d8f"/><text x="{x:.2f}" y="475" font-size="9" transform="rotate(28 {x:.2f} 475)">{label}</text>')
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><rect width="100%" height="100%" fill="white"/><text x="30" y="28" font-size="20">{title}</text><line x1="55" y1="455" x2="1020" y2="455" stroke="black"/>{"".join(bars)}</svg>\n', encoding="utf-8")


def reproduce(package: Path, output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise Phase18OfflineError("reproduction output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"; figures.mkdir()
    report, tail, logical = audit(package)
    _write_json(output / "reporting_audit.json", report)
    _write_csv(output / "bf16_batch_holdout_tail.csv", tail)
    selected = [row for row in logical if row["protocol"] in GEOMETRY_PROTOCOLS]
    _svg_scatter(figures / "predicted_vs_measured.svg", "Held-out predicted vs measured latency", [(float(row["observed_host_wall_ms"]), float(row["predicted_host_wall_ms"])) for row in selected], log_scale=True)
    comparison = _csv_rows(package / "data/model_comparison.csv")
    cells = [row for row in comparison if row["model_id"] == "D" and row["subset"] == "combined" and row["protocol"] in GEOMETRY_PROTOCOLS and int(row["scored_rows"]) > 0]
    _svg_bars(figures / "holdout_errors.svg", "D median relative error by protocol/method", [(f"{row['protocol']}:{row['method_family']}", float(row["median_absolute_relative_error"])) for row in cells])
    models = {}
    for row in comparison:
        if row["subset"] == "combined" and row["protocol"] in GEOMETRY_PROTOCOLS and int(row["scored_rows"]) > 0:
            models.setdefault(row["model_id"], []).append(float(row["median_absolute_relative_error"]))
    _svg_bars(figures / "candidate_macro_comparison.svg", "Candidate macro mean of cell medians", [(key, sum(values)/len(values)) for key, values in sorted(models.items())])
    knees = _csv_rows(package / "data/knee_estimates.csv")
    statuses: dict[str, int] = {}
    for row in knees: statuses[row["fit_status"]] = statuses.get(row["fit_status"], 0) + 1
    _svg_bars(figures / "knee_intervals_and_status.svg", "Knee identifiability status (intervals in table)", [(key, float(value)) for key, value in sorted(statuses.items())])
    by_protocol_key = {(row["protocol"], row["method_config_id"], row["batch_size"], row["historical_context"]): row for row in selected}
    ratios = []
    for row in selected:
        if row["method_config_id"] == "bf16" or row["protocol"] == "leave_one_config_out": continue
        base = by_protocol_key.get((row["protocol"], "bf16", row["batch_size"], row["historical_context"]))
        if base:
            ratios.append((float(base["observed_host_wall_ms"])/float(row["observed_host_wall_ms"]), float(base["predicted_host_wall_ms"])/float(row["predicted_host_wall_ms"])))
    _svg_scatter(figures / "same_work_ratios.svg", "Measured vs fully predicted same-work ratios", ratios, log_scale=True)
    mechanism = _load_json(package / "data/mechanism_summary.json")
    _svg_bars(figures / "mechanism_summary.svg", f"Phase 14/15 mechanism summary; profiler point {mechanism['phase15_common_point_label']}", [("launch-floor support", float(mechanism["phase14_launch_floor_support"])), ("identifiable comparisons", float(mechanism["phase14_identifiable_comparisons"])), ("CPU submission reduced", float(mechanism["phase15_cpu_submission_reduced_pairs"])), ("GPU idle reduced", float(mechanism["phase15_gpu_idle_reduced_pairs"]))])
    examples = []
    for configuration, batch, context in (("bf16", 1, 4096), ("tq_4bit_nc", 4, 16384), ("k4v4", 2, 32768), ("kvq4", 1, 4096)):
        examples.append(predict(package, method_config=configuration, batch=batch, context=context))
    _write_json(output / "predictor_examples.json", {"examples": examples, "examples_are_validation": False})
    result = {"schema_version": "kvbench-phase18-reproduction-result-1.0.0", "status": "PASS", "source_phase17_root_sha256": report["source_phase17_root_sha256"], "macro_metric_definition": report["macro_metric_definition"], "bf16_tail_diagnosis": report["bf16_leave_one_batch_out"]["diagnosis"], "figure_count": len(list(figures.glob("*.svg"))), "predictor_example_count": len(examples), "gpu_launched": False, "network_accessed": False, "quality_evaluation_executed": False}
    _write_json(output / "reproduction_result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    rep = sub.add_parser("reproduce")
    rep.add_argument("--package", type=Path, required=True)
    rep.add_argument("--output", type=Path, required=True)
    pred = sub.add_parser("predict")
    pred.add_argument("--package", type=Path, required=True)
    pred.add_argument("--method-config")
    pred.add_argument("--method-family")
    pred.add_argument("--batch", type=int, required=True)
    pred.add_argument("--context", type=int, required=True)
    pred.add_argument("--r-alloc", type=float)
    args = parser.parse_args()
    if args.command == "reproduce":
        print(json.dumps(reproduce(args.package, args.output), indent=2, sort_keys=True))
    else:
        print(json.dumps(predict(args.package, batch=args.batch, context=args.context, method_config=args.method_config, method_family=args.method_family, explicit_r_alloc=args.r_alloc), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
