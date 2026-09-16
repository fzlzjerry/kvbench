#!/usr/bin/env python3
"""Offline Phase 17 latency predictor; CPU formulas and exported coefficients only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scripts.phase13_pilot import cache_allocated_bytes
from scripts.phase17_modeling import predict_predictive


CONFIG_FAMILY = {
    "bf16": "bf16",
    "tq_4bit_nc": "turboquant", "tq_k3v4_nc": "turboquant", "tq_3bit_nc": "turboquant",
    "k4v4": "kivi", "k2v4": "kivi", "k2v2": "kivi",
    "kvq4": "kvquant", "kvq3": "kvquant", "kvq2": "kvquant",
}


def derived_r_alloc(configuration: str, batch: int, historical_context: int) -> float:
    capacity = historical_context + 1
    logical = 2 * 32 * batch * 8 * capacity * 128 * 2
    return logical / cache_allocated_bytes(configuration, batch, capacity)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def predict(
    bundle: Path, *, configuration: str, batch: int, historical_context: int,
    explicit_r_alloc: float | None = None,
) -> dict[str, Any]:
    if configuration not in CONFIG_FAMILY:
        raise ValueError("unsupported method configuration")
    index = _load(bundle / "models/index.json")
    model_id = str(index["deployment_model_id"])
    family = CONFIG_FAMILY[configuration]
    r_alloc = explicit_r_alloc if explicit_r_alloc is not None else derived_r_alloc(
        configuration, batch, historical_context
    )
    row = {
        "method_config_id": configuration, "method_family": family,
        "batch_size": batch, "historical_context": historical_context,
        "r_alloc": r_alloc,
    }
    model_path = bundle / "models" / index["families"][family][model_id]
    model = _load(model_path)
    latency = float(predict_predictive(model, [row])[0])
    interval = model.get("new_process_log_residual_interval_95")
    uncertainty = None
    if isinstance(interval, list) and len(interval) == 2:
        uncertainty = [latency * math.exp(float(interval[0])), latency * math.exp(float(interval[1]))]
    domain = model["training_domain"]
    interpolation = (
        domain["batch_min"] <= batch <= domain["batch_max"]
        and domain["historical_context_min"] <= historical_context <= domain["historical_context_max"]
        and domain["r_alloc_min"] <= r_alloc <= domain["r_alloc_max"]
        and configuration in model["training_configurations"]
    )
    baseline_r = derived_r_alloc("bf16", batch, historical_context)
    baseline_path = bundle / "models" / index["families"]["bf16"][model_id]
    baseline_model = _load(baseline_path)
    baseline = float(predict_predictive(baseline_model, [{
        "method_config_id": "bf16", "method_family": "bf16",
        "batch_size": batch, "historical_context": historical_context,
        "r_alloc": baseline_r,
    }])[0])
    return {
        "schema_version": "kvbench-phase17-offline-prediction-1.0.0",
        "method_config_id": configuration, "method_family": family,
        "batch_size": batch, "historical_context": historical_context,
        "r_alloc": r_alloc,
        "r_alloc_source": "explicit" if explicit_r_alloc is not None else "frozen_cpu_byte_formula",
        "predicted_host_wall_ms": latency,
        "prediction_interval_95_new_process_ms": uncertainty,
        "predicted_bf16_host_wall_ms": baseline,
        "predicted_same_work_ratio": baseline / latency,
        "units": "milliseconds_per_full_batch_decode_step",
        "model_id": model_id,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "feature_requirements": model["feature_requirements"],
        "training_domain": domain,
        "domain_status": "interpolation" if interpolation else "extrapolation",
        "quality_status": "unvalidated",
        "performance_claim_eligible": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--method-config", required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--context", type=int, required=True, help="actual historical context")
    parser.add_argument("--r-alloc", type=float)
    args = parser.parse_args()
    if args.batch <= 0 or args.context <= 0 or (args.r_alloc is not None and args.r_alloc <= 0):
        parser.error("batch, context, and optional r_alloc must be positive")
    print(json.dumps(predict(args.bundle, configuration=args.method_config, batch=args.batch, historical_context=args.context, explicit_r_alloc=args.r_alloc), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
