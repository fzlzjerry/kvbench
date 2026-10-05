#!/usr/bin/env python3
"""Addendum timing worker: one (configuration, B, L, replicate) per process.

Runs inside the authorized measurement container with the Full Scan execution
repository (ec534d99) mounted read-only at /home/rockrock/cmu_paper.  The
measurement is the Full Scan worker itself (scripts.phase16_full_scan.run_worker:
session construction, 64 warmup replays, 5 x 256 graph replays with host
wall-clock and CUDA-event timing, allocation/path/finiteness audits).  The
addendum only (a) installs an implementation variant from overrides.py before
the session is built and (b) derives the host-wall process median with the
rule of scripts/phase16_wall_closure.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import overrides  # noqa: E402

RESULT_PREFIX = "ADDENDUM_TIMING_RESULT="
SCHEMA = "kvbench-addendum-20261005-timing-1.0.0"


def install_variant(variant: dict) -> dict | None:
    if not variant:
        return None
    if set(variant) == {"tq_splits"}:
        return overrides.install_tq_split_override(int(variant["tq_splits"]))
    if set(variant) == {"kivi_grouped_residual"} and variant["kivi_grouped_residual"] is True:
        return overrides.install_kivi_grouped_residual("m4")
    if set(variant) == {"kivi_grouped_residual"} and variant["kivi_grouped_residual"] == "m1x4":
        return overrides.install_kivi_grouped_residual("m1x4")
    if (set(variant) == {"kivi_grouped_residual", "fp16_reduced_precision_reduction"}
            and variant["kivi_grouped_residual"]):
        # Diagnostic option: cuBLAS FP16 reduction mode for the residual bmm calls
        # (the only FP16 cuBLAS calls in KIVI decode; the model runs in BF16).
        import torch
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = bool(
            variant["fp16_reduced_precision_reduction"])
        state = overrides.install_kivi_grouped_residual()
        state["allow_fp16_reduced_precision_reduction"] = bool(variant["fp16_reduced_precision_reduction"])
        return state
    raise RuntimeError(f"unknown variant {variant}")


def run(args: argparse.Namespace) -> dict:
    variant = json.loads(args.variant)
    if variant.get("tq_splits") is not None and not args.configuration.startswith("tq_"):
        raise RuntimeError("tq_splits applies to TurboQuant configurations only")
    state = install_variant(variant)
    import scripts.phase16_full_scan as p16

    historical = p16.actual_historical_context(args.context_label)
    record = {
        "method_config_id": args.configuration,
        "batch_size": args.batch_size,
        "context_label": args.context_label,
        "historical_context": historical,
        "replicate_index": args.replicate_index,
        "order_index": args.order_index,
    }
    entry = p16.prefix_entry(record, container_paths=True,
                             logical_root=common.FAMILY / "logical-prefixes")
    run_root = Path(args.output_dir) / "run"
    run_root.mkdir()
    (run_root / "stage-progress").mkdir()
    common.write_new(run_root / "prefix-entry.json", common.json_text(entry))
    common.write_new(run_root / "worker-record.json", common.json_text(record))
    payload = p16.run_worker(run_id=args.run_id, record=record, git_sha=common.EXECUTION_SHA,
                             run_root=run_root, entry=entry)
    payload_text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    common.write_new(run_root / "result.json", payload_text + "\n")
    wall_median, wall_batches = common.host_wall_process_median_ms(payload)
    checks = {key: payload.get(key) for key in (
        "finite_output", "no_backend_fallback", "allocation_stable", "kernel_path_stable",
        "gpu_exclusive", "fitting_eligible")}
    return {
        "schema_version": SCHEMA,
        "status": "completed",
        "addendum_id": common.ADDENDUM_ID,
        "run_id": args.run_id,
        "task": args.task,
        "method_config_id": args.configuration,
        "batch_size": args.batch_size,
        "context_label": args.context_label,
        "historical_context": historical,
        "replicate_index": args.replicate_index,
        "order_index": args.order_index,
        "variant": variant,
        "variant_state": overrides.describe(state),
        "latency_basis": "host_wall",
        "host_wall_process_median_ms": wall_median,
        "host_wall_batch_ms_per_op": wall_batches,
        "cuda_process_median_ms": payload.get("process_median_ms"),
        "warmup_replays": payload.get("warmup_replays"),
        "measured_steps": payload.get("measured_steps"),
        "measured_batches": payload.get("measured_batches"),
        "checks": checks,
        "kernel_count": payload.get("kernel_count"),
        "output_checksum": payload.get("output_checksum"),
        "sm_clock_min_mhz": payload.get("sm_clock_min_mhz"),
        "sm_clock_max_mhz": payload.get("sm_clock_max_mhz"),
        "prefix_source": entry.get("source"),
        "full_scan_payload_path": "run/result.json",
        "full_scan_payload_sha256": hashlib.sha256((payload_text + "\n").encode()).hexdigest(),
        "execution_git_sha": common.EXECUTION_SHA,
        "addendum_git_sha": args.addendum_commit,
        "container_digest": os.environ.get("KVBENCH_AUTHORIZED_IMAGE_DIGEST"),
        "worker_sha256": common.sha256_file(Path(__file__)),
        "overrides_sha256": common.sha256_file(Path(overrides.__file__)),
        "common_sha256": common.sha256_file(Path(common.__file__)),
        "finished_at_utc": common.utc_now(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--context-label", type=int, required=True)
    parser.add_argument("--replicate-index", type=int, required=True)
    parser.add_argument("--order-index", type=int, required=True)
    parser.add_argument("--variant", default="{}")
    parser.add_argument("--addendum-commit", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    code = 0
    try:
        result = run(args)
    except BaseException as error:  # the driver records the failure and continues
        result = {"schema_version": SCHEMA, "status": "failed", "run_id": args.run_id,
                  "error_type": type(error).__name__, "error": str(error),
                  "traceback": traceback.format_exc(), "finished_at_utc": common.utc_now()}
        code = 3
    text = json.dumps(result, sort_keys=True, default=str)
    common.write_new(Path(args.output_dir) / "worker_result.json", text + "\n")
    sys.stdout.write(RESULT_PREFIX + text + "\n")
    sys.stdout.flush()
    os._exit(code)  # same exit path as the Full Scan / Phase 15 workers


if __name__ == "__main__":
    main()
