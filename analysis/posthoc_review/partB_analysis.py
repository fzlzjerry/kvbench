#!/usr/bin/env python3
"""POST-HOC Part B (GPU results): TurboQuant split diagnostic and traffic-model check.

POST-HOC DIAGNOSTIC.  Reads one sealed raw artifact written by
partB_gpu_diagnostics.py (verified against its root hash) and the sealed Part A
artifact (for the traffic model), and seals a derived artifact.  Profiler
durations are reported as traced GPU kernel time and ratios, never as
benchmark timing.

Split diagnostic (Nsight Systems, Graph mode, 8 replays): per decode step,
stage-1 kernel time, total GPU kernel time, and GPU span, each as split-32 over
split-4 ratios, plus TurboQuant split-32 kernel time over BF16 kernel time at
the same point.  ALERT if any TurboQuant split-32 step kernel time is within
10% of, or below, BF16's.

Traffic-model check (Nsight Compute, Phase 15 metrics and defaults): measured
total, cache-path (Phase 15 rules = v1; v2 adds KVQuant key-path kernels) and
non-cache DRAM bytes against the Part A model at the same (B, L).  The kernel
classifier below is copied from scripts/phase15_profiler_subset.py and is
checked against every Phase 15 kernel record before use.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import pyarrow.parquet as pq

import posthoc_common as pc
from partB_offline import TQ_STAGE1, TQ_STAGE2, find, trace_summary

REPO = pc.REPO
PH15_DIR = REPO / "artifacts/phase15/phase15-20260828t144810363697z-446b334e-90460f"
PH15_ROOT = "641fc02d8fa598097885b74a336b1b1f454d9844b90025cf0c4b427bee02d5e8"
WALL_DIR = REPO / "artifacts/phase16/phase16-20260831t123029614620z-ec534d99-de80ac/wall-closure"
WALL_ROOT = "5605558be0483ddfeffd251977306d3397aa27a66309324c6011e5043584103e"
PART_A_ROOT = "3dc84c64946d748bc4b73b49fa4fad90118dd04b3baa7fe5ad86b5326036d224"
ALERT_RATIO = 1.10
FAMILY_OF = {"bf16": "bf16", "tq_4bit_nc": "turboquant", "tq_k3v4_nc": "turboquant",
             "tq_3bit_nc": "turboquant", "k4v4": "kivi", "k2v4": "kivi", "k2v2": "kivi",
             "kvq4": "kvquant", "kvq3": "kvquant", "kvq2": "kvquant"}
KVQ_KEY_PATTERNS = ("matmulkernelnuqperchanneltransposedropemhabatchedfusedopt",
                    "spmv_atomic_rope_balanced")


# ---- copied from scripts/phase15_profiler_subset.py (classify_kernel, kernel_is_cache_path)
def classify_kernel(name: str, *, configuration: str, nvtx_ranges: tuple[str, ...] = ()) -> tuple[str, str]:
    text = " ".join([name, *nvtx_ranges]).lower()
    family = FAMILY_OF[configuration]
    family_rules: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
        "turboquant": (
            ("dense_cache_attention", ("_tq_decode_stage1",)),
            ("output_merge", ("_fwd_kernel_stage2",)),
            ("quantize", ("_tq_fused_store_mse", "normtwoops")),
        ),
        "kivi": (
            ("dense_cache_attention", ("bgemv2_kernel_outer_dim", "bgemv4_kernel_outer_dim",
                                       "cunn_softmaxforward")),
        ),
        "kvquant": (
            ("dense_cache_attention", ("matmulkernelnuqperchanneltransposedmhabatchedfusedoptdeterministictiles",
                                       "cunn_softmaxforward")),
            ("output_merge", ("matmulkernelnuqperchanneltransposedmhabatchedfusedoptdeterministicreduce",)),
            ("kvquant_sparse_selection", ("selectfixedoutliers1024cap12kernel",)),
            ("kvquant_sparse_correction", ("writekeysparseresidual1024cap12kernel",
                                           "writevaluemetadataandsparsekernel")),
            ("cache_append", ("vecquant2appendveck", "vecquant3appendveck", "vecquant4appendveck",
                              "appendvaluesparsedeviceoutkernel", "clearvaluepackedcolumnkernel")),
        ),
    }
    for role, patterns in family_rules.get(family, ()):
        if any(pattern in text for pattern in patterns):
            return role, "exact_kernel_symbol_plus_method_authority"
    rules: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("kvquant_sparse_selection", ("select_fixed_outlier", "sparse_selection")),
        ("kvquant_sparse_correction", ("kvquant_sparse", "sparse_correction", "value_sparse", "key_sparse_residual")),
        ("kivi_residual", ("kivi_residual", "residual_attention", "residual_cache")),
        ("sink_attention", ("sink_attention", "sink_cache", "sink_")),
        ("cache_append", ("cache_append", "append_decode", "update_cache", "scatter_kernel")),
        ("quantize", ("quantize", "pack_kv", "pack_key", "pack_value")),
        ("dequantize", ("dequant", "unpack_kv", "decompress")),
        ("dense_cache_attention", ("flash_fwd", "fmha", "attention", "attn")),
        ("output_merge", ("output_merge", "merge_output", "correction_merge")),
        ("model_projection", ("gemm", "cutlass", "cublas", "projection", "linear")),
        ("other_model", ("rmsnorm", "layer_norm", "meanops", "indexselectsmallindex", "rotary", "rope",
                         "silu", "elementwise")),
    )
    for role, patterns in rules:
        if any(pattern in text for pattern in patterns):
            if role.startswith("kvquant") and family != "kvquant":
                continue
            if role == "kivi_residual" and family != "kivi":
                continue
            return role, "kernel_name_plus_method_authority"
    return "unknown", "insufficient_unambiguous_evidence"


def kernel_is_cache_path(role: str) -> bool:
    return role in {"cache_append", "dense_cache_attention", "quantize", "dequantize", "kivi_residual",
                    "kvquant_sparse_selection", "kvquant_sparse_correction", "sink_attention", "output_merge"}
# ---- end of copy


def verify_classifier(ledger: pc.LedgerRoot) -> dict[str, Any]:
    columns = ["run_id", "kernel_name", "kernel_role", "cache_path"]
    rows = pq.read_table(ledger.path("kernel_metrics.parquet"), columns=columns).to_pylist()
    mismatches = 0
    for row in rows:
        configuration = row["run_id"].split("-")[1]
        role, _ = classify_kernel(str(row["kernel_name"]), configuration=configuration,
                                  nvtx_ranges=("phase15_decode",))
        if role != row["kernel_role"] or kernel_is_cache_path(role) != bool(row["cache_path"]):
            mismatches += 1
    if mismatches:
        raise pc.PosthocError(f"copied classifier disagrees with Phase 15 on {mismatches} kernels")
    return {"kernels_checked": len(rows), "mismatches": 0}


UNIT_SCALE = {"byte": 1.0, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "Tbyte": 1e12}


def ncu_traffic(path: Path, configuration: str) -> dict[str, Any]:
    lines = [line for line in path.read_text(errors="replace").splitlines()
             if line.strip() and not line.startswith("==")]
    rows = list(csv.reader(io.StringIO("\n".join(lines))))
    header, units = [h.strip() for h in rows[0]], rows[1]
    index = {name: i for i, name in enumerate(header)}
    read_i, write_i = index["dram__bytes_op_read.sum"], index["dram__bytes_op_write.sum"]
    name_i = index["Kernel Name"]
    scale_r = UNIT_SCALE.get(units[read_i].strip() or "byte", 1.0)
    scale_w = UNIT_SCALE.get(units[write_i].strip() or "byte", 1.0)
    total = cache_v1 = moved = 0.0
    kernels = 0
    roles: dict[str, float] = {}
    for raw in rows[2:]:
        if len(raw) != len(header):
            continue
        value = (float(raw[read_i].replace(",", "") or 0) * scale_r
                 + float(raw[write_i].replace(",", "") or 0) * scale_w)
        name = raw[name_i]
        role, _ = classify_kernel(name, configuration=configuration, nvtx_ranges=("phase15_decode",))
        kernels += 1
        total += value
        roles[role] = roles.get(role, 0.0) + value
        if kernel_is_cache_path(role):
            cache_v1 += value
        elif FAMILY_OF[configuration] == "kvquant" and any(p in name.lower() for p in KVQ_KEY_PATTERNS):
            moved += value
    return {"kernels": kernels, "total": total, "cache_v1": cache_v1, "cache_v2": cache_v1 + moved,
            "bytes_by_role": roles}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("raw_artifact", type=Path, help="sealed posthoc-b-gpu-... directory")
    parser.add_argument("--part-a", type=Path, required=True, help="sealed Part A artifact directory")
    parser.add_argument("--parent", type=Path, default=pc.ARTIFACT_PARENT)
    parser.add_argument("--excluded-staging", action="append", default=[], metavar="PATH=REASON_CODE",
                        help="an unsealed staging directory that is retained but not analyzed, with its reason")
    parser.add_argument("--preview", action="store_true",
                        help="read an unsealed staging directory and print the report; seal nothing")
    args = parser.parse_args(argv)
    log = pc.WarningLog()
    git_sha = pc.git_head()

    from scripts.r2_artifact import validate_local_artifact

    raw_root = None if args.preview else validate_local_artifact(args.raw_artifact, environ={}).root_sha256
    part_a_root = validate_local_artifact(args.part_a, environ={}).root_sha256
    if part_a_root != PART_A_ROOT:
        raise pc.PosthocError("Part A artifact root differs")
    wall = pc.verify_full_root(WALL_DIR, WALL_ROOT)
    ledger = pc.LedgerRoot(PH15_DIR, PH15_ROOT)
    classifier_check = verify_classifier(ledger)
    model = json.loads((args.part_a / "summary.json").read_text())["traffic_model"]
    points = {(r["method_config_id"], int(r["batch_size"]), int(r["historical_context"])): r
              for r in pq.read_table(WALL_DIR / "point_summary.parquet").to_pylist()}

    jobs = []
    for job_dir in sorted((args.raw_artifact / "jobs").iterdir()):
        status_path = job_dir / "status.json"
        if not status_path.exists():
            continue
        status = json.loads(status_path.read_text())
        job = json.loads((job_dir / "job.json").read_text())
        worker_path = job_dir / "worker_result.json"
        worker = json.loads(worker_path.read_text()) if worker_path.exists() else None
        jobs.append({"dir": job_dir, "job": job, "status": status, "worker": worker})
    exclusions = [{"job_id": j["job"]["job_id"], "reason_code": j["status"].get("reason_code"),
                   "detail": j["status"].get("worker_error") or j["status"].get("detail")}
                  for j in jobs if j["status"]["status"] != "completed"]
    for item in args.excluded_staging:
        path_text, _, reason = item.partition("=")
        path = Path(path_text)
        files = sorted(f for f in path.rglob("*") if f.is_file())
        exclusions.append({
            "job_id": None, "staging_path": str(path), "reason_code": reason,
            "detail": "retained in place, not sealed into the analyzed run, and not analyzed",
            "files": {str(f.relative_to(path)): pc.sha256_file(f) for f in files},
        })

    # ---- split diagnostic
    traces: dict[tuple, dict[str, Any]] = {}
    worker_checks = []
    for item in jobs:
        job, worker = item["job"], item["worker"]
        if item["status"]["status"] != "completed":
            continue
        worker_checks.append({
            "job_id": job["job_id"], "output_finite": worker.get("output_finite"),
            "cache_pointers_stable": worker.get("cache_pointers_stable"),
            "historical_cache_unchanged": worker.get("historical_cache_unchanged"),
            "adapter_fingerprint_validation_error": worker.get("adapter_fingerprint_validation_error"),
            "tq_split_override": worker.get("tq_split_override"),
            "prefix_restore_mode": (worker.get("prefix_source") or {}).get("restore_mode"),
        })
        if job["kind"] != "nsys":
            continue
        summary = log.guard(f"trace_{job['job_id']}",
                            lambda d=item["dir"]: trace_summary(d / "report.sqlite", 8))
        if summary is None:
            exclusions.append({"job_id": job["job_id"], "reason_code": "trace_parse_failed", "detail": None})
            continue
        key = (job["configuration"], job["batch_size"], job["context_label"], job["tq_splits"])
        traces[key] = summary

    split_rows = []
    alerts = []
    for (configuration, batch, label, splits), summary in sorted(traces.items(), key=lambda p: str(p[0])):
        if not configuration.startswith("tq_") or splits != 32:
            continue
        base = traces.get((configuration, batch, label, 4))
        bf16 = traces.get(("bf16", batch, label, None))
        stage1 = find(summary["symbols"], TQ_STAGE1)
        stage1_base = find(base["symbols"], TQ_STAGE1) if base else None
        stage2 = find(summary["symbols"], TQ_STAGE2)
        stage2_base = find(base["symbols"], TQ_STAGE2) if base else None
        historical = 131071 if label == 131072 else label
        measured = points.get((configuration, batch, historical))
        row = {
            "method_config_id": configuration, "batch_size": batch, "context_label": label,
            "stage1_grid_split4": stage1_base["geometries"][0]["grid"] if stage1_base else None,
            "stage1_grid_split32": stage1["geometries"][0]["grid"],
            "stage1_ms_split4": stage1_base["kernel_ms_per_step"] if stage1_base else None,
            "stage1_ms_split32": stage1["kernel_ms_per_step"],
            "stage2_ms_split4": stage2_base["kernel_ms_per_step"] if stage2_base else None,
            "stage2_ms_split32": stage2["kernel_ms_per_step"] if stage2 else None,
            "step_kernel_ms_split4": base["kernel_ms_per_step_median"] if base else None,
            "step_kernel_ms_split32": summary["kernel_ms_per_step_median"],
            "step_span_ms_split4": base["gpu_span_ms_per_step_median"] if base else None,
            "step_span_ms_split32": summary["gpu_span_ms_per_step_median"],
            "bf16_step_kernel_ms": bf16["kernel_ms_per_step_median"] if bf16 else None,
            "bf16_step_span_ms": bf16["gpu_span_ms_per_step_median"] if bf16 else None,
            "frozen_host_wall_ms_split4": float(measured["median_ms"]) if measured and measured["median_ms"] else None,
        }
        row["stage1_ratio_32_over_4"] = (row["stage1_ms_split32"] / row["stage1_ms_split4"]
                                         if row["stage1_ms_split4"] else None)
        row["step_kernel_ratio_32_over_4"] = (row["step_kernel_ms_split32"] / row["step_kernel_ms_split4"]
                                              if row["step_kernel_ms_split4"] else None)
        row["step_span_ratio_32_over_4"] = (row["step_span_ms_split32"] / row["step_span_ms_split4"]
                                            if row["step_span_ms_split4"] else None)
        row["split32_over_bf16_kernel"] = (row["step_kernel_ms_split32"] / row["bf16_step_kernel_ms"]
                                           if row["bf16_step_kernel_ms"] else None)
        row["split32_over_bf16_span"] = (row["step_span_ms_split32"] / row["bf16_step_span_ms"]
                                         if row["bf16_step_span_ms"] else None)
        grid = row["stage1_grid_split32"]
        row["grid_matches_split"] = grid == [batch, 32, 32]
        if not row["grid_matches_split"]:
            log.warn("unexpected_grid", f"{configuration} B{batch} L{label}: stage-1 grid {grid}")
        for field in ("split32_over_bf16_kernel", "split32_over_bf16_span"):
            if row[field] is not None and row[field] <= ALERT_RATIO:
                alerts.append({"method_config_id": configuration, "batch_size": batch, "context_label": label,
                               "measure": field, "ratio": row[field]})
        split_rows.append(row)

    # ---- traffic-model check
    traffic_rows = []
    for item in jobs:
        job = item["job"]
        if job["kind"] != "ncu" or item["status"]["status"] != "completed":
            continue
        configuration, batch, label = job["configuration"], job["batch_size"], job["context_label"]
        historical = 131071 if label == 131072 else label
        measured = log.guard(f"ncu_{job['job_id']}",
                             lambda d=item["dir"], c=configuration: ncu_traffic(d / "report_raw.csv", c))
        if measured is None:
            exclusions.append({"job_id": job["job_id"], "reason_code": "ncu_parse_failed", "detail": None})
            continue
        allocated = float(points[(configuration, batch, historical)]["allocated_bytes"])
        family = FAMILY_OF[configuration]
        row = {"method_config_id": configuration, "batch_size": batch, "context_label": label,
               "kernels": measured["kernels"], "measured_total_bytes": measured["total"],
               "measured_cache_v1_bytes": measured["cache_v1"], "measured_cache_v2_bytes": measured["cache_v2"]}
        for version in ("v1", "v2"):
            fit = model["noncache_fit"][version][family]
            alpha = model["alpha"][version][configuration]
            cache_model = alpha * allocated
            noncache_model = fit["n0_bytes"] + fit["n1_bytes_per_sequence_token"] * batch * (historical + 1)
            row[f"modeled_cache_{version}_bytes"] = cache_model
            row[f"modeled_total_{version}_bytes"] = cache_model + noncache_model
            row[f"cache_{version}_relative_error"] = cache_model / measured[f"cache_{version}"] - 1
            row[f"noncache_{version}_relative_error"] = (
                noncache_model / (measured["total"] - measured[f"cache_{version}"]) - 1)
            row[f"total_{version}_relative_error"] = (cache_model + noncache_model) / measured["total"] - 1
        traffic_rows.append(row)
    for row in traffic_rows:
        bf16 = next((r for r in traffic_rows if r["method_config_id"] == "bf16"
                     and r["batch_size"] == row["batch_size"] and r["context_label"] == row["context_label"]), None)
        if bf16 and row["method_config_id"] != "bf16":
            row["measured_s_eq"] = bf16["measured_total_bytes"] / row["measured_total_bytes"]
            row["modeled_s_eq_v2"] = bf16["modeled_total_v2_bytes"] / row["modeled_total_v2_bytes"]

    if args.preview:
        stage = None
    else:
        run_id = pc.new_run_id("b-analysis", git_sha)
        stage = pc.new_stage(run_id, args.parent)
    if stage is not None and split_rows:
        pc.write_csv(stage / "split_diagnostic.csv", split_rows, list(split_rows[0]))
    summary = {"label": pc.POSTHOC_LABEL, "alert_ratio": ALERT_RATIO, "alerts": alerts,
               "classifier_check": classifier_check, "split_rows": split_rows, "traffic_rows": traffic_rows}
    if stage is not None:
        if traffic_rows:
            columns = sorted({k for r in traffic_rows for k in r},
                             key=lambda k: list(traffic_rows[0]).index(k) if k in traffic_rows[0] else 999)
            pc.write_csv(stage / "traffic_model_check.csv", traffic_rows, columns)
        pc.write_new(stage / "worker_checks.json", pc.json_text(worker_checks))
        pc.write_new(stage / "exclusions.json", pc.json_text({
            "schema_version": "kvbench-posthoc-exclusions-1.0.0", "exclusions": exclusions}))
        pc.write_new(stage / "summary.json", pc.json_text(json.loads(json.dumps(summary, default=str))))

    def fmt(value: Any, digits: int = 2) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    lines = ["# Part B GPU diagnostics: analysis", "", f"> {pc.POSTHOC_LABEL}.", "",
             "Traced GPU kernel time per decode step (Nsight Systems, Graph mode, median of 8 replays). "
             "Profiler observations, not benchmark timing.", "",
             f"ALERTS (TurboQuant split 32 within {int((ALERT_RATIO - 1) * 100)}% of or below BF16): "
             + ("none" if not alerts else json.dumps(alerts)), "",
             "| Config | B | L | stage-1 grid (32) | stage-1 ms 4 -> 32 | ratio | step kernel ms 4 -> 32 | ratio | "
             "BF16 step kernel ms | 32 / BF16 |", "|---|---:|---:|---|---|---:|---|---:|---:|---:|"]
    for row in split_rows:
        lines.append(f"| {pc.CONFIG_LABELS[row['method_config_id']]} | {row['batch_size']} | {row['context_label']} | "
                     f"{tuple(row['stage1_grid_split32'])} | {fmt(row['stage1_ms_split4'])} -> "
                     f"{fmt(row['stage1_ms_split32'])} | {fmt(row['stage1_ratio_32_over_4'], 3)} | "
                     f"{fmt(row['step_kernel_ms_split4'])} -> {fmt(row['step_kernel_ms_split32'])} | "
                     f"{fmt(row['step_kernel_ratio_32_over_4'], 3)} | {fmt(row['bf16_step_kernel_ms'])} | "
                     f"{fmt(row['split32_over_bf16_kernel'], 2)} |")
    if traffic_rows:
        lines += ["", "## Traffic-model check (Nsight Compute, cold cache, base clocks)", "",
                  "| Config | B | L | measured total GB | model total v2 error | cache v2 error | non-cache v2 error |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for row in traffic_rows:
            lines.append(f"| {pc.CONFIG_LABELS[row['method_config_id']]} | {row['batch_size']} | "
                         f"{row['context_label']} | {row['measured_total_bytes'] / 1e9:.3f} | "
                         f"{100 * row['total_v2_relative_error']:.2f}% | {100 * row['cache_v2_relative_error']:.2f}% | "
                         f"{100 * row['noncache_v2_relative_error']:.2f}% |")
    if exclusions:
        lines += ["", "## Exclusions", ""] + [f"- {e.get('job_id') or e.get('staging_path')}: {e['reason_code']} {e.get('detail') or ''}"
                                             for e in exclusions]
    lines.append("")
    if stage is None:
        print("\n".join(lines))
        print(json.dumps({"worker_checks": worker_checks}, indent=1, default=str)[:4000])
        return 0
    pc.write_new(stage / "partB_analysis_report.md", "\n".join(lines))
    for source in (Path(__file__), Path(pc.__file__), Path(__file__).with_name("partB_offline.py")):
        pc.write_new(stage / "code" / source.name, source.read_bytes())
    manifest = {
        "schema_version": "kvbench-posthoc-review-manifest-1.0.0", "status": "PASS", "posthoc": True,
        "label": pc.POSTHOC_LABEL, "analysis": "review_round_part_b_gpu_analysis", "created_at_utc": pc.utc_now(),
        "git_head": git_sha, "code_sha256": {p.name: pc.sha256_file(p) for p in (
            Path(__file__), Path(pc.__file__), Path(__file__).with_name("partB_offline.py"))},
        "environment": pc.environment_record(),
        "inputs": {"part_b_gpu_raw": {"path": str(args.raw_artifact), "root_sha256": raw_root},
                   "part_a": {"path": str(args.part_a), "root_sha256": part_a_root},
                   "phase16r_host_wall_closure": wall, "phase15_profiler": ledger.record()},
        "profiler_duration_is_normal_timing": False, "performance_claim_eligible": False,
        "alerts": alerts, "warnings": log.entries,
    }
    final, root = pc.seal(stage, run_id, manifest, "posthoc_review_part_b_gpu_analysis", args.parent)
    print(f"artifact: {final}\nroot_sha256: {root}\nalerts: {len(alerts)}\nwarnings: {len(log.entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
