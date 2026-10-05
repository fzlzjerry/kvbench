#!/usr/bin/env python3
"""POST-HOC Part D: cache-only ceiling S_cache and the KIVI-k4v4 fixed-cost kernel breakdown.

POST-HOC.  Both analyses were defined during the final revision (2026-10-05), after
the frozen performance and quality results and the Part A-C analyses were known.  The
script re-runs no timing, launches no CUDA work, and never writes into an existing
evidence directory.  It reuses the Part A traffic model unchanged (imported from
partA_analysis.py, whose sealed copy is in the Part A artifact) and reads:

  * Phase 16R host-wall closure (5605558b...), Phase 16 outer root (d7458767...), and
    Phase 15 profiler root (641fc02d...), verified as in Part A;
  * the sealed Part A artifact (3dc84c64...), only to check that the recomputed model,
    fits, and ratios equal the published ones;
  * Phase 15 Nsight Systems events of the Graph-mode traces of BF16 and KIVI-k4v4 at
    B = 1, 4K (nsys_events_classified.parquet, nsys_run_index.parquet).

D.1 cache-only ceiling.  At each same-work point (m, B, L):

  S_cache = T_BF16 / (T_BF16 - D_cache,BF16 / BW_eff,BF16(B) + D_ideal,cache,m / BW_peak)

  T_BF16          measured BF16 step time (frozen host wall);
  D_cache,BF16    modeled BF16 cache-path traffic, alpha_BF16 * C_BF16(B, L);
  BW_eff,BF16(B)  BF16's effective cache bandwidth from the Part A decomposition fit
                  T = c0 + D_cache / BW_eff at batch size B;
  D_ideal,cache,m alpha_BF16 * C_m(B, L): m reads its allocation as efficiently as BF16
                  reads its own (A = 1); the stored variant excludes workspace and padding;
  BW_peak         1,792 GB/s.

BF16's non-cache time is kept at its measured value and only the cache path becomes
ideal, so S_cache bounds what a better cache kernel could gain.  S_roof (Part A) also
lets the non-cache work run at peak bandwidth and is reported next to it.

D.2 KIVI fixed-cost breakdown.  Per decode step (eight Graph replays per trace), GPU
kernel counts and summed kernel durations by kernel name in the BF16 and KIVI-k4v4
traces at B = 1, 4K, grouped into categories fixed by name patterns.  These are
profiler observations (Nsight Systems GPU kernel time), never benchmark timing.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any

import pyarrow.compute as pac
import pyarrow.parquet as pq

import partA_analysis as pa
import posthoc_common as pc

PART_A_DIR = pc.ARTIFACT_PARENT / "posthoc-a-20261004t143713987904z-0641de4b-ab87cf"
PART_A_ROOT = "3dc84c64946d748bc4b73b49fa4fad90118dd04b3baa7fe5ad86b5326036d224"
PEAK = pa.PEAK_BW_DATASHEET
SHORT_CONTEXT = 4096
KIVI_RUNS = {"bf16": "nsys-bf16-b1-l4096-cuda_graph-attempt0",
             "k4v4": "nsys-k4v4-b1-l4096-cuda_graph-attempt0"}
# Categories of the KIVI-k4v4 trace, fixed before the breakdown was tabulated.
KIVI_CATEGORIES = (
    ("kivi_quantized_cache_gemv", lambda n: n.startswith("bgemv")),
    ("fp16_bmm", lambda n: ("cutlass" in n and "_f16_" in n) or "gemvx::kernel<int, int, __half," in n),
    ("fp16_add", lambda n: "CUDAFunctor_add<c10::Half>" in n),
    ("bf16_flash_attention", lambda n: "flash_fwd" in n),
)
D1_COLUMNS = ["method_config_id", "family", "batch_size", "historical_context", "t_bf16_ms",
              "s_measured", "s_eq_port_v2", "s_roof_alg", "s_cache", "s_cache_stored",
              "s_over_s_cache", "bf16_cache_time_ms", "bf16_noncache_time_ms",
              "bf16_cache_time_share", "bw_eff_bf16_bytes_per_s", "d_cache_bf16_bytes",
              "d_ideal_cache_method_bytes", "d_ideal_cache_method_stored_bytes"]


def close(a: float, b: float, rel: float = 1e-9) -> bool:
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-300)


def check_against_part_a(model: pa.TrafficModel, a1: list[dict[str, Any]],
                         decomposition: list[dict[str, Any]]) -> dict[str, Any]:
    """The recomputed rows must equal the sealed Part A outputs."""

    record = pc.verify_full_root(PART_A_DIR, PART_A_ROOT)
    sealed_a1 = {(r["method_config_id"], int(r["batch_size"]), int(r["historical_context"])): r
                 for r in csv.DictReader(open(PART_A_DIR / "a1_roofline_ratios.csv"))}
    compared = 0
    for row in a1:
        key = (row["method_config_id"], row["batch_size"], row["historical_context"])
        sealed = sealed_a1[key]
        for field in ("t_bf16_ms", "s_measured", "s_eq_port_v2", "s_roof_alg"):
            if row[field] is None:
                if sealed[field] not in ("", None):
                    raise pc.PosthocError(f"{key} {field}: sealed value present, recomputed missing")
                continue
            if not close(float(sealed[field]), float(row[field]), 1e-12):
                raise pc.PosthocError(f"{key} {field}: {row[field]} differs from Part A {sealed[field]}")
            compared += 1
    sealed_fits = {(r["method_config_id"], int(r["batch_size"])): r
                   for r in csv.DictReader(open(PART_A_DIR / "a5_decomposition_fits.csv"))}
    for row in decomposition:
        if row.get("status") != "fitted":
            continue
        sealed = sealed_fits[(row["method_config_id"], row["batch_size"])]
        if not close(float(sealed["bw_eff_cache_bytes_per_s"]), row["bw_eff_cache_bytes_per_s"], 1e-12):
            raise pc.PosthocError(f"decomposition differs from Part A at {row['method_config_id']} B={row['batch_size']}")
        compared += 1
    return {**record, "values_compared_equal": compared}


def cache_ceiling(data: dict[str, Any], model: pa.TrafficModel, a1: list[dict[str, Any]],
                  decomposition: list[dict[str, Any]], exclusions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bw_bf16 = {row["batch_size"]: row["bw_eff_cache_bytes_per_s"] for row in decomposition
               if row["method_config_id"] == "bf16" and row.get("status") == "fitted"}
    if sorted(bw_bf16) != list(pa.BATCHES):
        raise pc.PosthocError("BF16 decomposition fit missing at some batch size")
    alpha_bf16 = model.alpha["v2"]["bf16"]
    rows = []
    for row in a1:
        key = (row["method_config_id"], row["batch_size"], row["historical_context"])
        if row["s_measured"] is None or row["t_bf16_ms"] is None:
            exclusions.append({"analysis": "D1_s_cache", "key": list(key),
                               "reason_code": "no_same_work_ratio",
                               "detail": "S_cache is computed only at the 357 same-work points"})
            continue
        configuration, batch, historical = key
        seconds = row["t_bf16_ms"] * 1e-3
        d_cache_bf16 = model.cache_traffic("v2", "bf16", batch, historical)
        cache_time = d_cache_bf16 / bw_bf16[batch]
        noncache_time = seconds - cache_time
        if noncache_time <= 0:
            raise pc.PosthocError(f"non-positive BF16 non-cache time at {key}")
        ideal = alpha_bf16 * model.cache_bytes(configuration, batch, historical)
        ideal_stored = alpha_bf16 * model.stored_bytes(configuration, batch, historical)
        s_cache = seconds / (noncache_time + ideal / PEAK)
        rows.append({
            "method_config_id": configuration, "family": pa.FAMILY_OF[configuration],
            "batch_size": batch, "historical_context": historical,
            "t_bf16_ms": row["t_bf16_ms"], "s_measured": row["s_measured"],
            "s_eq_port_v2": row["s_eq_port_v2"], "s_roof_alg": row["s_roof_alg"],
            "s_cache": s_cache, "s_cache_stored": seconds / (noncache_time + ideal_stored / PEAK),
            "s_over_s_cache": row["s_measured"] / s_cache,
            "bf16_cache_time_ms": cache_time * 1e3, "bf16_noncache_time_ms": noncache_time * 1e3,
            "bf16_cache_time_share": cache_time / seconds, "bw_eff_bf16_bytes_per_s": bw_bf16[batch],
            "d_cache_bf16_bytes": d_cache_bf16, "d_ideal_cache_method_bytes": ideal,
            "d_ideal_cache_method_stored_bytes": ideal_stored,
        })
    if len(rows) != 357:
        raise pc.PosthocError(f"expected 357 same-work points, found {len(rows)}")
    return rows


def span(values: list[float]) -> list[float]:
    return [min(values), max(values)]


def ceiling_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "points": len(rows),
        "s_cache_range": span([r["s_cache"] for r in rows]),
        "s_roof_alg_range": span([r["s_roof_alg"] for r in rows]),
        "count_s_measured_gt_s_cache": sum(r["s_measured"] > r["s_cache"] for r in rows),
        "count_s_measured_gt_s_cache_stored": sum(r["s_measured"] > r["s_cache_stored"] for r in rows),
        "bf16_noncache_time_ms_min": min(r["bf16_noncache_time_ms"] for r in rows),
        "by_family": {},
        "by_batch_at_4k": {},
        "b1_4k": [],
    }
    for family in pa.FAMILIES:
        subset = [r for r in rows if r["family"] == family]
        summary["by_family"][family] = {
            "points": len(subset),
            "s_measured_range": span([r["s_measured"] for r in subset]),
            "s_cache_range": span([r["s_cache"] for r in subset]),
            "s_cache_stored_range": span([r["s_cache_stored"] for r in subset]),
            "s_roof_alg_range": span([r["s_roof_alg"] for r in subset]),
            "s_over_s_cache_range": span([r["s_over_s_cache"] for r in subset]),
            "s_eq_port_v2_range": span([r["s_eq_port_v2"] for r in subset]),
        }
    for batch in pa.BATCHES:
        subset = [r for r in rows if r["batch_size"] == batch and r["historical_context"] == SHORT_CONTEXT]
        if subset:
            summary["by_batch_at_4k"][str(batch)] = {
                "points": len(subset),
                "s_eq_port_v2_range": span([r["s_eq_port_v2"] for r in subset]),
                "s_cache_range": span([r["s_cache"] for r in subset]),
                "bf16_cache_time_share": subset[0]["bf16_cache_time_share"],
            }
    for r in rows:
        if r["batch_size"] == 1 and r["historical_context"] == SHORT_CONTEXT:
            summary["b1_4k"].append({k: r[k] for k in ("method_config_id", "s_measured", "s_eq_port_v2",
                                                       "s_roof_alg", "s_cache", "bf16_cache_time_share")})
    # Smallest B * (L + 1) at which any configuration's S_cache exceeds thresholds.
    for threshold in (1.05, 1.10, 1.25):
        above = [r for r in rows if r["s_cache"] > threshold]
        summary[f"min_tokens_with_s_cache_gt_{threshold}"] = (
            min(r["batch_size"] * (r["historical_context"] + 1) for r in above) if above else None)
    for threshold in (1.05, 1.10):
        above = [r for r in rows if r["s_eq_port_v2"] > threshold]
        summary[f"min_tokens_with_s_eq_gt_{threshold}"] = (
            min(r["batch_size"] * (r["historical_context"] + 1) for r in above) if above else None)
    return summary


def kivi_breakdown(data: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ph15 = data["ph15"]
    index = {r["run_id"]: r for r in pq.read_table(ph15.path("nsys_run_index.parquet")).to_pylist()}
    events = pq.read_table(ph15.path("nsys_events_classified.parquet"))
    per_name: dict[str, dict[str, list[float]]] = {}
    runs: dict[str, Any] = {}
    for configuration, run_id in KIVI_RUNS.items():
        run = index[run_id]
        if run["status"] != "completed" or run["graph_mode"] != "cuda_graph":
            raise pc.PosthocError(f"{run_id} is not a completed Graph-mode trace")
        if int(run["batch_size"]) != 1 or int(run["historical_context"]) != SHORT_CONTEXT:
            raise pc.PosthocError(f"{run_id} is not at B = 1, 4K")
        steps = int(run["graph_launch_count"])
        mask = pac.and_(pac.equal(events["run_id"], run_id), pac.equal(events["event_kind"], "cuda_kernel"))
        kernels = events.filter(mask).to_pylist()
        if len(kernels) != int(run["kernel_count"]):
            raise pc.PosthocError(f"{run_id}: {len(kernels)} kernel events, run index says {run['kernel_count']}")
        totals: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
        for kernel in kernels:
            name = re.sub(r"\s+", " ", str(kernel["name"]))
            totals[name][0] += 1
            totals[name][1] += float(kernel["duration_ns"])
        per_name[configuration] = {k: [v[0] / steps, v[1] / steps / 1e6] for k, v in totals.items()}
        runs[configuration] = {
            "run_id": run_id, "graph_replays": steps,
            "kernels_per_step": len(kernels) / steps,
            "kernel_time_ms_per_step": sum(v[1] for v in per_name[configuration].values()),
            "profile_region_ms_per_step": float(run["total_profile_region_duration_ns"]) / steps / 1e6,
            "inter_kernel_idle_ms_per_step": float(run["gpu_inter_kernel_idle_total_ns"]) / steps / 1e6,
            "raw_report_sha256": run["raw_report_sha256"],
        }
    rows = []
    for name in sorted(set(per_name["bf16"]) | set(per_name["k4v4"])):
        b_count, b_ms = per_name["bf16"].get(name, [0.0, 0.0])
        k_count, k_ms = per_name["k4v4"].get(name, [0.0, 0.0])
        category = next((label for label, rule in KIVI_CATEGORIES if rule(name)), None)
        if category is None:
            category = "shared_with_bf16" if b_count and k_count else (
                "other_kivi_only" if k_count else "other_bf16_only")
        rows.append({"category": category, "kernel_name": name,
                     "bf16_count_per_step": b_count, "bf16_ms_per_step": b_ms,
                     "k4v4_count_per_step": k_count, "k4v4_ms_per_step": k_ms,
                     "extra_ms_per_step": k_ms - b_ms})
    categories: dict[str, dict[str, float]] = {}
    for row in rows:
        entry = categories.setdefault(row["category"], {"bf16_count_per_step": 0.0, "bf16_ms_per_step": 0.0,
                                                         "k4v4_count_per_step": 0.0, "k4v4_ms_per_step": 0.0})
        for field in entry:
            entry[field] += row[field]
    summary = {
        "label": "Nsight Systems GPU kernel time per decode step (eight Graph replays); profiler "
                 "observation, not benchmark timing",
        "runs": runs,
        "categories": categories,
        "extra_kernel_ms_per_step": runs["k4v4"]["kernel_time_ms_per_step"] - runs["bf16"]["kernel_time_ms_per_step"],
        "extra_idle_ms_per_step": runs["k4v4"]["inter_kernel_idle_ms_per_step"] - runs["bf16"]["inter_kernel_idle_ms_per_step"],
        "frozen_wall_ms": {c: float(data["points"][(c, 1, SHORT_CONTEXT)]["median_ms"]) for c in KIVI_RUNS},
        "attribution_note": ("fp16_bmm and fp16_add are the per-query-head torch.bmm and add_ calls of the "
                             "KIVI adapter's residual-window and current-token attention "
                             "(src/kvbench/adapters/kivi.py, loops over 32 query heads); static code reading"),
    }
    return rows, summary


def report(ceiling: dict[str, Any], kivi: dict[str, Any]) -> str:
    def r(pair: list[float], digits: int = 3) -> str:
        return f"{pair[0]:.{digits}f}-{pair[1]:.{digits}f}"

    lines = [f"# {pc.POSTHOC_LABEL}", "", "## D.1 Cache-only ceiling S_cache", "",
             f"- Points: {ceiling['points']}; S_cache {r(ceiling['s_cache_range'])}; "
             f"S_roof {r(ceiling['s_roof_alg_range'])}; S > S_cache at {ceiling['count_s_measured_gt_s_cache']} points.",
             f"- Smallest BF16 non-cache time: {ceiling['bf16_noncache_time_ms_min']:.2f} ms.", "",
             "| Family | S | S_cache | S_cache (stored) | S_roof | S / S_cache | S_eq |",
             "|---|---|---|---|---|---|---|"]
    for family, item in ceiling["by_family"].items():
        lines.append(f"| {family} | {r(item['s_measured_range'])} | {r(item['s_cache_range'])} | "
                     f"{r(item['s_cache_stored_range'])} | {r(item['s_roof_alg_range'])} | "
                     f"{r(item['s_over_s_cache_range'])} | {r(item['s_eq_port_v2_range'])} |")
    lines += ["", "At 4K by batch size (BF16 cache-path share of the step):", ""]
    for batch, item in ceiling["by_batch_at_4k"].items():
        lines.append(f"- B = {batch}: S_eq {r(item['s_eq_port_v2_range'])}, S_cache {r(item['s_cache_range'])}, "
                     f"BF16 cache share {100 * item['bf16_cache_time_share']:.1f}%")
    lines += ["", "## D.2 KIVI-k4v4 fixed-cost kernel breakdown (B = 1, 4K, Graph)", "",
              kivi["label"], "",
              "| Category | BF16 kernels | BF16 ms | KIVI kernels | KIVI ms |", "|---|---|---|---|---|"]
    for category, item in sorted(kivi["categories"].items()):
        lines.append(f"| {category} | {item['bf16_count_per_step']:.0f} | {item['bf16_ms_per_step']:.3f} | "
                     f"{item['k4v4_count_per_step']:.0f} | {item['k4v4_ms_per_step']:.3f} |")
    lines += ["", f"- Extra kernel time per step: {kivi['extra_kernel_ms_per_step']:.3f} ms; "
                  f"extra inter-kernel idle: {kivi['extra_idle_ms_per_step']:.3f} ms.",
              f"- Frozen wall-clock medians: {kivi['frozen_wall_ms']}.", f"- {kivi['attribution_note']}.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--parent", type=Path, default=pc.ARTIFACT_PARENT,
                        help="directory that receives the sealed artifact (default: paper/posthoc/artifacts)")
    args = parser.parse_args(argv)
    log = pc.WarningLog()
    git_sha = pc.git_head()
    run_id = pc.new_run_id("d", git_sha)
    data = pa.load_inputs(log)
    traffic, _ = pa.run_traffic(data)
    model = pa.TrafficModel(data, traffic)
    part_a_exclusions: list[dict[str, Any]] = []
    a1 = pa.roofline_rows(data, model, PEAK, part_a_exclusions)
    decomposition = pa.decomposition_rows(data, model)
    part_a = check_against_part_a(model, a1, decomposition)
    exclusions: list[dict[str, Any]] = []
    ceiling_rows = cache_ceiling(data, model, a1, decomposition, exclusions)
    ceiling = ceiling_summary(ceiling_rows)
    kivi_rows, kivi = kivi_breakdown(data)
    summary = {"label": pc.POSTHOC_LABEL, "peak_bw_bytes_per_s": PEAK,
               "s_cache_definition": ("T_BF16 / (T_BF16 - D_cache,BF16 / BW_eff,BF16(B) + "
                                      "alpha_BF16 * C_m / BW_peak); BW_eff,BF16 from the Part A decomposition"),
               "d1_cache_ceiling": ceiling, "d2_kivi_fixed_cost": kivi}

    stage = pc.new_stage(run_id, args.parent)
    try:
        pc.write_csv(stage / "d1_cache_ceiling_points.csv", ceiling_rows, D1_COLUMNS)
        pc.write_csv(stage / "d2_kivi_kernel_breakdown.csv", kivi_rows, list(kivi_rows[0]))
        pc.write_new(stage / "exclusions.json", pc.json_text({
            "schema_version": "kvbench-posthoc-exclusions-1.0.0", "exclusions": exclusions,
            "counts": dict(Counter(f"{e['analysis']}:{e['reason_code']}" for e in exclusions))}))
        pc.write_new(stage / "summary.json", pc.json_text(summary))
        pc.write_new(stage / "partD_report.md", report(ceiling, kivi))
        sources = (Path(__file__), Path(pa.__file__), Path(pc.__file__))
        for source in sources:
            pc.write_new(stage / "code" / source.name, source.read_bytes())
        manifest = {
            "schema_version": "kvbench-posthoc-review-manifest-1.0.0",
            "status": "PASS",
            "posthoc": True,
            "label": pc.POSTHOC_LABEL,
            "analysis": "final_revision_part_d",
            "created_at_utc": pc.utc_now(),
            "git_head": git_sha,
            "code_sha256": {p.name: pc.sha256_file(p) for p in sources},
            "environment": pc.environment_record(),
            "inputs": {"phase16r_host_wall_closure": data["roots"]["wall"],
                       "phase16_outer": data["roots"]["outer"],
                       "phase15_profiler": data["ph15"].record(),
                       "part_a": part_a},
            "parameters": {"peak_bw_bytes_per_s": PEAK, "short_context": SHORT_CONTEXT,
                           "kivi_runs": KIVI_RUNS,
                           "kivi_categories": [label for label, _ in KIVI_CATEGORIES]},
            "timing_source": ("Phase 16R host-wall process medians (frozen); Nsight Systems kernel times "
                              "appear only in D.2 as profiler observations"),
            "profiler_duration_is_normal_timing": False,
            "performance_claim_eligible": False,
            "warnings": log.entries,
        }
        final, root = pc.seal(stage, run_id, manifest, "posthoc_final_revision_part_d", args.parent)
    except Exception:
        print(f"stage left for inspection: {stage}", file=sys.stderr)
        raise
    print(f"artifact: {final}")
    print(f"root_sha256: {root}")
    print(f"warnings: {len(log.entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
