#!/usr/bin/env python3
"""POST-HOC Part E: KIVI same-work ratio with its fixed-cost excess removed (S_adj).

POST-HOC.  Defined during the final revision (2026-10-05), after all frozen results and the
Part A-D analyses were known.  No timing is re-run and no model is re-fitted: the script
reads only sealed numbers.

  * Part A artifact (3dc84c64...): the decomposition fits T = c0 + D_cache / BW_eff at each
    batch size (a5_decomposition_fits.csv) and the per-point times and same-work ratios
    (a1_roofline_ratios.csv); r_DRAM at B = 1, 128K (a2_bandwidth_b1.csv).
  * Phase 16R host-wall closure (5605558b...): the cross-process CV of each point.

For each KIVI configuration m and same-work point (B, L):

  dc0(m, B) = c0_m(B) - c0_BF16(B)
  S_adj     = T_BF16(B, L) / (T_m(B, L) - dc0(m, B))

dc0 contains every fixed cost that KIVI adds over BF16 (the adapter's per-query-head
kernels, FP16 staging, and the official kernels' fixed cost), so S_adj is an optimistic
bound on what removing the adapter overhead could give.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

import posthoc_common as pc

PART_A_DIR = pc.ARTIFACT_PARENT / "posthoc-a-20261004t143713987904z-0641de4b-ab87cf"
PART_A_ROOT = "3dc84c64946d748bc4b73b49fa4fad90118dd04b3baa7fe5ad86b5326036d224"
WALL_DIR = pc.REPO / "artifacts/phase16/phase16-20260831t123029614620z-ec534d99-de80ac/wall-closure"
WALL_ROOT = "5605558be0483ddfeffd251977306d3397aa27a66309324c6011e5043584103e"
KIVI = ("k4v4", "k2v4", "k2v2")
BATCHES = (1, 2, 4, 8, 16)
COLUMNS = ["method_config_id", "batch_size", "historical_context", "t_method_ms", "t_bf16_ms",
           "s_measured", "delta_c0_ms", "t_method_minus_delta_c0_ms", "s_adj",
           "method_fit_max_abs_relative_residual", "bf16_fit_max_abs_relative_residual",
           "method_point_cv", "bf16_point_cv"]


def span(values: list[float]) -> list[float]:
    return [min(values), max(values)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--parent", type=Path, default=pc.ARTIFACT_PARENT)
    args = parser.parse_args(argv)
    git_sha = pc.git_head()
    run_id = pc.new_run_id("e", git_sha)
    part_a = pc.verify_full_root(PART_A_DIR, PART_A_ROOT)
    wall = pc.verify_full_root(WALL_DIR, WALL_ROOT)

    fits = {(r["method_config_id"], int(r["batch_size"])): r
            for r in csv.DictReader(open(PART_A_DIR / "a5_decomposition_fits.csv"))}
    for key in [(c, b) for c in ("bf16", *KIVI) for b in BATCHES]:
        if fits[key]["status"] != "fitted":
            raise pc.PosthocError(f"no sealed decomposition fit for {key}")
    cv = {}
    for row in pq.read_table(WALL_DIR / "point_summary.parquet").to_pylist():
        cv[(str(row["method_config_id"]), int(row["batch_size"]), int(row["historical_context"]))] = row["cv"]
    r_dram = {r["method_config_id"]: float(r["r_dram_v1"])
              for r in csv.DictReader(open(PART_A_DIR / "a2_bandwidth_b1.csv"))}

    rows = []
    for r in csv.DictReader(open(PART_A_DIR / "a1_roofline_ratios.csv")):
        if r["method_config_id"] not in KIVI or not r["s_measured"] or not r["t_bf16_ms"]:
            continue
        m, b, l = r["method_config_id"], int(r["batch_size"]), int(r["historical_context"])
        delta = float(fits[(m, b)]["c0_ms"]) - float(fits[("bf16", b)]["c0_ms"])
        t_m, t_b = float(r["t_method_ms"]), float(r["t_bf16_ms"])
        rows.append({
            "method_config_id": m, "batch_size": b, "historical_context": l,
            "t_method_ms": t_m, "t_bf16_ms": t_b, "s_measured": float(r["s_measured"]),
            "delta_c0_ms": delta, "t_method_minus_delta_c0_ms": t_m - delta,
            "s_adj": t_b / (t_m - delta),
            "method_fit_max_abs_relative_residual": float(fits[(m, b)]["max_abs_relative_residual"]),
            "bf16_fit_max_abs_relative_residual": float(fits[("bf16", b)]["max_abs_relative_residual"]),
            "method_point_cv": cv[(m, b, l)], "bf16_point_cv": cv[("bf16", b, l)],
        })
    if len(rows) != 120:
        raise pc.PosthocError(f"expected 120 KIVI same-work points, found {len(rows)}")
    rows.sort(key=lambda r: (r["method_config_id"], r["batch_size"], r["historical_context"]))
    best = max(rows, key=lambda r: r["s_adj"])
    above = [r for r in rows if r["s_adj"] > 1]
    summary: dict[str, Any] = {
        "label": pc.POSTHOC_LABEL,
        "definition": "S_adj = T_BF16 / (T_KIVI - (c0_KIVI(B) - c0_BF16(B))); sealed Part A fits, no refit",
        "points": len(rows),
        "delta_c0_ms_range": span([r["delta_c0_ms"] for r in rows]),
        "delta_c0_ms_by_config_and_batch": {f"{m}:B{b}": float(fits[(m, b)]["c0_ms"]) - float(fits[("bf16", b)]["c0_ms"])
                                            for m in KIVI for b in BATCHES},
        "s_adj_range": span([r["s_adj"] for r in rows]),
        "s_adj_max": best,
        "points_with_s_adj_above_1": above,
        "kivi_fit_max_abs_relative_residual_range": span([r["method_fit_max_abs_relative_residual"] for r in rows]),
        "bw_eff_bytes_per_s": {f"{c}:B{b}": float(fits[(c, b)]["bw_eff_cache_bytes_per_s"])
                               for c in ("bf16", *KIVI) for b in BATCHES},
        "r_dram_b1_128k_frozen": {c: r_dram[c] for c in KIVI},
        "bf16_capacity_infeasible_note": ("B = 16 at 24K has no BF16 measurement (capacity-infeasible), so S_adj "
                                          "is undefined there"),
    }
    lines = [f"# {pc.POSTHOC_LABEL}", "", "## Part E: KIVI S_adj", "",
             f"- Points: {len(rows)}; dc0 {summary['delta_c0_ms_range'][0]:.2f}-{summary['delta_c0_ms_range'][1]:.2f} ms; "
             f"S_adj {summary['s_adj_range'][0]:.4f}-{summary['s_adj_range'][1]:.4f}.",
             f"- Maximum: {best['s_adj']:.4f} at {best['method_config_id']}, B = {best['batch_size']}, L = "
             f"{best['historical_context']} (T_KIVI {best['t_method_ms']:.2f} ms, dc0 {best['delta_c0_ms']:.2f} ms, "
             f"T_BF16 {best['t_bf16_ms']:.2f} ms; KIVI fit residual {100 * best['method_fit_max_abs_relative_residual']:.2f}%, "
             f"KIVI point CV {100 * best['method_point_cv']:.2f}%).",
             f"- Points above 1: {len(above)} ("
             + ", ".join(f"{r['method_config_id']} B={r['batch_size']} L={r['historical_context']}: {r['s_adj']:.4f}" for r in above)
             + ").", ""]
    stage = pc.new_stage(run_id, args.parent)
    try:
        pc.write_csv(stage / "e1_kivi_s_adj_points.csv", rows, COLUMNS)
        pc.write_new(stage / "summary.json", pc.json_text(summary))
        pc.write_new(stage / "partE_report.md", "\n".join(lines))
        sources = (Path(__file__), Path(pc.__file__))
        for source in sources:
            pc.write_new(stage / "code" / source.name, source.read_bytes())
        manifest = {
            "schema_version": "kvbench-posthoc-review-manifest-1.0.0",
            "status": "PASS", "posthoc": True, "label": pc.POSTHOC_LABEL,
            "analysis": "final_revision_part_e_kivi_s_adj",
            "created_at_utc": pc.utc_now(), "git_head": git_sha,
            "code_sha256": {p.name: pc.sha256_file(p) for p in sources},
            "environment": pc.environment_record(),
            "inputs": {"part_a": part_a, "phase16r_host_wall_closure": wall},
            "timing_source": "Phase 16R host-wall process medians (frozen), through the sealed Part A rows",
            "refit": False,
            "performance_claim_eligible": False,
        }
        final, root = pc.seal(stage, run_id, manifest, "posthoc_final_revision_part_e", args.parent)
    except Exception:
        print(f"stage left for inspection: {stage}", file=sys.stderr)
        raise
    print(f"artifact: {final}")
    print(f"root_sha256: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
