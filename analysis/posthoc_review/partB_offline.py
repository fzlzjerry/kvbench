#!/usr/bin/env python3
"""POST-HOC Part B (offline): kernel breakdown and launch geometry from the
existing Phase 15 Nsight traces (root 641fc02d...).  CPU only.

POST-HOC.  Defined after the frozen results were known.  Kernel durations come
from profiler traces (Nsight Systems kernel records, Graph mode, 8 replays per
trace; Nsight Compute only where noted) and are reported as shares of the
traced GPU kernel time per decode step, never as benchmark timing.

Outputs per Graph-mode nsys trace (all at B = 1):
  * per-step kernel time, GPU span, kernel count;
  * per-symbol launches per step, share of kernel time, launch geometry
    (grid, block, CTAs, warps) and mean duration per launch;
  * TurboQuant: stage-1 / stage-2 geometry and share; occupancy arithmetic
    against 188 SMs x 48 resident warps;
  * KVQuant: SelectFixedOutliers1024Cap12Kernel share, geometry, and growth
    with context.
TurboQuant k3v4 / 3bit have no nsys trace; their stage-1 geometry is read from
the Phase 15 Nsight Compute reports at 128K (noncritical; needs the ncu CLI).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import io
import json
from pathlib import Path
import shutil
import sqlite3
import statistics
import subprocess
import sys
from typing import Any

import pyarrow.parquet as pq

import posthoc_common as pc

REPO = pc.REPO
PH15_DIR = REPO / "artifacts/phase15/phase15-20260828t144810363697z-446b334e-90460f"
PH15_ROOT = "641fc02d8fa598097885b74a336b1b1f454d9844b90025cf0c4b427bee02d5e8"
SM_COUNT = 188
MAX_WARPS_PER_SM = 48  # 1,536 resident threads per SM on compute capability 12.0
TQ_STAGE1 = "_tq_decode_stage1"
TQ_STAGE2 = "_fwd_kernel_stage2"
KVQ_SELECT = "SelectFixedOutliers1024Cap12Kernel"


def symbol(name: str) -> str:
    """Short kernel symbol used for grouping (template and argument lists dropped)."""

    base = name.split("(")[0].strip()
    if base.startswith("void "):
        base = base[5:]
    for prefix in ("<unnamed>::", "(anonymous namespace)::"):
        base = base.replace(prefix, "")
    if "<" in base:
        base = base[:base.index("<")]
    return base.split("::")[-1] or name


def trace_summary(path: Path, decode_operations: int) -> dict[str, Any]:
    database = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = database.execute(
        "select k.start, k.end, s.value, k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY, k.blockZ "
        "from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id = k.demangledName order by k.start")
    kernels = [dict(zip(("start", "end", "name", "gx", "gy", "gz", "bx", "by", "bz"), r)) for r in rows]
    ranges = list(database.execute("select text, start, end from NVTX_EVENTS where text = 'phase15_decode'"))
    database.close()
    if not kernels or len(ranges) != 1:
        raise pc.PosthocError(f"{path.name}: unexpected trace content")
    if len(kernels) % decode_operations:
        raise pc.PosthocError(f"{path.name}: kernel count is not a multiple of the replays")
    per_step = len(kernels) // decode_operations
    steps = [kernels[i * per_step:(i + 1) * per_step] for i in range(decode_operations)]
    step_kernel_ns = [sum(k["end"] - k["start"] for k in step) for step in steps]
    step_span_ns = [step[-1]["end"] - step[0]["start"] for step in steps]
    symbols: dict[str, dict[str, Any]] = defaultdict(lambda: {"launches": 0, "kernel_ns": 0, "geometries": defaultdict(int)})
    for kernel in kernels:
        item = symbols[symbol(kernel["name"])]
        item["launches"] += 1
        item["kernel_ns"] += kernel["end"] - kernel["start"]
        item["geometries"][(kernel["gx"], kernel["gy"], kernel["gz"], kernel["bx"], kernel["by"], kernel["bz"])] += 1
    total_ns = sum(step_kernel_ns)
    table = []
    for name, item in sorted(symbols.items(), key=lambda pair: -pair[1]["kernel_ns"]):
        geometries = []
        for (gx, gy, gz, bx, by, bz), count in sorted(item["geometries"].items(), key=lambda p: -p[1]):
            ctas = gx * gy * gz
            warps_per_cta = -(-(bx * by * bz) // 32)
            geometries.append({"grid": [gx, gy, gz], "block": [bx, by, bz], "ctas": ctas,
                               "warps_per_cta": warps_per_cta, "launches": count})
        table.append({
            "symbol": name,
            "launches_per_step": item["launches"] / decode_operations,
            "kernel_ms_per_step": item["kernel_ns"] / decode_operations / 1e6,
            "share_of_kernel_time": item["kernel_ns"] / total_ns,
            "mean_us_per_launch": item["kernel_ns"] / item["launches"] / 1e3,
            "geometries": geometries,
        })
    return {
        "decode_operations": decode_operations,
        "kernels_per_step": per_step,
        "kernel_ms_per_step_median": statistics.median(step_kernel_ns) / 1e6,
        "gpu_span_ms_per_step_median": statistics.median(step_span_ns) / 1e6,
        "nvtx_region_ms": (ranges[0][2] - ranges[0][1]) / 1e6,
        "symbols": table,
    }


def find(table: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return next((row for row in table if row["symbol"] == name), None)


def ncu_stage1_geometry(ledger: pc.LedgerRoot, relative: str, log: pc.WarningLog) -> dict[str, Any] | None:
    ncu = shutil.which("ncu")
    if ncu is None:
        log.warn("ncu_unavailable", f"stage-1 geometry not read from {relative}")
        return None

    def read() -> dict[str, Any]:
        result = subprocess.run([ncu, "--import", str(ledger.path(relative)), "--page", "raw", "--csv",
                                 "--kernel-name", f"regex:{TQ_STAGE1}", "--launch-count", "1"],
                                check=True, capture_output=True, text=True, timeout=1200)
        rows = list(csv.reader(io.StringIO(result.stdout)))
        table = dict(zip(rows[0], rows[2]))
        return {"report": relative, "kernel": table.get("Kernel Name"), "grid": table.get("Grid Size"),
                "block": table.get("Block Size"),
                "achieved_occupancy_pct": table.get("sm__warps_active.avg.pct_of_peak_sustained_active"),
                "sm_throughput_pct": table.get("sm__throughput.avg.pct_of_peak_sustained_elapsed"),
                "source": "Nsight Compute report (profiler-instrumented; geometry only)"}

    return log.guard(f"ncu_geometry_{relative}", read)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--parent", type=Path, default=pc.ARTIFACT_PARENT)
    args = parser.parse_args(argv)
    log = pc.WarningLog()
    git_sha = pc.git_head()
    ledger = pc.LedgerRoot(PH15_DIR, PH15_ROOT)
    index = pq.read_table(ledger.path("nsys_run_index.parquet")).to_pylist()
    traces = {}
    for row in index:
        if row["status"] != "completed" or row["graph_mode"] != "cuda_graph":
            continue
        command = json.loads(ledger.path(f"runs/{row['run_id']}/command.json").read_text())
        argv_list = command["argv"]
        operations = int(argv_list[argv_list.index("--decode-operations") + 1])
        sqlite_relative = str(row["raw_report_path"]).replace(".nsys-rep", ".sqlite")
        summary = trace_summary(ledger.path(sqlite_relative), operations)
        summary.update({"run_id": row["run_id"], "method_config_id": row["method_config_id"],
                        "batch_size": int(row["batch_size"]), "historical_context": int(row["historical_context"]),
                        "trace": sqlite_relative})
        traces[row["run_id"]] = summary

    turboquant, kvquant = [], []
    for summary in sorted(traces.values(), key=lambda s: (s["method_config_id"], s["historical_context"])):
        if summary["method_config_id"].startswith("tq_"):
            stage1 = find(summary["symbols"], TQ_STAGE1)
            stage2 = find(summary["symbols"], TQ_STAGE2)
            geometry = stage1["geometries"][0]
            resident = SM_COUNT * MAX_WARPS_PER_SM
            turboquant.append({
                "method_config_id": summary["method_config_id"], "batch_size": summary["batch_size"],
                "historical_context": summary["historical_context"],
                "stage1_grid": geometry["grid"], "stage1_block": geometry["block"],
                "stage1_ctas": geometry["ctas"], "stage1_warps": geometry["ctas"] * geometry["warps_per_cta"],
                "stage1_warps_fraction_of_resident_capacity": geometry["ctas"] * geometry["warps_per_cta"] / resident,
                "stage1_ctas_per_sm": geometry["ctas"] / SM_COUNT,
                "stage1_launches_per_step": stage1["launches_per_step"],
                "stage1_share_of_kernel_time": stage1["share_of_kernel_time"],
                "stage1_kernel_ms_per_step": stage1["kernel_ms_per_step"],
                "stage2_grid": stage2["geometries"][0]["grid"] if stage2 else None,
                "stage2_share_of_kernel_time": stage2["share_of_kernel_time"] if stage2 else None,
                "kernel_ms_per_step": summary["kernel_ms_per_step_median"],
                "gpu_span_ms_per_step": summary["gpu_span_ms_per_step_median"],
                "source": "Nsight Systems Graph-mode trace (profiler-traced kernel durations)",
            })
        if summary["method_config_id"].startswith("kvq"):
            select = find(summary["symbols"], KVQ_SELECT)
            geometry = select["geometries"][0]
            kvquant.append({
                "method_config_id": summary["method_config_id"], "batch_size": summary["batch_size"],
                "historical_context": summary["historical_context"],
                "select_grid": geometry["grid"], "select_block": geometry["block"],
                "select_threads_per_cta": geometry["block"][0] * geometry["block"][1] * geometry["block"][2],
                "select_launches_per_step": select["launches_per_step"],
                "select_kernel_ms_per_step": select["kernel_ms_per_step"],
                "select_mean_us_per_launch": select["mean_us_per_launch"],
                "select_share_of_kernel_time": select["share_of_kernel_time"],
                "kernel_ms_per_step": summary["kernel_ms_per_step_median"],
                "gpu_span_ms_per_step": summary["gpu_span_ms_per_step_median"],
                "top_symbols": [{k: s[k] for k in ("symbol", "share_of_kernel_time", "kernel_ms_per_step",
                                                   "launches_per_step")} for s in summary["symbols"][:6]],
                "source": "Nsight Systems Graph-mode trace (profiler-traced kernel durations)",
            })

    ncu_geometry = [g for g in (
        ncu_stage1_geometry(ledger, f"raw/ncu/ncu-{c}-b1-l131072-cuda_graph-attempt2.ncu-rep", log)
        for c in ("tq_4bit_nc", "tq_k3v4_nc", "tq_3bit_nc")) if g]

    stage = pc.new_stage(pc.new_run_id("b-offline", git_sha), args.parent)
    run_id = stage.name.removeprefix(".staging-")
    pc.write_new(stage / "traces.json", pc.json_text(traces))
    pc.write_new(stage / "turboquant_geometry.json", pc.json_text({"nsys": turboquant, "ncu_128k": ncu_geometry,
                                                                   "sm_count": SM_COUNT,
                                                                   "max_resident_warps_per_sm": MAX_WARPS_PER_SM}))
    pc.write_new(stage / "kvquant_select_kernel.json", pc.json_text(kvquant))
    lines = ["# Part B offline: existing Phase 15 traces", "", f"> {pc.POSTHOC_LABEL}.", "",
             "Shares are of the traced GPU kernel time per decode step (Nsight Systems, Graph mode, B = 1); "
             "they are profiler observations, not benchmark timing.", "",
             "## TurboQuant stage 1", "",
             "| Config | L | grid | block | CTAs | warps (% of 9,024 resident) | launches/step | share | stage-1 ms/step |",
             "|---|---:|---|---|---:|---:|---:|---:|---:|"]
    for row in turboquant:
        lines.append(f"| {pc.CONFIG_LABELS[row['method_config_id']]} | {row['historical_context']} | "
                     f"{tuple(row['stage1_grid'])} | {tuple(row['stage1_block'])} | {row['stage1_ctas']} | "
                     f"{row['stage1_warps']} ({100 * row['stage1_warps_fraction_of_resident_capacity']:.1f}%) | "
                     f"{row['stage1_launches_per_step']:.0f} | {100 * row['stage1_share_of_kernel_time']:.1f}% | "
                     f"{row['stage1_kernel_ms_per_step']:.1f} |")
    lines += ["", "Nsight Compute geometry at 128K: " + "; ".join(
        f"{g['report'].split('/')[-1].split('-b1')[0].removeprefix('ncu-')}: grid {g['grid']}, block {g['block']}"
        for g in ncu_geometry), "", "## KVQuant SelectFixedOutliers1024Cap12Kernel", "",
        "| Config | L | grid | block | launches/step | ms/step | us/launch | share |",
        "|---|---:|---|---|---:|---:|---:|---:|"]
    for row in kvquant:
        lines.append(f"| {pc.CONFIG_LABELS[row['method_config_id']]} | {row['historical_context']} | "
                     f"{tuple(row['select_grid'])} | {tuple(row['select_block'])} | "
                     f"{row['select_launches_per_step']:.0f} | {row['select_kernel_ms_per_step']:.1f} | "
                     f"{row['select_mean_us_per_launch']:.0f} | {100 * row['select_share_of_kernel_time']:.1f}% |")
    lines.append("")
    pc.write_new(stage / "partB_offline_report.md", "\n".join(lines))
    for source in (Path(__file__), Path(pc.__file__)):
        pc.write_new(stage / "code" / source.name, source.read_bytes())
    manifest = {
        "schema_version": "kvbench-posthoc-review-manifest-1.0.0", "status": "PASS", "posthoc": True,
        "label": pc.POSTHOC_LABEL, "analysis": "review_round_part_b_offline", "created_at_utc": pc.utc_now(),
        "git_head": git_sha, "code_note": "analysis code is not committed at git_head; exact copies are in code/",
        "code_sha256": {p.name: pc.sha256_file(p) for p in (Path(__file__), Path(pc.__file__))},
        "environment": pc.environment_record(),
        "inputs": {"phase15_profiler": ledger.record()},
        "profiler_duration_is_normal_timing": False, "performance_claim_eligible": False,
        "warnings": log.entries,
    }
    final, root = pc.seal(stage, run_id, manifest, "posthoc_review_part_b_offline", args.parent)
    print(f"artifact: {final}\nroot_sha256: {root}\nwarnings: {len(log.entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
