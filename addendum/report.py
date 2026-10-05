#!/usr/bin/env python3
"""Build results/addendum-20261005/REPORT.md from the append-only records.

    python3 addendum/report.py            # writes REPORT.md (+ report-data.json)

Reads only files under results/addendum-20261005/ and the Part A alpha values
(post hoc, sealed root 3dc84c64...).  Ratios use same-run BF16 only.
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import statistics
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

R = c.RESULTS
# Part A traffic model, alpha_v2 = measured cache-path DRAM bytes per allocated cache
# byte (Nsight Compute, as-ported kernels; sealed post-hoc root 3dc84c64...).
ALPHA_V2 = {"bf16": 1.0114151938896763, "k2v2": 1.2105694991479592, "k4v4": 1.1282050658644103,
            "tq_3bit_nc": 0.7667821100163751, "tq_4bit_nc": 0.8119153652807217,
            "tq_k3v4_nc": 0.79533061518971}
LABEL_K = {32768: "32K", 131072: "128K", 4096: "4K", 24576: "24K"}


def points(task: str) -> list[dict[str, Any]]:
    rows = c.read_jsonl(R / task / "points.jsonl")
    latest: dict[tuple, dict[str, Any]] = {}
    for row in rows:  # append-only: the last row of a point wins
        latest[(row.get("block", row["configuration"]), json.dumps(row["variant"], sort_keys=True),
                row["batch_size"], row["context_label"])] = row
    return list(latest.values())


def cache_bytes(task: str, row: dict[str, Any]) -> int:
    run = row["run_ids"][0]
    payload = json.loads((R / task / "raw" / run / "run" / "result.json").read_text())
    accounting = payload["runner"].get("cache_accounting") or {}
    return int(accounting["allocated_bytes"])


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def ols(xs: list[float], ys: list[float]) -> tuple[float, float]:
    mx, my = statistics.mean(xs), statistics.mean(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - slope * mx, slope


def same_work_table(task: str, lines: list[str], label: str | None = None) -> list[dict[str, Any]]:
    rows = points(task)
    bf16 = {(r["batch_size"], r["context_label"]): r for r in rows if r["configuration"] == "bf16"}
    out = []
    lines += ["| Configuration | Variant | B | L | T_BF16 (ms) | CV_BF16 | T_method (ms) | CV_method | S = T_BF16/T | Status |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted((r for r in rows if r["configuration"] != "bf16"),
                    key=lambda r: (r["label"], json.dumps(r["variant"]), r["batch_size"], r["context_label"])):
        b = bf16.get((r["batch_size"], r["context_label"]))
        s = b["point_median_ms"] / r["point_median_ms"] if b else None
        variant = ", ".join(f"{k}={v}" for k, v in r["variant"].items()) or "existing"
        status = r["disposition"] + (" (unstable)" if r["unstable"] else "")
        if label:
            status += f"; {label}"
        lines.append(f"| {r['label']} | {variant} | {r['batch_size']} | {LABEL_K.get(r['context_label'], r['context_label'])} | "
                     f"{fmt(b and b['point_median_ms'])} | {fmt(b and 100 * b['cv'], 2)}% | {fmt(r['point_median_ms'])} | "
                     f"{fmt(100 * r['cv'], 2)}% | {fmt(s)} | {status} |")
        out.append({**r, "bf16_point_median_ms": b and b["point_median_ms"], "bf16_cv": b and b["cv"], "S": s})
    return out


def eta_table(task: str, lines: list[str], note: str) -> list[dict[str, Any]]:
    rows = [r for r in points(task) if r["disposition"] in ("stable", "unstable")]
    fits: dict[tuple, dict[str, Any]] = {}
    for key in sorted({(r["configuration"], json.dumps(r["variant"], sort_keys=True), r["batch_size"]) for r in rows}):
        configuration, variant, batch = key
        pts = sorted((r for r in rows if (r["configuration"], json.dumps(r["variant"], sort_keys=True), r["batch_size"]) == key),
                     key=lambda r: r["context_label"])
        if len(pts) < 2:
            continue
        xs = [ALPHA_V2[configuration] * cache_bytes(task, p) for p in pts]
        ys = [p["point_median_ms"] for p in pts]
        c0, slope = ols(xs, ys)  # ms per byte
        fits[key] = {"configuration": configuration, "variant": json.loads(variant), "batch_size": batch,
                     "contexts": [p["context_label"] for p in pts], "c0_ms": c0,
                     "bw_eff_gb_s": (1e-6 / slope) if slope > 0 else None}
    lines += ["| Configuration | Variant | B | Contexts | c0 (ms) | BW_eff (GB/s) | eta = BW_eff / BW_eff,BF16 |",
              "|---|---|---|---|---|---|---|"]
    out = []
    for key, fit in fits.items():
        if fit["configuration"] == "bf16":
            continue
        base = fits.get(("bf16", "{}", fit["batch_size"]))
        eta = (fit["bw_eff_gb_s"] / base["bw_eff_gb_s"]) if base and fit["bw_eff_gb_s"] and base["bw_eff_gb_s"] else None
        fit["eta"] = eta
        variant = ", ".join(f"{k}={v}" for k, v in fit["variant"].items()) or "existing"
        lines.append(f"| {c.LABELS.get(fit['configuration'])} | {variant} | {fit['batch_size']} | "
                     f"{', '.join(LABEL_K.get(x, str(x)) for x in fit['contexts'])} | {fmt(fit['c0_ms'], 2)} | "
                     f"{fmt(fit['bw_eff_gb_s'], 1)} | {fmt(eta)} |")
        out.append(fit)
    for key, fit in fits.items():
        if fit["configuration"] == "bf16":
            lines.append(f"| BF16 | — | {fit['batch_size']} | {', '.join(LABEL_K.get(x, str(x)) for x in fit['contexts'])} | "
                         f"{fmt(fit['c0_ms'], 2)} | {fmt(fit['bw_eff_gb_s'], 1)} | 1 |")
            out.append(fit)
    lines += ["", note]
    return out


def nsys_summary(name: str) -> dict[str, Any] | None:
    path = R / "task2" / "nsys" / name / "report.sqlite"
    if not path.exists():
        return None
    db = sqlite3.connect(path)
    rows = db.execute("select s.value, k.end - k.start from CUPTI_ACTIVITY_KIND_KERNEL k "
                      "join StringIds s on s.id = k.shortName").fetchall()
    replays = json.loads((R / "task2" / "nsys" / name / "status.json").read_text())["replays"]
    per: dict[str, list[float]] = {}
    for kernel, duration in rows:
        per.setdefault(kernel, [0, 0])
        per[kernel][0] += 1
        per[kernel][1] += duration
    return {"kernels_per_step": len(rows) / replays, "kernel_ms_per_step": sum(d for _, d in rows) / replays / 1e6,
            "by_kernel": {k: {"count_per_step": v[0] / replays, "ms_per_step": v[1] / replays / 1e6}
                          for k, v in sorted(per.items(), key=lambda kv: -kv[1][1])}}


def main() -> None:
    lines: list[str] = []
    data: dict[str, Any] = {"generated_at_utc": c.utc_now()}
    lines += ["# Addendum 2026-10-05: results", "",
              "Post-hoc addendum to \"Bytes Are Not Latency\" (preregistered amendment `docs/amendment-20261005.md`, "
              "committed before any addendum timing; decisions after gate results are in its Section 10). "
              "No frozen root or table was changed. Same-work speedups and fixed-memory throughput are reported "
              "separately. All latencies are host wall-clock, CUDA Graph, fixed L, single decode step: 64 warmup "
              "replays, then 5 batches x 256 replays per process; process median = median of the 5 batch values; "
              "point = median of 3 process medians; CV = sample SD / mean of the process medians; "
              "CV > 3 % = unstable (none were rerun). BF16 was remeasured in the same run for every ratio.", ""]
    pre = json.loads((R / "gpu-state" / "preflight-initial.json").read_text())
    lines += ["## Identity", "",
              f"- Measurement container: `{c.IMAGE}`; execution repository `{c.EXECUTION_SHA}` (Full Scan timing commit) "
              "mounted read-only; addendum code on branch `addendum-20261005` (per-row `addendum_git_sha`).",
              "- Model: meta-llama/Llama-3.1-8B-Instruct @ 0e9e39f249a16976918f6564b8830bc894c89659.",
              f"- GPU {c.GPU_UUID}; driver/persistence/clocks at start: `{pre['gpu_state']['query_csv'].splitlines()[-1]}`.",
              "- Clocks at driver defaults (not locked), as in the Full Scan; per-process SM clock ranges are in the raw records.", ""]

    # Task status
    lines += ["## Task status", "",
              "| Task | Status |", "|---|---|"]
    status_rows = []
    for task, text in (("task1", "Task 1 TurboQuant 32 splits"), ("task1-sweep", "Task 1 split sweep"),
                       ("task2", "Task 2 KIVI grouped residual (gate FAILED; diagnostic timing)"),
                       ("task3", "Task 3 vLLM FP8 positive control"), ("task4-gate", "Task 4 synthetic-cache gate"),
                       ("task4-bmax", "Task 4 fixed-memory throughput")):
        sessions = c.read_jsonl(R / task / "driver-sessions.jsonl")
        procs = c.read_jsonl(R / task / "processes.jsonl")
        done = sum(1 for p in procs if p["status"] == "completed")
        status_rows.append((text, sessions[-1] if sessions else None, done))
        last = sessions[-1] if sessions else None
        state = ("not run" if not procs else
                 f"{done} completed processes" + (f"; stopped: {last['stopped']}" if last and last.get("stopped") else ""))
        lines.append(f"| {text} | {state} |")
    lines += ["", "Failures, caps and decisions: see `FAILURES.md` (copied at the end).", ""]

    # Task 1
    lines += ["## Task 1: TurboQuant with 32 KV splits", ""]
    data["task1"] = same_work_table("task1", lines)
    lines += ["", "Split sweep, TQ-k3v4, B = 1, 128K (same-run BF16):", ""]
    data["task1_sweep"] = same_work_table("task1-sweep", lines)
    lines += ["", "Effective cache-path bandwidth (fit T = c0 + D_cache / BW_eff over the addendum contexts at fixed B):", ""]
    data["task1_eta"] = eta_table("task1", lines,
                                  "D_cache = alpha_v2 x allocated cache bytes (Part A traffic model; alpha measured on the "
                                  "as-ported 4-split kernels). Modeled, not measured, HBM traffic; two contexts per fit.")
    comparisons = c.read_jsonl(R / "task1" / "checks" / "comparisons.jsonl")
    lines += ["", "Correctness at (B = 1, 4K), frozen Q0 core-l4096 probe, eager growing cache, 100 greedy tokens:", "",
              "| Reference | Candidate | Step-0 max abs logit diff | Max abs diff before divergence | Agreeing positions | First divergence |",
              "|---|---|---|---|---|---|"]
    for row in comparisons:
        lines.append(f"| {row['reference']} | {row['candidate']} | {fmt(row['max_abs_logit_diff_step0'], 4)} | "
                     f"{fmt(row['max_abs_logit_diff_same_input_steps'], 4)} | {row['agreeing_positions']}/{row['steps']} | "
                     f"{fmt(row['first_divergence'])} |")
    lines += ["", "(BF16 rows are supplementary context, not preregistered.) Compute Sanitizer memcheck on the "
              "split-32 path: 0 errors, filtered to the TurboQuant decode kernels and unfiltered.", ""]

    # Task 2
    lines += ["## Task 2: KIVI with KV-head-grouped residual (gate FAILED; timing is diagnostic only)", ""]
    comparisons2 = c.read_jsonl(R / "task2" / "checks" / "comparisons.jsonl")
    lines += ["Gate 1 (100 identical greedy tokens versus the existing adapter, B = 1, 4K): **FAIL**.", "",
              "| Reference | Candidate | Step-0 max abs logit diff | Agreeing positions | First divergence |",
              "|---|---|---|---|---|"]
    for row in comparisons2:
        lines.append(f"| {row['reference']} | {row['candidate']} | {fmt(row['max_abs_logit_diff_step0'], 4)} | "
                     f"{row['agreeing_positions']}/{row['steps']} | {fmt(row['first_divergence'])} |")
    gates = {}
    for name in ("t2-gate-k4v4", "t2-gate-k2v2", "t2-gate-k4v4-control", "t2-gate-k2v2-control"):
        path = R / "task2" / "checks" / name / "worker_result.json"
        if path.exists():
            result = json.loads(path.read_text())
            gates[name] = {"gates_pass": result.get("gates_pass"), "components": result.get("gates_pass_components"),
                           "native_gqa_literal": result.get("native_gqa_literal", {}).get("pass")}
    data["task2_gates"] = gates
    lines += ["", "Other gates (grouped adapter; `-control` = existing adapter through the same harness):", "",
              "| Check | G3/G4 + G1 + graph tests (semantic native_gqa) | Literal native_gqa |", "|---|---|---|"]
    for name, g in gates.items():
        lines.append(f"| {name} | {g['gates_pass']} | {g['native_gqa_literal']} |")
    lines += ["", "Diagnostic timing (label `gate_failed_diagnostic_only`; not an admitted result):", ""]
    data["task2"] = same_work_table("task2", lines, "diagnostic only")
    lines += ["", "Effective cache-path bandwidth (diagnostic):", ""]
    data["task2_eta"] = eta_table("task2", lines,
                                  "D_cache from the Part A traffic model (alpha_v2 of the existing KIVI kernels; the "
                                  "quantized-history kernels are unchanged). Three contexts per fit.")
    lines += ["", "Nsight Systems at (B = 1, 4K), 8 graph replays (profiler kernel time, not wall-clock):", "",
              "| Trace | Kernels per step | Kernel time per step (ms) |", "|---|---|---|"]
    data["task2_nsys"] = {}
    for name in ("t2-nsys-k4v4-orig", "t2-nsys-k4v4-grouped", "t2-nsys-k2v2-orig", "t2-nsys-k2v2-grouped"):
        summary = nsys_summary(name)
        data["task2_nsys"][name] = summary
        if summary:
            lines.append(f"| {name} | {summary['kernels_per_step']:.0f} | {summary['kernel_ms_per_step']:.3f} |")

    # Task 3
    lines += ["", "## Task 3: vLLM FP8 KV-cache positive control", ""]
    summary_path = R / "task3" / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        data["task3"] = summary
        manifest = json.loads((R / "task3" / "manifest.json").read_text())
        lines += [f"vLLM/torch/CUDA: `{manifest.get('versions')}`; engine settings: {manifest.get('engine_settings')}.", "",
                  "| KV dtype | B | L (input_len) | Step (ms) | CV | n | Attention backend(s) | FP8 / BF16 step | BF16 / FP8 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for row in summary["points"]:
            lines.append(f"| {row['kv_cache_dtype']} | {row['batch_size']} | {LABEL_K.get(row['context_label'])} ({row['input_len']}) | "
                         f"{fmt(row['step_ms'])} | {fmt(row['cv'] and 100 * row['cv'], 2)}% | {row['n']} | "
                         f"{', '.join(row['attention_backends']) or '—'} | {fmt(row.get('fp8_over_bf16'))} | "
                         f"{fmt(row.get('speedup_bf16_over_fp8'))} |")
    else:
        lines += ["Not measured (see FAILURES.md)."]

    # Task 4
    lines += ["", "## Task 4: fixed-memory throughput (synthetic cache, timing only)", ""]
    gate_files = sorted((R / "task4-gate").glob("gate-*.json"))
    if gate_files:
        gate = json.loads(gate_files[-1].read_text())
        data["task4_gate"] = gate
        lines += [f"Synthetic-cache validity gate at (B = 1, 32K): **{'PASS' if gate['pass'] else 'FAIL'}** "
                  f"({gate['rule']}).", "", "| Configuration | T_real (ms) | T_synthetic (ms) | Relative difference | Threshold | Pass |",
                  "|---|---|---|---|---|---|"]
        for name, entry in gate["configurations"].items():
            lines.append(f"| {c.LABELS.get(name)} | {fmt(entry['real'] and entry['real']['point_median_ms'])} | "
                         f"{fmt(entry['synthetic'] and entry['synthetic']['point_median_ms'])} | "
                         f"{fmt(entry.get('relative_difference') and 100 * entry['relative_difference'], 2)}% | "
                         f"{fmt(entry.get('threshold') and 100 * entry['threshold'], 2)}% | {entry['pass']} |")
    bmax_rows = points("task4-bmax")
    if bmax_rows:
        lines += ["", "L = 128K at B_max (steady-state formula, limit 89,733,904,465 bytes; addendum/feasibility.py). "
                  "**Synthetic cache, timing only.**", "",
                  "| Configuration | B_max | Step (ms) | CV | tokens/s = B / step | Status |", "|---|---|---|---|---|---|"]
        data["task4_bmax"] = []
        for r in sorted(bmax_rows, key=lambda r: r["configuration"]):
            tps = r["batch_size"] / (r["point_median_ms"] / 1e3) if r["point_median_ms"] else None
            data["task4_bmax"].append({**r, "tokens_per_s": tps})
            lines.append(f"| {r['label']} | {r['batch_size']} | {fmt(r['point_median_ms'])} | {fmt(100 * r['cv'], 2)}% | "
                         f"{fmt(tps, 1)} | {r['disposition']} |")
    # Failures + checksums
    failures = R / "FAILURES.md"
    lines += ["", "## FAILURES.md (verbatim)", "", failures.read_text() if failures.exists() else "(none)", "",
              "## Files and checksums", "",
              "SHA-256 of every file under results/addendum-20261005/ except REPORT.md, report-data.json and "
              "SHA256SUMS are in `SHA256SUMS` (generated with this report).", ""]
    (R / "report-data.json").write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")
    (R / "REPORT.md").write_text("\n".join(lines) + "\n")
    sums = []
    for path in sorted(p for p in R.rglob("*") if p.is_file()):
        rel = path.relative_to(R).as_posix()
        if rel in ("REPORT.md", "SHA256SUMS"):
            continue
        sums.append(f"{c.sha256_file(path)}  {rel}")
    (R / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(f"wrote {R / 'REPORT.md'} ({len(lines)} lines), {len(sums)} checksummed files")


if __name__ == "__main__":
    main()
