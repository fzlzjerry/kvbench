#!/usr/bin/env python3
"""CPU-only supplementary calculations for the 2026-10-05 addendum (no new measurement).

    python3 analysis/addendum_20261005/supplement.py

Reads the addendum records (results/addendum-20261005, checked against its
SHA256SUMS, identical to the sealed R2 copy 57f8ff27...) and two sealed
post-hoc files (Part E S_adj, Part B split diagnostic, checked against their
own checksum ledgers).  Writes supplement.json, SUPPLEMENT.md and
appG_tables.tex next to this file.  The sealed results are not modified.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
R = REPO / "results" / "addendum-20261005"
POSTHOC = Path("/home/rockrock/cmu_paper/paper/posthoc/artifacts")
PART_E = POSTHOC / "posthoc-e-20261005t074653228884z-bbefea1c-dec6cf"
PART_B = POSTHOC / "posthoc-b-analysis-20261004t202531220352z-0641de4b-74462d"
sys.path.insert(0, str(REPO / "addendum"))

ALPHA_V2 = {"bf16": 1.0114151938896763, "k2v2": 1.2105694991479592, "k4v4": 1.1282050658644103,
            "tq_3bit_nc": 0.7667821100163751, "tq_4bit_nc": 0.8119153652807217, "tq_k3v4_nc": 0.79533061518971}
LABEL = {"tq_4bit_nc": "TQ-4bit", "tq_k3v4_nc": "TQ-k3v4", "tq_3bit_nc": "TQ-3bit",
         "k4v4": "KIVI-k4v4", "k2v2": "KIVI-k2v2", "bf16": "BF16"}
LK = {4096: "4K", 24576: "24K", 32768: "32K", 131072: "128K"}
HIST = {4096: 4096, 24576: 24576, 32768: 32768, 131072: 131071}
inputs: dict[str, str] = {}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ledger(path: Path) -> dict[str, str]:
    return {line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in path.read_text().splitlines() if line}


SUMS = ledger(R / "SHA256SUMS")


def addendum_file(rel: str) -> Path:
    path = R / rel
    digest = sha256(path)
    if SUMS.get(rel) != digest:
        raise SystemExit(f"{rel} differs from results SHA256SUMS")
    inputs[f"results/addendum-20261005/{rel}"] = digest
    return path


def sealed_file(root: Path, rel: str) -> Path:
    path = root / rel
    digest = sha256(path)
    if ledger(root / "checksums.sha256").get(rel) != digest:
        raise SystemExit(f"{root.name}/{rel} differs from its sealed ledger")
    inputs[f"{root.name}/{rel}"] = digest
    return path


def jsonl(rel: str) -> list[dict]:
    return [json.loads(line) for line in addendum_file(rel).read_text().splitlines() if line.strip()]


def pairwise(numerators: list[float], denominators: list[float]) -> tuple[float, float]:
    ratios = [a / b for a, b in itertools.product(numerators, denominators)]
    return min(ratios), max(ratios)


def ols(xs: list[float], ys: list[float]) -> tuple[float, float]:
    mx, my = statistics.mean(xs), statistics.mean(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - slope * mx, slope


def cache_bytes(task: str, point: dict) -> int:
    rel = f"{task}/raw/{point['run_ids'][0]}/run/result.json"
    return int(json.loads(addendum_file(rel).read_text())["runner"]["cache_accounting"]["allocated_bytes"])


def same_work(task: str) -> list[dict]:
    rows = jsonl(f"{task}/points.jsonl")
    bf16 = {(r["batch_size"], r["context_label"]): r for r in rows if r["configuration"] == "bf16"}
    out = []
    for r in rows:
        if r["configuration"] == "bf16":
            continue
        b = bf16[(r["batch_size"], r["context_label"])]
        low, high = pairwise(b["process_medians_ms"], r["process_medians_ms"])
        out.append({"configuration": r["configuration"], "variant": r["variant"], "batch_size": r["batch_size"],
                    "context_label": r["context_label"], "t_bf16_ms": b["point_median_ms"], "cv_bf16": b["cv"],
                    "t_method_ms": r["point_median_ms"], "cv_method": r["cv"],
                    "s": b["point_median_ms"] / r["point_median_ms"], "s_pairwise_min": low, "s_pairwise_max": high,
                    "bf16_process_medians_ms": b["process_medians_ms"],
                    "method_process_medians_ms": r["process_medians_ms"], "disposition": r["disposition"]})
    return out


def fits(task: str) -> dict:
    rows = [r for r in jsonl(f"{task}/points.jsonl") if r["disposition"] == "stable"]
    result = {}
    keys = sorted({(r["configuration"], r["batch_size"]) for r in rows})
    for configuration, batch in keys:
        pts = sorted((r for r in rows if (r["configuration"], r["batch_size"]) == (configuration, batch)),
                     key=lambda r: r["context_label"])
        if len(pts) < 2:
            continue
        xs = [ALPHA_V2[configuration] * cache_bytes(task, p) for p in pts]
        c0, slope = ols(xs, [p["point_median_ms"] for p in pts])
        result[f"{configuration}/B{batch}"] = {"c0_ms": c0, "bw_eff_gb_s": 1e-6 / slope,
                                              "contexts": [p["context_label"] for p in pts]}
    for key, value in result.items():
        base = result.get(f"bf16/B{key.split('/B')[1]}")
        value["eta"] = value["bw_eff_gb_s"] / base["bw_eff_gb_s"] if base else None
        value["c0_minus_bf16_ms"] = value["c0_ms"] - base["c0_ms"] if base else None
    return result


def main() -> None:
    out: dict = {"schema_version": "kvbench-addendum-20261005-supplement-1.0.0",
                 "note": "CPU-only recalculation from existing records; no new measurement."}

    # Task 1: TurboQuant 32 splits vs the sealed Nsight split diagnostic (Part B).
    tq = same_work("task1")
    traced = {}
    with sealed_file(PART_B, "split_diagnostic.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["split32_over_bf16_kernel"]:
                traced[(row["method_config_id"], int(row["batch_size"]), int(row["context_label"]))] = float(
                    row["split32_over_bf16_kernel"])
    for row in tq:
        row["wall_t_over_bf16"] = 1 / row["s"]
        row["traced_kernel_32_over_bf16"] = traced.get((row["configuration"], row["batch_size"], row["context_label"]))
    sweep = same_work("task1-sweep")
    frozen = {(p["method_config_id"], p["batch_size"], p["context_label"]): p["median_ms"]
              for p in json.loads(addendum_file("inputs/frozen_reference.json").read_text())["points"]}
    out["task1"] = {"points": tq, "sweep": sweep, "fits": fits("task1"),
                    "wall_over_bf16_range": [min(r["wall_t_over_bf16"] for r in tq), max(r["wall_t_over_bf16"] for r in tq)],
                    "traced_over_bf16_range": [min(r["traced_kernel_32_over_bf16"] for r in tq),
                                               max(r["traced_kernel_32_over_bf16"] for r in tq)],
                    "s_range": [min(r["s"] for r in tq), max(r["s"] for r in tq)],
                    "correctness": jsonl("task1/checks/comparisons.jsonl")}

    # Task 2: grouped KIVI (diagnostic) vs the post-hoc S_adj of Part E (sealed before the addendum timing).
    kivi = same_work("task2")
    s_adj = {}
    with sealed_file(PART_E, "e1_kivi_s_adj_points.csv").open() as handle:
        for row in csv.DictReader(handle):
            s_adj[(row["method_config_id"], int(row["batch_size"]), int(row["historical_context"]))] = float(row["s_adj"])
    part_e_manifest = json.loads(sealed_file(PART_E, "manifest.json").read_text())
    first_task2 = min(r["started_at_utc"] for r in jsonl("task2/processes.jsonl"))
    for row in kivi:
        row["s_adj_post_hoc"] = s_adj[(row["configuration"], row["batch_size"], HIST[row["context_label"]])]
        row["abs_error_vs_s_adj"] = abs(row["s"] - row["s_adj_post_hoc"])
        row["s_existing_frozen"] = (frozen[("bf16", row["batch_size"], row["context_label"])]
                                    / frozen[(row["configuration"], row["batch_size"], row["context_label"])])
    kivi_fits = fits("task2")
    best = max(kivi, key=lambda r: r["s"])
    out["task2"] = {"label": "gate_failed_diagnostic_only", "points": kivi, "fits": kivi_fits,
                    "best_point": {k: best[k] for k in ("configuration", "batch_size", "context_label", "s",
                                                        "s_pairwise_min", "s_pairwise_max", "cv_bf16", "cv_method")},
                    "points_with_s_above_1": [{k: r[k] for k in ("configuration", "batch_size", "context_label", "s",
                                                                 "s_pairwise_min", "s_pairwise_max")}
                                              for r in kivi if r["s"] > 1],
                    "max_abs_error_vs_s_adj": max(r["abs_error_vs_s_adj"] for r in kivi),
                    "s_adj_source": {"run_id": part_e_manifest["run_id"], "created_at_utc": part_e_manifest["created_at_utc"],
                                     "first_task2_process_started_at_utc": first_task2,
                                     "predictions_predate_measurement": part_e_manifest["created_at_utc"] < first_task2,
                                     "refit": False},
                    "c0_max_abs_diff_vs_bf16_ms": max(abs(v["c0_minus_bf16_ms"]) for k, v in kivi_fits.items()
                                                      if not k.startswith("bf16")),
                    "gate": jsonl("task2/checks/comparisons.jsonl")}

    # Task 3: vLLM FP8 vs BF16 (task3b) and the cache-halving comparison (nominal bytes).
    summary = json.loads(addendum_file("task3b/summary.json").read_text())["points"]
    step = {("fp8" if p["kv_cache_dtype"] == "fp8" else "bf16", p["batch_size"], p["context_label"]): p for p in summary}
    input_len = {32768: 32768, 131072: 131007}

    def nominal_bf16_bytes(batch: int, label: int) -> float:
        # K and V, 32 layers, 8 KV heads, head dim 128, 2 bytes; mean attended context of the
        # 64 differenced decode steps (input_len + 1 ... input_len + 64).
        return 2 * 32 * 8 * 128 * 2 * batch * (input_len[label] + 32.5)

    bf16_1_32, bf16_1_128 = step[("bf16", 1, 32768)]["step_ms"], step[("bf16", 1, 131072)]["step_ms"]
    bw_bf16 = (nominal_bf16_bytes(1, 131072) - nominal_bf16_bytes(1, 32768)) / ((bf16_1_128 - bf16_1_32) / 1e3)
    fp8_rows = []
    for batch, label in ((1, 32768), (1, 131072), (8, 32768)):
        b, f = step[("bf16", batch, label)], step[("fp8", batch, label)]
        predicted = b["step_ms"] - nominal_bf16_bytes(batch, label) / 2 / bw_bf16 * 1e3
        low, high = pairwise(b["step_ms_per_replicate"], f["step_ms_per_replicate"])
        fp8_rows.append({"batch_size": batch, "context_label": label, "input_len": input_len[label],
                         "bf16_step_ms": b["step_ms"], "bf16_cv": b["cv"], "fp8_step_ms": f["step_ms"], "fp8_cv": f["cv"],
                         "bf16_over_fp8": b["step_ms"] / f["step_ms"], "bf16_over_fp8_pairwise_min": low,
                         "bf16_over_fp8_pairwise_max": high, "fp8_predicted_cache_halved_ms": predicted,
                         "fp8_measured_over_predicted": f["step_ms"] / predicted,
                         "backends": {"bf16": b["attention_backends"], "fp8": f["attention_backends"]}})
    out["task3"] = {"run": "task3b (prefix caching on)", "points": fp8_rows,
                    "cache_halving_model": {
                        "formula": "T_pred,FP8(B,L) = T_BF16(B,L) - D_BF16(B,L) / (2 * BW_BF16)",
                        "D_BF16": "2 * 32 * 8 * 128 * 2 bytes * B * (input_len + 32.5): nominal BF16 K/V bytes per step",
                        "BW_BF16_nominal_gb_s": bw_bf16 / 1e9,
                        "BW_BF16_source": "two-point fit of the same vLLM BF16 baseline at B=1, 32K and 128K",
                        "assumptions": ["same non-cache time for both KV dtypes (same vLLM BF16 baseline)",
                                        "FP8 reads exactly half of the BF16 nominal bytes (per-tensor scales ignored)",
                                        "the B=1 nominal bandwidth also applies at B=8",
                                        "nominal bytes, not measured DRAM traffic; BW is a fitted parameter, not a hardware counter"],
                        "ratio_direction": "measured/predicted > 1: FP8 slower than the cache-halving prediction"}}

    # Task 4: synthetic-cache feasibility and throughput at 128K.
    import feasibility
    table = feasibility.compute()
    predicted = {(r["method_config_id"], r["batch_size"]): r for r in table["rows"]}
    t4 = []
    for p in jsonl("task4-bmax/points.jsonl"):
        memories = []
        for run in p["run_ids"]:
            worker = json.loads(addendum_file(f"task4-bmax/raw/{run}/worker_result.json").read_text())
            memories.append(worker["memory_observed"])
        steady = predicted[(p["configuration"], p["batch_size"])]
        t4.append({"configuration": p["configuration"], "batch_size": p["batch_size"], "context_label": p["context_label"],
                   "step_ms": p["point_median_ms"], "cv": p["cv"], "process_medians_ms": p["process_medians_ms"],
                   "tokens_per_s": p["batch_size"] / (p["point_median_ms"] / 1e3),
                   "steady_state_predicted_bytes": steady["steady_state_bytes"],
                   "observed_peak_reserved_bytes_max": max(m["timing_after"]["peak_reserved_bytes"] for m in memories),
                   "observed_allocated_after_setup_bytes": max(m["post_setup"]["allocated_bytes"] for m in memories),
                   "observed_scope": "PyTorch caching allocator in the worker process (model weights, cache, workspaces, "
                                     "CUDA Graph pool); excludes CUDA context and driver memory"})
    by = {r["configuration"]: r for r in t4}
    kivi_tps = [by["k4v4"]["batch_size"] / (m / 1e3) for m in by["k4v4"]["process_medians_ms"]]
    bf16_tps = [by["bf16"]["batch_size"] / (m / 1e3) for m in by["bf16"]["process_medians_ms"]]
    low, high = pairwise(kivi_tps, bf16_tps)
    out["task4"] = {"points": t4, "kivi_over_bf16_tokens_per_s": by["k4v4"]["tokens_per_s"] / by["bf16"]["tokens_per_s"],
                    "kivi_over_bf16_pairwise": [low, high], "b_max": table["b_max"], "limit_bytes": table["limit_bytes"],
                    "gate": json.loads(addendum_file(sorted(p.relative_to(R).as_posix()
                                                            for p in (R / "task4-gate").glob("gate-*.json"))[-1]).read_text()),
                    "scope": "full Llama-3.1-8B decode step with BF16 weights resident, CUDA Graph replay of one decode "
                             "step at fixed L; cache filled with synthetic K/V through the method's store path; no "
                             "prefill, no request-level serving"}
    out["gpu_state_after_addendum"] = "persistence mode restored to Disabled on 2026-10-06 (~09:10 CST), its state before the addendum"
    out["inputs_sha256"] = inputs
    (HERE / "supplement.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    write_tables(out)
    write_markdown(out)
    print(json.dumps({"task1_wall_over_bf16": out["task1"]["wall_over_bf16_range"],
                      "task2_best": out["task2"]["best_point"], "task2_max_abs_err": out["task2"]["max_abs_error_vs_s_adj"],
                      "task2_c0_max_diff": out["task2"]["c0_max_abs_diff_vs_bf16_ms"],
                      "task3": [(r["bf16_over_fp8"], r["bf16_over_fp8_pairwise_min"], r["bf16_over_fp8_pairwise_max"],
                                 r["fp8_measured_over_predicted"]) for r in fp8_rows],
                      "task3_bw": bw_bf16 / 1e9, "task4": out["task4"]["kivi_over_bf16_tokens_per_s"],
                      "task4_pairwise": out["task4"]["kivi_over_bf16_pairwise"]}, indent=1))


def f(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}"


def write_tables(out: dict) -> None:
    lines = ["% Generated by analysis/addendum_20261005/supplement.py; do not edit by hand.", ""]
    lines += ["% Table: TurboQuant 32 splits", ]
    for r in sorted(out["task1"]["points"], key=lambda r: (r["configuration"] != "tq_4bit_nc", r["configuration"] != "tq_k3v4_nc",
                                                          r["batch_size"], r["context_label"])):
        lines.append(f"{LABEL[r['configuration']]} & {r['batch_size']} & {LK[r['context_label']]} & {f(r['t_bf16_ms'], 2)} & "
                     f"{f(r['t_method_ms'], 2)} & {f(r['s'])} & {f(r['s_pairwise_min'])}--{f(r['s_pairwise_max'])} & "
                     f"{f(r['wall_t_over_bf16'], 2)} & {f(r['traced_kernel_32_over_bf16'], 2)} \\\\")
    lines.append("% sweep (TQ-k3v4, B = 1, 128K)")
    for r in sorted(out["task1"]["sweep"], key=lambda r: r["variant"]["tq_splits"]):
        lines.append(f"TQ-k3v4, {r['variant']['tq_splits']} splits & 1 & 128K & {f(r['t_bf16_ms'], 2)} & {f(r['t_method_ms'], 2)} & "
                     f"{f(r['s'])} & {f(r['s_pairwise_min'])}--{f(r['s_pairwise_max'])} & {f(1 / r['s'], 2)} & --- \\\\")
    lines += ["", "% Table: grouped KIVI (diagnostic)"]
    for r in sorted(out["task2"]["points"], key=lambda r: (r["configuration"] != "k4v4", r["batch_size"], r["context_label"])):
        lines.append(f"{LABEL[r['configuration']]} & {r['batch_size']} & {LK[r['context_label']]} & "
                     f"{f(r['t_bf16_ms'], 2)} & {f(r['t_method_ms'], 2)} & {f(r['s'])} & "
                     f"{f(r['s_pairwise_min'])}--{f(r['s_pairwise_max'])} & {f(r['s_existing_frozen'])} & "
                     f"{f(r['s_adj_post_hoc'])} & {f(r['abs_error_vs_s_adj'])} \\\\")
    lines += ["", "% Table: vLLM FP8"]
    for r in out["task3"]["points"]:
        lines.append(f"{r['batch_size']} & {LK[r['context_label']]} & {f(r['bf16_step_ms'], 2)} & {f(r['fp8_step_ms'], 2)} & "
                     f"{f(r['bf16_over_fp8'], 2)} & {f(r['bf16_over_fp8_pairwise_min'], 2)}--{f(r['bf16_over_fp8_pairwise_max'], 2)} & "
                     f"{f(r['fp8_predicted_cache_halved_ms'], 2)} & {f(r['fp8_measured_over_predicted'])} \\\\")
    lines += ["", "% Table: fixed-memory throughput (synthetic cache)"]
    for r in sorted(out["task4"]["points"], key=lambda r: r["configuration"]):
        lines.append(f"{LABEL[r['configuration']]} & {r['batch_size']} & {f(r['steady_state_predicted_bytes'] / 1e9, 1)} & "
                     f"{f(r['observed_peak_reserved_bytes_max'] / 1e9, 1)} & {f(r['step_ms'], 2)} & {f(r['tokens_per_s'], 1)} \\\\")
    (HERE / "appG_tables.tex").write_text("\n".join(lines) + "\n")


def write_markdown(out: dict) -> None:
    t2, t3, t4 = out["task2"], out["task3"], out["task4"]
    text = f"""# Addendum 2026-10-05: supplementary CPU calculations

No new measurement. Inputs (SHA-256 in supplement.json) are the addendum records, checked against
results/addendum-20261005/SHA256SUMS (= sealed R2 root 57f8ff27...), and two sealed post-hoc files
(Part E S_adj, Part B split diagnostic) checked against their ledgers.

- Task 1: wall-clock T_TQ/T_BF16 with 32 splits {f(out['task1']['wall_over_bf16_range'][0])}--{f(out['task1']['wall_over_bf16_range'][1])}
  (traced kernel time, Part B: {f(out['task1']['traced_over_bf16_range'][0])}--{f(out['task1']['traced_over_bf16_range'][1])}).
- Task 2 (gate failed; diagnostic): best S = {f(t2['best_point']['s'])} ({t2['best_point']['configuration']}, B={t2['best_point']['batch_size']},
  {LK[t2['best_point']['context_label']]}); the nine pairwise ratios of process medians span
  {f(t2['best_point']['s_pairwise_min'])}--{f(t2['best_point']['s_pairwise_max'])}. Max |S - S_adj| = {f(t2['max_abs_error_vs_s_adj'])};
  S_adj from Part E ({t2['s_adj_source']['created_at_utc']}), before the first Task 2 process
  ({t2['s_adj_source']['first_task2_process_started_at_utc']}); not refitted. Max |c0 - c0,BF16| = {f(t2['c0_max_abs_diff_vs_bf16_ms'], 2)} ms.
- Task 3: {t3['cache_halving_model']['formula']}; BW_BF16 = {f(t3['cache_halving_model']['BW_BF16_nominal_gb_s'], 0)} GB/s (nominal,
  fitted); measured/predicted FP8 step = {', '.join(f(r['fp8_measured_over_predicted']) for r in t3['points'])}
  ({t3['cache_halving_model']['ratio_direction']}).
- Task 4: KIVI-k4v4 (B=9) / BF16 (B=3) tokens/s = {f(t4['kivi_over_bf16_tokens_per_s'])} (pairwise {f(t4['kivi_over_bf16_pairwise'][0])}--{f(t4['kivi_over_bf16_pairwise'][1])});
  scope: {t4['scope']}.
- GPU state: {out['gpu_state_after_addendum']}.
"""
    (HERE / "SUPPLEMENT.md").write_text(text)


if __name__ == "__main__":
    main()
