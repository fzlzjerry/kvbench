#!/usr/bin/env python3
"""Task 3: FP8 KV-cache positive control with vLLM's own latency benchmark.

    sudo python3 addendum/vllm_fp8.py run --venv /home/rockrock/addendum-vllm-env
    python3 addendum/vllm_fp8.py summarize

Runs `vllm bench latency` from a separate virtual environment (the measurement
container is not used or modified), same model and revision, CUDA Graphs on
(no --enforce-eager), every other engine setting at vLLM's defaults.  For each
(KV dtype, point, output_len in {1, 65}) three separate processes run in
block-randomized rounds; T = median iteration latency of a process; the
per-step decode time of replicate i is (T65_i - T1_i) / 64; the point value is
the median over replicates and CV is taken over the three differences.
Logs are kept whole; the attention backend, KV-cache dtype and vLLM version
are extracted from them.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

TASK = "task3"
ROOT = c.RESULTS / TASK
SNAPSHOT = c.MODEL / "snapshots" / "0e9e39f249a16976918f6564b8830bc894c89659"
DTYPES = ("auto", "fp8")  # auto = model dtype (BF16)
# (batch, label, input_len): input + 65 must not exceed the model's 131,072 positions.
POINTS = ((1, 32768, 32768), (1, 131072, 131007), (8, 32768, 32768))
OUTPUT_LENS = (1, 65)
WARMUP_ITERS = 5
ITERS = 15
CAP_SECONDS = 3 * 3600 - 45 * 60  # Task 3 cap (3 h) minus the installation budget
# Task 3 started with the first installation attempt at 2026-10-05 20:05:23 CST;
# its 3 h cap (installation included) ends at 23:05:23 CST = 15:05:23Z.
TASK3_DEADLINE_EPOCH = 1791213923


def job_order() -> list[dict[str, Any]]:
    jobs = []
    for round_index in range(c.PROCESSES_PER_POINT):
        seed = 2026100500 + 30 + round_index
        rng = random.Random(seed)
        blocks = list(DTYPES)
        rng.shuffle(blocks)
        order = 0
        for dtype in blocks:
            cells = [(b, l, i, o) for (b, l, i) in POINTS for o in OUTPUT_LENS]
            rng.shuffle(cells)
            for batch, label, input_len, output_len in cells:
                jobs.append({"task": TASK, "round": round_index, "seed": seed, "order_index": order,
                             "kv_cache_dtype": dtype, "batch_size": batch, "context_label": label,
                             "input_len": input_len, "output_len": output_len,
                             "run_id": f"{TASK}-r{round_index}-o{order:03d}-{dtype}-b{batch}-l{label}-out{output_len}"})
                order += 1
    return jobs


BACKEND_PATTERNS = (
    re.compile(r"Using ([A-Za-z0-9_.]+) (?:attention )?backend", re.I),
    re.compile(r"attn(?:ention)?[_ ]backend[=: ]+([A-Za-z0-9_.]+)", re.I),
    re.compile(r"Using AttentionBackendEnum\.([A-Z0-9_]+)", re.I),
)


def log_facts(text: str) -> dict[str, Any]:
    backends = sorted({m.group(1) for p in BACKEND_PATTERNS for m in p.finditer(text)})
    kv_lines = sorted({line.strip()[:300] for line in text.splitlines()
                       if re.search(r"kv_cache_dtype|fp8 data type|kv cache dtype|fp8_e4m3|fp8_e5m2", line, re.I)})
    fallback_lines = sorted({line.strip()[:300] for line in text.splitlines()
                             if re.search(r"fall(ing)? ?back|not supported|unsupported|eager mode|enforce_eager=True",
                                          line, re.I)})
    graph_lines = sorted({line.strip()[:300] for line in text.splitlines()
                          if re.search(r"cuda ?graph|cudagraph", line, re.I)})[:20]
    version = re.search(r"\(v(\d+\.\d+\.\d+[^\s)]*)\)", text)
    return {"attention_backends": backends, "kv_dtype_lines": kv_lines[:20],
            "fallback_lines": fallback_lines[:20], "cuda_graph_lines": graph_lines,
            "vllm_version_in_log": version.group(1) if version else None}


def run(venv: Path) -> None:
    if os.geteuid() != 0:
        raise SystemExit("run as root (model snapshot is root-only)")
    ROOT.mkdir(parents=True, exist_ok=True)
    jobs = job_order()
    manifest_path = ROOT / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if [j["run_id"] for j in manifest["jobs"]] != [j["run_id"] for j in jobs]:
            raise SystemExit("job order differs from the recorded manifest")
    else:
        version = subprocess.run([str(venv / "bin/python"), "-c", "import vllm, torch; print(vllm.__version__, torch.__version__, torch.version.cuda)"],
                                 capture_output=True, text=True).stdout.strip()
        manifest = {"addendum_id": c.ADDENDUM_ID, "task": TASK, "amendment": c.AMENDMENT,
                    "started_at_utc": c.utc_now(), "started_at_epoch": time.time(),
                    "cap_seconds": CAP_SECONDS, "venv": str(venv), "versions": version,
                    "model_snapshot": str(SNAPSHOT), "warmup_iters": WARMUP_ITERS, "iters": ITERS,
                    "engine_settings": "vLLM defaults except --kv-cache-dtype; CUDA Graphs on (no --enforce-eager)",
                    "jobs": jobs}
        c.write_new(manifest_path, c.json_text(manifest))
    deadline = min(manifest["started_at_epoch"] + manifest["cap_seconds"], TASK3_DEADLINE_EPOCH)
    done = {r["run_id"] for r in c.read_jsonl(ROOT / "processes.jsonl") if r["status"] == "completed"}
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1",
           "DO_NOT_TRACK": "1", "CUDA_VISIBLE_DEVICES": c.GPU_UUID}
    for job in jobs:
        if job["run_id"] in done:
            continue
        if time.time() + 300 > deadline:
            c.append_failure(TASK, f"stopped at cap before {job['run_id']}")
            break
        for attempt in range(2):
            run_dir = ROOT / "raw" / f"{job['run_id']}.attempt{attempt}"
            if run_dir.exists():
                continue
            run_dir.mkdir(parents=True)
            apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                  capture_output=True, text=True).stdout.strip()
            command = [str(venv / "bin/vllm"), "bench", "latency", "--model", str(SNAPSHOT),
                       "--input-len", str(job["input_len"]), "--output-len", str(job["output_len"]),
                       "--batch-size", str(job["batch_size"]), "--num-iters-warmup", str(WARMUP_ITERS),
                       "--num-iters", str(ITERS), "--output-json", str(run_dir / "latency.json")]
            if job["kv_cache_dtype"] != "auto":
                command += ["--kv-cache-dtype", job["kv_cache_dtype"]]
            c.write_new(run_dir / "command.json", c.json_text(command))
            row: dict[str, Any] = {**job, "attempt": attempt, "started_at_utc": c.utc_now(), "gpu_apps_before": apps}
            if apps:
                row.update({"status": "infrastructure_failed", "reason_code": "gpu_not_idle"})
            else:
                began = time.monotonic()
                try:
                    done_proc = subprocess.run(command, capture_output=True, text=True, timeout=1800, env=env)
                    returncode: int | None = done_proc.returncode
                    log = done_proc.stdout + "\n" + done_proc.stderr
                except subprocess.TimeoutExpired as error:
                    returncode, log = None, f"timeout\n{error.stdout or ''}\n{error.stderr or ''}"
                c.write_new(run_dir / "vllm.log", log)
                row["wall_seconds"] = time.monotonic() - began
                row["returncode"] = returncode
                row["log_facts"] = log_facts(log)
                result_path = run_dir / "latency.json"
                if returncode == 0 and result_path.exists():
                    result = json.loads(result_path.read_text())
                    latencies = [float(v) for v in result["latencies"]]
                    row.update({"status": "completed", "process_median_s": statistics.median(latencies),
                                "latencies_s": latencies, "latency_json_sha256": c.sha256_file(result_path)})
                else:
                    row.update({"status": "failed" if returncode not in (None,) else "infrastructure_failed",
                                "reason_code": "timeout" if returncode is None else "vllm_failed",
                                "log_tail": log[-2000:]})
            row["finished_at_utc"] = c.utc_now()
            row["log_sha256"] = c.sha256_file(run_dir / "vllm.log") if (run_dir / "vllm.log").exists() else None
            c.append_jsonl(ROOT / "processes.jsonl", row)
            print(json.dumps({k: row.get(k) for k in ("run_id", "status", "process_median_s", "wall_seconds")}), flush=True)
            if row["status"] == "completed":
                break
            c.append_failure(TASK, f"`{job['run_id']}` attempt {attempt}: {row['status']} {row.get('reason_code')}\n\n"
                                   f"```\n{row.get('log_tail', '')[-1500:]}\n```")
            if row["status"] != "infrastructure_failed":
                break
    summarize()


def summarize() -> None:
    rows = [r for r in c.read_jsonl(ROOT / "processes.jsonl") if r["status"] == "completed"]
    out = []
    for dtype in DTYPES:
        for batch, label, input_len in POINTS:
            per_round = {}
            for r in rows:
                if (r["kv_cache_dtype"], r["batch_size"], r["context_label"]) == (dtype, batch, label):
                    per_round.setdefault(r["round"], {})[r["output_len"]] = r["process_median_s"]
            diffs = [(v[65] - v[1]) / 64 * 1000 for k, v in sorted(per_round.items()) if 1 in v and 65 in v]
            stats = c.point_statistics(diffs)
            backends = sorted({b for r in rows if (r["kv_cache_dtype"], r["batch_size"], r["context_label"]) == (dtype, batch, label)
                               for b in r["log_facts"]["attention_backends"]})
            out.append({"kv_cache_dtype": "bf16 (auto)" if dtype == "auto" else dtype, "batch_size": batch,
                        "context_label": label, "input_len": input_len, "step_ms_per_replicate": diffs,
                        "step_ms": stats["median_ms"], "cv": stats["cv"], "n": len(diffs),
                        "unstable": stats["cv"] is not None and stats["cv"] > c.CV_THRESHOLD,
                        "attention_backends": backends})
    for row in out:
        if row["kv_cache_dtype"] == "fp8":
            ref = next(r for r in out if r["kv_cache_dtype"] != "fp8" and r["batch_size"] == row["batch_size"]
                       and r["context_label"] == row["context_label"])
            row["fp8_over_bf16"] = (row["step_ms"] / ref["step_ms"]) if row["step_ms"] and ref["step_ms"] else None
            row["speedup_bf16_over_fp8"] = (ref["step_ms"] / row["step_ms"]) if row["step_ms"] and ref["step_ms"] else None
    path = ROOT / "summary.json"
    text = c.json_text({"generated_at_utc": c.utc_now(), "points": out})
    path.write_text(text)
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("run", "summarize", "order"))
    parser.add_argument("--venv", type=Path, default=Path("/home/rockrock/addendum-vllm-env"))
    args = parser.parse_args()
    if args.action == "run":
        run(args.venv)
    elif args.action == "order":
        for job in job_order():
            print(job["run_id"])
    else:
        summarize()


if __name__ == "__main__":
    main()
