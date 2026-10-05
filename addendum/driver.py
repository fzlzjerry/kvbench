#!/usr/bin/env python3
"""Host-side driver for the 2026-10-05 addendum (run as root, unattended in tmux).

    python3 addendum/driver.py preflight
    python3 addendum/driver.py run TASK          # resumable; completed processes are skipped
    python3 addendum/driver.py status TASK

Each process is one `docker run` of the authorized measurement image with the
Full Scan mounts (paper/posthoc/partB_gpu_diagnostics.py, itself copied from the
Full Scan recovery launcher), plus the addendum code (read-only) and one output
directory.  Results are appended to results/addendum-20261005/<task>/ as they
finish; failures are appended to results/addendum-20261005/FAILURES.md.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

TQ = ("tq_4bit_nc", "tq_k3v4_nc", "tq_3bit_nc")
KIVI = ("k4v4", "k2v2")

# Predicted seconds per process (worker span + ~7 s start-up).  BF16 and KIVI:
# Full Scan per-process durations (cache rebuild dominates KIVI).  TurboQuant:
# Full Scan setup (total minus 1,350 replays at the as-ported step) plus 1,350
# replays at the Appendix D 32-split step time, rounded up.
PREDICTED_SECONDS = {
    ("bf16", 1, 4096): 40, ("bf16", 1, 32768): 45, ("bf16", 1, 131072): 60,
    ("bf16", 8, 4096): 50, ("bf16", 8, 24576): 75, ("bf16", 8, 32768): 85,
    ("tq", 1, 32768): 70, ("tq", 1, 131072): 160, ("tq", 8, 32768): 130,
    ("kivi", 1, 4096): 90, ("kivi", 1, 32768): 245, ("kivi", 1, 131072): 850,
    ("kivi", 8, 4096): 100, ("kivi", 8, 24576): 240, ("kivi", 8, 32768): 290,
}
SPLIT_SWEEP_SECONDS = {8: 320, 16: 220, 64: 160}  # TQ-k3v4, B=1, 128K


def family(configuration: str) -> str:
    if configuration == "bf16":
        return "bf16"
    if configuration.startswith("tq_"):
        return "tq"
    if configuration in ("k4v4", "k2v4", "k2v2", "k4v2"):
        return "kivi"
    return "kvquant"


def predicted(configuration: str, batch: int, label: int, variant: dict[str, Any]) -> float:
    splits = variant.get("tq_splits")
    if splits in SPLIT_SWEEP_SECONDS and (batch, label) == (1, 131072):
        return SPLIT_SWEEP_SECONDS[splits]
    return PREDICTED_SECONDS.get((family(configuration), batch, label), 900)


# --------------------------------------------------------------------- tasks

def task_definitions() -> dict[str, dict[str, Any]]:
    """Blocks are the randomization units; each names one configuration and variant."""
    t1_points = [(1, 32768), (1, 131072), (8, 32768)]
    t2_points = [(1, 4096), (1, 32768), (1, 131072), (8, 4096), (8, 24576), (8, 32768)]

    def block(configuration: str, variant: dict[str, Any], points: list) -> dict[str, Any]:
        return {"configuration": configuration, "variant": variant, "points": points}

    return {
        "task1": {
            "task_number": 1, "cap_seconds": 3 * 3600,
            "blocks": {**{f"{cfg}-s32": block(cfg, {"tq_splits": 32}, t1_points) for cfg in TQ},
                       "bf16": block("bf16", {}, t1_points)},
        },
        # Optional split sweep; shares Task 1's cap (deadline taken from task1's manifest).
        "task1-sweep": {
            "task_number": 11, "cap_seconds": 3 * 3600, "cap_from": "task1",
            "blocks": {**{f"tq_k3v4_nc-s{s}": block("tq_k3v4_nc", {"tq_splits": s}, [(1, 131072)])
                          for s in (8, 16, 64)},
                       "bf16": block("bf16", {}, [(1, 131072)])},
        },
        "task2": {
            "task_number": 2, "cap_seconds": 4 * 3600,
            "blocks": {**{f"{cfg}-grouped": block(cfg, {"kivi_grouped_residual": True}, t2_points)
                          for cfg in KIVI},
                       "bf16": block("bf16", {}, t2_points)},
        },
        "smoke": {  # infrastructure smoke test; never reported
            "task_number": 0, "cap_seconds": 1800, "rounds": 1,
            "blocks": {"tq_k3v4_nc-s32": block("tq_k3v4_nc", {"tq_splits": 32}, [(1, 4096)]),
                       "bf16": block("bf16", {}, [(1, 4096)])},
        },
    }


def job_order(name: str, definition: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = []
    for round_index in range(definition.get("rounds", c.PROCESSES_PER_POINT)):
        seed = 2026100500 + 10 * definition["task_number"] + round_index
        rng = random.Random(seed)
        blocks = sorted(definition["blocks"])
        rng.shuffle(blocks)
        order = 0
        for block_name in blocks:
            block = definition["blocks"][block_name]
            configuration = block["configuration"]
            points = list(block["points"])
            rng.shuffle(points)
            for batch, label in points:
                jobs.append({
                    "task": name, "round": round_index, "seed": seed, "order_index": order,
                    "block": block_name, "configuration": configuration, "variant": block["variant"],
                    "batch_size": batch, "context_label": label,
                    "run_id": f"{name}-r{round_index}-o{order:03d}-{block_name}-b{batch}-l{label}",
                    "predicted_seconds": predicted(configuration, batch, label, block["variant"]),
                })
                order += 1
    return jobs


# ------------------------------------------------------------------ host side

def run_command(command: list[str], *, check: bool = True, timeout: float | None = None
                ) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, capture_output=True, text=True, timeout=timeout)


def git(*args: str, repo: Path) -> str:
    return run_command(["git", "-c", "safe.directory=*", "-C", str(repo), *args]).stdout.strip()


def addendum_commit() -> str:
    status = git("status", "--porcelain", "--", "addendum", "docs", repo=c.ADDENDUM_REPO)
    if status:
        raise SystemExit(f"addendum code or amendment has uncommitted changes:\n{status}")
    amendment = c.ADDENDUM_REPO / c.AMENDMENT
    if not amendment.exists() or "Status: DRAFT" in amendment.read_text():
        raise SystemExit("the amendment is missing or still marked DRAFT; no timing before it is committed")
    tracked = git("ls-files", "--error-unmatch", c.AMENDMENT, repo=c.ADDENDUM_REPO)
    if not tracked:
        raise SystemExit("the amendment is not committed")
    return git("rev-parse", "HEAD", repo=c.ADDENDUM_REPO)


def gpu_compute_apps() -> str:
    return run_command(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                        "--format=csv,noheader"], check=False).stdout.strip()


def gpu_state_record() -> dict[str, Any]:
    query = run_command(["nvidia-smi", "--query-gpu=uuid,name,driver_version,persistence_mode,"
                         "clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,clocks.applications.graphics,"
                         "clocks.applications.memory,power.limit,temperature.gpu,memory.used,"
                         "compute_mode,pstate", "--format=csv"], check=False).stdout
    detail = run_command(["nvidia-smi", "-q", "-d", "CLOCK,PERFORMANCE,POWER,TEMPERATURE"],
                         check=False).stdout
    return {"captured_at_utc": c.utc_now(), "query_csv": query, "nvidia_smi_q": detail,
            "compute_apps": gpu_compute_apps(),
            "all_processes": run_command(["nvidia-smi"], check=False).stdout}


def preflight() -> dict[str, Any]:
    checks: dict[str, Any] = {}

    def require(name: str, condition: bool, detail: str = "") -> None:
        checks[name] = {"pass": bool(condition), "detail": detail}
        if not condition:
            raise SystemExit(f"PREFLIGHT FAILED: {name} {detail}")

    require("running_as_root", os.geteuid() == 0)
    image = run_command(["docker", "image", "inspect", c.IMAGE, "--format", "{{.Id}}"], check=False)
    require("authorized_image_present", image.stdout.strip() == c.IMAGE, image.stdout.strip())
    head = git("rev-parse", "HEAD", repo=c.EXECUTION_REPO)
    require("execution_repo_head", head == c.EXECUTION_SHA, head)
    status = git("status", "--short", repo=c.EXECUTION_REPO)
    require("execution_repo_clean", status == "", status[:200])
    require("execution_repo_no_env", not (c.EXECUTION_REPO / ".env").exists())
    kvq = git("rev-parse", "HEAD", repo=c.CONTROL / "kvquant-source")
    require("kvquant_source_head", kvq == c.KVQUANT_SOURCE_SHA, kvq)
    require("kvquant_extension_sha256", c.sha256_file(c.KVQUANT_EXTENSION) == c.KVQUANT_EXTENSION_SHA)
    require("kivi_extension_sha256", c.sha256_file(c.KIVI_EXTENSION) == c.KIVI_EXTENSION_SHA)
    for name, path in (("calibration", c.CALIBRATION), ("model", c.MODEL), ("prefix_phase13", c.PREFIX_PHASE13),
                       ("prefix_phase13d", c.PREFIX_PHASE13D), ("prefix_phase16g", c.PREFIX_PHASE16G),
                       ("phase13b", c.PHASE13B), ("phase13rq4", c.PHASE13RQ4), ("family", c.FAMILY)):
        require(f"path_{name}", path.exists(), str(path))
    uuid = run_command(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"]).stdout.strip()
    require("gpu_uuid", uuid == c.GPU_UUID, uuid)
    persistence = run_command(["nvidia-smi", "--query-gpu=persistence_mode", "--format=csv,noheader"]
                              ).stdout.strip()
    require("persistence_mode_enabled", persistence == "Enabled", persistence)
    require("gpu_idle", gpu_compute_apps() == "", gpu_compute_apps())
    checks["addendum_commit"] = addendum_commit()
    checks["docker_version"] = run_command(["docker", "version", "--format", "{{.Server.Version}}"],
                                           check=False).stdout.strip()
    checks["gpu_state"] = gpu_state_record()
    return checks


def mounts(job_dir: Path) -> list[tuple[Path, str, bool]]:
    return [
        (c.EXECUTION_REPO, c.CONTAINER_REPO, True),
        (c.FAMILY, str(c.FAMILY), True),
        (c.PHASE13B, str(c.PHASE13B), True),
        (c.PHASE13RQ4, str(c.PHASE13RQ4), True),
        (c.CONTROL / "kivi-source", "/opt/kivi-source", True),
        (c.KIVI_EXTENSION, "/opt/kvbench/.phase3/site-packages/kivi_gemv.cpython-312-x86_64-linux-gnu.so", True),
        (c.CONTROL / "kvquant-source", "/opt/kvquant-source", True),
        (c.CONTROL / "kvquant-build", "/opt/kvquant-build", True),
        (c.CALIBRATION, "/opt/kvquant-calibration/kvqcal-cdb724c806d64d095c040d2673a987a3", True),
        (c.MODEL, str(c.MODEL), True),
        (c.PREFIX_PHASE13, "/opt/kvbench-prefix-phase13", True),
        (c.PREFIX_PHASE13D, "/opt/kvbench-prefix-phase13d", True),
        (c.PREFIX_PHASE16G, "/opt/kvbench-prefix-phase16g", True),
        (c.ADDENDUM_REPO / "addendum", c.CONTAINER_ADDENDUM, True),
        (c.RESULTS / "inputs", c.CONTAINER_INPUTS, True),
        (job_dir, c.CONTAINER_OUT, False),
    ]


ENVIRONMENT = {
    "PYTHONPATH": f"/opt/kivi-source:/opt/kvbench/.phase3/site-packages:{c.CONTAINER_REPO}/src:{c.CONTAINER_REPO}",
    "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "TORCH_CUDA_ARCH_LIST": "12.0+PTX", "TRITON_CACHE_DIR": "/root/.triton",
    "KVBENCH_AUTHORIZED_IMAGE_DIGEST": c.IMAGE,
    "KVBENCH_EXECUTION_ENVIRONMENT": "measurement_container",
    "KVBENCH_KIVI_SOURCE_ROOT": "/opt/kivi-source", "KVBENCH_KVQUANT_SOURCE_ROOT": "/opt/kvquant-source",
    "KVBENCH_KVQUANT_CALIBRATION_ROOT": "/opt/kvquant-calibration/kvqcal-cdb724c806d64d095c040d2673a987a3",
    "KVBENCH_KVQUANT_EXTENSION": "/opt/kvquant-build/quant_cuda.cpython-312-x86_64-linux-gnu.so",
    "KVBENCH_KVQUANT_EXTENSION_SHA256": c.KVQUANT_EXTENSION_SHA,
}


def docker_command(name: str, job_dir: Path, script: str) -> list[str]:
    command = ["docker", "run", "--rm", "--network", "none", "--gpus", f"device={c.GPU_UUID}",
               "--name", name[:120], "--entrypoint", "/usr/bin/bash"]
    for source, target, readonly in mounts(job_dir):
        command += ["--mount", f"type=bind,src={source},dst={target}" + (",readonly" if readonly else "")]
    for key, value in ENVIRONMENT.items():
        command += ["-e", f"{key}={value}"]
    command += ["-w", c.CONTAINER_REPO, c.IMAGE, "--noprofile", "--norc", "-c", script]
    return command


def timing_script(job: dict[str, Any], commit: str) -> str:
    worker = ["/opt/kvbench/.venv/bin/python", f"{c.CONTAINER_ADDENDUM}/timing_worker.py",
              "--task", job["task"], "--run-id", job["run_id"],
              "--configuration", job["configuration"], "--batch-size", str(job["batch_size"]),
              "--context-label", str(job["context_label"]), "--replicate-index", str(job["round"]),
              "--order-index", str(job["order_index"]), "--variant", json.dumps(job["variant"]),
              "--addendum-commit", commit, "--output-dir", c.CONTAINER_OUT]
    out = c.CONTAINER_OUT
    return (f"{shlex.join(worker)} > {out}/worker.stdout 2> {out}/worker.stderr; rc=$?; "
            f"echo $rc > {out}/worker.returncode; exit $rc")


def run_process(job: dict[str, Any], task_root: Path, commit: str) -> dict[str, Any]:
    """Run one process; preserve any earlier failed attempt; never overwrite."""
    raw = task_root / "raw"
    job_dir = raw / job["run_id"]
    attempt = 0
    while (raw / f"{job['run_id']}.attempt{attempt}").exists():
        attempt += 1
    if job_dir.exists():
        os.rename(job_dir, raw / f"{job['run_id']}.attempt{attempt}")
        attempt += 1
    job_dir.mkdir(parents=True)
    job_dir.chmod(0o777)
    c.write_new(job_dir / "job.json", c.json_text({**job, "attempt": attempt}))
    status: dict[str, Any] = {"run_id": job["run_id"], "attempt": attempt, "started_at_utc": c.utc_now()}
    apps = gpu_compute_apps()
    if apps:
        status.update({"status": "infrastructure_failed", "reason_code": "gpu_not_idle", "detail": apps})
    else:
        command = docker_command(f"kvbench-{job['run_id']}", job_dir, timing_script(job, commit))
        c.write_new(job_dir / "docker_command.json", c.json_text(command))
        timeout = max(15 * 60, job["predicted_seconds"] * 2.5)
        began = time.monotonic()
        try:
            completed = run_command(command, check=False, timeout=timeout)
            returncode: int | None = completed.returncode
            c.write_new(job_dir / "docker.stdout", completed.stdout)
            c.write_new(job_dir / "docker.stderr", completed.stderr)
        except subprocess.TimeoutExpired:
            returncode = None
            run_command(["docker", "rm", "-f", f"kvbench-{job['run_id']}"[:120]], check=False)
            c.write_new(job_dir / "docker.stderr", f"timeout after {timeout:.0f} s\n")
        status["wall_seconds_including_setup"] = time.monotonic() - began
        status["gpu_apps_after"] = gpu_compute_apps()
        result_path = job_dir / "worker_result.json"
        worker = json.loads(result_path.read_text()) if result_path.exists() else None
        if returncode is None:
            status.update({"status": "infrastructure_failed", "reason_code": "timeout"})
        elif worker is None:
            status.update({"status": "infrastructure_failed", "reason_code": "worker_result_missing",
                           "docker_returncode": returncode})
        elif worker.get("status") != "completed":
            status.update({"status": "failed", "reason_code": "worker_failed",
                           "error": f"{worker.get('error_type')}: {worker.get('error')}"})
        else:
            failed_checks = [k for k, v in worker["checks"].items() if v is not True]
            status.update({"status": "completed",
                           "failed_checks": failed_checks,
                           "host_wall_process_median_ms": worker["host_wall_process_median_ms"],
                           "worker_result_sha256": c.sha256_file(result_path)})
    status["finished_at_utc"] = c.utc_now()
    c.write_new(job_dir / "status.json", c.json_text(status))
    return status


def completed_runs(task_root: Path) -> dict[str, dict[str, Any]]:
    return {row["run_id"]: row for row in c.read_jsonl(task_root / "processes.jsonl")
            if row["status"] == "completed"}


def point_key(job: dict[str, Any]) -> tuple:
    return (job["configuration"], json.dumps(job["variant"], sort_keys=True),
            job["batch_size"], job["context_label"])


def write_point(task_root: Path, jobs: list[dict[str, Any]], key: tuple, commit: str,
                final: bool = False) -> None:
    done = completed_runs(task_root)
    rows = [done[j["run_id"]] for j in jobs if point_key(j) == key and j["run_id"] in done]
    existing = {(r["configuration"], json.dumps(r["variant"], sort_keys=True), r["batch_size"],
                 r["context_label"]) for r in c.read_jsonl(task_root / "points.jsonl")}
    if key in existing:
        return
    if len(rows) < c.PROCESSES_PER_POINT and not final:
        return
    rows.sort(key=lambda r: r["round"])
    medians = [r["host_wall_process_median_ms"] for r in rows]
    stats = c.point_statistics(medians)
    configuration, variant, batch, label = key
    disposition = ("incomplete" if len(rows) < c.PROCESSES_PER_POINT else
                   "check_failed" if any(r.get("failed_checks") for r in rows) else
                   "unstable" if stats["cv"] is not None and stats["cv"] > c.CV_THRESHOLD else "stable")
    c.append_jsonl(task_root / "points.jsonl", {
        "addendum_id": c.ADDENDUM_ID, "task": task_root.name, "configuration": configuration,
        "label": c.LABELS.get(configuration, configuration), "variant": json.loads(variant),
        "batch_size": batch, "context_label": label, "latency_basis": "host_wall",
        "process_medians_ms": medians, "point_median_ms": stats["median_ms"], "cv": stats["cv"],
        "n_processes": len(rows), "disposition": disposition,
        "unstable": disposition == "unstable",
        "run_ids": [r["run_id"] for r in rows],
        "worker_result_sha256": [r["worker_result_sha256"] for r in rows],
        "failed_checks": sorted({k for r in rows for k in r.get("failed_checks", [])}),
        "addendum_git_sha": commit, "execution_git_sha": c.EXECUTION_SHA,
        "container_digest": c.IMAGE, "written_at_utc": c.utc_now(),
    })


def run_task(name: str) -> None:
    definitions = task_definitions()
    if name not in definitions:
        raise SystemExit(f"unknown task {name}")
    definition = definitions[name]
    task_root = c.RESULTS / name
    task_root.mkdir(parents=True, exist_ok=True)
    pre = preflight()
    commit = pre["addendum_commit"]
    jobs = job_order(name, definition)
    manifest_path = task_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if [j["run_id"] for j in manifest["jobs"]] != [j["run_id"] for j in jobs]:
            raise SystemExit("job order differs from the recorded manifest")
    else:
        manifest = {"addendum_id": c.ADDENDUM_ID, "task": name, "amendment": c.AMENDMENT,
                    "started_at_utc": c.utc_now(), "started_at_epoch": time.time(),
                    "cap_seconds": definition["cap_seconds"], "addendum_git_sha_at_start": commit,
                    "jobs": jobs}
        c.write_new(manifest_path, c.json_text(manifest))
    c.write_new(task_root / f"preflight-{int(time.time())}.json", c.json_text(pre))
    deadline = manifest["started_at_epoch"] + manifest["cap_seconds"]
    if definition.get("cap_from"):
        parent = json.loads((c.RESULTS / definition["cap_from"] / "manifest.json").read_text())
        deadline = parent["started_at_epoch"] + parent["cap_seconds"]
    keys = []
    for job in jobs:
        if point_key(job) not in keys:
            keys.append(point_key(job))
    stopped = None
    for job in jobs:
        if job["run_id"] in completed_runs(task_root):
            continue
        if time.time() + job["predicted_seconds"] * 1.2 > deadline:
            stopped = f"cap: next process {job['run_id']} (predicted {job['predicted_seconds']} s) would cross the {manifest['cap_seconds'] / 3600:.1f} h cap"
            break
        for attempt in range(2):  # one rerun, for infrastructure failures only
            status = run_process(job, task_root, commit)
            row = {**{k: job[k] for k in ("task", "round", "order_index", "configuration", "variant",
                                          "batch_size", "context_label", "run_id")}, **status,
                   "addendum_git_sha": commit, "container_digest": c.IMAGE}
            c.append_jsonl(task_root / "processes.jsonl", row)
            print(json.dumps({k: row.get(k) for k in ("run_id", "status", "reason_code",
                                                     "host_wall_process_median_ms")}), flush=True)
            if status["status"] == "completed":
                break
            c.append_failure(name, f"process `{job['run_id']}` attempt {status['attempt']}: "
                                   f"{status['status']} ({status.get('reason_code')}) "
                                   f"{status.get('error') or status.get('detail') or ''}")
            if status["status"] != "infrastructure_failed":
                break
        write_point(task_root, jobs, point_key(job), commit)
    for key in keys:
        write_point(task_root, jobs, key, commit, final=True)
    if stopped:
        c.append_failure(name, f"stopped before completion: {stopped}")
    summary = {"task": name, "finished_at_utc": c.utc_now(), "stopped": stopped,
               "completed_processes": len(completed_runs(task_root)), "planned_processes": len(jobs)}
    c.append_jsonl(task_root / "driver-sessions.jsonl", summary)
    print(json.dumps(summary), flush=True)


def run_check(task: str, name: str, configuration: str, variant: dict[str, Any],
              extra: list[str], sanitize: str | None = None, timeout: float = 3600) -> dict[str, Any]:
    """Untimed correctness / sanitizer job (check_worker.py); results under <task>/checks/<name>/."""
    pre = preflight()
    commit = pre["addendum_commit"]
    job_dir = c.RESULTS / task / "checks" / name
    if (job_dir / "status.json").exists():
        previous = json.loads((job_dir / "status.json").read_text())
        if previous.get("status") == "completed":
            print(json.dumps(previous))
            return previous
        attempt = 0
        while (job_dir.parent / f"{name}.attempt{attempt}").exists():
            attempt += 1
        os.rename(job_dir, job_dir.parent / f"{name}.attempt{attempt}")
    job_dir.mkdir(parents=True)
    job_dir.chmod(0o777)
    c.write_new(job_dir / "preflight.json", c.json_text(pre))
    out = c.CONTAINER_OUT
    worker = ["/opt/kvbench/.venv/bin/python", f"{c.CONTAINER_ADDENDUM}/check_worker.py",
              "--configuration", configuration, "--variant", json.dumps(variant),
              "--addendum-commit", commit, "--output-dir", out, *extra]
    if sanitize:
        kernel_filter = [] if sanitize == "all" else ["--kernel-name", sanitize]
        worker = ["/usr/local/cuda-13.0/bin/compute-sanitizer", "--tool", "memcheck",
                  *kernel_filter, "--error-exitcode", "17",
                  "--log-file", f"{out}/sanitizer.log", *worker]
    script = (f"{shlex.join(worker)} > {out}/worker.stdout 2> {out}/worker.stderr; rc=$?; "
              f"echo $rc > {out}/worker.returncode; exit $rc")
    command = docker_command(f"kvbench-check-{task}-{name}", job_dir, script)
    c.write_new(job_dir / "docker_command.json", c.json_text(command))
    status: dict[str, Any] = {"task": task, "check": name, "started_at_utc": c.utc_now(),
                              "configuration": configuration, "variant": variant, "sanitize": sanitize}
    try:
        completed = run_command(command, check=False, timeout=timeout)
        status["docker_returncode"] = completed.returncode
        c.write_new(job_dir / "docker.stdout", completed.stdout)
        c.write_new(job_dir / "docker.stderr", completed.stderr)
    except subprocess.TimeoutExpired:
        run_command(["docker", "rm", "-f", f"kvbench-check-{task}-{name}"[:120]], check=False)
        status["docker_returncode"] = None
    result_path = job_dir / "worker_result.json"
    worker_result = json.loads(result_path.read_text()) if result_path.exists() else None
    ok = status["docker_returncode"] == 0 and worker_result is not None and worker_result.get("status") == "completed"
    status.update({"status": "completed" if ok else "failed", "finished_at_utc": c.utc_now(),
                   "worker_error": None if worker_result is None else worker_result.get("error")})
    c.write_new(job_dir / "status.json", c.json_text(status))
    if not ok:
        c.append_failure(task, f"check `{name}` failed: returncode {status['docker_returncode']} "
                               f"{status['worker_error'] or ''}")
    print(json.dumps(status), flush=True)
    return status


CHECKS = {
    # name: (task, configuration, variant, extra args, sanitizer kernel filter)
    "t1-greedy-s4": ("task1", "tq_k3v4_nc", {}, [], None),
    "t1-greedy-s32": ("task1", "tq_k3v4_nc", {"tq_splits": 32}, [], None),
    "t1-sanitizer-s32": ("task1", "tq_k3v4_nc", {"tq_splits": 32},
                         ["--prefix-tokens", "512", "--steps", "4"], "regex=_tq_decode_stage1|_fwd_kernel_stage2"),
    "t1-sanitizer-s32-all": ("task1", "tq_k3v4_nc", {"tq_splits": 32},
                             ["--prefix-tokens", "128", "--steps", "4"], "all"),
    "smoke-greedy-bf16": ("smoke", "bf16", {}, ["--prefix-tokens", "512", "--steps", "4"], None),
}


def show_status(name: str) -> None:
    task_root = c.RESULTS / name
    for row in c.read_jsonl(task_root / "points.jsonl"):
        print(f"{row['label']:10s} B={row['batch_size']:<2d} L={row['context_label']:<6d} "
              f"{row['point_median_ms']:9.3f} ms  CV={row['cv'] if row['cv'] is None else round(100 * row['cv'], 2)}%  "
              f"{row['disposition']}")
    done = completed_runs(task_root)
    print(f"completed processes: {len(done)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("preflight", "run", "status", "order", "check"))
    parser.add_argument("task", nargs="?")
    args = parser.parse_args()
    if args.action == "preflight":
        print(c.json_text(preflight()))
    elif args.action == "run":
        run_task(args.task)
    elif args.action == "check":
        task, configuration, variant, extra, sanitize = CHECKS[args.task]
        run_check(task, args.task, configuration, variant, extra, sanitize)
    elif args.action == "order":
        for job in job_order(args.task, task_definitions()[args.task]):
            print(job["run_id"], job["predicted_seconds"])
    else:
        show_status(args.task)


if __name__ == "__main__":
    main()
