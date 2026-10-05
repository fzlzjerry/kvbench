#!/usr/bin/env python3
"""POST-HOC Part B GPU diagnostics runner (run as root on the measurement host).

POST-HOC DIAGNOSTIC, NOT A TIMING RUN.  Nothing here changes a measured
configuration, an adapter, or frozen evidence, and no profiler duration is
reported as benchmark timing.  Each job runs in its own container from the
authorized measurement image with the Full Scan execution repository
(ec534d99, control/execution-repo-ec534) mounted read-only, using the same
mounts and environment as the Full Scan recovery launcher
(control/run-replicate4-after-recovery.sh) plus two posthoc mounts:
paper/posthoc (read-only, the worker) and one job output directory.

Jobs, in priority order:

  1. TurboQuant split diagnostic (Nsight Systems, Graph mode, Phase 15 nsys
     flags, 64 warmup + 8 profiled replays):
       TQ-4bit, TQ-k3v4, TQ-3bit x {B1 L32K, B1 L128K, B8 L32K, B8 L48K}
       x splits {4 (as-ported), 32 (vLLM v0.25.1 default)},
     plus BF16 at {B1 L32K, B1 L128K, B8 L32K} for the same-point comparison.
     B8 L128K is capacity-infeasible for TurboQuant (predicted 202.5 GB end to
     end, 95.0 GB steady state, limit 89.7 GB); B8 L48K is the largest B = 8
     context in the Full Scan and replaces it.  BF16 is infeasible at B8 L48K.
  2. Traffic-model check (Nsight Compute, Phase 15 metrics and defaults:
     --cache-control all, --clock-control base; 64 warmup + 1 profiled replay):
       BF16, TQ-4bit, KIVI-k4v4 at B8 L32K and B16 L16K.
  3. Optional KVQuant-4 Nsight Compute at the same two points (--with-kvquant-ncu).
     Estimated reports: ~42 GB (B8) and ~78 GB (B16), ~3.2 h and ~6 h.  Each
     ncu job is skipped with a recorded reason if free disk space is below its
     estimated need.

Estimated wall time: jobs 1 ~45-75 min; jobs 2 ~2.5-3 h; jobs 3 ~9-10 h.

Outputs: artifacts/posthoc/.staging-partb-<stamp>/ while running, then a sealed,
read-only artifacts/posthoc/posthoc-b-gpu-<stamp>-<sha>-<rand>/ that
scripts/r2_artifact.py can publish.  A failed job is recorded and the run
continues; --resume <staging dir> skips jobs that already completed.

Usage (repository root, as root):
  python3 paper/posthoc/partB_gpu_diagnostics.py plan
  python3 paper/posthoc/partB_gpu_diagnostics.py smoke         # 2 cheap jobs (~3 min), no seal
  python3 paper/posthoc/partB_gpu_diagnostics.py run [--with-kvquant-ncu] [--resume DIR]
  python3 paper/posthoc/partB_gpu_diagnostics.py seal DIR      # only if a run was interrupted
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import posthoc_common as pc  # noqa: E402

REPO = pc.REPO
FAMILY = REPO / "artifacts/phase16/phase16-20260831t123029614620z-ec534d99-de80ac"
CONTROL = FAMILY / "control"
EXECUTION_REPO = CONTROL / "execution-repo-ec534"
EXECUTION_SHA = "ec534d9958d5616eef981c628b87a92c7c809872"
KVQUANT_SOURCE_SHA = "34b0bdfa83082e1f30387d9ac5cca369006e089c"
KVQUANT_EXTENSION = CONTROL / "kvquant-build/quant_cuda.cpython-312-x86_64-linux-gnu.so"
KVQUANT_EXTENSION_SHA = "b3c33badb8e55b19d6b2ce535182e964ce51e5102d8413b29701dd3d817ad73d"
KIVI_EXTENSION = CONTROL / "kivi-image-source/quant/kivi_gemv.cpython-312-x86_64-linux-gnu.so"
KIVI_EXTENSION_SHA = "45d29ec1a3cecc4b253d1d1dd6139ef4f91cff88993db61a9d73685314851aa9"
IMAGE = "sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e"
GPU_UUID = "GPU-75bd273e-6b20-0d22-1b0b-5fbb6fb0025b"
CALIBRATION = REPO / "calibration/kvquant/kvqcal-cdb724c806d64d095c040d2673a987a3"
MODEL = Path("/root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct")
PREFIX_PHASE13 = REPO / "artifacts/phase13_prefix_catalogs/phase13-20260822t150835736582z-4ddd7b17-3a8fb3/catalog"
PREFIX_PHASE13D = Path("/home/rockrock/phase13d_prefix_states/phase13d-20260825t030556684636z-a06837a3-83761a")
PREFIX_PHASE16G = Path("/home/rockrock/phase16g_prefix_states/phase16g-20260830t061918945302z-6c829eda-f16c0d")
PHASE13B = REPO / "artifacts/phase13b/phase13b-20260801t143138050263z-b862af64-batch-admission"
PHASE13RQ4 = REPO / "artifacts/phase13rq4/phase13rq4-20260820t094629495794z-ab4e0b84-b8c7bd"
POSTHOC_DIR = Path(__file__).resolve().parent  # mounted read-only at /opt/posthoc
OUTPUT_PARENT = REPO / "artifacts/posthoc"
CONTAINER_REPO = "/home/rockrock/cmu_paper"

NCU_METRICS = (
    "dram__bytes_op_read.sum,dram__bytes_op_write.sum,lts__t_sectors_op_read.sum,"
    "lts__t_sectors_op_write.sum,lts__t_sector_hit_rate.pct,"
    "dram__throughput.avg.pct_of_peak_sustained_elapsed,"
    "sm__throughput.avg.pct_of_peak_sustained_elapsed,"
    "sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warps_active.avg.per_cycle_active,"
    "gpu__time_duration.sum,smsp__sass_thread_inst_executed_op_memory_pred_on.sum,"
    "smsp__inst_executed.sum"
)
# Kernels per decode step from the Full Scan records; ncu cost ~0.45 s per
# kernel and report size ~1.65 MB per kernel (Phase 15 observations).
KERNELS_PER_STEP = {("bf16", 8): 1482, ("bf16", 16): 1482, ("tq_4bit_nc", 8): 1678,
                    ("tq_4bit_nc", 16): 1678, ("k4v4", 8): 6954, ("k4v4", 16): 6954,
                    ("kvq4", 8): 25834, ("kvq4", 16): 47594}
NCU_SECONDS_PER_KERNEL = 0.45
NCU_BYTES_PER_KERNEL = 1.65e6
DISK_SAFETY = 1.6  # report + CSV export + margin


def job_list(with_kvquant: bool) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []

    def add(kind: str, configuration: str, batch: int, label: int, splits: int | None,
            priority: str) -> None:
        name = f"{kind}-{configuration}-b{batch}-l{label}" + (f"-s{splits}" if splits else "")
        jobs.append({"job_id": name, "kind": kind, "configuration": configuration,
                     "batch_size": batch, "context_label": label, "tq_splits": splits,
                     "priority": priority})

    split_points = ((1, 32768), (1, 131072), (8, 32768), (8, 49152))
    for batch, label in ((1, 32768), (1, 131072), (8, 32768)):
        add("nsys", "bf16", batch, label, None, "1_split_diagnostic")
    for configuration in ("tq_4bit_nc", "tq_k3v4_nc", "tq_3bit_nc"):
        for batch, label in split_points:
            for splits in (4, 32):
                add("nsys", configuration, batch, label, splits, "1_split_diagnostic")
    for batch, label in ((8, 32768), (16, 16384)):
        for configuration in ("bf16", "tq_4bit_nc", "k4v4"):
            add("ncu", configuration, batch, label, 4 if configuration.startswith("tq_") else None,
                "2_traffic_model_check")
    if with_kvquant:
        for batch, label in ((8, 32768), (16, 16384)):
            add("ncu", "kvq4", batch, label, None, "3_optional_kvquant_ncu")
    for job in jobs:
        if job["kind"] == "nsys":
            job["estimated_minutes"] = 2.5 if job["batch_size"] == 8 else 1.5
            job["estimated_bytes"] = 2e8
        else:
            kernels = KERNELS_PER_STEP[(job["configuration"], job["batch_size"])]
            job["estimated_minutes"] = kernels * NCU_SECONDS_PER_KERNEL / 60 + 4
            job["estimated_bytes"] = kernels * NCU_BYTES_PER_KERNEL
    return jobs


def run_command(command: list[str], *, timeout: float | None = None,
                check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, capture_output=True, text=True, timeout=timeout)


def sha256_file(path: Path) -> str:
    return pc.sha256_file(path)


def preflight() -> dict[str, Any]:
    checks: dict[str, Any] = {}

    def require(name: str, condition: bool, detail: str = "") -> None:
        checks[name] = {"pass": bool(condition), "detail": detail}
        if not condition:
            raise SystemExit(f"PREFLIGHT FAILED: {name} {detail}")

    require("running_as_root", os.geteuid() == 0)
    image = run_command(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], check=False)
    require("authorized_image_present", image.stdout.strip() == IMAGE, image.stdout.strip())
    git = ["git", "-c", "safe.directory=*", "-C"]
    head = run_command([*git, str(EXECUTION_REPO), "rev-parse", "HEAD"]).stdout.strip()
    require("execution_repo_head", head == EXECUTION_SHA, head)
    status = run_command([*git, str(EXECUTION_REPO), "status", "--short"]).stdout
    require("execution_repo_clean", status == "", status[:200])
    require("execution_repo_no_env", not (EXECUTION_REPO / ".env").exists())
    kvq = run_command([*git, str(CONTROL / "kvquant-source"), "rev-parse", "HEAD"]).stdout.strip()
    require("kvquant_source_head", kvq == KVQUANT_SOURCE_SHA, kvq)
    require("kvquant_extension_sha256", sha256_file(KVQUANT_EXTENSION) == KVQUANT_EXTENSION_SHA)
    require("kivi_extension_sha256", sha256_file(KIVI_EXTENSION) == KIVI_EXTENSION_SHA)
    for name, path in (("calibration", CALIBRATION), ("model", MODEL), ("prefix_phase13", PREFIX_PHASE13),
                       ("prefix_phase13d", PREFIX_PHASE13D), ("prefix_phase16g", PREFIX_PHASE16G),
                       ("phase13b", PHASE13B), ("phase13rq4", PHASE13RQ4), ("family", FAMILY)):
        require(f"path_{name}", path.exists(), str(path))
    uuid = run_command(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"]).stdout.strip()
    require("gpu_uuid", uuid == GPU_UUID, uuid)
    checks["worker_sha256"] = sha256_file(POSTHOC_DIR / "partB_worker.py")
    checks["runner_sha256"] = sha256_file(Path(__file__))
    checks["host_nsys"] = shutil.which("nsys")
    checks["docker_version"] = run_command(["docker", "version", "--format", "{{.Server.Version}}"],
                                           check=False).stdout.strip()
    checks["driver"] = run_command(["nvidia-smi", "--query-gpu=driver_version,name",
                                    "--format=csv,noheader"], check=False).stdout.strip()
    return checks


def gpu_idle() -> bool:
    apps = run_command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                       check=False).stdout.strip()
    return apps == ""


def docker_command(job: dict[str, Any], job_dir: Path) -> list[str]:
    out = "/opt/posthoc-out"
    worker = ["/opt/kvbench/.venv/bin/python", "/opt/posthoc/partB_worker.py",
              "--configuration", job["configuration"], "--batch-size", str(job["batch_size"]),
              "--context-label", str(job["context_label"]),
              "--decode-operations", "8" if job["kind"] == "nsys" else "1",
              "--warmup-steps", "64", "--family-root", str(FAMILY), "--output-dir", out]
    if job["tq_splits"]:
        worker += ["--tq-splits", str(job["tq_splits"])]
    if job["kind"] == "nsys":
        profile = ["nsys", "profile", "--trace=cuda,nvtx,osrt", "--sample=none", "--cpuctxsw=none",
                   "--backtrace=none", "--capture-range=nvtx", "--nvtx-capture=phase15_decode",
                   "--capture-range-end=stop", "--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0",
                   "--cuda-graph-trace=node", "--force-overwrite=true", f"--output={out}/report",
                   *worker]
        export = ["nsys", "export", "--type=sqlite", "--force-overwrite=true",
                  f"--output={out}/report.sqlite", f"{out}/report.nsys-rep"]
        report = f"{out}/report.nsys-rep"
    else:
        profile = ["ncu", "--target-processes=application-only", "--nvtx",
                   "--nvtx-include=phase15_decode/", "--replay-mode=kernel", "--graph-profiling=node",
                   "--disable-extra-suffixes", "--force-overwrite", "--metrics", NCU_METRICS,
                   "--export", f"{out}/report", *worker]
        export = ["ncu", "--import", f"{out}/report.ncu-rep", "--page=raw", "--csv",
                  "--print-units=base", "--log-file", f"{out}/report_raw.csv"]
        report = f"{out}/report.ncu-rep"
    script = (f"set +e; {shlex.join(profile)} > {out}/profiler.stdout 2> {out}/profiler.stderr; rc=$?; "
              f"echo $rc > {out}/profiler.returncode; "
              f"if [ -f {report} ]; then {shlex.join(export)} > {out}/export.stdout 2> {out}/export.stderr; "
              f"echo $? > {out}/export.returncode; fi; "
              f"{shlex.join(['nsys' if job['kind'] == 'nsys' else 'ncu', '--version'])} "
              f"> {out}/tool_version.txt 2>&1; exit $rc")
    mounts = [
        (EXECUTION_REPO, CONTAINER_REPO, True),
        (FAMILY, str(FAMILY), True),
        (PHASE13B, str(PHASE13B), True),
        (PHASE13RQ4, str(PHASE13RQ4), True),
        (CONTROL / "kivi-source", "/opt/kivi-source", True),
        (KIVI_EXTENSION, "/opt/kvbench/.phase3/site-packages/kivi_gemv.cpython-312-x86_64-linux-gnu.so", True),
        (CONTROL / "kvquant-source", "/opt/kvquant-source", True),
        (CONTROL / "kvquant-build", "/opt/kvquant-build", True),
        (CALIBRATION, "/opt/kvquant-calibration/kvqcal-cdb724c806d64d095c040d2673a987a3", True),
        (MODEL, str(MODEL), True),
        (PREFIX_PHASE13, "/opt/kvbench-prefix-phase13", True),
        (PREFIX_PHASE13D, "/opt/kvbench-prefix-phase13d", True),
        (PREFIX_PHASE16G, "/opt/kvbench-prefix-phase16g", True),
        (POSTHOC_DIR, "/opt/posthoc", True),
        (job_dir, out, False),
    ]
    env = {
        "PYTHONPATH": f"/opt/kivi-source:/opt/kvbench/.phase3/site-packages:{CONTAINER_REPO}/src:{CONTAINER_REPO}",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "TORCH_CUDA_ARCH_LIST": "12.0+PTX", "TRITON_CACHE_DIR": "/root/.triton",
        "KVBENCH_AUTHORIZED_IMAGE_DIGEST": IMAGE,
        "KVBENCH_EXECUTION_ENVIRONMENT": "measurement_container",
        "KVBENCH_KIVI_SOURCE_ROOT": "/opt/kivi-source", "KVBENCH_KVQUANT_SOURCE_ROOT": "/opt/kvquant-source",
        "KVBENCH_KVQUANT_CALIBRATION_ROOT": "/opt/kvquant-calibration/kvqcal-cdb724c806d64d095c040d2673a987a3",
        "KVBENCH_KVQUANT_EXTENSION": "/opt/kvquant-build/quant_cuda.cpython-312-x86_64-linux-gnu.so",
        "KVBENCH_KVQUANT_EXTENSION_SHA256": KVQUANT_EXTENSION_SHA,
    }
    command = ["docker", "run", "--rm", "--network", "none", "--gpus", f"device={GPU_UUID}",
               "--name", f"kvbench-posthoc-{job['job_id'][:48]}-{os.getpid()}",
               "--entrypoint", "/usr/bin/bash"]
    for source, target, readonly in mounts:
        command += ["--mount", f"type=bind,src={source},dst={target}" + (",readonly" if readonly else "")]
    for key, value in env.items():
        command += ["-e", f"{key}={value}"]
    command += ["-w", CONTAINER_REPO, IMAGE, "--noprofile", "--norc", "-c", script]
    return command


def run_job(job: dict[str, Any], stage: Path, log: pc.WarningLog) -> dict[str, Any]:
    job_dir = stage / "jobs" / job["job_id"]
    status_path = job_dir / "status.json"
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if previous.get("status") == "completed":
            return previous
        attempt = int(previous.get("attempt", 0)) + 1
        # A failed attempt is preserved, never overwritten.
        os.rename(job_dir, stage / "jobs" / f"{job['job_id']}.failed-attempt{attempt - 1}")
    else:
        attempt = 0
    job_dir.mkdir(parents=True)
    job_dir.chmod(0o777)
    pc.write_new(job_dir / "job.json", pc.json_text(job))
    status: dict[str, Any] = {"job_id": job["job_id"], "attempt": attempt, "started_at_utc": pc.utc_now()}
    free = shutil.disk_usage(stage).free
    if free < job["estimated_bytes"] * DISK_SAFETY:
        status.update({"status": "skipped", "reason_code": "insufficient_free_disk",
                       "detail": f"free {free / 1e9:.1f} GB < estimated need "
                                 f"{job['estimated_bytes'] * DISK_SAFETY / 1e9:.1f} GB"})
    elif not gpu_idle():
        status.update({"status": "skipped", "reason_code": "gpu_not_idle",
                       "detail": "another compute process was running before the job"})
    else:
        command = docker_command(job, job_dir)
        pc.write_new(job_dir / "docker_command.json", pc.json_text(command))
        timeout = max(30 * 60, job["estimated_minutes"] * 60 * 2.5)
        began = time.monotonic()
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
            returncode: int | None = completed.returncode
            pc.write_new(job_dir / "docker.stdout", completed.stdout)
            pc.write_new(job_dir / "docker.stderr", completed.stderr)
        except subprocess.TimeoutExpired as error:
            returncode = None
            subprocess.run(["docker", "rm", "-f", command[command.index("--name") + 1]],
                           capture_output=True, text=True)
            pc.write_new(job_dir / "docker.stderr", f"timeout after {timeout:.0f} s\n{error}")
        elapsed = time.monotonic() - began
        result_path = job_dir / "worker_result.json"
        worker = json.loads(result_path.read_text()) if result_path.exists() else None
        report = job_dir / ("report.nsys-rep" if job["kind"] == "nsys" else "report.ncu-rep")
        export = job_dir / ("report.sqlite" if job["kind"] == "nsys" else "report_raw.csv")
        ok = (returncode == 0 and worker is not None and worker.get("status") == "completed"
              and report.exists() and export.exists())
        status.update({
            "status": "completed" if ok else "failed",
            "reason_code": None if ok else (
                "timeout" if returncode is None else
                "worker_failed" if worker is not None and worker.get("status") != "completed" else
                "worker_result_missing" if worker is None else
                "report_or_export_missing"),
            "docker_returncode": returncode,
            "wall_seconds_including_setup": elapsed,
            "wall_seconds_note": "container, model load, prefix build and profiler overhead; not timing",
            "worker_error": None if worker is None else worker.get("error"),
        })
    status["finished_at_utc"] = pc.utc_now()
    pc.write_new(status_path, pc.json_text(status))
    if status["status"] != "completed":
        log.warn(f"job_{status['status']}", f"{job['job_id']}: {status.get('reason_code')} "
                                            f"{status.get('worker_error') or status.get('detail') or ''}")
    return status


def seal_stage(stage: Path, preflight_record: dict[str, Any] | None) -> None:
    jobs = sorted((stage / "jobs").iterdir()) if (stage / "jobs").exists() else []
    statuses = []
    for job_dir in jobs:
        path = job_dir / "status.json"
        if path.exists():
            statuses.append(json.loads(path.read_text()))
    counts: dict[str, int] = {}
    for item in statuses:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    for path in sorted(stage.rglob("*"), reverse=True):  # container wrote files as root
        path.chmod(0o755 if path.is_dir() else 0o644)
    git_sha = pc.git_head()
    run_id = pc.new_run_id("b-gpu", git_sha)
    manifest = {
        "schema_version": "kvbench-posthoc-review-manifest-1.0.0",
        "status": "PASS" if counts.get("completed") else "FAILED",
        "posthoc": True,
        "label": pc.POSTHOC_LABEL,
        "analysis": "review_round_part_b_gpu_diagnostics",
        "created_at_utc": pc.utc_now(),
        "git_head": git_sha,
        "execution_repository_sha": EXECUTION_SHA,
        "authorized_container_digest": IMAGE,
        "gpu_uuid": GPU_UUID,
        "job_status_counts": counts,
        "jobs": statuses,
        "preflight": preflight_record,
        "profiler_duration_is_normal_timing": False,
        "performance_claim_eligible": False,
        "diagnostic_note": ("TurboQuant split=32 runs swap the decode split count per call; "
                            "they are not a measured configuration"),
    }
    for source in (Path(__file__), POSTHOC_DIR / "partB_worker.py", Path(pc.__file__)):
        destination = stage / "code" / source.name
        if not destination.exists():
            pc.write_new(destination, source.read_bytes())
    final, root = pc.seal(stage, run_id, manifest, "posthoc_review_part_b_gpu_raw", OUTPUT_PARENT)
    print(f"\nSEALED: {final}\nroot_sha256: {root}")
    print("\nPublish to R2 (as root, from the repository root):")
    print("  set -a; . ./.env; set +a")
    print(f"  python3 scripts/r2_artifact.py publish {final.relative_to(REPO)} "
          f"> /tmp/{run_id}-publish.json; cat /tmp/{run_id}-publish.json")
    print(f"  python3 scripts/r2_artifact.py verify {root} > /tmp/{run_id}-verify.json; "
          f"cat /tmp/{run_id}-verify.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--with-kvquant-ncu", action="store_true")
    run = sub.add_parser("run")
    run.add_argument("--with-kvquant-ncu", action="store_true")
    run.add_argument("--resume", type=Path)
    seal = sub.add_parser("seal")
    seal.add_argument("stage", type=Path)
    sub.add_parser("smoke")
    args = parser.parse_args()

    if args.action == "plan":
        jobs = job_list(args.with_kvquant_ncu)
        total = sum(j["estimated_minutes"] for j in jobs)
        for job in jobs:
            print(f"{job['priority']:24s} {job['job_id']:44s} ~{job['estimated_minutes']:6.1f} min "
                  f"~{job['estimated_bytes'] / 1e9:5.1f} GB")
        print(f"{len(jobs)} jobs, estimated {total / 60:.1f} h; free disk "
              f"{shutil.disk_usage(REPO).free / 1e9:.0f} GB")
        return 0
    if args.action == "seal":
        seal_stage(args.stage.resolve(), None)
        return 0

    log = pc.WarningLog()
    record = preflight()
    if args.action == "smoke":
        jobs = [j for j in job_list(False)
                if j["job_id"] in ("nsys-bf16-b1-l32768", "nsys-tq_4bit_nc-b1-l32768-s32")]
        args.with_kvquant_ncu, args.resume = False, None
    else:
        jobs = job_list(args.with_kvquant_ncu)
    if args.resume:
        stage = args.resume.resolve()
        pc.write_new(stage / f"plan-resume-{datetime.now(timezone.utc):%Y%m%dt%H%M%Sz}.json",
                     pc.json_text({"jobs": jobs, "preflight": record, "created_at_utc": pc.utc_now()}))
    else:
        OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)
        stage = OUTPUT_PARENT / f".staging-partb-{datetime.now(timezone.utc):%Y%m%dt%H%M%Sz}"
        stage.mkdir()
        pc.write_new(stage / "plan.json", pc.json_text({"jobs": jobs, "preflight": record,
                                                         "created_at_utc": pc.utc_now()}))
    print(f"staging: {stage}")
    total = sum(j["estimated_minutes"] for j in jobs)
    print(f"{len(jobs)} jobs, estimated {total / 60:.1f} h")
    for index, job in enumerate(jobs, 1):
        print(f"[{index}/{len(jobs)}] {job['job_id']} (~{job['estimated_minutes']:.0f} min) ...", flush=True)
        status = run_job(job, stage, log)
        print(f"    -> {status['status']} {status.get('reason_code') or ''}", flush=True)
    pc.write_new(stage / f"runner_warnings-{datetime.now(timezone.utc):%Y%m%dt%H%M%Sz}.json",
                 pc.json_text(log.entries))
    if args.action == "smoke":
        for job in jobs:
            print(f"--- {job['job_id']}: {stage / 'jobs' / job['job_id']}")
            result = stage / "jobs" / job["job_id"] / "worker_result.json"
            if result.exists():
                payload = json.loads(result.read_text())
                print(json.dumps({k: payload.get(k) for k in (
                    "status", "error", "output_finite", "cache_pointers_stable", "tq_split_override",
                    "prefix_source", "adapter_fingerprint_validation_error")}, indent=1, default=str))
        print(f"\nSmoke run kept unsealed. Continue with:\n  python3 {Path(__file__).resolve()} "
              f"run --resume {stage}")
        return 0
    seal_stage(stage, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
