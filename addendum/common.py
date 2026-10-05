"""Shared constants and helpers for the 2026-10-05 post-hoc addendum.

Host-side code (driver.py) and container-side code (workers, mounted read-only
at /opt/addendum) both import this module.  It only reads frozen inputs; every
output goes under results/addendum-20261005/.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping

ADDENDUM_ID = "addendum-20261005"
AMENDMENT = "docs/amendment-20261005.md"

# Frozen identities (Full Scan timing environment).
REPO = Path("/home/rockrock/cmu_paper")
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
CONTAINER_REPO = "/home/rockrock/cmu_paper"
CONTAINER_ADDENDUM = "/opt/addendum"
CONTAINER_OUT = "/opt/addendum-out"
CONTAINER_INPUTS = "/opt/addendum-inputs"

ADDENDUM_REPO = Path(__file__).resolve().parents[1]
RESULTS = ADDENDUM_REPO / "results" / ADDENDUM_ID

# Full Scan protocol constants (scripts/phase16_full_scan.py).
WARMUP_STEPS = 64
MEASURED_STEPS = 256
MEASURED_BATCHES = 5
CV_THRESHOLD = 0.03
PROCESSES_PER_POINT = 3

LABELS = {"bf16": "BF16", "tq_4bit_nc": "TQ-4bit", "tq_k3v4_nc": "TQ-k3v4",
          "tq_3bit_nc": "TQ-3bit", "k4v4": "KIVI-k4v4", "k2v2": "KIVI-k2v2"}


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, default=str) + "\n"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path: Path, text: str) -> None:
    """Create a file that must not exist yet (raw data is append-only)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_failure(task: str, text: str) -> None:
    path = RESULTS / "FAILURES.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(f"\n## {utc_now()} {task}\n\n{text.rstrip()}\n")
        handle.flush()
        os.fsync(handle.fileno())


def host_wall_process_median_ms(payload: Mapping[str, Any]) -> tuple[float, list[float]]:
    """Same rule as scripts/phase16_wall_closure.py exact_wall_median."""
    samples = payload["runner"]["timing"]["samples"]
    if len(samples) != MEASURED_BATCHES:
        raise ValueError("raw timing sample count differs")
    values = []
    for sample in samples:
        if sample["completed_operations"] != MEASURED_STEPS or sample["failed_operations"] != 0:
            raise ValueError("raw timing operation count differs")
        value = float(sample["host_ns_per_operation"])
        if not value > 0:
            raise ValueError("invalid raw host timing")
        values.append(value / 1_000_000)
    return statistics.median(values), values


def point_statistics(process_medians: Iterable[float]) -> dict[str, Any]:
    """Same rule as scripts/phase16_full_scan.py point_statistics."""
    values = list(process_medians)
    if len(values) < 2:
        return {"n": len(values), "median_ms": values[0] if values else None, "cv": None}
    mean = statistics.mean(values)
    return {"n": len(values), "median_ms": statistics.median(values),
            "cv": statistics.stdev(values) / mean}
