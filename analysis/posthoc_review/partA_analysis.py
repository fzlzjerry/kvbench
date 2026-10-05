#!/usr/bin/env python3
"""POST-HOC Part A: traffic ceilings, effective bandwidth, within-family byte
elasticity, and steady-state capacity, computed from frozen KVBench evidence.

POST-HOC.  Every quantity in this script was defined after the frozen
performance and quality results were known (review round of 2026-10-04).  It
re-runs no timing, launches no CUDA work, never uses profiler durations as
timing, and never writes into an existing evidence directory.  Its inputs are
three published, content-addressed roots, each verified before use:

  * Phase 16R host-wall closure (5605558b...): per-session process medians,
    point medians, allocated-byte accounting, same-work ratios.
  * Phase 16 outer root (d7458767...): the preregistered feasibility formula
    and its per-point memory components (predictions, not measurements).
  * Phase 15 profiler root (641fc02d...): Nsight Compute DRAM bytes per kernel
    at B = 1 (cold cache, base clocks; see LIMITS below).

Definitions (all modeled quantities are labeled "modeled"):

  C_m(B, L)      allocated cache bytes of configuration m (byte accounting).
  alpha_m        DRAM bytes moved by m's cache-path kernels per allocated byte,
                 measured once at B = 1, L = 128K (Phase 15).
  N_f(B, L)      non-cache DRAM bytes, n0 + n1 * B * (L + 1), one fit per
                 family over every existing B = 1 profile of that family.  The
                 B-extension assumes the L-dependent term is per sequence.
  D_m(B, L)      modeled total traffic, N_f(B, L) + alpha_m * C_m(B, L).
  D_alg,m(B, L)  algorithm version (A = 1): N_BF16(B, L) + alpha_BF16 * C_m(B, L),
                 i.e. m reads its allocation with BF16's efficiency and adds no
                 method-specific non-cache traffic.
  S              measured same-work ratio T_BF16 / T_m (host wall, frozen).
  S_eq           D_BF16 / D_m (equal-efficiency ratio; modeled).
  S_roof         T_BF16 * BW_peak / D_m (roofline ceiling; modeled).

Kernel classification.  "v1" is the Phase 15 classification as published.
"v2" is a post-hoc correction that moves KVQuant's key-path kernels
(VecQuant{2,3,4}MatMul...TransposedRopeMHABatchedFusedOpt, which reads
packed_key_cache, and SPMV_ATOMIC_ROPE_BALANCED, the sparse key outliers) from
other_model into the cache path; their names contain "rope", which matched the
generic other_model rule.  Only KVQuant changes.  v2 is used for the traffic
model; v1 numbers are reported next to it.

LIMITS.  Phase 15 Nsight Compute ran with its defaults (--cache-control all,
--clock-control base): DRAM bytes are cold-L2 per kernel replay.  Every memory
number in the capacity section is a prediction of the preregistered
feasibility formula, not a measurement.  The session bootstrap reflects
process-to-process timing noise only.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import pyarrow.parquet as pq

import posthoc_common as pc

REPO = pc.REPO
FULL_SCAN = REPO / "artifacts/phase16/phase16-20260831t123029614620z-ec534d99-de80ac"
WALL_DIR = FULL_SCAN / "wall-closure"
WALL_ROOT = "5605558be0483ddfeffd251977306d3397aa27a66309324c6011e5043584103e"
OUTER_DIR = FULL_SCAN / "outer"
OUTER_ROOT = "d74587675dd59b464d81c6e82885d3c9706c681a9da216ad1a7fe4c6ccd88daa"
PH15_DIR = REPO / "artifacts/phase15/phase15-20260828t144810363697z-446b334e-90460f"
PH15_ROOT = "641fc02d8fa598097885b74a336b1b1f454d9844b90025cf0c4b427bee02d5e8"

CONFIGS = ("bf16", "tq_4bit_nc", "tq_k3v4_nc", "tq_3bit_nc", "k4v4", "k2v4", "k2v2",
           "kvq4", "kvq3", "kvq2")
FAMILY_OF = {"bf16": "bf16", "tq_4bit_nc": "turboquant", "tq_k3v4_nc": "turboquant",
             "tq_3bit_nc": "turboquant", "k4v4": "kivi", "k2v4": "kivi", "k2v2": "kivi",
             "kvq4": "kvquant", "kvq3": "kvquant", "kvq2": "kvquant"}
FAMILIES = ("turboquant", "kivi", "kvquant")
FAMILY_LABELS = {"bf16": "BF16", "turboquant": "TurboQuant (as-ported)", "kivi": "KIVI",
                 "kvquant": "KVQuant (as-ported)"}
FAMILY_CONFIGS = {family: tuple(c for c in CONFIGS if FAMILY_OF[c] == family)
                  for family in ("bf16", *FAMILIES)}
BATCHES = (1, 2, 4, 8, 16)
L128 = 131071
L4 = 4096
GB = 1e9

PEAK_BW_DATASHEET = 1792e9  # bytes per second
PEAK_BW_SOURCE = ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition published memory "
                  "bandwidth, 1,792 GB/s (96 GB GDDR7, 512-bit); cross-checked against the "
                  "Nsight Compute device attributes recorded in the Phase 15 reports")
KVQ_KEY_PATTERNS = ("matmulkernelnuqperchanneltransposedropemhabatchedfusedopt",
                    "spmv_atomic_rope_balanced")
# Fixed before any slope was computed (review decision: report elasticity only
# where cache traffic is a sizeable share of the step's modeled DRAM traffic).
ELASTICITY_MIN_CACHE_SHARE = 0.25
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20261004
SESSIONS = (0, 1, 2, 3, 4)

# Feasibility-formula constants (scripts/phase13_pilot.py), used only to give
# BF16 a byte count at adaptive contexts where the grid has no BF16 point.
MODEL_WEIGHT_BYTES = 16_060_556_288
REFERENCE_CAPACITY = 4097


def bf16_cache_bytes(batch: int, historical: int) -> int:
    capacity = historical + 1
    return 2 * 32 * batch * 8 * capacity * 128 * 2 + 163_840


# ---------------------------------------------------------------- loading

def load_inputs(log: pc.WarningLog) -> dict[str, Any]:
    wall = pc.verify_full_root(WALL_DIR, WALL_ROOT)
    outer = pc.verify_full_root(OUTER_DIR, OUTER_ROOT)
    ph15 = pc.LedgerRoot(PH15_DIR, PH15_ROOT)

    points = {}
    for row in pq.read_table(WALL_DIR / "point_summary.parquet").to_pylist():
        key = (str(row["method_config_id"]), int(row["batch_size"]), int(row["historical_context"]))
        if key in points:
            raise pc.PosthocError(f"duplicate point {key}")
        points[key] = row

    sessions: dict[tuple, dict[int, float]] = defaultdict(dict)
    columns = ["method_config_id", "batch_size", "historical_context", "replicate_index",
               "status", "host_wall_process_median_ms", "latency_basis"]
    for row in pq.read_table(WALL_DIR / "raw_run_index.parquet", columns=columns).to_pylist():
        if row["status"] != "completed":
            continue
        if row["latency_basis"] != "host_wall":
            raise pc.PosthocError("non host-wall process median in the wall closure")
        key = (str(row["method_config_id"]), int(row["batch_size"]), int(row["historical_context"]))
        replicate = int(row["replicate_index"])
        if replicate in sessions[key]:
            raise pc.PosthocError(f"two completed runs for {key} session {replicate}")
        sessions[key][replicate] = float(row["host_wall_process_median_ms"])

    for key, point in points.items():
        if point["disposition"] != "stable":
            continue
        values = sessions.get(key, {})
        if sorted(values) != list(SESSIONS):
            raise pc.PosthocError(f"{key} lacks one process median per session")
        if abs(float(np.median(list(values.values()))) - float(point["median_ms"])) > 1e-9:
            raise pc.PosthocError(f"{key} session medians do not reproduce the point median")
        recorded = point["process_medians_ms"]
        if isinstance(recorded, str):
            recorded = json.loads(recorded)
        if sorted(values.values()) != sorted(float(v) for v in recorded):
            raise pc.PosthocError(f"{key} session medians differ from the point record")

    same_work = {}
    for row in pq.read_table(WALL_DIR / "same_work_ratios.parquet").to_pylist():
        if row["latency_basis"] != "host_wall":
            raise pc.PosthocError("same-work table is not host-wall")
        if row["calculated"]:
            key = (str(row["method_config_id"]), int(row["batch_size"]), int(row["historical_context"]))
            same_work[key] = float(row["performance_only_ratio"])
    if len(same_work) != 357:
        raise pc.PosthocError(f"expected 357 same-work ratios, found {len(same_work)}")

    feasibility = {}
    for row in pq.read_table(OUTER_DIR / "feasibility.parquet").to_pylist():
        key = (str(row["method_config_id"]), int(row["batch_size"]), int(row["historical_context"]))
        feasibility[key] = row
    capacity_points = pq.read_table(OUTER_DIR / "capacity_amplification.parquet").to_pylist()

    ncu_index = pq.read_table(ph15.path("ncu_run_index.parquet")).to_pylist()
    kernel_columns = ["run_id", "kernel_name", "kernel_role", "cache_path", "dram_bytes",
                      "kernel_duration_ns"]
    kernels = pq.read_table(ph15.path("kernel_metrics.parquet"), columns=kernel_columns).to_pylist()

    return {
        "roots": {"wall": wall, "outer": outer},
        "ph15": ph15,
        "points": points,
        "sessions": {key: value for key, value in sessions.items()},
        "same_work": same_work,
        "feasibility": feasibility,
        "capacity_points": capacity_points,
        "ncu_index": ncu_index,
        "kernels": kernels,
    }


def device_peak(ph15: pc.LedgerRoot, log: pc.WarningLog) -> dict[str, Any]:
    """Read the memory clock and bus width that Nsight Compute recorded.

    Noncritical: if ncu is unavailable the published figure is used alone.
    """

    record: dict[str, Any] = {
        "peak_bw_bytes_per_s": PEAK_BW_DATASHEET,
        "peak_bw_source": PEAK_BW_SOURCE,
    }
    ncu = shutil.which("ncu")
    if ncu is None:
        log.warn("ncu_unavailable", "device attributes not re-read; published peak used alone")
        return record
    relative = "raw/ncu/ncu-bf16-b1-l131072-cuda_graph-attempt2.ncu-rep"

    def read() -> dict[str, Any]:
        report = ph15.path(relative)
        command = [ncu, "--import", str(report), "--page", "raw", "--csv",
                   "--kernel-name", "regex:flash_fwd_splitkv_kernel", "--launch-count", "1"]
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=900)
        rows = list(csv.reader(io.StringIO(result.stdout)))
        header, values = rows[0], rows[2]
        table = dict(zip(header, values))

        def number(name: str) -> float:
            return float(table[name].replace(",", ""))

        clock_khz = number("device__attribute_memory_clock_rate")
        bus_bits = number("device__attribute_global_memory_bus_width")
        derived = clock_khz * 1e3 * 2 * bus_bits / 8
        read_bytes = number("dram__bytes_op_read.sum") * 1e9
        write_bytes = number("dram__bytes_op_write.sum") * 1e6
        duration_s = number("gpu__time_duration.sum") * 1e-6
        throughput_pct = number("dram__throughput.avg.pct_of_peak_sustained_elapsed")
        sustained = (read_bytes + write_bytes) / duration_s / (throughput_pct / 100.0)
        return {
            "report": relative,
            "device_name": table.get("device__attribute_display_name"),
            "memory_clock_rate_khz": clock_khz,
            "global_memory_bus_width_bits": bus_bits,
            "derived_peak_bytes_per_s": derived,
            "derivation": "memory_clock_rate x 2 (double data rate) x bus_width / 8",
            "l2_cache_size_bytes": number("device__attribute_l2_cache_size"),
            "multiprocessor_count": number("device__attribute_multiprocessor_count"),
            "ncu_peak_sustained_estimate_bytes_per_s": sustained,
            "ncu_peak_sustained_derivation": (
                "BF16 flash_fwd_splitkv_kernel, first launch: DRAM bytes / duration / "
                "dram__throughput.avg.pct_of_peak_sustained_elapsed (profiler clocks)"),
            "command": [Path(command[0]).name, *command[1:2], relative, *command[3:]],
        }

    attributes = log.guard("ncu_device_attributes", read)
    if attributes:
        record["ncu_device_attributes"] = attributes
        if abs(attributes["derived_peak_bytes_per_s"] - PEAK_BW_DATASHEET) / PEAK_BW_DATASHEET > 0.01:
            log.warn("peak_mismatch", "ncu-derived peak differs from the published figure by >1%")
    return record


# ---------------------------------------------------------------- traffic

def run_traffic(data: dict[str, Any]) -> tuple[dict[tuple, dict[str, Any]], list[dict[str, Any]]]:
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for kernel in data["kernels"]:
        by_run[kernel["run_id"]].append(kernel)
    traffic: dict[tuple, dict[str, Any]] = {}
    reclassified: list[dict[str, Any]] = []
    for row in data["ncu_index"]:
        if row["status"] != "completed" or row["graph_mode"] != "cuda_graph":
            continue
        configuration = str(row["method_config_id"])
        key = (configuration, int(row["batch_size"]), int(row["historical_context"]))
        kernels = by_run[row["run_id"]]
        total = sum(float(k["dram_bytes"] or 0) for k in kernels)
        cache_v1 = sum(float(k["dram_bytes"] or 0) for k in kernels if k["cache_path"])
        moved = 0.0
        per_symbol: dict[str, float] = defaultdict(float)
        if FAMILY_OF[configuration] == "kvquant":
            for kernel in kernels:
                name = str(kernel["kernel_name"]).lower()
                if not kernel["cache_path"] and any(p in name for p in KVQ_KEY_PATTERNS):
                    moved += float(kernel["dram_bytes"] or 0)
                    per_symbol[str(kernel["kernel_name"]).split("(")[0]] += float(kernel["dram_bytes"] or 0)
        for name, value in sorted(per_symbol.items()):
            reclassified.append({"run_id": row["run_id"], "method_config_id": configuration,
                                 "historical_context": key[2], "kernel_symbol": name,
                                 "v1_role": "other_model", "v2_role": "cache_path_key",
                                 "dram_bytes": value})
        if abs(total - float(row["total_decode_dram_bytes"])) > 1e-6 * total:
            raise pc.PosthocError(f"{row['run_id']} kernel bytes do not sum to the run total")
        if abs(cache_v1 - float(row["cache_path_dram_bytes"])) > 1e-6 * max(cache_v1, 1.0):
            raise pc.PosthocError(f"{row['run_id']} cache-path bytes differ from the run index")
        if key in traffic:
            raise pc.PosthocError(f"two completed profiles for {key}")
        traffic[key] = {
            "run_id": row["run_id"],
            "total": total,
            "cache_v1": cache_v1,
            "cache_v2": cache_v1 + moved,
            "kernel_count": len(kernels),
        }
    return traffic, reclassified


def ols(xs: list[float], ys: list[float]) -> tuple[float, float]:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    xc = x - x.mean()
    slope = float((xc * (y - y.mean())).sum() / (xc * xc).sum())
    return float(y.mean() - slope * x.mean()), slope


class TrafficModel:
    def __init__(self, data: dict[str, Any], traffic: dict[tuple, dict[str, Any]]) -> None:
        self.points = data["points"]
        self.feasibility = data["feasibility"]
        self.traffic = traffic
        self.alpha: dict[str, dict[str, float]] = {"v1": {}, "v2": {}}
        self.noncache: dict[str, dict[str, dict[str, Any]]] = {"v1": {}, "v2": {}}
        for version in ("v1", "v2"):
            for configuration in CONFIGS:
                measured = traffic[(configuration, 1, L128)][f"cache_{version}"]
                self.alpha[version][configuration] = measured / self.cache_bytes(configuration, 1, L128)
            for family in ("bf16", *FAMILIES):
                xs, ys, used = [], [], []
                for (configuration, batch, historical), row in sorted(traffic.items()):
                    if FAMILY_OF[configuration] != family or batch != 1:
                        continue
                    xs.append(float(historical + 1))
                    ys.append(row["total"] - row[f"cache_{version}"])
                    used.append(row["run_id"])
                n0, n1 = ols(xs, ys)
                residuals = [y - (n0 + n1 * x) for x, y in zip(xs, ys)]
                self.noncache[version][family] = {
                    "n0_bytes": n0, "n1_bytes_per_sequence_token": n1,
                    "profiles": used, "max_abs_residual_bytes": max(abs(r) for r in residuals),
                    "max_abs_residual_fraction_of_total": max(
                        abs(r) / traffic[_key_of(run, traffic)]["total"]
                        for r, run in zip(residuals, used)),
                }

    def cache_bytes(self, configuration: str, batch: int, historical: int) -> float:
        """Allocated bytes: byte accounting of measured points, else the
        feasibility formula (capacity-infeasible points were never allocated)."""

        key = (configuration, batch, historical)
        point = self.points.get(key)
        recorded = None
        if point is not None and point["allocated_bytes"] is not None:
            recorded = int(point["allocated_bytes"])
        elif key in self.feasibility:
            recorded = int(self.feasibility[key]["cache_allocated_bytes"])
        if configuration == "bf16":
            value = bf16_cache_bytes(batch, historical)
            if recorded is not None and recorded != value:
                raise pc.PosthocError("BF16 byte formula differs from the byte accounting")
            return float(value)
        if recorded is None:
            raise pc.PosthocError(f"no byte accounting for {key}")
        if point is not None and point["allocated_bytes"] is not None and key in self.feasibility:
            if int(self.feasibility[key]["cache_allocated_bytes"]) != recorded:
                raise pc.PosthocError(f"byte accounting and feasibility formula differ at {key}")
        return float(recorded)

    def stored_bytes(self, configuration: str, batch: int, historical: int) -> float:
        if configuration == "bf16":
            return self.cache_bytes(configuration, batch, historical)
        point = self.points[(configuration, batch, historical)]
        return float(point["allocated_bytes"]) - float(point["workspace_bytes"] or 0) - float(
            point["padding_bytes"] or 0)

    def noncache_bytes(self, version: str, family: str, batch: int, historical: int) -> float:
        fit = self.noncache[version][family]
        return fit["n0_bytes"] + fit["n1_bytes_per_sequence_token"] * batch * (historical + 1)

    def cache_traffic(self, version: str, configuration: str, batch: int, historical: int) -> float:
        return self.alpha[version][configuration] * self.cache_bytes(configuration, batch, historical)

    def total(self, version: str, configuration: str, batch: int, historical: int) -> float:
        return (self.noncache_bytes(version, FAMILY_OF[configuration], batch, historical)
                + self.cache_traffic(version, configuration, batch, historical))

    def algorithm_total(self, configuration: str, batch: int, historical: int,
                        stored: bool = False) -> float:
        cache = (self.stored_bytes if stored else self.cache_bytes)(configuration, batch, historical)
        return self.noncache_bytes("v2", "bf16", batch, historical) + self.alpha["v2"]["bf16"] * cache

    def paper_amplification(self, version: str, configuration: str) -> float:
        return self.alpha[version][configuration] / self.alpha[version]["bf16"]


def _key_of(run_id: str, traffic: dict[tuple, dict[str, Any]]) -> tuple:
    for key, row in traffic.items():
        if row["run_id"] == run_id:
            return key
    raise KeyError(run_id)


def traffic_validation(model: TrafficModel) -> list[dict[str, Any]]:
    rows = []
    for (configuration, batch, historical), measured in sorted(model.traffic.items()):
        if historical == L128:
            continue
        row = {"method_config_id": configuration, "batch_size": batch,
               "historical_context": historical, "run_id": measured["run_id"],
               "measured_total_bytes": measured["total"]}
        for version in ("v1", "v2"):
            cache_model = model.cache_traffic(version, configuration, batch, historical)
            total_model = model.total(version, configuration, batch, historical)
            row[f"measured_cache_{version}_bytes"] = measured[f"cache_{version}"]
            row[f"modeled_cache_{version}_bytes"] = cache_model
            row[f"cache_{version}_relative_error"] = cache_model / measured[f"cache_{version}"] - 1
            row[f"modeled_total_{version}_bytes"] = total_model
            row[f"total_{version}_relative_error"] = total_model / measured["total"] - 1
        row["note"] = ("cache term is out of sample (alpha from 128K); non-cache fit includes "
                       "this profile")
        rows.append(row)
    return rows


# ---------------------------------------------------------------- A.1 ratios

def roofline_rows(data: dict[str, Any], model: TrafficModel, peak: float,
                  exclusions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    points = data["points"]
    for (configuration, batch, historical), point in sorted(points.items()):
        if configuration == "bf16" or point["disposition"] != "stable":
            continue
        key = (configuration, batch, historical)
        bf16 = points.get(("bf16", batch, historical))
        bf16_time = float(bf16["median_ms"]) if bf16 and bf16["disposition"] == "stable" else None
        method_time = float(point["median_ms"])
        measured = data["same_work"].get(key)
        bf16_total = model.total("v2", "bf16", batch, historical)
        method_total = {v: model.total(v, configuration, batch, historical) for v in ("v1", "v2")}
        algorithm_total = model.algorithm_total(configuration, batch, historical)
        stored_total = model.algorithm_total(configuration, batch, historical, stored=True)
        row = {
            "method_config_id": configuration, "family": FAMILY_OF[configuration],
            "batch_size": batch, "historical_context": historical,
            "grid_source": point["grid_source"],
            "t_method_ms": method_time, "t_bf16_ms": bf16_time,
            "s_measured": measured,
            "modeled_traffic_bf16_bytes": bf16_total,
            "modeled_traffic_method_v2_bytes": method_total["v2"],
            "modeled_traffic_method_v1_bytes": method_total["v1"],
            "modeled_traffic_algorithm_bytes": algorithm_total,
            "modeled_cache_share_v2": model.cache_traffic("v2", configuration, batch, historical)
            / method_total["v2"],
            "s_eq_port_v2": bf16_total / method_total["v2"],
            "s_eq_port_v1": bf16_total / method_total["v1"],
            "s_eq_alg": bf16_total / algorithm_total,
            "s_eq_alg_stored": bf16_total / stored_total,
            "method_bw_fraction_of_peak_v2": method_total["v2"] / (method_time * 1e-3) / peak,
        }
        if bf16_time is None:
            reason = ("bf16_capacity_infeasible" if ("bf16", batch, historical) in points
                      else "no_bf16_point_at_adaptive_context")
            exclusions.append({"analysis": "A1_s_roof", "key": list(key), "reason_code": reason,
                               "detail": "S_roof needs a measured BF16 time at the same (B, L)"})
            for name in ("s_roof_port_v2", "s_roof_port_v1", "s_roof_alg", "s_roof_alg_stored",
                         "bf16_bw_fraction_of_peak", "s_over_s_roof_alg"):
                row[name] = None
        else:
            seconds = bf16_time * 1e-3
            row["s_roof_port_v2"] = seconds * peak / method_total["v2"]
            row["s_roof_port_v1"] = seconds * peak / method_total["v1"]
            row["s_roof_alg"] = seconds * peak / algorithm_total
            row["s_roof_alg_stored"] = seconds * peak / stored_total
            row["bf16_bw_fraction_of_peak"] = bf16_total / seconds / peak
            row["s_over_s_roof_alg"] = measured / row["s_roof_alg"] if measured is not None else None
        if measured is None and bf16_time is not None:
            exclusions.append({"analysis": "A1_s_measured", "key": list(key),
                               "reason_code": "no_same_work_ratio",
                               "detail": "frozen same-work table has no ratio at this point"})
        rows.append(row)
    return rows


# ---------------------------------------------------------------- A.2 BW_eff

def bandwidth_rows(data: dict[str, Any], model: TrafficModel, peak: float) -> list[dict[str, Any]]:
    rows = []
    for configuration in CONFIGS:
        t128 = float(data["points"][(configuration, 1, L128)]["median_ms"])
        t4 = float(data["points"][(configuration, 1, L4)]["median_ms"])
        measured = model.traffic[(configuration, 1, L128)]
        row = {
            "method_config_id": configuration, "family": FAMILY_OF[configuration],
            "t_b1_128k_ms": t128, "t_b1_4k_ms": t4,
            "total_dram_128k_bytes": measured["total"],
            "bw_total_bytes_per_s": measured["total"] / (t128 * 1e-3),
        }
        row["bw_total_pct_peak"] = 100 * row["bw_total_bytes_per_s"] / peak
        delta = (t128 - t4) * 1e-3
        for version in ("v1", "v2"):
            cache128 = measured[f"cache_{version}"]
            low = model.traffic.get((configuration, 1, L4))
            if low is not None:
                cache4, source = low[f"cache_{version}"], "measured"
            else:
                cache4, source = model.cache_traffic(version, configuration, 1, L4), "modeled"
            row[f"cache_dram_128k_{version}_bytes"] = cache128
            row[f"cache_dram_4k_{version}_bytes"] = cache4
            row[f"cache_dram_4k_{version}_source"] = source
            row[f"bw_cache_{version}_bytes_per_s"] = cache128 / delta
            row[f"bw_cache_{version}_pct_peak"] = 100 * cache128 / delta / peak
            row[f"bw_cache_marginal_{version}_bytes_per_s"] = (cache128 - cache4) / delta
            row[f"bw_cache_marginal_{version}_pct_peak"] = 100 * (cache128 - cache4) / delta / peak
            row[f"r_dram_{version}"] = model.traffic[("bf16", 1, L128)][f"cache_{version}"] / cache128
            row[f"amplification_A_{version}"] = model.paper_amplification(version, configuration)
        rows.append(row)
    return rows


# ---------------------------------------------------------------- A.5 decomposition

def decomposition_rows(data: dict[str, Any], model: TrafficModel) -> list[dict[str, Any]]:
    """Descriptive fit T = c0 + D_cache / BW_eff along L at each fixed B."""

    rows = []
    baseline: dict[int, float] = {}
    for configuration in CONFIGS:
        for batch in BATCHES:
            xs, ys = [], []
            for (c, b, historical), point in sorted(data["points"].items()):
                if c == configuration and b == batch and point["disposition"] == "stable":
                    xs.append(model.cache_traffic("v2", configuration, batch, historical))
                    ys.append(float(point["median_ms"]))
            if len(xs) < 3 or max(xs) < 2 * min(xs):
                rows.append({"method_config_id": configuration, "batch_size": batch,
                             "points": len(xs), "status": "not_fitted_insufficient_span"})
                continue
            c0, c1 = ols(xs, ys)
            fitted = [c0 + c1 * x for x in xs]
            ss_res = sum((y - f) ** 2 for y, f in zip(ys, fitted))
            ss_tot = sum((y - sum(ys) / len(ys)) ** 2 for y in ys)
            if configuration == "bf16":
                baseline[batch] = c0
            rows.append({
                "method_config_id": configuration, "family": FAMILY_OF[configuration],
                "batch_size": batch, "points": len(xs), "status": "fitted",
                "c0_ms": c0, "slope_ms_per_gb": c1 * GB,
                "bw_eff_cache_bytes_per_s": (1e3 / c1) if c1 > 0 else None,
                "r_squared": 1 - ss_res / ss_tot if ss_tot > 0 else None,
                "max_abs_relative_residual": max(abs(y / f - 1) for y, f in zip(ys, fitted)),
                "cache_traffic_min_bytes": min(xs), "cache_traffic_max_bytes": max(xs),
            })
    for row in rows:
        if row.get("status") == "fitted":
            base = baseline.get(row["batch_size"])
            row["c0_excess_over_bf16_ms"] = (row["c0_ms"] - base) if base is not None else None
    return rows


# ---------------------------------------------------------------- A.3 elasticity

def elasticity(data: dict[str, Any], model: TrafficModel,
               exclusions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(SESSIONS), size=(BOOTSTRAP_DRAWS, len(SESSIONS)))
    rows: list[dict[str, Any]] = []
    pooled_inputs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    geometries = sorted({(b, h) for (_, b, h) in data["points"]})
    for family in FAMILIES:
        configurations = FAMILY_CONFIGS[family]
        for batch, historical in geometries:
            keys = [(c, batch, historical) for c in configurations]
            present = [k for k in keys if k in data["points"] and data["points"][k]["disposition"] == "stable"]
            if len(present) < 3:
                if any(k in data["points"] for k in keys):
                    exclusions.append({"analysis": "A3_elasticity", "key": [family, batch, historical],
                                       "reason_code": "fewer_than_three_configs_measured",
                                       "detail": f"{len(present)} of 3 configurations measured"})
                continue
            log_c = np.log([model.cache_bytes(*k) for k in keys])
            sessions = np.array([[data["sessions"][k][s] for s in SESSIONS] for k in keys])
            log_t_point = np.log(np.median(sessions, axis=1))
            log_d = np.log([model.total("v2", *k) for k in keys])
            share = [model.cache_traffic("v2", *k) / model.total("v2", *k) for k in keys]
            xc = log_c - log_c.mean()
            slope = float((xc * (log_t_point - log_t_point.mean())).sum() / (xc * xc).sum())
            expected = float((xc * (log_d - log_d.mean())).sum() / (xc * xc).sum())
            boot_t = np.log(np.median(sessions[:, draws], axis=2))  # (3, draws)
            boot = (xc[:, None] * (boot_t - boot_t.mean(axis=0))).sum(axis=0) / (xc * xc).sum()
            mean_share = float(np.mean(share))
            included = mean_share >= ELASTICITY_MIN_CACHE_SHARE
            if not included:
                exclusions.append({"analysis": "A3_elasticity", "key": [family, batch, historical],
                                   "reason_code": "expected_elasticity_below_threshold",
                                   "detail": f"mean modeled cache share {mean_share:.3f} < "
                                             f"{ELASTICITY_MIN_CACHE_SHARE}"})
            row = {
                "family": family, "batch_size": batch, "historical_context": historical,
                "included": included, "mean_cache_share_v2": mean_share,
                "expected_slope_bandwidth_bound": expected,
                "observed_slope": slope,
                "observed_slope_ci_low": float(np.percentile(boot, 2.5)),
                "observed_slope_ci_high": float(np.percentile(boot, 97.5)),
                "ci_scope": "process-session bootstrap; timing noise only",
                "log_cache_bytes_span": float(log_c.max() - log_c.min()),
                "configs": list(configurations),
            }
            rows.append(row)
            if included:
                pooled_inputs[family].append({"log_c": log_c, "log_d": log_d, "sessions": sessions,
                                              "log_t": log_t_point})
    pooled = []
    for family in FAMILIES:
        items = pooled_inputs.get(family, [])
        if not items:
            pooled.append({"family": family, "points": 0})
            continue
        numerator = denominator = expected_num = 0.0
        boot_num = np.zeros(BOOTSTRAP_DRAWS)
        for item in items:
            xc = item["log_c"] - item["log_c"].mean()
            numerator += float((xc * (item["log_t"] - item["log_t"].mean())).sum())
            expected_num += float((xc * (item["log_d"] - item["log_d"].mean())).sum())
            denominator += float((xc * xc).sum())
            boot_t = np.log(np.median(item["sessions"][:, draws], axis=2))
            boot_num += (xc[:, None] * (boot_t - boot_t.mean(axis=0))).sum(axis=0)
        boot = boot_num / denominator
        pooled.append({
            "family": family, "points": len(items),
            "pooled_observed_slope": numerator / denominator,
            "pooled_ci_low": float(np.percentile(boot, 2.5)),
            "pooled_ci_high": float(np.percentile(boot, 97.5)),
            "pooled_expected_slope_bandwidth_bound": expected_num / denominator,
            "ci_scope": "process-session bootstrap; timing noise only",
            "confounding_note": (
                "bit width also changes the kernel specialization (TurboQuant Triton constexpr "
                "variants); the slope mixes byte and kernel-variant effects"
                if family == "turboquant" else
                "bit width changes cache bytes and packing; kernels are the same symbols"),
        })
    return rows, pooled


# ---------------------------------------------------------------- A.4 capacity

def capacity(data: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    by_key = {}
    for key, record in sorted(data["feasibility"].items()):
        fixed = int(record["fixed_decode_input_bytes"])
        steady = (int(record["model_weight_bytes"]) + int(record["cache_allocated_bytes"])
                  + int(record["endpoint_workspace_bytes"])
                  + int(record["graph_pool_or_capture_reserve_bytes"]) + fixed)
        transient = (int(record["prefix_control_tensor_bytes"]) - fixed
                     + int(record["prefix_compute_peak_bytes"])
                     + int(record["kvquant_prefix_chunk_workspace_bytes"] or 0))
        predicted = int(record["predicted_required_bytes"])
        if steady + transient != predicted:
            raise pc.PosthocError(f"memory components of {key} do not add up")
        limit = int(record["limit_bytes"])
        dominant = ("mlp" if int(record["mlp_peak_bytes"]) >= int(record["attention_output_projection_peak_bytes"])
                    else "attention_output_projection")
        row = {
            "method_config_id": key[0], "batch_size": key[1], "historical_context": key[2],
            "grid_source": record["grid_source"],
            "steady_state_predicted_bytes": steady,
            "transient_predicted_bytes": transient,
            "prefix_compute_peak_bytes": int(record["prefix_compute_peak_bytes"]),
            "prefix_compute_peak_term": dominant,
            "graph_pool_predicted_bytes": int(record["graph_pool_or_capture_reserve_bytes"]),
            "end_to_end_predicted_bytes": predicted, "limit_bytes": limit,
            "feasible_end_to_end": record["status"] == "feasible",
            "feasible_steady_state": steady <= limit,
            "memory_numbers": "predicted by the preregistered feasibility formula; not measured",
        }
        rows.append(row)
        by_key[key] = row

    def bf16_steady(batch: int, historical: int) -> int:
        row = by_key.get(("bf16", batch, historical))
        if row is not None:
            return row["steady_state_predicted_bytes"]
        reference = next(r for k, r in data["feasibility"].items() if k[0] == "bf16")
        capacity_tokens = historical + 1
        graph = math.ceil((int(reference["graph_reserve_reference_bytes"])
                           - int(reference["graph_reserve_reference_endpoint_workspace_bytes"]))
                          * batch * capacity_tokens / REFERENCE_CAPACITY)
        endpoint = 32 * batch * (32 + 8) * 64 * 2
        fixed = batch * 8 + 8 + 2 * 128 * 2
        return MODEL_WEIGHT_BYTES + bf16_cache_bytes(batch, historical) + endpoint + graph + fixed

    for (configuration, batch, historical), row in by_key.items():
        if configuration != "bf16":
            continue
        reference = data["feasibility"][(configuration, batch, historical)]
        capacity_tokens = historical + 1
        graph = math.ceil((int(reference["graph_reserve_reference_bytes"])
                           - int(reference["graph_reserve_reference_endpoint_workspace_bytes"]))
                          * batch * capacity_tokens / REFERENCE_CAPACITY)
        formula = (MODEL_WEIGHT_BYTES + bf16_cache_bytes(batch, historical)
                   + 32 * batch * (32 + 8) * 64 * 2 + graph + batch * 8 + 8 + 2 * 128 * 2)
        if formula != row["steady_state_predicted_bytes"]:
            raise pc.PosthocError(f"BF16 steady-state formula differs from the grid at {batch}, {historical}")

    limit = rows[0]["limit_bytes"]
    published = []
    for item in data["capacity_points"]:
        key = (str(item["method_config_id"]), int(item["batch_size"]), int(item["historical_context"]))
        steady = bf16_steady(key[1], key[2])
        published.append({"method_config_id": key[0], "batch_size": key[1],
                          "historical_context": key[2],
                          "bf16_end_to_end_predicted_bytes": int(item["bf16_predicted_bytes"]),
                          "bf16_steady_state_predicted_bytes": steady,
                          "bf16_feasible_steady_state": steady <= limit,
                          "method_end_to_end_predicted_bytes": int(item["method_predicted_bytes"])})
    steady_only = []
    for row in rows:
        if row["method_config_id"] == "bf16" or not row["feasible_steady_state"]:
            continue
        b, h = row["batch_size"], row["historical_context"]
        steady = bf16_steady(b, h)
        if steady > limit:
            steady_only.append({"method_config_id": row["method_config_id"], "batch_size": b,
                                "historical_context": h,
                                "method_steady_state_predicted_bytes": row["steady_state_predicted_bytes"],
                                "bf16_steady_state_predicted_bytes": steady,
                                "method_measured": row["feasible_end_to_end"]})
    summary = {
        "limit_bytes": limit,
        "feasible_end_to_end_points": sum(r["feasible_end_to_end"] for r in rows),
        "feasible_steady_state_points": sum(r["feasible_steady_state"] for r in rows),
        "steady_feasible_but_end_to_end_infeasible": sum(
            r["feasible_steady_state"] and not r["feasible_end_to_end"] for r in rows),
        "end_to_end_feasible_but_steady_infeasible": sum(
            r["feasible_end_to_end"] and not r["feasible_steady_state"] for r in rows),
        "published_capacity_points": published,
        "published_capacity_points_where_bf16_fits_steady_state": sum(
            p["bf16_feasible_steady_state"] for p in published),
        "steady_state_capacity_amplification_points": steady_only,
        "steady_state_capacity_amplification_points_measured": sum(
            p["method_measured"] for p in steady_only),
        "by_configuration": {
            c: {"end_to_end_feasible": sum(r["feasible_end_to_end"] for r in rows if r["method_config_id"] == c),
                "steady_state_feasible": sum(r["feasible_steady_state"] for r in rows if r["method_config_id"] == c),
                "planned_points": sum(1 for r in rows if r["method_config_id"] == c)}
            for c in CONFIGS},
        "note": "all memory numbers are predictions of the preregistered feasibility formula",
    }
    return rows, summary


# ---------------------------------------------------------------- report

def fmt(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def build_report(summary: dict[str, Any], bandwidth: list[dict[str, Any]],
                 decomposition: list[dict[str, Any]], pooled: list[dict[str, Any]],
                 validation: list[dict[str, Any]]) -> str:
    lines = [
        "# Part A post-hoc analysis report",
        "",
        f"> {pc.POSTHOC_LABEL}.",
        "",
        "Inputs: Phase 16R host-wall closure `5605558b…`, Phase 16 outer root `d7458767…`, "
        "Phase 15 profiler root `641fc02d…` (all verified before use).",
        "",
        "## A.2 Effective bandwidth at B = 1 (profiled point 128K; % of 1,792 GB/s)",
        "",
        "| Config | total DRAM / T(128K) | cache v1 / ΔT | cache v2 / ΔT | marginal v2 | r_DRAM v1 | r_DRAM v2 | A v1 | A v2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in bandwidth:
        lines.append(
            f"| {pc.CONFIG_LABELS[row['method_config_id']]} | {fmt(row['bw_total_pct_peak'], 1)}% | "
            f"{fmt(row['bw_cache_v1_pct_peak'], 1)}% | {fmt(row['bw_cache_v2_pct_peak'], 1)}% | "
            f"{fmt(row['bw_cache_marginal_v2_pct_peak'], 1)}% | {fmt(row['r_dram_v1'], 2)} | "
            f"{fmt(row['r_dram_v2'], 2)} | {fmt(row['amplification_A_v1'], 2)} | "
            f"{fmt(row['amplification_A_v2'], 2)} |")
    lines += ["", "ΔT = T(B=1, 128K) − T(B=1, 4K), host wall. v1 = Phase 15 classification as "
              "published; v2 = post-hoc KVQuant key-path reclassification (see "
              "traffic_reclassified_kernels.csv).", "",
              "## A.1 Same-work ratio against modeled ceilings (357 same-work points)", "",
              "| Family | points | S measured | S_eq port | S_eq alg | S_roof port | S_roof alg | "
              "S_roof alg stored | S / S_roof alg | method BW / peak |",
              "|---|---:|---|---|---|---|---|---|---|---|"]
    a1 = summary["a1"]

    def span(pair: list[float], digits: int = 3) -> str:
        return f"{fmt(pair[0], digits)}–{fmt(pair[1], digits)}"

    for family, item in a1["by_family"].items():
        lines.append(
            f"| {FAMILY_LABELS[family]} | {item['points']} | {span(item['s_measured_range'])} | "
            f"{span(item['s_eq_port_v2_range'], 2)} | {span(item['s_eq_alg_range'], 2)} | "
            f"{span(item['s_roof_port_v2_range'], 2)} | {span(item['s_roof_alg_range'], 2)} | "
            f"{span(item['s_roof_alg_stored_range'], 2)} | {span(item['s_over_s_roof_alg_range'])} | "
            f"{span(item['method_bw_fraction_of_peak_v2_range'])} |")
    lines += ["",
              f"- BF16 modeled DRAM traffic / measured time at the same points: "
              f"{span(a1['bf16_bw_fraction_of_peak_range'])} of peak.",
              f"- Points with S > S_eq (port, v2): {a1['count_s_measured_gt_s_eq_port_v2']}; "
              f"points with S > S_roof (port, v2): {a1['count_s_measured_gt_s_roof_port_v2']}.",
              f"- Compressed feasible points: {a1['compressed_feasible_points']}; with a measured S: "
              f"{a1['points_with_measured_s']}."]
    lines += ["", "## A.5 Decomposition T = c0 + D_cache / BW_eff (descriptive, per B)", "",
              "| Config | B | n | c0 (ms) | BW_eff (GB/s) | R² |", "|---|---:|---:|---:|---:|---:|"]
    for row in decomposition:
        if row.get("status") == "fitted":
            bw = row["bw_eff_cache_bytes_per_s"]
            lines.append(f"| {pc.CONFIG_LABELS[row['method_config_id']]} | {row['batch_size']} | "
                         f"{row['points']} | {fmt(row['c0_ms'], 2)} | "
                         f"{fmt(bw / GB if bw else None, 1)} | {fmt(row['r_squared'], 4)} |")
    lines += ["", "## A.3 Within-family byte elasticity (secondary evidence)", ""]
    for row in pooled:
        if row.get("points"):
            lines.append(
                f"- {row['family']}: pooled slope {fmt(row['pooled_observed_slope'])} "
                f"[{fmt(row['pooled_ci_low'])}, {fmt(row['pooled_ci_high'])}] over {row['points']} "
                f"(B, L) points; bandwidth-bound expectation {fmt(row['pooled_expected_slope_bandwidth_bound'])}. "
                f"CI: {row['ci_scope']}.")
        else:
            lines.append(f"- {row['family']}: no (B, L) point passes the cache-share threshold.")
    lines += ["", "## A.4 Steady-state capacity (predicted)", ""]
    for key, value in summary["a4"].items():
        lines.append(f"- {key}: {value}")
    lines += ["", "## Traffic-model validation at existing B = 1 profiles", "",
              "| Config | L | cache v2 error | total v2 error |", "|---|---:|---:|---:|"]
    for row in validation:
        lines.append(f"| {pc.CONFIG_LABELS[row['method_config_id']]} | {row['historical_context']} | "
                     f"{fmt(100 * row['cache_v2_relative_error'], 2)}% | "
                     f"{fmt(100 * row['total_v2_relative_error'], 2)}% |")
    lines.append("")
    return "\n".join(lines)


def figure(rows: list[dict[str, Any]], x_field: str, x_label: str) -> str:
    panels = []
    xs, ys = [], []
    for family in FAMILIES:
        points = []
        for row in rows:
            if row["family"] == family and row.get(x_field) and row.get("s_measured"):
                points.append((row[x_field], row["s_measured"], row["method_config_id"],
                               row["batch_size"], False))
                xs.append(row[x_field])
                ys.append(row["s_measured"])
        panels.append({"title": FAMILY_LABELS[family], "points": points})
    if min(xs) >= 1.0:
        x_high = next(v for v in (1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0, 100.0) if v >= max(xs) * 1.02)
        x_range = (1.0, x_high)
        x_ticks = [v for v in (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0) if v <= x_high]
    else:
        x_range = (10 ** math.floor(math.log10(min(xs))), 10 ** math.ceil(math.log10(max(xs))))
        x_ticks = None
    y_low = max(v for v in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1) if v <= min(ys) * 0.95)
    y_range = (y_low, max(2.0, 10 ** math.ceil(math.log10(max(ys)))))
    legend = [(c, pc.CONFIG_LABELS[c]) for c in CONFIGS if c != "bf16"]
    return pc.render_loglog_panels(
        panels, x_label=x_label, y_label="measured same-work ratio S = T_BF16 / T_method",
        x_range=x_range, y_range=y_range, legend=legend, diagonal=True, unity_y=True,
        x_ticks=x_ticks)


# ---------------------------------------------------------------- main

A1_COLUMNS = ["method_config_id", "family", "batch_size", "historical_context", "grid_source",
              "t_method_ms", "t_bf16_ms", "s_measured", "s_eq_port_v2", "s_eq_port_v1",
              "s_eq_alg", "s_eq_alg_stored", "s_roof_port_v2", "s_roof_port_v1", "s_roof_alg",
              "s_roof_alg_stored", "s_over_s_roof_alg", "bf16_bw_fraction_of_peak",
              "method_bw_fraction_of_peak_v2", "modeled_cache_share_v2",
              "modeled_traffic_bf16_bytes", "modeled_traffic_method_v2_bytes",
              "modeled_traffic_method_v1_bytes", "modeled_traffic_algorithm_bytes"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--parent", type=Path, default=pc.ARTIFACT_PARENT,
                        help="directory that receives the sealed artifact (default: paper/posthoc/artifacts)")
    args = parser.parse_args(argv)
    log = pc.WarningLog()
    git_sha = pc.git_head()
    run_id = pc.new_run_id("a", git_sha)
    data = load_inputs(log)
    peak_record = device_peak(data["ph15"], log)
    peak = PEAK_BW_DATASHEET
    traffic, reclassified = run_traffic(data)
    model = TrafficModel(data, traffic)
    exclusions: list[dict[str, Any]] = []
    validation = traffic_validation(model)
    a1 = roofline_rows(data, model, peak, exclusions)
    bandwidth = bandwidth_rows(data, model, peak)
    decomposition = decomposition_rows(data, model)
    elastic_rows, pooled = elasticity(data, model, exclusions)
    capacity_rows, capacity_summary = capacity(data)

    measured = [r for r in a1 if r["s_measured"] is not None]
    with_roof = [r for r in measured if r["s_roof_alg"] is not None]
    a1_summary = {
        "compressed_feasible_points": len(a1),
        "points_with_measured_s": len(measured),
        "points_with_s_roof": len(with_roof),
        "count_s_measured_gt_s_eq_port_v2": sum(r["s_measured"] > r["s_eq_port_v2"] for r in measured),
        "count_s_measured_gt_s_roof_port_v2": sum(r["s_measured"] > r["s_roof_port_v2"] for r in with_roof),
        "by_family": {},
    }
    for family in FAMILIES:
        subset = [r for r in with_roof if r["family"] == family]
        if not subset:
            continue
        a1_summary["by_family"][family] = {
            "s_measured_range": [min(r["s_measured"] for r in subset), max(r["s_measured"] for r in subset)],
            "s_eq_port_v2_range": [min(r["s_eq_port_v2"] for r in subset), max(r["s_eq_port_v2"] for r in subset)],
            "s_eq_alg_range": [min(r["s_eq_alg"] for r in subset), max(r["s_eq_alg"] for r in subset)],
            "s_roof_port_v2_range": [min(r["s_roof_port_v2"] for r in subset), max(r["s_roof_port_v2"] for r in subset)],
            "s_roof_alg_range": [min(r["s_roof_alg"] for r in subset), max(r["s_roof_alg"] for r in subset)],
            "s_roof_alg_stored_range": [min(r["s_roof_alg_stored"] for r in subset),
                                        max(r["s_roof_alg_stored"] for r in subset)],
            "s_eq_alg_stored_range": [min(r["s_eq_alg_stored"] for r in subset),
                                      max(r["s_eq_alg_stored"] for r in subset)],
            "s_over_s_roof_alg_range": [min(r["s_over_s_roof_alg"] for r in subset),
                                        max(r["s_over_s_roof_alg"] for r in subset)],
            "method_bw_fraction_of_peak_v2_range": [
                min(r["method_bw_fraction_of_peak_v2"] for r in subset),
                max(r["method_bw_fraction_of_peak_v2"] for r in subset)],
            "points": len(subset),
        }
    bf16_util = [r["bf16_bw_fraction_of_peak"] for r in with_roof]
    a1_summary["bf16_bw_fraction_of_peak_range"] = [min(bf16_util), max(bf16_util)]
    a4_summary = {k: v for k, v in capacity_summary.items()
                  if k not in ("published_capacity_points", "steady_state_capacity_amplification_points",
                               "by_configuration")}

    summary = {
        "label": pc.POSTHOC_LABEL,
        "peak": peak_record,
        "traffic_model": {
            "alpha": model.alpha,
            "paper_amplification_A": {v: {c: model.paper_amplification(v, c) for c in CONFIGS}
                                      for v in ("v1", "v2")},
            "noncache_fit": model.noncache,
            "b_extension": "N_f(B, L) = n0 + n1 * B * (L + 1): L-dependent non-cache traffic assumed per sequence",
            "algorithm_version": "D_alg = N_BF16(B, L) + alpha_BF16 * C_m(B, L) (A = 1, no method-specific non-cache traffic)",
            "algorithm_stored_variant": "as D_alg but C_m excludes workspace and padding bytes",
        },
        "a1": a1_summary,
        "a4": a4_summary,
        "elasticity_threshold_min_cache_share": ELASTICITY_MIN_CACHE_SHARE,
        "elasticity_pooled": pooled,
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "unit": "process session (replicate segment)"},
    }

    stage = pc.new_stage(run_id, args.parent)
    try:
        pc.write_csv(stage / "a1_roofline_ratios.csv", a1, A1_COLUMNS)
        pc.write_csv(stage / "a2_bandwidth_b1.csv", bandwidth, list(bandwidth[0]))
        decomposition_columns = ["method_config_id", "family", "batch_size", "points", "status",
                                 "c0_ms", "c0_excess_over_bf16_ms", "slope_ms_per_gb",
                                 "bw_eff_cache_bytes_per_s", "r_squared", "max_abs_relative_residual",
                                 "cache_traffic_min_bytes", "cache_traffic_max_bytes"]
        pc.write_csv(stage / "a5_decomposition_fits.csv", decomposition, decomposition_columns)
        pc.write_csv(stage / "a3_elasticity_points.csv", elastic_rows, list(elastic_rows[0]))
        pc.write_new(stage / "a3_elasticity_pooled.json", pc.json_text(pooled))
        pc.write_csv(stage / "a4_capacity_points.csv", capacity_rows, list(capacity_rows[0]))
        pc.write_new(stage / "a4_capacity_summary.json", pc.json_text(capacity_summary))
        traffic_rows = [{"method_config_id": k[0], "batch_size": k[1], "historical_context": k[2], **v}
                        for k, v in sorted(traffic.items())]
        pc.write_csv(stage / "traffic_profiles.csv", traffic_rows, list(traffic_rows[0]))
        pc.write_csv(stage / "traffic_reclassified_kernels.csv", reclassified, list(reclassified[0]))
        pc.write_csv(stage / "traffic_model_validation.csv", validation, list(validation[0]))
        pc.write_new(stage / "exclusions.json", pc.json_text({
            "schema_version": "kvbench-posthoc-exclusions-1.0.0", "exclusions": exclusions,
            "counts": dict(Counter(f"{e['analysis']}:{e['reason_code']}" for e in exclusions))}))
        pc.write_new(stage / "summary.json", pc.json_text(summary))
        pc.write_new(stage / "partA_report.md", build_report(summary | {"a1": a1_summary, "a4": a4_summary},
                                                             bandwidth, decomposition, pooled, validation))
        for name, field, label in (
            ("fig_s_vs_s_roof_alg.svg", "s_roof_alg",
             "modeled roofline ceiling S_roof (A = 1; BF16 measured time x peak BW / ideal traffic)"),
            ("fig_s_vs_s_eq_port.svg", "s_eq_port_v2",
             "modeled equal-efficiency ratio S_eq = D_BF16 / D_method (as-ported traffic)"),
        ):
            svg = log.guard(f"figure_{field}", lambda f=field, l=label: figure(a1, f, l))
            if svg:
                pc.write_new(stage / "figures" / name, svg)
        for source in (Path(__file__), Path(pc.__file__)):
            pc.write_new(stage / "code" / source.name, source.read_bytes())
        manifest = {
            "schema_version": "kvbench-posthoc-review-manifest-1.0.0",
            "status": "PASS",
            "posthoc": True,
            "label": pc.POSTHOC_LABEL,
            "analysis": "review_round_part_a",
            "created_at_utc": pc.utc_now(),
            "git_head": git_sha,
            "code_note": "analysis code is not committed at git_head; exact copies are in code/",
            "code_sha256": {p.name: pc.sha256_file(p) for p in (Path(__file__), Path(pc.__file__))},
            "environment": pc.environment_record(),
            "inputs": {"phase16r_host_wall_closure": data["roots"]["wall"],
                       "phase16_outer": data["roots"]["outer"],
                       "phase15_profiler": data["ph15"].record()},
            "parameters": {"peak_bw_bytes_per_s": peak, "peak_bw_source": PEAK_BW_SOURCE,
                           "elasticity_min_cache_share": ELASTICITY_MIN_CACHE_SHARE,
                           "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED,
                           "kvquant_key_reclassification_patterns": list(KVQ_KEY_PATTERNS)},
            "timing_source": "Phase 16R host-wall process medians (frozen); no profiler duration used as timing",
            "profiler_settings_note": "Phase 15 ncu used defaults: --cache-control all, --clock-control base",
            "performance_claim_eligible": False,
            "warnings": log.entries,
        }
        final, root = pc.seal(stage, run_id, manifest, "posthoc_review_part_a", args.parent)
    except Exception:
        print(f"stage left for inspection: {stage}", file=sys.stderr)
        raise
    print(f"artifact: {final}")
    print(f"root_sha256: {root}")
    print(f"warnings: {len(log.entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
