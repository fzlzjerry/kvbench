#!/usr/bin/env python3
"""Task 4 B_max at L = 128K (host-side, pure Python, no torch, no GPU).

    python3 addendum/feasibility.py            # table + B_max
    python3 addendum/feasibility.py --json     # machine-readable result

Steady-state memory = weights + allocated cache + endpoint workspace + graph
reserve + fixed decode inputs: exactly the five fields that
paper/posthoc/partA_analysis.py capacity() adds from each Phase 16
feasibility record.  Every component is produced by the frozen Phase 13F
formula itself (scripts/phase13_pilot.py feasibility_record ->
cache_allocated_bytes / _kivi_cache_bytes, endpoint_workspace_bytes,
_reference_graph_reserve_bytes, prefix_construction_memory), imported from
the frozen repository (the import needs no torch).  The only intervention is
the batch domain: endpoint_workspace_bytes and prefix_construction_memory
guard on phase13.BATCH_SIZES, so it is widened to 1..32 for the duration of
the call, the same mechanism phase16_full_scan._configure_reused_phase13 uses
to admit B in {1,2,4,8,16}.  It is restored afterwards.

Cross-checks (any failure raises SystemExit):
  1. grid points B in {1,2,4,8,16}: every field of the computed record that
     also appears in the frozen per-point records of
     docs/plans/phase16-full-scan-execution-orders.json (all five replicate
     segments) is identical;
  2. BF16 at every B: the steady state equals partA_analysis.bf16_steady's
     closed form (re-stated verbatim below);
  3. the limit equals floor(101,970,345,984 x 0.88) = 89,733,904,465 bytes;
  4. steady state increases with B, so B_max is a threshold.

Memory numbers are predictions of the preregistered formula, not
measurements.  The KIVI grouped-residual variant (overrides.py) allocates
nothing, so the k4v4 row applies to both adapters.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterator, Sequence

FROZEN_REPO = Path("/home/rockrock/cmu_paper")
ORDER_PATH = FROZEN_REPO / "docs/plans/phase16-full-scan-execution-orders.json"
LIMIT_BYTES = 89_733_904_465
CONTEXT_LABEL = 131072
CONFIGURATIONS = ("bf16", "k4v4")
BATCHES = tuple(range(1, 33))
GRID_BATCHES = (1, 2, 4, 8, 16)
STEADY_COMPONENTS = (
    "model_weight_bytes",
    "cache_allocated_bytes",
    "endpoint_workspace_bytes",
    "graph_pool_or_capture_reserve_bytes",
    "fixed_decode_input_bytes",
)
# paper/posthoc/partA_analysis.py constants (copied from scripts/phase13_pilot.py there).
PARTA_MODEL_WEIGHT_BYTES = 16_060_556_288
PARTA_REFERENCE_CAPACITY = 4097


class FeasibilityError(SystemExit):
    pass


def _frozen_modules(repo: Path = FROZEN_REPO) -> tuple[Any, Any]:
    sys.dont_write_bytecode = True  # never write into the root-owned frozen tree
    for path in (str(repo / "src"), str(repo)):
        if path not in sys.path:
            sys.path.insert(0, path)
    import scripts.phase13_pilot as phase13
    import scripts.phase16_full_scan as phase16

    if phase13.REPOSITORY_ROOT.resolve() != repo.resolve():
        raise FeasibilityError(f"imported phase13_pilot is not the frozen copy: {phase13.__file__}")
    return phase13, phase16


@contextmanager
def _batch_domain(phase13: Any, batches: Sequence[int]) -> Iterator[None]:
    original = phase13.BATCH_SIZES
    phase13.BATCH_SIZES = tuple(sorted(set(original) | set(batches)))
    try:
        yield
    finally:
        phase13.BATCH_SIZES = original


def _parta_bf16_steady(batch: int, historical: int, reference: int, reference_endpoint: int) -> int:
    """paper/posthoc/partA_analysis.py capacity().bf16_steady, verbatim arithmetic."""
    capacity_tokens = historical + 1
    graph = math.ceil((reference - reference_endpoint) * batch * capacity_tokens
                      / PARTA_REFERENCE_CAPACITY)
    endpoint = 32 * batch * (32 + 8) * 64 * 2
    fixed = batch * 8 + 8 + 2 * 128 * 2
    cache = 2 * 32 * batch * 8 * capacity_tokens * 128 * 2 + 163_840
    return PARTA_MODEL_WEIGHT_BYTES + cache + endpoint + graph + fixed


def _frozen_grid_records(configurations: Sequence[str], historical: int
                         ) -> tuple[dict[tuple[str, int], list[dict]], int]:
    order = json.loads(ORDER_PATH.read_text())
    found: dict[tuple[str, int], list[dict]] = {}
    for segment in order["segments"]:
        for record in segment["records"]:
            if record["method_config_id"] in configurations and record["historical_context"] == historical:
                found.setdefault((record["method_config_id"], int(record["batch_size"])), []).append(record)
    return found, len(order["segments"])


def compute(configurations: Sequence[str] = CONFIGURATIONS, batches: Sequence[int] = BATCHES,
            context_label: int = CONTEXT_LABEL) -> dict[str, Any]:
    """Return the steady-state table, B_max per configuration and the cross-checks."""
    phase13, phase16 = _frozen_modules()
    historical = phase16.actual_historical_context(context_label)
    if tuple(phase16.BATCH_SIZES) != GRID_BATCHES:
        raise FeasibilityError("Full Scan batch grid differs")
    limit = math.floor(phase13.GPU_TOTAL_MEMORY_BYTES * phase13.MAX_MEMORY_FRACTION)
    if limit != LIMIT_BYTES:
        raise FeasibilityError(f"memory limit differs: {limit}")
    rows: list[dict[str, Any]] = []
    with _batch_domain(phase13, batches):
        for configuration in configurations:
            for batch in batches:
                record = phase13.feasibility_record({
                    "method_config_id": configuration,
                    "batch_size": batch,
                    "historical_context": historical,
                })
                components = {key: int(record[key]) for key in STEADY_COMPONENTS}
                steady = sum(components.values())
                rows.append({
                    "method_config_id": configuration,
                    "batch_size": batch,
                    "context_label": context_label,
                    "historical_context": historical,
                    "capacity": int(record["capacity"]),
                    **components,
                    "steady_state_bytes": steady,
                    "limit_bytes": int(record["limit_bytes"]),
                    "headroom_bytes": int(record["limit_bytes"]) - steady,
                    "fits_steady_state": steady <= int(record["limit_bytes"]),
                    "end_to_end_predicted_bytes": int(record["predicted_required_bytes"]),
                    "end_to_end_status": record["status"],
                    "full_scan_admitted_batch": batch in GRID_BATCHES,
                    "graph_reserve_reference_bytes": int(record["graph_reserve_reference_bytes"]),
                    "graph_reserve_reference_endpoint_workspace_bytes": int(
                        record["graph_reserve_reference_endpoint_workspace_bytes"]),
                    "_record": record,
                })

    # Cross-check 1: frozen grid records.
    frozen, segments = _frozen_grid_records(configurations, historical)
    grid_checked = 0
    for row in rows:
        if row["batch_size"] not in GRID_BATCHES:
            continue
        key = (row["method_config_id"], row["batch_size"])
        references = frozen.get(key, [])
        if len(references) != segments:  # one record per replicate segment
            raise FeasibilityError(f"expected {segments} frozen records for {key}, found {len(references)}")
        computed = row["_record"]
        for reference in references:
            shared = sorted(set(computed) & set(reference))
            if not set(STEADY_COMPONENTS) <= set(shared):
                raise FeasibilityError(f"frozen record {key} lacks steady-state components")
            differing = [name for name in shared if computed[name] != reference[name]]
            if differing:
                raise FeasibilityError(f"{key} differs from the frozen record in {differing}")
        grid_checked += 1
    expected_grid = len(configurations) * len([b for b in batches if b in GRID_BATCHES])
    if grid_checked != expected_grid:
        raise FeasibilityError("grid cross-check coverage differs")

    # Cross-check 2: partA closed form for BF16.
    parta_checked = 0
    for row in rows:
        if row["method_config_id"] != "bf16":
            continue
        closed = _parta_bf16_steady(row["batch_size"], historical, row["graph_reserve_reference_bytes"],
                                    row["graph_reserve_reference_endpoint_workspace_bytes"])
        if closed != row["steady_state_bytes"]:
            raise FeasibilityError(f"BF16 B={row['batch_size']} differs from partA bf16_steady")
        parta_checked += 1

    # Cross-check 4 and B_max.
    b_max: dict[str, int | None] = {}
    for configuration in configurations:
        series = [r for r in rows if r["method_config_id"] == configuration]
        series.sort(key=lambda r: r["batch_size"])
        values = [r["steady_state_bytes"] for r in series]
        if any(later <= earlier for earlier, later in zip(values, values[1:])):
            raise FeasibilityError(f"{configuration} steady state is not increasing in B")
        fitting = [r["batch_size"] for r in series if r["fits_steady_state"]]
        if fitting and fitting != list(range(series[0]["batch_size"], max(fitting) + 1)):
            raise FeasibilityError(f"{configuration} feasible batches are not a prefix")
        b_max[configuration] = max(fitting) if fitting else None
        if b_max[configuration] == batches[-1]:
            raise FeasibilityError(f"{configuration} fits at the largest scanned B; widen the scan")

    for row in rows:
        del row["_record"]
    return {
        "schema_version": "kvbench-addendum-20261005-task4-bmax-1.0.0",
        "rule": "largest B with weights + allocated cache + endpoint workspace + graph reserve "
                "+ fixed decode inputs <= limit (amendment Section 6; partA capacity() steady state)",
        "formula_source": "scripts/phase13_pilot.py feasibility_record (frozen, imported)",
        "context_label": context_label,
        "historical_context": historical,
        "limit_bytes": LIMIT_BYTES,
        "b_max": b_max,
        "cross_checks": {
            "frozen_grid_records_identical": grid_checked,
            "parta_bf16_closed_form_identical": parta_checked,
            "limit_bytes": True,
            "monotone_in_batch": True,
        },
        "memory_numbers": "predicted by the preregistered feasibility formula; not measured",
        "rows": rows,
    }


def b_max(configurations: Sequence[str] = CONFIGURATIONS) -> dict[str, int | None]:
    return compute(configurations)["b_max"]


def _table(result: dict[str, Any]) -> str:
    gb = 1e9
    lines = [f"L label {result['context_label']} (historical {result['historical_context']}), "
             f"limit {result['limit_bytes']:,} B; GB = 1e9 B; steady = W + cache + endpoint + graph + fixed",
             f"{'config':6s} {'B':>3s} {'cache GB':>10s} {'graph GB':>9s} {'endpoint':>9s} {'fixed':>5s} "
             f"{'steady B':>16s} {'headroom GB':>12s} fits  grid"]
    for row in result["rows"]:
        lines.append(
            f"{row['method_config_id']:6s} {row['batch_size']:3d} {row['cache_allocated_bytes'] / gb:10.3f} "
            f"{row['graph_pool_or_capture_reserve_bytes'] / gb:9.3f} {row['endpoint_workspace_bytes']:9d} "
            f"{row['fixed_decode_input_bytes']:5d} {row['steady_state_bytes']:16,d} "
            f"{row['headroom_bytes'] / gb:12.3f} {'yes' if row['fits_steady_state'] else 'no ':4s} "
            f"{'*' if row['full_scan_admitted_batch'] else ''}")
    lines.append("weights W = 16,060,556,288 B in every row; * = Full Scan grid batch (cross-checked)")
    lines.append("B_max: " + ", ".join(f"{k} = {v}" for k, v in result["b_max"].items()))
    lines.append("cross-checks: " + json.dumps(result["cross_checks"], sort_keys=True))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--max-batch", type=int, default=BATCHES[-1])
    args = parser.parse_args()
    result = compute(batches=tuple(range(1, args.max_batch + 1)))
    print(json.dumps(result, indent=2, sort_keys=True) if args.json else _table(result))


if __name__ == "__main__":
    main()
