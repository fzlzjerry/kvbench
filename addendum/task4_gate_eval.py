#!/usr/bin/env python3
"""Task 4 synthetic-cache validity gate (amendment Section 6).

Pass for a configuration iff both the real-cache and the synthetic-cache point
at (B=1, 32K) have 3 completed processes with all worker checks true, and
|T_syn - T_real| / T_real <= max(CV_real, CV_syn, 1 %).  Task 4 continues to
the B_max points only if BF16 and KIVI-k4v4 both pass.  Exit status 0 = pass.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402


def main() -> int:
    root = c.RESULTS / "task4-gate"
    points = {row["block"]: row for row in c.read_jsonl(root / "points.jsonl")}
    result: dict = {"addendum_id": c.ADDENDUM_ID, "rule": "|T_syn - T_real| / T_real <= max(CV_real, CV_syn, 0.01)",
                    "configurations": {}, "evaluated_at_utc": c.utc_now()}
    for configuration in ("bf16", "k4v4"):
        real, synthetic = points.get(f"{configuration}-real"), points.get(f"{configuration}-synthetic")
        entry: dict = {"real": real and {k: real[k] for k in ("point_median_ms", "cv", "n_processes", "disposition")},
                       "synthetic": synthetic and {k: synthetic[k] for k in ("point_median_ms", "cv", "n_processes", "disposition")}}
        complete = (real is not None and synthetic is not None
                    and real["n_processes"] == c.PROCESSES_PER_POINT
                    and synthetic["n_processes"] == c.PROCESSES_PER_POINT
                    and not real["failed_checks"] and not synthetic["failed_checks"])
        if complete:
            difference = abs(synthetic["point_median_ms"] - real["point_median_ms"]) / real["point_median_ms"]
            threshold = max(real["cv"], synthetic["cv"], 0.01)
            entry.update({"relative_difference": difference, "threshold": threshold,
                          "pass": difference <= threshold})
        else:
            entry.update({"pass": False, "reason": "incomplete_or_check_failed"})
        result["configurations"][configuration] = entry
    result["pass"] = all(e["pass"] for e in result["configurations"].values())
    path = root / f"gate-{c.utc_now().replace(':', '')}.json"
    c.write_new(path, c.json_text(result))
    print(c.json_text(result))
    if not result["pass"]:
        c.append_failure("task4", "synthetic-cache validity gate failed; Task 4 stops before the B_max points "
                                  "(reason `synthetic_cache_validity_gate_failed`):\n\n```\n"
                                  + json.dumps(result["configurations"], indent=2) + "\n```")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
