#!/usr/bin/env python3
"""Compare two greedy check runs offline (CPU; needs torch).

    python addendum/compare_checks.py TASK REFERENCE_CHECK CANDIDATE_CHECK

Reports the maximum absolute logit difference at the first decode step (same
input in both runs), the maximum over the steps before the first token
divergence (inputs still identical), the number of agreeing positions among
the generated tokens, and the first divergence.  Appends one JSONL row to
<task>/checks/comparisons.jsonl.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402


def main() -> None:
    import torch

    task, reference_name, candidate_name = sys.argv[1:4]
    root = c.RESULTS / task / "checks"
    runs = {}
    for name in (reference_name, candidate_name):
        result = json.loads((root / name / "worker_result.json").read_text())
        if result.get("status") != "completed":
            raise SystemExit(f"{name} did not complete")
        if c.sha256_file(root / name / "greedy.pt") != result["greedy_pt_sha256"]:
            raise SystemExit(f"{name} greedy.pt checksum differs")
        runs[name] = (result, torch.load(root / name / "greedy.pt"))
    (ref_result, ref), (cand_result, cand) = runs[reference_name], runs[candidate_name]
    ref_tokens, cand_tokens = ref["tokens"], cand["tokens"]
    if len(ref_tokens) != len(cand_tokens):
        raise SystemExit("step counts differ")
    agree = sum(int(a == b) for a, b in zip(ref_tokens, cand_tokens))
    first = next((i for i, (a, b) in enumerate(zip(ref_tokens, cand_tokens)) if a != b), None)
    # Step i's input is identical in both runs while all tokens before i agree.
    same_input_steps = len(ref_tokens) if first is None else first + 1
    diff = (ref["logits"][:same_input_steps] - cand["logits"][:same_input_steps]).abs()
    row = {
        "task": task, "reference": reference_name, "candidate": candidate_name,
        "reference_variant": ref_result["variant"], "candidate_variant": cand_result["variant"],
        "configuration": cand_result["configuration"], "probe_sha256": cand_result["probe_sha256"],
        "prefix_tokens": cand_result["prefix_tokens"], "steps": len(ref_tokens),
        "max_abs_logit_diff_step0": float(diff[0].max()),
        "max_abs_logit_diff_same_input_steps": float(diff.max()),
        "same_input_steps": same_input_steps,
        "agreeing_positions": agree, "identical_tokens": agree == len(ref_tokens),
        "first_divergence": first,
        "reference_finite": ref_result["evidence"]["finite"],
        "candidate_finite": cand_result["evidence"]["finite"],
        "reference_result_sha256": c.sha256_file(root / reference_name / "worker_result.json"),
        "candidate_result_sha256": c.sha256_file(root / candidate_name / "worker_result.json"),
        "compared_at_utc": c.utc_now(),
    }
    c.append_jsonl(root / "comparisons.jsonl", row)
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
