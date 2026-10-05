#!/usr/bin/env python3
"""POST-HOC exclusion record (AGENTS.md #12) for the review-round revision.

Seals one machine-readable record: the three KVQuant configurations are excluded
from the paper's quality conclusions because their Fast-perplexity degradation
(Delta NLL near 3.4 at 4, 3 and 2 bits) does not shrink with bit width, which is
inconsistent with quantization error and points to a suspected defect in the
port.  The preregistered FAIL status is unchanged; nothing in the frozen quality
evidence is edited.  The values are read from the committed Fast-perplexity
evidence (docs/evidence/q1a/fast-ppl.json) and checked before sealing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import posthoc_common as pc

REPO = pc.REPO
FAST_PPL = REPO / "docs/evidence/q1a/fast-ppl.json"
FAST_PPL_RECEIPT = REPO / "docs/evidence/q1a/r2-publication.json"
JOINT_RECEIPT = REPO / "docs/evidence/q3-q4/r2-publication.json"
KVQUANT = ("kvq4", "kvq3", "kvq2")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--parent", type=Path, default=pc.ARTIFACT_PARENT)
    args = parser.parse_args(argv)
    fast = json.loads(FAST_PPL.read_text())
    decisions = {item["configuration"]: item for item in fast["decisions"]}
    observed = {}
    for configuration in KVQUANT:
        item = decisions[configuration]
        if item["fast_stage_status"] != "fail" or not 3.3 < item["delta_nll"] < 3.5:
            raise pc.PosthocError(f"{configuration} Fast-perplexity evidence differs from the record basis")
        observed[configuration] = {
            "fast_stage_status": item["fast_stage_status"],
            "delta_nll": item["delta_nll"],
            "ci95_delta_nll": item["ci95_delta_nll"],
            "relative_ppl_change": item["relative_ppl_change"],
        }
    q1a_root = json.loads(FAST_PPL_RECEIPT.read_text())["root_sha256"]
    joint_root = json.loads(JOINT_RECEIPT.read_text())["publish"]["root_sha256"]
    record = {
        "schema_version": "kvbench-posthoc-exclusion-record-1.0.0",
        "posthoc": True,
        "label": pc.POSTHOC_LABEL,
        "decided_on": "2026-10-05",
        "decided_by": "author, review-round revision",
        "exclusions": [{
            "scope": "paper_quality_conclusions",
            "configurations": list(KVQUANT),
            "stage": "fast_perplexity",
            "preregistered_status": "FAIL",
            "preregistered_status_changed": False,
            "frozen_tables_changed": False,
            "reason_code": "quality_result_invalid_suspected_port_defect",
            "reason": ("Delta NLL is about 3.4 at 4, 3 and 2 bits; an error that does not shrink with bit width is "
                       "inconsistent with quantization error and points to a defect in the integrated port "
                       "(GQA/RoPE compatibility, outlier handling, calibration, or the graph-safe kernels); the "
                       "cause was not established"),
            "observed": observed,
            "effect_on_paper": ("KVQuant is removed from the headline and joint-quality statements; its rows stay in "
                                "the performance tables labeled as ported, and its preregistered FAIL stays in the "
                                "joint-outcome table with a post-hoc annotation; quality details move to the appendix"),
            "evidence": {
                "fast_ppl_summary": {"path": "docs/evidence/q1a/fast-ppl.json",
                                     "sha256": pc.sha256_file(FAST_PPL)},
                "fast_ppl_campaign_root_sha256": q1a_root,
                "joint_results_root_sha256": joint_root,
            },
        }],
    }
    git_sha = pc.git_head()
    run_id = pc.new_run_id("c-exclusions", git_sha)
    stage = pc.new_stage(run_id, args.parent)
    pc.write_new(stage / "exclusion_record.json", pc.json_text(record))
    pc.write_new(stage / "code" / Path(__file__).name, Path(__file__).read_bytes())
    manifest = {
        "schema_version": "kvbench-posthoc-review-manifest-1.0.0", "status": "PASS", "posthoc": True,
        "label": pc.POSTHOC_LABEL, "analysis": "review_round_exclusion_record", "created_at_utc": pc.utc_now(),
        "git_head": git_sha, "code_sha256": {Path(__file__).name: pc.sha256_file(Path(__file__))},
        "inputs": {"fast_ppl_summary_sha256": pc.sha256_file(FAST_PPL), "fast_ppl_root_sha256": q1a_root,
                   "joint_results_root_sha256": joint_root},
        "performance_claim_eligible": False,
    }
    final, root = pc.seal(stage, run_id, manifest, "posthoc_review_exclusion_record", args.parent)
    print(f"artifact: {final}\nroot_sha256: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
