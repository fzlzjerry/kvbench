from __future__ import annotations

import math
import tempfile
from pathlib import Path
import unittest

from scripts import q1b_full_ppl as q1b


class Q1BFullPPLTests(unittest.TestCase):
    def test_authority_and_exact_configuration_scope(self) -> None:
        self.assertEqual(q1b.verify_authority()["status"], "PASS")
        self.assertEqual(q1b.CONFIGS, ("bf16", "k4v4"))
        rows = {row["configuration"]: row for row in q1b.eligibility()["rows"]}
        self.assertTrue(rows["bf16"]["selected_for_q1b"])
        self.assertTrue(rows["k4v4"]["selected_for_q1b"])
        self.assertEqual(rows["tq_4bit_nc"]["q1b_status"], "not_run_due_to_fast_stage_status")

    def test_full_plan_counts_alignment_and_model_bound(self) -> None:
        plan = q1b.full_plan()
        self.assertEqual(plan["anchors_per_configuration"], 1024)
        self.assertEqual(plan["scored_tokens_per_configuration"], 262144)
        self.assertEqual(plan["total_anchor_units"], 2048)
        self.assertEqual(plan["total_scored_tokens"], 524288)
        self.assertTrue(all(row["target_start"] == row["burn_in_index"] + 1 for row in plan["anchors"]))
        self.assertTrue(all(row["target_end_exclusive"] - row["target_start"] == 256 for row in plan["anchors"]))
        self.assertTrue(all(row["prefix_length"] + 1 + 256 <= 131072 for row in plan["anchors"]))

    def test_fast_observations_are_not_reused_as_full(self) -> None:
        overlap = q1b.full_plan()["fast_full_overlap"]
        self.assertEqual(overlap["exact_full_observations"], 0)
        self.assertEqual(overlap["same_prefix_partial_horizon"], 2)

    def test_full_gate_uses_both_datasets_and_hard_length_guard(self) -> None:
        margins = q1b.q1a.quality_margins()
        margin = math.log1p(0.01)
        passing = [
            {"dataset": "wikitext2_test", "ci_lower_delta_nll": 0.0, "ci_upper_delta_nll": margin - 1e-6},
            {"dataset": "c4_validation", "ci_lower_delta_nll": 0.0, "ci_upper_delta_nll": margin - 1e-6},
        ]
        self.assertEqual(q1b.classify_full(passing, [{"relative_ppl_change": 0.0}], margins)[0], "pass")
        mixed = [passing[0], {"dataset": "c4_validation", "ci_lower_delta_nll": 0.0, "ci_upper_delta_nll": margin + 1e-6}]
        self.assertEqual(q1b.classify_full(mixed, [{"relative_ppl_change": 0.0}], margins)[0], "inconclusive")
        self.assertEqual(q1b.classify_full(passing, [{"dataset": "c4_validation", "prefix_length": 4096, "relative_ppl_change": 0.050001}], margins)[0], "fail")

    def test_unit_is_append_only_and_resumable(self) -> None:
        anchor = q1b.full_plan()["anchors"][0]
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)
            result = {"status": "PASS", "scored_token_count": 256}
            q1b._accept_anchor_result(campaign, "bf16", anchor, result)
            self.assertTrue(q1b._unit_complete(campaign, "bf16", anchor))
            with self.assertRaises(q1b.Q1BError):
                q1b._finalize_unit(campaign, "bf16", anchor, result)

    def test_worker_mount_is_absolute_and_q1b_scoped(self) -> None:
        with tempfile.TemporaryDirectory(dir=q1b.ROOT) as directory:
            command = q1b._worker_command(Path(directory), "bf16", "0" * 40)
            mounts = [value for value in command if value.startswith("type=bind,") and f"src={directory}" in value]
            self.assertEqual(len(mounts), 1)
            self.assertIn("dst=/home/rockrock/cmu_paper/artifacts/q1b/", mounts[0])
            self.assertIn("scripts.q1b_full_ppl", command)


if __name__ == "__main__":
    unittest.main()
