from __future__ import annotations

import math
import tempfile
from pathlib import Path
import unittest

import torch

from scripts import q1a_fast_ppl as q1a


class Q1AFastPPLTests(unittest.TestCase):
    def test_amendment_preserves_original_and_is_approved(self) -> None:
        result = q1a.verify_amendment()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["amendment"]["scope"]["quality_evaluated_batch_sizes"], [1])
        self.assertEqual(result["amendment"]["scope"]["original_cross_batch_gate"], "failed")

    def test_all_ten_configs_reuse_existing_b1_q0_passes(self) -> None:
        result = q1a.b1_q0_eligibility()
        self.assertEqual(result["eligible_count"], 10)
        self.assertTrue(all(row["eligible"] for row in result["rows"]))
        self.assertTrue(all(row["original_cross_batch_gate"] == "FAIL" for row in result["rows"]))

    def test_fast_plan_has_frozen_counts_and_scoring_split(self) -> None:
        plan = q1a.fast_plan()
        self.assertEqual(plan["anchors_per_configuration"], 128)
        self.assertEqual(plan["scored_tokens_per_configuration"], 16384)
        self.assertEqual(plan["total_anchor_units"], 1280)
        self.assertEqual(plan["total_scored_tokens"], 163840)
        self.assertTrue(all(row["target_start"] == row["burn_in_index"] + 1 for row in plan["anchors"]))

    def test_first_target_is_scored_from_existing_logits(self) -> None:
        logits = torch.tensor([0.0, 2.0, -1.0])
        target = torch.tensor(1)
        nll, count = q1a.teacher_forced_loss(logits, target, 1)
        expected = -torch.log_softmax(logits.float(), dim=-1)[1].item()
        self.assertAlmostEqual(nll, expected)
        self.assertEqual(count, 1)

    def test_mask_and_nll_to_ppl(self) -> None:
        logits = torch.tensor([0.0, 2.0])
        nll, count = q1a.teacher_forced_loss(logits, torch.tensor(1), 0)
        self.assertEqual((nll, count), (0.0, 0))
        self.assertAlmostEqual(math.exp(math.log(3.0)), 3.0)

    def test_paired_bootstrap_is_deterministic_and_clustered(self) -> None:
        rows = [
            {"cluster_id": "a", "delta_nll": 0.0, "scored_token_count": 128},
            {"cluster_id": "b", "delta_nll": 0.01, "scored_token_count": 128},
        ]
        first = q1a.paired_cluster_bootstrap(rows, seed=7, draws=100)
        second = q1a.paired_cluster_bootstrap(rows, seed=7, draws=100)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["point_delta_nll"], 0.005)

    def test_decision_logic_retains_three_states(self) -> None:
        margin = math.log1p(0.01)
        self.assertEqual(q1a.classify_fast(
            {"ci_lower_delta_nll": 0.0, "ci_upper_delta_nll": margin - 1e-6},
            [{"relative_ppl_change": 0.0}],
        ), "pass")
        self.assertEqual(q1a.classify_fast(
            {"ci_lower_delta_nll": margin + 1e-6, "ci_upper_delta_nll": margin + 2e-6},
            [{"relative_ppl_change": 0.0}],
        ), "fail")
        self.assertEqual(q1a.classify_fast(
            {"ci_lower_delta_nll": 0.0, "ci_upper_delta_nll": margin + 1e-6},
            [{"relative_ppl_change": 0.0}],
        ), "inconclusive")

    def test_length_hard_fail_is_not_hidden_by_global_average(self) -> None:
        self.assertEqual(q1a.classify_fast(
            {"ci_lower_delta_nll": 0.0, "ci_upper_delta_nll": 0.0},
            [{"relative_ppl_change": 0.050001}],
        ), "fail")

    def test_b_greater_than_one_is_not_in_effective_scope(self) -> None:
        scope = q1a.verify_amendment()["amendment"]["scope"]
        self.assertNotIn(2, scope["quality_evaluated_batch_sizes"])
        self.assertEqual(scope["quality_transfer_to_other_batches"], "not_established")

    def test_completed_anchor_is_not_rewritten(self) -> None:
        anchor = q1a.fast_plan()["anchors"][0]
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)
            result = {"status": "PASS"}
            q1a._finalize_unit(campaign, "bf16", anchor, result)
            self.assertTrue(q1a._unit_complete(campaign, "bf16", anchor))
            with self.assertRaises(q1a.Q1AError):
                q1a._finalize_unit(campaign, "bf16", anchor, result)


if __name__ == "__main__":
    unittest.main()
