"""Focused Q2A quality-only contract tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import q2a_longbench_e as q2a


class Q2ATests(unittest.TestCase):
    def test_exact_scope_and_frozen_plan(self) -> None:
        plan = q2a.frozen_plan()
        self.assertEqual(plan["configurations"], ["bf16", "k4v4"])
        self.assertEqual(plan["physical_batch_size"], 1)
        self.assertEqual(plan["planned_pairs"], 3668)
        self.assertEqual(plan["planned_outputs"], 7336)
        self.assertEqual(tuple(plan["tasks"]), q2a.TASKS)
        self.assertEqual(len(plan["entries"]), 3668)
        self.assertEqual(sum(plan["task_counts"].values()), 3668)
        self.assertEqual(plan["authority"]["original_cross_batch_gate"], "FAILED")

    def test_suffix_concatenation(self) -> None:
        prompt = list(range(33))
        prefix, suffix = q2a.split_prompt(prompt)
        self.assertEqual(prefix + suffix, prompt)
        self.assertEqual(suffix, list(range(17, 33)))
        with self.assertRaises(q2a.Q2AError):
            q2a.split_prompt(list(range(16)))

    def test_autoregressive_feeds_generated_tokens_only(self) -> None:
        fed = []
        with patch.object(q2a, "_next_token", side_effect=[17, 23, 128009]):
            tokens, reason = q2a.autoregressive_tokens(
                object(), task="qasper", budget=8, eos=128009,
                newline=None, advance=lambda token, index: fed.append((token, index)) or object(),
            )
        self.assertEqual(tokens, [17, 23, 128009])
        self.assertEqual(fed, [(17, 0), (23, 1)])
        self.assertEqual(reason, "eos")

    def test_samsum_newline_and_budget_stops(self) -> None:
        self.assertEqual(q2a._stop_reason(198, "samsum", 128009, 198), "samsum_newline")
        self.assertIsNone(q2a._stop_reason(198, "qasper", 128009, 198))
        with patch.object(q2a, "_next_token", side_effect=[5, 6]):
            tokens, reason = q2a.autoregressive_tokens(
                object(), task="qasper", budget=2, eos=128009,
                newline=198, advance=lambda *_: object(),
            )
        self.assertEqual(tokens, [5, 6])
        self.assertEqual(reason, "budget_exhausted")
        with patch.object(q2a, "_next_token", side_effect=[7, 198]) as select:
            tokens, reason = q2a.autoregressive_tokens(
                object(), task="samsum", budget=4, eos=128009,
                newline=198, advance=lambda *_: object(),
            )
        self.assertEqual(tokens, [7, 198])
        self.assertEqual(reason, "samsum_newline")
        self.assertEqual(select.call_args_list[0].kwargs["blocked_tokens"], (128009, 198))

    def test_bucket_weighting_is_not_sample_micro(self) -> None:
        rows = [
            {"length_bucket": "0-4k", "bf16_score": 1.0, "k4v4_score": 0.8},
            {"length_bucket": "4-8k", "bf16_score": 0.5, "k4v4_score": 0.4},
            {"length_bucket": "8k+", "bf16_score": 0.0, "k4v4_score": 0.0},
        ]
        self.assertEqual(q2a._task_score(rows, "bf16"), (50.0, {"0-4k": 100.0, "4-8k": 50.0, "8k+": 0.0}))

    def test_gate_boundaries_and_guardrails(self) -> None:
        margins = {"macro_drop_max_score_points": 2, "category_hard_fail_score_points": 5,
                   "invalid_output_increase_max_percentage_points": 1}
        self.assertEqual(q2a.classify_gate(1.5, [0.5, 2.0], {"qa": 5.0}, 1.0, complete=True, margins=margins)[0], "PASS")
        self.assertEqual(q2a.classify_gate(2.0, [1.0, 2.5], {}, 0, complete=True, margins=margins)[0], "INCONCLUSIVE")
        self.assertEqual(q2a.classify_gate(3.0, [2.1, 4.0], {}, 0, complete=True, margins=margins)[0], "FAIL")
        self.assertEqual(q2a.classify_gate(0.0, [-1.0, 1.0], {"qa": 5.1}, 0, complete=True, margins=margins)[0], "FAIL")
        self.assertEqual(q2a.classify_gate(0.0, [-1.0, 1.0], {}, 1.1, complete=True, margins=margins)[0], "FAIL")
        self.assertEqual(q2a.classify_gate(0.0, [-1.0, 1.0], {}, 0, complete=False, margins=margins)[0], "INCONCLUSIVE")

    def test_official_metric_multi_reference_and_parser(self) -> None:
        score, parsed = q2a._official_metric("qasper", "The answer", ["wrong", "the answer"], None)
        self.assertEqual(score, 1.0)
        self.assertEqual(parsed, "The answer")
        score, parsed = q2a._official_metric("trec", "HUM\nOther", ["HUM"], ["HUM", "LOC"])
        self.assertEqual(score, 1.0)
        self.assertEqual(parsed, "HUM")
        score, parsed = q2a._official_metric("passage_count", "There are 7 passages.", ["7"], None)
        self.assertEqual(score, 1.0)
        self.assertEqual(parsed, "There are 7 passages.")

    def test_paired_task_bootstrap_is_deterministic(self) -> None:
        rows = [
            {"task": task, "length_bucket": bucket, "bf16_score": 0.5, "k4v4_score": 0.49}
            for task in q2a.TASKS for bucket in q2a.BUCKETS for _ in range(100)
        ]
        self.assertEqual(q2a._bootstrap(rows, seed=20260722, draws=10), (1.0, 1.0))

    def test_completed_unit_is_not_regenerated(self) -> None:
        item = {"task": "qasper", "sample_id": "sample", "prompt_token_sha256": "a" * 64,
                "generation_budget": 32}
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary)
            result = {**item, "status": "COMPLETED", "configuration": "bf16",
                      "contract_sha256": q2a.q0.CONTRACT_SHA256,
                      "amendment_sha256": q2a.q1a.AMENDMENT_SHA256,
                      "quality_image_digest": q2a.q0.QUALITY_IMAGE,
                      "physical_batch_size": 1, "execution_mode": "cache_sensitive_growing_eager",
                      "generated_text": "ok", "generated_text_sha256": q2a.q0.sha256_bytes(b"ok")}
            q2a._write_attempt(campaign, "bf16", item, result, "attempt-00")
            self.assertEqual(q2a._accepted(campaign, "bf16", item)["generated_text"], "ok")
            with self.assertRaises(q2a.Q2AError):
                q2a._write_attempt(campaign, "bf16", item, result, "attempt-00")


if __name__ == "__main__":
    unittest.main()
