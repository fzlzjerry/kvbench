import gzip
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import q2b_longbench_v2 as q


class Q2BTests(unittest.TestCase):
    def row(self, b, m, *, category="qa", bucket="8-16k", invalid=False):
        return {"bf16_correct": bool(b), "k4v4_correct": bool(m),
                "bf16_invalid": False, "k4v4_invalid": invalid,
                "category": category, "length_bucket": bucket}

    def test_frozen_primary_plan_and_finalist_deduplication(self):
        p = q.frozen_plan()
        self.assertEqual(p["configurations"], ["bf16", "k4v4"])
        self.assertEqual(p["physical_batch_size"], 1)
        self.assertEqual((p["selected_count"], p["planned_pairs"], p["excluded_count"], p["planned_outputs"]), (503, 321, 182, 642))
        self.assertEqual(p["finalist_roles"], {"k4v4": ["best_quality", "highest_r_alloc"]})
        self.assertEqual(len({r["sample_id"] for r in p["eligibility_records"]}), 503)
        self.assertTrue(all(r["task"] == q.TASK and r["eligible"] and r["generation_budget"] == 8 for r in p["entries"]))
        self.assertTrue(all("answer" not in r and "reference_answer" not in r for r in p["entries"]))
        self.assertFalse(p["generation"]["cot"])
        self.assertFalse(p["generation"]["parser_early_stop"])
        self.assertTrue(all("cot_stress" not in k for k in p["input_files"]))
        self.assertEqual(p["margins"]["bootstrap"]["longbench_v2_unit"], "paired_sample_stratified_by_length_and_category")

    def test_exact_official_parser_not_bare_letter_or_favorable_option(self):
        for text, answer in (("The correct answer is (A)", "A"), ("**The correct answer is D**", "D"),
                             ("The correct answer is B and The correct answer is (C)", "C"),
                             ("A", None), ("Answer: B", None), ("the correct answer is C", None),
                             ("The correct answer is (E)", None), ("", None)):
            self.assertEqual(q.parse_answer(text)["parsed_answer"], answer)
            self.assertEqual(q.parse_answer(text)["invalid"], answer is None)

    def test_suffix_and_first_answer_eight_token_accounting(self):
        prefix, suffix = q.q2a.split_prompt(list(range(40)))
        self.assertEqual(prefix + suffix, list(range(40)))
        self.assertEqual(len(suffix), 16)
        fed = []
        with patch.object(q.q2a, "_next_token", side_effect=list(range(10, 18))):
            tokens, reason = q.q2a.autoregressive_tokens(object(), task=q.TASK, budget=8,
                eos=128009, newline=None, advance=lambda token, index: fed.append((token, index)) or object())
        self.assertEqual(tokens, list(range(10, 18)))
        self.assertEqual(fed, list(zip(range(10, 17), range(7))))
        self.assertEqual(reason, "budget_exhausted")
        with patch.object(q.q2a, "_next_token", return_value=128009):
            tokens, reason = q.q2a.autoregressive_tokens(object(), task=q.TASK, budget=8,
                eos=128009, newline=None, advance=lambda *_: self.fail("must not advance after EOS"))
        self.assertEqual(tokens, [128009])
        self.assertEqual(reason, "eos")

    def test_streaming_tokens_exact_without_retokens_or_full_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokens.gz"
            data = q.q0._i32_bytes(range(100))
            q.q0.write_new(path, gzip.compress(data))
            item = {"offset": 3, "prompt_length": 17, "prompt_token_sha256": q.q0.sha256_bytes(data[12:80])}
            with gzip.open(path, "rb") as stream:
                self.assertEqual(q.read_prompt(stream, item), list(range(3, 20)))
                with self.assertRaises(q.Q2BError):
                    q.read_prompt(stream, {**item, "prompt_token_sha256": "bad"})

    def test_contingency_is_overall_not_baseline_correct_only(self):
        rows = [self.row(1, 1)] * 9 + [self.row(1, 0)] + [self.row(0, 1)] * 2 + [self.row(0, 0, invalid=True)] * 8
        result = q.contingency(rows)
        self.assertEqual([result[k] for k in ("n11", "n10", "n01", "n00")], [9, 1, 2, 8])
        self.assertEqual(result["pairs"], 20)
        self.assertEqual(result["bf16_accuracy"], 0.5)
        self.assertEqual(result["k4v4_accuracy"], 0.55)
        self.assertEqual(result["accuracy_drop_pp"], -5)
        self.assertEqual(result["retention"], 0.9)
        self.assertEqual(result["retention_denominator"], 10)
        self.assertIsNone(q.contingency([self.row(0, 1)])["retention"])

    def test_bootstrap_pairing_determinism_and_strata(self):
        rows = [self.row(1, 1)] * 8 + [self.row(0, 0, category="code", bucket="32-64k")] * 2
        a = q.paired_summary(rows, draws=100, seed=20260722)
        b = q.paired_summary(rows, draws=100, seed=20260722)
        self.assertEqual(a, b)
        self.assertEqual(a["paired_ci95_pp"], [0, 0])
        self.assertEqual(a["retention_ci95"], [1, 1])
        self.assertEqual(sorted(r["count"] for r in a["bootstrap_strata"]), [2, 8])
        self.assertIsNone(q.paired_summary([self.row(0, 0)], draws=10)["retention_ci95"])

    def test_gate_boundaries_units_and_retention_point_not_lower_ci(self):
        margins = {"accuracy_drop_max_percentage_points": 2, "category_hard_fail_percentage_points": 5,
                   "length_hard_fail_percentage_points": 5, "invalid_output_increase_max_percentage_points": 1,
                   "bf16_correct_retention_min": 0.95}
        s = {"paired_ci95_pp": [-1, 2], "retention": 0.95, "retention_ci95": [0.8, 1], "invalid_increase_pp": 1}
        groups = [{"name": "x", "pairs": 10, "accuracy_drop_pp": 5}]
        classify = lambda value, c=groups, l=groups, complete=True: q.classify_gate(value, c, l, complete=complete, margins=margins)[0]
        self.assertEqual(classify(s), "PASS")
        self.assertEqual(classify({**s, "paired_ci95_pp": [1, 2.01]}), "INCONCLUSIVE")
        self.assertEqual(classify({**s, "paired_ci95_pp": [2.01, 4]}), "FAIL")
        self.assertEqual(classify({**s, "retention": 0.949}), "FAIL")
        self.assertEqual(classify({**s, "invalid_increase_pp": 1.01}), "FAIL")
        self.assertEqual(classify(s, c=[{**groups[0], "accuracy_drop_pp": 5.01}]), "FAIL")
        self.assertEqual(classify(s, l=[{**groups[0], "accuracy_drop_pp": 5.01}]), "FAIL")
        self.assertEqual(classify(s, complete=False), "INCONCLUSIVE")

    def test_invalid_generated_answer_is_accepted_not_missing_or_retried(self):
        item = {"task": q.TASK, "sample_id": "sample", "prompt_token_sha256": "a" * 64, "generation_budget": 8, "prompt_length": 40}
        plan = {"method_fingerprints": {"bf16": "b" * 64}, "authority": {"parser_source_sha256": "c" * 64}}
        result = {**item, "status": "COMPLETED", "configuration": "bf16", "contract_sha256": q.q0.CONTRACT_SHA256,
            "amendment_sha256": q.q1a.AMENDMENT_SHA256, "quality_image_digest": q.q0.QUALITY_IMAGE,
            "physical_batch_size": 1, "execution_mode": "cache_sensitive_growing_eager", "cot": False,
            "plan_sha256": q.q0.sha256_bytes(q.q0.canonical_bytes(plan)), "contract_method_config_fingerprint": "b" * 64,
            "parser_source_sha256": "c" * 64, "prefill_length": 24, "conditioning_length": 16,
            "generated_token_ids": [128009], "generated_token_count": 1,
            "generated_ids_sha256": q.q0.sha256_bytes(q.q0._i32_bytes([128009])),
            "generated_text": "", "generated_text_sha256": q.q0.sha256_bytes(b""), "active_context": 40,
            "attempt_id": "attempt-00", **q.parse_answer("")}
        with tempfile.TemporaryDirectory() as tmp:
            campaign = Path(tmp)
            self.assertIsNone(q.accepted(campaign, "bf16", item, plan))
            q.q2a._write_attempt(campaign, "bf16", item, result, "attempt-00")
            a = q.accepted(campaign, "bf16", item, plan)
            self.assertTrue(a["invalid"])
            self.assertEqual(a, q.accepted(campaign, "bf16", item, plan))
            with self.assertRaises(q.q2a.Q2AError):
                q.q2a._write_attempt(campaign, "bf16", item, result, "attempt-00")
        with tempfile.TemporaryDirectory() as tmp:
            campaign = Path(tmp)
            path = q.q2a._unit(campaign, "bf16", item) / "attempt-00/result.json"
            q.q0.write_json_new(path, result)
            digest = q.sha(path)
            self.assertTrue(q.accepted(campaign, "bf16", item, plan)["invalid"])
            self.assertEqual(q.sha(path), digest)

    def test_parser_tamper_rejection(self):
        q.pinned_parser.cache_clear()
        with patch.object(q, "sha", return_value="tampered"):
            with self.assertRaises(q.Q2BError):
                q.pinned_parser()
        q.pinned_parser.cache_clear()


if __name__ == "__main__":
    unittest.main()
