"""Focused CPU checks on the new evidence-only join, not prior experiments."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pyarrow.parquet as pq

from scripts import q3_q4_joint_results as q


class JointResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = q.source_paths(q.ROOT)
        cls.data = {}
        files = []
        for key, relative in cls.paths.items():
            path = q.ROOT / relative
            cls.data[key] = (pq.read_table(path).to_pylist() if path.suffix == ".parquet" else
                             path.read_text() if path.suffix == ".md" else q.load(path))
            files.append({"key": key, "source_path": str(relative), "sha256": q.sha(path)})
        cls.data["provenance"] = {"files": files, "starting_head": q.STARTING_HEAD, "execution_head": "test"}
        cls.reg = q.registry(cls.data)
        cls.by_name = {r["method_config_id"]: r for r in cls.reg}

    def test_original_scientific_stages_not_promoted(self):
        r = self.by_name["k4v4"]
        self.assertEqual([r[k] for k in ("fast_ppl_status", "full_ppl_status", "longbench_e_status")], ["PASS"] * 3)
        self.assertEqual(r["longbench_v2_status"], "FAIL")
        self.assertEqual(r["v2_primary_noninferiority"], "INCONCLUSIVE")
        self.assertEqual((r["joint_quality_status"], r["failure_stage"]), ("quality_fail", "Q2B"))
        self.assertTrue(r["entered_v2_finalist_validation"])

    def test_nine_eight_one_zero_and_baseline_separate(self):
        c = [r for r in self.reg if r["is_compressed"]]
        self.assertEqual(len(c), 9)
        self.assertEqual(sum(r["joint_quality_status"] == "quality_fail" for r in c), 8)
        self.assertEqual(sum(r["joint_quality_status"] == "quality_inconclusive" for r in c), 1)
        self.assertEqual(sum(r["fully_qualified_compressed"] for r in c), 0)
        self.assertEqual(self.by_name["bf16"]["joint_quality_status"], "reference_baseline")
        self.assertEqual(self.by_name["tq_4bit_nc"]["full_ppl_status"], "NOT_RUN_DUE_TO_FAST_STAGE_STATUS")

    def test_original_batch_fail_effective_b1_and_no_unexecuted_pass(self):
        for r in self.reg:
            self.assertEqual(r["original_cross_batch_gate"], "FAIL")
            self.assertEqual(r["physical_quality_batch_size"], 1)
            self.assertEqual(r["b1_q0_eligibility"], "eligible")
            self.assertEqual(r["q2c_status"], "not_run_no_eligible_finalist")
            self.assertEqual(r["native_prefill_status"], "not_run_conditional_finalists_only")

    def test_saved_pair_arithmetic_and_discordance(self):
        rows = q.arithmetic_check(self.data["q2b_summary"], self.data["q2b_pairs"]["rows"])
        self.assertEqual(len(rows), 11)
        self.assertEqual(sum(r["transition"] == "correct_to_wrong" for r in rows), 8)
        self.assertEqual(sum(r["transition"] == "wrong_to_correct" for r in rows), 3)
        bad = deepcopy(self.data["q2b_summary"])
        bad["overall"]["retention_denominator"] = 321
        with self.assertRaisesRegex(q.JointError, "arithmetic"):
            q.arithmetic_check(bad, self.data["q2b_pairs"]["rows"])

    def test_exact_join_grain_replacements_and_infeasible_rows(self):
        rows, counts, _ = q.join_performance(self.data, self.reg)
        self.assertEqual([counts[k] for k in ("slots", "accepted", "capacity_infeasible", "replacements")], [2670, 2205, 465, 38])
        self.assertEqual(len({r["logical_record_id"] for r in rows}), 2670)
        self.assertEqual(counts["unmatched_identity_rows"], 0)
        self.assertTrue(all(r["host_wall_process_median_ms"] is None for r in rows if r["run_status"] == "capacity_infeasible"))
        self.assertTrue(all(r["quality_status"] == "unvalidated" and r["r_hbm"] is None for r in rows))

    def test_unknown_source_preserved_unmatched_not_name_joined(self):
        data = dict(self.data)
        data["freeze_performance_inventory"] = deepcopy(data["freeze_performance_inventory"])
        data["freeze_performance_inventory"][0]["adapter_source_hash"] = "0" * 64
        rows, counts, _ = q.join_performance(data, self.reg)
        self.assertEqual(counts["unmatched_identity_rows"], 1)
        self.assertEqual(len(rows), 2670)
        self.assertIsNone(rows[0]["b1_config_quality_outcome"])
        self.assertFalse(rows[0]["row_quality_claim_eligible"])

    def test_duplicate_stage_or_slot_never_multiplies_rows(self):
        with self.assertRaisesRegex(q.JointError, "duplicate quality fingerprint"):
            q.join_performance(self.data, self.reg + [self.reg[0]])
        data = dict(self.data)
        data["freeze_performance_inventory"] = self.data["freeze_performance_inventory"] + self.data["freeze_performance_inventory"][:1]
        with self.assertRaisesRegex(q.JointError, "duplicate inventory"):
            q.join_performance(data, self.reg)

    def test_shape_mode_and_accepted_run_mismatch_rejected(self):
        for field, value in (("graph_mode", "eager"), ("batch_size", 2), ("run_id", "incorrect-run")):
            data = dict(self.data)
            data["wall_raw_run_index"] = deepcopy(data["wall_raw_run_index"])
            data["wall_raw_run_index"][0][field] = value
            with self.assertRaisesRegex(q.JointError, "identity mismatch"):
                q.join_performance(data, self.reg)

    def test_b1_not_sufficient_and_bigger_batch_not_quality_failure(self):
        rows, _, maps = q.join_performance(self.data, self.reg)
        self.assertFalse(any(r["row_quality_claim_eligible"] for r in rows))
        self.assertTrue(all(r["quality_scope_status"] == "outside_evaluated_quality_scope" for r in rows if r["batch_size"] != 1))
        self.assertTrue(all("metadata_only" in r["quality_scope_status"] for r in rows if r["batch_size"] == 1))
        self.assertTrue(all(m["graph_eager_relationship"]["fixed_l_graph_relationship_only"] for m in maps.values()))

    def test_stored_same_work_ratio_not_recomputed_or_filled(self):
        rows, _, _ = q.join_performance(self.data, self.reg)
        ratios = {q.point_key(r): r for r in self.data["wall_same_work_ratios"]}
        for r in rows:
            original = ratios.get(q.point_key(r))
            self.assertEqual(r["performance_only_ratio"], original["performance_only_ratio"] if original else None)

    def test_cpu_reproduction_empty_schema_and_historical_preservation(self):
        before = {str(p): q.sha(q.ROOT / p) for p in self.paths.values()}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "inputs-bundle"
            root.mkdir()
            files = []
            for key, relative in self.paths.items():
                p = q.ROOT / relative
                name = key + (".parquet" if p.suffix == ".parquet" else ".md" if p.suffix == ".md" else ".json")
                q.write_new(root / name, p.read_bytes())
                files.append({"key": key, "bundle_path": name, "sha256": q.sha(p)})
            q.write_json(root / "source_manifest.json", {**self.data["provenance"], "files": files})
            first, second = Path(tmp) / "one", Path(tmp) / "two"
            q.derive(q.read_inputs(root), first)
            subprocess.run([sys.executable, str(q.ROOT / "scripts/q3_q4_joint_results.py"), "--reproduce", str(root), "--output", str(second)], check=True, capture_output=True)
            self.assertEqual({p.name: q.sha(p) for p in first.iterdir()}, {p.name: q.sha(p) for p in second.iterdir()})
            empty = pq.read_table(first / "qualified_compressed_candidates.parquet")
            self.assertEqual(empty.num_rows, 0)
            self.assertIn("method_config_fingerprint", empty.column_names)
            summary = q.load(first / "joint_result_summary.json")
            self.assertIsNone(summary["best_qualified_compressed_speedup"])
            self.assertEqual(summary["reason"], q.EMPTY_REASON)
            with self.assertRaises(FileExistsError):
                q.derive(q.read_inputs(root), first)
            q.write_new(root / "tampered.json", b"{}")
            provenance = q.load(root / "source_manifest.json")
            provenance["files"][0]["bundle_path"] = "tampered.json"
            (root / "source_manifest.json").write_bytes(q.encoded(provenance))
            with self.assertRaisesRegex(q.JointError, "compact source changed"):
                q.read_inputs(root)
        self.assertEqual(before, {str(p): q.sha(q.ROOT / p) for p in self.paths.values()})

    def test_no_model_or_scorer_import_in_cpu_helper(self):
        tree = ast.parse((q.ROOT / "scripts/q3_q4_joint_results.py").read_text())
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [n.name for v in ast.walk(tree) if isinstance(v, ast.Import) for n in v.names]
        self.assertFalse(any(x and any(v in x for v in ("torch", "transformers", "q2a", "q2b", "phase17_modeling")) for x in imports))


if __name__ == "__main__":
    unittest.main()
