from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import pyarrow.parquet as pq

import scripts.phase17_predict as phase17_predict
import scripts.phase18_offline as offline
import scripts.phase18_package as package


class Phase18ReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name) / "package"
        (cls.root / "data").mkdir(parents=True)
        shutil.copytree(package.PHASE17_ROOT / "models", cls.root / "models")
        for stem in ("analysis_frame", "out_of_fold_predictions", "model_comparison", "knee_estimates"):
            package.parquet_to_csv(package.PHASE17_ROOT / f"{stem}.parquet", cls.root / "data" / f"{stem}.csv")
        for name in ("split_manifest.json", "candidate_spec.json"):
            shutil.copy2(package.PHASE17_ROOT / name, cls.root / "data" / name)
        source = package.source_manifest("0" * 40)
        (cls.root / "source_manifest.json").write_text(json.dumps(source), encoding="utf-8")
        mechanism = {"phase15_common_point_label": "B1/L131071 Graph", "phase14_launch_floor_support": 0, "phase14_identifiable_comparisons": 14, "phase15_cpu_submission_reduced_pairs": 16, "phase15_gpu_idle_reduced_pairs": 16}
        (cls.root / "data/mechanism_summary.json").write_text(json.dumps(mechanism), encoding="utf-8")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_macro_labels_and_percentage_units(self) -> None:
        report, _, _ = offline.audit(self.root)
        metric = report["macro_metric_definition"]
        self.assertEqual(metric["median_label"], "macro_mean_of_cell_median_relative_errors")
        self.assertEqual(metric["p95_label"], "macro_mean_of_cell_p95_relative_errors")
        self.assertFalse(metric["pooled_quantile"])
        self.assertAlmostEqual(metric["median_value"], 0.0791985632276268, places=15)
        self.assertAlmostEqual(metric["p95_value"], 1.645283992575687, places=15)

    def test_outlier_and_not_applicable_are_preserved(self) -> None:
        report, tail, _ = offline.audit(self.root)
        self.assertEqual(len(tail), 20)
        self.assertEqual(tail[0]["held_out_batch"], 1)
        self.assertGreater(tail[0]["absolute_relative_error"], 50.0)
        self.assertEqual(report["bf16_leave_one_batch_out"]["diagnosis"], "real_predictive_failure_concentrated_in_B1_short_context_edge_extrapolation")
        rows = offline._csv_rows(self.root / "data/model_comparison.csv")
        bf16_config = next(row for row in rows if row["protocol"] == "leave_one_config_out" and row["method_family"] == "bf16" and row["model_id"] == "D" and row["subset"] == "combined")
        self.assertEqual(int(bf16_config["scored_rows"]), 0)

    def test_predictor_serialization_agrees_with_phase17(self) -> None:
        for configuration, batch, context in (("bf16", 1, 4096), ("tq_4bit_nc", 4, 16384), ("k4v4", 2, 32768), ("kvq4", 1, 4096)):
            new = offline.predict(self.root, method_config=configuration, batch=batch, context=context)
            old = phase17_predict.predict(package.PHASE17_ROOT, configuration=configuration, batch=batch, historical_context=context)
            self.assertAlmostEqual(new["predicted_host_wall_ms"], old["predicted_host_wall_ms"], places=12)
            self.assertAlmostEqual(new["predicted_same_work_ratio"], old["predicted_same_work_ratio"], places=12)

    def test_ratio_scope_and_unvalidated_labels(self) -> None:
        result = offline.predict(self.root, method_config="kvq4", batch=1, context=4096)
        self.assertEqual(result["ratio_scope"], "fully_predicted_same_work")
        self.assertEqual(result["quality_status"], "unvalidated")
        self.assertFalse(result["performance_claim_eligible"])
        self.assertEqual(result["feasibility_status"], "not_predicted")

    def test_default_reproduction_has_no_gpu_dependency(self) -> None:
        self.assertNotIn("torch", sys.modules)
        with tempfile.TemporaryDirectory() as output:
            target = Path(output) / "fresh"
            result = offline.reproduce(self.root, target)
            self.assertFalse(result["gpu_launched"])
            self.assertFalse(result["network_accessed"])
            self.assertEqual(result["figure_count"], 6)

    def test_source_references_and_figures(self) -> None:
        source = json.loads((self.root / "source_manifest.json").read_text())
        self.assertEqual(source["phase17_root_sha256"], package.PHASE17_SHA256)
        self.assertFalse(source["historical_bundles_copied"])
        with tempfile.TemporaryDirectory() as output:
            target = Path(output) / "fresh"
            offline.reproduce(self.root, target)
            self.assertEqual(len(list((target / "figures").glob("*.svg"))), 6)


if __name__ == "__main__":
    unittest.main()
