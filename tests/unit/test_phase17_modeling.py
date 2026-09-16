from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np

import scripts.phase17_modeling as phase17
import scripts.phase17_predict as predictor


def row(*, config: str = "kvq4", family: str = "kvquant", batch: int = 1,
        context: int = 4096, replicate: int = 0, latency: float = 1.0,
        r_alloc: float = 3.8) -> dict[str, object]:
    return {
        "run_id": f"{config}-b{batch}-l{context}-r{replicate}",
        "status": "completed", "method_config_id": config,
        "method_family": family, "batch_size": batch,
        "context_label": context, "historical_context": context,
        "total_attended_context": context + 1, "replicate_index": replicate,
        "grid_source": "base", "response_host_wall_ms": latency,
        "r_alloc": r_alloc, "metadata_fraction": 0.01,
        "full_precision_fraction": 0.02, "outlier_fraction": 0.03,
        "workspace_fraction": 0.04, "kernel_count": 7,
    }


class Phase17ModelingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame, cls.summary = phase17.build_analysis_frame()

    def test_host_wall_units_and_supplement_join(self) -> None:
        completed = [r for r in self.frame if r["status"] == "completed"]
        self.assertEqual(len(completed), phase17.EXPECTED_COMPLETED)
        self.assertTrue(all(r["latency_basis"] == "host_wall" for r in completed))
        self.assertTrue(all(r["latency_units"] == "milliseconds_per_full_batch_decode_step" for r in completed))

    def test_unique_replacements_and_infeasible_exclusion(self) -> None:
        self.assertEqual(sum(r["replacement_of"] is not None for r in self.frame), 38)
        eligible = [r for r in self.frame if r["status"] == "completed"]
        self.assertEqual(len(eligible), 2205)
        self.assertTrue(all(r["response_host_wall_ms"] is None for r in self.frame if r["status"] == "capacity_infeasible"))

    def test_grouped_splits_have_no_leakage(self) -> None:
        manifest = phase17.build_split_manifest(self.frame)
        for fold in manifest["folds"]:
            overlap = set(fold["train_group_keys"]) & set(fold["test_group_keys"])
            if fold["protocol"] == "session_holdout":
                self.assertTrue(fold["same_geometry_across_sides"])
            else:
                self.assertFalse(overlap)

    def test_preprocessing_is_training_only(self) -> None:
        train = np.asarray([[0.0], [2.0]])
        transformed, state = phase17._standardize_fit(train)
        held = phase17._standardize_apply(np.asarray([[100.0]]), state)
        self.assertAlmostEqual(float(transformed.mean()), 0.0)
        self.assertGreater(float(held[0, 0]), 90.0)

    def test_deterministic_fit_and_metrics(self) -> None:
        values = [row(context=4096 * (i + 1), latency=1.0 + i) for i in range(5)]
        one = phase17.fit_linear_predictive(values, "E")
        two = phase17.fit_linear_predictive(values, "E")
        self.assertEqual(one, two)
        metrics = phase17._metric_values([{"absolute_error_ms": 1.0, "absolute_relative_error": 0.1}, {"absolute_error_ms": 3.0, "absolute_relative_error": 0.3}])
        self.assertAlmostEqual(metrics["mae_ms"], 2.0)
        self.assertAlmostEqual(metrics["median_absolute_relative_error"], 0.2)

    def test_non_identifiable_knee_is_not_forced(self) -> None:
        values = [row(context=4096 * (i + 1), replicate=r, latency=1.0) for i in range(5) for r in range(5)]
        fitted = phase17.local_curve_fit(values)
        self.assertNotEqual(fitted["fit_status"], "identified_in_range")

    def test_same_work_requires_exact_geometry(self) -> None:
        predictions = []
        for config, family, observed, predicted in (("bf16", "bf16", 2.0, 2.2), ("kvq4", "kvquant", 1.0, 1.1)):
            predictions.append({"protocol": "leave_one_batch_out", "fold_id": "B1", "model_id": "E", "method_family": family, "method_config_id": config, "batch_size": 1, "historical_context": 4096, "grid_source": "base", "domain_status": "edge", "observed_host_wall_ms": observed, "predicted_host_wall_ms": predicted, "absolute_error_ms": abs(observed-predicted), "absolute_relative_error": abs(observed-predicted)/observed})
        summary = phase17.ratio_and_ranking_metrics(predictions)
        match = next(r for r in summary["speedup_sign"] if r["model_id"] == "E")
        self.assertEqual(match["eligible"], 1)

    def test_method_family_mapping(self) -> None:
        self.assertEqual(phase17._family("k4v4"), "kivi")
        self.assertEqual(phase17._family("kvq4"), "kvquant")

    def test_exported_predictor_matches_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "models").mkdir()
            family_rows = [
                row(context=4096 * (index + 1), latency=1.0 + index * 0.2,
                    r_alloc=predictor.derived_r_alloc("kvq4", 1, 4096 * (index + 1)))
                for index in range(5)
            ]
            bf16_rows = [
                row(config="bf16", family="bf16", context=4096 * (index + 1),
                    latency=2.0 + index * 0.2,
                    r_alloc=predictor.derived_r_alloc("bf16", 1, 4096 * (index + 1)))
                for index in range(5)
            ]
            kv_model = phase17.fit_linear_predictive(family_rows, "E")
            bf_model = phase17.fit_linear_predictive(bf16_rows, "E")
            for model, family, configs in ((kv_model, "kvquant", ["kvq4"]), (bf_model, "bf16", ["bf16"])):
                model.update({"training_domain": {"batch_min": 1, "batch_max": 1, "historical_context_min": 4096, "historical_context_max": 20480, "r_alloc_min": min(float(r["r_alloc"]) for r in (family_rows if family == "kvquant" else bf16_rows)), "r_alloc_max": max(float(r["r_alloc"]) for r in (family_rows if family == "kvquant" else bf16_rows))}, "training_configurations": configs, "feature_requirements": ["B", "L", "r_alloc"], "new_process_log_residual_interval_95": [-0.1, 0.1]})
                (root / "models" / f"{family}-E.json").write_text(__import__("json").dumps(model), encoding="utf-8")
            (root / "models/index.json").write_text(__import__("json").dumps({"deployment_model_id": "E", "families": {"kvquant": {"E": "kvquant-E.json"}, "bf16": {"E": "bf16-E.json"}}}), encoding="utf-8")
            actual = predictor.predict(root, configuration="kvq4", batch=1, historical_context=4096)
            expected = float(phase17.predict_linear_predictive(kv_model, [family_rows[0]])[0])
            self.assertAlmostEqual(actual["predicted_host_wall_ms"], expected)


if __name__ == "__main__":
    unittest.main()
