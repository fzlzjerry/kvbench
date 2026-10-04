"""Strict configuration-schema tests for Phase 2 contracts."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from kvbench.config import (
    load_config,
    load_json_compatible_yaml,
    parse_config,
    validate_all_example_configs,
)
from kvbench.errors import ConfigLoadError, SchemaValidationError
from kvbench.schema import (
    ExperimentConfig,
    HardwareManifest,
    MethodConfig,
    ModelIdentity,
    ModelIdentityV2,
    Phase3AdmissionPlan,
    canonical_json_bytes,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPOSITORY_ROOT / "configs"


def _example(
    *, document_type: str, discriminator: tuple[str, str] | None = None
) -> tuple[Path, dict[str, object]]:
    for path in sorted(CONFIG_ROOT.glob("*/*.yaml")):
        raw = load_json_compatible_yaml(path)
        if raw.get("document_type") != document_type:
            continue
        if discriminator is not None and raw.get(discriminator[0]) != discriminator[1]:
            continue
        return path, raw
    raise AssertionError(
        f"missing example document_type={document_type!r}, discriminator={discriminator!r}"
    )


class ExampleConfigurationTests(unittest.TestCase):
    def test_all_versioned_examples_validate_with_references(self) -> None:
        validated = validate_all_example_configs(repository_root=REPOSITORY_ROOT)
        paths = {path for path, _ in validated}
        expected = {
            "configs/hardware/rtx_pro_6000.yaml",
            "configs/models/primary_gqa_model.yaml",
            "configs/methods/bf16.yaml",
            "configs/methods/turboquant.yaml",
            "configs/methods/kivi.yaml",
            "configs/methods/kvquant.yaml",
            "configs/plans/smoke.yaml",
            "configs/plans/pilot.yaml",
            "configs/plans/graph_ab.yaml",
            "configs/plans/profiler_subset.yaml",
            "configs/plans/full_scan.yaml",
            "configs/plans/phase3_bf16_fixed_l.yaml",
            "configs/plans/phase3_bf16_growing.yaml",
        }
        self.assertEqual(paths, expected)
        self.assertEqual(len({fingerprint for _, fingerprint in validated}), len(validated))

    def test_each_document_family_has_a_typed_model(self) -> None:
        expected_types = {
            "hardware": HardwareManifest,
            "model": (ModelIdentity, ModelIdentityV2),
            "method": MethodConfig,
            "experiment": (ExperimentConfig, Phase3AdmissionPlan),
        }
        observed: set[str] = set()
        for path in sorted(CONFIG_ROOT.glob("*/*.yaml")):
            document = load_config(path)
            document_type = document.to_dict()["document_type"]
            self.assertIsInstance(document, expected_types[document_type])
            observed.add(document_type)
        self.assertEqual(observed, set(expected_types))

    def test_primary_model_identity_is_exactly_resolved(self) -> None:
        _, raw_model = _example(document_type="model")
        model = parse_config(raw_model)
        self.assertIsInstance(model, ModelIdentityV2)
        self.assertEqual(model.resolution.status.value, "resolved")
        self.assertEqual(model.model_id, "meta-llama/Llama-3.1-8B-Instruct")
        self.assertEqual(
            model.revision, "0e9e39f249a16976918f6564b8830bc894c89659"
        )
        self.assertEqual(
            model.config_sha256,
            "29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e",
        )
        self.assertIsNotNone(model.geometry)
        assert model.geometry is not None
        self.assertEqual(model.geometry.num_query_heads, 32)
        self.assertEqual(model.geometry.num_kv_heads, 8)
        self.assertEqual(model.geometry.max_context_length, 131072)


class StrictConfigurationTests(unittest.TestCase):
    def test_valid_minimal_configuration_is_accepted(self) -> None:
        path, _ = _example(document_type="hardware")
        document = load_config(path)
        self.assertIsInstance(document, HardwareManifest)
        self.assertEqual(document.g0_status, "PASS")

    def test_unknown_field_is_rejected_without_echoing_value(self) -> None:
        _, raw = _example(document_type="hardware")
        mutated = copy.deepcopy(raw)
        mutated["unexpected_secret"] = "do-not-echo-this-value"
        with self.assertRaises(SchemaValidationError) as caught:
            parse_config(mutated)
        self.assertIn("unknown field", str(caught.exception))
        self.assertNotIn("do-not-echo-this-value", str(caught.exception))

    def test_invalid_enum_is_rejected(self) -> None:
        _, raw = _example(document_type="experiment", discriminator=("plan_kind", "smoke"))
        mutated = copy.deepcopy(raw)
        mutated["graph_modes"] = ["graphish"]
        with self.assertRaises(SchemaValidationError) as caught:
            parse_config(mutated)
        self.assertIn("invalid enum value", str(caught.exception))

    def test_nonpositive_batch_size_is_rejected(self) -> None:
        _, raw = _example(document_type="experiment", discriminator=("plan_kind", "smoke"))
        mutated = copy.deepcopy(raw)
        mutated["grid"]["batch_sizes"] = [0]
        with self.assertRaises(SchemaValidationError) as caught:
            parse_config(mutated)
        self.assertIn("positive integers", str(caught.exception))

    def test_nonpositive_context_length_is_rejected(self) -> None:
        _, raw = _example(document_type="experiment", discriminator=("plan_kind", "smoke"))
        mutated = copy.deepcopy(raw)
        mutated["grid"]["context_lengths"] = [-1]
        with self.assertRaises(SchemaValidationError) as caught:
            parse_config(mutated)
        self.assertIn("positive integers", str(caught.exception))

    def test_nonpositive_model_context_is_rejected(self) -> None:
        _, raw = _example(document_type="model")
        mutated = copy.deepcopy(raw)
        mutated["target_context_length"] = 0
        with self.assertRaises(SchemaValidationError) as caught:
            parse_config(mutated)
        self.assertIn("target_context_length must be positive", str(caught.exception))

    def test_malformed_method_configuration_is_rejected(self) -> None:
        _, raw = _example(document_type="method", discriminator=("method", "turboquant"))
        mutated = copy.deepcopy(raw)
        mutated["variants"][0]["parameters"]["key_bits"] = 7
        with self.assertRaises(SchemaValidationError):
            parse_config(mutated)

    def test_resolved_quantized_variant_rejects_unresolved_parameters(self) -> None:
        for method in ("turboquant", "kvquant"):
            with self.subTest(method=method):
                _, raw = _example(document_type="method", discriminator=("method", method))
                mutated = copy.deepcopy(raw)
                missing_parameter = {
                    "turboquant": "cache_dtype_name",
                    "kvquant": "calibration_artifact_sha256",
                }[method]
                mutated["variants"][0]["parameters"][
                    missing_parameter
                ] = None
                mutated["variants"][0]["resolution"] = {
                    "schema_version": "kvbench.resolution.v1",
                    "status": "resolved",
                    "blockers": [],
                    "reason": None,
                }
                with self.assertRaises(SchemaValidationError):
                    parse_config(mutated)

    def test_kivi_preregistered_group_geometry_is_strict(self) -> None:
        _, raw = _example(document_type="method", discriminator=("method", "kivi"))
        mutated = copy.deepcopy(raw)
        mutated["variants"][0]["parameters"]["group_size"] = 64
        with self.assertRaises(SchemaValidationError):
            parse_config(mutated)

    def test_method_and_parameter_type_mismatch_is_rejected(self) -> None:
        _, raw = _example(document_type="method", discriminator=("method", "bf16"))
        mutated = copy.deepcopy(raw)
        mutated["method"] = "kivi"
        with self.assertRaises(SchemaValidationError):
            parse_config(mutated)

    def test_incompatible_plan_and_run_kind_is_rejected(self) -> None:
        _, raw = _example(document_type="experiment", discriminator=("plan_kind", "smoke"))
        mutated = copy.deepcopy(raw)
        mutated["run_kinds"] = ["ncu"]
        with self.assertRaises(SchemaValidationError) as caught:
            parse_config(mutated)
        self.assertIn("profiler run kinds require profiler_subset", str(caught.exception))

    def test_preregistered_pilot_grid_cannot_be_changed(self) -> None:
        _, raw = _example(document_type="experiment", discriminator=("plan_kind", "pilot"))
        mutated = copy.deepcopy(raw)
        mutated["grid"]["batch_sizes"] = [1, 4]
        with self.assertRaises(SchemaValidationError) as caught:
            parse_config(mutated)
        self.assertIn("pilot grid does not match preregistration", str(caught.exception))

    def test_performance_plan_cannot_claim_quality_validation(self) -> None:
        _, raw = _example(document_type="experiment", discriminator=("plan_kind", "smoke"))
        mutated = copy.deepcopy(raw)
        mutated["quality"]["quality_status"] = "not_applicable"
        mutated["quality"]["claim_eligibility"] = "none"
        with self.assertRaises(SchemaValidationError) as caught:
            parse_config(mutated)
        self.assertIn("unvalidated/performance_only", str(caught.exception))

    def test_admission_controls_and_plan_modes_fail_closed(self) -> None:
        _, raw = _example(document_type="experiment", discriminator=("plan_kind", "smoke"))
        for mutate in (
            lambda item: item["admission"].__setitem__("require_admission_pass", False),
            lambda item: item["admission"].__setitem__("required_gates", ["G0"]),
            lambda item: item.__setitem__("graph_modes", ["cuda_graph"]),
            lambda item: item.__setitem__("runner_kind", "growing_context"),
        ):
            with self.subTest(mutation=mutate):
                mutated = copy.deepcopy(raw)
                mutate(mutated)
                with self.assertRaises(SchemaValidationError):
                    parse_config(mutated)

    def test_missing_required_fingerprint_is_rejected(self) -> None:
        _, raw = _example(document_type="hardware")
        mutated = copy.deepcopy(raw)
        del mutated["e00_manifest_sha256"]
        with self.assertRaises(SchemaValidationError) as caught:
            parse_config(mutated)
        self.assertIn("missing required field", str(caught.exception))
        self.assertIn("e00_manifest_sha256", str(caught.exception))

    def test_schema_version_is_serialized(self) -> None:
        path, _ = _example(document_type="hardware")
        document = load_config(path)
        self.assertEqual(document.to_dict()["schema_version"], "kvbench.hardware.v1")
        self.assertIn(b'"schema_version":"kvbench.hardware.v1"', document.canonical_bytes())

    def test_canonical_serialization_is_deterministic(self) -> None:
        path, raw = _example(document_type="hardware")
        document = load_config(path)
        reordered = {key: raw[key] for key in reversed(tuple(raw))}
        reparsed = parse_config(reordered)
        self.assertEqual(document.canonical_bytes(), reparsed.canonical_bytes())
        self.assertEqual(document.fingerprint(), reparsed.fingerprint())
        self.assertEqual(
            canonical_json_bytes({"z": 1, "a": 2}),
            canonical_json_bytes({"a": 2, "z": 1}),
        )

    def test_duplicate_key_in_json_compatible_yaml_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.yaml"
            path.write_text(
                '{"document_type":"model","document_type":"hardware"}\n',
                encoding="utf-8",
            )
            with self.assertRaises(ConfigLoadError) as caught:
                load_json_compatible_yaml(path)
        self.assertIn("duplicate object key", str(caught.exception))

    def test_nonfinite_number_in_json_compatible_yaml_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nonfinite.yaml"
            path.write_text('{"value": NaN}\n', encoding="utf-8")
            with self.assertRaises(ConfigLoadError):
                load_json_compatible_yaml(path)

    def test_fixture_documents_are_json_compatible_yaml(self) -> None:
        for path in sorted(CONFIG_ROOT.glob("*/*.yaml")):
            raw = path.read_text(encoding="utf-8")
            self.assertIsInstance(json.loads(raw), dict)


if __name__ == "__main__":
    unittest.main()
