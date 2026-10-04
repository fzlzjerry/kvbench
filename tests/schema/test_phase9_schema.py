"""Strict schema tests for the single Phase 9 calibration contract."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from kvbench.errors import SchemaValidationError
from kvbench.schema import (
    KVQuantCalibrationReference,
    MethodConfig,
    Phase9CalibrationManifest,
    RunStatus,
)
from kvbench.schema.phase9 import (
    CALIBRATION_DOCKERFILE_SHA256,
    CALIBRATION_IMAGE_DIGEST,
    DATASET_CONTENT_SHA256,
    DATASET_CONVERSION_REVISION,
    DATASET_REPOSITORY,
    DATASET_REVISION,
    METHOD_IDENTIFIER,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SNAPSHOT_MANIFEST_SHA256,
    PATCH_SHA256,
    PATCHED_COMMIT,
    PATCHED_TREE,
    UPSTREAM_BASE_COMMIT,
    UPSTREAM_BASE_TREE,
    UPSTREAM_REPOSITORY,
)


ROOT = Path(__file__).resolve().parents[2]
ROOT_DIGEST = "a" * 64
KVQ4_DIGEST = "4" * 64
KVQ3_DIGEST = "3" * 64
KVQ2_DIGEST = "2" * 64
CALIBRATION_ID = "kvqcal-" + "b" * 32


def calibration_reference() -> dict[str, object]:
    return {
        "method_identifier": METHOD_IDENTIFIER,
        "calibration_id": CALIBRATION_ID,
        "local_bundle_path": f"calibration/kvquant/{CALIBRATION_ID}",
        "local_compact_manifest_path": (
            "docs/evidence/phase9/calibration-manifest.json"
        ),
        "source_base_commit": UPSTREAM_BASE_COMMIT,
        "source_base_tree": UPSTREAM_BASE_TREE,
        "patch_sha256": PATCH_SHA256,
        "patched_commit": PATCHED_COMMIT,
        "patched_tree": PATCHED_TREE,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "dataset_conversion_revision": DATASET_CONVERSION_REVISION,
        "dataset_root": "c" * 64,
        "token_tensor_root": "d" * 64,
        "fisher_root": "e" * 64,
        "kvq4_sha256": KVQ4_DIGEST,
        "kvq3_sha256": KVQ3_DIGEST,
        "kvq2_sha256": KVQ2_DIGEST,
        "sink_tokens": 5,
        "key_outlier_cap": 12,
        "value_outlier_cap": 12,
        "outlier_value_dtype": "float32",
        "outlier_index_dtype": "int32",
        "metadata_dtype": "float32",
        "durable_r2_uri": (
            "r2://kvbench-artifacts/kvbench/sha256/"
            f"{ROOT_DIGEST}/"
        ),
        "calibration_root_digest": ROOT_DIGEST,
    }


def phase9_manifest() -> dict[str, object]:
    return {
        "schema_version": Phase9CalibrationManifest.SCHEMA_VERSION,
        "artifact_schema_version": (
            Phase9CalibrationManifest.ARTIFACT_SCHEMA_VERSION
        ),
        "run_id": CALIBRATION_ID,
        "status": "created",
        "created_at_utc": "2026-07-28T08:00:00Z",
        "started_at_utc": None,
        "finished_at_utc": None,
        "phase": "phase9_kvquant_calibration",
        "run_kind": "offline_calibration",
        "claim_class": "none",
        "git_sha": "f" * 40,
        "git_dirty": False,
        "container_digest": CALIBRATION_IMAGE_DIGEST,
        "dockerfile_sha256": CALIBRATION_DOCKERFILE_SHA256,
        "method_identifier": METHOD_IDENTIFIER,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_base_commit": UPSTREAM_BASE_COMMIT,
        "upstream_base_tree": UPSTREAM_BASE_TREE,
        "patch_sha256": PATCH_SHA256,
        "patched_commit": PATCHED_COMMIT,
        "patched_tree": PATCHED_TREE,
        "decision": "0021",
        "source_reconstruction_command": (
            "make KVQUANT_GQA_SOURCE_ROOT=/private/kvquant "
            "validate-kvquant-gqa-patch"
        ),
        "source_reconstruction_passed": True,
        "reconstructed_source_checksum_result": "PASS",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_id": MODEL_ID,
        "tokenizer_revision": MODEL_REVISION,
        "model_snapshot_manifest_sha256": (
            MODEL_SNAPSHOT_MANIFEST_SHA256
        ),
        "dataset_repository": DATASET_REPOSITORY,
        "dataset_revision": DATASET_REVISION,
        "dataset_conversion_revision": DATASET_CONVERSION_REVISION,
        "dataset_content_sha256": DATASET_CONTENT_SHA256,
        "dataset_split": "train",
        "number_of_examples": 16,
        "sequence_length": 2048,
        "random_seed": 20260721,
        "attempt_sequence": 1,
        "calibration_contract_sha256": "1" * 64,
        "command_argv": ["make", "calibrate-kvquant"],
        "performance_measurement": False,
        "profiler_execution": False,
        "quality_evaluation": False,
        "quality_execution": "locked",
        "performance_data_frozen": False,
        "full_scan_state": "closed",
        "phase10_started": False,
        "inventory_path": None,
        "failure_reason": None,
    }


class Phase9SchemaTests(unittest.TestCase):
    def test_exact_created_and_completed_lifecycle_parse(self) -> None:
        created = Phase9CalibrationManifest.from_dict(phase9_manifest())
        self.assertIs(created.status, RunStatus.CREATED)

        completed_payload = phase9_manifest()
        completed_payload.update(
            {
                "status": "completed",
                "started_at_utc": "2026-07-28T08:00:01Z",
                "finished_at_utc": "2026-07-28T09:00:00Z",
                "inventory_path": "artifact_inventory.json",
            }
        )
        completed = Phase9CalibrationManifest.from_dict(completed_payload)
        self.assertIs(completed.status, RunStatus.COMPLETED)

    def test_manifest_rejects_identity_and_boundary_drift(self) -> None:
        mutations = {
            "container_digest": "sha256:" + "0" * 64,
            "patch_sha256": "0" * 64,
            "patched_tree": "0" * 40,
            "model_revision": "0" * 40,
            "dataset_content_sha256": "0" * 64,
            "dataset_split": "test",
            "number_of_examples": 15,
            "sequence_length": 1024,
            "random_seed": 1,
            "performance_measurement": True,
            "profiler_execution": True,
            "quality_evaluation": True,
            "phase10_started": True,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                payload = phase9_manifest()
                payload[field] = value
                with self.assertRaises(SchemaValidationError):
                    Phase9CalibrationManifest.from_dict(payload)

    def test_calibration_reference_is_exact_and_root_bound(self) -> None:
        parsed = KVQuantCalibrationReference.from_dict(
            calibration_reference()
        )
        self.assertEqual(parsed.calibration_root_digest, ROOT_DIGEST)
        for field, value in (
            ("method_identifier", "kvquant"),
            ("source_base_commit", "0" * 40),
            ("patch_sha256", "0" * 64),
            ("sink_tokens", 4),
            ("key_outlier_cap", 13),
            ("outlier_index_dtype", "int64"),
            (
                "durable_r2_uri",
                "r2://kvbench-artifacts/kvbench/sha256/" + "0" * 64 + "/",
            ),
        ):
            with self.subTest(field=field):
                payload = calibration_reference()
                payload[field] = value
                with self.assertRaises(SchemaValidationError):
                    KVQuantCalibrationReference.from_dict(payload)

    def test_legacy_method_documents_serialize_without_new_null_field(self) -> None:
        for name in ("bf16", "turboquant", "kivi", "kvquant"):
            with self.subTest(name=name):
                path = ROOT / "configs/methods" / f"{name}.yaml"
                payload = json.loads(path.read_text(encoding="utf-8"))
                parsed = MethodConfig.from_dict(payload)
                self.assertEqual(parsed.to_dict(), payload)

    def test_kvquant_config_links_each_variant_to_its_safe_artifact(self) -> None:
        path = ROOT / "configs/methods/kvquant.yaml"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["calibration"] = calibration_reference()
        checksums = {
            "kvq4": KVQ4_DIGEST,
            "kvq3": KVQ3_DIGEST,
            "kvq2": KVQ2_DIGEST,
        }
        for variant in payload["variants"]:
            parameters = variant["parameters"]
            parameters.update(
                {
                    "outlier_cap": 12,
                    "calibration_artifact_sha256": checksums[
                        variant["variant_id"]
                    ],
                    "sparse_index_dtype": "int32",
                    "lut_scale_dtype": "float32",
                }
            )
        parsed = MethodConfig.from_dict(payload)
        self.assertIsNotNone(parsed.calibration)

        payload["variants"][0]["parameters"][
            "calibration_artifact_sha256"
        ] = "0" * 64
        with self.assertRaises(SchemaValidationError):
            MethodConfig.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
