"""Focused lifecycle and safe-format tests for Phase 9 calibration."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from kvbench.errors import ArtifactConflictError, ArtifactStateError
from kvbench.runtime.artifacts import (
    phase9_calibration_artifact_store,
    validate_run_directory,
)
from kvbench.schema import Phase9CalibrationManifest, RunStatus
from scripts import phase9_kvquant_calibration as host
from scripts import phase9_kvquant_worker as worker
from tests.schema.test_phase9_schema import phase9_manifest


def created_manifest(run_id: str) -> Phase9CalibrationManifest:
    payload = phase9_manifest()
    payload["run_id"] = run_id
    return Phase9CalibrationManifest.from_dict(payload)


def required_completed_payloads(run: object) -> None:
    paths = (
        "authority_manifest.json",
        "calibration_config.json",
        "dataset_manifest.json",
        "environment.json",
        "fisher/fisher.safetensors",
        "fisher_manifest.json",
        "inventory.json",
        "layer_stats.parquet",
        "model_manifest.json",
        "outlier_policy.json",
        "quantizers/kvq2.safetensors",
        "quantizers/kvq3.safetensors",
        "quantizers/kvq4.safetensors",
        "tokenizer_manifest.json",
        "tokens/input_ids.safetensors",
    )
    for relative in paths:
        run.write_bytes(relative, b"phase9-test")


class Phase9ArtifactLifecycleTests(unittest.TestCase):
    def test_completed_bundle_is_no_replace_complete_last_and_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "docs/evidence").mkdir(parents=True)
            store = phase9_calibration_artifact_store(root)
            run_id = "kvqcal-" + "1" * 32
            initial = created_manifest(run_id)
            run = store.create(run_id, initial)
            run.start()
            required_completed_payloads(run)
            completed = dataclasses.replace(
                initial,
                status=RunStatus.COMPLETED,
                started_at_utc="2026-07-28T08:00:01Z",
                finished_at_utc="2026-07-28T09:00:00Z",
                inventory_path="artifact_inventory.json",
            )
            final = run.finalize(completed)
            validation = validate_run_directory(final)
            self.assertTrue(validation.valid)
            self.assertTrue(validation.complete)
            self.assertTrue((final / "COMPLETE").is_file())
            self.assertFalse(final.stat().st_mode & 0o222)
            with self.assertRaises(ArtifactConflictError):
                store.create(run_id, initial)

    def test_completed_bundle_rejects_missing_required_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "docs/evidence").mkdir(parents=True)
            run_id = "kvqcal-" + "2" * 32
            initial = created_manifest(run_id)
            run = phase9_calibration_artifact_store(root).create(
                run_id,
                initial,
            )
            run.start()
            completed = dataclasses.replace(
                initial,
                status=RunStatus.COMPLETED,
                started_at_utc="2026-07-28T08:00:01Z",
                finished_at_utc="2026-07-28T09:00:00Z",
                inventory_path="artifact_inventory.json",
            )
            with self.assertRaises(ArtifactStateError):
                run.finalize(completed)

    def test_failed_attempt_can_finalize_without_partial_quantizers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "docs/evidence").mkdir(parents=True)
            run_id = "kvqcal-" + "3" * 32
            initial = created_manifest(run_id)
            run = phase9_calibration_artifact_store(root).create(
                run_id,
                initial,
            )
            run.start()
            run.write_json(
                "failure.json",
                {"stage": "full-fisher", "partial_fisher_used": False},
            )
            failed = dataclasses.replace(
                initial,
                status=RunStatus.RUNTIME_FAILED,
                started_at_utc="2026-07-28T08:00:01Z",
                finished_at_utc="2026-07-28T08:10:00Z",
                inventory_path="artifact_inventory.json",
                failure_reason="full Fisher failed",
            )
            final = run.finalize(failed)
            validation = validate_run_directory(final)
            self.assertTrue(validation.valid)
            self.assertEqual(validation.status, "runtime_failed")


class Phase9WorkerTests(unittest.TestCase):
    def test_trusted_pickle_is_converted_once_to_explicit_safe_tensors(self) -> None:
        provenance = {
            "bit_width": 2,
            "method_identifier": worker.METHOD_IDENTIFIER,
            "patch_digest": worker.PATCH_SHA256,
            "patched_commit": worker.PATCHED_COMMIT,
            "patched_tree": worker.PATCHED_TREE,
            "nuq": True,
            "dense_and_sparse": True,
            "key_outlier_cap": 12,
            "value_outlier_cap": 12,
            "sink_tokens": 5,
            "fisher_root": "f" * 64,
            "dataset_root": "d" * 64,
            "token_tensor_sha256": "t" * 64,
        }
        key = (
            torch.ones((1, 4), dtype=torch.float32),
            -torch.ones((1, 4), dtype=torch.float32),
            [np.zeros((4, 1), dtype=np.float32)],
        )
        value = (
            torch.ones((2, 1), dtype=torch.float32),
            -torch.ones((2, 1), dtype=torch.float32),
            [np.zeros((4, 1), dtype=np.float32)],
        )
        payload = {
            "format": "kvquant_gqa_quantizer_v1",
            "provenance": provenance,
            "quantizers": {
                "model.layers.0.self_attn.k_proj": key,
                "model.layers.0.self_attn.v_proj": value,
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native = root / "native.pickle"
            safe = root / "kvq2.safetensors"
            manifest = root / "kvq2.manifest.json"
            native.write_bytes(
                pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
            )
            native_sha = worker.sha256_file(native)
            with patch.multiple(
                worker,
                LAYERS=1,
                NUM_EXAMPLES=1,
                SEQUENCE_LENGTH=2,
                KV_WIDTH=4,
            ):
                worker._trusted_convert_quantizer(
                    native,
                    safe,
                    manifest,
                    2,
                )
            self.assertFalse(native.exists())
            record = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(
                record["source_native_pickle_sha256"],
                native_sha,
            )
            self.assertFalse(record["source_native_pickle_published"])
            self.assertEqual(record["tensor_count"], 10)
            with safe_open(safe, framework="pt", device="cpu") as handle:
                self.assertEqual(len(handle.keys()), 10)
                self.assertEqual(
                    handle.metadata()["format"],
                    "kvbench-kvquant-safe-v1",
                )

    def test_safe_quantizer_comparison_uses_frozen_numerical_fallback(
        self,
    ) -> None:
        metadata = {
            "format": "kvbench-kvquant-safe-v1",
            "bit_width": "4",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "original.safetensors"
            exact = root / "exact.safetensors"
            near = root / "near.safetensors"
            far = root / "far.safetensors"
            values = torch.tensor([1.0, -2.0], dtype=torch.float32)
            save_file({"values": values}, original, metadata=metadata)
            save_file({"values": values.clone()}, exact, metadata=metadata)
            save_file(
                {"values": values + torch.tensor([1e-6, 0.0])},
                near,
                metadata=metadata,
            )
            save_file(
                {"values": values + torch.tensor([1e-2, 0.0])},
                far,
                metadata=metadata,
            )

            exact_result = worker._compare_safe_quantizers(original, exact)
            near_result = worker._compare_safe_quantizers(original, near)
            far_result = worker._compare_safe_quantizers(original, far)

        self.assertEqual(exact_result["status"], "PASS")
        self.assertTrue(exact_result["all_tensors_exact"])
        self.assertTrue(exact_result["all_tensors_numerically_equivalent"])
        self.assertEqual(near_result["status"], "PASS")
        self.assertFalse(near_result["all_tensors_exact"])
        self.assertTrue(near_result["all_tensors_numerically_equivalent"])
        self.assertEqual(near_result["rtol"], worker.REPLAY_RTOL)
        self.assertEqual(near_result["atol"], worker.REPLAY_ATOL)
        self.assertEqual(far_result["status"], "FAIL")
        self.assertFalse(far_result["all_tensors_numerically_equivalent"])

    def test_contract_and_docker_invocation_are_single_lane_and_secret_free(self) -> None:
        first = host.calibration_identity()
        second = host.calibration_identity()
        self.assertEqual(first, second)
        self.assertRegex(first[0], r"\Akvqcal-[0-9a-f]{32}\Z")
        self.assertEqual(first[2]["quantizers"], ["kvq4", "kvq3", "kvq2"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entry = {
                "source_root": root,
                "model_cache": root,
                "dataset": root / "train.parquet",
            }
            command = host._docker_command(
                entry,
                root,
                ["policy-check", "--output", "/output/policy.json"],
            )
        rendered = " ".join(str(item) for item in command)
        for forbidden in (
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "CLOUDFLARE_API_TOKEN",
            "R2_ACCOUNT_ID",
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "dst=/workspace",
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertIn("--network=none", rendered)
        self.assertIn(host.CALIBRATION_IMAGE_DIGEST, command)

    def test_entry_document_status_accepts_exact_frozen_markdown_forms(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            decision = root / "decision.md"
            report = root / "report.md"
            rejected = root / "rejected.md"
            decision.write_text("- Status: Accepted\n", encoding="utf-8")
            report.write_text("Status: **PASS**\n", encoding="utf-8")
            rejected.write_text("Status: PARTIAL\n", encoding="utf-8")
            self.assertTrue(
                host._document_has_status(decision, "Accepted")
            )
            self.assertTrue(host._document_has_status(report, "PASS"))
            self.assertFalse(host._document_has_status(rejected, "PASS"))

    def test_source_validation_requires_exact_structured_authority(
        self,
    ) -> None:
        manifest = host._load_json(host.PATCH_MANIFEST_PATH)
        paths = [record["path"] for record in manifest["patched_files"]]
        validation = {
            "status": "PASS",
            "changed_paths": paths,
            "patch_path": manifest["patch"]["path"],
            "patch_sha256": host.PATCH_SHA256,
            "patch_size_bytes": host.PATCH_SIZE_BYTES,
            "reconstruction": {
                "base_commit": host.UPSTREAM_BASE_COMMIT,
                "base_tree": host.UPSTREAM_BASE_TREE,
                "applied_patch_sha256": host.PATCH_SHA256,
                "changed_file_count": len(paths),
                "patched_tree": host.PATCHED_TREE,
            },
        }
        self.assertTrue(host._source_validation_is_exact(validation))

        wrong_tree = json.loads(json.dumps(validation))
        wrong_tree["reconstruction"]["patched_tree"] = "0" * 40
        self.assertFalse(host._source_validation_is_exact(wrong_tree))

        unexpected = json.loads(json.dumps(validation))
        unexpected["extra"] = "not-accepted"
        self.assertFalse(host._source_validation_is_exact(unexpected))

    def test_tokenizer_default_behavior_uses_actual_encode_path(
        self,
    ) -> None:
        class ExactTokenizer:
            bos_token_id = 128000

            def build_inputs_with_special_tokens(
                self,
                _token_ids: list[int],
            ) -> list[int]:
                return []

            def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
                self.assert_call(text, kwargs)
                return {"input_ids": [self.bos_token_id]}

            @staticmethod
            def assert_call(text: str, kwargs: dict[str, object]) -> None:
                if text != "" or kwargs != {
                    "add_special_tokens": True,
                    "truncation": False,
                    "padding": False,
                    "return_attention_mask": False,
                }:
                    raise AssertionError("unexpected tokenizer call")

        exact = ExactTokenizer()
        self.assertEqual(exact.build_inputs_with_special_tokens([]), [])
        self.assertEqual(
            worker._default_special_token_ids(exact),
            [exact.bos_token_id],
        )

    def test_worker_exposes_no_general_campaign_or_quality_command(self) -> None:
        parser = worker._parser()
        subparsers = next(
            action
            for action in parser._actions
            if action.dest == "command"
        )
        self.assertEqual(
            set(subparsers.choices),
            {
                "freeze-dataset",
                "run-fisher",
                "fisher-manifest",
                "run-quantizer",
                "compare-quantizers",
                "layer-stats",
                "reconstruct-tokens",
                "replay-fisher",
                "policy-check",
                "validate-payloads",
            },
        )
        source = Path(worker.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "LongBench",
            "perplexity",
            "make pilot",
            "make full-scan",
            "make profile-subset",
            "make fit",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
