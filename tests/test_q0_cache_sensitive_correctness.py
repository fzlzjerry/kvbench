from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts import q0_cache_sensitive_correctness as q0


class Q0ContractTests(unittest.TestCase):
    def test_approval_binds_exact_frozen_contract_without_mutation(self) -> None:
        before = q0.CONTRACT.read_bytes()
        receipt = q0.verify_approval()
        self.assertEqual(receipt["contract_id"], q0.CONTRACT_ID)
        self.assertEqual(receipt["contract_sha256"], q0.CONTRACT_SHA256)
        self.assertEqual(hashlib.sha256(before).hexdigest(), q0.CONTRACT_SHA256)
        self.assertEqual(q0.CONTRACT.read_bytes(), before)

    def test_frozen_input_plan_has_exact_bounded_semantics(self) -> None:
        plan = q0.frozen_input_plan()
        self.assertEqual([item["length"] for item in plan["core_probes"]], list(q0.CORE_LENGTHS))
        self.assertTrue(all(len(item["target_tokens"]) == q0.TEACHER_TARGETS for item in plan["core_probes"]))
        self.assertEqual(len(plan["longbench"]["conditioning_ids"]), 16)
        self.assertEqual(len(plan["invariance"]["tokens"]), 100)
        self.assertEqual(plan["invariance"]["batch_sizes"], [1, 4, 8])

    def test_compact_plan_contains_no_bulk_token_payload(self) -> None:
        compact = q0.compact_input_plan(q0.frozen_input_plan())
        self.assertTrue(all("input_ids" not in item for item in compact["core_probes"]))
        self.assertNotIn("prefill_ids", compact["longbench"])
        self.assertNotIn("conditioning_ids", compact["longbench"])

    def test_locked_hot_paths_still_match_freeze(self) -> None:
        result = q0.verify_locked_paths()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["locked_path_count"], 68)

    def test_unit_topology_is_exact(self) -> None:
        self.assertEqual(q0.EXPECTED_UNIT_COUNT, 110)
        self.assertEqual(len(q0.UNIT_STAGES), 11)
        self.assertEqual(q0.UNIT_STAGES[-4:], (
            "longbench-suffix",
            "graph-invariance",
            "batch-invariance",
            "cache-dependence",
        ))

    def test_kvquant_configuration_family_routing(self) -> None:
        self.assertEqual(q0.family("kvq4"), "kvquant")
        self.assertEqual(q0.family("kvq3"), "kvquant")
        self.assertEqual(q0.family("kvq2"), "kvquant")
        self.assertEqual(q0.family("k4v4"), "kivi")

    def test_existing_unit_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = {"status": "PASS", "configuration": "bf16", "stage": "core-l512"}
            q0._finalize_unit(root, "bf16", "core-l512", result)
            original = (root / "units/bf16/core-l512/result.json").read_bytes()
            with self.assertRaisesRegex(q0.Q0Error, "already exists"):
                q0._finalize_unit(root, "bf16", "core-l512", result)
            self.assertEqual((root / "units/bf16/core-l512/result.json").read_bytes(), original)

    def test_exact_kvquant_harness_failure_is_replaced_append_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            failed = {
                "status": "FAIL",
                "configuration": "kvq4",
                "stage": "core-l512",
                "error": "config_load_error: KIVI requires one explicit frozen configuration",
            }
            q0._finalize_unit(root, "kvq4", "core-l512", failed)
            original_path = root / "units/kvq4/core-l512/result.json"
            original = original_path.read_bytes()
            passed = {"status": "PASS", "configuration": "kvq4", "stage": "core-l512"}
            q0._finalize_unit(root, "kvq4", "core-l512", passed)
            replacement_path = root / "continuations/kvquant-family-routing-fix/units/kvq4/core-l512/result.json"
            replacement = json.loads(replacement_path.read_text())
            self.assertEqual(original_path.read_bytes(), original)
            self.assertEqual(replacement["replacement_of"], "units/kvq4/core-l512")
            self.assertTrue(replacement["original_failure_preserved"])
            self.assertTrue(q0._unit_complete(root, "kvq4", "core-l512"))

    def test_docker_worker_uses_absolute_campaign_mount(self) -> None:
        campaign = q0.ROOT / "artifacts/q0/q0-test-absolute-path"
        command = q0._docker_worker_command(campaign, "bf16", "0" * 40)
        mounts = [command[index + 1] for index, value in enumerate(command[:-1]) if value == "--mount"]
        self.assertTrue(any(f"src={campaign}" in mount for mount in mounts))
        self.assertTrue(all("src=artifacts/" not in mount for mount in mounts))

    def test_approval_receipt_scope_excludes_later_quality_stages(self) -> None:
        receipt = json.loads(q0.APPROVAL.read_text())
        self.assertEqual(receipt["authorization_scope"], ["Q0_CACHE_SENSITIVE_CORRECTNESS"])
        self.assertTrue({"FAST_PPL", "FULL_PPL", "LONGBENCH_SCORING"}.issubset(receipt["not_authorized"]))


if __name__ == "__main__":
    unittest.main()
