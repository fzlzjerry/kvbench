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

    def test_existing_unit_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = {"status": "PASS", "configuration": "bf16", "stage": "core-l512"}
            q0._finalize_unit(root, "bf16", "core-l512", result)
            original = (root / "units/bf16/core-l512/result.json").read_bytes()
            with self.assertRaisesRegex(q0.Q0Error, "already exists"):
                q0._finalize_unit(root, "bf16", "core-l512", result)
            self.assertEqual((root / "units/bf16/core-l512/result.json").read_bytes(), original)

    def test_approval_receipt_scope_excludes_later_quality_stages(self) -> None:
        receipt = json.loads(q0.APPROVAL.read_text())
        self.assertEqual(receipt["authorization_scope"], ["Q0_CACHE_SENSITIVE_CORRECTNESS"])
        self.assertTrue({"FAST_PPL", "FULL_PPL", "LONGBENCH_SCORING"}.issubset(receipt["not_authorized"]))


if __name__ == "__main__":
    unittest.main()
