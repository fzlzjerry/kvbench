from __future__ import annotations

import unittest

from kvbench.runtime.gqa_taxonomy import (
    classify_gqa_evidence,
    gqa_failure_status,
)
from kvbench.schema import GQAVerdict, RunStatus


class GQATaxonomyTests(unittest.TestCase):
    def classify(self, **overrides: bool) -> GQAVerdict:
        evidence = {
            "materialization_evidence": False,
            "dispatch_verified": True,
            "no_replication_kernel_verified": True,
            "allocation_verified": True,
            "source_verified": True,
            "shape_verified": True,
        }
        evidence.update(overrides)
        return classify_gqa_evidence(**evidence)

    def test_positive_evidence_always_means_materialization_detected(self) -> None:
        verdict = self.classify(
            materialization_evidence=True,
            dispatch_verified=False,
            allocation_verified=False,
        )
        self.assertEqual(verdict, GQAVerdict.MATERIALIZATION_DETECTED)
        self.assertEqual(
            gqa_failure_status(verdict),
            RunStatus.GQA_MATERIALIZATION_DETECTED,
        )

    def test_missing_dispatch_proof_is_not_materialization(self) -> None:
        verdict = self.classify(dispatch_verified=False)
        self.assertEqual(verdict, GQAVerdict.DISPATCH_UNVERIFIED)
        self.assertEqual(
            gqa_failure_status(verdict), RunStatus.GQA_DISPATCH_UNVERIFIED
        )

    def test_incomplete_nonmaterialization_proof_is_unproven(self) -> None:
        for field in (
            "no_replication_kernel_verified",
            "allocation_verified",
            "source_verified",
            "shape_verified",
        ):
            with self.subTest(field=field):
                verdict = self.classify(**{field: False})
                self.assertEqual(
                    verdict, GQAVerdict.NONMATERIALIZATION_UNPROVEN
                )
                self.assertEqual(
                    gqa_failure_status(verdict),
                    RunStatus.GQA_NONMATERIALIZATION_UNPROVEN,
                )

    def test_complete_proof_is_verified_and_has_no_failure_status(self) -> None:
        verdict = self.classify()
        self.assertEqual(verdict, GQAVerdict.NONMATERIALIZATION_VERIFIED)
        self.assertIsNone(gqa_failure_status(verdict))


if __name__ == "__main__":
    unittest.main()
