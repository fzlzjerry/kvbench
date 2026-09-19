from __future__ import annotations

import unittest

import torch

from scripts import q0_batch_diagnosis as diagnosis
from scripts import q0_cache_sensitive_correctness as q0


class Q0BatchDiagnosisTests(unittest.TestCase):
    def test_carrier_mapping_is_explicit_and_row_correct(self) -> None:
        carrier, mapping = diagnosis.build_carrier([11, 22, 33], 8)
        self.assertEqual(carrier, [11, 22, 33, 11, 22, 33, 11, 22])
        self.assertEqual(mapping, {0: [0, 3, 6], 1: [1, 4, 7], 2: [2, 5]})
        for logical, rows in mapping.items():
            self.assertTrue(all(carrier[row] == [11, 22, 33][logical] for row in rows))

    def test_permutation_mapping_restores_logical_rows(self) -> None:
        carrier, mapping = diagnosis.permuted_carrier([11, 22, 33])
        self.assertEqual(carrier, [33, 11, 11, 22])
        self.assertEqual([carrier[mapping[index]] for index in range(3)], [11, 22, 33])

    def test_vector_metrics_uses_frozen_elementwise_predicate(self) -> None:
        reference = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float32)
        observed = torch.tensor([0.019, 1.039, 3.081], dtype=torch.float32)
        result = diagnosis.vector_metrics(reference, observed, atol=0.02, rtol=0.02)
        self.assertFalse(result["passed"])
        self.assertEqual(result["violating_element_count"], 1)
        self.assertEqual(result["element_count"], 3)
        self.assertEqual(result["predicate"], "abs(z_batch-z_b1) <= atol + rtol*abs(z_b1)")

    def test_q0_comparator_scales_relative_tolerance_by_reference(self) -> None:
        reference = torch.tensor([1.0], dtype=torch.float32)
        observed = torch.tensor([1.0405], dtype=torch.float32)
        result = q0.tensor_compare(reference, observed, atol=0.02, rtol=0.02)
        self.assertFalse(result["passed"])
        self.assertTrue(torch.allclose(reference, observed, atol=0.02, rtol=0.02))

    def test_boundary_localization_keeps_first_nonexact_boundary(self) -> None:
        reference = {name: torch.zeros(2) for name in diagnosis.BOUNDARIES}
        observed = {name: value.clone() for name, value in reference.items()}
        observed["layer00_q_proj"][0] = 0.1
        observed["final_norm"][1] = 0.2
        rows, first = diagnosis.boundary_metrics(reference, observed)
        self.assertEqual(first, "layer00_q_proj")
        self.assertTrue(rows[0]["exact"])
        self.assertFalse(next(row for row in rows if row["boundary"] == "layer00_q_proj")["exact"])

    def test_invalid_carrier_geometry_fails_closed(self) -> None:
        with self.assertRaises(diagnosis.DiagnosisError):
            diagnosis.build_carrier([1, 2, 3], 2)

    def test_batch_worker_mount_uses_diagnosis_namespace(self) -> None:
        campaign = diagnosis.ROOT / "artifacts/q0_batch_diagnosis/q0bd-test"
        command = diagnosis._batch_worker_command(campaign, "bf16", "0" * 40)
        mounts = [command[index + 1] for index, value in enumerate(command[:-1]) if value == "--mount"]
        selected = next(value for value in mounts if f"src={campaign}," in value)
        self.assertEqual(
            selected,
            f"type=bind,src={campaign},dst=/home/rockrock/cmu_paper/artifacts/q0_batch_diagnosis/{campaign.name}",
        )


if __name__ == "__main__":
    unittest.main()
