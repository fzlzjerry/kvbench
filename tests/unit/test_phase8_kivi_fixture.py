"""Focused immutable Phase 7 fixture compatibility tests for Phase 8."""

from __future__ import annotations

import hashlib
import importlib
import unittest

from kvbench.runtime.kivi_fixture import (
    KIVI_BASE_TREE,
    KIVI_EXTENSION_SHA256,
    KIVI_FIXTURE_CONFIGS,
    KIVI_FIXTURE_ROOT,
    KIVI_OFFICIAL_COMMIT,
    KIVI_PATCHED_TREE,
    load_kivi_fixture,
    tensor_from_record,
    validate_all_kivi_fixtures,
)


class Phase8KIVIFixtureTests(unittest.TestCase):
    def test_all_four_authorities_and_member_checksums(self) -> None:
        fixtures = validate_all_kivi_fixtures()
        self.assertEqual(
            tuple(fixture.config_name for fixture in fixtures),
            KIVI_FIXTURE_CONFIGS,
        )
        for fixture in fixtures:
            with self.subTest(config=fixture.config_name):
                self.assertEqual(
                    fixture.payload["source"]["commit"], KIVI_OFFICIAL_COMMIT
                )
                self.assertEqual(
                    fixture.payload["source"]["base_tree"], KIVI_BASE_TREE
                )
                self.assertEqual(
                    fixture.payload["source"]["patched_tree"], KIVI_PATCHED_TREE
                )
                self.assertEqual(
                    fixture.manifest["extension_sha256"], KIVI_EXTENSION_SHA256
                )
                expected = fixture.manifest["fixture"]["sha256"]
                self.assertEqual(
                    hashlib.sha256(fixture.fixture_path.read_bytes()).hexdigest(),
                    expected,
                )

    def test_tensor_payload_and_input_checksums(self) -> None:
        fixture = load_kivi_fixture("k4v4")
        record = fixture.tensor_record("inputs.key_0_33")
        self.assertIsNotNone(record)
        assert record is not None
        tensor = tensor_from_record(record, device="cpu")
        torch = importlib.import_module("torch")
        self.assertEqual(tuple(tensor.shape), (1, 8, 34, 128))
        self.assertEqual(str(tensor.dtype), "torch.bfloat16")
        self.assertEqual(
            hashlib.sha256(
                bytes(tensor.contiguous().view(torch.uint8).flatten().tolist())
            ).hexdigest(),
            record["payload_sha256"],
        )

    def test_legacy_ratio_is_explicitly_renamed_and_reciprocal(self) -> None:
        for config in KIVI_FIXTURE_CONFIGS:
            with self.subTest(config=config):
                records = load_kivi_fixture(config).legacy_allocation_records()
                self.assertEqual([item["context"] for item in records], [31, 32, 33, 64])
                for item in records:
                    self.assertNotIn("r_alloc", item)
                    self.assertAlmostEqual(
                        item["rho_alloc_legacy"]
                        * item["canonical_r_alloc"],
                        1.0,
                        places=12,
                    )
                    self.assertIsNone(item["r_hbm"])

    def test_reference_files_remain_immutable(self) -> None:
        self.assertTrue((KIVI_FIXTURE_ROOT / "checksums.sha256").is_file())
        fixture = load_kivi_fixture("k2v2")
        self.assertTrue(fixture.fixture_path.is_file())


if __name__ == "__main__":
    unittest.main()
