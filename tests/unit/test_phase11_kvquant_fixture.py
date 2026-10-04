"""Focused read-only tests for the corrected KVQuant fixture oracle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from kvbench.runtime.kvquant_fixture import (
    KVQUANT_AGGREGATE_PATCH_SHA256,
    KVQUANT_CALIBRATION_ID,
    KVQUANT_CALIBRATION_ROOT_SHA256,
    KVQUANT_CASES,
    KVQUANT_CORRECTED_COMMIT,
    KVQUANT_CORRECTED_TREE,
    KVQUANT_DECODE_ATOL,
    KVQUANT_DECODE_RTOL,
    KVQUANT_EXECUTION_SOURCE_IDENTIFIER,
    KVQUANT_FAMILIES,
    KVQUANT_FIXTURE_ID,
    KVQUANT_FIXTURE_ROOT,
    KVQUANT_FIXTURE_ROOT_SHA256,
    KVQUANT_HISTORICAL_FIXTURE_ID,
    KVQUANT_HISTORICAL_FIXTURE_ROOT,
    KVQUANT_HISTORICAL_ROOT_SHA256,
    KVQUANT_KEY_LOGITS_ATOL,
    KVQUANT_KEY_LOGITS_RTOL,
    KVQUANT_METHOD_IDENTIFIER,
    KVQUANT_TENSOR_FILES,
    KVQUANT_UPSTREAM_BASE_COMMIT,
    KVQUANT_UPSTREAM_BASE_TREE,
    KVQuantFixtureError,
    compare_decode_output_untimed,
    compare_exact_fixture_tensor_untimed,
    compare_key_logits_untimed,
    load_all_kvquant_fixtures,
    load_fixture_tensor_file_untimed,
    load_kvquant_fixture,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class Phase11KVQuantFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.historical_before = _file_hashes(KVQUANT_HISTORICAL_FIXTURE_ROOT)
        cls.corrected_before = _file_hashes(KVQUANT_FIXTURE_ROOT)
        cls.fixtures = load_all_kvquant_fixtures()
        cls.historical_after = _file_hashes(KVQUANT_HISTORICAL_FIXTURE_ROOT)
        cls.corrected_after = _file_hashes(KVQUANT_FIXTURE_ROOT)

    def test_exact_nine_case_authority_and_tensor_headers(self) -> None:
        self.assertEqual(
            [(item.family, item.case_name) for item in self.fixtures],
            [
                (family, case_name)
                for family in KVQUANT_FAMILIES
                for case_name in KVQUANT_CASES
            ],
        )
        self.assertEqual(len(self.fixtures), 9)
        for fixture in self.fixtures:
            with self.subTest(
                family=fixture.family,
                case_name=fixture.case_name,
            ):
                authority = fixture.authority
                self.assertEqual(authority.fixture_id, KVQUANT_FIXTURE_ID)
                self.assertEqual(
                    authority.root_sha256,
                    KVQUANT_FIXTURE_ROOT_SHA256,
                )
                self.assertEqual(
                    authority.historical_root_sha256,
                    KVQUANT_HISTORICAL_ROOT_SHA256,
                )
                self.assertEqual(
                    authority.calibration_id,
                    KVQUANT_CALIBRATION_ID,
                )
                self.assertEqual(
                    authority.calibration_root_sha256,
                    KVQUANT_CALIBRATION_ROOT_SHA256,
                )
                self.assertEqual(
                    authority.execution_source_identifier,
                    KVQUANT_EXECUTION_SOURCE_IDENTIFIER,
                )
                self.assertEqual(
                    authority.method_identifier,
                    KVQUANT_METHOD_IDENTIFIER,
                )
                self.assertEqual(
                    authority.upstream_base_commit,
                    KVQUANT_UPSTREAM_BASE_COMMIT,
                )
                self.assertEqual(
                    authority.upstream_base_tree,
                    KVQUANT_UPSTREAM_BASE_TREE,
                )
                self.assertEqual(
                    authority.aggregate_patch_sha256,
                    KVQUANT_AGGREGATE_PATCH_SHA256,
                )
                self.assertEqual(
                    authority.corrected_commit,
                    KVQUANT_CORRECTED_COMMIT,
                )
                self.assertEqual(
                    authority.corrected_tree,
                    KVQUANT_CORRECTED_TREE,
                )
                self.assertEqual(
                    set(fixture.tensor_records),
                    set(KVQUANT_TENSOR_FILES),
                )
                for file_records in fixture.tensor_records.values():
                    self.assertTrue(file_records)
                    for record in file_records.values():
                        self.assertTrue(record.name)
                        self.assertIn(
                            record.dtype,
                            {
                                "bool",
                                "bfloat16",
                                "float16",
                                "float32",
                                "float64",
                                "int32",
                                "int64",
                            },
                        )
                        self.assertTrue(all(value >= 0 for value in record.shape))
                        self.assertGreater(record.logical_nbytes, 0)
                        self.assertEqual(len(record.payload_sha256), 64)

    def test_mixed_provenance_is_explicit(self) -> None:
        for fixture in self.fixtures:
            with self.subTest(
                family=fixture.family,
                case_name=fixture.case_name,
            ):
                source = fixture.manifest["source"]
                if fixture.family == "kvq3":
                    self.assertEqual(fixture.fixture_id, KVQUANT_FIXTURE_ID)
                    self.assertEqual(source["decision"], "0025")
                    self.assertEqual(
                        source["patch_sha256"],
                        KVQUANT_AGGREGATE_PATCH_SHA256,
                    )
                    self.assertEqual(
                        source["patched_commit"],
                        KVQUANT_CORRECTED_COMMIT,
                    )
                    self.assertEqual(
                        source["patched_tree"],
                        KVQUANT_CORRECTED_TREE,
                    )
                else:
                    self.assertEqual(
                        fixture.fixture_id,
                        KVQUANT_HISTORICAL_FIXTURE_ID,
                    )
                    self.assertEqual(source["decision"], "0021")

    def test_safe_tensor_load_and_untimed_comparisons(self) -> None:
        fixture = next(
            value
            for value in self.fixtures
            if value.family == "kvq3"
            and value.case_name == "key_few_value_fixed12"
        )
        decode = load_fixture_tensor_file_untimed(
            fixture,
            "decode_output.safetensors",
        )
        exact = compare_exact_fixture_tensor_untimed(
            fixture,
            "decode_output.safetensors",
            "source_decode_output",
            decode["source_decode_output"],
        )
        self.assertTrue(exact.passed)

        decode_comparison = compare_decode_output_untimed(
            fixture,
            decode["source_decode_output"],
        )
        self.assertTrue(decode_comparison.passed)
        self.assertEqual(decode_comparison.atol, KVQUANT_DECODE_ATOL)
        self.assertEqual(decode_comparison.rtol, KVQUANT_DECODE_RTOL)

        key_comparison = compare_key_logits_untimed(
            fixture,
            decode["source_nonsink_key_logits"],
        )
        self.assertTrue(key_comparison.passed)
        self.assertEqual(key_comparison.atol, KVQUANT_KEY_LOGITS_ATOL)
        self.assertEqual(key_comparison.rtol, KVQUANT_KEY_LOGITS_RTOL)

    def test_historical_and_reused_fixture_bytes_remain_untouched(self) -> None:
        self.assertEqual(self.historical_after, self.historical_before)
        self.assertEqual(self.corrected_after, self.corrected_before)
        for family in ("kvq4", "kvq2"):
            old_family = KVQUANT_HISTORICAL_FIXTURE_ROOT / family
            corrected_family = KVQUANT_FIXTURE_ROOT / family
            self.assertEqual(
                _file_hashes(old_family),
                _file_hashes(corrected_family),
            )

    def test_root_or_member_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied_root = Path(temporary) / "fixtures"
            shutil.copytree(KVQUANT_FIXTURE_ROOT, copied_root)
            manifest_path = copied_root / "manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["decision"] = "untrusted"
            manifest_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(KVQuantFixtureError):
                load_all_kvquant_fixtures(copied_root)

    def test_unsupported_fixture_fails_closed(self) -> None:
        with self.assertRaises(KVQuantFixtureError):
            load_kvquant_fixture("kvq5", "key_zero_value_fixed12")
        with self.assertRaises(KVQuantFixtureError):
            load_kvquant_fixture("kvq4", "no_outlier")


if __name__ == "__main__":
    unittest.main()
