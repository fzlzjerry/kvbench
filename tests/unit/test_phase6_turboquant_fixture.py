"""CPU-only tests for frozen TurboQuant fixture conformance."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest

from kvbench.runtime.turboquant_fixture import (
    DECODE_ATOL,
    DECODE_RTOL,
    DEFAULT_FIXTURE_ROOT,
    ENVIRONMENT_MANIFEST_SHA256,
    MANDATORY_CONFIGURATIONS,
    SOURCE_MANIFEST_SHA256,
    TurboQuantFixtureError,
    compare_decode_output_untimed,
    compare_packed_output_untimed,
    compare_slot_layout,
    load_inputs_cpu,
    load_mandatory_fixtures,
    load_tensor_cpu,
    load_turboquant_fixture,
)
from kvbench.schema.phase6 import (
    FIXTURE_ROOT_LEDGER_SHA256,
    FIXTURE_SET_SHA256,
    MANDATORY_CONFIG_SLOT_SIZES,
    PINNED_SOURCE_COMMIT,
    PINNED_SOURCE_TREE,
)


class Phase6TurboQuantFixtureTests(unittest.TestCase):
    def test_mandatory_authority_inputs_and_slots_are_exact(self) -> None:
        fixtures = load_mandatory_fixtures()
        self.assertEqual(
            tuple(item.configuration for item in fixtures),
            MANDATORY_CONFIGURATIONS,
        )
        for fixture in fixtures:
            with self.subTest(configuration=fixture.configuration):
                self.assertEqual(
                    fixture.slot_size,
                    MANDATORY_CONFIG_SLOT_SIZES[fixture.configuration],
                )
                self.assertEqual(
                    sum(fixture.byte_breakdown_per_head_token.values()),
                    fixture.slot_size,
                )
                authority = fixture.authority
                self.assertEqual(
                    authority.fixture_set_sha256,
                    FIXTURE_SET_SHA256,
                )
                self.assertEqual(
                    authority.root_ledger_sha256,
                    FIXTURE_ROOT_LEDGER_SHA256,
                )
                self.assertEqual(
                    authority.source_manifest_sha256,
                    SOURCE_MANIFEST_SHA256,
                )
                self.assertEqual(
                    authority.environment_manifest_sha256,
                    ENVIRONMENT_MANIFEST_SHA256,
                )
                self.assertEqual(
                    authority.source_commit,
                    PINNED_SOURCE_COMMIT,
                )
                self.assertEqual(authority.source_tree, PINNED_SOURCE_TREE)
                self.assertEqual(
                    set(fixture.inputs),
                    {
                        "prefill_key",
                        "prefill_value",
                        "append_key",
                        "append_value",
                        "decode_query",
                    },
                )

    def test_cpu_loaders_preserve_frozen_raw_bytes(self) -> None:
        import torch

        fixture = load_turboquant_fixture("turboquant_4bit_nc")
        for name, tensor in load_inputs_cpu(fixture).items():
            with self.subTest(name=name):
                record = fixture.input_record(name)
                raw = (
                    tensor.contiguous()
                    .view(-1)
                    .view(torch.uint8)
                    .numpy()
                    .tobytes()
                )
                self.assertEqual(tuple(tensor.shape), record.shape)
                self.assertEqual(hashlib.sha256(raw).hexdigest(), record.sha256)
        expected = load_tensor_cpu(fixture.output_record("decode_output"))
        self.assertEqual(tuple(expected.shape), (1, 32, 128))
        self.assertEqual(str(expected.dtype), "torch.bfloat16")

    def test_exact_packed_helpers_reject_one_byte_difference(self) -> None:
        fixture = load_turboquant_fixture("turboquant_k3v4_nc")
        for output_name in (
            "cache_after_store",
            "cache_after_append",
            "append_slot",
        ):
            with self.subTest(output_name=output_name):
                expected = load_tensor_cpu(
                    fixture.output_record(output_name)
                )
                self.assertTrue(
                    compare_packed_output_untimed(
                        fixture,
                        output_name,
                        expected,
                    ).passed
                )
                changed = expected.clone()
                changed.view(-1)[0] ^= 1
                mismatch = compare_packed_output_untimed(
                    fixture,
                    output_name,
                    changed,
                )
                self.assertFalse(mismatch.passed)
                self.assertNotEqual(
                    mismatch.observed_sha256,
                    mismatch.expected_sha256,
                )

    def test_decode_helper_uses_frozen_tolerance_and_finiteness(self) -> None:
        import torch

        fixture = load_turboquant_fixture("turboquant_3bit_nc")
        expected = load_tensor_cpu(fixture.output_record("decode_output"))
        comparison = compare_decode_output_untimed(fixture, expected)
        self.assertTrue(comparison.passed)
        self.assertTrue(comparison.finite)
        self.assertEqual(comparison.atol, DECODE_ATOL)
        self.assertEqual(comparison.rtol, DECODE_RTOL)

        nonfinite = expected.clone()
        nonfinite.view(-1)[0] = torch.inf
        failed = compare_decode_output_untimed(fixture, nonfinite)
        self.assertFalse(failed.passed)
        self.assertFalse(failed.finite)

    def test_slot_layout_comparison_is_exact(self) -> None:
        fixture = load_turboquant_fixture("turboquant_4bit_nc")
        self.assertTrue(
            compare_slot_layout(
                fixture,
                slot_size=fixture.slot_size,
                byte_breakdown_per_head_token=(
                    fixture.byte_breakdown_per_head_token
                ),
            ).passed
        )
        changed = dict(fixture.byte_breakdown_per_head_token)
        changed["packed_keys"] -= 1
        self.assertFalse(
            compare_slot_layout(
                fixture,
                slot_size=fixture.slot_size,
                byte_breakdown_per_head_token=changed,
            ).passed
        )

    def test_tampered_packed_bytes_fail_before_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied_root = Path(temporary) / "fixtures"
            shutil.copytree(DEFAULT_FIXTURE_ROOT, copied_root)
            target = (
                copied_root
                / "turboquant_4bit_nc"
                / "cache_after_store.uint8.bin"
            )
            raw = bytearray(target.read_bytes())
            raw[0] ^= 1
            target.write_bytes(raw)
            with self.assertRaisesRegex(
                TurboQuantFixtureError,
                "checksum mismatch",
            ):
                load_turboquant_fixture(
                    "turboquant_4bit_nc",
                    copied_root,
                )

    def test_unpinned_configuration_is_rejected(self) -> None:
        for configuration in ("turboquant_k8v4", "turboquant", "unknown"):
            with self.subTest(configuration=configuration):
                with self.assertRaises(TurboQuantFixtureError):
                    load_turboquant_fixture(configuration)


if __name__ == "__main__":
    unittest.main()
