from __future__ import annotations

import math
import unittest

from kvbench.schema.phase8 import (
    RECIPROCAL_ABS_TOLERANCE,
    Phase7LegacyAllocationRatio,
    Phase8AllocationRatios,
)


class Phase8RatioTerminologyTests(unittest.TestCase):
    def test_canonical_ratios_are_derived_in_the_right_direction(self) -> None:
        ratios = Phase8AllocationRatios.from_bytes(
            allocated_bytes=80, bf16_allocated_bytes=100
        )
        self.assertEqual(ratios.rho_alloc, 0.8)
        self.assertEqual(ratios.r_alloc, 1.25)
        self.assertLessEqual(
            abs(ratios.r_alloc * ratios.rho_alloc - 1.0),
            RECIPROCAL_ABS_TOLERANCE,
        )

    def test_swapped_ratio_names_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "rho_alloc"):
            Phase8AllocationRatios(
                allocated_bytes=80,
                bf16_allocated_bytes=100,
                rho_alloc=1.25,
                r_alloc=0.8,
            )

    def test_phase7_field_is_explicitly_legacy_rho(self) -> None:
        legacy = Phase7LegacyAllocationRatio.from_phase7_r_alloc(0.65625)
        self.assertEqual(legacy.rho_alloc_legacy, 0.65625)
        self.assertTrue(
            math.isclose(
                legacy.canonical_r_alloc,
                1.0 / 0.65625,
                rel_tol=1e-9,
                abs_tol=0.0,
            )
        )

    def test_nonfinite_or_nonpositive_legacy_values_are_rejected(self) -> None:
        for value in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    Phase7LegacyAllocationRatio.from_phase7_r_alloc(value)


if __name__ == "__main__":
    unittest.main()
