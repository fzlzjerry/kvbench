from __future__ import annotations

import unittest

from scripts.qp0_performance_freeze import LOCKED_PATHS, ROOTS, method_family


class QP0FreezeTests(unittest.TestCase):
    def test_method_family_mapping(self) -> None:
        self.assertEqual(method_family("bf16"), "bf16")
        self.assertEqual(method_family("tq_4bit_nc"), "turboquant")
        self.assertEqual(method_family("k2v2"), "kivi")
        self.assertEqual(method_family("kvq3"), "kvquant")

    def test_locked_paths_cover_measured_hot_path(self) -> None:
        for required in ("src/kvbench/adapters/", "src/kvbench/runtime/", "configs/methods/", "configs/models/"):
            self.assertIn(required, LOCKED_PATHS)

    def test_release_roles_remain_separate(self) -> None:
        self.assertNotEqual(ROOTS["phase14_graph_ab_timing"], ROOTS["phase15_profiler_mechanism"])
        self.assertNotEqual(ROOTS["phase16_outer_cuda_event_secondary"], ROOTS["phase16_host_wall_primary"])


if __name__ == "__main__":
    unittest.main()
