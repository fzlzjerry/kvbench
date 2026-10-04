"""Synthetic fixed-shape CUDA Graph controls for Phase 3."""

from __future__ import annotations

import unittest

import torch

from kvbench.runtime.allocation import audit_cuda_allocations
from kvbench.runtime.cuda_graph import capture_fixed_graph


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class Phase3GraphRuntimeTests(unittest.TestCase):
    def test_capture_replay_pointer_and_allocation_stability(self) -> None:
        value = torch.ones((4096,), dtype=torch.float32, device="cuda:0")

        def operation() -> torch.Tensor:
            return value.mul_(1.0001)

        captured = capture_fixed_graph(operation, warmup_steps=0, device="cuda:0")
        pointer = captured.output_data_ptr
        first = captured.replay()
        torch.cuda.synchronize()
        second = captured.replay()
        torch.cuda.synchronize()
        self.assertEqual(int(first.data_ptr()), pointer)
        self.assertEqual(int(second.data_ptr()), pointer)
        audit = audit_cuda_allocations(captured.replay, device="cuda:0")
        self.assertTrue(audit.passed, audit.to_dict())


if __name__ == "__main__":
    unittest.main()
