"""Focused CUDA Graph capture/replay test for the BF16 adapter boundary."""

from __future__ import annotations

import unittest

import torch

from kvbench.adapters import MethodRuntimeContext, build_method_adapter
from kvbench.runtime.allocation import audit_cuda_allocations
from kvbench.runtime.cuda_graph import capture_fixed_graph


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class Phase4AdapterGraphTests(unittest.TestCase):
    def test_adapter_append_capture_replay_is_pointer_stable_and_zero_alloc(self) -> None:
        adapter = build_method_adapter(
            "bf16",
            MethodRuntimeContext(
                model_id="meta-llama/Llama-3.1-8B-Instruct",
                model_revision="0" * 40,
                backend_id="torch_sdpa_flash_gqa",
                backend_fingerprint="a" * 64,
                num_layers=32,
                num_query_heads=32,
                num_kv_heads=8,
                head_dim=128,
            ),
        )
        self.assertTrue(adapter.supports_cuda_graph())
        cache = adapter.allocate(
            batch_size=1,
            capacity=3,
            device="cuda:0",
        )
        cache.reset_active_length(2)
        cache.prepare_fixed(2)
        key = torch.full(
            (1, 8, 1, 128),
            7,
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        value = torch.full_like(key, 11)
        cache_position = torch.tensor([2], device="cuda:0")
        pointers_before = cache.pointers()

        def operation() -> object:
            return adapter.append_decode(
                cache,
                key,
                value,
                0,
                cache_position,
            )[0]

        graph = capture_fixed_graph(
            operation,
            warmup_steps=3,
            device="cuda:0",
        )
        first = graph.replay()
        first_copy = first.detach().cpu().clone()
        second = graph.replay()
        second_copy = second.detach().cpu().clone()
        replay_audit = audit_cuda_allocations(graph.replay, device="cuda:0")
        self.assertTrue(torch.equal(first_copy, second_copy))
        self.assertEqual(int(first.data_ptr()), graph.output_data_ptr)
        self.assertEqual(int(second.data_ptr()), graph.output_data_ptr)
        self.assertEqual(cache.pointers(), pointers_before)
        self.assertTrue(replay_audit.passed, replay_audit.to_dict())
        self.assertEqual(replay_audit.allocation_event_count, 0)
        self.assertFalse(graph.to_dict()["fallback"])


if __name__ == "__main__":
    unittest.main()
