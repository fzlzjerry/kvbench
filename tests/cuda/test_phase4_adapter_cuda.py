"""Focused CUDA checks for the BF16 adapter cache boundary."""

from __future__ import annotations

import unittest

import torch

from kvbench.adapters import MethodRuntimeContext, build_method_adapter
from kvbench.runtime.allocation import audit_cuda_allocations
from kvbench.runtime.gqa_audit import audit_cache_geometry


def _adapter() -> object:
    context = MethodRuntimeContext(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        model_revision="0" * 40,
        backend_id="torch_sdpa_flash_gqa",
        backend_fingerprint="a" * 64,
        num_layers=32,
        num_query_heads=32,
        num_kv_heads=8,
        head_dim=128,
    )
    return build_method_adapter("bf16", context)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class Phase4AdapterCudaTests(unittest.TestCase):
    def test_native_kv_append_is_allocation_free(self) -> None:
        adapter = _adapter()
        cache = adapter.allocate(
            batch_size=1,
            capacity=4,
            device="cuda:0",
        )
        cache.prepare_prefill(2)
        prefix_key = torch.zeros(
            (1, 8, 2, 128),
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        prefix_value = torch.ones_like(prefix_key)
        for layer in range(32):
            adapter.store_prefill(
                cache,
                prefix_key,
                prefix_value,
                layer,
                torch.arange(2, device="cuda:0"),
            )
        cache.complete_prefill()
        cache.prepare_fixed(2)
        decode_key = torch.full(
            (1, 8, 1, 128),
            2,
            dtype=torch.bfloat16,
            device="cuda:0",
        )
        decode_value = torch.full_like(decode_key, 3)

        cache_position = torch.tensor([2], device="cuda:0")

        def operation() -> object:
            return adapter.append_decode(
                cache,
                decode_key,
                decode_value,
                0,
                cache_position,
            )[0]

        operation()
        audit = audit_cuda_allocations(operation, device="cuda:0")
        self.assertTrue(audit.passed, audit.to_dict())
        self.assertEqual(audit.allocation_event_count, 0)
        geometry = audit_cache_geometry(cache, num_query_heads=32)
        self.assertTrue(geometry["uses_kv_head_geometry"])
        self.assertFalse(geometry["query_head_storage_detected"])
        self.assertEqual(cache.num_kv_heads, 8)
        self.assertEqual(
            sum(adapter.byte_breakdown(cache).values()),
            adapter.allocated_bytes(cache),
        )


if __name__ == "__main__":
    unittest.main()
