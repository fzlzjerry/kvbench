"""Synthetic CUDA controls for the Phase 3 runtime core."""

from __future__ import annotations

import unittest

import torch

from kvbench.runtime.allocation import audit_cuda_allocations
from kvbench.runtime.backend import forced_flash_execution
from kvbench.runtime.gqa_audit import (
    audit_cache_geometry,
    audit_gqa_operator,
    audit_mha_operator_control,
)
from kvbench.runtime.numerical import (
    compare_tensors_untimed,
    small_attention_reference,
)
from kvbench.runtime.static_cache import BF16StaticCache


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class Phase3CudaRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cuda:0")

    def test_static_cache_write_and_fixed_history(self) -> None:
        cache = BF16StaticCache(
            num_layers=2,
            batch_size=1,
            num_kv_heads=8,
            capacity=17,
            head_dim=128,
            device=self.device,
        )
        cache.prepare_prefill(16)
        prefix = torch.randn(
            (1, 8, 16, 128),
            dtype=torch.bfloat16,
            device=self.device,
        )
        for layer in range(2):
            cache.update(prefix, prefix, layer)
        cache.complete_prefill()
        history = cache.keys[:, :, :, :16, :].clone()
        cache.prepare_fixed(16)
        token = torch.randn(
            (1, 8, 1, 128),
            dtype=torch.bfloat16,
            device=self.device,
        )
        for layer in range(2):
            cache.update(token, token, layer)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(cache.keys[:, :, :, :16, :], history))

    def test_attention_matrix_covers_batches_lengths_causal_decode_and_mha(
        self,
    ) -> None:
        torch.manual_seed(11)
        for batch in (1, 2):
            for length in (7, 17):
                with self.subTest(batch=batch, length=length, mode="causal_gqa"):
                    query = torch.randn(
                        (batch, 32, length, 128),
                        dtype=torch.bfloat16,
                        device=self.device,
                    )
                    key = torch.randn(
                        (batch, 8, length, 128),
                        dtype=torch.bfloat16,
                        device=self.device,
                    )
                    value = torch.randn_like(key)
                    with forced_flash_execution():
                        observed = (
                            torch.nn.functional.scaled_dot_product_attention(
                                query,
                                key,
                                value,
                                dropout_p=0.0,
                                is_causal=True,
                                scale=128**-0.5,
                                enable_gqa=True,
                            )
                        )
                    reference = small_attention_reference(
                        query,
                        key,
                        value,
                        is_causal=True,
                        scale=128**-0.5,
                    )
                    comparison = compare_tensors_untimed(
                        observed,
                        reference,
                        atol=0.02,
                        rtol=0.02,
                    )
                    self.assertTrue(comparison.passed, comparison.to_dict())
                    self.assertTrue(torch.isfinite(observed[..., 0, :]).all())
                    self.assertTrue(torch.isfinite(observed[..., -1, :]).all())

                with self.subTest(batch=batch, length=length, mode="decode_gqa"):
                    query = torch.randn(
                        (batch, 32, 1, 128),
                        dtype=torch.bfloat16,
                        device=self.device,
                    )
                    key = torch.randn(
                        (batch, 8, length, 128),
                        dtype=torch.bfloat16,
                        device=self.device,
                    )
                    value = torch.randn_like(key)
                    with forced_flash_execution():
                        observed = (
                            torch.nn.functional.scaled_dot_product_attention(
                                query,
                                key,
                                value,
                                dropout_p=0.0,
                                is_causal=False,
                                scale=128**-0.5,
                                enable_gqa=True,
                            )
                        )
                    reference = small_attention_reference(
                        query,
                        key,
                        value,
                        is_causal=False,
                        scale=128**-0.5,
                    )
                    comparison = compare_tensors_untimed(
                        observed,
                        reference,
                        atol=0.02,
                        rtol=0.02,
                    )
                    self.assertTrue(comparison.passed, comparison.to_dict())

                with self.subTest(batch=batch, length=length, mode="causal_mha"):
                    query = torch.randn(
                        (batch, 32, length, 128),
                        dtype=torch.bfloat16,
                        device=self.device,
                    )
                    key = torch.randn_like(query)
                    value = torch.randn_like(query)
                    with forced_flash_execution():
                        observed = (
                            torch.nn.functional.scaled_dot_product_attention(
                                query,
                                key,
                                value,
                                dropout_p=0.0,
                                is_causal=True,
                                scale=128**-0.5,
                                enable_gqa=True,
                            )
                        )
                    reference = small_attention_reference(
                        query,
                        key,
                        value,
                        is_causal=True,
                        scale=128**-0.5,
                    )
                    comparison = compare_tensors_untimed(
                        observed,
                        reference,
                        atol=0.02,
                        rtol=0.02,
                    )
                    self.assertTrue(comparison.passed, comparison.to_dict())

    def test_strict_allocation_audit_distinguishes_in_place(self) -> None:
        value = torch.ones((1024,), dtype=torch.float32, device=self.device)
        torch.cuda.synchronize()
        in_place = audit_cuda_allocations(lambda: value.add_(1), device=self.device)
        self.assertTrue(in_place.audit_available)
        self.assertTrue(in_place.passed, in_place.to_dict())
        allocating = audit_cuda_allocations(lambda: value + 1, device=self.device)
        self.assertFalse(allocating.passed)
        self.assertGreater(allocating.allocation_event_count, 0)

    def test_gqa_operator_dispatch_has_no_expansion(self) -> None:
        query = torch.randn(
            (1, 32, 1, 128),
            dtype=torch.bfloat16,
            device=self.device,
        )
        key = torch.randn(
            (1, 8, 128, 128),
            dtype=torch.bfloat16,
            device=self.device,
        )
        audit = audit_gqa_operator(
            query,
            key,
            key,
            is_causal=False,
            scale=128**-0.5,
        )
        self.assertTrue(audit.passed, audit.to_dict())
        self.assertFalse(audit.query_head_sized_kv_temporary)
        cache = BF16StaticCache(
            num_layers=2,
            batch_size=1,
            num_kv_heads=8,
            capacity=128,
            head_dim=128,
            device=self.device,
        )
        geometry = audit_cache_geometry(cache, num_query_heads=32)
        self.assertTrue(geometry["uses_kv_head_geometry"])
        self.assertFalse(geometry["query_head_storage_detected"])

    def test_same_geometry_mha_operator_control_is_separate(self) -> None:
        query = torch.randn(
            (1, 32, 1, 128),
            dtype=torch.bfloat16,
            device=self.device,
        )
        key = torch.randn(
            (1, 32, 128, 128),
            dtype=torch.bfloat16,
            device=self.device,
        )
        audit = audit_mha_operator_control(
            query,
            key,
            key,
            is_causal=False,
            scale=128**-0.5,
        )
        self.assertTrue(audit.passed, audit.to_dict())
        self.assertTrue(audit.backend["control_only"])
        self.assertFalse(audit.backend["system_under_test"])


if __name__ == "__main__":
    unittest.main()
