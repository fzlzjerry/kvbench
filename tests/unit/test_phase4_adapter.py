"""CPU-only tests for the minimal Phase 4 method boundary."""

from __future__ import annotations

import ast
import inspect
import unittest

import torch

from kvbench.adapters import (
    BF16MethodAdapter,
    KVCacheMethod,
    MethodRuntimeContext,
    build_method_adapter,
)
from kvbench.errors import ConfigLoadError, ErrorCode, PhaseNotImplementedError
from kvbench.runtime.fixed_l_runner import run_fixed_l
from kvbench.runtime.growing_context_runner import run_growing_context
from kvbench.runtime.method_harness import (
    execution_path_audit_facade,
    summarize_allocation_harness,
)
from kvbench.schema import MethodName


def _context() -> MethodRuntimeContext:
    return MethodRuntimeContext(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        model_revision="0" * 40,
        backend_id="torch_sdpa_flash_gqa",
        backend_fingerprint="a" * 64,
        num_layers=32,
        num_query_heads=32,
        num_kv_heads=8,
        head_dim=128,
    )


class _ProtocolFake:
    name = "fake"
    adapter_version = "fake-1"

    def allocate(self, **kwargs: object) -> object:
        return kwargs

    def store_prefill(
        self,
        cache_state: object,
        key_states: object,
        value_states: object,
        layer_idx: int,
        cache_position: object,
    ) -> tuple[object, object]:
        del cache_state, layer_idx, cache_position
        return key_states, value_states

    append_decode = store_prefill

    def decode_attention(
        self,
        attention: object,
        query_states: object,
        key_states: object,
        value_states: object,
        *,
        scaling: float,
    ) -> object:
        del attention, key_states, value_states, scaling
        return query_states

    def allocated_bytes(self, cache_state: object) -> int:
        del cache_state
        return 0

    def byte_breakdown(self, cache_state: object) -> dict[str, int]:
        del cache_state
        return {"data_bytes": 0}

    def logical_bf16_bytes(self, cache_state: object) -> int:
        del cache_state
        return 0

    def config_fingerprint(self, cache_layout_fingerprint: str) -> str:
        del cache_layout_fingerprint
        return "0" * 64

    def supports_cuda_graph(self) -> bool:
        return False


class Phase4AdapterTests(unittest.TestCase):
    def test_factory_is_explicit_and_fail_closed(self) -> None:
        adapter = build_method_adapter(MethodName.BF16, _context())
        self.assertIsInstance(adapter, BF16MethodAdapter)
        self.assertEqual(adapter.name, "bf16")
        for method in ("kivi", "kvquant"):
            with self.subTest(method=method):
                with self.assertRaises(PhaseNotImplementedError) as caught:
                    build_method_adapter(method, _context())
                self.assertEqual(caught.exception.code, ErrorCode.PHASE_NOT_IMPLEMENTED)
        for method in ("turboquant", "turboquant_k8v4", "unknown"):
            with self.subTest(method=method):
                with self.assertRaises(ConfigLoadError):
                    build_method_adapter(method, _context())

    def test_protocol_conforming_fake(self) -> None:
        self.assertIsInstance(_ProtocolFake(), KVCacheMethod)

    def test_fingerprint_and_byte_breakdown_are_deterministic(self) -> None:
        adapter = build_method_adapter("bf16", _context())
        cache = adapter.allocate(
            batch_size=1,
            capacity=4,
            device="cpu",
            workspace_bytes=4096,
        )
        layout = cache.layout_fingerprint()
        first = adapter.config_fingerprint(layout)
        second = adapter.config_fingerprint(layout)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        breakdown = dict(adapter.byte_breakdown(cache))
        self.assertEqual(sum(breakdown.values()), adapter.allocated_bytes(cache))
        self.assertEqual(
            adapter.logical_bf16_bytes(cache),
            cache.predicted_tensor_bytes,
        )
        self.assertEqual(breakdown["workspace_bytes"], 4096)
        self.assertEqual(breakdown["scale_bytes"], 0)
        self.assertEqual(breakdown["zero_point_bytes"], 0)

    def test_cache_writes_keep_native_kv_geometry(self) -> None:
        adapter = build_method_adapter("bf16", _context())
        cache = adapter.allocate(
            batch_size=1,
            capacity=4,
            device="cpu",
        )
        cache.prepare_prefill(2)
        key = torch.full((1, 8, 2, 128), 3, dtype=torch.bfloat16)
        value = torch.full((1, 8, 2, 128), 5, dtype=torch.bfloat16)
        attended_key, attended_value = adapter.store_prefill(
            cache,
            key,
            value,
            0,
            torch.arange(2),
        )
        self.assertEqual(tuple(attended_key.shape), (1, 8, 2, 128))
        self.assertTrue(torch.equal(attended_key, key))
        self.assertTrue(torch.equal(attended_value, value))
        cache.complete_prefill()
        cache.prepare_fixed(2)
        decode_key = torch.full((1, 8, 1, 128), 7, dtype=torch.bfloat16)
        decode_value = torch.full((1, 8, 1, 128), 11, dtype=torch.bfloat16)
        attended_key, attended_value = adapter.append_decode(
            cache,
            decode_key,
            decode_value,
            0,
            torch.tensor([2]),
        )
        self.assertEqual(tuple(attended_key.shape), (1, 8, 3, 128))
        self.assertTrue(torch.equal(attended_key[:, :, 2:, :], decode_key))
        self.assertTrue(torch.equal(attended_value[:, :, 2:, :], decode_value))

    def test_allocation_and_execution_facades_reuse_derived_verdicts(self) -> None:
        adapter = build_method_adapter("bf16", _context())
        cache = adapter.allocate(batch_size=1, capacity=2, device="cpu")
        evidence = {
            "passed": True,
            "allocated_delta": 0,
            "reserved_delta": 0,
            "event_counts": {},
        }
        allocation = summarize_allocation_harness(
            adapter,
            cache,
            eager_evidence=evidence,
            graph_replay_evidence=evidence,
        )
        self.assertTrue(allocation.passed)
        self.assertEqual(allocation.persistent_allocated_delta, 0)
        self.assertEqual(allocation.graph_replay_allocations, {})
        path = execution_path_audit_facade(
            backend_identity_verified=True,
            device_kernel_family_verified=True,
            allocation_categories_verified=True,
            temporary_tensor_shapes_verified=True,
            gqa_replication_detected=False,
            full_prefix_temporary_detected=False,
            host_synchronization_detected=False,
            backend_fallback_detected=False,
        )
        self.assertTrue(path.passed)
        self.assertEqual(path.full_prefix_dequantization, "not_applicable")

    def test_runners_use_common_session_facades(self) -> None:
        for runner in (run_fixed_l, run_growing_context):
            source = inspect.getsource(runner)
            session_attributes = {
                node.attr
                for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "session"
            }
            self.assertNotIn("cache", session_attributes)
            self.assertNotIn("BF16StaticCache", source)
            self.assertIn("method_cache_accounting", source)
            self.assertIn("adapter_config_fingerprint", source)


if __name__ == "__main__":
    unittest.main()
