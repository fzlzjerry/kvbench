"""Fixed-L CUDA Graph checks for the Phase 11 KVQuant adapter."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import unittest
from unittest import mock

from kvbench.adapters.base import MethodRuntimeContext
from kvbench.adapters.kvquant import (
    KVQUANT_AUTHORIZED_CONTAINER_DIGEST,
    KVQuantMethodAdapter,
)
from kvbench.runtime.allocation import audit_cuda_allocations
from kvbench.runtime.cuda_graph import capture_fixed_graph
from kvbench.runtime.kvquant_fixture import (
    compare_decode_output_untimed,
    load_fixture_tensor_file_untimed,
    load_kvquant_fixture,
)
from kvbench.runtime.turboquant_admission import (
    PHASE6_CONTAINER_ENVIRONMENT_VALUE,
    PHASE6_CONTAINER_ENVIRONMENT_VARIABLE,
    PHASE6_IMAGE_ENVIRONMENT_VARIABLE,
    require_authorized_cuda_environment,
)


LAYER = 0
PREFIX_LENGTH = 17
CAPACITY = 18
SINK_TOKENS = 5
SCALING = 1.0 / math.sqrt(128)
GRAPH_CASES = (
    ("kvq4", "key_cap_value_fixed12"),
    ("kvq3", "key_few_value_fixed12"),
    ("kvq2", "key_zero_value_fixed12"),
)


def _authorized_environment_declared() -> bool:
    extension = os.environ.get("KVBENCH_KVQUANT_EXTENSION")
    return (
        Path("/.dockerenv").is_file()
        and os.environ.get(PHASE6_IMAGE_ENVIRONMENT_VARIABLE)
        == KVQUANT_AUTHORIZED_CONTAINER_DIGEST
        and os.environ.get(PHASE6_CONTAINER_ENVIRONMENT_VARIABLE)
        == PHASE6_CONTAINER_ENVIRONMENT_VALUE
        and extension is not None
        and Path(extension).is_file()
    )


def _runtime_context() -> MethodRuntimeContext:
    return MethodRuntimeContext(
        model_id="phase11-kvquant-graph-fixture",
        model_revision="0e9e39f249a16976918f6564b8830bc894c89659",
        backend_id="kvquant-gqa-longctx-deterministic-q23-v4",
        backend_fingerprint=hashlib.sha256(
            b"phase11-kvquant-fixed-graph"
        ).hexdigest(),
        num_layers=32,
        num_query_heads=32,
        num_kv_heads=8,
        head_dim=128,
    )


@unittest.skipUnless(
    _authorized_environment_declared(),
    "Phase 11 KVQuant Graph is authorized only in the Measurement Container",
)
class Phase11KVQuantGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.environment = require_authorized_cuda_environment(
            KVQUANT_AUTHORIZED_CONTAINER_DIGEST
        )
        cls.torch = __import__("torch")
        cls.device = cls.torch.device("cuda:0")
        cls.attention = SimpleNamespace(layer_idx=LAYER)

    def _prepared_operation(
        self,
        family: str,
        case_name: str,
    ) -> tuple[Any, Any, Any, tuple[Any, ...], tuple[Any, ...]]:
        fixture = load_kvquant_fixture(family, case_name)
        inputs = load_fixture_tensor_file_untimed(
            fixture,
            "inputs.safetensors",
        )
        sink = load_fixture_tensor_file_untimed(
            fixture,
            "sink.safetensors",
        )
        decode = load_fixture_tensor_file_untimed(
            fixture,
            "decode_output.safetensors",
        )
        key_pre_rope = inputs["key_pre_rope"].to(device=self.device)
        value = inputs["value_after_v_proj"].to(device=self.device)
        key_attention = key_pre_rope.clone()
        key_attention[:, :, :SINK_TOKENS, :].copy_(
            sink["sink_key_attention_fp16"]
            .transpose(2, 3)
            .to(device=self.device, dtype=self.torch.bfloat16)
        )
        positions = inputs["position_ids"].reshape(-1).to(device=self.device)
        query = decode["query_attention_ready"].to(device=self.device)

        method = KVQuantMethodAdapter(_runtime_context(), family)
        method.prepare_runtime()
        cache = method.allocate(
            batch_size=1,
            capacity=CAPACITY,
            device=self.device,
        )
        method.initialize_cache_untimed(cache)
        cache.prepare_prefill(PREFIX_LENGTH)
        method.store_prefill(
            cache,
            key_attention[:, :, :PREFIX_LENGTH, :],
            value[:, :, :PREFIX_LENGTH, :],
            LAYER,
            positions[:PREFIX_LENGTH],
            key_pre_rope_states=key_pre_rope[:, :, :PREFIX_LENGTH, :],
        )
        cache.complete_prefill()
        append_position = positions[PREFIX_LENGTH:CAPACITY]
        cache.bind_fixed_position_tensor_untimed(
            append_position,
            logical_position=PREFIX_LENGTH,
        )
        cache.prepare_fixed(PREFIX_LENGTH)

        def operation() -> Any:
            handles = method.append_decode(
                cache,
                key_attention[:, :, PREFIX_LENGTH:CAPACITY, :],
                value[:, :, PREFIX_LENGTH:CAPACITY, :],
                LAYER,
                append_position,
                key_pre_rope_states=key_pre_rope[
                    :, :, PREFIX_LENGTH:CAPACITY, :
                ],
            )
            return method.decode_attention(
                self.attention,
                query,
                handles[0],
                handles[1],
                scaling=SCALING,
            )

        tracked = (
            key_pre_rope,
            key_attention,
            value,
            positions,
            query,
        )
        quantized_prefix = PREFIX_LENGTH - SINK_TOKENS
        historical = (
            cache.packed_key_cache[
                LAYER, 0, :, :, :quantized_prefix
            ].detach().cpu().clone(),
            cache.packed_value_cache[
                LAYER, 0, :, :, :quantized_prefix
            ].detach().cpu().clone(),
            cache.key_sparse_values[
                LAYER, 0, :quantized_prefix
            ].detach().cpu().clone(),
            cache.key_sparse_indices[
                LAYER, 0, :quantized_prefix
            ].detach().cpu().clone(),
            cache.value_sparse_values[
                LAYER, 0, :quantized_prefix
            ].detach().cpu().clone(),
            cache.value_sparse_indices[
                LAYER, 0, :quantized_prefix
            ].detach().cpu().clone(),
            cache.key_active_counts[
                LAYER, 0, :quantized_prefix
            ].detach().cpu().clone(),
            cache.value_active_counts[
                LAYER, 0, :quantized_prefix
            ].detach().cpu().clone(),
            cache.value_lookup_cache[
                LAYER, 0, :quantized_prefix
            ].detach().cpu().clone(),
            cache.sink_key[LAYER].detach().cpu().clone(),
            cache.sink_value[LAYER].detach().cpu().clone(),
        )
        return fixture, cache, operation, tracked, historical

    @staticmethod
    def _pointer_snapshot(
        cache: Any,
        tracked: tuple[Any, ...],
    ) -> dict[str, int]:
        pointers = dict(cache.pointers())
        pointers.update(
            {
                f"input_{index}_data_ptr": int(tensor.data_ptr())
                for index, tensor in enumerate(tracked)
            }
        )
        return pointers

    def _assert_history_unchanged(
        self,
        cache: Any,
        expected: tuple[Any, ...],
    ) -> None:
        quantized_prefix = PREFIX_LENGTH - SINK_TOKENS
        observed = (
            cache.packed_key_cache[LAYER, 0, :, :, :quantized_prefix],
            cache.packed_value_cache[LAYER, 0, :, :, :quantized_prefix],
            cache.key_sparse_values[LAYER, 0, :quantized_prefix],
            cache.key_sparse_indices[LAYER, 0, :quantized_prefix],
            cache.value_sparse_values[LAYER, 0, :quantized_prefix],
            cache.value_sparse_indices[LAYER, 0, :quantized_prefix],
            cache.key_active_counts[LAYER, 0, :quantized_prefix],
            cache.value_active_counts[LAYER, 0, :quantized_prefix],
            cache.value_lookup_cache[LAYER, 0, :quantized_prefix],
            cache.sink_key[LAYER],
            cache.sink_value[LAYER],
        )
        for actual, frozen in zip(observed, expected, strict=True):
            self.assertTrue(self.torch.equal(actual.detach().cpu(), frozen))

    def test_all_bit_widths_capture_append_and_direct_decode(self) -> None:
        for family, case_name in GRAPH_CASES:
            with self.subTest(family=family, case=case_name):
                fixture, cache, operation, tracked, historical = (
                    self._prepared_operation(family, case_name)
                )
                for _ in range(3):
                    operation()
                self.torch.cuda.synchronize(device=self.device)
                eager = operation().detach().cpu().clone()
                eager_fixture = compare_decode_output_untimed(fixture, eager)
                self.assertTrue(eager_fixture.passed, eager_fixture)
                eager_audit = audit_cuda_allocations(
                    operation,
                    device=self.device,
                )
                pointers_before = self._pointer_snapshot(cache, tracked)
                if family == "kvq4":
                    self.assertEqual(
                        tuple(cache.q4_value_decode_workspace.shape),
                        cache.q4_value_decode_workspace_shape,
                    )
                    self.assertIn(
                        "q4_value_decode_workspace_data_ptr",
                        pointers_before,
                    )
                else:
                    self.assertIsNone(cache.q4_value_decode_workspace)
                    self.assertIsNotNone(cache.q23_value_decode_workspace)
                    self.assertIn(
                        "q23_value_decode_workspace_data_ptr",
                        pointers_before,
                    )

                with mock.patch(
                    "kvbench.adapters.kvquant.flash_attention_forward",
                    side_effect=AssertionError(
                        "KVQuant compressed decode fell back to BF16 attention"
                    ),
                ) as fallback:
                    graph = capture_fixed_graph(
                        operation,
                        warmup_steps=0,
                        device=self.device,
                    )
                    first = graph.replay()
                    self.torch.cuda.synchronize(device=self.device)
                    first_copy = first.detach().cpu().clone()
                    second = graph.replay()
                    self.torch.cuda.synchronize(device=self.device)
                    second_copy = second.detach().cpu().clone()
                    replay_audit = audit_cuda_allocations(
                        graph.replay,
                        device=self.device,
                    )
                    fallback.assert_not_called()

                self.torch.testing.assert_close(
                    first_copy,
                    eager,
                    atol=0.01,
                    rtol=0.01,
                )
                self.assertTrue(self.torch.equal(first_copy, second_copy))
                self.assertTrue(bool(self.torch.isfinite(second_copy).all()))
                fixture_comparison = compare_decode_output_untimed(
                    fixture,
                    second_copy,
                )
                self.assertTrue(fixture_comparison.passed, fixture_comparison)
                self.assertEqual(int(first.data_ptr()), graph.output_data_ptr)
                self.assertEqual(int(second.data_ptr()), graph.output_data_ptr)
                self.assertEqual(
                    self._pointer_snapshot(cache, tracked),
                    pointers_before,
                )
                self._assert_history_unchanged(cache, historical)
                for audit in (eager_audit, replay_audit):
                    self.assertTrue(audit.audit_available)
                    self.assertTrue(audit.passed, audit.to_dict())
                    self.assertEqual(audit.allocation_event_count, 0)
                    self.assertEqual(audit.allocation_event_bytes, 0)
                    self.assertEqual(
                        audit.allocated_after,
                        audit.allocated_before,
                    )
                    self.assertEqual(
                        audit.reserved_after,
                        audit.reserved_before,
                    )
                self.assertFalse(graph.to_dict()["fallback"])
                self.assertEqual(cache.active_context, PREFIX_LENGTH)
                self.assertEqual(cache.gqa_geometry()["num_kv_heads"], 8)
                self.assertFalse(
                    cache.gqa_geometry()["query_head_sized_kv_cache"]
                )


if __name__ == "__main__":
    unittest.main()
