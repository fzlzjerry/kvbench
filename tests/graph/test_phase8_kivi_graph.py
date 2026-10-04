"""Exact-container CUDA Graph checks for the static Phase 8 KIVI adapter."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from kvbench.adapters.base import MethodRuntimeContext
from kvbench.adapters.kivi import KIVIMethodAdapter
from kvbench.runtime.allocation import audit_cuda_allocations
from kvbench.runtime.cuda_graph import capture_fixed_graph
from kvbench.runtime.kivi_cache import KIVI_MANDATORY_CONFIGS
from kvbench.runtime.turboquant_admission import (
    PHASE6_CONTAINER_ENVIRONMENT_VALUE,
    PHASE6_CONTAINER_ENVIRONMENT_VARIABLE,
    PHASE6_IMAGE_ENVIRONMENT_VARIABLE,
    require_authorized_cuda_environment,
)
from kvbench.schema.phase6 import AUTHORIZED_CONTAINER_DIGEST


PREFIX_LENGTH = 128
CAPACITY = PREFIX_LENGTH + 1
DECODE_ATOL = 0.02
DECODE_RTOL = 0.02


def _authorized_environment_declared() -> bool:
    """Never authorize this test merely because native-host CUDA is present."""

    return (
        Path("/.dockerenv").is_file()
        and os.environ.get(PHASE6_IMAGE_ENVIRONMENT_VARIABLE)
        == AUTHORIZED_CONTAINER_DIGEST
        and os.environ.get(PHASE6_CONTAINER_ENVIRONMENT_VARIABLE)
        == PHASE6_CONTAINER_ENVIRONMENT_VALUE
    )


def _runtime_context() -> MethodRuntimeContext:
    return MethodRuntimeContext(
        model_id="phase8-kivi-graph-fixture",
        model_revision="phase8-kivi-graph-fixture-revision",
        backend_id="patched-official-kivi-direct-compressed-decode",
        backend_fingerprint=hashlib.sha256(
            b"phase8-kivi-direct-compressed-decode-graph"
        ).hexdigest(),
        num_layers=32,
        num_query_heads=32,
        num_kv_heads=8,
        head_dim=128,
    )


@unittest.skipUnless(
    _authorized_environment_declared(),
    "Phase 8 KIVI CUDA Graph is authorized only in the exact Measurement Container",
)
class Phase8KIVIGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.environment = require_authorized_cuda_environment(
            AUTHORIZED_CONTAINER_DIGEST
        )
        cls.torch = __import__("torch")
        cls.device = cls.torch.device("cuda:0")
        cls.attention = SimpleNamespace(layer_idx=0)
        cls.scaling = 1.0 / math.sqrt(128)

    def _inputs(self, seed: int) -> tuple[object, ...]:
        generator = self.torch.Generator(device=self.device)
        generator.manual_seed(seed)
        key_history = self.torch.randn(
            (1, 8, PREFIX_LENGTH, 128),
            dtype=self.torch.bfloat16,
            device=self.device,
            generator=generator,
        )
        value_history = self.torch.randn(
            key_history.shape,
            dtype=self.torch.bfloat16,
            device=self.device,
            generator=generator,
        )
        pending_key = self.torch.randn(
            (1, 8, 1, 128),
            dtype=self.torch.bfloat16,
            device=self.device,
            generator=generator,
        )
        pending_value = self.torch.randn(
            pending_key.shape,
            dtype=self.torch.bfloat16,
            device=self.device,
            generator=generator,
        )
        query = self.torch.randn(
            (1, 32, 1, 128),
            dtype=self.torch.bfloat16,
            device=self.device,
            generator=generator,
        )
        positions = self.torch.arange(
            PREFIX_LENGTH,
            dtype=self.torch.int64,
            device=self.device,
        )
        decode_position = self.torch.tensor(
            [PREFIX_LENGTH],
            dtype=self.torch.int64,
            device=self.device,
        )
        return (
            key_history,
            value_history,
            pending_key,
            pending_value,
            query,
            positions,
            decode_position,
        )

    @staticmethod
    def _pointer_snapshot(
        cache: object,
        tensors: tuple[object, ...],
    ) -> dict[str, int]:
        pointers = dict(cache.pointers())
        pointers.update(
            {
                f"input_{index}_data_ptr": int(tensor.data_ptr())
                for index, tensor in enumerate(tensors)
            }
        )
        return pointers

    def test_mandatory_configs_capture_direct_decode_without_replay_allocation(
        self,
    ) -> None:
        for seed, configuration in enumerate(
            sorted(KIVI_MANDATORY_CONFIGS),
            start=800,
        ):
            with self.subTest(configuration=configuration):
                method = KIVIMethodAdapter(_runtime_context(), configuration)
                method.prepare_runtime()
                cache = method.allocate(
                    batch_size=1,
                    capacity=CAPACITY,
                    device=self.device,
                )
                cache.initialize_deterministic()
                (
                    key_history,
                    value_history,
                    pending_key,
                    pending_value,
                    query,
                    positions,
                    decode_position,
                ) = self._inputs(seed)

                cache.prepare_prefill(PREFIX_LENGTH)
                method.store_prefill(
                    cache,
                    key_history,
                    value_history,
                    0,
                    positions,
                )
                cache.complete_prefill()
                cache.prepare_fixed(PREFIX_LENGTH)

                tracked_tensors = (
                    key_history,
                    value_history,
                    pending_key,
                    pending_value,
                    query,
                    positions,
                    decode_position,
                )

                def operation() -> object:
                    handles = method.append_decode(
                        cache,
                        pending_key,
                        pending_value,
                        0,
                        decode_position,
                    )
                    return method.decode_attention(
                        self.attention,
                        query,
                        handles[0],
                        handles[1],
                        scaling=self.scaling,
                    )

                with mock.patch(
                    "kvbench.adapters.kivi.flash_attention_forward",
                    side_effect=AssertionError(
                        "compressed decode fell back to a BF16 attention backend"
                    ),
                ) as fallback:
                    # Runtime loading, Triton compilation, cache population, and
                    # every exact decode kernel shape are prepared before capture.
                    for _ in range(3):
                        operation()
                    self.torch.cuda.synchronize(device=self.device)
                    eager = operation().detach().cpu().clone()
                    eager_audit = audit_cuda_allocations(
                        operation,
                        device=self.device,
                    )
                    pointers_before = self._pointer_snapshot(
                        cache,
                        tracked_tensors,
                    )

                    graph = capture_fixed_graph(
                        operation,
                        warmup_steps=0,
                        device=self.device,
                    )
                    first = graph.replay()
                    first_copy = first.detach().cpu().clone()
                    second = graph.replay()
                    second_copy = second.detach().cpu().clone()
                    replay_audit = audit_cuda_allocations(
                        graph.replay,
                        device=self.device,
                    )

                self.torch.testing.assert_close(
                    first_copy,
                    eager,
                    atol=DECODE_ATOL,
                    rtol=DECODE_RTOL,
                )
                self.assertTrue(self.torch.equal(first_copy, second_copy))
                self.assertTrue(self.torch.isfinite(second_copy).all().item())
                self.assertEqual(
                    int(first.data_ptr()),
                    int(cache.output_buffer.data_ptr()),
                )
                self.assertEqual(
                    int(second.data_ptr()),
                    graph.output_data_ptr,
                )
                self.assertEqual(
                    self._pointer_snapshot(cache, tracked_tensors),
                    pointers_before,
                )
                self.assertTrue(replay_audit.audit_available)
                self.assertTrue(replay_audit.passed, replay_audit.to_dict())
                self.assertTrue(eager_audit.audit_available)
                self.assertTrue(eager_audit.passed, eager_audit.to_dict())
                self.assertEqual(eager_audit.allocation_event_count, 0)
                self.assertEqual(eager_audit.allocation_event_bytes, 0)
                self.assertEqual(
                    eager_audit.allocated_after,
                    eager_audit.allocated_before,
                )
                self.assertEqual(
                    eager_audit.reserved_after,
                    eager_audit.reserved_before,
                )
                self.assertEqual(replay_audit.allocation_event_count, 0)
                self.assertEqual(replay_audit.allocation_event_bytes, 0)
                self.assertEqual(
                    replay_audit.allocated_after,
                    replay_audit.allocated_before,
                )
                self.assertEqual(
                    replay_audit.reserved_after,
                    replay_audit.reserved_before,
                )
                self.assertFalse(graph.to_dict()["fallback"])
                fallback.assert_not_called()
                self.assertFalse(cache.gqa_geometry()["gqa_materialized"])
                self.assertEqual(cache.gqa_geometry()["num_kv_heads"], 8)


if __name__ == "__main__":
    unittest.main()
