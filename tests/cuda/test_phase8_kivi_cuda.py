"""Exact-container CUDA conformance against the frozen Phase 7 KIVI fixtures."""

from __future__ import annotations

import gc
import hashlib
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import unittest

from kvbench.adapters.base import MethodRuntimeContext
from kvbench.adapters.kivi import KIVIMethodAdapter
from kvbench.runtime.kivi_fixture import (
    KIVI_FIXTURE_CONFIGS,
    KIVIFixture,
    load_kivi_fixture,
    tensor_from_record,
)
from kvbench.runtime.turboquant_admission import (
    PHASE6_CONTAINER_ENVIRONMENT_VALUE,
    PHASE6_CONTAINER_ENVIRONMENT_VARIABLE,
    PHASE6_IMAGE_ENVIRONMENT_VARIABLE,
    require_authorized_cuda_environment,
)
from kvbench.runtime.static_cache import CacheStateError
from kvbench.schema.phase6 import AUTHORIZED_CONTAINER_DIGEST


DECODE_ATOL = 0.02
DECODE_RTOL = 0.02
TEST_CAPACITY = 35
TEST_LAYER = 0
_STATE_TENSORS = (
    "quantized_key_payload",
    "quantized_value_payload",
    "key_scales",
    "key_minimum_offsets",
    "value_scales",
    "value_minimum_offsets",
    "residual_key",
    "residual_value",
)


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
        model_id="phase8-frozen-fixture-model",
        model_revision="phase8-frozen-fixture-revision",
        backend_id="patched-official-kivi-fixture",
        backend_fingerprint=hashlib.sha256(
            b"phase8-patched-official-kivi-fixture"
        ).hexdigest(),
        num_layers=32,
        num_query_heads=32,
        num_kv_heads=8,
        head_dim=128,
    )


def _fixture_value(fixture: KIVIFixture, dotted_path: str) -> Any:
    value: Any = fixture.payload
    for component in dotted_path.split("."):
        if type(value) is not dict or component not in value:
            raise AssertionError(f"fixture path is absent: {dotted_path}")
        value = value[component]
    return value


@unittest.skipUnless(
    _authorized_environment_declared(),
    "Phase 8 KIVI CUDA is authorized only in the exact Measurement Container",
)
class Phase8KIVIFixtureCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.environment = require_authorized_cuda_environment(
            AUTHORIZED_CONTAINER_DIGEST
        )
        cls.torch = __import__("torch")
        cls.device = cls.torch.device("cuda:0")
        cls.attention = SimpleNamespace(layer_idx=TEST_LAYER)
        cls.scaling = 1.0 / math.sqrt(128)

    def _load_inputs(self, fixture: KIVIFixture) -> tuple[Any, Any, Any]:
        key_record = fixture.tensor_record("inputs.key_0_33")
        value_record = fixture.tensor_record("inputs.value_0_33")
        query_record = fixture.tensor_record("inputs.selected_queries")
        self.assertIsNotNone(key_record)
        self.assertIsNotNone(value_record)
        self.assertIsNotNone(query_record)
        key = tensor_from_record(key_record, device=self.device)
        value = tensor_from_record(value_record, device=self.device)
        query = tensor_from_record(query_record, device=self.device)
        self.assertEqual(key.dtype, self.torch.bfloat16)
        self.assertEqual(value.dtype, self.torch.bfloat16)
        self.assertEqual(query.dtype, self.torch.bfloat16)
        return key, value, query

    def _new_cache(
        self,
        method: KIVIMethodAdapter,
        *,
        capacity: int = TEST_CAPACITY,
    ) -> Any:
        cache = method.allocate(
            batch_size=1,
            capacity=capacity,
            device=self.device,
        )
        cache.initialize_deterministic()
        return cache

    def _prefill(
        self,
        method: KIVIMethodAdapter,
        cache: Any,
        key: Any,
        value: Any,
        length: int,
    ) -> None:
        cache.prepare_prefill(length)
        method.store_prefill(
            cache,
            key[:, :, :length, :],
            value[:, :, :length, :],
            TEST_LAYER,
            self.torch.arange(length, dtype=self.torch.int64, device=self.device),
        )
        cache.complete_prefill()

    def _ordered_value_residual(self, cache: Any) -> Any | None:
        count = cache._value_residual_counts[TEST_LAYER]
        if count == 0:
            return None
        head = cache._value_residual_heads[TEST_LAYER]
        source = cache.value_residual_ring[TEST_LAYER]
        ordered = cache.value_residual_ordered_staging[TEST_LAYER]
        first = min(count, cache.residual_length - head)
        ordered[:, :, :first, :].copy_(source[:, :, head : head + first, :])
        if count > first:
            ordered[:, :, first:count, :].copy_(
                source[:, :, : count - first, :]
            )
        return ordered[:, :, :count, :]

    def _observed_state_tensors(self, cache: Any) -> dict[str, Any | None]:
        key_history = cache._key_history_counts[TEST_LAYER]
        key_residual = cache._key_residual_counts[TEST_LAYER]
        value_history = cache._value_history_counts[TEST_LAYER]
        key_words = key_history * cache.k_bits // 32
        key_groups = key_history // cache.group_size
        layer = TEST_LAYER
        return {
            "quantized_key_payload": (
                None
                if key_history == 0
                else cache.packed_key_history[
                    layer, :, :, :key_words, :
                ].transpose(-1, -2)
            ),
            "quantized_value_payload": (
                None
                if value_history == 0
                else cache.packed_value_history[
                    layer, :, :, :, :value_history
                ].transpose(-1, -2)
            ),
            "key_scales": (
                None
                if key_history == 0
                else cache.key_scales[
                    layer, :, :, :key_groups, :
                ].transpose(-1, -2)
            ),
            "key_minimum_offsets": (
                None
                if key_history == 0
                else cache.key_minimums[
                    layer, :, :, :key_groups, :
                ].transpose(-1, -2)
            ),
            "value_scales": (
                None
                if value_history == 0
                else cache.value_scales[
                    layer, :, :, :, :value_history
                ].transpose(-1, -2)
            ),
            "value_minimum_offsets": (
                None
                if value_history == 0
                else cache.value_minimums[
                    layer, :, :, :, :value_history
                ].transpose(-1, -2)
            ),
            "residual_key": (
                None
                if key_residual == 0
                else cache.key_residual[
                    layer, :, :, :key_residual, :
                ]
            ),
            "residual_value": self._ordered_value_residual(cache),
        }

    def _assert_state_matches(
        self,
        fixture: KIVIFixture,
        cache: Any,
        state_path: str,
    ) -> None:
        state = _fixture_value(fixture, state_path)
        self.assertEqual(cache.active_context, state["length"])
        token_state = cache.token_index_state(TEST_LAYER)
        expected_tokens = {
            "quantized_key_tokens": "quantized_key_tokens",
            "residual_key_tokens": "residual_key_tokens",
            "quantized_value_tokens": "quantized_value_tokens",
            "residual_value_tokens": "residual_value_tokens",
        }
        for observed_name, fixture_name in expected_tokens.items():
            self.assertEqual(
                token_state[observed_name].tolist(),
                state[fixture_name],
                f"{fixture.config_name} {state_path} {fixture_name}",
            )

        observed = self._observed_state_tensors(cache)
        for name in _STATE_TENSORS:
            record = fixture.tensor_record(f"{state_path}.tensors.{name}")
            actual = observed[name]
            if record is None:
                self.assertIsNone(
                    actual,
                    f"{fixture.config_name} {state_path} unexpectedly owns {name}",
                )
                continue
            self.assertIsNotNone(actual)
            expected = tensor_from_record(record, device=self.device)
            self.assertEqual(tuple(actual.shape), tuple(expected.shape))
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertTrue(
                self.torch.equal(actual, expected),
                f"{fixture.config_name} {state_path} differs at {name}",
            )

    def _decode_pending(
        self,
        method: KIVIMethodAdapter,
        cache: Any,
        key: Any,
        value: Any,
        query: Any,
        token: int,
        *,
        growing_step: int | None = None,
    ) -> Any:
        if growing_step is not None:
            cache.select_growing_step(growing_step)
        handles = method.append_decode(
            cache,
            key[:, :, token : token + 1, :],
            value[:, :, token : token + 1, :],
            TEST_LAYER,
            self.torch.tensor(
                [token], dtype=self.torch.int64, device=self.device
            ),
        )
        return method.decode_attention(
            self.attention,
            query,
            handles[0],
            handles[1],
            scaling=self.scaling,
        )

    def _assert_decode_matches(
        self,
        fixture: KIVIFixture,
        cache: Any,
        output: Any,
        output_path: str,
    ) -> None:
        record = fixture.tensor_record(output_path)
        self.assertIsNotNone(record)
        expected = tensor_from_record(record, device=self.device)
        actual_fp16 = cache.decode_output_fp16.unsqueeze(2)
        self.assertTrue(self.torch.isfinite(actual_fp16).all().item())
        self.assertTrue(self.torch.isfinite(output).all().item())
        self.torch.testing.assert_close(
            actual_fp16,
            expected,
            atol=DECODE_ATOL,
            rtol=DECODE_RTOL,
        )
        self.torch.testing.assert_close(
            output,
            expected.to(dtype=self.torch.bfloat16),
            atol=DECODE_ATOL,
            rtol=DECODE_RTOL,
        )

    def _fixed_output(
        self,
        method: KIVIMethodAdapter,
        key: Any,
        value: Any,
        selected_queries: Any,
        *,
        context: int,
        query_index: int,
    ) -> tuple[Any, Any]:
        cache = self._new_cache(method)
        prefix = context - 1
        self._prefill(method, cache, key, value, prefix)
        cache.prepare_fixed(prefix)
        output = self._decode_pending(
            method,
            cache,
            key,
            value,
            selected_queries[:, :, query_index : query_index + 1, :],
            prefix,
        )
        return cache, output

    def test_cache_position_requires_cuda_int64(self) -> None:
        method = KIVIMethodAdapter(_runtime_context(), "k4v4")
        key = self.torch.zeros(
            (1, 8, 1, 128),
            dtype=self.torch.bfloat16,
            device=self.device,
        )
        value = self.torch.zeros_like(key)
        for dtype in (self.torch.int32, self.torch.float32):
            with self.subTest(dtype=str(dtype)):
                cache = self._new_cache(method)
                cache.prepare_prefill(1)
                position = self.torch.zeros(
                    (1,),
                    dtype=dtype,
                    device=self.device,
                )
                with self.assertRaises(CacheStateError):
                    method.store_prefill(
                        cache,
                        key,
                        value,
                        TEST_LAYER,
                        position,
                    )
                self.assertEqual(cache.active_context, 0)

    def test_all_four_frozen_configurations_store_append_and_rollover(self) -> None:
        output_cases = (
            (17, 0, "basic.store_output"),
            (31, 2, "rollover.before.output"),
        )
        rollover_cases = (
            (0, 31, 3, "rollover.boundary"),
            (1, 32, 4, "rollover.after"),
            (2, 33, 5, "rollover.post_rollover_decode"),
        )
        for configuration in KIVI_FIXTURE_CONFIGS:
            with self.subTest(configuration=configuration):
                fixture = load_kivi_fixture(configuration)
                method = KIVIMethodAdapter(_runtime_context(), configuration)
                method.prepare_runtime()
                key, value, selected_queries = self._load_inputs(fixture)
                allocation_records = fixture.legacy_allocation_records()
                accounting_capacity = max(
                    int(record["context"]) for record in allocation_records
                )
                accounting_cache = self._new_cache(
                    method,
                    capacity=accounting_capacity,
                )
                for record in allocation_records:
                    observed_bytes = (
                        accounting_cache.reference_active_byte_breakdown(
                            record["context"]
                        )
                    )
                    self.assertEqual(
                        observed_bytes,
                        record["categories"],
                    )
                    self.assertEqual(
                        sum(observed_bytes.values()),
                        record["actual_total"],
                    )
                    self.assertAlmostEqual(
                        record["rho_alloc_legacy"]
                        * record["canonical_r_alloc"],
                        1.0,
                        places=12,
                    )
                    self.assertIsNone(record["r_hbm"])

                store_cache = self._new_cache(method)
                self._prefill(method, store_cache, key, value, 17)
                self._assert_state_matches(
                    fixture, store_cache, "basic.store_state"
                )

                for context, query_index, output_path in output_cases:
                    output_cache, output = self._fixed_output(
                        method,
                        key,
                        value,
                        selected_queries,
                        context=context,
                        query_index=query_index,
                    )
                    self._assert_decode_matches(
                        fixture, output_cache, output, output_path
                    )

                append_cache = self._new_cache(method)
                self._prefill(method, append_cache, key, value, 17)
                append_cache.prepare_growing(17, 1)
                append_output = self._decode_pending(
                    method,
                    append_cache,
                    key,
                    value,
                    selected_queries[:, :, 1:2, :],
                    17,
                    growing_step=0,
                )
                self._assert_decode_matches(
                    fixture,
                    append_cache,
                    append_output,
                    "basic.decode_output",
                )
                self._assert_state_matches(
                    fixture, append_cache, "basic.append_state"
                )

                rollover_cache = self._new_cache(method)
                self._prefill(method, rollover_cache, key, value, 31)
                self._assert_state_matches(
                    fixture, rollover_cache, "rollover.before.state"
                )
                rollover_cache.prepare_growing(31, 3)
                for (
                    step,
                    token,
                    query_index,
                    stage_path,
                ) in rollover_cases:
                    output = self._decode_pending(
                        method,
                        rollover_cache,
                        key,
                        value,
                        selected_queries[
                            :, :, query_index : query_index + 1, :
                        ],
                        token,
                        growing_step=step,
                    )
                    self._assert_decode_matches(
                        fixture,
                        rollover_cache,
                        output,
                        f"{stage_path}.output",
                    )
                    self._assert_state_matches(
                        fixture,
                        rollover_cache,
                        f"{stage_path}.state",
                    )

                del (
                    accounting_cache,
                    append_cache,
                    append_output,
                    key,
                    method,
                    output,
                    output_cache,
                    rollover_cache,
                    selected_queries,
                    store_cache,
                    value,
                )
                gc.collect()
                self.torch.cuda.synchronize()
                self.torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
