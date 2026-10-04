"""Focused common-runner session tests for the bounded Phase 8 KIVI grid."""

from __future__ import annotations

import hashlib
import inspect
import json
import struct
from types import SimpleNamespace
import unittest

from kvbench.runtime.kivi_cache import KIVIStaticCache
from kvbench.runtime.kivi_session import (
    KIVIEndpointSession,
    _historical_prefix_sha256,
    _logical_prefix_sha256,
    build_kivi_endpoint_session,
    build_kivi_operation_keys,
    kivi_runtime_context,
    load_frozen_kivi_method_config,
    phase8_kivi_backend_fingerprint,
)
from kvbench.runtime.numerical import tensor_sha256_untimed
from kvbench.runtime.turboquant_session import (
    MeasurementEndpointSession,
    require_endpoint_session,
)
from kvbench.schema import GraphMode, MeasurementScope, RunnerKind


class Phase8KIVISessionTests(unittest.TestCase):
    def test_untimed_checksum_preserves_raw_bytes_without_numpy(self) -> None:
        import torch

        tensor = torch.tensor([[1, -2]], dtype=torch.int32)
        header = json.dumps(
            {"shape": [1, 2], "dtype": "torch.int32"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        expected = hashlib.sha256()
        expected.update(header)
        expected.update(b"\0")
        expected.update(struct.pack("<ii", 1, -2))
        self.assertEqual(
            tensor_sha256_untimed(tensor),
            expected.hexdigest(),
        )

    def test_operation_builder_accepts_exact_ten_point_grid(self) -> None:
        fixed_points = (
            ("k4v4", GraphMode.EAGER, 128),
            ("k4v4", GraphMode.CUDA_GRAPH, 128),
            ("k2v4", GraphMode.EAGER, 128),
            ("k2v4", GraphMode.CUDA_GRAPH, 128),
            ("k2v2", GraphMode.EAGER, 128),
            ("k2v2", GraphMode.CUDA_GRAPH, 128),
            ("k4v4", GraphMode.EAGER, 4096),
            ("k4v4", GraphMode.CUDA_GRAPH, 4096),
            ("k4v2", GraphMode.EAGER, 128),
        )
        keys = [
            build_kivi_operation_keys(
                configuration=configuration,
                runner_kind=RunnerKind.FIXED_L,
                graph_mode=graph_mode,
                starting_context=context,
                output_steps=1,
            )[0]
            for configuration, graph_mode, context in fixed_points
        ]
        growing = build_kivi_operation_keys(
            configuration="k4v4",
            runner_kind=RunnerKind.GROWING_CONTEXT,
            graph_mode=GraphMode.EAGER,
            starting_context=31,
            output_steps=4,
        )
        self.assertEqual(len(keys) + 1, 10)
        self.assertEqual(
            [item.historical_context for item in growing],
            [31, 32, 33, 34],
        )
        self.assertEqual(
            [item.attended_context for item in growing],
            [32, 33, 34, 35],
        )
        self.assertEqual({item.capacity for item in growing}, {35})
        self.assertTrue(
            all(
                len(item.operation_fingerprint_sha256) == 64
                for item in (*keys, *growing)
            )
        )

    def test_operation_builder_rejects_every_campaign_expansion(self) -> None:
        rejected = (
            ("k4v2", RunnerKind.FIXED_L, GraphMode.CUDA_GRAPH, 128, 1),
            ("k2v4", RunnerKind.FIXED_L, GraphMode.EAGER, 4096, 1),
            ("k4v4", RunnerKind.FIXED_L, GraphMode.EAGER, 64, 1),
            ("k4v4", RunnerKind.FIXED_L, GraphMode.EAGER, 128, 2),
            (
                "k4v4",
                RunnerKind.GROWING_CONTEXT,
                GraphMode.CUDA_GRAPH,
                31,
                4,
            ),
            (
                "k4v4",
                RunnerKind.GROWING_CONTEXT,
                GraphMode.EAGER,
                31,
                5,
            ),
            (
                "k2v2",
                RunnerKind.GROWING_CONTEXT,
                GraphMode.EAGER,
                31,
                4,
            ),
        )
        for configuration, runner, graph, context, steps in rejected:
            with self.subTest(
                configuration=configuration,
                runner=runner,
                graph=graph,
                context=context,
                steps=steps,
            ):
                with self.assertRaises(ValueError):
                    build_kivi_operation_keys(
                        configuration=configuration,
                        runner_kind=runner,
                        graph_mode=graph,
                        starting_context=context,
                        output_steps=steps,
                    )

    def test_runtime_context_is_frozen_and_source_bound(self) -> None:
        first = phase8_kivi_backend_fingerprint()
        second = phase8_kivi_backend_fingerprint()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        self.assertNotEqual(first, "0" * 64)
        context = kivi_runtime_context()
        self.assertEqual(
            (
                context.num_layers,
                context.num_query_heads,
                context.num_kv_heads,
                context.head_dim,
            ),
            (32, 32, 8, 128),
        )
        self.assertEqual(
            context.backend_id,
            "pytorch_flash_patched_official_kivi",
        )

    def test_logical_prefix_stays_stable_across_real_rollover(self) -> None:
        cache = KIVIStaticCache(
            config_name="k4v4",
            num_layers=1,
            batch_size=1,
            num_query_heads=32,
            num_kv_heads=8,
            capacity=35,
            head_dim=128,
            device="cpu",
        )
        for token in range(31):
            cache.update(layer_idx=0, token_index=token)
        before = _logical_prefix_sha256(cache, 31)
        for token in range(31, 35):
            cache.update(layer_idx=0, token_index=token)
        after = _logical_prefix_sha256(cache, 31)
        self.assertEqual(before, after)
        state = cache.token_index_state(0)
        self.assertEqual(
            state["quantized_key_tokens"].tolist(),
            list(range(32)),
        )
        self.assertEqual(
            state["residual_key_tokens"].tolist(),
            [32, 33, 34],
        )
        self.assertEqual(
            state["quantized_value_tokens"].tolist(),
            [0, 1, 2],
        )
        self.assertEqual(
            state["residual_value_tokens"].tolist(),
            list(range(3, 35)),
        )

    def test_session_reuses_common_protocol_and_canonical_accounting(
        self,
    ) -> None:
        cache = KIVIStaticCache(
            config_name="k4v4",
            num_layers=1,
            batch_size=1,
            num_query_heads=32,
            num_kv_heads=8,
            capacity=129,
            head_dim=128,
            device="cpu",
        )
        for token in range(128):
            cache.update(layer_idx=0, token_index=token)
        keys = build_kivi_operation_keys(
            configuration="k4v4",
            runner_kind=RunnerKind.FIXED_L,
            graph_mode=GraphMode.EAGER,
            starting_context=128,
            output_steps=1,
        )
        allocated = cache.accounting().allocated_bytes
        method = SimpleNamespace(
            allocated_bytes=lambda value: value.accounting().allocated_bytes,
            byte_breakdown=lambda value: value.byte_breakdown(),
        )
        session = KIVIEndpointSession(
            loaded=SimpleNamespace(
                receipt=SimpleNamespace(receipt_sha256="1" * 64)
            ),
            operation_keys=keys,
            endpoint=object(),
            cache=cache,
            method=method,
            adapter_config_fingerprint="2" * 64,
            model_memory=SimpleNamespace(allocated_bytes=0),
            cache_memory=SimpleNamespace(allocated_bytes=allocated),
            fixed_operation=lambda: None,
            graph=None,
            graph_evidence=None,
            eager_graph_comparison=None,
            growing_operations=(),
            reset_growing=None,
            warmed_outputs=(("3" * 64, True),),
            prefix_sha256=_historical_prefix_sha256(cache, keys[0]),
        )
        self.assertEqual(
            session.current_historical_prefix_sha256(),
            session._prefix_sha256,
        )
        self.assertIs(require_endpoint_session(session), session)
        self.assertIsInstance(session, MeasurementEndpointSession)
        self.assertEqual(
            session.measurement_scope,
            MeasurementScope.MEASUREMENT_CONTAINER_ADMISSION,
        )
        accounting = session.method_cache_accounting()
        self.assertEqual(accounting["allocated_bytes"], allocated)
        self.assertEqual(
            accounting["logical_bf16_allocated_bytes"],
            cache.logical_bf16_storage_bytes,
        )
        ratios = session.method_allocation_ratios()
        self.assertAlmostEqual(
            ratios["rho_alloc"] * ratios["r_alloc"],
            1.0,
        )
        self.assertAlmostEqual(ratios["reciprocal_product"], 1.0)
        self.assertIsNone(cache.r_hbm)

    def test_runtime_authority_is_prepared_before_prefill_or_capture(
        self,
    ) -> None:
        source = inspect.getsource(build_kivi_endpoint_session)
        self.assertIn("load_frozen_kivi_method_config()", source)
        self.assertIn("variant_id=first.configuration", source)
        config = load_frozen_kivi_method_config()
        self.assertEqual(config.method_config_id, "kivi")
        self.assertEqual(
            tuple(variant.variant_id for variant in config.variants),
            ("k4v4", "k2v4", "k2v2", "k4v2"),
        )
        prepared = source.index("method.prepare_runtime()")
        prefill = source.index("endpoint.prefill(prefix_input_ids)")
        capture = source.index("capture_fixed_graph(")
        self.assertLess(prepared, prefill)
        self.assertLess(prepared, capture)


if __name__ == "__main__":
    unittest.main()
