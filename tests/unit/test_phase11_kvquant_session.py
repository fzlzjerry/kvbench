"""Focused CPU tests for the bounded Phase 11 KVQuant session bridge."""

from __future__ import annotations

import gc
import inspect
from types import SimpleNamespace
from typing import Any, cast
import unittest

from kvbench.adapters.kvquant import (
    KVQUANT_AGGREGATE_PATCH_SHA256,
    KVQUANT_AUTHORIZED_CONTAINER_DIGEST,
    KVQUANT_CORRECTED_COMMIT,
    KVQUANT_CORRECTED_TREE,
    KVQUANT_DECISIONS,
    KVQUANT_DETERMINISTIC_VALUE_DECODE_APIS,
    KVQUANT_EXECUTION_SOURCE_IDENTIFIER,
    KVQUANT_EXTENSION_SHA256,
    KVQUANT_METHOD_IDENTIFIER,
)
from kvbench.runtime.kvquant_cache import (
    KVQUANT_Q4_VALUE_DECODE_WORKSPACE_FORMULA_VERSION,
    KVQuantStaticCache,
    kvquant_q4_value_decode_workspace_bytes,
    kvquant_q4_value_decode_workspace_shape,
)
from kvbench.runtime.kvquant_session import (
    KVQuantEndpointSession,
    _historical_prefix_sha256,
    build_kvquant_endpoint_session,
    build_kvquant_operation_keys,
    kvquant_runtime_context,
    load_frozen_kvquant_method_config,
    phase11_kvquant_backend_fingerprint,
)
from kvbench.runtime.turboquant_session import (
    MeasurementEndpointSession,
    require_endpoint_session,
)
from kvbench.schema import GraphMode, MeasurementScope, RunnerKind
from kvbench.schema.phase11 import (
    PHASE11_AGGREGATE_PATCH_SHA256,
    PHASE11_AUTHORIZED_CONTAINER_DIGEST,
    PHASE11_BOUNDED_POINT_SIGNATURES,
    PHASE11_CORRECTED_COMMIT,
    PHASE11_CORRECTED_TREE,
    PHASE11_DECISIONS,
    PHASE11_EXTENSION_SHA256,
    PHASE11_METHOD_IDENTIFIER,
)


def _endpoint_with_frozen_rope_scratch(cache: KVQuantStaticCache) -> Any:
    import torch

    query = torch.empty(
        (
            cache.num_layers,
            cache.batch_size,
            cache.num_query_heads,
            1,
            cache.head_dim // 2,
        ),
        dtype=torch.bfloat16,
        device=cache.device,
    )
    key = torch.empty(
        (
            cache.num_layers,
            cache.batch_size,
            cache.num_kv_heads,
            1,
            cache.head_dim // 2,
        ),
        dtype=torch.bfloat16,
        device=cache.device,
    )
    return SimpleNamespace(
        query_rope_scratch=query,
        key_rope_scratch=key,
        workspace_bytes=(
            query.untyped_storage().nbytes()
            + key.untyped_storage().nbytes()
        ),
    )


def _session_for_accounting(
    cache: KVQuantStaticCache,
    *,
    operation_keys: tuple[Any, ...],
) -> KVQuantEndpointSession:
    endpoint = _endpoint_with_frozen_rope_scratch(cache)
    method = SimpleNamespace(
        allocated_bytes=lambda value: value.accounting().allocated_bytes,
        byte_breakdown=lambda value: value.byte_breakdown(),
    )
    return KVQuantEndpointSession(
        loaded=SimpleNamespace(
            receipt=SimpleNamespace(receipt_sha256="1" * 64)
        ),
        operation_keys=operation_keys,
        endpoint=endpoint,
        cache=cache,
        method=method,
        adapter_config_fingerprint="2" * 64,
        model_memory=SimpleNamespace(allocated_bytes=0),
        cache_memory=SimpleNamespace(
            allocated_bytes=(
                cache.accounting().allocated_bytes + endpoint.workspace_bytes
            )
        ),
        fixed_operation=lambda: None,
        graph=None,
        graph_evidence=None,
        eager_graph_comparison=None,
        growing_operations=(),
        reset_growing=None,
        warmed_outputs=(("3" * 64, True),),
        prefix_sha256="4" * 64,
    )


class Phase11KVQuantSessionTests(unittest.TestCase):
    def test_operation_builder_accepts_exact_nine_point_grid(self) -> None:
        built = []
        for (
            configuration,
            runner_kind,
            graph_mode,
            context,
            output_steps,
        ) in PHASE11_BOUNDED_POINT_SIGNATURES:
            keys = build_kvquant_operation_keys(
                configuration=configuration,
                runner_kind=runner_kind,
                graph_mode=graph_mode,
                starting_context=context,
                output_steps=output_steps,
            )
            built.append(keys)
            self.assertEqual(len(keys), output_steps)
            self.assertEqual(
                [key.historical_context for key in keys],
                list(range(context, context + output_steps)),
            )
            self.assertEqual(
                [key.attended_context for key in keys],
                list(range(context + 1, context + output_steps + 1)),
            )
            self.assertEqual(
                {key.capacity for key in keys},
                {context + output_steps},
            )
            self.assertTrue(
                all(
                    len(key.operation_fingerprint_sha256) == 64
                    for key in keys
                )
            )
        self.assertEqual(len(built), 9)
        self.assertEqual(sum(len(keys) for keys in built), 12)

    def test_operation_builder_rejects_campaign_expansion(self) -> None:
        rejected = (
            ("kvq3", RunnerKind.FIXED_L, GraphMode.EAGER, 4096, 1),
            ("kvq2", RunnerKind.FIXED_L, GraphMode.EAGER, 64, 1),
            ("kvq4", RunnerKind.FIXED_L, GraphMode.EAGER, 128, 2),
            (
                "kvq4",
                RunnerKind.GROWING_CONTEXT,
                GraphMode.CUDA_GRAPH,
                17,
                4,
            ),
            (
                "kvq3",
                RunnerKind.GROWING_CONTEXT,
                GraphMode.EAGER,
                17,
                4,
            ),
            (
                "kvq4",
                RunnerKind.GROWING_CONTEXT,
                GraphMode.EAGER,
                17,
                5,
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
                    build_kvquant_operation_keys(
                        configuration=configuration,
                        runner_kind=runner,
                        graph_mode=graph,
                        starting_context=context,
                        output_steps=steps,
                    )

    def test_runtime_context_is_exact_and_source_bound(self) -> None:
        self.assertEqual(
            KVQUANT_METHOD_IDENTIFIER,
            PHASE11_METHOD_IDENTIFIER,
        )
        self.assertNotEqual(
            KVQUANT_AGGREGATE_PATCH_SHA256,
            PHASE11_AGGREGATE_PATCH_SHA256,
        )
        self.assertNotEqual(KVQUANT_CORRECTED_COMMIT, PHASE11_CORRECTED_COMMIT)
        self.assertNotEqual(KVQUANT_CORRECTED_TREE, PHASE11_CORRECTED_TREE)
        self.assertNotEqual(KVQUANT_EXTENSION_SHA256, PHASE11_EXTENSION_SHA256)
        self.assertEqual(
            KVQUANT_AUTHORIZED_CONTAINER_DIGEST,
            PHASE11_AUTHORIZED_CONTAINER_DIGEST,
        )
        first = phase11_kvquant_backend_fingerprint()
        second = phase11_kvquant_backend_fingerprint()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        context = kvquant_runtime_context()
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
            "pytorch_flash_kvquant_longctx_deterministic_q23_v4",
        )
        self.assertEqual(KVQUANT_DECISIONS[:-1], PHASE11_DECISIONS)
        self.assertEqual(KVQUANT_DECISIONS[-1], "0029")
        self.assertNotEqual(
            KVQUANT_EXECUTION_SOURCE_IDENTIFIER,
            PHASE11_METHOD_IDENTIFIER,
        )

    def test_backend_fingerprint_binds_q23_and_capacity_q4_workspace(
        self,
    ) -> None:
        source = inspect.getsource(phase11_kvquant_backend_fingerprint)
        self.assertIn(
            "KVQUANT_DETERMINISTIC_VALUE_DECODE_APIS",
            source,
        )
        self.assertIn(
            "KVQUANT_Q4_VALUE_DECODE_WORKSPACE_FORMULA_VERSION",
            source,
        )
        self.assertIn("declared_cache_capacity", source)
        self.assertTrue(
            all(
                api.endswith("_deterministic_out")
                for api in KVQUANT_DETERMINISTIC_VALUE_DECODE_APIS.values()
            )
        )
        self.assertEqual(
            kvquant_q4_value_decode_workspace_shape(
                batch_size=1,
                total_attended_capacity=4097,
            ),
            (1, 32, 32, 128),
        )
        self.assertEqual(
            kvquant_q4_value_decode_workspace_bytes(
                batch_size=1,
                total_attended_capacity=4097,
            ),
            524_288,
        )
        self.assertIn(
            "capacity-v1",
            KVQUANT_Q4_VALUE_DECODE_WORKSPACE_FORMULA_VERSION,
        )

    def test_method_config_binds_calibration_without_license_gate(self) -> None:
        config = load_frozen_kvquant_method_config()
        self.assertEqual(config.method_config_id, "kvquant")
        self.assertEqual(
            tuple(variant.variant_id for variant in config.variants),
            ("kvq4", "kvq3", "kvq2"),
        )
        self.assertIsNotNone(config.calibration)
        source = inspect.getsource(load_frozen_kvquant_method_config)
        self.assertNotIn("B-006", source)
        self.assertNotIn("resolution.status", source)

    def test_session_builder_orders_authority_before_execution(self) -> None:
        source = inspect.getsource(build_kvquant_endpoint_session)
        config = source.index("load_frozen_kvquant_method_config()")
        factory = source.index("build_method_adapter(")
        prepare = source.index("method.prepare_runtime()")
        allocate = source.index("method.allocate(")
        initialize = source.index("method.initialize_cache_untimed(cache)")
        endpoint = source.index("BF16DecodeEndpoint(")
        prefill = source.index("endpoint.prefill(prefix_input_ids)")
        capture = source.index("capture_fixed_graph(")
        self.assertLess(config, factory)
        self.assertLess(factory, prepare)
        self.assertLess(prepare, allocate)
        self.assertLess(allocate, initialize)
        self.assertLess(initialize, endpoint)
        self.assertLess(endpoint, prefill)
        self.assertLess(prepare, capture)
        self.assertIn("variant_id=first.configuration", source)

    def test_prefix_digest_excludes_only_fixed_scratch_slot(self) -> None:
        import torch

        def scalar() -> Any:
            return torch.zeros((1,), dtype=torch.float32)

        def sparse() -> Any:
            return torch.zeros((1, 2, 1), dtype=torch.float32)

        def sparse_index() -> Any:
            return torch.zeros((1, 2, 1), dtype=torch.int32)

        cache = cast(
            KVQuantStaticCache,
            SimpleNamespace(
                sink_tokens=1,
                active_context=2,
                config_name="kvq4",
                packed_key_cache=torch.zeros(
                    (1, 1, 1, 1, 2), dtype=torch.int32
                ),
                packed_value_cache=torch.zeros(
                    (1, 1, 1, 1, 2), dtype=torch.int32
                ),
                key_codebook=scalar(),
                key_lookup_table=scalar(),
                key_lower_threshold=scalar(),
                key_upper_threshold=scalar(),
                key_zero_point=scalar(),
                rope_inv_freq=scalar(),
                value_codebook=scalar(),
                value_lookup_cache=torch.zeros((1, 1, 2, 1)),
                key_sparse_values=sparse().unsqueeze(1),
                key_sparse_indices=sparse_index().unsqueeze(1),
                value_sparse_values=sparse().unsqueeze(1),
                value_sparse_indices=sparse_index().unsqueeze(1),
                key_active_counts=torch.zeros((1, 1, 2), dtype=torch.int32),
                value_active_counts=torch.zeros((1, 1, 2), dtype=torch.int32),
                sink_key=scalar(),
                sink_value=scalar(),
            ),
        )
        original = _historical_prefix_sha256(cache, 2)
        cache.packed_key_cache[0, 0, 0, 0, 1] = 7
        self.assertEqual(
            _historical_prefix_sha256(cache, 2),
            original,
        )
        cache.packed_key_cache[0, 0, 0, 0, 0] = 9
        self.assertNotEqual(
            _historical_prefix_sha256(cache, 2),
            original,
        )

    def test_session_reuses_common_protocol_and_exact_accounting(self) -> None:
        cache = KVQuantStaticCache(
            config_name="kvq4",
            num_layers=32,
            batch_size=1,
            num_query_heads=32,
            num_kv_heads=8,
            capacity=129,
            head_dim=128,
            device="cpu",
        )
        cache.reset_active_length(128, key_active_entries=0)
        keys = build_kvquant_operation_keys(
            configuration="kvq4",
            runner_kind=RunnerKind.FIXED_L,
            graph_mode=GraphMode.EAGER,
            starting_context=128,
            output_steps=1,
        )
        allocated = cache.accounting().allocated_bytes
        session = _session_for_accounting(
            cache,
            operation_keys=keys,
        )
        endpoint_bytes = 163_840
        self.assertIs(require_endpoint_session(session), session)
        self.assertIsInstance(session, MeasurementEndpointSession)
        self.assertEqual(
            session.measurement_scope,
            MeasurementScope.MEASUREMENT_CONTAINER_ADMISSION,
        )
        accounting = session.method_cache_accounting()
        self.assertEqual(
            accounting["allocated_bytes"],
            allocated + endpoint_bytes,
        )
        self.assertEqual(
            accounting["endpoint_rope_scratch_bytes"],
            endpoint_bytes,
        )
        self.assertEqual(
            accounting["staging_bytes"],
            cache.accounting().staging_bytes + endpoint_bytes,
        )
        self.assertEqual(
            sum(session.method_byte_breakdown().values()),
            accounting["allocated_bytes"],
        )
        self.assertEqual(accounting["key_active_entries"], 0)
        self.assertGreater(accounting["active_storage_bytes"], 0)
        pointers_before = session.current_cache_pointers()
        self.assertIn(
            "endpoint_query_rope_scratch_data_ptr",
            pointers_before,
        )
        self.assertIn(
            "endpoint_key_rope_scratch_data_ptr",
            pointers_before,
        )
        session.endpoint.query_rope_scratch.zero_()
        session.endpoint.key_rope_scratch.zero_()
        self.assertEqual(
            session.current_cache_pointers(),
            pointers_before,
        )
        ratios = session.method_allocation_ratios()
        self.assertAlmostEqual(
            float(ratios["rho_alloc"]) * float(ratios["r_alloc"]),
            1.0,
        )
        self.assertLessEqual(float(ratios["reciprocal_error"]), 1.0e-9)
        self.assertIsNone(ratios["r_hbm"])

    def test_composite_accounting_is_exact_at_required_contexts(self) -> None:
        for family in ("kvq4", "kvq3", "kvq2"):
            for context in (5, 17, 18, 128, 4096):
                with self.subTest(family=family, context=context):
                    cache = KVQuantStaticCache(
                        config_name=family,
                        num_layers=32,
                        batch_size=1,
                        num_query_heads=32,
                        num_kv_heads=8,
                        capacity=context,
                        head_dim=128,
                        device="cpu",
                    )
                    cache.reset_active_length(
                        context,
                        key_active_entries=0,
                    )
                    session = _session_for_accounting(
                        cache,
                        operation_keys=(
                            SimpleNamespace(historical_context=context),
                        ),
                    )
                    cache_accounting = cache.accounting()
                    composite = session.method_cache_accounting()
                    self.assertEqual(
                        composite["endpoint_rope_scratch_bytes"],
                        163_840,
                    )
                    self.assertEqual(
                        composite["allocated_bytes"],
                        cache_accounting.allocated_bytes + 163_840,
                    )
                    self.assertEqual(
                        composite["predicted_tensor_bytes"],
                        cache_accounting.predicted_tensor_bytes + 163_840,
                    )
                    self.assertEqual(
                        composite["measured_tensor_bytes"],
                        cache_accounting.measured_tensor_bytes + 163_840,
                    )
                    self.assertLess(
                        float(composite["relative_error"]),
                        0.01,
                    )
                    self.assertEqual(
                        sum(session.method_byte_breakdown().values()),
                        composite["allocated_bytes"],
                    )
                    pointers = session.current_cache_pointers()
                    self.assertEqual(
                        pointers,
                        session.current_cache_pointers(),
                    )
                    self.assertEqual(
                        pointers,
                        session._pointers,
                    )
                    del session
                    del cache
                    gc.collect()

    def test_bridge_contains_no_runner_or_claim_expansion(self) -> None:
        module_source = inspect.getsource(
            __import__(
                "kvbench.runtime.kvquant_session",
                fromlist=["kvquant_session"],
            )
        )
        self.assertNotIn("run_fixed_l", module_source)
        self.assertNotIn("run_growing_context", module_source)
        self.assertNotIn("speedup", module_source.lower())
        self.assertNotIn("quality evaluation", module_source.lower())


if __name__ == "__main__":
    unittest.main()
