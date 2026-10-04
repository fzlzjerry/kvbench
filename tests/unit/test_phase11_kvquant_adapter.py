"""Focused CPU controls for the Phase 11 adapter-boundary correction."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
import unittest

import torch

from kvbench.adapters import (
    BF16MethodAdapter,
    KIVIMethodAdapter,
    KVCacheMethod,
    TurboQuantMethodAdapter,
)
from kvbench.adapters.base import MethodRuntimeContext, method_requires_pre_rope_key
from kvbench.adapters.kvquant import (
    KVQUANT_AGGREGATE_PATCH_SHA256,
    KVQUANT_AUTHORIZED_CONTAINER_DIGEST,
    KVQUANT_CORRECTED_COMMIT,
    KVQUANT_CORRECTED_TREE,
    KVQUANT_DECISIONS,
    KVQUANT_DETERMINISTIC_VALUE_DECODE_APIS,
    KVQUANT_EXTENSION_SHA256,
    KVQUANT_Q4_DETERMINISTIC_VALUE_DECODE_API,
    KVQUANT_QUANTIZER_SHA256,
    KVQuantMethodAdapter,
    _required_extension_symbols,
)
from kvbench.runtime.bf16_endpoint import (
    BF16DecodeEndpoint,
    EndpointGeometryError,
    preserve_pre_rope_key_in_query_scratch,
)
from kvbench.runtime.kvquant_fixture import (
    load_fixture_tensor_file_untimed,
    load_kvquant_fixture,
)
from kvbench.runtime.static_cache import CacheStateError


class _Projection:
    def __init__(self, output: torch.Tensor) -> None:
        self.output = output

    def __call__(self, _hidden_states: torch.Tensor) -> torch.Tensor:
        return self.output.clone()


class _LegacyMethod:
    """Old boundary shape: it deliberately accepts no new keyword."""

    def __init__(self) -> None:
        self.appended_key: torch.Tensor | None = None

    def append_decode(
        self,
        cache_state: object,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_state, layer_idx, cache_position
        self.appended_key = key_states
        return key_states, value_states

    store_prefill = append_decode

    def decode_attention(
        self,
        attention: object,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *,
        scaling: float,
    ) -> torch.Tensor:
        del attention, key_states, value_states, scaling
        return query_states


class _PreRoPEMethod(_LegacyMethod):
    requires_pre_rope_key = True

    def __init__(self) -> None:
        super().__init__()
        self.pre_rope_key: torch.Tensor | None = None

    def append_decode(
        self,
        cache_state: object,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_position: torch.Tensor,
        *,
        key_pre_rope_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.pre_rope_key = key_pre_rope_states
        return super().append_decode(
            cache_state,
            key_states,
            value_states,
            layer_idx,
            cache_position,
        )

    store_prefill = append_decode


def _endpoint(method: object) -> BF16DecodeEndpoint:
    endpoint = object.__new__(BF16DecodeEndpoint)
    endpoint.cache = object()
    endpoint.method = method
    endpoint.method_requires_pre_rope_key = method_requires_pre_rope_key(method)
    endpoint.num_kv_heads = 8
    endpoint.head_dim = 128
    endpoint.query_rope_scratch = torch.empty(
        (1, 1, 32, 1, 64),
        dtype=torch.bfloat16,
    )
    endpoint.key_rope_scratch = torch.empty(
        (1, 1, 8, 1, 64),
        dtype=torch.bfloat16,
    )
    return endpoint


def _attention_inputs() -> tuple[
    SimpleNamespace,
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor],
]:
    query_projection = torch.arange(
        4096,
        dtype=torch.bfloat16,
    ).reshape(1, 1, 4096)
    key_projection = (
        torch.arange(1024, dtype=torch.bfloat16).reshape(1, 1, 1024) + 3
    )
    value_projection = torch.full(
        (1, 1, 1024),
        7,
        dtype=torch.bfloat16,
    )
    attention = SimpleNamespace(
        q_proj=_Projection(query_projection),
        k_proj=_Projection(key_projection),
        v_proj=_Projection(value_projection),
        layer_idx=0,
        scaling=128**-0.5,
        o_proj=lambda output: output,
    )
    hidden = torch.zeros((1, 1, 4096), dtype=torch.bfloat16)
    cos = torch.zeros((1, 1, 128), dtype=torch.bfloat16)
    sin = torch.ones((1, 1, 128), dtype=torch.bfloat16)
    return attention, hidden, key_projection, (cos, sin)


class Phase11KVQuantBoundaryTests(unittest.TestCase):
    def test_protocol_pre_rope_input_is_optional_keyword_only(self) -> None:
        for operation in (
            KVCacheMethod.store_prefill,
            KVCacheMethod.append_decode,
        ):
            parameter = inspect.signature(operation).parameters[
                "key_pre_rope_states"
            ]
            self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
            self.assertIsNone(parameter.default)

    def test_existing_adapters_default_to_post_rope_only(self) -> None:
        for adapter_type in (
            BF16MethodAdapter,
            TurboQuantMethodAdapter,
            KIVIMethodAdapter,
        ):
            with self.subTest(adapter=adapter_type.__name__):
                adapter = object.__new__(adapter_type)
                self.assertFalse(method_requires_pre_rope_key(adapter))

    def test_capability_is_static_and_fails_closed(self) -> None:
        class _Invalid:
            requires_pre_rope_key = "yes"

        self.assertTrue(method_requires_pre_rope_key(_PreRoPEMethod()))
        with self.assertRaisesRegex(TypeError, "class-level bool"):
            method_requires_pre_rope_key(_Invalid())

    def test_legacy_method_receives_only_attention_ready_key(self) -> None:
        method = _LegacyMethod()
        endpoint = _endpoint(method)
        attention, hidden, original_key, positions = _attention_inputs()
        endpoint._attention(
            attention,
            hidden,
            positions,
            torch.tensor([0]),
            measured_decode=True,
        )
        self.assertIsNotNone(method.appended_key)
        expected_pre_rope = original_key.view(1, 1, 8, 128).transpose(1, 2)
        self.assertFalse(torch.equal(method.appended_key, expected_pre_rope))

    def test_declared_method_receives_pre_and_post_rope_keys(self) -> None:
        method = _PreRoPEMethod()
        endpoint = _endpoint(method)
        attention, hidden, original_key, positions = _attention_inputs()
        endpoint._attention(
            attention,
            hidden,
            positions,
            torch.tensor([0]),
            measured_decode=True,
        )
        expected_pre_rope = original_key.view(1, 1, 8, 128).transpose(1, 2)
        self.assertIsNotNone(method.pre_rope_key)
        self.assertTrue(torch.equal(method.pre_rope_key, expected_pre_rope))
        self.assertFalse(torch.equal(method.appended_key, method.pre_rope_key))
        self.assertEqual(tuple(method.pre_rope_key.shape), (1, 8, 1, 128))
        self.assertEqual(
            method.pre_rope_key.untyped_storage().data_ptr(),
            endpoint.query_rope_scratch[0].untyped_storage().data_ptr(),
        )

    def test_query_scratch_reuse_fails_closed_when_storage_is_insufficient(
        self,
    ) -> None:
        key = torch.empty((1, 8, 1, 128), dtype=torch.bfloat16)
        scratch = torch.empty((1, 7, 1, 64), dtype=torch.bfloat16)
        with self.assertRaisesRegex(EndpointGeometryError, "too small"):
            preserve_pre_rope_key_in_query_scratch(scratch, key)

    def test_boundary_helper_has_no_forbidden_measured_operations(self) -> None:
        source = inspect.getsource(preserve_pre_rope_key_in_query_scratch)
        for forbidden in (
            "torch.cat",
            "repeat_kv",
            "repeat_interleave",
            ".item(",
            ".tolist(",
            ".cpu(",
            ".numpy(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


def _runtime_context() -> MethodRuntimeContext:
    return MethodRuntimeContext(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        model_revision="0e9e39f249a16976918f6564b8830bc894c89659",
        backend_id="kvquant_corrected_direct_compressed",
        backend_fingerprint="0" * 64,
        num_layers=32,
        num_query_heads=32,
        num_kv_heads=8,
        head_dim=128,
    )


class Phase11KVQuantMethodTests(unittest.TestCase):
    def test_exact_authority_and_required_cuda_surface(self) -> None:
        self.assertEqual(
            KVQUANT_AGGREGATE_PATCH_SHA256,
            "7b9d3cc6773e8ef37697601c885f2c5ec581dffd57cf59424d03e68f147bd55a",
        )
        self.assertEqual(
            KVQUANT_CORRECTED_COMMIT,
            "34b0bdfa83082e1f30387d9ac5cca369006e089c",
        )
        self.assertEqual(
            KVQUANT_CORRECTED_TREE,
            "1f85af65fe03061583ffe8bd91e47d7ecffdd312",
        )
        self.assertEqual(
            KVQUANT_EXTENSION_SHA256,
            "b3c33badb8e55b19d6b2ce535182e964ce51e5102d8413b29701dd3d817ad73d",
        )
        self.assertEqual(KVQUANT_DECISIONS[-1], "0029")
        self.assertEqual(
            KVQUANT_AUTHORIZED_CONTAINER_DIGEST,
            (
                "sha256:"
                "059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e"
            ),
        )
        symbols = set(_required_extension_symbols())
        self.assertEqual(len(symbols), 18)
        self.assertEqual(
            KVQUANT_DETERMINISTIC_VALUE_DECODE_APIS[4],
            KVQUANT_Q4_DETERMINISTIC_VALUE_DECODE_API,
        )
        self.assertTrue(
            set(KVQUANT_DETERMINISTIC_VALUE_DECODE_APIS.values())
            <= symbols
        )
        for bits in (4, 3, 2):
            self.assertIn(f"vecquant{bits}appendvecKsparse", symbols)
            self.assertIn(
                f"vecquant{bits}appendvecVsparseParallel",
                symbols,
            )

    def test_all_three_static_configurations_and_fail_closed_geometry(
        self,
    ) -> None:
        context = _runtime_context()
        for configuration, bits in (("kvq4", 4), ("kvq3", 3), ("kvq2", 2)):
            with self.subTest(configuration=configuration):
                method = KVQuantMethodAdapter(context, configuration)
                cache = method.allocate(
                    batch_size=1,
                    capacity=18,
                    device="cpu",
                )
                self.assertEqual(method.bits, bits)
                self.assertEqual(
                    method.quantizer_sha256,
                    KVQUANT_QUANTIZER_SHA256[configuration],
                )
                self.assertTrue(method.requires_pre_rope_key)
                self.assertEqual(
                    tuple(cache.packed_key_cache.shape),
                    (32, 1, 8, bits * 4, 18),
                )
        with self.assertRaisesRegex(ValueError, "unsupported"):
            KVQuantMethodAdapter(context, "kvq5")

    def test_append_rejects_unbound_position_identity_before_cuda_path(
        self,
    ) -> None:
        method = KVQuantMethodAdapter(_runtime_context(), "kvq4")
        cache = method.allocate(
            batch_size=1,
            capacity=18,
            device="cpu",
        )
        method.initialize_cache_untimed(cache)
        cache.reset_active_length(17, key_active_entries=0)
        bound_position = torch.tensor([17], dtype=torch.int64)
        cache.bind_fixed_position_tensor_untimed(
            bound_position,
            logical_position=17,
        )
        cache.prepare_fixed(17)
        key = torch.zeros((1, 8, 1, 128), dtype=torch.bfloat16)
        value = torch.zeros_like(key)
        with self.assertRaisesRegex(CacheStateError, "physical slot"):
            method.append_decode(
                cache,
                key,
                value,
                0,
                torch.tensor([17], dtype=torch.int64),
                key_pre_rope_states=key,
            )

    def test_frozen_layer_zero_metadata_matches_corrected_oracle(self) -> None:
        context = _runtime_context()
        for configuration in ("kvq4", "kvq3", "kvq2"):
            with self.subTest(configuration=configuration):
                method = KVQuantMethodAdapter(context, configuration)
                cache = method.allocate(
                    batch_size=1,
                    capacity=18,
                    device="cpu",
                )
                method.initialize_cache_untimed(cache)
                fixture = load_kvquant_fixture(
                    configuration,
                    "key_few_value_fixed12",
                )
                metadata = load_fixture_tensor_file_untimed(
                    fixture,
                    "metadata.safetensors",
                )
                comparisons = (
                    (
                        cache.key_lookup_table[0],
                        metadata["key_lookup_table"],
                    ),
                    (
                        cache.key_lower_threshold[0].reshape(-1),
                        metadata["key_runtime_lower_threshold"],
                    ),
                    (
                        cache.key_upper_threshold[0].reshape(-1),
                        metadata["key_runtime_upper_threshold"],
                    ),
                    (cache.key_codebook[0], metadata["key_codebook"]),
                    (cache.value_codebook[0], metadata["value_codebook"]),
                    (cache.rope_inv_freq, metadata["rope_inv_freq"]),
                )
                for actual, expected in comparisons:
                    self.assertTrue(torch.equal(actual, expected))

    def test_measured_adapter_source_has_no_forbidden_host_or_growth_path(
        self,
    ) -> None:
        sources = "\n".join(
            inspect.getsource(operation)
            for operation in (
                KVQuantMethodAdapter._pack_nonsink_token,
                KVQuantMethodAdapter.append_decode,
                KVQuantMethodAdapter._decode_compressed,
                KVQuantMethodAdapter._decode_quantized_value,
            )
        )
        for forbidden in (
            "torch.cat",
            "torch.nonzero",
            "torch.argsort",
            "repeat_kv",
            "repeat_interleave",
            ".item(",
            ".tolist(",
            ".cpu(",
            ".numpy(",
            "synchronize(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, sources)

    def test_all_bits_use_deterministic_caller_owned_workspace(
        self,
    ) -> None:
        class _Runtime:
            def __init__(self) -> None:
                self.calls: list[tuple[str, tuple[object, ...]]] = []

            def __getattr__(self, name: str) -> object:
                def call(*args: object) -> None:
                    self.calls.append((name, args))

                return call

        for configuration in ("kvq4", "kvq3", "kvq2"):
            with self.subTest(configuration=configuration):
                method = KVQuantMethodAdapter(
                    _runtime_context(),
                    configuration,
                )
                cache = method.allocate(
                    batch_size=1,
                    capacity=18,
                    device="cpu",
                )
                runtime = _Runtime()
                value_weights = torch.empty((1, 32, 13))
                method._decode_quantized_value(
                    runtime=runtime,
                    cache=cache,
                    layer_idx=0,
                    value_weights=value_weights,
                    quantized=13,
                    batch_idx=0,
                )
                self.assertEqual(len(runtime.calls), 1)
                name, arguments = runtime.calls[0]
                self.assertEqual(len(arguments), 8)
                self.assertEqual(
                    name,
                    KVQUANT_DETERMINISTIC_VALUE_DECODE_APIS[method.bits],
                )
                if configuration == "kvq4":
                    self.assertEqual(
                        arguments[-1].untyped_storage().data_ptr(),
                        cache.q4_value_decode_workspace.untyped_storage().data_ptr(),
                    )
                else:
                    self.assertEqual(
                        arguments[-1].untyped_storage().data_ptr(),
                        cache.q23_value_decode_workspace.untyped_storage().data_ptr(),
                    )
                    self.assertIsNone(cache.q4_value_decode_workspace)

    def test_prefill_uses_corrected_parallel_value_store(self) -> None:
        source = inspect.getsource(KVQuantMethodAdapter.store_prefill)
        self.assertIn(
            'f"vecquant{self.bits}appendvecVsparseParallel"',
            source,
        )
        self.assertIn("value_store_lower_bounds", source)
        self.assertIn("value_store_upper_bounds", source)


if __name__ == "__main__":
    unittest.main()
