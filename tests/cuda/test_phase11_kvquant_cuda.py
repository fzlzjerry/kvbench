"""Exact-container CUDA conformance for the Phase 11 KVQuant adapter."""

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
from kvbench.adapters.kvquant import (
    KVQUANT_AUTHORIZED_CONTAINER_DIGEST,
    KVQUANT_EXTENSION_SHA256,
    KVQuantMethodAdapter,
)
from kvbench.runtime.kvquant_fixture import (
    KVQUANT_CASES,
    KVQUANT_FAMILIES,
    KVQuantFixture,
    compare_decode_output_untimed,
    compare_exact_fixture_tensor_untimed,
    load_all_kvquant_fixtures,
    load_fixture_tensor_file_untimed,
)
from kvbench.runtime.turboquant_admission import (
    PHASE6_CONTAINER_ENVIRONMENT_VALUE,
    PHASE6_CONTAINER_ENVIRONMENT_VARIABLE,
    PHASE6_IMAGE_ENVIRONMENT_VARIABLE,
    require_authorized_cuda_environment,
)


LAYER = 0
STORE_CONTEXT = 17
TOTAL_CONTEXT = 18
SINK_TOKENS = 5
SCALING = 1.0 / math.sqrt(128)
_TENSOR_FILES = (
    "inputs.safetensors",
    "dense_payload.safetensors",
    "metadata.safetensors",
    "sparse_values.safetensors",
    "sparse_indices.safetensors",
    "sink.safetensors",
    "store_state.safetensors",
    "append_state.safetensors",
    "decode_output.safetensors",
)


def _authorized_environment_declared() -> bool:
    """Never authorize native-host CUDA or a floating image identity."""

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
        model_id="phase11-kvquant-corrected-fixture",
        model_revision="0e9e39f249a16976918f6564b8830bc894c89659",
        backend_id="kvquant-gqa-longctx-deterministic-q23-v4",
        backend_fingerprint=hashlib.sha256(
            b"phase11-kvquant-gqa-longctx-deterministic-q23-v4"
        ).hexdigest(),
        num_layers=32,
        num_query_heads=32,
        num_kv_heads=8,
        head_dim=128,
    )


def _fixture_files(
    fixture: KVQuantFixture,
) -> dict[str, dict[str, Any]]:
    return {
        file_name: dict(
            load_fixture_tensor_file_untimed(fixture, file_name)
        )
        for file_name in _TENSOR_FILES
    }


def _tensor_nbytes(tensor: Any) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


@unittest.skipUnless(
    _authorized_environment_declared(),
    "Phase 11 KVQuant CUDA is authorized only in the exact Measurement Container",
)
class Phase11KVQuantCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.environment = require_authorized_cuda_environment(
            KVQUANT_AUTHORIZED_CONTAINER_DIGEST
        )
        cls.torch = __import__("torch")
        cls.device = cls.torch.device("cuda:0")
        cls.attention = SimpleNamespace(layer_idx=LAYER)
        cls.fixtures = {
            (fixture.family, fixture.case_name): fixture
            for fixture in load_all_kvquant_fixtures()
        }
        if set(cls.fixtures) != {
            (family, case_name)
            for family in KVQUANT_FAMILIES
            for case_name in KVQUANT_CASES
        }:
            raise AssertionError("corrected KVQuant fixture matrix differs")

    def _new_method_cache(
        self,
        family: str,
    ) -> tuple[KVQuantMethodAdapter, Any]:
        method = KVQuantMethodAdapter(_runtime_context(), family)
        method.prepare_runtime()
        cache = method.allocate(
            batch_size=1,
            capacity=TOTAL_CONTEXT,
            device=self.device,
        )
        method.initialize_cache_untimed(cache)
        return method, cache

    def _cuda_inputs(
        self,
        files: dict[str, dict[str, Any]],
    ) -> tuple[Any, Any, Any, Any, Any]:
        inputs = files["inputs.safetensors"]
        sink = files["sink.safetensors"]
        decode = files["decode_output.safetensors"]
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
        return key_pre_rope, key_attention, value, positions, query

    def _assert_exact(
        self,
        fixture: KVQuantFixture,
        file_name: str,
        tensor_name: str,
        observed: Any,
    ) -> None:
        comparison = compare_exact_fixture_tensor_untimed(
            fixture,
            file_name,
            tensor_name,
            observed,
        )
        self.assertTrue(
            comparison.passed,
            (
                f"{fixture.family}/{fixture.case_name} "
                f"{file_name}:{tensor_name} differs: {comparison}"
            ),
        )

    def _assert_metadata(
        self,
        fixture: KVQuantFixture,
        cache: Any,
        *,
        stage: str,
    ) -> None:
        expected_names = (
            ("key_codebook", cache.key_codebook[LAYER]),
            ("key_lookup_table", cache.key_lookup_table[LAYER]),
            (
                "key_runtime_lower_threshold",
                cache.key_lower_threshold[LAYER].reshape(-1),
            ),
            (
                "key_runtime_upper_threshold",
                cache.key_upper_threshold[LAYER].reshape(-1),
            ),
            (
                "key_runtime_zero",
                cache.key_zero_point[LAYER].reshape(-1),
            ),
            ("rope_inv_freq", cache.rope_inv_freq),
            ("value_codebook", cache.value_codebook[LAYER]),
            (
                f"value_lookup_after_{stage}",
                cache.value_lookup_cache[LAYER, 0],
            ),
        )
        for tensor_name, observed in expected_names:
            self._assert_exact(
                fixture,
                "metadata.safetensors",
                tensor_name,
                observed,
            )
        quantized_tokens = (
            STORE_CONTEXT - SINK_TOKENS
            if stage == "store"
            else TOTAL_CONTEXT - SINK_TOKENS
        )
        metadata = load_fixture_tensor_file_untimed(
            fixture,
            "metadata.safetensors",
        )
        for tensor_name, observed in (
            (
                "value_dense_lower_bound",
                cache.value_store_lower_bounds[0, :quantized_tokens],
            ),
            (
                "value_dense_upper_bound",
                cache.value_store_upper_bounds[0, :quantized_tokens],
            ),
        ):
            expected = metadata[tensor_name][
                SINK_TOKENS : SINK_TOKENS + quantized_tokens
            ]
            self.assertTrue(
                self.torch.equal(observed.detach().cpu(), expected),
                (
                    f"{fixture.family}/{fixture.case_name} "
                    f"metadata.safetensors:{tensor_name} differs "
                    f"after {stage}"
                ),
            )

    def _assert_fixture_byte_breakdown(
        self,
        fixture: KVQuantFixture,
        cache: Any,
    ) -> None:
        observed = {
            "dense_k_payload_bytes": _tensor_nbytes(
                cache.packed_key_cache[LAYER, 0]
            ),
            "dense_v_payload_bytes": _tensor_nbytes(
                cache.packed_value_cache[LAYER, 0]
            ),
            "key_metadata_bytes": sum(
                _tensor_nbytes(tensor)
                for tensor in (
                    cache.key_codebook[LAYER],
                    cache.key_lookup_table[LAYER],
                    cache.key_lower_threshold[LAYER],
                    cache.key_upper_threshold[LAYER],
                    cache.key_zero_point[LAYER],
                    cache.rope_inv_freq,
                )
            ),
            "value_metadata_bytes": sum(
                _tensor_nbytes(tensor)
                for tensor in (
                    cache.value_codebook[LAYER],
                    cache.value_lookup_cache[LAYER, 0],
                )
            ),
            "key_sparse_value_bytes": _tensor_nbytes(
                cache.key_sparse_values[LAYER, 0]
            ),
            "key_sparse_index_bytes": _tensor_nbytes(
                cache.key_sparse_indices[LAYER, 0]
            ),
            "value_sparse_value_bytes": _tensor_nbytes(
                cache.value_sparse_values[LAYER, 0]
            ),
            "value_sparse_index_bytes": _tensor_nbytes(
                cache.value_sparse_indices[LAYER, 0]
            ),
            "sink_k_bytes": _tensor_nbytes(cache.sink_key[LAYER]),
            "sink_v_bytes": _tensor_nbytes(cache.sink_value[LAYER]),
        }
        expected = {
            name: int(fixture.byte_breakdown[name])
            for name in observed
        }
        self.assertEqual(
            observed,
            expected,
            (
                f"{fixture.family}/{fixture.case_name} "
                "source-owned byte categories differ"
            ),
        )
        self.assertEqual(
            sum(observed.values()),
            int(fixture.byte_breakdown["actual_allocated_total_bytes"]),
        )

    def _assert_sparse_counts(
        self,
        files: dict[str, dict[str, Any]],
        cache: Any,
        *,
        quantized_tokens: int,
    ) -> None:
        declared = files["sparse_indices.safetensors"]
        for source_name, observed in (
            ("key_active_count_by_position", cache.key_active_counts[LAYER, 0]),
            (
                "value_active_count_by_position",
                cache.value_active_counts[LAYER, 0],
            ),
        ):
            expected = self.torch.zeros_like(observed)
            expected[:quantized_tokens].copy_(
                declared[source_name][
                    SINK_TOKENS : SINK_TOKENS + quantized_tokens
                ].to(device=self.device)
            )
            self.assertTrue(
                self.torch.equal(observed, expected),
                f"{source_name} differs after {quantized_tokens} tokens",
            )

    def _assert_state(
        self,
        fixture: KVQuantFixture,
        files: dict[str, dict[str, Any]],
        cache: Any,
        *,
        stage: str,
    ) -> None:
        state_file = f"{stage}_state.safetensors"
        quantized_tokens = (
            STORE_CONTEXT - SINK_TOKENS
            if stage == "store"
            else TOTAL_CONTEXT - SINK_TOKENS
        )
        expected_state = files[state_file]
        expected_length = int(expected_state["k_length"][0])
        self.assertEqual(expected_length, int(expected_state["v_length"][0]))
        self.assertEqual(
            expected_length,
            STORE_CONTEXT if stage == "store" else TOTAL_CONTEXT,
        )
        state_tensors = (
            ("k_dense_allocated", cache.packed_key_cache[LAYER, 0]),
            ("v_dense_allocated", cache.packed_value_cache[LAYER, 0]),
            ("v_lookup_allocated", cache.value_lookup_cache[LAYER, 0]),
            (
                "k_sparse_values_allocated",
                cache.key_sparse_values[LAYER, 0],
            ),
            (
                "k_sparse_indices_allocated",
                cache.key_sparse_indices[LAYER, 0],
            ),
            (
                "v_sparse_values_allocated",
                cache.value_sparse_values[LAYER, 0],
            ),
            (
                "v_sparse_indices_allocated",
                cache.value_sparse_indices[LAYER, 0],
            ),
            ("sink_k", cache.sink_key[LAYER]),
            ("sink_v", cache.sink_value[LAYER]),
        )
        for tensor_name, observed in state_tensors:
            self._assert_exact(
                fixture,
                state_file,
                tensor_name,
                observed,
            )

        suffix = "after_store" if stage == "store" else "after_append"
        dense = files["dense_payload.safetensors"]
        for tensor_name, observed in (
            (
                f"key_packed_{suffix}",
                cache.packed_key_cache[LAYER, 0, :, :, :quantized_tokens],
            ),
            (
                f"value_packed_{suffix}",
                cache.packed_value_cache[LAYER, 0, :, :, :quantized_tokens],
            ),
        ):
            self._assert_exact(
                fixture,
                "dense_payload.safetensors",
                tensor_name,
                observed,
            )
        for file_name, source_prefix, observed in (
            (
                "sparse_values.safetensors",
                "key_cache",
                cache.key_sparse_values[LAYER, 0],
            ),
            (
                "sparse_values.safetensors",
                "value_cache",
                cache.value_sparse_values[LAYER, 0],
            ),
            (
                "sparse_indices.safetensors",
                "key_cache",
                cache.key_sparse_indices[LAYER, 0],
            ),
            (
                "sparse_indices.safetensors",
                "value_cache",
                cache.value_sparse_indices[LAYER, 0],
            ),
        ):
            self._assert_exact(
                fixture,
                file_name,
                f"{source_prefix}_{suffix}",
                observed,
            )
        self._assert_sparse_counts(
            files,
            cache,
            quantized_tokens=quantized_tokens,
        )
        self._assert_metadata(fixture, cache, stage=stage)
        self._assert_fixture_byte_breakdown(fixture, cache)
        self._assert_exact(
            fixture,
            "sink.safetensors",
            "sink_key_attention_fp16",
            cache.sink_key[LAYER],
        )
        self._assert_exact(
            fixture,
            "sink.safetensors",
            "sink_value_fp16",
            cache.sink_value[LAYER],
        )
        if stage == "append":
            self._assert_exact(
                fixture,
                "dense_payload.safetensors",
                "key_appended_slot",
                cache.packed_key_cache[LAYER, 0, :, :, quantized_tokens - 1],
            )
            self._assert_exact(
                fixture,
                "dense_payload.safetensors",
                "value_appended_slot",
                cache.packed_value_cache[LAYER, 0, :, :, quantized_tokens - 1],
            )

    def _prefill_append_decode(
        self,
        fixture: KVQuantFixture,
        files: dict[str, dict[str, Any]],
    ) -> tuple[Any, Any, dict[str, int], dict[str, int]]:
        method, cache = self._new_method_cache(fixture.family)
        key_pre_rope, key_attention, value, positions, query = self._cuda_inputs(
            files
        )
        pointers_before = cache.pointers()
        cache.prepare_prefill(STORE_CONTEXT)
        method.store_prefill(
            cache,
            key_attention[:, :, :STORE_CONTEXT, :],
            value[:, :, :STORE_CONTEXT, :],
            LAYER,
            positions[:STORE_CONTEXT],
            key_pre_rope_states=key_pre_rope[:, :, :STORE_CONTEXT, :],
        )
        cache.complete_prefill()
        self._assert_state(
            fixture,
            files,
            cache,
            stage="store",
        )
        append_position = positions[STORE_CONTEXT:TOTAL_CONTEXT]
        cache.bind_fixed_position_tensor_untimed(
            append_position,
            logical_position=STORE_CONTEXT,
        )
        cache.prepare_fixed(STORE_CONTEXT)
        handles = method.append_decode(
            cache,
            key_attention[:, :, STORE_CONTEXT:TOTAL_CONTEXT, :],
            value[:, :, STORE_CONTEXT:TOTAL_CONTEXT, :],
            LAYER,
            append_position,
            key_pre_rope_states=key_pre_rope[
                :, :, STORE_CONTEXT:TOTAL_CONTEXT, :
            ],
        )
        output = method.decode_attention(
            self.attention,
            query,
            handles[0],
            handles[1],
            scaling=SCALING,
        )
        self.torch.cuda.synchronize(device=self.device)
        self._assert_state(
            fixture,
            files,
            cache,
            stage="append",
        )
        decode = compare_decode_output_untimed(fixture, output)
        self.assertTrue(decode.passed, decode)
        self.assertTrue(bool(self.torch.isfinite(output).all()))
        self.assertEqual(cache.active_context, STORE_CONTEXT)
        return cache, output, pointers_before, cache.pointers()

    def test_all_nine_corrected_fixtures_conform_through_adapter(self) -> None:
        self.assertEqual(len(self.fixtures), 9)
        for family in KVQUANT_FAMILIES:
            for case_name in KVQUANT_CASES:
                with self.subTest(family=family, case=case_name):
                    fixture = self.fixtures[(family, case_name)]
                    files = _fixture_files(fixture)
                    cache, output, pointers_before, pointers_after = (
                        self._prefill_append_decode(fixture, files)
                    )
                    self.assertEqual(pointers_after, pointers_before)
                    if family == "kvq4":
                        self.assertEqual(
                            tuple(cache.q4_value_decode_workspace.shape),
                            cache.q4_value_decode_workspace_shape,
                        )
                    else:
                        self.assertIsNone(
                            cache.q4_value_decode_workspace
                        )
                        self.assertIsNotNone(
                            cache.q23_value_decode_workspace
                        )
                    self.assertEqual(
                        cache.gqa_geometry(),
                        {
                            "num_query_heads": 32,
                            "num_kv_heads": 8,
                            "gqa_group_size": 4,
                            "native_kv_head_storage": True,
                            "query_head_sized_kv_cache": False,
                        },
                    )
                    self.assertEqual(
                        cache.key_active_counts[
                            LAYER, 0, : TOTAL_CONTEXT - SINK_TOKENS
                        ].unique().cpu().tolist(),
                        [fixture.key_active_count],
                    )
                    self.assertEqual(
                        cache.value_active_counts[
                            LAYER, 0, : TOTAL_CONTEXT - SINK_TOKENS
                        ].unique().cpu().tolist(),
                        [12],
                    )
                    del cache, files, output
                    gc.collect()
                    self.torch.cuda.empty_cache()

    def test_non_default_stream_orders_store_append_and_decode(self) -> None:
        fixture = self.fixtures[("kvq4", "key_cap_value_fixed12")]
        files = _fixture_files(fixture)
        method, cache = self._new_method_cache(fixture.family)
        source = self._cuda_inputs(files)
        self.torch.cuda.synchronize(device=self.device)
        key_pre_rope = self.torch.zeros_like(source[0])
        key_attention = self.torch.zeros_like(source[1])
        value = self.torch.zeros_like(source[2])
        positions = source[3]
        query = source[4]
        append_position = positions[STORE_CONTEXT:TOTAL_CONTEXT]
        cache.bind_fixed_position_tensor_untimed(
            append_position,
            logical_position=STORE_CONTEXT,
        )
        stream = self.torch.cuda.Stream(device=self.device)
        complete = self.torch.cuda.Event()
        pointers_before = cache.pointers()
        with self.torch.cuda.stream(stream):
            sleep = getattr(self.torch.cuda, "_sleep", None)
            if callable(sleep):
                sleep(5_000_000)
            key_pre_rope.copy_(source[0])
            key_attention.copy_(source[1])
            value.copy_(source[2])
            cache.prepare_prefill(STORE_CONTEXT)
            method.store_prefill(
                cache,
                key_attention[:, :, :STORE_CONTEXT, :],
                value[:, :, :STORE_CONTEXT, :],
                LAYER,
                positions[:STORE_CONTEXT],
                key_pre_rope_states=key_pre_rope[:, :, :STORE_CONTEXT, :],
            )
            cache.complete_prefill()
            cache.prepare_fixed(STORE_CONTEXT)
            handles = method.append_decode(
                cache,
                key_attention[:, :, STORE_CONTEXT:TOTAL_CONTEXT, :],
                value[:, :, STORE_CONTEXT:TOTAL_CONTEXT, :],
                LAYER,
                append_position,
                key_pre_rope_states=key_pre_rope[
                    :, :, STORE_CONTEXT:TOTAL_CONTEXT, :
                ],
            )
            output = method.decode_attention(
                self.attention,
                query,
                handles[0],
                handles[1],
                scaling=SCALING,
            )
            complete.record(stream)
        self.torch.cuda.current_stream(device=self.device).wait_event(complete)
        self.torch.cuda.synchronize(device=self.device)
        self._assert_state(fixture, files, cache, stage="append")
        comparison = compare_decode_output_untimed(fixture, output)
        self.assertTrue(comparison.passed, comparison)
        self.assertEqual(cache.pointers(), pointers_before)
        self.assertNotEqual(
            int(stream.cuda_stream),
            int(self.torch.cuda.default_stream(self.device).cuda_stream),
        )
        extension_path = Path(
            os.environ["KVBENCH_KVQUANT_EXTENSION"]
        ).resolve(strict=True)
        self.assertEqual(
            hashlib.sha256(extension_path.read_bytes()).hexdigest(),
            KVQUANT_EXTENSION_SHA256,
        )

    def test_value_tie_control_replays_caller_owned_selector(self) -> None:
        fixture = self.fixtures[("kvq3", "key_zero_value_fixed12")]
        files = _fixture_files(fixture)
        inputs = files["inputs.safetensors"]
        rows = inputs["value_tie_control_rows"].to(
            device=self.device,
            dtype=self.torch.float32,
        )
        sink_mask = inputs["value_tie_control_sink_mask"].to(
            device=self.device
        )
        sparse_values = self.torch.empty(
            (rows.shape[0], 12),
            dtype=self.torch.float32,
            device=self.device,
        )
        sparse_indices = self.torch.empty(
            (rows.shape[0], 12),
            dtype=self.torch.int32,
            device=self.device,
        )
        active_counts = self.torch.empty(
            (rows.shape[0],),
            dtype=self.torch.int32,
            device=self.device,
        )
        dense_lower = self.torch.empty(
            (rows.shape[0],),
            dtype=self.torch.float32,
            device=self.device,
        )
        dense_upper = self.torch.empty_like(dense_lower)
        dummy_thresholds = self.torch.zeros(
            (rows.shape[1],),
            dtype=self.torch.float32,
            device=self.device,
        )
        method = KVQuantMethodAdapter(_runtime_context(), "kvq3")
        method.prepare_runtime()
        outputs = (
            sparse_values,
            sparse_indices,
            active_counts,
            dense_lower,
            dense_upper,
        )
        pointers_before = tuple(tensor.data_ptr() for tensor in outputs)
        method._runtime().select_fixed_outliers_1024_cap12_out(
            rows,
            dummy_thresholds,
            dummy_thresholds,
            sink_mask,
            sparse_values,
            sparse_indices,
            active_counts,
            dense_lower,
            dense_upper,
            1,
        )
        self.torch.cuda.synchronize(device=self.device)
        self.assertEqual(
            tuple(tensor.data_ptr() for tensor in outputs),
            pointers_before,
        )
        self._assert_exact(
            fixture,
            "sparse_values.safetensors",
            "value_tie_control",
            sparse_values,
        )
        self._assert_exact(
            fixture,
            "sparse_indices.safetensors",
            "value_tie_control",
            sparse_indices,
        )
        self._assert_exact(
            fixture,
            "sparse_indices.safetensors",
            "value_tie_control_counts",
            active_counts,
        )


if __name__ == "__main__":
    unittest.main()
