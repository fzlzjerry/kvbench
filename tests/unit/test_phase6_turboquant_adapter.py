"""Focused CPU/static tests for the minimal Phase 6 TurboQuant adapter."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock
import weakref

from kvbench.adapters import (
    BF16MethodAdapter,
    KVCacheMethod,
    MethodRuntimeContext,
    TurboQuantMethodAdapter,
    build_method_adapter,
)
from kvbench.config import load_config
from kvbench.errors import ConfigLoadError, PhaseNotImplementedError
from kvbench.runtime.static_cache import CacheStateError
from kvbench.runtime.turboquant_admission import (
    release_fixture_cuda_resources_for_sanitizer,
)
from kvbench.runtime.turboquant_cache import (
    TURBOQUANT_BF16_LAYERS,
    TURBOQUANT_COMPRESSED_LAYERS,
    TURBOQUANT_SLOT_SIZES,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANDATORY = {
    "turboquant_4bit_nc": 134,
    "turboquant_k3v4_nc": 118,
    "turboquant_3bit_nc": 102,
}


def _context(*, head_dim: int = 128) -> MethodRuntimeContext:
    return MethodRuntimeContext(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        model_revision="0" * 40,
        backend_id="pytorch_flash",
        backend_fingerprint="1" * 64,
        num_layers=32,
        num_query_heads=32,
        num_kv_heads=8,
        head_dim=head_dim,
    )


class TurboQuantAdapterTests(unittest.TestCase):
    def test_explicit_factory_mapping_and_fail_closed_methods(self) -> None:
        for config_name in MANDATORY:
            with self.subTest(config_name=config_name):
                adapter = build_method_adapter(config_name, _context())
                self.assertIsInstance(adapter, TurboQuantMethodAdapter)
                self.assertIsInstance(adapter, KVCacheMethod)
                self.assertEqual(adapter.config_name, config_name)
        method_config = load_config(
            REPOSITORY_ROOT / "configs/methods/turboquant.yaml"
        )
        adapter = build_method_adapter(
            method_config,
            _context(),
            variant_id="turboquant_4bit_nc",
        )
        self.assertEqual(adapter.config_name, "turboquant_4bit_nc")
        for rejected in ("turboquant", "turboquant_k8v4", "unknown"):
            with self.subTest(rejected=rejected):
                with self.assertRaises(ConfigLoadError):
                    build_method_adapter(rejected, _context())
        for deferred in ("kivi", "kvquant"):
            with self.subTest(deferred=deferred):
                with self.assertRaises(PhaseNotImplementedError):
                    build_method_adapter(deferred, _context())

    def test_static_layout_accounting_and_fingerprints(self) -> None:
        expected_layers = set(range(32))
        self.assertEqual(TURBOQUANT_BF16_LAYERS, (0, 1, 30, 31))
        self.assertEqual(TURBOQUANT_COMPRESSED_LAYERS, tuple(range(2, 30)))
        self.assertEqual(
            set(TURBOQUANT_BF16_LAYERS)
            | set(TURBOQUANT_COMPRESSED_LAYERS),
            expected_layers,
        )
        self.assertFalse(
            set(TURBOQUANT_BF16_LAYERS)
            & set(TURBOQUANT_COMPRESSED_LAYERS)
        )
        for config_name, slot_size in MANDATORY.items():
            with self.subTest(config_name=config_name):
                adapter = build_method_adapter(config_name, _context())
                cache = adapter.allocate(
                    batch_size=1,
                    capacity=17,
                    device="cpu",
                )
                self.assertEqual(cache.block_size, 16)
                self.assertEqual(cache.block_count, 2)
                self.assertEqual(cache.rounded_capacity, 32)
                self.assertEqual(cache.block_table.tolist(), [[0, 1]])
                self.assertEqual(cache.slot_mapping.tolist(), list(range(32)))
                self.assertEqual(cache.slot_size, slot_size)
                self.assertEqual(TURBOQUANT_SLOT_SIZES[config_name], slot_size)
                expected_packed = 28 * 32 * 8 * slot_size
                expected_bf16 = 2 * 4 * 1 * 8 * 32 * 128 * 2
                self.assertEqual(
                    cache.packed_cache.untyped_storage().nbytes(),
                    expected_packed,
                )
                self.assertEqual(
                    cache.bf16_cache.keys.untyped_storage().nbytes()
                    + cache.bf16_cache.values.untyped_storage().nbytes(),
                    expected_bf16,
                )
                breakdown = cache.byte_breakdown()
                accounting = cache.accounting()
                self.assertEqual(sum(breakdown.values()), accounting.allocated_bytes)
                self.assertEqual(
                    accounting.predicted_tensor_bytes,
                    accounting.allocated_bytes,
                )
                self.assertEqual(
                    accounting.measured_tensor_bytes,
                    accounting.allocated_bytes,
                )
                self.assertEqual(accounting.temporary_peak_bytes, 0)
                self.assertIsNone(cache.r_hbm)
                first_layout = cache.layout_fingerprint()
                second_layout = cache.layout_fingerprint()
                self.assertEqual(first_layout, second_layout)
                self.assertEqual(
                    adapter.config_fingerprint(first_layout),
                    adapter.config_fingerprint(second_layout),
                )
                self.assertEqual(cache.pointers(), cache.pointers())

    def test_sanitizer_release_breaks_all_tensor_owners(self) -> None:
        method = build_method_adapter("turboquant_4bit_nc", _context())
        cache = method.allocate(
            batch_size=1,
            capacity=18,
            device="cpu",
        )
        owned_references = tuple(
            weakref.ref(tensor) for tensor in cache._owned_tensors()
        )
        handle = cache.attended_handle(
            2,
            key_states=cache.store_key_float,
            value_states=cache.store_value_float,
            prefill=False,
        )

        cache.release_owned_cuda_resources_for_sanitizer()
        cache.release_owned_cuda_resources_for_sanitizer()
        gc.collect()

        self.assertEqual(cache.mode, "released")
        self.assertIsNone(cache.bf16_cache)
        self.assertEqual(cache._handles, {})
        self.assertEqual(cache._single_slot_mappings, ())
        self.assertEqual(cache._single_seq_lens, ())
        self.assertIsNone(cache._current_slot_mapping)
        self.assertIsNone(cache._current_seq_lens)
        self.assertIsNone(handle.key_states)
        self.assertIsNone(handle.value_states)
        self.assertTrue(all(reference() is None for reference in owned_references))

    def test_sanitizer_release_zeroes_shared_storage_aliases(self) -> None:
        method = build_method_adapter("turboquant_4bit_nc", _context())
        cache = method.allocate(
            batch_size=1,
            capacity=18,
            device="cpu",
        )
        alias = cache.packed_cache.view(-1)
        source = cache._sanitizer_hadamard_source()
        self.assertIsNotNone(source)
        retained = (*cache._owned_tensors(), alias, source)

        cache.release_owned_cuda_resources_for_sanitizer()

        self.assertTrue(
            all(
                int(tensor.untyped_storage().nbytes()) == 0
                for tensor in retained
            )
        )

    def test_fixture_release_zeroes_inputs_and_positions(self) -> None:
        method = build_method_adapter("turboquant_4bit_nc", _context())
        cache = method.allocate(
            batch_size=1,
            capacity=18,
            device="cpu",
        )
        fixture_input = cache.packed_cache.new_empty((5,))
        position = cache.block_table.new_empty((1,))
        fixture_alias = fixture_input.view(-1)
        position_alias = position.view(-1)
        resources = {
            "inputs": {"fixture": fixture_input},
            "positions": [position],
            "cache": cache,
        }

        release_fixture_cuda_resources_for_sanitizer(resources)

        self.assertEqual(resources, {})
        self.assertEqual(
            int(fixture_alias.untyped_storage().nbytes()),
            0,
        )
        self.assertEqual(
            int(position_alias.untyped_storage().nbytes()),
            0,
        )
        self.assertEqual(cache.mode, "released")

    def test_attention_handles_do_not_own_the_cache(self) -> None:
        method = build_method_adapter("turboquant_4bit_nc", _context())
        cache = method.allocate(
            batch_size=1,
            capacity=18,
            device="cpu",
        )
        handle = cache.attended_handle(
            2,
            key_states=None,
            value_states=None,
            prefill=False,
        )
        cache_reference = weakref.ref(cache)
        del cache
        gc.collect()
        self.assertIsNone(cache_reference())
        with self.assertRaises(ReferenceError):
            _ = handle.cache.config_name

    def test_prefill_attention_handle_is_one_shot(self) -> None:
        method = build_method_adapter("turboquant_4bit_nc", _context())
        cache = method.allocate(
            batch_size=1,
            capacity=18,
            device="cpu",
        )
        attention = SimpleNamespace(layer_idx=2)
        query_states = object()

        key_states = object()
        value_states = object()
        handle = cache.attended_handle(
            2,
            key_states=key_states,
            value_states=value_states,
            prefill=True,
        )
        output = object()
        with mock.patch(
            "kvbench.adapters.turboquant.flash_attention_forward",
            return_value=(output, None),
        ) as flash:
            self.assertIs(
                method.decode_attention(
                    attention,
                    query_states,
                    handle,
                    handle,
                    scaling=0.125,
                ),
                output,
            )
            flash.assert_called_once_with(
                attention,
                query_states,
                key_states,
                value_states,
                None,
                0.125,
                dropout=0.0,
            )
        self.assertIs(cache._handles[2], handle)
        self.assertIsNone(handle.key_states)
        self.assertIsNone(handle.value_states)
        self.assertFalse(handle.prefill)

        key_states = object()
        value_states = object()
        handle = cache.attended_handle(
            2,
            key_states=key_states,
            value_states=value_states,
            prefill=True,
        )
        with mock.patch(
            "kvbench.adapters.turboquant.flash_attention_forward",
            side_effect=RuntimeError("forced Flash failure"),
        ) as flash:
            with self.assertRaisesRegex(
                RuntimeError,
                "forced Flash failure",
            ):
                method.decode_attention(
                    attention,
                    query_states,
                    handle,
                    handle,
                    scaling=0.125,
                )
            flash.assert_called_once_with(
                attention,
                query_states,
                key_states,
                value_states,
                None,
                0.125,
                dropout=0.0,
            )
        self.assertIs(cache._handles[2], handle)
        self.assertIsNone(handle.key_states)
        self.assertIsNone(handle.value_states)
        self.assertFalse(handle.prefill)

        key_states = object()
        handle = cache.attended_handle(
            2,
            key_states=key_states,
            value_states=None,
            prefill=True,
        )
        with self.assertRaisesRegex(
            CacheStateError,
            "TurboQuant prefill handle lost raw K/V",
        ):
            method.decode_attention(
                attention,
                query_states,
                handle,
                handle,
                scaling=0.125,
            )
        self.assertIs(cache._handles[2], handle)
        self.assertIsNone(handle.key_states)
        self.assertIsNone(handle.value_states)
        self.assertFalse(handle.prefill)

    def test_unsupported_geometry_and_wrong_variant_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TurboQuantMethodAdapter(_context(head_dim=64), "turboquant_4bit_nc")
        with self.assertRaises(ValueError):
            TurboQuantMethodAdapter(_context(), "turboquant_k8v4")
        with self.assertRaises(ValueError):
            build_method_adapter("turboquant_4bit_nc", _context()).allocate(
                batch_size=2,
                capacity=17,
                device="cpu",
            )

    def test_bf16_factory_and_allocation_are_unchanged(self) -> None:
        adapter = build_method_adapter("bf16", _context())
        self.assertIsInstance(adapter, BF16MethodAdapter)
        cache = adapter.allocate(
            batch_size=1,
            capacity=17,
            device="cpu",
            workspace_bytes=4096,
        )
        self.assertEqual(
            adapter.allocated_bytes(cache),
            2 * 32 * 1 * 8 * 17 * 128 * 2 + 4096,
        )

    def test_pinned_source_provenance_and_kernel_bodies(self) -> None:
        root = (
            REPOSITORY_ROOT
            / "src/kvbench/third_party/vllm_turboquant"
        )
        manifest = json.loads((root / "provenance.json").read_text())
        self.assertFalse(manifest["algorithmic_source_modified"])
        self.assertEqual(
            manifest["authority"]["commit"],
            "752a3a504485790a2e8491cacbb35c137339ad34",
        )
        for record in manifest["files"]:
            carried = REPOSITORY_ROOT / record["carried_path"]
            self.assertEqual(
                hashlib.sha256(carried.read_bytes()).hexdigest(),
                record["carried_sha256"],
            )
        adapter_source = (
            REPOSITORY_ROOT / "src/kvbench/adapters/turboquant.py"
        ).read_text()
        cache_source = (
            REPOSITORY_ROOT / "src/kvbench/runtime/turboquant_cache.py"
        ).read_text()
        self.assertNotIn("from kvbench.third_party", adapter_source)
        self.assertNotIn("torch.cat", adapter_source)
        self.assertNotIn("repeat_kv", adapter_source)
        self.assertNotIn("repeat_interleave", adapter_source)
        self.assertNotIn(".tolist(", adapter_source)
        self.assertIn("self._store_kernel", cache_source)
        self.assertIn("self._decode_stage1_kernel", cache_source)
        self.assertIn("self._decode_stage2_kernel", cache_source)


if __name__ == "__main__":
    unittest.main()
