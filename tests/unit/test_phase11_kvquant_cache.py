"""Focused static-layout tests for the Phase 11 KVQuant cache state."""

from __future__ import annotations

import gc
from pathlib import Path
import unittest

import torch

from kvbench.runtime.kvquant_cache import (
    KVQUANT_CONFIG_BITS,
    KVQUANT_HEAD_DIM,
    KVQUANT_KEY_CAP,
    KVQUANT_NUM_KV_HEADS,
    KVQUANT_NUM_LAYERS,
    KVQUANT_NUM_QUERY_HEADS,
    kvquant_q4_value_decode_workspace_shape,
    KVQUANT_SINK_TOKENS,
    KVQUANT_VALUE_CAP,
    KVQuantStaticCache,
)
from kvbench.runtime.static_cache import CacheBoundsError, CacheStateError


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _cache(
    *,
    config_name: str = "kvq4",
    capacity: int = 18,
    workspace_bytes: int = 0,
) -> KVQuantStaticCache:
    return KVQuantStaticCache(
        config_name=config_name,
        num_layers=KVQUANT_NUM_LAYERS,
        batch_size=1,
        num_query_heads=KVQUANT_NUM_QUERY_HEADS,
        num_kv_heads=KVQUANT_NUM_KV_HEADS,
        capacity=capacity,
        head_dim=KVQUANT_HEAD_DIM,
        device="cpu",
        workspace_bytes=workspace_bytes,
    )


class KVQuantStaticCacheTests(unittest.TestCase):
    def test_frozen_configurations_and_geometry_fail_closed(self) -> None:
        self.assertEqual(KVQUANT_CONFIG_BITS, {"kvq4": 4, "kvq3": 3, "kvq2": 2})
        for rejected in ("kvq1", "kvq5", "kvquant"):
            with self.subTest(rejected=rejected):
                with self.assertRaisesRegex(ValueError, "unsupported KVQuant"):
                    _cache(config_name=rejected)
        with self.assertRaisesRegex(ValueError, "frozen"):
            KVQuantStaticCache(
                config_name="kvq4",
                num_layers=31,
                batch_size=1,
                num_query_heads=32,
                num_kv_heads=8,
                capacity=18,
                head_dim=128,
                device="cpu",
            )
        with self.assertRaisesRegex(ValueError, "five sink"):
            _cache(capacity=4)

    def test_native_hkv_layout_dtypes_and_required_regions(self) -> None:
        for family, bits in KVQUANT_CONFIG_BITS.items():
            with self.subTest(family=family):
                cache = _cache(config_name=family)
                packed_rows = bits * 128 // 32
                self.assertEqual(
                    cache.packed_key_cache.shape,
                    (32, 1, 8, packed_rows, 18),
                )
                self.assertEqual(
                    cache.packed_value_cache.shape,
                    cache.packed_key_cache.shape,
                )
                self.assertEqual(
                    cache.key_lookup_table.shape,
                    (32, 8, 128, 1 << bits),
                )
                self.assertEqual(
                    cache.value_lookup_cache.shape,
                    (32, 1, 18, 1 << bits),
                )
                self.assertEqual(cache.key_sparse_values.shape, (32, 1, 18, 12))
                self.assertEqual(cache.value_sparse_values.shape, (32, 1, 18, 12))
                self.assertEqual(cache.key_active_counts.shape, (32, 1, 18))
                self.assertEqual(cache.value_active_counts.shape, (32, 1, 18))
                self.assertEqual(cache.sink_key.shape, (32, 1, 8, 128, 5))
                self.assertEqual(cache.sink_value.shape, (32, 1, 8, 5, 128))
                self.assertEqual(cache.payload_slot_for_position(5), 0)
                self.assertTrue(cache.is_sink_position(4))
                self.assertFalse(cache.is_sink_position(5))
                self.assertEqual(
                    cache.key_pre_rope_bf16_staging.shape,
                    (1, 8, 1, 128),
                )
                self.assertEqual(cache.key_float_staging.shape, (1, 1024))
                self.assertEqual(
                    cache.value_store_lower_bounds.shape,
                    (1, 18),
                )
                self.assertEqual(
                    cache.value_store_upper_bounds.shape,
                    (1, 18),
                )
                self.assertEqual(cache.key_selector_lower.shape, (1024,))
                self.assertEqual(cache.key_selector_upper.shape, (1024,))
                self.assertEqual(float(cache.key_selector_lower[0]), -1.0)
                self.assertEqual(float(cache.key_selector_upper[1023]), 1.0)
                self.assertEqual(cache.query_float_staging.shape, (1, 32, 128))
                self.assertEqual(cache.decode_logits.shape, (1, 32, 18))
                self.assertEqual(cache.decode_logits_bf16.shape, (1, 32, 18))
                self.assertEqual(cache.sink_logits_fp16.shape, (1, 32, 5))
                self.assertEqual(cache.sink_output_fp16.shape, (1, 32, 128))
                if family == "kvq4":
                    self.assertIsNotNone(cache.q4_value_decode_workspace)
                    self.assertEqual(
                        tuple(cache.q4_value_decode_workspace.shape),
                        kvquant_q4_value_decode_workspace_shape(
                            batch_size=1,
                            total_attended_capacity=18,
                        ),
                    )
                    self.assertEqual(
                        str(cache.q4_value_decode_workspace.dtype),
                        "torch.float32",
                    )
                    self.assertTrue(
                        cache.q4_value_decode_workspace.is_contiguous()
                    )
                    self.assertEqual(
                        cache.storage_geometry()[
                            "q4_value_decode_workspace"
                        ],
                        kvquant_q4_value_decode_workspace_shape(
                            batch_size=1,
                            total_attended_capacity=18,
                        ),
                    )
                else:
                    self.assertIsNone(cache.q4_value_decode_workspace)
                    self.assertIsNotNone(cache.q23_value_decode_workspace)
                    self.assertNotIn(
                        "q4_value_decode_workspace",
                        cache.storage_geometry(),
                    )
                self.assertEqual(str(cache.packed_key_cache.dtype), "torch.int32")
                self.assertEqual(str(cache.key_sparse_values.dtype), "torch.float32")
                self.assertEqual(str(cache.key_sparse_indices.dtype), "torch.int32")
                self.assertEqual(str(cache.sink_key.dtype), "torch.float16")
                self.assertEqual(
                    str(cache.decode_logits_bf16.dtype),
                    "torch.bfloat16",
                )
                self.assertEqual(
                    str(cache.sink_logits_fp16.dtype),
                    "torch.float16",
                )
                self.assertEqual(
                    str(cache.sink_output_fp16.dtype),
                    "torch.float16",
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
                    cache.storage_geometry()["dense_k"],
                    (32, 1, 8, packed_rows, 18),
                )
                self.assertEqual(
                    cache.storage_geometry()["sink_logits_fp16"],
                    (1, 32, 5),
                )
                del cache
                gc.collect()

    def test_q3_q2_workspace_reuses_owned_logits_storage(self) -> None:
        for family in ("kvq3", "kvq2"):
            with self.subTest(family=family):
                cache = _cache(config_name=family, capacity=257)
                workspace = cache.q23_value_decode_workspace
                self.assertIsNotNone(workspace)
                assert workspace is not None
                self.assertEqual(tuple(workspace.shape), (1, 32, 1, 128))
                self.assertEqual(
                    workspace.untyped_storage().data_ptr(),
                    cache.decode_logits.untyped_storage().data_ptr(),
                )
                self.assertEqual(
                    cache.layout_fingerprint(),
                    cache.layout_fingerprint(),
                )
                self.assertEqual(
                    cache.byte_breakdown(),
                    cache.predicted_byte_breakdown(),
                )
                self.assertIn(
                    "q23_value_decode_workspace_data_ptr",
                    cache.pointers(),
                )

    def test_physical_accounting_is_exact_for_all_families(self) -> None:
        for family in KVQUANT_CONFIG_BITS:
            with self.subTest(family=family):
                cache = _cache(
                    config_name=family,
                    capacity=128,
                    workspace_bytes=257,
                )
                observed = cache.byte_breakdown()
                predicted = cache.predicted_byte_breakdown()
                accounting = cache.accounting()
                self.assertEqual(observed, predicted)
                self.assertEqual(sum(observed.values()), accounting.allocated_bytes)
                self.assertEqual(
                    accounting.predicted_tensor_bytes,
                    accounting.measured_tensor_bytes,
                )
                self.assertLess(accounting.relative_error, 0.01)
                self.assertEqual(accounting.temporary_peak_bytes, 0)
                self.assertEqual(
                    observed["persistent_workspace"],
                    (
                        cache.decode_logits.untyped_storage().nbytes()
                        + cache.decode_logits_bf16.untyped_storage().nbytes()
                        + cache.decode_softmax.untyped_storage().nbytes()
                        + cache.sink_logits_fp16.untyped_storage().nbytes()
                        + cache.decode_merge.untyped_storage().nbytes()
                        + cache.decode_sparse_correction.untyped_storage().nbytes()
                        + cache.decode_sink_contribution.untyped_storage().nbytes()
                        + cache.decode_quantized_output.untyped_storage().nbytes()
                        + cache.sink_output_fp16.untyped_storage().nbytes()
                        + 257
                    ),
                )
                if family == "kvq4":
                    self.assertEqual(
                        observed["q4_value_decode_workspace"],
                        cache.q4_value_decode_workspace_bytes,
                    )
                else:
                    self.assertNotIn("q4_value_decode_workspace", observed)
                self.assertEqual(
                    accounting.workspace_bytes,
                    observed["persistent_workspace"]
                    + observed.get("q4_value_decode_workspace", 0),
                )
                ratios = cache.ratios()
                self.assertLessEqual(ratios.reciprocal_error, 1e-9)
                self.assertLessEqual(
                    abs(ratios.rho_alloc * ratios.r_alloc - 1.0),
                    1e-9,
                )
                self.assertIsNone(ratios.r_hbm)
                self.assertIsNone(cache.r_hbm)
                del cache
                gc.collect()

    def test_required_context_accounting_and_active_sparse_semantics(self) -> None:
        for context in (5, 17, 18, 128, 4096):
            with self.subTest(context=context):
                cache = _cache(config_name="kvq4", capacity=context)
                nonsink_rows = 32 * max(0, context - 5)
                key_entries = nonsink_rows * 6
                active = cache.active_byte_breakdown(
                    context,
                    key_active_entries=key_entries,
                )
                dense_each = 32 * 8 * 128 * max(0, context - 5) * 4 // 8
                self.assertEqual(active["dense_k_payload"], dense_each)
                self.assertEqual(active["dense_v_payload"], dense_each)
                self.assertEqual(active["key_sparse_values"], key_entries * 4)
                self.assertEqual(active["key_sparse_indices"], key_entries * 4)
                self.assertEqual(
                    active["value_sparse_values"],
                    nonsink_rows * KVQUANT_VALUE_CAP * 4,
                )
                self.assertEqual(
                    active["value_sparse_indices"],
                    nonsink_rows * KVQUANT_VALUE_CAP * 4,
                )
                self.assertEqual(
                    cache.active_storage_bytes(
                        context,
                        key_active_entries=key_entries,
                    ),
                    sum(active.values()),
                )
                accounting = cache.accounting()
                self.assertLess(accounting.relative_error, 0.01)
                ratios = cache.ratios()
                self.assertLessEqual(
                    abs(ratios.rho_alloc * ratios.r_alloc - 1.0),
                    1e-9,
                )
                self.assertEqual(
                    cache.active_logical_bf16_bytes(context),
                    2 * 32 * 1 * 8 * context * 128 * 2,
                )
                del cache
                gc.collect()

    def test_exact_active_key_count_is_fail_closed(self) -> None:
        cache = _cache()
        cache.begin_prefill()
        cache.finish_prefill(17)
        with self.assertRaisesRegex(CacheStateError, "exact active Key"):
            cache.active_storage_bytes()
        cache.record_key_active_entries(32 * 12 * 6)
        self.assertGreater(cache.active_storage_bytes(), 0)
        with self.assertRaisesRegex(CacheBoundsError, "fixed sparse capacity"):
            cache.record_key_active_entries(32 * 12 * 12 + 1)

    def test_lifecycle_slots_are_static_ordered_and_bounds_checked(self) -> None:
        cache = _cache(capacity=22)
        cache.begin_prefill()
        cache.finish_prefill(17, key_active_entries=32 * 12 * 6)
        self.assertEqual(cache.active_context, 17)

        fixed_position = torch.tensor([17], dtype=torch.int64)
        cache.bind_fixed_position_tensor_untimed(
            fixed_position,
            logical_position=17,
        )
        cache.begin_fixed(17)
        self.assertEqual(cache.fixed_slot(0), 12)
        cache.packed_key_cache[0, 0, 0, 0, 12].fill_(7)
        cache.value_sparse_indices[0, 0, 12].fill_(7)
        self.assertEqual(cache.fixed_scratch_overwrite(layer_idx=0), 12)
        self.assertEqual(
            int(cache.packed_key_cache[0, 0, 0, 0, 12]),
            0,
        )
        self.assertEqual(
            int(cache.value_sparse_indices[0, 0, 12].count_nonzero()),
            0,
        )
        self.assertEqual(cache.active_context, 17)
        with self.assertRaises(CacheBoundsError):
            cache.fixed_slot(32)

        cache.reset_active_length(17, key_active_entries=32 * 12 * 6)
        growing_positions = tuple(
            torch.tensor([position], dtype=torch.int64)
            for position in range(17, 21)
        )
        cache.bind_growing_position_tensors_untimed(
            growing_positions,
            starting_position=17,
        )
        cache.begin_growing(17, 4)
        for step in range(4):
            cache.select_growing_step(step)
            self.assertEqual(cache.growing_slot(31), 12 + step)
            cache.validate_decode_position_binding(
                growing_positions[step],
                payload_slot=12 + step,
            )
            cache.growing_scratch_overwrite(layer_idx=31)
            cache.commit_growing(key_active_entries_added=32 * 6)
            self.assertEqual(cache.active_context, 18 + step)
        cache.reset_growing()
        self.assertEqual(cache.mode, "ready")
        with self.assertRaisesRegex(CacheBoundsError, "trajectory"):
            cache.begin_growing(21, 2)

    def test_decode_position_binding_rejects_identity_and_slot_drift(self) -> None:
        cache = _cache(capacity=22)
        cache.begin_prefill()
        cache.finish_prefill(17, key_active_entries=0)
        fixed_position = torch.tensor([17], dtype=torch.int64)
        cache.bind_fixed_position_tensor_untimed(
            fixed_position,
            logical_position=17,
        )
        cache.begin_fixed(17)
        cache.validate_decode_position_binding(
            fixed_position,
            payload_slot=12,
        )
        with self.assertRaisesRegex(CacheStateError, "physical slot"):
            cache.validate_decode_position_binding(
                torch.tensor([17], dtype=torch.int64),
                payload_slot=12,
            )
        with self.assertRaisesRegex(CacheStateError, "physical slot"):
            cache.validate_decode_position_binding(
                fixed_position,
                payload_slot=13,
            )

        unbound = _cache(capacity=22)
        unbound.begin_prefill()
        unbound.finish_prefill(17, key_active_entries=0)
        with self.assertRaisesRegex(CacheStateError, "not bound"):
            unbound.begin_fixed(17)

        growing = _cache(capacity=22)
        growing.begin_prefill()
        growing.finish_prefill(17, key_active_entries=0)
        positions = tuple(
            torch.tensor([position], dtype=torch.int64)
            for position in range(17, 21)
        )
        growing.bind_growing_position_tensors_untimed(
            positions,
            starting_position=17,
        )
        growing.begin_growing(17, 4)
        growing.select_growing_step(0)
        growing.validate_decode_position_binding(
            positions[0],
            payload_slot=12,
        )
        growing.commit_growing(key_active_entries_added=0)
        growing.select_growing_step(1)
        growing.validate_decode_position_binding(
            positions[1],
            payload_slot=13,
        )
        with self.assertRaisesRegex(CacheStateError, "physical slot"):
            growing.validate_decode_position_binding(
                positions[0],
                payload_slot=13,
            )
        with self.assertRaisesRegex(CacheStateError, "binding changed"):
            growing.bind_growing_position_tensors_untimed(
                tuple(
                    torch.tensor([position], dtype=torch.int64)
                    for position in range(17, 21)
                ),
                starting_position=17,
            )

    def test_layout_and_pointer_identity_are_stable(self) -> None:
        cache = _cache()
        pointers = cache.pointers()
        fingerprint = cache.layout_fingerprint()
        self.assertEqual(cache.pointers(), pointers)
        self.assertEqual(cache.layout_fingerprint(), fingerprint)
        self.assertEqual(len(pointers), len(cache._owned_tensors()))
        self.assertTrue(all(pointer >= 0 for pointer in pointers.values()))

    def test_hot_state_module_has_no_forbidden_growth_primitives(self) -> None:
        source = (
            REPOSITORY_ROOT / "src/kvbench/runtime/kvquant_cache.py"
        ).read_text(encoding="utf-8")
        for forbidden in (
            "torch.cat(",
            "repeat_kv(",
            "repeat_interleave(",
            ".item(",
            ".tolist(",
            ".cpu(",
            ".numpy(",
            "synchronize(",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)
