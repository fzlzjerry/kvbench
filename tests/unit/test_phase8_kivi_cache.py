from __future__ import annotations

import unittest

from kvbench.runtime.kivi_cache import (
    KIVI_CONFIG_BITS,
    KIVI_GROUP_SIZE,
    KIVI_RESIDUAL_LENGTH,
    KIVIStaticCache,
)
from kvbench.runtime.kivi_fixture import load_kivi_fixture
from kvbench.runtime.static_cache import CacheBoundsError


def _cache(*, config_name: str = "k4v4", capacity: int = 128) -> KIVIStaticCache:
    return KIVIStaticCache(
        config_name=config_name,
        num_layers=1,
        batch_size=1,
        num_query_heads=32,
        num_kv_heads=8,
        capacity=capacity,
        head_dim=128,
        device="cpu",
    )


class KIVIStaticCacheTests(unittest.TestCase):
    def test_frozen_config_mapping_and_geometry_rejection(self) -> None:
        self.assertEqual(KIVI_CONFIG_BITS["k4v4"], (4, 4))
        self.assertEqual(KIVI_CONFIG_BITS["k2v4"], (2, 4))
        self.assertEqual(KIVI_CONFIG_BITS["k2v2"], (2, 2))
        self.assertEqual(KIVI_CONFIG_BITS["k4v2"], (4, 2))
        self.assertEqual(KIVI_GROUP_SIZE, 32)
        self.assertEqual(KIVI_RESIDUAL_LENGTH, 32)
        with self.assertRaisesRegex(ValueError, "unsupported KIVI"):
            _cache(config_name="k3v4")
        with self.assertRaisesRegex(ValueError, "frozen"):
            KIVIStaticCache(
                config_name="k4v4",
                num_layers=1,
                batch_size=1,
                num_query_heads=32,
                num_kv_heads=16,
                capacity=128,
                head_dim=128,
                device="cpu",
            )

    def test_rollover_states_are_source_faithful_at_31_through_34(self) -> None:
        cache = _cache()
        observed: dict[int, dict[str, list[int]]] = {}
        for token in range(34):
            cache.update(layer_idx=0, token_index=token)
            if token + 1 in {31, 32, 33, 34}:
                state = cache.token_index_state(0)
                observed[token + 1] = {
                    name: state[name].tolist()
                    for name in (
                        "quantized_key_tokens",
                        "residual_key_tokens",
                        "quantized_value_tokens",
                        "residual_value_tokens",
                    )
                }
        self.assertEqual(observed[31]["quantized_key_tokens"], [])
        self.assertEqual(observed[31]["residual_key_tokens"], list(range(31)))
        self.assertEqual(observed[31]["quantized_value_tokens"], [])
        self.assertEqual(observed[31]["residual_value_tokens"], list(range(31)))
        self.assertEqual(observed[32]["quantized_key_tokens"], list(range(32)))
        self.assertEqual(observed[32]["residual_key_tokens"], [])
        self.assertEqual(observed[32]["quantized_value_tokens"], [])
        self.assertEqual(observed[32]["residual_value_tokens"], list(range(32)))
        self.assertEqual(observed[33]["residual_key_tokens"], [32])
        self.assertEqual(observed[33]["quantized_value_tokens"], [0])
        self.assertEqual(observed[33]["residual_value_tokens"], list(range(1, 33)))
        self.assertEqual(observed[34]["residual_key_tokens"], [32, 33])
        self.assertEqual(observed[34]["quantized_value_tokens"], [0, 1])
        self.assertEqual(observed[34]["residual_value_tokens"], list(range(2, 34)))
        final = observed[34]
        tokens = (
            final["quantized_key_tokens"] + final["residual_key_tokens"]
        )
        self.assertEqual(tokens, list(range(34)))
        self.assertEqual(len(tokens), len(set(tokens)))
        tokens = (
            final["quantized_value_tokens"] + final["residual_value_tokens"]
        )
        self.assertEqual(tokens, list(range(34)))
        self.assertEqual(len(tokens), len(set(tokens)))

    def test_all_fixture_byte_categories_totals_and_legacy_ratios_match(self) -> None:
        for configuration in KIVI_CONFIG_BITS:
            fixture = load_kivi_fixture(configuration)
            records = fixture.legacy_allocation_records()
            self.assertEqual(
                [record["context"] for record in records],
                [31, 32, 33, 64],
            )
            cache = _cache(config_name=configuration)
            for record in records:
                context = record["context"]
                with self.subTest(
                    configuration=configuration,
                    context=context,
                ):
                    observed = cache.reference_active_byte_breakdown(context)
                    self.assertEqual(observed, record["categories"])
                    self.assertEqual(sum(observed.values()), record["actual_total"])
                    self.assertTrue(record["storage_agreement"])
                    self.assertEqual(
                        record["calculation_mode"],
                        (
                            "source_layout_formula_no_runtime_campaign"
                            if context == 64
                            else "actual_source_owned_tensor_storage"
                        ),
                    )
                    self.assertAlmostEqual(
                        record["rho_alloc_legacy"],
                        record["actual_total"]
                        / record["logical_bf16_bytes"],
                        places=15,
                    )
                    self.assertAlmostEqual(
                        record["canonical_r_alloc"],
                        record["logical_bf16_bytes"]
                        / record["actual_total"],
                        places=15,
                    )
                    self.assertLessEqual(
                        abs(
                            record["rho_alloc_legacy"]
                            * record["canonical_r_alloc"]
                            - 1.0
                        ),
                        1e-9,
                    )
                    self.assertIsNone(record["r_hbm"])

    def test_extension_abi_rows_and_rollover_scratch_are_preallocated(self) -> None:
        cache = _cache(config_name="k4v4", capacity=128)
        # The adapter transforms logical K/V to this raw extension layout:
        # K [packed_time, D] and V [packed_D, history_time].
        self.assertEqual(cache.packed_key_history.shape, (1, 1, 8, 16, 128))
        self.assertEqual(cache.packed_value_history.shape, (1, 1, 8, 16, 96))
        self.assertEqual(cache.key_scales.shape, (1, 1, 8, 4, 128))
        self.assertEqual(cache.value_scales.shape, (1, 1, 8, 4, 96))
        self.assertFalse(hasattr(cache, "value_payload_shift_scratch"))
        self.assertFalse(hasattr(cache, "value_metadata_shift_scratch"))
        self.assertEqual(str(cache.decode_softmax.dtype), "torch.float32")
        self.assertEqual(cache.quantization_fp16_staging.shape, (1, 8, 128, 32))
        self.assertEqual(cache.quantization_int_staging.shape, (1, 8, 128, 32))
        self.assertEqual(cache.quantization_packed_staging.shape, (1, 8, 128, 8))
        self.assertEqual(
            cache.byte_breakdown()["value_rollover_shift_scratch"],
            0,
        )

    def test_preallocated_accounting_is_exact_and_pointers_stable(self) -> None:
        for configuration in KIVI_CONFIG_BITS:
            for capacity in (128, 4096):
                with self.subTest(
                    configuration=configuration,
                    capacity=capacity,
                ):
                    cache = _cache(
                        config_name=configuration,
                        capacity=capacity,
                    )
                    before = cache.pointers()
                    breakdown = cache.byte_breakdown()
                    accounting = cache.accounting()
                    self.assertEqual(
                        sum(breakdown.values()),
                        accounting.allocated_bytes,
                    )
                    self.assertEqual(
                        accounting.predicted_tensor_bytes,
                        accounting.measured_tensor_bytes,
                    )
                    self.assertEqual(
                        breakdown["other_metadata"],
                        (
                            cache.num_layers
                            * (
                                cache.key_history_capacity
                                + cache.residual_length
                                + cache.value_history_capacity
                                + cache.residual_length
                            )
                            * 8
                        ),
                    )
                    self.assertEqual(cache.pointers(), before)
                    ratios = cache.ratios()
                    self.assertLessEqual(
                        abs(ratios.r_alloc * ratios.rho_alloc - 1.0),
                        1e-9,
                    )
                    self.assertGreater(cache.logical_bf16_storage_bytes, 0)
                    self.assertIsNone(cache.r_hbm)

    def test_fixed_scratch_preserves_physical_historical_state(self) -> None:
        cache = _cache(capacity=64)
        cache.initialize_deterministic()
        for token in range(33):
            cache.update(layer_idx=0, token_index=token)
        cache.reset_active_length(33)
        cache.prepare_fixed(33)
        before = cache.physical_history_checksum(0)
        cache.fixed_scratch_overwrite(layer_idx=0, token_index=33)
        self.assertEqual(cache.physical_history_checksum(0), before)

        active_regions = (
            cache.packed_key_history[0, 0, 0, 0, 0],
            cache.packed_value_history[0, 0, 0, 0, 0],
            cache.key_scales[0, 0, 0, 0, 0],
            cache.key_minimums[0, 0, 0, 0, 0],
            cache.value_scales[0, 0, 0, 0, 0],
            cache.value_minimums[0, 0, 0, 0, 0],
            cache.key_residual[0, 0, 0, 0, 0],
            cache.value_residual_ring[0, 0, 0, 0, 0],
        )
        for index, value in enumerate(active_regions):
            with self.subTest(region=index):
                value.fill_(1)
                self.assertNotEqual(cache.physical_history_checksum(0), before)
                value.zero_()
                self.assertEqual(cache.physical_history_checksum(0), before)

    def test_fixed_scratch_preserves_logical_history(self) -> None:
        cache = _cache()
        for token in range(33):
            cache.update(layer_idx=0, token_index=token)
        cache.reset_active_length(33)
        cache.prepare_fixed(33)
        before = cache.fixed_scratch_history_checksum(0)
        cache.fixed_scratch_overwrite(layer_idx=0, token_index=33)
        self.assertEqual(cache.fixed_scratch_history_checksum(0), before)
        with self.assertRaises(CacheBoundsError):
            cache.fixed_scratch_overwrite(layer_idx=0, token_index=34)
        self.assertEqual(cache._fixed_scratch_tokens[0], 33)

    def test_deterministic_reset_clears_rollover_ledgers(self) -> None:
        cache = _cache(capacity=64)
        for token in range(34):
            cache.update(layer_idx=0, token_index=token)
        cache.initialize_deterministic()
        state = cache.token_index_state(0)
        self.assertEqual(cache.active_context, 0)
        self.assertTrue(
            all(int(value.numel()) == 0 for value in state.values())
        )
        cache.update(layer_idx=0, token_index=0)
        self.assertEqual(
            cache.token_index_state(0)["residual_key_tokens"].tolist(), [0]
        )

    def test_capacity_is_bounded(self) -> None:
        cache = _cache(capacity=32)
        for token in range(32):
            cache.update(layer_idx=0, token_index=token)
        before = cache.history_checksum(0)
        with self.assertRaises(CacheBoundsError):
            cache.update(layer_idx=0, token_index=32)
        self.assertEqual(cache.history_checksum(0), before)
        state = cache.token_index_state(0)
        self.assertEqual(state["quantized_key_tokens"].tolist(), list(range(32)))
        self.assertEqual(state["residual_key_tokens"].tolist(), [])
        self.assertEqual(state["quantized_value_tokens"].tolist(), [])
        self.assertEqual(state["residual_value_tokens"].tolist(), list(range(32)))


if __name__ == "__main__":
    unittest.main()
