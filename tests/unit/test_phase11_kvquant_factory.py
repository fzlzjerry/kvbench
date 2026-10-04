"""Focused explicit-factory checks for the Phase 11 KVQuant adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path
import unittest

from kvbench.adapters import (
    BF16MethodAdapter,
    KIVIMethodAdapter,
    KVQUANT_ADAPTER_VERSION,
    KVQUANT_METHOD_IDENTIFIER,
    KVQuantMethodAdapter,
    TurboQuantMethodAdapter,
    build_method_adapter,
)
from kvbench.adapters.factory import (
    KVQUANT_METHOD_CONFIG_AUTHORITY_SHA256,
)
from kvbench.adapters.kvquant import (
    KVQUANT_QUANTIZER_SHA256,
    KVQUANT_UPSTREAM_BASE_COMMIT,
)
from kvbench.config import load_config
from kvbench.errors import ConfigLoadError, PhaseNotImplementedError
from kvbench.schema import MethodConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
KVQUANT_CONFIG = REPOSITORY_ROOT / "configs/methods/kvquant.yaml"


def _context():
    from kvbench.adapters import MethodRuntimeContext

    return MethodRuntimeContext(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        model_revision="0e9e39f249a16976918f6564b8830bc894c89659",
        backend_id="pytorch_flash",
        backend_fingerprint=hashlib.sha256(b"backend").hexdigest(),
        num_layers=32,
        num_query_heads=32,
        num_kv_heads=8,
        head_dim=128,
    )


def _load_kvquant_config() -> MethodConfig:
    config = load_config(KVQUANT_CONFIG)
    if not isinstance(config, MethodConfig):
        raise AssertionError("KVQuant config did not load as MethodConfig")
    return config


class Phase11KVQuantFactoryTests(unittest.TestCase):
    def test_explicit_presets_and_pinned_method_config_construct(self) -> None:
        expected = {"kvq4": 4, "kvq3": 3, "kvq2": 2}
        config = _load_kvquant_config()
        # Stale phase-boundary resolution is not an adapter-construction gate;
        # exact technical authority is the gate.
        self.assertEqual(config.resolution.status.value, "unresolved")
        for name, bits in expected.items():
            with self.subTest(name=name, source="preset"):
                adapter = build_method_adapter(name, _context())
                self.assertIsInstance(adapter, KVQuantMethodAdapter)
                self.assertEqual(adapter.config_name, name)
                self.assertEqual(adapter.bits, bits)
            with self.subTest(name=name, source="method_config"):
                adapter = build_method_adapter(
                    config,
                    _context(),
                    variant_id=name,
                )
                self.assertIsInstance(adapter, KVQuantMethodAdapter)
                self.assertEqual(
                    adapter.quantizer_sha256,
                    KVQUANT_QUANTIZER_SHA256[name],
                )

    def test_factory_binds_complete_source_calibration_variant_authority(self) -> None:
        config = _load_kvquant_config()
        self.assertEqual(config.source_revision, KVQUANT_UPSTREAM_BASE_COMMIT)
        self.assertEqual(
            KVQUANT_METHOD_CONFIG_AUTHORITY_SHA256,
            "a229b99b7edcf77289cba33422023139adc0562b24822da415b7185520d83a57",
        )
        self.assertEqual(KVQUANT_METHOD_IDENTIFIER, "kvquant_gqa_upstream_patch_v1")
        self.assertEqual(
            KVQUANT_ADAPTER_VERSION,
            "kvbench-kvquant-method-adapter-1.1.0",
        )
        self.assertIsNotNone(config.calibration)
        assert config.calibration is not None
        self.assertEqual(
            config.calibration.calibration_root_digest,
            "8148306d08205af376994b022f189a0d6837915cd279ca8af6b104e1f4b46ccf",
        )

    def test_unqualified_ambiguous_and_unsupported_selection_fails_closed(
        self,
    ) -> None:
        with self.assertRaises(PhaseNotImplementedError):
            build_method_adapter("kvquant", _context())
        for rejected in ("kvq1", "kvq5", "unknown"):
            with self.subTest(rejected=rejected):
                with self.assertRaises(ConfigLoadError):
                    build_method_adapter(rejected, _context())
        with self.assertRaisesRegex(ConfigLoadError, "ambiguous"):
            build_method_adapter("kvq4", _context(), variant_id="kvq3")
        with self.assertRaisesRegex(ConfigLoadError, "explicit frozen"):
            build_method_adapter("kvquant", _context(), variant_id="kvq5")
        with self.assertRaisesRegex(PhaseNotImplementedError, "explicit Phase 11"):
            build_method_adapter(_load_kvquant_config(), _context())

    def test_method_config_authority_drift_fails_closed(self) -> None:
        source_drift = _load_kvquant_config()
        object.__setattr__(source_drift, "source_revision", "0" * 40)
        with self.assertRaisesRegex(ConfigLoadError, "pinned source"):
            build_method_adapter(source_drift, _context(), variant_id="kvq4")

        calibration_drift = _load_kvquant_config()
        assert calibration_drift.calibration is not None
        object.__setattr__(
            calibration_drift.calibration,
            "calibration_root_digest",
            "0" * 64,
        )
        with self.assertRaisesRegex(ConfigLoadError, "pinned source"):
            build_method_adapter(
                calibration_drift,
                _context(),
                variant_id="kvq3",
            )

        parameter_drift = _load_kvquant_config()
        variant = next(
            item for item in parameter_drift.variants if item.variant_id == "kvq2"
        )
        object.__setattr__(variant.parameters, "outlier_cap", 11)
        with self.assertRaisesRegex(ConfigLoadError, "pinned source"):
            build_method_adapter(parameter_drift, _context(), variant_id="kvq2")

    def test_existing_factory_mappings_are_unchanged(self) -> None:
        self.assertIsInstance(
            build_method_adapter("bf16", _context()),
            BF16MethodAdapter,
        )
        self.assertIsInstance(
            build_method_adapter("turboquant_4bit_nc", _context()),
            TurboQuantMethodAdapter,
        )
        self.assertIsInstance(
            build_method_adapter("k4v4", _context()),
            KIVIMethodAdapter,
        )


if __name__ == "__main__":
    unittest.main()
