"""Focused CPU/unit checks for the narrow Phase 8 KIVI adapter boundary."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from kvbench.adapters.base import MethodRuntimeContext
from kvbench.adapters import build_method_adapter
from kvbench.adapters.kivi import (
    KIVI_DECISION_0018_PATCH_SHA256,
    KIVI_EXTENSION_SHA256,
    KIVI_FIXTURE_ROOT_SHA256,
    KIVI_GROUP_SIZE,
    KIVI_OFFICIAL_BASE_TREE,
    KIVI_OFFICIAL_COMMIT,
    KIVI_PATCHED_TREE,
    KIVI_RESIDUAL_LENGTH,
    KIVIMethodAdapter,
    _load_authorized_kivi_runtime,
)
from kvbench.config import load_config
from kvbench.errors import ConfigLoadError, PhaseNotImplementedError
from kvbench.runtime.static_cache import CacheStateError


def _context(*, head_dim: int = 128) -> MethodRuntimeContext:
    return MethodRuntimeContext(
        model_id="frozen-model",
        model_revision="frozen-revision",
        backend_id="pytorch-flash",
        backend_fingerprint=hashlib.sha256(b"backend").hexdigest(),
        num_layers=32,
        num_query_heads=32,
        num_kv_heads=8,
        head_dim=head_dim,
    )


class KIVIMethodAdapterTests(unittest.TestCase):
    def test_explicit_factory_mapping_and_pinned_method_config(self) -> None:
        expected = {
            "k4v4": (4, 4),
            "k2v4": (2, 4),
            "k2v2": (2, 2),
            "k4v2": (4, 2),
        }
        config = load_config(
            Path(__file__).parents[2] / "configs" / "methods" / "kivi.yaml"
        )
        for name, bits in expected.items():
            with self.subTest(name=name, input="preset"):
                method = build_method_adapter(name, _context())
                self.assertIsInstance(method, KIVIMethodAdapter)
                self.assertEqual((method.k_bits, method.v_bits), bits)
            with self.subTest(name=name, input="method_config"):
                method = build_method_adapter(
                    config,
                    _context(),
                    variant_id=name,
                )
                self.assertIsInstance(method, KIVIMethodAdapter)
                self.assertEqual(method.config_name, name)

    def test_factory_keeps_unqualified_and_unknown_selection_closed(self) -> None:
        with self.assertRaises(PhaseNotImplementedError):
            build_method_adapter("kivi", _context())
        with self.assertRaises(PhaseNotImplementedError):
            build_method_adapter("kvquant", _context())
        with self.assertRaises(ConfigLoadError):
            build_method_adapter("unknown", _context())
        with self.assertRaisesRegex(ConfigLoadError, "ambiguous"):
            build_method_adapter("k4v4", _context(), variant_id="k2v4")
        with self.assertRaisesRegex(ConfigLoadError, "explicit frozen"):
            build_method_adapter("kivi", _context(), variant_id="not_registered")

    def test_factory_rejects_method_config_authority_or_variant_drift(self) -> None:
        repository = Path(__file__).parents[2]
        wrong_source = load_config(repository / "configs" / "methods" / "kivi.yaml")
        object.__setattr__(wrong_source, "source_revision", "0" * 40)
        with self.assertRaisesRegex(ConfigLoadError, "pinned method config"):
            build_method_adapter(
                wrong_source,
                _context(),
                variant_id="k4v4",
            )

        wrong_role = load_config(repository / "configs" / "methods" / "kivi.yaml")
        selected = next(
            variant for variant in wrong_role.variants if variant.variant_id == "k4v2"
        )
        object.__setattr__(selected, "role", selected.role.MAIN)
        with self.assertRaisesRegex(ConfigLoadError, "frozen preset"):
            build_method_adapter(
                wrong_role,
                _context(),
                variant_id="k4v2",
            )

    def test_only_frozen_configuration_mapping_is_accepted(self) -> None:
        expected = {
            "k4v4": (4, 4),
            "k2v4": (2, 4),
            "k2v2": (2, 2),
            "k4v2": (4, 2),
        }
        for name, bits in expected.items():
            with self.subTest(name=name):
                method = KIVIMethodAdapter(_context(), name)
                self.assertEqual((method.k_bits, method.v_bits), bits)
                self.assertTrue(method.supports_cuda_graph())
        with self.assertRaisesRegex(ValueError, "unsupported KIVI"):
            KIVIMethodAdapter(_context(), "k3v4")

    def test_geometry_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen Llama GQA"):
            KIVIMethodAdapter(_context(head_dim=64), "k4v4")

    def test_fingerprint_binds_authority_boundary_and_layout(self) -> None:
        method = KIVIMethodAdapter(_context(), "k2v4")
        layout = hashlib.sha256(b"layout").hexdigest()
        first = method.config_fingerprint(layout)
        self.assertEqual(first, method.config_fingerprint(layout))
        self.assertNotEqual(first, KIVIMethodAdapter(_context(), "k4v4").config_fingerprint(layout))
        source = Path(__file__).parents[2] / "src/kvbench/adapters/kivi.py"
        text = source.read_text(encoding="utf-8")
        for authority in (
            KIVI_OFFICIAL_COMMIT,
            KIVI_OFFICIAL_BASE_TREE,
            KIVI_PATCHED_TREE,
            KIVI_DECISION_0018_PATCH_SHA256,
            KIVI_EXTENSION_SHA256,
            KIVI_FIXTURE_ROOT_SHA256,
        ):
            self.assertIn(authority, text)
        self.assertIn("bfloat16_to_float16_to_bfloat16", text)
        self.assertEqual(KIVI_GROUP_SIZE, 32)
        self.assertEqual(KIVI_RESIDUAL_LENGTH, 32)

    def test_loader_rejects_unbound_extension_before_import(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            extension = Path(temporary) / "kivi_gemv_fake.so"
            extension.write_bytes(b"not-the-pinned-extension")
            spec = SimpleNamespace(origin=str(extension))
            with mock.patch(
                "kvbench.adapters.kivi.importlib.util.find_spec",
                return_value=spec,
            ), mock.patch(
                "kvbench.adapters.kivi.importlib.import_module",
                side_effect=AssertionError(
                    "unverified extension import attempted"
                ),
            ) as importer:
                with self.assertRaisesRegex(CacheStateError, "SHA-256 mismatch"):
                    _load_authorized_kivi_runtime()
            importer.assert_not_called()

    def test_loader_rejects_unbound_source_before_extension_import(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extension = root / "kivi_gemv_fake.so"
            extension.write_bytes(b"test-extension-placeholder")
            source = root / "source" / "quant" / "new_pack.py"
            source.parent.mkdir(parents=True)
            source.write_text("import triton\n", encoding="utf-8")
            spec = SimpleNamespace(origin=str(extension))

            def observed_sha256(path: Path) -> str:
                return (
                    KIVI_EXTENSION_SHA256
                    if path.resolve() == extension.resolve()
                    else "0" * 64
                )

            with mock.patch(
                "kvbench.adapters.kivi.importlib.util.find_spec",
                return_value=spec,
            ), mock.patch(
                "kvbench.adapters.kivi._kivi_source_root",
                return_value=source.parents[1],
            ), mock.patch(
                "kvbench.adapters.kivi._sha256_file",
                side_effect=observed_sha256,
            ), mock.patch(
                "kvbench.adapters.kivi.importlib.import_module",
                side_effect=AssertionError(
                    "extension imported before source verification"
                ),
            ) as importer:
                with self.assertRaisesRegex(
                    CacheStateError,
                    "new_pack.py SHA-256 mismatch",
                ):
                    _load_authorized_kivi_runtime()
            importer.assert_not_called()

    def test_source_keeps_direct_static_kernels_and_native_gqa(self) -> None:
        source = (Path(__file__).parents[2] / "src/kvbench/adapters/kivi.py").read_text(
            encoding="utf-8"
        )
        for required in (
            "_minmax_along_last_dim",
            "_pack_along_last_dim",
            "gemv_forward_cuda_outer_dim",
            "query_head // KIVI_GQA_GROUP_SIZE",
            "out=",
            "quantization_fp16_staging",
            "quantization_int_staging",
            "quantization_packed_staging",
            'cache.decode_softmax.fill_(float("-inf"))',
            "_torch().softmax(\n            cache.decode_softmax,",
            "out=cache.decode_softmax,",
        ):
            self.assertIn(required, source)
        for forbidden in ("torch.cat", "repeat_kv", "repeat_interleave"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
