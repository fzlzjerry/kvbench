"""Explicit fail-closed cache-method adapter construction."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from kvbench.adapters.base import KVCacheMethod, MethodRuntimeContext
from kvbench.adapters.bf16 import BF16MethodAdapter
from kvbench.adapters.kivi import KIVI_OFFICIAL_COMMIT, KIVIMethodAdapter
from kvbench.adapters.kvquant import (
    KVQUANT_QUANTIZER_SHA256,
    KVQUANT_UPSTREAM_BASE_COMMIT,
    KVQuantMethodAdapter,
)
from kvbench.adapters.turboquant import (
    TURBOQUANT_SOURCE_COMMIT,
    TurboQuantMethodAdapter,
)
from kvbench.errors import ConfigLoadError, PhaseNotImplementedError
from kvbench.schema import (
    MethodConfig,
    MethodName,
    canonical_json_bytes,
    sha256_hex,
)
from kvbench.schema.config import (
    KiviParameters,
    KVQuantParameters,
    TurboQuantParameters,
    VariantRole,
)


AdapterBuilder = Callable[[MethodRuntimeContext], KVCacheMethod]


def _build_bf16(runtime_context: MethodRuntimeContext) -> KVCacheMethod:
    return BF16MethodAdapter(runtime_context)


_BUILDERS: Mapping[str, AdapterBuilder] = MappingProxyType(
    {"bf16": _build_bf16}
)
_KIVI_CONFIGS = frozenset({"k4v4", "k2v4", "k2v2", "k4v2"})
_KIVI_RUNTIME_PARAMETERS = {
    "k4v4": (4, 4, 32, 32, VariantRole.MAIN),
    "k2v4": (2, 4, 32, 32, VariantRole.MAIN),
    "k2v2": (2, 2, 32, 32, VariantRole.MAIN),
    "k4v2": (4, 2, 32, 32, VariantRole.HELD_OUT),
}
_TURBOQUANT_CONFIGS = frozenset(
    {
        "turboquant_4bit_nc",
        "turboquant_k3v4_nc",
        "turboquant_3bit_nc",
    }
)
_TURBOQUANT_RUNTIME_PARAMETERS = {
    "turboquant_4bit_nc": (4, 4, "mse", True),
    "turboquant_k3v4_nc": (3, 4, "mse", True),
    "turboquant_3bit_nc": (3, 3, "mse", True),
}
_KVQUANT_CONFIGS = frozenset({"kvq4", "kvq3", "kvq2"})
_KVQUANT_RUNTIME_PARAMETERS = {
    "kvq4": (4, 5, 12, KVQUANT_QUANTIZER_SHA256["kvq4"], "int32", "float32"),
    "kvq3": (3, 5, 12, KVQUANT_QUANTIZER_SHA256["kvq3"], "int32", "float32"),
    "kvq2": (2, 5, 12, KVQUANT_QUANTIZER_SHA256["kvq2"], "int32", "float32"),
}
# Canonical digest of the source, calibration, and variant authority fields in
# configs/methods/kvquant.yaml. Resolution text/status is deliberately excluded:
# Phase 11 gates the exact technical authority independently.
KVQUANT_METHOD_CONFIG_AUTHORITY_SHA256 = (
    "a229b99b7edcf77289cba33422023139adc0562b24822da415b7185520d83a57"
)


def _method_name(method_config: MethodConfig | MethodName | str) -> str:
    if isinstance(method_config, MethodConfig):
        return method_config.method.value
    if isinstance(method_config, MethodName):
        return method_config.value
    if type(method_config) is str:
        if method_config in _KIVI_CONFIGS:
            return MethodName.KIVI.value
        if method_config in _TURBOQUANT_CONFIGS:
            return MethodName.TURBOQUANT.value
        if method_config in _KVQUANT_CONFIGS:
            return MethodName.KVQUANT.value
        return method_config
    raise ConfigLoadError("method adapter selection has an invalid type")


def _turboquant_config_name(
    method_config: MethodConfig | MethodName | str,
    variant_id: str | None,
) -> str:
    inferred = (
        method_config
        if type(method_config) is str
        and method_config in _TURBOQUANT_CONFIGS
        else None
    )
    if inferred is not None and variant_id not in {None, inferred}:
        raise ConfigLoadError("TurboQuant preset selection is ambiguous")
    selected = inferred if inferred is not None else variant_id
    if selected not in _TURBOQUANT_CONFIGS:
        raise ConfigLoadError(
            "TurboQuant requires one explicit mandatory configuration"
        )
    if isinstance(method_config, MethodConfig):
        if (
            method_config.method is not MethodName.TURBOQUANT
            or method_config.method_config_id != "turboquant"
            or method_config.source_revision != TURBOQUANT_SOURCE_COMMIT
        ):
            raise ConfigLoadError(
                "TurboQuant adapter requires the pinned method config"
            )
        variant = next(
            (
                item
                for item in method_config.variants
                if item.variant_id == selected
            ),
            None,
        )
        if variant is None or variant.role is not VariantRole.MAIN:
            raise ConfigLoadError(
                "TurboQuant configuration is not a mandatory main variant"
            )
        parameters = variant.parameters
        if not isinstance(parameters, TurboQuantParameters):
            raise ConfigLoadError("TurboQuant parameters have the wrong type")
        observed = (
            parameters.key_bits,
            parameters.value_bits,
            parameters.key_path,
            parameters.norm_correction,
        )
        if observed != _TURBOQUANT_RUNTIME_PARAMETERS[selected]:
            raise ConfigLoadError(
                "TurboQuant parameters differ from the pinned preset"
            )
        optional_runtime = (
            parameters.cache_dtype_name,
            parameters.skipped_layers,
            parameters.block_size,
            parameters.decode_split_count,
        )
        expected_runtime = (selected, (0, 1, 30, 31), 16, 4)
        if any(value is not None for value in optional_runtime):
            if optional_runtime != expected_runtime:
                raise ConfigLoadError(
                    "TurboQuant runtime parameters differ from Phase 6"
                )
    return selected


def _kivi_config_name(
    method_config: MethodConfig | MethodName | str,
    variant_id: str | None,
) -> str:
    inferred = (
        method_config
        if type(method_config) is str and method_config in _KIVI_CONFIGS
        else None
    )
    if inferred is not None and variant_id not in {None, inferred}:
        raise ConfigLoadError("KIVI preset selection is ambiguous")
    selected = inferred if inferred is not None else variant_id
    if selected not in _KIVI_CONFIGS:
        raise ConfigLoadError("KIVI requires one explicit frozen configuration")
    if isinstance(method_config, MethodConfig):
        if (
            method_config.method is not MethodName.KIVI
            or method_config.method_config_id != "kivi"
            or method_config.source_revision != KIVI_OFFICIAL_COMMIT
        ):
            raise ConfigLoadError("KIVI adapter requires the pinned method config")
        variant = next(
            (
                item
                for item in method_config.variants
                if item.variant_id == selected
            ),
            None,
        )
        if variant is None:
            raise ConfigLoadError("KIVI configuration is absent from the method config")
        parameters = variant.parameters
        if not isinstance(parameters, KiviParameters):
            raise ConfigLoadError("KIVI parameters have the wrong type")
        observed = (
            parameters.k_bits,
            parameters.v_bits,
            parameters.group_size,
            parameters.residual_length,
            variant.role,
        )
        if observed != _KIVI_RUNTIME_PARAMETERS[selected]:
            raise ConfigLoadError("KIVI parameters differ from the frozen preset")
    return selected


def _kvquant_authority_sha256(method_config: MethodConfig) -> str:
    calibration = method_config.calibration
    if calibration is None:
        raise ConfigLoadError(
            "KVQuant adapter requires the frozen calibration reference"
        )
    payload = {
        "method_config_id": method_config.method_config_id,
        "method": method_config.method.value,
        "source_lock_id": method_config.source_lock_id,
        "source_revision": method_config.source_revision,
        "calibration": calibration.to_dict(),
        "variants": [
            {
                "variant_id": variant.variant_id,
                "role": variant.role.value,
                "parameters": variant.parameters.to_dict(),
            }
            for variant in sorted(
                method_config.variants,
                key=lambda candidate: candidate.variant_id,
            )
        ],
    }
    return sha256_hex(canonical_json_bytes(payload))


def _kvquant_config_name(
    method_config: MethodConfig | MethodName | str,
    variant_id: str | None,
) -> str:
    inferred = (
        method_config
        if type(method_config) is str and method_config in _KVQUANT_CONFIGS
        else None
    )
    if inferred is not None and variant_id not in {None, inferred}:
        raise ConfigLoadError("KVQuant preset selection is ambiguous")
    selected = inferred if inferred is not None else variant_id
    if selected not in _KVQUANT_CONFIGS:
        raise ConfigLoadError(
            "KVQuant requires one explicit frozen configuration"
        )
    if isinstance(method_config, MethodConfig):
        if (
            method_config.method is not MethodName.KVQUANT
            or method_config.method_config_id != "kvquant"
            or method_config.source_lock_id != "kvquant"
            or method_config.source_revision != KVQUANT_UPSTREAM_BASE_COMMIT
            or _kvquant_authority_sha256(method_config)
            != KVQUANT_METHOD_CONFIG_AUTHORITY_SHA256
        ):
            raise ConfigLoadError(
                "KVQuant adapter requires the pinned source and calibration config"
            )
        variant = next(
            (
                item
                for item in method_config.variants
                if item.variant_id == selected
            ),
            None,
        )
        if variant is None or variant.role is not VariantRole.MAIN:
            raise ConfigLoadError(
                "KVQuant configuration is not a mandatory main variant"
            )
        parameters = variant.parameters
        if not isinstance(parameters, KVQuantParameters):
            raise ConfigLoadError("KVQuant parameters have the wrong type")
        observed = (
            parameters.bits,
            parameters.sink_tokens,
            parameters.outlier_cap,
            parameters.calibration_artifact_sha256,
            parameters.sparse_index_dtype,
            parameters.lut_scale_dtype,
        )
        if observed != _KVQUANT_RUNTIME_PARAMETERS[selected]:
            raise ConfigLoadError(
                "KVQuant parameters differ from the frozen preset"
            )
    return selected


def build_method_adapter(
    method_config: MethodConfig | MethodName | str,
    runtime_context: MethodRuntimeContext,
    *,
    variant_id: str | None = None,
) -> KVCacheMethod:
    """Construct one explicit adapter; unimplemented methods remain closed."""

    name = _method_name(method_config)
    if (
        name == MethodName.KIVI.value
        and variant_id is None
        and (
            not isinstance(method_config, str)
            or method_config == MethodName.KIVI.value
        )
    ):
        raise PhaseNotImplementedError(
            "kivi requires one explicit Phase 8 configuration"
        )
    if (
        name == MethodName.KVQUANT.value
        and variant_id is None
        and (
            not isinstance(method_config, str)
            or method_config == MethodName.KVQUANT.value
        )
    ):
        raise PhaseNotImplementedError(
            "kvquant requires one explicit Phase 11 configuration"
        )
    if name == MethodName.KIVI.value:
        selected = _kivi_config_name(method_config, variant_id)
        return KIVIMethodAdapter(runtime_context, selected)
    if name == MethodName.TURBOQUANT.value:
        selected = _turboquant_config_name(method_config, variant_id)
        return TurboQuantMethodAdapter(runtime_context, selected)
    if name == MethodName.KVQUANT.value:
        selected = _kvquant_config_name(method_config, variant_id)
        return KVQuantMethodAdapter(runtime_context, selected)
    if variant_id is not None:
        raise ConfigLoadError(
            "variant_id is accepted only for quantized method adapters"
        )
    try:
        builder = _BUILDERS[name]
    except KeyError as error:
        raise ConfigLoadError("unknown method adapter") from error
    if isinstance(method_config, MethodConfig):
        if (
            method_config.method is not MethodName.BF16
            or method_config.method_config_id != "bf16"
        ):
            raise ConfigLoadError("BF16 adapter requires the BF16 method config")
    return builder(runtime_context)
