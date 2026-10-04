"""Static Phase 11 adapter for the corrected KVQuant CUDA authority.

The adapter is intentionally narrow.  It binds one checksum-verified CUDA
extension and the frozen Phase 9 quantizers to caller-owned storage from
``KVQuantStaticCache``.  It does not contain a replacement quantizer or
attention implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import struct
from types import ModuleType
from typing import Any, Mapping
import weakref

from kvbench.adapters.base import MethodRuntimeContext
from kvbench.runtime.backend import flash_attention_forward
from kvbench.runtime.kvquant_cache import (
    KVQUANT_CONFIG_BITS,
    KVQUANT_HEAD_DIM,
    KVQUANT_KEY_CAP,
    KVQUANT_NUM_KV_HEADS,
    KVQUANT_NUM_LAYERS,
    KVQUANT_NUM_QUERY_HEADS,
    KVQUANT_Q4_VALUE_DECODE_WORKSPACE_FORMULA_VERSION,
    KVQUANT_SINK_TOKENS,
    KVQUANT_VALUE_CAP,
    KVQuantStaticCache,
)
from kvbench.runtime.static_cache import CacheStateError
from kvbench.schema import canonical_json_bytes, sha256_hex
from kvbench.schema.base import require_sha256


KVQUANT_ADAPTER_VERSION = "kvbench-kvquant-method-adapter-1.3.0"
KVQUANT_ADAPTER_FINGERPRINT_SCHEMA_VERSION = (
    "kvbench-kvquant-method-adapter-config-1.3.0"
)
KVQUANT_METHOD_IDENTIFIER = "kvquant_gqa_upstream_patch_v1"
KVQUANT_Q4_ADAPTER_VERSION = "kvbench-kvquant-method-adapter-1.4.0"
KVQUANT_Q4_ADAPTER_FINGERPRINT_SCHEMA_VERSION = (
    "kvbench-kvquant-method-adapter-config-1.4.0"
)
KVQUANT_LEGACY_ADAPTER_IMPLEMENTATION_SHA256 = (
    "922ad8ef951b815118043b17792ecf737969ade437f96cacabd27d81c6e94b91"
)
KVQUANT_EXECUTION_SOURCE_IDENTIFIER = (
    "kvquant_gqa_longctx_deterministic_q23_v4"
)
KVQUANT_UPSTREAM_BASE_COMMIT = "57a238357f0ffe50084670fcd5781c9848f80ea2"
KVQUANT_UPSTREAM_BASE_TREE = "094e0f736f77ee327e5350cbd1eefb1c936aa77b"
KVQUANT_DECISION_0021_PATCH_SHA256 = (
    "db3b6fb7ec0a72e25001e1c83a5158d86512248db5c3a06c61895598d1d482d6"
)
KVQUANT_AGGREGATE_PATCH_SHA256 = (
    "7b9d3cc6773e8ef37697601c885f2c5ec581dffd57cf59424d03e68f147bd55a"
)
KVQUANT_CORRECTED_COMMIT = "34b0bdfa83082e1f30387d9ac5cca369006e089c"
KVQUANT_CORRECTED_TREE = "1f85af65fe03061583ffe8bd91e47d7ecffdd312"
KVQUANT_CORRECTED_CUDA_SHA256 = (
    "457a1d8d7bd07ba2e7e8420cbe3b722b52471f131ecc88f7f5442584a8b212f9"
)
KVQUANT_EXTENSION_SHA256 = (
    "b3c33badb8e55b19d6b2ce535182e964ce51e5102d8413b29701dd3d817ad73d"
)
KVQUANT_AUTHORIZED_CONTAINER_DIGEST = (
    "sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e"
)
KVQUANT_CALIBRATION_ID = "kvqcal-cdb724c806d64d095c040d2673a987a3"
KVQUANT_CALIBRATION_ROOT_SHA256 = (
    "8148306d08205af376994b022f189a0d6837915cd279ca8af6b104e1f4b46ccf"
)
KVQUANT_HISTORICAL_FIXTURE_ID = "kvqref-a50af6511c314b6394e58a7f81ceefb8"
KVQUANT_HISTORICAL_ROOT_SHA256 = (
    "32cdf465a361dd6695b66ccbea0a462bddc075fd9778d0aa8cdaa3f94e6f63ab"
)
KVQUANT_FIXTURE_ID = "kvqref-2e0a0e9022c50cbc6fb497d88cae973e"
KVQUANT_FIXTURE_ROOT_SHA256 = (
    "c28682d58706b58812dc1db69ba5eb4982339ba13f39bf67f751794cdaabfdec"
)
KVQUANT_QUANTIZER_SHA256 = {
    "kvq4": "a8c009633ac4cad952deb2a2fa96c44ef928a1510dadcf11dee29a7a3efe1bf6",
    "kvq3": "97518129cc64ffa445722cb0802b3082631841de50835cbdf2c85c36a0c1579f",
    "kvq2": "b9bb3a8699aa38fb2a5707ff036814971552462692a180431f6f68df9624560e",
}
KVQUANT_ZERO_CODE = {"kvq4": 7, "kvq3": 3, "kvq2": 1}
KVQUANT_DECISIONS = (
    "0021",
    "0023",
    "0024",
    "0025",
    "0026",
    "0027",
    "0029",
)
KVQUANT_Q4_DECISIONS = (*KVQUANT_DECISIONS, "0036")
KVQUANT_Q4_DETERMINISTIC_VALUE_DECODE_API = (
    "vecquant4matmul_nuq_perchannel_transposed_"
    "mha_batched_fused_opt2_deterministic_out"
)
KVQUANT_DETERMINISTIC_VALUE_DECODE_APIS = {
    bits: (
        f"vecquant{bits}matmul_nuq_perchannel_transposed_"
        "mha_batched_fused_opt2_deterministic_out"
    )
    for bits in (4, 3, 2)
}
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CALIBRATION_ROOT = (
    _REPOSITORY_ROOT
    / "calibration"
    / "kvquant"
    / KVQUANT_CALIBRATION_ID
)
_TORCH: Any | None = None


def _torch() -> Any:
    global _TORCH
    if _TORCH is None:
        try:
            _TORCH = importlib.import_module("torch")
        except ModuleNotFoundError as error:  # pragma: no cover - environment
            raise CacheStateError(
                "PyTorch is required for the KVQuant adapter"
            ) from error
    return _TORCH


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise CacheStateError("KVQuant authority path must be a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _calibration_root() -> Path:
    raw = os.environ.get(
        "KVBENCH_KVQUANT_CALIBRATION_ROOT",
        str(_DEFAULT_CALIBRATION_ROOT),
    )
    try:
        root = Path(raw).resolve(strict=True)
    except FileNotFoundError as error:
        raise CacheStateError(
            "frozen KVQuant calibration root is unavailable"
        ) from error
    if root.is_symlink() or not root.is_dir():
        raise CacheStateError("KVQuant calibration root is not a directory")
    return root


def _extension_path() -> Path:
    raw = os.environ.get("KVBENCH_KVQUANT_EXTENSION")
    if not raw:
        raise CacheStateError(
            "KVBENCH_KVQUANT_EXTENSION must name the checksum-bound binary"
        )
    try:
        path = Path(raw).resolve(strict=True)
    except FileNotFoundError as error:
        raise CacheStateError("KVQuant extension is unavailable") from error
    if (
        path.is_symlink()
        or not path.is_file()
        or not path.name.startswith("quant_cuda.")
        or path.suffix != ".so"
        or _sha256_file(path) != KVQUANT_EXTENSION_SHA256
    ):
        raise CacheStateError("KVQuant extension identity differs")
    return path


def _required_extension_symbols() -> tuple[str, ...]:
    common = (
        "select_fixed_outliers_1024_cap12_out",
        "key_sparse_residual_1024_cap12_out",
        "append_value_sparse_1024_cap12_out",
        *KVQUANT_DETERMINISTIC_VALUE_DECODE_APIS.values(),
    )
    per_bit: list[str] = []
    for bits in (4, 3, 2):
        per_bit.extend(
            (
                f"vecquant{bits}appendvecKsparse",
                f"vecquant{bits}appendvecVsparseParallel",
                (
                    f"vecquant{bits}matmul_nuq_perchannel_transposed_"
                    "rope_mha_batched_fused_opt2"
                ),
                (
                    f"vecquant{bits}matmul_nuq_perchannel_transposed_"
                    "mha_batched_fused_opt2"
                ),
            )
        )
    return (*common, *per_bit)


def _load_authorized_extension() -> ModuleType:
    path = _extension_path()
    existing = importlib.util.find_spec("quant_cuda")
    if existing is not None and existing.origin:
        try:
            existing_path = Path(existing.origin).resolve(strict=True)
        except FileNotFoundError as error:
            raise CacheStateError(
                "loaded quant_cuda authority is unavailable"
            ) from error
        if existing_path != path:
            raise CacheStateError(
                "a different quant_cuda module is already discoverable"
            )
    specification = importlib.util.spec_from_file_location("quant_cuda", path)
    if specification is None or specification.loader is None:
        raise CacheStateError("KVQuant extension specification is unavailable")
    module = importlib.util.module_from_spec(specification)
    try:
        specification.loader.exec_module(module)
    except (ImportError, OSError, RuntimeError) as error:
        raise CacheStateError("KVQuant extension cannot be loaded") from error
    try:
        loaded_path = Path(str(module.__file__)).resolve(strict=True)
    except (AttributeError, FileNotFoundError) as error:
        raise CacheStateError(
            "loaded KVQuant extension lacks exact identity"
        ) from error
    if loaded_path != path:
        raise CacheStateError("loaded KVQuant extension path differs")
    missing = [
        name for name in _required_extension_symbols() if not hasattr(module, name)
    ]
    if missing:
        raise CacheStateError(
            f"KVQuant extension lacks required APIs: {', '.join(missing)}"
        )
    return module


def _read_required_float_tensors(
    path: Path,
    required: set[str],
) -> dict[str, Any]:
    """Read only frozen float32 tensors without executable serialization."""

    torch = _torch()
    if _sha256_file(path) == "":
        raise CacheStateError("unreachable KVQuant quantizer digest")
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise CacheStateError("KVQuant safetensors header is truncated")
            header_size = struct.unpack("<Q", prefix)[0]
            header_bytes = handle.read(header_size)
            header = json.loads(header_bytes)
            payload_offset = 8 + header_size
            tensors: dict[str, Any] = {}
            for name in sorted(required):
                record = header.get(name)
                if (
                    type(record) is not dict
                    or record.get("dtype") != "F32"
                    or type(record.get("shape")) is not list
                    or type(record.get("data_offsets")) is not list
                    or len(record["data_offsets"]) != 2
                ):
                    raise CacheStateError(
                        f"KVQuant quantizer tensor header differs: {name}"
                    )
                start, stop = record["data_offsets"]
                if (
                    type(start) is not int
                    or type(stop) is not int
                    or start < 0
                    or stop <= start
                ):
                    raise CacheStateError(
                        f"KVQuant quantizer tensor offsets differ: {name}"
                    )
                handle.seek(payload_offset + start)
                storage = bytearray(handle.read(stop - start))
                if len(storage) != stop - start:
                    raise CacheStateError(
                        f"KVQuant quantizer tensor is truncated: {name}"
                    )
                tensor = torch.frombuffer(
                    storage,
                    dtype=torch.float32,
                ).clone()
                expected = 1
                for dimension in record["shape"]:
                    if type(dimension) is not int or dimension <= 0:
                        raise CacheStateError(
                            f"KVQuant quantizer shape differs: {name}"
                        )
                    expected *= dimension
                if int(tensor.numel()) != expected:
                    raise CacheStateError(
                        f"KVQuant quantizer tensor size differs: {name}"
                    )
                tensors[name] = tensor.reshape(tuple(record["shape"]))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CacheStateError("KVQuant quantizer cannot be read safely") from error
    if set(tensors) != required:
        raise CacheStateError("KVQuant quantizer tensor set differs")
    return tensors


def _required_quantizer_names() -> set[str]:
    names: set[str] = set()
    for layer in range(KVQUANT_NUM_LAYERS):
        prefix = f"layer_{layer:02d}"
        names.update(
            {
                f"{prefix}.k.codebook",
                f"{prefix}.k.lower_threshold",
                f"{prefix}.k.upper_threshold",
                f"{prefix}.v.codebook",
            }
        )
    return names


def _rope_inv_freq_cpu() -> Any:
    torch = _torch()
    base = 1.0 / (
        500000.0
        ** (
            torch.arange(0, KVQUANT_HEAD_DIM, 2, dtype=torch.float32)
            / KVQUANT_HEAD_DIM
        )
    )
    wavelength = 2.0 * torch.pi / base
    scaled = torch.where(wavelength > 8192.0, base / 8.0, base)
    smooth = (8192.0 / wavelength - 1.0) / 3.0
    smoothed = (1.0 - smooth) * base / 8.0 + smooth * base
    medium = (wavelength <= 8192.0) & (wavelength >= 2048.0)
    return torch.where(medium, smoothed, scaled).contiguous()


@dataclass(slots=True)
class KVQuantAttentionHandle:
    """Precreated identity handle consumed by the common model endpoint."""

    cache: KVQuantStaticCache
    layer_idx: int
    prefill: bool = False
    prefill_key_states: Any | None = None
    prefill_value_states: Any | None = None
    payload_slot: int = -1
    commit_after_decode: bool = False


class KVQuantMethodAdapter:
    """One static adapter for kvq4, kvq3, and kvq2."""

    name = "kvquant"
    adapter_version = KVQUANT_ADAPTER_VERSION
    requires_pre_rope_key = True

    def __init__(
        self,
        runtime_context: MethodRuntimeContext,
        config_name: str,
    ) -> None:
        if type(runtime_context) is not MethodRuntimeContext:
            raise TypeError("KVQuant adapter requires MethodRuntimeContext")
        if (
            runtime_context.num_layers,
            runtime_context.num_query_heads,
            runtime_context.num_kv_heads,
            runtime_context.head_dim,
        ) != (
            KVQUANT_NUM_LAYERS,
            KVQUANT_NUM_QUERY_HEADS,
            KVQUANT_NUM_KV_HEADS,
            KVQUANT_HEAD_DIM,
        ):
            raise ValueError("KVQuant adapter requires frozen Llama GQA geometry")
        if config_name not in KVQUANT_CONFIG_BITS:
            raise ValueError("unsupported KVQuant configuration")
        self.runtime_context = runtime_context
        self.config_name = config_name
        self.bits = KVQUANT_CONFIG_BITS[config_name]
        self.levels = 1 << self.bits
        self.quantizer_sha256 = KVQUANT_QUANTIZER_SHA256[config_name]
        self.adapter_version = (
            KVQUANT_Q4_ADAPTER_VERSION
            if config_name == "kvq4"
            else KVQUANT_ADAPTER_VERSION
        )
        self.fingerprint_schema_version = (
            KVQUANT_Q4_ADAPTER_FINGERPRINT_SCHEMA_VERSION
            if config_name == "kvq4"
            else KVQUANT_ADAPTER_FINGERPRINT_SCHEMA_VERSION
        )
        self.decisions = (
            KVQUANT_Q4_DECISIONS
            if config_name == "kvq4"
            else KVQUANT_DECISIONS
        )
        self._extension: ModuleType | None = None

    def prepare_runtime(self) -> None:
        """Validate the exact binary before prefill, warmup, or capture."""

        if self._extension is None:
            self._extension = _load_authorized_extension()

    def _runtime(self) -> ModuleType:
        if self._extension is None:
            raise CacheStateError(
                "KVQuant runtime authority was not prepared before execution"
            )
        return self._extension

    def allocate(
        self,
        *,
        batch_size: int,
        capacity: int,
        device: Any,
        workspace_bytes: int = 0,
    ) -> KVQuantStaticCache:
        cache = KVQuantStaticCache(
            config_name=self.config_name,
            num_layers=self.runtime_context.num_layers,
            batch_size=batch_size,
            num_query_heads=self.runtime_context.num_query_heads,
            num_kv_heads=self.runtime_context.num_kv_heads,
            capacity=capacity,
            head_dim=self.runtime_context.head_dim,
            device=device,
            workspace_bytes=workspace_bytes,
        )
        cache._kvquant_handles = {
            layer: KVQuantAttentionHandle(weakref.proxy(cache), layer)
            for layer in range(cache.num_layers)
        }
        cache._kvquant_initialized = False
        return cache

    def _require_cache(self, value: Any) -> KVQuantStaticCache:
        if type(value) is not KVQuantStaticCache:
            raise TypeError(
                "KVQuant adapter cache state must be KVQuantStaticCache"
            )
        if value.config_name != self.config_name:
            raise CacheStateError(
                "KVQuant adapter and cache configurations differ"
            )
        return value

    def initialize_cache_untimed(self, cache_state: Any) -> None:
        """Copy frozen calibration metadata into caller-owned CUDA tensors."""

        cache = self._require_cache(cache_state)
        root = _calibration_root()
        quantizer = root / "quantizers" / f"{self.config_name}.safetensors"
        if _sha256_file(quantizer) != self.quantizer_sha256:
            raise CacheStateError("KVQuant quantizer SHA-256 mismatch")
        tensors = _read_required_float_tensors(
            quantizer,
            _required_quantizer_names(),
        )
        torch = _torch()
        for layer in range(cache.num_layers):
            prefix = f"layer_{layer:02d}"
            codebook = tensors[f"{prefix}.k.codebook"].reshape(-1)
            value_codebook = tensors[f"{prefix}.v.codebook"].reshape(-1)
            if (
                int(codebook.numel()) != cache.levels
                or int(value_codebook.numel()) != cache.levels
            ):
                raise CacheStateError("KVQuant codebook level count differs")
            codebook = torch.sort(codebook.float()).values.contiguous()
            value_codebook = torch.sort(
                value_codebook.float()
            ).values.contiguous()
            # The frozen source rounds thresholds through FP16 before
            # constructing its float32 runtime metadata.
            lower_half = tensors[
                f"{prefix}.k.lower_threshold"
            ].reshape(-1).to(torch.float16)
            upper_half = tensors[
                f"{prefix}.k.upper_threshold"
            ].reshape(-1).to(torch.float16)
            if (
                int(lower_half.numel()) != KVQUANT_NUM_KV_HEADS * KVQUANT_HEAD_DIM
                or int(upper_half.numel()) != int(lower_half.numel())
            ):
                raise CacheStateError("KVQuant Key threshold geometry differs")
            midpoint = ((upper_half + lower_half) / 2.0).float()
            half_range = ((upper_half - lower_half) / 2.0).float()
            lookup = (
                half_range.reshape(-1, 1) * codebook.reshape(1, -1)
                + midpoint.reshape(-1, 1)
            ).reshape(
                KVQUANT_NUM_KV_HEADS,
                KVQUANT_HEAD_DIM,
                cache.levels,
            )
            cache.key_codebook[layer].copy_(codebook)
            cache.value_codebook[layer].copy_(value_codebook)
            cache.key_lower_threshold[layer].copy_(
                lower_half.float().reshape(
                    KVQUANT_NUM_KV_HEADS,
                    KVQUANT_HEAD_DIM,
                )
            )
            cache.key_upper_threshold[layer].copy_(
                upper_half.float().reshape(
                    KVQUANT_NUM_KV_HEADS,
                    KVQUANT_HEAD_DIM,
                )
            )
            cache.key_zero_point[layer].copy_(
                midpoint.reshape(
                    KVQUANT_NUM_KV_HEADS,
                    KVQUANT_HEAD_DIM,
                )
            )
            cache.key_lookup_table[layer].copy_(lookup)
        cache.rope_inv_freq.copy_(_rope_inv_freq_cpu())
        self.reset_cache_untimed(cache)
        cache._kvquant_initialized = True

    def reset_cache_untimed(self, cache_state: Any) -> None:
        """Clear mutable state for one fresh prefix without reallocating."""

        cache = self._require_cache(cache_state)
        for tensor in (
            cache.packed_key_cache,
            cache.packed_value_cache,
            cache.value_lookup_cache,
            cache.key_sparse_values,
            cache.key_sparse_indices,
            cache.value_sparse_values,
            cache.value_sparse_indices,
            cache.key_active_counts,
            cache.value_active_counts,
            cache.sink_key,
            cache.sink_value,
            cache.value_store_lower_bounds,
            cache.value_store_upper_bounds,
        ):
            tensor.zero_()
        cache.reset_active_length()

    @staticmethod
    def _handle(
        cache: KVQuantStaticCache,
        layer_idx: int,
    ) -> KVQuantAttentionHandle:
        try:
            handle = cache._kvquant_handles[layer_idx]
        except (AttributeError, KeyError) as error:
            raise CacheStateError(
                "KVQuant precreated attention handle is absent"
            ) from error
        handle.prefill = False
        handle.prefill_key_states = None
        handle.prefill_value_states = None
        handle.payload_slot = -1
        handle.commit_after_decode = False
        return handle

    @staticmethod
    def _validate_update(
        cache: KVQuantStaticCache,
        key_states: Any,
        value_states: Any,
        key_pre_rope_states: Any | None,
        layer_idx: int,
        cache_position: Any,
    ) -> int:
        torch = _torch()
        if not getattr(cache, "_kvquant_initialized", False):
            raise CacheStateError("KVQuant cache metadata is not initialized")
        if type(layer_idx) is not int or not 0 <= layer_idx < cache.num_layers:
            raise CacheStateError("KVQuant layer is outside static allocation")
        if key_pre_rope_states is None:
            raise CacheStateError("KVQuant requires pre-RoPE Key input")
        shape = tuple(int(item) for item in key_states.shape)
        if (
            int(key_states.ndim) != 4
            or tuple(int(item) for item in value_states.shape) != shape
            or tuple(int(item) for item in key_pre_rope_states.shape) != shape
            or shape[:2] != (cache.batch_size, cache.num_kv_heads)
            or shape[3] != cache.head_dim
        ):
            raise CacheStateError("KVQuant update has unsupported geometry")
        tensors = (key_states, value_states, key_pre_rope_states)
        if (
            any(tensor.dtype != torch.bfloat16 for tensor in tensors)
            or any(tensor.device != cache.device for tensor in tensors)
            or cache_position.device != cache.device
            or cache_position.dtype != torch.int64
        ):
            raise CacheStateError(
                "KVQuant update must use BF16 tensors on the cache device"
            )
        tokens = shape[2]
        if int(cache_position.ndim) != 1 or int(cache_position.shape[0]) != tokens:
            raise CacheStateError(
                "KVQuant cache position differs from update length"
            )
        return tokens

    def _pack_nonsink_token(
        self,
        cache: KVQuantStaticCache,
        *,
        layer_idx: int,
        batch_idx: int,
        payload_slot: int,
        key_pre_rope: Any,
        value: Any,
    ) -> None:
        runtime = self._runtime()
        key_staging = cache.key_pre_rope_bf16_staging[
            batch_idx : batch_idx + 1
        ]
        value_staging = cache.value_bf16_staging[
            batch_idx : batch_idx + 1
        ]
        key_staging.copy_(key_pre_rope)
        value_staging.copy_(value)
        cache.key_float_staging.copy_(
            key_staging.reshape(
                1, KVQUANT_NUM_KV_HEADS * KVQUANT_HEAD_DIM
            )
        )
        cache.value_float_staging.copy_(
            value_staging.reshape(
                1, KVQUANT_NUM_KV_HEADS * KVQUANT_HEAD_DIM
            )
        )
        cache.key_rescaled_staging.copy_(cache.key_float_staging)
        key_cache = cache.packed_key_cache[layer_idx, batch_idx]
        key_lookup = cache.key_lookup_table[layer_idx]
        key_lower = cache.key_lower_threshold[layer_idx].reshape(-1)
        key_upper = cache.key_upper_threshold[layer_idx].reshape(-1)
        getattr(runtime, f"vecquant{self.bits}appendvecKsparse")(
            key_cache,
            key_lookup,
            cache.key_float_staging.reshape(-1),
            cache.key_rescaled_staging.reshape(-1),
            key_lower,
            key_upper,
            payload_slot,
        )
        runtime.select_fixed_outliers_1024_cap12_out(
            cache.key_rescaled_staging,
            cache.key_selector_lower,
            cache.key_selector_upper,
            cache.selector_sink_mask,
            cache.selector_values,
            cache.selector_indices,
            cache.selector_count,
            cache.selector_dense_lower,
            cache.selector_dense_upper,
            0,
        )
        runtime.key_sparse_residual_1024_cap12_out(
            cache.key_float_staging,
            key_lookup,
            cache.selector_values,
            cache.selector_indices,
            cache.selector_count,
            cache.key_sparse_values[layer_idx, batch_idx],
            cache.key_sparse_indices[layer_idx, batch_idx],
            cache.key_active_counts[layer_idx, batch_idx],
            payload_slot,
            self.bits,
        )

        runtime.select_fixed_outliers_1024_cap12_out(
            cache.value_float_staging,
            cache.dummy_thresholds,
            cache.dummy_thresholds,
            cache.selector_sink_mask,
            cache.selector_values,
            cache.selector_indices,
            cache.selector_count,
            cache.selector_dense_lower,
            cache.selector_dense_upper,
            1,
        )
        cache.value_lower_bound_staging.copy_(cache.selector_dense_lower)
        cache.value_upper_bound_staging.copy_(cache.selector_dense_upper)
        cache.value_store_lower_bounds[
            batch_idx, payload_slot : payload_slot + 1
        ].copy_(cache.value_lower_bound_staging)
        cache.value_store_upper_bounds[
            batch_idx, payload_slot : payload_slot + 1
        ].copy_(cache.value_upper_bound_staging)
        cache.value_scale_staging.copy_(
            cache.value_upper_bound_staging
        ).sub_(cache.value_lower_bound_staging).mul_(0.5)
        cache.value_offset_staging.copy_(
            cache.value_upper_bound_staging
        ).add_(cache.value_lower_bound_staging).mul_(0.5)
        cache.value_metadata_row_staging.copy_(
            cache.value_codebook[layer_idx]
        ).mul_(cache.value_scale_staging).add_(cache.value_offset_staging)
        zero_code = KVQUANT_ZERO_CODE[self.config_name]
        cache.value_zero_point_staging.copy_(
            cache.value_metadata_row_staging[
                zero_code : zero_code + 1
            ]
        )
        runtime.append_value_sparse_1024_cap12_out(
            cache.packed_value_cache[layer_idx, batch_idx],
            cache.value_lookup_cache[layer_idx, batch_idx],
            cache.value_float_staging.reshape(-1),
            cache.value_metadata_row_staging,
            cache.value_lower_bound_staging,
            cache.value_upper_bound_staging,
            cache.value_zero_point_staging,
            cache.selector_values.reshape(-1),
            cache.selector_indices.reshape(-1),
            cache.selector_count,
            cache.value_sparse_values[layer_idx, batch_idx],
            cache.value_sparse_indices[layer_idx, batch_idx],
            cache.value_active_counts[layer_idx, batch_idx],
            payload_slot,
            self.bits,
        )

    def store_prefill(
        self,
        cache_state: Any,
        key_states: Any,
        value_states: Any,
        layer_idx: int,
        cache_position: Any,
        *,
        key_pre_rope_states: Any | None = None,
    ) -> tuple[Any, Any]:
        cache = self._require_cache(cache_state)
        if cache.mode != "prefill":
            raise CacheStateError("KVQuant prefill store requires prefill mode")
        tokens = self._validate_update(
            cache,
            key_states,
            value_states,
            key_pre_rope_states,
            layer_idx,
            cache_position,
        )
        if tokens > cache.capacity:
            raise CacheStateError("KVQuant prefill exceeds static capacity")
        sink = min(tokens, cache.sink_tokens)
        if sink:
            cache.sink_key[
                layer_idx, :, :, :, :sink
            ].copy_(key_states[:, :, :sink, :].transpose(2, 3))
            cache.sink_value[
                layer_idx, :, :, :sink, :
            ].copy_(value_states[:, :, :sink, :])
        if key_pre_rope_states is None:  # narrowed by _validate_update
            raise CacheStateError("KVQuant pre-RoPE Key disappeared")
        for position in range(cache.sink_tokens, tokens):
            payload_slot = position - cache.sink_tokens
            cache.zero_payload_slot(
                layer_idx=layer_idx,
                payload_slot=payload_slot,
            )
            for batch_idx in range(cache.batch_size):
                self._pack_nonsink_token(
                    cache,
                    layer_idx=layer_idx,
                    batch_idx=batch_idx,
                    payload_slot=payload_slot,
                    key_pre_rope=key_pre_rope_states[
                        batch_idx : batch_idx + 1,
                        :,
                        position : position + 1,
                        :,
                    ],
                    value=value_states[
                        batch_idx : batch_idx + 1,
                        :,
                        position : position + 1,
                        :,
                    ],
                )
        quantized_tokens = tokens - sink
        if quantized_tokens:
            # Phase 11P-R corrected the exact upstream parallel Value-store
            # kernel.  The caller-owned append API above freezes metadata and
            # sparse state per row; this final store-only pass overwrites the
            # dense prefix through that corrected source path.  The temporary
            # FP32 layout conversion occurs only during untimed prefill, never
            # in measured append/decode or graph replay.
            for batch_idx in range(cache.batch_size):
                value_parallel = (
                    value_states[batch_idx, :, sink:tokens, :]
                    .transpose(1, 2)
                    .float()
                    .contiguous()
                )
                cache.packed_value_cache[
                    layer_idx,
                    batch_idx,
                    :,
                    :,
                    :quantized_tokens,
                ].zero_()
                getattr(
                    self._runtime(),
                    f"vecquant{self.bits}appendvecVsparseParallel",
                )(
                    cache.packed_value_cache[layer_idx, batch_idx],
                    cache.value_lookup_cache[layer_idx, batch_idx],
                    value_parallel,
                    cache.value_store_lower_bounds[
                        batch_idx, :quantized_tokens
                    ],
                    cache.value_store_upper_bounds[
                        batch_idx, :quantized_tokens
                    ],
                )
        handle = self._handle(cache, layer_idx)
        handle.prefill = True
        handle.prefill_key_states = key_states
        handle.prefill_value_states = value_states
        return handle, handle

    def append_decode(
        self,
        cache_state: Any,
        key_states: Any,
        value_states: Any,
        layer_idx: int,
        cache_position: Any,
        *,
        key_pre_rope_states: Any | None = None,
    ) -> tuple[Any, Any]:
        cache = self._require_cache(cache_state)
        if cache.mode not in {"fixed", "growing_step"}:
            raise CacheStateError(
                "KVQuant append requires fixed or growing mode"
            )
        tokens = self._validate_update(
            cache,
            key_states,
            value_states,
            key_pre_rope_states,
            layer_idx,
            cache_position,
        )
        if tokens != 1 or key_pre_rope_states is None:
            raise CacheStateError("KVQuant append requires exactly one token")
        if cache.mode == "fixed":
            payload_slot = cache.fixed_slot(layer_idx)
        else:
            payload_slot = cache.growing_slot(layer_idx)
        cache.validate_decode_position_binding(
            cache_position,
            payload_slot=payload_slot,
        )
        cache.zero_payload_slot(
            layer_idx=layer_idx,
            payload_slot=payload_slot,
        )
        for batch_idx in range(cache.batch_size):
            self._pack_nonsink_token(
                cache,
                layer_idx=layer_idx,
                batch_idx=batch_idx,
                payload_slot=payload_slot,
                key_pre_rope=key_pre_rope_states[
                    batch_idx : batch_idx + 1
                ],
                value=value_states[batch_idx : batch_idx + 1],
            )
        handle = self._handle(cache, layer_idx)
        handle.payload_slot = payload_slot
        handle.commit_after_decode = cache.mode == "growing_step"
        return handle, handle

    def _decode_compressed(
        self,
        handle: KVQuantAttentionHandle,
        query_states: Any,
        scaling: float,
    ) -> Any:
        torch = _torch()
        cache = handle.cache
        if cache.device.type != "cuda":
            raise CacheStateError("KVQuant compressed kernels require CUDA")
        if (
            tuple(int(item) for item in query_states.shape)
            != (
                cache.batch_size,
                KVQUANT_NUM_QUERY_HEADS,
                1,
                KVQUANT_HEAD_DIM,
            )
            or query_states.dtype != torch.bfloat16
            or query_states.device != cache.device
            or type(scaling) is not float
            or scaling <= 0.0
        ):
            raise CacheStateError("KVQuant decode query geometry differs")
        total = cache.active_context + 1
        quantized = total - cache.sink_tokens
        if quantized <= 0 or total > cache.capacity:
            raise CacheStateError("KVQuant decode context is invalid")
        runtime = self._runtime()
        cache.query_bf16_staging.copy_(query_states[:, :, 0, :])
        cache.query_float_staging.copy_(cache.query_bf16_staging)
        # The source stores attention-ready sink K in FP16.
        cache.sink_output_fp16.copy_(cache.query_bf16_staging)
        for query_head in range(KVQUANT_NUM_QUERY_HEADS):
            kv_head = query_head // (
                KVQUANT_NUM_QUERY_HEADS // KVQUANT_NUM_KV_HEADS
            )
            torch.bmm(
                cache.sink_output_fp16[
                    :, query_head : query_head + 1, :
                ],
                cache.sink_key[
                    handle.layer_idx, :, kv_head, :, :
                ],
                out=cache.sink_logits_fp16[
                    :, query_head : query_head + 1, :
                ],
            )

        cache.decode_logits_bf16.fill_(float("-inf"))
        cache.decode_logits_bf16[
            :, :, : cache.sink_tokens
        ].copy_(cache.sink_logits_fp16)
        key_api = getattr(
            runtime,
            (
                f"vecquant{self.bits}matmul_nuq_perchannel_transposed_"
                "rope_mha_batched_fused_opt2"
            ),
        )
        for batch_idx in range(cache.batch_size):
            key_output = cache.decode_logits[batch_idx].reshape(-1)[
                : KVQUANT_NUM_QUERY_HEADS * quantized
            ].view(1, KVQUANT_NUM_QUERY_HEADS, quantized)
            key_output.zero_()
            key_api(
                cache.query_float_staging[batch_idx : batch_idx + 1],
                cache.packed_key_cache[handle.layer_idx, batch_idx],
                key_output,
                cache.key_lookup_table[handle.layer_idx],
                quantized,
                cache.key_sparse_values[handle.layer_idx, batch_idx],
                cache.key_sparse_indices[handle.layer_idx, batch_idx],
                cache.rope_inv_freq,
                cache.sink_tokens,
            )
            for query_head in range(KVQUANT_NUM_QUERY_HEADS):
                cache.decode_logits_bf16[
                    batch_idx : batch_idx + 1,
                    query_head,
                    cache.sink_tokens : total,
                ].copy_(key_output[:, query_head, :])
        cache.decode_logits_bf16.mul_(scaling)
        # Passing a BF16 input together with ``dtype=float32`` makes PyTorch
        # allocate a full logits-sized conversion temporary even when ``out``
        # is caller-owned.  Preserve the same BF16-rounded input semantics by
        # copying into the preallocated FP32 softmax workspace first, then run
        # the FP32 softmax in place.
        cache.decode_softmax.copy_(cache.decode_logits_bf16)
        torch.softmax(
            cache.decode_softmax,
            dim=-1,
            out=cache.decode_softmax,
        )
        cache.decode_logits_bf16.copy_(cache.decode_softmax)

        cache.decode_quantized_output.zero_()
        for batch_idx in range(cache.batch_size):
            if self.bits == 4:
                value_weights = cache.decode_logits[batch_idx].reshape(-1)[
                    : KVQUANT_NUM_QUERY_HEADS * quantized
                ].view(1, KVQUANT_NUM_QUERY_HEADS, quantized)
                for query_head in range(KVQUANT_NUM_QUERY_HEADS):
                    value_weights[:, query_head, :].copy_(
                        cache.decode_logits_bf16[
                            batch_idx : batch_idx + 1,
                            query_head,
                            cache.sink_tokens : total,
                        ]
                    )
            else:
                value_weights = cache.decode_softmax[
                    batch_idx : batch_idx + 1,
                    :,
                    cache.sink_tokens : total,
                ]
            self._decode_quantized_value(
                runtime=runtime,
                cache=cache,
                layer_idx=handle.layer_idx,
                batch_idx=batch_idx,
                value_weights=value_weights,
                quantized=quantized,
            )
        cache.output_bf16_staging.copy_(cache.decode_quantized_output)
        cache.sink_logits_fp16.copy_(
            cache.decode_logits_bf16[:, :, : cache.sink_tokens]
        )
        for query_head in range(KVQUANT_NUM_QUERY_HEADS):
            kv_head = query_head // (
                KVQUANT_NUM_QUERY_HEADS // KVQUANT_NUM_KV_HEADS
            )
            torch.bmm(
                cache.sink_logits_fp16[
                    :, query_head : query_head + 1, :
                ],
                cache.sink_value[
                    handle.layer_idx, :, kv_head, :, :
                ],
                out=cache.sink_output_fp16[
                    :, query_head : query_head + 1, :
                ],
            )
        cache.output_bf16_staging.add_(cache.sink_output_fp16)
        if (
            handle.commit_after_decode
            and handle.layer_idx == cache.num_layers - 1
        ):
            cache.finish_growing_step()
        return cache.output_bf16_staging.unsqueeze(2)

    def _decode_quantized_value(
        self,
        *,
        runtime: ModuleType,
        cache: KVQuantStaticCache,
        layer_idx: int,
        batch_idx: int,
        value_weights: Any,
        quantized: int,
    ) -> None:
        """Dispatch every bit width through its deterministic out API."""

        if self.bits == 4:
            workspace = cache.require_q4_value_decode_workspace(
                quantized_length=quantized
            )
        else:
            workspace = cache.q23_value_decode_workspace
            if workspace is None:
                raise CacheStateError(
                    "KVQuant q3/q2 deterministic workspace differs"
                )
        output = cache.decode_quantized_output[
            batch_idx : batch_idx + 1
        ]
        output.zero_()
        getattr(
            runtime,
            KVQUANT_DETERMINISTIC_VALUE_DECODE_APIS[self.bits],
        )(
            value_weights,
            cache.packed_value_cache[layer_idx, batch_idx],
            output,
            cache.value_lookup_cache[layer_idx, batch_idx],
            quantized,
            cache.value_sparse_values[layer_idx, batch_idx],
            cache.value_sparse_indices[layer_idx, batch_idx],
            workspace[batch_idx : batch_idx + 1],
        )

    def decode_attention(
        self,
        attention: Any,
        query_states: Any,
        key_states: Any,
        value_states: Any,
        *,
        scaling: float,
    ) -> Any:
        if (
            type(key_states) is not KVQuantAttentionHandle
            or key_states is not value_states
        ):
            raise CacheStateError("KVQuant attention handle identity differs")
        handle = key_states
        if handle.prefill:
            if (
                handle.prefill_key_states is None
                or handle.prefill_value_states is None
            ):
                raise CacheStateError("KVQuant prefill attention state is absent")
            output, _ = flash_attention_forward(
                attention,
                query_states,
                handle.prefill_key_states,
                handle.prefill_value_states,
                None,
                scaling,
                dropout=0.0,
            )
            handle.prefill_key_states = None
            handle.prefill_value_states = None
            return output
        return self._decode_compressed(handle, query_states, scaling)

    def allocated_bytes(self, cache_state: Any) -> int:
        return self._require_cache(cache_state).accounting().allocated_bytes

    def byte_breakdown(self, cache_state: Any) -> Mapping[str, int]:
        return self._require_cache(cache_state).byte_breakdown()

    def logical_bf16_bytes(self, cache_state: Any) -> int:
        return self._require_cache(cache_state).logical_bf16_storage_bytes

    def config_fingerprint(self, cache_layout_fingerprint: str) -> str:
        require_sha256(
            cache_layout_fingerprint,
            field_name="cache_layout_fingerprint",
        )
        context = self.runtime_context
        payload = {
            "schema_version": self.fingerprint_schema_version,
            "adapter_version": self.adapter_version,
            "method_name": self.name,
            "configuration": self.config_name,
            "bits": self.bits,
            "method_identifier": KVQUANT_METHOD_IDENTIFIER,
            "execution_source_identifier": (
                KVQUANT_EXECUTION_SOURCE_IDENTIFIER
            ),
            "upstream_base_commit": KVQUANT_UPSTREAM_BASE_COMMIT,
            "upstream_base_tree": KVQUANT_UPSTREAM_BASE_TREE,
            "decision_0021_patch_sha256": (
                KVQUANT_DECISION_0021_PATCH_SHA256
            ),
            "aggregate_patch_sha256": KVQUANT_AGGREGATE_PATCH_SHA256,
            "corrected_commit": KVQUANT_CORRECTED_COMMIT,
            "corrected_tree": KVQUANT_CORRECTED_TREE,
            "corrected_cuda_sha256": KVQUANT_CORRECTED_CUDA_SHA256,
            "extension_sha256": KVQUANT_EXTENSION_SHA256,
            "decisions": list(self.decisions),
            "calibration_id": KVQUANT_CALIBRATION_ID,
            "calibration_root_sha256": KVQUANT_CALIBRATION_ROOT_SHA256,
            "quantizer_sha256": self.quantizer_sha256,
            "historical_fixture_id": KVQUANT_HISTORICAL_FIXTURE_ID,
            "historical_fixture_root_sha256": (
                KVQUANT_HISTORICAL_ROOT_SHA256
            ),
            "fixture_id": KVQUANT_FIXTURE_ID,
            "fixture_root_sha256": KVQUANT_FIXTURE_ROOT_SHA256,
            "authorized_container_digest": (
                KVQUANT_AUTHORIZED_CONTAINER_DIGEST
            ),
            "cache_layout_fingerprint": cache_layout_fingerprint,
            "model_id": context.model_id,
            "model_revision": context.model_revision,
            "backend_id": context.backend_id,
            "backend_fingerprint": context.backend_fingerprint,
            "geometry": {
                "layers": context.num_layers,
                "query_heads": context.num_query_heads,
                "kv_heads": context.num_kv_heads,
                "groups": context.num_query_heads // context.num_kv_heads,
                "head_dim": context.head_dim,
            },
            "semantics": {
                "quantized_key": "pre_rope",
                "sink_key": "post_rope_attention_ready_fp16",
                "value": "post_v_proj_native_hkv",
                "query_to_kv": "query_head//4",
                "key_cap": KVQUANT_KEY_CAP,
                "value_cap": KVQUANT_VALUE_CAP,
                "value_nonsink_active": KVQUANT_VALUE_CAP,
                "value_sink_active": 0,
                "sparse_value_dtype": "float32",
                "sparse_index_dtype": "int32",
                "value_decode_api": (
                    KVQUANT_DETERMINISTIC_VALUE_DECODE_APIS[self.bits]
                ),
                "value_decode_workspace": (
                    "dedicated_q4"
                    if self.bits == 4
                    else "decode_logits_alias_no_additional_storage"
                ),
                "q4_value_decode_workspace_shape": (
                    "capacity_bound_in_cache_layout_fingerprint"
                    if self.bits == 4
                    else None
                ),
            },
            "supports_cuda_graph": self.supports_cuda_graph(),
            "r_hbm": None,
            "implementation_sha256": (
                hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                if self.bits == 4
                else KVQUANT_LEGACY_ADAPTER_IMPLEMENTATION_SHA256
            ),
        }

        if self.bits == 4:
            payload["semantics"].update(
                {
                    "q4_value_decode_tile_size": 128,
                    "q4_value_decode_workspace_formula_version": (
                        KVQUANT_Q4_VALUE_DECODE_WORKSPACE_FORMULA_VERSION
                    ),
                    "q4_value_decode_total_attended_capacity_source": (
                        "declared_cache_capacity"
                    ),
                    "q4_value_decode_workspace_preallocated": True,
                    "q4_value_decode_workspace_resizable": False,
                    "decision_0036": True,
                }
            )
        return sha256_hex(canonical_json_bytes(payload))

    def supports_cuda_graph(self) -> bool:
        return True
