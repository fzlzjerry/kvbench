"""Static Phase 8 adapter for the checksum-bound patched KIVI reference.

This is deliberately a small wrapper: the quantization kernels and compressed
GEMV remain the pinned upstream implementation.  It owns no alternative
quantizer, packing format, or attention backend.
"""

from __future__ import annotations

import ast
import ctypes
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping
import weakref

from kvbench.adapters.base import MethodRuntimeContext
from kvbench.runtime.backend import flash_attention_forward
from kvbench.runtime.kivi_cache import (
    KIVI_CONFIG_BITS,
    KIVI_GROUP_SIZE,
    KIVI_RESIDUAL_LENGTH,
    KIVIStaticCache,
)
from kvbench.runtime.static_cache import CacheStateError
from kvbench.schema import canonical_json_bytes, sha256_hex
from kvbench.schema.base import require_sha256


KIVI_ADAPTER_VERSION = "kvbench-kivi-method-adapter-1.1.0"
KIVI_ADAPTER_FINGERPRINT_SCHEMA_VERSION = "kvbench-kivi-method-adapter-config-1.1.0"
KIVI_OFFICIAL_COMMIT = "876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6"
KIVI_OFFICIAL_BASE_TREE = "c94c31b2cfd44eeb9a18cff9dcdf03adff4ac49b"
KIVI_PATCHED_TREE = "b617493dea5aff1a754cd27ad6be12ac512b2aee"
KIVI_DECISION_0018_PATCH_SHA256 = (
    "c9c2dd52d4c81b844d1d1d7218ad2cd60a5b31574a387f716d466cb01310423d"
)
KIVI_EXTENSION_SHA256 = (
    "45d29ec1a3cecc4b253d1d1dd6139ef4f91cff88993db61a9d73685314851aa9"
)
KIVI_NEW_PACK_SHA256 = (
    "3678af0e34a0ba18e5d80a4128acf11d4070667c800a15540a16d07253a4f75e"
)
KIVI_FIXTURE_ROOT_SHA256 = (
    "abd164da0adf9e0c1404e8fba1f6a6e42e57944481cdf060b91e8cef175ed302"
)
KIVI_GQA_GROUP_SIZE = 4
# Local host-stub offsets in the exact checksum-bound ELF.  They name the
# original compiled bgemv kernel families; no replacement CUDA is compiled.
KIVI_BGEMV4_HOST_STUB_OFFSET = 0xA970
KIVI_BGEMV2_HOST_STUB_OFFSET = 0xAB70
_TORCH: Any | None = None


def _torch() -> Any:
    global _TORCH
    if _TORCH is None:
        try:
            _TORCH = importlib.import_module("torch")
        except ModuleNotFoundError as error:  # pragma: no cover - environment
            raise CacheStateError("PyTorch is required for the KIVI adapter") from error
    return _TORCH


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise CacheStateError("KIVI authority path must be a regular file")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _kivi_source_root() -> Path:
    """Return only the mounted, checksum-bound KIVI source root.

    The override is solely for the authorized container mount used by tests and
    admission; it is never a package-install or source-discovery mechanism.
    """

    raw = os.environ.get("KVBENCH_KIVI_SOURCE_ROOT", "/opt/kivi-source")
    try:
        path = Path(raw).resolve(strict=True)
    except FileNotFoundError as error:
        raise CacheStateError("KIVI source root is unavailable") from error
    if path.is_symlink() or not path.is_dir():
        raise CacheStateError("KIVI source root is not an authorized directory")
    return path


@dataclass(frozen=True, slots=True)
class KIVIKernelLaunchRecord:
    """One actual successful launch collected outside measured execution."""

    kernel_family: str
    bits: int
    input_shape: tuple[int, ...]
    packed_shape: tuple[int, ...]
    metadata_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    group_size: int
    num_query_heads: int
    num_kv_heads: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "kernel_family": self.kernel_family,
            "bits": self.bits,
            "input_shape": list(self.input_shape),
            "packed_shape": list(self.packed_shape),
            "metadata_shape": list(self.metadata_shape),
            "output_shape": list(self.output_shape),
            "group_size": self.group_size,
            "num_query_heads": self.num_query_heads,
            "num_kv_heads": self.num_kv_heads,
        }


class _Dim3(ctypes.Structure):
    _fields_ = (
        ("x", ctypes.c_uint),
        ("y", ctypes.c_uint),
        ("z", ctypes.c_uint),
    )


class _KIVIDirectGEMVLauncher:
    """Launch the exact registered KIVI kernels into caller-owned storage."""

    def __init__(self, extension_path: Path) -> None:
        base_address: int | None = None
        try:
            mappings = Path("/proc/self/maps").read_text(
                encoding="utf-8"
            ).splitlines()
        except (OSError, UnicodeDecodeError) as error:
            raise CacheStateError("KIVI extension mapping cannot be read") from error
        for line in mappings:
            fields = line.split(maxsplit=5)
            if (
                len(fields) == 6
                and fields[5] == str(extension_path)
                and int(fields[2], 16) == 0
            ):
                base_address = int(fields[0].split("-", 1)[0], 16)
                break
        if base_address is None:
            raise CacheStateError("KIVI extension mapping is absent")
        try:
            cudart = ctypes.CDLL("libcudart.so.13")
        except OSError as error:
            raise CacheStateError("authorized CUDA runtime is unavailable") from error
        launch = cudart.cudaLaunchKernel
        launch.argtypes = (
            ctypes.c_void_p,
            _Dim3,
            _Dim3,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_void_p,
        )
        launch.restype = ctypes.c_int
        self._launch = launch
        self._kernel_pointers = {
            2: ctypes.c_void_p(
                base_address + KIVI_BGEMV2_HOST_STUB_OFFSET
            ),
            4: ctypes.c_void_p(
                base_address + KIVI_BGEMV4_HOST_STUB_OFFSET
            ),
        }
        self._observation: list[KIVIKernelLaunchRecord] | None = None

    def begin_observation(self) -> None:
        """Enable a narrow untimed launch audit."""

        if self._observation is not None:
            raise CacheStateError("KIVI launch observation is already active")
        self._observation = []

    def end_observation(self) -> tuple[KIVIKernelLaunchRecord, ...]:
        """Return actual launches and disable observation."""

        if self._observation is None:
            raise CacheStateError("KIVI launch observation is not active")
        observed = tuple(self._observation)
        self._observation = None
        return observed

    def launch_into(
        self,
        *,
        input_tensor: Any,
        packed: Any,
        scales: Any,
        minimums: Any,
        output: Any,
        bits: int,
        group_size: int,
        num_query_heads: int,
        num_kv_heads: int,
    ) -> None:
        """Launch without constructing an extension-owned output tensor."""

        torch = _torch()
        if (
            bits not in self._kernel_pointers
            or group_size != KIVI_GROUP_SIZE
            or num_query_heads != 32
            or num_kv_heads != 8
            or input_tensor.dtype != torch.float16
            or packed.dtype != torch.int32
            or scales.dtype != torch.float16
            or minimums.dtype != torch.float16
            or output.dtype != torch.float16
            or not all(
                tensor.device == input_tensor.device
                for tensor in (packed, scales, minimums, output)
            )
            or input_tensor.device.type != "cuda"
            or not all(
                tensor.is_contiguous()
                for tensor in (
                    input_tensor,
                    packed,
                    scales,
                    minimums,
                    output,
                )
            )
        ):
            raise CacheStateError("KIVI direct GEMV launch operands differ")
        batch_heads = int(input_tensor.shape[0])
        rows = int(input_tensor.shape[1])
        input_channels = int(input_tensor.shape[2])
        output_channels = int(minimums.shape[1]) * group_size
        if int(output.numel()) != batch_heads * rows * output_channels:
            raise CacheStateError("KIVI direct GEMV output geometry differs")
        values = (
            ctypes.c_void_p(int(input_tensor.data_ptr())),
            ctypes.c_void_p(int(packed.data_ptr())),
            ctypes.c_void_p(int(minimums.data_ptr())),
            ctypes.c_void_p(int(scales.data_ptr())),
            ctypes.c_void_p(int(output.data_ptr())),
            ctypes.c_int(input_channels),
            ctypes.c_int(output_channels),
            ctypes.c_int(group_size),
            ctypes.c_int(num_query_heads),
            ctypes.c_int(num_kv_heads),
        )
        arguments = (ctypes.c_void_p * len(values))(
            *(
                ctypes.cast(ctypes.pointer(value), ctypes.c_void_p)
                for value in values
            )
        )
        pack_factor = 32 // bits
        grid = _Dim3(
            batch_heads,
            (output_channels // pack_factor + 3) // 4,
            rows,
        )
        block = _Dim3(32, 4, 1)
        stream = ctypes.c_void_p(
            int(torch.cuda.current_stream(input_tensor.device).cuda_stream)
        )
        error = self._launch(
            self._kernel_pointers[bits],
            grid,
            block,
            arguments,
            0,
            stream,
        )
        if error != 0:
            raise CacheStateError(
                f"official KIVI CUDA launch failed with error {error}"
            )
        if self._observation is not None:
            self._observation.append(
                KIVIKernelLaunchRecord(
                    kernel_family=f"bgemv{bits}_kernel_outer_dim",
                    bits=bits,
                    input_shape=tuple(int(item) for item in input_tensor.shape),
                    packed_shape=tuple(int(item) for item in packed.shape),
                    metadata_shape=tuple(int(item) for item in scales.shape),
                    output_shape=tuple(int(item) for item in output.shape),
                    group_size=group_size,
                    num_query_heads=num_query_heads,
                    num_kv_heads=num_kv_heads,
                )
            )


def _load_authorized_kivi_runtime(
) -> tuple[ModuleType, ModuleType, _KIVIDirectGEMVLauncher]:
    """Load the extension and only the two checksum-bound Triton kernels."""

    try:
        specification = importlib.util.find_spec("kivi_gemv")
    except (ImportError, ValueError) as error:
        raise CacheStateError("KIVI extension specification is unavailable") from error
    if specification is None or not specification.origin:
        raise CacheStateError("KIVI extension specification is unavailable")
    try:
        extension_path = Path(specification.origin).resolve(strict=True)
    except FileNotFoundError as error:
        raise CacheStateError("KIVI extension path is unavailable") from error
    if not extension_path.name.startswith("kivi_gemv") or extension_path.suffix != ".so":
        raise CacheStateError("KIVI extension module identity is invalid")
    if _sha256_file(extension_path) != KIVI_EXTENSION_SHA256:
        raise CacheStateError("KIVI extension SHA-256 mismatch")

    source_path = _kivi_source_root() / "quant" / "new_pack.py"
    if _sha256_file(source_path) != KIVI_NEW_PACK_SHA256:
        raise CacheStateError("KIVI new_pack.py SHA-256 mismatch")
    extension = importlib.import_module("kivi_gemv")
    try:
        loaded_path = Path(str(getattr(extension, "__file__", ""))).resolve(
            strict=True
        )
    except FileNotFoundError as error:
        raise CacheStateError("loaded KIVI extension path is unavailable") from error
    if loaded_path != extension_path:
        raise CacheStateError("loaded KIVI extension differs from verified authority")

    # The authorized Measurement Container deliberately has no NumPy.  NumPy
    # is used only by upstream allocating convenience functions that Phase 8
    # forbids.  Parse the checksum-verified file and compile the two exact
    # official Triton definitions plus their imports, without vendoring or
    # rewriting either kernel.
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), source_path)
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise CacheStateError("KIVI new_pack.py cannot be parsed") from error
    kernel_names = {"_minmax_along_last_dim", "_pack_along_last_dim"}
    imports = [
        node
        for node in tree.body
        if isinstance(node, ast.Import)
        and all(alias.name in {"triton", "triton.language"} for alias in node.names)
    ]
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in kernel_names
    ]
    if {node.name for node in definitions} != kernel_names or len(definitions) != 2:
        raise CacheStateError("authorized KIVI pack kernel definitions differ")
    selected = ast.Module(body=[*imports, *definitions], type_ignores=[])
    ast.fix_missing_locations(selected)
    pack = ModuleType("kvbench_authorized_kivi_new_pack_kernels")
    try:
        exec(compile(selected, str(source_path), "exec"), pack.__dict__)
    except (ImportError, RuntimeError) as error:
        raise CacheStateError(
            "required KIVI Triton kernel dependency is unavailable"
        ) from error
    for name in ("_minmax_along_last_dim", "_pack_along_last_dim"):
        if not hasattr(pack, name):
            raise CacheStateError(f"authorized KIVI pack kernel is absent: {name}")
    if not hasattr(extension, "gemv_forward_cuda_outer_dim"):
        raise CacheStateError("authorized KIVI GEMV entry point is absent")
    return extension, pack, _KIVIDirectGEMVLauncher(extension_path)


@dataclass(slots=True)
class KIVIAttentionHandle:
    """Identity-only cache handle consumed by the existing model endpoint."""

    cache: KIVIStaticCache
    layer_idx: int
    prefill: bool
    prefill_key_states: Any | None = None
    prefill_value_states: Any | None = None
    pending_key: Any | None = None
    pending_value: Any | None = None
    commit_after_decode: bool = False


class KIVIMethodAdapter:
    """One static adapter for the four frozen KIVI bit configurations."""

    name = "kivi"
    adapter_version = KIVI_ADAPTER_VERSION

    def __init__(self, runtime_context: MethodRuntimeContext, config_name: str) -> None:
        if type(runtime_context) is not MethodRuntimeContext:
            raise TypeError("KIVI adapter requires MethodRuntimeContext")
        if (
            runtime_context.num_layers,
            runtime_context.num_query_heads,
            runtime_context.num_kv_heads,
            runtime_context.head_dim,
        ) != (32, 32, 8, 128):
            raise ValueError("KIVI adapter requires frozen Llama GQA geometry")
        if config_name not in KIVI_CONFIG_BITS:
            raise ValueError("unsupported KIVI configuration")
        self.runtime_context = runtime_context
        self.config_name = config_name
        self.k_bits, self.v_bits = KIVI_CONFIG_BITS[config_name]
        self._runtime_modules: (
            tuple[ModuleType, ModuleType, _KIVIDirectGEMVLauncher] | None
        ) = None

    def prepare_runtime(self) -> None:
        """Validate and cache authority before any measured operation."""

        if self._runtime_modules is None:
            self._runtime_modules = _load_authorized_kivi_runtime()

    def _runtime(
        self,
    ) -> tuple[ModuleType, ModuleType, _KIVIDirectGEMVLauncher]:
        if self._runtime_modules is None:
            raise CacheStateError(
                "KIVI runtime authority was not prepared before execution"
            )
        return self._runtime_modules

    def allocate(
        self,
        *,
        batch_size: int,
        capacity: int,
        device: Any,
        workspace_bytes: int = 0,
    ) -> KIVIStaticCache:
        cache = KIVIStaticCache(
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
        cache._kivi_handles = {
            layer: KIVIAttentionHandle(weakref.proxy(cache), layer, False)
            for layer in range(cache.num_layers)
        }
        return cache

    def _require_cache(self, value: Any) -> KIVIStaticCache:
        if type(value) is not KIVIStaticCache:
            raise TypeError("KIVI adapter cache state must be KIVIStaticCache")
        if value.config_name != self.config_name:
            raise CacheStateError("KIVI adapter and cache configurations differ")
        return value

    @staticmethod
    def _handle(cache: KIVIStaticCache, layer_idx: int) -> KIVIAttentionHandle:
        try:
            handle = cache._kivi_handles[layer_idx]
        except (AttributeError, KeyError) as error:
            raise CacheStateError("KIVI precreated attention handle is absent") from error
        handle.prefill = False
        handle.prefill_key_states = None
        handle.prefill_value_states = None
        handle.pending_key = None
        handle.pending_value = None
        handle.commit_after_decode = False
        return handle

    @staticmethod
    def _validate_update(
        cache: KIVIStaticCache,
        key_states: Any,
        value_states: Any,
        layer_idx: int,
        cache_position: Any,
    ) -> int:
        if type(layer_idx) is not int or not 0 <= layer_idx < cache.num_layers:
            raise CacheStateError("KIVI layer is outside the static allocation")
        shape = tuple(int(item) for item in key_states.shape)
        if (
            int(key_states.ndim) != 4
            or tuple(int(item) for item in value_states.shape) != shape
            or shape[:2] != (cache.batch_size, cache.num_kv_heads)
            or shape[3] != cache.head_dim
        ):
            raise CacheStateError("KIVI cache update has unsupported geometry")
        if (
            key_states.dtype != _torch().bfloat16
            or value_states.dtype != _torch().bfloat16
            or key_states.device != cache.device
            or value_states.device != cache.device
            or cache_position.device != cache.device
            or cache_position.dtype != _torch().int64
        ):
            raise CacheStateError("KIVI update must use BF16 tensors on the cache device")
        tokens = shape[2]
        if int(cache_position.ndim) != 1 or int(cache_position.shape[0]) != tokens:
            raise CacheStateError("KIVI cache position differs from update length")
        return tokens

    @staticmethod
    def _quantization_scratch(cache: KIVIStaticCache) -> tuple[Any, Any, Any]:
        """Require explicit static scratch; never substitute allocating helpers."""

        try:
            fp16 = cache.quantization_fp16_staging
            integers = cache.quantization_int_staging
            packed = cache.quantization_packed_staging
        except AttributeError as error:
            raise CacheStateError(
                "KIVI static cache lacks required preallocated quantization scratch"
            ) from error
        return fp16, integers, packed

    @staticmethod
    def _quantize_into(
        *,
        source: Any,
        scale: Any,
        minimum: Any,
        destination: Any,
        bits: int,
        pack: ModuleType,
        fp16_scratch: Any,
        int_scratch: Any,
        packed_scratch: Any,
    ) -> None:
        """Run the two official Triton kernels with only preallocated buffers."""

        width = int(source.shape[-1])
        if width not in (KIVI_GROUP_SIZE, 128) or bits not in (2, 4):
            raise CacheStateError("KIVI quantization has unsupported group geometry")
        rows = int(source.shape[0] * source.shape[1] * source.shape[2])
        # The staging allocation is physically [B,H,D,32].  Its leading
        # contiguous rows also present the one-token V [B,H,1,128] ABI
        # without allocating or relaying a prefix.
        scratch = fp16_scratch.view(-1)[: rows * width].view(*source.shape)
        integer = int_scratch.view(-1)[: rows * width].view(*source.shape)
        packed_width = width * bits // 32
        packed = packed_scratch.view(-1)[: rows * packed_width].view(
            *source.shape[:-1], packed_width
        )
        if tuple(scratch.shape) != tuple(source.shape):
            raise CacheStateError("KIVI quantization staging geometry is invalid")
        scratch.copy_(source)
        groups = width // KIVI_GROUP_SIZE
        block = 128
        grid = ((rows * groups + block - 1) // block,)
        # These are the checksum-verified official kernel objects, invoked
        # directly rather than the upstream allocating convenience launcher.
        pack._minmax_along_last_dim[grid](
            scratch,
            minimum,
            scale,
            int(scratch.numel()),
            rows,
            groups,
            KIVI_GROUP_SIZE,
            BLOCK_SIZE_N=block,
            num_warps=8,
        )
        scale.sub_(minimum).div_(float((1 << bits) - 1))
        grouped = scratch.view(rows, groups, KIVI_GROUP_SIZE)
        grouped.sub_(minimum.view(rows, groups, 1)).div_(
            scale.view(rows, groups, 1)
        )
        grouped.clamp_(0, (1 << bits) - 1).round_()
        integer.copy_(scratch)
        pack_grid = ((rows + block - 1) // block, packed_width)
        pack._pack_along_last_dim[pack_grid](
            bits,
            integer.view(-1, width),
            packed.view(-1, packed_width),
            rows,
            width,
            32 // bits,
            BLOCK_SIZE_N=block,
            num_warps=8,
        )
        destination.copy_(packed)

    def _store_historical_k(self, cache: KIVIStaticCache, layer: int, group: int) -> None:
        _, pack, _ = self._runtime()
        fp16, integers, packed = self._quantization_scratch(cache)
        source = cache.key_residual[layer].transpose(2, 3)
        target = cache.packed_key_history[layer, :, :, group * self.k_bits : (group + 1) * self.k_bits, :].transpose(-1, -2)
        scale = cache.key_scale_fp16_staging
        minimum = cache.key_minimum_fp16_staging
        self._quantize_into(
            source=source, scale=scale, minimum=minimum, destination=target,
            bits=self.k_bits, pack=pack, fp16_scratch=fp16, int_scratch=integers,
            packed_scratch=packed,
        )
        cache.key_scales[layer, :, :, group : group + 1, :].copy_(
            scale.transpose(-1, -2)
        )
        cache.key_minimums[layer, :, :, group : group + 1, :].copy_(
            minimum.transpose(-1, -2)
        )

    def _store_historical_v(self, cache: KIVIStaticCache, layer: int, position: int, value: Any) -> None:
        _, pack, _ = self._runtime()
        fp16, integers, packed = self._quantization_scratch(cache)
        source = value
        target = cache.packed_value_history[
            layer, :, :, :, position : position + 1
        ].transpose(-1, -2)
        scale = cache.value_scale_fp16_staging
        minimum = cache.value_minimum_fp16_staging
        self._quantize_into(
            source=source, scale=scale, minimum=minimum, destination=target,
            bits=self.v_bits, pack=pack, fp16_scratch=fp16, int_scratch=integers,
            packed_scratch=packed,
        )
        cache.value_scales[layer, :, :, :, position : position + 1].copy_(
            scale.transpose(-1, -2)
        )
        cache.value_minimums[layer, :, :, :, position : position + 1].copy_(
            minimum.transpose(-1, -2)
        )

    def _commit_token(self, cache: KIVIStaticCache, layer: int, key: Any, value: Any, token: int) -> None:
        key_count = cache._key_residual_counts[layer]
        value_count = cache._value_residual_counts[layer]
        value_head = cache._value_residual_heads[layer]
        cache.key_residual[layer, :, :, key_count : key_count + 1, :].copy_(key)
        if value_count == cache.residual_length:
            self._store_historical_v(
                cache,
                layer,
                cache._value_history_counts[layer],
                cache.value_residual_ring[
                    layer, :, :, value_head : value_head + 1, :
                ],
            )
            cache.value_residual_ring[
                layer, :, :, value_head : value_head + 1, :
            ].copy_(value)
        else:
            slot = (value_head + value_count) % cache.residual_length
            cache.value_residual_ring[layer, :, :, slot : slot + 1, :].copy_(value)
        if key_count + 1 == KIVI_RESIDUAL_LENGTH:
            self._store_historical_k(cache, layer, cache._key_history_counts[layer] // KIVI_GROUP_SIZE)
        cache.update(layer_idx=layer, token_index=token)

    @staticmethod
    def _layer_context(cache: KIVIStaticCache, layer: int) -> int:
        """Read only Python-side static ledgers; never read a CUDA scalar."""

        key_length = cache._key_history_counts[layer] + cache._key_residual_counts[layer]
        value_length = cache._value_history_counts[layer] + cache._value_residual_counts[layer]
        if key_length != value_length:
            raise CacheStateError("KIVI K/V layer ledgers have diverged")
        return key_length

    def store_prefill(self, cache_state: Any, key_states: Any, value_states: Any, layer_idx: int, cache_position: Any) -> tuple[Any, Any]:
        cache = self._require_cache(cache_state)
        if cache.mode != "prefill":
            raise CacheStateError("KIVI prefill store requires prefill mode")
        tokens = self._validate_update(cache, key_states, value_states, layer_idx, cache_position)
        layer_context = self._layer_context(cache, layer_idx)
        if layer_context + tokens > cache.capacity:
            raise CacheStateError("KIVI prefill exceeds static capacity")
        for offset in range(tokens):
            self._commit_token(
                cache, layer_idx, key_states[:, :, offset : offset + 1, :],
                value_states[:, :, offset : offset + 1, :], layer_context + offset,
            )
        handle = self._handle(cache, layer_idx)
        handle.prefill = True
        handle.prefill_key_states = key_states
        handle.prefill_value_states = value_states
        return handle, handle

    def append_decode(self, cache_state: Any, key_states: Any, value_states: Any, layer_idx: int, cache_position: Any) -> tuple[Any, Any]:
        cache = self._require_cache(cache_state)
        if cache.mode not in {"fixed", "growing_step"}:
            raise CacheStateError("KIVI append requires fixed or growing mode")
        if self._validate_update(cache, key_states, value_states, layer_idx, cache_position) != 1:
            raise CacheStateError("KIVI append requires exactly one token")
        cache.key_fp16_staging.copy_(key_states)
        cache.value_fp16_staging.copy_(value_states)
        handle = self._handle(cache, layer_idx)
        handle.pending_key = cache.key_fp16_staging
        handle.pending_value = cache.value_fp16_staging
        if cache.mode == "fixed":
            cache.fixed_scratch_overwrite(
                layer_idx=layer_idx,
                token_index=self._layer_context(cache, layer_idx),
            )
        else:
            handle.commit_after_decode = True
        return handle, handle

    def _decode_compressed(self, handle: KIVIAttentionHandle, query_states: Any, scaling: float) -> Any:
        cache = handle.cache
        if cache.device.type != "cuda":
            raise CacheStateError("KIVI compressed kernels require CUDA execution")
        if tuple(int(x) for x in query_states.shape) != (
            cache.batch_size,
            cache.num_query_heads,
            1,
            cache.head_dim,
        ):
            raise CacheStateError("KIVI decode query has unsupported geometry")
        if query_states.dtype != _torch().bfloat16 or query_states.device != cache.device:
            raise CacheStateError("KIVI decode query differs from BF16 cache device")
        _, _, launcher = self._runtime()
        cache.query_fp16_staging.copy_(query_states[:, :, 0, :])
        historical = cache._key_history_counts[handle.layer_idx]
        residual = cache._key_residual_counts[handle.layer_idx]
        total = historical + residual + 1
        if total > cache.capacity or handle.pending_key is None or handle.pending_value is None:
            raise CacheStateError("KIVI decode state is incomplete")

        logits = cache.decode_logits[:, :, :total]
        if historical:
            kernel_history = cache.key_history_capacity
            packed = cache.packed_key_history[handle.layer_idx]
            scales = cache.key_scales[handle.layer_idx]
            minimums = cache.key_minimums[handle.layer_idx]
            kernel_output = cache.key_kernel_output_fp16.view(-1)[
                : cache.batch_size * cache.num_query_heads * kernel_history
            ].view(
                cache.batch_size * cache.num_query_heads,
                1,
                kernel_history,
            )
            launcher.launch_into(
                input_tensor=cache.query_fp16_staging.view(
                    cache.batch_size * cache.num_query_heads,
                    1,
                    cache.head_dim,
                ),
                packed=packed.view(
                    cache.batch_size * cache.num_kv_heads,
                    -1,
                    cache.head_dim,
                ),
                scales=scales.view(
                    cache.batch_size * cache.num_kv_heads,
                    -1,
                    cache.head_dim,
                ),
                minimums=minimums.view(
                    cache.batch_size * cache.num_kv_heads,
                    -1,
                    cache.head_dim,
                ),
                output=kernel_output,
                bits=self.k_bits,
                group_size=KIVI_GROUP_SIZE,
                num_query_heads=32,
                num_kv_heads=8,
            )
            logits[:, :, :historical].copy_(
                kernel_output[:, :, :historical].view(
                    cache.batch_size,
                    cache.num_query_heads,
                    historical,
                )
            )
        for query_head in range(32):
            kv_head = query_head // KIVI_GQA_GROUP_SIZE
            if residual:
                _torch().bmm(
                    cache.query_fp16_staging[:, query_head : query_head + 1, :],
                    cache.key_residual[handle.layer_idx, :, kv_head, :residual, :].transpose(-1, -2),
                    out=logits[:, query_head : query_head + 1, historical : historical + residual],
                )
            _torch().bmm(
                cache.query_fp16_staging[:, query_head : query_head + 1, :],
                handle.pending_key[:, kv_head, :, :].transpose(-1, -2),
                out=logits[:, query_head : query_head + 1, historical + residual : total],
            )
        # The frozen reference applies the scale while scores are still FP16,
        # then requests an FP32 softmax accumulator.
        logits.mul_(float(scaling))
        # Keep the CUDA softmax geometry fixed at the declared capacity.
        # PyTorch's variable-width out= path selects context-dependent kernels
        # which may allocate an internal contiguous input and workspace for
        # some widths.  An inactive -inf tail is mathematically neutral and
        # lets every growing step reuse the same preallocated workspace.
        cache.decode_softmax.fill_(float("-inf"))
        cache.decode_softmax[:, :, :total].copy_(logits)
        _torch().softmax(
            cache.decode_softmax,
            dim=-1,
            out=cache.decode_softmax,
        )
        logits.copy_(cache.decode_softmax[:, :, :total])

        value_history = cache._value_history_counts[handle.layer_idx]
        cache.decode_output_fp16.zero_()
        if value_history:
            kernel_history = cache.value_history_capacity
            value_weights = cache.key_kernel_output_fp16.view(-1)[
                : cache.batch_size * cache.num_query_heads * kernel_history
            ].view(
                cache.batch_size * cache.num_query_heads,
                1,
                kernel_history,
            )
            value_weights.zero_()
            value_weights[:, :, :value_history].copy_(
                logits[:, :, :value_history].view(
                    cache.batch_size * cache.num_query_heads,
                    1,
                    value_history,
                )
            )
            launcher.launch_into(
                input_tensor=value_weights,
                packed=cache.packed_value_history[
                    handle.layer_idx
                ].view(
                    cache.batch_size * cache.num_kv_heads,
                    -1,
                    kernel_history,
                ),
                scales=cache.value_scales[
                    handle.layer_idx
                ].view(
                    cache.batch_size * cache.num_kv_heads,
                    -1,
                    kernel_history,
                ),
                minimums=cache.value_minimums[
                    handle.layer_idx
                ].view(
                    cache.batch_size * cache.num_kv_heads,
                    -1,
                    kernel_history,
                ),
                output=cache.decode_output_fp16.view(
                    cache.batch_size * cache.num_query_heads,
                    1,
                    cache.head_dim,
                ),
                bits=self.v_bits,
                group_size=KIVI_GROUP_SIZE,
                num_query_heads=32,
                num_kv_heads=8,
            )
        value_residual = cache._value_residual_counts[handle.layer_idx]
        residual_values = cache.value_residual_ring[handle.layer_idx]
        if value_residual:
            head = cache._value_residual_heads[handle.layer_idx]
            ordered = cache.value_residual_ordered_staging[handle.layer_idx]
            first = min(value_residual, cache.residual_length - head)
            ordered[:, :, :first, :].copy_(
                residual_values[:, :, head : head + first, :]
            )
            if value_residual > first:
                ordered[:, :, first:value_residual, :].copy_(
                    residual_values[:, :, : value_residual - first, :]
                )
            residual_values = ordered
        for query_head in range(32):
            kv_head = query_head // KIVI_GQA_GROUP_SIZE
            if value_residual:
                _torch().bmm(
                    logits[:, query_head : query_head + 1, value_history : value_history + value_residual],
                    residual_values[:, kv_head, :value_residual, :],
                    out=cache.decode_merge[:, query_head : query_head + 1, :],
                )
                cache.decode_output_fp16[:, query_head : query_head + 1, :].add_(cache.decode_merge[:, query_head : query_head + 1, :])
            _torch().bmm(
                logits[:, query_head : query_head + 1, value_history + value_residual : total],
                handle.pending_value[:, kv_head, :, :],
                out=cache.decode_merge[:, query_head : query_head + 1, :],
            )
            cache.decode_output_fp16[:, query_head : query_head + 1, :].add_(cache.decode_merge[:, query_head : query_head + 1, :])
        cache.output_buffer.copy_(cache.decode_output_fp16)
        if handle.commit_after_decode:
            self._commit_token(
                cache,
                handle.layer_idx,
                handle.pending_key,
                handle.pending_value,
                self._layer_context(cache, handle.layer_idx),
            )
            handle.commit_after_decode = False
        return cache.output_buffer.unsqueeze(2)

    def decode_attention(self, attention: Any, query_states: Any, key_states: Any, value_states: Any, *, scaling: float) -> Any:
        if not isinstance(key_states, KIVIAttentionHandle) or key_states is not value_states:
            raise CacheStateError("KIVI attended handles must be identical")
        handle = key_states
        if int(attention.layer_idx) != handle.layer_idx:
            raise CacheStateError("KIVI attended handle identity mismatch")
        if handle.prefill:
            if handle.prefill_key_states is None or handle.prefill_value_states is None:
                raise CacheStateError("KIVI prefill handle lost raw K/V")
            try:
                output, _ = flash_attention_forward(attention, query_states, handle.prefill_key_states, handle.prefill_value_states, None, scaling, dropout=0.0)
                return output
            finally:
                handle.prefill_key_states = None
                handle.prefill_value_states = None
                handle.prefill = False
        return self._decode_compressed(handle, query_states, scaling)

    def allocated_bytes(self, cache_state: Any) -> int:
        return self._require_cache(cache_state).accounting().allocated_bytes

    def byte_breakdown(self, cache_state: Any) -> Mapping[str, int]:
        return self._require_cache(cache_state).byte_breakdown()

    def logical_bf16_bytes(self, cache_state: Any) -> int:
        return self._require_cache(cache_state).logical_bf16_storage_bytes

    def config_fingerprint(self, cache_layout_fingerprint: str) -> str:
        require_sha256(cache_layout_fingerprint, field_name="cache_layout_fingerprint")
        context = self.runtime_context
        payload = {
            "schema_version": KIVI_ADAPTER_FINGERPRINT_SCHEMA_VERSION,
            "adapter_version": self.adapter_version,
            "method_name": self.name,
            "configuration": self.config_name,
            "key_bits": self.k_bits,
            "value_bits": self.v_bits,
            "group_size": KIVI_GROUP_SIZE,
            "residual_length": KIVI_RESIDUAL_LENGTH,
            "cuda_abi": "float16",
            "model_boundary": "bfloat16_to_float16_to_bfloat16",
            "gqa_mapping": "query_head // 4",
            "cache_layout_fingerprint": cache_layout_fingerprint,
            "model_id": context.model_id,
            "model_revision": context.model_revision,
            "backend_id": context.backend_id,
            "backend_fingerprint": context.backend_fingerprint,
            "num_layers": context.num_layers,
            "num_query_heads": context.num_query_heads,
            "num_kv_heads": context.num_kv_heads,
            "head_dim": context.head_dim,
            "official_commit": KIVI_OFFICIAL_COMMIT,
            "official_base_tree": KIVI_OFFICIAL_BASE_TREE,
            "patched_tree": KIVI_PATCHED_TREE,
            "decision_0018_patch_sha256": KIVI_DECISION_0018_PATCH_SHA256,
            "extension_sha256": KIVI_EXTENSION_SHA256,
            "bgemv2_host_stub_offset": KIVI_BGEMV2_HOST_STUB_OFFSET,
            "bgemv4_host_stub_offset": KIVI_BGEMV4_HOST_STUB_OFFSET,
            "fixture_root_sha256": KIVI_FIXTURE_ROOT_SHA256,
            "supports_cuda_graph": self.supports_cuda_graph(),
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        return sha256_hex(canonical_json_bytes(payload))

    def supports_cuda_graph(self) -> bool:
        return True
