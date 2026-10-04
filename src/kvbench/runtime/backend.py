"""Fail-closed direct Flash SDPA backend for the Phase 3 BF16 baseline."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
from pathlib import Path
from typing import Any, Iterator


ATTENTION_IMPLEMENTATION = "kvbench_bf16_flash"
_FLASH_CONTEXT_ACTIVE: ContextVar[bool] = ContextVar(
    "kvbench_flash_context_active",
    default=False,
)
_TORCH: Any | None = None
_BACKEND_IDENTITY_VERIFIED = False

BACKEND_IDENTITY = {
    "schema_version": "kvbench.phase3-bf16-backend.v1",
    "backend_id": "torch_sdpa_flash_gqa",
    "torch_version": "2.12.1+cu130",
    "torch_git_sha": "7269437d655783a26cba32aa88195b741ff496aa",
    "cuda_runtime_version": "13.0",
    "cudnn_version": "9.20.0",
    "triton_version": "3.7.1",
    "flash_generation": "FA2",
    "flash_version": "2.5.7",
    "dispatch_api": "torch.nn.functional.scaled_dot_product_attention",
    "selected_backend": "flash_attention",
    "enable_gqa": True,
    "compile_mode": "disabled",
}
_EXPECTED_SOURCE_DIGESTS = {
    "include/ATen/native/transformers/cuda/flash_attn/flash_api.h": (
        "1474aa79d8aa6ce39984dbc3c0aad9dba283ab819f034370e5cfb70980524ee7"
    ),
    "lib/libtorch_cuda.so": (
        "b248fb7e9935440965e4736eea48868b315ba41012734b7ce058fc0a2d0b1984"
    ),
    "nn/attention/__init__.py": (
        "56e10b6f965cc050db782dd4dc472097c9b02ec5b5fe3ab2c8b04055c0b0bbe0"
    ),
    "nn/attention/varlen.py": (
        "2f5384e0bc8ce371d00a1c09d38ad019517009798e7cb3434f56cf4b9fa351ea"
    ),
    "nn/functional.py": (
        "27493186ee22f811b553e31d9c804d4d46716d1be62d034d731537f66f27ef19"
    ),
}


class BackendUnsupportedError(RuntimeError):
    """The frozen Flash backend cannot execute the requested geometry."""


class BackendFallbackError(RuntimeError):
    """Execution escaped the sole allowed Flash backend."""


@dataclass(frozen=True, slots=True)
class BackendAudit:
    """Untimed evidence for one exact direct-SDPA dispatch decision."""

    backend_value: int
    backend_name: str
    query_shape: tuple[int, ...]
    key_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
    dtype: str
    device: str
    is_causal: bool
    enable_gqa: bool
    scale: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_value": self.backend_value,
            "backend_name": self.backend_name,
            "query_shape": list(self.query_shape),
            "key_shape": list(self.key_shape),
            "value_shape": list(self.value_shape),
            "dtype": self.dtype,
            "device": self.device,
            "is_causal": self.is_causal,
            "enable_gqa": self.enable_gqa,
            "scale": self.scale,
        }


def _torch() -> Any:
    global _TORCH
    if _TORCH is None:
        try:
            _TORCH = importlib.import_module("torch")
        except ModuleNotFoundError as error:
            raise BackendUnsupportedError(
                "the frozen Phase 3 PyTorch runtime is unavailable"
            ) from error
    return _TORCH


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cudnn_version_text(raw_version: int) -> str:
    if raw_version <= 0:
        raise BackendFallbackError("cuDNN reported an invalid version scalar")
    major = raw_version // 10000
    minor = (raw_version % 10000) // 100
    patch = raw_version % 100
    return f"{major}.{minor}.{patch}"


def _verify_backend_identity() -> None:
    global _BACKEND_IDENTITY_VERIFIED
    if _BACKEND_IDENTITY_VERIFIED:
        return
    torch = _torch()
    observed_scalars = {
        "torch_version": str(torch.__version__),
        "torch_git_sha": str(torch.version.git_version),
        "cuda_runtime_version": str(torch.version.cuda),
        "cudnn_version": _cudnn_version_text(int(torch.backends.cudnn.version())),
        "triton_version": importlib.metadata.version("triton"),
    }
    for field, observed in observed_scalars.items():
        if observed != BACKEND_IDENTITY[field]:
            raise BackendFallbackError(
                f"installed backend scalar differs from Decision 0007: {field}"
            )
    if not callable(torch.nn.functional.scaled_dot_product_attention):
        raise BackendFallbackError("the frozen SDPA dispatch API is unavailable")
    if not hasattr(torch.nn.attention.SDPBackend, "FLASH_ATTENTION"):
        raise BackendFallbackError("the frozen Flash backend enum is unavailable")
    torch_root = Path(torch.__file__).resolve().parent
    for relative, expected in sorted(_EXPECTED_SOURCE_DIGESTS.items()):
        path = torch_root / relative
        if not path.is_file():
            raise BackendFallbackError(
                f"frozen backend source artifact is absent: {relative}"
            )
        if _sha256_file(path) != expected:
            raise BackendFallbackError(
                f"frozen backend source hash differs: {relative}"
            )
    _BACKEND_IDENTITY_VERIFIED = True


def backend_identity() -> dict[str, Any]:
    """Verify and return the strict Decision-0007 backend manifest payload."""

    _verify_backend_identity()
    payload = dict(BACKEND_IDENTITY)
    payload["source_artifacts"] = [
        {"path": path, "sha256": digest}
        for path, digest in sorted(_EXPECTED_SOURCE_DIGESTS.items())
    ]
    return payload


@contextmanager
def forced_flash_execution() -> Iterator[None]:
    """Enable only Flash SDPA for a complete untimed or measured lane."""

    torch = _torch()
    token = _FLASH_CONTEXT_ACTIVE.set(True)
    try:
        with torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.FLASH_ATTENTION
        ):
            yield
    finally:
        _FLASH_CONTEXT_ACTIVE.reset(token)


def _validate_geometry(query: Any, key: Any, value: Any) -> None:
    torch = _torch()
    if not _FLASH_CONTEXT_ACTIVE.get():
        raise BackendFallbackError(
            "direct Flash attention was called outside the forced backend context"
        )
    if query.device.type != "cuda":
        raise BackendUnsupportedError("the Phase 3 SUT requires a CUDA device")
    if query.dtype != torch.bfloat16:
        raise BackendUnsupportedError("the Phase 3 SUT requires BF16 Q/K/V")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise BackendUnsupportedError("Q, K, and V dtypes must match exactly")
    if key.device != query.device or value.device != query.device:
        raise BackendUnsupportedError("Q, K, and V devices must match exactly")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise BackendUnsupportedError("Q, K, and V must be rank-four tensors")
    if tuple(key.shape) != tuple(value.shape):
        raise BackendUnsupportedError("K and V shapes must match exactly")
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise BackendUnsupportedError("Q and K batch/head-dimension geometry differs")
    if query.shape[1] <= key.shape[1]:
        raise BackendUnsupportedError("the selected baseline requires GQA geometry")
    if query.shape[1] % key.shape[1] != 0:
        raise BackendUnsupportedError("query heads must be divisible by KV heads")


def flash_attention_forward(
    module: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any | None,
    scaling: float,
    dropout: float = 0.0,
    **_: Any,
) -> tuple[Any, None]:
    """Transformers attention-interface entry for the frozen direct Flash call."""

    torch = _torch()
    _validate_geometry(query, key, value)
    if attention_mask is not None:
        raise BackendUnsupportedError("the frozen direct Flash path accepts no mask")
    if dropout != 0.0 or getattr(module, "training", False):
        raise BackendUnsupportedError("dropout and training mode are forbidden")
    query_length = int(query.shape[-2])
    key_length = int(key.shape[-2])
    is_causal = query_length > 1
    if is_causal and query_length != key_length:
        raise BackendUnsupportedError(
            "prefill requires equal query and attended-key lengths"
        )
    if not is_causal and query_length != 1:
        raise BackendUnsupportedError("decode requires exactly one query token")
    try:
        output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=float(scaling),
            enable_gqa=True,
        )
    except RuntimeError as error:
        raise BackendUnsupportedError(
            "the forced Flash SDPA operator rejected the requested geometry"
        ) from error
    return output.transpose(1, 2), None


def register_transformers_attention() -> None:
    """Register the unique Phase 3 attention key without overwriting another key."""

    try:
        modeling_utils = importlib.import_module("transformers.modeling_utils")
    except ModuleNotFoundError as error:
        raise BackendUnsupportedError(
            "Transformers 4.57.6 is unavailable in the Phase 3 dependency target"
        ) from error
    interface = modeling_utils.ALL_ATTENTION_FUNCTIONS
    if ATTENTION_IMPLEMENTATION in interface.valid_keys():
        if interface[ATTENTION_IMPLEMENTATION] is not flash_attention_forward:
            raise BackendFallbackError(
                "the Phase 3 attention registry key was already replaced"
            )
        return
    interface.register(ATTENTION_IMPLEMENTATION, flash_attention_forward)
    if interface[ATTENTION_IMPLEMENTATION] is not flash_attention_forward:
        raise BackendFallbackError("custom attention registration did not persist")


def audit_backend_choice(
    query: Any,
    key: Any,
    value: Any,
    *,
    is_causal: bool,
    scale: float,
) -> BackendAudit:
    """Probe the fused choice outside timing and require the exact Flash enum."""

    torch = _torch()
    with forced_flash_execution():
        _validate_geometry(query, key, value)
        try:
            choice = int(
                torch._fused_sdp_choice(
                    query,
                    key,
                    value,
                    None,
                    0.0,
                    is_causal,
                    scale=float(scale),
                    enable_gqa=True,
                )
            )
        except RuntimeError as error:
            raise BackendUnsupportedError(
                "the fused-SDPA choice probe rejected the geometry"
            ) from error
    expected = int(torch.nn.attention.SDPBackend.FLASH_ATTENTION.value)
    if choice != expected:
        raise BackendFallbackError(
            f"fused SDPA chose backend enum {choice}, expected {expected}"
        )
    return BackendAudit(
        backend_value=choice,
        backend_name="FLASH_ATTENTION",
        query_shape=tuple(int(item) for item in query.shape),
        key_shape=tuple(int(item) for item in key.shape),
        value_shape=tuple(int(item) for item in value.shape),
        dtype=str(query.dtype),
        device=str(query.device),
        is_causal=is_causal,
        enable_gqa=True,
        scale=float(scale),
    )
