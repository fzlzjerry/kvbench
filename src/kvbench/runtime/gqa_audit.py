"""Untimed GQA source, dispatch, tensor-shape, and storage audit helpers."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any
import warnings

from kvbench.runtime.backend import audit_backend_choice, forced_flash_execution
from kvbench.runtime.static_cache import BF16StaticCache


FORBIDDEN_SOURCE_PATTERNS = (
    "repeat_kv",
    "repeat_interleave",
    "expand(",
    "torch.cat",
    "DynamicCache",
)
FORBIDDEN_OPERATOR_FRAGMENTS = (
    "repeat_interleave",
    "repeat",
    "clone",
)


@dataclass(frozen=True, slots=True)
class SourceAudit:
    """Static scan of the selected SUT implementation files only."""

    passed: bool
    findings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "findings": list(self.findings)}


@dataclass(frozen=True, slots=True)
class OperatorAudit:
    """Direct attention operator trace collected outside timing."""

    passed: bool
    operations: tuple[str, ...]
    output_shapes: tuple[tuple[int, ...], ...]
    warnings: tuple[str, ...]
    backend: dict[str, Any]
    query_head_sized_kv_temporary: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "operations": list(self.operations),
            "output_shapes": [list(shape) for shape in self.output_shapes],
            "warnings": list(self.warnings),
            "backend": self.backend,
            "query_head_sized_kv_temporary": (
                self.query_head_sized_kv_temporary
            ),
        }


def audit_source_paths(paths: tuple[str | Path, ...]) -> SourceAudit:
    findings: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            findings.append(f"unreadable:{path}")
            continue
        for pattern in FORBIDDEN_SOURCE_PATTERNS:
            if pattern in source:
                findings.append(f"{path}:{pattern}")
    return SourceAudit(passed=not findings, findings=tuple(findings))


def audit_cache_geometry(
    cache: BF16StaticCache,
    *,
    num_query_heads: int,
) -> dict[str, Any]:
    forbidden_bytes = (
        cache.predicted_tensor_bytes
        * num_query_heads
        // cache.num_kv_heads
    )
    return {
        "cache_shape": list(cache.keys.shape),
        "num_query_heads": num_query_heads,
        "num_kv_heads": cache.num_kv_heads,
        "uses_kv_head_geometry": cache.keys.shape[2] == cache.num_kv_heads,
        "measured_storage_bytes": cache.tensor_storage_bytes,
        "predicted_kv_head_bytes": cache.predicted_tensor_bytes,
        "forbidden_query_head_bytes": forbidden_bytes,
        "query_head_storage_detected": (
            cache.tensor_storage_bytes == forbidden_bytes
        ),
    }


def _tensor_shapes(value: Any, torch: Any) -> list[tuple[int, ...]]:
    if isinstance(value, torch.Tensor):
        return [tuple(int(item) for item in value.shape)]
    if isinstance(value, (tuple, list)):
        shapes: list[tuple[int, ...]] = []
        for item in value:
            shapes.extend(_tensor_shapes(item, torch))
        return shapes
    return []


def audit_gqa_operator(
    query: Any,
    key: Any,
    value: Any,
    *,
    is_causal: bool,
    scale: float,
) -> OperatorAudit:
    """Trace one direct forced-Flash call; its duration is never timing data."""

    torch = importlib.import_module("torch")
    dispatch_module = importlib.import_module("torch.utils._python_dispatch")
    operation_names: list[str] = []
    output_shapes: list[tuple[int, ...]] = []

    class Recorder(dispatch_module.TorchDispatchMode):
        def __torch_dispatch__(
            self,
            function: Any,
            types: Any,
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any:
            del types
            actual_kwargs = {} if kwargs is None else kwargs
            output = function(*args, **actual_kwargs)
            operation_names.append(str(function))
            output_shapes.extend(_tensor_shapes(output, torch))
            return output

    backend = audit_backend_choice(
        query,
        key,
        value,
        is_causal=is_causal,
        scale=scale,
    )
    captured_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        with forced_flash_execution(), Recorder():
            torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=is_causal,
                scale=float(scale),
                enable_gqa=True,
            )
        captured_warnings.extend(str(item.message) for item in warning_records)
    forbidden_operation = any(
        fragment in name
        for name in operation_names
        for fragment in FORBIDDEN_OPERATOR_FRAGMENTS
    )
    query_head_temp = False
    if int(key.shape[-2]) != int(query.shape[-2]):
        forbidden_shape = (
            int(key.shape[0]),
            int(query.shape[1]),
            int(key.shape[-2]),
            int(key.shape[-1]),
        )
        query_head_temp = forbidden_shape in output_shapes
    fused_flash = any("scaled_dot_product_flash_attention" in name for name in operation_names)
    passed = (
        fused_flash
        and not forbidden_operation
        and not query_head_temp
        and not captured_warnings
    )
    return OperatorAudit(
        passed=passed,
        operations=tuple(operation_names),
        output_shapes=tuple(output_shapes),
        warnings=tuple(captured_warnings),
        backend=backend.to_dict(),
        query_head_sized_kv_temporary=query_head_temp,
    )


def audit_mha_operator_control(
    query: Any,
    key: Any,
    value: Any,
    *,
    is_causal: bool,
    scale: float,
) -> OperatorAudit:
    """Trace a same-head MHA control without relaxing the GQA SUT contract."""

    torch = importlib.import_module("torch")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("MHA control Q/K/V must be rank four")
    if tuple(key.shape) != tuple(value.shape):
        raise ValueError("MHA control K/V shapes differ")
    if int(query.shape[1]) != int(key.shape[1]):
        raise ValueError("MHA control requires equal query and KV head counts")
    if query.dtype != torch.bfloat16 or key.dtype != query.dtype:
        raise ValueError("MHA control requires BF16 Q/K/V")
    if value.dtype != query.dtype or key.device != query.device:
        raise ValueError("MHA control Q/K/V dtype or device differs")
    if value.device != query.device or query.device.type != "cuda":
        raise ValueError("MHA control requires one CUDA device")

    with forced_flash_execution():
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
            raise RuntimeError("forced-Flash MHA control was rejected") from error
    expected = int(torch.nn.attention.SDPBackend.FLASH_ATTENTION.value)
    if choice != expected:
        raise RuntimeError("MHA control did not dispatch to forced Flash")
    backend = {
        "backend_value": choice,
        "backend_name": "FLASH_ATTENTION",
        "query_shape": list(query.shape),
        "key_shape": list(key.shape),
        "value_shape": list(value.shape),
        "dtype": str(query.dtype),
        "device": str(query.device),
        "is_causal": is_causal,
        "enable_gqa": True,
        "scale": float(scale),
        "control_only": True,
        "system_under_test": False,
    }
    dispatch_module = importlib.import_module("torch.utils._python_dispatch")
    operation_names: list[str] = []
    output_shapes: list[tuple[int, ...]] = []

    class Recorder(dispatch_module.TorchDispatchMode):
        def __torch_dispatch__(
            self,
            function: Any,
            types: Any,
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any:
            del types
            actual_kwargs = {} if kwargs is None else kwargs
            output = function(*args, **actual_kwargs)
            operation_names.append(str(function))
            output_shapes.extend(_tensor_shapes(output, torch))
            return output

    captured_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        with forced_flash_execution(), Recorder():
            torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=is_causal,
                scale=float(scale),
                enable_gqa=True,
            )
        captured_warnings.extend(str(item.message) for item in warning_records)
    forbidden_operation = any(
        fragment in name
        for name in operation_names
        for fragment in FORBIDDEN_OPERATOR_FRAGMENTS
    )
    fused_flash = any(
        "scaled_dot_product_flash_attention" in name for name in operation_names
    )
    passed = fused_flash and not forbidden_operation and not captured_warnings
    return OperatorAudit(
        passed=passed,
        operations=tuple(operation_names),
        output_shapes=tuple(output_shapes),
        warnings=tuple(captured_warnings),
        backend=backend,
        query_head_sized_kv_temporary=False,
    )
