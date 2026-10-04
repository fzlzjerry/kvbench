"""Independent small reference and explicitly untimed checksum helpers."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import importlib
import json
from typing import Any, Protocol, runtime_checkable

from kvbench.adapters import KVCacheMethod


_TORCH: Any | None = None
FULL_MODEL_ATOL = 0.125
FULL_MODEL_RTOL = 0.02


@dataclass(frozen=True, slots=True)
class NumericalComparison:
    """Finite/tolerance evidence for one untimed tensor comparison."""

    passed: bool
    finite: bool
    max_absolute_error: float
    max_relative_error: float
    atol: float
    rtol: float

    def to_dict(self) -> dict[str, bool | float]:
        return {
            "passed": self.passed,
            "finite": self.finite,
            "max_absolute_error": self.max_absolute_error,
            "max_relative_error": self.max_relative_error,
            "atol": self.atol,
            "rtol": self.rtol,
        }


@dataclass(frozen=True, slots=True)
class FullModelStepEvidence:
    """Untimed full-vocabulary comparison for one declared decode step."""

    mode: str
    step: int
    position: int
    reference_checksum: str
    observed_checksum: str
    comparison: NumericalComparison

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "step": self.step,
            "position": self.position,
            "reference_checksum": self.reference_checksum,
            "observed_checksum": self.observed_checksum,
            "comparison": self.comparison.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class FullModelReferenceResult:
    """Trusted eager/DynamicCache evidence; this object contains no timing."""

    passed: bool
    reference_implementation: str
    reference_cache_type: str
    reference_implementation_restored: bool
    tolerance_atol: float
    tolerance_rtol: float
    fixed_repeat_exact: bool
    fixed_historical_cache_unchanged: bool
    fixed_steps: tuple[FullModelStepEvidence, ...]
    growing_steps: tuple[FullModelStepEvidence, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reference_implementation": self.reference_implementation,
            "reference_cache_type": self.reference_cache_type,
            "reference_implementation_restored": (
                self.reference_implementation_restored
            ),
            "tolerance_atol": self.tolerance_atol,
            "tolerance_rtol": self.tolerance_rtol,
            "fixed_repeat_exact": self.fixed_repeat_exact,
            "fixed_historical_cache_unchanged": (
                self.fixed_historical_cache_unchanged
            ),
            "fixed_steps": [item.to_dict() for item in self.fixed_steps],
            "growing_steps": [item.to_dict() for item in self.growing_steps],
            "timing_collected": False,
            "performance_claim_eligible": False,
        }


def _torch() -> Any:
    global _TORCH
    if _TORCH is None:
        try:
            _TORCH = importlib.import_module("torch")
        except ModuleNotFoundError as error:
            raise RuntimeError("PyTorch is unavailable") from error
    return _TORCH


def small_attention_reference(
    query: Any,
    key: Any,
    value: Any,
    *,
    is_causal: bool,
    scale: float,
) -> Any:
    """Independent FP32 GQA reference intended only for small tensors."""

    torch = _torch()
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("reference Q/K/V must be rank four")
    if tuple(key.shape) != tuple(value.shape):
        raise ValueError("reference K/V shapes differ")
    query_heads = int(query.shape[1])
    kv_heads = int(key.shape[1])
    if query_heads % kv_heads != 0:
        raise ValueError("reference query heads are not divisible by KV heads")
    head_map = torch.arange(query_heads, device=query.device) // (
        query_heads // kv_heads
    )
    selected_key = torch.index_select(key.float(), 1, head_map)
    selected_value = torch.index_select(value.float(), 1, head_map)
    scores = torch.matmul(query.float(), selected_key.transpose(-2, -1))
    scores.mul_(float(scale))
    if is_causal:
        query_length = int(query.shape[-2])
        key_length = int(key.shape[-2])
        if query_length != key_length:
            raise ValueError("causal reference requires square prefill geometry")
        row = torch.arange(query_length, device=query.device).unsqueeze(1)
        column = torch.arange(key_length, device=query.device).unsqueeze(0)
        scores.masked_fill_(column > row, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
    return torch.matmul(probabilities, selected_value)


def compare_tensors_untimed(
    observed: Any,
    reference: Any,
    *,
    atol: float,
    rtol: float,
) -> NumericalComparison:
    """Synchronizing comparison helper forbidden from measured lanes."""

    torch = _torch()
    if tuple(observed.shape) != tuple(reference.shape):
        raise ValueError("comparison tensor shapes differ")
    observed_float = observed.float()
    reference_float = reference.float()
    finite = bool(
        torch.isfinite(observed_float).all().item()
        and torch.isfinite(reference_float).all().item()
    )
    difference = (observed_float - reference_float).abs()
    denominator = reference_float.abs().clamp_min(1.0e-12)
    max_absolute = float(difference.max().item())
    max_relative = float((difference / denominator).max().item())
    passed = finite and bool(
        torch.allclose(observed_float, reference_float, atol=atol, rtol=rtol)
    )
    return NumericalComparison(
        passed=passed,
        finite=finite,
        max_absolute_error=max_absolute,
        max_relative_error=max_relative,
        atol=float(atol),
        rtol=float(rtol),
    )


def tensor_sha256_untimed(tensor: Any) -> str:
    """Hash canonical raw tensor bytes after leaving measured execution."""

    torch = _torch()
    contiguous = (
        tensor.detach()
        .contiguous()
        .view(torch.uint8)
        .to(device="cpu", copy=True)
    )
    byte_count = int(contiguous.numel())
    header = json.dumps(
        {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    if byte_count:
        # ``bytes(UntypedStorage)`` iterates one Python integer per byte.  That
        # is exact but makes untimed custody checks over long-context caches
        # take hours.  The tensor above is a fresh contiguous CPU uint8 copy,
        # so a read-only buffer over its declared byte range has identical raw
        # byte semantics without materializing or iterating a Python sequence.
        raw_array = (ctypes.c_ubyte * byte_count).from_address(
            int(contiguous.data_ptr())
        )
        raw_view = memoryview(raw_array).cast("B")
        if len(raw_view) != byte_count:
            raise RuntimeError("untimed tensor checksum storage is incomplete")
        digest.update(raw_view)
    return digest.hexdigest()


@runtime_checkable
class CacheHistoryCapability(Protocol):
    """Cache state that can hash its own exact untimed historical prefix."""

    def history_sha256(self, historical_length: int) -> str: ...


def cache_history_sha256_untimed(
    keys: Any,
    values: Any,
    *,
    historical_length: int,
) -> str:
    """Stream exact historical K/V bytes to a digest outside timing."""

    torch = _torch()
    if historical_length <= 0 or historical_length > int(keys.shape[-2]):
        raise ValueError("historical_length is outside cache storage")
    if tuple(keys.shape) != tuple(values.shape):
        raise ValueError("cache K/V shapes differ")
    digest = hashlib.sha256()
    digest.update(str(tuple(keys.shape)).encode("ascii"))
    digest.update(str(keys.dtype).encode("ascii"))
    for label, tensor in ((b"K", keys), (b"V", values)):
        digest.update(label)
        for layer in range(int(tensor.shape[0])):
            for batch in range(int(tensor.shape[1])):
                chunk = (
                    tensor[layer, batch, :, :historical_length, :]
                    .detach()
                    .contiguous()
                    .view(torch.uint8)
                    .cpu()
                )
                digest.update(chunk.numpy().tobytes(order="C"))
    return digest.hexdigest()


def cache_state_history_sha256_untimed(
    cache_state: Any,
    *,
    historical_length: int,
) -> str:
    """Dispatch untimed history hashing without assuming a BF16 layout."""

    if (
        isinstance(historical_length, bool)
        or not isinstance(historical_length, int)
        or historical_length <= 0
    ):
        raise ValueError("historical_length must be a positive integer")
    if isinstance(cache_state, CacheHistoryCapability):
        digest = cache_state.history_sha256(historical_length)
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("cache history capability returned an invalid digest")
        return digest
    try:
        keys = cache_state.keys
        values = cache_state.values
    except AttributeError as error:
        raise TypeError("cache state has no exact history capability") from error
    return cache_history_sha256_untimed(
        keys,
        values,
        historical_length=historical_length,
    )


def validate_full_model_reference(
    model: Any,
    prefix_input_ids: Any,
    decode_input_ids: Any,
    *,
    method: KVCacheMethod | None = None,
    atol: float = FULL_MODEL_ATOL,
    rtol: float = FULL_MODEL_RTOL,
) -> FullModelReferenceResult:
    """Compare the manual static endpoint with trusted eager DynamicCache.

    This correctness helper is intentionally synchronizing and untimed.  The
    trusted lane may materialize GQA through Transformers' eager reference;
    that implementation is never the system-under-test or a timing endpoint.
    """

    torch = _torch()
    if prefix_input_ids.ndim != 2 or decode_input_ids.ndim != 2:
        raise ValueError("full-model reference inputs must be rank two")
    batch = int(prefix_input_ids.shape[0])
    prefix_length = int(prefix_input_ids.shape[1])
    steps = int(decode_input_ids.shape[1])
    if batch <= 0 or prefix_length <= 0:
        raise ValueError("full-model reference prefix must be nonempty")
    if int(decode_input_ids.shape[0]) != batch or steps < 3:
        raise ValueError("full-model reference requires at least three decode steps")
    if prefix_input_ids.device != decode_input_ids.device:
        raise ValueError("full-model reference input devices differ")

    original_implementation = model.config._attn_implementation
    reference_outputs: list[Any] = []
    reference_cache_type = ""
    restoration_error: BaseException | None = None
    try:
        model.set_attn_implementation("eager")
        with torch.inference_mode():
            prefix_positions = torch.arange(
                prefix_length,
                dtype=torch.long,
                device=prefix_input_ids.device,
            )
            reference = model(
                input_ids=prefix_input_ids,
                position_ids=prefix_positions.unsqueeze(0),
                cache_position=prefix_positions,
                use_cache=True,
                logits_to_keep=1,
            )
            dynamic_cache = reference.past_key_values
            reference_cache_type = type(dynamic_cache).__name__
            if reference_cache_type != "DynamicCache":
                raise RuntimeError("trusted reference did not return DynamicCache")
            for step in range(steps):
                position = torch.tensor(
                    [prefix_length + step],
                    dtype=torch.long,
                    device=prefix_input_ids.device,
                )
                reference = model(
                    input_ids=decode_input_ids[:, step : step + 1],
                    position_ids=position.unsqueeze(0),
                    cache_position=position,
                    past_key_values=dynamic_cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                dynamic_cache = reference.past_key_values
                reference_outputs.append(reference.logits.detach().cpu().clone())
    finally:
        try:
            model.set_attn_implementation(original_implementation)
        except BaseException as error:
            restoration_error = error
    if restoration_error is not None:
        raise RuntimeError("failed to restore model attention implementation") from (
            restoration_error
        )
    restored = model.config._attn_implementation == original_implementation
    if not restored:
        raise RuntimeError("model attention implementation was not restored")

    from kvbench.adapters import (
        build_method_adapter,
        declared_bf16_runtime_context,
    )
    from kvbench.runtime.backend import forced_flash_execution
    from kvbench.runtime.bf16_endpoint import BF16DecodeEndpoint

    config = model.config
    num_layers = int(config.num_hidden_layers)
    query_heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(getattr(config, "head_dim", config.hidden_size // query_heads))
    workspace_bytes = (
        num_layers * batch * (query_heads + kv_heads) * (head_dim // 2) * 2
    )

    adapter = method
    if adapter is None:
        adapter = build_method_adapter(
            "bf16",
            declared_bf16_runtime_context(model),
        )
    fixed_cache = adapter.allocate(
        batch_size=batch,
        capacity=prefix_length + 1,
        device=prefix_input_ids.device,
        workspace_bytes=workspace_bytes,
    )
    fixed_endpoint = BF16DecodeEndpoint(model, fixed_cache, adapter)
    fixed_outputs: list[Any] = []
    with torch.inference_mode(), forced_flash_execution():
        fixed_endpoint.prefill(prefix_input_ids)
        fixed_cache.prepare_fixed(prefix_length)
        fixed_position = torch.tensor(
            [prefix_length],
            dtype=torch.long,
            device=prefix_input_ids.device,
        )
        fixed_embeddings = fixed_endpoint.prepare_position_embeddings(
            fixed_position.unsqueeze(0)
        )
        historical_before = cache_state_history_sha256_untimed(
            fixed_cache,
            historical_length=prefix_length,
        )
        for _ in range(steps):
            fixed_outputs.append(
                fixed_endpoint.decode(
                    decode_input_ids[:, :1],
                    fixed_position,
                    fixed_embeddings,
                )
                .detach()
                .cpu()
                .clone()
            )
        historical_after = cache_state_history_sha256_untimed(
            fixed_cache,
            historical_length=prefix_length,
        )

    growing_cache = adapter.allocate(
        batch_size=batch,
        capacity=prefix_length + steps,
        device=prefix_input_ids.device,
        workspace_bytes=workspace_bytes,
    )
    growing_endpoint = BF16DecodeEndpoint(model, growing_cache, adapter)
    growing_outputs: list[Any] = []
    with torch.inference_mode(), forced_flash_execution():
        growing_endpoint.prefill(prefix_input_ids)
        growing_cache.prepare_growing(prefix_length, steps)
        for step in range(steps):
            position = torch.tensor(
                [prefix_length + step],
                dtype=torch.long,
                device=prefix_input_ids.device,
            )
            embeddings = growing_endpoint.prepare_position_embeddings(
                position.unsqueeze(0)
            )
            growing_cache.select_growing_step(step)
            output = growing_endpoint.decode(
                decode_input_ids[:, step : step + 1],
                position,
                embeddings,
            )
            growing_cache.finish_growing_step()
            growing_outputs.append(output.detach().cpu().clone())

    fixed_evidence = tuple(
        FullModelStepEvidence(
            mode="fixed_l",
            step=step,
            position=prefix_length,
            reference_checksum=tensor_sha256_untimed(reference_outputs[0]),
            observed_checksum=tensor_sha256_untimed(output),
            comparison=compare_tensors_untimed(
                output,
                reference_outputs[0],
                atol=atol,
                rtol=rtol,
            ),
        )
        for step, output in enumerate(fixed_outputs)
    )
    growing_evidence = tuple(
        FullModelStepEvidence(
            mode="growing_context",
            step=step,
            position=prefix_length + step,
            reference_checksum=tensor_sha256_untimed(reference_outputs[step]),
            observed_checksum=tensor_sha256_untimed(output),
            comparison=compare_tensors_untimed(
                output,
                reference_outputs[step],
                atol=atol,
                rtol=rtol,
            ),
        )
        for step, output in enumerate(growing_outputs)
    )
    fixed_repeat_exact = all(
        torch.equal(fixed_outputs[0], output) for output in fixed_outputs[1:]
    )
    fixed_history_unchanged = historical_before == historical_after
    passed = (
        restored
        and fixed_repeat_exact
        and fixed_history_unchanged
        and all(item.comparison.passed for item in fixed_evidence)
        and all(item.comparison.passed for item in growing_evidence)
    )
    return FullModelReferenceResult(
        passed=passed,
        reference_implementation="transformers_eager_dynamic_cache",
        reference_cache_type=reference_cache_type,
        reference_implementation_restored=restored,
        tolerance_atol=float(atol),
        tolerance_rtol=float(rtol),
        fixed_repeat_exact=fixed_repeat_exact,
        fixed_historical_cache_unchanged=fixed_history_unchanged,
        fixed_steps=fixed_evidence,
        growing_steps=growing_evidence,
    )
