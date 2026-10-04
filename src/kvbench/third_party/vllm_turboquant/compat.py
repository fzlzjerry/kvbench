"""Narrow import compatibility for the pinned vLLM TurboQuant source."""

from __future__ import annotations

import functools
import math

import torch


class _CurrentPlatform:
    """Only the vLLM platform predicate used by the pinned FP8 helper."""

    @staticmethod
    def is_cuda_alike() -> bool:
        return bool(torch.cuda.is_available())


current_platform = _CurrentPlatform()
_FP8_E4B15: dict[int, int] = {}


def use_fp8_e4b15(device: int = 0) -> int:
    """Match the pinned vLLM CUDA capability selection without vLLM imports."""

    if device not in _FP8_E4B15:
        if current_platform.is_cuda_alike():
            capability = torch.cuda.get_device_capability(device)
            _FP8_E4B15[device] = 1 if capability < (8, 9) else 0
        else:
            _FP8_E4B15[device] = 0
    return _FP8_E4B15[device]


def _build_hadamard(d: int, device_str: str) -> torch.Tensor:
    """Orthonormal Hadamard matrix (Sylvester construction), cached per (d, device).

    Precomputed D×D matrix enables matmul-based WHT — single cuBLAS GEMM
    instead of log2(D) butterfly kernel launches. 64KB for D=128.
    """
    # Normalize device string so "cuda" and "cuda:0" hit the same cache entry.
    return _build_hadamard_cached(d, str(torch.device(device_str)))


@functools.cache
def _build_hadamard_cached(d: int, device_str: str) -> torch.Tensor:
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(torch.device(device_str))


build_hadamard = _build_hadamard
