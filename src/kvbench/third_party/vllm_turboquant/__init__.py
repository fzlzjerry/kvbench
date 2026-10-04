"""Minimal pinned vLLM TurboQuant surface for the Phase 6 adapter."""

from .centroids import get_centroids, solve_lloyd_max
from .compat import build_hadamard
from .config import TQ_PRESETS, TurboQuantConfig
from .triton_decode_attention import _fwd_kernel_stage2
from .triton_turboquant_decode import _tq_decode_stage1
from .triton_turboquant_store import _tq_fused_store_mse

__all__ = [
    "TQ_PRESETS",
    "TurboQuantConfig",
    "_fwd_kernel_stage2",
    "_tq_decode_stage1",
    "_tq_fused_store_mse",
    "build_hadamard",
    "get_centroids",
    "solve_lloyd_max",
]
