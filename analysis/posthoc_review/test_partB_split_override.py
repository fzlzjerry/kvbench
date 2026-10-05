#!/usr/bin/env python3
"""Mechanism test for the POST-HOC TurboQuant split override (no model, small CUDA work).

Builds two identical TurboQuant caches from the Phase 13B sanitizer-probe recipe
(random BF16 K/V, one compressed layer), runs append + decode eagerly and as a
captured CUDA graph, once as-ported (4 splits) and once with the override
(32 splits), and checks that:

  * the override launches with 32 splits through a caller-owned (B, Hq, 32, D+1)
    scratch and swaps nothing outside each call;
  * cache-owned state is untouched: pointers, layout fingerprint, byte breakdown;
  * graph replay equals eager output under the override, and the 32-split output
    agrees with the 4-split output to the Phase 12 TurboQuant tolerance
    (split-KV changes only the reduction order).

Diagnostic development test; not evidence and not timing.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]

import torch  # noqa: E402

from kvbench.adapters.base import MethodRuntimeContext  # noqa: E402
from kvbench.adapters.turboquant import TurboQuantMethodAdapter  # noqa: E402
import kvbench.adapters.turboquant as adapter_module  # noqa: E402
from kvbench.runtime.cuda_graph import capture_fixed_graph  # noqa: E402

spec = importlib.util.spec_from_file_location("partB_worker", Path(__file__).with_name("partB_worker.py"))
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)

BATCH, PREFIX, CAPACITY, LAYER = 2, 4096, 4097, 2


def context() -> MethodRuntimeContext:
    return MethodRuntimeContext(
        model_id="posthoc-split-override-test", model_revision="test",
        backend_id="posthoc-tq", backend_fingerprint=hashlib.sha256(b"posthoc-tq").hexdigest(),
        num_layers=32, num_query_heads=32, num_kv_heads=8, head_dim=128)


def build(seed: int = 1234):
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    key = torch.randn((1, 8, CAPACITY, 128), generator=generator, dtype=torch.bfloat16,
                      device=device).expand(BATCH, -1, -1, -1).clone()
    value = torch.randn((1, 8, CAPACITY, 128), generator=generator, dtype=torch.bfloat16,
                        device=device).expand(BATCH, -1, -1, -1).clone()
    query = torch.randn((1, 32, 1, 128), generator=generator, dtype=torch.bfloat16,
                        device=device).expand(BATCH, -1, -1, -1).clone()
    positions = torch.arange(CAPACITY, dtype=torch.int32, device=device)
    method = TurboQuantMethodAdapter(context(), "turboquant_4bit_nc")
    if hasattr(method, "prepare_runtime"):
        method.prepare_runtime()
    cache = method.allocate(batch_size=BATCH, capacity=CAPACITY, device=device)
    cache.initialize_deterministic()
    cache.prepare_prefill(PREFIX)
    method.store_prefill(cache, key[:, :, :PREFIX], value[:, :, :PREFIX], LAYER, positions[:PREFIX])
    cache.complete_prefill()
    append_position = positions[PREFIX:PREFIX + 1]
    cache.prepare_fixed(PREFIX)
    attention = SimpleNamespace(layer_idx=LAYER)

    def operation():
        handles = method.append_decode(cache, key[:, :, PREFIX:PREFIX + 1], value[:, :, PREFIX:PREFIX + 1],
                                       LAYER, append_position)
        return method.decode_attention(attention, query, handles[0], handles[1],
                                       scaling=1.0 / math.sqrt(128))

    return method, cache, operation


def state(cache) -> dict:
    return {"pointers": cache.pointers(), "layout": cache.layout_fingerprint(),
            "breakdown": cache.byte_breakdown(), "mid_o_shape": list(cache.decode_mid_o.shape),
            "mid_o_ptr": cache.decode_mid_o.data_ptr()}


def run(override: bool) -> dict:
    method, cache, operation = build()
    before = state(cache)
    eager = operation().detach().clone()
    graph = capture_fixed_graph(operation, warmup_steps=1, device=cache.device)
    replay = graph.replay().detach().clone()
    torch.cuda.synchronize()
    after = state(cache)
    graph.graph.reset()
    return {"eager": eager.float().cpu(), "replay": replay.float().cpu(), "before": before, "after": after,
            "module_splits_after": adapter_module.TURBOQUANT_MAX_KV_SPLITS}


def main() -> int:
    baseline = run(override=False)
    original = TurboQuantMethodAdapter.__dict__["_decode_compressed"]
    override = worker.install_tq_split_override(32)
    try:
        split32 = run(override=True)
    finally:
        TurboQuantMethodAdapter._decode_compressed = original
    difference = (split32["replay"] - baseline["replay"]).abs()
    tolerance = 0.02 + 0.02 * baseline["replay"].abs()
    checks = {
        "as_ported_cache_state_unchanged": baseline["before"] == baseline["after"],
        "override_cache_state_unchanged": split32["before"] == split32["after"],
        "layout_fingerprint_same_as_as_ported": split32["after"]["layout"] == baseline["after"]["layout"],
        "byte_breakdown_same_as_as_ported": split32["after"]["breakdown"] == baseline["after"]["breakdown"],
        "cache_mid_o_still_4_splits": split32["after"]["mid_o_shape"][2] == 4,
        "module_constant_restored": split32["module_splits_after"] == 4,
        "scratch_is_32_splits": [list(t.shape) for t in override["scratch"].values()] == [[BATCH, 32, 32, 129]],
        "decode_called_during_capture": override["calls_during_capture"] >= 1,
        "override_graph_equals_eager": bool(torch.equal(split32["eager"], split32["replay"])),
        "split32_within_phase12_tq_tolerance": bool((difference <= tolerance).all()),
        "outputs_finite": bool(torch.isfinite(split32["replay"]).all()),
    }
    report = {"checks": checks, "max_abs_difference_split32_vs_split4": float(difference.max()),
              "override_calls": override["calls"], "calls_during_capture": override["calls_during_capture"],
              "all_pass": all(checks.values())}
    print(json.dumps(report, indent=2))
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
