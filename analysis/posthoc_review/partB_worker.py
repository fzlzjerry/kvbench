#!/usr/bin/env python3
"""POST-HOC Part B diagnostic profiling worker (one decode session per process).

POST-HOC DIAGNOSTIC, NOT A TIMING RUN.  This worker runs inside the authorized
measurement container (sha256:059bc9be...) with the Full Scan execution
repository (ec534d99) mounted read-only at /home/rockrock/cmu_paper.  Profiler
durations it produces are never benchmark timing.

Session construction is the Full Scan's own: scripts.phase16_full_scan decides
the prefix source for the point (BF16 restores its layout-compatible snapshot;
TurboQuant, KIVI and KVQuant reconstruct the cache from the frozen logical
prefix by prefill), and scripts.phase15_profiler_subset builds and captures the
session exactly as the Phase 15 profiler worker did.  After 64 graph replays of
warmup, the worker replays the captured graph inside the Phase 15 NVTX range
("phase15_decode"), which is the only region nsys or ncu captures.

--tq-splits N (TurboQuant only; diagnostic): every
TurboQuantMethodAdapter._decode_compressed call launches the stage-1 and stage-2
kernels with N KV splits and a caller-owned FP32 scratch of shape
(B, Hq, N, D + 1).  The scratch and the split constant are swapped in only for
the duration of each call and restored before it returns, so the cache's owned
tensors, byte accounting, layout and adapter fingerprints, and pointer checks
are those of the measured configuration.  N = 4 (the as-ported value) installs
nothing.  vLLM v0.25.1 defaults to 32 (tq_max_kv_splits_for_cuda_graph).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any

NVTX_RANGE = "phase15_decode"
AS_PORTED_SPLITS = 4
RESULT_PREFIX = "POSTHOC_PARTB_WORKER_RESULT="


def install_tq_split_override(splits: int) -> dict[str, Any]:
    import torch
    import kvbench.adapters.turboquant as adapter_module

    if adapter_module.TURBOQUANT_MAX_KV_SPLITS != AS_PORTED_SPLITS:
        raise RuntimeError("as-ported TurboQuant split count differs from 4")
    adapter = adapter_module.TurboQuantMethodAdapter
    raw = adapter.__dict__["_decode_compressed"]
    original = raw.__func__ if isinstance(raw, staticmethod) else raw
    state: dict[str, Any] = {
        "splits": splits,
        "as_ported_splits": AS_PORTED_SPLITS,
        "scratch": {},
        "calls": 0,
        "calls_during_capture": 0,
    }

    def decode_with_splits(handle: Any, query_states: Any, scaling: float) -> Any:
        cache = handle.cache
        capturing = bool(torch.cuda.is_current_stream_capturing())
        scratch = state["scratch"].get(id(cache))
        if scratch is None:
            if capturing:
                raise RuntimeError("split scratch would be allocated during graph capture")
            scratch = torch.empty(
                (cache.batch_size, cache.num_query_heads, splits, cache.head_dim + 1),
                dtype=torch.float32,
                device=cache.device,
            )
            state["scratch"][id(cache)] = scratch
        saved_scratch = cache.decode_mid_o
        saved_splits = adapter_module.TURBOQUANT_MAX_KV_SPLITS
        cache.decode_mid_o = scratch
        adapter_module.TURBOQUANT_MAX_KV_SPLITS = splits
        state["calls"] += 1
        state["calls_during_capture"] += int(capturing)
        try:
            return original(handle, query_states, scaling)
        finally:
            cache.decode_mid_o = saved_scratch
            adapter_module.TURBOQUANT_MAX_KV_SPLITS = saved_splits

    adapter._decode_compressed = staticmethod(decode_with_splits)
    return state


def run(args: argparse.Namespace) -> dict[str, Any]:
    import scripts.phase12_unified_admission as phase12

    attestation = phase12._require_authorized_container_runtime()
    import scripts.phase15_profiler_subset as p15
    import scripts.phase16_full_scan as p16
    import torch
    from kvbench.runtime.numerical import tensor_sha256_untimed
    from kvbench.runtime.timing import warmup_operations

    configuration = args.configuration
    override_state = None
    if args.tq_splits != AS_PORTED_SPLITS:
        if not configuration.startswith("tq_"):
            raise RuntimeError("--tq-splits applies to TurboQuant configurations only")
        override_state = install_tq_split_override(args.tq_splits)
    p15._install_profiler_adapter_nvtx_annotations()

    historical = p16.actual_historical_context(args.context_label)
    record = {
        "method_config_id": configuration,
        "batch_size": args.batch_size,
        "context_label": args.context_label,
        "historical_context": historical,
    }
    entry = p16.prefix_entry(record, container_paths=True,
                             logical_root=Path(args.family_root) / "logical-prefixes")
    snapshot_root = entry["snapshot_root"] or str(Path(args.output_dir) / "no-prefix-snapshot")
    expected_sha = entry["state_file_sha256"] or "0" * 64
    with (p16._worker_overrides(batch=args.batch_size, entry=entry),
          p16._worker_context_label_override(args.context_label)):
        session, receipt = p15._build_profiler_session(
            configuration=configuration,
            batch=args.batch_size,
            context_label=args.context_label,
            graph_mode="cuda_graph",
            prefix_entry={"snapshot_root": snapshot_root, "state_file_sha256": expected_sha},
        )
        if session.graph is None:
            raise RuntimeError("Graph session did not capture a CUDA graph")
        # Return prefix-construction blocks cached by the allocator to the
        # driver so profiler replay buffers have device memory.  The captured
        # graph's private pool and every cache tensor stay allocated.
        free_before, total = torch.cuda.mem_get_info()
        reserved_before = torch.cuda.memory_reserved()
        torch.cuda.empty_cache()
        free_after, _ = torch.cuda.mem_get_info()
        memory_record = {
            "free_bytes_before_empty_cache": free_before,
            "free_bytes_after_empty_cache": free_after,
            "reserved_bytes_before_empty_cache": reserved_before,
            "reserved_bytes_after_empty_cache": torch.cuda.memory_reserved(),
            "allocated_bytes": torch.cuda.memory_allocated(),
            "total_bytes": total,
        }
        pointers_before = phase12._phase12_session_pointers(session)
        history_before = session.current_historical_prefix_sha256()
        operation = session.graph.replay
        warmed = warmup_operations(operation, count=args.warmup_steps, device=session.cache_device)
        torch.cuda.synchronize(device=session.cache_device)
        if not bool(torch.isfinite(warmed).all()):
            raise RuntimeError("warmup output is non-finite")
        output = None
        torch.cuda.nvtx.range_push(NVTX_RANGE)
        try:
            for _ in range(args.decode_operations):
                output = operation()
            # Profiler-boundary synchronization; never used as timing.
            torch.cuda.synchronize(device=session.cache_device)
        finally:
            torch.cuda.nvtx.range_pop()
        output_cpu = output.detach().to(device="cpu", copy=True).clone()
        pointers_after = phase12._phase12_session_pointers(session)
        history_after = session.current_historical_prefix_sha256()
        family = phase12._method_family(configuration)
        try:
            replayed_fingerprint = phase12._validate_runtime_adapter_fingerprint(
                method=session.method, cache=session.cache,
                observed=session.adapter_config_fingerprint)
            fingerprint_error = None
        except Exception as error:  # recorded, not raised: diagnostic only
            replayed_fingerprint, fingerprint_error = None, f"{type(error).__name__}: {error}"
        result = {
            "schema_version": "kvbench-posthoc-partb-worker-1.0.0",
            "status": "completed",
            "posthoc_diagnostic": True,
            "profiler_duration_is_normal_timing": False,
            "performance_claim_eligible": False,
            "method_config_id": configuration,
            "method_family": family,
            "batch_size": args.batch_size,
            "context_label": args.context_label,
            "historical_context": historical,
            "graph_mode": "cuda_graph",
            "warmup_steps": args.warmup_steps,
            "decode_operations": args.decode_operations,
            "nvtx_range": NVTX_RANGE,
            "tq_splits": args.tq_splits if configuration.startswith("tq_") else None,
            "tq_split_override": None if override_state is None else {
                "splits": override_state["splits"],
                "as_ported_splits": override_state["as_ported_splits"],
                "scratch_shapes": [list(t.shape) for t in override_state["scratch"].values()],
                "decode_calls": override_state["calls"],
                "decode_calls_during_capture": override_state["calls_during_capture"],
                "swap_scope": "per _decode_compressed call; cache-owned tensors unchanged",
            },
            "prefix_source": {k: entry[k] for k in ("kind", "restore_mode", "source", "snapshot_root",
                                                    "state_file_sha256") if k in entry},
            "logical_prefix_id": entry.get("logical_prefix", {}).get("logical_prefix_id"),
            "prefix_receipt": json.loads(json.dumps(receipt, default=str)),
            "output_checksum": tensor_sha256_untimed(output_cpu),
            "output_finite": bool(torch.isfinite(output_cpu).all()),
            "cache_pointers_stable": pointers_before == pointers_after,
            "historical_cache_unchanged": history_before == history_after,
            "cache_layout_fingerprint": session.cache_layout_fingerprint(),
            "adapter_config_fingerprint": session.adapter_config_fingerprint,
            "adapter_config_fingerprint_replayed": replayed_fingerprint,
            "adapter_fingerprint_validation_error": fingerprint_error,
            "cache_accounting": json.loads(json.dumps(session.method_cache_accounting(), default=str)),
            "graph_evidence": json.loads(json.dumps(session.graph_evidence, default=str)),
            "device_memory_after_build": memory_record,
            "container_runtime_attestation": attestation,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0),
            "execution_repository_head": os.popen(
                "/usr/bin/git -C /home/rockrock/cmu_paper rev-parse HEAD").read().strip(),
            "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="POST-HOC Part B diagnostic profiling worker")
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--context-label", type=int, required=True)
    parser.add_argument("--tq-splits", type=int, default=AS_PORTED_SPLITS)
    parser.add_argument("--decode-operations", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=64)
    parser.add_argument("--family-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result_path = Path(args.output_dir) / "worker_result.json"
    code = 0
    try:
        payload = run(args)
    except BaseException as error:  # the runner records the failure and continues
        payload = {
            "schema_version": "kvbench-posthoc-partb-worker-1.0.0",
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        code = 3
    text = json.dumps(payload, sort_keys=True, default=str)
    with result_path.open("x") as handle:
        handle.write(text + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    sys.stdout.write(RESULT_PREFIX + text + "\n")
    sys.stdout.flush()
    # Same exit path as the Phase 15 profiler worker: skip interpreter teardown.
    os._exit(code)


if __name__ == "__main__":
    main()
