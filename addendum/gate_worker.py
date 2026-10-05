#!/usr/bin/env python3
"""Addendum gate worker (never timing): the original G3/G4 items on a KIVI variant.

Runs inside the measurement container like check_worker.py.  The original
_decode_compressed is captured, the variant is installed (timing_worker.install_variant),
and the frozen Phase 8 admission derivation is reused unchanged from
scripts/phase8_kivi_admission.py (_preallocate_first_identity,
_hot_path_temporary_shapes, _static_execution_path_precheck,
_capture_launcher_probe, _audit_session, _execution_path_audit,
_derive_local_candidate) over kvbench.runtime.kivi_admission,
kivi_allocation, allocation and cuda_graph.  Because inspect.getsource
resolves the class attribute, the static and path audits read the variant's
source.

Differences from the Phase 8 bundle (all recorded in the result):
* scope: the configuration's two Phase 8 grid points (fixed-L, L=128, B=1,
  eager and cuda_graph) plus one companion eager L=128 point whose launcher
  probe is pooled, as Phase 8 pooled its ten points, so that both official
  kernel families (bgemv2 and bgemv4) can be observed;
* the Phase 8 exact-container tests run in this process through unittest
  (the variant is a process-local patch), not as supervised children;
* the fixed-L common runner (a timing harness) is not executed;
* Compute Sanitizer is a separate driver check.

native_gqa is reported literally (a source-string test that the grouped form
cannot satisfy) and semantically: per-head numerical equivalence with the
captured original, K/V operand provenance (no K/V tensor with a 32-head axis),
and source tokens.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
from typing import Any, Callable, Iterator
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import overrides  # noqa: E402
import timing_worker  # noqa: E402
from timing_worker import install_variant  # noqa: E402

RESULT_PREFIX = "ADDENDUM_GATE_RESULT="
SCHEMA = "kvbench-addendum-20261005-gate-1.0.0"
DEFAULT_VARIANT = '{"kivi_grouped_residual": true}'
ALLOWED_VARIANTS = ({}, {"kivi_grouped_residual": True})
CONFIGURATIONS = ("k4v4", "k2v2")
AUTO_COMPANION = {"k4v4": "k2v2", "k2v2": "k4v4"}
PHASE8_CONTEXT = 128  # PHASE8_ADMISSION_GRID fixed-L points for k4v4/k2v2 (output_steps 1)
G3_ITEMS = ("no_measured_torch_cat", "direct_compressed_decode", "native_gqa",
            "no_unknown_allocation", "no_backend_fallback")
G4_ITEMS = ("graph_capture_replay", "graph_zero_replay_allocation", "no_backend_fallback")
NATIVE_GQA_REASON = "native_gqa_indexing_unverified"
LITERAL_NATIVE_GQA_TOKEN = "kv_head = query_head // kivi_gqa_group_size"
SEMANTIC_FORBIDDEN_TOKENS = ("repeat_kv", "repeat_interleave", ".expand(", "torch.cat(", ".cat(",
                             ".item(", ".cpu(", ".tolist(", "synchronize(")
TEST_MODULES = (("fixture_conformance", "tests/cuda/test_phase8_kivi_cuda.py"),
                ("graph_harness", "tests/graph/test_phase8_kivi_graph.py"))
# (batch, prefix): 17 = residual only, 100 = K history 96 + K residual 4 and a
# wrapped V ring (head 4), 128 = K residual empty, V residual 32.
EQUIVALENCE_FIXED = ((1, 17), (1, 100), (1, 128), (8, 17), (8, 100), (8, 128))
# (batch, prefix, steps): 31 crosses the 32-token K group flush and V eviction.
EQUIVALENCE_GROWING = ((1, 31, 3), (1, 100, 3), (8, 31, 3), (8, 100, 3))
MIN_EQUIVALENCE_CAPACITY = 35  # TEST_CAPACITY of tests/cuda/test_phase8_kivi_cuda.py
KV_STORAGE_ROLES = {"key_residual": "qk_residual", "key_fp16_staging": "qk_pending",
                    "value_residual_ordered_staging": "pv_residual",
                    "value_residual_ring": "pv_residual_ring", "value_fp16_staging": "pv_pending"}
NUM_KV_HEADS = 8
REPO = Path(common.CONTAINER_REPO)
INPUTS = Path(common.CONTAINER_INPUTS)
PROVENANCE_SOURCES = (
    "scripts/phase8_kivi_admission.py", "scripts/phase12_unified_admission.py",
    "src/kvbench/runtime/kivi_admission.py", "src/kvbench/runtime/kivi_allocation.py",
    "src/kvbench/runtime/allocation.py", "src/kvbench/runtime/cuda_graph.py",
    "src/kvbench/runtime/kivi_session.py", "src/kvbench/runtime/kivi_cache.py",
    "src/kvbench/adapters/kivi.py", "tests/cuda/test_phase8_kivi_cuda.py",
    "tests/graph/test_phase8_kivi_graph.py",
)
P8 = "scripts/phase8_kivi_admission.py"
PATH_AUDIT = (f"{P8}::_execution_path_audit -> kvbench.runtime.kivi_admission.audit_kivi_execution_path"
              f"(kivi_adapter_hot_path_source; kernels from {P8}::_capture_launcher_probe; temporary "
              f"shapes from {P8}::_hot_path_temporary_shapes(_preallocate_first_identity cache))")
ALLOCATION = (f"{P8}::_audit_session -> kvbench.runtime.kivi_allocation.collect_kivi_allocation_attribution"
              " (kvbench.runtime.allocation.collect_cuda_allocator_raw)")
GRAPH = ("graph_harness unittest (kvbench.runtime.cuda_graph.capture_fixed_graph, "
         "kvbench.runtime.allocation.audit_cuda_allocations) + "
         f"{P8}::_audit_session graph_passed (kivi_session.build_kivi_endpoint_session -> capture_fixed_graph)")
DERIVED_WITH = {
    "no_measured_torch_cat": f"{P8}::_derive_local_candidate <- {PATH_AUDIT}",
    "direct_compressed_decode": f"{P8}::_derive_local_candidate <- {PATH_AUDIT}",
    "native_gqa": (f"{P8}::_derive_local_candidate <- {PATH_AUDIT} (source string, gqa materialization, "
                   "_query_head_sized_kv_temporary) + KIVIStaticCache.gqa_geometry per point"),
    "no_unknown_allocation": f"{P8}::_derive_local_candidate <- {ALLOCATION}",
    "no_backend_fallback": f"{P8}::_derive_local_candidate <- {PATH_AUDIT}; graph_harness unittest; launcher probes",
    "graph_capture_replay": f"{P8}::_derive_local_candidate <- {GRAPH}",
    "graph_zero_replay_allocation": (f"{P8}::_derive_local_candidate <- {GRAPH}; "
                                     f"{ALLOCATION} strict_graph_zero_events"),
}
NOT_REPRODUCED = [
    "Phase 8 pooled launcher probes and allocation audits over its ten-point grid (k4v4/k2v4/k2v2/k4v2, "
    "L=128 and 4096, growing L=31); this worker uses the configuration's fixed-L L=128 eager and "
    "cuda_graph points plus one companion-configuration eager L=128 point (pooled for the kernel-family "
    "criteria two_bit/four_bit, which a single configuration cannot satisfy).",
    "Phase 8 ran the fixture and graph tests as supervised child processes; here they run in-process via "
    "unittest because the variant is a process-local patch (child_process_supervision not evaluated).",
    "The fixed-L common runner (a timing harness) is not executed: point native_gqa uses "
    "session.gqa_cache_geometry() (the call that runner makes) and cache_pointers_stable uses "
    "_audit_session pointers_stable.",
    "Compute Sanitizer and the bounded-grid / byte-accounting / G1-G2 items are out of scope here.",
    "native_gqa: the literal source-string criterion is reported as-is; the gate uses a semantic replacement.",
]


# ------------------------------------------------------------------ helpers

def _error_record(error: BaseException) -> dict[str, Any]:
    return {"error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}


def _write_json(path: Path, value: Any) -> None:
    common.write_new(path, common.json_text(value))


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _file_sha256(path: Path) -> str | None:
    return common.sha256_file(path) if path.is_file() else None


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - best-effort cleanup between stages
        pass


class StageLog:
    """Run stages independently; each stage writes an append-only progress record."""

    def __init__(self, root: Path) -> None:
        self.root = root / "stages"
        self.count = 0
        self.errors: dict[str, dict[str, Any]] = {}

    def run(self, name: str, function: Callable[[], Any]) -> Any:
        self.count += 1
        started = common.utc_now()
        try:
            value = function()
            record: dict[str, Any] = {"stage": name, "status": "completed"}
        except Exception as error:  # noqa: BLE001 - recorded, later stages still run
            value = None
            record = {"stage": name, "status": "failed", **_error_record(error)}
            self.errors[name] = record
        record.update({"started_at_utc": started, "finished_at_utc": common.utc_now(), "value": value})
        path = self.root / f"{self.count:02d}-{name}.json"
        try:
            _write_json(path, record)
        except Exception as error:  # noqa: BLE001 - serialization happens before the file is created
            failure = {"stage": name, "status": "progress_record_unserializable", **_error_record(error)}
            self.errors[f"{name}.progress_record"] = failure
            _write_json(path, failure)
        return value


# ------------------------------------------------------- unittest modules

def _test_id(test: Any) -> str:
    try:
        return str(test.id())
    except Exception:  # noqa: BLE001
        return str(test)


class RecordingResult(unittest.TestResult):
    """Per-test outcomes; a skip is recorded as a skip (never as a pass)."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict[str, Any]] = []
        self.successes: list[str] = []

    def _record(self, test: Any, outcome: str, message: str | None = None) -> None:
        self.records.append({"test": _test_id(test), "outcome": outcome,
                             "message": None if message is None else message[-20000:]})

    def addSuccess(self, test: Any) -> None:  # noqa: N802
        super().addSuccess(test)
        self.successes.append(_test_id(test))
        self._record(test, "pass")

    def addFailure(self, test: Any, err: Any) -> None:  # noqa: N802
        super().addFailure(test, err)
        self._record(test, "fail", self.failures[-1][1])

    def addError(self, test: Any, err: Any) -> None:  # noqa: N802
        super().addError(test, err)
        self._record(test, "error", self.errors[-1][1])

    def addSkip(self, test: Any, reason: str) -> None:  # noqa: N802
        super().addSkip(test, reason)
        self._record(test, "skip", reason)

    def addExpectedFailure(self, test: Any, err: Any) -> None:  # noqa: N802
        super().addExpectedFailure(test, err)
        self._record(test, "expected_failure", self.expectedFailures[-1][1])

    def addUnexpectedSuccess(self, test: Any) -> None:  # noqa: N802
        super().addUnexpectedSuccess(test)
        self._record(test, "unexpected_success")

    def addSubTest(self, test: Any, subtest: Any, err: Any) -> None:  # noqa: N802
        super().addSubTest(test, subtest, err)
        if err is None:
            self._record(subtest, "subtest_pass")
        else:
            failed = issubclass(err[0], test.failureException)
            self._record(subtest, "subtest_fail" if failed else "subtest_error",
                         (self.failures if failed else self.errors)[-1][1])


def run_unittest_module(label: str, relative: str) -> dict[str, Any]:
    """Load one frozen test file and run it in this (variant-patched) process."""
    primary = REPO / relative
    fallback = INPUTS / Path(relative).name
    if primary.is_file():
        path, origin = primary, "execution_repository"
    elif fallback.is_file():
        path, origin = fallback, "addendum_inputs"
    else:
        return {"label": label, "relative_path": relative, "available": False, "passed": False,
                "reason": "absent from the execution repository and from /opt/addendum-inputs"}
    record: dict[str, Any] = {"label": label, "relative_path": relative, "available": True,
                              "origin": origin, "path": str(path), "sha256": common.sha256_file(path)}
    try:
        spec = importlib.util.spec_from_file_location(f"addendum_gate_{label}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    except Exception as error:  # noqa: BLE001
        return {**record, "passed": False, "load_error": _error_record(error)}
    declared = getattr(module, "_authorized_environment_declared", None)
    record["authorized_environment_declared"] = declared() if callable(declared) else None
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    result = RecordingResult()
    suite.run(result)
    passed = bool(
        result.testsRun > 0
        and len(result.successes) == result.testsRun
        and not result.failures and not result.errors and not result.skipped
        and not result.expectedFailures and not result.unexpectedSuccesses
    )
    _release_cuda()
    return {**record, "passed": passed, "tests_run": result.testsRun,
            "counts": {"pass": len(result.successes), "fail": len(result.failures),
                       "error": len(result.errors), "skip": len(result.skipped),
                       "expected_failure": len(result.expectedFailures),
                       "unexpected_success": len(result.unexpectedSuccesses)},
            "skip_counts_as_not_passed": True, "outcomes": result.records}


# ------------------------------------------- semantic native GQA replacement

def _owned_storage_names(cache: Any) -> dict[int, str]:
    names = [key[: -len("_data_ptr")] for key in cache.pointers()]
    result: dict[int, str] = {}
    for name, tensor in zip(names, cache._owned_tensors(), strict=True):
        if tensor.device.type == "cuda" and tensor.numel() > 0:
            result[int(tensor.untyped_storage().data_ptr())] = name
    return result


def _describe(tensor: Any, names: dict[int, str]) -> dict[str, Any]:
    return {"shape": [int(v) for v in tensor.shape], "stride": [int(v) for v in tensor.stride()],
            "alias": names.get(int(tensor.untyped_storage().data_ptr()), "unaliased")}


@contextlib.contextmanager
def _recording_bmm(torch: Any, cache: Any) -> Iterator[list[dict[str, Any]]]:
    """Record bmm operand geometry and storage (host-side only; numerics unchanged)."""
    names = _owned_storage_names(cache)
    calls: list[dict[str, Any]] = []
    real = torch.bmm

    def bmm(input: Any, mat2: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: A002
        out = kwargs.get("out")
        calls.append({"input": _describe(input, names), "mat2": _describe(mat2, names),
                      "out": None if out is None else _describe(out, names)})
        return real(input, mat2, *args, **kwargs)

    torch.bmm = bmm
    try:
        yield calls
    finally:
        torch.bmm = real


@contextlib.contextmanager
def _decode_implementation(adapter_class: Any, function: Any) -> Iterator[None]:
    saved = adapter_class.__dict__["_decode_compressed"]
    adapter_class._decode_compressed = function
    try:
        yield
    finally:
        adapter_class._decode_compressed = saved


def _counts(cache: Any) -> dict[str, Any]:
    return {"key_history": cache._key_history_counts[0], "key_residual": cache._key_residual_counts[0],
            "value_history": cache._value_history_counts[0],
            "value_residual": cache._value_residual_counts[0],
            "value_residual_head": cache._value_residual_heads[0],
            "active_context": int(cache.active_context), "mode": cache.mode}


def _state(cache: Any) -> dict[str, Any]:
    return {"history_checksum_layer0": cache.history_checksum(0),
            "pointers_sha256": _text_sha256(json.dumps(cache.pointers(), sort_keys=True)), **_counts(cache)}


def _inputs(ctx: Any, batch: int, tokens: int, queries: int, seed: int) -> tuple[Any, Any, Any]:
    torch = ctx.torch
    generator = torch.Generator(device=ctx.device)
    generator.manual_seed(seed)

    def draw(shape: tuple[int, ...]) -> Any:
        return torch.randn(shape, dtype=torch.bfloat16, device=ctx.device, generator=generator)

    return (draw((batch, NUM_KV_HEADS, tokens, 128)), draw((batch, NUM_KV_HEADS, tokens, 128)),
            draw((batch, 32, queries, 128)))


def _prefilled_cache(ctx: Any, method: Any, batch: int, capacity: int, key: Any, value: Any,
                     prefix: int) -> Any:
    torch = ctx.torch
    cache = method.allocate(batch_size=batch, capacity=capacity, device=ctx.device)
    cache.initialize_deterministic()
    cache.prepare_prefill(prefix)
    method.store_prefill(cache, key[:, :, :prefix, :], value[:, :, :prefix, :], 0,
                         torch.arange(prefix, dtype=torch.int64, device=ctx.device))
    cache.complete_prefill()
    return cache


def _decode_once(ctx: Any, method: Any, cache: Any, function: Any, key: Any, value: Any,
                 query: Any, token: int) -> dict[str, Any]:
    """append_decode + decode_attention once, with `function` as _decode_compressed."""
    torch = ctx.torch
    position = torch.tensor([token], dtype=torch.int64, device=ctx.device)
    with _decode_implementation(ctx.adapter_class, function), \
            _recording_bmm(torch, cache) as calls, \
            mock.patch("kvbench.adapters.kivi.flash_attention_forward",
                       side_effect=AssertionError("compressed decode fell back to a BF16 attention backend")):
        handles = method.append_decode(cache, key, value, 0, position)
        output = method.decode_attention(ctx.attention, query, handles[0], handles[1], scaling=ctx.scaling)
        torch.cuda.synchronize(device=ctx.device)
    return {"output": output.detach().cpu().clone(),
            "output_fp16": cache.decode_output_fp16.detach().cpu().clone(),
            "output_is_cache_buffer": int(output.data_ptr()) == int(cache.output_buffer.data_ptr()),
            "calls": calls}


def _compare(ctx: Any, observed: Any, reference: Any) -> dict[str, Any]:
    comparison = ctx.compare(observed, reference, atol=ctx.atol, rtol=ctx.rtol)
    return {**comparison.to_dict(), "bitwise_equal": bool(ctx.torch.equal(observed, reference))}


def _fixed_case(ctx: Any, method: Any, implementations: tuple, batch: int, prefix: int,
                seed: int) -> tuple[dict[str, Any], dict[str, list]]:
    key, value, query = _inputs(ctx, batch, prefix + 1, 1, seed)
    cache = _prefilled_cache(ctx, method, batch, max(prefix + 1, MIN_EQUIVALENCE_CAPACITY), key, value, prefix)
    cache.prepare_fixed(prefix)
    states = [_state(cache)]
    runs = {}
    for name, function in implementations:
        runs[name] = _decode_once(ctx, method, cache, function, key[:, :, prefix:prefix + 1, :],
                                  value[:, :, prefix:prefix + 1, :], query[:, :, 0:1, :], prefix)
        states.append(_state(cache))
    reference, observed = runs[implementations[0][0]], runs[implementations[1][0]]
    output_bf16 = _compare(ctx, observed["output"], reference["output"])
    output_fp16 = _compare(ctx, observed["output_fp16"], reference["output_fp16"])
    unchanged = all(state == states[0] for state in states[1:])
    record = {"kind": "fixed", "batch_size": batch, "prefix_length": prefix, "seed": seed,
              "capacity": cache.capacity, "state": states[0], "state_unchanged_across_calls": unchanged,
              "output_bf16": output_bf16, "output_fp16": output_fp16,
              "passed": bool(output_bf16["passed"] and output_fp16["passed"] and unchanged
                             and observed["output_is_cache_buffer"])}
    return record, {name: [(batch, call) for call in run["calls"]] for name, run in runs.items()}


def _growing_case(ctx: Any, method: Any, implementations: tuple, batch: int, prefix: int, steps: int,
                  seed: int) -> tuple[dict[str, Any], dict[str, list]]:
    """Two identically built caches, one per implementation; exercises commit and rollover."""
    key, value, query = _inputs(ctx, batch, prefix + steps, steps, seed)
    runs: dict[str, dict[str, Any]] = {}
    for name, function in implementations:
        cache = _prefilled_cache(ctx, method, batch, max(prefix + steps, MIN_EQUIVALENCE_CAPACITY),
                                 key, value, prefix)
        cache.prepare_growing(prefix, steps)
        step_records, calls = [], []
        for step in range(steps):
            cache.select_growing_step(step)
            before = _counts(cache)
            token = prefix + step
            run = _decode_once(ctx, method, cache, function, key[:, :, token:token + 1, :],
                               value[:, :, token:token + 1, :], query[:, :, step:step + 1, :], token)
            calls.extend((batch, call) for call in run["calls"])
            step_records.append({"counts_before": before, **run})
        final = {"history_checksum_layer0": cache.history_checksum(0),
                 "token_state": {k: v.tolist() for k, v in cache.token_index_state(0).items()},
                 **_counts(cache)}
        runs[name] = {"steps": step_records, "final": final, "calls": calls}
        del cache
    reference, observed = runs[implementations[0][0]], runs[implementations[1][0]]
    step_out = []
    for step in range(steps):
        ref, obs = reference["steps"][step], observed["steps"][step]
        step_out.append({"step": step, "counts_before": ref["counts_before"],
                         "counts_match": ref["counts_before"] == obs["counts_before"],
                         "output_bf16": _compare(ctx, obs["output"], ref["output"]),
                         "output_fp16": _compare(ctx, obs["output_fp16"], ref["output_fp16"])})
    final_equal = reference["final"] == observed["final"]
    passed = bool(final_equal and all(s["counts_match"] and s["output_bf16"]["passed"]
                                      and s["output_fp16"]["passed"] for s in step_out))
    record = {"kind": "growing", "batch_size": batch, "prefix_length": prefix, "steps": steps, "seed": seed,
              "step_comparisons": step_out, "final_cache_state_bitwise_equal": final_equal,
              "final_state": {k: v for k, v in reference["final"].items() if k != "token_state"},
              "passed": passed}
    return record, {name: run["calls"] for name, run in runs.items()}


def _summarize_operands(records: list, heuristic: Callable[[str, Any], bool]) -> dict[str, Any]:
    """K/V bmm operands: native-KV storage views, no broadcast, head axis <= 8."""
    rows: dict[tuple, dict[str, Any]] = {}
    for batch, call in records:
        operand = call["mat2"]
        shape, stride, alias = operand["shape"], operand["stride"], operand["alias"]
        checks = {
            "aliases_native_kv_storage": alias in KV_STORAGE_ROLES,
            "no_broadcast_stride": not any(size > 1 and step == 0 for size, step in zip(shape, stride)),
            "kv_head_axis_at_most_8": bool(len(shape) == 3 and shape[0] % batch == 0
                                           and shape[0] // batch <= NUM_KV_HEADS),
            "all_operands_cache_owned": all(part is not None and part["alias"] != "unaliased"
                                            for part in (call["input"], operand, call["out"])),
        }
        signature = (batch, alias, tuple(shape), tuple(stride))
        if signature not in rows:
            try:
                flagged: Any = bool(heuristic(alias, shape))
            except ValueError as error:
                flagged = f"invalid: {error}"
            rows[signature] = {"batch_size": batch, "role": KV_STORAGE_ROLES.get(alias, "unknown"),
                               "alias": alias, "shape": shape, "stride": stride, **checks,
                               "phase8_query_head_sized_kv_temporary_heuristic": flagged, "calls": 0}
        else:
            for name, value in checks.items():
                rows[signature][name] = rows[signature][name] and value
        rows[signature]["calls"] += 1
    listed = list(rows.values())
    passed = bool(listed) and all(row["aliases_native_kv_storage"] and row["no_broadcast_stride"]
                                  and row["kv_head_axis_at_most_8"] and row["all_operands_cache_owned"]
                                  for row in listed)
    return {"pass": passed, "bmm_calls": len(records), "distinct_kv_operands": listed}


def _kv_temporaries(calls: dict[str, list], temporary_shapes: dict[str, tuple] | None,
                    heuristic: Callable[[str, Any], bool]) -> dict[str, Any]:
    phase8 = None
    if temporary_shapes is not None:
        phase8 = {role: {"shape": list(shape), "query_head_sized_kv_temporary": bool(heuristic(role, shape))}
                  for role, shape in temporary_shapes.items()}
    variant = _summarize_operands(calls["variant"], heuristic)
    original = _summarize_operands(calls["original_per_head"], heuristic)
    passed = bool(phase8 is not None and not any(v["query_head_sized_kv_temporary"] for v in phase8.values())
                  and variant["pass"])
    return {
        "pass": passed,
        "criterion": ("_query_head_sized_kv_temporary is False for every Phase 8 temporary shape (the "
                      "audit's own input), and every K/V operand of every decode bmm is a zero-copy view of "
                      "cache-owned 8-KV-head storage (no stride-0 broadcast, leading axis <= B*8); all bmm "
                      "operands and outputs are cache-owned"),
        "phase8_temporary_shapes": phase8,
        "variant_kv_operands": variant,
        "original_kv_operands_diagnostic": original,
        "heuristic_note": ("phase8_query_head_sized_kv_temporary_heuristic flags any K/V-named 3-D shape with a "
                           "32 in axis 0 or 1; the V residual operand has 32 residual *tokens* (residual_length "
                           "= 32), so it is flagged for the original per-head code as well; diagnostic only"),
    }


def _source_tokens(active_source: str, hot_path_source: str) -> dict[str, Any]:
    active, hot = active_source.casefold(), hot_path_source.casefold()
    in_active = [token for token in SEMANTIC_FORBIDDEN_TOKENS if token in active]
    in_hot = [token for token in SEMANTIC_FORBIDDEN_TOKENS if token in hot]
    return {"pass": not in_active and not in_hot, "tokens": list(SEMANTIC_FORBIDDEN_TOKENS),
            "present_in_decode_compressed": in_active, "present_in_hot_path": in_hot,
            "decode_compressed_source_sha256": _text_sha256(active_source),
            "hot_path_source_sha256": _text_sha256(hot_path_source)}


def semantic_native_gqa(p8: Any, configuration: str, original: Any, active: Any,
                        temporary_shapes: dict[str, tuple] | None, hot_path_source: str) -> dict[str, Any]:
    import torch
    import kvbench.adapters.kivi as kivi_module
    from kvbench.runtime.kivi_admission import _query_head_sized_kv_temporary
    from kvbench.runtime.kivi_session import PHASE8_DECODE_ATOL, PHASE8_DECODE_RTOL
    from kvbench.runtime.numerical import compare_tensors_untimed

    ctx = SimpleNamespace(torch=torch, device=torch.device("cuda:0"), attention=SimpleNamespace(layer_idx=0),
                          scaling=1.0 / math.sqrt(128), adapter_class=kivi_module.KIVIMethodAdapter,
                          compare=compare_tensors_untimed, atol=PHASE8_DECODE_ATOL, rtol=PHASE8_DECODE_RTOL)
    method = p8._canonical_factory_method(configuration)
    method.prepare_runtime()
    implementations = (("original_per_head", original), ("variant", active))
    cases: list[dict[str, Any]] = []
    calls: dict[str, list] = {"original_per_head": [], "variant": []}
    for batch, prefix in EQUIVALENCE_FIXED:
        record, case_calls = _fixed_case(ctx, method, implementations, batch, prefix,
                                         seed=20261005 + 1000 * batch + prefix)
        cases.append(record)
        for name in calls:
            calls[name].extend(case_calls[name])
    for batch, prefix, steps in EQUIVALENCE_GROWING:
        record, case_calls = _growing_case(ctx, method, implementations, batch, prefix, steps,
                                           seed=20261012 + 1000 * batch + prefix)
        cases.append(record)
        for name in calls:
            calls[name].extend(case_calls[name])
    comparisons = ([(c["output_bf16"], c["output_fp16"]) for c in cases if c["kind"] == "fixed"]
                   + [(s["output_bf16"], s["output_fp16"]) for c in cases if c["kind"] == "growing"
                      for s in c["step_comparisons"]])
    fixed_states = [c["state"] for c in cases if c["kind"] == "fixed"]
    coverage = {
        "fixed_with_key_history_key_residual_and_value_residual": any(
            s["key_history"] > 0 and s["key_residual"] > 0 and s["value_residual"] > 0 for s in fixed_states),
        "fixed_without_history": any(s["key_history"] == 0 and s["value_history"] == 0 for s in fixed_states),
        "fixed_wrapped_value_ring": any(s["value_residual_head"] > 0 for s in fixed_states),
        "growing_key_group_flush": any(s["counts_before"]["key_residual"] == 31 for c in cases
                                       if c["kind"] == "growing" for s in c["step_comparisons"]),
        "batch_sizes": sorted({c["batch_size"] for c in cases}),
    }
    equivalence = {
        "pass": bool(all(c["passed"] for c in cases)
                     and coverage["fixed_with_key_history_key_residual_and_value_residual"]
                     and coverage["growing_key_group_flush"]),
        "reference": "captured original KIVIMethodAdapter._decode_compressed (per-query-head loops)",
        "atol": PHASE8_DECODE_ATOL, "rtol": PHASE8_DECODE_RTOL,
        "all_bitwise_equal_bf16": all(bf16["bitwise_equal"] for bf16, _ in comparisons),
        "all_bitwise_equal_fp16": all(fp16["bitwise_equal"] for _, fp16 in comparisons),
        "max_abs_diff_bf16": max(bf16["max_absolute_error"] for bf16, _ in comparisons),
        "max_abs_diff_fp16": max(fp16["max_absolute_error"] for _, fp16 in comparisons),
        "coverage": coverage, "cases": cases,
    }
    kv = _kv_temporaries(calls, temporary_shapes, _query_head_sized_kv_temporary)
    tokens = _source_tokens(inspect.getsource(active), hot_path_source)
    del method
    _release_cuda()
    return {"indexing_verified": bool(equivalence["pass"] and kv["pass"] and tokens["pass"]),
            "per_head_equivalence": equivalence, "kv_temporaries": kv, "source_tokens": tokens}


# ---------------------------------------------------------- model points

def _operation_brief(operation: dict[str, Any]) -> dict[str, Any]:
    criterion = operation.get("criterion", {})
    return {key: criterion.get(key) for key in (
        "passed", "failure_reasons", "allocation_event_count", "allocation_event_bytes",
        "expected_allocation_event_count", "unknown_allocation_count", "persistent_allocated_delta",
        "persistent_reserved_delta", "strict_graph_zero_events", "categories")}


def _model_point(p8: Any, loaded: Any, configuration: str, graph_mode: Any, role: str, name: str,
                 out: Path, state: dict[str, Any] | None) -> dict[str, Any]:
    """phase8 _execute_point up to session.admit, without the fixed-L timing runner."""
    import torch
    from kvbench.runtime.backend import forced_flash_execution
    from kvbench.runtime.kivi_session import build_kivi_endpoint_session, build_kivi_operation_keys
    from kvbench.schema import RunnerKind

    calls_before = None if state is None else state.get("decode_calls")
    canonical_method = p8._canonical_factory_method(configuration)
    del canonical_method
    keys = build_kivi_operation_keys(configuration=configuration, runner_kind=RunnerKind.FIXED_L,
                                     graph_mode=graph_mode, starting_context=PHASE8_CONTEXT, output_steps=1)
    offset = 10_000 + list(p8._BITS).index(configuration) * 20_000 + PHASE8_CONTEXT
    device = torch.device("cuda:0")
    prefix = p8._deterministic_ids(torch, length=PHASE8_CONTEXT, offset=offset, device=device)
    decode = p8._deterministic_ids(torch, length=1, offset=offset + PHASE8_CONTEXT + 257, device=device)
    with forced_flash_execution():
        session = build_kivi_endpoint_session(loaded=loaded, operation_keys=keys,
                                              prefix_input_ids=prefix, decode_input_ids=decode)
        launcher_probe = p8._capture_launcher_probe(session)
        observed, audit, evidence_files = p8._audit_session(session)
    for relative, payload in sorted(evidence_files.items()):
        _write_bytes(out / "allocation" / name / relative, payload)
    geometry = session.gqa_cache_geometry()
    accounting = p8._phase8_byte_accounting(session.cache, active_context=PHASE8_CONTEXT)
    breakdown_sum = sum(int(v) for v in session.method_byte_breakdown().values())
    post_warmup_stable = tuple(observed) == tuple(session._warmed_outputs)
    comparison = session.eager_graph_comparison
    point = {
        "schema_version": "kvbench-addendum-20261005-gate-point-1.0.0",
        "point_name": name, "role": role, "configuration": configuration,
        "runner_kind": RunnerKind.FIXED_L.value, "graph_mode": graph_mode.value, "batch_size": 1,
        "context_length": PHASE8_CONTEXT, "output_steps": 1, "capacity": PHASE8_CONTEXT + 1,
        "prefix_ids_rule": "phase8 _deterministic_ids, offset 10000 + 20000 * bits-index + L",
        "cache_layout_fingerprint": session.cache_layout_fingerprint(),
        "adapter_config_fingerprint": session.adapter_config_fingerprint,
        "allocation": audit, "launcher_probe": launcher_probe,
        "accounting": accounting.to_dict(), "byte_breakdown_sum": breakdown_sum,
        "allocated_bytes": accounting.allocated_bytes,
        "reciprocal_product_error": abs(accounting.rho_alloc * accounting.r_alloc - 1.0),
        "cache_pointers_stable": audit["pointers_stable"],
        "cache_pointers_stable_source": "_audit_session pointers_stable (fixed-L runner not executed)",
        "gqa_cache_geometry": geometry,
        "native_gqa": bool(geometry["native_kv_head_storage"] and not geometry["gqa_materialized"]),
        "post_warmup_outputs_stable": post_warmup_stable,
        "graph_evidence": session.graph_evidence,
        "eager_graph_comparison": None if comparison is None else comparison.to_dict(),
        "variant_decode_calls_delta": (None if state is None
                                       else state.get("decode_calls") - calls_before),
        "allocation_evidence_root": f"allocation/{name}",
        "r_hbm": None, "performance_claim_eligible": False, "timing_runner_executed": False,
    }
    point["passed"] = bool(
        launcher_probe["passed"] and audit["passed"] and post_warmup_stable and point["native_gqa"]
        and accounting.predicted_relative_error < 0.01 and breakdown_sum == accounting.allocated_bytes
        and point["reciprocal_product_error"] <= p8.RECIPROCAL_ABS_TOLERANCE)
    point["passed_scope"] = "phase8 _execute_point conditions that do not need the fixed-L timing runner"
    session = None
    return point


def model_points(p8: Any, configuration: str, companion: str | None, out: Path,
                 state: dict[str, Any] | None) -> dict[str, Any]:
    from kvbench.runtime.model_loader import load_frozen_model, validate_loaded_frozen_model_receipt
    from kvbench.schema import GraphMode, canonical_json_bytes, sha256_hex

    loaded = load_frozen_model()
    validate_loaded_frozen_model_receipt(loaded)
    model_identity = sha256_hex(canonical_json_bytes(loaded.identity.to_dict()))
    specs = [("target", configuration, GraphMode.EAGER), ("target", configuration, GraphMode.CUDA_GRAPH)]
    if companion is not None:
        specs.append(("companion_kernel_family", companion, GraphMode.EAGER))
    points, errors = [], {}
    for role, selected, mode in specs:
        name = f"{selected}-fixed_l-l{PHASE8_CONTEXT}-{mode.value}"
        try:
            points.append(_model_point(p8, loaded, selected, mode, role, name, out, state))
        except Exception as error:  # noqa: BLE001 - recorded; derivation sees the missing point
            errors[name] = _error_record(error)
        p8._release_cuda_objects()
    loaded = None
    p8._release_cuda_objects()
    return {"model_identity_sha256": model_identity, "specs": [[r, c, m.value] for r, c, m in specs],
            "points": points, "errors": errors}


# ---------------------------------------------------------------- derivation

def _point_brief(point: dict[str, Any]) -> dict[str, Any]:
    return {"point": point["point_name"], "role": point["role"], "graph_mode": point["graph_mode"],
            "launcher_probe_passed": point["launcher_probe"]["passed"],
            "kernel_families": sorted({r["kernel_family"] for r in point["launcher_probe"]["first_sequence"]}),
            "allocation_passed": point["allocation"]["passed"],
            "graph_passed": point["allocation"]["graph_passed"],
            "native_gqa": point["native_gqa"], "cache_pointers_stable": point["cache_pointers_stable"],
            "operations": [_operation_brief(op) for op in point["allocation"]["operation_allocations"]]}


def derive(p8: Any, *, modules: dict[str, Any], semantic: dict[str, Any] | None,
           model: dict[str, Any] | None, temporary_shapes: dict[str, tuple] | None) -> dict[str, Any]:
    if not model or not model["points"]:
        raise RuntimeError("no model point was built; the execution-path audit cannot be derived")
    if temporary_shapes is None:
        raise RuntimeError("Phase 8 temporary shapes are unavailable")
    points = model["points"]
    fixture_pass = bool(modules.get("fixture_conformance") and modules["fixture_conformance"]["passed"] is True)
    graph_pass = bool(modules.get("graph_harness") and modules["graph_harness"]["passed"] is True)
    probes = tuple(point["launcher_probe"] for point in points)
    fallback_observed = bool(not graph_pass or any(probe.get("passed") is not True for probe in probes))
    growth_observed = any(point["cache_pointers_stable"] is not True for point in points)
    literal_audit = p8._execution_path_audit(temporary_shapes=temporary_shapes, launcher_probes=probes,
                                             backend_fallback_observed=fallback_observed,
                                             cache_growth_observed=growth_observed)
    semantic_verified = bool(semantic is not None and semantic["indexing_verified"] is True
                             and not literal_audit.gqa_materialization_detected)
    reasons = tuple(r for r in literal_audit.reasons if r != NATIVE_GQA_REASON)
    if not semantic_verified:
        reasons += (NATIVE_GQA_REASON,)
    replaced_audit = dataclasses.replace(literal_audit, native_gqa_indexing_verified=semantic_verified,
                                         reasons=reasons, passed=not reasons)
    own = tuple(point["launcher_probe"] for point in points if point["role"] == "target")
    try:
        own_audit: Any = list(p8._execution_path_audit(
            temporary_shapes=temporary_shapes, launcher_probes=own,
            backend_fallback_observed=bool(not graph_pass or any(p.get("passed") is not True for p in own)),
            cache_growth_observed=growth_observed).reasons)
    except Exception as error:  # noqa: BLE001 - diagnostic only
        own_audit = _error_record(error)
    inputs = {"git_sha": common.EXECUTION_SHA, "fixture": {"passed": fixture_pass}, "graph": {"passed": graph_pass},
              "sanitizer": {"passed": None, "not_evaluated": "separate driver check"},
              "point_records": points, "run_ids": [point["point_name"] for point in points]}
    literal = p8._derive_local_candidate(execution_path=literal_audit, **inputs)
    replaced = p8._derive_local_candidate(execution_path=replaced_audit, **inputs)

    briefs = [_point_brief(point) for point in points]
    graph_briefs = [b for b in briefs if b["graph_mode"] == "cuda_graph"]
    path = {"execution_path_passed_literal": literal_audit.passed,
            "execution_path_reasons_literal": list(literal_audit.reasons),
            "execution_path_passed_native_gqa_replaced": replaced_audit.passed,
            "execution_path_reasons_native_gqa_replaced": list(replaced_audit.reasons)}
    details = {
        "no_measured_torch_cat": {**path, "measured_torch_cat_detected": literal_audit.measured_torch_cat_detected},
        "direct_compressed_decode": {
            **path, "two_bit_kernel_verified": literal_audit.two_bit_kernel_verified,
            "four_bit_kernel_verified": literal_audit.four_bit_kernel_verified,
            "full_prefix_dequantization_detected": literal_audit.full_prefix_dequantization_detected,
            "full_prefix_temporary_detected": literal_audit.full_prefix_temporary_detected,
            "kernel_families_by_point": {b["point"]: b["kernel_families"] for b in briefs}},
        "native_gqa": {
            "native_gqa_indexing_verified_literal": literal_audit.native_gqa_indexing_verified,
            "native_gqa_indexing_verified_semantic": replaced_audit.native_gqa_indexing_verified,
            "gqa_materialization_detected": literal_audit.gqa_materialization_detected,
            "query_head_sized_kv_temporary_detected": literal_audit.query_head_sized_kv_temporary_detected,
            "point_native_gqa": {b["point"]: b["native_gqa"] for b in briefs}},
        "no_unknown_allocation": {"operations_by_point": {b["point"]: b["operations"] for b in briefs}},
        "no_backend_fallback": {
            **path, "backend_fallback_detected": literal_audit.backend_fallback_detected,
            "backend_fallback_observed_input": fallback_observed, "graph_harness_passed": graph_pass,
            "launcher_probes_passed": {b["point"]: b["launcher_probe_passed"] for b in briefs}},
        "graph_capture_replay": {"graph_harness_passed": graph_pass, "graph_points": [
            {"point": b["point"], "graph_passed": b["graph_passed"]} for b in graph_briefs]},
        "graph_zero_replay_allocation": {"graph_points": [
            {"point": b["point"], "strict_graph_zero_events": [op["strict_graph_zero_events"]
                                                               for op in b["operations"]]}
            for b in graph_briefs]},
    }

    def item(name: str) -> dict[str, Any]:
        record = {"pass_literal": literal[name], "detail": details[name], "derived_with": DERIVED_WITH[name]}
        if name == "native_gqa":
            return {"pass": literal[name], **record, "pass_semantic": replaced[name],
                    "pass_basis": "literal Phase 8 derivation (source-string criterion); see pass_semantic"}
        return {"pass": replaced[name], **record,
                "pass_basis": ("Phase 8 derivation with the audit's native_gqa_indexing_unverified reason "
                               "decided by the semantic replacement instead of the source string"),
                "literal_fails_only_through_native_gqa_source_string": bool(
                    literal[name] is not True and replaced[name] is True)}

    return {
        "items": {"G3": {name: item(name) for name in G3_ITEMS}, "G4": {name: item(name) for name in G4_ITEMS}},
        "execution_path_audit_literal": literal_audit.to_dict(),
        "execution_path_audit_native_gqa_replaced": replaced_audit.to_dict(),
        "execution_path_reasons_target_points_only_diagnostic": own_audit,
        "native_gqa_literal": {
            "pass": literal["native_gqa"],
            "source_token": LITERAL_NATIVE_GQA_TOKEN,
            "audit_native_gqa_indexing_verified": literal_audit.native_gqa_indexing_verified,
        },
        "native_gqa_semantic_phase8_composition": replaced["native_gqa"],
        "inputs": {"fixture_passed": fixture_pass, "graph_passed": graph_pass,
                   "backend_fallback_observed": fallback_observed, "cache_growth_observed": growth_observed,
                   "launcher_probe_points": [point["point_name"] for point in points],
                   "temporary_shapes": {k: list(v) for k, v in temporary_shapes.items()}},
    }


# ---------------------------------------------------------------- worker

def _environment() -> dict[str, Any]:
    import kvbench.adapters.kivi as kivi_module
    from kvbench.runtime.kivi_admission import require_authorized_kivi_environment
    from kvbench.schema.phase8 import PHASE8_AUTHORIZED_CONTAINER_DIGEST

    identity = require_authorized_kivi_environment(PHASE8_AUTHORIZED_CONTAINER_DIGEST)
    spec = importlib.util.find_spec("kivi_gemv")
    extension = Path(spec.origin).resolve() if spec is not None and spec.origin else None
    digest = None if extension is None else common.sha256_file(extension)
    return {"identity": identity, "dockerenv": Path("/.dockerenv").is_file(),
            "kivi_extension_path": None if extension is None else str(extension),
            "kivi_extension_sha256": digest,
            "kivi_extension_matches_frozen": digest == kivi_module.KIVI_EXTENSION_SHA256 == common.KIVI_EXTENSION_SHA}


def _first_identity(p8: Any) -> dict[str, Any]:
    first = p8.PHASE8_ADMISSION_GRID[0]
    method, cache, layout, fingerprint, _ = p8._preallocate_first_identity()
    shapes = p8._hot_path_temporary_shapes(cache)
    del method, cache
    return {"grid_point": [first.configuration, first.runner_kind.value, first.graph_mode.value,
                           first.context_length, first.output_steps],
            "cache_layout_fingerprint": layout, "method_fingerprint": fingerprint,
            "temporary_shapes": {role: [int(v) for v in shape] for role, shape in shapes.items()}}


def run(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    variant = json.loads(args.variant)
    if variant not in ALLOWED_VARIANTS:
        raise RuntimeError(f"gate worker accepts only {ALLOWED_VARIANTS}, got {variant}")
    companion = AUTO_COMPANION[args.configuration] if args.companion_configuration == "auto" else (
        None if args.companion_configuration == "none" else args.companion_configuration)
    if companion == args.configuration:
        raise RuntimeError("companion configuration must differ from the configuration")

    import kvbench.adapters.kivi as kivi_module

    adapter_class = kivi_module.KIVIMethodAdapter
    original = adapter_class.__dict__["_decode_compressed"]  # captured before installation
    original_source = inspect.getsource(original)
    state = install_variant(variant)
    active = adapter_class.__dict__["_decode_compressed"]
    active_source = inspect.getsource(adapter_class._decode_compressed)
    import scripts.phase8_kivi_admission as p8
    from kvbench.runtime.kivi_admission import kivi_adapter_hot_path_source

    hot_path = kivi_adapter_hot_path_source()
    installed = (active is original) if not variant else (
        active is not original and state is not None
        and _text_sha256(active_source) == state.get("override_source_sha256")
        and state.get("replaced_source_sha256") == _text_sha256(original_source))
    installation = {
        "variant_installed": active is not original, "installed_as_requested": bool(installed),
        "original_source_sha256": _text_sha256(original_source),
        "active_source_sha256": _text_sha256(active_source),
        "hot_path_contains_active_source": active_source in hot_path,
        "hot_path_source_sha256": _text_sha256(hot_path),
        "literal_native_gqa_token_in_hot_path": LITERAL_NATIVE_GQA_TOKEN in hot_path.casefold(),
    }
    common.write_new(out / "sources" / "original_decode_compressed.py", original_source)
    common.write_new(out / "sources" / "active_decode_compressed.py", active_source)
    common.write_new(out / "sources" / "hot_path_source.py", hot_path)

    stages = StageLog(out)
    environment = stages.run("environment", _environment)
    identity = stages.run("phase8_first_identity", lambda: _first_identity(p8))
    _release_cuda()
    temporary_shapes = None if identity is None else {
        role: tuple(shape) for role, shape in identity["temporary_shapes"].items()}
    static = stages.run("static_execution_path_precheck", p8._static_execution_path_precheck)
    modules = {label: stages.run(f"unittest_{label}", lambda label=label, rel=rel: run_unittest_module(label, rel))
               for label, rel in TEST_MODULES}
    semantic = stages.run("semantic_native_gqa", lambda: semantic_native_gqa(
        p8, args.configuration, original, active, temporary_shapes, hot_path))
    model = stages.run("model_points", lambda: model_points(p8, args.configuration, companion, out, state))
    if model is not None and model["errors"]:
        stages.errors["model_points"] = {"stage": "model_points", "status": "point_failed",
                                         "error": json.dumps(sorted(model["errors"]))}
    derived = stages.run("derive_g3_g4", lambda: derive(p8, modules=modules, semantic=semantic, model=model,
                                                        temporary_shapes=temporary_shapes))

    items = None if derived is None else derived["items"]

    def item_pass(gate: str, name: str, key: str = "pass") -> bool:
        return bool(items is not None and items[gate][name][key] is True)

    semantic_pass = bool(semantic is not None and semantic["indexing_verified"] is True
                         and item_pass("G3", "native_gqa", "pass_semantic"))
    components = {
        "variant_installed_as_requested": bool(installed),
        "static_execution_path_precheck": bool(static is not None and static.get("passed") is True),
        **{f"G3.{name}": item_pass("G3", name) for name in G3_ITEMS if name != "native_gqa"},
        "G3.native_gqa_semantic_replacement": semantic_pass,
        **{f"G4.{name}": item_pass("G4", name) for name in G4_ITEMS},
        **{f"unittest.{label}": bool(modules[label] is not None and modules[label]["passed"] is True)
           for label, _ in TEST_MODULES},
        "no_stage_errors": not stages.errors,
    }
    import torch

    return {
        "schema_version": SCHEMA,
        "status": "completed" if not stages.errors else "stage_failed",
        "error": None if not stages.errors else f"stage errors: {sorted(stages.errors)}",
        "mode": "gate", "configuration": args.configuration, "companion_configuration": companion,
        "variant": variant, "variant_state": overrides.describe(state), "variant_installation": installation,
        "gates_pass": all(components.values()), "gates_pass_components": components,
        "items": items,
        "native_gqa_literal": None if derived is None else derived["native_gqa_literal"],
        "native_gqa_semantic": None if semantic is None else {
            "pass": semantic_pass,
            "phase8_composition_pass": None if derived is None else derived["native_gqa_semantic_phase8_composition"],
            "definition": ("(a) per-head equivalence with the captured original at PHASE8_DECODE_ATOL/RTOL "
                           "(fixed and growing, residual tokens present) AND (b) no K/V temporary with a "
                           "32-head axis AND (c) no forbidden source token; substituted for the source-string "
                           "test inside Phase 8's native_gqa composition (with gqa materialization, "
                           "_query_head_sized_kv_temporary and per-point cache geometry)"),
            **semantic},
        "unittest_modules": modules,
        "static_execution_path_precheck": static,
        "environment": environment,
        "phase8_first_identity": identity,
        "model": model,
        "derivation": None if derived is None else {k: v for k, v in derived.items() if k != "items"},
        "stage_errors": stages.errors,
        "not_reproduced": NOT_REPRODUCED,
        "execution_git_sha": common.EXECUTION_SHA,
        "addendum_git_sha": args.addendum_commit,
        "container_digest": os.environ.get("KVBENCH_AUTHORIZED_IMAGE_DIGEST"),
        "expected_container_digest": common.IMAGE,
        "worker_sha256": common.sha256_file(Path(__file__)),
        "overrides_sha256": common.sha256_file(Path(overrides.__file__)),
        "timing_worker_sha256": common.sha256_file(Path(timing_worker.__file__)),
        "common_sha256": common.sha256_file(Path(common.__file__)),
        "frozen_sources_sha256": {rel: _file_sha256(REPO / rel) for rel in PROVENANCE_SOURCES},
        "torch_version": torch.__version__,
        "performance_timing_collected": False,
        "profiler_used": False,
        "finished_at_utc": common.utc_now(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", required=True, choices=CONFIGURATIONS)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--companion-configuration", default="auto",
                        choices=("auto", "none", "k4v4", "k2v4", "k2v2"))
    parser.add_argument("--addendum-commit", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    code = 0
    try:
        result = run(args, out)
        code = 0 if result["status"] == "completed" else 4
        text = json.dumps(result, sort_keys=True, default=str)
    except BaseException as error:  # noqa: BLE001 - the driver records the failure
        result = {"schema_version": SCHEMA, "status": "failed", "error_type": type(error).__name__,
                  "error": str(error), "traceback": traceback.format_exc(),
                  "performance_timing_collected": False, "finished_at_utc": common.utc_now()}
        code = 3
        text = json.dumps(result, sort_keys=True, default=str)
    common.write_new(out / "worker_result.json", text + "\n")
    sys.stdout.write(RESULT_PREFIX + json.dumps({k: result.get(k) for k in (
        "status", "configuration", "variant", "gates_pass", "gates_pass_components", "error")},
        sort_keys=True, default=str) + "\n")
    sys.stdout.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
