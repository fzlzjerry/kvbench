#!/usr/bin/env python3
"""Bounded, quality-only diagnosis for the Q0 BF16 cross-batch failure."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from scripts import q0_cache_sensitive_correctness as q0


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IDS = (0, 1, 2)
BOUNDARIES = (
    "embedding",
    "layer00_input",
    "layer00_q_proj",
    "layer00_k_proj",
    "layer00_v_proj",
    "layer00_attention_output",
    "layer01_input",
    "layer16_input",
    "layer31_input",
    "final_norm",
    "logits",
)


class DiagnosisError(RuntimeError):
    """The bounded diagnosis failed closed."""


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_carrier(sample_tokens: Sequence[int], batch: int) -> tuple[list[int], dict[int, list[int]]]:
    """Build one explicit equal-length carrier and its logical-row mapping."""

    if batch <= 0 or not sample_tokens or len(sample_tokens) > batch:
        raise DiagnosisError("carrier geometry is invalid")
    carrier = [int(sample_tokens[index % len(sample_tokens)]) for index in range(batch)]
    mapping: dict[int, list[int]] = {index: [] for index in range(len(sample_tokens))}
    for row in range(batch):
        mapping[row % len(sample_tokens)].append(row)
    return carrier, mapping


def permuted_carrier(sample_tokens: Sequence[int]) -> tuple[list[int], dict[int, int]]:
    """Return the frozen B=4 permutation control for three diagnosis samples."""

    if len(sample_tokens) != 3:
        raise DiagnosisError("permutation control requires three tokens")
    logical_ids = (2, 0, 0, 1)
    return [int(sample_tokens[index]) for index in logical_ids], {0: 1, 1: 3, 2: 0}


def vector_metrics(reference: Any, observed: Any, *, atol: float, rtol: float) -> dict[str, Any]:
    """Calculate diagnostics without changing the frozen allclose predicate."""

    import torch
    import torch.nn.functional as functional

    left = reference.detach().to(device="cpu", dtype=torch.float64, copy=True).reshape(-1)
    right = observed.detach().to(device="cpu", dtype=torch.float64, copy=True).reshape(-1)
    if left.shape != right.shape or not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
        raise DiagnosisError("aligned finite vectors are required")
    delta = (right - left).abs()
    threshold = atol + rtol * left.abs()
    normalized = delta / threshold
    violations = delta > threshold
    ref_values, ref_ids = torch.topk(left, 2)
    obs_values, obs_ids = torch.topk(right, 2)
    ref_logp = functional.log_softmax(left, dim=-1)
    obs_logp = functional.log_softmax(right, dim=-1)
    kl = torch.sum(ref_logp.exp() * (ref_logp - obs_logp))
    epsilon = float(delta.max().item())
    margin = float((ref_values[0] - ref_values[1]).item())
    return {
        "predicate": "abs(z_batch-z_b1) <= atol + rtol*abs(z_b1)",
        "atol": atol,
        "rtol": rtol,
        "passed": bool(torch.all(delta <= threshold)),
        "element_count": int(left.numel()),
        "violating_element_count": int(violations.sum().item()),
        "violating_element_fraction": float(violations.double().mean().item()),
        "abs_error_max": epsilon,
        "abs_error_median": float(torch.quantile(delta, 0.5).item()),
        "abs_error_p95": float(torch.quantile(delta, 0.95).item()),
        "normalized_error_max": float(normalized.max().item()),
        "normalized_error_median": float(torch.quantile(normalized, 0.5).item()),
        "normalized_error_p95": float(torch.quantile(normalized, 0.95).item()),
        "kl_b1_to_batch": float(kl.item()),
        "b1_top1_id": int(ref_ids[0].item()),
        "b1_top2_id": int(ref_ids[1].item()),
        "b1_top1_score": float(ref_values[0].item()),
        "b1_top2_score": float(ref_values[1].item()),
        "batch_top1_id": int(obs_ids[0].item()),
        "batch_top2_id": int(obs_ids[1].item()),
        "batch_top1_score": float(obs_values[0].item()),
        "batch_top2_score": float(obs_values[1].item()),
        "selected_token_agrees": int(ref_ids[0].item()) == int(obs_ids[0].item()),
        "b1_top_two_margin": margin,
        "epsilon": epsilon,
        "argmax_preservation_bound_satisfied": margin > 2.0 * epsilon,
    }


def boundary_metrics(reference: Mapping[str, Any], observed: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    import torch

    rows: list[dict[str, Any]] = []
    first: str | None = None
    for name in BOUNDARIES:
        left = reference[name].detach().to(device="cpu", dtype=torch.float32, copy=True)
        right = observed[name].detach().to(device="cpu", dtype=torch.float32, copy=True)
        if left.shape != right.shape:
            raise DiagnosisError(f"boundary shape differs: {name}")
        delta = (right - left).abs()
        exact = bool(torch.equal(left, right))
        if not exact and first is None:
            first = name
        rows.append({
            "boundary": name,
            "shape": list(left.shape),
            "exact": exact,
            "max_abs_error": float(delta.max().item()),
            "mean_abs_error": float(delta.mean().item()),
        })
    return rows, first


def _capture_hooks(model: Any, captures: dict[str, list[Any]]) -> ExitStack:
    import torch

    stack = ExitStack()

    def store(name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            if isinstance(tensor, torch.Tensor) and tensor.ndim >= 2 and int(tensor.shape[1]) == 1:
                captures[name].append(tensor.detach().to(device="cpu", dtype=torch.float32, copy=True).squeeze(1))
        return hook

    def store_input(name: str):
        def hook(_module: Any, inputs: Any) -> None:
            tensor = inputs[0]
            if isinstance(tensor, torch.Tensor) and tensor.ndim >= 2 and int(tensor.shape[1]) == 1:
                captures[name].append(tensor.detach().to(device="cpu", dtype=torch.float32, copy=True).squeeze(1))
        return hook

    modules = model.model
    stack.callback(modules.embed_tokens.register_forward_hook(store("embedding")).remove)
    for index in (0, 1, 16, 31):
        stack.callback(modules.layers[index].input_layernorm.register_forward_pre_hook(store_input(f"layer{index:02d}_input")).remove)
    layer0 = modules.layers[0]
    stack.callback(layer0.self_attn.q_proj.register_forward_hook(store("layer00_q_proj")).remove)
    stack.callback(layer0.self_attn.k_proj.register_forward_hook(store("layer00_k_proj")).remove)
    stack.callback(layer0.self_attn.v_proj.register_forward_hook(store("layer00_v_proj")).remove)
    stack.callback(layer0.self_attn.o_proj.register_forward_hook(store("layer00_attention_output")).remove)
    stack.callback(modules.norm.register_forward_hook(store("final_norm")).remove)
    stack.callback(model.lm_head.register_forward_hook(store("logits")).remove)
    return stack


def run_fixed_capture(loaded: Any, prefix: Sequence[int], groups: Sequence[Sequence[int]]) -> tuple[list[Any], dict[str, Any], list[dict[str, Any]]]:
    batch = len(groups[0])
    if any(len(group) != batch for group in groups):
        raise DiagnosisError("token group batches differ")
    captures: dict[str, list[Any]] = {name: [] for name in BOUNDARIES}
    with _capture_hooks(loaded.model, captures):
        outputs, evidence = q0.fixed_outputs(
            loaded,
            "bf16",
            [list(prefix) for _ in range(batch)],
            groups,
            graph=False,
        )
    count = len(groups)
    if any(len(captures[name]) != count for name in BOUNDARIES):
        observed = {name: len(values) for name, values in captures.items()}
        raise DiagnosisError(f"boundary capture cardinality differs: {observed}")
    per_group = [{name: captures[name][index] for name in BOUNDARIES} for index in range(count)]
    return outputs, evidence, per_group


def _existing_batch_summary() -> list[dict[str, Any]]:
    closure = ROOT / "artifacts/q0/q0-20260919t054207209666z-b85111f1-27c21357-closure-5b143f1"
    rows: list[dict[str, Any]] = []
    for configuration in q0.CONFIGS:
        primary = closure / "units" / configuration / "batch-invariance/result.json"
        continuation = closure / "continuations/kvquant-q0-wrapper-fix/units" / configuration / "batch-invariance/result.json"
        path = continuation if continuation.exists() else primary
        result = q0.load_json(path)
        for batch in (4, 8):
            item = result["results"][str(batch)]
            rows.append({
                "configuration": configuration,
                "batch": batch,
                "logical_sample_denominator": int(item["logical_sample_count"]),
                "logit_closeness_status": "FAIL" if not item["passed"] else "PASS",
                "maximum_abs_error": float(item["maximum_abs_error"]),
                "selected_token_agreement_numerator": int(item["selected_token_agreement_count"]),
                "selected_token_agreement_denominator": int(item["logical_sample_count"]),
                "teacher_forced_step_agreement": None,
                "generated_sequence_agreement": None,
                "parsed_task_answer_agreement": None,
                "per_sample_logits_available": False,
                "source": path.relative_to(ROOT).as_posix(),
            })
    return rows


def run_bf16(output_dir: Path, execution_head: str) -> dict[str, Any]:
    if output_dir.exists():
        raise DiagnosisError("diagnosis output already exists")
    if os.environ.get("KVBENCH_QUALITY_IMAGE_DIGEST") != q0.QUALITY_IMAGE:
        raise DiagnosisError("Quality image identity differs")
    observed_head = subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    if observed_head != execution_head:
        raise DiagnosisError("diagnosis execution HEAD differs")
    q0.verify_approval()
    protected = q0.verify_locked_paths()

    import torch
    from kvbench.runtime.backend import forced_flash_execution
    from kvbench.runtime.model_loader import load_frozen_model
    from safetensors.torch import save_file

    plan = q0.frozen_input_plan()
    prefix = next(item["input_ids"] for item in plan["core_probes"] if item["length"] == 512)
    all_tokens = [int(value) for value in plan["invariance"]["tokens"]]
    tokens = [all_tokens[index] for index in SAMPLE_IDS]
    prefix_sha = q0.sha256_bytes(q0._i32_bytes(prefix))
    atol, rtol = q0.frozen_tolerance("bf16")
    loaded = load_frozen_model(device=torch.device("cuda:0"))

    with torch.inference_mode(), forced_flash_execution():
        b1_a, b1_evidence_a, b1_capture_a = run_fixed_capture(loaded, prefix, [[token] for token in tokens])
        b1_b, b1_evidence_b, _ = run_fixed_capture(loaded, prefix, [[token] for token in tokens])
        duplicate4, duplicate4_evidence, duplicate4_capture = run_fixed_capture(loaded, prefix, [[tokens[0]] * 4])
        duplicate8, duplicate8_evidence, duplicate8_capture = run_fixed_capture(loaded, prefix, [[tokens[0]] * 8])
        carrier4, mapping4 = build_carrier(tokens, 4)
        mixed4, mixed4_evidence, mixed4_capture = run_fixed_capture(loaded, prefix, [carrier4])
        carrier8, mapping8 = build_carrier(tokens, 8)
        mixed8, mixed8_evidence, mixed8_capture = run_fixed_capture(loaded, prefix, [carrier8])
        permuted, permutation_rows = permuted_carrier(tokens)
        permuted4, permuted4_evidence, permuted4_capture = run_fixed_capture(loaded, prefix, [permuted])

    b1_repeat = [vector_metrics(b1_a[index][0], b1_b[index][0], atol=atol, rtol=rtol) for index in range(3)]
    samples: list[dict[str, Any]] = []
    tensors: dict[str, Any] = {}
    for logical_index, sample_id in enumerate(SAMPLE_IDS):
        reference = b1_a[logical_index][0]
        row4 = mapping4[logical_index][0]
        row8 = mapping8[logical_index][0]
        comparison4 = vector_metrics(reference, mixed4[0][row4], atol=atol, rtol=rtol)
        comparison8 = vector_metrics(reference, mixed8[0][row8], atol=atol, rtol=rtol)
        boundary4, first4 = boundary_metrics(
            {name: b1_capture_a[logical_index][name][0] for name in BOUNDARIES},
            {name: mixed4_capture[0][name][row4] for name in BOUNDARIES},
        )
        boundary8, first8 = boundary_metrics(
            {name: b1_capture_a[logical_index][name][0] for name in BOUNDARIES},
            {name: mixed8_capture[0][name][row8] for name in BOUNDARIES},
        )
        samples.append({
            "configuration": "bf16",
            "sample_id": sample_id,
            "token_id": tokens[logical_index],
            "prefix_sha256": prefix_sha,
            "input_identity_matches": True,
            "historical_prefix_length": 512,
            "effective_attended_length": 513,
            "decode_step": 0,
            "mode": "eager",
            "b1_repeat": b1_repeat[logical_index],
            "b4": {"row": row4, "metrics": comparison4, "first_divergent_boundary": first4, "boundaries": boundary4},
            "b8": {"row": row8, "metrics": comparison8, "first_divergent_boundary": first8, "boundaries": boundary8},
            "first_divergent_decode_position": 0 if not (comparison4["passed"] and comparison8["passed"]) else None,
            "conditioning_histories_identical": True,
            "selected_token_semantics": "unrestricted_full_vocabulary_argmax",
        })
        tensors[f"sample_{sample_id}_b1"] = reference.contiguous()
        tensors[f"sample_{sample_id}_b4"] = mixed4[0][row4].contiguous()
        tensors[f"sample_{sample_id}_b8"] = mixed8[0][row8].contiguous()

    duplicate_controls: dict[str, Any] = {}
    for batch, output, capture in ((4, duplicate4[0], duplicate4_capture[0]), (8, duplicate8[0], duplicate8_capture[0])):
        duplicate_controls[str(batch)] = {
            "all_rows_logits_exact": all(bool(torch.equal(output[0], output[row])) for row in range(1, batch)),
            "all_rows_boundaries_exact": all(
                bool(torch.equal(capture[name][0], capture[name][row]))
                for name in BOUNDARIES
                for row in range(1, batch)
            ),
            "row_metrics_against_b1": [vector_metrics(b1_a[0][0], output[row], atol=atol, rtol=rtol) for row in range(batch)],
        }

    permutation_controls: list[dict[str, Any]] = []
    for logical_index, sample_id in enumerate(SAMPLE_IDS):
        original_row = mapping4[logical_index][0]
        permuted_row = permutation_rows[logical_index]
        permutation_controls.append({
            "sample_id": sample_id,
            "original_row": original_row,
            "permuted_row": permuted_row,
            "logits_exact_after_row_restore": bool(torch.equal(mixed4[0][original_row], permuted4[0][permuted_row])),
            "all_boundaries_exact_after_row_restore": all(
                bool(torch.equal(mixed4_capture[0][name][original_row], permuted4_capture[0][name][permuted_row]))
                for name in BOUNDARIES
            ),
        })

    result = {
        "schema_version": "kvbench-q0-batch-diagnosis-1.0.0",
        "status": "COMPLETE_GATE_REMAINS_FAILED",
        "execution_head": execution_head,
        "quality_image_digest": q0.QUALITY_IMAGE,
        "contract_id": q0.CONTRACT_ID,
        "contract_sha256": q0.CONTRACT_SHA256,
        "configuration": "bf16",
        "sample_selection": {"kind": "lowest_frozen_sample_ids", "sample_ids": list(SAMPLE_IDS), "diagnostic_only": True},
        "frozen_tolerance": {"atol": atol, "rtol": rtol},
        "protected_paths": protected,
        "input_mapping": {
            "equal_length": True,
            "prefix_sha256": prefix_sha,
            "tokens": tokens,
            "b4_carrier": carrier4,
            "b4_mapping": mapping4,
            "b8_carrier": carrier8,
            "b8_mapping": mapping8,
            "permuted_b4_carrier": permuted,
            "permuted_b4_rows": permutation_rows,
            "padding_or_masking_used": False,
        },
        "evidence": {
            "b1_first": b1_evidence_a,
            "b1_second": b1_evidence_b,
            "duplicate_b4": duplicate4_evidence,
            "duplicate_b8": duplicate8_evidence,
            "mixed_b4": mixed4_evidence,
            "mixed_b8": mixed8_evidence,
            "permuted_b4": permuted4_evidence,
        },
        "samples": samples,
        "duplicate_controls": duplicate_controls,
        "permutation_controls": permutation_controls,
        "existing_q0_batch_summary": _existing_batch_summary(),
        "full_vocabulary_logits_file": "bf16_logits.safetensors",
        "normal_performance_timing": False,
        "fast_ppl_started": False,
    }
    output_dir.mkdir(parents=True)
    save_file(tensors, output_dir / "bf16_logits.safetensors")
    (output_dir / "diagnosis.json").write_bytes(canonical_bytes(result))
    checksums = []
    for path in sorted(output_dir.iterdir()):
        checksums.append(f"{q0.sha256_file(path)}  {path.name}\n")
    (output_dir / "checksums.sha256").write_text("".join(checksums), encoding="utf-8")
    complete = {
        "schema_version": "kvbench-q0-batch-diagnosis-complete-1.0.0",
        "status": result["status"],
        "execution_head": execution_head,
        "diagnosis_sha256": q0.sha256_file(output_dir / "diagnosis.json"),
        "checksums_sha256": q0.sha256_file(output_dir / "checksums.sha256"),
        "written_last": True,
    }
    (output_dir / "COMPLETE").write_bytes(canonical_bytes(complete))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-bf16", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-head", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.run_bf16:
        raise DiagnosisError("--run-bf16 is required")
    result = run_bf16(args.output_dir, args.execution_head)
    print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
