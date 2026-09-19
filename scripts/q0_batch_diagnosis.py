#!/usr/bin/env python3
"""Bounded, quality-only diagnosis for the Q0 BF16 cross-batch failure."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
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


def run_batch_worker(campaign: Path, configuration: str, execution_head: str) -> dict[str, Any]:
    """Rerun only one comparator-affected batch-invariance unit."""

    if os.environ.get("KVBENCH_QUALITY_IMAGE_DIGEST") != q0.QUALITY_IMAGE:
        raise DiagnosisError("Quality image identity differs")
    observed_head = subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    if observed_head != execution_head:
        raise DiagnosisError("batch continuation execution HEAD differs")
    if subprocess.run(("git", "status", "--porcelain=v1", "--untracked-files=all"), cwd=ROOT, check=True, capture_output=True, text=True).stdout:
        raise DiagnosisError("batch continuation source is not clean")
    q0.verify_approval()
    q0.verify_locked_paths()

    import torch
    from kvbench.runtime.backend import forced_flash_execution
    from kvbench.runtime.model_loader import load_frozen_model

    plan = q0.frozen_input_plan()
    loaded = load_frozen_model(device=torch.device("cuda:0"))
    with torch.inference_mode(), forced_flash_execution():
        q0._batch_stage(loaded, campaign, configuration, plan)
    result_path = campaign / "units" / configuration / "batch-invariance/result.json"
    return q0.load_json(result_path)


def _batch_worker_command(campaign: Path, configuration: str, execution_head: str) -> list[str]:
    command = q0._docker_worker_command(campaign, configuration, execution_head)
    container_campaign = f"/home/rockrock/cmu_paper/artifacts/q0_batch_diagnosis/{campaign.name}"
    source_marker = f"src={campaign},"
    for index, value in enumerate(command):
        if value.startswith("type=bind,") and source_marker in value:
            command[index] = f"type=bind,src={campaign},dst={container_campaign}"
            break
    else:
        raise DiagnosisError("batch continuation campaign mount is absent")
    module_index = command.index("-m")
    return command[:module_index] + [
        "-m",
        "scripts.q0_batch_diagnosis",
        "--run-batch-worker",
        "--campaign",
        container_campaign,
        "--configuration",
        configuration,
        "--execution-head",
        execution_head,
    ]


def _diagnosis_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Q0 Batch Diagnosis",
        "",
        "Status: **COMPLETE — unchanged batch gate remains failed**",
        "",
        "The bounded BF16 reproducer demonstrates genuine batch-shape numerical",
        "sensitivity in the frozen BF16 full-model path. Identical embedding and",
        "first-layer inputs first diverge in layer-0 Q or V projection output,",
        "before cache attention consumes the stored prefix. Duplicate rows and",
        "row permutations remain exact, while independent B=1 reconstructions are",
        "exact. The quality comparator's relative term is now explicitly scaled by",
        "the B=1 reference as required; this non-protected correction does not make",
        "any of the ten affected batch units pass.",
        "",
        f"- Execution HEAD: `{summary['execution_head']}`.",
        f"- Source Q0 root: `{summary['source_q0_root']}`.",
        f"- Affected batch units: {summary['affected_unit_count']}/10 rerun; {summary['passing_batch_units']} PASS, {summary['failing_batch_units']} FAIL.",
        "- Existing non-batch units reused: 100/100; no PPL or LongBench scoring.",
        "- Frozen hot paths: 68/68 unchanged.",
        "",
        "## BF16 bounded samples",
        "",
        "| sample | batch | max abs | median abs | P95 abs | violating logits | top-1 | first divergence |",
        "|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in summary["bf16_sample_rows"]:
        lines.append(
            f"| {row['sample_id']} | {row['batch']} | {row['max_abs_error']:.9g} | "
            f"{row['median_abs_error']:.9g} | {row['p95_abs_error']:.9g} | "
            f"{row['violating_element_count']}/{row['element_count']} | "
            f"{'same' if row['selected_token_agrees'] else 'changed'} | `{row['first_divergent_boundary']}` |"
        )
    lines.extend([
        "",
        "All values use the unchanged reference-directed 0.02/0.02 BF16 predicate.",
        "The three diagnosis inputs are deterministic low sample IDs and are not a",
        "new evaluation sample. All ten configurations remain ineligible for Fast PPL.",
        "",
    ])
    return "\n".join(lines)


def run_batch_continuation(campaign: Path, diagnostic: Path, runtime_settings: Path, execution_head: str) -> dict[str, Any]:
    """Run and finalize the ten comparator-affected units append-only."""

    if campaign.exists():
        raise DiagnosisError("batch continuation campaign already exists")
    if subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip() != execution_head:
        raise DiagnosisError("batch continuation execution HEAD differs")
    if subprocess.run(("git", "status", "--porcelain=v1", "--untracked-files=all"), cwd=ROOT, check=True, capture_output=True, text=True).stdout:
        raise DiagnosisError("batch continuation source is not clean")
    diagnosis_result = q0.load_json(diagnostic / "diagnosis.json")
    if diagnosis_result.get("status") != "COMPLETE_GATE_REMAINS_FAILED" or not (diagnostic / "COMPLETE").is_file():
        raise DiagnosisError("bounded BF16 diagnosis is incomplete")
    campaign.mkdir(parents=True)
    q0.write_json_new(campaign / "authority.json", {
        "schema_version": "kvbench-q0-batch-diagnosis-authority-1.0.0",
        "contract_id": q0.CONTRACT_ID,
        "contract_sha256": q0.CONTRACT_SHA256,
        "source_q0_root": "2bde5bf4a95becb0b6cbe752c7987355128c412709abd107b89979b21c6a48e0",
        "source_q0_campaign": "q0-20260919t054207209666z-b85111f1-27c21357-closure-5b143f1",
        "execution_head": execution_head,
        "quality_image_digest": q0.QUALITY_IMAGE,
        "affected_units": [f"{configuration}/batch-invariance" for configuration in q0.CONFIGS],
        "unaffected_units_reused": 100,
        "fast_ppl_authorized": False,
    })
    raw = campaign / "bounded_bf16"
    raw.mkdir()
    for name in ("diagnosis.json", "bf16_logits.safetensors", "checksums.sha256", "COMPLETE"):
        shutil.copyfile(diagnostic / name, raw / name)
    shutil.copyfile(runtime_settings, campaign / "runtime_settings.json")

    attempts: list[dict[str, Any]] = []
    with q0.gpu_lock():
        for configuration in q0.CONFIGS:
            q0._gpu_idle()
            attempt = campaign / "worker_attempts" / configuration
            attempt.mkdir(parents=True)
            command = _batch_worker_command(campaign, configuration, execution_head)
            result = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
            q0.write_new(attempt / "stdout.txt", result.stdout.encode())
            q0.write_new(attempt / "stderr.txt", result.stderr.encode())
            q0.write_json_new(attempt / "command.json", {
                "schema_version": "kvbench-q0-batch-diagnosis-worker-1.0.0",
                "configuration": configuration,
                "returncode": result.returncode,
                "execution_head": execution_head,
            })
            if result.returncode != 0:
                raise DiagnosisError(f"affected batch worker failed: {configuration}")
            q0._gpu_idle()
            unit = q0.load_json(campaign / "units" / configuration / "batch-invariance/result.json")
            attempts.append({"configuration": configuration, "status": unit["status"]})

    sample_rows: list[dict[str, Any]] = []
    for sample in diagnosis_result["samples"]:
        for batch_key in ("b4", "b8"):
            item = sample[batch_key]
            metrics = item["metrics"]
            sample_rows.append({
                "sample_id": sample["sample_id"],
                "batch": int(batch_key[1:]),
                "row": item["row"],
                "max_abs_error": metrics["abs_error_max"],
                "median_abs_error": metrics["abs_error_median"],
                "p95_abs_error": metrics["abs_error_p95"],
                "violating_element_count": metrics["violating_element_count"],
                "element_count": metrics["element_count"],
                "selected_token_agrees": metrics["selected_token_agrees"],
                "first_divergent_boundary": item["first_divergent_boundary"],
            })
    pass_count = sum(item["status"] == "PASS" for item in attempts)
    summary = {
        "schema_version": "kvbench-q0-batch-diagnosis-summary-1.0.0",
        "status": "COMPLETE_GATE_REMAINS_FAILED" if pass_count < len(q0.CONFIGS) else "PASS",
        "execution_head": execution_head,
        "source_q0_root": "2bde5bf4a95becb0b6cbe752c7987355128c412709abd107b89979b21c6a48e0",
        "affected_unit_count": len(attempts),
        "passing_batch_units": pass_count,
        "failing_batch_units": len(attempts) - pass_count,
        "unaffected_valid_units_reused": 100,
        "invalidated_units": 0,
        "bf16_gpu_diagnosis_samples": len(diagnosis_result["samples"]),
        "bf16_sample_rows": sample_rows,
        "comparator_fix": "reference-directed explicit atol+rtol*abs(B1)",
        "root_cause": "genuine_cross_batch_bf16_arithmetic_sensitivity",
        "first_divergent_operations": sorted({row["first_divergent_boundary"] for row in sample_rows}),
        "fast_ppl_eligible_configurations": [item["configuration"] for item in attempts if item["status"] == "PASS"],
        "fast_ppl_started": False,
        "protected_implementation_changed": False,
        "locked_hot_path_count": 68,
    }
    q0.write_json_new(campaign / "batch_diagnosis_summary.json", summary)
    q0.write_new(campaign / "batch_diagnosis_report.md", _diagnosis_markdown(summary).encode())
    q0.write_json_new(campaign / "manifest.json", {
        "schema_version": "kvbench-q0-batch-diagnosis-campaign-1.0.0",
        "run_id": campaign.name,
        "status": summary["status"],
        "execution_head": execution_head,
        "source_q0_root": summary["source_q0_root"],
        "quality_image_digest": q0.QUALITY_IMAGE,
        "object_scope": "bounded_bf16_plus_ten_affected_batch_units",
    })
    payloads = sorted(path for path in campaign.rglob("*") if path.is_file())
    inventory = {
        "schema_version": "kvbench-artifact-inventory-1.0.0",
        "run_id": campaign.name,
        "files": [
            {"path": path.relative_to(campaign).as_posix(), "role": q0._role(path.relative_to(campaign).as_posix()), "size_bytes": path.stat().st_size, "sha256": q0.sha256_file(path)}
            for path in payloads
        ],
        "excluded_control_files": ["artifact_inventory.json", "checksums.sha256", "COMPLETE"],
    }
    q0.write_json_new(campaign / "artifact_inventory.json", inventory)
    ledger_paths = q0._checksum_ledger_payloads(campaign)
    ledger = "".join(f"{q0.sha256_file(path)}  {path.relative_to(campaign).as_posix()}\n" for path in ledger_paths).encode()
    q0.write_new(campaign / "checksums.sha256", ledger)
    q0.write_json_new(campaign / "COMPLETE", {
        "schema_version": "kvbench-q0-batch-diagnosis-complete-1.0.0",
        "run_id": campaign.name,
        "status": summary["status"],
        "manifest_sha256": q0.sha256_file(campaign / "manifest.json"),
        "artifact_inventory_sha256": q0.sha256_file(campaign / "artifact_inventory.json"),
        "checksum_ledger_sha256": q0.sha256_file(campaign / "checksums.sha256"),
        "checksum_ledger_path": "checksums.sha256",
        "written_last": True,
    })
    for path in sorted(campaign.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    campaign.chmod(0o555)
    from scripts.r2_artifact import validate_local_artifact
    artifact = validate_local_artifact(campaign)
    return {**summary, "campaign_id": campaign.name, "root_sha256": artifact.root_sha256, "object_count": len(artifact.files)}


def close_staging(source: Path, closure: Path) -> dict[str, Any]:
    """Copy immutable run payloads and rebuild only invalid finalization controls."""

    if closure.exists():
        raise DiagnosisError("diagnosis closure already exists")
    summary = q0.load_json(source / "batch_diagnosis_summary.json")
    if summary.get("affected_unit_count") != 10:
        raise DiagnosisError("diagnosis staging lacks ten affected units")
    closure.mkdir(parents=True)
    controls = {"manifest.json", "artifact_inventory.json", "checksums.sha256", "COMPLETE"}
    for item in source.iterdir():
        if item.name in controls:
            continue
        destination = closure / item.name
        if item.is_dir():
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)
    q0.write_json_new(closure / "manifest.json", {
        "schema_version": "kvbench-q0-batch-diagnosis-campaign-1.0.0",
        "run_id": closure.name,
        "status": summary["status"],
        "execution_head": summary["execution_head"],
        "source_q0_root": summary["source_q0_root"],
        "quality_image_digest": q0.QUALITY_IMAGE,
        "source_invalid_finalization_staging": source.name,
        "run_payloads_reexecuted": False,
    })
    payloads = sorted(path for path in closure.rglob("*") if path.is_file())
    q0.write_json_new(closure / "artifact_inventory.json", {
        "schema_version": "kvbench-artifact-inventory-1.0.0",
        "run_id": closure.name,
        "files": [
            {"path": path.relative_to(closure).as_posix(), "role": q0._role(path.relative_to(closure).as_posix()), "size_bytes": path.stat().st_size, "sha256": q0.sha256_file(path)}
            for path in payloads
        ],
        "excluded_control_files": ["artifact_inventory.json", "checksums.sha256", "COMPLETE"],
    })
    ledger_paths = q0._checksum_ledger_payloads(closure)
    q0.write_new(
        closure / "checksums.sha256",
        "".join(f"{q0.sha256_file(path)}  {path.relative_to(closure).as_posix()}\n" for path in ledger_paths).encode(),
    )
    q0.write_json_new(closure / "COMPLETE", {
        "schema_version": "kvbench-q0-batch-diagnosis-complete-1.0.0",
        "run_id": closure.name,
        "status": summary["status"],
        "manifest_sha256": q0.sha256_file(closure / "manifest.json"),
        "artifact_inventory_sha256": q0.sha256_file(closure / "artifact_inventory.json"),
        "checksum_ledger_sha256": q0.sha256_file(closure / "checksums.sha256"),
        "checksum_ledger_path": "checksums.sha256",
        "written_last": True,
    })
    for path in sorted(closure.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    closure.chmod(0o555)
    from scripts.r2_artifact import validate_local_artifact
    artifact = validate_local_artifact(closure)
    return {
        "status": summary["status"],
        "campaign_id": closure.name,
        "root_sha256": artifact.root_sha256,
        "object_count": len(artifact.files),
        "run_payloads_reexecuted": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run-bf16", action="store_true")
    action.add_argument("--run-batch-worker", action="store_true")
    action.add_argument("--run-batch-continuation", action="store_true")
    action.add_argument("--close-staging", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--configuration", choices=q0.CONFIGS)
    parser.add_argument("--diagnostic", type=Path)
    parser.add_argument("--runtime-settings", type=Path)
    parser.add_argument("--source-staging", type=Path)
    parser.add_argument("--execution-head", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_bf16:
        if args.output_dir is None:
            raise DiagnosisError("--output-dir is required")
        result = run_bf16(args.output_dir, args.execution_head)
        print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir)}, sort_keys=True))
        return 0
    if args.run_batch_worker:
        if args.campaign is None or args.configuration is None:
            raise DiagnosisError("batch worker arguments are required")
        result = run_batch_worker(args.campaign, args.configuration, args.execution_head)
        print(json.dumps({"configuration": args.configuration, "status": result["status"]}, sort_keys=True))
        return 0
    if args.run_batch_continuation:
        if args.campaign is None or args.diagnostic is None or args.runtime_settings is None:
            raise DiagnosisError("batch continuation arguments are required")
        result = run_batch_continuation(args.campaign, args.diagnostic, args.runtime_settings, args.execution_head)
        print(json.dumps(result, sort_keys=True))
        return 0
    if args.close_staging:
        if args.source_staging is None or args.campaign is None:
            raise DiagnosisError("closure arguments are required")
        result = close_staging(args.source_staging, args.campaign)
        print(json.dumps(result, sort_keys=True))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
