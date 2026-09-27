#!/usr/bin/env python3
"""Approved physical-B=1 LongBench-E execution for BF16 and KIVI k4v4."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import traceback
from typing import Any, Mapping, Sequence

from scripts import q0_cache_sensitive_correctness as q0
from scripts import q1a_fast_ppl as q1a
from scripts import q1b_full_ppl as q1b


ROOT = q0.ROOT
CONTRACT = q0.CONTRACT_BUNDLE
SOURCE = CONTRACT / "selected_inputs/longbench_e"
CONFIGS = ("bf16", "k4v4")
TASKS = (
    "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report",
    "multi_news", "trec", "triviaqa", "samsum", "passage_count",
    "passage_retrieval_en", "lcc", "repobench-p",
)
BUCKETS = ("0-4k", "4-8k", "8k+")
CATEGORIES = {
    "single_document_qa": ("qasper", "multifieldqa_en"),
    "multi_document_qa": ("hotpotqa", "2wikimqa"),
    "summarization": ("gov_report", "multi_news", "samsum"),
    "few_shot_learning": ("trec", "triviaqa"),
    "synthetic": ("passage_count", "passage_retrieval_en"),
    "code_completion": ("lcc", "repobench-p"),
}
Q1B_ROOT = "725f28a2b5cfa00fc68642b5661755233f20c10b0c7272998abfa3200bf39dd8"
Q1B_EVIDENCE = ROOT / "docs/evidence/q1b/full-ppl.json"
Q1B_REPORT = ROOT / "docs/phase_reports/q1b-full-ppl.md"
Q1B_CAMPAIGN = ROOT / "artifacts/q1b/q1b-20260921t130810654798z-e5961237-fe8adff2"
PROMPT_LOCK = CONTRACT / "prompt_lock.json"
INPUT_MANIFEST = CONTRACT / "input_manifest.json"
MARGINS = CONTRACT / "quality_margins.json"
STOP_FIRST_LINE_TASKS = frozenset(("trec", "triviaqa", "samsum"))
MAX_MODEL_TOKENS = 131072
CONDITIONING = 16


class Q2AError(RuntimeError):
    """Q2A failed closed."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json(path: Path) -> dict[str, Any]:
    return q0.load_json(path)


def _sha(path: Path) -> str:
    return q0.sha256_file(path)


def _input_ledger() -> dict[str, str]:
    return {row["path"]: row["sha256"] for row in _json(INPUT_MANIFEST)["files"]}


def _verify_input(relative: str, ledger: Mapping[str, str]) -> Path:
    path = CONTRACT / relative
    if relative not in ledger or not path.is_file() or _sha(path) != ledger[relative]:
        raise Q2AError(f"frozen LongBench-E input differs: {relative}")
    return path


def verify_authority() -> dict[str, Any]:
    q1b.verify_authority()
    if _json(Q1B_EVIDENCE).get("campaign_root_sha256") != Q1B_ROOT:
        raise Q2AError("Q1B evidence root differs")
    if _json(Q1B_EVIDENCE).get("k4v4_full_ppl_status") != "PASS":
        raise Q2AError("k4v4 is not Q2A eligible")
    if _json(Q1B_EVIDENCE).get("original_cross_batch_gate") != "FAILED":
        raise Q2AError("historical cross-batch failure differs")
    if _sha(q0.CONTRACT) != q0.CONTRACT_SHA256 or _sha(CONTRACT / "contract_snapshot.yaml") != q0.CONTRACT_SHA256:
        raise Q2AError("approved contract bytes differ")
    q1a.verify_amendment()
    lock = _json(PROMPT_LOCK)
    for item in lock["files"]:
        path = ROOT / ".qp1-cache/sources" / item["path"]
        if _sha(path) != item["sha256"]:
            raise Q2AError(f"official LongBench source differs: {item['path']}")
    return {
        "contract_id": q0.CONTRACT_ID,
        "contract_sha256": q0.CONTRACT_SHA256,
        "amendment_id": q1a.AMENDMENT_ID,
        "amendment_sha256": q1a.AMENDMENT_SHA256,
        "q1b_root_sha256": Q1B_ROOT,
        "qp1_root_sha256": q0.QP1_ROOT,
        "quality_image_digest": q0.QUALITY_IMAGE,
        "prompt_lock_sha256": _sha(PROMPT_LOCK),
        "input_manifest_sha256": _sha(INPUT_MANIFEST),
        "official_metric_source_sha256": next(row["sha256"] for row in lock["files"] if row["path"] == "LongBench/metrics.py"),
        "original_cross_batch_gate": "FAILED",
        "physical_batch_size": 1,
    }


def _source_rows(task: str) -> list[dict[str, Any]]:
    with gzip.open(SOURCE / f"{task}.jsonl.gz", "rt", encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def frozen_plan() -> dict[str, Any]:
    authority = verify_authority()
    ledger = _input_ledger()
    contract = _json(q0.CONTRACT)["quality_contract"]
    if tuple(contract["longbench_e"]["tasks"]) != TASKS:
        raise Q2AError("contract task order differs")
    expected = contract["fixture_manifest"]["longbench_e"]
    entries: list[dict[str, Any]] = []
    task_counts: dict[str, int] = {}
    for task in TASKS:
        base = f"selected_inputs/longbench_e/{task}"
        for suffix in ("index.json", "jsonl.gz", "token_ids.i32.gz"):
            _verify_input(f"{base}.{suffix}", ledger)
        index = _json(SOURCE / f"{task}.index.json")
        rows = _source_rows(task)
        if len(rows) != len(index["records"]) or len(rows) != int(expected[task]["eligible"]):
            raise Q2AError(f"frozen task cardinality differs: {task}")
        if [row["sample_id"] for row in rows] != contract["longbench_e"]["sample_ids"][task]:
            raise Q2AError(f"frozen sample order differs: {task}")
        for ordinal, (row, token_index) in enumerate(zip(rows, index["records"], strict=True)):
            if not row["eligible"] or row["sample_id"] != token_index["sample_id"]:
                raise Q2AError(f"frozen sample identity differs: {task}/{ordinal}")
            prompt_length = int(token_index["length"])
            budget = int(row["generation_budget"])
            if prompt_length <= CONDITIONING or prompt_length + budget + int(row["safety_margin"]) > MAX_MODEL_TOKENS:
                raise Q2AError(f"frozen sample capacity differs: {task}/{ordinal}")
            if int(row["conditioning_tokens"]) != CONDITIONING or int(row["prefill_token_count"]) != prompt_length - CONDITIONING:
                raise Q2AError(f"frozen suffix split differs: {task}/{ordinal}")
            if row["length_bucket"] not in BUCKETS:
                raise Q2AError(f"frozen E bucket differs: {task}/{ordinal}")
            entries.append({
                "task": task, "ordinal": ordinal, "sample_id": row["sample_id"],
                "offset": int(token_index["offset"]), "prompt_length": prompt_length,
                "prompt_token_sha256": token_index["token_ids_sha256"],
                "generation_budget": budget, "length_bucket": row["length_bucket"],
                "formatting_policy": row["formatting_policy"],
            })
        task_counts[task] = len(rows)
    if len(entries) != 3668 or sum(task_counts.values()) != 3668:
        raise Q2AError("LongBench-E eligible count differs from frozen manifest")
    if set().union(*map(set, CATEGORIES.values())) != set(TASKS):
        raise Q2AError("LongBench-E category coverage differs")
    return {
        "schema_version": "kvbench-q2a-longbench-e-plan-1.0.0",
        "authority": authority, "configurations": list(CONFIGS),
        "tasks": list(TASKS), "task_counts": task_counts,
        "physical_batch_size": 1, "execution_mode": "cache_sensitive_growing_eager",
        "decode_conditioning_tokens": CONDITIONING,
        "generation": {"do_sample": False, "temperature": 0.0, "top_p": 1.0},
        "stop": {"eos_token_id": 128009, "samsum_additional_newline_token": "tokenizer.encode_newline_last_id"},
        "source_order": "frozen_task_then_frozen_sample_order",
        "recovery": "one_bounded_retry_only_for_infrastructure_or_incomplete_output; completed_units_are_immutable",
        "planned_pairs": 3668, "planned_outputs": 7336,
        "entries": entries,
    }


def _unit(campaign: Path, configuration: str, item: Mapping[str, Any]) -> Path:
    return campaign / "units" / configuration / str(item["task"]) / str(item["sample_id"])


def _accepted(campaign: Path, configuration: str, item: Mapping[str, Any]) -> dict[str, Any] | None:
    root = _unit(campaign, configuration, item)
    accepted = root / "accepted.json"
    if not accepted.exists():
        # A crash between an immutable completed attempt and its small pointer
        # is a finalization failure, not a reason to regenerate an answer.
        completed = sorted(root.glob("attempt-*/COMPLETE")) if root.exists() else []
        valid = []
        for marker in completed:
            result_path = marker.parent / "result.json"
            if result_path.is_file() and _json(marker).get("result_sha256") == _sha(result_path):
                if _json(result_path).get("status") == "COMPLETED":
                    valid.append(marker.parent)
        if len(valid) > 1:
            raise Q2AError("multiple completed outputs for one sample")
        if not valid:
            return None
        if (campaign / "COMPLETE").exists():
            raise Q2AError("finalized campaign has no accepted pointer")
        q0.write_json_new(accepted, {
            "schema_version": "kvbench-q2a-accepted-output-1.0.0",
            "attempt_id": valid[0].name,
            "result_sha256": _sha(valid[0] / "result.json"),
        })
    pointer = _json(accepted)
    attempt = root / pointer["attempt_id"]
    result_path = attempt / "result.json"
    marker = attempt / "COMPLETE"
    if not result_path.is_file() or not marker.is_file():
        raise Q2AError("accepted output is incomplete")
    if _json(marker).get("result_sha256") != _sha(result_path):
        raise Q2AError("accepted output checksum differs")
    if pointer.get("result_sha256") != _sha(result_path):
        raise Q2AError("accepted pointer checksum differs")
    result = _json(result_path)
    if result.get("status") != "COMPLETED" or result.get("configuration") != configuration:
        raise Q2AError("accepted output identity differs")
    for key in ("task", "sample_id", "prompt_token_sha256", "generation_budget"):
        if result.get(key) != item.get(key):
            raise Q2AError(f"accepted output input identity differs: {key}")
    if result.get("contract_sha256") != q0.CONTRACT_SHA256 or result.get("amendment_sha256") != q1a.AMENDMENT_SHA256:
        raise Q2AError("accepted output contract identity differs")
    if result.get("quality_image_digest") != q0.QUALITY_IMAGE or result.get("physical_batch_size") != 1:
        raise Q2AError("accepted output execution identity differs")
    if result.get("execution_mode") != "cache_sensitive_growing_eager":
        raise Q2AError("accepted output generation mode differs")
    if result.get("generated_text_sha256") != q0.sha256_bytes(result["generated_text"].encode("utf-8")):
        raise Q2AError("accepted output text checksum differs")
    return result


def _write_attempt(campaign: Path, configuration: str, item: Mapping[str, Any], result: Mapping[str, Any], attempt_id: str) -> None:
    root = _unit(campaign, configuration, item)
    attempt = root / attempt_id
    if attempt.exists():
        raise Q2AError("attempt ID already exists")
    attempt.mkdir(parents=True)
    q0.write_json_new(attempt / "result.json", result)
    q0.write_json_new(attempt / "COMPLETE", {
        "schema_version": "kvbench-q2a-unit-complete-1.0.0",
        "attempt_id": attempt_id, "status": result["status"],
        "result_sha256": _sha(attempt / "result.json"), "written_last": True,
    })
    if result["status"] == "COMPLETED":
        q0.write_json_new(root / "accepted.json", {
            "schema_version": "kvbench-q2a-accepted-output-1.0.0",
            "attempt_id": attempt_id, "result_sha256": _sha(attempt / "result.json"),
        })


def _next_token(logits: Any, *, blocked_tokens: Sequence[int] = ()) -> int:
    import torch
    vector = logits[0, -1, :]
    if not bool(torch.isfinite(vector).all()):
        raise Q2AError("generation logits contain NaN/Inf")
    if blocked_tokens:
        vector = vector.clone()
        for token in blocked_tokens:
            vector[int(token)] = -float("inf")
    # Copy the selected scalar before the endpoint reuses its output buffer.
    return int(torch.argmax(vector).item())


def _stop_reason(token: int, task: str, eos: int, newline: int | None) -> str | None:
    if token == eos:
        return "eos"
    if task == "samsum" and newline is not None and token == newline:
        return "samsum_newline"
    return None


def split_prompt(prompt_ids: Sequence[int]) -> tuple[list[int], list[int]]:
    if len(prompt_ids) <= CONDITIONING:
        raise Q2AError("prompt is too short for frozen conditioning split")
    prefix_ids = [int(value) for value in prompt_ids[:-CONDITIONING]]
    conditioning_ids = [int(value) for value in prompt_ids[-CONDITIONING:]]
    if len(conditioning_ids) != CONDITIONING or [*prefix_ids, *conditioning_ids] != list(prompt_ids):
        raise Q2AError("LongBench-E suffix concatenation differs")
    return prefix_ids, conditioning_ids


def autoregressive_tokens(first_logits: Any, *, task: str, budget: int, eos: int,
                          newline: int | None, advance: Any) -> tuple[list[int], str]:
    if budget < 1:
        raise Q2AError("generation budget must be positive")
    logits = first_logits
    generated: list[int] = []
    reason = "budget_exhausted"
    for index in range(budget):
        # Official samsum uses min_length=context_length+1. Both EOS and its
        # newline stop are suppressed for the first generated answer token.
        blocked = tuple(token for token in (eos, newline) if token is not None) if task == "samsum" and index == 0 else ()
        token = _next_token(logits, blocked_tokens=blocked)
        generated.append(token)
        stop = _stop_reason(token, task, eos, newline)
        if stop is not None:
            reason = stop
            break
        if index + 1 == budget:
            break
        logits = advance(token, index)
    return generated, reason


def generate_sample(loaded: Any, configuration: str, item: Mapping[str, Any], prompt_ids: Sequence[int], newline_id: int | None) -> dict[str, Any]:
    import torch

    if len(prompt_ids) != int(item["prompt_length"]) or q0.sha256_bytes(q0._i32_bytes(prompt_ids)) != item["prompt_token_sha256"]:
        raise Q2AError("frozen prompt token checksum differs")
    prefix_ids, conditioning_ids = split_prompt(prompt_ids)
    budget = int(item["generation_budget"])
    steps = CONDITIONING + budget
    method, cache, endpoint, positions, rope = q0._allocate_state(
        loaded, configuration, batch=1, capacity=len(prompt_ids) + budget,
        mode="growing", prefix_length=len(prefix_ids), output_steps=steps,
    )
    selected = q0.family(configuration)
    prefix = torch.tensor(prefix_ids, dtype=torch.long, device=cache.device).reshape(1, -1)
    conditioning = torch.tensor(conditioning_ids, dtype=torch.long, device=cache.device)
    generated: list[int] = []
    reason = "budget_exhausted"
    with torch.inference_mode():
        q0._prefill(endpoint, prefix, selected)
        cache.prepare_growing(len(prefix_ids), steps)
        logits = None
        for index in range(CONDITIONING):
            cache.select_growing_step(index)
            logits = endpoint.decode(conditioning[index].reshape(1, 1), positions[index], rope[index])
            q0._finish_step(cache, selected)
        if logits is None:
            raise Q2AError("no conditioning logits")
        def advance(token: int, index: int) -> Any:
            cache.select_growing_step(CONDITIONING + index)
            next_logits = endpoint.decode(
                torch.tensor([[token]], dtype=torch.long, device=cache.device),
                positions[CONDITIONING + index], rope[CONDITIONING + index],
            )
            q0._finish_step(cache, selected)
            return next_logits

        generated, reason = autoregressive_tokens(
            logits, task=item["task"], budget=budget,
            eos=int(loaded.tokenizer.eos_token_id), newline=newline_id,
            advance=advance,
        )
    if int(cache.active_context) < len(prompt_ids) or len(generated) < 1:
        raise Q2AError("growing cache state did not advance")
    text = loaded.tokenizer.decode(generated, skip_special_tokens=True)
    return {
        "schema_version": "kvbench-q2a-generated-output-1.0.0",
        "status": "COMPLETED", "configuration": configuration,
        "task": item["task"], "sample_id": item["sample_id"], "ordinal": item["ordinal"],
        "contract_id": q0.CONTRACT_ID, "contract_sha256": q0.CONTRACT_SHA256,
        "amendment_id": q1a.AMENDMENT_ID, "amendment_sha256": q1a.AMENDMENT_SHA256,
        "quality_image_digest": q0.QUALITY_IMAGE, "physical_batch_size": 1,
        "execution_mode": "cache_sensitive_growing_eager",
        "method_config_fingerprint": method.config_fingerprint(cache.layout_fingerprint()),
        "cache_layout_fingerprint": cache.layout_fingerprint(),
        "prompt_token_sha256": item["prompt_token_sha256"],
        "prompt_length": len(prompt_ids), "prefill_length": len(prefix_ids),
        "conditioning_length": CONDITIONING, "generation_budget": budget,
        "generated_token_ids": generated, "generated_token_count": len(generated),
        "generated_text": text, "generated_text_sha256": q0.sha256_bytes(text.encode("utf-8")),
        "stop_reason": reason, "budget_exhausted": reason == "budget_exhausted",
        "length_bucket": item["length_bucket"],
        "active_context": int(cache.active_context),
        "allocated_bytes": int(cache.accounting().allocated_bytes),
        "score_computed_in_worker": False,
    }


def run_worker(campaign: Path, configuration: str, execution_head: str) -> dict[str, Any]:
    if configuration not in CONFIGS:
        raise Q2AError("configuration outside Q2A scope")
    if os.environ.get("KVBENCH_QUALITY_IMAGE_DIGEST") != q0.QUALITY_IMAGE:
        raise Q2AError("Quality image identity differs")
    observed = subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    if observed != execution_head:
        raise Q2AError("execution HEAD differs")
    q0.verify_locked_paths()
    plan = _json(campaign / "execution_plan.json")
    expected_fingerprints = q0.contract_config_fingerprints()

    import gc
    import torch
    from kvbench.runtime.backend import forced_flash_execution
    from kvbench.runtime.model_loader import load_frozen_model

    loaded = load_frozen_model(device=torch.device("cuda:0"))
    newline_ids = loaded.tokenizer.encode("\n", add_special_tokens=False)
    newline_id = int(newline_ids[-1]) if newline_ids else None
    completed = 0
    with torch.inference_mode(), forced_flash_execution():
        for task in TASKS:
            values = q0._read_i32_gzip(SOURCE / f"{task}.token_ids.i32.gz")
            for item in (row for row in plan["entries"] if row["task"] == task):
                if _accepted(campaign, configuration, item) is not None:
                    completed += 1
                    continue
                prompt_ids = values[item["offset"]:item["offset"] + item["prompt_length"]]
                root = _unit(campaign, configuration, item)
                existing = sorted(root.glob("attempt-*")) if root.exists() else []
                if len(existing) > 1:
                    raise Q2AError("bounded retry policy exceeded")
                if existing:
                    if (existing[0] / "COMPLETE").exists():
                        previous = _json(existing[0] / "result.json")
                        if previous.get("failure_class") not in {"infrastructure_failed", "incomplete_output"}:
                            raise Q2AError("non-retryable output already exists")
                attempt_id = f"attempt-{len(existing):02d}"
                try:
                    result = generate_sample(loaded, configuration, item, prompt_ids, newline_id)
                    result["contract_method_config_fingerprint"] = expected_fingerprints[configuration]
                    result["execution_head"] = execution_head
                    result["attempt_id"] = attempt_id
                    _write_attempt(campaign, configuration, item, result, attempt_id)
                except BaseException as error:
                    if not (root / attempt_id / "COMPLETE").exists():
                        _write_attempt(campaign, configuration, item, {
                            "schema_version": "kvbench-q2a-generation-failure-1.0.0",
                            "status": "FAILED", "configuration": configuration,
                            "task": task, "sample_id": item["sample_id"],
                            "attempt_id": attempt_id, "execution_head": execution_head,
                            "failure_class": "runtime_or_correctness_failure",
                            "error_type": type(error).__name__, "error": str(error),
                            "traceback": traceback.format_exc(), "retry_permitted": False,
                        }, attempt_id)
                    raise
                completed += 1
                print(json.dumps({"configuration": configuration, "task": task, "completed": completed, "total": 3668}, sort_keys=True), flush=True)
                del result, prompt_ids
                gc.collect()
                torch.cuda.empty_cache()
    return {"configuration": configuration, "completed": completed}


def _worker_command(campaign: Path, configuration: str, execution_head: str) -> list[str]:
    command = q1b._worker_command(campaign, configuration, execution_head)
    old = f"dst=/home/rockrock/cmu_paper/artifacts/q1b/{campaign.name}"
    new = f"dst=/home/rockrock/cmu_paper/artifacts/q2a/{campaign.name}"
    for index, value in enumerate(command):
        if value.startswith("type=bind,") and old in value:
            command[index] = value.replace(old, new)
            break
    else:
        raise Q2AError("Q2A campaign mount is absent")
    index = command.index("-m")
    return command[:index] + [
        "-m", "scripts.q2a_longbench_e", "--run-worker",
        "--campaign", f"/home/rockrock/cmu_paper/artifacts/q2a/{campaign.name}",
        "--configuration", configuration, "--execution-head", execution_head,
    ]


def start_campaign(campaign: Path, execution_head: str) -> dict[str, Any]:
    if campaign.exists():
        raise Q2AError("campaign ID already exists")
    observed = subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    if observed != execution_head:
        raise Q2AError("execution HEAD differs")
    authority = verify_authority()
    protected = q0.verify_locked_paths()
    plan = frozen_plan()
    campaign.mkdir(parents=True)
    q0.write_json_new(campaign / "execution_authorization.json", {
        "schema_version": "kvbench-q2a-execution-authorization-1.0.0",
        "status": "APPROVED", "approved_by": "operator", "recorded_at_utc": _now(),
        "authorization_scope": ["Q2A_B1_LONGBENCH_E"],
        "authorized_configurations": list(CONFIGS),
        "original_contract_id": q0.CONTRACT_ID, "original_contract_sha256": q0.CONTRACT_SHA256,
        "amendment_id": q1a.AMENDMENT_ID, "amendment_sha256": q1a.AMENDMENT_SHA256,
        "approval_text": "Execute frozen LongBench-E only for bf16 and k4v4 at physical B=1; do not start Q2B or quality-performance join.",
    })
    q0.write_json_new(campaign / "execution_plan.json", plan)
    q0.write_json_new(campaign / "entry.json", {
        "schema_version": "kvbench-q2a-entry-1.0.0", "status": "PASS",
        "execution_head": execution_head, "authority": authority,
        "protected_paths": protected, "quality_image_digest": q0.QUALITY_IMAGE,
        "untracked_operator_prompt": "Q2A_B1_LONGBENCH_E_PROMPT.md",
    })
    return {"campaign_id": campaign.name, "planned_outputs": plan["planned_outputs"]}


def run_campaign(campaign: Path, execution_head: str) -> None:
    if not (campaign / "execution_plan.json").is_file():
        start_campaign(campaign, execution_head)
    elif (campaign / "COMPLETE").exists():
        raise Q2AError("finalized campaign cannot resume")
    with q0.gpu_lock():
        for configuration in CONFIGS:
            plan = _json(campaign / "execution_plan.json")
            if all(_accepted(campaign, configuration, item) is not None for item in plan["entries"]):
                continue
            q0._gpu_idle()
            attempts_root = campaign / "worker_attempts" / configuration
            attempt_no = len(list(attempts_root.glob("attempt-*"))) if attempts_root.exists() else 0
            if attempt_no > 1:
                raise Q2AError("bounded worker retry policy exceeded")
            attempt = attempts_root / f"attempt-{attempt_no:02d}"
            attempt.mkdir(parents=True)
            command = _worker_command(campaign, configuration, execution_head)
            with (attempt / "stdout.txt").open("wb") as stdout, (attempt / "stderr.txt").open("wb") as stderr:
                process = subprocess.run(command, cwd=ROOT, check=False, stdout=stdout, stderr=stderr)
            q0.write_json_new(attempt / "attempt.json", {
                "schema_version": "kvbench-q2a-worker-attempt-1.0.0",
                "configuration": configuration, "returncode": process.returncode,
                "execution_head": execution_head, "attempt_id": attempt.name,
            })
            q0._gpu_idle()
            if process.returncode != 0:
                raise Q2AError(f"Q2A worker failed for {configuration}; completed outputs preserved")


def _official_metric(task: str, prediction: str, answers: Sequence[str], all_classes: Any) -> tuple[float, str]:
    source = ROOT / ".qp1-cache/sources/LongBench"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    import metrics  # type: ignore[import-not-found]

    mapping = {
        "qasper": metrics.qa_f1_score, "multifieldqa_en": metrics.qa_f1_score,
        "hotpotqa": metrics.qa_f1_score, "2wikimqa": metrics.qa_f1_score,
        "gov_report": metrics.rouge_score, "multi_news": metrics.rouge_score,
        "trec": metrics.classification_score, "triviaqa": metrics.qa_f1_score,
        "samsum": metrics.rouge_score, "passage_count": metrics.count_score,
        "passage_retrieval_en": metrics.retrieval_score,
        "lcc": metrics.code_sim_score, "repobench-p": metrics.code_sim_score,
    }
    if task in STOP_FIRST_LINE_TASKS:
        prediction = prediction.lstrip("\n").split("\n")[0]
    score = max((float(mapping[task](prediction, answer, all_classes=all_classes)) for answer in answers), default=0.0)
    if not math.isfinite(score) or score < 0.0 or score > 1.0:
        raise Q2AError("official metric returned invalid score")
    return score, prediction


def _task_score(rows: Sequence[Mapping[str, Any]], configuration: str) -> tuple[float, dict[str, float]]:
    values: dict[str, float] = {}
    for bucket in BUCKETS:
        subset = [float(row[f"{configuration}_score"]) for row in rows if row["length_bucket"] == bucket]
        if not subset:
            raise Q2AError("LongBench-E task has an empty frozen E bucket")
        values[bucket] = round(100.0 * statistics.mean(subset), 2)
    return statistics.mean(values.values()), values


def _bootstrap(paired: Sequence[Mapping[str, Any]], *, seed: int, draws: int) -> tuple[float, float]:
    by_task = {task: [row for row in paired if row["task"] == task] for task in TASKS}
    if any(not rows for rows in by_task.values()):
        raise Q2AError("paired task coverage is incomplete")
    rng = random.Random(seed)
    distribution: list[float] = []
    for _ in range(draws):
        task_drops = []
        for task in TASKS:
            rows = by_task[task]
            selected = [rows[rng.randrange(len(rows))] for _ in rows]
            base, _ = _task_score(selected, "bf16")
            method, _ = _task_score(selected, "k4v4")
            task_drops.append(base - method)
        distribution.append(statistics.mean(task_drops))
    distribution.sort()
    return distribution[int(0.025 * draws)], distribution[min(draws - 1, int(0.975 * draws))]


def classify_gate(drop: float, ci: Sequence[float], category_drops: Mapping[str, float], invalid_increase_pp: float, *, complete: bool, margins: Mapping[str, Any]) -> tuple[str, list[str]]:
    if not complete:
        return "INCONCLUSIVE", ["incomplete_or_invalid_pairing"]
    threshold = float(margins["macro_drop_max_score_points"])
    category_hard = float(margins["category_hard_fail_score_points"])
    invalid_hard = float(margins["invalid_output_increase_max_percentage_points"])
    reasons = [f"category_hard_fail:{name}" for name, value in category_drops.items() if value > category_hard]
    if invalid_increase_pp > invalid_hard:
        reasons.append("invalid_output_increase_hard_fail")
    if ci[0] > threshold:
        reasons.append("macro_lower_ci_exceeds_margin")
    if reasons:
        return "FAIL", reasons
    if ci[1] <= threshold:
        return "PASS", ["macro_upper_ci_within_margin", "hard_guardrails_pass"]
    return "INCONCLUSIVE", ["macro_ci_crosses_margin"]


def analyze(campaign: Path) -> dict[str, Any]:
    plan = _json(campaign / "execution_plan.json")
    all_rows: dict[str, list[dict[str, Any]]] = {}
    for task in TASKS:
        all_rows[task] = _source_rows(task)
    paired: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    output_index: list[dict[str, Any]] = []
    for item in plan["entries"]:
        source = all_rows[item["task"]][item["ordinal"]]
        outputs: dict[str, dict[str, Any]] = {}
        for configuration in CONFIGS:
            result = _accepted(campaign, configuration, item)
            if result is None:
                failures.append({"configuration": configuration, "task": item["task"], "sample_id": item["sample_id"], "reason": "missing_output"})
                continue
            outputs[configuration] = result
            output_index.append({
                "configuration": configuration, "task": item["task"], "sample_id": item["sample_id"],
                "attempt_id": result["attempt_id"], "execution_head": result["execution_head"],
                "result_sha256": _sha(_unit(campaign, configuration, item) / result["attempt_id"] / "result.json"),
            })
        if len(outputs) != 2:
            continue
        scored: dict[str, Any] = {}
        for configuration in CONFIGS:
            result = outputs[configuration]
            score, parsed = _official_metric(item["task"], result["generated_text"], source["source"]["answers"], source["source"]["all_classes"])
            # An ordinary wrong answer remains valid; only an empty parsed answer is invalid.
            invalid = parsed.strip() == ""
            scored[f"{configuration}_score"] = score
            scored[f"{configuration}_invalid"] = invalid
            scored[f"{configuration}_parsed_text"] = parsed
            scored[f"{configuration}_generated_length"] = result["generated_token_count"]
            scored[f"{configuration}_stop_reason"] = result["stop_reason"]
        paired.append({
            "task": item["task"], "sample_id": item["sample_id"],
            "length_bucket": item["length_bucket"], "source_length": source["source"]["length"],
            "reference_ids": [q0.sha256_bytes(str(answer).encode()) for answer in source["source"]["answers"]],
            **scored,
        })
    complete = len(paired) == plan["planned_pairs"] and not failures
    task_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    length_rows: list[dict[str, Any]] = []
    gate_status = "INCONCLUSIVE"
    gate_reasons = ["incomplete_or_invalid_pairing"]
    macro: dict[str, Any] = {}
    invalid: dict[str, Any] = {}
    if complete:
        for task in TASKS:
            subset = [row for row in paired if row["task"] == task]
            base, base_buckets = _task_score(subset, "bf16")
            method, method_buckets = _task_score(subset, "k4v4")
            task_rows.append({"task": task, "pairs": len(subset), "bf16_score_points": base,
                              "k4v4_score_points": method, "drop_score_points": base - method,
                              "bf16_buckets": base_buckets, "k4v4_buckets": method_buckets})
        for name, tasks in CATEGORIES.items():
            selected = [row for row in task_rows if row["task"] in tasks]
            base = statistics.mean(row["bf16_score_points"] for row in selected)
            method = statistics.mean(row["k4v4_score_points"] for row in selected)
            category_rows.append({"category": name, "tasks": list(tasks), "pairs": sum(row["pairs"] for row in selected),
                                  "bf16_score_points": base, "k4v4_score_points": method,
                                  "drop_score_points": base - method})
        for bucket in BUCKETS:
            base = statistics.mean(row["bf16_buckets"][bucket] for row in task_rows)
            method = statistics.mean(row["k4v4_buckets"][bucket] for row in task_rows)
            length_rows.append({"bucket": bucket, "pairs": sum(row["length_bucket"] == bucket for row in paired),
                                "bf16_score_points": base, "k4v4_score_points": method,
                                "drop_score_points": base - method})
        base = statistics.mean(row["bf16_score_points"] for row in task_rows)
        method = statistics.mean(row["k4v4_score_points"] for row in task_rows)
        margins = _json(MARGINS)
        ci = _bootstrap(paired, seed=int(margins["bootstrap"]["seed"]), draws=int(margins["bootstrap"]["draws"]))
        invalid = {configuration: {"count": sum(bool(row[f"{configuration}_invalid"]) for row in paired),
                                   "denominator": len(paired)} for configuration in CONFIGS}
        invalid_increase_pp = 100.0 * (invalid["k4v4"]["count"] - invalid["bf16"]["count"]) / len(paired)
        category_drops = {row["category"]: row["drop_score_points"] for row in category_rows}
        gate_status, gate_reasons = classify_gate(base - method, ci, category_drops, invalid_increase_pp,
                                                 complete=True, margins=margins["longbench_e"])
        for row in category_rows:
            row["review_flag"] = row["drop_score_points"] > float(margins["longbench_e"]["category_review_score_points"])
            row["hard_fail"] = row["drop_score_points"] > float(margins["longbench_e"]["category_hard_fail_score_points"])
        macro = {"bf16_score_points": base, "k4v4_score_points": method,
                 "drop_score_points": base - method, "paired_ci95_score_points": list(ci),
                 "aggregation": "equal_13_task_macro_of_official_E_bucket_means",
                 "bootstrap_draws": int(margins["bootstrap"]["draws"]),
                 "bootstrap_seed": int(margins["bootstrap"]["seed"])}
        invalid["increase_percentage_points"] = invalid_increase_pp
    generation = {}
    for configuration in CONFIGS:
        lengths = [int(row[f"{configuration}_generated_length"]) for row in paired]
        generation[configuration] = {
            "paired_output_count": len(lengths),
            "mean_generated_tokens": statistics.mean(lengths) if lengths else None,
            "median_generated_tokens": statistics.median(lengths) if lengths else None,
            "stop_reasons": dict(Counter(row[f"{configuration}_stop_reason"] for row in paired)),
        }
    return {
        "paired_rows": paired, "output_index": output_index, "failures": failures,
        "summary": {
            "schema_version": "kvbench-q2a-longbench-e-summary-1.0.0",
            "execution_status": "COMPLETE" if complete else "PARTIAL",
            "planned_outputs": plan["planned_outputs"], "completed_outputs": len(output_index),
            "missing_or_failed_outputs": len(failures), "planned_pairs": plan["planned_pairs"],
            "accepted_pairs": len(paired), "configurations": list(CONFIGS),
            "physical_batch_size": 1, "original_cross_batch_gate": "FAILED",
            "task_results": task_rows, "category_results": category_rows,
            "length_results": length_rows, "macro": macro, "invalid_outputs": invalid,
            "generation": generation, "gate_status": gate_status,
            "gate_reasons": gate_reasons, "q2b_eligible": gate_status == "PASS",
            "quality_transfer_to_other_batches": "not_established",
        },
    }


def close_campaign(campaign: Path, execution_head: str) -> dict[str, Any]:
    if (campaign / "COMPLETE").exists():
        raise Q2AError("campaign already finalized")
    analysis = analyze(campaign)
    summary = analysis["summary"]
    q0.write_json_new(campaign / "output_index.json", {"rows": analysis["output_index"]})
    q0.write_json_new(campaign / "paired_scores.json", {"rows": analysis["paired_rows"]})
    q0.write_json_new(campaign / "failures.json", {"rows": analysis["failures"]})
    q0.write_json_new(campaign / "longbench_e_summary.json", summary)
    report = [
        "# Q2A B=1 LongBench-E", "",
        f"Execution: **{summary['execution_status']}**; gate: **{summary['gate_status']}**.",
        f"Accepted pairs: {summary['accepted_pairs']}/{summary['planned_pairs']}.",
        "Original cross-batch gate: FAILED; no transfer beyond physical B=1.", "",
        "| Task | BF16 | k4v4 | BF16 − k4v4 |", "|---|---:|---:|---:|",
    ]
    for row in summary["task_results"]:
        report.append(f"| {row['task']} | {row['bf16_score_points']:.3f} | {row['k4v4_score_points']:.3f} | {row['drop_score_points']:.3f} |")
    if summary["macro"]:
        m = summary["macro"]
        report.extend(["", f"Task macro: BF16 {m['bf16_score_points']:.4f}, k4v4 {m['k4v4_score_points']:.4f}; "
                       f"drop {m['drop_score_points']:.4f}, paired 95% CI {m['paired_ci95_score_points']}.  ",
                       f"Gate reasons: {', '.join(summary['gate_reasons'])}."])
    report.extend(["", "Category, length, invalid-output and generation summaries are in longbench_e_summary.json.",
                   "Q2B was not started.", ""])
    q0.write_new(campaign / "q2a_report.md", "\n".join(report).encode())
    q0.write_json_new(campaign / "manifest.json", {
        "schema_version": "kvbench-q2a-longbench-e-campaign-1.0.0",
        "run_id": campaign.name, "execution_head": execution_head,
        "quality_image_digest": q0.QUALITY_IMAGE,
        "contract_id": q0.CONTRACT_ID, "contract_sha256": q0.CONTRACT_SHA256,
        "amendment_id": q1a.AMENDMENT_ID, "amendment_sha256": q1a.AMENDMENT_SHA256,
        "status": summary["execution_status"], "gate_status": summary["gate_status"],
        "physical_batch_size": 1,
    })
    payloads = sorted(path for path in campaign.rglob("*") if path.is_file())
    q0.write_json_new(campaign / "artifact_inventory.json", {
        "schema_version": "kvbench-artifact-inventory-1.0.0", "run_id": campaign.name,
        "files": [{"path": path.relative_to(campaign).as_posix(), "role": q0._role(path.relative_to(campaign).as_posix()),
                   "size_bytes": path.stat().st_size, "sha256": _sha(path)} for path in payloads],
        "excluded_control_files": ["artifact_inventory.json", "checksums.sha256", "COMPLETE"],
    })
    ledger_paths = q0._checksum_ledger_payloads(campaign)
    q0.write_new(campaign / "checksums.sha256", "".join(f"{_sha(path)}  {path.relative_to(campaign).as_posix()}\n" for path in ledger_paths).encode())
    q0.write_json_new(campaign / "COMPLETE", {
        "schema_version": "kvbench-q2a-longbench-e-complete-1.0.0", "run_id": campaign.name,
        "status": summary["execution_status"],
        "manifest_sha256": _sha(campaign / "manifest.json"),
        "artifact_inventory_sha256": _sha(campaign / "artifact_inventory.json"),
        "checksum_ledger_sha256": _sha(campaign / "checksums.sha256"),
        "checksum_ledger_path": "checksums.sha256", "written_last": True,
    })
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--run-worker", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--configuration")
    parser.add_argument("--execution-head")
    args = parser.parse_args(argv)
    if args.plan:
        value = frozen_plan()
        print(json.dumps({key: value[key] for key in ("task_counts", "planned_pairs", "planned_outputs")}, sort_keys=True))
        return 0
    if args.campaign is None or args.execution_head is None:
        parser.error("campaign and execution HEAD are required")
    if args.run_worker:
        print(json.dumps(run_worker(args.campaign, args.configuration, args.execution_head), sort_keys=True))
    elif args.run:
        run_campaign(args.campaign, args.execution_head)
    elif args.analyze:
        result = analyze(args.campaign)
        print(json.dumps(result["summary"], sort_keys=True))
    elif args.finalize:
        print(json.dumps(close_campaign(args.campaign, args.execution_head), sort_keys=True))
    else:
        parser.error("select an action")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
