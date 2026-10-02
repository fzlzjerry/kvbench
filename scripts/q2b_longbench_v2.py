#!/usr/bin/env python3
"""Q2B-only glue around the frozen Q2A physical-B=1 generation path."""
from __future__ import annotations

import argparse
import array
import ast
from collections import Counter, defaultdict
from functools import lru_cache
import gzip
import json
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import traceback
from typing import Any, Mapping, Sequence

from scripts import q0_cache_sensitive_correctness as q0
from scripts import q1a_fast_ppl as q1a
from scripts import q2a_longbench_e as q2a

ROOT = q0.ROOT
CONFIGS = ("bf16", "k4v4")
TASK = "longbench_v2_primary_no_cot"
BASE = "selected_inputs/longbench_v2/primary_no_cot"
SOURCE = q0.CONTRACT_BUNDLE / (BASE + ".jsonl.gz")
TOKENS = q0.CONTRACT_BUNDLE / (BASE + ".token_ids.i32.gz")
BUCKETS = ("8-16k", "16-32k", "32-64k", "64-128k")
Q2A_ROOT = "9b7c4f5a3631afb634a0d9fd50232c3aba3418a518bc86ebeedd1ccbabdeedfa"
STARTING_HEAD = "df09bbf272dd9ef40611b57d1141a6739eb27d35"
PROMPT = ROOT / "Q2B_B1_LONGBENCH_V2_NO_COT_PROMPT.md"
load = q0.load_json
sha = q0.sha256_file


class Q2BError(RuntimeError):
    """Q2B evidence or execution fails closed without changing frozen rules."""


def verify_authority() -> dict[str, Any]:
    # Small accepted handoffs only: never invoke historical campaign validators.
    q0.verify_approval()
    q1a.verify_amendment()
    contract = load(q0.CONTRACT)["quality_contract"]
    margins = load(q2a.MARGINS)
    lock = load(q2a.PROMPT_LOCK)
    if margins != contract["gates"] or lock != contract["prompt_lock"]:
        raise Q2BError("approved metric/prompt lock differs")
    q1b = load(ROOT / "docs/evidence/q1b/full-ppl.json")
    q2a_evidence = load(ROOT / "docs/evidence/q2a/longbench-e.json")
    receipt = load(ROOT / "docs/evidence/q2a/r2-publication.json")
    if q1b.get("k4v4_full_ppl_status") != "PASS" or q1b.get("campaign_root_sha256") != q2a.Q1B_ROOT:
        raise Q2BError("k4v4 Full-PPL authority differs")
    if (q2a_evidence.get("gate_status"), q2a_evidence.get("execution_status"),
        q2a_evidence.get("q2b_eligible_configuration"), q2a_evidence.get("original_cross_batch_gate"),
        q2a_evidence.get("campaign_root_sha256")) != ("PASS", "COMPLETE", "k4v4", "FAILED", Q2A_ROOT):
        raise Q2BError("Q2A finalist authority differs")
    if receipt.get("root_sha256") != Q2A_ROOT or receipt.get("clean_retrieval") != "PASS" or not receipt.get("complete_last"):
        raise Q2BError("Q2A publication receipt differs")
    sources = {r["path"]: r["sha256"] for r in lock["files"]}
    for path in ("pred.py", "result.py", "prompts/0shot.txt"):
        if sha(ROOT / ".qp1-cache/sources" / path) != sources[path]:
            raise Q2BError(f"pinned evaluator differs: {path}")
    ledger = {line.split("  ", 1)[1]: line.split("  ", 1)[0]
              for line in (q0.CONTRACT_BUNDLE / "checksums.sha256").read_text().splitlines()}
    if sha(q2a.INPUT_MANIFEST) != ledger["input_manifest.json"]:
        raise Q2BError("frozen input manifest checksum differs")
    return {
        "contract_id": q0.CONTRACT_ID, "contract_sha256": q0.CONTRACT_SHA256,
        "amendment_id": q1a.AMENDMENT_ID, "amendment_sha256": q1a.AMENDMENT_SHA256,
        "performance_freeze_tag": q0.FREEZE_TAG, "quality_image_digest": q0.QUALITY_IMAGE,
        "qp1_root": q0.QP1_ROOT, "q1b_root": q2a.Q1B_ROOT, "q2a_root": Q2A_ROOT,
        "input_manifest_sha256": sha(q2a.INPUT_MANIFEST), "prompt_lock_sha256": sha(q2a.PROMPT_LOCK),
        "parser_source_sha256": sources["pred.py"], "evaluator_source_sha256": sources["result.py"],
        "original_cross_batch_gate": "FAILED", "physical_batch_size": 1,
        "quality_transfer_to_other_batches": "not_established",
    }


def source_rows():
    with gzip.open(SOURCE, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def frozen_plan() -> dict[str, Any]:
    authority = verify_authority()
    contract = load(q0.CONTRACT)["quality_contract"]
    v2 = contract["longbench_v2"]
    if v2["cot_primary"] or v2["max_new_tokens_no_cot"] != 8 or v2["decode_conditioning_tokens"] != 16:
        raise Q2BError("Q2B primary generation contract differs")
    input_hashes = q2a._input_ledger()
    for suffix in ("index.json", "jsonl.gz", "token_ids.i32.gz"):
        q2a._verify_input(f"{BASE}.{suffix}", input_hashes)
    index = load(q0.CONTRACT_BUNDLE / (BASE + ".index.json"))
    if index["dtype"] != "little_endian_int32" or len(index["records"]) != 503:
        raise Q2BError("primary token index differs")
    records = []
    offset = 0
    for ordinal, (row, token) in enumerate(zip(source_rows(), index["records"], strict=True)):
        length = int(token["length"])
        eligible = length + 8 + 128 <= int(contract["model"]["max_model_length"])
        if (row["sample_id"] != v2["sample_ids"][ordinal] or row["sample_id"] != token["sample_id"]
            or row["token_count"] != length or row["token_ids_sha256"] != token["token_ids_sha256"]
            or token["offset"] != offset or row["eligible"] != eligible
            or row["generation_budget"] != 8 or row["safety_margin"] != 128
            or row["conditioning_tokens"] != 16 or row["prefill_token_count"] != length - 16
            or row["length_bucket"] not in BUCKETS
            or row["parser"] != "official_A_B_C_D_extract_answer"
            or q0.sha256_bytes(row["effective_prompt"].encode()) != row["prompt_text_sha256"]):
            raise Q2BError(f"frozen primary identity/eligibility differs: {ordinal}")
        if row["exclusion_reason"] != (None if eligible else "model_length_budget"):
            raise Q2BError("predeclared exclusion reason differs")
        # Gold labels are deliberately absent from the inference plan.
        records.append({
            "task": TASK, "sample_id": row["sample_id"], "ordinal": ordinal,
            "eligible": eligible, "exclusion_reason": row["exclusion_reason"],
            "offset": offset, "prompt_length": length, "prompt_token_sha256": token["token_ids_sha256"],
            "prompt_text_sha256": row["prompt_text_sha256"], "generation_budget": 8,
            "length_bucket": row["length_bucket"], "category": row["source"]["domain"],
            "sub_category": row["source"]["sub_domain"], "source_length_label": row["source"]["length"],
            "difficulty": row["source"]["difficulty"], "under_8k": length < 8192,
            "reference_label_link": f"{BASE}.jsonl.gz#sample_id={row['sample_id']}",
        })
        offset += length
    if len(records) != 503 or sum(r["eligible"] for r in records) != 321 or offset != index["total_tokens"]:
        raise Q2BError("frozen 503/321/182 cardinality differs")
    return {
        "schema_version": "kvbench-q2b-plan-1.0.0", "authority": authority,
        "configurations": list(CONFIGS), "finalist_roles": {"k4v4": ["best_quality", "highest_r_alloc"]},
        "finalist_deduplicated": True, "physical_batch_size": 1,
        "execution_mode": "cache_sensitive_growing_eager", "model": contract["model"],
        "dataset": v2["revision"], "method_fingerprints": {c: q0.contract_config_fingerprints()[c] for c in CONFIGS},
        "generation": {**contract["execution"], "cot": False, "max_new_tokens": 8,
                       "eos_token_id": contract["model"]["tokenizer_assets"]["eos_token_id"],
                       "additional_stop_tokens": [], "parser_early_stop": False},
        "input_files": {k: v for k, v in input_hashes.items() if k.startswith(BASE + ".")},
        "margins": contract["gates"], "statistics": {
            "aggregation": "sample_accuracy", "strata": ["length_bucket", "category"],
            "interval": "paired_percentile_order_statistics_at_floor_0.025N_and_floor_0.975N",
            "retention_zero_denominator": "undefined; report count and null CI if any bootstrap denominator is zero",
            "mcnemar": "discordant_table_only; contract_does_not_require_a_p_value",
        },
        "planned_pairs": 321, "planned_outputs": 642, "selected_count": 503, "excluded_count": 182,
        "source_order": "frozen_primary_order; bf16_then_k4v4",
        "entries": [r for r in records if r["eligible"]], "eligibility_records": records,
    }


@lru_cache(maxsize=1)
def pinned_parser():
    path = ROOT / ".qp1-cache/sources/pred.py"
    expected = next(r["sha256"] for r in load(q2a.PROMPT_LOCK)["files"] if r["path"] == "pred.py")
    if sha(path) != expected:
        raise Q2BError("official parser source changed")
    tree = ast.parse(path.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "extract_answer"]
    if len(functions) != 1:
        raise Q2BError("official parser definition is absent or ambiguous")
    # Execute the exact pinned function only, avoiding upstream API/model imports.
    namespace = {"re": re}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["extract_answer"]


def parse_answer(text: str) -> dict[str, Any]:
    answer = pinned_parser()(text.strip())  # upstream get_pred strips before extraction
    return {"parsed_answer": answer, "invalid": answer is None,
            "invalid_reason": "official_parser_no_valid_answer" if answer is None else None}


def validate_result(result: Mapping[str, Any], configuration: str, item: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    expected = {
        "status": "COMPLETED", "configuration": configuration, "task": TASK,
        "sample_id": item["sample_id"], "prompt_token_sha256": item["prompt_token_sha256"],
        "prompt_length": item["prompt_length"], "generation_budget": 8, "conditioning_length": 16,
        "physical_batch_size": 1, "execution_mode": "cache_sensitive_growing_eager",
        "contract_sha256": q0.CONTRACT_SHA256, "amendment_sha256": q1a.AMENDMENT_SHA256,
        "quality_image_digest": q0.QUALITY_IMAGE, "cot": False,
        "plan_sha256": q0.sha256_bytes(q0.canonical_bytes(plan)),
        "contract_method_config_fingerprint": plan["method_fingerprints"][configuration],
        "parser_source_sha256": plan["authority"]["parser_source_sha256"],
    }
    if any(result.get(k) != v for k, v in expected.items()):
        raise Q2BError("completed Q2B output identity differs")
    ids = result["generated_token_ids"]
    if not 1 <= len(ids) <= 8 or result["generated_token_count"] != len(ids):
        raise Q2BError("generated token accounting differs")
    if result["generated_ids_sha256"] != q0.sha256_bytes(q0._i32_bytes(ids)):
        raise Q2BError("generated IDs checksum differs")
    if result["generated_text_sha256"] != q0.sha256_bytes(result["generated_text"].encode()):
        raise Q2BError("generated text checksum differs")
    if result["prefill_length"] + 16 != item["prompt_length"] or result["active_context"] != item["prompt_length"] + len(ids) - 1:
        raise Q2BError("growing state/conditioning accounting differs")
    if any(result.get(k) != v for k, v in parse_answer(result["generated_text"]).items()):
        raise Q2BError("stored pinned parse differs")


def accepted(campaign: Path, configuration: str, item: Mapping[str, Any], plan: Mapping[str, Any]):
    root = q2a._unit(campaign, configuration, item)
    # Recover only fully written validated generation. Never generate again for a
    # missing small COMPLETE/accepted pointer after a report-writing interruption.
    for result_path in sorted(root.glob("attempt-*/result.json")):
        marker = result_path.parent / "COMPLETE"
        if not marker.exists():
            result = load(result_path)
            if result.get("status") == "COMPLETED":
                validate_result(result, configuration, item, plan)
                q0.write_json_new(marker, {"schema_version": "kvbench-q2b-unit-complete-1.0.0",
                                         "result_sha256": sha(result_path), "written_last": True})
    result = q2a._accepted(campaign, configuration, item)
    if result is not None:
        validate_result(result, configuration, item, plan)
    return result


def read_prompt(stream, item: Mapping[str, Any]) -> list[int]:
    stream.seek(int(item["offset"]) * 4)
    data = stream.read(int(item["prompt_length"]) * 4)
    if len(data) != item["prompt_length"] * 4 or q0.sha256_bytes(data) != item["prompt_token_sha256"]:
        raise Q2BError("primary token checksum/length differs")
    values = array.array("i")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    return values.tolist()


def _head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def run_worker(campaign: Path, configuration: str, execution_head: str) -> dict[str, Any]:
    if configuration not in CONFIGS or _head() != execution_head or os.environ.get("KVBENCH_QUALITY_IMAGE_DIGEST") != q0.QUALITY_IMAGE:
        raise Q2BError("worker scope/source/image differs")
    plan = load(campaign / "execution_plan.json")
    if sha(campaign / "execution_plan.json") != load(campaign / "entry.json")["plan_sha256"]:
        raise Q2BError("execution plan changed")
    import gc
    import torch
    from kvbench.runtime.backend import forced_flash_execution
    from kvbench.runtime.model_loader import load_frozen_model

    random.seed(plan["generation"]["seed"])
    torch.manual_seed(plan["generation"]["seed"])
    loaded = load_frozen_model(device=torch.device("cuda:0"))
    if int(loaded.tokenizer.eos_token_id) != plan["generation"]["eos_token_id"]:
        raise Q2BError("frozen EOS differs")
    completed = 0
    with torch.inference_mode(), forced_flash_execution(), gzip.open(TOKENS, "rb") as stream:
        for item in plan["entries"]:
            if accepted(campaign, configuration, item, plan) is not None:
                completed += 1
                continue
            root = q2a._unit(campaign, configuration, item)
            existing = sorted(root.glob("attempt-*"))
            if len(existing) > 1:
                raise Q2BError("bounded unit retry exceeded")
            if existing and (existing[0] / "COMPLETE").exists():
                previous = load(existing[0] / "result.json")
                if previous.get("failure_class") not in {"infrastructure_failed", "incomplete_output"}:
                    raise Q2BError("non-retryable prior failure")
            attempt_id = f"attempt-{len(existing):02d}"
            started = q0.utc_now()
            print(json.dumps({"event": "sample_started", "configuration": configuration,
                              "sample_id": item["sample_id"], "prompt_length": item["prompt_length"],
                              "completed": completed, "at_utc": started}), flush=True)
            try:
                prompt_ids = read_prompt(stream, item)
                result = q2a.generate_sample(loaded, configuration, item, prompt_ids, None)
                result.update({
                    "schema_version": "kvbench-q2b-generated-output-1.0.0",
                    "cot": False, "plan_sha256": sha(campaign / "execution_plan.json"),
                    "execution_head": execution_head, "attempt_id": attempt_id,
                    "replacement_of": existing[0].name if existing else None,
                    "contract_method_config_fingerprint": plan["method_fingerprints"][configuration],
                    "parser_source_sha256": plan["authority"]["parser_source_sha256"],
                    "generated_ids_sha256": q0.sha256_bytes(q0._i32_bytes(result["generated_token_ids"])),
                    "category": item["category"], "source_length_label": item["source_length_label"],
                    "reference_label_link": item["reference_label_link"], "dataset_revision": plan["dataset"]["revision"],
                    "input_manifest_sha256": plan["authority"]["input_manifest_sha256"],
                    "started_at_utc": started, "generated_at_utc": q0.utc_now(),
                    **parse_answer(result["generated_text"]),
                })
                validate_result(result, configuration, item, plan)
            except BaseException as error:
                q2a._write_attempt(campaign, configuration, item, {
                    "status": "FAILED", "configuration": configuration, "task": TASK,
                    "sample_id": item["sample_id"], "attempt_id": attempt_id,
                    "execution_head": execution_head, "failure_class": "runtime_or_correctness_failure",
                    "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
                    "retry_permitted": False, "started_at_utc": started,
                }, attempt_id)
                raise
            # Generation and report finalization are deliberately separate. A
            # failed write must not cause a valid wrong/invalid answer to rerun.
            q2a._write_attempt(campaign, configuration, item, result, attempt_id)
            completed += 1
            print(json.dumps({"event": "sample_completed", "configuration": configuration,
                              "sample_id": item["sample_id"], "completed": completed,
                              "total": 321, "at_utc": q0.utc_now()}), flush=True)
            del result, prompt_ids
            gc.collect()
            torch.cuda.empty_cache()  # existing Q2A between-sample lifecycle, never decode
    return {"configuration": configuration, "completed": completed}


def worker_command(campaign: Path, configuration: str, execution_head: str) -> list[str]:
    command = q2a._worker_command(campaign, configuration, execution_head)
    old = f"/home/rockrock/cmu_paper/artifacts/q2a/{campaign.name}"
    new = f"/home/rockrock/cmu_paper/artifacts/q2b/{campaign.name}"
    return [value.replace(old, new).replace("scripts.q2a_longbench_e", "scripts.q2b_longbench_v2") for value in command]


def start_campaign(campaign: Path, execution_head: str):
    if campaign.exists() or _head() != execution_head:
        raise Q2BError("campaign ID already exists or execution HEAD differs")
    plan = frozen_plan()
    protected = q0.verify_locked_paths()  # one targeted comparison, not historical validation
    campaign.mkdir(parents=True)
    q0.write_json_new(campaign / "execution_plan.json", plan)
    q0.write_json_new(campaign / "entry.json", {
        "schema_version": "kvbench-q2b-entry-1.0.0", "status": "PASS",
        "starting_head": STARTING_HEAD, "execution_head": execution_head,
        "plan_sha256": sha(campaign / "execution_plan.json"), "protected_paths": protected,
        "quality_image_digest": q0.QUALITY_IMAGE, "operator_prompt_sha256": sha(PROMPT),
        "q2a_generation_source_sha256": sha(ROOT / "scripts/q2a_longbench_e.py"),
        "q2b_glue_sha256": sha(Path(__file__)),
    })
    q0.write_json_new(campaign / "execution_authorization.json", {
        "schema_version": "kvbench-q2b-execution-authorization-1.0.0", "status": "APPROVED",
        "approved_by": "operator", "recorded_at_utc": q0.utc_now(),
        "authorization_scope": ["Q2B_B1_LONGBENCH_V2_PRIMARY_NO_COT"],
        "authorized_configurations": list(CONFIGS), "physical_batch_size": 1,
        "contract_id": q0.CONTRACT_ID, "contract_sha256": q0.CONTRACT_SHA256,
        "amendment_id": q1a.AMENDMENT_ID, "amendment_sha256": q1a.AMENDMENT_SHA256,
        "approval_text": "Execute Q2B exactly per Q2B_B1_LONGBENCH_V2_NO_COT_PROMPT.md: only bf16 + k4v4, B=1, LongBench-v2 no-CoT. Do not start Q2C, KVQuant diagnosis, Q3, or Q4. Operator authorizes this task.",
        "operator_prompt_sha256": sha(PROMPT), "frozen_contract_modified": False,
    })
    q0.write_json_new(campaign / "primary_eligibility.json", {"records": plan["eligibility_records"]})


def run_campaign(campaign: Path, execution_head: str):
    if not campaign.exists():
        start_campaign(campaign, execution_head)
    if (campaign / "COMPLETE").exists() or _head() != execution_head:
        raise Q2BError("finalized campaign or changed source cannot execute")
    plan = load(campaign / "execution_plan.json")
    with q0.gpu_lock():
        for configuration in CONFIGS:
            if all(accepted(campaign, configuration, item, plan) is not None for item in plan["entries"]):
                continue
            q0._gpu_idle()
            root = campaign / "worker_attempts" / configuration
            previous = sorted(root.glob("attempt-*"))
            if len(previous) > 1:
                raise Q2BError("bounded worker retry exceeded")
            attempt = root / f"attempt-{len(previous):02d}"
            attempt.mkdir(parents=True)
            command = worker_command(campaign, configuration, execution_head)
            q0.write_json_new(attempt / "started.json", {"at_utc": q0.utc_now(), "execution_head": execution_head,
                "configuration": configuration, "command": command, "replacement_of": previous[0].name if previous else None})
            with (attempt / "stdout.txt").open("xb") as stdout, (attempt / "stderr.txt").open("xb") as stderr:
                process = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr, check=False)
            q0.write_json_new(attempt / "attempt.json", {"returncode": process.returncode,
                "configuration": configuration, "execution_head": execution_head, "finished_at_utc": q0.utc_now()})
            # Accepted outputs survive a post-worker query failure.
            if process.returncode != 0:
                raise Q2BError(f"worker failed: {configuration}; preserve outputs and inspect failure before retry")
            q0._gpu_idle()


def contingency(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    counts = {f"n{b}{m}": sum(int(r["bf16_correct"]) == b and int(r["k4v4_correct"]) == m for r in rows)
              for b in (0, 1) for m in (0, 1)}
    base = counts["n11"] + counts["n10"]
    method = counts["n11"] + counts["n01"]
    return {**counts, "pairs": n,
        "bf16_accuracy": base / n if n else None, "k4v4_accuracy": method / n if n else None,
        "accuracy_drop_pp": 100 * (counts["n10"] - counts["n01"]) / n if n else None,
        "correct_to_wrong": counts["n10"], "wrong_to_correct": counts["n01"],
        "retention": counts["n11"] / base if base else None, "retention_denominator": base,
    }


def percentile_interval(values: Sequence[float]):
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    return [ordered[int(0.025 * n)], ordered[min(n - 1, int(0.975 * n))]]


def paired_summary(rows: Sequence[Mapping[str, Any]], *, draws: int = 10000, seed: int = 20260722) -> dict[str, Any]:
    result = contingency(rows)
    if not rows:
        return {**result, "status": "unrepresented", "paired_ci95_pp": None, "retention_ci95": None}
    strata = defaultdict(list)
    for row in rows:
        strata[(row["length_bucket"], row["category"])].append(row)
    rng = random.Random(seed)
    drops, retentions, bases, methods = [], [], [], []
    zero = 0
    for _ in range(draws):
        sample = [group[rng.randrange(len(group))] for _, group in sorted(strata.items()) for _ in group]
        s = contingency(sample)
        drops.append(s["accuracy_drop_pp"])
        bases.append(s["bf16_accuracy"])
        methods.append(s["k4v4_accuracy"])
        if s["retention"] is None:
            zero += 1
        else:
            retentions.append(s["retention"])
    invalid = {c: sum(bool(r[f"{c}_invalid"]) for r in rows) for c in CONFIGS}
    return {**result, "status": "observed", "paired_ci95_pp": percentile_interval(drops),
        "bf16_accuracy_ci95": percentile_interval(bases), "k4v4_accuracy_ci95": percentile_interval(methods),
        "retention_ci95": None if zero else percentile_interval(retentions),
        "retention_zero_denominator_draws": zero, "bootstrap_draws": draws, "bootstrap_seed": seed,
        "bootstrap_strata": [{"length_bucket": k[0], "category": k[1], "count": len(v)} for k, v in sorted(strata.items())],
        "invalid_counts": invalid, "invalid_rates": {c: invalid[c] / len(rows) for c in CONFIGS},
        "invalid_increase_pp": 100 * (invalid["k4v4"] - invalid["bf16"]) / len(rows),
    }


def classify_gate(overall, categories, lengths, *, complete: bool, margins):
    if not complete or overall.get("retention") is None or overall.get("paired_ci95_pp") is None:
        return "INCONCLUSIVE", ["incomplete_pairing_or_undefined_statistic"]
    reasons = []
    for kind, groups, limit in (("category", categories, margins["category_hard_fail_percentage_points"]),
                               ("length", lengths, margins["length_hard_fail_percentage_points"])):
        for row in groups:
            if row["pairs"] and row["accuracy_drop_pp"] > limit:
                reasons.append(f"{kind}_hard_fail:{row['name']}")
    if overall["invalid_increase_pp"] > margins["invalid_output_increase_max_percentage_points"]:
        reasons.append("invalid_increase_hard_fail")
    if overall["retention"] < margins["bf16_correct_retention_min"]:
        reasons.append("retention_point_hard_fail")
    ci = overall["paired_ci95_pp"]
    if ci[0] > margins["accuracy_drop_max_percentage_points"]:
        reasons.append("accuracy_lower_ci_exceeds_margin")
    if reasons:
        return "FAIL", reasons
    if ci[1] <= margins["accuracy_drop_max_percentage_points"]:
        return "PASS", ["accuracy_upper_ci_within_margin", "all_hard_guardrails_pass"]
    return "INCONCLUSIVE", ["accuracy_ci_crosses_margin"]


def analyze(campaign: Path) -> dict[str, Any]:
    plan = load(campaign / "execution_plan.json")
    gold = {r["sample_id"]: r["source"]["answer"] for r in source_rows() if r["eligible"]}
    paired, index, missing = [], [], []
    for item in plan["entries"]:
        scored = {}
        for configuration in CONFIGS:
            result = accepted(campaign, configuration, item, plan)
            if result is None:
                missing.append({"configuration": configuration, "sample_id": item["sample_id"], "reason": "missing_execution"})
                continue
            answer = parse_answer(result["generated_text"])
            scored.update({f"{configuration}_correct": answer["parsed_answer"] == gold[item["sample_id"]],
                f"{configuration}_invalid": answer["invalid"], f"{configuration}_answer": answer["parsed_answer"],
                f"{configuration}_generated_length": result["generated_token_count"],
                f"{configuration}_stop_reason": result["stop_reason"]})
            index.append({"configuration": configuration, "sample_id": item["sample_id"],
                "attempt_id": result["attempt_id"], "replacement_of": result["replacement_of"],
                "execution_head": result["execution_head"],
                "result_sha256": sha(q2a._unit(campaign, configuration, item) / result["attempt_id"] / "result.json"),
                "correct": scored[f"{configuration}_correct"], **answer})
        if len(scored) == 10:
            paired.append({"sample_id": item["sample_id"], "category": item["category"],
                "length_bucket": item["length_bucket"], "prompt_length": item["prompt_length"],
                "reference_answer": gold[item["sample_id"]], **scored})
    complete = len(paired) == 321 and len(index) == 642 and not missing
    b = plan["margins"]["bootstrap"]
    kwargs = {"draws": b["draws"], "seed": b["seed"]}
    # No primary pass or reduced-denominator primary estimate on incomplete data.
    overall = paired_summary(paired, **kwargs) if complete else {}
    categories = [{"name": name, **paired_summary([r for r in paired if r["category"] == name], **kwargs)}
                  for name in sorted({r["category"] for r in plan["entries"]})] if complete else []
    lengths = [{"name": name, **paired_summary([r for r in paired if r["length_bucket"] == name], **kwargs)}
               for name in BUCKETS] if complete else []
    gate, reasons = classify_gate(overall, categories, lengths, complete=complete, margins=plan["margins"]["longbench_v2"])
    generation = {}
    for configuration in CONFIGS:
        values = [r[f"{configuration}_generated_length"] for r in paired]
        generation[configuration] = {"paired_outputs": len(values),
            "mean_tokens": statistics.mean(values) if values else None,
            "median_tokens": statistics.median(values) if values else None,
            "stop_reasons": dict(Counter(r[f"{configuration}_stop_reason"] for r in paired))}
    summary = {
        "schema_version": "kvbench-q2b-summary-1.0.0", "execution_status": "COMPLETE" if complete else "PARTIAL",
        "scientific_verdict": gate, "gate_reasons": reasons, "selected": 503, "eligible": 321, "excluded": 182,
        "planned_outputs": 642, "completed_outputs": len(index), "missing_outputs": len(missing), "paired_count": len(paired),
        "overall": overall, "category_results": categories, "length_results": lengths, "generation": generation,
        "margins": plan["margins"]["longbench_v2"], "inference_replacements": sum(r["replacement_of"] is not None for r in index),
        "original_cross_batch_gate": "FAILED", "physical_batch_size": 1, "quality_transfer_to_other_batches": "not_established",
        "q2c_readiness": "eligible_for_separate_authorization" if gate == "PASS" else "not_qualified_by_Q2B",
        "q2c_started": False, "q3_started": False, "q4_started": False,
    }
    return {"summary": summary, "paired_scores": paired, "output_index": index, "missing": missing}


def write_idempotent(path: Path, value: Any):
    data = q0.canonical_bytes(value)
    if path.exists():
        if path.read_bytes() != data:
            raise Q2BError(f"existing finalized payload differs: {path.name}")
    else:
        q0.write_new(path, data)


def finalize(campaign: Path, execution_head: str):
    if (campaign / "COMPLETE").exists():
        raise Q2BError("campaign already finalized")
    analysis = analyze(campaign)
    summary = analysis["summary"]
    if summary["execution_status"] != "COMPLETE":
        raise Q2BError("cannot seal incomplete Q2B as complete")
    for key, value in analysis.items():
        write_idempotent(campaign / f"{key}.json", value if isinstance(value, dict) else {"rows": value})
    s = summary["overall"]
    report = (f"# Q2B B=1 LongBench-v2 primary no-CoT\n\nExecution COMPLETE; scientific {summary['scientific_verdict']}.\n"
              f"321 paired samples; 642 outputs; 182 predeclared length exclusions.\n"
              f"BF16 accuracy {s['bf16_accuracy']:.6f}; k4v4 {s['k4v4_accuracy']:.6f}; drop {s['accuracy_drop_pp']:.6f} pp, CI95 {s['paired_ci95_pp']}.\n"
              f"Retention {s['retention']}; CI95 {s['retention_ci95']}; denominator {s['retention_denominator']}.\n"
              f"Gate reasons: {summary['gate_reasons']}.\n"
              "Category/length results and all guardrails are in summary.json; scores in paired_scores.json.\n"
              "Original cross-batch gate FAILED; no B>1 quality transfer. Q2C, Q3, Q4 not started.\n")
    path = campaign / "q2b_report.md"
    if path.exists():
        if path.read_text() != report:
            raise Q2BError("existing report differs")
    else:
        q0.write_new(path, report.encode())
    write_idempotent(campaign / "manifest.json", {
        "schema_version": "kvbench-q2b-campaign-1.0.0", "run_id": campaign.name,
        "starting_head": STARTING_HEAD, "execution_head": execution_head, "quality_image_digest": q0.QUALITY_IMAGE,
        "contract_id": q0.CONTRACT_ID, "contract_sha256": q0.CONTRACT_SHA256,
        "amendment_id": q1a.AMENDMENT_ID, "amendment_sha256": q1a.AMENDMENT_SHA256,
        "status": "COMPLETE", "scientific_verdict": summary["scientific_verdict"], "physical_batch_size": 1,
        "source_roots": {"qp1": q0.QP1_ROOT, "q1b": q2a.Q1B_ROOT, "q2a": Q2A_ROOT},
    })
    payloads = sorted(p for p in campaign.rglob("*") if p.is_file() and p.name not in {"artifact_inventory.json", "checksums.sha256"})
    write_idempotent(campaign / "artifact_inventory.json", {
        "schema_version": "kvbench-artifact-inventory-1.0.0", "run_id": campaign.name,
        "files": [{"path": p.relative_to(campaign).as_posix(), "role": "manifest" if p.name == "manifest.json" else "q2b_evidence",
                   "size_bytes": p.stat().st_size, "sha256": sha(p)} for p in payloads],
        "excluded_control_files": ["artifact_inventory.json", "checksums.sha256", "COMPLETE"],
    })
    ledger = "".join(f"{sha(p)}  {p.relative_to(campaign).as_posix()}\n" for p in q0._checksum_ledger_payloads(campaign)).encode()
    path = campaign / "checksums.sha256"
    if path.exists():
        if path.read_bytes() != ledger:
            raise Q2BError("checksum ledger differs")
    else:
        q0.write_new(path, ledger)
    q0.write_json_new(campaign / "COMPLETE", {
        "schema_version": "kvbench-q2b-complete-1.0.0", "run_id": campaign.name, "status": "COMPLETE",
        "manifest_sha256": sha(campaign / "manifest.json"), "artifact_inventory_sha256": sha(campaign / "artifact_inventory.json"),
        "checksum_ledger_sha256": sha(path), "checksum_ledger_path": "checksums.sha256", "written_last": True,
    })
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--run-worker", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--configuration", choices=CONFIGS)
    parser.add_argument("--execution-head")
    args = parser.parse_args()
    if args.plan:
        p = frozen_plan()
        print(json.dumps({k: p[k] for k in ("configurations", "selected_count", "planned_pairs", "planned_outputs", "excluded_count")}))
        return
    if not args.campaign or not args.execution_head:
        parser.error("campaign and execution HEAD required")
    if args.start:
        start_campaign(args.campaign, args.execution_head)
    elif args.run_worker:
        print(json.dumps(run_worker(args.campaign, args.configuration, args.execution_head)))
    elif args.run:
        run_campaign(args.campaign, args.execution_head)
    elif args.analyze:
        print(json.dumps(analyze(args.campaign)["summary"]))
    elif args.finalize:
        print(json.dumps(finalize(args.campaign, args.execution_head)))
    else:
        parser.error("select an action")


if __name__ == "__main__":
    main()
