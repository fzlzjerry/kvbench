#!/usr/bin/env python3
"""Materialize and validate the QP-1 quality contract without model inference."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import shutil
import struct
import sys
import urllib.request
import zipfile

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".qp1-cache"
TOKENIZER = CACHE / "tokenizer-public"
DATA = CACHE / "data"
SOURCES = CACHE / "sources"
SEED = 20260722
MODEL_MAX = 131072
CONTRACT_ID = "quality-qp1-20260918t170812117127z-170b638c-e8f4a2"
TOKENIZER_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
LONG_BENCH_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
LONG_BENCH_V2_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
LONG_BENCH_SOURCE_COMMIT = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
C4_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
TASKS = [
    "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report",
    "multi_news", "trec", "triviaqa", "samsum", "passage_count",
    "passage_retrieval_en", "lcc", "repobench-p",
]
NO_CHAT = {"trec", "triviaqa", "samsum", "lcc", "repobench-p"}
CONFIG_IDS = [
    "bf16", "tq_4bit_nc", "tq_k3v4_nc", "tq_3bit_nc", "k4v4",
    "k2v4", "k2v2", "kvq4", "kvq3", "kvq2",
]


class QP1Error(RuntimeError):
    pass


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != value:
            raise QP1Error(f"resume would rewrite completed unit: {path}")
        return
    path.write_bytes(value)


def write_json(path: Path, value: object) -> None:
    write(path, canonical(value))


def gzip_bytes(value: bytes) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as handle:
        handle.write(value)
    return output.getvalue()


def write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    raw = b"".join(
        (json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                    allow_nan=False) + "\n").encode()
        for row in rows
    )
    write(path, gzip_bytes(raw))


def write_i32_gz(path: Path, values: list[int]) -> None:
    array = np.asarray(values, dtype="<i4")
    write(path, gzip_bytes(array.tobytes(order="C")))


def fetch_c4_frame() -> Path:
    destination = DATA / "c4-validation-frame-000000-004095.jsonl.gz"
    if destination.exists():
        return destination
    rows: list[dict] = []
    for offset in range(0, 4096, 100):
        length = min(100, 4096 - offset)
        url = (
            "https://datasets-server.huggingface.co/rows?dataset=allenai/c4"
            f"&config=en&split=validation&offset={offset}&length={length}"
        )
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = json.load(response)
        if payload.get("num_rows_total") != 364608 or not isinstance(payload.get("partial"), bool):
            raise QP1Error("C4 Dataset Viewer frame identity changed")
        for item in payload["rows"]:
            if item.get("truncated_cells"):
                raise QP1Error("C4 Dataset Viewer returned truncated text")
            row = item["row"]
            rows.append({"row_idx": item["row_idx"], "text": row["text"],
                         "timestamp": row["timestamp"], "url": row["url"]})
    if [row["row_idx"] for row in rows] != list(range(4096)):
        raise QP1Error("C4 sampling frame is incomplete")
    write_jsonl_gz(destination, rows)
    return destination


def read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def tokenizer() -> object:
    required = {
        "tokenizer.json": "79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4",
        "tokenizer_config.json": "177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424",
        "special_tokens_map.json": "6f38c73729248f6c127296386e3cdde96e254636cc58b4169d3fd32328d9a8ec",
    }
    for name, digest in required.items():
        if sha256_file(TOKENIZER / name) != digest:
            raise QP1Error(f"frozen tokenizer asset mismatch: {name}")
    return AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)


def ppl_stream(tok: object, documents: list[tuple[str, str]]) -> tuple[list[int], list[dict]]:
    stream: list[int] = []
    spans: list[dict] = []
    for document_id, text in documents:
        if not text.strip():
            continue
        start = len(stream)
        ids = tok.encode(text, add_special_tokens=False)
        stream.extend(ids)
        end = len(stream)
        stream.append(tok.eos_token_id)
        spans.append({"document_id": document_id, "start": start, "end": end,
                      "separator_index": end, "token_count": len(ids)})
    return stream, spans


def choose_anchors(spans: list[dict], length: int, horizon: int, count: int,
                   seed: int) -> list[dict]:
    candidates: list[tuple[int, str, int, int]] = []
    stride = max(1, horizon)
    for span in spans:
        low = max(span["start"] + 32, length + 1)
        high = span["end"] - horizon
        for target in range(low, high + 1, stride):
            start = target - length - 1
            candidates.append((target, span["document_id"], start, span["end"]))
    if len(candidates) < count:
        raise QP1Error(f"insufficient anchors for L={length}: {len(candidates)}")
    rng = random.Random(seed)
    chosen = rng.sample(candidates, count)
    chosen.sort()
    return [{
        "anchor_id": f"L{length}-t{target}", "source_document_id": doc,
        "stream_offset": start, "prefix_start": start,
        "prefix_end": start + length, "burn_in_index": target - 1,
        "target_start": target, "target_end_exclusive": target + horizon,
        "scored_token_count": horizon, "cluster_id": doc,
        "boundary_policy": "target_horizon_must_not_cross_document_eos",
        "scoring_mask": {"value": 1, "length": horizon},
    } for target, doc, start, _ in chosen]


def render_chat(tok: object, text: str) -> tuple[str, list[int]]:
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True,
    )
    ids = tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=True,
        add_generation_prompt=True,
    )
    return rendered, list(ids)


def write_tokenized_records(base: Path, rows: list[dict], prompts: list[str],
                            ids_list: list[list[int]]) -> dict:
    if not (len(rows) == len(prompts) == len(ids_list)):
        raise QP1Error("record/token count mismatch")
    offsets: list[dict] = []
    flat: list[int] = []
    stored: list[dict] = []
    for row, prompt, ids in zip(rows, prompts, ids_list):
        start = len(flat)
        flat.extend(ids)
        sample_id = row["sample_id"]
        offsets.append({"sample_id": sample_id, "offset": start, "length": len(ids),
                        "token_ids_sha256": sha256_bytes(np.asarray(ids, dtype="<i4").tobytes())})
        stored.append({**row, "effective_prompt": prompt,
                       "prompt_text_sha256": sha256_bytes(prompt.encode()),
                       "token_ids_sha256": offsets[-1]["token_ids_sha256"],
                       "token_count": len(ids)})
    write_jsonl_gz(base.with_suffix(".jsonl.gz"), stored)
    write_i32_gz(base.with_suffix(".token_ids.i32.gz"), flat)
    write_json(base.with_suffix(".index.json"), {
        "schema_version": "kvbench-quality-token-index-1.0.0",
        "dtype": "little_endian_int32", "records": offsets,
        "total_tokens": len(flat),
    })
    return {"records": len(rows), "total_tokens": len(flat),
            "eligible": sum(bool(row.get("eligible", True)) for row in rows),
            "excluded": sum(not bool(row.get("eligible", True)) for row in rows)}


def dependency_lock() -> dict:
    installed = {
        "python": "3.12.3", "torch": "2.12.1+cu130", "triton": "3.7.1",
        "numpy": "2.5.1", "transformers": "4.57.6", "tokenizers": "0.22.2",
        "huggingface-hub": "0.36.2", "safetensors": "0.8.0",
        "jinja2": "3.1.6", "pyarrow": "25.0.0",
    }
    scoring = {
        "rouge": "1.0.1", "jieba": "0.42.1", "fuzzywuzzy": "0.18.0",
    }
    artifacts: dict[str, list[dict]] = {}
    for name, version in scoring.items():
        with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json") as response:
            payload = json.load(response)
        artifacts[name] = sorted([
            {"filename": item["filename"], "packagetype": item["packagetype"],
             "sha256": item["digests"]["sha256"], "size": item["size"]}
            for item in payload["urls"]
        ], key=lambda item: item["filename"])
    return {
        "schema_version": "kvbench-quality-dependency-lock-1.0.0",
        "quality_image_digest": "sha256:1d8056643bf001fb83ee19e6bb19ec93379c296f2be6ba02e907d0b2f4ecca32",
        "measurement_image_digest": "sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e",
        "measured_runtime_unchanged": True, "installed_cpu_preparation": installed,
        "official_metric_dependencies_locked_for_future_execution": scoring,
        "pypi_artifacts": artifacts, "model_weights_loaded": False,
        "cuda_initialized": False,
    }


def prepare(contract_id: str) -> Path:
    if contract_id != CONTRACT_ID:
        raise QP1Error("unexpected contract ID")
    final = ROOT / "artifacts" / "quality_contract" / contract_id
    stage = final.with_name(final.name + ".staging")
    if final.exists():
        raise QP1Error("contract artifact already exists")
    stage.mkdir(parents=True, exist_ok=True)
    tok = tokenizer()
    c4_frame = fetch_c4_frame()
    selected = stage / "selected_inputs"

    # PPL streams and deterministic anchors.
    wiki_table = pq.read_table(DATA / "wikitext-test.parquet", columns=["text"])
    wiki_docs = [(f"wikitext-test-{index:05d}", text.as_py())
                 for index, text in enumerate(wiki_table.column("text"))]
    c4_rows = read_jsonl_gz(c4_frame)
    chosen_c4_ids = sorted(random.Random(SEED).sample(range(4096), 2048))
    c4_docs = [(f"c4-validation-row-{row['row_idx']:06d}", row["text"])
               for row in c4_rows if row["row_idx"] in set(chosen_c4_ids)]
    ppl_manifest = {"schema_version": "kvbench-quality-ppl-fixtures-1.0.0",
                    "seed": SEED, "burn_in_tokens": 1,
                    "ignored_tokens_after_document_boundary": 32,
                    "document_separator_token_id": tok.eos_token_id,
                    "datasets": {}}
    for dataset_name, docs in [("wikitext2_test", wiki_docs), ("c4_validation", c4_docs)]:
        stream, spans = ppl_stream(tok, docs)
        base = selected / "ppl" / dataset_name
        write_i32_gz(base / "token_stream.i32.gz", stream)
        write_json(base / "document_spans.json", {"documents": spans})
        stages: dict[str, dict] = {}
        for stage_name, lengths, count, horizon in [
            ("fast", [4096, 24576, 32768, 65536], 16, 128),
            ("full", [4096, 16384, 24576, 28672, 32768, 65536, 98304, 130560], 64, 256),
        ]:
            stages[stage_name] = {}
            for length in lengths:
                anchors = choose_anchors(spans, length, horizon, count,
                                         SEED + length + (0 if stage_name == "fast" else 1_000_000))
                stages[stage_name][str(length)] = anchors
        ppl_manifest["datasets"][dataset_name] = {
            "document_count": len(spans), "stream_token_count": len(stream),
            "stream_sha256": sha256_bytes(np.asarray(stream, dtype="<i4").tobytes()),
            "stages": stages,
        }
    write_json(selected / "ppl" / "anchor_manifest.json", ppl_manifest)
    write_json(selected / "ppl" / "c4_selection.json", {
        "source_revision": C4_REVISION, "config": "en", "split": "validation",
        "sampling_frame": {"first_row_inclusive": 0, "last_row_inclusive": 4095,
                           "frame_size": 4096, "total_validation_rows": 364608,
                           "viewer_partial": True},
        "selection_seed": SEED, "selected_count": 2048,
        "row_indices": chosen_c4_ids,
        "frame_file_sha256": sha256_file(c4_frame),
        "scope_note": "uniform without replacement within the declared first-4096-row partial-viewer frame, not all C4 validation",
    })

    # LongBench-E fixtures.
    prompts = json.load(open(SOURCES / "LongBench/config/dataset2prompt.json"))
    maxlens = json.load(open(SOURCES / "LongBench/config/dataset2maxlen.json"))
    lb_counts: dict[str, dict] = {}
    with zipfile.ZipFile(DATA / "longbench-data.zip") as archive:
        for task in TASKS:
            source_rows = [json.loads(line) for line in archive.read(f"data/{task}_e.jsonl").decode().split("\n") if line]
            rows: list[dict] = []
            rendered: list[str] = []
            ids_list: list[list[int]] = []
            for source in source_rows:
                plain = prompts[task].format(**source)
                if task in NO_CHAT:
                    effective = plain
                    ids = list(tok.encode(plain, add_special_tokens=True))
                    wrapper = "official_exception_no_chat_wrapper"
                else:
                    effective, ids = render_chat(tok, plain)
                    wrapper = "frozen_llama31_chat_template"
                budget = int(maxlens[task])
                eligible = len(ids) + budget + 128 <= MODEL_MAX
                rows.append({
                    "sample_id": source["_id"], "task": task,
                    "source": source, "generation_budget": budget,
                    "safety_margin": 128, "eligible": eligible,
                    "exclusion_reason": None if eligible else "model_length_budget",
                    "formatting_policy": wrapper, "conditioning_tokens": 16,
                    "prefill_token_count": max(0, len(ids) - 16),
                    "length_bucket": "0-4k" if source["length"] < 4000 else
                                     "4-8k" if source["length"] < 8000 else "8k+",
                })
                rendered.append(effective); ids_list.append(ids)
            lb_counts[task] = write_tokenized_records(selected / "longbench_e" / task,
                                                       rows, rendered, ids_list)

    # LongBench v2 primary and preregistered CoT-stress subset.
    v2 = json.load(open(DATA / "longbench-v2-data.json"))
    no_cot_template = (SOURCES / "prompts/0shot.txt").read_text()
    cot_template = (SOURCES / "prompts/0shot_cot.txt").read_text()
    v2_rows: list[dict] = []
    v2_prompts: list[str] = []
    v2_ids: list[list[int]] = []
    for item in v2:
        plain = (no_cot_template.replace("$DOC$", item["context"].strip())
                 .replace("$Q$", item["question"].strip())
                 .replace("$C_A$", item["choice_A"].strip())
                 .replace("$C_B$", item["choice_B"].strip())
                 .replace("$C_C$", item["choice_C"].strip())
                 .replace("$C_D$", item["choice_D"].strip()))
        effective, ids = render_chat(tok, plain)
        n = len(ids)
        if n < 16384: bucket = "8-16k"
        elif n < 32768: bucket = "16-32k"
        elif n < 65536: bucket = "32-64k"
        else: bucket = "64-128k"
        eligible = n + 8 + 128 <= MODEL_MAX
        v2_rows.append({"sample_id": item["_id"], "source": item,
                        "generation_budget": 8, "safety_margin": 128,
                        "eligible": eligible,
                        "exclusion_reason": None if eligible else "model_length_budget",
                        "length_bucket": bucket, "conditioning_tokens": 16,
                        "prefill_token_count": max(0, n - 16),
                        "parser": "official_A_B_C_D_extract_answer"})
        v2_prompts.append(effective); v2_ids.append(ids)
    v2_counts = write_tokenized_records(selected / "longbench_v2" / "primary_no_cot",
                                         v2_rows, v2_prompts, v2_ids)
    strata: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(v2_rows):
        strata.setdefault((row["source"]["length"], row["source"]["difficulty"]), []).append(index)
    rng = random.Random(SEED)
    subset: list[int] = []
    for key in sorted(strata):
        indices = list(strata[key]); rng.shuffle(indices)
        subset.extend(indices[:math.ceil(128 * len(indices) / len(v2_rows))])
    subset = sorted(subset[:128], key=lambda index: v2_rows[index]["sample_id"])
    cot_rows: list[dict] = []
    cot_prompts: list[str] = []
    cot_ids: list[list[int]] = []
    for index in subset:
        item = v2_rows[index]["source"]
        plain = (cot_template.replace("$DOC$", item["context"].strip())
                 .replace("$Q$", item["question"].strip())
                 .replace("$C_A$", item["choice_A"].strip())
                 .replace("$C_B$", item["choice_B"].strip())
                 .replace("$C_C$", item["choice_C"].strip())
                 .replace("$C_D$", item["choice_D"].strip()))
        effective, ids = render_chat(tok, plain)
        eligible = len(ids) + 1024 + 128 <= MODEL_MAX
        cot_rows.append({"sample_id": item["_id"], "source": item,
                         "generation_budget": 1024, "answer_stage_budget": 8,
                         "safety_margin": 128, "eligible": eligible,
                         "exclusion_reason": None if eligible else "model_length_budget",
                         "selection_rule": "seeded_stratified_length_label_x_difficulty_128"})
        cot_prompts.append(effective); cot_ids.append(ids)
    cot_counts = write_tokenized_records(selected / "longbench_v2" / "cot_stress_subset",
                                          cot_rows, cot_prompts, cot_ids)

    dep_lock = dependency_lock()
    source_lock = {
        "schema_version": "kvbench-quality-dataset-lock-1.0.0",
        "wikitext2": {"repository": "Salesforce/wikitext", "revision": WIKITEXT_REVISION,
                      "config": "wikitext-2-raw-v1", "split": "test",
                      "file": "wikitext-2-raw-v1/test-00000-of-00001.parquet",
                      "sha256": sha256_file(DATA / "wikitext-test.parquet")},
        "c4": {"repository": "allenai/c4", "revision": C4_REVISION,
               "config": "en", "split": "validation", "selected_frame": "rows 0..4095",
               "selected_count": 2048, "seed": SEED},
        "longbench_e": {"repository": "zai-org/LongBench", "revision": LONG_BENCH_REVISION,
                        "source_file": "data.zip", "sha256": sha256_file(DATA / "longbench-data.zip"),
                        "tasks": TASKS, "splits": [f"{task}_e/test" for task in TASKS]},
        "longbench_v2": {"repository": "zai-org/LongBench-v2", "revision": LONG_BENCH_V2_REVISION,
                         "source_file": "data.json", "sha256": sha256_file(DATA / "longbench-v2-data.json"),
                         "config": "default", "split": "train"},
        "benchmark_source": {"repository": "THUDM/LongBench", "commit": LONG_BENCH_SOURCE_COMMIT},
    }
    prompt_files = sorted(path for path in SOURCES.rglob("*") if path.is_file())
    prompt_lock = {
        "schema_version": "kvbench-quality-prompt-lock-1.0.0",
        "benchmark_source_commit": LONG_BENCH_SOURCE_COMMIT,
        "files": [{"path": path.relative_to(SOURCES).as_posix(), "sha256": sha256_file(path)}
                  for path in prompt_files],
        "tokenizer": {"repository": "meta-llama/Llama-3.1-8B-Instruct",
                      "revision": TOKENIZER_REVISION,
                      "tokenizer_json_sha256": sha256_file(TOKENIZER / "tokenizer.json"),
                      "tokenizer_config_sha256": sha256_file(TOKENIZER / "tokenizer_config.json"),
                      "chat_template_sha256": sha256_bytes(tok.chat_template.encode()),
                      "bos_token_id": tok.bos_token_id, "eos_token_id": tok.eos_token_id,
                      "add_generation_prompt": True,
                      "asset_recovery_mirror": {"repository": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
                                                "commit": "db1f81ad4b8c7e39777509fac66c652eb0a52f91",
                                                "verified_by_frozen_sha256": True}},
        "longbench_e_no_chat_wrapper_tasks": sorted(NO_CHAT),
    }
    fixture_manifest = {
        "schema_version": "kvbench-quality-fixture-manifest-1.0.0",
        "ppl": {name: {"documents": value["document_count"],
                       "stream_tokens": value["stream_token_count"],
                       "fast_anchor_records": 4 * 16, "full_anchor_records": 8 * 64}
                for name, value in ppl_manifest["datasets"].items()},
        "longbench_e": lb_counts,
        "longbench_v2_primary": v2_counts,
        "longbench_v2_cot_stress": cot_counts,
        "method_independent": True, "model_outputs_present": False,
        "scores_present": False, "model_weights_present": False,
    }
    gates = {
        "schema_version": "kvbench-quality-margins-1.0.0",
        "decision_states": ["pass", "fail", "inconclusive"],
        "bootstrap": {"confidence_level": 0.95, "draws": 10000, "seed": SEED,
                      "ppl_unit": "paired_anchor_clustered_by_source_document",
                      "longbench_e_unit": "paired_sample_stratified_by_task",
                      "longbench_v2_unit": "paired_sample_stratified_by_length_and_category",
                      "insufficient_sample_state": "inconclusive",
                      "extension_budget": "one preregistered additional equal-size sample only after human approval"},
        "ppl": {"global_relative_ppl_increase_max": 0.01,
                "length_review_relative_increase": 0.02,
                "length_hard_fail_relative_increase": 0.05,
                "primary_comparison": "paired_delta_nll",
                "reported": ["nll", "delta_nll", "exp(delta_nll)-1"]},
        "longbench_e": {"macro_drop_max_score_points": 2.0,
                        "category_review_score_points": 3.0,
                        "category_hard_fail_score_points": 5.0,
                        "invalid_output_increase_max_percentage_points": 1.0,
                        "aggregation": "equal_task_macro"},
        "longbench_v2": {"accuracy_drop_max_percentage_points": 2.0,
                         "category_hard_fail_percentage_points": 5.0,
                         "length_hard_fail_percentage_points": 5.0,
                         "invalid_output_increase_max_percentage_points": 1.0,
                         "bf16_correct_retention_min": 0.95},
    }
    resolved = {
        "chronology": "resolved in QP-1 after performance results, before any quality outputs",
        "c4_sampling_frame": "first 4096 validation rows; seeded sample 2048 without replacement",
        "ppl_boundary": "reject candidate if scored horizon crosses document EOS",
        "bootstrap": "10000 draws, seed 20260722, paired clustered/stratified units",
        "longbench_v2_cot_subset": "128 seeded samples stratified by source length label and difficulty",
        "longbench_v2_cot_budgets": {"reasoning": 1024, "answer": 8, "safety_margin": 128},
        "finalist_ties": "quality: LongBench-E macro then global PPL then lexical config ID; compression: r_alloc at B1/L131071 then lexical config ID",
        "r_alloc_comparison_geometry": {"batch": 1, "historical_context": 131071,
                                        "total_attended": 131072, "graph_mode": "cuda_graph"},
        "length_buckets": "left-inclusive/right-exclusive except 64-128k includes the model-limit endpoint",
    }

    config_path = ROOT / "configs/quality/quality_contract.yaml"
    qp0 = json.loads(config_path.read_text())["quality_contract"]
    qp0["id"] = contract_id
    qp0["status"] = "requires_human_approval"
    qp0["quality_execution"] = "LOCKED"
    qp0["approval"] = {"approved": False, "approved_at_utc": None, "approved_by": None}
    qp0["pending_qp1_items"] = ["formal_human_approval"]
    qp0["completed_qp1_items"] = ["quality_evaluation_dependency_lock", "chat_template_hash",
                                   "dataset_revisions", "sample_id_sets", "prompt_template_hashes",
                                   "dataset_and_prompt_materialization"]
    qp0["dependency_lock"] = dep_lock
    qp0["model"]["chat_template_hash"] = {"algorithm": "sha256",
                                                   "value": prompt_lock["tokenizer"]["chat_template_sha256"]}
    qp0["model"]["tokenizer_assets"] = prompt_lock["tokenizer"]
    qp0["datasets"] = source_lock
    qp0["fixture_manifest"] = fixture_manifest
    qp0["prompt_lock"] = prompt_lock
    qp0["gates"] = gates
    qp0["resolved_choices"] = resolved
    qp0["longbench_e"]["revision"] = source_lock["longbench_e"]
    qp0["longbench_e"]["sample_ids"] = {task: [record["sample_id"] for record in
        read_jsonl_gz(selected / "longbench_e" / f"{task}.jsonl.gz")] for task in TASKS}
    qp0["longbench_v2"]["revision"] = source_lock["longbench_v2"]
    qp0["longbench_v2"]["sample_ids"] = [row["sample_id"] for row in v2_rows]
    qp0["ppl"]["datasets"][0]["revision"] = source_lock["wikitext2"]
    qp0["ppl"]["datasets"][1]["revision"] = source_lock["c4"]
    qp0["ppl"]["datasets"][1]["sample_ids_file"] = "selected_inputs/ppl/c4_selection.json"
    contract = {"quality_contract": qp0}

    # Snapshot/config files are byte-identical and finalized before the digest is recorded.
    contract_bytes = canonical(contract)
    contract_sha = sha256_bytes(contract_bytes)
    write(stage / "contract_snapshot.yaml", contract_bytes)
    input_files = sorted(path for path in selected.rglob("*") if path.is_file())
    input_manifest = {
        "schema_version": "kvbench-quality-input-manifest-1.0.0",
        "contract_id": contract_id, "contract_sha256": contract_sha,
        "files": [{"path": path.relative_to(stage).as_posix(), "size_bytes": path.stat().st_size,
                   "sha256": sha256_file(path)} for path in input_files],
        "fixture_manifest": fixture_manifest,
    }
    write_json(stage / "input_manifest.json", input_manifest)
    write_json(stage / "dependency_lock.json", dep_lock)
    write_json(stage / "dataset_lock.json", source_lock)
    write_json(stage / "prompt_lock.json", prompt_lock)
    write_json(stage / "quality_margins.json", gates)
    write_json(stage / "resolved_choices.json", resolved)
    manifest = {
        "schema_version": "kvbench-qp1-quality-contract-bundle-1.0.0",
        "run_id": contract_id, "status": "PASS", "contract_sha256": contract_sha,
        "quality_execution": "LOCKED", "quality_scores_computed": False,
        "performance_freeze_tag": "perf-freeze-20260917-83536c37-r1",
        "performance_manifest_sha256": "91db28a33940e9bbdda6c723a2678ae9459e73f813b20bfecb6c97222ba38fc5",
        "performance_freeze_root": "9996171e9c0ee737dba15fb0609684e3574e4633ec2f841f2c660b48319d213f",
        "qp0_correction_root": "44c9594079e372d708d509449589a428f4b27c0392e2f1b1e491acc3d17e2559",
        "formal_human_approval": "pending",
    }
    write_json(stage / "manifest.json", manifest)

    # Copy final contract and compact config projections before bundle controls.
    config_path.write_bytes(contract_bytes)
    config_files = {
        ROOT / "configs/quality/dependencies.lock.json": dep_lock,
        ROOT / "configs/quality/datasets/dataset_lock.json": source_lock,
        ROOT / "configs/quality/prompts/prompt_lock.json": prompt_lock,
        ROOT / "configs/quality/fixtures/fixture_manifest.json": fixture_manifest,
        ROOT / "configs/quality/gates/quality_margins.yaml": gates,
    }
    for path, value in config_files.items(): write_json(path, value)

    payload = sorted(path for path in stage.rglob("*") if path.is_file())
    inventory_items = [{"path": path.relative_to(stage).as_posix(), "role": "quality_contract_evidence",
                        "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
                       for path in payload]
    write_json(stage / "artifact_inventory.json", {
        "schema_version": "kvbench-artifact-inventory-1.0.0", "run_id": contract_id,
        "files": inventory_items,
        "excluded_control_files": ["artifact_inventory.json", "checksums.sha256", "COMPLETE"],
    })
    ledger_paths = sorted(path for path in stage.rglob("*") if path.is_file()
                          and path.name not in {"checksums.sha256", "COMPLETE"})
    ledger = "".join(f"{sha256_file(path)}  {path.relative_to(stage).as_posix()}\n" for path in ledger_paths)
    write(stage / "checksums.sha256", ledger.encode())
    write_json(stage / "COMPLETE", {
        "run_id": contract_id, "status": "PASS", "written_last": True,
        "manifest_sha256": sha256_file(stage / "manifest.json"),
        "artifact_inventory_sha256": sha256_file(stage / "artifact_inventory.json"),
        "checksum_ledger_sha256": sha256_file(stage / "checksums.sha256"),
        "checksum_ledger_path": "checksums.sha256",
    })
    stage.rename(final)
    for path in sorted(final.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    final.chmod(0o555)
    print(json.dumps({"status": "PASS", "contract_id": contract_id,
                      "contract_sha256": contract_sha, "artifact": str(final),
                      "fixture_manifest": fixture_manifest}, indent=2))
    return final


def validate(root: Path) -> dict:
    contract = json.loads((root / "contract_snapshot.yaml").read_text())["quality_contract"]
    if contract["quality_execution"] != "LOCKED" or contract["approval"]["approved"] is not False:
        raise QP1Error("quality gate is not locked")
    if contract["pending_qp1_items"] != ["formal_human_approval"]:
        raise QP1Error("technical pending items remain")
    if [item["method_config_id"] for item in contract["configurations"]["items"]] != CONFIG_IDS:
        raise QP1Error("configuration set drifted")
    if contract["provenance"]["performance_freeze_tag"] != "perf-freeze-20260917-83536c37-r1":
        raise QP1Error("freeze tag drifted")
    input_manifest = json.loads((root / "input_manifest.json").read_text())
    if input_manifest["contract_sha256"] != sha256_file(root / "contract_snapshot.yaml"):
        raise QP1Error("contract hash mismatch")
    for item in input_manifest["files"]:
        path = root / item["path"]
        if path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
            raise QP1Error("input fixture mismatch")
    # Exact LongBench conditioning split identity on the first item of every suite.
    for index_path in sorted((root / "selected_inputs").rglob("*.index.json")):
        index = json.loads(index_path.read_text())
        if index["records"] and index["records"][0]["length"] < 16:
            raise QP1Error("LongBench prompt shorter than conditioning split")
    ppl = json.loads((root / "selected_inputs/ppl/anchor_manifest.json").read_text())
    for dataset in ppl["datasets"].values():
        for stage in dataset["stages"].values():
            for length, anchors in stage.items():
                for anchor in anchors:
                    if anchor["prefix_end"] != anchor["burn_in_index"] or anchor["target_start"] != anchor["burn_in_index"] + 1:
                        raise QP1Error(f"PPL alignment mismatch at L={length}")
    result = {"status": "PASS", "contract_id": contract["id"],
              "contract_sha256": sha256_file(root / "contract_snapshot.yaml"),
              "quality_execution": "LOCKED", "quality_scores_computed": False}
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare"); prep.add_argument("--contract-id", default=CONTRACT_ID)
    check = sub.add_parser("validate"); check.add_argument("artifact", type=Path)
    args = parser.parse_args()
    if args.command == "prepare": prepare(args.contract_id)
    else: validate(args.artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
