#!/usr/bin/env python3
"""Validate the narrow Phase 7 KIVI B-019 GQA remediation patch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import types
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    REPOSITORY_ROOT / "third_party/patches/kivi/manifest.json"
)


class ValidationError(RuntimeError):
    """Raised when the pinned patch or its semantic contract is invalid."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValidationError(f"{path} must contain a JSON object")
    return value


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob_sha1(data: bytes) -> str:
    prefix = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(prefix + data, usedforsecurity=False).hexdigest()


def _extract_added_file(patch_bytes: bytes, path: str) -> bytes:
    patch_text = patch_bytes.decode("utf-8")
    marker = f"diff --git a/{path} b/{path}\n"
    if patch_text.count(marker) != 1:
        raise ValidationError(f"patch must add exactly one {path} section")
    section = patch_text.split(marker, 1)[1]
    if "\ndiff --git " in section:
        section = section.split("\ndiff --git ", 1)[0]
    lines = section.splitlines()
    try:
        hunk_index = next(
            index for index, line in enumerate(lines) if line.startswith("@@ ")
        )
    except StopIteration as error:
        raise ValidationError(f"patch section for {path} has no hunk") from error
    if "--- /dev/null" not in lines[:hunk_index]:
        raise ValidationError(f"patch section for {path} is not a new file")
    content: list[str] = []
    for line in lines[hunk_index + 1 :]:
        if line.startswith("+"):
            content.append(line[1:])
        elif line == r"\ No newline at end of file":
            raise ValidationError(f"added file {path} must end with a newline")
        else:
            raise ValidationError(
                f"new-file patch for {path} contains a non-added line"
            )
    return ("\n".join(content) + "\n").encode("utf-8")


def _run_git(source_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=source_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise ValidationError(
            f"git {' '.join(arguments)} failed with exit {result.returncode}"
        )
    return result.stdout.rstrip("\n")


def _validate_source_root(
    source_root: Path,
    manifest: dict[str, Any],
    patch_path: Path,
) -> dict[str, Any]:
    source_root = source_root.resolve(strict=True)
    source = manifest["source"]
    if _run_git(source_root, "rev-parse", "HEAD") != source["base_commit"]:
        raise ValidationError("KIVI checkout HEAD does not match base_commit")
    if (
        _run_git(source_root, "rev-parse", "HEAD^{tree}")
        != source["base_tree"]
    ):
        raise ValidationError("KIVI checkout base tree does not match lock")

    expected_paths = {
        record["path"] for record in manifest["patched_files"]
    }
    status_paths = {
        line[3:]
        for line in _run_git(
            source_root, "status", "--porcelain", "--untracked-files=all"
        ).splitlines()
        if line
    }
    if status_paths != expected_paths:
        raise ValidationError(
            "patched checkout must differ from HEAD at exactly the locked files"
        )

    observed_files: list[dict[str, str]] = []
    for record in manifest["patched_files"]:
        path = source_root / record["path"]
        data = path.read_bytes()
        observed = {
            "path": record["path"],
            "git_blob": _git_blob_sha1(data),
            "sha256": _sha256(data),
        }
        if observed["git_blob"] != record["patched_git_blob"]:
            raise ValidationError(f"patched Git blob mismatch: {record['path']}")
        if observed["sha256"] != record["patched_sha256"]:
            raise ValidationError(f"patched SHA-256 mismatch: {record['path']}")
        observed_files.append(observed)

    _run_git(
        source_root,
        "apply",
        "--unidiff-zero",
        "--reverse",
        "--check",
        str(patch_path),
    )

    with tempfile.TemporaryDirectory(prefix="kivi-b019-index-") as directory:
        index_path = Path(directory) / "index"
        environment = dict(os.environ)
        environment["GIT_INDEX_FILE"] = str(index_path)
        for arguments in (
            ("read-tree", "HEAD"),
            ("add", "--", *sorted(expected_paths)),
            ("write-tree",),
        ):
            result = subprocess.run(
                ("git", *arguments),
                cwd=source_root,
                env=environment,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode != 0:
                raise ValidationError(
                    "failed to derive patched tree in isolated Git index"
                )
            if arguments[0] == "write-tree":
                patched_tree = result.stdout.strip()
    if patched_tree != source["patched_tree"]:
        raise ValidationError("derived patched tree does not match manifest")

    return {
        "base_commit": source["base_commit"],
        "base_tree": source["base_tree"],
        "patched_files": observed_files,
        "patched_tree": patched_tree,
        "reverse_patch_check": "PASS",
        "status_path_count": len(status_paths),
    }


def _load_helper_module(source_bytes: bytes) -> types.ModuleType:
    module = types.ModuleType("kivi_b019_candidate")
    exec(
        compile(source_bytes, "models/kivi_gqa.py", "exec"),
        module.__dict__,
    )
    return module


def _validate_semantics(
    helper_source: bytes,
    device_name: str,
) -> dict[str, Any]:
    import torch

    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValidationError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    helper = _load_helper_module(helper_source)
    cases = (
        {"query_length": 3, "key_length": 17},
        {"query_length": 1, "key_length": 33},
    )
    case_results: list[dict[str, Any]] = []
    for case in cases:
        torch.manual_seed(20260726 + case["key_length"])
        query = torch.randn(
            1,
            32,
            case["query_length"],
            128,
            dtype=torch.bfloat16,
            device=device,
        )
        key = torch.randn(
            1,
            8,
            case["key_length"],
            128,
            dtype=torch.bfloat16,
            device=device,
        )
        value = torch.randn(
            1,
            8,
            case["key_length"],
            128,
            dtype=torch.bfloat16,
            device=device,
        )
        weights = torch.randn(
            1,
            32,
            case["query_length"],
            case["key_length"],
            dtype=torch.bfloat16,
            device=device,
        )
        key_pointer = key.untyped_storage().data_ptr()
        value_pointer = value.untyped_storage().data_ptr()

        bmm_operands: list[dict[str, Any]] = []
        original_bmm = torch.bmm

        def recording_bmm(
            left: Any,
            right: Any,
        ) -> Any:
            bmm_operands.append(
                {
                    "left_shape": list(left.shape),
                    "right_shape": list(right.shape),
                    "device": left.device.type,
                }
            )
            return original_bmm(left, right)

        torch.bmm = recording_bmm
        try:
            scores = helper.gqa_query_key_matmul(query, key)
            output = helper.gqa_attention_value_matmul(weights, value)
        finally:
            torch.bmm = original_bmm

        repeated_key = torch.repeat_interleave(key, 4, dim=1)
        repeated_value = torch.repeat_interleave(value, 4, dim=1)
        reference_scores = torch.matmul(
            query, repeated_key.transpose(2, 3)
        )
        reference_output = torch.matmul(weights, repeated_value)
        if not torch.equal(scores, reference_scores):
            raise ValidationError("BF16 query/key equivalence failed")
        if not torch.equal(output, reference_output):
            raise ValidationError("BF16 attention/value equivalence failed")

        expected_batch = 8
        if any(
            record["left_shape"][0] != expected_batch
            or record["right_shape"][0] != expected_batch
            for record in bmm_operands
        ):
            raise ValidationError("an H_Q-sized K/V BMM operand was observed")
        if list(key.shape) != [1, 8, case["key_length"], 128]:
            raise ValidationError("key storage geometry changed")
        if list(value.shape) != [1, 8, case["key_length"], 128]:
            raise ValidationError("value storage geometry changed")
        if key.untyped_storage().data_ptr() != key_pointer:
            raise ValidationError("key storage identity changed")
        if value.untyped_storage().data_ptr() != value_pointer:
            raise ValidationError("value storage identity changed")

        case_results.append(
            {
                **case,
                "bmm_operands": bmm_operands,
                "key_shape": list(key.shape),
                "value_shape": list(value.shape),
                "query_key_bf16_exact": True,
                "attention_value_bf16_exact": True,
                "expanded_kv_operand": False,
            }
        )

    invalid_query = torch.zeros(
        1, 31, 1, 8, dtype=torch.float32, device=device
    )
    invalid_key = torch.zeros(
        1, 8, 1, 8, dtype=torch.float32, device=device
    )
    try:
        helper.gqa_query_key_matmul(invalid_query, invalid_key)
    except ValueError:
        unsupported_geometry_rejected = True
    else:
        unsupported_geometry_rejected = False
    if not unsupported_geometry_rejected:
        raise ValidationError("non-divisible GQA geometry was not rejected")

    result: dict[str, Any] = {
        "cases": case_results,
        "device": device.type,
        "dtype": "bfloat16",
        "head_mapping": [head // 4 for head in range(32)],
        "h_q": 32,
        "h_kv": 8,
        "unsupported_geometry_rejected": True,
    }
    if device.type == "cuda":
        result["cuda_device_name"] = torch.cuda.get_device_name(device)
        result["cuda_capability"] = list(
            torch.cuda.get_device_capability(device)
        )
    return result


def validate(
    *,
    device: str,
    source_root: Path | None = None,
) -> dict[str, Any]:
    manifest = _read_json(MANIFEST_PATH)
    patch_path = REPOSITORY_ROOT / manifest["patch"]["path"]
    patch_bytes = patch_path.read_bytes()
    if _sha256(patch_bytes) != manifest["patch"]["sha256"]:
        raise ValidationError("patch SHA-256 does not match manifest")

    helper_path = "models/kivi_gqa.py"
    helper_source = _extract_added_file(patch_bytes, helper_path)
    helper_record = next(
        record
        for record in manifest["patched_files"]
        if record["path"] == helper_path
    )
    if _sha256(helper_source) != helper_record["patched_sha256"]:
        raise ValidationError("extracted helper SHA-256 mismatch")
    if _git_blob_sha1(helper_source) != helper_record["patched_git_blob"]:
        raise ValidationError("extracted helper Git blob mismatch")

    added_lines = [
        line[1:]
        for line in patch_bytes.decode("utf-8").splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    added_source = "\n".join(added_lines)
    for prohibited in ("repeat_kv(", "repeat_interleave(", ".expand("):
        if prohibited in added_source:
            raise ValidationError(
                f"patch adds prohibited operation: {prohibited}"
            )
    if added_source.count("torch.bmm(") != 2:
        raise ValidationError("patch must add exactly two grouped BMM calls")

    source_validation = None
    if source_root is not None:
        source_validation = _validate_source_root(
            source_root, manifest, patch_path
        )
    return {
        "schema_version": "kvbench-kivi-b019-validation-result-1.0.0",
        "status": "PASS",
        "performance_measurement": False,
        "patch": {
            "path": manifest["patch"]["path"],
            "sha256": manifest["patch"]["sha256"],
            "prohibited_added_operations": False,
        },
        "semantics": _validate_semantics(helper_source, device),
        "source_checkout": source_validation,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--source-root", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    try:
        result = validate(
            device=arguments.device,
            source_root=arguments.source_root,
        )
    except (OSError, ValueError, ValidationError) as error:
        print(
            json.dumps(
                {
                    "schema_version": (
                        "kvbench-kivi-b019-validation-result-1.0.0"
                    ),
                    "status": "FAIL",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
