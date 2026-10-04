#!/usr/bin/env python3
"""Prepare and verify the isolated Phase 5 TurboQuant reference environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPOSITORY_ROOT / "reference" / "turboquant"
ENVIRONMENT_PATH = REFERENCE_ROOT / "environment.json"
SOURCE_MANIFEST_PATH = REFERENCE_ROOT / "source_manifest.json"
FREEZE_PATH = REFERENCE_ROOT / "python-freeze.txt"
DEFAULT_VENV = REPOSITORY_ROOT / ".reference" / "turboquant-v0.25.1"
DEFAULT_SOURCE = REPOSITORY_ROOT / ".reference" / "vllm-source-v0.25.1"


class ReferenceEnvironmentError(RuntimeError):
    """Raised when the frozen reference environment does not match."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ReferenceEnvironmentError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ReferenceEnvironmentError(f"non-finite JSON value in {path}: {value}")

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_pairs,
        parse_constant=reject_constant,
    )
    if not isinstance(payload, dict):
        raise ReferenceEnvironmentError(f"JSON root must be an object: {path}")
    return payload


def _run(
    args: list[str],
    *,
    cwd: Path | None = None,
    capture: bool = True,
) -> str:
    completed = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=capture,
        env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
    )
    return completed.stdout.strip() if capture else ""


def _source_entry() -> dict[str, Any]:
    payload = _load_json(SOURCE_MANIFEST_PATH)
    source = payload.get("source")
    if not isinstance(source, dict):
        raise ReferenceEnvironmentError("source_manifest.json lacks source object")
    return source


def prepare_source(source_root: Path) -> None:
    source = _source_entry()
    revision = str(source["commit"])
    repository = str(source["repository"])
    if not source_root.exists():
        source_root.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "init", str(source_root)], capture=False)
        _run(["git", "remote", "add", "origin", repository], cwd=source_root)
        _run(
            ["git", "fetch", "--depth=1", "origin", revision],
            cwd=source_root,
            capture=False,
        )
        _run(
            ["git", "update-ref", "refs/turboquant/locked", "FETCH_HEAD"],
            cwd=source_root,
        )
    verify_source(source_root)


def verify_source(source_root: Path) -> dict[str, Any]:
    source = _source_entry()
    if not (source_root / ".git").is_dir():
        raise ReferenceEnvironmentError(f"missing pinned source checkout: {source_root}")
    repository = _run(["git", "remote", "get-url", "origin"], cwd=source_root)
    if repository != source["repository"]:
        raise ReferenceEnvironmentError(
            f"source remote mismatch: expected {source['repository']}, found {repository}"
        )
    revision = str(source["commit"])
    commit = _run(["git", "rev-parse", f"{revision}^{{commit}}"], cwd=source_root)
    tree = _run(["git", "rev-parse", f"{revision}^{{tree}}"], cwd=source_root)
    if commit != revision or tree != source["tree"]:
        raise ReferenceEnvironmentError("pinned source commit/tree identity mismatch")
    for record in source["relevant_source_files"]:
        path = record["path"]
        blob = _run(["git", "rev-parse", f"{revision}:{path}"], cwd=source_root)
        raw = subprocess.run(
            ["git", "show", f"{revision}:{path}"],
            cwd=source_root,
            check=True,
            capture_output=True,
        ).stdout
        if blob != record["git_blob"] or _sha256(raw) != record["sha256"]:
            raise ReferenceEnvironmentError(f"source file identity mismatch: {path}")
    return {
        "repository": repository,
        "commit": commit,
        "tree": tree,
        "relevant_file_count": len(source["relevant_source_files"]),
    }


def prepare_venv(venv_root: Path) -> None:
    python = venv_root / "bin" / "python"
    if not python.exists():
        if platform.python_version_tuple()[:2] != ("3", "12"):
            raise ReferenceEnvironmentError(
                "Python 3.12 is required to create the frozen reference environment"
            )
        venv_root.parent.mkdir(parents=True, exist_ok=True)
        _run([sys.executable, "-m", "venv", str(venv_root)], capture=False)
        _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                "-r",
                str(FREEZE_PATH),
            ],
            capture=False,
        )
    verify_venv(venv_root)


def _runtime_probe(venv_python: Path) -> dict[str, Any]:
    command = r'''
import hashlib, importlib.metadata, json, platform, subprocess
from pathlib import Path
import torch, triton, vllm
root = Path(vllm.__file__).resolve().parent.parent
files = {}
for relative in json.loads(__import__("os").environ["TQ_INSTALLED_SOURCE_FILES"]):
    data = (root / relative).read_bytes()
    files[relative] = hashlib.sha256(data).hexdigest()
driver = subprocess.run(
    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
    check=True, capture_output=True, text=True,
).stdout.splitlines()[0].strip()
probe = torch.empty((1,), device="cuda:0")
print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "triton": triton.__version__,
    "vllm": importlib.metadata.version("vllm"),
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0),
    "compute_capability": "%d.%d" % torch.cuda.get_device_capability(0),
    "driver": driver,
    "probe_device": str(probe.device),
    "installed_source_files": files,
}, sort_keys=True))
'''
    source = _source_entry()
    relative_files = [
        record["path"]
        for record in source["relevant_source_files"]
        if str(record["path"]).startswith("vllm/")
    ]
    env = {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TQ_INSTALLED_SOURCE_FILES": json.dumps(relative_files),
    }
    completed = subprocess.run(
        [str(venv_python), "-c", command],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(completed.stdout)


def verify_venv(venv_root: Path) -> dict[str, Any]:
    environment = _load_json(ENVIRONMENT_PATH)
    source = _source_entry()
    python = venv_root / "bin" / "python"
    if not python.is_file():
        raise ReferenceEnvironmentError(f"missing reference Python: {python}")
    if venv_root.resolve() == (REPOSITORY_ROOT / ".venv").resolve():
        raise ReferenceEnvironmentError("reference environment may not be the Measurement Lane")
    if _sha256(FREEZE_PATH.read_bytes()) != environment["python_environment"][
        "freeze_sha256"
    ]:
        raise ReferenceEnvironmentError("python-freeze.txt checksum mismatch")
    actual_freeze = _run([str(python), "-m", "pip", "freeze", "--all"])
    expected_freeze = FREEZE_PATH.read_text(encoding="utf-8").strip()
    if actual_freeze != expected_freeze:
        raise ReferenceEnvironmentError("installed package freeze differs from lock")
    _run([str(python), "-m", "pip", "check"])
    runtime = _runtime_probe(python)
    expected = environment["runtime"]
    for key in (
        "python",
        "torch",
        "cuda_runtime",
        "triton",
        "vllm",
        "gpu",
        "compute_capability",
        "driver",
    ):
        if runtime[key] != expected[key]:
            raise ReferenceEnvironmentError(
                f"runtime mismatch for {key}: expected {expected[key]!r}, "
                f"found {runtime[key]!r}"
            )
    if not runtime["cuda_available"] or runtime["probe_device"] != "cuda:0":
        raise ReferenceEnvironmentError("CUDA probe failed or silently fell back to CPU")
    expected_files = {
        item["path"]: item["sha256"]
        for item in source["relevant_source_files"]
        if str(item["path"]).startswith("vllm/")
    }
    if runtime["installed_source_files"] != expected_files:
        raise ReferenceEnvironmentError("installed vLLM TurboQuant source hashes differ")
    return runtime


def verify_all(venv_root: Path, source_root: Path) -> dict[str, Any]:
    return {
        "schema_version": "turboquant-reference-bootstrap-result-1.0.0",
        "status": "ready",
        "lane": "reference",
        "measurement_environment_modified": False,
        "source": verify_source(source_root),
        "runtime": verify_venv(venv_root),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "verify"))
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            prepare_source(args.source)
            prepare_venv(args.venv)
        result = verify_all(args.venv, args.source)
    except (OSError, subprocess.CalledProcessError, ReferenceEnvironmentError) as error:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "error": type(error).__name__,
                    "message": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
