#!/usr/bin/env python3
"""Run the narrow Phase 11D-Q23 checks inside the authorized container."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


AUTHORIZED_IMAGE = (
    "sha256:059bc9be89387369d7de9e3e9b26d85b6e9902c41e7dbf002ebc45edd188fb7e"
)
EXPECTED_FIXTURE_ROOT = (
    "c28682d58706b58812dc1db69ba5eb4982339ba13f39bf67f751794cdaabfdec"
)
MANIFEST_RELATIVE = Path(
    "third_party/patches/kvquant/"
    "deterministic-long-context-q3-q2-manifest.json"
)
HARNESS_RELATIVE = Path(
    "tests/cuda/phase11dq23_kvquant_long_context_validation.py"
)
Q4_HARNESS_RELATIVE = Path(
    "tests/cuda/phase11d_kvquant_long_context_validation.py"
)
SANITIZER = Path("/usr/local/cuda-13.0/bin/compute-sanitizer")
CUOBJDUMP = Path("/usr/local/cuda-13.0/bin/cuobjdump")
EXISTING_MHA_GQA_CUDA_TESTS = (
    (
        "tests_phase9p.test_deployment_gqa.DeploymentCudaTests."
        "test_4_3_2_bit_native_gqa_matches_explicit_repeat_reference"
    ),
    (
        "tests_phase9p.test_deployment_gqa.DeploymentCudaTests."
        "test_mha_groups_one_matches_direct_unpacked_4_3_2_bit_reference"
    ),
    (
        "tests_phase9p.test_deployment_gqa.DeploymentCudaTests."
        "test_cap_reached_sparse_4_3_2_bit_gqa_matches_native_kv_reference"
    ),
    (
        "tests_phase9p.test_deployment_gqa.DeploymentCudaTests."
        "test_all_changed_kernels_capture_and_allocate_nothing"
    ),
)


class Phase11DQ23ValidationError(RuntimeError):
    """The narrow source-remediation validation failed closed."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _write_exclusive(root: Path, relative: str, payload: bytes) -> Path:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return target


def _last_json(stdout: bytes) -> dict[str, Any] | None:
    text = stdout.decode("utf-8", errors="replace")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        return payload
    for line in reversed(text.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _zero_sanitizer(stdout: bytes, stderr: bytes) -> bool:
    text = (stdout + b"\n" + stderr).decode("utf-8", errors="replace")
    summaries = re.findall(r"ERROR SUMMARY:\s*(\d+)\s+errors?", text)
    leaks = re.findall(
        r"LEAK SUMMARY:\s*(\d+)\s+bytes leaked in\s+(\d+)\s+allocations?",
        text,
    )
    return bool(summaries) and all(int(item) == 0 for item in summaries) and (
        not leaks
        or all(
            int(byte_count) == 0 and int(count) == 0
            for byte_count, count in leaks
        )
    )


def _child_environment() -> dict[str, str]:
    allowed = (
        "CUBLAS_WORKSPACE_CONFIG",
        "CUDA_VISIBLE_DEVICES",
        "KVBENCH_AUTHORIZED_IMAGE_DIGEST",
        "KVBENCH_EXECUTION_ENVIRONMENT",
        "LD_LIBRARY_PATH",
        "PATH",
        "PYTHONPATH",
    )
    environment = {
        name: os.environ[name]
        for name in allowed
        if os.environ.get(name)
    }
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONIOENCODING": "utf-8",
        }
    )
    return environment


def _run(
    command: tuple[str, ...],
    *,
    cwd: Path,
    environment: dict[str, str],
    timeout: int,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            check=False,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise Phase11DQ23ValidationError(
            f"command timed out: {command[0]}"
        ) from error


def _record_command(
    output_root: Path,
    name: str,
    command: tuple[str, ...],
    result: subprocess.CompletedProcess[bytes],
) -> dict[str, Any]:
    stdout_path = _write_exclusive(
        output_root,
        f"raw/{name}.stdout.txt",
        result.stdout,
    )
    stderr_path = _write_exclusive(
        output_root,
        f"raw/{name}.stderr.txt",
        result.stderr,
    )
    payload = _last_json(result.stdout)
    record = {
        "command_argv": list(command),
        "returncode": result.returncode,
        "stdout_path": stdout_path.relative_to(output_root).as_posix(),
        "stdout_sha256": _sha256_bytes(result.stdout),
        "stderr_path": stderr_path.relative_to(output_root).as_posix(),
        "stderr_sha256": _sha256_bytes(result.stderr),
        "result": payload,
    }
    _write_exclusive(
        output_root,
        f"checks/{name}.json",
        _json_bytes(record),
    )
    return record


def _require_result(record: dict[str, Any], name: str) -> dict[str, Any]:
    payload = record.get("result")
    if (
        record.get("returncode") != 0
        or not isinstance(payload, dict)
        or payload.get("passed") is not True
    ):
        raise Phase11DQ23ValidationError(f"{name} failed")
    return payload


def _validate_code_objects(extension: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for arguments, marker, label in (
        (("--list-elf", str(extension)), b".sm_120.cubin", "sm_120_cubin"),
        (("--dump-ptx", str(extension)), b".target sm_120", "compute_120_ptx"),
    ):
        result = _run(
            (str(CUOBJDUMP), *arguments),
            cwd=extension.parent,
            environment=_child_environment(),
            timeout=300,
        )
        passed = result.returncode == 0 and marker in result.stdout
        records.append(
            {
                "label": label,
                "returncode": result.returncode,
                "stdout_sha256": _sha256_bytes(result.stdout),
                "stderr_sha256": _sha256_bytes(result.stderr),
                "passed": passed,
            }
        )
        if not passed:
            raise Phase11DQ23ValidationError(f"extension lacks {label}")
    return {"records": records, "passed": True}


def _harness_command(
    repository_root: Path,
    extension: Path,
    extension_sha256: str,
    mode: str,
    *,
    fixture_root: Path | None = None,
    source_root: Path | None = None,
    harness_relative: Path = HARNESS_RELATIVE,
) -> tuple[str, ...]:
    command = [
        sys.executable,
        str(repository_root / harness_relative),
        "--mode",
        mode,
        "--extension",
        str(extension),
        "--extension-sha256",
        extension_sha256,
    ]
    if fixture_root is not None:
        command.extend(("--fixture-root", str(fixture_root)))
    if source_root is not None:
        command.extend(("--source-root", str(source_root)))
    return tuple(command)


def _run_sanitizer(
    output_root: Path,
    repository_root: Path,
    extension: Path,
    extension_sha256: str,
) -> dict[str, Any]:
    version = _run(
        (str(SANITIZER), "--version"),
        cwd=repository_root,
        environment=_child_environment(),
        timeout=60,
    )
    if version.returncode != 0:
        raise Phase11DQ23ValidationError(
            "Compute Sanitizer version is unavailable"
        )
    records: list[dict[str, Any]] = []
    for tool in ("memcheck", "initcheck"):
        command = (
            str(SANITIZER),
            "--tool",
            tool,
            "--error-exitcode",
            "99",
            "--target-processes",
            "application-only",
            *(("--leak-check", "full") if tool == "memcheck" else ()),
            *_harness_command(
                repository_root,
                extension,
                extension_sha256,
                "sanitizer",
            ),
        )
        environment = _child_environment()
        environment["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
        result = _run(
            command,
            cwd=repository_root,
            environment=environment,
            timeout=1800,
        )
        record = _record_command(
            output_root,
            f"sanitizer-{tool}",
            command,
            result,
        )
        payload = record.get("result")
        passed = (
            result.returncode == 0
            and _zero_sanitizer(result.stdout, result.stderr)
            and isinstance(payload, dict)
            and payload.get("passed") is True
        )
        record["tool"] = tool
        record["zero_error_and_leak_summary"] = _zero_sanitizer(
            result.stdout,
            result.stderr,
        )
        record["passed"] = passed
        records.append(record)
        if not passed:
            raise Phase11DQ23ValidationError(
                f"Compute Sanitizer {tool} failed"
            )
    return {
        "version_stdout_sha256": _sha256_bytes(version.stdout),
        "version_text": version.stdout.decode(
            "utf-8",
            errors="replace",
        ).strip(),
        "records": records,
        "passed": True,
    }


def _finalize(output_root: Path, summary: dict[str, Any]) -> dict[str, Any]:
    _write_exclusive(output_root, "summary.json", _json_bytes(summary))
    excluded = {"artifact_inventory.json", "checksums.sha256", "COMPLETE"}
    payload_files = sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file()
        and path.relative_to(output_root).as_posix() not in excluded
    )
    inventory = {
        "schema_version": "kvbench-artifact-inventory-1.0.0",
        "files": [
            {
                "path": path.relative_to(output_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in payload_files
        ],
        "excluded_control_files": sorted(excluded),
    }
    _write_exclusive(
        output_root,
        "artifact_inventory.json",
        _json_bytes(inventory),
    )
    ledger_paths = sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file()
        and path.relative_to(output_root).as_posix()
        not in {"checksums.sha256", "COMPLETE"}
    )
    ledger = "".join(
        f"{_sha256_file(path)}  {path.relative_to(output_root).as_posix()}\n"
        for path in ledger_paths
    ).encode("utf-8")
    _write_exclusive(output_root, "checksums.sha256", ledger)
    completion = {
        "schema_version": "kvbench-completion-1.0.0",
        "status": "completed",
        "artifact_inventory_sha256": _sha256_file(
            output_root / "artifact_inventory.json"
        ),
        "checksum_ledger_sha256": _sha256_bytes(ledger),
        "written_last": True,
    }
    _write_exclusive(output_root, "COMPLETE", _json_bytes(completion))
    for path in sorted(output_root.rglob("*"), reverse=True):
        path.chmod(0o444 if path.is_file() else 0o555)
    output_root.chmod(0o555)
    return {
        "evidence_root_sha256": _sha256_bytes(ledger),
        "artifact_count": len(ledger_paths) + 2,
        "complete_last": True,
    }


def main() -> int:
    arguments = _parse_args()
    repository_root = arguments.repository_root.resolve(strict=True)
    source_root = arguments.source_root.resolve(strict=True)
    extension = arguments.extension.resolve(strict=True)
    fixture_root = arguments.fixture_root.resolve(strict=True)
    calibration_root = arguments.calibration_root.resolve(strict=True)
    output_root = arguments.output_root.resolve(strict=True)
    if any(output_root.iterdir()):
        raise Phase11DQ23ValidationError("output root must start empty")
    if (
        not Path("/.dockerenv").is_file()
        or os.environ.get("KVBENCH_AUTHORIZED_IMAGE_DIGEST") != AUTHORIZED_IMAGE
        or os.environ.get("KVBENCH_EXECUTION_ENVIRONMENT")
        != "measurement_container"
    ):
        raise Phase11DQ23ValidationError(
            "authorized container identity differs"
        )

    manifest = json.loads(
        (repository_root / MANIFEST_RELATIVE).read_text(encoding="utf-8")
    )
    extension_sha256 = manifest["extension"]["sha256"]
    if _sha256_file(extension) != extension_sha256:
        raise Phase11DQ23ValidationError("fresh extension identity differs")

    environment = _child_environment()
    source_command = (
        sys.executable,
        "-m",
        "scripts.validate_kvquant_q23_long_context_patch",
        "--source-root",
        str(source_root),
    )
    source_record = _record_command(
        output_root,
        "source",
        source_command,
        _run(
            source_command,
            cwd=repository_root,
            environment=environment,
            timeout=300,
        ),
    )
    source_result = source_record.get("result")
    if (
        source_record["returncode"] != 0
        or not isinstance(source_result, dict)
        or source_result.get("status") != "PASS"
    ):
        raise Phase11DQ23ValidationError(
            "source reconstruction validation failed"
        )

    fixture_command = (
        sys.executable,
        str(
            repository_root
            / "reference/kvquant_phase11pr/validate_corrected_bundle.py"
        ),
        "--fixtures",
        str(fixture_root),
        "--old-fixtures",
        str(repository_root / "reference/kvquant/fixtures"),
    )
    fixture_custody_record = _record_command(
        output_root,
        "fixture-custody",
        fixture_command,
        _run(
            fixture_command,
            cwd=repository_root,
            environment=environment,
            timeout=300,
        ),
    )
    fixture_custody = fixture_custody_record.get("result")
    if (
        fixture_custody_record["returncode"] != 0
        or not isinstance(fixture_custody, dict)
        or fixture_custody.get("status") != "PASS"
        or fixture_custody.get("local_root_sha256")
        != EXPECTED_FIXTURE_ROOT
    ):
        raise Phase11DQ23ValidationError("fixture custody validation failed")

    results: dict[str, dict[str, Any]] = {}
    for mode in ("determinism", "fixtures", "stream", "graph"):
        command = _harness_command(
            repository_root,
            extension,
            extension_sha256,
            mode,
            fixture_root=fixture_root if mode == "fixtures" else None,
            source_root=source_root if mode == "fixtures" else None,
        )
        record = _record_command(
            output_root,
            mode,
            command,
            _run(
                command,
                cwd=repository_root,
                environment=environment,
                timeout=1800,
            ),
        )
        results[mode] = _require_result(record, mode)

    q4_regression: dict[str, dict[str, Any]] = {}
    for mode in ("determinism", "stream", "graph"):
        command = _harness_command(
            repository_root,
            extension,
            extension_sha256,
            mode,
            harness_relative=Q4_HARNESS_RELATIVE,
        )
        record = _record_command(
            output_root,
            f"q4-regression-{mode}",
            command,
            _run(
                command,
                cwd=repository_root,
                environment=environment,
                timeout=1800,
            ),
        )
        q4_regression[mode] = _require_result(
            record,
            f"q4-regression-{mode}",
        )

    adapter_environment = dict(environment)
    adapter_environment.update(
        {
            "KVBENCH_KVQUANT_EXTENSION": str(extension),
            "KVBENCH_KVQUANT_CALIBRATION_ROOT": str(calibration_root),
        }
    )
    adapter_command = (
        sys.executable,
        "-m",
        "unittest",
        (
            "tests.cuda.test_phase11_kvquant_cuda."
            "Phase11KVQuantCudaTests."
            "test_all_nine_corrected_fixtures_conform_through_adapter"
        ),
        (
            "tests.graph.test_phase11_kvquant_graph."
            "Phase11KVQuantGraphTests."
            "test_all_bit_widths_capture_append_and_direct_decode"
        ),
        "-v",
    )
    adapter_record = _record_command(
        output_root,
        "adapter-fixture-graph",
        adapter_command,
        _run(
            adapter_command,
            cwd=repository_root,
            environment=adapter_environment,
            timeout=1800,
        ),
    )
    if adapter_record["returncode"] != 0:
        raise Phase11DQ23ValidationError(
            "current Adapter fixture/graph regression failed"
        )

    source_helper_command = _harness_command(
        repository_root,
        extension,
        extension_sha256,
        "mha_gqa",
        source_root=source_root,
    )
    source_helper_record = _record_command(
        output_root,
        "mha-gqa-source-helper-control",
        source_helper_command,
        _run(
            source_helper_command,
            cwd=repository_root,
            environment=environment,
            timeout=1800,
        ),
    )
    source_helper_result = _require_result(
        source_helper_record,
        "mha-gqa-source-helper-control",
    )

    gqa_environment = dict(environment)
    gqa_environment.update(
        {
            "KVQUANT_RUN_CUDA_TESTS": "1",
            "KVQUANT_CUDA_EXTENSION_DIR": str(extension.parent),
            "PYTHONPATH": (
                f"{source_root}:"
                f"{gqa_environment.get('PYTHONPATH', '')}"
            ),
        }
    )
    gqa_command = (
        sys.executable,
        "-m",
        "unittest",
        *EXISTING_MHA_GQA_CUDA_TESTS,
        "-v",
    )
    gqa_record = _record_command(
        output_root,
        "mha-gqa-regression",
        gqa_command,
        _run(
            gqa_command,
            cwd=source_root,
            environment=gqa_environment,
            timeout=1800,
        ),
    )
    if gqa_record["returncode"] != 0:
        raise Phase11DQ23ValidationError("MHA/GQA regression failed")

    code_objects = _validate_code_objects(extension)
    sanitizer = _run_sanitizer(
        output_root,
        repository_root,
        extension,
        extension_sha256,
    )
    summary = {
        "schema_version": "kvbench-phase11dq23-validation-summary-1.0.0",
        "status": "PASS",
        "source": source_result,
        "extension_sha256": extension_sha256,
        "fixture_custody": fixture_custody,
        "determinism": results["determinism"],
        "fixtures": results["fixtures"],
        "stream": results["stream"],
        "graph": results["graph"],
        "q4_regression": q4_regression,
        "adapter_fixture_graph": {
            "returncode": adapter_record["returncode"],
            "stdout_sha256": adapter_record["stdout_sha256"],
            "stderr_sha256": adapter_record["stderr_sha256"],
            "passed": True,
        },
        "mha_gqa_regression": {
            "frozen_source_helper_control": source_helper_result,
            "existing_cuda_test_count": len(EXISTING_MHA_GQA_CUDA_TESTS),
            "existing_cuda_tests": list(EXISTING_MHA_GQA_CUDA_TESTS),
            "existing_cuda_returncode": gqa_record["returncode"],
            "existing_cuda_stdout_sha256": gqa_record["stdout_sha256"],
            "existing_cuda_stderr_sha256": gqa_record["stderr_sha256"],
            "passed": True,
        },
        "code_objects": code_objects,
        "sanitizer": sanitizer,
        "adapter_modified": True,
        "adapter_workspace_additional_allocated_bytes": 0,
        "admission_grid_run": False,
        "method_admission_report_published": False,
        "performance_timing": False,
        "quality_execution": False,
    }
    finalization = _finalize(output_root, summary)
    print(
        json.dumps(
            {**summary, **finalization},
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
