#!/usr/bin/env python3
"""Fail-closed Phase 1 / E00 hardware and CUDA certification collector."""

from __future__ import annotations

import argparse
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import signal
import shutil
import stat
import socket
import subprocess
import sys
import tarfile
import time
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = ROOT / "docs" / "evidence" / "e00"
MEASUREMENT_CONTAINER_EVIDENCE_ROOT = (
    ROOT / "artifacts" / "phase6a" / "container_g0"
)
SCHEMA_PATH = ROOT / "preflight" / "e00_manifest.schema.json"
PYTHON_LOCK_PATH = ROOT / "preflight" / "requirements-e00.txt"
SYSTEM_LOCK_PATH = ROOT / "preflight" / "system-packages.lock.json"
MEASUREMENT_CONTAINER_SYSTEM_LOCK_PATH = (
    ROOT
    / "preflight"
    / "measurement-container-system-packages.expected.json"
)
MEASUREMENT_CONTAINER_OBSERVED_SYSTEM_LOCK_PATH = (
    ROOT
    / "preflight"
    / "measurement-container-system-packages.lock.json"
)
PROCESS_QUERY_PATH = ROOT / "preflight" / "process_query.py"
PYTHON_PROBE_PATH = ROOT / "preflight" / "python_probe.py"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
MEASUREMENT_CONTAINER_PYTHON = Path("/opt/kvbench/.venv/bin/python")
PYTHON_INTEGRITY_PATH = ROOT / "preflight" / "python_integrity_probe.py"
CUDA_HOME_DEFAULT = Path("/usr/local/cuda-13.0")
EXPECTED_GPU_NAME = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
EXPECTED_GPU_UUID = "GPU-75bd273e-6b20-0d22-1b0b-5fbb6fb0025b"
MEASUREMENT_BASE_DIGEST_LABEL = "org.kvbench.measurement.base.manifest"
IMAGE_SECRET_VARIABLES = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "CLOUDFLARE_API_TOKEN",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "R2_ACCOUNT_ID",
)

# The pinned NVIDIA base image deliberately configures dpkg to exclude these
# non-executable payloads.  The Dockerfile also removes apt's transient list
# directory.  Package versions and the locked executable hashes remain exact;
# every other dpkg verification line is still rejected.
DPKG_EXPECTED_MISSING_PREFIXES = (
    "/usr/share/doc/",
    "/usr/share/locale/",
    "/usr/share/man/",
)
DPKG_EXPECTED_MISSING_PATHS = frozenset({"/var/lib/apt/lists/partial"})

MEASUREMENT_CONTAINER_LOCK_PURPOSE = (
    "Exact Measurement Container system toolchain observed from an "
    "uncertified candidate build; requires explicit review and promotion"
)
MEASUREMENT_CONTAINER_LOCK_AUTHORITY = (
    "candidate_observation_requires_explicit_review_and_promotion"
)
MEASUREMENT_CONTAINER_LOCK_TOOL_SPECS = (
    ("bash", "/usr/bin/bash", "bash", None),
    ("cat", "/usr/bin/cat", "coreutils", None),
    ("cc", "/usr/bin/gcc", "gcc-13-x86-64-linux-gnu", None),
    (
        "compute-sanitizer-real",
        "/usr/local/cuda-13.0/compute-sanitizer/compute-sanitizer",
        "cuda-sanitizer-13-0",
        "compute-sanitizer",
    ),
    ("dpkg", "/usr/bin/dpkg", "dpkg", None),
    ("dpkg-query", "/usr/bin/dpkg-query", "dpkg", None),
    ("env", "/usr/bin/env", "coreutils", None),
    ("git", "/usr/bin/git", "git", None),
    ("hostname", "/usr/bin/hostname", "hostname", None),
    ("lscpu", "/usr/bin/lscpu", "util-linux", None),
    ("readlink", "/usr/bin/readlink", "coreutils", None),
    ("sha256sum", "/usr/bin/sha256sum", "coreutils", None),
    ("sh", "/bin/sh", "dash", None),
    (
        "systemd-detect-virt",
        "/usr/bin/systemd-detect-virt",
        "systemd",
        None,
    ),
    ("uname", "/usr/bin/uname", "coreutils", None),
    (
        "compute-sanitizer",
        "/usr/local/cuda-13.0/bin/compute-sanitizer",
        "cuda-sanitizer-13-0",
        None,
    ),
    (
        "cuobjdump",
        "/usr/local/cuda-13.0/bin/cuobjdump",
        "cuda-cuobjdump-13-0",
        "cuobjdump",
    ),
    (
        "nvdisasm",
        "/usr/local/cuda-13.0/bin/nvdisasm",
        "cuda-nvdisasm-13-0",
        "nvdisasm",
    ),
    ("c++", "/usr/bin/c++", "g++-13-x86-64-linux-gnu", "c++"),
    ("ld", "/usr/bin/ld", "binutils-x86-64-linux-gnu", None),
    (
        "ncu",
        "/opt/nvidia/nsight-compute/2026.2.1/ncu",
        "nsight-compute-2026.2.1",
        "ncu",
    ),
    ("ninja", "/usr/bin/ninja", "ninja-build", "ninja"),
    (
        "nsys",
        "/usr/local/bin/nsys",
        "nsight-systems-2026.1.3",
        "nsys",
    ),
    (
        "nvcc",
        "/usr/local/cuda-13.0/bin/nvcc",
        "cuda-nvcc-13-0",
        "nvcc",
    ),
    (
        "python",
        "/opt/kvbench/.venv/bin/python",
        "python3.12-minimal",
        "python",
    ),
)

GPU_QUERY_FIELDS = (
    "index",
    "name",
    "uuid",
    "pci.bus_id",
    "pci.device_id",
    "compute_cap",
    "memory.total",
    "ecc.mode.current",
    "ecc.mode.pending",
    "power.draw",
    "power.limit",
    "power.default_limit",
    "power.min_limit",
    "power.max_limit",
    "clocks.current.sm",
    "clocks.current.memory",
    "clocks.max.sm",
    "clocks.max.memory",
    "temperature.gpu",
    "vbios_version",
    "persistence_mode",
    "compute_mode",
    "mig.mode.current",
    "display_active",
    "pstate",
    "driver_version",
)

NATIVE_HOST_GATE_NAMES = (
    "committed_clean_code",
    "native_host_environment_verified",
    "single_uuid_selected_gpu",
    "target_sku_match",
    "hardware_identity_complete",
    "power_clock_state_complete",
    "required_toolchain_complete",
    "torch_cuda_available",
    "capability_identity_match",
    "extension_build",
    "native_sass_execution",
    "forced_ptx_jit_execution",
    "compute_sanitizer_clean",
    "no_kernel_image_error",
    "no_foreign_compute_process",
    "manifest_schema_valid",
    "evidence_checksums_valid",
)
MEASUREMENT_CONTAINER_GATE_NAMES = (
    "committed_clean_code",
    "measurement_container_environment_verified",
    *NATIVE_HOST_GATE_NAMES[2:],
)
# Preserve the historical public constant and default no-argument gate order.
GATE_NAMES = NATIVE_HOST_GATE_NAMES
ALL_GATE_NAMES = frozenset(
    (*NATIVE_HOST_GATE_NAMES, *MEASUREMENT_CONTAINER_GATE_NAMES)
)

ENVIRONMENT_ALLOWLIST = (
    "LC_ALL",
    "LANG",
    "TZ",
    "PATH",
    "GIT_CONFIG_GLOBAL",
    "GIT_CONFIG_NOSYSTEM",
    "CUDA_HOME",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "TORCH_CUDA_ARCH_LIST",
    "TORCH_EXTENSIONS_DIR",
    "CUDA_FORCE_PTX_JIT",
    "CUDA_DISABLE_PTX_JIT",
    "CUDA_CACHE_DISABLE",
    "CUDA_CACHE_PATH",
    "PYTHONNOUSERSITE",
    "PYTHONOPTIMIZE",
    "PYTHONPATH",
    "PYTORCH_NO_CUDA_MEMORY_CACHING",
    "MAX_JOBS",
    "CC",
    "CXX",
    "PYTHONHASHSEED",
    "E00_DETECTED_ARCH",
    "E00_EXTENSION_PATH",
    "E00_EXTENSION_MODULE",
    "E00_ALLOCATION_ITERATIONS",
)

CONTRACT_PATHS = (
    "Makefile",
    "scripts/preflight.sh",
    "preflight/run_preflight.py",
    "preflight/process_query.py",
    "preflight/python_probe.py",
    "preflight/python_integrity_probe.py",
    "preflight/audit_checkpoint.py",
    "preflight/__init__.py",
    "preflight/e00_cuda/build.py",
    "preflight/e00_cuda/binding.cpp",
    "preflight/e00_cuda/xor_kernel.cu",
    "preflight/e00_cuda/xor_kernel.h",
    "tests/cuda/e00_runtime_probe.py",
    "tests/cuda/e00_sanitizer_probe.py",
    "tests/golden/test_e00_numerical.py",
    "tests/graph_capture/test_e00_graph.py",
    "tests/allocation/test_e00_allocation.py",
    "AGENTS.md",
    "CODEX_WORKFLOW.md",
    "docs/decisions/0002-e00-native-host-evidence.md",
    "docs/decisions/0003-e00-certification-protocol.md",
    "docs/decisions/0004-e00-failure-evidence-and-memcheck.md",
    "preflight/e00_manifest.schema.json",
    "preflight/requirements-e00.txt",
    "preflight/system-packages.lock.json",
)
MEASUREMENT_CONTAINER_CONTRACT_PATHS = (
    *tuple(
        (
            "preflight/measurement-container-system-packages.expected.json"
            if path == "preflight/system-packages.lock.json"
            else path
        )
        for path in CONTRACT_PATHS
    ),
    "preflight/measurement-container-system-packages.lock.json",
    ".dockerignore",
    "docker/measurement.Dockerfile",
    "preflight/requirements-phase3.txt",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", value)
    if not cleaned or not cleaned[0].isalnum():
        cleaned = "x_" + cleaned
    return cleaned[:128]


def artifact_id(relative_path: str) -> str:
    return "file_" + sha256_bytes(relative_path.encode("utf-8"))[:20]


def write_exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def publish_atomic_exclusive(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    write_exclusive(temporary, data)
    try:
        os.link(temporary, path, follow_symlinks=False)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def rename_noreplace(source: Path, target: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = library.renameat2
    except AttributeError as error:
        raise RuntimeError("renameat2 is required for append-only E00 finalization") from error
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), str(target))


def output_ref(path: Path, relative_path: str) -> dict[str, Any]:
    target = path / relative_path
    return {
        "path": relative_path,
        "sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
    }


def relevant_environment(environment: dict[str, str]) -> dict[str, str]:
    result = {
        key: environment[key]
        for key in ENVIRONMENT_ALLOWLIST
        if key in environment
    }
    result["LC_ALL"] = environment.get("LC_ALL", "C")
    result["TZ"] = environment.get("TZ", "UTC")
    return result


def environment_with(
    base: dict[str, str],
    updates: dict[str, str] | None = None,
    unset: Iterable[str] = (),
) -> dict[str, str]:
    result = dict(base)
    for key in unset:
        result.pop(key, None)
    if updates:
        result.update(updates)
    return result


def process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_process_group(
    process: subprocess.Popen[bytes],
    *,
    grace_seconds: float = 2.0,
) -> bool:
    process_group_id = process.pid
    if not process_group_exists(process_group_id):
        return True
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + grace_seconds
    while process_group_exists(process_group_id) and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.05)
    if process_group_exists(process_group_id):
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            return True
    deadline = time.monotonic() + grace_seconds
    while process_group_exists(process_group_id) and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.05)
    return not process_group_exists(process_group_id)


def run_memory(
    argv: Sequence[str],
    *,
    environment: dict[str, str],
    cwd: Path = ROOT,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    started_at = utc_now()
    start = time.monotonic()
    timed_out = False
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            list(argv),
            cwd=str(cwd),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        exit_code = int(process.returncode)
        if process_group_exists(process.pid):
            terminated = terminate_process_group(process)
            exit_code = 125 if exit_code == 0 else exit_code
            stderr += (
                b"\nE00 command left descendant processes; process group terminated="
                + str(terminated).encode("ascii")
                + b".\n"
            )
            if not terminated:
                raise RuntimeError("could not drain command process group")
    except subprocess.TimeoutExpired as error:
        timed_out = True
        exit_code = 124
        if process is None:
            stdout = error.stdout or b""
            stderr = error.stderr or b""
            terminated = True
        else:
            terminated = terminate_process_group(process)
            if not terminated:
                raise RuntimeError("could not drain timed-out command process group")
            try:
                stdout, stderr = process.communicate(timeout=5.0)
            except subprocess.TimeoutExpired as final_error:
                final_terminated = terminate_process_group(
                    process, grace_seconds=0.5
                )
                if not final_terminated:
                    raise RuntimeError(
                        "could not reap timed-out command process group"
                    )
                stdout = final_error.stdout or b""
                stderr = final_error.stderr or b""
        stderr += (
            b"\nE00 command timed out; process group terminated="
            + str(terminated).encode("ascii")
            + b".\n"
        )
    except FileNotFoundError as error:
        exit_code = 127
        stdout = b""
        stderr = (str(error) + "\n").encode("utf-8")
    except OSError as error:
        exit_code = 126
        stdout = b""
        stderr = (f"{type(error).__name__}: {error}\n").encode("utf-8")
    finished_at = utc_now()
    return {
        "argv": [str(item) for item in argv],
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "duration_seconds": max(0.0, time.monotonic() - start),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "stdout_bytes": stdout,
        "stderr_bytes": stderr,
    }


def proc_identity(pid: int) -> dict[str, Any] | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    closing = raw.rfind(")")
    opening = raw.find("(")
    if opening < 0 or closing < opening:
        return None
    fields = raw[closing + 2 :].split()
    if len(fields) <= 19:
        return None
    try:
        return {
            "pid": int(pid),
            "ppid": int(fields[1]),
            "start_time_ticks": int(fields[19]),
            "comm": raw[opening + 1 : closing],
        }
    except ValueError:
        return None


def parse_last_json(text: str) -> dict[str, Any] | None:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _canonical_inspect_name(name: str) -> str:
    with_word_boundaries = re.sub(
        r"(?<=[a-z0-9])(?=[A-Z])",
        "_",
        name,
    )
    return re.sub(r"[^A-Za-z0-9]+", "_", with_word_boundaries).strip("_").upper()


def configured_secret_value_names(
    data: bytes,
    environ: Mapping[str, str],
) -> list[str]:
    """Return only names of configured credentials whose bytes are present."""

    return sorted(
        name
        for name in IMAGE_SECRET_VARIABLES
        if (value := environ.get(name))
        and value.encode("utf-8") in data
    )


def _stream_configured_secret_value_names(
    stream: Any,
    environ: Mapping[str, str],
) -> set[str]:
    configured = {
        name: value.encode("utf-8")
        for name in IMAGE_SECRET_VARIABLES
        if (value := environ.get(name))
    }
    if not configured:
        return set()
    overlap = max(len(value) for value in configured.values()) - 1
    tail = b""
    found: set[str] = set()
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            break
        window = tail + chunk
        for name, value in configured.items():
            if name not in found and value in window:
                found.add(name)
        tail = window[-overlap:] if overlap > 0 else b""
    return found


def _safe_tar_parts(name: str) -> tuple[str, ...] | None:
    candidate = PurePosixPath(name)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    parts = tuple(part for part in candidate.parts if part not in ("", "."))
    return parts if parts else None


def _contains_operator_env_filename(parts: Sequence[str]) -> bool:
    return any(part == ".env" or part.startswith(".env.") for part in parts)


def _contains_model_weight_filename(parts: Sequence[str]) -> bool:
    name = parts[-1].lower()
    return (
        name.endswith((".safetensors", ".gguf"))
        or (
            name.endswith(".bin")
            and name.startswith(("pytorch_model", "adapter_model"))
        )
        or (name.endswith(".pth") and name.startswith("consolidated"))
    )


def validate_docker_image_save_archive(
    path: Path,
    *,
    secret_environment: Mapping[str, str],
) -> dict[str, Any]:
    """Validate a Docker image-save archive without exposing secret values."""

    errors: list[str] = []
    configured_secret_names: set[str] = set()
    outer_member_count = 0
    layer_count = 0
    layer_member_count = 0
    operator_env_path_count = 0
    model_weight_path_count = 0
    source_size_bytes: int | None = None

    try:
        file_stat = path.lstat()
        if not stat.S_ISREG(file_stat.st_mode):
            errors.append("image-save input is not a regular file")
        elif file_stat.st_size <= 0:
            errors.append("image-save input is empty")
        else:
            source_size_bytes = file_stat.st_size
    except OSError as error:
        errors.append(
            "image-save input is unavailable: "
            f"{type(error).__name__}"
        )

    if not errors:
        try:
            with tarfile.open(path, mode="r:*") as outer:
                outer_members = outer.getmembers()
                outer_member_count = len(outer_members)
                members_by_name: dict[str, list[tarfile.TarInfo]] = {}
                for member_index, member in enumerate(outer_members):
                    skip_member_content = False
                    members_by_name.setdefault(member.name, []).append(member)
                    parts = _safe_tar_parts(member.name)
                    if parts is None:
                        errors.append(
                            "unsafe outer archive member path at "
                            f"member[{member_index}]"
                        )
                    elif _contains_operator_env_filename(parts):
                        skip_member_content = True
                        operator_env_path_count += 1
                        errors.append(
                            "operator environment file present in outer "
                            f"archive at member[{member_index}]"
                        )
                    metadata = "\0".join(
                        (
                            member.name,
                            member.linkname,
                            member.uname,
                            member.gname,
                            json.dumps(
                                member.pax_headers,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        )
                    ).encode("utf-8", errors="surrogateescape")
                    configured_secret_names.update(
                        configured_secret_value_names(
                            metadata,
                            secret_environment,
                        )
                    )
                    if member.isfile() and not skip_member_content:
                        stream = outer.extractfile(member)
                        if stream is None:
                            errors.append(
                                "outer archive regular member is unreadable at "
                                f"member[{member_index}]"
                            )
                        else:
                            with stream:
                                configured_secret_names.update(
                                    _stream_configured_secret_value_names(
                                        stream,
                                        secret_environment,
                                    )
                                )

                manifest_members = members_by_name.get("manifest.json", [])
                layer_names: list[str] = []
                if (
                    len(manifest_members) != 1
                    or not manifest_members[0].isfile()
                    or manifest_members[0].size > 16 * 1024 * 1024
                ):
                    errors.append(
                        "image-save archive must contain one bounded regular "
                        "manifest.json"
                    )
                else:
                    stream = outer.extractfile(manifest_members[0])
                    if stream is None:
                        errors.append("image-save manifest is unreadable")
                    else:
                        with stream:
                            manifest_raw = stream.read(16 * 1024 * 1024 + 1)
                        try:
                            manifest = json.loads(manifest_raw.decode("utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            errors.append(
                                "image-save manifest is not valid UTF-8 JSON"
                            )
                        else:
                            if (
                                not isinstance(manifest, list)
                                or len(manifest) != 1
                                or not isinstance(manifest[0], dict)
                                or not isinstance(
                                    manifest[0].get("Layers"),
                                    list,
                                )
                                or not manifest[0]["Layers"]
                                or not all(
                                    isinstance(item, str)
                                    and _safe_tar_parts(item) is not None
                                    for item in manifest[0]["Layers"]
                                )
                                or len(set(manifest[0]["Layers"]))
                                != len(manifest[0]["Layers"])
                            ):
                                errors.append(
                                    "image-save manifest does not identify "
                                    "exactly one image with unique safe layers"
                                )
                            else:
                                layer_names = list(manifest[0]["Layers"])

                layer_count = len(layer_names)
                for layer_index, layer_name in enumerate(layer_names):
                    candidates = members_by_name.get(layer_name, [])
                    if len(candidates) != 1 or not candidates[0].isfile():
                        errors.append(
                            "manifest layer is missing, duplicated, or not "
                            f"regular at layer[{layer_index}]"
                        )
                        continue
                    layer_stream = outer.extractfile(candidates[0])
                    if layer_stream is None:
                        errors.append(
                            "manifest layer is unreadable at "
                            f"layer[{layer_index}]"
                        )
                        continue
                    try:
                        with layer_stream:
                            with tarfile.open(
                                fileobj=layer_stream,
                                mode="r|*",
                            ) as layer:
                                for member_index, member in enumerate(layer):
                                    skip_member_content = False
                                    layer_member_count += 1
                                    parts = _safe_tar_parts(member.name)
                                    if parts is None:
                                        errors.append(
                                            "unsafe layer member path at "
                                            f"layer[{layer_index}].member"
                                            f"[{member_index}]"
                                        )
                                    elif _contains_operator_env_filename(parts):
                                        skip_member_content = True
                                        operator_env_path_count += 1
                                        errors.append(
                                            "operator environment file present "
                                            "in image layer at "
                                            f"layer[{layer_index}].member"
                                            f"[{member_index}]"
                                        )
                                    elif _contains_model_weight_filename(parts):
                                        model_weight_path_count += 1
                                        errors.append(
                                            "model weight file present in image "
                                            "layer at "
                                            f"layer[{layer_index}].member"
                                            f"[{member_index}]"
                                        )
                                    metadata = "\0".join(
                                        (
                                            member.name,
                                            member.linkname,
                                            member.uname,
                                            member.gname,
                                            json.dumps(
                                                member.pax_headers,
                                                sort_keys=True,
                                                separators=(",", ":"),
                                            ),
                                        )
                                    ).encode(
                                        "utf-8",
                                        errors="surrogateescape",
                                    )
                                    configured_secret_names.update(
                                        configured_secret_value_names(
                                            metadata,
                                            secret_environment,
                                        )
                                    )
                                    if member.isfile() and not skip_member_content:
                                        member_stream = layer.extractfile(
                                            member
                                        )
                                        if member_stream is None:
                                            errors.append(
                                                "layer regular member is "
                                                "unreadable at "
                                                f"layer[{layer_index}].member"
                                                f"[{member_index}]"
                                            )
                                        else:
                                            with member_stream:
                                                configured_secret_names.update(
                                                    _stream_configured_secret_value_names(
                                                        member_stream,
                                                        secret_environment,
                                                    )
                                                )
                    except (OSError, tarfile.TarError) as error:
                        errors.append(
                            "image layer cannot be parsed at "
                            f"layer[{layer_index}]: "
                            f"{type(error).__name__}"
                        )
        except (OSError, tarfile.TarError) as error:
            errors.append(
                "image-save archive cannot be parsed: "
                f"{type(error).__name__}"
            )

    if configured_secret_names:
        errors.append("image-save archive contains configured credential bytes")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "source_size_bytes": source_size_bytes,
        "outer_member_count": outer_member_count,
        "layer_count": layer_count,
        "layer_member_count": layer_member_count,
        "operator_env_path_count": operator_env_path_count,
        "model_weight_path_count": model_weight_path_count,
        "configured_secret_variables": sorted(configured_secret_names),
    }


def validate_container_image_build_record(
    history_path: Path,
    scan_path: Path,
    *,
    expected_image_config_digest: str,
    secret_environment: Mapping[str, str] | None = None,
) -> tuple[bytes | None, bytes | None, dict[str, Any]]:
    """Validate bounded host-derived image history and layer scan evidence."""

    errors: list[str] = []

    def bounded(path: Path, limit: int, label: str) -> bytes | None:
        try:
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                errors.append(f"{label} is not a regular file")
            elif metadata.st_size <= 0 or metadata.st_size > limit:
                errors.append(f"{label} size is invalid")
            else:
                return path.read_bytes()
        except OSError as error:
            errors.append(f"{label} is unavailable: {type(error).__name__}")
        return None

    history_raw = bounded(history_path, 16 * 1024 * 1024, "image history")
    scan_raw = bounded(scan_path, 1024 * 1024, "image build scan")
    payload: Any = None
    if scan_raw is not None:
        try:
            payload = json.loads(scan_raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            errors.append(
                "image build scan is invalid JSON: "
                f"{type(error).__name__}"
            )
    if isinstance(payload, dict) and scan_raw != json_bytes(payload):
        errors.append("image build scan is not canonical")
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "image_config_digest",
        "image_history",
        "image_layers",
    }:
        errors.append("image build scan field closure differs")
        payload = {}
    history = payload.get("image_history")
    layers = payload.get("image_layers")
    history_keys = {
        "sha256",
        "size_bytes",
        "line_count",
        "configured_secret_variables",
    }
    layer_keys = {
        "status",
        "errors",
        "source_size_bytes",
        "outer_member_count",
        "layer_count",
        "layer_member_count",
        "operator_env_path_count",
        "model_weight_path_count",
        "configured_secret_variables",
    }
    if not isinstance(history, dict) or set(history) != history_keys:
        errors.append("image history scan field closure differs")
        history = {}
    if not isinstance(layers, dict) or set(layers) != layer_keys:
        errors.append("image layer scan field closure differs")
        layers = {}
    if (
        payload.get("schema_version")
        != "kvbench-measurement-container-build-scan-1.0.0"
        or payload.get("image_config_digest")
        != expected_image_config_digest
        or history_raw is None
        or history.get("sha256") != sha256_bytes(history_raw or b"")
        or history.get("size_bytes") != len(history_raw or b"")
        or history.get("line_count") != len((history_raw or b"").splitlines())
        or history.get("configured_secret_variables") != []
        or layers.get("status") != "PASS"
        or layers.get("errors") != []
        or type(layers.get("source_size_bytes")) is not int
        or int(layers.get("source_size_bytes", 0)) <= 0
        or type(layers.get("outer_member_count")) is not int
        or int(layers.get("outer_member_count", 0)) <= 0
        or type(layers.get("layer_count")) is not int
        or int(layers.get("layer_count", 0)) <= 0
        or type(layers.get("layer_member_count")) is not int
        or int(layers.get("layer_member_count", 0)) <= 0
        or layers.get("operator_env_path_count") != 0
        or layers.get("model_weight_path_count") != 0
        or layers.get("configured_secret_variables") != []
    ):
        errors.append("image build scan did not pass exact requirements")
    if secret_environment is not None:
        leaked_names = set()
        for raw in (history_raw, scan_raw):
            if raw is not None:
                leaked_names.update(
                    configured_secret_value_names(raw, secret_environment)
                )
        if leaked_names:
            errors.append(
                "image build evidence contains configured credential bytes"
            )
    return history_raw, scan_raw, {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "image_config_digest": expected_image_config_digest,
        "history_sha256": (
            sha256_bytes(history_raw) if history_raw is not None else None
        ),
        "build_scan_sha256": (
            sha256_bytes(scan_raw) if scan_raw is not None else None
        ),
        "model_weight_path_count": layers.get("model_weight_path_count"),
        "configured_secret_variables": layers.get(
            "configured_secret_variables"
        ),
    }


def _secret_named_inspect_fields(payload: Any) -> list[str]:
    forbidden_parts = {
        "AUTH",
        "AUTHORIZATION",
        "CREDENTIAL",
        "CREDENTIALS",
        "PASSWORD",
        "PASSWD",
        "SECRET",
        "TOKEN",
    }
    forbidden_names = {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "CLOUDFLARE_API_TOKEN",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "KVBENCH_R2_PREFIX",
        "R2_ACCOUNT_ID",
        "R2_BUCKET",
        "R2_ENDPOINT",
    }
    forbidden_scalar_patterns = (
        (
            "credential variable name",
            re.compile(
                r"(?i)\b(?:AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|"
                r"AWS_SESSION_TOKEN|CLOUDFLARE_API_TOKEN|HF_TOKEN|"
                r"HUGGING_FACE_HUB_TOKEN|R2_ACCOUNT_ID)\b"
            ),
        ),
        (
            "authorization header",
            re.compile(
                r"(?i)\bauthorization\s*:\s*(?:bearer|aws4-hmac-sha256)\b"
            ),
        ),
        (
            "private key marker",
            re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
        ),
        (
            "signed credential parameter",
            re.compile(r"(?i)\b(?:x-amz-credential|credential)\s*="),
        ),
        (
            "operator environment file path",
            re.compile(r"(?:^|[/\\])\.env(?:\.|[/\\]|$)"),
        ),
    )
    findings: list[str] = []

    def secret_named(name: str) -> bool:
        canonical = _canonical_inspect_name(name)
        parts = set(canonical.split("_"))
        return (
            canonical in forbidden_names
            or bool(parts & forbidden_parts)
            or "ACCESS_KEY" in canonical
            or "API_KEY" in canonical
            or "PRIVATE_KEY" in canonical
        )

    def visit(value: Any, location: str) -> None:
        if isinstance(value, dict):
            for key_index, (raw_key, child) in enumerate(value.items()):
                key = str(raw_key)
                key_location = f"{location}.key[{key_index}]"
                child_location = f"{location}.value[{key_index}]"
                if secret_named(key):
                    findings.append(
                        f"secret-named JSON key at {key_location}"
                    )
                if _canonical_inspect_name(key) == "ENV" and isinstance(
                    child,
                    list,
                ):
                    for index, entry in enumerate(child):
                        if not isinstance(entry, str) or "=" not in entry:
                            findings.append(
                                "malformed environment entry at "
                                f"{child_location}.item[{index}]"
                            )
                            continue
                        name, _value = entry.split("=", 1)
                        if secret_named(name):
                            findings.append(
                                "secret-named environment variable at "
                                f"{child_location}.item[{index}]"
                            )
                visit(child, child_location)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{location}.item[{index}]")
        elif isinstance(value, str):
            for label, pattern in forbidden_scalar_patterns:
                if pattern.search(value):
                    findings.append(
                        f"secret-shaped scalar ({label}) at {location}"
                    )

    visit(payload, "$")
    return sorted(set(findings))


def validate_container_image_inspect(
    path: Path,
    *,
    expected_image_config_digest: str,
    expected_base_image_digest: str,
    expected_revision: str | None = None,
    secret_environment: Mapping[str, str] | None = None,
) -> tuple[bytes | None, dict[str, Any]]:
    """Validate a sanitized host-side image inspect without exposing values."""
    errors: list[str] = []
    raw: bytes | None = None
    payload: Any = None
    try:
        file_stat = path.lstat()
        if not stat.S_ISREG(file_stat.st_mode):
            errors.append("image inspect input is not a regular file")
        elif file_stat.st_size <= 0:
            errors.append("image inspect input is empty")
        elif file_stat.st_size > 16 * 1024 * 1024:
            errors.append("image inspect input exceeds 16 MiB")
        else:
            raw = path.read_bytes()
    except OSError as error:
        errors.append(
            "image inspect input is unavailable: "
            f"{type(error).__name__}"
        )

    duplicate_key_count = 0

    def reject_duplicate_keys(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        nonlocal duplicate_key_count
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                duplicate_key_count += 1
            result[key] = value
        return result

    if raw is not None and not errors:
        try:
            payload = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=reject_duplicate_keys,
            )
        except UnicodeDecodeError as error:
            errors.append(
                "image inspect input is not UTF-8: "
                f"byte offset {error.start}"
            )
        except json.JSONDecodeError as error:
            errors.append(
                "image inspect input is not valid JSON: "
                f"{error.msg} at line {error.lineno} column {error.colno}"
            )
    if duplicate_key_count:
        errors.append(
            "duplicate JSON key names detected: "
            f"count={duplicate_key_count}"
        )

    if isinstance(payload, list):
        if len(payload) == 1 and isinstance(payload[0], dict):
            payload = payload[0]
        else:
            errors.append(
                "image inspect JSON array must contain exactly one object"
            )
            payload = None
    elif payload is not None and not isinstance(payload, dict):
        errors.append("image inspect JSON root must be one object")
        payload = None

    secret_findings = (
        _secret_named_inspect_fields(payload)
        if isinstance(payload, dict)
        else []
    )
    errors.extend(secret_findings)
    configured_secret_names = (
        configured_secret_value_names(raw, secret_environment)
        if raw is not None and secret_environment is not None
        else []
    )
    errors.extend(
        f"image inspect contains configured credential bytes for {name}"
        for name in configured_secret_names
    )

    observed_config_digest: str | None = None
    observed_base_digest: str | None = None
    observed_revision: str | None = None
    label_matches = False
    forbidden_startup_environment_count = 0
    if isinstance(payload, dict):
        image_id = payload.get("Id")
        if isinstance(image_id, str):
            observed_config_digest = image_id
        else:
            errors.append("image inspect .Id is missing or invalid")
        config = payload.get("Config")
        if not isinstance(config, dict):
            errors.append("image inspect .Config is missing or invalid")
        else:
            environment = config.get("Env")
            if not isinstance(environment, list) or not all(
                isinstance(item, str) and "=" in item
                for item in environment
            ):
                errors.append("image inspect .Config.Env is missing or invalid")
            else:
                environment_names: set[str] = set()
                forbidden_startup_names = {
                    "LD_LIBRARY_PATH",
                    "LD_PRELOAD",
                    "PYTHONHOME",
                    "PYTHONPATH",
                }
                for environment_index, entry in enumerate(environment):
                    name, value = entry.split("=", 1)
                    if name in environment_names:
                        errors.append(
                            "duplicate image environment variable at "
                            f"Config.Env item[{environment_index}]"
                        )
                    environment_names.add(name)
                    if name in forbidden_startup_names and value:
                        forbidden_startup_environment_count += 1
                        errors.append(
                            "nonempty forbidden startup environment variable "
                            f"at Config.Env item[{environment_index}]"
                        )
            labels = config.get("Labels")
            if not isinstance(labels, dict):
                errors.append(
                    "image inspect .Config.Labels is missing or invalid"
                )
            else:
                base_digest = labels.get(MEASUREMENT_BASE_DIGEST_LABEL)
                if isinstance(base_digest, str):
                    observed_base_digest = base_digest
                else:
                    errors.append(
                        "image inspect base-image digest label is missing "
                        "or invalid"
                    )
                expected_labels = {
                    "org.kvbench.measurement.lane": "phase3-phase4-bf16",
                    MEASUREMENT_BASE_DIGEST_LABEL: expected_base_image_digest,
                    "org.kvbench.measurement.requirements-e00.sha256": (
                        sha256_file(PYTHON_LOCK_PATH)
                    ),
                    "org.kvbench.measurement.requirements-phase3.sha256": (
                        sha256_file(ROOT / "preflight" / "requirements-phase3.txt")
                    ),
                }
                if expected_revision is not None:
                    expected_labels["org.opencontainers.image.revision"] = (
                        expected_revision
                    )
                revision = labels.get("org.opencontainers.image.revision")
                if isinstance(revision, str):
                    observed_revision = revision
                label_matches = all(
                    labels.get(name) == expected
                    for name, expected in expected_labels.items()
                )
                if not label_matches:
                    errors.append(
                        "image inspect Measurement Container labels do not "
                        "match committed inputs"
                    )

    config_digest_matches = (
        observed_config_digest == expected_image_config_digest
    )
    base_digest_matches = observed_base_digest == expected_base_image_digest
    if observed_config_digest is not None and not config_digest_matches:
        errors.append("image inspect .Id does not match the supplied digest")
    if observed_base_digest is not None and not base_digest_matches:
        errors.append(
            "image inspect base-image label does not match the supplied digest"
        )

    status = "PASS" if not errors else "FAIL"
    validation = {
        "status": status,
        "errors": errors,
        "image_config_digest": observed_config_digest,
        "image_config_digest_matches": config_digest_matches,
        "base_image_digest_label": MEASUREMENT_BASE_DIGEST_LABEL,
        "base_image_digest": observed_base_digest,
        "base_image_digest_matches": base_digest_matches,
        "source_revision": observed_revision,
        "source_revision_matches": (
            expected_revision is None or observed_revision == expected_revision
        ),
        "measurement_labels_match": label_matches,
        "forbidden_startup_environment_count": (
            forbidden_startup_environment_count
        ),
        "secret_named_field_count": len(secret_findings),
        "configured_secret_value_count": len(configured_secret_names),
        "source_sha256": sha256_bytes(raw) if raw is not None else None,
        "source_size_bytes": len(raw) if raw is not None else None,
    }
    return (raw if status == "PASS" else None), validation


def validate_container_runtime_inspect(
    path: Path,
    *,
    expected_image_config_digest: str,
    expected_hostname: str,
    secret_environment: Mapping[str, str] | None = None,
) -> tuple[bytes | None, dict[str, Any]]:
    """Bind a pre-start Docker container inspect to this running collector."""

    errors: list[str] = []
    raw: bytes | None = None
    payload: Any = None
    try:
        file_stat = path.lstat()
        if not stat.S_ISREG(file_stat.st_mode):
            errors.append("runtime inspect input is not a regular file")
        elif file_stat.st_size <= 0:
            errors.append("runtime inspect input is empty")
        elif file_stat.st_size > 16 * 1024 * 1024:
            errors.append("runtime inspect input exceeds 16 MiB")
        else:
            raw = path.read_bytes()
    except OSError as error:
        errors.append(
            "runtime inspect input is unavailable: "
            f"{type(error).__name__}"
        )

    duplicate_key_count = 0

    def reject_duplicate_keys(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        nonlocal duplicate_key_count
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                duplicate_key_count += 1
            result[key] = value
        return result

    if raw is not None and not errors:
        try:
            payload = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=reject_duplicate_keys,
            )
        except UnicodeDecodeError as error:
            errors.append(
                "runtime inspect input is not UTF-8: "
                f"byte offset {error.start}"
            )
        except json.JSONDecodeError as error:
            errors.append(
                "runtime inspect input is not valid JSON: "
                f"{error.msg} at line {error.lineno} column {error.colno}"
            )
    if duplicate_key_count:
        errors.append(
            "duplicate runtime JSON key names detected: "
            f"count={duplicate_key_count}"
        )
    if isinstance(payload, list):
        if len(payload) == 1 and isinstance(payload[0], dict):
            payload = payload[0]
        else:
            errors.append(
                "runtime inspect JSON array must contain exactly one object"
            )
            payload = None
    elif payload is not None and not isinstance(payload, dict):
        errors.append("runtime inspect JSON root must be one object")
        payload = None

    secret_findings = (
        _secret_named_inspect_fields(payload)
        if isinstance(payload, dict)
        else []
    )
    errors.extend(secret_findings)
    configured_secret_names = (
        configured_secret_value_names(raw, secret_environment)
        if raw is not None and secret_environment is not None
        else []
    )
    errors.extend(
        f"runtime inspect contains configured credential bytes for {name}"
        for name in configured_secret_names
    )

    runtime_id: str | None = None
    observed_image: str | None = None
    configured_image: str | None = None
    configured_hostname: str | None = None
    state_status: str | None = None
    network_mode: str | None = None
    pid_mode: str | None = None
    read_only_root = False
    exact_gpu_request = False
    exact_tmpfs = False
    required_mounts_match = False
    if isinstance(payload, dict):
        candidate_runtime_id = payload.get("Id")
        if (
            isinstance(candidate_runtime_id, str)
            and re.fullmatch(r"[0-9a-f]{64}", candidate_runtime_id)
        ):
            runtime_id = candidate_runtime_id
        else:
            errors.append("runtime inspect .Id is missing or invalid")
        candidate_image = payload.get("Image")
        if isinstance(candidate_image, str):
            observed_image = candidate_image
        else:
            errors.append("runtime inspect .Image is missing or invalid")
        config = payload.get("Config")
        if isinstance(config, dict):
            if isinstance(config.get("Image"), str):
                configured_image = config["Image"]
            else:
                errors.append(
                    "runtime inspect .Config.Image is missing or invalid"
                )
            if isinstance(config.get("Hostname"), str):
                configured_hostname = config["Hostname"]
            else:
                errors.append(
                    "runtime inspect .Config.Hostname is missing or invalid"
                )
        else:
            errors.append("runtime inspect .Config is missing or invalid")
        state = payload.get("State")
        if isinstance(state, dict) and isinstance(state.get("Status"), str):
            state_status = state["Status"]
        else:
            errors.append("runtime inspect .State.Status is missing or invalid")
        host_config = payload.get("HostConfig")
        if isinstance(host_config, dict):
            if isinstance(host_config.get("NetworkMode"), str):
                network_mode = host_config["NetworkMode"]
            else:
                errors.append(
                    "runtime inspect .HostConfig.NetworkMode is missing or invalid"
                )
            if isinstance(host_config.get("PidMode"), str):
                pid_mode = host_config["PidMode"]
            else:
                errors.append(
                    "runtime inspect .HostConfig.PidMode is missing or invalid"
                )
            read_only_root = host_config.get("ReadonlyRootfs") is True
            device_requests = host_config.get("DeviceRequests")
            if isinstance(device_requests, list):
                exact_gpu_request = (
                    len(device_requests) == 1
                    and isinstance(device_requests[0], dict)
                    and device_requests[0].get("DeviceIDs")
                    == [EXPECTED_GPU_UUID]
                    and device_requests[0].get("Capabilities") == [["gpu"]]
                )
            else:
                errors.append(
                    "runtime inspect GPU device requests are missing or invalid"
                )
            tmpfs = host_config.get("Tmpfs")
            expected_tmpfs = {
                "/root": {"rw", "nosuid", "nodev", "size=2g"},
                "/tmp": {"rw", "nosuid", "nodev", "size=8g"},
            }
            if isinstance(tmpfs, dict) and all(
                isinstance(path, str) and isinstance(options, str)
                for path, options in tmpfs.items()
            ):
                tmpfs_options_well_formed = all(
                    len(options.split(",")) == len(set(options.split(",")))
                    for options in tmpfs.values()
                )
                observed_tmpfs = {
                    path: set(options.split(","))
                    for path, options in tmpfs.items()
                }
                exact_tmpfs = (
                    tmpfs_options_well_formed
                    and observed_tmpfs == expected_tmpfs
                )
            else:
                errors.append(
                    "runtime inspect tmpfs configuration is missing or invalid"
                )
        else:
            errors.append("runtime inspect .HostConfig is missing or invalid")
        mounts = payload.get("Mounts")
        if isinstance(mounts, list):
            expected_bind_modes = {
                "/workspace": False,
                "/workspace/artifacts/phase6a/container_g0": True,
                "/run/kvbench/image-inspect.json": False,
                "/run/kvbench/runtime-inspect.json": False,
                "/run/kvbench/image-history.jsonl": False,
                "/run/kvbench/image-build-scan.json": False,
            }
            expected_tmpfs_modes = {
                "/root": True,
                "/tmp": True,
            }
            valid_mounts = [
                item
                for item in mounts
                if isinstance(item, dict)
                and item.get("Type") in {"bind", "tmpfs"}
                and isinstance(item.get("Destination"), str)
                and isinstance(item.get("RW"), bool)
            ]
            bind_modes = {
                item["Destination"]: item["RW"]
                for item in valid_mounts
                if item["Type"] == "bind"
            }
            tmpfs_modes = {
                item["Destination"]: item["RW"]
                for item in valid_mounts
                if item["Type"] == "tmpfs"
            }
            required_mounts_match = (
                len(valid_mounts) == len(mounts)
                and len(
                    {
                        item["Destination"]
                        for item in valid_mounts
                    }
                )
                == len(mounts)
                and bind_modes == expected_bind_modes
                and tmpfs_modes in ({}, expected_tmpfs_modes)
                and len(mounts)
                == len(expected_bind_modes) + len(tmpfs_modes)
            )
        else:
            errors.append("runtime inspect .Mounts is missing or invalid")

    if observed_image != expected_image_config_digest:
        errors.append("runtime inspect .Image does not match the supplied digest")
    if configured_image != expected_image_config_digest:
        errors.append(
            "runtime inspect .Config.Image does not match the supplied digest"
        )
    if (
        runtime_id is None
        or configured_hostname != runtime_id[:12]
        or expected_hostname != configured_hostname
    ):
        errors.append(
            "runtime inspect container ID/hostname does not match the collector"
        )
    if state_status != "created":
        errors.append("runtime inspect was not captured before container start")
    if network_mode != "none":
        errors.append("runtime inspect network mode is not none")
    if pid_mode != "host":
        errors.append("runtime inspect PID mode is not host")
    if not read_only_root:
        errors.append("runtime inspect root filesystem is not read-only")
    if not exact_gpu_request:
        errors.append("runtime inspect does not bind the exact certified GPU UUID")
    if not exact_tmpfs:
        errors.append("runtime inspect tmpfs isolation does not exactly match")
    if not required_mounts_match:
        errors.append("runtime inspect required mount isolation does not match")

    status = "PASS" if not errors else "FAIL"
    validation = {
        "status": status,
        "errors": errors,
        "runtime_container_id": runtime_id,
        "image_config_digest": observed_image,
        "image_config_digest_matches": (
            observed_image == expected_image_config_digest
            and configured_image == expected_image_config_digest
        ),
        "hostname_matches": (
            runtime_id is not None
            and configured_hostname == runtime_id[:12]
            and expected_hostname == configured_hostname
        ),
        "pre_start_state": state_status == "created",
        "network_disabled": network_mode == "none",
        "host_pid_mode": pid_mode == "host",
        "read_only_root": read_only_root,
        "exact_gpu_request": exact_gpu_request,
        "exact_tmpfs": exact_tmpfs,
        "required_mounts_match": required_mounts_match,
        "secret_named_field_count": len(secret_findings),
        "configured_secret_value_count": len(configured_secret_names),
        "source_sha256": sha256_bytes(raw) if raw is not None else None,
        "source_size_bytes": len(raw) if raw is not None else None,
    }
    return (raw if status == "PASS" else None), validation


class EvidenceRecorder:
    def __init__(
        self,
        stage: Path,
        base_environment: dict[str, str],
        *,
        project_python: Path = VENV_PYTHON,
    ) -> None:
        self.stage = stage
        self.base_environment = base_environment
        self.project_python = project_python
        self.commands: list[dict[str, Any]] = []
        self.command_index: dict[str, dict[str, Any]] = {}
        self.snapshots: list[dict[str, Any]] = []
        self.audit_errors: list[str] = []
        self.audit_outcome_paths: list[str] = []

    def _finish_command(
        self,
        *,
        command_id: str,
        purpose: str,
        result: dict[str, Any],
        environment: dict[str, str],
        expected_exit_codes: Sequence[int],
        stdout_relative: str,
        stderr_relative: str,
    ) -> dict[str, Any]:
        if command_id in self.command_index:
            raise RuntimeError(f"duplicate evidence command id: {command_id}")
        record = {
            "id": command_id,
            "purpose": purpose,
            "started_at_utc": result["started_at_utc"],
            "finished_at_utc": result["finished_at_utc"],
            "duration_seconds": float(result["duration_seconds"]),
            "argv": list(result["argv"]),
            "cwd": str(ROOT),
            "relevant_environment": relevant_environment(environment),
            "exit_code": int(result["exit_code"]),
            "expected_exit_codes": [int(item) for item in expected_exit_codes],
            "matched_expectation": int(result["exit_code"])
            in set(expected_exit_codes),
            "stdout": output_ref(self.stage, stdout_relative),
            "stderr": output_ref(self.stage, stderr_relative),
        }
        self.commands.append(record)
        self.command_index[command_id] = record
        return record

    def record_precollected(
        self,
        *,
        command_id: str,
        purpose: str,
        result: dict[str, Any],
        environment: dict[str, str],
        expected_exit_codes: Sequence[int] = (0,),
    ) -> dict[str, Any]:
        command_id = safe_id(command_id)
        stdout_relative = f"commands/{command_id}.stdout.txt"
        stderr_relative = f"commands/{command_id}.stderr.txt"
        write_exclusive(self.stage / stdout_relative, result["stdout_bytes"])
        write_exclusive(self.stage / stderr_relative, result["stderr_bytes"])
        return self._finish_command(
            command_id=command_id,
            purpose=purpose,
            result=result,
            environment=environment,
            expected_exit_codes=expected_exit_codes,
            stdout_relative=stdout_relative,
            stderr_relative=stderr_relative,
        )

    def run(
        self,
        command_id: str,
        purpose: str,
        argv: Sequence[str],
        *,
        environment: dict[str, str] | None = None,
        expected_exit_codes: Sequence[int] = (0,),
        timeout_seconds: float = 60.0,
    ) -> dict[str, Any]:
        command_id = safe_id(command_id)
        env = self.base_environment if environment is None else environment
        print(f"[e00] {command_id}: {' '.join(str(item) for item in argv)}", flush=True)
        result = run_memory(
            argv,
            environment=env,
            cwd=ROOT,
            timeout_seconds=timeout_seconds,
        )
        return self.record_precollected(
            command_id=command_id,
            purpose=purpose,
            result=result,
            environment=env,
            expected_exit_codes=expected_exit_codes,
        )

    def _snapshot_from_command(
        self,
        *,
        phase: str,
        command_record: dict[str, Any],
    ) -> dict[str, Any]:
        stdout_path = self.stage / command_record["stdout"]["path"]
        payload = parse_last_json(stdout_path.read_text(encoding="utf-8", errors="replace"))
        required_list_fields = (
            "graphics_processes",
            "allowed_compute_processes",
            "foreign_compute_processes",
            "unknown_processes",
        )
        payload_errors: list[str] = []
        if payload is None:
            payload_errors.append("process query emitted no JSON object")
        else:
            if not isinstance(
                payload.get("captured_at_utc"), str
            ) or not payload["captured_at_utc"]:
                payload_errors.append("captured_at_utc is missing or invalid")
            if type(payload.get("query_exit_code")) is not int:
                payload_errors.append("query_exit_code is missing or invalid")
            if not isinstance(payload.get("errors"), list):
                payload_errors.append("errors is missing or invalid")
            if not isinstance(payload.get("subcommands"), list):
                payload_errors.append("subcommands is missing or invalid")
            for key in required_list_fields:
                if not isinstance(payload.get(key), list):
                    payload_errors.append(f"{key} is missing or invalid")
            if (
                type(payload.get("query_exit_code")) is int
                and int(payload["query_exit_code"])
                != int(command_record["exit_code"])
            ):
                payload_errors.append(
                    "payload query_exit_code disagrees with command exit code"
                )
            if (
                isinstance(payload.get("errors"), list)
                and payload["errors"]
                and payload.get("query_exit_code") == 0
            ):
                payload_errors.append("process query reported errors with exit zero")
        snapshot_exit_code = int(command_record["exit_code"])
        if payload_errors:
            self.audit_errors.append(
                f"invalid process snapshot {command_record['id']}: "
                + "; ".join(payload_errors)
            )
            payload = {}
        snapshot = {
            "phase": phase,
            "command_id": command_record["id"],
            "captured_at_utc": payload.get(
                "captured_at_utc", command_record["finished_at_utc"]
            ),
            "query_argv": command_record["argv"],
            "query_exit_code": snapshot_exit_code,
            "raw_stdout": command_record["stdout"],
            "raw_stderr": command_record["stderr"],
            "graphics_processes": payload.get("graphics_processes", []),
            "allowed_compute_processes": payload.get(
                "allowed_compute_processes", []
            ),
            "foreign_compute_processes": payload.get(
                "foreign_compute_processes", []
            ),
            "unknown_processes": payload.get("unknown_processes", []),
        }
        self.snapshots.append(snapshot)
        return snapshot

    def record_initial_snapshot(
        self,
        *,
        command_record: dict[str, Any],
    ) -> dict[str, Any]:
        return self._snapshot_from_command(
            phase="before",
            command_record=command_record,
        )

    def capture_snapshot(
        self,
        *,
        target_command_id: str,
        phase: str,
        supervised_root: dict[str, Any] | None,
        environment: dict[str, str],
    ) -> dict[str, Any]:
        query_id = safe_id(f"process_{target_command_id}_{phase}")
        argv = [str(self.project_python), str(PROCESS_QUERY_PATH)]
        if supervised_root is not None:
            argv.extend(
                [
                    "--supervised-root-pid",
                    str(supervised_root["pid"]),
                    "--supervised-root-start-ticks",
                    str(supervised_root["start_time_ticks"]),
                ]
            )
        record = self.run(
            query_id,
            f"GPU process isolation snapshot {phase} {target_command_id}",
            argv,
            environment=environment,
            expected_exit_codes=(0,),
            timeout_seconds=20.0,
        )
        return self._snapshot_from_command(phase=phase, command_record=record)

    def run_supervised(
        self,
        command_id: str,
        purpose: str,
        argv: Sequence[str],
        *,
        environment: dict[str, str],
        expected_exit_codes: Sequence[int] = (0,),
        timeout_seconds: float = 300.0,
        audit_ready_timeout_seconds: float = 120.0,
    ) -> dict[str, Any]:
        command_id = safe_id(command_id)
        self.capture_snapshot(
            target_command_id=command_id,
            phase="before",
            supervised_root=None,
            environment=self.base_environment,
        )

        if command_id in self.command_index:
            raise RuntimeError(f"duplicate evidence command id: {command_id}")
        stdout_relative = f"commands/{command_id}.stdout.txt"
        stderr_relative = f"commands/{command_id}.stderr.txt"
        stdout_path = self.stage / stdout_relative
        stderr_path = self.stage / stderr_relative
        stdout_path.parent.mkdir(parents=True, exist_ok=True)

        audit_directory = self.stage / "audit"
        audit_directory.mkdir(parents=True, exist_ok=True)
        ready_path = audit_directory / f"{command_id}.ready.json"
        release_path = audit_directory / f"{command_id}.release"
        if ready_path.exists() or release_path.exists():
            raise RuntimeError(f"duplicate audit handshake path for {command_id}")
        effective_argv = [
            *[str(item) for item in argv],
            "--audit-ready-file",
            str(ready_path),
            "--audit-release-file",
            str(release_path),
            "--audit-timeout-seconds",
            str(timeout_seconds),
        ]

        print(f"[e00] {command_id}: {' '.join(effective_argv)}", flush=True)
        started_at = utc_now()
        start = time.monotonic()
        timed_out = False
        audit_error_start = len(self.audit_errors)
        checkpoint_verified = False
        premature_release = False
        release_published = False
        process_group_drained = True
        ready_identity_record: dict[str, int] | None = None
        root: dict[str, Any] | None = None
        with stdout_path.open("xb") as stdout_handle, stderr_path.open("xb") as stderr_handle:
            try:
                process = subprocess.Popen(
                    effective_argv,
                    cwd=str(ROOT),
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    start_new_session=True,
                )
            except FileNotFoundError as error:
                stderr_handle.write((str(error) + "\n").encode("utf-8"))
                exit_code = 127
                root = None
            else:
                root = proc_identity(process.pid)
                if root is None:
                    self.audit_errors.append(
                        f"could not read PID/start identity for {command_id}"
                    )
                ready_deadline = min(
                    start + timeout_seconds,
                    time.monotonic() + audit_ready_timeout_seconds,
                )
                while not ready_path.exists() and process.poll() is None and time.monotonic() < ready_deadline:
                    time.sleep(0.05)

                premature_release = release_path.exists()
                if premature_release:
                    self.audit_errors.append(
                        f"supervised command created its own release file: {command_id}"
                    )
                if (
                    ready_path.exists()
                    and not premature_release
                    and process.poll() is None
                    and root is not None
                ):
                    try:
                        ready_stat = ready_path.lstat()
                        if (
                            not stat.S_ISREG(ready_stat.st_mode)
                            or ready_stat.st_nlink != 1
                        ):
                            raise ValueError(
                                "audit ready path is not a singly linked regular file"
                            )
                        ready_payload = json.loads(
                            ready_path.read_text(encoding="utf-8")
                        )
                        ready_identity = (
                            int(ready_payload["pid"]),
                            int(ready_payload["process_start_time_ticks"]),
                        )
                        ready_identity_record = {
                            "pid": ready_identity[0],
                            "process_start_time_ticks": ready_identity[1],
                        }
                        if ready_payload.get("protocol") != "e00-process-audit-v1":
                            raise ValueError("unexpected audit checkpoint protocol")
                        if ready_payload.get("cuda_work_complete") is not True:
                            raise ValueError("CUDA work-complete marker is absent")
                        live_ready = proc_identity(ready_identity[0])
                        if (
                            live_ready is None
                            or live_ready["start_time_ticks"] != ready_identity[1]
                        ):
                            raise ValueError(
                                "audit checkpoint PID/start identity is not live"
                            )
                        snapshot = self.capture_snapshot(
                            target_command_id=command_id,
                            phase="during",
                            supervised_root=root,
                            environment=self.base_environment,
                        )
                        observed = {
                            (int(item["pid"]), int(item["process_start_time_ticks"]))
                            for item in snapshot["allowed_compute_processes"]
                            if item["gpu_uuid"]
                            == environment.get("CUDA_VISIBLE_DEVICES")
                        }
                        if ready_identity not in observed:
                            raise ValueError(
                                "during snapshot did not observe the ready CUDA "
                                "PID/start identity"
                            )
                        checkpoint_verified = True
                    except Exception as error:
                        self.audit_errors.append(
                            f"invalid audit checkpoint for {command_id}: "
                            f"{type(error).__name__}: {error}"
                        )
                else:
                    self.audit_errors.append(
                        f"supervised command did not reach a releasable CUDA "
                        f"audit checkpoint: {command_id}"
                    )

                if checkpoint_verified:
                    try:
                        publish_atomic_exclusive(release_path, b"release\n")
                        release_published = True
                    except (FileExistsError, OSError) as error:
                        checkpoint_verified = False
                        self.audit_errors.append(
                            f"could not release supervised command {command_id}: {error}"
                        )
                if not checkpoint_verified and process.poll() is None:
                    terminated = terminate_process_group(process)
                    process_group_drained = terminated
                    stderr_handle.write(
                        f"\nE00 audit checkpoint failed; process group "
                        f"terminated={terminated}.\n".encode("utf-8")
                    )
                    if not terminated:
                        raise RuntimeError(
                            f"could not drain failed supervised process group: "
                            f"{command_id}"
                        )
                remaining = max(0.1, timeout_seconds - (time.monotonic() - start))
                try:
                    exit_code = int(process.wait(timeout=remaining))
                except subprocess.TimeoutExpired:
                    timed_out = True
                    terminated = terminate_process_group(process)
                    process_group_drained = terminated
                    if not terminated:
                        raise RuntimeError(
                            f"could not drain timed-out supervised process group: "
                            f"{command_id}"
                        )
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired as error:
                        raise RuntimeError(
                            f"could not reap timed-out supervised process: "
                            f"{command_id}"
                        ) from error
                    exit_code = 124
                    stderr_handle.write(
                        b"\nE00 supervised command timed out; process group terminated="
                        + str(terminated).encode("ascii")
                        + b".\n"
                    )
                else:
                    if process_group_exists(process.pid):
                        terminated = terminate_process_group(process)
                        process_group_drained = terminated
                        self.audit_errors.append(
                            f"supervised command left descendant processes: {command_id}"
                        )
                        if not terminated:
                            raise RuntimeError(
                                f"could not drain lingering supervised process "
                                f"group: {command_id}"
                            )
                        if exit_code == 0:
                            exit_code = 125
                        stderr_handle.write(
                            f"\nE00 terminated lingering process group: {terminated}.\n".encode("utf-8")
                        )
            stdout_handle.flush()
            stderr_handle.flush()
            os.fsync(stdout_handle.fileno())
            os.fsync(stderr_handle.fileno())

        outcome_relative = f"audit/{command_id}.outcome.json"
        write_exclusive(
            self.stage / outcome_relative,
            json_bytes(
                {
                    "protocol": "e00-process-audit-v1",
                    "command_id": command_id,
                    "checkpoint_verified": checkpoint_verified,
                    "premature_release_detected": premature_release,
                    "release_published_by_collector": release_published,
                    "ready_identity": ready_identity_record,
                    "supervised_root_identity": root,
                    "timed_out": timed_out,
                    "process_group_drained": process_group_drained,
                    "audit_errors": self.audit_errors[audit_error_start:],
                }
            ),
        )
        self.audit_outcome_paths.append(outcome_relative)

        finished_at = utc_now()
        result = {
            "argv": effective_argv,
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "duration_seconds": max(0.0, time.monotonic() - start),
            "exit_code": exit_code,
            "timed_out": timed_out,
        }
        record = self._finish_command(
            command_id=command_id,
            purpose=purpose,
            result=result,
            environment=environment,
            expected_exit_codes=expected_exit_codes,
            stdout_relative=stdout_relative,
            stderr_relative=stderr_relative,
        )
        self.capture_snapshot(
            target_command_id=command_id,
            phase="after",
            supervised_root=None,
            environment=self.base_environment,
        )
        return record

    def command_ok(self, command_id: str) -> bool:
        record = self.command_index.get(command_id)
        return bool(record and record["matched_expectation"])

    def command_stdout(self, command_id: str) -> str:
        record = self.command_index[command_id]
        return (
            self.stage / record["stdout"]["path"]
        ).read_text(encoding="utf-8", errors="replace")

    def command_stderr(self, command_id: str) -> str:
        record = self.command_index[command_id]
        return (
            self.stage / record["stderr"]["path"]
        ).read_text(encoding="utf-8", errors="replace")

    def command_text(self, command_id: str) -> str:
        record = self.command_index[command_id]
        stdout = (
            self.stage / record["stdout"]["path"]
        ).read_text(encoding="utf-8", errors="replace")
        stderr = (
            self.stage / record["stderr"]["path"]
        ).read_text(encoding="utf-8", errors="replace")
        return stdout + "\n" + stderr

    def command_payload(self, command_id: str) -> dict[str, Any] | None:
        if command_id not in self.command_index:
            return None
        return parse_last_json(self.command_text(command_id))


def parse_gpu_rows(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in csv.reader(text.splitlines(), skipinitialspace=True):
        if not row:
            continue
        if len(row) != len(GPU_QUERY_FIELDS):
            raise ValueError(
                f"GPU query returned {len(row)} columns; expected {len(GPU_QUERY_FIELDS)}"
            )
        values = {
            field: value.strip()
            for field, value in zip(GPU_QUERY_FIELDS, row, strict=True)
        }
        if any(not value for value in values.values()):
            raise ValueError("GPU query returned an empty required field")
        rows.append(values)
    return rows


def nullable_float(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or stripped in {"[N/A]", "N/A", "Unknown", "Not Supported"}:
        return None
    try:
        numeric = float(stripped)
    except ValueError:
        return None
    return numeric if math.isfinite(numeric) else None


def nullable_nonnegative_float(value: str | None) -> float | None:
    numeric = nullable_float(value)
    return numeric if numeric is not None and numeric >= 0 else None


def nullable_int(value: str | None) -> int | None:
    numeric = nullable_float(value)
    if numeric is None or numeric < 0 or not numeric.is_integer():
        return None
    return int(numeric)


def nonnegative_int_value(value: Any, *, minimum: int = 0) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        return None
    return value


def boolean_value(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def capability_from_mapping(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    major = nonnegative_int_value(value.get("major"), minimum=1)
    minor = nonnegative_int_value(value.get("minor"))
    return capability(f"{major}.{minor}") if major is not None and minor is not None else None


def nonempty_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def matching_string(
    value: Any,
    pattern: str,
) -> str | None:
    normalized = nonempty_string(value)
    if normalized is None:
        return None
    return normalized if re.fullmatch(pattern, normalized) else None


def normalized_mode(value: str | None, *, not_supported: bool = False) -> str | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered.startswith("enabled"):
        return "enabled"
    if lowered.startswith("disabled"):
        return "disabled"
    if not_supported and ("n/a" in lowered or "not supported" in lowered):
        return "not_supported"
    return "unknown"


def capability(raw: str | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    match = re.fullmatch(r"([0-9]+)\.([0-9]+)", raw.strip())
    if match is None:
        return None
    major = int(match.group(1))
    minor = int(match.group(2))
    if major <= 0:
        return None
    return {"major": major, "minor": minor, "text": f"{major}.{minor}"}


def parse_key_value_file(text: str, delimiter: str = "=") -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if delimiter not in line:
            continue
        key, value = line.split(delimiter, 1)
        values[key.strip()] = value.strip().strip('"')
    return values


def parse_lscpu(text: str) -> dict[str, str]:
    payload = json.loads(text)
    result: dict[str, str] = {}
    for item in payload.get("lscpu", []):
        field = str(item.get("field", "")).rstrip(":")
        result[field] = str(item.get("data", ""))
    return result


def parse_memtotal_bytes(text: str) -> int | None:
    match = re.search(r"^MemTotal:\s+([0-9]+)\s+kB$", text, re.MULTILINE)
    return None if match is None else int(match.group(1)) * 1024


def native_host_validation_errors(
    *,
    container_detection_ok: bool,
    container_detection_output: str,
    cgroup_evidence: dict[str, str | None],
    present_container_markers: Sequence[str],
    cgroup_container_tokens: Sequence[str],
    caller_container_marker: bool,
) -> list[str]:
    errors: list[str] = []
    if not container_detection_ok:
        errors.append(
            "systemd-detect-virt did not return non-container exit 1"
        )
    if container_detection_output != "none":
        errors.append(
            f"unexpected systemd-detect-virt output: "
            f"{container_detection_output!r}"
        )
    for command_id, output in cgroup_evidence.items():
        if output is None or not output.strip():
            errors.append(
                f"required cgroup evidence is unavailable: {command_id}"
            )
    if present_container_markers:
        errors.append(
            f"container markers present: {list(present_container_markers)!r}"
        )
    if cgroup_container_tokens:
        errors.append(
            f"container cgroup tokens present: "
            f"{list(cgroup_container_tokens)!r}"
        )
    if caller_container_marker:
        errors.append("caller environment contains a container marker")
    return errors


def measurement_container_validation_errors(
    *,
    container_detection_ok: bool,
    container_detection_output: str,
    cgroup_evidence: dict[str, str | None],
    present_container_markers: Sequence[str],
    image_identity_ok: bool,
) -> list[str]:
    """Fail closed unless the collector is visibly running in Docker."""
    errors: list[str] = []
    if not container_detection_ok:
        errors.append("systemd-detect-virt did not return container exit 0")
    if container_detection_output != "docker":
        errors.append(
            "systemd-detect-virt did not identify the Docker runtime: "
            f"{container_detection_output!r}"
        )
    for command_id, output in cgroup_evidence.items():
        if output is None or not output.strip():
            errors.append(
                f"required cgroup evidence is unavailable: {command_id}"
            )
    if "/.dockerenv" not in present_container_markers:
        errors.append("required Docker marker /.dockerenv is absent")
    if not image_identity_ok:
        errors.append(
            "sanitized container image inspect identity verification failed"
        )
    return errors


def verification_outputs_are_empty(
    *,
    command_ok: bool,
    stdout: str,
    stderr: str,
) -> bool:
    return command_ok and stdout == "" and stderr == ""


def dpkg_verification_result(
    *,
    command_ok: bool,
    stdout: str,
    stderr: str,
    allow_container_policy_omissions: bool,
) -> dict[str, Any]:
    """Accept only dpkg omissions imposed by the pinned image policy."""
    expected_omissions: list[str] = []
    unexpected_lines: list[str] = []
    for line in stdout.splitlines():
        marker = "missing     "
        if not line.startswith(marker):
            unexpected_lines.append(line)
            continue
        path_text = line[len(marker) :]
        path = PurePosixPath(path_text)
        normalized = path.is_absolute() and not ({".", ".."} & set(path.parts))
        allowed = allow_container_policy_omissions and (
            path_text in DPKG_EXPECTED_MISSING_PATHS
            or any(
                path_text.startswith(prefix)
                for prefix in DPKG_EXPECTED_MISSING_PREFIXES
            )
        )
        if normalized and allowed:
            expected_omissions.append(path_text)
        else:
            unexpected_lines.append(line)
    status = (
        "PASS"
        if command_ok and stderr == "" and not unexpected_lines
        else "FAIL"
    )
    return {
        "status": status,
        "command_ok": command_ok,
        "container_policy_omissions_allowed": (
            allow_container_policy_omissions
        ),
        "stderr_empty": stderr == "",
        "byte_empty": stdout == "" and stderr == "",
        "expected_omission_count": len(expected_omissions),
        "allowed_missing_prefixes": list(DPKG_EXPECTED_MISSING_PREFIXES),
        "allowed_missing_paths": sorted(DPKG_EXPECTED_MISSING_PATHS),
        "unexpected_lines": unexpected_lines,
    }


def dpkg_ownership_matches(
    *,
    command_ok: bool,
    stdout: str,
    stderr: str,
    expected_package: str,
    expected_path: str,
) -> bool:
    return (
        command_ok
        and stderr == ""
        and stdout == f"{expected_package}: {expected_path}\n"
    )


def extract_version(tool: str, text: str) -> str | None:
    patterns = {
        "nvcc": (r"V([0-9]+(?:\.[0-9]+)+)",),
        "compute-sanitizer": (r"Version\s+([^\s]+)",),
        "cuobjdump": (r"V([0-9]+(?:\.[0-9]+)+)",),
        "nvdisasm": (r"V([0-9]+(?:\.[0-9]+)+)",),
        "ncu": (r"Version\s+([^\s]+)", r"([0-9]{4}\.[0-9.]+)"),
        "nsys": (r"version\s+([^\s]+)",),
        "python": (r"Python\s+([^\s]+)",),
        "c++": (r"\)\s+([^\s]+)",),
        "ninja": (r"^([^\s]+)",),
    }
    for pattern in patterns.get(tool, (r"([0-9]+(?:\.[0-9]+)+)",)):
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1)
    return None


def parse_requirements_lock(path: Path) -> dict[str, str]:
    requirements: dict[str, str] = {}
    pattern = re.compile(
        r"^([A-Za-z0-9_.-]+)==([^\s]+)\s+--hash=sha256:[0-9a-f]{64}$"
    )
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("--"):
            continue
        match = pattern.fullmatch(stripped)
        if match is None:
            raise ValueError(f"unhashed or malformed requirement: {stripped}")
        normalized = re.sub(r"[-_.]+", "-", match.group(1)).lower()
        requirements[normalized] = match.group(2)
    if not requirements:
        raise ValueError("Python requirements lock is empty")
    return requirements


def parse_exact_dpkg_inventory(
    text: str,
) -> tuple[dict[str, tuple[str, str]], list[str]]:
    observed: dict[str, tuple[str, str]] = {}
    errors: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        parts = line.split("\t")
        if (
            len(parts) != 3
            or not parts[0]
            or not parts[1]
            or not parts[2]
        ):
            errors.append(
                f"malformed dpkg inventory row at line {line_number}"
            )
            continue
        name, version, architecture = parts
        if name in observed:
            errors.append(
                f"duplicate dpkg inventory package at line {line_number}"
            )
            continue
        observed[name] = (version, architecture)
    if not observed:
        errors.append("dpkg inventory is empty")
    return observed, errors


def parse_exact_python_freeze(
    text: str,
) -> tuple[dict[str, str], list[str]]:
    observed: dict[str, str] = {}
    errors: list[str] = []
    pattern = re.compile(
        r"([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s]+)"
    )
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = pattern.fullmatch(stripped)
        if match is None:
            errors.append(
                f"malformed Python freeze row at line {line_number}"
            )
            continue
        name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
        if name in observed:
            errors.append(
                f"duplicate Python freeze distribution at line {line_number}"
            )
            continue
        observed[name] = match.group(2)
    if not observed:
        errors.append("Python freeze is empty")
    return observed, errors


def verify_dependency_locks(
    system_lock: dict[str, Any],
    dpkg_text: str,
    pip_payload: Any,
    *,
    full_dpkg_text: str | None = None,
    phase3_freeze_text: str | None = None,
    phase3_dependency_check_ok: bool | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    package_observations: list[dict[str, Any]] = []
    observed_dpkg: dict[str, tuple[str, str]] = {}
    for line in dpkg_text.splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            observed_dpkg[parts[0]] = (parts[1], parts[2])
    for package in system_lock.get("dpkg_packages", []):
        observed = observed_dpkg.get(package["name"])
        expected = (package["version"], package["architecture"])
        matches = observed == expected
        package_observations.append(
            {
                "name": package["name"],
                "expected_version": expected[0],
                "expected_architecture": expected[1],
                "observed_version": observed[0] if observed is not None else None,
                "observed_architecture": (
                    observed[1] if observed is not None else None
                ),
                "matches": matches,
            }
        )
        if not matches:
            errors.append(
                f"dpkg mismatch {package['name']}: "
                f"observed={observed!r} expected={expected!r}"
            )

    tool_observations: list[dict[str, Any]] = []
    for tool in system_lock.get("tools", []):
        invocation = Path(tool["invocation_path"])
        resolved = invocation.resolve() if invocation.exists() else None
        expected_resolved = Path(tool["resolved_path"])
        observed_hash = (
            sha256_file(resolved)
            if resolved is not None and resolved.is_file()
            else None
        )
        path_matches = resolved == expected_resolved
        hash_matches = observed_hash == tool["sha256"]
        tool_observations.append(
            {
                "name": tool["name"],
                "expected_invocation_path": tool["invocation_path"],
                "expected_resolved_path": tool["resolved_path"],
                "observed_resolved_path": (
                    str(resolved) if resolved is not None else None
                ),
                "expected_sha256": tool["sha256"],
                "observed_sha256": observed_hash,
                "path_matches": path_matches,
                "hash_matches": hash_matches,
            }
        )
        if not path_matches:
            errors.append(
                f"tool path mismatch {tool['name']}: "
                f"{resolved!s} != {expected_resolved!s}"
            )
        if not hash_matches:
            errors.append(
                f"tool hash mismatch {tool['name']}: "
                f"{observed_hash} != {tool['sha256']}"
            )

    locked_python = parse_requirements_lock(PYTHON_LOCK_PATH)
    installed: dict[str, str] = {}
    if isinstance(pip_payload, list):
        for package in pip_payload:
            if not isinstance(package, dict):
                continue
            name = re.sub(r"[-_.]+", "-", str(package.get("name", ""))).lower()
            if name in installed:
                errors.append(f"duplicate installed Python distribution: {name}")
            installed[name] = str(package.get("version", ""))
    else:
        errors.append("pip list did not return a JSON array")
    for name in sorted(set(installed) - set(locked_python)):
        errors.append(f"unlocked Python distribution present: {name}=={installed[name]}")
    for name, version in locked_python.items():
        if installed.get(name) != version:
            errors.append(
                f"Python package mismatch {name}: "
                f"{installed.get(name)!r} != {version!r}"
            )

    result: dict[str, Any] = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "python_locked_package_count": len(locked_python),
        "python_installed_package_count": len(installed),
        "dpkg_locked_package_count": len(system_lock.get("dpkg_packages", [])),
        "tool_hash_count": len(system_lock.get("tools", [])),
        "package_observations": package_observations,
        "tool_hash_observations": tool_observations,
    }
    container_inputs = (
        full_dpkg_text,
        phase3_freeze_text,
        phase3_dependency_check_ok,
    )
    if any(item is not None for item in container_inputs):
        if (
            not isinstance(full_dpkg_text, str)
            or not isinstance(phase3_freeze_text, str)
            or not isinstance(phase3_dependency_check_ok, bool)
        ):
            errors.append(
                "container dependency validation inputs are incomplete"
            )
        else:
            expected_dpkg: dict[str, tuple[str, str]] = {}
            expected_dpkg_errors: list[str] = []
            for package_index, package in enumerate(
                system_lock.get("dpkg_packages", [])
            ):
                if not isinstance(package, dict):
                    expected_dpkg_errors.append(
                        "invalid expected dpkg package at "
                        f"index {package_index}"
                    )
                    continue
                name = package.get("name")
                version = package.get("version")
                architecture = package.get("architecture")
                if not all(
                    isinstance(item, str) and item
                    for item in (name, version, architecture)
                ):
                    expected_dpkg_errors.append(
                        "invalid expected dpkg package fields at "
                        f"index {package_index}"
                    )
                    continue
                if name in expected_dpkg:
                    expected_dpkg_errors.append(
                        "duplicate expected dpkg package at "
                        f"index {package_index}"
                    )
                    continue
                expected_dpkg[name] = (version, architecture)

            observed_full_dpkg, full_dpkg_errors = (
                parse_exact_dpkg_inventory(full_dpkg_text)
            )
            errors.extend(expected_dpkg_errors)
            errors.extend(full_dpkg_errors)
            for name in sorted(set(expected_dpkg) - set(observed_full_dpkg)):
                errors.append(f"full dpkg inventory missing package: {name}")
            for name in sorted(set(observed_full_dpkg) - set(expected_dpkg)):
                errors.append(f"full dpkg inventory has extra package: {name}")
            for name in sorted(set(expected_dpkg) & set(observed_full_dpkg)):
                if observed_full_dpkg[name] != expected_dpkg[name]:
                    errors.append(
                        f"full dpkg inventory mismatch {name}: "
                        f"observed={observed_full_dpkg[name]!r} "
                        f"expected={expected_dpkg[name]!r}"
                    )

            expected_phase3 = parse_requirements_lock(
                ROOT / "preflight" / "requirements-phase3.txt"
            )
            observed_phase3, phase3_freeze_errors = (
                parse_exact_python_freeze(phase3_freeze_text)
            )
            errors.extend(phase3_freeze_errors)
            for name in sorted(set(expected_phase3) - set(observed_phase3)):
                errors.append(
                    f"Phase 3 target freeze missing distribution: {name}"
                )
            for name in sorted(set(observed_phase3) - set(expected_phase3)):
                errors.append(
                    f"Phase 3 target freeze has extra distribution: {name}"
                )
            for name in sorted(set(expected_phase3) & set(observed_phase3)):
                if observed_phase3[name] != expected_phase3[name]:
                    errors.append(
                        f"Phase 3 target freeze mismatch {name}: "
                        f"{observed_phase3[name]!r} != "
                        f"{expected_phase3[name]!r}"
                    )
            if phase3_dependency_check_ok is not True:
                errors.append(
                    "Phase 3 target-aware dependency check failed"
                )
            result["container_dependency_validation"] = {
                "full_dpkg_inventory_matches": (
                    not expected_dpkg_errors
                    and not full_dpkg_errors
                    and observed_full_dpkg == expected_dpkg
                ),
                "expected_dpkg_package_count": len(expected_dpkg),
                "observed_dpkg_package_count": len(observed_full_dpkg),
                "phase3_target_freeze_matches": (
                    not phase3_freeze_errors
                    and observed_phase3 == expected_phase3
                ),
                "expected_phase3_distribution_count": len(expected_phase3),
                "observed_phase3_distribution_count": len(observed_phase3),
                "phase3_target_dependency_check_passed": (
                    phase3_dependency_check_ok is True
                ),
            }
    result["status"] = "PASS" if not errors else "FAIL"
    return result


def verify_platform_lock(
    system_lock: dict[str, Any],
    observed: dict[str, Any],
) -> dict[str, Any]:
    expected = {
        **system_lock["platform"],
        **system_lock["scope"],
        "cuda_home": system_lock["environment"]["cuda_home"],
        "python_environment": system_lock["environment"]["python_environment"],
        "python_requirements_lock": system_lock["environment"][
            "python_requirements_lock"
        ],
    }
    comparisons = {
        key: {
            "expected": expected[key],
            "observed": observed.get(key),
            "matches": observed.get(key) == expected[key],
        }
        for key in expected
    }
    errors = [
        f"platform lock mismatch {key}: "
        f"{item['observed']!r} != {item['expected']!r}"
        for key, item in comparisons.items()
        if not item["matches"]
    ]
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "comparisons": comparisons,
    }


def validate_measurement_container_system_lock(
    payload: Any,
    *,
    expected_image_config_digest: str | None = None,
    expected_source_revision: str | None = None,
) -> dict[str, Any]:
    """Validate the reviewed, build-observed container lock schema."""

    errors: list[str] = []
    if not isinstance(payload, dict):
        return {
            "status": "FAIL",
            "errors": [
                "Measurement Container system lock root is not an object"
            ],
            "dpkg_package_count": 0,
            "tool_count": 0,
        }

    expected_top_level = {
        "schema_version",
        "purpose",
        "scope",
        "platform",
        "environment",
        "dpkg_packages",
        "tools",
        "observation",
    }
    if set(payload) != expected_top_level:
        errors.append("Measurement Container system lock top-level keys differ")
    if type(payload.get("schema_version")) is not int or payload.get(
        "schema_version"
    ) != 1:
        errors.append("Measurement Container system lock schema version differs")
    if payload.get("purpose") != MEASUREMENT_CONTAINER_LOCK_PURPOSE:
        errors.append("Measurement Container system lock purpose differs")
    if payload.get("scope") != {
        "execution_environment_kind": "measurement_container",
        "container_digest_status": (
            "verified_against_sanitized_image_inspect"
        ),
        "performance_claim_eligible": False,
    }:
        errors.append("Measurement Container system lock scope differs")
    if payload.get("platform") != {
        "distribution": "ubuntu",
        "distribution_version": "24.04",
        "dpkg_architecture": "amd64",
        "machine": "x86_64",
        "python_abi": "cp312",
    }:
        errors.append("Measurement Container system lock platform differs")
    if payload.get("environment") != {
        "cuda_home": "/usr/local/cuda-13.0",
        "python_environment": "/opt/kvbench/.venv",
        "python_requirements_lock": "preflight/requirements-e00.txt",
    }:
        errors.append("Measurement Container system lock environment differs")

    packages = payload.get("dpkg_packages")
    package_names: list[str] = []
    if not isinstance(packages, list) or not packages:
        errors.append("Measurement Container dpkg inventory is absent")
        packages = []
    for index, package in enumerate(packages):
        if not isinstance(package, dict) or set(package) != {
            "name",
            "version",
            "architecture",
        }:
            errors.append(f"invalid dpkg inventory item at index {index}")
            continue
        values = tuple(
            package.get(field) for field in ("name", "version", "architecture")
        )
        if not all(
            isinstance(value, str)
            and bool(value)
            and re.search(r"[\x00-\x1f\x7f]", value) is None
            for value in values
        ):
            errors.append(f"invalid dpkg inventory fields at index {index}")
            continue
        package_names.append(package["name"])
        if package["architecture"] not in {"all", "amd64"}:
            errors.append(f"invalid dpkg architecture at index {index}")
    if package_names != sorted(package_names):
        errors.append("Measurement Container dpkg inventory is not sorted")
    if len(package_names) != len(set(package_names)):
        errors.append("Measurement Container dpkg inventory has duplicate names")

    expected_specs = {
        name: (invocation_path, dpkg_package, version_parser)
        for name, invocation_path, dpkg_package, version_parser in (
            MEASUREMENT_CONTAINER_LOCK_TOOL_SPECS
        )
    }
    tools = payload.get("tools")
    tool_names: list[str] = []
    if not isinstance(tools, list):
        errors.append("Measurement Container tool inventory is absent")
        tools = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            errors.append(f"invalid tool inventory item at index {index}")
            continue
        name = tool.get("name")
        tool_names.append(name if isinstance(name, str) else "")
        spec = expected_specs.get(name)
        if spec is None:
            errors.append(f"unexpected tool inventory item at index {index}")
            continue
        invocation_path, dpkg_package, version_parser = spec
        expected_fields = {
            "name",
            "invocation_path",
            "resolved_path",
            "dpkg_package",
            "version",
            "sha256",
        }
        if version_parser is not None:
            expected_fields.add("reported_version")
        if set(tool) != expected_fields:
            errors.append(f"tool inventory fields differ for {name}")
        if tool.get("invocation_path") != invocation_path:
            errors.append(f"tool invocation path differs for {name}")
        if tool.get("dpkg_package") != dpkg_package:
            errors.append(f"tool package owner differs for {name}")
        resolved_path = tool.get("resolved_path")
        if (
            not isinstance(resolved_path, str)
            or not PurePosixPath(resolved_path).is_absolute()
            or str(PurePosixPath(resolved_path)) != resolved_path
        ):
            errors.append(f"tool resolved path is invalid for {name}")
        if not isinstance(tool.get("version"), str) or not tool.get("version"):
            errors.append(f"tool version is invalid for {name}")
        if re.fullmatch(r"[0-9a-f]{64}", str(tool.get("sha256", ""))) is None:
            errors.append(f"tool SHA-256 is invalid for {name}")
        if version_parser is not None and (
            not isinstance(tool.get("reported_version"), str)
            or not tool.get("reported_version")
        ):
            errors.append(f"reported tool version is invalid for {name}")
        if dpkg_package not in package_names:
            errors.append(f"tool package owner is absent for {name}")
    expected_tool_names = sorted(expected_specs)
    if tool_names != expected_tool_names:
        errors.append("Measurement Container tool inventory set/order differs")
    if len(tool_names) != len(set(tool_names)):
        errors.append("Measurement Container tool inventory has duplicate names")

    observation = payload.get("observation")
    if not isinstance(observation, dict) or set(observation) != {
        "authority",
        "candidate_image_config_digest",
        "candidate_source_revision",
    }:
        errors.append("Measurement Container lock observation metadata differs")
        observation = {}
    if observation.get("authority") != MEASUREMENT_CONTAINER_LOCK_AUTHORITY:
        errors.append("Measurement Container lock observation authority differs")
    candidate_digest = observation.get("candidate_image_config_digest")
    candidate_revision = observation.get("candidate_source_revision")
    if re.fullmatch(r"sha256:[0-9a-f]{64}", str(candidate_digest)) is None:
        errors.append("candidate image config digest is invalid")
    if re.fullmatch(r"[0-9a-f]{40}", str(candidate_revision)) is None:
        errors.append("candidate source revision is invalid")
    if (
        expected_image_config_digest is not None
        and candidate_digest != expected_image_config_digest
    ):
        errors.append("candidate image config digest differs")
    if (
        expected_source_revision is not None
        and candidate_revision != expected_source_revision
    ):
        errors.append("candidate source revision differs")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "dpkg_package_count": len(packages),
        "tool_count": len(tools),
    }


def load_measurement_container_system_lock(
    path: Path,
    *,
    expected_image_config_digest: str | None = None,
    expected_source_revision: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Load only a regular, canonical, structurally valid container lock."""

    errors: list[str] = []
    raw: bytes | None = None
    payload: Any = None
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            errors.append("Measurement Container system lock is not regular")
        elif metadata.st_size <= 0 or metadata.st_size > 16 * 1024 * 1024:
            errors.append("Measurement Container system lock size is invalid")
        else:
            raw = path.read_bytes()
    except OSError as error:
        errors.append(
            "Measurement Container system lock is unavailable: "
            f"{type(error).__name__}"
        )

    duplicate_key_count = 0

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate_key_count
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                duplicate_key_count += 1
            result[key] = value
        return result

    if raw is not None:
        try:
            payload = json.loads(
                raw.decode("utf-8"), object_pairs_hook=reject_duplicate_keys
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            errors.append(
                "Measurement Container system lock JSON is invalid: "
                f"{type(error).__name__}"
            )
    if duplicate_key_count:
        errors.append("Measurement Container system lock has duplicate keys")
    validation = validate_measurement_container_system_lock(
        payload,
        expected_image_config_digest=expected_image_config_digest,
        expected_source_revision=expected_source_revision,
    )
    errors.extend(validation["errors"])
    if raw is not None and isinstance(payload, dict) and raw != json_bytes(payload):
        errors.append("Measurement Container system lock is not canonical JSON")
    result = {
        **validation,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "canonical_json": not errors,
    }
    return (payload if not errors else None), result


def _run_measurement_lock_observation_command(
    command_id: str,
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
) -> tuple[bytes, bytes]:
    try:
        result = subprocess.run(
            list(command),
            cwd=ROOT,
            env=dict(environment),
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(
            f"lock observation command unavailable: {command_id}; "
            f"error={type(error).__name__}"
        ) from None
    if result.returncode != 0:
        raise RuntimeError(
            f"lock observation command failed: {command_id}; "
            f"exit_code={result.returncode}"
        )
    if (
        len(result.stdout) > 16 * 1024 * 1024
        or len(result.stderr) > 16 * 1024 * 1024
    ):
        raise RuntimeError(f"lock observation output is too large: {command_id}")
    return result.stdout, result.stderr


def _measurement_lock_owner_matches(
    text: str,
    *,
    expected_package: str,
    expected_path: str,
) -> bool:
    for line in text.splitlines():
        owner_text, separator, owned_path = line.partition(": ")
        if not separator or owned_path != expected_path:
            continue
        owners = {
            item.strip().split(":", 1)[0]
            for item in owner_text.split(",")
            if item.strip()
        }
        if expected_package in owners:
            return True
    return False


def observe_measurement_container_system_lock(
    *,
    image_config_digest: str,
    base_image_digest: str,
    source_revision: str,
    image_inspect_path: Path,
) -> dict[str, Any]:
    """Observe one untrusted candidate; this does not promote authority."""

    if any(os.environ.get(name) for name in IMAGE_SECRET_VARIABLES):
        raise RuntimeError("credential variables reached the lock observer")
    _raw_inspect, image_validation = validate_container_image_inspect(
        image_inspect_path,
        expected_image_config_digest=image_config_digest,
        expected_base_image_digest=base_image_digest,
        expected_revision=source_revision,
        secret_environment=os.environ,
    )
    if image_validation["status"] != "PASS":
        raise RuntimeError("candidate image identity validation failed")
    if not Path("/.dockerenv").is_file():
        raise RuntimeError("lock observation is not running inside Docker")

    environment = {
        "LC_ALL": "C",
        "LANG": "C",
        "TZ": "UTC",
        "PATH": (
            "/opt/kvbench/.venv/bin:/usr/local/cuda-13.0/bin:"
            "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        ),
        "CUDA_HOME": "/usr/local/cuda-13.0",
        "LD_LIBRARY_PATH": "",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    detection_stdout, detection_stderr = (
        _run_measurement_lock_observation_command(
            "container_detection",
            ["/usr/bin/systemd-detect-virt", "--container"],
            environment=environment,
        )
    )
    if decode(detection_stdout).strip() != "docker" or detection_stderr:
        raise RuntimeError("Docker runtime detection did not match")

    dpkg_format = "${Package}\t${Version}\t${Architecture}\n"
    dpkg_stdout, dpkg_stderr = _run_measurement_lock_observation_command(
        "complete_dpkg_inventory",
        ["/usr/bin/dpkg-query", "-W", f"-f={dpkg_format}"],
        environment=environment,
    )
    if dpkg_stderr:
        raise RuntimeError("complete dpkg inventory wrote stderr")
    dpkg_text = decode(dpkg_stdout)
    observed_dpkg, dpkg_errors = parse_exact_dpkg_inventory(dpkg_text)
    if dpkg_errors or not observed_dpkg:
        raise RuntimeError("complete dpkg inventory is malformed")
    packages = [
        {"name": name, "version": values[0], "architecture": values[1]}
        for name, values in sorted(observed_dpkg.items())
    ]

    tools: list[dict[str, Any]] = []
    for name, invocation_text, dpkg_package, version_parser in sorted(
        MEASUREMENT_CONTAINER_LOCK_TOOL_SPECS
    ):
        invocation = Path(invocation_text)
        try:
            resolved = invocation.resolve(strict=True)
        except OSError as error:
            raise RuntimeError(
                f"required tool is unavailable: {name}; "
                f"error={type(error).__name__}"
            ) from None
        if not resolved.is_file():
            raise RuntimeError(f"required tool is not a regular file: {name}")
        if dpkg_package not in observed_dpkg:
            raise RuntimeError(f"tool owner package is absent: {name}")
        owner_stdout, owner_stderr = _run_measurement_lock_observation_command(
            f"owner_{safe_id(name)}",
            ["/usr/bin/dpkg-query", "-S", str(resolved)],
            environment=environment,
        )
        if owner_stderr or not _measurement_lock_owner_matches(
            decode(owner_stdout),
            expected_package=dpkg_package,
            expected_path=str(resolved),
        ):
            raise RuntimeError(f"tool package ownership differs: {name}")
        tool: dict[str, Any] = {
            "name": name,
            "invocation_path": invocation_text,
            "resolved_path": str(resolved),
            "dpkg_package": dpkg_package,
            "version": observed_dpkg[dpkg_package][0],
            "sha256": sha256_file(resolved),
        }
        if version_parser is not None:
            version_stdout, version_stderr = (
                _run_measurement_lock_observation_command(
                    f"version_{safe_id(name)}",
                    [invocation_text, "--version"],
                    environment=environment,
                )
            )
            reported_version = extract_version(
                version_parser,
                decode(version_stdout + b"\n" + version_stderr),
            )
            if reported_version is None:
                raise RuntimeError(f"tool version is unparseable: {name}")
            tool["version"] = reported_version
            tool["reported_version"] = reported_version
        tools.append(tool)

    pip_stdout, _pip_stderr = _run_measurement_lock_observation_command(
        "base_pip_list",
        [
            str(MEASUREMENT_CONTAINER_PYTHON),
            "-m",
            "pip",
            "list",
            "--format=json",
        ],
        environment=environment,
    )
    try:
        pip_payload = json.loads(decode(pip_stdout))
    except json.JSONDecodeError:
        raise RuntimeError("base Python inventory is invalid JSON") from None
    _run_measurement_lock_observation_command(
        "base_pip_check",
        [str(MEASUREMENT_CONTAINER_PYTHON), "-m", "pip", "check"],
        environment=environment,
    )
    phase3_stdout, _phase3_stderr = (
        _run_measurement_lock_observation_command(
            "phase3_python_freeze",
            [
                str(MEASUREMENT_CONTAINER_PYTHON),
                "-m",
                "pip",
                "freeze",
                "--path",
                "/opt/kvbench/.phase3/site-packages",
            ],
            environment=environment,
        )
    )
    phase3_environment = dict(environment)
    phase3_environment["PYTHONPATH"] = (
        "/opt/kvbench/.phase3/site-packages"
    )
    _run_measurement_lock_observation_command(
        "phase3_pip_check",
        [str(MEASUREMENT_CONTAINER_PYTHON), "-m", "pip", "check"],
        environment=phase3_environment,
    )

    lock = {
        "schema_version": 1,
        "purpose": MEASUREMENT_CONTAINER_LOCK_PURPOSE,
        "scope": {
            "execution_environment_kind": "measurement_container",
            "container_digest_status": (
                "verified_against_sanitized_image_inspect"
            ),
            "performance_claim_eligible": False,
        },
        "platform": {
            "distribution": "ubuntu",
            "distribution_version": "24.04",
            "dpkg_architecture": "amd64",
            "machine": "x86_64",
            "python_abi": "cp312",
        },
        "environment": {
            "cuda_home": "/usr/local/cuda-13.0",
            "python_environment": "/opt/kvbench/.venv",
            "python_requirements_lock": "preflight/requirements-e00.txt",
        },
        "dpkg_packages": packages,
        "tools": tools,
        "observation": {
            "authority": MEASUREMENT_CONTAINER_LOCK_AUTHORITY,
            "candidate_image_config_digest": image_config_digest,
            "candidate_source_revision": source_revision,
        },
    }
    structure_validation = validate_measurement_container_system_lock(
        lock,
        expected_image_config_digest=image_config_digest,
        expected_source_revision=source_revision,
    )
    if structure_validation["status"] != "PASS":
        raise RuntimeError("observed container lock is structurally invalid")
    dependency_validation = verify_dependency_locks(
        lock,
        dpkg_text,
        pip_payload,
        full_dpkg_text=dpkg_text,
        phase3_freeze_text=decode(phase3_stdout),
        phase3_dependency_check_ok=True,
    )
    if dependency_validation["status"] != "PASS":
        raise RuntimeError("candidate package/tool/Python identity differs")
    os_release = parse_key_value_file(
        Path("/etc/os-release").read_text(encoding="utf-8")
    )
    architecture_stdout, architecture_stderr = (
        _run_measurement_lock_observation_command(
            "dpkg_architecture",
            ["/usr/bin/dpkg", "--print-architecture"],
            environment=environment,
        )
    )
    machine_stdout, machine_stderr = _run_measurement_lock_observation_command(
        "machine_architecture",
        ["/usr/bin/uname", "-m"],
        environment=environment,
    )
    if architecture_stderr or machine_stderr:
        raise RuntimeError("platform identity commands wrote stderr")
    platform_validation = verify_platform_lock(
        lock,
        {
            "distribution": str(os_release.get("ID", "")).lower(),
            "distribution_version": os_release.get("VERSION_ID"),
            "dpkg_architecture": decode(architecture_stdout).strip(),
            "machine": decode(machine_stdout).strip(),
            "python_abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
            "execution_environment_kind": "measurement_container",
            "container_digest_status": (
                "verified_against_sanitized_image_inspect"
            ),
            "performance_claim_eligible": False,
            "cuda_home": "/usr/local/cuda-13.0",
            "python_environment": "/opt/kvbench/.venv",
            "python_requirements_lock": "preflight/requirements-e00.txt",
        },
    )
    if platform_validation["status"] != "PASS":
        raise RuntimeError("candidate platform identity differs")
    return lock


def result_envelope(
    status: str,
    *,
    reason: str | None,
    command_ids: Sequence[str] = (),
    evidence_file_ids: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "status": status,
        "reason": None if status == "PASS" else (reason or "required check did not pass"),
        "command_ids": list(dict.fromkeys(command_ids)),
        "evidence_file_ids": list(dict.fromkeys(evidence_file_ids)),
    }


def gate_check(
    name: str,
    status: str,
    *,
    reason: str | None = None,
    command_ids: Sequence[str] = (),
    evidence_file_ids: Sequence[str] = (),
) -> dict[str, Any]:
    if name not in ALL_GATE_NAMES:
        raise ValueError(f"unknown G0 check: {name}")
    payload = result_envelope(
        status,
        reason=reason,
        command_ids=command_ids,
        evidence_file_ids=evidence_file_ids,
    )
    return {"name": name, **payload}


def command_file_ids(
    recorder: EvidenceRecorder,
    command_ids: Sequence[str],
) -> list[str]:
    result: list[str] = []
    for command_id in command_ids:
        command = recorder.command_index.get(command_id)
        if command is None:
            continue
        result.extend(
            [
                artifact_id(command["stdout"]["path"]),
                artifact_id(command["stderr"]["path"]),
            ]
        )
    return list(dict.fromkeys(result))


def file_role(relative: str) -> str:
    suffix = Path(relative).suffix.lower()
    if relative.startswith("commands/"):
        return "command_output"
    if relative.startswith("validation/"):
        return "validation_record"
    if relative.endswith(".so"):
        return "extension_binary"
    if suffix in {".o", ".d"} or "build.ninja" in relative:
        return "build_intermediate"
    if relative.endswith("build_metadata.json"):
        return "build_metadata"
    return "supporting_evidence"


def enumerate_evidence_files(stage: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(
        stage.rglob("*"),
        key=lambda item: item.relative_to(stage).as_posix(),
    ):
        if not path.is_file():
            continue
        if path.is_symlink():
            raise RuntimeError(f"symlink is not allowed in E00 evidence: {path}")
        relative = path.relative_to(stage).as_posix()
        if relative in {"manifest.json", "checksums.sha256", "COMPLETE"}:
            continue
        records.append(
            {
                "id": artifact_id(relative),
                "path": relative,
                "role": file_role(relative),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def evidence_reference_errors(
    stage: Path,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    actual_files = enumerate_evidence_files(stage)
    declared_files = manifest["evidence"]["files"]
    if declared_files != actual_files:
        errors.append("manifest evidence.files does not exactly match staged payload files")
    actual_by_path = {item["path"]: item for item in actual_files}
    actual_ids = {item["id"] for item in actual_files}
    if len(actual_by_path) != len(actual_files) or len(actual_ids) != len(actual_files):
        errors.append("staged evidence paths or artifact IDs are not unique")

    commands = manifest["evidence"]["commands"]
    declared_command_ids = [item["id"] for item in commands]
    command_id_set = set(declared_command_ids)
    if len(declared_command_ids) != len(command_id_set):
        errors.append("duplicate evidence command IDs")

    referenced_file_ids: set[str] = set()
    referenced_command_ids: set[str] = set()

    def collect(value: Any, location: str) -> None:
        if isinstance(value, dict):
            path_value = value.get("path")
            sha_value = value.get("sha256")
            size_value = value.get("size_bytes")
            has_artifact_fields = all(
                key in value for key in ("path", "sha256", "size_bytes")
            )
            if has_artifact_fields and not (
                path_value is None and sha_value is None and size_value is None
            ):
                if (
                    not isinstance(path_value, str)
                    or not isinstance(sha_value, str)
                    or not isinstance(size_value, int)
                ):
                    errors.append(f"incomplete artifact reference at {location}")
                else:
                    observed = actual_by_path.get(path_value)
                    if observed is None:
                        errors.append(
                            f"artifact reference at {location} is missing: {path_value}"
                        )
                    elif (
                        sha_value != observed["sha256"]
                        or size_value != observed["size_bytes"]
                    ):
                        errors.append(
                            f"artifact reference at {location} does not match staged bytes: {path_value}"
                        )

            for key, child in value.items():
                child_location = f"{location}/{key}"
                if key == "evidence_file_ids" and isinstance(child, list):
                    referenced_file_ids.update(str(item) for item in child)
                elif key == "verification_evidence_file_id" and isinstance(child, str):
                    referenced_file_ids.add(child)
                elif key in {"command_ids", "cuobjdump_command_ids"} and isinstance(
                    child, list
                ):
                    referenced_command_ids.update(str(item) for item in child)
                elif key == "command_id" and isinstance(child, str):
                    referenced_command_ids.add(child)
                collect(child, child_location)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                collect(child, f"{location}/{index}")

    collect(manifest, "<root>")

    missing_file_ids = sorted(referenced_file_ids - actual_ids)
    if missing_file_ids:
        errors.append(
            f"missing referenced evidence file IDs: {missing_file_ids!r}"
        )
    missing_command_ids = sorted(referenced_command_ids - command_id_set)
    if missing_command_ids:
        errors.append(
            f"missing referenced evidence command IDs: {missing_command_ids!r}"
        )
    return errors


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def verify_contract_git_state(
    *,
    staged_output: bytes,
    flags_output: bytes,
    paths: Sequence[str],
) -> dict[str, Any]:
    errors: list[str] = []
    expected_paths = set(paths)
    staged_entries: dict[str, dict[str, str]] = {}
    for raw_entry in staged_output.split(b"\0"):
        if not raw_entry:
            continue
        try:
            metadata, raw_path = raw_entry.split(b"\t", 1)
            mode, object_id, stage_number = metadata.decode("ascii").split()
            relative = raw_path.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as error:
            errors.append(f"malformed git index entry: {error}")
            continue
        if relative in staged_entries:
            errors.append(f"duplicate git index entry: {relative}")
        staged_entries[relative] = {
            "mode": mode,
            "object_id": object_id,
            "stage": stage_number,
        }

    flag_entries: dict[str, str] = {}
    for raw_entry in flags_output.split(b"\0"):
        if not raw_entry:
            continue
        try:
            tag, raw_path = raw_entry[:1].decode("ascii"), raw_entry[2:]
            if raw_entry[1:2] != b" ":
                raise ValueError("missing flag separator")
            relative = raw_path.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as error:
            errors.append(f"malformed git flag entry: {error}")
            continue
        flag_entries[relative] = tag

    if set(staged_entries) != expected_paths:
        errors.append(
            "Git index contract paths differ: "
            f"missing={sorted(expected_paths - set(staged_entries))!r}, "
            f"extra={sorted(set(staged_entries) - expected_paths)!r}"
        )
    if set(flag_entries) != expected_paths:
        errors.append(
            "Git flag contract paths differ: "
            f"missing={sorted(expected_paths - set(flag_entries))!r}, "
            f"extra={sorted(set(flag_entries) - expected_paths)!r}"
        )

    observations: list[dict[str, Any]] = []
    for relative in sorted(expected_paths):
        entry = staged_entries.get(relative)
        path = ROOT / relative
        local_mode: str | None = None
        local_object_id: str | None = None
        try:
            file_stat = path.lstat()
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError("contract path is not a regular file")
            local_mode = (
                "100755" if file_stat.st_mode & 0o111 else "100644"
            )
            local_object_id = git_blob_sha1(path)
        except (OSError, ValueError) as error:
            errors.append(f"cannot authenticate contract path {relative}: {error}")

        index_matches = bool(
            entry is not None
            and entry["stage"] == "0"
            and entry["mode"] == local_mode
            and entry["object_id"] == local_object_id
        )
        flag = flag_entries.get(relative)
        safe_flag = flag == "H"
        observations.append(
            {
                "path": relative,
                "index_mode": entry["mode"] if entry is not None else None,
                "local_mode": local_mode,
                "index_object_id": (
                    entry["object_id"] if entry is not None else None
                ),
                "local_object_id": local_object_id,
                "index_stage": entry["stage"] if entry is not None else None,
                "git_flag": flag,
                "index_matches_local_bytes": index_matches,
                "ordinary_tracked_flag": safe_flag,
            }
        )
        if not index_matches:
            errors.append(
                f"contract bytes/mode do not match the Git index: {relative}"
            )
        if not safe_flag:
            errors.append(
                f"contract path has assume-unchanged/skip-worktree or "
                f"unexpected Git flag {flag!r}: {relative}"
            )

    return {
        "status": "PASS" if not errors else "FAIL",
        "git_object_format": "sha1",
        "errors": errors,
        "files": observations,
    }


def source_artifacts(paths: Sequence[str]) -> list[dict[str, str]]:
    artifacts: list[dict[str, str]] = []
    for relative in paths:
        path = ROOT / relative
        artifacts.append({"path": relative, "sha256": sha256_file(path)})
    return artifacts


def status_from_commands(
    recorder: EvidenceRecorder,
    command_ids: Sequence[str],
) -> bool:
    return bool(command_ids) and all(recorder.command_ok(item) for item in command_ids)


def sanitizer_error_count(tool: str, text: str) -> int | None:
    if tool == "racecheck":
        matches = re.findall(
            r"RACECHECK SUMMARY:\s*([0-9]+) hazards displayed \(([0-9]+) errors, ([0-9]+) warnings\)",
            text,
        )
        if len(matches) != 1:
            return None
        return sum(int(value) for value in matches[0])

    matches = re.findall(r"ERROR SUMMARY:\s*([0-9]+)\s+errors?", text)
    if len(matches) != 1:
        return None
    errors = int(matches[0])
    if tool == "memcheck":
        leak_matches = re.findall(
            r"LEAK SUMMARY:\s*([0-9]+) bytes leaked in ([0-9]+) allocations?", text
        )
        if len(leak_matches) != 1:
            return None
        leaked_bytes, allocations = (int(value) for value in leak_matches[0])
        if leaked_bytes != 0 or allocations != 0:
            return errors + 1
    return errors


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    import jsonschema

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(
        schema,
        format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER,
    )
    return [
        f"{'/'.join(str(item) for item in error.absolute_path) or '<root>'}: {error.message}"
        for error in sorted(validator.iter_errors(manifest), key=lambda item: list(item.path))
    ]


def finalize_stage(
    *,
    stage: Path,
    final: Path,
    manifest: dict[str, Any],
) -> None:
    manifest_data = json_bytes(manifest)
    write_exclusive(stage / "manifest.json", manifest_data)

    reference_errors = evidence_reference_errors(stage, manifest)
    if reference_errors:
        raise RuntimeError(f"manifest evidence cross-reference failed: {reference_errors[0]}")
    container_inventory = manifest.get("schema_version") in {
        "e00-manifest-1.1.0",
        "kvbench-phase6a-bf16-container-parity-1.0.0",
    }
    if container_inventory:
        inventory_items = []
        for path in sorted(
            stage.rglob("*"),
            key=lambda item: item.relative_to(stage).as_posix(),
        ):
            if not path.is_file():
                continue
            relative = path.relative_to(stage).as_posix()
            if relative in {
                "artifact_inventory.json",
                "checksums.sha256",
                "COMPLETE",
            }:
                continue
            inventory_items.append(
                {
                    "path": relative,
                    "role": (
                        "manifest"
                        if relative == "manifest.json"
                        else file_role(relative)
                    ),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        write_exclusive(
            stage / "artifact_inventory.json",
            json_bytes(
                {
                    "schema_version": "kvbench-artifact-inventory-1.0.0",
                    "run_id": manifest["run"]["id"],
                    "files": inventory_items,
                    "excluded_control_files": [
                        "artifact_inventory.json",
                        "checksums.sha256",
                        "COMPLETE",
                    ],
                }
            ),
        )
    ledger_entries: list[tuple[str, str]] = []
    for path in sorted(
        stage.rglob("*"),
        key=lambda item: item.relative_to(stage).as_posix(),
    ):
        if not path.is_file():
            continue
        relative = path.relative_to(stage).as_posix()
        if relative in {"checksums.sha256", "COMPLETE"}:
            continue
        ledger_entries.append((sha256_file(path), relative))
    ledger_data = "".join(
        f"{digest}  {relative}\n" for digest, relative in ledger_entries
    ).encode("utf-8")
    write_exclusive(stage / "checksums.sha256", ledger_data)
    ledger_sha256 = sha256_bytes(ledger_data)

    ledger_paths = {relative for _, relative in ledger_entries}
    current_payload_paths = {
        path.relative_to(stage).as_posix()
        for path in stage.rglob("*")
        if path.is_file()
        and path.relative_to(stage).as_posix()
        not in {"checksums.sha256", "COMPLETE"}
    }
    if current_payload_paths != ledger_paths:
        raise RuntimeError("checksum ledger does not exactly cover current payload files")

    complete_payload = {
        "run_id": manifest["run"]["id"],
        "status": manifest["run"]["status"],
        "manifest_sha256": sha256_bytes(manifest_data),
        **(
            {
                "artifact_inventory_sha256": sha256_file(
                    stage / "artifact_inventory.json"
                )
            }
            if container_inventory
            else {}
        ),
        "checksum_ledger_path": "checksums.sha256",
        "checksum_ledger_sha256": ledger_sha256,
        "written_last": True,
        "finished_at_utc": manifest["run"]["finished_at_utc"],
    }
    write_exclusive(stage / "COMPLETE", json_bytes(complete_payload))

    for expected, relative in ledger_entries:
        observed = sha256_file(stage / relative)
        if observed != expected:
            raise RuntimeError(
                f"checksum verification failed before final rename: {relative}"
            )
    if sha256_file(stage / "checksums.sha256") != ledger_sha256:
        raise RuntimeError("checksum ledger self-verification failed")
    loaded_complete = json.loads((stage / "COMPLETE").read_text(encoding="utf-8"))
    if loaded_complete["checksum_ledger_sha256"] != ledger_sha256:
        raise RuntimeError("completion marker does not authenticate checksum ledger")

    for path in sorted(stage.rglob("*"), reverse=True):
        if path.is_file():
            os.chmod(path, 0o444)
        elif path.is_dir():
            os.chmod(path, 0o555)
    os.chmod(stage, 0o555)
    rename_noreplace(stage, final)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--measurement-container",
        action="store_true",
        help="run E00 inside the certified Measurement Container candidate",
    )
    mode_group.add_argument(
        "--observe-measurement-container-system-lock",
        action="store_true",
        help="observe an untrusted candidate lock without granting authority",
    )
    parser.add_argument(
        "--container-image-reference",
        help="non-authoritative human-readable OCI image reference",
    )
    parser.add_argument(
        "--container-image-config-digest",
        help="immutable OCI image configuration digest (sha256:<64 lowercase hex>)",
    )
    parser.add_argument(
        "--container-base-image-digest",
        help="immutable OCI base-image manifest digest (sha256:<64 lowercase hex>)",
    )
    parser.add_argument(
        "--container-image-inspect-json",
        type=Path,
        help=(
            "absolute path to one sanitized host-generated Docker image "
            "inspect JSON document"
        ),
    )
    parser.add_argument(
        "--container-runtime-inspect-json",
        type=Path,
        help=(
            "absolute path to one sanitized host-generated pre-start Docker "
            "container inspect JSON document"
        ),
    )
    parser.add_argument(
        "--container-image-history-jsonl",
        type=Path,
        help="absolute path to sanitized host-generated Docker image history",
    )
    parser.add_argument(
        "--container-image-build-scan-json",
        type=Path,
        help="absolute path to the host-generated image layer scan record",
    )
    parser.add_argument(
        "--container-source-revision",
        help="source revision recorded in the immutable container image",
    )
    args = parser.parse_args(list(argv))
    common_identity_values = {
        "--container-image-reference": args.container_image_reference,
        "--container-image-config-digest": args.container_image_config_digest,
        "--container-base-image-digest": args.container_base_image_digest,
        "--container-image-inspect-json": args.container_image_inspect_json,
    }
    all_identity_values = {
        **common_identity_values,
        "--container-runtime-inspect-json": (
            args.container_runtime_inspect_json
        ),
        "--container-image-history-jsonl": (
            args.container_image_history_jsonl
        ),
        "--container-image-build-scan-json": (
            args.container_image_build_scan_json
        ),
        "--container-source-revision": args.container_source_revision,
    }
    container_mode = bool(
        args.measurement_container
        or args.observe_measurement_container_system_lock
    )
    if args.measurement_container:
        required = {
            **common_identity_values,
            "--container-runtime-inspect-json": (
                args.container_runtime_inspect_json
            ),
            "--container-source-revision": args.container_source_revision,
            "--container-image-history-jsonl": (
                args.container_image_history_jsonl
            ),
            "--container-image-build-scan-json": (
                args.container_image_build_scan_json
            ),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            parser.error(
                "--measurement-container requires " + ", ".join(missing)
            )
    elif args.observe_measurement_container_system_lock:
        required = {
            **common_identity_values,
            "--container-source-revision": args.container_source_revision,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            parser.error(
                "--observe-measurement-container-system-lock requires "
                + ", ".join(missing)
            )
        forbidden = {
            "--container-runtime-inspect-json": (
                args.container_runtime_inspect_json
            ),
            "--container-image-history-jsonl": (
                args.container_image_history_jsonl
            ),
            "--container-image-build-scan-json": (
                args.container_image_build_scan_json
            ),
        }
        supplied_forbidden = [
            name for name, value in forbidden.items() if value is not None
        ]
        if supplied_forbidden:
            parser.error(
                "runtime/build evidence is forbidden during lock observation: "
                + ", ".join(supplied_forbidden)
            )
    else:
        supplied = [
            name for name, value in all_identity_values.items() if value
        ]
        if supplied:
            parser.error(
                "container identity arguments require a container mode: "
                + ", ".join(supplied)
            )

    if container_mode:
        digest_pattern = re.compile(r"sha256:[0-9a-f]{64}")
        for name in (
            "--container-image-config-digest",
            "--container-base-image-digest",
        ):
            value = all_identity_values[name]
            if (
                not isinstance(value, str)
                or digest_pattern.fullmatch(value) is None
            ):
                parser.error(f"{name} must be sha256:<64 lowercase hex>")
        image_reference = args.container_image_reference
        if (
            not isinstance(image_reference, str)
            or len(image_reference) > 512
            or re.search(r"[\s\x00-\x1f\x7f]", image_reference)
        ):
            parser.error(
                "--container-image-reference must be a non-empty "
                "single-line OCI reference"
            )
        inspect_paths = [
            (
                "--container-image-inspect-json",
                args.container_image_inspect_json,
            )
        ]
        if args.measurement_container:
            inspect_paths.extend(
                (
                    (
                        "--container-runtime-inspect-json",
                        args.container_runtime_inspect_json,
                    ),
                    (
                        "--container-image-history-jsonl",
                        args.container_image_history_jsonl,
                    ),
                    (
                        "--container-image-build-scan-json",
                        args.container_image_build_scan_json,
                    ),
                )
            )
        for name, inspect_path in inspect_paths:
            if (
                not isinstance(inspect_path, Path)
                or not inspect_path.is_absolute()
            ):
                parser.error(f"{name} must be an absolute path")
            if inspect_path.is_relative_to(ROOT):
                parser.error(
                    f"{name} must be outside the repository checkout"
                )
        if (
            not isinstance(args.container_source_revision, str)
            or re.fullmatch(r"[0-9a-f]{40}", args.container_source_revision)
            is None
        ):
            parser.error(
                "--container-source-revision must be 40 lowercase hex"
            )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.observe_measurement_container_system_lock:
        try:
            lock = observe_measurement_container_system_lock(
                image_config_digest=args.container_image_config_digest,
                base_image_digest=args.container_base_image_digest,
                source_revision=args.container_source_revision,
                image_inspect_path=args.container_image_inspect_json,
            )
        except (OSError, RuntimeError, ValueError) as error:
            print(
                "Measurement Container lock observation failed: "
                f"{error}",
                file=sys.stderr,
            )
            return 2
        sys.stdout.buffer.write(json_bytes(lock))
        return 0
    measurement_container_mode = bool(args.measurement_container)
    evidence_root = (
        MEASUREMENT_CONTAINER_EVIDENCE_ROOT
        if measurement_container_mode
        else EVIDENCE_ROOT
    )
    system_lock_path = (
        MEASUREMENT_CONTAINER_SYSTEM_LOCK_PATH
        if measurement_container_mode
        else SYSTEM_LOCK_PATH
    )
    contract_paths = (
        MEASUREMENT_CONTAINER_CONTRACT_PATHS
        if measurement_container_mode
        else CONTRACT_PATHS
    )
    gate_names = (
        MEASUREMENT_CONTAINER_GATE_NAMES
        if measurement_container_mode
        else NATIVE_HOST_GATE_NAMES
    )
    project_python = (
        MEASUREMENT_CONTAINER_PYTHON
        if measurement_container_mode
        else VENV_PYTHON
    )
    cuda_home = CUDA_HOME_DEFAULT.resolve()
    original_environment = dict(os.environ)
    startup_forbidden = sorted(
        key
        for key in (
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
            "PYTHONHOME",
            "PYTHONPATH",
        )
        if original_environment.get(key)
    )
    if sys.flags.optimize != 0 or startup_forbidden:
        print(
            "E00 refused before run-directory creation: contaminated Python startup; keys="
            + repr(startup_forbidden),
            file=sys.stderr,
        )
        return 2
    base_environment: dict[str, str] = {}
    sanitized_keys: list[str] = []
    for key in tuple(original_environment):
        if (
            key.startswith("CUDA_")
            or key.startswith("E00_")
            or key.startswith("GIT_")
            or key.startswith("LD_")
            or key.startswith("NVCC_")
            or key.startswith("PYTORCH_")
            or key.startswith("TORCH_")
            or key
            in {
                "CC",
                "CFLAGS",
                "CPPFLAGS",
                "CPATH",
                "CPLUS_INCLUDE_PATH",
                "CXX",
                "CXXFLAGS",
                "C_INCLUDE_PATH",
                "CUDAHOSTCXX",
                "LIBRARY_PATH",
                "LDFLAGS",
                "MAX_JOBS",
                "PATH",
                "PYTHONHOME",
                "PYTHONPATH",
                "PYTHONOPTIMIZE",
                "PYTORCH_NO_CUDA_MEMORY_CACHING",
                "TORCH_CUDA_ARCH_LIST",
                "TORCH_EXTENSIONS_DIR",
            }
        ):
            sanitized_keys.append(key)
    base_environment.update(
        {
            "LC_ALL": "C",
            "LANG": "C",
            "TZ": "UTC",
            "CUDA_HOME": str(cuda_home),
            "PATH": f"{cuda_home / 'bin'}:/usr/bin:/bin",
            "CC": "/usr/bin/gcc",
            "CXX": "/usr/bin/c++",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONOPTIMIZE": "0",
        }
    )

    git_head_raw = run_memory(
        ["/usr/bin/git", "rev-parse", "HEAD"],
        environment=base_environment,
    )
    git_status_raw = run_memory(
        [
            "/usr/bin/git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        environment=base_environment,
    )
    git_contract_index_raw = run_memory(
        [
            "/usr/bin/git",
            "ls-files",
            "--stage",
            "-z",
            "--",
            *contract_paths,
        ],
        environment=base_environment,
    )
    git_contract_flags_raw = run_memory(
        [
            "/usr/bin/git",
            "ls-files",
            "-v",
            "-z",
            "--",
            *contract_paths,
        ],
        environment=base_environment,
    )
    contract_git_validation = verify_contract_git_state(
        staged_output=git_contract_index_raw["stdout_bytes"],
        flags_output=git_contract_flags_raw["stdout_bytes"],
        paths=contract_paths,
    )
    git_sha = decode(git_head_raw["stdout_bytes"]).strip()
    dirty_text = decode(git_status_raw["stdout_bytes"])
    clean = (
        git_head_raw["exit_code"] == 0
        and git_status_raw["exit_code"] == 0
        and git_contract_index_raw["exit_code"] == 0
        and git_contract_flags_raw["exit_code"] == 0
        and re.fullmatch(r"[0-9a-f]{40}", git_sha) is not None
        and dirty_text == ""
        and contract_git_validation["status"] == "PASS"
    )
    if not clean:
        print(
            "E00 refused before run-directory creation: implementation tree is not clean.",
            file=sys.stderr,
        )
        if dirty_text:
            print(dirty_text, file=sys.stderr, end="")
        return 2

    initial_process_raw = run_memory(
        [str(project_python), str(PROCESS_QUERY_PATH)],
        environment=base_environment,
        timeout_seconds=20.0,
    )

    started_at = utc_now()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_id = f"e00-{timestamp}-{git_sha[:12]}-{secrets.token_hex(4)}"
    evidence_root.mkdir(parents=True, exist_ok=True)
    stage = evidence_root / f".{run_id}.tmp"
    final = evidence_root / run_id
    os.mkdir(stage, 0o700)
    recorder = EvidenceRecorder(
        stage,
        base_environment,
        project_python=project_python,
    )

    try:
        recorder.record_precollected(
            command_id="git_head",
            purpose="Record the committed implementation SHA before evidence creation",
            result=git_head_raw,
            environment=base_environment,
        )
        recorder.record_precollected(
            command_id="git_status_clean",
            purpose="Prove the worktree was clean before evidence-directory creation",
            result=git_status_raw,
            environment=base_environment,
        )
        recorder.record_precollected(
            command_id="git_contract_index",
            purpose="Record committed index object IDs for all executed contract files",
            result=git_contract_index_raw,
            environment=base_environment,
        )
        recorder.record_precollected(
            command_id="git_contract_flags",
            purpose="Reject assume-unchanged or skip-worktree contract files",
            result=git_contract_flags_raw,
            environment=base_environment,
        )
        source_git_relative = "validation/source_git_integrity.json"
        write_exclusive(
            stage / source_git_relative,
            json_bytes(contract_git_validation),
        )
        input_environment_relative = "validation/input_environment.json"
        write_exclusive(
            stage / input_environment_relative,
            json_bytes(
                {
                    "original_allowlisted_environment": {
                        key: original_environment[key]
                        for key in ENVIRONMENT_ALLOWLIST
                        if key in original_environment
                    },
                    "sanitized_keys": sorted(sanitized_keys),
                    "effective_base_environment": relevant_environment(base_environment),
                }
            ),
        )
        container_inspect_relative: str | None = None
        container_identity_validation_relative: str | None = None
        container_identity_validation: dict[str, Any] | None = None
        container_runtime_inspect_relative: str | None = None
        container_runtime_validation_relative: str | None = None
        container_runtime_validation: dict[str, Any] | None = None
        container_image_history_relative: str | None = None
        container_image_build_scan_relative: str | None = None
        container_image_build_validation_relative: str | None = None
        container_image_build_validation: dict[str, Any] | None = None
        if measurement_container_mode:
            container_inspect_relative = (
                "inputs/container_image_inspect.json"
            )
            container_identity_validation_relative = (
                "validation/container_image_identity.json"
            )
            container_runtime_inspect_relative = (
                "inputs/container_runtime_inspect.json"
            )
            container_runtime_validation_relative = (
                "validation/container_runtime_identity.json"
            )
            container_image_history_relative = (
                "inputs/container_image_history.jsonl"
            )
            container_image_build_scan_relative = (
                "inputs/container_image_build_scan.json"
            )
            container_image_build_validation_relative = (
                "validation/container_image_build_record.json"
            )
            inspect_raw, container_identity_validation = (
                validate_container_image_inspect(
                    args.container_image_inspect_json,
                    expected_image_config_digest=(
                        args.container_image_config_digest
                    ),
                    expected_base_image_digest=(
                        args.container_base_image_digest
                    ),
                    expected_revision=args.container_source_revision,
                )
            )
            if inspect_raw is not None:
                write_exclusive(
                    stage / container_inspect_relative,
                    inspect_raw,
                )
                container_identity_validation["copied_evidence"] = output_ref(
                    stage,
                    container_inspect_relative,
                )
            else:
                container_identity_validation["copied_evidence"] = None
            write_exclusive(
                stage / container_identity_validation_relative,
                json_bytes(container_identity_validation),
            )
            runtime_raw, container_runtime_validation = (
                validate_container_runtime_inspect(
                    args.container_runtime_inspect_json,
                    expected_image_config_digest=(
                        args.container_image_config_digest
                    ),
                    expected_hostname=socket.gethostname(),
                )
            )
            if runtime_raw is not None:
                write_exclusive(
                    stage / container_runtime_inspect_relative,
                    runtime_raw,
                )
                container_runtime_validation["copied_evidence"] = output_ref(
                    stage,
                    container_runtime_inspect_relative,
                )
            else:
                container_runtime_validation["copied_evidence"] = None
            write_exclusive(
                stage / container_runtime_validation_relative,
                json_bytes(container_runtime_validation),
            )
            history_raw, build_scan_raw, container_image_build_validation = (
                validate_container_image_build_record(
                    args.container_image_history_jsonl,
                    args.container_image_build_scan_json,
                    expected_image_config_digest=(
                        args.container_image_config_digest
                    ),
                    secret_environment=original_environment,
                )
            )
            if history_raw is not None:
                write_exclusive(
                    stage / container_image_history_relative,
                    history_raw,
                )
                container_image_build_validation["history_evidence"] = (
                    output_ref(stage, container_image_history_relative)
                )
            else:
                container_image_build_validation["history_evidence"] = None
            if build_scan_raw is not None:
                write_exclusive(
                    stage / container_image_build_scan_relative,
                    build_scan_raw,
                )
                container_image_build_validation["scan_evidence"] = (
                    output_ref(stage, container_image_build_scan_relative)
                )
            else:
                container_image_build_validation["scan_evidence"] = None
            write_exclusive(
                stage / container_image_build_validation_relative,
                json_bytes(container_image_build_validation),
            )
        container_identity_ok = (
            not measurement_container_mode
            or (
                container_identity_validation is not None
                and container_identity_validation["status"] == "PASS"
                and container_runtime_validation is not None
                and container_runtime_validation["status"] == "PASS"
                and container_image_build_validation is not None
                and container_image_build_validation["status"] == "PASS"
            )
        )
        if measurement_container_mode and not container_identity_ok:
            raise RuntimeError(
                "Measurement Container identity validation failed before "
                "CUDA execution"
            )
        initial_process_command = recorder.record_precollected(
            command_id="process_initial_before_directory",
            purpose="Check GPU process isolation before evidence-directory creation",
            result=initial_process_raw,
            environment=base_environment,
        )
        initial_snapshot = recorder.record_initial_snapshot(
            command_record=initial_process_command
        )

        if measurement_container_mode:
            system_lock, system_lock_validation = (
                load_measurement_container_system_lock(system_lock_path)
            )
            observed_system_lock, observed_system_lock_validation = (
                load_measurement_container_system_lock(
                    MEASUREMENT_CONTAINER_OBSERVED_SYSTEM_LOCK_PATH
                )
            )
            if system_lock is None or observed_system_lock is None:
                raise RuntimeError(
                    "reviewed Measurement Container system locks are invalid; "
                    f"expected_errors={system_lock_validation['errors']!r}; "
                    "observed_errors="
                    f"{observed_system_lock_validation['errors']!r}"
                )
            if (
                system_lock != observed_system_lock
                or system_lock_path.read_bytes()
                != MEASUREMENT_CONTAINER_OBSERVED_SYSTEM_LOCK_PATH.read_bytes()
            ):
                raise RuntimeError(
                    "reviewed Measurement Container system lock bytes differ"
                )
        else:
            system_lock = json.loads(
                system_lock_path.read_text(encoding="utf-8")
            )
        lock_tools = {
            item["name"]: item for item in system_lock.get("tools", [])
        }

        gpu_query = ",".join(GPU_QUERY_FIELDS)
        recorder.run(
            "nvidia_smi_list",
            "Record NVIDIA GPU list",
            ["/usr/bin/nvidia-smi", "-L"],
        )
        recorder.run(
            "nvidia_smi_query_full",
            "Record full NVIDIA hardware and driver inventory",
            ["/usr/bin/nvidia-smi", "-q"],
        )
        recorder.run(
            "nvidia_smi_table",
            "Record driver-supported CUDA version and current process table",
            ["/usr/bin/nvidia-smi"],
        )
        recorder.run(
            "nvidia_smi_gpu_csv",
            "Record parseable GPU identity, memory, power, clocks, thermals, and firmware",
            [
                "/usr/bin/nvidia-smi",
                f"--query-gpu={gpu_query}",
                "--format=csv,noheader,nounits",
            ],
        )
        recorder.run("host_lscpu", "Record CPU topology", ["/usr/bin/lscpu", "--json"])
        recorder.run("host_meminfo", "Record host memory", ["/usr/bin/cat", "/proc/meminfo"])
        recorder.run("host_os_release", "Record host OS release", ["/usr/bin/cat", "/etc/os-release"])
        recorder.run("host_kernel_release", "Record host kernel release", ["/usr/bin/uname", "-r"])
        recorder.run("host_kernel_version", "Record host kernel version", ["/usr/bin/uname", "-v"])
        recorder.run("host_machine", "Record host machine architecture", ["/usr/bin/uname", "-m"])
        recorder.run("host_hostname", "Record host name", ["/usr/bin/hostname"])
        recorder.run("host_cgroup", "Record PID 1 cgroup for environment classification", ["/usr/bin/cat", "/proc/1/cgroup"])
        recorder.run(
            "host_container_detection",
            (
                "Verify the Docker execution environment"
                if measurement_container_mode
                else "Detect container virtualization without fabricating a digest"
            ),
            ["/usr/bin/systemd-detect-virt", "--container"],
            expected_exit_codes=(0,) if measurement_container_mode else (1,),
        )
        recorder.run("host_self_cgroup", "Record collector cgroup for environment classification", ["/usr/bin/cat", "/proc/self/cgroup"])

        version_commands = {
            "nvcc_version": (
                "Record CUDA compiler version",
                [lock_tools["nvcc"]["invocation_path"], "--version"],
            ),
            "compute_sanitizer_version": (
                "Record Compute Sanitizer version",
                [
                    lock_tools["compute-sanitizer-real"]["invocation_path"],
                    "--version",
                ],
            ),
            "cuobjdump_version": (
                "Record CUDA binary inspection version",
                [lock_tools["cuobjdump"]["invocation_path"], "--version"],
            ),
            "nvdisasm_version": (
                "Record CUDA disassembler version",
                [lock_tools["nvdisasm"]["invocation_path"], "--version"],
            ),
            "ncu_version": (
                "Record Nsight Compute version",
                [lock_tools["ncu"]["invocation_path"], "--version"],
            ),
            "nsys_version": (
                "Record Nsight Systems version",
                [lock_tools["nsys"]["invocation_path"], "--version"],
            ),
            "cxx_version": (
                "Record host C++ compiler version",
                [lock_tools["c++"]["invocation_path"], "--version"],
            ),
            "ninja_version": (
                "Record Ninja version",
                [lock_tools["ninja"]["invocation_path"], "--version"],
            ),
            "python_version": (
                "Record project Python version",
                [str(project_python), "--version"],
            ),
        }
        version_specs = {
            "nvcc_version": ("nvcc", "nvcc"),
            "compute_sanitizer_version": (
                "compute-sanitizer-real",
                "compute-sanitizer",
            ),
            "cuobjdump_version": ("cuobjdump", "cuobjdump"),
            "nvdisasm_version": ("nvdisasm", "nvdisasm"),
            "ncu_version": ("ncu", "ncu"),
            "nsys_version": ("nsys", "nsys"),
            "cxx_version": ("c++", "c++"),
            "ninja_version": ("ninja", "ninja"),
            "python_version": ("python", "python"),
        }
        for command_id, (purpose, command) in version_commands.items():
            recorder.run(command_id, purpose, command)

        dpkg_names = [item["name"] for item in system_lock["dpkg_packages"]]
        dpkg_format = chr(36) + "{Package}\t" + chr(36) + "{Version}\t" + chr(36) + "{Architecture}\n"
        recorder.run(
            "dpkg_locked_versions",
            "Verify exact system package versions",
            ["/usr/bin/dpkg-query", "-W", f"-f={dpkg_format}", *dpkg_names],
        )
        recorder.run(
            "dpkg_architecture",
            "Verify the native dpkg architecture",
            [lock_tools["dpkg"]["invocation_path"], "--print-architecture"],
        )
        recorder.run(
            "nvdisasm_package_ownership",
            "Verify nvdisasm package ownership",
            [
                lock_tools["dpkg-query"]["invocation_path"],
                "-S",
                lock_tools["nvdisasm"]["resolved_path"],
            ],
        )
        recorder.run(
            "dpkg_verify_locked_files",
            "Verify package-managed toolchain and transitive executable files",
            [
                lock_tools["dpkg"]["invocation_path"],
                "--verify-format=rpm",
                "--verify",
                *dpkg_names,
            ],
        )

        recorder.run(
            "pip_check",
            "Verify installed Python dependency consistency",
            [str(project_python), "-m", "pip", "check"],
        )
        recorder.run(
            "pip_list",
            "Record installed Python dependency versions",
            [str(project_python), "-m", "pip", "list", "--format=json"],
        )
        container_inventory_ids: tuple[str, ...] = ()
        if measurement_container_mode:
            recorder.run(
                "container_dpkg_inventory",
                "Record the complete installed container package inventory",
                ["/usr/bin/dpkg-query", "-W", f"-f={dpkg_format}"],
            )
            recorder.run(
                "container_python_freeze",
                "Record the complete Measurement Container Python freeze",
                [str(project_python), "-m", "pip", "freeze", "--all"],
            )
            recorder.run(
                "container_phase3_python_freeze",
                "Record the frozen Phase 3 additive Python target",
                [
                    str(project_python),
                    "-m",
                    "pip",
                    "freeze",
                    "--path",
                    "/opt/kvbench/.phase3/site-packages",
                ],
            )
            recorder.run(
                "container_phase3_pip_check",
                "Verify Phase 3 target dependencies against the frozen base",
                [str(project_python), "-m", "pip", "check"],
                environment=environment_with(
                    base_environment,
                    {
                        "PYTHONPATH": (
                            "/opt/kvbench/.phase3/site-packages"
                        )
                    },
                ),
            )
            container_inventory_ids = (
                "container_dpkg_inventory",
                "container_python_freeze",
                "container_phase3_python_freeze",
                "container_phase3_pip_check",
            )
        recorder.run(
            "python_record_integrity",
            "Verify installed wheel RECORD hashes and file ownership",
            [str(project_python), str(PYTHON_INTEGRITY_PATH)],
            timeout_seconds=300.0,
        )

        gpu_rows: list[dict[str, str]] = []
        gpu_parse_error: str | None = None
        try:
            gpu_rows = parse_gpu_rows(recorder.command_text("nvidia_smi_gpu_csv").split("\n\n", 1)[0])
        except Exception as error:
            gpu_parse_error = str(error)
        matches = [
            row for row in gpu_rows if row.get("name") == EXPECTED_GPU_NAME
        ]
        selected = matches[0] if len(matches) == 1 else (gpu_rows[0] if len(gpu_rows) == 1 else None)
        selected_full_name = nonempty_string(selected.get("name") if selected else None)
        selected_uuid = matching_string(selected.get("uuid") if selected else None, r"GPU-[0-9A-Fa-f-]+")
        selected_pci_device_id = matching_string(selected.get("pci.device_id") if selected else None, r"0x[0-9A-Fa-f]{8}")
        selected_pci_bus_id = matching_string(selected.get("pci.bus_id") if selected else None, r"(?:[0-9A-Fa-f]{8}:)?[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]")
        selected_vbios_version = nonempty_string(selected.get("vbios_version") if selected else None)
        selected_capability = capability(selected.get("compute_cap") if selected else None)

        selection_collection_ok = (
            selected is not None
            and gpu_parse_error is None
            and selected_uuid is not None
        )
        selection_ok = (
            len(matches) == 1
            and selection_collection_ok
            and (
                not measurement_container_mode
                or selected_uuid == EXPECTED_GPU_UUID
            )
        )
        gpu_selection = {
            "collection_status": "PASS" if selection_collection_ok else "FAIL",
            "collection_error": None if selection_collection_ok else (gpu_parse_error or "no valid GPU UUID inventory row"),
            "selection_method": "gpu_uuid",
            "cuda_visible_devices": selected_uuid,
            "visible_device_count": 1 if selection_collection_ok else None,
            "selected_logical_index": 0 if selection_collection_ok else None,
            "selected_gpu_uuid": selected_uuid,
            "selection_is_unique": selection_ok if selection_collection_ok else None,
        }

        gpu_power = {
            "draw_w": nullable_nonnegative_float(selected.get("power.draw") if selected else None),
            "limit_w": nullable_nonnegative_float(selected.get("power.limit") if selected else None),
            "default_limit_w": nullable_nonnegative_float(selected.get("power.default_limit") if selected else None),
            "min_limit_w": nullable_nonnegative_float(selected.get("power.min_limit") if selected else None),
            "max_limit_w": nullable_nonnegative_float(selected.get("power.max_limit") if selected else None),
        }
        gpu_clocks = {
            "sm_current_mhz": nullable_nonnegative_float(selected.get("clocks.current.sm") if selected else None),
            "memory_current_mhz": nullable_nonnegative_float(selected.get("clocks.current.memory") if selected else None),
            "sm_max_mhz": nullable_nonnegative_float(selected.get("clocks.max.sm") if selected else None),
            "memory_max_mhz": nullable_nonnegative_float(selected.get("clocks.max.memory") if selected else None),
        }
        gpu_ecc_mode = normalized_mode(selected.get("ecc.mode.current") if selected else None, not_supported=True)
        gpu_persistence_mode = normalized_mode(selected.get("persistence_mode") if selected else None)
        gpu_display_mode = normalized_mode(selected.get("display_active") if selected else None)
        gpu_performance_state = matching_string(selected.get("pstate") if selected else None, r"P[0-9]{1,2}")
        gpu_temperature_c = nullable_float(selected.get("temperature.gpu") if selected else None)
        if gpu_temperature_c is not None and not -100 <= gpu_temperature_c <= 200:
            gpu_temperature_c = None
        vram_mib = nullable_int(selected.get("memory.total") if selected else None)

        gpu_required_values = [
            selected_full_name,
            selected_uuid,
            selected_pci_bus_id,
            selected_pci_device_id,
            selected_capability,
            vram_mib,
            gpu_ecc_mode,
            gpu_persistence_mode,
            gpu_display_mode,
            gpu_performance_state,
            *gpu_power.values(),
            *gpu_clocks.values(),
            gpu_temperature_c,
            selected_vbios_version,
        ]
        gpu_values_valid = (
            vram_mib is not None and vram_mib > 0
            and gpu_ecc_mode in {"enabled", "disabled", "not_supported"}
            and gpu_persistence_mode in {"enabled", "disabled"}
            and gpu_display_mode in {"enabled", "disabled"}
            and gpu_power["draw_w"] is not None and gpu_power["draw_w"] >= 0
            and all(value is not None and value > 0 for name, value in gpu_power.items() if name != "draw_w")
            and all(value is not None and value > 0 for value in gpu_clocks.values())
        )
        gpu_collection_ok = (
            all(
                recorder.command_ok(command_id)
                for command_id in (
                    "nvidia_smi_list",
                    "nvidia_smi_query_full",
                    "nvidia_smi_table",
                    "nvidia_smi_gpu_csv",
                )
            )
            and selected is not None
            and gpu_parse_error is None
            and all(value is not None for value in gpu_required_values)
            and gpu_values_valid
        )
        gpu_manifest = {
            "collection_status": "PASS" if gpu_collection_ok else "FAIL",
            "collection_error": None if gpu_collection_ok else (gpu_parse_error or "required GPU identity field missing"),
            "captured_at_utc": recorder.command_index["nvidia_smi_gpu_csv"]["finished_at_utc"],
            "full_name": selected_full_name,
            "uuid": selected_uuid,
            "pci_device_id": selected_pci_device_id,
            "pci_bus_id": selected_pci_bus_id,
            "compute_capability": selected_capability,
            "vram_total_bytes": vram_mib * 1024 * 1024 if vram_mib is not None else None,
            "ecc_mode": gpu_ecc_mode,
            "persistence_mode": gpu_persistence_mode,
            "display_mode": gpu_display_mode,
            "performance_state": gpu_performance_state,
            "power": gpu_power,
            "clocks": gpu_clocks,
            "temperature_c": gpu_temperature_c,
            "vbios_version": selected_vbios_version,
        }

        host_error: str | None = None
        try:
            lscpu = parse_lscpu(recorder.command_text("host_lscpu").split("\n\n", 1)[0])
            os_release = parse_key_value_file(
                recorder.command_text("host_os_release").split("\n\n", 1)[0]
            )
            cores_per_socket = int(lscpu["Core(s) per socket"])
            sockets = int(lscpu["Socket(s)"])
            physical_cores = cores_per_socket * sockets
            logical_cores = int(lscpu["CPU(s)"])
            memory_total = parse_memtotal_bytes(
                recorder.command_text("host_meminfo").split("\n\n", 1)[0]
            )
            cpu_model = lscpu["Model name"]
        except Exception as error:
            host_error = str(error)
            os_release = {}
            physical_cores = None
            logical_cores = None
            memory_total = None
            cpu_model = None
        host_command_ids = (
            "host_lscpu",
            "host_meminfo",
            "host_os_release",
            "host_kernel_release",
            "host_kernel_version",
            "host_machine",
            "host_hostname",
        )
        host_hostname = nonempty_string(recorder.command_stdout("host_hostname"))
        host_os_name = nonempty_string(os_release.get("NAME"))
        host_os_version = nonempty_string(os_release.get("VERSION_ID"))
        host_kernel_release = nonempty_string(recorder.command_stdout("host_kernel_release"))
        host_kernel_version = nonempty_string(recorder.command_stdout("host_kernel_version"))
        host_machine_arch = nonempty_string(recorder.command_stdout("host_machine"))
        host_cpu_model = nonempty_string(cpu_model)

        host_ok = (
            host_error is None
            and all(recorder.command_ok(item) for item in host_command_ids)
            and all(
                value is not None
                for value in (
                    host_hostname,
                    host_os_name,
                    host_os_version,
                    host_kernel_release,
                    host_kernel_version,
                    host_machine_arch,
                    host_cpu_model,
                )
            )
            and all(
                isinstance(value, int) and value > 0
                for value in (physical_cores, logical_cores, memory_total)
            )
        )
        container_marker_paths = (
            "/.dockerenv",
            "/run/.containerenv",
            "/run/systemd/container",
        )
        present_container_markers = [
            path for path in container_marker_paths if os.path.lexists(path)
        ]
        cgroup_text = recorder.command_stdout("host_cgroup") + recorder.command_stdout("host_self_cgroup")
        cgroup_container_tokens = sorted(
            set(re.findall(r"(?:docker|kubepods|containerd|libpod|podman|lxc)", cgroup_text, re.IGNORECASE))
        )
        container_detection_output = recorder.command_stdout(
            "host_container_detection"
        ).strip()
        caller_container_marker = bool(
            original_environment.get("container")
            or original_environment.get("CONTAINER")
        )
        cgroup_evidence = {
            command_id: (
                recorder.command_stdout(command_id)
                if recorder.command_ok(command_id)
                else None
            )
            for command_id in ("host_cgroup", "host_self_cgroup")
        }
        if measurement_container_mode:
            environment_errors = measurement_container_validation_errors(
                container_detection_ok=recorder.command_ok(
                    "host_container_detection"
                ),
                container_detection_output=container_detection_output,
                cgroup_evidence=cgroup_evidence,
                present_container_markers=present_container_markers,
                image_identity_ok=container_identity_ok,
            )
            environment_relative = (
                "validation/measurement_container_detection.json"
            )
        else:
            environment_errors = native_host_validation_errors(
                container_detection_ok=recorder.command_ok(
                    "host_container_detection"
                ),
                container_detection_output=container_detection_output,
                cgroup_evidence=cgroup_evidence,
                present_container_markers=present_container_markers,
                cgroup_container_tokens=cgroup_container_tokens,
                caller_container_marker=caller_container_marker,
            )
            environment_relative = "validation/native_host_detection.json"
        environment_ok = not environment_errors

        host_manifest = {
            "collection_status": "PASS" if host_ok else "FAIL",
            "collection_error": None if host_ok else (host_error or "host inventory command failed"),
            "hostname": host_hostname,
            "os_name": host_os_name,
            "os_version": host_os_version,
            "kernel_release": host_kernel_release,
            "kernel_version": host_kernel_version,
            "machine_arch": host_machine_arch,
            "cpu_model": host_cpu_model,
            "physical_core_count": physical_cores,
            "logical_cpu_count": logical_cores,
            "memory_total_bytes": memory_total,
        }

        write_exclusive(
            stage / environment_relative,
            json_bytes(
                {
                    **(
                        {
                            "mode": "measurement_container",
                            "container_image_identity": {
                                "status": container_identity_validation[
                                    "status"
                                ],
                                "validation_evidence_file_id": artifact_id(
                                    container_identity_validation_relative
                                ),
                                "inspect_evidence_file_id": (
                                    artifact_id(container_inspect_relative)
                                    if container_identity_ok
                                    else None
                                ),
                            },
                            "container_runtime_identity": {
                                "status": container_runtime_validation[
                                    "status"
                                ],
                                "validation_evidence_file_id": artifact_id(
                                    container_runtime_validation_relative
                                ),
                                "inspect_evidence_file_id": (
                                    artifact_id(
                                        container_runtime_inspect_relative
                                    )
                                    if container_identity_ok
                                    else None
                                ),
                            },
                        }
                        if measurement_container_mode
                        else {}
                    ),
                    "status": "PASS" if environment_ok else "FAIL",
                    "errors": environment_errors,
                    "systemd_detect_virt": {
                        "exit_code": recorder.command_index["host_container_detection"]["exit_code"],
                        "stdout": container_detection_output,
                    },
                    "present_container_markers": present_container_markers,
                    "cgroup_container_tokens": cgroup_container_tokens,
                    "caller_container_marker_present": caller_container_marker,
                }
            ),
        )

        pip_payload = recorder.command_payload("pip_list")
        if pip_payload is None:
            try:
                pip_payload = json.loads(
                    recorder.command_text("pip_list").split("\n\n", 1)[0]
                )
            except json.JSONDecodeError:
                pip_payload = None
        container_dependency_inputs: dict[str, Any] = {}
        if measurement_container_mode:
            container_dependency_inputs = {
                "full_dpkg_text": recorder.command_stdout(
                    "container_dpkg_inventory"
                ),
                "phase3_freeze_text": recorder.command_stdout(
                    "container_phase3_python_freeze"
                ),
                "phase3_dependency_check_ok": recorder.command_ok(
                    "container_phase3_pip_check"
                ),
            }
        dependency_validation = verify_dependency_locks(
            system_lock,
            recorder.command_text("dpkg_locked_versions").split("\n\n", 1)[0],
            pip_payload,
            **container_dependency_inputs,
        )
        dependency_errors = dependency_validation["errors"]
        nvdisasm_owner_ok = dpkg_ownership_matches(
            command_ok=recorder.command_ok("nvdisasm_package_ownership"),
            stdout=recorder.command_stdout("nvdisasm_package_ownership"),
            stderr=recorder.command_stderr("nvdisasm_package_ownership"),
            expected_package=lock_tools["nvdisasm"]["dpkg_package"],
            expected_path=lock_tools["nvdisasm"]["resolved_path"],
        )
        dependency_validation["nvdisasm_package_ownership"] = {
            "status": "PASS" if nvdisasm_owner_ok else "FAIL",
            "command_id": "nvdisasm_package_ownership",
            "expected_package": lock_tools["nvdisasm"]["dpkg_package"],
            "expected_path": lock_tools["nvdisasm"]["resolved_path"],
        }
        if not nvdisasm_owner_ok:
            dependency_errors.append(
                "nvdisasm package ownership did not exactly match the lock"
            )
        tool_integrity_ok = {
            observation["name"]: (
                observation["path_matches"] and observation["hash_matches"]
            )
            for observation in dependency_validation["tool_hash_observations"]
        }

        try:
            venv_python_resolved = project_python.resolve(strict=True)
        except OSError as error:
            venv_python_resolved = None
            dependency_errors.append(f"project Python cannot be resolved: {error}")
        expected_python = Path(lock_tools["python"]["resolved_path"])
        if venv_python_resolved != expected_python:
            dependency_errors.append(
                f"project Python target mismatch: "
                f"{venv_python_resolved} != {expected_python}"
            )
        python_executable_present = (
            tool_integrity_ok.get("python", False)
            and venv_python_resolved == expected_python
        )
        locked_cuda_home = Path(system_lock["environment"]["cuda_home"]).resolve()
        if cuda_home != locked_cuda_home:
            dependency_errors.append(
                f"CUDA root mismatch: {cuda_home} != {locked_cuda_home}"
            )
        nvcc_from_root = (cuda_home / "bin" / "nvcc").resolve()
        if nvcc_from_root != Path(lock_tools["nvcc"]["resolved_path"]):
            dependency_errors.append(
                f"CUDA-root nvcc mismatch: {nvcc_from_root}"
            )

        observed_platform = {
            "distribution": str(os_release.get("ID", "")).lower(),
            "distribution_version": os_release.get("VERSION_ID"),
            "dpkg_architecture": (
                recorder.command_stdout("dpkg_architecture").strip()
                if recorder.command_ok("dpkg_architecture")
                else None
            ),
            "machine": (
                recorder.command_stdout("host_machine").strip()
                if recorder.command_ok("host_machine")
                else None
            ),
            "python_abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
            "execution_environment_kind": (
                (
                    "measurement_container"
                    if measurement_container_mode
                    else "native_host"
                )
                if environment_ok
                else "unverified"
            ),
            "container_digest_status": (
                (
                    "verified_against_sanitized_image_inspect"
                    if container_identity_ok
                    else "image_inspect_verification_failed"
                )
                if measurement_container_mode
                else (
                    "not_applicable_native_host"
                    if environment_ok
                    else "unavailable_unverified_environment"
                )
            ),
            "performance_claim_eligible": False,
            "cuda_home": str(cuda_home),
            "python_environment": (
                project_python.parents[1].as_posix()
                if measurement_container_mode
                else project_python.parents[1].relative_to(ROOT).as_posix()
            ),
            "python_requirements_lock": PYTHON_LOCK_PATH.relative_to(
                ROOT
            ).as_posix(),
        }
        platform_validation = verify_platform_lock(
            system_lock, observed_platform
        )
        dependency_errors.extend(platform_validation["errors"])
        dependency_validation["platform_validation"] = platform_validation
        dependency_validation["requirements_lock_sha256"] = sha256_file(
            PYTHON_LOCK_PATH
        )
        if measurement_container_mode:
            dependency_validation["system_lock_path"] = (
                system_lock_path.relative_to(ROOT).as_posix()
            )
            dependency_validation["system_lock_sha256"] = sha256_file(
                system_lock_path
            )
            dependency_validation["observed_system_lock_path"] = (
                MEASUREMENT_CONTAINER_OBSERVED_SYSTEM_LOCK_PATH.relative_to(
                    ROOT
                ).as_posix()
            )
            dependency_validation["observed_system_lock_sha256"] = sha256_file(
                MEASUREMENT_CONTAINER_OBSERVED_SYSTEM_LOCK_PATH
            )
            dependency_validation["reviewed_system_lock_bytes_equal"] = True

        dpkg_verify = dpkg_verification_result(
            command_ok=recorder.command_ok("dpkg_verify_locked_files"),
            stdout=recorder.command_stdout("dpkg_verify_locked_files"),
            stderr=recorder.command_stderr("dpkg_verify_locked_files"),
            allow_container_policy_omissions=measurement_container_mode,
        )
        dpkg_verify_ok = dpkg_verify["status"] == "PASS"
        dependency_validation["dpkg_verify"] = dpkg_verify
        dependency_validation["dpkg_verify_clean"] = dpkg_verify_ok
        dependency_validation["dpkg_verify_byte_empty"] = dpkg_verify[
            "byte_empty"
        ]
        if not dpkg_verify_ok:
            dependency_errors.append(
                "dpkg --verify reported an unexpected mismatch or command failure"
            )

        python_integrity = recorder.command_payload(
            "python_record_integrity"
        )
        python_integrity_ok = (
            recorder.command_ok("python_record_integrity")
            and python_integrity is not None
            and python_integrity.get("status") == "PASS"
            and not python_integrity.get("errors")
        )
        dependency_validation["python_record_integrity"] = (
            python_integrity
            if python_integrity is not None
            else {"status": "FAIL", "errors": ["probe returned no JSON"]}
        )
        if not python_integrity_ok:
            dependency_errors.append(
                "installed Python RECORD/file-integrity verification failed"
            )

        table_text = recorder.command_text("nvidia_smi_table")
        driver_cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", table_text)
        driver_version = nonempty_string(selected.get("driver_version") if selected else None)
        driver_cuda_version = nonempty_string(
            driver_cuda_match.group(1) if driver_cuda_match else None
        )

        def command_version(command_id: str, tool: str) -> str | None:
            if not recorder.command_ok(command_id):
                return None
            return extract_version(tool, recorder.command_text(command_id))

        tool_version_observations: list[dict[str, Any]] = []
        for command_id, (tool_name, parser_name) in version_specs.items():
            observed_version = command_version(command_id, parser_name)
            expected_version = lock_tools[tool_name].get("reported_version")
            matches = (
                isinstance(expected_version, str)
                and observed_version == expected_version
            )
            tool_version_observations.append(
                {
                    "command_id": command_id,
                    "tool_name": tool_name,
                    "expected_version": expected_version,
                    "observed_version": observed_version,
                    "matches": matches,
                }
            )
            if not matches:
                dependency_errors.append(
                    f"reported tool version mismatch {tool_name}: "
                    f"{observed_version!r} != {expected_version!r}"
                )
        dependency_validation["tool_version_observations"] = (
            tool_version_observations
        )
        dependency_validation["status"] = (
            "PASS" if not dependency_errors else "FAIL"
        )

        dependency_relative = "validation/dependency_locks.json"
        write_exclusive(
            stage / dependency_relative, json_bytes(dependency_validation)
        )

        version_ids = tuple(version_commands)
        required_tool_commands_ok = (
            all(recorder.command_ok(item) for item in version_ids)
            and recorder.command_ok("pip_check")
            and recorder.command_ok("pip_list")
            and all(
                recorder.command_ok(command_id)
                for command_id in container_inventory_ids
            )
            and recorder.command_ok("python_record_integrity")
            and recorder.command_ok("dpkg_locked_versions")
            and recorder.command_ok("dpkg_architecture")
            and recorder.command_ok("nvdisasm_package_ownership")
            and nvdisasm_owner_ok
            and dpkg_verify_ok
            and dependency_validation["status"] == "PASS"
        )

        initial_foreign = initial_snapshot["foreign_compute_processes"]
        initial_unknown = initial_snapshot["unknown_processes"]
        initial_isolation_ok = (
            initial_snapshot["query_exit_code"] == 0
            and not initial_foreign
            and not initial_unknown
        )

        torch_payload: dict[str, Any] | None = None
        torch_command_ran = False
        selected_environment: dict[str, str] | None = None
        if selected_uuid and initial_isolation_ok and environment_ok:
            selected_environment = environment_with(
                base_environment,
                {
                    "CUDA_VISIBLE_DEVICES": selected_uuid,
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                },
            )
            recorder.run_supervised(
                "torch_probe",
                "Verify PyTorch CUDA availability, selected GPU identity, and capability",
                [str(project_python), str(PYTHON_PROBE_PATH)],
                environment=selected_environment,
                timeout_seconds=120.0,
            )
            torch_command_ran = True
            torch_payload = recorder.command_payload("torch_probe")

        torch_payload_parsed = (
            torch_command_ran
            and isinstance(torch_payload, dict)
        )
        torch_command_ok = torch_payload_parsed and recorder.command_ok("torch_probe")
        parsed_torch_payload = torch_payload if torch_payload_parsed else {}
        python_payload = parsed_torch_payload.get("python")
        if not isinstance(python_payload, dict):
            python_payload = {}
        cuda_available = boolean_value(parsed_torch_payload.get("cuda_is_available"))
        torch_device_count = nonnegative_int_value(parsed_torch_payload.get("device_count"))
        torch_selected_index = nonnegative_int_value(parsed_torch_payload.get("current_device"))
        torch_device_name = nonempty_string(parsed_torch_payload.get("device_name"))
        torch_uuid = matching_string(
            parsed_torch_payload.get("device_uuid"), r"GPU-[0-9A-Fa-f-]+"
        )
        torch_capability = capability_from_mapping(
            parsed_torch_payload.get("compute_capability")
        )
        torch_total_memory_bytes = nonnegative_int_value(
            parsed_torch_payload.get("total_memory_bytes")
        )
        torch_python_executable = nonempty_string(python_payload.get("executable"))
        if (
            torch_python_executable is not None
            and not Path(torch_python_executable).is_absolute()
        ):
            torch_python_executable = None
        torch_optimize_level = nonnegative_int_value(python_payload.get("optimize_level"))
        try:
            python_executable_match = (
                torch_python_executable is not None
                and Path(torch_python_executable).resolve(strict=True)
                == Path(lock_tools["python"]["resolved_path"])
            )
        except (OSError, TypeError):
            python_executable_match = False
        uuid_match = (
            torch_uuid == selected_uuid
            if torch_uuid is not None and selected_uuid is not None
            else False
        )
        capability_match = (
            torch_capability == selected_capability
            if torch_capability is not None and selected_capability is not None
            else False
        )
        torch_collection_ok = (
            torch_payload_parsed
            and torch_command_ok
            and cuda_available is not None
            and torch_device_count is not None
            and torch_python_executable is not None
            and python_executable_match
            and torch_optimize_level == 0
            and (
                not cuda_available
                or (
                    torch_device_count >= 1
                    and torch_selected_index is not None
                    and torch_device_name is not None
                    and torch_uuid is not None
                    and torch_capability is not None
                    and torch_total_memory_bytes is not None
                    and torch_total_memory_bytes > 0
                )
            )
        )
        torch_probe_manifest = {
            "collection_status": "PASS" if torch_collection_ok else "FAIL",
            "collection_error": None if torch_collection_ok else ("probe was not run due to process isolation or GPU selection" if not torch_command_ran else "PyTorch probe failed validation"),
            "cuda_available": cuda_available,
            "device_count": torch_device_count,
            "selected_logical_index": torch_selected_index,
            "selected_device_name": torch_device_name,
            "python_executable": torch_python_executable,
            "python_executable_matches_lock": python_executable_match if torch_payload_parsed else None,
            "python_optimization_level": torch_optimize_level,
            "selected_device_uuid": torch_uuid,
            "selected_device_capability": torch_capability,
            "selected_device_total_memory_bytes": torch_total_memory_bytes,
            "selected_uuid_matches_manifest": uuid_match if torch_payload_parsed else None,
            "capability_matches_manifest": capability_match if torch_payload_parsed else None,
        }

        def package_observation(
            presence_observed: bool,
            version: str | None,
            error: str,
        ) -> dict[str, Any]:
            normalized_version = nonempty_string(version)
            installed = presence_observed
            return {
                "installed": installed,
                "version": normalized_version if installed else None,
                "observation_error": (
                    None if installed and normalized_version is not None else error
                ),
            }

        def complete_package_observation(observation: dict[str, Any]) -> bool:
            return (
                observation["installed"] is True
                and observation["version"] is not None
                and observation["observation_error"] is None
            )

        payload_package_validity: dict[str, bool] = {}
        def payload_package_observation(package_name: str) -> dict[str, Any]:
            package_payload = parsed_torch_payload.get(package_name)
            if not isinstance(package_payload, dict):
                payload_package_validity[package_name] = False
                return package_observation(
                    False, None, f"{package_name} observation unavailable"
                )
            reported_installed = boolean_value(package_payload.get("installed"))
            reported_version = package_payload.get("version")
            version = nonempty_string(reported_version)
            payload_package_validity[package_name] = (
                torch_payload_parsed
                and reported_installed is not None
                and ((reported_installed and version is not None)
                     or (not reported_installed and reported_version is None))
            )
            installed = torch_payload_parsed and reported_installed is True
            if installed:
                return {
                    "installed": True,
                    "version": version,
                    "observation_error": (
                        None if version is not None else f"{package_name} version unavailable"
                    ),
                }
            error = (
                f"{package_name} not installed"
                if torch_payload_parsed and reported_installed is False
                else f"{package_name} observation invalid or unavailable"
            )
            return package_observation(False, None, error)

        torch_details = parsed_torch_payload.get("torch")
        if not isinstance(torch_details, dict):
            torch_details = {}
        torch_version = nonempty_string(torch_details.get("version"))
        torch_compiled_cuda_version = nonempty_string(torch_details.get("built_cuda"))
        torch_installed = (
            torch_payload_parsed
            and boolean_value(torch_details.get("installed")) is True
        )
        torch_observation = {
            "installed": torch_installed,
            "version": torch_version if torch_installed else None,
            "compiled_cuda_version": (
                torch_compiled_cuda_version if torch_installed else None
            ),
            "observation_error": (
                None
                if torch_installed
                and torch_version is not None
                and torch_compiled_cuda_version is not None
                else "PyTorch observation invalid or unavailable"
            ),
        }
        nvcc_version = command_version("nvcc_version", "nvcc")
        cuda_home_value = nonempty_string(str(cuda_home))
        nvcc_path_value = nonempty_string(lock_tools["nvcc"].get("invocation_path"))
        cuda_toolkit_installed = tool_integrity_ok.get("nvcc", False)
        ncu_observation = package_observation(
            tool_integrity_ok.get("ncu", False),
            command_version("ncu_version", "ncu"),
            "ncu version command failed or returned unparsable output",
        )
        nsys_observation = package_observation(
            tool_integrity_ok.get("nsys", False),
            command_version("nsys_version", "nsys"),
            "nsys version command failed or returned unparsable output",
        )
        sanitizer_observation = package_observation(
            tool_integrity_ok.get("compute-sanitizer-real", False),
            command_version("compute_sanitizer_version", "compute-sanitizer"),
            "Compute Sanitizer version command failed or returned unparsable output",
        )
        python_observation = package_observation(
            python_executable_present,
            command_version("python_version", "python"),
            "Python version command failed or returned unparsable output",
        )
        triton_observation = payload_package_observation("triton")
        vllm_observation = payload_package_observation("vllm")
        required_package_observations = (
            ncu_observation,
            nsys_observation,
            sanitizer_observation,
            python_observation,
            triton_observation,
        )
        vllm_observation_complete = (
            payload_package_validity.get("vllm", False)
            and (not vllm_observation["installed"]
                 or complete_package_observation(vllm_observation))
        )
        software_collection_ok = (
            required_tool_commands_ok
            and torch_collection_ok
            and driver_version is not None
            and driver_cuda_version is not None
            and cuda_toolkit_installed
            and nvcc_version is not None
            and cuda_home_value is not None
            and nvcc_path_value is not None
            and all(
                complete_package_observation(observation)
                for observation in required_package_observations
            )
            and torch_observation["installed"]
            and torch_observation["version"] is not None
            and torch_observation["compiled_cuda_version"] is not None
            and torch_observation["observation_error"] is None
            and vllm_observation_complete
        )
        software_manifest = {
            "collection_status": "PASS" if software_collection_ok else "FAIL",
            "collection_error": None if software_collection_ok else "one or more locked toolchain observations failed validation",
            "nvidia_driver": {
                "version": driver_version,
                "driver_supported_cuda_version": driver_cuda_version,
            },
            "cuda_toolkit": {
                "installed": cuda_toolkit_installed,
                "version": nvcc_version if cuda_toolkit_installed else None,
                "cuda_home": cuda_home_value if cuda_toolkit_installed else None,
                "nvcc_path": nvcc_path_value if cuda_toolkit_installed else None,
                "observation_error": (
                    None
                    if cuda_toolkit_installed
                    and nvcc_version is not None
                    and cuda_home_value is not None
                    and nvcc_path_value is not None
                    else "nvcc observation invalid or unavailable"
                ),
            },
            "ncu": ncu_observation,
            "nsys": nsys_observation,
            "compute_sanitizer": sanitizer_observation,
            "python": python_observation,
            "torch": torch_observation,
            "triton": triton_observation,
            "vllm": vllm_observation,
        }

        extension_sources = (
            "preflight/e00_cuda/binding.cpp",
            "preflight/e00_cuda/xor_kernel.cu",
            "preflight/e00_cuda/xor_kernel.h",
        )
        source_records = source_artifacts(extension_sources)
        forbidden_patterns = {
            "torch_cat": r"torch\s*::?\s*cat|torch\.cat",
            "cuda_malloc": r"cudaMalloc(?:Async)?\s*\(",
            "cuda_sync": r"cudaDeviceSynchronize\s*\(",
            "tensor_to_list": r"\.tolist\s*\(",
        }
        forbidden_hits: dict[str, list[str]] = {}
        for relative in extension_sources:
            text = (ROOT / relative).read_text(encoding="utf-8")
            for name, pattern in forbidden_patterns.items():
                if re.search(pattern, text):
                    forbidden_hits.setdefault(name, []).append(relative)
        source_scan = {
            "status": "PASS" if not forbidden_hits else "FAIL",
            "forbidden_hits": forbidden_hits,
            "sources": source_records,
        }
        source_scan_relative = "validation/extension_source_scan.json"
        write_exclusive(stage / source_scan_relative, json_bytes(source_scan))

        build_metadata_relative = "extension/build_metadata.json"
        build_payload: dict[str, Any] | None = None
        binary_path: Path | None = None
        build_command_ran = False
        build_preconditions = (
            environment_ok
            and initial_isolation_ok
            and selection_ok
            and gpu_collection_ok
            and required_tool_commands_ok
            and torch_collection_ok
            and cuda_available is True
            and capability_match
            and uuid_match
            and source_scan["status"] == "PASS"
            and selected_environment is not None
        )
        if build_preconditions:
            arch_text = selected_capability["text"]
            build_directory = stage / "extension" / "build"
            build_environment = environment_with(
                selected_environment,
                {
                    "E00_DETECTED_ARCH": arch_text,
                    "TORCH_EXTENSIONS_DIR": str(build_directory),
                    "MAX_JOBS": "4",
                    "CXX": lock_tools["c++"]["invocation_path"],
                },
                unset=("TORCH_CUDA_ARCH_LIST",),
            )
            recorder.run(
                "extension_build",
                "Compile the certification extension with exact native SASS and PTX targets",
                [
                    str(project_python),
                    str(ROOT / "preflight" / "e00_cuda" / "build.py"),
                    "--build-directory",
                    str(build_directory),
                    "--json-output",
                    str(stage / build_metadata_relative),
                    "--verbose",
                ],
                environment=build_environment,
                timeout_seconds=900.0,
            )
            build_command_ran = True
            if recorder.command_ok("extension_build") and (stage / build_metadata_relative).is_file():
                try:
                    build_payload = json.loads(
                        (stage / build_metadata_relative).read_text(encoding="utf-8")
                    )
                    candidate = Path(build_payload["module_path"]).resolve(strict=True)
                    candidate.relative_to(stage.resolve())
                    binary_path = candidate
                except Exception as error:
                    build_payload = {"status": "invalid", "error": str(error)}
                    binary_path = None
        if not (stage / build_metadata_relative).exists():
            write_exclusive(
                stage / build_metadata_relative,
                json_bytes(
                    {
                        "status": "NOT_RUN" if not build_command_ran else "FAIL",
                        "reason": "build preconditions did not pass" if not build_command_ran else "build did not produce metadata",
                    }
                ),
            )

        inspection_ids: list[str] = []
        binary_inspection_ok = False
        sass_present = False
        ptx_present = False
        if binary_path is not None:
            cuobjdump = lock_tools["cuobjdump"]["invocation_path"]
            inspection_commands = (
                ("cuobjdump_list_elf", "--list-elf"),
                ("cuobjdump_list_ptx", "--list-ptx"),
                ("cuobjdump_dump_sass", "--dump-sass"),
                ("cuobjdump_dump_ptx", "--dump-ptx"),
            )
            for command_id, option in inspection_commands:
                recorder.run(
                    command_id,
                    f"Inspect extension binary with cuobjdump {option}",
                    [cuobjdump, option, str(binary_path)],
                    timeout_seconds=120.0,
                )
                inspection_ids.append(command_id)
            arch_code = f"{selected_capability['major']}{selected_capability['minor']}"
            sass_text = recorder.command_text("cuobjdump_list_elf") + recorder.command_text("cuobjdump_dump_sass")
            ptx_text = recorder.command_text("cuobjdump_list_ptx") + recorder.command_text("cuobjdump_dump_ptx")
            sass_present = f"sm_{arch_code}" in sass_text
            ptx_present = (
                f"sm_{arch_code}" in ptx_text
                or f"compute_{arch_code}" in ptx_text
            )
            binary_inspection_ok = (
                status_from_commands(recorder, inspection_ids)
                and sass_present
                and ptx_present
            )

        runtime_ids: list[str] = []
        sanitizer_ids: list[str] = []
        native_ok = False
        forced_ptx_ok = False
        numerical_ok = False
        graph_ok = False
        allocation_ok = False
        sanitizer_ok = False
        sanitizer_errors: list[int | None] = []
        if binary_path is not None and binary_inspection_ok and selected_environment is not None:
            common_test_updates = {
                "E00_EXTENSION_PATH": str(binary_path),
                "E00_EXTENSION_MODULE": "e00_cuda_cert",
                "E00_ALLOCATION_ITERATIONS": "1000",
            }
            native_environment = environment_with(
                selected_environment,
                common_test_updates | {"CUDA_DISABLE_PTX_JIT": "1"},
                unset=(
                    "CUDA_FORCE_PTX_JIT",
                    "CUDA_FORCE_JIT",
                    "CUDA_DISABLE_JIT",
                    "CUDA_CACHE_DISABLE",
                    "CUDA_CACHE_PATH",
                    "TORCH_CUDA_ARCH_LIST",
                ),
            )
            recorder.run_supervised(
                "native_runtime",
                "Prove native SASS execution with PTX JIT disabled",
                [
                    str(project_python),
                    str(ROOT / "tests" / "cuda" / "e00_runtime_probe.py"),
                    "--mode",
                    "native",
                ],
                environment=native_environment,
                timeout_seconds=180.0,
            )
            runtime_ids.append("native_runtime")
            native_payload = recorder.command_payload("native_runtime")
            native_ok = (
                recorder.command_ok("native_runtime")
                and native_payload is not None
                and native_payload.get("status") == "pass"
                and native_payload.get("mode") == "native"
            )

            for command_id, purpose, script in (
                (
                    "numerical_golden",
                    "Run exact int32 numerical golden certification",
                    ROOT / "tests" / "golden" / "test_e00_numerical.py",
                ),
                (
                    "cuda_graph",
                    "Run CUDA Graph capture and changed-input replay certification",
                    ROOT / "tests" / "graph_capture" / "test_e00_graph.py",
                ),
                (
                    "allocation_stability",
                    "Run eager and Graph post-warmup allocation stability certification",
                    ROOT / "tests" / "allocation" / "test_e00_allocation.py",
                ),
            ):
                recorder.run_supervised(
                    command_id,
                    purpose,
                    [str(project_python), str(script)],
                    environment=native_environment,
                    timeout_seconds=240.0,
                )
                runtime_ids.append(command_id)

            numerical_payload = recorder.command_payload("numerical_golden")
            graph_payload = recorder.command_payload("cuda_graph")
            allocation_payload = recorder.command_payload("allocation_stability")
            graph_replays = nonnegative_int_value(
                graph_payload.get("replays") if graph_payload is not None else None
            )
            eager_iterations = nonnegative_int_value(
                allocation_payload.get("eager_iterations") if allocation_payload is not None else None
            )
            allocation_graph_replays = nonnegative_int_value(
                allocation_payload.get("graph_replays") if allocation_payload is not None else None
            )
            numerical_ok = (
                recorder.command_ok("numerical_golden")
                and numerical_payload is not None
                and numerical_payload.get("status") == "pass"
                and numerical_payload.get("lengths") == [1, 255, 256, 257, 4097]
            )
            graph_ok = (
                recorder.command_ok("cuda_graph")
                and graph_payload is not None
                and graph_payload.get("status") == "pass"
                and graph_replays is not None
                and graph_replays >= 3
            )
            allocation_ok = (
                recorder.command_ok("allocation_stability")
                and allocation_payload is not None
                and allocation_payload.get("status") == "pass"
                and eager_iterations is not None
                and eager_iterations >= 1000
                and allocation_graph_replays is not None
                and allocation_graph_replays >= 1000
            )

            ptx_cache = stage / "extension" / "ptx_cache"
            os.mkdir(ptx_cache, 0o700)
            ptx_environment = environment_with(
                selected_environment,
                common_test_updates
                | {
                    "CUDA_FORCE_PTX_JIT": "1",
                    "CUDA_CACHE_DISABLE": "1",
                    "CUDA_CACHE_PATH": str(ptx_cache),
                },
                unset=(
                    "CUDA_DISABLE_PTX_JIT",
                    "CUDA_DISABLE_JIT",
                    "CUDA_FORCE_JIT",
                    "TORCH_CUDA_ARCH_LIST",
                ),
            )
            recorder.run_supervised(
                "forced_ptx_runtime",
                "Prove forced PTX JIT execution in a fresh process",
                [
                    str(project_python),
                    str(ROOT / "tests" / "cuda" / "e00_runtime_probe.py"),
                    "--mode",
                    "forced-ptx",
                ],
                environment=ptx_environment,
                timeout_seconds=300.0,
            )
            runtime_ids.append("forced_ptx_runtime")
            forced_payload = recorder.command_payload("forced_ptx_runtime")
            forced_ptx_ok = (
                recorder.command_ok("forced_ptx_runtime")
                and forced_payload is not None
                and forced_payload.get("status") == "pass"
                and forced_payload.get("mode") == "forced-ptx"
            )

            sanitizer_base = environment_with(
                native_environment,
                {"PYTORCH_NO_CUDA_MEMORY_CACHING": "1"},
            )
            sanitizer_binary = lock_tools["compute-sanitizer-real"][
                "invocation_path"
            ]
            for tool in ("memcheck", "initcheck", "racecheck", "synccheck"):
                command_id = f"sanitizer_{tool}"
                command = [
                    sanitizer_binary,
                    "--tool",
                    tool,
                    "--error-exitcode",
                    "99",
                    "--target-processes",
                    "application-only",
                ]
                if tool == "memcheck":
                    command.extend(["--leak-check", "full"])
                command.extend(
                    [
                        str(project_python),
                        str(ROOT / "tests" / "cuda" / "e00_sanitizer_probe.py"),
                    ]
                )
                recorder.run_supervised(
                    command_id,
                    f"Run Compute Sanitizer {tool} on the assertion-based probe",
                    command,
                    environment=sanitizer_base,
                    timeout_seconds=900.0,
                )
                sanitizer_ids.append(command_id)
                sanitizer_errors.append(
                    sanitizer_error_count(tool, recorder.command_text(command_id))
                )
            sanitizer_ok = (
                status_from_commands(recorder, sanitizer_ids)
                and all(item == 0 for item in sanitizer_errors)
                and all(
                    (recorder.command_payload(command_id) or {}).get("status")
                    == "pass"
                    for command_id in sanitizer_ids
                )
            )

        all_log_text = "\n".join(
            recorder.command_text(command_id)
            for command_id in recorder.command_index
        )
        kernel_image_error = "no kernel image" in all_log_text.lower()
        any_runtime_ran = bool(runtime_ids or sanitizer_ids)
        kernel_execution_proven = native_ok and forced_ptx_ok

        required_audited_targets = (
            (["torch_probe"] if torch_command_ran else [])
            + runtime_ids
            + sanitizer_ids
        )
        snapshot_ids = {snapshot["command_id"] for snapshot in recorder.snapshots}
        audit_coverage_errors: list[str] = list(recorder.audit_errors)
        for target in required_audited_targets:
            for phase in ("before", "during", "after"):
                expected = safe_id(f"process_{target}_{phase}")
                if expected not in snapshot_ids:
                    audit_coverage_errors.append(
                        f"missing {phase} process snapshot for {target}"
                    )
            expected_outcome = f"audit/{target}.outcome.json"
            if expected_outcome not in recorder.audit_outcome_paths:
                audit_coverage_errors.append(
                    f"missing machine-readable audit outcome for {target}"
                )
        query_failures = sum(
            1 for snapshot in recorder.snapshots if snapshot["query_exit_code"] != 0
        )
        foreign_count = sum(
            len(snapshot["foreign_compute_processes"])
            + len(snapshot["unknown_processes"])
            for snapshot in recorder.snapshots
        )
        graphics_identities = {
            (
                process["gpu_uuid"],
                process["pid"],
                process["process_start_time_ticks"],
            )
            for snapshot in recorder.snapshots
            for process in snapshot["graphics_processes"]
        }
        supervised_identities = {
            (
                process["gpu_uuid"],
                process["pid"],
                process["process_start_time_ticks"],
            )
            for snapshot in recorder.snapshots
            for process in snapshot["allowed_compute_processes"]
        }
        process_ok = (
            query_failures == 0
            and foreign_count == 0
            and not audit_coverage_errors
        )
        collector_identity = proc_identity(os.getpid())
        if collector_identity is None:
            raise RuntimeError("cannot read collector PID/start identity")
        process_audit = {
            "policy": {
                "graphics_only_allowed": True,
                "foreign_compute_allowed": False,
                "supervised_identity_basis": "pid_and_process_start_time",
                "executable_name_whitelist_used": False,
                "unknown_process_type_fails_closed": True,
            },
            "collector_identity": {
                "pid": collector_identity["pid"],
                "start_time_ticks": collector_identity["start_time_ticks"],
            },
            "query_failure_count": query_failures,
            "graphics_process_count": len(graphics_identities),
            "supervised_compute_process_count": len(supervised_identities),
            "foreign_or_unknown_process_count": foreign_count,
            "snapshots": recorder.snapshots,
        }

        power_clock_complete = gpu_values_valid
        torch_cuda_ok = (
            torch_collection_ok
            and cuda_available is True
            and torch_device_count == 1
        )
        build_ok = (
            build_command_ran
            and recorder.command_ok("extension_build")
            and binary_path is not None
            and binary_inspection_ok
            and source_scan["status"] == "PASS"
        )
        native_cert_ok = native_ok and numerical_ok and graph_ok and allocation_ok

        build_status = "PASS" if build_ok else ("FAIL" if build_command_ran else "NOT_RUN")
        inspection_status = "PASS" if binary_inspection_ok else ("FAIL" if inspection_ids else "NOT_RUN")
        numerical_status = "PASS" if numerical_ok else ("FAIL" if "numerical_golden" in runtime_ids else "NOT_RUN")
        native_status = "PASS" if native_ok else ("FAIL" if "native_runtime" in runtime_ids else "NOT_RUN")
        ptx_status = "PASS" if forced_ptx_ok else ("FAIL" if "forced_ptx_runtime" in runtime_ids else "NOT_RUN")
        sanitizer_status = "PASS" if sanitizer_ok else ("FAIL" if sanitizer_ids else "NOT_RUN")
        graph_status = "PASS" if graph_ok else ("FAIL" if "cuda_graph" in runtime_ids else "NOT_RUN")
        allocation_status = "PASS" if allocation_ok else ("FAIL" if "allocation_stability" in runtime_ids else "NOT_RUN")
        kernel_status = (
            "FAIL"
            if kernel_image_error
            else "PASS"
            if kernel_execution_proven
            else "FAIL"
            if any_runtime_ran
            else "NOT_RUN"
        )
        kernel_reason = (
            None
            if kernel_status == "PASS"
            else "a no-kernel-image error was detected"
            if kernel_image_error
            else "required native and forced-PTX kernel execution was not proven"
            if any_runtime_ran
            else "runtime was not executed"
        )

        binary_relative = (
            binary_path.relative_to(stage.resolve()).as_posix()
            if binary_path is not None
            else None
        )
        binary_record = {
            "available": binary_path is not None,
            "path": binary_relative,
            "sha256": sha256_file(binary_path) if binary_path is not None else None,
            "size_bytes": binary_path.stat().st_size if binary_path is not None else None,
        }
        build_metadata_ref = output_ref(stage, build_metadata_relative)
        certified_arch = selected_capability if capability_match else None
        certified_arch_code = (
            f"{certified_arch['major']}{certified_arch['minor']}"
            if certified_arch is not None
            else None
        )

        extension_manifest = {
            "name": "e00_cuda_certification",
            "sources": source_records,
            "architecture": {
                "derived_from": (
                    "nvidia_smi_and_torch_cuda_capability_agreement"
                ),
                "compute_capability": certified_arch,
                "equivalent_torch_cuda_arch_list": (
                    f"{certified_arch['text']}+PTX"
                    if certified_arch is not None
                    else None
                ),
                "compiled_sass_targets": (
                    [f"sm_{certified_arch_code}"] if binary_inspection_ok else []
                ),
                "compiled_ptx_targets": (
                    [f"compute_{certified_arch_code}"]
                    if binary_inspection_ok
                    else []
                ),
            },
            "binary": binary_record,
            "build": result_envelope(
                build_status,
                reason=None if build_status == "PASS" else "extension build or binary inspection did not pass",
                command_ids=("extension_build",) if build_command_ran else (),
                evidence_file_ids=(
                    command_file_ids(recorder, ("extension_build",))
                    + [artifact_id(build_metadata_relative)]
                ),
            ),
            "binary_inspection": {
                **result_envelope(
                    inspection_status,
                    reason=None if inspection_status == "PASS" else "required SASS/PTX binary evidence missing",
                    command_ids=inspection_ids,
                    evidence_file_ids=command_file_ids(recorder, inspection_ids),
                ),
                "sass_target_present": sass_present if inspection_ids else None,
                "ptx_target_present": ptx_present if inspection_ids else None,
            },
            "numerical_golden": {
                **result_envelope(
                    numerical_status,
                    reason=None if numerical_status == "PASS" else "exact int32 golden command did not pass",
                    command_ids=("numerical_golden",) if "numerical_golden" in runtime_ids else (),
                    evidence_file_ids=command_file_ids(recorder, ("numerical_golden",)),
                ),
                "dtype": "int32",
                "case_count": 5 if numerical_ok else None,
                "atol": 0.0 if numerical_ok else None,
                "rtol": 0.0 if numerical_ok else None,
                "max_abs_error": 0.0 if numerical_ok else None,
                "max_rel_error": 0.0 if numerical_ok else None,
            },
            "native_execution": {
                **result_envelope(
                    native_status,
                    reason=None if native_status == "PASS" else "native process with PTX JIT disabled did not pass",
                    command_ids=("native_runtime",) if "native_runtime" in runtime_ids else (),
                    evidence_file_ids=command_file_ids(recorder, ("native_runtime",)),
                ),
                "separate_process": True if "native_runtime" in runtime_ids else None,
                "cuda_disable_ptx_jit": True if "native_runtime" in runtime_ids else None,
            },
            "forced_ptx_jit": {
                **result_envelope(
                    ptx_status,
                    reason=None if ptx_status == "PASS" else "forced-PTX fresh process did not pass",
                    command_ids=("forced_ptx_runtime",) if "forced_ptx_runtime" in runtime_ids else (),
                    evidence_file_ids=command_file_ids(recorder, ("forced_ptx_runtime",)),
                ),
                "separate_process": True if "forced_ptx_runtime" in runtime_ids else None,
                "cuda_force_ptx_jit": True if "forced_ptx_runtime" in runtime_ids else None,
                "fresh_cuda_cache": True if "forced_ptx_runtime" in runtime_ids else None,
            },
            "compute_sanitizer": {
                **result_envelope(
                    sanitizer_status,
                    reason=None if sanitizer_status == "PASS" else "one or more required Compute Sanitizer tools did not pass cleanly",
                    command_ids=sanitizer_ids,
                    evidence_file_ids=command_file_ids(recorder, sanitizer_ids),
                ),
                "tools_run": [
                    command_id.removeprefix("sanitizer_")
                    for command_id in sanitizer_ids
                ],
                "leak_check_full": (
                    True if "sanitizer_memcheck" in sanitizer_ids else None
                ),
                "error_count": (
                    sum(item for item in sanitizer_errors if item is not None)
                    if sanitizer_errors
                    and all(item is not None for item in sanitizer_errors)
                    else None
                ),
            },
            "cuda_graph_capture_replay": {
                **result_envelope(
                    graph_status,
                    reason=None if graph_status == "PASS" else "CUDA Graph capture/replay command did not pass",
                    command_ids=("cuda_graph",) if "cuda_graph" in runtime_ids else (),
                    evidence_file_ids=command_file_ids(recorder, ("cuda_graph",)),
                ),
                "capture_succeeded": True if graph_ok else None,
                "replay_count": 3 if graph_ok else None,
                "output_matches_golden": True if graph_ok else None,
            },
            "memory_allocation": {
                **result_envelope(
                    allocation_status,
                    reason=None if allocation_status == "PASS" else "post-warmup allocation counters were not certified stable",
                    command_ids=("allocation_stability",) if "allocation_stability" in runtime_ids else (),
                    evidence_file_ids=command_file_ids(recorder, ("allocation_stability",)),
                ),
                "preallocated_input_output": True if allocation_ok else None,
                "allocated_delta_bytes": 0 if allocation_ok else None,
                "reserved_delta_bytes": 0 if allocation_ok else None,
                "output_pointer_stable": True if allocation_ok else None,
            },
            "no_kernel_image_error": {
                **result_envelope(
                    kernel_status,
                    reason=kernel_reason,
                    command_ids=runtime_ids + sanitizer_ids,
                    evidence_file_ids=command_file_ids(recorder, runtime_ids + sanitizer_ids),
                ),
                "error_found": kernel_image_error if any_runtime_ran else None,
            },
            "build_metadata": build_metadata_ref,
            "cuobjdump_command_ids": inspection_ids,
            "kernel_image_error_detected": kernel_image_error if any_runtime_ran else None,
        }

        checks = [
            gate_check(
                "committed_clean_code",
                "PASS",
                command_ids=(
                    "git_head",
                    "git_status_clean",
                    "git_contract_index",
                    "git_contract_flags",
                ),
                evidence_file_ids=command_file_ids(
                    recorder,
                    (
                        "git_head",
                        "git_status_clean",
                        "git_contract_index",
                        "git_contract_flags",
                    ),
                )
                + [artifact_id(source_git_relative)],
            ),
            gate_check(
                gate_names[1],
                "PASS" if environment_ok else "FAIL",
                reason=None if environment_ok else "; ".join(environment_errors),
                command_ids=("host_container_detection", "host_cgroup", "host_self_cgroup"),
                evidence_file_ids=(
                    command_file_ids(
                        recorder,
                        (
                            "host_container_detection",
                            "host_cgroup",
                            "host_self_cgroup",
                        ),
                    )
                    + [artifact_id(environment_relative)]
                    + (
                        [
                            artifact_id(
                                container_identity_validation_relative
                            ),
                            artifact_id(
                                container_runtime_validation_relative
                            ),
                        ]
                        if measurement_container_mode
                        else []
                    )
                    + (
                        [
                            artifact_id(container_inspect_relative),
                            artifact_id(container_runtime_inspect_relative),
                        ]
                        if measurement_container_mode
                        and container_identity_ok
                        else []
                    )
                ),
            ),
            gate_check(
                "single_uuid_selected_gpu",
                "PASS" if selection_ok else "FAIL",
                reason=None if selection_ok else "target GPU selection was not unique by UUID",
                command_ids=("nvidia_smi_gpu_csv",),
                evidence_file_ids=command_file_ids(recorder, ("nvidia_smi_gpu_csv",)),
            ),
            gate_check(
                "target_sku_match",
                "PASS" if selected is not None and selected.get("name") == EXPECTED_GPU_NAME else "FAIL",
                reason=None if selected is not None and selected.get("name") == EXPECTED_GPU_NAME else "full GPU SKU did not exactly match the Phase 1 target",
                command_ids=("nvidia_smi_gpu_csv",),
                evidence_file_ids=command_file_ids(recorder, ("nvidia_smi_gpu_csv",)),
            ),
            gate_check(
                "hardware_identity_complete",
                "PASS" if host_ok and gpu_collection_ok else "FAIL",
                reason=None if host_ok and gpu_collection_ok else "host or GPU identity inventory is incomplete",
                command_ids=host_command_ids + ("nvidia_smi_list", "nvidia_smi_query_full", "nvidia_smi_gpu_csv"),
                evidence_file_ids=command_file_ids(recorder, host_command_ids + ("nvidia_smi_list", "nvidia_smi_query_full", "nvidia_smi_gpu_csv")),
            ),
            gate_check(
                "power_clock_state_complete",
                "PASS" if power_clock_complete else "FAIL",
                reason=None if power_clock_complete else "required power, clock, ECC, or thermal state is missing",
                command_ids=("nvidia_smi_query_full", "nvidia_smi_gpu_csv"),
                evidence_file_ids=command_file_ids(recorder, ("nvidia_smi_query_full", "nvidia_smi_gpu_csv")),
            ),
            gate_check(
                "required_toolchain_complete",
                "PASS" if required_tool_commands_ok and software_manifest["collection_status"] == "PASS" else "FAIL",
                reason=None if required_tool_commands_ok and software_manifest["collection_status"] == "PASS" else "locked CUDA/Python/profiler toolchain did not verify",
                command_ids=version_ids + ("dpkg_locked_versions", "dpkg_architecture", "nvdisasm_package_ownership", "dpkg_verify_locked_files", "pip_check", "pip_list") + container_inventory_ids + ("python_record_integrity",),
                evidence_file_ids=command_file_ids(recorder, version_ids + ("dpkg_locked_versions", "dpkg_architecture", "nvdisasm_package_ownership", "dpkg_verify_locked_files", "pip_check", "pip_list") + container_inventory_ids + ("python_record_integrity",)) + [artifact_id(dependency_relative)],
            ),
            gate_check(
                "torch_cuda_available",
                "PASS" if torch_cuda_ok else ("FAIL" if torch_command_ran else "NOT_RUN"),
                reason=None if torch_cuda_ok else "torch.cuda availability/device-count probe did not pass",
                command_ids=("torch_probe",) if torch_command_ran else (),
                evidence_file_ids=command_file_ids(recorder, ("torch_probe",)),
            ),
            gate_check(
                "capability_identity_match",
                "PASS" if capability_match and uuid_match else ("FAIL" if torch_command_ran else "NOT_RUN"),
                reason=None if capability_match and uuid_match else "PyTorch and NVIDIA UUID/capability observations did not match",
                command_ids=("nvidia_smi_gpu_csv", "torch_probe") if torch_command_ran else ("nvidia_smi_gpu_csv",),
                evidence_file_ids=command_file_ids(recorder, ("nvidia_smi_gpu_csv", "torch_probe")),
            ),
            gate_check(
                "extension_build",
                "PASS" if build_ok else ("FAIL" if build_command_ran else "NOT_RUN"),
                reason=None if build_ok else "extension build, source audit, or SASS/PTX inspection did not pass",
                command_ids=(("extension_build",) if build_command_ran else ()) + tuple(inspection_ids),
                evidence_file_ids=command_file_ids(recorder, (("extension_build",) if build_command_ran else ()) + tuple(inspection_ids)) + [artifact_id(source_scan_relative), artifact_id(build_metadata_relative)],
            ),
            gate_check(
                "native_sass_execution",
                "PASS" if native_cert_ok else ("FAIL" if runtime_ids else "NOT_RUN"),
                reason=None if native_cert_ok else "native, numerical, Graph, or allocation certification did not pass",
                command_ids=tuple(item for item in ("native_runtime", "numerical_golden", "cuda_graph", "allocation_stability") if item in runtime_ids),
                evidence_file_ids=command_file_ids(recorder, tuple(item for item in ("native_runtime", "numerical_golden", "cuda_graph", "allocation_stability") if item in runtime_ids)),
            ),
            gate_check(
                "forced_ptx_jit_execution",
                "PASS" if forced_ptx_ok else ("FAIL" if "forced_ptx_runtime" in runtime_ids else "NOT_RUN"),
                reason=None if forced_ptx_ok else "forced PTX/JIT execution did not pass",
                command_ids=("forced_ptx_runtime",) if "forced_ptx_runtime" in runtime_ids else (),
                evidence_file_ids=command_file_ids(recorder, ("forced_ptx_runtime",)),
            ),
            gate_check(
                "compute_sanitizer_clean",
                "PASS" if sanitizer_ok else ("FAIL" if sanitizer_ids else "NOT_RUN"),
                reason=None if sanitizer_ok else "memcheck/initcheck/racecheck/synccheck did not all report zero errors",
                command_ids=sanitizer_ids,
                evidence_file_ids=command_file_ids(recorder, sanitizer_ids),
            ),
            gate_check(
                "no_kernel_image_error",
                kernel_status,
                reason=kernel_reason,
                command_ids=runtime_ids + sanitizer_ids,
                evidence_file_ids=command_file_ids(recorder, runtime_ids + sanitizer_ids),
            ),
            gate_check(
                "no_foreign_compute_process",
                "PASS" if process_ok else "FAIL",
                reason=None if process_ok else "; ".join(audit_coverage_errors or [f"query_failures={query_failures}, foreign_or_unknown={foreign_count}"]),
                command_ids=tuple(snapshot["command_id"] for snapshot in recorder.snapshots),
                evidence_file_ids=command_file_ids(recorder, tuple(snapshot["command_id"] for snapshot in recorder.snapshots))
                + [
                    artifact_id(relative)
                    for relative in recorder.audit_outcome_paths
                ],
            ),
            gate_check("manifest_schema_valid", "NOT_RUN", reason="manifest schema validation pending"),
            gate_check("evidence_checksums_valid", "NOT_RUN", reason="evidence cross-reference and checksum validation pending"),
        ]

        schema_check_index = gate_names.index("manifest_schema_valid")
        evidence_check_index = gate_names.index("evidence_checksums_valid")

        def failure_classes_for(current_checks: Sequence[dict[str, Any]]) -> list[str]:
            statuses = {item["name"]: item["status"] for item in current_checks}
            classes: list[str] = []

            def add(value: str) -> None:
                if value not in classes:
                    classes.append(value)

            if statuses[gate_names[1]] == "FAIL":
                add("environment_unverified")
            if any(
                statuses[name] == "FAIL"
                for name in (
                    "single_uuid_selected_gpu",
                    "target_sku_match",
                    "hardware_identity_complete",
                    "power_clock_state_complete",
                )
            ):
                add("hardware_inventory_failed")
            if statuses["required_toolchain_complete"] == "FAIL":
                add("toolchain_failed")
            if any(
                statuses[name] != "PASS"
                for name in ("torch_cuda_available", "capability_identity_match")
            ):
                add("cuda_capability_mismatch")
            if statuses["extension_build"] != "PASS":
                add("build_failed")
            if statuses["native_sass_execution"] != "PASS":
                add("runtime_failed")
                if "numerical_golden" in runtime_ids and not numerical_ok:
                    add("numerical_failed")
                if "cuda_graph" in runtime_ids and not graph_ok:
                    add("graph_capture_failed")
                if "allocation_stability" in runtime_ids and not allocation_ok:
                    add("allocation_failed")
            if any(
                statuses[name] != "PASS"
                for name in ("forced_ptx_jit_execution", "no_kernel_image_error")
            ):
                add("runtime_failed")
            if statuses["compute_sanitizer_clean"] != "PASS":
                add("sanitizer_failed")
            if statuses["no_foreign_compute_process"] == "FAIL":
                add("process_isolation_failed")
            if statuses["manifest_schema_valid"] != "PASS":
                add("manifest_validation_failed")
            if statuses["evidence_checksums_valid"] != "PASS":
                add("evidence_integrity_failed")
            return classes

        finished_at = utc_now()
        aggregate_status = (
            "PASS" if all(check["status"] == "PASS" for check in checks) else "FAIL"
        )
        failure_reasons = list(
            dict.fromkeys(
                check["reason"]
                for check in checks
                if check["status"] != "PASS" and check["reason"]
            )
        )

        contract_records = source_artifacts(contract_paths)
        if measurement_container_mode:
            execution_environment = {
                "kind": (
                    "measurement_container"
                    if environment_ok
                    else "unverified"
                ),
                "verification_status": "PASS" if environment_ok else "FAIL",
                "verification_evidence_file_id": artifact_id(
                    environment_relative
                ),
                "performance_claim_eligible": False,
                "performance_ineligibility_reasons": [
                    *(
                        []
                        if environment_ok
                        else ["execution_environment_not_verified"]
                    ),
                    "container_bf16_parity_not_yet_established",
                    "hardware_certification_not_benchmark_timing",
                ],
                "container": {
                    "runtime": "docker",
                    "image_reference": args.container_image_reference,
                    "image_config_digest": args.container_image_config_digest,
                    "base_image_digest": args.container_base_image_digest,
                    "image_inspect_validation": output_ref(
                        stage,
                        container_identity_validation_relative,
                    ),
                    "image_inspect": (
                        output_ref(stage, container_inspect_relative)
                        if container_identity_ok
                        else None
                    ),
                    "runtime_container_id": container_runtime_validation[
                        "runtime_container_id"
                    ],
                    "runtime_inspect_validation": output_ref(
                        stage,
                        container_runtime_validation_relative,
                    ),
                    "runtime_inspect": (
                        output_ref(stage, container_runtime_inspect_relative)
                        if container_identity_ok
                        else None
                    ),
                    "digest_status": (
                        "verified_against_sanitized_image_inspect"
                        if container_identity_ok
                        else "image_inspect_verification_failed"
                    ),
                },
                "container_parity_required_before_e02": True,
            }
        else:
            execution_environment = {
                "kind": "native_host" if environment_ok else "unverified",
                "verification_status": "PASS" if environment_ok else "FAIL",
                "verification_evidence_file_id": artifact_id(
                    environment_relative
                ),
                "performance_claim_eligible": False,
                "performance_ineligibility_reasons": ([
                    "native_host_without_container_digest"
                    if environment_ok
                    else "execution_environment_not_verified",
                    "hardware_certification_not_benchmark_timing",
                ]),
                "container": {
                    "runtime": None,
                    "image_reference": None,
                    "image_digest": None,
                    "digest_status": (
                        "not_applicable_native_host"
                        if environment_ok
                        else "unavailable_unverified_environment"
                    ),
                },
                "container_parity_required_before_e02": True,
            }

        recorder.commands.sort(key=lambda item: (item["started_at_utc"], item["id"]))
        manifest: dict[str, Any] = {
            "schema_version": (
                "e00-manifest-1.1.0"
                if measurement_container_mode
                else "e00-manifest-1.0.0"
            ),
            "run": {
                "id": run_id,
                "phase": "E00",
                "gate": "G0",
                "kind": "hardware_preflight",
                "status": aggregate_status,
                "started_at_utc": started_at,
                "finished_at_utc": finished_at,
                "completed": True,
                "benchmark_timing_collected": False,
                "failure_reasons": failure_reasons,
                "failure_classes": failure_classes_for(checks),
            },
            "code": {
                "git_sha": git_sha,
                "git_dirty": False,
                "worktree_clean_before_run_directory_creation": True,
                "contract_files": contract_records,
            },
            "execution_environment": execution_environment,
            "host": host_manifest,
            "gpu_selection": gpu_selection,
            "gpu": gpu_manifest,
            "software": software_manifest,
            "torch_probe": torch_probe_manifest,
            "process_audit": process_audit,
            "extension": extension_manifest,
            "evidence": {
                "storage": {
                    "root": evidence_root.relative_to(ROOT).as_posix(),
                    "run_directory": final.relative_to(ROOT).as_posix(),
                    "append_only": True,
                    "exclusive_temporary_sibling": True,
                    "atomic_final_rename": True,
                    "no_overwrite_final_rename": True,
                },
                "commands": recorder.commands,
                "files": [],
                "checksum_ledger": {
                    "algorithm": "sha256",
                    "path": "checksums.sha256",
                    "covers_all_payload_files": True,
                    "exclusions": ["checksums.sha256", "COMPLETE"],
                },
                "checksum_verification": result_envelope(
                    "NOT_RUN",
                    reason="evidence cross-reference and checksum validation pending",
                ),
                "completion_marker": {
                    "path": "COMPLETE",
                    "written_last": True,
                    "contains_ledger_sha256": True,
                },
            },
            "gate": {
                "name": "G0",
                "aggregate_status": aggregate_status,
                "checks": checks,
            },
        }

        def refresh_manifest_outcome() -> None:
            current_status = (
                "PASS"
                if all(check["status"] == "PASS" for check in checks)
                else "FAIL"
            )
            current_reasons = list(
                dict.fromkeys(
                    check["reason"]
                    for check in checks
                    if check["status"] != "PASS" and check["reason"]
                )
            )
            manifest["run"]["status"] = current_status
            manifest["run"]["failure_reasons"] = current_reasons
            manifest["run"]["failure_classes"] = failure_classes_for(checks)
            manifest["gate"]["aggregate_status"] = current_status
            manifest["gate"]["checks"] = checks

        schema_validation_relative = "validation/manifest_schema.json"
        manifest["evidence"]["files"] = enumerate_evidence_files(stage)
        preliminary_schema_errors = validate_manifest(manifest)
        checks[schema_check_index] = gate_check(
            "manifest_schema_valid",
            "PASS" if not preliminary_schema_errors else "FAIL",
            reason=(
                None
                if not preliminary_schema_errors
                else f"manifest failed schema validation with {len(preliminary_schema_errors)} error(s)"
            ),
            evidence_file_ids=[artifact_id(schema_validation_relative)],
        )
        refresh_manifest_outcome()
        schema_validation_payload = {
            "status": "PASS" if not preliminary_schema_errors else "FAIL",
            "schema_path": "preflight/e00_manifest.schema.json",
            "schema_sha256": sha256_file(SCHEMA_PATH),
            "errors": preliminary_schema_errors,
        }
        write_exclusive(
            stage / schema_validation_relative,
            json_bytes(schema_validation_payload),
        )
        manifest["evidence"]["files"] = enumerate_evidence_files(stage)

        post_record_schema_errors = validate_manifest(manifest)
        if post_record_schema_errors:
            fatal_relative = "validation/post_record_manifest_schema_errors.json"
            write_exclusive(
                stage / fatal_relative,
                json_bytes({"errors": post_record_schema_errors}),
            )
            raise RuntimeError(
                "E00 manifest does not validate after adding its schema-validation record: "
                + post_record_schema_errors[0]
            )

        reference_errors = evidence_reference_errors(stage, manifest)
        checks[evidence_check_index] = gate_check(
            "evidence_checksums_valid",
            "PASS" if not reference_errors else "FAIL",
            reason=(
                None
                if not reference_errors
                else f"evidence cross-reference failed with {len(reference_errors)} error(s)"
            ),
            evidence_file_ids=[
                item["id"] for item in manifest["evidence"]["files"]
            ],
        )
        manifest["evidence"]["checksum_verification"] = result_envelope(
            "PASS" if not reference_errors else "FAIL",
            reason=(
                None
                if not reference_errors
                else f"evidence cross-reference failed with {len(reference_errors)} error(s)"
            ),
            evidence_file_ids=[
                item["id"] for item in manifest["evidence"]["files"]
            ],
        )
        refresh_manifest_outcome()

        if reference_errors:
            reference_error_relative = "validation/evidence_reference_errors.json"
            write_exclusive(
                stage / reference_error_relative,
                json_bytes({"errors": reference_errors}),
            )
            raise RuntimeError(
                f"E00 evidence cross-reference failed: {reference_errors[0]}"
            )

        final_errors = validate_manifest(manifest)
        if final_errors:
            fatal_relative = "validation/final_manifest_schema_errors.json"
            write_exclusive(
                stage / fatal_relative,
                json_bytes({"errors": final_errors}),
            )
            raise RuntimeError(
                f"final E00 manifest does not validate: {final_errors[0]}"
            )

        finalize_stage(stage=stage, final=final, manifest=manifest)
        print(f"[e00] finalized {final.relative_to(ROOT)}", flush=True)
        print(f"[e00] G0 {manifest['gate']['aggregate_status']}", flush=True)
        return 0 if manifest["gate"]["aggregate_status"] == "PASS" else 2
    except Exception as error:
        try:
            write_exclusive(
                stage / "UNFINALIZED_ERROR.json",
                json_bytes(
                    {
                        "run_id": run_id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "captured_at_utc": utc_now(),
                        "append_only_retention_required": True,
                    }
                ),
            )
        except Exception:
            pass
        print(
            f"E00 collector failed; retained unfinalized staging evidence at {stage}: {error}",
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
