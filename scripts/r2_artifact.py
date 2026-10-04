#!/usr/bin/env python3
"""Publish and verify finalized kvbench artifacts in Cloudflare R2.

This is deliberately provider-specific.  It implements only the two Phase 6A
operations, ``publish`` and ``verify``, and uses the existing finalized
artifact controls as the authority for file identity.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import hmac
import http.client
import json
import mimetypes
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


REGION = "auto"
SERVICE = "s3"
EXPECTED_R2_PREFIX = "kvbench/sha256"
ENDPOINT_CLASS = "cloudflare_r2_s3"
SINGLE_PUT_MAX_BYTES = 64 * 1024**2
MULTIPART_PART_SIZE_BYTES = 32 * 1024**2
MULTIPART_UPLOAD_ATTEMPTS = 5
OBJECT_READ_ATTEMPTS = 8
OBJECT_RANGE_SIZE_BYTES = 32 * 1024**2
R2_READ_WORKERS = 8
R2_UPLOAD_WORKERS = 4
MAX_MULTIPART_PARTS = 10_000
RETRYABLE_MULTIPART_STATUSES = frozenset({429, 500, 502, 503, 504})
CONTROL_FILES = (
    "manifest.json",
    "artifact_inventory.json",
    "checksums.sha256",
    "COMPLETE",
)
INVENTORY_SCHEMA_VERSION = "kvbench-artifact-inventory-1.0.0"
INVENTORY_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "files",
        "excluded_control_files",
    }
)
INVENTORY_ITEM_KEYS = frozenset({"path", "role", "size_bytes", "sha256"})
INVENTORY_EXCLUDED_CONTROLS = (
    "artifact_inventory.json",
    "checksums.sha256",
    "COMPLETE",
)
E00_CONTAINER_SCHEMA_VERSION = "e00-manifest-1.1.0"
PHASE6A_PARITY_SCHEMA_VERSION = (
    "kvbench-phase6a-bf16-container-parity-1.0.0"
)
OBJECT_ACCESS_VARIABLES = (
    "R2_ACCOUNT_ID",
    "R2_BUCKET",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "KVBENCH_R2_PREFIX",
)
LOCK_VARIABLE = "CLOUDFLARE_API_TOKEN"
REQUIRED_VARIABLES = (*OBJECT_ACCESS_VARIABLES, LOCK_VARIABLE)
STATUS_VARIABLES = (
    "R2_ACCOUNT_ID",
    "R2_BUCKET",
    "R2_ENDPOINT",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "CLOUDFLARE_API_TOKEN",
    "KVBENCH_R2_PREFIX",
)
SECRET_VARIABLES = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "CLOUDFLARE_API_TOKEN",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "R2_ACCOUNT_ID",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_BUCKET_RE = re.compile(r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]")
_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
_PROHIBITED_JSON_KEYS = frozenset(
    {
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "cloudflare_api_token",
        "hf_token",
        "hugging_face_hub_token",
        "r2_account_id",
        "authorization",
    }
)


class R2ArtifactError(RuntimeError):
    """Base class for safe, redacted Phase 6A R2 failures."""


class ArtifactValidationError(R2ArtifactError):
    """The local or reconstructed artifact is not safely finalized."""


class ObjectConflictError(R2ArtifactError):
    """An existing content-addressed object has different bytes."""


class RemoteRequestError(R2ArtifactError):
    """A provider request failed without retaining a secret-bearing message."""

    def __init__(
        self,
        *,
        status: int | None,
        code: str,
        failed_check: str | None = None,
        failed_path: str | None = None,
        provider_message: str | None = None,
    ) -> None:
        self.status = status
        self.code = code
        self.failed_check = failed_check
        self.failed_path = failed_path
        self.provider_message = provider_message
        status_text = str(status) if status is not None else "unavailable"
        details = [f"HTTP {status_text}", f"code {code}"]
        if failed_check is not None:
            details.append(f"check {failed_check}")
        if failed_path is not None:
            details.append(f"path {failed_path}")
        if provider_message is not None:
            details.append(f"provider message {provider_message}")
        super().__init__(f"Cloudflare R2 request failed ({', '.join(details)})")

    def to_dict(self) -> dict[str, object]:
        return {
            "http_status": self.status,
            "provider_error_code": self.code,
            "provider_error_message": self.provider_message,
            "failed_check": self.failed_check,
            "failed_path": self.failed_path,
        }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_variable_status(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return names and PRESENT/MISSING only; values are never returned."""

    source = os.environ if environ is None else environ
    return {
        name: "PRESENT" if bool(source.get(name)) else "MISSING"
        for name in STATUS_VARIABLES
    }


def redact_text(
    text: str,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Redact configured credentials without echoing their values."""

    source = os.environ if environ is None else environ
    redacted = str(text)
    replacements = sorted(
        (
            (value, f"<redacted:{name}>")
            for name in SECRET_VARIABLES
            if (value := source.get(name))
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for value, marker in replacements:
        redacted = redacted.replace(value, marker)
    redacted = re.sub(
        r"(?i)(authorization\s*:\s*(?:bearer|aws4-hmac-sha256)\s+)\S+",
        r"\1<redacted>",
        redacted,
    )
    return redacted


def _require_safe_relative(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ArtifactValidationError("artifact path is not a safe POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ArtifactValidationError("artifact path is not a safe POSIX path")
    return path.as_posix()


def normalize_prefix(value: str) -> str:
    """Return a canonical, non-empty R2 prefix without a trailing slash."""

    if not isinstance(value, str) or not value or value != value.strip():
        raise R2ArtifactError("KVBENCH_R2_PREFIX is invalid")
    if value.startswith(("/", "\\")) or "\\" in value:
        raise R2ArtifactError("KVBENCH_R2_PREFIX is invalid")
    candidate = value[:-1] if value.endswith("/") else value
    if not candidate or candidate.endswith("/"):
        raise R2ArtifactError("KVBENCH_R2_PREFIX is invalid")
    try:
        normalized = _require_safe_relative(candidate)
    except ArtifactValidationError as error:
        raise R2ArtifactError("KVBENCH_R2_PREFIX is invalid") from error
    return normalized


def validate_root_sha256(value: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise R2ArtifactError("root SHA-256 must be 64 lowercase hexadecimal digits")
    return value


def artifact_object_prefix(base_prefix: str, root_sha256: str) -> str:
    return f"{normalize_prefix(base_prefix)}/{validate_root_sha256(root_sha256)}/"


def artifact_object_key(
    base_prefix: str,
    root_sha256: str,
    relative_path: str,
) -> str:
    prefix = artifact_object_prefix(base_prefix, root_sha256)
    relative = _require_safe_relative(relative_path)
    key = f"{prefix}{relative}"
    if not key.startswith(prefix):
        raise R2ArtifactError("object key escaped the configured prefix")
    return key


def _validate_endpoint(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise R2ArtifactError("R2_ENDPOINT must be an HTTPS origin without credentials")
    return f"https://{parsed.netloc}"


@dataclass(frozen=True)
class R2Config:
    endpoint: str = field(repr=False)
    bucket: str
    prefix: str
    account_id: str = field(repr=False)
    access_key_id: str = field(repr=False)
    secret_access_key: str = field(repr=False)
    cloudflare_api_token: str = field(repr=False)

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "R2Config":
        source = os.environ if environ is None else environ
        if any(not source.get(name) for name in REQUIRED_VARIABLES):
            raise R2ArtifactError("required R2 variables are missing")
        account_id = source["R2_ACCOUNT_ID"]
        if re.fullmatch(r"[0-9a-f]{32}", account_id) is None:
            raise R2ArtifactError("R2_ACCOUNT_ID is invalid")
        expected_endpoint = (
            f"https://{account_id}.r2.cloudflarestorage.com"
        )
        endpoint = source.get(
            "R2_ENDPOINT",
            expected_endpoint,
        )
        validated_endpoint = _validate_endpoint(endpoint)
        if validated_endpoint != expected_endpoint:
            raise R2ArtifactError(
                "R2_ENDPOINT does not match the configured Cloudflare account"
            )
        bucket = source["R2_BUCKET"]
        if _BUCKET_RE.fullmatch(bucket) is None:
            raise R2ArtifactError("R2_BUCKET is invalid")
        prefix = normalize_prefix(source["KVBENCH_R2_PREFIX"])
        if prefix != EXPECTED_R2_PREFIX:
            raise R2ArtifactError(
                f"KVBENCH_R2_PREFIX must normalize to {EXPECTED_R2_PREFIX}"
            )
        return cls(
            endpoint=validated_endpoint,
            bucket=bucket,
            prefix=prefix,
            account_id=account_id,
            access_key_id=source["AWS_ACCESS_KEY_ID"],
            secret_access_key=source["AWS_SECRET_ACCESS_KEY"],
            cloudflare_api_token=source["CLOUDFLARE_API_TOKEN"],
        )

    def endpoint_identity(self) -> str:
        """Return a recordable endpoint identity without the account ID."""

        return self.endpoint.replace(self.account_id, "<R2_ACCOUNT_ID>")

    def public_identity(self) -> dict[str, str]:
        return {
            "provider": "cloudflare_r2",
            "endpoint_class": ENDPOINT_CLASS,
            "endpoint": self.endpoint_identity(),
            "bucket": self.bucket,
            "prefix": self.prefix,
            "region": REGION,
        }


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite value {value}")
            ),
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise ArtifactValidationError(f"{label} is invalid") from error
    if not isinstance(payload, dict):
        raise ArtifactValidationError(f"{label} is invalid")
    return payload


def _reject_credential_keys(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and key.lower() in _PROHIBITED_JSON_KEYS:
                raise ArtifactValidationError(
                    "artifact contains prohibited credential material"
                )
            _reject_credential_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_credential_keys(child)


def _contains_bytes(path: Path, needle: bytes) -> bool:
    if not needle:
        return False
    overlap = max(0, len(needle) - 1)
    tail = b""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            data = tail + chunk
            if needle in data:
                return True
            tail = data[-overlap:] if overlap else b""
    return False


def _is_env_path(relative: str) -> bool:
    return any(part == ".env" or part.startswith(".env.") for part in PurePosixPath(relative).parts)


@dataclass(frozen=True)
class ArtifactFile:
    relative_path: str
    path: Path
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ValidatedArtifact:
    directory: Path
    root_sha256: str
    files: tuple[ArtifactFile, ...]

    def by_path(self) -> dict[str, ArtifactFile]:
        return {item.relative_path: item for item in self.files}


def _collect_files(directory: Path, *, require_immutable: bool) -> list[Path]:
    try:
        root_metadata = directory.lstat()
    except OSError as error:
        raise ArtifactValidationError("artifact directory is missing or unsafe") from error
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise ArtifactValidationError("artifact directory is missing or unsafe")
    if require_immutable and root_metadata.st_mode & _WRITE_BITS:
        raise ArtifactValidationError("finalized artifact remains writable")

    files: list[Path] = []
    for path in sorted(directory.rglob("*")):
        metadata = path.lstat()
        relative = path.relative_to(directory).as_posix()
        _require_safe_relative(relative)
        if stat.S_ISLNK(metadata.st_mode):
            raise ArtifactValidationError("artifact contains a symlink")
        if stat.S_ISDIR(metadata.st_mode):
            if require_immutable and metadata.st_mode & _WRITE_BITS:
                raise ArtifactValidationError("finalized artifact remains writable")
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ArtifactValidationError("artifact contains an unsafe file")
        if require_immutable and metadata.st_mode & _WRITE_BITS:
            raise ArtifactValidationError("finalized artifact remains writable")
        if _is_env_path(relative):
            raise ArtifactValidationError("artifact contains a prohibited secret file")
        files.append(path)
    return files


def _parse_checksum_ledger(path: Path) -> dict[str, str]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise ArtifactValidationError("checksum ledger is unreadable") from error
    if raw and not raw.endswith(b"\n"):
        raise ArtifactValidationError("checksum ledger is not canonical")
    entries: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or _SHA256_RE.fullmatch(parts[0]) is None:
            raise ArtifactValidationError("checksum ledger is malformed")
        relative = _require_safe_relative(parts[1])
        if relative in entries:
            raise ArtifactValidationError("checksum ledger contains duplicate paths")
        entries[relative] = parts[0]
    if not entries or list(entries) != sorted(entries):
        raise ArtifactValidationError("checksum ledger is not canonical")
    return entries


def _tree_root_sha256(files: Sequence[ArtifactFile]) -> str:
    canonical = "".join(
        f"{item.sha256}  {item.relative_path}\n"
        for item in sorted(files, key=lambda item: item.relative_path)
    ).encode("utf-8")
    return sha256_bytes(canonical)


def _manifest_identity(manifest: Mapping[str, object]) -> tuple[str, str | None]:
    if manifest.get("schema_version") == E00_CONTAINER_SCHEMA_VERSION:
        run = manifest.get("run")
        if not isinstance(run, Mapping):
            raise ArtifactValidationError("E00 manifest run identity is invalid")
        run_id = run.get("id")
        status = run.get("status")
    else:
        run_id = manifest.get("run_id")
        status = manifest.get("status")
    if not isinstance(run_id, str) or not run_id:
        raise ArtifactValidationError("manifest run identity is invalid")
    if status is not None and not isinstance(status, str):
        raise ArtifactValidationError("manifest status is invalid")
    return run_id, status


def _validate_inventory_structure(
    inventory: Mapping[str, object],
) -> list[Mapping[str, object]]:
    if (
        set(inventory) != INVENTORY_KEYS
        or inventory.get("schema_version") != INVENTORY_SCHEMA_VERSION
        or inventory.get("excluded_control_files")
        != list(INVENTORY_EXCLUDED_CONTROLS)
        or not isinstance(inventory.get("run_id"), str)
        or not inventory.get("run_id")
    ):
        raise ArtifactValidationError("artifact inventory is invalid")
    files = inventory.get("files")
    if not isinstance(files, list):
        raise ArtifactValidationError("artifact inventory is invalid")
    result: list[Mapping[str, object]] = []
    for item in files:
        if (
            not isinstance(item, Mapping)
            or set(item) != INVENTORY_ITEM_KEYS
            or not isinstance(item.get("role"), str)
            or not item["role"].strip()
        ):
            raise ArtifactValidationError("artifact inventory is invalid")
        result.append(item)
    return result


def _validate_e00_artifact(
    root: Path,
    manifest: dict[str, Any],
    inventory_files: Sequence[Mapping[str, object]],
) -> None:
    try:
        from preflight import run_preflight

        manifest_errors = run_preflight.validate_manifest(manifest)
    except (ImportError, OSError, TypeError, ValueError) as error:
        raise ArtifactValidationError(
            "E00 manifest validator is unavailable"
        ) from error
    if manifest_errors:
        raise ArtifactValidationError("E00 manifest validation failed")

    for item in inventory_files:
        relative = item["path"]
        expected_role = (
            "manifest"
            if relative == "manifest.json"
            else run_preflight.file_role(str(relative))
        )
        if item["role"] != expected_role:
            raise ArtifactValidationError("E00 inventory role is invalid")

    try:
        actual_evidence = run_preflight.enumerate_evidence_files(root)
        inventory_records = [
            item
            for item in actual_evidence
            if item.get("path") == "artifact_inventory.json"
        ]
        evidence = manifest.get("evidence")
        if (
            len(inventory_records) != 1
            or not isinstance(evidence, Mapping)
            or not isinstance(evidence.get("files"), list)
            or any(
                isinstance(item, Mapping)
                and item.get("path") == "artifact_inventory.json"
                for item in evidence["files"]
            )
        ):
            raise ArtifactValidationError(
                "E00 evidence inventory projection is invalid"
            )
        projected_manifest = dict(manifest)
        projected_evidence = dict(evidence)
        projected_evidence["files"] = sorted(
            [*evidence["files"], inventory_records[0]],
            key=lambda item: item["path"],
        )
        projected_manifest["evidence"] = projected_evidence
        reference_errors = run_preflight.evidence_reference_errors(
            root,
            projected_manifest,
        )
    except ArtifactValidationError:
        raise
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise ArtifactValidationError(
            "E00 evidence-reference validator is unavailable"
        ) from error
    if reference_errors:
        raise ArtifactValidationError(
            "E00 evidence-reference validation failed"
        )


def _validate_artifact(
    directory: str | Path,
    *,
    environ: Mapping[str, str] | None,
    require_immutable: bool,
    expect_final_name: bool,
) -> ValidatedArtifact:
    root = Path(directory).absolute()
    files = _collect_files(root, require_immutable=require_immutable)
    relatives = {path.relative_to(root).as_posix() for path in files}
    missing = sorted(set(CONTROL_FILES) - relatives)
    if missing:
        raise ArtifactValidationError("artifact lacks required finalization controls")

    manifest = _load_json_object(root / "manifest.json", "manifest")
    inventory = _load_json_object(
        root / "artifact_inventory.json", "artifact inventory"
    )
    completion = _load_json_object(root / "COMPLETE", "completion marker")
    for payload in (manifest, inventory, completion):
        _reject_credential_keys(payload)
    manifest_run_id, manifest_status = _manifest_identity(manifest)
    inventory_files = _validate_inventory_structure(inventory)
    if (
        inventory.get("run_id") != manifest_run_id
        or completion.get("run_id") != manifest_run_id
    ):
        raise ArtifactValidationError("artifact run identities do not agree")
    completion_status = completion.get("status")
    if completion_status is not None and not isinstance(completion_status, str):
        raise ArtifactValidationError("completion marker status is invalid")
    if manifest_status is not None and completion_status != manifest_status:
        raise ArtifactValidationError(
            "manifest and completion marker statuses do not agree"
        )

    source = os.environ if environ is None else environ
    secret_values = tuple(
        value.encode("utf-8")
        for name in SECRET_VARIABLES
        if (value := source.get(name))
    )
    for path in files:
        if any(_contains_bytes(path, value) for value in secret_values):
            raise ArtifactValidationError(
                "artifact contains prohibited credential material"
            )

    ledger = _parse_checksum_ledger(root / "checksums.sha256")
    ledger_expected = relatives - {"checksums.sha256", "COMPLETE"}
    if set(ledger) != ledger_expected:
        raise ArtifactValidationError(
            "checksum ledger does not exactly cover artifact files"
        )
    for relative, expected in ledger.items():
        if sha256_file(root / relative) != expected:
            raise ArtifactValidationError("artifact checksum verification failed")

    declared: dict[str, tuple[str, int]] = {}
    for item in inventory_files:
        relative_value = item.get("path")
        digest = item.get("sha256")
        size = item.get("size_bytes")
        if (
            not isinstance(relative_value, str)
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            raise ArtifactValidationError("artifact inventory is invalid")
        relative = _require_safe_relative(relative_value)
        if relative in declared:
            raise ArtifactValidationError("artifact inventory has duplicate paths")
        declared[relative] = (digest, size)
    inventory_expected = relatives - {
        "artifact_inventory.json",
        "checksums.sha256",
        "COMPLETE",
    }
    if set(declared) != inventory_expected or list(declared) != sorted(declared):
        raise ArtifactValidationError(
            "artifact inventory does not exactly cover payload files"
        )
    for relative, (digest, size) in declared.items():
        target = root / relative
        if target.stat().st_size != size or sha256_file(target) != digest:
            raise ArtifactValidationError("artifact inventory verification failed")

    if completion.get("written_last") is not True:
        raise ArtifactValidationError("COMPLETE is not a valid final marker")
    expected_controls = {
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "artifact_inventory_sha256": sha256_file(
            root / "artifact_inventory.json"
        ),
        "checksum_ledger_sha256": sha256_file(root / "checksums.sha256"),
    }
    if any(completion.get(key) != value for key, value in expected_controls.items()):
        raise ArtifactValidationError("COMPLETE control hashes do not match")
    if completion.get("checksum_ledger_path", "checksums.sha256") != "checksums.sha256":
        raise ArtifactValidationError("COMPLETE references an unexpected ledger")

    if manifest.get("schema_version") == E00_CONTAINER_SCHEMA_VERSION:
        _validate_e00_artifact(root, manifest, inventory_files)
    elif manifest.get("schema_version") == PHASE6A_PARITY_SCHEMA_VERSION:
        from scripts.phase6a_bf16_parity import (
            Phase6AParityError,
            validate_finalized_parity_artifact,
        )

        try:
            validate_finalized_parity_artifact(
                root, manifest, inventory_files, relatives
            )
        except (
            Phase6AParityError,
            IndexError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise ArtifactValidationError(
                "Phase 6A parity artifact is invalid"
            ) from error

    if "manifest.initial.json" in relatives and require_immutable:
        try:
            from kvbench.runtime.artifacts import validate_run_directory

            repository_validation = validate_run_directory(
                root,
                expect_final_name=expect_final_name,
            )
        except (ImportError, OSError) as error:
            raise ArtifactValidationError(
                "repository artifact validator is unavailable"
            ) from error
        if not repository_validation.valid or not repository_validation.complete:
            raise ArtifactValidationError(
                "artifact failed repository lifecycle validation"
            )

    records = tuple(
        ArtifactFile(
            relative_path=path.relative_to(root).as_posix(),
            path=path,
            size_bytes=path.stat().st_size,
            sha256=sha256_file(path),
        )
        for path in files
    )
    return ValidatedArtifact(
        directory=root.resolve(strict=True),
        root_sha256=_tree_root_sha256(records),
        files=records,
    )


def validate_local_artifact(
    directory: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> ValidatedArtifact:
    """Validate a finalized immutable artifact before any network operation."""

    return _validate_artifact(
        directory,
        environ=environ,
        require_immutable=True,
        expect_final_name=True,
    )


def publication_order(artifact: ValidatedArtifact) -> tuple[ArtifactFile, ...]:
    by_path = artifact.by_path()
    payload = sorted(
        (
            item
            for item in artifact.files
            if item.relative_path not in CONTROL_FILES
        ),
        key=lambda item: item.relative_path,
    )
    ordered = (
        *payload,
        by_path["manifest.json"],
        by_path["artifact_inventory.json"],
        by_path["checksums.sha256"],
        by_path["COMPLETE"],
    )
    if len({item.relative_path for item in ordered}) != len(artifact.files):
        raise ArtifactValidationError("publication order does not cover artifact")
    return ordered


def _content_type(path: str) -> str | None:
    if path.endswith(".json"):
        return "application/json"
    if path.endswith((".txt", ".sha256")) or path == "COMPLETE":
        return "text/plain"
    guessed, _ = mimetypes.guess_type(path, strict=True)
    return guessed


@dataclass(frozen=True)
class PublicationResult:
    root_sha256: str
    uri: str
    ordered_paths: tuple[str, ...]
    uploaded: tuple[str, ...]
    verified_existing: tuple[str, ...]
    published_at_utc: str

    def to_dict(self) -> dict[str, object]:
        order_bytes = "".join(f"{item}\n" for item in self.ordered_paths).encode()
        return {
            "provider": "cloudflare_r2",
            "root_sha256": self.root_sha256,
            "uri": self.uri,
            "object_count": len(self.ordered_paths),
            "uploaded_count": len(self.uploaded),
            "verified_existing_count": len(self.verified_existing),
            "complete_last": bool(
                self.ordered_paths and self.ordered_paths[-1] == "COMPLETE"
            ),
            "publication_order_sha256": sha256_bytes(order_bytes),
            "published_at_utc": self.published_at_utc,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def publish_artifact(
    client: "R2S3Client",
    config: R2Config,
    artifact: ValidatedArtifact,
) -> PublicationResult:
    """Conditionally create every object, publishing COMPLETE last."""

    uploaded: list[str] = []
    verified_existing: list[str] = []
    ordered = publication_order(artifact)
    for item in ordered:
        multipart = item.size_bytes > SINGLE_PUT_MAX_BYTES
        data: bytes | None = None
        if multipart:
            if sha256_file(item.path) != item.sha256:
                raise ArtifactValidationError(
                    "local artifact changed after final validation"
                )
        else:
            data = item.path.read_bytes()
            if sha256_bytes(data) != item.sha256:
                raise ArtifactValidationError(
                    "local artifact changed after final validation"
                )
        key = artifact_object_key(
            config.prefix,
            artifact.root_sha256,
            item.relative_path,
        )
        content_type = _content_type(item.relative_path)
        existing = _remote_object_identity(client, key)
        if existing is not None:
            if existing != (item.sha256, item.size_bytes):
                raise ObjectConflictError(
                    "existing content-addressed object has different bytes"
                )
            verified_existing.append(item.relative_path)
            continue
        try:
            if multipart:
                client.put_file_if_absent(
                    key,
                    item.path,
                    expected_size=item.size_bytes,
                    expected_sha256=item.sha256,
                    content_type=content_type,
                )
            else:
                if data is None:
                    raise ArtifactValidationError(
                        "single-part object data is absent"
                    )
                client.put_object_if_absent(
                    key,
                    data,
                    content_type=content_type,
                )
        except RemoteRequestError as error:
            if (
                error.status != 412
                and error.code != "ObjectLockedByBucketPolicy"
            ):
                raise
            raced = _remote_object_identity(client, key)
            if raced != (item.sha256, item.size_bytes):
                raise ObjectConflictError(
                    "conditional object creation did not preserve exact bytes"
                ) from error
            verified_existing.append(item.relative_path)
            continue
        retrieved = _remote_object_identity(client, key)
        if retrieved != (item.sha256, item.size_bytes):
            raise R2ArtifactError(
                "uploaded object failed authoritative SHA-256 check"
            )
        uploaded.append(item.relative_path)

    return PublicationResult(
        root_sha256=artifact.root_sha256,
        uri=(
            f"r2://{config.bucket}/"
            f"{artifact_object_prefix(config.prefix, artifact.root_sha256)}"
        ),
        ordered_paths=tuple(item.relative_path for item in ordered),
        uploaded=tuple(uploaded),
        verified_existing=tuple(verified_existing),
        published_at_utc=_utc_now(),
    )


def _remote_object_identity(
    client: object,
    key: str,
) -> tuple[str, int] | None:
    streaming = getattr(client, "get_object_sha256_or_none", None)
    if callable(streaming):
        identity = streaming(key)
        if identity is None:
            return None
        digest, size = identity
        return str(digest), int(size)
    materialized = getattr(client, "get_object_or_none")(key)
    if materialized is None:
        return None
    return sha256_bytes(materialized), len(materialized)


def _ensure_empty_directory(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ArtifactValidationError("retrieval destination is unsafe")
        if any(path.iterdir()):
            raise ArtifactValidationError("retrieval destination must be empty")
    else:
        path.mkdir(mode=0o700, parents=False)
    return path.resolve(strict=True)


def _create_downloaded_file(root: Path, relative: str) -> int:
    safe = PurePosixPath(_require_safe_relative(relative))
    current = root
    for part in safe.parent.parts:
        current = current / part
        try:
            current.mkdir(mode=0o700)
        except FileExistsError:
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise ArtifactValidationError("retrieval path is unsafe") from None
    target = root.joinpath(*safe.parts)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(target, flags, 0o600)
    except FileExistsError as error:
        raise ArtifactValidationError("retrieval produced a duplicate path") from error
    return descriptor


def _write_downloaded_file(root: Path, relative: str, data: bytes) -> None:
    descriptor = _create_downloaded_file(root, relative)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(descriptor, view) :]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _make_tree_immutable(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    root.chmod(0o555)


@dataclass(frozen=True)
class RetrievalResult:
    root_sha256: str
    uri: str
    object_count: int
    retrieved_at_utc: str

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": "cloudflare_r2",
            "root_sha256": self.root_sha256,
            "uri": self.uri,
            "object_count": self.object_count,
            "complete_marker_valid": True,
            "inventory_valid": True,
            "checksum_ledger_valid": True,
            "unexpected_objects": False,
            "verification_result": "PASS",
            "retrieved_at_utc": self.retrieved_at_utc,
        }


def verify_remote_artifact(
    client: "R2S3Client",
    config: R2Config,
    root_sha256: str,
    destination: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> RetrievalResult:
    """Download a remote artifact into a clean directory and fully revalidate it."""

    digest = validate_root_sha256(root_sha256)
    remote_prefix = artifact_object_prefix(config.prefix, digest)
    keys = client.list_keys(remote_prefix)
    if len(keys) != len(set(keys)):
        raise ArtifactValidationError("remote listing contains duplicate objects")
    relatives: list[str] = []
    for key in keys:
        if not key.startswith(remote_prefix):
            raise ArtifactValidationError("remote object escaped the configured prefix")
        relative = key[len(remote_prefix) :]
        relatives.append(_require_safe_relative(relative))
    if "COMPLETE" not in relatives:
        raise ArtifactValidationError("remote artifact prefix is incomplete")

    target_root = _ensure_empty_directory(Path(destination).absolute())
    for relative in sorted(relatives, key=lambda item: (item == "COMPLETE", item)):
        key = artifact_object_key(config.prefix, digest, relative)
        streaming = getattr(client, "download_object_to_descriptor", None)
        if callable(streaming):
            descriptor = _create_downloaded_file(target_root, relative)
            try:
                identity = streaming(key, descriptor)
            finally:
                os.close(descriptor)
            if identity is None:
                raise ArtifactValidationError(
                    "remote listing changed during retrieval"
                )
        else:
            data = client.get_object_or_none(key)
            if data is None:
                raise ArtifactValidationError(
                    "remote listing changed during retrieval"
                )
            _write_downloaded_file(target_root, relative, data)

    reconstructed = _validate_artifact(
        target_root,
        environ=environ,
        require_immutable=False,
        expect_final_name=False,
    )
    if reconstructed.root_sha256 != digest:
        raise ArtifactValidationError("reconstructed root digest does not match prefix")
    if set(relatives) != {item.relative_path for item in reconstructed.files}:
        raise ArtifactValidationError("remote prefix contains unexpected objects")
    _make_tree_immutable(target_root)
    immutable = _validate_artifact(
        target_root,
        environ=environ,
        require_immutable=True,
        expect_final_name=False,
    )
    if immutable.root_sha256 != digest:
        raise ArtifactValidationError("immutable retrieval digest changed")
    return RetrievalResult(
        root_sha256=digest,
        uri=f"r2://{config.bucket}/{remote_prefix}",
        object_count=len(relatives),
        retrieved_at_utc=_utc_now(),
    )


def _xml_error_code(data: bytes) -> str:
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return "Unspecified"
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] == "Code" and element.text:
            candidate = element.text[:80]
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,79}", candidate):
                return candidate
            return "Unspecified"
    return "Unspecified"


class R2S3Client:
    """Small SigV4 client for the exact R2 object operations Phase 6A needs."""

    def __init__(
        self,
        config: R2Config,
        *,
        opener: Callable[..., Any] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._config = config
        self._opener = urllib.request.urlopen if opener is None else opener
        self._clock = (
            (lambda: datetime.now(timezone.utc)) if clock is None else clock
        )
        parsed = urllib.parse.urlsplit(config.endpoint)
        self._host = parsed.netloc

    @staticmethod
    def _canonical_query(parameters: Sequence[tuple[str, str]]) -> str:
        encoded = [
            (
                urllib.parse.quote(key, safe="-_.~"),
                urllib.parse.quote(value, safe="-_.~"),
            )
            for key, value in parameters
        ]
        return "&".join(f"{key}={value}" for key, value in sorted(encoded))

    def _object_path(self, key: str | None = None) -> str:
        raw = f"/{self._config.bucket}"
        if key is not None:
            raw += f"/{key}"
        return urllib.parse.quote(raw, safe="/-_.~")

    def _authorization_headers(
        self,
        *,
        method: str,
        canonical_path: str,
        canonical_query: str,
        body: bytes,
        extra_headers: Mapping[str, str],
    ) -> dict[str, str]:
        now = self._clock().astimezone(timezone.utc)
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        short_date = now.strftime("%Y%m%d")
        payload_hash = sha256_bytes(body)
        headers = {
            "host": self._host,
            "x-amz-content-sha256": payload_hash,
            "x-amz-date": amz_date,
            **{key.lower(): " ".join(value.split()) for key, value in extra_headers.items()},
        }
        signed_header_names = sorted(headers)
        canonical_headers = "".join(
            f"{name}:{headers[name]}\n" for name in signed_header_names
        )
        signed_headers = ";".join(signed_header_names)
        canonical_request = "\n".join(
            (
                method,
                canonical_path,
                canonical_query,
                canonical_headers,
                signed_headers,
                payload_hash,
            )
        )
        scope = f"{short_date}/{REGION}/{SERVICE}/aws4_request"
        string_to_sign = "\n".join(
            (
                "AWS4-HMAC-SHA256",
                amz_date,
                scope,
                sha256_bytes(canonical_request.encode("utf-8")),
            )
        )
        key_date = hmac.new(
            f"AWS4{self._config.secret_access_key}".encode(),
            short_date.encode(),
            hashlib.sha256,
        ).digest()
        key_region = hmac.new(key_date, REGION.encode(), hashlib.sha256).digest()
        key_service = hmac.new(key_region, SERVICE.encode(), hashlib.sha256).digest()
        signing_key = hmac.new(
            key_service, b"aws4_request", hashlib.sha256
        ).digest()
        signature = hmac.new(
            signing_key, string_to_sign.encode(), hashlib.sha256
        ).hexdigest()
        authorization = (
            "AWS4-HMAC-SHA256 "
            f"Credential={self._config.access_key_id}/{scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )
        output = {name: value for name, value in headers.items() if name != "host"}
        output["Authorization"] = authorization
        return output

    def _request_with_metadata(
        self,
        method: str,
        *,
        key: str | None = None,
        query: Sequence[tuple[str, str]] = (),
        body: bytes = b"",
        headers: Mapping[str, str] | None = None,
    ) -> tuple[bytes, dict[str, str], int]:
        canonical_path = self._object_path(key)
        canonical_query = self._canonical_query(query)
        extra = {} if headers is None else dict(headers)
        signed = self._authorization_headers(
            method=method,
            canonical_path=canonical_path,
            canonical_query=canonical_query,
            body=body,
            extra_headers=extra,
        )
        url = f"{self._config.endpoint}{canonical_path}"
        if canonical_query:
            url += f"?{canonical_query}"
        request = urllib.request.Request(
            url,
            data=body if method in {"POST", "PUT"} else None,
            headers=signed,
            method=method,
        )
        try:
            with self._opener(request, timeout=120) as response:
                response_body = response.read()
                raw_headers = getattr(response, "headers", {})
                response_headers = {
                    str(name).lower(): str(value)
                    for name, value in raw_headers.items()
                }
                raw_status = getattr(response, "status", None)
                if not isinstance(raw_status, int):
                    getcode = getattr(response, "getcode", None)
                    raw_status = getcode() if callable(getcode) else 200
                if not isinstance(raw_status, int):
                    raise RemoteRequestError(
                        status=None,
                        code="InvalidHTTPStatus",
                    )
                return response_body, response_headers, raw_status
        except urllib.error.HTTPError as error:
            try:
                response_body = error.read(65536)
            except OSError:
                response_body = b""
            code = _xml_error_code(response_body)
            if any(
                secret and secret in code
                for secret in (
                    self._config.account_id,
                    self._config.access_key_id,
                    self._config.secret_access_key,
                    self._config.cloudflare_api_token,
                )
            ):
                code = "Unspecified"
            raise RemoteRequestError(
                status=error.code,
                code=code,
            ) from None
        except (
            urllib.error.URLError,
            http.client.IncompleteRead,
            TimeoutError,
            OSError,
        ):
            raise RemoteRequestError(status=None, code="TransportError") from None

    def _request(
        self,
        method: str,
        *,
        key: str | None = None,
        query: Sequence[tuple[str, str]] = (),
        body: bytes = b"",
        headers: Mapping[str, str] | None = None,
    ) -> bytes:
        response_body, _, _ = self._request_with_metadata(
            method,
            key=key,
            query=query,
            body=body,
            headers=headers,
        )
        return response_body

    def get_object_or_none(self, key: str) -> bytes | None:
        for attempt in range(1, OBJECT_READ_ATTEMPTS + 1):
            try:
                return self._request("GET", key=key)
            except RemoteRequestError as error:
                if error.status == 404:
                    return None
                retryable = (
                    error.status is None and error.code == "TransportError"
                )
                if not retryable or attempt == OBJECT_READ_ATTEMPTS:
                    raise
        raise AssertionError("object read retry loop exhausted")

    def _object_size_or_none(self, key: str) -> int | None:
        for attempt in range(1, OBJECT_READ_ATTEMPTS + 1):
            try:
                _, headers, _ = self._request_with_metadata("HEAD", key=key)
            except RemoteRequestError as error:
                if error.status == 404:
                    return None
                retryable = (
                    error.status is None and error.code == "TransportError"
                )
                if not retryable or attempt == OBJECT_READ_ATTEMPTS:
                    raise
                continue
            raw_size = headers.get("content-length")
            try:
                size = int(raw_size) if raw_size is not None else -1
            except ValueError as error:
                raise RemoteRequestError(
                    status=200,
                    code="InvalidContentLength",
                ) from error
            if size < 0:
                raise RemoteRequestError(
                    status=200,
                    code="InvalidContentLength",
                )
            return size
        raise AssertionError("object HEAD retry loop exhausted")

    def _get_object_range(self, key: str, start: int, end: int) -> bytes:
        expected = end - start + 1
        for attempt in range(1, OBJECT_READ_ATTEMPTS + 1):
            try:
                data, _, status = self._request_with_metadata(
                    "GET",
                    key=key,
                    headers={"Range": f"bytes={start}-{end}"},
                )
            except RemoteRequestError as error:
                retryable = (
                    error.status is None and error.code == "TransportError"
                ) or error.status in RETRYABLE_MULTIPART_STATUSES
                if not retryable or attempt == OBJECT_READ_ATTEMPTS:
                    raise
                continue
            if status != 206 or len(data) != expected:
                raise RemoteRequestError(
                    status=status,
                    code="InvalidRangeResponse",
                )
            return data
        raise AssertionError("object range retry loop exhausted")

    def _consume_object_ranges(
        self,
        key: str,
        size: int,
        consume: Callable[[bytes], None],
    ) -> None:
        if size == 0:
            return
        with ThreadPoolExecutor(max_workers=R2_READ_WORKERS) as executor:
            start = 0
            while start < size:
                ranges: list[tuple[int, int]] = []
                for _ in range(R2_READ_WORKERS):
                    if start >= size:
                        break
                    end = min(size - 1, start + OBJECT_RANGE_SIZE_BYTES - 1)
                    ranges.append((start, end))
                    start = end + 1
                futures = [
                    executor.submit(self._get_object_range, key, begin, end)
                    for begin, end in ranges
                ]
                for future in futures:
                    consume(future.result())

    def get_object_sha256_or_none(
        self,
        key: str,
    ) -> tuple[str, int] | None:
        size = self._object_size_or_none(key)
        if size is None:
            return None
        digest = hashlib.sha256()
        self._consume_object_ranges(key, size, digest.update)
        return digest.hexdigest(), size

    def download_object_to_descriptor(
        self,
        key: str,
        descriptor: int,
    ) -> tuple[str, int] | None:
        size = self._object_size_or_none(key)
        if size is None:
            return None
        digest = hashlib.sha256()
        written = 0

        def consume(data: bytes) -> None:
            nonlocal written
            digest.update(data)
            view = memoryview(data)
            while view:
                count = os.write(descriptor, view)
                if count <= 0:
                    raise OSError("retrieval write made no progress")
                written += count
                view = view[count:]

        self._consume_object_ranges(key, size, consume)
        if written != size:
            raise RemoteRequestError(
                status=200,
                code="InvalidRetrievedSize",
            )
        os.fsync(descriptor)
        return digest.hexdigest(), size

    def put_object_if_absent(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str | None,
    ) -> None:
        headers = {"If-None-Match": "*"}
        if content_type is not None:
            headers["Content-Type"] = content_type
        self._request("PUT", key=key, body=data, headers=headers)

    @staticmethod
    def _multipart_upload_id(data: bytes) -> str:
        try:
            root = ET.fromstring(data)
        except ET.ParseError as error:
            raise RemoteRequestError(
                status=200,
                code="MalformedMultipartResponse",
            ) from error
        for element in root.iter():
            if element.tag.rsplit("}", 1)[-1] == "UploadId" and element.text:
                upload_id = element.text
                if (
                    len(upload_id) <= 2048
                    and all(ord(character) >= 32 for character in upload_id)
                ):
                    return upload_id
        raise RemoteRequestError(
            status=200,
            code="MalformedMultipartResponse",
        )

    def _abort_multipart_upload(self, key: str, upload_id: str) -> None:
        self._request(
            "DELETE",
            key=key,
            query=(("uploadId", upload_id),),
        )

    def _upload_multipart_part(
        self,
        *,
        key: str,
        upload_id: str,
        part_number: int,
        data: bytes,
    ) -> str:
        for attempt in range(MULTIPART_UPLOAD_ATTEMPTS):
            try:
                _, response_headers, _ = self._request_with_metadata(
                    "PUT",
                    key=key,
                    query=(
                        ("partNumber", str(part_number)),
                        ("uploadId", upload_id),
                    ),
                    body=data,
                )
            except RemoteRequestError as error:
                retryable = (
                    error.status is None
                    or error.status in RETRYABLE_MULTIPART_STATUSES
                )
                if (
                    not retryable
                    or attempt + 1 >= MULTIPART_UPLOAD_ATTEMPTS
                ):
                    raise
                continue
            etag = response_headers.get("etag", "")
            if re.fullmatch(
                r'(?:[0-9a-fA-F]{32}|"[0-9a-fA-F]{32}")',
                etag,
            ) is None:
                raise RemoteRequestError(
                    status=200,
                    code="InvalidMultipartETag",
                )
            return etag
        raise RemoteRequestError(
            status=None,
            code="MultipartRetryExhausted",
        )

    def put_file_if_absent(
        self,
        key: str,
        path: Path,
        *,
        expected_size: int,
        expected_sha256: str,
        content_type: str | None,
    ) -> None:
        headers = {"If-None-Match": "*"}
        if content_type is not None:
            headers["Content-Type"] = content_type
        create_body, _, _ = self._request_with_metadata(
            "POST",
            key=key,
            query=(("uploads", ""),),
            headers=headers,
        )
        upload_id = self._multipart_upload_id(create_body)
        parts: list[tuple[int, str]] = []
        digest = hashlib.sha256()
        total_size = 0
        try:
            with (
                path.open("rb") as source,
                ThreadPoolExecutor(max_workers=R2_UPLOAD_WORKERS) as executor,
            ):
                part_number = 1
                while True:
                    batch: list[tuple[int, bytes]] = []
                    for _ in range(R2_UPLOAD_WORKERS):
                        data = source.read(MULTIPART_PART_SIZE_BYTES)
                        if not data:
                            break
                        if part_number > MAX_MULTIPART_PARTS:
                            raise ArtifactValidationError(
                                "multipart object exceeds the part-count limit"
                            )
                        digest.update(data)
                        total_size += len(data)
                        batch.append((part_number, data))
                        part_number += 1
                    if not batch:
                        break
                    futures = [
                        (
                            number,
                            executor.submit(
                                self._upload_multipart_part,
                                key=key,
                                upload_id=upload_id,
                                part_number=number,
                                data=data,
                            ),
                        )
                        for number, data in batch
                    ]
                    parts.extend(
                        (number, future.result())
                        for number, future in futures
                    )
            if (
                not parts
                or total_size != expected_size
                or digest.hexdigest() != expected_sha256
            ):
                raise ArtifactValidationError(
                    "local artifact changed during multipart upload"
                )
            completion = ET.Element("CompleteMultipartUpload")
            for part_number, etag in parts:
                part = ET.SubElement(completion, "Part")
                ET.SubElement(part, "PartNumber").text = str(part_number)
                ET.SubElement(part, "ETag").text = etag
            complete_body = ET.tostring(
                completion,
                encoding="utf-8",
                xml_declaration=True,
            )
            response_body, _, _ = self._request_with_metadata(
                "POST",
                key=key,
                query=(("uploadId", upload_id),),
                body=complete_body,
                headers={"Content-Type": "application/xml"},
            )
            try:
                response_root = ET.fromstring(response_body)
            except ET.ParseError as error:
                raise RemoteRequestError(
                    status=200,
                    code="MalformedMultipartResponse",
                ) from error
            if (
                response_root.tag.rsplit("}", 1)[-1]
                != "CompleteMultipartUploadResult"
            ):
                raise RemoteRequestError(
                    status=200,
                    code="MultipartCompletionFailed",
                )
        except Exception:
            try:
                self._abort_multipart_upload(key, upload_id)
            except (OSError, R2ArtifactError):
                pass
            raise

    def list_keys(self, prefix: str) -> list[str]:
        keys: list[str] = []
        continuation: str | None = None
        seen_tokens: set[str] = set()
        while True:
            query = [("list-type", "2"), ("prefix", prefix)]
            if continuation is not None:
                query.append(("continuation-token", continuation))
            raw = self._request("GET", query=query)
            try:
                root = ET.fromstring(raw)
            except ET.ParseError as error:
                raise RemoteRequestError(
                    status=200, code="MalformedListResponse"
                ) from error
            truncated = False
            next_token: str | None = None
            for element in root.iter():
                name = element.tag.rsplit("}", 1)[-1]
                if name == "Contents":
                    for child in element:
                        if child.tag.rsplit("}", 1)[-1] == "Key" and child.text:
                            keys.append(child.text)
                elif name == "IsTruncated":
                    truncated = (element.text or "").lower() == "true"
                elif name == "NextContinuationToken":
                    next_token = element.text
            if not truncated:
                return keys
            if not next_token or next_token in seen_tokens:
                raise RemoteRequestError(
                    status=200, code="InvalidContinuationToken"
                )
            seen_tokens.add(next_token)
            continuation = next_token


def _cloudflare_error_details(
    payload: object,
    *,
    environ: Mapping[str, str],
) -> tuple[str, str | None]:
    if not isinstance(payload, Mapping):
        return "CloudflareAPIError", None
    errors = payload.get("errors")
    if not isinstance(errors, list):
        return "CloudflareAPIError", None
    for item in errors:
        if not isinstance(item, Mapping):
            continue
        raw_code = item.get("code")
        if isinstance(raw_code, int) and not isinstance(raw_code, bool):
            code = str(raw_code)
        elif isinstance(raw_code, str) and re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", raw_code
        ):
            code = raw_code
        else:
            code = "CloudflareAPIError"
        raw_message = item.get("message")
        if not isinstance(raw_message, str):
            return code, None
        normalized = " ".join(raw_message.split())
        message = redact_text(normalized, environ)[:512]
        return code, message if message else None
    return "CloudflareAPIError", None


class CloudflareReadClient:
    """Read-only Cloudflare management API client for bucket certification."""

    def __init__(
        self,
        api_token: str,
        *,
        opener: Callable[..., Any] | None = None,
        redaction_environment: Mapping[str, str] | None = None,
    ) -> None:
        self._api_token = api_token
        self._opener = urllib.request.urlopen if opener is None else opener
        source = (
            os.environ
            if redaction_environment is None
            else redaction_environment
        )
        self._redaction_environment = {
            name: value
            for name in SECRET_VARIABLES
            if (value := source.get(name))
        }
        self._redaction_environment["CLOUDFLARE_API_TOKEN"] = api_token

    def get_json(
        self,
        path: str,
        *,
        check: str = "cloudflare_management",
    ) -> dict[str, Any]:
        if not path.startswith("/") or ".." in PurePosixPath(path).parts:
            raise R2ArtifactError("Cloudflare API path is unsafe")
        if re.fullmatch(r"[a-z0-9_]{1,64}", check) is None:
            raise R2ArtifactError("Cloudflare API check name is unsafe")
        safe_path = redact_text(path, self._redaction_environment)
        request = urllib.request.Request(
            f"https://api.cloudflare.com/client/v4{path}",
            headers={"Authorization": f"Bearer {self._api_token}"},
            method="GET",
        )
        try:
            with self._opener(request, timeout=60) as response:
                raw = response.read(4 * 1024 * 1024 + 1)
        except urllib.error.HTTPError as error:
            try:
                response_body = error.read(65536)
                error_payload: object = json.loads(response_body)
            except (OSError, UnicodeError, json.JSONDecodeError):
                error_payload = None
            provider_code, provider_message = _cloudflare_error_details(
                error_payload,
                environ=self._redaction_environment,
            )
            raise RemoteRequestError(
                status=error.code,
                code=provider_code,
                failed_check=check,
                failed_path=safe_path,
                provider_message=provider_message,
            ) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise RemoteRequestError(
                status=None,
                code="TransportError",
                failed_check=check,
                failed_path=safe_path,
            ) from None
        if len(raw) > 4 * 1024 * 1024:
            raise RemoteRequestError(
                status=200,
                code="OversizedAPIResponse",
                failed_check=check,
                failed_path=safe_path,
            )
        try:
            payload = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise RemoteRequestError(
                status=200,
                code="MalformedAPIResponse",
                failed_check=check,
                failed_path=safe_path,
            ) from error
        if not isinstance(payload, dict):
            raise RemoteRequestError(
                status=200,
                code="MalformedAPIResponse",
                failed_check=check,
                failed_path=safe_path,
            )
        if payload.get("success") is not True:
            provider_code, provider_message = _cloudflare_error_details(
                payload,
                environ=self._redaction_environment,
            )
            raise RemoteRequestError(
                status=200,
                code=provider_code,
                failed_check=check,
                failed_path=safe_path,
                provider_message=provider_message,
            )
        if not isinstance(payload.get("result"), dict):
            raise RemoteRequestError(
                status=200,
                code="MalformedAPIResponse",
                failed_check=check,
                failed_path=safe_path,
            )
        return payload


@dataclass(frozen=True)
class BucketLockEvidence:
    bucket: str
    endpoint_identity: str
    endpoint_class: str
    bucket_exists: bool
    managed_r2_dev_enabled: bool
    custom_domain_count: int
    enabled_custom_domain_count: int
    public_state_result: str
    rule_id: str
    rule_name: str | None
    covered_prefix: str
    rule_prefix: str
    scope_kind: str
    retention_type: str
    verified_at_utc: str

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": "cloudflare_r2",
            "bucket": self.bucket,
            "endpoint": self.endpoint_identity,
            "endpoint_class": self.endpoint_class,
            "bucket_exists": self.bucket_exists,
            "bucket_public": False,
            "public_state_result": self.public_state_result,
            "managed_r2_dev_enabled": self.managed_r2_dev_enabled,
            "public_r2_dev": self.managed_r2_dev_enabled,
            "custom_domain_count": self.custom_domain_count,
            "enabled_custom_domain_count": self.enabled_custom_domain_count,
            "public_custom_domain": self.enabled_custom_domain_count > 0,
            "lock_rule_id": self.rule_id,
            "lock_rule_name": self.rule_name,
            "covered_prefix": self.covered_prefix,
            "lock_prefix": self.rule_prefix,
            "lock_scope": self.scope_kind,
            "enabled": True,
            "retention_type": self.retention_type,
            "retention_condition": self.retention_type,
            "verification_result": "PASS",
            "verified_at_utc": self.verified_at_utc,
        }


def _cloudflare_result(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if payload.get("success") is not True or not isinstance(
        payload.get("result"), Mapping
    ):
        raise R2ArtifactError("Cloudflare API response failed validation")
    return payload["result"]


def verify_cloudflare_bucket_lock(
    config: R2Config,
    client: CloudflareReadClient | None = None,
) -> BucketLockEvidence:
    """Read and verify bucket existence, direct public state, and Bucket Lock."""

    reader = (
        CloudflareReadClient(
            config.cloudflare_api_token,
            redaction_environment={
                "R2_ACCOUNT_ID": config.account_id,
                "CLOUDFLARE_API_TOKEN": config.cloudflare_api_token,
            },
        )
        if client is None
        else client
    )
    account = urllib.parse.quote(config.account_id, safe="")
    bucket = urllib.parse.quote(config.bucket, safe="")
    base = f"/accounts/{account}/r2/buckets/{bucket}"

    bucket_result = _cloudflare_result(
        reader.get_json(base, check="bucket_exists")
    )
    if bucket_result.get("name") != config.bucket:
        raise R2ArtifactError("configured R2 bucket does not exist")

    managed = _cloudflare_result(
        reader.get_json(
            f"{base}/domains/managed",
            check="managed_r2_dev_public_access",
        )
    )
    custom = _cloudflare_result(
        reader.get_json(
            f"{base}/domains/custom",
            check="custom_domain_public_access",
        )
    )
    domains = custom.get("domains")
    if not isinstance(managed.get("enabled"), bool) or not isinstance(domains, list):
        raise R2ArtifactError("R2 public-access response failed validation")
    if any(
        not isinstance(item, Mapping) or not isinstance(item.get("enabled"), bool)
        for item in domains
    ):
        raise R2ArtifactError("R2 public-access response failed validation")
    enabled_custom_domain_count = sum(
        item.get("enabled") is True
        for item in domains
        if isinstance(item, Mapping)
    )
    if managed["enabled"] or enabled_custom_domain_count:
        raise R2ArtifactError("R2 bucket direct public access is enabled")

    lock_result = _cloudflare_result(
        reader.get_json(f"{base}/lock", check="bucket_lock")
    )
    rules = lock_result.get("rules")
    if not isinstance(rules, list):
        raise R2ArtifactError("R2 Bucket Lock response failed validation")
    expected_lock_prefix = f"{config.prefix}/"
    candidates: list[tuple[str, str | None, str]] = []
    for rule in rules:
        if not isinstance(rule, Mapping) or rule.get("enabled") is not True:
            continue
        rule_id = rule.get("id")
        rule_name = rule.get("name")
        condition = rule.get("condition")
        raw_prefix = rule.get("prefix", "")
        if (
            not isinstance(rule_id, str)
            or not rule_id
            or (
                rule_name is not None
                and (
                    not isinstance(rule_name, str)
                    or not rule_name
                    or rule_name != rule_name.strip()
                )
            )
            or not isinstance(condition, Mapping)
            or condition.get("type") != "Indefinite"
            or not isinstance(raw_prefix, str)
        ):
            continue
        try:
            normalized_rule_prefix = normalize_prefix(raw_prefix)
        except R2ArtifactError:
            continue
        if normalized_rule_prefix == config.prefix:
            candidates.append((rule_id, rule_name, raw_prefix))
    if not candidates:
        raise R2ArtifactError(
            "no enabled indefinite Bucket Lock rule exactly matches the evidence prefix"
        )
    rule_id, rule_name, rule_prefix = sorted(
        candidates,
        key=lambda item: (item[0], item[1] or "", item[2]),
    )[0]
    return BucketLockEvidence(
        bucket=config.bucket,
        endpoint_identity=config.endpoint_identity(),
        endpoint_class=ENDPOINT_CLASS,
        bucket_exists=True,
        managed_r2_dev_enabled=managed["enabled"],
        custom_domain_count=len(domains),
        enabled_custom_domain_count=enabled_custom_domain_count,
        public_state_result="PASS",
        rule_id=rule_id,
        rule_name=rule_name,
        covered_prefix=expected_lock_prefix,
        rule_prefix=rule_prefix,
        scope_kind="exact",
        retention_type="Indefinite",
        verified_at_utc=_utc_now(),
    )


def _output(
    payload: Mapping[str, object],
    *,
    environ: Mapping[str, str] | None = None,
) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    print(redact_text(rendered, environ))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish or verify one finalized artifact in Cloudflare R2"
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)
    publish = subparsers.add_parser("publish")
    publish.add_argument("artifact", type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("root_sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    statuses = required_variable_status()
    config: R2Config | None = None
    lock: BucketLockEvidence | None = None
    try:
        config = R2Config.from_environment()
        lock = verify_cloudflare_bucket_lock(config)
        client = R2S3Client(config)
        if arguments.operation == "publish":
            artifact = validate_local_artifact(arguments.artifact)
            result: Mapping[str, object] = publish_artifact(
                client, config, artifact
            ).to_dict()
            _output(
                {
                    "status": "PASS",
                    "required_variables": statuses,
                    "r2": config.public_identity(),
                    "bucket_lock": lock.to_dict(),
                    arguments.operation: result,
                }
            )
        else:
            with tempfile.TemporaryDirectory(prefix="kvbench-r2-verify-") as temporary:
                result = verify_remote_artifact(
                    client,
                    config,
                    arguments.root_sha256,
                    Path(temporary) / "artifact",
                ).to_dict()
                _output(
                    {
                        "status": "PASS",
                        "required_variables": statuses,
                        "r2": config.public_identity(),
                        "bucket_lock": lock.to_dict(),
                        arguments.operation: result,
                    }
                )
        return 0
    except (R2ArtifactError, OSError, ValueError) as error:
        failure: dict[str, object] = {
            "status": "FAIL",
            "required_variables": statuses,
            "error": redact_text(str(error)),
            "error_type": type(error).__name__,
        }
        if config is not None:
            failure["r2"] = config.public_identity()
        if lock is not None:
            failure["bucket_lock"] = lock.to_dict()
        if isinstance(error, RemoteRequestError):
            failure["remote_error"] = error.to_dict()
        _output(failure)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
