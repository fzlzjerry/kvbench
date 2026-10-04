"""Append-only preregistration and finalization for Phase 3 campaigns."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import stat
from typing import Any

from kvbench.schema import canonical_json_bytes, sha256_hex


CAMPAIGN_ID_PATTERN = re.compile(
    r"phase3-[0-9]{8}t[0-9]{12}z-[0-9a-f]{8}-[0-9a-f]{6}\Z"
)
_MAX_JSON_BYTES = 8 * 1024 * 1024


class Phase3CampaignError(RuntimeError):
    """Campaign preregistration or immutable validation failed closed."""


def _exclusive_write(path: Path, data: bytes, *, mode: int = 0o444) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(dict(payload)) + b"\n"


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise Phase3CampaignError(f"campaign evidence is absent: {path}") from error
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size > _MAX_JSON_BYTES
    ):
        raise Phase3CampaignError(f"campaign evidence is unsafe: {path}")

    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    raw = path.read_bytes()
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")),
        )
    except (UnicodeError, ValueError) as error:
        raise Phase3CampaignError(f"campaign JSON is invalid: {path}") from error
    if not isinstance(payload, dict) or raw != _json_bytes(payload):
        raise Phase3CampaignError(f"campaign JSON is not canonical: {path}")
    return payload


def _validate_identifier_sequence(
    values: object,
    *,
    label: str,
    expected_count: int,
) -> tuple[str, ...]:
    if (
        not isinstance(values, list)
        or len(values) != expected_count
        or any(not isinstance(value, str) or not value for value in values)
        or len(set(values)) != expected_count
    ):
        raise Phase3CampaignError(f"campaign {label} is not an exact unique list")
    return tuple(values)


def campaign_root(repository_root: str | Path) -> Path:
    root = Path(repository_root).resolve(strict=True) / "artifacts" / "phase3_campaigns"
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise Phase3CampaignError("campaign root is unsafe")
    return root


def _existing_preregistrations(root: Path) -> tuple[dict[str, Any], ...]:
    if not root.exists():
        return ()
    records: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        metadata = child.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise Phase3CampaignError("campaign root contains unsafe content")
        if not CAMPAIGN_ID_PATTERN.fullmatch(child.name):
            raise Phase3CampaignError("campaign root contains an unknown directory")
        payload = _strict_json(child / "preregistered.json")
        if payload.get("campaign_id") != child.name:
            raise Phase3CampaignError("campaign preregistration identity differs")
        records.append(payload)
    return tuple(records)


def assert_unique_plan_campaign(
    root: Path,
    *,
    plan_path: str,
    git_sha: str,
    selected_campaign_id: str | None = None,
) -> None:
    """Reject multiple attempts for an exact plan and implementation SHA."""

    matches = [
        record
        for record in _existing_preregistrations(root)
        if record.get("plan_path") == plan_path and record.get("git_sha") == git_sha
    ]
    expected = 0 if selected_campaign_id is None else 1
    if len(matches) != expected:
        raise Phase3CampaignError(
            "exact plan/Git SHA already has a campaign attempt; selective rerun is closed"
        )
    if selected_campaign_id is not None and matches[0].get("campaign_id") != selected_campaign_id:
        raise Phase3CampaignError("selected campaign is not the unique preregistered attempt")


@dataclass(slots=True)
class Phase3CampaignRecorder:
    """One campaign whose selection is fixed before any worker starts."""

    directory: Path
    preregistration: dict[str, Any]
    finalized: bool = False

    @classmethod
    def create(
        cls,
        *,
        repository_root: str | Path,
        campaign_id: str,
        created_at_utc: str,
        git_sha: str,
        plan_path: str,
        plan_fingerprint: str,
        point_ids: Sequence[str],
        run_ids: Sequence[str],
    ) -> Phase3CampaignRecorder:
        if not CAMPAIGN_ID_PATTERN.fullmatch(campaign_id):
            raise Phase3CampaignError("campaign ID is invalid")
        root = campaign_root(repository_root)
        root.mkdir(mode=0o755, parents=True, exist_ok=True)
        assert_unique_plan_campaign(root, plan_path=plan_path, git_sha=git_sha)
        points = tuple(point_ids)
        runs = tuple(run_ids)
        if (
            not points
            or len(points) != len(runs)
            or len(set(points)) != len(points)
            or len(set(runs)) != len(runs)
            or any(run_id != f"{campaign_id}-{point_id}" for point_id, run_id in zip(points, runs))
        ):
            raise Phase3CampaignError("campaign run selection is inconsistent")
        directory = root / campaign_id
        os.mkdir(directory, 0o755)
        preregistration = {
            "schema_version": "kvbench-phase3-campaign-preregistration-1.0.0",
            "campaign_id": campaign_id,
            "created_at_utc": created_at_utc,
            "git_sha": git_sha,
            "plan_path": plan_path,
            "plan_fingerprint": plan_fingerprint,
            "expected_process_count": len(points),
            "point_ids": list(points),
            "run_ids": list(runs),
            "selection_frozen_before_execution": True,
            "retry_policy": "no_second_campaign_for_exact_plan_and_git_sha",
            "selective_rerun_allowed": False,
            "performance_claim_eligible": False,
            "measurement_scope": "native_host_admission",
        }
        _exclusive_write(directory / "preregistered.json", _json_bytes(preregistration))
        _exclusive_write(
            directory / "lifecycle.0001-created.json",
            _json_bytes(
                {
                    "schema_version": "kvbench-phase3-campaign-lifecycle-1.0.0",
                    "campaign_id": campaign_id,
                    "sequence": 1,
                    "state": "created",
                }
            ),
        )
        return cls(directory=directory, preregistration=preregistration)

    def finalize(self, result: Mapping[str, Any]) -> Path:
        if self.finalized:
            raise Phase3CampaignError("campaign has already been finalized")
        expected_runs = tuple(self.preregistration["run_ids"])
        result_runs = result.get("runs")
        if not isinstance(result_runs, list):
            raise Phase3CampaignError("campaign result lacks its ordered run list")
        observed_runs = tuple(
            item.get("run_id") if isinstance(item, Mapping) else None
            for item in result_runs
        )
        if (
            result.get("campaign_id") != self.preregistration["campaign_id"]
            or result.get("plan") != self.preregistration["plan_path"]
            or result.get("plan_fingerprint")
            != self.preregistration["plan_fingerprint"]
            or observed_runs != expected_runs[: len(observed_runs)]
            or len(observed_runs) > len(expected_runs)
            or result.get("selective_rerun_performed") is not False
        ):
            raise Phase3CampaignError("campaign result differs from preregistration")
        _exclusive_write(self.directory / "result.json", _json_bytes(dict(result)))
        _exclusive_write(
            self.directory / "lifecycle.0002-finalized.json",
            _json_bytes(
                {
                    "schema_version": "kvbench-phase3-campaign-lifecycle-1.0.0",
                    "campaign_id": self.preregistration["campaign_id"],
                    "sequence": 2,
                    "state": "finalized",
                }
            ),
        )
        payload_paths = sorted(
            path for path in self.directory.iterdir() if path.is_file()
        )
        ledger = b"".join(
            f"{sha256_hex(path.read_bytes())}  {path.name}\n".encode("utf-8")
            for path in payload_paths
        )
        _exclusive_write(self.directory / "checksums.sha256", ledger)
        completion = {
            "schema_version": "kvbench-phase3-campaign-completion-1.0.0",
            "campaign_id": self.preregistration["campaign_id"],
            "preregistration_sha256": sha256_hex(
                (self.directory / "preregistered.json").read_bytes()
            ),
            "result_sha256": sha256_hex(
                (self.directory / "result.json").read_bytes()
            ),
            "checksum_ledger_sha256": sha256_hex(ledger),
            "written_last": True,
        }
        _exclusive_write(self.directory / "COMPLETE", _json_bytes(completion))
        for path in sorted(self.directory.iterdir(), reverse=True):
            path.chmod(0o444)
        self.directory.chmod(0o555)
        self.finalized = True
        validated = validate_phase3_campaign_directory(self.directory)
        if not validated["valid"]:
            raise Phase3CampaignError("final campaign record failed validation")
        return self.directory


def validate_phase3_campaign_directory(path: str | Path) -> dict[str, Any]:
    """Validate exact files, hashes, selection, and terminal campaign state."""

    errors: list[str] = []
    preregistration: dict[str, Any] = {}
    result: dict[str, Any] = {}
    try:
        lexical = Path(path).absolute()
        lexical_metadata = lexical.lstat()
        if stat.S_ISLNK(lexical_metadata.st_mode) or not stat.S_ISDIR(
            lexical_metadata.st_mode
        ):
            raise Phase3CampaignError("campaign directory is a symlink or non-directory")
        directory = lexical.resolve(strict=True)
        if not CAMPAIGN_ID_PATTERN.fullmatch(directory.name):
            raise Phase3CampaignError("campaign directory name is invalid")
        required = {
            "COMPLETE",
            "checksums.sha256",
            "lifecycle.0001-created.json",
            "lifecycle.0002-finalized.json",
            "preregistered.json",
            "result.json",
        }
        actual = {child.name for child in directory.iterdir()}
        if actual != required:
            raise Phase3CampaignError("campaign exact file set differs")
        write_bits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        for target in (directory, *directory.iterdir()):
            metadata = target.lstat()
            if (
                stat.S_ISLNK(metadata.st_mode)
                or metadata.st_mode & write_bits
                or target.is_file()
                and metadata.st_nlink != 1
            ):
                raise Phase3CampaignError("final campaign content is unsafe or writable")
        preregistration = _strict_json(directory / "preregistered.json")
        result = _strict_json(directory / "result.json")
        completion = _strict_json(directory / "COMPLETE")
        created = _strict_json(directory / "lifecycle.0001-created.json")
        finalized = _strict_json(directory / "lifecycle.0002-finalized.json")
        campaign_id = directory.name
        if (
            preregistration.get("schema_version")
            != "kvbench-phase3-campaign-preregistration-1.0.0"
            or preregistration.get("campaign_id") != campaign_id
            or preregistration.get("selection_frozen_before_execution") is not True
            or preregistration.get("selective_rerun_allowed") is not False
            or preregistration.get("retry_policy")
            != "no_second_campaign_for_exact_plan_and_git_sha"
        ):
            raise Phase3CampaignError("campaign preregistration policy differs")
        expected_count = preregistration.get("expected_process_count")
        if not isinstance(expected_count, int) or isinstance(expected_count, bool):
            raise Phase3CampaignError("campaign process count is invalid")
        points = _validate_identifier_sequence(
            preregistration.get("point_ids"),
            label="points",
            expected_count=expected_count,
        )
        runs = _validate_identifier_sequence(
            preregistration.get("run_ids"),
            label="runs",
            expected_count=expected_count,
        )
        if any(run_id != f"{campaign_id}-{point_id}" for point_id, run_id in zip(points, runs)):
            raise Phase3CampaignError("campaign point/run join differs")
        result_runs = result.get("runs")
        if not isinstance(result_runs, list):
            raise Phase3CampaignError("campaign result runs are invalid")
        observed_runs = tuple(
            item.get("run_id") if isinstance(item, Mapping) else None
            for item in result_runs
        )
        if (
            result.get("campaign_id") != campaign_id
            or result.get("plan") != preregistration.get("plan_path")
            or result.get("plan_fingerprint")
            != preregistration.get("plan_fingerprint")
            or result.get("selective_rerun_performed") is not False
            or observed_runs != runs[: len(observed_runs)]
        ):
            raise Phase3CampaignError("campaign result selection differs")
        if (
            created
            != {
                "schema_version": "kvbench-phase3-campaign-lifecycle-1.0.0",
                "campaign_id": campaign_id,
                "sequence": 1,
                "state": "created",
            }
            or finalized
            != {
                "schema_version": "kvbench-phase3-campaign-lifecycle-1.0.0",
                "campaign_id": campaign_id,
                "sequence": 2,
                "state": "finalized",
            }
        ):
            raise Phase3CampaignError("campaign lifecycle differs")
        ledger_path = directory / "checksums.sha256"
        ledger_bytes = ledger_path.read_bytes()
        entries: dict[str, str] = {}
        for line in ledger_bytes.decode("utf-8").splitlines():
            digest, separator, relative = line.partition("  ")
            lexical = PurePosixPath(relative)
            if (
                not separator
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or lexical.is_absolute()
                or ".." in lexical.parts
                or "/" in relative
                or relative in entries
            ):
                raise Phase3CampaignError("campaign checksum ledger is malformed")
            entries[relative] = digest
        expected_ledger_paths = required - {"COMPLETE", "checksums.sha256"}
        if set(entries) != expected_ledger_paths or list(entries) != sorted(entries):
            raise Phase3CampaignError("campaign checksum coverage differs")
        if any(
            sha256_hex((directory / relative).read_bytes()) != digest
            for relative, digest in entries.items()
        ):
            raise Phase3CampaignError("campaign checksum differs")
        if (
            completion.get("schema_version")
            != "kvbench-phase3-campaign-completion-1.0.0"
            or completion.get("campaign_id") != campaign_id
            or completion.get("preregistration_sha256")
            != sha256_hex((directory / "preregistered.json").read_bytes())
            or completion.get("result_sha256")
            != sha256_hex((directory / "result.json").read_bytes())
            or completion.get("checksum_ledger_sha256") != sha256_hex(ledger_bytes)
            or completion.get("written_last") is not True
        ):
            raise Phase3CampaignError("campaign completion marker differs")
    except (OSError, UnicodeError, ValueError, Phase3CampaignError) as error:
        errors.append(f"campaign validation failed closed: {type(error).__name__}")
    return {
        "schema_version": "kvbench-phase3-campaign-validation-1.0.0",
        "valid": not errors,
        "errors": errors,
        "preregistration": preregistration,
        "result": result,
    }
