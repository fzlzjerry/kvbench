"""Command-line validation skeleton for Phase 2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, NoReturn, Sequence

from kvbench.config import (
    REPOSITORY_ROOT,
    ExperimentConfig,
    load_config,
    load_experiment_bundle,
    load_phase3_admission_bundle,
)
from kvbench.errors import (
    AdmissionError,
    ErrorCode,
    KVBenchError,
    PhaseNotImplementedError,
)
from kvbench.runtime.artifacts import summarize_run_directory, validate_run_directory
from kvbench.validation import evaluate_admission
from kvbench.schema import Phase3AdmissionPlan, expand_phase3_process_points


def _emit(payload: dict[str, Any], *, stream: Any | None = None) -> None:
    target = sys.stdout if stream is None else stream
    print(
        json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")),
        file=target,
    )


def _error(error: KVBenchError, *, exit_code: int = 2) -> NoReturn:
    _emit({"error": error.to_dict(), "ok": False}, stream=sys.stderr)
    raise SystemExit(exit_code)


def _repository_relative(path: Path) -> str:
    resolved = path.resolve(strict=True)
    root = REPOSITORY_ROOT.resolve(strict=True)
    if resolved.is_relative_to(root):
        return resolved.relative_to(root).as_posix()
    return resolved.as_posix()


def command_validate_config(args: argparse.Namespace) -> int:
    path = Path(args.path)
    document = load_config(path)
    references_validated = False
    blockers: tuple[str, ...] = ()
    if isinstance(document, ExperimentConfig):
        bundle = load_experiment_bundle(path)
        references_validated = True
        blockers = bundle.blockers
    _emit(
        {
            "schema_version": "kvbench-cli-result-1.0.0",
            "ok": True,
            "command": "validate-config",
            "path": _repository_relative(path),
            "document_type": document.to_dict()["document_type"],
            "document_schema_version": document.to_dict()["schema_version"],
            "canonical_sha256": document.fingerprint(),
            "references_validated": references_validated,
            "blockers": list(blockers),
            "execution_attempted": False,
        }
    )
    return 0


def command_preflight(args: argparse.Namespace) -> int:
    argv = ["/usr/bin/make", "-C", str(REPOSITORY_ROOT), "preflight"]
    if args.dry_run:
        _emit(
            {
                "schema_version": "kvbench-cli-result-1.0.0",
                "ok": True,
                "command": "preflight",
                "delegated_argv": argv,
                "certified_semantics_preserved": True,
                "execution_attempted": False,
            }
        )
        return 0
    environment = dict(os.environ)
    result = subprocess.run(argv, check=False, env=environment)
    return int(result.returncode)


def command_run(args: argparse.Namespace) -> int:
    bundle = load_experiment_bundle(args.plan)
    if isinstance(bundle.plan, Phase3AdmissionPlan):
        phase3_bundle = load_phase3_admission_bundle(args.plan)
        plan_path = _repository_relative(phase3_bundle.plan_path)
        points = expand_phase3_process_points(phase3_bundle.plan)
        if args.dry_run:
            _emit(
                {
                    "schema_version": "kvbench-phase3-dry-run-1.0.0",
                    "ok": True,
                    "command": "run",
                    "plan": plan_path,
                    "plan_sha256": phase3_bundle.plan.fingerprint(),
                    "intended_argv": [
                        "kvbench",
                        "run",
                        "--plan",
                        plan_path,
                        "--dry-run",
                    ],
                    "point_ids": [point.point_id for point in points],
                    "expected_process_count": len(points),
                    "narrow_native_host_execution_ready": (
                        phase3_bundle.execution_ready
                    ),
                    "formal_blockers_retained": list(
                        phase3_bundle.all_blockers
                    ),
                    "performance_execution_implemented": True,
                    "execution_attempted": False,
                    "timing_collected": False,
                    "profiler_executed": False,
                    "quality_executed": False,
                    "performance_claim_eligible": False,
                    "measurement_scope": "native_host_admission",
                }
            )
            return 0
        try:
            from kvbench.runtime.phase3_coordinator import run_phase3_campaign

            result = run_phase3_campaign(plan_path)
        except RuntimeError as error:
            raise AdmissionError(
                f"Phase 3 coordination failed closed: {error}"
            ) from error
        _emit(result)
        return 0 if result["ok"] else 1
    admission = evaluate_admission(bundle)
    plan_path = _repository_relative(bundle.plan_path)
    intended = ("kvbench", "run", "--plan", plan_path, "--dry-run")
    payload = {
        "schema_version": "kvbench-dry-run-1.0.0",
        "ok": bool(args.dry_run),
        "command": "run",
        "plan": plan_path,
        "plan_sha256": bundle.plan.fingerprint(),
        "intended_argv": list(intended),
        "admission": admission.to_dict(),
        "referenced_fingerprints": dict(bundle.canonical_fingerprints),
        "performance_execution_implemented": False,
        "execution_attempted": False,
        "timing_collected": False,
        "profiler_executed": False,
        "quality_executed": False,
    }
    if args.dry_run:
        _emit(payload)
        return 0
    error = PhaseNotImplementedError(
        "performance execution begins in Phase 3; configuration was validated only"
    )
    payload["error"] = error.to_dict()
    _emit(payload, stream=sys.stderr)
    return 3


def command_phase3_worker(args: argparse.Namespace) -> int:
    """Internal supervised worker; direct invocation fails without IPC state."""

    from kvbench.runtime.phase3_worker import execute_worker, emit_worker_result

    result = execute_worker(
        plan_path=args.plan,
        point_id=args.point_id,
        replicate=args.replicate,
        run_id=args.run_id,
    )
    emit_worker_result(result)
    return 0


def command_validate_run(args: argparse.Namespace) -> int:
    result = validate_run_directory(args.run_dir)
    _emit(
        {
            "schema_version": "kvbench-cli-result-1.0.0",
            "ok": result.valid and result.complete,
            "command": "validate-run",
            "result": result.to_dict(),
            "execution_attempted": False,
        }
    )
    return 0 if result.valid and result.complete else 1


def command_summarize(args: argparse.Namespace) -> int:
    result = summarize_run_directory(args.run_dir)
    _emit(
        {
            "schema_version": "kvbench-cli-result-1.0.0",
            "ok": True,
            "command": "summarize",
            "summary": result,
            "execution_attempted": False,
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kvbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_config = subparsers.add_parser("validate-config")
    validate_config.add_argument("path")
    validate_config.set_defaults(handler=command_validate_config)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--dry-run", action="store_true")
    preflight.set_defaults(handler=command_preflight)

    run = subparsers.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(handler=command_run)

    phase3_worker = subparsers.add_parser("phase3-worker")
    phase3_worker.add_argument("--plan", required=True)
    phase3_worker.add_argument("--point-id", required=True)
    phase3_worker.add_argument("--replicate", required=True, type=int)
    phase3_worker.add_argument("--run-id", required=True)
    phase3_worker.set_defaults(handler=command_phase3_worker)

    validate_run = subparsers.add_parser("validate-run")
    validate_run.add_argument("run_dir")
    validate_run.set_defaults(handler=command_validate_run)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("run_dir")
    summarize.set_defaults(handler=command_summarize)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except KVBenchError as error:
        _error(error)
    except (OSError, ValueError) as error:
        safe = KVBenchError(ErrorCode.CONFIG_LOAD_ERROR, str(error))
        _error(safe)


if __name__ == "__main__":
    raise SystemExit(main())
