#!/usr/bin/env python3
"""Seal a copy of results/addendum-20261005 for R2 (same control files as the post-hoc seals).

    python3 addendum/seal.py [--launch-scripts DIR]

Copies (file by file, never onto a repository directory) the results, the
addendum code, the amendment and the launch scripts into
/home/rockrock/kvbench-addendum-sealed/<run_id>/, writes manifest.json,
artifact_inventory.json, checksums.sha256 and COMPLETE (written last), makes
the tree read-only and validates it with scripts/r2_artifact.validate_local_artifact.
Publish afterwards as root from the repository root:

    set -a; . ./.env; set +a
    python3 scripts/r2_artifact.py publish <sealed dir>
    python3 scripts/r2_artifact.py verify <root_sha256>
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

PARENT = Path("/home/rockrock/kvbench-addendum-sealed")


def write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def copy_tree(source: Path, destination: Path, skip: set[str] = frozenset()) -> int:
    count = 0
    for path in sorted(source.rglob("*")):
        rel = path.relative_to(source)
        if any(part in skip for part in rel.parts) or not path.is_file() or path.is_symlink():
            continue
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--launch-scripts", type=Path, default=None)
    args = parser.parse_args()
    commit = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(c.ADDENDUM_REPO), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dt%H%M%S%fz")
    run_id = f"addendum-20261005-{stamp}-{commit[:8]}-{secrets.token_hex(3)}"
    PARENT.mkdir(parents=True, exist_ok=True)
    stage = PARENT / f".staging-{run_id}"
    stage.mkdir()
    counts = {
        "results": copy_tree(c.RESULTS, stage / "results" / c.ADDENDUM_ID),
        "code": copy_tree(c.ADDENDUM_REPO / "addendum", stage / "code" / "addendum", {"__pycache__"}),
    }
    (stage / "code" / c.AMENDMENT).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(c.ADDENDUM_REPO / c.AMENDMENT, stage / "code" / c.AMENDMENT)
    if args.launch_scripts:
        counts["launch_scripts"] = copy_tree(args.launch_scripts, stage / "code" / "launch-scripts")
    log = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(c.ADDENDUM_REPO), "log", "--format=%H %s",
                          "121c24d..HEAD"], capture_output=True, text=True, check=True).stdout
    write_new(stage / "code" / "git-log.txt", log)
    manifest = {
        "schema_version": "kvbench-addendum-20261005-seal-1.0.0",
        "run_id": run_id,
        "status": "complete_with_recorded_failures",
        "addendum_id": c.ADDENDUM_ID,
        "branch": "addendum-20261005",
        "addendum_git_sha": commit,
        "base_commit": "121c24d",
        "execution_git_sha": c.EXECUTION_SHA,
        "container_digest": c.IMAGE,
        "gpu_uuid": c.GPU_UUID,
        "amendment": c.AMENDMENT,
        "post_hoc": True,
        "frozen_roots_unchanged": True,
        "file_counts": counts,
        "sealed_at_utc": c.utc_now(),
        "notes": ("Task 2 timing is diagnostic only (gate 1 failed); Task 3 measured as task3b; Task 4 uses a "
                  "synthetic cache for timing only; see results/addendum-20261005/REPORT.md and FAILURES.md."),
    }
    write_new(stage / "manifest.json", c.json_text(manifest))
    controls = {"artifact_inventory.json", "checksums.sha256", "COMPLETE"}
    payload = sorted(p for p in stage.rglob("*") if p.is_file() and p.relative_to(stage).as_posix() not in controls)
    items = [{"path": p.relative_to(stage).as_posix(), "role": "addendum_20261005",
              "size_bytes": p.stat().st_size, "sha256": c.sha256_file(p)} for p in payload]
    write_new(stage / "artifact_inventory.json", c.json_text({
        "schema_version": "kvbench-artifact-inventory-1.0.0", "run_id": run_id, "files": items,
        "excluded_control_files": sorted(controls)}))
    ledger_files = sorted(p for p in stage.rglob("*") if p.is_file()
                          and p.relative_to(stage).as_posix() not in {"checksums.sha256", "COMPLETE"})
    write_new(stage / "checksums.sha256",
              "".join(f"{c.sha256_file(p)}  {p.relative_to(stage).as_posix()}\n" for p in ledger_files))
    write_new(stage / "COMPLETE", c.json_text({
        "schema_version": "kvbench-completion-1.0.0", "run_id": run_id, "status": manifest["status"],
        "manifest_sha256": c.sha256_file(stage / "manifest.json"),
        "artifact_inventory_sha256": c.sha256_file(stage / "artifact_inventory.json"),
        "checksum_ledger_path": "checksums.sha256",
        "checksum_ledger_sha256": c.sha256_file(stage / "checksums.sha256"),
        "written_last": True}))
    final = PARENT / run_id
    if final.exists():
        raise SystemExit(f"refusing to replace {final}")
    os.rename(stage, final)
    for path in sorted(final.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    final.chmod(0o555)
    sys.path.insert(0, str(c.REPO))
    from scripts.r2_artifact import validate_local_artifact
    root = validate_local_artifact(final, environ={}).root_sha256
    print(json.dumps({"sealed": str(final), "root_sha256": root, "files": len(items) + 4,
                      "bytes": sum(i["size_bytes"] for i in items)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
