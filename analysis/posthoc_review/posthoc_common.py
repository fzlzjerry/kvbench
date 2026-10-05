"""Shared helpers for the POST-HOC review-round analyses (CPU only).

POST-HOC: every analysis that uses this module was defined after the frozen
performance and quality results were known.  Nothing here launches CUDA,
re-runs timing, or writes into an existing evidence directory.  New outputs are
sealed into a fresh, content-addressable directory with the same control files
(manifest.json, artifact_inventory.json, checksums.sha256, COMPLETE) that
scripts/r2_artifact.py requires, so they can be published to R2 unchanged.

Noncritical failures (optional telemetry, figure rendering) are recorded in a
warning log and never abort the run.  Input-integrity failures do abort,
because they mean the analysis would not be reading the frozen evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import secrets
import subprocess
import sys
import traceback
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

POSTHOC_LABEL = (
    "POST-HOC: defined after the frozen performance and quality results were "
    "known; not preregistered; not a performance claim"
)
ARTIFACT_PARENT = REPO / "paper/posthoc/artifacts"


class PosthocError(RuntimeError):
    """A critical failure: the analysis cannot be trusted and must stop."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={REPO}", "-C", str(REPO), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=30,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


class WarningLog:
    """Collects noncritical failures; the run continues after each one."""

    def __init__(self) -> None:
        self.entries: list[dict[str, str]] = []

    def warn(self, code: str, message: str) -> None:
        self.entries.append({"code": code, "message": message, "at_utc": utc_now()})
        print(f"WARNING [{code}] {message}", file=sys.stderr)

    def guard(self, code: str, function: Callable[[], Any], default: Any = None) -> Any:
        try:
            return function()
        except Exception as error:  # noncritical by construction
            self.warn(code, f"{type(error).__name__}: {error}\n{traceback.format_exc(limit=3)}")
            return default


# ---------------------------------------------------------------- input checks

def verify_full_root(directory: Path, expected_root: str) -> dict[str, Any]:
    """Full validation through the repository's own R2 validator."""

    from scripts.r2_artifact import validate_local_artifact

    artifact = validate_local_artifact(directory, environ={})
    if artifact.root_sha256 != expected_root:
        raise PosthocError(f"input root differs for {directory}: {artifact.root_sha256}")
    return {
        "path": str(Path(directory).relative_to(REPO)),
        "root_sha256": artifact.root_sha256,
        "verification": "full: every file hashed by scripts/r2_artifact.validate_local_artifact",
        "file_count": len(artifact.files),
    }


class LedgerRoot:
    """Ledger-anchored verification for very large roots.

    The R2 root hash is SHA-256 over the canonical "<sha256>  <path>" list of
    every file.  The checksum ledger lists every file except itself and
    COMPLETE, so the root can be recomputed from the ledger plus those two
    small files.  Each payload file that is actually read is then hashed and
    compared with its ledger entry.  This proves the bytes read are the bytes
    of the published root without rehashing the whole tree.
    """

    def __init__(self, directory: Path, expected_root: str) -> None:
        self.directory = Path(directory)
        ledger: dict[str, str] = {}
        for line in (self.directory / "checksums.sha256").read_text().splitlines():
            digest, relative = line.split("  ", 1)
            ledger[relative] = digest
        entries = dict(ledger)
        entries["checksums.sha256"] = sha256_file(self.directory / "checksums.sha256")
        entries["COMPLETE"] = sha256_file(self.directory / "COMPLETE")
        canonical = "".join(f"{entries[key]}  {key}\n" for key in sorted(entries)).encode()
        root = hashlib.sha256(canonical).hexdigest()
        if root != expected_root:
            raise PosthocError(f"ledger-anchored root differs for {directory}: {root}")
        self.root = root
        self.ledger = ledger
        self.verified: dict[str, str] = {}

    def path(self, relative: str) -> Path:
        if relative not in self.verified:
            expected = self.ledger.get(relative)
            if expected is None:
                raise PosthocError(f"{relative} is not in the ledger of {self.directory}")
            actual = sha256_file(self.directory / relative)
            if actual != expected:
                raise PosthocError(f"{relative} differs from its ledger entry")
            self.verified[relative] = actual
        return self.directory / relative

    def record(self) -> dict[str, Any]:
        return {
            "path": str(self.directory.relative_to(REPO)),
            "root_sha256": self.root,
            "verification": "ledger-anchored root plus per-file hash of every file read",
            "files_read": dict(sorted(self.verified.items())),
        }


# ---------------------------------------------------------------- sealing

def new_run_id(kind: str, git_sha: str | None) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S%fz")
    return f"posthoc-{kind}-{stamp}-{(git_sha or 'nogit')[:8]}-{secrets.token_hex(3)}"


def new_stage(run_id: str, parent: Path = ARTIFACT_PARENT) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    stage = parent / f".staging-{run_id}"
    stage.mkdir()
    return stage


def write_new(path: Path, data: bytes | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = data.encode() if isinstance(data, str) else data
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _payload(stage: Path, excluded: set[str]) -> list[Path]:
    return sorted(
        path for path in stage.rglob("*")
        if path.is_file() and path.relative_to(stage).as_posix() not in excluded
    )


def seal(stage: Path, run_id: str, manifest: Mapping[str, Any], role: str,
         parent: Path = ARTIFACT_PARENT) -> tuple[Path, str]:
    """Write the control files, promote without replacement, make read-only."""

    status = str(manifest["status"])
    write_new(stage / "manifest.json", json_text({**manifest, "run_id": run_id}))
    controls = {"artifact_inventory.json", "checksums.sha256", "COMPLETE"}
    items = [
        {
            "path": path.relative_to(stage).as_posix(),
            "role": role,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in _payload(stage, controls)
    ]
    write_new(stage / "artifact_inventory.json", json_text({
        "schema_version": "kvbench-artifact-inventory-1.0.0",
        "run_id": run_id,
        "files": items,
        "excluded_control_files": ["artifact_inventory.json", "checksums.sha256", "COMPLETE"],
    }))
    ledger = "".join(
        f"{sha256_file(path)}  {path.relative_to(stage).as_posix()}\n"
        for path in _payload(stage, {"checksums.sha256", "COMPLETE"})
    )
    write_new(stage / "checksums.sha256", ledger)
    write_new(stage / "COMPLETE", json_text({
        "schema_version": "kvbench-completion-1.0.0",
        "run_id": run_id,
        "status": status,
        "manifest_sha256": sha256_file(stage / "manifest.json"),
        "artifact_inventory_sha256": sha256_file(stage / "artifact_inventory.json"),
        "checksum_ledger_path": "checksums.sha256",
        "checksum_ledger_sha256": sha256_file(stage / "checksums.sha256"),
        "written_last": True,
    }))
    final = parent / run_id
    if final.exists():
        raise PosthocError(f"refusing to replace existing artifact {final}")
    os.rename(stage, final)
    for path in sorted(final.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    final.chmod(0o555)
    from scripts.r2_artifact import validate_local_artifact

    root = validate_local_artifact(final, environ={}).root_sha256
    return final, root


def environment_record() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("numpy", "pyarrow"):
        try:
            module = __import__(name)
            packages[name] = getattr(module, "__version__", None)
        except ImportError:
            packages[name] = None
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_used": False,
        "container_digest": None,
        "container_note": "CPU-only post-hoc analysis of frozen evidence; no container, GPU, or timing run",
    }


# ---------------------------------------------------------------- SVG output

CONFIG_COLORS = {
    "tq_4bit_nc": "#08519c", "tq_k3v4_nc": "#3182bd", "tq_3bit_nc": "#6baed6",
    "k4v4": "#a63603", "k2v4": "#e6550d", "k2v2": "#fd8d3c",
    "kvq4": "#006d2c", "kvq3": "#31a354", "kvq2": "#74c476",
}
CONFIG_LABELS = {
    "bf16": "BF16",
    "tq_4bit_nc": "TQ-4bit", "tq_k3v4_nc": "TQ-k3v4", "tq_3bit_nc": "TQ-3bit",
    "k4v4": "KIVI-k4v4", "k2v4": "KIVI-k2v4", "k2v2": "KIVI-k2v2",
    "kvq4": "KVQuant-4", "kvq3": "KVQuant-3", "kvq2": "KVQuant-2",
}


def escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def marker(batch: int, x: float, y: float, color: str, size: float = 3.4,
           hollow: bool = False) -> str:
    """Same batch-size marker shapes as the paper's Figure 1."""

    paint = f'fill="white" stroke="{color}" stroke-width="1.3"' if hollow else f'fill="{color}"'
    if batch == 1:
        return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{size:.1f}" {paint}/>'
    if batch == 2:
        return (f'<rect x="{x - size:.2f}" y="{y - size:.2f}" width="{2 * size:.2f}" '
                f'height="{2 * size:.2f}" {paint}/>')
    if batch == 4:
        return (f'<path d="M {x:.2f} {y - size - 0.8:.2f} L {x + size + 0.6:.2f} {y + size:.2f} '
                f'L {x - size - 0.6:.2f} {y + size:.2f} Z" {paint}/>')
    if batch == 8:
        return (f'<path d="M {x:.2f} {y - size - 0.8:.2f} L {x + size + 0.8:.2f} {y:.2f} '
                f'L {x:.2f} {y + size + 0.8:.2f} L {x - size - 0.8:.2f} {y:.2f} Z" {paint}/>')
    return (f'<path d="M {x:.2f} {y + size + 0.8:.2f} L {x + size + 0.6:.2f} {y - size:.2f} '
            f'L {x - size - 0.6:.2f} {y - size:.2f} Z" {paint}/>')


def _decade_ticks(low: float, high: float) -> list[float]:
    ticks = []
    exponent = math.floor(math.log10(low))
    while 10 ** exponent <= high * 1.0000001:
        for mantissa in (1, 2, 5):
            value = mantissa * 10 ** exponent
            if low * 0.9999999 <= value <= high * 1.0000001:
                ticks.append(value)
        exponent += 1
    return ticks


def _tick_label(value: float) -> str:
    if value >= 1:
        return f"{value:g}"
    return f"{value:.3g}"


def render_loglog_panels(
    panels: Sequence[Mapping[str, Any]],
    *,
    x_label: str,
    y_label: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    legend: Sequence[tuple[str, str]],
    diagonal: bool = True,
    unity_y: bool = True,
    unity_x: bool = False,
    x_ticks: Sequence[float] | None = None,
) -> str:
    """Side-by-side log-log scatter panels with shared axes.

    Each panel: {"title": str, "points": [(x, y, config, batch, hollow)]}.
    """

    panel_width, panel_height = 300.0, 300.0
    left_margin, gap, top, bottom_margin = 70.0, 34.0, 34.0, 60.0
    legend_width = 150.0
    width = left_margin + len(panels) * panel_width + (len(panels) - 1) * gap + legend_width
    height = top + panel_height + bottom_margin
    lx0, lx1 = (math.log10(value) for value in x_range)
    ly0, ly1 = (math.log10(value) for value in y_range)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}">',
        f'<rect width="{width:.0f}" height="{height:.0f}" fill="white"/>',
        f'<text x="16" y="{top + panel_height / 2:.1f}" transform="rotate(-90 16 {top + panel_height / 2:.1f})" '
        f'text-anchor="middle" font-family="sans-serif" font-size="13">{escape(y_label)}</text>',
    ]
    x_ticks = list(x_ticks) if x_ticks is not None else _decade_ticks(*x_range)
    y_ticks = _decade_ticks(*y_range)
    for index, panel in enumerate(panels):
        x0 = left_margin + index * (panel_width + gap)
        x1 = x0 + panel_width
        y0, y1 = top, top + panel_height

        def px(value: float, x0: float = x0) -> float:
            return x0 + (math.log10(value) - lx0) / (lx1 - lx0) * panel_width

        def py(value: float) -> float:
            return y1 - (math.log10(value) - ly0) / (ly1 - ly0) * panel_height

        elements.append(
            f'<text x="{(x0 + x1) / 2:.1f}" y="{top - 12:.1f}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="13">{escape(str(panel["title"]))}</text>')
        for value in y_ticks:
            y = py(value)
            major = abs(math.log10(value) - round(math.log10(value))) < 1e-9
            elements.append(f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x1:.1f}" y2="{y:.1f}" '
                            f'stroke="{"#dddddd" if major else "#f2f2f2"}"/>')
            if index == 0:
                elements.append(f'<text x="{x0 - 6:.1f}" y="{y + 4:.1f}" text-anchor="end" '
                                f'font-family="sans-serif" font-size="10">{_tick_label(value)}</text>')
        for value in x_ticks:
            x = px(value)
            major = abs(math.log10(value) - round(math.log10(value))) < 1e-9
            elements.append(f'<line x1="{x:.1f}" y1="{y0:.1f}" x2="{x:.1f}" y2="{y1:.1f}" '
                            f'stroke="{"#dddddd" if major else "#f2f2f2"}"/>')
            elements.append(f'<text x="{x:.1f}" y="{y1 + 15:.1f}" text-anchor="middle" '
                            f'font-family="sans-serif" font-size="10">{_tick_label(value)}</text>')
        elements.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{panel_width:.1f}" '
                        f'height="{panel_height:.1f}" fill="none" stroke="black"/>')
        if unity_y and y_range[0] < 1 < y_range[1]:
            elements.append(f'<line x1="{x0:.1f}" y1="{py(1):.1f}" x2="{x1:.1f}" y2="{py(1):.1f}" '
                            'stroke="#555555" stroke-dasharray="5 3"/>')
        if unity_x and x_range[0] < 1 < x_range[1]:
            elements.append(f'<line x1="{px(1):.1f}" y1="{y0:.1f}" x2="{px(1):.1f}" y2="{y1:.1f}" '
                            'stroke="#555555" stroke-dasharray="5 3"/>')
        if diagonal:
            low = max(x_range[0], y_range[0])
            high = min(x_range[1], y_range[1])
            if low < high:
                elements.append(f'<line x1="{px(low):.1f}" y1="{py(low):.1f}" x2="{px(high):.1f}" '
                                f'y2="{py(high):.1f}" stroke="#c00000" stroke-width="1.2"/>')
        for x_value, y_value, configuration, batch, hollow in panel["points"]:
            if not (x_range[0] <= x_value <= x_range[1] and y_range[0] <= y_value <= y_range[1]):
                continue
            elements.append(marker(int(batch), px(x_value), py(y_value),
                                   CONFIG_COLORS.get(configuration, "#444444"), hollow=bool(hollow)))
    elements.append(
        f'<text x="{left_margin + (len(panels) * panel_width + (len(panels) - 1) * gap) / 2:.1f}" '
        f'y="{top + panel_height + 40:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="13">{escape(x_label)}</text>')
    legend_x = left_margin + len(panels) * panel_width + (len(panels) - 1) * gap + 18
    legend_y = top + 8
    for index, (configuration, label) in enumerate(legend):
        y = legend_y + 17 * index
        elements.append(f'<rect x="{legend_x:.1f}" y="{y - 6:.1f}" width="12" height="9" '
                        f'fill="{CONFIG_COLORS.get(configuration, "#444444")}"/>')
        elements.append(f'<text x="{legend_x + 18:.1f}" y="{y + 2:.1f}" font-family="sans-serif" '
                        f'font-size="11">{escape(label)}</text>')
    batch_y = legend_y + 17 * len(legend) + 12
    for index, batch in enumerate((1, 2, 4, 8, 16)):
        y = batch_y + 17 * index
        elements.append(marker(batch, legend_x + 6, y - 2, "#444444"))
        elements.append(f'<text x="{legend_x + 18:.1f}" y="{y + 2:.1f}" font-family="sans-serif" '
                        f'font-size="11">B = {batch}</text>')
    elements.append("</svg>")
    return "".join(elements)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> None:
    import csv
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n",
                            extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _csv_value(row.get(key)) for key in columns})
    write_new(path, buffer.getvalue())


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return repr(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True)
    return value
