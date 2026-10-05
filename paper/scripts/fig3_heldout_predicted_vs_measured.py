"""Redraw Figure 3 (held-out predicted vs measured latency) with axes, ticks, and axis labels.

Source data: the modeling reproduction package's out-of-fold prediction table, loaded
through the package's own `audit()` and filtered exactly as its `reproduce()` does
(geometry holdout protocols only). `--original <svg>` checks that the package's
`_svg_scatter` reproduces that figure byte for byte. `render_readable` keeps the
same points, log10 transform, shared axis range, square plot area, and identity line,
and adds only axes, ticks, tick labels, and axis titles. It has no in-figure title (the
caption carries it); the canvas is cropped through the SVG viewBox.

Usage (from the repository root; standard library only):
    python3 paper/scripts/fig3_heldout_predicted_vs_measured.py \
        --package <modeling package root> \
        --output paper/figures/fig3_heldout_predicted_vs_measured.svg
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import tempfile
from pathlib import Path

TITLE = "Held-out predicted vs measured latency"  # original title, used only with --original


def load_package_module(package: Path):
    spec = importlib.util.spec_from_file_location("package_reproduce", package / "reproduce.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_rows(module, package: Path) -> list[tuple[float, float]]:
    """Same selection as reproduce(): logical out-of-fold rows from geometry protocols."""
    _, _, logical = module.audit(package)
    selected = [row for row in logical if row["protocol"] in module.GEOMETRY_PROTOCOLS]
    return [(float(row["observed_host_wall_ms"]), float(row["predicted_host_wall_ms"])) for row in selected]


def render_readable(rows: list[tuple[float, float]]) -> str:
    # Unchanged geometry and transform from reproduce.py::_svg_scatter (log_scale=True).
    width, height, left, top, plot = 900, 560, 80, 55, 430
    transformed = [(math.log10(max(x, 1e-9)), math.log10(max(y, 1e-9))) for x, y in rows]
    xs = [x for x, _ in transformed] or [0.0, 1.0]
    ys = [y for _, y in transformed] or [0.0, 1.0]
    low, high = min(min(xs), min(ys)), max(max(xs), max(ys))
    span = high - low or 1.0
    circles = "".join(
        f'<circle cx="{left+(x-low)/span*plot:.2f}" cy="{top+plot-(y-low)/span*plot:.2f}" r="2.3" fill="#235789" fill-opacity="0.55"/>'
        for x, y in transformed)
    diagonal = f'<line x1="{left}" y1="{top+plot}" x2="{left+plot}" y2="{top}" stroke="#d1495b" stroke-width="2"/>'

    # Added: axes, 1-2-5 ticks in milliseconds on both log10 axes, and axis titles.
    def position(log_value: float) -> float:
        return (log_value - low) / span * plot

    ticks = []
    for decade in range(math.floor(low) - 1, math.ceil(high) + 1):
        for multiplier in (1, 2, 5):
            value = multiplier * 10 ** decade
            if low <= math.log10(value) <= high:
                ticks.append(value)
    parts = [f'<line x1="{left}" y1="{top+plot}" x2="{left+plot}" y2="{top+plot}" stroke="black"/>',
             f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot}" stroke="black"/>']
    for value in ticks:
        offset = position(math.log10(value))
        label = f"{value:,.0f}" if value >= 1 else f"{value:g}"
        x, y = left + offset, top + plot - offset
        parts.extend((
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+plot}" stroke="#eeeeee"/>',
            f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot}" y2="{y:.2f}" stroke="#eeeeee"/>',
            f'<line x1="{x:.2f}" y1="{top+plot}" x2="{x:.2f}" y2="{top+plot+5}" stroke="black"/>',
            f'<text x="{x:.2f}" y="{top+plot+19}" text-anchor="middle" font-family="sans-serif" font-size="12">{label}</text>',
            f'<line x1="{left-5}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="black"/>',
            f'<text x="{left-8}" y="{y+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{label}</text>',
        ))
    parts.append(f'<text x="{left+plot/2}" y="{top+plot+40}" text-anchor="middle" font-family="sans-serif" '
                 'font-size="13">measured wall-clock latency (ms per full-batch step, log scale)</text>')
    parts.append(f'<text x="22" y="{top+plot/2}" transform="rotate(-90 22 {top+plot/2})" text-anchor="middle" '
                 'font-family="sans-serif" font-size="13">predicted latency (ms, log scale)</text>')
    parts.append(f'<line x1="{left+plot+30}" y1="80" x2="{left+plot+58}" y2="80" stroke="#d1495b" stroke-width="2"/>'
                 f'<text x="{left+plot+66}" y="84" font-family="sans-serif" font-size="12">y = x</text>'
                 f'<circle cx="{left+plot+44}" cy="104" r="3" fill="#235789" fill-opacity="0.55"/>'
                 f'<text x="{left+plot+66}" y="108" font-family="sans-serif" font-size="12">held-out logical point (model D)</text>')
    # Same 900 x 560 drawing as the original; the viewBox crops the empty title band.
    crop_top = 40
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height - crop_top}" '
            f'viewBox="0 {crop_top} {width} {height - crop_top}"><rect y="{crop_top}" width="100%" height="100%" fill="white"/>'
            f'{"".join(parts)}{diagonal}{circles}'
            f'<text x="{left}" y="{height-8}" font-size="12" font-family="sans-serif">x=observed, y=predicted; log10 scale; outliers retained</text></svg>\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--original", "--verify-original", dest="original", type=Path,
                        help="original SVG to compare against byte for byte")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    module = load_package_module(args.package)
    rows = load_rows(module, args.package)
    if args.original is not None:
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "original.svg"
            module._svg_scatter(path, TITLE, rows, log_scale=True)
            same = path.read_bytes() == args.original.read_bytes()
        print("original plot reproduced byte-for-byte:", same, f"({len(rows)} points)")
        if not same:
            raise SystemExit(1)
    if args.output is not None:
        args.output.write_text(render_readable(rows), encoding="utf-8")
        print("wrote", args.output)


if __name__ == "__main__":
    main()
