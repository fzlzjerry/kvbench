"""Redraw Figure 1 (wall-clock same-work ratios) with a complete legend and readable ticks.

Source data: the measured same-work ratio table, shipped in the joint-results
reproduction package as inputs/wall_same_work_ratios.parquet (read only).
`render_original` is a verbatim copy of the plotting function that produced the
original figure; `--original <svg>` checks that it reproduces that file byte for byte.
`render_readable` keeps the same series, point set, and x/y coordinate mapping, and
changes only the legend, tick placement, axis labels, and the color/marker assignment
used to tell series apart. It has no in-figure title (the caption carries it) and no
footer note; the canvas is cropped to the plot through the SVG viewBox.

Usage (from the repository root, with pyarrow available):
    python3 paper/scripts/fig1_same_work_ratios.py \
        --table <joint-results package>/inputs/wall_same_work_ratios.parquet \
        --output paper/figures/fig1_same_work_ratios.svg
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pyarrow.parquet as pq

CONFIGURATIONS = ("bf16", "tq_4bit_nc", "tq_k3v4_nc", "tq_3bit_nc", "k4v4", "k2v4", "k2v2",
                  "kvq4", "kvq3", "kvq2")
BATCH_SIZES = (1, 2, 4, 8, 16)
TITLE = "Host-wall same-work ratio (quality unvalidated)"  # original title, used only with --original
FIELD = "performance_only_ratio"
NOTE = "Performance-only; quality unvalidated; not claim eligible"  # original note, used only with --original


def load_series(table: Path) -> dict[str, list[tuple[float, float]]]:
    """Same series construction as the original figure."""
    rows = pq.read_table(table).to_pylist()
    return {f"{config}/B{batch}": [
        (float(r["historical_context"]), float(r[FIELD])) for r in rows
        if r["method_config_id"] == config and r["batch_size"] == batch
        and r.get(FIELD) is not None]
        for config in CONFIGURATIONS for batch in BATCH_SIZES}


def escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def normalize(series):
    normalized: dict[str, list[tuple[float, float]]] = {}
    for name, points in series.items():
        finite = sorted(
            ((float(x), float(y)) for x, y in points
             if x > 0 and math.isfinite(float(x)) and math.isfinite(float(y))),
            key=lambda item: item[0])
        if finite:
            normalized[name] = finite
    return normalized


def mapping(normalized):
    """Unchanged coordinate mapping from the original plot function."""
    all_points = [point for points in normalized.values() for point in points]
    transformed_x = [math.log2(point[0]) for point in all_points]
    y_values = [point[1] for point in all_points]
    x_min, x_max = min(transformed_x), max(transformed_x)
    y_min, y_max = min(y_values), max(y_values)
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    if y_min == y_max:
        padding = max(abs(y_min) * 0.05, 1e-9)
    else:
        padding = (y_max - y_min) * 0.08
    y_min -= padding
    y_max += padding
    left, right, top, bottom = 82.0, 790.0, 70.0, 520.0

    def x_coordinate(value: float) -> float:
        return left + (math.log2(value) - x_min) * (right - left) / (x_max - x_min)

    def y_coordinate(value: float) -> float:
        return bottom - (value - y_min) * (bottom - top) / (y_max - y_min)

    return all_points, (y_min, y_max), (left, right, top, bottom), x_coordinate, y_coordinate


def render_original(series, *, title: str, y_label: str, note: str) -> str:
    """Verbatim logic of the original plotting function (returns the SVG text)."""
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
              "#7f7f7f", "#bcbd22", "#17becf", "#3366cc", "#dc3912")
    normalized = normalize(series)
    all_points, (y_min, y_max), (left, right, top, bottom), x_coordinate, y_coordinate = mapping(normalized)
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="640" '
        'viewBox="0 0 1100 640">',
        '<rect width="1100" height="640" fill="white"/>',
        f'<text x="52" y="38" font-family="sans-serif" font-size="24">{escape(title)}</text>',
        f'<text x="18" y="295" transform="rotate(-90 18 295)" font-family="sans-serif" font-size="14">{escape(y_label)}</text>',
        '<text x="400" y="585" font-family="sans-serif" font-size="14">context length (log2 axis)</text>',
    ]
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = y_coordinate(value)
        elements.extend((
            f'<line x1="{left:.2f}" y1="{y:.2f}" x2="{right:.2f}" y2="{y:.2f}" stroke="#e5e5e5"/>',
            f'<text x="{left - 8:.2f}" y="{y + 4:.2f}" text-anchor="end" font-family="monospace" font-size="11">{value:.4g}</text>',
        ))
    x_ticks = sorted({point[0] for point in all_points})
    for value in x_ticks:
        x = x_coordinate(value)
        label = f"{int(value // 1024)}K" if value >= 1024 else f"{int(value)}"
        elements.extend((
            f'<line x1="{x:.2f}" y1="{top:.2f}" x2="{x:.2f}" y2="{bottom:.2f}" stroke="#f0f0f0"/>',
            f'<text x="{x:.2f}" y="{bottom + 22:.2f}" text-anchor="middle" font-family="monospace" font-size="10">{label}</text>',
        ))
    elements.extend((
        f'<line x1="{left:.2f}" y1="{bottom:.2f}" x2="{right:.2f}" y2="{bottom:.2f}" stroke="black"/>',
        f'<line x1="{left:.2f}" y1="{top:.2f}" x2="{left:.2f}" y2="{bottom:.2f}" stroke="black"/>',
    ))
    for index, (name, points) in enumerate(normalized.items()):
        color = colors[index % len(colors)]
        coordinates = [(x_coordinate(x), y_coordinate(y)) for x, y in points]
        path_data = " ".join(f"{'M' if point_index == 0 else 'L'} {x:.2f} {y:.2f}"
                             for point_index, (x, y) in enumerate(coordinates))
        elements.append(f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="1.8"/>')
        elements.extend(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"/>' for x, y in coordinates)
        legend_y = 86 + 25 * index
        elements.extend((
            f'<line x1="820" y1="{legend_y}" x2="846" y2="{legend_y}" stroke="{color}" stroke-width="2"/>',
            f'<text x="854" y="{legend_y + 4}" font-family="sans-serif" font-size="12">{escape(name)}</text>',
        ))
    elements.extend((f'<text x="52" y="620" font-family="sans-serif" font-size="12">{escape(note)}</text>', "</svg>"))
    return "".join(elements)


# Readability changes: one color per configuration (shades by method family) and one
# marker shape per batch size, so that all 45 series are identifiable from a 14-entry legend.
CONFIG_COLORS = {
    "tq_4bit_nc": "#08519c", "tq_k3v4_nc": "#3182bd", "tq_3bit_nc": "#6baed6",
    "k4v4": "#a63603", "k2v4": "#e6550d", "k2v2": "#fd8d3c",
    "kvq4": "#006d2c", "kvq3": "#31a354", "kvq2": "#74c476",
}
CONFIG_LABELS = {
    "tq_4bit_nc": "TQ-4bit", "tq_k3v4_nc": "TQ-k3v4", "tq_3bit_nc": "TQ-3bit",
    "k4v4": "KIVI-k4v4", "k2v4": "KIVI-k2v4", "k2v2": "KIVI-k2v2",
    "kvq4": "KVQuant-4", "kvq3": "KVQuant-3", "kvq2": "KVQuant-2",
}
BASE_CONTEXTS = ((4096, "4K"), (8192, "8K"), (16384, "16K"), (24576, "24K"), (32768, "32K"),
                 (49152, "48K"), (65536, "64K"), (98304, "96K"), (131071, "128K"))


def marker(batch: int, x: float, y: float, color: str, size: float = 3.4) -> str:
    if batch == 1:
        return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{size:.1f}" fill="{color}"/>'
    if batch == 2:
        return (f'<rect x="{x - size:.2f}" y="{y - size:.2f}" width="{2 * size:.2f}" '
                f'height="{2 * size:.2f}" fill="{color}"/>')
    if batch == 4:
        return (f'<path d="M {x:.2f} {y - size - 0.8:.2f} L {x + size + 0.6:.2f} {y + size:.2f} '
                f'L {x - size - 0.6:.2f} {y + size:.2f} Z" fill="{color}"/>')
    if batch == 8:
        return (f'<path d="M {x:.2f} {y - size - 0.8:.2f} L {x + size + 0.8:.2f} {y:.2f} '
                f'L {x:.2f} {y + size + 0.8:.2f} L {x - size - 0.8:.2f} {y:.2f} Z" fill="{color}"/>')
    return (f'<path d="M {x:.2f} {y + size + 0.8:.2f} L {x + size + 0.6:.2f} {y - size:.2f} '
            f'L {x - size - 0.6:.2f} {y - size:.2f} Z" fill="{color}"/>')


def render_readable(series) -> str:
    normalized = normalize(series)
    all_points, (y_min, y_max), (left, right, top, bottom), x_coordinate, y_coordinate = mapping(normalized)
    elements = [
        # Same 1100 x 640 drawing as the original; the viewBox crops the empty title and note bands.
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="545" viewBox="0 50 1100 545">',
        '<rect width="1100" height="640" fill="white"/>',
        '<text x="22" y="295" transform="rotate(-90 22 295)" text-anchor="middle" font-family="sans-serif" '
        'font-size="14">same-work ratio T_BF16 / T_method</text>',
        f'<text x="{(left + right) / 2:.2f}" y="575" text-anchor="middle" font-family="sans-serif" '
        'font-size="14">historical context length (tokens, log2 axis)</text>',
    ]
    tick = math.ceil(y_min * 10) / 10
    while tick <= y_max + 1e-12:
        y = y_coordinate(tick)
        elements.extend((
            f'<line x1="{left:.2f}" y1="{y:.2f}" x2="{right:.2f}" y2="{y:.2f}" stroke="#e5e5e5"/>',
            f'<text x="{left - 8:.2f}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{tick:.1f}</text>',
        ))
        tick = round(tick + 0.1, 10)
    for value, label in BASE_CONTEXTS:
        x = x_coordinate(value)
        elements.extend((
            f'<line x1="{x:.2f}" y1="{top:.2f}" x2="{x:.2f}" y2="{bottom:.2f}" stroke="#f0f0f0"/>',
            f'<line x1="{x:.2f}" y1="{bottom:.2f}" x2="{x:.2f}" y2="{bottom + 5:.2f}" stroke="black"/>',
            f'<text x="{x:.2f}" y="{bottom + 22:.2f}" text-anchor="middle" font-family="sans-serif" font-size="12">{label}</text>',
        ))
    elements.extend((
        f'<line x1="{left:.2f}" y1="{bottom:.2f}" x2="{right:.2f}" y2="{bottom:.2f}" stroke="black"/>',
        f'<line x1="{left:.2f}" y1="{top:.2f}" x2="{left:.2f}" y2="{bottom:.2f}" stroke="black"/>',
    ))
    for name, points in normalized.items():
        config, batch_text = name.split("/B")
        batch = int(batch_text)
        color = CONFIG_COLORS[config]
        coordinates = [(x_coordinate(x), y_coordinate(y)) for x, y in points]
        path_data = " ".join(f"{'M' if i == 0 else 'L'} {x:.2f} {y:.2f}" for i, (x, y) in enumerate(coordinates))
        elements.append(f'<path d="{path_data}" fill="none" stroke="{color}" stroke-width="1.5" stroke-opacity="0.9"/>')
        elements.extend(marker(batch, x, y, color) for x, y in coordinates)
    legend_x, y = 815, 80
    elements.append(f'<text x="{legend_x}" y="{y}" font-family="sans-serif" font-size="13" font-weight="bold">Configuration (color)</text>')
    for config, label in CONFIG_LABELS.items():
        y += 22
        elements.extend((
            f'<line x1="{legend_x}" y1="{y - 4}" x2="{legend_x + 28}" y2="{y - 4}" stroke="{CONFIG_COLORS[config]}" stroke-width="3"/>',
            f'<text x="{legend_x + 36}" y="{y}" font-family="sans-serif" font-size="12">{label}</text>',
        ))
    y += 34
    elements.append(f'<text x="{legend_x}" y="{y}" font-family="sans-serif" font-size="13" font-weight="bold">Batch size (marker)</text>')
    for batch in BATCH_SIZES:
        y += 22
        elements.extend((marker(batch, legend_x + 14, y - 4, "#444444", size=4.0),
                         f'<text x="{legend_x + 36}" y="{y}" font-family="sans-serif" font-size="12">B = {batch}</text>'))
    elements.append("</svg>")
    return "".join(elements) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--original", "--verify-original", dest="original", type=Path,
                        help="original SVG to compare against byte for byte")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    series = load_series(args.table)
    if args.original is not None:
        original = render_original(series, title=TITLE, y_label=FIELD, note=NOTE)
        stored = args.original.read_text(encoding="utf-8")
        print("original plot reproduced byte-for-byte:", original == stored)
        if original != stored:
            raise SystemExit(1)
    if args.output is not None:
        args.output.write_text(render_readable(series), encoding="utf-8")
        print("wrote", args.output)


if __name__ == "__main__":
    main()
