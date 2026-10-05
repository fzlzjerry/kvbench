"""Redraw Figure 2 (eager-to-Graph latency ratio versus context) with manuscript names.

Source data: the eager/Graph pair table graph_ab_pairs.parquet in the eager-versus-Graph
experiment root (read only); `graph_ratio` is the eager/Graph wall-clock median ratio.
`render_original` is a verbatim copy of the plotting function that produced the original
figure (scripts/phase13_pilot.py::_svg_line_plot, called from
scripts/phase14_graph_ab.py::_plots); `--verify-original <svg>` checks that it reproduces
that file byte for byte. `render_readable` draws the same series, points, coordinate
mapping, and ticks with the color scheme of Figure 1 (one color per configuration, one
marker per batch size, manuscript names in the legend), the y-axis title "Eager / Graph
latency ratio", no in-figure title (the caption carries it), and no footer note; the
empty bands are cropped through the SVG viewBox.

Usage (from the repository root, with pyarrow available):
    python3 paper/scripts/fig2_graph_ab_ratio.py \
        --pairs <eager/Graph experiment root>/graph_ab_pairs.parquet \
        --verify-original <eager/Graph experiment root>/plots/graph-ratio-vs-context.svg \
        --output paper/figures/fig2_graph_ab_ratio.svg
"""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

# Arguments of the original call in scripts/phase14_graph_ab.py::_plots.
TITLE = "Graph ratio versus context"
KEY = "graph_ratio"
NOTE = "Proxy only; no direct launch-gap or quality claim"
CONFIG_LABELS = {
    "bf16": "BF16", "tq_4bit_nc": "TQ-4bit", "tq_k3v4_nc": "TQ-k3v4", "tq_3bit_nc": "TQ-3bit",
    "k4v4": "KIVI-k4v4", "k2v4": "KIVI-k2v4", "k2v2": "KIVI-k2v2",
    "kvq4": "KVQuant-4", "kvq3": "KVQuant-3", "kvq2": "KVQuant-2",
}
COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
          "#7f7f7f", "#bcbd22", "#17becf", "#3366cc", "#dc3912")


def load_series(pairs: Path) -> dict[str, list[tuple[float, float]]]:
    """Same series construction as the original call: stable pairs, one series per configuration and batch."""
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in pq.read_table(pairs).to_pylist():
        if row["pair_status"] == "stable":
            series[f"{row['method_config_id']}/B{row['batch_size']}"].append(
                (float(row["context_label"]), float(row[KEY])))
    return series


def escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_original(series, *, title: str, y_label: str, note: str) -> str:
    """Verbatim logic of the original plotting function (returns the SVG text)."""
    normalized: dict[str, list[tuple[float, float]]] = {}
    for name, points in series.items():
        finite = sorted(
            ((float(x), float(y)) for x, y in points
             if x > 0 and math.isfinite(float(x)) and math.isfinite(float(y))),
            key=lambda item: item[0])
        if finite:
            normalized[name] = finite
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
        color = COLORS[index % len(COLORS)]
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


# Figure 1 color scheme: one color per configuration (shades by method family), one marker per batch size.
CONFIG_COLORS = {
    "bf16": "#252525",
    "tq_4bit_nc": "#08519c", "tq_k3v4_nc": "#3182bd", "tq_3bit_nc": "#6baed6",
    "k4v4": "#a63603", "k2v4": "#e6550d", "k2v2": "#fd8d3c",
    "kvq4": "#006d2c", "kvq3": "#31a354", "kvq2": "#74c476",
}
Y_TITLE = "Eager / Graph latency ratio"


def marker(batch: int, x: float, y: float, color: str, size: float = 3.4) -> str:
    """Same marker shapes as Figure 1 (B = 1 circle, B = 4 triangle)."""
    if batch == 1:
        return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{size:.1f}" fill="{color}"/>'
    if batch == 4:
        return (f'<path d="M {x:.2f} {y - size - 0.8:.2f} L {x + size + 0.6:.2f} {y + size:.2f} '
                f'L {x - size - 0.6:.2f} {y + size:.2f} Z" fill="{color}"/>')
    raise ValueError(f"no marker for B = {batch}")


def render_readable(series) -> str:
    # Same normalization, coordinate mapping, and ticks as render_original.
    normalized = {}
    for name, points in series.items():
        finite = sorted(((float(x), float(y)) for x, y in points
                         if x > 0 and math.isfinite(float(x)) and math.isfinite(float(y))), key=lambda item: item[0])
        if finite:
            normalized[name] = finite
    all_points = [point for points in normalized.values() for point in points]
    x_min = min(math.log2(point[0]) for point in all_points)
    x_max = max(math.log2(point[0]) for point in all_points)
    y_min = min(point[1] for point in all_points)
    y_max = max(point[1] for point in all_points)
    padding = (y_max - y_min) * 0.08
    y_min -= padding
    y_max += padding
    left, right, top, bottom = 82.0, 790.0, 70.0, 520.0

    def x_coordinate(value: float) -> float:
        return left + (math.log2(value) - x_min) * (right - left) / (x_max - x_min)

    def y_coordinate(value: float) -> float:
        return bottom - (value - y_min) * (bottom - top) / (y_max - y_min)

    elements = [
        # Same 1100 x 640 drawing as the original; the viewBox crops the empty title and note bands.
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="545" viewBox="0 50 1100 545">',
        '<rect width="1100" height="640" fill="white"/>',
        f'<text x="22" y="{(top + bottom) / 2:.0f}" transform="rotate(-90 22 {(top + bottom) / 2:.0f})" text-anchor="middle" '
        f'font-family="sans-serif" font-size="14">{escape(Y_TITLE)}</text>',
        '<text x="400" y="585" font-family="sans-serif" font-size="14">context length (log2 axis)</text>',
    ]
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = y_coordinate(value)
        elements.extend((
            f'<line x1="{left:.2f}" y1="{y:.2f}" x2="{right:.2f}" y2="{y:.2f}" stroke="#e5e5e5"/>',
            f'<text x="{left - 8:.2f}" y="{y + 4:.2f}" text-anchor="end" font-family="monospace" font-size="11">{value:.4g}</text>',
        ))
    for value in sorted({point[0] for point in all_points}):
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
    batches = []
    for name, points in normalized.items():
        config, batch_text = name.split("/B")
        batch = int(batch_text)
        batches.append(batch)
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
    for batch in sorted(set(batches)):
        y += 22
        elements.extend((marker(batch, legend_x + 14, y - 4, "#444444", size=4.0),
                         f'<text x="{legend_x + 36}" y="{y}" font-family="sans-serif" font-size="12">B = {batch}</text>'))
    elements.append("</svg>")
    return "".join(elements) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", type=Path, required=True, help="graph_ab_pairs.parquet of the eager/Graph experiment")
    parser.add_argument("--verify-original", type=Path, help="original SVG to compare against byte for byte")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    series = load_series(args.pairs)
    if args.verify_original is not None:
        original = render_original(series, title=TITLE, y_label=KEY, note=NOTE)
        same = original.encode("utf-8") == args.verify_original.read_bytes()
        print("original plot reproduced byte-for-byte:", same, f"({len(series)} series)")
        if not same:
            raise SystemExit(1)
    if args.output is not None:
        args.output.write_text(render_readable(series), encoding="utf-8")
        print("wrote", args.output)


if __name__ == "__main__":
    main()
