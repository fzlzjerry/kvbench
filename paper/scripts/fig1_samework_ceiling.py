"""Figure 1 (revised): same-work ratios faceted by family, and their modeled ceilings.

Row (a): measured wall-clock same-work ratio S = T_BF16 / T_method versus historical context,
one panel per compressed family, log y-axis (frozen Phase 16R host-wall table).
Row (b): the same ratios against the post-hoc modeled roofline ceiling S_roof (A = 1; BF16's
measured time x 1,792 GB/s / ideal traffic) from the sealed Part A artifact.

Inputs (read only):
  --table   Phase 16R host-wall closure same_work_ratios.parquet (root 5605558b...)
  --ceiling posthoc Part A artifact a1_roofline_ratios.csv (root 3dc84c64...)

Usage (repository root, with pyarrow available):
    python3 paper/scripts/fig1_samework_ceiling.py \
        --table artifacts/phase16/phase16-20260831t123029614620z-ec534d99-de80ac/wall-closure/same_work_ratios.parquet \
        --ceiling paper/posthoc/artifacts/posthoc-a-20261004t143713987904z-0641de4b-ab87cf/a1_roofline_ratios.csv \
        --output paper/figures/fig1_samework_ceiling.svg
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import pyarrow.parquet as pq

FAMILIES = (
    ("TurboQuant (as ported)", ("tq_4bit_nc", "tq_k3v4_nc", "tq_3bit_nc")),
    ("KIVI", ("k4v4", "k2v4", "k2v2")),
    ("KVQuant (as ported)", ("kvq4", "kvq3", "kvq2")),
)
COLORS = {
    "tq_4bit_nc": "#08519c", "tq_k3v4_nc": "#3182bd", "tq_3bit_nc": "#6baed6",
    "k4v4": "#a63603", "k2v4": "#e6550d", "k2v2": "#fd8d3c",
    "kvq4": "#006d2c", "kvq3": "#31a354", "kvq2": "#74c476",
}
LABELS = {
    "tq_4bit_nc": "TQ-4bit", "tq_k3v4_nc": "TQ-k3v4", "tq_3bit_nc": "TQ-3bit",
    "k4v4": "KIVI-k4v4", "k2v4": "KIVI-k2v4", "k2v2": "KIVI-k2v2",
    "kvq4": "KVQuant-4", "kvq3": "KVQuant-3", "kvq2": "KVQuant-2",
}
BATCHES = (1, 2, 4, 8, 16)
CONTEXT_TICKS = ((4096, "4K"), (8192, "8K"), (16384, "16K"), (32768, "32K"), (65536, "64K"),
                 (131071, "128K"))
Y_RANGE = (0.005, 1.0)
Y_TICKS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
Y_RANGE_B = (0.005, 3.0)
Y_TICKS_B = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
X2_RANGE = (1.0, 3.0)
X2_TICKS = (1.0, 1.5, 2.0, 2.5, 3.0)
FONT = 'font-family="Helvetica, Arial, sans-serif"'

WIDTH, LEFT, PANEL_W, GAP = 900, 64, 256, 30
ROW_A_TOP, ROW_B_TOP, PANEL_H = 30, 262, 168


def marker(batch: int, x: float, y: float, color: str, size: float = 3.3) -> str:
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


def tick_label(value: float) -> str:
    return f"{value:g}"


def panel_frame(x0: float, top: float, x_ticks, x_pos, y_pos, show_y: bool, y_ticks=Y_TICKS) -> list[str]:
    out = []
    for value in y_ticks:
        y = y_pos(value)
        out.append(f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x0 + PANEL_W:.1f}" y2="{y:.1f}" stroke="#e6e6e6"/>')
        if show_y:
            out.append(f'<text x="{x0 - 5:.1f}" y="{y + 4:.1f}" text-anchor="end" {FONT} '
                       f'font-size="12">{tick_label(value)}</text>')
    ticks = list(x_ticks)
    for position, (value, label) in enumerate(ticks):
        x = x_pos(value)
        anchor = "start" if position == 0 else "end" if position == len(ticks) - 1 else "middle"
        out.append(f'<line x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" y2="{top + PANEL_H:.1f}" stroke="#f0f0f0"/>')
        out.append(f'<text x="{x:.1f}" y="{top + PANEL_H + 15:.1f}" text-anchor="{anchor}" {FONT} '
                   f'font-size="12">{label}</text>')
    out.append(f'<rect x="{x0:.1f}" y="{top:.1f}" width="{PANEL_W}" height="{PANEL_H}" fill="none" '
               f'stroke="black"/>')
    unity = y_pos(1.0)
    out.append(f'<line x1="{x0:.1f}" y1="{unity:.1f}" x2="{x0 + PANEL_W:.1f}" y2="{unity:.1f}" '
               'stroke="#444444" stroke-dasharray="5 3"/>')
    return out


def render(series: dict, ceiling: list[dict]) -> str:
    height = ROW_B_TOP + PANEL_H + 82
    e = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
         f'viewBox="0 0 {WIDTH} {height}">', f'<rect width="{WIDTH}" height="{height}" fill="white"/>']
    ly0, ly1 = math.log10(Y_RANGE[0]), math.log10(Y_RANGE[1])
    lb0, lb1 = math.log10(Y_RANGE_B[0]), math.log10(Y_RANGE_B[1])
    lx0, lx1 = math.log2(4096), math.log2(131071)
    bx0, bx1 = math.log10(X2_RANGE[0]), math.log10(X2_RANGE[1])
    for row_top, row_label in ((ROW_A_TOP, "(a)"), (ROW_B_TOP, "(b)")):
        mid = row_top + PANEL_H / 2
        e.append(f'<text x="14" y="{mid:.1f}" transform="rotate(-90 14 {mid:.1f})" text-anchor="middle" '
                 f'{FONT} font-size="13">{row_label} S = T_BF16 / T_method</text>')
    for index, (title, configs) in enumerate(FAMILIES):
        x0 = LEFT + index * (PANEL_W + GAP)

        def ya(v, top=ROW_A_TOP):
            return top + PANEL_H - (math.log10(v) - ly0) / (ly1 - ly0) * PANEL_H

        def yb(v, top=ROW_B_TOP):
            return top + PANEL_H - (math.log10(v) - lb0) / (lb1 - lb0) * PANEL_H

        def xa(v, x0=x0):
            return x0 + (math.log2(v) - lx0) / (lx1 - lx0) * PANEL_W

        def xb(v, x0=x0):
            return x0 + (math.log10(v) - bx0) / (bx1 - bx0) * PANEL_W

        e.append(f'<text x="{x0 + PANEL_W / 2:.1f}" y="{ROW_A_TOP - 10:.1f}" text-anchor="middle" {FONT} '
                 f'font-size="13">{title}</text>')
        e += panel_frame(x0, ROW_A_TOP, CONTEXT_TICKS, xa, ya, index == 0)
        e += panel_frame(x0, ROW_B_TOP, [(v, tick_label(v)) for v in X2_TICKS], xb, yb, index == 0, Y_TICKS_B)
        # Ceiling reference y = x (a method running at its modeled ceiling).
        e.append(f'<line x1="{xb(1.0):.1f}" y1="{yb(1.0):.1f}" x2="{xb(3.0):.1f}" y2="{yb(3.0):.1f}" '
                 'stroke="#c00000" stroke-width="1.3"/>')
        if index == 0:
            e.append(f'<text x="{xb(1.55):.1f}" y="{yb(2.4):.1f}" text-anchor="end" {FONT} '
                     'font-size="11" fill="#c00000">S = S_roof</text>')
        for config in configs:
            color = COLORS[config]
            for batch in BATCHES:
                points = sorted(series.get((config, batch), []))
                if not points:
                    continue
                path = " ".join(f"{'M' if i == 0 else 'L'} {xa(x):.2f} {ya(y):.2f}"
                                for i, (x, y) in enumerate(points))
                e.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.2" '
                         'stroke-opacity="0.8"/>')
                e += [marker(batch, xa(x), ya(y), color, 2.8) for x, y in points]
            for row in ceiling:
                if row["method_config_id"] == config:
                    e.append(marker(int(row["batch_size"]), xb(row["s_roof_alg"]), yb(row["s_measured"]),
                                    color, 2.8))
    e.append(f'<text x="{LEFT + 1.5 * PANEL_W + GAP:.1f}" y="{ROW_A_TOP + PANEL_H + 32:.1f}" '
             f'text-anchor="middle" {FONT} font-size="13">historical context (tokens, log2 axis)</text>')
    e.append(f'<text x="{LEFT + 1.5 * PANEL_W + GAP:.1f}" y="{ROW_B_TOP + PANEL_H + 32:.1f}" '
             f'text-anchor="middle" {FONT} font-size="13">modeled ceiling S_roof '
             f'(post hoc; ideal traffic at peak DRAM bandwidth)</text>')
    legend_y = ROW_B_TOP + PANEL_H + 56
    x = LEFT
    for _, configs in FAMILIES:
        for config in configs:
            e.append(f'<rect x="{x:.1f}" y="{legend_y - 9:.1f}" width="11" height="9" fill="{COLORS[config]}"/>')
            e.append(f'<text x="{x + 15:.1f}" y="{legend_y:.1f}" {FONT} font-size="12">{LABELS[config]}</text>')
            x += 15 + 6.4 * len(LABELS[config]) + 9
        x += 6
    x = LEFT
    legend_y += 19
    for batch in BATCHES:
        e.append(marker(batch, x + 4, legend_y - 4, "#444444", 3.0))
        e.append(f'<text x="{x + 11:.1f}" y="{legend_y:.1f}" {FONT} font-size="12">B={batch}</text>')
        x += 46
    e.append("</svg>")
    return "".join(e)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--ceiling", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    series: dict = {}
    for row in pq.read_table(args.table).to_pylist():
        if row.get("calculated") and row.get("performance_only_ratio") is not None:
            key = (row["method_config_id"], int(row["batch_size"]))
            series.setdefault(key, []).append((float(row["historical_context"]),
                                               float(row["performance_only_ratio"])))
    ceiling = []
    with args.ceiling.open() as handle:
        for row in csv.DictReader(handle):
            if row["s_measured"] and row["s_roof_alg"]:
                ceiling.append({"method_config_id": row["method_config_id"], "batch_size": row["batch_size"],
                                "s_measured": float(row["s_measured"]), "s_roof_alg": float(row["s_roof_alg"])})
    values = [y for points in series.values() for _, y in points]
    if min(values) < Y_RANGE[0] or max(values) > Y_RANGE[1]:
        raise SystemExit("same-work ratio outside the plotted range")
    if any(not X2_RANGE[0] <= r["s_roof_alg"] <= X2_RANGE[1] for r in ceiling):
        raise SystemExit("ceiling outside the plotted range")
    args.output.write_text(render(series, ceiling))
    print(f"wrote {args.output}: {sum(len(p) for p in series.values())} measured ratios, {len(ceiling)} ceilings")


if __name__ == "__main__":
    main()
