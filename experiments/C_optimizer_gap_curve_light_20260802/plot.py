#!/usr/bin/env python3
"""Generate a dependency-free SVG of the frozen-variance gap curve."""

import json
import math
from pathlib import Path


ROOT = Path("/scratch/dexuan1/runs/C_optimizer_gap_curve_light_20260802")
summary = json.loads((ROOT / "summary.json").read_text())
records = summary["records"]
shared = next(row for row in records if row["method"] == "shared_kfac_pre1000")


def series(method):
    rows = sorted(
        (row for row in records if row["method"] == method),
        key=lambda row: row["epoch"],
    )
    return (
        [shared["epoch"]] + [row["epoch"] for row in rows],
        [shared["raw_variance"]] + [row["raw_variance"] for row in rows],
    )


def esc(value):
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


width, height = 1100, 450
top, bottom = 55, 390
panels = [(75, 510), (635, 1025)]
elements = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
    '<rect width="100%" height="100%" fill="white"/>',
    '<style>text{font-family:Arial,sans-serif;fill:#222}.tick{font-size:12px}.label{font-size:15px}.title{font-size:19px;font-weight:600}.legend{font-size:13px}</style>',
    f'<text x="{width/2}" y="27" text-anchor="middle" class="title">C matched optimizer frozen-variance gap curve</text>',
]


def log_x(epoch, left, right):
    return left + (math.log10(epoch) - 3.0) / 2.0 * (right - left)


left, right = panels[0]
log_y_min, log_y_max = math.log10(0.003), math.log10(3.0)


def log_y(value):
    return bottom - (math.log10(value) - log_y_min) / (log_y_max - log_y_min) * (bottom - top)


elements += [
    f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222"/>',
    f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#222"/>',
]
for epoch, label in ((1000, "1k"), (5000, "5k"), (20000, "20k"), (50000, "50k"), (100000, "100k")):
    x = log_x(epoch, left, right)
    elements += [
        f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="#ddd"/>',
        f'<text x="{x:.1f}" y="410" text-anchor="middle" class="tick">{label}</text>',
    ]
for value in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
    y = log_y(value)
    elements += [
        f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#ddd"/>',
        f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" class="tick">{value:g}</text>',
    ]

colors = {"SPRING replicate2": "#1f77b4", "WSSR rank1600": "#d62728"}
for label, method in (("SPRING replicate2", "spring_replicate2"), ("WSSR rank1600", "wssr_rank1600_eta03_lr004")):
    xs, ys = series(method)
    points = [(log_x(x, left, right), log_y(y)) for x, y in zip(xs, ys)]
    elements.append(
        '<polyline fill="none" stroke="{}" stroke-width="2.5" points="{}"/>'.format(
            colors[label], " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        )
    )
    for x, y in points:
        elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors[label]}"/>')
elements += [
    f'<text x="{(left+right)/2}" y="438" text-anchor="middle" class="label">Checkpoint epoch</text>',
    f'<text x="18" y="{(top+bottom)/2}" text-anchor="middle" class="label" transform="rotate(-90 18 {(top+bottom)/2})">Frozen raw variance (Ha²)</text>',
]
for i, label in enumerate(colors):
    y = 72 + 20 * i
    elements += [
        f'<line x1="{left+18}" y1="{y}" x2="{left+45}" y2="{y}" stroke="{colors[label]}" stroke-width="3"/>',
        f'<text x="{left+52}" y="{y+4}" class="legend">{esc(label)}</text>',
    ]

left, right = panels[1]
ratio_max = 8.0


def ratio_y(value):
    return bottom - value / ratio_max * (bottom - top)


elements += [
    f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222"/>',
    f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#222"/>',
]
for epoch, label in ((5000, "5k"), (20000, "20k"), (50000, "50k"), (100000, "100k")):
    x = log_x(epoch, left, right)
    elements += [
        f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="#ddd"/>',
        f'<text x="{x:.1f}" y="410" text-anchor="middle" class="tick">{label}</text>',
    ]
for value in (0, 1, 2, 4, 6, 8):
    y = ratio_y(value)
    stroke = "#777" if value == 1 else "#ddd"
    dash = ' stroke-dasharray="5 4"' if value == 1 else ""
    elements += [
        f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{stroke}"{dash}/>',
        f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" class="tick">{value:g}</text>',
    ]
comparisons = summary["comparisons"]
points = [
    (log_x(row["epoch"], left, right), ratio_y(row["variance_ratio"]))
    for row in comparisons
]
elements.append(
    '<polyline fill="none" stroke="#d62728" stroke-width="2.5" points="{}"/>'.format(
        " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    )
)
for (x, y), row in zip(points, comparisons):
    elements += [
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#d62728"/>',
        f'<text x="{x:.1f}" y="{y-9:.1f}" text-anchor="middle" class="tick">{row["variance_ratio"]:.2f}×</text>',
    ]
elements += [
    f'<text x="{(left+right)/2}" y="438" text-anchor="middle" class="label">Checkpoint epoch</text>',
    f'<text x="575" y="{(top+bottom)/2}" text-anchor="middle" class="label" transform="rotate(-90 575 {(top+bottom)/2})">WSSR / SPRING variance</text>',
    '</svg>',
]
(ROOT / "gap_curve.svg").write_text("\n".join(elements) + "\n")
