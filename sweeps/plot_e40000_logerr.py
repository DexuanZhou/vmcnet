#!/usr/bin/env python3
"""Plot the two completed C E40000 log-error trajectories."""
import html
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path("/scratch/dexuan1/runs/e40000/C")
REF = -37.84471
RUNS = [
    ("SPRING (lr=0.02)", "spring_lr02", "#D55E00", 3.0),
    ("WSSR (lr=0.002, eta=0.2, tikhonov)", "wssr_eta02_tik", "#0072B2", 2.4),
]


def rolling(values, window=50):
    out, buf, total = [], [], 0.0
    for value in values:
        buf.append(value)
        total += value
        if len(buf) > window:
            total -= buf.pop(0)
        out.append(total / len(buf))
    return out


def segments(points):
    result, current = [], []
    for point in points:
        if point is None:
            if current:
                result.append(current)
                current = []
        else:
            current.append(point)
    if current:
        result.append(current)
    return result


def main():
    series = []
    for label, run, color, width in RUNS:
        energies = [float(x) for x in (ROOT / run / "energy.txt").read_text().splitlines()]
        smooth = rolling(energies)
        points = [
            (epoch, math.log10(energy - REF)) if energy > REF else None
            for epoch, energy in enumerate(smooth, 1)
        ]
        series.append((label, color, width, segments(points), sum(p is None for p in points)))

    W, H = 1080, 620
    left, right, top, bottom = 90, 300, 48, 72
    x0, x1, y0, y1 = left, W - right, top, H - bottom
    valid = [p for _, _, _, segs, _ in series for seg in segs for p in seg]
    xmin, xmax = 1, 40000
    ymin, ymax = min(y for _, y in valid), max(y for _, y in valid)
    pad = 0.04 * (ymax - ymin)
    ymin, ymax = ymin - pad, ymax + pad
    sx = lambda x: x0 + (x - xmin) * (x1 - x0) / (xmax - xmin)
    sy = lambda y: y1 - (y - ymin) * (y1 - y0) / (ymax - ymin)
    parts = [
        f'<rect width="{W}" height="{H}" fill="white"/>',
        f'<text x="{(x0+x1)/2}" y="27" text-anchor="middle" font-size="18">C E40000 log-error convergence (50-epoch moving mean)</text>',
        f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#222"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#222"/>',
    ]
    for i in range(6):
        f = i / 5
        xv, yv = xmin + f * (xmax - xmin), ymin + f * (ymax - ymin)
        xx, yy = sx(xv), sy(yv)
        parts += [
            f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y1}" stroke="#e5e5e5"/>',
            f'<text x="{xx:.1f}" y="{y1+22}" text-anchor="middle" font-size="11">{xv:.0f}</text>',
            f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="#e5e5e5"/>',
            f'<text x="{x0-8}" y="{yy+4:.1f}" text-anchor="end" font-size="11">{yv:.2f}</text>',
        ]
    parts += [
        f'<text x="{(x0+x1)/2}" y="{H-20}" text-anchor="middle" font-size="14">Epoch</text>',
        f'<text x="20" y="{(y0+y1)/2}" text-anchor="middle" font-size="14" transform="rotate(-90 20 {(y0+y1)/2})">log10(E − E_ref) [Ha]</text>',
    ]
    for i, (label, color, width, segs, omitted) in enumerate(series):
        for seg in segs:
            coords = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in seg)
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"/>')
        ly = y0 + 20 + i * 48
        parts += [
            f'<line x1="{x1+20}" y1="{ly}" x2="{x1+48}" y2="{ly}" stroke="{color}" stroke-width="{width}"/>',
            f'<text x="{x1+55}" y="{ly+4}" font-size="11">{html.escape(label)}</text>',
            f'<text x="{x1+55}" y="{ly+20}" font-size="10" fill="#666">undefined points omitted: {omitted}</text>',
        ]
    parts.append(f'<text x="{x1+20}" y="{y1-18}" font-size="10" fill="#666">E_ref = −37.84471 Ha</text>')
    out = HERE / "report_figs" / "logerr_C_e40000.svg"
    out.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="sans-serif">{"".join(parts)}</svg>\n')
    print(out)


if __name__ == "__main__":
    main()
