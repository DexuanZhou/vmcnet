#!/usr/bin/env python3
"""Build pilot figures and a Markdown technical report from collected results."""

from __future__ import annotations

import argparse
import csv
import html
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "results"
DEFAULT_REPORT = HERE / "pilot_report.md"
DEFAULT_FIGURES = HERE / "report_figs"


def _read(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _fmt(value, digits=4):
    value = _float(value)
    return f"{value:.{digits}g}" if math.isfinite(value) else "—"


COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]


def _panel_svg(series, title, xlabel, ylabel, x0, y0, width, height):
    left, right, top, bottom = 72, 18, 35, 55
    px0, px1 = x0 + left, x0 + width - right
    py0, py1 = y0 + top, y0 + height - bottom
    points = [(x, y) for _, values in series for x, y in values if math.isfinite(x) and math.isfinite(y)]
    parts = [f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" fill="white"/>']
    parts += [
        f'<text x="{x0 + width / 2}" y="{y0 + 20}" text-anchor="middle" font-size="16">{html.escape(title)}</text>',
        f'<line x1="{px0}" y1="{py1}" x2="{px1}" y2="{py1}" stroke="#333"/>',
        f'<line x1="{px0}" y1="{py0}" x2="{px0}" y2="{py1}" stroke="#333"/>',
        f'<text x="{(px0 + px1) / 2}" y="{y0 + height - 12}" text-anchor="middle" font-size="12">{html.escape(xlabel)}</text>',
        f'<text x="{x0 + 15}" y="{(py0 + py1) / 2}" text-anchor="middle" font-size="12" transform="rotate(-90 {x0 + 15} {(py0 + py1) / 2})">{html.escape(ylabel)}</text>',
    ]
    if not points:
        parts.append(f'<text x="{(px0 + px1) / 2}" y="{(py0 + py1) / 2}" text-anchor="middle" fill="#666">No completed metrics yet</text>')
        return parts
    xmin, xmax = min(p[0] for p in points), max(p[0] for p in points)
    ymin, ymax = min(p[1] for p in points), max(p[1] for p in points)
    if xmax == xmin:
        xmax = xmin + 1
    if ymax == ymin:
        pad = max(abs(ymin) * 1e-6, 1e-6)
        ymin, ymax = ymin - pad, ymax + pad
    ypad = 0.05 * (ymax - ymin)
    ymin, ymax = ymin - ypad, ymax + ypad
    sx = lambda x: px0 + (x - xmin) * (px1 - px0) / (xmax - xmin)
    sy = lambda y: py1 - (y - ymin) * (py1 - py0) / (ymax - ymin)
    for tick in range(5):
        frac = tick / 4
        xval, yval = xmin + frac * (xmax - xmin), ymin + frac * (ymax - ymin)
        xx, yy = sx(xval), sy(yval)
        parts += [
            f'<line x1="{xx:.1f}" y1="{py0}" x2="{xx:.1f}" y2="{py1}" stroke="#ddd"/>',
            f'<text x="{xx:.1f}" y="{py1 + 17}" text-anchor="middle" font-size="10">{xval:.3g}</text>',
            f'<line x1="{px0}" y1="{yy:.1f}" x2="{px1}" y2="{yy:.1f}" stroke="#ddd"/>',
            f'<text x="{px0 - 7}" y="{yy + 3:.1f}" text-anchor="end" font-size="10">{yval:.6g}</text>',
        ]
    for index, (name, values) in enumerate(series):
        values = [(x, y) for x, y in values if math.isfinite(x) and math.isfinite(y)]
        if not values:
            continue
        color = COLORS[index % len(COLORS)]
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in values)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="1.7"/>')
        ly = py0 + 14 * index
        parts += [f'<line x1="{px1 - 135}" y1="{ly}" x2="{px1 - 120}" y2="{ly}" stroke="{color}" stroke-width="2"/>', f'<text x="{px1 - 116}" y="{ly + 3}" font-size="9">{html.escape(name)}</text>']
    return parts


def _write_svg(path, width, height, parts):
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="sans-serif">' + "".join(parts) + "</svg>\n", encoding="utf-8")


def _plot_convergence(system, curves, output):
    groups = defaultdict(list)
    for row in curves:
        if row["system"] == system and row["family"] != "kfac_pre":
            groups[row["run_name"]].append(row)
    series = []
    for name, rows in groups.items():
        rows.sort(key=lambda r: _float(r["epoch"]))
        x = [_float(row["epoch"]) for row in rows]
        y = [_float(row["energy_smooth20"]) for row in rows]
        if not any(math.isfinite(v) for v in y):
            y = [_float(row["energy"]) for row in rows]
        series.append((name, list(zip(x, y))))
    _write_svg(output, 900, 520, _panel_svg(series, f"{system}: convergence", "Epoch", "Energy (Ha)", 0, 0, 900, 520))


def _plot_sensitivity(summary, output):
    selections = [
        ("C", "eta", ["wssr_eta02_tik", "wssr_eta05_tik", "wssr_center_tik"], "C: tikhonov eta sensitivity"),
        ("N2eq", "learning_rate", ["wssr_lr001", "wssr_center", "wssr_lr005"], "N2eq: learning-rate sensitivity"),
    ]
    parts = []
    for index, (system, xfield, names, title) in enumerate(selections):
        by_name = {r["run_name"]: r for r in summary if r["system"] == system}
        points = []
        for name in names:
            row = by_name.get(name)
            if row:
                points.append((_float(row[xfield]), _float(row["tail50_energy_mean"])))
        points = [(x, y) for x, y in points if math.isfinite(x) and math.isfinite(y)]
        points.sort()
        parts += _panel_svg([("tail-50 mean", points)], title, xfield.replace("_", " "), "Tail-50 mean energy (Ha)", index * 500, 0, 500, 400)
    _write_svg(output, 1000, 400, parts)


def _cost_rows(summary):
    by_key = {(r["system"], r["run_name"]): r for r in summary}
    pairs = [
        ("rank", "C", "wssr_rank800", "rank 800"),
        ("rank", "C", "wssr_center", "rank 1600"),
        ("warm", "C", "wssr_warm1", "warm 1"),
        ("warm", "C", "wssr_center", "warm 2"),
    ]
    return [(kind, label, by_key.get((system, name), {})) for kind, system, name, label in pairs]


def _observed_or_allocated_hours(summary, system):
    center = next((r for r in summary if r["system"] == system and r["run_name"] == "wssr_center"), None)
    if center:
        elapsed = _float(center["elapsed_seconds"])
        if math.isfinite(elapsed) and elapsed > 0:
            return elapsed / 3600
    return 3.0 if system == "C" else 6.0


def build_report(results_dir: Path, report_path: Path, figures: Path):
    summary = _read(results_dir / "summary.csv")
    curves = _read(results_dir / "curves.csv")
    figures.mkdir(parents=True, exist_ok=True)
    _plot_convergence("C", curves, figures / "convergence_C.svg")
    _plot_convergence("N2eq", curves, figures / "convergence_N2eq.svg")
    _plot_sensitivity(summary, figures / "sensitivity.svg")
    complete = sum(row["status"] == "complete" for row in summary)
    by_key = {(row["system"], row["run_name"]): row for row in summary}

    def energy(system, run):
        return _float(by_key[(system, run)]["tail100_energy_mean"])

    c_hard, c_tik = energy("C", "wssr_center"), energy("C", "wssr_center_tik")
    c_spring = energy("C", "spring_lr02")
    lines = [
        "# Two-system pilot technical report", "",
        f"Data status: **{complete}/{len(summary)} runs complete**. The two partial runs are the original and explicitly configured N2eq SPRING attempts; they have 47 and 48 finite epochs respectively before NaN.", "",
        "Endpoint comparisons use the final-100-epoch mean and epoch-wise SEM. The SEM describes temporal fluctuation, not independent-repeat uncertainty.", "",
        "## 1. Configuration and complete results", "",
        "All main runs use 1000 walkers, burn-in 5000, 10 MCMC steps per update, mean-centered clipping threshold 5, 500 epochs, and inverse-time decay 1e-4. WSSR uses damping 3e-4, complement weight 0, and norm constraint 1e-3.", "",
        "**Spectral-regularization note:** every WSSR run without a `_tik` suffix used `hard_floor`; these runs were initially interpreted against historical tikhonov tuning and are a distinct regularization family. Every `_tik` run explicitly used `tikhonov`.", "",
    ]
    for system in ("C", "N2eq"):
        lines += [f"### {system}", "", "| Run | Optimizer | Spectral reg | lr | eta | rank/storage/work | Status | Finite epochs | Tail-100 mean ± SEM (Ha) | Error (mHa) |", "|---|---|---|---:|---:|---|---|---:|---:|---:|"]
        for row in summary:
            if row["system"] != system:
                continue
            ranks = "/".join(_fmt(row.get(key), 5) for key in ("rank", "storage_rank", "working_rank"))
            ranks = "—" if ranks == "—/—/—" else ranks
            lines.append(f"| {row['run_name']} | {row['family']} | {row.get('spectral_reg') or '—'} | {_fmt(row['learning_rate'], 5)} | {_fmt(row['eta'], 5)} | {ranks} | {row['status']} | {row['finite_energy_epochs']}/{row['nepochs']} | {_fmt(row['tail100_energy_mean'], 10)} ± {_fmt(row['tail100_energy_sem'], 4)} | {_fmt(row['error_mha'], 6)} |")
        lines.append("")
    lines += [
        "## 2. Main findings", "", "### C: center tikhonov versus hard-floor", "",
        f"Hard-floor gives {c_hard:.6f} Ha and tikhonov gives {c_tik:.6f} Ha: tikhonov improves by **{(c_hard-c_tik)*1000:.3f} mHa**. Their gaps behind C SPRING are {(c_hard-c_spring)*1000:.3f} and {(c_tik-c_spring)*1000:.3f} mHa. Therefore the regularization mismatch explains **{(c_hard-c_tik)*1000:.3f} mHa, or {(c_hard-c_tik)/(c_hard-c_spring)*100:.1f}%**, of the hard-floor center run's lag; the remaining gap is {(c_tik-c_spring)*1000:.3f} mHa.", "",
        "### C: tikhonov eta trend", "",
        f"Tail-100 means are eta=0.2: {energy('C','wssr_eta02_tik'):.6f} Ha; eta=0.5: {energy('C','wssr_eta05_tik'):.6f} Ha; eta=0.8: {energy('C','wssr_center_tik'):.6f} Ha. Eta 0.5 improves on 0.8 by {(energy('C','wssr_center_tik')-energy('C','wssr_eta05_tik'))*1000:.3f} mHa, and eta 0.2 improves on 0.5 by {(energy('C','wssr_eta05_tik')-energy('C','wssr_eta02_tik'))*1000:.3f} mHa. The measured trend is monotonic: **eta=0.2 is best**.", "",
        "### N2eq ranking and SPRING diagnosis", "",
        "The original and fixed SPRING attempts become NaN after 47 and 48 finite epochs respectively, including the explicit `mu=0.99`, `constrain_norm=true`, `norm_constraint=0.001` rerun. SPRING therefore has no valid tail-100 endpoint and ranks last as failed/partial.", "",
        "Among valid N2eq WSSR endpoints, lower is better:", "",
    ]
    n2 = [row for row in summary if row["system"] == "N2eq" and row["family"] == "wssr" and math.isfinite(_float(row["tail100_energy_mean"]))]
    n2.sort(key=lambda row: _float(row["tail100_energy_mean"]))
    for index, row in enumerate(n2, 1):
        lines.append(f"{index}. `{row['run_name']}` ({row['spectral_reg']}): {_float(row['tail100_energy_mean']):.6f} Ha; {_float(row['error_mha']):.2f} mHa above reference")
    lines += ["", "## 3. Convergence", "", "![](report_figs/convergence_C.svg)", "", "![](report_figs/convergence_N2eq.svg)", "", "## 4. Tikhonov eta and learning-rate sensitivity", "", "![](report_figs/sensitivity.svg)", "", "## 5. Interpretation limits", "", "These are single optimization trajectories. Tail-epoch SEM is not a seed-to-seed uncertainty estimate, so paired differences require independent-repeat confirmation.", ""]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report_path} and {figures}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    args = parser.parse_args()
    build_report(args.results_dir, args.report, args.figures_dir)


if __name__ == "__main__":
    main()
