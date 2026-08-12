#!/usr/bin/env python3
"""Formal E500 log-error analysis from collected curves and run metadata."""

from __future__ import annotations

import csv
import html
import math
from collections import defaultdict
from functools import cmp_to_key
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = {"C": -37.84471, "N2eq": -109.5388}
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
          "#56B4E9", "#000000", "#8C564B", "#17BECF", "#7F7F7F"]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rolling_mean(values, window=50):
    out, buf, total = [], [], 0.0
    for value in values:
        buf.append(value)
        total += value
        if len(buf) > window:
            total -= buf.pop(0)
        out.append(total / len(buf))
    return out


def linreg(points):
    n = len(points)
    xbar = sum(x for x, _ in points) / n
    ybar = sum(y for _, y in points) / n
    den = sum((x - xbar) ** 2 for x, _ in points)
    return sum((x - xbar) * (y - ybar) for x, y in points) / den


def rank_cmp(a, b):
    # If closed 2-SEM intervals overlap, the more negative tail slope wins.
    overlap = max(a["lo"], b["lo"]) <= min(a["hi"], b["hi"])
    if overlap and a["slope"] != b["slope"]:
        return -1 if a["slope"] < b["slope"] else 1
    return -1 if a["error"] < b["error"] else (1 if a["error"] > b["error"] else 0)


def write_svg(system, series, path):
    width, height = 1080, 620
    left, right, top, bottom = 88, 250, 42, 70
    x0, x1, y0, y1 = left, width - right, top, height - bottom
    points = [(x, y) for s in series for x, y in s["points"]]
    xmin, xmax = min(x for x, _ in points), max(x for x, _ in points)
    ymin, ymax = min(y for _, y in points), max(y for _, y in points)
    ypad = 0.04 * (ymax - ymin)
    ymin, ymax = ymin - ypad, ymax + ypad
    sx = lambda x: x0 + (x - xmin) * (x1 - x0) / (xmax - xmin)
    sy = lambda y: y1 - (y - ymin) * (y1 - y0) / (ymax - ymin)
    p = [f'<rect width="{width}" height="{height}" fill="white"/>',
         f'<text x="{(x0+x1)/2:.1f}" y="25" text-anchor="middle" font-size="18">{system}: E500 log-error convergence (50-epoch moving mean)</text>',
         f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#222"/>',
         f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#222"/>']
    for i in range(6):
        f = i / 5
        xv, yv = xmin + f * (xmax-xmin), ymin + f * (ymax-ymin)
        xx, yy = sx(xv), sy(yv)
        p += [f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y1}" stroke="#e5e5e5"/>',
              f'<text x="{xx:.1f}" y="{y1+22}" text-anchor="middle" font-size="11">{xv:.0f}</text>',
              f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="#e5e5e5"/>',
              f'<text x="{x0-9}" y="{yy+4:.1f}" text-anchor="end" font-size="11">{yv:.2f}</text>']
    p += [f'<text x="{(x0+x1)/2:.1f}" y="{height-18}" text-anchor="middle" font-size="14">Epoch</text>',
          f'<text x="20" y="{(y0+y1)/2:.1f}" text-anchor="middle" font-size="14" transform="rotate(-90 20 {(y0+y1)/2:.1f})">log10(E - E_ref) [Ha]</text>']
    for i, s in enumerate(series):
        color = COLORS[i % len(COLORS)]
        coords = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in s["points"])
        spring = s["family"] == "spring"
        sw = 3.5 if spring else 1.6
        weight = "bold" if spring else "normal"
        ly = y0 + 16 + i * 21
        p += [f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round"/>',
              f'<line x1="{x1+20}" y1="{ly}" x2="{x1+43}" y2="{ly}" stroke="{color}" stroke-width="{sw}"/>',
              f'<text x="{x1+49}" y="{ly+4}" font-size="11" font-weight="{weight}">{html.escape(s["name"])}</text>']
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="sans-serif">{"".join(p)}</svg>\n', encoding="utf-8")


def main():
    curves = read_csv(HERE / "results" / "curves.csv")
    summary = read_csv(HERE / "results" / "summary.csv")
    complete = {(r["system"], r["run_name"]): r for r in summary if r["status"] == "complete"}
    grouped = defaultdict(list)
    for row in curves:
        key = (row["system"], row["run_name"])
        if key in complete and math.isfinite(float(row["energy"])):
            grouped[key].append((int(float(row["epoch"])), float(row["energy"])))

    metrics, plot_data = [], defaultdict(list)
    for (system, name), vals in grouped.items():
        vals.sort()
        epochs = [x for x, _ in vals]
        smooth = rolling_mean([e for _, e in vals], 50)
        logpoints = [(x, math.log10(e - REF[system])) for x, e in zip(epochs, smooth) if e > REF[system]]
        last_epoch = epochs[-1]
        fitpoints = [(x, y) for x, y in logpoints if x > last_epoch - 200]
        srow = complete[(system, name)]
        error = (float(srow["tail100_energy_mean"]) - REF[system]) * 1000
        two_sem = 2 * float(srow["tail100_energy_sem"]) * 1000
        metrics.append({"system": system, "run": name, "family": srow["family"],
                        "spectral_reg": srow["spectral_reg"], "learning_rate": srow["learning_rate"],
                        "eta": srow["eta"], "error": error, "two_sem": two_sem,
                        "lo": error-two_sem, "hi": error+two_sem,
                        "slope": linreg(fitpoints), "fit_n": len(fitpoints),
                        "fit_start": min(x for x, _ in fitpoints), "fit_end": max(x for x, _ in fitpoints)})
        plot_data[system].append({"name": name, "family": srow["family"], "points": logpoints})

    figs = HERE / "report_figs"
    figs.mkdir(exist_ok=True)
    for system in ("C", "N2eq"):
        plot_data[system].sort(key=lambda s: (s["family"] != "spring", s["name"]))
        write_svg(system, plot_data[system], figs / f"logerr_{system}.svg")

    fieldnames = ["system", "rank", "run", "family", "spectral_reg", "learning_rate", "eta",
                  "tail100_error_mHa", "two_sem_mHa", "tail_log10_slope_per_epoch",
                  "fit_n", "fit_epoch_start", "fit_epoch_end"]
    ranked = []
    for system in ("C", "N2eq"):
        rows = sorted((m for m in metrics if m["system"] == system), key=cmp_to_key(rank_cmp))
        for rank, m in enumerate(rows, 1):
            ranked.append({"system": system, "rank": rank, "run": m["run"], "family": m["family"],
                           "spectral_reg": m["spectral_reg"], "learning_rate": m["learning_rate"], "eta": m["eta"],
                           "tail100_error_mHa": f'{m["error"]:.6f}', "two_sem_mHa": f'{m["two_sem"]:.6f}',
                           "tail_log10_slope_per_epoch": f'{m["slope"]:.9g}', "fit_n": m["fit_n"],
                           "fit_epoch_start": m["fit_start"], "fit_epoch_end": m["fit_end"]})
    with (HERE / "results" / "e500_logerr_ranking.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(ranked)

    lines = ["# E500 formal log-error analysis", "",
             "Definition: endpoint = tail-100 mean energy minus reference; uncertainty = ±2 epoch-wise SEM. "
             "Tail slope is OLS on the final 200 epochs of the 50-epoch moving-mean log10 error. "
             "Ranking uses endpoint error, with overlapping closed ±2SEM intervals treated as ties and broken by the more negative slope.", ""]
    for system in ("C", "N2eq"):
        lines += [f"## {system}", "", "| Rank | Run | Tail-100 error ±2SEM (mHa) | Tail slope (log10 Ha / epoch) | Fit epochs |", "|---:|---|---:|---:|---:|"]
        for r in (x for x in ranked if x["system"] == system):
            lines.append(f'| {r["rank"]} | `{r["run"]}` | {float(r["tail100_error_mHa"]):.3f} ± {float(r["two_sem_mHa"]):.3f} | {float(r["tail_log10_slope_per_epoch"]):+.3e} | {r["fit_epoch_start"]}–{r["fit_epoch_end"]} ({r["fit_n"]}) |')
        lines += [""]
    (HERE / "e500_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote E500 figures, ranking CSV, and e500_analysis.md")


if __name__ == "__main__":
    main()
