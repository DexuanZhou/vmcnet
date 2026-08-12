#!/usr/bin/env python3
"""Collect the three 200-step delayed-subspace timing arms."""

import csv
import json
import statistics
from pathlib import Path


ROOT = Path("/scratch/dexuan1/runs/C_delayed_subspace_timing_20260807")
HERE = Path(__file__).resolve().parent
rows = []
for period in (1, 2, 5):
    arm = f"refresh{period}"
    path = ROOT / "metadata" / arm / "monitor.csv"
    with path.open(newline="") as handle:
        data = list(csv.DictReader(handle))
    steady = [row for row in data if int(row["epoch"]) >= 21]
    times = [float(row["time"]) for row in steady]
    durations = [right - left for left, right in zip(times, times[1:])]
    reductions = [
        float(row["residual_reduction_fraction"]) for row in steady
    ]
    refresh_values = [
        float(row["subspace_refreshed"])
        for row in steady
        if row["subspace_refreshed"].lower() != "nan"
    ]
    rows.append(
        {
            "period": period,
            "steps": len(data),
            "steady_duration_count": len(durations),
            "mean_s_per_step": statistics.mean(durations),
            "std_s_per_step": statistics.pstdev(durations),
            "median_residual_reduction_fraction": statistics.median(
                reductions
            ),
            "observed_refresh_fraction": (
                statistics.mean(refresh_values) if refresh_values else 1.0
            ),
        }
    )

(HERE / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
baseline = rows[0]["mean_s_per_step"]
lines = [
    "# Delayed-subspace 200-step timing result",
    "",
    "| refresh period | s/step mean | std | speedup vs period1 | "
    "median residual reduction | observed refresh fraction |",
    "|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['period']} | {row['mean_s_per_step']:.6f} | "
        f"{row['std_s_per_step']:.6f} | "
        f"{baseline / row['mean_s_per_step']:.3f}x | "
        f"{row['median_residual_reduction_fraction']:.6f} | "
        f"{row['observed_refresh_fraction']:.3f} |"
    )
(HERE / "REPORT.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
