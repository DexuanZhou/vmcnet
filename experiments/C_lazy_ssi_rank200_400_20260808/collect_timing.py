#!/usr/bin/env python3
"""Collect steady timing and short stability for the lazy-SSI screen."""

import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path("/scratch/dexuan1/runs/C_lazy_ssi_rank200_400_20260808")
HERE = Path(__file__).resolve().parent
ARMS = (
    ("spring", 0, 1),
    ("r200_k1", 200, 1),
    ("r200_k10", 200, 10),
    ("r200_k20", 200, 20),
    ("r400_k1", 400, 1),
    ("r400_k10", 400, 10),
    ("r400_k20", 400, 20),
)

rows = []
for arm, rank, period in ARMS:
    path = ROOT / "metadata" / arm / "monitor.csv"
    with path.open(newline="") as handle:
        data = list(csv.DictReader(handle))
    steady = [row for row in data if int(row["epoch"]) >= 21]
    timestamps = [float(row["time"]) for row in steady]
    durations = [right - left for left, right in zip(timestamps, timestamps[1:])]
    variances = [float(row["variance"]) for row in data]
    refreshed = [
        float(row["subspace_refreshed"])
        for row in steady
        if math.isfinite(float(row["subspace_refreshed"]))
    ]
    rows.append(
        {
            "arm": arm,
            "rank": rank,
            "refresh_period": period,
            "steps": len(data),
            "mean_s_per_step": statistics.mean(durations),
            "std_s_per_step": statistics.pstdev(durations),
            "last20_online_variance_median": statistics.median(variances[-20:]),
            "all_finite": all(math.isfinite(value) for value in variances),
            "observed_refresh_fraction": (
                statistics.mean(refreshed) if refreshed else None
            ),
        }
    )

spring_time = rows[0]["mean_s_per_step"]
for row in rows:
    row["time_ratio_to_spring"] = row["mean_s_per_step"] / spring_time
    row["passes_time_gate"] = (
        row["arm"] != "spring"
        and row["time_ratio_to_spring"] <= 1.0
        and row["all_finite"]
    )

(HERE / "timing_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
lines = [
    "# C lazy-SSI 200-step timing screen",
    "",
    "| arm | rank | K | s/step | ratio to SPRING | last20 online variance | refresh fraction | gate |",
    "|---|---:|---:|---:|---:|---:|---:|:---:|",
]
for row in rows:
    refresh = row["observed_refresh_fraction"]
    lines.append(
        f"| {row['arm']} | {row['rank']} | {row['refresh_period']} | "
        f"{row['mean_s_per_step']:.6f} +/- {row['std_s_per_step']:.6f} | "
        f"{row['time_ratio_to_spring']:.3f}x | "
        f"{row['last20_online_variance_median']:.6f} | "
        f"{refresh if refresh is not None else 'n/a'} | "
        f"{'PASS' if row['passes_time_gate'] else 'FAIL'} |"
    )
(HERE / "TIMING_REPORT.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
