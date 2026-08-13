#!/usr/bin/env python3
"""Collect frozen and Grassmann telemetry for the four-arm screen."""

import csv
import json
import os
from pathlib import Path

import numpy as np


DEFAULT_ROOT = (
    Path(os.environ.get("SCRATCH", f"/tmp/{os.environ.get('USER', 'user')}"))
    / "runs"
    / "C_grassmann_ritz_E1000_20260812"
)
ROOT = Path(os.environ.get("C_GRASSMANN_ROOT", str(DEFAULT_ROOT)))
ARMS = (
    ("alpha100", 1.0),
    ("alpha075", 0.75),
    ("alpha050", 0.5),
    ("alpha025", 0.25),
)
rows = []
for arm, alpha in ARMS:
    stats_path = ROOT / "frozen" / arm / "eval" / "statistics.json"
    monitor_path = ROOT / "metadata" / arm / "monitor.csv"
    if not stats_path.exists() or not monitor_path.exists():
        continue
    stats = json.loads(stats_path.read_text())
    with monitor_path.open() as handle:
        telemetry = list(csv.DictReader(handle))

    def median(name, start, stop):
        values = [
            float(row[name])
            for row in telemetry[start:stop]
            if np.isfinite(float(row[name]))
        ]
        return float(np.median(values)) if values else float("nan")

    wall = float(telemetry[-1]["wall_clock"]) - float(
        telemetry[0]["wall_clock"]
    )
    rows.append(
        {
            "arm": arm,
            "alpha": alpha,
            "frozen_energy": stats["average"],
            "frozen_sem": stats["std_err"],
            "frozen_variance": stats["variance"],
            "train_seconds_per_step": wall / max(len(telemetry) - 1, 1),
            "overlap_mean_early": median("overlap_mean", 1, 101),
            "overlap_mean_middle": median("overlap_mean", 450, 550),
            "overlap_mean_late": median("overlap_mean", 900, 1000),
            "overlap_min_late": median("overlap_min", 900, 1000),
            "gradient_capture_late": median(
                "gradient_capture", 900, 1000
            ),
        }
    )

ROOT.mkdir(parents=True, exist_ok=True)
(ROOT / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
if rows:
    with (ROOT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
print(json.dumps(rows, indent=2))
