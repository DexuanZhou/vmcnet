#!/usr/bin/env python3
"""Collect the pure suggestion-1 paired screen."""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path("/scratch/dexuan1/runs/C_anisotropic_matrix_history_E5000_20260812")
OUT = Path(__file__).resolve().parent
ARMS = ("current_only", "uniform_matrix", "anisotropic_matrix")
rows = []
for arm in ARMS:
    for seed in (0, 1):
        stats_path = ROOT / "frozen" / arm / f"seed{seed}" / "eval" / "statistics.json"
        monitor_path = ROOT / "metadata" / arm / f"seed{seed}" / "monitor.csv"
        if not stats_path.exists() or not monitor_path.exists():
            continue
        stats = json.loads(stats_path.read_text())
        with monitor_path.open() as handle:
            monitor = list(csv.DictReader(handle))
        clocks = np.asarray([float(row["wall_clock"]) for row in monitor])
        tail = monitor[-100:]
        rows.append(
            {
                "arm": arm,
                "seed": seed,
                "frozen_energy": float(stats["average"]),
                "frozen_sem": float(stats["std_err"]),
                "frozen_variance": float(stats["variance"]),
                "seconds_per_step": float(np.median(np.diff(clocks)[100:])),
                "eta_head": float(
                    np.nanmedian([float(row["matrix_eta_head"]) for row in tail])
                ),
                "eta_mean": float(
                    np.nanmedian([float(row["matrix_eta_mean"]) for row in tail])
                ),
                "eta_tail": float(
                    np.nanmedian([float(row["matrix_eta_tail"]) for row in tail])
                ),
            }
        )

(OUT / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
print("| arm | seed | frozen E | frozen variance | s/step | eta head/mean/tail |")
print("|---|---:|---:|---:|---:|---:|")
for row in rows:
    print(
        f"| {row['arm']} | {row['seed']} | {row['frozen_energy']:.9f} | "
        f"{row['frozen_variance']:.8f} | {row['seconds_per_step']:.4f} | "
        f"{row['eta_head']:.3f}/{row['eta_mean']:.3f}/{row['eta_tail']:.3f} |"
    )
