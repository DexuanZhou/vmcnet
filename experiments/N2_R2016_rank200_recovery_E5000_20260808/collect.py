#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/dexuan1/runs/N2_R2016_rank200_recovery_E5000_20260808")
HERE = Path(__file__).resolve().parent
ARMS = [
    "A_hard_rank200",
    "B_isotropic_floor",
    "C_rotating_EF",
    "D_momentum_floor",
]
POINTS = [1000, 5000]


def stable_seconds_per_step(run: Path) -> float:
    energy = run / "energy.txt"
    if not energy.exists():
        return math.nan
    # File mtimes include compilation.  Keep this definition aligned with the
    # existing N2 E5000 report and expose it as an approximate all-in rate.
    checkpoint = run / "checkpoints" / "5000.npz"
    if not checkpoint.exists():
        return math.nan
    return (checkpoint.stat().st_mtime - energy.stat().st_ctime) / 5000.0


rows = []
for arm in ARMS:
    run = ROOT / "train" / arm
    rate = stable_seconds_per_step(run)
    for step in POINTS:
        stats_path = ROOT / "frozen" / f"{arm}_e{step}" / "eval" / "statistics.json"
        if stats_path.exists():
            stats = json.loads(stats_path.read_text())
            rows.append(
                {
                    "arm": arm,
                    "step": step,
                    "energy": stats["average"],
                    "std_err": stats["std_err"],
                    "variance": stats["variance"],
                    "approx_train_s_per_step": rate,
                    "statistics_path": str(stats_path),
                }
            )

out = ROOT / "summary.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as handle:
    fieldnames = list(rows[0]) if rows else ["arm", "step"]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

report = [
    "# N2 rank-200 complement-recovery results",
    "",
    "| arm | step | frozen energy | stderr | frozen variance | approx s/step |",
    "|---|---:|---:|---:|---:|---:|",
]
for row in rows:
    report.append(
        f"| {row['arm']} | {row['step']} | {row['energy']:.9f} | "
        f"{row['std_err']:.3g} | {row['variance']:.9f} | "
        f"{row['approx_train_s_per_step']:.4f} |"
    )
(ROOT / "REPORT.md").write_text("\n".join(report) + "\n")
print(json.dumps(rows, indent=2))
print(out)
print(ROOT / "REPORT.md")
