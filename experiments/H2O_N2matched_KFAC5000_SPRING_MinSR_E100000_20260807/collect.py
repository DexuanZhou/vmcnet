#!/usr/bin/env python3
"""Collect completed H2O optimizer endpoints without assuming all arms survive."""

import json
from pathlib import Path


RUNS = {
    "SPRING_mu095": Path(
        "/scratch/dexuan1/runs/"
        "H2O_SPRING_mu095_lr002_after_kfac5000_E100000_20260807"
    ),
    "MinSR_seed0": Path(
        "/scratch/dexuan1/runs/"
        "H2O_MinSR_lr002_after_kfac5000_E100000_20260807"
    ),
    "MinSR_seed1": Path(
        "/scratch/dexuan1/runs/"
        "H2O_MinSR_lr002_after_kfac5000_E100000_seed1_20260807"
    ),
    "KFAC": Path(
        "/scratch/dexuan1/runs/"
        "H2O_KFAC_lr005_after_kfac5000_E100000_20260807"
    ),
}
OUT = Path(__file__).resolve().parent


def latest_checkpoint_epoch(run):
    epochs = []
    checkpoint_dir = run / "checkpoints"
    if checkpoint_dir.exists():
        for path in checkpoint_dir.glob("*.npz"):
            try:
                epochs.append(int(path.stem))
            except ValueError:
                continue
    return max(epochs, default=None)


rows = []
for method, run in RUNS.items():
    stats_path = run / "eval/statistics.json"
    row = {
        "method": method,
        "run": str(run),
        "latest_regular_checkpoint": latest_checkpoint_epoch(run),
        "completed_frozen_evaluation": stats_path.exists(),
    }
    if stats_path.exists():
        with stats_path.open() as handle:
            stats = json.load(handle)
        row.update(
            energy=float(stats["average"]),
            variance=float(stats["variance"]),
            std_err=float(stats["std_err"]),
        )
    nan_checkpoints = sorted((run / "checkpoints").glob("nans_*.npz"))
    row["nan_checkpoints"] = [path.name for path in nan_checkpoints]
    rows.append(row)

summary = {"rows": rows}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
lines = [
    "# H2O optimizer comparison status",
    "",
    "| method | latest checkpoint | frozen energy | frozen variance | status |",
    "|---|---:|---:|---:|---|",
]
for row in rows:
    if row["completed_frozen_evaluation"]:
        energy = f"{row['energy']:.10f}"
        variance = f"{row['variance']:.10f}"
        status = "complete"
    elif row["nan_checkpoints"]:
        energy = variance = "--"
        status = "NaN: " + ",".join(row["nan_checkpoints"])
    elif Path(row["run"]).exists():
        energy = variance = "--"
        status = "running/incomplete"
    else:
        energy = variance = "--"
        status = "not started"
    latest = row["latest_regular_checkpoint"]
    lines.append(
        f"| {row['method']} | {latest if latest is not None else '--'} | "
        f"{energy} | {variance} | {status} |"
    )
(OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
