#!/usr/bin/env python3
"""Collect late-checkpoint WSSR eta-sweep frozen results."""

import json
from pathlib import Path

import numpy as np


ROOT = Path("/scratch/dexuan1/runs/C_late_spring50k_eta_drift_20260803")
ARMS = ("spring55k", "eta03", "eta08", "eta095")


def load(arm):
    path = ROOT / "frozen" / arm / "eval" / "statistics.json"
    with path.open() as handle:
        stats = json.load(handle)
    return {
        "energy": float(stats["average"]),
        "variance": float(stats["variance"]),
        "sem": float(stats["std_err"]),
        "iat": float(stats["integrated_autocorrelation"]),
        "source": str(path.parent),
    }


results = {arm: load(arm) for arm in ARMS}
spring_var = results["spring55k"]["variance"]
for arm in ARMS[1:]:
    results[arm]["variance_over_spring"] = results[arm]["variance"] / spring_var

etas = np.array([0.3, 0.8, 0.95])
variances = np.array([results[f"eta{tag}"]["variance"] for tag in ("03", "08", "095")])
if np.all(np.diff(variances) > 0):
    ordering = "longer_memory_monotonically_worse"
elif np.all(np.diff(variances) < 0):
    ordering = "late_ordering_reversed_longer_memory_better"
else:
    ordering = "late_ordering_flat_or_nonmonotone"

summary = {
    "question": "does_long_WSSR_memory_stop_being_toxic_at_SPRING_epoch50000",
    "source_checkpoint": (
        "/scratch/dexuan1/runs/C_spring_official_kfac1000_E100000_replicate2/"
        "checkpoints/50000.npz"
    ),
    "protocol": {
        "wssr_updates": 5000,
        "rank": 1600,
        "etas": etas.tolist(),
        "walkers": 1000,
        "reburn": False,
        "frozen_walkers": 1000,
        "frozen_burn": 5000,
        "frozen_measurements": 2000,
        "frozen_mcmc_steps": 10,
    },
    "results": results,
    "eta_variance_ordering": ordering,
}
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

lines = [
    "# C late-SPRING-checkpoint WSSR memory test",
    "",
    "| arm | frozen energy | frozen variance | variance / SPRING |",
    "|---|---:|---:|---:|",
]
for arm in ARMS:
    row = results[arm]
    ratio = row.get("variance_over_spring", 1.0)
    lines.append(
        f"| {arm} | {row['energy']:.10f} | {row['variance']:.10f} | {ratio:.4f} |"
    )
lines += ["", f"Eta ordering: **{ordering}**."]
(ROOT / "report.md").write_text("\n".join(lines) + "\n")
print(json.dumps(summary, indent=2))
