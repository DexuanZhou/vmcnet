#!/usr/bin/env python3
"""Collect the two-seed late fresh-SPRING control."""

import json
import math
from pathlib import Path


NEW = Path("/scratch/dexuan1/runs/C_fresh_spring50k_control_20260807")
OLD = Path("/scratch/dexuan1/runs/C_late_spring50k_eta_drift_20260803")


def load(path):
    with path.open() as handle:
        stats = json.load(handle)
    return float(stats["average"]), float(stats["variance"])


rows = []
for seed in (0, 1):
    suffix = "frozen" if seed == 0 else "frozen_seed1"
    paths = {
        "native_spring": OLD / suffix / "spring55k" / "eval/statistics.json",
        "fresh_spring": NEW / f"frozen_seed{seed}" / "eval/statistics.json",
        "wssr_eta03": OLD / suffix / "eta03" / "eval/statistics.json",
    }
    values = {name: load(path) for name, path in paths.items()}
    baseline = values["native_spring"][1]
    for name, (energy, variance) in values.items():
        rows.append(
            {
                "seed": seed,
                "method": name,
                "energy": energy,
                "variance": variance,
                "variance_over_native_spring": variance / baseline,
            }
        )

aggregates = []
for method in ("native_spring", "fresh_spring", "wssr_eta03"):
    method_rows = [row for row in rows if row["method"] == method]
    energies = [row["energy"] for row in method_rows]
    variances = [row["variance"] for row in method_rows]
    aggregates.append(
        {
            "method": method,
            "mean_energy": sum(energies) / len(energies),
            "mean_variance": sum(variances) / len(variances),
            "geometric_mean_variance": math.prod(variances) ** (1.0 / len(variances)),
        }
    )

native = next(row for row in aggregates if row["method"] == "native_spring")
for row in aggregates:
    row["mean_variance_over_native_spring"] = (
        row["mean_variance"] / native["mean_variance"]
    )

summary = {"rows": rows, "aggregates": aggregates}
(NEW / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
lines = [
    "# C epoch-50000 fresh-SPRING-state control",
    "",
    "| seed | method | frozen energy | frozen variance | variance/native SPRING |",
    "|---:|---|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['seed']} | {row['method']} | {row['energy']:.10f} | "
        f"{row['variance']:.10f} | {row['variance_over_native_spring']:.4f} |"
    )
lines.extend(
    [
        "",
        "## Two-seed aggregate",
        "",
        "| method | mean energy | mean variance | geometric-mean variance | mean variance/native |",
        "|---|---:|---:|---:|---:|",
    ]
)
for row in aggregates:
    lines.append(
        f"| {row['method']} | {row['mean_energy']:.10f} | "
        f"{row['mean_variance']:.10f} | {row['geometric_mean_variance']:.10f} | "
        f"{row['mean_variance_over_native_spring']:.4f} |"
    )
lines.extend(
    [
        "",
        "## Interpretation",
        "",
        "Fresh SPRING used the same globally continuous late learning-rate schedule as native SPRING, "
        "but reset the SPRING optimizer state at epoch 50000. Its two-seed mean frozen variance is "
        "9.6% below native SPRING and 2.4% below WSSR eta=0.3. However, the per-seed ranking crosses: "
        "WSSR is best for seed 0, while native SPRING is best for seed 1. Thus the earlier seed-0-only "
        "claim that WSSR is intrinsically better in the late regime is not supported. Resetting the "
        "SPRING state accounts for the aggregate variance advantage at this precision, while no robust "
        "method ordering is established by only two frozen-evaluation seeds.",
    ]
)
(NEW / "REPORT.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
