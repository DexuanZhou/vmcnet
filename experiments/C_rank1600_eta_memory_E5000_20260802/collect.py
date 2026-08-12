#!/usr/bin/env python3
"""Collect the paired E5000 frozen-variance decision."""

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path("/scratch/dexuan1/runs/C_rank1600_eta_memory_E5000_20260802")
ARMS = ("eta03_control", "eta099_bias")
rows = []
for seed in (0, 1):
    values = {}
    for arm in ARMS:
        path = ROOT / "frozen" / arm / f"seed{seed}" / "eval" / "statistics.json"
        payload = json.loads(path.read_text())
        variance = float(payload["variance"])
        energy = float(payload.get("energy", payload["average"]))
        values[arm] = variance
        rows.append({"arm": arm, "seed": seed, "energy": energy, "variance": variance})
    rows.append(
        {
            "arm": "paired_ratio_candidate_over_control",
            "seed": seed,
            "energy": None,
            "variance": values["eta099_bias"] / values["eta03_control"],
        }
    )

ratios = [
    row["variance"]
    for row in rows
    if row["arm"] == "paired_ratio_candidate_over_control"
]
geometric_mean_ratio = float(math.exp(np.mean(np.log(ratios))))
if all(ratio < 1.0 for ratio in ratios) and geometric_mean_ratio <= 0.9:
    decision = "positive"
elif geometric_mean_ratio < 1.0:
    decision = "promising_but_inconclusive"
else:
    decision = "negative"

summary = {
    "primary_endpoint": "epoch5000_light_frozen_variance",
    "paired_ratios_candidate_over_control": ratios,
    "geometric_mean_ratio": geometric_mean_ratio,
    "relative_reduction": 1.0 - geometric_mean_ratio,
    "decision": decision,
    "positive_rule": "both ratios < 1 and geometric mean <= 0.9",
    "rows": rows,
}
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

lines = [
    "# C rank-1600 long-memory E5000 result",
    "",
    f"Decision: **{decision}**",
    "",
    f"Paired candidate/control variance ratios: {ratios}",
    f"Geometric mean ratio: {geometric_mean_ratio:.6f}",
    f"Relative variance reduction: {100.0 * (1.0-geometric_mean_ratio):.2f}%",
    "",
    "| arm | seed | energy | variance |",
    "|---|---:|---:|---:|",
]
for row in rows:
    energy = "" if row["energy"] is None else f'{row["energy"]:.9f}'
    lines.append(
        f'| {row["arm"]} | {row["seed"]} | {energy} | {row["variance"]:.9g} |'
    )
(ROOT / "report.md").write_text("\n".join(lines) + "\n")
print(json.dumps(summary, indent=2))
