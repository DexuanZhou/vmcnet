#!/usr/bin/env python3
"""Collect matched function-space C training and frozen-evaluation results."""

import csv
import json
import math
from pathlib import Path

import numpy as np


root = Path("/scratch/dexuan1/runs/C_function_space_matched_E5000_20260809")
records = []
for method in ("spring", "wssr_rank1600"):
    prefix = "spring" if method == "spring" else "wssr"
    for seed in (0, 1):
        train = root / "train" / method / f"seed{seed}"
        requested_eval = root / "eval" / method / f"seed{seed}"
        candidates = [
            requested_eval / "eval",
            requested_eval.with_name(requested_eval.name + "_1") / "eval",
        ]
        completed = [path for path in candidates if (path / "statistics.json").is_file()]
        if len(completed) != 1:
            raise RuntimeError(
                f"expected one completed evaluation for {method} seed {seed}, "
                f"found {completed}"
            )
        evaluation = completed[0]
        with (evaluation / "statistics.json").open() as handle:
            stats = json.load(handle)
        active_key = f"{prefix}_function_norm_constraint_active"
        scale_key = f"{prefix}_function_norm_constraint_scale"
        norm_key = f"{prefix}_function_update_norm_constrained"
        raw_norm_key = f"{prefix}_function_update_norm_unconstrained"
        active = np.atleast_1d(np.loadtxt(train / f"{active_key}.txt"))
        scales = np.atleast_1d(np.loadtxt(train / f"{scale_key}.txt"))
        norms = np.atleast_1d(np.loadtxt(train / f"{norm_key}.txt"))
        raw_norms = np.atleast_1d(np.loadtxt(train / f"{raw_norm_key}.txt"))
        training_rows = int(np.atleast_1d(np.loadtxt(train / "energy.txt")).size)
        wall = float(
            (
                root
                / "metadata_train"
                / method
                / f"seed{seed}"
                / "wall_time_seconds.txt"
            ).read_text()
        )
        records.append(
            {
                "method": method,
                "seed": seed,
                "frozen_energy": float(stats["average"]),
                "frozen_energy_sem": float(stats["std_err"]),
                "frozen_variance": float(stats["variance"]),
                "training_rows": training_rows,
                "constraint_active_fraction": float(active.mean()),
                "constraint_scale_median": float(np.median(scales)),
                "unconstrained_function_norm_median": float(
                    np.median(raw_norms)
                ),
                "function_norm_mean": float(norms.mean()),
                "function_norm_max_abs_error_from_c": float(
                    np.max(np.abs(norms[active > 0.5] - 8e-4))
                    if np.any(active > 0.5)
                    else math.nan
                ),
                "wall_time_seconds": wall,
                "wall_seconds_per_training_row": wall / training_rows,
            }
        )

by_method = {}
for method in ("spring", "wssr_rank1600"):
    subset = [row for row in records if row["method"] == method]
    by_method[method] = {
        "geometric_mean_variance": float(
            math.exp(np.mean(np.log([row["frozen_variance"] for row in subset])))
        ),
        "mean_energy": float(np.mean([row["frozen_energy"] for row in subset])),
        "mean_active_fraction": float(
            np.mean([row["constraint_active_fraction"] for row in subset])
        ),
        "mean_seconds_per_training_row": float(
            np.mean([row["wall_seconds_per_training_row"] for row in subset])
        ),
    }

summary = {
    "function_norm_radius": 8e-4,
    "records": records,
    "method_summary": by_method,
    "wssr_over_spring_variance_ratio": (
        by_method["wssr_rank1600"]["geometric_mean_variance"]
        / by_method["spring"]["geometric_mean_variance"]
    ),
    "interpretation_guard": (
        "The step radii are actually matched only on updates for which both "
        "constraint_active_fraction values are near one."
    ),
}
root.mkdir(parents=True, exist_ok=True)
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
with (root / "summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
print(json.dumps(summary, indent=2))
