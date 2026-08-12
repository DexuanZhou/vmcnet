#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path("/scratch/dexuan1/runs/N2_R2016_EF_device_E1000_20260807")
HERE = Path("/scratch/dexuan1/vmcnet/experiments/N2_R2016_EF_device_E1000_20260807")
ARMS = ("A_vanilla", "B_EF_mu095", "SPRING_mu095")
STEPS = (500, 1000)


def series(arm, metric):
    path = ROOT / "train" / arm / f"{metric}.txt"
    if not path.exists():
        return None
    return np.loadtxt(path, ndmin=1)


def at(values, step):
    if values is None or len(values) < step:
        return math.nan
    return float(values[step - 1])


def median_window(values, step, width=100):
    if values is None or len(values) < step:
        return math.nan
    return float(np.median(values[max(0, step - width):step]))


rows = []
for arm in ARMS:
    loaded = {
        name: series(arm, name)
        for name in (
            "wssr_residual_norm_reduction",
            "wssr_error_feedback_norm",
            "wssr_error_feedback_uncapped_norm",
            "wssr_error_feedback_cap_norm",
            "wssr_error_feedback_to_sample_ratio",
            "wssr_error_feedback_to_parameter_conflict_ratio",
            "wssr_error_feedback_clip_count",
            "wssr_error_feedback_clip_increment",
        )
    }
    for step in STEPS:
        stats_path = ROOT / "frozen" / f"{arm}_e{step}" / "eval" / "statistics.json"
        if not stats_path.exists():
            raise FileNotFoundError(stats_path)
        stats = json.loads(stats_path.read_text())
        clip_count = at(loaded["wssr_error_feedback_clip_count"], step)
        row = {
            "arm": arm,
            "step": step,
            "frozen_energy": float(stats["average"]),
            "frozen_energy_stderr": float(stats["std_err"]),
            "frozen_variance": float(stats["variance"]),
            "RR_sample_exact": at(loaded["wssr_residual_norm_reduction"], step),
            "RR_sample_last100_median": median_window(
                loaded["wssr_residual_norm_reduction"], step
            ),
            "feedback_norm_exact": at(loaded["wssr_error_feedback_norm"], step),
            "feedback_uncapped_norm_exact": at(
                loaded["wssr_error_feedback_uncapped_norm"], step
            ),
            "feedback_cap_norm_exact": at(
                loaded["wssr_error_feedback_cap_norm"], step
            ),
            "feedback_to_sample_exact": at(
                loaded["wssr_error_feedback_to_sample_ratio"], step
            ),
            "feedback_to_sample_last100_median": median_window(
                loaded["wssr_error_feedback_to_sample_ratio"], step
            ),
            "feedback_to_parameter_conflict_exact": at(
                loaded["wssr_error_feedback_to_parameter_conflict_ratio"], step
            ),
            "feedback_to_parameter_conflict_last100_median": median_window(
                loaded["wssr_error_feedback_to_parameter_conflict_ratio"], step
            ),
            "clip_count": clip_count,
            "clip_frequency": clip_count / step if math.isfinite(clip_count) else math.nan,
            "clip_last100_frequency": (
                float(np.mean(loaded["wssr_error_feedback_clip_increment"][step - 100:step]))
                if loaded["wssr_error_feedback_clip_increment"] is not None
                and len(loaded["wssr_error_feedback_clip_increment"]) >= step
                else math.nan
            ),
        }
        rows.append(row)

HERE.mkdir(parents=True, exist_ok=True)
with (HERE / "results.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

by_key = {(row["arm"], row["step"]): row for row in rows}
lines = [
    "# N2 rank-800 error-feedback E1000 result",
    "",
    "All arms start from the same KFAC-pre5000 checkpoint with 1000 training walkers.",
    "Frozen endpoints use 2000 fresh walkers, 10000 burn-in steps and 2000 measurements.",
    "",
    "| arm | endpoint | frozen energy (Ha) | stderr | frozen variance (Ha^2) | RR_sample last100 median | m/zeta last100 median | cap frequency |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    def fmt(value, digits=6):
        return "--" if not math.isfinite(value) else f"{value:.{digits}f}"
    lines.append(
        f"| {row['arm']} | {row['step']} | {row['frozen_energy']:.9f} | "
        f"{row['frozen_energy_stderr']:.6f} | {row['frozen_variance']:.9f} | "
        f"{fmt(row['RR_sample_last100_median'])} | "
        f"{fmt(row['feedback_to_sample_last100_median'])} | "
        f"{fmt(row['clip_frequency'], 4)} |"
    )

lines.extend(["", "## Paired variance comparisons", ""])
for step in STEPS:
    a = by_key[("A_vanilla", step)]["frozen_variance"]
    b = by_key[("B_EF_mu095", step)]["frozen_variance"]
    spring = by_key[("SPRING_mu095", step)]["frozen_variance"]
    lines.append(
        f"- E{step}: EF versus vanilla improvement = {(a-b)/a*100:.2f}%; "
        f"EF/SPRING variance ratio = {b/spring:.3f}."
    )

b1000 = by_key[("B_EF_mu095", 1000)]
gate = b1000["clip_frequency"] < 0.05
lines.extend(
    [
        "",
        "## EF health gate",
        "",
        f"- cumulative cap frequency at E1000: {b1000['clip_frequency']:.4%}; "
        f"the requested <5% gate {'passes' if gate else 'fails'}.",
        "- `m/zeta` is reported because it defines this experiment's cap, but it compares parameter and sample spaces. The parameter-consistent `m/(O.T@zeta)` values are retained in results.csv.",
        "- This is a one-seed short screen; it is not a 100k accuracy result.",
        "",
    ]
)
(HERE / "RESULT.md").write_text("\n".join(lines))
print("\n".join(lines))
