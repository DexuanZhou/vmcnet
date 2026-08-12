#!/usr/bin/env python3
"""Collect the preregistered 200-step dual-residual screening endpoint."""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(
    "/scratch/dexuan1/runs/C_galerkin_residual_recurrence_20260803"
)
MONITOR = ROOT / "metadata_train/prescreen_r800_dual/seed0/monitor.csv"
rows = list(csv.DictReader(MONITOR.open()))
if len(rows) != 200:
    raise RuntimeError(f"expected 200 monitor rows, found {len(rows)}")


def values(field):
    result = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise RuntimeError(f"nonfinite {field}")
    return result


difference = values("correction_relative_difference")
times = values("time")
summary = {
    "steps": len(rows),
    "correction_relative_difference": {
        "median": float(np.median(difference)),
        "p25": float(np.quantile(difference, 0.25)),
        "p75": float(np.quantile(difference, 0.75)),
        "p95": float(np.quantile(difference, 0.95)),
        "max": float(np.max(difference)),
    },
    "sample_residual_norm_median": float(
        np.median(values("sample_residual_norm"))
    ),
    "correctable_residual_norm_median": float(
        np.median(values("correctable_residual_norm"))
    ),
    "correctable_residual_ratio_median": float(
        np.median(values("correctable_residual_ratio"))
    ),
    "active_rank_median": float(np.median(values("active_rank"))),
    "hypothetical_gradient_drift_beta_median": float(
        np.median(values("hypothetical_gradient_drift_beta"))
    ),
    # Drop the first compiled step. Each timestamp is taken after device sync.
    "seconds_per_step_postcompile": float((times[-1] - times[0]) / 199.0),
}
threshold = 1e-3
summary["decision"] = {
    "threshold": threshold,
    "median_below_threshold": (
        summary["correction_relative_difference"]["median"] < threshold
    ),
    "next": (
        "run_full_only_and_compare_existing_rank_coordinate"
        if summary["correction_relative_difference"]["median"] < threshold
        else "run_full_P0_as_formal_complete_contrast"
    ),
}
(ROOT / "prescreen_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
print(json.dumps(summary, indent=2))
