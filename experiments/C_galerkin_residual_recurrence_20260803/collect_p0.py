#!/usr/bin/env python3
"""Collect P0 paired endpoints and apply the preregistered 15% gate."""

import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(
    "/scratch/dexuan1/runs/C_galerkin_residual_recurrence_20260803"
)
ARM = "p0_full_r800_mu099"
CHECKPOINTS = (1000, 2500, 5000)
RANK_COORDINATE = {
    0: {"energy": -37.843985875848766, "variance": 0.04922198650234436},
    1: {"energy": -37.84432710253048, "variance": 0.05219799935163433},
}


def load_frozen(seed, checkpoint):
    path = (
        ROOT
        / f"frozen/{ARM}/seed{seed}/epoch{checkpoint}/eval/statistics.json"
    )
    payload = json.loads(path.read_text())
    return {
        "energy": float(payload.get("energy", payload["average"])),
        "variance": float(payload["variance"]),
        "std_err": float(payload["std_err"]),
    }


def telemetry(seed):
    path = ROOT / f"metadata_train/{ARM}/seed{seed}/monitor.csv"
    rows = list(csv.DictReader(path.open()))
    if len(rows) != 5000:
        raise RuntimeError(f"seed {seed}: expected 5000 rows, found {len(rows)}")

    def vals(field):
        output = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        if not np.all(np.isfinite(output)):
            raise RuntimeError(f"seed {seed}: nonfinite {field}")
        return output

    timestamps = vals("time")
    return {
        "seconds_per_step_postcompile": float(
            (timestamps[-1] - timestamps[0]) / 4999.0
        ),
        "sample_residual_norm_median": float(
            np.median(vals("sample_residual_norm"))
        ),
        "correctable_residual_norm_median": float(
            np.median(vals("correctable_residual_norm"))
        ),
        "correctable_residual_ratio_median": float(
            np.median(vals("correctable_residual_ratio"))
        ),
        "correction_relative_difference_median": float(
            np.median(vals("correction_relative_difference"))
        ),
        "correction_relative_difference_p95": float(
            np.quantile(vals("correction_relative_difference"), 0.95)
        ),
        "active_rank_median": float(np.median(vals("active_rank"))),
        "hypothetical_gradient_drift_beta_median": float(
            np.median(vals("hypothetical_gradient_drift_beta"))
        ),
    }


summary = {
    "stage": "P0",
    "arm": ARM,
    "configuration": {
        "rank": 800,
        "mu": 0.99,
        "residual_evaluation": "full_current_batch",
        "eta_S": 0.0,
        "error_feedback": False,
    },
    "seeds": {},
}
for seed in (0, 1):
    curve = {str(cp): load_frozen(seed, cp) for cp in CHECKPOINTS}
    final = curve["5000"]
    summary["seeds"][str(seed)] = {
        "light_frozen_curve": curve,
        "rank_coordinate_reference_E5000": RANK_COORDINATE[seed],
        "full_over_rank_coordinate_variance": (
            final["variance"] / RANK_COORDINATE[seed]["variance"]
        ),
        "paired_variance_improvement_fraction": (
            1.0 - final["variance"] / RANK_COORDINATE[seed]["variance"]
        ),
        "telemetry": telemetry(seed),
    }

old_geo = math.sqrt(
    RANK_COORDINATE[0]["variance"] * RANK_COORDINATE[1]["variance"]
)
new_geo = math.sqrt(
    summary["seeds"]["0"]["light_frozen_curve"]["5000"]["variance"]
    * summary["seeds"]["1"]["light_frozen_curve"]["5000"]["variance"]
)
improvement = 1.0 - new_geo / old_geo
summary["paired_geometric_mean"] = {
    "rank_coordinate_variance": old_geo,
    "full_current_batch_variance": new_geo,
    "improvement_fraction": improvement,
}
summary["decision"] = {
    "threshold_improvement_fraction": 0.15,
    "pass": improvement >= 0.15,
    "next": "continue_P0b_P1_P2_P3" if improvement >= 0.15 else "stop",
}
(ROOT / "p0_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
