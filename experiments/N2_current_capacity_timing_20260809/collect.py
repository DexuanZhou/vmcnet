#!/usr/bin/env python3
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/dexuan1/runs/N2_current_capacity_timing_tailfix_20260809")
rows = []
for rank in (200, 384, 512):
    arm = f"current_rank{rank}"
    run = ROOT / arm
    meta = ROOT / "metadata" / arm
    progress = np.genfromtxt(meta / "progress.tsv", delimiter="\t", names=True)
    times = np.atleast_1d(progress["time_ns"]).astype(float) / 1e9
    epochs = np.atleast_1d(progress["epochs"]).astype(int)
    memory = np.atleast_1d(progress["gpu_memory_mib"]).astype(float)
    keep = epochs >= 20
    unique = np.r_[True, np.diff(epochs) != 0] & keep
    t = times[unique]
    e = epochs[unique]
    if e.size < 2:
        seconds_per_step = float("nan")
    else:
        seconds_per_step = float((t[-1] - t[0]) / (e[-1] - e[0]))

    def tail(name):
        x = np.atleast_1d(np.loadtxt(run / f"{name}.txt"))
        return float(np.median(x[20:]))

    row = {
        "rank": rank,
        "seconds_per_step": seconds_per_step,
        "peak_device_memory_mib": float(np.max(memory)),
        "numerical_rank": tail("wssr_envelope_numerical_rank"),
        "gradient_capture_fraction": tail("wssr_envelope_gradient_capture_fraction"),
        "curvature_min": tail("wssr_envelope_projected_curvature_min"),
        "curvature_max": tail("wssr_envelope_projected_curvature_max"),
        "regularized_condition": tail("wssr_envelope_projected_curvature_condition"),
        "boundary_update_mass": tail("wssr_envelope_boundary_update_mass"),
        "spectral_tail_ratio_sigma_d_over_sigma_dplus1": tail(
            "wssr_envelope_spectral_tail_ratio"
        ),
        "spectral_tail_available": tail(
            "wssr_envelope_spectral_tail_available"
        ),
        "finite": tail("wssr_diag_finite"),
    }
    if row["spectral_tail_available"] != 1.0:
        raise RuntimeError(f"{arm}: d+1 spectral boundary was not recorded")
    rows.append(row)

(ROOT / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
with (ROOT / "summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(rows, indent=2))
