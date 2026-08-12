#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np


RUN = Path("/scratch/dexuan1/runs/N2_cluster_snr_rank200_20260809/health_e200")


def tail(name):
    values = np.atleast_1d(np.loadtxt(RUN / f"{name}.txt"))
    return float(np.median(values[100:])), float(np.min(values)), float(np.max(values))


names = (
    "wssr_envelope_snr_cluster_count",
    "wssr_envelope_snr_weight_min",
    "wssr_envelope_snr_weight_mean",
    "wssr_envelope_snr_weight_max",
    "wssr_envelope_snr_weak_boundary_weight",
    "wssr_envelope_snr_strong_boundary_weight",
    "wssr_envelope_snr_cross_force_cosine",
    "wssr_envelope_snr_retained_update_mass",
    "wssr_envelope_snr_update_cosine_baseline",
    "wssr_diag_finite",
    "variance",
    "energy",
)
summary = {name: dict(zip(("tail_median", "minimum", "maximum"), tail(name))) for name in names}
weak = summary["wssr_envelope_snr_weak_boundary_weight"]["tail_median"]
strong = summary["wssr_envelope_snr_strong_boundary_weight"]["tail_median"]
mass = summary["wssr_envelope_snr_retained_update_mass"]["tail_median"]
cosine = summary["wssr_envelope_snr_update_cosine_baseline"]["tail_median"]
finite = summary["wssr_diag_finite"]["minimum"] == 1.0
summary["health_gate"] = {
    "finite": finite,
    "weak_weight_below_strong": weak < strong,
    "retained_update_mass_at_least_0p25": mass >= 0.25,
    "candidate_baseline_cosine_at_least_0p8": cosine >= 0.8,
}
summary["healthy"] = bool(finite and weak < strong and mass >= 0.25 and cosine >= 0.8)
(RUN.parent / "health_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
