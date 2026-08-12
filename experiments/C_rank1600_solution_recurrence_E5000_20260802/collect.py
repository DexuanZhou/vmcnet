#!/usr/bin/env python3
"""Collect the one-seed causal decision for WSSR solution recurrence."""

import json
import math
from pathlib import Path


ROOT = Path(
    "/scratch/dexuan1/runs/C_rank1600_solution_recurrence_E5000_20260802"
)
CONTROL_ROOT = Path(
    "/scratch/dexuan1/runs/C_rank1600_eta_memory_E5000_20260802/"
    "frozen/eta03_control"
)
SPRING_E5000_VARIANCE = 0.03784060855346502


def load(path):
    payload = json.loads(path.read_text())
    return {
        "energy": float(payload.get("energy", payload["average"])),
        "variance": float(payload["variance"]),
    }


results = {}
paired_ratios = []
for seed in (0, 1):
    control_path = CONTROL_ROOT / f"seed{seed}/eval/statistics.json"
    residual_path = ROOT / f"frozen/residual_mu099/seed{seed}/eval/statistics.json"
    if not residual_path.exists():
        continue
    control = load(control_path)
    residual = load(residual_path)
    ratio = residual["variance"] / control["variance"]
    residual["variance_ratio_over_control"] = ratio
    results[f"eta03_control_seed{seed}"] = control
    results[f"residual_mu099_seed{seed}"] = residual
    paired_ratios.append(ratio)

naive_path = ROOT / "frozen/naive_mu099/seed0/eval/statistics.json"
if naive_path.exists():
    naive = load(naive_path)
    naive["variance_ratio_over_control"] = (
        naive["variance"] / results["eta03_control_seed0"]["variance"]
    )
    results["naive_mu099_seed0"] = naive

geometric_mean_ratio = math.exp(
    sum(math.log(ratio) for ratio in paired_ratios) / len(paired_ratios)
)
if len(paired_ratios) == 2:
    if all(ratio < 1.0 for ratio in paired_ratios) and geometric_mean_ratio <= 0.9:
        decision = "residual_correction_positive_replicated"
    else:
        decision = "residual_correction_not_replicated"
    second_seed_required = False
else:
    decision = "residual_correction_positive_seed0"
    second_seed_required = True

summary = {
    "primary_endpoint": "epoch5000_light_frozen_variance",
    "decision": decision,
    "second_seed_required": second_seed_required,
    "paired_residual_over_control_ratios": paired_ratios,
    "geometric_mean_ratio": geometric_mean_ratio,
    "spring_e5000_context": {
        "variance": SPRING_E5000_VARIANCE,
        "residual_seed0_over_spring": (
            results["residual_mu099_seed0"]["variance"]
            / SPRING_E5000_VARIANCE
        ),
    },
    "rules": {
        "positive": "residual < control and residual < 0.9 * naive",
        "advance": "only a positive result receives a second seed",
    },
    "results": results,
}
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
