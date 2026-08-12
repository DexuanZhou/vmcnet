#!/usr/bin/env python3
"""Collect the paired E5000 frozen-gradient-transport decision."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path("/scratch/dexuan1/runs/C_wssr_gradient_transport_etaG02_20260803")
ARMS = ("expG02_vanilla", "expG02_transported")
rows = []
ratios = []
for seed in (0, 1):
    values = {}
    for arm in ARMS:
        path = ROOT / "frozen" / arm / f"seed{seed}" / "eval" / "statistics.json"
        payload = json.loads(path.read_text())
        energy = float(payload.get("energy", payload["average"]))
        variance = float(payload["variance"])
        values[arm] = variance
        rows.append({"arm": arm, "seed": seed, "energy": energy, "variance": variance})
    ratios.append(values["expG02_transported"] / values["expG02_vanilla"])

geometric_mean_ratio = float(math.exp(np.mean(np.log(ratios))))
if all(ratio < 1.0 for ratio in ratios) and geometric_mean_ratio <= 0.9:
    decision = "positive"
elif geometric_mean_ratio < 1.0:
    decision = "promising_but_inconclusive"
else:
    decision = "negative"
summary = {
    "primary_endpoint": "epoch5000_light_frozen_variance",
    "transported_over_vanilla_ratios": ratios,
    "geometric_mean_ratio": geometric_mean_ratio,
    "relative_reduction": 1.0 - geometric_mean_ratio,
    "decision": decision,
    "positive_rule": "both paired ratios < 1 and geometric mean <= 0.9",
    "rows": rows,
}
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
