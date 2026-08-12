#!/usr/bin/env python3
"""Compare the one-seed middle-eta screen with its paired eta=.3 control."""

import json
from pathlib import Path


ROOT = Path("/scratch/dexuan1/runs/C_rank1600_eta_memory_E5000_20260802")


def load(arm):
    path = ROOT / "frozen" / arm / "seed0" / "eval" / "statistics.json"
    payload = json.loads(path.read_text())
    return {
        "energy": float(payload.get("energy", payload["average"])),
        "variance": float(payload["variance"]),
    }


control = load("eta03_control")
arms = {arm: load(arm) for arm in ("eta08_bias", "eta095_bias")}
for values in arms.values():
    values["variance_ratio_over_eta03"] = values["variance"] / control["variance"]
best_arm = min(arms, key=lambda arm: arms[arm]["variance_ratio_over_eta03"])
summary = {
    "control": control,
    "candidates": arms,
    "best_arm": best_arm,
    "best_ratio": arms[best_arm]["variance_ratio_over_eta03"],
    "second_seed_required": arms[best_arm]["variance_ratio_over_eta03"] < 1.0,
}
(ROOT / "mid_eta_screen.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
