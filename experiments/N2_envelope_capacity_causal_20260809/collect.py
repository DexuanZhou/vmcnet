#!/usr/bin/env python3
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path("/scratch/dexuan1/runs/N2_envelope_capacity_causal_20260809_retry1")
ARMS = ("H96_current_only", "E96_current32_history2")


def one(path: Path) -> dict:
    candidates = [path / "eval" / "statistics.json"]
    candidates.extend(sorted(path.parent.glob(path.name + "_*/eval/statistics.json")))
    stats_path = next((p for p in candidates if p.is_file()), None)
    if stats_path is None:
        raise FileNotFoundError(path)
    stats = json.loads(stats_path.read_text())
    values = np.loadtxt(stats_path.parent / "local_energies.txt")
    return {
        "statistics_path": str(stats_path),
        "energy": float(np.mean(values)),
        "standard_error": float(np.std(values) / np.sqrt(values.size)),
        "variance": float(np.var(values)),
        "samples": int(values.size),
    }


rows = []
for arm in ARMS:
    result = one(ROOT / "frozen" / f"{arm}_e1000")
    train = ROOT / arm
    result.update(
        arm=arm,
        training_tail100_energy=float(np.median(np.loadtxt(train / "energy_noclip.txt")[-100:])),
        training_tail100_variance=float(np.median(np.loadtxt(train / "variance_noclip.txt")[-100:])),
        envelope_rank=float(np.median(np.loadtxt(train / "wssr_envelope_numerical_rank.txt")[-100:])),
        projected_condition=float(np.median(np.loadtxt(train / "wssr_envelope_projected_curvature_condition.txt")[-100:])),
    )
    rows.append(result)

out = ROOT / "capacity_causal_summary.json"
out.write_text(json.dumps(rows, indent=2) + "\n")
with (ROOT / "capacity_causal_summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(rows, indent=2))
