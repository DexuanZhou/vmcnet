#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np

root = Path("/scratch/dexuan1/runs/N2_envelope_selected_history_20260809")
base = root / "frozen" / "S96_current32_pool160_select64_e1000"
candidates = [base / "eval" / "statistics.json"]
candidates.extend(sorted(base.parent.glob(base.name + "_*/eval/statistics.json")))
stats = next(p for p in candidates if p.is_file())
x = np.loadtxt(stats.parent / "local_energies.txt")
train = root / "S96_current32_pool160_select64"
result = {
    "energy": float(np.mean(x)),
    "standard_error": float(np.std(x) / np.sqrt(x.size)),
    "variance": float(np.var(x)),
    "samples": int(x.size),
    "training_tail100_energy": float(np.median(np.loadtxt(train / "energy_noclip.txt")[-100:])),
    "training_tail100_variance": float(np.median(np.loadtxt(train / "variance_noclip.txt")[-100:])),
    "envelope_rank": float(np.median(np.loadtxt(train / "wssr_envelope_numerical_rank.txt")[-100:])),
    "history_pool_rank": float(np.median(np.loadtxt(train / "wssr_envelope_history_pool_numerical_rank.txt")[-100:])),
    "selected_history_count": float(np.median(np.loadtxt(train / "wssr_envelope_selected_history_count.txt")[-100:])),
    "projected_condition": float(np.median(np.loadtxt(train / "wssr_envelope_projected_curvature_condition.txt")[-100:])),
}
(root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
