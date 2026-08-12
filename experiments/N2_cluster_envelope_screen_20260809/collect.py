from pathlib import Path
import json
import numpy as np

ROOT = Path("/scratch/dexuan1/runs/N2_cluster_envelope_screen_20260809_retry1")
OUT = ROOT / "screen_summary.json"


def load(run: Path, name: str):
    path = run / f"{name}.txt"
    if not path.exists():
        return None
    return np.atleast_1d(np.loadtxt(path, dtype=float))


def summary(values):
    if values is None or values.size == 0:
        return None
    tail = values[-min(100, values.size):]
    return {
        "n": int(values.size),
        "median_all": float(np.nanmedian(values)),
        "median_tail100": float(np.nanmedian(tail)),
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
    }


metrics = [
    "energy_noclip",
    "variance_noclip",
    "wssr_envelope_numerical_rank",
    "wssr_envelope_history_numerical_rank",
    "wssr_envelope_current_novelty_fraction",
    "wssr_envelope_gradient_capture_fraction",
    "wssr_envelope_cluster_update_norm",
    "wssr_envelope_complement_update_norm",
    "wssr_envelope_complement_cluster_ratio",
    "wssr_envelope_bar_lambda",
    "wssr_envelope_gamma",
    "wssr_diag_finite",
]
result = {}
for arm in ("A_hard_rank200", "B_cluster_envelope"):
    run = ROOT / arm
    result[arm] = {name: summary(load(run, name)) for name in metrics}

candidate = result["B_cluster_envelope"]
rank = candidate["wssr_envelope_numerical_rank"]
novelty = candidate["wssr_envelope_current_novelty_fraction"]
cluster = candidate["wssr_envelope_cluster_update_norm"]
complement = candidate["wssr_envelope_complement_update_norm"]
finite = candidate["wssr_diag_finite"]
gate = {
    "rank_tail100_at_least_48": bool(rank and rank["median_tail100"] >= 48),
    "novelty_tail100_positive": bool(
        novelty and novelty["median_tail100"] > 1e-3
    ),
    "cluster_component_active": bool(cluster and cluster["median_tail100"] > 0),
    "complement_component_active": bool(
        complement and complement["median_tail100"] > 0
    ),
    "all_finite": bool(finite and finite["min"] == 1.0),
}
result["gate"] = gate
result["advance_to_E5000"] = bool(all(gate.values()))
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
