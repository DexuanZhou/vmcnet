from pathlib import Path
import json
import numpy as np

ROOT = Path("/scratch/dexuan1/runs/N2_cluster_envelope_E5000_20260809")
ARMS = ("A_hard_rank200", "B_cluster_envelope")
STEPS = (1000, 2500, 5000)


def frozen_stats(path):
    # VMCNet appends a numeric suffix when the requested evaluation directory
    # already exists (the Slurm wrapper creates the unsuffixed metadata path).
    # Resolve that layout here so collection does not mistake a completed
    # frozen evaluation for a missing result.
    if not (path / "eval" / "statistics.json").exists():
        candidates = sorted(
            path.parent.glob(f"{path.name}_*/eval/statistics.json")
        )
        if candidates:
            path = candidates[-1].parent.parent
    json_path = path / "eval" / "statistics.json"
    if json_path.exists():
        raw = json.loads(json_path.read_text())
        return raw
    energy = np.loadtxt(path / "energy.txt", dtype=float)
    variance = np.loadtxt(path / "variance.txt", dtype=float)
    return {
        "energy_mean": float(np.mean(energy)),
        "energy_stderr": float(np.std(energy, ddof=1) / np.sqrt(energy.size)),
        "variance_mean": float(np.mean(variance)),
        "n": int(energy.size),
    }


result = {}
for arm in ARMS:
    result[arm] = {}
    for step in STEPS:
        result[arm][str(step)] = frozen_stats(
            ROOT / "frozen" / f"{arm}_e{step}"
        )
    train = ROOT / arm
    result[arm]["train_tail100"] = {
        name: float(np.median(np.loadtxt(train / f"{name}.txt")[-100:]))
        for name in ("energy_noclip", "variance_noclip")
    }
    slurm = ROOT / "metadata" / arm / "slurm_job.txt"
    result[arm]["slurm_metadata"] = str(slurm)

result["paired_comparison"] = {}
benchmark = -109.5423
for step in STEPS:
    hard = result["A_hard_rank200"][str(step)]
    envelope = result["B_cluster_envelope"][str(step)]
    result["paired_comparison"][str(step)] = {
        "envelope_minus_hard_mHa": 1000.0
        * (envelope["average"] - hard["average"]),
        "envelope_to_hard_variance_ratio": envelope["variance"]
        / hard["variance"],
        "hard_energy_error_mHa": 1000.0 * (hard["average"] - benchmark),
        "envelope_energy_error_mHa": 1000.0
        * (envelope["average"] - benchmark),
    }

path = ROOT / "summary.json"
path.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
