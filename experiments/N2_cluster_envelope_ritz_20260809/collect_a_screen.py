from pathlib import Path
import json

import numpy as np


NEW = Path(
    "/scratch/dexuan1/runs/N2_cluster_envelope_ritz_A200_20260809_retry1"
)
OLD = Path("/scratch/dexuan1/runs/N2_cluster_envelope_screen_20260809_retry1")


def tail(path: Path, window: int = 100) -> dict[str, float]:
    result = {}
    for name in ("energy", "variance", "energy_noclip", "variance_noclip"):
        values = np.loadtxt(path / f"{name}.txt", dtype=float)
        result[f"{name}_tail{window}_median"] = float(
            np.median(values[-window:])
        )
        result[f"{name}_tail{window}_mean"] = float(
            np.mean(values[-window:])
        )
    return result


result = {
    "hard_rank200_previous_paired_protocol": tail(OLD / "A_hard_rank200"),
    "scalar_envelope_previous": tail(OLD / "B_cluster_envelope"),
    "rayleigh_ritz_envelope": tail(NEW / "A_ritz96_alpha0"),
}

telemetry = {}
run = NEW / "A_ritz96_alpha0"
for file in sorted(run.glob("wssr_envelope_*.txt")):
    values = np.loadtxt(file, dtype=float)
    telemetry[file.stem] = {
        "tail100_median": float(np.median(values[-100:])),
        "all_median": float(np.median(values)),
    }
result["rayleigh_ritz_telemetry"] = telemetry

hard = result["hard_rank200_previous_paired_protocol"]
ritz = result["rayleigh_ritz_envelope"]
result["comparison"] = {
    "ritz_minus_hard_energy_noclip_mHa": 1000.0
    * (
        ritz["energy_noclip_tail100_median"]
        - hard["energy_noclip_tail100_median"]
    ),
    "ritz_to_hard_variance_noclip_ratio": (
        ritz["variance_noclip_tail100_median"]
        / hard["variance_noclip_tail100_median"]
    ),
}

output = NEW / "screen_summary.json"
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
