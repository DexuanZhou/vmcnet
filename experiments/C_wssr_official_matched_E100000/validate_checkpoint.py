#!/usr/bin/env python3
"""Read-only validation of the common C KFAC-pre1000 checkpoint."""

import json
from pathlib import Path

import numpy as np

SOURCE = Path("/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1")
CHECKPOINT = SOURCE / "checkpoints/1000.npz"
SPRING_CONFIG = Path(
    "/scratch/dexuan1/runs/C_spring_official_kfac1000_E100000/config.json"
)
OUTPUT = Path(__file__).resolve().parent / "results/checkpoint_validation.json"


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(v) for v in value)
    array = np.asarray(value)
    return array.dtype.kind not in "fc" or bool(np.isfinite(array).all())


with np.load(CHECKPOINT, allow_pickle=True) as checkpoint:
    epoch = int(np.asarray(checkpoint["e"]).item())
    data = checkpoint["d"].tolist()
    parameters = checkpoint["p"].tolist()
    optimizer = checkpoint["o"].tolist()
    key = np.asarray(checkpoint["k"])

position = np.asarray(data["walker_data"]["position"])
amplitude = np.asarray(data["walker_data"]["amplitude"])
unique = np.unique(
    np.ascontiguousarray(position.reshape(position.shape[0], -1)), axis=0
).shape[0]
source_config = json.loads((SOURCE / "config.json").read_text())
spring_config = json.loads(SPRING_CONFIG.read_text())

result = {
    "checkpoint": str(CHECKPOINT),
    "internal_epoch": epoch,
    "expected_internal_epoch": 999,
    "nchains": int(position.shape[0]),
    "unique_walkers": int(unique),
    "position_shape": list(position.shape),
    "amplitude_shape": list(amplitude.shape),
    "prng_shape": list(key.shape),
    "parameters_finite": finite_tree(parameters),
    "optimizer_state_finite": finite_tree(optimizer),
    "walker_state_finite": finite_tree(data),
    "ion_pos": source_config["problem"]["ion_pos"],
    "ion_charges": source_config["problem"]["ion_charges"],
    "nelec": source_config["problem"]["nelec"],
    "dtype": source_config["dtype"],
    "architecture_matches_completed_spring": (
        source_config["model"] == spring_config["model"]
    ),
    "complete_walker_state": (
        "walker_data" in data
        and position.shape == (1000, 6, 3)
        and amplitude.shape == (1000,)
    ),
}
passed = (
    epoch == 999
    and position.shape == (1000, 6, 3)
    and amplitude.shape == (1000,)
    and unique == 1000
    and result["ion_pos"] == [[0.0, 0.0, 0.0]]
    and result["ion_charges"] == [6.0]
    and result["nelec"] == [4, 2]
    and result["parameters_finite"]
    and result["optimizer_state_finite"]
    and result["walker_state_finite"]
    and key.size > 0
    and result["architecture_matches_completed_spring"]
)
result["status"] = "PASS" if passed else "FAIL"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
if not passed:
    raise SystemExit(5)
