#!/usr/bin/env python3
"""Read-only validation of the common N2 KFAC-pre5000 checkpoint."""
import json
from pathlib import Path
import numpy as np

ROOT = Path("/scratch/dexuan1/runs/fir_n4096_corrected_retry1/N2/kfac_preliminary")
CKPT = ROOT / "checkpoints/5000.npz"
OUT = Path(__file__).resolve().parent / "results/checkpoint_validation.json"

def finite_tree(x):
    if isinstance(x, dict): return all(finite_tree(v) for v in x.values())
    if isinstance(x, (tuple, list)): return all(finite_tree(v) for v in x)
    a = np.asarray(x)
    return a.dtype.kind not in "fc" or bool(np.isfinite(a).all())

with np.load(CKPT, allow_pickle=True) as z:
    epoch = int(np.asarray(z["e"]).item())
    data = z["d"].tolist()
    params = z["p"].tolist()
    optimizer = z["o"].tolist()
    key = np.asarray(z["k"])
position = np.asarray(data["walker_data"]["position"])
amplitude = np.asarray(data["walker_data"]["amplitude"])
unique = np.unique(np.ascontiguousarray(position.reshape(4096, -1)), axis=0).shape[0]
cfg = json.loads((ROOT / "config.json").read_text())
ions = np.asarray(cfg["problem"]["ion_pos"], dtype=float)
result = {
    "checkpoint": str(CKPT), "internal_epoch": epoch,
    "nchains": int(position.shape[0]), "unique_walkers": int(unique),
    "position_shape": list(position.shape), "amplitude_shape": list(amplitude.shape),
    "prng_shape": list(key.shape), "parameters_finite": finite_tree(params),
    "optimizer_state_finite": finite_tree(optimizer), "walker_state_finite": finite_tree(data),
    "bond_length_bohr": float(np.linalg.norm(ions[1] - ions[0])),
    "ion_pos": cfg["problem"]["ion_pos"], "ion_charges": cfg["problem"]["ion_charges"],
    "nelec": cfg["problem"]["nelec"], "dtype": cfg["dtype"],
    "complete_state": all(k in data for k in ("walker_data",)),
}
passed = (epoch in (4999, 5000) and position.shape == (4096, 14, 3)
          and amplitude.shape == (4096,) and unique == 4096
          and abs(result["bond_length_bohr"] - 2.068) < 1e-12
          and result["ion_charges"] == [7.0, 7.0] and result["nelec"] == [7, 7]
          and result["parameters_finite"] and result["optimizer_state_finite"]
          and result["walker_state_finite"] and key.size > 0)
result["status"] = "PASS" if passed else "FAIL"
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
if not passed: raise SystemExit(5)
