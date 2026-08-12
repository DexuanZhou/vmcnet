#!/usr/bin/env python3
"""Run one paired C arm after folding the reloaded checkpoint PRNG key."""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import jax

from vmcnet.train import runners
from vmcnet.utils import io


metadata = Path(os.environ["FUNCTION_CAP_METADATA"])
metadata.mkdir(parents=True, exist_ok=False)
seed = int(os.environ["FUNCTION_CAP_SEED"])
method = os.environ["FUNCTION_CAP_METHOD"]
fold_tag = 0x46534E00 + seed

scoped_files = [
    "vmcnet/updates/update_param_fns.py",
    "vmcnet/updates/spring.py",
    "vmcnet/updates/wssr.py",
    "vmcnet/train/default_config.py",
]
diff = subprocess.check_output(
    ["git", "diff", "--", *scoped_files], text=True
)
protocol = {
    "question": "SPRING versus rank-1600 WSSR at matched function-space radius",
    "method": method,
    "paired_seed": seed,
    "checkpoint_key_fold_in": fold_tag,
    "source_checkpoint": (
        "/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1/checkpoints/1000.npz"
    ),
    "function_norm": "centered score action divided by sqrt(num_walkers)",
    "function_norm_constraint": float(os.environ["FUNCTION_CAP_RADIUS"]),
    "target_epoch": int(os.environ["FUNCTION_CAP_TARGET_EPOCH"]),
    "eta_S": os.environ.get("FUNCTION_CAP_ETA_S"),
    "eta_g": os.environ.get("FUNCTION_CAP_ETA_G"),
    "complement_weight": os.environ.get("FUNCTION_CAP_COMPLEMENT_WEIGHT"),
    "ssi_warm_iterations": os.environ.get("FUNCTION_CAP_SSI_WARM"),
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "scoped_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
}
(metadata / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
(metadata / "scoped.patch").write_text(diff)

original_reload = io.reload_vmc_state


def paired_reload(*args, **kwargs):
    epoch, data, params, optimizer_state, key = original_reload(*args, **kwargs)
    jax.block_until_ready((data, params, optimizer_state, key))
    if getattr(key, "ndim", 0) == 2:
        key = jax.vmap(lambda item: jax.random.fold_in(item, fold_tag))(key)
    else:
        key = jax.random.fold_in(key, fold_tag)
    return epoch, data, params, optimizer_state, key


io.reload_vmc_state = paired_reload
start = time.time()
try:
    runners.run_molecule()
finally:
    elapsed = time.time() - start
    (metadata / "wall_time_seconds.txt").write_text(f"{elapsed:.9f}\n")
