#!/usr/bin/env python3
"""Paired-seed launcher with long-history divergence monitoring."""

import csv
import json
import os
import socket
import subprocess
import time
from pathlib import Path

import jax
import numpy as np

from vmcnet.train import runners
from vmcnet.train import vmc as vmc_module
from vmcnet.utils import io


OUT = Path(os.environ["WSSR_MONITOR_DIR"])
OUT.mkdir(parents=True, exist_ok=False)
SEED = int(os.environ["WSSR_PAIRED_SEED"])
ARM = os.environ["WSSR_ARM"]
FOLD_TAG = 0x4D454D30 + SEED


def _git_output(*args):
    return subprocess.check_output(args, text=True).strip()


protocol = {
    "scientific_question": "does_fixed_lambda_long_memory_improve_early_C_WSSR",
    "arm": ARM,
    "paired_seed": SEED,
    "checkpoint_key_fold_in": FOLD_TAG,
    "hostname": socket.gethostname(),
    "git_commit": _git_output("git", "rev-parse", "HEAD"),
    "source_checkpoint": (
        "/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1/checkpoints/1000.npz"
    ),
    "source_epoch": 1000,
    "target_optimizer_epoch": 5000,
    "wssr_updates": 5000,
    "reburn": False,
    "new_optimizer_state": True,
    "walkers": 1000,
    "mcmc_steps_per_update": 10,
    "rank": 1600,
    "ssi_initial": 40,
    "ssi_warm": 2,
    "learning_rate": 0.04,
    "learning_decay_rate": 1e-4,
    "fixed_tikhonov_lambda": 1e-3,
    "relative_singular_value_cutoff": 3e-4,
    "norm_constraint": 1e-3,
    "complement_weight": 0.0,
    "mixed_precision_solve": True,
    "global_jax_x64": bool(jax.config.x64_enabled),
    "monitor": {
        "warmup_updates": 20,
        "variance_multiple": 100.0,
        "raw_direction_multiple": 100.0,
    },
}
(OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")


def _append_monitor(epoch, variance, direction, finite_flag):
    path = OUT / "monitor.csv"
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(
                ["epoch", "variance", "raw_direction_norm", "finite_flag", "time"]
            )
        writer.writerow([epoch, variance, direction, finite_flag, time.time()])


original_reload = io.reload_vmc_state


def paired_reload(*args, **kwargs):
    epoch, data, params, optimizer_state, key = original_reload(*args, **kwargs)
    jax.block_until_ready((data, params, optimizer_state, key))
    if getattr(key, "ndim", 0) == 2:
        key = jax.vmap(lambda item: jax.random.fold_in(item, FOLD_TAG))(key)
    else:
        key = jax.random.fold_in(key, FOLD_TAG)
    return epoch, data, params, optimizer_state, key


io.reload_vmc_state = paired_reload
original_append = vmc_module._append_training_metrics_csv_row
variances = []
directions = []


def monitored_append(logdir, epoch, metrics):
    value = original_append(logdir, epoch, metrics)
    jax.block_until_ready(metrics)
    variance = float(metrics["variance"])
    energy = float(metrics["energy"])
    direction = float(metrics.get("wssr_diag_raw_direction_norm", np.nan))
    finite_flag = bool(metrics.get("wssr_diag_finite", False))
    epoch_out = int(epoch) + 1
    variances.append(variance)
    directions.append(direction)
    _append_monitor(epoch_out, variance, direction, finite_flag)

    reason = None
    if not np.isfinite(energy) or not np.isfinite(variance):
        reason = "nonfinite_energy_or_variance"
    elif not np.isfinite(direction) or not finite_flag:
        reason = "nonfinite_optimizer_direction"
    elif len(variances) >= 20:
        variance_baseline = float(np.median(variances[:20]))
        direction_baseline = float(np.median(directions[:20]))
        if variance_baseline > 0 and variance > 100.0 * variance_baseline:
            reason = "variance_gt_100x_first20_median"
        elif direction_baseline > 0 and direction > 100.0 * direction_baseline:
            reason = "raw_direction_gt_100x_first20_median"
    if reason is not None:
        payload = {"triggered": True, "reason": reason, "epoch": epoch_out}
        (OUT / "early_stop.json").write_text(json.dumps(payload, indent=2) + "\n")
        raise RuntimeError(reason)
    return value


vmc_module._append_training_metrics_csv_row = monitored_append
try:
    runners.run_molecule()
finally:
    early_stop = OUT / "early_stop.json"
    if not early_stop.exists():
        payload = {"triggered": False, "reason": "none", "epoch": None}
        early_stop.write_text(json.dumps(payload, indent=2) + "\n")
