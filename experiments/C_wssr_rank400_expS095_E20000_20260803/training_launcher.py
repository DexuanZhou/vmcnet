#!/usr/bin/env python3
"""Run the matched C rank-400 WSSR trajectory with divergence monitoring."""

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
SEED = int(os.environ.get("WSSR_PAIRED_SEED", "0"))
FOLD_TAG = 0x4D454D30 + SEED
PROTOCOL_LEARNING_RATE = float(os.environ.get("WSSR_PROTOCOL_LEARNING_RATE", "0.04"))
PROTOCOL_EXACT_FIRST = os.environ.get("WSSR_PROTOCOL_EXACT_FIRST", "False") == "True"
PROTOCOL_LEGACY_REGULARIZATION = (
    os.environ.get("WSSR_PROTOCOL_LEGACY_REGULARIZATION", "False") == "True"
)
PROTOCOL_QUESTION = os.environ.get(
    "WSSR_PROTOCOL_QUESTION",
    "rank400_exponential_S_history_with_current_gradient_C_E20000",
)


def _git_output(*args):
    return subprocess.check_output(args, text=True).strip()


protocol = {
    "scientific_question": PROTOCOL_QUESTION,
    "paired_seed": SEED,
    "checkpoint_key_fold_in": FOLD_TAG,
    "hostname": socket.gethostname(),
    "git_commit": _git_output("git", "rev-parse", "HEAD"),
    "source_checkpoint": (
        "/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1/checkpoints/1000.npz"
    ),
    "source_epoch": 1000,
    "wssr_updates": 20000,
    "reburn": False,
    "new_optimizer_state": True,
    "walkers": 1000,
    "mcmc_steps_per_update": 10,
    "rank": 400,
    "ssi_initial": 40,
    "ssi_warm": 2,
    "learning_rate": PROTOCOL_LEARNING_RATE,
    "learning_decay_rate": 1e-4,
    "eta_g": 0.0,
    "eta_S_schedule": "exponential_growth",
    "eta_S_max": 0.95,
    "eta_S_tau": 1000.0,
    "exact_first": PROTOCOL_EXACT_FIRST,
    "legacy_regularization": PROTOCOL_LEGACY_REGULARIZATION,
    "fixed_tikhonov_lambda": None if PROTOCOL_LEGACY_REGULARIZATION else 1e-3,
    "relative_singular_value_cutoff": (
        None if PROTOCOL_LEGACY_REGULARIZATION else 3e-4
    ),
    "norm_constraint": 1e-3,
    "complement_weight": 0.0,
    "mixed_precision_solve": True,
    "global_jax_x64": bool(jax.config.x64_enabled),
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
