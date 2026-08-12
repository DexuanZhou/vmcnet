#!/usr/bin/env python3
"""Paired-key launcher and lightweight telemetry for the spectral-history screen."""

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
MODE = os.environ["WSSR_REDUCED_METRIC_HISTORY_MODE"]
ETA_S = float(os.environ["WSSR_ETA_S"])
ANISOTROPIC_MATRIX_HISTORY = (
    os.environ.get("WSSR_ANISOTROPIC_MATRIX_HISTORY", "0") == "1"
)
FOLD_TAG = 0x53504543 + SEED


def _git_output(*args):
    return subprocess.check_output(args, text=True).strip()


protocol = {
    "scientific_question": "current_subspace_with_spectrum_only_history",
    "arm": ARM,
    "paired_seed": SEED,
    "checkpoint_key_fold_in": FOLD_TAG,
    "hostname": socket.gethostname(),
    "git_commit": _git_output("git", "rev-parse", "HEAD"),
    "git_diff_sha256": subprocess.check_output(
        ["bash", "-c", "git diff | sha256sum | cut -d' ' -f1"], text=True
    ).strip(),
    "source_checkpoint": (
        "/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1/checkpoints/1000.npz"
    ),
    "updates": 5000,
    "walkers": 1000,
    "mcmc_steps_per_update": 10,
    "rank": 800,
    "ssi_initial": 40,
    "ssi_warm": 2,
    "eta_S": ETA_S,
    "eta_g": 0.0,
    "reduced_metric_history_mode": MODE,
    "anisotropic_matrix_history": ANISOTROPIC_MATRIX_HISTORY,
    "spectral_history_cluster_gap": 0.01,
    "spectral_history_noise_scale": 1.0,
    "spectral_history_drift_scale": 1.0,
    "learning_rate": 0.04,
    "learning_decay_rate": 1e-4,
    "tikhonov_lambda": 1e-3,
    "relative_singular_value_cutoff": 3e-4,
    "norm_constraint": 1e-3,
    "reburn": False,
    "new_optimizer_state": True,
    "global_jax_x64": bool(jax.config.x64_enabled),
}
(OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")


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
FIELDS = (
    "epoch",
    "energy",
    "variance",
    "active_rank",
    "eta_mean",
    "eta_head",
    "eta_tail",
    "noise_floor",
    "history_overlap",
    "cluster_count",
    "drift_ratio_mean",
    "matrix_eta_mean",
    "matrix_eta_head",
    "matrix_eta_tail",
    "matrix_eta_spectral_mean",
    "matrix_current_weight",
    "wall_clock",
)


def _metric(metrics, name):
    return float(metrics.get(name, np.nan))


def monitored_append(logdir, epoch, metrics):
    value = original_append(logdir, epoch, metrics)
    jax.block_until_ready(metrics)
    row = {
        "epoch": int(epoch) + 1,
        "energy": _metric(metrics, "energy"),
        "variance": _metric(metrics, "variance"),
        "active_rank": _metric(metrics, "wssr_active_rank"),
        "eta_mean": _metric(metrics, "wssr_spectral_history_eta_mean"),
        "eta_head": _metric(metrics, "wssr_spectral_history_eta_head"),
        "eta_tail": _metric(metrics, "wssr_spectral_history_eta_tail"),
        "noise_floor": _metric(metrics, "wssr_spectral_history_noise_floor"),
        "history_overlap": _metric(metrics, "wssr_spectral_history_overlap"),
        "cluster_count": _metric(metrics, "wssr_spectral_history_cluster_count"),
        "drift_ratio_mean": _metric(
            metrics, "wssr_spectral_history_drift_ratio_mean"
        ),
        "matrix_eta_mean": _metric(
            metrics, "wssr_anisotropic_matrix_eta_mean"
        ),
        "matrix_eta_head": _metric(
            metrics, "wssr_anisotropic_matrix_eta_head"
        ),
        "matrix_eta_tail": _metric(
            metrics, "wssr_anisotropic_matrix_eta_tail"
        ),
        "matrix_eta_spectral_mean": _metric(
            metrics, "wssr_anisotropic_matrix_eta_spectral_mean"
        ),
        "matrix_current_weight": _metric(
            metrics, "wssr_anisotropic_matrix_current_weight"
        ),
        "wall_clock": time.time(),
    }
    path = OUT / "monitor.csv"
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)
    if not np.isfinite(row["energy"]) or not np.isfinite(row["variance"]):
        raise RuntimeError("nonfinite energy or variance")
    return value


vmc_module._append_training_metrics_csv_row = monitored_append
runners.run_molecule()
