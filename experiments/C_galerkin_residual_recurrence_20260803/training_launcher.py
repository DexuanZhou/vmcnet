#!/usr/bin/env python3
"""Run one manifest-selected Galerkin recurrence training arm."""

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
SEED = int(os.environ["WSSR_SEED"])
RANK = int(os.environ["WSSR_RANK"])
UPDATES = int(os.environ["WSSR_UPDATES"])
ARM = os.environ["WSSR_ARM"]
MU = float(os.environ["WSSR_MU"])
RESIDUAL_EVALUATION = os.environ["WSSR_RESIDUAL_EVALUATION"]
DUAL_DIAGNOSTICS = os.environ["WSSR_DUAL_DIAGNOSTICS"] == "1"
ERROR_FEEDBACK = os.environ["WSSR_ERROR_FEEDBACK"] == "1"
SUBSPACE_ETA_S = float(os.environ["WSSR_SUBSPACE_ETA_S"])
SUBSPACE_REFRESH_PERIOD = int(
    os.environ.get("WSSR_SUBSPACE_REFRESH_PERIOD", "1")
)
SUBSPACE_REFRESH_MODE = os.environ.get(
    "WSSR_SUBSPACE_REFRESH_MODE", "ritz"
)
FOLD_TAG = 0x47414C30 + 100 * SEED + RANK


def _git_output(*args):
    return subprocess.check_output(args, text=True).strip()


protocol = {
    "scientific_question": "galerkin_consistent_sketched_residual_recurrence",
    "arm": ARM,
    "paired_seed": SEED,
    "rank": RANK,
    "updates": UPDATES,
    "solution_recurrence_mode": "residual",
    "solution_recurrence_mu": MU,
    "residual_evaluation": RESIDUAL_EVALUATION,
    "residual_dual_mode_diagnostics": DUAL_DIAGNOSTICS,
    "solution_error_feedback": ERROR_FEEDBACK,
    "error_feedback_norm_cap": 10.0,
    "subspace_eta_S": SUBSPACE_ETA_S,
    "subspace_refresh_period": SUBSPACE_REFRESH_PERIOD,
    "subspace_refresh_mode": SUBSPACE_REFRESH_MODE,
    "checkpoint_key_fold_in": FOLD_TAG,
    "hostname": socket.gethostname(),
    "git_commit": _git_output("git", "rev-parse", "HEAD"),
    "git_diff_sha256": subprocess.check_output(
        ["bash", "-c", "git diff | sha256sum | cut -d' ' -f1"], text=True
    ).strip(),
    "source_checkpoint": (
        "/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1/checkpoints/1000.npz"
    ),
    "source_epoch": 1000,
    "reburn": False,
    "new_optimizer_state": True,
    "walkers": 1000,
    "mcmc_steps_per_update": 10,
    "ssi_initial": 40,
    "ssi_warm": 2,
    "learning_rate": 0.04,
    "learning_decay_rate": 1e-4,
    "eta_S": 0.0,
    "eta_g": 0.0,
    "fixed_tikhonov_lambda": 1e-3,
    "relative_singular_value_cutoff": 3e-4,
    "norm_constraint": 1e-3,
    "mixed_precision_rank_arithmetic": True,
    "global_jax_x64": bool(jax.config.x64_enabled),
}
(OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")


MONITOR_FIELDS = (
    "epoch",
    "energy",
    "variance",
    "raw_direction_norm",
    "prior_norm",
    "correction_norm",
    "active_rank",
    "sample_residual_norm",
    "correctable_residual_norm",
    "correctable_residual_ratio",
    "captured_sample_residual_ratio",
    "residual_norm_reduction",
    "residual_reduction_fraction",
    "subspace_refreshed",
    "correction_relative_difference",
    "error_feedback_norm",
    "error_feedback_clip_count",
    "hypothetical_gradient_drift_beta",
    "time",
)


def _metric_float(metrics, name):
    return float(metrics.get(name, np.nan))


def _append_monitor(epoch, metrics):
    path = OUT / "monitor.csv"
    new = not path.exists()
    row = {
        "epoch": epoch,
        "energy": _metric_float(metrics, "energy"),
        "variance": _metric_float(metrics, "variance"),
        "raw_direction_norm": _metric_float(
            metrics, "wssr_diag_raw_direction_norm"
        ),
        "prior_norm": _metric_float(metrics, "wssr_solution_prior_norm"),
        "correction_norm": _metric_float(
            metrics, "wssr_solution_correction_norm"
        ),
        "active_rank": _metric_float(metrics, "wssr_active_rank"),
        "sample_residual_norm": _metric_float(
            metrics, "wssr_residual_sample_norm"
        ),
        "correctable_residual_norm": _metric_float(
            metrics, "wssr_residual_correctable_norm"
        ),
        "correctable_residual_ratio": _metric_float(
            metrics, "wssr_residual_correctable_ratio"
        ),
        "captured_sample_residual_ratio": _metric_float(
            metrics, "wssr_residual_captured_sample_ratio"
        ),
        "residual_norm_reduction": _metric_float(
            metrics, "wssr_residual_norm_reduction"
        ),
        "residual_reduction_fraction": _metric_float(
            metrics, "wssr_residual_reduction_fraction"
        ),
        "subspace_refreshed": _metric_float(
            metrics, "wssr_subspace_refreshed"
        ),
        "correction_relative_difference": _metric_float(
            metrics, "wssr_residual_correction_relative_difference"
        ),
        "error_feedback_norm": _metric_float(
            metrics, "wssr_error_feedback_norm"
        ),
        "error_feedback_clip_count": _metric_float(
            metrics, "wssr_error_feedback_clip_count"
        ),
        "hypothetical_gradient_drift_beta": _metric_float(
            metrics, "wssr_hypothetical_gradient_drift_beta"
        ),
        "time": time.time(),
    }
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MONITOR_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


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
    _append_monitor(int(epoch) + 1, metrics)
    variance = float(metrics["variance"])
    energy = float(metrics["energy"])
    direction = _metric_float(metrics, "wssr_diag_raw_direction_norm")
    finite_flag = bool(metrics.get("wssr_diag_finite", False))
    variances.append(variance)
    directions.append(direction)
    values = (energy, variance, direction)
    reason = None
    if not all(np.isfinite(item) for item in values) or not finite_flag:
        reason = "nonfinite_training_or_optimizer_value"
    elif len(variances) >= 20:
        variance_baseline = float(np.median(variances[:20]))
        direction_baseline = float(np.median(directions[:20]))
        if variance_baseline > 0 and variance > 100.0 * variance_baseline:
            reason = "variance_gt_100x_first20_median"
        elif direction_baseline > 0 and direction > 100.0 * direction_baseline:
            reason = "raw_direction_gt_100x_first20_median"
    if reason is not None:
        payload = {"triggered": True, "reason": reason, "epoch": epoch + 1}
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
