#!/usr/bin/env python3
"""Paired-key launcher and telemetry for the Grassmann WSSR screen."""

import csv
import json
import os
import time
from pathlib import Path

import jax
import numpy as np

from vmcnet.train import runners
from vmcnet.train import vmc as vmc_module
from vmcnet.utils import io


OUT = Path(os.environ["WSSR_MONITOR_DIR"])
OUT.mkdir(parents=True, exist_ok=False)
ALPHA = float(os.environ["WSSR_GRASSMANN_ALPHA"])
FOLD_TAG = 0x47524153

(OUT / "protocol.json").write_text(
    json.dumps(
        {
            "system": "C",
            "source_checkpoint": os.environ["WSSR_SOURCE_CHECKPOINT"],
            "paired_seed": 0,
            "updates": 1000,
            "walkers": 1000,
            "rank": 200,
            "grassmann_alpha": ALPHA,
            "eta_S": 0.0,
            "eta_g": 0.0,
            "learning_rate": 0.04,
            "tikhonov_lambda": 1e-3,
            "norm_constraint": 1e-3,
        },
        indent=2,
    )
    + "\n"
)

original_reload = io.reload_vmc_state


def paired_reload(*args, **kwargs):
    epoch, data, params, optimizer_state, key = original_reload(*args, **kwargs)
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
    "overlap_mean",
    "overlap_min",
    "history_active",
    "gradient_capture",
    "raw_direction_norm",
    "constraint_scale",
    "wall_clock",
)


def metric(metrics, name):
    return float(metrics.get(name, np.nan))


def monitored_append(logdir, epoch, metrics):
    result = original_append(logdir, epoch, metrics)
    jax.block_until_ready(metrics)
    row = {
        "epoch": int(epoch) + 1,
        "energy": metric(metrics, "energy"),
        "variance": metric(metrics, "variance"),
        "active_rank": metric(metrics, "wssr_active_rank"),
        "overlap_mean": metric(metrics, "wssr_grassmann_overlap_mean"),
        "overlap_min": metric(metrics, "wssr_grassmann_overlap_min"),
        "history_active": metric(metrics, "wssr_grassmann_history_active"),
        "gradient_capture": metric(
            metrics, "wssr_envelope_gradient_capture_fraction"
        ),
        "raw_direction_norm": metric(metrics, "wssr_diag_raw_direction_norm"),
        "constraint_scale": metric(
            metrics, "wssr_diag_norm_constraint_scale"
        ),
        "wall_clock": time.time(),
    }
    path = OUT / "monitor.csv"
    new_file = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    if not np.isfinite(row["energy"]) or not np.isfinite(row["variance"]):
        raise RuntimeError("nonfinite energy or variance")
    return result


vmc_module._append_training_metrics_csv_row = monitored_append
runners.run_molecule()
