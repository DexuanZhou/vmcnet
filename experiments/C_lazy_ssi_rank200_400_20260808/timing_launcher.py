#!/usr/bin/env python3
"""Minimal paired timing monitor shared by SPRING and lazy WSSR arms."""

import csv
import json
import os
import socket
import time
from pathlib import Path

import jax
import numpy as np

from vmcnet.train import runners
from vmcnet.train import vmc as vmc_module
from vmcnet.utils import io


OUT = Path(os.environ["TIMING_MONITOR_DIR"])
OUT.mkdir(parents=True, exist_ok=False)
ARM = os.environ["TIMING_ARM"]
SEED = int(os.environ.get("TIMING_SEED", "0"))
FOLD_TAG = int(os.environ.get("TIMING_FOLD_TAG", "127934600")) + SEED

(OUT / "protocol.json").write_text(
    json.dumps(
        {
            "arm": ARM,
            "seed": SEED,
            "fold_tag": FOLD_TAG,
            "hostname": socket.gethostname(),
            "source_checkpoint": (
                "/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1/"
                "checkpoints/1000.npz"
            ),
        },
        indent=2,
    )
    + "\n"
)

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


def monitored_append(logdir, epoch, metrics):
    value = original_append(logdir, epoch, metrics)
    jax.block_until_ready(metrics)
    path = OUT / "monitor.csv"
    new_file = not path.exists()
    row = {
        "epoch": int(epoch) + 1,
        "energy": float(metrics["energy"]),
        "variance": float(metrics["variance"]),
        "active_rank": float(metrics.get("wssr_active_rank", np.nan)),
        "subspace_refreshed": float(
            metrics.get("wssr_subspace_refreshed", np.nan)
        ),
        "residual_reduction_fraction": float(
            metrics.get("wssr_residual_reduction_fraction", np.nan)
        ),
        "time": time.time(),
    }
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    if not np.isfinite(row["energy"]) or not np.isfinite(row["variance"]):
        raise RuntimeError("non-finite short timing trajectory")
    return value


vmc_module._append_training_metrics_csv_row = monitored_append
runners.run_molecule()
