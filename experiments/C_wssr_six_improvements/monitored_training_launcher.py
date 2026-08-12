#!/usr/bin/env python3
"""Timing and in-process scientific-stop wrapper for the C WSSR ablation."""
import csv
import json
import os
import time

import jax
import numpy as np

from vmcnet.train import runners
from vmcnet.train import vmc as vmc_module


OUT = os.environ["SMOKE_TIMING_DIR"]
os.makedirs(OUT, exist_ok=True)


def event(name, epoch=""):
    path = os.path.join(OUT, "phase_timing.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["event", "epoch", "wall_time_unix", "monotonic_seconds"])
        writer.writerow([name, epoch, time.time(), time.perf_counter()])
        f.flush()


event("process_start")
original_reload = runners.utils.io.reload_vmc_state


def timed_reload(*args, **kwargs):
    event("checkpoint_reload_start")
    value = original_reload(*args, **kwargs)
    jax.block_until_ready(value)
    event("checkpoint_reload_end")
    return value


runners.utils.io.reload_vmc_state = timed_reload
original_append = vmc_module._append_training_metrics_csv_row
variances = []
directions = []


def monitored_append(logdir, epoch, metrics):
    value = original_append(logdir, epoch, metrics)
    jax.block_until_ready(metrics)
    event("epoch_end", int(epoch) + 1)
    variance = float(metrics["variance"])
    direction = float(metrics.get("wssr_diag_raw_direction_norm", np.nan))
    finite_flag = int(metrics.get("wssr_diag_finite", 1))
    variances.append(variance)
    directions.append(direction)
    reason = None
    if not np.isfinite(variance) or not np.isfinite(direction) or not finite_flag:
        reason = "nonfinite_metric_or_optimizer_direction"
    elif len(variances) >= 20:
        variance_baseline = float(np.median(variances[:20]))
        direction_baseline = float(np.median(directions[:20]))
        if variance_baseline > 0 and variance > 100 * variance_baseline:
            reason = "variance_gt_100x_first20_median"
        elif direction_baseline > 0 and direction > 100 * direction_baseline:
            reason = "raw_direction_gt_100x_first20_median"
    if reason:
        with open(os.path.join(OUT, "early_stop.json"), "w") as f:
            json.dump({"triggered": True, "reason": reason, "epoch": int(epoch) + 1}, f, indent=2)
        raise RuntimeError(reason)
    return value


vmc_module._append_training_metrics_csv_row = monitored_append
try:
    runners.run_molecule()
finally:
    if not os.path.exists(os.path.join(OUT, "early_stop.json")):
        with open(os.path.join(OUT, "early_stop.json"), "w") as f:
            json.dump({"triggered": False, "reason": "none", "epoch": None}, f, indent=2)
    event("process_end")
