#!/usr/bin/env python3
"""Run 100 timed C updates plus a five-update profiler tail."""

import csv
import json
import os
import time
from pathlib import Path

import jax

from vmcnet.train import runners
from vmcnet.train import vmc as vmc_module


OUT = Path(os.environ["TIMING_META_DIR"])
OUT.mkdir(parents=True, exist_ok=True)
TRACE_DIR = OUT / "jax_trace"
PROFILE_FIRST_EPOCH = int(os.environ.get("TIMING_PROFILE_START", "101"))
PROFILE_LAST_EPOCH = PROFILE_FIRST_EPOCH + 4


def event(name, epoch=""):
    path = OUT / "phase_timing.csv"
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(["event", "epoch", "wall_time_unix", "monotonic_seconds"])
        writer.writerow([name, epoch, time.time(), time.perf_counter()])
        handle.flush()


event("process_start")
fresh = {}
original_setup = runners._setup_vmc


def setup_with_fresh_walkers(*args, **kwargs):
    """Retain walkers initialized at the requested timing-run chain count."""
    result = original_setup(*args, **kwargs)
    fresh["data"] = result[6]
    fresh["key"] = result[8]
    shapes = {
        str(index): list(getattr(leaf, "shape", ()))
        for index, leaf in enumerate(jax.tree_util.tree_leaves(result[6]))
    }
    (OUT / "fresh_data_leaf_shapes.json").write_text(
        json.dumps(shapes, indent=2) + "\n"
    )
    event("fresh_walkers_initialized")
    return result


runners._setup_vmc = setup_with_fresh_walkers
original_reload = runners.utils.io.reload_vmc_state


def timed_reload(*args, **kwargs):
    event("checkpoint_reload_start")
    epoch, _checkpoint_data, params, optimizer_state, _checkpoint_key = (
        original_reload(*args, **kwargs)
    )
    if "data" not in fresh:
        raise RuntimeError("fresh walker state was not initialized before reload")
    # This is a timing run: take trained model parameters from the checkpoint,
    # but do not silently restore its N=1000 sampler state for N=4096/8192 arms.
    value = epoch, fresh["data"], params, optimizer_state, fresh["key"]
    jax.block_until_ready(value)
    event("checkpoint_reload_end")
    return value


runners.utils.io.reload_vmc_state = timed_reload
original_append = vmc_module._append_training_metrics_csv_row
trace_started = False


def timed_append(logdir, epoch, metrics):
    global trace_started
    value = original_append(logdir, epoch, metrics)
    jax.block_until_ready(metrics)
    completed_epoch = int(epoch) + 1
    event("epoch_end", completed_epoch)
    # The requested steady timing is epochs 21--100.  A disjoint five-update
    # tail captures the compiled kernel structure without perturbing that window.
    if completed_epoch == PROFILE_FIRST_EPOCH - 1:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        jax.profiler.start_trace(str(TRACE_DIR))
        trace_started = True
        event("profile_start", PROFILE_FIRST_EPOCH)
    elif completed_epoch == PROFILE_LAST_EPOCH and trace_started:
        jax.profiler.stop_trace()
        trace_started = False
        event("profile_end", PROFILE_LAST_EPOCH)
    return value


vmc_module._append_training_metrics_csv_row = timed_append
try:
    runners.run_molecule()
finally:
    if trace_started:
        jax.profiler.stop_trace()
    event("process_end")
