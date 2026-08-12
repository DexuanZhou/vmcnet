"""Timing-only launcher: wraps burn-in and metrics writes, not optimizer code."""
import csv
import os
import time

import jax
from vmcnet.mcmc import metropolis
from vmcnet.train import runners
from vmcnet.train import vmc as vmc_loop_module


OUT = os.environ["SMOKE_TIMING_DIR"]
os.makedirs(OUT, exist_ok=True)
PROCESS_START = time.time()
PROCESS_START_MONO = time.perf_counter()


def append_event(event, epoch="", wall_time="", monotonic=""):
    path = os.path.join(OUT, "phase_timing.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["event", "epoch", "wall_time_unix", "monotonic_seconds"])
        writer.writerow([event, epoch, wall_time or time.time(), monotonic or time.perf_counter()])
        f.flush()


append_event("process_start", wall_time=PROCESS_START, monotonic=PROCESS_START_MONO)

# The stock reload path restores checkpoint walkers after constructing new data.
# Capture the freshly initialized 4096-walker state and return it from reload while
# taking only model parameters from the source checkpoint. The reloaded optimizer
# state is returned for API compatibility but is ignored because the CLI sets
# reload.new_optimizer_state=True.
_fresh = {}
_setup_vmc = runners._setup_vmc


def setup_with_fresh_walkers(*args, **kwargs):
    result = _setup_vmc(*args, **kwargs)
    _fresh["data"] = result[6]
    _fresh["key"] = result[8]
    append_event("fresh_walkers_initialized")
    return result


runners._setup_vmc = setup_with_fresh_walkers

_reload_vmc_state = runners.utils.io.reload_vmc_state


def reload_params_only(*args, **kwargs):
    epoch, _old_data, params, optimizer_state, _old_key = _reload_vmc_state(
        *args, **kwargs
    )
    if "data" not in _fresh:
        raise RuntimeError("fresh walker state was not initialized before reload")
    append_event("checkpoint_params_loaded_fresh_walkers_retained")
    return epoch, _fresh["data"], params, optimizer_state, _fresh["key"]


runners.utils.io.reload_vmc_state = reload_params_only

_burn_data = metropolis.burn_data


def timed_burn_data(*args, **kwargs):
    append_event("burn_start")
    result = _burn_data(*args, **kwargs)
    jax.block_until_ready(result)
    append_event("burn_end")
    return result


metropolis.burn_data = timed_burn_data

_append_metrics = vmc_loop_module._append_training_metrics_csv_row


def timed_append_metrics(logdir, epoch, metrics):
    result = _append_metrics(logdir, epoch, metrics)
    jax.block_until_ready(metrics)
    append_event("epoch_end", epoch=int(epoch) + 1)
    return result


vmc_loop_module._append_training_metrics_csv_row = timed_append_metrics

try:
    runners.run_molecule()
finally:
    append_event("process_end")
