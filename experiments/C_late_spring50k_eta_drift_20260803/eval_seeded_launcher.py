#!/usr/bin/env python3
"""Use one independent, identical PRNG key for every frozen endpoint."""

import os

import jax

from vmcnet.train import runners
from vmcnet.utils import io


SEED = int(os.environ.get("FROZEN_SEED", "0"))
original_reload = io.reload_vmc_state


def paired_reload(*args, **kwargs):
    epoch, data, params, optimizer_state, _ = original_reload(*args, **kwargs)
    key = jax.random.PRNGKey(0x4C415445 + SEED)
    return epoch, data, params, optimizer_state, key


io.reload_vmc_state = paired_reload
runners.run_molecule()
