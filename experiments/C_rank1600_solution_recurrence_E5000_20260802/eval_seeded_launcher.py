#!/usr/bin/env python3
"""Fold an independent but paired key into frozen evaluation."""

import os

import jax

from vmcnet.train import runners
from vmcnet.utils import io


SEED = int(os.environ["WSSR_PAIRED_SEED"])
FOLD_TAG = 0x4556414C + SEED
original_reload = io.reload_vmc_state


def paired_reload(*args, **kwargs):
    epoch, data, params, optimizer_state, key = original_reload(*args, **kwargs)
    if getattr(key, "ndim", 0) == 2:
        key = jax.vmap(lambda item: jax.random.fold_in(item, FOLD_TAG))(key)
    else:
        key = jax.random.fold_in(key, FOLD_TAG)
    return epoch, data, params, optimizer_state, key


io.reload_vmc_state = paired_reload
runners.run_molecule()
