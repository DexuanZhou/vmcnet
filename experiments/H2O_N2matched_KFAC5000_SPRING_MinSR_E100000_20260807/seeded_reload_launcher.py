#!/usr/bin/env python3
"""Replace only the checkpoint PRNG key for an independent reload replicate."""

import os

import jax

from vmcnet.train import runners
from vmcnet.utils import io


SEED = int(os.environ["VMC_RELOAD_SEED"])
_original_reload = io.reload_vmc_state


def _seeded_reload(*args, **kwargs):
    epoch, data, params, optimizer_state, _ = _original_reload(*args, **kwargs)
    return epoch, data, params, optimizer_state, jax.random.PRNGKey(SEED)


io.reload_vmc_state = _seeded_reload
runners.run_molecule()
