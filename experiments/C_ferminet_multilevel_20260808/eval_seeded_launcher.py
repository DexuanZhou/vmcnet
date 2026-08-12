"""Use one paired, independent key for all frozen endpoint evaluations."""

import jax

from vmcnet.train import runners
from vmcnet.utils import io


FOLD_TAG = 0x4D554C54
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
