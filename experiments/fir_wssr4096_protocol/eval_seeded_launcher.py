"""Run standard frozen evaluation with a reproducibly independent reload PRNG."""
import os
import jax
from vmcnet.train import runners

tag = int(os.environ["EVAL_PRNG_TAG"])
original = runners.utils.io.reload_vmc_state

def seeded_reload(*args, **kwargs):
    epoch, data, params, optimizer_state, key = original(*args, **kwargs)
    key = jax.vmap(lambda k: jax.random.fold_in(k, tag))(key) if key.ndim == 2 else jax.random.fold_in(key, tag)
    return epoch, data, params, optimizer_state, key

runners.utils.io.reload_vmc_state = seeded_reload
runners.run_molecule()
