"""Audit strict equality and parameter counts for the proposed C hierarchy."""

import copy
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from vmcnet.models import construct, multilevel
from vmcnet.train import default_config


SPECS = (
    ((64, 8), (64,)),
    ((128, 16), (128, 16), (128,)),
    ((256, 16), (256, 16), (256, 16), (256,)),
)


def _config(spec):
    config = default_config.get_default_config()
    config.model = default_config.choose_model_type_in_model_config(config.model)
    config.model.ndeterminants = 16
    config.model.backflow.ndense_list = spec
    return config.model


def _count(params):
    return int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(params)))


def main():
    nelec = jnp.asarray((4, 2))
    ion_pos = jnp.zeros((1, 3))
    ion_charges = jnp.asarray((6.0,))
    positions = jax.random.normal(jax.random.PRNGKey(0), (8, 6, 3))
    configs = [_config(spec) for spec in SPECS]
    models = [
        construct.get_model_from_config(
            config, nelec, ion_pos, ion_charges, dtype=jnp.float32
        )
        for config in configs
    ]
    params = [models[0].init(jax.random.PRNGKey(1), positions[:1])]
    transition_errors = []
    for level in range(1, len(models)):
        template = models[level].init(jax.random.PRNGKey(level + 1), positions[:1])
        params.append(
            multilevel.prolongate_ferminet_params(
                params[-1],
                template,
                configs[level - 1],
                configs[level],
                nspins=2,
            )
        )
        coarse_sign, coarse_log = models[level - 1].apply(params[-2], positions)
        fine_sign, fine_log = models[level].apply(params[-1], positions)
        transition_errors.append(
            {
                "transition": f"L{level - 1}->L{level}",
                "sign_mismatches": int(np.count_nonzero(coarse_sign != fine_sign)),
                "max_logabs_error": float(np.max(np.abs(fine_log - coarse_log))),
            }
        )
    report = {
        "specs": SPECS,
        "parameter_counts": [_count(level_params) for level_params in params],
        "transition_errors": transition_errors,
    }
    path = Path(__file__).with_name("production_level_audit.json")
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
