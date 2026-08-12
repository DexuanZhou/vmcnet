"""End-to-end checkpoint conversion test for strict FermiNet levels."""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import optax

from tests.test_utils import get_default_config_with_chosen_model
from vmcnet.mcmc import position_amplitude_core as pacore
from vmcnet.models import construct
from vmcnet.train import multilevel_checkpoint
from vmcnet.utils import io


def test_spring_checkpoint_round_trip_and_wavefunction_equality(tmp_path):
    coarse_config = get_default_config_with_chosen_model("ferminet")
    fine_config = copy.deepcopy(coarse_config)
    for config in (coarse_config, fine_config):
        config.problem.ion_pos = ((0.0, 0.0, 0.0),)
        config.problem.ion_charges = (2.0,)
        config.problem.nelec = (1, 1)
        config.model.ndeterminants = 2
        config.vmc.optimizer_type = "spring"
    coarse_config.model.backflow.ndense_list = ((8, 3), (8,))
    fine_config.model.backflow.ndense_list = ((12, 6), (12, 6), (12,))

    positions = jax.random.normal(jax.random.PRNGKey(0), (4, 2, 3))
    model = construct.get_model_from_config(
        coarse_config.model,
        jnp.asarray((1, 1)),
        jnp.zeros((1, 3)),
        jnp.asarray((2.0,)),
        dtype=jnp.float32,
    )
    coarse_params = model.init(jax.random.PRNGKey(1), positions[:1])
    _, logabs = model.apply(coarse_params, positions)
    data = pacore.make_position_amplitude_data(positions, logabs, None)
    optimizer = optax.sgd(lambda step: 0.1 / (step + 1), momentum=0)
    optimizer_state = optimizer.init(coarse_params)
    _, optimizer_state = optimizer.update(
        jax.tree_util.tree_map(jnp.ones_like, coarse_params),
        optimizer_state,
        coarse_params,
    )
    checkpoint = (7, data, coarse_params, optimizer_state, jax.random.PRNGKey(2))

    converted, report = multilevel_checkpoint.convert_checkpoint(
        coarse_config,
        fine_config,
        checkpoint,
        template_seed=3,
        verify_samples=4,
    )
    assert report["sign_mismatches"] == 0
    assert report["max_logabs_error"] < report["equality_atol"]
    assert report["fine_parameter_count"] > report["coarse_parameter_count"]
    fine_model = construct.get_model_from_config(
        fine_config.model,
        jnp.asarray((1, 1)),
        jnp.zeros((1, 3)),
        jnp.asarray((2.0,)),
        dtype=jnp.float32,
    )
    _, expected_cached_logabs = fine_model.apply(converted[2], positions)
    np.testing.assert_allclose(
        pacore.get_amplitude_from_data(converted[1]), expected_cached_logabs
    )

    io.save_vmc_state(str(tmp_path), "converted.npz", converted)
    reloaded = io.reload_vmc_state(str(tmp_path), "converted.npz")
    assert reloaded[0] == 7
    np.testing.assert_array_equal(reloaded[4], checkpoint[4])
    np.testing.assert_array_equal(
        reloaded[3][1].count, checkpoint[3][1].count
    )
