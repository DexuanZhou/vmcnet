"""Tests for multilevel SPRING and WSSR state prolongation."""

import copy

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import optax

from tests.test_utils import get_default_config_with_chosen_model
from vmcnet.models import construct, multilevel as model_multilevel
from vmcnet.updates import multilevel, wssr


def _make_nested_params():
    coarse_config = get_default_config_with_chosen_model("ferminet").model
    fine_config = copy.deepcopy(coarse_config)
    coarse_config.ndeterminants = 2
    fine_config.ndeterminants = 2
    coarse_config.backflow.ndense_list = ((8, 3), (8,))
    fine_config.backflow.ndense_list = ((12, 6), (12, 6), (12,))
    positions = jnp.zeros((1, 2, 3))
    ion_pos = jnp.zeros((1, 3))
    ion_charges = jnp.asarray([2.0])
    nelec = jnp.asarray([1, 1])

    def initialize(config, key):
        model = construct.get_model_from_config(
            config, nelec, ion_pos, ion_charges, dtype=jnp.float32
        )
        return model.init(key, positions)

    coarse_params = initialize(coarse_config, jax.random.PRNGKey(0))
    fine_template = initialize(fine_config, jax.random.PRNGKey(1))
    fine_params = model_multilevel.prolongate_ferminet_params(
        coarse_params,
        fine_template,
        coarse_config,
        fine_config,
        nspins=2,
    )
    return coarse_config, fine_config, coarse_params, fine_params


def test_spring_trace_and_schedule_count_are_preserved():
    coarse_config, fine_config, coarse_params, fine_params = _make_nested_params()
    optimizer = optax.sgd(lambda step: 0.1 / (step + 1), momentum=0)
    coarse_state = optimizer.init(coarse_params)
    coarse_update = jax.tree_util.tree_map(jnp.ones_like, coarse_params)
    _, coarse_state = optimizer.update(coarse_update, coarse_state, coarse_params)

    fine_state = multilevel.prolongate_spring_optimizer_state(
        coarse_state,
        fine_params,
        coarse_config,
        fine_config,
        nspins=2,
    )
    expected_trace = model_multilevel.prolongate_ferminet_tangent(
        coarse_state[0].trace,
        fine_params,
        coarse_config,
        fine_config,
        nspins=2,
    )
    for actual, expected in zip(
        jax.tree_util.tree_leaves(fine_state[0].trace),
        jax.tree_util.tree_leaves(expected_trace),
    ):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(fine_state[1].count, coarse_state[1].count)


def test_wssr_parameter_axis_memories_are_zero_extended():
    coarse_config, fine_config, coarse_params, fine_params = _make_nested_params()
    coarse_flat, _ = jax.flatten_util.ravel_pytree(coarse_params)
    fine_flat, _ = jax.flatten_util.ravel_pytree(fine_params)
    rows = model_multilevel.ferminet_flat_prolongation_indices(
        coarse_params,
        fine_params,
        coarse_config,
        fine_config,
        nspins=2,
    )

    core = wssr.initialize_wssr_warm_svd_core_state(
        coarse_flat.shape[0], 2, 3
    )
    sr_o = jnp.arange(coarse_flat.shape[0] * 3, dtype=jnp.float32).reshape(
        coarse_flat.shape[0], 3
    )
    u = sr_o + 0.5
    core = core._replace(sr_o=sr_o, u=u, has_u=jnp.asarray(True))
    optimizer = optax.sgd(lambda step: 0.1 / (step + 1), momentum=0)
    optax_state = optimizer.init(coarse_params)
    _, optax_state = optimizer.update(
        jax.tree_util.tree_map(jnp.ones_like, coarse_params),
        optax_state,
        coarse_params,
    )
    coarse_state = wssr.WSSRSolutionRecurrenceOptimizerState(
        core_state=core,
        optax_state=optax_state,
        solution_state=jnp.arange(coarse_flat.shape[0], dtype=jnp.float32),
    )

    fine_state = multilevel.prolongate_wssr_optimizer_state(
        coarse_state,
        coarse_params,
        fine_params,
        coarse_config,
        fine_config,
        nspins=2,
    )
    np.testing.assert_array_equal(fine_state.core_state.sr_o[rows], sr_o)
    np.testing.assert_array_equal(fine_state.core_state.u[rows], u)
    np.testing.assert_array_equal(
        fine_state.solution_state[rows], coarse_state.solution_state
    )
    new_mask = jnp.ones((fine_flat.shape[0],), dtype=bool).at[rows].set(False)
    np.testing.assert_array_equal(fine_state.core_state.sr_o[new_mask], 0)
    np.testing.assert_array_equal(fine_state.core_state.u[new_mask], 0)
    np.testing.assert_array_equal(fine_state.solution_state[new_mask], 0)
    np.testing.assert_array_equal(
        fine_state.optax_state[1].count, coarse_state.optax_state[1].count
    )


def test_wssr_previous_theta_uses_actual_fine_parameters():
    coarse_config, fine_config, coarse_params, fine_params = _make_nested_params()
    coarse_flat, _ = jax.flatten_util.ravel_pytree(coarse_params)
    fine_flat, _ = jax.flatten_util.ravel_pytree(fine_params)
    core = wssr.initialize_wssr_warm_svd_core_state(
        coarse_flat.shape[0], 2, 3
    )
    optimizer = optax.sgd(lambda step: 0.1 / (step + 1), momentum=0)
    coarse_state = wssr.WSSRTransportedGradientOptimizerState(
        core_state=core,
        optax_state=optimizer.init(coarse_params),
        transported_gradient=jnp.ones_like(coarse_flat),
        previous_theta=coarse_flat,
        transport_initialized=jnp.asarray(True),
    )
    fine_state = multilevel.prolongate_wssr_optimizer_state(
        coarse_state,
        coarse_params,
        fine_params,
        coarse_config,
        fine_config,
        nspins=2,
    )
    np.testing.assert_array_equal(fine_state.previous_theta, fine_flat)
