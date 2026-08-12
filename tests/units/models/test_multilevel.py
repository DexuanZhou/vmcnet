"""Tests for function-preserving multilevel FermiNet prolongation."""

import copy

import jax
import jax.numpy as jnp
import numpy as np

from tests.test_utils import get_default_config_with_chosen_model
from vmcnet.models import construct, multilevel


def _make_level(config, key, positions):
    ion_pos = jnp.zeros((1, 3))
    ion_charges = jnp.asarray([2.0])
    nelec = jnp.asarray([1, 1])
    model = construct.get_model_from_config(
        config, nelec, ion_pos, ion_charges, dtype=jnp.float32
    )
    return model, model.init(key, positions[:1])


def _make_nested_configs():
    coarse = get_default_config_with_chosen_model("ferminet").model
    fine = copy.deepcopy(coarse)
    coarse.ndeterminants = 2
    fine.ndeterminants = 2
    # The raw two-electron stream has width four.  Avoid choosing that exact
    # width for only one level, which would change whether block zero has a
    # residual skip and therefore is not a nested architecture pair.
    coarse.backflow.ndense_list = ((8, 3), (8,))
    fine.backflow.ndense_list = ((12, 6), (12, 6), (12,))
    return coarse, fine


def test_widen_and_deepen_preserve_wavefunction_and_position_derivatives():
    coarse_config, fine_config = _make_nested_configs()
    positions = jax.random.normal(jax.random.PRNGKey(1), (3, 2, 3))
    coarse_model, coarse_params = _make_level(
        coarse_config, jax.random.PRNGKey(2), positions
    )
    fine_model, fine_template = _make_level(
        fine_config, jax.random.PRNGKey(3), positions
    )

    fine_params = multilevel.prolongate_ferminet_params(
        coarse_params,
        fine_template,
        coarse_config,
        fine_config,
        nspins=2,
    )

    coarse_sign, coarse_logabs = coarse_model.apply(coarse_params, positions)
    fine_sign, fine_logabs = fine_model.apply(fine_params, positions)
    np.testing.assert_array_equal(fine_sign, coarse_sign)
    np.testing.assert_allclose(fine_logabs, coarse_logabs, rtol=2e-6, atol=2e-6)

    coarse_position_grad = jax.grad(
        lambda x: jnp.sum(coarse_model.apply(coarse_params, x)[1])
    )(positions)
    fine_position_grad = jax.grad(
        lambda x: jnp.sum(fine_model.apply(fine_params, x)[1])
    )(positions)
    np.testing.assert_allclose(
        fine_position_grad, coarse_position_grad, rtol=2e-5, atol=2e-5
    )

    # The kinetic energy depends on second position derivatives, so equality of
    # values and first derivatives alone is not a sufficient VMC regression.
    coarse_hessian = jax.hessian(
        lambda x: coarse_model.apply(coarse_params, x[None, ...])[1][0]
    )(positions[0])
    fine_hessian = jax.hessian(
        lambda x: fine_model.apply(fine_params, x[None, ...])[1][0]
    )(positions[0])
    np.testing.assert_allclose(fine_hessian, coarse_hessian, rtol=5e-5, atol=5e-5)


def test_sidecar_is_disconnected_but_has_a_trainable_orbital_coupling():
    coarse_config, fine_config = _make_nested_configs()
    positions = jax.random.normal(jax.random.PRNGKey(4), (2, 2, 3))
    _, coarse_params = _make_level(coarse_config, jax.random.PRNGKey(5), positions)
    fine_model, fine_template = _make_level(
        fine_config, jax.random.PRNGKey(6), positions
    )
    fine_params = multilevel.prolongate_ferminet_params(
        coarse_params,
        fine_template,
        coarse_config,
        fine_config,
        nspins=2,
        keep_trainable_sidecar=True,
        sidecar_scale=0.1,
    )

    gradient = jax.grad(lambda p: jnp.sum(fine_model.apply(p, positions)[1]))(
        fine_params
    )
    orbital_gradient = gradient["params"]["FermiNetOrbitalLayer_0"][
        "SplitDense_0"
    ]["_dense_layers_0"]["kernel"]
    coarse_width = coarse_config.backflow.ndense_list[-1][0]
    assert jnp.linalg.norm(orbital_gradient[coarse_width:, :]) > 0


def test_old_parameter_score_coordinates_are_preserved():
    """The fine score agrees with the coarse score on embedded coordinates."""
    coarse_config, fine_config = _make_nested_configs()
    positions = jax.random.normal(jax.random.PRNGKey(9), (2, 2, 3))
    coarse_model, coarse_params = _make_level(
        coarse_config, jax.random.PRNGKey(10), positions
    )
    fine_model, fine_template = _make_level(
        fine_config, jax.random.PRNGKey(11), positions
    )
    fine_params = multilevel.prolongate_ferminet_params(
        coarse_params,
        fine_template,
        coarse_config,
        fine_config,
        nspins=2,
    )

    coarse_score = jax.grad(
        lambda p: jnp.sum(coarse_model.apply(p, positions)[1])
    )(coarse_params)
    fine_score = jax.grad(
        lambda p: jnp.sum(fine_model.apply(p, positions)[1])
    )(fine_params)
    coarse_score_flat, _ = jax.flatten_util.ravel_pytree(coarse_score)
    fine_score_flat, _ = jax.flatten_util.ravel_pytree(fine_score)
    fine_rows = multilevel.ferminet_flat_prolongation_indices(
        coarse_params,
        fine_params,
        coarse_config,
        fine_config,
        nspins=2,
    )
    np.testing.assert_allclose(
        fine_score_flat[fine_rows], coarse_score_flat, rtol=2e-5, atol=2e-5
    )


def test_tangent_prolongation_is_zero_on_all_new_coordinates():
    coarse_config, fine_config = _make_nested_configs()
    positions = jnp.zeros((1, 2, 3))
    _, coarse_params = _make_level(coarse_config, jax.random.PRNGKey(7), positions)
    _, fine_params = _make_level(fine_config, jax.random.PRNGKey(8), positions)
    coarse_tangent = jax.tree_util.tree_map(jnp.ones_like, coarse_params)

    fine_tangent = multilevel.prolongate_ferminet_tangent(
        coarse_tangent,
        fine_params,
        coarse_config,
        fine_config,
        nspins=2,
    )
    coarse_size = sum(leaf.size for leaf in jax.tree_util.tree_leaves(coarse_tangent))
    fine_nonzero = sum(
        int(jnp.count_nonzero(leaf)) for leaf in jax.tree_util.tree_leaves(fine_tangent)
    )
    assert fine_nonzero == coarse_size


def test_determinant_growth_is_rejected_until_safe_gates_exist():
    coarse_config, fine_config = _make_nested_configs()
    fine_config.ndeterminants = 4
    try:
        multilevel.validate_nested_ferminet_configs(coarse_config, fine_config)
    except ValueError as error:
        assert "determinant-count growth" in str(error)
    else:
        raise AssertionError("unsafe determinant-count growth was accepted")
