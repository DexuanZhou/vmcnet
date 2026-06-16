"""Tests for pure WSSR SVD numerical core."""

import jax
import jax.numpy as jnp
import numpy as np

import vmcnet.physics as physics
import vmcnet.train.default_config as default_config
from vmcnet.updates.parse_optimizer_config import initialize_optimizer
from tests.test_utils import assert_pytree_allclose
from vmcnet.updates import wssr


def _tiny_params():
    return {
        "w": jnp.array([1.0, -0.5]),
        "b": jnp.array(0.25),
    }


def _tiny_positions():
    return jnp.array(
        [
            [1.0, -2.0],
            [0.5, 1.5],
            [-1.0, 0.25],
            [2.0, 0.0],
        ]
    )


def _log_psi_apply(params, position):
    return (
        params["w"][0] * position[0]
        + params["w"][1] * position[1]
        + params["b"] * position[0] * position[1]
    )


def _local_energy_fn(params, position):
    del params
    return jnp.sum(position**2)


def _assert_tree_all_finite(tree):
    for leaf in jax.tree_util.tree_leaves(tree):
        assert jnp.all(jnp.isfinite(leaf))


def test_initialize_wssr_core_state_shapes_and_ranks():
    state = wssr.initialize_wssr_core_state(
        num_params=5, sr_rank=3, sr_rank_max=7, dtype=jnp.float32
    )

    assert state.sr_o.shape == (5, 7)
    assert state.ek.shape == (7,)
    assert state.sr_rank0 == 0
    assert 0 <= state.sr_rank0 <= state.sr_rank <= 7


def test_default_config_contains_wssr_svd_without_changing_default_optimizer():
    config = default_config.get_default_config()

    assert config.vmc.optimizer_type == "spring"
    assert config.vmc.optimizer.wssr_svd.learning_rate == 5e-2
    assert config.vmc.optimizer.wssr_svd.sr_rank == 10
    assert config.vmc.optimizer.wssr_svd.sr_rank_max == 100
    assert "damping_decay" not in config.vmc.optimizer.wssr_svd
    assert "damping_min" not in config.vmc.optimizer.wssr_svd


def test_default_config_contains_wssr_sketch_placeholder():
    config = default_config.get_default_config()

    assert config.vmc.optimizer.wssr_sketch.learning_rate == 5e-2
    assert config.vmc.optimizer.wssr_sketch.sr_rank == 10
    assert config.vmc.optimizer.wssr_sketch.sr_rank_max == 100


def test_center_and_scale_score_matrix_uses_julia_convention():
    params = _tiny_params()
    positions = _tiny_positions()

    o_cur, _ = wssr.center_and_scale_score_matrix(
        _log_psi_apply, params, positions
    )

    def ravel_grad(position):
        grad = jax.grad(_log_psi_apply, argnums=0)(params, position)
        return jax.flatten_util.ravel_pytree(grad)[0]

    raw_scores = jax.vmap(ravel_grad)(positions)
    expected = (raw_scores - jnp.mean(raw_scores, axis=0, keepdims=True)).T
    expected = expected / jnp.sqrt(positions.shape[0])

    assert o_cur.shape == (3, positions.shape[0])
    np.testing.assert_allclose(o_cur, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(jnp.mean(o_cur, axis=1), jnp.zeros((3,)), atol=1e-6)


def test_center_and_scale_energy_residuals_uses_julia_convention():
    local_energies = jnp.array([1.0, 3.0, -2.0, 6.0])
    energy = jnp.mean(local_energies)

    e_cur = wssr.center_and_scale_energy_residuals(local_energies, energy)

    expected = (local_energies - energy) / jnp.sqrt(local_energies.shape[0])
    np.testing.assert_allclose(e_cur, expected, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(jnp.sum(e_cur), 0.0, atol=1e-7)


def test_augment_wssr_system_uses_current_batch_without_history():
    o_cur = jnp.arange(12.0).reshape(3, 4)
    e_cur = jnp.arange(4.0)
    state = wssr.initialize_wssr_core_state(3, sr_rank=2, sr_rank_max=5)

    o_aug, e_aug = wssr.augment_wssr_system(o_cur, e_cur, state, eta=0.9)

    np.testing.assert_allclose(o_aug, o_cur)
    np.testing.assert_allclose(e_aug, e_cur)


def test_augment_wssr_system_mixes_history_and_current_batch():
    o_cur = jnp.ones((3, 2))
    e_cur = jnp.array([2.0, 4.0])
    state = wssr.WSSRCoreState(
        sr_o=jnp.arange(15.0).reshape(3, 5),
        ek=jnp.arange(5.0),
        sr_rank0=2,
        sr_rank=3,
    )

    o_aug, e_aug = wssr.augment_wssr_system(o_cur, e_cur, state, eta=0.25)

    expected_o = jnp.concatenate(
        [jnp.sqrt(0.25) * state.sr_o[:, :2], jnp.sqrt(0.75) * o_cur],
        axis=1,
    )
    expected_e = jnp.concatenate(
        [jnp.sqrt(0.25) * state.ek[:2], jnp.sqrt(0.75) * e_cur],
        axis=0,
    )
    np.testing.assert_allclose(o_aug, expected_o, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(e_aug, expected_e, rtol=1e-6, atol=1e-6)


def test_wssr_svd_core_returns_finite_gradient_like_update_and_updates_history():
    state = wssr.initialize_wssr_core_state(
        num_params=3, sr_rank=2, sr_rank_max=4, dtype=jnp.float32
    )
    o_aug = jnp.array(
        [
            [1.0, 0.0, 2.0],
            [0.0, 1.0, -1.0],
            [1.0, 1.0, 0.5],
        ]
    )
    e_aug = jnp.array([0.25, -0.5, 0.75])

    result = wssr.wssr_svd_core_update(
        o_aug,
        e_aug,
        state,
        damping=0.05,
        norm_constraint=10.0,
        sr_rank_max=4,
        constrain_update_norm=False,
    )

    assert result.grad_like_update.shape == (3,)
    assert jnp.all(jnp.isfinite(result.grad_like_update))
    assert result.active_rank == result.state.sr_rank0
    assert 0 <= result.state.sr_rank0 <= result.state.sr_rank <= 4
    assert jnp.any(result.state.sr_o != 0.0)
    assert jnp.any(result.state.ek != 0.0)


def test_wssr_svd_core_matches_spec_formula_without_lr_or_minus_sign():
    state = wssr.initialize_wssr_core_state(
        num_params=3, sr_rank=2, sr_rank_max=4, dtype=jnp.float32
    )
    o_aug = jnp.array(
        [
            [1.0, 0.0, 2.0],
            [0.0, 1.0, -1.0],
            [1.0, 1.0, 0.5],
        ]
    )
    e_aug = jnp.array([0.25, -0.5, 0.75])
    damping = 0.05

    result = wssr.wssr_svd_core_update(
        o_aug,
        e_aug,
        state,
        damping=damping,
        norm_constraint=10.0,
        sr_rank_max=4,
        constrain_update_norm=False,
    )

    u, singular_values, vh = jnp.linalg.svd(o_aug, full_matrices=False)
    singular_values = singular_values[: state.sr_rank]
    u = u[:, : state.sr_rank]
    active_rank = int(jnp.sum(singular_values / singular_values[0] > damping))
    u_active = u[:, :active_rank]
    s_active = singular_values[:active_rank]
    sigma0 = 1.0 / jnp.square(damping * jnp.abs(singular_values[0]))
    force = o_aug @ e_aug
    projected_force = u_active.T @ force
    projected_force = projected_force * (jnp.square(1.0 / s_active) - sigma0)
    expected = u_active @ projected_force + sigma0 * force

    np.testing.assert_allclose(
        result.grad_like_update, expected, rtol=1e-5, atol=1e-5
    )
    assert not np.allclose(result.grad_like_update, -expected, rtol=1e-5, atol=1e-5)


def test_wssr_svd_core_applies_safe_norm_constraint():
    update = jnp.array([3.0, 4.0])
    constrained = wssr.constrain_norm(update, norm_constraint=1.0)

    np.testing.assert_allclose(jnp.linalg.norm(constrained), 1.0, rtol=1e-6)

    zero_update = jnp.zeros((2,))
    zero_constrained = wssr.constrain_norm(zero_update, norm_constraint=1.0)
    np.testing.assert_allclose(zero_constrained, zero_update)
    assert jnp.all(jnp.isfinite(zero_constrained))


def test_wssr_svd_core_handles_zero_score_matrix_without_nans():
    state = wssr.initialize_wssr_core_state(
        num_params=3, sr_rank=2, sr_rank_max=4, dtype=jnp.float32
    )

    result = wssr.wssr_svd_core_update(
        jnp.zeros((3, 4)),
        jnp.ones((4,)),
        state,
        damping=0.05,
        norm_constraint=1.0,
        sr_rank_max=4,
    )

    np.testing.assert_allclose(result.grad_like_update, jnp.zeros((3,)))
    assert result.active_rank == 0
    assert result.state.sr_rank0 == 0
    assert jnp.all(jnp.isfinite(result.grad_like_update))


def test_compute_wssr_svd_core_update_preserves_param_tree_structure():
    params = _tiny_params()
    positions = _tiny_positions()
    local_energies = jnp.array([1.0, 3.0, -2.0, 6.0])
    energy = jnp.mean(local_energies)
    flat_params, _ = jax.flatten_util.ravel_pytree(params)
    state = wssr.initialize_wssr_core_state(
        num_params=flat_params.shape[0], sr_rank=2, sr_rank_max=4
    )

    update_tree, new_state, active_rank = wssr.compute_wssr_svd_core_update(
        _log_psi_apply,
        params,
        positions,
        local_energies,
        energy,
        state,
        eta=0.9,
        damping=0.05,
        norm_constraint=10.0,
        sr_rank_max=4,
        constrain_update_norm=False,
    )

    assert jax.tree_util.tree_structure(update_tree) == jax.tree_util.tree_structure(
        params
    )
    _assert_tree_all_finite(update_tree)
    assert active_rank == new_state.sr_rank0
    assert 0 <= new_state.sr_rank0 <= new_state.sr_rank <= 4

    zero_tree = jax.tree_util.tree_map(jnp.zeros_like, update_tree)
    try:
        assert_pytree_allclose(update_tree, zero_tree)
    except AssertionError:
        pass
    else:
        raise AssertionError("Expected nonzero WSSR update for tiny fixture")


def test_initialize_optimizer_dispatches_wssr_svd_and_constructs_state():
    params = _tiny_params()
    data = _tiny_positions()
    config = default_config.get_default_config()
    config.vmc.nchains = data.shape[0]
    config.vmc.optimizer_type = "wssr_svd"

    update_param_fn, optimizer_state, key = initialize_optimizer(
        _log_psi_apply,
        _local_energy_fn,
        None,
        config.vmc,
        params,
        data,
        lambda x: x,
        lambda d, p: d,
        jax.random.PRNGKey(0),
        apply_pmap=False,
    )

    assert callable(update_param_fn)
    assert isinstance(optimizer_state, wssr.WSSROptimizerState)
    assert optimizer_state.core_state.sr_o.shape[0] == 3
    assert optimizer_state.core_state.ek.shape == (
        config.vmc.optimizer.wssr_svd.sr_rank_max,
    )
    assert 0 <= optimizer_state.core_state.sr_rank0 <= optimizer_state.core_state.sr_rank
    assert key.shape == (2,)


def test_wssr_svd_integrated_update_applies_optax_sign_and_lr_once():
    params = _tiny_params()
    data = _tiny_positions()
    config = default_config.get_default_config()
    config.vmc.nchains = data.shape[0]
    config.vmc.optimizer_type = "wssr_svd"
    config.vmc.optimizer.wssr_svd.schedule_type = "constant"
    config.vmc.optimizer.wssr_svd.learning_rate = 0.125
    config.vmc.optimizer.wssr_svd.constrain_norm = False

    update_param_fn, optimizer_state, key = initialize_optimizer(
        _log_psi_apply,
        _local_energy_fn,
        None,
        config.vmc,
        params,
        data,
        lambda x: x,
        lambda d, p: d,
        jax.random.PRNGKey(1),
        apply_pmap=False,
    )

    local_energies = jax.vmap(_local_energy_fn, in_axes=(None, 0))(params, data)
    energy, _, _ = physics.core.get_clipped_energies_and_stats(
        local_energies,
        config.vmc.nchains,
        None,
        config.vmc.nan_safe,
    )
    raw_update_tree, _, _ = wssr.compute_wssr_svd_core_update(
        _log_psi_apply,
        params,
        data,
        local_energies,
        energy,
        optimizer_state.core_state,
        config.vmc.optimizer.wssr_svd.eta,
        config.vmc.optimizer.wssr_svd.damping,
        config.vmc.optimizer.wssr_svd.norm_constraint,
        config.vmc.optimizer.wssr_svd.sr_rank_max,
        sr_scale=config.vmc.optimizer.wssr_svd.sr_scale,
        constrain_update_norm=False,
    )
    expected_params = jax.tree_util.tree_map(
        lambda param, grad_like: param
        - config.vmc.optimizer.wssr_svd.learning_rate * grad_like,
        params,
        raw_update_tree,
    )

    new_params, new_data, new_optimizer_state, metrics, new_key = update_param_fn(
        params, data, optimizer_state, key
    )

    assert_pytree_allclose(new_params, expected_params, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(new_data, data)
    assert isinstance(new_optimizer_state, wssr.WSSROptimizerState)
    assert new_optimizer_state.core_state.sr_rank0 > 0
    assert set(metrics).issuperset({"energy", "variance", "energy_noclip"})
    assert jnp.all(jnp.isfinite(jnp.asarray(list(metrics.values()))))
    np.testing.assert_allclose(new_key, key)


def test_wssr_svd_integrated_norm_constraint_bounds_final_update_not_raw_sr():
    params = _tiny_params()
    data = _tiny_positions()
    config = default_config.get_default_config()
    config.vmc.nchains = data.shape[0]
    config.vmc.optimizer_type = "wssr_svd"
    config.vmc.optimizer.wssr_svd.schedule_type = "constant"
    config.vmc.optimizer.wssr_svd.learning_rate = 10.0
    config.vmc.optimizer.wssr_svd.constrain_norm = True
    config.vmc.optimizer.wssr_svd.norm_constraint = 0.01

    update_param_fn, optimizer_state, key = initialize_optimizer(
        _log_psi_apply,
        _local_energy_fn,
        None,
        config.vmc,
        params,
        data,
        lambda x: x,
        lambda d, p: d,
        jax.random.PRNGKey(2),
        apply_pmap=False,
    )

    local_energies = jax.vmap(_local_energy_fn, in_axes=(None, 0))(params, data)
    energy, _, _ = physics.core.get_clipped_energies_and_stats(
        local_energies,
        config.vmc.nchains,
        None,
        config.vmc.nan_safe,
    )
    raw_update_tree, _, _ = wssr.compute_wssr_svd_core_update(
        _log_psi_apply,
        params,
        data,
        local_energies,
        energy,
        optimizer_state.core_state,
        config.vmc.optimizer.wssr_svd.eta,
        config.vmc.optimizer.wssr_svd.damping,
        config.vmc.optimizer.wssr_svd.norm_constraint,
        config.vmc.optimizer.wssr_svd.sr_rank_max,
        sr_scale=config.vmc.optimizer.wssr_svd.sr_scale,
        constrain_update_norm=False,
    )
    unconstrained_optax_update = jax.tree_util.tree_map(
        lambda grad_like: -config.vmc.optimizer.wssr_svd.learning_rate * grad_like,
        raw_update_tree,
    )
    sqrt_constraint = jnp.sqrt(config.vmc.optimizer.wssr_svd.norm_constraint)
    assert wssr.tree_l2_norm(unconstrained_optax_update) > sqrt_constraint

    new_params, *_ = update_param_fn(params, data, optimizer_state, key)

    final_update = jax.tree_util.tree_map(
        lambda new_param, old_param: new_param - old_param,
        new_params,
        params,
    )
    final_update_norm = wssr.tree_l2_norm(final_update)

    assert final_update_norm <= sqrt_constraint + 1e-6
    assert final_update_norm > 0.0
    assert final_update_norm < (
        config.vmc.optimizer.wssr_svd.learning_rate * sqrt_constraint
    )


def test_initialize_wssr_svd_rejects_pmap_until_jit_safe_core_exists():
    params = _tiny_params()
    data = _tiny_positions()
    config = default_config.get_default_config()
    config.vmc.nchains = data.shape[0]
    config.vmc.optimizer_type = "wssr_svd"

    try:
        initialize_optimizer(
            _log_psi_apply,
            _local_energy_fn,
            None,
            config.vmc,
            params,
            data,
            lambda x: x,
            lambda d, p: d,
            jax.random.PRNGKey(0),
            apply_pmap=True,
        )
    except NotImplementedError as err:
        assert "apply_pmap=False" in str(err)
    else:
        raise AssertionError("Expected wssr_svd to reject apply_pmap=True")


def test_initialize_wssr_sketch_placeholder_raises_clear_error():
    try:
        wssr.initialize_wssr_sketch()
    except NotImplementedError as err:
        assert str(err) == "wssr_sketch is not implemented yet"
    else:
        raise AssertionError("Expected wssr_sketch placeholder to raise")


def test_initialize_optimizer_dispatches_wssr_sketch_placeholder():
    params = _tiny_params()
    data = _tiny_positions()
    config = default_config.get_default_config()
    config.vmc.nchains = data.shape[0]
    config.vmc.optimizer_type = "wssr_sketch"

    try:
        initialize_optimizer(
            _log_psi_apply,
            _local_energy_fn,
            None,
            config.vmc,
            params,
            data,
            lambda x: x,
            lambda d, p: d,
            jax.random.PRNGKey(0),
            apply_pmap=False,
        )
    except NotImplementedError as err:
        assert str(err) == "wssr_sketch is not implemented yet"
    else:
        raise AssertionError("Expected wssr_sketch dispatch to reach placeholder")
