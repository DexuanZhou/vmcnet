"""Tests for pure WSSR SVD numerical core."""

import jax
import jax.numpy as jnp
import numpy as np
import chex

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


def _assert_wssr_state_all_finite(state):
    assert jnp.all(jnp.isfinite(state.sr_o))
    assert jnp.all(jnp.isfinite(state.ek))
    assert jnp.all(jnp.isfinite(state.sr_rank0))
    assert jnp.all(jnp.isfinite(state.sr_rank))
    if hasattr(state, "u"):
        assert jnp.all(jnp.isfinite(state.u))
    if hasattr(state, "has_u"):
        assert jnp.all(jnp.isfinite(state.has_u))


def _legacy_dynamic_augment_wssr_system(o_cur, e_cur, state, eta):
    active_rank = int(state.sr_rank0)
    if active_rank == 0:
        return o_cur, e_cur

    return (
        jnp.concatenate(
            [
                jnp.sqrt(eta) * state.sr_o[:, :active_rank],
                jnp.sqrt(1.0 - eta) * o_cur,
            ],
            axis=1,
        ),
        jnp.concatenate(
            [
                jnp.sqrt(eta) * state.ek[:active_rank],
                jnp.sqrt(1.0 - eta) * e_cur,
            ],
            axis=0,
        ),
    )


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
    assert config.vmc.optimizer.wssr_sketch.sketch_oversampling == 5
    assert config.vmc.optimizer.wssr_sketch.sketch_n_iter == 1


def test_default_config_contains_wssr_warm_svd():
    config = default_config.get_default_config()

    assert config.vmc.optimizer.wssr_warm_svd.learning_rate == 5e-2
    assert config.vmc.optimizer.wssr_warm_svd.sr_rank == 10
    assert config.vmc.optimizer.wssr_warm_svd.sr_rank_max == 100
    assert config.vmc.optimizer.wssr_warm_svd.svd_maxiter_initial == 8
    assert config.vmc.optimizer.wssr_warm_svd.svd_maxiter_warm == 2


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


def test_center_and_scale_score_matrix_matches_per_sample_ravel_construction():
    params = {
        "dense": {
            "w": jnp.array([[0.25, -0.75], [1.5, 0.5]]),
            "b": jnp.array([0.1, -0.2]),
        },
        "scale": jnp.array(0.3),
    }
    positions = _tiny_positions()

    def log_psi_apply(params, position):
        hidden = jnp.tanh(params["dense"]["w"] @ position + params["dense"]["b"])
        return jnp.sum(hidden) + params["scale"] * position[0] * position[1]

    o_cur, _ = wssr.center_and_scale_score_matrix(log_psi_apply, params, positions)

    def legacy_ravel_grad(position):
        grad = jax.grad(log_psi_apply, argnums=0)(params, position)
        return jax.flatten_util.ravel_pytree(grad)[0]

    legacy_scores = jax.vmap(legacy_ravel_grad)(positions)
    expected = legacy_scores - jnp.mean(legacy_scores, axis=0, keepdims=True)
    expected = expected.T / jnp.sqrt(positions.shape[0])

    np.testing.assert_allclose(o_cur, expected, rtol=1e-6, atol=1e-6)


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

    expected_o = jnp.concatenate([jnp.zeros((3, 5)), o_cur], axis=1)
    expected_e = jnp.concatenate([jnp.zeros((5,)), e_cur], axis=0)
    assert o_aug.shape == (3, 9)
    assert e_aug.shape == (9,)
    np.testing.assert_allclose(o_aug, expected_o)
    np.testing.assert_allclose(e_aug, expected_e)


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

    expected_hist_o = jnp.sqrt(0.25) * state.sr_o.at[:, 2:].set(0.0)
    expected_hist_e = jnp.sqrt(0.25) * state.ek.at[2:].set(0.0)
    expected_o = jnp.concatenate(
        [expected_hist_o, jnp.sqrt(0.75) * o_cur],
        axis=1,
    )
    expected_e = jnp.concatenate(
        [expected_hist_e, jnp.sqrt(0.75) * e_cur],
        axis=0,
    )
    assert o_aug.shape == (3, 7)
    assert e_aug.shape == (7,)
    np.testing.assert_allclose(o_aug, expected_o, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(e_aug, expected_e, rtol=1e-6, atol=1e-6)


def test_augment_wssr_system_is_jit_safe():
    o_cur = jnp.ones((3, 2))
    e_cur = jnp.array([2.0, 4.0])
    state = wssr.WSSRCoreState(
        sr_o=jnp.arange(15.0).reshape(3, 5),
        ek=jnp.arange(5.0),
        sr_rank0=jnp.array(2),
        sr_rank=jnp.array(3),
    )

    jitted_augment = jax.jit(wssr.augment_wssr_system)
    o_aug, e_aug = jitted_augment(o_cur, e_cur, state, eta=0.25)

    assert o_aug.shape == (3, 7)
    assert e_aug.shape == (7,)
    np.testing.assert_allclose(o_aug[:, 2:5], 0.0, atol=1e-6)
    np.testing.assert_allclose(e_aug[2:5], 0.0, atol=1e-6)
    np.testing.assert_allclose(o_aug[:, 5:], jnp.sqrt(0.75) * o_cur)
    np.testing.assert_allclose(e_aug[5:], jnp.sqrt(0.75) * e_cur)


def test_augment_wssr_system_force_matches_legacy_without_history():
    o_cur = jnp.arange(12.0, dtype=jnp.float32).reshape(3, 4)
    e_cur = jnp.array([0.25, -0.5, 0.75, 1.25], dtype=jnp.float32)
    state = wssr.WSSRCoreState(
        sr_o=jnp.arange(15.0, dtype=jnp.float32).reshape(3, 5),
        ek=jnp.arange(5.0, dtype=jnp.float32),
        sr_rank0=jnp.array(0),
        sr_rank=jnp.array(3),
    )

    o_fixed, e_fixed = wssr.augment_wssr_system(o_cur, e_cur, state, eta=0.25)
    o_dyn, e_dyn = _legacy_dynamic_augment_wssr_system(o_cur, e_cur, state, eta=0.25)

    chex.assert_trees_all_close(o_fixed @ e_fixed, o_dyn @ e_dyn)


def test_augment_wssr_system_force_matches_legacy_with_history():
    o_cur = jnp.array(
        [
            [1.0, -0.5, 0.25],
            [0.5, 1.5, -1.0],
            [2.0, 0.0, 0.75],
        ],
        dtype=jnp.float32,
    )
    e_cur = jnp.array([0.25, -0.5, 0.75], dtype=jnp.float32)
    state = wssr.WSSRCoreState(
        sr_o=jnp.arange(15.0, dtype=jnp.float32).reshape(3, 5) / 10.0,
        ek=jnp.array([1.0, -2.0, 0.5, 3.0, -1.0], dtype=jnp.float32),
        sr_rank0=jnp.array(2),
        sr_rank=jnp.array(3),
    )

    o_fixed, e_fixed = wssr.augment_wssr_system(o_cur, e_cur, state, eta=0.25)
    o_dyn, e_dyn = _legacy_dynamic_augment_wssr_system(o_cur, e_cur, state, eta=0.25)

    chex.assert_trees_all_close(o_fixed @ e_fixed, o_dyn @ e_dyn)


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


def test_randomized_svd_clips_rank_for_tiny_matrices():
    o_aug = jnp.arange(6.0).reshape(2, 3)

    u, singular_values, vh = wssr.randomized_svd(
        o_aug,
        target_rank=10,
        sketch_oversampling=5,
        sketch_n_iter=1,
        key=jax.random.PRNGKey(0),
    )

    assert u.shape == (2, 2)
    assert singular_values.shape == (2,)
    assert vh.shape == (2, 3)
    assert jnp.all(jnp.isfinite(singular_values))


def test_wssr_sketch_core_update_is_deterministic_with_fixed_key():
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
    key = jax.random.PRNGKey(3)

    result_1 = wssr.wssr_sketch_core_update(
        o_aug,
        e_aug,
        state,
        key,
        damping=0.05,
        norm_constraint=10.0,
        sr_rank_max=4,
        sketch_oversampling=2,
        sketch_n_iter=1,
        constrain_update_norm=False,
    )
    result_2 = wssr.wssr_sketch_core_update(
        o_aug,
        e_aug,
        state,
        key,
        damping=0.05,
        norm_constraint=10.0,
        sr_rank_max=4,
        sketch_oversampling=2,
        sketch_n_iter=1,
        constrain_update_norm=False,
    )

    np.testing.assert_allclose(result_1.grad_like_update, result_2.grad_like_update)
    np.testing.assert_allclose(result_1.state.sr_o, result_2.state.sr_o)
    np.testing.assert_allclose(result_1.state.ek, result_2.state.ek)
    assert result_1.active_rank == result_2.active_rank


def test_fixed_shape_randomized_svd_masks_inactive_rank_columns():
    state = wssr.initialize_wssr_core_state(
        num_params=4, sr_rank=2, sr_rank_max=4, dtype=jnp.float32
    )
    o_aug = jnp.array(
        [
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [1.0, -1.0, 0.5, 1.0],
        ],
        dtype=jnp.float32,
    )

    u, singular_values, vh = wssr.fixed_shape_randomized_svd(
        o_aug,
        state,
        key=jax.random.PRNGKey(17),
        sr_rank_max=4,
        sketch_oversampling=2,
        sketch_n_iter=0,
    )

    assert u.shape == (4, 4)
    assert singular_values.shape == (4,)
    assert vh.shape == (4, 4)
    assert jnp.all(jnp.isfinite(u))
    assert jnp.all(jnp.isfinite(singular_values))
    assert jnp.all(jnp.isfinite(vh))
    np.testing.assert_allclose(u[:, 2:], 0.0, atol=1e-6)
    np.testing.assert_allclose(singular_values[2:], 0.0, atol=1e-6)
    np.testing.assert_allclose(vh[2:, :], 0.0, atol=1e-6)


def test_wssr_sketch_core_jitted_matches_eager_update_and_history():
    state = wssr.WSSRCoreState(
        sr_o=jnp.array(
            [
                [0.2, -0.1, 0.0, 0.0],
                [0.4, 0.3, 0.0, 0.0],
                [-0.5, 0.25, 0.0, 0.0],
                [0.1, -0.2, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        ),
        ek=jnp.array([0.5, -0.25, 0.0, 0.0], dtype=jnp.float32),
        sr_rank0=jnp.array(2),
        sr_rank=jnp.array(2),
    )
    o_aug = jnp.array(
        [
            [2.0, 0.5, -1.0, 0.25],
            [0.0, 1.5, 0.5, -0.75],
            [1.0, -0.25, 1.25, 0.5],
            [0.25, 1.0, -0.5, 1.5],
        ],
        dtype=jnp.float32,
    )
    e_aug = jnp.array([0.5, -0.25, 0.125, -0.75], dtype=jnp.float32)
    key = jax.random.PRNGKey(123)
    kwargs = dict(
        damping=0.05,
        norm_constraint=10.0,
        sr_rank_max=4,
        sr_scale=1.5,
        sketch_oversampling=2,
        sketch_n_iter=1,
        constrain_update_norm=False,
    )

    eager = wssr.wssr_sketch_core_update(o_aug, e_aug, state, key=key, **kwargs)
    jitted_update = jax.jit(
        wssr.wssr_sketch_core_update,
        static_argnames=(
            "sr_rank_max",
            "sketch_oversampling",
            "sketch_n_iter",
            "constrain_update_norm",
        ),
    )
    jitted = jitted_update(o_aug, e_aug, state, key=key, **kwargs)
    jitted.grad_like_update.block_until_ready()

    chex.assert_trees_all_close(
        eager.grad_like_update, jitted.grad_like_update, rtol=1e-5, atol=1e-5
    )
    chex.assert_trees_all_close(eager.state.sr_o, jitted.state.sr_o)
    chex.assert_trees_all_close(eager.state.ek, jitted.state.ek)
    chex.assert_trees_all_close(eager.state.sr_rank0, jitted.state.sr_rank0)
    chex.assert_trees_all_close(eager.state.sr_rank, jitted.state.sr_rank)
    chex.assert_trees_all_close(eager.active_rank, jitted.active_rank)


def test_wssr_sketch_core_jits_and_preserves_rank_growth_history():
    state = wssr.initialize_wssr_core_state(
        num_params=4, sr_rank=2, sr_rank_max=4, dtype=jnp.float32
    )
    o_aug = jnp.array(
        [
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    e_aug = jnp.array([0.5, -0.25, 0.125, -0.75], dtype=jnp.float32)
    jitted_update = jax.jit(
        wssr.wssr_sketch_core_update,
        static_argnames=(
            "sr_rank_max",
            "sketch_oversampling",
            "sketch_n_iter",
            "constrain_update_norm",
        ),
    )

    first = jitted_update(
        o_aug,
        e_aug,
        state,
        key=jax.random.PRNGKey(5),
        damping=1e-4,
        norm_constraint=10.0,
        sr_rank_max=4,
        sr_scale=2.0,
        sketch_oversampling=2,
        sketch_n_iter=0,
        constrain_update_norm=False,
    )
    first.grad_like_update.block_until_ready()
    second = jitted_update(
        o_aug,
        e_aug,
        first.state,
        key=jax.random.PRNGKey(6),
        damping=1e-4,
        norm_constraint=10.0,
        sr_rank_max=4,
        sr_scale=2.0,
        sketch_oversampling=2,
        sketch_n_iter=0,
        constrain_update_norm=False,
    )
    second.grad_like_update.block_until_ready()

    _assert_wssr_state_all_finite(first.state)
    _assert_wssr_state_all_finite(second.state)
    assert first.state.sr_rank0 == 2
    assert first.state.sr_rank == 4
    assert 0 <= second.state.sr_rank0 <= second.state.sr_rank <= 4
    assert jnp.any(first.state.sr_o[:, :2] != 0.0)
    assert jnp.all(first.state.sr_o[:, 2:] == 0.0)


def test_initialize_optimizer_dispatches_wssr_sketch_and_constructs_state():
    params = _tiny_params()
    data = _tiny_positions()
    config = default_config.get_default_config()
    config.vmc.nchains = data.shape[0]
    config.vmc.optimizer_type = "wssr_sketch"

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
        config.vmc.optimizer.wssr_sketch.sr_rank_max,
    )
    assert 0 <= optimizer_state.core_state.sr_rank0 <= optimizer_state.core_state.sr_rank
    assert key.shape == (2,)


def test_wssr_sketch_integrated_update_preserves_structure_and_has_no_nans():
    params = _tiny_params()
    data = _tiny_positions()
    config = default_config.get_default_config()
    config.vmc.nchains = data.shape[0]
    config.vmc.optimizer_type = "wssr_sketch"
    config.vmc.optimizer.wssr_sketch.schedule_type = "constant"
    config.vmc.optimizer.wssr_sketch.learning_rate = 0.125
    config.vmc.optimizer.wssr_sketch.constrain_norm = False
    config.vmc.optimizer.wssr_sketch.sketch_oversampling = 2
    config.vmc.optimizer.wssr_sketch.sketch_n_iter = 1

    update_param_fn, optimizer_state, key = initialize_optimizer(
        _log_psi_apply,
        _local_energy_fn,
        None,
        config.vmc,
        params,
        data,
        lambda x: x,
        lambda d, p: d,
        jax.random.PRNGKey(4),
        apply_pmap=False,
    )

    new_params, new_data, new_optimizer_state, metrics, new_key = update_param_fn(
        params, data, optimizer_state, key
    )

    assert jax.tree_util.tree_structure(new_params) == jax.tree_util.tree_structure(
        params
    )
    _assert_tree_all_finite(new_params)
    _assert_wssr_state_all_finite(new_optimizer_state.core_state)
    assert isinstance(new_optimizer_state, wssr.WSSROptimizerState)
    assert (
        0
        <= new_optimizer_state.core_state.sr_rank0
        <= new_optimizer_state.core_state.sr_rank
        <= config.vmc.optimizer.wssr_sketch.sr_rank_max
    )
    assert set(metrics).issuperset({"energy", "variance", "energy_noclip"})
    assert jnp.all(jnp.isfinite(jnp.asarray(list(metrics.values()))))
    np.testing.assert_allclose(new_data, data)
    assert new_key.shape == key.shape
    assert not np.allclose(new_key, key)


def test_initialize_wssr_sketch_rejects_pmap_until_jit_safe_core_exists():
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
            apply_pmap=True,
        )
    except NotImplementedError as err:
        assert "apply_pmap=False" in str(err)
    else:
        raise AssertionError("Expected wssr_sketch to reject apply_pmap=True")


def test_wssr_warm_svd_initial_state_has_fixed_u_shape():
    state = wssr.initialize_wssr_warm_svd_core_state(
        num_params=3, sr_rank=2, sr_rank_max=4, dtype=jnp.float32
    )

    assert state.sr_o.shape == (3, 4)
    assert state.ek.shape == (4,)
    assert state.u.shape == (3, 4)
    assert state.has_u == jnp.array(False)
    assert 0 <= state.sr_rank0 <= state.sr_rank <= 4


def test_wssr_warm_svd_core_clips_rank_for_tiny_matrices():
    state = wssr.initialize_wssr_warm_svd_core_state(
        num_params=2, sr_rank=5, sr_rank_max=5, dtype=jnp.float32
    )
    o_aug = jnp.arange(6.0).reshape(2, 3)

    u, singular_values, vh, rank = wssr.warm_start_svd(
        o_aug,
        state,
        key=jax.random.PRNGKey(0),
        maxiter_initial=1,
        maxiter_warm=1,
    )

    assert rank == 2
    assert u.shape == (2, 2)
    assert singular_values.shape == (2,)
    assert vh.shape == (2, 3)
    assert jnp.all(jnp.isfinite(singular_values))


def test_wssr_warm_svd_first_and_second_updates_store_and_reuse_u():
    state = wssr.initialize_wssr_warm_svd_core_state(
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

    first = wssr.wssr_warm_svd_core_update(
        o_aug,
        e_aug,
        state,
        key=jax.random.PRNGKey(5),
        damping=0.05,
        norm_constraint=10.0,
        sr_rank_max=4,
        svd_maxiter_initial=2,
        svd_maxiter_warm=1,
        constrain_update_norm=False,
    )
    second = wssr.wssr_warm_svd_core_update(
        o_aug,
        e_aug,
        first.state,
        key=jax.random.PRNGKey(999),
        damping=0.05,
        norm_constraint=10.0,
        sr_rank_max=4,
        svd_maxiter_initial=2,
        svd_maxiter_warm=1,
        constrain_update_norm=False,
    )

    assert first.state.has_u == jnp.array(True)
    assert second.state.has_u == jnp.array(True)
    _assert_wssr_state_all_finite(first.state)
    _assert_wssr_state_all_finite(second.state)
    assert first.active_rank == first.state.sr_rank0
    assert second.active_rank == second.state.sr_rank0
    assert 0 <= second.state.sr_rank0 <= second.state.sr_rank <= 4
    assert jnp.any(first.state.u[:, :2] != 0.0)
    assert jnp.any(second.state.u[:, :2] != 0.0)


def test_wssr_warm_svd_core_jits_and_preserves_rank_growth_history():
    state = wssr.initialize_wssr_warm_svd_core_state(
        num_params=4, sr_rank=2, sr_rank_max=4, dtype=jnp.float32
    )
    o_aug = jnp.array(
        [
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    e_aug = jnp.array([0.5, -0.25, 0.125, -0.75], dtype=jnp.float32)
    jitted_update = jax.jit(
        wssr.wssr_warm_svd_core_update,
        static_argnames=(
            "sr_rank_max",
            "svd_maxiter_initial",
            "svd_maxiter_warm",
            "constrain_update_norm",
        ),
    )

    first = jitted_update(
        o_aug,
        e_aug,
        state,
        key=jax.random.PRNGKey(5),
        damping=1e-4,
        norm_constraint=10.0,
        sr_rank_max=4,
        sr_scale=2.0,
        svd_maxiter_initial=3,
        svd_maxiter_warm=1,
        constrain_update_norm=False,
    )
    first.grad_like_update.block_until_ready()
    second = jitted_update(
        o_aug,
        e_aug,
        first.state,
        key=jax.random.PRNGKey(6),
        damping=1e-4,
        norm_constraint=10.0,
        sr_rank_max=4,
        sr_scale=2.0,
        svd_maxiter_initial=3,
        svd_maxiter_warm=1,
        constrain_update_norm=False,
    )
    second.grad_like_update.block_until_ready()

    _assert_wssr_state_all_finite(first.state)
    _assert_wssr_state_all_finite(second.state)
    assert first.state.has_u == jnp.array(True)
    assert second.state.has_u == jnp.array(True)
    assert first.state.sr_rank0 == 2
    assert first.state.sr_rank == 4
    assert 0 <= second.state.sr_rank0 <= second.state.sr_rank <= 4
    assert jnp.any(first.state.sr_o[:, :2] != 0.0)
    assert jnp.all(first.state.sr_o[:, 2:] == 0.0)
    assert jnp.any(second.state.u != 0.0)


def test_wssr_warm_svd_core_jitted_matches_eager_update_and_history():
    state = wssr.WSSRWarmSVDCoreState(
        sr_o=jnp.array(
            [
                [0.2, -0.1, 0.0],
                [0.4, 0.3, 0.0],
                [-0.5, 0.25, 0.0],
            ],
            dtype=jnp.float32,
        ),
        ek=jnp.array([0.5, -0.25, 0.0], dtype=jnp.float32),
        sr_rank0=jnp.array(2),
        sr_rank=jnp.array(2),
        u=jnp.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=jnp.float32,
        ),
        has_u=jnp.array(True),
    )
    o_aug = jnp.array(
        [
            [2.0, 0.5, -1.0, 0.25],
            [0.0, 1.5, 0.5, -0.75],
            [1.0, -0.25, 1.25, 0.5],
        ],
        dtype=jnp.float32,
    )
    e_aug = jnp.array([0.5, -0.25, 0.125, -0.75], dtype=jnp.float32)
    key = jax.random.PRNGKey(123)

    kwargs = dict(
        damping=0.05,
        norm_constraint=10.0,
        sr_rank_max=3,
        sr_scale=1.5,
        svd_maxiter_initial=2,
        svd_maxiter_warm=1,
        constrain_update_norm=False,
    )
    eager = wssr.wssr_warm_svd_core_update(o_aug, e_aug, state, key=key, **kwargs)
    jitted_update = jax.jit(
        wssr.wssr_warm_svd_core_update,
        static_argnames=(
            "sr_rank_max",
            "svd_maxiter_initial",
            "svd_maxiter_warm",
            "constrain_update_norm",
        ),
    )
    jitted = jitted_update(o_aug, e_aug, state, key=key, **kwargs)
    jitted.grad_like_update.block_until_ready()

    chex.assert_trees_all_close(
        eager.grad_like_update, jitted.grad_like_update, rtol=1e-5, atol=1e-5
    )
    chex.assert_trees_all_close(eager.state.sr_o, jitted.state.sr_o)
    chex.assert_trees_all_close(eager.state.ek, jitted.state.ek)
    chex.assert_trees_all_close(eager.state.sr_rank0, jitted.state.sr_rank0)
    chex.assert_trees_all_close(eager.state.sr_rank, jitted.state.sr_rank)
    chex.assert_trees_all_close(eager.state.has_u, jitted.state.has_u)
    assert eager.state.u.shape == jitted.state.u.shape == state.u.shape
    assert jnp.all(jnp.isfinite(eager.state.u))
    assert jnp.all(jnp.isfinite(jitted.state.u))


def test_wssr_warm_svd_augmented_core_jits():
    state = wssr.initialize_wssr_warm_svd_core_state(
        num_params=4, sr_rank=2, sr_rank_max=4, dtype=jnp.float32
    )
    o_cur = jnp.array(
        [
            [4.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 2.0],
            [1.0, -1.0, 0.5],
        ],
        dtype=jnp.float32,
    )
    e_cur = jnp.array([0.5, -0.25, 0.125], dtype=jnp.float32)

    def augmented_core_update(o_cur, e_cur, state, key):
        o_aug, e_aug = wssr.augment_wssr_system(o_cur, e_cur, state, eta=0.25)
        return wssr.wssr_warm_svd_core_update(
            o_aug,
            e_aug,
            state,
            key=key,
            damping=1e-4,
            norm_constraint=10.0,
            sr_rank_max=4,
            sr_scale=2.0,
            svd_maxiter_initial=2,
            svd_maxiter_warm=1,
            constrain_update_norm=False,
        )

    jitted_update = jax.jit(augmented_core_update)
    first = jitted_update(o_cur, e_cur, state, jax.random.PRNGKey(11))
    first.grad_like_update.block_until_ready()
    second = jitted_update(o_cur, e_cur, first.state, jax.random.PRNGKey(12))
    second.grad_like_update.block_until_ready()

    _assert_wssr_state_all_finite(first.state)
    _assert_wssr_state_all_finite(second.state)
    assert first.grad_like_update.shape == (4,)
    assert second.grad_like_update.shape == (4,)
    assert first.state.has_u == jnp.array(True)
    assert second.state.has_u == jnp.array(True)
    assert first.state.sr_rank == 4


def test_initialize_optimizer_dispatches_wssr_warm_svd_and_constructs_state():
    params = _tiny_params()
    data = _tiny_positions()
    config = default_config.get_default_config()
    config.vmc.nchains = data.shape[0]
    config.vmc.optimizer_type = "wssr_warm_svd"

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
    assert isinstance(optimizer_state.core_state, wssr.WSSRWarmSVDCoreState)
    assert optimizer_state.core_state.u.shape == (
        3,
        config.vmc.optimizer.wssr_warm_svd.sr_rank_max,
    )
    assert optimizer_state.core_state.has_u == jnp.array(False)
    assert key.shape == (2,)


def test_wssr_warm_svd_integrated_two_updates_preserve_structure_and_have_no_nans():
    params = _tiny_params()
    data = _tiny_positions()
    config = default_config.get_default_config()
    config.vmc.nchains = data.shape[0]
    config.vmc.optimizer_type = "wssr_warm_svd"
    config.vmc.optimizer.wssr_warm_svd.schedule_type = "constant"
    config.vmc.optimizer.wssr_warm_svd.learning_rate = 0.125
    config.vmc.optimizer.wssr_warm_svd.constrain_norm = False
    config.vmc.optimizer.wssr_warm_svd.svd_maxiter_initial = 2
    config.vmc.optimizer.wssr_warm_svd.svd_maxiter_warm = 1

    update_param_fn, optimizer_state, key = initialize_optimizer(
        _log_psi_apply,
        _local_energy_fn,
        None,
        config.vmc,
        params,
        data,
        lambda x: x,
        lambda d, p: d,
        jax.random.PRNGKey(6),
        apply_pmap=False,
    )

    params_1, data_1, state_1, metrics_1, key_1 = update_param_fn(
        params, data, optimizer_state, key
    )
    params_2, data_2, state_2, metrics_2, key_2 = update_param_fn(
        params_1, data_1, state_1, key_1
    )

    assert jax.tree_util.tree_structure(params_2) == jax.tree_util.tree_structure(
        params
    )
    _assert_tree_all_finite(params_2)
    _assert_wssr_state_all_finite(state_1.core_state)
    _assert_wssr_state_all_finite(state_2.core_state)
    assert state_1.core_state.has_u == jnp.array(True)
    assert state_2.core_state.has_u == jnp.array(True)
    assert (
        0
        <= state_2.core_state.sr_rank0
        <= state_2.core_state.sr_rank
        <= config.vmc.optimizer.wssr_warm_svd.sr_rank_max
    )
    assert set(metrics_1).issuperset({"energy", "variance", "energy_noclip"})
    assert set(metrics_2).issuperset({"energy", "variance", "energy_noclip"})
    assert jnp.all(jnp.isfinite(jnp.asarray(list(metrics_1.values()))))
    assert jnp.all(jnp.isfinite(jnp.asarray(list(metrics_2.values()))))
    np.testing.assert_allclose(data_2, data)
    assert key_1.shape == key.shape
    assert key_2.shape == key.shape
    assert not np.allclose(key_1, key)
    assert not np.allclose(key_2, key_1)


def test_initialize_wssr_warm_svd_rejects_pmap_until_jit_safe_core_exists():
    params = _tiny_params()
    data = _tiny_positions()
    config = default_config.get_default_config()
    config.vmc.nchains = data.shape[0]
    config.vmc.optimizer_type = "wssr_warm_svd"

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
        raise AssertionError("Expected wssr_warm_svd to reject apply_pmap=True")
