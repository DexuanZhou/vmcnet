"""Tests for averaged-MinSR with SVD-compressed WSSR-style history."""

import jax
import jax.flatten_util
import jax.numpy as jnp
from ml_collections import ConfigDict
import optax
import pytest

from vmcnet.updates import wssr


def _small_augmented_system():
    o_aug = jnp.array(
        [
            [1.0, -0.5, 0.25, 0.75],
            [0.25, 1.25, -0.75, 0.5],
            [1.5, 0.0, 0.5, -1.0],
        ],
        dtype=jnp.float32,
    )
    e_aug = jnp.array([0.5, -0.25, 0.75, -0.125], dtype=jnp.float32)
    state = wssr.initialize_wssr_core_state(
        num_params=o_aug.shape[0],
        sr_rank=3,
        sr_rank_max=4,
        dtype=o_aug.dtype,
    )
    return o_aug, e_aug, state


def _manual_tikhonov_dual_direction(o_aug, e_aug, lambda_reg):
    gram = o_aug.T @ o_aug
    alpha = jnp.linalg.solve(
        gram + lambda_reg * jnp.eye(o_aug.shape[1], dtype=o_aug.dtype),
        e_aug,
    )
    return o_aug @ alpha


def _manual_tikhonov_svd_direction(o_aug, e_aug, lambda_reg):
    u, singular_values, vh = jnp.linalg.svd(o_aug, full_matrices=False)
    projected_rhs = vh @ e_aug
    filtered_rhs = singular_values / (jnp.square(singular_values) + lambda_reg)
    return u @ (filtered_rhs * projected_rhs)


def test_delta_zero_no_history_matches_current_batch_tikhonov_dual_direction():
    o_cur = jnp.array(
        [
            [0.5, -1.0, 0.25],
            [1.25, 0.75, -0.5],
            [-0.25, 0.5, 1.0],
            [0.75, -0.25, -1.25],
        ],
        dtype=jnp.float32,
    )
    e_cur = jnp.array([0.25, -0.5, 0.75], dtype=jnp.float32)
    state = wssr.initialize_wssr_core_state(
        num_params=o_cur.shape[0],
        sr_rank=3,
        sr_rank_max=5,
        dtype=o_cur.dtype,
    )
    o_aug, e_aug = wssr.augment_wssr_system(o_cur, e_cur, state, eta=0.99)

    result = wssr.avg_minsr_svd_history_core_update(
        o_aug,
        e_aug,
        state,
        lambda_reg=0.1,
        damping=1e-6,
        norm_constraint=1.0,
        sr_rank_max=5,
        constrain_update_norm=False,
    )

    expected = _manual_tikhonov_dual_direction(o_cur, e_cur, 0.1)
    assert jnp.allclose(result.grad_like_update, expected, rtol=1e-5, atol=1e-5)


def test_core_update_matches_manual_dual_solve():
    o_aug, e_aug, state = _small_augmented_system()

    result = wssr.avg_minsr_svd_history_core_update(
        o_aug,
        e_aug,
        state,
        lambda_reg=0.2,
        damping=1e-6,
        norm_constraint=1.0,
        sr_rank_max=4,
        constrain_update_norm=False,
    )

    expected = _manual_tikhonov_dual_direction(o_aug, e_aug, 0.2)
    assert jnp.allclose(result.grad_like_update, expected, rtol=1e-5, atol=1e-5)


def test_core_update_matches_full_svd_tikhonov_formula():
    o_aug, e_aug, state = _small_augmented_system()

    result = wssr.avg_minsr_svd_history_core_update(
        o_aug,
        e_aug,
        state,
        lambda_reg=0.05,
        damping=1e-6,
        norm_constraint=1.0,
        sr_rank_max=4,
        constrain_update_norm=False,
    )

    expected = _manual_tikhonov_svd_direction(o_aug, e_aug, 0.05)
    assert jnp.allclose(result.grad_like_update, expected, rtol=1e-5, atol=1e-5)


def test_history_update_uses_retained_svd_factors():
    o_aug, e_aug, state = _small_augmented_system()
    u, singular_values, vh = jnp.linalg.svd(o_aug, full_matrices=False)

    result = wssr.avg_minsr_svd_history_core_update(
        o_aug,
        e_aug,
        state,
        lambda_reg=0.05,
        damping=1e-6,
        norm_constraint=1.0,
        sr_rank_max=4,
        constrain_update_norm=False,
    )
    rank = singular_values.shape[0]

    assert jnp.allclose(
        result.state.sr_o[:, :rank],
        u[:, :rank] * singular_values[:rank],
        rtol=1e-5,
        atol=1e-5,
    )
    assert jnp.allclose(
        result.state.ek[:rank],
        vh[:rank, :] @ e_aug,
        rtol=1e-5,
        atol=1e-5,
    )
    assert result.state.sr_rank0 == rank


def test_full_rank_history_preserves_score_covariance_and_force():
    o_aug, e_aug, state = _small_augmented_system()

    result = wssr.avg_minsr_svd_history_core_update(
        o_aug,
        e_aug,
        state,
        lambda_reg=0.05,
        damping=1e-6,
        norm_constraint=1.0,
        sr_rank_max=4,
        constrain_update_norm=False,
    )
    active_rank = int(result.state.sr_rank0)
    h_new = result.state.sr_o[:, :active_rank]
    c_new = result.state.ek[:active_rank]

    assert jnp.allclose(h_new @ h_new.T, o_aug @ o_aug.T, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(h_new @ c_new, o_aug @ e_aug, rtol=1e-5, atol=1e-5)


def test_jitted_core_update_preserves_shapes_when_active_rank_changes():
    o_aug, e_aug, state = _small_augmented_system()
    jitted_update = jax.jit(
        wssr.avg_minsr_svd_history_core_update,
        static_argnames=("sr_rank_max", "constrain_update_norm"),
    )

    result_full = jitted_update(
        o_aug,
        e_aug,
        state,
        0.05,
        1e-6,
        1.0,
        4,
        constrain_update_norm=False,
    )
    result_low_rank = jitted_update(
        o_aug.at[:, 1:].set(0.0),
        e_aug,
        state,
        0.05,
        0.5,
        1.0,
        4,
        constrain_update_norm=False,
    )

    assert result_full.state.sr_o.shape == state.sr_o.shape
    assert result_full.state.ek.shape == state.ek.shape
    assert result_low_rank.state.sr_o.shape == state.sr_o.shape
    assert result_low_rank.state.ek.shape == state.ek.shape
    assert jnp.all(jnp.isfinite(result_full.grad_like_update))
    assert jnp.all(jnp.isfinite(result_low_rank.grad_like_update))


def test_zero_singular_values_do_not_produce_nan_or_inf():
    o_aug = jnp.zeros((3, 4), dtype=jnp.float32)
    e_aug = jnp.array([1.0, -0.5, 0.25, 0.0], dtype=jnp.float32)
    state = wssr.initialize_wssr_core_state(
        num_params=3,
        sr_rank=3,
        sr_rank_max=4,
        dtype=o_aug.dtype,
    )

    result = wssr.avg_minsr_svd_history_core_update(
        o_aug,
        e_aug,
        state,
        lambda_reg=0.05,
        damping=0.001,
        norm_constraint=1.0,
        sr_rank_max=4,
        constrain_update_norm=False,
    )

    assert jnp.all(jnp.isfinite(result.grad_like_update))
    assert jnp.all(jnp.isfinite(result.state.sr_o))
    assert jnp.all(jnp.isfinite(result.state.ek))
    assert jnp.isfinite(result.dual_matrix_cond)
    assert jnp.isfinite(result.update_norm)
    assert result.active_rank == 0


def _tiny_log_psi_apply(params, position):
    return (
        params["w"][0] * position[0]
        + params["w"][1] * position[1]
        + params["b"] * position[0] * position[1]
    )


def _tiny_energy_and_statistics_fn(params, positions):
    del params
    local_energies = jnp.sum(jnp.square(positions), axis=1)
    energy = jnp.mean(local_energies)
    variance = jnp.var(local_energies)
    return (
        energy,
        local_energies,
        {
            "variance": variance,
            "energy_noclip": energy,
            "variance_noclip": variance,
        },
    )


def test_integrated_update_returns_avg_minsr_metrics_and_fixed_state_shapes():
    params = {
        "w": jnp.array([1.0, -0.5], dtype=jnp.float32),
        "b": jnp.array(0.25, dtype=jnp.float32),
    }
    positions = jnp.array(
        [
            [1.0, -2.0],
            [0.5, 1.5],
            [-1.0, 0.25],
            [2.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    flat_params, _ = jax.flatten_util.ravel_pytree(params)
    config = ConfigDict(
        {
            "lambda_reg": 0.05,
            "damping": 1e-6,
            "constrain_norm": True,
            "norm_constraint": 0.1,
            "eta": 0.99,
            "sr_rank": 3,
            "sr_rank_max": 5,
            "sr_scale": 1.1,
        }
    )
    optimizer = optax.sgd(learning_rate=0.01, momentum=0, nesterov=False)
    update_fn = wssr.construct_avg_minsr_svd_history_update_param_fn(
        _tiny_log_psi_apply,
        _tiny_energy_and_statistics_fn,
        optimizer,
        lambda data: data,
        lambda data, params: data,
        config,
    )
    state = wssr.WSSROptimizerState(
        core_state=wssr.initialize_wssr_core_state(
            flat_params.shape[0],
            config.sr_rank,
            config.sr_rank_max,
            dtype=flat_params.dtype,
        ),
        optax_state=optimizer.init(params),
    )

    new_params, new_data, new_state, metrics, key = update_fn(
        params,
        positions,
        state,
        jax.random.PRNGKey(0),
    )

    assert set(metrics).issuperset(
        {
            "energy",
            "variance",
            "energy_noclip",
            "variance_noclip",
            "avg_minsr_active_rank",
            "avg_minsr_sr_rank",
            "avg_minsr_sr_rank0",
            "avg_minsr_lambda_reg",
            "avg_minsr_dual_matrix_cond",
            "avg_minsr_update_norm",
        }
    )
    assert new_state.core_state.sr_o.shape == (flat_params.shape[0], 5)
    assert new_state.core_state.ek.shape == (5,)
    assert new_data.shape == positions.shape
    assert key.shape == (2,)
    for leaf in jax.tree_util.tree_leaves(new_params):
        assert jnp.all(jnp.isfinite(leaf))


def test_initialize_avg_minsr_svd_history_rejects_pmap():
    params = {"w": jnp.array([1.0, -0.5], dtype=jnp.float32)}
    config = ConfigDict(
        {
            "lambda_reg": 0.05,
            "damping": 0.001,
            "constrain_norm": True,
            "norm_constraint": 0.1,
            "eta": 0.99,
            "sr_rank": 2,
            "sr_rank_max": 4,
            "sr_scale": 1.1,
        }
    )

    with pytest.raises(NotImplementedError, match="apply_pmap=False"):
        wssr.initialize_avg_minsr_svd_history(
            _tiny_log_psi_apply,
            _tiny_energy_and_statistics_fn,
            params,
            lambda data: data,
            lambda data, params: data,
            lambda t: 0.01,
            config,
            apply_pmap=True,
        )
