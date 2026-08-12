"""Tests for pure WSSR SVD numerical core."""

import jax
import jax.numpy as jnp
import numpy as np
import chex
import pytest

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


_WSSR_RANK_METRIC_KEYS = {
    "wssr_active_rank",
    "wssr_sr_rank",
    "wssr_sr_rank0",
    "wssr_storage_width",
    "wssr_svd_working_rank",
    "wssr_sr_rank_max",
}


def _assert_wssr_rank_metrics(
    metrics, sr_rank_max, svd_working_rank, storage_width=None
):
    if storage_width is None:
        storage_width = sr_rank_max
    assert set(metrics).issuperset(_WSSR_RANK_METRIC_KEYS)
    assert metrics["wssr_storage_width"] == jnp.asarray(storage_width)
    assert metrics["wssr_sr_rank_max"] == jnp.asarray(sr_rank_max)
    assert metrics["wssr_svd_working_rank"] == jnp.asarray(svd_working_rank)
    assert 0 <= metrics["wssr_active_rank"] <= metrics["wssr_svd_working_rank"]
    assert 0 <= metrics["wssr_sr_rank0"] <= metrics["wssr_sr_rank"]


def _spectral_regularization_fixture():
    state = wssr.initialize_wssr_core_state(
        num_params=3, sr_rank=3, sr_rank_max=3, dtype=jnp.float32
    )
    o_aug = jnp.array(
        [
            [1.0, -0.5, 0.25, 1.25],
            [0.25, 1.5, -1.0, -0.75],
            [1.5, 0.75, 0.5, -0.25],
        ],
        dtype=jnp.float32,
    )
    e_aug = jnp.array([0.5, -0.25, 0.75, -0.125], dtype=jnp.float32)
    u = jnp.eye(3, dtype=jnp.float32)
    singular_values = jnp.array([4.0, 2.0, 0.75], dtype=jnp.float32)
    vh = jnp.array(
        [
            [0.5, -0.5, 0.25, 0.25],
            [0.25, 0.75, -0.5, 0.125],
            [-0.5, 0.25, 0.5, -0.25],
        ],
        dtype=jnp.float32,
    )
    return state, o_aug, e_aug, u, singular_values, vh


def _expected_wssr_spectral_update(
    o_aug,
    e_aug,
    u,
    singular_values,
    damping,
    spectral_regularization,
    complement_weight,
):
    force = o_aug @ e_aug
    leading_sv = singular_values[0]
    retained = singular_values / leading_sv > damping
    retained_float = retained.astype(o_aug.dtype)
    sigma_floor = jnp.square(damping * jnp.abs(leading_sv))
    inv_floor = 1.0 / sigma_floor
    if spectral_regularization == "hard_floor":
        inv_cap = jnp.square(1.0 / singular_values)
    elif spectral_regularization == "tikhonov":
        inv_cap = 1.0 / (jnp.square(singular_values) + sigma_floor)
    else:
        raise ValueError("bad test spectral_regularization")
    inv_perp = complement_weight * inv_floor
    projected_force = u.T @ force
    projected_force = projected_force * (inv_cap - inv_perp) * retained_float
    return u @ projected_force + inv_perp * force


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


@pytest.mark.parametrize("eta_g", [0.5, 0.8, 0.95])
def test_transported_gradient_removes_fixed_quadratic_ema_bias(eta_g):
    hessian = jnp.array(
        [[4.0, 0.5, 0.0], [0.5, 2.0, 0.25], [0.0, 0.25, 1.0]],
        dtype=jnp.float32,
    )
    theta_previous = jnp.array([0.8, -0.3, 0.5], dtype=jnp.float32)
    theta_current = jnp.array([0.35, 0.1, -0.2], dtype=jnp.float32)
    gradient_previous = hessian @ theta_previous
    gradient_current = hessian @ theta_current
    operator_delta = hessian @ (theta_current - theta_previous)

    ordinary_ema = (
        eta_g * gradient_previous + (1.0 - eta_g) * gradient_current
    )
    transported = wssr.transport_gradient_memory(
        gradient_previous,
        gradient_current,
        operator_delta,
        eta_g,
        jnp.asarray(True),
    )
    gradient_error = jnp.linalg.norm(ordinary_ema - gradient_current)
    transported_gradient_error = jnp.linalg.norm(transported - gradient_current)
    print(
        f"eta_g={eta_g}: gradient error={float(gradient_error):.8e}, "
        "transported gradient error="
        f"{float(transported_gradient_error):.8e}"
    )

    assert gradient_error > 1e-2
    np.testing.assert_allclose(
        transported, gradient_current, rtol=1e-6, atol=1e-6
    )


def _split_eta_test_system():
    state = wssr.initialize_wssr_core_state(
        num_params=3, sr_rank=2, sr_rank_max=2, dtype=jnp.float32
    )._replace(
        sr_o=jnp.array(
            [[1.0, 0.25], [0.5, -1.0], [-0.5, 0.75]], dtype=jnp.float32
        ),
        ek=jnp.array([0.4, -0.2], dtype=jnp.float32),
        sr_rank0=jnp.asarray(2),
    )
    o_current = jnp.array(
        [[0.5, -0.25], [1.0, 0.5], [-0.75, 0.25]], dtype=jnp.float32
    )
    e_current = jnp.array([0.3, -0.6], dtype=jnp.float32)
    previous_gradient = state.sr_o @ state.ek
    current_gradient = o_current @ e_current
    return state, o_current, e_current, previous_gradient, current_gradient


def test_split_eta_S_history_with_current_batch_gradient():
    """eta_S=.95, eta_g=0 keeps S history but uses the current gradient."""
    state, o_current, e_current, previous_gradient, current_gradient = (
        _split_eta_test_system()
    )
    o_aug, _ = wssr.augment_wssr_system(
        o_current, e_current, state, eta=0.95
    )
    gradient_history = wssr.transport_gradient_memory(
        previous_gradient,
        current_gradient,
        jnp.zeros_like(current_gradient),
        eta_g=0.0,
        transport_initialized=jnp.asarray(True),
    )

    np.testing.assert_allclose(o_aug[:, :2], jnp.sqrt(0.95) * state.sr_o)
    np.testing.assert_allclose(
        o_aug[:, 2:], jnp.sqrt(0.05) * o_current, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(gradient_history, current_gradient)


def test_split_eta_g_history_without_S_averaging():
    """eta_S=0, eta_g=.95 removes S history but retains gradient history."""
    state, o_current, e_current, previous_gradient, current_gradient = (
        _split_eta_test_system()
    )
    o_aug, _ = wssr.augment_wssr_system(
        o_current, e_current, state, eta=0.0
    )
    gradient_history = wssr.transport_gradient_memory(
        previous_gradient,
        current_gradient,
        jnp.zeros_like(current_gradient),
        eta_g=0.95,
        transport_initialized=jnp.asarray(True),
    )

    np.testing.assert_allclose(o_aug[:, :2], jnp.zeros_like(state.sr_o))
    np.testing.assert_allclose(o_aug[:, 2:], o_current)
    np.testing.assert_allclose(
        gradient_history,
        0.95 * previous_gradient + 0.05 * current_gradient,
        rtol=1e-6,
        atol=1e-6,
    )


def test_equal_split_etas_match_legacy_augmented_force():
    """eta_S=eta_g follows the exact legacy shared-eta RHS."""
    state, o_current, e_current, previous_gradient, current_gradient = (
        _split_eta_test_system()
    )
    o_aug, e_aug = wssr.augment_wssr_system(
        o_current, e_current, state, eta=0.95
    )
    split_gradient_history = wssr.transport_gradient_memory(
        previous_gradient,
        current_gradient,
        jnp.zeros_like(current_gradient),
        eta_g=0.95,
        transport_initialized=jnp.asarray(True),
    )

    np.testing.assert_allclose(
        split_gradient_history, o_aug @ e_aug, rtol=1e-6, atol=1e-6
    )


def test_split_eta_resolution_prefers_new_keys_and_falls_back_to_eta():
    config = default_config.get_default_config().vmc.optimizer.wssr_warm_svd_right
    config.eta = 0.8
    assert wssr.resolve_wssr_averaging_weights(config) == (0.8, 0.8)

    config.eta_S = 0.95
    config.eta_g = 0.0
    assert wssr.resolve_wssr_averaging_weights(config) == (0.95, 0.0)


def test_adaptive_eta_resolution_defaults_to_current_gradient():
    config = default_config.get_default_config().vmc.optimizer.wssr_warm_svd_right
    config.eta = 0.8
    config.eta_S = -1.0
    config.eta_g = -1.0
    config.adaptive_S_average = True
    config.eta_S_max = 0.95

    assert wssr.resolve_wssr_averaging_weights(config) == (0.95, 0.0)


def test_adaptive_gradient_eta_resolution_uses_independent_maximum():
    config = default_config.get_default_config().vmc.optimizer.wssr_warm_svd_right
    config.eta = 0.8
    config.eta_S = -1.0
    config.eta_g = -1.0
    config.adaptive_S_average = True
    config.eta_S_max = 0.95
    config.adaptive_g_average = True
    config.eta_g_max = 0.2

    assert wssr.resolve_wssr_averaging_weights(config) == (0.95, 0.2)


def test_adaptive_s_averaging_eta_schedules():
    steps = jnp.array([0, 2, 5, 10], dtype=jnp.int32)
    constant = jax.vmap(
        lambda step: wssr.adaptive_s_averaging_eta(
            "constant", 0.95, step, 10, 5.0
        )
    )(steps)
    linear = jax.vmap(
        lambda step: wssr.adaptive_s_averaging_eta(
            "linear_warmup", 0.95, step, 10, 5.0
        )
    )(steps)
    exponential = jax.vmap(
        lambda step: wssr.adaptive_s_averaging_eta(
            "exponential_growth", 0.95, step, 10, 5.0
        )
    )(steps)

    np.testing.assert_allclose(constant, 0.95 * np.ones(4), rtol=1e-6)
    np.testing.assert_allclose(
        linear, 0.95 * np.array([0.0, 0.2, 0.5, 1.0]), rtol=1e-6
    )
    np.testing.assert_allclose(
        exponential,
        0.95 * (1.0 - np.exp(-np.array([0.0, 2.0, 5.0, 10.0]) / 5.0)),
        rtol=1e-6,
    )


def test_adaptive_s_average_integrated_metrics_use_actual_parameter_delta():
    params = _tiny_params()
    initial_params = jax.tree_util.tree_map(lambda value: value.copy(), params)
    data = _tiny_positions()
    config = default_config.get_default_config()
    config.vmc.nchains = data.shape[0]
    config.vmc.optimizer_type = "wssr_warm_svd_right"
    opt = config.vmc.optimizer.wssr_warm_svd_right
    opt.schedule_type = "constant"
    opt.learning_rate = 0.01
    opt.constrain_norm = False
    opt.adaptive_S_average = True
    opt.eta_S_schedule = "linear_warmup"
    opt.eta_S_max = 0.8
    opt.eta_S_warmup_steps = 4
    opt.eta_g = -1.0
    opt.enable_gradient_transport = False
    opt.sr_rank = 2
    opt.sr_rank_max = 3
    opt.sr_storage_rank = 3
    opt.svd_working_rank = 3
    opt.svd_maxiter_initial = 2
    opt.svd_maxiter_warm = 1
    opt.spectral_regularization = "tikhonov"
    opt.complement_weight = 0.0
    opt.relative_singular_value_cutoff = 0.0
    opt.tikhonov_lambda = 0.1

    update_param_fn, state, key = initialize_optimizer(
        _log_psi_apply,
        _local_energy_fn,
        None,
        config.vmc,
        params,
        data,
        lambda x: x,
        lambda d, p: d,
        jax.random.PRNGKey(92),
        apply_pmap=False,
    )
    assert isinstance(state, wssr.WSSRTransportedGradientOptimizerState)

    params, data, state, metrics_1, key = update_param_fn(
        params, data, state, key
    )
    params_after_first = params
    params, _, state, metrics_2, _ = update_param_fn(
        params, data, state, key
    )
    actual_first_update = jax.tree_util.tree_map(
        lambda current, initial: current - initial,
        params_after_first,
        initial_params,
    )

    assert metrics_1["adaptive_S_step"] == 0
    assert metrics_1["eta_S_current"] == pytest.approx(0.0)
    assert metrics_1["eta_g"] == pytest.approx(0.0)
    assert metrics_2["adaptive_S_step"] == 1
    assert metrics_2["eta_S_current"] == pytest.approx(0.2)
    assert metrics_2["delta_theta_norm"] == pytest.approx(
        float(wssr.tree_l2_norm(actual_first_update)), rel=1e-5
    )
    assert jnp.isfinite(metrics_2["S_current_minus_ema_action_norm"])
    assert jnp.isfinite(metrics_2["S_transport_norm"])


def test_adaptive_gradient_average_integrated_schedule():
    params = _tiny_params()
    data = _tiny_positions()
    config = default_config.get_default_config()
    config.vmc.nchains = data.shape[0]
    config.vmc.optimizer_type = "wssr_warm_svd_right"
    opt = config.vmc.optimizer.wssr_warm_svd_right
    opt.schedule_type = "constant"
    opt.learning_rate = 0.01
    opt.constrain_norm = False
    opt.eta_S = 0.0
    opt.adaptive_g_average = True
    opt.eta_g_schedule = "exponential_growth"
    opt.eta_g_max = 0.2
    opt.eta_g_tau = 4.0
    opt.enable_gradient_transport = True
    opt.sr_rank = 2
    opt.sr_rank_max = 3
    opt.sr_storage_rank = 3
    opt.svd_working_rank = 3
    opt.svd_maxiter_initial = 2
    opt.svd_maxiter_warm = 1
    opt.spectral_regularization = "tikhonov"
    opt.complement_weight = 0.0
    opt.relative_singular_value_cutoff = 0.0
    opt.tikhonov_lambda = 0.1

    update_param_fn, state, key = initialize_optimizer(
        _log_psi_apply,
        _local_energy_fn,
        None,
        config.vmc,
        params,
        data,
        lambda x: x,
        lambda d, p: d,
        jax.random.PRNGKey(192),
        apply_pmap=False,
    )
    assert isinstance(state, wssr.WSSRTransportedGradientOptimizerState)
    params, data, state, metrics_1, key = update_param_fn(
        params, data, state, key
    )
    params, _, _, metrics_2, _ = update_param_fn(params, data, state, key)

    assert metrics_1["eta_g_current"] == pytest.approx(0.0)
    assert metrics_2["eta_g_current"] == pytest.approx(
        0.2 * (1.0 - np.exp(-0.25)), rel=1e-6
    )


@pytest.mark.parametrize("eta_S,eta_g", [(0.95, 0.0), (0.0, 0.95)])
def test_split_eta_optimizer_interface_smoke(eta_S, eta_g):
    """Independent weights run through the integrated optimizer state path."""
    params = _tiny_params()
    data = _tiny_positions()
    config = default_config.get_default_config()
    config.vmc.nchains = data.shape[0]
    config.vmc.optimizer_type = "wssr_warm_svd_right"
    opt = config.vmc.optimizer.wssr_warm_svd_right
    opt.schedule_type = "constant"
    opt.learning_rate = 0.01
    opt.constrain_norm = False
    opt.eta_S = eta_S
    opt.eta_g = eta_g
    opt.enable_gradient_transport = False
    opt.sr_rank = 2
    opt.sr_rank_max = 3
    opt.sr_storage_rank = 3
    opt.svd_working_rank = 3
    opt.svd_maxiter_initial = 2
    opt.svd_maxiter_warm = 1
    opt.spectral_regularization = "tikhonov"
    opt.complement_weight = 0.0
    opt.relative_singular_value_cutoff = 0.0
    opt.tikhonov_lambda = 0.1

    update_param_fn, state, key = initialize_optimizer(
        _log_psi_apply,
        _local_energy_fn,
        None,
        config.vmc,
        params,
        data,
        lambda x: x,
        lambda d, p: d,
        jax.random.PRNGKey(91),
        apply_pmap=False,
    )
    assert isinstance(state, wssr.WSSRTransportedGradientOptimizerState)

    params, data, state, metrics, key = update_param_fn(
        params, data, state, key
    )
    params, _, state, metrics, _ = update_param_fn(
        params, data, state, key
    )
    _assert_tree_all_finite(params)
    assert metrics["eta_S"] == pytest.approx(eta_S)
    assert metrics["eta_g"] == pytest.approx(eta_g)
    assert jnp.isfinite(metrics["gradient_norm"])
    assert jnp.isfinite(metrics["gradient_history_norm"])
    assert "S_transport_norm" not in metrics


def test_equal_explicit_split_etas_resolve_to_legacy_pair():
    def make_config(explicit_split):
        config = default_config.get_default_config()
        config.vmc.nchains = _tiny_positions().shape[0]
        config.vmc.optimizer_type = "wssr_warm_svd_right"
        opt = config.vmc.optimizer.wssr_warm_svd_right
        opt.schedule_type = "constant"
        opt.learning_rate = 0.01
        opt.constrain_norm = False
        opt.eta = 0.95
        if explicit_split:
            opt.eta_S = 0.95
            opt.eta_g = 0.95
        opt.enable_gradient_transport = False
        opt.sr_rank = 2
        opt.sr_rank_max = 3
        opt.sr_storage_rank = 3
        opt.svd_working_rank = 3
        opt.svd_maxiter_initial = 2
        opt.svd_maxiter_warm = 1
        opt.spectral_regularization = "tikhonov"
        opt.complement_weight = 0.0
        opt.relative_singular_value_cutoff = 0.0
        opt.tikhonov_lambda = 0.1
        return config

    params = _tiny_params()
    data = _tiny_positions()
    initialized = []
    for explicit_split in (False, True):
        config = make_config(explicit_split)
        update_fn, state, key = initialize_optimizer(
            _log_psi_apply,
            _local_energy_fn,
            None,
            config.vmc,
            params,
            data,
            lambda x: x,
            lambda d, p: d,
            jax.random.PRNGKey(101),
            apply_pmap=False,
        )
        assert type(state) is wssr.WSSROptimizerState
        initialized.append((update_fn, state, key))

    outputs = [
        update_fn(params, data, state, key)
        for update_fn, state, key in initialized
    ]
    assert_pytree_allclose(outputs[0][0], outputs[1][0])
    assert_pytree_allclose(outputs[0][2], outputs[1][2])
    assert outputs[0][3]["eta_S"] == pytest.approx(0.95)
    assert outputs[0][3]["eta_g"] == pytest.approx(0.95)
    assert outputs[1][3]["eta_S"] == pytest.approx(0.95)
    assert outputs[1][3]["eta_g"] == pytest.approx(0.95)


def test_apply_wssr_history_operator_matches_low_rank_product():
    state = wssr.initialize_wssr_core_state(
        num_params=4, sr_rank=2, sr_rank_max=3, dtype=jnp.float32
    )._replace(
        sr_o=jnp.array(
            [
                [1.0, 0.5, 9.0],
                [0.0, -1.0, 9.0],
                [2.0, 0.25, 9.0],
                [-0.5, 0.75, 9.0],
            ],
            dtype=jnp.float32,
        ),
        sr_rank0=jnp.asarray(2),
    )
    vector = jnp.array([0.25, -0.5, 1.25, 0.75], dtype=jnp.float32)
    active_factor = state.sr_o[:, :2]
    expected = active_factor @ (active_factor.T @ vector)

    np.testing.assert_allclose(
        wssr.apply_wssr_history_operator(state, vector),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("mixed_precision_solve", [False, True])
def test_wssr_force_override_replaces_legacy_gradient_history(
    mixed_precision_solve,
):
    state, o_aug, e_aug, u, singular_values, vh = (
        _spectral_regularization_fixture()
    )
    force_override = jnp.array([0.75, -0.5, 1.25], dtype=jnp.float32)
    result = wssr._wssr_update_from_svd(
        o_aug=o_aug,
        e_aug=e_aug,
        state=state,
        u=u,
        singular_values=singular_values,
        vh=vh,
        damping=0.1,
        norm_constraint=10.0,
        sr_rank_max=3,
        sr_scale=1.1,
        constrain_update_norm=False,
        spectral_regularization="tikhonov",
        complement_weight=0.0,
        relative_singular_value_cutoff=0.0,
        tikhonov_lambda=0.5,
        mixed_precision_solve=mixed_precision_solve,
        mixed_precision_history_width=2,
        force_override=force_override,
    )
    expected = u @ (
        (u.T @ force_override) / (jnp.square(singular_values) + 0.5)
    )

    np.testing.assert_allclose(
        result.grad_like_update, expected, rtol=2e-6, atol=2e-6
    )
    np.testing.assert_array_equal(result.state.ek, jnp.zeros_like(result.state.ek))


def test_initialize_wssr_core_state_shapes_and_ranks():
    state = wssr.initialize_wssr_core_state(
        num_params=5, sr_rank=3, sr_rank_max=7, dtype=jnp.float32
    )

    assert state.sr_o.shape == (5, 7)
    assert state.ek.shape == (7,)
    assert state.sr_rank0 == 0
    assert 0 <= state.sr_rank0 <= state.sr_rank <= 7


def test_resolve_wssr_storage_rank_preserves_default_and_validates_width():
    assert wssr.resolve_wssr_storage_rank(10, 100, -1) == 100
    assert wssr.resolve_wssr_storage_rank(10, 100, None) == 100
    assert wssr.resolve_wssr_storage_rank(10, 100, 80) == 80
    assert wssr.resolve_wssr_storage_rank(10, 100, 120) == 100

    with pytest.raises(ValueError, match="sr_storage_rank"):
        wssr.resolve_wssr_storage_rank(10, 100, 5)


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
    assert config.vmc.optimizer.wssr_warm_svd.svd_working_rank == -1
    assert config.vmc.optimizer.wssr_warm_svd.store_warm_u is True


def test_default_config_contains_wssr_warm_svd_right():
    config = default_config.get_default_config()

    assert config.vmc.optimizer.wssr_warm_svd_right.learning_rate == 5e-2
    assert config.vmc.optimizer.wssr_warm_svd_right.sr_rank == 10
    assert config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max == 100
    assert config.vmc.optimizer.wssr_warm_svd_right.sr_storage_rank == -1
    assert config.vmc.optimizer.wssr_warm_svd_right.eta_S == -1.0
    assert config.vmc.optimizer.wssr_warm_svd_right.eta_g == -1.0
    assert (
        config.vmc.optimizer.wssr_warm_svd_right.reduced_metric_history_mode
        == "none"
    )
    assert (
        config.vmc.optimizer.wssr_warm_svd_right.spectral_history_cluster_gap
        == 0.01
    )
    assert (
        config.vmc.optimizer.wssr_warm_svd_right.spectral_history_noise_scale
        == 1.0
    )
    assert (
        config.vmc.optimizer.wssr_warm_svd_right.spectral_history_drift_scale
        == 1.0
    )
    assert (
        config.vmc.optimizer.wssr_warm_svd_right.anisotropic_matrix_history
        is False
    )
    assert (
        config.vmc.optimizer.wssr_warm_svd_right
        .anisotropic_matrix_history_noise_scale
        == 1.0
    )
    assert config.vmc.optimizer.wssr_warm_svd_right.adaptive_S_average is False
    assert config.vmc.optimizer.wssr_warm_svd_right.eta_S_schedule == "constant"
    assert config.vmc.optimizer.wssr_warm_svd_right.eta_S_max == 0.95
    assert config.vmc.optimizer.wssr_warm_svd_right.eta_S_warmup_steps == 1000
    assert config.vmc.optimizer.wssr_warm_svd_right.eta_S_tau == 1000.0
    assert config.vmc.optimizer.wssr_warm_svd_right.adaptive_g_average is False
    assert config.vmc.optimizer.wssr_warm_svd_right.eta_g_schedule == "constant"
    assert config.vmc.optimizer.wssr_warm_svd_right.eta_g_max == 0.2
    assert config.vmc.optimizer.wssr_warm_svd_right.eta_g_warmup_steps == 1000
    assert config.vmc.optimizer.wssr_warm_svd_right.eta_g_tau == 1000.0
    assert config.vmc.optimizer.wssr_warm_svd_right.eta_bias_correction is False
    assert (
        config.vmc.optimizer.wssr_warm_svd_right.enable_gradient_transport
        is False
    )
    assert config.vmc.optimizer.wssr_warm_svd_right.mixed_precision_solve is False
    assert (
        config.vmc.optimizer.wssr_warm_svd_right.solution_recurrence_mode
        == "none"
    )
    assert (
        config.vmc.optimizer.wssr_warm_svd_right.solution_recurrence_mu
        == 0.99
    )
    assert (
        config.vmc.optimizer.wssr_warm_svd_right.residual_evaluation
        == "rank_coordinate"
    )
    assert (
        config.vmc.optimizer.wssr_warm_svd_right.galerkin_solve_backend
        == "host_fp64"
    )
    assert (
        config.vmc.optimizer.wssr_warm_svd_right.residual_dual_mode_diagnostics
        is False
    )
    assert (
        config.vmc.optimizer.wssr_warm_svd_right.solution_error_feedback
        is False
    )
    assert config.vmc.optimizer.wssr_warm_svd_right.error_feedback_decay == 1.0
    assert (
        config.vmc.optimizer.wssr_warm_svd_right.error_feedback_cap_reference
        == "correction"
    )
    assert config.vmc.optimizer.wssr_warm_svd_right.subspace_eta_S == 0.0
    assert (
        config.vmc.optimizer.wssr_warm_svd_right.subspace_refresh_mode
        == "ritz"
    )
    assert config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_initial == 8
    assert config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm == 2
    assert config.vmc.optimizer.wssr_warm_svd_right.exact_first is False
    assert config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank == -1
    assert config.vmc.optimizer.wssr_warm_svd_right.spectral_regularization == (
        "hard_floor"
    )
    assert config.vmc.optimizer.wssr_warm_svd_right.complement_weight == 1.0
    assert (
        config.vmc.optimizer.wssr_warm_svd_right.relative_singular_value_cutoff
        == -1.0
    )
    assert config.vmc.optimizer.wssr_warm_svd_right.tikhonov_lambda == -1.0
    assert config.vmc.optimizer.wssr_warm_svd_right.store_warm_u is True
    assert (
        config.vmc.optimizer.wssr_warm_svd_right.semi_matrix_free_augmented
        is False
    )
    assert config.vmc.optimizer.wssr_warm_svd_right.reliability_diagnostics is False
    assert config.vmc.optimizer.wssr_warm_svd_right.experimental_mode == "none"
    assert config.vmc.optimizer.wssr_warm_svd_right.experimental_target_rank == -1
    assert config.vmc.optimizer.wssr_warm_svd_right.cluster_gap_threshold == 0.002
    assert config.vmc.optimizer.wssr_warm_svd_right.near_tail_modes == 0
    assert config.vmc.optimizer.wssr_warm_svd_right.adaptive_complement_beta == 0.0
    assert (
        config.vmc.optimizer.wssr_warm_svd_right.adaptive_complement_beta_function
        == 0.0
    )
    assert (
        config.vmc.optimizer.wssr_warm_svd_right.euclidean_safety_constraint
        == -1.0
    )
    assert config.vmc.optimizer.wssr_warm_svd_right.smooth_transition_start == -1
    assert config.vmc.optimizer.wssr_warm_svd_right.smooth_transition_end == -1
    assert config.vmc.optimizer.wssr_warm_svd_right.force_aware_krylov_vectors == 0
    assert config.vmc.optimizer.wssr_warm_svd_right.iterative_complement_iterations == 0


def test_default_config_contains_wssr_warm_svd_right_matfree():
    config = default_config.get_default_config()

    assert config.vmc.optimizer.wssr_warm_svd_right_matfree.learning_rate == 5e-2
    assert config.vmc.optimizer.wssr_warm_svd_right_matfree.sr_rank == 10
    assert config.vmc.optimizer.wssr_warm_svd_right_matfree.sr_rank_max == 100
    assert config.vmc.optimizer.wssr_warm_svd_right_matfree.sr_storage_rank == -1
    assert config.vmc.optimizer.wssr_warm_svd_right_matfree.svd_maxiter_initial == 8
    assert config.vmc.optimizer.wssr_warm_svd_right_matfree.svd_maxiter_warm == 2
    assert config.vmc.optimizer.wssr_warm_svd_right_matfree.svd_working_rank == -1
    assert config.vmc.optimizer.wssr_warm_svd_right_matfree.spectral_regularization == (
        "hard_floor"
    )
    assert config.vmc.optimizer.wssr_warm_svd_right_matfree.complement_weight == 1.0
    assert (
        config.vmc.optimizer.wssr_warm_svd_right_matfree.relative_singular_value_cutoff
        == -1.0
    )
    assert config.vmc.optimizer.wssr_warm_svd_right_matfree.tikhonov_lambda == -1.0
    assert config.vmc.optimizer.wssr_warm_svd_right_matfree.store_warm_u is True


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


def _assert_matrix_free_score_products_match_explicit(
    log_psi_apply, params, positions, sample_weights
):
    o_cur, _ = wssr.center_and_scale_score_matrix(log_psi_apply, params, positions)
    flat_params, _ = jax.flatten_util.ravel_pytree(params)
    param_vector = jnp.linspace(-0.7, 0.9, flat_params.shape[0])
    sample_weight_matrix = jnp.stack(
        [
            sample_weights,
            jnp.linspace(0.8, -1.2, positions.shape[0]),
            jnp.linspace(-0.4, 1.6, positions.shape[0]),
        ],
        axis=1,
    )
    param_matrix = jnp.stack(
        [
            param_vector,
            jnp.linspace(1.1, -0.3, flat_params.shape[0]),
            jnp.linspace(-1.4, 0.2, flat_params.shape[0]),
        ],
        axis=1,
    )

    matvec = wssr.score_matvec_current(
        log_psi_apply, params, positions, sample_weights
    )
    rmatvec = wssr.score_rmatvec_current(
        log_psi_apply, params, positions, param_vector
    )
    matmat = wssr.score_matmat_current(
        log_psi_apply, params, positions, sample_weight_matrix
    )
    rmatmat = wssr.score_rmatmat_current(
        log_psi_apply, params, positions, param_matrix
    )

    np.testing.assert_allclose(matvec, o_cur @ sample_weights, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(rmatvec, o_cur.T @ param_vector, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        matmat, o_cur @ sample_weight_matrix, rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        rmatmat, o_cur.T @ param_matrix, rtol=1e-4, atol=1e-4
    )


def test_matrix_free_score_products_match_explicit_when_samples_exceed_params():
    params = _tiny_params()
    positions = _tiny_positions()
    sample_weights = jnp.array([1.5, -0.25, 2.0, 0.75])

    assert positions.shape[0] > jax.flatten_util.ravel_pytree(params)[0].shape[0]
    assert not np.allclose(jnp.mean(sample_weights), 0.0)
    _assert_matrix_free_score_products_match_explicit(
        _log_psi_apply, params, positions, sample_weights
    )


def test_matrix_free_score_products_match_explicit_when_params_exceed_samples():
    params = {
        "w": jnp.array([[0.2, -0.1, 0.3], [0.7, -0.4, 0.5]]),
        "b": jnp.array([0.1, -0.2]),
        "scale": jnp.array(0.4),
    }
    positions = jnp.array(
        [
            [0.5, -1.0, 0.25],
            [1.5, 0.2, -0.75],
            [-0.25, 0.4, 1.0],
        ]
    )
    sample_weights = jnp.array([2.0, -1.0, 0.5])

    def log_psi_apply(params, position):
        hidden = jnp.tanh(params["w"] @ position + params["b"])
        return jnp.sum(hidden) + params["scale"] * jnp.prod(position)

    assert jax.flatten_util.ravel_pytree(params)[0].shape[0] > positions.shape[0]
    assert not np.allclose(jnp.mean(sample_weights), 0.0)
    _assert_matrix_free_score_products_match_explicit(
        log_psi_apply, params, positions, sample_weights
    )


def _assert_augmented_matrix_free_products_match_explicit(
    log_psi_apply, params, positions, state, eta
):
    o_cur, _ = wssr.center_and_scale_score_matrix(log_psi_apply, params, positions)
    e_cur = jnp.linspace(-0.2, 0.4, positions.shape[0])
    o_aug, _ = wssr.augment_wssr_system(o_cur, e_cur, state, eta)
    flat_params, _ = jax.flatten_util.ravel_pytree(params)
    aug_vector = jnp.linspace(-0.8, 1.1, state.sr_o.shape[1] + positions.shape[0])
    param_vector = jnp.linspace(0.7, -0.9, flat_params.shape[0])
    aug_matrix = jnp.stack(
        [
            aug_vector,
            jnp.linspace(1.2, -0.6, aug_vector.shape[0]),
            jnp.linspace(-1.5, 0.25, aug_vector.shape[0]),
        ],
        axis=1,
    )
    param_matrix = jnp.stack(
        [
            param_vector,
            jnp.linspace(-0.4, 1.3, flat_params.shape[0]),
            jnp.linspace(1.6, -0.2, flat_params.shape[0]),
        ],
        axis=1,
    )

    matvec = wssr.wssr_augmented_matvec(
        log_psi_apply, params, positions, state, eta, aug_vector
    )
    rmatvec = wssr.wssr_augmented_rmatvec(
        log_psi_apply, params, positions, state, eta, param_vector
    )
    matmat = wssr.wssr_augmented_matmat(
        log_psi_apply, params, positions, state, eta, aug_matrix
    )
    rmatmat = wssr.wssr_augmented_rmatmat(
        log_psi_apply, params, positions, state, eta, param_matrix
    )

    np.testing.assert_allclose(matvec, o_aug @ aug_vector, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(rmatvec, o_aug.T @ param_vector, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(matmat, o_aug @ aug_matrix, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        rmatmat, o_aug.T @ param_matrix, rtol=1e-4, atol=1e-4
    )


def test_augmented_matrix_free_products_match_explicit_without_active_history():
    params = _tiny_params()
    positions = _tiny_positions()
    state = wssr.WSSRCoreState(
        sr_o=jnp.array(
            [
                [1.0, -2.0, 3.0, -4.0, 5.0],
                [0.5, 1.5, -0.5, 2.5, -1.0],
                [-1.0, 0.25, 0.75, -0.75, 1.25],
            ]
        ),
        ek=jnp.zeros((5,)),
        sr_rank0=jnp.array(0),
        sr_rank=jnp.array(3),
    )

    assert positions.shape[0] > jax.flatten_util.ravel_pytree(params)[0].shape[0]
    _assert_augmented_matrix_free_products_match_explicit(
        _log_psi_apply, params, positions, state, eta=0.99
    )


def test_augmented_matrix_free_products_match_explicit_with_active_history():
    params = {
        "w": jnp.array([[0.2, -0.1, 0.3], [0.7, -0.4, 0.5]]),
        "b": jnp.array([0.1, -0.2]),
        "scale": jnp.array(0.4),
    }
    positions = jnp.array(
        [
            [0.5, -1.0, 0.25],
            [1.5, 0.2, -0.75],
            [-0.25, 0.4, 1.0],
        ]
    )
    state = wssr.WSSRCoreState(
        sr_o=jnp.array(
            [
                [0.5, -1.0, 10.0, 20.0, -30.0],
                [1.5, 0.25, -10.0, 40.0, 50.0],
                [-0.5, 0.75, 60.0, -70.0, 80.0],
                [0.2, -0.4, -90.0, 100.0, 110.0],
                [1.2, 0.8, 120.0, 130.0, -140.0],
                [-1.1, 0.3, 150.0, -160.0, 170.0],
                [0.9, -0.7, -180.0, 190.0, 200.0],
                [0.4, 1.4, 210.0, -220.0, 230.0],
                [-0.8, 0.6, -240.0, 250.0, -260.0],
            ]
        ),
        ek=jnp.zeros((5,)),
        sr_rank0=jnp.array(2),
        sr_rank=jnp.array(4),
    )

    def log_psi_apply(params, position):
        hidden = jnp.tanh(params["w"] @ position + params["b"])
        return jnp.sum(hidden) + params["scale"] * jnp.prod(position)

    assert state.sr_o.shape[1] > int(state.sr_rank0)
    assert jax.flatten_util.ravel_pytree(params)[0].shape[0] > positions.shape[0]
    _assert_augmented_matrix_free_products_match_explicit(
        log_psi_apply, params, positions, state, eta=0.99
    )


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


def test_wssr_update_hard_floor_default_matches_previous_formula():
    state, o_aug, e_aug, u, singular_values, vh = _spectral_regularization_fixture()
    damping = 0.1

    result = wssr._wssr_update_from_svd(
        o_aug,
        e_aug,
        state,
        u,
        singular_values,
        vh,
        damping=damping,
        norm_constraint=10.0,
        sr_rank_max=state.sr_o.shape[1],
        sr_scale=1.1,
        constrain_update_norm=False,
        spectral_regularization="hard_floor",
        complement_weight=1.0,
    )

    inv_floor = 1.0 / jnp.square(damping * jnp.abs(singular_values[0]))
    force = o_aug @ e_aug
    projected_force = u.T @ force
    projected_force = projected_force * (jnp.square(1.0 / singular_values) - inv_floor)
    expected = u @ projected_force + inv_floor * force
    np.testing.assert_allclose(
        result.grad_like_update, expected, rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("complement_weight", [0.0, 0.3])
def test_wssr_update_hard_floor_complement_weight_matches_formula(
    complement_weight,
):
    state, o_aug, e_aug, u, singular_values, vh = _spectral_regularization_fixture()
    damping = 0.1

    result = wssr._wssr_update_from_svd(
        o_aug,
        e_aug,
        state,
        u,
        singular_values,
        vh,
        damping=damping,
        norm_constraint=10.0,
        sr_rank_max=state.sr_o.shape[1],
        sr_scale=1.1,
        constrain_update_norm=False,
        spectral_regularization="hard_floor",
        complement_weight=complement_weight,
    )

    expected = _expected_wssr_spectral_update(
        o_aug,
        e_aug,
        u,
        singular_values,
        damping,
        spectral_regularization="hard_floor",
        complement_weight=complement_weight,
    )
    np.testing.assert_allclose(
        result.grad_like_update, expected, rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("complement_weight", [1.0, 0.0])
def test_wssr_update_tikhonov_complement_weight_matches_formula(
    complement_weight,
):
    state, o_aug, e_aug, u, singular_values, vh = _spectral_regularization_fixture()
    damping = 0.1

    result = wssr._wssr_update_from_svd(
        o_aug,
        e_aug,
        state,
        u,
        singular_values,
        vh,
        damping=damping,
        norm_constraint=10.0,
        sr_rank_max=state.sr_o.shape[1],
        sr_scale=1.1,
        constrain_update_norm=False,
        spectral_regularization="tikhonov",
        complement_weight=complement_weight,
    )

    expected = _expected_wssr_spectral_update(
        o_aug,
        e_aug,
        u,
        singular_values,
        damping,
        spectral_regularization="tikhonov",
        complement_weight=complement_weight,
    )
    np.testing.assert_allclose(
        result.grad_like_update, expected, rtol=1e-5, atol=1e-5
    )


def test_wssr_fixed_tikhonov_lambda_is_independent_of_cutoff():
    state, o_aug, e_aug, u, singular_values, vh = _spectral_regularization_fixture()
    fixed_lambda = 0.5
    result = wssr._wssr_update_from_svd(
        o_aug,
        e_aug,
        state,
        u,
        singular_values,
        vh,
        damping=0.1,
        norm_constraint=10.0,
        sr_rank_max=state.sr_o.shape[1],
        sr_scale=1.1,
        constrain_update_norm=False,
        spectral_regularization="tikhonov",
        complement_weight=0.0,
        relative_singular_value_cutoff=0.1,
        tikhonov_lambda=fixed_lambda,
    )

    force = o_aug @ e_aug
    expected = u @ ((u.T @ force) / (jnp.square(singular_values) + fixed_lambda))
    np.testing.assert_allclose(
        result.grad_like_update, expected, rtol=1e-5, atol=1e-5
    )
    assert int(result.active_rank) == 3


def test_wssr_mixed_precision_rank_solve_matches_fixed_tikhonov_update():
    state, o_aug, e_aug, u, singular_values, vh = _spectral_regularization_fixture()
    common = dict(
        o_aug=o_aug,
        e_aug=e_aug,
        state=state,
        u=u,
        singular_values=singular_values,
        vh=vh,
        damping=0.1,
        norm_constraint=10.0,
        sr_rank_max=state.sr_o.shape[1],
        sr_scale=1.1,
        constrain_update_norm=False,
        spectral_regularization="tikhonov",
        complement_weight=0.0,
        relative_singular_value_cutoff=0.1,
        tikhonov_lambda=0.5,
    )
    reference = wssr._wssr_update_from_svd(**common)
    mixed = wssr._wssr_update_from_svd(
        mixed_precision_solve=True,
        mixed_precision_history_width=2,
        **common,
    )

    assert mixed.grad_like_update.dtype == jnp.float32
    chex.assert_trees_all_close(
        mixed.grad_like_update,
        reference.grad_like_update,
        rtol=2e-6,
        atol=2e-6,
    )
    chex.assert_trees_all_equal(mixed.state, reference.state)


@pytest.mark.parametrize("mode", ["naive", "residual"])
def test_wssr_solution_recurrence_matches_rank_space_formula(mode):
    state, o_aug, e_aug, u, singular_values, vh = _spectral_regularization_fixture()
    prior = jnp.array([0.2, -0.3, 0.7], dtype=jnp.float32)
    fixed_lambda = 0.5
    history_width = 2
    result = wssr._wssr_update_from_svd(
        o_aug=o_aug,
        e_aug=e_aug,
        state=state,
        u=u,
        singular_values=singular_values,
        vh=vh,
        damping=0.1,
        norm_constraint=10.0,
        sr_rank_max=state.sr_o.shape[1],
        sr_scale=1.1,
        constrain_update_norm=False,
        spectral_regularization="tikhonov",
        complement_weight=0.0,
        relative_singular_value_cutoff=0.3,
        tikhonov_lambda=fixed_lambda,
        mixed_precision_solve=True,
        mixed_precision_history_width=history_width,
        solution_prior=prior,
        solution_recurrence_mode=mode,
    )

    current_force = o_aug[:, history_width:] @ e_aug[history_width:]
    retained = singular_values / singular_values[0] > 0.3
    numerator = u.T @ current_force
    if mode == "residual":
        numerator = numerator - jnp.square(singular_values) * (u.T @ prior)
    correction = jnp.where(
        retained,
        numerator / (jnp.square(singular_values) + fixed_lambda),
        0.0,
    )
    expected = prior + u @ correction
    chex.assert_trees_all_close(
        result.grad_like_update, expected, rtol=2e-6, atol=2e-6
    )
    # The third mode is truncated, so its prior component must survive exactly.
    assert result.grad_like_update[2] == pytest.approx(prior[2])


def test_full_current_batch_galerkin_matches_full_rank_recurrence():
    """The full-current residual is the code-coordinate SPRING projection.

    VMCNet stores ``o_cur = O_bar.T`` (parameters by samples), so a basis
    spanning ``range(o_cur)`` must reproduce the existing full-rank recurrence.
    This is the small analogue of the r=1600, eta_S=0 C regression anchor.
    """
    o_cur = jnp.array(
        [
            [1.0, 0.0, 2.0],
            [0.0, 1.0, -1.0],
            [2.0, 1.0, 0.0],
            [-1.0, 2.0, 1.0],
            [0.5, -0.25, 0.75],
        ],
        dtype=jnp.float32,
    )
    e_cur = jnp.array([0.4, -0.7, 0.3], dtype=jnp.float32)
    prior = jnp.array([0.2, -0.1, 0.3, 0.4, -0.2], dtype=jnp.float32)
    u, singular_values, vh = jnp.linalg.svd(o_cur, full_matrices=False)
    state = wssr.initialize_wssr_core_state(
        o_cur.shape[0], sr_rank=3, sr_rank_max=3
    )
    rank_coordinate = wssr._wssr_update_from_svd(
        o_aug=o_cur,
        e_aug=e_cur,
        state=state,
        u=u,
        singular_values=singular_values,
        vh=vh,
        damping=0.0,
        norm_constraint=10.0,
        sr_rank_max=3,
        sr_scale=1.0,
        constrain_update_norm=False,
        spectral_regularization="tikhonov",
        complement_weight=0.0,
        relative_singular_value_cutoff=0.0,
        tikhonov_lambda=1e-3,
        mixed_precision_solve=True,
        mixed_precision_history_width=0,
        solution_prior=prior,
        solution_recurrence_mode="residual",
    )
    full_current = wssr.galerkin_residual_solution_recurrence(
        o_cur,
        e_cur,
        u,
        prior,
        lambda_reg=1e-3,
    )

    chex.assert_trees_all_close(
        full_current.direction,
        rank_coordinate.grad_like_update,
        rtol=5e-5,
        atol=5e-5,
    )


def test_cached_current_action_matches_explicit_fp64():
    """An exact SVD factor recovers O.T @ U without approximation."""
    with jax.experimental.enable_x64():
        history = jnp.array(
            [[0.3, -0.2], [0.5, 0.1], [-0.7, 0.4], [0.2, 0.8]],
            dtype=jnp.float64,
        )
        o_cur = jnp.array(
            [[0.6, -0.1, 0.4], [0.2, 0.9, -0.5],
             [-0.3, 0.7, 0.8], [0.1, -0.4, 0.5]],
            dtype=jnp.float64,
        )
        current_scale = jnp.asarray(0.7, dtype=jnp.float64)
        o_aug = jnp.concatenate([history, current_scale * o_cur], axis=1)
        u, singular_values, vh = jnp.linalg.svd(o_aug, full_matrices=False)
        cached = wssr.cached_current_action_from_svd(
            vh,
            singular_values,
            active_rank=jnp.asarray(singular_values.shape[0]),
            storage_width=singular_values.shape[0],
            current_sample_width=o_cur.shape[1],
            current_block_scale=current_scale,
        )
        explicit = o_cur.T @ u

        np.testing.assert_allclose(cached, explicit, rtol=1e-11, atol=1e-11)


def test_right_ssi_returned_current_action_matches_explicit_fp64():
    """A refresh step must not substitute approximate right-SSI V*s."""
    with jax.experimental.enable_x64():
        o_aug = jnp.array(
            [
                [0.4, -0.2, 0.7, 0.1, -0.5],
                [0.8, 0.3, -0.4, 0.6, 0.2],
                [-0.1, 0.9, 0.5, -0.7, 0.4],
                [0.6, -0.8, 0.2, 0.3, 0.9],
                [-0.5, 0.1, 0.8, -0.2, 0.7],
                [0.2, 0.5, -0.6, 0.9, -0.3],
            ],
            dtype=jnp.float64,
        )
        warm_u, _ = jnp.linalg.qr(
            jnp.array(
                [
                    [1.0, 0.2],
                    [0.1, 1.0],
                    [0.5, -0.3],
                    [-0.2, 0.4],
                    [0.3, 0.6],
                    [-0.4, 0.1],
                ],
                dtype=jnp.float64,
            ),
            mode="reduced",
        )
        state = wssr.initialize_wssr_warm_svd_core_state(
            num_params=o_aug.shape[0],
            sr_rank=2,
            sr_rank_max=2,
            dtype=jnp.float64,
            store_warm_u=True,
        )._replace(u=warm_u, has_u=jnp.asarray(True))
        result, current_action = wssr.wssr_warm_svd_right_core_update(
            o_aug,
            jnp.array([0.3, -0.2, 0.4, 0.1, -0.5], dtype=jnp.float64),
            state,
            jax.random.PRNGKey(23),
            damping=0.0,
            norm_constraint=10.0,
            sr_rank_max=2,
            sr_scale=1.0,
            svd_maxiter_initial=2,
            svd_maxiter_warm=1,
            svd_working_rank=2,
            constrain_update_norm=False,
            spectral_regularization="tikhonov",
            complement_weight=0.0,
            relative_singular_value_cutoff=0.0,
            tikhonov_lambda=1e-3,
            return_current_action=True,
            current_sample_width=3,
            reuse_warm_subspace=jnp.asarray(False),
        )
        explicit = o_aug[:, -3:].T @ result.state.u

        np.testing.assert_allclose(
            current_action, explicit, rtol=1e-11, atol=1e-11
        )


def test_cached_and_explicit_galerkin_corrections_match_fp64():
    with jax.experimental.enable_x64():
        o_cur = jnp.array(
            [[1.0, 0.0, 2.0], [0.0, 1.0, -1.0],
             [2.0, 1.0, 0.0], [-1.0, 2.0, 1.0]],
            dtype=jnp.float64,
        )
        e_cur = jnp.array([0.4, -0.7, 0.3], dtype=jnp.float64)
        prior = jnp.array([0.2, -0.1, 0.3, 0.4], dtype=jnp.float64)
        basis, singular_values, vh = jnp.linalg.svd(
            o_cur, full_matrices=False
        )
        cached_action = wssr.cached_current_action_from_svd(
            vh,
            singular_values,
            active_rank=jnp.asarray(singular_values.shape[0]),
            storage_width=singular_values.shape[0],
            current_sample_width=o_cur.shape[1],
        )
        explicit = wssr.galerkin_residual_solution_recurrence(
            o_cur, e_cur, basis, prior, lambda_reg=1e-3
        )
        cached = wssr.galerkin_residual_solution_recurrence(
            o_cur,
            e_cur,
            basis,
            prior,
            lambda_reg=1e-3,
            current_action=cached_action,
        )

        np.testing.assert_allclose(
            cached.correction, explicit.correction, rtol=1e-11, atol=1e-11
        )
        assert 0.0 <= float(cached.residual_reduction_fraction) <= 1.0
        assert cached.residual_reduction_fraction > 0.0


def test_exact_residual_capture_separates_sample_and_parameter_spaces():
    """The two valid projection ratios are bounded but generally unequal."""
    o_cur = jnp.array(
        [[2.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=jnp.float32
    )
    basis = jnp.array([[1.0], [0.0], [0.0]], dtype=jnp.float32)
    sample_residual = jnp.array([3.0, 4.0], dtype=jnp.float32)
    diagnostics = wssr.exact_residual_capture_diagnostics(
        o_cur,
        sample_residual,
        basis,
        applied_coefficients=jnp.array([1.0], dtype=jnp.float32),
    )

    assert diagnostics.sample_projection_ratio == pytest.approx(3.0 / 5.0)
    assert diagnostics.parameter_projection_ratio == pytest.approx(
        6.0 / np.sqrt(52.0)
    )
    assert diagnostics.applied_residual_norm_reduction == pytest.approx(
        1.0 - np.sqrt(17.0) / 5.0
    )
    assert diagnostics.sample_projection_ratio != pytest.approx(
        diagnostics.parameter_projection_ratio
    )


def test_exact_parameter_capture_accepts_an_explicit_full_update_target():
    o_cur = jnp.eye(3, 2, dtype=jnp.float32)
    basis = jnp.array([[1.0], [0.0], [0.0]], dtype=jnp.float32)
    full_current_batch_update = jnp.array([3.0, 4.0, 0.0], dtype=jnp.float32)
    diagnostics = wssr.exact_residual_capture_diagnostics(
        o_cur,
        sample_residual=jnp.array([1.0, 1.0], dtype=jnp.float32),
        basis=basis,
        applied_coefficients=jnp.array([0.0], dtype=jnp.float32),
        parameter_target=full_current_batch_update,
    )

    assert diagnostics.parameter_projection_ratio == pytest.approx(3.0 / 5.0)


def test_applied_residual_capture_metrics_are_bounded_and_consistent():
    o_cur = jnp.array(
        [[1.0, 2.0], [0.5, -1.0], [2.0, 0.25]], dtype=jnp.float32
    )
    basis, _ = jnp.linalg.qr(jnp.eye(3, 2, dtype=jnp.float32))
    result = wssr.galerkin_residual_solution_recurrence(
        o_cur,
        jnp.array([0.8, -0.4], dtype=jnp.float32),
        basis,
        jnp.array([0.3, -0.2, 0.1], dtype=jnp.float32),
        lambda_reg=1e-3,
    )

    assert 0.0 <= result.captured_sample_residual_ratio <= 1.0
    assert 0.0 <= result.residual_norm_reduction <= 1.0
    np.testing.assert_allclose(
        result.residual_norm_reduction,
        1.0 - jnp.sqrt(1.0 - result.residual_reduction_fraction),
        rtol=1e-5,
        atol=1e-6,
    )


def test_device_cholesky_galerkin_matches_host_fp64():
    """The callback-free device backend solves the same regularized system."""
    with jax.experimental.enable_x64():
        o_cur = jnp.array(
            [[1.0, 0.0, 2.0], [0.0, 1.0, -1.0],
             [2.0, 1.0, 0.0], [-1.0, 2.0, 1.0]],
            dtype=jnp.float64,
        )
        e_cur = jnp.array([0.4, -0.7, 0.3], dtype=jnp.float64)
        prior = jnp.array([0.2, -0.1, 0.3, 0.4], dtype=jnp.float64)
        basis, _ = jnp.linalg.qr(o_cur, mode="reduced")
        host = wssr.galerkin_residual_solution_recurrence(
            o_cur,
            e_cur,
            basis,
            prior,
            lambda_reg=1e-3,
            solve_backend="host_fp64",
        )
        device = wssr.galerkin_residual_solution_recurrence(
            o_cur,
            e_cur,
            basis,
            prior,
            lambda_reg=1e-3,
            solve_backend="device_cholesky",
        )

        np.testing.assert_allclose(
            device.correction, host.correction, rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            device.direction, host.direction, rtol=1e-10, atol=1e-10
        )


def test_device_cholesky_galerkin_jaxpr_has_no_callback():
    """GPU-direct mode must remain a pure JAX computation after tracing."""
    o_cur = jnp.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]], dtype=jnp.float32
    )
    e_cur = jnp.array([0.25, -0.5], dtype=jnp.float32)
    prior = jnp.array([0.1, -0.2, 0.3], dtype=jnp.float32)
    basis, _ = jnp.linalg.qr(o_cur, mode="reduced")

    def correction_fn(score, residual, subspace, history):
        return wssr.galerkin_residual_solution_recurrence(
            score,
            residual,
            subspace,
            history,
            lambda_reg=1e-3,
            solve_backend="device_cholesky",
        ).correction

    jaxpr = str(
        jax.make_jaxpr(correction_fn)(o_cur, e_cur, basis, prior)
    )
    assert "pure_callback" not in jaxpr
    assert "cholesky" in jaxpr


def test_galerkin_rejects_unknown_solve_backend():
    o_cur = jnp.eye(2, dtype=jnp.float32)
    with pytest.raises(ValueError, match="solve_backend"):
        wssr.galerkin_residual_solution_recurrence(
            o_cur,
            jnp.ones(2, dtype=jnp.float32),
            o_cur,
            jnp.zeros(2, dtype=jnp.float32),
            lambda_reg=1e-3,
            solve_backend="unknown",
        )


def test_delayed_refresh_reuses_stored_left_subspace():
    o_aug = jnp.array(
        [[1.0, 0.2, -0.4, 0.1], [0.3, 1.2, 0.7, -0.2],
         [-0.5, 0.4, 1.1, 0.6], [0.8, -0.3, 0.2, 0.9],
         [0.1, 0.5, -0.7, 1.3]],
        dtype=jnp.float32,
    )
    warm_u, _ = jnp.linalg.qr(
        jnp.array(
            [[1.0, 0.2], [0.1, 1.0], [0.5, -0.3],
             [-0.2, 0.4], [0.3, 0.6]],
            dtype=jnp.float32,
        ),
        mode="reduced",
    )
    state = wssr.WSSRWarmSVDCoreState(
        sr_o=warm_u,
        ek=jnp.zeros(2, dtype=jnp.float32),
        sr_rank0=jnp.asarray(2, dtype=jnp.int32),
        sr_rank=jnp.asarray(2, dtype=jnp.int32),
        u=warm_u,
        has_u=jnp.asarray(True),
    )
    reused_u, singular_values, vh, rank = (
        wssr._wssr_right_svd_decomposition(
            o_aug,
            state,
            jax.random.PRNGKey(7),
            working_rank_max=2,
            svd_maxiter_initial=3,
            svd_maxiter_warm=2,
            exact_first=False,
            eps=1e-12,
            reuse_warm_subspace=jnp.asarray(True),
        )
    )

    outside = reused_u - warm_u @ (warm_u.T @ reused_u)
    np.testing.assert_allclose(outside, 0.0, rtol=0.0, atol=2e-6)
    np.testing.assert_allclose(
        o_aug.T @ reused_u,
        vh.T * singular_values[None, :],
        rtol=2e-5,
        atol=2e-5,
    )
    assert rank == 2


def test_lazy_fixed_basis_keeps_u_and_recomputes_exact_current_action():
    """Lazy SSI must not rotate U between scheduled refreshes."""
    o_aug = jnp.array(
        [[1.0, 0.2, -0.4, 0.1], [0.3, 1.2, 0.7, -0.2],
         [-0.5, 0.4, 1.1, 0.6], [0.8, -0.3, 0.2, 0.9],
         [0.1, 0.5, -0.7, 1.3]],
        dtype=jnp.float32,
    )
    warm_u, _ = jnp.linalg.qr(
        jnp.array(
            [[1.0, 0.2], [0.1, 1.0], [0.5, -0.3],
             [-0.2, 0.4], [0.3, 0.6]],
            dtype=jnp.float32,
        ),
        mode="reduced",
    )
    state = wssr.WSSRWarmSVDCoreState(
        sr_o=warm_u * jnp.array([2.0, 0.5]),
        ek=jnp.zeros(2, dtype=jnp.float32),
        sr_rank0=jnp.asarray(2, dtype=jnp.int32),
        sr_rank=jnp.asarray(2, dtype=jnp.int32),
        u=warm_u,
        has_u=jnp.asarray(True),
    )
    reused_u, singular_values, vh, rank = (
        wssr._wssr_right_svd_decomposition(
            o_aug,
            state,
            jax.random.PRNGKey(11),
            working_rank_max=2,
            svd_maxiter_initial=3,
            svd_maxiter_warm=2,
            exact_first=False,
            eps=1e-12,
            reuse_warm_subspace=jnp.asarray(True),
            fixed_warm_subspace=True,
        )
    )

    np.testing.assert_allclose(reused_u, warm_u, rtol=0.0, atol=2e-6)
    np.testing.assert_allclose(
        vh.T * singular_values[None, :],
        o_aug.T @ warm_u,
        rtol=2e-6,
        atol=2e-6,
    )
    assert rank == 2


def test_full_current_batch_large_lambda_suppresses_correction():
    o_cur = jnp.array(
        [[1.0, 2.0], [0.5, -1.0], [2.0, 0.25]], dtype=jnp.float32
    )
    basis, _ = jnp.linalg.qr(jnp.eye(3, 2, dtype=jnp.float32))
    prior = jnp.array([0.3, -0.2, 0.1], dtype=jnp.float32)
    result = wssr.galerkin_residual_solution_recurrence(
        o_cur,
        jnp.array([0.8, -0.4], dtype=jnp.float32),
        basis,
        prior,
        lambda_reg=1e12,
    )

    assert jnp.linalg.norm(result.correction) < 1e-9
    chex.assert_trees_all_close(result.direction, prior, rtol=0.0, atol=1e-8)


def test_full_current_batch_error_feedback_vanishes_for_complete_basis():
    o_cur = jnp.array(
        [[1.0, 2.0], [0.5, -1.0], [2.0, 0.25]], dtype=jnp.float32
    )
    result = wssr.galerkin_residual_solution_recurrence(
        o_cur,
        jnp.array([0.8, -0.4], dtype=jnp.float32),
        jnp.eye(3, dtype=jnp.float32),
        jnp.array([0.3, -0.2, 0.1], dtype=jnp.float32),
        lambda_reg=1e-3,
        error_feedback=jnp.array([0.2, -0.1, 0.3], dtype=jnp.float32),
        enable_error_feedback=True,
    )

    chex.assert_trees_all_close(
        result.error_feedback, jnp.zeros(3), rtol=0.0, atol=1e-7
    )
    assert result.error_feedback_clip_increment == 0


def test_error_feedback_history_is_decayed_before_reinjection():
    result = wssr.galerkin_residual_solution_recurrence(
        jnp.zeros((3, 1), dtype=jnp.float32),
        jnp.ones((1,), dtype=jnp.float32),
        jnp.array([[1.0], [0.0], [0.0]], dtype=jnp.float32),
        jnp.zeros((3,), dtype=jnp.float32),
        lambda_reg=1e-3,
        error_feedback=jnp.array([0.0, 2.0, 0.0], dtype=jnp.float32),
        enable_error_feedback=True,
        error_feedback_decay=0.5,
        error_feedback_norm_cap=100.0,
        error_feedback_cap_reference="sample_residual",
        solve_backend="device_cholesky",
    )

    np.testing.assert_allclose(
        result.error_feedback,
        jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32),
        rtol=0.0,
        atol=1e-6,
    )
    assert result.error_feedback_clip_increment == 0
    assert result.error_feedback_to_sample_ratio == pytest.approx(1.0)


def test_error_feedback_sample_residual_cap_and_health_ratios():
    result = wssr.galerkin_residual_solution_recurrence(
        jnp.array([[0.0], [100.0], [0.0]], dtype=jnp.float32),
        jnp.ones((1,), dtype=jnp.float32),
        jnp.array([[1.0], [0.0], [0.0]], dtype=jnp.float32),
        jnp.zeros((3,), dtype=jnp.float32),
        lambda_reg=1e-3,
        error_feedback=jnp.zeros((3,), dtype=jnp.float32),
        enable_error_feedback=True,
        error_feedback_decay=0.95,
        error_feedback_norm_cap=10.0,
        error_feedback_cap_reference="sample_residual",
        solve_backend="device_cholesky",
    )

    assert result.error_feedback_uncapped_norm == pytest.approx(100.0)
    assert result.error_feedback_cap_norm == pytest.approx(10.0)
    assert result.error_feedback_norm == pytest.approx(10.0)
    assert result.error_feedback_to_sample_ratio == pytest.approx(10.0)
    assert result.error_feedback_to_parameter_conflict_ratio == pytest.approx(
        0.1
    )
    assert result.error_feedback_clip_increment == 1


def test_subspace_eta_zero_augmented_path_matches_current_batch_operator():
    o_cur = jnp.arange(12.0, dtype=jnp.float32).reshape(3, 4)
    e_cur = jnp.array([0.5, -0.25, 0.75, -1.0], dtype=jnp.float32)
    state = wssr.WSSRCoreState(
        sr_o=jnp.arange(15.0, dtype=jnp.float32).reshape(3, 5),
        ek=jnp.arange(5.0, dtype=jnp.float32),
        sr_rank0=jnp.asarray(3, dtype=jnp.int32),
        sr_rank=jnp.asarray(3, dtype=jnp.int32),
    )
    o_aug, e_aug = wssr.augment_wssr_subspace_system(
        o_cur, e_cur, state, eta=0.0
    )

    chex.assert_trees_all_close(
        o_aug @ o_aug.T, o_cur @ o_cur.T, rtol=0.0, atol=1e-6
    )
    chex.assert_trees_all_close(
        o_aug @ e_aug, o_cur @ e_cur, rtol=0.0, atol=1e-6
    )


def test_subspace_history_keeps_current_batch_at_unit_weight():
    o_cur = jnp.ones((3, 2), dtype=jnp.float32)
    e_cur = jnp.array([2.0, 4.0], dtype=jnp.float32)
    state = wssr.WSSRCoreState(
        sr_o=jnp.arange(15.0, dtype=jnp.float32).reshape(3, 5),
        ek=jnp.arange(5.0, dtype=jnp.float32),
        sr_rank0=jnp.asarray(2, dtype=jnp.int32),
        sr_rank=jnp.asarray(3, dtype=jnp.int32),
    )
    o_aug, e_aug = wssr.augment_wssr_subspace_system(
        o_cur, e_cur, state, eta=0.3
    )

    chex.assert_trees_all_close(o_aug[:, -2:], o_cur)
    chex.assert_trees_all_close(e_aug[-2:], e_cur)
    chex.assert_trees_all_close(
        o_aug[:, :2], jnp.sqrt(0.3) * state.sr_o[:, :2]
    )


def test_bias_corrected_history_eta_ramps_to_target():
    steps = jnp.arange(6, dtype=jnp.int32)
    actual = jax.vmap(lambda step: wssr.bias_corrected_history_eta(0.99, step))(
        steps
    )
    expected = jnp.array([0.0, 0.5, 2.0 / 3.0, 0.75, 0.8, 5.0 / 6.0])
    chex.assert_trees_all_close(actual, expected)
    assert wssr.bias_corrected_history_eta(0.8, 100) == pytest.approx(0.8)


def test_wssr_relative_cutoff_is_independent_of_legacy_tikhonov_lambda():
    state, o_aug, e_aug, u, singular_values, vh = _spectral_regularization_fixture()
    damping = 0.1
    relative_cutoff = 0.3
    result = wssr._wssr_update_from_svd(
        o_aug,
        e_aug,
        state,
        u,
        singular_values,
        vh,
        damping=damping,
        norm_constraint=10.0,
        sr_rank_max=state.sr_o.shape[1],
        sr_scale=1.1,
        constrain_update_norm=False,
        spectral_regularization="tikhonov",
        complement_weight=0.0,
        relative_singular_value_cutoff=relative_cutoff,
        tikhonov_lambda=-1.0,
    )

    force = o_aug @ e_aug
    retained = singular_values / singular_values[0] > relative_cutoff
    legacy_lambda = jnp.square(damping * singular_values[0])
    expected = u @ (
        (u.T @ force)
        * retained.astype(u.dtype)
        / (jnp.square(singular_values) + legacy_lambda)
    )
    np.testing.assert_allclose(
        result.grad_like_update, expected, rtol=1e-5, atol=1e-5
    )
    assert int(result.active_rank) == 2


def test_wssr_update_rejects_invalid_spectral_regularization():
    state, o_aug, e_aug, u, singular_values, vh = _spectral_regularization_fixture()

    with pytest.raises(ValueError, match="spectral_regularization"):
        wssr._wssr_update_from_svd(
            o_aug,
            e_aug,
            state,
            u,
            singular_values,
            vh,
            damping=0.1,
            norm_constraint=10.0,
            sr_rank_max=state.sr_o.shape[1],
            sr_scale=1.1,
            constrain_update_norm=False,
            spectral_regularization="bad_mode",
        )


def test_wssr_update_rejects_negative_complement_weight():
    state, o_aug, e_aug, u, singular_values, vh = _spectral_regularization_fixture()

    with pytest.raises(ValueError, match="complement_weight"):
        wssr._wssr_update_from_svd(
            o_aug,
            e_aug,
            state,
            u,
            singular_values,
            vh,
            damping=0.1,
            norm_constraint=10.0,
            sr_rank_max=state.sr_o.shape[1],
            sr_scale=1.1,
            constrain_update_norm=False,
            complement_weight=-0.1,
        )


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


def test_wssr_warm_svd_working_rank_keeps_fixed_state_shape_and_masks_tail():
    state = wssr.initialize_wssr_warm_svd_core_state(
        num_params=5, sr_rank=4, sr_rank_max=6, dtype=jnp.float32
    )
    o_aug = jnp.array(
        [
            [3.0, 0.5, -1.0, 0.0, 1.0, -0.5, 0.25, 0.75],
            [0.0, 2.5, 0.5, -1.5, 0.0, 1.0, -0.25, 0.5],
            [1.0, -0.25, 2.0, 0.75, -1.0, 0.0, 0.5, -0.5],
            [0.5, 1.0, -0.75, 1.5, 0.25, -1.0, 0.0, 0.25],
            [-1.0, 0.25, 0.5, -0.5, 2.0, 0.75, -0.75, 1.0],
        ],
        dtype=jnp.float32,
    )
    e_aug = jnp.array([0.5, -0.25, 0.125, -0.75, 0.6, -0.4, 0.2, -0.1])

    result = wssr.wssr_warm_svd_core_update(
        o_aug,
        e_aug,
        state,
        key=jax.random.PRNGKey(17),
        damping=1e-5,
        norm_constraint=10.0,
        sr_rank_max=6,
        sr_scale=2.0,
        svd_maxiter_initial=2,
        svd_maxiter_warm=1,
        svd_working_rank=3,
        constrain_update_norm=False,
        spectral_regularization="tikhonov",
        complement_weight=0.3,
    )

    assert result.state.sr_o.shape == (5, 6)
    assert result.state.ek.shape == (6,)
    assert result.state.u.shape == (5, 6)
    assert result.state.sr_rank0 <= 3
    assert result.state.sr_rank <= 3
    np.testing.assert_allclose(result.state.sr_o[:, 3:], 0.0, atol=1e-6)
    np.testing.assert_allclose(result.state.ek[3:], 0.0, atol=1e-6)
    np.testing.assert_allclose(result.state.u[:, 3:], 0.0, atol=1e-6)
    _assert_wssr_state_all_finite(result.state)


def test_wssr_warm_svd_working_rank_caps_stale_rank_growth():
    state = wssr.initialize_wssr_warm_svd_core_state(
        num_params=4, sr_rank=6, sr_rank_max=6, dtype=jnp.float32
    )
    o_aug = jnp.array(
        [
            [4.0, 0.0, 0.0, 0.5, -1.0, 0.25],
            [0.0, 3.0, 0.0, -0.25, 0.5, -0.75],
            [0.0, 0.0, 2.0, 1.0, 0.25, 0.5],
            [1.0, -1.0, 0.5, 0.0, 2.0, -0.5],
        ],
        dtype=jnp.float32,
    )
    e_aug = jnp.array([0.5, -0.25, 0.125, -0.75, 0.4, -0.2])

    result = wssr.wssr_warm_svd_core_update(
        o_aug,
        e_aug,
        state,
        key=jax.random.PRNGKey(19),
        damping=1e-5,
        norm_constraint=10.0,
        sr_rank_max=6,
        sr_scale=2.0,
        svd_maxiter_initial=2,
        svd_maxiter_warm=1,
        svd_working_rank=2,
        constrain_update_norm=False,
    )

    assert result.state.sr_rank0 <= 2
    assert result.state.sr_rank <= 2
    np.testing.assert_allclose(result.state.sr_o[:, 2:], 0.0, atol=1e-6)
    np.testing.assert_allclose(result.state.ek[2:], 0.0, atol=1e-6)
    np.testing.assert_allclose(result.state.u[:, 2:], 0.0, atol=1e-6)


def test_wssr_warm_svd_full_working_rank_matches_fallback_behavior():
    state = wssr.initialize_wssr_warm_svd_core_state(
        num_params=4, sr_rank=3, sr_rank_max=4, dtype=jnp.float32
    )
    o_aug = jnp.array(
        [
            [2.0, 0.5, -1.0, 0.25, 0.75],
            [0.0, 1.5, 0.5, -0.75, -0.25],
            [1.0, -0.25, 1.25, 0.5, 0.0],
            [0.5, 0.75, -0.5, 2.0, -1.0],
        ],
        dtype=jnp.float32,
    )
    e_aug = jnp.array([0.5, -0.25, 0.125, -0.75, 0.4])
    kwargs = dict(
        key=jax.random.PRNGKey(23),
        damping=0.05,
        norm_constraint=10.0,
        sr_rank_max=4,
        sr_scale=1.5,
        svd_maxiter_initial=2,
        svd_maxiter_warm=1,
        constrain_update_norm=False,
    )

    fallback = wssr.wssr_warm_svd_core_update(o_aug, e_aug, state, **kwargs)
    explicit_full = wssr.wssr_warm_svd_core_update(
        o_aug,
        e_aug,
        state,
        svd_working_rank=4,
        **kwargs,
    )

    chex.assert_trees_all_close(
        fallback.grad_like_update, explicit_full.grad_like_update, rtol=1e-5, atol=1e-5
    )
    chex.assert_trees_all_close(
        fallback.state, explicit_full.state, rtol=1e-5, atol=1e-5
    )
    chex.assert_trees_all_close(fallback.active_rank, explicit_full.active_rank)


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
            "svd_working_rank",
            "constrain_update_norm",
            "spectral_regularization",
            "complement_weight",
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
            "svd_working_rank",
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
    np.testing.assert_allclose(eager.state.u[:, 2:], 0.0, atol=1e-6)
    np.testing.assert_allclose(jitted.state.u[:, 2:], 0.0, atol=1e-6)
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


def test_initialize_wssr_warm_svd_core_state_store_warm_u_shapes():
    full_state = wssr.initialize_wssr_warm_svd_core_state(
        num_params=4,
        sr_rank=2,
        sr_rank_max=5,
        dtype=jnp.float32,
        store_warm_u=True,
    )
    derived_state = wssr.initialize_wssr_warm_svd_core_state(
        num_params=4,
        sr_rank=2,
        sr_rank_max=5,
        dtype=jnp.float32,
        store_warm_u=False,
    )

    assert full_state.u.shape == (4, 5)
    assert derived_state.u.shape == (4, 0)
    assert full_state.sr_o.shape == derived_state.sr_o.shape == (4, 5)


def test_explicit_current_block_operator_matches_materialized_augmentation():
    o_cur = jnp.array(
        [[0.4, -0.2, 0.1], [0.3, 0.5, -0.4], [-0.1, 0.2, 0.6]],
        dtype=jnp.float32,
    )
    e_cur = jnp.array([0.2, -0.1, 0.3], dtype=jnp.float32)
    state = wssr.initialize_wssr_warm_svd_core_state(
        3, 2, 2, dtype=jnp.float32, store_warm_u=False
    )._replace(
        sr_o=jnp.array(
            [[0.5, 0.0], [0.1, 0.4], [-0.2, 0.3]], dtype=jnp.float32
        ),
        ek=jnp.array([0.7, -0.2], dtype=jnp.float32),
        sr_rank0=jnp.array(2),
    )
    eta = 0.3
    o_aug, e_aug = wssr.augment_wssr_system(o_cur, e_cur, state, eta)
    probe = jnp.arange(15, dtype=jnp.float32).reshape(5, 3) / 10.0
    left_probe = jnp.arange(9, dtype=jnp.float32).reshape(3, 3) / 7.0

    np.testing.assert_allclose(
        wssr.explicit_current_augmented_matmat(
            o_cur, state, eta, probe
        ),
        o_aug @ probe,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        wssr.explicit_current_augmented_rmatmat(
            o_cur, state, eta, left_probe
        ),
        o_aug.T @ left_probe,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        wssr.explicit_current_augmented_matvec(
            o_cur, state, eta, e_aug
        ),
        o_aug @ e_aug,
        rtol=1e-6,
        atol=1e-6,
    )


def test_explicit_current_block_core_matches_materialized_residual_recurrence():
    o_cur = jnp.array(
        [
            [0.4, -0.2, -0.2],
            [0.1, 0.3, -0.4],
            [-0.5, 0.2, 0.3],
            [0.2, -0.1, -0.1],
        ],
        dtype=jnp.float32,
    )
    e_cur = jnp.array([0.2, -0.3, 0.1], dtype=jnp.float32)
    state = wssr.initialize_wssr_warm_svd_core_state(
        4, 3, 3, dtype=jnp.float32, store_warm_u=False
    )
    e_aug = wssr.augment_wssr_residuals(e_cur, state, eta=0.0)
    o_aug, _ = wssr.augment_wssr_system(o_cur, e_cur, state, eta=0.0)
    prior = jnp.array([0.05, -0.02, 0.01, 0.03], dtype=jnp.float32)
    kwargs = dict(
        key=jax.random.PRNGKey(7),
        damping=0.001,
        norm_constraint=0.001,
        sr_rank_max=3,
        svd_maxiter_initial=4,
        svd_maxiter_warm=2,
        svd_working_rank=3,
        constrain_update_norm=False,
        spectral_regularization="tikhonov",
        complement_weight=0.0,
        relative_singular_value_cutoff=0.0,
        tikhonov_lambda=0.001,
        mixed_precision_solve=True,
        solution_prior=prior,
        solution_recurrence_mode="residual",
    )
    explicit = wssr.wssr_warm_svd_right_core_update(
        o_aug, e_aug, state, **kwargs
    )
    blockwise = (
        wssr.wssr_warm_svd_right_core_update_explicit_current_blocks(
            o_cur,
            e_cur,
            e_aug,
            state,
            eta=0.0,
            **kwargs,
        )
    )

    np.testing.assert_allclose(
        blockwise.grad_like_update,
        explicit.grad_like_update,
        rtol=2e-5,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        blockwise.state.sr_o,
        explicit.state.sr_o,
        rtol=2e-5,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        blockwise.state.ek,
        explicit.state.ek,
        rtol=2e-5,
        atol=2e-6,
    )


def test_explicit_current_block_core_matches_materialized_dual_cap_complement():
    o_cur = jnp.array(
        [
            [0.4, -0.2, -0.2],
            [0.1, 0.3, -0.4],
            [-0.5, 0.2, 0.3],
            [0.2, -0.1, -0.1],
        ],
        dtype=jnp.float32,
    )
    e_cur = jnp.array([0.2, -0.3, 0.1], dtype=jnp.float32)
    state = wssr.initialize_wssr_warm_svd_core_state(
        4, 2, 2, dtype=jnp.float32, store_warm_u=False
    )
    e_aug = wssr.augment_wssr_residuals(e_cur, state, eta=0.0)
    o_aug, _ = wssr.augment_wssr_system(o_cur, e_cur, state, eta=0.0)
    kwargs = dict(
        key=jax.random.PRNGKey(11),
        damping=0.001,
        norm_constraint=0.001,
        sr_rank_max=2,
        svd_maxiter_initial=4,
        svd_maxiter_warm=2,
        svd_working_rank=2,
        constrain_update_norm=False,
        spectral_regularization="tikhonov",
        complement_weight=1e-4,
        relative_singular_value_cutoff=0.0,
        tikhonov_lambda=0.001,
        experimental_mode="adaptive_complement",
        adaptive_complement_beta=0.1,
        adaptive_complement_beta_function=0.1,
    )
    explicit = wssr.wssr_warm_svd_right_core_update(
        o_aug,
        e_aug,
        state,
        complement_function_operator=o_cur,
        **kwargs,
    )
    blockwise = wssr.wssr_warm_svd_right_core_update_explicit_current_blocks(
        o_cur, e_cur, e_aug, state, eta=0.0, **kwargs
    )

    np.testing.assert_allclose(
        blockwise.grad_like_update,
        explicit.grad_like_update,
        rtol=2e-5,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        blockwise.state.sr_o, explicit.state.sr_o, rtol=2e-5, atol=2e-6
    )


def test_recover_u_from_sr_o_masks_inactive_and_zero_columns():
    sr_o = jnp.array(
        [
            [3.0, 0.0, 1.0, 2.0],
            [4.0, 0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )

    recovered = wssr.recover_u_from_sr_o(
        sr_o,
        sr_rank0=jnp.array(3),
        rank_capacity=5,
    )

    expected = jnp.array(
        [
            [0.6, 0.0, 1.0, 0.0, 0.0],
            [0.8, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    assert recovered.dtype == sr_o.dtype
    assert recovered.shape == (2, 5)
    np.testing.assert_allclose(recovered, expected, rtol=1e-6, atol=1e-6)


def test_recover_right_basis_from_sr_o_projection_matches_recovered_u_projection():
    o_aug = jnp.array(
        [
            [1.0, -0.2, 0.3, 0.1],
            [0.5, 0.4, -0.7, 0.2],
            [-0.1, 0.8, 0.6, -0.3],
        ],
        dtype=jnp.float32,
    )
    sr_o = jnp.array(
        [
            [3.0, 0.0, 1.0],
            [4.0, 0.0, -2.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=jnp.float32,
    )
    sr_rank0 = jnp.array(2)
    rank_capacity = 3

    recovered_u = wssr.recover_u_from_sr_o(sr_o, sr_rank0, rank_capacity)
    expected = o_aug.T @ recovered_u
    projection = wssr.recover_right_basis_from_sr_o_projection(
        o_aug,
        sr_o,
        sr_rank0,
        rank_capacity,
    )

    assert projection.shape == (o_aug.shape[1], rank_capacity)
    np.testing.assert_allclose(projection, expected, rtol=1e-6, atol=1e-6)


def test_right_warm_start_svd_recovers_left_basis_without_persistent_u():
    o_aug = jnp.array(
        [
            [1.0, 0.2, 0.0, 0.1],
            [0.0, 1.0, 0.3, 0.2],
            [0.2, 0.0, 1.0, 0.4],
        ],
        dtype=jnp.float32,
    )
    state = wssr.initialize_wssr_warm_svd_core_state(
        num_params=3,
        sr_rank=2,
        sr_rank_max=3,
        dtype=jnp.float32,
        store_warm_u=False,
    )._replace(
        sr_o=jnp.array(
            [
                [3.0, 0.0, 0.0],
                [4.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        ),
        sr_rank0=jnp.array(2),
        has_u=jnp.array(False),
    )

    u, singular_values, vh, rank = wssr.right_warm_start_svd(
        o_aug,
        state,
        key=jax.random.PRNGKey(3),
        maxiter_initial=2,
        maxiter_warm=1,
        svd_working_rank=2,
    )

    assert state.u.shape == (3, 0)
    assert u.shape == (3, 2)
    assert singular_values.shape == (2,)
    assert vh.shape == (2, 4)
    assert rank == jnp.asarray(2)
    assert jnp.all(jnp.isfinite(u))
    assert jnp.all(jnp.isfinite(singular_values))
    assert jnp.all(jnp.isfinite(vh))


def test_wssr_warm_svd_right_uses_left_basis_state_shape():
    right_state = wssr.initialize_wssr_warm_svd_core_state(
        num_params=3, sr_rank=2, sr_rank_max=5, dtype=jnp.float32
    )
    left_state = wssr.initialize_wssr_warm_svd_core_state(
        num_params=3, sr_rank=2, sr_rank_max=5, dtype=jnp.float32
    )

    assert right_state.sr_o.shape == (3, 5)
    assert right_state.ek.shape == (5,)
    assert right_state.u.shape == (3, 5)
    assert right_state.has_u == jnp.array(False)
    assert hasattr(left_state, "u")
    assert not hasattr(left_state, "v")
    chex.assert_trees_all_close(right_state, left_state)


def _assert_right_warm_start_svd_matfree_matches_explicit(
    log_psi_apply,
    params,
    positions,
    state,
    eta,
    key,
    svd_working_rank,
    maxiter_initial,
    maxiter_warm,
):
    o_cur, _ = wssr.center_and_scale_score_matrix(log_psi_apply, params, positions)
    e_cur = jnp.linspace(-0.3, 0.5, positions.shape[0])
    o_aug, e_aug = wssr.augment_wssr_system(o_cur, e_cur, state, eta)
    explicit = wssr.right_warm_start_svd(
        o_aug,
        state,
        key,
        maxiter_initial=maxiter_initial,
        maxiter_warm=maxiter_warm,
        svd_working_rank=svd_working_rank,
    )
    matfree = wssr.right_warm_start_svd_matfree(
        log_psi_apply,
        params,
        positions,
        state,
        eta,
        key,
        maxiter_initial=maxiter_initial,
        maxiter_warm=maxiter_warm,
        svd_working_rank=svd_working_rank,
    )
    u_explicit, s_explicit, vh_explicit, rank_explicit = explicit
    u_matfree, s_matfree, vh_matfree, rank_matfree = matfree
    active_rank = int(rank_explicit)

    chex.assert_trees_all_close(rank_matfree, rank_explicit)
    np.testing.assert_allclose(s_matfree, s_explicit, rtol=1e-4, atol=1e-4)
    assert u_matfree.shape == u_explicit.shape
    assert vh_matfree.shape == vh_explicit.shape

    u_projector_explicit = u_explicit[:, :active_rank] @ u_explicit[:, :active_rank].T
    u_projector_matfree = u_matfree[:, :active_rank] @ u_matfree[:, :active_rank].T
    vh_basis_explicit = vh_explicit[:active_rank, :].T
    vh_basis_matfree = vh_matfree[:active_rank, :].T
    vh_projector_explicit = vh_basis_explicit @ vh_basis_explicit.T
    vh_projector_matfree = vh_basis_matfree @ vh_basis_matfree.T

    np.testing.assert_allclose(
        u_projector_matfree, u_projector_explicit, rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        vh_projector_matfree, vh_projector_explicit, rtol=1e-4, atol=1e-4
    )

    explicit_update = wssr._wssr_update_from_svd(
        o_aug,
        e_aug,
        state,
        u_explicit,
        s_explicit,
        vh_explicit,
        damping=0.05,
        norm_constraint=10.0,
        sr_rank_max=state.sr_o.shape[1],
        sr_scale=1.1,
        constrain_update_norm=False,
        rank_update_max=svd_working_rank,
    )
    matfree_update = wssr._wssr_update_from_svd(
        o_aug,
        e_aug,
        state,
        u_matfree,
        s_matfree,
        vh_matfree,
        damping=0.05,
        norm_constraint=10.0,
        sr_rank_max=state.sr_o.shape[1],
        sr_scale=1.1,
        constrain_update_norm=False,
        rank_update_max=svd_working_rank,
    )
    np.testing.assert_allclose(
        matfree_update.grad_like_update,
        explicit_update.grad_like_update,
        rtol=1e-4,
        atol=1e-4,
    )


def test_right_warm_start_svd_matfree_matches_explicit_initial_branch():
    params = _tiny_params()
    positions = _tiny_positions()
    state = wssr.WSSRWarmSVDCoreState(
        sr_o=jnp.array(
            [
                [1.0, -2.0, 30.0, -40.0, 50.0],
                [0.5, 1.5, -50.0, 20.0, -10.0],
                [-1.0, 0.25, 70.0, -80.0, 90.0],
            ]
        ),
        ek=jnp.array([0.2, -0.1, 10.0, -20.0, 30.0]),
        sr_rank0=jnp.array(0),
        sr_rank=jnp.array(2),
        u=jnp.zeros((3, 5)),
        has_u=jnp.array(False),
    )

    assert positions.shape[0] > jax.flatten_util.ravel_pytree(params)[0].shape[0]
    _assert_right_warm_start_svd_matfree_matches_explicit(
        _log_psi_apply,
        params,
        positions,
        state,
        eta=0.99,
        key=jax.random.PRNGKey(21),
        svd_working_rank=2,
        maxiter_initial=2,
        maxiter_warm=1,
    )


def test_right_warm_start_svd_matfree_matches_explicit_warm_branch_with_history():
    params = {
        "w": jnp.array([[0.2, -0.1, 0.3], [0.7, -0.4, 0.5]]),
        "b": jnp.array([0.1, -0.2]),
        "scale": jnp.array(0.4),
    }
    positions = jnp.array(
        [
            [0.5, -1.0, 0.25],
            [1.5, 0.2, -0.75],
            [-0.25, 0.4, 1.0],
        ]
    )
    state = wssr.WSSRWarmSVDCoreState(
        sr_o=jnp.array(
            [
                [0.5, -1.0, 10.0, 20.0, -30.0],
                [1.5, 0.25, -10.0, 40.0, 50.0],
                [-0.5, 0.75, 60.0, -70.0, 80.0],
                [0.2, -0.4, -90.0, 100.0, 110.0],
                [1.2, 0.8, 120.0, 130.0, -140.0],
                [-1.1, 0.3, 150.0, -160.0, 170.0],
                [0.9, -0.7, -180.0, 190.0, 200.0],
                [0.4, 1.4, 210.0, -220.0, 230.0],
                [-0.8, 0.6, -240.0, 250.0, -260.0],
            ]
        ),
        ek=jnp.array([0.3, -0.2, 100.0, 200.0, -300.0]),
        sr_rank0=jnp.array(2),
        sr_rank=jnp.array(3),
        u=jnp.eye(9, 5),
        has_u=jnp.array(True),
    )

    def log_psi_apply(params, position):
        hidden = jnp.tanh(params["w"] @ position + params["b"])
        return jnp.sum(hidden) + params["scale"] * jnp.prod(position)

    assert state.sr_o.shape[1] > int(state.sr_rank0)
    assert jax.flatten_util.ravel_pytree(params)[0].shape[0] > positions.shape[0]
    _assert_right_warm_start_svd_matfree_matches_explicit(
        log_psi_apply,
        params,
        positions,
        state,
        eta=0.99,
        key=jax.random.PRNGKey(22),
        svd_working_rank=3,
        maxiter_initial=2,
        maxiter_warm=1,
    )


def _assert_wssr_warm_svd_right_core_matfree_matches_explicit(
    log_psi_apply,
    params,
    positions,
    local_energies,
    state,
    key,
    eta,
    svd_working_rank,
    constrain_update_norm,
):
    energy = jnp.mean(local_energies)
    kwargs = dict(
        key=key,
        eta=eta,
        damping=0.05,
        norm_constraint=0.5,
        sr_rank_max=state.sr_o.shape[1],
        sr_scale=1.1,
        svd_maxiter_initial=2,
        svd_maxiter_warm=1,
        svd_working_rank=svd_working_rank,
        constrain_update_norm=constrain_update_norm,
    )
    explicit_update, explicit_state, explicit_rank = (
        wssr.compute_wssr_warm_svd_right_core_update(
            log_psi_apply,
            params,
            positions,
            local_energies,
            energy,
            state,
            **kwargs,
        )
    )
    matfree_update, matfree_state, matfree_rank = (
        wssr.compute_wssr_warm_svd_right_core_update_matfree(
            log_psi_apply,
            params,
            positions,
            local_energies,
            energy,
            state,
            **kwargs,
        )
    )
    flat_explicit_update, _ = jax.flatten_util.ravel_pytree(explicit_update)
    flat_matfree_update, _ = jax.flatten_util.ravel_pytree(matfree_update)
    active_rank = int(explicit_state.sr_rank0)

    np.testing.assert_allclose(
        flat_matfree_update, flat_explicit_update, rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        matfree_state.sr_o, explicit_state.sr_o, rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        matfree_state.ek, explicit_state.ek, rtol=1e-4, atol=1e-4
    )
    chex.assert_trees_all_close(matfree_state.sr_rank0, explicit_state.sr_rank0)
    chex.assert_trees_all_close(matfree_state.sr_rank, explicit_state.sr_rank)
    chex.assert_trees_all_close(matfree_state.has_u, explicit_state.has_u)
    chex.assert_trees_all_close(matfree_rank, explicit_rank)

    u_projector_explicit = (
        explicit_state.u[:, :active_rank] @ explicit_state.u[:, :active_rank].T
    )
    u_projector_matfree = (
        matfree_state.u[:, :active_rank] @ matfree_state.u[:, :active_rank].T
    )
    np.testing.assert_allclose(
        u_projector_matfree, u_projector_explicit, rtol=1e-4, atol=1e-4
    )


def test_wssr_warm_svd_right_core_matfree_matches_explicit_initial_branch():
    params = _tiny_params()
    positions = _tiny_positions()
    local_energies = jax.vmap(_local_energy_fn, in_axes=(None, 0))(params, positions)
    state = wssr.WSSRWarmSVDCoreState(
        sr_o=jnp.array(
            [
                [1.0, -2.0, 30.0, -40.0, 50.0],
                [0.5, 1.5, -50.0, 20.0, -10.0],
                [-1.0, 0.25, 70.0, -80.0, 90.0],
            ]
        ),
        ek=jnp.array([0.2, -0.1, 10.0, -20.0, 30.0]),
        sr_rank0=jnp.array(0),
        sr_rank=jnp.array(2),
        u=jnp.zeros((3, 5)),
        has_u=jnp.array(False),
    )

    _assert_wssr_warm_svd_right_core_matfree_matches_explicit(
        _log_psi_apply,
        params,
        positions,
        local_energies,
        state,
        key=jax.random.PRNGKey(31),
        eta=0.99,
        svd_working_rank=2,
        constrain_update_norm=False,
    )


def test_wssr_warm_svd_right_core_matfree_matches_explicit_warm_branch_constrained():
    params = {
        "w": jnp.array([[0.2, -0.1, 0.3], [0.7, -0.4, 0.5]]),
        "b": jnp.array([0.1, -0.2]),
        "scale": jnp.array(0.4),
    }
    positions = jnp.array(
        [
            [0.5, -1.0, 0.25],
            [1.5, 0.2, -0.75],
            [-0.25, 0.4, 1.0],
        ]
    )

    def log_psi_apply(params, position):
        hidden = jnp.tanh(params["w"] @ position + params["b"])
        return jnp.sum(hidden) + params["scale"] * jnp.prod(position)

    local_energies = jnp.array([1.0, -0.5, 0.25])
    state = wssr.WSSRWarmSVDCoreState(
        sr_o=jnp.array(
            [
                [0.5, -1.0, 10.0, 20.0, -30.0],
                [1.5, 0.25, -10.0, 40.0, 50.0],
                [-0.5, 0.75, 60.0, -70.0, 80.0],
                [0.2, -0.4, -90.0, 100.0, 110.0],
                [1.2, 0.8, 120.0, 130.0, -140.0],
                [-1.1, 0.3, 150.0, -160.0, 170.0],
                [0.9, -0.7, -180.0, 190.0, 200.0],
                [0.4, 1.4, 210.0, -220.0, 230.0],
                [-0.8, 0.6, -240.0, 250.0, -260.0],
            ]
        ),
        ek=jnp.array([0.3, -0.2, 100.0, 200.0, -300.0]),
        sr_rank0=jnp.array(2),
        sr_rank=jnp.array(3),
        u=jnp.eye(9, 5),
        has_u=jnp.array(True),
    )

    _assert_wssr_warm_svd_right_core_matfree_matches_explicit(
        log_psi_apply,
        params,
        positions,
        local_energies,
        state,
        key=jax.random.PRNGKey(32),
        eta=0.99,
        svd_working_rank=3,
        constrain_update_norm=True,
    )


def test_wssr_warm_svd_right_matches_exact_svd_on_small_rank_controlled_case():
    state = wssr.initialize_wssr_warm_svd_core_state(
        num_params=4,
        sr_rank=3,
        sr_rank_max=4,
        dtype=jnp.float32,
    )
    core_state = wssr.initialize_wssr_core_state(
        num_params=4, sr_rank=3, sr_rank_max=4, dtype=jnp.float32
    )
    o_aug = jnp.array(
        [
            [4.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    e_aug = jnp.array([0.5, -0.25, 0.125, -0.75, 0.25], dtype=jnp.float32)
    key = jax.random.PRNGKey(42)

    _, singular_values, _, rank = wssr.right_warm_start_svd(
        o_aug,
        state,
        key,
        maxiter_initial=1,
        maxiter_warm=1,
        svd_working_rank=3,
    )
    exact_singular_values = jnp.linalg.svd(o_aug, full_matrices=False)[1][:3]
    chex.assert_trees_all_close(
        singular_values[:3], exact_singular_values, rtol=1e-5, atol=1e-5
    )
    assert rank == 3

    right_result = wssr.wssr_warm_svd_right_core_update(
        o_aug,
        e_aug,
        state,
        key=key,
        damping=0.05,
        norm_constraint=10.0,
        sr_rank_max=4,
        sr_scale=1.1,
        svd_maxiter_initial=1,
        svd_maxiter_warm=1,
        svd_working_rank=3,
        constrain_update_norm=False,
    )
    exact_result = wssr.wssr_svd_core_update(
        o_aug,
        e_aug,
        core_state,
        damping=0.05,
        norm_constraint=10.0,
        sr_rank_max=4,
        sr_scale=1.1,
        constrain_update_norm=False,
    )

    chex.assert_trees_all_close(
        right_result.grad_like_update,
        exact_result.grad_like_update,
        rtol=1e-4,
        atol=1e-4,
    )
    chex.assert_trees_all_close(
        right_result.state.sr_o @ right_result.state.ek,
        exact_result.state.sr_o @ exact_result.state.ek,
        rtol=1e-5,
        atol=1e-5,
    )


def test_wssr_warm_svd_right_core_jitted_matches_eager_update_and_history():
    state = wssr.WSSRWarmSVDCoreState(
        sr_o=jnp.array(
            [
                [0.2, -0.1, 0.0, 0.0],
                [0.4, 0.3, 0.0, 0.0],
                [-0.5, 0.25, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        ),
        ek=jnp.array([0.5, -0.25, 0.0, 0.0], dtype=jnp.float32),
        sr_rank0=jnp.array(2),
        sr_rank=jnp.array(2),
        u=jnp.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=jnp.float32,
        ),
        has_u=jnp.array(True),
    )
    o_aug = jnp.array(
        [
            [2.0, 0.5, -1.0, 0.25, 0.0],
            [0.0, 1.5, 0.5, -0.75, 1.0],
            [1.0, -0.25, 1.25, 0.5, -0.5],
        ],
        dtype=jnp.float32,
    )
    e_aug = jnp.array([0.5, -0.25, 0.125, -0.75, 0.25], dtype=jnp.float32)
    key = jax.random.PRNGKey(123)
    kwargs = dict(
        damping=0.05,
        norm_constraint=10.0,
        sr_rank_max=4,
        sr_scale=1.5,
        svd_maxiter_initial=2,
        svd_maxiter_warm=1,
        svd_working_rank=3,
        constrain_update_norm=False,
    )

    eager = wssr.wssr_warm_svd_right_core_update(
        o_aug, e_aug, state, key=key, **kwargs
    )
    jitted_update = jax.jit(
        wssr.wssr_warm_svd_right_core_update,
        static_argnames=(
            "sr_rank_max",
            "svd_maxiter_initial",
            "svd_maxiter_warm",
            "svd_working_rank",
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
    np.testing.assert_allclose(eager.state.u[:, 3:], 0.0, atol=1e-6)
    np.testing.assert_allclose(jitted.state.u[:, 3:], 0.0, atol=1e-6)
    assert not hasattr(eager.state, "v")
    assert jnp.all(jnp.isfinite(eager.state.u))
    assert jnp.all(jnp.isfinite(jitted.state.u))


def test_wssr_warm_svd_right_exact_first_matches_exact_reference_and_is_once():
    """Exact-first matches rank-r SVD, stores it, then takes the warm branch."""
    key = jax.random.PRNGKey(713)
    o_aug = jax.random.normal(key, (9, 7), dtype=jnp.float32)
    e_aug = jax.random.normal(jax.random.fold_in(key, 1), (7,), dtype=jnp.float32)
    warm_state = wssr.initialize_wssr_warm_svd_core_state(
        9, 3, 3, dtype=jnp.float32, store_warm_u=True
    )
    exact_state = wssr.WSSRCoreState(
        warm_state.sr_o,
        warm_state.ek,
        warm_state.sr_rank0,
        warm_state.sr_rank,
    )
    kwargs = dict(
        damping=3e-4,
        norm_constraint=1e-3,
        sr_rank_max=3,
        constrain_update_norm=False,
        spectral_regularization="tikhonov",
        complement_weight=0.0,
    )
    exact_first = wssr.wssr_warm_svd_right_core_update(
        o_aug,
        e_aug,
        warm_state,
        jax.random.fold_in(key, 2),
        svd_maxiter_initial=2,
        svd_maxiter_warm=1,
        svd_working_rank=3,
        exact_first=True,
        **kwargs,
    )
    ref_u, ref_s, ref_vh = jnp.linalg.svd(o_aug, full_matrices=False)
    reference = wssr._wssr_update_from_svd(
        o_aug,
        e_aug,
        exact_state,
        ref_u[:, :3],
        ref_s[:3],
        ref_vh[:3, :],
        rank_update_max=3,
        sr_scale=1.1,
        **kwargs,
    )
    chex.assert_trees_all_close(
        exact_first.grad_like_update,
        reference.grad_like_update,
        rtol=2e-5,
        atol=2e-6,
    )

    exact_u = jnp.linalg.svd(o_aug, full_matrices=False)[0][:, :3]
    chex.assert_trees_all_close(
        exact_first.state.u @ exact_first.state.u.T,
        exact_u @ exact_u.T,
        rtol=2e-5,
        atol=2e-5,
    )
    assert bool(exact_first.state.has_u)

    o_aug_2 = o_aug + 0.01 * jax.random.normal(
        jax.random.fold_in(key, 3), o_aug.shape, dtype=o_aug.dtype
    )
    e_aug_2 = e_aug + 0.01 * jax.random.normal(
        jax.random.fold_in(key, 4), e_aug.shape, dtype=e_aug.dtype
    )
    common = dict(
        o_aug=o_aug_2,
        e_aug=e_aug_2,
        state=exact_first.state,
        key=jax.random.fold_in(key, 5),
        svd_maxiter_initial=2,
        svd_maxiter_warm=1,
        svd_working_rank=3,
        **kwargs,
    )
    second_exact_flag = wssr.wssr_warm_svd_right_core_update(
        exact_first=True, **common
    )
    second_normal = wssr.wssr_warm_svd_right_core_update(
        exact_first=False, **common
    )
    chex.assert_trees_all_equal(second_exact_flag, second_normal)

    diagnostic, diagnostic_metrics = (
        wssr.wssr_warm_svd_right_reference_diagnostic_core_update(
            exact_first=False, **common
        )
    )
    chex.assert_trees_all_equal(diagnostic, second_normal)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in diagnostic_metrics.values())


def test_wssr_warm_svd_right_rank_growth_caps_to_working_width_and_stores_history():
    state = wssr.initialize_wssr_warm_svd_core_state(
        num_params=4,
        sr_rank=5,
        sr_rank_max=6,
        dtype=jnp.float32,
    )
    o_aug = jnp.array(
        [
            [5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    e_aug = jnp.array([0.5, -0.25, 0.125, -0.75, 0.25, 0.1], dtype=jnp.float32)

    result = wssr.wssr_warm_svd_right_core_update(
        o_aug,
        e_aug,
        state,
        key=jax.random.PRNGKey(7),
        damping=1e-4,
        norm_constraint=10.0,
        sr_rank_max=6,
        sr_scale=2.0,
        svd_maxiter_initial=2,
        svd_maxiter_warm=1,
        svd_working_rank=3,
        constrain_update_norm=False,
    )

    _assert_wssr_state_all_finite(result.state)
    assert result.state.u.shape == (4, 6)
    assert result.state.has_u == jnp.array(True)
    assert result.state.sr_rank <= 3
    assert result.state.sr_rank0 <= 3
    assert jnp.any(result.state.u[:, :3] != 0.0)
    assert jnp.all(result.state.u[:, 3:] == 0.0)
    assert jnp.all(result.state.sr_o[:, 3:] == 0.0)
    assert jnp.all(result.state.ek[3:] == 0.0)


def test_wssr_warm_svd_right_thin_storage_matches_zero_padded_full_storage():
    num_params = 8
    storage_rank = 7
    sr_rank_max = 20
    num_samples = 6
    dtype = jnp.float32

    history = (
        jnp.arange(num_params * storage_rank, dtype=dtype)
        .reshape(num_params, storage_rank)
        / 50.0
    )
    ek = jnp.linspace(-0.2, 0.25, storage_rank, dtype=dtype)
    u_basis = jnp.eye(num_params, storage_rank, dtype=dtype)
    o_cur = (
        jnp.arange(num_params * num_samples, dtype=dtype)
        .reshape(num_params, num_samples)
        / 30.0
    )
    e_cur = jnp.linspace(-0.4, 0.5, num_samples, dtype=dtype)

    full_state = wssr.initialize_wssr_warm_svd_core_state(
        num_params=num_params,
        sr_rank=storage_rank,
        sr_rank_max=sr_rank_max,
        dtype=dtype,
    )
    full_state = full_state._replace(
        sr_o=full_state.sr_o.at[:, :storage_rank].set(history),
        ek=full_state.ek.at[:storage_rank].set(ek),
        sr_rank0=jnp.array(storage_rank),
        u=full_state.u.at[:, :storage_rank].set(u_basis),
        has_u=jnp.array(True),
    )
    thin_state = wssr.initialize_wssr_warm_svd_core_state(
        num_params=num_params,
        sr_rank=storage_rank,
        sr_rank_max=storage_rank,
        dtype=dtype,
    )
    thin_state = thin_state._replace(
        sr_o=history,
        ek=ek,
        sr_rank0=jnp.array(storage_rank),
        u=u_basis,
        has_u=jnp.array(True),
    )

    eta = 0.99
    o_aug_full, e_aug_full = wssr.augment_wssr_system(
        o_cur, e_cur, full_state, eta
    )
    o_aug_thin, e_aug_thin = wssr.augment_wssr_system(
        o_cur, e_cur, thin_state, eta
    )
    kwargs = dict(
        key=jax.random.PRNGKey(17),
        damping=0.02,
        norm_constraint=10.0,
        sr_rank_max=sr_rank_max,
        sr_scale=1.1,
        svd_maxiter_initial=2,
        svd_maxiter_warm=2,
        svd_working_rank=storage_rank,
        constrain_update_norm=False,
    )

    full_result = wssr.wssr_warm_svd_right_core_update(
        o_aug_full, e_aug_full, full_state, **kwargs
    )
    thin_result = wssr.wssr_warm_svd_right_core_update(
        o_aug_thin, e_aug_thin, thin_state, **kwargs
    )

    np.testing.assert_allclose(
        thin_result.grad_like_update,
        full_result.grad_like_update,
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        thin_result.state.sr_o,
        full_result.state.sr_o[:, :storage_rank],
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        thin_result.state.ek,
        full_result.state.ek[:storage_rank],
        rtol=1e-4,
        atol=1e-4,
    )
    chex.assert_trees_all_close(thin_result.active_rank, full_result.active_rank)
    chex.assert_trees_all_close(thin_result.state.sr_rank, full_result.state.sr_rank)
    chex.assert_trees_all_close(
        thin_result.state.sr_rank0, full_result.state.sr_rank0
    )
    assert thin_result.state.sr_o.shape[1] == storage_rank
    assert full_result.state.sr_o.shape[1] == sr_rank_max
    assert jnp.all(full_result.state.sr_o[:, storage_rank:] == 0.0)
    assert jnp.all(full_result.state.ek[storage_rank:] == 0.0)


def test_initialize_optimizer_dispatches_wssr_warm_svd_right_and_constructs_state():
    params = _tiny_params()
    data = _tiny_positions()
    config = default_config.get_default_config()
    config.vmc.nchains = data.shape[0]
    config.vmc.optimizer_type = "wssr_warm_svd_right"
    config.vmc.optimizer.wssr_warm_svd_right.sr_rank = 2
    config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max = 5
    config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank = 3

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
        config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max,
    )
    assert optimizer_state.core_state.sr_o.shape == (
        3,
        config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max,
    )
    assert not hasattr(optimizer_state.core_state, "v")
    assert optimizer_state.core_state.has_u == jnp.array(False)
    assert key.shape == (2,)


def test_wssr_warm_svd_right_native_proximal_uses_previous_direction_state():
    params = _tiny_params()
    data = _tiny_positions()
    config = default_config.get_default_config()
    config.vmc.nchains = data.shape[0]
    config.vmc.optimizer_type = "wssr_warm_svd_right"
    opt = config.vmc.optimizer.wssr_warm_svd_right
    opt.schedule_type = "constant"
    opt.constrain_norm = False
    opt.sr_rank = 2
    opt.sr_rank_max = 5
    opt.svd_working_rank = 3
    opt.svd_maxiter_initial = 2
    opt.svd_maxiter_warm = 1
    opt.spectral_regularization = "tikhonov"
    opt.complement_weight = 0.0
    opt.experimental_mode = "native_proximal"
    opt.native_proximal_gamma = 3e-4

    update_param_fn, state, key = initialize_optimizer(
        _log_psi_apply,
        _local_energy_fn,
        None,
        config.vmc,
        params,
        data,
        lambda x: x,
        lambda d, p: d,
        jax.random.PRNGKey(17),
        apply_pmap=False,
    )
    assert isinstance(state, wssr.WSSRComplementEMAOptimizerState)
    assert jnp.all(state.complement_ema == 0)

    params, data, state, metrics, key = update_param_fn(
        params, data, state, key
    )
    first_direction = state.complement_ema
    assert jnp.all(jnp.isfinite(first_direction))
    assert jnp.linalg.norm(first_direction) > 0
    assert metrics["wssr_diag_native_proximal_gamma"] == pytest.approx(3e-4)

    _, _, state, metrics, _ = update_param_fn(params, data, state, key)
    assert jnp.all(jnp.isfinite(state.complement_ema))
    assert jnp.linalg.norm(state.complement_ema - first_direction) > 0
    assert jnp.isfinite(metrics["wssr_diag_native_proximal_previous_cosine"])


def test_initialize_wssr_warm_svd_right_uses_thin_storage_rank_when_set():
    params = _tiny_params()
    data = _tiny_positions()
    config = default_config.get_default_config()
    config.vmc.nchains = data.shape[0]
    config.vmc.optimizer_type = "wssr_warm_svd_right"
    config.vmc.optimizer.wssr_warm_svd_right.sr_rank = 4
    config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max = 20
    config.vmc.optimizer.wssr_warm_svd_right.sr_storage_rank = 7
    config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank = 7

    _, optimizer_state, _ = initialize_optimizer(
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

    assert optimizer_state.core_state.sr_o.shape == (3, 7)
    assert optimizer_state.core_state.ek.shape == (7,)
    assert optimizer_state.core_state.u.shape == (3, 7)
    assert optimizer_state.core_state.sr_rank == jnp.asarray(4)


def test_initialize_wssr_warm_svd_right_rejects_storage_rank_below_initial_rank():
    params = _tiny_params()
    data = _tiny_positions()
    config = default_config.get_default_config()
    config.vmc.nchains = data.shape[0]
    config.vmc.optimizer_type = "wssr_warm_svd_right"
    config.vmc.optimizer.wssr_warm_svd_right.sr_rank = 10
    config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max = 20
    config.vmc.optimizer.wssr_warm_svd_right.sr_storage_rank = 5

    with pytest.raises(ValueError, match="sr_storage_rank"):
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


def test_wssr_function_constraint_applies_euclidean_safety_afterward():
    params = _tiny_params()
    data = _tiny_positions()
    config = default_config.get_default_config()
    config.vmc.nchains = data.shape[0]
    config.vmc.optimizer_type = "wssr_warm_svd_right"
    opt = config.vmc.optimizer.wssr_warm_svd_right
    opt.schedule_type = "constant"
    opt.learning_rate = 1.0
    opt.sr_rank = 2
    opt.sr_rank_max = 3
    opt.sr_storage_rank = 3
    opt.svd_working_rank = 3
    opt.svd_maxiter_initial = 2
    opt.svd_maxiter_warm = 1
    opt.norm_constraint_mode = "function_space"
    opt.function_norm_constraint = 1e6
    opt.euclidean_safety_constraint = 1e-8

    update_param_fn, state, key = initialize_optimizer(
        _log_psi_apply,
        _local_energy_fn,
        None,
        config.vmc,
        params,
        data,
        lambda x: x,
        lambda d, p: d,
        jax.random.PRNGKey(23),
        apply_pmap=False,
    )
    new_params, _, _, metrics, _ = update_param_fn(
        params, data, state, key
    )
    displacement = jax.tree_util.tree_map(
        lambda new, old: new - old, new_params, params
    )

    assert float(wssr.tree_l2_norm(displacement)) <= 1.0001e-4
    assert bool(metrics["wssr_euclidean_safety_active"])
    assert float(metrics["wssr_euclidean_safety_scale"]) < 1.0


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("spectral_regularization", "bad_mode", "spectral_regularization"),
        ("complement_weight", -0.1, "complement_weight"),
    ],
)
def test_initialize_wssr_warm_svd_right_rejects_invalid_regularization_config(
    field, value, match
):
    params = _tiny_params()
    data = _tiny_positions()
    config = default_config.get_default_config()
    config.vmc.nchains = data.shape[0]
    config.vmc.optimizer_type = "wssr_warm_svd_right"
    setattr(config.vmc.optimizer.wssr_warm_svd_right, field, value)

    with pytest.raises(ValueError, match=match):
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


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("euclidean_safety_constraint", 0.0, "euclidean_safety_constraint"),
        (
            "adaptive_complement_beta_function",
            -0.1,
            "adaptive_complement_beta_function",
        ),
    ],
)
def test_initialize_wssr_rejects_invalid_dual_cap_config(field, value, match):
    params = _tiny_params()
    data = _tiny_positions()
    config = default_config.get_default_config()
    config.vmc.nchains = data.shape[0]
    config.vmc.optimizer_type = "wssr_warm_svd_right"
    setattr(config.vmc.optimizer.wssr_warm_svd_right, field, value)

    with pytest.raises(ValueError, match=match):
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


def test_wssr_warm_svd_right_update_returns_rank_diagnostics():
    params = _tiny_params()
    data = _tiny_positions()
    config = default_config.get_default_config()
    config.vmc.nchains = data.shape[0]
    config.vmc.optimizer_type = "wssr_warm_svd_right"
    config.vmc.optimizer.wssr_warm_svd_right.schedule_type = "constant"
    config.vmc.optimizer.wssr_warm_svd_right.constrain_norm = False
    config.vmc.optimizer.wssr_warm_svd_right.sr_rank = 2
    config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max = 5
    config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank = 3
    config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_initial = 2
    config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm = 1

    update_param_fn, optimizer_state, key = initialize_optimizer(
        _log_psi_apply,
        _local_energy_fn,
        None,
        config.vmc,
        params,
        data,
        lambda x: x,
        lambda d, p: d,
        jax.random.PRNGKey(5),
        apply_pmap=False,
    )
    _, _, _, metrics, _ = update_param_fn(params, data, optimizer_state, key)

    _assert_wssr_rank_metrics(
        metrics,
        sr_rank_max=config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max,
        svd_working_rank=config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank,
    )


def test_wssr_long_history_candidate_uses_eta_ramp_and_finite_mixed_solve():
    params = _tiny_params()
    data = _tiny_positions()
    config = default_config.get_default_config()
    config.vmc.nchains = data.shape[0]
    config.vmc.optimizer_type = "wssr_warm_svd_right"
    opt = config.vmc.optimizer.wssr_warm_svd_right
    opt.constrain_norm = False
    opt.eta = 0.99
    opt.eta_bias_correction = True
    opt.mixed_precision_solve = True
    opt.sr_rank = 2
    opt.sr_rank_max = 3
    opt.sr_storage_rank = 3
    opt.svd_working_rank = 3
    opt.svd_maxiter_initial = 2
    opt.svd_maxiter_warm = 1
    opt.spectral_regularization = "tikhonov"
    opt.complement_weight = 0.0
    opt.relative_singular_value_cutoff = 0.1
    opt.tikhonov_lambda = 0.5

    update_param_fn, state, key = initialize_optimizer(
        _log_psi_apply,
        _local_energy_fn,
        None,
        config.vmc,
        params,
        data,
        lambda x: x,
        lambda d, p: d,
        jax.random.PRNGKey(29),
        apply_pmap=False,
    )
    params, data, state, first_metrics, key = update_param_fn(
        params, data, state, key
    )
    assert first_metrics["wssr_eta_used"] == pytest.approx(0.0)
    assert bool(first_metrics["wssr_diag_finite"])
    assert jnp.isfinite(first_metrics["wssr_diag_raw_direction_norm"])

    _, _, _, second_metrics, _ = update_param_fn(params, data, state, key)
    assert second_metrics["wssr_eta_used"] == pytest.approx(0.5)
    assert bool(second_metrics["wssr_diag_finite"])
    assert jnp.isfinite(second_metrics["wssr_diag_raw_direction_norm"])


@pytest.mark.parametrize("mode", ["naive", "residual"])
def test_wssr_solution_recurrence_tracks_full_parameter_history(mode):
    params = _tiny_params()
    data = _tiny_positions()
    config = default_config.get_default_config()
    config.vmc.nchains = data.shape[0]
    config.vmc.optimizer_type = "wssr_warm_svd_right"
    opt = config.vmc.optimizer.wssr_warm_svd_right
    opt.schedule_type = "constant"
    opt.constrain_norm = False
    opt.eta = 0.0
    opt.mixed_precision_solve = True
    opt.solution_recurrence_mode = mode
    opt.solution_recurrence_mu = 0.99
    opt.sr_rank = 2
    opt.sr_rank_max = 3
    opt.sr_storage_rank = 3
    opt.svd_working_rank = 3
    opt.svd_maxiter_initial = 2
    opt.svd_maxiter_warm = 1
    opt.spectral_regularization = "tikhonov"
    opt.complement_weight = 0.0
    opt.relative_singular_value_cutoff = 0.1
    opt.tikhonov_lambda = 0.5

    update_param_fn, state, key = initialize_optimizer(
        _log_psi_apply,
        _local_energy_fn,
        None,
        config.vmc,
        params,
        data,
        lambda x: x,
        lambda d, p: d,
        jax.random.PRNGKey(37),
        apply_pmap=False,
    )
    assert isinstance(state, wssr.WSSRSolutionRecurrenceOptimizerState)
    assert jnp.linalg.norm(state.solution_state) == pytest.approx(0.0)

    params, data, state, first_metrics, key = update_param_fn(
        params, data, state, key
    )
    assert bool(first_metrics["wssr_diag_finite"])
    assert first_metrics["wssr_solution_prior_norm"] == pytest.approx(0.0)
    assert jnp.linalg.norm(state.solution_state) > 0.0

    _, _, next_state, second_metrics, _ = update_param_fn(
        params, data, state, key
    )
    assert bool(second_metrics["wssr_diag_finite"])
    assert second_metrics["wssr_solution_prior_norm"] > 0.0
    assert second_metrics["wssr_solution_correction_norm"] > 0.0
    _assert_tree_all_finite(next_state)


def test_full_current_batch_device_cholesky_runs_through_optimizer():
    """The configured GPU-direct backend reaches the production update path."""
    params = _tiny_params()
    data = _tiny_positions()
    config = default_config.get_default_config()
    config.vmc.nchains = data.shape[0]
    config.vmc.optimizer_type = "wssr_warm_svd_right"
    opt = config.vmc.optimizer.wssr_warm_svd_right
    opt.schedule_type = "constant"
    opt.constrain_norm = False
    opt.eta = 0.0
    opt.mixed_precision_solve = True
    opt.solution_recurrence_mode = "residual"
    opt.solution_recurrence_mu = 0.99
    opt.residual_evaluation = "full_current_batch"
    opt.galerkin_solve_backend = "device_cholesky"
    opt.sr_rank = 2
    opt.sr_rank_max = 3
    opt.sr_storage_rank = 3
    opt.svd_working_rank = 3
    opt.svd_maxiter_initial = 2
    opt.svd_maxiter_warm = 1
    opt.spectral_regularization = "tikhonov"
    opt.complement_weight = 0.0
    opt.relative_singular_value_cutoff = 0.1
    opt.tikhonov_lambda = 0.5

    update_param_fn, state, key = initialize_optimizer(
        _log_psi_apply,
        _local_energy_fn,
        None,
        config.vmc,
        params,
        data,
        lambda x: x,
        lambda d, p: d,
        jax.random.PRNGKey(107),
        apply_pmap=False,
    )
    params, data, state, first_metrics, key = update_param_fn(
        params, data, state, key
    )
    _, _, next_state, second_metrics, _ = update_param_fn(
        params, data, state, key
    )

    assert bool(first_metrics["wssr_diag_finite"])
    assert bool(second_metrics["wssr_diag_finite"])
    assert second_metrics["wssr_solution_prior_norm"] > 0.0
    assert second_metrics["wssr_solution_correction_norm"] > 0.0
    _assert_tree_all_finite(next_state)


def test_wssr_multilevel_complement_uses_two_buffers_and_resets_at_boundary():
    params = _tiny_params()
    data = _tiny_positions()
    config = default_config.get_default_config()
    config.vmc.nchains = data.shape[0]
    config.vmc.optimizer_type = "wssr_warm_svd_right"
    opt = config.vmc.optimizer.wssr_warm_svd_right
    opt.schedule_type = "constant"
    opt.constrain_norm = False
    opt.sr_rank = 2
    opt.sr_rank_max = 3
    opt.sr_storage_rank = 3
    opt.svd_working_rank = 3
    opt.svd_maxiter_initial = 2
    opt.svd_maxiter_warm = 1
    opt.experimental_mode = "adaptive_complement"
    opt.experimental_target_rank = 2
    opt.adaptive_complement_beta = 0.001
    opt.complement_weight = 0.0001
    opt.complement_state_decay = 0.99
    opt.multilevel_complement_period = 2
    opt.multilevel_complement_cosine_threshold = 0.0

    update_param_fn, state, key = initialize_optimizer(
        _log_psi_apply,
        _local_energy_fn,
        None,
        config.vmc,
        params,
        data,
        lambda x: x,
        lambda d, p: d,
        jax.random.PRNGKey(5),
        apply_pmap=False,
    )
    assert isinstance(state, wssr.WSSRMultilevelComplementOptimizerState)

    params, data, state, metrics, key = update_param_fn(
        params, data, state, key
    )
    assert int(state.complement_step) == 1
    assert int(metrics["wssr_diag_multilevel_boundary"]) == 0

    _, _, state, metrics, _ = update_param_fn(params, data, state, key)
    assert int(state.complement_step) == 2
    assert int(metrics["wssr_diag_multilevel_boundary"]) == 1
    assert jnp.all(state.complement_ema == 0)
    assert jnp.all(state.complement_ema_b == 0)
    assert state.complement_weight_a == 0
    assert state.complement_weight_b == 0


def test_initialize_optimizer_dispatches_wssr_warm_svd_right_matfree_and_constructs_state():
    params = _tiny_params()
    data = _tiny_positions()
    config = default_config.get_default_config()
    config.vmc.nchains = data.shape[0]
    config.vmc.optimizer_type = "wssr_warm_svd_right_matfree"
    config.vmc.optimizer.wssr_warm_svd_right_matfree.sr_rank = 2
    config.vmc.optimizer.wssr_warm_svd_right_matfree.sr_rank_max = 5
    config.vmc.optimizer.wssr_warm_svd_right_matfree.svd_working_rank = 3

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
        config.vmc.optimizer.wssr_warm_svd_right_matfree.sr_rank_max,
    )
    assert optimizer_state.core_state.sr_o.shape == (
        3,
        config.vmc.optimizer.wssr_warm_svd_right_matfree.sr_rank_max,
    )
    assert not hasattr(optimizer_state.core_state, "v")
    assert optimizer_state.core_state.has_u == jnp.array(False)
    assert key.shape == (2,)


def test_wssr_warm_svd_right_matfree_one_update_matches_explicit_right_path():
    params = _tiny_params()
    data = _tiny_positions()
    explicit_config = default_config.get_default_config()
    explicit_config.vmc.nchains = data.shape[0]
    explicit_config.vmc.optimizer_type = "wssr_warm_svd_right"
    explicit_config.vmc.optimizer.wssr_warm_svd_right.schedule_type = "constant"
    explicit_config.vmc.optimizer.wssr_warm_svd_right.learning_rate = 0.125
    explicit_config.vmc.optimizer.wssr_warm_svd_right.constrain_norm = False
    explicit_config.vmc.optimizer.wssr_warm_svd_right.sr_rank = 2
    explicit_config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max = 5
    explicit_config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank = 3
    explicit_config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_initial = 2
    explicit_config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm = 1

    matfree_config = default_config.get_default_config()
    matfree_config.vmc.nchains = data.shape[0]
    matfree_config.vmc.optimizer_type = "wssr_warm_svd_right_matfree"
    matfree_config.vmc.optimizer.wssr_warm_svd_right_matfree.schedule_type = (
        explicit_config.vmc.optimizer.wssr_warm_svd_right.schedule_type
    )
    matfree_config.vmc.optimizer.wssr_warm_svd_right_matfree.learning_rate = (
        explicit_config.vmc.optimizer.wssr_warm_svd_right.learning_rate
    )
    matfree_config.vmc.optimizer.wssr_warm_svd_right_matfree.constrain_norm = (
        explicit_config.vmc.optimizer.wssr_warm_svd_right.constrain_norm
    )
    matfree_config.vmc.optimizer.wssr_warm_svd_right_matfree.sr_rank = (
        explicit_config.vmc.optimizer.wssr_warm_svd_right.sr_rank
    )
    matfree_config.vmc.optimizer.wssr_warm_svd_right_matfree.sr_rank_max = (
        explicit_config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max
    )
    matfree_config.vmc.optimizer.wssr_warm_svd_right_matfree.svd_working_rank = (
        explicit_config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank
    )
    matfree_config.vmc.optimizer.wssr_warm_svd_right_matfree.svd_maxiter_initial = (
        explicit_config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_initial
    )
    matfree_config.vmc.optimizer.wssr_warm_svd_right_matfree.svd_maxiter_warm = (
        explicit_config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm
    )

    key = jax.random.PRNGKey(8)
    explicit_update_fn, explicit_state, explicit_key = initialize_optimizer(
        _log_psi_apply,
        _local_energy_fn,
        None,
        explicit_config.vmc,
        params,
        data,
        lambda x: x,
        lambda d, p: d,
        key,
        apply_pmap=False,
    )
    matfree_update_fn, matfree_state, matfree_key = initialize_optimizer(
        _log_psi_apply,
        _local_energy_fn,
        None,
        matfree_config.vmc,
        params,
        data,
        lambda x: x,
        lambda d, p: d,
        key,
        apply_pmap=False,
    )

    explicit_params, _, explicit_state_1, explicit_metrics, explicit_key_1 = (
        explicit_update_fn(params, data, explicit_state, explicit_key)
    )
    matfree_params, _, matfree_state_1, matfree_metrics, matfree_key_1 = (
        matfree_update_fn(params, data, matfree_state, matfree_key)
    )

    assert_pytree_allclose(matfree_params, explicit_params, rtol=2e-4, atol=2e-4)
    assert set(matfree_metrics).issuperset({"energy", "variance", "energy_noclip"})
    assert set(explicit_metrics).issuperset({"energy", "variance", "energy_noclip"})
    chex.assert_trees_all_close(
        matfree_state_1.core_state.sr_o,
        explicit_state_1.core_state.sr_o,
        rtol=1e-5,
        atol=1e-5,
    )
    chex.assert_trees_all_close(
        matfree_state_1.core_state.ek,
        explicit_state_1.core_state.ek,
        rtol=1e-5,
        atol=1e-5,
    )
    chex.assert_trees_all_close(
        matfree_state_1.core_state.sr_rank0, explicit_state_1.core_state.sr_rank0
    )
    chex.assert_trees_all_close(
        matfree_state_1.core_state.sr_rank, explicit_state_1.core_state.sr_rank
    )
    assert matfree_state_1.core_state.u.shape == explicit_state_1.core_state.u.shape
    assert not hasattr(matfree_state_1.core_state, "v")
    chex.assert_trees_all_close(matfree_key_1, explicit_key_1)


def test_wssr_warm_svd_right_integrated_two_updates_have_no_nans():
    params = _tiny_params()
    data = _tiny_positions()
    config = default_config.get_default_config()
    config.vmc.nchains = data.shape[0]
    config.vmc.optimizer_type = "wssr_warm_svd_right"
    config.vmc.optimizer.wssr_warm_svd_right.schedule_type = "constant"
    config.vmc.optimizer.wssr_warm_svd_right.learning_rate = 0.125
    config.vmc.optimizer.wssr_warm_svd_right.constrain_norm = False
    config.vmc.optimizer.wssr_warm_svd_right.sr_rank = 2
    config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max = 5
    config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank = 3
    config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_initial = 2
    config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm = 1

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
    assert state_2.core_state.u.shape == (
        3,
        config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max,
    )
    assert not hasattr(state_2.core_state, "v")
    assert set(metrics_1).issuperset({"energy", "variance", "energy_noclip"})
    assert set(metrics_2).issuperset({"energy", "variance", "energy_noclip"})
    assert jnp.all(jnp.isfinite(jnp.asarray(list(metrics_1.values()))))
    assert jnp.all(jnp.isfinite(jnp.asarray(list(metrics_2.values()))))
    np.testing.assert_allclose(data_2, data)
    assert key_1.shape == key.shape
    assert key_2.shape == key.shape
    assert not np.allclose(key_1, key)
    assert not np.allclose(key_2, key_1)


@pytest.mark.parametrize(
    ("experimental_mode", "curvature_mode"),
    [
        ("cluster_envelope", "scalar"),
        ("cluster_envelope", "rayleigh_ritz"),
        ("cluster_envelope_ritz", "scalar"),
        ("cluster_envelope_ritz_ef", "scalar"),
        ("cluster_envelope_ritz_selected", "scalar"),
        ("cluster_envelope_ritz_snr", "scalar"),
    ],
)
def test_cluster_envelope_integrated_two_updates_expand_state_and_remain_finite(
    experimental_mode,
    curvature_mode,
):
    params = _tiny_params()
    data = _tiny_positions()
    config = default_config.get_default_config()
    config.vmc.nchains = data.shape[0]
    config.vmc.optimizer_type = "wssr_warm_svd_right"
    opt = config.vmc.optimizer.wssr_warm_svd_right
    opt.schedule_type = "constant"
    opt.learning_rate = 0.05
    opt.constrain_norm = True
    opt.eta = 0.0
    opt.eta_S = 0.0
    opt.eta_g = 0.0
    opt.sr_rank = 2
    opt.sr_rank_max = 2
    opt.sr_storage_rank = 2
    opt.svd_working_rank = 2
    opt.svd_maxiter_initial = 2
    opt.svd_maxiter_warm = 1
    opt.spectral_regularization = "tikhonov"
    opt.tikhonov_lambda = 0.1
    opt.complement_weight = 0.0
    opt.experimental_mode = experimental_mode
    opt.cluster_envelope_rank = 2
    opt.cluster_envelope_history = 3
    if experimental_mode == "cluster_envelope_ritz_selected":
        opt.cluster_envelope_capacity = 3
    opt.cluster_envelope_alpha = 0.2
    opt.cluster_envelope_gamma = -1.0
    opt.cluster_envelope_curvature_mode = curvature_mode

    update_param_fn, state_0, key = initialize_optimizer(
        _log_psi_apply,
        _local_energy_fn,
        None,
        config.vmc,
        params,
        data,
        lambda x: x,
        lambda d, p: d,
        jax.random.PRNGKey(81),
        apply_pmap=False,
    )
    if experimental_mode == "cluster_envelope_ritz_ef":
        assert isinstance(state_0, wssr.WSSRClusterEnvelopeEFOptimizerState)
    else:
        assert isinstance(state_0, wssr.WSSRClusterEnvelopeOptimizerState)
    assert state_0.envelope_history.shape == (3, 4)

    params_1, data_1, state_1, metrics_1, key_1 = update_param_fn(
        params, data, state_0, key
    )
    params_2, _, state_2, metrics_2, _ = update_param_fn(
        params_1, data_1, state_1, key_1
    )

    _assert_tree_all_finite(params_2)
    assert state_1.envelope_count == 1
    assert state_2.envelope_count == 2
    assert jnp.any(state_2.envelope_history != 0.0)
    if experimental_mode == "cluster_envelope_ritz_ef":
        assert jnp.linalg.norm(state_2.error_feedback_state) > 0.0
        assert "wssr_envelope_ef_clip_count" in metrics_2
    for metrics in (metrics_1, metrics_2):
        assert metrics["wssr_diag_finite"] == 1
        assert "wssr_envelope_numerical_rank" in metrics
        assert jnp.all(jnp.isfinite(jnp.asarray(list(metrics.values()))))
    if experimental_mode == "cluster_envelope_ritz_selected":
        assert metrics_2["wssr_envelope_capacity"] == 3
        assert metrics_2["wssr_envelope_selected_history_count"] == 1
    if experimental_mode == "cluster_envelope_ritz_snr":
        assert 0.0 <= metrics_2["wssr_envelope_snr_weight_min"] <= 1.0
        assert 0.0 <= metrics_2["wssr_envelope_snr_weight_max"] <= 1.0
        assert metrics_2["wssr_envelope_snr_cluster_count"] >= 1


def test_wssr_transported_gradient_uses_actual_constrained_parameter_delta():
    params = _tiny_params()
    data = _tiny_positions()
    config = default_config.get_default_config()
    config.vmc.nchains = data.shape[0]
    config.vmc.optimizer_type = "wssr_warm_svd_right"
    opt = config.vmc.optimizer.wssr_warm_svd_right
    opt.schedule_type = "constant"
    opt.learning_rate = 10.0
    opt.constrain_norm = True
    opt.norm_constraint = 0.01
    opt.eta = 0.8
    opt.enable_gradient_transport = True
    opt.sr_rank = 2
    opt.sr_rank_max = 3
    opt.sr_storage_rank = 3
    opt.svd_working_rank = 3
    opt.svd_maxiter_initial = 2
    opt.svd_maxiter_warm = 1
    opt.spectral_regularization = "tikhonov"
    opt.complement_weight = 0.0
    opt.relative_singular_value_cutoff = 0.0
    opt.tikhonov_lambda = 0.1

    update_param_fn, state_0, key = initialize_optimizer(
        _log_psi_apply,
        _local_energy_fn,
        None,
        config.vmc,
        params,
        data,
        lambda x: x,
        lambda d, p: d,
        jax.random.PRNGKey(73),
        apply_pmap=False,
    )
    assert isinstance(state_0, wssr.WSSRTransportedGradientOptimizerState)
    flat_params_0, _ = jax.flatten_util.ravel_pytree(params)
    np.testing.assert_allclose(state_0.previous_theta, flat_params_0)
    assert not bool(state_0.transport_initialized)

    params_1, data_1, state_1, metrics_1, key_1 = update_param_fn(
        params, data, state_0, key
    )
    flat_params_1, _ = jax.flatten_util.ravel_pytree(params_1)
    actual_delta = flat_params_1 - flat_params_0
    np.testing.assert_allclose(state_1.previous_theta, flat_params_0)
    assert bool(state_1.transport_initialized)
    assert metrics_1["transport_norm"] == pytest.approx(0.0)
    assert jnp.linalg.norm(actual_delta) == pytest.approx(
        jnp.sqrt(opt.norm_constraint), rel=1e-5
    )

    local_energies_1 = jax.vmap(_local_energy_fn, in_axes=(None, 0))(
        params_1, data_1
    )
    energy_1, _, _ = physics.core.get_clipped_energies_and_stats(
        local_energies_1,
        config.vmc.nchains,
        None,
        config.vmc.nan_safe,
    )
    o_cur_1, _ = wssr.center_and_scale_score_matrix(
        _log_psi_apply, params_1, data_1
    )
    e_cur_1 = wssr.center_and_scale_energy_residuals(
        local_energies_1, energy_1
    )
    current_gradient_1 = o_cur_1 @ e_cur_1
    operator_delta = wssr.apply_wssr_history_operator(
        state_1.core_state, actual_delta
    )
    current_operator_delta = o_cur_1 @ (o_cur_1.T @ actual_delta)
    expected_transport = (
        opt.eta * (state_1.transported_gradient + operator_delta)
        + (1.0 - opt.eta) * current_gradient_1
    )

    _, _, state_2, metrics_2, _ = update_param_fn(
        params_1, data_1, state_1, key_1
    )
    np.testing.assert_allclose(
        state_2.transported_gradient,
        expected_transport,
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        metrics_2["transport_norm"], jnp.linalg.norm(operator_delta), rtol=1e-5
    )
    np.testing.assert_allclose(
        metrics_2["gradient_norm"], jnp.linalg.norm(current_gradient_1), rtol=1e-5
    )
    np.testing.assert_allclose(
        metrics_2["transport_ratio"],
        jnp.linalg.norm(operator_delta) / jnp.linalg.norm(current_gradient_1),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        metrics_2["S_transport_norm_ema"],
        jnp.linalg.norm(operator_delta),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        metrics_2["S_transport_norm_current"],
        jnp.linalg.norm(current_operator_delta),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        metrics_2["transport_ratio_ema"],
        jnp.linalg.norm(operator_delta) / jnp.linalg.norm(current_gradient_1),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        metrics_2["transport_ratio_current"],
        jnp.linalg.norm(current_operator_delta)
        / jnp.linalg.norm(current_gradient_1),
        rtol=1e-5,
    )


def test_wssr_reliability_diagnostics_do_not_change_two_step_update():
    """Reliability metrics must be observational, including on a warm step."""
    params = _tiny_params()
    data = _tiny_positions()

    def _make_config(enabled):
        config = default_config.get_default_config()
        config.vmc.nchains = data.shape[0]
        config.vmc.optimizer_type = "wssr_warm_svd_right"
        opt = config.vmc.optimizer.wssr_warm_svd_right
        opt.schedule_type = "constant"
        opt.learning_rate = 0.04
        opt.constrain_norm = False
        opt.sr_rank = 2
        opt.sr_rank_max = 3
        opt.sr_storage_rank = 3
        opt.svd_working_rank = 3
        opt.svd_maxiter_initial = 3
        opt.svd_maxiter_warm = 2
        opt.spectral_regularization = "tikhonov"
        opt.relative_singular_value_cutoff = 3e-4
        opt.tikhonov_lambda = 1e-3
        opt.complement_weight = 0.0
        opt.reliability_diagnostics = enabled
        return config

    baseline_fn, baseline_state, baseline_key = initialize_optimizer(
        _log_psi_apply,
        _local_energy_fn,
        None,
        _make_config(False).vmc,
        params,
        data,
        lambda x: x,
        lambda d, p: d,
        jax.random.PRNGKey(19),
        apply_pmap=False,
    )
    diagnostic_fn, diagnostic_state, diagnostic_key = initialize_optimizer(
        _log_psi_apply,
        _local_energy_fn,
        None,
        _make_config(True).vmc,
        params,
        data,
        lambda x: x,
        lambda d, p: d,
        jax.random.PRNGKey(19),
        apply_pmap=False,
    )

    baseline_params = diagnostic_params = params
    baseline_data = diagnostic_data = data
    diagnostic_metrics = None
    for _ in range(2):
        (
            baseline_params,
            baseline_data,
            baseline_state,
            _,
            baseline_key,
        ) = baseline_fn(
            baseline_params, baseline_data, baseline_state, baseline_key
        )
        (
            diagnostic_params,
            diagnostic_data,
            diagnostic_state,
            diagnostic_metrics,
            diagnostic_key,
        ) = diagnostic_fn(
            diagnostic_params,
            diagnostic_data,
            diagnostic_state,
            diagnostic_key,
        )

    assert_pytree_allclose(
        diagnostic_params, baseline_params, rtol=1e-6, atol=1e-7
    )
    assert_pytree_allclose(
        diagnostic_state, baseline_state, rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(diagnostic_data, baseline_data)
    np.testing.assert_array_equal(diagnostic_key, baseline_key)
    reliability_metrics = {
        key: value
        for key, value in diagnostic_metrics.items()
        if key.startswith("wssr_reliability_")
    }
    assert set(reliability_metrics) == {
        "wssr_reliability_update_weighted_ritz_residual",
        "wssr_reliability_previous_subspace_retained_fraction",
        "wssr_reliability_temporal_update_novelty",
        "wssr_reliability_ssi_comparison_iterations",
        "wssr_reliability_ssi_update_relative_change",
        "wssr_reliability_ssi_update_cosine",
        "wssr_reliability_raw_energy_median",
        "wssr_reliability_raw_energy_mad",
        "wssr_reliability_raw_tail_q99_robust_z",
        "wssr_reliability_raw_tail_max_robust_z",
        "wssr_reliability_raw_tail_fraction_gt_10_robust_z",
    }
    assert jnp.all(
        jnp.isfinite(jnp.asarray(list(reliability_metrics.values())))
    )
    assert diagnostic_metrics[
        "wssr_reliability_ssi_comparison_iterations"
    ] == jnp.asarray(1)


def test_wssr_warm_svd_right_two_step_no_persistent_u_matches_persistent_metrics():
    params = _tiny_params()
    data = _tiny_positions()

    def _make_config(store_warm_u):
        config = default_config.get_default_config()
        config.vmc.nchains = data.shape[0]
        config.vmc.optimizer_type = "wssr_warm_svd_right"
        config.vmc.optimizer.wssr_warm_svd_right.schedule_type = "constant"
        config.vmc.optimizer.wssr_warm_svd_right.learning_rate = 0.125
        config.vmc.optimizer.wssr_warm_svd_right.constrain_norm = False
        config.vmc.optimizer.wssr_warm_svd_right.sr_rank = 2
        config.vmc.optimizer.wssr_warm_svd_right.sr_rank_max = 5
        config.vmc.optimizer.wssr_warm_svd_right.svd_working_rank = 3
        config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_initial = 2
        config.vmc.optimizer.wssr_warm_svd_right.svd_maxiter_warm = 1
        config.vmc.optimizer.wssr_warm_svd_right.store_warm_u = store_warm_u
        return config

    persistent_update_fn, persistent_state, persistent_key = initialize_optimizer(
        _log_psi_apply,
        _local_energy_fn,
        None,
        _make_config(True).vmc,
        params,
        data,
        lambda x: x,
        lambda d, p: d,
        jax.random.PRNGKey(11),
        apply_pmap=False,
    )
    derived_update_fn, derived_state, derived_key = initialize_optimizer(
        _log_psi_apply,
        _local_energy_fn,
        None,
        _make_config(False).vmc,
        params,
        data,
        lambda x: x,
        lambda d, p: d,
        jax.random.PRNGKey(11),
        apply_pmap=False,
    )

    persistent_params_1, _, persistent_state_1, persistent_metrics_1, persistent_key_1 = (
        persistent_update_fn(params, data, persistent_state, persistent_key)
    )
    derived_params_1, _, derived_state_1, derived_metrics_1, derived_key_1 = (
        derived_update_fn(params, data, derived_state, derived_key)
    )
    persistent_params_2, _, persistent_state_2, persistent_metrics_2, _ = (
        persistent_update_fn(
            persistent_params_1, data, persistent_state_1, persistent_key_1
        )
    )
    derived_params_2, _, derived_state_2, derived_metrics_2, _ = derived_update_fn(
        derived_params_1, data, derived_state_1, derived_key_1
    )

    _assert_tree_all_finite(persistent_params_2)
    _assert_tree_all_finite(derived_params_2)
    _assert_wssr_state_all_finite(persistent_state_2.core_state)
    _assert_wssr_state_all_finite(derived_state_2.core_state)
    assert persistent_state_2.core_state.u.shape == (3, 5)
    assert derived_state_2.core_state.u.shape == (3, 0)
    assert derived_state_2.core_state.sr_o.shape == persistent_state_2.core_state.sr_o.shape
    assert set(persistent_metrics_1).issuperset({"energy", "variance", "energy_noclip"})
    assert set(derived_metrics_1).issuperset({"energy", "variance", "energy_noclip"})
    for key in ("energy", "variance", "energy_noclip"):
        np.testing.assert_allclose(
            derived_metrics_1[key], persistent_metrics_1[key], rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            derived_metrics_2[key], persistent_metrics_2[key], rtol=1e-6, atol=1e-6
        )


def test_initialize_wssr_warm_svd_right_rejects_pmap_until_jit_safe_core_exists():
    params = _tiny_params()
    data = _tiny_positions()
    config = default_config.get_default_config()
    config.vmc.nchains = data.shape[0]
    config.vmc.optimizer_type = "wssr_warm_svd_right"

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
        raise AssertionError("Expected wssr_warm_svd_right to reject apply_pmap=True")


def test_initialize_wssr_warm_svd_right_matfree_rejects_pmap_until_supported():
    params = _tiny_params()
    data = _tiny_positions()
    config = default_config.get_default_config()
    config.vmc.nchains = data.shape[0]
    config.vmc.optimizer_type = "wssr_warm_svd_right_matfree"

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
        raise AssertionError(
            "Expected wssr_warm_svd_right_matfree to reject apply_pmap=True"
        )


def test_current_subspace_uniform_spectrum_matches_current_tikhonov_without_history():
    o_cur = jnp.array(
        [
            [1.0, -0.5, 0.2, 0.7, -0.1],
            [0.3, 0.8, -0.4, 0.1, -0.8],
            [-0.2, 0.6, 0.9, -0.5, -0.8],
            [0.4, -0.1, 0.5, -0.9, 0.1],
        ],
        dtype=jnp.float32,
    )
    e_cur = jnp.array([0.7, -0.2, 0.5, -0.4, -0.6], dtype=jnp.float32)
    state = wssr.initialize_wssr_warm_svd_core_state(
        4, 3, 3, dtype=jnp.float32, store_warm_u=True
    )
    common = dict(
        key=jax.random.PRNGKey(91),
        damping=1e-3,
        norm_constraint=1.0,
        sr_rank_max=3,
        sr_scale=1.0,
        svd_maxiter_initial=2,
        svd_maxiter_warm=1,
        exact_first=True,
        svd_working_rank=3,
        constrain_update_norm=False,
        relative_singular_value_cutoff=0.0,
        tikhonov_lambda=0.1,
    )
    reference = wssr.wssr_warm_svd_right_core_update(
        o_cur,
        e_cur,
        state,
        spectral_regularization="tikhonov",
        complement_weight=0.0,
        **common,
    )
    candidate = wssr.wssr_current_subspace_reduced_metric_core_update(
        o_cur,
        e_cur,
        state,
        eta_S=0.8,
        history_mode="uniform_spectrum",
        **common,
    )

    np.testing.assert_allclose(
        candidate.grad_like_update,
        reference.grad_like_update,
        rtol=2e-5,
        atol=2e-6,
    )
    assert candidate.active_rank == reference.active_rank
    np.testing.assert_allclose(candidate.diagnostics.eta_by_mode, 0.0)


def test_anisotropic_matrix_history_weights_tail_more_than_head():
    diagnostics = wssr.spectral_position_history_weights(
        jnp.array([100.0, 10.0, 1.0, 0.1]),
        jnp.ones((4,), dtype=bool),
        jnp.asarray(True),
        eta_max=0.95,
        num_samples=1000,
        noise_scale=1.0,
    )

    eta = np.asarray(diagnostics.eta_by_mode)
    assert np.all(np.diff(eta) > 0.0)
    assert diagnostics.eta_tail > diagnostics.eta_head
    assert 0.0 < diagnostics.eta_spectral_mean < 0.95
    assert diagnostics.current_weight == pytest.approx(
        1.0 - float(diagnostics.eta_spectral_mean), abs=1e-7
    )


def test_anisotropic_matrix_augmentation_recovers_legacy_for_uniform_weights():
    o_cur = jnp.array([[1.0, -0.5, 0.2], [0.3, 0.8, -0.4]])
    e_cur = jnp.array([0.2, -0.1, -0.1])
    state = wssr.WSSRWarmSVDCoreState(
        sr_o=jnp.array([[1.2, 0.0], [0.0, 0.7]]),
        ek=jnp.array([0.3, -0.2]),
        sr_rank0=jnp.asarray(2),
        sr_rank=jnp.asarray(2),
        u=jnp.eye(2),
        has_u=jnp.asarray(True),
    )
    eta = 0.3
    diagnostics = wssr.WSSRAnisotropicMatrixHistoryDiagnostics(
        eta_by_mode=jnp.full((2,), eta),
        eta_mean=jnp.asarray(eta),
        eta_head=jnp.asarray(eta),
        eta_tail=jnp.asarray(eta),
        eta_spectral_mean=jnp.asarray(eta),
        current_weight=jnp.asarray(1.0 - eta),
        noise_floor=jnp.asarray(0.0),
    )
    candidate = wssr.augment_wssr_anisotropic_matrix_history(
        o_cur, state, diagnostics
    )
    reference, _ = wssr.augment_wssr_system(o_cur, e_cur, state, eta)

    np.testing.assert_allclose(candidate, reference, rtol=1e-7, atol=1e-7)


def test_anisotropic_matrix_history_matches_current_tikhonov_without_history():
    o_cur = jnp.array(
        [
            [1.0, -0.5, 0.2, 0.7, -0.1],
            [0.3, 0.8, -0.4, 0.1, -0.8],
            [-0.2, 0.6, 0.9, -0.5, -0.8],
            [0.4, -0.1, 0.5, -0.9, 0.1],
        ],
        dtype=jnp.float32,
    )
    e_cur = jnp.array([0.7, -0.2, 0.5, -0.4, -0.6], dtype=jnp.float32)
    state = wssr.initialize_wssr_warm_svd_core_state(
        4, 3, 3, dtype=jnp.float32, store_warm_u=True
    )
    common = dict(
        key=jax.random.PRNGKey(96),
        damping=1e-3,
        norm_constraint=1.0,
        sr_rank_max=3,
        sr_scale=1.0,
        svd_maxiter_initial=2,
        svd_maxiter_warm=1,
        exact_first=True,
        svd_working_rank=3,
        constrain_update_norm=False,
        relative_singular_value_cutoff=0.0,
        tikhonov_lambda=0.1,
    )
    reference = wssr.wssr_warm_svd_right_core_update(
        o_cur,
        e_cur,
        state,
        spectral_regularization="tikhonov",
        complement_weight=0.0,
        **common,
    )
    candidate = wssr.wssr_anisotropic_matrix_history_core_update(
        o_cur,
        e_cur,
        state,
        eta_S=0.95,
        **common,
    )

    np.testing.assert_allclose(
        candidate.grad_like_update,
        reference.grad_like_update,
        rtol=2e-5,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        candidate.state.sr_o @ candidate.state.sr_o.T,
        reference.state.sr_o @ reference.state.sr_o.T,
        rtol=2e-5,
        atol=2e-6,
    )
    np.testing.assert_allclose(candidate.diagnostics.eta_by_mode, 0.0)
    assert candidate.diagnostics.current_weight == pytest.approx(1.0)


def test_anisotropic_matrix_eta_zero_matches_two_current_only_steps():
    o_first = jnp.array(
        [
            [1.0, -0.5, 0.2, 0.7],
            [0.3, 0.8, -0.4, 0.1],
            [-0.2, 0.6, 0.9, -0.5],
        ],
        dtype=jnp.float32,
    )
    o_second = jnp.array(
        [
            [0.9, -0.4, 0.1, 0.8],
            [0.4, 0.7, -0.5, 0.2],
            [-0.1, 0.5, 0.8, -0.6],
        ],
        dtype=jnp.float32,
    )
    e_first = jnp.array([0.5, -0.3, 0.2, -0.4], dtype=jnp.float32)
    e_second = jnp.array([0.4, -0.2, 0.3, -0.5], dtype=jnp.float32)
    initial = wssr.initialize_wssr_warm_svd_core_state(
        3, 3, 3, dtype=jnp.float32, store_warm_u=True
    )
    common = dict(
        damping=1e-3,
        norm_constraint=1.0,
        sr_rank_max=3,
        sr_scale=1.0,
        svd_maxiter_initial=2,
        svd_maxiter_warm=1,
        exact_first=True,
        svd_working_rank=3,
        constrain_update_norm=False,
        relative_singular_value_cutoff=0.0,
        tikhonov_lambda=0.1,
    )

    reference_first = wssr.wssr_warm_svd_right_core_update(
        o_first,
        e_first,
        initial,
        key=jax.random.PRNGKey(97),
        spectral_regularization="tikhonov",
        complement_weight=0.0,
        **common,
    )
    candidate_first = wssr.wssr_anisotropic_matrix_history_core_update(
        o_first,
        e_first,
        initial,
        key=jax.random.PRNGKey(97),
        eta_S=0.0,
        **common,
    )
    reference_second = wssr.wssr_warm_svd_right_core_update(
        o_second,
        e_second,
        reference_first.state,
        key=jax.random.PRNGKey(98),
        spectral_regularization="tikhonov",
        complement_weight=0.0,
        **common,
    )
    candidate_second = wssr.wssr_anisotropic_matrix_history_core_update(
        o_second,
        e_second,
        candidate_first.state,
        key=jax.random.PRNGKey(98),
        eta_S=0.0,
        **common,
    )

    np.testing.assert_allclose(
        candidate_second.grad_like_update,
        reference_second.grad_like_update,
        rtol=3e-5,
        atol=3e-6,
    )
    np.testing.assert_allclose(
        candidate_second.state.sr_o @ candidate_second.state.sr_o.T,
        reference_second.state.sr_o @ reference_second.state.sr_o.T,
        rtol=3e-5,
        atol=3e-6,
    )


def test_cluster_adaptive_spectral_history_weights_tail_more_than_head():
    eigenvalues = jnp.array([100.0, 10.0, 1.0, 0.1])
    history = jnp.diag(eigenvalues)
    _, diagnostics = wssr.average_current_reduced_metric(
        eigenvalues,
        history,
        jnp.ones((4,), dtype=bool),
        jnp.asarray(True),
        "cluster_adaptive",
        eta_max=0.95,
        num_samples=1000,
        cluster_gap_threshold=0.01,
        noise_scale=1.0,
        drift_scale=1.0,
        history_total_mass=jnp.sum(eigenvalues),
    )

    eta = np.asarray(diagnostics.eta_by_mode)
    assert np.all(np.diff(eta) > 0.0)
    assert diagnostics.eta_tail > diagnostics.eta_head
    assert diagnostics.drift_ratio_mean == pytest.approx(0.0, abs=1e-7)


def test_uniform_spectrum_history_ignores_historical_off_diagonal_rotation():
    eigenvalues = jnp.array([3.0, 1.0])
    history = jnp.array([[2.0, 0.9], [0.9, 0.5]])
    metric, diagnostics = wssr.average_current_reduced_metric(
        eigenvalues,
        history,
        jnp.ones((2,), dtype=bool),
        jnp.asarray(True),
        "uniform_spectrum",
        eta_max=0.25,
        num_samples=1000,
    )

    expected = jnp.diag(jnp.array([2.75, 0.875]))
    np.testing.assert_allclose(metric, expected, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(diagnostics.eta_by_mode, 0.25)


def test_cluster_adaptive_reduced_metric_is_rotation_invariant_inside_cluster():
    eigenvalues = jnp.array([2.0, 2.0], dtype=jnp.float32)
    history = jnp.array([[1.5, 0.2], [0.2, 1.0]], dtype=jnp.float32)
    angle = 0.37
    rotation = jnp.array(
        [
            [jnp.cos(angle), -jnp.sin(angle)],
            [jnp.sin(angle), jnp.cos(angle)],
        ],
        dtype=jnp.float32,
    )
    kwargs = dict(
        active_mask=jnp.ones((2,), dtype=bool),
        has_history=jnp.asarray(True),
        mode="cluster_adaptive",
        eta_max=0.8,
        num_samples=1000,
        cluster_gap_threshold=0.01,
        noise_scale=1.0,
        drift_scale=1.0,
        history_total_mass=jnp.trace(history),
    )
    metric_a, diagnostics_a = wssr.average_current_reduced_metric(
        eigenvalues, history, **kwargs
    )
    metric_b, diagnostics_b = wssr.average_current_reduced_metric(
        eigenvalues, rotation.T @ history @ rotation, **kwargs
    )
    force_a = jnp.array([0.4, -0.7], dtype=jnp.float32)
    force_b = rotation.T @ force_a
    damping = 0.1
    direction_a = jnp.linalg.solve(
        metric_a + damping * jnp.eye(2), force_a
    )
    direction_b = rotation @ jnp.linalg.solve(
        metric_b + damping * jnp.eye(2), force_b
    )

    np.testing.assert_allclose(direction_b, direction_a, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(
        diagnostics_b.eta_by_mode,
        diagnostics_a.eta_by_mode,
        rtol=2e-5,
        atol=2e-6,
    )


def test_current_subspace_spectral_history_integrates_and_logs_diagnostics():
    params = _tiny_params()
    data = _tiny_positions()
    config = default_config.get_default_config()
    config.vmc.nchains = data.shape[0]
    config.vmc.optimizer_type = "wssr_warm_svd_right"
    opt = config.vmc.optimizer.wssr_warm_svd_right
    opt.schedule_type = "constant"
    opt.learning_rate = 0.02
    opt.constrain_norm = False
    opt.eta = 0.0
    opt.eta_S = 0.8
    opt.eta_g = 0.0
    opt.reduced_metric_history_mode = "cluster_adaptive"
    opt.spectral_regularization = "tikhonov"
    opt.tikhonov_lambda = 1e-3
    opt.relative_singular_value_cutoff = 0.0
    opt.complement_weight = 0.0
    opt.sr_rank = 2
    opt.sr_rank_max = 2
    opt.sr_storage_rank = 2
    opt.svd_working_rank = 2
    opt.svd_maxiter_initial = 2
    opt.svd_maxiter_warm = 1
    opt.exact_first = True

    update_fn, state, key = initialize_optimizer(
        _log_psi_apply,
        _local_energy_fn,
        None,
        config.vmc,
        params,
        data,
        lambda x: x,
        lambda d, p: d,
        jax.random.PRNGKey(92),
        apply_pmap=False,
    )
    next_params, _, next_state, metrics, _ = update_fn(
        params, data, state, key
    )

    _assert_tree_all_finite(next_params)
    _assert_wssr_state_all_finite(next_state.core_state)
    assert set(metrics).issuperset(
        {
            "wssr_spectral_history_eta_mean",
            "wssr_spectral_history_eta_head",
            "wssr_spectral_history_eta_tail",
            "wssr_spectral_history_noise_floor",
            "wssr_spectral_history_overlap",
            "wssr_spectral_history_cluster_count",
            "wssr_spectral_history_drift_ratio_mean",
        }
    )
    assert next_state.core_state.ek.shape == (2,)
    np.testing.assert_allclose(next_state.core_state.ek, 0.0)


def test_current_subspace_spectral_history_rejects_gradient_ema():
    params = _tiny_params()
    data = _tiny_positions()
    config = default_config.get_default_config()
    config.vmc.nchains = data.shape[0]
    config.vmc.optimizer_type = "wssr_warm_svd_right"
    opt = config.vmc.optimizer.wssr_warm_svd_right
    opt.eta_S = 0.8
    opt.eta_g = 0.3
    opt.reduced_metric_history_mode = "uniform_spectrum"
    opt.spectral_regularization = "tikhonov"
    opt.tikhonov_lambda = 1e-3
    opt.complement_weight = 0.0

    with pytest.raises(ValueError, match="requires eta_g=0"):
        initialize_optimizer(
            _log_psi_apply,
            _local_energy_fn,
            None,
            config.vmc,
            params,
            data,
            lambda x: x,
            lambda d, p: d,
            jax.random.PRNGKey(93),
            apply_pmap=False,
        )


def test_anisotropic_matrix_history_integrates_without_gradient_memory():
    params = _tiny_params()
    data = _tiny_positions()
    config = default_config.get_default_config()
    config.vmc.nchains = data.shape[0]
    config.vmc.optimizer_type = "wssr_warm_svd_right"
    opt = config.vmc.optimizer.wssr_warm_svd_right
    opt.schedule_type = "constant"
    opt.learning_rate = 0.02
    opt.constrain_norm = False
    opt.eta = 0.0
    opt.eta_S = 0.8
    opt.eta_g = 0.0
    opt.anisotropic_matrix_history = True
    opt.spectral_regularization = "tikhonov"
    opt.tikhonov_lambda = 1e-3
    opt.relative_singular_value_cutoff = 0.0
    opt.complement_weight = 0.0
    opt.sr_rank = 2
    opt.sr_rank_max = 2
    opt.sr_storage_rank = 2
    opt.svd_working_rank = 2
    opt.svd_maxiter_initial = 2
    opt.svd_maxiter_warm = 1
    opt.exact_first = True

    update_fn, state, key = initialize_optimizer(
        _log_psi_apply,
        _local_energy_fn,
        None,
        config.vmc,
        params,
        data,
        lambda x: x,
        lambda d, p: d,
        jax.random.PRNGKey(94),
        apply_pmap=False,
    )
    params_1, data_1, state_1, _, key_1 = update_fn(
        params, data, state, key
    )
    params_2, _, state_2, metrics, _ = update_fn(
        params_1, data_1, state_1, key_1
    )

    _assert_tree_all_finite(params_2)
    _assert_wssr_state_all_finite(state_2.core_state)
    assert set(metrics).issuperset(
        {
            "wssr_anisotropic_matrix_eta_mean",
            "wssr_anisotropic_matrix_eta_head",
            "wssr_anisotropic_matrix_eta_tail",
            "wssr_anisotropic_matrix_eta_spectral_mean",
            "wssr_anisotropic_matrix_current_weight",
            "wssr_anisotropic_matrix_noise_floor",
        }
    )
    assert metrics["wssr_anisotropic_matrix_eta_tail"] >= metrics[
        "wssr_anisotropic_matrix_eta_head"
    ]
    np.testing.assert_allclose(state_2.core_state.ek, 0.0)


def test_anisotropic_matrix_history_rejects_gradient_ema():
    params = _tiny_params()
    data = _tiny_positions()
    config = default_config.get_default_config()
    config.vmc.nchains = data.shape[0]
    config.vmc.optimizer_type = "wssr_warm_svd_right"
    opt = config.vmc.optimizer.wssr_warm_svd_right
    opt.eta_S = 0.8
    opt.eta_g = 0.3
    opt.anisotropic_matrix_history = True
    opt.spectral_regularization = "tikhonov"
    opt.tikhonov_lambda = 1e-3
    opt.complement_weight = 0.0

    with pytest.raises(ValueError, match="requires eta_g=0"):
        initialize_optimizer(
            _log_psi_apply,
            _local_energy_fn,
            None,
            config.vmc,
            params,
            data,
            lambda x: x,
            lambda d, p: d,
            jax.random.PRNGKey(95),
            apply_pmap=False,
        )
