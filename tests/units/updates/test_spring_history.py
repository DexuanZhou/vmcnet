"""Deterministic tests for SPRING history-state semantics."""
import math
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax

from vmcnet.updates.spring import constrain_norm, get_spring_step

def linear_logpsi(params, positions):
    return positions @ params


X = jnp.asarray([[1.0, -2.0, 0.5], [0.3, 1.1, -0.7], [-1.2, 0.4, 1.5], [0.2, 0.5, -0.1]])
ENERGY_RESIDUALS = [
    jnp.asarray([0.7, -0.2, 0.4, -0.9]),
    jnp.asarray([-0.3, 0.8, -0.6, 0.1]),
    jnp.asarray([0.2, 0.5, -0.4, -0.3]),
]
DAMPING = 0.07


def reference_code_sign(centered_energy, previous_q, mu):
    """Reference q=-phi, matching the sign passed into Optax by VMCNet."""
    ns = X.shape[0]
    o_bar = (X - jnp.mean(X, axis=0, keepdims=True)) / jnp.sqrt(ns)
    p = jnp.ones((ns, ns)) / ns
    rhs = centered_energy / jnp.sqrt(ns) - o_bar @ (mu * previous_q)
    return o_bar.T @ jnp.linalg.solve(o_bar @ o_bar.T + DAMPING * jnp.eye(ns) + p, rhs) + mu * previous_q


def assert_same(actual, expected):
    # Do not mutate JAX's process-wide x64 flag during test collection: doing
    # so invalidates fp32 executables cached by unrelated fwdlap tests.
    tolerance = 2e-11 if jax.config.x64_enabled else 2e-6
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    na, ne = np.linalg.norm(actual), np.linalg.norm(expected)
    assert abs(na - ne) <= tolerance * max(1.0, ne)
    if na and ne:
        assert np.dot(actual, expected) / (na * ne) > 1 - tolerance


def run_current(mu, learning_rate, constrain=False, state=None, steps=3):
    spring = get_spring_step(linear_logpsi, DAMPING, mu)
    tx = optax.sgd(learning_rate=learning_rate, momentum=0, nesterov=False)
    params = jnp.asarray([0.2, -0.1, 0.4])
    state = tx.init(params) if state is None else state
    records = []
    for energy in ENERGY_RESIDUALS[:steps]:
        q = spring(energy, params, state[0].trace, X)
        lr_update, state = tx.update(q, state, params)
        applied = constrain_norm(lr_update, 0.001) if constrain else lr_update
        new_params = optax.apply_updates(params, applied)
        records.append((q, state[0].trace, applied, new_params - params))
        params = new_params
    return records, state


def test_mu_zero_is_minsr():
    records, _ = run_current(0.0, 0.13, steps=1)
    assert_same(records[0][0], reference_code_sign(ENERGY_RESIDUALS[0], jnp.zeros(3), 0.0))


def test_diagnostics_do_not_change_direction():
    spring = get_spring_step(linear_logpsi, DAMPING, 0.83)
    params = jnp.asarray([0.2, -0.1, 0.4])
    history = jnp.asarray([0.1, -0.2, 0.3])
    plain = spring(ENERGY_RESIDUALS[0], params, history, X)
    diagnosed, metrics = spring(
        ENERGY_RESIDUALS[0], params, history, X, return_diagnostics=True
    )
    assert_same(plain, diagnosed)
    for key in (
        "spring_diag_input_norm",
        "spring_diag_history_norm",
        "spring_diag_mu_history_norm",
        "spring_diag_nonhistory_norm",
        "spring_diag_raw_solution_norm",
        "spring_diag_history_projection_norm",
    ):
        assert key in metrics


def test_spectral_and_float64_replay_diagnostics_do_not_change_direction():
    params = jnp.asarray([0.2, -0.1, 0.4])
    history = jnp.asarray([0.1, -0.2, 0.3])
    plain = get_spring_step(linear_logpsi, DAMPING, 0.83)(
        ENERGY_RESIDUALS[0], params, history, X
    )
    diagnosed, metrics = get_spring_step(
        linear_logpsi,
        DAMPING,
        0.83,
        diagnostics_spectral=True,
        diagnostics_replay_epochs=(1,),
    )(
        ENERGY_RESIDUALS[0],
        params,
        history,
        X,
        return_diagnostics=True,
        diagnostic_step=1,
    )
    assert_same(plain, diagnosed)
    # Host replay returns the solve as float32 before the unchanged model VJP.
    replay_tolerance = 1e-6 if jax.config.x64_enabled else 2e-6
    assert metrics["spring_replay_relative_direction_error"] < replay_tolerance
    assert metrics["spring_replay_direction_cosine"] > 1 - replay_tolerance
    assert "spring_spec_low_rhs_fraction" in metrics


def test_float32_training_path_replays_in_float64_without_changing_direction():
    with jax.experimental.disable_x64():
        params = jnp.asarray([0.2, -0.1, 0.4], dtype=jnp.float32)
        history = jnp.asarray([0.1, -0.2, 0.3], dtype=jnp.float32)
        positions = X.astype(jnp.float32)
        residual = ENERGY_RESIDUALS[0].astype(jnp.float32)
        plain = get_spring_step(linear_logpsi, DAMPING, 0.83)(
            residual, params, history, positions
        )
        diagnosed, metrics = get_spring_step(
            linear_logpsi, DAMPING, 0.83,
            diagnostics_spectral=True, diagnostics_replay_epochs=(1,),
            diagnostics_decomposition=True,
        )(residual, params, history, positions, return_diagnostics=True, diagnostic_step=1)
    np.testing.assert_array_equal(np.asarray(plain), np.asarray(diagnosed))
    assert math.isfinite(float(metrics["spring_replay_relative_direction_error"]))
    assert metrics["spring_replay_direction_cosine"] > 0.999
    assert math.isfinite(float(metrics["spring_decomp_true_history_ratio"]))
    assert math.isfinite(float(metrics["spring_replay_history_relative_error"]))


def test_mixed_precision_solve_is_locally_float64_and_returns_float32():
    with jax.experimental.disable_x64():
        params = jnp.asarray([0.2, -0.1, 0.4], dtype=jnp.float32)
        history = jnp.asarray([0.1, -0.2, 0.3], dtype=jnp.float32)
        step = get_spring_step(
            linear_logpsi,
            DAMPING,
            0.5,
            diagnostics_spectral=True,
            mixed_precision_solve=True,
        )
        direction, metrics = jax.jit(
            lambda residual, p, h, x: step(
                residual, p, h, x, return_diagnostics=True
            )
        )(
            ENERGY_RESIDUALS[0].astype(jnp.float32),
            params,
            history,
            X.astype(jnp.float32),
        )
        assert direction.dtype == jnp.float32
        assert math.isfinite(float(metrics["spring_spec_max_eigenvalue"]))
        assert not jax.config.x64_enabled


def test_two_and_three_steps_match_reference_and_no_double_momentum():
    mu = 0.83
    records, _ = run_current(mu, 0.13)
    previous = jnp.zeros(3)
    for i, (q, trace, _, _) in enumerate(records):
        expected = reference_code_sign(ENERGY_RESIDUALS[i], previous, mu)
        assert_same(q, expected); assert_same(trace, expected)
        previous = expected


def test_history_is_learning_rate_independent_and_manual_scaling_matches_optax():
    low, _ = run_current(0.83, 0.01)
    high, _ = run_current(0.83, 0.37)
    for (ql, tl, ul, _), (qh, th, uh, _) in zip(low, high):
        assert_same(ql, qh)
        assert_same(tl, th)
        assert_same(ul, -0.01 * ql)
        assert_same(uh, -0.37 * qh)


def test_optax_is_only_learning_rate_scaling_not_an_extra_recurrence():
    mu, learning_rate = 0.83, 0.13
    optax_records, _ = run_current(mu, learning_rate)
    manual_history = jnp.zeros(3)
    for i, (q_optax, trace, update_optax, _) in enumerate(optax_records):
        q_manual = reference_code_sign(
            ENERGY_RESIDUALS[i], manual_history, mu
        )
        update_manual = -learning_rate * q_manual
        assert_same(q_optax, q_manual)
        assert_same(trace, q_manual)
        assert_same(update_optax, update_manual)
        manual_history = q_manual


def test_sign_convention_two_step_example():
    records, _ = run_current(0.83, 0.13, steps=2)
    q0, _, displacement0, _ = records[0]
    assert np.dot(q0, displacement0) < 0
    correct = reference_code_sign(ENERGY_RESIDUALS[1], q0, 0.83)
    wrong_sign = reference_code_sign(ENERGY_RESIDUALS[1], -q0, 0.83)
    assert_same(records[1][0], correct)
    assert np.linalg.norm(records[1][0] - wrong_sign) > 1e-3


def test_norm_constraint_preserves_unconstrained_history_as_intended():
    records, _ = run_current(0.83, 2.0, constrain=True, steps=2)
    for q, trace, applied, displacement in records:
        assert_same(trace, q)
        assert_same(applied, displacement)
        expected = -q * min(2.0, np.sqrt(0.001) / np.linalg.norm(q))
        assert_same(applied, expected)
        cap_tolerance = 1e-12 if jax.config.x64_enabled else 2e-7
        assert np.linalg.norm(applied) <= np.sqrt(0.001) * (1 + cap_tolerance)


def test_optimizer_state_roundtrip_exact_next_update():
    _, state = run_current(0.83, 0.13, steps=2)
    restored = pickle.loads(pickle.dumps(state))
    spring = get_spring_step(linear_logpsi, DAMPING, 0.83)
    params = jnp.asarray([0.2, -0.1, 0.4])
    q_a = spring(ENERGY_RESIDUALS[2], params, state[0].trace, X)
    q_b = spring(ENERGY_RESIDUALS[2], params, restored[0].trace, X)
    assert_same(q_a, q_b)
    tx = optax.sgd(learning_rate=0.13, momentum=0, nesterov=False)
    update_a, next_a = tx.update(q_a, state, params)
    update_b, next_b = tx.update(q_b, restored, params)
    assert_same(update_a, update_b)
    assert_same(next_a[0].trace, next_b[0].trace)
