"""Tests for the matrix-free function-space update constraint."""

import jax.numpy as jnp
import numpy as np

from vmcnet.updates.update_param_fns import constrain_update_function_norm


def _linear_log_psi(params, position):
    return jnp.vdot(params, position)


def test_function_space_constraint_matches_centered_score_action():
    params = jnp.array([0.2, -0.1, 0.3])
    positions = jnp.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 2.0, -1.0],
            [2.0, -1.0, 0.0],
            [-1.0, 1.0, 2.0],
        ]
    )
    update = jnp.array([0.4, -0.2, 0.1])
    radius = 0.07

    constrained, norm_before, scale, norm_after = (
        constrain_update_function_norm(
            _linear_log_psi, params, positions, update, radius
        )
    )

    raw_action = positions @ update
    expected_norm = np.linalg.norm(
        (np.asarray(raw_action) - np.asarray(raw_action).mean())
        / np.sqrt(positions.shape[0])
    )
    expected_scale = min(1.0, radius / expected_norm)
    np.testing.assert_allclose(norm_before, expected_norm, rtol=1e-6)
    np.testing.assert_allclose(scale, expected_scale, rtol=1e-6)
    np.testing.assert_allclose(constrained, update * expected_scale, rtol=1e-6)
    np.testing.assert_allclose(norm_after, radius, rtol=1e-6)


def test_function_space_constraint_leaves_update_below_radius_unchanged():
    params = jnp.zeros(2)
    positions = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    update = jnp.array([1e-4, -1e-4])

    constrained, _, scale, norm_after = constrain_update_function_norm(
        _linear_log_psi, params, positions, update, 1.0
    )

    np.testing.assert_allclose(scale, 1.0)
    np.testing.assert_allclose(constrained, update)
    assert float(norm_after) < 1.0
