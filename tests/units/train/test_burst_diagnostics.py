"""Tests for default-disabled, host-side burst snapshots."""

import json

import jax.numpy as jnp
import numpy as np

from vmcnet.train import burst_diagnostics


def _state():
    params = {"w": jnp.array([1.0, -2.0])}
    optimizer_state = {"trace": jnp.array([0.25, 0.5])}
    data = {
        "walker_data": {
            "position": jnp.array([[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]]),
            "amplitude": jnp.array([-1.0, -2.0]),
        }
    }
    return params, optimizer_state, data, jnp.array([3, 4], dtype=jnp.uint32)


def test_disabled_is_exact_noop(tmp_path):
    params, optimizer_state, data, key = _state()
    metrics = {
        "variance": jnp.array(1.0),
        "variance_noclip": jnp.array(1.0e9),
        "burst_diag_local_energies": jnp.array([-2.0, 3.0]),
    }
    original_keys = tuple(metrics)
    before = np.asarray(params["w"]).copy()
    result = burst_diagnostics.maybe_save_burst(
        epoch=0,
        logdir=str(tmp_path),
        config={"enabled": False, "raw_variance_threshold": 0.0},
        metrics=metrics,
        params=params,
        optimizer_state=optimizer_state,
        data=data,
        key=key,
    )
    assert result is None
    assert tuple(metrics) == original_keys
    np.testing.assert_array_equal(params["w"], before)
    assert list(tmp_path.iterdir()) == []


def test_trigger_saves_payload_and_preserves_state(tmp_path):
    params, optimizer_state, data, key = _state()
    metrics = {
        "variance": jnp.array(2.0),
        "variance_noclip": jnp.array(30.0),
        "burst_diag_local_energies": jnp.array([-8.0, 2.0]),
        "burst_diag_hamiltonian_components": jnp.array(
            [[-5.0, 1.0], [-3.0, 1.0]]
        ),
        "burst_diag_resolved_update": jnp.array([1.0, 2.0]),
        "burst_diag_complement_update": jnp.array([0.1, 0.2]),
    }
    params_before = np.asarray(params["w"]).copy()
    state_before = np.asarray(optimizer_state["trace"]).copy()
    event = burst_diagnostics.maybe_save_burst(
        epoch=6,
        logdir=str(tmp_path),
        config={
            "enabled": True,
            "raw_variance_threshold": 10.0,
            "raw_clipped_ratio_threshold": 100.0,
            "output_dir": "bursts",
        },
        metrics=metrics,
        params=params,
        optimizer_state=optimizer_state,
        data=data,
        key=key,
    )
    assert event is not None
    payload = np.load(f"{event}/burst_payload.npz")
    np.testing.assert_array_equal(payload["local_energies"], [-8.0, 2.0])
    np.testing.assert_array_equal(payload["resolved_update"], [1.0, 2.0])
    assert "burst_diag_local_energies" not in metrics
    manifest = json.loads((tmp_path / "bursts/epoch_00000007/manifest.json").read_text())
    assert manifest["hamiltonian_components_available"]
    np.testing.assert_array_equal(params["w"], params_before)
    np.testing.assert_array_equal(optimizer_state["trace"], state_before)
