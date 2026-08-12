"""Testing main VMC routine."""

import csv
import logging

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import wandb

import vmcnet.mcmc as mcmc
import vmcnet.train as train
import vmcnet.utils as utils

from tests.test_utils import make_dummy_data_params_and_key, make_dummy_metropolis_fn


def test_compute_rolling_smooth_metrics():
    """Test rolling tail averages and configured window sizes."""
    avg, window = train.vmc.compute_rolling_smooth_metrics([], total_count=10)
    assert avg is None
    assert window == 2

    avg, window = train.vmc.compute_rolling_smooth_metrics([3.0], total_count=10)
    assert avg == 3.0
    assert window == 2

    avg, window = train.vmc.compute_rolling_smooth_metrics(
        list(range(10)), total_count=10
    )
    assert avg == pytest.approx(8.5)
    assert window == 2

    avg, window = train.vmc.compute_rolling_smooth_metrics(
        list(range(100)), total_count=100
    )
    assert avg == pytest.approx(np.mean(np.arange(80, 100)))
    assert window == 20


def _make_different_pmappable_data(data):
    """Adding (0, 1, ..., ndevices - 1) to the data and concatenating."""
    ndevices = jax.local_device_count()
    return jnp.concatenate([data + i for i in range(ndevices)])


@pytest.mark.slow
def test_vmc_loop_logging(caplog):
    """Test vmc_loop logging. Uses pytest's caplog fixture to capture logs."""
    nburn = 4
    nepochs = 13  # eventual number of parameter updates
    nsteps_per_param_update = 10

    fixed_metrics = {
        "energy": 1.0,
        "energy_noclip": 2.5,
        "variance": 3.0,
        "variance_noclip": np.pi,
    }

    def update_param_fn(params, data, optimizer_state, key):
        return params, data, optimizer_state, fixed_metrics, key

    wandb.init(mode="disabled")

    for pmapped in [True, False]:
        caplog.clear()
        data, params, key = make_dummy_data_params_and_key()
        metrop_step_fn = make_dummy_metropolis_fn()
        nchains = data.shape[0]

        if pmapped:
            data = _make_different_pmappable_data(data)
            (
                data,
                params,
                optimizer_state,
                key,
            ) = utils.distribute.distribute_vmc_state(data, params, None, key)

        burning_step = mcmc.metropolis.make_jitted_burning_step(
            metrop_step_fn, apply_pmap=pmapped
        )
        walker_fn = mcmc.metropolis.make_jitted_walker_fn(
            nsteps_per_param_update, metrop_step_fn, apply_pmap=pmapped
        )

        with caplog.at_level(logging.INFO):
            data, key = mcmc.metropolis.burn_data(
                burning_step, nburn, params, data, key
            )
            train.vmc.vmc_loop(
                params,
                optimizer_state,
                data,
                nchains,
                nepochs,
                walker_fn,
                update_param_fn,
                key,
                is_pmapped=pmapped,
            )

        # 1 line for burning, nepochs lines for training
        assert len(caplog.records) == 1 + nepochs
        assert "Energy smooth20" in caplog.records[-1].message
        assert "Variance smooth20" in caplog.records[-1].message
        assert "Smooth20 window" in caplog.records[-1].message


def test_vmc_loop_writes_training_metrics_csv(tmp_path):
    """Test that vmc_loop writes one CSV row per epoch with smoothed metrics."""
    nepochs = 10
    nsteps_per_param_update = 1
    data, params, key = make_dummy_data_params_and_key()
    metrop_step_fn = make_dummy_metropolis_fn()
    nchains = data.shape[0]
    counter = {"epoch": 0}

    def update_param_fn(params, data, optimizer_state, key):
        counter["epoch"] += 1
        epoch_value = float(counter["epoch"])
        metrics = {
            "energy": epoch_value,
            "energy_noclip": epoch_value + 0.25,
            "variance": epoch_value + 10.0,
            "variance_noclip": epoch_value + 20.0,
        }
        return params, data, optimizer_state, metrics, key

    walker_fn = mcmc.metropolis.make_jitted_walker_fn(
        nsteps_per_param_update, metrop_step_fn, apply_pmap=False
    )
    wandb.init(mode="disabled")

    train.vmc.vmc_loop(
        params,
        None,
        data,
        nchains,
        nepochs,
        walker_fn,
        update_param_fn,
        key,
        logdir=str(tmp_path),
        checkpoint_every=None,
        best_checkpoint_every=None,
        is_pmapped=False,
    )

    csv_path = tmp_path / "training_metrics.csv"
    assert csv_path.exists()
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == nepochs
    last_row = rows[-1]
    assert int(last_row["epoch"]) == nepochs
    assert float(last_row["energy"]) == pytest.approx(10.0)
    assert float(last_row["energy_noclip"]) == pytest.approx(10.25)
    assert float(last_row["variance"]) == pytest.approx(20.0)
    assert float(last_row["variance_noclip"]) == pytest.approx(30.0)
    assert int(last_row["smooth20_window"]) == 2
    assert float(last_row["energy_smooth20"]) == pytest.approx(9.5)
    assert float(last_row["variance_smooth20"]) == pytest.approx(19.5)
    assert float(last_row["accept_smooth20"]) == pytest.approx(0.5)


def test_training_writer_preserves_spring_diagnostics(tmp_path):
    metrics = {
        "energy": jnp.asarray([[-109.5]]),
        "variance": jnp.asarray([[2.0]]),
        "spring_diag_history_norm": jnp.asarray([[3.25]]),
        "spring_spec_low_rhs_fraction": jnp.asarray([[0.125]]),
    }
    train.vmc._append_training_metrics_csv_row(str(tmp_path), 0, metrics)
    with (tmp_path / "spring_diagnostics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert int(rows[0]["epoch"]) == 1
    assert float(rows[0]["spring_diag_history_norm"]) == pytest.approx(3.25)
    assert float(rows[0]["spring_spec_low_rhs_fraction"]) == pytest.approx(0.125)


@pytest.mark.slow
def test_vmc_loop_number_of_updates():
    """Test number of updates.

    Make sure that data is updated nsteps_per_param_update * nepochs + nburn times and
    params is updated nepoch times.
    """
    data, params, key = make_dummy_data_params_and_key()
    metrop_step_fn = make_dummy_metropolis_fn()
    nchains = data.shape[0]

    data = _make_different_pmappable_data(data)
    # storing the number of parameter updates in optimizer_state, replicated on each
    # device; with a real optimizer this is probably something more exciting and
    # possibly data-dependent (e.g. KFAC/Adam's running metrics)
    (
        data,
        params,
        optimizer_state,
        key,
    ) = utils.distribute.distribute_vmc_state(data, params, 0, key)

    nburn = 5
    nepochs = 17  # eventual number of parameter updates
    nsteps_per_param_update = 2

    burning_step = mcmc.metropolis.make_jitted_burning_step(metrop_step_fn)
    walker_fn = mcmc.metropolis.make_jitted_walker_fn(
        nsteps_per_param_update, metrop_step_fn
    )

    def update_param_fn(params, data, optimizer_state, key):
        optimizer_state += 1
        return params, data, optimizer_state, None, key

    wandb.init(mode="disabled")

    data, key = mcmc.metropolis.burn_data(burning_step, nburn, params, data, key)
    _, new_optimizer_state, new_data, _, _ = train.vmc.vmc_loop(
        params,
        optimizer_state,
        data,
        nchains,
        nepochs,
        walker_fn,
        update_param_fn,
        key,
    )

    new_optimizer_state = utils.distribute.get_first(new_optimizer_state)

    num_updates = nsteps_per_param_update * nepochs + nburn

    # check that nepochs "parameter updates" have been done
    assert nepochs == new_optimizer_state
    for device_index in range(jax.local_device_count()):
        np.testing.assert_allclose(
            new_data[device_index],
            jnp.array([num_updates, 0, 3 * num_updates, 0]) + device_index,
        )
