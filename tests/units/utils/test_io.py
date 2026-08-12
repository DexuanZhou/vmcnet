"""Testing io routines."""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import vmcnet.utils.distribute as distribute
import vmcnet.utils.io as io

from tests.test_utils import make_dummy_data_params_and_key, assert_pytree_allclose


class _NestedOptimizerState(NamedTuple):
    count: Any
    history: Any


def test_pmapped_save_and_reload_vmc_state(tmp_path):
    """Test round-trip of vmc state to and from disk, with pmapping."""
    # Set up log directory within pytest tmp_dir
    subdir = "logs"
    directory = tmp_path / subdir
    file_name = "checkpoint_file.npz"

    # Create dummy vmc state and distribute it across fake devices
    epoch = 0
    (_, params, key) = make_dummy_data_params_and_key()
    data = {"position": jnp.arange(jax.local_device_count() * 2)}
    opt_state = {"momentum": 2.0}
    (data, params, opt_state, key) = distribute.distribute_vmc_state(
        data, params, opt_state, key
    )

    # Save the vmc state to file, then reload it and redistribute it
    processed_data = io.process_checkpoint_data_for_saving(
        (epoch, data, params, opt_state, key), is_distributed=True
    )
    io.save_vmc_state(directory, file_name, processed_data)
    (
        restored_epoch,
        restored_data,
        restored_params,
        restored_opt_state,
        restored_key,
    ) = io.reload_vmc_state(directory, file_name)
    (
        restored_data,
        restored_params,
        restored_opt_state,
        restored_key,
    ) = distribute.distribute_vmc_state_from_checkpoint(
        restored_data, restored_params, restored_opt_state, restored_key
    )

    # Verify that restored data is same as original data
    np.testing.assert_equal(restored_epoch, epoch)
    assert_pytree_allclose(restored_data, data)
    assert_pytree_allclose(restored_params, params)
    assert_pytree_allclose(restored_opt_state, opt_state)
    np.testing.assert_allclose(restored_key, key)


def test_non_pmapped_save_and_reload_vmc_state(tmp_path):
    """Test round-trip of vmc state to and from disk, without pmapping."""
    # Set up log directory within pytest tmp_dir
    subdir = "logs"
    directory = tmp_path / subdir
    file_name = "checkpoint_file.npz"

    # Create dummy vmc state and distribute it across fake devices
    epoch = 0
    (_, params, key) = make_dummy_data_params_and_key()
    data = {"position": jnp.arange(10)}
    opt_state = {"momentum": 2.0}

    # Save the vmc state to file, then reload it and redistribute it
    processed_data = io.process_checkpoint_data_for_saving(
        (epoch, data, params, opt_state, key), is_distributed=False
    )
    io.save_vmc_state(directory, file_name, processed_data)
    (
        restored_epoch,
        restored_data,
        restored_params,
        restored_opt_state,
        restored_key,
    ) = io.reload_vmc_state(directory, file_name)

    # Verify that restored data is same as original data
    np.testing.assert_equal(restored_epoch, epoch)
    assert_pytree_allclose(restored_data, data)
    assert_pytree_allclose(restored_params, params)
    assert_pytree_allclose(restored_opt_state, opt_state)
    np.testing.assert_allclose(restored_key, key)


def test_array_data_save_and_reload_vmc_state(tmp_path):
    """Test round-trip of vmc state to and from disk, without pmapping."""
    # Set up log directory within pytest tmp_dir
    subdir = "logs"
    directory = tmp_path / subdir
    file_name = "checkpoint_file.npz"

    # Create dummy vmc state and distribute it across fake devices
    epoch = 0
    # This data is a simple ndarray, NOT a dict
    (data, params, key) = make_dummy_data_params_and_key()
    opt_state = {"momentum": 2.0}

    # Save the vmc state to file, then reload it and redistribute it
    processed_data = io.process_checkpoint_data_for_saving(
        (epoch, data, params, opt_state, key), is_distributed=False
    )
    io.save_vmc_state(directory, file_name, processed_data)
    (
        restored_epoch,
        restored_data,
        restored_params,
        restored_opt_state,
        restored_key,
    ) = io.reload_vmc_state(directory, file_name)

    # Verify that restored data is same as original data
    assert_pytree_allclose(restored_data, data)


def test_nested_optimizer_pytree_roundtrip(tmp_path):
    """Optimizer NamedTuples and their numerical leaves survive a round-trip."""
    (_, params, key) = make_dummy_data_params_and_key()
    optimizer_state = _NestedOptimizerState(
        count=np.asarray(7, dtype=np.int32),
        history={
            "matrix": np.arange(12, dtype=np.float64).reshape(3, 4),
            "nested": (jnp.asarray([1.0, 2.0]), None),
        },
    )
    checkpoint_data = io.process_checkpoint_data_for_saving(
        (3, np.arange(2), params, optimizer_state, key), is_distributed=False
    )

    io.save_vmc_state(tmp_path, "nested.npz", checkpoint_data)
    restored = io.reload_vmc_state(tmp_path, "nested.npz")

    assert isinstance(restored[3], _NestedOptimizerState)
    assert_pytree_allclose(restored[3], optimizer_state)


def test_optimizer_leaves_are_stored_as_separate_numeric_members(tmp_path):
    """Large optimizer leaves do not enter NumPy's object-array pickle path."""
    (_, params, key) = make_dummy_data_params_and_key()
    optimizer_state = {
        "large_leaf": np.arange(1024, dtype=np.float64),
        "step": np.asarray(4, dtype=np.int32),
    }
    checkpoint_data = io.process_checkpoint_data_for_saving(
        (4, np.arange(2), params, optimizer_state, key), is_distributed=False
    )

    io.save_vmc_state(tmp_path, "leaf_sharded.npz", checkpoint_data)

    with np.load(tmp_path / "leaf_sharded.npz", allow_pickle=True) as npz_data:
        assert npz_data["o_format"].item() == "pytree_leaves_v1"
        assert npz_data["o"].dtype != np.dtype("object")
        assert npz_data["o_treedef"].dtype == np.dtype(np.uint8)
        num_leaves = int(npz_data["o_num_leaves"].item())
        assert num_leaves == 2
        for index in range(num_leaves):
            assert npz_data[f"o_leaf_{index:06d}"].dtype != np.dtype("object")


def test_reload_legacy_object_array_optimizer_checkpoint(tmp_path):
    """The new reader remains compatible with existing checkpoints."""
    epoch = 9
    (data, params, key) = make_dummy_data_params_and_key()
    optimizer_state = ({"momentum": np.asarray([1.0, 2.0])}, np.asarray(5))

    with open(tmp_path / "legacy.npz", "wb") as file_handle:
        np.savez(
            file_handle,
            e=epoch,
            d=data,
            p=params,
            o=io.wrap_singleton(optimizer_state),
            k=key,
        )

    restored = io.reload_vmc_state(tmp_path, "legacy.npz")

    assert restored[0] == epoch
    assert_pytree_allclose(restored[3], optimizer_state)
