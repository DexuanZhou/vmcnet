"""Audit every WSSR state leaf across the five-level checkpoint conversions."""

import gc
import json
from pathlib import Path

import jax
import jax.flatten_util
import numpy as np

from vmcnet.mcmc import position_amplitude_core as pacore
from vmcnet.models import multilevel
from vmcnet.utils import io


ROOT = Path(
    "/scratch/dexuan1/runs/C_ferminet_multilevel_pilot_20260808/"
    "pilot/wssr_warm_svd_right/five_level"
)
PAIRS = (
    ("L0-L05", "train_L0_to250", "250.npz", "converted_L05"),
    ("L05-L1", "train_L05_to500", "500.npz", "converted_L1"),
    ("L1-L15", "train_L1_to750", "750.npz", "converted_L15"),
    ("L15-L2", "train_L15_to1000", "1000.npz", "converted_L2"),
)


def _max_abs(array):
    array = np.asarray(array)
    return float(np.max(np.abs(array))) if array.size else 0.0


def main():
    result = {}
    for name, coarse_name, checkpoint_name, fine_name in PAIRS:
        coarse_root = ROOT / coarse_name
        fine_root = ROOT / fine_name
        coarse_config = io.load_config_dict(str(coarse_root), "config.json")
        fine_config = io.load_config_dict(str(fine_root), "config.json")
        coarse = io.reload_vmc_state(
            str(coarse_root / "checkpoints"), checkpoint_name
        )
        fine = io.reload_vmc_state(str(fine_root), "multilevel.npz")
        coarse_epoch, coarse_data, coarse_params, coarse_state, coarse_key = coarse
        fine_epoch, fine_data, fine_params, fine_state, fine_key = fine
        coarse_flat, _ = jax.flatten_util.ravel_pytree(coarse_params)
        fine_flat, _ = jax.flatten_util.ravel_pytree(fine_params)
        rows = np.asarray(
            multilevel.ferminet_flat_prolongation_indices(
                coarse_params,
                fine_params,
                coarse_config.model,
                fine_config.model,
                nspins=2,
            )
        )
        new_rows = np.ones((fine_flat.shape[0],), dtype=bool)
        new_rows[rows] = False
        coarse_core = coarse_state.core_state
        fine_core = fine_state.core_state
        result[name] = {
            "epoch_equal": bool(int(coarse_epoch) == int(fine_epoch)),
            "key_equal": bool(
                np.array_equal(np.asarray(coarse_key), np.asarray(fine_key))
            ),
            "positions_equal": bool(
                np.array_equal(
                    np.asarray(pacore.get_position_from_data(coarse_data)),
                    np.asarray(pacore.get_position_from_data(fine_data)),
                )
            ),
            "old_parameter_max_error": _max_abs(
                np.asarray(fine_flat)[rows] - np.asarray(coarse_flat)
            ),
            "old_sr_o_max_error": _max_abs(
                np.asarray(fine_core.sr_o)[rows] - np.asarray(coarse_core.sr_o)
            ),
            "new_sr_o_max_abs": _max_abs(np.asarray(fine_core.sr_o)[new_rows]),
            "old_u_max_error": _max_abs(
                np.asarray(fine_core.u)[rows] - np.asarray(coarse_core.u)
            ),
            "new_u_max_abs": _max_abs(np.asarray(fine_core.u)[new_rows]),
            "ek_max_error": _max_abs(
                np.asarray(fine_core.ek) - np.asarray(coarse_core.ek)
            ),
            "sr_rank0_equal": bool(
                np.array_equal(
                    np.asarray(fine_core.sr_rank0),
                    np.asarray(coarse_core.sr_rank0),
                )
            ),
            "sr_rank_equal": bool(
                np.array_equal(
                    np.asarray(fine_core.sr_rank),
                    np.asarray(coarse_core.sr_rank),
                )
            ),
            "has_u_equal": bool(
                np.array_equal(
                    np.asarray(fine_core.has_u), np.asarray(coarse_core.has_u)
                )
            ),
            "coarse_schedule_count": int(
                np.asarray(coarse_state.optax_state[1].count)
            ),
            "fine_schedule_count": int(np.asarray(fine_state.optax_state[1].count)),
        }
        print(name, result[name], flush=True)
        del coarse, fine, coarse_params, fine_params, coarse_state, fine_state
        gc.collect()

    output = ROOT / "wssr_state_prolongation_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
