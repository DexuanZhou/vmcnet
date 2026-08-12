"""Measure how quickly WSSR's padded basis enters newly added coordinates."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import jax.flatten_util
import numpy as np

from vmcnet.models import multilevel
from vmcnet.utils import io


SOURCE_ROOT = Path(
    "/scratch/dexuan1/runs/C_ferminet_multilevel_fair_scale001_20260808"
)
ROOT = Path("/scratch/dexuan1/runs/C_wssr_transition_basis_refresh_20260808")


def _load(run: Path, checkpoint: str):
    return io.reload_vmc_state(str(run), checkpoint)


def _fraction_in_new_rows(array: np.ndarray, new_rows: np.ndarray) -> float:
    square = np.square(np.asarray(array, dtype=np.float64))
    total = float(np.sum(square))
    if total == 0.0:
        return 0.0
    return float(np.sum(square[new_rows])) / total


def _state_metrics(state, new_rows: np.ndarray) -> dict[str, float | int]:
    core = state.core_state
    rank = int(np.asarray(core.sr_rank))
    u = np.asarray(core.u)[:, :rank]
    sr_o = np.asarray(core.sr_o)[:, :rank]
    return {
        "active_rank": rank,
        "u_new_coordinate_mass": _fraction_in_new_rows(u, new_rows),
        "sr_o_new_coordinate_mass": _fraction_in_new_rows(sr_o, new_rows),
    }


def _parameter_metrics(params, reference_flat, new_rows):
    flat, _ = jax.flatten_util.ravel_pytree(params)
    displacement = np.asarray(flat) - reference_flat
    return {
        "parameter_displacement_norm": float(np.linalg.norm(displacement)),
        "parameter_displacement_new_coordinate_mass": _fraction_in_new_rows(
            displacement, new_rows
        ),
    }


def main() -> None:
    output = {}
    for replicate in (0, 1):
        source = SOURCE_ROOT / "three_level" / f"seed{replicate}"
        coarse_run = source / "train_L1_to1000"
        converted_run = source / "converted_L2"
        coarse_config = io.load_config_dict(str(coarse_run), "config.json")
        fine_config = io.load_config_dict(str(converted_run), "config.json")
        _, _, coarse_params, _, _ = _load(
            coarse_run / "checkpoints", "1000.npz"
        )
        _, _, fine_params, fine_state, _ = _load(converted_run, "multilevel.npz")
        fine_flat, _ = jax.flatten_util.ravel_pytree(fine_params)
        fine_flat = np.asarray(fine_flat)
        rows = np.asarray(
            multilevel.ferminet_flat_prolongation_indices(
                coarse_params,
                fine_params,
                coarse_config.model,
                fine_config.model,
                nspins=2,
            )
        )
        new_rows = np.ones(fine_flat.shape[0], dtype=bool)
        new_rows[rows] = False
        seed_result = {
            "num_parameters": int(fine_flat.shape[0]),
            "num_new_parameters": int(np.sum(new_rows)),
            "converted": _state_metrics(fine_state, new_rows),
        }
        del coarse_params, fine_params, fine_state
        gc.collect()

        for mode in ("warm2", "refresh40"):
            mode_root = ROOT / f"seed{replicate}" / mode
            mode_result = {}
            for label, run, checkpoint in (
                (
                    "after_one_step",
                    mode_root / "first_step" / "checkpoints",
                    "1001.npz",
                ),
                (
                    "after_200_steps",
                    mode_root / "continue_to1200" / "checkpoints",
                    "1200.npz",
                ),
            ):
                _, _, params, state, _ = _load(run, checkpoint)
                metrics = _state_metrics(state, new_rows)
                metrics.update(_parameter_metrics(params, fine_flat, new_rows))
                mode_result[label] = metrics
                del params, state
                gc.collect()
            seed_result[mode] = mode_result
        output[f"seed{replicate}"] = seed_result

    ROOT.mkdir(parents=True, exist_ok=True)
    output_path = ROOT / "transition_basis_audit.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
