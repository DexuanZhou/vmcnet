"""Measure the fp32 function discrepancy at the failed L15-to-L2 conversion."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from vmcnet.mcmc import position_amplitude_core as pacore
from vmcnet.models import multilevel
from vmcnet.train import multilevel_checkpoint
from vmcnet.utils import io


ROOT = Path(
    "/scratch/dexuan1/runs/C_ferminet_multilevel_fair_scale001_20260808/"
    "five_level/seed0"
)


def main():
    coarse_config = io.load_config_dict(str(ROOT / "train_L15_to1000"), "config.json")
    fine_config = io.load_config_dict(str(ROOT / "config_L2"), "config.json")
    _, data, coarse_params, _, _ = io.reload_vmc_state(
        str(ROOT / "train_L15_to1000/checkpoints"), "1000.npz"
    )
    positions = jnp.asarray(pacore.get_position_from_data(data))
    coarse_model, _, nelec = multilevel_checkpoint._make_model_and_template(
        coarse_config, positions, 314159
    )
    fine_model, fine_template, _ = multilevel_checkpoint._make_model_and_template(
        fine_config, positions, 314159
    )
    fine_params = multilevel.prolongate_ferminet_params(
        coarse_params,
        fine_template,
        coarse_config.model,
        fine_config.model,
        nspins=int(len(nelec)),
        sidecar_scale=0.01,
    )
    for count in (8, 128):
        coarse_sign, coarse_log = coarse_model.apply(coarse_params, positions[:count])
        fine_sign, fine_log = fine_model.apply(fine_params, positions[:count])
        error = np.abs(np.asarray(fine_log - coarse_log))
        print(
            {
                "samples": count,
                "sign_mismatches": int(
                    np.count_nonzero(np.asarray(coarse_sign) != np.asarray(fine_sign))
                ),
                "max_abs": float(np.max(error)),
                "median_abs": float(np.median(error)),
                "p99_abs": float(np.quantile(error, 0.99)),
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
