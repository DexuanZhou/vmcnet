"""Convert a coarse FermiNet checkpoint into a strictly nested fine checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from vmcnet.mcmc import position_amplitude_core as pacore
from vmcnet.models import construct, multilevel as model_multilevel
from vmcnet.train import runners
from vmcnet.updates import multilevel as update_multilevel
from vmcnet.utils import io


def _load_config(path: str):
    config_path = Path(path).resolve()
    return io.load_config_dict(str(config_path.parent), config_path.name)


def _verify_same_problem(coarse_config: Any, fine_config: Any) -> None:
    if coarse_config.problem.to_dict() != fine_config.problem.to_dict():
        raise ValueError("coarse and fine checkpoints must describe the same problem")
    if coarse_config.dtype != fine_config.dtype:
        raise ValueError("coarse and fine checkpoints must use the same dtype")
    if coarse_config.vmc.optimizer_type != fine_config.vmc.optimizer_type:
        raise ValueError("optimizer type cannot change during state prolongation")


def _make_model_and_template(config: Any, positions: Any, seed: int):
    dtype = runners._get_dtype(config)
    ion_pos, ion_charges, nelec = runners._get_electron_ion_config_as_arrays(
        config, dtype=dtype
    )
    model = construct.get_model_from_config(
        config.model, nelec, ion_pos, ion_charges, dtype=dtype
    )
    template = model.init(jax.random.PRNGKey(seed), jnp.asarray(positions[:1]))
    return model, template, nelec


def convert_checkpoint(
    coarse_config: Any,
    fine_config: Any,
    checkpoint_data: Any,
    *,
    template_seed: int,
    verify_samples: int = 8,
    sidecar_scale: float = 1.0,
    equality_atol: float | None = None,
):
    """Return a fine-shaped checkpoint and a numerical conversion report."""
    _verify_same_problem(coarse_config, fine_config)
    epoch, data, coarse_params, coarse_optimizer_state, key = checkpoint_data
    positions = jnp.asarray(pacore.get_position_from_data(data))
    coarse_model, _, nelec = _make_model_and_template(
        coarse_config, positions, template_seed
    )
    fine_model, fine_template, _ = _make_model_and_template(
        fine_config, positions, template_seed
    )
    nspins = int(len(nelec))
    fine_params = model_multilevel.prolongate_ferminet_params(
        coarse_params,
        fine_template,
        coarse_config.model,
        fine_config.model,
        nspins=nspins,
        sidecar_scale=sidecar_scale,
    )

    optimizer_type = fine_config.vmc.optimizer_type
    if optimizer_type == "spring":
        fine_optimizer_state = update_multilevel.prolongate_spring_optimizer_state(
            coarse_optimizer_state,
            fine_params,
            coarse_config.model,
            fine_config.model,
            nspins=nspins,
        )
    elif optimizer_type.startswith("wssr"):
        fine_optimizer_state = update_multilevel.prolongate_wssr_optimizer_state(
            coarse_optimizer_state,
            coarse_params,
            fine_params,
            coarse_config.model,
            fine_config.model,
            nspins=nspins,
        )
    else:
        raise ValueError(
            "checkpoint state prolongation currently supports SPRING and WSSR only"
        )

    sample_count = min(int(verify_samples), positions.shape[0])
    verify_positions = positions[:sample_count]
    coarse_sign, coarse_logabs = coarse_model.apply(coarse_params, verify_positions)
    fine_sign, fine_logabs = fine_model.apply(fine_params, verify_positions)
    log_difference = np.asarray(fine_logabs - coarse_logabs)
    sign_mismatches = int(
        np.count_nonzero(np.asarray(fine_sign) != np.asarray(coarse_sign))
    )
    max_logabs_error = float(np.max(np.abs(log_difference)))
    # A wider GEMM includes extra, exactly-zero products and can use a different
    # accumulation order.  The embedding is algebraically exact, while fp32
    # log-amplitudes can differ by a few ulps at production widths.
    if equality_atol is None:
        equality_atol = 1e-5 if fine_logabs.dtype == jnp.float32 else 1e-10
    equality_atol = float(equality_atol)
    if (
        sign_mismatches
        or not np.isfinite(max_logabs_error)
        or max_logabs_error > equality_atol
    ):
        raise ValueError("prolonged checkpoint failed wavefunction equality")

    # Keep the walker locations and move metadata, but refresh cached amplitudes
    # with the fine numerical graph.  Algebraically they are unchanged; this
    # prevents a few fp32 accumulation-order ulps from entering the next
    # Metropolis acceptance ratio.  No burn-in or resampling is performed.
    _, fine_cached_logabs = fine_model.apply(fine_params, positions)
    old_cached_logabs = jnp.asarray(pacore.get_amplitude_from_data(data))
    fine_data = pacore.make_position_amplitude_data(
        positions, fine_cached_logabs, data["move_metadata"]
    )

    coarse_size = sum(
        leaf.size for leaf in jax.tree_util.tree_leaves(coarse_params)
    )
    fine_size = sum(leaf.size for leaf in jax.tree_util.tree_leaves(fine_params))
    report = {
        "epoch": int(epoch),
        "optimizer_type": optimizer_type,
        "template_seed": int(template_seed),
        "sidecar_scale": float(sidecar_scale),
        "verify_samples": sample_count,
        "sign_mismatches": sign_mismatches,
        "max_logabs_error": max_logabs_error,
        "equality_atol": equality_atol,
        "coarse_parameter_count": int(coarse_size),
        "fine_parameter_count": int(fine_size),
        "parameter_growth_ratio": float(fine_size / coarse_size),
        "max_cached_logabs_refresh": float(
            np.max(np.abs(np.asarray(fine_cached_logabs - old_cached_logabs)))
        ),
    }
    return (
        epoch,
        fine_data,
        fine_params,
        fine_optimizer_state,
        key,
    ), report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-config", required=True)
    parser.add_argument("--fine-config", required=True)
    parser.add_argument("--input-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", default="multilevel.npz")
    parser.add_argument("--template-seed", type=int, default=314159)
    parser.add_argument("--verify-samples", type=int, default=8)
    parser.add_argument("--sidecar-scale", type=float, default=1.0)
    parser.add_argument("--equality-atol", type=float)
    args = parser.parse_args()

    coarse_config = _load_config(args.coarse_config)
    fine_config = _load_config(args.fine_config)
    checkpoint_path = Path(args.input_checkpoint).resolve()
    checkpoint_data = io.reload_vmc_state(
        str(checkpoint_path.parent), checkpoint_path.name
    )
    converted, report = convert_checkpoint(
        coarse_config,
        fine_config,
        checkpoint_data,
        template_seed=args.template_seed,
        verify_samples=args.verify_samples,
        sidecar_scale=args.sidecar_scale,
        equality_atol=args.equality_atol,
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    io.save_vmc_state(str(output_dir), args.output_name, converted)
    io.save_config_dict_to_json(fine_config, str(output_dir), "config")
    with open(output_dir / "multilevel_conversion_report.json", "w") as report_file:
        json.dump(report, report_file, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
