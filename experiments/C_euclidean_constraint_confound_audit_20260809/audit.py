#!/usr/bin/env python3
"""Read-only audit of Euclidean-step versus function-space displacement.

The code convention is O_cur in R^{N_param x N_sample}; hence the paper's
``O_bar p`` is evaluated here as ``O_cur.T @ p``.  O_cur is centered over
walkers and scaled by 1/sqrt(N), so its norm is the RMS log-wavefunction
displacement induced by the parameter direction.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from vmcnet.mcmc import position_amplitude_core as pacore
from vmcnet.physics import core as physics_core
from vmcnet.train import runners
from vmcnet.updates import spring, wssr
from vmcnet.utils import io


EPS = 1.0e-12


def scalar(value) -> float:
    return float(jax.device_get(value))


def optional_float(config, name: str, fallback: float) -> float:
    value = config.get(name, fallback)
    return float(fallback if value is None else value)


def summary(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("wssr", "spring"), required=True)
    parser.add_argument("--config-run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument(
        "--direction-source", choices=("stored", "recompute"), default="stored"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.batches < 1:
        raise ValueError("batches must be positive")

    config = io.load_config_dict(str(args.config_run), "config.json")
    first = io.reload_vmc_state(
        str(args.checkpoint[0].parent), args.checkpoint[0].name
    )
    _, first_data, first_params, _, first_key = first
    nchains = int(config.vmc.nchains)
    dtype = runners._get_dtype(config)
    ions, charges, nelec = runners._get_electron_ion_config_as_arrays(config, dtype)
    initial_positions = pacore.get_position_from_data(first_data)
    log_psi, _, _ = runners._get_and_init_model(
        config.model,
        ions,
        charges,
        nelec,
        initial_positions,
        first_key,
        dtype=dtype,
        apply_pmap=False,
    )
    local_energy = runners._assemble_mol_local_energy_fn(
        ions,
        charges,
        config.problem.ei_softening,
        config.problem.ee_softening,
        log_psi,
    )
    energy_fn = jax.jit(
        physics_core.create_energy_and_statistics_fn(
            local_energy,
            nchains,
            runners._get_clipping_fn(config.vmc),
            config.vmc.nan_safe,
            record_raw_local_energies=True,
        )
    )
    _, walker_fn = runners._get_mcmc_fns(
        config.vmc, log_psi, apply_pmap=False
    )

    if args.method == "wssr":
        if config.vmc.optimizer_type != "wssr_warm_svd_right":
            raise ValueError(config.vmc.optimizer_type)
        opt = config.vmc.optimizer.wssr_warm_svd_right
        working_rank = min(int(opt.svd_working_rank), int(opt.sr_rank_max))
        relative_cutoff = optional_float(opt, "relative_singular_value_cutoff", -1.0)
        tikhonov_lambda = optional_float(opt, "tikhonov_lambda", -1.0)

        def direction_fn(score, epsilon, direction_key, core_state):
            augmented, rhs = wssr.augment_wssr_system(
                score, epsilon, core_state, float(opt.eta)
            )
            u, singular_values, vh, _ = wssr._wssr_right_svd_decomposition(
                augmented,
                core_state,
                direction_key,
                working_rank,
                int(opt.svd_maxiter_initial),
                int(opt.svd_maxiter_warm),
                bool(opt.exact_first),
                EPS,
                exact_first_force=bool(opt.get("exact_first_force", False)),
            )
            result = wssr._wssr_update_from_svd(
                augmented,
                rhs,
                core_state,
                u,
                singular_values,
                vh,
                float(opt.damping),
                float(opt.norm_constraint),
                int(opt.sr_rank_max),
                float(opt.sr_scale),
                False,
                rank_update_max=working_rank,
                spectral_regularization=str(opt.spectral_regularization),
                complement_weight=float(opt.complement_weight),
                relative_singular_value_cutoff=relative_cutoff,
                tikhonov_lambda=tikhonov_lambda,
                experimental_mode="none",
                eps=EPS,
            )
            return result.grad_like_update

        direction_jit = jax.jit(direction_fn)

        def compute_direction(params, positions, centered, state, key):
            score, _ = wssr.center_and_scale_score_matrix(
                log_psi, params, positions
            )
            epsilon = centered / jnp.sqrt(nchains)
            return direction_jit(score, epsilon, key, state.core_state)

        def stored_direction(state):
            """Recover the preceding raw WSSR solution from its saved SVD factor."""
            core_state = state.core_state
            squared_singular_values = jnp.sum(jnp.square(core_state.sr_o), axis=0)
            leading_singular_value = jnp.sqrt(
                jnp.maximum(squared_singular_values[0], EPS)
            )
            configured_lambda = optional_float(opt, "tikhonov_lambda", -1.0)
            lambda_reg = jnp.where(
                configured_lambda >= 0.0,
                configured_lambda,
                jnp.square(float(opt.damping) * leading_singular_value),
            )
            active = (
                jnp.arange(core_state.sr_o.shape[1]) < core_state.sr_rank0
            ).astype(core_state.sr_o.dtype)
            coefficients = (
                active
                * core_state.ek
                / jnp.maximum(squared_singular_values + lambda_reg, EPS)
            )
            return core_state.sr_o @ coefficients

    else:
        if config.vmc.optimizer_type != "spring":
            raise ValueError(config.vmc.optimizer_type)
        opt = config.vmc.optimizer.spring
        spring_step = spring.get_spring_step(log_psi, opt.damping, opt.mu)

        def compute_direction(params, positions, centered, state, key):
            del key
            previous_direction = state[0].trace
            tree = spring_step(centered, params, previous_direction, positions)
            return jax.flatten_util.ravel_pytree(tree)[0]

        def stored_direction(state):
            return jax.flatten_util.ravel_pytree(state[0].trace)[0]

    learning_rate = float(opt.learning_rate)
    learning_decay = float(opt.learning_decay_rate)
    norm_constraint = float(opt.norm_constraint)
    constrain_norm = bool(opt.constrain_norm)
    records = []
    started = time.perf_counter()

    for checkpoint in args.checkpoint:
        stored_epoch, base_data, params, optimizer_state, base_key = (
            io.reload_vmc_state(str(checkpoint.parent), checkpoint.name)
        )
        filename_epoch = int(checkpoint.stem)
        if int(stored_epoch) != filename_epoch - 1:
            raise ValueError(
                f"{checkpoint}: stored epoch {stored_epoch} != {filename_epoch - 1}"
            )
        data = base_data
        key = base_key
        for batch in range(args.batches):
            acceptance, data, key = walker_fn(params, data, key)
            positions = pacore.get_position_from_data(data)
            if args.direction_source == "recompute":
                energy, clipped, stats = energy_fn(params, positions)
                centered = clipped - energy
                key, direction_key = jax.random.split(key)
                direction = compute_direction(
                    params, positions, centered, optimizer_state, direction_key
                )
                energy_value = scalar(energy)
                variance_value = scalar(
                    jnp.var(stats["local_energies_noclip"], ddof=1)
                )
                # This is the schedule count for the next, hypothetical update.
                schedule_step = int(stored_epoch) + 1
            else:
                direction = stored_direction(optimizer_state)
                energy_value = None
                variance_value = None
                # This direction produced the saved checkpoint's last update.
                schedule_step = int(stored_epoch)
            direction_norm = jnp.linalg.norm(direction)
            unit_direction = direction / jnp.maximum(direction_norm, EPS)
            # One exact JVP; no dense Fisher/S matrix is constructed.
            function_action = wssr.score_rmatvec_current(
                log_psi, params, positions, unit_direction
            )
            function_gain = jnp.linalg.norm(function_action)
            rho_inverse_gain = 1.0 / jnp.maximum(function_gain, EPS)

            lr_current = learning_rate / (1.0 + learning_decay * schedule_step)
            lr_update_norm = lr_current * direction_norm
            if constrain_norm:
                constraint_scale = jnp.minimum(
                    1.0,
                    jnp.sqrt(norm_constraint)
                    / jnp.maximum(lr_update_norm, EPS),
                )
            else:
                constraint_scale = jnp.asarray(1.0)
            actual_update_norm = constraint_scale * lr_update_norm
            actual_function_step = actual_update_norm * function_gain
            jax.block_until_ready(actual_function_step)

            row = {
                "method": args.method,
                "checkpoint": str(checkpoint),
                "epoch": filename_epoch,
                "batch": batch,
                "acceptance": scalar(acceptance),
                "direction_source": args.direction_source,
                "energy": energy_value,
                "raw_variance": variance_value,
                "direction_norm": scalar(direction_norm),
                "function_gain_O_p_hat": scalar(function_gain),
                "rho_inverse_function_gain": scalar(rho_inverse_gain),
                "learning_rate": lr_current,
                "lr_scaled_update_norm": scalar(lr_update_norm),
                "constraint_scale": scalar(constraint_scale),
                "constraint_active": scalar(constraint_scale) < 1.0 - 1.0e-7,
                "actual_update_norm": scalar(actual_update_norm),
                "actual_function_step_norm": scalar(actual_function_step),
            }
            if not all(
                value is None
                or isinstance(value, (str, bool))
                or math.isfinite(value)
                for value in row.values()
            ):
                raise FloatingPointError(row)
            records.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    per_epoch = {}
    for epoch in sorted({row["epoch"] for row in records}):
        rows = [row for row in records if row["epoch"] == epoch]
        per_epoch[str(epoch)] = {
            "function_gain_O_p_hat": summary(
                [row["function_gain_O_p_hat"] for row in rows]
            ),
            "rho_inverse_function_gain": summary(
                [row["rho_inverse_function_gain"] for row in rows]
            ),
            "actual_function_step_norm": summary(
                [row["actual_function_step_norm"] for row in rows]
            ),
            "constraint_active_fraction": float(
                np.mean([row["constraint_active"] for row in rows])
            ),
            "constraint_scale": summary([row["constraint_scale"] for row in rows]),
        }

    payload = {
        "experiment": "C Euclidean norm-constraint confound audit",
        "method": args.method,
        "direction_source": args.direction_source,
        "read_only": True,
        "parameters_updated": False,
        "optimizer_state_updated": False,
        "score_convention": (
            "code O_cur has shape N_param x N_sample, is walker-centered and "
            "scaled by 1/sqrt(N); paper O_bar @ p equals code O_cur.T @ p"
        ),
        "walkers": nchains,
        "mcmc_steps_per_batch": int(config.vmc.nsteps_per_param_update),
        "batches_per_checkpoint": args.batches,
        "optimizer_config": {
            "optimizer_type": str(config.vmc.optimizer_type),
            "learning_rate": learning_rate,
            "learning_decay_rate": learning_decay,
            "norm_constraint_squared_radius": norm_constraint,
            "euclidean_radius": math.sqrt(norm_constraint),
            "constrain_norm": constrain_norm,
        },
        "aggregate": {
            "function_gain_O_p_hat": summary(
                [row["function_gain_O_p_hat"] for row in records]
            ),
            "rho_inverse_function_gain": summary(
                [row["rho_inverse_function_gain"] for row in records]
            ),
            "actual_function_step_norm": summary(
                [row["actual_function_step_norm"] for row in records]
            ),
            "constraint_active_fraction": float(
                np.mean([row["constraint_active"] for row in records])
            ),
        },
        "per_epoch": per_epoch,
        "records": records,
        "elapsed_seconds": time.perf_counter() - started,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
