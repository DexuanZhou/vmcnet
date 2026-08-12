#!/usr/bin/env python3
"""Read-only fixed-checkpoint direction-consistency audit for WSSR or SPRING."""

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


NCHAINS = 1000
EPS = 1.0e-12


def scalar(value) -> float:
    return float(jax.device_get(value))


def safe_cosine(left, right):
    return jnp.vdot(left, right) / jnp.maximum(
        jnp.linalg.norm(left) * jnp.linalg.norm(right), EPS
    )


def residual_ratio(target, prediction):
    return jnp.linalg.norm(target - prediction) / jnp.maximum(
        jnp.linalg.norm(target), EPS
    )


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("wssr", "spring"), required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--batches", type=int, default=30)
    parser.add_argument("--replicate-keys", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = args.run / "checkpoints" / f"{args.epoch}.npz"
    config = io.load_config_dict(str(args.run), "config.json")
    stored_epoch, data, params, optimizer_state, key = io.reload_vmc_state(
        str(checkpoint.parent), checkpoint.name
    )
    if int(stored_epoch) != args.epoch - 1:
        raise ValueError(f"stored epoch {stored_epoch} != {args.epoch - 1}")
    if int(config.vmc.nchains) != NCHAINS:
        raise ValueError(f"expected {NCHAINS} chains")
    if args.replicate_keys < 1:
        raise ValueError("replicate-keys must be positive")
    if args.method == "spring" and args.replicate_keys != 1:
        raise ValueError("replicate-key audit only applies to WSSR")

    dtype = runners._get_dtype(config)
    ions, charges, nelec = runners._get_electron_ion_config_as_arrays(config, dtype)
    positions = pacore.get_position_from_data(data)
    log_psi, _, _ = runners._get_and_init_model(
        config.model,
        ions,
        charges,
        nelec,
        positions,
        key,
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
            NCHAINS,
            runners._get_clipping_fn(config.vmc),
            config.vmc.nan_safe,
            record_raw_local_energies=True,
        )
    )
    _, walker_fn = runners._get_mcmc_fns(config.vmc, log_psi, apply_pmap=False)

    config_verified = {
        "optimizer_type": str(config.vmc.optimizer_type),
        "nchains": int(config.vmc.nchains),
        "mcmc_steps": int(config.vmc.nsteps_per_param_update),
        "clip_threshold": float(config.vmc.clip_threshold),
        "clip_center": str(config.vmc.clip_center),
    }
    fixed_anchor = None
    if args.method == "wssr":
        if config.vmc.optimizer_type != "wssr_warm_svd_right":
            raise ValueError(config.vmc.optimizer_type)
        opt = config.vmc.optimizer.wssr_warm_svd_right
        expected = {
            "rank": (int(opt.sr_rank), 1600),
            "eta": (float(opt.eta), 0.3),
            "learning_rate": (float(opt.learning_rate), 0.04),
            "lambda": (float(opt.tikhonov_lambda), 1.0e-3),
            "warm_iterations": (int(opt.svd_maxiter_warm), 2),
            "complement_weight": (float(opt.complement_weight), 0.0),
        }
        mismatches = {
            name: pair for name, pair in expected.items() if pair[0] != pair[1]
        }
        if mismatches:
            raise ValueError(f"unexpected WSSR config: {mismatches}")
        state = optimizer_state.core_state
        working_rank = min(int(opt.svd_working_rank), int(opt.sr_rank_max))

        def direction_fn(score, epsilon, direction_key, state_arg):
            augmented, rhs = wssr.augment_wssr_system(
                score, epsilon, state_arg, float(opt.eta)
            )
            u, singular_values, vh, _ = wssr._wssr_right_svd_decomposition(
                augmented,
                state_arg,
                direction_key,
                working_rank,
                int(opt.svd_maxiter_initial),
                int(opt.svd_maxiter_warm),
                bool(opt.exact_first),
                EPS,
                exact_first_force=False,
            )
            result = wssr._wssr_update_from_svd(
                augmented,
                rhs,
                state_arg,
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
                complement_weight=0.0,
                relative_singular_value_cutoff=float(
                    opt.relative_singular_value_cutoff
                ),
                tikhonov_lambda=float(opt.tikhonov_lambda),
                eps=EPS,
            )
            return result.grad_like_update

        direction_jit = jax.jit(direction_fn)

        def compute_direction(score, epsilon, centered, positions, direction_key):
            return direction_jit(score, epsilon, direction_key, state)

        config_verified.update({name: actual for name, (actual, _) in expected.items()})
    else:
        if config.vmc.optimizer_type != "spring":
            raise ValueError(config.vmc.optimizer_type)
        opt = config.vmc.optimizer.spring
        expected = {
            "mu": (float(opt.mu), 0.99),
            "learning_rate": (float(opt.learning_rate), 0.02),
            "damping": (float(opt.damping), 1.0e-3),
            "norm_constraint": (float(opt.norm_constraint), 1.0e-3),
        }
        mismatches = {
            name: pair for name, pair in expected.items() if pair[0] != pair[1]
        }
        if mismatches:
            raise ValueError(f"unexpected SPRING config: {mismatches}")
        spring_step = spring.get_spring_step(log_psi, opt.damping, opt.mu)
        previous_direction = optimizer_state[0].trace
        fixed_anchor = jax.flatten_util.ravel_pytree(
            jax.tree_util.tree_map(lambda x: opt.mu * x, previous_direction)
        )[0]

        def compute_direction(score, epsilon, centered, positions, direction_key):
            del score, epsilon, direction_key
            tree = spring_step(centered, params, previous_direction, positions)
            return jax.flatten_util.ravel_pytree(tree)[0]

        config_verified.update({name: actual for name, (actual, _) in expected.items()})

    records = []
    previous = None
    sum_direction = None
    sum_unit = None
    sum_squared_norm = 0.0
    started = time.perf_counter()
    for batch in range(args.batches):
        acceptance, data, key = walker_fn(params, data, key)
        positions = pacore.get_position_from_data(data)
        energy, clipped, stats = energy_fn(params, positions)
        centered = clipped - energy
        epsilon = centered / jnp.sqrt(NCHAINS)
        score, _ = wssr.center_and_scale_score_matrix(log_psi, params, positions)
        force = score @ epsilon
        split_keys = jax.random.split(key, args.replicate_keys + 1)
        key, direction_keys = split_keys[0], split_keys[1:]
        direction = compute_direction(
            score, epsilon, centered, positions, direction_keys[0]
        )
        jax.block_until_ready(direction)
        replicated_key_cosines = []
        for replicate_key in direction_keys[1:]:
            replicated = compute_direction(
                score, epsilon, centered, positions, replicate_key
            )
            jax.block_until_ready(replicated)
            replicated_key_cosines.append(scalar(safe_cosine(direction, replicated)))
        direction_norm = jnp.linalg.norm(direction)
        unit = direction / jnp.maximum(direction_norm, EPS)

        row = {
            "batch": batch,
            "acceptance": scalar(acceptance),
            "raw_variance": scalar(jnp.var(stats["local_energies_noclip"], ddof=1)),
            "direction_norm": scalar(direction_norm),
            "current_force_norm": scalar(jnp.linalg.norm(force)),
            "current_force_direction_cosine": scalar(safe_cosine(force, direction)),
            "current_q": scalar(residual_ratio(epsilon, score.T @ direction)),
            "consecutive_direction_cosine": None,
            "previous_direction_heldout_force_cosine": None,
            "previous_direction_heldout_q": None,
            "fixed_anchor_direction_cosine": (
                scalar(safe_cosine(fixed_anchor, direction))
                if fixed_anchor is not None
                else None
            ),
            "same_batch_replicate_key_cosine_mean": (
                float(np.mean(replicated_key_cosines))
                if replicated_key_cosines
                else None
            ),
            "same_batch_replicate_key_cosine_min": (
                float(np.min(replicated_key_cosines))
                if replicated_key_cosines
                else None
            ),
        }
        if previous is not None:
            row["consecutive_direction_cosine"] = scalar(
                safe_cosine(previous, direction)
            )
            row["previous_direction_heldout_force_cosine"] = scalar(
                safe_cosine(force, previous)
            )
            row["previous_direction_heldout_q"] = scalar(
                residual_ratio(epsilon, score.T @ previous)
            )
        if not all(
            value is None or not isinstance(value, float) or math.isfinite(value)
            for value in row.values()
        ):
            raise FloatingPointError(row)
        records.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

        if sum_direction is None:
            sum_direction = direction
            sum_unit = unit
        else:
            sum_direction = sum_direction + direction
            sum_unit = sum_unit + unit
        sum_squared_norm += scalar(jnp.square(direction_norm))
        previous = direction

    count = len(records)
    mean_direction = sum_direction / count
    mean_norm = scalar(jnp.linalg.norm(mean_direction))
    rms_deviation = math.sqrt(
        max(sum_squared_norm / count - mean_norm * mean_norm, 0.0)
    )
    unit_resultant = scalar(jnp.linalg.norm(sum_unit) / count)
    mean_pairwise_cosine = (
        (scalar(jnp.vdot(sum_unit, sum_unit)) - count) / (count * (count - 1))
        if count > 1
        else math.nan
    )

    def present(name):
        return [row[name] for row in records if row[name] is not None]

    heldout_cosines = present("previous_direction_heldout_force_cosine")
    payload = {
        "experiment": f"C {args.method} fixed-checkpoint direction consistency",
        "method": args.method,
        "read_only_checkpoint": True,
        "parameters_updated": False,
        "optimizer_state_updated": False,
        "source_checkpoint": str(checkpoint),
        "stored_epoch": int(stored_epoch),
        "batches": args.batches,
        "replicate_keys": args.replicate_keys,
        "walkers_per_batch": NCHAINS,
        "config_verified": config_verified,
        "aggregate": {
            "direction_norm": summarize(present("direction_norm")),
            "current_force_direction_cosine": summarize(
                present("current_force_direction_cosine")
            ),
            "current_q": summarize(present("current_q")),
            "consecutive_direction_cosine": summarize(
                present("consecutive_direction_cosine")
            ),
            "previous_direction_heldout_force_cosine": summarize(heldout_cosines),
            "previous_direction_heldout_q": summarize(
                present("previous_direction_heldout_q")
            ),
            "positive_heldout_force_cosine_fraction": float(
                np.mean(np.asarray(heldout_cosines) > 0.0)
            ),
            "raw_direction_mean_norm": mean_norm,
            "raw_direction_rms_deviation": rms_deviation,
            "raw_direction_snr": mean_norm / max(rms_deviation, EPS),
            "unit_direction_resultant_length": unit_resultant,
            "mean_pairwise_direction_cosine": mean_pairwise_cosine,
            "same_batch_replicate_key_cosine_mean": (
                summarize(present("same_batch_replicate_key_cosine_mean"))
                if args.replicate_keys > 1
                else None
            ),
            "same_batch_replicate_key_cosine_min": (
                summarize(present("same_batch_replicate_key_cosine_min"))
                if args.replicate_keys > 1
                else None
            ),
        },
        "per_batch": records,
        "finite": True,
        "elapsed_seconds": time.perf_counter() - started,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
