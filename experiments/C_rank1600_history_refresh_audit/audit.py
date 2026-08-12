#!/usr/bin/env python3
"""Read-only one-step stale-versus-refreshed WSSR history audit for carbon.

Each replicate starts from the identical checkpoint.  It advances the walkers,
records the score matrix and residual, and applies exactly one in-memory WSSR
parameter update.  On the *same electronic coordinates*, the score matrix and
residual are recomputed at the updated parameters.  These two otherwise matched
history blocks are then compared on a new current batch and a third held-out
batch while the updated parameters remain fixed.

Nothing is written back to the source run and no training checkpoint is saved.
"""

from __future__ import annotations

import argparse
import gc
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
from vmcnet.updates import parse_optimizer_config, wssr
from vmcnet.utils import io


NCHAINS = 1000
ETA = 0.3
EPS = 1.0e-12
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260802


def scalar(value) -> float:
    return float(jax.device_get(value))


def tree_l2_difference(left, right) -> float:
    leaves = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(lambda x, y: x - y, left, right)
    )
    return scalar(jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves)))


def safe_cosine(left, right):
    return jnp.vdot(left, right) / jnp.maximum(
        jnp.linalg.norm(left) * jnp.linalg.norm(right), EPS
    )


def residual_ratio(target, prediction):
    return jnp.linalg.norm(target - prediction) / jnp.maximum(
        jnp.linalg.norm(target), EPS
    )


def bootstrap_median_ci(values, seed_offset=0):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    medians = np.median(values[draws], axis=1)
    return [float(x) for x in np.quantile(medians, [0.025, 0.975])]


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def optional_float(config, name, default):
    value = config.get(name, default)
    return float(default if value is None else value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--replicates", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = args.run / "checkpoints" / f"{args.epoch}.npz"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config = io.load_config_dict(str(args.run), "config.json")
    stored_epoch, base_data, base_params, base_optimizer_state, base_key = (
        io.reload_vmc_state(str(checkpoint.parent), checkpoint.name)
    )
    if int(stored_epoch) != args.epoch - 1:
        raise ValueError(f"stored epoch {stored_epoch} != {args.epoch - 1}")
    if config.vmc.optimizer_type != "wssr_warm_svd_right":
        raise ValueError(config.vmc.optimizer_type)
    opt = config.vmc.optimizer.wssr_warm_svd_right
    expected = {
        "nchains": (int(config.vmc.nchains), NCHAINS),
        "rank": (int(opt.sr_rank), 1600),
        "rank_max": (int(opt.sr_rank_max), 1600),
        "eta": (float(opt.eta), ETA),
        "learning_rate": (float(opt.learning_rate), 0.04),
        "warm_iterations": (int(opt.svd_maxiter_warm), 2),
        "complement_weight": (float(opt.complement_weight), 0.0),
    }
    mismatches = {
        name: pair for name, pair in expected.items() if pair[0] != pair[1]
    }
    if mismatches:
        raise ValueError(f"unexpected source configuration: {mismatches}")

    dtype = runners._get_dtype(config)
    ions, charges, nelec = runners._get_electron_ion_config_as_arrays(config, dtype)
    initial_positions = pacore.get_position_from_data(base_data)
    log_psi, _, _ = runners._get_and_init_model(
        config.model,
        ions,
        charges,
        nelec,
        initial_positions,
        base_key,
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
    clipping_fn = runners._get_clipping_fn(config.vmc)
    diagnostic_energy_fn = jax.jit(
        physics_core.create_energy_and_statistics_fn(
            local_energy,
            NCHAINS,
            clipping_fn,
            config.vmc.nan_safe,
            record_raw_local_energies=True,
        )
    )
    _, walker_fn = runners._get_mcmc_fns(config.vmc, log_psi, apply_pmap=False)
    update_data_fn = pacore.get_update_data_fn(log_psi)
    update_param_fn, _, _ = parse_optimizer_config.initialize_optimizer(
        log_psi,
        local_energy,
        clipping_fn,
        config.vmc,
        base_params,
        base_data,
        pacore.get_position_from_data,
        update_data_fn,
        base_key,
        apply_pmap=False,
    )

    damping = float(opt.damping)
    relative_cutoff = optional_float(
        opt, "relative_singular_value_cutoff", -1.0
    )
    tikhonov_lambda = optional_float(opt, "tikhonov_lambda", -1.0)
    spectral_regularization = str(opt.spectral_regularization)
    sr_scale = float(opt.get("sr_scale", 1.1))
    raw_history_width = NCHAINS

    def evaluate_batch(params, data):
        positions = pacore.get_position_from_data(data)
        energy, clipped, stats = diagnostic_energy_fn(params, positions)
        score, _ = wssr.center_and_scale_score_matrix(log_psi, params, positions)
        epsilon = wssr.center_and_scale_energy_residuals(clipped, energy)
        return score, epsilon, stats["local_energies_noclip"]

    def raw_history_state(template, score, epsilon, active):
        if score.shape[1] != raw_history_width:
            raise ValueError(score.shape)
        rank_dtype = template.sr_rank.dtype
        return type(template)(
            sr_o=score,
            ek=epsilon,
            sr_rank0=jnp.asarray(raw_history_width if active else 0, dtype=rank_dtype),
            sr_rank=jnp.asarray(raw_history_width, dtype=rank_dtype),
            u=template.u[:, :raw_history_width],
            has_u=template.has_u,
        )

    def make_candidate(score, epsilon, candidate_key, history_state):
        augmented, augmented_rhs = wssr.augment_wssr_system(
            score, epsilon, history_state, ETA
        )
        working_rank = history_state.sr_o.shape[1]
        u, singular_values, vh, approx_rank = wssr._wssr_right_svd_decomposition(
            augmented,
            history_state,
            candidate_key,
            working_rank,
            int(opt.svd_maxiter_initial),
            int(opt.svd_maxiter_warm),
            bool(opt.get("exact_first", False)),
            EPS,
            exact_first_force=False,
        )
        result = wssr._wssr_update_from_svd(
            augmented,
            augmented_rhs,
            history_state,
            u,
            singular_values,
            vh,
            damping,
            float(opt.norm_constraint),
            working_rank,
            sr_scale,
            False,
            rank_update_max=working_rank,
            spectral_regularization=spectral_regularization,
            complement_weight=0.0,
            relative_singular_value_cutoff=relative_cutoff,
            tikhonov_lambda=tikhonov_lambda,
            eps=EPS,
        )
        prediction = score.T @ result.grad_like_update
        return (
            result.grad_like_update,
            residual_ratio(epsilon, prediction),
            result.active_rank,
            approx_rank,
        )

    make_candidate_jit = jax.jit(make_candidate)
    rows = []
    started = time.perf_counter()
    for replicate in range(args.replicates):
        key = jax.random.fold_in(base_key, replicate + 1)
        acceptance_old, old_data, key = walker_fn(base_params, base_data, key)
        old_score, old_epsilon, old_raw = evaluate_batch(base_params, old_data)

        updated_params, updated_data, updated_optimizer_state, _, key = (
            update_param_fn(
                base_params, old_data, base_optimizer_state, key
            )
        )
        jax.block_until_ready(jax.tree_util.tree_leaves(updated_params)[0])
        displacement_norm = tree_l2_difference(updated_params, base_params)

        refreshed_score, refreshed_epsilon, refreshed_raw = evaluate_batch(
            updated_params, updated_data
        )
        acceptance_current, current_data, key = walker_fn(
            updated_params, updated_data, key
        )
        current_score, current_epsilon, current_raw = evaluate_batch(
            updated_params, current_data
        )

        template = base_optimizer_state.core_state
        stale_state = raw_history_state(template, old_score, old_epsilon, True)
        refreshed_state = raw_history_state(
            template, refreshed_score, refreshed_epsilon, True
        )
        current_only_state = raw_history_state(template, old_score, old_epsilon, False)
        key, candidate_key = jax.random.split(key)
        candidates = {}
        current_metrics = {}
        for name, state in (
            ("current_only", current_only_state),
            ("stale_last_batch", stale_state),
            ("refreshed_last_batch", refreshed_state),
        ):
            update, current_q, active_rank, approx_rank = make_candidate_jit(
                current_score, current_epsilon, candidate_key, state
            )
            jax.block_until_ready(update)
            force = current_score @ current_epsilon
            candidates[name] = update
            current_metrics[name] = {
                "current_q": scalar(current_q),
                "current_force_dot_update": scalar(jnp.vdot(force, update)),
                "current_force_update_cosine": scalar(safe_cosine(force, update)),
                "update_norm": scalar(jnp.linalg.norm(update)),
                "active_rank": int(jax.device_get(active_rank)),
                "approx_rank": int(jax.device_get(approx_rank)),
            }

        acceptance_heldout, heldout_data, key = walker_fn(
            updated_params, current_data, key
        )
        heldout_score, heldout_epsilon, heldout_raw = evaluate_batch(
            updated_params, heldout_data
        )
        heldout_force = heldout_score @ heldout_epsilon
        metrics = {}
        for name, update in candidates.items():
            prediction = heldout_score.T @ update
            metrics[name] = {
                **current_metrics[name],
                "heldout_q": scalar(residual_ratio(heldout_epsilon, prediction)),
                "heldout_force_dot_update": scalar(jnp.vdot(heldout_force, update)),
                "heldout_force_update_cosine": scalar(
                    safe_cosine(heldout_force, update)
                ),
            }

        old_force_on_refreshed_rhs = old_score @ refreshed_epsilon
        refreshed_force = refreshed_score @ refreshed_epsilon
        row = {
            "replicate": replicate,
            "parameter_displacement_norm": displacement_norm,
            "acceptance_old": scalar(acceptance_old),
            "acceptance_current": scalar(acceptance_current),
            "acceptance_heldout": scalar(acceptance_heldout),
            "old_raw_variance": scalar(jnp.var(old_raw, ddof=1)),
            "refreshed_same_coordinates_raw_variance": scalar(
                jnp.var(refreshed_raw, ddof=1)
            ),
            "current_raw_variance": scalar(jnp.var(current_raw, ddof=1)),
            "heldout_raw_variance": scalar(jnp.var(heldout_raw, ddof=1)),
            "same_coordinate_epsilon_relative_change": scalar(
                jnp.linalg.norm(refreshed_epsilon - old_epsilon)
                / jnp.maximum(jnp.linalg.norm(refreshed_epsilon), EPS)
            ),
            "same_rhs_score_action_relative_change": scalar(
                jnp.linalg.norm(refreshed_force - old_force_on_refreshed_rhs)
                / jnp.maximum(jnp.linalg.norm(refreshed_force), EPS)
            ),
            "same_rhs_score_action_cosine": scalar(
                safe_cosine(refreshed_force, old_force_on_refreshed_rhs)
            ),
            "candidates": metrics,
        }
        if not all(
            math.isfinite(value)
            for value in (
                row["parameter_displacement_norm"],
                row["same_coordinate_epsilon_relative_change"],
                row["same_rhs_score_action_relative_change"],
                row["same_rhs_score_action_cosine"],
            )
        ):
            raise FloatingPointError(row)
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

        del (
            old_data,
            old_score,
            old_epsilon,
            old_raw,
            updated_params,
            updated_data,
            updated_optimizer_state,
            refreshed_score,
            refreshed_epsilon,
            refreshed_raw,
            current_data,
            current_score,
            current_epsilon,
            current_raw,
            heldout_data,
            heldout_score,
            heldout_epsilon,
            heldout_raw,
            stale_state,
            refreshed_state,
            current_only_state,
            candidates,
            current_metrics,
            metrics,
            force,
            heldout_force,
            old_force_on_refreshed_rhs,
            refreshed_force,
        )
        gc.collect()

    stale_q = np.asarray(
        [row["candidates"]["stale_last_batch"]["heldout_q"] for row in rows]
    )
    refreshed_q = np.asarray(
        [row["candidates"]["refreshed_last_batch"]["heldout_q"] for row in rows]
    )
    current_q = np.asarray(
        [row["candidates"]["current_only"]["heldout_q"] for row in rows]
    )
    refresh_improvement = (stale_q - refreshed_q) / stale_q
    fixed_theta_improvement = (current_q - refreshed_q) / current_q
    refreshed_positive = np.asarray(
        [
            row["candidates"]["refreshed_last_batch"]["heldout_force_dot_update"]
            > 0.0
            for row in rows
        ]
    )
    stale_gate = {
        "median_relative_heldout_q_improvement": float(
            np.median(refresh_improvement)
        ),
        "bootstrap_95_ci": bootstrap_median_ci(refresh_improvement, 1),
        "win_fraction": float(np.mean(refresh_improvement > 0.0)),
        "refreshed_positive_heldout_force_fraction": float(
            np.mean(refreshed_positive)
        ),
    }
    stale_gate["passed"] = bool(
        stale_gate["median_relative_heldout_q_improvement"] >= 0.05
        and stale_gate["bootstrap_95_ci"][0] > 0.0
        and stale_gate["win_fraction"] >= 0.75
        and stale_gate["refreshed_positive_heldout_force_fraction"] >= 0.95
    )
    fixed_theta_gate = {
        "median_relative_heldout_q_improvement": float(
            np.median(fixed_theta_improvement)
        ),
        "bootstrap_95_ci": bootstrap_median_ci(fixed_theta_improvement, 2),
        "win_fraction": float(np.mean(fixed_theta_improvement > 0.0)),
    }
    fixed_theta_gate["passed"] = bool(
        fixed_theta_gate["median_relative_heldout_q_improvement"] >= 0.05
        and fixed_theta_gate["bootstrap_95_ci"][0] > 0.0
        and fixed_theta_gate["win_fraction"] >= 0.75
    )

    payload = {
        "experiment": "C rank1600 one-step stale-versus-refreshed history audit",
        "read_only_checkpoint": True,
        "source_checkpoint": str(checkpoint),
        "stored_epoch": int(stored_epoch),
        "replicates": args.replicates,
        "parameters_persisted": False,
        "optimizer_state_persisted": False,
        "training_checkpoint_written": False,
        "walkers_per_batch": NCHAINS,
        "eta": ETA,
        "raw_history_width": raw_history_width,
        "config_verified": {name: actual for name, (actual, _) in expected.items()},
        "spectral_controls": {
            "damping": damping,
            "relative_singular_value_cutoff": relative_cutoff,
            "tikhonov_lambda": tikhonov_lambda,
            "spectral_regularization": spectral_regularization,
        },
        "pre_registered_staleness_gate": {
            "median_improvement_min": 0.05,
            "bootstrap_lower_must_exceed": 0.0,
            "win_fraction_min": 0.75,
            "positive_force_fraction_min": 0.95,
        },
        "staleness_gate": stale_gate,
        "fixed_theta_multibatch_gate": fixed_theta_gate,
        "stale_history_supported": stale_gate["passed"],
        "fixed_theta_multibatch_supported": fixed_theta_gate["passed"],
        "summaries": {
            "parameter_displacement_norm": summarize(
                [row["parameter_displacement_norm"] for row in rows]
            ),
            "same_coordinate_epsilon_relative_change": summarize(
                [row["same_coordinate_epsilon_relative_change"] for row in rows]
            ),
            "same_rhs_score_action_relative_change": summarize(
                [row["same_rhs_score_action_relative_change"] for row in rows]
            ),
            "same_rhs_score_action_cosine": summarize(
                [row["same_rhs_score_action_cosine"] for row in rows]
            ),
        },
        "per_replicate": rows,
        "elapsed_seconds": time.perf_counter() - started,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status_short": subprocess.check_output(
            ["git", "status", "--short"], text=True
        ).splitlines(),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
