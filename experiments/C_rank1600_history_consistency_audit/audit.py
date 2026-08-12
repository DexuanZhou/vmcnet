#!/usr/bin/env python3
"""Read-only forward validation of the WSSR history weight on carbon.

For a fixed checkpoint, candidate directions are constructed from batch k for a
pre-registered grid of history weights.  Parameters and optimizer state are never
updated.  Each direction is scored on the independently advanced batch k+1.  This
separates fitting the current stochastic SR equations from generalizing to the next
sample from the same fixed wavefunction.
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
from vmcnet.updates import wssr
from vmcnet.utils import io


ETA_GRID = (0.0, 0.05, 0.1, 0.2, 0.3)
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


def bootstrap_ci(values, *, seed=20260802, replicates=10000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(replicates, len(values)))
    statistics = np.median(values[draws], axis=1)
    return [float(x) for x in np.quantile(statistics, [0.025, 0.975])]


def eta_name(eta: float) -> str:
    return f"eta_{eta:.2f}"


def optional(config, name, default):
    return config.get(name, default)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--transitions", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = args.run / "checkpoints" / f"{args.epoch}.npz"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config = io.load_config_dict(str(args.run), "config.json")
    stored_epoch, data, params, optimizer_state, key = io.reload_vmc_state(
        str(checkpoint.parent), checkpoint.name
    )
    if int(stored_epoch) != args.epoch - 1:
        raise ValueError(f"stored epoch {stored_epoch} != {args.epoch - 1}")
    opt = config.vmc.optimizer.wssr_warm_svd_right
    expected = {
        "nchains": (int(config.vmc.nchains), NCHAINS),
        "rank": (int(opt.sr_rank), 1600),
        "rank_max": (int(opt.sr_rank_max), 1600),
        "eta": (float(opt.eta), 0.3),
        "learning_rate": (float(opt.learning_rate), 0.04),
        "warm_iterations": (int(opt.svd_maxiter_warm), 2),
        "complement_weight": (float(opt.complement_weight), 0.0),
    }
    mismatches = {name: pair for name, pair in expected.items() if pair[0] != pair[1]}
    if mismatches:
        raise ValueError(f"unexpected source configuration: {mismatches}")

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
    state = optimizer_state.core_state

    damping = float(opt.damping)
    relative_cutoff = float(optional(opt, "relative_singular_value_cutoff", -1.0))
    tikhonov_lambda = float(optional(opt, "tikhonov_lambda", -1.0))
    spectral_regularization = str(opt.spectral_regularization)
    sr_scale = float(optional(opt, "sr_scale", 1.1))
    working_rank = min(
        int(optional(opt, "svd_working_rank", opt.sr_rank_max)),
        int(opt.sr_rank_max),
        state.sr_o.shape[1],
    )

    def make_candidate(score, epsilon, candidate_key, eta, state_arg):
        augmented, augmented_rhs = wssr.augment_wssr_system(
            score, epsilon, state_arg, eta
        )
        u, singular_values, vh, approx_rank = wssr._wssr_right_svd_decomposition(
            augmented,
            state_arg,
            candidate_key,
            working_rank,
            int(opt.svd_maxiter_initial),
            int(opt.svd_maxiter_warm),
            bool(optional(opt, "exact_first", False)),
            EPS,
            exact_first_force=False,
        )
        result = wssr._wssr_update_from_svd(
            augmented,
            augmented_rhs,
            state_arg,
            u,
            singular_values,
            vh,
            damping,
            float(opt.norm_constraint),
            int(opt.sr_rank_max),
            sr_scale,
            False,
            rank_update_max=working_rank,
            spectral_regularization=spectral_regularization,
            complement_weight=0.0,
            relative_singular_value_cutoff=relative_cutoff,
            tikhonov_lambda=tikhonov_lambda,
            eps=EPS,
        )
        current_prediction = score.T @ result.grad_like_update
        augmented_prediction = augmented.T @ result.grad_like_update
        return (
            result.grad_like_update,
            result.active_rank,
            residual_ratio(epsilon, current_prediction),
            residual_ratio(augmented_rhs, augmented_prediction),
            singular_values[0],
            approx_rank,
        )

    make_candidate_jit = jax.jit(make_candidate)

    def sample_batch(data, key):
        acceptance, data, key = walker_fn(params, data, key)
        positions = pacore.get_position_from_data(data)
        energy, clipped_energies, stats = energy_fn(params, positions)
        score, _ = wssr.center_and_scale_score_matrix(log_psi, params, positions)
        epsilon = wssr.center_and_scale_energy_residuals(clipped_energies, energy)
        raw = stats["local_energies_noclip"]
        raw_epsilon = (raw - jnp.mean(raw)) / jnp.sqrt(NCHAINS)
        return data, key, acceptance, score, epsilon, raw_epsilon, raw

    started = time.perf_counter()
    data, key, acceptance, score, epsilon, raw_epsilon, raw = sample_batch(data, key)
    rows = []
    for transition in range(args.transitions):
        key, candidate_key = jax.random.split(key)
        current_force = score @ epsilon
        history_mask = (
            jnp.arange(state.sr_o.shape[1]) < state.sr_rank0
        ).astype(score.dtype)
        pure_history_force = state.sr_o @ (state.ek * history_mask)

        candidates = {}
        current_metrics = {}
        for eta in ETA_GRID:
            candidate = make_candidate_jit(
                score,
                epsilon,
                candidate_key,
                jnp.asarray(eta, dtype=score.dtype),
                state,
            )
            update, active_rank, current_q, augmented_q, leading_sv, approx_rank = candidate
            # Materialize before the next candidate to keep peak device use bounded.
            jax.block_until_ready(update)
            name = eta_name(eta)
            candidates[name] = update
            current_metrics[name] = {
                "eta": eta,
                "current_q": scalar(current_q),
                "augmented_q": scalar(augmented_q),
                "update_norm": scalar(jnp.linalg.norm(update)),
                "current_force_dot_update": scalar(jnp.vdot(current_force, update)),
                "current_force_update_cosine": scalar(safe_cosine(current_force, update)),
                "active_rank": int(jax.device_get(active_rank)),
                "approx_rank": int(jax.device_get(approx_rank)),
                "leading_singular_value": scalar(leading_sv),
            }

        (
            data,
            key,
            next_acceptance,
            next_score,
            next_epsilon,
            next_raw_epsilon,
            next_raw,
        ) = sample_batch(data, key)
        next_force = next_score @ next_epsilon
        next_metrics = {}
        for eta in ETA_GRID:
            name = eta_name(eta)
            update = candidates[name]
            prediction = next_score.T @ update
            next_metrics[name] = {
                "next_q": scalar(residual_ratio(next_epsilon, prediction)),
                "next_raw_q": scalar(residual_ratio(next_raw_epsilon, prediction)),
                "next_force_dot_update": scalar(jnp.vdot(next_force, update)),
                "next_force_update_cosine": scalar(safe_cosine(next_force, update)),
            }

        row = {
            "transition": transition,
            "acceptance_current": scalar(acceptance),
            "acceptance_next": scalar(next_acceptance),
            "raw_variance_current": scalar(jnp.var(raw, ddof=1)),
            "raw_variance_next": scalar(jnp.var(next_raw, ddof=1)),
            "current_force_norm": scalar(jnp.linalg.norm(current_force)),
            "pure_history_force_norm": scalar(jnp.linalg.norm(pure_history_force)),
            "current_pure_history_force_cosine": scalar(
                safe_cosine(current_force, pure_history_force)
            ),
            "eta_metrics": {
                name: {**current_metrics[name], **next_metrics[name]}
                for name in current_metrics
            },
        }
        if not all(
            math.isfinite(value)
            for metrics in row["eta_metrics"].values()
            for value in metrics.values()
            if isinstance(value, float)
        ):
            raise FloatingPointError(f"nonfinite candidate metrics: {row}")
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

        acceptance = next_acceptance
        score, epsilon, raw_epsilon, raw = (
            next_score,
            next_epsilon,
            next_raw_epsilon,
            next_raw,
        )
        del candidates, current_metrics, next_metrics, current_force, next_force

    baseline = eta_name(0.3)
    comparisons = {}
    qualifying = []
    for index, eta in enumerate(ETA_GRID[:-1]):
        name = eta_name(eta)
        improvements = np.asarray(
            [
                (row["eta_metrics"][baseline]["next_q"] - row["eta_metrics"][name]["next_q"])
                / row["eta_metrics"][baseline]["next_q"]
                for row in rows
            ],
            dtype=np.float64,
        )
        positive_descent = np.asarray(
            [row["eta_metrics"][name]["next_force_dot_update"] > 0 for row in rows]
        )
        median_improvement = float(np.median(improvements))
        ci = bootstrap_ci(improvements, seed=20260802 + index)
        win_fraction = float(np.mean(improvements > 0.0))
        descent_fraction = float(np.mean(positive_descent))
        passed = (
            median_improvement >= 0.10
            and ci[0] > 0.0
            and win_fraction >= 0.70
            and descent_fraction >= 0.95
        )
        comparisons[name] = {
            "median_relative_next_q_improvement_vs_eta_0.30": median_improvement,
            "bootstrap_95_ci": ci,
            "win_fraction": win_fraction,
            "positive_next_force_dot_update_fraction": descent_fraction,
            "passed_pre_registered_gate": passed,
        }
        if passed:
            qualifying.append((median_improvement, eta))

    selected_eta = max(qualifying)[1] if qualifying else None
    cosines = np.asarray(
        [row["current_pure_history_force_cosine"] for row in rows], dtype=np.float64
    )
    payload = {
        "experiment": "C rank1600 WSSR forward held-out eta audit",
        "read_only_checkpoint": True,
        "parameters_updated": False,
        "optimizer_state_updated": False,
        "mcmc_walkers_advanced_in_memory": True,
        "source_checkpoint": str(checkpoint),
        "stored_epoch": int(stored_epoch),
        "transitions": args.transitions,
        "walkers_per_batch": NCHAINS,
        "mcmc_steps_between_batches": int(config.vmc.nsteps_per_param_update),
        "eta_grid": ETA_GRID,
        "config_verified": {name: actual for name, (actual, _) in expected.items()},
        "spectral_controls": {
            "damping": damping,
            "relative_singular_value_cutoff": relative_cutoff,
            "tikhonov_lambda": tikhonov_lambda,
            "spectral_regularization": spectral_regularization,
            "working_rank": working_rank,
        },
        "pre_registered_gate": {
            "median_relative_next_q_improvement_min": 0.10,
            "bootstrap_95_ci_lower_must_exceed": 0.0,
            "win_fraction_min": 0.70,
            "positive_next_force_dot_update_fraction_min": 0.95,
        },
        "comparisons": comparisons,
        "selected_eta": selected_eta,
        "stale_history_supported": selected_eta is not None,
        "pure_history_force": {
            "cosine_median": float(np.median(cosines)),
            "cosine_min": float(np.min(cosines)),
            "cosine_max": float(np.max(cosines)),
        },
        "per_transition": rows,
        "elapsed_seconds": time.perf_counter() - started,
        "finite": True,
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
