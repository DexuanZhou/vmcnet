#!/usr/bin/env python3
"""Read-only correlated one-step response audit for C rank-1600 WSSR.

Directions are constructed on batch k.  On the independently advanced batch k+1,
we evaluate parameter perturbations by importance reweighting from the unchanged
checkpoint distribution.  Neither parameters, walkers on disk, nor optimizer state
are written back.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np

from vmcnet.mcmc import position_amplitude_core as pacore
from vmcnet.physics import core as physics_core
from vmcnet.train import runners
from vmcnet.updates import wssr
from vmcnet.utils import io


NCHAINS = 1000
ALPHAS = (0.25, 0.5, 1.0)
EPS = 1.0e-12


def scalar(value) -> float:
    return float(jax.device_get(value))


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


def key_for(direction, alpha):
    return f"{direction}_alpha_{alpha:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--transitions", type=int, default=10)
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
    opt = config.vmc.optimizer.wssr_warm_svd_right
    expected = {
        "nchains": (int(config.vmc.nchains), NCHAINS),
        "rank": (int(opt.sr_rank), 1600),
        "eta": (float(opt.eta), 0.3),
        "learning_rate": (float(opt.learning_rate), 0.04),
        "learning_decay_rate": (float(opt.learning_decay_rate), 1.0e-4),
        "warm_iterations": (int(opt.svd_maxiter_warm), 2),
        "relative_cutoff": (float(opt.relative_singular_value_cutoff), 3.0e-4),
        "tikhonov_lambda": (float(opt.tikhonov_lambda), 1.0e-3),
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
    log_batch = jax.jit(jax.vmap(log_psi, in_axes=(None, 0)))
    _, walker_fn = runners._get_mcmc_fns(config.vmc, log_psi, apply_pmap=False)
    state = optimizer_state.core_state
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)

    working_rank = min(int(opt.svd_working_rank), int(opt.sr_rank_max))

    def make_direction(score, epsilon, candidate_key, eta, state_arg):
        augmented, rhs = wssr.augment_wssr_system(score, epsilon, state_arg, eta)
        u, singular_values, vh, approx_rank = wssr._wssr_right_svd_decomposition(
            augmented,
            state_arg,
            candidate_key,
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
            relative_singular_value_cutoff=float(opt.relative_singular_value_cutoff),
            tikhonov_lambda=float(opt.tikhonov_lambda),
            eps=EPS,
        )
        return (
            result.grad_like_update,
            result.active_rank,
            residual_ratio(epsilon, score.T @ result.grad_like_update),
        )

    make_direction_jit = jax.jit(make_direction)

    def evaluate_new_params(
        new_params, eval_positions, old_log_psi, baseline_energy
    ):
        _, _, stats = energy_fn(new_params, eval_positions)
        raw_local_energies = stats["local_energies_noclip"]
        new_log_psi = log_batch(new_params, eval_positions)
        log_weights = 2.0 * (new_log_psi - old_log_psi)
        log_weights = log_weights - jnp.max(log_weights)
        weights = jnp.exp(log_weights)
        weight_sum = jnp.sum(weights)
        normalized = weights / weight_sum
        # Form the paired energy difference directly.  Subtracting two float32
        # total energies near -37.8 Ha would otherwise lose the microhartree-scale
        # one-step signal to cancellation.
        delta_energy = jnp.sum(
            normalized * (raw_local_energies - baseline_energy)
        )
        energy = baseline_energy + delta_energy
        variance = jnp.sum(normalized * jnp.square(raw_local_energies - energy))
        ess = jnp.square(weight_sum) / jnp.sum(jnp.square(weights))
        return (
            energy,
            delta_energy,
            variance,
            ess,
            jnp.max(normalized),
            jnp.max(jnp.abs(log_weights)),
        )

    evaluate_new_params_jit = jax.jit(evaluate_new_params)

    def sample_batch(data, key):
        acceptance, data, key = walker_fn(params, data, key)
        positions = pacore.get_position_from_data(data)
        energy, clipped, stats = energy_fn(params, positions)
        score, _ = wssr.center_and_scale_score_matrix(log_psi, params, positions)
        epsilon = wssr.center_and_scale_energy_residuals(clipped, energy)
        raw = stats["local_energies_noclip"]
        logs = log_batch(params, positions)
        return data, key, acceptance, positions, score, epsilon, raw, logs

    effective_lr = float(opt.learning_rate) / (
        1.0 + float(opt.learning_decay_rate) * float(args.epoch)
    )
    norm_cap = math.sqrt(float(opt.norm_constraint))
    started = time.perf_counter()
    (
        data,
        key,
        acceptance,
        positions,
        score,
        epsilon,
        raw,
        logs,
    ) = sample_batch(data, key)
    rows = []
    for transition in range(args.transitions):
        key, candidate_key = jax.random.split(key)
        directions = {}
        construction = {}
        for name, eta in (("native", 0.3), ("current_only", 0.0)):
            direction, active_rank, q = make_direction_jit(
                score,
                epsilon,
                candidate_key,
                jnp.asarray(eta, dtype=score.dtype),
                state,
            )
            jax.block_until_ready(direction)
            directions[name] = direction
            construction[name] = {
                "eta": eta,
                "active_rank": int(jax.device_get(active_rank)),
                "current_q": scalar(q),
                "direction_norm": scalar(jnp.linalg.norm(direction)),
            }

        native_unconstrained_step = -effective_lr * directions["native"]
        native_scale = min(
            1.0,
            norm_cap / max(scalar(jnp.linalg.norm(native_unconstrained_step)), EPS),
        )
        native_step = native_scale * native_unconstrained_step
        current_only_direction = directions["current_only"]
        current_only_step = -effective_lr * current_only_direction
        current_only_scale = min(
            1.0,
            norm_cap / max(scalar(jnp.linalg.norm(current_only_step)), EPS),
        )
        current_only_step = current_only_scale * current_only_step
        current_only_matched = -current_only_direction * (
            scalar(jnp.linalg.norm(native_step))
            / max(scalar(jnp.linalg.norm(current_only_direction)), EPS)
        )

        (
            data,
            key,
            next_acceptance,
            next_positions,
            next_score,
            next_epsilon,
            next_raw,
            next_logs,
        ) = sample_batch(data, key)
        baseline_energy = jnp.mean(next_raw)
        baseline_variance = jnp.var(next_raw)
        response = {}
        for direction_name, step in (
            ("native", native_step),
            ("current_only_matched_norm", current_only_matched),
        ):
            for alpha in ALPHAS:
                new_params = unravel(flat_params + alpha * step)
                (
                    energy_new,
                    delta_energy,
                    variance_new,
                    ess,
                    max_weight,
                    max_abs_logw,
                ) = (
                    evaluate_new_params_jit(
                        new_params, next_positions, next_logs, baseline_energy
                    )
                )
                jax.block_until_ready(energy_new)
                response[key_for(direction_name, alpha)] = {
                    "direction": direction_name,
                    "alpha": alpha,
                    "energy": scalar(energy_new),
                    "delta_energy": scalar(delta_energy),
                    "variance": scalar(variance_new),
                    "delta_variance": scalar(variance_new - baseline_variance),
                    "ess": scalar(ess),
                    "ess_fraction": scalar(ess / NCHAINS),
                    "max_normalized_weight": scalar(max_weight),
                    "max_abs_shifted_log_weight": scalar(max_abs_logw),
                }

        row = {
            "transition": transition,
            "acceptance_current": scalar(acceptance),
            "acceptance_next": scalar(next_acceptance),
            "baseline_energy": scalar(baseline_energy),
            "baseline_variance": scalar(baseline_variance),
            "effective_learning_rate": effective_lr,
            "native_step_norm": scalar(jnp.linalg.norm(native_step)),
            "current_only_production_step_norm": scalar(
                jnp.linalg.norm(current_only_step)
            ),
            "native_norm_constraint_scale": native_scale,
            "current_only_norm_constraint_scale": current_only_scale,
            "construction": construction,
            "response": response,
        }
        if not all(
            math.isfinite(value)
            for metrics in response.values()
            for value in metrics.values()
            if isinstance(value, float)
        ):
            raise FloatingPointError(f"nonfinite response: {row}")
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

        acceptance, positions, score, epsilon, raw, logs = (
            next_acceptance,
            next_positions,
            next_score,
            next_epsilon,
            next_raw,
            next_logs,
        )
        del directions, construction, response

    summaries = {}
    for direction in ("native", "current_only_matched_norm"):
        for alpha_index, alpha in enumerate(ALPHAS):
            name = key_for(direction, alpha)
            deltas = [row["response"][name]["delta_energy"] for row in rows]
            variances = [row["response"][name]["delta_variance"] for row in rows]
            ess = [row["response"][name]["ess_fraction"] for row in rows]
            summaries[name] = {
                "median_delta_energy": float(np.median(deltas)),
                "bootstrap_95_ci_delta_energy": bootstrap_ci(
                    deltas, seed=20260802 + alpha_index
                ),
                "median_delta_variance": float(np.median(variances)),
                "median_ess_fraction": float(np.median(ess)),
                "minimum_ess_fraction": float(np.min(ess)),
            }

    native_full = np.asarray(
        [row["response"][key_for("native", 1.0)]["delta_energy"] for row in rows]
    )
    native_half = np.asarray(
        [row["response"][key_for("native", 0.5)]["delta_energy"] for row in rows]
    )
    direction_gap = np.asarray(
        [
            row["response"][key_for("native", 1.0)]["delta_energy"]
            - row["response"][key_for("current_only_matched_norm", 1.0)][
                "delta_energy"
            ]
            for row in rows
        ]
    )
    full_minus_half = native_full - native_half
    classification = {
        "native_full_minus_half_median": float(np.median(full_minus_half)),
        "native_full_minus_half_bootstrap_95_ci": bootstrap_ci(
            full_minus_half, seed=20260901
        ),
        "native_minus_current_only_matched_norm_median": float(
            np.median(direction_gap)
        ),
        "native_minus_current_only_matched_norm_bootstrap_95_ci": bootstrap_ci(
            direction_gap, seed=20260902
        ),
        "step_too_large_supported": bool(
            np.median(full_minus_half) >= 2.0e-6
            and bootstrap_ci(full_minus_half, seed=20260901)[0] > 0.0
        ),
        "native_direction_worse_supported": bool(
            np.median(direction_gap) >= 2.0e-6
            and bootstrap_ci(direction_gap, seed=20260902)[0] > 0.0
        ),
    }
    payload = {
        "experiment": "C rank1600 WSSR correlated one-step response audit",
        "read_only_checkpoint": True,
        "parameters_updated": False,
        "optimizer_state_updated": False,
        "source_checkpoint": str(checkpoint),
        "stored_epoch": int(stored_epoch),
        "transitions": args.transitions,
        "walkers_per_batch": NCHAINS,
        "mcmc_steps_between_batches": int(config.vmc.nsteps_per_param_update),
        "alphas": ALPHAS,
        "effective_learning_rate": effective_lr,
        "config_verified": {name: actual for name, (actual, _) in expected.items()},
        "pre_registered_gates": {
            "median_energy_gap_min_ha": 2.0e-6,
            "bootstrap_95_ci_lower_must_exceed_ha": 0.0,
            "importance_ess_fraction_required_for_interpretation": 0.5,
        },
        "summary": summaries,
        "classification": classification,
        "per_transition": rows,
        "finite": True,
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
