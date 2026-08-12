#!/usr/bin/env python3
"""Read-only same-coordinate audit of the WSSR gradient transport model.

For each independently advanced walker batch, construct the actual constrained
WSSR step at the checkpoint parameters.  Re-evaluate the VMC force at the new
parameters on exactly the same electron coordinates and compare its observed
change with both signs of S_current @ delta_theta and S_ema @ delta_theta.
Nothing from the checkpoint is mutated or written back.
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


EPS = 1.0e-12


def scalar(value) -> float:
    return float(jax.device_get(value))


def vector_metrics(observed, prediction):
    observed_norm = jnp.linalg.norm(observed)
    prediction_norm = jnp.linalg.norm(prediction)
    denominator = jnp.maximum(observed_norm, EPS)
    cosine_denominator = jnp.maximum(observed_norm * prediction_norm, EPS)
    prediction_norm_sq = jnp.maximum(jnp.vdot(prediction, prediction), EPS)
    return {
        "observed_norm": scalar(observed_norm),
        "prediction_norm": scalar(prediction_norm),
        "prediction_to_observed_norm_ratio": scalar(prediction_norm / denominator),
        "cosine": scalar(jnp.vdot(observed, prediction) / cosine_denominator),
        "relative_error": scalar(jnp.linalg.norm(observed - prediction) / denominator),
        "optimal_scale": scalar(jnp.vdot(observed, prediction) / prediction_norm_sq),
    }


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
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--transitions", type=int, default=8)
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
    if config.vmc.optimizer_type != "wssr_warm_svd_right":
        raise ValueError(config.vmc.optimizer_type)
    opt = config.vmc.optimizer.wssr_warm_svd_right
    expected = {
        "nchains": (int(config.vmc.nchains), 1000),
        "rank": (int(opt.sr_rank), 1600),
        "learning_rate": (float(opt.learning_rate), 0.04),
        "learning_decay_rate": (float(opt.learning_decay_rate), 1.0e-4),
        "lambda": (float(opt.tikhonov_lambda), 1.0e-3),
        "norm_constraint": (float(opt.norm_constraint), 1.0e-3),
        "eta_S_max": (float(opt.eta_S_max), 0.95),
        "eta_S_tau": (float(opt.eta_S_tau), 1000.0),
    }
    mismatches = {name: pair for name, pair in expected.items() if pair[0] != pair[1]}
    if mismatches:
        raise ValueError(f"unexpected source configuration: {mismatches}")
    if not bool(opt.adaptive_S_average) or opt.eta_S_schedule != "exponential_growth":
        raise ValueError("source must use exponential adaptive S averaging")
    if not isinstance(optimizer_state, wssr.WSSRTransportedGradientOptimizerState):
        raise TypeError(type(optimizer_state))

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
            int(config.vmc.nchains),
            runners._get_clipping_fn(config.vmc),
            config.vmc.nan_safe,
        )
    )
    _, walker_fn = runners._get_mcmc_fns(config.vmc, log_psi, apply_pmap=False)
    flat_params, unravel = jax.flatten_util.ravel_pytree(params)
    state = optimizer_state.core_state
    eta_S = float(opt.eta_S_max) * (
        1.0 - math.exp(-float(args.epoch) / float(opt.eta_S_tau))
    )
    effective_lr = float(opt.learning_rate) / (
        1.0 + float(opt.learning_decay_rate) * float(args.epoch)
    )
    norm_cap = math.sqrt(float(opt.norm_constraint))

    def force_at(candidate_params, candidate_positions):
        energy, local_energies, _ = energy_fn(candidate_params, candidate_positions)
        score, _ = wssr.center_and_scale_score_matrix(
            log_psi, candidate_params, candidate_positions
        )
        epsilon = wssr.center_and_scale_energy_residuals(local_energies, energy)
        return energy, score, epsilon, score @ epsilon

    force_at_jit = jax.jit(force_at)

    def direction_at(candidate_positions, local_energies, energy, direction_key):
        result = wssr.compute_wssr_warm_svd_right_transported_gradient_update(
            log_psi_apply=log_psi,
            params=params,
            positions=candidate_positions,
            local_energies=local_energies,
            energy=energy,
            state=state,
            previous_gradient=jnp.zeros_like(flat_params),
            previous_theta=flat_params,
            transport_initialized=jnp.asarray(False),
            key=direction_key,
            eta_S=jnp.asarray(eta_S, dtype=flat_params.dtype),
            eta_g=jnp.asarray(0.0, dtype=flat_params.dtype),
            enable_gradient_transport=False,
            record_S_lag_diagnostics=False,
            damping=float(opt.damping),
            norm_constraint=float(opt.norm_constraint),
            sr_rank_max=int(opt.sr_rank_max),
            sr_scale=float(opt.sr_scale),
            svd_maxiter_initial=int(opt.svd_maxiter_initial),
            svd_maxiter_warm=int(opt.svd_maxiter_warm),
            exact_first=bool(opt.exact_first),
            exact_first_force=False,
            svd_working_rank=int(opt.svd_working_rank),
            spectral_regularization=str(opt.spectral_regularization),
            complement_weight=0.0,
            relative_singular_value_cutoff=float(opt.relative_singular_value_cutoff),
            tikhonov_lambda=float(opt.tikhonov_lambda),
            mixed_precision_solve=bool(opt.mixed_precision_solve),
        )
        return result[0]

    direction_at_jit = jax.jit(direction_at)

    rows = []
    started = time.perf_counter()
    for transition in range(args.transitions):
        _, data, key = walker_fn(params, data, key)
        positions = pacore.get_position_from_data(data)
        energy_old, local_energies_old, _ = energy_fn(params, positions)
        score_old, _ = wssr.center_and_scale_score_matrix(log_psi, params, positions)
        epsilon_old = wssr.center_and_scale_energy_residuals(
            local_energies_old, energy_old
        )
        gradient_old = score_old @ epsilon_old
        key, direction_key = jax.random.split(key)
        direction_tree = direction_at_jit(
            positions, local_energies_old, energy_old, direction_key
        )
        flat_direction, _ = jax.flatten_util.ravel_pytree(direction_tree)
        unconstrained_delta = -effective_lr * flat_direction
        scale = jnp.minimum(
            1.0,
            norm_cap / jnp.maximum(jnp.linalg.norm(unconstrained_delta), EPS),
        )
        delta_theta = scale * unconstrained_delta
        new_params = unravel(flat_params + delta_theta)
        energy_new, score_new, epsilon_new, gradient_new = force_at_jit(
            new_params, positions
        )
        observed = gradient_new - gradient_old
        current_action = score_old @ (score_old.T @ delta_theta)
        history_action = wssr.apply_wssr_history_operator(state, delta_theta)
        ema_action = eta_S * history_action + (1.0 - eta_S) * current_action
        row = {
            "transition": transition,
            "energy_old": scalar(energy_old),
            "energy_new_same_coordinates": scalar(energy_new),
            "delta_theta_norm": scalar(jnp.linalg.norm(delta_theta)),
            "norm_constraint_scale": scalar(scale),
            "gradient_old_norm": scalar(jnp.linalg.norm(gradient_old)),
            "gradient_new_norm": scalar(jnp.linalg.norm(gradient_new)),
            "current_plus": vector_metrics(observed, current_action),
            "current_minus": vector_metrics(observed, -current_action),
            "history_plus": vector_metrics(observed, history_action),
            "history_minus": vector_metrics(observed, -history_action),
            "ema_plus": vector_metrics(observed, ema_action),
            "ema_minus": vector_metrics(observed, -ema_action),
        }
        if not all(
            math.isfinite(value)
            for candidate in (
                row["current_plus"], row["current_minus"],
                row["history_plus"], row["history_minus"],
                row["ema_plus"], row["ema_minus"],
            )
            for value in candidate.values()
        ):
            raise FloatingPointError(row)
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    summary = {}
    for name in (
        "current_plus", "current_minus", "history_plus", "history_minus",
        "ema_plus", "ema_minus",
    ):
        summary[name] = {
            metric: summarize([row[name][metric] for row in rows])
            for metric in (
                "prediction_to_observed_norm_ratio",
                "cosine",
                "relative_error",
                "optimal_scale",
            )
        }
    best_ema_sign = min(
        ("ema_plus", "ema_minus"),
        key=lambda name: summary[name]["relative_error"]["median"],
    )
    best = summary[best_ema_sign]
    gate = {
        "best_ema_sign": best_ema_sign,
        "median_cosine_positive": best["cosine"]["median"] > 0.0,
        "median_relative_error_better_than_zero_predictor": (
            best["relative_error"]["median"] < 1.0
        ),
        "win_fraction_relative_error_below_one": float(np.mean([
            row[best_ema_sign]["relative_error"] < 1.0 for row in rows
        ])),
    }
    gate["passed"] = bool(
        gate["median_cosine_positive"]
        and gate["median_relative_error_better_than_zero_predictor"]
        and gate["win_fraction_relative_error_below_one"] >= 0.75
    )
    payload = {
        "experiment": "C WSSR same-coordinate gradient transport audit",
        "read_only_checkpoint": True,
        "source_checkpoint": str(checkpoint),
        "stored_epoch": int(stored_epoch),
        "transitions": args.transitions,
        "eta_S": eta_S,
        "effective_learning_rate": effective_lr,
        "norm_cap": norm_cap,
        "summary": summary,
        "gate": gate,
        "rows": rows,
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
