#!/usr/bin/env python3
"""Read-only audit of a direction-sensitive adaptive WSSR S-history indicator."""

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
from vmcnet.updates import wssr
from vmcnet.utils import io


FIXED_LAMBDA = 1.0e-3
NORM_CONSTRAINT = 1.0e-3
DEFAULT_LEARNING_RATE = 0.04
RANK = 1600
STATIC_DEPTH = 16
FUTURE_BATCHES = 4
EPS = 1.0e-12


def scalar(value) -> float:
    return float(jax.device_get(value))


def require_finite(label: str, *arrays) -> None:
    for index, array in enumerate(arrays):
        if not bool(jax.device_get(jnp.all(jnp.isfinite(array)))):
            raise FloatingPointError(f"{label}[{index}] contains non-finite values")


def core_state_from_optimizer(optimizer_state):
    if not hasattr(optimizer_state, "core_state"):
        raise TypeError(
            f"checkpoint optimizer has no WSSR core_state: {type(optimizer_state)}"
        )
    return optimizer_state.core_state


def normalize_columns(matrix):
    norms = jnp.linalg.norm(matrix, axis=0, keepdims=True)
    return matrix / jnp.maximum(norms, EPS)


def safe_cosine(left, right):
    return jnp.vdot(left, right) / jnp.maximum(
        jnp.linalg.norm(left) * jnp.linalg.norm(right), EPS
    )


def actual_scaled_direction(direction, learning_rate):
    coefficient = jnp.minimum(
        jnp.asarray(learning_rate, dtype=direction.dtype),
        jnp.sqrt(jnp.asarray(NORM_CONSTRAINT, dtype=direction.dtype))
        / jnp.maximum(jnp.linalg.norm(direction), EPS),
    )
    return coefficient * direction, coefficient


@jax.jit
def current_only_direction(score, epsilon):
    """Exact current-batch MinSR direction with fixed Tikhonov damping."""
    count = score.shape[1]
    ones = jnp.ones((count, 1), dtype=score.dtype)
    gram = score.T @ score + ones @ ones.T / count
    gram = (gram + gram.T) / 2.0
    values, vectors = jnp.linalg.eigh(gram)
    filtered = jnp.maximum(values, 0.0) + FIXED_LAMBDA
    sample_solution = vectors @ ((vectors.T @ epsilon) / filtered)
    sample_solution = sample_solution - jnp.mean(sample_solution)
    return score @ sample_solution


def mixed_direction(score, epsilon, state, eta, candidate_key):
    """Use historical S but force the RHS to the current stochastic gradient."""
    augmented, augmented_rhs = wssr.augment_wssr_system(
        score, epsilon, state, jnp.asarray(eta, dtype=score.dtype)
    )
    working_rank = min(RANK, state.sr_o.shape[1])
    u, singular_values, vh, _ = wssr._wssr_right_svd_decomposition(
        augmented,
        state,
        candidate_key,
        working_rank,
        40,
        2,
        False,
        EPS,
        exact_first_force=False,
    )
    current_gradient = score @ epsilon
    result = wssr._wssr_update_from_svd(
        o_aug=augmented,
        e_aug=augmented_rhs,
        state=state,
        u=u,
        singular_values=singular_values,
        vh=vh,
        damping=3.0e-4,
        norm_constraint=NORM_CONSTRAINT,
        sr_rank_max=state.sr_o.shape[1],
        sr_scale=1.1,
        constrain_update_norm=False,
        rank_update_max=state.sr_o.shape[1],
        spectral_regularization="tikhonov",
        complement_weight=0.0,
        relative_singular_value_cutoff=3.0e-4,
        tikhonov_lambda=FIXED_LAMBDA,
        experimental_mode="none",
        mixed_precision_solve=True,
        mixed_precision_history_width=state.sr_o.shape[1],
        force_override=current_gradient,
        eps=EPS,
    )
    if hasattr(state, "u"):
        warm_u = jnp.zeros_like(state.u)
        warm_width = min(u.shape[1], state.u.shape[1])
        warm_mask = (
            jnp.arange(warm_width) < result.active_rank
        ).astype(u.dtype)
        warm_u = warm_u.at[:, :warm_width].set(
            u[:, :warm_width] * warm_mask
        )
        next_state = wssr.WSSRWarmSVDCoreState(
            sr_o=result.state.sr_o,
            ek=result.state.ek,
            sr_rank0=result.state.sr_rank0,
            sr_rank=result.state.sr_rank,
            u=warm_u,
            has_u=jnp.where(
                result.active_rank > 0, jnp.asarray(True), state.has_u
            ),
        )
    else:
        next_state = result.state
    return result.grad_like_update, next_state, result.active_rank


@jax.jit
def indicator_diagnostics(
    score, state, current_direction, history_direction, current_gradient
):
    """Compute action-only noise and drift diagnostics on relevant directions."""
    probe = normalize_columns(
        jnp.stack(
            [
                current_direction,
                history_direction,
                current_gradient,
            ],
            axis=1,
        )
    )

    current_action = score @ (score.T @ probe)
    history_mask = (
        jnp.arange(state.sr_o.shape[1]) < state.sr_rank0
    ).astype(score.dtype)
    history_coordinates = state.sr_o.T @ probe
    history_action = state.sr_o @ (history_mask[:, None] * history_coordinates)

    half = score.shape[1] // 2
    score_a = score[:, :half]
    score_b = score[:, half : 2 * half]
    score_a = jnp.sqrt(2.0) * (
        score_a - jnp.mean(score_a, axis=1, keepdims=True)
    )
    score_b = jnp.sqrt(2.0) * (
        score_b - jnp.mean(score_b, axis=1, keepdims=True)
    )
    action_a = score_a @ (score_a.T @ probe)
    action_b = score_b @ (score_b.T @ probe)
    noise_squared = 0.25 * jnp.sum(jnp.square(action_a - action_b))
    drift_squared = jnp.sum(jnp.square(current_action - history_action))
    drift_ratio = drift_squared / jnp.maximum(noise_squared, EPS)
    inverse_action_ratio = jnp.linalg.norm(
        (current_action - history_action)[:, 0]
    ) / jnp.maximum(
        jnp.linalg.norm(current_action[:, 0])
        + FIXED_LAMBDA * jnp.linalg.norm(current_direction),
        EPS,
    )
    return noise_squared, drift_squared, drift_ratio, inverse_action_ratio


def initialize_static_history(sample_batch, data, sample_key, base_key, score_shape):
    state = wssr.initialize_wssr_warm_svd_core_state(
        score_shape[0], RANK, RANK, dtype=jnp.float32, store_warm_u=True
    )
    acceptances = []
    for step in range(1, STATIC_DEPTH + 1):
        data, sample_key, acceptance, score, epsilon, _ = sample_batch(
            data, sample_key
        )
        candidate_key = jax.random.fold_in(base_key, 920000 + step)
        _, state, _ = mixed_direction(
            score, epsilon, state, 0.95, candidate_key
        )
        jax.block_until_ready(state.sr_o)
        require_finite("static_history", state.sr_o, state.ek)
        acceptances.append(scalar(acceptance))
    return state, data, sample_key, acceptances


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--static-history", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not 0.0 < args.eta < 1.0:
        raise ValueError("eta must be strictly between zero and one")

    config = io.load_config_dict(str(args.run), "config.json")
    checkpoint = args.checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    stored_epoch, base_data, params, optimizer_state, base_key = io.reload_vmc_state(
        str(checkpoint.parent), checkpoint.name
    )
    if int(config.vmc.nchains) != 1000:
        raise ValueError(f"expected 1000 walkers, got {config.vmc.nchains}")

    dtype = runners._get_dtype(config)
    ions, charges, nelec = runners._get_electron_ion_config_as_arrays(
        config, dtype
    )
    if tuple(int(value) for value in nelec) != (4, 2):
        raise ValueError(f"unexpected spin sector {nelec}")
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
    energy_fn = jax.jit(
        physics_core.create_energy_and_statistics_fn(
            local_energy,
            1000,
            runners._get_clipping_fn(config.vmc),
            config.vmc.nan_safe,
            record_raw_local_energies=True,
        )
    )
    _, walker_fn = runners._get_mcmc_fns(config.vmc, log_psi, apply_pmap=False)

    def sample_batch(data, sample_key):
        acceptance, data, sample_key = walker_fn(params, data, sample_key)
        positions = pacore.get_position_from_data(data)
        energy, clipped, stats = energy_fn(params, positions)
        score, _ = wssr.center_and_scale_score_matrix(log_psi, params, positions)
        epsilon = wssr.center_and_scale_energy_residuals(clipped, energy)
        return data, sample_key, acceptance, score, epsilon, stats

    if args.static_history:
        template_data = base_data
        template_key = jax.random.fold_in(base_key, 919999)
        template_data, template_key, _, template_score, _, _ = sample_batch(
            template_data, template_key
        )
        template_state = None
        score_shape = template_score.shape
        del template_score, template_data, template_key
    else:
        template_state = core_state_from_optimizer(optimizer_state)
        score_shape = None
    del optimizer_state
    gc.collect()

    optimizer_config = config.vmc.optimizer.wssr_warm_svd_right
    learning_rate = float(
        optimizer_config.get("learning_rate", DEFAULT_LEARNING_RATE)
    )
    rows = []
    started = time.perf_counter()
    for replicate in range(args.replicates):
        data = base_data
        sample_key = jax.random.fold_in(base_key, 930000 + replicate)
        history_acceptances = []
        if args.static_history:
            state, data, sample_key, history_acceptances = initialize_static_history(
                sample_batch,
                data,
                sample_key,
                jax.random.fold_in(base_key, replicate),
                score_shape,
            )
        else:
            state = template_state

        data, sample_key, acceptance, score, epsilon, stats = sample_batch(
            data, sample_key
        )
        current = current_only_direction(score, epsilon)
        candidate_key = jax.random.fold_in(base_key, 940000 + replicate)
        history, _, active_rank = mixed_direction(
            score, epsilon, state, args.eta, candidate_key
        )
        current_applied, current_scale = actual_scaled_direction(
            current, learning_rate
        )
        history_applied, history_scale = actual_scaled_direction(
            history, learning_rate
        )
        current_gradient = score @ epsilon
        noise_sq, drift_sq, drift_ratio, inverse_action_ratio = (
            indicator_diagnostics(
                score, state, current, history, current_gradient
            )
        )
        require_finite(
            "candidates",
            current,
            history,
            current_applied,
            history_applied,
            noise_sq,
            drift_sq,
            drift_ratio,
        )

        future_current_dots = []
        future_history_dots = []
        future_current_cosines = []
        future_history_cosines = []
        future_acceptances = []
        future_variances = []
        for _ in range(FUTURE_BATCHES):
            (
                data,
                sample_key,
                future_acceptance,
                future_score,
                future_epsilon,
                future_stats,
            ) = sample_batch(data, sample_key)
            future_force = future_score @ future_epsilon
            require_finite("future", future_force)
            future_current_dots.append(scalar(jnp.vdot(future_force, current_applied)))
            future_history_dots.append(scalar(jnp.vdot(future_force, history_applied)))
            future_current_cosines.append(scalar(safe_cosine(future_force, current)))
            future_history_cosines.append(scalar(safe_cosine(future_force, history)))
            future_acceptances.append(scalar(future_acceptance))
            future_variances.append(scalar(future_stats["variance_noclip"]))

        current_dot = float(np.mean(future_current_dots))
        history_dot = float(np.mean(future_history_dots))
        gain = (history_dot - current_dot) / max(abs(current_dot), EPS)
        row = {
            "replicate": replicate,
            "current_acceptance": scalar(acceptance),
            "history_acceptances": history_acceptances,
            "future_acceptances": future_acceptances,
            "current_batch_variance": scalar(stats["variance_noclip"]),
            "future_variances": future_variances,
            "active_rank": int(jax.device_get(active_rank)),
            "current_raw_norm": scalar(jnp.linalg.norm(current)),
            "history_raw_norm": scalar(jnp.linalg.norm(history)),
            "current_scale": scalar(current_scale),
            "history_scale": scalar(history_scale),
            "candidate_cosine": scalar(safe_cosine(current, history)),
            "noise_squared": scalar(noise_sq),
            "drift_squared": scalar(drift_sq),
            "noise_drift_ratio": scalar(drift_ratio),
            "inverse_action_ratio": scalar(inverse_action_ratio),
            "future_current_dots": future_current_dots,
            "future_history_dots": future_history_dots,
            "future_current_cosines": future_current_cosines,
            "future_history_cosines": future_history_cosines,
            "mean_future_current_dot": current_dot,
            "mean_future_history_dot": history_dot,
            "paired_predicted_descent_gain": gain,
            "mean_future_cosine_gain": float(
                np.mean(future_history_cosines) - np.mean(future_current_cosines)
            ),
            "baseline_dot_positive": current_dot > 0.0,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        del data, state, score, epsilon, current, history
        gc.collect()

    payload = {
        "experiment": "C adaptive-S direction indicator audit",
        "read_only": True,
        "label": args.label,
        "source_run": str(args.run),
        "source_checkpoint": str(checkpoint),
        "stored_epoch": int(stored_epoch),
        "eta": args.eta,
        "static_history": args.static_history,
        "static_history_depth": STATIC_DEPTH if args.static_history else 0,
        "replicates": args.replicates,
        "future_batches": FUTURE_BATCHES,
        "fixed_lambda": FIXED_LAMBDA,
        "rank": RANK,
        "learning_rate_for_scaling": learning_rate,
        "norm_constraint": NORM_CONSTRAINT,
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
