#!/usr/bin/env python3
"""Read-only early-regime WSSR memory-length and SPRING replay audit."""

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


NCHAINS = 1000
RANK = 1600
DAMPING = 0.0003
CUTOFF = 0.0003
FIXED_LAMBDA = 0.001
LEARNING_RATE = 0.04
NORM_CONSTRAINT = 0.001
SPRING_MU = 0.99
SPRING_DAMPING = 0.001
DEPTHS = (3, 5, 8, 16)
FUTURE_BATCHES = 4
EPS = 1.0e-12

WSSR_ARMS = {
    "wssr_eta03": (0.3, False),
    "wssr_eta08": (0.8, False),
    "wssr_eta095": (0.95, False),
    "wssr_eta099": (0.99, False),
    "wssr_eta099_bias_corrected": (0.99, True),
}
VALID_ARMS = ("single_current", *WSSR_ARMS, "spring_mu099")


def mixed_precision_rank_filter(
    history_coefficients, current_coefficients, singular_values
):
    """Add cancellation-prone rank coefficients and filter them in fp64."""
    history64 = np.asarray(history_coefficients, dtype=np.float64)
    current64 = np.asarray(current_coefficients, dtype=np.float64)
    singular64 = np.asarray(singular_values, dtype=np.float64)
    leading = max(float(singular64[0]), EPS)
    retained = singular64 / leading > CUTOFF
    filtered = np.where(
        retained,
        (history64 + current64)
        / (np.square(singular64) + FIXED_LAMBDA),
        0.0,
    )
    if not np.all(np.isfinite(filtered)):
        raise FloatingPointError("mixed-precision rank filter became non-finite")
    return np.asarray(filtered, dtype=np.float32)


def scalar(value) -> float:
    return float(jax.device_get(value))


def integer(value) -> int:
    return int(jax.device_get(value))


def safe_cosine(left, right):
    return jnp.vdot(left, right) / jnp.maximum(
        jnp.linalg.norm(left) * jnp.linalg.norm(right), EPS
    )


def residual_ratio(target, prediction):
    return jnp.linalg.norm(target - prediction) / jnp.maximum(
        jnp.linalg.norm(target), EPS
    )


def require_finite(label: str, *arrays) -> None:
    """Fail immediately instead of allowing a non-finite replay to be scored."""
    for index, array in enumerate(arrays):
        if not bool(jax.device_get(jnp.all(jnp.isfinite(array)))):
            raise FloatingPointError(f"{label}[{index}] contains non-finite values")


def effective_eta(target: float, bias_corrected: bool, step: int) -> float:
    if not bias_corrected:
        return target
    return min(target, 1.0 - 1.0 / float(step))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--arm", choices=VALID_ARMS, required=True)
    parser.add_argument("--replicates", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = args.run / "checkpoints" / f"{args.epoch}.npz"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config = io.load_config_dict(str(args.run), "config.json")
    stored_epoch, base_data, params, optimizer_state, base_key = io.reload_vmc_state(
        str(checkpoint.parent), checkpoint.name
    )
    if int(stored_epoch) != args.epoch - 1:
        raise ValueError(f"stored epoch {stored_epoch} != {args.epoch - 1}")
    if int(config.vmc.nchains) != NCHAINS:
        raise ValueError(f"expected {NCHAINS} walkers, got {config.vmc.nchains}")

    dtype = runners._get_dtype(config)
    if dtype != jnp.float32:
        raise ValueError(f"expected a float32 source model, got {dtype}")
    ions, charges, nelec = runners._get_electron_ion_config_as_arrays(config, dtype)
    if tuple(int(x) for x in nelec) != (4, 2):
        raise ValueError(f"unexpected C spin sector: {nelec}")
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
            NCHAINS,
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
        return (
            data,
            sample_key,
            acceptance,
            score,
            epsilon,
            stats["local_energies_noclip"],
        )

    @jax.jit
    def wssr_step(score, epsilon, state, candidate_key, eta):
        augmented, augmented_rhs = wssr.augment_wssr_system(
            score, epsilon, state, eta
        )
        history_force = eta * (state.sr_o @ state.ek)
        current_force = jnp.where(
            state.sr_rank0 > 0,
            (1.0 - eta) * (score @ epsilon),
            score @ epsilon,
        )
        history_mass = eta * jnp.sum(jnp.square(state.sr_o))
        current_mass = jnp.where(
            state.sr_rank0 > 0,
            (1.0 - eta) * jnp.sum(jnp.square(score)),
            jnp.sum(jnp.square(score)),
        )
        u, singular_values, vh, rank = wssr._wssr_right_svd_decomposition(
            augmented,
            state,
            candidate_key,
            RANK,
            40,
            2,
            False,
            EPS,
            exact_first_force=False,
        )

        common = dict(
            o_aug=augmented,
            e_aug=augmented_rhs,
            state=state,
            u=u,
            singular_values=singular_values,
            vh=vh,
            damping=DAMPING,
            norm_constraint=NORM_CONSTRAINT,
            sr_rank_max=RANK,
            sr_scale=1.1,
            constrain_update_norm=False,
            rank_update_max=RANK,
            spectral_regularization="tikhonov",
            complement_weight=0.0,
            relative_singular_value_cutoff=CUTOFF,
            experimental_mode="none",
            eps=EPS,
        )
        fixed32 = wssr._wssr_update_from_svd(
            tikhonov_lambda=FIXED_LAMBDA, **common
        )

        # Mixed-precision hard-subspace solve.  Form the history and current
        # force terms separately, cast them before addition, and perform the
        # spectral filtering in fp64.  The compressed state remains in the
        # production dtype and is generated with the identical retained mask.
        full_force = history_force + current_force
        # Keep the large P-by-rank projections in the production dtype, but
        # add the independently projected history/current terms and apply the
        # inverse spectral filter in fp64.  This is the production-feasible
        # mixed-precision path: only O(rank) values need cross the precision
        # boundary, while the cancellation-prone operation is protected.
        history_coefficients = u.T @ history_force
        current_coefficients = u.T @ current_force
        coefficients = jax.pure_callback(
            mixed_precision_rank_filter,
            jax.ShapeDtypeStruct(singular_values.shape, score.dtype),
            history_coefficients,
            current_coefficients,
            singular_values,
        )
        fixed = u @ coefficients

        u_state = jnp.zeros_like(state.u)
        warm_width = min(u.shape[1], state.u.shape[1])
        rank_mask = (jnp.arange(warm_width) < rank).astype(u.dtype)
        u_state = u_state.at[:, :warm_width].set(
            u[:, :warm_width] * rank_mask
        )
        new_state = wssr.WSSRWarmSVDCoreState(
            sr_o=fixed32.state.sr_o,
            ek=fixed32.state.ek,
            sr_rank0=fixed32.state.sr_rank0,
            sr_rank=fixed32.state.sr_rank,
            u=u_state,
            has_u=jnp.where(rank > 0, jnp.array(True), state.has_u),
        )
        compressed_force = new_state.sr_o @ new_state.ek
        force_relerr = jnp.linalg.norm(compressed_force - full_force) / jnp.maximum(
            jnp.linalg.norm(full_force), EPS
        )
        force_cosine = safe_cosine(compressed_force, full_force)
        history_force_ratio = jnp.linalg.norm(history_force) / jnp.maximum(
            jnp.linalg.norm(current_force), EPS
        )
        history_mass_fraction = history_mass / jnp.maximum(
            history_mass + current_mass, EPS
        )
        old_force_projection = jnp.linalg.norm(u.T @ history_force) / jnp.maximum(
            jnp.linalg.norm(history_force), EPS
        )
        diagnostics = jnp.asarray(
            [
                force_relerr,
                force_cosine,
                history_force_ratio,
                history_mass_fraction,
                old_force_projection,
                singular_values[0],
            ]
        )
        return (
            fixed,
            new_state,
            fixed32.active_rank,
            diagnostics,
        )

    @jax.jit
    def history_spectral_quality(old_state, new_state, eta, probe_key):
        width = old_state.sr_o.shape[1]
        old_mask = (
            jnp.arange(width) < old_state.sr_rank0
        ).astype(old_state.sr_o.dtype)
        old_factor = jnp.sqrt(eta) * old_state.sr_o * old_mask[None, :]
        coefficients = jax.random.rademacher(
            probe_key, (width, 8), dtype=old_state.sr_o.dtype
        )
        old_vectors = old_factor @ coefficients
        new_u = wssr.recover_u_from_sr_o(
            new_state.sr_o, new_state.sr_rank0, RANK, eps=EPS
        )
        projected_coordinates = new_u.T @ old_vectors
        return jnp.sum(jnp.square(projected_coordinates)) / jnp.maximum(
            jnp.sum(jnp.square(old_vectors)), EPS
        )

    @jax.jit
    def spring_step(score, epsilon, previous):
        mu_previous = SPRING_MU * previous
        count = score.shape[1]
        ones = jnp.ones((count, 1), dtype=score.dtype)
        gram = score.T @ score + ones @ ones.T / count
        gram = (gram + gram.T) / 2.0
        values, vectors = jnp.linalg.eigh(gram)
        filtered = jnp.maximum(values, 0.0) + SPRING_DAMPING
        rhs = epsilon - score.T @ mu_previous
        zeta = vectors @ ((vectors.T @ rhs) / filtered)
        zeta = zeta - jnp.mean(zeta)
        correction = score @ zeta.astype(score.dtype)
        return (
            mu_previous + correction
        )

    def zero_state(score):
        return wssr.initialize_wssr_warm_svd_core_state(
            score.shape[0], RANK, RANK, dtype=score.dtype, store_warm_u=True
        )

    def cap_scale(update):
        raw_step_norm = LEARNING_RATE * jnp.linalg.norm(update)
        return jnp.minimum(
            1.0,
            math.sqrt(NORM_CONSTRAINT) / jnp.maximum(raw_step_norm, EPS),
        )

    del optimizer_state
    gc.collect()
    rows = []
    started = time.perf_counter()

    for replicate in range(args.replicates):
        sample_key = jax.random.fold_in(base_key, replicate + 61001)
        data = base_data
        state = None
        spring_history = None
        snapshots = {}
        step_diagnostics = {}
        acceptances = []
        raw_variances = []
        previous_direction = None
        adjacent_cosines = []

        for step in range(1, max(DEPTHS) + 1):
            data, sample_key, acceptance, score, epsilon, raw = sample_batch(
                data, sample_key
            )
            jax.block_until_ready(score)
            require_finite("history_batch", acceptance, score, epsilon, raw)
            acceptances.append(scalar(acceptance))
            raw_variances.append(scalar(jnp.var(raw, ddof=1)))
            candidate_key = jax.random.fold_in(
                base_key, 710000 + 100 * replicate + step
            )

            if args.arm == "single_current":
                if step in DEPTHS:
                    fixed, _, active_rank, diagnostics = wssr_step(
                        score,
                        epsilon,
                        zero_state(score),
                        candidate_key,
                        jnp.asarray(0.3, dtype=score.dtype),
                    )
                    jax.block_until_ready(fixed)
                    require_finite("single_current", fixed, diagnostics)
                    snapshots[str(step)] = {
                        "fixed": fixed,
                    }
                    step_diagnostics[str(step)] = {
                        "active_rank": integer(active_rank),
                        "compression": [scalar(x) for x in diagnostics],
                        "eta_used": 0.0,
                    }
                    current_direction = fixed
                else:
                    current_direction = None
            elif args.arm in WSSR_ARMS:
                if state is None:
                    state = zero_state(score)
                target_eta, bias_corrected = WSSR_ARMS[args.arm]
                eta_step = effective_eta(target_eta, bias_corrected, step)
                old_state = state
                fixed, state, active_rank, diagnostics = wssr_step(
                    score,
                    epsilon,
                    state,
                    candidate_key,
                    jnp.asarray(eta_step, dtype=score.dtype),
                )
                jax.block_until_ready(fixed)
                require_finite("wssr", fixed, state.sr_o, state.ek, diagnostics)
                current_direction = fixed
                if step in DEPTHS:
                    probe_key = jax.random.fold_in(
                        base_key, 810000 + 100 * replicate + step
                    )
                    spectral_quality = history_spectral_quality(
                        old_state,
                        state,
                        jnp.asarray(eta_step, dtype=score.dtype),
                        probe_key,
                    )
                    require_finite("history_spectral_quality", spectral_quality)
                    snapshots[str(step)] = {
                        "fixed": fixed,
                    }
                    step_diagnostics[str(step)] = {
                        "active_rank": integer(active_rank),
                        "compression": [scalar(x) for x in diagnostics],
                        "history_spectral_quality_retention": scalar(
                            spectral_quality
                        ),
                        "eta_used": eta_step,
                    }
            else:
                if spring_history is None:
                    spring_history = jnp.zeros((score.shape[0],), dtype=score.dtype)
                spring_history = spring_step(score, epsilon, spring_history)
                jax.block_until_ready(spring_history)
                require_finite("spring", spring_history)
                current_direction = spring_history
                if step in DEPTHS:
                    snapshots[str(step)] = {"spring": spring_history}
                    step_diagnostics[str(step)] = {
                        "active_rank": score.shape[1] - 1,
                        "eta_used": SPRING_MU,
                    }

            if current_direction is not None and previous_direction is not None:
                adjacent_cosines.append(scalar(safe_cosine(
                    previous_direction, current_direction
                )))
            if current_direction is not None:
                previous_direction = current_direction
            del score, epsilon, raw
            gc.collect()

        future_force = None
        future_metrics = {
            depth: {name: {"q": [], "force_cosine": [], "force_dot": []}
                    for name in candidates}
            for depth, candidates in snapshots.items()
        }
        future_acceptances = []
        for _ in range(FUTURE_BATCHES):
            data, sample_key, acceptance, score, epsilon, raw = sample_batch(
                data, sample_key
            )
            jax.block_until_ready(score)
            force = score @ epsilon
            require_finite("future_batch", acceptance, score, epsilon, raw, force)
            future_force = force if future_force is None else future_force + force
            future_acceptances.append(scalar(acceptance))
            for depth, candidates in snapshots.items():
                for name, update in candidates.items():
                    metrics = future_metrics[depth][name]
                    metrics["q"].append(scalar(
                        residual_ratio(epsilon, score.T @ update)
                    ))
                    metrics["force_cosine"].append(scalar(
                        safe_cosine(force, update)
                    ))
                    metrics["force_dot"].append(scalar(jnp.vdot(force, update)))
            del score, epsilon, raw, force
            gc.collect()

        future_force = future_force / FUTURE_BATCHES
        depth_rows = {}
        for depth, candidates in snapshots.items():
            candidate_rows = {}
            for name, update in candidates.items():
                future = future_metrics[depth][name]
                candidate_rows[name] = {
                    "future_consensus_cosine": scalar(
                        safe_cosine(future_force, update)
                    ),
                    "future_mean_q": float(sum(future["q"]) / FUTURE_BATCHES),
                    "future_mean_force_cosine": float(
                        sum(future["force_cosine"]) / FUTURE_BATCHES
                    ),
                    "future_positive_force_fraction": float(
                        sum(value > 0.0 for value in future["force_dot"])
                        / FUTURE_BATCHES
                    ),
                    "raw_direction_norm": scalar(jnp.linalg.norm(update)),
                    "norm_constraint_scale": scalar(cap_scale(update)),
                }
            depth_rows[depth] = {
                "candidates": candidate_rows,
                "step_diagnostics": step_diagnostics[depth],
            }

        row = {
            "replicate": replicate,
            "acceptances": acceptances,
            "future_acceptances": future_acceptances,
            "raw_variances": raw_variances,
            "adjacent_direction_cosines": adjacent_cosines,
            "depths": depth_rows,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        del (
            data,
            state,
            spring_history,
            snapshots,
            future_metrics,
            future_force,
            previous_direction,
        )
        gc.collect()

    payload = {
        "experiment": "C rank1600 early memory-scaling audit",
        "read_only_checkpoint": True,
        "source_checkpoint": str(checkpoint),
        "stored_epoch": int(stored_epoch),
        "arm": args.arm,
        "replicates": args.replicates,
        "walkers_per_batch": NCHAINS,
        "history_depths": list(DEPTHS),
        "future_batches": FUTURE_BATCHES,
        "rank": RANK,
        "damping": DAMPING,
        "relative_singular_value_cutoff": CUTOFF,
        "fixed_lambda": FIXED_LAMBDA,
        "mixed_precision_spectral_solve": True,
        "jax_x64_enabled_globally": bool(jax.config.x64_enabled),
        "rank_space_host_callback_solve_dtype": "float64",
        "learning_rate_for_cap_diagnostic": LEARNING_RATE,
        "norm_constraint": NORM_CONSTRAINT,
        "spring_mu": SPRING_MU,
        "spring_damping": SPRING_DAMPING,
        "parameters_persisted": False,
        "optimizer_state_persisted": False,
        "sampler_state_persisted": False,
        "training_checkpoint_written": False,
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
