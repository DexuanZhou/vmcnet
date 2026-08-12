#!/usr/bin/env python3
"""Read-only fixed-theta WSSR history-compression fidelity audit."""

from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import time
from functools import partial
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
ETA = 0.3
RANK = 1600
EPS = 1.0e-12


def scalar(x) -> float:
    return float(jax.device_get(x))


def integer(x) -> int:
    return int(jax.device_get(x))


def safe_cosine(x, y):
    return jnp.vdot(x, y) / jnp.maximum(
        jnp.linalg.norm(x) * jnp.linalg.norm(y), EPS
    )


def residual_ratio(target, prediction):
    return jnp.linalg.norm(target - prediction) / jnp.maximum(
        jnp.linalg.norm(target), EPS
    )


def optional_float(config, name, default):
    value = config.get(name, default)
    return float(default if value is None else value)


def raw_weights(depth: int) -> tuple[float, ...]:
    if depth < 1:
        raise ValueError(depth)
    if depth == 1:
        return (1.0,)
    weights = [ETA ** (depth - 1)]
    weights.extend((1.0 - ETA) * ETA ** (depth - 1 - i) for i in range(1, depth))
    if not np.isclose(sum(weights), 1.0):
        raise AssertionError(weights)
    return tuple(weights)


@jax.jit
def spectrum(matrix):
    gram = matrix.T @ matrix
    gram = 0.5 * (gram + gram.T)
    values, vectors = jnp.linalg.eigh(gram)
    values = jnp.maximum(values[::-1], 0.0)
    vectors = vectors[:, ::-1]
    return values, vectors


@partial(jax.jit, static_argnames=("rank", "make_factor"))
def solve_from_spectrum(
    matrix,
    rhs,
    values,
    vectors,
    rank: int,
    relative_cutoff,
    damping,
    tikhonov_lambda,
    make_factor: bool,
):
    singular_values = jnp.sqrt(values)
    leading = jnp.maximum(singular_values[0], EPS)
    configured_lambda = jnp.asarray(tikhonov_lambda, dtype=matrix.dtype)
    legacy_lambda = jnp.square(jnp.asarray(damping, matrix.dtype) * leading)
    lambda_reg = jnp.where(configured_lambda >= 0.0, configured_lambda, legacy_lambda)
    lambda_reg = jnp.maximum(lambda_reg, EPS)
    configured_cutoff = jnp.asarray(relative_cutoff, dtype=matrix.dtype)
    cutoff = jnp.where(configured_cutoff >= 0.0, configured_cutoff, damping)
    retained = singular_values / leading > cutoff
    selected = retained & (jnp.arange(values.shape[0]) < rank)
    projected_rhs = vectors.T @ rhs
    coefficients = (
        projected_rhs
        / jnp.maximum(values + lambda_reg, EPS)
        * selected.astype(matrix.dtype)
    )
    update = matrix @ (vectors @ coefficients)
    if make_factor:
        width = min(rank, matrix.shape[1])
        factor = matrix @ vectors[:, :width]
        factor_rhs = projected_rhs[:width]
        singular_head = singular_values[:width]
        retained_head = retained[:width]
    else:
        factor = jnp.zeros((matrix.shape[0], 0), dtype=matrix.dtype)
        factor_rhs = jnp.zeros((0,), dtype=matrix.dtype)
        singular_head = jnp.zeros((0,), dtype=matrix.dtype)
        retained_head = jnp.zeros((0,), dtype=jnp.bool_)
    return (
        update,
        factor,
        factor_rhs,
        singular_head,
        retained_head,
        lambda_reg,
        jnp.sum(selected.astype(jnp.int32)),
    )


@partial(jax.jit, static_argnames=("nprobes",))
def operator_probe_metrics(raw_matrix, factor, key, nprobes=3):
    coeff = jax.random.normal(key, (raw_matrix.shape[1], nprobes), raw_matrix.dtype)
    probes = raw_matrix @ coeff
    raw_action = raw_matrix @ (raw_matrix.T @ probes)
    factor_action = factor @ (factor.T @ probes)
    relative = jnp.linalg.norm(factor_action - raw_action, axis=0) / jnp.maximum(
        jnp.linalg.norm(raw_action, axis=0), EPS
    )
    cosines = jnp.sum(factor_action * raw_action, axis=0) / jnp.maximum(
        jnp.linalg.norm(factor_action, axis=0) * jnp.linalg.norm(raw_action, axis=0),
        EPS,
    )
    return jnp.median(relative), jnp.median(cosines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
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
    if config.vmc.optimizer_type != "wssr_warm_svd_right":
        raise ValueError(config.vmc.optimizer_type)
    opt = config.vmc.optimizer.wssr_warm_svd_right
    expected = {
        "nchains": (int(config.vmc.nchains), NCHAINS),
        "rank": (int(opt.sr_rank), RANK),
        "rank_max": (int(opt.sr_rank_max), RANK),
        "eta": (float(opt.eta), ETA),
        "learning_rate": (float(opt.learning_rate), 0.04),
        "warm_iterations": (int(opt.svd_maxiter_warm), 2),
        "initial_iterations": (int(opt.svd_maxiter_initial), 40),
        "complement_weight": (float(opt.complement_weight), 0.0),
        "experimental_mode": (str(opt.experimental_mode), "none"),
    }
    mismatches = {name: pair for name, pair in expected.items() if pair[0] != pair[1]}
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
    energy_fn = jax.jit(
        physics_core.create_energy_and_statistics_fn(
            local_energy,
            NCHAINS,
            clipping_fn,
            config.vmc.nan_safe,
            record_raw_local_energies=True,
        )
    )
    _, walker_fn = runners._get_mcmc_fns(config.vmc, log_psi, apply_pmap=False)

    damping = float(opt.damping)
    relative_cutoff = optional_float(opt, "relative_singular_value_cutoff", -1.0)
    tikhonov_lambda = optional_float(opt, "tikhonov_lambda", -1.0)
    spectral_regularization = str(opt.spectral_regularization)
    if spectral_regularization != "tikhonov":
        raise ValueError(spectral_regularization)
    sr_scale = float(opt.get("sr_scale", 1.1))

    def sample_batch(data, key):
        acceptance, data, key = walker_fn(params, data, key)
        positions = pacore.get_position_from_data(data)
        energy, clipped, stats = energy_fn(params, positions)
        score, _ = wssr.center_and_scale_score_matrix(log_psi, params, positions)
        epsilon = wssr.center_and_scale_energy_residuals(clipped, energy)
        return data, key, acceptance, score, epsilon, stats["local_energies_noclip"]

    @jax.jit
    def production_step(score, epsilon, state, candidate_key):
        augmented, augmented_rhs = wssr.augment_wssr_system(
            score, epsilon, state, ETA
        )
        result = wssr.wssr_warm_svd_right_core_update(
            augmented,
            augmented_rhs,
            state,
            candidate_key,
            damping,
            float(opt.norm_constraint),
            RANK,
            sr_scale=sr_scale,
            svd_maxiter_initial=int(opt.svd_maxiter_initial),
            svd_maxiter_warm=int(opt.svd_maxiter_warm),
            exact_first=False,
            exact_first_force=False,
            svd_working_rank=RANK,
            constrain_update_norm=False,
            spectral_regularization=spectral_regularization,
            complement_weight=0.0,
            relative_singular_value_cutoff=relative_cutoff,
            tikhonov_lambda=tikhonov_lambda,
            experimental_mode="none",
            eps=EPS,
        )
        return result

    def exact_candidate(matrix, rhs, rank, make_factor):
        values, vectors = spectrum(matrix)
        result = solve_from_spectrum(
            matrix,
            rhs,
            values,
            vectors,
            rank,
            relative_cutoff,
            damping,
            tikhonov_lambda,
            make_factor,
        )
        jax.block_until_ready(result[0])
        return result, values, vectors

    def candidate_metrics(update, hold_score, hold_epsilon, raw_reference):
        hold_force = hold_score @ hold_epsilon
        prediction = hold_score.T @ update
        return {
            "heldout_q": scalar(residual_ratio(hold_epsilon, prediction)),
            "heldout_force_dot_update": scalar(jnp.vdot(hold_force, update)),
            "heldout_force_update_cosine": scalar(safe_cosine(hold_force, update)),
            "update_norm": scalar(jnp.linalg.norm(update)),
            "direction_cosine_to_raw_full": scalar(
                safe_cosine(update, raw_reference)
            ),
        }

    # The checkpoint optimizer state is intentionally not used as history.  The
    # audit rebuilds a matched fixed-theta history from sampled raw batches.
    del optimizer_state
    gc.collect()

    rows = []
    started = time.perf_counter()
    for replicate in range(args.replicates):
        key = jax.random.fold_in(base_key, replicate + 4001)
        data = base_data
        batches = []
        acceptances = []
        raw_variances = []
        for _ in range(4):
            data, key, acceptance, score, epsilon, raw = sample_batch(data, key)
            jax.block_until_ready(score)
            batches.append((score, epsilon))
            acceptances.append(scalar(acceptance))
            raw_variances.append(scalar(jnp.var(raw, ddof=1)))
            del raw

        hold_score, hold_epsilon = batches[3]
        num_params = batches[0][0].shape[0]

        # Recursively compressed exact factors.  The unfiltered and production-
        # masked histories are carried separately.
        unfiltered_factor = None
        unfiltered_rhs = None
        production_exact_factor = None
        production_exact_rhs = None
        depth_rows = {}

        # Actual production SSI recurrence starts from a fresh empty history.
        production_state = wssr.initialize_wssr_warm_svd_core_state(
            num_params, RANK, RANK, dtype=batches[0][0].dtype, store_warm_u=True
        )
        for batch_index in range(3):
            score, epsilon = batches[batch_index]
            if batch_index == 0:
                unfiltered_matrix = score
                unfiltered_aug_rhs = epsilon
                production_exact_matrix = score
                production_exact_aug_rhs = epsilon
            else:
                unfiltered_matrix = jnp.concatenate(
                    [jnp.sqrt(ETA) * unfiltered_factor,
                     jnp.sqrt(1.0 - ETA) * score], axis=1
                )
                unfiltered_aug_rhs = jnp.concatenate(
                    [jnp.sqrt(ETA) * unfiltered_rhs,
                     jnp.sqrt(1.0 - ETA) * epsilon], axis=0
                )
                production_exact_matrix = jnp.concatenate(
                    [jnp.sqrt(ETA) * production_exact_factor,
                     jnp.sqrt(1.0 - ETA) * score], axis=1
                )
                production_exact_aug_rhs = jnp.concatenate(
                    [jnp.sqrt(ETA) * production_exact_rhs,
                     jnp.sqrt(1.0 - ETA) * epsilon], axis=0
                )

            unfiltered_result, unfiltered_values, unfiltered_vectors = exact_candidate(
                unfiltered_matrix, unfiltered_aug_rhs, RANK, True
            )
            (
                unfiltered_update,
                next_unfiltered_factor,
                next_unfiltered_rhs,
                _,
                unfiltered_retained,
                unfiltered_lambda,
                unfiltered_active,
            ) = unfiltered_result

            production_exact_result, production_values, production_vectors = (
                exact_candidate(
                    production_exact_matrix, production_exact_aug_rhs, RANK, True
                )
            )
            (
                production_exact_update,
                next_production_exact_factor,
                next_production_exact_rhs,
                _,
                production_exact_retained,
                production_exact_lambda,
                production_exact_active,
            ) = production_exact_result
            next_production_exact_factor = (
                next_production_exact_factor
                * production_exact_retained.astype(score.dtype)[None, :]
            )
            next_production_exact_rhs = (
                next_production_exact_rhs
                * production_exact_retained.astype(score.dtype)
            )

            key, production_key = jax.random.split(key)
            production_result = production_step(
                score, epsilon, production_state, production_key
            )
            jax.block_until_ready(production_result.grad_like_update)
            production_state = production_result.state

            unfiltered_factor = next_unfiltered_factor
            unfiltered_rhs = next_unfiltered_rhs
            production_exact_factor = next_production_exact_factor
            production_exact_rhs = next_production_exact_rhs

            depth = batch_index + 1
            if depth >= 2:
                current_score, current_epsilon = batches[depth - 1]
                weights = raw_weights(depth)
                raw_matrix = jnp.concatenate(
                    [
                        jnp.sqrt(weight) * batches[i][0]
                        for i, weight in enumerate(weights)
                    ],
                    axis=1,
                )
                raw_rhs = jnp.concatenate(
                    [
                        jnp.sqrt(weight) * batches[i][1]
                        for i, weight in enumerate(weights)
                    ],
                    axis=0,
                )
                raw_values, raw_vectors = spectrum(raw_matrix)
                raw_full = solve_from_spectrum(
                    raw_matrix,
                    raw_rhs,
                    raw_values,
                    raw_vectors,
                    raw_matrix.shape[1],
                    relative_cutoff,
                    damping,
                    tikhonov_lambda,
                    False,
                )
                raw_rank = solve_from_spectrum(
                    raw_matrix,
                    raw_rhs,
                    raw_values,
                    raw_vectors,
                    RANK,
                    relative_cutoff,
                    damping,
                    tikhonov_lambda,
                    True,
                )
                current_values, current_vectors = spectrum(current_score)
                current = solve_from_spectrum(
                    current_score,
                    current_epsilon,
                    current_values,
                    current_vectors,
                    current_score.shape[1],
                    relative_cutoff,
                    damping,
                    tikhonov_lambda,
                    False,
                )
                jax.block_until_ready(raw_rank[0])

                raw_full_update = raw_full[0]
                raw_rank_update, raw_rank_factor, raw_rank_rhs = raw_rank[:3]
                candidates = {
                    "single_current_exact": current[0],
                    "raw_full_exact": raw_full_update,
                    "raw_rank1600_exact": raw_rank_update,
                    "recursive_factor_rank1600_exact": unfiltered_update,
                    "recursive_production_exact": production_exact_update,
                    "recursive_production_ssi": production_result.grad_like_update,
                }
                metrics = {
                    name: candidate_metrics(
                        update, hold_score, hold_epsilon, raw_full_update
                    )
                    for name, update in candidates.items()
                }

                raw_force = raw_matrix @ raw_rhs
                state_diagnostics = {}
                factor_candidates = {
                    "raw_rank1600_exact": (raw_rank_factor, raw_rank_rhs),
                    "recursive_factor_rank1600_exact": (
                        unfiltered_factor,
                        unfiltered_rhs,
                    ),
                    "recursive_production_exact": (
                        production_exact_factor,
                        production_exact_rhs,
                    ),
                    "recursive_production_ssi": (
                        production_state.sr_o,
                        production_state.ek,
                    ),
                }
                for name, (factor, factor_rhs) in factor_candidates.items():
                    key, probe_key = jax.random.split(key)
                    probe_rel, probe_cos = operator_probe_metrics(
                        raw_matrix, factor, probe_key
                    )
                    factor_force = factor @ factor_rhs
                    state_diagnostics[name] = {
                        "force_relative_error_to_raw": scalar(
                            jnp.linalg.norm(factor_force - raw_force)
                            / jnp.maximum(jnp.linalg.norm(raw_force), EPS)
                        ),
                        "force_cosine_to_raw": scalar(
                            safe_cosine(factor_force, raw_force)
                        ),
                        "gram_probe_median_relative_error": scalar(probe_rel),
                        "gram_probe_median_cosine": scalar(probe_cos),
                    }

                depth_rows[str(depth)] = {
                    "depth": depth,
                    "raw_weights": list(weights),
                    "raw_operator_shape": list(map(int, raw_matrix.shape)),
                    "raw_full_active_rank": integer(raw_full[6]),
                    "raw_rank1600_active_rank": integer(raw_rank[6]),
                    "recursive_factor_active_rank": integer(unfiltered_active),
                    "recursive_production_exact_active_rank": integer(
                        production_exact_active
                    ),
                    "recursive_production_ssi_active_rank": integer(
                        production_result.active_rank
                    ),
                    "raw_full_lambda": scalar(raw_full[5]),
                    "raw_rank1600_lambda": scalar(raw_rank[5]),
                    "recursive_factor_lambda": scalar(unfiltered_lambda),
                    "recursive_production_exact_lambda": scalar(
                        production_exact_lambda
                    ),
                    "metrics": metrics,
                    "state_diagnostics": state_diagnostics,
                }
                if not all(
                    math.isfinite(value)
                    for candidate in metrics.values()
                    for value in candidate.values()
                ):
                    raise FloatingPointError(depth_rows[str(depth)])

                del (
                    raw_matrix,
                    raw_rhs,
                    raw_values,
                    raw_vectors,
                    raw_full,
                    raw_rank,
                    current_values,
                    current_vectors,
                    current,
                    raw_full_update,
                    raw_rank_update,
                    raw_rank_factor,
                    raw_rank_rhs,
                    candidates,
                    metrics,
                    raw_force,
                    state_diagnostics,
                    factor_candidates,
                )
                gc.collect()

            del (
                unfiltered_matrix,
                unfiltered_aug_rhs,
                production_exact_matrix,
                production_exact_aug_rhs,
                unfiltered_result,
                unfiltered_values,
                unfiltered_vectors,
                production_exact_result,
                production_values,
                production_vectors,
                next_unfiltered_factor,
                next_unfiltered_rhs,
                next_production_exact_factor,
                next_production_exact_rhs,
                production_result,
            )
            gc.collect()

        row = {
            "replicate": replicate,
            "acceptances": acceptances,
            "raw_variances": raw_variances,
            "depths": depth_rows,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

        del (
            data,
            batches,
            hold_score,
            hold_epsilon,
            unfiltered_factor,
            unfiltered_rhs,
            production_exact_factor,
            production_exact_rhs,
            production_state,
            depth_rows,
        )
        gc.collect()

    payload = {
        "experiment": "C rank1600 fixed-theta history-compression fidelity audit",
        "read_only_checkpoint": True,
        "source_checkpoint": str(checkpoint),
        "stored_epoch": int(stored_epoch),
        "replicates": args.replicates,
        "walkers_per_batch": NCHAINS,
        "depths": [2, 3],
        "eta": ETA,
        "rank": RANK,
        "spectral_controls": {
            "spectral_regularization": spectral_regularization,
            "damping": damping,
            "relative_singular_value_cutoff": relative_cutoff,
            "tikhonov_lambda": tikhonov_lambda,
        },
        "production_ssi": {
            "initial_iterations": int(opt.svd_maxiter_initial),
            "warm_iterations": int(opt.svd_maxiter_warm),
        },
        "parameters_persisted": False,
        "optimizer_state_persisted": False,
        "training_checkpoint_written": False,
        "config_verified": {name: actual for name, (actual, _) in expected.items()},
        "pre_registered_protocol": str(
            Path(__file__).resolve().with_name("README.md")
        ),
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
