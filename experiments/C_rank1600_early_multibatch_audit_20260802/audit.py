#!/usr/bin/env python3
"""Read-only early-regime fixed-lambda single-vs-multibatch WSSR audit."""

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

from vmcnet.mcmc import position_amplitude_core as pacore
from vmcnet.physics import core as physics_core
from vmcnet.train import runners
from vmcnet.updates import wssr
from vmcnet.utils import io


NCHAINS = 1000
ETA = 0.3
RANK = 1600
DAMPING = 0.0003
CUTOFF = 0.0003
FIXED_LAMBDA = 0.001
LEARNING_RATE = 0.04
NORM_CONSTRAINT = 0.001
EPS = 1.0e-12


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


def main():
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
    if int(config.vmc.nchains) != NCHAINS:
        raise ValueError(f"expected {NCHAINS} walkers, got {config.vmc.nchains}")

    dtype = runners._get_dtype(config)
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

    def sample_batch(data, key):
        acceptance, data, key = walker_fn(params, data, key)
        positions = pacore.get_position_from_data(data)
        energy, clipped, stats = energy_fn(params, positions)
        score, _ = wssr.center_and_scale_score_matrix(log_psi, params, positions)
        epsilon = wssr.center_and_scale_energy_residuals(clipped, energy)
        return data, key, acceptance, score, epsilon, stats["local_energies_noclip"]

    @jax.jit
    def fixed_step(score, epsilon, state, candidate_key):
        augmented, augmented_rhs = wssr.augment_wssr_system(
            score, epsilon, state, ETA
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
        result = wssr._wssr_update_from_svd(
            augmented,
            augmented_rhs,
            state,
            u,
            singular_values,
            vh,
            damping=DAMPING,
            norm_constraint=NORM_CONSTRAINT,
            sr_rank_max=RANK,
            sr_scale=1.1,
            constrain_update_norm=False,
            rank_update_max=RANK,
            spectral_regularization="tikhonov",
            complement_weight=0.0,
            relative_singular_value_cutoff=CUTOFF,
            tikhonov_lambda=FIXED_LAMBDA,
            experimental_mode="none",
            eps=EPS,
        )
        u_state = jnp.zeros_like(state.u)
        warm_width = min(u.shape[1], state.u.shape[1])
        rank_mask = (jnp.arange(warm_width) < rank).astype(u.dtype)
        u_state = u_state.at[:, :warm_width].set(
            u[:, :warm_width] * rank_mask
        )
        new_state = wssr.WSSRWarmSVDCoreState(
            sr_o=result.state.sr_o,
            ek=result.state.ek,
            sr_rank0=result.state.sr_rank0,
            sr_rank=result.state.sr_rank,
            u=u_state,
            has_u=jnp.where(rank > 0, jnp.array(True), state.has_u),
        )
        leading = singular_values[0]
        return result.grad_like_update, new_state, result.active_rank, leading

    def zero_state(score):
        return wssr.initialize_wssr_warm_svd_core_state(
            score.shape[0], RANK, RANK, dtype=score.dtype, store_warm_u=True
        )

    def metrics(update, hold_score, hold_epsilon, hold_force):
        update_norm = jnp.linalg.norm(update)
        raw_step_norm = LEARNING_RATE * update_norm
        cap = math.sqrt(NORM_CONSTRAINT)
        cap_scale = jnp.minimum(1.0, cap / jnp.maximum(raw_step_norm, EPS))
        actual_step = LEARNING_RATE * cap_scale * update
        return {
            "heldout_q": scalar(
                residual_ratio(hold_epsilon, hold_score.T @ update)
            ),
            "heldout_force_dot_update": scalar(jnp.vdot(hold_force, update)),
            "heldout_force_update_cosine": scalar(safe_cosine(hold_force, update)),
            "raw_direction_norm": scalar(update_norm),
            "nominal_parameter_step_norm": scalar(raw_step_norm),
            "norm_constraint_scale": scalar(cap_scale),
            "actual_parameter_step_norm": scalar(jnp.linalg.norm(actual_step)),
        }

    del optimizer_state
    gc.collect()
    rows = []
    started = time.perf_counter()
    for replicate in range(args.replicates):
        key = jax.random.fold_in(base_key, replicate + 17001)
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

        multi_state = zero_state(batches[0][0])
        candidates = {}
        step_diagnostics = {}
        for batch_index in range(3):
            key, candidate_key = jax.random.split(key)
            score, epsilon = batches[batch_index]
            multi_update, multi_state, multi_rank, multi_leading = fixed_step(
                score, epsilon, multi_state, candidate_key
            )
            single_update, _, single_rank, single_leading = fixed_step(
                score, epsilon, zero_state(score), candidate_key
            )
            jax.block_until_ready(multi_update)
            depth = batch_index + 1
            if depth >= 2:
                candidates[depth] = (single_update, multi_update)
                step_diagnostics[depth] = {
                    "single_active_rank": integer(single_rank),
                    "multi_active_rank": integer(multi_rank),
                    "single_leading_singular_value": scalar(single_leading),
                    "multi_leading_singular_value": scalar(multi_leading),
                }

        hold_score, hold_epsilon = batches[3]
        hold_force = hold_score @ hold_epsilon
        depth_rows = {}
        for depth in (2, 3):
            single_update, multi_update = candidates[depth]
            single_metrics = metrics(
                single_update, hold_score, hold_epsilon, hold_force
            )
            multi_metrics = metrics(
                multi_update, hold_score, hold_epsilon, hold_force
            )
            depth_rows[str(depth)] = {
                "depth": depth,
                "single_batch_fixed_lambda": single_metrics,
                "multibatch_fixed_lambda": multi_metrics,
                "single_multi_direction_cosine": scalar(
                    safe_cosine(single_update, multi_update)
                ),
                "step_diagnostics": step_diagnostics[depth],
            }
            values = list(single_metrics.values()) + list(multi_metrics.values())
            if not all(math.isfinite(value) for value in values):
                raise FloatingPointError(depth_rows[str(depth)])

        row = {
            "replicate": replicate,
            "acceptances": acceptances,
            "raw_variances": raw_variances,
            "depths": depth_rows,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        del data, batches, multi_state, candidates, hold_score, hold_epsilon, hold_force
        gc.collect()

    payload = {
        "experiment": "C rank1600 early fixed-lambda single-vs-multibatch audit",
        "read_only_checkpoint": True,
        "source_checkpoint": str(checkpoint),
        "stored_epoch": int(stored_epoch),
        "replicates": args.replicates,
        "walkers_per_batch": NCHAINS,
        "eta": ETA,
        "rank": RANK,
        "damping": DAMPING,
        "relative_singular_value_cutoff": CUTOFF,
        "fixed_lambda": FIXED_LAMBDA,
        "learning_rate_for_cap_diagnostic": LEARNING_RATE,
        "norm_constraint": NORM_CONSTRAINT,
        "ssi_initial_iterations": 40,
        "ssi_warm_iterations": 2,
        "parameters_persisted": False,
        "optimizer_state_persisted": False,
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
