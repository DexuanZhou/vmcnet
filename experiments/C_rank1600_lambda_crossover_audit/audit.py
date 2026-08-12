#!/usr/bin/env python3
"""Read-only same-checkpoint dynamic-vs-fixed lambda crossover."""

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
    opt = config.vmc.optimizer.wssr_warm_svd_right
    expected = {
        "nchains": (int(config.vmc.nchains), NCHAINS),
        "rank": (int(opt.sr_rank), RANK),
        "rank_max": (int(opt.sr_rank_max), RANK),
        "eta": (float(opt.eta), ETA),
        "learning_rate": (float(opt.learning_rate), 0.04),
        "damping": (float(opt.damping), DAMPING),
        "warm_iterations": (int(opt.svd_maxiter_warm), 2),
        "initial_iterations": (int(opt.svd_maxiter_initial), 40),
        "complement_weight": (float(opt.complement_weight), 0.0),
        "spectral_regularization": (str(opt.spectral_regularization), "tikhonov"),
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
    sr_scale = float(opt.get("sr_scale", 1.1))

    def sample_batch(data, key):
        acceptance, data, key = walker_fn(params, data, key)
        positions = pacore.get_position_from_data(data)
        energy, clipped, stats = energy_fn(params, positions)
        score, _ = wssr.center_and_scale_score_matrix(log_psi, params, positions)
        epsilon = wssr.center_and_scale_energy_residuals(clipped, energy)
        return data, key, acceptance, score, epsilon, stats["local_energies_noclip"]

    @jax.jit
    def dual_lambda_step(score, epsilon, state, candidate_key):
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
        common = dict(
            damping=DAMPING,
            norm_constraint=float(opt.norm_constraint),
            sr_rank_max=RANK,
            sr_scale=sr_scale,
            constrain_update_norm=False,
            rank_update_max=RANK,
            spectral_regularization="tikhonov",
            complement_weight=0.0,
            relative_singular_value_cutoff=CUTOFF,
            experimental_mode="none",
            eps=EPS,
        )
        dynamic = wssr._wssr_update_from_svd(
            augmented,
            augmented_rhs,
            state,
            u,
            singular_values,
            vh,
            tikhonov_lambda=-1.0,
            **common,
        )
        fixed = wssr._wssr_update_from_svd(
            augmented,
            augmented_rhs,
            state,
            u,
            singular_values,
            vh,
            tikhonov_lambda=FIXED_LAMBDA,
            **common,
        )
        u_state = jnp.zeros_like(state.u)
        warm_width = min(u.shape[1], state.u.shape[1])
        rank_mask = (jnp.arange(warm_width) < rank).astype(u.dtype)
        u_state = u_state.at[:, :warm_width].set(
            u[:, :warm_width] * rank_mask
        )
        new_state = wssr.WSSRWarmSVDCoreState(
            sr_o=dynamic.state.sr_o,
            ek=dynamic.state.ek,
            sr_rank0=dynamic.state.sr_rank0,
            sr_rank=dynamic.state.sr_rank,
            u=u_state,
            has_u=jnp.where(rank > 0, jnp.array(True), state.has_u),
        )
        leading = singular_values[0]
        dynamic_lambda = jnp.square(DAMPING * jnp.abs(leading))
        state_force_delta = jnp.linalg.norm(dynamic.state.sr_o - fixed.state.sr_o)
        state_rhs_delta = jnp.linalg.norm(dynamic.state.ek - fixed.state.ek)
        return (
            dynamic.grad_like_update,
            fixed.grad_like_update,
            new_state,
            dynamic.active_rank,
            leading,
            singular_values[-1],
            dynamic_lambda,
            state_force_delta,
            state_rhs_delta,
        )

    del optimizer_state
    gc.collect()
    rows = []
    started = time.perf_counter()
    for replicate in range(args.replicates):
        key = jax.random.fold_in(base_key, replicate + 9001)
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

        state = wssr.initialize_wssr_warm_svd_core_state(
            batches[0][0].shape[0],
            RANK,
            RANK,
            dtype=batches[0][0].dtype,
            store_warm_u=True,
        )
        candidates = {}
        step_diagnostics = {}
        for batch_index in range(3):
            key, candidate_key = jax.random.split(key)
            result = dual_lambda_step(
                batches[batch_index][0],
                batches[batch_index][1],
                state,
                candidate_key,
            )
            jax.block_until_ready(result[0])
            dynamic_update, fixed_update, state = result[:3]
            depth = batch_index + 1
            if depth >= 2:
                candidates[depth] = (dynamic_update, fixed_update)
                step_diagnostics[depth] = {
                    "active_rank": integer(result[3]),
                    "leading_singular_value": scalar(result[4]),
                    "tail_singular_value": scalar(result[5]),
                    "dynamic_lambda": scalar(result[6]),
                    "fixed_lambda": FIXED_LAMBDA,
                    "history_factor_delta_between_lambdas": scalar(result[7]),
                    "history_rhs_delta_between_lambdas": scalar(result[8]),
                }
            del result

        hold_score, hold_epsilon = batches[3]
        hold_force = hold_score @ hold_epsilon
        depth_rows = {}
        for depth in (2, 3):
            dynamic_update, fixed_update = candidates[depth]
            metrics = {}
            for name, update in (
                ("dynamic_lambda", dynamic_update),
                ("fixed_lambda", fixed_update),
            ):
                prediction = hold_score.T @ update
                metrics[name] = {
                    "heldout_q": scalar(residual_ratio(hold_epsilon, prediction)),
                    "heldout_force_dot_update": scalar(jnp.vdot(hold_force, update)),
                    "heldout_force_update_cosine": scalar(
                        safe_cosine(hold_force, update)
                    ),
                    "update_norm": scalar(jnp.linalg.norm(update)),
                }
            metrics["fixed_dynamic_direction_cosine"] = scalar(
                safe_cosine(fixed_update, dynamic_update)
            )
            depth_rows[str(depth)] = {
                "depth": depth,
                "metrics": metrics,
                "step_diagnostics": step_diagnostics[depth],
            }
            if not all(
                math.isfinite(value)
                for name in ("dynamic_lambda", "fixed_lambda")
                for value in metrics[name].values()
            ):
                raise FloatingPointError(depth_rows[str(depth)])

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
            state,
            candidates,
            step_diagnostics,
            hold_score,
            hold_epsilon,
            hold_force,
            depth_rows,
        )
        gc.collect()

    payload = {
        "experiment": "C rank1600 same-checkpoint lambda crossover audit",
        "read_only_checkpoint": True,
        "source_checkpoint": str(checkpoint),
        "stored_epoch": int(stored_epoch),
        "replicates": args.replicates,
        "walkers_per_batch": NCHAINS,
        "eta": ETA,
        "rank": RANK,
        "damping": DAMPING,
        "relative_singular_value_cutoff": CUTOFF,
        "dynamic_lambda_rule": "(damping * leading_singular_value)^2",
        "fixed_lambda": FIXED_LAMBDA,
        "ssi_initial_iterations": 40,
        "ssi_warm_iterations": 2,
        "parameters_persisted": False,
        "optimizer_state_persisted": False,
        "training_checkpoint_written": False,
        "config_verified": {name: actual for name, (actual, _) in expected.items()},
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
