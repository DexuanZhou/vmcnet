#!/usr/bin/env python3
"""Fixed-checkpoint native SPRING audit matched to the C WSSR tail audit."""

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

from audit import (
    EPS,
    NCHAINS,
    ROBUST_Z_THRESHOLD,
    TOP_TAIL,
    constrained,
    minsr_update,
    residual_ratio,
    safe_cosine,
    scalar,
    summarize,
)


DEFAULT_RUN = Path("/scratch/dexuan1/runs/C_spring_official_kfac1000_E100000")
DEFAULT_EPOCH = 100000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--epoch", type=int, default=DEFAULT_EPOCH)
    parser.add_argument("--batches", type=int, default=20)
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
    opt = config.vmc.optimizer.spring
    expected = {
        "optimizer": (config.vmc.optimizer_type, "spring"),
        "nchains": (int(config.vmc.nchains), NCHAINS),
        "mu": (float(opt.mu), 0.99),
        "learning_rate": (float(opt.learning_rate), 0.02),
        "damping": (float(opt.damping), 1.0e-3),
        "norm_constraint": (float(opt.norm_constraint), 1.0e-3),
    }
    mismatches = {name: pair for name, pair in expected.items() if pair[0] != pair[1]}
    if mismatches:
        raise ValueError(f"unexpected SPRING source configuration: {mismatches}")

    dtype = runners._get_dtype(config)
    ions, charges, nelec = runners._get_electron_ion_config_as_arrays(config, dtype)
    positions = pacore.get_position_from_data(data)
    log_psi, _, _ = runners._get_and_init_model(
        config.model, ions, charges, nelec, positions, key,
        dtype=dtype, apply_pmap=False,
    )
    local_energy = runners._assemble_mol_local_energy_fn(
        ions, charges, config.problem.ei_softening, config.problem.ee_softening,
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
    spring_step = spring.get_spring_step(log_psi, opt.damping, opt.mu)
    previous_direction = optimizer_state[0].trace
    mu_history = jax.tree_util.tree_map(lambda value: opt.mu * value, previous_direction)
    mu_history_flat = jax.flatten_util.ravel_pytree(mu_history)[0]
    lambda_reg = jnp.asarray(opt.damping, dtype=dtype)
    norm_constraint = jnp.asarray(opt.norm_constraint, dtype=dtype)

    records = []
    started = time.perf_counter()
    for batch in range(args.batches):
        acceptance, data, key = walker_fn(params, data, key)
        positions = pacore.get_position_from_data(data)
        energy, clipped_energies, stats = energy_fn(params, positions)
        raw_energies = stats["local_energies_noclip"]
        centered_clipped = clipped_energies - energy
        clipped_epsilon = centered_clipped / jnp.sqrt(NCHAINS)
        raw_mean = jnp.mean(raw_energies)
        raw_epsilon = (raw_energies - raw_mean) / jnp.sqrt(NCHAINS)
        score, _ = wssr.center_and_scale_score_matrix(log_psi, params, positions)

        native_tree = spring_step(
            centered_clipped, params, previous_direction, positions
        )
        native = jax.flatten_util.ravel_pytree(native_tree)[0]
        current_correction = native - mu_history_flat
        minsr_clipped = minsr_update(score, clipped_epsilon, lambda_reg)
        minsr_raw = minsr_update(score, raw_epsilon, lambda_reg)

        median = jnp.median(raw_energies)
        mad = jnp.median(jnp.abs(raw_energies - median))
        robust_z = jnp.abs(raw_energies - median) / jnp.maximum(1.4826 * mad, EPS)
        _, tail_indices = jax.lax.top_k(robust_z, TOP_TAIL)
        tail_mask = jnp.zeros((NCHAINS,), dtype=bool).at[tail_indices].set(True)
        raw_tail_force = score @ jnp.where(tail_mask, raw_epsilon, 0.0)
        clipped_tail_force = score @ jnp.where(tail_mask, clipped_epsilon, 0.0)

        predictions = {
            "native": score.T @ native,
            "minsr_clipped": score.T @ minsr_clipped,
            "minsr_raw": score.T @ minsr_raw,
            "native_constrained": score.T @ constrained(native, norm_constraint),
            "minsr_clipped_constrained": score.T @ constrained(
                minsr_clipped, norm_constraint
            ),
        }
        row = {
            "batch": batch,
            "acceptance": scalar(acceptance),
            "raw_energy_mean": scalar(raw_mean),
            "raw_variance": scalar(jnp.var(raw_energies, ddof=1)),
            "raw_energy_min": scalar(jnp.min(raw_energies)),
            "raw_energy_max": scalar(jnp.max(raw_energies)),
            "raw_max_robust_z": scalar(jnp.max(robust_z)),
            "robust_z_above_10": int(jax.device_get(jnp.sum(robust_z > ROBUST_Z_THRESHOLD))),
            "raw_vs_clipped_tail_force_cosine": scalar(
                safe_cosine(raw_tail_force, clipped_tail_force)
            ),
            "tail_clipped_force_over_raw_force": scalar(
                jnp.linalg.norm(clipped_tail_force)
                / jnp.maximum(jnp.linalg.norm(raw_tail_force), EPS)
            ),
            "raw_tail_alignment_spring_native": scalar(
                safe_cosine(raw_tail_force, native)
            ),
            "raw_tail_alignment_spring_history": scalar(
                safe_cosine(raw_tail_force, mu_history_flat)
            ),
            "raw_tail_alignment_spring_current": scalar(
                safe_cosine(raw_tail_force, current_correction)
            ),
            "raw_tail_alignment_minsr_clipped": scalar(
                safe_cosine(raw_tail_force, minsr_clipped)
            ),
            "raw_tail_alignment_minsr_raw": scalar(
                safe_cosine(raw_tail_force, minsr_raw)
            ),
            "raw_tail_q_spring_native": scalar(
                residual_ratio(raw_epsilon, predictions["native"], tail_mask)
            ),
            "raw_tail_q_minsr_clipped": scalar(
                residual_ratio(raw_epsilon, predictions["minsr_clipped"], tail_mask)
            ),
            "raw_tail_q_minsr_raw": scalar(
                residual_ratio(raw_epsilon, predictions["minsr_raw"], tail_mask)
            ),
            "raw_tail_q_spring_native_constrained": scalar(
                residual_ratio(
                    raw_epsilon, predictions["native_constrained"], tail_mask
                )
            ),
            "raw_tail_q_minsr_clipped_constrained": scalar(
                residual_ratio(
                    raw_epsilon, predictions["minsr_clipped_constrained"], tail_mask
                )
            ),
            "spring_native_norm": scalar(jnp.linalg.norm(native)),
            "spring_history_norm": scalar(jnp.linalg.norm(mu_history_flat)),
            "spring_current_correction_norm": scalar(
                jnp.linalg.norm(current_correction)
            ),
            "minsr_clipped_update_norm": scalar(jnp.linalg.norm(minsr_clipped)),
            "minsr_raw_update_norm": scalar(jnp.linalg.norm(minsr_raw)),
        }
        if not all(
            math.isfinite(value)
            for value in row.values()
            if isinstance(value, float)
        ):
            raise FloatingPointError(f"nonfinite batch metrics: {row}")
        records.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    metric_names = [
        "raw_variance",
        "raw_max_robust_z",
        "raw_vs_clipped_tail_force_cosine",
        "tail_clipped_force_over_raw_force",
        "raw_tail_alignment_spring_native",
        "raw_tail_alignment_spring_history",
        "raw_tail_alignment_spring_current",
        "raw_tail_alignment_minsr_clipped",
        "raw_tail_alignment_minsr_raw",
        "raw_tail_q_spring_native",
        "raw_tail_q_minsr_clipped",
        "raw_tail_q_minsr_raw",
    ]
    q_native = np.asarray(
        [row["raw_tail_q_spring_native"] for row in records], dtype=np.float64
    )
    q_clipped = np.asarray(
        [row["raw_tail_q_minsr_clipped"] for row in records], dtype=np.float64
    )
    q_raw = np.asarray(
        [row["raw_tail_q_minsr_raw"] for row in records], dtype=np.float64
    )
    med_clipping_gap = float(np.median(q_clipped - q_raw))
    med_native_gap = float(np.median(q_native - q_clipped))
    shared_clipping = med_clipping_gap >= 0.05 and med_native_gap >= -0.05
    classification = {
        "pre_registered_thresholds": {
            "median_q_minsr_clipped_minus_minsr_raw_min": 0.05,
            "native_spring_advantage_over_minsr_clipped_required": 0.05,
        },
        "median_q_minsr_clipped_minus_minsr_raw": med_clipping_gap,
        "median_q_spring_native_minus_minsr_clipped": med_native_gap,
        "shared_clipping_confirmed": bool(shared_clipping),
        "conclusion": (
            "shared_clipping_confirmed"
            if shared_clipping
            else "spring_native_tail_response_requires_followup"
        ),
    }
    payload = {
        "experiment": "C SPRING fixed-checkpoint native tail-response audit",
        "read_only_checkpoint": True,
        "parameters_updated": False,
        "optimizer_state_updated": False,
        "mcmc_walkers_advanced_in_memory": True,
        "source_checkpoint": str(checkpoint),
        "stored_epoch": int(stored_epoch),
        "batches": int(args.batches),
        "walkers_per_batch": NCHAINS,
        "mcmc_steps_between_batches": int(config.vmc.nsteps_per_param_update),
        "tail_definition": f"top {TOP_TAIL} absolute robust-z walkers per batch",
        "config_verified": {name: actual for name, (actual, _) in expected.items()},
        "aggregate": {
            name: summarize([row[name] for row in records]) for name in metric_names
        },
        "classification": classification,
        "per_batch": records,
        "elapsed_seconds": time.perf_counter() - started,
        "finite": True,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
