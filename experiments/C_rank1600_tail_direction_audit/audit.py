#!/usr/bin/env python3
"""Fixed-checkpoint audit of whether WSSR drops C local-energy tail forces.

The checkpoint, parameters, and optimizer state are never modified.  MCMC walkers are
advanced in memory to obtain independent batches.  For every batch we compare the
native rank-1600 WSSR direction against current-batch MinSR and against a one-vector
tail-subspace extension of the native WSSR direction.
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


DEFAULT_RUN = Path(
    "/scratch/dexuan1/runs/C_wssr_rank1600_fixed_lambda_stage2_E2000"
)
DEFAULT_EPOCH = 102000
NCHAINS = 1000
TOP_TAIL = 10
ROBUST_Z_THRESHOLD = 10.0
EPS = 1.0e-12


def scalar(value) -> float:
    return float(jax.device_get(value))


def safe_cosine(left, right):
    denominator = jnp.maximum(jnp.linalg.norm(left) * jnp.linalg.norm(right), EPS)
    return jnp.vdot(left, right) / denominator


def residual_ratio(target, prediction, mask):
    selected_target = target[mask]
    selected_residual = (target - prediction)[mask]
    return jnp.linalg.norm(selected_residual) / jnp.maximum(
        jnp.linalg.norm(selected_target), EPS
    )


def minsr_update(score, residual, lambda_reg):
    """Solve O^T (O O^T + lambda I)^-1 e in sample space."""
    gram = score.T @ score
    gram = 0.5 * (gram + gram.T)
    values, vectors = jnp.linalg.eigh(gram)
    values = jnp.maximum(values, 0.0)
    dual = vectors @ ((vectors.T @ residual) / (values + lambda_reg))
    return score @ dual


def constrained(update, norm_constraint):
    norm = jnp.linalg.norm(update)
    scale = jnp.minimum(1.0, jnp.sqrt(norm_constraint) / jnp.maximum(norm, EPS))
    return update * scale


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def classify(records):
    def column(name):
        return np.asarray([row[name] for row in records], dtype=np.float64)

    q_wssr = column("raw_tail_q_wssr")
    q_minsr = column("raw_tail_q_minsr_clipped")
    q_raw = column("raw_tail_q_minsr_raw")
    q_tail = column("raw_tail_q_wssr_plus_tail")
    wssr_minsr_gap = q_wssr - q_minsr
    clipping_gap = q_minsr - q_raw
    tail_gain = q_wssr - q_tail
    positive_gap = wssr_minsr_gap > 1.0e-8
    recovery = np.zeros_like(tail_gain)
    recovery[positive_gap] = tail_gain[positive_gap] / wssr_minsr_gap[positive_gap]

    med_leakage = float(np.median(column("raw_tail_active_leakage")))
    med_gap = float(np.median(wssr_minsr_gap))
    med_clipping_gap = float(np.median(clipping_gap))
    med_recovery = float(np.median(recovery[positive_gap])) if np.any(positive_gap) else 0.0

    # These thresholds were fixed before examining the audit output.  They require
    # both geometric leakage and a material loss in the actual tail residual.
    truncation_supported = (
        med_leakage >= 0.50 and med_gap >= 0.05 and med_recovery >= 0.50
    )
    clipping_supported = med_clipping_gap >= 0.05
    if truncation_supported:
        conclusion = "hard_truncation_supported"
    elif clipping_supported:
        conclusion = "clipping_dominant_or_shared"
    else:
        conclusion = "hard_truncation_not_supported"
    return {
        "pre_registered_thresholds": {
            "median_raw_tail_active_leakage_min": 0.50,
            "median_q_wssr_minus_minsr_clipped_min": 0.05,
            "median_tail_basis_gap_recovery_min": 0.50,
            "median_q_minsr_clipped_minus_minsr_raw_for_clipping": 0.05,
        },
        "median_raw_tail_active_leakage": med_leakage,
        "median_q_wssr_minus_minsr_clipped": med_gap,
        "median_q_minsr_clipped_minus_minsr_raw": med_clipping_gap,
        "median_tail_basis_gap_recovery": med_recovery,
        "hard_truncation_supported": bool(truncation_supported),
        "clipping_supported": bool(clipping_supported),
        "conclusion": conclusion,
    }


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
        "relative_cutoff": (float(opt.relative_singular_value_cutoff), 3.0e-4),
        "tikhonov_lambda": (float(opt.tikhonov_lambda), 1.0e-3),
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
    lambda_reg = jnp.asarray(opt.tikhonov_lambda, dtype=dtype)
    norm_constraint = jnp.asarray(opt.norm_constraint, dtype=dtype)

    records = []
    previous_tail = None
    consensus = None
    started = time.perf_counter()
    for batch in range(args.batches):
        acceptance, data, key = walker_fn(params, data, key)
        positions = pacore.get_position_from_data(data)
        energy, clipped_energies, stats = energy_fn(params, positions)
        raw_energies = stats["local_energies_noclip"]
        score, _ = wssr.center_and_scale_score_matrix(log_psi, params, positions)
        clipped_epsilon = wssr.center_and_scale_energy_residuals(
            clipped_energies, energy
        )
        raw_mean = jnp.mean(raw_energies)
        raw_epsilon = (raw_energies - raw_mean) / jnp.sqrt(NCHAINS)
        augmented, augmented_rhs = wssr.augment_wssr_system(
            score, clipped_epsilon, optimizer_state.core_state, opt.eta
        )
        force = augmented @ augmented_rhs
        key, svd_key = jax.random.split(key)
        u, singular_values, _, approx_rank = wssr._wssr_right_svd_decomposition(
            augmented,
            optimizer_state.core_state,
            svd_key,
            1600,
            opt.svd_maxiter_initial,
            opt.svd_maxiter_warm,
            False,
            EPS,
        )
        retained = (
            singular_values / jnp.maximum(jnp.abs(singular_values[0]), EPS)
            > opt.relative_singular_value_cutoff
        )
        retained = retained & (jnp.arange(singular_values.shape[0]) < approx_rank)
        active_u = u * retained.astype(u.dtype)[None, :]
        wssr_update = active_u @ (
            (active_u.T @ force) / (jnp.square(singular_values) + lambda_reg)
        )
        minsr_clipped = minsr_update(score, clipped_epsilon, lambda_reg)
        minsr_raw = minsr_update(score, raw_epsilon, lambda_reg)

        median = jnp.median(raw_energies)
        mad = jnp.median(jnp.abs(raw_energies - median))
        robust_z = jnp.abs(raw_energies - median) / jnp.maximum(1.4826 * mad, EPS)
        _, tail_indices = jax.lax.top_k(robust_z, TOP_TAIL)
        tail_mask = jnp.zeros((NCHAINS,), dtype=bool).at[tail_indices].set(True)
        raw_tail_residual = jnp.where(tail_mask, raw_epsilon, 0.0)
        clipped_tail_residual = jnp.where(tail_mask, clipped_epsilon, 0.0)
        raw_tail_force = score @ raw_tail_residual
        clipped_tail_force = score @ clipped_tail_residual

        raw_projection = active_u @ (active_u.T @ raw_tail_force)
        clipped_projection = active_u @ (active_u.T @ clipped_tail_force)
        raw_perp = raw_tail_force - raw_projection
        clipped_perp = clipped_tail_force - clipped_projection
        raw_perp_norm = jnp.linalg.norm(raw_perp)
        tail_basis = raw_perp / jnp.maximum(raw_perp_norm, EPS)

        # Add one raw-tail direction but choose its coefficient using the unchanged
        # clipped, history-augmented Tikhonov objective.  Improvement is therefore
        # not guaranteed by construction.
        normal_residual = force - (
            augmented @ (augmented.T @ wssr_update) + lambda_reg * wssr_update
        )
        basis_aug_projection = augmented.T @ tail_basis
        alpha = jnp.vdot(tail_basis, normal_residual) / jnp.maximum(
            jnp.vdot(basis_aug_projection, basis_aug_projection) + lambda_reg,
            EPS,
        )
        wssr_plus_tail = wssr_update + alpha * tail_basis

        predictions = {
            "wssr": score.T @ wssr_update,
            "minsr_clipped": score.T @ minsr_clipped,
            "minsr_raw": score.T @ minsr_raw,
            "wssr_plus_tail": score.T @ wssr_plus_tail,
            "wssr_constrained": score.T @ constrained(wssr_update, norm_constraint),
            "minsr_clipped_constrained": score.T @ constrained(
                minsr_clipped, norm_constraint
            ),
            "wssr_plus_tail_constrained": score.T @ constrained(
                wssr_plus_tail, norm_constraint
            ),
        }

        consecutive_cosine = math.nan
        consensus_cosine = math.nan
        if consensus is not None:
            consecutive_cosine = scalar(safe_cosine(tail_basis, previous_tail))
            consensus_cosine = scalar(safe_cosine(tail_basis, consensus))
            consensus = 0.75 * consensus + 0.25 * tail_basis
            consensus = consensus / jnp.maximum(jnp.linalg.norm(consensus), EPS)
        else:
            consensus = tail_basis
        previous_tail = tail_basis

        row = {
            "batch": batch,
            "acceptance": scalar(acceptance),
            "raw_energy_mean": scalar(raw_mean),
            "raw_variance": scalar(jnp.var(raw_energies, ddof=1)),
            "raw_energy_min": scalar(jnp.min(raw_energies)),
            "raw_energy_max": scalar(jnp.max(raw_energies)),
            "raw_max_robust_z": scalar(jnp.max(robust_z)),
            "robust_z_above_10": int(jax.device_get(jnp.sum(robust_z > ROBUST_Z_THRESHOLD))),
            "active_rank": int(jax.device_get(jnp.sum(retained))),
            "raw_tail_active_leakage": scalar(
                raw_perp_norm / jnp.maximum(jnp.linalg.norm(raw_tail_force), EPS)
            ),
            "clipped_tail_active_leakage": scalar(
                jnp.linalg.norm(clipped_perp)
                / jnp.maximum(jnp.linalg.norm(clipped_tail_force), EPS)
            ),
            "raw_vs_clipped_tail_force_cosine": scalar(
                safe_cosine(raw_tail_force, clipped_tail_force)
            ),
            "tail_clipped_force_over_raw_force": scalar(
                jnp.linalg.norm(clipped_tail_force)
                / jnp.maximum(jnp.linalg.norm(raw_tail_force), EPS)
            ),
            "tail_basis_alpha_standard_objective": scalar(alpha),
            "tail_basis_consecutive_cosine": consecutive_cosine,
            "tail_basis_consensus_cosine": consensus_cosine,
            "raw_tail_alignment_wssr": scalar(safe_cosine(raw_tail_force, wssr_update)),
            "raw_tail_alignment_minsr_clipped": scalar(
                safe_cosine(raw_tail_force, minsr_clipped)
            ),
            "raw_tail_alignment_minsr_raw": scalar(
                safe_cosine(raw_tail_force, minsr_raw)
            ),
            "raw_tail_alignment_wssr_plus_tail": scalar(
                safe_cosine(raw_tail_force, wssr_plus_tail)
            ),
            "raw_tail_q_wssr": scalar(
                residual_ratio(raw_epsilon, predictions["wssr"], tail_mask)
            ),
            "raw_tail_q_minsr_clipped": scalar(
                residual_ratio(raw_epsilon, predictions["minsr_clipped"], tail_mask)
            ),
            "raw_tail_q_minsr_raw": scalar(
                residual_ratio(raw_epsilon, predictions["minsr_raw"], tail_mask)
            ),
            "raw_tail_q_wssr_plus_tail": scalar(
                residual_ratio(raw_epsilon, predictions["wssr_plus_tail"], tail_mask)
            ),
            "raw_tail_q_wssr_constrained": scalar(
                residual_ratio(raw_epsilon, predictions["wssr_constrained"], tail_mask)
            ),
            "raw_tail_q_minsr_clipped_constrained": scalar(
                residual_ratio(
                    raw_epsilon, predictions["minsr_clipped_constrained"], tail_mask
                )
            ),
            "raw_tail_q_wssr_plus_tail_constrained": scalar(
                residual_ratio(
                    raw_epsilon, predictions["wssr_plus_tail_constrained"], tail_mask
                )
            ),
            "wssr_update_norm": scalar(jnp.linalg.norm(wssr_update)),
            "minsr_clipped_update_norm": scalar(jnp.linalg.norm(minsr_clipped)),
            "minsr_raw_update_norm": scalar(jnp.linalg.norm(minsr_raw)),
            "wssr_plus_tail_update_norm": scalar(jnp.linalg.norm(wssr_plus_tail)),
        }
        if not all(
            math.isfinite(value)
            for value in row.values()
            if isinstance(value, float) and not math.isnan(value)
        ):
            raise FloatingPointError(f"nonfinite batch metrics: {row}")
        records.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

        del (
            score, augmented, force, u, active_u, wssr_update, minsr_clipped,
            minsr_raw, raw_tail_force, clipped_tail_force, raw_perp, clipped_perp,
            wssr_plus_tail, predictions,
        )

    metric_names = [
        "raw_variance",
        "raw_max_robust_z",
        "raw_tail_active_leakage",
        "clipped_tail_active_leakage",
        "raw_vs_clipped_tail_force_cosine",
        "tail_clipped_force_over_raw_force",
        "raw_tail_alignment_wssr",
        "raw_tail_alignment_minsr_clipped",
        "raw_tail_alignment_minsr_raw",
        "raw_tail_alignment_wssr_plus_tail",
        "raw_tail_q_wssr",
        "raw_tail_q_minsr_clipped",
        "raw_tail_q_minsr_raw",
        "raw_tail_q_wssr_plus_tail",
    ]
    payload = {
        "experiment": "C rank1600 WSSR fixed-checkpoint tail-direction audit",
        "read_only_checkpoint": True,
        "parameters_updated": False,
        "optimizer_state_updated": False,
        "mcmc_walkers_advanced_in_memory": True,
        "source_checkpoint": str(checkpoint),
        "stored_epoch": int(stored_epoch),
        "batches": int(args.batches),
        "walkers_per_batch": NCHAINS,
        "mcmc_steps_between_batches": int(config.vmc.nsteps_per_param_update),
        "tail_definition": {
            "direction": f"top {TOP_TAIL} absolute robust-z walkers per batch",
            "event_count_threshold": ROBUST_Z_THRESHOLD,
        },
        "config_verified": {name: actual for name, (actual, _) in expected.items()},
        "aggregate": {
            name: summarize([row[name] for row in records]) for name in metric_names
        },
        "classification": classify(records),
        "per_batch": records,
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
