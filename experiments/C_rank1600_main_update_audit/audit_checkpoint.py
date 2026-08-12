#!/usr/bin/env python3
"""Read-only one-step audit of the C rank-1600 WSSR main update."""

import argparse
import json
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


def sync(value):
    return jax.block_until_ready(value)


def scalar(value):
    return float(jax.device_get(value))


def vector_metrics(value, reference):
    value_norm = jnp.linalg.norm(value)
    reference_norm = jnp.linalg.norm(reference)
    denominator = jnp.maximum(reference_norm, 1e-30)
    cosine_denominator = jnp.maximum(value_norm * reference_norm, 1e-30)
    return {
        "relative_error": scalar(jnp.linalg.norm(value - reference) / denominator),
        "cosine": scalar(jnp.vdot(value, reference) / cosine_denominator),
        "norm": scalar(value_norm),
        "reference_norm": scalar(reference_norm),
    }


def exact_update_from_sample_eigh(matrix, rhs, vectors, values, rank, damping):
    """Return A V_r (V_r^T e / (sigma_r^2 + lambda))."""
    selected_vectors = vectors[:, :rank]
    selected_values = values[:rank]
    coefficients = (selected_vectors.T @ rhs) / (selected_values + damping)
    return matrix @ (selected_vectors @ coefficients)


def normal_residual(matrix, force, update, damping):
    residual = matrix @ (matrix.T @ update) + damping * update - force
    return scalar(jnp.linalg.norm(residual) / jnp.maximum(jnp.linalg.norm(force), 1e-30))


def current_metrics(score, residuals, force, update):
    predicted = score.T @ update
    return {
        "current_linear_residual": scalar(
            jnp.linalg.norm(predicted - residuals)
            / jnp.maximum(jnp.linalg.norm(residuals), 1e-30)
        ),
        "current_fisher_norm_sq": scalar(jnp.vdot(predicted, predicted)),
        "current_force_dot_update": scalar(jnp.vdot(force, update)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    run = Path(args.run)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    config = io.load_config_dict(str(run), "config.json")
    checkpoint = run / "checkpoints" / f"{args.epoch}.npz"
    stored_epoch, data, params, optimizer_state, key = io.reload_vmc_state(
        str(checkpoint.parent), checkpoint.name
    )
    # VMCNet checkpoint filenames are one-based update counts, while the epoch
    # stored inside the checkpoint is the corresponding zero-based loop index.
    expected_stored_epoch = args.epoch - 1
    if int(stored_epoch) != expected_stored_epoch:
        raise ValueError(
            f"stored epoch {stored_epoch} != expected zero-based epoch "
            f"{expected_stored_epoch} for checkpoint {args.epoch}.npz"
        )

    dtype = runners._get_dtype(config)
    ions, charges, nelec = runners._get_electron_ion_config_as_arrays(config, dtype)
    positions = pacore.get_position_from_data(data)
    if positions.shape[0] != 1000:
        raise ValueError(f"expected 1000 walkers, got {positions.shape[0]}")
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
    energy_fn = physics_core.create_energy_and_statistics_fn(
        local_energy,
        1000,
        runners._get_clipping_fn(config.vmc),
        config.vmc.nan_safe,
    )
    energy, local_energies, _ = energy_fn(params, positions)
    score, _ = wssr.center_and_scale_score_matrix(log_psi, params, positions)
    epsilon = wssr.center_and_scale_energy_residuals(local_energies, energy)
    opt = config.vmc.optimizer.wssr_warm_svd_right
    augmented, augmented_rhs = wssr.augment_wssr_system(
        score, epsilon, optimizer_state.core_state, opt.eta
    )
    force = augmented @ augmented_rhs
    current_force = score @ epsilon
    _, svd_key = jax.random.split(key)
    sync((energy, local_energies, score, augmented, force))

    # Actual warm-SSI decomposition at this checkpoint.
    started = time.perf_counter()
    approx_u, approx_s, approx_vh, approx_rank = wssr._wssr_right_svd_decomposition(
        augmented,
        optimizer_state.core_state,
        svd_key,
        1600,
        opt.svd_maxiter_initial,
        opt.svd_maxiter_warm,
        False,
        1e-12,
    )
    sync((approx_u, approx_s, approx_vh))
    approx_seconds = time.perf_counter() - started
    approx_lambda = jnp.square(opt.damping * jnp.abs(approx_s[0]))
    approx_retained = approx_s / jnp.maximum(jnp.abs(approx_s[0]), 1e-30) > opt.damping
    approx_coefficients = (approx_u.T @ force) / (jnp.square(approx_s) + approx_lambda)
    approx_update = approx_u @ (approx_coefficients * approx_retained.astype(approx_s.dtype))
    sync(approx_update)

    # Exact sample-space decomposition of A^T A, ordered from largest to smallest.
    started = time.perf_counter()
    gram = augmented.T @ augmented
    gram = 0.5 * (gram + gram.T)
    values_ascending, vectors_ascending = jnp.linalg.eigh(gram)
    order = jnp.argsort(values_ascending)[::-1]
    values = jnp.maximum(values_ascending[order], 0.0)
    vectors = vectors_ascending[:, order]
    sync((values, vectors))
    exact_seconds = time.perf_counter() - started
    singular_values = jnp.sqrt(values)
    relative_lambda = jnp.square(opt.damping * singular_values[0])
    fixed_lambda = jnp.asarray(0.001, dtype=values.dtype)
    full_rank = int(augmented.shape[1])

    exact_full_relative = exact_update_from_sample_eigh(
        augmented, augmented_rhs, vectors, values, full_rank, relative_lambda
    )
    exact_full_fixed = exact_update_from_sample_eigh(
        augmented, augmented_rhs, vectors, values, full_rank, fixed_lambda
    )
    sync((exact_full_relative, exact_full_fixed))

    rows = []
    exact_rank1600_relative = None
    exact_rank1600_fixed = None
    for rank in (400, 800, 1200, 1600, 2000, full_rank):
        if rank > full_rank:
            continue
        relative_update = exact_update_from_sample_eigh(
            augmented, augmented_rhs, vectors, values, rank, relative_lambda
        )
        fixed_update = exact_update_from_sample_eigh(
            augmented, augmented_rhs, vectors, values, rank, fixed_lambda
        )
        sync((relative_update, fixed_update))
        projected_rhs = vectors[:, :rank].T @ augmented_rhs
        all_projected_rhs = vectors.T @ augmented_rhs
        force_coordinates = singular_values[:rank] * projected_rhs
        all_force_coordinates = singular_values * all_projected_rhs
        update_coefficients = force_coordinates / (
            values[:rank] + relative_lambda
        )
        all_update_coefficients = all_force_coordinates / (
            values + relative_lambda
        )
        row = {
            "rank": rank,
            "force_coordinate_capture": scalar(
                jnp.sum(jnp.square(force_coordinates))
                / jnp.maximum(jnp.sum(jnp.square(all_force_coordinates)), 1e-30)
            ),
            "update_coefficient_capture": scalar(
                jnp.sum(jnp.square(update_coefficients))
                / jnp.maximum(jnp.sum(jnp.square(all_update_coefficients)), 1e-30)
            ),
            "relative_damping_vs_full": vector_metrics(relative_update, exact_full_relative),
            "fixed_damping_vs_full": vector_metrics(fixed_update, exact_full_fixed),
            "relative_normal_residual": normal_residual(
                augmented, force, relative_update, relative_lambda
            ),
            "fixed_normal_residual": normal_residual(
                augmented, force, fixed_update, fixed_lambda
            ),
        }
        rows.append(row)
        if rank == 1600:
            exact_rank1600_relative = relative_update
            exact_rank1600_fixed = fixed_update

    # Current-batch MinSR with the SPRING paper's fixed lambda.
    current_gram = score.T @ score
    current_gram = 0.5 * (current_gram + current_gram.T)
    current_values_asc, current_vectors_asc = jnp.linalg.eigh(current_gram)
    current_order = jnp.argsort(current_values_asc)[::-1]
    current_values = jnp.maximum(current_values_asc[current_order], 0.0)
    current_vectors = current_vectors_asc[:, current_order]
    minsr_fixed = exact_update_from_sample_eigh(
        score, epsilon, current_vectors, current_values, score.shape[1], fixed_lambda
    )
    sync(minsr_fixed)

    # Subspace overlap without materializing exact full left singular vectors.
    exact_v1600 = vectors[:, :1600]
    exact_s1600 = singular_values[:1600]
    overlap = (approx_u.T @ augmented) @ exact_v1600
    overlap = overlap / jnp.maximum(exact_s1600[None, :], 1e-30)
    overlap_singular = jnp.linalg.svd(overlap, compute_uv=False)

    result = {
        "source_checkpoint": str(checkpoint),
        "stored_epoch": int(stored_epoch),
        "nchains": int(positions.shape[0]),
        "operator_shape": list(map(int, augmented.shape)),
        "eta": float(opt.eta),
        "configured_damping": float(opt.damping),
        "leading_singular_value": scalar(singular_values[0]),
        "relative_effective_lambda": scalar(relative_lambda),
        "spring_fixed_lambda": 0.001,
        "relative_lambda_over_fixed": scalar(relative_lambda / fixed_lambda),
        "approx_active_rank": int(jnp.sum(approx_retained)),
        "approx_decomposition_rank": int(approx_rank),
        "approx_seconds": approx_seconds,
        "exact_gram_eigh_seconds": exact_seconds,
        "ssi_vs_exact_rank1600": vector_metrics(approx_update, exact_rank1600_relative),
        "ssi_augmented_normal_residual": normal_residual(
            augmented, force, approx_update, relative_lambda
        ),
        "ssi_subspace_min_cosine": scalar(jnp.min(overlap_singular)),
        "ssi_subspace_max_principal_angle": scalar(
            jnp.arccos(jnp.clip(jnp.min(overlap_singular), -1.0, 1.0))
        ),
        "rank1600_relative_vs_full": vector_metrics(
            exact_rank1600_relative, exact_full_relative
        ),
        "rank1600_fixed_vs_full": vector_metrics(exact_rank1600_fixed, exact_full_fixed),
        "relative_full_vs_fixed_full": vector_metrics(
            exact_full_relative, exact_full_fixed
        ),
        "minsr_fixed_vs_wssr_full_relative": vector_metrics(
            minsr_fixed, exact_full_relative
        ),
        "minsr_fixed_vs_wssr_full_fixed": vector_metrics(minsr_fixed, exact_full_fixed),
        "current_batch_metrics": {
            "ssi": current_metrics(score, epsilon, current_force, approx_update),
            "exact_rank1600_relative": current_metrics(
                score, epsilon, current_force, exact_rank1600_relative
            ),
            "exact_full_relative": current_metrics(
                score, epsilon, current_force, exact_full_relative
            ),
            "exact_full_fixed": current_metrics(
                score, epsilon, current_force, exact_full_fixed
            ),
            "minsr_fixed": current_metrics(score, epsilon, current_force, minsr_fixed),
        },
        "rank_sweep": rows,
        "finite": bool(
            jax.device_get(
                jnp.all(jnp.isfinite(approx_update))
                & jnp.all(jnp.isfinite(exact_full_relative))
                & jnp.all(jnp.isfinite(exact_full_fixed))
                & jnp.all(jnp.isfinite(minsr_fixed))
            )
        ),
        "update_applied": False,
        "walkers_advanced": False,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
