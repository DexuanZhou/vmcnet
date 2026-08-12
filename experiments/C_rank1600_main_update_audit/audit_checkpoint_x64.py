#!/usr/bin/env python3
"""Read-only same-coordinate fp64 linear-algebra audit for C rank-1600 WSSR."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from vmcnet.mcmc import position_amplitude_core as pacore
from vmcnet.physics import core as physics_core
from vmcnet.train import runners
from vmcnet.updates import wssr
from vmcnet.utils import io


CHUNK_ROWS = 4096


def sync(value):
    return jax.block_until_ready(value)


def scalar(value):
    return float(jax.device_get(value))


def vector_metrics(value, reference):
    value_norm = jnp.linalg.norm(value)
    reference_norm = jnp.linalg.norm(reference)
    return {
        "relative_error": scalar(
            jnp.linalg.norm(value - reference) / jnp.maximum(reference_norm, 1e-300)
        ),
        "cosine": scalar(
            jnp.vdot(value, reference)
            / jnp.maximum(value_norm * reference_norm, 1e-300)
        ),
        "norm": scalar(value_norm),
        "reference_norm": scalar(reference_norm),
    }


def spectral_band_fractions(coordinates, retained):
    """Return squared update-coordinate mass in fixed one-indexed bands."""
    weighted = jnp.square(coordinates * retained.astype(coordinates.dtype))
    total = jnp.maximum(jnp.sum(weighted), 1e-300)
    result = {}
    for start, end in ((0, 400), (400, 800), (800, 1200), (1200, 1600), (1600, coordinates.shape[0])):
        if start >= coordinates.shape[0]:
            continue
        clipped_end = min(end, coordinates.shape[0])
        result[f"{start + 1}-{clipped_end}"] = scalar(
            jnp.sum(weighted[start:clipped_end]) / total
        )
    return result


def chunked_gram(matrix, chunk_rows=CHUNK_ROWS):
    """Accumulate matrix.T @ matrix in fp64 without a full fp64 matrix copy."""
    gram = jnp.zeros((matrix.shape[1], matrix.shape[1]), dtype=jnp.float64)
    for start in range(0, matrix.shape[0], chunk_rows):
        block = matrix[start : start + chunk_rows].astype(jnp.float64)
        if block.dtype != jnp.float64:
            raise TypeError(f"x64 block cast failed: {block.dtype}")
        gram = gram + block.T @ block
    return gram


def chunked_matvec(matrix, vector, chunk_rows=CHUNK_ROWS):
    """Compute matrix @ vector with fp64 block products."""
    output = jnp.zeros((matrix.shape[0],), dtype=jnp.float64)
    for start in range(0, matrix.shape[0], chunk_rows):
        end = min(start + chunk_rows, matrix.shape[0])
        block = matrix[start:end].astype(jnp.float64)
        output = output.at[start:end].set(block @ vector)
    return output


def chunked_rmatvec(matrix, vector, chunk_rows=CHUNK_ROWS):
    """Compute matrix.T @ vector with fp64 block products."""
    output = jnp.zeros((matrix.shape[1],), dtype=jnp.float64)
    for start in range(0, matrix.shape[0], chunk_rows):
        end = min(start + chunk_rows, matrix.shape[0])
        block = matrix[start:end].astype(jnp.float64)
        output = output + block.T @ vector[start:end]
    return output


def chunked_cross_gram(left, right, chunk_rows=CHUNK_ROWS):
    """Compute left.T @ right with fp64 block products."""
    output = jnp.zeros((left.shape[1], right.shape[1]), dtype=jnp.float64)
    for start in range(0, left.shape[0], chunk_rows):
        end = min(start + chunk_rows, left.shape[0])
        left_block = left[start:end].astype(jnp.float64)
        right_block = right[start:end].astype(jnp.float64)
        output = output + left_block.T @ right_block
    return output


def exact_update(matrix, rhs, vectors, values, rank, cutoff, lambda_reg):
    selected_vectors = vectors[:, :rank]
    selected_values = values[:rank]
    singular_values = jnp.sqrt(selected_values)
    retained = singular_values / jnp.sqrt(values[0]) > cutoff
    coefficients = (
        selected_vectors.T @ rhs
        * retained.astype(rhs.dtype)
        / (selected_values + lambda_reg)
    )
    return chunked_matvec(matrix, selected_vectors @ coefficients), retained


def normal_residual(matrix, force, update, lambda_reg):
    dual = chunked_rmatvec(matrix, update)
    residual = chunked_matvec(matrix, dual) + lambda_reg * update - force
    return scalar(jnp.linalg.norm(residual) / jnp.maximum(jnp.linalg.norm(force), 1e-300))


def current_residual(score, residuals, update):
    return scalar(
        jnp.linalg.norm(chunked_rmatvec(score, update) - residuals)
        / jnp.maximum(jnp.linalg.norm(residuals), 1e-300)
    )


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
    expected_stored_epoch = args.epoch - 1
    if int(stored_epoch) != expected_stored_epoch:
        raise ValueError(
            f"stored epoch {stored_epoch} != expected {expected_stored_epoch}"
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
    _, svd_key = jax.random.split(key)
    sync((energy, local_energies, score, augmented, force))

    # Reproduce the actual late-checkpoint warm-SSI decomposition in fp32.
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

    # ``runners._get_dtype`` deliberately follows the training config and turns
    # x64 off for this fp32 checkpoint. Restore x64 only after constructing the
    # same-coordinate fp32 linear system, then audit its linear algebra in fp64.
    jax.config.update("jax_enable_x64", True)
    if not jax.config.x64_enabled:
        raise RuntimeError("failed to enable JAX x64 for the audit linear algebra")

    # Keep the large same-coordinate operator in fp32. Each row block is cast
    # transiently inside the fp64 Gram/matvec helpers, avoiding a 13 GiB full
    # copy plus another full-size transpose buffer.
    rhs64 = augmented_rhs.astype(jnp.float64)
    epsilon64 = epsilon.astype(jnp.float64)
    force64 = chunked_matvec(augmented, rhs64)
    sync((rhs64, epsilon64, force64))
    if rhs64.dtype != jnp.float64 or force64.dtype != jnp.float64:
        raise TypeError(
            f"x64 audit cast failed: rhs={rhs64.dtype}, force={force64.dtype}"
        )

    started = time.perf_counter()
    gram64 = chunked_gram(augmented)
    gram64 = 0.5 * (gram64 + gram64.T)
    raw_values_ascending, vectors_ascending = jnp.linalg.eigh(gram64)
    order = jnp.argsort(raw_values_ascending)[::-1]
    raw_values = raw_values_ascending[order]
    values = jnp.maximum(raw_values, 0.0)
    vectors = vectors_ascending[:, order]
    sync((values, vectors))
    gram_eigh_seconds = time.perf_counter() - started

    singular_values = jnp.sqrt(values)
    leading = singular_values[0]
    legacy_cutoff = jnp.asarray(opt.damping, dtype=jnp.float64)
    legacy_lambda = jnp.square(legacy_cutoff * leading)
    fixed_lambda = jnp.asarray(0.001, dtype=jnp.float64)
    full_rank = int(augmented.shape[1])

    exact_full_legacy, full_retained = exact_update(
        augmented,
        rhs64,
        vectors,
        values,
        full_rank,
        legacy_cutoff,
        legacy_lambda,
    )
    exact_full_fixed, _ = exact_update(
        augmented,
        rhs64,
        vectors,
        values,
        full_rank,
        legacy_cutoff,
        fixed_lambda,
    )
    exact_rank1600_legacy, retained1600 = exact_update(
        augmented,
        rhs64,
        vectors,
        values,
        1600,
        legacy_cutoff,
        legacy_lambda,
    )
    exact_rank1600_fixed, _ = exact_update(
        augmented,
        rhs64,
        vectors,
        values,
        1600,
        legacy_cutoff,
        fixed_lambda,
    )

    approx_u64 = approx_u.astype(jnp.float64)
    approx_s64 = approx_s.astype(jnp.float64)
    approx_force64 = force.astype(jnp.float64)
    approx_retained = approx_s64 / jnp.maximum(approx_s64[0], 1e-300) > legacy_cutoff
    approx_projected = approx_u64.T @ approx_force64
    approx_legacy = approx_u64 @ (
        approx_projected
        * approx_retained.astype(jnp.float64)
        / (jnp.square(approx_s64) + legacy_lambda)
    )
    approx_fixed = approx_u64 @ (
        approx_projected
        * approx_retained.astype(jnp.float64)
        / (jnp.square(approx_s64) + fixed_lambda)
    )
    sync(
        (
            exact_full_legacy,
            exact_full_fixed,
            exact_rank1600_legacy,
            exact_rank1600_fixed,
            approx_legacy,
            approx_fixed,
        )
    )

    all_rhs_coordinates = vectors.T @ rhs64
    all_force_coordinates = singular_values * all_rhs_coordinates
    all_force_retained = all_force_coordinates * full_retained.astype(jnp.float64)
    all_legacy_update_coordinates = all_force_retained / (values + legacy_lambda)
    all_fixed_update_coordinates = all_force_retained / (values + fixed_lambda)
    current_force64 = chunked_matvec(score, epsilon64)
    history_force64 = force64 - current_force64
    rows = []
    for rank in (400, 800, 1200, 1600, 2000, full_rank):
        update_legacy, retained = exact_update(
            augmented,
            rhs64,
            vectors,
            values,
            rank,
            legacy_cutoff,
            legacy_lambda,
        )
        update_fixed, _ = exact_update(
            augmented,
            rhs64,
            vectors,
            values,
            rank,
            legacy_cutoff,
            fixed_lambda,
        )
        force_coordinates = all_force_coordinates[:rank] * retained.astype(jnp.float64)
        update_coordinates = force_coordinates / (values[:rank] + legacy_lambda)
        all_update_coordinates = all_legacy_update_coordinates
        rows.append(
            {
                "rank": rank,
                "retained_rank": int(jax.device_get(jnp.sum(retained))),
                "singular_value_at_rank": scalar(singular_values[rank - 1]),
                "relative_singular_value_at_rank": scalar(
                    singular_values[rank - 1] / leading
                ),
                "force_coordinate_capture": scalar(
                    jnp.sum(jnp.square(force_coordinates))
                    / jnp.maximum(jnp.sum(jnp.square(all_force_retained)), 1e-300)
                ),
                "update_coefficient_capture": scalar(
                    jnp.sum(jnp.square(update_coordinates))
                    / jnp.maximum(jnp.sum(jnp.square(all_update_coordinates)), 1e-300)
                ),
                "legacy_vs_full": vector_metrics(update_legacy, exact_full_legacy),
                "fixed_vs_full": vector_metrics(update_fixed, exact_full_fixed),
                "legacy_normal_residual": normal_residual(
                    augmented, force64, update_legacy, legacy_lambda
                ),
                "fixed_normal_residual": normal_residual(
                    augmented, force64, update_fixed, fixed_lambda
                ),
            }
        )

    exact_v1600 = vectors[:, :1600]
    exact_s1600 = singular_values[:1600]
    overlap = chunked_cross_gram(approx_u, augmented) @ exact_v1600
    overlap = overlap / jnp.maximum(exact_s1600[None, :], 1e-300)
    overlap_singular = jnp.linalg.svd(overlap, compute_uv=False)

    result = {
        "source_checkpoint": str(checkpoint),
        "stored_epoch": int(stored_epoch),
        "nchains": int(positions.shape[0]),
        "operator_shape": list(map(int, augmented.shape)),
        "model_linear_system_dtype": str(augmented.dtype),
        "audit_linear_algebra_dtype": "float64",
        "audit_chunk_rows": CHUNK_ROWS,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "eta": float(opt.eta),
        "legacy_relative_cutoff": scalar(legacy_cutoff),
        "legacy_effective_lambda": scalar(legacy_lambda),
        "fixed_tikhonov_lambda": scalar(fixed_lambda),
        "leading_singular_value": scalar(leading),
        "full_retained_rank": int(jax.device_get(jnp.sum(full_retained))),
        "rank1600_retained_rank": int(jax.device_get(jnp.sum(retained1600))),
        "raw_min_gram_eigenvalue": scalar(jnp.min(raw_values)),
        "negative_gram_eigenvalues": int(jax.device_get(jnp.sum(raw_values < 0.0))),
        "gram_eigh_seconds": gram_eigh_seconds,
        "approx_decomposition_rank": int(approx_rank),
        "approx_active_rank": int(jax.device_get(jnp.sum(approx_retained))),
        "ssi_legacy_vs_x64_rank1600": vector_metrics(
            approx_legacy, exact_rank1600_legacy
        ),
        "ssi_fixed_vs_x64_rank1600": vector_metrics(
            approx_fixed, exact_rank1600_fixed
        ),
        "ssi_fixed_vs_legacy": vector_metrics(approx_fixed, approx_legacy),
        "x64_rank1600_fixed_vs_legacy": vector_metrics(
            exact_rank1600_fixed, exact_rank1600_legacy
        ),
        "x64_full_fixed_vs_legacy": vector_metrics(
            exact_full_fixed, exact_full_legacy
        ),
        "spectral_band_update_mass": {
            "legacy": spectral_band_fractions(
                all_legacy_update_coordinates, full_retained
            ),
            "fixed_lambda_1e-3": spectral_band_fractions(
                all_fixed_update_coordinates, full_retained
            ),
        },
        "force_decomposition": {
            "current_force_norm": scalar(jnp.linalg.norm(current_force64)),
            "history_force_norm": scalar(jnp.linalg.norm(history_force64)),
            "augmented_force_norm": scalar(jnp.linalg.norm(force64)),
            "current_history_cosine": scalar(
                jnp.vdot(current_force64, history_force64)
                / jnp.maximum(
                    jnp.linalg.norm(current_force64)
                    * jnp.linalg.norm(history_force64),
                    1e-300,
                )
            ),
        },
        "force_dot_update": {
            "ssi_legacy": scalar(jnp.vdot(force64, approx_legacy)),
            "ssi_fixed_lambda_1e-3": scalar(jnp.vdot(force64, approx_fixed)),
            "x64_rank1600_legacy": scalar(
                jnp.vdot(force64, exact_rank1600_legacy)
            ),
            "x64_rank1600_fixed_lambda_1e-3": scalar(
                jnp.vdot(force64, exact_rank1600_fixed)
            ),
        },
        "ssi_x64_subspace_min_cosine": scalar(jnp.min(overlap_singular)),
        "ssi_legacy_current_residual": current_residual(
            score, epsilon64, approx_legacy
        ),
        "ssi_fixed_current_residual": current_residual(
            score, epsilon64, approx_fixed
        ),
        "x64_rank1600_legacy_current_residual": current_residual(
            score, epsilon64, exact_rank1600_legacy
        ),
        "x64_rank1600_fixed_current_residual": current_residual(
            score, epsilon64, exact_rank1600_fixed
        ),
        "x64_full_legacy_normal_residual": normal_residual(
            augmented, force64, exact_full_legacy, legacy_lambda
        ),
        "x64_full_fixed_normal_residual": normal_residual(
            augmented, force64, exact_full_fixed, fixed_lambda
        ),
        "rank_sweep": rows,
        "finite": bool(
            jax.device_get(
                jnp.all(jnp.isfinite(exact_full_legacy))
                & jnp.all(jnp.isfinite(exact_full_fixed))
                & jnp.all(jnp.isfinite(approx_legacy))
                & jnp.all(jnp.isfinite(approx_fixed))
            )
        ),
        "update_applied": False,
        "walkers_advanced": False,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
