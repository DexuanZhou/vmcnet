#!/usr/bin/env python3
"""Read-only causal audit of C rank-1600 WSSR direction instability.

The checkpoint parameters and optimizer state are held fixed.  Only the walkers
are advanced between statistically adjacent batches.  On every batch we compare
the production two-step warm SSI solve, a four-step warm SSI solve, and an exact
sample-space rank-1600 eigensolve of the same augmented operator.
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


NCHAINS = 1000
BATCHES = 30
RANK = 1600
TOP_MODES = 32
RITZ_MODES = 16
EPS = 1.0e-12
BLOCK_LENGTH = 5
BOOTSTRAP_REPLICATES = 4000
BOOTSTRAP_SEED = 20260802
PROXIMAL_GAMMA_MULTIPLIERS = (0.1, 0.3, 1.0, 3.0)


def scalar(value) -> float:
    return float(jax.device_get(value))


def safe_cosine(left, right):
    return jnp.vdot(left, right) / jnp.maximum(
        jnp.linalg.norm(left) * jnp.linalg.norm(right), EPS
    )


def safe_angle_from_cosine(cosine):
    return jnp.arccos(jnp.clip(cosine, -1.0, 1.0))


def circular_indices(length, rng):
    count = math.ceil(length / BLOCK_LENGTH)
    starts = rng.integers(0, length, size=count)
    offsets = np.arange(BLOCK_LENGTH)
    return ((starts[:, None] + offsets[None, :]) % length).reshape(-1)[:length]


def bootstrap_median_ci(values, seed_offset=0):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    statistics = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = values[circular_indices(len(values), rng)]
        statistics.append(float(np.median(sample)))
    return [float(x) for x in np.quantile(statistics, [0.025, 0.975])]


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skip-warm4",
        action="store_true",
        help=(
            "Do not recompute the already-validated warm4 control. Warm4-valued "
            "record fields are native placeholders and must not be interpreted."
        ),
    )
    parser.add_argument(
        "--test-proximal",
        action="store_true",
        help=(
            "Run the pre-registered basis-invariant temporal-proximal coefficient "
            "grid and matched memoryless extra-damping controls."
        ),
    )
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
        "nchains": (int(config.vmc.nchains), NCHAINS),
        "rank": (int(opt.sr_rank), RANK),
        "rank_max": (int(opt.sr_rank_max), RANK),
        "working_rank": (int(opt.svd_working_rank), RANK),
        "eta": (float(opt.eta), 0.3),
        "learning_rate": (float(opt.learning_rate), 0.04),
        "spectral_regularization": (str(opt.spectral_regularization), "tikhonov"),
        "tikhonov_lambda": (float(opt.tikhonov_lambda), 1.0e-3),
        "relative_cutoff": (float(opt.relative_singular_value_cutoff), 3.0e-4),
        "warm_iterations": (int(opt.svd_maxiter_warm), 2),
        "complement_weight": (float(opt.complement_weight), 0.0),
    }
    mismatches = {
        name: pair for name, pair in expected.items() if pair[0] != pair[1]
    }
    if mismatches:
        raise ValueError(f"unexpected source configuration: {mismatches}")

    dtype = runners._get_dtype(config)
    # Keep the scientific path in the checkpoint's configured fp32 precision.
    # x64 is enabled only around the two diagnostic JITs below; leaving it on
    # globally changes weak scalar types inside the dynamic-width MCMC kernel.
    jax.config.update("jax_enable_x64", False)
    ions, charges, nelec = runners._get_electron_ion_config_as_arrays(config, dtype)
    positions = pacore.get_position_from_data(data)
    if tuple(map(int, nelec)) != (4, 2):
        raise ValueError(f"expected C spin sector (4,2), got {nelec}")
    if positions.shape[0] != NCHAINS:
        raise ValueError(positions.shape)
    if not bool(jax.device_get(jnp.all(jnp.isfinite(positions)))):
        raise ValueError("non-finite checkpoint walkers")

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
            NCHAINS,
            runners._get_clipping_fn(config.vmc),
            config.vmc.nan_safe,
            record_raw_local_energies=True,
        )
    )
    _, walker_fn = runners._get_mcmc_fns(config.vmc, log_psi, apply_pmap=False)
    state = optimizer_state.core_state
    cutoff = float(opt.relative_singular_value_cutoff)
    tikhonov_lambda = float(opt.tikhonov_lambda)

    def native_solve(score, epsilon, direction_key, state_arg, warm_iterations):
        augmented, rhs = wssr.augment_wssr_system(
            score, epsilon, state_arg, float(opt.eta)
        )
        u, singular_values, vh, _ = wssr._wssr_right_svd_decomposition(
            augmented,
            state_arg,
            direction_key,
            RANK,
            int(opt.svd_maxiter_initial),
            warm_iterations,
            bool(opt.exact_first),
            EPS,
            exact_first_force=False,
        )
        result = wssr._wssr_update_from_svd(
            augmented,
            rhs,
            state_arg,
            u,
            singular_values,
            vh,
            float(opt.damping),
            float(opt.norm_constraint),
            RANK,
            float(opt.sr_scale),
            False,
            rank_update_max=RANK,
            spectral_regularization="tikhonov",
            complement_weight=0.0,
            relative_singular_value_cutoff=cutoff,
            tikhonov_lambda=tikhonov_lambda,
            eps=EPS,
        )
        return result.grad_like_update, u, singular_values, vh, augmented, rhs

    native2_jit = jax.jit(
        lambda score, epsilon, k, state_arg: native_solve(
            score, epsilon, k, state_arg, 2
        )
    )

    def native4_direction(score, epsilon, direction_key, state_arg):
        return native_solve(score, epsilon, direction_key, state_arg, 4)[0]

    native4_jit = jax.jit(native4_direction)

    @jax.jit
    def exact_rank_solve(augmented, rhs):
        gram = augmented.T @ augmented
        gram = 0.5 * (gram + gram.T)
        values_ascending, vectors_ascending = jnp.linalg.eigh(gram)
        order = jnp.argsort(values_ascending)[::-1]
        values_all = jnp.maximum(values_ascending[order], 0.0)
        vectors_all = vectors_ascending[:, order]
        values = values_all[:RANK]
        singular_values = jnp.sqrt(values)
        vectors = vectors_all[:, :RANK]
        leading = jnp.maximum(singular_values[0], EPS)
        active = singular_values / leading > cutoff
        rhs_coordinates = vectors.T @ rhs
        dual_coefficients = (
            rhs_coordinates / (values + tikhonov_lambda)
        ) * active.astype(rhs.dtype)
        update = augmented @ (vectors @ dual_coefficients)
        left_coefficients = singular_values * dual_coefficients
        top_indices = jnp.argsort(jnp.abs(left_coefficients))[::-1][:TOP_MODES]
        top_singular = singular_values[top_indices]
        top_basis = (augmented @ vectors[:, top_indices]) / jnp.maximum(
            top_singular[None, :], EPS
        )
        top_basis, _ = jnp.linalg.qr(top_basis, mode="reduced")
        # In exact arithmetic this matrix is diagonal with entries s**2.  In
        # fp32, however, the near-cutoff eigenvectors are not orthogonal enough
        # for diag(1/s**2) to be a trustworthy projector.  Retain the actual
        # basis Gram and solve it in fp64 when decomposing the next transition.
        vectors64 = vectors.astype(jnp.float64)
        gram64 = gram.astype(jnp.float64)
        basis_gram64 = vectors64.T @ (gram64 @ vectors64)
        basis_gram64 = 0.5 * (basis_gram64 + basis_gram64.T)
        next_values = values_all[jnp.minimum(top_indices + 1, values_all.shape[0] - 1)]
        next_singular = jnp.sqrt(next_values)
        selected_relative_gaps = (
            singular_values[top_indices] - next_singular
        ) / jnp.maximum(singular_values[top_indices], EPS)
        return (
            update,
            singular_values,
            vectors,
            left_coefficients,
            top_basis,
            basis_gram64,
            selected_relative_gaps,
            jnp.sum(active.astype(jnp.int32)),
        )

    @jax.jit
    def proximal_candidate_solve(
        augmented, rhs, singular_values, vectors, previous_proximal
    ):
        """Solve temporal-proximal and matched memoryless controls.

        In the current retained left-singular basis U, the temporal candidate is

          c_i = (s_i r_i + gamma b_i) / (s_i^2 + lambda + gamma),

        where b_i = <u_i, d_previous>.  The matched control sets b_i=0 and
        therefore has exactly the same effective damping lambda+gamma.  A
        common gamma inside the retained space makes the construction invariant
        to rotations within an exactly degenerate singular-value cluster.
        """
        value_dtype = augmented.dtype
        gammas = jnp.asarray(PROXIMAL_GAMMA_MULTIPLIERS, dtype=value_dtype) * jnp.asarray(
            tikhonov_lambda, dtype=value_dtype
        )
        values = jnp.square(singular_values)
        leading = jnp.maximum(singular_values[0], jnp.asarray(EPS, value_dtype))
        active = singular_values / leading > cutoff
        rhs_coordinates = vectors.T @ rhs
        previous_coordinates = vectors.T @ (augmented.T @ previous_proximal)
        previous_left = previous_coordinates / jnp.maximum(
            singular_values[:, None], jnp.asarray(EPS, value_dtype)
        )
        denominators = values[:, None] + jnp.asarray(
            tikhonov_lambda, dtype=value_dtype
        ) + gammas[None, :]
        data_numerator = singular_values[:, None] * rhs_coordinates[:, None]
        proximal_left = (
            data_numerator + gammas[None, :] * previous_left
        ) / denominators
        control_left = data_numerator / denominators
        active_scale = active.astype(value_dtype)[:, None] / jnp.maximum(
            singular_values[:, None], jnp.asarray(EPS, value_dtype)
        )
        proximal = augmented @ (vectors @ (proximal_left * active_scale))
        control = augmented @ (vectors @ (control_left * active_scale))
        return proximal, control

    @jax.jit
    def candidate_consecutive_cosines(previous, current):
        dots = jnp.sum(previous * current, axis=0)
        norms = jnp.linalg.norm(previous, axis=0) * jnp.linalg.norm(current, axis=0)
        return dots / jnp.maximum(norms, EPS)

    @jax.jit
    def candidate_heldout_metrics(
        score, epsilon, native_reference, proximal, control
    ):
        candidates = jnp.concatenate((proximal, control), axis=1)
        reference_norm = jnp.linalg.norm(native_reference)
        candidates = candidates * (
            reference_norm / jnp.maximum(jnp.linalg.norm(candidates, axis=0), EPS)
        )[None, :]
        predictions = score.T @ candidates
        force = score @ epsilon
        force_norm = jnp.linalg.norm(force)
        candidate_norms = jnp.linalg.norm(candidates, axis=0)
        alignments = (force @ candidates) / jnp.maximum(
            force_norm * candidate_norms, EPS
        )
        q_values = jnp.linalg.norm(epsilon[:, None] - predictions, axis=0) / jnp.maximum(
            jnp.linalg.norm(epsilon), EPS
        )
        force_dots = force @ candidates
        return alignments, q_values, force_dots

    @jax.jit
    def batch_metrics(
        augmented,
        native_update,
        native_u,
        native_s,
        native_vh,
        warm4_update,
        exact_update,
        exact_s,
        exact_v,
        exact_left_coefficients,
        exact_top_basis,
        exact_basis_gram64,
        exact_selected_gaps,
        previous_native,
        previous_warm4,
        previous_exact,
        previous_native_top_basis,
        previous_exact_top_basis,
    ):
        native_active = native_s / jnp.maximum(native_s[0], EPS) > cutoff
        native_coefficients = native_u.T @ native_update
        native_top_indices = jnp.argsort(jnp.abs(native_coefficients))[::-1][
            :TOP_MODES
        ]
        native_top_basis = native_u[:, native_top_indices]
        native_top_basis, _ = jnp.linalg.qr(native_top_basis, mode="reduced")
        ritz_indices = native_top_indices[:RITZ_MODES]
        ritz_actual = augmented.T @ native_u[:, ritz_indices]
        ritz_model = (
            native_vh[ritz_indices, :].T * native_s[ritz_indices][None, :]
        )
        ritz_residuals = jnp.linalg.norm(ritz_actual - ritz_model, axis=0) / jnp.maximum(
            jnp.linalg.norm(ritz_actual, axis=0), EPS
        )

        native_projection = native_u @ (
            (native_u.T @ previous_native) * native_active.astype(native_u.dtype)
        )
        exact_active = exact_s / jnp.maximum(exact_s[0], EPS) > cutoff
        exact_at_previous = augmented.T @ previous_exact
        exact_projection_rhs64 = (exact_v.T @ exact_at_previous).astype(jnp.float64)
        projector_ridge64 = jnp.maximum(
            jnp.max(jnp.diag(exact_basis_gram64)) * 1.0e-12,
            jnp.asarray(1.0e-18, dtype=jnp.float64),
        )
        exact_projection_coefficients64 = jnp.linalg.solve(
            exact_basis_gram64
            + projector_ridge64 * jnp.eye(RANK, dtype=jnp.float64),
            exact_projection_rhs64,
        )
        exact_projection = augmented @ (
            exact_v @ exact_projection_coefficients64.astype(exact_v.dtype)
        )

        native_subspace_cosine = jnp.linalg.norm(native_projection) / jnp.maximum(
            jnp.linalg.norm(previous_native), EPS
        )
        exact_projected_norm_sq64 = jnp.maximum(
            jnp.vdot(exact_projection_rhs64, exact_projection_coefficients64),
            0.0,
        )
        exact_subspace_cosine = (
            jnp.sqrt(exact_projected_norm_sq64)
            / jnp.maximum(
                jnp.linalg.norm(previous_exact).astype(jnp.float64),
                jnp.asarray(EPS, dtype=jnp.float64),
            )
        ).astype(exact_s.dtype)
        native_subspace_angle = safe_angle_from_cosine(native_subspace_cosine)
        exact_subspace_angle = safe_angle_from_cosine(exact_subspace_cosine)
        native_coefficient_angle = safe_angle_from_cosine(
            safe_cosine(native_projection, native_update)
        )
        exact_coefficient_angle = safe_angle_from_cosine(
            safe_cosine(exact_projection, exact_update)
        )

        native_overlap_sv = jnp.linalg.svd(
            previous_native_top_basis.T @ native_top_basis,
            compute_uv=False,
        )
        exact_overlap_sv = jnp.linalg.svd(
            previous_exact_top_basis.T @ exact_top_basis,
            compute_uv=False,
        )
        native_overlap_angles = safe_angle_from_cosine(native_overlap_sv)
        exact_overlap_angles = safe_angle_from_cosine(exact_overlap_sv)

        native_total_angle = safe_angle_from_cosine(
            safe_cosine(previous_native, native_update)
        )
        exact_total_angle = safe_angle_from_cosine(
            safe_cosine(previous_exact, exact_update)
        )
        native_attribution = native_subspace_angle / jnp.maximum(
            native_subspace_angle + native_coefficient_angle, EPS
        )
        exact_attribution = exact_subspace_angle / jnp.maximum(
            exact_subspace_angle + exact_coefficient_angle, EPS
        )

        return {
            "native_top_basis": native_top_basis,
            "native_active_rank": jnp.sum(native_active.astype(jnp.int32)),
            "native_vs_exact_cosine": safe_cosine(native_update, exact_update),
            "warm4_vs_exact_cosine": safe_cosine(warm4_update, exact_update),
            "warm2_vs_warm4_cosine": safe_cosine(native_update, warm4_update),
            "native_vs_exact_relative_error": jnp.linalg.norm(
                native_update - exact_update
            ) / jnp.maximum(jnp.linalg.norm(exact_update), EPS),
            "warm4_vs_exact_relative_error": jnp.linalg.norm(
                warm4_update - exact_update
            ) / jnp.maximum(jnp.linalg.norm(exact_update), EPS),
            "warm4_norm_ratio": jnp.linalg.norm(warm4_update)
            / jnp.maximum(jnp.linalg.norm(native_update), EPS),
            "exact_norm_ratio": jnp.linalg.norm(exact_update)
            / jnp.maximum(jnp.linalg.norm(native_update), EPS),
            "native_consecutive_cosine": safe_cosine(previous_native, native_update),
            "warm4_consecutive_cosine": safe_cosine(previous_warm4, warm4_update),
            "exact_consecutive_cosine": safe_cosine(previous_exact, exact_update),
            "native_total_angle": native_total_angle,
            "native_subspace_angle": native_subspace_angle,
            "native_coefficient_angle": native_coefficient_angle,
            "native_subspace_attribution": native_attribution,
            "exact_total_angle": exact_total_angle,
            "exact_subspace_angle": exact_subspace_angle,
            "exact_coefficient_angle": exact_coefficient_angle,
            "exact_subspace_attribution": exact_attribution,
            "native_top32_min_overlap_cosine": jnp.min(native_overlap_sv),
            "native_top32_median_principal_angle": jnp.median(native_overlap_angles),
            "native_top32_max_principal_angle": jnp.max(native_overlap_angles),
            "exact_top32_min_overlap_cosine": jnp.min(exact_overlap_sv),
            "exact_top32_median_principal_angle": jnp.median(exact_overlap_angles),
            "exact_top32_max_principal_angle": jnp.max(exact_overlap_angles),
            "native_top16_ritz_residual_median": jnp.median(ritz_residuals),
            "native_top16_ritz_residual_max": jnp.max(ritz_residuals),
            "exact_top32_relative_gap_median": jnp.median(exact_selected_gaps),
            "exact_top32_relative_gap_min": jnp.min(exact_selected_gaps),
            "exact_top32_gap_fraction_lt_1e3": jnp.mean(
                (exact_selected_gaps < 1.0e-3).astype(jnp.float32)
            ),
            "exact_top32_gap_fraction_lt_1e2": jnp.mean(
                (exact_selected_gaps < 1.0e-2).astype(jnp.float32)
            ),
            "exact_top32_coefficient_capture": jnp.sum(
                jnp.square(
                    exact_left_coefficients[
                        jnp.argsort(jnp.abs(exact_left_coefficients))[::-1][:TOP_MODES]
                    ]
                )
            ) / jnp.maximum(jnp.sum(jnp.square(exact_left_coefficients)), EPS),
            "exact_projector_ridge": projector_ridge64,
        }

    @jax.jit
    def heldout_metrics(score, epsilon, native, warm4, exact):
        native_norm = jnp.linalg.norm(native)
        warm4 = warm4 * native_norm / jnp.maximum(jnp.linalg.norm(warm4), EPS)
        exact = exact * native_norm / jnp.maximum(jnp.linalg.norm(exact), EPS)
        candidates = jnp.stack((native, warm4, exact), axis=1)
        predictions = score.T @ candidates
        force = score @ epsilon
        force_norm = jnp.linalg.norm(force)
        candidate_norms = jnp.linalg.norm(candidates, axis=0)
        alignments = (force @ candidates) / jnp.maximum(
            force_norm * candidate_norms, EPS
        )
        q_values = jnp.linalg.norm(epsilon[:, None] - predictions, axis=0) / jnp.maximum(
            jnp.linalg.norm(epsilon), EPS
        )
        force_dots = force @ candidates
        return alignments, q_values, force_dots

    def sample_batch(data_arg, key_arg):
        acceptance, data_arg, key_arg = walker_fn(params, data_arg, key_arg)
        positions_arg = pacore.get_position_from_data(data_arg)
        energy, clipped, stats = energy_fn(params, positions_arg)
        score, _ = wssr.center_and_scale_score_matrix(log_psi, params, positions_arg)
        epsilon = wssr.center_and_scale_energy_residuals(clipped, energy)
        return data_arg, key_arg, acceptance, energy, score, epsilon, stats

    flat_params, _ = jax.flatten_util.ravel_pytree(params)
    zero_direction = jnp.zeros_like(flat_params)
    zero_top_basis = jnp.zeros(
        (flat_params.shape[0], TOP_MODES), dtype=flat_params.dtype
    )
    previous_native = zero_direction
    previous_warm4 = zero_direction
    previous_exact = zero_direction
    previous_native_top_basis = zero_top_basis
    previous_exact_top_basis = zero_top_basis
    gamma_values = [
        float(multiplier * tikhonov_lambda)
        for multiplier in PROXIMAL_GAMMA_MULTIPLIERS
    ]
    previous_proximal = jnp.zeros(
        (flat_params.shape[0], len(gamma_values)), dtype=flat_params.dtype
    )
    previous_damping_control = jnp.zeros_like(previous_proximal)

    records = []
    candidate_records = []
    native_seconds = []
    warm4_seconds = []
    exact_seconds = []
    candidate_seconds = []
    started = time.perf_counter()
    for batch in range(BATCHES):
        # Sampling, local energies, score construction, and both production
        # WSSR directions must compile and execute with the normal fp32 policy.
        jax.config.update("jax_enable_x64", False)
        data, key, acceptance, energy, score, epsilon, stats = sample_batch(data, key)

        heldout = None
        candidate_heldout = None
        if batch > 0:
            alignments, q_values, force_dots = heldout_metrics(
                score,
                epsilon,
                previous_native,
                previous_warm4,
                previous_exact,
            )
            alignments, q_values, force_dots = jax.device_get(
                (alignments, q_values, force_dots)
            )
            heldout = {
                name: {
                    "alignment": float(alignments[index]),
                    "q": float(q_values[index]),
                    "force_dot": float(force_dots[index]),
                }
                for index, name in enumerate(("native_warm2", "warm4", "exact_rank1600"))
            }
            if args.test_proximal:
                candidate_metrics = candidate_heldout_metrics(
                    score,
                    epsilon,
                    previous_native,
                    previous_proximal,
                    previous_damping_control,
                )
                candidate_metrics = jax.device_get(candidate_metrics)
                candidate_heldout = {
                    "proximal": {
                        str(gamma): {
                            "alignment": float(candidate_metrics[0][index]),
                            "q": float(candidate_metrics[1][index]),
                            "force_dot": float(candidate_metrics[2][index]),
                        }
                        for index, gamma in enumerate(gamma_values)
                    },
                    "damping_control": {
                        str(gamma): {
                            "alignment": float(
                                candidate_metrics[0][index + len(gamma_values)]
                            ),
                            "q": float(candidate_metrics[1][index + len(gamma_values)]),
                            "force_dot": float(
                                candidate_metrics[2][index + len(gamma_values)]
                            ),
                        }
                        for index, gamma in enumerate(gamma_values)
                    },
                }

        key, direction_key = jax.random.split(key)
        if not args.skip_warm4:
            # Run the direction-only warm4 solve first so its large temporary basis
            # can be released before the warm2 factors needed by the causal
            # decomposition are materialized.
            begin = time.perf_counter()
            warm4_update = native4_jit(score, epsilon, direction_key, state)
            jax.block_until_ready(warm4_update)
            warm4_seconds.append(time.perf_counter() - begin)

        begin = time.perf_counter()
        native_update, native_u, native_s, native_vh, augmented, rhs = native2_jit(
            score, epsilon, direction_key, state
        )
        jax.block_until_ready(native_update)
        native_seconds.append(time.perf_counter() - begin)
        if args.skip_warm4:
            # Explicit placeholder solely to keep the fixed-shape metrics interface.
            # The payload marks every warm4-valued field as non-interpretable.
            warm4_update = native_update
            warm4_seconds.append(float("nan"))

        # Only the small basis-Gram construction/projector solve is fp64.  The
        # exact direction itself still consumes fp32 inputs and remains fp32.
        jax.config.update("jax_enable_x64", True)
        if not jax.config.x64_enabled:
            raise RuntimeError("fp64 diagnostic projector was not enabled")
        begin = time.perf_counter()
        (
            exact_update,
            exact_s,
            exact_v,
            exact_left_coefficients,
            exact_top_basis,
            exact_basis_gram64,
            exact_selected_gaps,
            exact_active_rank,
        ) = exact_rank_solve(augmented, rhs)
        jax.block_until_ready(exact_update)
        exact_seconds.append(time.perf_counter() - begin)

        current_proximal = previous_proximal
        current_damping_control = previous_damping_control
        proximal_cosines = jnp.zeros((len(gamma_values),), dtype=flat_params.dtype)
        damping_control_cosines = jnp.zeros_like(proximal_cosines)
        if args.test_proximal:
            begin = time.perf_counter()
            current_proximal, current_damping_control = proximal_candidate_solve(
                augmented,
                rhs,
                exact_s,
                exact_v,
                previous_proximal,
            )
            jax.block_until_ready(current_proximal)
            candidate_seconds.append(time.perf_counter() - begin)
            proximal_cosines = candidate_consecutive_cosines(
                previous_proximal, current_proximal
            )
            damping_control_cosines = candidate_consecutive_cosines(
                previous_damping_control, current_damping_control
            )

        metrics = batch_metrics(
            augmented,
            native_update,
            native_u,
            native_s,
            native_vh,
            warm4_update,
            exact_update,
            exact_s,
            exact_v,
            exact_left_coefficients,
            exact_top_basis,
            exact_basis_gram64,
            exact_selected_gaps,
            previous_native,
            previous_warm4,
            previous_exact,
            previous_native_top_basis,
            previous_exact_top_basis,
        )
        metrics = jax.device_get(metrics)
        jax.config.update("jax_enable_x64", False)
        current_native_top_basis = metrics.pop("native_top_basis")
        record = {
            "batch": batch,
            "acceptance": scalar(acceptance),
            "energy_internal_only": scalar(energy),
            "raw_variance": scalar(jnp.var(stats["local_energies_noclip"], ddof=1)),
            "exact_active_rank": int(jax.device_get(exact_active_rank)),
            "heldout_next_batch_for_previous_directions": heldout,
            **{name: float(value) for name, value in metrics.items()},
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

        if args.test_proximal:
            proximal_cosines, damping_control_cosines = jax.device_get(
                (proximal_cosines, damping_control_cosines)
            )
            candidate_record = {
                "batch": batch,
                "heldout_next_batch_for_previous_directions": candidate_heldout,
                "proximal_consecutive_cosine": {
                    str(gamma): float(proximal_cosines[index])
                    for index, gamma in enumerate(gamma_values)
                },
                "damping_control_consecutive_cosine": {
                    str(gamma): float(damping_control_cosines[index])
                    for index, gamma in enumerate(gamma_values)
                },
            }
            candidate_records.append(candidate_record)
            print(json.dumps({"candidate": candidate_record}, sort_keys=True), flush=True)

        previous_native = native_update
        previous_warm4 = warm4_update
        previous_exact = exact_update
        previous_native_top_basis = current_native_top_basis
        previous_exact_top_basis = exact_top_basis
        previous_proximal = current_proximal
        previous_damping_control = current_damping_control

    transition_records = records[1:]

    def values(name):
        return [record[name] for record in transition_records]

    native_cos = values("native_consecutive_cosine")
    warm4_cos = values("warm4_consecutive_cosine")
    exact_cos = values("exact_consecutive_cosine")
    exact_minus_native = [e - n for e, n in zip(exact_cos, native_cos)]
    warm4_minus_native = [w - n for w, n in zip(warm4_cos, native_cos)]
    exact_attribution = values("exact_subspace_attribution")
    exact_attribution_ci = bootstrap_median_ci(exact_attribution, seed_offset=1)
    exact_native_ci = bootstrap_median_ci(exact_minus_native, seed_offset=2)
    warm4_native_ci = bootstrap_median_ci(warm4_minus_native, seed_offset=3)

    heldout_records = [
        record["heldout_next_batch_for_previous_directions"]
        for record in transition_records
    ]
    warm4_alignment_delta = [
        h["warm4"]["alignment"] - h["native_warm2"]["alignment"]
        for h in heldout_records
    ]
    warm4_q_ratio = [
        h["warm4"]["q"] / h["native_warm2"]["q"] for h in heldout_records
    ]
    exact_alignment_delta = [
        h["exact_rank1600"]["alignment"] - h["native_warm2"]["alignment"]
        for h in heldout_records
    ]
    exact_q_ratio = [
        h["exact_rank1600"]["q"] / h["native_warm2"]["q"]
        for h in heldout_records
    ]

    median_exact_gain = float(np.median(exact_minus_native))
    solver_material = median_exact_gain >= 0.15 and exact_native_ci[0] >= 0.10
    median_attribution = float(np.median(exact_attribution))
    if solver_material:
        mechanism = "approximate_ssi_solver"
    elif median_attribution > 0.70 and exact_attribution_ci[0] > 0.50:
        mechanism = "exact_rank1600_subspace_rotation"
    elif median_attribution < 0.30 and exact_attribution_ci[1] < 0.50:
        mechanism = "within_subspace_coefficients"
    else:
        mechanism = "mixed_subspace_and_coefficients"

    median_native_time = float(np.median(native_seconds[1:]))
    median_warm4_time = float(np.median(warm4_seconds[1:]))
    warm4_overhead = (
        float("nan")
        if args.skip_warm4
        else median_warm4_time / max(median_native_time, EPS) - 1.0
    )
    native_positive = float(
        np.mean([h["native_warm2"]["force_dot"] > 0.0 for h in heldout_records])
    )
    warm4_positive = float(
        np.mean([h["warm4"]["force_dot"] > 0.0 for h in heldout_records])
    )
    warm4_gates = {
        "coherence": float(np.median(warm4_minus_native)) >= 0.15
        and warm4_native_ci[0] >= 0.10,
        "alignment": float(np.median(warm4_alignment_delta)) >= -0.001
        and bootstrap_median_ci(warm4_alignment_delta, seed_offset=4)[0] >= -0.002,
        "residual": float(np.median(warm4_q_ratio)) <= 1.02
        and bootstrap_median_ci(warm4_q_ratio, seed_offset=5)[1] <= 1.05,
        "positive_force": native_positive - warm4_positive <= 0.05,
        "overhead": warm4_overhead <= 0.10,
    }
    if args.skip_warm4:
        warm4_gates = {name: False for name in warm4_gates}

    proximal_test = None
    if args.test_proximal:
        candidate_transitions = candidate_records[1:]
        train_count = 15
        train_indices = list(range(train_count))
        validation_indices = list(range(train_count, len(candidate_transitions)))

        def candidate_series(gamma, family, metric):
            key = str(gamma)
            if metric == "cosine":
                field = f"{family}_consecutive_cosine"
                return [record[field][key] for record in candidate_transitions]
            return [
                record["heldout_next_batch_for_previous_directions"][family][key][
                    metric
                ]
                for record in candidate_transitions
            ]

        def take(series, indices):
            return [series[index] for index in indices]

        proximal_grid = {}
        eligible = []
        for gamma_index, gamma in enumerate(gamma_values):
            proximal_cos = candidate_series(gamma, "proximal", "cosine")
            control_cos = candidate_series(gamma, "damping_control", "cosine")
            proximal_alignment = candidate_series(gamma, "proximal", "alignment")
            control_alignment = candidate_series(
                gamma, "damping_control", "alignment"
            )
            proximal_q = candidate_series(gamma, "proximal", "q")
            control_q = candidate_series(gamma, "damping_control", "q")
            proximal_force_dot = candidate_series(gamma, "proximal", "force_dot")
            control_force_dot = candidate_series(
                gamma, "damping_control", "force_dot"
            )
            coherence_vs_control = [
                left - right for left, right in zip(proximal_cos, control_cos)
            ]
            coherence_vs_native = [
                left - right for left, right in zip(proximal_cos, native_cos)
            ]
            alignment_vs_control = [
                left - right
                for left, right in zip(proximal_alignment, control_alignment)
            ]
            alignment_vs_native = [
                left - h["native_warm2"]["alignment"]
                for left, h in zip(proximal_alignment, heldout_records)
            ]
            q_ratio_vs_control = [
                left / right for left, right in zip(proximal_q, control_q)
            ]
            q_ratio_vs_native = [
                left / h["native_warm2"]["q"]
                for left, h in zip(proximal_q, heldout_records)
            ]

            train = {
                "coherence_vs_control": summarize(
                    take(coherence_vs_control, train_indices)
                ),
                "coherence_vs_native": summarize(
                    take(coherence_vs_native, train_indices)
                ),
                "alignment_vs_control": summarize(
                    take(alignment_vs_control, train_indices)
                ),
                "alignment_vs_native": summarize(
                    take(alignment_vs_native, train_indices)
                ),
                "q_ratio_vs_control": summarize(
                    take(q_ratio_vs_control, train_indices)
                ),
                "q_ratio_vs_native": summarize(
                    take(q_ratio_vs_native, train_indices)
                ),
            }
            train_safe = (
                train["coherence_vs_control"]["median"] >= 0.05
                and train["alignment_vs_control"]["median"] >= -0.001
                and train["q_ratio_vs_control"]["median"] <= 1.02
                and train["alignment_vs_native"]["median"] >= -0.002
                and train["q_ratio_vs_native"]["median"] <= 1.05
            )
            if train_safe:
                eligible.append(
                    (train["coherence_vs_control"]["median"], gamma_index, gamma)
                )
            proximal_grid[str(gamma)] = {
                "gamma_multiplier_of_lambda": PROXIMAL_GAMMA_MULTIPLIERS[
                    gamma_index
                ],
                "train": train,
                "train_safe": train_safe,
                "raw": {
                    "proximal_cosine": proximal_cos,
                    "damping_control_cosine": control_cos,
                    "proximal_alignment": proximal_alignment,
                    "damping_control_alignment": control_alignment,
                    "proximal_q": proximal_q,
                    "damping_control_q": control_q,
                    "proximal_force_dot": proximal_force_dot,
                    "damping_control_force_dot": control_force_dot,
                    "coherence_vs_control": coherence_vs_control,
                    "coherence_vs_native": coherence_vs_native,
                    "alignment_vs_control": alignment_vs_control,
                    "alignment_vs_native": alignment_vs_native,
                    "q_ratio_vs_control": q_ratio_vs_control,
                    "q_ratio_vs_native": q_ratio_vs_native,
                },
            }

        selected_gamma = max(eligible)[2] if eligible else None
        validation = None
        validation_gates = {
            "incremental_coherence": False,
            "absolute_coherence": False,
            "alignment_vs_control": False,
            "residual_vs_control": False,
            "alignment_vs_native": False,
            "residual_vs_native": False,
            "positive_force": False,
        }
        if selected_gamma is not None:
            raw = proximal_grid[str(selected_gamma)]["raw"]

            def validation_summary(name, seed_offset):
                series = take(raw[name], validation_indices)
                return {
                    **summarize(series),
                    "block_bootstrap_95_ci_median": bootstrap_median_ci(
                        series, seed_offset=seed_offset
                    ),
                }

            validation = {
                "coherence_vs_control": validation_summary(
                    "coherence_vs_control", 20
                ),
                "coherence_vs_native": validation_summary(
                    "coherence_vs_native", 21
                ),
                "alignment_vs_control": validation_summary(
                    "alignment_vs_control", 22
                ),
                "q_ratio_vs_control": validation_summary(
                    "q_ratio_vs_control", 23
                ),
                "alignment_vs_native": validation_summary(
                    "alignment_vs_native", 24
                ),
                "q_ratio_vs_native": validation_summary(
                    "q_ratio_vs_native", 25
                ),
            }
            validation_gates = {
                "incremental_coherence": (
                    validation["coherence_vs_control"]["median"] >= 0.05
                    and validation["coherence_vs_control"][
                        "block_bootstrap_95_ci_median"
                    ][0]
                    >= 0.02
                ),
                "absolute_coherence": (
                    validation["coherence_vs_native"]["median"] >= 0.10
                    and validation["coherence_vs_native"][
                        "block_bootstrap_95_ci_median"
                    ][0]
                    >= 0.05
                ),
                "alignment_vs_control": (
                    validation["alignment_vs_control"]["median"] >= -0.001
                    and validation["alignment_vs_control"][
                        "block_bootstrap_95_ci_median"
                    ][0]
                    >= -0.002
                ),
                "residual_vs_control": (
                    validation["q_ratio_vs_control"]["median"] <= 1.02
                    and validation["q_ratio_vs_control"][
                        "block_bootstrap_95_ci_median"
                    ][1]
                    <= 1.05
                ),
                "alignment_vs_native": (
                    validation["alignment_vs_native"]["median"] >= -0.002
                    and validation["alignment_vs_native"][
                        "block_bootstrap_95_ci_median"
                    ][0]
                    >= -0.004
                ),
                "residual_vs_native": (
                    validation["q_ratio_vs_native"]["median"] <= 1.05
                    and validation["q_ratio_vs_native"][
                        "block_bootstrap_95_ci_median"
                    ][1]
                    <= 1.08
                ),
                "positive_force": (
                    np.mean(
                        [
                            raw["proximal_force_dot"][index] > 0.0
                            for index in validation_indices
                        ]
                    )
                    >= np.mean(
                        [
                            raw["damping_control_force_dot"][index] > 0.0
                            for index in validation_indices
                        ]
                    )
                    - 0.05
                    and np.mean(
                        [
                            raw["proximal_force_dot"][index] > 0.0
                            for index in validation_indices
                        ]
                    )
                    >= np.mean(
                        [
                            heldout_records[index]["native_warm2"]["force_dot"] > 0.0
                            for index in validation_indices
                        ]
                    )
                    - 0.05
                ),
            }

        proximal_test = {
            "mathematical_form": (
                "c_i=(s_i*r_i+gamma*<u_i,d_prev>)/(s_i^2+lambda+gamma)"
            ),
            "matched_control": (
                "same denominator s_i^2+lambda+gamma with zero history term"
            ),
            "gamma_values": gamma_values,
            "chronological_train_transitions": [1, train_count],
            "chronological_validation_transitions": [
                train_count + 1,
                len(candidate_transitions),
            ],
            "selection_rule": (
                "among train-safe gammas, maximize median coherence gain over "
                "the matched extra-damping control"
            ),
            "grid": proximal_grid,
            "selected_gamma": selected_gamma,
            "validation": validation,
            "validation_gates": validation_gates,
            "offline_passed": bool(selected_gamma is not None)
            and all(validation_gates.values()),
            "e500_authorized": bool(selected_gamma is not None)
            and all(validation_gates.values()),
            "candidate_records": candidate_records,
        }

    payload = {
        "experiment": "C rank1600 WSSR subspace/solver/coefficients causal audit",
        "read_only_checkpoint": True,
        "parameters_updated": False,
        "optimizer_state_updated": False,
        "walkers_advanced": True,
        "warm4_recomputed": not args.skip_warm4,
        "warm4_fields_are_native_placeholders": args.skip_warm4,
        "warm4_external_valid_source": (
            "/scratch/dexuan1/runs/"
            "C_wssr_subspace_causal_audit_20260802_retry1/audit.json"
            if args.skip_warm4
            else None
        ),
        "source_checkpoint": str(checkpoint),
        "stored_epoch": int(stored_epoch),
        "config_verified": {name: actual for name, (actual, _) in expected.items()},
        "batches": BATCHES,
        "transitions": BATCHES - 1,
        "walkers_per_batch": NCHAINS,
        "mcmc_steps_between_batches": int(config.vmc.nsteps_per_param_update),
        "precision": {
            "model_score_and_updates": str(dtype),
            "basis_projector_solve": "float64",
        },
        "pre_registered_classification": {
            "solver_if_exact_minus_native_median_ge": 0.15,
            "solver_if_exact_minus_native_ci_lower_ge": 0.10,
            "rotation_if_attribution_median_gt": 0.70,
            "rotation_if_attribution_ci_lower_gt": 0.50,
            "coefficients_if_attribution_median_lt": 0.30,
            "coefficients_if_attribution_ci_upper_lt": 0.50,
            "otherwise": "mixed",
        },
        "pre_registered_proximal_test": {
            "gamma_multipliers_of_lambda": PROXIMAL_GAMMA_MULTIPLIERS,
            "train_transitions": 15,
            "validation_transitions": 14,
            "validation_incremental_coherence_median_ge": 0.05,
            "validation_incremental_coherence_ci_lower_ge": 0.02,
            "validation_absolute_coherence_median_ge": 0.10,
            "validation_absolute_coherence_ci_lower_ge": 0.05,
            "validation_alignment_vs_control_median_ge": -0.001,
            "validation_alignment_vs_control_ci_lower_ge": -0.002,
            "validation_q_ratio_vs_control_median_le": 1.02,
            "validation_q_ratio_vs_control_ci_upper_le": 1.05,
            "validation_alignment_vs_native_median_ge": -0.002,
            "validation_alignment_vs_native_ci_lower_ge": -0.004,
            "validation_q_ratio_vs_native_median_le": 1.05,
            "validation_q_ratio_vs_native_ci_upper_le": 1.08,
        },
        "proximal_test": proximal_test,
        "summaries": {
            "native_consecutive_cosine": summarize(native_cos),
            "warm4_consecutive_cosine": summarize(warm4_cos),
            "exact_rank1600_consecutive_cosine": summarize(exact_cos),
            "exact_minus_native_consecutive_cosine": {
                **summarize(exact_minus_native),
                "block_bootstrap_95_ci_median": exact_native_ci,
            },
            "warm4_minus_native_consecutive_cosine": {
                **summarize(warm4_minus_native),
                "block_bootstrap_95_ci_median": warm4_native_ci,
            },
            "exact_subspace_attribution": {
                **summarize(exact_attribution),
                "block_bootstrap_95_ci_median": exact_attribution_ci,
            },
            "exact_subspace_angle_rad": summarize(values("exact_subspace_angle")),
            "exact_coefficient_angle_rad": summarize(values("exact_coefficient_angle")),
            "exact_top32_median_principal_angle_rad": summarize(
                values("exact_top32_median_principal_angle")
            ),
            "exact_top32_min_overlap_cosine": summarize(
                values("exact_top32_min_overlap_cosine")
            ),
            "exact_top32_selected_relative_gap": {
                "median_gap_per_batch": summarize(
                    [record["exact_top32_relative_gap_median"] for record in records]
                ),
                "minimum_gap_per_batch": summarize(
                    [record["exact_top32_relative_gap_min"] for record in records]
                ),
                "fraction_below_1e-3": summarize(
                    [record["exact_top32_gap_fraction_lt_1e3"] for record in records]
                ),
                "fraction_below_1e-2": summarize(
                    [record["exact_top32_gap_fraction_lt_1e2"] for record in records]
                ),
            },
            "native_top16_ritz_residual_median": summarize(
                [record["native_top16_ritz_residual_median"] for record in records]
            ),
            "native_vs_exact_cosine": summarize(
                [record["native_vs_exact_cosine"] for record in records]
            ),
            "warm4_vs_exact_cosine": summarize(
                [record["warm4_vs_exact_cosine"] for record in records]
            ),
            "warm4_heldout_alignment_delta": {
                **summarize(warm4_alignment_delta),
                "block_bootstrap_95_ci_median": bootstrap_median_ci(
                    warm4_alignment_delta, seed_offset=4
                ),
            },
            "warm4_heldout_q_ratio": {
                **summarize(warm4_q_ratio),
                "block_bootstrap_95_ci_median": bootstrap_median_ci(
                    warm4_q_ratio, seed_offset=5
                ),
            },
            "exact_heldout_alignment_delta": summarize(exact_alignment_delta),
            "exact_heldout_q_ratio": summarize(exact_q_ratio),
        },
        "timing": {
            "median_native_warm2_seconds": median_native_time,
            "median_warm4_seconds": median_warm4_time,
            "warm4_replacement_overhead_fraction": warm4_overhead,
            "median_exact_rank1600_seconds": float(np.median(exact_seconds[1:])),
            "median_four_gamma_proximal_grid_seconds": (
                float(np.median(candidate_seconds[1:]))
                if len(candidate_seconds) > 1
                else None
            ),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "decision": {
            "classified_primary_mechanism": mechanism,
            "solver_material": solver_material,
            "warm4_offline_gates": warm4_gates,
            "warm4_offline_passed": all(warm4_gates.values()),
            "e500_authorized_for_warm4": all(warm4_gates.values()),
            "next_branch": {
                "approximate_ssi_solver": (
                    "Test an adaptive residual-based SSI stopping rule; do not smooth the final vector."
                ),
                "exact_rank1600_subspace_rotation": (
                    "Test a persistent cluster-level subspace with a fresh current-RHS solve."
                ),
                "within_subspace_coefficients": (
                    "Test modewise coefficient stabilization inside the current exact subspace."
                ),
                "mixed_subspace_and_coefficients": (
                    "Stop adding gates; the rank-1600 truncated update is intrinsically batch-sensitive."
                ),
            }[mechanism],
        },
        "records": records,
        "finite": bool(
            all(
                np.isfinite(value)
                for record in records
                for name, value in record.items()
                if isinstance(value, (int, float)) and name != "batch"
            )
        ),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    summary = args.output.with_name("summary.md")
    summary.write_text(
        "\n".join(
            [
                "# C rank-1600 WSSR causal subspace audit",
                "",
                "This was read-only with fixed parameters and optimizer state; only walkers advanced.",
                "Absolute C energies are not used for physical claims.",
                "",
                "## Decision",
                "",
                f"- Classified primary mechanism: **{mechanism}**.",
                f"- Median adjacent cosine: native warm2 {np.median(native_cos):.6f}, "
                f"warm4 {np.median(warm4_cos):.6f}, exact rank1600 {np.median(exact_cos):.6f}.",
                f"- Median exact-minus-native adjacent cosine: {median_exact_gain:.6f}; "
                f"95% block-bootstrap CI [{exact_native_ci[0]:.6f}, {exact_native_ci[1]:.6f}].",
                f"- Median exact subspace-attribution ratio: {median_attribution:.6f}; "
                f"95% CI [{exact_attribution_ci[0]:.6f}, {exact_attribution_ci[1]:.6f}].",
                f"- Warm4 replacement overhead: {100.0 * warm4_overhead:.2f}%.",
                f"- Warm4 offline gates: `{json.dumps(warm4_gates, sort_keys=True)}`.",
                f"- Warm4 E500 authorized: **{all(warm4_gates.values())}**.",
                *(
                    [
                        f"- Proximal selected gamma: `{proximal_test['selected_gamma']}`.",
                        f"- Proximal validation gates: `{json.dumps(proximal_test['validation_gates'], sort_keys=True)}`.",
                        f"- Proximal E500 authorized: **{proximal_test['e500_authorized']}**.",
                    ]
                    if proximal_test is not None
                    else []
                ),
                "",
                "## Next branch",
                "",
                payload["decision"]["next_branch"],
                "",
                "Full per-batch results, uncertainty intervals, held-out diagnostics, Ritz residuals,",
                "selected-mode gaps and contribution-weighted top-32 subspace overlaps are in `audit.json`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
