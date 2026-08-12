#!/usr/bin/env python3
"""Read-only attribution audit for the C rank-1600 WSSR trajectory.

The checkpoint parameters are frozen.  Independent MCMC batches update only a
local copy of the WSSR history factor.  This isolates stochastic subspace
motion from parameter drift and measures:

1. current-batch versus history-augmented projector mismatch on the actual
   WSSR force;
2. consecutive history-augmented projector drift (Hutchinson estimate); and
3. how much of a hypothetical complement would be removed by Euclidean and
   function-space relative caps.

The code uses ``o_cur = O_bar.T`` with shape ``(nparam, nsample)``.  No dense
parameter-space projector is formed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from vmcnet.mcmc import position_amplitude_core as pacore
from vmcnet.physics import core as physics_core
from vmcnet.train import runners
from vmcnet.updates import wssr
from vmcnet.utils import io


EPS = 1.0e-12
NCHAINS = 1000


def scalar(x) -> float:
    return float(jax.device_get(x))


def optional(config, name, default):
    return config.get(name, default)


def summarize(rows, name):
    values = np.asarray([row[name] for row in rows], dtype=np.float64)
    return {
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "q10": float(np.quantile(values, 0.10)),
        "q90": float(np.quantile(values, 0.90)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=5000)
    parser.add_argument("--transitions", type=int, default=20)
    parser.add_argument("--probes", type=int, default=16)
    parser.add_argument("--beta-e", type=float, default=0.15)
    parser.add_argument("--beta-f", type=float, default=0.15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    checkpoint = args.run / "checkpoints" / f"{args.epoch}.npz"
    config = io.load_config_dict(str(args.run), "config.json")
    stored_epoch, data, params, optimizer_state, key = io.reload_vmc_state(
        str(checkpoint.parent), checkpoint.name
    )
    if int(stored_epoch) != args.epoch - 1:
        raise ValueError(f"stored epoch {stored_epoch} != {args.epoch - 1}")
    opt = config.vmc.optimizer.wssr_warm_svd_right
    eta = float(opt.eta if float(optional(opt, "eta_S", -1.0)) < 0 else opt.eta_S)
    expected = {
        "nchains": (int(config.vmc.nchains), NCHAINS),
        "rank": (int(opt.sr_rank), 1600),
        "rank_max": (int(opt.sr_rank_max), 1600),
        "eta_S": (eta, 0.3),
        "complement_weight": (float(opt.complement_weight), 0.0),
    }
    mismatches = {k: v for k, v in expected.items() if v[0] != v[1]}
    if mismatches:
        raise ValueError(f"unexpected source configuration: {mismatches}")

    if not isinstance(optimizer_state.core_state, wssr.WSSRWarmSVDCoreState):
        raise TypeError(type(optimizer_state.core_state))
    state = optimizer_state.core_state
    dtype = runners._get_dtype(config)
    # ``_get_dtype`` intentionally disables global x64 for float32 training.
    # Sampling must remain in that original precision because some MCMC state
    # leaves are float32 while Python literals are weakly typed.  The loop
    # below enables x64 only around the isolated projector audit.
    ions, charges, nelec = runners._get_electron_ion_config_as_arrays(config, dtype)
    positions = pacore.get_position_from_data(data)
    log_psi, _, _ = runners._get_and_init_model(
        config.model, ions, charges, nelec, positions, key,
        dtype=dtype, apply_pmap=False,
    )
    local_energy = runners._assemble_mol_local_energy_fn(
        ions, charges, config.problem.ei_softening,
        config.problem.ee_softening, log_psi,
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

    damping = float(opt.damping)
    relative_cutoff_cfg = float(optional(opt, "relative_singular_value_cutoff", -1.0))
    tikhonov_lambda = float(optional(opt, "tikhonov_lambda", -1.0))
    working_rank = min(
        int(optional(opt, "svd_working_rank", opt.sr_rank_max)),
        int(opt.sr_rank_max),
        state.sr_o.shape[1],
    )
    beta_e = float(args.beta_e)
    beta_f = float(args.beta_f)

    def analyze_batch(o_cur, e_cur, state_arg, svd_key, probe_key):
        o_aug, e_aug = wssr.augment_wssr_system(o_cur, e_cur, state_arg, eta)
        u, singular_values, vh, ssi_rank = wssr._wssr_right_svd_decomposition(
            o_aug,
            state_arg,
            svd_key,
            working_rank,
            int(opt.svd_maxiter_initial),
            int(opt.svd_maxiter_warm),
            bool(optional(opt, "exact_first", False)),
            EPS,
            exact_first_force=False,
        )
        leading = jnp.maximum(jnp.abs(singular_values[0]), EPS)
        relative_cutoff, lambda_reg = wssr._wssr_resolve_spectral_controls(
            leading,
            damping,
            relative_cutoff_cfg,
            tikhonov_lambda,
            EPS,
        )
        retained = singular_values / leading > relative_cutoff
        retained_f = retained.astype(o_cur.dtype)
        basis = u * retained_f[None, :]

        force = o_aug @ e_aug
        hist_projection = basis @ (basis.T @ force)
        complement_force = force - hist_projection
        inv_cap, _ = wssr._wssr_inverse_spectral_coefficients(
            jnp.where(retained, singular_values, 1.0),
            leading,
            jnp.maximum(jnp.abs(jnp.asarray(damping, o_cur.dtype)), EPS),
            str(opt.spectral_regularization),
            0.0,
            tikhonov_lambda=tikhonov_lambda,
        )
        retained_direction = basis @ ((basis.T @ force) * inv_cap)

        # Exact projector onto the numerically retained current-batch range.
        # The current Gram has retained eigenvalues down to roughly 1e-7 of
        # the leading value.  A fp32 pseudoinverse is not an orthogonal
        # projector at that condition number (and can even increase a vector
        # norm), so isolate this audit-only projector calculation in fp64.
        o_cur_projector = o_cur.astype(jnp.float64)
        gram = o_cur_projector.T @ o_cur_projector
        gram = 0.5 * (gram + gram.T)
        evals, rotation = jnp.linalg.eigh(gram)
        gram_leading = jnp.maximum(jnp.max(evals), EPS)
        current_retained = evals > jnp.square(relative_cutoff) * gram_leading
        current_inverse = jnp.where(
            current_retained,
            1.0 / jnp.maximum(evals, EPS),
            0.0,
        )

        def project_current(vector):
            rhs = o_cur_projector.T @ vector.astype(jnp.float64)
            coeff = rotation @ (current_inverse * (rotation.T @ rhs))
            return o_cur_projector @ coeff

        current_projection = project_current(force)
        current_component_of_complement = project_current(complement_force)
        hist_of_current = basis @ (basis.T @ current_projection)

        # Trace(P_current P_history), computed entirely through the sample Gram.
        current_action = o_cur_projector.T @ basis.astype(jnp.float64)
        rotated_action = rotation.T @ current_action
        trace_overlap = jnp.sum(
            current_inverse[:, None] * jnp.square(rotated_action)
        )
        current_rank = jnp.sum(current_retained.astype(jnp.int32))
        active_rank = jnp.sum(retained.astype(jnp.int32))
        current_containment = trace_overlap / jnp.maximum(
            current_rank.astype(o_cur.dtype), 1.0
        )

        # Consecutive history-projector drift.  Exact U.T@U_prev is prohibitive
        # at FermiNet scale, so use a reproducible Hutchinson trace estimate.
        probes = jax.random.rademacher(
            probe_key,
            (o_cur.shape[0], args.probes),
            dtype=o_cur.dtype,
        )
        previous_mask = (
            jnp.arange(state_arg.u.shape[1]) < state_arg.sr_rank0
        ).astype(o_cur.dtype)
        previous_basis = state_arg.u * previous_mask[None, :]
        projected_now = basis @ (basis.T @ probes)
        projected_previous = previous_basis @ (previous_basis.T @ probes)
        projector_diff_probes = projected_now - projected_previous
        projector_frob2 = jnp.mean(jnp.sum(jnp.square(projector_diff_probes), axis=0))
        projector_drift = jnp.sqrt(
            projector_frob2
            / jnp.maximum(
                active_rank.astype(o_cur.dtype)
                + state_arg.sr_rank0.astype(o_cur.dtype),
                1.0,
            )
        )
        previous_force_projection = previous_basis @ (previous_basis.T @ force)

        retained_action = o_cur.T @ retained_direction
        complement_action = o_cur.T @ complement_force
        retained_norm = jnp.linalg.norm(retained_direction)
        complement_norm = jnp.linalg.norm(complement_force)
        retained_action_norm = jnp.linalg.norm(retained_action)
        complement_action_norm = jnp.linalg.norm(complement_action)
        alpha_e = jnp.minimum(
            1.0,
            beta_e * retained_norm / jnp.maximum(complement_norm, EPS),
        )
        alpha_f = jnp.minimum(
            1.0,
            beta_f * retained_action_norm
            / jnp.maximum(complement_action_norm, EPS),
        )
        alpha_dual = jnp.minimum(alpha_e, alpha_f)
        raw_total_action = retained_action + complement_action
        removed_action = (1.0 - alpha_dual) * complement_action

        core_result = wssr._wssr_update_from_svd(
            o_aug,
            e_aug,
            state_arg,
            u,
            singular_values,
            vh,
            damping,
            float(opt.norm_constraint),
            int(opt.sr_rank_max),
            float(optional(opt, "sr_scale", 1.1)),
            False,
            rank_update_max=working_rank,
            spectral_regularization=str(opt.spectral_regularization),
            complement_weight=0.0,
            relative_singular_value_cutoff=relative_cutoff_cfg,
            tikhonov_lambda=tikhonov_lambda,
            eps=EPS,
        )
        u_state = jnp.zeros_like(state_arg.u)
        rank_mask = (jnp.arange(u.shape[1]) < ssi_rank).astype(o_cur.dtype)
        u_state = u_state.at[:, :u.shape[1]].set(u * rank_mask[None, :])
        next_state = wssr.WSSRWarmSVDCoreState(
            core_result.state.sr_o,
            core_result.state.ek,
            core_result.state.sr_rank0,
            core_result.state.sr_rank,
            u_state,
            jnp.asarray(True),
        )
        metrics = {
            "current_rank": current_rank,
            "history_active_rank": active_rank,
            "ssi_rank": ssi_rank,
            "timeline_force_mismatch": jnp.linalg.norm(
                current_projection - hist_projection
            ) / jnp.maximum(jnp.linalg.norm(force), EPS),
            "current_force_missed_by_history": jnp.linalg.norm(
                current_projection - hist_of_current
            ) / jnp.maximum(jnp.linalg.norm(current_projection), EPS),
            "current_subspace_containment": jnp.clip(current_containment, 0.0, 1.0),
            "current_subspace_miss_rms": jnp.sqrt(
                jnp.maximum(1.0 - current_containment, 0.0)
            ),
            "complement_current_component_ratio": jnp.linalg.norm(
                current_component_of_complement
            ) / jnp.maximum(complement_norm, EPS),
            "projector_drift_hutchinson": projector_drift,
            "projector_drift_on_force": jnp.linalg.norm(
                hist_projection - previous_force_projection
            ) / jnp.maximum(jnp.linalg.norm(force), EPS),
            "retained_direction_norm": retained_norm,
            "complement_force_norm": complement_norm,
            "retained_function_action_norm": retained_action_norm,
            "complement_function_action_norm": complement_action_norm,
            "alpha_euclidean": alpha_e,
            "alpha_function": alpha_f,
            "alpha_dual": alpha_dual,
            "complement_function_action_removed_fraction": 1.0 - alpha_dual,
            "removed_action_relative_to_raw_total": jnp.linalg.norm(removed_action)
            / jnp.maximum(jnp.linalg.norm(raw_total_action), EPS),
            "lambda_reg": lambda_reg,
            "leading_singular_value": leading,
        }
        return next_state, metrics

    analyze_batch_jit = jax.jit(analyze_batch)

    def sample_batch(data_arg, key_arg):
        acceptance, next_data, next_key = walker_fn(params, data_arg, key_arg)
        pos = pacore.get_position_from_data(next_data)
        energy, clipped_energies, _ = energy_fn(params, pos)
        score, _ = wssr.center_and_scale_score_matrix(log_psi, params, pos)
        epsilon = wssr.center_and_scale_energy_residuals(clipped_energies, energy)
        return next_data, next_key, acceptance, energy, score, epsilon

    rows = []
    started = time.perf_counter()
    for transition in range(args.transitions):
        jax.config.update("jax_enable_x64", False)
        data, key, acceptance, energy, o_cur, e_cur = sample_batch(data, key)
        jax.config.update("jax_enable_x64", True)
        if not jax.config.x64_enabled:
            raise RuntimeError("fp64 current-projector audit requires JAX x64")
        key, svd_key, probe_key = jax.random.split(key, 3)
        state, metrics = analyze_batch_jit(
            o_cur, e_cur, state, svd_key, probe_key
        )
        metrics = jax.device_get(metrics)
        row = {
            "transition": transition,
            "energy": scalar(energy),
            "acceptance": scalar(acceptance),
            **{name: float(value) for name, value in metrics.items()},
        }
        if row["complement_current_component_ratio"] > 1.0001:
            raise FloatingPointError(
                "current projector increased complement norm: "
                f"{row['complement_current_component_ratio']}"
            )
        if not (-1.0e-4 <= row["current_subspace_containment"] <= 1.0001):
            raise FloatingPointError(
                "invalid current-subspace containment: "
                f"{row['current_subspace_containment']}"
            )
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    csv_path = args.output / "metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metric_names = [
        "timeline_force_mismatch",
        "current_force_missed_by_history",
        "current_subspace_containment",
        "current_subspace_miss_rms",
        "complement_current_component_ratio",
        "projector_drift_hutchinson",
        "projector_drift_on_force",
        "alpha_euclidean",
        "alpha_function",
        "alpha_dual",
        "complement_function_action_removed_fraction",
        "removed_action_relative_to_raw_total",
    ]
    summary = {
        "source_run": str(args.run),
        "source_epoch": args.epoch,
        "transitions": args.transitions,
        "hutchinson_probes": args.probes,
        "eta_S": eta,
        "beta_E": beta_e,
        "beta_F": beta_f,
        "wall_seconds": time.perf_counter() - started,
        "metrics": {name: summarize(rows, name) for name in metric_names},
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
