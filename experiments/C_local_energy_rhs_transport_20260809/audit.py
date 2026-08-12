#!/usr/bin/env python3
"""Early-C local-energy RHS staleness and solve-level transport audit."""

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
from vmcnet.updates import parse_optimizer_config, wssr
from vmcnet.utils import io


NCHAINS = 1000
K_VALUES = (1, 5, 20)
FINAL_STEP = 20
EPS = 1.0e-14


def scalar(x) -> float:
    return float(jax.device_get(x))


def flat64(tree):
    return jax.flatten_util.ravel_pytree(
        jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float64), tree)
    )[0]


def cast64(tree):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float64), tree)


def tree_delta(new, old):
    return jax.tree_util.tree_map(
        lambda x, y: jnp.asarray(x, jnp.float64) - jnp.asarray(y, jnp.float64),
        new,
        old,
    )


def safe_cosine(a, b) -> float:
    aa = np.asarray(jax.device_get(a), dtype=np.float64).ravel()
    bb = np.asarray(jax.device_get(b), dtype=np.float64).ravel()
    return float(np.dot(aa, bb) / max(np.linalg.norm(aa) * np.linalg.norm(bb), EPS))


def configure_optimizer(config, method: str) -> dict:
    config.vmc.optimizer_type = "spring" if method == "spring" else "wssr_warm_svd_right"
    if method == "spring":
        opt = config.vmc.optimizer.spring
        opt.schedule_type = "inverse_time"
        opt.learning_rate = 0.02
        opt.learning_decay_rate = 1.0e-4
        opt.mu = 0.99
        opt.damping = 1.0e-3
        opt.constrain_norm = True
        opt.norm_constraint = 1.0e-3
        return {
            "optimizer": "spring", "mu": 0.99, "learning_rate": 0.02,
            "damping": 1.0e-3, "norm_constraint": 1.0e-3,
        }

    opt = config.vmc.optimizer.wssr_warm_svd_right
    settings = {
        "schedule_type": "inverse_time",
        "learning_rate": 0.04,
        "learning_decay_rate": 1.0e-4,
        "eta": 0.3,
        "eta_S": 0.3,
        "eta_g": 0.3,
        "adaptive_S_average": False,
        "adaptive_g_average": False,
        "enable_gradient_transport": False,
        "damping": 3.0e-4,
        "constrain_norm": True,
        "norm_constraint": 1.0e-3,
        "sr_rank": 1600,
        "sr_rank_max": 1600,
        "sr_storage_rank": 1600,
        "svd_working_rank": 1600,
        "sr_scale": 1.1,
        "svd_maxiter_initial": 40,
        "svd_maxiter_warm": 2,
        "store_warm_u": True,
        "exact_first": False,
        "exact_first_force": False,
        "spectral_regularization": "tikhonov",
        "relative_singular_value_cutoff": 3.0e-4,
        "tikhonov_lambda": 1.0e-3,
        "mixed_precision_solve": True,
        "experimental_mode": "none",
        "complement_weight": 0.0,
        "adaptive_complement_beta": 0.0,
        "solution_recurrence_mode": "none",
        "update_diagnostics": False,
        "reliability_diagnostics": False,
    }
    for name, value in settings.items():
        opt[name] = value
    return {
        "optimizer": "wssr_warm_svd_right", "rank": 1600,
        "eta": 0.3, "learning_rate": 0.04, "tikhonov_lambda": 1.0e-3,
        "norm_constraint": 1.0e-3, "ssi": "40/2",
    }


def microbatched_values_and_jvp(batch_fn, params, tangent, positions, microbatch):
    values, tangents = [], []
    for start in range(0, positions.shape[0], microbatch):
        x = positions[start : start + microbatch]
        value, derivative = batch_fn(params, tangent, x)
        values.append(np.asarray(jax.device_get(value), dtype=np.float64))
        tangents.append(np.asarray(jax.device_get(derivative), dtype=np.float64))
    return np.concatenate(values), np.concatenate(tangents)


def microbatched_values(batch_fn, params, positions, microbatch):
    values = []
    for start in range(0, positions.shape[0], microbatch):
        x = positions[start : start + microbatch]
        value = batch_fn(params, x)
        values.append(np.asarray(jax.device_get(value), dtype=np.float64))
    return np.concatenate(values)


def microbatched_logpsi_jvp(batch_fn, params, tangent, positions, microbatch):
    _, derivative = microbatched_values_and_jvp(
        batch_fn, params, tangent, positions, microbatch
    )
    return derivative


def microbatched_logpsi_vjp(batch_fn, params, positions, weights, microbatch):
    accumulator = jax.tree_util.tree_map(jnp.zeros_like, params)
    for start in range(0, positions.shape[0], microbatch):
        x = positions[start : start + microbatch]
        w = jnp.asarray(weights[start : start + microbatch], jnp.float64)

        contribution = batch_fn(params, x, w)
        accumulator = jax.tree_util.tree_map(jnp.add, accumulator, contribution)
    return flat64(accumulator)


def centered(x):
    x = np.asarray(x, dtype=np.float64)
    return x - np.mean(x)


def evaluate_equation(energy_fn, log_psi, params, data, include_score=True):
    positions = pacore.get_position_from_data(data)
    energy, clipped, stats = energy_fn(params, positions)
    score = None
    if include_score:
        score, _ = wssr.center_and_scale_score_matrix(log_psi, params, positions)
    epsilon = wssr.center_and_scale_energy_residuals(clipped, energy)
    return score, epsilon, stats


def solve_k2(old_score, current_score, old_rhs, current_rhs, correction, eta, rank, damping):
    # A=[sqrt(eta) O_old, sqrt(1-eta) O_current], with O shaped P x N.
    so, sc = math.sqrt(eta), math.sqrt(1.0 - eta)
    # Keep the large score factors and their Gram products on the GPU.  Moving
    # these P x N arrays to NumPy would turn the audit into a CPU GEMM test.
    oo = jnp.asarray(old_score, jnp.float64)
    oc = jnp.asarray(current_score, jnp.float64)
    blocks = jnp.block([
        [eta * (oo.T @ oo), so * sc * (oo.T @ oc)],
        [so * sc * (oc.T @ oo), (1.0 - eta) * (oc.T @ oc)],
    ])
    evals, vecs = jnp.linalg.eigh(blocks)
    order = jnp.argsort(evals)[::-1][:rank]
    evals = jnp.maximum(evals[order], 0.0)
    vecs = vecs[:, order]
    rhs = jnp.concatenate(
        (so * jnp.asarray(old_rhs, jnp.float64), sc * jnp.asarray(current_rhs, jnp.float64))
    )
    corrected_rhs = jnp.concatenate(
        (
            so * (jnp.asarray(old_rhs, jnp.float64) + jnp.asarray(correction, jnp.float64)),
            sc * jnp.asarray(current_rhs, jnp.float64),
        )
    )

    def coefficients(target):
        return vecs @ ((vecs.T @ target) / (evals + damping))

    alpha_raw = coefficients(rhs)
    alpha_corrected = coefficients(corrected_rhs)
    raw = so * (oo @ alpha_raw[:NCHAINS]) + sc * (oc @ alpha_raw[NCHAINS:])
    corrected_update = (
        so * (oo @ alpha_corrected[:NCHAINS])
        + sc * (oc @ alpha_corrected[NCHAINS:])
    )
    return raw, corrected_update, {
        "leading_gram_eigenvalue": scalar(evals[0]),
        "trailing_retained_gram_eigenvalue": scalar(evals[-1]),
    }


def heldout_residual(logpsi_jvp_batch, params64, positions64, epsilon, direction, unravel, microbatch):
    direction_tree = unravel(jnp.asarray(direction, jnp.float64))
    action = centered(
        microbatched_logpsi_jvp(
            logpsi_jvp_batch, params64, direction_tree, positions64, microbatch
        )
    ) / math.sqrt(positions64.shape[0])
    target = np.asarray(jax.device_get(epsilon), dtype=np.float64)
    return float(np.linalg.norm(target - action) / max(np.linalg.norm(target), EPS))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("wssr", "spring"), required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--microbatch", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not jax.config.x64_enabled:
        raise RuntimeError("JAX_ENABLE_X64 must be enabled")

    config = io.load_config_dict(str(args.source), "config.json")
    protocol = configure_optimizer(config, args.method)
    checkpoint = args.source / "checkpoints" / "1000.npz"
    epoch, data, params, _, key = io.reload_vmc_state(str(checkpoint.parent), checkpoint.name)
    if int(epoch) != 999 or int(config.vmc.nchains) != NCHAINS:
        raise ValueError((epoch, config.vmc.nchains))
    key = jax.random.fold_in(key, 9000 + args.replicate)

    dtype = runners._get_dtype(config)
    ions, charges, nelec = runners._get_electron_ion_config_as_arrays(config, dtype)
    positions = pacore.get_position_from_data(data)
    log_psi, _, _ = runners._get_and_init_model(
        config.model, ions, charges, nelec, positions, key,
        dtype=dtype, apply_pmap=False,
    )
    local_energy = runners._assemble_mol_local_energy_fn(
        ions, charges, config.problem.ei_softening, config.problem.ee_softening, log_psi
    )
    clipping_fn = runners._get_clipping_fn(config.vmc)
    energy_fn = jax.jit(physics_core.create_energy_and_statistics_fn(
        local_energy, NCHAINS, clipping_fn, config.vmc.nan_safe,
        record_raw_local_energies=True,
    ))
    equation_jit = jax.jit(
        lambda p, d: evaluate_equation(energy_fn, log_psi, p, d)
    )
    _, walker_fn = runners._get_mcmc_fns(config.vmc, log_psi, apply_pmap=False)
    update_fn, optimizer_state, key = parse_optimizer_config.initialize_optimizer(
        log_psi, local_energy, clipping_fn, config.vmc, params, data,
        pacore.get_position_from_data, pacore.get_update_data_fn(log_psi), key,
        apply_pmap=False,
    )

    snapshots = {}
    trajectory_metrics = []
    started = time.perf_counter()
    for step in range(FINAL_STEP):
        acceptance, data, key = walker_fn(params, data, key)
        if step in {FINAL_STEP - k for k in K_VALUES}:
            snapshots[step] = (
                jax.tree_util.tree_map(lambda x: x.copy(), params), data.copy()
            )
        params, data, optimizer_state, metrics, key = update_fn(
            params, data, optimizer_state, key
        )
        trajectory_metrics.append({
            "step": step + 1,
            "energy": scalar(metrics["energy"]),
            "variance": scalar(metrics["variance"]),
            "acceptance": scalar(acceptance),
        })
        print(json.dumps(trajectory_metrics[-1]), flush=True)

    # Two independently advanced current-parameter batches: the first is part
    # of K=2; only the second is used for the held-out decision.
    _, current_data, key = walker_fn(params, data, key)
    _, heldout_data, key = walker_fn(params, current_data, key)
    current_score, current_epsilon, _ = equation_jit(params, current_data)
    _, heldout_epsilon, heldout_stats = evaluate_equation(
        energy_fn, log_psi, params, heldout_data, include_score=False
    )

    # Materialize every matched fp32 SR equation before changing JAX's global
    # x64 mode.  Calling the fp32 model closure after that switch would retrace
    # fwdlap with mixed captured constants.  Host storage also avoids keeping
    # four P x N score matrices resident on the GPU during the fp64 JVP phase.
    current_score_host = np.asarray(jax.device_get(current_score), dtype=np.float32)
    current_epsilon_host = np.asarray(jax.device_get(current_epsilon), dtype=np.float32)
    heldout_epsilon_host = np.asarray(jax.device_get(heldout_epsilon), dtype=np.float32)
    old_equations = {}
    for k in K_VALUES:
        old_step = FINAL_STEP - k
        old_params, old_data = snapshots[old_step]
        old_score, old_epsilon, old_stats = equation_jit(old_params, old_data)
        old_equations[old_step] = {
            "score": np.asarray(jax.device_get(old_score), dtype=np.float32),
            "epsilon": np.asarray(jax.device_get(old_epsilon), dtype=np.float32),
            "raw_variance": scalar(old_stats["variance_noclip"]),
        }
        del old_score, old_epsilon, old_stats
    del current_score, current_epsilon, heldout_epsilon, equation_jit

    # Optimizer state is deliberately excluded from the diagnostic and can be
    # released before constructing the two score factors.
    del optimizer_state
    gc.collect()

    # runners._get_dtype follows the matched training config and therefore
    # switches global JAX x64 off for the fp32 trajectory.  Re-enable it only
    # after the trajectory is complete: every diagnostic cast/JVP/Gram solve
    # below is required to be genuinely fp64, while the optimizer path remains
    # exactly the matched fp32 path.
    jax.config.update("jax_enable_x64", True)
    if not jax.config.x64_enabled:
        raise RuntimeError("failed to re-enable x64 for the diagnostic phase")
    dtype_probe = jnp.asarray(0.0, jnp.float64)
    if dtype_probe.dtype != jnp.float64:
        raise RuntimeError(f"diagnostic dtype probe is {dtype_probe.dtype}, not float64")

    params64 = cast64(params)
    _, unravel64 = jax.flatten_util.ravel_pytree(params64)
    ions64 = jnp.asarray(ions, jnp.float64)
    charges64 = jnp.asarray(charges, jnp.float64)
    diagnostic_positions64 = jnp.asarray(
        pacore.get_position_from_data(current_data), jnp.float64
    )
    # The training log_psi closure captures fp32 ion/model constants.  Build an
    # identical apply closure with fp64 captured constants; otherwise mixing
    # fp64 positions with the original closure fails inside fwdlap primitives.
    log_psi64, diagnostic_init_params, _ = runners._get_and_init_model(
        config.model, ions64, charges64, nelec, diagnostic_positions64, key,
        dtype=jnp.float64, apply_pmap=False,
    )
    if jax.tree_util.tree_structure(diagnostic_init_params) != jax.tree_util.tree_structure(params64):
        raise RuntimeError("fp64 diagnostic model parameter structure mismatch")
    del diagnostic_init_params
    local_energy64 = runners._assemble_mol_local_energy_fn(
        ions64, charges64, config.problem.ei_softening,
        config.problem.ee_softening, log_psi64,
    )
    local_energy_jvp_batch = jax.jit(
        lambda p, d, x: jax.jvp(
            lambda q: jax.vmap(local_energy64, in_axes=(None, 0))(q, x),
            (p,), (d,),
        )
    )
    local_energy_batch = jax.jit(
        lambda p, x: jax.vmap(local_energy64, in_axes=(None, 0))(p, x)
    )
    logpsi_jvp_batch = jax.jit(
        lambda p, d, x: jax.jvp(
            lambda q: jax.vmap(log_psi64, in_axes=(None, 0))(q, x),
            (p,), (d,),
        )
    )
    def logpsi_vjp_impl(p, x, weights):
        _, pullback = jax.vjp(
            lambda q: jax.vmap(log_psi64, in_axes=(None, 0))(q, x), p
        )
        return pullback(weights)[0]
    logpsi_vjp_batch = jax.jit(logpsi_vjp_impl)
    records = []
    for k in K_VALUES:
        old_step = FINAL_STEP - k
        old_params, old_data = snapshots[old_step]
        old_params64 = cast64(old_params)
        delta64 = tree_delta(params, old_params)
        old_positions64 = jnp.asarray(pacore.get_position_from_data(old_data), jnp.float64)
        old_values, jdelta = microbatched_values_and_jvp(
            local_energy_jvp_batch, old_params64, delta64,
            old_positions64, args.microbatch
        )
        new_same_values = microbatched_values(
            local_energy_batch, params64, old_positions64, args.microbatch
        )
        fd = new_same_values - old_values
        eps_raw = centered(old_values)
        j_center = centered(jdelta)
        fd_center = centered(fd)
        j_norm = np.linalg.norm(jdelta)

        force_den = microbatched_logpsi_vjp(
            logpsi_vjp_batch, old_params64, old_positions64,
            eps_raw / old_positions64.shape[0], args.microbatch,
        )
        force_j = microbatched_logpsi_vjp(
            logpsi_vjp_batch, old_params64, old_positions64,
            j_center / old_positions64.shape[0], args.microbatch,
        )
        score_delta = centered(microbatched_logpsi_jvp(
            logpsi_jvp_batch, old_params64, delta64,
            old_positions64, args.microbatch
        ))
        fisher_delta = microbatched_logpsi_vjp(
            logpsi_vjp_batch, old_params64, old_positions64,
            score_delta / old_positions64.shape[0], args.microbatch,
        )

        old_equation = old_equations[old_step]
        old_score = old_equation["score"]
        old_epsilon = old_equation["epsilon"]
        correction = j_center / math.sqrt(NCHAINS)
        raw_direction, corrected_direction, spectral = solve_k2(
            old_score, current_score_host, old_epsilon, current_epsilon_host,
            correction, eta=0.3, rank=1600, damping=1.0e-3,
        )
        held_positions64 = jnp.asarray(
            pacore.get_position_from_data(heldout_data), jnp.float64
        )
        raw_residual = heldout_residual(
            logpsi_jvp_batch, params64, held_positions64, heldout_epsilon_host,
            raw_direction, unravel64, args.microbatch,
        )
        corrected_residual = heldout_residual(
            logpsi_jvp_batch, params64, held_positions64, heldout_epsilon_host,
            corrected_direction, unravel64, args.microbatch,
        )
        record = {
            "method": args.method,
            "replicate": args.replicate,
            "k": k,
            "old_step": old_step,
            "actual_delta_norm": float(np.linalg.norm(np.asarray(jax.device_get(flat64(delta64))))),
            "R_J": float(np.linalg.norm(j_center) / max(np.linalg.norm(eps_raw), EPS)),
            "R_FD": float(np.linalg.norm(fd_center) / max(np.linalg.norm(eps_raw), EPS)),
            "R_mean": float(math.sqrt(NCHAINS) * abs(np.mean(jdelta)) / max(j_norm, EPS)),
            "R_param": float(np.linalg.norm(np.asarray(jax.device_get(force_j))) / max(np.linalg.norm(np.asarray(jax.device_get(force_den))), EPS)),
            "fisher_proxy_ratio": float(np.linalg.norm(np.asarray(jax.device_get(fisher_delta))) / max(np.linalg.norm(np.asarray(jax.device_get(force_den))), EPS)),
            "jvp_fd_cosine_centered": safe_cosine(j_center, fd_center),
            "linearization_relative_error_centered": float(np.linalg.norm(j_center - fd_center) / max(np.linalg.norm(fd_center), EPS)),
            "old_raw_variance": old_equation["raw_variance"],
            "direction_cosine_raw_vs_transport": safe_cosine(raw_direction, corrected_direction),
            "heldout_residual_raw": raw_residual,
            "heldout_residual_transport": corrected_residual,
            "heldout_residual_improvement": float((raw_residual - corrected_residual) / max(raw_residual, EPS)),
            **spectral,
        }
        if not all(math.isfinite(v) for v in record.values() if isinstance(v, float)):
            raise FloatingPointError(record)
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        del old_score, old_epsilon, raw_direction, corrected_direction
        gc.collect()

    result = {
        "experiment": "C local-energy RHS transport audit",
        "diagnostic_only": True,
        "source_checkpoint": str(checkpoint),
        "method": args.method,
        "replicate": args.replicate,
        "walkers": NCHAINS,
        "mcmc_steps_per_update": int(config.vmc.nsteps_per_param_update),
        "trajectory_updates": FINAL_STEP,
        "microbatch": args.microbatch,
        "x64": bool(jax.config.x64_enabled),
        "trajectory_dtype": "float32",
        "diagnostic_dtype": str(dtype_probe.dtype),
        "optimizer_protocol": protocol,
        "heldout_current_raw_variance": scalar(heldout_stats["variance_noclip"]),
        "trajectory_metrics": trajectory_metrics,
        "records": records,
        "elapsed_seconds": time.perf_counter() - started,
        "git_commit": subprocess.check_output(
            ["git", "-C", "/scratch/dexuan1/vmcnet", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
