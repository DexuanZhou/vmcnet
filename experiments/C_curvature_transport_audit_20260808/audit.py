#!/usr/bin/env python3
"""Read-only one-step curvature-transport audit for early-carbon WSSR.

For each replicate, one real optimizer step is applied in memory.  On the same
electronic coordinates we evaluate score factors at theta, theta + delta/2,
and theta + delta.  We then compare the old Fisher/SR action with the finite-
difference first-order transported action

    S_transport v = 2 S(theta + delta/2) v - S(theta) v

against S(theta + delta) v.  No dense parameter-space S matrix is formed and
nothing is written back to the source checkpoint.
"""

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


EPS = 1.0e-12
NCHAINS = 1000
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260808


def scalar(value) -> float:
    return float(jax.device_get(value))


def safe_cosine(left, right):
    return jnp.vdot(left, right) / jnp.maximum(
        jnp.linalg.norm(left) * jnp.linalg.norm(right), EPS
    )


def relative_error(candidate, target):
    return jnp.linalg.norm(candidate - target) / jnp.maximum(
        jnp.linalg.norm(target), EPS
    )


def fisher_action(score, probes):
    """Apply S=O O^T to parameter-space probe columns."""
    return score @ (score.T @ probes)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def bootstrap_median_ci(values, seed_offset=0):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    draws = rng.integers(
        0, len(values), size=(BOOTSTRAP_REPLICATES, len(values))
    )
    medians = np.median(values[draws], axis=1)
    return [float(x) for x in np.quantile(medians, [0.025, 0.975])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = args.run / "checkpoints" / f"{args.epoch}.npz"
    config = io.load_config_dict(str(args.run), "config.json")
    stored_epoch, base_data, base_params, base_optimizer_state, base_key = (
        io.reload_vmc_state(str(checkpoint.parent), checkpoint.name)
    )
    if int(stored_epoch) != args.epoch - 1:
        raise ValueError(f"stored epoch {stored_epoch} != {args.epoch - 1}")
    if int(config.vmc.nchains) != NCHAINS:
        raise ValueError(f"expected {NCHAINS} walkers, got {config.vmc.nchains}")
    if config.vmc.optimizer_type != "wssr_warm_svd_right":
        raise ValueError(config.vmc.optimizer_type)
    opt = config.vmc.optimizer.wssr_warm_svd_right
    expected = {
        "rank": (int(opt.sr_rank), 1600),
        "eta_S": (float(opt.get("eta_S", opt.eta)), 0.0),
        "eta_g": (float(opt.get("eta_g", opt.eta)), 0.0),
        "learning_rate": (float(opt.learning_rate), 0.04),
        "tikhonov_lambda": (float(opt.tikhonov_lambda), 1.0e-3),
        "norm_constraint": (float(opt.norm_constraint), 1.0e-3),
    }
    mismatches = {
        name: pair for name, pair in expected.items() if pair[0] != pair[1]
    }
    if mismatches:
        raise ValueError(f"unexpected source configuration: {mismatches}")

    dtype = runners._get_dtype(config)
    ions, charges, nelec = runners._get_electron_ion_config_as_arrays(
        config, dtype
    )
    positions = pacore.get_position_from_data(base_data)
    log_psi, _, _ = runners._get_and_init_model(
        config.model,
        ions,
        charges,
        nelec,
        positions,
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
    _, walker_fn = runners._get_mcmc_fns(
        config.vmc, log_psi, apply_pmap=False
    )
    update_data_fn = pacore.get_update_data_fn(log_psi)
    update_param_fn, _, _ = parse_optimizer_config.initialize_optimizer(
        log_psi,
        local_energy,
        clipping_fn,
        config.vmc,
        base_params,
        base_data,
        pacore.get_position_from_data,
        update_data_fn,
        base_key,
        apply_pmap=False,
    )
    flat_base, unravel = jax.flatten_util.ravel_pytree(base_params)

    def evaluate(params, data):
        pos = pacore.get_position_from_data(data)
        energy, clipped, stats = energy_fn(params, pos)
        score, _ = wssr.center_and_scale_score_matrix(log_psi, params, pos)
        epsilon = wssr.center_and_scale_energy_residuals(clipped, energy)
        return score, epsilon, stats["local_energies_noclip"]

    evaluate_jit = jax.jit(evaluate)

    @jax.jit
    def action_metrics(old_score, mid_score, new_score, batch_score, probes):
        old_action = fisher_action(old_score, probes)
        mid_action = fisher_action(mid_score, probes)
        new_action = fisher_action(new_score, probes)
        batch_action = fisher_action(batch_score, probes)
        transported_action = 2.0 * mid_action - old_action
        per_probe_old_error = jnp.linalg.norm(
            old_action - new_action, axis=0
        ) / jnp.maximum(jnp.linalg.norm(new_action, axis=0), EPS)
        per_probe_transport_error = jnp.linalg.norm(
            transported_action - new_action, axis=0
        ) / jnp.maximum(jnp.linalg.norm(new_action, axis=0), EPS)
        return {
            "old_error": relative_error(old_action, new_action),
            "transport_error": relative_error(transported_action, new_action),
            "transport_error_ratio": relative_error(
                transported_action, new_action
            ) / jnp.maximum(relative_error(old_action, new_action), EPS),
            "old_cosine": safe_cosine(old_action, new_action),
            "transport_cosine": safe_cosine(transported_action, new_action),
            "sampling_error": relative_error(new_action, batch_action),
            "sampling_to_staleness_ratio": relative_error(
                new_action, batch_action
            ) / jnp.maximum(relative_error(old_action, new_action), EPS),
            "old_total_batch_error": relative_error(old_action, batch_action),
            "transport_total_batch_error": relative_error(
                transported_action, batch_action
            ),
            "per_probe_old_error": per_probe_old_error,
            "per_probe_transport_error": per_probe_transport_error,
        }

    rows = []
    started = time.perf_counter()
    probe_names = ("actual_delta", "new_force", "random_0", "random_1")
    for replicate in range(args.replicates):
        key = jax.random.fold_in(base_key, replicate + 1)
        _, old_data, key = walker_fn(base_params, base_data, key)
        old_score, old_epsilon, old_raw = evaluate_jit(base_params, old_data)

        updated_params, updated_data, _, _, key = update_param_fn(
            base_params, old_data, base_optimizer_state, key
        )
        flat_updated, _ = jax.flatten_util.ravel_pytree(updated_params)
        delta = flat_updated - flat_base
        midpoint_params = jax.tree_util.tree_map(
            lambda old, new: old + 0.5 * (new - old),
            base_params,
            updated_params,
        )
        mid_score, _, _ = evaluate_jit(midpoint_params, updated_data)
        new_score, new_epsilon, new_raw = evaluate_jit(
            updated_params, updated_data
        )
        new_force = new_score @ new_epsilon

        _, batch_data, key = walker_fn(updated_params, updated_data, key)
        batch_score, _, batch_raw = evaluate_jit(updated_params, batch_data)

        key, probe_key = jax.random.split(key)
        random_probes = jax.random.rademacher(
            probe_key, (flat_base.shape[0], 2), dtype=flat_base.dtype
        )
        probes = jnp.column_stack((delta, new_force, random_probes))
        probes = probes / jnp.maximum(
            jnp.linalg.norm(probes, axis=0, keepdims=True), EPS
        )
        metrics = jax.device_get(
            action_metrics(old_score, mid_score, new_score, batch_score, probes)
        )
        per_old = np.asarray(metrics.pop("per_probe_old_error"))
        per_transport = np.asarray(metrics.pop("per_probe_transport_error"))
        row = {
            "replicate": replicate,
            "actual_delta_norm": scalar(jnp.linalg.norm(delta)),
            "old_raw_variance": scalar(jnp.var(old_raw, ddof=1)),
            "new_same_coordinate_raw_variance": scalar(
                jnp.var(new_raw, ddof=1)
            ),
            "new_batch_raw_variance": scalar(jnp.var(batch_raw, ddof=1)),
            **{name: float(value) for name, value in metrics.items()},
            "per_probe": {
                name: {
                    "old_error": float(per_old[index]),
                    "transport_error": float(per_transport[index]),
                    "transport_error_ratio": float(
                        per_transport[index] / max(per_old[index], EPS)
                    ),
                }
                for index, name in enumerate(probe_names)
            },
        }
        if not all(
            math.isfinite(value)
            for name, value in row.items()
            if isinstance(value, float)
        ):
            raise FloatingPointError(row)
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

        del (
            old_data,
            old_score,
            old_epsilon,
            old_raw,
            updated_params,
            updated_data,
            midpoint_params,
            mid_score,
            new_score,
            new_epsilon,
            new_raw,
            new_force,
            batch_data,
            batch_score,
            batch_raw,
            probes,
            random_probes,
            metrics,
        )
        gc.collect()

    fields = (
        "actual_delta_norm",
        "old_error",
        "transport_error",
        "transport_error_ratio",
        "old_cosine",
        "transport_cosine",
        "sampling_error",
        "sampling_to_staleness_ratio",
        "old_total_batch_error",
        "transport_total_batch_error",
    )
    summaries = {
        field: summarize([row[field] for row in rows]) for field in fields
    }
    transport_ratio = np.asarray(
        [row["transport_error_ratio"] for row in rows], dtype=np.float64
    )
    sampling_ratio = np.asarray(
        [row["sampling_to_staleness_ratio"] for row in rows], dtype=np.float64
    )
    # A useful transport must halve the same-coordinate action error and that
    # corrected error source must not be dwarfed by ordinary batch noise.
    gate = {
        "transport_error_ratio_required_max": 0.5,
        "sampling_to_staleness_ratio_required_max": 2.0,
        "transport_win_fraction_required_min": 0.75,
        "median_transport_error_ratio": float(np.median(transport_ratio)),
        "median_sampling_to_staleness_ratio": float(np.median(sampling_ratio)),
        "transport_win_fraction": float(np.mean(transport_ratio < 1.0)),
        "transport_error_ratio_bootstrap_95_ci": bootstrap_median_ci(
            transport_ratio, 1
        ),
    }
    gate["passed"] = bool(
        gate["median_transport_error_ratio"] <= 0.5
        and gate["median_sampling_to_staleness_ratio"] <= 2.0
        and gate["transport_win_fraction"] >= 0.75
    )
    result = {
        "experiment": "early-C one-step curvature transport action audit",
        "read_only_checkpoint": True,
        "source_checkpoint": str(checkpoint),
        "stored_epoch": int(stored_epoch),
        "replicates": args.replicates,
        "walkers": NCHAINS,
        "transport_definition": "2*S(theta+delta/2)-S(theta)",
        "delta_definition": "actual post-lr/post-norm-constraint parameter update",
        "probe_names": probe_names,
        "summaries": summaries,
        "pre_registered_gate": gate,
        "records": rows,
        "elapsed_seconds": time.perf_counter() - started,
        "git_commit": subprocess.check_output(
            ["git", "-C", "/scratch/dexuan1/vmcnet", "rev-parse", "HEAD"],
            text=True,
        ).strip(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
