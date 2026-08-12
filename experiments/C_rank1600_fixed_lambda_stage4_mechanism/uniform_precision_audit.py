#!/usr/bin/env python3
"""Read-only, uniformly sampled paired-coordinate C fp32/fp64 audit."""

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

from vmcnet.mcmc import metropolis
from vmcnet.mcmc import position_amplitude_core as pacore
from vmcnet.train import runners
from vmcnet.utils import io


NCHAINS = 2000
NBURN = 5000
MEASUREMENT_EPOCHS = 50
MCMC_STEPS = 10
AUDIT_SEED = 20260802
EXPECTED_SAMPLES = NCHAINS * MEASUREMENT_EPOCHS
TRIM_FRACTIONS = (0.0001, 0.001, 0.01)
TAIL_THRESHOLDS = (1.0, 3.0, 10.0)
COMPONENT_NAMES = ("kinetic", "electron_ion", "electron_electron", "ion_ion")


RUNS = {
    "legacy_epoch102000": {
        "run": Path(
            "/scratch/dexuan1/runs/C_wssr_rank1600_hard_matched_control_E5000"
        ),
        "epoch": 102000,
        "reference_statistics": Path(
            "/scratch/dexuan1/runs/C_wssr_rank1600_hard_matched_control_frozen/"
            "epoch102000/eval/statistics.json"
        ),
    },
    "fixed_lambda_1e-3_epoch102000": {
        "run": Path(
            "/scratch/dexuan1/runs/C_wssr_rank1600_fixed_lambda_stage2_E2000"
        ),
        "epoch": 102000,
        "reference_statistics": Path(
            "/scratch/dexuan1/runs/"
            "C_wssr_rank1600_fixed_lambda_stage2_E2000_frozen/"
            "epoch102000/eval/statistics.json"
        ),
    },
}
OUTPUT_ROOT = Path(
    "/scratch/dexuan1/runs/C_wssr_rank1600_uniform_precision_audit_N100k_retry1"
)


def finite_tree(tree) -> bool:
    leaves = jax.tree_util.tree_leaves(tree)
    return all(bool(np.all(np.isfinite(np.asarray(jax.device_get(x))))) for x in leaves)


def cast_inexact_tree(tree, dtype):
    def cast(value):
        value = jnp.asarray(value)
        if jnp.issubdtype(value.dtype, jnp.inexact):
            return value.astype(dtype)
        return value

    return jax.tree_util.tree_map(cast, tree)


def make_component_fn(local_energy_fn):
    terms = local_energy_fn._vmcnet_local_energy_terms

    def evaluate(params, positions):
        return jnp.stack(
            [jax.vmap(term, in_axes=(None, 0))(params, positions) for term in terms],
            axis=0,
        )

    return jax.jit(evaluate)


def scalar(value) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite statistic: {result}")
    return result


def quantiles(values) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    probabilities = (0.0, 0.01, 0.5, 0.9, 0.99, 0.999, 0.9999, 1.0)
    labels = ("min", "p01", "p50", "p90", "p99", "p99_9", "p99_99", "max")
    return {
        label: scalar(value)
        for label, value in zip(labels, np.quantile(values, probabilities))
    }


def correlation(x, y) -> float:
    value = float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])
    return value if math.isfinite(value) else 0.0


def geometry(positions: np.ndarray, ion_positions: np.ndarray) -> dict[str, np.ndarray]:
    electron_ion = np.linalg.norm(
        positions[:, :, None, :] - ion_positions[None, None, :, :], axis=-1
    )
    minimum_ei = np.min(electron_ion, axis=(1, 2))
    displacement = positions[:, :, None, :] - positions[:, None, :, :]
    distances = np.linalg.norm(displacement, axis=-1)
    upper = np.triu_indices(positions.shape[1], 1)
    minimum_ee = np.min(distances[:, upper[0], upper[1]], axis=1)
    return {"minimum_electron_ion": minimum_ei, "minimum_electron_electron": minimum_ee}


def distribution_summary(values: np.ndarray, reference_energy: float) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    mean = scalar(np.mean(values, dtype=np.float64))
    variance = scalar(np.var(values, dtype=np.float64))
    deviation = values - mean
    total_ss = scalar(np.dot(deviation, deviation))
    trimmed = {}
    for fraction in TRIM_FRACTIONS:
        remove = max(1, int(math.floor(fraction * values.size)))
        keep = values.size - remove
        retained_indices = np.argpartition(np.abs(deviation), keep - 1)[:keep]
        retained = values[retained_indices]
        removed_ss = total_ss - scalar(np.dot(deviation[retained_indices], deviation[retained_indices]))
        tail_contribution = removed_ss / total_ss if total_ss > 0.0 else 0.0
        trimmed[f"{100.0 * fraction:.4f}%"] = {
            "removed_count": remove,
            "retained_count": keep,
            "trimmed_mean": scalar(np.mean(retained, dtype=np.float64)),
            "trimmed_variance": scalar(np.var(retained, dtype=np.float64)),
            "removed_tail_variance_contribution": scalar(tail_contribution),
        }
    return {
        "samples": int(values.size),
        "mean": mean,
        "raw_variance": variance,
        "naive_sem": scalar(math.sqrt(variance / values.size)),
        "energy_quantiles": quantiles(values),
        "absolute_deviation_from_own_mean_quantiles": quantiles(np.abs(deviation)),
        "events_over_absolute_deviation_from_own_mean": {
            str(threshold): int(np.count_nonzero(np.abs(deviation) > threshold))
            for threshold in TAIL_THRESHOLDS
        },
        "events_over_absolute_deviation_from_reference_energy": {
            str(threshold): int(
                np.count_nonzero(np.abs(values - reference_energy) > threshold)
            )
            for threshold in TAIL_THRESHOLDS
        },
        "trimmed": trimmed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=sorted(RUNS), required=True)
    args = parser.parse_args()
    spec = RUNS[args.label]
    output = OUTPUT_ROOT / args.label
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)

    run = spec["run"]
    epoch = int(spec["epoch"])
    checkpoint = run / "checkpoints" / f"{epoch}.npz"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    reference = json.loads(spec["reference_statistics"].read_text(encoding="utf-8"))
    reference_energy = float(reference["average"])
    config = io.load_config_dict(str(run), "config.json")
    config.distribute = False
    config.eval.nchains = NCHAINS
    config.eval.nburn = NBURN
    config.eval.nepochs = MEASUREMENT_EPOCHS
    config.eval.nsteps_per_param_update = MCMC_STEPS
    config.eval.nmoves_per_width_update = 100
    config.eval.std_move = 0.25

    stored_epoch, _, params32, _, _ = io.reload_vmc_state(
        str(checkpoint.parent), checkpoint.name
    )
    if int(stored_epoch) != epoch - 1:
        raise ValueError(f"stored epoch {stored_epoch} != expected {epoch - 1}")
    if not finite_tree(params32):
        raise ValueError("checkpoint parameters are nonfinite")

    # Sample the normal inference chain and evaluate production local energies in fp32.
    jax.config.update("jax_enable_x64", False)
    ions32, charges32, nelec = runners._get_electron_ion_config_as_arrays(
        config, jnp.float32
    )
    sampling_key = jax.random.PRNGKey(AUDIT_SEED)
    init_key, model_key = jax.random.split(sampling_key)
    nelec_total = int(jnp.sum(nelec))
    init_key, init_positions = runners.physics.core.initialize_molecular_pos(
        init_key,
        NCHAINS,
        ions32,
        charges32,
        nelec_total,
        dtype=jnp.float32,
    )
    log_psi32, _, _ = runners._get_and_init_model(
        config.model,
        ions32,
        charges32,
        nelec,
        init_positions,
        model_key,
        dtype=jnp.float32,
        apply_pmap=False,
    )
    data = runners._make_initial_data(
        log_psi32,
        config.eval,
        init_positions,
        params32,
        dtype=jnp.float32,
        apply_pmap=False,
    )
    burning_step, walker_fn = runners._get_mcmc_fns(
        config.eval, log_psi32, apply_pmap=False
    )
    local_energy32 = runners._assemble_mol_local_energy_fn(
        ions32,
        charges32,
        config.problem.ei_softening,
        config.problem.ee_softening,
        log_psi32,
    )
    total32_fn = jax.jit(jax.vmap(local_energy32, in_axes=(None, 0), out_axes=0))
    components32_fn = make_component_fn(local_energy32)
    logabs32_fn = jax.jit(jax.vmap(log_psi32, in_axes=(None, 0)))

    print(f"burning {NBURN} fp32 MCMC steps", flush=True)
    started = time.perf_counter()
    data, sampling_key = metropolis.burn_data(
        burning_step, NBURN, params32, data, sampling_key
    )
    burn_seconds = time.perf_counter() - started

    positions_batches = []
    sampling_energy_batches = []
    acceptances = []
    print(
        f"collecting {MEASUREMENT_EPOCHS} complete batches "
        f"({EXPECTED_SAMPLES} uniformly retained coordinates)",
        flush=True,
    )
    started = time.perf_counter()
    for measurement in range(MEASUREMENT_EPOCHS):
        acceptance, data, sampling_key = walker_fn(params32, data, sampling_key)
        positions = pacore.get_position_from_data(data)
        energies = total32_fn(params32, positions)
        positions_batches.append(
            np.asarray(jax.device_get(positions), dtype=np.float32)
        )
        sampling_energy_batches.append(
            np.asarray(jax.device_get(energies), dtype=np.float32)
        )
        acceptances.append(float(jax.device_get(acceptance)))
        if (measurement + 1) % 10 == 0:
            print(f"measurement {measurement + 1}/{MEASUREMENT_EPOCHS}", flush=True)
    sampling_seconds = time.perf_counter() - started
    positions32 = np.stack(positions_batches, axis=0)
    sampling_energy32 = np.stack(sampling_energy_batches, axis=0)
    if positions32.shape[0:2] != (MEASUREMENT_EPOCHS, NCHAINS):
        raise ValueError(f"unexpected coordinate shape {positions32.shape}")

    # Recompute every retained coordinate in the original batch shape in fp32.
    recomputed_energy32 = np.empty((MEASUREMENT_EPOCHS, NCHAINS), dtype=np.float64)
    components32 = np.empty(
        (MEASUREMENT_EPOCHS, len(COMPONENT_NAMES), NCHAINS), dtype=np.float64
    )
    logabs32 = np.empty((MEASUREMENT_EPOCHS, NCHAINS), dtype=np.float64)
    started = time.perf_counter()
    for measurement in range(MEASUREMENT_EPOCHS):
        batch = jnp.asarray(positions32[measurement], dtype=jnp.float32)
        recomputed_energy32[measurement] = np.asarray(
            jax.device_get(total32_fn(params32, batch)), dtype=np.float64
        )
        components32[measurement] = np.asarray(
            jax.device_get(components32_fn(params32, batch)), dtype=np.float64
        )
        logabs32[measurement] = np.asarray(
            jax.device_get(logabs32_fn(params32, batch)), dtype=np.float64
        )
    fp32_recompute_seconds = time.perf_counter() - started

    # Cast parameters and coordinates once per process and evaluate the entire
    # forward/derivative/Laplacian/Hamiltonian path in fp64, preserving batch=2000.
    jax.config.update("jax_enable_x64", True)
    if not jax.config.x64_enabled:
        raise RuntimeError("failed to enable x64")
    ions64, charges64, nelec64 = runners._get_electron_ion_config_as_arrays(
        config, jnp.float64
    )
    params64 = cast_inexact_tree(params32, jnp.float64)
    first_positions64 = jnp.asarray(positions32[0], dtype=jnp.float64)
    log_psi64, _, _ = runners._get_and_init_model(
        config.model,
        ions64,
        charges64,
        nelec64,
        first_positions64,
        jax.random.PRNGKey(AUDIT_SEED + 1),
        dtype=jnp.float64,
        apply_pmap=False,
    )
    local_energy64 = runners._assemble_mol_local_energy_fn(
        ions64,
        charges64,
        config.problem.ei_softening,
        config.problem.ee_softening,
        log_psi64,
    )
    total64_fn = jax.jit(jax.vmap(local_energy64, in_axes=(None, 0), out_axes=0))
    components64_fn = make_component_fn(local_energy64)
    logabs64_fn = jax.jit(jax.vmap(log_psi64, in_axes=(None, 0)))
    recomputed_energy64 = np.empty_like(recomputed_energy32)
    components64 = np.empty_like(components32)
    logabs64 = np.empty_like(logabs32)
    started = time.perf_counter()
    for measurement in range(MEASUREMENT_EPOCHS):
        batch = jnp.asarray(positions32[measurement], dtype=jnp.float64)
        recomputed_energy64[measurement] = np.asarray(
            jax.device_get(total64_fn(params64, batch)), dtype=np.float64
        )
        components64[measurement] = np.asarray(
            jax.device_get(components64_fn(params64, batch)), dtype=np.float64
        )
        logabs64[measurement] = np.asarray(
            jax.device_get(logabs64_fn(params64, batch)), dtype=np.float64
        )
        if (measurement + 1) % 10 == 0:
            print(f"fp64 batch {measurement + 1}/{MEASUREMENT_EPOCHS}", flush=True)
    fp64_seconds = time.perf_counter() - started

    flat_positions = positions32.reshape((-1,) + positions32.shape[2:])
    flat_sampling32 = sampling_energy32.astype(np.float64).reshape(-1)
    flat_energy32 = recomputed_energy32.reshape(-1)
    flat_energy64 = recomputed_energy64.reshape(-1)
    flat_components32 = components32.transpose(1, 0, 2).reshape(
        len(COMPONENT_NAMES), -1
    )
    flat_components64 = components64.transpose(1, 0, 2).reshape(
        len(COMPONENT_NAMES), -1
    )
    flat_logabs32 = logabs32.reshape(-1)
    flat_logabs64 = logabs64.reshape(-1)
    precision_delta = flat_energy64 - flat_energy32
    measurement_mean_delta = np.mean(
        recomputed_energy64 - recomputed_energy32, axis=1, dtype=np.float64
    )
    block_sem = scalar(
        np.std(measurement_mean_delta, ddof=1) / math.sqrt(MEASUREMENT_EPOCHS)
    )
    geometry_data = geometry(flat_positions, np.asarray(ions32))

    component_summary = {}
    for index, name in enumerate(COMPONENT_NAMES):
        delta = flat_components64[index] - flat_components32[index]
        component_summary[name] = {
            "fp32": distribution_summary(flat_components32[index], 0.0),
            "fp64": distribution_summary(flat_components64[index], 0.0),
            "signed_precision_delta": quantiles(delta),
            "absolute_precision_delta": quantiles(np.abs(delta)),
            "mean_signed_precision_delta": scalar(np.mean(delta, dtype=np.float64)),
        }

    paired_thresholds = {}
    for threshold in TAIL_THRESHOLDS:
        fp32_event = np.abs(flat_energy32 - reference_energy) > threshold
        fp64_event = np.abs(flat_energy64 - reference_energy) > threshold
        paired_thresholds[str(threshold)] = {
            "fp32_events": int(np.count_nonzero(fp32_event)),
            "fp64_events": int(np.count_nonzero(fp64_event)),
            "fp32_events_resolved_in_fp64": int(np.count_nonzero(fp32_event & ~fp64_event)),
            "fp64_new_events": int(np.count_nonzero(~fp32_event & fp64_event)),
        }

    component_sum32 = np.sum(
        flat_components32.astype(np.float32), axis=0, dtype=np.float32
    ).astype(np.float64)
    component_sum64 = np.sum(flat_components64, axis=0, dtype=np.float64)
    fp32_summary = distribution_summary(flat_energy32, reference_energy)
    fp64_summary = distribution_summary(flat_energy64, reference_energy)
    payload = {
        "experiment": "uniform same-coordinate C full-path fp32/fp64 audit",
        "label": args.label,
        "source_checkpoint": str(checkpoint),
        "stored_epoch": int(stored_epoch),
        "read_only": True,
        "parameters_updated": False,
        "optimizer_state_updated": False,
        "source_checkpoint_modified": False,
        "sampling_measure": "coordinates sampled from the fp32 wavefunction; both local energies evaluated at identical coordinates",
        "sampling_protocol": {
            "sampling_dtype": "float32",
            "walkers": NCHAINS,
            "burn_in_steps": NBURN,
            "measurement_epochs": MEASUREMENT_EPOCHS,
            "mcmc_steps_per_measurement": MCMC_STEPS,
            "uniformly_retained_coordinates": EXPECTED_SAMPLES,
            "paired_evaluation_batch_size": NCHAINS,
            "batch_shape_matches_sampling": True,
            "audit_seed": AUDIT_SEED,
            "reference_energy_from_40m_frozen_evaluation": reference_energy,
            "mean_acceptance": scalar(np.mean(acceptances)),
        },
        "validity": {
            "sampling_vs_recomputed_fp32_max_abs_difference": scalar(
                np.max(np.abs(flat_sampling32 - flat_energy32))
            ),
            "production_fp32_vs_fp32_component_sum_max_abs_difference": scalar(
                np.max(np.abs(flat_energy32 - component_sum32))
            ),
            "production_fp64_vs_fp64_component_sum_max_abs_difference": scalar(
                np.max(np.abs(flat_energy64 - component_sum64))
            ),
            "all_coordinates_retained_without_energy_based_selection": True,
            "finite": bool(
                np.all(np.isfinite(flat_positions))
                and np.all(np.isfinite(flat_energy32))
                and np.all(np.isfinite(flat_energy64))
                and np.all(np.isfinite(flat_components32))
                and np.all(np.isfinite(flat_components64))
            ),
        },
        "fp32_distribution": fp32_summary,
        "fp64_same_coordinate_distribution": fp64_summary,
        "paired_precision": {
            "mean_signed_local_energy_delta_fp64_minus_fp32": scalar(
                np.mean(precision_delta, dtype=np.float64)
            ),
            "naive_sem_of_mean_delta": scalar(
                np.std(precision_delta, ddof=1) / math.sqrt(precision_delta.size)
            ),
            "measurement_block_sem_of_mean_delta": block_sem,
            "mean_delta_over_block_sem": scalar(
                np.mean(precision_delta, dtype=np.float64) / block_sem
            ),
            "signed_local_energy_precision_delta": quantiles(precision_delta),
            "absolute_local_energy_precision_delta": quantiles(np.abs(precision_delta)),
            "raw_variance_ratio_fp64_over_fp32": scalar(
                fp64_summary["raw_variance"] / fp32_summary["raw_variance"]
            ),
            "raw_variance_change_fraction": scalar(
                fp64_summary["raw_variance"] / fp32_summary["raw_variance"] - 1.0
            ),
            "reference_threshold_event_reclassification": paired_thresholds,
            "components": component_summary,
            "logabs_signed_precision_delta": quantiles(flat_logabs64 - flat_logabs32),
            "logabs_absolute_precision_delta": quantiles(
                np.abs(flat_logabs64 - flat_logabs32)
            ),
            "correlation_abs_fp32_energy_deviation_vs_abs_precision_delta": correlation(
                np.abs(flat_energy32 - reference_energy), np.abs(precision_delta)
            ),
        },
        "geometry": {
            "minimum_electron_ion": quantiles(geometry_data["minimum_electron_ion"]),
            "minimum_electron_electron": quantiles(
                geometry_data["minimum_electron_electron"]
            ),
            "correlation_abs_fp32_energy_deviation_vs_inverse_min_ei": correlation(
                np.abs(flat_energy32 - reference_energy),
                1.0 / np.maximum(geometry_data["minimum_electron_ion"], 1e-12),
            ),
            "correlation_abs_fp32_energy_deviation_vs_inverse_min_ee": correlation(
                np.abs(flat_energy32 - reference_energy),
                1.0 / np.maximum(geometry_data["minimum_electron_electron"], 1e-12),
            ),
        },
        "timing_seconds": {
            "burn_in": burn_seconds,
            "sampling": sampling_seconds,
            "fp32_recomputation_including_compile": fp32_recompute_seconds,
            "fp64_recomputation_including_compile": fp64_seconds,
        },
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    if not payload["validity"]["finite"]:
        raise ValueError("nonfinite result")

    np.savez_compressed(
        output / "uniform_coordinates_and_components.npz",
        positions_fp32=flat_positions,
        sampling_energy_fp32=flat_sampling32,
        recomputed_energy_fp32=flat_energy32,
        recomputed_energy_fp64=flat_energy64,
        components_fp32=flat_components32,
        components_fp64=flat_components64,
        logabs_fp32=flat_logabs32,
        logabs_fp64=flat_logabs64,
        measurement_index=np.repeat(
            np.arange(1, MEASUREMENT_EPOCHS + 1, dtype=np.int32), NCHAINS
        ),
        chain_index=np.tile(np.arange(NCHAINS, dtype=np.int32), MEASUREMENT_EPOCHS),
        minimum_electron_ion=geometry_data["minimum_electron_ion"],
        minimum_electron_electron=geometry_data["minimum_electron_electron"],
    )
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
