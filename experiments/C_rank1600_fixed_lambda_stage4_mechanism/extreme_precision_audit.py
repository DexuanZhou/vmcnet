#!/usr/bin/env python3
"""Read-only paired-coordinate fp32/fp64 audit of rare C local-energy tails."""

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
MEASUREMENT_EPOCHS = 2000
MCMC_STEPS = 10
TOP_PER_EPOCH = 32
TOP_FINAL = 1000
AUDIT_SEED = 20260801
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
    "/scratch/dexuan1/runs/"
    "C_wssr_rank1600_fixed_lambda_stage4_extreme_precision_retry2"
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


def quantiles(values) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    probs = (0.0, 0.5, 0.9, 0.99, 1.0)
    return {
        label: float(value)
        for label, value in zip(
            ("min", "p50", "p90", "p99", "max"), np.quantile(values, probs)
        )
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

    # Reproduce normal inference sampling entirely in fp32.
    jax.config.update("jax_enable_x64", False)
    dtype32 = jnp.float32
    ions32, charges32, nelec = runners._get_electron_ion_config_as_arrays(
        config, dtype32
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
        dtype=dtype32,
    )
    log_psi32, _, _ = runners._get_and_init_model(
        config.model,
        ions32,
        charges32,
        nelec,
        init_positions,
        model_key,
        dtype=dtype32,
        apply_pmap=False,
    )
    data = runners._make_initial_data(
        log_psi32,
        config.eval,
        init_positions,
        params32,
        dtype=dtype32,
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
    components32_fn = make_component_fn(local_energy32)
    # Use the production sequential local-energy path for event selection.  The
    # separately evaluated components are diagnostic only: summing already-rounded
    # fp32 components in fp64 is not equivalent in severe cancellation regions.
    total32_fn = jax.jit(
        jax.vmap(local_energy32, in_axes=(None, 0), out_axes=0)
    )

    print(f"burning {NBURN} fp32 MCMC steps", flush=True)
    started = time.perf_counter()
    data, sampling_key = metropolis.burn_data(
        burning_step, NBURN, params32, data, sampling_key
    )
    burn_seconds = time.perf_counter() - started

    candidate_deviation = []
    candidate_energy = []
    candidate_position = []
    candidate_iteration = []
    candidate_chain = []
    acceptances = []
    sample_count = 0
    sample_sum = 0.0
    sample_sum_square = 0.0
    sample_min = math.inf
    sample_max = -math.inf
    threshold_counts = {1.0: 0, 3.0: 0, 10.0: 0}
    print(f"sampling {MEASUREMENT_EPOCHS} measurement epochs", flush=True)
    started = time.perf_counter()
    for measurement in range(1, MEASUREMENT_EPOCHS + 1):
        acceptance, data, sampling_key = walker_fn(
            params32, data, sampling_key
        )
        positions = pacore.get_position_from_data(data)
        energies = total32_fn(params32, positions)
        deviation = jnp.abs(energies - jnp.asarray(reference_energy, energies.dtype))
        _, indices = jax.lax.top_k(deviation, TOP_PER_EPOCH)
        host_energy = np.asarray(jax.device_get(energies), dtype=np.float64)
        host_indices = np.asarray(jax.device_get(indices), dtype=np.int64)
        host_positions = np.asarray(jax.device_get(positions[indices]), dtype=np.float32)
        host_deviation = np.abs(host_energy - reference_energy)
        candidate_deviation.append(host_deviation[host_indices])
        candidate_energy.append(host_energy[host_indices])
        candidate_position.append(host_positions)
        candidate_iteration.append(np.full(TOP_PER_EPOCH, measurement, dtype=np.int32))
        candidate_chain.append(host_indices.astype(np.int32))
        acceptances.append(float(jax.device_get(acceptance)))
        sample_count += host_energy.size
        sample_sum += float(np.sum(host_energy, dtype=np.float64))
        sample_sum_square += float(np.dot(host_energy, host_energy))
        sample_min = min(sample_min, float(np.min(host_energy)))
        sample_max = max(sample_max, float(np.max(host_energy)))
        for threshold in threshold_counts:
            threshold_counts[threshold] += int(np.count_nonzero(host_deviation > threshold))
        if measurement % 100 == 0:
            print(
                f"measurement {measurement}/{MEASUREMENT_EPOCHS}, "
                f"running extrema=({sample_min:.6f}, {sample_max:.6f})",
                flush=True,
            )
    sampling_seconds = time.perf_counter() - started

    all_deviation = np.concatenate(candidate_deviation)
    all_energy = np.concatenate(candidate_energy)
    all_positions = np.concatenate(candidate_position)
    all_iterations = np.concatenate(candidate_iteration)
    all_chains = np.concatenate(candidate_chain)
    selected = np.argsort(all_deviation)[-TOP_FINAL:][::-1]
    extreme_positions32 = all_positions[selected]
    sampling_energy32 = all_energy[selected]
    extreme_iterations = all_iterations[selected]
    extreme_chains = all_chains[selected]

    # Preserve the production sampling batch shape.  In fp32, XLA may choose a
    # different derivative/Laplacian implementation for batch=1000 versus
    # batch=2000, which is itself enough to move cancellation-sensitive local
    # energies.  Duplicate the selected coordinates to the original batch size and
    # retain only the first TOP_FINAL outputs.
    paired_positions32 = np.concatenate(
        (extreme_positions32, extreme_positions32), axis=0
    )
    if paired_positions32.shape[0] != NCHAINS:
        raise ValueError(f"paired evaluation batch has shape {paired_positions32.shape}")

    # Re-evaluate the exact same selected coordinates and Hamiltonian terms in fp32.
    components32 = np.asarray(
        jax.device_get(components32_fn(params32, jnp.asarray(paired_positions32))),
        dtype=np.float64,
    )[:, :TOP_FINAL]
    recomputed_energy32 = np.asarray(
        jax.device_get(total32_fn(params32, jnp.asarray(paired_positions32))),
        dtype=np.float64,
    )[:TOP_FINAL]
    component_sum32 = np.sum(
        components32.astype(np.float32), axis=0, dtype=np.float32
    ).astype(np.float64)

    # Build the same architecture in fp64, cast parameters and coordinates once, and
    # run the entire forward/derivative/Laplacian/Hamiltonian path in fp64.
    jax.config.update("jax_enable_x64", True)
    if not jax.config.x64_enabled:
        raise RuntimeError("failed to enable x64")
    ions64, charges64, nelec64 = runners._get_electron_ion_config_as_arrays(
        config, jnp.float64
    )
    paired_positions64 = jnp.asarray(paired_positions32, dtype=jnp.float64)
    params64 = cast_inexact_tree(params32, jnp.float64)
    log_psi64, _, _ = runners._get_and_init_model(
        config.model,
        ions64,
        charges64,
        nelec64,
        paired_positions64,
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
    components64_fn = make_component_fn(local_energy64)
    total64_fn = jax.jit(
        jax.vmap(local_energy64, in_axes=(None, 0), out_axes=0)
    )
    started = time.perf_counter()
    recomputed_energy64 = np.asarray(
        jax.device_get(total64_fn(params64, paired_positions64)), dtype=np.float64
    )[:TOP_FINAL]
    components64 = np.asarray(
        jax.device_get(components64_fn(params64, paired_positions64)),
        dtype=np.float64,
    )[:, :TOP_FINAL]
    fp64_seconds = time.perf_counter() - started
    component_sum64 = np.sum(components64, axis=0, dtype=np.float64)
    logabs32 = np.asarray(
        jax.device_get(jax.jit(jax.vmap(log_psi32, in_axes=(None, 0)))(
            params32, jnp.asarray(paired_positions32)
        )),
        dtype=np.float64,
    )[:TOP_FINAL]
    logabs64 = np.asarray(
        jax.device_get(jax.jit(jax.vmap(log_psi64, in_axes=(None, 0)))(
            params64, paired_positions64
        )),
        dtype=np.float64,
    )[:TOP_FINAL]

    precision_delta = recomputed_energy64 - recomputed_energy32
    geometry_data = geometry(extreme_positions32, np.asarray(ions32))
    sample_mean = sample_sum / sample_count
    sample_variance = sample_sum_square / sample_count - sample_mean**2
    event_thresholds = {}
    for threshold in (1.0, 3.0, 10.0):
        fp32_event = np.abs(recomputed_energy32 - reference_energy) > threshold
        fp64_event = np.abs(recomputed_energy64 - reference_energy) > threshold
        event_thresholds[str(threshold)] = {
            "fp32_selected_events": int(np.count_nonzero(fp32_event)),
            "fp64_selected_events": int(np.count_nonzero(fp64_event)),
            "fp32_events_resolved_in_fp64": int(np.count_nonzero(fp32_event & ~fp64_event)),
        }

    component_summary = {}
    for index, name in enumerate(COMPONENT_NAMES):
        delta = components64[index] - components32[index]
        component_summary[name] = {
            "fp32_range": [float(np.min(components32[index])), float(np.max(components32[index]))],
            "fp64_range": [float(np.min(components64[index])), float(np.max(components64[index]))],
            "absolute_precision_delta": quantiles(np.abs(delta)),
        }

    np.savez_compressed(
        output / "extreme_coordinates_and_components.npz",
        positions_fp32=extreme_positions32,
        sampling_energy_fp32=sampling_energy32,
        recomputed_energy_fp32=recomputed_energy32,
        recomputed_energy_fp64=recomputed_energy64,
        components_fp32=components32,
        components_fp64=components64,
        component_sum_fp32=component_sum32,
        component_sum_fp64=component_sum64,
        measurement_iteration=extreme_iterations,
        chain_index=extreme_chains,
        logabs_fp32=logabs32,
        logabs_fp64=logabs64,
        minimum_electron_ion=geometry_data["minimum_electron_ion"],
        minimum_electron_electron=geometry_data["minimum_electron_electron"],
    )
    payload = {
        "experiment": "paired-coordinate C extreme local-energy fp32/fp64 audit",
        "label": args.label,
        "source_checkpoint": str(checkpoint),
        "stored_epoch": int(stored_epoch),
        "read_only": True,
        "parameters_updated": False,
        "optimizer_state_updated": False,
        "source_checkpoint_modified": False,
        "sampling_protocol": {
            "dtype": "float32",
            "walkers": NCHAINS,
            "burn_in_steps": NBURN,
            "measurement_epochs": MEASUREMENT_EPOCHS,
            "mcmc_steps_per_measurement": MCMC_STEPS,
            "total_local_energy_samples": sample_count,
            "audit_seed": AUDIT_SEED,
            "reference_energy": reference_energy,
            "mean_acceptance": float(np.mean(acceptances)),
            "sample_mean": sample_mean,
            "sample_raw_variance": sample_variance,
            "sample_minimum": sample_min,
            "sample_maximum": sample_max,
            "events_over_absolute_deviation_threshold": {
                str(key): value for key, value in threshold_counts.items()
            },
        },
        "selection": {
            "candidates_per_measurement": TOP_PER_EPOCH,
            "final_extreme_coordinates": TOP_FINAL,
            "paired_evaluation_batch_size": NCHAINS,
            "batch_shape_matches_sampling": True,
            "criterion": "absolute fp32 local-energy deviation from reference frozen mean",
        },
        "paired_precision": {
            "sampling_vs_recomputed_fp32_max_abs_difference": float(
                np.max(np.abs(sampling_energy32 - recomputed_energy32))
            ),
            "production_fp32_vs_fp32_component_sum_max_abs_difference": float(
                np.max(np.abs(recomputed_energy32 - component_sum32))
            ),
            "production_fp64_vs_fp64_component_sum_max_abs_difference": float(
                np.max(np.abs(recomputed_energy64 - component_sum64))
            ),
            "local_energy_fp32_range": [
                float(np.min(recomputed_energy32)), float(np.max(recomputed_energy32))
            ],
            "local_energy_fp64_range": [
                float(np.min(recomputed_energy64)), float(np.max(recomputed_energy64))
            ],
            "absolute_local_energy_precision_delta": quantiles(np.abs(precision_delta)),
            "signed_local_energy_precision_delta": quantiles(precision_delta),
            "correlation_abs_fp32_energy_deviation_vs_abs_precision_delta": correlation(
                np.abs(recomputed_energy32 - reference_energy), np.abs(precision_delta)
            ),
            "event_thresholds": event_thresholds,
            "components": component_summary,
            "logabs_absolute_precision_delta": quantiles(np.abs(logabs64 - logabs32)),
        },
        "geometry": {
            "minimum_electron_ion": quantiles(geometry_data["minimum_electron_ion"]),
            "minimum_electron_electron": quantiles(
                geometry_data["minimum_electron_electron"]
            ),
            "correlation_abs_fp32_energy_deviation_vs_inverse_min_ei": correlation(
                np.abs(recomputed_energy32 - reference_energy),
                1.0 / np.maximum(geometry_data["minimum_electron_ion"], 1e-12),
            ),
            "correlation_abs_fp32_energy_deviation_vs_inverse_min_ee": correlation(
                np.abs(recomputed_energy32 - reference_energy),
                1.0 / np.maximum(geometry_data["minimum_electron_electron"], 1e-12),
            ),
        },
        "timing_seconds": {
            "burn_in": burn_seconds,
            "sampling": sampling_seconds,
            "fp64_extreme_evaluation_including_compile": fp64_seconds,
        },
        "finite": bool(
            np.all(np.isfinite(components32))
            and np.all(np.isfinite(components64))
            and np.all(np.isfinite(extreme_positions32))
        ),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
