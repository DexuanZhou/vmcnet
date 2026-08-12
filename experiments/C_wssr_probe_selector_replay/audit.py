#!/usr/bin/env python3
"""Read-only three-batch cross-fit audit of probe-selected WSSR smoothing."""

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
EPS = 1.0e-12
RHO = 0.9
WARMUP_BATCHES = 20
TRIPLETS = 60
ALPHAS = (0.0, 0.1, 0.25, 0.4)
PROBE_Q_RATIO_MAX = 1.02
PROBE_ALIGNMENT_DROP_MAX = 0.001
NEAR_TIE_Q_FRACTION = 0.001
LOW_FREQUENCY_PERIOD = 20
BLOCK_LENGTH = 5
BOOTSTRAP_REPLICATES = 4000
BOOTSTRAP_SEED = 20260802


def scalar(value) -> float:
    return float(jax.device_get(value))


def alpha_name(alpha: float) -> str:
    return f"alpha_{alpha:.2f}"


def safe_cosine(left, right):
    return jnp.vdot(left, right) / jnp.maximum(
        jnp.linalg.norm(left) * jnp.linalg.norm(right), EPS
    )


def normalize(vector):
    return vector / jnp.maximum(jnp.linalg.norm(vector), EPS)


def residual_ratio(target, prediction):
    return jnp.linalg.norm(target - prediction) / jnp.maximum(
        jnp.linalg.norm(target), EPS
    )


def circular_indices(length, rng):
    count = math.ceil(length / BLOCK_LENGTH)
    starts = rng.integers(0, length, size=count)
    offsets = np.arange(BLOCK_LENGTH)
    return ((starts[:, None] + offsets[None, :]) % length).reshape(-1)[:length]


def bootstrap_ci(values, *, seed_offset):
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


def select_from_probe(probe_metrics):
    native = probe_metrics[alpha_name(0.0)]
    feasible = []
    for alpha in ALPHAS:
        name = alpha_name(alpha)
        metrics = probe_metrics[name]
        if (
            metrics["q"] / native["q"] <= PROBE_Q_RATIO_MAX
            and metrics["alignment"] >= native["alignment"] - PROBE_ALIGNMENT_DROP_MAX
            and metrics["force_dot"] > 0.0
        ):
            feasible.append((alpha, name, metrics["q"]))
    if not feasible:
        return alpha_name(0.0), {
            "fallback_to_native": True,
            "feasible": [],
            "q_min": native["q"],
        }
    q_min = min(item[2] for item in feasible)
    near = [
        item
        for item in feasible
        if item[2] <= q_min + NEAR_TIE_Q_FRACTION * native["q"]
    ]
    selected = min(near, key=lambda item: item[0])
    return selected[1], {
        "fallback_to_native": False,
        "feasible": [name for _, name, _ in feasible],
        "q_min": q_min,
    }


def aggregate_strategy(records, names, label, *, require_stage1=True, seed_base=0):
    if len(names) != len(records):
        raise ValueError((len(names), len(records)))
    native = alpha_name(0.0)
    coherence_delta = []
    for index in range(1, len(records)):
        current_name = names[index]
        previous_name = names[index - 1]
        selected_cosine = records[index]["cross_consecutive_cosines"][current_name][
            previous_name
        ]
        native_cosine = records[index]["cross_consecutive_cosines"][native][native]
        coherence_delta.append(selected_cosine - native_cosine)
    alignment_delta = [
        record["final_metrics"][name]["alignment"]
        - record["final_metrics"][native]["alignment"]
        for record, name in zip(records, names)
    ]
    q_ratio = [
        record["final_metrics"][name]["q"]
        / record["final_metrics"][native]["q"]
        for record, name in zip(records, names)
    ]
    native_positive = float(
        np.mean(
            [record["final_metrics"][native]["force_dot"] > 0.0 for record in records]
        )
    )
    selected_positive = float(
        np.mean(
            [
                record["final_metrics"][name]["force_dot"] > 0.0
                for record, name in zip(records, names)
            ]
        )
    )
    nonzero_fraction = float(np.mean([name != native for name in names]))
    coherence_ci = bootstrap_ci(coherence_delta, seed_offset=seed_base + 1)
    alignment_ci = bootstrap_ci(alignment_delta, seed_offset=seed_base + 2)
    q_ratio_ci = bootstrap_ci(q_ratio, seed_offset=seed_base + 3)
    med_coherence = float(np.median(coherence_delta))
    med_alignment = float(np.median(alignment_delta))
    med_q_ratio = float(np.median(q_ratio))
    individual_gates = {
        "coherence": med_coherence >= 0.15 and coherence_ci[0] >= 0.10,
        "alignment": med_alignment >= -0.001 and alignment_ci[0] >= -0.002,
        "residual": med_q_ratio <= 1.02 and q_ratio_ci[1] <= 1.05,
        "positive_force": native_positive - selected_positive <= 0.05,
        "nonzero_selection": 0.20 <= nonzero_fraction <= 0.80,
    }
    passed = all(individual_gates.values()) and require_stage1
    return {
        "label": label,
        "selected_names": names,
        "selection_counts": {
            name: names.count(name) for name in [alpha_name(alpha) for alpha in ALPHAS]
        },
        "nonzero_selection_fraction": nonzero_fraction,
        "median_consecutive_cosine_improvement": med_coherence,
        "block_bootstrap_95_ci_consecutive_improvement": coherence_ci,
        "median_final_alignment_difference": med_alignment,
        "block_bootstrap_95_ci_alignment_difference": alignment_ci,
        "median_final_q_ratio": med_q_ratio,
        "block_bootstrap_95_ci_q_ratio": q_ratio_ci,
        "native_positive_final_force_fraction": native_positive,
        "selected_positive_final_force_fraction": selected_positive,
        "positive_force_fraction_drop": native_positive - selected_positive,
        "individual_gates": individual_gates,
        "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
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
    if config.vmc.optimizer_type != "wssr_warm_svd_right":
        raise ValueError(config.vmc.optimizer_type)
    opt = config.vmc.optimizer.wssr_warm_svd_right
    expected = {
        "nchains": (int(config.vmc.nchains), NCHAINS),
        "rank": (int(opt.sr_rank), 1600),
        "rank_max": (int(opt.sr_rank_max), 1600),
        "eta": (float(opt.eta), 0.3),
        "learning_rate": (float(opt.learning_rate), 0.04),
        "tikhonov_lambda": (float(opt.tikhonov_lambda), 1.0e-3),
        "warm_iterations": (int(opt.svd_maxiter_warm), 2),
        "complement_weight": (float(opt.complement_weight), 0.0),
    }
    mismatches = {
        name: pair for name, pair in expected.items() if pair[0] != pair[1]
    }
    if mismatches:
        raise ValueError(f"unexpected source configuration: {mismatches}")

    dtype = runners._get_dtype(config)
    ions, charges, nelec = runners._get_electron_ion_config_as_arrays(config, dtype)
    positions = pacore.get_position_from_data(data)
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
    working_rank = min(
        int(opt.svd_working_rank), int(opt.sr_rank_max), state.sr_o.shape[1]
    )

    def native_direction(score, epsilon, direction_key, state_arg):
        augmented, rhs = wssr.augment_wssr_system(
            score, epsilon, state_arg, float(opt.eta)
        )
        u, singular_values, vh, _ = wssr._wssr_right_svd_decomposition(
            augmented,
            state_arg,
            direction_key,
            working_rank,
            int(opt.svd_maxiter_initial),
            int(opt.svd_maxiter_warm),
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
            int(opt.sr_rank_max),
            float(opt.sr_scale),
            False,
            rank_update_max=working_rank,
            spectral_regularization=str(opt.spectral_regularization),
            complement_weight=0.0,
            relative_singular_value_cutoff=float(opt.relative_singular_value_cutoff),
            tikhonov_lambda=float(opt.tikhonov_lambda),
            eps=EPS,
        )
        return result.grad_like_update

    native_jit = jax.jit(native_direction)

    @jax.jit
    def blend(direction, consensus, alpha):
        norm = jnp.linalg.norm(direction)
        mixed = normalize((1.0 - alpha) * normalize(direction) + alpha * consensus)
        return norm * mixed

    def sample_batch(data_arg, key_arg):
        acceptance, data_arg, key_arg = walker_fn(params, data_arg, key_arg)
        positions_arg = pacore.get_position_from_data(data_arg)
        energy, clipped, stats = energy_fn(params, positions_arg)
        score, _ = wssr.center_and_scale_score_matrix(
            log_psi, params, positions_arg
        )
        epsilon = wssr.center_and_scale_energy_residuals(clipped, energy)
        force = score @ epsilon
        return data_arg, key_arg, acceptance, score, epsilon, force, stats

    def make_native(score, epsilon, key_arg):
        key_arg, direction_key = jax.random.split(key_arg)
        begin = time.perf_counter()
        direction = native_jit(score, epsilon, direction_key, state)
        jax.block_until_ready(direction)
        return direction, key_arg, time.perf_counter() - begin

    def evaluate(score, epsilon, force, candidate):
        return {
            "alignment": scalar(safe_cosine(force, candidate)),
            "force_dot": scalar(jnp.vdot(force, candidate)),
            "q": scalar(residual_ratio(epsilon, score.T @ candidate)),
        }

    started = time.perf_counter()
    consensus = None
    solve_seconds = []
    candidate_seconds = []
    for warmup in range(WARMUP_BATCHES):
        data, key, _, score, epsilon, _, _ = sample_batch(data, key)
        direction, key, seconds = make_native(score, epsilon, key)
        solve_seconds.append(seconds)
        unit = normalize(direction)
        consensus = unit if consensus is None else normalize(
            RHO * consensus + (1.0 - RHO) * unit
        )
        print(json.dumps({"phase": "warmup", "batch": warmup}), flush=True)

    pending = []
    records = []
    previous_grid = None
    previous_selected = None
    names_grid = [alpha_name(alpha) for alpha in ALPHAS]
    for batch in range(TRIPLETS + 2):
        data, key, acceptance, score, epsilon, force, stats = sample_batch(data, key)

        completed = []
        for item in pending:
            age = batch - item["origin"]
            if age == 1:
                item["probe_metrics"] = {
                    name: evaluate(score, epsilon, force, candidate)
                    for name, candidate in item["candidates"].items()
                }
                selected_name, selection = select_from_probe(item["probe_metrics"])
                item["selected_name"] = selected_name
                item["selection"] = selection
                selected = item["candidates"][selected_name]
                item["selected_consecutive_cosine"] = (
                    scalar(safe_cosine(previous_selected, selected))
                    if previous_selected is not None
                    else None
                )
                previous_selected = selected
            elif age == 2:
                item["final_metrics"] = {
                    name: evaluate(score, epsilon, force, candidate)
                    for name, candidate in item["candidates"].items()
                }
                record = {
                    key_: value
                    for key_, value in item.items()
                    if key_ != "candidates"
                }
                record.update(
                    {
                        "final_acceptance": scalar(acceptance),
                        "final_raw_variance": scalar(
                            jnp.var(stats["local_energies_noclip"], ddof=1)
                        ),
                    }
                )
                records.append(record)
                print(json.dumps({"phase": "triplet", **record}, sort_keys=True), flush=True)
                completed.append(item)
        for item in completed:
            pending.remove(item)

        if batch < TRIPLETS:
            direction, key, seconds = make_native(score, epsilon, key)
            solve_seconds.append(seconds)
            candidate_begin = time.perf_counter()
            candidates = {alpha_name(0.0): direction}
            for alpha in ALPHAS[1:]:
                candidates[alpha_name(alpha)] = blend(direction, consensus, alpha)
            jax.block_until_ready(candidates[alpha_name(ALPHAS[-1])])
            candidate_seconds.append(time.perf_counter() - candidate_begin)
            cross = {}
            if previous_grid is not None:
                for current_name, current in candidates.items():
                    cross[current_name] = {
                        previous_name: scalar(safe_cosine(current, previous))
                        for previous_name, previous in previous_grid.items()
                    }
            item = {
                "origin": batch,
                "origin_acceptance": scalar(acceptance),
                "origin_raw_variance": scalar(
                    jnp.var(stats["local_energies_noclip"], ddof=1)
                ),
                "direction_norm": scalar(jnp.linalg.norm(direction)),
                "cross_consecutive_cosines": cross,
                "candidates": candidates,
            }
            pending.append(item)
            previous_grid = candidates
            consensus = normalize(RHO * consensus + (1.0 - RHO) * normalize(direction))

    if len(records) != TRIPLETS or pending:
        raise RuntimeError((len(records), len(pending)))

    stepwise_names = [record["selected_name"] for record in records]
    stage1 = aggregate_strategy(
        records, stepwise_names, "per-triplet held-out probe selector", seed_base=0
    )

    low_frequency_names = []
    refresh_decisions = []
    for start in range(0, TRIPLETS, LOW_FREQUENCY_PERIOD):
        selected_name, selection = select_from_probe(records[start]["probe_metrics"])
        stop = min(start + LOW_FREQUENCY_PERIOD, TRIPLETS)
        low_frequency_names.extend([selected_name] * (stop - start))
        refresh_decisions.append(
            {
                "start_origin": start,
                "stop_origin_exclusive": stop,
                "selected_name": selected_name,
                "selection": selection,
            }
        )
    stage2 = aggregate_strategy(
        records,
        low_frequency_names,
        "refresh alpha every 20 WSSR steps",
        require_stage1=stage1["passed"],
        seed_base=100,
    )
    stage2["refresh_decisions"] = refresh_decisions
    stage2["eligible_only_if_stage1_passed"] = True

    median_solve = float(np.median(solve_seconds[1:]))
    median_candidate = float(np.median(candidate_seconds[1:]))
    payload = {
        "experiment": "C rank1600 WSSR three-batch held-out probe selector",
        "read_only_checkpoint": True,
        "parameters_updated": False,
        "optimizer_state_updated": False,
        "source_checkpoint": str(checkpoint),
        "stored_epoch": int(stored_epoch),
        "config_verified": {name: actual for name, (actual, _) in expected.items()},
        "warmup_batches": WARMUP_BATCHES,
        "triplets": TRIPLETS,
        "walkers_per_batch": NCHAINS,
        "mcmc_steps_between_batches": int(config.vmc.nsteps_per_param_update),
        "consensus_rho": RHO,
        "alphas": ALPHAS,
        "probe_selector": {
            "q_ratio_max": PROBE_Q_RATIO_MAX,
            "alignment_drop_max": PROBE_ALIGNMENT_DROP_MAX,
            "force_dot_must_be_positive": True,
            "near_tie_q_fraction": NEAR_TIE_Q_FRACTION,
            "near_tie_prefers_smaller_alpha": True,
        },
        "pre_registered_final_gates": {
            "median_consecutive_cosine_improvement_min": 0.15,
            "coherence_ci_lower_min": 0.10,
            "median_alignment_difference_min": -0.001,
            "alignment_ci_lower_min": -0.002,
            "median_q_ratio_max": 1.02,
            "q_ratio_ci_upper_max": 1.05,
            "positive_force_fraction_drop_max": 0.05,
            "nonzero_selection_fraction_range": [0.20, 0.80],
        },
        "stage1_stepwise_crossfit": stage1,
        "stage2_low_frequency_emulation": stage2,
        "timing": {
            "median_native_wssr_solve_seconds": median_solve,
            "median_three_blend_vector_ops_seconds": median_candidate,
            "blend_overhead_fraction": median_candidate / max(median_solve, EPS),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "decision": {
            "stage1_passed": stage1["passed"],
            "stage2_passed": stage2["passed"],
            "short_training_authorized": stage1["passed"] and stage2["passed"],
            "conclusion": (
                "Both replay stages passed; a matched E500 is authorized."
                if stage1["passed"] and stage2["passed"]
                else (
                    "Stepwise probe selection generalized, but a 20-step refresh did not; "
                    "the selector is not deployable at low overhead."
                    if stage1["passed"]
                    else "An independent probe batch could not select temporally smoothed "
                    "directions that pass the next-batch safeguards; stop without training."
                )
            ),
        },
        "records": records,
        "finite": True,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
