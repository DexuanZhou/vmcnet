#!/usr/bin/env python3
"""Read-only validation of a WSSR temporal-consistency indicator.

The checkpoint parameters and optimizer state remain fixed.  Native WSSR directions
are constructed on batch k and evaluated on independently advanced batch k+1.  A
short-term consensus is formed only from *native* WSSR directions, so candidate
filtering cannot feed back into its own indicator.
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
EPS = 1.0e-12
RHO = 0.9
CALIBRATION_TRANSITIONS = 40
VALIDATION_TRANSITIONS = 80
GMAX_GRID = (0.25, 0.5)
BLOCK_LENGTH = 5
BOOTSTRAP_REPLICATES = 4000
BOOTSTRAP_SEED = 20260802


def scalar(value) -> float:
    return float(jax.device_get(value))


def safe_cosine(left, right):
    return jnp.vdot(left, right) / jnp.maximum(
        jnp.linalg.norm(left) * jnp.linalg.norm(right), EPS
    )


def residual_ratio(target, prediction):
    return jnp.linalg.norm(target - prediction) / jnp.maximum(
        jnp.linalg.norm(target), EPS
    )


def normalize(vector):
    return vector / jnp.maximum(jnp.linalg.norm(vector), EPS)


def rankdata(values: np.ndarray) -> np.ndarray:
    """Return average ranks, matching scipy.stats.rankdata(method='average')."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman(left, right) -> float:
    left_rank = rankdata(np.asarray(left, dtype=np.float64))
    right_rank = rankdata(np.asarray(right, dtype=np.float64))
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return math.nan
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def circular_block_indices(length: int, rng: np.random.Generator) -> np.ndarray:
    nblocks = math.ceil(length / BLOCK_LENGTH)
    starts = rng.integers(0, length, size=nblocks)
    offsets = np.arange(BLOCK_LENGTH)
    return ((starts[:, None] + offsets[None, :]) % length).reshape(-1)[:length]


def block_bootstrap_stat(records, statistic, *, seed_offset=0):
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    values = []
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = circular_block_indices(len(records), rng)
        value = statistic([records[index] for index in indices])
        if math.isfinite(value):
            values.append(value)
    if len(values) < BOOTSTRAP_REPLICATES // 2:
        raise RuntimeError("too few finite block-bootstrap replicates")
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def candidate_name(gmax: float) -> str:
    return f"gated_gmax_{gmax:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
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

    native_direction_jit = jax.jit(native_direction)

    @jax.jit
    def blend_direction(direction, consensus, gate):
        norm = jnp.linalg.norm(direction)
        unit = direction / jnp.maximum(norm, EPS)
        mixed = normalize((1.0 - gate) * unit + gate * consensus)
        return norm * mixed

    def sample_batch(data_arg, key_arg):
        acceptance, data_arg, key_arg = walker_fn(params, data_arg, key_arg)
        positions_arg = pacore.get_position_from_data(data_arg)
        energy, clipped, stats = energy_fn(params, positions_arg)
        score, _ = wssr.center_and_scale_score_matrix(
            log_psi, params, positions_arg
        )
        epsilon = wssr.center_and_scale_energy_residuals(clipped, energy)
        return data_arg, key_arg, acceptance, score, epsilon, stats

    def make_native(score, epsilon, key_arg):
        key_arg, direction_key = jax.random.split(key_arg)
        started = time.perf_counter()
        direction = native_direction_jit(score, epsilon, direction_key, state)
        jax.block_until_ready(direction)
        return direction, key_arg, time.perf_counter() - started

    # One independent batch initializes the native-only consensus.  It is excluded
    # from both calibration and validation statistics.
    started = time.perf_counter()
    data, key, _, score, epsilon, _ = sample_batch(data, key)
    initial, key, initial_solve_seconds = make_native(score, epsilon, key)
    consensus = normalize(initial)

    calibration = []
    solve_seconds = [initial_solve_seconds]
    data, key, acceptance, score, epsilon, stats = sample_batch(data, key)
    for index in range(CALIBRATION_TRANSITIONS):
        direction, key, seconds = make_native(score, epsilon, key)
        solve_seconds.append(seconds)
        unit = normalize(direction)
        indicator = scalar(safe_cosine(unit, consensus))
        force = score @ epsilon

        data, key, next_acceptance, next_score, next_epsilon, next_stats = sample_batch(
            data, key
        )
        next_force = next_score @ next_epsilon
        row = {
            "transition": index,
            "indicator": indicator,
            "acceptance": scalar(acceptance),
            "next_acceptance": scalar(next_acceptance),
            "raw_variance": scalar(
                jnp.var(stats["local_energies_noclip"], ddof=1)
            ),
            "next_raw_variance": scalar(
                jnp.var(next_stats["local_energies_noclip"], ddof=1)
            ),
            "native_current_alignment": scalar(safe_cosine(force, direction)),
            "native_next_alignment": scalar(safe_cosine(next_force, direction)),
            "native_next_q": scalar(
                residual_ratio(next_epsilon, next_score.T @ direction)
            ),
        }
        calibration.append(row)
        print(json.dumps({"phase": "calibration", **row}, sort_keys=True), flush=True)
        consensus = normalize(RHO * consensus + (1.0 - RHO) * unit)
        acceptance, score, epsilon, stats = (
            next_acceptance,
            next_score,
            next_epsilon,
            next_stats,
        )

    calibration_indicators = np.asarray(
        [row["indicator"] for row in calibration], dtype=np.float64
    )
    c_bad = float(np.quantile(calibration_indicators, 0.20))
    c_on = float(np.quantile(calibration_indicators, 0.50))
    if not c_on > c_bad:
        raise RuntimeError(f"degenerate calibration thresholds: {c_bad}, {c_on}")

    validation = []
    previous_candidates = {
        "native": None,
        "always_blend_0.25": None,
        **{candidate_name(gmax): None for gmax in GMAX_GRID},
    }
    candidate_seconds = []
    for index in range(VALIDATION_TRANSITIONS):
        direction, key, seconds = make_native(score, epsilon, key)
        solve_seconds.append(seconds)
        unit = normalize(direction)
        indicator = scalar(safe_cosine(unit, consensus))
        force = score @ epsilon
        gate_base = float(np.clip((c_on - indicator) / (c_on - c_bad), 0.0, 1.0))

        candidate_started = time.perf_counter()
        candidates = {
            "native": direction,
            "always_blend_0.25": blend_direction(direction, consensus, 0.25),
        }
        gates = {"native": 0.0, "always_blend_0.25": 0.25}
        for gmax in GMAX_GRID:
            name = candidate_name(gmax)
            gate = gmax * gate_base
            candidates[name] = blend_direction(direction, consensus, gate)
            gates[name] = gate
        jax.block_until_ready(candidates[candidate_name(GMAX_GRID[-1])])
        candidate_seconds.append(time.perf_counter() - candidate_started)

        data, key, next_acceptance, next_score, next_epsilon, next_stats = sample_batch(
            data, key
        )
        next_force = next_score @ next_epsilon
        candidate_metrics = {}
        for name, candidate in candidates.items():
            previous = previous_candidates[name]
            candidate_metrics[name] = {
                "gate": gates[name],
                "direction_norm": scalar(jnp.linalg.norm(candidate)),
                "current_alignment": scalar(safe_cosine(force, candidate)),
                "next_alignment": scalar(safe_cosine(next_force, candidate)),
                "next_force_dot": scalar(jnp.vdot(next_force, candidate)),
                "next_q": scalar(
                    residual_ratio(next_epsilon, next_score.T @ candidate)
                ),
                "consecutive_cosine": (
                    scalar(safe_cosine(previous, candidate))
                    if previous is not None
                    else None
                ),
            }
            previous_candidates[name] = candidate

        row = {
            "transition": index,
            "indicator": indicator,
            "indicator_bottom20": indicator <= c_bad,
            "indicator_top50": indicator >= c_on,
            "gate_base": gate_base,
            "acceptance": scalar(acceptance),
            "next_acceptance": scalar(next_acceptance),
            "raw_variance": scalar(
                jnp.var(stats["local_energies_noclip"], ddof=1)
            ),
            "next_raw_variance": scalar(
                jnp.var(next_stats["local_energies_noclip"], ddof=1)
            ),
            "candidate_metrics": candidate_metrics,
        }
        if not all(
            value is None or math.isfinite(value)
            for metrics in candidate_metrics.values()
            for value in metrics.values()
        ):
            raise FloatingPointError(row)
        validation.append(row)
        print(json.dumps({"phase": "validation", **row}, sort_keys=True), flush=True)
        consensus = normalize(RHO * consensus + (1.0 - RHO) * unit)
        acceptance, score, epsilon, stats = (
            next_acceptance,
            next_score,
            next_epsilon,
            next_stats,
        )

    def native_values(records, metric):
        return np.asarray(
            [row["candidate_metrics"]["native"][metric] for row in records],
            dtype=np.float64,
        )

    indicators = np.asarray([row["indicator"] for row in validation])
    native_alignment = native_values(validation, "next_alignment")
    native_minus_q = -native_values(validation, "next_q")
    alignment_rho = spearman(indicators, native_alignment)
    minus_q_rho = spearman(indicators, native_minus_q)
    alignment_rho_ci = block_bootstrap_stat(
        validation,
        lambda rows: spearman(
            [row["indicator"] for row in rows],
            [row["candidate_metrics"]["native"]["next_alignment"] for row in rows],
        ),
        seed_offset=1,
    )
    minus_q_rho_ci = block_bootstrap_stat(
        validation,
        lambda rows: spearman(
            [row["indicator"] for row in rows],
            [-row["candidate_metrics"]["native"]["next_q"] for row in rows],
        ),
        seed_offset=2,
    )

    def group_difference(rows, metric, negate=False):
        bottom = [
            row["candidate_metrics"]["native"][metric]
            for row in rows
            if row["indicator"] <= c_bad
        ]
        top = [
            row["candidate_metrics"]["native"][metric]
            for row in rows
            if row["indicator"] >= c_on
        ]
        if len(bottom) < 2 or len(top) < 2:
            return math.nan
        sign = -1.0 if negate else 1.0
        return sign * (float(np.median(top)) - float(np.median(bottom)))

    alignment_group_difference = group_difference(validation, "next_alignment")
    minus_q_group_difference = group_difference(validation, "next_q", negate=True)
    bottom_count = int(np.sum(indicators <= c_bad))
    top_count = int(np.sum(indicators >= c_on))
    group_test_valid = bottom_count >= 2 and top_count >= 2
    if group_test_valid:
        alignment_group_ci = block_bootstrap_stat(
            validation,
            lambda rows: group_difference(rows, "next_alignment"),
            seed_offset=3,
        )
        minus_q_group_ci = block_bootstrap_stat(
            validation,
            lambda rows: group_difference(rows, "next_q", negate=True),
            seed_offset=4,
        )
    else:
        alignment_group_ci = None
        minus_q_group_ci = None
    corr_significant = (
        alignment_rho_ci[0] > 0.0 or minus_q_rho_ci[0] > 0.0
    )
    corr_other_nonnegative = (
        minus_q_rho >= 0.0 if alignment_rho_ci[0] > 0.0 else alignment_rho >= 0.0
    )
    group_significant = bool(
        group_test_valid
        and (
            alignment_group_ci[0] > 0.0 or minus_q_group_ci[0] > 0.0
        )
    )
    group_other_nonnegative = (
        minus_q_group_difference >= 0.0
        if group_test_valid and alignment_group_ci[0] > 0.0
        else alignment_group_difference >= 0.0
    )
    indicator_passed = bool(
        corr_significant
        and corr_other_nonnegative
        and group_significant
        and group_other_nonnegative
    )

    median_native_solve = float(np.median(solve_seconds[1:]))
    median_candidate_seconds = float(np.median(candidate_seconds))
    overhead_fraction = median_candidate_seconds / max(median_native_solve, EPS)
    candidate_summaries = {}
    qualifying = []
    for candidate_index, gmax in enumerate(GMAX_GRID):
        name = candidate_name(gmax)
        coherence_differences = np.asarray(
            [
                row["candidate_metrics"][name]["consecutive_cosine"]
                - row["candidate_metrics"]["native"]["consecutive_cosine"]
                for row in validation
                if row["candidate_metrics"][name]["consecutive_cosine"] is not None
            ],
            dtype=np.float64,
        )
        alignment_differences = np.asarray(
            [
                row["candidate_metrics"][name]["next_alignment"]
                - row["candidate_metrics"]["native"]["next_alignment"]
                for row in validation
            ],
            dtype=np.float64,
        )
        q_ratios = np.asarray(
            [
                row["candidate_metrics"][name]["next_q"]
                / row["candidate_metrics"]["native"]["next_q"]
                for row in validation
            ],
            dtype=np.float64,
        )
        native_positive = float(
            np.mean(native_values(validation, "next_force_dot") > 0.0)
        )
        candidate_positive = float(
            np.mean(
                np.asarray(
                    [row["candidate_metrics"][name]["next_force_dot"] for row in validation]
                )
                > 0.0
            )
        )
        gate_fraction = float(
            np.mean(
                np.asarray(
                    [row["candidate_metrics"][name]["gate"] for row in validation]
                )
                > 0.0
            )
        )
        coherence_ci = block_bootstrap_stat(
            validation[1:],
            lambda rows: float(
                np.median(
                    [
                        row["candidate_metrics"][name]["consecutive_cosine"]
                        - row["candidate_metrics"]["native"]["consecutive_cosine"]
                        for row in rows
                    ]
                )
            ),
            seed_offset=10 + candidate_index,
        )
        alignment_ci = block_bootstrap_stat(
            validation,
            lambda rows: float(
                np.median(
                    [
                        row["candidate_metrics"][name]["next_alignment"]
                        - row["candidate_metrics"]["native"]["next_alignment"]
                        for row in rows
                    ]
                )
            ),
            seed_offset=20 + candidate_index,
        )
        q_ratio_ci = block_bootstrap_stat(
            validation,
            lambda rows: float(
                np.median(
                    [
                        row["candidate_metrics"][name]["next_q"]
                        / row["candidate_metrics"]["native"]["next_q"]
                        for row in rows
                    ]
                )
            ),
            seed_offset=30 + candidate_index,
        )
        median_coherence_improvement = float(np.median(coherence_differences))
        median_alignment_difference = float(np.median(alignment_differences))
        median_q_ratio = float(np.median(q_ratios))
        passed = bool(
            indicator_passed
            and median_coherence_improvement >= 0.15
            and coherence_ci[0] >= 0.10
            and median_alignment_difference >= -0.001
            and alignment_ci[0] >= -0.002
            and median_q_ratio <= 1.02
            and q_ratio_ci[1] <= 1.05
            and native_positive - candidate_positive <= 0.05
            and 0.20 <= gate_fraction <= 0.60
            and overhead_fraction < 0.05
        )
        candidate_summaries[name] = {
            "gmax": gmax,
            "median_consecutive_cosine_improvement": median_coherence_improvement,
            "block_bootstrap_95_ci_consecutive_improvement": coherence_ci,
            "median_next_alignment_difference": median_alignment_difference,
            "block_bootstrap_95_ci_alignment_difference": alignment_ci,
            "median_next_q_ratio": median_q_ratio,
            "block_bootstrap_95_ci_q_ratio": q_ratio_ci,
            "native_positive_next_force_fraction": native_positive,
            "candidate_positive_next_force_fraction": candidate_positive,
            "positive_fraction_drop": native_positive - candidate_positive,
            "gate_activation_fraction": gate_fraction,
            "passed_all_pre_registered_gates": passed,
        }
        if passed:
            qualifying.append(name)

    fixed_name = "always_blend_0.25"
    fixed_alignment_delta = np.asarray(
        [
            row["candidate_metrics"][fixed_name]["next_alignment"]
            - row["candidate_metrics"]["native"]["next_alignment"]
            for row in validation
        ]
    )
    fixed_q_ratio = np.asarray(
        [
            row["candidate_metrics"][fixed_name]["next_q"]
            / row["candidate_metrics"]["native"]["next_q"]
            for row in validation
        ]
    )
    fixed_coherence_delta = np.asarray(
        [
            row["candidate_metrics"][fixed_name]["consecutive_cosine"]
            - row["candidate_metrics"]["native"]["consecutive_cosine"]
            for row in validation
            if row["candidate_metrics"][fixed_name]["consecutive_cosine"] is not None
        ]
    )

    selected = qualifying[0] if qualifying else None
    payload = {
        "experiment": "C rank1600 WSSR temporal-consistency indicator replay",
        "read_only_checkpoint": True,
        "parameters_updated": False,
        "optimizer_state_updated": False,
        "source_checkpoint": str(checkpoint),
        "stored_epoch": int(stored_epoch),
        "native_only_consensus": True,
        "consensus_rho": RHO,
        "consensus_initialization_batches_excluded": 1,
        "calibration_transitions": CALIBRATION_TRANSITIONS,
        "validation_transitions": VALIDATION_TRANSITIONS,
        "walkers_per_batch": NCHAINS,
        "mcmc_steps_between_batches": int(config.vmc.nsteps_per_param_update),
        "config_verified": {name: actual for name, (actual, _) in expected.items()},
        "calibration_thresholds": {"c_bad_p20": c_bad, "c_on_p50": c_on},
        "pre_registered_indicator_gate": {
            "at_least_one_spearman_ci_lower_above_zero": True,
            "other_spearman_point_estimate_nonnegative": True,
            "at_least_one_top_minus_bottom_ci_lower_above_zero": True,
            "other_top_minus_bottom_point_estimate_nonnegative": True,
            "block_length": BLOCK_LENGTH,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
        "indicator_validation": {
            "spearman_c_vs_next_alignment": alignment_rho,
            "block_bootstrap_95_ci_spearman_alignment": alignment_rho_ci,
            "spearman_c_vs_negative_next_q": minus_q_rho,
            "block_bootstrap_95_ci_spearman_negative_q": minus_q_rho_ci,
            "top50_minus_bottom20_next_alignment": alignment_group_difference,
            "block_bootstrap_95_ci_group_alignment": alignment_group_ci,
            "top50_minus_bottom20_negative_next_q": minus_q_group_difference,
            "block_bootstrap_95_ci_group_negative_q": minus_q_group_ci,
            "bottom20_count": bottom_count,
            "top50_count": top_count,
            "group_test_valid": group_test_valid,
            "passed": indicator_passed,
        },
        "pre_registered_candidate_gate": {
            "median_consecutive_cosine_improvement_min": 0.15,
            "coherence_ci_lower_min": 0.10,
            "median_next_alignment_difference_min": -0.001,
            "alignment_ci_lower_min": -0.002,
            "median_next_q_ratio_max": 1.02,
            "q_ratio_ci_upper_max": 1.05,
            "positive_next_force_fraction_drop_max": 0.05,
            "gate_activation_fraction_range": [0.20, 0.60],
            "measured_vector_overhead_fraction_max": 0.05,
        },
        "candidate_summaries": candidate_summaries,
        "always_blend_control": {
            "gate": 0.25,
            "median_consecutive_cosine_improvement": float(
                np.median(fixed_coherence_delta)
            ),
            "median_next_alignment_difference": float(
                np.median(fixed_alignment_delta)
            ),
            "median_next_q_ratio": float(np.median(fixed_q_ratio)),
            "eligible_for_selection": False,
        },
        "timing": {
            "median_native_wssr_solve_seconds": median_native_solve,
            "median_all_three_candidate_vector_ops_seconds": median_candidate_seconds,
            "measured_candidate_overhead_fraction": overhead_fraction,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "decision": {
            "indicator_supported": indicator_passed,
            "qualifying_candidates": qualifying,
            "selected_candidate": selected,
            "short_training_authorized_by_replay": selected is not None,
            "conclusion": (
                "The low-consensus indicator predicts held-out quality and a gated "
                "candidate passes all offline safeguards."
                if selected is not None
                else (
                    "The indicator is predictive, but no candidate passes the full "
                    "coherence/held-out/cost gate; do not train it."
                    if indicator_passed
                    else "Low consensus cosine is not a validated control indicator; "
                    "stop without training the temporal filter."
                )
            ),
        },
        "calibration": calibration,
        "validation": validation,
        "finite": True,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
