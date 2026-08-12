#!/usr/bin/env python3
"""Read-only native-factor proximal replay for C rank-1600 WSSR.

Parameters and optimizer state stay fixed; only walkers advance.  The temporal
candidate is constructed inside the production warm2 SSI factorization and is
compared with a memoryless control having the same extra spectral damping.
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
EPS = 1.0e-12
GAMMA_MULTIPLIERS = (0.1, 0.3, 1.0, 3.0)
TRAIN_TRANSITIONS = 15
BLOCK_LENGTH = 5
BOOTSTRAP_REPLICATES = 4000
BOOTSTRAP_SEED = 20260803


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
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-fold-in", type=int, default=0)
    parser.add_argument(
        "--confirmation-gamma",
        type=float,
        default=None,
        help="Pre-fix one gamma and validate it on every independent transition.",
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
    if args.key_fold_in:
        key = jax.random.fold_in(key, args.key_fold_in)
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
    gamma_values = [
        float(multiplier * tikhonov_lambda) for multiplier in GAMMA_MULTIPLIERS
    ]

    def native_solve(score, epsilon, direction_key, state_arg):
        augmented, rhs = wssr.augment_wssr_system(
            score, epsilon, state_arg, float(opt.eta)
        )
        u, singular_values, vh, _ = wssr._wssr_right_svd_decomposition(
            augmented,
            state_arg,
            direction_key,
            RANK,
            int(opt.svd_maxiter_initial),
            2,
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
        return result.grad_like_update, u, singular_values

    native_solve_jit = jax.jit(native_solve)

    @jax.jit
    def native_factor_candidates(
        native_update, u, singular_values, previous_proximal
    ):
        value_dtype = native_update.dtype
        gammas = jnp.asarray(GAMMA_MULTIPLIERS, dtype=value_dtype) * jnp.asarray(
            tikhonov_lambda, dtype=value_dtype
        )
        active = (
            singular_values
            / jnp.maximum(singular_values[0], jnp.asarray(EPS, value_dtype))
            > cutoff
        )
        current_left = u.T @ native_update
        previous_left = u.T @ previous_proximal
        base = jnp.square(singular_values) + jnp.asarray(
            tikhonov_lambda, dtype=value_dtype
        )
        denominators = base[:, None] + gammas[None, :]
        proximal_left = (
            base[:, None] * current_left[:, None]
            + gammas[None, :] * previous_left
        ) / denominators
        control_left = base[:, None] * current_left[:, None] / denominators
        active = active.astype(value_dtype)[:, None]
        return u @ (proximal_left * active), u @ (control_left * active)

    @jax.jit
    def consecutive_cosines(previous, current):
        dots = jnp.sum(previous * current, axis=0)
        norms = jnp.linalg.norm(previous, axis=0) * jnp.linalg.norm(current, axis=0)
        return dots / jnp.maximum(norms, EPS)

    @jax.jit
    def heldout_metrics(score, epsilon, native, proximal, control):
        directions = jnp.concatenate((native[:, None], proximal, control), axis=1)
        native_norm = jnp.linalg.norm(native)
        directions = directions * (
            native_norm / jnp.maximum(jnp.linalg.norm(directions, axis=0), EPS)
        )[None, :]
        predictions = score.T @ directions
        force = score @ epsilon
        alignments = (force @ directions) / jnp.maximum(
            jnp.linalg.norm(force) * jnp.linalg.norm(directions, axis=0), EPS
        )
        q_values = jnp.linalg.norm(epsilon[:, None] - predictions, axis=0) / jnp.maximum(
            jnp.linalg.norm(epsilon), EPS
        )
        return alignments, q_values, force @ directions

    def sample_batch(data_arg, key_arg):
        acceptance, data_arg, key_arg = walker_fn(params, data_arg, key_arg)
        positions_arg = pacore.get_position_from_data(data_arg)
        energy, clipped, stats = energy_fn(params, positions_arg)
        score, _ = wssr.center_and_scale_score_matrix(log_psi, params, positions_arg)
        epsilon = wssr.center_and_scale_energy_residuals(clipped, energy)
        return data_arg, key_arg, acceptance, energy, score, epsilon, stats

    flat_params, _ = jax.flatten_util.ravel_pytree(params)
    previous_native = jnp.zeros_like(flat_params)
    previous_proximal = jnp.zeros(
        (flat_params.shape[0], len(gamma_values)), dtype=flat_params.dtype
    )
    previous_control = jnp.zeros_like(previous_proximal)
    records = []
    native_seconds = []
    candidate_seconds = []
    started = time.perf_counter()

    for batch in range(BATCHES):
        data, key, acceptance, energy, score, epsilon, stats = sample_batch(data, key)
        heldout = None
        if batch > 0:
            heldout_values = jax.device_get(
                heldout_metrics(
                    score,
                    epsilon,
                    previous_native,
                    previous_proximal,
                    previous_control,
                )
            )
            heldout = {
                "native": {
                    "alignment": float(heldout_values[0][0]),
                    "q": float(heldout_values[1][0]),
                    "force_dot": float(heldout_values[2][0]),
                },
                "proximal": {},
                "damping_control": {},
            }
            for index, gamma in enumerate(gamma_values):
                heldout["proximal"][str(gamma)] = {
                    "alignment": float(heldout_values[0][1 + index]),
                    "q": float(heldout_values[1][1 + index]),
                    "force_dot": float(heldout_values[2][1 + index]),
                }
                control_index = 1 + len(gamma_values) + index
                heldout["damping_control"][str(gamma)] = {
                    "alignment": float(heldout_values[0][control_index]),
                    "q": float(heldout_values[1][control_index]),
                    "force_dot": float(heldout_values[2][control_index]),
                }

        key, direction_key = jax.random.split(key)
        begin = time.perf_counter()
        native_update, u, singular_values = native_solve_jit(
            score, epsilon, direction_key, state
        )
        jax.block_until_ready(native_update)
        native_seconds.append(time.perf_counter() - begin)

        begin = time.perf_counter()
        proximal, control = native_factor_candidates(
            native_update, u, singular_values, previous_proximal
        )
        jax.block_until_ready(proximal)
        candidate_seconds.append(time.perf_counter() - begin)
        native_cosine = float(
            jax.device_get(
                jnp.vdot(previous_native, native_update)
                / jnp.maximum(
                    jnp.linalg.norm(previous_native) * jnp.linalg.norm(native_update),
                    EPS,
                )
            )
        )
        proximal_cosines, control_cosines = jax.device_get(
            (
                consecutive_cosines(previous_proximal, proximal),
                consecutive_cosines(previous_control, control),
            )
        )
        record = {
            "batch": batch,
            "acceptance": float(jax.device_get(acceptance)),
            "energy_internal_only": float(jax.device_get(energy)),
            "raw_variance": float(
                jax.device_get(jnp.var(stats["local_energies_noclip"], ddof=1))
            ),
            "native_consecutive_cosine": native_cosine,
            "proximal_consecutive_cosine": {
                str(gamma): float(proximal_cosines[index])
                for index, gamma in enumerate(gamma_values)
            },
            "damping_control_consecutive_cosine": {
                str(gamma): float(control_cosines[index])
                for index, gamma in enumerate(gamma_values)
            },
            "heldout_next_batch_for_previous_directions": heldout,
        }
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        previous_native = native_update
        previous_proximal = proximal
        previous_control = control
        # Do not keep the previous batch's multi-GB score and retained basis
        # alive while Python evaluates the next sample_batch call.  Only the
        # compact direction states above are causally required across batches.
        del score, epsilon, stats, u, singular_values
        del native_update, proximal, control

    transitions = records[1:]
    train_indices = list(range(TRAIN_TRANSITIONS))
    validation_indices = (
        list(range(len(transitions)))
        if args.confirmation_gamma is not None
        else list(range(TRAIN_TRANSITIONS, len(transitions)))
    )
    native_cosines = [record["native_consecutive_cosine"] for record in transitions]

    def series(gamma, family, metric):
        key = str(gamma)
        if metric == "cosine":
            return [
                record[f"{family}_consecutive_cosine"][key]
                for record in transitions
            ]
        return [
            record["heldout_next_batch_for_previous_directions"][family][key][metric]
            for record in transitions
        ]

    def native_series(metric):
        return [
            record["heldout_next_batch_for_previous_directions"]["native"][metric]
            for record in transitions
        ]

    native_alignment = native_series("alignment")
    native_q = native_series("q")
    native_force = native_series("force_dot")
    grid = {}
    eligible = []
    for gamma_index, gamma in enumerate(gamma_values):
        prox_cos = series(gamma, "proximal", "cosine")
        ctrl_cos = series(gamma, "damping_control", "cosine")
        prox_alignment = series(gamma, "proximal", "alignment")
        ctrl_alignment = series(gamma, "damping_control", "alignment")
        prox_q = series(gamma, "proximal", "q")
        ctrl_q = series(gamma, "damping_control", "q")
        prox_force = series(gamma, "proximal", "force_dot")
        ctrl_force = series(gamma, "damping_control", "force_dot")
        raw = {
            "coherence_vs_control": [a - b for a, b in zip(prox_cos, ctrl_cos)],
            "coherence_vs_native": [a - b for a, b in zip(prox_cos, native_cosines)],
            "alignment_vs_control": [
                a - b for a, b in zip(prox_alignment, ctrl_alignment)
            ],
            "alignment_vs_native": [
                a - b for a, b in zip(prox_alignment, native_alignment)
            ],
            "q_ratio_vs_control": [a / b for a, b in zip(prox_q, ctrl_q)],
            "q_ratio_vs_native": [a / b for a, b in zip(prox_q, native_q)],
            "proximal_force_dot": prox_force,
            "control_force_dot": ctrl_force,
        }
        train = {
            name: summarize([values[index] for index in train_indices])
            for name, values in raw.items()
            if name not in ("proximal_force_dot", "control_force_dot")
        }
        train_safe = (
            train["coherence_vs_control"]["median"] >= 0.05
            and train["alignment_vs_control"]["median"] >= -0.001
            and train["q_ratio_vs_control"]["median"] <= 1.02
            and train["alignment_vs_native"]["median"] >= -0.002
            and train["q_ratio_vs_native"]["median"] <= 1.05
        )
        if train_safe:
            eligible.append((train["coherence_vs_control"]["median"], gamma_index, gamma))
        grid[str(gamma)] = {
            "gamma_multiplier_of_lambda": GAMMA_MULTIPLIERS[gamma_index],
            "train": train,
            "train_safe": train_safe,
            "raw": raw,
        }

    if args.confirmation_gamma is not None:
        selected_gamma = float(args.confirmation_gamma)
        if str(selected_gamma) not in grid:
            raise ValueError(
                f"confirmation gamma {selected_gamma} is not in {gamma_values}"
            )
    else:
        selected_gamma = max(eligible)[2] if eligible else None
    validation = None
    gates = {
        "incremental_coherence": False,
        "absolute_coherence": False,
        "alignment_vs_control": False,
        "residual_vs_control": False,
        "alignment_vs_native": False,
        "residual_vs_native": False,
        "positive_force": False,
    }
    if selected_gamma is not None:
        raw = grid[str(selected_gamma)]["raw"]

        def validation_summary(name, seed):
            values = [raw[name][index] for index in validation_indices]
            return {
                **summarize(values),
                "block_bootstrap_95_ci_median": bootstrap_median_ci(values, seed),
            }

        validation = {
            "coherence_vs_control": validation_summary("coherence_vs_control", 1),
            "coherence_vs_native": validation_summary("coherence_vs_native", 2),
            "alignment_vs_control": validation_summary("alignment_vs_control", 3),
            "q_ratio_vs_control": validation_summary("q_ratio_vs_control", 4),
            "alignment_vs_native": validation_summary("alignment_vs_native", 5),
            "q_ratio_vs_native": validation_summary("q_ratio_vs_native", 6),
        }
        gates = {
            "incremental_coherence": (
                validation["coherence_vs_control"]["median"] >= 0.05
                and validation["coherence_vs_control"]["block_bootstrap_95_ci_median"][0]
                >= 0.02
            ),
            "absolute_coherence": (
                validation["coherence_vs_native"]["median"] >= 0.10
                and validation["coherence_vs_native"]["block_bootstrap_95_ci_median"][0]
                >= 0.05
            ),
            "alignment_vs_control": (
                validation["alignment_vs_control"]["median"] >= -0.001
                and validation["alignment_vs_control"]["block_bootstrap_95_ci_median"][0]
                >= -0.002
            ),
            "residual_vs_control": (
                validation["q_ratio_vs_control"]["median"] <= 1.02
                and validation["q_ratio_vs_control"]["block_bootstrap_95_ci_median"][1]
                <= 1.05
            ),
            "alignment_vs_native": (
                validation["alignment_vs_native"]["median"] >= -0.002
                and validation["alignment_vs_native"]["block_bootstrap_95_ci_median"][0]
                >= -0.004
            ),
            "residual_vs_native": (
                validation["q_ratio_vs_native"]["median"] <= 1.05
                and validation["q_ratio_vs_native"]["block_bootstrap_95_ci_median"][1]
                <= 1.08
            ),
            "positive_force": (
                np.mean([raw["proximal_force_dot"][i] > 0.0 for i in validation_indices])
                >= np.mean([raw["control_force_dot"][i] > 0.0 for i in validation_indices])
                - 0.05
                and np.mean([raw["proximal_force_dot"][i] > 0.0 for i in validation_indices])
                >= np.mean([native_force[i] > 0.0 for i in validation_indices]) - 0.05
            ),
        }

    gates = {name: bool(value) for name, value in gates.items()}
    passed = bool(selected_gamma is not None and all(gates.values()))
    payload = {
        "experiment": "C rank1600 native-factor proximal coefficient replay",
        "read_only_checkpoint": True,
        "parameters_updated": False,
        "optimizer_state_updated": False,
        "walkers_advanced": True,
        "source_checkpoint": str(checkpoint),
        "stored_epoch": int(stored_epoch),
        "key_fold_in": int(args.key_fold_in),
        "confirmation_mode": args.confirmation_gamma is not None,
        "confirmation_gamma_fixed_before_replay": args.confirmation_gamma,
        "config_verified": {name: actual for name, (actual, _) in expected.items()},
        "gamma_values": gamma_values,
        "mathematical_form": (
            "c_i=((s_i^2+lambda)c_native_i+gamma<u_i,d_prev>)/"
            "(s_i^2+lambda+gamma)"
        ),
        "matched_control": (
            "same native warm2 factors and denominator, with zero history term"
        ),
        "pre_registered_protocol": {
            "chronological_train_transitions": 15,
            "chronological_validation_transitions": (
                29 if args.confirmation_gamma is not None else 14
            ),
            "selection": (
                f"independent confirmation fixed gamma={args.confirmation_gamma}"
                if args.confirmation_gamma is not None
                else "maximize train coherence gain among train-safe gammas"
            ),
            "validation_thresholds": {
                "coherence_vs_control_median_ci_lower": [0.05, 0.02],
                "coherence_vs_native_median_ci_lower": [0.10, 0.05],
                "alignment_vs_control_median_ci_lower": [-0.001, -0.002],
                "q_vs_control_median_ci_upper": [1.02, 1.05],
                "alignment_vs_native_median_ci_lower": [-0.002, -0.004],
                "q_vs_native_median_ci_upper": [1.05, 1.08],
            },
        },
        "grid": grid,
        "selected_gamma": selected_gamma,
        "validation": validation,
        "validation_gates": gates,
        "offline_passed": passed,
        "e500_authorized": passed,
        "timing": {
            "median_native_seconds": float(np.median(native_seconds[1:])),
            "median_four_gamma_candidate_seconds": float(
                np.median(candidate_seconds[1:])
            ),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "finite": bool(
            all(
                np.isfinite(value)
                for record in records
                for name, value in record.items()
                if isinstance(value, (int, float)) and name != "batch"
            )
        ),
        "records": records,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.output.with_name("summary.md").write_text(
        "\n".join(
            [
                "# C rank-1600 native-factor proximal replay",
                "",
                "Fixed parameters and optimizer state; only walkers advanced.",
                "Absolute C energies are not used for physical claims.",
                "",
                f"- Selected gamma: `{selected_gamma}`.",
                f"- Validation gates: `{json.dumps(gates, sort_keys=True)}`.",
                f"- Offline passed: **{passed}**.",
                f"- E500 authorized: **{passed}**.",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
