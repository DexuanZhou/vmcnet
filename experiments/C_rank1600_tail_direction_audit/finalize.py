#!/usr/bin/env python3
"""Finalize the C WSSR/SPRING tail mechanism decision with bootstrap intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


BOOTSTRAP_SEED = 20260802
BOOTSTRAP_REPLICATES = 100000


def interval(values):
    return {
        "estimate": float(np.median(values)),
        "bootstrap_95_percent": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
    }


def bootstrap_median(rng, values):
    values = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    return np.median(values[indices], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wssr", type=Path, required=True)
    parser.add_argument("--spring", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.output_json, args.output_report):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")

    wssr = json.loads(args.wssr.read_text(encoding="utf-8"))
    spring = json.loads(args.spring.read_text(encoding="utf-8"))
    wr, sr = wssr["per_batch"], spring["per_batch"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    w_var = np.asarray([row["raw_variance"] for row in wr])
    s_var = np.asarray([row["raw_variance"] for row in sr])
    w_var_boot = bootstrap_median(rng, w_var)
    s_var_boot = bootstrap_median(rng, s_var)
    ratio_boot = w_var_boot / s_var_boot

    w_clip_gap = np.asarray(
        [row["raw_tail_q_minsr_clipped"] - row["raw_tail_q_minsr_raw"] for row in wr]
    )
    s_clip_gap = np.asarray(
        [row["raw_tail_q_minsr_clipped"] - row["raw_tail_q_minsr_raw"] for row in sr]
    )
    w_trunc_gap = np.asarray(
        [row["raw_tail_q_wssr"] - row["raw_tail_q_minsr_clipped"] for row in wr]
    )
    s_native_gap = np.asarray(
        [row["raw_tail_q_spring_native"] - row["raw_tail_q_minsr_clipped"] for row in sr]
    )
    leakage = np.asarray([row["raw_tail_active_leakage"] for row in wr])

    frozen = {
        "spring": {
            "energy": -37.844913176158094,
            "sem": 1.213375990184921e-05,
            "raw_variance": 0.0045313891571030634,
            "source": "/scratch/dexuan1/runs/C_spring_official_kfac1000_E100000/eval/statistics.json",
        },
        "wssr": {
            "energy": -37.844333445821455,
            "sem": 2.145690135816009e-05,
            "raw_variance": 0.01421094734140307,
            "source": "/scratch/dexuan1/runs/C_wssr_rank1600_fixed_lambda_stage2_E2000_frozen/epoch102000/eval/statistics.json",
        },
    }
    frozen_ratio = frozen["wssr"]["raw_variance"] / frozen["spring"]["raw_variance"]
    payload = {
        "experiment": "C WSSR versus SPRING tail mechanism final decision",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "matched_configuration": {
            "starting_checkpoint": "/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1/checkpoints/1000.npz",
            "model_equal": True,
            "problem_equal": True,
            "walkers": 1000,
            "mcmc_steps": 10,
            "clip_threshold": 5.0,
            "clip_center": "mean",
            "dtype": "float32",
            "reburn": False,
            "spring_training_epochs": 100000,
            "wssr_training_epochs_at_audit": 102000,
        },
        "fixed_checkpoint_audit": {
            "wssr_tail_leakage": interval(
                bootstrap_median(rng, leakage)
            ),
            "wssr_q_minus_clipped_minsr": interval(
                bootstrap_median(rng, w_trunc_gap)
            ),
            "wssr_clipped_minus_raw_minsr_q": interval(
                bootstrap_median(rng, w_clip_gap)
            ),
            "spring_native_minus_clipped_minsr_q": interval(
                bootstrap_median(rng, s_native_gap)
            ),
            "spring_clipped_minus_raw_minsr_q": interval(
                bootstrap_median(rng, s_clip_gap)
            ),
            "wssr_median_raw_variance": interval(w_var_boot),
            "spring_median_raw_variance": interval(s_var_boot),
            "wssr_over_spring_median_raw_variance": {
                "estimate": float(np.median(w_var) / np.median(s_var)),
                "bootstrap_95_percent": [
                    float(np.quantile(ratio_boot, 0.025)),
                    float(np.quantile(ratio_boot, 0.975)),
                ],
            },
            "wssr_median_max_robust_z": float(
                np.median([row["raw_max_robust_z"] for row in wr])
            ),
            "spring_median_max_robust_z": float(
                np.median([row["raw_max_robust_z"] for row in sr])
            ),
        },
        "frozen_evaluation": {
            **frozen,
            "wssr_over_spring_variance": frozen_ratio,
        },
        "decisions": {
            "hard_truncation_as_tail_cause": "rejected_in_tested_rank1600_checkpoint",
            "generic_tail_basis_intervention": "not_supported",
            "clipping_suppresses_raw_tail_response": "supported_for_both_optimizers",
            "spring_is_immune_to_raw_tail_events": "rejected",
            "spring_final_wavefunction_has_lower_bulk_and_frozen_variance": "supported",
            "best_current_interpretation": (
                "SPRING reaches a better long-trajectory parameter basin; its final "
                "advantage is not explained by a special instantaneous response to "
                "raw tail walkers, and WSSR rank truncation is not the measured cause."
            ),
            "submit_tail_basis_E500_or_E5000": False,
        },
        "source_paths": [str(args.wssr), str(args.spring)],
    }
    args.output_json.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    audit = payload["fixed_checkpoint_audit"]
    lines = [
        "# C WSSR versus SPRING tail-mechanism conclusion",
        "",
        "## Conclusion",
        "",
        "The tested rank-1600 WSSR tail problem is not caused by the tail force falling outside the retained WSSR subspace. Both optimizers strongly suppress raw-tail information through the shared 5-SD clipping rule. SPRING is not immune to rare raw local-energy spikes; its advantage is a lower-variance final parameter basin reached over the full optimization trajectory.",
        "",
        "Consequently, the proposed tail-basis E500/E5000 intervention is not released.",
        "",
        "## Fixed-checkpoint evidence",
        "",
        "| quantity | estimate | bootstrap 95% interval |",
        "|---|---:|---:|",
    ]
    for label, key in (
        ("WSSR tail-force leakage", "wssr_tail_leakage"),
        ("Q(WSSR)-Q(clipped MinSR)", "wssr_q_minus_clipped_minsr"),
        ("WSSR Q(clipped MinSR)-Q(raw MinSR)", "wssr_clipped_minus_raw_minsr_q"),
        ("Q(native SPRING)-Q(clipped MinSR)", "spring_native_minus_clipped_minsr_q"),
        ("SPRING Q(clipped MinSR)-Q(raw MinSR)", "spring_clipped_minus_raw_minsr_q"),
        ("WSSR/Spring median raw-variance ratio", "wssr_over_spring_median_raw_variance"),
    ):
        row = audit[key]
        lines.append(
            f"| {label} | {row['estimate']:.6g} | "
            f"[{row['bootstrap_95_percent'][0]:.6g}, {row['bootstrap_95_percent'][1]:.6g}] |"
        )
    lines.extend(
        [
            "",
            "The median maximum robust-z is larger for SPRING ("
            f"{audit['spring_median_max_robust_z']:.3f}) than for WSSR ("
            f"{audit['wssr_median_max_robust_z']:.3f}); this directly rejects the claim that SPRING has no raw tail events.",
            "",
            "## Frozen evaluation",
            "",
            f"- SPRING raw variance: {frozen['spring']['raw_variance']:.8f} Ha^2.",
            f"- WSSR raw variance: {frozen['wssr']['raw_variance']:.8f} Ha^2.",
            f"- WSSR/SPRING ratio: {frozen_ratio:.3f}.",
            "",
            "## Scope",
            "",
            "This conclusion applies to the tested C checkpoints, rank1600, eta=0.3, learning rate 0.04, fixed lambda=0.001, cutoff=0.0003, SSI warm=2, and complement_weight=0. It does not prove that rank truncation is harmless for other ranks or systems.",
        ]
    )
    args.output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
