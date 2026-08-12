#!/usr/bin/env python3
"""Recover the pre-registered replay summary from complete per-transition JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


BLOCK_LENGTH = 5
BOOTSTRAP_REPLICATES = 4000
BOOTSTRAP_SEED = 20260802
GATED_NAMES = ("gated_gmax_0.25", "gated_gmax_0.50")


def rankdata(values):
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


def spearman(left, right):
    left = rankdata(left)
    right = rankdata(right)
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def circular_indices(length, rng):
    count = math.ceil(length / BLOCK_LENGTH)
    starts = rng.integers(0, length, size=count)
    offsets = np.arange(BLOCK_LENGTH)
    return ((starts[:, None] + offsets[None, :]) % length).reshape(-1)[:length]


def bootstrap(records, statistic, seed_offset):
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    values = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = [records[i] for i in circular_indices(len(records), rng)]
        value = statistic(sampled)
        if math.isfinite(value):
            values.append(value)
    if not values:
        return {"ci": None, "finite_replicates": 0}
    return {
        "ci": [float(x) for x in np.quantile(values, [0.025, 0.975])],
        "finite_replicates": len(values),
    }


def metric(rows, name, field):
    return np.asarray(
        [row["candidate_metrics"][name][field] for row in rows], dtype=np.float64
    )


def group_difference(rows, field, c_bad, c_on, negate=False):
    bottom = [
        row["candidate_metrics"]["native"][field]
        for row in rows
        if row["indicator"] <= c_bad
    ]
    top = [
        row["candidate_metrics"]["native"][field]
        for row in rows
        if row["indicator"] >= c_on
    ]
    if len(bottom) < 2 or len(top) < 2:
        return math.nan
    difference = float(np.median(top)) - float(np.median(bottom))
    return -difference if negate else difference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    lines = [line for line in args.log.read_text(encoding="utf-8").splitlines() if line]
    rows = [json.loads(line) for line in lines]
    calibration = [row for row in rows if row["phase"] == "calibration"]
    validation = [row for row in rows if row["phase"] == "validation"]
    if len(calibration) != 40 or len(validation) != 80:
        raise ValueError((len(calibration), len(validation)))

    calibration_c = np.asarray([row["indicator"] for row in calibration])
    c_bad, c_on = [float(x) for x in np.quantile(calibration_c, [0.20, 0.50])]
    indicators = np.asarray([row["indicator"] for row in validation])
    alignment = metric(validation, "native", "next_alignment")
    negative_q = -metric(validation, "native", "next_q")
    alignment_rho = spearman(indicators, alignment)
    negative_q_rho = spearman(indicators, negative_q)
    alignment_rho_boot = bootstrap(
        validation,
        lambda sample: spearman(
            [row["indicator"] for row in sample],
            [row["candidate_metrics"]["native"]["next_alignment"] for row in sample],
        ),
        1,
    )
    negative_q_rho_boot = bootstrap(
        validation,
        lambda sample: spearman(
            [row["indicator"] for row in sample],
            [-row["candidate_metrics"]["native"]["next_q"] for row in sample],
        ),
        2,
    )
    bottom_count = int(np.sum(indicators <= c_bad))
    top_count = int(np.sum(indicators >= c_on))
    alignment_group = group_difference(
        validation, "next_alignment", c_bad, c_on
    )
    negative_q_group = group_difference(
        validation, "next_q", c_bad, c_on, negate=True
    )
    alignment_group_boot = bootstrap(
        validation,
        lambda sample: group_difference(sample, "next_alignment", c_bad, c_on),
        3,
    )
    negative_q_group_boot = bootstrap(
        validation,
        lambda sample: group_difference(
            sample, "next_q", c_bad, c_on, negate=True
        ),
        4,
    )

    group_test_valid = bottom_count >= 2 and top_count >= 2
    correlation_pass = bool(
        (
            alignment_rho_boot["ci"] is not None
            and alignment_rho_boot["ci"][0] > 0.0
            and negative_q_rho >= 0.0
        )
        or (
            negative_q_rho_boot["ci"] is not None
            and negative_q_rho_boot["ci"][0] > 0.0
            and alignment_rho >= 0.0
        )
    )
    group_pass = bool(
        group_test_valid
        and (
            (
                alignment_group_boot["ci"] is not None
                and alignment_group_boot["ci"][0] > 0.0
                and negative_q_group >= 0.0
            )
            or (
                negative_q_group_boot["ci"] is not None
                and negative_q_group_boot["ci"][0] > 0.0
                and alignment_group >= 0.0
            )
        )
    )
    indicator_pass = correlation_pass and group_pass

    candidate_summaries = {}
    for index, name in enumerate(GATED_NAMES):
        coherence_delta = np.asarray(
            [
                row["candidate_metrics"][name]["consecutive_cosine"]
                - row["candidate_metrics"]["native"]["consecutive_cosine"]
                for row in validation[1:]
            ]
        )
        alignment_delta = metric(validation, name, "next_alignment") - alignment
        q_ratio = metric(validation, name, "next_q") / metric(
            validation, "native", "next_q"
        )
        native_positive = float(np.mean(metric(validation, "native", "next_force_dot") > 0))
        candidate_positive = float(np.mean(metric(validation, name, "next_force_dot") > 0))
        activation = float(np.mean(metric(validation, name, "gate") > 0))
        coherence_boot = bootstrap(
            validation[1:],
            lambda sample: float(
                np.median(
                    [
                        row["candidate_metrics"][name]["consecutive_cosine"]
                        - row["candidate_metrics"]["native"]["consecutive_cosine"]
                        for row in sample
                    ]
                )
            ),
            10 + index,
        )
        alignment_boot = bootstrap(
            validation,
            lambda sample: float(
                np.median(
                    [
                        row["candidate_metrics"][name]["next_alignment"]
                        - row["candidate_metrics"]["native"]["next_alignment"]
                        for row in sample
                    ]
                )
            ),
            20 + index,
        )
        q_boot = bootstrap(
            validation,
            lambda sample: float(
                np.median(
                    [
                        row["candidate_metrics"][name]["next_q"]
                        / row["candidate_metrics"]["native"]["next_q"]
                        for row in sample
                    ]
                )
            ),
            30 + index,
        )
        candidate_summaries[name] = {
            "median_consecutive_cosine_improvement": float(np.median(coherence_delta)),
            "block_bootstrap_consecutive_improvement": coherence_boot,
            "median_next_alignment_difference": float(np.median(alignment_delta)),
            "block_bootstrap_alignment_difference": alignment_boot,
            "median_next_q_ratio": float(np.median(q_ratio)),
            "block_bootstrap_q_ratio": q_boot,
            "native_positive_next_force_fraction": native_positive,
            "candidate_positive_next_force_fraction": candidate_positive,
            "positive_fraction_drop": native_positive - candidate_positive,
            "gate_activation_fraction": activation,
            "passed_all_pre_registered_gates": False,
            "failure_reasons": [
                "indicator gate failed",
                "gate activation is outside the pre-registered 20%-60% range",
            ],
        }

    fixed = "always_blend_0.25"
    fixed_coherence = np.asarray(
        [
            row["candidate_metrics"][fixed]["consecutive_cosine"]
            - row["candidate_metrics"]["native"]["consecutive_cosine"]
            for row in validation[1:]
        ]
    )
    fixed_alignment = metric(validation, fixed, "next_alignment") - alignment
    fixed_q_ratio = metric(validation, fixed, "next_q") / metric(
        validation, "native", "next_q"
    )
    payload = {
        "experiment": "C rank1600 WSSR temporal-consistency indicator replay",
        "recovered_after_summary_only_failure": True,
        "source_job_id": args.job_id,
        "source_job_state": "FAILED",
        "source_job_failure_class": "collector/statistics implementation",
        "scientific_gpu_transitions_complete": True,
        "source_log": str(args.log),
        "source_log_sha256": hashlib.sha256(args.log.read_bytes()).hexdigest(),
        "read_only_checkpoint": True,
        "parameters_updated": False,
        "optimizer_state_updated": False,
        "calibration_transitions": len(calibration),
        "validation_transitions": len(validation),
        "consensus_rho": 0.9,
        "calibration_thresholds": {"c_bad_p20": c_bad, "c_on_p50": c_on},
        "validation_indicator_summary": {
            "min": float(np.min(indicators)),
            "median": float(np.median(indicators)),
            "max": float(np.max(indicators)),
        },
        "indicator_validation": {
            "spearman_c_vs_next_alignment": alignment_rho,
            "block_bootstrap_spearman_alignment": alignment_rho_boot,
            "spearman_c_vs_negative_next_q": negative_q_rho,
            "block_bootstrap_spearman_negative_q": negative_q_rho_boot,
            "bottom20_count_using_frozen_calibration_threshold": bottom_count,
            "top50_count_using_frozen_calibration_threshold": top_count,
            "group_test_valid": group_test_valid,
            "top50_minus_bottom20_next_alignment": (
                alignment_group if math.isfinite(alignment_group) else None
            ),
            "block_bootstrap_group_alignment": alignment_group_boot,
            "top50_minus_bottom20_negative_next_q": (
                negative_q_group if math.isfinite(negative_q_group) else None
            ),
            "block_bootstrap_group_negative_q": negative_q_group_boot,
            "correlation_gate_passed": correlation_pass,
            "group_gate_passed": group_pass,
            "passed": indicator_pass,
        },
        "candidate_summaries": candidate_summaries,
        "always_blend_control": {
            "gate": 0.25,
            "median_consecutive_cosine_improvement": float(np.median(fixed_coherence)),
            "median_next_alignment_difference": float(np.median(fixed_alignment)),
            "median_next_q_ratio": float(np.median(fixed_q_ratio)),
            "eligible_for_selection": False,
        },
        "timing": {
            "candidate_overhead_fraction": None,
            "note": "not recoverable from JSONL; unnecessary because the indicator already failed",
        },
        "decision": {
            "indicator_supported": False,
            "qualifying_candidates": [],
            "selected_candidate": None,
            "short_training_authorized_by_replay": False,
            "conclusion": (
                "The frozen calibration thresholds became nonstationary as the native-only "
                "consensus warmed up: validation contained no bottom-20% events and the "
                "gate activated on only 1/80 transitions.  The pre-registered indicator "
                "test therefore fails and no temporal-filter training should be run."
            ),
        },
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
