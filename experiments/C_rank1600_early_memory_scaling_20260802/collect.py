#!/usr/bin/env python3
"""Collect and gate the paired early memory-scaling audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ARMS = (
    "single_current",
    "wssr_eta03",
    "wssr_eta08",
    "wssr_eta095",
    "wssr_eta099",
    "wssr_eta099_bias_corrected",
    "spring_mu099",
)
WSSR_ARMS = ARMS[1:6]
DEPTHS = (3, 5, 8, 16)
SEED = 20260802


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def bootstrap_ci(values, offset):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(SEED + offset)
    draws = rng.integers(0, len(values), (10000, len(values)))
    medians = np.median(values[draws], axis=1)
    return [float(x) for x in np.quantile(medians, (0.025, 0.975))]


def candidate_name(arm):
    if arm == "spring_mu099":
        return "spring"
    return "fixed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.root / "summary.json"
    report_path = args.root / "report.md"
    if summary_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite collector output")

    payloads = {}
    for arm in ARMS:
        path = args.root / arm / "audit.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["arm"] != arm or payload["replicates"] != 6:
            raise ValueError((arm, payload["arm"], payload["replicates"]))
        payloads[arm] = payload

    results = {}
    passing = []
    single_rows = payloads["single_current"]["per_replicate"]
    short_rows = payloads["wssr_eta03"]["per_replicate"]
    for arm_index, arm in enumerate(ARMS[1:]):
            rows = payloads[arm]["per_replicate"]
            name = candidate_name(arm)
            depth_results = {}
            for depth in DEPTHS:
                key = str(depth)
                single = np.asarray([
                    row["depths"][key]["candidates"]["fixed"][
                        "future_consensus_cosine"
                    ]
                    for row in single_rows
                ])
                short = np.asarray([
                    row["depths"][key]["candidates"]["fixed"][
                        "future_consensus_cosine"
                    ]
                    for row in short_rows
                ])
                candidate = np.asarray([
                    row["depths"][key]["candidates"][name][
                        "future_consensus_cosine"
                    ]
                    for row in rows
                ])
                delta = candidate - short
                depth_results[key] = {
                    "single_future_consensus_cosine": stats(single),
                    "eta03_future_consensus_cosine": stats(short),
                    "candidate_future_consensus_cosine": stats(candidate),
                    "paired_cosine_change_vs_eta03": stats(delta),
                    "bootstrap_median_improvement_95_ci": bootstrap_ci(
                        delta, 1000 * arm_index + 10 * depth
                    ),
                    "win_fraction": float(np.mean(delta > 0.0)),
                    "candidate_future_mean_q": stats([
                        row["depths"][key]["candidates"][name]["future_mean_q"]
                        for row in rows
                    ]),
                    "candidate_norm_constraint_scale": stats([
                        row["depths"][key]["candidates"][name][
                            "norm_constraint_scale"
                        ]
                        for row in rows
                    ]),
                }
                if arm in WSSR_ARMS:
                    compression = [
                        row["depths"][key]["step_diagnostics"]["compression"]
                        for row in rows
                    ]
                    labels = (
                        "compressed_force_relative_error",
                        "compressed_force_cosine",
                        "history_current_force_norm_ratio",
                        "history_operator_mass_fraction",
                        "old_force_projection_retention",
                        "leading_singular_value",
                    )
                    depth_results[key]["compression"] = {
                        label: stats([values[index] for values in compression])
                        for index, label in enumerate(labels)
                    }
                    depth_results[key]["compression"][
                        "history_spectral_quality_retention"
                    ] = stats([
                        row["depths"][key]["step_diagnostics"][
                            "history_spectral_quality_retention"
                        ]
                        for row in rows
                    ])

            k16 = depth_results["16"]
            if arm in WSSR_ARMS:
                compression = k16["compression"]
                compression_passed = bool(
                    compression["compressed_force_cosine"]["median"] >= 0.98
                    and compression["compressed_force_relative_error"]["median"]
                    <= 0.05
                    and compression[
                        "history_spectral_quality_retention"
                    ]["median"] >= 0.80
                )
            else:
                compression_passed = True
            gate = {
                "median_cosine_change_vs_eta03_at_k16": k16[
                    "paired_cosine_change_vs_eta03"
                ]["median"],
                "bootstrap_lower_bound_at_least_minus_0p03": (
                    k16["bootstrap_median_improvement_95_ci"][0] >= -0.03
                ),
                "severe_harm_fraction_at_most_1_of_6": float(np.mean(
                    np.asarray([
                        row["depths"]["16"]["candidates"][name][
                            "future_consensus_cosine"
                        ]
                        for row in rows
                    ])
                    - np.asarray([
                        row["depths"]["16"]["candidates"]["fixed"][
                            "future_consensus_cosine"
                        ]
                        for row in short_rows
                    ])
                    < -0.02
                )) <= 1.0 / 6.0,
                "compression_passed": compression_passed,
            }
            gate["passed"] = bool(
                arm in WSSR_ARMS
                and arm != "wssr_eta03"
                and gate["median_cosine_change_vs_eta03_at_k16"] >= -0.01
                and gate["bootstrap_lower_bound_at_least_minus_0p03"]
                and gate["severe_harm_fraction_at_most_1_of_6"]
                and gate["compression_passed"]
            )
            results[arm] = {
                "depths": depth_results,
                "gate": gate,
            }
            if gate["passed"]:
                passing.append((
                    k16["candidate_future_consensus_cosine"]["median"],
                    arm,
                ))

    passing.sort(reverse=True)
    selected = None
    if passing:
        cosine, arm = passing[0]
        selected = {
            "arm": arm,
            "regularization": "fixed_lambda_0.001_mixed_precision",
            "median_future_consensus_cosine_at_k16": cosine,
        }
        conclusion = "long_memory_wssr_passed_direction_gate"
        next_action = "Implement the selected arm as a matched two-seed E5000 test."
    else:
        conclusion = "long_memory_wssr_failed_direction_gate"
        next_action = (
            "Stop operator-EMA training candidates; use the SPRING comparator to "
            "decide whether solution-space recurrence remains justified."
        )

    output = {
        "experiment": "C rank1600 early memory-scaling collection",
        "source_checkpoint": payloads["single_current"]["source_checkpoint"],
        "replicates": 6,
        "primary_metric": "future_consensus_force_cosine",
        "results": results,
        "passing_wssr_candidates": [
            {"arm": arm, "future_consensus_cosine": value}
            for value, arm in passing
        ],
        "eta08_beats_eta03_at_fixed_lambda_k16": bool(
            results["wssr_eta08"]["depths"]["16"][
                "candidate_future_consensus_cosine"
            ]["median"]
            > results["wssr_eta03"]["depths"]["16"][
                "candidate_future_consensus_cosine"
            ]["median"]
        ),
        "selected_candidate": selected,
        "conclusion": conclusion,
        "next_action": next_action,
    }
    summary_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# C rank-1600 early memory-scaling audit",
        "",
        f"Conclusion: **{conclusion}**",
        "",
        "| arm | K=3 cos | K=8 cos | K=16 cos | "
        "K=16 delta vs eta03 | 95% CI | compression | gate |",
        "|---|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for arm, item in results.items():
            depths = item["depths"]
            gate = item["gate"]
            ci = depths["16"]["bootstrap_median_improvement_95_ci"]
            lines.append(
                f"| {arm} | "
                f"{depths['3']['candidate_future_consensus_cosine']['median']:.4f} | "
                f"{depths['8']['candidate_future_consensus_cosine']['median']:.4f} | "
                f"{depths['16']['candidate_future_consensus_cosine']['median']:.4f} | "
                f"{depths['16']['paired_cosine_change_vs_eta03']['median']:+.4f} | "
                f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | "
                f"{'PASS' if gate['compression_passed'] else 'FAIL'} | "
                f"{'PASS' if gate['passed'] else 'FAIL'} |"
            )
    lines += ["", next_action, ""]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
