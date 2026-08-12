#!/usr/bin/env python3
"""Collect the early fixed-lambda single-vs-multibatch audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


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


def summarize(payload, depth):
    rows = [row["depths"][str(depth)] for row in payload["per_replicate"]]
    single_q = np.asarray(
        [row["single_batch_fixed_lambda"]["heldout_q"] for row in rows]
    )
    multi_q = np.asarray(
        [row["multibatch_fixed_lambda"]["heldout_q"] for row in rows]
    )
    improvement = (single_q - multi_q) / single_q
    active = single_q < 1.0
    positive_force = np.asarray(
        [
            row["multibatch_fixed_lambda"]["heldout_force_dot_update"] > 0.0
            for row in rows
        ]
    )
    result = {
        "single_heldout_q": stats(single_q),
        "multibatch_heldout_q": stats(multi_q),
        "multibatch_relative_improvement": stats(improvement),
        "bootstrap_median_improvement_95_ci": bootstrap_ci(improvement, depth),
        "single_active_signal_fraction": float(np.mean(active)),
        "multibatch_win_fraction": float(np.mean(improvement > 0.0)),
        "multibatch_positive_force_fraction": float(np.mean(positive_force)),
        "single_multi_direction_cosine": stats(
            [row["single_multi_direction_cosine"] for row in rows]
        ),
        "single_norm_constraint_scale": stats(
            [
                row["single_batch_fixed_lambda"]["norm_constraint_scale"]
                for row in rows
            ]
        ),
        "multibatch_norm_constraint_scale": stats(
            [
                row["multibatch_fixed_lambda"]["norm_constraint_scale"]
                for row in rows
            ]
        ),
    }
    result["passed"] = bool(
        result["single_active_signal_fraction"] >= 4.0 / 6.0
        and result["multibatch_relative_improvement"]["median"] >= 0.10
        and result["bootstrap_median_improvement_95_ci"][0] > 0.0
        and result["multibatch_win_fraction"] >= 5.0 / 6.0
        and result["multibatch_positive_force_fraction"] >= 5.0 / 6.0
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    audit_path = args.root / "audit.json"
    summary_path = args.root / "summary.json"
    report_path = args.root / "report.md"
    if summary_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite collector output")
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    depths = {str(depth): summarize(payload, depth) for depth in (2, 3)}
    passed = depths["3"]["passed"]
    conclusion = (
        "early_fixed_lambda_multibatch_candidate_passed"
        if passed
        else "early_fixed_lambda_multibatch_candidate_failed"
    )
    output = {
        "experiment": payload["experiment"],
        "source_checkpoint": payload["source_checkpoint"],
        "replicates": payload["replicates"],
        "depths": depths,
        "stage2_passed": passed,
        "conclusion": conclusion,
        "next_action": (
            "Run the pre-registered 2x2-seed E500 baseline-vs-candidate test."
            if passed
            else "Stop this multibatch route; do not train the candidate."
        ),
    }
    summary_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Early fixed-lambda single-vs-multibatch WSSR audit",
        "",
        f"Conclusion: **{conclusion}**",
        "",
        "| K | single q | multibatch q | improvement | 95% CI | active | wins | gate |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for depth in (2, 3):
        row = depths[str(depth)]
        ci = row["bootstrap_median_improvement_95_ci"]
        lines.append(
            f"| {depth} | {row['single_heldout_q']['median']:.6f} | "
            f"{row['multibatch_heldout_q']['median']:.6f} | "
            f"{row['multibatch_relative_improvement']['median']:.2%} | "
            f"[{ci[0]:.2%}, {ci[1]:.2%}] | "
            f"{row['single_active_signal_fraction']:.0%} | "
            f"{row['multibatch_win_fraction']:.0%} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |"
        )
    lines += [
        "",
        output["next_action"],
        "",
        "This is a read-only fixed-parameter audit, not a frozen VMC result.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
