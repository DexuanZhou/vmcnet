#!/usr/bin/env python3
"""Collect the same-checkpoint lambda crossover and apply its fixed gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SEED = 20260802


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def bootstrap_median_ci(values, offset):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(SEED + offset)
    draws = rng.integers(0, len(values), (10000, len(values)))
    return [float(x) for x in np.quantile(np.median(values[draws], axis=1), (0.025, 0.975))]


def gate(payload, depth, offset):
    rows = [row["depths"][str(depth)] for row in payload["per_replicate"]]
    dynamic_q = np.asarray(
        [row["metrics"]["dynamic_lambda"]["heldout_q"] for row in rows]
    )
    fixed_q = np.asarray(
        [row["metrics"]["fixed_lambda"]["heldout_q"] for row in rows]
    )
    improvement = (dynamic_q - fixed_q) / dynamic_q
    positive_force = np.asarray(
        [
            row["metrics"]["fixed_lambda"]["heldout_force_dot_update"] > 0.0
            for row in rows
        ]
    )
    result = {
        "dynamic_heldout_q": summarize(dynamic_q),
        "fixed_heldout_q": summarize(fixed_q),
        "fixed_relative_improvement": summarize(improvement),
        "bootstrap_median_95_ci": bootstrap_median_ci(improvement, offset),
        "win_fraction": float(np.mean(improvement > 0.0)),
        "positive_fixed_force_fraction": float(np.mean(positive_force)),
        "fixed_dynamic_direction_cosine": summarize(
            [row["metrics"]["fixed_dynamic_direction_cosine"] for row in rows]
        ),
        "dynamic_lambda": summarize(
            [row["step_diagnostics"]["dynamic_lambda"] for row in rows]
        ),
        "active_rank": summarize(
            [row["step_diagnostics"]["active_rank"] for row in rows]
        ),
        "history_factor_delta_between_lambdas": summarize(
            [
                row["step_diagnostics"]["history_factor_delta_between_lambdas"]
                for row in rows
            ]
        ),
        "history_rhs_delta_between_lambdas": summarize(
            [
                row["step_diagnostics"]["history_rhs_delta_between_lambdas"]
                for row in rows
            ]
        ),
    }
    result["passed"] = bool(
        result["fixed_relative_improvement"]["median"] >= 0.05
        and result["bootstrap_median_95_ci"][0] > 0.0
        and result["win_fraction"] >= 0.75
        and result["positive_fixed_force_fraction"] >= 0.95
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.root / "summary.json"
    report_path = args.root / "summary.md"
    if summary_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite existing summary")

    results = []
    for index, label in enumerate(("legacy_epoch100000", "fixedlambda_epoch102000")):
        path = args.root / label / "audit.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        results.append(
            {
                "label": label,
                "source_checkpoint": payload["source_checkpoint"],
                "replicates": payload["replicates"],
                "elapsed_seconds": payload["elapsed_seconds"],
                "depths": {
                    str(depth): gate(payload, depth, 100 * index + depth)
                    for depth in (2, 3)
                },
            }
        )

    k3_all = all(item["depths"]["3"]["passed"] for item in results)
    conclusion = (
        "fixed_lambda_causally_stabilizes_production_ssi"
        if k3_all
        else "lambda_alone_not_sufficient"
    )
    next_action = (
        "No further offline compression/SSI tuning; use fixed lambda as the "
        "numerically justified WSSR control and evaluate training separately."
        if k3_all
        else "Audit warm iterations 4/8/40 at matched lambda before any training."
    )
    output = {
        "experiment": "C rank1600 same-checkpoint lambda crossover collection",
        "k3_passed_both_checkpoint_states": k3_all,
        "conclusion": conclusion,
        "next_action": next_action,
        "per_checkpoint": results,
    }
    summary_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# C rank-1600 same-checkpoint lambda crossover",
        "",
        f"Conclusion: **{conclusion}**",
        "",
        f"Next action: {next_action}",
        "",
        "| checkpoint | K | dynamic q | fixed q | fixed improvement | 95% CI | gate |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for item in results:
        for depth in (2, 3):
            value = item["depths"][str(depth)]
            ci = value["bootstrap_median_95_ci"]
            lines.append(
                f"| {item['label']} | {depth} | "
                f"{value['dynamic_heldout_q']['median']:.6f} | "
                f"{value['fixed_heldout_q']['median']:.6f} | "
                f"{value['fixed_relative_improvement']['median']:.2%} | "
                f"[{ci[0]:.2%}, {ci[1]:.2%}] | "
                f"{'PASS' if value['passed'] else 'FAIL'} |"
            )
    lines.extend(
        [
            "",
            "This is a fixed-parameter inverse-solve diagnostic, not a frozen VMC "
            "energy/variance result.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
