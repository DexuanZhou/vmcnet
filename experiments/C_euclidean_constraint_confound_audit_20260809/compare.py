#!/usr/bin/env python3
"""Compare completed SPRING/WSSR Euclidean-constraint audits."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path("/scratch/dexuan1/runs/C_euclidean_constraint_confound_audit_20260809")


def geometric_mean(values):
    return float(np.exp(np.mean(np.log(np.asarray(values, dtype=np.float64)))))


def bootstrap_geomean_ci(values, seed=20260809, replicates=20000):
    logs = np.log(np.asarray(values, dtype=np.float64))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(logs), size=(replicates, len(logs)))
    means = np.mean(logs[indices], axis=1)
    lower, upper = np.quantile(np.exp(means), [0.025, 0.975])
    return [float(lower), float(upper)]


def main():
    payloads = {}
    for method in ("spring", "wssr"):
        path = ROOT / f"stored_{method}" / "audit.json"
        payloads[method] = json.loads(path.read_text(encoding="utf-8"))

    records = [
        row for method in ("spring", "wssr") for row in payloads[method]["records"]
    ]
    csv_path = ROOT / "rho_trajectory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    epochs = sorted({row["epoch"] for row in records})
    per_epoch = []
    paired_gain_ratios = []
    paired_actual_ratios = []
    paired_parameter_step_ratios = []
    for epoch in epochs:
        selected = {
            method: [
                row
                for row in payloads[method]["records"]
                if row["epoch"] == epoch
            ]
            for method in ("spring", "wssr")
        }
        if len(selected["spring"]) != len(selected["wssr"]):
            raise ValueError(f"unpaired batch counts at epoch {epoch}")
        gain_s = np.asarray(
            [row["function_gain_O_p_hat"] for row in selected["spring"]]
        )
        gain_w = np.asarray(
            [row["function_gain_O_p_hat"] for row in selected["wssr"]]
        )
        actual_s = np.asarray(
            [row["actual_function_step_norm"] for row in selected["spring"]]
        )
        actual_w = np.asarray(
            [row["actual_function_step_norm"] for row in selected["wssr"]]
        )
        parameter_s = np.asarray(
            [row["actual_update_norm"] for row in selected["spring"]]
        )
        parameter_w = np.asarray(
            [row["actual_update_norm"] for row in selected["wssr"]]
        )
        paired_gain_ratios.extend((gain_w / gain_s).tolist())
        paired_actual_ratios.extend((actual_w / actual_s).tolist())
        paired_parameter_step_ratios.extend((parameter_w / parameter_s).tolist())
        spring_direction_norm = float(selected["spring"][0]["direction_norm"])
        wssr_direction_norm = float(selected["wssr"][0]["direction_norm"])
        schedule_multiplier = 1.0 + 1.0e-4 * (epoch - 1)
        per_epoch.append(
            {
                "epoch": epoch,
                "spring_gain_median": float(np.median(gain_s)),
                "wssr_gain_median": float(np.median(gain_w)),
                "gain_ratio_wssr_over_spring": float(
                    np.median(gain_w) / np.median(gain_s)
                ),
                "spring_rho_median": float(np.median(1.0 / gain_s)),
                "wssr_rho_median": float(np.median(1.0 / gain_w)),
                "spring_actual_function_step_median": float(np.median(actual_s)),
                "wssr_actual_function_step_median": float(np.median(actual_w)),
                "actual_step_ratio_wssr_over_spring": float(
                    np.median(actual_w) / np.median(actual_s)
                ),
                "spring_actual_parameter_step": float(np.median(parameter_s)),
                "wssr_actual_parameter_step": float(np.median(parameter_w)),
                "parameter_step_ratio_wssr_over_spring": float(
                    np.median(parameter_w) / np.median(parameter_s)
                ),
                "spring_base_lr_uncap_threshold": float(
                    math.sqrt(0.001) * schedule_multiplier / spring_direction_norm
                ),
                "wssr_base_lr_uncap_threshold": float(
                    math.sqrt(0.001) * schedule_multiplier / wssr_direction_norm
                ),
                "spring_cap_active_fraction": float(
                    np.mean([row["constraint_active"] for row in selected["spring"]])
                ),
                "wssr_cap_active_fraction": float(
                    np.mean([row["constraint_active"] for row in selected["wssr"]])
                ),
            }
        )

    aggregate = {}
    for method in ("spring", "wssr"):
        median_gains = [row[f"{method}_gain_median"] for row in per_epoch]
        aggregate[method] = {
            "function_gain_trajectory_max_over_min": float(
                max(median_gains) / min(median_gains)
            ),
            "constraint_active_fraction": payloads[method]["aggregate"][
                "constraint_active_fraction"
            ],
        }
    aggregate["paired_wssr_over_spring"] = {
        "function_gain_geometric_mean_ratio": geometric_mean(paired_gain_ratios),
        "function_gain_bootstrap_95pct_ci": bootstrap_geomean_ci(
            paired_gain_ratios
        ),
        "actual_function_step_geometric_mean_ratio": geometric_mean(
            paired_actual_ratios
        ),
        "actual_function_step_bootstrap_95pct_ci": bootstrap_geomean_ci(
            paired_actual_ratios, seed=20260810
        ),
        "actual_parameter_step_geometric_mean_ratio": geometric_mean(
            paired_parameter_step_ratios
        ),
    }
    output = {
        "per_epoch": per_epoch,
        "aggregate": aggregate,
        "csv": str(csv_path),
    }
    (ROOT / "comparison.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# C Euclidean norm-constraint confound audit",
        "",
        "Both source checkpoints are read only. `gain` is "
        r"$\|\bar O\hat p\|_2$ and `rho` is its inverse.",
        "",
        "| epoch | SPRING gain | WSSR gain | W/S gain | SPRING rho | WSSR rho | "
        "SPRING actual function step | WSSR actual function step | W/S actual | "
        "cap active S/W |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_epoch:
        lines.append(
            f"| {row['epoch']} | {row['spring_gain_median']:.6g} | "
            f"{row['wssr_gain_median']:.6g} | "
            f"{row['gain_ratio_wssr_over_spring']:.4f} | "
            f"{row['spring_rho_median']:.6g} | {row['wssr_rho_median']:.6g} | "
            f"{row['spring_actual_function_step_median']:.6g} | "
            f"{row['wssr_actual_function_step_median']:.6g} | "
            f"{row['actual_step_ratio_wssr_over_spring']:.4f} | "
            f"{row['spring_cap_active_fraction']:.0%}/"
            f"{row['wssr_cap_active_fraction']:.0%} |"
        )
    paired = aggregate["paired_wssr_over_spring"]
    lines.extend(
        [
            "",
            "| epoch | SPRING actual parameter step | WSSR actual parameter step | "
            "W/S parameter step | SPRING base-lr uncap threshold | "
            "WSSR base-lr uncap threshold |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in per_epoch:
        lines.append(
            f"| {row['epoch']} | {row['spring_actual_parameter_step']:.6g} | "
            f"{row['wssr_actual_parameter_step']:.6g} | "
            f"{row['parameter_step_ratio_wssr_over_spring']:.4f} | "
            f"{row['spring_base_lr_uncap_threshold']:.6g} | "
            f"{row['wssr_base_lr_uncap_threshold']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "- WSSR/SPRING function-gain geometric mean ratio: "
            f"{paired['function_gain_geometric_mean_ratio']:.4f} "
            f"(batch bootstrap 95% CI "
            f"{paired['function_gain_bootstrap_95pct_ci'][0]:.4f}–"
            f"{paired['function_gain_bootstrap_95pct_ci'][1]:.4f}).",
            "- WSSR/SPRING actual function-step geometric mean ratio: "
            f"{paired['actual_function_step_geometric_mean_ratio']:.4f} "
            f"(95% CI {paired['actual_function_step_bootstrap_95pct_ci'][0]:.4f}–"
            f"{paired['actual_function_step_bootstrap_95pct_ci'][1]:.4f}).",
            "- WSSR/SPRING actual Euclidean parameter-step geometric mean ratio: "
            f"{paired['actual_parameter_step_geometric_mean_ratio']:.4f}.",
            "- Cap-active fraction over all audited points: SPRING "
            f"{aggregate['spring']['constraint_active_fraction']:.1%}; WSSR "
            f"{aggregate['wssr']['constraint_active_fraction']:.1%}.",
            "- Function-gain trajectory max/min: SPRING "
            f"{aggregate['spring']['function_gain_trajectory_max_over_min']:.3f}; "
            f"WSSR {aggregate['wssr']['function_gain_trajectory_max_over_min']:.3f}.",
            "",
            "The gain ratio quantifies the metric mismatch. It does not by itself "
            "identify a causal fraction of the frozen-variance ratio; that would "
            "require a matched rerun under a common function-space constraint.",
        ]
    )
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
