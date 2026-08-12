#!/usr/bin/env python3
"""Collect and gate the fixed-theta history-compression audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


BOOTSTRAP_SEED = 20260802
BOOTSTRAP_REPLICATES = 10000
CANDIDATES = (
    "raw_rank1600_exact",
    "recursive_factor_rank1600_exact",
    "recursive_production_exact",
    "recursive_production_ssi",
)


def median_ci(values, seed_offset):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    draws = rng.integers(0, len(values), (BOOTSTRAP_REPLICATES, len(values)))
    medians = np.median(values[draws], axis=1)
    return [float(x) for x in np.quantile(medians, (0.025, 0.975))]


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def collect_depth(payload, depth, seed_offset):
    rows = [row["depths"][str(depth)] for row in payload["per_replicate"]]
    single_q = np.asarray(
        [row["metrics"]["single_current_exact"]["heldout_q"] for row in rows]
    )
    raw_q = np.asarray(
        [row["metrics"]["raw_full_exact"]["heldout_q"] for row in rows]
    )
    raw_improvement = (single_q - raw_q) / single_q
    raw_gate = {
        "relative_heldout_q_improvement": summary(raw_improvement),
        "bootstrap_median_95_ci": median_ci(raw_improvement, seed_offset),
        "win_fraction": float(np.mean(raw_improvement > 0.0)),
    }
    raw_gate["passed"] = bool(
        raw_gate["relative_heldout_q_improvement"]["median"] >= 0.05
        and raw_gate["bootstrap_median_95_ci"][0] > 0.0
        and raw_gate["win_fraction"] >= 0.75
    )

    denominator = single_q - raw_q
    positive_denom = denominator > 0.0
    candidate_gates = {}
    for candidate_index, candidate in enumerate(CANDIDATES):
        candidate_q = np.asarray(
            [row["metrics"][candidate]["heldout_q"] for row in rows]
        )
        retention = (single_q[positive_denom] - candidate_q[positive_denom]) / (
            denominator[positive_denom]
        )
        cosines = np.asarray(
            [
                row["metrics"][candidate]["direction_cosine_to_raw_full"]
                for row in rows
            ]
        )
        positive_force = np.asarray(
            [
                row["metrics"][candidate]["heldout_force_dot_update"] > 0.0
                for row in rows
            ]
        )
        q_relative_to_raw = (candidate_q - raw_q) / raw_q
        gate = {
            "positive_raw_gain_denominator_count": int(np.sum(positive_denom)),
            "retained_gain": summary(retention) if len(retention) else None,
            "direction_cosine_to_raw_full": summary(cosines),
            "positive_heldout_force_fraction": float(np.mean(positive_force)),
            "relative_q_excess_over_raw_full": summary(q_relative_to_raw),
        }
        gate["passed"] = bool(
            len(retention) > 0
            and gate["retained_gain"]["median"] >= 0.80
            and gate["direction_cosine_to_raw_full"]["median"] >= 0.98
            and gate["positive_heldout_force_fraction"] >= 0.95
        )
        candidate_gates[candidate] = gate

    secondary = {}
    for candidate in CANDIDATES:
        diagnostics = [row["state_diagnostics"][candidate] for row in rows]
        secondary[candidate] = {
            key: summary([item[key] for item in diagnostics])
            for key in diagnostics[0]
        }
    return {
        "raw_multibatch_gate": raw_gate,
        "candidate_gates": candidate_gates,
        "secondary_state_diagnostics": secondary,
        "heldout_q": {
            name: summary([row["metrics"][name]["heldout_q"] for row in rows])
            for name in (
                "single_current_exact",
                "raw_full_exact",
                *CANDIDATES,
            )
        },
        "active_rank": {
            key: summary([row[key] for row in rows])
            for key in (
                "raw_full_active_rank",
                "raw_rank1600_active_rank",
                "recursive_factor_active_rank",
                "recursive_production_exact_active_rank",
                "recursive_production_ssi_active_rank",
            )
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    output = args.root / "summary.json"
    report = args.root / "summary.md"
    if output.exists() or report.exists():
        raise FileExistsError("refusing to overwrite an existing summary")

    paths = (
        args.root / "legacy_epoch100000" / "audit.json",
        args.root / "fixedlambda_epoch102000" / "audit.json",
    )
    checkpoint_results = []
    for checkpoint_index, path in enumerate(paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        checkpoint_results.append(
            {
                "label": path.parent.name,
                "source_checkpoint": payload["source_checkpoint"],
                "replicates": payload["replicates"],
                "spectral_controls": payload["spectral_controls"],
                "elapsed_seconds": payload["elapsed_seconds"],
                "depths": {
                    str(depth): collect_depth(
                        payload, depth, 100 * checkpoint_index + depth
                    )
                    for depth in (2, 3)
                },
            }
        )

    k3 = [result["depths"]["3"] for result in checkpoint_results]
    raw_all = all(result["raw_multibatch_gate"]["passed"] for result in k3)
    passes = {
        candidate: all(
            result["candidate_gates"][candidate]["passed"] for result in k3
        )
        for candidate in CANDIDATES
    }
    if not raw_all:
        branch = "raw_full_not_robust"
        next_test = "Stop; reconcile with the earlier rank-1000 two-batch gate."
    elif not passes["raw_rank1600_exact"]:
        branch = "one_shot_rank_cap"
        next_test = "Audit one-shot ranks 2400 and 3200; no training."
    elif not passes["recursive_factor_rank1600_exact"]:
        branch = "recursive_compression"
        next_test = "Audit K=5/8 with storage ranks 1600 and 3200; no training."
    elif not passes["recursive_production_exact"]:
        branch = "production_state_masking"
        next_test = "Audit unfiltered versus configured-cutoff history state only."
    elif not passes["recursive_production_ssi"]:
        branch = "warm_ssi_approximation"
        next_test = "Audit warm iterations 4, 8, and 40 offline; no training."
    else:
        branch = "compression_not_bottleneck"
        next_test = (
            "Stop compression experiments; compare recurrence/update semantics or "
            "long-trajectory statistics."
        )

    result = {
        "experiment": "C rank1600 history-compression fidelity collection",
        "pre_registered_branch_depth": 3,
        "checkpoint_count": len(checkpoint_results),
        "raw_multibatch_passed_both_checkpoints": raw_all,
        "candidate_passed_both_checkpoints": passes,
        "selected_branch": branch,
        "next_test": next_test,
        "per_checkpoint": checkpoint_results,
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# C rank-1600 history-compression fidelity audit",
        "",
        f"Selected branch: **{branch}**",
        "",
        f"Next action: {next_test}",
        "",
        "| checkpoint | K | raw gain | raw gate | rank1600 R | recursive R | "
        "production-exact R | production-SSI R |",
        "|---|---:|---:|:---:|---:|---:|---:|---:|",
    ]
    for checkpoint in checkpoint_results:
        for depth in (2, 3):
            item = checkpoint["depths"][str(depth)]
            raw = item["raw_multibatch_gate"]
            gates = item["candidate_gates"]
            retained = []
            for candidate in CANDIDATES:
                value = gates[candidate]["retained_gain"]
                retained.append("unknown" if value is None else f"{value['median']:.4f}")
            lines.append(
                f"| {checkpoint['label']} | {depth} | "
                f"{raw['relative_heldout_q_improvement']['median']:.4f} | "
                f"{'PASS' if raw['passed'] else 'FAIL'} | "
                + " | ".join(retained)
                + " |"
            )
    lines.extend(
        [
            "",
            "All values are fixed-parameter replay diagnostics. They are not frozen "
            "VMC energies or variances and do not establish a physical-energy result.",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
