#!/usr/bin/env python3
"""Recover the completed eight-replicate audit after final provenance failure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FIELDS = (
    "actual_delta_norm",
    "old_error",
    "transport_error",
    "transport_error_ratio",
    "old_cosine",
    "transport_cosine",
    "sampling_error",
    "sampling_to_staleness_ratio",
    "old_total_batch_error",
    "transport_total_batch_error",
)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def bootstrap_median_ci(values, seed=20260809, draws=10000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    return [float(x) for x in np.quantile(np.median(values[indices], axis=1), [0.025, 0.975])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for line in args.stdout.read_text().splitlines():
        if line.startswith("{") and '"replicate"' in line:
            rows.append(json.loads(line))
    rows.sort(key=lambda row: row["replicate"])
    if [row["replicate"] for row in rows] != list(range(8)):
        raise ValueError(f"expected replicates 0..7, found {[r['replicate'] for r in rows]}")
    summaries = {
        field: summarize([row[field] for row in rows]) for field in FIELDS
    }
    transport_ratios = np.asarray(
        [row["transport_error_ratio"] for row in rows], dtype=np.float64
    )
    sampling_ratios = np.asarray(
        [row["sampling_to_staleness_ratio"] for row in rows], dtype=np.float64
    )
    gate = {
        "transport_error_ratio_required_max": 0.5,
        "sampling_to_staleness_ratio_required_max": 2.0,
        "transport_win_fraction_required_min": 0.75,
        "median_transport_error_ratio": float(np.median(transport_ratios)),
        "median_sampling_to_staleness_ratio": float(np.median(sampling_ratios)),
        "transport_win_fraction": float(np.mean(transport_ratios < 1.0)),
        "transport_error_ratio_bootstrap_95_ci": bootstrap_median_ci(
            transport_ratios
        ),
    }
    gate["passed"] = bool(
        gate["median_transport_error_ratio"] <= 0.5
        and gate["median_sampling_to_staleness_ratio"] <= 2.0
        and gate["transport_win_fraction"] >= 0.75
    )
    old_total = np.asarray(
        [row["old_total_batch_error"] for row in rows], dtype=np.float64
    )
    transported_total = np.asarray(
        [row["transport_total_batch_error"] for row in rows], dtype=np.float64
    )
    result = {
        "experiment": "early-C one-step curvature transport action audit",
        "recovered_from_complete_stdout": True,
        "slurm_job": 53763134,
        "slurm_final_failure_scope": "post-replicate git provenance lookup only",
        "read_only_checkpoint": True,
        "source_checkpoint": "/scratch/dexuan1/runs/C_wssr_split_eta_E5000_20260803/train/D_etaS0_etaG0/seed0/checkpoints/1000.npz",
        "replicates": 8,
        "walkers": 1000,
        "transport_definition": "2*S(theta+delta/2)-S(theta)",
        "delta_definition": "actual post-lr/post-norm-constraint parameter update",
        "summaries": summaries,
        "new_batch_total_error_relative_improvement": summarize(
            (old_total - transported_total) / old_total
        ),
        "pre_registered_gate": gate,
        "decision": (
            "stop_before_training: first-order transport accurately removes the "
            "small same-coordinate parameter-staleness component, but ordinary "
            "batch sampling variation is dominant"
        ),
        "records": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
