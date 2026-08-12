#!/usr/bin/env python3
"""Compare matched-size WSSR and SPRING direction-consistency audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def bootstrap_median_difference(spring, wssr, *, seed, replicates=20000):
    spring = np.asarray(spring, dtype=np.float64)
    wssr = np.asarray(wssr, dtype=np.float64)
    rng = np.random.default_rng(seed)
    si = rng.integers(0, spring.size, size=(replicates, spring.size))
    wi = rng.integers(0, wssr.size, size=(replicates, wssr.size))
    delta = np.median(spring[si], axis=1) - np.median(wssr[wi], axis=1)
    return float(np.median(spring) - np.median(wssr)), [
        float(x) for x in np.quantile(delta, [0.025, 0.975])
    ]


def column(payload, name):
    return [row[name] for row in payload["per_batch"] if row[name] is not None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wssr", type=Path, required=True)
    parser.add_argument("--spring", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    wssr = json.loads(args.wssr.read_text())
    spring = json.loads(args.spring.read_text())
    if wssr["batches"] != spring["batches"]:
        raise ValueError("batch counts must match")

    comparisons = {}
    for index, metric in enumerate(
        (
            "consecutive_direction_cosine",
            "previous_direction_heldout_force_cosine",
        )
    ):
        difference, ci = bootstrap_median_difference(
            column(spring, metric), column(wssr, metric), seed=20260802 + index
        )
        comparisons[metric] = {
            "spring_minus_wssr_median_difference": difference,
            "bootstrap_95_ci": ci,
        }

    wa = wssr["aggregate"]
    sa = spring["aggregate"]
    resultant_ratio = sa["unit_direction_resultant_length"] / max(
        wa["unit_direction_resultant_length"], 1.0e-12
    )
    snr_ratio = sa["raw_direction_snr"] / max(wa["raw_direction_snr"], 1.0e-12)
    consecutive_supported = (
        comparisons["consecutive_direction_cosine"]["bootstrap_95_ci"][0] > 0.0
    )
    heldout_supported = (
        comparisons["previous_direction_heldout_force_cosine"][
            "bootstrap_95_ci"
        ][0]
        > 0.0
    )
    resultant_supported = resultant_ratio >= 1.25
    signal_count = sum(
        (consecutive_supported, heldout_supported, resultant_supported)
    )
    payload = {
        "experiment": "C WSSR versus SPRING direction-consistency comparison",
        "wssr_source": str(args.wssr),
        "spring_source": str(args.spring),
        "comparisons": comparisons,
        "unit_direction_resultant_ratio_spring_over_wssr": resultant_ratio,
        "raw_direction_snr_ratio_spring_over_wssr": snr_ratio,
        "positive_heldout_fraction": {
            "wssr": wa["positive_heldout_force_cosine_fraction"],
            "spring": sa["positive_heldout_force_cosine_fraction"],
        },
        "pre_registered_gate": {
            "consecutive_cosine_ci_lower_above_zero": True,
            "heldout_force_cosine_ci_lower_above_zero": True,
            "unit_resultant_ratio_min": 1.25,
            "signals_required": 2,
        },
        "signals": {
            "consecutive_cosine": consecutive_supported,
            "heldout_force_alignment": heldout_supported,
            "unit_resultant": resultant_supported,
        },
        "signal_count": signal_count,
        "spring_direction_consistency_advantage_supported": signal_count >= 2,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
