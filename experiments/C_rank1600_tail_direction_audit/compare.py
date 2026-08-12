#!/usr/bin/env python3
"""Compare the matched WSSR and SPRING fixed-checkpoint tail audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wssr", type=Path, required=True)
    parser.add_argument("--spring", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    wssr = json.loads(args.wssr.read_text(encoding="utf-8"))
    spring = json.loads(args.spring.read_text(encoding="utf-8"))
    wa, sa = wssr["aggregate"], spring["aggregate"]
    variance_ratio = wa["raw_variance"]["median"] / sa["raw_variance"]["median"]
    maxz_ratio = wa["raw_max_robust_z"]["median"] / sa["raw_max_robust_z"]["median"]
    shared = bool(spring["classification"]["shared_clipping_confirmed"])
    truncation = bool(wssr["classification"]["hard_truncation_supported"])
    if shared and not truncation and variance_ratio > 1.25:
        conclusion = "long_trajectory_parameter_basin_supported"
    elif shared and not truncation:
        conclusion = "shared_clipping_but_basin_difference_not_resolved"
    else:
        conclusion = "optimizer_update_mechanism_requires_followup"
    payload = {
        "experiment": "C WSSR versus SPRING tail-mechanism decision",
        "wssr_source": str(args.wssr),
        "spring_source": str(args.spring),
        "wssr_median_raw_variance_over_spring": variance_ratio,
        "wssr_median_max_robust_z_over_spring": maxz_ratio,
        "hard_truncation_supported": truncation,
        "shared_clipping_confirmed": shared,
        "conclusion": conclusion,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
