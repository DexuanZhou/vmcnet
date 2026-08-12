#!/usr/bin/env python3
"""Create a compact report from the fixed-checkpoint tail-direction audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    data = json.loads(args.input.read_text(encoding="utf-8"))
    classification = data["classification"]
    aggregate = data["aggregate"]
    lines = [
        "# C rank1600 fixed-checkpoint tail-direction audit",
        "",
        "No parameters or optimizer state were updated. MCMC walkers were advanced only in memory.",
        "",
        f"Conclusion: **{classification['conclusion']}**.",
        "",
        "| metric | median | mean | min | max |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in (
        "raw_variance",
        "raw_max_robust_z",
        "raw_tail_active_leakage",
        "clipped_tail_active_leakage",
        "tail_clipped_force_over_raw_force",
        "raw_tail_alignment_wssr",
        "raw_tail_alignment_minsr_clipped",
        "raw_tail_alignment_minsr_raw",
        "raw_tail_alignment_wssr_plus_tail",
        "raw_tail_q_wssr",
        "raw_tail_q_minsr_clipped",
        "raw_tail_q_minsr_raw",
        "raw_tail_q_wssr_plus_tail",
    ):
        row = aggregate[name]
        lines.append(
            f"| {name} | {row['median']:.6g} | {row['mean']:.6g} | "
            f"{row['min']:.6g} | {row['max']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Pre-registered decision values",
            "",
            f"- Median tail leakage: {classification['median_raw_tail_active_leakage']:.6g}",
            f"- Median Q(WSSR)-Q(MinSR clipped): {classification['median_q_wssr_minus_minsr_clipped']:.6g}",
            f"- Median Q(MinSR clipped)-Q(MinSR raw): {classification['median_q_minsr_clipped_minus_minsr_raw']:.6g}",
            f"- Median one-tail-basis recovery of the WSSR/MinSR gap: {classification['median_tail_basis_gap_recovery']:.6g}",
            "",
            "The one-vector tail extension uses the unchanged clipped, history-augmented Tikhonov objective to select its coefficient.",
        ]
    )
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
