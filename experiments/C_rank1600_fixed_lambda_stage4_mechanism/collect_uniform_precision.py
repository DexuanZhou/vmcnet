#!/usr/bin/env python3
"""Collect the uniform same-coordinate C fp32/fp64 audits."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(
    "/scratch/dexuan1/runs/C_wssr_rank1600_uniform_precision_audit_N100k_retry1"
)
LABELS = ("legacy_epoch102000", "fixed_lambda_1e-3_epoch102000")
EXPECTED_SAMPLES = 100_000


def main() -> None:
    summary_path = ROOT / "summary.json"
    report_path = ROOT / "report.md"
    if summary_path.exists() or report_path.exists():
        raise FileExistsError(f"refusing to overwrite collector output in {ROOT}")
    runs = {}
    for label in LABELS:
        data = json.loads((ROOT / label / "summary.json").read_text(encoding="utf-8"))
        validity = data["validity"]
        if not data["read_only"] or not validity["finite"]:
            raise ValueError(f"invalid or mutating audit {label}")
        if not validity["all_coordinates_retained_without_energy_based_selection"]:
            raise ValueError(f"nonuniform coordinate selection in {label}")
        if validity["sampling_vs_recomputed_fp32_max_abs_difference"] != 0.0:
            raise ValueError(f"fp32 replay mismatch in {label}: {validity}")
        if data["fp32_distribution"]["samples"] != EXPECTED_SAMPLES:
            raise ValueError(f"wrong sample count in {label}")
        runs[label] = data

    payload = {
        "experiment": "uniform same-coordinate C full-path fp32/fp64 audit",
        "read_only": True,
        "valid": True,
        "important_scope_note": (
            "Both precisions are evaluated at coordinates sampled from the fp32 "
            "wavefunction. This isolates arithmetic precision and is not an "
            "independent full fp64 MCMC inference."
        ),
        "runs": runs,
    }
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# C uniform same-coordinate fp32/fp64 audit",
        "",
        "All 100,000 coordinates per checkpoint were retained without local-energy selection.",
        "Both paths used the original batch size of 2000. This is a read-only arithmetic-precision audit, not an independent fp64 MCMC inference.",
        "",
        "| checkpoint | mean ΔE (64-32) | block SEM | fp32 variance | fp64 variance | variance change | >10 Ha fp32→fp64 | median / p99 / max |ΔE| |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        data = runs[label]
        paired = data["paired_precision"]
        fp32 = data["fp32_distribution"]
        fp64 = data["fp64_same_coordinate_distribution"]
        events = paired["reference_threshold_event_reclassification"]["10.0"]
        delta = paired["absolute_local_energy_precision_delta"]
        lines.append(
            f"| {label} | "
            f"{paired['mean_signed_local_energy_delta_fp64_minus_fp32']:.8f} | "
            f"{paired['measurement_block_sem_of_mean_delta']:.3g} | "
            f"{fp32['raw_variance']:.8f} | {fp64['raw_variance']:.8f} | "
            f"{100.0 * paired['raw_variance_change_fraction']:+.2f}% | "
            f"{events['fp32_events']}→{events['fp64_events']} | "
            f"{delta['p50']:.4g} / {delta['p99']:.4g} / {delta['max']:.4g} |"
        )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
