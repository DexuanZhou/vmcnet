#!/usr/bin/env python3
"""Collect the pre-registered adaptive-S indicator audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


LABELS = (
    "static_pre1000",
    "early_eta095_e1000",
    "early_eta095_e3000",
    "early_eta095_e5000",
    "late_eta03",
    "late_eta08",
    "late_eta095",
)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.root / "summary.json"
    report_path = args.root / "report.md"
    if summary_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite collector output")

    results = {}
    for label in LABELS:
        audit_path = args.root / label / "audit.json"
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
        rows = payload["rows"]
        results[label] = {
            "eta": payload["eta"],
            "static_history": payload["static_history"],
            "elapsed_seconds": payload["elapsed_seconds"],
            "paired_predicted_descent_gain": summarize(
                [row["paired_predicted_descent_gain"] for row in rows]
            ),
            "future_cosine_gain": summarize(
                [row["mean_future_cosine_gain"] for row in rows]
            ),
            "noise_drift_ratio": summarize(
                [row["noise_drift_ratio"] for row in rows]
            ),
            "candidate_cosine": summarize(
                [row["candidate_cosine"] for row in rows]
            ),
            "baseline_positive_fraction": float(
                np.mean([row["baseline_dot_positive"] for row in rows])
            ),
            "mix_gain_positive_fraction": float(
                np.mean(
                    [row["paired_predicted_descent_gain"] > 0.0 for row in rows]
                )
            ),
        }

    static_positive = (
        results["static_pre1000"]["paired_predicted_descent_gain"]["median"] > 0.0
    )
    early_rejections = []
    for label in LABELS[1:4]:
        result = results[label]
        early_rejections.append(
            result["paired_predicted_descent_gain"]["median"] <= 0.0
            or result["noise_drift_ratio"]["median"] > 1.0
        )
    finite_baseline = all(
        result["baseline_positive_fraction"] >= 0.75
        for result in results.values()
    )
    passed = bool(static_positive and sum(early_rejections) >= 2 and finite_baseline)
    conclusion = (
        "indicator_passed_regime_discrimination"
        if passed
        else "indicator_failed_regime_discrimination"
    )
    next_action = (
        "Implement a default-off hysteretic eta_S gate and run matched C E5000."
        if passed
        else "Stop: do not launch adaptive eta_S training from this indicator."
    )
    summary = {
        "experiment": "C adaptive-S indicator collection",
        "results": results,
        "decision": {
            "static_positive_control_passed": static_positive,
            "early_rejection_count": int(sum(early_rejections)),
            "early_rejections": early_rejections,
            "baseline_dot_quality_passed": finite_baseline,
            "passed": passed,
        },
        "conclusion": conclusion,
        "next_action": next_action,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# C adaptive-S indicator audit",
        "",
        f"Conclusion: **{conclusion}**",
        "",
        "| regime | eta | median G | positive fraction | median R | "
        "candidate cosine | baseline dot positive |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        result = results[label]
        lines.append(
            f"| {label} | {result['eta']:.2f} | "
            f"{result['paired_predicted_descent_gain']['median']:+.5f} | "
            f"{result['mix_gain_positive_fraction']:.2f} | "
            f"{result['noise_drift_ratio']['median']:.3f} | "
            f"{result['candidate_cosine']['median']:.5f} | "
            f"{result['baseline_positive_fraction']:.2f} |"
        )
    lines += [
        "",
        f"Static positive control: **{'PASS' if static_positive else 'FAIL'}**.",
        f"Early rejection count: **{sum(early_rejections)}/3**.",
        f"Baseline-dot quality: **{'PASS' if finite_baseline else 'FAIL'}**.",
        "",
        f"Next action: {next_action}",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
