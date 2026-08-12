#!/usr/bin/env python3
"""Collect the four diagnostic arms and apply the pre-registered stop rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FIELDS = (
    "R_J", "R_FD", "R_mean", "R_param", "fisher_proxy_ratio",
    "jvp_fd_cosine_centered", "linearization_relative_error_centered",
    "direction_cosine_raw_vs_transport", "heldout_residual_raw",
    "heldout_residual_transport", "heldout_residual_improvement",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result_dir = args.root / "results"
    paths = [result_dir / f"{method}_rep{rep}.json" for method in ("wssr", "spring") for rep in (0, 1)]
    missing = [str(path) for path in paths if not path.exists()]
    payloads = [json.loads(path.read_text()) for path in paths if path.exists()]
    records = [record for payload in payloads for record in payload["records"]]

    summaries = {}
    for method in ("wssr", "spring"):
        summaries[method] = {}
        for k in (1, 5, 20):
            selected = [r for r in records if r["method"] == method and r["k"] == k]
            summaries[method][str(k)] = {
                field: {
                    "values": [float(r[field]) for r in selected],
                    "median": float(np.median([r[field] for r in selected])) if selected else None,
                }
                for field in FIELDS
            }

    wssr_go_ks = []
    for k in (1, 5, 20):
        selected = [r for r in records if r["method"] == "wssr" and r["k"] == k]
        if len(selected) == 2 and all(
            r["heldout_residual_improvement"] >= 0.10
            and r["linearization_relative_error_centered"] <= 0.25
            for r in selected
        ):
            wssr_go_ks.append(k)
    gate = {
        "complete": not missing,
        "direction_cosine_materiality_threshold": 0.98,
        "heldout_improvement_training_threshold": 0.10,
        "required_independent_wssr_replicates": 2,
        "linearization_error_max": 0.25,
        "passing_k_values": wssr_go_ks,
        "short_training_admissible": bool(not missing and wssr_go_ks),
        "automatic_training_submission": False,
    }
    output = {
        "experiment": "C local-energy RHS transport audit",
        "missing": missing,
        "summaries": summaries,
        "pre_registered_gate": gate,
        "records": records,
    }
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "summary.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    lines = [
        "# C local-energy RHS transport audit: results", "",
        f"Complete: **{gate['complete']}**", "",
        "No training job was automatically submitted.", "",
    ]
    if missing:
        lines += ["## Missing outputs", ""] + [f"- `{p}`" for p in missing] + [""]
    lines += [
        "## Per-replicate result", "",
        "| method | rep | k | R_J | R_FD | R_mean | R_param | lin err | solve cos | held-out improvement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(records, key=lambda x: (x["method"], x["replicate"], x["k"])):
        lines.append(
            f"| {r['method']} | {r['replicate']} | {r['k']} | {r['R_J']:.4g} | "
            f"{r['R_FD']:.4g} | {r['R_mean']:.4g} | {r['R_param']:.4g} | "
            f"{r['linearization_relative_error_centered']:.4g} | "
            f"{r['direction_cosine_raw_vs_transport']:.6f} | "
            f"{100*r['heldout_residual_improvement']:.3f}% |"
        )
    lines += ["", "## Pre-registered decision", ""]
    if not gate["complete"]:
        lines.append("No scientific decision: one or more diagnostic jobs did not produce output.")
    elif gate["short_training_admissible"]:
        lines.append(
            "GO for human review of a short transport training test: the held-out >=10% "
            f"and linearization gates passed at k={wssr_go_ks}. No training was submitted."
        )
    else:
        lines.append(
            "NO-GO for transport training: changing the RHS did not pass the paired, "
            "independent held-out benefit gate. A direction cosine below 0.98, if present, "
            "is interpreted only as materiality."
        )
    (args.root / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(gate, sort_keys=True))


if __name__ == "__main__":
    main()
