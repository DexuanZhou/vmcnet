#!/usr/bin/env python3
"""Collect the matched light frozen-evaluation gap curve."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path("/scratch/dexuan1/runs/C_optimizer_gap_curve_light_20260802")
METHODS = {
    "shared_kfac_pre1000": [1000],
    "spring_replicate2": [5000, 20000, 50000, 100000],
    "wssr_rank1600_eta03_lr004": [5000, 20000, 50000, 100000],
}


def load_record(method: str, epoch: int) -> dict:
    requested = ROOT / method / f"epoch{epoch}"
    candidates = [requested / "eval", requested.with_name(requested.name + "_1") / "eval"]
    complete = [path for path in candidates if (path / "statistics.json").is_file()]
    if len(complete) != 1:
        raise RuntimeError(
            f"expected exactly one completed eval for {method} epoch {epoch}, "
            f"found {complete}"
        )
    eval_dir = complete[0]
    with (eval_dir / "statistics.json").open() as handle:
        stats = json.load(handle)
    acceptance = np.loadtxt(eval_dir / "accept_ratio.txt")
    if np.size(acceptance) != 2000:
        raise ValueError(
            f"expected 2000 acceptance records in {eval_dir}, got {np.size(acceptance)}"
        )
    return {
        "method": method,
        "epoch": epoch,
        "energy": float(stats["average"]),
        "sem": float(stats["std_err"]),
        "raw_variance": float(stats["variance"]),
        "iat": float(stats["integrated_autocorrelation"]),
        "acceptance": float(np.mean(acceptance)),
        "walkers": 1000,
        "burn_in": 5000,
        "measurement_epochs": 2000,
        "mcmc_steps": 10,
        "source_path": str(eval_dir),
    }


records = [
    load_record(method, epoch)
    for method, epochs in METHODS.items()
    for epoch in epochs
]
by_key = {(row["method"], row["epoch"]): row for row in records}

comparisons = []
for epoch in (5000, 20000, 50000, 100000):
    spring = by_key[("spring_replicate2", epoch)]
    wssr = by_key[("wssr_rank1600_eta03_lr004", epoch)]
    ratio = wssr["raw_variance"] / spring["raw_variance"]
    comparisons.append(
        {
            "epoch": epoch,
            "spring_variance": spring["raw_variance"],
            "wssr_variance": wssr["raw_variance"],
            "variance_ratio": ratio,
            "log_variance_gap": math.log(ratio),
        }
    )

final_gap = comparisons[-1]["log_variance_gap"]
for row in comparisons:
    row["fraction_of_final_log_gap"] = (
        row["log_variance_gap"] / final_gap if final_gap != 0.0 else None
    )

f5 = comparisons[0]["fraction_of_final_log_gap"]
f20 = comparisons[1]["fraction_of_final_log_gap"]
gaps = [row["log_variance_gap"] for row in comparisons]
monotone = all(a <= b for a, b in zip(gaps, gaps[1:]))
if f5 is not None and f5 >= 0.5:
    classification = "early_gap_opening_by_epoch5000"
    next_checkpoint = 1000
elif f20 is not None and f20 < 0.5 and monotone:
    classification = "gradual_accumulation_after_epoch20000"
    next_checkpoint = 20000
else:
    classification = "mixed_or_ambiguous"
    next_checkpoint = None

summary = {
    "experiment": "C SPRING-WSSR matched light frozen-variance gap curve",
    "protocol": {
        "walkers": 1000,
        "burn_in": 5000,
        "measurement_epochs": 2000,
        "mcmc_steps": 10,
        "training": False,
    },
    "records": records,
    "comparisons": comparisons,
    "classification": classification,
    "pre_registered_rules": {
        "early": "fraction_of_final_log_gap_at_5000 >= 0.5",
        "gradual": "fraction_at_20000 < 0.5 and sampled log gaps monotone",
        "otherwise": "mixed_or_ambiguous",
    },
    "recommended_stage2_checkpoint": next_checkpoint,
    "scientific_caution": (
        "Absolute C energies are not used for physical conclusions; this is a "
        "matched internal optimizer comparison."
    ),
}

ROOT.mkdir(parents=True, exist_ok=True)
with (ROOT / "summary.json").open("w") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
    handle.write("\n")

with (ROOT / "summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
    writer.writeheader()
    writer.writerows(comparisons)

lines = [
    "# C matched frozen-variance gap curve",
    "",
    "All points use the same light frozen protocol: 1000 walkers, 5000 burn-in "
    "steps, 2000 measurement epochs and 10 MCMC steps per measurement.",
    "",
    "| checkpoint epoch | SPRING variance | WSSR variance | ratio | log gap | fraction of final log gap |",
    "|---:|---:|---:|---:|---:|---:|",
]
for row in comparisons:
    lines.append(
        f"| {row['epoch']} | {row['spring_variance']:.8f} | "
        f"{row['wssr_variance']:.8f} | {row['variance_ratio']:.4f} | "
        f"{row['log_variance_gap']:.4f} | "
        f"{row['fraction_of_final_log_gap']:.2%} |"
    )
lines += [
    "",
    f"- Classification: **{classification}**.",
    f"- Recommended stage-2 checkpoint: `{next_checkpoint}`.",
    "- Absolute C energies are not used for physical conclusions.",
]
(ROOT / "report.md").write_text("\n".join(lines) + "\n")

print(json.dumps(summary, indent=2, sort_keys=True))
