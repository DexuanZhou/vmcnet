#!/usr/bin/env python3
"""Collect paired frozen results for the three SPRING->WSSR switch points."""

import csv
import json
import math
import re
import subprocess
from pathlib import Path


ROOT = Path("/scratch/dexuan1/runs/C_stagewise_switch_scan_20260807")
HERE = Path(__file__).resolve().parent
LATE = Path("/scratch/dexuan1/runs/C_late_spring50k_eta_drift_20260803")


def read_statistics(path):
    with path.open() as handle:
        return json.load(handle)


def training_elapsed_seconds(arm):
    metadata = ROOT / "metadata" / arm / "slurm_job.txt"
    text = metadata.read_text()
    match = re.search(r"ArrayJobId=(\d+)", text)
    if not match:
        return math.nan
    array_task = 0 if arm == "switch5k" else 1
    job_id = f"{match.group(1)}_{array_task}"
    result = subprocess.run(
        [
            "sacct",
            "-n",
            "-X",
            "-j",
            job_id,
            "--format=ElapsedRaw",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [
        int(line.strip())
        for line in result.stdout.splitlines()
        if line.strip().isdigit()
    ]
    return max(values) if values else math.nan


pairs = [
    (5000, ROOT / "frozen" / "switch5k", ROOT / "frozen" / "spring10k"),
    (20000, ROOT / "frozen" / "switch20k", ROOT / "frozen" / "spring25k"),
    (50000, LATE / "frozen" / "eta03", LATE / "frozen" / "spring55k"),
]
rows = []
for switch_epoch, wssr_dir, spring_dir in pairs:
    wssr = read_statistics(wssr_dir / "eval" / "statistics.json")
    spring = read_statistics(spring_dir / "eval" / "statistics.json")
    arm = f"switch{switch_epoch // 1000}k"
    elapsed = (
        training_elapsed_seconds(arm) if switch_epoch < 50000 else math.nan
    )
    rows.append(
        {
            "switch_epoch": switch_epoch,
            "wssr_energy": wssr["average"],
            "spring_energy": spring["average"],
            "wssr_variance": wssr["variance"],
            "spring_variance": spring["variance"],
            "wssr_over_spring_variance": (
                wssr["variance"] / spring["variance"]
            ),
            "wssr_elapsed_seconds": elapsed,
            "wssr_conservative_s_per_step": elapsed / 5000,
        }
    )

with (HERE / "summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

lines = [
    "# C stage-wise SPRING -> WSSR result",
    "",
    "The variance ratio is paired at the same total optimization epoch and "
    "frozen-evaluation seed. A value below one favors the WSSR switch.",
    "",
    "| switch epoch | WSSR frozen E | SPRING frozen E | WSSR frozen var | "
    "SPRING frozen var | var ratio | conservative WSSR s/step |",
    "|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    step_time = row["wssr_conservative_s_per_step"]
    step_time_text = "existing run" if math.isnan(step_time) else f"{step_time:.6f}"
    lines.append(
        f"| {row['switch_epoch']} | {row['wssr_energy']:.9f} | "
        f"{row['spring_energy']:.9f} | {row['wssr_variance']:.8f} | "
        f"{row['spring_variance']:.8f} | "
        f"{row['wssr_over_spring_variance']:.4f} | {step_time_text} |"
    )
(HERE / "REPORT.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
