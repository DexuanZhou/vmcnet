#!/usr/bin/env python3
"""Collect compressed recurrence accuracy and steady-state timing endpoints."""

import csv
import json
import math
from pathlib import Path


ROOT = Path(
    "/scratch/dexuan1/runs/C_compressed_residual_recurrence_E5000_20260803"
)
RANK1600_ROOT = Path(
    "/scratch/dexuan1/runs/C_rank1600_solution_recurrence_E5000_20260802/"
    "frozen/residual_mu099"
)
SPRING_E5000_VARIANCE = 0.03784060855346502
SPRING_E8000_VARIANCE_PATH = (
    ROOT
    / "frozen_equal_time/spring_replicate2_epoch8000/eval/statistics.json"
)


def load(path):
    payload = json.loads(path.read_text())
    return {
        "energy": float(payload.get("energy", payload["average"])),
        "variance": float(payload["variance"]),
    }


def monitor_elapsed_seconds(path):
    rows = list(csv.DictReader(path.open()))
    return float(rows[-1]["time"]) - float(rows[0]["time"])


summary = {
    "primary_endpoint": "epoch5000_light_frozen_variance",
    "efficiency_endpoint": "frozen_variance_versus_slurm_wall_time",
    "spring_e5000_variance_context": SPRING_E5000_VARIANCE,
    "spring_e8000_equal_time": load(SPRING_E8000_VARIANCE_PATH),
    "results": {},
}
for rank in (400, 800):
    variances = []
    for seed in (0, 1):
        frozen_path = (
            ROOT
            / f"frozen_h100/residual_rank{rank}/seed{seed}/eval/statistics.json"
        )
        if not frozen_path.exists():
            summary["results"][f"rank{rank}_seed{seed}"] = {
                "status": "not_run_after_early_rejection"
            }
            continue
        result = load(frozen_path)
        reference = load(
            RANK1600_ROOT / f"seed{seed}/eval/statistics.json"
        )
        result["variance_over_rank1600_same_seed"] = (
            result["variance"] / reference["variance"]
        )
        result["variance_over_spring_context"] = (
            result["variance"] / SPRING_E5000_VARIANCE
        )
        result["variance_over_spring_e8000_equal_time"] = (
            result["variance"]
            / summary["spring_e8000_equal_time"]["variance"]
        )
        result["training_monitor_elapsed_seconds"] = monitor_elapsed_seconds(
            ROOT
            / f"metadata_train/residual_rank{rank}/seed{seed}/monitor.csv"
        )
        result["training_seconds_per_step"] = (
            result["training_monitor_elapsed_seconds"] / 4999.0
        )
        summary["results"][f"rank{rank}_seed{seed}"] = result
        variances.append(result["variance"])
    if len(variances) == 2:
        summary["results"][f"rank{rank}_two_seed_geomean_variance"] = (
            math.sqrt(variances[0] * variances[1])
        )

rank800_geomean = summary["results"]["rank800_two_seed_geomean_variance"]
rank1600_geomean = math.sqrt(
    load(RANK1600_ROOT / "seed0/eval/statistics.json")["variance"]
    * load(RANK1600_ROOT / "seed1/eval/statistics.json")["variance"]
)
summary["decision"] = {
    "rank400": "rejected_after_seed0_equal_time_variance_ratio_3.99",
    "rank800": "rejected_no_accuracy_time_pareto_advantage_over_spring",
    "rank800_geomean_variance": rank800_geomean,
    "rank800_geomean_over_spring_e8000": (
        rank800_geomean / summary["spring_e8000_equal_time"]["variance"]
    ),
    "rank1600_recurrence_geomean_variance": rank1600_geomean,
    "rank800_geomean_over_rank1600": rank800_geomean / rank1600_geomean,
    "interpretation": (
        "Rank-800 genuine compression preserves the recurrence endpoint to "
        "within about 5 percent of rank 1600 and is much faster than that "
        "implementation, but it remains about 1.93x worse than SPRING at "
        "matched training time."
    ),
}

(ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
report = f"""# C compressed residual-recurrence E5000 result

The current-batch score rank is at most 999, so ranks 400 and 800 are genuine
compressions.  All arms use residual recurrence with `mu=.99`, fixed
Tikhonov lambda `1e-3`, `eta_S=eta_g=0`, SSI 40/2, 1000 walkers and
`reburn=False`.

| method | seed | frozen variance | steady train time | vs SPRING-8k |
|---|---:|---:|---:|---:|
| rank400 recurrence | 0 | {summary['results']['rank400_seed0']['variance']:.8f} | {summary['results']['rank400_seed0']['training_monitor_elapsed_seconds']:.1f} s | {summary['results']['rank400_seed0']['variance_over_spring_e8000_equal_time']:.3f}x |
| rank800 recurrence | 0 | {summary['results']['rank800_seed0']['variance']:.8f} | {summary['results']['rank800_seed0']['training_monitor_elapsed_seconds']:.1f} s | {summary['results']['rank800_seed0']['variance_over_spring_e8000_equal_time']:.3f}x |
| rank800 recurrence | 1 | {summary['results']['rank800_seed1']['variance']:.8f} | {summary['results']['rank800_seed1']['training_monitor_elapsed_seconds']:.1f} s | {summary['results']['rank800_seed1']['variance_over_spring_e8000_equal_time']:.3f}x |
| SPRING replicate2 epoch8000 | - | {summary['spring_e8000_equal_time']['variance']:.8f} | about 640 s from checkpoint 1k to 8k | 1.000x |

Rank400 seed1 was not evaluated after seed0 missed the equal-time target by
3.99x.  Rank800 has a two-seed geometric-mean variance of
`{rank800_geomean:.8f}`, which is
`{summary['decision']['rank800_geomean_over_spring_e8000']:.3f}x` SPRING-8k.
It therefore does not produce an accuracy-time Pareto improvement.

Compression itself is not catastrophic: the rank800 two-seed endpoint is only
`{summary['decision']['rank800_geomean_over_rank1600']:.3f}x` the uncompressed
rank1600 recurrence endpoint while its measured per-step time is substantially
lower.  The remaining failure is that even this faster compressed recurrence
is slower and less accurate than native SPRING on C.
"""
(ROOT / "report.md").write_text(report)
print(json.dumps(summary, indent=2))
