"""Collect the C rank1600 fixed-lambda E2000 stage-2 experiment."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import fmean


TRAIN = Path("/scratch/dexuan1/runs/C_wssr_rank1600_fixed_lambda_stage2_E2000")
FIXED_EVAL = Path(
    "/scratch/dexuan1/runs/C_wssr_rank1600_fixed_lambda_stage2_E2000_frozen/epoch102000/eval"
)
LEGACY_EVAL = Path(
    "/scratch/dexuan1/runs/C_wssr_rank1600_hard_matched_control_frozen/epoch102000/eval"
)
STAGE1 = Path(
    "/scratch/dexuan1/runs/C_wssr_rank1600_fixed_lambda_stage1_E500_retry1/results/summary.json"
)
RESULTS = Path(
    "/scratch/dexuan1/runs/C_wssr_rank1600_fixed_lambda_stage2_E2000_results"
)


def read_stats(path: Path) -> dict[str, float]:
    stats = json.loads((path / "statistics.json").read_text(encoding="utf-8"))
    result = {
        "frozen_energy": float(stats["average"]),
        "frozen_sem": float(stats["std_err"]),
        "frozen_variance": float(stats["variance"]),
        "frozen_iat": float(stats["integrated_autocorrelation"]),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError(f"non-finite statistics in {path}")
    acceptance = [
        float(line)
        for line in (path / "accept_ratio.txt").read_text(encoding="utf-8").splitlines()
    ]
    if len(acceptance) != 20000 or not all(
        math.isfinite(value) for value in acceptance
    ):
        raise ValueError(f"invalid acceptance series in {path}")
    result.update(
        {
            "frozen_acceptance_mean": fmean(acceptance),
            "frozen_walkers": 2000,
            "frozen_measurement_iterations": 20000,
            "frozen_local_energy_samples": 40000000,
            "source_path": str(path),
        }
    )
    return result


def read_training() -> dict[str, object]:
    with (TRAIN / "training_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if 100500 < int(row["epoch"]) <= 102000]
    if len(selected) != 1500:
        raise ValueError(f"expected 1500 stage-2 rows, got {len(selected)}")
    for field in ("energy_noclip", "variance_noclip", "accept_ratio"):
        if not all(math.isfinite(float(row[field])) for row in selected):
            raise ValueError(f"non-finite training field {field}")
    config = json.loads((TRAIN / "config.json").read_text(encoding="utf-8"))
    wssr = config["vmc"]["optimizer"]["wssr_warm_svd_right"]
    return {
        "additional_rows": len(selected),
        "last100_energy_raw_mean": fmean(
            float(row["energy_noclip"]) for row in selected[-100:]
        ),
        "last100_variance_raw_mean": fmean(
            float(row["variance_noclip"]) for row in selected[-100:]
        ),
        "last100_acceptance_mean": fmean(
            float(row["accept_ratio"]) for row in selected[-100:]
        ),
        "actual_parameters": {
            "rank": int(wssr["sr_rank"]),
            "eta": float(wssr["eta"]),
            "learning_rate": float(wssr["learning_rate"]),
            "damping": float(wssr["damping"]),
            "relative_singular_value_cutoff": float(
                wssr["relative_singular_value_cutoff"]
            ),
            "tikhonov_lambda": float(wssr["tikhonov_lambda"]),
            "warm_iterations": int(wssr["svd_maxiter_warm"]),
            "complement_weight": float(wssr["complement_weight"]),
        },
    }


def main() -> None:
    if RESULTS.exists():
        raise FileExistsError(f"refusing to overwrite {RESULTS}")
    fixed = read_stats(FIXED_EVAL)
    legacy = read_stats(LEGACY_EVAL)
    combined_sem = math.hypot(fixed["frozen_sem"], legacy["frozen_sem"])
    delta_energy = fixed["frozen_energy"] - legacy["frozen_energy"]
    variance_ratio = fixed["frozen_variance"] / legacy["frozen_variance"]
    comparison = {
        "fixed_minus_legacy_frozen_energy": delta_energy,
        "combined_sem": combined_sem,
        "energy_z_score": delta_energy / combined_sem,
        "fixed_over_legacy_frozen_variance": variance_ratio,
        "variance_reduction_fraction": 1.0 - variance_ratio,
        "energy_noninferior_at_2sem": delta_energy <= 2.0 * combined_sem,
        "variance_reduction_at_least_10pct": variance_ratio <= 0.90,
        "variance_reduction_at_least_20pct": variance_ratio <= 0.80,
    }
    comparison["stage2_success"] = bool(
        comparison["energy_noninferior_at_2sem"]
        and comparison["variance_reduction_at_least_10pct"]
    )
    stage1 = json.loads(STAGE1.read_text(encoding="utf-8"))
    payload = {
        "experiment": "C rank1600 WSSR fixed-lambda stage 2 matched E2000",
        "scientific_caution": (
            "C absolute energies remain unsuitable for physical claims; this is an "
            "internal matched optimizer comparison."
        ),
        "training": read_training(),
        "runs": {"legacy_hard": legacy, "fixed_lambda_1e-3": fixed},
        "comparison": comparison,
        "stage1_variance_reduction_fraction": stage1["comparison"][
            "variance_reduction_fraction"
        ],
        "legacy_control_justification": (
            "The stage-1 legacy replay matched the existing legacy trajectory bitwise "
            "for epochs 100001-100500, so the existing epoch-102000 checkpoint and "
            "frozen evaluation are reused as the strict continuation control."
        ),
    }
    RESULTS.mkdir(parents=True)
    (RESULTS / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# C rank1600 fixed-lambda stage-2 E2000 result",
        "",
        "Internal matched comparison only; absolute C energies are not used for physical claims.",
        "",
        "| method | frozen energy (Ha) | SEM | raw variance (Ha^2) | IAT | acceptance |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, run in (("legacy_hard", legacy), ("fixed_lambda_1e-3", fixed)):
        lines.append(
            f"| {label} | {run['frozen_energy']:.10f} | {run['frozen_sem']:.3e} | "
            f"{run['frozen_variance']:.8f} | {run['frozen_iat']:.4f} | "
            f"{run['frozen_acceptance_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"- Fixed minus legacy energy: {delta_energy:.8e} Ha ({comparison['energy_z_score']:.3f} combined SEM).",
            f"- Fixed/legacy variance ratio: {variance_ratio:.6f} ({100.0 * (1.0 - variance_ratio):.2f}% reduction).",
            f"- Stage-1 variance reduction: {100.0 * payload['stage1_variance_reduction_fraction']:.2f}%.",
            f"- Stage-2 success under the predeclared rule: {comparison['stage2_success']}.",
            "",
        ]
    )
    (RESULTS / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
