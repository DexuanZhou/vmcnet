"""Collect the fixed-lambda variance-gain localization study."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import fmean


OLD_LEGACY_TRAIN = Path(
    "/scratch/dexuan1/runs/C_wssr_rank1600_hard_matched_control_E5000/training_metrics.csv"
)
NEW_LEGACY_TRAIN = Path(
    "/scratch/dexuan1/runs/C_wssr_rank1600_fixed_lambda_stage3_legacy_E1500/training_metrics.csv"
)
STAGE1 = Path(
    "/scratch/dexuan1/runs/C_wssr_rank1600_fixed_lambda_stage1_E500_retry1/results/summary.json"
)
STAGE2 = Path(
    "/scratch/dexuan1/runs/C_wssr_rank1600_fixed_lambda_stage2_E2000_results/summary.json"
)
MID = Path("/scratch/dexuan1/runs/C_wssr_rank1600_fixed_lambda_stage3_frozen")
RESULTS = Path(
    "/scratch/dexuan1/runs/C_wssr_rank1600_fixed_lambda_stage3_localization_results"
)


def read_eval(path: Path) -> dict[str, object]:
    stats = json.loads((path / "statistics.json").read_text(encoding="utf-8"))
    acceptance = [
        float(line)
        for line in (path / "accept_ratio.txt").read_text(encoding="utf-8").splitlines()
    ]
    result: dict[str, object] = {
        "energy": float(stats["average"]),
        "sem": float(stats["std_err"]),
        "variance": float(stats["variance"]),
        "iat": float(stats["integrated_autocorrelation"]),
        "acceptance": fmean(acceptance),
        "walkers": 2000,
        "measurement_iterations": len(acceptance),
        "local_energy_samples": len(acceptance) * 2000,
        "source_path": str(path),
    }
    numeric = ("energy", "sem", "variance", "iat", "acceptance")
    if len(acceptance) != 20000 or not all(
        math.isfinite(float(result[key])) for key in numeric
    ):
        raise ValueError(f"invalid frozen evaluation {path}")
    return result


def comparison(legacy: dict[str, object], fixed: dict[str, object]) -> dict[str, object]:
    delta_energy = float(fixed["energy"]) - float(legacy["energy"])
    combined_sem = math.hypot(float(fixed["sem"]), float(legacy["sem"]))
    variance_ratio = float(fixed["variance"]) / float(legacy["variance"])
    return {
        "fixed_minus_legacy_energy": delta_energy,
        "combined_sem": combined_sem,
        "energy_z_score": delta_energy / combined_sem,
        "variance_ratio": variance_ratio,
        "variance_reduction_fraction": 1.0 - variance_ratio,
        "energy_noninferior_at_2sem": delta_energy <= 2.0 * combined_sem,
        "variance_reduction_at_least_10pct": variance_ratio <= 0.90,
    }


def from_prior(run: dict[str, object]) -> dict[str, object]:
    return {
        "energy": run["frozen_energy"],
        "sem": run["frozen_sem"],
        "variance": run["frozen_variance"],
        "iat": run["frozen_iat"],
        "acceptance": run["frozen_acceptance_mean"],
        "walkers": run["frozen_walkers"],
        "measurement_iterations": run["frozen_measurement_iterations"],
        "local_energy_samples": run["frozen_local_energy_samples"],
        "source_path": run["source_path"] if "source_path" in run else run["checkpoint"],
    }


def read_rows(path: Path, start: int, end: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if start < int(row["epoch"]) <= end]
    if len(selected) != end - start:
        raise ValueError(f"expected {end-start} rows in {path}, got {len(selected)}")
    return selected


def reproduction_check() -> dict[str, object]:
    old = read_rows(OLD_LEGACY_TRAIN, 101000, 101500)
    new = read_rows(NEW_LEGACY_TRAIN, 101000, 101500)
    fields = (
        "energy",
        "energy_noclip",
        "variance",
        "variance_noclip",
        "accept_ratio",
    )
    maxima = {
        field: max(abs(float(a[field]) - float(b[field])) for a, b in zip(old, new))
        for field in fields
    }
    return {
        "compared_rows": 500,
        "max_absolute_differences": maxima,
        "bitwise_numeric_match": all(value == 0.0 for value in maxima.values()),
        "existing_trajectory": str(OLD_LEGACY_TRAIN),
        "replay_trajectory": str(NEW_LEGACY_TRAIN),
    }


def main() -> None:
    if RESULTS.exists():
        raise FileExistsError(f"refusing to overwrite {RESULTS}")
    stage1 = json.loads(STAGE1.read_text(encoding="utf-8"))
    stage2 = json.loads(STAGE2.read_text(encoding="utf-8"))

    evaluations: dict[int, dict[str, dict[str, object]]] = {
        100500: {
            label: from_prior(run) for label, run in stage1["runs"].items()
        },
        101000: {
            "legacy_hard": read_eval(
                Path(
                    "/scratch/dexuan1/runs/C_wssr_rank1600_hard_matched_control_frozen/epoch101000/eval"
                )
            ),
            "fixed_lambda_1e-3": read_eval(
                MID / "fixed_lambda_1e-3_epoch101000" / "eval"
            ),
        },
        101500: {
            "legacy_hard": read_eval(MID / "legacy_hard_epoch101500" / "eval"),
            "fixed_lambda_1e-3": read_eval(
                MID / "fixed_lambda_1e-3_epoch101500" / "eval"
            ),
        },
        102000: {
            label: from_prior(run) for label, run in stage2["runs"].items()
        },
    }
    comparisons = {
        epoch: comparison(runs["legacy_hard"], runs["fixed_lambda_1e-3"])
        for epoch, runs in evaluations.items()
    }
    successful = [
        epoch
        for epoch, result in comparisons.items()
        if bool(result["variance_reduction_at_least_10pct"])
    ]
    failed = [
        epoch
        for epoch, result in comparisons.items()
        if not bool(result["variance_reduction_at_least_10pct"])
    ]
    payload = {
        "experiment": "C rank1600 fixed-lambda stage-3 gain localization",
        "scientific_caution": (
            "C absolute energies remain unsuitable for physical claims; this is an "
            "internal matched optimizer comparison."
        ),
        "evaluations": evaluations,
        "comparisons": comparisons,
        "last_epoch_with_at_least_10pct_variance_gain": max(successful)
        if successful
        else None,
        "first_epoch_without_10pct_variance_gain": min(failed) if failed else None,
        "legacy_reproduction": reproduction_check(),
    }
    RESULTS.mkdir(parents=True)
    (RESULTS / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# C rank1600 fixed-lambda stage-3 localization",
        "",
        "Internal matched comparison only; absolute C energies are not used for physical claims.",
        "",
        "| epoch | legacy variance | fixed variance | reduction | energy delta (mHa) | z |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for epoch in sorted(comparisons):
        comp = comparisons[epoch]
        runs = evaluations[epoch]
        lines.append(
            f"| {epoch} | {runs['legacy_hard']['variance']:.8f} | "
            f"{runs['fixed_lambda_1e-3']['variance']:.8f} | "
            f"{100.0 * float(comp['variance_reduction_fraction']):.2f}% | "
            f"{1000.0 * float(comp['fixed_minus_legacy_energy']):.5f} | "
            f"{float(comp['energy_z_score']):.3f} |"
        )
    lines.extend(
        [
            "",
            f"- Last epoch with at least 10% variance gain: {payload['last_epoch_with_at_least_10pct_variance_gain']}.",
            f"- First epoch without 10% variance gain: {payload['first_epoch_without_10pct_variance_gain']}.",
            f"- Legacy replay bitwise match: {payload['legacy_reproduction']['bitwise_numeric_match']}.",
            "",
        ]
    )
    (RESULTS / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
