"""Collect the matched 500-step fixed-lambda WSSR stage-1 experiment."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import fmean


ROOT = Path("/scratch/dexuan1/runs/C_wssr_rank1600_fixed_lambda_stage1_E500_retry1")
OLD_CONTROL = Path(
    "/scratch/dexuan1/runs/C_wssr_rank1600_hard_matched_control_E5000"
)
LABELS = ("legacy_hard", "fixed_lambda_1e-3")
START_EPOCH = 100000
END_EPOCH = 100500


def read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def continuation_rows(path: Path) -> list[dict[str, str]]:
    rows = read_metrics(path)
    selected = [
        row
        for row in rows
        if START_EPOCH < int(row["epoch"]) <= END_EPOCH
    ]
    if len(selected) != END_EPOCH - START_EPOCH:
        raise ValueError(f"expected 500 continuation rows in {path}, got {len(selected)}")
    return selected


def mean_field(rows: list[dict[str, str]], field: str) -> float:
    values = [float(row[field]) for row in rows]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"non-finite {field}")
    return fmean(values)


def read_float_lines(path: Path) -> list[float]:
    values = [float(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"missing or non-finite values in {path}")
    return values


def load_run(label: str) -> dict[str, object]:
    train_dir = ROOT / label
    eval_dir = ROOT / "frozen" / f"{label}_epoch100500" / "eval"
    checkpoint = train_dir / "checkpoints" / "100500.npz"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    rows = continuation_rows(train_dir / "training_metrics.csv")
    stats = json.loads((eval_dir / "statistics.json").read_text(encoding="utf-8"))
    acceptance = read_float_lines(eval_dir / "accept_ratio.txt")
    config = json.loads((train_dir / "config.json").read_text(encoding="utf-8"))
    wssr = config["vmc"]["optimizer"]["wssr_warm_svd_right"]
    result = {
        "checkpoint": str(checkpoint),
        "training_rows": len(rows),
        "training_last100_energy_raw_mean": mean_field(rows[-100:], "energy_noclip"),
        "training_last100_variance_raw_mean": mean_field(rows[-100:], "variance_noclip"),
        "training_last100_acceptance_mean": mean_field(rows[-100:], "accept_ratio"),
        "frozen_energy": float(stats["average"]),
        "frozen_sem": float(stats["std_err"]),
        "frozen_variance": float(stats["variance"]),
        "frozen_iat": float(stats["integrated_autocorrelation"]),
        "frozen_acceptance_mean": fmean(acceptance),
        "frozen_measurement_iterations": len(acceptance),
        "frozen_walkers": 2000,
        "frozen_local_energy_samples": len(acceptance) * 2000,
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
    for key in ("frozen_energy", "frozen_sem", "frozen_variance", "frozen_iat"):
        if not math.isfinite(float(result[key])):
            raise ValueError(f"non-finite {key} for {label}")
    return result


def reproduction_check() -> dict[str, object]:
    rerun = continuation_rows(ROOT / "legacy_hard" / "training_metrics.csv")
    old = continuation_rows(OLD_CONTROL / "training_metrics.csv")
    fields = (
        "energy",
        "energy_noclip",
        "variance",
        "variance_noclip",
        "accept_ratio",
    )
    max_abs = {
        field: max(abs(float(a[field]) - float(b[field])) for a, b in zip(rerun, old))
        for field in fields
    }
    return {
        "reference_path": str(OLD_CONTROL / "training_metrics.csv"),
        "compared_rows": len(rerun),
        "max_absolute_differences": max_abs,
        "bitwise_numeric_match": all(value == 0.0 for value in max_abs.values()),
    }


def main() -> None:
    if (ROOT / "results").exists():
        raise FileExistsError(f"refusing to overwrite {ROOT / 'results'}")
    runs = {label: load_run(label) for label in LABELS}
    control = runs["legacy_hard"]
    fixed = runs["fixed_lambda_1e-3"]
    combined_sem = math.hypot(
        float(control["frozen_sem"]), float(fixed["frozen_sem"])
    )
    delta_energy = float(fixed["frozen_energy"]) - float(control["frozen_energy"])
    variance_ratio = float(fixed["frozen_variance"]) / float(
        control["frozen_variance"]
    )
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
    comparison["advance_to_stage2"] = bool(
        comparison["energy_noninferior_at_2sem"]
        and comparison["variance_reduction_at_least_10pct"]
    )
    payload = {
        "experiment": "C rank1600 WSSR fixed-lambda stage 1",
        "source_epoch": START_EPOCH,
        "target_epoch": END_EPOCH,
        "scientific_caution": (
            "C absolute energies remain unsuitable for physical claims; this is an "
            "internal matched optimizer comparison."
        ),
        "runs": runs,
        "comparison": comparison,
        "legacy_reproduction": reproduction_check(),
    }

    result_dir = ROOT / "results"
    result_dir.mkdir(parents=True)
    (result_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# C rank1600 fixed-lambda stage-1 result",
        "",
        "This is an internal matched comparison; no physical conclusion is drawn from the absolute C energies.",
        "",
        "| method | frozen energy (Ha) | SEM | raw variance (Ha^2) | IAT | acceptance |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        run = runs[label]
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
            f"- Energy non-inferior at 2 SEM: {comparison['energy_noninferior_at_2sem']}.",
            f"- At least 10% variance reduction: {comparison['variance_reduction_at_least_10pct']}.",
            f"- Advance to stage 2 under the predeclared rule: {comparison['advance_to_stage2']}.",
            "",
        ]
    )
    (result_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
