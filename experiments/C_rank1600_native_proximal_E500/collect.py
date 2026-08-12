"""Collect the matched C rank1600 native-proximal E500 experiment."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import fmean, median


TRAIN_ROOT = Path(
    "/scratch/dexuan1/runs/"
    "C_wssr_rank1600_native_proximal_gamma3e4_matched_E500_20260802"
)
EVAL_ROOT = Path(f"{TRAIN_ROOT}_frozen")
RESULTS = Path(f"{TRAIN_ROOT}_results")
LABELS = ("hard_control", "native_proximal")


def read_training(label: str) -> dict[str, object]:
    run = TRAIN_ROOT / label
    with (run / "training_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["epoch"]) > 102000]
    if len(rows) != 500:
        raise ValueError(f"{label}: expected 500 rows, got {len(rows)}")
    fields = ("energy_noclip", "variance_noclip", "accept_ratio")
    for field in fields:
        values = [float(row[field]) for row in rows]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{label}: non-finite {field}")
    result: dict[str, object] = {
        "rows": len(rows),
        "last100_energy_raw_mean": fmean(
            float(row["energy_noclip"]) for row in rows[-100:]
        ),
        "last100_variance_raw_mean": fmean(
            float(row["variance_noclip"]) for row in rows[-100:]
        ),
        "last100_acceptance_mean": fmean(
            float(row["accept_ratio"]) for row in rows[-100:]
        ),
    }
    if label == "native_proximal":
        for field in (
            "wssr_diag_native_proximal_previous_cosine",
            "wssr_diag_native_proximal_relative_change",
            "wssr_diag_norm_constraint_scale",
            "wssr_diag_finite",
        ):
            # Auxiliary metrics are logged as one-value-per-line text files;
            # training_metrics.csv intentionally contains only the core series.
            values = [
                float(line)
                for line in (run / f"{field}.txt")
                .read_text(encoding="utf-8")
                .splitlines()
            ][-500:]
            if len(values) != 500:
                raise ValueError(f"{label}: expected 500 values for {field}")
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{label}: non-finite {field}")
            result[f"{field}_median"] = median(values)
    return result


def read_eval(label: str) -> dict[str, float | int | str]:
    path = EVAL_ROOT / label / "epoch102500" / "eval"
    stats = json.loads((path / "statistics.json").read_text(encoding="utf-8"))
    acceptance = [
        float(line)
        for line in (path / "accept_ratio.txt").read_text(encoding="utf-8").splitlines()
    ]
    if len(acceptance) != 20000:
        raise ValueError(f"{label}: expected 20000 acceptance rows")
    result: dict[str, float | int | str] = {
        "frozen_energy": float(stats["average"]),
        "frozen_sem": float(stats["std_err"]),
        "frozen_variance": float(stats["variance"]),
        "frozen_iat": float(stats["integrated_autocorrelation"]),
        "frozen_acceptance_mean": fmean(acceptance),
        "frozen_walkers": 2000,
        "frozen_burn_in": 5000,
        "frozen_measurement_iterations": 20000,
        "frozen_mcmc_steps": 10,
        "frozen_local_energy_samples": 40000000,
        "source_path": str(path),
    }
    if not all(
        math.isfinite(value)
        for key, value in result.items()
        if key.startswith("frozen_") and isinstance(value, float)
    ):
        raise ValueError(f"{label}: non-finite frozen statistic")
    return result


def main() -> None:
    if RESULTS.exists():
        raise FileExistsError(f"refusing to overwrite {RESULTS}")
    training = {label: read_training(label) for label in LABELS}
    frozen = {label: read_eval(label) for label in LABELS}
    hard = frozen["hard_control"]
    prox = frozen["native_proximal"]
    delta_energy = float(prox["frozen_energy"]) - float(hard["frozen_energy"])
    combined_sem = math.hypot(
        float(prox["frozen_sem"]), float(hard["frozen_sem"])
    )
    variance_ratio = float(prox["frozen_variance"]) / float(
        hard["frozen_variance"]
    )
    comparison = {
        "proximal_minus_hard_frozen_energy": delta_energy,
        "combined_sem": combined_sem,
        "energy_z_score": delta_energy / combined_sem,
        "proximal_over_hard_frozen_variance": variance_ratio,
        "variance_reduction_fraction": 1.0 - variance_ratio,
        "energy_noninferior_at_2sem": delta_energy <= 2.0 * combined_sem,
        "variance_reduction_at_least_5pct": variance_ratio <= 0.95,
    }
    comparison["screen_success"] = bool(
        comparison["energy_noninferior_at_2sem"]
        and comparison["variance_reduction_at_least_5pct"]
    )
    payload = {
        "experiment": "C rank1600 WSSR native-factor proximal matched E500",
        "scientific_caution": (
            "C absolute energies remain unsuitable for physical claims; this is an "
            "internal matched optimizer comparison. One E500 continuation is a "
            "screen, not a multi-seed final claim."
        ),
        "source_checkpoint": (
            "/scratch/dexuan1/runs/C_wssr_rank1600_fixed_lambda_stage2_E2000/"
            "checkpoints/102000.npz"
        ),
        "common_settings": {
            "rank": 1600,
            "eta": 0.3,
            "learning_rate": 0.04,
            "learning_decay_rate": 0.0001,
            "damping": 0.0003,
            "relative_singular_value_cutoff": 0.0003,
            "tikhonov_lambda": 0.001,
            "norm_constraint": 0.001,
            "warm_iterations": 2,
            "complement_weight": 0.0,
            "additional_training_steps": 500,
        },
        "candidate": {
            "mode": "native_proximal",
            "gamma": 0.0003,
            "first_step_previous_direction": "zero_initialized",
        },
        "training": training,
        "frozen": frozen,
        "comparison": comparison,
    }
    RESULTS.mkdir(parents=True)
    (RESULTS / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# C rank1600 native-factor proximal matched E500",
        "",
        "Internal matched comparison only; absolute C energies are not physical claims.",
        "",
        "| method | frozen energy (Ha) | SEM | raw variance (Ha^2) | IAT | acceptance |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        run = frozen[label]
        lines.append(
            f"| {label} | {run['frozen_energy']:.10f} | {run['frozen_sem']:.3e} | "
            f"{run['frozen_variance']:.8f} | {run['frozen_iat']:.4f} | "
            f"{run['frozen_acceptance_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"- Proximal minus hard energy: {delta_energy:.8e} Ha "
            f"({comparison['energy_z_score']:.3f} combined SEM).",
            f"- Proximal/hard variance ratio: {variance_ratio:.6f} "
            f"({100.0 * (1.0 - variance_ratio):.2f}% reduction).",
            f"- Predeclared E500 screen success: {comparison['screen_success']}.",
            "",
        ]
    )
    (RESULTS / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
