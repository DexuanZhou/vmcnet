"""Audit whether the fixed-lambda variance crossover is caused by rare tails."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


OUTPUT = Path(
    "/scratch/dexuan1/runs/"
    "C_wssr_rank1600_fixed_lambda_stage4_tail_audit"
)
RUNS = {
    "101500": {
        "legacy_hard": Path(
            "/scratch/dexuan1/runs/"
            "C_wssr_rank1600_fixed_lambda_stage3_frozen/"
            "legacy_hard_epoch101500/eval/local_energies.txt"
        ),
        "fixed_lambda_1e-3": Path(
            "/scratch/dexuan1/runs/"
            "C_wssr_rank1600_fixed_lambda_stage3_frozen/"
            "fixed_lambda_1e-3_epoch101500/eval/local_energies.txt"
        ),
    },
    "102000": {
        "legacy_hard": Path(
            "/scratch/dexuan1/runs/"
            "C_wssr_rank1600_hard_matched_control_frozen/"
            "epoch102000/eval/local_energies.txt"
        ),
        "fixed_lambda_1e-3": Path(
            "/scratch/dexuan1/runs/"
            "C_wssr_rank1600_fixed_lambda_stage2_E2000_frozen/"
            "epoch102000/eval/local_energies.txt"
        ),
    },
}
TAIL_FRACTIONS = (0.0001, 0.001, 0.01)
EXPECTED_SAMPLES = 40_000_000
CHUNK = 1_000_000


def scalar(value: np.generic | float | int) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite statistic: {result}")
    return result


def audit_file(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.fromfile(path, dtype=np.float64, sep=" ")
    if values.size != EXPECTED_SAMPLES:
        raise ValueError(f"expected {EXPECTED_SAMPLES} values in {path}, got {values.size}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"nonfinite local energy in {path}")

    mean = scalar(np.mean(values, dtype=np.float64))
    variance = scalar(np.var(values, dtype=np.float64))
    minimum = scalar(np.min(values))
    maximum = scalar(np.max(values))

    absolute_deviation = np.abs(values - mean)
    absolute_deviation.sort()
    total_ss = variance * values.size
    tails: dict[str, object] = {}
    for fraction in TAIL_FRACTIONS:
        keep = int(math.floor((1.0 - fraction) * values.size))
        threshold = scalar(absolute_deviation[keep - 1])
        count = 0
        sum_value = 0.0
        sum_square = 0.0
        lower_count = 0
        upper_count = 0
        lower_ss = 0.0
        upper_ss = 0.0
        for start in range(0, values.size, CHUNK):
            block = values[start : start + CHUNK]
            deviation = block - mean
            retained = np.abs(deviation) <= threshold
            kept = block[retained]
            count += int(kept.size)
            sum_value += float(np.sum(kept, dtype=np.float64))
            sum_square += float(np.dot(kept, kept))
            lower = deviation < -threshold
            upper = deviation > threshold
            lower_count += int(np.count_nonzero(lower))
            upper_count += int(np.count_nonzero(upper))
            lower_ss += float(np.dot(deviation[lower], deviation[lower]))
            upper_ss += float(np.dot(deviation[upper], deviation[upper]))
        retained_mean = sum_value / count
        retained_variance = max(sum_square / count - retained_mean**2, 0.0)
        tails[f"{100.0 * fraction:.4f}%"] = {
            "removed_fraction": fraction,
            "absolute_deviation_threshold": threshold,
            "retained_count": count,
            "retained_mean": scalar(retained_mean),
            "trimmed_variance": scalar(retained_variance),
            "trimmed_variance_over_raw": scalar(retained_variance / variance),
            "lower_tail_count": lower_count,
            "upper_tail_count": upper_count,
            "lower_tail_variance_contribution": scalar(lower_ss / total_ss),
            "upper_tail_variance_contribution": scalar(upper_ss / total_ss),
            "total_tail_variance_contribution": scalar((lower_ss + upper_ss) / total_ss),
        }

    return {
        "source_path": str(path),
        "samples": int(values.size),
        "mean": mean,
        "raw_variance": variance,
        "minimum": minimum,
        "maximum": maximum,
        "tails": tails,
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    results: dict[str, dict[str, object]] = {}
    for epoch, methods in RUNS.items():
        results[epoch] = {}
        for method, path in methods.items():
            print(f"auditing {epoch} {method}: {path}", flush=True)
            results[epoch][method] = audit_file(path)

    comparisons: dict[str, object] = {}
    for epoch, methods in results.items():
        legacy = methods["legacy_hard"]
        fixed = methods["fixed_lambda_1e-3"]
        trimmed = {}
        for label in legacy["tails"]:
            legacy_var = legacy["tails"][label]["trimmed_variance"]
            fixed_var = fixed["tails"][label]["trimmed_variance"]
            trimmed[label] = {
                "legacy_variance": legacy_var,
                "fixed_variance": fixed_var,
                "fixed_over_legacy": fixed_var / legacy_var,
                "fixed_reduction_fraction": 1.0 - fixed_var / legacy_var,
            }
        comparisons[epoch] = {
            "raw_fixed_over_legacy": fixed["raw_variance"] / legacy["raw_variance"],
            "raw_fixed_reduction_fraction": 1.0
            - fixed["raw_variance"] / legacy["raw_variance"],
            "trimmed": trimmed,
        }

    payload = {
        "experiment": "C rank1600 fixed-lambda stage-4 tail audit",
        "scientific_caution": (
            "Absolute C energies are not used for physical claims; this audit only "
            "localizes an internal matched variance crossover."
        ),
        "results": results,
        "comparisons": comparisons,
    }
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# C rank1600 fixed-lambda tail audit",
        "",
        "| epoch | trimming | legacy variance | fixed variance | fixed reduction |",
        "|---:|---:|---:|---:|---:|",
    ]
    for epoch in sorted(results):
        legacy = results[epoch]["legacy_hard"]
        fixed = results[epoch]["fixed_lambda_1e-3"]
        raw_ratio = fixed["raw_variance"] / legacy["raw_variance"]
        lines.append(
            f"| {epoch} | none | {legacy['raw_variance']:.8f} | "
            f"{fixed['raw_variance']:.8f} | {100.0 * (1.0 - raw_ratio):.2f}% |"
        )
        for label, comp in comparisons[epoch]["trimmed"].items():
            lines.append(
                f"| {epoch} | {label} | {comp['legacy_variance']:.8f} | "
                f"{comp['fixed_variance']:.8f} | "
                f"{100.0 * comp['fixed_reduction_fraction']:.2f}% |"
            )
    lines.append("")
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload["comparisons"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
