"""Collect and assess the C WSSR reliability-indicator replay."""

from __future__ import annotations

import csv
import json
import math
import zipfile
from pathlib import Path

import numpy as np
from scipy import stats


SOURCE = Path("/scratch/dexuan1/runs/C_wssr_rank1600_fixed_lambda_stage2_E2000")
RUN = Path("/scratch/dexuan1/runs/C_wssr_rank1600_reliability_indicator_replay_E500")
RESULTS = Path(
    "/scratch/dexuan1/runs/C_wssr_rank1600_reliability_indicator_replay_E500_results"
)
SOURCE_CHECKPOINT_LABEL = 101500
FIRST_REPLAY_EPOCH = 101500
END_EPOCH = 102000

INDICATORS = {
    "update_weighted_ritz_residual": (
        "wssr_reliability_update_weighted_ritz_residual"
    ),
    "ssi_update_relative_change": (
        "wssr_reliability_ssi_update_relative_change"
    ),
    "ssi_update_one_minus_cosine": (
        "wssr_reliability_ssi_update_cosine"
    ),
    "temporal_update_novelty": (
        "wssr_reliability_temporal_update_novelty"
    ),
}
TARGETS = {
    "raw_tail_max_robust_z": "wssr_reliability_raw_tail_max_robust_z",
    "raw_tail_q99_robust_z": "wssr_reliability_raw_tail_q99_robust_z",
}


def read_metric(name: str) -> np.ndarray:
    path = RUN / f"{name}.txt"
    values = np.asarray(
        [float(line) for line in path.read_text(encoding="utf-8").splitlines()],
        dtype=np.float64,
    )
    expected_count = END_EPOCH - FIRST_REPLAY_EPOCH + 1
    if values.shape != (expected_count,):
        raise ValueError(
            f"{path}: expected {expected_count} values, got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: non-finite values")
    return values


def checkpoint_crc_manifest(path: Path) -> dict[str, tuple[int, int]]:
    with zipfile.ZipFile(path) as archive:
        return {
            item.filename: (item.CRC, item.file_size)
            for item in archive.infolist()
        }


def correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return {"pearson": math.nan, "spearman": math.nan}
    return {
        "pearson": float(stats.pearsonr(x, y).statistic),
        "spearman": float(stats.spearmanr(x, y).statistic),
    }


def future_max(values: np.ndarray, horizon: int) -> np.ndarray:
    return np.asarray(
        [np.max(values[index + 1 : index + horizon + 1])
         for index in range(len(values) - horizon)],
        dtype=np.float64,
    )


def predictive_summary(indicator: np.ndarray, target: np.ndarray, horizon: int) -> dict:
    x = indicator[:-horizon]
    y = future_max(target, horizon)
    indicator_cutoff = float(np.quantile(x, 0.9))
    target_cutoff = float(np.quantile(y, 0.9))
    selected = x >= indicator_cutoff
    event = y >= target_cutoff
    selected_mean = float(np.mean(y[selected]))
    unselected_mean = float(np.mean(y[~selected]))
    return {
        **correlation(x, y),
        "indicator_top_decile_cutoff": indicator_cutoff,
        "future_target_top_decile_cutoff": target_cutoff,
        "future_target_mean_indicator_top_decile": selected_mean,
        "future_target_mean_indicator_bottom_90pct": unselected_mean,
        "top_decile_enrichment_ratio": selected_mean / unselected_mean,
        "event_precision_at_indicator_top_decile": float(np.mean(event[selected])),
        "event_base_rate": float(np.mean(event)),
    }


def source_rows() -> list[dict[str, str]]:
    with (SOURCE / "training_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row for row in rows
        if FIRST_REPLAY_EPOCH <= int(row["epoch"]) <= END_EPOCH
    ]
    expected_count = END_EPOCH - FIRST_REPLAY_EPOCH + 1
    if len(selected) != expected_count:
        raise ValueError(
            f"expected {expected_count} source rows, got {len(selected)}"
        )
    return selected


def replay_rows() -> list[dict[str, str]]:
    with (RUN / "training_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if [int(row["epoch"]) for row in rows] != list(
        range(FIRST_REPLAY_EPOCH, END_EPOCH + 1)
    ):
        raise ValueError("replay epochs are not exactly 101500..102000")
    return rows


def main() -> None:
    if RESULTS.exists():
        raise FileExistsError(f"refusing to overwrite {RESULTS}")

    indicators = {name: read_metric(path) for name, path in INDICATORS.items()}
    indicators["ssi_update_one_minus_cosine"] = (
        1.0 - indicators["ssi_update_one_minus_cosine"]
    )
    source = source_rows()
    replay = replay_rows()
    targets = {name: read_metric(path) for name, path in TARGETS.items()}
    targets["raw_variance"] = np.asarray(
        [float(row["variance_noclip"]) for row in replay], dtype=np.float64
    )

    trajectory_comparison = {}
    for field in ("energy", "energy_noclip", "variance", "variance_noclip", "accept_ratio"):
        source_values = np.asarray([float(row[field]) for row in source])
        replay_values = np.asarray([float(row[field]) for row in replay])
        trajectory_comparison[field] = {
            "array_equal": bool(np.array_equal(source_values, replay_values)),
            "max_absolute_difference": float(
                np.max(np.abs(source_values - replay_values))
            ),
            "mean_absolute_difference": float(
                np.mean(np.abs(source_values - replay_values))
            ),
        }

    source_manifest = checkpoint_crc_manifest(SOURCE / "checkpoints/102000.npz")
    replay_manifest = checkpoint_crc_manifest(RUN / "checkpoints/102000.npz")
    checkpoint_exact = source_manifest == replay_manifest

    predictive = {
        indicator_name: {
            target_name: {
                str(horizon): predictive_summary(
                    indicator_values, target_values, horizon
                )
                for horizon in (1, 5, 10, 20)
            }
            for target_name, target_values in targets.items()
        }
        for indicator_name, indicator_values in indicators.items()
    }
    contemporaneous = {
        indicator_name: {
            target_name: correlation(indicator_values, target_values)
            for target_name, target_values in targets.items()
        }
        for indicator_name, indicator_values in indicators.items()
    }
    strongest = max(
        (
            (abs(result["spearman"]), indicator, target, horizon, result)
            for indicator, target_results in predictive.items()
            for target, horizon_results in target_results.items()
            for horizon, result in horizon_results.items()
            if math.isfinite(result["spearman"])
        ),
        default=None,
    )

    payload = {
        "experiment": "C rank1600 fixed-lambda WSSR read-only reliability replay",
        "source_checkpoint_label": SOURCE_CHECKPOINT_LABEL,
        "first_replay_epoch": FIRST_REPLAY_EPOCH,
        "target_epoch": END_EPOCH,
        "steps": END_EPOCH - FIRST_REPLAY_EPOCH + 1,
        "scientific_caution": (
            "This single 500-step trajectory can identify associations and lead-lag "
            "signals, not establish a causal adaptive rule. C absolute energies remain "
            "unsuitable for physical claims."
        ),
        "update_unchanged_validation": {
            "checkpoint_npz_member_crc_and_size_exact": checkpoint_exact,
            "source_checkpoint": str(SOURCE / "checkpoints/102000.npz"),
            "replay_checkpoint": str(RUN / "checkpoints/102000.npz"),
            "trajectory": trajectory_comparison,
        },
        "indicator_ranges": {
            name: {
                "min": float(np.min(values)),
                "median": float(np.median(values)),
                "max": float(np.max(values)),
            }
            for name, values in indicators.items()
        },
        "target_ranges": {
            name: {
                "min": float(np.min(values)),
                "median": float(np.median(values)),
                "max": float(np.max(values)),
            }
            for name, values in targets.items()
        },
        "contemporaneous_correlations": contemporaneous,
        "future_max_predictive_statistics": predictive,
        "strongest_absolute_future_spearman": (
            None if strongest is None else {
                "absolute_spearman": strongest[0],
                "indicator": strongest[1],
                "target": strongest[2],
                "horizon": int(strongest[3]),
                "statistics": strongest[4],
            }
        ),
        "source_paths": [str(SOURCE), str(RUN)],
    }

    RESULTS.mkdir(parents=True)
    (RESULTS / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# C WSSR reliability-indicator replay",
        "",
        "This is a read-only diagnostic replay; no optimizer update formula was changed.",
        "",
        f"- Epochs: {FIRST_REPLAY_EPOCH}–{END_EPOCH} (501 updates)",
        f"- Final checkpoint arrays exactly match source by NPZ member CRC/size: {checkpoint_exact}",
        "- C absolute energies are not used for physical claims.",
        "",
        "## Strongest lead-lag association",
        "",
        json.dumps(payload["strongest_absolute_future_spearman"], indent=2),
        "",
        "The full contemporaneous and horizon 1/5/10/20 statistics are in `summary.json`.",
        "",
    ]
    (RESULTS / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True))


if __name__ == "__main__":
    main()
