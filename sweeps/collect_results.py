#!/usr/bin/env python3
"""Collect all manifest-listed pilot runs into tidy CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "pilot" / "runs.csv"
DEFAULT_OUT = HERE / "results"
LOG_DIR = Path("/scratch/dexuan1/runs/logs")
REFERENCE_ENERGIES = {"C": -37.84471, "N2eq": -109.5388}
NUMBER = r"(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan)"
EPOCH_RE = re.compile(
    rf"INFO:root:Epoch\s+(?P<epoch>\d+), Energy:\s*(?P<energy>{NUMBER})"
    rf"\s*\([^)]*\), Variance:\s*(?P<variance>{NUMBER})\s*\([^)]*\),"
    rf" Accept ratio:\s*(?P<accept_ratio>{NUMBER}), Energy smooth20:\s*"
    rf"(?P<energy_smooth20>{NUMBER}), Variance smooth20:\s*"
    rf"(?P<variance_smooth20>{NUMBER}), Accept smooth20:\s*"
    rf"(?P<accept_smooth20>{NUMBER})"
)


def _float(value, default=math.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean_sem(values):
    if not values:
        return math.nan, math.nan
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, math.nan
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance / len(values))


def _read_csv(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _job_name(script_path: Path):
    if not script_path.exists():
        return ""
    with script_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#SBATCH --job-name="):
                return line.split("=", 1)[1].strip()
    return ""


def _stderr_metrics(script_name: str):
    if not script_name:
        return []
    job_name = _job_name(HERE / "pilot" / script_name)
    logs = list(LOG_DIR.glob(f"{job_name}-*.err")) if job_name else []
    if not logs:
        return []
    path = max(logs, key=lambda candidate: candidate.stat().st_mtime)
    rows = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = EPOCH_RE.search(line)
            if match:
                rows.append(match.groupdict())
    return rows


def _run_dirs(run_dir: Path):
    """Return the base run directory and VMCNet's numeric-suffix variants."""
    candidates = []
    if run_dir.is_dir():
        candidates.append((0, run_dir))
    for path in run_dir.parent.glob(f"{run_dir.name}_*"):
        suffix = path.name.removeprefix(f"{run_dir.name}_")
        if path.is_dir() and suffix.isdigit():
            candidates.append((int(suffix), path))
    return [path for _, path in sorted(candidates)]


def _data_dir(run_dir: Path, run_dirs):
    """Prefer the newest directory containing an actual checkpoint."""
    checkpoint_dirs = []
    for path in run_dirs:
        checkpoints = list((path / "checkpoints").glob("*.npz"))
        if checkpoints:
            checkpoint_dirs.append(
                (max(item.stat().st_mtime for item in checkpoints), path)
            )
    if checkpoint_dirs:
        return max(checkpoint_dirs, key=lambda item: item[0])[1]
    metric_dirs = [path for path in run_dirs if (path / "training_metrics.csv").exists()]
    if metric_dirs:
        return max(
            metric_dirs,
            key=lambda path: (path / "training_metrics.csv").stat().st_mtime,
        )
    return run_dir


def _first_file(run_dirs, filename: str, preferred: Path):
    for path in [preferred, *reversed(run_dirs)]:
        candidate = path / filename
        if candidate.exists():
            return candidate
    return None


def _peak_gpu_memory(paths):
    values = []
    for path in paths:
        for row in _read_csv(path):
            # nvidia-smi may retain a leading space after the comma.
            value = row.get("memory_used_mib")
            if value is None:
                value = row.get(" memory_used_mib")
            number = _float(value)
            if math.isfinite(number):
                values.append(number)
    return max(values, default=math.nan)


def _timing(path: Path):
    rows = _read_csv(path)
    if not rows:
        return math.nan, ""
    return _float(rows[-1].get("elapsed_seconds")), rows[-1].get("status", "")


def _last_finite_scalar(path: Path):
    if not path.exists():
        return math.nan
    values = [_float(value.strip()) for value in path.read_text().splitlines()]
    return next((value for value in reversed(values) if math.isfinite(value)), math.nan)


def collect(manifest_path: Path, out_dir: Path):
    manifest = _read_csv(manifest_path)
    if not manifest:
        raise ValueError("Manifest is empty")
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries, curves = [], []

    for spec in manifest:
        output_dir = spec.get("output_dir", "")
        if output_dir:
            run_dir = Path(output_dir)
        else:
            batch = manifest_path.parent.name
            run_dir = Path("/scratch/dexuan1/runs") / batch / spec["system"] / spec["run_name"]
        run_dirs = _run_dirs(run_dir)
        data_dir = _data_dir(run_dir, run_dirs)
        # Prefer the run-local metrics: repeated submissions can share a Slurm job
        # name, so selecting the newest stderr would attach one run to another.
        metrics = _read_csv(data_dir / "training_metrics.csv")
        if not metrics:
            metrics = _stderr_metrics(spec.get("script", ""))
        metadata = {}
        metadata_path = _first_file(run_dirs, "run_metadata.json", data_dir)
        if metadata_path is not None:
            with metadata_path.open(encoding="utf-8") as handle:
                metadata = json.load(handle)
        config = {}
        config_path = data_dir / "config.json"
        if config_path.exists():
            with config_path.open(encoding="utf-8") as handle:
                config = json.load(handle)
        wssr_config = (
            config.get("vmc", {})
            .get("optimizer", {})
            .get("wssr_warm_svd_right", {})
        )
        timing_path = _first_file(run_dirs, "run_timing.csv", data_dir)
        elapsed, exit_status = (
            _timing(timing_path) if timing_path is not None else (math.nan, "")
        )
        expected_epochs = int(spec.get("nepochs") or metadata.get("nepochs") or 0)
        observed_epochs = len(metrics)
        energies = [_float(row.get("energy")) for row in metrics]
        finite_energy_epochs = sum(math.isfinite(value) for value in energies)
        first_nonfinite_epoch = next(
            (index for index, value in enumerate(energies, start=1) if not math.isfinite(value)),
            "",
        )
        if not metrics:
            status = "missing"
        elif (
            exit_status == "0"
            and observed_epochs >= expected_epochs
            and finite_energy_epochs >= expected_epochs
        ):
            status = "complete"
        else:
            status = "partial"

        tail = [value for value in energies[-50:] if math.isfinite(value)]
        tail100 = [value for value in energies[-100:] if math.isfinite(value)]
        tail100_mean, tail100_sem = _mean_sem(tail100)
        tail500 = [value for value in energies[-500:] if math.isfinite(value)]
        tail500_mean, tail500_sem = _mean_sem(tail500)
        tail1000 = [value for value in energies[-1000:] if math.isfinite(value)]
        tail1000_mean, tail1000_sem = _mean_sem(tail1000)
        variances = [_float(row.get("variance")) for row in metrics]
        accepts = [_float(row.get("accept_ratio")) for row in metrics]
        variance_tail = [value for value in variances[-1000:] if math.isfinite(value)]
        acceptance_tail = [value for value in accepts[-1000:] if math.isfinite(value)]
        reference_energy = REFERENCE_ENERGIES.get(spec["system"], math.nan)
        final = metrics[-1] if metrics else {}
        summary = dict(spec)
        if spec["family"] == "wssr":
            summary["spectral_reg"] = wssr_config.get(
                "spectral_regularization", spec.get("spectral_reg", "")
            )
        summary.update(
            {
                "status": status,
                "observed_epochs": observed_epochs,
                "finite_energy_epochs": finite_energy_epochs,
                "first_nonfinite_epoch": first_nonfinite_epoch,
                "final_energy": _float(final.get("energy")),
                "final_energy_smooth20": _float(final.get("energy_smooth20")),
                "tail50_energy_mean": sum(tail) / len(tail) if tail else math.nan,
                "tail100_energy_mean": tail100_mean,
                "tail100_energy_sem": tail100_sem,
                "tail500_energy_mean": tail500_mean,
                "tail500_energy_sem": tail500_sem,
                "tail500_error_mha": (tail500_mean - reference_energy) * 1000
                if math.isfinite(tail500_mean) and math.isfinite(reference_energy)
                else math.nan,
                "tail1000_energy_mean": tail1000_mean,
                "tail1000_energy_sem": tail1000_sem,
                "tail1000_error_mha": (tail1000_mean - reference_energy) * 1000
                if math.isfinite(tail1000_mean) and math.isfinite(reference_energy)
                else math.nan,
                "variance_tail1000_mean": sum(variance_tail) / len(variance_tail)
                if variance_tail else math.nan,
                "acceptance_tail1000_mean": sum(acceptance_tail) / len(acceptance_tail)
                if acceptance_tail else math.nan,
                "active_rank": _last_finite_scalar(data_dir / "wssr_active_rank.txt"),
                "reference_energy": reference_energy,
                "error_mha": (tail100_mean - reference_energy) * 1000
                if math.isfinite(tail100_mean) and math.isfinite(reference_energy)
                else math.nan,
                "final_variance": _float(final.get("variance")),
                "peak_gpu_memory_mib": _peak_gpu_memory(
                    path / "gpumem.csv" for path in run_dirs
                ),
                "elapsed_seconds": elapsed,
                "seconds_per_epoch": elapsed / observed_epochs
                if math.isfinite(elapsed) and observed_epochs
                else math.nan,
                "exit_status": exit_status,
                "metadata_present": bool(metadata),
                "data_dir": str(data_dir),
                "checkpoint_present": any((data_dir / "checkpoints").glob("*.npz")),
            }
        )
        summaries.append(summary)

        for index, row in enumerate(metrics, start=1):
            curves.append(
                {
                    "system": spec["system"],
                    "run_name": spec["run_name"],
                    "family": spec["family"],
                    "epoch": row.get("epoch", index),
                    "energy": _float(row.get("energy")),
                    "energy_smooth20": _float(row.get("energy_smooth20")),
                    "variance": _float(row.get("variance")),
                    "accept_ratio": _float(row.get("accept_ratio")),
                }
            )

    summary_fields = list(summaries[0])
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)
    curve_fields = ["system", "run_name", "family", "epoch", "energy", "energy_smooth20", "variance", "accept_ratio"]
    with (out_dir / "curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=curve_fields)
        writer.writeheader()
        writer.writerows(curves)
    return summaries, curves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    summaries, curves = collect(args.manifest, args.out_dir)
    complete = sum(row["status"] == "complete" for row in summaries)
    print(f"Collected {len(summaries)} runs ({complete} complete), {len(curves)} metric rows into {args.out_dir}")


if __name__ == "__main__":
    main()
