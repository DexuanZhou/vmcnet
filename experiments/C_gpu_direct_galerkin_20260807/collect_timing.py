#!/usr/bin/env python3
"""Collect paired host/device Galerkin timing without third-party packages."""

import csv
import gzip
import json
import math
from pathlib import Path
import statistics


ROOT = Path("/scratch/dexuan1/runs/C_gpu_direct_galerkin_20260807")
OUT = Path(__file__).resolve().parent


def read_epoch_times(path):
    times = {}
    unix_times = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["event"] == "epoch_end":
                epoch = int(row["epoch"])
                times[epoch] = float(row["monotonic_seconds"])
                unix_times[epoch] = float(row["wall_time_unix"])
    durations = [times[epoch] - times[epoch - 1] for epoch in range(21, 121)]
    return durations, unix_times


def peak_memory(path, unix_times):
    values = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            timestamp = float(row["timestamp_unix"])
            if unix_times[20] <= timestamp <= unix_times[120]:
                values.append(float(row["memory_used_mib"]))
    return max(values) / 1024.0


def callback_profile_ms(meta):
    traces = list(meta.glob("jax_trace/plugins/profile/*/*.trace.json.gz"))
    if not traces:
        return math.nan, 0
    with gzip.open(traces[0], "rt") as handle:
        events = json.load(handle).get("traceEvents", [])
    matches = [
        event
        for event in events
        if "_mixed_precision_galerkin_solution_callback"
        in event.get("name", "")
        and "dur" in event
    ]
    # Chrome trace duration is in microseconds.
    return sum(float(event["dur"]) for event in matches) / 1000.0, len(matches)


rows = []
for backend in ("host_fp64", "device_cholesky"):
    for walkers in (1000, 4096):
        arm = f"recurrence_r800_n{walkers}"
        meta = ROOT / backend / "metadata" / arm
        run = ROOT / backend / "runs" / arm
        durations, unix_times = read_epoch_times(meta / "phase_timing.csv")
        callback_ms, callback_count = callback_profile_ms(meta)
        final_energy = float((run / "energy.txt").read_text().splitlines()[-1])
        rows.append(
            {
                "backend": backend,
                "walkers": walkers,
                "rank": 800,
                "mean_s_per_step": statistics.mean(durations),
                "std_s_per_step": statistics.stdev(durations),
                "peak_gpu_gib": peak_memory(
                    meta / "gpu_memory_poll.csv", unix_times
                ),
                "galerkin_host_callback_ms_5_steps": callback_ms,
                "galerkin_host_callback_count": callback_count,
                "final_energy_timing_only": final_energy,
            }
        )

csv_path = OUT / "timing_summary.csv"
with csv_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

lines = [
    "# GPU-direct Galerkin timing result",
    "",
    "| backend | walkers | mean s/step | std | peak GPU GiB | host Galerkin callback / 5 steps |",
    "|---|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['backend']} | {row['walkers']} | "
        f"{row['mean_s_per_step']:.6f} | {row['std_s_per_step']:.6f} | "
        f"{row['peak_gpu_gib']:.3f} | "
        f"{row['galerkin_host_callback_ms_5_steps']:.3f} ms "
        f"({row['galerkin_host_callback_count']} calls) |"
    )
for walkers in (1000, 4096):
    host = next(
        row
        for row in rows
        if row["backend"] == "host_fp64" and row["walkers"] == walkers
    )
    device = next(
        row
        for row in rows
        if row["backend"] == "device_cholesky" and row["walkers"] == walkers
    )
    speedup = host["mean_s_per_step"] / device["mean_s_per_step"]
    lines.append(
        f"\n- N={walkers}: device speedup = {speedup:.3f}x "
        f"({100.0 * (1.0 - 1.0 / speedup):.2f}% wall-time reduction)."
    )
(OUT / "RESULT.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
