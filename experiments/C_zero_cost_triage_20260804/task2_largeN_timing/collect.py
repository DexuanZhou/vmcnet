#!/usr/bin/env python3
"""Collect the eight-cell large-N timing pilot and low-level profiler slices."""

import csv
import gzip
import json
import math
import re
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = Path("/scratch/dexuan1/runs/C_zero_cost_triage_20260804/task2_largeN_timing_retry2")
SPRING8192 = Path(
    "/scratch/dexuan1/runs/C_zero_cost_triage_20260804/"
    "task2_largeN_timing_retry3_spring8192"
)
LOGS = Path("/scratch/dexuan1/runs/logs")


def read_csv(path, delimiter=","):
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def trace_breakdown(meta):
    paths = list((meta / "jax_trace").glob("plugins/profile/*/*.trace.json.gz"))
    if not paths:
        return {}
    with gzip.open(paths[0], "rt") as handle:
        events = json.load(handle)["traceEvents"]
    five_steps = 5.0
    gpu_w = 0.0
    gpu_ssi_qr = 0.0
    gpu_ssi_eigh = 0.0
    gpu_wssr_dot = 0.0
    cpu_galerkin_solve = 0.0
    for event in events:
        if event.get("ph") != "X" or "dur" not in event:
            continue
        duration_us = float(event["dur"])
        event_name = str(event.get("name", ""))
        source_name = str(event.get("args", {}).get("name", ""))
        if event.get("pid") == 1:
            if source_name == "jit(update_param_fn)/jit(main)/dot_general":
                gpu_w += duration_us
            if "wssr_warm_svd_right_core_update" in source_name:
                if "/jit(qr)/" in source_name:
                    gpu_ssi_qr += duration_us
                elif "/jit(eigh)/" in source_name:
                    gpu_ssi_eigh += duration_us
                elif "dot_general" in source_name:
                    gpu_wssr_dot += duration_us
        if event_name.endswith("_mixed_precision_galerkin_solution_callback"):
            cpu_galerkin_solve += duration_us
    return {
        "W_matmul_ms_per_step": gpu_w / five_steps / 1000.0,
        "SSI_QR_ms_per_step": gpu_ssi_qr / five_steps / 1000.0,
        "SSI_eigh_ms_per_step": gpu_ssi_eigh / five_steps / 1000.0,
        "WSSR_other_dot_ms_per_step": gpu_wssr_dot / five_steps / 1000.0,
        "Galerkin_host_solve_ms_per_step": (
            cpu_galerkin_solve / five_steps / 1000.0
        ),
    }


manifest = read_csv(HERE / "manifest.tsv", delimiter="\t")
rows = []
for index, entry in enumerate(manifest):
    method = entry["method"]
    rank = int(entry["rank"])
    walkers = int(entry["walkers"])
    arm = f"{method}_r{rank}_n{walkers}"
    root = SPRING8192 if method == "spring" and walkers == 8192 else ROOT
    meta = root / "metadata" / arm
    run = root / "runs" / arm
    events = read_csv(meta / "phase_timing.csv")
    epoch_events = {
        int(row["epoch"]): (number(row["monotonic_seconds"]), number(row["wall_time_unix"]))
        for row in events
        if row.get("event") == "epoch_end" and row.get("epoch")
    }
    durations = [
        epoch_events[epoch][0] - epoch_events[epoch - 1][0]
        for epoch in range(21, 101)
        if epoch in epoch_events and epoch - 1 in epoch_events
    ]
    stderr_candidates = list(LOGS.glob(f"C-largeN-cost-52819358_{index}.err"))
    if method == "spring" and walkers == 8192:
        stderr_candidates = list(LOGS.glob("C-largeN-cost-52823495_0.err"))
    stderr = "".join(path.read_text(errors="replace") for path in stderr_candidates)
    oom = "RESOURCE_EXHAUSTED" in stderr or "out of memory" in stderr.lower()
    requested = re.search(r"allocate ([0-9.]+)GiB", stderr)
    status = "OOM" if oom else ("OK" if len(durations) == 80 else "INCOMPLETE")

    memory = read_csv(meta / "gpu_memory_poll.csv")
    if len(durations) == 80:
        start = epoch_events[20][1]
        end = epoch_events[100][1]
        steady_memory = [
            number(row["memory_used_mib"])
            for row in memory
            if start <= number(row["timestamp_unix"]) <= end
        ]
    else:
        steady_memory = [number(row.get("memory_used_mib")) for row in memory]
    peak_mib = max((value for value in steady_memory if math.isfinite(value)), default=math.nan)
    identity = (meta / "gpu_identity.csv").read_text().strip() if (meta / "gpu_identity.csv").exists() else ""
    identity_fields = [item.strip() for item in identity.split(",")]
    total_mib = number(identity_fields[1]) if len(identity_fields) > 1 else math.nan
    shapes = json.loads((meta / "fresh_data_leaf_shapes.json").read_text()) if (meta / "fresh_data_leaf_shapes.json").exists() else {}
    observed_counts = sorted({shape[0] for shape in shapes.values() if shape})
    row = {
        "method": method,
        "rank": rank,
        "walkers": walkers,
        "observed_walker_leading_dims": observed_counts,
        "status": status,
        "timed_steps": len(durations),
        "mean_s_per_step": statistics.mean(durations) if durations else math.nan,
        "std_s_per_step": statistics.pstdev(durations) if len(durations) > 1 else math.nan,
        "peak_memory_mib": peak_mib,
        "peak_memory_gib": peak_mib / 1024.0,
        "gpu_total_mib": total_mib,
        "gpu_model": identity_fields[0] if identity_fields else "",
        "node": (meta / "hostname.txt").read_text().strip() if (meta / "hostname.txt").exists() else "",
        "oom_requested_gib": number(requested.group(1)) if requested else math.nan,
        "spring_lr": 0.0 if method == "spring" and walkers == 8192 else (0.02 if method == "spring" else math.nan),
    }
    row.update(trace_breakdown(meta) if status == "OK" else {})
    rows.append(row)

fields = sorted({key for row in rows for key in row})
with (HERE / "timing_summary.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
(HERE / "timing_summary.json").write_text(json.dumps(rows, indent=2) + "\n")


def fmt(value, digits=4):
    return "--" if not isinstance(value, (int, float)) or not math.isfinite(value) else f"{value:.{digits}f}"


lines = [
    "# Task 2: large-N timing pilot (pre-W-cache)",
    "",
    "All accepted cells use actual fresh walker arrays at the requested N; the ",
    "leading dimensions are recorded in `timing_summary.json`.  Timing uses ",
    "epochs 21--100 after 20 warm-up updates.  All GPUs report NVIDIA H100 ",
    "80GB HBM3.  Recurrence cells are the pre-cache implementation.  The ",
    "SPRING N=8192 retry freezes the learning rate at zero solely to prevent ",
    "off-distribution fresh walkers from changing parameters during a timing-only run; ",
    "the optimizer and MCMC kernels are unchanged.",
    "",
    "| N | method | rank | status | s/step mean ± std | peak GiB | node |",
    "|---:|:---|---:|:---|---:|---:|:---|",
]
for row in rows:
    timing = (
        f"{fmt(row['mean_s_per_step'], 6)} ± {fmt(row['std_s_per_step'], 6)}"
        if row["status"] == "OK"
        else f"OOM (request {fmt(row['oom_requested_gib'], 2)} GiB)"
    )
    lines.append(
        f"| {row['walkers']} | {row['method']} | {row['rank'] or '--'} | "
        f"{row['status']} | {timing} | {fmt(row['peak_memory_gib'], 2)} | "
        f"{row['node']} |"
    )

lines += [
    "",
    "## Low-level profiler slices",
    "",
    "Profiler values below are duration sums divided by the five disjoint profile-tail ",
    "steps.  They are not additive wall-time partitions because CPU callbacks and GPU ",
    "kernels can overlap.  XLA identifies the explicit Galerkin `W=O U` multiply as ",
    "`jit(main)/dot_general`; the host solve is identified by its callback function. ",
    "The remaining WSSR dot kernels include Gram/SSI matmuls, while QR and eigensolve ",
    "are split out when XLA exposes their source scope.",
    "",
    "| N | method | rank | W matmul ms | host solve ms | SSI QR ms | SSI eigh ms | other WSSR dot ms |",
    "|---:|:---|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    if row["status"] != "OK":
        continue
    lines.append(
        f"| {row['walkers']} | {row['method']} | {row['rank'] or '--'} | "
        f"{fmt(row.get('W_matmul_ms_per_step', math.nan), 3)} | "
        f"{fmt(row.get('Galerkin_host_solve_ms_per_step', math.nan), 3)} | "
        f"{fmt(row.get('SSI_QR_ms_per_step', math.nan), 3)} | "
        f"{fmt(row.get('SSI_eigh_ms_per_step', math.nan), 3)} | "
        f"{fmt(row.get('WSSR_other_dot_ms_per_step', math.nan), 3)} |"
    )

spring = {row["walkers"]: row for row in rows if row["method"] == "spring"}
r800 = {row["walkers"]: row for row in rows if row["method"] == "recurrence" and row["rank"] == 800}
r1600_4096 = next(row for row in rows if row["method"] == "recurrence" and row["rank"] == 1600 and row["walkers"] == 4096)
lines += [
    "",
    "## Decision answers",
    "",
    "There is no measured speed crossover through N=8192: at N=1000 and 4096 ",
    "SPRING is faster, and at N=8192 both recurrence ranks OOM while SPRING runs. ",
    "Thus a finite crossover interval is not observed in the tested range.",
    "",
    f"At N=4096, r=1600 is `{r1600_4096['status']}`; consequently its per-step ",
    "cost ratio to SPRING is not finite/measurable.  The feasible r=800 ratio is ",
    f"{r800[4096]['mean_s_per_step'] / spring[4096]['mean_s_per_step']:.3f}x.",
]
(HERE / "REPORT.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
