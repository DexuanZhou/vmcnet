#!/usr/bin/env python3
"""Collect corrected N2 n=4096 WSSR smoke results without touching old data."""
import csv
import math
import pathlib
import re
import statistics
import os

PKG = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path(os.environ.get("CORRECTED_ROOT", "/scratch/dexuan1/runs/fir_n4096_corrected/N2"))
TAG = os.environ.get("CORRECTED_TAG", "")
OLD_ROOT = pathlib.Path("/scratch/dexuan1/runs/fir_n4096_smoke")
RESULTS = PKG / "results"
LOG_ROOT = pathlib.Path("/scratch/dexuan1/runs/logs")
COMBOS = [(400, 1), (800, 2), (1600, 2)]
EXPECTED_ENERGY = -109.48


def rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(value, default=math.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def finite_metric_rows(data):
    return [r for r in data if all(math.isfinite(number(r.get(k))) for k in ("energy", "variance", "accept_ratio"))]


def epoch_times(meta):
    data = rows(meta / "phase_timing.csv")
    stamps = {}
    for r in data:
        if r.get("event") == "epoch_end":
            stamps[int(r["epoch"])] = number(r["monotonic_seconds"])
    return [stamps[e] - stamps[e - 1] for e in range(11, 51) if e in stamps and e - 1 in stamps]


def gpu_stats(meta):
    identity = rows(meta / "gpu_identity.csv")
    if identity:
        first = identity[0]
        model = first.get("name", "")
        total = number(first.get("memory.total [MiB]", first.get("memory_total_mib")))
    else:
        raw = (meta / "gpu_identity.csv").read_text().strip().split(",") if (meta / "gpu_identity.csv").exists() else []
        model = raw[0].strip() if raw else ""
        total = number(raw[1]) if len(raw) > 1 else math.nan
    samples = rows(meta / "gpu_memory_poll.csv")
    used = [number(r.get("memory_used_mib")) for r in samples]
    used = [x for x in used if math.isfinite(x)]
    maximum = max(used) if used else math.nan
    return model, total, maximum, total - maximum


def active_rank(run):
    path = run / "wssr_active_rank.txt"
    if not path.exists():
        return ""
    values = re.findall(r"[-+]?\d+(?:\.\d+)?", path.read_text())
    return number(values[-1]) if values else ""


def old_run(rank, warm):
    prefix = f"N2eq_R2068_rank{rank}_warm{warm}_n4096_e50"
    candidates = sorted(OLD_ROOT.glob(prefix + "*/training_metrics.csv"))
    if not candidates:
        return {}, []
    return candidates[-1].parent, rows(candidates[-1])


def fmt(value, digits=3):
    if value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


RESULTS.mkdir(exist_ok=True)
output = []
comparison = []
for task, (rank, warm) in enumerate(COMBOS):
    name = f"N2eq_R2068_rank{rank}_warm{warm}_n4096_corrected_e50"
    run = ROOT / "wssr_smoke" / name
    meta = ROOT / "metadata" / name
    metrics = rows(run / "training_metrics.csv")
    finite = finite_metric_rows(metrics)
    times = epoch_times(meta)
    model, total, maximum, margin = gpu_stats(meta)
    stderr = "".join(p.read_text(errors="replace") for p in LOG_ROOT.glob(f"N2-wssr4096-corrected-*_{task}.err"))
    oom = bool(re.search(r"out of memory|RESOURCE_EXHAUSTED|CUDA_ERROR_OUT_OF_MEMORY", stderr, re.I))
    nonfinite = len(finite) != len(metrics)
    first = metrics[0] if metrics else {}
    final = metrics[-1] if metrics else {}
    first_energy = number(first.get("energy"))
    final_energy = number(final.get("energy"))
    final_variance = number(final.get("variance"))
    acceptance = number(final.get("accept_ratio"))
    scientific = (
        len(metrics) == 50 and len(finite) == 50
        and abs(first_energy - EXPECTED_ENERGY) < 2.0
        and abs(final_energy - EXPECTED_ENERGY) < 2.0
        and final_variance < 100.0
        and 0.2 <= acceptance <= 0.8
    )
    if oom:
        status = "OOM"
    elif nonfinite:
        status = "NONFINITE"
    elif len(metrics) == 50 and scientific:
        status = "SCIENTIFICALLY_VALID_RUNNABLE"
    elif len(metrics) == 50:
        status = "RESOURCE_RUNNABLE_BUT_SAMPLING_INVALID"
    else:
        status = "FAILED_OTHER"
    mean = statistics.fmean(times) if times else math.nan
    median = statistics.median(times) if times else math.nan
    std = statistics.stdev(times) if len(times) > 1 else math.nan
    old_dir, old_metrics = old_run(rank, warm)
    old_first = old_metrics[0] if old_metrics else {}
    old_times = epoch_times(old_dir) if old_dir else []
    old_median = statistics.median(old_times) if old_times else math.nan
    output.append({
        "system": "N2_R2068", "rank": rank, "warm": warm, "gpu_model": model,
        "gpu_total_mib": total, "max_gpu_memory_mib": maximum, "memory_margin_mib": margin,
        "steady_mean_sec_per_epoch": mean, "steady_median_sec_per_epoch": median,
        "steady_std_sec_per_epoch": std,
        "estimated_25000_hours": median * 25000 / 3600,
        "estimated_100000_hours": median * 100000 / 3600,
        "epoch1_energy": first_energy, "final_energy": final_energy,
        "final_variance": final_variance, "acceptance": acceptance,
        "final_active_rank": active_rank(run), "finite_epochs": len(finite), "status": status,
    })
    comparison.append({
        "rank": rank, "warm": warm,
        "old_epoch1_energy": number(old_first.get("energy")),
        "corrected_epoch1_energy": first_energy,
        "old_epoch1_acceptance": number(old_first.get("accept_ratio")),
        "corrected_epoch1_acceptance": number(first.get("accept_ratio")),
        "old_epoch1_variance": number(old_first.get("variance")),
        "corrected_epoch1_variance": number(first.get("variance")),
        "old_steady_median_sec_per_epoch": old_median,
        "corrected_steady_median_sec_per_epoch": median,
        "steady_time_ratio_corrected_over_old": median / old_median if math.isfinite(median) and math.isfinite(old_median) else math.nan,
    })

fields = list(output[0])
with (RESULTS / f"n2_corrected_smoke_summary{TAG}.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader(); writer.writerows(output)

lines = [
    "# Corrected N2 n=4096 WSSR smoke results", "",
    "| rank | warm | GPU used/total MiB | margin MiB | steady mean/median/std s | 25k h | 100k h | epoch-1 E | final E | final variance | acceptance | active rank | status |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
]
for r in output:
    lines.append(f"| {r['rank']} | {r['warm']} | {fmt(r['max_gpu_memory_mib'],0)}/{fmt(r['gpu_total_mib'],0)} | {fmt(r['memory_margin_mib'],0)} | {fmt(r['steady_mean_sec_per_epoch'])}/{fmt(r['steady_median_sec_per_epoch'])}/{fmt(r['steady_std_sec_per_epoch'])} | {fmt(r['estimated_25000_hours'],2)} | {fmt(r['estimated_100000_hours'],2)} | {fmt(r['epoch1_energy'],7)} | {fmt(r['final_energy'],7)} | {fmt(r['final_variance'],5)} | {fmt(r['acceptance'],4)} | {fmt(r['final_active_rank'],0)} | {r['status']} |")
lines += ["", "## Previous invalid fresh-walker smoke comparison", "",
          "| rank | warm | old/corrected epoch-1 E | old/corrected acceptance | old/corrected variance | old/corrected steady median s | corrected/old timing |",
          "|---:|---:|---:|---:|---:|---:|---:|"]
for r in comparison:
    lines.append(f"| {r['rank']} | {r['warm']} | {fmt(r['old_epoch1_energy'],7)}/{fmt(r['corrected_epoch1_energy'],7)} | {fmt(r['old_epoch1_acceptance'],4)}/{fmt(r['corrected_epoch1_acceptance'],4)} | {fmt(r['old_epoch1_variance'],4)}/{fmt(r['corrected_epoch1_variance'],4)} | {fmt(r['old_steady_median_sec_per_epoch'])}/{fmt(r['corrected_steady_median_sec_per_epoch'])} | {fmt(r['steady_time_ratio_corrected_over_old'])} |")
(RESULTS / f"n2_corrected_smoke_summary{TAG}.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
