#!/usr/bin/env python3
"""Collect the matched 500-epoch warm-SVD-right screening runs."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path("/scratch/dexuan1/runs/wssr4096_screen")
OUT = Path(__file__).resolve().parent / "results"
REF_C = -37.84471
TAIL = 200
ROLLING = 100


def mean_sem(xs):
    return statistics.mean(xs), statistics.stdev(xs) / math.sqrt(len(xs))


def collect(system, run):
    path = ROOT / system / run
    meta = ROOT / system / "metadata" / run
    cfg = json.loads((path / "config.json").read_text())
    reload = json.loads((path / "reload_config.json").read_text())
    opt = cfg["vmc"]["optimizer"]["wssr_warm_svd_right"]
    rows = list(csv.DictReader((path / "training_metrics.csv").open()))
    numeric = []
    for r in rows:
        try:
            numeric.append({k: float(r[k]) for k in ("epoch", "energy", "variance", "accept_ratio")})
        except (KeyError, ValueError):
            pass
    finite = [r for r in numeric if all(math.isfinite(r[k]) for k in ("energy", "variance", "accept_ratio"))]
    tail = finite[-TAIL:]
    emean, esem = mean_sem([r["energy"] for r in tail])
    vmean = statistics.mean(r["variance"] for r in tail)
    amed = statistics.median(r["accept_ratio"] for r in tail)
    roll = [(statistics.mean(r["energy"] for r in finite[i-ROLLING+1:i+1]), int(finite[i]["epoch"])) for i in range(ROLLING-1, len(finite))]
    best_e, best_epoch = min(roll) if roll else (float("nan"), -1)
    timing = list(csv.DictReader((meta / "phase_timing.csv").open()))
    ends = [(int(r["epoch"]), float(r["monotonic_seconds"])) for r in timing if r["event"] == "epoch_end"]
    deltas = [ends[i][1] - ends[i-1][1] for i in range(1, len(ends)) if ends[i][0] >= 11]
    steady = statistics.median(deltas) if deltas else float("nan")
    mem_rows = list(csv.DictReader((meta / "gpu_memory_poll.csv").open()))
    max_mem = max((float(r["memory_used_mib"]) for r in mem_rows), default=float("nan"))
    early = json.loads((meta / "early_stop.json").read_text()) if (meta / "early_stop.json").exists() else {}
    variances = [r["variance"] for r in finite]
    baseline = statistics.median(variances[:20]) if len(variances) >= 20 else float("nan")
    runaway = bool(baseline > 0 and max(variances, default=0) > 100 * baseline)
    return {
        "system": system, "run": run,
        "checkpoint": str(Path(reload["logdir"]) / reload["checkpoint_relative_file_path"]),
        "nchains": cfg["vmc"]["nchains"], "optimizer_type": cfg["vmc"]["optimizer_type"],
        "rank": opt["sr_rank"], "eta": opt["eta"], "learning_rate": opt["learning_rate"],
        "damping": opt["damping"], "warm": opt["svd_maxiter_warm"],
        "norm_constraint": opt["norm_constraint"], "constrain_norm": opt["constrain_norm"],
        "completed_epochs": len(numeric), "finite_epochs": len(finite),
        "termination_reason": early.get("reason", "missing"),
        "tail_energy": emean, "tail_sem": esem, "tail_variance": vmean,
        "median_acceptance": amed, "steady_median_sec_epoch": steady,
        "best_rolling_energy": best_e, "best_rolling_end_epoch": best_epoch,
        "max_gpu_memory_mib": max_mem, "max_variance": max(variances, default=float("nan")),
        "variance_runaway": runaway, "nonfinite": len(finite) != len(numeric),
        "c_error_mha": 1000 * (emean - REF_C) if system == "C" else "",
        "stable": len(numeric) == EXPECTED and len(finite) == EXPECTED and not runaway and not early.get("triggered", False),
    }


def main():
    global ROOT, TAIL, ROLLING, EXPECTED
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--expected", type=int, default=500)
    p.add_argument("--tail", type=int, default=200)
    p.add_argument("--rolling", type=int, default=100)
    p.add_argument("--output", default="screening_summary.csv")
    a = p.parse_args()
    ROOT, EXPECTED, TAIL, ROLLING = a.root, a.expected, a.tail, a.rolling
    results = []
    for system in ("C", "N2"):
        for path in sorted((ROOT / system).iterdir()):
            if path.is_dir() and path.name != "metadata" and (path / "training_metrics.csv").exists():
                results.append(collect(system, path.name))
    OUT.mkdir(parents=True, exist_ok=True)
    fields = list(results[0])
    with (OUT / a.output).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(results)
    print(f"system run epochs finite E_tail{TAIL} SEM var_tail{TAIL} accept med_s/ep maxMiB best{ROLLING}@end stable")
    for r in results:
        print(f"{r['system']:2} {r['run']:30} {r['completed_epochs']:3} {r['finite_epochs']:3} "
              f"{r['tail_energy']:.8f} {r['tail_sem']:.2g} {r['tail_variance']:.5g} "
              f"{r['median_acceptance']:.4f} {r['steady_median_sec_epoch']:.3f} {r['max_gpu_memory_mib']:.0f} "
              f"{r['best_rolling_energy']:.8f}@{r['best_rolling_end_epoch']} {r['stable']}")


if __name__ == "__main__":
    main()
