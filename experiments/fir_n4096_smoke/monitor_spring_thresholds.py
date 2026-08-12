#!/usr/bin/env python3
"""Terminate one diagnostic run when a requested stability threshold is crossed."""
import argparse, csv, json, math, os, pathlib, signal, statistics, time

p = argparse.ArgumentParser()
p.add_argument("--pid", type=int, required=True)
p.add_argument("--run", type=pathlib.Path, required=True)
p.add_argument("--output", type=pathlib.Path, required=True)
a = p.parse_args()

def alive():
    try: os.kill(a.pid, 0); return True
    except ProcessLookupError: return False

def read(path):
    try:
        with path.open(newline="") as f: return list(csv.DictReader(f))
    except (FileNotFoundError, OSError): return []

status = {"triggered": False, "reason": "none", "epoch": None}
seen = 0
while alive():
    training = read(a.run / "training_metrics.csv")
    diagnostics = read(a.run / "spring_diagnostics.csv")
    n = min(len(training), len(diagnostics))
    if n > seen:
        for i in range(seen, n):
            epoch = i + 1
            try:
                energy = float(training[i]["energy"])
                variance = float(training[i]["variance"])
                acceptance = float(training[i]["accept_ratio"])
                raw = float(diagnostics[i]["spring_diag_raw_solution_norm"])
            except (KeyError, TypeError, ValueError):
                continue
            if not all(math.isfinite(x) for x in (energy, variance, acceptance, raw)):
                status = {"triggered": True, "reason": "nonfinite_metric", "epoch": epoch}
            elif n >= 10:
                initial_variance = statistics.median(
                    float(row["variance"]) for row in training[:10]
                )
                initial_raw = statistics.median(
                    float(row["spring_diag_raw_solution_norm"])
                    for row in diagnostics[:10]
                )
                if variance > 100 * initial_variance:
                    status = {"triggered": True, "reason": "variance_gt_100x_initial10_median", "epoch": epoch}
                elif raw > 100 * initial_raw:
                    status = {"triggered": True, "reason": "raw_direction_gt_100x_initial10_median", "epoch": epoch}
            if status["triggered"]:
                a.output.parent.mkdir(parents=True, exist_ok=True)
                a.output.write_text(json.dumps(status, indent=2) + "\n")
                os.kill(a.pid, signal.SIGTERM)
                raise SystemExit(0)
        seen = n
    time.sleep(0.2)

a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(status, indent=2) + "\n")
