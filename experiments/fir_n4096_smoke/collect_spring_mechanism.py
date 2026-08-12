#!/usr/bin/env python3
"""Collect the matched N2 SPRING mechanism trajectories and make diagnostics plots."""
import csv
import math
import pathlib
import shutil
import statistics

import matplotlib.pyplot as plt

ROOT = pathlib.Path("/scratch/dexuan1/runs/fir_n4096_spring_mechanism_retry2/N2")
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "results" / "spring_mechanism"
OUT.mkdir(parents=True, exist_ok=True)

RUNS = {
    "mu=0": "N2eq_R2068_spring_lr0005_mu000_n4096_e80_diag",
    "mu=0.99": "N2eq_R2068_spring_lr0005_mu099_n4096_e80_diag",
}


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def finite(row, keys):
    return all(math.isfinite(number(row.get(key))) for key in keys)


trajectories = {}
summaries = []
for label, name in RUNS.items():
    source = ROOT / name / "training_metrics.csv"
    if not source.exists():
        continue
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    diagnostics_source = ROOT / name / "spring_diagnostics.csv"
    with diagnostics_source.open(newline="") as handle:
        diagnostic_rows = {row["epoch"]: row for row in csv.DictReader(handle)}
    for row in rows:
        row.update(diagnostic_rows.get(row["epoch"], {}))
        epoch = int(float(row["epoch"]))
        replay_norm = number(row.get("spring_replay_float64_direction_norm"))
        learning_rate = 0.0005 / (1.0 + 1e-4 * (epoch - 1))
        row["spring_replay_float64_norm_constraint_scale_corrected"] = (
            min(1.0, math.sqrt(0.001) / (learning_rate * replay_norm))
            if math.isfinite(replay_norm) and replay_norm > 0 else math.nan
        )
    with (OUT / f"{name}_per_epoch.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    trajectories[label] = rows
    finite_rows = [
        row for row in rows if finite(row, ("energy", "variance", "accept_ratio"))
    ]
    first10 = finite_rows[:10]
    med_raw = statistics.median(number(r["spring_diag_raw_solution_norm"]) for r in first10)
    med_var = statistics.median(number(r["variance"]) for r in first10)
    med_res = statistics.median(number(r["spring_spec_relative_residual"]) for r in first10)
    triggers = []
    for row in finite_rows:
        epoch = int(float(row["epoch"]))
        conditions = {
            "history/nonhistory > 1": number(row.get("spring_diag_mu_history_over_nonhistory")) > 1,
            "raw norm > 5x initial median": number(row.get("spring_diag_raw_solution_norm")) > 5 * med_raw,
            "variance > 5x initial median": number(row.get("variance")) > 5 * med_var,
            "constraint scale < 0.2": number(row.get("spring_diag_norm_constraint_scale")) < 0.2,
            "float32/64 relative error > 1e-2": number(row.get("spring_replay_relative_direction_error")) > 1e-2,
            "solve residual > 100x initial median": number(row.get("spring_spec_relative_residual")) > 100 * med_res,
        }
        for condition, met in conditions.items():
            if met and not any(old[1] == condition for old in triggers):
                triggers.append((epoch, condition))
    onset_epoch, onset_condition = min(triggers, default=("", "none"), key=lambda x: x[0])
    last = finite_rows[-1] if finite_rows else {}
    summaries.append(
        {
            "run": label,
            "rows": len(rows),
            "finite_epochs": len(finite_rows),
            "last_finite_epoch": int(float(last["epoch"])) if last else "",
            "first_nonfinite_epoch": "" if len(rows) == 80 else len(finite_rows) + 1,
            "onset_epoch": onset_epoch,
            "onset_condition": onset_condition,
            "initial_variance_median": med_var,
            "initial_raw_norm_median": med_raw,
            "initial_residual_median": med_res,
            "last_energy": number(last.get("energy")),
            "last_variance": number(last.get("variance")),
            "last_acceptance": number(last.get("accept_ratio")),
            "last_history_ratio": number(last.get("spring_diag_mu_history_over_nonhistory")),
            "last_constraint_scale": number(last.get("spring_diag_norm_constraint_scale")),
        }
    )

if summaries:
    with (OUT / "trajectory_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

spectral_fields = [
    "spring_spec_min_eigenvalue", "spring_spec_max_eigenvalue",
    "spring_spec_min_positive_eigenvalue", "spring_spec_negative_count",
    "spring_spec_below_1e8_count", "spring_spec_below_1e6_count",
    "spring_spec_below_1e4_count", "spring_spec_below_damping_count",
    "spring_spec_below_10damping_count", "spring_spec_effective_condition",
    "spring_spec_max_inverse_filter", "spring_spec_rhs_norm",
    "spring_spec_relative_residual", "spring_spec_low_rhs_fraction",
    "spring_spec_low_correction_fraction",
    "spring_spec_low_history_projection_fraction",
]
selected_spectral = []
for label, rows in trajectories.items():
    selected = {1, 5, 10, 20, 30, int(float(rows[-1]["epoch"]))}
    for row in rows:
        if int(float(row["epoch"])) in selected:
            selected_spectral.append(
                {"run": label, "epoch": row["epoch"],
                 **{field: row.get(field, "") for field in spectral_fields}}
            )
if selected_spectral:
    with (OUT / "selected_spectral_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected_spectral[0].keys())
        writer.writeheader(); writer.writerows(selected_spectral)

selected_replay = []
for label, rows in trajectories.items():
    for row in rows:
        error = number(row.get("spring_replay_relative_direction_error"))
        if math.isfinite(error) and int(float(row["epoch"])) in {1, 8, 10, 20, 30}:
            selected_replay.append({
                "run": label, "epoch": row["epoch"],
                "float32_direction_norm": row.get("spring_diag_raw_solution_norm"),
                "float64_direction_norm": row.get("spring_replay_float64_direction_norm"),
                "relative_direction_error": row.get("spring_replay_relative_direction_error"),
                "direction_cosine": row.get("spring_replay_direction_cosine"),
                "float32_residual": row.get("spring_spec_relative_residual"),
                "float64_residual": row.get("spring_replay_float64_relative_residual"),
                "float32_low_rhs_fraction": row.get("spring_spec_low_rhs_fraction"),
                "float64_low_rhs_fraction": row.get("spring_replay_float64_low_rhs_fraction"),
                "float32_constraint_scale": row.get("spring_diag_norm_constraint_scale"),
                "float64_constraint_scale": row.get("spring_replay_float64_norm_constraint_scale_corrected"),
            })
if selected_replay:
    with (OUT / "selected_float32_float64_replay.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected_replay[0].keys())
        writer.writeheader(); writer.writerows(selected_replay)


def series(rows, key):
    return [number(row.get(key)) for row in rows]


def plot(name, panels, logy=None):
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.0, 2.7 * len(panels)), sharex=True)
    if len(panels) == 1:
        axes = [axes]
    styles = {"mu=0": ("-", "o"), "mu=0.99": ("--", "s")}
    for axis, (ylabel, keys) in zip(axes, panels):
        for run_label, rows in trajectories.items():
            epochs = series(rows, "epoch")
            for key, suffix in keys:
                ls, marker = styles[run_label]
                axis.plot(epochs, series(rows, key), ls=ls, marker=marker,
                          markevery=max(1, len(rows) // 12), ms=3,
                          label=run_label + suffix)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
        if logy and ylabel in logy:
            axis.set_yscale("log")
    axes[-1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)


if trajectories:
    plot("energy_variance", [("Energy (Ha)", [("energy", "")]),
                              ("Variance (Ha²)", [("variance", "")])])
    plot("history_nonhistory_norm", [("Direction norm", [
        ("spring_diag_history_norm", " history"),
        ("spring_diag_nonhistory_norm", " non-history")])], {"Direction norm"})
    plot("history_ratio", [("||mu h|| / ||c||", [
        ("spring_diag_mu_history_over_nonhistory", "")])])
    plot("raw_displacement", [("Norm", [
        ("spring_diag_raw_solution_norm", " raw"),
        ("spring_diag_displacement_norm", " displacement")])], {"Norm"})
    plot("constraint_scale", [("Constraint scale", [
        ("spring_diag_norm_constraint_scale", "")])])
    plot("low_eigen_rhs_fraction", [("Low-mode RHS fraction", [
        ("spring_spec_low_rhs_fraction", " float32")])])
    plot("float32_float64_replay_error", [("Relative direction error", [
        ("spring_replay_relative_direction_error", "")])], {"Relative direction error"})

print(f"Collected {len(trajectories)} trajectories into {OUT}")
for summary in summaries:
    print(summary)
