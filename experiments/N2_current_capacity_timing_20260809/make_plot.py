#!/usr/bin/env python3
"""Plot the N2 current-only capacity audit."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("/scratch/dexuan1/runs/N2_current_capacity_timing_tailfix_20260809")
rows = json.loads((ROOT / "summary.json").read_text())
ranks = [row["rank"] for row in rows]

fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)

axes[0, 0].plot(ranks, [row["seconds_per_step"] for row in rows], "o-")
axes[0, 0].set_ylabel("seconds / step")
axes[0, 0].set_title("Steady throughput")

axes[0, 1].plot(
    ranks, [row["peak_device_memory_mib"] / 1024 for row in rows], "o-"
)
axes[0, 1].set_ylabel("peak GPU memory (GiB)")
axes[0, 1].set_title("Device memory")

axes[1, 0].plot(
    ranks,
    [row["spectral_tail_ratio_sigma_d_over_sigma_dplus1"] for row in rows],
    "o-",
)
axes[1, 0].axhline(1.0, color="0.5", linestyle="--", linewidth=1)
axes[1, 0].set_ylabel(r"$\sigma_d/\sigma_{d+1}$")
axes[1, 0].set_title("Spectral boundary")

axes[1, 1].plot(
    ranks, [100 * row["boundary_update_mass"] for row in rows], "o-"
)
axes[1, 1].axhline(10.0, color="0.5", linestyle="--", linewidth=1)
axes[1, 1].set_ylabel("weakest-10% update mass (%)")
axes[1, 1].set_title("Ritz boundary importance")

for ax in axes.flat:
    ax.set_xlabel("update rank d")
    ax.grid(alpha=0.25)

fig.savefig(ROOT / "capacity_audit.png", dpi=180)
