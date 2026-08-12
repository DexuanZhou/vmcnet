"""Matched float32/float64 local-energy audit for the C seed-0 frozen run.

This is analysis-only.  It recreates the first 25 saved frozen-evaluation
snapshots from the final training checkpoint and the recorded evaluation PRNG
tag, then evaluates exactly the same 102400 configurations in float32 and
float64 without changing parameters.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vmcnet.models import construct
from vmcnet.physics import core
from vmcnet.train import runners
from vmcnet.utils import io


SOURCE = Path(
    "/scratch/dexuan1/runs/C_wssr6_stage3_E500/"
    "adaptive_complement_0.2/seed0"
)
ORIGINAL_EVAL = Path(
    "/scratch/dexuan1/runs/C_wssr6_stage3_frozen/"
    "adaptive_complement_0.2/seed0/eval"
)
OUT = Path(
    "/scratch/dexuan1/vmcnet/experiments/C_wssr_six_improvements/"
    "results/energy_bias_audit/p2_c_fp64"
)
CHECKPOINT = SOURCE / "checkpoints/500.npz"
EVAL_PRNG_TAG = 2026072200
NSNAPSHOTS = 25
BATCH = 128


def tree_cast(tree, dtype):
    return jax.tree_util.tree_map(
        lambda x: jnp.asarray(x, dtype=dtype)
        if np.issubdtype(np.asarray(x).dtype, np.floating)
        else jnp.asarray(x),
        tree,
    )


def summarize(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "q0001": float(np.quantile(x, 0.0001)),
        "q001": float(np.quantile(x, 0.001)),
        "q01": float(np.quantile(x, 0.01)),
        "q50": float(np.quantile(x, 0.5)),
        "q99": float(np.quantile(x, 0.99)),
        "q999": float(np.quantile(x, 0.999)),
        "q9999": float(np.quantile(x, 0.9999)),
    }


def batched_apply(fn, params, positions, dtype):
    out = []
    for start in range(0, len(positions), BATCH):
        p = jnp.asarray(positions[start : start + BATCH], dtype=dtype)
        out.append(np.asarray(fn(params, p), dtype=np.float64))
    return np.concatenate(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = io.load_config_dict(str(SOURCE), "config.json")
    epoch, _, params0, _, key = io.reload_vmc_state(
        str(CHECKPOINT.parent), CHECKPOINT.name
    )
    if int(epoch) not in (499, 500):
        raise RuntimeError(f"unexpected checkpoint epoch {epoch}")

    # Exactly reproduce eval_seeded_launcher.py and _make_new_data_for_eval.
    key = jax.random.fold_in(jnp.asarray(key), EVAL_PRNG_TAG)
    ion32, charges32, nelec = runners._get_electron_ion_config_as_arrays(
        config, dtype=jnp.float32
    )
    key, init_pos = core.initialize_molecular_pos(
        key,
        4096,
        ion32,
        charges32,
        int(sum(config.problem.nelec)),
        dtype=jnp.float32,
    )
    model32 = construct.get_model_from_config(
        config.model, nelec, ion32, charges32, dtype=jnp.float32
    )
    logpsi32 = construct.slog_psi_to_log_psi_apply(model32.apply)
    params32 = tree_cast(params0, jnp.float32)
    data = runners._make_initial_single_device_data(
        logpsi32, config.eval, init_pos, params32, dtype=jnp.float32
    )
    burning_step, walker_fn = runners._get_mcmc_fns(
        config.eval, logpsi32, apply_pmap=False
    )

    t0 = time.time()
    data, key = runners.mcmc.metropolis.burn_data(
        burning_step, 10000, params32, data, key
    )
    burn_seconds = time.time() - t0

    snapshots = []
    accept = []
    for _ in range(NSNAPSHOTS):
        ar, data, key = walker_fn(params32, data, key)
        accept.append(float(ar))
        snapshots.append(np.asarray(runners.pacore.get_position_from_data(data)))
    positions = np.concatenate(snapshots, axis=0)
    if positions.shape[0] < 100000:
        raise RuntimeError(f"only {positions.shape[0]} positions")

    ion64 = jnp.asarray(config.problem.ion_pos, dtype=jnp.float64)
    charges64 = jnp.asarray(config.problem.ion_charges, dtype=jnp.float64)
    model64 = construct.get_model_from_config(
        config.model, nelec, ion64, charges64, dtype=jnp.float64
    )
    logpsi64 = construct.slog_psi_to_log_psi_apply(model64.apply)
    params64 = tree_cast(params0, jnp.float64)

    el32_single = runners._assemble_mol_local_energy_fn(
        ion32, charges32, config.problem.ei_softening,
        config.problem.ee_softening, logpsi32
    )
    el64_single = runners._assemble_mol_local_energy_fn(
        ion64, charges64, config.problem.ei_softening,
        config.problem.ee_softening, logpsi64
    )
    el32 = jax.jit(jax.vmap(el32_single, in_axes=(None, 0)))
    el64 = jax.jit(jax.vmap(el64_single, in_axes=(None, 0)))
    lp32 = jax.jit(jax.vmap(logpsi32, in_axes=(None, 0)))
    lp64 = jax.jit(jax.vmap(logpsi64, in_axes=(None, 0)))

    t1 = time.time()
    e32 = batched_apply(el32, params32, positions, jnp.float32)
    e64 = batched_apply(el64, params64, positions, jnp.float64)
    logpsi = batched_apply(lp64, params64, positions, jnp.float64)
    eval_seconds = time.time() - t1
    de = e64 - e32

    original = np.loadtxt(ORIGINAL_EVAL / "local_energies.txt", max_rows=len(e32))
    replay_delta = e32 - original
    finite = np.isfinite(e32) & np.isfinite(e64) & np.isfinite(logpsi)
    if not np.all(finite):
        raise RuntimeError(f"{np.count_nonzero(~finite)} nonfinite matched samples")

    abs50_32 = np.abs(e32) <= 50.0
    abs50_64 = np.abs(e64) <= 50.0
    common50 = abs50_32 & abs50_64

    # Equal-count log|psi| bins expose near-node precision error without relying
    # on visually sparse scatter points.
    edges = np.quantile(logpsi, np.linspace(0, 1, 21))
    bin_rows = []
    for i in range(20):
        take = (logpsi >= edges[i]) & (
            logpsi <= edges[i + 1] if i == 19 else logpsi < edges[i + 1]
        )
        bin_rows.append(
            [
                i,
                int(np.sum(take)),
                float(edges[i]),
                float(edges[i + 1]),
                float(np.mean(logpsi[take])),
                float(np.mean(de[take])),
                float(np.std(de[take], ddof=1)),
                float(np.mean(np.abs(de[take]))),
                float(np.max(np.abs(de[take]))),
            ]
        )
    np.savetxt(
        OUT / "dE_vs_logpsi_bins.csv",
        np.asarray(bin_rows),
        delimiter=",",
        header=(
            "bin,count,logpsi_left,logpsi_right,logpsi_mean,dE_mean,dE_std,"
            "abs_dE_mean,abs_dE_max"
        ),
        comments="",
    )
    np.savez_compressed(
        OUT / "matched_samples.npz",
        energy_float32=e32,
        energy_float64=e64,
        delta_energy=de,
        logabspsi_float64=logpsi,
        original_saved_energy=original,
    )

    corr = float(np.corrcoef(logpsi, np.abs(de))[0, 1])
    summary = {
        "source_checkpoint": str(CHECKPOINT),
        "checkpoint_internal_epoch": int(epoch),
        "original_eval_dir": str(ORIGINAL_EVAL),
        "eval_prng_tag": EVAL_PRNG_TAG,
        "nchains": 4096,
        "snapshots": NSNAPSHOTS,
        "mcmc_steps_between_snapshots": 10,
        "sample_count": int(len(e32)),
        "burn_steps": 10000,
        "burn_seconds": burn_seconds,
        "evaluation_seconds": eval_seconds,
        "mean_acceptance": float(np.mean(accept)),
        "float32_energy": summarize(e32),
        "float64_energy": summarize(e64),
        "delta_energy_float64_minus_float32": summarize(de),
        "mean_shift_Ha": float(np.mean(e64) - np.mean(e32)),
        "variance_float32": float(np.var(e32, ddof=1)),
        "variance_float64": float(np.var(e64, ddof=1)),
        "variance_difference": float(np.var(e64, ddof=1)-np.var(e32, ddof=1)),
        "original_saved_replay_delta": summarize(replay_delta),
        "original_saved_replay_allclose": bool(
            np.allclose(e32, original, rtol=2e-5, atol=2e-5)
        ),
        "common_abs_energy_le_50_count": int(np.sum(common50)),
        "common_abs_energy_le_50_fraction": float(np.mean(common50)),
        "trimmed_float32_mean": float(np.mean(e32[common50])),
        "trimmed_float64_mean": float(np.mean(e64[common50])),
        "trimmed_mean_shift_Ha": float(
            np.mean(e64[common50])-np.mean(e32[common50])
        ),
        "corr_logpsi_abs_delta_energy": corr,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    rng = np.random.default_rng(20260723)
    idx = rng.choice(len(de), min(20000, len(de)), replace=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(logpsi[idx], de[idx] * 1000, s=3, alpha=0.18, rasterized=True)
    br = np.asarray(bin_rows)
    ax.plot(br[:, 4], br[:, 5] * 1000, "o-", color="black", lw=1.5,
            label="equal-count-bin mean")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel(r"$\\log|\\psi|$")
    ax.set_ylabel(r"$E_L^{64}-E_L^{32}$ (mHa)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "dE_vs_logpsi.svg")
    fig.savefig(OUT / "dE_vs_logpsi.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    lim = np.quantile(np.abs(de), 0.999)
    ax.hist(de[np.abs(de) <= lim] * 1000, bins=160, histtype="step", lw=1.2)
    ax.set_yscale("log")
    ax.set_xlabel(r"$E_L^{64}-E_L^{32}$ (mHa)")
    ax.set_ylabel("count (log scale)")
    ax.set_title("Central 99.9% of matched local-energy differences")
    fig.tight_layout()
    fig.savefig(OUT / "dE_histogram.svg")
    fig.savefig(OUT / "dE_histogram.png", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
