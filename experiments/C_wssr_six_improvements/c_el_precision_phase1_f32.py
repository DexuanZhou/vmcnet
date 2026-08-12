"""Phase 1: reproduce the seed-0 frozen trajectory entirely in float32."""
from pathlib import Path
import json
import time

import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
import numpy as np

from vmcnet.models import construct
from vmcnet.physics import core
from vmcnet.train import runners
from vmcnet.utils import io

SOURCE = Path("/scratch/dexuan1/runs/C_wssr6_stage3_E500/adaptive_complement_0.2/seed0")
OLD = Path("/scratch/dexuan1/runs/C_wssr6_stage3_frozen/adaptive_complement_0.2/seed0/eval")
OUT = Path("/scratch/dexuan1/vmcnet/experiments/C_wssr_six_improvements/results/energy_bias_audit/p2_c_fp64")
TAG = 2026072200
NSNAP, BATCH = 25, 128


def cast32(tree):
    return jax.tree_util.tree_map(
        lambda x: jnp.asarray(x, jnp.float32)
        if np.issubdtype(np.asarray(x).dtype, np.floating) else jnp.asarray(x),
        tree,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = io.load_config_dict(str(SOURCE), "config.json")
    epoch, _, params0, _, key = io.reload_vmc_state(
        str(SOURCE / "checkpoints"), "500.npz"
    )
    key = jax.random.fold_in(jnp.asarray(key), TAG)
    ions, charges, nelec = runners._get_electron_ion_config_as_arrays(
        cfg, dtype=jnp.float32
    )
    key, init_pos = core.initialize_molecular_pos(
        key, 4096, ions, charges, 6, dtype=jnp.float32
    )
    model = construct.get_model_from_config(
        cfg.model, nelec, ions, charges, dtype=jnp.float32
    )
    logpsi = construct.slog_psi_to_log_psi_apply(model.apply)
    params = cast32(params0)
    data = runners._make_initial_single_device_data(
        logpsi, cfg.eval, init_pos, params, dtype=jnp.float32
    )
    burn, walk = runners._get_mcmc_fns(cfg.eval, logpsi, apply_pmap=False)
    t0 = time.time()
    data, key = runners.mcmc.metropolis.burn_data(burn, 10000, params, data, key)
    burn_seconds = time.time() - t0
    pos, acc = [], []
    for _ in range(NSNAP):
        ar, data, key = walk(params, data, key)
        acc.append(float(ar))
        pos.append(np.asarray(runners.pacore.get_position_from_data(data)))
    pos = np.concatenate(pos)
    local_single = runners._assemble_mol_local_energy_fn(
        ions, charges, cfg.problem.ei_softening, cfg.problem.ee_softening, logpsi
    )
    local_batch = jax.jit(jax.vmap(local_single, in_axes=(None, 0)))
    e32 = np.concatenate([
        np.asarray(local_batch(params, jnp.asarray(pos[i:i+BATCH], jnp.float32)))
        for i in range(0, len(pos), BATCH)
    ]).astype(np.float32)
    # The file has one row per evaluation epoch and one column per walker.
    # np.loadtxt(max_rows=...) limits rows, not scalar entries, so flatten before
    # selecting the matched first NSNAP * nchains samples.
    original = (
        np.loadtxt(OLD / "local_energies.txt", max_rows=NSNAP)
        .reshape(-1)[: len(e32)]
        .astype(np.float32)
    )
    np.savez(
        OUT / "phase1_f32.npz", positions=pos, energy_float32=e32,
        original_saved_energy=original
    )
    report = {
        "phase": 1, "dtype": "float32", "checkpoint_internal_epoch": int(epoch),
        "checkpoint": str(SOURCE / "checkpoints/500.npz"), "eval_prng_tag": TAG,
        "sample_count": len(e32), "nchains": 4096, "snapshots": NSNAP,
        "burn_steps": 10000, "burn_seconds": burn_seconds,
        "mean_acceptance": float(np.mean(acc)),
        "energy_float32_mean": float(np.mean(e32, dtype=np.float64)),
        "original_mean": float(np.mean(original, dtype=np.float64)),
        "replay_max_abs_difference": float(np.max(np.abs(e32-original))),
        "replay_allclose": bool(np.allclose(e32, original, rtol=2e-5, atol=2e-5)),
    }
    (OUT / "phase1_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
