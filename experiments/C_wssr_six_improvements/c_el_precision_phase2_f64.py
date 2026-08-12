"""Phase 2: evaluate saved fixed positions with a wholly float64 model/Hamiltonian."""
from pathlib import Path
import json
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from vmcnet.models import construct
from vmcnet.train import runners
from vmcnet.utils import io

SOURCE = Path("/scratch/dexuan1/runs/C_wssr6_stage3_E500/adaptive_complement_0.2/seed0")
OUT = Path("/scratch/dexuan1/vmcnet/experiments/C_wssr_six_improvements/results/energy_bias_audit/p2_c_fp64")
BATCH = 128


def cast64(tree):
    return jax.tree_util.tree_map(
        lambda x: jnp.asarray(x, jnp.float64)
        if np.issubdtype(np.asarray(x).dtype, np.floating) else jnp.asarray(x),
        tree,
    )


def apply_batches(fn, params, pos):
    return np.concatenate([
        np.asarray(fn(params, jnp.asarray(pos[i:i+BATCH], jnp.float64)))
        for i in range(0, len(pos), BATCH)
    ]).astype(np.float64)


def main():
    inp = np.load(OUT / "phase1_f32.npz")
    pos = inp["positions"]
    cfg = io.load_config_dict(str(SOURCE), "config.json")
    epoch, _, params0, _, _ = io.reload_vmc_state(
        str(SOURCE / "checkpoints"), "500.npz"
    )
    ions = jnp.asarray(cfg.problem.ion_pos, jnp.float64)
    charges = jnp.asarray(cfg.problem.ion_charges, jnp.float64)
    nelec = jnp.asarray(cfg.problem.nelec)
    model = construct.get_model_from_config(
        cfg.model, nelec, ions, charges, dtype=jnp.float64
    )
    logpsi = construct.slog_psi_to_log_psi_apply(model.apply)
    params = cast64(params0)
    local_single = runners._assemble_mol_local_energy_fn(
        ions, charges, cfg.problem.ei_softening, cfg.problem.ee_softening, logpsi
    )
    local_batch = jax.jit(jax.vmap(local_single, in_axes=(None, 0)))
    logpsi_batch = jax.jit(jax.vmap(logpsi, in_axes=(None, 0)))
    t0 = time.time()
    e64 = apply_batches(local_batch, params, pos)
    lp64 = apply_batches(logpsi_batch, params, pos)
    seconds = time.time()-t0
    np.savez(OUT/"phase2_f64.npz", energy_float64=e64, logabspsi_float64=lp64)
    report = {
        "phase": 2, "dtype": "float64", "checkpoint_internal_epoch": int(epoch),
        "sample_count": len(e64), "energy_float64_mean": float(np.mean(e64)),
        "energy_float64_variance": float(np.var(e64, ddof=1)),
        "logabspsi_min": float(np.min(lp64)), "logabspsi_max": float(np.max(lp64)),
        "evaluation_seconds": seconds, "all_finite": bool(
            np.all(np.isfinite(e64)) and np.all(np.isfinite(lp64))
        ),
    }
    (OUT/"phase2_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
