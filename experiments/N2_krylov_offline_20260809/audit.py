#!/usr/bin/env python3
"""Fixed-state N2 Krylov audit; no MCMC advance and no parameter update."""

import csv
import json
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from vmcnet.mcmc import position_amplitude_core as pacore
from vmcnet.physics import core as physics_core
from vmcnet.train import runners
from vmcnet.updates import wssr
from vmcnet.utils import io


SOURCE = Path(
    "/scratch/dexuan1/runs/old/N2_R2016/e100000_followup/N2eq/"
    "N2eq_kfac_pre5000"
)
CHECKPOINT = "5000.npz"
OUTPUT = Path("/scratch/dexuan1/runs/N2_krylov_offline_20260809")
LAMBDA = 1.0e-3
KRYLOV_DIMS = (4, 8, 16)


def sync(value):
    return jax.block_until_ready(value)


def device_memory():
    stats = jax.devices()[0].memory_stats() or {}
    return {
        "bytes_in_use": int(stats.get("bytes_in_use", -1)),
        "peak_bytes_in_use": int(stats.get("peak_bytes_in_use", -1)),
    }


def append_orthonormal(basis, vector):
    """Twice-reorthogonalize one Krylov vector against an existing basis."""
    if basis:
        q = jnp.stack(basis, axis=1)
        vector = vector - q @ (q.T @ vector)
        vector = vector - q @ (q.T @ vector)
    return vector / jnp.maximum(jnp.linalg.norm(vector), 1.0e-30)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cfg = io.load_config_dict(str(SOURCE), "config.json")
    dtype = runners._get_dtype(cfg)
    ions, charges, nelec = runners._get_electron_ion_config_as_arrays(cfg, dtype)
    epoch, data, params, _optimizer_state, key = io.reload_vmc_state(
        str(SOURCE / "checkpoints"), CHECKPOINT
    )
    positions = pacore.get_position_from_data(data)
    if positions.shape[0] != 1000:
        raise ValueError(f"expected 1000 walkers, got {positions.shape[0]}")

    logpsi, _, _ = runners._get_and_init_model(
        cfg.model,
        ions,
        charges,
        nelec,
        positions,
        key,
        dtype=dtype,
        apply_pmap=False,
    )
    local_energy = runners._assemble_mol_local_energy_fn(
        ions,
        charges,
        cfg.problem.ei_softening,
        cfg.problem.ee_softening,
        logpsi,
    )
    energy_fn = physics_core.create_energy_and_statistics_fn(
        local_energy,
        1000,
        runners._get_clipping_fn(cfg.vmc),
        cfg.vmc.nan_safe,
    )
    energy, local_energies, _ = energy_fn(params, positions)
    # VMCNet stores the centered score operator as parameters x walkers.
    operator, _ = wssr.center_and_scale_score_matrix(logpsi, params, positions)
    rhs_sample = wssr.center_and_scale_energy_residuals(local_energies, energy)
    force = operator @ rhs_sample
    sync((operator, rhs_sample, force))

    def fisher_action(vector):
        return operator @ (operator.T @ vector)

    basis = []
    q = append_orthonormal([], force)
    rows = []
    start = time.perf_counter()
    for one_based_dim in range(1, max(KRYLOV_DIMS) + 1):
        basis.append(q)
        if one_based_dim in KRYLOV_DIMS:
            qmat = jnp.stack(basis, axis=1)
            sample_images = operator.T @ qmat
            projected_fisher = sample_images.T @ sample_images
            projected_rhs = qmat.T @ force
            coefficients = jnp.linalg.solve(
                projected_fisher + LAMBDA * jnp.eye(one_based_dim, dtype=dtype),
                projected_rhs,
            )
            update = qmat @ coefficients
            normal_residual = fisher_action(update) + LAMBDA * update - force
            sample_residual = operator.T @ update - rhs_sample
            sync((update, normal_residual, sample_residual))
            row = {
                "krylov_dim": one_based_dim,
                "elapsed_seconds": time.perf_counter() - start,
                "normal_residual_ratio": float(
                    sync(
                        jnp.linalg.norm(normal_residual)
                        / jnp.maximum(jnp.linalg.norm(force), 1.0e-30)
                    )
                ),
                "normal_residual_reduction": float(
                    sync(
                        1.0
                        - jnp.linalg.norm(normal_residual)
                        / jnp.maximum(jnp.linalg.norm(force), 1.0e-30)
                    )
                ),
                "sample_residual_ratio": float(
                    sync(
                        jnp.linalg.norm(sample_residual)
                        / jnp.maximum(jnp.linalg.norm(rhs_sample), 1.0e-30)
                    )
                ),
                "update_norm": float(sync(jnp.linalg.norm(update))),
            }
            row.update(device_memory())
            rows.append(row)
        if one_based_dim < max(KRYLOV_DIMS):
            q = append_orthonormal(basis, fisher_action(q))
            sync(q)

    with (OUTPUT / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "source_checkpoint": str(SOURCE / "checkpoints" / CHECKPOINT),
        "stored_epoch": int(epoch),
        "operator_shape_parameter_by_walker": list(map(int, operator.shape)),
        "lambda": LAMBDA,
        "mcmc_advanced": False,
        "parameter_update_applied": False,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "rows": rows,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
