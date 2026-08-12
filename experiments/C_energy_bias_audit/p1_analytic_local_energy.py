#!/usr/bin/env python3
"""Pure analytic audit of VMCNet's molecular local-energy operator.

No network parameters or Monte Carlo state are involved.  The supplied wave
function is log|psi(r)| = -Z |r| for a one-electron ion of charge Z.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from vmcnet.physics import core, kinetic, potential


def make_points(npoints: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed random 3-D points, half deliberately close to the nucleus."""
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(npoints, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    nnear = npoints // 2
    # Log-uniform radii stress cancellation at the Coulomb cusp.
    near_r = 10.0 ** rng.uniform(-6.0, -1.000001, size=nnear)
    # Ordinary points span the bulk and tail of a hydrogenic distribution.
    far_r = 10.0 ** rng.uniform(-1.0, 1.0, size=npoints - nnear)
    radii = np.concatenate((near_r, far_r))
    return (directions * radii[:, None])[:, None, :], radii


def make_local_energy(z: float, dtype: jnp.dtype):
    ion_pos = jnp.zeros((1, 3), dtype=dtype)
    ion_charges = jnp.asarray([z], dtype=dtype)

    def log_psi_apply(_params, x):
        return -jnp.asarray(z, dtype=dtype) * jnp.linalg.norm(x[0])

    return core.combine_local_energy_terms(
        [
            kinetic.create_laplacian_kinetic_energy(log_psi_apply),
            potential.create_electron_ion_coulomb_potential(
                ion_pos, ion_charges
            ),
        ]
    )


def summarize(values: np.ndarray, expected: float, radii: np.ndarray) -> dict:
    abs_error = np.abs(values - expected)
    near = radii < 0.1
    return {
        "expected_ha": expected,
        "npoints": int(values.size),
        "n_near_nucleus_r_lt_0p1": int(np.count_nonzero(near)),
        "all": {
            "mean_signed_error_ha": float(np.mean(values - expected)),
            "mean_abs_error_ha": float(np.mean(abs_error)),
            "max_abs_error_ha": float(np.max(abs_error)),
            "rms_error_ha": float(np.sqrt(np.mean(abs_error**2))),
            "min_local_energy_ha": float(np.min(values)),
            "max_local_energy_ha": float(np.max(values)),
        },
        "near_nucleus_r_lt_0p1": {
            "mean_signed_error_ha": float(np.mean(values[near] - expected)),
            "mean_abs_error_ha": float(np.mean(abs_error[near])),
            "max_abs_error_ha": float(np.max(abs_error[near])),
            "rms_error_ha": float(np.sqrt(np.mean(abs_error[near] ** 2))),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--npoints", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    args = parser.parse_args()

    # fwdlap constructs identity tangents using the process-wide default dtype.
    # Consequently float32 and float64 must be audited in separate processes.
    # This is test infrastructure, not a change to the physics implementation.
    use_x64 = args.dtype == "float64"
    jax.config.update("jax_enable_x64", use_x64)
    dtype = jnp.float64 if use_x64 else jnp.float32
    positions64, radii = make_points(args.npoints, args.seed)

    output = {
        "jax_backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "jax_enable_x64": bool(jax.config.x64_enabled),
        "point_generation": {
            "seed": args.seed,
            "near_r_distribution": "log-uniform [1e-6, 0.1)",
            "far_r_distribution": "log-uniform [0.1, 10]",
        },
        "dtype": args.dtype,
        "results": {},
    }

    for z in (1.0, 2.0):
        expected = -(z**2) / 2.0
        local_energy_fn = make_local_energy(z, dtype)
        positions = jnp.asarray(positions64, dtype=dtype)
        values = jax.jit(
            jax.vmap(local_energy_fn, in_axes=(None, 0), out_axes=0)
        )(None, positions)
        values_np = np.asarray(values, dtype=np.float64)
        output["results"][f"Z={z:g}"] = summarize(values_np, expected, radii)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
