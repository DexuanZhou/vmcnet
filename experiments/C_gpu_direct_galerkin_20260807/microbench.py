#!/usr/bin/env python3
"""Isolate host-callback versus device Galerkin latency on one GPU."""

import json
import os
from pathlib import Path
import statistics
import time

import jax
import jax.numpy as jnp

from vmcnet.updates import wssr


OUT = Path(os.environ["MICROBENCH_OUT"])
OUT.mkdir(parents=True, exist_ok=True)
LAMBDA = 1e-3
RANK = 800
REPEATS = 50


def host_solve(action, sample_residual, feedback):
    shape = jax.ShapeDtypeStruct((action.shape[1],), action.dtype)
    return jax.pure_callback(
        wssr._mixed_precision_galerkin_solution_callback,
        shape,
        action,
        sample_residual,
        feedback,
        jnp.asarray(LAMBDA, dtype=action.dtype),
    )


def device_solve(action, sample_residual, feedback):
    return wssr._device_galerkin_solution(
        action,
        action.T @ sample_residual,
        feedback,
        LAMBDA,
    )


host_jit = jax.jit(host_solve)
device_jit = jax.jit(device_solve)
rows = []
for samples in (1000, 4096):
    key = jax.random.PRNGKey(samples)
    action = jax.random.normal(key, (samples, RANK), dtype=jnp.float32)
    residual = jax.random.normal(
        jax.random.fold_in(key, 1), (samples,), dtype=jnp.float32
    )
    feedback = jnp.zeros((RANK,), dtype=jnp.float32)
    host_value = host_jit(action, residual, feedback)
    device_value = device_jit(action, residual, feedback)
    jax.block_until_ready((host_value, device_value))
    relative_error = float(
        jnp.linalg.norm(host_value - device_value)
        / jnp.maximum(jnp.linalg.norm(host_value), 1e-12)
    )
    for backend, function in (
        ("host_fp64", host_jit),
        ("device_cholesky", device_jit),
    ):
        durations = []
        for _ in range(REPEATS):
            start = time.perf_counter()
            value = function(action, residual, feedback)
            jax.block_until_ready(value)
            durations.append(time.perf_counter() - start)
        rows.append(
            {
                "backend": backend,
                "samples": samples,
                "rank": RANK,
                "mean_ms": 1000.0 * statistics.mean(durations),
                "std_ms": 1000.0 * statistics.stdev(durations),
                "relative_coefficient_error_vs_host": relative_error,
            }
        )

(OUT / "microbench.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows, indent=2))
