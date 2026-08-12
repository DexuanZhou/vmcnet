"""Optimizer-state prolongation for strict multilevel FermiNet transitions."""

from __future__ import annotations

from typing import Any

import jax
import jax.flatten_util
import jax.numpy as jnp

from vmcnet.models import multilevel as model_multilevel


def _prolongate_optax_state(
    optimizer_state: Any,
    fine_params: Any,
    coarse_model_config: Any,
    fine_model_config: Any,
    *,
    nspins: int,
) -> Any:
    """Prolong parameter-shaped Optax traces and preserve scalar counters."""
    if not isinstance(optimizer_state, tuple):
        raise TypeError("expected an Optax optimizer state tuple")
    converted = []
    for state_part in optimizer_state:
        if hasattr(state_part, "trace"):
            converted.append(
                state_part._replace(
                    trace=model_multilevel.prolongate_ferminet_tangent(
                        state_part.trace,
                        fine_params,
                        coarse_model_config,
                        fine_model_config,
                        nspins=nspins,
                    )
                )
            )
        else:
            # ScaleByScheduleState.count and EmptyState are intentionally copied.
            converted.append(state_part)
    return tuple(converted)


def prolongate_spring_optimizer_state(
    coarse_optimizer_state: Any,
    fine_params: Any,
    coarse_model_config: Any,
    fine_model_config: Any,
    *,
    nspins: int,
) -> Any:
    """Prolong SPRING's full-parameter history and retain its LR step count."""
    return _prolongate_optax_state(
        coarse_optimizer_state,
        fine_params,
        coarse_model_config,
        fine_model_config,
        nspins=nspins,
    )


def _embed_flat_rows(
    coarse_array: Any,
    fine_num_params: int,
    fine_rows: jnp.ndarray,
) -> jnp.ndarray:
    coarse_array = jnp.asarray(coarse_array)
    if coarse_array.ndim == 1:
        result = jnp.zeros((fine_num_params,), dtype=coarse_array.dtype)
    elif coarse_array.ndim == 2:
        result = jnp.zeros(
            (fine_num_params, coarse_array.shape[1]), dtype=coarse_array.dtype
        )
    else:
        raise ValueError("a parameter-axis WSSR state must be a vector or matrix")
    return result.at[fine_rows].set(coarse_array)


def _prolongate_wssr_core(
    core_state: Any,
    fine_num_params: int,
    fine_rows: jnp.ndarray,
) -> Any:
    replacements = {
        "sr_o": _embed_flat_rows(core_state.sr_o, fine_num_params, fine_rows)
    }
    if hasattr(core_state, "u"):
        replacements["u"] = _embed_flat_rows(
            core_state.u, fine_num_params, fine_rows
        )
    return core_state._replace(**replacements)


def prolongate_wssr_optimizer_state(
    coarse_optimizer_state: Any,
    coarse_params: Any,
    fine_params: Any,
    coarse_model_config: Any,
    fine_model_config: Any,
    *,
    nspins: int,
) -> Any:
    """Prolong all parameter-axis memories in an integrated WSSR state."""
    if not hasattr(coarse_optimizer_state, "core_state"):
        raise TypeError("optimizer state is not an integrated WSSR state")
    coarse_flat, _ = jax.flatten_util.ravel_pytree(coarse_params)
    fine_flat, _ = jax.flatten_util.ravel_pytree(fine_params)
    fine_rows = model_multilevel.ferminet_flat_prolongation_indices(
        coarse_params,
        fine_params,
        coarse_model_config,
        fine_model_config,
        nspins=nspins,
    )
    if fine_rows.shape[0] != coarse_flat.shape[0]:
        raise ValueError("flat prolongation map has the wrong source size")

    replacements = {
        "core_state": _prolongate_wssr_core(
            coarse_optimizer_state.core_state,
            fine_flat.shape[0],
            fine_rows,
        ),
        "optax_state": _prolongate_optax_state(
            coarse_optimizer_state.optax_state,
            fine_params,
            coarse_model_config,
            fine_model_config,
            nspins=nspins,
        ),
    }
    for field in coarse_optimizer_state._fields:
        if field in replacements or field in ("core_state", "optax_state"):
            continue
        value = getattr(coarse_optimizer_state, field)
        if field == "previous_theta":
            replacements[field] = fine_flat.astype(jnp.asarray(value).dtype)
            continue
        value_array = jnp.asarray(value)
        if value_array.ndim == 1 and value_array.shape[0] == coarse_flat.shape[0]:
            replacements[field] = _embed_flat_rows(
                value_array, fine_flat.shape[0], fine_rows
            )
    return coarse_optimizer_state._replace(**replacements)
