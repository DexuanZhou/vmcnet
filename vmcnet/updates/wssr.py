"""WSSR low-rank stochastic reconfiguration helpers."""
from typing import NamedTuple, Optional, Tuple

import chex
import jax
import jax.flatten_util
import jax.numpy as jnp
from ml_collections import ConfigDict
import optax

from vmcnet.utils.pytree_helpers import tree_reduce_l1
from vmcnet.utils.typing import (
    Array,
    D,
    GetPositionFromData,
    LearningRateSchedule,
    ModelApply,
    P,
    UpdateDataFn,
)

from .update_param_fns import UpdateParamFn, update_metrics_with_noclip


class WSSRCoreState(NamedTuple):
    """Pure WSSR history state.

    This core state intentionally does not include Optax state. Optimizer integration
    state belongs at the dispatch layer.
    """

    sr_o: Array
    ek: Array
    sr_rank0: Array
    sr_rank: Array


class WSSRWarmSVDCoreState(NamedTuple):
    """Warm-start WSSR SVD state with previous left singular subspace."""

    sr_o: Array
    ek: Array
    sr_rank0: Array
    sr_rank: Array
    u: Array
    has_u: Array


class WSSRSVDResult(NamedTuple):
    """Result of one pure WSSR SVD core update."""

    grad_like_update: Array
    state: WSSRCoreState
    active_rank: Array


class AvgMinSRSVDHistoryResult(NamedTuple):
    """Result of one averaged-MinSR SVD-history core update."""

    grad_like_update: Array
    state: WSSRCoreState
    active_rank: Array
    dual_matrix_cond: Array
    update_norm: Array


class WSSROptimizerState(NamedTuple):
    """Integrated WSSR optimizer state."""

    core_state: WSSRCoreState
    optax_state: optax.OptState


def recover_u_from_sr_o(
    sr_o: Array,
    sr_rank0: Array,
    rank_capacity: int,
    eps: chex.Numeric = 1e-12,
) -> Array:
    """Recover a normalized warm left basis from stored WSSR score history."""
    if rank_capacity < 0:
        raise ValueError("rank_capacity must be nonnegative")

    recovered = jnp.zeros((sr_o.shape[0], rank_capacity), dtype=sr_o.dtype)
    history_width = min(sr_o.shape[1], rank_capacity)
    sr_o_active = sr_o[:, :history_width]
    column_norms = jnp.linalg.norm(sr_o_active, axis=0, keepdims=True)
    eps_array = jnp.asarray(eps, dtype=sr_o.dtype)
    safe_norms = jnp.maximum(column_norms, eps_array)
    active_columns = jnp.arange(history_width) < sr_rank0
    nonzero_columns = column_norms[0] > eps_array
    column_mask = active_columns & nonzero_columns
    u_active = sr_o_active / safe_norms
    u_active = jnp.where(column_mask[None, :], u_active, jnp.zeros_like(u_active))
    return recovered.at[:, :history_width].set(u_active)


def recover_right_basis_from_sr_o_projection(
    o_aug: Array,
    sr_o: Array,
    sr_rank0: Array,
    rank_capacity: int,
    eps: chex.Numeric = 1e-12,
) -> Array:
    """Project recovered WSSR history directions into right space without forming U."""
    if rank_capacity < 0:
        raise ValueError("rank_capacity must be nonnegative")

    recovered_projection = jnp.zeros(
        (o_aug.shape[1], rank_capacity), dtype=o_aug.dtype
    )
    history_width = min(sr_o.shape[1], rank_capacity)
    sr_o_active = sr_o[:, :history_width]
    column_norms = jnp.linalg.norm(sr_o_active, axis=0)
    eps_array = jnp.asarray(eps, dtype=sr_o.dtype)
    active_columns = jnp.arange(history_width) < sr_rank0
    nonzero_columns = column_norms > eps_array
    inv_norms = jnp.where(
        active_columns & nonzero_columns,
        1.0 / jnp.maximum(column_norms, eps_array),
        jnp.zeros_like(column_norms),
    )
    projection = (o_aug.T @ sr_o_active) * inv_norms[None, :]
    return recovered_projection.at[:, :history_width].set(projection)


def _slice_or_zero_pad_columns(matrix: Array, rank_capacity: int) -> Array:
    padded = jnp.zeros((matrix.shape[0], rank_capacity), dtype=matrix.dtype)
    width = min(matrix.shape[1], rank_capacity)
    return padded.at[:, :width].set(matrix[:, :width])


def _select_warm_left_basis(
    state: WSSRWarmSVDCoreState,
    rank_capacity: int,
    eps: chex.Numeric = 1e-12,
) -> Tuple[Array, Array]:
    """Choose stored warm ``u`` when available, otherwise derive it from ``sr_o``."""
    recovered_u = recover_u_from_sr_o(
        state.sr_o,
        state.sr_rank0,
        rank_capacity,
        eps=eps,
    )
    if state.u.shape[1] > 0:
        stored_u = _slice_or_zero_pad_columns(state.u, rank_capacity)
        warm_u = jax.lax.cond(
            state.has_u,
            lambda _: stored_u,
            lambda _: recovered_u,
            operand=None,
        )
        has_warm_u = state.has_u | (state.sr_rank0 > 0)
    else:
        warm_u = recovered_u
        has_warm_u = state.sr_rank0 > 0
    return warm_u, has_warm_u


def initialize_wssr_core_state(
    num_params: int,
    sr_rank: int,
    sr_rank_max: int,
    dtype=jnp.float32,
) -> WSSRCoreState:
    """Initialize the pure WSSR low-rank history state."""
    if sr_rank_max < 0:
        raise ValueError("sr_rank_max must be nonnegative")
    if sr_rank < 0:
        raise ValueError("sr_rank must be nonnegative")

    sr_rank = min(sr_rank, sr_rank_max)
    return WSSRCoreState(
        sr_o=jnp.zeros((num_params, sr_rank_max), dtype=dtype),
        ek=jnp.zeros((sr_rank_max,), dtype=dtype),
        sr_rank0=jnp.array(0),
        sr_rank=jnp.array(sr_rank),
    )


def initialize_wssr_warm_svd_core_state(
    num_params: int,
    sr_rank: int,
    sr_rank_max: int,
    dtype=jnp.float32,
    store_warm_u: bool = True,
) -> WSSRWarmSVDCoreState:
    """Initialize WSSR history plus fixed-shape warm-SVD subspace."""
    core_state = initialize_wssr_core_state(num_params, sr_rank, sr_rank_max, dtype)
    u_width = sr_rank_max if store_warm_u else 0
    return WSSRWarmSVDCoreState(
        sr_o=core_state.sr_o,
        ek=core_state.ek,
        sr_rank0=core_state.sr_rank0,
        sr_rank=core_state.sr_rank,
        u=jnp.zeros((num_params, u_width), dtype=dtype),
        has_u=jnp.array(False),
    )


def center_and_scale_score_matrix(
    log_psi_apply: ModelApply[P],
    params: P,
    positions: Array,
) -> Tuple[Array, P]:
    """Build centered score matrix with Julia WSSR scaling.

    Returns:
        A pair ``(o_cur, unravel_fn)`` where ``o_cur`` has shape
        ``num_params x num_samples`` and is divided by ``sqrt(num_samples)``.
    """

    _, unravel_fn = jax.flatten_util.ravel_pytree(params)

    grad_samples = jax.vmap(
        jax.grad(log_psi_apply, argnums=0),
        in_axes=(None, 0),
    )(params, positions)
    flat_score_leaves = [
        jnp.reshape(leaf, (positions.shape[0], -1))
        for leaf in jax.tree_util.tree_leaves(grad_samples)
    ]
    score_samples = jnp.concatenate(flat_score_leaves, axis=1)
    score_samples = score_samples - jnp.mean(score_samples, axis=0, keepdims=True)
    scale = jnp.sqrt(positions.shape[0])
    return score_samples.T / scale, unravel_fn


def score_matvec_current(
    log_psi_apply: ModelApply[P],
    params: P,
    positions: Array,
    sample_weights: Array,
) -> Array:
    """Compute ``O_cur @ sample_weights`` without materializing ``O_cur``.

    ``O_cur`` is the centered and ``1 / sqrt(num_samples)``-scaled score
    matrix returned by :func:`center_and_scale_score_matrix`.
    """

    def log_psi_samples(params):
        return jax.vmap(log_psi_apply, in_axes=(None, 0))(params, positions)

    centered_weights = sample_weights - jnp.mean(sample_weights)
    scaled_weights = centered_weights / jnp.sqrt(positions.shape[0])
    _, pullback = jax.vjp(log_psi_samples, params)
    grad_tree = pullback(scaled_weights)[0]
    flat_grad, _ = jax.flatten_util.ravel_pytree(grad_tree)
    return flat_grad


def score_matmat_current(
    log_psi_apply: ModelApply[P],
    params: P,
    positions: Array,
    sample_weight_matrix: Array,
) -> Array:
    """Compute ``O_cur @ sample_weight_matrix`` without materializing ``O_cur``.

    ``O_cur`` is the centered and ``1 / sqrt(num_samples)``-scaled score
    matrix returned by :func:`center_and_scale_score_matrix`.
    """

    def log_psi_samples(params):
        return jax.vmap(log_psi_apply, in_axes=(None, 0))(params, positions)

    centered_weights = sample_weight_matrix - jnp.mean(
        sample_weight_matrix, axis=0, keepdims=True
    )
    scaled_weights = centered_weights / jnp.sqrt(positions.shape[0])
    _, pullback = jax.vjp(log_psi_samples, params)

    def pullback_flat(sample_weights):
        grad_tree = pullback(sample_weights)[0]
        flat_grad, _ = jax.flatten_util.ravel_pytree(grad_tree)
        return flat_grad

    return jax.vmap(pullback_flat, in_axes=1, out_axes=1)(scaled_weights)


def score_rmatvec_current(
    log_psi_apply: ModelApply[P],
    params: P,
    positions: Array,
    param_vector: Array,
) -> Array:
    """Compute ``O_cur.T @ param_vector`` without materializing ``O_cur``.

    ``O_cur`` is the centered and ``1 / sqrt(num_samples)``-scaled score
    matrix returned by :func:`center_and_scale_score_matrix`.
    """

    _, unravel_fn = jax.flatten_util.ravel_pytree(params)
    param_tangent = unravel_fn(param_vector)

    def log_psi_samples(params):
        return jax.vmap(log_psi_apply, in_axes=(None, 0))(params, positions)

    _, jvp_values = jax.jvp(log_psi_samples, (params,), (param_tangent,))
    centered_jvp_values = jvp_values - jnp.mean(jvp_values)
    return centered_jvp_values / jnp.sqrt(positions.shape[0])


def score_rmatmat_current(
    log_psi_apply: ModelApply[P],
    params: P,
    positions: Array,
    param_matrix: Array,
) -> Array:
    """Compute ``O_cur.T @ param_matrix`` without materializing ``O_cur``.

    ``O_cur`` is the centered and ``1 / sqrt(num_samples)``-scaled score
    matrix returned by :func:`center_and_scale_score_matrix`.
    """

    _, unravel_fn = jax.flatten_util.ravel_pytree(params)

    def log_psi_samples(params):
        return jax.vmap(log_psi_apply, in_axes=(None, 0))(params, positions)

    _, linear_fn = jax.linearize(log_psi_samples, params)

    def jvp_values(param_vector):
        param_tangent = unravel_fn(param_vector)
        return linear_fn(param_tangent)

    jvp_matrix = jax.vmap(jvp_values, in_axes=1, out_axes=1)(param_matrix)
    centered_jvp_matrix = jvp_matrix - jnp.mean(jvp_matrix, axis=0, keepdims=True)
    return centered_jvp_matrix / jnp.sqrt(positions.shape[0])


def wssr_augmented_matvec(
    log_psi_apply: ModelApply[P],
    params: P,
    positions: Array,
    state: WSSRCoreState,
    eta: chex.Numeric,
    aug_vector: Array,
) -> Array:
    """Compute ``O_aug @ aug_vector`` without materializing current scores.

    The augmented matrix matches :func:`augment_wssr_system`. History columns
    remain explicit through ``state.sr_o``; the current score block is applied
    through :func:`score_matvec_current`.
    """
    history_width = state.sr_o.shape[1]
    v_hist = aug_vector[:history_width]
    v_cur = aug_vector[history_width:]

    history_mask = (jnp.arange(history_width) < state.sr_rank0).astype(v_hist.dtype)
    has_history = state.sr_rank0 > 0
    sqrt_eta = jnp.sqrt(eta)
    current_scale = jnp.where(has_history, jnp.sqrt(1.0 - eta), 1.0).astype(
        v_cur.dtype
    )

    history_term = sqrt_eta * (state.sr_o @ (history_mask * v_hist))
    current_term = current_scale * score_matvec_current(
        log_psi_apply, params, positions, v_cur
    )
    return history_term + current_term


def wssr_augmented_rmatvec(
    log_psi_apply: ModelApply[P],
    params: P,
    positions: Array,
    state: WSSRCoreState,
    eta: chex.Numeric,
    param_vector: Array,
) -> Array:
    """Compute ``O_aug.T @ param_vector`` without materializing current scores."""
    history_width = state.sr_o.shape[1]
    history_mask = (jnp.arange(history_width) < state.sr_rank0).astype(
        param_vector.dtype
    )
    has_history = state.sr_rank0 > 0
    sqrt_eta = jnp.sqrt(eta)
    current_scale = jnp.where(has_history, jnp.sqrt(1.0 - eta), 1.0).astype(
        param_vector.dtype
    )

    history_term = sqrt_eta * history_mask * (state.sr_o.T @ param_vector)
    current_term = current_scale * score_rmatvec_current(
        log_psi_apply, params, positions, param_vector
    )
    return jnp.concatenate([history_term, current_term], axis=0)


def wssr_augmented_matmat(
    log_psi_apply: ModelApply[P],
    params: P,
    positions: Array,
    state: WSSRCoreState,
    eta: chex.Numeric,
    aug_matrix: Array,
) -> Array:
    """Compute ``O_aug @ aug_matrix`` without materializing current scores."""
    history_width = state.sr_o.shape[1]
    v_hist = aug_matrix[:history_width, :]
    v_cur = aug_matrix[history_width:, :]

    history_mask = (jnp.arange(history_width) < state.sr_rank0).astype(v_hist.dtype)
    has_history = state.sr_rank0 > 0
    sqrt_eta = jnp.sqrt(eta)
    current_scale = jnp.where(has_history, jnp.sqrt(1.0 - eta), 1.0).astype(
        v_cur.dtype
    )

    history_term = sqrt_eta * (state.sr_o @ (history_mask[:, None] * v_hist))
    current_term = current_scale * score_matmat_current(
        log_psi_apply, params, positions, v_cur
    )
    return history_term + current_term


def wssr_augmented_rmatmat(
    log_psi_apply: ModelApply[P],
    params: P,
    positions: Array,
    state: WSSRCoreState,
    eta: chex.Numeric,
    param_matrix: Array,
) -> Array:
    """Compute ``O_aug.T @ param_matrix`` without materializing current scores."""
    history_width = state.sr_o.shape[1]
    history_mask = (jnp.arange(history_width) < state.sr_rank0).astype(
        param_matrix.dtype
    )
    has_history = state.sr_rank0 > 0
    sqrt_eta = jnp.sqrt(eta)
    current_scale = jnp.where(has_history, jnp.sqrt(1.0 - eta), 1.0).astype(
        param_matrix.dtype
    )

    history_term = sqrt_eta * history_mask[:, None] * (state.sr_o.T @ param_matrix)
    current_term = current_scale * score_rmatmat_current(
        log_psi_apply, params, positions, param_matrix
    )
    return jnp.concatenate([history_term, current_term], axis=0)


def center_and_scale_energy_residuals(local_energies: Array, energy: Array) -> Array:
    """Center local-energy residuals and apply Julia WSSR scaling."""
    residuals = local_energies - energy
    return residuals / jnp.sqrt(local_energies.shape[0])


def augment_wssr_residuals(
    e_cur: Array,
    state: WSSRCoreState,
    eta: chex.Numeric,
) -> Array:
    """Create the history-augmented WSSR residual vector only.

    This matches the residual side of :func:`augment_wssr_system` without
    requiring an explicit current score matrix.
    """
    history_width = state.sr_o.shape[1]
    history_mask = (jnp.arange(history_width) < state.sr_rank0).astype(e_cur.dtype)
    has_history = state.sr_rank0 > 0
    sqrt_eta = jnp.sqrt(eta)
    current_scale = jnp.where(has_history, jnp.sqrt(1.0 - eta), 1.0).astype(
        e_cur.dtype
    )
    e_hist = sqrt_eta * state.ek * history_mask
    e_scaled = current_scale * e_cur
    return jnp.concatenate([e_hist, e_scaled], axis=0)


def augment_wssr_system(
    o_cur: Array,
    e_cur: Array,
    state: WSSRCoreState,
    eta: chex.Numeric,
) -> Tuple[Array, Array]:
    """Create history-augmented WSSR score matrix and residual vector.

    The output has fixed width ``sr_rank_max + num_samples``. Inactive history
    columns are zeroed so that the WSSR equations match the dynamic active-history
    augmentation while remaining JIT-safe.
    """
    history_width = state.sr_o.shape[1]
    history_mask = (jnp.arange(history_width) < state.sr_rank0).astype(o_cur.dtype)
    has_history = state.sr_rank0 > 0
    sqrt_eta = jnp.sqrt(eta)
    current_scale = jnp.where(has_history, jnp.sqrt(1.0 - eta), 1.0).astype(
        o_cur.dtype
    )
    o_hist = sqrt_eta * state.sr_o * history_mask
    e_hist = sqrt_eta * state.ek * history_mask
    o_scaled = current_scale * o_cur
    e_scaled = current_scale * e_cur
    return jnp.concatenate([o_hist, o_scaled], axis=1), jnp.concatenate(
        [e_hist, e_scaled], axis=0
    )


def constrain_norm(
    update: Array,
    norm_constraint: chex.Numeric,
    eps: chex.Numeric = 1e-12,
) -> Array:
    """Apply the WSSR Euclidean norm constraint with zero-norm safety."""
    update_norm = jnp.linalg.norm(update)
    safe_scale = jnp.where(
        update_norm > eps,
        jnp.minimum(1.0, jnp.sqrt(norm_constraint) / update_norm),
        1.0,
    )
    return update * safe_scale


def tree_l2_norm(tree) -> Array:
    """Compute the Euclidean norm of all leaves in a PyTree."""
    squared_norms = [
        jnp.sum(jnp.square(leaf)) for leaf in jax.tree_util.tree_leaves(tree)
    ]
    return jnp.sqrt(sum(squared_norms))


def constrain_update_tree_norm(
    updates: P,
    norm_constraint: chex.Numeric,
    eps: chex.Numeric = 1e-12,
) -> P:
    """Constrain the final Optax update PyTree to the WSSR norm bound."""
    update_norm = tree_l2_norm(updates)
    safe_scale = jnp.where(
        update_norm > eps,
        jnp.minimum(1.0, jnp.sqrt(norm_constraint) / update_norm),
        1.0,
    )
    return jax.tree_util.tree_map(lambda update: update * safe_scale, updates)


def _resolve_svd_working_rank(
    svd_working_rank: Optional[int],
    sr_rank_max: int,
) -> int:
    """Resolve the static warm-SVD compute width.

    ``sr_rank_max`` remains the history/checkpoint capacity. The resolved value is
    only the maximum width used by the warm-start SVD computation and rank growth.
    Nonpositive values are treated as unset so command-line integer overrides can
    use a default sentinel while preserving full-width behavior.
    """
    if svd_working_rank is None or svd_working_rank <= 0:
        return sr_rank_max
    return min(svd_working_rank, sr_rank_max)


def resolve_wssr_storage_rank(
    sr_rank: int,
    sr_rank_max: int,
    sr_storage_rank: Optional[int] = None,
) -> int:
    """Resolve the fixed WSSR history allocation width.

    ``sr_rank_max`` remains the logical maximum rank used by optimizer configs and
    diagnostics. ``sr_storage_rank`` optionally narrows the allocated history and
    warm-start buffers. Nonpositive values preserve the historical full-width
    allocation.
    """
    if sr_storage_rank is None or sr_storage_rank <= 0:
        storage_rank = sr_rank_max
    else:
        storage_rank = min(sr_storage_rank, sr_rank_max)

    if storage_rank < sr_rank:
        raise ValueError("sr_storage_rank must be at least sr_rank")
    return storage_rank


_WSSR_SPECTRAL_REGULARIZATIONS = ("hard_floor", "tikhonov")


def _validate_wssr_spectral_regularization(
    spectral_regularization: str,
    complement_weight: chex.Numeric,
) -> None:
    if spectral_regularization not in _WSSR_SPECTRAL_REGULARIZATIONS:
        raise ValueError(
            "spectral_regularization must be one of "
            f"{_WSSR_SPECTRAL_REGULARIZATIONS}"
        )
    if complement_weight < 0:
        raise ValueError("complement_weight must be nonnegative")


def _wssr_inverse_spectral_coefficients(
    safe_singular_values: Array,
    safe_leading_sv: Array,
    safe_damping: Array,
    spectral_regularization: str,
    complement_weight: chex.Numeric,
) -> Tuple[Array, Array]:
    """Return range and complement inverse coefficients for the WSSR update."""
    _validate_wssr_spectral_regularization(
        spectral_regularization, complement_weight
    )
    sigma_floor = jnp.square(safe_damping * jnp.abs(safe_leading_sv))
    inv_floor = 1.0 / sigma_floor
    if spectral_regularization == "hard_floor":
        inv_cap = jnp.square(1.0 / safe_singular_values)
    else:
        inv_cap = 1.0 / (jnp.square(safe_singular_values) + sigma_floor)
    inv_perp = jnp.asarray(complement_weight, dtype=safe_singular_values.dtype)
    inv_perp = inv_perp * inv_floor
    return inv_cap, inv_perp


def _update_metrics_with_wssr_rank_diagnostics(
    metrics,
    active_rank: Array,
    core_state: WSSRCoreState,
    optimizer_config: ConfigDict,
    svd_working_rank: Optional[int] = None,
):
    """Add WSSR rank/storage diagnostics to an update metrics dictionary."""
    resolved_working_rank = _resolve_svd_working_rank(
        svd_working_rank, optimizer_config.sr_rank_max
    )
    resolved_working_rank = min(resolved_working_rank, core_state.sr_o.shape[1])
    metric_dtype = core_state.sr_rank.dtype
    metrics.update(
        {
            "wssr_active_rank": jnp.asarray(active_rank, dtype=metric_dtype),
            "wssr_sr_rank": core_state.sr_rank,
            "wssr_sr_rank0": core_state.sr_rank0,
            "wssr_storage_width": jnp.asarray(
                core_state.sr_o.shape[1], dtype=metric_dtype
            ),
            "wssr_svd_working_rank": jnp.asarray(
                resolved_working_rank, dtype=metric_dtype
            ),
            "wssr_sr_rank_max": jnp.asarray(
                optimizer_config.sr_rank_max, dtype=metric_dtype
            ),
        }
    )
    return metrics


def _update_metrics_with_avg_minsr_diagnostics(
    metrics,
    result: AvgMinSRSVDHistoryResult,
    lambda_reg: chex.Numeric,
):
    """Add averaged-MinSR SVD-history diagnostics to metrics."""
    metric_dtype = result.state.sr_rank.dtype
    metrics.update(
        {
            "avg_minsr_active_rank": jnp.asarray(
                result.active_rank, dtype=metric_dtype
            ),
            "avg_minsr_sr_rank": result.state.sr_rank,
            "avg_minsr_sr_rank0": result.state.sr_rank0,
            "avg_minsr_lambda_reg": jnp.asarray(
                lambda_reg, dtype=result.update_norm.dtype
            ),
            "avg_minsr_dual_matrix_cond": result.dual_matrix_cond,
            "avg_minsr_update_norm": result.update_norm,
        }
    )
    return metrics


def _update_working_rank(active_rank: Array, sr_rank: Array, sr_rank_max: int, sr_scale):
    capped_sr_rank = jnp.minimum(
        sr_rank,
        jnp.asarray(sr_rank_max, dtype=sr_rank.dtype),
    )
    proposed_rank = jnp.minimum(
        jnp.ceil(capped_sr_rank * sr_scale).astype(sr_rank.dtype),
        jnp.asarray(sr_rank_max, dtype=sr_rank.dtype),
    )
    should_grow = (active_rank == capped_sr_rank) & (capped_sr_rank < sr_rank_max)
    return jnp.where(should_grow, proposed_rank, capped_sr_rank)


def _zero_history_like(state: WSSRCoreState) -> WSSRCoreState:
    return WSSRCoreState(
        sr_o=jnp.zeros_like(state.sr_o),
        ek=jnp.zeros_like(state.ek),
        sr_rank0=jnp.array(0),
        sr_rank=state.sr_rank,
    )


def _zero_warm_svd_history_like(state: WSSRWarmSVDCoreState) -> WSSRWarmSVDCoreState:
    return WSSRWarmSVDCoreState(
        sr_o=jnp.zeros_like(state.sr_o),
        ek=jnp.zeros_like(state.ek),
        sr_rank0=jnp.array(0),
        sr_rank=state.sr_rank,
        u=state.u,
        has_u=state.has_u,
    )


def _wssr_update_from_svd(
    o_aug: Array,
    e_aug: Array,
    state: WSSRCoreState,
    u: Array,
    singular_values: Array,
    vh: Array,
    damping: chex.Numeric,
    norm_constraint: chex.Numeric,
    sr_rank_max: int,
    sr_scale: chex.Numeric,
    constrain_update_norm: bool,
    rank_update_max: Optional[int] = None,
    spectral_regularization: str = "hard_floor",
    complement_weight: chex.Numeric = 1.0,
    eps: chex.Numeric = 1e-12,
) -> WSSRSVDResult:
    """Apply the shared post-SVD WSSR update formula."""
    if rank_update_max is None:
        rank_update_max = sr_rank_max
    rank_update_max = min(rank_update_max, sr_rank_max)

    if singular_values.shape[0] == 0:
        zero_state = _zero_history_like(state)
        return WSSRSVDResult(
            jnp.zeros(o_aug.shape[0], dtype=o_aug.dtype),
            zero_state,
            jnp.asarray(0, dtype=state.sr_rank.dtype),
        )

    leading_sv = singular_values[0]
    valid_leading = leading_sv > eps
    safe_leading_sv = jnp.where(valid_leading, leading_sv, 1.0)
    retained = (singular_values / safe_leading_sv > damping) & valid_leading
    active_rank = jnp.sum(retained.astype(state.sr_rank.dtype))
    has_active_rank = active_rank > 0
    valid_update = valid_leading & has_active_rank
    retained_float = retained.astype(o_aug.dtype)

    safe_singular_values = jnp.where(retained, singular_values, 1.0)
    safe_damping = jnp.maximum(jnp.abs(damping), eps)
    inv_cap, inv_perp = _wssr_inverse_spectral_coefficients(
        safe_singular_values,
        safe_leading_sv,
        safe_damping,
        spectral_regularization,
        complement_weight,
    )
    force = o_aug @ e_aug
    projected_force = u.T @ force
    projected_force = projected_force * (inv_cap - inv_perp)
    projected_force = projected_force * retained_float
    grad_like_update = u @ projected_force + inv_perp * force
    grad_like_update = jnp.where(valid_update, grad_like_update, 0.0)

    if constrain_update_norm:
        grad_like_update = constrain_norm(grad_like_update, norm_constraint, eps=eps)

    sr_o = jnp.zeros_like(state.sr_o)
    ek = jnp.zeros_like(state.ek)
    history_width = min(singular_values.shape[0], sr_rank_max)
    sr_o_values = (
        u[:, :history_width]
        * singular_values[:history_width]
        * retained_float[:history_width]
    )
    ek_values = (vh[:history_width, :] @ e_aug) * retained_float[:history_width]
    sr_o = sr_o.at[:, :history_width].set(sr_o_values)
    ek = ek.at[:history_width].set(ek_values)

    updated_sr_rank = _update_working_rank(
        active_rank, state.sr_rank, rank_update_max, sr_scale
    )
    capped_sr_rank = jnp.minimum(
        state.sr_rank,
        jnp.asarray(rank_update_max, dtype=state.sr_rank.dtype),
    )
    new_sr_rank = jnp.where(valid_update, updated_sr_rank, capped_sr_rank)
    new_state = WSSRCoreState(
        sr_o=jnp.where(valid_update, sr_o, jnp.zeros_like(sr_o)),
        ek=jnp.where(valid_update, ek, jnp.zeros_like(ek)),
        sr_rank0=jnp.where(
            valid_update, active_rank, jnp.asarray(0, dtype=state.sr_rank0.dtype)
        ),
        sr_rank=new_sr_rank,
    )
    return WSSRSVDResult(grad_like_update, new_state, active_rank)


def _wssr_update_from_svd_matfree(
    log_psi_apply: ModelApply[P],
    params: P,
    positions: Array,
    e_aug: Array,
    state: WSSRCoreState,
    eta: chex.Numeric,
    u: Array,
    singular_values: Array,
    vh: Array,
    damping: chex.Numeric,
    norm_constraint: chex.Numeric,
    sr_rank_max: int,
    sr_scale: chex.Numeric,
    constrain_update_norm: bool,
    rank_update_max: Optional[int] = None,
    spectral_regularization: str = "hard_floor",
    complement_weight: chex.Numeric = 1.0,
    eps: chex.Numeric = 1e-12,
) -> WSSRSVDResult:
    """Matrix-free analogue of :func:`_wssr_update_from_svd`."""
    if rank_update_max is None:
        rank_update_max = sr_rank_max
    rank_update_max = min(rank_update_max, sr_rank_max)

    flat_params, _ = jax.flatten_util.ravel_pytree(params)
    if singular_values.shape[0] == 0:
        zero_state = _zero_history_like(state)
        return WSSRSVDResult(
            jnp.zeros(flat_params.shape[0], dtype=flat_params.dtype),
            zero_state,
            jnp.asarray(0, dtype=state.sr_rank.dtype),
        )

    leading_sv = singular_values[0]
    valid_leading = leading_sv > eps
    safe_leading_sv = jnp.where(valid_leading, leading_sv, 1.0)
    retained = (singular_values / safe_leading_sv > damping) & valid_leading
    active_rank = jnp.sum(retained.astype(state.sr_rank.dtype))
    has_active_rank = active_rank > 0
    valid_update = valid_leading & has_active_rank
    retained_float = retained.astype(flat_params.dtype)

    safe_singular_values = jnp.where(retained, singular_values, 1.0)
    safe_damping = jnp.maximum(jnp.abs(damping), eps)
    inv_cap, inv_perp = _wssr_inverse_spectral_coefficients(
        safe_singular_values,
        safe_leading_sv,
        safe_damping,
        spectral_regularization,
        complement_weight,
    )
    force = wssr_augmented_matvec(
        log_psi_apply,
        params,
        positions,
        state,
        eta,
        e_aug,
    )
    projected_force = u.T @ force
    projected_force = projected_force * (inv_cap - inv_perp)
    projected_force = projected_force * retained_float
    grad_like_update = u @ projected_force + inv_perp * force
    grad_like_update = jnp.where(valid_update, grad_like_update, 0.0)

    if constrain_update_norm:
        grad_like_update = constrain_norm(grad_like_update, norm_constraint, eps=eps)

    sr_o = jnp.zeros_like(state.sr_o)
    ek = jnp.zeros_like(state.ek)
    history_width = min(singular_values.shape[0], sr_rank_max)
    sr_o_values = (
        u[:, :history_width]
        * singular_values[:history_width]
        * retained_float[:history_width]
    )
    ek_values = (vh[:history_width, :] @ e_aug) * retained_float[:history_width]
    sr_o = sr_o.at[:, :history_width].set(sr_o_values)
    ek = ek.at[:history_width].set(ek_values)

    updated_sr_rank = _update_working_rank(
        active_rank, state.sr_rank, rank_update_max, sr_scale
    )
    capped_sr_rank = jnp.minimum(
        state.sr_rank,
        jnp.asarray(rank_update_max, dtype=state.sr_rank.dtype),
    )
    new_sr_rank = jnp.where(valid_update, updated_sr_rank, capped_sr_rank)
    new_state = WSSRCoreState(
        sr_o=jnp.where(valid_update, sr_o, jnp.zeros_like(sr_o)),
        ek=jnp.where(valid_update, ek, jnp.zeros_like(ek)),
        sr_rank0=jnp.where(
            valid_update, active_rank, jnp.asarray(0, dtype=state.sr_rank0.dtype)
        ),
        sr_rank=new_sr_rank,
    )
    return WSSRSVDResult(grad_like_update, new_state, active_rank)


def wssr_svd_core_update(
    o_aug: Array,
    e_aug: Array,
    state: WSSRCoreState,
    damping: chex.Numeric,
    norm_constraint: chex.Numeric,
    sr_rank_max: int,
    sr_scale: chex.Numeric = 1.1,
    constrain_update_norm: bool = True,
    eps: chex.Numeric = 1e-12,
) -> WSSRSVDResult:
    """Compute one exact-SVD WSSR core update.

    The returned vector is a gradient-like raw SR vector:

    - no learning-rate factor is applied,
    - no ``-1`` is applied,
    - inputs are expected to already use the Julia ``sqrt(N)`` scaling.

    This PR 1 helper uses eager Python integer rank handling. The integrated PR 2 path
    must revisit dynamic rank slicing if the function is jitted.
    """
    sr_rank = int(state.sr_rank)
    if sr_rank == 0 or sr_rank_max == 0:
        zero_state = _zero_history_like(state)
        return WSSRSVDResult(jnp.zeros(o_aug.shape[0], dtype=o_aug.dtype), zero_state, 0)

    working_rank = min(sr_rank, o_aug.shape[0], o_aug.shape[1])
    if working_rank == 0:
        zero_state = _zero_history_like(state)
        return WSSRSVDResult(jnp.zeros(o_aug.shape[0], dtype=o_aug.dtype), zero_state, 0)

    u, singular_values, vh = jnp.linalg.svd(o_aug, full_matrices=False)
    singular_values = singular_values[:working_rank]
    u = u[:, :working_rank]
    vh = vh[:working_rank, :]

    return _wssr_update_from_svd(
        o_aug,
        e_aug,
        state,
        u,
        singular_values,
        vh,
        damping,
        norm_constraint,
        sr_rank_max,
        sr_scale,
        constrain_update_norm,
        eps=eps,
    )


def avg_minsr_svd_history_core_update(
    o_aug: Array,
    e_aug: Array,
    state: WSSRCoreState,
    lambda_reg: chex.Numeric,
    damping: chex.Numeric,
    norm_constraint: chex.Numeric,
    sr_rank_max: int,
    sr_scale: chex.Numeric = 1.1,
    constrain_update_norm: bool = True,
    eps: chex.Numeric = 1e-12,
) -> AvgMinSRSVDHistoryResult:
    """Compute averaged-MinSR with SVD-compressed WSSR-style history.

    This ablation baseline uses a Tikhonov dual solve for the update direction:
    ``A @ solve(A.T @ A + lambda_reg * I, b)``. The SVD is used only to update
    the compressed history buffers with the same retained-rank rule as WSSR.
    """
    width = o_aug.shape[1]
    safe_lambda = jnp.maximum(
        jnp.asarray(lambda_reg, dtype=o_aug.dtype),
        jnp.asarray(eps, dtype=o_aug.dtype),
    )
    gram = o_aug.T @ o_aug
    regularized_gram = gram + safe_lambda * jnp.eye(width, dtype=o_aug.dtype)
    alpha = jnp.linalg.solve(regularized_gram, e_aug)
    grad_like_update = o_aug @ alpha
    raw_update_norm = jnp.linalg.norm(grad_like_update)
    if constrain_update_norm:
        grad_like_update = constrain_norm(grad_like_update, norm_constraint, eps=eps)

    if width == 0 or o_aug.shape[0] == 0:
        zero_state = _zero_history_like(state)
        return AvgMinSRSVDHistoryResult(
            grad_like_update,
            zero_state,
            jnp.asarray(0, dtype=state.sr_rank.dtype),
            jnp.asarray(1.0, dtype=o_aug.dtype),
            raw_update_norm,
        )

    u, singular_values, vh = jnp.linalg.svd(o_aug, full_matrices=False)
    if singular_values.shape[0] == 0:
        zero_state = _zero_history_like(state)
        return AvgMinSRSVDHistoryResult(
            grad_like_update,
            zero_state,
            jnp.asarray(0, dtype=state.sr_rank.dtype),
            jnp.asarray(1.0, dtype=o_aug.dtype),
            raw_update_norm,
        )

    leading_sv = singular_values[0]
    valid_leading = leading_sv > eps
    safe_leading_sv = jnp.where(valid_leading, leading_sv, 1.0)
    rank_capacity = min(singular_values.shape[0], state.sr_o.shape[1], sr_rank_max)
    target_rank = jnp.minimum(
        state.sr_rank,
        jnp.asarray(rank_capacity, dtype=state.sr_rank.dtype),
    )
    rank_mask = jnp.arange(singular_values.shape[0]) < target_rank
    retained = (
        (singular_values / safe_leading_sv > damping) & valid_leading & rank_mask
    )
    active_rank = jnp.sum(retained.astype(state.sr_rank.dtype))
    valid_history = valid_leading & (active_rank > 0)
    retained_float = retained.astype(o_aug.dtype)

    sr_o = jnp.zeros_like(state.sr_o)
    ek = jnp.zeros_like(state.ek)
    history_width = rank_capacity
    sr_o_values = (
        u[:, :history_width]
        * singular_values[:history_width]
        * retained_float[:history_width]
    )
    ek_values = (vh[:history_width, :] @ e_aug) * retained_float[:history_width]
    sr_o = sr_o.at[:, :history_width].set(sr_o_values)
    ek = ek.at[:history_width].set(ek_values)

    updated_sr_rank = _update_working_rank(
        active_rank, state.sr_rank, sr_rank_max, sr_scale
    )
    capped_sr_rank = jnp.minimum(
        state.sr_rank,
        jnp.asarray(sr_rank_max, dtype=state.sr_rank.dtype),
    )
    new_sr_rank = jnp.where(valid_history, updated_sr_rank, capped_sr_rank)
    new_state = WSSRCoreState(
        sr_o=jnp.where(valid_history, sr_o, jnp.zeros_like(sr_o)),
        ek=jnp.where(valid_history, ek, jnp.zeros_like(ek)),
        sr_rank0=jnp.where(
            valid_history, active_rank, jnp.asarray(0, dtype=state.sr_rank0.dtype)
        ),
        sr_rank=new_sr_rank,
    )

    regularized_spectrum = jnp.square(singular_values) + safe_lambda
    max_eig = jnp.max(regularized_spectrum)
    min_eig = jnp.min(regularized_spectrum)
    min_eig = jnp.where(width > singular_values.shape[0], safe_lambda, min_eig)
    dual_matrix_cond = max_eig / jnp.maximum(min_eig, eps)
    return AvgMinSRSVDHistoryResult(
        grad_like_update,
        new_state,
        active_rank,
        dual_matrix_cond,
        raw_update_norm,
    )


_jitted_avg_minsr_svd_history_core_update = jax.jit(
    avg_minsr_svd_history_core_update,
    static_argnames=("sr_rank_max", "constrain_update_norm"),
)


def randomized_svd(
    o_aug: Array,
    target_rank: int,
    sketch_oversampling: int,
    sketch_n_iter: int,
    key: Array,
) -> Tuple[Array, Array, Array]:
    """Compute a randomized low-rank SVD approximation of ``o_aug``."""
    if target_rank < 0:
        raise ValueError("target_rank must be nonnegative")
    if sketch_oversampling < 0:
        raise ValueError("sketch_oversampling must be nonnegative")
    if sketch_n_iter < 0:
        raise ValueError("sketch_n_iter must be nonnegative")

    clipped_rank = min(target_rank, o_aug.shape[0], o_aug.shape[1])
    if clipped_rank == 0:
        return (
            jnp.zeros((o_aug.shape[0], 0), dtype=o_aug.dtype),
            jnp.zeros((0,), dtype=o_aug.dtype),
            jnp.zeros((0, o_aug.shape[1]), dtype=o_aug.dtype),
        )

    sketch_rank = min(
        clipped_rank + sketch_oversampling,
        o_aug.shape[0],
        o_aug.shape[1],
    )
    omega = jax.random.normal(key, (o_aug.shape[1], sketch_rank), dtype=o_aug.dtype)
    y = o_aug @ omega
    for _ in range(sketch_n_iter):
        y = o_aug @ (o_aug.T @ y)

    q, _ = jnp.linalg.qr(y, mode="reduced")
    b = q.T @ o_aug
    u_hat, singular_values, vh = jnp.linalg.svd(b, full_matrices=False)
    u = q @ u_hat
    return (
        u[:, :clipped_rank],
        singular_values[:clipped_rank],
        vh[:clipped_rank, :],
    )


def fixed_shape_randomized_svd(
    o_aug: Array,
    state: WSSRCoreState,
    key: Array,
    sr_rank_max: int,
    sketch_oversampling: int,
    sketch_n_iter: int,
) -> Tuple[Array, Array, Array]:
    """Compute a fixed-shape randomized SVD approximation for JIT-safe WSSR."""
    if sketch_oversampling < 0:
        raise ValueError("sketch_oversampling must be nonnegative")
    if sketch_n_iter < 0:
        raise ValueError("sketch_n_iter must be nonnegative")

    rank_capacity = min(state.sr_o.shape[1], sr_rank_max, o_aug.shape[0], o_aug.shape[1])
    if rank_capacity == 0:
        return (
            jnp.zeros((o_aug.shape[0], 0), dtype=o_aug.dtype),
            jnp.zeros((0,), dtype=o_aug.dtype),
            jnp.zeros((0, o_aug.shape[1]), dtype=o_aug.dtype),
        )

    sketch_capacity = min(
        rank_capacity + sketch_oversampling,
        o_aug.shape[0],
        o_aug.shape[1],
    )
    rank = jnp.minimum(
        state.sr_rank,
        jnp.asarray(rank_capacity, dtype=state.sr_rank.dtype),
    )
    sketch_rank = jnp.minimum(
        rank + jnp.asarray(sketch_oversampling, dtype=state.sr_rank.dtype),
        jnp.asarray(sketch_capacity, dtype=state.sr_rank.dtype),
    )
    rank_mask = (jnp.arange(rank_capacity) < rank).astype(o_aug.dtype)
    sketch_mask = (jnp.arange(sketch_capacity) < sketch_rank).astype(o_aug.dtype)

    omega = jax.random.normal(
        key, (o_aug.shape[1], sketch_capacity), dtype=o_aug.dtype
    )
    omega = omega * sketch_mask
    y = o_aug @ omega
    y = y * sketch_mask
    for _ in range(sketch_n_iter):
        y = o_aug @ (o_aug.T @ y)
        y = y * sketch_mask

    q, _ = jnp.linalg.qr(y, mode="reduced")
    q = q * sketch_mask
    b = q.T @ o_aug
    u_hat, singular_values, vh = jnp.linalg.svd(b, full_matrices=False)
    u = q @ u_hat
    return (
        u[:, :rank_capacity] * rank_mask,
        singular_values[:rank_capacity] * rank_mask,
        vh[:rank_capacity, :] * rank_mask[:, None],
    )


def wssr_sketch_core_update(
    o_aug: Array,
    e_aug: Array,
    state: WSSRCoreState,
    key: Array,
    damping: chex.Numeric,
    norm_constraint: chex.Numeric,
    sr_rank_max: int,
    sr_scale: chex.Numeric = 1.1,
    sketch_oversampling: int = 5,
    sketch_n_iter: int = 1,
    constrain_update_norm: bool = True,
    eps: chex.Numeric = 1e-12,
) -> WSSRSVDResult:
    """Compute one randomized-SVD WSSR core update."""
    if sr_rank_max == 0:
        zero_state = _zero_history_like(state)
        return WSSRSVDResult(
            jnp.zeros(o_aug.shape[0], dtype=o_aug.dtype),
            zero_state,
            jnp.asarray(0, dtype=state.sr_rank.dtype),
        )

    u, singular_values, vh = fixed_shape_randomized_svd(
        o_aug,
        state,
        key,
        sr_rank_max,
        sketch_oversampling,
        sketch_n_iter,
    )
    return _wssr_update_from_svd(
        o_aug,
        e_aug,
        state,
        u,
        singular_values,
        vh,
        damping,
        norm_constraint,
        sr_rank_max,
        sr_scale,
        constrain_update_norm,
        eps=eps,
    )


_jitted_wssr_sketch_core_update = jax.jit(
    wssr_sketch_core_update,
    static_argnames=(
        "sr_rank_max",
        "sketch_oversampling",
        "sketch_n_iter",
        "constrain_update_norm",
    ),
)


def warm_start_svd(
    o_aug: Array,
    state: WSSRWarmSVDCoreState,
    key: Array,
    maxiter_initial: int,
    maxiter_warm: int,
    svd_working_rank: Optional[int] = None,
    eps: chex.Numeric = 1e-12,
) -> Tuple[Array, Array, Array, Array]:
    """Compute a warm-start subspace-iteration SVD approximation."""
    if maxiter_initial < 0:
        raise ValueError("maxiter_initial must be nonnegative")
    if maxiter_warm < 0:
        raise ValueError("maxiter_warm must be nonnegative")

    storage_width = state.sr_o.shape[1]
    working_rank_max = _resolve_svd_working_rank(svd_working_rank, storage_width)
    rank_capacity = min(
        working_rank_max,
        storage_width,
        o_aug.shape[0],
        o_aug.shape[1],
    )
    if rank_capacity == 0:
        return (
            jnp.zeros((o_aug.shape[0], 0), dtype=o_aug.dtype),
            jnp.zeros((0,), dtype=o_aug.dtype),
            jnp.zeros((0, o_aug.shape[1]), dtype=o_aug.dtype),
            jnp.asarray(0, dtype=state.sr_rank.dtype),
        )

    rank = jnp.minimum(
        state.sr_rank,
        jnp.asarray(rank_capacity, dtype=state.sr_rank.dtype),
    )
    rank_mask = (jnp.arange(rank_capacity) < rank).astype(o_aug.dtype)

    def _masked_qr(y):
        q, _ = jnp.linalg.qr(y, mode="reduced")
        return q * rank_mask

    def _subspace_iterate(u, n_iter):
        for _ in range(n_iter):
            z = o_aug.T @ u
            y = o_aug @ z
            u = _masked_qr(y)
        return u

    warm_u, has_warm_u = _select_warm_left_basis(
        state,
        rank_capacity,
        eps=eps,
    )

    def _warm_branch(_):
        u0 = warm_u * rank_mask
        u, _ = jnp.linalg.qr(u0, mode="reduced")
        u = u * rank_mask
        return _subspace_iterate(u, maxiter_warm)

    def _initial_branch(_):
        omega = jax.random.normal(
            key, (o_aug.shape[1], rank_capacity), dtype=o_aug.dtype
        )
        omega = omega * rank_mask
        y = o_aug @ omega
        u = _masked_qr(y)
        return _subspace_iterate(u, maxiter_initial)

    u = jax.lax.cond(has_warm_u, _warm_branch, _initial_branch, operand=None)
    b = u.T @ o_aug
    u_hat, singular_values, vh = jnp.linalg.svd(b, full_matrices=False)
    factor_mask = (jnp.arange(singular_values.shape[0]) < rank).astype(o_aug.dtype)
    singular_values = singular_values * factor_mask
    u_final = (u @ u_hat) * factor_mask
    vh = vh * factor_mask[:, None]
    return u_final, singular_values, vh, rank


def right_warm_start_svd(
    o_aug: Array,
    state: WSSRWarmSVDCoreState,
    key: Array,
    maxiter_initial: int,
    maxiter_warm: int,
    svd_working_rank: Optional[int] = None,
    eps: chex.Numeric = 1e-12,
) -> Tuple[Array, Array, Array, Array, Array]:
    """Compute a warm-start SVD approximation from the right subspace.

    This variant still persists the left basis ``state.u`` for checkpoint
    compatibility. On each update it projects that warm start into the right
    augmented-column space, performs right-subspace iteration there, and recovers
    a fresh left basis for the shared WSSR post-SVD update.
    """
    if maxiter_initial < 0:
        raise ValueError("maxiter_initial must be nonnegative")
    if maxiter_warm < 0:
        raise ValueError("maxiter_warm must be nonnegative")

    storage_width = state.sr_o.shape[1]
    working_rank_max = _resolve_svd_working_rank(svd_working_rank, storage_width)
    rank_capacity = min(
        working_rank_max,
        storage_width,
        o_aug.shape[0],
        o_aug.shape[1],
    )
    if rank_capacity == 0:
        return (
            jnp.zeros((o_aug.shape[0], 0), dtype=o_aug.dtype),
            jnp.zeros((0,), dtype=o_aug.dtype),
            jnp.zeros((0, o_aug.shape[1]), dtype=o_aug.dtype),
            jnp.asarray(0, dtype=state.sr_rank.dtype),
        )

    rank = jnp.minimum(
        state.sr_rank,
        jnp.asarray(rank_capacity, dtype=state.sr_rank.dtype),
    )
    rank_mask = (jnp.arange(rank_capacity) < rank).astype(o_aug.dtype)

    def _masked_qr(z):
        q, _ = jnp.linalg.qr(z, mode="reduced")
        return q * rank_mask

    def _subspace_iterate(v, n_iter):
        for _ in range(n_iter):
            y = o_aug @ v
            z = o_aug.T @ y
            v = _masked_qr(z)
        return v

    def _warm_branch(_):
        if state.u.shape[1] > 0:
            warm_u, _ = _select_warm_left_basis(
                state,
                rank_capacity,
                eps=eps,
            )
            projection = o_aug.T @ (warm_u * rank_mask)
        else:
            projection = recover_right_basis_from_sr_o_projection(
                o_aug,
                state.sr_o,
                state.sr_rank0,
                rank_capacity,
                eps=eps,
            )
            projection = projection * rank_mask
        v = _masked_qr(projection)
        return _subspace_iterate(v, maxiter_warm)

    def _initial_branch(_):
        omega = jax.random.normal(
            key, (o_aug.shape[1], rank_capacity), dtype=o_aug.dtype
        )
        omega = omega * rank_mask
        v = _masked_qr(omega)
        return _subspace_iterate(v, maxiter_initial)

    has_warm_u = state.has_u | (state.sr_rank0 > 0)
    v = jax.lax.cond(has_warm_u, _warm_branch, _initial_branch, operand=None)
    y = o_aug @ v
    gram = y.T @ y
    eigvals, right_rot = jnp.linalg.eigh(gram)
    order = jnp.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    right_rot = right_rot[:, order]
    singular_values = jnp.sqrt(jnp.maximum(eigvals, 0.0)) * rank_mask

    safe_singular_values = jnp.where(singular_values > eps, singular_values, 1.0)
    u = (y @ right_rot) / safe_singular_values
    u = u * rank_mask
    vh = (v @ right_rot).T * rank_mask[:, None]
    return u, singular_values, vh, rank


def right_warm_start_svd_matfree(
    log_psi_apply: ModelApply[P],
    params: P,
    positions: Array,
    state: WSSRWarmSVDCoreState,
    eta: chex.Numeric,
    key: Array,
    maxiter_initial: int,
    maxiter_warm: int,
    svd_working_rank: Optional[int] = None,
    eps: chex.Numeric = 1e-12,
) -> Tuple[Array, Array, Array, Array]:
    """Matrix-free analogue of :func:`right_warm_start_svd`.

    This prototype uses semi-matrix-free augmented products. The history block is
    explicit in ``state.sr_o``; the current score block is applied via JVP/VJP
    without materializing ``O_cur`` or ``O_aug``.
    """
    if maxiter_initial < 0:
        raise ValueError("maxiter_initial must be nonnegative")
    if maxiter_warm < 0:
        raise ValueError("maxiter_warm must be nonnegative")

    flat_params, _ = jax.flatten_util.ravel_pytree(params)
    num_params = flat_params.shape[0]
    aug_width = state.sr_o.shape[1] + positions.shape[0]
    storage_width = state.sr_o.shape[1]
    working_rank_max = _resolve_svd_working_rank(svd_working_rank, storage_width)
    rank_capacity = min(
        working_rank_max,
        storage_width,
        num_params,
        aug_width,
    )
    if rank_capacity == 0:
        return (
            jnp.zeros((num_params, 0), dtype=flat_params.dtype),
            jnp.zeros((0,), dtype=flat_params.dtype),
            jnp.zeros((0, aug_width), dtype=flat_params.dtype),
            jnp.asarray(0, dtype=state.sr_rank.dtype),
        )

    rank = jnp.minimum(
        state.sr_rank,
        jnp.asarray(rank_capacity, dtype=state.sr_rank.dtype),
    )
    rank_mask = (jnp.arange(rank_capacity) < rank).astype(flat_params.dtype)

    def _masked_qr(z):
        q, _ = jnp.linalg.qr(z, mode="reduced")
        return q * rank_mask

    def _matmat(v):
        return wssr_augmented_matmat(
            log_psi_apply, params, positions, state, eta, v
        )

    def _rmatmat(y):
        return wssr_augmented_rmatmat(
            log_psi_apply, params, positions, state, eta, y
        )

    def _subspace_iterate(v, n_iter):
        for _ in range(n_iter):
            y = _matmat(v)
            z = _rmatmat(y)
            v = _masked_qr(z)
        return v

    warm_u, has_warm_u = _select_warm_left_basis(
        state,
        rank_capacity,
        eps=eps,
    )

    def _warm_branch(_):
        u0 = warm_u * rank_mask
        v = _masked_qr(_rmatmat(u0))
        return _subspace_iterate(v, maxiter_warm)

    def _initial_branch(_):
        omega = jax.random.normal(key, (aug_width, rank_capacity), dtype=flat_params.dtype)
        omega = omega * rank_mask
        v = _masked_qr(omega)
        return _subspace_iterate(v, maxiter_initial)

    v = jax.lax.cond(has_warm_u, _warm_branch, _initial_branch, operand=None)
    y = _matmat(v)
    gram = y.T @ y
    eigvals, right_rot = jnp.linalg.eigh(gram)
    order = jnp.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    right_rot = right_rot[:, order]
    singular_values = jnp.sqrt(jnp.maximum(eigvals, 0.0)) * rank_mask

    safe_singular_values = jnp.where(singular_values > eps, singular_values, 1.0)
    u = (y @ right_rot) / safe_singular_values
    u = u * rank_mask
    vh = (v @ right_rot).T * rank_mask[:, None]
    return u, singular_values, vh, rank


def wssr_warm_svd_core_update(
    o_aug: Array,
    e_aug: Array,
    state: WSSRWarmSVDCoreState,
    key: Array,
    damping: chex.Numeric,
    norm_constraint: chex.Numeric,
    sr_rank_max: int,
    sr_scale: chex.Numeric = 1.1,
    svd_maxiter_initial: int = 8,
    svd_maxiter_warm: int = 2,
    svd_working_rank: Optional[int] = None,
    constrain_update_norm: bool = True,
    spectral_regularization: str = "hard_floor",
    complement_weight: chex.Numeric = 1.0,
    eps: chex.Numeric = 1e-12,
) -> WSSRSVDResult:
    """Compute one warm-start subspace-iteration WSSR core update."""
    if sr_rank_max == 0:
        zero_state = _zero_warm_svd_history_like(state)
        return WSSRSVDResult(
            jnp.zeros(o_aug.shape[0], dtype=o_aug.dtype),
            zero_state,
            jnp.asarray(0, dtype=state.sr_rank.dtype),
        )

    working_rank_max = _resolve_svd_working_rank(svd_working_rank, sr_rank_max)
    u, singular_values, vh, rank = warm_start_svd(
        o_aug,
        state,
        key,
        svd_maxiter_initial,
        svd_maxiter_warm,
        svd_working_rank=working_rank_max,
        eps=eps,
    )
    result = _wssr_update_from_svd(
        o_aug,
        e_aug,
        state,
        u,
        singular_values,
        vh,
        damping,
        norm_constraint,
        sr_rank_max,
        sr_scale,
        constrain_update_norm,
        rank_update_max=working_rank_max,
        spectral_regularization=spectral_regularization,
        complement_weight=complement_weight,
        eps=eps,
    )
    u_state = jnp.zeros_like(state.u)
    warm_width = min(u.shape[1], state.u.shape[1])
    rank_mask = (jnp.arange(warm_width) < rank).astype(u.dtype)
    u_state = u_state.at[:, :warm_width].set(u[:, :warm_width] * rank_mask)
    warm_state = WSSRWarmSVDCoreState(
        sr_o=result.state.sr_o,
        ek=result.state.ek,
        sr_rank0=result.state.sr_rank0,
        sr_rank=result.state.sr_rank,
        u=u_state,
        has_u=jnp.where(rank > 0, jnp.array(True), state.has_u),
    )
    return WSSRSVDResult(result.grad_like_update, warm_state, result.active_rank)


_jitted_wssr_warm_svd_core_update = jax.jit(
    wssr_warm_svd_core_update,
    static_argnames=(
        "sr_rank_max",
        "svd_maxiter_initial",
        "svd_maxiter_warm",
        "svd_working_rank",
        "constrain_update_norm",
        "spectral_regularization",
        "complement_weight",
    ),
)


def wssr_warm_svd_right_core_update(
    o_aug: Array,
    e_aug: Array,
    state: WSSRWarmSVDCoreState,
    key: Array,
    damping: chex.Numeric,
    norm_constraint: chex.Numeric,
    sr_rank_max: int,
    sr_scale: chex.Numeric = 1.1,
    svd_maxiter_initial: int = 8,
    svd_maxiter_warm: int = 2,
    svd_working_rank: Optional[int] = None,
    constrain_update_norm: bool = True,
    spectral_regularization: str = "hard_floor",
    complement_weight: chex.Numeric = 1.0,
    eps: chex.Numeric = 1e-12,
) -> WSSRSVDResult:
    """Compute one right-subspace warm-start SVD WSSR core update."""
    storage_rank_max = min(sr_rank_max, state.sr_o.shape[1])
    if storage_rank_max == 0:
        zero_state = _zero_warm_svd_history_like(state)
        return WSSRSVDResult(
            jnp.zeros(o_aug.shape[0], dtype=o_aug.dtype),
            zero_state,
            jnp.asarray(0, dtype=state.sr_rank.dtype),
        )

    working_rank_max = _resolve_svd_working_rank(svd_working_rank, sr_rank_max)
    working_rank_max = min(working_rank_max, storage_rank_max)
    u, singular_values, vh, rank = right_warm_start_svd(
        o_aug,
        state,
        key,
        svd_maxiter_initial,
        svd_maxiter_warm,
        svd_working_rank=working_rank_max,
        eps=eps,
    )
    result = _wssr_update_from_svd(
        o_aug,
        e_aug,
        state,
        u,
        singular_values,
        vh,
        damping,
        norm_constraint,
        storage_rank_max,
        sr_scale,
        constrain_update_norm,
        rank_update_max=working_rank_max,
        spectral_regularization=spectral_regularization,
        complement_weight=complement_weight,
        eps=eps,
    )
    u_state = jnp.zeros_like(state.u)
    warm_width = min(u.shape[1], state.u.shape[1])
    rank_mask = (jnp.arange(warm_width) < rank).astype(u.dtype)
    u_state = u_state.at[:, :warm_width].set(u[:, :warm_width] * rank_mask)
    warm_state = WSSRWarmSVDCoreState(
        sr_o=result.state.sr_o,
        ek=result.state.ek,
        sr_rank0=result.state.sr_rank0,
        sr_rank=result.state.sr_rank,
        u=u_state,
        has_u=jnp.where(rank > 0, jnp.array(True), state.has_u),
    )
    return WSSRSVDResult(result.grad_like_update, warm_state, result.active_rank)


_jitted_wssr_warm_svd_right_core_update = jax.jit(
    wssr_warm_svd_right_core_update,
    static_argnames=(
        "sr_rank_max",
        "svd_maxiter_initial",
        "svd_maxiter_warm",
        "svd_working_rank",
        "constrain_update_norm",
        "spectral_regularization",
        "complement_weight",
    ),
)


def compute_wssr_svd_core_update(
    log_psi_apply: ModelApply[P],
    params: P,
    positions: Array,
    local_energies: Array,
    energy: Array,
    state: WSSRCoreState,
    eta: chex.Numeric,
    damping: chex.Numeric,
    norm_constraint: chex.Numeric,
    sr_rank_max: int,
    sr_scale: chex.Numeric = 1.1,
    constrain_update_norm: bool = True,
) -> Tuple[P, WSSRCoreState, int]:
    """Compute an exact-SVD WSSR core update and unflatten it to param structure."""
    o_cur, unravel_fn = center_and_scale_score_matrix(
        log_psi_apply, params, positions
    )
    e_cur = center_and_scale_energy_residuals(local_energies, energy)
    o_aug, e_aug = augment_wssr_system(o_cur, e_cur, state, eta)
    result = wssr_svd_core_update(
        o_aug,
        e_aug,
        state,
        damping,
        norm_constraint,
        sr_rank_max,
        sr_scale=sr_scale,
        constrain_update_norm=constrain_update_norm,
    )
    return unravel_fn(result.grad_like_update), result.state, result.active_rank


def compute_avg_minsr_svd_history_core_update(
    log_psi_apply: ModelApply[P],
    params: P,
    positions: Array,
    local_energies: Array,
    energy: Array,
    state: WSSRCoreState,
    eta: chex.Numeric,
    lambda_reg: chex.Numeric,
    damping: chex.Numeric,
    norm_constraint: chex.Numeric,
    sr_rank_max: int,
    sr_scale: chex.Numeric = 1.1,
    constrain_update_norm: bool = True,
) -> Tuple[P, AvgMinSRSVDHistoryResult]:
    """Compute averaged-MinSR SVD-history update and unflatten it."""
    o_cur, unravel_fn = center_and_scale_score_matrix(
        log_psi_apply, params, positions
    )
    e_cur = center_and_scale_energy_residuals(local_energies, energy)
    o_aug, e_aug = augment_wssr_system(o_cur, e_cur, state, eta)
    result = _jitted_avg_minsr_svd_history_core_update(
        o_aug,
        e_aug,
        state,
        lambda_reg,
        damping,
        norm_constraint,
        sr_rank_max,
        sr_scale=sr_scale,
        constrain_update_norm=constrain_update_norm,
    )
    result = AvgMinSRSVDHistoryResult(
        unravel_fn(result.grad_like_update),
        result.state,
        result.active_rank,
        result.dual_matrix_cond,
        result.update_norm,
    )
    return result.grad_like_update, result


def compute_wssr_sketch_core_update(
    log_psi_apply: ModelApply[P],
    params: P,
    positions: Array,
    local_energies: Array,
    energy: Array,
    state: WSSRCoreState,
    key: Array,
    eta: chex.Numeric,
    damping: chex.Numeric,
    norm_constraint: chex.Numeric,
    sr_rank_max: int,
    sr_scale: chex.Numeric = 1.1,
    sketch_oversampling: int = 5,
    sketch_n_iter: int = 1,
    constrain_update_norm: bool = True,
) -> Tuple[P, WSSRCoreState, int]:
    """Compute a randomized-SVD WSSR core update and unflatten it."""
    o_cur, unravel_fn = center_and_scale_score_matrix(
        log_psi_apply, params, positions
    )
    e_cur = center_and_scale_energy_residuals(local_energies, energy)
    o_aug, e_aug = augment_wssr_system(o_cur, e_cur, state, eta)
    result = _jitted_wssr_sketch_core_update(
        o_aug,
        e_aug,
        state,
        key,
        damping,
        norm_constraint,
        sr_rank_max,
        sr_scale=sr_scale,
        sketch_oversampling=sketch_oversampling,
        sketch_n_iter=sketch_n_iter,
        constrain_update_norm=constrain_update_norm,
    )
    return unravel_fn(result.grad_like_update), result.state, result.active_rank


def compute_wssr_warm_svd_core_update(
    log_psi_apply: ModelApply[P],
    params: P,
    positions: Array,
    local_energies: Array,
    energy: Array,
    state: WSSRWarmSVDCoreState,
    key: Array,
    eta: chex.Numeric,
    damping: chex.Numeric,
    norm_constraint: chex.Numeric,
    sr_rank_max: int,
    sr_scale: chex.Numeric = 1.1,
    svd_maxiter_initial: int = 8,
    svd_maxiter_warm: int = 2,
    svd_working_rank: Optional[int] = None,
    constrain_update_norm: bool = True,
    spectral_regularization: str = "hard_floor",
    complement_weight: chex.Numeric = 1.0,
) -> Tuple[P, WSSRWarmSVDCoreState, int]:
    """Compute a warm-start SVD WSSR core update and unflatten it."""
    o_cur, unravel_fn = center_and_scale_score_matrix(
        log_psi_apply, params, positions
    )
    e_cur = center_and_scale_energy_residuals(local_energies, energy)
    o_aug, e_aug = augment_wssr_system(o_cur, e_cur, state, eta)
    result = _jitted_wssr_warm_svd_core_update(
        o_aug,
        e_aug,
        state,
        key,
        damping,
        norm_constraint,
        sr_rank_max,
        sr_scale=sr_scale,
        svd_maxiter_initial=svd_maxiter_initial,
        svd_maxiter_warm=svd_maxiter_warm,
        svd_working_rank=svd_working_rank,
        constrain_update_norm=constrain_update_norm,
        spectral_regularization=spectral_regularization,
        complement_weight=complement_weight,
    )
    return unravel_fn(result.grad_like_update), result.state, result.active_rank


def compute_wssr_warm_svd_right_core_update(
    log_psi_apply: ModelApply[P],
    params: P,
    positions: Array,
    local_energies: Array,
    energy: Array,
    state: WSSRWarmSVDCoreState,
    key: Array,
    eta: chex.Numeric,
    damping: chex.Numeric,
    norm_constraint: chex.Numeric,
    sr_rank_max: int,
    sr_scale: chex.Numeric = 1.1,
    svd_maxiter_initial: int = 8,
    svd_maxiter_warm: int = 2,
    svd_working_rank: Optional[int] = None,
    constrain_update_norm: bool = True,
    spectral_regularization: str = "hard_floor",
    complement_weight: chex.Numeric = 1.0,
) -> Tuple[P, WSSRWarmSVDCoreState, int]:
    """Compute a right-subspace warm-start SVD WSSR update and unflatten it."""
    o_cur, unravel_fn = center_and_scale_score_matrix(
        log_psi_apply, params, positions
    )
    e_cur = center_and_scale_energy_residuals(local_energies, energy)
    o_aug, e_aug = augment_wssr_system(o_cur, e_cur, state, eta)
    result = _jitted_wssr_warm_svd_right_core_update(
        o_aug,
        e_aug,
        state,
        key,
        damping,
        norm_constraint,
        sr_rank_max,
        sr_scale=sr_scale,
        svd_maxiter_initial=svd_maxiter_initial,
        svd_maxiter_warm=svd_maxiter_warm,
        svd_working_rank=svd_working_rank,
        constrain_update_norm=constrain_update_norm,
        spectral_regularization=spectral_regularization,
        complement_weight=complement_weight,
    )
    return unravel_fn(result.grad_like_update), result.state, result.active_rank


def compute_wssr_warm_svd_right_core_update_matfree(
    log_psi_apply: ModelApply[P],
    params: P,
    positions: Array,
    local_energies: Array,
    energy: Array,
    state: WSSRWarmSVDCoreState,
    key: Array,
    eta: chex.Numeric,
    damping: chex.Numeric,
    norm_constraint: chex.Numeric,
    sr_rank_max: int,
    sr_scale: chex.Numeric = 1.1,
    svd_maxiter_initial: int = 8,
    svd_maxiter_warm: int = 2,
    svd_working_rank: Optional[int] = None,
    constrain_update_norm: bool = True,
    spectral_regularization: str = "hard_floor",
    complement_weight: chex.Numeric = 1.0,
) -> Tuple[P, WSSRWarmSVDCoreState, int]:
    """Matrix-free right-subspace warm-start SVD WSSR core update."""
    _, unravel_fn = jax.flatten_util.ravel_pytree(params)
    e_cur = center_and_scale_energy_residuals(local_energies, energy)
    e_aug = augment_wssr_residuals(e_cur, state, eta)
    storage_rank_max = min(sr_rank_max, state.sr_o.shape[1])
    working_rank_max = _resolve_svd_working_rank(svd_working_rank, sr_rank_max)
    working_rank_max = min(working_rank_max, storage_rank_max)
    u, singular_values, vh, rank = right_warm_start_svd_matfree(
        log_psi_apply,
        params,
        positions,
        state,
        eta,
        key,
        svd_maxiter_initial,
        svd_maxiter_warm,
        svd_working_rank=working_rank_max,
    )
    result = _wssr_update_from_svd_matfree(
        log_psi_apply,
        params,
        positions,
        e_aug,
        state,
        eta,
        u,
        singular_values,
        vh,
        damping,
        norm_constraint,
        storage_rank_max,
        sr_scale,
        constrain_update_norm,
        rank_update_max=working_rank_max,
        spectral_regularization=spectral_regularization,
        complement_weight=complement_weight,
    )
    u_state = jnp.zeros_like(state.u)
    warm_width = min(u.shape[1], state.u.shape[1])
    rank_mask = (jnp.arange(warm_width) < rank).astype(u.dtype)
    u_state = u_state.at[:, :warm_width].set(u[:, :warm_width] * rank_mask)
    warm_state = WSSRWarmSVDCoreState(
        sr_o=result.state.sr_o,
        ek=result.state.ek,
        sr_rank0=result.state.sr_rank0,
        sr_rank=result.state.sr_rank,
        u=u_state,
        has_u=jnp.where(rank > 0, jnp.array(True), state.has_u),
    )
    return unravel_fn(result.grad_like_update), warm_state, result.active_rank


def construct_wssr_svd_update_param_fn(
    log_psi_apply: ModelApply[P],
    energy_and_statistics_fn,
    optimizer: optax.GradientTransformation,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
) -> UpdateParamFn[P, D, WSSROptimizerState]:
    """Create the integrated WSSR SVD update function.

    The pure WSSR core returns a gradient-like raw SR vector. This integration layer
    passes that object to ``optax.sgd``, which applies the descent sign and learning
    rate exactly once. If enabled, the norm constraint is applied after Optax so the
    final parameter displacement is bounded by ``sqrt(norm_constraint)``.

    This PR 2 integration intentionally returns an untraced callable because the PR 1
    core uses dynamic Python rank handling. A future integration can add a JIT-safe
    fixed-shape/masked implementation.
    """

    def update_param_fn(params, data, optimizer_state, key):
        position = get_position_fn(data)
        energy, local_energies, stats = energy_and_statistics_fn(params, position)

        grad_like_update, core_state, active_rank = compute_wssr_svd_core_update(
            log_psi_apply,
            params,
            position,
            local_energies,
            energy,
            optimizer_state.core_state,
            optimizer_config.eta,
            optimizer_config.damping,
            optimizer_config.norm_constraint,
            optimizer_config.sr_rank_max,
            sr_scale=optimizer_config.sr_scale,
            constrain_update_norm=False,
        )

        updates, optax_state = optimizer.update(
            grad_like_update, optimizer_state.optax_state, params
        )
        if optimizer_config.constrain_norm:
            updates = constrain_update_tree_norm(
                updates, optimizer_config.norm_constraint
            )
        params = optax.apply_updates(params, updates)
        data = update_data_fn(data, params)

        metrics = {"energy": energy, "variance": stats["variance"]}
        metrics = update_metrics_with_noclip(
            stats["energy_noclip"],
            stats["variance_noclip"],
            metrics,
        )
        metrics = _update_metrics_with_wssr_rank_diagnostics(
            metrics, active_rank, core_state, optimizer_config
        )
        if record_param_l1_norm:
            metrics.update({"param_l1_norm": tree_reduce_l1(params)})

        return (
            params,
            data,
            WSSROptimizerState(core_state=core_state, optax_state=optax_state),
            metrics,
            key,
        )

    return update_param_fn


def construct_wssr_sketch_update_param_fn(
    log_psi_apply: ModelApply[P],
    energy_and_statistics_fn,
    optimizer: optax.GradientTransformation,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
) -> UpdateParamFn[P, D, WSSROptimizerState]:
    """Create the integrated randomized-SVD WSSR update function."""

    def update_param_fn(params, data, optimizer_state, key):
        position = get_position_fn(data)
        energy, local_energies, stats = energy_and_statistics_fn(params, position)
        key, sketch_key = jax.random.split(key)

        grad_like_update, core_state, active_rank = compute_wssr_sketch_core_update(
            log_psi_apply,
            params,
            position,
            local_energies,
            energy,
            optimizer_state.core_state,
            sketch_key,
            optimizer_config.eta,
            optimizer_config.damping,
            optimizer_config.norm_constraint,
            optimizer_config.sr_rank_max,
            sr_scale=optimizer_config.sr_scale,
            sketch_oversampling=optimizer_config.sketch_oversampling,
            sketch_n_iter=optimizer_config.sketch_n_iter,
            constrain_update_norm=False,
        )

        updates, optax_state = optimizer.update(
            grad_like_update, optimizer_state.optax_state, params
        )
        if optimizer_config.constrain_norm:
            updates = constrain_update_tree_norm(
                updates, optimizer_config.norm_constraint
            )
        params = optax.apply_updates(params, updates)
        data = update_data_fn(data, params)

        metrics = {"energy": energy, "variance": stats["variance"]}
        metrics = update_metrics_with_noclip(
            stats["energy_noclip"],
            stats["variance_noclip"],
            metrics,
        )
        metrics = _update_metrics_with_wssr_rank_diagnostics(
            metrics, active_rank, core_state, optimizer_config
        )
        if record_param_l1_norm:
            metrics.update({"param_l1_norm": tree_reduce_l1(params)})

        return (
            params,
            data,
            WSSROptimizerState(core_state=core_state, optax_state=optax_state),
            metrics,
            key,
        )

    return jax.jit(update_param_fn)


def construct_wssr_warm_svd_update_param_fn(
    log_psi_apply: ModelApply[P],
    energy_and_statistics_fn,
    optimizer: optax.GradientTransformation,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
) -> UpdateParamFn[P, D, WSSROptimizerState]:
    """Create the integrated warm-start SVD WSSR update function."""

    def update_param_fn(params, data, optimizer_state, key):
        position = get_position_fn(data)
        energy, local_energies, stats = energy_and_statistics_fn(params, position)
        key, svd_key = jax.random.split(key)

        grad_like_update, core_state, active_rank = compute_wssr_warm_svd_core_update(
            log_psi_apply,
            params,
            position,
            local_energies,
            energy,
            optimizer_state.core_state,
            svd_key,
            optimizer_config.eta,
            optimizer_config.damping,
            optimizer_config.norm_constraint,
            optimizer_config.sr_rank_max,
            sr_scale=optimizer_config.sr_scale,
            svd_maxiter_initial=optimizer_config.svd_maxiter_initial,
            svd_maxiter_warm=optimizer_config.svd_maxiter_warm,
            svd_working_rank=optimizer_config.get("svd_working_rank", None),
            constrain_update_norm=False,
        )

        updates, optax_state = optimizer.update(
            grad_like_update, optimizer_state.optax_state, params
        )
        if optimizer_config.constrain_norm:
            updates = constrain_update_tree_norm(
                updates, optimizer_config.norm_constraint
            )
        params = optax.apply_updates(params, updates)
        data = update_data_fn(data, params)

        metrics = {"energy": energy, "variance": stats["variance"]}
        metrics = update_metrics_with_noclip(
            stats["energy_noclip"],
            stats["variance_noclip"],
            metrics,
        )
        metrics = _update_metrics_with_wssr_rank_diagnostics(
            metrics,
            active_rank,
            core_state,
            optimizer_config,
            svd_working_rank=optimizer_config.get("svd_working_rank", None),
        )
        if record_param_l1_norm:
            metrics.update({"param_l1_norm": tree_reduce_l1(params)})

        return (
            params,
            data,
            WSSROptimizerState(core_state=core_state, optax_state=optax_state),
            metrics,
            key,
        )

    return jax.jit(update_param_fn)


def construct_wssr_warm_svd_right_update_param_fn(
    log_psi_apply: ModelApply[P],
    energy_and_statistics_fn,
    optimizer: optax.GradientTransformation,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
) -> UpdateParamFn[P, D, WSSROptimizerState]:
    """Create the integrated right-subspace warm-start SVD WSSR update function."""
    _validate_wssr_spectral_regularization(
        optimizer_config.get("spectral_regularization", "hard_floor"),
        optimizer_config.get("complement_weight", 1.0),
    )

    def update_param_fn(params, data, optimizer_state, key):
        position = get_position_fn(data)
        energy, local_energies, stats = energy_and_statistics_fn(params, position)
        key, svd_key = jax.random.split(key)

        grad_like_update, core_state, active_rank = (
            compute_wssr_warm_svd_right_core_update(
                log_psi_apply,
                params,
                position,
                local_energies,
                energy,
                optimizer_state.core_state,
                svd_key,
                optimizer_config.eta,
                optimizer_config.damping,
                optimizer_config.norm_constraint,
                optimizer_config.sr_rank_max,
                sr_scale=optimizer_config.sr_scale,
                svd_maxiter_initial=optimizer_config.svd_maxiter_initial,
                svd_maxiter_warm=optimizer_config.svd_maxiter_warm,
                svd_working_rank=optimizer_config.get("svd_working_rank", None),
                constrain_update_norm=False,
                spectral_regularization=optimizer_config.get(
                    "spectral_regularization", "hard_floor"
                ),
                complement_weight=optimizer_config.get("complement_weight", 1.0),
            )
        )

        updates, optax_state = optimizer.update(
            grad_like_update, optimizer_state.optax_state, params
        )
        if optimizer_config.constrain_norm:
            updates = constrain_update_tree_norm(
                updates, optimizer_config.norm_constraint
            )
        params = optax.apply_updates(params, updates)
        data = update_data_fn(data, params)

        metrics = {"energy": energy, "variance": stats["variance"]}
        metrics = update_metrics_with_noclip(
            stats["energy_noclip"],
            stats["variance_noclip"],
            metrics,
        )
        metrics = _update_metrics_with_wssr_rank_diagnostics(
            metrics,
            active_rank,
            core_state,
            optimizer_config,
            svd_working_rank=optimizer_config.get("svd_working_rank", None),
        )
        if record_param_l1_norm:
            metrics.update({"param_l1_norm": tree_reduce_l1(params)})

        return (
            params,
            data,
            WSSROptimizerState(core_state=core_state, optax_state=optax_state),
            metrics,
            key,
        )

    return jax.jit(update_param_fn)


def construct_wssr_warm_svd_right_matfree_update_param_fn(
    log_psi_apply: ModelApply[P],
    energy_and_statistics_fn,
    optimizer: optax.GradientTransformation,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
) -> UpdateParamFn[P, D, WSSROptimizerState]:
    """Create the experimental matrix-free right warm-SVD WSSR update function."""
    _validate_wssr_spectral_regularization(
        optimizer_config.get("spectral_regularization", "hard_floor"),
        optimizer_config.get("complement_weight", 1.0),
    )

    def update_param_fn(params, data, optimizer_state, key):
        position = get_position_fn(data)
        energy, local_energies, stats = energy_and_statistics_fn(params, position)
        key, svd_key = jax.random.split(key)

        grad_like_update, core_state, active_rank = (
            compute_wssr_warm_svd_right_core_update_matfree(
                log_psi_apply,
                params,
                position,
                local_energies,
                energy,
                optimizer_state.core_state,
                svd_key,
                optimizer_config.eta,
                optimizer_config.damping,
                optimizer_config.norm_constraint,
                optimizer_config.sr_rank_max,
                sr_scale=optimizer_config.sr_scale,
                svd_maxiter_initial=optimizer_config.svd_maxiter_initial,
                svd_maxiter_warm=optimizer_config.svd_maxiter_warm,
                svd_working_rank=optimizer_config.get("svd_working_rank", None),
                constrain_update_norm=False,
                spectral_regularization=optimizer_config.get(
                    "spectral_regularization", "hard_floor"
                ),
                complement_weight=optimizer_config.get("complement_weight", 1.0),
            )
        )

        updates, optax_state = optimizer.update(
            grad_like_update, optimizer_state.optax_state, params
        )
        if optimizer_config.constrain_norm:
            updates = constrain_update_tree_norm(
                updates, optimizer_config.norm_constraint
            )
        params = optax.apply_updates(params, updates)
        data = update_data_fn(data, params)

        metrics = {"energy": energy, "variance": stats["variance"]}
        metrics = update_metrics_with_noclip(
            stats["energy_noclip"],
            stats["variance_noclip"],
            metrics,
        )
        metrics = _update_metrics_with_wssr_rank_diagnostics(
            metrics,
            active_rank,
            core_state,
            optimizer_config,
            svd_working_rank=optimizer_config.get("svd_working_rank", None),
        )
        if record_param_l1_norm:
            metrics.update({"param_l1_norm": tree_reduce_l1(params)})

        return (
            params,
            data,
            WSSROptimizerState(core_state=core_state, optax_state=optax_state),
            metrics,
            key,
        )

    return update_param_fn


def construct_avg_minsr_svd_history_update_param_fn(
    log_psi_apply: ModelApply[P],
    energy_and_statistics_fn,
    optimizer: optax.GradientTransformation,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
) -> UpdateParamFn[P, D, WSSROptimizerState]:
    """Create the integrated SVD-compressed averaged-MinSR update function."""
    if optimizer_config.lambda_reg < 0:
        raise ValueError("lambda_reg must be nonnegative")

    def update_param_fn(params, data, optimizer_state, key):
        position = get_position_fn(data)
        energy, local_energies, stats = energy_and_statistics_fn(params, position)

        grad_like_update, result = compute_avg_minsr_svd_history_core_update(
            log_psi_apply,
            params,
            position,
            local_energies,
            energy,
            optimizer_state.core_state,
            optimizer_config.eta,
            optimizer_config.lambda_reg,
            optimizer_config.damping,
            optimizer_config.norm_constraint,
            optimizer_config.sr_rank_max,
            sr_scale=optimizer_config.sr_scale,
            constrain_update_norm=False,
        )

        updates, optax_state = optimizer.update(
            grad_like_update, optimizer_state.optax_state, params
        )
        if optimizer_config.constrain_norm:
            updates = constrain_update_tree_norm(
                updates, optimizer_config.norm_constraint
            )
        params = optax.apply_updates(params, updates)
        data = update_data_fn(data, params)

        metrics = {"energy": energy, "variance": stats["variance"]}
        metrics = update_metrics_with_noclip(
            stats["energy_noclip"],
            stats["variance_noclip"],
            metrics,
        )
        metrics = _update_metrics_with_avg_minsr_diagnostics(
            metrics,
            result,
            optimizer_config.lambda_reg,
        )
        if record_param_l1_norm:
            metrics.update({"param_l1_norm": tree_reduce_l1(params)})

        return (
            params,
            data,
            WSSROptimizerState(
                core_state=result.state,
                optax_state=optax_state,
            ),
            metrics,
            key,
        )

    return jax.jit(update_param_fn)


def initialize_avg_minsr_svd_history(
    log_psi_apply: ModelApply[P],
    energy_and_statistics_fn,
    params: P,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    learning_rate_schedule: LearningRateSchedule,
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
) -> Tuple[UpdateParamFn[P, D, WSSROptimizerState], WSSROptimizerState]:
    """Get an update function and initial state for averaged-MinSR SVD history."""
    if apply_pmap:
        raise NotImplementedError(
            "avg_minsr_svd_history currently supports apply_pmap=False only"
        )

    flat_params, _ = jax.flatten_util.ravel_pytree(params)
    core_state = initialize_wssr_core_state(
        flat_params.shape[0],
        optimizer_config.sr_rank,
        optimizer_config.sr_rank_max,
        dtype=flat_params.dtype,
    )
    optimizer = optax.sgd(
        learning_rate=learning_rate_schedule, momentum=0, nesterov=False
    )
    optax_state = optimizer.init(params)
    update_param_fn = construct_avg_minsr_svd_history_update_param_fn(
        log_psi_apply,
        energy_and_statistics_fn,
        optimizer,
        get_position_fn,
        update_data_fn,
        optimizer_config,
        record_param_l1_norm=record_param_l1_norm,
    )
    return (
        update_param_fn,
        WSSROptimizerState(core_state=core_state, optax_state=optax_state),
    )


def initialize_wssr_svd(
    log_psi_apply: ModelApply[P],
    energy_and_statistics_fn,
    params: P,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    learning_rate_schedule: LearningRateSchedule,
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
) -> Tuple[UpdateParamFn[P, D, WSSROptimizerState], WSSROptimizerState]:
    """Get an update function and initial state for exact-SVD WSSR."""
    if apply_pmap:
        raise NotImplementedError("wssr_svd currently supports apply_pmap=False only")

    flat_params, _ = jax.flatten_util.ravel_pytree(params)
    core_state = initialize_wssr_core_state(
        flat_params.shape[0],
        optimizer_config.sr_rank,
        optimizer_config.sr_rank_max,
        dtype=flat_params.dtype,
    )
    optimizer = optax.sgd(
        learning_rate=learning_rate_schedule, momentum=0, nesterov=False
    )
    optax_state = optimizer.init(params)
    update_param_fn = construct_wssr_svd_update_param_fn(
        log_psi_apply,
        energy_and_statistics_fn,
        optimizer,
        get_position_fn,
        update_data_fn,
        optimizer_config,
        record_param_l1_norm=record_param_l1_norm,
    )
    return (
        update_param_fn,
        WSSROptimizerState(core_state=core_state, optax_state=optax_state),
    )


def initialize_wssr_warm_svd_right_matfree(
    log_psi_apply: ModelApply[P],
    energy_and_statistics_fn,
    params: P,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    learning_rate_schedule: LearningRateSchedule,
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
) -> Tuple[UpdateParamFn[P, D, WSSROptimizerState], WSSROptimizerState]:
    """Get an experimental matrix-free right warm-start SVD WSSR updater."""
    if apply_pmap:
        raise NotImplementedError(
            "wssr_warm_svd_right_matfree currently supports apply_pmap=False only"
        )

    flat_params, _ = jax.flatten_util.ravel_pytree(params)
    storage_rank = resolve_wssr_storage_rank(
        optimizer_config.sr_rank,
        optimizer_config.sr_rank_max,
        optimizer_config.get("sr_storage_rank", -1),
    )
    core_state = initialize_wssr_warm_svd_core_state(
        flat_params.shape[0],
        optimizer_config.sr_rank,
        storage_rank,
        dtype=flat_params.dtype,
        store_warm_u=optimizer_config.get("store_warm_u", True),
    )
    optimizer = optax.sgd(
        learning_rate=learning_rate_schedule, momentum=0, nesterov=False
    )
    optax_state = optimizer.init(params)
    update_param_fn = construct_wssr_warm_svd_right_matfree_update_param_fn(
        log_psi_apply,
        energy_and_statistics_fn,
        optimizer,
        get_position_fn,
        update_data_fn,
        optimizer_config,
        record_param_l1_norm=record_param_l1_norm,
    )
    return (
        update_param_fn,
        WSSROptimizerState(core_state=core_state, optax_state=optax_state),
    )


def initialize_wssr_sketch(
    log_psi_apply: ModelApply[P],
    energy_and_statistics_fn,
    params: P,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    learning_rate_schedule: LearningRateSchedule,
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
) -> Tuple[UpdateParamFn[P, D, WSSROptimizerState], WSSROptimizerState]:
    """Get an update function and initial state for randomized-SVD WSSR."""
    if apply_pmap:
        raise NotImplementedError("wssr_sketch currently supports apply_pmap=False only")

    flat_params, _ = jax.flatten_util.ravel_pytree(params)
    core_state = initialize_wssr_core_state(
        flat_params.shape[0],
        optimizer_config.sr_rank,
        optimizer_config.sr_rank_max,
        dtype=flat_params.dtype,
    )
    optimizer = optax.sgd(
        learning_rate=learning_rate_schedule, momentum=0, nesterov=False
    )
    optax_state = optimizer.init(params)
    update_param_fn = construct_wssr_sketch_update_param_fn(
        log_psi_apply,
        energy_and_statistics_fn,
        optimizer,
        get_position_fn,
        update_data_fn,
        optimizer_config,
        record_param_l1_norm=record_param_l1_norm,
    )
    return (
        update_param_fn,
        WSSROptimizerState(core_state=core_state, optax_state=optax_state),
    )


def initialize_wssr_warm_svd(
    log_psi_apply: ModelApply[P],
    energy_and_statistics_fn,
    params: P,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    learning_rate_schedule: LearningRateSchedule,
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
) -> Tuple[UpdateParamFn[P, D, WSSROptimizerState], WSSROptimizerState]:
    """Get an update function and initial state for warm-start SVD WSSR."""
    if apply_pmap:
        raise NotImplementedError(
            "wssr_warm_svd currently supports apply_pmap=False only"
        )
    if not optimizer_config.get("store_warm_u", True):
        raise NotImplementedError(
            "wssr_warm_svd with store_warm_u=False is unsupported; use "
            "wssr_warm_svd_right for zero-width warm-U storage"
        )

    flat_params, _ = jax.flatten_util.ravel_pytree(params)
    core_state = initialize_wssr_warm_svd_core_state(
        flat_params.shape[0],
        optimizer_config.sr_rank,
        optimizer_config.sr_rank_max,
        dtype=flat_params.dtype,
        store_warm_u=optimizer_config.get("store_warm_u", True),
    )
    optimizer = optax.sgd(
        learning_rate=learning_rate_schedule, momentum=0, nesterov=False
    )
    optax_state = optimizer.init(params)
    update_param_fn = construct_wssr_warm_svd_update_param_fn(
        log_psi_apply,
        energy_and_statistics_fn,
        optimizer,
        get_position_fn,
        update_data_fn,
        optimizer_config,
        record_param_l1_norm=record_param_l1_norm,
    )
    return (
        update_param_fn,
        WSSROptimizerState(core_state=core_state, optax_state=optax_state),
    )


def initialize_wssr_warm_svd_right(
    log_psi_apply: ModelApply[P],
    energy_and_statistics_fn,
    params: P,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    learning_rate_schedule: LearningRateSchedule,
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
) -> Tuple[UpdateParamFn[P, D, WSSROptimizerState], WSSROptimizerState]:
    """Get an update function and initial state for right warm-start SVD WSSR."""
    if apply_pmap:
        raise NotImplementedError(
            "wssr_warm_svd_right currently supports apply_pmap=False only"
        )

    flat_params, _ = jax.flatten_util.ravel_pytree(params)
    storage_rank = resolve_wssr_storage_rank(
        optimizer_config.sr_rank,
        optimizer_config.sr_rank_max,
        optimizer_config.get("sr_storage_rank", -1),
    )
    core_state = initialize_wssr_warm_svd_core_state(
        flat_params.shape[0],
        optimizer_config.sr_rank,
        storage_rank,
        dtype=flat_params.dtype,
        store_warm_u=optimizer_config.get("store_warm_u", True),
    )
    optimizer = optax.sgd(
        learning_rate=learning_rate_schedule, momentum=0, nesterov=False
    )
    optax_state = optimizer.init(params)
    update_param_fn = construct_wssr_warm_svd_right_update_param_fn(
        log_psi_apply,
        energy_and_statistics_fn,
        optimizer,
        get_position_fn,
        update_data_fn,
        optimizer_config,
        record_param_l1_norm=record_param_l1_norm,
    )
    return (
        update_param_fn,
        WSSROptimizerState(core_state=core_state, optax_state=optax_state),
    )
