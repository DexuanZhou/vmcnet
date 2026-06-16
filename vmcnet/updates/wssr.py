"""WSSR low-rank stochastic reconfiguration helpers."""

import math
from typing import NamedTuple, Tuple

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
    active_rank: int


class WSSROptimizerState(NamedTuple):
    """Integrated WSSR optimizer state."""

    core_state: WSSRCoreState
    optax_state: optax.OptState


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
) -> WSSRWarmSVDCoreState:
    """Initialize WSSR history plus fixed-shape warm-SVD subspace."""
    core_state = initialize_wssr_core_state(num_params, sr_rank, sr_rank_max, dtype)
    return WSSRWarmSVDCoreState(
        sr_o=core_state.sr_o,
        ek=core_state.ek,
        sr_rank0=core_state.sr_rank0,
        sr_rank=core_state.sr_rank,
        u=jnp.zeros((num_params, sr_rank_max), dtype=dtype),
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

    def ravel_grad_log_psi(position):
        grad = jax.grad(log_psi_apply, argnums=0)(params, position)
        return jax.flatten_util.ravel_pytree(grad)[0]

    score_samples = jax.vmap(ravel_grad_log_psi, in_axes=0)(positions)
    score_samples = score_samples - jnp.mean(score_samples, axis=0, keepdims=True)
    scale = jnp.sqrt(positions.shape[0])
    return score_samples.T / scale, unravel_fn


def center_and_scale_energy_residuals(local_energies: Array, energy: Array) -> Array:
    """Center local-energy residuals and apply Julia WSSR scaling."""
    residuals = local_energies - energy
    return residuals / jnp.sqrt(local_energies.shape[0])


def augment_wssr_system(
    o_cur: Array,
    e_cur: Array,
    state: WSSRCoreState,
    eta: chex.Numeric,
) -> Tuple[Array, Array]:
    """Create history-augmented WSSR score matrix and residual vector.

    This PR 1 helper uses Python integer slicing and is intended for eager core tests,
    not as the final jitted integration path.
    """
    active_rank = int(state.sr_rank0)
    if active_rank == 0:
        return o_cur, e_cur

    sqrt_eta = jnp.sqrt(eta)
    sqrt_one_minus_eta = jnp.sqrt(1.0 - eta)
    o_hist = sqrt_eta * state.sr_o[:, :active_rank]
    e_hist = sqrt_eta * state.ek[:active_rank]
    o_scaled = sqrt_one_minus_eta * o_cur
    e_scaled = sqrt_one_minus_eta * e_cur
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


def _update_working_rank(active_rank: int, sr_rank: int, sr_rank_max: int, sr_scale):
    if active_rank == sr_rank and sr_rank < sr_rank_max:
        return min(int(math.ceil(sr_rank * sr_scale)), sr_rank_max)
    return sr_rank


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
    eps: chex.Numeric = 1e-12,
) -> WSSRSVDResult:
    """Apply the shared post-SVD WSSR update formula."""
    if singular_values.shape[0] == 0:
        zero_state = _zero_history_like(state)
        return WSSRSVDResult(jnp.zeros(o_aug.shape[0], dtype=o_aug.dtype), zero_state, 0)

    leading_sv = singular_values[0]
    if bool(leading_sv <= eps):
        zero_state = _zero_history_like(state)
        return WSSRSVDResult(jnp.zeros(o_aug.shape[0], dtype=o_aug.dtype), zero_state, 0)

    retained = singular_values / leading_sv > damping
    active_rank = int(jnp.sum(retained))
    if active_rank == 0:
        zero_state = _zero_history_like(state)
        return WSSRSVDResult(jnp.zeros(o_aug.shape[0], dtype=o_aug.dtype), zero_state, 0)

    u_active = u[:, :active_rank]
    s_active = singular_values[:active_rank]
    v_active = vh[:active_rank, :].T

    safe_damping = jnp.maximum(jnp.abs(damping), eps)
    sigma0 = 1.0 / jnp.square(safe_damping * jnp.abs(leading_sv))
    force = o_aug @ e_aug
    projected_force = u_active.T @ force
    projected_force = projected_force * (jnp.square(1.0 / s_active) - sigma0)
    grad_like_update = u_active @ projected_force + sigma0 * force

    if constrain_update_norm:
        grad_like_update = constrain_norm(grad_like_update, norm_constraint, eps=eps)

    sr_o = jnp.zeros_like(state.sr_o)
    ek = jnp.zeros_like(state.ek)
    sr_o = sr_o.at[:, :active_rank].set(u_active * s_active)
    ek = ek.at[:active_rank].set(v_active.T @ e_aug)

    new_sr_rank = _update_working_rank(
        active_rank, int(state.sr_rank), sr_rank_max, sr_scale
    )
    new_state = WSSRCoreState(
        sr_o=sr_o,
        ek=ek,
        sr_rank0=jnp.array(active_rank),
        sr_rank=jnp.array(new_sr_rank),
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
    sr_rank = int(state.sr_rank)
    if sr_rank == 0 or sr_rank_max == 0:
        zero_state = _zero_history_like(state)
        return WSSRSVDResult(jnp.zeros(o_aug.shape[0], dtype=o_aug.dtype), zero_state, 0)

    working_rank = min(sr_rank, o_aug.shape[0], o_aug.shape[1])
    if working_rank == 0:
        zero_state = _zero_history_like(state)
        return WSSRSVDResult(jnp.zeros(o_aug.shape[0], dtype=o_aug.dtype), zero_state, 0)

    u, singular_values, vh = randomized_svd(
        o_aug,
        working_rank,
        sketch_oversampling,
        sketch_n_iter,
        key,
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


def warm_start_svd(
    o_aug: Array,
    state: WSSRWarmSVDCoreState,
    key: Array,
    maxiter_initial: int,
    maxiter_warm: int,
) -> Tuple[Array, Array, Array, int]:
    """Compute a warm-start subspace-iteration SVD approximation."""
    if maxiter_initial < 0:
        raise ValueError("maxiter_initial must be nonnegative")
    if maxiter_warm < 0:
        raise ValueError("maxiter_warm must be nonnegative")

    rank = min(int(state.sr_rank), o_aug.shape[0], o_aug.shape[1])
    if rank == 0:
        return (
            jnp.zeros((o_aug.shape[0], 0), dtype=o_aug.dtype),
            jnp.zeros((0,), dtype=o_aug.dtype),
            jnp.zeros((0, o_aug.shape[1]), dtype=o_aug.dtype),
            0,
        )

    if bool(state.has_u):
        u0 = state.u[:, :rank]
        u, _ = jnp.linalg.qr(u0, mode="reduced")
        n_iter = maxiter_warm
    else:
        omega = jax.random.normal(key, (o_aug.shape[1], rank), dtype=o_aug.dtype)
        y = o_aug @ omega
        u, _ = jnp.linalg.qr(y, mode="reduced")
        n_iter = maxiter_initial

    for _ in range(n_iter):
        z = o_aug.T @ u
        y = o_aug @ z
        u, _ = jnp.linalg.qr(y, mode="reduced")

    b = u.T @ o_aug
    u_hat, singular_values, vh = jnp.linalg.svd(b, full_matrices=False)
    u_final = u @ u_hat
    return u_final[:, :rank], singular_values[:rank], vh[:rank, :], rank


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
    constrain_update_norm: bool = True,
    eps: chex.Numeric = 1e-12,
) -> WSSRSVDResult:
    """Compute one warm-start subspace-iteration WSSR core update."""
    if int(state.sr_rank) == 0 or sr_rank_max == 0:
        zero_state = _zero_warm_svd_history_like(state)
        return WSSRSVDResult(jnp.zeros(o_aug.shape[0], dtype=o_aug.dtype), zero_state, 0)

    u, singular_values, vh, rank = warm_start_svd(
        o_aug,
        state,
        key,
        svd_maxiter_initial,
        svd_maxiter_warm,
    )
    if rank == 0:
        zero_state = _zero_warm_svd_history_like(state)
        return WSSRSVDResult(jnp.zeros(o_aug.shape[0], dtype=o_aug.dtype), zero_state, 0)

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
        eps=eps,
    )
    u_state = jnp.zeros_like(state.u)
    u_state = u_state.at[:, :rank].set(u[:, :rank])
    warm_state = WSSRWarmSVDCoreState(
        sr_o=result.state.sr_o,
        ek=result.state.ek,
        sr_rank0=result.state.sr_rank0,
        sr_rank=result.state.sr_rank,
        u=u_state,
        has_u=jnp.array(True),
    )
    return WSSRSVDResult(result.grad_like_update, warm_state, result.active_rank)


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
    result = wssr_sketch_core_update(
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
    constrain_update_norm: bool = True,
) -> Tuple[P, WSSRWarmSVDCoreState, int]:
    """Compute a warm-start SVD WSSR core update and unflatten it."""
    o_cur, unravel_fn = center_and_scale_score_matrix(
        log_psi_apply, params, positions
    )
    e_cur = center_and_scale_energy_residuals(local_energies, energy)
    o_aug, e_aug = augment_wssr_system(o_cur, e_cur, state, eta)
    result = wssr_warm_svd_core_update(
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
        constrain_update_norm=constrain_update_norm,
    )
    return unravel_fn(result.grad_like_update), result.state, result.active_rank


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

        grad_like_update, core_state, _ = compute_wssr_svd_core_update(
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

        grad_like_update, core_state, _ = compute_wssr_sketch_core_update(
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

        grad_like_update, core_state, _ = compute_wssr_warm_svd_core_update(
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

    flat_params, _ = jax.flatten_util.ravel_pytree(params)
    core_state = initialize_wssr_warm_svd_core_state(
        flat_params.shape[0],
        optimizer_config.sr_rank,
        optimizer_config.sr_rank_max,
        dtype=flat_params.dtype,
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
