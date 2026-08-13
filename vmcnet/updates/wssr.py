"""WSSR low-rank stochastic reconfiguration helpers."""
from typing import NamedTuple, Optional, Tuple

import chex
import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
from ml_collections import ConfigDict
import numpy as np
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

from .update_param_fns import (
    UpdateParamFn,
    constrain_update_function_norm,
    update_metrics_with_noclip,
)
from . import wssr_experimental


def _mixed_precision_rank_filter_callback(
    history_coefficients,
    current_coefficients,
    singular_values,
    relative_cutoff,
    lambda_reg,
):
    """Apply the cancellation-prone rank-space Tikhonov solve in fp64."""
    history64 = np.asarray(history_coefficients, dtype=np.float64)
    current64 = np.asarray(current_coefficients, dtype=np.float64)
    singular64 = np.asarray(singular_values, dtype=np.float64)
    cutoff64 = float(np.asarray(relative_cutoff, dtype=np.float64))
    lambda64 = float(np.asarray(lambda_reg, dtype=np.float64))
    leading = max(float(singular64[0]), 1e-12)
    retained = singular64 / leading > cutoff64
    coefficients = np.where(
        retained,
        (history64 + current64) / (np.square(singular64) + lambda64),
        0.0,
    )
    if not np.all(np.isfinite(coefficients)):
        raise FloatingPointError("WSSR mixed-precision rank solve became non-finite")
    return np.asarray(coefficients, dtype=np.float32)


def _mixed_precision_naive_solution_recurrence_callback(
    current_coefficients,
    singular_values,
    relative_cutoff,
    lambda_reg,
):
    """Resolve the current-batch correction for naive solution recurrence."""
    current64 = np.asarray(current_coefficients, dtype=np.float64)
    singular64 = np.asarray(singular_values, dtype=np.float64)
    cutoff64 = float(np.asarray(relative_cutoff, dtype=np.float64))
    lambda64 = float(np.asarray(lambda_reg, dtype=np.float64))
    leading = max(float(singular64[0]), 1e-12)
    retained = singular64 / leading > cutoff64
    correction = np.where(
        retained,
        current64 / (np.square(singular64) + lambda64),
        0.0,
    )
    if not np.all(np.isfinite(correction)):
        raise FloatingPointError(
            "WSSR naive solution recurrence became non-finite"
        )
    return np.asarray(correction, dtype=np.float32)


def _mixed_precision_residual_solution_recurrence_callback(
    current_coefficients,
    prior_coefficients,
    singular_values,
    relative_cutoff,
    lambda_reg,
):
    """Resolve current-batch residual correction in fp64 rank coordinates."""
    current64 = np.asarray(current_coefficients, dtype=np.float64)
    prior64 = np.asarray(prior_coefficients, dtype=np.float64)
    singular64 = np.asarray(singular_values, dtype=np.float64)
    cutoff64 = float(np.asarray(relative_cutoff, dtype=np.float64))
    lambda64 = float(np.asarray(lambda_reg, dtype=np.float64))
    leading = max(float(singular64[0]), 1e-12)
    retained = singular64 / leading > cutoff64
    residual_coefficients = current64 - np.square(singular64) * prior64
    correction = np.where(
        retained,
        residual_coefficients / (np.square(singular64) + lambda64),
        0.0,
    )
    if not np.all(np.isfinite(correction)):
        raise FloatingPointError(
            "WSSR residual solution recurrence became non-finite"
        )
    return np.asarray(correction, dtype=np.float32)


def _mixed_precision_galerkin_solution_callback(
    current_action,
    sample_residual,
    projected_error_feedback,
    lambda_reg,
):
    """Solve the current-batch Galerkin normal equation in host fp64.

    ``current_action`` is ``W = O_cur.T @ U_r`` in the code's transposed
    score convention. Only the small sample-by-rank matrix crosses the host
    boundary; the parameter-space basis remains on device.
    """
    output_dtype = np.asarray(current_action).dtype
    action64 = np.asarray(current_action, dtype=np.float64)
    residual64 = np.asarray(sample_residual, dtype=np.float64)
    feedback64 = np.asarray(projected_error_feedback, dtype=np.float64)
    lambda64 = float(np.asarray(lambda_reg, dtype=np.float64))
    gram = action64.T @ action64
    gram.flat[:: gram.shape[0] + 1] += lambda64
    rhs = action64.T @ residual64 + feedback64
    coefficients = np.linalg.solve(gram, rhs)
    if not np.all(np.isfinite(coefficients)):
        raise FloatingPointError(
            "WSSR full-current-batch Galerkin solve became non-finite"
        )
    # The production path supplies fp32 arrays, while x64 diagnostics and unit
    # tests legitimately request an fp64 callback result.  Preserve the caller
    # dtype so it matches the JAX ShapeDtypeStruct contract in both cases.
    return np.asarray(coefficients, dtype=output_dtype)


def _device_galerkin_solution(
    current_action: Array,
    projected_residual: Array,
    projected_error_feedback: Array,
    lambda_reg: chex.Numeric,
) -> Array:
    """Solve the regularized Galerkin system entirely on the JAX device.

    Keeping Gram construction and the Cholesky solve in JAX avoids the
    synchronization and device-to-host transfer imposed by ``pure_callback``.
    The solve follows the caller dtype: standard training uses fp32, while an
    explicitly x64-enabled run uses fp64. JAX cannot safely introduce local
    fp64 operations inside the outer fp32 training ``jit`` without enabling
    x64 globally, which would change unrelated model computations.
    """
    gram = current_action.T @ current_action
    half = jnp.asarray(0.5, dtype=gram.dtype)
    gram = half * (gram + gram.T)
    width = gram.shape[0]
    regularized_gram = gram + jnp.asarray(
        lambda_reg, dtype=gram.dtype
    ) * jnp.eye(width, dtype=gram.dtype)
    rhs = projected_residual + projected_error_feedback
    cholesky_factor = jnp.linalg.cholesky(regularized_gram)
    intermediate = jsp_linalg.solve_triangular(
        cholesky_factor, rhs, lower=True
    )
    return jsp_linalg.solve_triangular(
        cholesky_factor.T, intermediate, lower=False
    )


def bias_corrected_history_eta(target_eta: chex.Numeric, step):
    """Ramp history weight as min(target_eta, 1 - 1 / (step + 1))."""
    target = jnp.asarray(target_eta)
    one = jnp.asarray(1.0, dtype=target.dtype)
    iteration = jnp.asarray(step, dtype=target.dtype) + one
    return jnp.minimum(target, one - one / iteration)


def adaptive_s_averaging_eta(
    mode: str,
    eta_max: chex.Numeric,
    step,
    warmup_steps: int,
    tau: chex.Numeric,
):
    """Return the S-history weight for the requested adaptive schedule."""
    eta_max_array = jnp.asarray(eta_max)
    step_array = jnp.asarray(step, dtype=eta_max_array.dtype)
    if mode == "constant":
        return eta_max_array
    if mode == "linear_warmup":
        warmup = jnp.asarray(warmup_steps, dtype=eta_max_array.dtype)
        return eta_max_array * jnp.minimum(step_array / warmup, 1.0)
    if mode == "exponential_growth":
        tau_array = jnp.asarray(tau, dtype=eta_max_array.dtype)
        return eta_max_array * (1.0 - jnp.exp(-step_array / tau_array))
    raise ValueError(
        "eta_S_schedule must be constant, linear_warmup, or "
        "exponential_growth"
    )


def resolve_wssr_averaging_weights(optimizer_config: ConfigDict):
    """Resolve independent S and gradient averaging weights.

    Negative ``eta_S``/``eta_g`` values are sentinels for the legacy shared
    ``eta``.  This keeps old configuration files and command lines unchanged,
    while an explicitly configured split weight takes precedence. Adaptive
    S averaging instead resolves ``eta_S`` to its schedule maximum and defaults
    an unspecified gradient weight to zero (the current minibatch gradient).
    """
    legacy_eta = float(optimizer_config.get("eta", 0.99))
    adaptive_S_average = bool(
        optimizer_config.get("adaptive_S_average", False)
    )
    adaptive_g_average = bool(
        optimizer_config.get("adaptive_g_average", False)
    )
    eta_S = float(optimizer_config.get("eta_S", -1.0))
    eta_g = float(optimizer_config.get("eta_g", -1.0))
    if adaptive_S_average:
        # Adaptive WSSR uses the current stochastic gradient by default. An
        # explicitly supplied eta_g remains available for controlled ablations.
        eta_S = float(optimizer_config.get("eta_S_max", 0.95))
        eta_g = 0.0 if eta_g < 0.0 else eta_g
    else:
        eta_S = legacy_eta if eta_S < 0.0 else eta_S
        eta_g = legacy_eta if eta_g < 0.0 else eta_g
    if adaptive_g_average:
        eta_g = float(optimizer_config.get("eta_g_max", 0.2))
    for name, value in (("eta_S", eta_S), ("eta_g", eta_g)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    return eta_S, eta_g


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


class WSSRReducedMetricHistoryDiagnostics(NamedTuple):
    """Diagnostics for current-subspace reduced-metric averaging."""

    eta_by_mode: Array
    eta_mean: Array
    eta_head: Array
    eta_tail: Array
    noise_floor: Array
    history_overlap: Array
    cluster_count: Array
    drift_ratio_mean: Array


class WSSRReducedMetricHistoryResult(NamedTuple):
    """Update and telemetry from spectrum-only WSSR history."""

    grad_like_update: Array
    state: WSSRWarmSVDCoreState
    active_rank: Array
    diagnostics: WSSRReducedMetricHistoryDiagnostics


class WSSRAnisotropicMatrixHistoryDiagnostics(NamedTuple):
    """Diagnostics for spectral-position weights on the legacy S history."""

    eta_by_mode: Array
    eta_mean: Array
    eta_head: Array
    eta_tail: Array
    eta_spectral_mean: Array
    current_weight: Array
    noise_floor: Array


class WSSRAnisotropicMatrixHistoryResult(NamedTuple):
    """Update and telemetry from anisotropically weighted matrix history."""

    grad_like_update: Array
    state: WSSRWarmSVDCoreState
    active_rank: Array
    diagnostics: WSSRAnisotropicMatrixHistoryDiagnostics


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


class WSSRTransportedGradientOptimizerState(NamedTuple):
    """WSSR state with full-parameter gradient memory and optional transport."""

    core_state: WSSRCoreState
    optax_state: optax.OptState
    transported_gradient: Array
    previous_theta: Array
    transport_initialized: Array


class WSSRSolutionRecurrenceOptimizerState(NamedTuple):
    """WSSR state with an uncompressed full-parameter solution history."""

    core_state: WSSRCoreState
    optax_state: optax.OptState
    solution_state: Array


class WSSRErrorFeedbackRecurrenceOptimizerState(NamedTuple):
    """Solution recurrence plus an uncompressed error-feedback accumulator."""

    core_state: WSSRCoreState
    optax_state: optax.OptState
    solution_state: Array
    error_feedback_state: Array
    error_feedback_clip_count: Array


class WSSRClusterEnvelopeOptimizerState(NamedTuple):
    """WSSR state with a short, uncompressed history of cluster bases."""

    core_state: WSSRCoreState
    optax_state: optax.OptState
    envelope_history: Array
    envelope_count: Array


class WSSRClusterEnvelopeEFOptimizerState(NamedTuple):
    """Cluster-envelope state with decayed parameter-force error feedback."""

    core_state: WSSRCoreState
    optax_state: optax.OptState
    envelope_history: Array
    envelope_count: Array
    error_feedback_state: Array
    error_feedback_clip_count: Array


class GalerkinRecurrenceResult(NamedTuple):
    """Galerkin correction, next feedback state, and scalar diagnostics."""

    direction: Array
    correction: Array
    error_feedback: Array
    error_feedback_clip_increment: Array
    sample_residual_norm: Array
    correctable_residual_norm: Array
    correctable_residual_ratio: Array
    captured_sample_residual_ratio: Array
    residual_norm_reduction: Array
    residual_reduction_fraction: Array
    error_feedback_norm: Array
    error_feedback_uncapped_norm: Array
    error_feedback_cap_norm: Array
    error_feedback_to_sample_ratio: Array
    error_feedback_to_parameter_conflict_ratio: Array


class SolutionRecurrenceDiagnostics(NamedTuple):
    """State and telemetry returned by optional recurrence extensions."""

    error_feedback: Array
    error_feedback_clip_increment: Array
    sample_residual_norm: Array
    correctable_residual_norm: Array
    correctable_residual_ratio: Array
    captured_sample_residual_ratio: Array
    residual_norm_reduction: Array
    residual_reduction_fraction: Array
    error_feedback_norm: Array
    error_feedback_uncapped_norm: Array
    error_feedback_cap_norm: Array
    error_feedback_to_sample_ratio: Array
    error_feedback_to_parameter_conflict_ratio: Array
    correction_relative_difference: Array


class ExactResidualCaptureDiagnostics(NamedTuple):
    """Dimensionally distinct sample- and parameter-space capture metrics."""

    sample_projection_ratio: Array
    parameter_projection_ratio: Array
    applied_residual_norm_reduction: Array


class WSSRComplementEMAOptimizerState(NamedTuple):
    """WSSR state with an experimental EMA of the adaptive complement."""

    core_state: WSSRCoreState
    optax_state: optax.OptState
    complement_ema: Array


class WSSRMultilevelComplementOptimizerState(NamedTuple):
    """WSSR state for gated, low-frequency, two-buffer complement updates."""

    core_state: WSSRCoreState
    optax_state: optax.OptState
    # Keep this compatibility marker for checkpoint-state migration.
    complement_ema: Array
    complement_ema_b: Array
    complement_weight_a: Array
    complement_weight_b: Array
    complement_step: Array


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


def explicit_current_augmented_matmat(
    o_cur: Array,
    state: WSSRCoreState,
    eta: chex.Numeric,
    aug_matrix: Array,
    subspace_only_averaging: bool = False,
) -> Array:
    """Apply the augmented factor while keeping its two blocks separate.

    Unlike :func:`augment_wssr_system`, this helper never allocates the
    ``num_params x (history_rank + num_samples)`` concatenated matrix.  The
    current score block is still explicit, which avoids the repeated JVP/VJP
    cost of the fully matrix-free prototype.
    """
    history_width = state.sr_o.shape[1]
    v_hist = aug_matrix[:history_width, :]
    v_cur = aug_matrix[history_width:, :]
    history_mask = (jnp.arange(history_width) < state.sr_rank0).astype(
        o_cur.dtype
    )
    has_history = state.sr_rank0 > 0
    history_scale = jnp.sqrt(jnp.asarray(eta, dtype=o_cur.dtype))
    if subspace_only_averaging:
        current_scale = jnp.asarray(1.0, dtype=o_cur.dtype)
    else:
        current_scale = jnp.where(
            has_history,
            jnp.sqrt(1.0 - jnp.asarray(eta, dtype=o_cur.dtype)),
            1.0,
        )
    history_term = history_scale * (
        state.sr_o @ (history_mask[:, None] * v_hist)
    )
    return history_term + current_scale * (o_cur @ v_cur)


def explicit_current_augmented_rmatmat(
    o_cur: Array,
    state: WSSRCoreState,
    eta: chex.Numeric,
    param_matrix: Array,
    subspace_only_averaging: bool = False,
) -> Array:
    """Apply the transpose of the blockwise augmented factor."""
    history_width = state.sr_o.shape[1]
    history_mask = (jnp.arange(history_width) < state.sr_rank0).astype(
        o_cur.dtype
    )
    has_history = state.sr_rank0 > 0
    history_scale = jnp.sqrt(jnp.asarray(eta, dtype=o_cur.dtype))
    if subspace_only_averaging:
        current_scale = jnp.asarray(1.0, dtype=o_cur.dtype)
    else:
        current_scale = jnp.where(
            has_history,
            jnp.sqrt(1.0 - jnp.asarray(eta, dtype=o_cur.dtype)),
            1.0,
        )
    history_term = (
        history_scale
        * history_mask[:, None]
        * (state.sr_o.T @ param_matrix)
    )
    current_term = current_scale * (o_cur.T @ param_matrix)
    return jnp.concatenate([history_term, current_term], axis=0)


def explicit_current_augmented_matvec(
    o_cur: Array,
    state: WSSRCoreState,
    eta: chex.Numeric,
    aug_vector: Array,
    subspace_only_averaging: bool = False,
) -> Array:
    """Vector analogue of :func:`explicit_current_augmented_matmat`."""
    return explicit_current_augmented_matmat(
        o_cur,
        state,
        eta,
        aug_vector[:, None],
        subspace_only_averaging=subspace_only_averaging,
    )[:, 0]


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


def augment_wssr_subspace_residuals(
    e_cur: Array,
    state: WSSRCoreState,
    eta: chex.Numeric,
) -> Array:
    """Residual companion for subspace-only augmentation without ``O_aug``."""
    history_width = state.sr_o.shape[1]
    history_mask = (jnp.arange(history_width) < state.sr_rank0).astype(
        e_cur.dtype
    )
    e_hist = (
        jnp.sqrt(jnp.asarray(eta, dtype=e_cur.dtype))
        * state.ek
        * history_mask
    )
    return jnp.concatenate([e_hist, e_cur], axis=0)


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


def augment_wssr_subspace_system(
    o_cur: Array,
    e_cur: Array,
    state: WSSRCoreState,
    eta: chex.Numeric,
) -> Tuple[Array, Array]:
    """Augment only the SSI subspace, leaving the current batch unscaled.

    This realizes the P3 factor ``[sqrt(eta) A_(t-1), O_bar_t.T]``.  Unlike
    the standard averaged-S WSSR factor, there is deliberately no
    ``sqrt(1-eta)`` multiplier on the current block.  The returned residual is
    present only because the shared SVD state updater has a paired interface;
    the Galerkin residual and solve are evaluated separately on ``o_cur`` and
    ``e_cur``.
    """
    history_width = state.sr_o.shape[1]
    history_mask = (jnp.arange(history_width) < state.sr_rank0).astype(
        o_cur.dtype
    )
    sqrt_eta = jnp.sqrt(jnp.asarray(eta, dtype=o_cur.dtype))
    o_hist = sqrt_eta * state.sr_o * history_mask
    e_hist = sqrt_eta * state.ek * history_mask
    return jnp.concatenate([o_hist, o_cur], axis=1), jnp.concatenate(
        [e_hist, e_cur], axis=0
    )


def apply_wssr_history_operator(state: WSSRCoreState, vector: Array) -> Array:
    """Apply the previous low-rank WSSR operator without forming it densely."""
    history_width = state.sr_o.shape[1]
    history_mask = (jnp.arange(history_width) < state.sr_rank0).astype(
        state.sr_o.dtype
    )
    coefficients = state.sr_o.T @ vector
    return state.sr_o @ (history_mask * coefficients)


def transport_gradient_memory(
    previous_gradient: Array,
    current_gradient: Array,
    operator_delta: Array,
    eta_g: chex.Numeric,
    transport_initialized: Array,
) -> Array:
    """Update gradient EMA after transporting its history to current parameters."""
    # Transport historical gradients from previous parameter points to current
    # parameter point using first-order Taylor correction. WSSR stores a positive
    # gradient-like RHS, so the Taylor correction is +S @ (theta_t-theta_{t-1}).
    transported_history = previous_gradient + operator_delta
    transported_ema = (
        eta_g * transported_history + (1.0 - eta_g) * current_gradient
    )
    return jnp.where(transport_initialized, transported_ema, current_gradient)


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


def exact_residual_capture_diagnostics(
    o_cur: Array,
    sample_residual: Array,
    basis: Array,
    applied_coefficients: Array,
    parameter_target: Optional[Array] = None,
    relative_singular_value_cutoff: chex.Numeric = 1e-7,
    eps: chex.Numeric = 1e-12,
) -> ExactResidualCaptureDiagnostics:
    """Measure residual capture without conflating sample and parameter spaces.

    ``o_cur`` stores ``O_bar.T`` and ``W = O_bar @ basis``.  The exact sample
    metric uses the orthogonal projector onto ``col(W)``.  The parameter metric
    separately projects ``parameter_target`` onto ``span(basis)``.  Callers can
    therefore audit, for example, either a full current-batch update or the
    detected conflict.  When no target is supplied, the detected conflict
    ``O_bar.T @ zeta`` is used.  These two ratios are not generally equal.  This
    SVD-based helper is intended for audits and unit tests, not the hot training
    path.
    """
    current_action = o_cur.T @ basis
    left_vectors, singular_values, _ = jnp.linalg.svd(
        current_action, full_matrices=False
    )
    leading = jnp.maximum(
        singular_values[0], jnp.asarray(eps, dtype=o_cur.dtype)
    )
    retained = singular_values > (
        jnp.asarray(relative_singular_value_cutoff, dtype=o_cur.dtype) * leading
    )
    sample_projection = left_vectors @ (
        retained.astype(o_cur.dtype) * (left_vectors.T @ sample_residual)
    )
    sample_norm = jnp.linalg.norm(sample_residual)
    sample_projection_ratio = jnp.linalg.norm(sample_projection) / jnp.maximum(
        sample_norm, jnp.asarray(eps, dtype=o_cur.dtype)
    )

    if parameter_target is None:
        parameter_target = o_cur @ sample_residual
    parameter_projection = basis @ (basis.T @ parameter_target)
    parameter_projection_ratio = jnp.linalg.norm(
        parameter_projection
    ) / jnp.maximum(
        jnp.linalg.norm(parameter_target),
        jnp.asarray(eps, dtype=o_cur.dtype),
    )

    applied_residual = sample_residual - current_action @ applied_coefficients
    applied_residual_norm_reduction = jnp.clip(
        1.0
        - jnp.linalg.norm(applied_residual)
        / jnp.maximum(sample_norm, jnp.asarray(eps, dtype=o_cur.dtype)),
        0.0,
        1.0,
    )
    return ExactResidualCaptureDiagnostics(
        sample_projection_ratio=sample_projection_ratio,
        parameter_projection_ratio=parameter_projection_ratio,
        applied_residual_norm_reduction=applied_residual_norm_reduction,
    )


def galerkin_residual_solution_recurrence(
    o_cur: Array,
    e_cur: Array,
    basis: Array,
    solution_prior: Array,
    lambda_reg: chex.Numeric,
    error_feedback: Optional[Array] = None,
    enable_error_feedback: bool = False,
    error_feedback_norm_cap: chex.Numeric = 10.0,
    error_feedback_decay: chex.Numeric = 1.0,
    error_feedback_cap_reference: str = "correction",
    current_action: Optional[Array] = None,
    solve_backend: str = "host_fp64",
    eps: chex.Numeric = 1e-12,
) -> GalerkinRecurrenceResult:
    """Apply a Galerkin-consistent current-batch residual correction.

    The code stores the centered score as ``o_cur = O_bar.T`` with shape
    ``(num_params, num_samples)``. Consequently the mathematical operations
    ``O_bar @ prior`` and ``O_bar @ U_r`` are implemented as
    ``o_cur.T @ prior`` and ``o_cur.T @ basis`` respectively.
    """
    if solution_prior is None:
        raise ValueError("Galerkin recurrence requires solution_prior")
    if enable_error_feedback and error_feedback is None:
        raise ValueError("error feedback requires a parameter-space state")

    sample_residual = e_cur - o_cur.T @ solution_prior
    # The final right-subspace iteration already produces the equivalent
    # current-batch action. Reusing it avoids a second num-samples x
    # num-parameters x rank product. The explicit path remains available as a
    # reference and for callers that do not own an SSI cache.
    if current_action is None:
        current_action = o_cur.T @ basis
    projected_residual = current_action.T @ sample_residual
    if enable_error_feedback:
        decayed_feedback = jnp.asarray(
            error_feedback_decay, dtype=o_cur.dtype
        ) * error_feedback
        projected_feedback = basis.T @ decayed_feedback
    else:
        decayed_feedback = jnp.zeros_like(solution_prior)
        projected_feedback = jnp.zeros(
            (basis.shape[1],), dtype=o_cur.dtype
        )

    if solve_backend == "host_fp64":
        callback_shape = jax.ShapeDtypeStruct((basis.shape[1],), o_cur.dtype)
        coefficients = jax.pure_callback(
            _mixed_precision_galerkin_solution_callback,
            callback_shape,
            current_action,
            sample_residual,
            projected_feedback,
            jnp.asarray(lambda_reg, dtype=o_cur.dtype),
        )
    elif solve_backend == "device_cholesky":
        coefficients = _device_galerkin_solution(
            current_action,
            projected_residual,
            projected_feedback,
            lambda_reg,
        )
    else:
        raise ValueError(
            "solve_backend must be host_fp64 or device_cholesky"
        )
    correction = basis @ coefficients
    direction = solution_prior + correction
    captured_sample_residual = current_action @ coefficients
    corrected_sample_residual = sample_residual - captured_sample_residual
    sample_residual_squared = jnp.sum(jnp.square(sample_residual))
    corrected_residual_squared = jnp.sum(
        jnp.square(corrected_sample_residual)
    )
    residual_reduction_fraction = jnp.clip(
        1.0
        - corrected_residual_squared
        / jnp.maximum(sample_residual_squared, jnp.asarray(eps, o_cur.dtype)),
        0.0,
        1.0,
    )
    sample_residual_norm = jnp.sqrt(sample_residual_squared)
    captured_sample_residual_ratio = jnp.linalg.norm(
        captured_sample_residual
    ) / jnp.maximum(sample_residual_norm, jnp.asarray(eps, o_cur.dtype))
    residual_norm_reduction = jnp.clip(
        1.0
        - jnp.sqrt(corrected_residual_squared)
        / jnp.maximum(sample_residual_norm, jnp.asarray(eps, o_cur.dtype)),
        0.0,
        1.0,
    )

    parameter_conflict = o_cur @ sample_residual
    parameter_conflict_norm = jnp.linalg.norm(parameter_conflict)
    if enable_error_feedback:
        detected_conflict = parameter_conflict + decayed_feedback
        next_feedback = detected_conflict - basis @ (basis.T @ detected_conflict)
        feedback_norm_uncapped = jnp.linalg.norm(next_feedback)
        correction_norm = jnp.linalg.norm(correction)
        if error_feedback_cap_reference == "correction":
            cap_reference_norm = correction_norm
        elif error_feedback_cap_reference == "sample_residual":
            cap_reference_norm = sample_residual_norm
        elif error_feedback_cap_reference == "parameter_conflict":
            cap_reference_norm = parameter_conflict_norm
        else:
            raise ValueError(
                "error_feedback_cap_reference must be correction, "
                "sample_residual, or parameter_conflict"
            )
        maximum_feedback_norm = jnp.asarray(
            error_feedback_norm_cap, dtype=o_cur.dtype
        ) * cap_reference_norm
        feedback_scale = jnp.minimum(
            1.0,
            maximum_feedback_norm
            / jnp.maximum(feedback_norm_uncapped, jnp.asarray(eps, o_cur.dtype)),
        )
        next_feedback = next_feedback * feedback_scale
        clip_increment = (
            feedback_norm_uncapped > maximum_feedback_norm
        ).astype(jnp.int32)
    else:
        next_feedback = jnp.zeros_like(solution_prior)
        feedback_norm_uncapped = jnp.asarray(0.0, dtype=o_cur.dtype)
        maximum_feedback_norm = jnp.asarray(0.0, dtype=o_cur.dtype)
        clip_increment = jnp.asarray(0, dtype=jnp.int32)

    error_feedback_norm = jnp.linalg.norm(next_feedback)
    error_feedback_to_sample_ratio = error_feedback_norm / jnp.maximum(
        sample_residual_norm, jnp.asarray(eps, dtype=o_cur.dtype)
    )
    error_feedback_to_parameter_conflict_ratio = (
        error_feedback_norm
        / jnp.maximum(
            parameter_conflict_norm,
            jnp.asarray(eps, dtype=o_cur.dtype),
        )
    )

    correctable_residual_norm = jnp.linalg.norm(projected_residual)
    # Legacy action ratio retained for backward-compatible logs. It is not a
    # projection fraction and may exceed one; use the capture metrics above.
    correctable_residual_ratio = correctable_residual_norm / jnp.maximum(
        sample_residual_norm, jnp.asarray(eps, dtype=o_cur.dtype)
    )
    return GalerkinRecurrenceResult(
        direction=direction,
        correction=correction,
        error_feedback=next_feedback,
        error_feedback_clip_increment=clip_increment,
        sample_residual_norm=sample_residual_norm,
        correctable_residual_norm=correctable_residual_norm,
        correctable_residual_ratio=correctable_residual_ratio,
        captured_sample_residual_ratio=captured_sample_residual_ratio,
        residual_norm_reduction=residual_norm_reduction,
        residual_reduction_fraction=residual_reduction_fraction,
        error_feedback_norm=error_feedback_norm,
        error_feedback_uncapped_norm=feedback_norm_uncapped,
        error_feedback_cap_norm=maximum_feedback_norm,
        error_feedback_to_sample_ratio=error_feedback_to_sample_ratio,
        error_feedback_to_parameter_conflict_ratio=(
            error_feedback_to_parameter_conflict_ratio
        ),
    )


def cached_current_action_from_svd(
    vh: Array,
    singular_values: Array,
    active_rank: Array,
    storage_width: int,
    current_sample_width: int,
    current_block_scale: chex.Numeric = 1.0,
    eps: chex.Numeric = 1e-12,
) -> Array:
    """Recover ``O_current.T @ U`` from the final SSI factors.

    For the approximate SVD ``O_aug = U diag(s) V.T``, the last current-batch
    rows of ``V diag(s)`` equal the current block applied to ``U``.  Dividing
    by the augmentation scale yields the unscaled current-batch action used by
    the Galerkin residual solve.  No parameter-by-rank product is repeated.
    """
    rank_width = singular_values.shape[0]
    if current_sample_width <= 0:
        return jnp.zeros((0, storage_width), dtype=vh.dtype)
    safe_scale = jnp.maximum(
        jnp.asarray(current_block_scale, dtype=vh.dtype),
        jnp.asarray(eps, dtype=vh.dtype),
    )
    retained = (
        jnp.arange(rank_width) < jnp.asarray(active_rank)
    ).astype(vh.dtype)
    action_values = (
        vh[:, -current_sample_width:].T
        * singular_values[None, :]
        * retained[None, :]
        / safe_scale
    )
    action = jnp.zeros(
        (current_sample_width, storage_width), dtype=vh.dtype
    )
    copy_width = min(rank_width, storage_width)
    return action.at[:, :copy_width].set(action_values[:, :copy_width])


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
    tikhonov_lambda: chex.Numeric = -1.0,
) -> Tuple[Array, Array]:
    """Return range and complement inverse coefficients for the WSSR update."""
    _validate_wssr_spectral_regularization(
        spectral_regularization, complement_weight
    )
    legacy_lambda = jnp.square(safe_damping * jnp.abs(safe_leading_sv))
    configured_lambda = jnp.asarray(
        tikhonov_lambda, dtype=safe_singular_values.dtype
    )
    sigma_floor = jnp.where(
        configured_lambda >= 0.0, configured_lambda, legacy_lambda
    )
    sigma_floor = jnp.maximum(
        sigma_floor, jnp.asarray(jnp.finfo(safe_singular_values.dtype).tiny)
    )
    inv_floor = 1.0 / sigma_floor
    if spectral_regularization == "hard_floor":
        inv_cap = jnp.square(1.0 / safe_singular_values)
    else:
        inv_cap = 1.0 / (jnp.square(safe_singular_values) + sigma_floor)
    inv_perp = jnp.asarray(complement_weight, dtype=safe_singular_values.dtype)
    inv_perp = inv_perp * inv_floor
    return inv_cap, inv_perp


def _wssr_resolve_spectral_controls(
    safe_leading_sv: Array,
    damping: chex.Numeric,
    relative_singular_value_cutoff: chex.Numeric = -1.0,
    tikhonov_lambda: chex.Numeric = -1.0,
    eps: chex.Numeric = 1e-12,
) -> Tuple[Array, Array]:
    """Resolve backward-compatible cutoff and Tikhonov lambda controls."""
    safe_damping = jnp.maximum(jnp.abs(damping), eps)
    configured_cutoff = jnp.asarray(
        relative_singular_value_cutoff, dtype=safe_leading_sv.dtype
    )
    cutoff = jnp.where(
        configured_cutoff >= 0.0, configured_cutoff, safe_damping
    )
    legacy_lambda = jnp.square(safe_damping * jnp.abs(safe_leading_sv))
    configured_lambda = jnp.asarray(
        tikhonov_lambda, dtype=safe_leading_sv.dtype
    )
    lambda_reg = jnp.where(
        configured_lambda >= 0.0, configured_lambda, legacy_lambda
    )
    lambda_reg = jnp.maximum(lambda_reg, eps)
    return cutoff, lambda_reg


_WSSR_REDUCED_METRIC_HISTORY_MODES = (
    "none",
    "uniform_spectrum",
    "cluster_adaptive",
)


def spectral_position_history_weights(
    history_eigenvalues: Array,
    active_mask: Array,
    has_history: Array,
    eta_max: chex.Numeric,
    num_samples: int,
    noise_scale: chex.Numeric = 1.0,
    eps: chex.Numeric = 1e-12,
) -> WSSRAnisotropicMatrixHistoryDiagnostics:
    """Return suggestion-1 weights for the legacy matrix-history factor.

    The columns of the ordinary WSSR history factor are ordered approximate
    eigenmodes, with squared column norms equal to their retained eigenvalues.
    This rule keeps that augmented-factor representation and replaces its one
    scalar history weight by ``delta_i``.  The SNR proxy is
    ``lambda_i / (lambda_1 / sqrt(N_s))``; consequently reliable leading modes
    use little history and noisy tail modes use more.

    A single scalar current-block weight is still needed because sample columns
    do not have a one-to-one correspondence with parameter-space eigenmodes.
    We use the spectral-mass-weighted mean history weight. It exactly recovers
    ``1-eta`` when all history weights are uniform and preserves aggregate
    trace scaling when consecutive spectra are comparable.
    """
    if not 0.0 <= eta_max < 1.0:
        raise ValueError("anisotropic matrix-history eta_max must be in [0, 1)")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if noise_scale <= 0.0:
        raise ValueError("noise_scale must be positive")

    dtype = history_eigenvalues.dtype
    eps_array = jnp.asarray(eps, dtype=dtype)
    active = active_mask.astype(bool)
    active_float = active.astype(dtype)
    active_count = jnp.sum(active.astype(jnp.int32))
    usable_history = jnp.asarray(has_history) & (active_count > 0)
    eigenvalues = jnp.where(
        active, jnp.maximum(history_eigenvalues, 0.0), 0.0
    )
    leading = jnp.maximum(eigenvalues[0], eps_array)
    estimated_noise = (
        jnp.asarray(noise_scale, dtype=dtype)
        * leading
        / jnp.sqrt(jnp.asarray(num_samples, dtype=dtype))
    )
    mode_snr = eigenvalues / jnp.maximum(estimated_noise, eps_array)
    eta_by_mode = (
        jnp.asarray(eta_max, dtype=dtype)
        / (1.0 + mode_snr)
        * active_float
        * usable_history.astype(dtype)
    )

    def _masked_mean(values, mask):
        mask_float = mask.astype(dtype)
        return jnp.sum(values * mask_float) / jnp.maximum(
            jnp.sum(mask_float), jnp.asarray(1.0, dtype=dtype)
        )

    spectral_mass = jnp.sum(eigenvalues)
    eta_spectral_mean = jnp.sum(eta_by_mode * eigenvalues) / jnp.maximum(
        spectral_mass, eps_array
    )
    eta_spectral_mean = jnp.where(
        usable_history, eta_spectral_mean, jnp.asarray(0.0, dtype=dtype)
    )
    quartile = jnp.maximum(active_count // 4, 1)
    indices = jnp.arange(history_eigenvalues.shape[0], dtype=jnp.int32)
    head_mask = active & (indices < quartile)
    tail_mask = active & (indices >= jnp.maximum(active_count - quartile, 0))
    return WSSRAnisotropicMatrixHistoryDiagnostics(
        eta_by_mode=eta_by_mode,
        eta_mean=_masked_mean(eta_by_mode, active),
        eta_head=_masked_mean(eta_by_mode, head_mask),
        eta_tail=_masked_mean(eta_by_mode, tail_mask),
        eta_spectral_mean=eta_spectral_mean,
        current_weight=jnp.where(
            usable_history,
            1.0 - eta_spectral_mean,
            jnp.asarray(1.0, dtype=dtype),
        ),
        noise_floor=jnp.where(
            usable_history,
            estimated_noise,
            jnp.asarray(0.0, dtype=dtype),
        ),
    )


def augment_wssr_anisotropic_matrix_history(
    o_cur: Array,
    state: WSSRWarmSVDCoreState,
    diagnostics: WSSRAnisotropicMatrixHistoryDiagnostics,
) -> Array:
    """Apply per-mode history weights to the legacy augmented factor."""
    history_width = state.sr_o.shape[1]
    history_mask = (
        jnp.arange(history_width) < state.sr_rank0
    ).astype(o_cur.dtype)
    weighted_history = state.sr_o * jnp.sqrt(
        diagnostics.eta_by_mode * history_mask
    )[None, :]
    weighted_current = o_cur * jnp.sqrt(diagnostics.current_weight)
    return jnp.concatenate([weighted_history, weighted_current], axis=1)


def average_current_reduced_metric(
    current_eigenvalues: Array,
    history_metric_in_current_basis: Array,
    active_mask: Array,
    has_history: Array,
    mode: str,
    eta_max: chex.Numeric,
    num_samples: int,
    cluster_gap_threshold: chex.Numeric = 0.01,
    noise_scale: chex.Numeric = 1.0,
    drift_scale: chex.Numeric = 1.0,
    history_total_mass: Optional[Array] = None,
    eps: chex.Numeric = 1e-12,
) -> Tuple[Array, WSSRReducedMetricHistoryDiagnostics]:
    """Average curvature history inside the *current* SSI subspace.

    ``current_eigenvalues`` and ``history_metric_in_current_basis`` are both
    expressed in the current left-singular basis.  History therefore changes
    only the reduced metric used by the solve; it never participates in the
    construction of the current basis itself.

    ``uniform_spectrum`` averages only diagonal Ritz values.  The
    ``cluster_adaptive`` mode retains complete blocks inside near-degenerate
    spectral clusters and assigns one history weight to every block.  Its
    weight decreases when the current eigenvalue has high sample-size-scaled
    SNR or when the historical block disagrees with the current block by more
    than the estimated Monte Carlo noise floor.  Sharing the weight across a
    cluster makes the rule invariant to rotations inside that cluster.
    """
    if mode not in _WSSR_REDUCED_METRIC_HISTORY_MODES:
        raise ValueError(
            "reduced metric history mode must be one of "
            f"{_WSSR_REDUCED_METRIC_HISTORY_MODES}"
        )
    if not 0.0 <= eta_max < 1.0:
        raise ValueError("reduced metric eta_max must be in [0, 1)")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if cluster_gap_threshold < 0.0:
        raise ValueError("cluster_gap_threshold must be nonnegative")
    if noise_scale <= 0.0:
        raise ValueError("noise_scale must be positive")
    if drift_scale < 0.0:
        raise ValueError("drift_scale must be nonnegative")

    dtype = current_eigenvalues.dtype
    eps_array = jnp.asarray(eps, dtype=dtype)
    active = active_mask.astype(bool)
    active_float = active.astype(dtype)
    active_count = jnp.sum(active.astype(jnp.int32))
    has_usable_history = jnp.asarray(has_history) & (active_count > 0)
    current_eigenvalues = jnp.where(
        active, jnp.maximum(current_eigenvalues, 0.0), 0.0
    )
    current_metric = jnp.diag(current_eigenvalues)
    history_metric = 0.5 * (
        history_metric_in_current_basis
        + history_metric_in_current_basis.T
    )

    leading = jnp.maximum(current_eigenvalues[0], eps_array)
    noise_floor = (
        jnp.asarray(noise_scale, dtype=dtype)
        * leading
        / jnp.sqrt(jnp.asarray(num_samples, dtype=dtype))
    )

    if current_eigenvalues.shape[0] > 1:
        adjacent_scale = jnp.maximum(
            current_eigenvalues[:-1], eps_array
        )
        relative_gaps = (
            current_eigenvalues[:-1] - current_eigenvalues[1:]
        ) / adjacent_scale
        boundaries = (
            (relative_gaps > jnp.asarray(cluster_gap_threshold, dtype=dtype))
            & active[:-1]
            & active[1:]
        )
        cluster_ids = jnp.concatenate(
            [
                jnp.zeros((1,), dtype=jnp.int32),
                jnp.cumsum(boundaries.astype(jnp.int32)),
            ]
        )
    else:
        cluster_ids = jnp.zeros((1,), dtype=jnp.int32)
    same_cluster = (
        (cluster_ids[:, None] == cluster_ids[None, :])
        & active[:, None]
        & active[None, :]
    )
    cluster_members = same_cluster.astype(dtype)
    cluster_sizes = jnp.maximum(
        jnp.sum(cluster_members, axis=1), jnp.asarray(1.0, dtype=dtype)
    )

    eta_target = jnp.asarray(eta_max, dtype=dtype)
    if mode == "none":
        eta_by_mode = jnp.zeros_like(current_eigenvalues)
        averaged_metric = current_metric
        drift_ratio = jnp.zeros_like(current_eigenvalues)
    elif mode == "uniform_spectrum":
        eta_by_mode = (
            eta_target
            * active_float
            * has_usable_history.astype(dtype)
        )
        history_eigenvalues = jnp.maximum(jnp.diag(history_metric), 0.0)
        averaged_eigenvalues = (
            (1.0 - eta_by_mode) * current_eigenvalues
            + eta_by_mode * history_eigenvalues
        )
        averaged_metric = jnp.diag(averaged_eigenvalues)
        drift_ratio = jnp.abs(
            history_eigenvalues - current_eigenvalues
        ) / jnp.maximum(noise_floor, eps_array)
    else:
        # The sample-size-scaled Ritz value is a cheap per-mode SNR proxy.
        # High-SNR leading modes remain current; noisy tail modes receive more
        # history.  Cluster averaging avoids basis-dependent decisions inside
        # nearly degenerate eigenspaces.
        mode_snr = current_eigenvalues / jnp.maximum(noise_floor, eps_array)
        base_eta = eta_target / (1.0 + mode_snr)
        cluster_eta = (
            cluster_members @ base_eta
        ) / cluster_sizes

        block_history = jnp.where(same_cluster, history_metric, 0.0)
        block_difference = block_history - jnp.where(
            same_cluster, current_metric, 0.0
        )
        row_difference_energy = jnp.sum(jnp.square(block_difference), axis=1)
        cluster_difference_energy = (
            cluster_members @ row_difference_energy
        ) / cluster_sizes
        drift_ratio = jnp.sqrt(jnp.maximum(cluster_difference_energy, 0.0))
        drift_ratio = drift_ratio / jnp.maximum(noise_floor, eps_array)
        drift_gate = 1.0 / (
            1.0
            + jnp.square(jnp.asarray(drift_scale, dtype=dtype) * drift_ratio)
        )
        eta_by_mode = (
            cluster_eta
            * drift_gate
            * active_float
            * has_usable_history.astype(dtype)
        )
        # eta is constant within a cluster. The geometric mean below keeps the
        # expression symmetric even under roundoff at a cluster boundary.
        history_weights = jnp.sqrt(
            eta_by_mode[:, None] * eta_by_mode[None, :]
        )
        averaged_metric = jnp.diag(
            (1.0 - eta_by_mode) * current_eigenvalues
        ) + jnp.where(
            same_cluster, history_weights * history_metric, 0.0
        )
        averaged_metric = 0.5 * (averaged_metric + averaged_metric.T)

    def _masked_mean(values, mask):
        mask_float = mask.astype(dtype)
        return jnp.sum(values * mask_float) / jnp.maximum(
            jnp.sum(mask_float), jnp.asarray(1.0, dtype=dtype)
        )

    quartile = jnp.maximum(active_count // 4, 1)
    indices = jnp.arange(current_eigenvalues.shape[0], dtype=jnp.int32)
    head_mask = active & (indices < quartile)
    tail_mask = active & (indices >= jnp.maximum(active_count - quartile, 0))
    if history_total_mass is None:
        history_total_mass = jnp.trace(history_metric)
    history_overlap = jnp.trace(history_metric) / jnp.maximum(
        jnp.asarray(history_total_mass, dtype=dtype), eps_array
    )
    cluster_count = jnp.where(
        active_count > 0,
        jnp.max(jnp.where(active, cluster_ids, 0)) + 1,
        0,
    )
    diagnostics = WSSRReducedMetricHistoryDiagnostics(
        eta_by_mode=eta_by_mode,
        eta_mean=_masked_mean(eta_by_mode, active),
        eta_head=_masked_mean(eta_by_mode, head_mask),
        eta_tail=_masked_mean(eta_by_mode, tail_mask),
        noise_floor=noise_floor,
        history_overlap=history_overlap,
        cluster_count=cluster_count,
        drift_ratio_mean=_masked_mean(drift_ratio, active),
    )
    return averaged_metric, diagnostics


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
    o_aug: Optional[Array],
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
    relative_singular_value_cutoff: chex.Numeric = -1.0,
    tikhonov_lambda: chex.Numeric = -1.0,
    experimental_mode: str = "none",
    experimental_target_rank: int = -1,
    cluster_gap_threshold: chex.Numeric = 0.002,
    near_tail_modes: int = 0,
    adaptive_complement_beta: chex.Numeric = 0.0,
    adaptive_complement_beta_function: chex.Numeric = 0.0,
    complement_function_operator: Optional[Array] = None,
    smooth_transition_start: int = -1,
    smooth_transition_end: int = -1,
    iterative_complement_iterations: int = 0,
    mixed_precision_solve: bool = False,
    mixed_precision_history_width: int = 0,
    solution_prior: Optional[Array] = None,
    solution_recurrence_mode: str = "none",
    force_override: Optional[Array] = None,
    force_override_keeps_residual_history: bool = False,
    eps: chex.Numeric = 1e-12,
) -> WSSRSVDResult:
    """Apply the shared post-SVD WSSR update formula."""
    if rank_update_max is None:
        rank_update_max = sr_rank_max
    rank_update_max = min(rank_update_max, sr_rank_max)

    if singular_values.shape[0] == 0:
        zero_state = _zero_history_like(state)
        return WSSRSVDResult(
            jnp.zeros(u.shape[0], dtype=u.dtype),
            zero_state,
            jnp.asarray(0, dtype=state.sr_rank.dtype),
        )

    if o_aug is None and force_override is None:
        raise ValueError("blockwise WSSR update requires a precomputed force")
    if o_aug is None and experimental_mode not in ("none", "adaptive_complement"):
        raise ValueError(
            "blockwise WSSR update only supports the adaptive complement"
        )
    wssr_experimental.validate_mode(experimental_mode)
    leading_sv = singular_values[0]
    valid_leading = leading_sv > eps
    safe_leading_sv = jnp.where(valid_leading, leading_sv, 1.0)
    relative_cutoff, lambda_reg = _wssr_resolve_spectral_controls(
        safe_leading_sv,
        damping,
        relative_singular_value_cutoff,
        tikhonov_lambda,
        eps,
    )
    retained = (singular_values / safe_leading_sv > relative_cutoff) & valid_leading
    spectral_weights = jnp.ones_like(singular_values)
    if experimental_mode != "none":
        target = singular_values.shape[0] if experimental_target_rank <= 0 else min(
            experimental_target_rank, singular_values.shape[0]
        )
        effective_rank = singular_values.shape[0]
        if experimental_mode == "cluster":
            effective_rank = wssr_experimental.cluster_effective_rank(
                singular_values, target, cluster_gap_threshold,
                singular_values.shape[0]
            )
        elif experimental_mode in ("near_tail", "capped_near_tail"):
            effective_rank = min(target + near_tail_modes, singular_values.shape[0])
        elif experimental_mode == "smooth":
            effective_rank = singular_values.shape[0]
            spectral_weights = wssr_experimental.cosine_taper(
                singular_values.shape[0], smooth_transition_start,
                smooth_transition_end, singular_values.dtype
            )
        else:
            effective_rank = target
        retained = retained & (jnp.arange(singular_values.shape[0]) < effective_rank)
        if experimental_mode == "smooth":
            retained = retained & (spectral_weights > eps)
    active_rank = jnp.sum(retained.astype(state.sr_rank.dtype))
    has_active_rank = active_rank > 0
    valid_update = valid_leading & has_active_rank
    retained_float = retained.astype(u.dtype)

    safe_singular_values = jnp.where(retained, singular_values, 1.0)
    safe_damping = jnp.maximum(jnp.abs(damping), eps)
    inv_cap, inv_perp = _wssr_inverse_spectral_coefficients(
        safe_singular_values,
        safe_leading_sv,
        safe_damping,
        spectral_regularization,
        complement_weight,
        tikhonov_lambda=tikhonov_lambda,
    )
    if mixed_precision_solve:
        if spectral_regularization != "tikhonov" or complement_weight != 0.0:
            raise ValueError(
                "mixed_precision_solve requires Tikhonov with zero complement"
            )
        if experimental_mode != "none":
            raise ValueError(
                "mixed_precision_solve is only supported for experimental_mode=none"
            )
        if solution_recurrence_mode not in ("none", "naive", "residual"):
            raise ValueError(
                "solution_recurrence_mode must be none, naive, or residual"
            )
        if solution_recurrence_mode != "none" and solution_prior is None:
            raise ValueError(
                "solution recurrence requires a full-parameter solution_prior"
            )
        history_width = min(mixed_precision_history_width, e_aug.shape[0])
        if force_override is None:
            history_force = (
                o_aug[:, :history_width] @ e_aug[:history_width]
                if history_width > 0
                else jnp.zeros(o_aug.shape[0], dtype=o_aug.dtype)
            )
            current_force = o_aug[:, history_width:] @ e_aug[history_width:]
        else:
            # The transported-gradient path has already combined history and the
            # current gradient in parameter space. Keep the fp64 rank solve, but
            # do not add the legacy residual history a second time.
            history_force = jnp.zeros_like(force_override)
            current_force = force_override
        history_coefficients = u.T @ history_force
        current_coefficients = u.T @ current_force
        callback_shape = jax.ShapeDtypeStruct(
            singular_values.shape, u.dtype
        )
        if solution_recurrence_mode == "naive":
            correction_coefficients = jax.pure_callback(
                _mixed_precision_naive_solution_recurrence_callback,
                callback_shape,
                current_coefficients,
                singular_values,
                relative_cutoff,
                lambda_reg,
            )
            grad_like_update = solution_prior + u @ correction_coefficients
        elif solution_recurrence_mode == "residual":
            prior_coefficients = u.T @ solution_prior
            correction_coefficients = jax.pure_callback(
                _mixed_precision_residual_solution_recurrence_callback,
                callback_shape,
                current_coefficients,
                prior_coefficients,
                singular_values,
                relative_cutoff,
                lambda_reg,
            )
            grad_like_update = solution_prior + u @ correction_coefficients
        else:
            resolved_coefficients = jax.pure_callback(
                _mixed_precision_rank_filter_callback,
                callback_shape,
                history_coefficients,
                current_coefficients,
                singular_values,
                relative_cutoff,
                lambda_reg,
            )
            grad_like_update = u @ resolved_coefficients
    else:
        force = o_aug @ e_aug if force_override is None else force_override
        projected_force = u.T @ force
    if experimental_mode == "none" and not mixed_precision_solve:
        # Preserve the historical operation ordering bit-for-bit.
        projected_force = projected_force * (inv_cap - inv_perp)
        projected_force = projected_force * retained_float
        grad_like_update = u @ projected_force + inv_perp * force
    elif not mixed_precision_solve:
        resolved = u @ (projected_force * inv_cap * retained_float * spectral_weights)
        projected_resolved_force = u @ (projected_force * retained_float)
        perp_force = force - projected_resolved_force
        complement = jnp.zeros_like(force)
        if experimental_mode == "adaptive_complement":
            complement, _ = wssr_experimental.adaptive_complement(
                resolved,
                perp_force,
                inv_perp,
                adaptive_complement_beta,
                function_operator=complement_function_operator,
                beta_function=adaptive_complement_beta_function,
            )
        elif experimental_mode == "residual_optimal_complement":
            regularizer = lambda_reg
            complement, _, _, _, _, _ = (
                wssr_experimental.residual_optimal_complement(
                    o_aug, force, resolved, perp_force, regularizer,
                    adaptive_complement_beta, eps
                )
            )
        elif experimental_mode == "capped_near_tail":
            main = jnp.arange(singular_values.shape[0]) < target
            near = retained & ~main
            resolved = u @ (
                projected_force * inv_cap * retained_float
                * main.astype(o_aug.dtype)
            )
            near_update = u @ (
                projected_force * inv_cap * near.astype(o_aug.dtype)
            )
            complement, _ = wssr_experimental.cap_contribution(
                resolved, near_update, adaptive_complement_beta, eps
            )
        elif experimental_mode == "iterative_complement":
            active_u = u * retained_float[None, :]
            complement, _ = wssr_experimental.projected_cg(
                o_aug, active_u, perp_force,
                lambda_reg,
                iterative_complement_iterations,
            )
        grad_like_update = resolved + complement
    grad_like_update = jnp.where(valid_update, grad_like_update, 0.0)

    if constrain_update_norm:
        grad_like_update = constrain_norm(grad_like_update, norm_constraint, eps=eps)

    sr_o = jnp.zeros_like(state.sr_o)
    ek = jnp.zeros_like(state.ek)
    history_width = min(singular_values.shape[0], sr_rank_max)
    history_weights = retained_float * spectral_weights
    sr_o_values = (
        u[:, :history_width]
        * singular_values[:history_width]
        * history_weights[:history_width]
    )
    if force_override is None or force_override_keeps_residual_history:
        ek_values = (vh[:history_width, :] @ e_aug) * history_weights[:history_width]
    else:
        # Gradient memory now lives in transported_gradient. ``ek`` must not
        # retain a second, stale copy of the legacy gradient EMA.
        ek_values = jnp.zeros(history_width, dtype=state.ek.dtype)
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
    relative_singular_value_cutoff: chex.Numeric = -1.0,
    tikhonov_lambda: chex.Numeric = -1.0,
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
    relative_cutoff, _ = _wssr_resolve_spectral_controls(
        safe_leading_sv,
        damping,
        relative_singular_value_cutoff,
        tikhonov_lambda,
        eps,
    )
    retained = (singular_values / safe_leading_sv > relative_cutoff) & valid_leading
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
        tikhonov_lambda=tikhonov_lambda,
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


def right_warm_start_svd_explicit_current_blocks(
    o_cur: Array,
    state: WSSRWarmSVDCoreState,
    eta: chex.Numeric,
    key: Array,
    maxiter_initial: int,
    maxiter_warm: int,
    svd_working_rank: Optional[int] = None,
    subspace_only_averaging: bool = False,
    eps: chex.Numeric = 1e-12,
) -> Tuple[Array, Array, Array, Array]:
    """Right-warm SVD without materializing the augmented score matrix.

    The current score matrix is evaluated once and retained explicitly.  Only
    the history/current concatenation is represented as a block operator.  A
    zero-width ``state.u`` is supported by recovering the warm left subspace
    from ``state.sr_o``.
    """
    if maxiter_initial < 0:
        raise ValueError("maxiter_initial must be nonnegative")
    if maxiter_warm < 0:
        raise ValueError("maxiter_warm must be nonnegative")

    num_params = o_cur.shape[0]
    aug_width = state.sr_o.shape[1] + o_cur.shape[1]
    storage_width = state.sr_o.shape[1]
    working_rank_max = _resolve_svd_working_rank(
        svd_working_rank, storage_width
    )
    rank_capacity = min(
        working_rank_max,
        storage_width,
        num_params,
        aug_width,
    )
    if rank_capacity == 0:
        return (
            jnp.zeros((num_params, 0), dtype=o_cur.dtype),
            jnp.zeros((0,), dtype=o_cur.dtype),
            jnp.zeros((0, aug_width), dtype=o_cur.dtype),
            jnp.asarray(0, dtype=state.sr_rank.dtype),
        )

    rank = jnp.minimum(
        state.sr_rank,
        jnp.asarray(rank_capacity, dtype=state.sr_rank.dtype),
    )
    rank_mask = (jnp.arange(rank_capacity) < rank).astype(o_cur.dtype)

    def _masked_qr(z):
        q, _ = jnp.linalg.qr(z, mode="reduced")
        return q * rank_mask

    def _matmat(v):
        return explicit_current_augmented_matmat(
            o_cur,
            state,
            eta,
            v,
            subspace_only_averaging=subspace_only_averaging,
        )

    def _rmatmat(y):
        return explicit_current_augmented_rmatmat(
            o_cur,
            state,
            eta,
            y,
            subspace_only_averaging=subspace_only_averaging,
        )

    def _subspace_iterate(v, n_iter):
        for _ in range(n_iter):
            v = _masked_qr(_rmatmat(_matmat(v)))
        return v

    warm_u, has_warm_u = _select_warm_left_basis(
        state, rank_capacity, eps=eps
    )

    def _warm_branch(_):
        v = _masked_qr(_rmatmat(warm_u * rank_mask))
        return _subspace_iterate(v, maxiter_warm)

    def _initial_branch(_):
        omega = jax.random.normal(
            key, (aug_width, rank_capacity), dtype=o_cur.dtype
        )
        return _subspace_iterate(_masked_qr(omega * rank_mask), maxiter_initial)

    v = jax.lax.cond(has_warm_u, _warm_branch, _initial_branch, operand=None)
    y = _matmat(v)
    gram = y.T @ y
    eigvals, right_rot = jnp.linalg.eigh(gram)
    order = jnp.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    right_rot = right_rot[:, order]
    singular_values = jnp.sqrt(jnp.maximum(eigvals, 0.0)) * rank_mask
    safe_singular_values = jnp.where(singular_values > eps, singular_values, 1.0)
    u = ((y @ right_rot) / safe_singular_values) * rank_mask
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
    exact_first: bool = False,
    exact_first_force: bool = False,
    svd_working_rank: Optional[int] = None,
    constrain_update_norm: bool = True,
    spectral_regularization: str = "hard_floor",
    complement_weight: chex.Numeric = 1.0,
    relative_singular_value_cutoff: chex.Numeric = -1.0,
    tikhonov_lambda: chex.Numeric = -1.0,
    experimental_mode: str = "none",
    experimental_target_rank: int = -1,
    cluster_gap_threshold: chex.Numeric = 0.002,
    near_tail_modes: int = 0,
    adaptive_complement_beta: chex.Numeric = 0.0,
    adaptive_complement_beta_function: chex.Numeric = 0.0,
    complement_function_operator: Optional[Array] = None,
    smooth_transition_start: int = -1,
    smooth_transition_end: int = -1,
    force_aware_krylov_vectors: int = 0,
    iterative_complement_iterations: int = 0,
    mixed_precision_solve: bool = False,
    solution_prior: Optional[Array] = None,
    solution_recurrence_mode: str = "none",
    force_override: Optional[Array] = None,
    return_current_action: bool = False,
    current_sample_width: int = 0,
    current_block_scale: chex.Numeric = 1.0,
    reuse_warm_subspace: Array = False,
    fixed_warm_subspace: bool = False,
    eps: chex.Numeric = 1e-12,
):
    """Compute one right-subspace warm-start SVD WSSR core update."""
    storage_rank_max = min(sr_rank_max, state.sr_o.shape[1])
    if storage_rank_max == 0:
        zero_state = _zero_warm_svd_history_like(state)
        zero_result = WSSRSVDResult(
            jnp.zeros(o_aug.shape[0], dtype=o_aug.dtype),
            zero_state,
            jnp.asarray(0, dtype=state.sr_rank.dtype),
        )
        if return_current_action:
            return zero_result, jnp.zeros(
                (current_sample_width, state.sr_o.shape[1]), dtype=o_aug.dtype
            )
        return zero_result

    working_rank_max = _resolve_svd_working_rank(svd_working_rank, sr_rank_max)
    working_rank_max = min(working_rank_max, storage_rank_max)
    u, singular_values, vh, rank = _wssr_right_svd_decomposition(
        o_aug,
        state,
        key,
        working_rank_max,
        svd_maxiter_initial,
        svd_maxiter_warm,
        exact_first,
        eps,
        exact_first_force=exact_first_force,
        reuse_warm_subspace=reuse_warm_subspace,
        fixed_warm_subspace=fixed_warm_subspace,
    )
    if experimental_mode == "force_aware":
        target = working_rank_max if experimental_target_rank <= 0 else min(
            experimental_target_rank, working_rank_max
        )
        u, singular_values, vh, _, _ = (
            wssr_experimental.force_aware_rayleigh_ritz(
                o_aug, u, vh, o_aug @ e_aug,
                force_aware_krylov_vectors, target
            )
        )
        rank = jnp.asarray(target, dtype=state.sr_rank.dtype)
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
        relative_singular_value_cutoff=relative_singular_value_cutoff,
        tikhonov_lambda=tikhonov_lambda,
        experimental_mode=experimental_mode,
        experimental_target_rank=experimental_target_rank,
        cluster_gap_threshold=cluster_gap_threshold,
        near_tail_modes=near_tail_modes,
        adaptive_complement_beta=adaptive_complement_beta,
        adaptive_complement_beta_function=adaptive_complement_beta_function,
        complement_function_operator=complement_function_operator,
        smooth_transition_start=smooth_transition_start,
        smooth_transition_end=smooth_transition_end,
        iterative_complement_iterations=iterative_complement_iterations,
        mixed_precision_solve=mixed_precision_solve,
        mixed_precision_history_width=state.sr_o.shape[1],
        solution_prior=solution_prior,
        solution_recurrence_mode=solution_recurrence_mode,
        force_override=force_override,
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
    warm_result = WSSRSVDResult(
        result.grad_like_update, warm_state, result.active_rank
    )
    if return_current_action:
        # For an exact SVD (and for the fixed-left-subspace Ritz path), the
        # current rows of V @ diag(s) equal O_current.T @ U.  The ordinary
        # right-SSI path does not satisfy that identity exactly: its final V
        # spans only an approximate invariant subspace.  Reusing V*s there
        # would therefore change the Galerkin equation.  Keep the explicit
        # action on refresh steps and use the free cache only on delayed steps,
        # where ``_reuse_left_subspace`` constructs an exact Ritz factorization
        # inside the stored left subspace.
        cached_action = cached_current_action_from_svd(
            vh,
            singular_values,
            result.active_rank,
            state.sr_o.shape[1],
            current_sample_width,
            current_block_scale=current_block_scale,
            eps=eps,
        )

        def _explicit_current_action(_):
            scaled_current_block = o_aug[:, -current_sample_width:]
            safe_scale = jnp.maximum(
                jnp.asarray(current_block_scale, dtype=o_aug.dtype),
                jnp.asarray(eps, dtype=o_aug.dtype),
            )
            explicit = scaled_current_block.T @ u / safe_scale
            action = jnp.zeros(
                (current_sample_width, state.sr_o.shape[1]), dtype=o_aug.dtype
            )
            copy_width = min(explicit.shape[1], state.sr_o.shape[1])
            return action.at[:, :copy_width].set(explicit[:, :copy_width])

        cache_is_exact = (
            jnp.asarray(reuse_warm_subspace)
            & (state.has_u | (state.sr_rank0 > 0))
        )
        current_action = jax.lax.cond(
            cache_is_exact,
            lambda _: cached_action,
            _explicit_current_action,
            operand=None,
        )
        return warm_result, current_action
    return warm_result


def wssr_warm_svd_right_core_update_explicit_current_blocks(
    o_cur: Array,
    e_cur: Array,
    e_aug: Array,
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
    relative_singular_value_cutoff: chex.Numeric = -1.0,
    tikhonov_lambda: chex.Numeric = -1.0,
    experimental_mode: str = "none",
    adaptive_complement_beta: chex.Numeric = 0.0,
    adaptive_complement_beta_function: chex.Numeric = 0.0,
    mixed_precision_solve: bool = False,
    solution_prior: Optional[Array] = None,
    solution_recurrence_mode: str = "none",
    subspace_only_averaging: bool = False,
    return_current_action: bool = False,
    eps: chex.Numeric = 1e-12,
):
    """Right-warm update using an explicit current block, never ``O_aug``.

    This is the memory-bounded counterpart of
    :func:`wssr_warm_svd_right_core_update`.  In addition to the production
    Tikhonov/recurrence path, it supports the adaptive complement because that
    correction needs only the current score block and the already available
    augmented force.
    """
    storage_rank_max = min(sr_rank_max, state.sr_o.shape[1])
    if storage_rank_max == 0:
        zero_state = _zero_warm_svd_history_like(state)
        zero_result = WSSRSVDResult(
            jnp.zeros(o_cur.shape[0], dtype=o_cur.dtype),
            zero_state,
            jnp.asarray(0, dtype=state.sr_rank.dtype),
        )
        if return_current_action:
            return zero_result, jnp.zeros(
                (o_cur.shape[1], state.sr_o.shape[1]), dtype=o_cur.dtype
            )
        return zero_result

    working_rank_max = min(
        _resolve_svd_working_rank(svd_working_rank, sr_rank_max),
        storage_rank_max,
    )
    u, singular_values, vh, rank = (
        right_warm_start_svd_explicit_current_blocks(
            o_cur,
            state,
            eta,
            key,
            svd_maxiter_initial,
            svd_maxiter_warm,
            svd_working_rank=working_rank_max,
            subspace_only_averaging=subspace_only_averaging,
            eps=eps,
        )
    )
    if solution_recurrence_mode == "none":
        force = explicit_current_augmented_matvec(
            o_cur,
            state,
            eta,
            e_aug,
            subspace_only_averaging=subspace_only_averaging,
        )
    else:
        # Solution recurrence corrects the current-batch equation; historical
        # residuals are retained only in ``ek`` for the next subspace update.
        force = o_cur @ e_cur
    result = _wssr_update_from_svd(
        None,
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
        relative_singular_value_cutoff=relative_singular_value_cutoff,
        tikhonov_lambda=tikhonov_lambda,
        experimental_mode=experimental_mode,
        adaptive_complement_beta=adaptive_complement_beta,
        adaptive_complement_beta_function=adaptive_complement_beta_function,
        complement_function_operator=o_cur,
        mixed_precision_solve=mixed_precision_solve,
        mixed_precision_history_width=state.sr_o.shape[1],
        solution_prior=solution_prior,
        solution_recurrence_mode=solution_recurrence_mode,
        force_override=force,
        force_override_keeps_residual_history=True,
        eps=eps,
    )
    u_state = jnp.zeros_like(state.u)
    warm_width = min(u.shape[1], state.u.shape[1])
    rank_mask = (jnp.arange(warm_width) < rank).astype(u.dtype)
    u_state = u_state.at[:, :warm_width].set(
        u[:, :warm_width] * rank_mask
    )
    warm_state = WSSRWarmSVDCoreState(
        sr_o=result.state.sr_o,
        ek=result.state.ek,
        sr_rank0=result.state.sr_rank0,
        sr_rank=result.state.sr_rank,
        u=u_state,
        has_u=jnp.where(rank > 0, jnp.array(True), state.has_u),
    )
    warm_result = WSSRSVDResult(
        result.grad_like_update, warm_state, result.active_rank
    )
    if return_current_action:
        # This decomposition is also produced by approximate right SSI, so its
        # V*s block is not an exact replacement for O_current.T @ U.  Preserve
        # the Galerkin update exactly; the delayed-refresh path above is where
        # the mathematically valid cache is used.
        explicit = o_cur.T @ u
        current_action = jnp.zeros(
            (o_cur.shape[1], state.sr_o.shape[1]), dtype=o_cur.dtype
        )
        copy_width = min(explicit.shape[1], state.sr_o.shape[1])
        current_action = current_action.at[:, :copy_width].set(
            explicit[:, :copy_width]
        )
        return warm_result, current_action
    return warm_result


_jitted_wssr_warm_svd_right_core_update_explicit_current_blocks = jax.jit(
    wssr_warm_svd_right_core_update_explicit_current_blocks,
    static_argnames=(
        "sr_rank_max",
        "svd_maxiter_initial",
        "svd_maxiter_warm",
        "svd_working_rank",
        "constrain_update_norm",
        "spectral_regularization",
        "complement_weight",
        "experimental_mode",
        "mixed_precision_solve",
        "solution_recurrence_mode",
        "subspace_only_averaging",
        "return_current_action",
    ),
)


def _wssr_right_svd_decomposition(
    o_aug: Array,
    state: WSSRWarmSVDCoreState,
    key: Array,
    working_rank_max: int,
    svd_maxiter_initial: int,
    svd_maxiter_warm: int,
    exact_first: bool,
    eps: chex.Numeric,
    exact_first_force: bool = False,
    reuse_warm_subspace: Array = False,
    fixed_warm_subspace: bool = False,
):
    """Return the decomposition used by right-warm WSSR."""
    has_warm_subspace = state.has_u | (state.sr_rank0 > 0)

    def _reuse_left_subspace(_):
        """Recompute Ritz pairs inside the stored left subspace only."""
        warm_u, _ = _select_warm_left_basis(
            state, working_rank_max, eps=eps
        )
        rank = jnp.minimum(
            state.sr_rank,
            jnp.asarray(working_rank_max, dtype=state.sr_rank.dtype),
        )
        rank_mask = (
            jnp.arange(working_rank_max) < rank
        ).astype(o_aug.dtype)
        q, _ = jnp.linalg.qr(warm_u * rank_mask, mode="reduced")
        q = q * rank_mask
        projected = q.T @ o_aug
        gram = projected @ projected.T
        eigvals, rotation = jnp.linalg.eigh(gram)
        order = jnp.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        rotation = rotation[:, order]
        singular_values = (
            jnp.sqrt(jnp.maximum(eigvals, 0.0)) * rank_mask
        )
        safe_singular_values = jnp.where(
            singular_values > eps, singular_values, 1.0
        )
        u = (q @ rotation) * rank_mask
        vh = (
            (rotation.T @ projected)
            / safe_singular_values[:, None]
            * rank_mask[:, None]
        )
        return u, singular_values, vh, rank

    def _reuse_fixed_left_subspace(_):
        """Keep the stored U fixed and only evaluate its current action.

        This is the true lazy-SSI path.  Unlike ``_reuse_left_subspace``, it
        performs no QR, eigendecomposition, or within-subspace rotation.  The
        row norms below are used only to maintain the legacy fixed-shape WSSR
        state; the full-current-batch Galerkin solve consumes the exact
        ``O_current.T @ U`` reconstructed from ``vh * singular_values``.
        """
        warm_u, _ = _select_warm_left_basis(
            state, working_rank_max, eps=eps
        )
        rank = jnp.minimum(
            state.sr_rank0,
            jnp.asarray(working_rank_max, dtype=state.sr_rank.dtype),
        )
        rank_mask = (
            jnp.arange(working_rank_max) < rank
        ).astype(o_aug.dtype)
        u = warm_u * rank_mask
        projected = (u.T @ o_aug) * rank_mask[:, None]
        singular_values = jnp.linalg.norm(projected, axis=1) * rank_mask
        safe_singular_values = jnp.where(
            singular_values > eps, singular_values, 1.0
        )
        vh = (
            projected / safe_singular_values[:, None]
        ) * rank_mask[:, None]
        return u, singular_values, vh, rank

    def _exact_initial(_):
        # This branch is used exactly once: only while the warm state is empty.
        # ``full_matrices=False`` computes the thin SVD of the same augmented
        # operator consumed by the existing right-warm implementation.
        exact_u, exact_s, exact_vh = jnp.linalg.svd(
            o_aug, full_matrices=False
        )
        rank = jnp.minimum(
            state.sr_rank,
            jnp.asarray(working_rank_max, dtype=state.sr_rank.dtype),
        )
        rank_mask = (jnp.arange(working_rank_max) < rank).astype(o_aug.dtype)
        return (
            exact_u[:, :working_rank_max] * rank_mask,
            exact_s[:working_rank_max] * rank_mask,
            exact_vh[:working_rank_max, :] * rank_mask[:, None],
            rank,
        )

    def _normal_warm_svd(_):
        return right_warm_start_svd(
            o_aug,
            state,
            key,
            svd_maxiter_initial,
            svd_maxiter_warm,
            svd_working_rank=working_rank_max,
            eps=eps,
        )

    def _fresh_decomposition(_):
        if exact_first:
            return jax.lax.cond(
                has_warm_subspace & ~jnp.asarray(exact_first_force),
                _normal_warm_svd,
                _exact_initial,
                operand=None,
            )
        return _normal_warm_svd(None)

    reuse_fn = (
        _reuse_fixed_left_subspace
        if fixed_warm_subspace
        else _reuse_left_subspace
    )
    return jax.lax.cond(
        jnp.asarray(reuse_warm_subspace) & has_warm_subspace,
        reuse_fn,
        _fresh_decomposition,
        operand=None,
    )


_jitted_wssr_warm_svd_right_core_update = jax.jit(
    wssr_warm_svd_right_core_update,
    static_argnames=(
        "sr_rank_max",
        "svd_maxiter_initial",
        "svd_maxiter_warm",
        "exact_first",
        "exact_first_force",
        "svd_working_rank",
        "constrain_update_norm",
        "spectral_regularization",
        "complement_weight",
        "experimental_mode",
        "experimental_target_rank",
        "near_tail_modes",
        "smooth_transition_start",
        "smooth_transition_end",
        "force_aware_krylov_vectors",
        "iterative_complement_iterations",
        "mixed_precision_solve",
        "solution_recurrence_mode",
        "return_current_action",
        "current_sample_width",
        "fixed_warm_subspace",
    ),
)


def wssr_anisotropic_matrix_history_core_update(
    o_cur: Array,
    e_cur: Array,
    state: WSSRWarmSVDCoreState,
    key: Array,
    eta_S: chex.Numeric,
    damping: chex.Numeric,
    norm_constraint: chex.Numeric,
    sr_rank_max: int,
    sr_scale: chex.Numeric = 1.1,
    svd_maxiter_initial: int = 8,
    svd_maxiter_warm: int = 2,
    exact_first: bool = False,
    exact_first_force: bool = False,
    svd_working_rank: Optional[int] = None,
    constrain_update_norm: bool = False,
    relative_singular_value_cutoff: chex.Numeric = -1.0,
    tikhonov_lambda: chex.Numeric = -1.0,
    noise_scale: chex.Numeric = 1.0,
) -> WSSRAnisotropicMatrixHistoryResult:
    """Run legacy augmented-factor WSSR with per-history-mode weights.

    Unlike current-subspace reduced-metric averaging, historical directions
    remain columns of the SSI operator and can therefore change the computed
    subspace. Only the shared scalar ``eta_S`` is replaced by spectral-position
    weights; the RHS remains the current batch force.
    """
    history_eigenvalues = jnp.sum(jnp.square(state.sr_o), axis=0)
    history_active = jnp.arange(state.sr_o.shape[1]) < state.sr_rank0
    diagnostics = spectral_position_history_weights(
        history_eigenvalues,
        history_active,
        state.sr_rank0 > 0,
        eta_S,
        o_cur.shape[1],
        noise_scale=noise_scale,
    )
    o_aug = augment_wssr_anisotropic_matrix_history(
        o_cur, state, diagnostics
    )
    # Gradient averaging is intentionally absent. e_aug is unused by the solve
    # because force_override supplies O_current @ epsilon_current, and it is
    # also omitted from the next state by the standard force-override path.
    e_aug = jnp.zeros((o_aug.shape[1],), dtype=o_cur.dtype)
    current_force = o_cur @ e_cur
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
        exact_first=exact_first,
        exact_first_force=exact_first_force,
        svd_working_rank=svd_working_rank,
        constrain_update_norm=constrain_update_norm,
        spectral_regularization="tikhonov",
        complement_weight=0.0,
        relative_singular_value_cutoff=relative_singular_value_cutoff,
        tikhonov_lambda=tikhonov_lambda,
        force_override=current_force,
    )
    return WSSRAnisotropicMatrixHistoryResult(
        result.grad_like_update,
        result.state,
        result.active_rank,
        diagnostics,
    )


def compute_wssr_anisotropic_matrix_history_update(
    log_psi_apply: ModelApply[P],
    params: P,
    positions: Array,
    local_energies: Array,
    energy: Array,
    state: WSSRWarmSVDCoreState,
    key: Array,
    eta_S: chex.Numeric,
    damping: chex.Numeric,
    norm_constraint: chex.Numeric,
    sr_rank_max: int,
    sr_scale: chex.Numeric = 1.1,
    svd_maxiter_initial: int = 8,
    svd_maxiter_warm: int = 2,
    exact_first: bool = False,
    exact_first_force: bool = False,
    svd_working_rank: Optional[int] = None,
    constrain_update_norm: bool = False,
    relative_singular_value_cutoff: chex.Numeric = -1.0,
    tikhonov_lambda: chex.Numeric = -1.0,
    noise_scale: chex.Numeric = 1.0,
):
    """Build current scores, apply anisotropic matrix history, and unflatten."""
    o_cur, unravel_fn = center_and_scale_score_matrix(
        log_psi_apply, params, positions
    )
    e_cur = center_and_scale_energy_residuals(local_energies, energy)
    result = wssr_anisotropic_matrix_history_core_update(
        o_cur,
        e_cur,
        state,
        key,
        eta_S,
        damping,
        norm_constraint,
        sr_rank_max,
        sr_scale=sr_scale,
        svd_maxiter_initial=svd_maxiter_initial,
        svd_maxiter_warm=svd_maxiter_warm,
        exact_first=exact_first,
        exact_first_force=exact_first_force,
        svd_working_rank=svd_working_rank,
        constrain_update_norm=constrain_update_norm,
        relative_singular_value_cutoff=relative_singular_value_cutoff,
        tikhonov_lambda=tikhonov_lambda,
        noise_scale=noise_scale,
    )
    current_gradient = o_cur @ e_cur
    return (
        unravel_fn(result.grad_like_update),
        result.state,
        result.active_rank,
        jnp.linalg.norm(current_gradient),
        result.diagnostics,
    )


def wssr_current_subspace_reduced_metric_core_update(
    o_cur: Array,
    e_cur: Array,
    state: WSSRWarmSVDCoreState,
    key: Array,
    eta_S: chex.Numeric,
    damping: chex.Numeric,
    norm_constraint: chex.Numeric,
    sr_rank_max: int,
    history_mode: str,
    sr_scale: chex.Numeric = 1.1,
    svd_maxiter_initial: int = 8,
    svd_maxiter_warm: int = 2,
    exact_first: bool = False,
    exact_first_force: bool = False,
    svd_working_rank: Optional[int] = None,
    constrain_update_norm: bool = False,
    relative_singular_value_cutoff: chex.Numeric = -1.0,
    tikhonov_lambda: chex.Numeric = -1.0,
    cluster_gap_threshold: chex.Numeric = 0.01,
    noise_scale: chex.Numeric = 1.0,
    drift_scale: chex.Numeric = 1.0,
    eps: chex.Numeric = 1e-12,
) -> WSSRReducedMetricHistoryResult:
    """Solve with a current-batch subspace and historical reduced metric.

    In the code convention ``o_cur`` is ``O_bar.T`` with shape
    ``(num_params, num_samples)``.  SSI is applied to ``o_cur`` alone.  The
    previous factor stored in ``state.sr_o`` is projected into that new basis,
    and only the resulting rank-by-rank metric is averaged.  The force is
    always the current ``o_cur @ e_cur``; there is no gradient EMA.
    """
    if history_mode not in _WSSR_REDUCED_METRIC_HISTORY_MODES:
        raise ValueError(
            "history_mode must be one of "
            f"{_WSSR_REDUCED_METRIC_HISTORY_MODES}"
        )
    storage_rank_max = min(sr_rank_max, state.sr_o.shape[1])
    working_rank_max = min(
        _resolve_svd_working_rank(svd_working_rank, sr_rank_max),
        storage_rank_max,
        o_cur.shape[0],
        o_cur.shape[1],
    )
    if working_rank_max == 0:
        zero_state = _zero_warm_svd_history_like(state)
        zero_diag = WSSRReducedMetricHistoryDiagnostics(
            eta_by_mode=jnp.zeros((0,), dtype=o_cur.dtype),
            eta_mean=jnp.asarray(0.0, dtype=o_cur.dtype),
            eta_head=jnp.asarray(0.0, dtype=o_cur.dtype),
            eta_tail=jnp.asarray(0.0, dtype=o_cur.dtype),
            noise_floor=jnp.asarray(0.0, dtype=o_cur.dtype),
            history_overlap=jnp.asarray(0.0, dtype=o_cur.dtype),
            cluster_count=jnp.asarray(0, dtype=jnp.int32),
            drift_ratio_mean=jnp.asarray(0.0, dtype=o_cur.dtype),
        )
        return WSSRReducedMetricHistoryResult(
            jnp.zeros(o_cur.shape[0], dtype=o_cur.dtype),
            zero_state,
            jnp.asarray(0, dtype=state.sr_rank.dtype),
            zero_diag,
        )

    # The decomposition sees only the current batch. ``state.u`` is merely an
    # iterative initial guess and therefore does not change the operator whose
    # Ritz subspace is computed.
    u, singular_values, _, rank = _wssr_right_svd_decomposition(
        o_cur,
        state,
        key,
        working_rank_max,
        svd_maxiter_initial,
        svd_maxiter_warm,
        exact_first,
        eps,
        exact_first_force=exact_first_force,
    )
    leading_sv = singular_values[0]
    valid_leading = leading_sv > eps
    safe_leading_sv = jnp.where(valid_leading, leading_sv, 1.0)
    relative_cutoff, lambda_reg = _wssr_resolve_spectral_controls(
        safe_leading_sv,
        damping,
        relative_singular_value_cutoff,
        tikhonov_lambda,
        eps,
    )
    retained = (
        (jnp.arange(working_rank_max) < rank)
        & (singular_values / safe_leading_sv > relative_cutoff)
        & valid_leading
    )
    active_rank = jnp.sum(retained.astype(state.sr_rank.dtype))
    valid_update = valid_leading & (active_rank > 0)
    current_eigenvalues = jnp.where(
        retained, jnp.square(singular_values), 0.0
    )

    history_mask = (
        jnp.arange(state.sr_o.shape[1]) < state.sr_rank0
    ).astype(o_cur.dtype)
    history_factor = state.sr_o * history_mask[None, :]
    history_projection = history_factor.T @ u
    history_metric = history_projection.T @ history_projection
    averaged_metric, diagnostics = average_current_reduced_metric(
        current_eigenvalues,
        history_metric,
        retained,
        state.sr_rank0 > 0,
        history_mode,
        eta_S,
        o_cur.shape[1],
        cluster_gap_threshold=cluster_gap_threshold,
        noise_scale=noise_scale,
        drift_scale=drift_scale,
        history_total_mass=jnp.sum(jnp.square(history_factor)),
        eps=eps,
    )

    # The diagonal spectrum-only arm needs no second decomposition.  Adaptive
    # spectral clusters can contain dense invariant blocks; there one reduced
    # eigendecomposition both solves those blocks and produces a factor F_t
    # with F_t F_t.T equal to the averaged metric. No dense parameter-space
    # Fisher matrix is ever formed.
    if history_mode == "cluster_adaptive":
        reduced_eigenvalues, reduced_rotation = jnp.linalg.eigh(
            0.5 * (averaged_metric + averaged_metric.T)
        )
        order = jnp.argsort(reduced_eigenvalues)[::-1]
        reduced_eigenvalues = jnp.maximum(reduced_eigenvalues[order], 0.0)
        reduced_rotation = reduced_rotation[:, order]
    else:
        reduced_eigenvalues = jnp.maximum(jnp.diag(averaged_metric), 0.0)
        reduced_rotation = jnp.eye(working_rank_max, dtype=o_cur.dtype)
    force = o_cur @ e_cur
    reduced_force = reduced_rotation.T @ (u.T @ force)
    solved_coefficients = reduced_force / (
        reduced_eigenvalues + lambda_reg
    )
    direction = u @ (reduced_rotation @ solved_coefficients)
    direction = jnp.where(valid_update, direction, jnp.zeros_like(direction))
    if constrain_update_norm:
        direction = constrain_norm(direction, norm_constraint, eps=eps)

    metric_basis = u @ reduced_rotation
    factor_mask = (
        jnp.arange(working_rank_max) < active_rank
    ).astype(o_cur.dtype)
    factor_values = (
        metric_basis
        * jnp.sqrt(reduced_eigenvalues)[None, :]
        * factor_mask[None, :]
    )
    next_sr_o = jnp.zeros_like(state.sr_o)
    next_sr_o = next_sr_o.at[:, :working_rank_max].set(factor_values)
    next_u = jnp.zeros_like(state.u)
    warm_width = min(working_rank_max, state.u.shape[1])
    next_u = next_u.at[:, :warm_width].set(
        u[:, :warm_width]
        * retained[:warm_width].astype(o_cur.dtype)[None, :]
    )
    updated_rank = _update_working_rank(
        active_rank, state.sr_rank, working_rank_max, sr_scale
    )
    capped_rank = jnp.minimum(
        state.sr_rank,
        jnp.asarray(working_rank_max, dtype=state.sr_rank.dtype),
    )
    next_state = WSSRWarmSVDCoreState(
        sr_o=jnp.where(valid_update, next_sr_o, jnp.zeros_like(next_sr_o)),
        # Gradient history is deliberately absent: RHS = current batch force.
        ek=jnp.zeros_like(state.ek),
        sr_rank0=jnp.where(
            valid_update,
            active_rank,
            jnp.asarray(0, dtype=state.sr_rank0.dtype),
        ),
        sr_rank=jnp.where(valid_update, updated_rank, capped_rank),
        u=jnp.where(valid_update, next_u, state.u),
        has_u=jnp.where(valid_update, jnp.asarray(True), state.has_u),
    )
    return WSSRReducedMetricHistoryResult(
        direction, next_state, active_rank, diagnostics
    )


def compute_wssr_current_subspace_reduced_metric_update(
    log_psi_apply: ModelApply[P],
    params: P,
    positions: Array,
    local_energies: Array,
    energy: Array,
    state: WSSRWarmSVDCoreState,
    key: Array,
    eta_S: chex.Numeric,
    damping: chex.Numeric,
    norm_constraint: chex.Numeric,
    sr_rank_max: int,
    history_mode: str,
    sr_scale: chex.Numeric = 1.1,
    svd_maxiter_initial: int = 8,
    svd_maxiter_warm: int = 2,
    exact_first: bool = False,
    exact_first_force: bool = False,
    svd_working_rank: Optional[int] = None,
    constrain_update_norm: bool = False,
    relative_singular_value_cutoff: chex.Numeric = -1.0,
    tikhonov_lambda: chex.Numeric = -1.0,
    cluster_gap_threshold: chex.Numeric = 0.01,
    noise_scale: chex.Numeric = 1.0,
    drift_scale: chex.Numeric = 1.0,
):
    """Build current scores, apply reduced-metric history, and unflatten."""
    o_cur, unravel_fn = center_and_scale_score_matrix(
        log_psi_apply, params, positions
    )
    e_cur = center_and_scale_energy_residuals(local_energies, energy)
    result = wssr_current_subspace_reduced_metric_core_update(
        o_cur,
        e_cur,
        state,
        key,
        eta_S,
        damping,
        norm_constraint,
        sr_rank_max,
        history_mode,
        sr_scale=sr_scale,
        svd_maxiter_initial=svd_maxiter_initial,
        svd_maxiter_warm=svd_maxiter_warm,
        exact_first=exact_first,
        exact_first_force=exact_first_force,
        svd_working_rank=svd_working_rank,
        constrain_update_norm=constrain_update_norm,
        relative_singular_value_cutoff=relative_singular_value_cutoff,
        tikhonov_lambda=tikhonov_lambda,
        cluster_gap_threshold=cluster_gap_threshold,
        noise_scale=noise_scale,
        drift_scale=drift_scale,
    )
    current_gradient = o_cur @ e_cur
    return (
        unravel_fn(result.grad_like_update),
        result.state,
        result.active_rank,
        jnp.linalg.norm(current_gradient),
        result.diagnostics,
    )


def wssr_warm_svd_right_reference_diagnostic_core_update(
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
    exact_first: bool = False,
    exact_first_force: bool = False,
    svd_working_rank: Optional[int] = None,
    constrain_update_norm: bool = False,
    compute_exact_reference: bool = True,
    spectral_regularization: str = "hard_floor",
    complement_weight: chex.Numeric = 1.0,
    relative_singular_value_cutoff: chex.Numeric = -1.0,
    tikhonov_lambda: chex.Numeric = -1.0,
    eps: chex.Numeric = 1e-12,
):
    """Run one update and return expensive exact-reference validation metrics."""
    storage_rank_max = min(sr_rank_max, state.sr_o.shape[1])
    working_rank_max = min(
        _resolve_svd_working_rank(svd_working_rank, sr_rank_max),
        storage_rank_max,
    )
    u, singular_values, vh, rank = _wssr_right_svd_decomposition(
        o_aug,
        state,
        key,
        working_rank_max,
        svd_maxiter_initial,
        svd_maxiter_warm,
        exact_first,
        eps,
        exact_first_force=exact_first_force,
    )
    base = _wssr_update_from_svd(
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
        relative_singular_value_cutoff=relative_singular_value_cutoff,
        tikhonov_lambda=tikhonov_lambda,
        eps=eps,
    )
    u_state = jnp.zeros_like(state.u)
    warm_width = min(u.shape[1], state.u.shape[1])
    rank_mask = (jnp.arange(warm_width) < rank).astype(u.dtype)
    u_state = u_state.at[:, :warm_width].set(u[:, :warm_width] * rank_mask)
    warm_state = WSSRWarmSVDCoreState(
        base.state.sr_o,
        base.state.ek,
        base.state.sr_rank0,
        base.state.sr_rank,
        u_state,
        jnp.where(rank > 0, jnp.array(True), state.has_u),
    )
    result = WSSRSVDResult(base.grad_like_update, warm_state, base.active_rank)

    actual_update = result.grad_like_update
    actual_norm = jnp.linalg.norm(actual_update)
    if compute_exact_reference:
        exact_u_all, exact_s_all, exact_vh_all = jnp.linalg.svd(
            o_aug, full_matrices=False
        )
        exact_u = exact_u_all[:, :working_rank_max]
        exact_s = exact_s_all[:working_rank_max]
        exact_vh = exact_vh_all[:working_rank_max, :]
        exact_result = _wssr_update_from_svd(
            o_aug, e_aug, state, exact_u, exact_s, exact_vh, damping,
            norm_constraint, storage_rank_max, sr_scale, constrain_update_norm,
            rank_update_max=working_rank_max,
            spectral_regularization=spectral_regularization,
            complement_weight=complement_weight,
            relative_singular_value_cutoff=relative_singular_value_cutoff,
            tikhonov_lambda=tikhonov_lambda,
            eps=eps,
        )
        reference_update = exact_result.grad_like_update
        reference_norm = jnp.linalg.norm(reference_update)
        update_relerr = jnp.linalg.norm(actual_update - reference_update) / jnp.maximum(reference_norm, eps)
        update_cosine = jnp.vdot(actual_update, reference_update) / jnp.maximum(actual_norm * reference_norm, eps)
        overlap_s = jnp.linalg.svd(u.T @ exact_u, compute_uv=False)
        max_principal_angle = jnp.arccos(jnp.clip(jnp.min(overlap_s), -1.0, 1.0))
        subspace_residual = jnp.linalg.norm(exact_u - u @ (u.T @ exact_u))
        singular_rank_relerr = jnp.abs(singular_values[-1] - exact_s[-1]) / jnp.maximum(jnp.abs(exact_s[-1]), eps)
        exact_singular_at_rank = exact_s[-1]
    else:
        nan = jnp.asarray(jnp.nan, dtype=o_aug.dtype)
        update_relerr = update_cosine = max_principal_angle = nan
        subspace_residual = singular_rank_relerr = exact_singular_at_rank = nan

    leading = jnp.maximum(jnp.abs(singular_values[0]), eps)
    relative_cutoff, _ = _wssr_resolve_spectral_controls(
        leading,
        damping,
        relative_singular_value_cutoff,
        tikhonov_lambda,
        eps,
    )
    retained = singular_values / leading > relative_cutoff
    inv_cap, inv_perp = _wssr_inverse_spectral_coefficients(
        jnp.where(retained, singular_values, 1.0),
        leading,
        jnp.maximum(jnp.abs(damping), eps),
        spectral_regularization,
        complement_weight,
        tikhonov_lambda=tikhonov_lambda,
    )
    force = o_aug @ e_aug
    projected = (u.T @ force) * retained.astype(o_aug.dtype)
    resolved = u @ (inv_cap * projected)
    complement = inv_perp * (force - u @ projected)
    resolved_norm = jnp.linalg.norm(resolved)
    complement_norm = jnp.linalg.norm(complement)
    total_norm = jnp.linalg.norm(resolved + complement)
    comp_cosine = jnp.vdot(resolved, complement) / jnp.maximum(
        resolved_norm * complement_norm, eps
    )
    diagnostics = {
        "wssr_diag_raw_direction_norm": actual_norm,
        "wssr_diag_raw_direction_max_abs": jnp.max(jnp.abs(actual_update)),
        "wssr_diag_singular_leading": singular_values[0],
        "wssr_diag_singular_at_rank": singular_values[-1],
        "wssr_diag_resolved_norm": resolved_norm,
        "wssr_diag_complement_norm": complement_norm,
        "wssr_diag_complement_resolved_ratio": complement_norm
        / jnp.maximum(resolved_norm, eps),
        "wssr_diag_complement_total_ratio": complement_norm
        / jnp.maximum(total_norm, eps),
        "wssr_diag_complement_resolved_cosine": comp_cosine,
        "wssr_diag_inv_perp": inv_perp,
        "wssr_diag_exact_initialization_used": (
            jnp.asarray(exact_first)
            & (jnp.asarray(exact_first_force) | ~(state.has_u | (state.sr_rank0 > 0)))
        ).astype(jnp.int32),
        "wssr_diag_finite": jnp.all(jnp.isfinite(actual_update)).astype(jnp.int32),
    }
    if compute_exact_reference:
        diagnostics.update(
            {
                "wssr_diag_update_relative_error": update_relerr,
                "wssr_diag_update_cosine": update_cosine,
                "wssr_diag_max_principal_angle": max_principal_angle,
                "wssr_diag_subspace_residual": subspace_residual,
                "wssr_diag_singular_rank_relerr": singular_rank_relerr,
                "wssr_diag_exact_singular_at_rank": exact_singular_at_rank,
            }
        )
    return result, diagnostics


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
    exact_first: bool = False,
    exact_first_force: bool = False,
    svd_working_rank: Optional[int] = None,
    constrain_update_norm: bool = True,
    spectral_regularization: str = "hard_floor",
    complement_weight: chex.Numeric = 1.0,
    relative_singular_value_cutoff: chex.Numeric = -1.0,
    tikhonov_lambda: chex.Numeric = -1.0,
    experimental_mode: str = "none",
    experimental_target_rank: int = -1,
    cluster_gap_threshold: chex.Numeric = 0.002,
    near_tail_modes: int = 0,
    adaptive_complement_beta: chex.Numeric = 0.0,
    adaptive_complement_beta_function: chex.Numeric = 0.0,
    smooth_transition_start: int = -1,
    smooth_transition_end: int = -1,
    force_aware_krylov_vectors: int = 0,
    iterative_complement_iterations: int = 0,
    mixed_precision_solve: bool = False,
    solution_prior: Optional[Array] = None,
    solution_recurrence_mode: str = "none",
    residual_evaluation: str = "rank_coordinate",
    residual_dual_mode_diagnostics: bool = False,
    solution_error_feedback: bool = False,
    error_feedback_state: Optional[Array] = None,
    error_feedback_norm_cap: chex.Numeric = 10.0,
    error_feedback_decay: chex.Numeric = 1.0,
    error_feedback_cap_reference: str = "correction",
    galerkin_solve_backend: str = "host_fp64",
    subspace_only_averaging: bool = False,
    semi_matrix_free_augmented: bool = False,
    reuse_warm_subspace: Array = False,
    fixed_warm_subspace: bool = False,
    return_gradient_diagnostics: bool = False,
    return_recurrence_diagnostics: bool = False,
):
    """Compute a right-subspace warm-start SVD WSSR update and unflatten it."""
    o_cur, unravel_fn = center_and_scale_score_matrix(
        log_psi_apply, params, positions
    )
    e_cur = center_and_scale_energy_residuals(local_energies, energy)
    needs_galerkin = (
        solution_recurrence_mode == "residual"
        and (
            residual_evaluation == "full_current_batch"
            or residual_dual_mode_diagnostics
            or solution_error_feedback
        )
    )
    if semi_matrix_free_augmented:
        e_aug = (
            augment_wssr_subspace_residuals(e_cur, state, eta)
            if subspace_only_averaging
            else augment_wssr_residuals(e_cur, state, eta)
        )
        o_aug = None
        core_result = (
            _jitted_wssr_warm_svd_right_core_update_explicit_current_blocks(
                o_cur,
                e_cur,
                e_aug,
                state,
                key,
                eta,
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
                relative_singular_value_cutoff=(
                    relative_singular_value_cutoff
                ),
                tikhonov_lambda=tikhonov_lambda,
                experimental_mode=experimental_mode,
                adaptive_complement_beta=adaptive_complement_beta,
                adaptive_complement_beta_function=(
                    adaptive_complement_beta_function
                ),
                mixed_precision_solve=mixed_precision_solve,
                solution_prior=solution_prior,
                solution_recurrence_mode=solution_recurrence_mode,
                subspace_only_averaging=subspace_only_averaging,
                return_current_action=needs_galerkin,
            )
        )
        if needs_galerkin:
            result, cached_current_action = core_result
        else:
            result = core_result
            cached_current_action = None
    else:
        if subspace_only_averaging:
            o_aug, e_aug = augment_wssr_subspace_system(
                o_cur, e_cur, state, eta
            )
        else:
            o_aug, e_aug = augment_wssr_system(o_cur, e_cur, state, eta)
        core_result = _jitted_wssr_warm_svd_right_core_update(
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
            exact_first=exact_first,
            exact_first_force=exact_first_force,
            svd_working_rank=svd_working_rank,
            constrain_update_norm=constrain_update_norm,
            spectral_regularization=spectral_regularization,
            complement_weight=complement_weight,
            relative_singular_value_cutoff=relative_singular_value_cutoff,
            tikhonov_lambda=tikhonov_lambda,
            experimental_mode=experimental_mode,
            experimental_target_rank=experimental_target_rank,
            cluster_gap_threshold=cluster_gap_threshold,
            near_tail_modes=near_tail_modes,
            adaptive_complement_beta=adaptive_complement_beta,
            adaptive_complement_beta_function=(
                adaptive_complement_beta_function
            ),
            complement_function_operator=o_cur,
            smooth_transition_start=smooth_transition_start,
            smooth_transition_end=smooth_transition_end,
            force_aware_krylov_vectors=force_aware_krylov_vectors,
            iterative_complement_iterations=iterative_complement_iterations,
            mixed_precision_solve=mixed_precision_solve,
            solution_prior=solution_prior,
            solution_recurrence_mode=solution_recurrence_mode,
            return_current_action=needs_galerkin,
            current_sample_width=o_cur.shape[1],
            current_block_scale=1.0,
            reuse_warm_subspace=reuse_warm_subspace,
            fixed_warm_subspace=fixed_warm_subspace,
        )
        if needs_galerkin:
            result, cached_current_action = core_result
        else:
            result = core_result
            cached_current_action = None
    zero_scalar = jnp.asarray(0.0, dtype=o_cur.dtype)
    recurrence_diagnostics = SolutionRecurrenceDiagnostics(
        error_feedback=jnp.zeros(o_cur.shape[0], dtype=o_cur.dtype),
        error_feedback_clip_increment=jnp.asarray(0, dtype=jnp.int32),
        sample_residual_norm=zero_scalar,
        correctable_residual_norm=zero_scalar,
        correctable_residual_ratio=zero_scalar,
        captured_sample_residual_ratio=zero_scalar,
        residual_norm_reduction=zero_scalar,
        residual_reduction_fraction=zero_scalar,
        error_feedback_norm=zero_scalar,
        error_feedback_uncapped_norm=zero_scalar,
        error_feedback_cap_norm=zero_scalar,
        error_feedback_to_sample_ratio=zero_scalar,
        error_feedback_to_parameter_conflict_ratio=zero_scalar,
        correction_relative_difference=zero_scalar,
    )
    if needs_galerkin:
        basis = recover_u_from_sr_o(
            result.state.sr_o,
            result.state.sr_rank0,
            result.state.sr_o.shape[1],
        )
        leading_sv = jnp.linalg.norm(result.state.sr_o[:, 0])
        _, lambda_reg = _wssr_resolve_spectral_controls(
            leading_sv,
            damping,
            relative_singular_value_cutoff,
            tikhonov_lambda,
        )
        feedback = (
            error_feedback_state
            if error_feedback_state is not None
            else jnp.zeros(o_cur.shape[0], dtype=o_cur.dtype)
        )
        galerkin = galerkin_residual_solution_recurrence(
            o_cur,
            e_cur,
            basis,
            solution_prior,
            lambda_reg,
            error_feedback=feedback,
            enable_error_feedback=solution_error_feedback,
            error_feedback_norm_cap=error_feedback_norm_cap,
            error_feedback_decay=error_feedback_decay,
            error_feedback_cap_reference=error_feedback_cap_reference,
            current_action=cached_current_action,
            solve_backend=galerkin_solve_backend,
        )
        rank_correction = result.grad_like_update - solution_prior
        correction_relative_difference = jnp.linalg.norm(
            galerkin.correction - rank_correction
        ) / jnp.maximum(jnp.linalg.norm(rank_correction), 1e-12)
        recurrence_diagnostics = SolutionRecurrenceDiagnostics(
            error_feedback=galerkin.error_feedback,
            error_feedback_clip_increment=(
                galerkin.error_feedback_clip_increment
            ),
            sample_residual_norm=galerkin.sample_residual_norm,
            correctable_residual_norm=galerkin.correctable_residual_norm,
            correctable_residual_ratio=galerkin.correctable_residual_ratio,
            captured_sample_residual_ratio=(
                galerkin.captured_sample_residual_ratio
            ),
            residual_norm_reduction=galerkin.residual_norm_reduction,
            residual_reduction_fraction=(
                galerkin.residual_reduction_fraction
            ),
            error_feedback_norm=galerkin.error_feedback_norm,
            error_feedback_uncapped_norm=(
                galerkin.error_feedback_uncapped_norm
            ),
            error_feedback_cap_norm=galerkin.error_feedback_cap_norm,
            error_feedback_to_sample_ratio=(
                galerkin.error_feedback_to_sample_ratio
            ),
            error_feedback_to_parameter_conflict_ratio=(
                galerkin.error_feedback_to_parameter_conflict_ratio
            ),
            correction_relative_difference=correction_relative_difference,
        )
        if residual_evaluation == "full_current_batch":
            galerkin_direction = jnp.where(
                result.active_rank > 0,
                galerkin.direction,
                jnp.zeros_like(galerkin.direction),
            )
            result = WSSRSVDResult(
                galerkin_direction, result.state, result.active_rank
            )
    output = (
        unravel_fn(result.grad_like_update),
        result.state,
        result.active_rank,
    )
    if return_gradient_diagnostics:
        current_gradient = o_cur @ e_cur
        if semi_matrix_free_augmented:
            gradient_history = explicit_current_augmented_matvec(
                o_cur,
                state,
                eta,
                e_aug,
                subspace_only_averaging=subspace_only_averaging,
            )
        else:
            gradient_history = o_aug @ e_aug
        output = output + (
            jnp.linalg.norm(current_gradient),
            jnp.linalg.norm(gradient_history),
        )
    if return_recurrence_diagnostics:
        output = output + (recurrence_diagnostics,)
    return output


def compute_wssr_warm_svd_right_transported_gradient_update(
    log_psi_apply: ModelApply[P],
    params: P,
    positions: Array,
    local_energies: Array,
    energy: Array,
    state: WSSRWarmSVDCoreState,
    previous_gradient: Array,
    previous_theta: Array,
    transport_initialized: Array,
    key: Array,
    eta_S: chex.Numeric,
    eta_g: chex.Numeric,
    enable_gradient_transport: bool,
    record_S_lag_diagnostics: bool,
    damping: chex.Numeric,
    norm_constraint: chex.Numeric,
    sr_rank_max: int,
    sr_scale: chex.Numeric = 1.1,
    svd_maxiter_initial: int = 8,
    svd_maxiter_warm: int = 2,
    exact_first: bool = False,
    exact_first_force: bool = False,
    svd_working_rank: Optional[int] = None,
    spectral_regularization: str = "hard_floor",
    complement_weight: chex.Numeric = 1.0,
    relative_singular_value_cutoff: chex.Numeric = -1.0,
    tikhonov_lambda: chex.Numeric = -1.0,
    mixed_precision_solve: bool = False,
):
    """Compute WSSR with independent gradient memory and optional transport."""
    o_cur, unravel_fn = center_and_scale_score_matrix(
        log_psi_apply, params, positions
    )
    e_cur = center_and_scale_energy_residuals(local_energies, energy)
    flat_params, _ = jax.flatten_util.ravel_pytree(params)
    delta_theta = flat_params - previous_theta
    if enable_gradient_transport or record_S_lag_diagnostics:
        # These low-rank actions diagnose S-history lag on the actual parameter
        # displacement. They never materialize the million-dimensional operators.
        history_operator_delta = apply_wssr_history_operator(state, delta_theta)
        current_operator_delta = o_cur @ (o_cur.T @ delta_theta)
    else:
        history_operator_delta = jnp.zeros_like(flat_params)
        current_operator_delta = jnp.zeros_like(flat_params)
    apply_transport = transport_initialized & jnp.asarray(
        enable_gradient_transport
    )
    transport_operator_delta = jnp.where(
        apply_transport,
        history_operator_delta,
        jnp.zeros_like(history_operator_delta),
    )
    current_gradient = o_cur @ e_cur
    gradient_history = transport_gradient_memory(
        previous_gradient,
        current_gradient,
        transport_operator_delta,
        eta_g,
        transport_initialized,
    )

    o_aug, e_aug = augment_wssr_system(o_cur, e_cur, state, eta_S)
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
        exact_first=exact_first,
        exact_first_force=exact_first_force,
        svd_working_rank=svd_working_rank,
        constrain_update_norm=False,
        spectral_regularization=spectral_regularization,
        complement_weight=complement_weight,
        relative_singular_value_cutoff=relative_singular_value_cutoff,
        tikhonov_lambda=tikhonov_lambda,
        experimental_mode="none",
        mixed_precision_solve=mixed_precision_solve,
        force_override=gradient_history,
    )
    gradient_norm = jnp.linalg.norm(current_gradient)
    transport_norm_ema = jnp.linalg.norm(history_operator_delta)
    transport_norm_current = jnp.linalg.norm(current_operator_delta)
    # The operator used this step is eta_S*S_ema_prev+(1-eta_S)*S_current
    # once history exists, so its difference from S_current has this action.
    s_difference_scale = jnp.where(state.sr_rank0 > 0, eta_S, 0.0)
    s_difference_action_norm = jnp.linalg.norm(
        s_difference_scale
        * (current_operator_delta - history_operator_delta)
    )
    delta_theta_norm = jnp.linalg.norm(delta_theta)
    transport_ratio_ema = transport_norm_ema / jnp.maximum(gradient_norm, 1e-12)
    transport_ratio_current = (
        transport_norm_current / jnp.maximum(gradient_norm, 1e-12)
    )
    return (
        unravel_fn(result.grad_like_update),
        result.state,
        result.active_rank,
        gradient_history,
        flat_params,
        transport_norm_ema,
        transport_norm_current,
        gradient_norm,
        jnp.linalg.norm(gradient_history),
        transport_ratio_ema,
        transport_ratio_current,
        s_difference_action_norm,
        delta_theta_norm,
    )


def compute_wssr_cluster_envelope_update(
    log_psi_apply,
    params,
    positions,
    local_energies,
    energy,
    state,
    key,
    damping,
    norm_constraint,
    sr_rank_max,
    envelope_history,
    envelope_count,
    cluster_rank=32,
    envelope_length=3,
    envelope_decay=0.8,
    envelope_capacity=-1,
    complement_alpha=0.2,
    complement_gamma=-1.0,
    envelope_eigenvalue_cutoff=1e-6,
    curvature_mode="scalar",
    enable_cross_batch_snr_shrinkage=False,
    snr_cluster_gap_threshold=0.05,
    enable_grassmann_smoothing=False,
    grassmann_smoothing_alpha=0.5,
    error_feedback_state=None,
    enable_rotation_error_feedback=False,
    rotation_error_feedback_decay=0.95,
    rotation_error_feedback_alpha=0.2,
    rotation_error_feedback_norm_cap=10.0,
    **kwargs,
):
    """Apply a cluster-invariant envelope-preconditioned gradient update.

    The code stores the centered score as ``o_cur = O_bar.T``.  Thus the
    current Fisher action on a parameter-space basis is ``o_cur.T @ U``.
    Recent top-cluster bases are retained as columns, but the enclosing-space
    projection is evaluated through a small Gram pseudoinverse rather than a
    tall QR or an explicit parameter-space projector.
    """
    eps = 1e-12
    o_cur, unravel_fn = center_and_scale_score_matrix(
        log_psi_apply, params, positions
    )
    e_cur = center_and_scale_energy_residuals(local_energies, energy)
    # This candidate deliberately does not average S or g.  The legacy core
    # history remains useful only as the warm SSI initializer.
    o_aug, e_aug = augment_wssr_system(o_cur, e_cur, state, 0.0)
    working = min(
        _resolve_svd_working_rank(
            kwargs.get("svd_working_rank"), sr_rank_max
        ),
        state.sr_o.shape[1],
    )
    u, singular_values, vh, rank = _wssr_right_svd_decomposition(
        o_aug,
        state,
        key,
        working,
        kwargs.get("svd_maxiter_initial", 8),
        kwargs.get("svd_maxiter_warm", 2),
        kwargs.get("exact_first", False),
        eps,
        exact_first_force=kwargs.get("exact_first_force", False),
    )

    k = min(cluster_rank, u.shape[1])
    current_mask = (
        (jnp.arange(k) < rank) & (singular_values[:k] > eps)
    ).astype(u.dtype)
    current_basis = u[:, :k] * current_mask[None, :]
    current_gradient = o_cur @ e_cur
    safe_leading = jnp.maximum(jnp.abs(singular_values[0]), eps)
    _, lambda_reg = _wssr_resolve_spectral_controls(
        safe_leading,
        damping,
        kwargs.get("relative_singular_value_cutoff", -1.0),
        kwargs.get("tikhonov_lambda", -1.0),
        eps,
    )

    history_blocks = max(envelope_length - 1, 0)
    grassmann_overlap_singular_values = jnp.ones((k,), dtype=u.dtype)
    if history_blocks > 0:
        block_ids = jnp.repeat(jnp.arange(history_blocks), k)
        history_mask = (block_ids < envelope_count).astype(u.dtype)
        # The current block has weight one; stored blocks have ages 1, 2, ... .
        history_scale = jnp.power(
            jnp.asarray(envelope_decay, dtype=u.dtype),
            0.5 * (block_ids.astype(u.dtype) + 1.0),
        )
        weighted_history = (
            envelope_history * (history_mask * history_scale)[None, :]
        )
        if envelope_capacity > 0:
            historical_slots = min(
                max(envelope_capacity - k, 0), weighted_history.shape[1]
            )
            (
                selected_history,
                history_pool_numerical_rank,
                selected_history_count,
                selected_history_scores,
                history_selection_gain,
            ) = wssr_experimental.select_history_by_current_value(
                current_basis,
                weighted_history,
                o_cur,
                current_gradient,
                historical_slots,
                lambda_reg,
                envelope_eigenvalue_cutoff,
            )
            envelope_basis = jnp.concatenate(
                [current_basis, selected_history], axis=1
            )
        else:
            history_pool_numerical_rank = jnp.asarray(0, dtype=jnp.int32)
            selected_history_count = jnp.asarray(
                weighted_history.shape[1], dtype=jnp.int32
            )
            selected_history_scores = jnp.zeros(
                (weighted_history.shape[1],), dtype=u.dtype
            )
            history_selection_gain = jnp.asarray(0.0, dtype=u.dtype)
            envelope_basis = jnp.concatenate(
                [current_basis, weighted_history], axis=1
            )
    else:
        weighted_history = jnp.zeros((u.shape[0], 0), dtype=u.dtype)
        history_pool_numerical_rank = jnp.asarray(0, dtype=jnp.int32)
        selected_history_count = jnp.asarray(0, dtype=jnp.int32)
        selected_history_scores = jnp.zeros((0,), dtype=u.dtype)
        history_selection_gain = jnp.asarray(0.0, dtype=u.dtype)
        envelope_basis = current_basis

    if enable_grassmann_smoothing:
        if history_blocks != 1:
            raise ValueError(
                "grassmann_ritz requires cluster_envelope_history=2"
            )
        previous_basis = envelope_history[:, :k]

        def _smooth(_):
            return wssr_experimental.procrustes_grassmann_average(
                previous_basis,
                current_basis,
                grassmann_smoothing_alpha,
            )

        def _current(_):
            return current_basis, jnp.ones((k,), dtype=u.dtype)

        envelope_basis, grassmann_overlap_singular_values = jax.lax.cond(
            envelope_count > 0,
            _smooth,
            _current,
            operand=None,
        )

    orthonormal_envelope, numerical_rank, gram_eigenvalues = (
        wssr_experimental.orthonormalize_basis_envelope(
            envelope_basis,
            envelope_eigenvalue_cutoff,
        )
    )
    envelope_gradient_coordinates = orthonormal_envelope.T @ current_gradient
    cluster_gradient = orthonormal_envelope @ envelope_gradient_coordinates
    perpendicular_gradient = current_gradient - cluster_gradient
    if enable_rotation_error_feedback:
        if error_feedback_state is None:
            raise ValueError(
                "cluster-envelope rotation EF requires an error-feedback state"
            )
        feedback_used = (
            jnp.asarray(rotation_error_feedback_alpha, dtype=u.dtype)
            * jnp.asarray(rotation_error_feedback_decay, dtype=u.dtype)
            * error_feedback_state
        )
    else:
        feedback_used = jnp.zeros_like(current_gradient)
    effective_gradient = current_gradient + feedback_used
    effective_gradient_coordinates = (
        orthonormal_envelope.T @ effective_gradient
    )

    if enable_grassmann_smoothing:
        history_numerical_rank = jnp.sum(
            grassmann_overlap_singular_values > envelope_eigenvalue_cutoff
        ).astype(jnp.int32)
        novelty_fraction = jnp.clip(
            1.0
            - jnp.sum(jnp.square(grassmann_overlap_singular_values))
            / jnp.maximum(jnp.asarray(k, dtype=u.dtype), 1.0),
            0.0,
            1.0,
        )
    elif history_blocks > 0:
        projected_current, history_numerical_rank, _ = (
            wssr_experimental.project_onto_basis_envelope(
                weighted_history,
                current_basis,
                envelope_eigenvalue_cutoff,
            )
        )
        current_energy = jnp.sum(jnp.square(current_basis))
        novelty_fraction = jnp.clip(
            1.0
            - jnp.sum(jnp.square(projected_current))
            / jnp.maximum(current_energy, eps),
            0.0,
            1.0,
        )
    else:
        history_numerical_rank = jnp.asarray(0, dtype=jnp.int32)
        novelty_fraction = jnp.asarray(1.0, dtype=u.dtype)

    current_action = o_cur.T @ current_basis
    curvature_by_mode = jnp.sum(jnp.square(current_action), axis=0)
    active_cluster_columns = jnp.sum(current_mask)
    bar_lambda = jnp.sum(curvature_by_mode * current_mask) / jnp.maximum(
        active_cluster_columns, 1.0
    )
    cluster_denominator = jnp.maximum(bar_lambda + lambda_reg, eps)
    configured_gamma = jnp.asarray(complement_gamma, dtype=u.dtype)
    gamma_used = jnp.where(
        configured_gamma > 0.0, configured_gamma, cluster_denominator
    )
    envelope_width = orthonormal_envelope.shape[1]
    snr_weights = jnp.ones((envelope_width,), dtype=u.dtype)
    snr_cluster_count = jnp.asarray(0, dtype=jnp.int32)
    snr_cross_force_cosine = jnp.asarray(0.0, dtype=u.dtype)
    snr_retained_update_mass = jnp.asarray(1.0, dtype=u.dtype)
    snr_update_cosine_baseline = jnp.asarray(1.0, dtype=u.dtype)
    projected_curvature_eigenvalues = None
    projected_curvature_vectors = None
    if curvature_mode == "scalar":
        cluster_update = cluster_gradient / cluster_denominator
        projected_curvature = jnp.diag(
            jnp.full(
                (orthonormal_envelope.shape[1],),
                bar_lambda,
                dtype=u.dtype,
            )
        )
    elif curvature_mode == "rayleigh_ritz":
        # Rayleigh--Ritz/Galerkin solve on the complete enclosing subspace.
        # ``o_cur`` stores O_bar.T, hence W = O_bar @ Q is o_cur.T @ Q.
        # This retains current-batch anisotropic Fisher curvature without
        # materializing either S or a parameter-space projector.
        envelope_action = o_cur.T @ orthonormal_envelope
        projected_curvature = envelope_action.T @ envelope_action
        projected_rhs = envelope_action.T @ e_cur + (
            orthonormal_envelope.T @ feedback_used
        )
        if enable_cross_batch_snr_shrinkage:
            projected_curvature_eigenvalues, projected_curvature_vectors = (
                jnp.linalg.eigh(
                    0.5 * (projected_curvature + projected_curvature.T)
                )
            )
            sample_ids = jnp.arange(e_cur.shape[0])
            mask_a = (sample_ids % 2 == 0).astype(e_cur.dtype)
            mask_b = 1.0 - mask_a
            scale_a = e_cur.shape[0] / jnp.maximum(jnp.sum(mask_a), 1.0)
            scale_b = e_cur.shape[0] / jnp.maximum(jnp.sum(mask_b), 1.0)
            half_rhs_a = scale_a * (envelope_action.T @ (e_cur * mask_a))
            half_rhs_b = scale_b * (envelope_action.T @ (e_cur * mask_b))
            force_a = projected_curvature_vectors.T @ half_rhs_a
            force_b = projected_curvature_vectors.T @ half_rhs_b
            (
                snr_weights,
                _,
                snr_cluster_count,
                _,
                _,
            ) = wssr_experimental.cluster_cross_batch_snr_weights(
                projected_curvature_eigenvalues,
                force_a,
                force_b,
                snr_cluster_gap_threshold,
                lambda_reg,
                eps,
            )
            baseline_eigen_coefficients = (
                projected_curvature_vectors.T @ projected_rhs
            ) / jnp.maximum(
                projected_curvature_eigenvalues + lambda_reg, eps
            )
            weighted_eigen_coefficients = (
                snr_weights * baseline_eigen_coefficients
            )
            envelope_coefficients = (
                projected_curvature_vectors @ weighted_eigen_coefficients
            )
            snr_retained_update_mass = jnp.sum(
                jnp.square(weighted_eigen_coefficients)
            ) / jnp.maximum(
                jnp.sum(jnp.square(baseline_eigen_coefficients)), eps
            )
            snr_update_cosine_baseline = jnp.vdot(
                baseline_eigen_coefficients, weighted_eigen_coefficients
            ) / jnp.maximum(
                jnp.linalg.norm(baseline_eigen_coefficients)
                * jnp.linalg.norm(weighted_eigen_coefficients),
                eps,
            )
            snr_cross_force_cosine = jnp.vdot(force_a, force_b) / jnp.maximum(
                jnp.linalg.norm(force_a) * jnp.linalg.norm(force_b), eps
            )
        else:
            envelope_coefficients = _device_galerkin_solution(
                envelope_action,
                projected_rhs,
                jnp.zeros_like(projected_rhs),
                lambda_reg,
            )
        cluster_update = orthonormal_envelope @ envelope_coefficients
    else:
        raise ValueError(
            "cluster_envelope_curvature_mode must be scalar or rayleigh_ritz"
        )
    complement_update = (
        jnp.asarray(complement_alpha, dtype=u.dtype)
        * perpendicular_gradient
        / jnp.maximum(gamma_used, eps)
    )
    direction = cluster_update + complement_update

    if enable_rotation_error_feedback:
        unapplied_force = effective_gradient - (
            orthonormal_envelope @ effective_gradient_coordinates
        )
        unapplied_force_norm = jnp.linalg.norm(unapplied_force)
        feedback_cap_norm = jnp.asarray(
            rotation_error_feedback_norm_cap, dtype=u.dtype
        ) * jnp.linalg.norm(current_gradient)
        feedback_scale = jnp.minimum(
            1.0,
            feedback_cap_norm / jnp.maximum(unapplied_force_norm, eps),
        )
        next_error_feedback = feedback_scale * unapplied_force
        feedback_clip_increment = (
            unapplied_force_norm > feedback_cap_norm
        ).astype(jnp.int32)
    else:
        next_error_feedback = jnp.zeros_like(current_gradient)
        unapplied_force_norm = jnp.asarray(0.0, dtype=u.dtype)
        feedback_cap_norm = jnp.asarray(0.0, dtype=u.dtype)
        feedback_clip_increment = jnp.asarray(0, dtype=jnp.int32)

    # Reuse the production state transition so warm-start SSI behavior remains
    # unchanged. Its update is intentionally ignored by this experimental arm.
    base = _wssr_update_from_svd(
        o_aug,
        e_aug,
        state,
        u,
        singular_values,
        vh,
        damping,
        norm_constraint,
        state.sr_o.shape[1],
        kwargs.get("sr_scale", 1.1),
        False,
        rank_update_max=working,
        spectral_regularization=kwargs.get(
            "spectral_regularization", "tikhonov"
        ),
        complement_weight=0.0,
        relative_singular_value_cutoff=kwargs.get(
            "relative_singular_value_cutoff", -1.0
        ),
        tikhonov_lambda=kwargs.get("tikhonov_lambda", -1.0),
        experimental_mode="none",
    )
    u_state = jnp.zeros_like(state.u)
    warm_width = min(u.shape[1], state.u.shape[1])
    warm_mask = (jnp.arange(warm_width) < rank).astype(u.dtype)
    u_state = u_state.at[:, :warm_width].set(
        u[:, :warm_width] * warm_mask[None, :]
    )
    warm_state = WSSRWarmSVDCoreState(
        base.state.sr_o,
        base.state.ek,
        base.state.sr_rank0,
        base.state.sr_rank,
        u_state,
        jnp.where(rank > 0, jnp.array(True), state.has_u),
    )

    if enable_grassmann_smoothing:
        next_history = jnp.zeros_like(envelope_history)
        next_history = next_history.at[:, :k].set(
            orthonormal_envelope[:, :k]
        )
        next_count = jnp.asarray(1, dtype=envelope_count.dtype)
    elif history_blocks > 0:
        retained_history = envelope_history[:, : max(history_blocks - 1, 0) * k]
        next_history = jnp.concatenate(
            [current_basis, retained_history], axis=1
        )
        next_count = jnp.minimum(
            envelope_count + 1,
            jnp.asarray(history_blocks, dtype=envelope_count.dtype),
        )
    else:
        next_history = envelope_history
        next_count = envelope_count

    gradient_norm = jnp.linalg.norm(current_gradient)
    cluster_gradient_norm = jnp.linalg.norm(cluster_gradient)
    perpendicular_gradient_norm = jnp.linalg.norm(perpendicular_gradient)
    cluster_update_norm = jnp.linalg.norm(cluster_update)
    complement_update_norm = jnp.linalg.norm(complement_update)
    if projected_curvature_eigenvalues is None:
        projected_curvature_eigenvalues, projected_curvature_vectors = (
            jnp.linalg.eigh(0.5 * (projected_curvature + projected_curvature.T))
        )
    projected_curvature_min = jnp.min(projected_curvature_eigenvalues)
    projected_curvature_max = jnp.max(projected_curvature_eigenvalues)
    if curvature_mode == "rayleigh_ritz":
        projected_update_eigen_coordinates = (
            projected_curvature_vectors.T @ projected_rhs
        ) / jnp.maximum(projected_curvature_eigenvalues + lambda_reg, eps)
        boundary_width = max(
            projected_curvature_eigenvalues.shape[0] // 10, 1
        )
        # eigh is ascending: the first modes are the weak-curvature boundary
        # of the retained current/envelope subspace.
        boundary_update_mass = jnp.sum(
            jnp.square(projected_update_eigen_coordinates[:boundary_width])
        ) / jnp.maximum(
            jnp.sum(jnp.square(projected_update_eigen_coordinates)), eps
        )
        snr_weak_boundary_weight = jnp.mean(snr_weights[:boundary_width])
        snr_strong_boundary_weight = jnp.mean(snr_weights[-boundary_width:])
    else:
        boundary_update_mass = jnp.asarray(0.0, dtype=u.dtype)
        snr_weak_boundary_weight = jnp.asarray(1.0, dtype=u.dtype)
        snr_strong_boundary_weight = jnp.asarray(1.0, dtype=u.dtype)
    finite_selected_scores = jnp.where(
        jnp.isfinite(selected_history_scores),
        selected_history_scores,
        jnp.inf,
    )
    selected_score_min = jnp.min(finite_selected_scores, initial=jnp.inf)
    selected_score_min = jnp.where(
        jnp.isfinite(selected_score_min), selected_score_min, 0.0
    )
    diagnostics = {
        "wssr_envelope_numerical_rank": numerical_rank,
        "wssr_envelope_capacity": jnp.asarray(
            envelope_basis.shape[1], dtype=jnp.int32
        ),
        "wssr_envelope_history_count": envelope_count,
        "wssr_envelope_history_numerical_rank": history_numerical_rank,
        "wssr_envelope_history_pool_numerical_rank": (
            history_pool_numerical_rank
        ),
        "wssr_envelope_selected_history_count": selected_history_count,
        "wssr_envelope_selected_history_score_max": jnp.max(
            selected_history_scores, initial=0.0
        ),
        "wssr_envelope_selected_history_score_min": selected_score_min,
        "wssr_envelope_history_selection_gain": history_selection_gain,
        "wssr_envelope_spectral_tail_ratio": (
            singular_values[k - 1]
            / jnp.maximum(singular_values[k], eps)
            if singular_values.shape[0] > k
            else jnp.asarray(0.0, dtype=u.dtype)
        ),
        "wssr_envelope_spectral_tail_available": jnp.asarray(
            singular_values.shape[0] > k, dtype=jnp.int32
        ),
        "wssr_envelope_current_novelty_fraction": novelty_fraction,
        "wssr_grassmann_alpha": jnp.asarray(
            grassmann_smoothing_alpha, dtype=u.dtype
        ),
        "wssr_grassmann_history_active": (
            (envelope_count > 0) & enable_grassmann_smoothing
        ).astype(jnp.int32),
        "wssr_grassmann_overlap_mean": jnp.mean(
            grassmann_overlap_singular_values
        ),
        "wssr_grassmann_overlap_min": jnp.min(
            grassmann_overlap_singular_values
        ),
        "wssr_envelope_gradient_capture_fraction": (
            cluster_gradient_norm / jnp.maximum(gradient_norm, eps)
        ),
        "wssr_envelope_bar_lambda": bar_lambda,
        "wssr_envelope_lambda": lambda_reg,
        "wssr_envelope_gamma": gamma_used,
        "wssr_envelope_gradient_norm": gradient_norm,
        "wssr_envelope_cluster_gradient_norm": cluster_gradient_norm,
        "wssr_envelope_perpendicular_gradient_norm": perpendicular_gradient_norm,
        "wssr_envelope_cluster_update_norm": cluster_update_norm,
        "wssr_envelope_complement_update_norm": complement_update_norm,
        "wssr_envelope_complement_cluster_ratio": (
            complement_update_norm / jnp.maximum(cluster_update_norm, eps)
        ),
        "wssr_envelope_smallest_gram_eigenvalue": jnp.min(gram_eigenvalues),
        "wssr_envelope_projected_curvature_min": projected_curvature_min,
        "wssr_envelope_projected_curvature_max": projected_curvature_max,
        "wssr_envelope_projected_curvature_condition": (
            (projected_curvature_max + lambda_reg)
            / jnp.maximum(projected_curvature_min + lambda_reg, eps)
        ),
        "wssr_envelope_boundary_update_mass": boundary_update_mass,
        "wssr_envelope_snr_cluster_count": snr_cluster_count,
        "wssr_envelope_snr_weight_min": jnp.min(snr_weights),
        "wssr_envelope_snr_weight_mean": jnp.mean(snr_weights),
        "wssr_envelope_snr_weight_max": jnp.max(snr_weights),
        "wssr_envelope_snr_weak_boundary_weight": snr_weak_boundary_weight,
        "wssr_envelope_snr_strong_boundary_weight": (
            snr_strong_boundary_weight
        ),
        "wssr_envelope_snr_cross_force_cosine": snr_cross_force_cosine,
        "wssr_envelope_snr_retained_update_mass": snr_retained_update_mass,
        "wssr_envelope_snr_update_cosine_baseline": (
            snr_update_cosine_baseline
        ),
        "wssr_envelope_ef_used_norm": jnp.linalg.norm(feedback_used),
        "wssr_envelope_ef_state_norm": jnp.linalg.norm(next_error_feedback),
        "wssr_envelope_ef_uncapped_norm": unapplied_force_norm,
        "wssr_envelope_ef_cap_norm": feedback_cap_norm,
        "wssr_envelope_ef_clip_increment": feedback_clip_increment,
        "wssr_envelope_ef_to_gradient_ratio": (
            jnp.linalg.norm(next_error_feedback)
            / jnp.maximum(gradient_norm, eps)
        ),
        "wssr_diag_raw_direction_norm": jnp.linalg.norm(direction),
        "wssr_diag_finite": jnp.all(jnp.isfinite(direction)).astype(jnp.int32),
    }
    return (
        unravel_fn(direction),
        warm_state,
        base.active_rank,
        next_history,
        next_count,
        next_error_feedback,
        feedback_clip_increment,
        diagnostics,
    )


def compute_wssr_warm_svd_right_experimental_diagnostic_update(
    log_psi_apply, params, positions, local_energies, energy, state, key,
    eta, damping, norm_constraint, sr_rank_max, **kwargs
):
    """Experimental update plus cheap same-operator diagnostics (no exact SVD)."""
    o_cur, unravel_fn = center_and_scale_score_matrix(log_psi_apply, params, positions)
    e_cur = center_and_scale_energy_residuals(local_energies, energy)
    o_aug, e_aug = augment_wssr_system(o_cur, e_cur, state, eta)
    working = min(_resolve_svd_working_rank(kwargs.get('svd_working_rank'), sr_rank_max), state.sr_o.shape[1])
    u, s, vh, rank = _wssr_right_svd_decomposition(
        o_aug, state, key, working, kwargs.get('svd_maxiter_initial',8),
        kwargs.get('svd_maxiter_warm',2), kwargs.get('exact_first',False), 1e-12,
        exact_first_force=kwargs.get('exact_first_force',False)
    )
    mode=kwargs.get('experimental_mode','none')
    wssr_experimental.validate_mode(mode)
    if mode=='force_aware':
        target=working if kwargs.get('experimental_target_rank',-1)<=0 else min(kwargs['experimental_target_rank'],working)
        u,s,vh,_,_=wssr_experimental.force_aware_rayleigh_ritz(
            o_aug,u,vh,o_aug@e_aug,kwargs.get('force_aware_krylov_vectors',1),target)
        rank=jnp.asarray(target,dtype=state.sr_rank.dtype)
    # The proximal candidate starts from the exact production warm2 update;
    # only the subsequent coefficient transform is experimental.
    base_mode='none' if mode=='native_proximal' else mode
    post=dict(sr_scale=kwargs.get('sr_scale',1.1),constrain_update_norm=False,
      rank_update_max=working,spectral_regularization=kwargs.get('spectral_regularization','hard_floor'),
      complement_weight=kwargs.get('complement_weight',1.0),experimental_mode=base_mode,
      relative_singular_value_cutoff=kwargs.get('relative_singular_value_cutoff',-1.0),
      tikhonov_lambda=kwargs.get('tikhonov_lambda',-1.0),
      experimental_target_rank=kwargs.get('experimental_target_rank',-1),
      cluster_gap_threshold=kwargs.get('cluster_gap_threshold',.002),near_tail_modes=kwargs.get('near_tail_modes',0),
      adaptive_complement_beta=kwargs.get('adaptive_complement_beta',0.),
      adaptive_complement_beta_function=kwargs.get('adaptive_complement_beta_function',0.),
      complement_function_operator=o_cur,
      smooth_transition_start=kwargs.get('smooth_transition_start',-1),
      smooth_transition_end=kwargs.get('smooth_transition_end',-1),iterative_complement_iterations=kwargs.get('iterative_complement_iterations',0))
    base=_wssr_update_from_svd(o_aug,e_aug,state,u,s,vh,damping,norm_constraint,state.sr_o.shape[1],**post)
    resolved=base.grad_like_update
    near_tail=jnp.zeros_like(base.grad_like_update)
    unscaled_near_tail=jnp.zeros_like(base.grad_like_update)
    if mode in ('adaptive_complement','residual_optimal_complement'):
        resolved=_wssr_update_from_svd(o_aug,e_aug,state,u,s,vh,damping,norm_constraint,state.sr_o.shape[1],**{**post,'adaptive_complement_beta':0.}).grad_like_update
    elif mode=='iterative_complement':
        resolved=_wssr_update_from_svd(o_aug,e_aug,state,u,s,vh,damping,norm_constraint,state.sr_o.shape[1],**{**post,'iterative_complement_iterations':0}).grad_like_update
    elif mode in ('near_tail','capped_near_tail'):
        target=working if kwargs.get('experimental_target_rank',-1)<=0 else min(kwargs['experimental_target_rank'],working)
        leading=jnp.maximum(jnp.abs(s[0]),1e-12)
        relative_cutoff,_=_wssr_resolve_spectral_controls(
            leading,damping,kwargs.get('relative_singular_value_cutoff',-1.0),
            kwargs.get('tikhonov_lambda',-1.0))
        retained=(s/leading>relative_cutoff)
        inv_cap,_=_wssr_inverse_spectral_coefficients(
            jnp.where(retained,s,1.),leading,jnp.maximum(jnp.abs(damping),1e-12),
            kwargs.get('spectral_regularization','hard_floor'),0.,
            tikhonov_lambda=kwargs.get('tikhonov_lambda',-1.0))
        projected=u.T@(o_aug@e_aug)
        main_mask=retained&(jnp.arange(s.shape[0])<target)
        tail_mask=retained&(jnp.arange(s.shape[0])>=target)&(jnp.arange(s.shape[0])<target+kwargs.get('near_tail_modes',0))
        resolved=u@(projected*inv_cap*main_mask.astype(u.dtype))
        near_tail=u@(projected*inv_cap*tail_mask.astype(u.dtype))
        unscaled_near_tail=near_tail
        if mode=='capped_near_tail':
            near_tail,_=wssr_experimental.cap_contribution(
                resolved,near_tail,kwargs.get('adaptive_complement_beta',0.))
    complement_ema = kwargs.get("complement_ema", None)
    proximal_state = None
    proximal_previous_cosine = jnp.asarray(0.0, dtype=base.grad_like_update.dtype)
    proximal_native_relative_change = jnp.asarray(
        0.0, dtype=base.grad_like_update.dtype
    )
    if mode == "native_proximal":
        if complement_ema is None:
            raise ValueError("native_proximal requires a previous-update state")
        leading = jnp.maximum(jnp.abs(s[0]), 1e-12)
        relative_cutoff, regularizer = _wssr_resolve_spectral_controls(
            leading,
            damping,
            kwargs.get("relative_singular_value_cutoff", -1.0),
            kwargs.get("tikhonov_lambda", -1.0),
        )
        native_update = base.grad_like_update
        proximal_update = wssr_experimental.native_factor_proximal_update(
            native_update,
            u,
            s,
            complement_ema,
            kwargs.get("native_proximal_gamma", 0.0),
            regularizer,
            relative_cutoff,
        )
        base = base._replace(grad_like_update=proximal_update)
        resolved = proximal_update
        proximal_state = proximal_update
        proximal_previous_cosine = jnp.vdot(
            proximal_update, complement_ema
        ) / jnp.maximum(
            jnp.linalg.norm(proximal_update) * jnp.linalg.norm(complement_ema),
            1e-12,
        )
        proximal_native_relative_change = jnp.linalg.norm(
            proximal_update - native_update
        ) / jnp.maximum(jnp.linalg.norm(native_update), 1e-12)
    comp=base.grad_like_update-resolved
    multilevel_ema_b = kwargs.get("multilevel_ema_b", None)
    complement_state_decay = kwargs.get("complement_state_decay", 0.0)
    complement_state_relative_cap = kwargs.get(
        "complement_state_relative_cap", 0.0
    )
    multilevel_state = None
    if multilevel_ema_b is not None:
        rho = jnp.asarray(complement_state_decay, dtype=comp.dtype)
        one_minus_rho = 1.0 - rho
        weight_a = kwargs["multilevel_weight_a"]
        weight_b = kwargs["multilevel_weight_b"]
        step = kwargs["multilevel_step"]
        update_a = (step % 2) == 0
        ema_a = jnp.where(
            update_a,
            rho * complement_ema + one_minus_rho * comp,
            complement_ema,
        )
        ema_b = jnp.where(
            update_a,
            multilevel_ema_b,
            rho * multilevel_ema_b + one_minus_rho * comp,
        )
        weight_a = jnp.where(
            update_a, rho * weight_a + one_minus_rho, weight_a
        )
        weight_b = jnp.where(
            update_a, weight_b, rho * weight_b + one_minus_rho
        )
        corrected_a = ema_a / jnp.maximum(weight_a, 1e-12)
        corrected_b = ema_b / jnp.maximum(weight_b, 1e-12)
        cosine = jnp.vdot(corrected_a, corrected_b) / jnp.maximum(
            jnp.linalg.norm(corrected_a) * jnp.linalg.norm(corrected_b), 1e-12
        )
        period = kwargs["multilevel_apply_period"]
        at_boundary = ((step + 1) % period) == 0
        accepted = at_boundary & (cosine > kwargs["multilevel_cosine_threshold"])
        candidate = 0.5 * (corrected_a + corrected_b)
        candidate, _ = wssr_experimental.cap_contribution(
            resolved,
            candidate,
            kwargs.get("adaptive_complement_beta", 0.0),
        )
        applied = jnp.where(accepted, candidate, jnp.zeros_like(candidate))
        base = base._replace(grad_like_update=resolved + applied)
        comp = applied
        reset = at_boundary
        ema_a = jnp.where(reset, jnp.zeros_like(ema_a), ema_a)
        ema_b = jnp.where(reset, jnp.zeros_like(ema_b), ema_b)
        weight_a = jnp.where(reset, jnp.zeros_like(weight_a), weight_a)
        weight_b = jnp.where(reset, jnp.zeros_like(weight_b), weight_b)
        multilevel_state = (ema_a, ema_b, weight_a, weight_b, step + 1)
    elif complement_ema is not None and mode != "native_proximal":
        rho = jnp.asarray(complement_state_decay, dtype=comp.dtype)
        complement_ema = rho * complement_ema + (1.0 - rho) * comp
        if complement_state_relative_cap > 0:
            complement_ema, _ = wssr_experimental.cap_contribution(
                resolved,
                complement_ema,
                complement_state_relative_cap,
            )
        base = base._replace(grad_like_update=resolved + complement_ema)
        comp = complement_ema
    adaptive_alpha=jnp.asarray(0.0,dtype=base.grad_like_update.dtype)
    adaptive_unclamped_norm=jnp.asarray(0.0,dtype=base.grad_like_update.dtype)
    adaptive_cap_active=jnp.asarray(0,dtype=jnp.int32)
    adaptive_requested_beta=jnp.asarray(
        kwargs.get('adaptive_complement_beta',0.),dtype=base.grad_like_update.dtype)
    residual_before=jnp.asarray(0.0,dtype=base.grad_like_update.dtype)
    residual_after=jnp.asarray(0.0,dtype=base.grad_like_update.dtype)
    residual_alpha_star=jnp.asarray(0.0,dtype=base.grad_like_update.dtype)
    if mode=='adaptive_complement':
        leading=jnp.maximum(jnp.abs(s[0]),1e-12)
        relative_cutoff,_=_wssr_resolve_spectral_controls(
            leading,damping,kwargs.get('relative_singular_value_cutoff',-1.0),
            kwargs.get('tikhonov_lambda',-1.0))
        safe_s=jnp.where(s/leading>relative_cutoff,s,1.)
        _,alpha_nominal=_wssr_inverse_spectral_coefficients(
            safe_s,leading,jnp.maximum(jnp.abs(damping),1e-12),
            kwargs.get('spectral_regularization','hard_floor'),
            kwargs.get('complement_weight',1.0),
            tikhonov_lambda=kwargs.get('tikhonov_lambda',-1.0))
        projected=u.T@(o_aug@e_aug)
        retained=(s/leading>relative_cutoff).astype(u.dtype)
        perp_force=(o_aug@e_aug)-u@(projected*retained)
        _, adaptive_alpha = wssr_experimental.adaptive_complement(
            resolved,
            perp_force,
            alpha_nominal,
            adaptive_requested_beta,
            function_operator=o_cur,
            beta_function=kwargs.get(
                'adaptive_complement_beta_function', 0.0
            ),
        )
        adaptive_unclamped_norm=alpha_nominal*jnp.linalg.norm(perp_force)
        adaptive_cap_active=(adaptive_alpha<alpha_nominal).astype(jnp.int32)
    elif mode=='residual_optimal_complement':
        leading=jnp.maximum(jnp.abs(s[0]),1e-12)
        relative_cutoff,regularizer=_wssr_resolve_spectral_controls(
            leading,damping,kwargs.get('relative_singular_value_cutoff',-1.0),
            kwargs.get('tikhonov_lambda',-1.0))
        retained=(s/leading>relative_cutoff).astype(u.dtype)
        projected=u.T@(o_aug@e_aug)
        perp_force=(o_aug@e_aug)-u@(projected*retained)
        _,adaptive_alpha,residual_alpha_star,alpha_cap,rb,ra=(
            wssr_experimental.residual_optimal_complement(
                o_aug,o_aug@e_aug,resolved,perp_force,regularizer,
                adaptive_requested_beta))
        adaptive_unclamped_norm=residual_alpha_star*jnp.linalg.norm(perp_force)
        adaptive_cap_active=(adaptive_alpha<residual_alpha_star).astype(jnp.int32)
        residual_before=jnp.linalg.norm(rb)
        residual_after=jnp.linalg.norm(ra)
    elif mode=='capped_near_tail':
        adaptive_alpha=jnp.linalg.norm(near_tail)/jnp.maximum(
            jnp.linalg.norm(unscaled_near_tail),1e-30)
        adaptive_unclamped_norm=jnp.linalg.norm(unscaled_near_tail)
        adaptive_cap_active=(adaptive_alpha<1.0).astype(jnp.int32)
    u_state=jnp.zeros_like(state.u);width=min(u.shape[1],state.u.shape[1]);mask=(jnp.arange(width)<rank).astype(u.dtype);u_state=u_state.at[:,:width].set(u[:,:width]*mask)
    warm=WSSRWarmSVDCoreState(base.state.sr_o,base.state.ek,base.state.sr_rank0,base.state.sr_rank,u_state,jnp.where(rank>0,jnp.array(True),state.has_u))
    reliability_diagnostics = kwargs.get("reliability_diagnostics", False)
    if reliability_diagnostics:
        # Weight the approximate singular-pair residual by the direction that
        # is actually sent to Optax.  Unlike a full block residual O^T U-VS,
        # this is a pair of matrix-vector products and emphasizes errors that
        # can influence the current parameter update.
        update_coefficients = u.T @ base.grad_like_update
        ritz_actual = o_aug.T @ base.grad_like_update
        ritz_model = vh.T @ (s * update_coefficients)
        update_weighted_ritz_residual = jnp.linalg.norm(
            ritz_actual - ritz_model
        ) / jnp.maximum(jnp.linalg.norm(ritz_actual), 1e-12)

        # Measure whether the current update points outside the previously
        # stored warm subspace.  This is basis-invariant when the active warm
        # columns are orthonormal and is explicitly weighted by the update.
        update_norm_sq = jnp.square(jnp.linalg.norm(base.grad_like_update))
        if state.u.shape[1] > 0:
            previous_width = min(state.u.shape[1], state.sr_o.shape[1])
            previous_mask = (
                jnp.arange(previous_width) < state.sr_rank0
            ).astype(state.u.dtype)
            previous_projection = (
                state.u[:, :previous_width].T @ base.grad_like_update
            ) * previous_mask
            retained_fraction = jnp.clip(
                jnp.sum(jnp.square(previous_projection))
                / jnp.maximum(update_norm_sq, 1e-24),
                0.0,
                1.0,
            )
            has_previous_subspace = state.has_u | (state.sr_rank0 > 0)
            retained_fraction = jnp.where(
                has_previous_subspace, retained_fraction, 0.0
            )
        else:
            retained_fraction = jnp.asarray(
                0.0, dtype=base.grad_like_update.dtype
            )
        temporal_update_novelty = jnp.sqrt(
            jnp.maximum(1.0 - retained_fraction, 0.0)
        )

        # Compare the production warm-SSI result with the same decomposition
        # using exactly one fewer iteration.  The alternative is diagnostic
        # only: neither its update nor its warm state is returned.
        comparison_warm_iterations = max(
            kwargs.get("svd_maxiter_warm", 2) - 1, 0
        )
        alt_u, alt_s, alt_vh, _ = _wssr_right_svd_decomposition(
            o_aug,
            state,
            key,
            working,
            kwargs.get("svd_maxiter_initial", 8),
            comparison_warm_iterations,
            kwargs.get("exact_first", False),
            1e-12,
            exact_first_force=kwargs.get("exact_first_force", False),
        )
        alt_base = _wssr_update_from_svd(
            o_aug,
            e_aug,
            state,
            alt_u,
            alt_s,
            alt_vh,
            damping,
            norm_constraint,
            state.sr_o.shape[1],
            **post,
        )
        alt_update = alt_base.grad_like_update
        base_update_norm = jnp.linalg.norm(base.grad_like_update)
        alt_update_norm = jnp.linalg.norm(alt_update)
        ssi_update_relative_change = jnp.linalg.norm(
            base.grad_like_update - alt_update
        ) / jnp.maximum(base_update_norm, 1e-12)
        ssi_update_cosine = jnp.vdot(
            base.grad_like_update, alt_update
        ) / jnp.maximum(base_update_norm * alt_update_norm, 1e-12)
    elif mode == "native_proximal":
        # Do not form the diagnostic block residual ``O.T @ U`` in the
        # production candidate: at rank 1600 it is much more expensive than
        # the proximal coefficient transform and is not used by the update.
        update_weighted_ritz_residual = jnp.asarray(
            0.0, dtype=base.grad_like_update.dtype
        )
    else:
        # Preserve the existing block-residual definition for legacy
        # update_diagnostics and experimental-mode reports.
        lhs = o_aug.T @ u
        rhs = vh.T * s
        update_weighted_ritz_residual = jnp.linalg.norm(
            lhs - rhs
        ) / jnp.maximum(jnp.linalg.norm(lhs), 1e-12)

    diag={'wssr_diag_raw_direction_norm':jnp.linalg.norm(base.grad_like_update),
      'wssr_diag_raw_direction_max_abs':jnp.max(jnp.abs(base.grad_like_update)),
      'wssr_diag_subspace_residual':update_weighted_ritz_residual,
      'wssr_diag_resolved_norm':jnp.linalg.norm(resolved),'wssr_diag_complement_norm':jnp.linalg.norm(comp),
      'wssr_diag_near_tail_norm':jnp.linalg.norm(near_tail),
      'wssr_diag_complement_resolved_ratio':jnp.linalg.norm(comp)/jnp.maximum(jnp.linalg.norm(resolved),1e-12),
      'wssr_diag_complement_state_decay':jnp.asarray(complement_state_decay,dtype=base.grad_like_update.dtype),
      'wssr_diag_native_proximal_gamma':jnp.asarray(kwargs.get('native_proximal_gamma',0.),dtype=base.grad_like_update.dtype),
      'wssr_diag_native_proximal_previous_cosine':proximal_previous_cosine,
      'wssr_diag_native_proximal_relative_change':proximal_native_relative_change,
      'wssr_diag_adaptive_requested_beta':adaptive_requested_beta,
      'wssr_diag_adaptive_alpha':adaptive_alpha,
      'wssr_diag_adaptive_unclamped_complement_norm':adaptive_unclamped_norm,
      'wssr_diag_adaptive_cap_active':adaptive_cap_active,
      'wssr_diag_residual_alpha_star':residual_alpha_star,
      'wssr_diag_residual_before':residual_before,
      'wssr_diag_residual_after':residual_after,
      'wssr_diag_finite':jnp.all(jnp.isfinite(base.grad_like_update)).astype(jnp.int32)}
    if reliability_diagnostics:
        diag.update({
            'wssr_reliability_update_weighted_ritz_residual': (
                update_weighted_ritz_residual
            ),
            'wssr_reliability_previous_subspace_retained_fraction': (
                retained_fraction
            ),
            'wssr_reliability_temporal_update_novelty': (
                temporal_update_novelty
            ),
            'wssr_reliability_ssi_comparison_iterations': jnp.asarray(
                comparison_warm_iterations, dtype=jnp.int32
            ),
            'wssr_reliability_ssi_update_relative_change': (
                ssi_update_relative_change
            ),
            'wssr_reliability_ssi_update_cosine': ssi_update_cosine,
        })
    if multilevel_ema_b is not None:
        diag.update({
            'wssr_diag_multilevel_cosine': cosine,
            'wssr_diag_multilevel_boundary':at_boundary.astype(jnp.int32),
            'wssr_diag_multilevel_accepted':accepted.astype(jnp.int32),
        })
    if kwargs.get("burst_diagnostics_payload", False):
        diag.update(
            {
                "burst_diag_raw_direction": base.grad_like_update,
                "burst_diag_resolved_update": resolved,
                "burst_diag_complement_update": comp,
                "burst_diag_near_tail_update": near_tail,
                "burst_diag_complement_resolved_ratio": jnp.linalg.norm(comp)
                / jnp.maximum(jnp.linalg.norm(resolved), 1e-12),
            }
        )
    returned_state = (
        proximal_state
        if proximal_state is not None
        else (multilevel_state if multilevel_state is not None else complement_ema)
    )
    return unravel_fn(base.grad_like_update),warm,base.active_rank,diag,returned_state


def compute_wssr_warm_svd_right_reference_diagnostic_update(
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
    exact_first: bool = False,
    compute_exact_reference: bool = True,
    svd_working_rank: Optional[int] = None,
    spectral_regularization: str = "hard_floor",
    complement_weight: chex.Numeric = 1.0,
    relative_singular_value_cutoff: chex.Numeric = -1.0,
    tikhonov_lambda: chex.Numeric = -1.0,
):
    """Compute the normal update plus exact-reference diagnostics."""
    o_cur, unravel_fn = center_and_scale_score_matrix(log_psi_apply, params, positions)
    e_cur = center_and_scale_energy_residuals(local_energies, energy)
    o_aug, e_aug = augment_wssr_system(o_cur, e_cur, state, eta)
    result, diagnostics = wssr_warm_svd_right_reference_diagnostic_core_update(
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
        exact_first=exact_first,
        compute_exact_reference=compute_exact_reference,
        svd_working_rank=svd_working_rank,
        spectral_regularization=spectral_regularization,
        complement_weight=complement_weight,
        relative_singular_value_cutoff=relative_singular_value_cutoff,
        tikhonov_lambda=tikhonov_lambda,
    )
    return (
        unravel_fn(result.grad_like_update),
        result.state,
        result.active_rank,
        diagnostics,
    )


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
    relative_singular_value_cutoff: chex.Numeric = -1.0,
    tikhonov_lambda: chex.Numeric = -1.0,
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
        relative_singular_value_cutoff=relative_singular_value_cutoff,
        tikhonov_lambda=tikhonov_lambda,
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
    norm_constraint_mode = optimizer_config.get(
        "norm_constraint_mode", "euclidean"
    )
    if norm_constraint_mode not in ("euclidean", "function_space"):
        raise ValueError(
            "WSSR norm_constraint_mode must be euclidean or function_space"
        )
    function_norm_constraint = optimizer_config.get(
        "function_norm_constraint", 0.001
    )
    if function_norm_constraint <= 0.0:
        raise ValueError("function_norm_constraint must be positive")
    euclidean_safety_constraint = optimizer_config.get(
        "euclidean_safety_constraint", -1.0
    )
    if euclidean_safety_constraint == 0.0:
        raise ValueError(
            "euclidean_safety_constraint must be positive or negative to disable"
        )
    adaptive_complement_beta_function = optimizer_config.get(
        "adaptive_complement_beta_function", 0.0
    )
    if adaptive_complement_beta_function < 0.0:
        raise ValueError("adaptive_complement_beta_function must be nonnegative")
    if (
        adaptive_complement_beta_function > 0.0
        and optimizer_config.get("experimental_mode", "none")
        != "adaptive_complement"
    ):
        raise ValueError(
            "function-space complement cap requires adaptive_complement mode"
        )

    _validate_wssr_spectral_regularization(
        optimizer_config.get("spectral_regularization", "hard_floor"),
        optimizer_config.get("complement_weight", 1.0),
    )
    enable_gradient_transport = optimizer_config.get(
        "enable_gradient_transport", False
    )
    adaptive_S_average = optimizer_config.get("adaptive_S_average", False)
    adaptive_g_average = optimizer_config.get("adaptive_g_average", False)
    eta_S_schedule = optimizer_config.get("eta_S_schedule", "constant")
    if adaptive_S_average:
        if eta_S_schedule not in (
            "constant",
            "linear_warmup",
            "exponential_growth",
        ):
            raise ValueError(
                "eta_S_schedule must be constant, linear_warmup, or "
                "exponential_growth"
            )
        if optimizer_config.get("eta_S_warmup_steps", 1000) <= 0:
            raise ValueError("eta_S_warmup_steps must be positive")
        if optimizer_config.get("eta_S_tau", 1000.0) <= 0.0:
            raise ValueError("eta_S_tau must be positive")
        if optimizer_config.get("eta_bias_correction", False):
            raise ValueError(
                "adaptive S averaging cannot be combined with eta bias correction"
            )
    eta_g_schedule = optimizer_config.get("eta_g_schedule", "constant")
    if adaptive_g_average:
        if eta_g_schedule not in (
            "constant",
            "linear_warmup",
            "exponential_growth",
        ):
            raise ValueError(
                "eta_g_schedule must be constant, linear_warmup, or "
                "exponential_growth"
            )
        if optimizer_config.get("eta_g_warmup_steps", 1000) <= 0:
            raise ValueError("eta_g_warmup_steps must be positive")
        if optimizer_config.get("eta_g_tau", 1000.0) <= 0.0:
            raise ValueError("eta_g_tau must be positive")
        if optimizer_config.get("eta_bias_correction", False):
            raise ValueError(
                "adaptive gradient averaging cannot be combined with eta bias correction"
            )
    eta_S_config, eta_g_config = resolve_wssr_averaging_weights(
        optimizer_config
    )
    reduced_metric_history_mode = optimizer_config.get(
        "reduced_metric_history_mode", "none"
    )
    if reduced_metric_history_mode not in _WSSR_REDUCED_METRIC_HISTORY_MODES:
        raise ValueError(
            "reduced_metric_history_mode must be one of "
            f"{_WSSR_REDUCED_METRIC_HISTORY_MODES}"
        )
    use_reduced_metric_history = reduced_metric_history_mode != "none"
    if use_reduced_metric_history:
        if eta_g_config != 0.0:
            raise ValueError(
                "reduced metric history requires eta_g=0 (current RHS)"
            )
        if adaptive_S_average or adaptive_g_average or enable_gradient_transport:
            raise ValueError(
                "reduced metric history owns its spectral adaptation and cannot "
                "be combined with averaging schedules or gradient transport"
            )
        if optimizer_config.get("experimental_mode", "none") != "none":
            raise ValueError(
                "reduced metric history cannot be combined with experimental_mode"
            )
        if optimizer_config.get("spectral_regularization") != "tikhonov":
            raise ValueError(
                "reduced metric history requires Tikhonov regularization"
            )
        if optimizer_config.get("complement_weight", 1.0) != 0.0:
            raise ValueError(
                "reduced metric history requires complement_weight=0"
            )
        if optimizer_config.get("tikhonov_lambda", -1.0) < 0.0:
            raise ValueError(
                "reduced metric history requires an explicit tikhonov_lambda"
            )
        if not optimizer_config.get("store_warm_u", True):
            raise ValueError(
                "reduced metric history requires store_warm_u=True"
            )
        if optimizer_config.get("semi_matrix_free_augmented", False):
            raise ValueError(
                "reduced metric history uses a current-only score block and "
                "cannot be combined with semi_matrix_free_augmented"
            )
        if optimizer_config.get("spectral_history_cluster_gap", 0.01) < 0.0:
            raise ValueError("spectral_history_cluster_gap must be nonnegative")
        if optimizer_config.get("spectral_history_noise_scale", 1.0) <= 0.0:
            raise ValueError("spectral_history_noise_scale must be positive")
        if optimizer_config.get("spectral_history_drift_scale", 1.0) < 0.0:
            raise ValueError(
                "spectral_history_drift_scale must be nonnegative"
            )
    use_anisotropic_matrix_history = optimizer_config.get(
        "anisotropic_matrix_history", False
    )
    if use_anisotropic_matrix_history:
        if use_reduced_metric_history:
            raise ValueError(
                "anisotropic matrix history and reduced metric history are "
                "mutually exclusive"
            )
        if not 0.0 <= eta_S_config < 1.0:
            raise ValueError(
                "anisotropic matrix history requires eta_S in [0, 1)"
            )
        if eta_g_config != 0.0:
            raise ValueError(
                "anisotropic matrix history requires eta_g=0 (current RHS)"
            )
        if adaptive_S_average or adaptive_g_average or enable_gradient_transport:
            raise ValueError(
                "anisotropic matrix history cannot be combined with averaging "
                "schedules or gradient transport"
            )
        if optimizer_config.get("experimental_mode", "none") != "none":
            raise ValueError(
                "anisotropic matrix history cannot be combined with experimental_mode"
            )
        if optimizer_config.get("spectral_regularization") != "tikhonov":
            raise ValueError(
                "anisotropic matrix history requires Tikhonov regularization"
            )
        if optimizer_config.get("complement_weight", 1.0) != 0.0:
            raise ValueError(
                "anisotropic matrix history requires complement_weight=0"
            )
        if optimizer_config.get("tikhonov_lambda", -1.0) < 0.0:
            raise ValueError(
                "anisotropic matrix history requires an explicit tikhonov_lambda"
            )
        if not optimizer_config.get("store_warm_u", True):
            raise ValueError(
                "anisotropic matrix history requires store_warm_u=True"
            )
        if optimizer_config.get("semi_matrix_free_augmented", False):
            raise ValueError(
                "anisotropic matrix history cannot be combined with "
                "semi_matrix_free_augmented"
            )
        if optimizer_config.get(
            "anisotropic_matrix_history_noise_scale", 1.0
        ) <= 0.0:
            raise ValueError(
                "anisotropic_matrix_history_noise_scale must be positive"
            )
    experimental_mode = optimizer_config.get("experimental_mode", "none")
    use_cluster_envelope = experimental_mode in (
        "cluster_envelope",
        "cluster_envelope_ritz",
        "cluster_envelope_ritz_ef",
        "cluster_envelope_ritz_selected",
        "cluster_envelope_ritz_snr",
        "grassmann_ritz",
    )
    use_cluster_envelope_ef = experimental_mode == "cluster_envelope_ritz_ef"
    use_cluster_envelope_selection = (
        experimental_mode == "cluster_envelope_ritz_selected"
    )
    if use_cluster_envelope:
        cluster_rank = optimizer_config.get("cluster_envelope_rank", 32)
        envelope_length = optimizer_config.get("cluster_envelope_history", 3)
        if eta_S_config != 0.0 or eta_g_config != 0.0:
            raise ValueError("cluster_envelope requires eta_S=eta_g=0")
        if cluster_rank < 1 or cluster_rank > optimizer_config.sr_rank:
            raise ValueError(
                "cluster_envelope_rank must be in [1, sr_rank]"
            )
        if envelope_length < 1:
            raise ValueError("cluster_envelope_history must be positive")
        if experimental_mode == "grassmann_ritz" and envelope_length != 2:
            raise ValueError(
                "grassmann_ritz requires cluster_envelope_history=2"
            )
        if not 0.0 <= optimizer_config.get(
            "grassmann_smoothing_alpha", 0.5
        ) <= 1.0:
            raise ValueError("grassmann_smoothing_alpha must be in [0, 1]")
        if optimizer_config.get("cluster_snr_gap_threshold", 0.05) < 0.0:
            raise ValueError("cluster_snr_gap_threshold must be nonnegative")
        envelope_capacity = optimizer_config.get(
            "cluster_envelope_capacity", -1
        )
        if use_cluster_envelope_selection and not (
            cluster_rank <= envelope_capacity
            <= cluster_rank * envelope_length
        ):
            raise ValueError(
                "selected cluster-envelope capacity must be between the "
                "current rank and total history-pool width"
            )
        if not 0.0 < optimizer_config.get(
            "cluster_envelope_decay", 0.8
        ) <= 1.0:
            raise ValueError("cluster_envelope_decay must be in (0, 1]")
        if not 0.0 <= optimizer_config.get(
            "cluster_envelope_alpha", 0.2
        ) <= 1.0:
            raise ValueError("cluster_envelope_alpha must be in [0, 1]")
        gamma = optimizer_config.get("cluster_envelope_gamma", -1.0)
        if gamma == 0.0:
            raise ValueError(
                "cluster_envelope_gamma must be positive or negative for adaptive"
            )
        if optimizer_config.get(
            "cluster_envelope_eigenvalue_cutoff", 1e-6
        ) <= 0.0:
            raise ValueError(
                "cluster_envelope_eigenvalue_cutoff must be positive"
            )
        if optimizer_config.get(
            "cluster_envelope_curvature_mode", "scalar"
        ) not in ("scalar", "rayleigh_ritz"):
            raise ValueError(
                "cluster_envelope_curvature_mode must be scalar or "
                "rayleigh_ritz"
            )
        if optimizer_config.get("spectral_regularization") != "tikhonov":
            raise ValueError("cluster_envelope requires Tikhonov regularization")
        if optimizer_config.get("complement_weight", 1.0) != 0.0:
            raise ValueError("cluster_envelope requires complement_weight=0")
        if not optimizer_config.get("store_warm_u", True):
            raise ValueError("cluster_envelope requires store_warm_u=True")
    use_independent_gradient_memory = (
        adaptive_S_average
        or adaptive_g_average
        or enable_gradient_transport
        or eta_S_config != eta_g_config
    ) and not (
        use_reduced_metric_history or use_anisotropic_matrix_history
    )
    if use_independent_gradient_memory:
        if optimizer_config.get("experimental_mode", "none") != "none":
            raise ValueError(
                "independent gradient memory cannot be combined with "
                "experimental modes"
            )
        if (
            optimizer_config.get("exact_reference_diagnostics", False)
            or optimizer_config.get("update_diagnostics", False)
            or optimizer_config.get("reliability_diagnostics", False)
        ):
            raise ValueError(
                "independent gradient memory cannot be combined with "
                "reference diagnostics"
            )
    if optimizer_config.get("experimental_mode", "none") == "native_proximal":
        if optimizer_config.get("native_proximal_gamma", 0.0) <= 0.0:
            raise ValueError("native_proximal_gamma must be positive")
        if optimizer_config.get("complement_state_decay", 0.0) > 0.0:
            raise ValueError(
                "native_proximal cannot be combined with complement_state_decay"
            )
        if optimizer_config.get("multilevel_complement_period", 1) > 1:
            raise ValueError(
                "native_proximal cannot be combined with multilevel complement"
            )
    if optimizer_config.get("mixed_precision_solve", False):
        if optimizer_config.get("spectral_regularization") != "tikhonov":
            raise ValueError(
                "mixed_precision_solve requires spectral_regularization=tikhonov"
            )
        if optimizer_config.get("complement_weight", 1.0) != 0.0:
            raise ValueError("mixed_precision_solve requires complement_weight=0")
        if optimizer_config.get("experimental_mode", "none") != "none":
            raise ValueError(
                "mixed_precision_solve cannot be combined with experimental modes"
            )
        if (optimizer_config.get("exact_reference_diagnostics", False)
                or optimizer_config.get("update_diagnostics", False)
                or optimizer_config.get("reliability_diagnostics", False)):
            raise ValueError(
                "mixed_precision_solve cannot be combined with reference diagnostics"
            )
    solution_recurrence_mode = optimizer_config.get(
        "solution_recurrence_mode", "none"
    )
    if use_reduced_metric_history or use_anisotropic_matrix_history:
        if solution_recurrence_mode != "none":
            raise ValueError(
                "spectral history modes cannot be combined with solution recurrence"
            )
        if (
            optimizer_config.get("exact_reference_diagnostics", False)
            or optimizer_config.get("update_diagnostics", False)
            or optimizer_config.get("reliability_diagnostics", False)
        ):
            raise ValueError(
                "spectral history modes cannot be combined with reference diagnostics"
            )
    residual_evaluation = optimizer_config.get(
        "residual_evaluation", "rank_coordinate"
    )
    if residual_evaluation not in ("rank_coordinate", "full_current_batch"):
        raise ValueError(
            "residual_evaluation must be rank_coordinate or full_current_batch"
        )
    galerkin_solve_backend = optimizer_config.get(
        "galerkin_solve_backend", "host_fp64"
    )
    if galerkin_solve_backend not in ("host_fp64", "device_cholesky"):
        raise ValueError(
            "galerkin_solve_backend must be host_fp64 or device_cholesky"
        )
    residual_dual_mode_diagnostics = optimizer_config.get(
        "residual_dual_mode_diagnostics", False
    )
    solution_error_feedback = optimizer_config.get(
        "solution_error_feedback", False
    )
    error_feedback_norm_cap = optimizer_config.get(
        "error_feedback_norm_cap", 10.0
    )
    if error_feedback_norm_cap <= 0.0:
        raise ValueError("error_feedback_norm_cap must be positive")
    error_feedback_decay = optimizer_config.get(
        "error_feedback_decay", 1.0
    )
    if not 0.0 <= error_feedback_decay <= 1.0:
        raise ValueError("error_feedback_decay must be in [0, 1]")
    error_feedback_cap_reference = optimizer_config.get(
        "error_feedback_cap_reference", "correction"
    )
    if error_feedback_cap_reference not in (
        "correction",
        "sample_residual",
        "parameter_conflict",
    ):
        raise ValueError(
            "error_feedback_cap_reference must be correction, "
            "sample_residual, or parameter_conflict"
        )
    subspace_eta_S = optimizer_config.get("subspace_eta_S", 0.0)
    if not 0.0 <= subspace_eta_S < 1.0:
        raise ValueError("subspace_eta_S must be in [0, 1)")
    subspace_refresh_period = optimizer_config.get(
        "subspace_refresh_period", 1
    )
    if subspace_refresh_period < 1:
        raise ValueError("subspace_refresh_period must be positive")
    subspace_refresh_mode = optimizer_config.get(
        "subspace_refresh_mode", "ritz"
    )
    if subspace_refresh_mode not in ("ritz", "fixed_basis"):
        raise ValueError(
            "subspace_refresh_mode must be ritz or fixed_basis"
        )
    drift_gate_hypothetical_eta_g = optimizer_config.get(
        "drift_gate_hypothetical_eta_g", 0.8
    )
    if not 0.0 <= drift_gate_hypothetical_eta_g < 1.0:
        raise ValueError("drift_gate_hypothetical_eta_g must be in [0, 1)")
    if solution_recurrence_mode not in ("none", "naive", "residual"):
        raise ValueError(
            "solution_recurrence_mode must be none, naive, or residual"
        )
    if solution_recurrence_mode != "none":
        if not optimizer_config.get("mixed_precision_solve", False):
            raise ValueError("solution recurrence requires mixed_precision_solve")
        if eta_S_config != 0.0 or eta_g_config != 0.0:
            raise ValueError(
                "solution recurrence requires eta_S=eta_g=0 to isolate "
                "solution history"
            )
        if optimizer_config.get("eta_bias_correction", False):
            raise ValueError(
                "solution recurrence cannot be combined with eta bias correction"
            )
        solution_mu = optimizer_config.get("solution_recurrence_mu", 0.99)
        if not 0.0 <= solution_mu < 1.0:
            raise ValueError("solution_recurrence_mu must be in [0, 1)")
        if subspace_eta_S != 0.0 and not (
            solution_recurrence_mode == "residual"
            and residual_evaluation == "full_current_batch"
        ):
            raise ValueError(
                "subspace_eta_S is supported only by full-current-batch "
                "residual recurrence"
            )
    elif (
        residual_evaluation != "rank_coordinate"
        or residual_dual_mode_diagnostics
        or solution_error_feedback
        or subspace_eta_S != 0.0
    ):
        raise ValueError(
            "residual extensions require solution_recurrence_mode=residual"
        )
    if solution_error_feedback and residual_evaluation != "full_current_batch":
        raise ValueError(
            "solution_error_feedback requires full_current_batch residuals"
        )
    if use_independent_gradient_memory and solution_recurrence_mode != "none":
        raise ValueError(
            "independent gradient memory cannot be combined with solution "
            "recurrence"
        )
    semi_matrix_free_augmented = optimizer_config.get(
        "semi_matrix_free_augmented", False
    )
    if semi_matrix_free_augmented:
        if optimizer_config.get("store_warm_u", True):
            raise ValueError(
                "semi_matrix_free_augmented requires store_warm_u=False"
            )
        if optimizer_config.get("exact_first", False) or optimizer_config.get(
            "exact_first_force", False
        ):
            raise ValueError(
                "semi_matrix_free_augmented does not support exact_first"
            )
        if optimizer_config.get("experimental_mode", "none") not in (
            "none",
            "adaptive_complement",
        ):
            raise ValueError(
                "semi_matrix_free_augmented supports only the "
                "adaptive_complement experimental mode"
            )
        if use_independent_gradient_memory:
            raise ValueError(
                "semi_matrix_free_augmented cannot be combined with "
                "independent gradient memory"
            )
        if (
            optimizer_config.get("exact_reference_diagnostics", False)
            or optimizer_config.get("update_diagnostics", False)
            or optimizer_config.get("reliability_diagnostics", False)
        ):
            raise ValueError(
                "semi_matrix_free_augmented cannot be combined with "
                "reference diagnostics"
            )
    if subspace_refresh_period > 1:
        if solution_recurrence_mode != "residual" or (
            residual_evaluation != "full_current_batch"
        ):
            raise ValueError(
                "delayed subspace refresh requires full-current-batch "
                "residual recurrence"
            )
        if semi_matrix_free_augmented:
            raise ValueError(
                "delayed subspace refresh currently requires a stored warm U"
            )
        if not optimizer_config.get("store_warm_u", True):
            raise ValueError(
                "delayed subspace refresh requires store_warm_u=True"
            )
    elif subspace_refresh_mode != "ritz":
        raise ValueError(
            "fixed_basis subspace refresh requires subspace_refresh_period > 1"
        )

    def update_param_fn(params, data, optimizer_state, key):
        position = get_position_fn(data)
        energy, local_energies, stats = energy_and_statistics_fn(params, position)
        key, svd_key = jax.random.split(key)

        reuse_warm_subspace = jnp.asarray(False)
        subspace_refreshed = jnp.asarray(True)
        if subspace_refresh_period > 1:
            step_counters = [
                leaf
                for leaf in jax.tree_util.tree_leaves(
                    optimizer_state.optax_state
                )
                if getattr(leaf, "shape", None) == ()
                and jnp.issubdtype(leaf.dtype, jnp.integer)
            ]
            if len(step_counters) != 1:
                raise ValueError(
                    "delayed subspace refresh requires exactly one scalar "
                    "Optax step counter"
                )
            refresh_due = (step_counters[0] % subspace_refresh_period) == 0
            has_reusable_subspace = optimizer_state.core_state.has_u
            reuse_warm_subspace = (~refresh_due) & has_reusable_subspace
            subspace_refreshed = ~reuse_warm_subspace

        if use_independent_gradient_memory and not isinstance(
            optimizer_state, WSSRTransportedGradientOptimizerState
        ):
            raise ValueError(
                "independent gradient memory requires its dedicated optimizer "
                "state"
            )
        expected_cluster_state = (
            WSSRClusterEnvelopeEFOptimizerState
            if use_cluster_envelope_ef
            else WSSRClusterEnvelopeOptimizerState
        )
        if use_cluster_envelope and not isinstance(
            optimizer_state, expected_cluster_state
        ):
            raise ValueError(
                "cluster_envelope requires its dedicated optimizer state"
            )

        if solution_recurrence_mode != "none":
            if not isinstance(
                optimizer_state,
                (
                    WSSRSolutionRecurrenceOptimizerState,
                    WSSRErrorFeedbackRecurrenceOptimizerState,
                ),
            ):
                raise ValueError(
                    "solution recurrence requires its dedicated optimizer state"
                )
            solution_prior = (
                optimizer_config.get("solution_recurrence_mu", 0.99)
                * optimizer_state.solution_state
            )
            error_feedback_state = (
                optimizer_state.error_feedback_state
                if isinstance(
                    optimizer_state,
                    WSSRErrorFeedbackRecurrenceOptimizerState,
                )
                else None
            )
        else:
            solution_prior = None
            error_feedback_state = None

        eta_S_used = eta_S_config
        eta_g_used = eta_g_config
        averaging_step = jnp.asarray(-1, dtype=jnp.int32)
        if adaptive_S_average or adaptive_g_average:
            eta_counters = [
                leaf for leaf in jax.tree_util.tree_leaves(
                    optimizer_state.optax_state
                )
                if getattr(leaf, "shape", None) == ()
                and jnp.issubdtype(leaf.dtype, jnp.integer)
            ]
            if len(eta_counters) != 1:
                raise ValueError(
                    "adaptive averaging requires exactly one scalar Optax "
                    "step counter"
                )
            averaging_step = eta_counters[0]
            if adaptive_S_average:
                eta_S_used = adaptive_s_averaging_eta(
                    eta_S_schedule,
                    eta_S_config,
                    averaging_step,
                    optimizer_config.get("eta_S_warmup_steps", 1000),
                    optimizer_config.get("eta_S_tau", 1000.0),
                )
            if adaptive_g_average:
                eta_g_used = adaptive_s_averaging_eta(
                    eta_g_schedule,
                    eta_g_config,
                    averaging_step,
                    optimizer_config.get("eta_g_warmup_steps", 1000),
                    optimizer_config.get("eta_g_tau", 1000.0),
                )
        elif optimizer_config.get("eta_bias_correction", False):
            eta_counters = [
                leaf for leaf in jax.tree_util.tree_leaves(
                    optimizer_state.optax_state
                )
                if getattr(leaf, "shape", None) == ()
                and jnp.issubdtype(leaf.dtype, jnp.integer)
            ]
            if len(eta_counters) != 1:
                raise ValueError(
                    "eta bias correction requires exactly one scalar Optax step counter"
                )
            eta_S_used = bias_corrected_history_eta(
                eta_S_config, eta_counters[0]
            )
            eta_g_used = bias_corrected_history_eta(
                eta_g_config, eta_counters[0]
            )

        adaptive_beta = optimizer_config.get("adaptive_complement_beta", 0.0)
        decay_steps = optimizer_config.get("adaptive_complement_decay_steps", 0)
        if decay_steps > 0:
            integer_scalars = [
                leaf for leaf in jax.tree_util.tree_leaves(optimizer_state.optax_state)
                if getattr(leaf, "shape", None) == ()
                and jnp.issubdtype(leaf.dtype, jnp.integer)
            ]
            if len(integer_scalars) != 1:
                raise ValueError(
                    "beta decay requires exactly one scalar Optax step counter"
                )
            local_step = jnp.clip(
                integer_scalars[0]
                - optimizer_config.get("adaptive_complement_decay_start", 0),
                0, decay_steps - 1,
            )
            adaptive_beta = wssr_experimental.linear_budget(
                adaptive_beta,
                optimizer_config.get("adaptive_complement_beta_final", adaptive_beta),
                local_step,
                decay_steps,
            )

        core_eta = (
            subspace_eta_S
            if solution_recurrence_mode == "residual"
            and residual_evaluation == "full_current_batch"
            else eta_S_used
        )
        core_kwargs = dict(
            log_psi_apply=log_psi_apply,
            params=params,
            positions=position,
            local_energies=local_energies,
            energy=energy,
            state=optimizer_state.core_state,
            key=svd_key,
            eta=core_eta,
            damping=optimizer_config.damping,
            norm_constraint=optimizer_config.norm_constraint,
            sr_rank_max=optimizer_config.sr_rank_max,
            sr_scale=optimizer_config.sr_scale,
            svd_maxiter_initial=optimizer_config.svd_maxiter_initial,
            svd_maxiter_warm=optimizer_config.svd_maxiter_warm,
            exact_first=optimizer_config.get("exact_first", False),
            exact_first_force=optimizer_config.get("exact_first_force", False),
            svd_working_rank=optimizer_config.get("svd_working_rank", None),
            spectral_regularization=optimizer_config.get(
                "spectral_regularization", "hard_floor"
            ),
            complement_weight=optimizer_config.get("complement_weight", 1.0),
            relative_singular_value_cutoff=optimizer_config.get(
                "relative_singular_value_cutoff", -1.0
            ),
            tikhonov_lambda=optimizer_config.get("tikhonov_lambda", -1.0),
            experimental_mode=optimizer_config.get("experimental_mode", "none"),
            experimental_target_rank=optimizer_config.get("experimental_target_rank", -1),
            cluster_gap_threshold=optimizer_config.get("cluster_gap_threshold", 0.002),
            near_tail_modes=optimizer_config.get("near_tail_modes", 0),
            adaptive_complement_beta=adaptive_beta,
            adaptive_complement_beta_function=(
                adaptive_complement_beta_function
            ),
            smooth_transition_start=optimizer_config.get("smooth_transition_start", -1),
            smooth_transition_end=optimizer_config.get("smooth_transition_end", -1),
            force_aware_krylov_vectors=optimizer_config.get("force_aware_krylov_vectors", 0),
            iterative_complement_iterations=optimizer_config.get("iterative_complement_iterations", 0),
        )
        use_reference_path = (
            optimizer_config.get("experimental_mode", "none") == "none"
            and (optimizer_config.get("exact_reference_diagnostics", False)
                 or optimizer_config.get("update_diagnostics", False))
        )
        inline_semimf_adaptive_complement = (
            semi_matrix_free_augmented
            and optimizer_config.get("experimental_mode", "none")
            == "adaptive_complement"
            and not optimizer_config.get("reliability_diagnostics", False)
        )
        use_experimental_path = (
            optimizer_config.get("experimental_mode", "none") != "none"
            or optimizer_config.get("reliability_diagnostics", False)
        ) and not inline_semimf_adaptive_complement
        gradient_metrics_available = False
        recurrence_diagnostics = None
        reduced_metric_diagnostics = None
        anisotropic_matrix_diagnostics = None
        if use_anisotropic_matrix_history:
            (
                grad_like_update,
                core_state,
                active_rank,
                gradient_norm,
                anisotropic_matrix_diagnostics,
            ) = compute_wssr_anisotropic_matrix_history_update(
                log_psi_apply,
                params,
                position,
                local_energies,
                energy,
                optimizer_state.core_state,
                svd_key,
                eta_S_used,
                optimizer_config.damping,
                optimizer_config.norm_constraint,
                optimizer_config.sr_rank_max,
                sr_scale=optimizer_config.sr_scale,
                svd_maxiter_initial=optimizer_config.svd_maxiter_initial,
                svd_maxiter_warm=optimizer_config.svd_maxiter_warm,
                exact_first=optimizer_config.get("exact_first", False),
                exact_first_force=optimizer_config.get(
                    "exact_first_force", False
                ),
                svd_working_rank=optimizer_config.get(
                    "svd_working_rank", None
                ),
                constrain_update_norm=False,
                relative_singular_value_cutoff=optimizer_config.get(
                    "relative_singular_value_cutoff", -1.0
                ),
                tikhonov_lambda=optimizer_config.get(
                    "tikhonov_lambda", -1.0
                ),
                noise_scale=optimizer_config.get(
                    "anisotropic_matrix_history_noise_scale", 1.0
                ),
            )
            gradient_history_norm = gradient_norm
            gradient_metrics_available = True
            reference_metrics = {}
            complement_state = None
        elif use_reduced_metric_history:
            (
                grad_like_update,
                core_state,
                active_rank,
                gradient_norm,
                reduced_metric_diagnostics,
            ) = compute_wssr_current_subspace_reduced_metric_update(
                log_psi_apply,
                params,
                position,
                local_energies,
                energy,
                optimizer_state.core_state,
                svd_key,
                eta_S_used,
                optimizer_config.damping,
                optimizer_config.norm_constraint,
                optimizer_config.sr_rank_max,
                reduced_metric_history_mode,
                sr_scale=optimizer_config.sr_scale,
                svd_maxiter_initial=optimizer_config.svd_maxiter_initial,
                svd_maxiter_warm=optimizer_config.svd_maxiter_warm,
                exact_first=optimizer_config.get("exact_first", False),
                exact_first_force=optimizer_config.get(
                    "exact_first_force", False
                ),
                svd_working_rank=optimizer_config.get(
                    "svd_working_rank", None
                ),
                constrain_update_norm=False,
                relative_singular_value_cutoff=optimizer_config.get(
                    "relative_singular_value_cutoff", -1.0
                ),
                tikhonov_lambda=optimizer_config.get(
                    "tikhonov_lambda", -1.0
                ),
                cluster_gap_threshold=optimizer_config.get(
                    "spectral_history_cluster_gap", 0.01
                ),
                noise_scale=optimizer_config.get(
                    "spectral_history_noise_scale", 1.0
                ),
                drift_scale=optimizer_config.get(
                    "spectral_history_drift_scale", 1.0
                ),
            )
            gradient_history_norm = gradient_norm
            gradient_metrics_available = True
            reference_metrics = {}
            complement_state = None
        elif use_independent_gradient_memory:
            transport_core_kwargs = {
                name: value
                for name, value in core_kwargs.items()
                if name
                not in {
                    "experimental_mode",
                    "experimental_target_rank",
                    "cluster_gap_threshold",
                    "near_tail_modes",
                    "adaptive_complement_beta",
                    "adaptive_complement_beta_function",
                    "smooth_transition_start",
                    "smooth_transition_end",
                    "force_aware_krylov_vectors",
                    "iterative_complement_iterations",
                    "eta",
                }
            }
            (
                grad_like_update,
                core_state,
                active_rank,
                transported_gradient,
                previous_theta,
                transport_norm_ema,
                transport_norm_current,
                gradient_norm,
                gradient_history_norm,
                transport_ratio_ema,
                transport_ratio_current,
                s_difference_action_norm,
                delta_theta_norm,
            ) = compute_wssr_warm_svd_right_transported_gradient_update(
                previous_gradient=optimizer_state.transported_gradient,
                previous_theta=optimizer_state.previous_theta,
                transport_initialized=optimizer_state.transport_initialized,
                eta_S=eta_S_used,
                eta_g=eta_g_used,
                enable_gradient_transport=enable_gradient_transport,
                record_S_lag_diagnostics=adaptive_S_average,
                mixed_precision_solve=optimizer_config.get(
                    "mixed_precision_solve", False
                ),
                **transport_core_kwargs,
            )
            gradient_metrics_available = True
            reference_metrics = {}
            complement_state = None
        elif use_cluster_envelope:
            (
                grad_like_update,
                core_state,
                active_rank,
                next_envelope_history,
                next_envelope_count,
                next_envelope_error_feedback,
                envelope_feedback_clip_increment,
                reference_metrics,
            ) = compute_wssr_cluster_envelope_update(
                envelope_history=optimizer_state.envelope_history,
                envelope_count=optimizer_state.envelope_count,
                cluster_rank=optimizer_config.get(
                    "cluster_envelope_rank", 32
                ),
                envelope_length=optimizer_config.get(
                    "cluster_envelope_history", 3
                ),
                envelope_decay=optimizer_config.get(
                    "cluster_envelope_decay", 0.8
                ),
                envelope_capacity=(
                    optimizer_config.get("cluster_envelope_capacity", -1)
                    if use_cluster_envelope_selection
                    else -1
                ),
                complement_alpha=(
                    0.0
                    if experimental_mode in (
                        "cluster_envelope_ritz",
                        "cluster_envelope_ritz_ef",
                        "cluster_envelope_ritz_selected",
                        "cluster_envelope_ritz_snr",
                        "grassmann_ritz",
                    )
                    else optimizer_config.get("cluster_envelope_alpha", 0.2)
                ),
                complement_gamma=optimizer_config.get(
                    "cluster_envelope_gamma", -1.0
                ),
                envelope_eigenvalue_cutoff=optimizer_config.get(
                    "cluster_envelope_eigenvalue_cutoff", 1e-6
                ),
                curvature_mode=(
                    "rayleigh_ritz"
                    if experimental_mode in (
                        "cluster_envelope_ritz",
                        "cluster_envelope_ritz_ef",
                        "cluster_envelope_ritz_selected",
                        "cluster_envelope_ritz_snr",
                        "grassmann_ritz",
                    )
                    else optimizer_config.get(
                        "cluster_envelope_curvature_mode", "scalar"
                    )
                ),
                enable_cross_batch_snr_shrinkage=(
                    experimental_mode == "cluster_envelope_ritz_snr"
                ),
                enable_grassmann_smoothing=(
                    experimental_mode == "grassmann_ritz"
                ),
                grassmann_smoothing_alpha=optimizer_config.get(
                    "grassmann_smoothing_alpha", 0.5
                ),
                snr_cluster_gap_threshold=optimizer_config.get(
                    "cluster_snr_gap_threshold", 0.05
                ),
                error_feedback_state=(
                    optimizer_state.error_feedback_state
                    if use_cluster_envelope_ef
                    else None
                ),
                enable_rotation_error_feedback=use_cluster_envelope_ef,
                rotation_error_feedback_decay=0.95,
                rotation_error_feedback_alpha=0.2,
                rotation_error_feedback_norm_cap=10.0,
                **core_kwargs,
            )
            if use_cluster_envelope_ef:
                reference_metrics["wssr_envelope_ef_clip_count"] = (
                    optimizer_state.error_feedback_clip_count
                    + envelope_feedback_clip_increment
                )
            gradient_metrics_available = False
            complement_state = None
        elif use_experimental_path:
            grad_like_update, core_state, active_rank, reference_metrics, complement_state = (
                compute_wssr_warm_svd_right_experimental_diagnostic_update(
                    burst_diagnostics_payload=optimizer_config.get(
                        "burst_diagnostics_payload", False
                    ),
                    reliability_diagnostics=optimizer_config.get(
                        "reliability_diagnostics", False
                    ),
                    complement_ema=(
                        optimizer_state.complement_ema
                        if isinstance(
                            optimizer_state,
                            (WSSRComplementEMAOptimizerState,
                             WSSRMultilevelComplementOptimizerState),
                        )
                        else None
                    ),
                    multilevel_ema_b=(
                        optimizer_state.complement_ema_b
                        if isinstance(optimizer_state, WSSRMultilevelComplementOptimizerState)
                        else None
                    ),
                    multilevel_weight_a=(
                        optimizer_state.complement_weight_a
                        if isinstance(optimizer_state, WSSRMultilevelComplementOptimizerState)
                        else None
                    ),
                    multilevel_weight_b=(
                        optimizer_state.complement_weight_b
                        if isinstance(optimizer_state, WSSRMultilevelComplementOptimizerState)
                        else None
                    ),
                    multilevel_step=(
                        optimizer_state.complement_step
                        if isinstance(optimizer_state, WSSRMultilevelComplementOptimizerState)
                        else None
                    ),
                    multilevel_apply_period=optimizer_config.get(
                        "multilevel_complement_period", 1
                    ),
                    multilevel_cosine_threshold=optimizer_config.get(
                        "multilevel_complement_cosine_threshold", 0.0
                    ),
                    complement_state_decay=optimizer_config.get(
                        "complement_state_decay", 0.0
                    ),
                    complement_state_relative_cap=optimizer_config.get(
                        "complement_state_relative_cap", 0.0
                    ),
                    native_proximal_gamma=optimizer_config.get(
                        "native_proximal_gamma", 0.0
                    ),
                    **core_kwargs,
                )
            )
        elif use_reference_path:
            # Experimental-only controls are not part of the normal reference
            # diagnostic API.  This branch is selected only for mode="none";
            # avoid forwarding unrelated disabled options into it.
            reference_core_kwargs = {
                key: value
                for key, value in core_kwargs.items()
                if key
                not in {
                    "experimental_mode",
                    "experimental_target_rank",
                    "cluster_gap_threshold",
                    "near_tail_modes",
                    "adaptive_complement_beta",
                    "adaptive_complement_beta_function",
                    "adaptive_complement_beta_final",
                    "adaptive_complement_decay_steps",
                    "adaptive_complement_decay_start",
                    "smooth_transition_start",
                    "smooth_transition_end",
                    "force_aware_krylov_vectors",
                    "iterative_complement_iterations",
                    "native_proximal_gamma",
                }
            }
            grad_like_update, core_state, active_rank, reference_metrics = (
                compute_wssr_warm_svd_right_reference_diagnostic_update(
                    compute_exact_reference=optimizer_config.get(
                        "exact_reference_diagnostics", False
                    ),
                    **reference_core_kwargs
                )
            )
        else:
            need_recurrence_diagnostics = (
                solution_recurrence_mode == "residual"
                and (
                    residual_evaluation == "full_current_batch"
                    or residual_dual_mode_diagnostics
                    or solution_error_feedback
                    or optimizer_config.get("recurrence_telemetry", False)
                )
            )
            core_output = compute_wssr_warm_svd_right_core_update(
                constrain_update_norm=False,
                mixed_precision_solve=optimizer_config.get(
                    "mixed_precision_solve", False
                ),
                solution_prior=solution_prior,
                solution_recurrence_mode=solution_recurrence_mode,
                residual_evaluation=residual_evaluation,
                residual_dual_mode_diagnostics=(
                    residual_dual_mode_diagnostics
                ),
                solution_error_feedback=solution_error_feedback,
                error_feedback_state=error_feedback_state,
                error_feedback_norm_cap=error_feedback_norm_cap,
                error_feedback_decay=error_feedback_decay,
                error_feedback_cap_reference=error_feedback_cap_reference,
                galerkin_solve_backend=galerkin_solve_backend,
                subspace_only_averaging=(subspace_eta_S != 0.0),
                semi_matrix_free_augmented=semi_matrix_free_augmented,
                reuse_warm_subspace=reuse_warm_subspace,
                fixed_warm_subspace=(
                    subspace_refresh_mode == "fixed_basis"
                ),
                return_gradient_diagnostics=True,
                return_recurrence_diagnostics=need_recurrence_diagnostics,
                **core_kwargs,
            )
            if need_recurrence_diagnostics:
                (
                    grad_like_update,
                    core_state,
                    active_rank,
                    gradient_norm,
                    gradient_history_norm,
                    recurrence_diagnostics,
                ) = core_output
            else:
                (
                    grad_like_update,
                    core_state,
                    active_rank,
                    gradient_norm,
                    gradient_history_norm,
                ) = core_output
                recurrence_diagnostics = None
            gradient_metrics_available = True
            reference_metrics = {}
            complement_state = None

        monitor_long_history = (
            optimizer_config.get("mixed_precision_solve", False)
            or optimizer_config.get("eta_bias_correction", False)
            or solution_recurrence_mode != "none"
        )
        if monitor_long_history:
            raw_direction_norm = tree_l2_norm(grad_like_update)
            direction_finite = jnp.all(jnp.stack([
                jnp.all(jnp.isfinite(leaf))
                for leaf in jax.tree_util.tree_leaves(grad_like_update)
            ]))
        if solution_recurrence_mode != "none":
            flat_grad_like_update, _ = jax.flatten_util.ravel_pytree(
                grad_like_update
            )
            solution_prior_norm = jnp.linalg.norm(solution_prior)
            solution_correction_norm = jnp.linalg.norm(
                flat_grad_like_update - solution_prior
            )

        updates, optax_state = optimizer.update(
            grad_like_update, optimizer_state.optax_state, params
        )
        unconstrained_update_norm = optax.global_norm(updates)
        constraint_scale = jnp.minimum(
            1.0,
            jnp.sqrt(optimizer_config.norm_constraint)
            / jnp.maximum(unconstrained_update_norm, 1e-12),
        )
        if optimizer_config.constrain_norm:
            if norm_constraint_mode == "function_space":
                (
                    updates,
                    function_update_norm,
                    constraint_scale,
                    constrained_function_update_norm,
                ) = constrain_update_function_norm(
                    log_psi_apply,
                    params,
                    position,
                    updates,
                    function_norm_constraint,
                )
            else:
                updates = constrain_update_tree_norm(
                    updates, optimizer_config.norm_constraint
                )
        euclidean_safety_scale = jnp.asarray(1.0, dtype=energy.dtype)
        if euclidean_safety_constraint > 0.0:
            before_safety_norm = optax.global_norm(updates)
            updates = constrain_update_tree_norm(
                updates, euclidean_safety_constraint
            )
            after_safety_norm = optax.global_norm(updates)
            euclidean_safety_scale = jnp.where(
                before_safety_norm > 0.0,
                after_safety_norm / before_safety_norm,
                1.0,
            )
        actual_update_norm = optax.global_norm(updates)
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
        if monitor_long_history:
            metrics.update(
                {
                    # Legacy alias follows the S-matrix history weight.
                    "wssr_eta_used": eta_S_used,
                    "wssr_diag_raw_direction_norm": raw_direction_norm,
                    "wssr_diag_finite": direction_finite,
                }
            )
        metrics.update(
            {
                "eta_S": eta_S_used,
                "eta_g": eta_g_used,
            }
        )
        if (
            norm_constraint_mode == "function_space"
            and optimizer_config.constrain_norm
        ):
            metrics.update(
                {
                    "wssr_function_update_norm_unconstrained": (
                        function_update_norm
                    ),
                    "wssr_function_update_norm_constrained": (
                        constrained_function_update_norm
                    ),
                    "wssr_function_norm_constraint_scale": constraint_scale,
                    "wssr_function_norm_constraint_active": (
                        constraint_scale < 1.0
                    ).astype(function_update_norm.dtype),
                }
            )
        if euclidean_safety_constraint > 0.0:
            metrics.update(
                {
                    "wssr_euclidean_safety_scale": euclidean_safety_scale,
                    "wssr_euclidean_safety_active": (
                        euclidean_safety_scale < 1.0
                    ).astype(energy.dtype),
                }
            )
        if adaptive_S_average:
            metrics.update(
                {
                    "adaptive_S_step": averaging_step,
                    "eta_S_current": eta_S_used,
                    # The full operators are too large to materialize. This is
                    # their difference applied to the previous actual update:
                    # ||(S_current - S_ema_used) delta_theta_prev||.
                    "S_current_minus_ema_action_norm": (
                        s_difference_action_norm
                    ),
                    "S_transport_norm": transport_norm_ema,
                    "delta_theta_norm": delta_theta_norm,
                }
            )
        if adaptive_g_average:
            metrics["eta_g_current"] = eta_g_used
        if gradient_metrics_available:
            metrics.update(
                {
                    "gradient_norm": gradient_norm,
                    "gradient_history_norm": gradient_history_norm,
                }
            )
        if reduced_metric_diagnostics is not None:
            metrics.update(
                {
                    "wssr_spectral_history_eta_mean": (
                        reduced_metric_diagnostics.eta_mean
                    ),
                    "wssr_spectral_history_eta_head": (
                        reduced_metric_diagnostics.eta_head
                    ),
                    "wssr_spectral_history_eta_tail": (
                        reduced_metric_diagnostics.eta_tail
                    ),
                    "wssr_spectral_history_noise_floor": (
                        reduced_metric_diagnostics.noise_floor
                    ),
                    "wssr_spectral_history_overlap": (
                        reduced_metric_diagnostics.history_overlap
                    ),
                    "wssr_spectral_history_cluster_count": (
                        reduced_metric_diagnostics.cluster_count
                    ),
                    "wssr_spectral_history_drift_ratio_mean": (
                        reduced_metric_diagnostics.drift_ratio_mean
                    ),
                }
            )
        if anisotropic_matrix_diagnostics is not None:
            metrics.update(
                {
                    "wssr_anisotropic_matrix_eta_mean": (
                        anisotropic_matrix_diagnostics.eta_mean
                    ),
                    "wssr_anisotropic_matrix_eta_head": (
                        anisotropic_matrix_diagnostics.eta_head
                    ),
                    "wssr_anisotropic_matrix_eta_tail": (
                        anisotropic_matrix_diagnostics.eta_tail
                    ),
                    "wssr_anisotropic_matrix_eta_spectral_mean": (
                        anisotropic_matrix_diagnostics.eta_spectral_mean
                    ),
                    "wssr_anisotropic_matrix_current_weight": (
                        anisotropic_matrix_diagnostics.current_weight
                    ),
                    "wssr_anisotropic_matrix_noise_floor": (
                        anisotropic_matrix_diagnostics.noise_floor
                    ),
                }
            )
        if solution_recurrence_mode != "none":
            metrics.update(
                {
                    "wssr_solution_prior_norm": solution_prior_norm,
                    "wssr_solution_correction_norm": solution_correction_norm,
                }
            )
        if recurrence_diagnostics is not None:
            metrics.update(
                {
                    "wssr_residual_sample_norm": (
                        recurrence_diagnostics.sample_residual_norm
                    ),
                    "wssr_residual_correctable_norm": (
                        recurrence_diagnostics.correctable_residual_norm
                    ),
                    "wssr_residual_correctable_ratio": (
                        recurrence_diagnostics.correctable_residual_ratio
                    ),
                    "wssr_residual_captured_sample_ratio": (
                        recurrence_diagnostics.captured_sample_residual_ratio
                    ),
                    "wssr_residual_norm_reduction": (
                        recurrence_diagnostics.residual_norm_reduction
                    ),
                    "wssr_residual_reduction_fraction": (
                        recurrence_diagnostics.residual_reduction_fraction
                    ),
                    "wssr_error_feedback_norm": (
                        recurrence_diagnostics.error_feedback_norm
                    ),
                    "wssr_error_feedback_uncapped_norm": (
                        recurrence_diagnostics.error_feedback_uncapped_norm
                    ),
                    "wssr_error_feedback_cap_norm": (
                        recurrence_diagnostics.error_feedback_cap_norm
                    ),
                    "wssr_error_feedback_to_sample_ratio": (
                        recurrence_diagnostics.error_feedback_to_sample_ratio
                    ),
                    "wssr_error_feedback_to_parameter_conflict_ratio": (
                        recurrence_diagnostics
                        .error_feedback_to_parameter_conflict_ratio
                    ),
                    "wssr_residual_correction_relative_difference": (
                        recurrence_diagnostics.correction_relative_difference
                    ),
                    "wssr_subspace_eta_S": jnp.asarray(
                        subspace_eta_S, dtype=gradient_norm.dtype
                    ),
                }
            )
            if isinstance(
                optimizer_state, WSSRErrorFeedbackRecurrenceOptimizerState
            ):
                metrics["wssr_error_feedback_clip_count"] = (
                    optimizer_state.error_feedback_clip_count
                    + recurrence_diagnostics.error_feedback_clip_increment
                )
                metrics["wssr_error_feedback_clip_increment"] = (
                    recurrence_diagnostics.error_feedback_clip_increment
                )
        if subspace_refresh_period > 1:
            metrics["wssr_subspace_refreshed"] = subspace_refreshed.astype(
                jnp.int32
            )
        if optimizer_config.get("drift_gate_monitoring", False):
            flat_direction, _ = jax.flatten_util.ravel_pytree(grad_like_update)
            hypothetical_eta = jnp.asarray(
                drift_gate_hypothetical_eta_g, dtype=flat_direction.dtype
            )
            drift_prefactor = hypothetical_eta / jnp.maximum(
                1.0 - hypothetical_eta,
                jnp.asarray(1e-12, dtype=flat_direction.dtype),
            )
            metrics["wssr_hypothetical_gradient_drift_beta"] = (
                drift_prefactor
                * actual_update_norm
                / jnp.maximum(jnp.linalg.norm(flat_direction), 1e-12)
            )
        if enable_gradient_transport:
            metrics.update(
                {
                    # Backward-compatible aliases keep existing result readers
                    # working; the explicit names distinguish the two operators.
                    "transport_norm": transport_norm_ema,
                    "gradient_norm": gradient_norm,
                    "transport_ratio": transport_ratio_ema,
                    "S_transport_norm_ema": transport_norm_ema,
                    "S_transport_norm_current": transport_norm_current,
                    "transport_ratio_ema": transport_ratio_ema,
                    "transport_ratio_current": transport_ratio_current,
                    "S_transport_norm": transport_norm_ema,
                }
            )
        if reference_metrics:
            metrics.update(reference_metrics)
        if optimizer_config.get("reliability_diagnostics", False):
            raw_local_energies = stats["local_energies_noclip"]
            raw_median = jnp.nanmedian(raw_local_energies)
            raw_absolute_deviation = jnp.abs(
                raw_local_energies - raw_median
            )
            raw_mad = jnp.nanmedian(raw_absolute_deviation)
            robust_scale = jnp.maximum(1.4826 * raw_mad, 1e-12)
            metrics.update(
                {
                    "wssr_reliability_raw_energy_median": raw_median,
                    "wssr_reliability_raw_energy_mad": raw_mad,
                    "wssr_reliability_raw_tail_q99_robust_z": (
                        jnp.nanquantile(raw_absolute_deviation, 0.99)
                        / robust_scale
                    ),
                    "wssr_reliability_raw_tail_max_robust_z": (
                        jnp.nanmax(raw_absolute_deviation) / robust_scale
                    ),
                    "wssr_reliability_raw_tail_fraction_gt_10_robust_z": (
                        jnp.nanmean(
                            (raw_absolute_deviation > 10.0 * robust_scale).astype(
                                raw_local_energies.dtype
                            )
                        )
                    ),
                }
            )
        if optimizer_config.get("burst_diagnostics_payload", False):
            metrics["burst_diag_local_energies"] = stats["local_energies_noclip"]
            if "hamiltonian_components" in stats:
                metrics["burst_diag_hamiltonian_components"] = stats[
                    "hamiltonian_components"
                ]
            metrics["burst_diag_walker_coordinates"] = position
            metrics["burst_diag_walker_amplitudes"] = data["walker_data"][
                "amplitude"
            ]
        if (reference_metrics
                or optimizer_config.get("update_diagnostics", False)
                or optimizer_config.get("reliability_diagnostics", False)
                or optimizer_config.get("experimental_mode", "none") != "none"
                or solution_recurrence_mode != "none"):
            metrics.update(
                {
                    "wssr_diag_lr_scaled_update_norm": unconstrained_update_norm,
                    "wssr_diag_norm_constraint_scale": jnp.where(
                        optimizer_config.constrain_norm, constraint_scale, 1.0
                    ),
                    "wssr_diag_constrained_update_norm": optax.global_norm(updates),
                }
            )
        if record_param_l1_norm:
            metrics.update({"param_l1_norm": tree_reduce_l1(params)})

        if isinstance(
            optimizer_state, WSSRErrorFeedbackRecurrenceOptimizerState
        ):
            next_optimizer_state = WSSRErrorFeedbackRecurrenceOptimizerState(
                core_state=core_state,
                optax_state=optax_state,
                solution_state=flat_grad_like_update,
                error_feedback_state=recurrence_diagnostics.error_feedback,
                error_feedback_clip_count=(
                    optimizer_state.error_feedback_clip_count
                    + recurrence_diagnostics.error_feedback_clip_increment
                ),
            )
        elif isinstance(
            optimizer_state, WSSRSolutionRecurrenceOptimizerState
        ):
            next_optimizer_state = WSSRSolutionRecurrenceOptimizerState(
                core_state=core_state,
                optax_state=optax_state,
                solution_state=flat_grad_like_update,
            )
        elif isinstance(
            optimizer_state, WSSRTransportedGradientOptimizerState
        ):
            next_optimizer_state = WSSRTransportedGradientOptimizerState(
                core_state=core_state,
                optax_state=optax_state,
                transported_gradient=transported_gradient,
                # Save theta_t. At the next call, theta_{t+1}-theta_t is the
                # actual update after learning-rate scaling and norm constraint.
                previous_theta=previous_theta,
                transport_initialized=jnp.asarray(True),
            )
        elif isinstance(
            optimizer_state, WSSRClusterEnvelopeEFOptimizerState
        ):
            next_optimizer_state = WSSRClusterEnvelopeEFOptimizerState(
                core_state=core_state,
                optax_state=optax_state,
                envelope_history=next_envelope_history,
                envelope_count=next_envelope_count,
                error_feedback_state=next_envelope_error_feedback,
                error_feedback_clip_count=(
                    optimizer_state.error_feedback_clip_count
                    + envelope_feedback_clip_increment
                ),
            )
        elif isinstance(optimizer_state, WSSRClusterEnvelopeOptimizerState):
            next_optimizer_state = WSSRClusterEnvelopeOptimizerState(
                core_state=core_state,
                optax_state=optax_state,
                envelope_history=next_envelope_history,
                envelope_count=next_envelope_count,
            )
        elif isinstance(optimizer_state, WSSRMultilevelComplementOptimizerState):
            ema_a, ema_b, weight_a, weight_b, step = complement_state
            next_optimizer_state = WSSRMultilevelComplementOptimizerState(
                core_state=core_state,
                optax_state=optax_state,
                complement_ema=ema_a,
                complement_ema_b=ema_b,
                complement_weight_a=weight_a,
                complement_weight_b=weight_b,
                complement_step=step,
            )
        elif isinstance(optimizer_state, WSSRComplementEMAOptimizerState):
            next_optimizer_state = WSSRComplementEMAOptimizerState(
                core_state=core_state,
                optax_state=optax_state,
                complement_ema=complement_state,
            )
        else:
            next_optimizer_state = WSSROptimizerState(
                core_state=core_state, optax_state=optax_state
            )
        return (
            params,
            data,
            next_optimizer_state,
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
                relative_singular_value_cutoff=optimizer_config.get(
                    "relative_singular_value_cutoff", -1.0
                ),
                tikhonov_lambda=optimizer_config.get("tikhonov_lambda", -1.0),
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
    use_multilevel = optimizer_config.get("multilevel_complement_period", 1) > 1
    use_native_proximal = (
        optimizer_config.get("experimental_mode", "none") == "native_proximal"
        and optimizer_config.get("native_proximal_gamma", 0.0) > 0.0
    )
    use_complement_ema = (
        optimizer_config.get("complement_state_decay", 0.0) > 0
        or use_native_proximal
    )
    solution_recurrence_mode = optimizer_config.get(
        "solution_recurrence_mode", "none"
    )
    enable_gradient_transport = optimizer_config.get(
        "enable_gradient_transport", False
    )
    eta_S, eta_g = resolve_wssr_averaging_weights(optimizer_config)
    use_reduced_metric_history = optimizer_config.get(
        "reduced_metric_history_mode", "none"
    ) != "none"
    use_anisotropic_matrix_history = optimizer_config.get(
        "anisotropic_matrix_history", False
    )
    use_independent_gradient_memory = (
        optimizer_config.get("adaptive_S_average", False)
        or optimizer_config.get("adaptive_g_average", False)
        or enable_gradient_transport
        or eta_S != eta_g
    ) and not (
        use_reduced_metric_history or use_anisotropic_matrix_history
    )
    use_cluster_envelope = optimizer_config.get(
        "experimental_mode", "none"
    ) in (
        "cluster_envelope",
        "cluster_envelope_ritz",
        "cluster_envelope_ritz_ef",
        "cluster_envelope_ritz_selected",
        "cluster_envelope_ritz_snr",
        "grassmann_ritz",
    )
    use_cluster_envelope_ef = (
        optimizer_config.get("experimental_mode", "none")
        == "cluster_envelope_ritz_ef"
    )
    if use_cluster_envelope:
        if (
            use_independent_gradient_memory
            or solution_recurrence_mode != "none"
            or use_multilevel
            or use_complement_ema
        ):
            raise ValueError(
                "cluster_envelope cannot be combined with gradient, solution, "
                "or complement memory"
            )
        cluster_rank = optimizer_config.get("cluster_envelope_rank", 32)
        envelope_length = optimizer_config.get(
            "cluster_envelope_history", 3
        )
        envelope_state_kwargs = dict(
            core_state=core_state,
            optax_state=optax_state,
            envelope_history=jnp.zeros(
                (
                    flat_params.shape[0],
                    cluster_rank * max(envelope_length - 1, 0),
                ),
                dtype=flat_params.dtype,
            ),
            envelope_count=jnp.asarray(0, dtype=jnp.int32),
        )
        if use_cluster_envelope_ef:
            optimizer_state = WSSRClusterEnvelopeEFOptimizerState(
                **envelope_state_kwargs,
                error_feedback_state=jnp.zeros_like(flat_params),
                error_feedback_clip_count=jnp.asarray(0, dtype=jnp.int32),
            )
        else:
            optimizer_state = WSSRClusterEnvelopeOptimizerState(
                **envelope_state_kwargs
            )
    elif use_independent_gradient_memory:
        if (
            solution_recurrence_mode != "none"
            or use_multilevel
            or use_complement_ema
        ):
            raise ValueError(
                "independent gradient memory cannot be combined with solution "
                "or complement state"
            )
        optimizer_state = WSSRTransportedGradientOptimizerState(
            core_state=core_state,
            optax_state=optax_state,
            transported_gradient=jnp.zeros_like(flat_params),
            previous_theta=flat_params,
            transport_initialized=jnp.asarray(False),
        )
    elif solution_recurrence_mode != "none":
        if use_multilevel or use_complement_ema:
            raise ValueError(
                "solution recurrence cannot be combined with complement state"
            )
        if optimizer_config.get("solution_error_feedback", False):
            optimizer_state = WSSRErrorFeedbackRecurrenceOptimizerState(
                core_state=core_state,
                optax_state=optax_state,
                solution_state=jnp.zeros_like(flat_params),
                error_feedback_state=jnp.zeros_like(flat_params),
                error_feedback_clip_count=jnp.asarray(0, dtype=jnp.int32),
            )
        else:
            optimizer_state = WSSRSolutionRecurrenceOptimizerState(
                core_state=core_state,
                optax_state=optax_state,
                solution_state=jnp.zeros_like(flat_params),
            )
    elif use_multilevel:
        zero_scalar = jnp.asarray(0.0, dtype=flat_params.dtype)
        optimizer_state = WSSRMultilevelComplementOptimizerState(
            core_state=core_state,
            optax_state=optax_state,
            complement_ema=jnp.zeros_like(flat_params),
            complement_ema_b=jnp.zeros_like(flat_params),
            complement_weight_a=zero_scalar,
            complement_weight_b=zero_scalar,
            complement_step=jnp.asarray(0, dtype=jnp.int32),
        )
    elif use_complement_ema:
        optimizer_state = WSSRComplementEMAOptimizerState(
            core_state=core_state,
            optax_state=optax_state,
            complement_ema=jnp.zeros_like(flat_params),
        )
    else:
        optimizer_state = WSSROptimizerState(
            core_state=core_state, optax_state=optax_state
        )
    return (
        update_param_fn,
        optimizer_state,
    )
