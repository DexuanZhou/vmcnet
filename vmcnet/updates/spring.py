"""SPRING implementation, see https://doi.org/10.1016/j.jcp.2024.113351."""

import os
from typing import Callable, Dict

import numpy as np
import jax
import jax.flatten_util
import jax.numpy as jnp
import neural_tangents as nt  # type: ignore
from ml_collections import ConfigDict
import chex
import optax

from vmcnet.utils.typing import Array, D, ModelApply, P, S, Tuple
from vmcnet.utils.pytree_helpers import (
    multiply_tree_by_scalar,
    tree_inner_product,
    tree_reduce_l1,
)
from vmcnet.utils.distribute import pmean_if_pmap
from vmcnet.utils.typing import UpdateDataFn, GetPositionFromData, LearningRateSchedule

from .update_param_fns import (
    UpdateParamFn,
    make_traced_fn_with_single_metrics,
    update_metrics_with_noclip,
)
from .optax_utils import initialize_optax_optimizer


def _tree_vector(tree):
    return jax.flatten_util.ravel_pytree(tree)[0]


def _tree_diagnostics(prefix, tree):
    vector = _tree_vector(tree)
    return {
        prefix + "_norm": jnp.linalg.norm(vector),
        prefix + "_max_abs": jnp.max(jnp.abs(vector)),
    }


def _tree_cosine(left, right, eps=1e-30):
    left_vec, right_vec = _tree_vector(left), _tree_vector(right)
    denom = jnp.linalg.norm(left_vec) * jnp.linalg.norm(right_vec)
    return jnp.where(denom > eps, jnp.vdot(left_vec, right_vec) / denom, 0.0)


def _save_spring_replay(directory, step, **arrays):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"epoch_{int(step):04d}.npz")
    np.savez_compressed(path, **{key: np.asarray(value) for key, value in arrays.items()})


def _float64_replay_solve(matrix, rhs, damping):
    """Host-only float64 replay; returned solve is cast for the unchanged f32 VJP."""
    matrix64 = np.asarray(matrix, dtype=np.float64)
    rhs64 = np.asarray(rhs, dtype=np.float64)
    damping64 = float(np.asarray(damping))
    values, vectors = np.linalg.eigh(matrix64)
    filt = np.maximum(values, 0.0) + damping64
    rhs_modes = vectors.T @ rhs64
    zeta = vectors @ (rhs_modes / filt)
    zeta -= np.mean(zeta)
    residual = np.linalg.norm(matrix64 @ zeta + damping64 * zeta - rhs64)
    residual /= max(np.linalg.norm(rhs64), 1e-300)
    low = values < damping64
    correction_weights = np.maximum(values, 0.0) * np.square(rhs_modes / filt)
    low_rhs = np.square(rhs_modes[low]).sum() / max(np.square(rhs_modes).sum(), 1e-300)
    low_correction = correction_weights[low].sum() / max(
        correction_weights.sum(), 1e-300
    )
    return (
        zeta.astype(np.float32),
        values.astype(np.float32),
        np.asarray(residual, dtype=np.float32),
        np.asarray(low_rhs, dtype=np.float32),
        np.asarray(low_correction, dtype=np.float32),
    )


def _float64_replay_decomposition(matrix, epsilon, projection, damping):
    matrix64 = np.asarray(matrix, dtype=np.float64)
    epsilon64 = np.asarray(epsilon, dtype=np.float64)
    projection64 = np.asarray(projection, dtype=np.float64)
    values, vectors = np.linalg.eigh(matrix64)
    filt = np.maximum(values, 0.0) + float(np.asarray(damping))

    def solve(rhs):
        value = vectors @ ((vectors.T @ rhs) / filt)
        value -= np.mean(value)
        return value.astype(np.float32)

    return solve(epsilon64), solve(projection64)


def _mixed_precision_sample_solves(
    uncentered_matrix, epsilon, history_projection, damping
):
    """Host float64 sample-space construction and eigensolve for diagnostics."""
    matrix = np.asarray(uncentered_matrix, dtype=np.float64)
    epsilon64 = np.asarray(epsilon, dtype=np.float64)
    projection64 = np.asarray(history_projection, dtype=np.float64)
    nchains = matrix.shape[0]
    matrix -= np.mean(matrix, axis=0, keepdims=True)
    matrix -= np.mean(matrix, axis=1, keepdims=True)
    matrix += np.ones((nchains, nchains), dtype=np.float64) / nchains
    matrix = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(matrix)
    filt = np.maximum(values, 0.0) + float(np.asarray(damping))

    def solve(rhs):
        value = vectors @ ((vectors.T @ rhs) / filt)
        return (value - np.mean(value)).astype(np.float32)

    return (
        solve(epsilon64 - projection64),
        solve(epsilon64),
        solve(projection64),
    )


def construct_spring_update_param_fn(
    energy_and_statistics_fn,
    optimizer_apply: Callable[[P, P, S, D, Dict[str, Array]], Tuple[P, S]],
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    apply_pmap: bool = True,
    record_param_l1_norm: bool = False,
) -> UpdateParamFn[P, D, S]:
    """Create the `update_param_fn` based on the gradient of the total energy."""

    def update_param_fn(params, data, optimizer_state, key):
        position = get_position_fn(data)

        energy, local_energies, stats = energy_and_statistics_fn(params, position)

        params, optimizer_state, spring_diagnostics = optimizer_apply(
            energy,
            local_energies,
            params,
            optimizer_state,
            data,
        )
        data = update_data_fn(data, params)

        metrics = {"energy": energy, "variance": stats["variance"]}
        metrics.update(spring_diagnostics)
        metrics = update_metrics_with_noclip(
            stats["energy_noclip"],
            stats["variance_noclip"],
            metrics,
        )
        if record_param_l1_norm:
            metrics.update({"param_l1_norm": tree_reduce_l1(params)})
        return params, data, optimizer_state, metrics, key

    traced_fn = make_traced_fn_with_single_metrics(update_param_fn, apply_pmap)

    return traced_fn


def initialize_spring(
    log_psi_apply: ModelApply[P],
    energy_and_statistics_fn,
    params: P,
    get_position_fn: GetPositionFromData[D],
    update_data_fn: UpdateDataFn[D, P],
    learning_rate_schedule: LearningRateSchedule,
    optimizer_config: ConfigDict,
    record_param_l1_norm: bool = False,
    apply_pmap: bool = True,
) -> Tuple[UpdateParamFn[P, D, optax.OptState], optax.OptState]:
    """Get an update param function and initial state for SPRING."""
    spring_step = get_spring_step(
        log_psi_apply,
        optimizer_config.damping,
        optimizer_config.mu,
        diagnostics_spectral=optimizer_config.get("diagnostics_spectral", False),
        diagnostics_replay_epochs=tuple(
            optimizer_config.get("diagnostics_replay_epochs", ())
        ),
        diagnostics_replay_dir=optimizer_config.get("diagnostics_replay_dir", ""),
        diagnostics_decomposition=optimizer_config.get(
            "diagnostics_decomposition", False
        ),
        mixed_precision_solve=optimizer_config.get("mixed_precision_solve", False),
    )

    descent_optimizer = optax.sgd(
        learning_rate=learning_rate_schedule, momentum=0, nesterov=False
    )

    def prev_update(optimizer_state):
        return optimizer_state[0].trace

    def optimizer_apply(energy, local_energies, params, optimizer_state, data):
        positions = get_position_fn(data)

        centered_local_energies = local_energies - energy
        diagnostics_enabled = optimizer_config.get("diagnostics", False)
        diagnostic_step = optimizer_state[1].count + 1
        spring_result = spring_step(
            centered_local_energies,
            params,
            prev_update(optimizer_state),
            positions,
            return_diagnostics=diagnostics_enabled,
            diagnostic_step=diagnostic_step,
        )
        if diagnostics_enabled:
            grad, diagnostics = spring_result
        else:
            grad, diagnostics = spring_result, {}

        updates_unconstrained, optimizer_state = descent_optimizer.update(
            grad, optimizer_state, params
        )
        updates = updates_unconstrained

        if optimizer_config.constrain_norm:
            updates = constrain_norm(
                updates,
                optimizer_config.norm_constraint,
            )

        new_params = optax.apply_updates(params, updates)
        if diagnostics_enabled:
            trace = prev_update(optimizer_state)
            displacement = jax.tree_map(lambda new, old: new - old, new_params, params)
            update_norm = jnp.linalg.norm(_tree_vector(updates_unconstrained))
            constrained_norm = jnp.linalg.norm(_tree_vector(updates))
            diagnostics.update(_tree_diagnostics("spring_diag_optax_trace", trace))
            diagnostics.update(
                _tree_diagnostics("spring_diag_lr_update", updates_unconstrained)
            )
            diagnostics.update(
                _tree_diagnostics("spring_diag_constrained_update", updates)
            )
            diagnostics.update(_tree_diagnostics("spring_diag_displacement", displacement))
            diagnostics["spring_diag_norm_constraint_scale"] = jnp.where(
                update_norm > 0, constrained_norm / update_norm, 1.0
            )
            history = diagnostics.pop("spring_diag_history_tree")
            diagnostics["spring_diag_cos_history_raw"] = _tree_cosine(history, grad)
            diagnostics["spring_diag_cos_history_nonhistory"] = _tree_cosine(
                history, diagnostics.pop("spring_diag_nonhistory_tree")
            )
            diagnostics["spring_diag_cos_history_displacement"] = _tree_cosine(
                history, displacement
            )
            diagnostics["spring_diag_cos_trace_raw"] = _tree_cosine(trace, grad)
            diagnostics["spring_diag_cos_trace_displacement"] = _tree_cosine(
                trace, displacement
            )
            diagnostics["spring_diag_cos_raw_displacement"] = _tree_cosine(
                grad, displacement
            )
            raw_norm = diagnostics["spring_diag_raw_solution_norm"]
            trace_norm = diagnostics["spring_diag_optax_trace_norm"]
            disp_norm = diagnostics["spring_diag_displacement_norm"]
            nonhistory_norm = diagnostics["spring_diag_nonhistory_norm"]
            mu_history_norm = diagnostics["spring_diag_mu_history_norm"]
            diagnostics["spring_diag_trace_over_raw"] = trace_norm / jnp.maximum(
                raw_norm, 1e-30
            )
            diagnostics["spring_diag_displacement_over_trace"] = (
                disp_norm / jnp.maximum(trace_norm, 1e-30)
            )
            diagnostics["spring_diag_mu_history_over_nonhistory"] = (
                mu_history_norm / jnp.maximum(nonhistory_norm, 1e-30)
            )
            diagnostics["spring_diag_raw_over_nonhistory"] = (
                raw_norm / jnp.maximum(nonhistory_norm, 1e-30)
            )
            diagnostics["spring_diag_displacement_over_raw"] = (
                disp_norm / jnp.maximum(raw_norm, 1e-30)
            )
            replay_norm = diagnostics.get(
                "spring_replay_float64_direction_norm", jnp.nan
            )
            effective_lr = update_norm / jnp.maximum(raw_norm, 1e-30)
            diagnostics["spring_replay_float64_constraint_scale"] = jnp.minimum(
                effective_lr,
                jnp.sqrt(optimizer_config.norm_constraint)
                / jnp.maximum(replay_norm, 1e-30),
            )

        return new_params, optimizer_state, diagnostics

    update_param_fn = construct_spring_update_param_fn(
        energy_and_statistics_fn,
        optimizer_apply,
        get_position_fn=get_position_fn,
        update_data_fn=update_data_fn,
        record_param_l1_norm=record_param_l1_norm,
        apply_pmap=apply_pmap,
    )
    optimizer_state = initialize_optax_optimizer(
        descent_optimizer, params, apply_pmap=apply_pmap
    )

    return update_param_fn, optimizer_state


def get_spring_step(
    log_psi_apply: ModelApply[P],
    damping: chex.Scalar = 0.001,
    mu: chex.Scalar = 0.99,
    diagnostics_spectral: bool = False,
    diagnostics_replay_epochs=(),
    diagnostics_replay_dir: str = "",
    diagnostics_decomposition: bool = False,
    mixed_precision_solve: bool = False,
):
    """Get the SPRING update function."""
    kernel_fn = nt.empirical_kernel_fn(log_psi_apply, vmap_axes=0, trace_axes=())

    def spring_step(
        centered_energies: P,
        params: P,
        prev_grad,
        positions: Array,
        return_diagnostics: bool = False,
        diagnostic_step=0,
    ) -> Tuple[Array, P]:
        nchains = positions.shape[0]
        solve_positions, solve_params, solve_prev_grad = positions, params, prev_grad
        mu_prev = jax.tree_map(lambda x: mu * x, solve_prev_grad)
        ones = jnp.ones((nchains, 1), dtype=positions.dtype)

        # Calculate T = Ohat @ Ohat^T using neural-tangents
        # Some GPUs, particularly A100s and A5000s, can exhibit large numerical
        # errors in these calculations. As a result, we explicitly symmetrize T
        # and, rather than using a Cholesky solver to solve against T, we
        # calculate its eigendecomposition and explicitly fix any negative
        # eigenvalues. We then use the fixed and regularized igendecomposition
        # to solve against T. This appears to be more stable than Cholesky
        # in practice.
        T_uncentered = (
            kernel_fn(solve_positions, solve_positions, "ntk", solve_params)
            / nchains
        )
        T = T_uncentered
        T = T - jnp.mean(T, axis=0, keepdims=True)
        T = T - jnp.mean(T, axis=1, keepdims=True)
        T = T + ones @ ones.T / nchains
        T = (T + T.T) / 2
        raw_Tvals, Tvecs = jnp.linalg.eigh(T)
        Tvals = jnp.maximum(raw_Tvals, 0) + damping

        epsilon_bar = centered_energies / jnp.sqrt(nchains)
        O_prev = jax.jvp(
            log_psi_apply,
            (solve_params, solve_positions),
            (mu_prev, jnp.zeros_like(solve_positions)),
        )[1] / jnp.sqrt(nchains)
        Ohat_prev = O_prev - jnp.mean(O_prev, axis=0, keepdims=True)
        epsilon_tilde = epsilon_bar - Ohat_prev

        zeta = Tvecs @ jnp.diag(1 / Tvals) @ Tvecs.T @ epsilon_tilde
        zeta_hat = zeta - jnp.mean(zeta)
        mixed_current = mixed_projection = None
        if mixed_precision_solve:
            zeta_hat, mixed_current, mixed_projection = jax.pure_callback(
                _mixed_precision_sample_solves,
                (
                    jax.ShapeDtypeStruct(epsilon_bar.shape, jnp.float32),
                    jax.ShapeDtypeStruct(epsilon_bar.shape, jnp.float32),
                    jax.ShapeDtypeStruct(epsilon_bar.shape, jnp.float32),
                ),
                T_uncentered,
                epsilon_bar,
                Ohat_prev,
                jnp.asarray(damping, dtype=T_uncentered.dtype),
            )
        dtheta_residual = jax.vjp(log_psi_apply, solve_params, solve_positions)[1](zeta_hat)[0]

        nonhistory = jax.tree_map(
            lambda dt: dt / jnp.sqrt(nchains), dtheta_residual
        )
        raw_solution = jax.tree_map(
            lambda dt, mup: dt / jnp.sqrt(nchains) + mup, dtheta_residual, mu_prev
        )
        decomposition = None
        if diagnostics_decomposition:
            zeta_current = Tvecs @ ((Tvecs.T @ epsilon_bar) / Tvals)
            zeta_current = zeta_current - jnp.mean(zeta_current)
            if mixed_current is not None:
                zeta_current = mixed_current
            current_residual = jax.vjp(
                log_psi_apply, solve_params, solve_positions
            )[1](zeta_current)[0]
            current_term = jax.tree_map(
                lambda value: value / jnp.sqrt(nchains), current_residual
            )
            zeta_projection = Tvecs @ ((Tvecs.T @ Ohat_prev) / Tvals)
            zeta_projection = zeta_projection - jnp.mean(zeta_projection)
            if mixed_projection is not None:
                zeta_projection = mixed_projection
            projected_residual = jax.vjp(
                log_psi_apply, solve_params, solve_positions
            )[1](zeta_projection)[0]
            projected_history = jax.tree_map(
                lambda value: value / jnp.sqrt(nchains), projected_residual
            )
            history_term = jax.tree_map(
                lambda history, projected: history - projected,
                mu_prev,
                projected_history,
            )
            decomposition = (current_term, history_term)
        if not return_diagnostics:
            return raw_solution
        diagnostics = {}
        diagnostics.update(_tree_diagnostics("spring_diag_input", centered_energies))
        diagnostics.update(_tree_diagnostics("spring_diag_history", prev_grad))
        diagnostics.update(_tree_diagnostics("spring_diag_mu_history", mu_prev))
        diagnostics.update(_tree_diagnostics("spring_diag_nonhistory", nonhistory))
        diagnostics.update(_tree_diagnostics("spring_diag_raw_solution", raw_solution))
        diagnostics["spring_diag_history_projection_norm"] = jnp.linalg.norm(
            Ohat_prev
        )
        if decomposition is not None:
            current_term, history_term = decomposition
            current_norm = jnp.linalg.norm(_tree_vector(current_term))
            history_term_norm = jnp.linalg.norm(_tree_vector(history_term))
            epsilon_norm = jnp.linalg.norm(epsilon_bar)
            projection_norm = jnp.linalg.norm(Ohat_prev)
            diagnostics.update(_tree_diagnostics("spring_decomp_current", current_term))
            diagnostics.update(_tree_diagnostics("spring_decomp_history", history_term))
            diagnostics["spring_decomp_epsilon_norm"] = epsilon_norm
            diagnostics["spring_decomp_projection_norm"] = projection_norm
            diagnostics["spring_decomp_rhs_ratio"] = projection_norm / jnp.maximum(
                epsilon_norm, 1e-30
            )
            diagnostics["spring_decomp_true_history_ratio"] = (
                history_term_norm / jnp.maximum(current_norm, 1e-30)
            )
            diagnostics["spring_decomp_cos_current_history"] = _tree_cosine(
                current_term, history_term
            )
            positive_max = jnp.maximum(jnp.max(raw_Tvals), 0)
            diagnostics["spring_contractivity_operator_norm_estimate"] = 1.0
            diagnostics["spring_contractivity_mu_operator_norm_estimate"] = abs(mu)
            diagnostics["spring_contractivity_symmetric_max_eigenvalue"] = 1.0
            diagnostics["spring_contractivity_symmetric_min_eigenvalue"] = (
                damping / (positive_max + damping)
            )
        diagnostics["spring_diag_history_tree"] = prev_grad
        diagnostics["spring_diag_nonhistory_tree"] = nonhistory
        if diagnostics_spectral:
            rhs_modes = Tvecs.T @ epsilon_tilde
            history_modes = Tvecs.T @ Ohat_prev
            low = raw_Tvals < damping
            positive = raw_Tvals > 0
            positive_eigs = jnp.where(positive, raw_Tvals, jnp.inf)
            correction_weights = jnp.maximum(raw_Tvals, 0) * jnp.square(
                rhs_modes / Tvals
            )
            residual = T @ zeta + damping * zeta - epsilon_tilde
            diagnostics.update(
                {
                    "spring_spec_min_eigenvalue": jnp.min(raw_Tvals),
                    "spring_spec_max_eigenvalue": jnp.max(raw_Tvals),
                    "spring_spec_min_positive_eigenvalue": jnp.min(positive_eigs),
                    "spring_spec_negative_count": jnp.sum(raw_Tvals < 0),
                    "spring_spec_below_1e8_count": jnp.sum(raw_Tvals < 1e-8),
                    "spring_spec_below_1e6_count": jnp.sum(raw_Tvals < 1e-6),
                    "spring_spec_below_1e4_count": jnp.sum(raw_Tvals < 1e-4),
                    "spring_spec_below_damping_count": jnp.sum(raw_Tvals < damping),
                    "spring_spec_below_10damping_count": jnp.sum(
                        raw_Tvals < 10 * damping
                    ),
                    "spring_spec_effective_condition": jnp.max(Tvals)
                    / jnp.min(Tvals),
                    "spring_spec_max_inverse_filter": jnp.max(1 / Tvals),
                    "spring_spec_rhs_norm": jnp.linalg.norm(epsilon_tilde),
                    "spring_spec_relative_residual": jnp.linalg.norm(residual)
                    / jnp.maximum(jnp.linalg.norm(epsilon_tilde), 1e-30),
                    "spring_spec_low_rhs_fraction": jnp.sum(
                        jnp.where(low, jnp.square(rhs_modes), 0)
                    )
                    / jnp.maximum(jnp.sum(jnp.square(rhs_modes)), 1e-30),
                    "spring_spec_low_correction_fraction": jnp.sum(
                        jnp.where(low, correction_weights, 0)
                    )
                    / jnp.maximum(jnp.sum(correction_weights), 1e-30),
                    "spring_spec_low_history_projection_fraction": jnp.sum(
                        jnp.where(low, jnp.square(history_modes), 0)
                    )
                    / jnp.maximum(jnp.sum(jnp.square(history_modes)), 1e-30),
                }
            )

            replay_epochs = jnp.asarray(diagnostics_replay_epochs, dtype=jnp.int32)
            do_replay = jnp.any(replay_epochs == diagnostic_step)
            damping_training_dtype = jnp.asarray(damping, dtype=raw_Tvals.dtype)

            def replay(_):
                zeta64, vals64, residual64, low_rhs64, low_correction64 = (
                    jax.pure_callback(
                        _float64_replay_solve,
                        (
                            jax.ShapeDtypeStruct(epsilon_tilde.shape, jnp.float32),
                            jax.ShapeDtypeStruct(raw_Tvals.shape, jnp.float32),
                            jax.ShapeDtypeStruct((), jnp.float32),
                            jax.ShapeDtypeStruct((), jnp.float32),
                            jax.ShapeDtypeStruct((), jnp.float32),
                        ),
                        T,
                        epsilon_tilde,
                        damping_training_dtype,
                    )
                )
                nonhistory64 = jax.vjp(log_psi_apply, params, positions)[1](
                    zeta64.astype(positions.dtype)
                )[0]
                nonhistory64 = jax.tree_map(
                    lambda value: value / jnp.sqrt(nchains), nonhistory64
                )
                raw64 = jax.tree_map(
                    lambda value, history: value + history,
                    nonhistory64,
                    mu_prev,
                )
                if diagnostics_decomposition:
                    zeta_current64, zeta_projection64 = jax.pure_callback(
                        _float64_replay_decomposition,
                        (
                            jax.ShapeDtypeStruct(epsilon_bar.shape, jnp.float32),
                            jax.ShapeDtypeStruct(Ohat_prev.shape, jnp.float32),
                        ),
                        T,
                        epsilon_bar,
                        Ohat_prev,
                        damping_training_dtype,
                    )
                    current64_residual = jax.vjp(
                        log_psi_apply, solve_params, solve_positions
                    )[1](zeta_current64.astype(solve_positions.dtype))[0]
                    projection64_residual = jax.vjp(
                        log_psi_apply, solve_params, solve_positions
                    )[1](zeta_projection64.astype(solve_positions.dtype))[0]
                    current64 = jax.tree_map(
                        lambda value: value / jnp.sqrt(nchains), current64_residual
                    )
                    projected64 = jax.tree_map(
                        lambda value: value / jnp.sqrt(nchains), projection64_residual
                    )
                    history64 = jax.tree_map(
                        lambda history, projected: history - projected,
                        mu_prev,
                        projected64,
                    )
                    phi64 = jax.tree_map(
                        lambda current, history: current + history,
                        current64,
                        history64,
                    )
                    current32, history32 = decomposition
                    current_relative_error = (
                        jnp.linalg.norm(_tree_vector(current32) - _tree_vector(current64))
                        / jnp.maximum(jnp.linalg.norm(_tree_vector(current64)), 1e-30)
                    )
                    current_cosine = _tree_cosine(
                        current32, current64
                    )
                    history_relative_error = (
                        jnp.linalg.norm(_tree_vector(history32) - _tree_vector(history64))
                        / jnp.maximum(jnp.linalg.norm(_tree_vector(history64)), 1e-30)
                    )
                    history_cosine = _tree_cosine(
                        history32, history64
                    )
                    total_relative_error = (
                        jnp.linalg.norm(_tree_vector(raw_solution) - _tree_vector(phi64))
                        / jnp.maximum(jnp.linalg.norm(_tree_vector(phi64)), 1e-30)
                    )
                    total_cosine = _tree_cosine(
                        raw_solution, phi64
                    )
                    decomposition_replay = jnp.asarray(
                        [current_relative_error, current_cosine,
                         history_relative_error, history_cosine,
                         total_relative_error, total_cosine,
                         jnp.linalg.norm(_tree_vector(current32)),
                         jnp.linalg.norm(_tree_vector(current64)),
                         jnp.linalg.norm(_tree_vector(history32)),
                         jnp.linalg.norm(_tree_vector(history64)),
                         jnp.linalg.norm(_tree_vector(raw_solution)),
                         jnp.linalg.norm(_tree_vector(phi64))],
                        dtype=raw_Tvals.dtype,
                    )
                else:
                    decomposition_replay = jnp.full(
                        (12,), jnp.nan, dtype=raw_Tvals.dtype
                    )
                raw32_vec, raw64_vec = _tree_vector(raw_solution), _tree_vector(raw64)
                relerr = jnp.linalg.norm(raw32_vec - raw64_vec) / jnp.maximum(
                    jnp.linalg.norm(raw64_vec), 1e-30
                )
                cosine = _tree_cosine(raw_solution, raw64)
                if diagnostics_replay_dir:
                    jax.debug.callback(
                        lambda step, matrix, rhs, history_projection, eig32, eig64, q32, q64: _save_spring_replay(
                            diagnostics_replay_dir,
                            step,
                            matrix=matrix,
                            rhs=rhs,
                            history_projection=history_projection,
                            eigenvalues_float32=eig32,
                            eigenvalues_float64=eig64,
                            raw_direction_float32=q32,
                            raw_direction_float64=q64,
                        ),
                        diagnostic_step,
                        T,
                        epsilon_tilde,
                        Ohat_prev,
                        raw_Tvals,
                        vals64,
                        raw32_vec,
                        raw64_vec,
                    )
                base_replay = jnp.asarray(
                    [
                        jnp.linalg.norm(raw64_vec),
                        relerr,
                        cosine,
                        residual64,
                        low_rhs64,
                        low_correction64,
                    ],
                    dtype=raw_Tvals.dtype,
                )
                return jnp.concatenate((base_replay, decomposition_replay))

            replay_result = jax.lax.cond(
                do_replay,
                replay,
                lambda _: jnp.full((18,), jnp.nan, dtype=raw_Tvals.dtype),
                operand=None,
            )
            diagnostics["spring_replay_float64_direction_norm"] = replay_result[0]
            diagnostics["spring_replay_relative_direction_error"] = replay_result[1]
            diagnostics["spring_replay_direction_cosine"] = replay_result[2]
            diagnostics["spring_replay_float64_relative_residual"] = replay_result[3]
            diagnostics["spring_replay_float64_low_rhs_fraction"] = replay_result[4]
            diagnostics["spring_replay_float64_low_correction_fraction"] = (
                replay_result[5]
            )
            diagnostics["spring_replay_current_relative_error"] = replay_result[6]
            diagnostics["spring_replay_current_cosine"] = replay_result[7]
            diagnostics["spring_replay_history_relative_error"] = replay_result[8]
            diagnostics["spring_replay_history_cosine"] = replay_result[9]
            diagnostics["spring_replay_total_relative_error"] = replay_result[10]
            diagnostics["spring_replay_total_cosine"] = replay_result[11]
            diagnostics["spring_replay_current_float32_norm"] = replay_result[12]
            diagnostics["spring_replay_current_float64_norm"] = replay_result[13]
            diagnostics["spring_replay_history_float32_norm"] = replay_result[14]
            diagnostics["spring_replay_history_float64_norm"] = replay_result[15]
            diagnostics["spring_replay_total_float32_norm"] = replay_result[16]
            diagnostics["spring_replay_total_float64_norm"] = replay_result[17]
        return raw_solution, diagnostics

    return spring_step


def constrain_norm(
    grad: P,
    norm_constraint: chex.Numeric = 0.001,
) -> P:
    """Euclidean norm constraint."""
    sq_norm_scaled_grads = tree_inner_product(grad, grad)

    # Sync the norms here, see:
    # https://github.com/deepmind/deepmind-research/blob/30799687edb1abca4953aec507be87ebe63e432d/kfac_ferminet_alpha/optimizer.py#L585
    sq_norm_scaled_grads = pmean_if_pmap(sq_norm_scaled_grads)

    norm_scale_factor = jnp.sqrt(norm_constraint / sq_norm_scaled_grads)
    coefficient = jnp.minimum(norm_scale_factor, 1)
    constrained_grads = multiply_tree_by_scalar(grad, coefficient)

    return constrained_grads
