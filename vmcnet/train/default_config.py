"""Create configuration of hyperparameters."""

import os
from typing import Dict

from ml_collections import ConfigDict

from vmcnet.utils.checkpoint import DEFAULT_CHECKPOINT_FILE_NAME

NO_NAME = "NONE"
NO_PATH = "NONE"
NO_RELOAD_LOG_DIR = "NONE"
DEFAULT_CONFIG_FILE_NAME = "config.json"
DEFAULT_PRESETS_DIR = "preset_configs"


def _copy_all_dicts(config: Dict) -> Dict:
    """Recursively copy the top-level Dict and all sub-Dicts.

    This ensures that command line flags used to override fields of the config will only
    apply to the intended flag, even if the code used to generate the default values is
    shared among several different configuration flags. For example, the following code
    could be problematic:

    subconfig = {"val": 2}
    config = {
        "sub1": subconfig,
        "sub2": subconfig
    }

    If this config is used directly, setting a command line flag like
    --config.sub1.val=3 will override both config.sub1.val and config.sub2.val, which is
    not the intended behavior. Calling config=copy_all_dicts(config) before turning this
    into a ConfigDict solves the problem by making separate copies of both subconfigs.

    Note: using copy.deepcopy is not a valid replacement for this method, as deepcopy
    will not generate multiple copies of the same object if encountered multiple times.
    In the example above, deepcopy will make a single copy subconfig_copy of subconfig,
    and use it for both sub1 and sub2 in the returned copy. This does not solve the
    problem!
    """
    result = {}
    for key in config:
        if isinstance(config[key], Dict):
            result[key] = _copy_all_dicts(config[key])
        else:
            result[key] = config[key]
    return result


def get_default_reload_config() -> ConfigDict:
    """Make a default reload configuration (no logdir but valid defaults otherwise)."""
    return ConfigDict(
        {
            "logdir": NO_RELOAD_LOG_DIR,
            "use_config_file": True,
            "config_relative_file_path": DEFAULT_CONFIG_FILE_NAME,
            "use_checkpoint_file": True,
            "checkpoint_relative_file_path": DEFAULT_CHECKPOINT_FILE_NAME,
            "new_optimizer_state": False,
            "reburn": False,
            "append": True,
            "same_logdir": False,
        }
    )


def get_default_config() -> ConfigDict:
    """Make a default configuration (single det FermiNet on LiH)."""
    config = ConfigDict(
        _copy_all_dicts(
            {
                "notes": "default",
                "problem": get_default_molecular_config(),
                "model": get_default_model_config(),
                "vmc": get_default_vmc_config(),
                "eval": get_default_eval_config(),
                "logdir": os.path.join(
                    os.curdir,  # this will be relative to the calling script
                    "logs",
                ),
                # if save_to_current_datetime_subfolder=True, will log into a subfolder
                # named according to the datetime at start
                "save_to_current_datetime_subfolder": True,
                "subfolder_name": NO_NAME,
                "logging_level": "INFO",
                "dtype": "float32",
                "distribute": False,
                "debug_nans": False,  # If true, OVERRIDES config.distribute to be False
                "initial_seed": 0,
                "wandb": {
                    "mode": "disabled",
                    "project": "default",
                    "name": "vmc-molecule_run",
                    "group": "default",
                },
            }
        )
    )
    config.base_logdir = config.logdir
    return config


def choose_model_type_in_model_config(model_config):
    """Given a model config with a specified type, select the specified model.

    The default config contains architecture hyperparameters for several types of models
    (in order to support command-line overwriting via absl.flags), but only one needs to
    be retained after the model type is chosen at the beginning of a run, so this
    function returns a ConfigDict with only the hyperparams associated with the model in
    model_config.type.
    """
    model_type = model_config.type
    model_config = model_config[model_type]
    model_config.type = model_type
    return model_config


def get_default_model_config() -> Dict:
    """Get a default model configuration from a model type."""
    orthogonal_init = {"type": "orthogonal", "scale": 1.0}
    normal_init = {"type": "normal"}

    input_streams = {
        "include_2e_stream": True,
        "include_ei_norm": True,
        "ei_norm_softening": 0.0,
        "include_ee_norm": True,
        "ee_norm_softening": 0.0,
    }

    base_backflow_config = {
        "kernel_init_unmixed": {"type": "orthogonal", "scale": 2.0},
        "kernel_init_mixed": orthogonal_init,
        "kernel_init_2e_1e_stream": orthogonal_init,
        "kernel_init_2e_2e_stream": {"type": "orthogonal", "scale": 2.0},
        "bias_init_1e_stream": normal_init,
        "bias_init_2e_stream": normal_init,
        "activation_fn": "tanh",
        "use_bias": True,
        "one_electron_skip": True,
        "one_electron_skip_scale": 1.0,
        "two_electron_skip": True,
        "two_electron_skip_scale": 1.0,
    }

    ferminet_backflow = {
        "ndense_list": ((256, 16), (256, 16), (256, 16), (256,)),
        **base_backflow_config,
    }

    base_ferminet_config = {
        "input_streams": input_streams,
        "backflow": ferminet_backflow,
        "ndeterminants": 16,
        "kernel_init_orbital_linear": {"type": "orthogonal", "scale": 2.0},
        "kernel_init_envelope_dim": {"type": "ones"},
        "kernel_init_envelope_ion": {"type": "ones"},
        "envelope_softening": 0.0,
        "bias_init_orbital_linear": normal_init,
        "orbitals_use_bias": True,
        "isotropic_decay": True,
        "full_det": True,
        "include_cusp_jastrow": False,
    }

    config = {
        "type": "ferminet",
        "ferminet": base_ferminet_config,
        "explicit_antisym": {
            "input_streams": input_streams,
            "backflow": ferminet_backflow,
            "antisym_type": "generic",  # factorized or generic
            "rank": 1,  # Only relevant for antisym_type=factorized
            "ndense_resnet": 64,
            "nlayers_resnet": 2,
            "kernel_init_resnet": {"type": "orthogonal", "scale": 2.0},
            "bias_init_resnet": normal_init,
            "activation_fn_resnet": "tanh",
            "resnet_use_bias": True,
            "jastrow": {
                # type must be a value in models.jastrow.VALID_JASTROW_TYPES
                "type": "backflow_based",
                "one_body_decay": {"kernel_init": {"type": "ones"}},
                "two_body_decay": {"init_ee_strength": 1.0, "trainable": True},
                "backflow_based": {
                    "use_separate_jastrow_backflow": True,
                    "backflow": {
                        "ndense_list": ((256, 16), (256, 16), (256, 16), (256,)),
                        "kernel_init_unmixed": orthogonal_init,
                        "kernel_init_mixed": orthogonal_init,
                        "kernel_init_2e_1e_stream": orthogonal_init,
                        "kernel_init_2e_2e_stream": orthogonal_init,
                        "bias_init_1e_stream": normal_init,
                        "bias_init_2e_stream": normal_init,
                        "activation_fn": "gelu",
                        "use_bias": True,
                        "one_electron_skip": True,
                        "one_electron_skip_scale": 1.0,
                        "two_electron_skip": True,
                        "two_electron_skip_scale": 1.0,
                    },
                },
            },
        },
    }
    return config


def get_default_molecular_config() -> Dict:
    """Get a default molecular configuration (LiH)."""
    problem_config = {
        "ion_pos": ((0.0, 0.0, -1.5069621), (0.0, 0.0, 1.5069621)),
        "ion_charges": (1.0, 3.0),
        "nelec": (2, 2),
        "ei_softening": 0.0,
        "ee_softening": 0.0,
    }
    return problem_config


def get_default_vmc_config() -> Dict:
    """Get a default VMC training configuration."""
    vmc_config = {
        "nchains": 1000,
        "nepochs": 200000,
        "nburn": 5000,
        "nsteps_per_param_update": 10,
        "nmoves_per_width_update": 100,
        "std_move": 0.25,
        "checkpoint_every": 5000,
        "best_checkpoint_every": 100,
        "checkpoint_dir": "checkpoints",
        "checkpoint_variance_scale": 10,
        "disable_checkpointing": False,
        "check_for_nans": False,
        # Rare-event snapshotting. Disabled means no device transfer, I/O, or
        # optimizer-path change. Thresholds are meaningful only when enabled.
        "burst_diagnostics": {
            "enabled": False,
            "raw_variance_threshold": float("inf"),
            "raw_clipped_ratio_threshold": float("inf"),
            "output_dir": "burst_diagnostics",
        },
        "nhistory_max": 200,
        "record_amplitudes": False,
        "record_param_l1_norm": False,
        "clip_threshold": 5.0,
        "clip_center": "mean",  # mean or median
        "nan_safe": True,
        "optimizer_type": "spring",
        "optimizer": {
            "kfac": {
                "l2_reg": 0.0,
                "norm_constraint": 0.001,
                "curvature_ema": 0.95,
                "inverse_update_period": 1,
                "min_damping": 1e-4,
                "register_only_generic": False,
                "estimation_mode": "fisher_exact",
                "damping": 0.001,
                "schedule_type": "inverse_time",  # constant or inverse_time
                "learning_rate": 5e-2,
                "learning_decay_rate": 1e-4,
            },
            "adam": {
                "b1": 0.9,
                "b2": 0.999,
                "eps": 1e-8,
                "eps_root": 0.0,
                "schedule_type": "inverse_time",  # constant or inverse_time
                "learning_rate": 5e-2,
                "learning_decay_rate": 1e-4,
            },
            "sgd": {
                "momentum": 0.0,
                "nesterov": False,
                "schedule_type": "inverse_time",  # constant or inverse_time
                "learning_rate": 5e-2,
                "learning_decay_rate": 1e-4,
            },
            "spring": {
                # Learning rate settings
                "schedule_type": "inverse_time",  # constant or inverse_time
                "learning_rate": 5e-2,
                "learning_decay_rate": 1e-4,
                # SPRING hyperparams
                "mu": 0.99,
                "damping": 0.001,
                "constrain_norm": True,
                "norm_constraint": 0.001,
                # Default preserves the paper's Euclidean update constraint.
                # In function_space mode, the direct radius below bounds
                # ||O_bar delta_theta||_2 on the current walker batch.
                "norm_constraint_mode": "euclidean",
                "function_norm_constraint": 0.001,
                # Disabled-by-default per-step state/scale diagnostics.
                "diagnostics": False,
                "diagnostics_spectral": False,
                "diagnostics_replay_epochs": (),
                "diagnostics_replay_dir": "",
                "diagnostics_decomposition": False,
                "mixed_precision_solve": False,
            },
            "wssr_svd": {
                # Learning rate settings
                "schedule_type": "inverse_time",  # constant or inverse_time
                "learning_rate": 5e-2,
                "learning_decay_rate": 1e-4,
                # WSSR SVD hyperparams
                "damping": 0.001,
                "constrain_norm": True,
                "norm_constraint": 0.001,
                "eta": 0.99,
                "sr_rank": 10,
                "sr_rank_max": 100,
                "sr_scale": 1.1,
            },
            "wssr_sketch": {
                # Learning rate settings
                "schedule_type": "inverse_time",  # constant or inverse_time
                "learning_rate": 5e-2,
                "learning_decay_rate": 1e-4,
                # WSSR sketch hyperparams
                "damping": 0.001,
                "constrain_norm": True,
                "norm_constraint": 0.001,
                "eta": 0.99,
                "sr_rank": 10,
                "sr_rank_max": 100,
                "sr_scale": 1.1,
                "sketch_oversampling": 5,
                "sketch_n_iter": 1,
            },
            "wssr_warm_svd": {
                # Learning rate settings
                "schedule_type": "inverse_time",  # constant or inverse_time
                "learning_rate": 5e-2,
                "learning_decay_rate": 1e-4,
                # WSSR warm-start SVD hyperparams
                "damping": 0.001,
                "constrain_norm": True,
                "norm_constraint": 0.001,
                "eta": 0.99,
                "sr_rank": 10,
                "sr_rank_max": 100,
                "sr_scale": 1.1,
                "svd_maxiter_initial": 8,
                "svd_maxiter_warm": 2,
                "svd_working_rank": -1,
                "store_warm_u": True,
            },
            "wssr_warm_svd_right": {
                # Learning rate settings
                "schedule_type": "inverse_time",  # constant or inverse_time
                "learning_rate": 5e-2,
                "learning_decay_rate": 1e-4,
                # WSSR right-subspace warm-start SVD hyperparams
                "damping": 0.001,
                # Optional independent spectral controls. Negative values keep
                # the historical coupling to ``damping`` exactly: damping is
                # both the relative singular-value cutoff and defines
                # lambda=(damping*sigma_max)^2. Set nonnegative values only in
                # controlled experiments.
                "relative_singular_value_cutoff": -1.0,
                "tikhonov_lambda": -1.0,
                "constrain_norm": True,
                "norm_constraint": 0.001,
                # Optional matrix-free function-space constraint. Disabled
                # by default so existing WSSR configurations are unchanged.
                "norm_constraint_mode": "euclidean",
                "function_norm_constraint": 0.001,
                "eta": 0.99,
                # Independent averaging weights. Negative values inherit the
                # legacy shared eta, preserving old configurations exactly.
                # With adaptive_S_average enabled, an unspecified eta_g uses
                # the current gradient (eta_g=0) instead.
                "eta_S": -1.0,
                "eta_g": -1.0,
                # Default-disabled alternative to augmented-factor averaging.
                # SSI determines U from the current batch only; history is
                # projected into U and averages either the diagonal spectrum
                # or near-degenerate reduced-metric blocks. The RHS remains
                # the current gradient (eta_g must be zero when enabled).
                "reduced_metric_history_mode": "none",
                "spectral_history_cluster_gap": 0.01,
                "spectral_history_noise_scale": 1.0,
                "spectral_history_drift_scale": 1.0,
                # Pure suggestion-1 ablation: retain legacy augmented-factor
                # matrix/subspace averaging but replace scalar eta_S by a
                # spectral-position weight on each stored history mode.
                "anisotropic_matrix_history": False,
                "anisotropic_matrix_history_noise_scale": 1.0,
                # Optional adaptive S-history schedule. New settings take
                # priority over eta/eta_S only when explicitly enabled.
                "adaptive_S_average": False,
                "eta_S_schedule": "constant",
                "eta_S_max": 0.95,
                "eta_S_warmup_steps": 1000,
                "eta_S_tau": 1000.0,
                # Optional independent schedule for gradient memory. Disabled
                # by default; eta_g remains constant unless explicitly enabled.
                "adaptive_g_average": False,
                "eta_g_schedule": "constant",
                "eta_g_max": 0.2,
                "eta_g_warmup_steps": 1000,
                "eta_g_tau": 1000.0,
                # Replace the legacy gradient EMA with a first-order transported
                # gradient memory. Disabled for exact legacy WSSR behavior.
                "enable_gradient_transport": False,
                # Optional startup ramp eta_t=min(eta, 1-1/t). This prevents
                # an empty history factor from receiving the full long-memory
                # weight in the first iterations. Disabled by default.
                "eta_bias_correction": False,
                # Keep the model/SVD in fp32, but add history/current rank
                # coefficients and apply the Tikhonov filter in fp64 via a
                # host callback. Disabled by default.
                "mixed_precision_solve": False,
                # Experimental full-parameter solution recurrence. ``naive``
                # adds the current independent solve to a decayed history;
                # ``residual`` first removes the history action visible to the
                # current batch, matching SPRING's projection correction.
                # Disabled by default.
                "solution_recurrence_mode": "none",
                "solution_recurrence_mu": 0.99,
                # Residual evaluation for solution recurrence. The historical
                # rank-coordinate approximation remains the default. The
                # Galerkin-consistent option evaluates the current batch in
                # sample space before solving inside the retained subspace.
                "residual_evaluation": "rank_coordinate",
                # Galerkin solve backend for full-current-batch residual
                # recurrence. ``host_fp64`` preserves the historical NumPy
                # callback; ``device_cholesky`` forms and solves the
                # regularized Gram system entirely on the JAX device.
                "galerkin_solve_backend": "host_fp64",
                "residual_dual_mode_diagnostics": False,
                "solution_error_feedback": False,
                "error_feedback_norm_cap": 10.0,
                # Error-feedback history is optionally decayed before it is
                # projected into the next Galerkin RHS. The historical
                # behavior is recovered with decay=1 and correction-relative
                # clipping.
                "error_feedback_decay": 1.0,
                "error_feedback_cap_reference": "correction",
                # Optional history is used only to choose the SSI subspace;
                # current-batch residual and Galerkin solve remain unchanged.
                "subspace_eta_S": 0.0,
                "recurrence_telemetry": False,
                # Refresh the SSI subspace every K updates. Intermediate
                # updates reuse the stored left subspace but recompute current
                # Ritz values, residuals, and Galerkin coefficients. K=1 is
                # the exact historical behavior.
                "subspace_refresh_period": 1,
                # ``ritz`` preserves the original delayed-refresh experiment.
                # ``fixed_basis`` keeps U exactly fixed between refreshes and
                # only recomputes the current W=O.T@U and Galerkin correction.
                # The latter is the genuinely lazy SSI fast path.
                "subspace_refresh_mode": "ritz",
                # Diagnostic-only hypothetical gradient-lag indicator.
                "drift_gate_monitoring": False,
                "drift_gate_hypothetical_eta_g": 0.8,
                "sr_rank": 10,
                "sr_rank_max": 100,
                "sr_storage_rank": -1,
                "sr_scale": 1.1,
                "svd_maxiter_initial": 8,
                "svd_maxiter_warm": 2,
                # Use one exact thin-SVD initialization, then the unchanged
                # right-warm truncated path. Disabled by default.
                "exact_first": False,
                "exact_first_force": False,
                # Expensive exact-reference metrics for short validation only.
                "exact_reference_diagnostics": False,
                # Lightweight update/complement metrics without an exact SVD.
                "update_diagnostics": False,
                # Diagnostic-only reliability indicators for the approximate
                # warm subspace and raw local-energy tails.  This recomputes a
                # one-fewer-SSI comparison but never changes the update used
                # by the optimizer.  Disabled by default.
                "reliability_diagnostics": False,
                # Emit large arrays only for host-side threshold-triggered burst
                # snapshots. Disabled by default.
                "burst_diagnostics_payload": False,
                "svd_working_rank": -1,
                "spectral_regularization": "hard_floor",
                "complement_weight": 1.0,
                "store_warm_u": True,
                # Keep O_current explicit but apply the history/current
                # augmented factor blockwise, avoiding a large concatenated
                # O_aug allocation. Disabled for backward compatibility.
                "semi_matrix_free_augmented": False,
                # Independent experimental ablations. ``none`` preserves the
                # historical warm-SVD-right update exactly.
                "experimental_mode": "none",
                "experimental_target_rank": -1,
                "cluster_gap_threshold": 0.002,
                # Default-disabled spectral-cluster envelope candidate.  It
                # projects the current gradient onto the span of the current
                # and recent top-k SSI bases without forming a parameter-space
                # projector. A negative gamma uses bar_lambda + lambda.
                "cluster_envelope_rank": 32,
                "cluster_envelope_history": 3,
                # A positive capacity enables the fixed-budget historical
                # selection mode. Current directions are always retained;
                # historical novelty competes only for the remaining slots.
                "cluster_envelope_capacity": -1,
                "cluster_envelope_decay": 0.8,
                "cluster_envelope_alpha": 0.2,
                "cluster_envelope_gamma": -1.0,
                "cluster_envelope_eigenvalue_cutoff": 1e-6,
                # ``scalar`` preserves the first prototype. ``rayleigh_ritz``
                # solves the current-batch projected Fisher equation on the
                # complete enclosing subspace instead of replacing its
                # anisotropic curvature by one mean eigenvalue.
                "cluster_envelope_curvature_mode": "scalar",
                # Default-disabled cross-half-batch empirical shrinkage for
                # the current-batch Ritz solve. Near-degenerate Ritz modes
                # share one SNR weight, avoiding basis-dependent filtering.
                "cluster_snr_gap_threshold": 0.05,
                "near_tail_modes": 0,
                "adaptive_complement_beta": 0.0,
                # Optional current-batch Fisher-metric cap on the same
                # complement. Zero preserves the legacy Euclidean-only cap.
                "adaptive_complement_beta_function": 0.0,
                # Optional continuation-local linear beta decay. A nonpositive
                # number of steps disables it exactly.
                "adaptive_complement_beta_final": 0.0,
                "adaptive_complement_decay_steps": 0,
                "adaptive_complement_decay_start": 0,
                # Experimental EMA of the already beta-capped complement.
                # A nonpositive decay disables the state exactly.
                "complement_state_decay": 0.0,
                "complement_state_relative_cap": 0.0,
                "multilevel_complement_period": 1,
                "multilevel_complement_cosine_threshold": 0.0,
                "smooth_transition_start": -1,
                "smooth_transition_end": -1,
                "force_aware_krylov_vectors": 0,
                "iterative_complement_iterations": 0,
                # Default-disabled temporal proximal regularization inside the
                # production warm2 SSI factors.  A positive value is gamma in
                # c_i=((s_i^2+lambda)c_i^native+gamma<u_i,d_prev>)/
                #     (s_i^2+lambda+gamma).
                "native_proximal_gamma": 0.0,
                # Optional Euclidean backstop applied after the selected
                # global norm constraint. A negative value disables it.
                "euclidean_safety_constraint": -1.0,
            },
            "wssr_warm_svd_right_matfree": {
                # Learning rate settings
                "schedule_type": "inverse_time",  # constant or inverse_time
                "learning_rate": 5e-2,
                "learning_decay_rate": 1e-4,
                # Experimental matrix-free right-subspace warm-start SVD hyperparams
                "damping": 0.001,
                "relative_singular_value_cutoff": -1.0,
                "tikhonov_lambda": -1.0,
                "constrain_norm": True,
                "norm_constraint": 0.001,
                "eta": 0.99,
                "sr_rank": 10,
                "sr_rank_max": 100,
                "sr_storage_rank": -1,
                "sr_scale": 1.1,
                "svd_maxiter_initial": 8,
                "svd_maxiter_warm": 2,
                "svd_working_rank": -1,
                "spectral_regularization": "hard_floor",
                "complement_weight": 1.0,
                "store_warm_u": True,
            },
            "avg_minsr_svd_history": {
                # Learning rate settings
                "schedule_type": "inverse_time",  # constant or inverse_time
                "learning_rate": 5e-2,
                "learning_decay_rate": 1e-4,
                # Averaged-MinSR Tikhonov dual-solve hyperparams
                "lambda_reg": 0.001,
                # Damping is used only for SVD-compressed history retention.
                "damping": 0.001,
                "constrain_norm": True,
                "norm_constraint": 0.001,
                "eta": 0.99,
                "sr_rank": 10,
                "sr_rank_max": 100,
                "sr_scale": 1.1,
            },
            "gauss_newton": {
                # Learning rate settings
                "schedule_type": "inverse_time",  # constant or inverse_time
                "learning_rate": 1.0,
                "learning_decay_rate": 1e-4,
                # GN hyperparams
                "E": 0.0,  # target energy
                "damping": 0.001,
                "constrain_norm": True,
                "norm_constraint": 0.001,
                "clip_threshold": 1000.0,  # GN works best with cusp Jastrow and no clipping
            },
        },
    }
    return vmc_config


def get_default_eval_config() -> Dict:
    """Get a default evaluation configuration."""
    eval_config = {
        "nchains": 1000,
        "nburn": 5000,
        "nepochs": 20000,
        "nsteps_per_param_update": 10,
        "nmoves_per_width_update": 100,
        "record_amplitudes": False,
        "std_move": 0.25,
        # if use_data_from_training=True, nchains, nmoves_per_width_update, and
        # std_move are completely ignored, and the data output from training is
        # used as the initial positions instead
        "use_data_from_training": False,
        "record_local_energies": True,  # save local energies and compute statistics
        "nan_safe": False,
    }
    return eval_config
