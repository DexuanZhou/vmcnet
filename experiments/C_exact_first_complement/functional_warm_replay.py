#!/usr/bin/env python3
"""No-update replay of one warm step from baseline and exact-first states."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from vmcnet.mcmc import position_amplitude_core as pacore
from vmcnet.physics import core as physics_core
from vmcnet.train import runners
from vmcnet.updates import wssr
from vmcnet.utils import io

SOURCE = Path('/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1')
OUT = Path('/scratch/dexuan1/vmcnet/experiments/C_exact_first_complement/results/functional_warm_replay.json')
config = io.load_config_dict(str(SOURCE), 'config.json')
dtype = runners._get_dtype(config)
ion_pos, ion_charges, nelec = runners._get_electron_ion_config_as_arrays(config, dtype)
_, data, params, _, key = io.reload_vmc_state(str(SOURCE / 'checkpoints'), '1000.npz')
positions = pacore.get_position_from_data(data)
log_psi_apply, _, _ = runners._get_and_init_model(
    config.model, ion_pos, ion_charges, nelec, positions, key,
    dtype=dtype, apply_pmap=False,
)
local_energy_fn = runners._assemble_mol_local_energy_fn(
    ion_pos, ion_charges, config.problem.ei_softening,
    config.problem.ee_softening, log_psi_apply,
)
energy_fn = physics_core.create_energy_and_statistics_fn(
    local_energy_fn, 1000, runners._get_clipping_fn(config.vmc), config.vmc.nan_safe
)
energy, local_energies, _ = energy_fn(params, positions)
o_cur, _ = wssr.center_and_scale_score_matrix(log_psi_apply, params, positions)
e_cur = wssr.center_and_scale_energy_residuals(local_energies, energy)
initial = wssr.initialize_wssr_warm_svd_core_state(
    o_cur.shape[0], 400, 400, dtype=o_cur.dtype, store_warm_u=True
)
o_aug, e_aug = wssr.augment_wssr_system(o_cur, e_cur, initial, 0.8)
_, svd_key = jax.random.split(key)
common = dict(
    damping=3e-4, norm_constraint=1e-3, sr_rank_max=400,
    sr_scale=1.1, svd_maxiter_initial=8, svd_maxiter_warm=1,
    svd_working_rank=400, constrain_update_norm=False,
    spectral_regularization='tikhonov', complement_weight=0.0,
)
baseline_first = wssr.wssr_warm_svd_right_core_update(
    o_aug, e_aug, initial, svd_key, exact_first=False, **common
)
exact_first = wssr.wssr_warm_svd_right_core_update(
    o_aug, e_aug, initial, svd_key, exact_first=True, **common
)
u_ref_all, s_ref_all, vh_ref_all = jnp.linalg.svd(o_aug, full_matrices=False)
u_ref, s_ref, vh_ref = u_ref_all[:, :400], s_ref_all[:400], vh_ref_all[:400, :]

def replay(label, first_state):
    u, s, vh, rank = wssr.right_warm_start_svd(
        o_aug, first_state, svd_key, 8, 1, svd_working_rank=400
    )
    got = wssr._wssr_update_from_svd(
        o_aug, e_aug, first_state, u, s, vh, 3e-4, 1e-3, 400,
        1.1, False, rank_update_max=400,
        spectral_regularization='tikhonov', complement_weight=0.0,
    ).grad_like_update
    ref = wssr._wssr_update_from_svd(
        o_aug, e_aug, first_state, u_ref, s_ref, vh_ref, 3e-4, 1e-3,
        400, 1.1, False, rank_update_max=400,
        spectral_regularization='tikhonov', complement_weight=0.0,
    ).grad_like_update
    ng, nr = jnp.linalg.norm(got), jnp.linalg.norm(ref)
    qa, _ = jnp.linalg.qr(u, mode='reduced'); qr, _ = jnp.linalg.qr(u_ref, mode='reduced')
    overlap = jnp.linalg.svd(qa.T @ qr, compute_uv=False)
    angles = jnp.arccos(jnp.clip(overlap, -1.0, 1.0))
    residual = jnp.linalg.norm(qr - qa @ (qa.T @ qr))
    leaves = jax.tree_util.tree_leaves(first_state)
    finite_state = all(bool(jnp.all(jnp.isfinite(x))) for x in leaves)
    return {
        'label': label,
        'active_rank': int(rank),
        'update_relative_error': float(jnp.linalg.norm(got-ref)/jnp.maximum(nr,1e-12)),
        'update_cosine': float(jnp.vdot(got,ref)/jnp.maximum(ng*nr,1e-12)),
        'filtered_force_relative_error': float(jnp.linalg.norm(got-ref)/jnp.maximum(nr,1e-12)),
        'subspace_residual': float(residual),
        'maximum_principal_angle_rad': float(jnp.max(angles)),
        'finite_update': bool(jnp.all(jnp.isfinite(got))),
        'finite_state': finite_state,
        'singular_relative_error_at_400': float(jnp.abs(s[399]-s_ref[399])/jnp.abs(s_ref[399])),
    }

spectrum=[]
s_np=np.asarray(s_ref_all)
for index in range(394,406):
    i=index-1
    spectrum.append({'index_1based':index,'singular_value':float(s_np[i]),'relative_gap_to_next':float(abs(s_np[i]-s_np[i+1])/abs(s_np[i]))})
result={
    'source_checkpoint':str(SOURCE/'checkpoints/1000.npz'),
    'operator_shape':list(map(int,o_aug.shape)),
    'parameters_walkers_prng_unchanged':True,
    'baseline_A':replay('baseline_A',baseline_first.state),
    'exact_first':replay('exact_first',exact_first.state),
    'singular_values_394_405':spectrum,
}
OUT.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
