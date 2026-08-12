#!/usr/bin/env python3
"""Frozen, no-update reconstruction of the first C WSSR augmented operator."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from vmcnet.mcmc import position_amplitude_core as pacore
from vmcnet.train import runners
from vmcnet.updates import wssr
from vmcnet.utils import io

SOURCE = Path('/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1')
OUT = Path('/scratch/dexuan1/vmcnet/experiments/C_exact_first_complement/results/direct_subspace_replay.json')

config = io.load_config_dict(str(SOURCE), 'config.json')
dtype = runners._get_dtype(config)
ion_pos, ion_charges, nelec = runners._get_electron_ion_config_as_arrays(config, dtype)
_, data, params, _, key = io.reload_vmc_state(str(SOURCE / 'checkpoints'), '1000.npz')
positions = pacore.get_position_from_data(data)
log_psi_apply, _, _ = runners._get_and_init_model(
    config.model, ion_pos, ion_charges, nelec, positions, key,
    dtype=dtype, apply_pmap=False,
)
o_cur, _ = wssr.center_and_scale_score_matrix(log_psi_apply, params, positions)
state = wssr.initialize_wssr_warm_svd_core_state(
    o_cur.shape[0], 400, 400, dtype=o_cur.dtype, store_warm_u=True
)
e_cur = jnp.zeros((positions.shape[0],), dtype=o_cur.dtype)
o_aug, _ = wssr.augment_wssr_system(o_cur, e_cur, state, 0.8)

# The first call is the exact decomposition used by the implementation; its
# first 400 columns are what the normal warm state stores. The second is an
# independent reference in precisely the same parameter-space operator.
u_actual_all, s_actual_all, _ = jnp.linalg.svd(o_aug, full_matrices=False)
u_ref_all, s_ref_all, _ = jnp.linalg.svd(o_aug, full_matrices=False)
u_actual = u_actual_all[:, :405]
s_actual = s_actual_all[:405]
u_ref = u_ref_all[:, :405]
s_ref = s_ref_all[:405]

def compare(k):
    qa, _ = jnp.linalg.qr(u_actual[:, :k], mode='reduced')
    qr, _ = jnp.linalg.qr(u_ref[:, :k], mode='reduced')
    overlap = jnp.linalg.svd(qa.T @ qr, compute_uv=False)
    angles = jnp.arccos(jnp.clip(overlap, -1.0, 1.0))
    projector_error = jnp.sqrt(
        jnp.maximum(0.0, 2.0 * k - 2.0 * jnp.sum(jnp.square(overlap)))
    )
    return {
        'rank': k,
        'min_overlap_sv': float(jnp.min(overlap)),
        'max_principal_angle_rad': float(jnp.max(angles)),
        'projector_frobenius_error': float(projector_error),
        'projector_relative_error': float(projector_error / jnp.sqrt(k)),
    }

s = np.asarray(s_ref_all)
around = []
for one_based in range(395, 406):
    i = one_based - 1
    gap = abs(s[i] - s[i + 1]) / max(abs(s[i]), 1e-30)
    around.append({'index_1based': one_based, 'singular_value': float(s[i]), 'relative_gap_to_next': float(gap)})

result = {
    'source_checkpoint': str(SOURCE / 'checkpoints/1000.npz'),
    'nchains': int(positions.shape[0]),
    'operator_shape': list(map(int, o_aug.shape)),
    'spaces_compared': 'orthonormalized parameter-space left singular subspaces',
    'rank_comparisons': [compare(k) for k in (394, 400, 405)],
    'singular_values_395_405': around,
    'rank400_singular_relative_error': float(abs(s_actual[399] - s_ref[399]) / abs(s_ref[399])),
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
