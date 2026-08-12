#!/usr/bin/env python3
import json
from pathlib import Path
import jax,numpy as np
from vmcnet.utils import io
from vmcnet.mcmc import position_amplitude_core as pacore
SRC=Path('/scratch/dexuan1/runs/N2_adcomp_E50000_compare/B_rank400_beta02')
CK=SRC/'checkpoints/30000.npz'; OUT=Path(__file__).resolve().parent/'results'
e,d,p,o,k=io.reload_vmc_state(str(CK.parent),CK.name)
pos=np.asarray(pacore.get_position_from_data(d)); amp=np.asarray(pacore.get_amplitude_from_data(d))
leaves=lambda x:jax.tree_util.tree_leaves(x)
finite=lambda x:all(np.isfinite(np.asarray(y)).all() for y in leaves(x))
cfg=json.loads((SRC/'config.json').read_text())
core=o.core_state
r=dict(status='PASS',checkpoint=str(CK),internal_epoch=int(e),nchains=int(pos.shape[0]),
 unique_walkers=int(np.unique(pos.reshape(pos.shape[0],-1),axis=0).shape[0]),amplitude_shape=list(amp.shape),
 prng_shape=list(np.asarray(k).shape),parameters_finite=finite(p),optimizer_finite=finite(o),walkers_finite=finite(d),
 warm_sr_o_shape=list(core.sr_o.shape),warm_ek_shape=list(core.ek.shape),warm_rank=int(core.sr_rank),
 warm_rank0=int(core.sr_rank0),warm_has_u=bool(core.has_u),bond_length=abs(cfg['problem']['ion_pos'][1][2]-cfg['problem']['ion_pos'][0][2]),nelec=cfg['problem']['nelec'])
assert r['internal_epoch']==29999 and r['nchains']==4096 and r['unique_walkers']==4096
assert abs(r['bond_length']-2.068)<1e-9 and r['nelec']==[7,7]
assert r['parameters_finite'] and r['optimizer_finite'] and r['walkers_finite'] and r['warm_rank']>=400
OUT.mkdir(parents=True,exist_ok=True);(OUT/'checkpoint_validation.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
