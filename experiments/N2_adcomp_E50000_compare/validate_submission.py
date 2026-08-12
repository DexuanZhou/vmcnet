#!/usr/bin/env python3
import copy,json
from pathlib import Path
P=Path('/scratch/dexuan1/vmcnet/experiments/N2_adcomp_E50000_compare')
SRC=Path('/scratch/dexuan1/runs/fir_n4096_corrected_retry1/N2/kfac_preliminary')
v=json.loads(Path('/scratch/dexuan1/vmcnet/experiments/N2_adaptive_complement/results/checkpoint_validation.json').read_text())
assert v['internal_epoch']==4999 and v['nchains']==4096 and v['unique_walkers']==4096
assert abs(v['bond_length_bohr']-2.068)<1e-9 and v['nelec']==[7,7]
base=json.loads((SRC/'config.json').read_text())
common=dict(optimizer_type='wssr_warm_svd_right',nchains=4096,nepochs=50000,nburn=5000,
 eta=.2,learning_rate=.002,schedule_type='inverse_time',learning_decay_rate=.0001,
 damping=.0003,spectral_regularization='tikhonov',norm_constraint=.001,
 svd_maxiter_initial=8,svd_maxiter_warm=2,exact_first=False,store_warm_u=False,
 complement_weight=0.,experimental_mode='none',adaptive_complement_beta=0.)
def resolved(rank,mode,beta,comp):
 d=copy.deepcopy(common); d.update(rank=rank,sr_rank=rank,sr_rank_max=rank,
  sr_storage_rank=rank,svd_working_rank=rank,experimental_target_rank=rank,
  experimental_mode=mode,adaptive_complement_beta=beta,complement_weight=comp,
  source_checkpoint=str(SRC/'checkpoints/5000.npz'),geometry_R=2.068,nelec=[7,7])
 return d
a=resolved(800,'none',0.,0.); b=resolved(400,'adaptive_complement',.2,.0001)
diff={k:{'A':a[k],'B':b[k]} for k in sorted(a) if a[k]!=b[k]}
allowed={'rank','sr_rank','sr_rank_max','sr_storage_rank','svd_working_rank','experimental_target_rank','experimental_mode','adaptive_complement_beta','complement_weight'}
assert set(diff)==allowed, (set(diff),allowed)
(P/'resolved_A.json').write_text(json.dumps(a,indent=2)+'\n')
(P/'resolved_B.json').write_text(json.dumps(b,indent=2)+'\n')
(P/'config_diff.json').write_text(json.dumps(diff,indent=2)+'\n')
print(json.dumps({'checkpoint_validation':'PASS','config_diff':diff},indent=2))
