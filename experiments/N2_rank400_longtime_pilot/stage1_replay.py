#!/usr/bin/env python3
import csv,json,subprocess,time
from pathlib import Path
import jax,jax.numpy as jnp,numpy as np
from vmcnet.mcmc import position_amplitude_core as pacore
from vmcnet.physics import core as physics_core
from vmcnet.train import runners
from vmcnet.updates import wssr,wssr_experimental as wx
from vmcnet.utils import io
SRC=Path('/scratch/dexuan1/runs/N2_adcomp_E50000_compare/B_rank400_beta02');OUT=Path(__file__).resolve().parent/'results/stage1';D=.0003;LR=.002;NC=.001
sync=jax.block_until_ready
def rc(x,y):
 return float(sync(jnp.linalg.norm(x-y)/jnp.maximum(jnp.linalg.norm(y),1e-30))),float(sync(jnp.vdot(x,y)/jnp.maximum(jnp.linalg.norm(x)*jnp.linalg.norm(y),1e-30)))
def residual(A,f,x,lam):return jnp.linalg.norm(f-(A@(A.T@x)+lam*x))/jnp.maximum(jnp.linalg.norm(f),1e-30)
def mem():
 try:return float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()[0])
 except:return float('nan')
cfg=io.load_config_dict(str(SRC),'config.json');dtype=runners._get_dtype(cfg);ions,ch,ne=runners._get_electron_ion_config_as_arrays(cfg,dtype);epoch,data,params,opt,key=io.reload_vmc_state(str(SRC/'checkpoints'),'30000.npz');pos=pacore.get_position_from_data(data)
logpsi,_,_=runners._get_and_init_model(cfg.model,ions,ch,ne,pos,key,dtype=dtype,apply_pmap=False);local=runners._assemble_mol_local_energy_fn(ions,ch,cfg.problem.ei_softening,cfg.problem.ee_softening,logpsi);ef=physics_core.create_energy_and_statistics_fn(local,4096,runners._get_clipping_fn(cfg.vmc),cfg.vmc.nan_safe)
@jax.jit
def fixed(p,x):
 en,le,_=ef(p,x);o,_=wssr.center_and_scale_score_matrix(logpsi,p,x);e=wssr.center_and_scale_energy_residuals(le,en);return en,o,e
en,oc,ec=fixed(params,pos);A,e=wssr.augment_wssr_system(oc,ec,opt.core_state,.2);f=A@e;_,sk=jax.random.split(key);sync((A,e,f))
def memory_safe_truncated_svd(matrix, rank, key, oversampling=32, n_iter=6):
 """Deterministic randomized thin SVD without materializing A.T @ A."""
 width=min(rank+oversampling,min(matrix.shape))
 omega=jax.random.normal(key,(matrix.shape[1],width),dtype=matrix.dtype)
 q,_=jnp.linalg.qr(matrix@omega,mode='reduced')
 for _ in range(n_iter):
  z,_=jnp.linalg.qr(matrix.T@q,mode='reduced')
  q,_=jnp.linalg.qr(matrix@z,mode='reduced')
 small=q.T@matrix
 ub,s,vh=jnp.linalg.svd(small,full_matrices=False)
 return q@ub[:,:rank],s[:rank],vh[:rank]
t=time.perf_counter();ref_key=jax.random.fold_in(sk,800);U,s,Vh=memory_safe_truncated_svd(A,800,ref_key);lam=(D*s[0])**2;ref800=U@((U.T@f)/(s**2+lam));sync(ref800);ref_time=time.perf_counter()-t
u,sv,vh,_=wssr.right_warm_start_svd(A,opt.core_state,sk,8,2,svd_working_rank=400);sync((u,sv,vh));ret=(sv/sv[0]>D);resolved=u@((u.T@f)*ret/(sv*sv+lam));perp=f-u@((u.T@f)*ret)
_,nom=wssr._wssr_inverse_spectral_coefficients(jnp.where(ret,sv,1.),sv[0],jnp.asarray(D,sv.dtype),'tikhonov',1e-4)
rows=[]
def add(name,x,res,comp,beta,seconds,extra={}):
 rr,co=rc(x,ref800);rn=float(sync(jnp.linalg.norm(res)));cn=float(sync(jnp.linalg.norm(comp)));scale=float(sync(jnp.minimum(1.,jnp.sqrt(NC)/jnp.maximum(LR*jnp.linalg.norm(x),1e-30))))
 rows.append(dict(variant=name,update_norm=float(sync(jnp.linalg.norm(x))),rank800_relative_error=rr,rank800_cosine=co,regularized_residual=float(sync(residual(A,f,x,lam))),residual_reduction_vs_A=0.,resolved_norm=rn,complement_or_near_norm=cn,realized_ratio=cn/max(rn,1e-30),norm_constraint_scale=scale,runtime_seconds=seconds,peak_gpu_memory_mib=mem(),finite=bool(sync(jnp.all(jnp.isfinite(x)))),requested_beta=beta,**extra))
for name,beta in [('A_beta02_control',.2),('B_beta01_fixed',.1),('C_beta_decay',.2)]:
 t=time.perf_counter();c,a=wx.adaptive_complement(resolved,perp,nom,beta);x=resolved+c;sync(x);add(name,x,resolved,c,beta,time.perf_counter()-t,{'effective_alpha':float(sync(a)),'cap_active':bool(sync(a<nom))})
t=time.perf_counter();c,a,astar,acap,rb,ra=wx.residual_optimal_complement(A,f,resolved,perp,lam,.2);x=resolved+c;sync(x);add('D_residual_optimal',x,resolved,c,.2,time.perf_counter()-t,{'effective_alpha':float(sync(a)),'alpha_star':float(sync(astar)),'alpha_cap':float(sync(acap)),'cap_active':bool(sync(a<astar)),'residual_before_abs':float(sync(jnp.linalg.norm(rb))),'residual_after_abs':float(sync(jnp.linalg.norm(ra)))})
main=U[:,:400]@((U[:,:400].T@f)/(s[:400]**2+lam));near=U[:,400:600]@((U[:,400:600].T@f)/(s[400:600]**2+lam));nearcap,sc=wx.cap_contribution(main,near,.2);x=main+nearcap;add('E_capped_near_tail',x,main,nearcap,.2,ref_time,{'unscaled_near_norm':float(sync(jnp.linalg.norm(near))),'near_scale':float(sync(sc)),'cap_active':bool(sync(sc<1)),'deterministic_initialization':'exact fixed-operator rank600'})
x=U[:,:600]@((U[:,:600].T@f)/(s[:600]**2+lam));add('F_rank600_hard',x,x,jnp.zeros_like(x),0.,ref_time,{'deterministic_initialization':'exact fixed-operator rank600'})
base=rows[0]['regularized_residual']
for r in rows:r['residual_reduction_vs_A']=(base-r['regularized_residual'])/base
OUT.mkdir(parents=True,exist_ok=True);fields=sorted({k for r in rows for k in r});
with (OUT/'replay.csv').open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
(OUT/'metadata.json').write_text(json.dumps({'checkpoint':str(SRC/'checkpoints/30000.npz'),'internal_epoch':int(epoch),'nchains':4096,'fixed_operator':True,'mcmc_advanced':False,'update_applied':False,'operator_shape':list(map(int,A.shape)),'rank800_reference':'deterministic oversampled randomized thin SVD (oversampling=32, SSI=6); no A.T@A materialization','rank600_initialization':'first 600 modes of the same deterministic fixed-operator thin SVD; no random uncontrolled completion'},indent=2)+'\n');print(json.dumps(rows,indent=2))
