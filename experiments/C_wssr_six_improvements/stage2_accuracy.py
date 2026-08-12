#!/usr/bin/env python3
"""Frozen exact-full-reference replay at a Stage-2 checkpoint."""
import argparse,csv,json,time
from pathlib import Path
import jax,jax.numpy as jnp
from vmcnet.mcmc import position_amplitude_core as pacore
from vmcnet.physics import core as physics_core
from vmcnet.train import runners
from vmcnet.updates import wssr
from vmcnet.utils import io
def sync(x):return jax.block_until_ready(x)
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--epoch',type=int,required=True);p.add_argument('--out',required=True);a=p.parse_args();run=Path(a.run);cfg=io.load_config_dict(str(run),'config.json');ck=f'{a.epoch}.npz';ep,data,params,opt,key=io.reload_vmc_state(str(run/'checkpoints'),ck);dtype=runners._get_dtype(cfg);ions,ch,ne=runners._get_electron_ion_config_as_arrays(cfg,dtype);pos=pacore.get_position_from_data(data);logpsi,_,_=runners._get_and_init_model(cfg.model,ions,ch,ne,pos,key,dtype=dtype,apply_pmap=False);local=runners._assemble_mol_local_energy_fn(ions,ch,cfg.problem.ei_softening,cfg.problem.ee_softening,logpsi);ef=physics_core.create_energy_and_statistics_fn(local,1000,runners._get_clipping_fn(cfg.vmc),cfg.vmc.nan_safe);en,le,_=ef(params,pos);oc,_=wssr.center_and_scale_score_matrix(logpsi,params,pos);ec=wssr.center_and_scale_energy_residuals(le,en);A,e=wssr.augment_wssr_system(oc,ec,opt.core_state,.8);o=cfg.vmc.optimizer.wssr_warm_svd_right
 kw=dict(damping=o.damping,norm_constraint=o.norm_constraint,sr_rank_max=o.sr_rank_max,sr_scale=o.sr_scale,svd_maxiter_initial=o.svd_maxiter_initial,svd_maxiter_warm=o.svd_maxiter_warm,exact_first=False,svd_working_rank=o.svd_working_rank,constrain_update_norm=False,spectral_regularization=o.spectral_regularization,complement_weight=o.complement_weight,experimental_mode=o.experimental_mode,experimental_target_rank=o.experimental_target_rank,cluster_gap_threshold=o.cluster_gap_threshold,near_tail_modes=o.near_tail_modes,adaptive_complement_beta=o.adaptive_complement_beta,smooth_transition_start=o.smooth_transition_start,smooth_transition_end=o.smooth_transition_end,force_aware_krylov_vectors=o.force_aware_krylov_vectors,iterative_complement_iterations=o.iterative_complement_iterations)
 t=time.perf_counter();got=wssr.wssr_warm_svd_right_core_update(A,e,opt.core_state,key,**kw).grad_like_update;sync(got);approx_time=time.perf_counter()-t;t=time.perf_counter();U,S,Vh=jnp.linalg.svd(A,full_matrices=False);sync((U,S,Vh));exact_time=time.perf_counter()-t;force=A@e;lam=(o.damping*S[0])**2;full=U@((U.T@force)/(S*S+lam));u400=U[:,:400];s400=S[:400];rank=u400@((u400.T@force)/(s400*s400+lam));ng=jnp.linalg.norm(got);nf=jnp.linalg.norm(full);nr=jnp.linalg.norm(rank)
 row=dict(run=run.name,epoch=a.epoch,stored_epoch=int(ep),full_relative_error=float(sync(jnp.linalg.norm(got-full)/nf)),full_cosine=float(sync(jnp.vdot(got,full)/(ng*nf))),rank400_relative_error=float(sync(jnp.linalg.norm(got-rank)/nr)),rank400_cosine=float(sync(jnp.vdot(got,rank)/(ng*nr))),linear_residual=float(sync(jnp.linalg.norm(A@(A.T@got)+lam*got-force)/jnp.linalg.norm(force))),approx_seconds=approx_time,exact_seconds=exact_time,finite=bool(sync(jnp.all(jnp.isfinite(got)))))
 out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(row,indent=2));print(json.dumps(row,indent=2))
if __name__=='__main__':main()
