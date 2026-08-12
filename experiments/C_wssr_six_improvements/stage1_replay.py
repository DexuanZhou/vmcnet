#!/usr/bin/env python3
"""Frozen C snapshot screening for six independent WSSR improvements."""
import argparse,csv,json,subprocess,time
from pathlib import Path
import jax,jax.numpy as jnp
import numpy as np
from vmcnet.mcmc import position_amplitude_core as pacore
from vmcnet.physics import core as physics_core
from vmcnet.train import runners
from vmcnet.updates import wssr
from vmcnet.utils import io

SOURCES={
'initial':('/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1','1000.npz'),
'A500':('/scratch/dexuan1/runs/C_exact_first_complement/stage2_500_retry2/A_warm_c0','500.npz'),
'B500':('/scratch/dexuan1/runs/C_exact_first_complement/stage2_500_retry1/B_exact_c0','500.npz'),
'C500':('/scratch/dexuan1/runs/C_exact_first_complement/stage2_500_retry1/C_exact_c1e4','500.npz')}

def sync(x): return jax.block_until_ready(x)
def gpu_memory_mib():
 try:
  return float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()[0])
 except (OSError,subprocess.SubprocessError,ValueError,IndexError):
  return float('nan')
def metrics(x,ref):
 n=jnp.linalg.norm(x);r=jnp.linalg.norm(ref)
 return float(sync(jnp.linalg.norm(x-ref)/jnp.maximum(r,1e-30))),float(sync(jnp.vdot(x,ref)/jnp.maximum(n*r,1e-30)))
def constrain(x):
 d=-.02*x; scale=jnp.minimum(1.,jnp.sqrt(.001)/jnp.maximum(jnp.linalg.norm(d),1e-30));return d*scale,float(sync(scale))
def filter_update(u,s,force,lam,weights=None):
 if weights is None:weights=jnp.ones_like(s)
 return u@((weights/(s*s+lam))*(u.T@force))
def warm_filtered_update(u,s,force,weights=None):
 if weights is None:weights=jnp.ones_like(s)
 retained=(s/jnp.maximum(jnp.abs(s[0]),1e-30)>3e-4).astype(s.dtype)
 lam=(3e-4*s[0])**2
 return filter_update(u,s,force,lam,weights*retained)
def subspace(u,uref):
 q,_=jnp.linalg.qr(u,mode='reduced');r,_=jnp.linalg.qr(uref[:,:u.shape[1]],mode='reduced');z=jnp.linalg.svd(q.T@r,compute_uv=False);a=jnp.arccos(jnp.clip(z,-1,1));return float(sync(jnp.max(a))),float(sync(jnp.sqrt(jnp.mean(a*a)))),float(sync(jnp.sqrt(jnp.maximum(0.,2*u.shape[1]-2*jnp.sum(z*z)))))
def projected_cg(A,u,b,lam,niter):
 proj=lambda x:x-u@(u.T@x)
 mat=lambda x:proj(A@(A.T@proj(x))+lam*proj(x))
 x=jnp.zeros_like(b);r=proj(b);p=r;rr=jnp.vdot(r,r)
 for _ in range(niter):
  ap=mat(p);alpha=rr/jnp.maximum(jnp.vdot(p,ap),1e-30);x=proj(x+alpha*p);rn=proj(r-alpha*ap);rrn=jnp.vdot(rn,rn);p=proj(rn+(rrn/jnp.maximum(rr,1e-30))*p);r,rr=rn,rrn
 return x,float(sync(jnp.sqrt(rr)/jnp.maximum(jnp.linalg.norm(b),1e-30)))
def rr_force_basis(A,u,vh,force,krylov):
 v=vh.T;v1=A.T@force;v1=v1-v@(v.T@v1);n1=jnp.linalg.norm(v1);extra=[(v1/jnp.maximum(n1,1e-30))[:,None]]
 if krylov:
  v2=A.T@(A@extra[0][:,0]);q=jnp.concatenate([v]+extra,axis=1);v2=v2-q@(q.T@v2);n2=jnp.linalg.norm(v2);extra.append((v2/jnp.maximum(n2,1e-30))[:,None])
 q,_=jnp.linalg.qr(jnp.concatenate([v]+extra,axis=1),mode='reduced');y=A@q;g=y.T@y;ev,rot=jnp.linalg.eigh(g);idx=jnp.argsort(ev)[::-1][:400];vr=q@rot[:,idx];yy=A@vr;s=jnp.sqrt(jnp.maximum(ev[idx],0));uu=yy/jnp.maximum(s,1e-30);return uu,s,vr.T,float(sync(n1)),float(sync(n2 if krylov else jnp.nan)),extra

def replay_warm_svd(A,state,key,width):
 """Actual right-SSI, reproducibly augmenting a narrower saved warm state."""
 stored=state.sr_o.shape[1]
 if width<=stored:
  return wssr.right_warm_start_svd(A,state,key,8,2,svd_working_rank=width)
 has_warm=bool(sync(state.has_u|(state.sr_rank0>0)))
 if not has_warm:
  fresh=wssr.initialize_wssr_warm_svd_core_state(A.shape[0],width,width,dtype=A.dtype,store_warm_u=True)
  return wssr.right_warm_start_svd(A,fresh,key,8,2,svd_working_rank=width)
 base_width=min(stored,int(sync(state.sr_rank)),width)
 warm_u,_=wssr._select_warm_left_basis(state,base_width)
 projection=A.T@warm_u[:,:base_width]
 extra=jax.random.normal(jax.random.fold_in(key,width),(A.shape[1],width-base_width),dtype=A.dtype)
 v,_=jnp.linalg.qr(jnp.concatenate([projection,extra],axis=1),mode='reduced')
 for _ in range(2):
  v,_=jnp.linalg.qr(A.T@(A@v),mode='reduced')
 y=A@v;gram=y.T@y;ev,rot=jnp.linalg.eigh(gram);order=jnp.argsort(ev)[::-1];ev=ev[order];rot=rot[:,order]
 s=jnp.sqrt(jnp.maximum(ev,0));u=(y@rot)/jnp.maximum(s,1e-30);vh=(v@rot).T
 return u,s,vh,jnp.asarray(width,dtype=state.sr_rank.dtype)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('snapshot',choices=SOURCES);ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 src,ck=SOURCES[a.snapshot];src=Path(src);cfg=io.load_config_dict(str(src),'config.json');dtype=runners._get_dtype(cfg);ions,ch,ne=runners._get_electron_ion_config_as_arrays(cfg,dtype);epoch,data,params,opt,key=io.reload_vmc_state(str(src/'checkpoints'),ck);pos=pacore.get_position_from_data(data);assert pos.shape[0]==1000
 logpsi,_,_=runners._get_and_init_model(cfg.model,ions,ch,ne,pos,key,dtype=dtype,apply_pmap=False);local=runners._assemble_mol_local_energy_fn(ions,ch,cfg.problem.ei_softening,cfg.problem.ee_softening,logpsi);ef=physics_core.create_energy_and_statistics_fn(local,1000,runners._get_clipping_fn(cfg.vmc),cfg.vmc.nan_safe);en,le,_=ef(params,pos);oc,_=wssr.center_and_scale_score_matrix(logpsi,params,pos);ec=wssr.center_and_scale_energy_residuals(le,en)
 if a.snapshot=='initial':state=wssr.initialize_wssr_warm_svd_core_state(oc.shape[0],400,400,dtype=oc.dtype,store_warm_u=True)
 else:state=opt.core_state
 A,e=wssr.augment_wssr_system(oc,ec,state,.8);force=A@e;_,svd_key=jax.random.split(key);sync((A,e,force));t=time.perf_counter();U,S,Vh=jnp.linalg.svd(A,full_matrices=False);sync((U,S,Vh));exact_time=time.perf_counter()-t;lam=(3e-4*S[0])**2;full=filter_update(U,S,force,lam);rank400=filter_update(U[:,:400],S[:400],force,lam)
 # Actual baseline q=2, preserving initial-vs-warm branch semantics.
 t0_baseline=time.perf_counter();ub,sb,vhb,_=replay_warm_svd(A,state,svd_key,400);sync((ub,sb,vhb));lam_b=(3e-4*sb[0])**2;baseline=warm_filtered_update(ub,sb,force);sync(baseline)
 rows=[]
 def add(method,param,x,u=None,s=None,res=0.,rn=0.,tn=0.,cn=0.,eff=400,extra=None,t0=None):
  rel,cos=metrics(x,full);rr,rc=metrics(x,rank400);dx,scale=constrain(x);dr,dc=metrics(dx,constrain(full)[0]);lin=float(sync(jnp.linalg.norm(A@(A.T@x)+lam*x-force)/jnp.maximum(jnp.linalg.norm(force),1e-30)));ma=ra=pe=float('nan')
  if u is not None:ma,ra,pe=subspace(u,U)
  row=dict(snapshot=a.snapshot,method=method,parameter=str(param),full_relative_error=rel,full_cosine=cos,rank400_relative_error=rr,rank400_cosine=rc,constrained_relative_error=dr,constrained_cosine=dc,linear_residual=lin,resolved_norm=rn,near_tail_norm=tn,complement_norm=cn,complement_resolved_ratio=cn/max(rn,1e-30),constraint_scale=scale,effective_rank=eff,decomposition_width=(int(s.shape[0]) if s is not None else 0),runtime_seconds=(time.perf_counter()-t0 if t0 else 0),exact_svd_seconds=exact_time,max_principal_angle=ma,rms_principal_angle=ra,projector_error=pe,gpu_memory_used_mib=gpu_memory_mib(),finite=bool(sync(jnp.all(jnp.isfinite(x)))))
  if extra:row.update(extra)
  rows.append(row)
 add('baseline','warm2',baseline,ub,sb,rn=float(sync(jnp.linalg.norm(baseline))),t0=t0_baseline)
 # 1 cluster-aware: use the actual oversampled warm/SSI decomposition.
 gaps=jnp.abs(S[:-1]-S[1:])/jnp.maximum(jnp.abs(S[:-1]),1e-30)
 uc,sc,vhc,_=replay_warm_svd(A,state,svd_key,512);sync((uc,sc,vhc));gaps_c=jnp.abs(sc[:-1]-sc[1:])/jnp.maximum(jnp.abs(sc[:-1]),1e-30);lam_c=(3e-4*sc[0])**2
 for th in (1e-3,2e-3,5e-3):
  cand=np.where(np.asarray(gaps_c[399:511])>th)[0];r=400+int(cand[0]) if len(cand) else 512;t0=time.perf_counter();x=warm_filtered_update(uc[:,:r],sc[:r],force);sync(x);add('cluster',th,x,uc[:,:r],sc[:r],rn=float(sync(jnp.linalg.norm(x))),eff=r,t0=t0)
 # 2 explicit near tail.
 for p in (32,64,128):
  t0=time.perf_counter();un,sn,vhn,_=replay_warm_svd(A,state,svd_key,400+p);sync((un,sn,vhn));all_update=warm_filtered_update(un,sn,force);main=warm_filtered_update(un[:,:400],sn[:400],force);tail=all_update-main;x=all_update;sync(x);add('near_tail',p,x,un,sn,rn=float(sync(jnp.linalg.norm(main))),tn=float(sync(jnp.linalg.norm(tail))),eff=400+p,t0=t0)
 # 3 norm-adaptive scalar complement, nominal current weight=1e-4.
 resolved=warm_filtered_update(ub,sb,force);active_b=(sb/jnp.maximum(jnp.abs(sb[0]),1e-30)>3e-4).astype(sb.dtype);perp=force-ub@((ub.T@force)*active_b);alpha_nom=1e-4/lam_b
 fixed_comp=alpha_nom*perp
 add('fixed_complement_reference','1e-4',resolved+fixed_comp,ub,sb,rn=float(sync(jnp.linalg.norm(resolved))),cn=float(sync(jnp.linalg.norm(fixed_comp))),extra={'alpha_effective':float(sync(alpha_nom))})
 for beta in (.05,.1,.2):
  t0=time.perf_counter();alpha=jnp.minimum(alpha_nom,beta*jnp.linalg.norm(resolved)/jnp.maximum(jnp.linalg.norm(perp),1e-30));comp=alpha*perp;x=resolved+comp;sync(x);add('adaptive_complement',beta,x,ub,sb,rn=float(sync(jnp.linalg.norm(resolved))),cn=float(sync(jnp.linalg.norm(comp))),extra={'alpha_effective':float(sync(alpha))},t0=t0)
 # 4 continuous cosine tapers on the actual oversampled warm decomposition.
 for start,end in ((385,432),(385,464),(369,464)):
  t0=time.perf_counter();n=end;us,ss,vhs,_=replay_warm_svd(A,state,svd_key,n);sync((us,ss,vhs));idx=jnp.arange(n)+1.;w=jnp.where(idx<start,1.,jnp.where(idx>=end,0.,.5*(1+jnp.cos(jnp.pi*(idx-start)/(end-start)))));x=warm_filtered_update(us,ss,force,w);sync(x);add('smooth',f'{start}-{end}',x,us,ss,rn=float(sync(jnp.linalg.norm(x))),eff=n,extra={'filter_jump_max':float(sync(jnp.max(jnp.abs(jnp.diff(w)))))},t0=t0)
 # 5 force-aware RR from the actual approximate warm basis.
 for kry in (False,True):
  t0=time.perf_counter();uf,sf,vf,n1,n2,injected=rr_force_basis(A,ub,vhb,force,kry);x=warm_filtered_update(uf,sf,force);sync(x);ov1=metrics(A@injected[0][:,0],full)[1];ov2=metrics(A@injected[1][:,0],full)[1] if kry else float('nan');add('force_aware','force+krylov' if kry else 'force',x,uf,sf,rn=float(sync(jnp.linalg.norm(x))),t0=t0,extra={'force_projected_norm':n1,'krylov_projected_norm':n2,'force_vector_exact_update_cosine':ov1,'krylov_vector_exact_update_cosine':ov2})
 # 6 projected CG complement on top of the actual rank-400 warm update.
 active_u=ub*active_b[None,:];rhs=force-active_u@(active_u.T@force)
 for nit in (3,5,10):
  t0=time.perf_counter();comp,res=projected_cg(A,active_u,rhs,lam_b,nit);x=resolved+comp;sync(x);add('iterative_complement',nit,x,ub,sb,res=res,rn=float(sync(jnp.linalg.norm(resolved))),cn=float(sync(jnp.linalg.norm(comp))),t0=t0,extra={'projected_solver_residual':res})
 fields=sorted({k for r in rows for k in r});
 with (out/'results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 meta={'source':str(src/'checkpoints'/ck),'stored_epoch':int(epoch),'operator_shape':list(map(int,A.shape)),'nchains':1000,'fixed_operator':True,'mcmc_advanced':False,'update_applied':False,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'working_tree_dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),'oversampled_replay_initialization':'actual saved right-warm basis plus checkpoint-key random orthogonal completion; fresh random initial branch when no warm state','spectrum_390_520':[float(x) for x in np.asarray(S[389:520])],'gaps_390_520':[float(x) for x in np.asarray(gaps[389:520])]};(out/'metadata.json').write_text(json.dumps(meta,indent=2));print('wrote',out)
if __name__=='__main__':main()
