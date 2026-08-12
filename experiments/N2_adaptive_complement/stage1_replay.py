#!/usr/bin/env python3
"""Fixed N2 snapshot replay for hard truncation and adaptive complement."""
import csv, json, subprocess, time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from vmcnet.mcmc import position_amplitude_core as pacore
from vmcnet.physics import core as physics_core
from vmcnet.train import runners
from vmcnet.updates import wssr
from vmcnet.utils import io

SOURCE = Path("/scratch/dexuan1/runs/fir_n4096_corrected_retry1/N2/kfac_preliminary")
OUT = Path(__file__).resolve().parent / "results/stage1"
DAMPING, LR, NORM = 3e-4, 2e-3, 1e-3

def sync(x): return jax.block_until_ready(x)
def relcos(x, y):
    nx, ny = jnp.linalg.norm(x), jnp.linalg.norm(y)
    return (float(sync(jnp.linalg.norm(x-y)/jnp.maximum(ny,1e-30))),
            float(sync(jnp.vdot(x,y)/jnp.maximum(nx*ny,1e-30))))
def constrained(x):
    d = -LR*x
    scale = jnp.minimum(1., jnp.sqrt(NORM)/jnp.maximum(jnp.linalg.norm(d),1e-30))
    return d*scale, float(sync(scale))
def gpu_mem():
    try: return float(subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True).splitlines()[0])
    except Exception: return float("nan")

cfg=io.load_config_dict(str(SOURCE),"config.json")
dtype=runners._get_dtype(cfg)
ions,charges,nelec=runners._get_electron_ion_config_as_arrays(cfg,dtype)
epoch,data,params,_,key=io.reload_vmc_state(str(SOURCE/"checkpoints"),"5000.npz")
positions=pacore.get_position_from_data(data)
assert positions.shape[0] == 4096
logpsi,_,_=runners._get_and_init_model(cfg.model,ions,charges,nelec,positions,key,dtype=dtype,apply_pmap=False)
local=runners._assemble_mol_local_energy_fn(ions,charges,cfg.problem.ei_softening,cfg.problem.ee_softening,logpsi)
efn=physics_core.create_energy_and_statistics_fn(local,4096,runners._get_clipping_fn(cfg.vmc),cfg.vmc.nan_safe)
@jax.jit
def build_fixed_operator_inputs(p, x):
    """Fuse score construction, centering and scaling to avoid eager copies."""
    en, le, _ = efn(p, x)
    o, _ = wssr.center_and_scale_score_matrix(logpsi, p, x)
    er = wssr.center_and_scale_energy_residuals(le, en)
    return en, o, er

energy,o_cur,e_cur=build_fixed_operator_inputs(params,positions)
# One common empty history allocation makes the fixed operator identical for A-D.
state=wssr.initialize_wssr_warm_svd_core_state(o_cur.shape[0],800,800,dtype=o_cur.dtype,store_warm_u=False)
A,e=wssr.augment_wssr_system(o_cur,e_cur,state,.2)
force=A@e
_,svd_key=jax.random.split(key)
sync((A,e,force))

# Exact full regularized action through the sample-space Gram matrix; this avoids
# materializing the full left-singular-vector matrix.
t=time.perf_counter()
gram=A.T@A
evals,V=jnp.linalg.eigh(gram)
order=jnp.argsort(evals)[::-1]
evals=jnp.maximum(evals[order],0.); V=V[:,order]
s=jnp.sqrt(evals); lam=(DAMPING*s[0])**2
full=A@(V@((V.T@e)/(evals+lam)))
sync(full); exact_seconds=time.perf_counter()-t

rows=[]
for label,rank,beta in (("A_rank400_hard",400,0.),("B_rank400_beta01",400,.1),
                        ("C_rank400_beta02",400,.2),("D_rank800_hard",800,0.)):
    t=time.perf_counter()
    u,sv,vh,_=wssr.right_warm_start_svd(A,state,svd_key,8,2,svd_working_rank=rank)
    sync((u,sv,vh))
    leading=jnp.maximum(jnp.abs(sv[0]),1e-12)
    retained=(sv/leading>DAMPING).astype(A.dtype)
    lam_r=(DAMPING*leading)**2
    projected=u.T@force
    resolved=u@(projected*retained/(sv*sv+lam_r))
    perp_force=force-u@(projected*retained)
    alpha_nominal=jnp.asarray(0.,A.dtype)
    comp=jnp.zeros_like(resolved); cap=False
    if beta:
        # Matches the tested adaptive path: complement_weight supplies only the
        # nominal ceiling; the fixed scalar-complement branch is not used.
        _,alpha_nominal=wssr._wssr_inverse_spectral_coefficients(
            jnp.where(retained.astype(bool),sv,1.),leading,jnp.asarray(DAMPING,A.dtype),"tikhonov",1e-4)
        alpha_cap=beta*jnp.linalg.norm(resolved)/jnp.maximum(jnp.linalg.norm(perp_force),1e-30)
        alpha=jnp.minimum(alpha_nominal,alpha_cap); comp=alpha*perp_force
        cap=bool(sync(alpha < alpha_nominal))
    else: alpha=jnp.asarray(0.,A.dtype)
    update=resolved+comp; sync(update); seconds=time.perf_counter()-t
    re,co=relcos(update,full); du,scale=constrained(update); df,_=constrained(full); dre,dco=relcos(du,df)
    residual=float(sync(jnp.linalg.norm(A@(A.T@update)+lam*update-force)/jnp.maximum(jnp.linalg.norm(force),1e-30)))
    rn=float(sync(jnp.linalg.norm(resolved))); cn=float(sync(jnp.linalg.norm(comp)))
    rows.append(dict(variant=label,rank=rank,requested_beta=beta,update_relative_error=re,
      update_cosine=co,constrained_relative_error=dre,constrained_cosine=dco,
      resolved_norm=rn,complement_norm=cn,complement_resolved_ratio=cn/max(rn,1e-30),
      unclamped_candidate_complement_norm=float(sync(alpha_nominal*jnp.linalg.norm(perp_force))),
      adaptive_scaling_factor=float(sync(alpha)),beta_cap_active=cap,regularized_residual=residual,
      norm_constraint_scale=scale,runtime_seconds=seconds,exact_full_seconds=exact_seconds,
      peak_gpu_memory_mib=gpu_mem(),finite=bool(sync(jnp.all(jnp.isfinite(update))))))
OUT.mkdir(parents=True,exist_ok=True)
with (OUT/"replay.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/"metadata.json").write_text(json.dumps({"checkpoint":str(SOURCE/"checkpoints/5000.npz"),
 "internal_epoch":int(epoch),"nchains":4096,"operator_shape":list(map(int,A.shape)),
 "fixed_operator":True,"mcmc_advanced":False,"update_applied":False,"energy":float(energy),
 "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()},indent=2)+"\n")
print(json.dumps(rows,indent=2))
