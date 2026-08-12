#!/usr/bin/env python3
"""Fixed-snapshot, no-MCMC, no-update SSI convergence replay."""
import argparse, csv, json, time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from vmcnet.mcmc import position_amplitude_core as pacore
from vmcnet.physics import core as physics_core
from vmcnet.train import runners
from vmcnet.updates import wssr
from vmcnet.utils import io

SOURCES = {
    "initial": (Path("/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1"), "1000.npz"),
    "A500": (Path("/scratch/dexuan1/runs/C_exact_first_complement/stage2_500_retry2/A_warm_c0"), "500.npz"),
    "B500": (Path("/scratch/dexuan1/runs/C_exact_first_complement/stage2_500_retry1/B_exact_c0"), "500.npz"),
    "C500": (Path("/scratch/dexuan1/runs/C_exact_first_complement/stage2_500_retry1/C_exact_c1e4"), "500.npz"),
}
BASE_Q = [0, 1, 2, 4, 8, 16, 32]
FIELDS = [
    "snapshot", "initialization", "q", "update_relative_error", "update_cosine",
    "constrained_relative_error", "constrained_cosine", "subspace_residual",
    "projector_frobenius_error", "max_principal_angle_rad", "rms_principal_angle_rad",
    "singular_relerr_395", "singular_relerr_396", "singular_relerr_397",
    "singular_relerr_398", "singular_relerr_399", "singular_relerr_400",
    "spectral_gap_400_401", "ssi_wall_seconds", "exact_svd_wall_seconds",
    "device_bytes_in_use", "device_peak_bytes_in_use", "job_peak_gpu_memory_mib",
    "resolved_relative_error", "complement_relative_error", "finite",
]

def sync(x):
    return jax.block_until_ready(x)

def norm_metrics(got, ref):
    ng, nr = jnp.linalg.norm(got), jnp.linalg.norm(ref)
    return (
        float(sync(jnp.linalg.norm(got-ref) / jnp.maximum(nr, 1e-12))),
        float(sync(jnp.vdot(got, ref) / jnp.maximum(ng*nr, 1e-12))),
    )

def constrained(x, lr, limit=1e-3):
    d = -lr*x
    return d * jnp.minimum(1.0, limit/jnp.maximum(jnp.linalg.norm(d), 1e-12))

def contributions(o, e, u, s, damping, weight):
    leading = jnp.maximum(jnp.abs(s[0]), 1e-12)
    retained = s/leading > damping
    inv, inv_perp = wssr._wssr_inverse_spectral_coefficients(
        jnp.where(retained, s, 1.0), leading, damping, "tikhonov", weight
    )
    force = o @ e
    projected = (u.T @ force)*retained.astype(o.dtype)
    resolved = u @ (inv*projected)
    complement = inv_perp*(force-u@projected)
    return resolved, complement

def memstats():
    m = jax.devices()[0].memory_stats() or {}
    return int(m.get("bytes_in_use", -1)), int(m.get("peak_bytes_in_use", -1))

def make_row(snapshot, label, q, o, e, state, key, exact, cfg, exact_seconds, lr):
    # Warm primitive compilation/caches are populated before the measured call.
    warm = wssr.right_warm_start_svd(o, state, key, 8, q, svd_working_rank=400)
    sync(warm)
    t0 = time.perf_counter()
    u, s, vh, rank = wssr.right_warm_start_svd(
        o, state, key, 8, q, svd_working_rank=400
    )
    sync((u, s, vh, rank)); elapsed = time.perf_counter()-t0
    ue, se, vhe = exact
    got = wssr._wssr_update_from_svd(
        o, e, state, u, s, vh, 3e-4, 1e-3, 400, 1.1, False,
        rank_update_max=400, spectral_regularization="tikhonov",
        complement_weight=cfg["complement_weight"],
    ).grad_like_update
    ref = wssr._wssr_update_from_svd(
        o, e, state, ue, se, vhe, 3e-4, 1e-3, 400, 1.1, False,
        rank_update_max=400, spectral_regularization="tikhonov",
        complement_weight=cfg["complement_weight"],
    ).grad_like_update
    sync((got, ref))
    rel, cos = norm_metrics(got, ref)
    crel, ccos = norm_metrics(constrained(got, lr), constrained(ref, lr))
    qa, _ = jnp.linalg.qr(u, mode="reduced"); qr, _ = jnp.linalg.qr(ue, mode="reduced")
    overlap = jnp.linalg.svd(qa.T@qr, compute_uv=False)
    angles = jnp.arccos(jnp.clip(overlap, -1., 1.))
    residual = jnp.linalg.norm(qr-qa@(qa.T@qr))
    proj = jnp.sqrt(jnp.maximum(0., 800.-2.*jnp.sum(overlap**2)))
    singular = {}
    for one in range(395, 401):
        i=one-1; singular[f"singular_relerr_{one}"] = float(sync(jnp.abs(s[i]-se[i])/jnp.maximum(jnp.abs(se[i]),1e-12)))
    rr=cr=float("nan")
    if snapshot == "C500":
        rg,cg=contributions(o,e,u,s,3e-4,cfg["complement_weight"])
        re,ce=contributions(o,e,ue,se,3e-4,cfg["complement_weight"])
        rr,_=norm_metrics(rg,re); cr,_=norm_metrics(cg,ce)
    used, peak = memstats()
    row = dict(
        snapshot=snapshot, initialization=label, q=q,
        update_relative_error=rel, update_cosine=cos,
        constrained_relative_error=crel, constrained_cosine=ccos,
        subspace_residual=float(sync(residual)), projector_frobenius_error=float(sync(proj)),
        max_principal_angle_rad=float(sync(jnp.max(angles))),
        rms_principal_angle_rad=float(sync(jnp.sqrt(jnp.mean(angles**2)))),
        spectral_gap_400_401=float(sync(jnp.abs(se[399]-se[400])/jnp.maximum(jnp.abs(se[399]),1e-12))),
        ssi_wall_seconds=elapsed, exact_svd_wall_seconds=exact_seconds,
        device_bytes_in_use=used, device_peak_bytes_in_use=peak,
        job_peak_gpu_memory_mib=float("nan"), resolved_relative_error=rr,
        complement_relative_error=cr,
        finite=bool(sync(jnp.all(jnp.isfinite(got)))) and bool(sync(jnp.all(jnp.isfinite(s)))),
    )
    row.update(singular)
    return row

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("snapshot", choices=SOURCES); ap.add_argument("--output", required=True)
    args=ap.parse_args(); source, ckname=SOURCES[args.snapshot]; out=Path(args.output); out.mkdir(parents=True, exist_ok=True)
    assert (source/"checkpoints"/ckname).is_file()
    cfg=io.load_config_dict(str(source), "config.json")
    dtype=runners._get_dtype(cfg); ions,charges,nelec=runners._get_electron_ion_config_as_arrays(cfg,dtype)
    epoch,data,params,opt_state,key=io.reload_vmc_state(str(source/"checkpoints"),ckname)
    positions=pacore.get_position_from_data(data)
    assert positions.shape[0] == 1000
    logpsi,_,_=runners._get_and_init_model(cfg.model,ions,charges,nelec,positions,key,dtype=dtype,apply_pmap=False)
    local=runners._assemble_mol_local_energy_fn(ions,charges,cfg.problem.ei_softening,cfg.problem.ee_softening,logpsi)
    energy_fn=physics_core.create_energy_and_statistics_fn(local,1000,runners._get_clipping_fn(cfg.vmc),cfg.vmc.nan_safe)
    energy,local_energies,_=energy_fn(params,positions)
    o_cur,_=wssr.center_and_scale_score_matrix(logpsi,params,positions)
    e_cur=wssr.center_and_scale_energy_residuals(local_energies,energy)
    if args.snapshot == "initial":
        base=wssr.initialize_wssr_warm_svd_core_state(o_cur.shape[0],400,400,dtype=o_cur.dtype,store_warm_u=True)
        states=[("baseline_warm_initial",base)]
        complement=0.0; step=0
    else:
        base=opt_state.core_state
        states=[("stored_warm_state",base)]
        complement=float(cfg.vmc.optimizer.wssr_warm_svd_right.complement_weight); step=500
    o,e=wssr.augment_wssr_system(o_cur,e_cur,base,0.8); sync((o,e))
    # One exact reference for the fixed operator. Keep 401 values for the cutoff gap.
    t0=time.perf_counter(); uall,sall,vhall=jnp.linalg.svd(o,full_matrices=False); sync((uall,sall,vhall)); exact_seconds=time.perf_counter()-t0
    exact=(uall[:,:400],sall[:400],vhall[:400,:])
    if args.snapshot == "initial":
        exact_state=wssr.WSSRWarmSVDCoreState(base.sr_o,base.ek,base.sr_rank0,base.sr_rank,exact[0],jnp.array(True))
        states.append(("exact_first_stored_initial",exact_state))
    lr=0.02/(1.0+1e-4*step)
    rows=[]
    for label,state in states:
        previous=None
        for q in BASE_Q:
            row=make_row(args.snapshot,label,q,o,e,state,key,exact,{"complement_weight":complement},exact_seconds,lr); rows.append(row)
            previous=row["update_relative_error"]
        if previous >= 1e-3:
            last=previous
            for q in (64,128):
                row=make_row(args.snapshot,label,q,o,e,state,key,exact,{"complement_weight":complement},exact_seconds,lr); rows.append(row)
                improvement=(last-row["update_relative_error"])/max(last,1e-30); last=row["update_relative_error"]
                if last<1e-3 or improvement<0.05: break
    meta={"snapshot":args.snapshot,"source":str(source/"checkpoints"/ckname),"stored_epoch":int(epoch),"nchains":int(positions.shape[0]),"operator_shape":list(map(int,o.shape)),"force_fixed":True,"mcmc_advanced":False,"update_applied":False,"complement_weight":complement,"exact_singular_values_395_405":[float(x) for x in np.asarray(sall[394:405])],"note":"Rank-400 SSI returns only 400 singular values; approximate errors 401-405 are mathematically unavailable without changing the working rank."}
    with (out/"results.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    (out/"metadata.json").write_text(json.dumps(meta,indent=2)); print(json.dumps(meta,indent=2));

if __name__ == "__main__": main()
