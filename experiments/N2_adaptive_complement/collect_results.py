#!/usr/bin/env python3
"""Collect N2 adaptive-complement training and frozen-evaluation results."""
import csv, json, math, statistics
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; OUT=HERE/"results"; OUT.mkdir(parents=True,exist_ok=True)
METHODS=[("A_rank400_hard",400,0.),("B_rank400_beta01",400,.1),
         ("C_rank400_beta02",400,.2),("D_rank800_hard",800,0.)]
ORIGINAL_SUCCESS={0,1,4,7,8,10}
RETRY23={2,3}
RETRY3={5}
RETRY4={6,9,11}

def txt(path):
    if not path.exists(): return np.array([])
    return np.loadtxt(path,ndmin=1)
def peak(meta):
    p=meta/"gpu_memory_poll.csv"
    if not p.exists(): return math.nan
    with p.open() as f: return max(float(r["memory_used_mib"]) for r in csv.DictReader(f))
def median_epoch(meta):
    p=meta/"phase_timing.csv"
    if not p.exists(): return math.nan
    with p.open() as f:t=[float(r["monotonic_seconds"]) for r in csv.DictReader(f) if r["event"]=="epoch_end"]
    return float(np.median(np.diff(t)[10:])) if len(t)>11 else math.nan

rows=[]
for variant,rank,beta in METHODS:
  for seed in range(3):
    task=len(rows)
    run=Path("/scratch/dexuan1/runs/N2_adaptive_complement_E500")/variant/f"seed{seed}"
    if task in ORIGINAL_SUCCESS:
      frozen_root=Path("/scratch/dexuan1/runs/N2_adaptive_complement_frozen_retry1")
    elif task in RETRY23:
      frozen_root=Path("/scratch/dexuan1/runs/N2_adaptive_complement_frozen_retry2")
    elif task in RETRY3:
      frozen_root=Path("/scratch/dexuan1/runs/N2_adaptive_complement_frozen_retry3")
    elif task in RETRY4:
      frozen_root=Path("/scratch/dexuan1/runs/N2_adaptive_complement_frozen_retry4")
    else:
      raise RuntimeError(f"unmapped frozen task {task}")
    frozen=frozen_root/variant/f"seed{seed}"
    with (run/"training_metrics.csv").open() as f:d=list(csv.DictReader(f))
    tail=d[-50:]; arr=lambda k:np.array([float(r[k]) for r in tail])
    diag=lambda k:txt(run/(k+".txt"))
    ratio=diag("wssr_diag_complement_resolved_ratio")
    cap=diag("wssr_diag_adaptive_cap_active")
    scale=diag("wssr_diag_norm_constraint_scale")
    stats_path=frozen/"eval/statistics.json"; local_path=frozen/"eval/local_energies.txt"
    metrics_path=frozen/"eval/training_metrics.csv"
    protocol_path=Path(str(frozen)+".metadata/evaluation_protocol.txt")
    for required in (stats_path,local_path,metrics_path,protocol_path):
      if not required.exists(): raise RuntimeError(f"missing task {task} output: {required}")
    stats=json.loads(stats_path.read_text())
    local=np.loadtxt(local_path)
    protocol=protocol_path.read_text()
    expected_checkpoint=str(run/"checkpoints/500.npz")
    expected_tag=2026072200+task
    with metrics_path.open() as f: eval_rows=list(csv.DictReader(f))
    if local.shape!=(1000,4096) or len(eval_rows)!=1000:
      raise RuntimeError(f"task {task} incomplete: local={local.shape}, epochs={len(eval_rows)}")
    if (f"source_checkpoint={expected_checkpoint}" not in protocol
        or f"eval_prng_tag={expected_tag}" not in protocol
        or "nchains=4096" not in protocol or "nburn=10000" not in protocol
        or "nepochs=1000" not in protocol or "nsteps_per_param_update=10" not in protocol):
      raise RuntimeError(f"task {task} protocol/checkpoint mismatch")
    if not np.isfinite(local).all(): raise RuntimeError(f"task {task} has nonfinite local energies")
    finite_local=local[np.isfinite(local)]
    quant=np.quantile(finite_local,[0,.001,.01,.05,.5,.95,.99,.999,1.])
    iat=float(stats["integrated_autocorrelation"]); total=int(local.size)
    finite_train=all(np.isfinite(arr(k)).all() for k in ("energy","variance","variance_noclip","accept_ratio"))
    rows.append(dict(variant=variant,rank=rank,requested_beta=beta,seed=seed,
      completed_epochs=len(d),status="STABLE" if len(d)==500 and finite_train else "FAILED",
      tail50_energy=float(arr("energy").mean()),tail50_clipped_variance=float(arr("variance").mean()),
      tail50_raw_variance=float(arr("variance_noclip").mean()),tail50_acceptance=float(np.median(arr("accept_ratio"))),
      median_seconds_epoch=median_epoch(Path(str(run)+".metadata")),peak_gpu_memory_mib=peak(Path(str(run)+".metadata")),
      complement_ratio_mean=float(np.mean(ratio)) if ratio.size else 0.,complement_ratio_max=float(np.max(ratio)) if ratio.size else 0.,
      beta_cap_active_fraction=float(np.mean(cap)) if cap.size else 0.,norm_constraint_scale_min=float(np.min(scale)) if scale.size else math.nan,
      norm_constraint_active_fraction=float(np.mean(scale<1-1e-7)) if scale.size else math.nan,
      frozen_energy=float(stats["average"]),frozen_energy_sem=float(stats["std_err"]),
      frozen_raw_variance=float(stats["variance"]),frozen_iat=iat,frozen_ess=float(total/max(iat,1e-30)),
      frozen_acceptance=float(np.mean(txt(frozen/"eval/accept_ratio.txt"))),frozen_nonfinite_count=int(local.size-finite_local.size),
      frozen_min=float(quant[0]),frozen_q001=float(quant[1]),frozen_q01=float(quant[2]),frozen_q05=float(quant[3]),
      frozen_median=float(quant[4]),frozen_q95=float(quant[5]),frozen_q99=float(quant[6]),frozen_q999=float(quant[7]),frozen_max=float(quant[8]),
      checkpoint=expected_checkpoint,frozen_output=str(frozen),evaluation_task_id=task))
if len(rows)!=12 or len({(r["variant"],r["seed"]) for r in rows})!=12 or len({r["frozen_output"] for r in rows})!=12:
  raise RuntimeError("collector requires exactly 12 unique method/seed/output combinations")
with (OUT/"per_seed.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

agg=[]
for variant,rank,beta in METHODS:
    rs=[r for r in rows if r["variant"]==variant]
    q=dict(variant=variant,rank=rank,requested_beta=beta,all_three_stable=all(r["status"]=="STABLE" for r in rs))
    for k in ("frozen_energy","frozen_raw_variance","median_seconds_epoch","peak_gpu_memory_mib","tail50_raw_variance","beta_cap_active_fraction","complement_ratio_mean","norm_constraint_active_fraction"):
        a=np.array([r[k] for r in rs],float);q[k+"_mean"]=float(a.mean());q[k+"_std"]=float(a.std(ddof=1))
    sems=np.array([r["frozen_energy_sem"] for r in rs],float)
    q["frozen_energy_within_run_sem"]=float(np.sqrt(np.sum(sems**2))/3)
    q["frozen_energy_cross_seed_sem"]=q["frozen_energy_std"]/np.sqrt(3)
    q["frozen_energy_total_uncertainty"]=float(np.hypot(q["frozen_energy_within_run_sem"],q["frozen_energy_cross_seed_sem"]))
    q["frozen_iat_mean"]=float(np.mean([r["frozen_iat"] for r in rs]))
    q["frozen_ess_total"]=float(np.sum([r["frozen_ess"] for r in rs]))
    q["frozen_acceptance_mean"]=float(np.mean([r["frozen_acceptance"] for r in rs]))
    agg.append(q)
base=agg[0]
for q in agg:
    q["raw_variance_improvement_percent"]=100*(base["frozen_raw_variance_mean"]-q["frozen_raw_variance_mean"])/base["frozen_raw_variance_mean"]
    q["runtime_multiplier"]=q["median_seconds_epoch_mean"]/base["median_seconds_epoch_mean"]
    q["variance_improvement_per_runtime"]=q["raw_variance_improvement_percent"]/q["runtime_multiplier"]
with (OUT/"cross_seed.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(agg[0]));w.writeheader();w.writerows(agg)

stable=[q for q in agg if q["all_three_stable"]]
winner=min(stable,key=lambda q:(q["frozen_raw_variance_mean"],q["frozen_raw_variance_std"],q["median_seconds_epoch_mean"]))
b1,b2,r8=agg[1],agg[2],agg[3]
lines=["# N2 adaptive-complement validation","",
"All methods restart independently from the validated N2 R=2.068 Bohr, n=4096 KFAC-pre5000 checkpoint.","",
"|variant|stable|frozen energy ± total uncertainty|frozen raw variance ± seed SD|IAT|ESS total|acceptance|train s/epoch|runtime ×|peak MiB|",
"|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for q in agg: lines.append(f"|{q['variant']}|{q['all_three_stable']}|{q['frozen_energy_mean']:.8f} ± {q['frozen_energy_total_uncertainty']:.2g}|{q['frozen_raw_variance_mean']:.6g} ± {q['frozen_raw_variance_std']:.3g}|{q['frozen_iat_mean']:.3f}|{q['frozen_ess_total']:.0f}|{q['frozen_acceptance_mean']:.4f}|{q['median_seconds_epoch_mean']:.4f}|{q['runtime_multiplier']:.3f}|{q['peak_gpu_memory_mib_mean']:.0f}|")
lines += ["","## Conclusions","",
f"1. Adaptive complement generalizes to N2: {'yes' if min(b1['frozen_raw_variance_mean'],b2['frozen_raw_variance_mean']) < base['frozen_raw_variance_mean'] else 'no'}.",
f"2. beta=0.2 is consistently better than beta=0.1: {'yes' if b2['frozen_raw_variance_mean'] < b1['frozen_raw_variance_mean'] and b2['frozen_raw_variance_std'] <= b1['frozen_raw_variance_std'] else 'not established'}.",
f"3. Cap-active fractions: beta=0.1 {b1['beta_cap_active_fraction_mean']:.1%}; beta=0.2 {b2['beta_cap_active_fraction_mean']:.1%}.",
f"4. Frozen raw variance changes, not merely clipped training variance: beta=0.1 {b1['raw_variance_improvement_percent']:.2f}%, beta=0.2 {b2['raw_variance_improvement_percent']:.2f}% versus rank400 hard.",
f"5. rank400 beta=0.2 {'matches or exceeds' if b2['frozen_raw_variance_mean'] <= r8['frozen_raw_variance_mean'] else 'does not match'} rank800 frozen variance; runtime ratio beta0.2/rank800 = {b2['median_seconds_epoch_mean']/r8['median_seconds_epoch_mean']:.3f}.",
f"6. Complement dominance / constraint saturation: max cross-seed mean ratio {max(q['complement_ratio_mean_mean'] for q in (b1,b2)):.3f}; constraint-active fraction beta0.2 {b2['norm_constraint_active_fraction_mean']:.1%}.",
f"7. Recommended matched E50000 candidate: {winner['variant']} (not submitted).","",
"Energy is used only as an internal relative diagnostic."]
(OUT/"report.md").write_text("\n".join(lines)+"\n")
(OUT/"selection.json").write_text(json.dumps({"winner":winner,"all_methods":agg,"E50000_submitted":False},indent=2)+"\n")
print("\n".join(lines))
