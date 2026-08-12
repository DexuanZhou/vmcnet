#!/usr/bin/env python3
import csv, math
from pathlib import Path
import numpy as np

BASE=Path('/scratch/dexuan1/runs/C_exact_first_complement/stage2_500_retry1')
A_RETRY=Path('/scratch/dexuan1/runs/C_exact_first_complement/stage2_500_retry2/A_warm_c0')
OUT=Path('/scratch/dexuan1/vmcnet/experiments/C_exact_first_complement/results');OUT.mkdir(parents=True,exist_ok=True)
def val(row,key):
    try:return float(row[key])
    except:return math.nan
def diag(run,key):
    p=run/f'{key}.txt';return [float(x) for x in p.read_text().split()] if p.exists() else []
def timing(meta):
    p=meta/'phase_timing.csv'
    if not p.exists():return math.nan
    with p.open() as f:r=[x for x in csv.DictReader(f) if x['event']=='epoch_end']
    t=np.array([float(x['monotonic_seconds']) for x in r]);return float(np.median(np.diff(t)[10:])) if len(t)>11 else math.nan
rows=[]
run_paths=[A_RETRY, BASE/'B_exact_c0', BASE/'C_exact_c1e4']
for run in run_paths:
    if run.name.endswith('.metadata'):continue
    p=run/'training_metrics.csv'
    if not p.exists():rows.append({'variant':run.name,'status':'MISSING'});continue
    with p.open() as f:d=list(csv.DictReader(f))
    tail=d[-100:];energy=np.array([val(r,'energy') for r in tail]);var=np.array([val(r,'variance') for r in tail]);raw=np.array([val(r,'variance_noclip') for r in tail]);acc=np.array([val(r,'accept_ratio') for r in tail])
    meta=Path(str(run)+'.metadata');gpu=meta/'gpu_memory_poll.csv';peak=math.nan
    if gpu.exists():
        with gpu.open() as f:peak=max((val(r,'memory_used_mib') for r in csv.DictReader(f)),default=math.nan)
    scale=np.array(diag(run,'wssr_diag_norm_constraint_scale')[-100:]);resolved=np.array(diag(run,'wssr_diag_resolved_norm')[-100:]);comp=np.array(diag(run,'wssr_diag_complement_norm')[-100:]);rawdir=np.array(diag(run,'wssr_diag_lr_scaled_update_norm')[-100:])
    stable=len(d)==500 and all(np.isfinite(x).all() for x in (energy,var,raw,acc,scale,resolved,comp,rawdir))
    rows.append(dict(variant=run.name,epochs=len(d),status='STABLE' if stable else 'FAILED',energy_mean=float(energy.mean()),energy_sem=float(energy.std(ddof=1)/np.sqrt(len(energy))),variance_clipped_mean=float(var.mean()),variance_raw_mean=float(raw.mean()),acceptance_median=float(np.median(acc)),seconds_per_epoch_median=timing(meta),peak_gpu_memory_mib=peak,constraint_scale_mean=float(scale.mean()),constraint_scale_min=float(scale.min()),raw_direction_norm_mean=float(rawdir.mean()),resolved_norm_mean=float(resolved.mean()),complement_norm_mean=float(comp.mean()),final_checkpoint=str(run/'checkpoints/500.npz')))
fields=sorted({k for r in rows for k in r})
with (OUT/'stage2_summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
with (OUT/'stage2_summary.md').open('w') as f:
 f.write('| '+' | '.join(fields)+' |\n|'+('|---'*len(fields))+'|\n')
 for r in rows:f.write('| '+' | '.join(str(r.get(k,'')) for k in fields)+' |\n')
print('\n'.join(str(r) for r in rows))
