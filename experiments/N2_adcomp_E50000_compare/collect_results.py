#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path('/scratch/dexuan1/runs/N2_adcomp_E50000_compare')
FROZEN=Path('/scratch/dexuan1/runs/N2_adcomp_E50000_compare_frozen')
OUT=Path('/scratch/dexuan1/vmcnet/experiments/N2_adcomp_E50000_compare/results')
METHODS=[('A_rank800_hard',800,0.),('B_rank400_beta02',400,.2)]
OUT.mkdir(parents=True,exist_ok=True)

def values(path):
    a=np.loadtxt(path,ndmin=1)
    return np.asarray(a,dtype=float).reshape(-1)

def block_sem(x):
    x=np.asarray(x,dtype=float); n=len(x)
    b=max(20,int(np.sqrt(n))); m=n//b
    if m<2: return float(np.std(x,ddof=1)/np.sqrt(n))
    means=x[:m*b].reshape(m,b).mean(1)
    return float(means.std(ddof=1)/np.sqrt(m))

rows=[]
for name,rank,beta in METHODS:
    run=ROOT/name; ev=FROZEN/name/'eval'
    cfg=json.loads((run/'config.json').read_text())
    met=pd.read_csv(run/'training_metrics.csv')
    if len(met)!=50000 or not np.isfinite(met.select_dtypes('number')).all().all():
        raise RuntimeError(f'{name}: incomplete/nonfinite training')
    for e in (10000,20000,30000,40000,50000):
        if not (run/f'checkpoints/{e}.npz').is_file(): raise RuntimeError(f'{name}: missing checkpoint {e}')
    stats=json.loads((ev/'statistics.json').read_text())
    le=values(ev/'local_energies.txt')
    if le.size != 4096000 or not np.isfinite(le).all(): raise RuntimeError(f'{name}: bad frozen samples')
    frozen_center = float(stats['average'])
    epoch_variances = le.reshape(-1, 4096).var(axis=1, ddof=1)
    block_variances = epoch_variances.reshape(-1, 100).mean(axis=1)
    phase=pd.read_csv(Path(str(run)+'.metadata')/'phase_timing.csv')
    pe=phase[phase.event=='epoch_end'].sort_values('epoch')
    dt=np.diff(pe.monotonic_seconds.to_numpy())
    gpu_memory = np.loadtxt(
        Path(str(run)+'.metadata')/'gpu_memory_poll.csv',
        delimiter=',',
        skiprows=1,
        usecols=1,
        ndmin=1,
    )
    row=dict(variant=name,rank=rank,beta=beta,completed_epochs=len(met),status='STABLE',
             median_seconds_epoch=float(np.median(dt[10:])),total_wall_hours=float((pe.monotonic_seconds.iloc[-1]-phase.monotonic_seconds.iloc[0])/3600),
             peak_gpu_memory_mib=float(gpu_memory.max()),final_checkpoint=str(run/'checkpoints/50000.npz'),
             frozen_energy=float(stats['average']),frozen_sem=float(stats['std_err']),frozen_raw_variance=float(stats['variance']),
             frozen_iat=float(stats['integrated_autocorrelation']),frozen_ess=float(le.size/stats['integrated_autocorrelation']),
             frozen_acceptance=float(values(ev/'accept_ratio.txt').mean()),frozen_nonfinite=0,
             frozen_min=float(le.min()),frozen_q001=float(np.quantile(le,.001)),frozen_q01=float(np.quantile(le,.01)),
             frozen_median=float(np.median(le)),frozen_q99=float(np.quantile(le,.99)),frozen_q999=float(np.quantile(le,.999)),frozen_max=float(le.max()))
    for threshold in (5.0, 10.0, 20.0):
        row[f'frozen_abs_deviation_gt_{int(threshold)}ha_fraction'] = float(
            np.mean(np.abs(le - frozen_center) > threshold)
        )
    row.update(
        frozen_epoch_variance_median=float(np.median(epoch_variances)),
        frozen_epoch_variance_q99=float(np.quantile(epoch_variances, .99)),
        frozen_epoch_variance_max=float(epoch_variances.max()),
        frozen_block100_variance_median=float(np.median(block_variances)),
        frozen_block100_variance_max=float(block_variances.max()),
        frozen_samples_path=str(ev/'local_energies.txt'),
        frozen_statistics_path=str(ev/'statistics.json'),
    )
    for label,n in [('last1000',1000),('last5000',5000),('last10pct',5000)]:
        d=met.tail(n)
        row[f'{label}_energy']=float(d.energy_noclip.mean())
        row[f'{label}_energy_sem']=block_sem(d.energy_noclip)
        row[f'{label}_clipped_variance']=float(d.variance.mean())
        row[f'{label}_raw_variance']=float(d.variance_noclip.mean())
        row[f'{label}_acceptance']=float(d.accept_ratio.median())
    ratio=values(run/'wssr_diag_complement_resolved_ratio.txt')
    scale=values(run/'wssr_diag_norm_constraint_scale.txt')
    cap=values(run/'wssr_diag_adaptive_cap_active.txt') if beta else np.zeros(len(ratio))
    row.update(beta_cap_active_fraction=float(cap.mean()),complement_ratio_mean=float(ratio.mean()),
               complement_ratio_median=float(np.median(ratio)),complement_ratio_max=float(ratio.max()),
               constraint_scale_min=float(scale.min()),constraint_active_fraction=float((scale<.999999).mean()))
    rows.append(row)
df=pd.DataFrame(rows); df.to_csv(OUT/'summary.csv',index=False)
a,b=rows
decision={
 'lower_frozen_variance': a['variant'] if a['frozen_raw_variance']<b['frozen_raw_variance'] else b['variant'],
 'rank800_variance_improvement_percent':100*(b['frozen_raw_variance']-a['frozen_raw_variance'])/b['frozen_raw_variance'],
 'rank800_runtime_premium_percent':100*(a['median_seconds_epoch']/b['median_seconds_epoch']-1),
 'rank800_memory_premium_percent':100*(a['peak_gpu_memory_mib']/b['peak_gpu_memory_mib']-1),
}
(OUT/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(3,2,figsize=(11,10),sharex=True); ax=ax.ravel()
for name,rank,beta in METHODS:
    run=ROOT/name; m=pd.read_csv(run/'training_metrics.csv'); x=m.epoch
    roll=1000
    ax[0].plot(x,m.energy_noclip.rolling(roll,min_periods=1).mean(),label=name)
    ax[1].plot(x,m.variance_noclip.rolling(roll,min_periods=1).mean(),label=name)
    ax[2].plot(x,m.variance.rolling(roll,min_periods=1).mean(),label=name)
    phase=pd.read_csv(Path(str(run)+'.metadata')/'phase_timing.csv'); p=phase[phase.event=='epoch_end'].sort_values('epoch')
    ax[3].plot(p.epoch.iloc[1:],np.diff(p.monotonic_seconds).clip(0,10),label=name,alpha=.6)
    ax[4].plot(x,values(run/'wssr_diag_norm_constraint_scale.txt'),label=name)
    if beta: ax[5].plot(x,values(run/'wssr_diag_complement_resolved_ratio.txt'),label=name)
for a0,t in zip(ax,['rolling raw energy','rolling raw variance','rolling clipped variance','seconds/epoch','constraint scale','complement/resolved ratio']): a0.set_title(t); a0.grid(alpha=.2)
ax[0].legend(); fig.tight_layout(); fig.savefig(OUT/'trajectories.png',dpi=180); fig.savefig(OUT/'trajectories.svg')
def dataframe_to_markdown(frame):
    """Render a small dataframe without pandas' optional tabulate dependency."""
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(lines)

lines=['# N2 E50000 rank800 vs adaptive rank400','',dataframe_to_markdown(df),'','## Decision','',json.dumps(decision,indent=2)]
(OUT/'report.md').write_text('\n'.join(lines)+'\n')
