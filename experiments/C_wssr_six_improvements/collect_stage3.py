#!/usr/bin/env python3
import csv,json,math
from pathlib import Path
import numpy as np
X=Path('/scratch/dexuan1/vmcnet/experiments/C_wssr_six_improvements');top=json.loads((X/'results/stage2/top3.json').read_text());methods=[{'method':'baseline','parameter':'warm2'}]+top;rows=[]
for z in methods:
 m=z['method'];p=str(z['parameter']);v=f'{m}_{p.replace("+","_")}'
 for seed in range(3):
  run=Path('/scratch/dexuan1/runs/C_wssr6_stage3_E500')/v/f'seed{seed}';ev=Path('/scratch/dexuan1/runs/C_wssr6_stage3_frozen')/v/f'seed{seed}'/'eval/statistics.json'
  with (run/'training_metrics.csv').open() as f:d=list(csv.DictReader(f));tail=d[-50:]
  arr=lambda k:np.array([float(r[k]) for r in tail]);stats=json.loads(ev.read_text());meta=Path(str(run)+'.metadata');peak=math.nan
  if (meta/'gpu_memory_poll.csv').exists():
   with (meta/'gpu_memory_poll.csv').open() as f:peak=max(float(r['memory_used_mib']) for r in csv.DictReader(f))
  sec=math.nan
  if (meta/'phase_timing.csv').exists():
   with (meta/'phase_timing.csv').open() as f:t=[float(r['monotonic_seconds']) for r in csv.DictReader(f) if r['event']=='epoch_end'];sec=float(np.median(np.diff(t)[10:]))
  rows.append(dict(method=m,parameter=p,seed=seed,status='STABLE' if len(d)==500 and all(np.isfinite(arr(k)).all() for k in ('energy','variance','variance_noclip','accept_ratio')) else 'FAILED',tail50_energy_mean=arr('energy').mean(),tail50_clipped_variance=arr('variance').mean(),tail50_raw_variance=arr('variance_noclip').mean(),tail50_acceptance=np.median(arr('accept_ratio')),median_seconds_epoch=sec,peak_gpu_memory_mib=peak,frozen_energy=stats['average'],frozen_energy_sem=stats['std_err'],frozen_raw_variance=stats['variance'],frozen_iat=stats['integrated_autocorrelation'],checkpoint=str(run/'checkpoints/500.npz')))
out=X/'results/stage3';out.mkdir(parents=True,exist_ok=True);fields=list(rows[0]);
with (out/'per_seed.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
agg=[]
for z in methods:
 rs=[r for r in rows if r['method']==z['method']]
 q={'method':z['method'],'parameter':str(z['parameter'])}
 for k in ('frozen_energy','frozen_raw_variance','median_seconds_epoch','tail50_raw_variance'):
  a=np.array([r[k] for r in rs]);q[k+'_mean']=a.mean();q[k+'_std']=a.std(ddof=1)
 q['all_stable']=all(r['status']=='STABLE' for r in rs);agg.append(q)
with (out/'cross_seed.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(agg[0]));w.writeheader();w.writerows(agg)
try:
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,2,figsize=(10,4));names=[r['method'] for r in agg];ax[0].bar(names,[r['frozen_raw_variance_mean'] for r in agg],yerr=[r['frozen_raw_variance_std'] for r in agg]);ax[1].scatter([r['median_seconds_epoch_mean'] for r in agg],[r['frozen_raw_variance_mean'] for r in agg]);ax[0].tick_params(axis='x',rotation=50);ax[0].set_ylabel('frozen raw variance');ax[1].set(xlabel='seconds/epoch',ylabel='frozen raw variance');fig.tight_layout();fig.savefig(out/'stage3_frozen_variance_runtime.svg');plt.close(fig)
except ModuleNotFoundError:
 (out/'plot_deferred.txt').write_text('matplotlib unavailable in training environment; final report generator will create plots.\n')
best_var=min(agg,key=lambda r:r['frozen_raw_variance_mean']);best_energy=min(agg,key=lambda r:r['frozen_energy_mean']);best_trade=min(agg,key=lambda r:(r['frozen_raw_variance_mean']*r['median_seconds_epoch_mean']));report={'best_frozen_variance':best_var,'best_relative_frozen_energy':best_energy,'best_speed_variance_tradeoff':best_trade,'absolute_energy_caveat':'C absolute-reference inconsistency remains unresolved; energies are internal relative comparisons only.'};(out/'conclusions.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
