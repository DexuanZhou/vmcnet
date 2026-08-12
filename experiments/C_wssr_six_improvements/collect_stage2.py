#!/usr/bin/env python3
import csv,json,math
from pathlib import Path
import numpy as np
X=Path('/scratch/dexuan1/vmcnet/experiments/C_wssr_six_improvements');sel=json.loads((X/'results/stage1/selection.json').read_text());methods=json.loads((X/'results/stage1/stage2_methods.json').read_text());rows=[]
for m in methods:
 p='warm2' if m=='baseline' else str(sel[m]['parameter']);v=f'{m}_{p.replace("+","_")}';run=Path('/scratch/dexuan1/runs/C_wssr6_stage2_E200')/v
 with (run/'training_metrics.csv').open() as f:d=list(csv.DictReader(f));tail=d[-50:]
 def a(k,z=tail):return np.array([float(r[k]) for r in z])
 finite=all(np.isfinite(a(k)).all() for k in ('energy','variance','variance_noclip','accept_ratio'));early=json.loads((Path(str(run)+'.metadata')/'early_stop.json').read_text()) if (Path(str(run)+'.metadata')/'early_stop.json').exists() else {}
 status='STABLE' if len(d)==200 and finite and not early.get('triggered') else ('UNSTABLE' if not finite or early.get('triggered') else 'MARGINAL')
 acc=[]
 # Epoch 1 is the common initial fixed snapshot; later values use actual checkpoints.
 with (X/'results/stage1/all_results.csv').open() as f:s1=list(csv.DictReader(f))
 sx=[r for r in s1 if r['snapshot']=='initial' and r['method']==m and r['parameter']==p];
 if sx:acc.append(float(sx[0]['full_relative_error']))
 for e in (50,100,200):acc.append(json.loads((X/f'results/stage2_accuracy/{v}_e{e}.json').read_text())['full_relative_error'])
 meta=Path(str(run)+'.metadata');peak=math.nan
 if (meta/'gpu_memory_poll.csv').exists():
  with (meta/'gpu_memory_poll.csv').open() as f:peak=max(float(x['memory_used_mib']) for x in csv.DictReader(f))
 sec=math.nan
 if (meta/'phase_timing.csv').exists():
  with (meta/'phase_timing.csv').open() as f:t=[float(x['monotonic_seconds']) for x in csv.DictReader(f) if x['event']=='epoch_end'];sec=float(np.median(np.diff(t)[10:]))
 rows.append(dict(method=m,parameter=p,variant=v,epochs=len(d),status=status,tail50_energy_mean=a('energy').mean(),tail50_energy_sem=a('energy').std(ddof=1)/np.sqrt(50),tail50_clipped_variance=a('variance').mean(),tail50_raw_variance=a('variance_noclip').mean(),tail50_acceptance_median=np.median(a('accept_ratio')),median_seconds_epoch=sec,peak_gpu_memory_mib=peak,full_error_epoch1=acc[0],full_error_epoch50=acc[1],full_error_epoch100=acc[2],full_error_epoch200=acc[3],mean_full_error=np.mean(acc),early_stop_reason=early.get('reason','')))
stable=[r for r in rows if r['method']!='baseline' and r['status']=='STABLE'];top=sorted(stable,key=lambda r:(r['tail50_raw_variance'],r['tail50_clipped_variance'],r['mean_full_error'],r['median_seconds_epoch']))[:3]
if len(top)!=3:raise SystemExit('fewer than three stable improvements')
out=X/'results/stage2';out.mkdir(parents=True,exist_ok=True);fields=list(rows[0]);
with (out/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
(out/'top3.json').write_text(json.dumps(top,indent=2))
try:
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,2,figsize=(10,4));ax[0].bar([r['method'] for r in rows],[r['tail50_raw_variance'] for r in rows]);ax[0].tick_params(axis='x',rotation=60);ax[0].set_ylabel('final-50 raw variance');ax[1].scatter([r['median_seconds_epoch'] for r in rows],[r['mean_full_error'] for r in rows]);ax[1].set(xlabel='seconds/epoch',ylabel='mean full-reference error');fig.tight_layout();fig.savefig(out/'stage2_variance_accuracy.svg');plt.close(fig)
except ModuleNotFoundError:
 (out/'plot_deferred.txt').write_text('matplotlib unavailable in training environment; final report generator will create plots.\n')
print(json.dumps(top,indent=2))
