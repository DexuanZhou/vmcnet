#!/usr/bin/env python3
import csv,json,math
from pathlib import Path
import numpy as np
R=Path('/scratch/dexuan1/vmcnet/experiments/C_wssr_six_improvements/results/stage1')
rows=[]
for d in (R/name for name in ('initial','A500','B500','C500')):
 p=d/'results.csv'
 if not p.exists():continue
 with p.open() as f:part=list(csv.DictReader(f))
 peak=math.nan;gp=d/'gpu_memory_poll.csv'
 if gp.exists():
  with gp.open() as f:peak=max((float(x['memory_used_mib']) for x in csv.DictReader(f)),default=math.nan)
 if not math.isfinite(peak):
  peak=max((float(x.get('gpu_memory_used_mib','nan')) for x in part),default=math.nan)
 for x in part:x['peak_gpu_memory_mib']=peak
 rows+=part
if len({x['snapshot'] for x in rows})!=4:raise SystemExit('Stage1 incomplete')
fields=sorted({k for x in rows for k in x})
with (R/'all_results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
groups={}
for x in rows:groups.setdefault((x['method'],x['parameter']),[]).append(x)
agg=[]
for (m,p),xs in groups.items():
 finite=len(xs)==4 and all(x['finite']=='True' for x in xs)
 def expected_width(x):
  if m=='cluster':return 512
  if m=='near_tail':return 400+int(p)
  if m=='smooth':return int(p.split('-')[1])
  return 400
 widths=[int(float(x.get('decomposition_width') or expected_width(x))) for x in xs]
 if m=='cluster':finite=finite and all(400<=int(float(x['effective_rank']))<=512 for x in xs)
 elif m=='near_tail':finite=finite and min(widths)>=400+int(p)
 elif m=='smooth':finite=finite and min(widths)>=int(p.split('-')[1])
 ratios=[float(x['complement_resolved_ratio']) for x in xs]
 runtime=[float(x['runtime_seconds']) for x in xs];exact=[float(x['exact_svd_seconds']) for x in xs]
 eligible=finite and (m!='iterative_complement' or max(ratios)<=.5)
 agg.append(dict(method=m,parameter=p,eligible=eligible,ineligible_reason=('complement/resolved exceeds 0.5' if m=='iterative_complement' and max(ratios)>.5 else ('' if finite else 'nonfinite, incomplete, or working-width gate failed')),mean_full_error=np.mean([float(x['full_relative_error']) for x in xs]),max_full_error=max(float(x['full_relative_error']) for x in xs),mean_cosine=np.mean([float(x['full_cosine']) for x in xs]),mean_runtime=np.mean(runtime),max_complement_ratio=max(ratios),mean_effective_rank=np.mean([float(x['effective_rank']) for x in xs])))
baseline_error=next(x['mean_full_error'] for x in agg if x['method']=='baseline')
for x in agg:
 if (x['method']=='iterative_complement' and x['mean_runtime']>=np.mean([
     float(y['exact_svd_seconds']) for y in rows if y['method']=='baseline'])
     and x['mean_full_error']>=baseline_error):
  x['eligible']=False
selected={}
for m in sorted({x['method'] for x in agg if x['method'] not in ('baseline','fixed_complement_reference')}):
 candidates=[x for x in agg if x['method']==m and x['eligible']]
 if candidates:selected[m]=min(candidates,key=lambda x:(x['mean_full_error'],x['mean_runtime']))
with (R/'aggregate.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(agg[0]));w.writeheader();w.writerows(agg)
(R/'selection.json').write_text(json.dumps(selected,indent=2))
order=['cluster','near_tail','adaptive_complement','smooth','force_aware','iterative_complement']
(R/'stage2_methods.json').write_text(json.dumps(['baseline']+[m for m in order if m in selected],indent=2))
try:
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(9,5))
 for m in sorted({x['method'] for x in agg}):
  z=[x for x in agg if x['method']==m];ax.scatter([x['mean_runtime'] for x in z],[x['mean_full_error'] for x in z],label=m)
 ax.set(xlabel='mean replay runtime (s)',ylabel='mean relative error vs exact full');ax.set_yscale('log');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(R/'stage1_error_runtime.svg');plt.close(fig)
except ModuleNotFoundError:
 (R/'plot_deferred.txt').write_text('matplotlib unavailable in training environment; final report generator will create plots.\n')
print(json.dumps(selected,indent=2))
