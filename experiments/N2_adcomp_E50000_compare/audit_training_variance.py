#!/usr/bin/env python3
import csv,json,math
from pathlib import Path
import numpy as np
ROOT=Path('/scratch/dexuan1/runs/N2_adcomp_E50000_compare')
OUT=Path('/scratch/dexuan1/vmcnet/experiments/N2_adcomp_E50000_compare/results/variance_audit')
RUNS=['A_rank800_hard','B_rank400_beta02'];N=4096
OUT.mkdir(parents=True,exist_ok=True); summary=[]; blocks=[]
for run in RUNS:
 rows=list(csv.DictReader(open(ROOT/run/'training_metrics.csv')))
 epochs=np.array([int(r['epoch']) for r in rows]);assert len(rows)==50000 and np.array_equal(epochs,np.arange(1,50001))
 for window in (1000,5000):
  z=rows[-window:];means=np.array([float(r['energy_noclip']) for r in z],dtype=np.float64);var=np.array([float(r['variance_noclip']) for r in z],dtype=np.float64);clip=np.array([float(r['variance']) for r in z],dtype=np.float64)
  grand=means.mean(dtype=np.float64);within_ss=(N-1)*var.sum(dtype=np.float64);between_ss=N*np.square(means-grand,dtype=np.float64).sum(dtype=np.float64);nt=N*window;pooled=(within_ss+between_ss)/(nt-1)
  # A float32 reconstruction audit from already-rounded logged summaries. This
  # does not recreate the original walker-level reduction.
  mf=means.astype(np.float32);vf=var.astype(np.float32);gf=mf.mean(dtype=np.float32);pooled32=((np.float32(N-1)*vf.sum(dtype=np.float32)+np.float32(N)*np.square(mf-gf,dtype=np.float32).sum(dtype=np.float32))/np.float32(nt-1))
  q=np.quantile(var,[0,.001,.01,.25,.5,.75,.99,.999,1.])
  summary.append(dict(run=run,window=window,definition_reported='arithmetic mean of per-epoch unbiased walker sample variances',nchains_per_epoch=N,epochs=window,total_walker_evaluations=nt,raw_energy_mean=grand,mean_per_epoch_raw_variance=var.mean(),median_per_epoch_raw_variance=np.median(var),variance_q000=q[0],variance_q001=q[1],variance_q01=q[2],variance_q25=q[3],variance_q50=q[4],variance_q75=q[5],variance_q99=q[6],variance_q999=q[7],variance_max=q[8],pooled_raw_variance=pooled,pooled_raw_std=math.sqrt(pooled),within_population_component=within_ss/nt,between_population_component=between_ss/nt,variance_of_epoch_means=means.var(ddof=1),mean_clipped_variance=clip.mean(),ratio_of_mean_raw_to_mean_clipped=var.mean()/clip.mean(),median_epoch_raw_clipped_ratio=np.median(var/clip),float32_summary_reconstruction=pooled32,float32_vs_float64_summary_relative_difference=abs(float(pooled32)-pooled)/pooled,training_sample_quantiles='unavailable: walker local energies not saved',extreme_sample_count='unavailable: walker local energies not saved'))
 if run:
  z=rows[-5000:]
  for b in range(50):
   a=z[100*b:100*(b+1)];vr=np.array([float(r['variance_noclip']) for r in a]);vc=np.array([float(r['variance']) for r in a]);en=np.array([float(r['energy_noclip']) for r in a]);blocks.append(dict(run=run,block=b+1,epoch_start=int(a[0]['epoch']),epoch_end=int(a[-1]['epoch']),raw_variance_mean=vr.mean(),raw_variance_median=np.median(vr),raw_variance_max=vr.max(),clipped_variance_mean=vc.mean(),raw_clipped_ratio=vr.mean()/vc.mean(),raw_energy_mean=en.mean()))
for fn,data in [('summary.csv',summary),('final5000_blocks.csv',blocks)]:
 with (OUT/fn).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
(OUT/'audit.json').write_text(json.dumps({'formula':{'per_epoch_unbiased_variance':'s_t^2 = sum_i (E_ti - mean_t)^2 / (4096 - 1)','reported_window_raw_variance':'mean_t(s_t^2)','pooled_unbiased_variance':'[(4095)*sum_t s_t^2 + 4096*sum_t(mean_t-grand_mean)^2] / (4096*T-1)','within_population_component':'(4095*sum_t s_t^2)/(4096*T)','between_population_component':'sum_t(mean_t-grand_mean)^2/T'},'units':{'energy':'Hartree','variance':'Hartree^2','standard_deviation':'Hartree'},'local_energy_samples_saved_during_training':False,'single_device':True,'nchains':4096},indent=2)+'\n')
try:
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(2,1,figsize=(11,7),sharex=True)
 for run in RUNS:
  rr=[r for r in blocks if r['run']==run];x=[r['epoch_end'] for r in rr];ax[0].plot(x,[r['raw_variance_mean'] for r in rr],marker='o',ms=2,label=run);ax[1].plot(x,[r['raw_clipped_ratio'] for r in rr],marker='o',ms=2,label=run)
 ax[0].set_ylabel('Raw variance (Ha²)');ax[1].set_ylabel('Raw/clipped variance');ax[1].set_xlabel('Epoch (100-epoch non-overlapping blocks)');ax[0].legend();ax[0].grid(alpha=.2);ax[1].grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'final5000_blocks.svg');fig.savefig(OUT/'final5000_blocks.png',dpi=180)
except ImportError:pass
print(json.dumps(summary,indent=2))
