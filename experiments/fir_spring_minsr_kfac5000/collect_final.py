#!/usr/bin/env python3
import csv,json,math,pathlib,statistics
BASE=pathlib.Path('/scratch/dexuan1/runs'); OUT=pathlib.Path(__file__).parent/'results';OUT.mkdir(exist_ok=True)
runs=[('C','MinSR',0.0,BASE/'C_minsr_after_kfac5000_E50000'),('C','SPRING',.99,BASE/'C_spring_mu099_after_kfac5000_E50000'),('N2','MinSR',0.0,BASE/'N2_minsr_after_kfac5000_E50000'),('N2','SPRING',.99,BASE/'N2_spring_mu099_after_kfac5000_E50000')]
rows=[]
for system,opt,mu,run in runs:
 if not (run/'training_metrics.csv').exists(): continue
 m=list(csv.DictReader(open(run/'training_metrics.csv'))); fin=[r for r in m if all(math.isfinite(float(r[k])) for k in ('energy','variance','accept_ratio'))]; tail=fin[-1000:]
 meta=pathlib.Path(str(run)+'_metadata')
 timing=list(csv.DictReader(open(meta/'phase_timing.csv'))) if (meta/'phase_timing.csv').exists() else []; ends=[(int(r['epoch']),float(r['monotonic_seconds'])) for r in timing if r['event']=='epoch_end']; dt=[ends[i][1]-ends[i-1][1] for i in range(1,len(ends)) if ends[i][0]>=11]
 mem=list(csv.DictReader(open(meta/'gpu_memory_poll.csv'))) if (meta/'gpu_memory_poll.csv').exists() else []
 d=list(csv.DictReader(open(run/'spring_diagnostics.csv'))) if (run/'spring_diagnostics.csv').exists() else []
 E=[float(r['energy']) for r in tail];V=[float(r['variance']) for r in tail];A=[float(r['accept_ratio']) for r in tail]
 row={'system':system,'optimizer':opt,'mu':mu,'pretraining_epochs':5000,'source_checkpoint':('/scratch/dexuan1/runs/wssr_C_4096_protocol/C_KFAC4096_pre5000/checkpoints/5000.npz' if system=='C' else '/scratch/dexuan1/runs/fir_n4096_corrected_retry1/N2/kfac_preliminary/checkpoints/5000.npz'),'completed_epochs':len(m),'finite_epochs':len(fin),'status':'COMPLETED' if len(m)==50000 and len(fin)==50000 else 'PARTIAL_OR_FAILED','tail1000_energy_mean':statistics.mean(E) if E else '','tail1000_energy_sem':statistics.stdev(E)/len(E)**.5 if len(E)>1 else '','tail1000_variance_mean':statistics.mean(V) if V else '','median_acceptance':statistics.median(A) if A else '','median_sec_epoch':statistics.median(dt) if dt else '','total_wall_seconds':ends[-1][1]-ends[0][1] if len(ends)>1 else '','peak_gpu_mib':max((float(r['memory_used_mib']) for r in mem),default=''),'final_energy':fin[-1]['energy'] if fin else '','final_variance':fin[-1]['variance'] if fin else '','final_checkpoint':str(run/'checkpoints/50000.npz') if (run/'checkpoints/50000.npz').exists() else ''}
 if opt=='SPRING' and d:
  def nums(k): return [float(x[k]) for x in d if x.get(k,'') not in ('','nan','NaN') and math.isfinite(float(x[k]))]
  hp=nums('spring_diag_history_projection_norm');ratio=nums('spring_diag_mu_history_over_nonhistory');scale=nums('spring_diag_norm_constraint_scale');hist=nums('spring_diag_history_norm')
  row.update(max_history_projection=max(hp,default=''),max_history_nonhistory_ratio=max(ratio,default=''),min_norm_constraint_scale=min(scale,default=''),sustained_history_growth=any(all(hist[j]>hist[j-1] for j in range(i+1,i+6)) for i in range(max(0,len(hist)-5))) if hist else '')
 rows.append(row)
fields=sorted({k for r in rows for k in r});
with (OUT/'final_summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
(OUT/'final_summary.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
