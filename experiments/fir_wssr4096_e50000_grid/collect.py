#!/usr/bin/env python3
import csv,json,math,pathlib,statistics
out=pathlib.Path(__file__).parent/'results';out.mkdir(exist_ok=True)
for sys in ('C','N2'):
 root=pathlib.Path(f'/scratch/dexuan1/runs/wssr_{sys}_4096_kfac1000_E50000_grid'); rows=[]
 for run in sorted(p for p in root.iterdir() if p.is_dir() and p.name!='metadata'):
  m=list(csv.DictReader(open(run/'training_metrics.csv'))); fin=[r for r in m if all(math.isfinite(float(r[k])) for k in ('energy','variance','accept_ratio'))]; tail=fin[-1000:]
  tim=list(csv.DictReader(open(root/'metadata'/run.name/'phase_timing.csv'))); ends=[(int(r['epoch']),float(r['monotonic_seconds'])) for r in tim if r['event']=='epoch_end'];dt=[ends[i][1]-ends[i-1][1] for i in range(1,len(ends)) if ends[i][0]>=11]
  mem=list(csv.DictReader(open(root/'metadata'/run.name/'gpu_memory_poll.csv'))); cfg=json.load(open(run/'config.json'));o=cfg['vmc']['optimizer']['wssr_warm_svd_right']; early=json.load(open(root/'metadata'/run.name/'early_stop.json')) if (root/'metadata'/run.name/'early_stop.json').exists() else {}
  E=[float(r['energy']) for r in tail];V=[float(r['variance']) for r in tail]; final=fin[-1] if fin else {}
  rows.append({'system':sys,'run':run.name,'rank':o['sr_rank'],'warm':o['svd_maxiter_warm'],'completed_epochs':len(m),'finite_epochs':len(fin),'termination_status':'COMPLETED' if len(m)==50000 and len(fin)==50000 else early.get('reason','PARTIAL'),'tail1000_energy_mean':statistics.mean(E) if E else '', 'tail1000_energy_sem':statistics.stdev(E)/math.sqrt(len(E)) if len(E)>1 else '','tail1000_variance_mean':statistics.mean(V) if V else '','median_acceptance':statistics.median(float(r['accept_ratio']) for r in tail) if tail else '','median_sec_epoch':statistics.median(dt) if dt else '','total_wall_seconds':ends[-1][1]-ends[0][1] if len(ends)>1 else '','peak_gpu_mib':max((float(r['memory_used_mib']) for r in mem),default=''),'final_energy':final.get('energy',''),'final_variance':final.get('variance',''),'final_checkpoint':str(run/'checkpoints/50000.npz') if (run/'checkpoints/50000.npz').exists() else ''})
 with open(out/f'{sys}_summary.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(sys);[print(r) for r in rows]
