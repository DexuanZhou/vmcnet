#!/usr/bin/env python3
import csv,json,math,os,pathlib,statistics,subprocess
base=pathlib.Path('/scratch/dexuan1/runs'); out=pathlib.Path(__file__).parent/'results'; out.mkdir(exist_ok=True)
allrows=[]; winners={}
for sys in ('C','N2'):
 root=base/f'wssr_{sys}_4096_kfac1000_selection'; rows=[]
 for run in sorted(p for p in root.iterdir() if p.is_dir() and p.name!='metadata'):
  m=list(csv.DictReader(open(run/'training_metrics.csv'))); finite=[r for r in m if all(math.isfinite(float(r[k])) for k in ('energy','variance','accept_ratio'))]; tail=finite[-500:]
  vals=[float(r['variance']) for r in tail]; ens=[float(r['energy']) for r in tail]; blocks=[statistics.mean(vals[i:i+50]) for i in range(0,500,50)]
  times=list(csv.DictReader(open(root/'metadata'/run.name/'phase_timing.csv'))); ends=[(int(r['epoch']),float(r['monotonic_seconds'])) for r in times if r['event']=='epoch_end']; dt=[ends[i][1]-ends[i-1][1] for i in range(1,len(ends)) if ends[i][0]>=11]
  mem=list(csv.DictReader(open(root/'metadata'/run.name/'gpu_memory_poll.csv'))); cfg=json.load(open(run/'config.json')); o=cfg['vmc']['optimizer']['wssr_warm_svd_right']; early=json.load(open(root/'metadata'/run.name/'early_stop.json'))
  r={'system':sys,'run':run.name,'stable':len(finite)==len(m)==2000 and not early['triggered'],'variance_mean':statistics.mean(vals),'variance_block_sem':statistics.stdev(blocks)/math.sqrt(len(blocks)),'energy_mean':statistics.mean(ens),'energy_sem':statistics.stdev(ens)/math.sqrt(500),'acceptance_median':statistics.median(float(x['accept_ratio']) for x in tail),'median_sec_epoch':statistics.median(dt),'peak_gpu_mib':max(float(x['memory_used_mib']) for x in mem),'rank':o['sr_rank'],'eta':o['eta'],'lr':o['learning_rate'],'warm':o['svd_maxiter_warm']}; rows.append(r); allrows.append(r)
 stable=sorted((r for r in rows if r['stable']),key=lambda r:r['variance_mean']); assert stable
 best=stable[0]; tied=[r for r in stable if abs(r['variance_mean']-best['variance_mean']) <= 2*math.sqrt(r['variance_block_sem']**2+best['variance_block_sem']**2)]
 winners[sys]=min(tied,key=lambda r:r['median_sec_epoch'])
with open(out/'selection_summary.csv','w',newline='') as f: w=csv.DictWriter(f,fieldnames=allrows[0]);w.writeheader();w.writerows(allrows)
(out/'winners.json').write_text(json.dumps(winners,indent=2)+'\n')
jobs={}
for sys,w in winners.items():
 env=f"ALL,SYSTEM={sys},RANK={w['rank']},ETA={w['eta']},LR={w['lr']},WARM={w['warm']}"
 jobs[sys]=subprocess.check_output(['sbatch','--parsable',f'--export={env}',str(pathlib.Path(__file__).parent/'scripts/production.sh')],text=True).strip()
(out/'production_job_ids.json').write_text(json.dumps(jobs,indent=2)+'\n');print(json.dumps({'winners':winners,'production_jobs':jobs},indent=2))
