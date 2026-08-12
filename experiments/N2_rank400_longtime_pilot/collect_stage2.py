#!/usr/bin/env python3
import csv,json,math,statistics,subprocess
from pathlib import Path
P=Path('/scratch/dexuan1/vmcnet/experiments/N2_rank400_longtime_pilot');ROOT=Path('/scratch/dexuan1/runs/N2_rank400_longtime_pilot_E500');OUT=P/'results';OUT.mkdir(exist_ok=True)
names=['A_beta02_control','B_beta01_fixed','C_beta_decay','D_residual_optimal','E_capped_near_tail','F_rank600_hard'];released=json.loads((OUT/'release_stage2.json').read_text())['eligible_tasks'];rows=[];stable=[]
def vals(p):return [float(x) for x in p.read_text().split()]
for i in released:
 n=names[i];r=ROOT/n;m=Path(str(r)+'.metadata');d=list(csv.DictReader(open(r/'training_metrics.csv')));num=['energy','energy_noclip','variance','variance_noclip','accept_ratio'];finite=len(d)==500 and all(math.isfinite(float(x[k])) for x in d for k in num);early=m/'early_stop.json';status='STABLE' if finite and not early.exists() else 'FAILED'
 if status=='STABLE':stable.append(i)
 tail=d[-100:];tim=list(csv.DictReader(open(m/'phase_timing.csv')));te=[float(x['monotonic_seconds']) for x in tim if x['event']=='epoch_end'];sec=[b-a for a,b in zip(te,te[1:])][10:];gpu=list(csv.DictReader(open(m/'gpu_memory_poll.csv')))
 def diag(fn,default=0.):
  p=r/fn;return vals(p) if p.exists() else [default]*len(d)
 ratio=diag('wssr_diag_complement_resolved_ratio.txt');scale=diag('wssr_diag_norm_constraint_scale.txt',1.);cap=diag('wssr_diag_adaptive_cap_active.txt');rb=diag('wssr_diag_residual_before.txt');ra=diag('wssr_diag_residual_after.txt')
 rows.append(dict(task=i,variant=n,completed_epochs=len(d),status=status,tail100_energy=statistics.mean(float(x['energy_noclip']) for x in tail),tail100_raw_variance=statistics.mean(float(x['variance_noclip']) for x in tail),tail100_clipped_variance=statistics.mean(float(x['variance']) for x in tail),acceptance=statistics.median(float(x['accept_ratio']) for x in tail),median_seconds_epoch=statistics.median(sec),peak_gpu_mib=max(float(x['memory_used_mib']) for x in gpu),ratio_mean=statistics.mean(ratio),ratio_max=max(ratio),cap_active_fraction=statistics.mean(cap),constraint_scale_min=min(scale),residual_before_mean=statistics.mean(rb),residual_after_mean=statistics.mean(ra),rolling_raw_clipped_ratio=statistics.mean(float(x['variance_noclip'])/max(float(x['variance']),1e-30) for x in tail),training_tail_quantiles='not available per epoch; frozen evaluation supplies sample quantiles'))
with (OUT/'stage2_summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
if not stable:raise RuntimeError('no stable E500 variants')
spec=','.join(map(str,stable));ev=subprocess.check_output(['sbatch','--parsable',f'--array={spec}%4',str(P/'frozen_eval_array.sh')],text=True).strip();col=subprocess.check_output(['sbatch','--parsable',f'--dependency=afterany:{ev}',str(P/'collect_final.sh')],text=True).strip();release={'stable_tasks':stable,'frozen_array_job_id':ev,'final_collector_job_id':col};(OUT/'release_frozen.json').write_text(json.dumps(release,indent=2)+'\n');print(json.dumps(release,indent=2))
