#!/usr/bin/env python3
import csv,json,math,pathlib,statistics,subprocess
BASE=pathlib.Path('/scratch/dexuan1/runs')
runs=[('C','MinSR',0.0,BASE/'C_minsr_after_kfac5000_E50000'),('C','SPRING',.99,BASE/'C_spring_mu099_after_kfac5000_E50000'),('N2','MinSR',0.0,BASE/'N2_minsr_after_kfac5000_E50000'),('N2','SPRING',.99,BASE/'N2_spring_mu099_after_kfac5000_E50000')]
out=pathlib.Path(__file__).parent/'results';out.mkdir(exist_ok=True)
rows=[]; stable=[]
for task,(system,opt,mu,longroot) in enumerate(runs):
 nominal=pathlib.Path(str(longroot)+'_smoke200')
 # The first submitted smoke used metadata inside the nominal directory, so
 # vmcnet correctly collision-avoided into the adjacent _1 directory.
 run=pathlib.Path(str(nominal)+'_1') if pathlib.Path(str(nominal)+'_1/training_metrics.csv').exists() else nominal
 m=list(csv.DictReader(open(run/'training_metrics.csv'))) if (run/'training_metrics.csv').exists() else []
 finite=[r for r in m if all(math.isfinite(float(r[k])) for k in ('energy','variance','accept_ratio'))]
 early_path=nominal/'metadata/early_stop.json'
 early=json.load(open(early_path)) if early_path.exists() else {}
 ok=len(m)==200 and len(finite)==200 and not early.get('triggered',False)
 source='/scratch/dexuan1/runs/wssr_C_4096_protocol/C_KFAC4096_pre5000/checkpoints/5000.npz' if system=='C' else '/scratch/dexuan1/runs/fir_n4096_corrected_retry1/N2/kfac_preliminary/checkpoints/5000.npz'
 rows.append({'task':task,'system':system,'optimizer':opt,'mu':mu,'pretraining_epochs':5000,'completed_epochs':len(m),'finite_epochs':len(finite),'status':'STABLE' if ok else early.get('reason','FAILED_OR_PARTIAL'),'source_checkpoint':source})
 if ok:stable.append(str(task))
with (out/'smoke_summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
submission={'stable_tasks':stable,'long_job_id':None}
if stable:
 cmd=['sbatch','--parsable',f'--array={",".join(stable)}%4','--export=ALL,PHASE=long',str(pathlib.Path(__file__).parent/'scripts/run_array.sh')]
 submission['command']=' '.join(cmd);submission['long_job_id']=subprocess.check_output(cmd,text=True).strip()
 finalcmd=['sbatch','--parsable',f'--dependency=afterany:{submission["long_job_id"].split(";")[0]}',str(pathlib.Path(__file__).parent/'scripts/final_collector.sh')]
 submission['final_collector_command']=' '.join(finalcmd);submission['final_collector_job_id']=subprocess.check_output(finalcmd,text=True).strip()
(out/'release.json').write_text(json.dumps(submission,indent=2)+'\n')
print(json.dumps({'rows':rows,**submission},indent=2))
