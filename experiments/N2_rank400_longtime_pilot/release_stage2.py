#!/usr/bin/env python3
import csv,json,subprocess
from pathlib import Path
P=Path('/scratch/dexuan1/vmcnet/experiments/N2_rank400_longtime_pilot');rows=list(csv.DictReader(open(P/'results/stage1/replay.csv')))
ref=next(r for r in rows if r['variant']=='F_rank600_hard');eligible=[];reasons={}
for i,r in enumerate(rows):
 reason=[]
 if r['finite']!='True':reason.append('nonfinite')
 if float(r['realized_ratio'])>.5:reason.append('complement_dominated')
 if float(r['runtime_seconds'])>float(ref['runtime_seconds']) and float(r['regularized_residual'])>float(ref['regularized_residual']) and float(r['rank800_relative_error'])>float(ref['rank800_relative_error']):reason.append('slower_and_worse_than_rank600_reference')
 reasons[r['variant']]=reason
 if not reason:eligible.append(i)
if not eligible:raise RuntimeError('no eligible variants')
spec=','.join(map(str,eligible));job=subprocess.check_output(['sbatch','--parsable',f'--array={spec}%4',str(P/'stage2_array.sh')],text=True).strip()
collector=subprocess.check_output(['sbatch','--parsable',f'--dependency=afterany:{job}',str(P/'collect_stage2.sh')],text=True).strip()
out={'eligible_tasks':eligible,'eligible_array_spec':spec,'exclusions':reasons,'stage2_array_job_id':job,'stage2_collector_job_id':collector}
(P/'results/release_stage2.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
