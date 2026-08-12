#!/usr/bin/env python3
import json,subprocess
from pathlib import Path
R=Path('/scratch/dexuan1/vmcnet/experiments/C_wssr_six_improvements');s=json.loads((R/'results/stage1/selection.json').read_text());required={'cluster','near_tail','adaptive_complement','smooth','force_aware'}
if not required.issubset(s):raise SystemExit(f'missing eligible Stage1 selections: {required-set(s)}')
methods=json.loads((R/'results/stage1/stage2_methods.json').read_text());n=len(methods)
j=subprocess.check_output(['sbatch','--parsable',f'--array=0-{n-1}%4',str(R/'stage2_array.sh')],text=True).strip()
acc=subprocess.check_output(['sbatch','--parsable',f'--array=0-{n-1}%4',f'--dependency=afterok:{j}',str(R/'stage2_accuracy_array.sh')],text=True).strip()
collect=subprocess.check_output(['sbatch','--parsable',f'--dependency=afterany:{acc}',str(R/'collect_stage2.sh')],text=True).strip()
release3=subprocess.check_output(['sbatch','--parsable',f'--dependency=afterok:{collect}',str(R/'release_stage3.sh')],text=True).strip()
(R/'stage2_job_ids.txt').write_text(f'stage2_array={j}\nstage2_accuracy={acc}\nstage2_collector={collect}\nstage3_release={release3}\n');print(j,acc,collect,release3)
