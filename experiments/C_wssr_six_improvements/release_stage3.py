#!/usr/bin/env python3
import json,subprocess
from pathlib import Path
R=Path('/scratch/dexuan1/vmcnet/experiments/C_wssr_six_improvements');x=json.loads((R/'results/stage2/top3.json').read_text());
if len(x)!=3 or not all(z['status']=='STABLE' for z in x):raise SystemExit('Stage3 requires three stable improvements')
j=subprocess.check_output(['sbatch','--parsable',str(R/'stage3_array.sh')],text=True).strip();e=subprocess.check_output(['sbatch','--parsable',f'--dependency=afterok:{j}',str(R/'stage3_frozen_eval.sh')],text=True).strip();c=subprocess.check_output(['sbatch','--parsable',f'--dependency=afterany:{e}',str(R/'collect_stage3.sh')],text=True).strip();(R/'stage3_job_ids.txt').write_text(f'stage3_array={j}\nfrozen_eval={e}\ncollector={c}\n');print(j,e,c)
