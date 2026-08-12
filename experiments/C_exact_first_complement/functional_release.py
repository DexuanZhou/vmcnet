#!/usr/bin/env python3
import json, subprocess
from pathlib import Path
ROOT=Path('/scratch/dexuan1/vmcnet/experiments/C_exact_first_complement');R=ROOT/'results'
x=json.loads((R/'functional_warm_replay.json').read_text());a=x['baseline_A'];e=x['exact_first']
passed=(e['update_relative_error']<1e-2 and e['update_cosine']>.9999 and e['subspace_residual']<.05 and e['finite_update'] and e['finite_state'] and e['update_relative_error']<a['update_relative_error'])
decision={'passed':passed,'criteria':{'relative_error_lt':1e-2,'cosine_gt':.9999,'subspace_residual_lt':.05,'finite':True,'exact_materially_better_than_baseline':True},'baseline_A':a,'exact_first':e}
(R/'functional_release_decision.json').write_text(json.dumps(decision,indent=2));print(json.dumps(decision,indent=2))
if not passed:raise SystemExit('Stage 2 blocked by functional warm-state gate')
array=subprocess.check_output(['sbatch','--parsable',str(ROOT/'scripts/stage2_array.sh')],text=True).strip()
collector=subprocess.check_output(['sbatch','--parsable',f'--dependency=afterany:{array}',str(ROOT/'scripts/collect_stage2.sh')],text=True).strip()
evaluation=subprocess.check_output(['sbatch','--parsable',f'--dependency=afterok:{array}',str(ROOT/'scripts/frozen_eval_array.sh')],text=True).strip()
text=f'stage2_array={array}\nstage2_collector={collector}\nfrozen_eval_array={evaluation}\n';(R/'stage2_job_ids.txt').write_text(text);print(text,end='')
