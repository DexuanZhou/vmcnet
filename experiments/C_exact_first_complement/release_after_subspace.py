#!/usr/bin/env python3
import json, subprocess
from pathlib import Path

ROOT=Path('/scratch/dexuan1/vmcnet/experiments/C_exact_first_complement')
RESULTS=ROOT/'results'
replay=json.loads((RESULTS/'direct_subspace_replay.json').read_text())
rank={x['rank']:x for x in replay['rank_comparisons']}
base=Path('/scratch/dexuan1/runs/C_exact_first_complement/stage1_10_retry1')
def first(run,key):return float((base/run/f'{key}.txt').read_text().split()[0])
update_ok=all(first(v,'wssr_diag_update_relative_error')<1e-4 and first(v,'wssr_diag_update_cosine')>.99999 for v in ('B_exact_c0','C_exact_c1e4'))
gaps={x['index_1based']:x['relative_gap_to_next'] for x in replay['singular_values_395_405']}
clustered=gaps[400] < 1e-4
same_space=(rank[400]['projector_relative_error']<1e-3 or (clustered and rank[394]['projector_relative_error']<1e-3 and rank[405]['projector_relative_error']<1e-3))
decision={'update_ok':update_ok,'rank400':rank[400],'cutoff_clustered':clustered,'same_space_consistent':same_space}
(RESULTS/'release_decision.json').write_text(json.dumps(decision,indent=2))
print(json.dumps(decision,indent=2))
if not (update_ok and same_space):
    raise SystemExit('Stage 2 blocked by direct same-space gate')
array=subprocess.check_output(['sbatch','--parsable',str(ROOT/'scripts/stage2_array.sh')],text=True).strip()
collector=subprocess.check_output(['sbatch','--parsable',f'--dependency=afterany:{array}',str(ROOT/'scripts/collect_stage2.sh')],text=True).strip()
evaluation=subprocess.check_output(['sbatch','--parsable',f'--dependency=afterok:{array}',str(ROOT/'scripts/frozen_eval_array.sh')],text=True).strip()
text=f'stage2_array={array}\nstage2_collector={collector}\nfrozen_eval_array={evaluation}\n'
(RESULTS/'stage2_job_ids.txt').write_text(text);print(text,end='')
