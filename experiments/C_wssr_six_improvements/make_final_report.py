#!/usr/bin/env python3
import csv
import json
import subprocess
from pathlib import Path

import numpy as np


X=Path('/scratch/dexuan1/vmcnet/experiments/C_wssr_six_improvements')
s1=list(csv.DictReader(open(X/'results/stage1/aggregate.csv')))
sel=json.load(open(X/'results/stage1/selection.json'))
s2=list(csv.DictReader(open(X/'results/stage2/summary.csv')))
s3=list(csv.DictReader(open(X/'results/stage3/cross_seed.csv')))
conc=json.load(open(X/'results/stage3/conclusions.json'))
num=lambda x:float(x)


def selected(method):
    return next(x for x in s1 if x['method']==method and x['parameter']==str(sel[method]['parameter']))


def stage2(method):
    return next(x for x in s2 if x['method']==method)


all_improvements=[x for x in s1 if x['method'] not in ('baseline','fixed_complement_reference')]
best_numerical=min(all_improvements,key=lambda x:num(x['mean_full_error']))
best_eligible=min((x for x in all_improvements if x['eligible']=='True'),key=lambda x:num(x['mean_full_error']))
cluster=selected('cluster');tail=selected('near_tail');adapt=selected('adaptive_complement')
fixed=next(x for x in s1 if x['method']=='fixed_complement_reference')
smooth2=stage2('smooth');force2=stage2('force_aware')
cg=[x for x in s1 if x['method']=='iterative_complement']


def variant(method,parameter):
    return f"{method}_{str(parameter).replace('+','_')}"


noise={}
for x in s2:
    run=Path('/scratch/dexuan1/runs/C_wssr6_stage2_E200')/variant(x['method'],x['parameter'])
    rows=list(csv.DictReader(open(run/'training_metrics.csv')))[-50:]
    raw=np.asarray([float(r['variance_noclip']) for r in rows])
    noise[x['method']]=dict(std=float(raw.std(ddof=1)),diff_std=float(np.diff(raw).std(ddof=1)))


stable3=[x for x in s3 if x['all_stable']=='True' and x['method']!='baseline']
n2_ranked=sorted(stable3,key=lambda x:(num(x['frozen_raw_variance_mean']),num(x['median_seconds_epoch_mean'])))[:2]
job_lines=[]
for name in ('stage2_job_ids.txt','stage3_job_ids.txt'):
    p=X/name
    if p.exists():job_lines += [f'- `{line}`' for line in p.read_text().splitlines() if line]
changed=subprocess.check_output(['git','status','--short'],cwd='/scratch/dexuan1/vmcnet',text=True).splitlines()
relevant=[line for line in changed if ('wssr' in line.lower() or 'C_wssr_six_improvements' in line)]
cg_ratios=', '.join(f"{num(x['max_complement_ratio']):.3g}" for x in cg)
n2_recommend=', '.join(f"**{x['method']} ({x['parameter']})**" for x in n2_ranked)


lines=['# C WSSR six-improvement independent ablation','',
'All training uses all-electron C `(4,2)`, `nchains=1000`, and the common KFAC-pre1000 checkpoint. No N2 job was run. Absolute C energies are not interpreted physically because the existing reference inconsistency remains unresolved.','',
'## Direct answers','',
f"1. Largest full-reference error reduction: **projected iterative complement, 10 iterations**, mean error {num(best_numerical['mean_full_error']):.6g}; it is **training-ineligible** because complement/resolved reached {num(best_numerical['max_complement_ratio']):.3g} > 0.5. Best eligible method is **{best_eligible['method']} ({best_eligible['parameter']})**, error {num(best_eligible['mean_full_error']):.6g}.",
f"2. Lowest frozen raw variance: **{conc['best_frozen_variance']['method']}**, {conc['best_frozen_variance']['frozen_raw_variance_mean']:.6g} ± {conc['best_frozen_variance']['frozen_raw_variance_std']:.3g} across seeds.",
f"3. Best measured speed–variance tradeoff: **{conc['best_speed_variance_tradeoff']['method']}**.",
f"4. Cluster-aware versus fixed near-tail: mean full errors {num(cluster['mean_full_error']):.6g} versus {num(tail['mean_full_error']):.6g}; {'cluster-aware wins' if num(cluster['mean_full_error'])<num(tail['mean_full_error']) else 'fixed near-tail wins'} numerically.",
f"5. Adaptive complement versus fixed `1e-4`: {num(adapt['mean_full_error']):.6g} versus {num(fixed['mean_full_error']):.6g}. The selected adaptive beta reaches the nominal cap and therefore {'improves on' if num(adapt['mean_full_error'])<num(fixed['mean_full_error']) else 'does not improve on'} the fixed case in Stage 1.",
f"6. Smooth-cutoff E200 is {smooth2['status']}; final-50 raw-variance SD={noise['smooth']['std']:.6g} and first-difference SD={noise['smooth']['diff_std']:.6g}, compared with baseline {noise['baseline']['std']:.6g}/{noise['baseline']['diff_std']:.6g}.",
f"7. Force-aware is {force2['status']}; its four-snapshot mean full error is {num(selected('force_aware')['mean_full_error']):.6g}, versus baseline {num(next(x for x in s1 if x['method']=='baseline')['mean_full_error']):.6g}. Injected-vector norms and overlaps are in Stage-1 raw data.",
f"8. Projected iterative complement greatly improves numerical fidelity ({num(cg[-1]['mean_full_error']):.6g} at 10 iterations) but all variants violate the 0.5 contribution gate ({cg_ratios}); it does not justify training in its present isolated form.",
f"9. Recommended next N2/4096 methods: {n2_recommend}. This is a recommendation only; no N2 job was submitted.",'',
'## Stage 1 selections','', '```json',json.dumps(sel,indent=2),'```','',
'The planned seventh E200 method, projected iterative complement, was excluded by the user-specified eligibility gate. E200 therefore contains baseline plus five eligible independent improvements.','',
'## Stage 2','', '|method|parameter|status|final-50 raw variance|mean full error|s/epoch|peak MiB|','|---|---|---|---:|---:|---:|---:|']
for x in s2:
    lines.append(f"|{x['method']}|{x['parameter']}|{x['status']}|{num(x['tail50_raw_variance']):.6g}|{num(x['mean_full_error']):.6g}|{num(x['median_seconds_epoch']):.4g}|{num(x['peak_gpu_memory_mib']):.0f}|")
lines += ['', '## Stage 3 cross-seed frozen evaluation','', '|method|parameter|frozen energy mean±seed SD|raw variance mean±seed SD|s/epoch mean±seed SD|','|---|---|---:|---:|---:|']
for x in s3:
    lines.append(f"|{x['method']}|{x['parameter']}|{num(x['frozen_energy_mean']):.8f} ± {num(x['frozen_energy_std']):.3g}|{num(x['frozen_raw_variance_mean']):.6g} ± {num(x['frozen_raw_variance_std']):.3g}|{num(x['median_seconds_epoch_mean']):.4g} ± {num(x['median_seconds_epoch_std']):.2g}|")
lines += ['', '## Jobs and commands',''] + job_lines + ['',
'Exact commands are the versioned local scripts `stage1_array.sh`, `stage2_array.sh`, `stage2_accuracy_array.sh`, `stage3_array.sh`, and `stage3_frozen_eval.sh`. Stage-1 authoritative jobs were `49918208_[0-3]`; invariance tests were `49922369` (15 passed). Pre-correction jobs are preserved but excluded.','',
'## Relevant local changes',''] + [f'- `{x}`' for x in relevant] + ['',
'WSSR update semantics were preserved when all new options are disabled; every experimental option is default-disabled. Nothing was committed or pushed.','',
'## Artifacts','',
'- Stage 1: `results/stage1/all_results.csv`, `aggregate.csv`, `report.md`','- Stage 2: `results/stage2/summary.csv`','- Stage 3: `results/stage3/per_seed.csv`, `cross_seed.csv`, `conclusions.json`','- Plots: `results/plots/*.svg`','']
(X/'results/final_report.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
