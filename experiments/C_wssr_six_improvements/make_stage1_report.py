#!/usr/bin/env python3
import csv
import json
from pathlib import Path

root=Path('/scratch/dexuan1/vmcnet/experiments/C_wssr_six_improvements')
r=root/'results/stage1'
agg=list(csv.DictReader(open(r/'aggregate.csv')))
sel=json.load(open(r/'selection.json'))
lines=['# Stage 1 — fixed-snapshot numerical screening','',
'All four C `(4,2)`, `nchains=1000` snapshots were replayed without moving walkers or applying an update. The primary reference is the full Tikhonov-regularized inverse on the identical frozen augmented operator. Oversampled late-state methods retain the saved rank-400 right-warm information and add a reproducible checkpoint-key orthogonal completion before normal q=2 SSI.','',
'Jobs: authoritative array `49918208` (all four tasks completed); invariance tests `49922369` (15 passed). Earlier jobs `49913264_0`, `49915718_0`, and `49913264_1` were failed/cancelled infrastructure or pre-correction attempts and are excluded.','',
'## Selected internal settings','', '```json',json.dumps(sel,indent=2),'```','',
'Projected iterative complement has no training-eligible setting: its maximum complement/resolved ratios are 8.17, 10.78 and 14.26 for 3, 5 and 10 iterations, all above the predeclared 0.5 gate.','',
'## Aggregate over four snapshots','',
'|method|parameter|eligible|mean full error|max full error|mean cosine|max comp/resolved|mean effective rank|','|---|---:|:---:|---:|---:|---:|---:|---:|']
for x in agg:
 lines.append(f"|{x['method']}|{x['parameter']}|{x['eligible']}|{float(x['mean_full_error']):.6f}|{float(x['max_full_error']):.6f}|{float(x['mean_cosine']):.5f}|{float(x['max_complement_ratio']):.4g}|{float(x['mean_effective_rank']):.1f}|")
lines += ['', 'Full per-snapshot results, constrained errors, residuals, principal angles, spectra, runtime and memory are in `all_results.csv`; the exact spectra are in each snapshot `metadata.json`.','']
(r/'report.md').write_text('\n'.join(lines))
print(r/'report.md')
