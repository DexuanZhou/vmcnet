#!/usr/bin/env python3
import csv
import math
import subprocess
from pathlib import Path

BASE = Path('/scratch/dexuan1/runs/C_exact_first_complement/stage1_10_retry1')
OUT = Path('/scratch/dexuan1/vmcnet/experiments/C_exact_first_complement/results')
OUT.mkdir(parents=True, exist_ok=True)

def number(row, key):
    try: return float(row[key])
    except (KeyError, TypeError, ValueError): return math.nan

def diagnostic(run, key):
    path = run / f'{key}.txt'
    return [float(x) for x in path.read_text().split()] if path.exists() else []

rows = []
for run in sorted(BASE.glob('*')):
    if run.name.endswith('.metadata'):
        continue
    path = run / 'training_metrics.csv'
    if not path.exists():
        rows.append({'variant': run.name, 'status': 'MISSING_METRICS'}); continue
    with path.open() as f: data = list(csv.DictReader(f))
    numeric = [number(r, k) for r in data for k in r if k != 'epoch']
    finite = all(math.isfinite(v) for v in numeric)
    exact_flags = diagnostic(run, 'wssr_diag_exact_initialization_used')
    exact_count = round(sum(exact_flags)) if exact_flags else -1
    def vals(key): return diagnostic(run, key)
    row = dict(
        variant=run.name, epochs=len(data), finite=finite,
        exact_initialization_count=exact_count,
        max_update_relative_error=max(vals('wssr_diag_update_relative_error')),
        min_update_cosine=min(vals('wssr_diag_update_cosine')),
        max_principal_angle_rad=max(vals('wssr_diag_max_principal_angle')),
        max_subspace_residual=max(vals('wssr_diag_subspace_residual')),
        max_rank_sv_relative_error=max(vals('wssr_diag_singular_rank_relerr')),
        max_complement_resolved_ratio=max(vals('wssr_diag_complement_resolved_ratio')),
        max_complement_total_ratio=max(vals('wssr_diag_complement_total_ratio')),
        min_constraint_scale=min(vals('wssr_diag_norm_constraint_scale')),
        status='PASS' if finite and len(data) == 10 else 'FAIL',
    )
    if run.name.startswith(('B_', 'C_', 'D_')):
        update_ok=(vals('wssr_diag_update_relative_error')[0] < 1e-4 and vals('wssr_diag_update_cosine')[0] > .99999)
        row['first_step_update_ok']=update_ok
        if exact_count != 1 or not update_ok: row['status']='FAIL'
    if run.name.startswith(('C_', 'D_')) and row['max_complement_total_ratio'] > 10:
        row['status']='PATHOLOGICAL_COMPLEMENT'
    if run.name.startswith('D_'):
        row['status']='EXCLUDED_STRESS_COMPLEMENT'
    rows.append(row)

fields=sorted({k for r in rows for k in r})
with (OUT/'stage1_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
with (OUT/'stage1_summary.md').open('w') as f:
    f.write('| '+' | '.join(fields)+' |\n|'+('|---'*len(fields))+'|\n')
    for row in rows: f.write('| '+' | '.join(str(row.get(k,'')) for k in fields)+' |\n')
print('\n'.join(str(r) for r in rows))
print('Collector only summarizes Stage 1; downstream release is gated by the direct same-space replay.')
