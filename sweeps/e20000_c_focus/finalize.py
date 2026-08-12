#!/usr/bin/env python3
"""Render the focused C E20000 completed-run comparison."""
import csv, math
from pathlib import Path

HERE = Path(__file__).resolve().parent
SWEEPS = HERE.parent
SUMMARY = SWEEPS / "results/e20000_c_focus/summary.csv"
SPRING = Path("/scratch/dexuan1/runs/e40000/C/spring_lr02")
REF = -37.84471

def mean_sem(values):
    mean = sum(values) / len(values)
    var = sum((x-mean)**2 for x in values)/(len(values)-1)
    return mean, (var/len(values))**0.5

def spring_row():
    energies=[float(x) for x in (SPRING/'energy.txt').read_text().splitlines()]
    metrics=list(csv.DictReader((SPRING/'training_metrics.csv').open()))
    t=energies[-1000:]; mean,sem=mean_sem(t)
    var=[float(r['variance']) for r in metrics[-1000:]]
    acc=[float(r['accept_ratio']) for r in metrics[-1000:]]
    return mean,sem,(mean-REF)*1000,sum(var)/len(var),sum(acc)/len(acc)

def fmt(x, spec):
    try:
        x=float(x)
        return format(x,spec) if math.isfinite(x) else '—'
    except: return '—'

def main():
    rows=list(csv.DictReader(SUMMARY.open()))
    valid=[r for r in rows if r['status']=='complete']
    valid.sort(key=lambda r:float(r['tail1000_error_mha']))
    sm,ss,se,sv,sa=spring_row()
    lines=['# Focused C WSSR E20000 results','',
           '| Rank | Run | Status | Finite epochs | First nonfinite | Tail-500 mean ± SEM | Tail-1000 mean ± SEM | Tail-1000 error (mHa) | Variance tail | Acceptance tail | GPU MiB | Wall time (s) | Active rank | Checkpoint |',
           '|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    ranks={r['run_name']:i+1 for i,r in enumerate(valid)}
    for r in sorted(rows,key=lambda x:(x['status']!='complete',float(x['tail1000_error_mha']) if x['tail1000_error_mha']!='nan' else math.inf)):
        lines.append(f"| {ranks.get(r['run_name'],'—')} | `{r['run_name']}` | {r['status']} | {r['finite_energy_epochs']} | {r['first_nonfinite_epoch'] or '—'} | {fmt(r['tail500_energy_mean'],'.9f')} ± {fmt(r['tail500_energy_sem'],'.2e')} | {fmt(r['tail1000_energy_mean'],'.9f')} ± {fmt(r['tail1000_energy_sem'],'.2e')} | {fmt(r['tail1000_error_mha'],'.3f')} | {fmt(r['variance_tail1000_mean'],'.5f')} | {fmt(r['acceptance_tail1000_mean'],'.5f')} | {fmt(r['peak_gpu_memory_mib'],'.0f')} | {fmt(r['elapsed_seconds'],'.0f')} | {fmt(r['active_rank'],'.0f')} | {r['checkpoint_present']} |")
    lines += ['', '## Existing C SPRING baseline', '',
              f'Tail-1000: {sm:.9f} ± {ss:.2e} Ha; error {se:.3f} mHa; variance tail {sv:.5f}; acceptance tail {sa:.5f}.', '']
    if valid:
        names=', '.join(f'`{r["run_name"]}`' for r in valid[:2])
        lines += ['## E100000 candidates', '', f'Recommend the top {min(2,len(valid))} finite settings by tail-1000 error for E100000 + inference: {names}.', '']
    else:
        lines += ['## E100000 candidates', '', 'No completed finite WSSR run; no recommendation.', '']
    (HERE/'final_summary.md').write_text('\n'.join(lines))

if __name__=='__main__': main()
