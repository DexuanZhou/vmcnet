#!/usr/bin/env python3
import csv,math,pathlib,statistics
HERE=pathlib.Path(__file__).resolve().parent
ROOT=pathlib.Path('/scratch/dexuan1/runs/fir_n4096_corrected_retry1/N2')
OUT=HERE/'results'; OUT.mkdir(exist_ok=True)
RUNS=[('0.0005','lr0005'),('0.0002','lr0002'),('0.0001','lr0001')]
def read(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def n(x):
    try:return float(x)
    except:return math.nan
rows=[]
for lr,label in RUNS:
    name=f'N2eq_R2068_spring_{label}_n4096_e200'; run=ROOT/'spring_stability'/name; meta=ROOT/'metadata'/name
    m=read(run/'training_metrics.csv') if (run/'training_metrics.csv').exists() else []
    finite=[all(math.isfinite(n(r.get(k))) for k in ('energy','variance','accept_ratio')) for r in m]
    first_bad=next((i+1 for i,x in enumerate(finite) if not x), '')
    if not first_bad and len(m)<200:first_bad=len(m)+1
    ev=read(meta/'phase_timing.csv') if (meta/'phase_timing.csv').exists() else []
    stamps={int(r['epoch']):n(r['monotonic_seconds']) for r in ev if r['event']=='epoch_end'}
    secs=[stamps[e]-stamps[e-1] for e in range(11,min(200,len(m))+1) if e in stamps and e-1 in stamps]
    med=statistics.median(secs) if secs else math.nan
    gpu=read(meta/'gpu_memory_poll.csv') if (meta/'gpu_memory_poll.csv').exists() else []
    used=[n(r.get('memory_used_mib')) for r in gpu]; used=[x for x in used if math.isfinite(x)]
    energies=[n(r.get('energy')) for r in m]; variances=[n(r.get('variance')) for r in m]
    normal=bool(energies) and all(abs(x+109.48)<2 for x in energies if math.isfinite(x))
    # Sustained explosion: any 10-epoch rolling mean above 100 Ha^2.
    explosive=any(statistics.fmean(variances[i:i+10])>100 for i in range(max(0,len(variances)-9)))
    stable=len(m)==200 and all(finite) and normal and not explosive
    row={'learning_rate':lr,'finite_epochs':sum(finite),'first_nonfinite_epoch':first_bad,'steady_median_sec_per_epoch':med,'max_gpu_memory_mib':max(used) if used else math.nan,'status':'STABLE' if stable else 'NONFINITE'}
    for ep in (1,10,50,100,200):
        r=m[ep-1] if len(m)>=ep else {}
        row[f'energy_epoch_{ep}']=n(r.get('energy')); row[f'variance_epoch_{ep}']=n(r.get('variance')); row[f'acceptance_epoch_{ep}']=n(r.get('accept_ratio'))
    rows.append(row)
fields=list(rows[0])
with (OUT/'n2_spring_n4096_stability_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
def f(x,d=4):return f'{x:.{d}f}' if isinstance(x,float) and math.isfinite(x) else str(x)
lines=['# N2 n=4096 SPRING stability sweep','', '> This 200-epoch sweep assesses stability and per-epoch cost only; it does not select a winner by energy.','', '| lr | finite | first nonfinite | E@1/10/50/100/200 | Var@1/10/50/100/200 | acceptance@200 | median s/epoch | max GPU MiB | status |','|---:|---:|---:|---|---|---:|---:|---:|---|']
for r in rows:
    es='/'.join(f(r[f'energy_epoch_{e}'],4) for e in (1,10,50,100,200)); vs='/'.join(f(r[f'variance_epoch_{e}'],3) for e in (1,10,50,100,200))
    lines.append(f"| {r['learning_rate']} | {r['finite_epochs']} | {r['first_nonfinite_epoch']} | {es} | {vs} | {f(r['acceptance_epoch_200'],4)} | {f(r['steady_median_sec_per_epoch'],3)} | {f(r['max_gpu_memory_mib'],0)} | {r['status']} |")
lines += ['', 'Timing comparison:', '', '| configuration | sec/epoch | ratio to SPRING lr=0.0005 |','|---|---:|---:|']
base=rows[0]['steady_median_sec_per_epoch']
for name,t in [('WSSR rank400/warm1',0.404),('WSSR rank800/warm2',0.520),('WSSR rank1600/warm2',0.770)]: lines.append(f'| {name} | {t:.3f} | {t/base:.3f} |')
(OUT/'n2_spring_n4096_stability_summary.md').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines))
