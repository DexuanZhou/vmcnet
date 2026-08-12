#!/usr/bin/env python3
import csv,math,pathlib,statistics
HERE=pathlib.Path(__file__).resolve().parent; ROOT=pathlib.Path('/scratch/dexuan1/runs/fir_n4096_corrected_retry1/N2'); OUT=HERE/'results'
def read(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def n(x):
 try:return float(x)
 except:return math.nan
rows=[]
for mu,label in [('0.90','mu090'),('0.95','mu095')]:
 name=f'N2eq_R2068_spring_lr0005_{label}_n4096_e200'; run=ROOT/'spring_mu_stability'/name; meta=ROOT/'metadata'/name
 m=read(run/'training_metrics.csv') if (run/'training_metrics.csv').exists() else []; finite=[all(math.isfinite(n(r.get(k))) for k in ('energy','variance','accept_ratio')) for r in m]
 ev=read(meta/'phase_timing.csv') if (meta/'phase_timing.csv').exists() else []; st={int(r['epoch']):n(r['monotonic_seconds']) for r in ev if r['event']=='epoch_end'}; sec=[st[e]-st[e-1] for e in range(11,min(200,len(m))+1) if e in st and e-1 in st]
 energies=[n(r.get('energy')) for r in m]; var=[n(r.get('variance')) for r in m]; explosive=any(statistics.fmean(var[i:i+10])>100 for i in range(max(0,len(var)-9)))
 stable=len(m)==200 and all(finite) and all(abs(x+109.48)<2 for x in energies) and not explosive
 row={'mu':mu,'learning_rate':0.0005,'finite_epochs':sum(finite),'first_nonfinite_epoch':'' if len(m)==200 and all(finite) else len(m)+1,'steady_median_sec_per_epoch':statistics.median(sec) if sec else math.nan,'status':'STABLE' if stable else 'NONFINITE'}
 for e in (1,10,50,100,200):
  r=m[e-1] if len(m)>=e else {}; row[f'energy_{e}']=n(r.get('energy')); row[f'variance_{e}']=n(r.get('variance')); row[f'acceptance_{e}']=n(r.get('accept_ratio'))
 rows.append(row)
with (OUT/'n2_spring_n4096_mu_stability.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
lines=['# N2 SPRING mu stability (lr=0.0005)','', '| mu | finite | first nonfinite | E@1/10/50/100/200 | Var@1/10/50/100/200 | median s/epoch | status |','|---:|---:|---:|---|---|---:|---|']
for r in rows:
 fmt=lambda x:f'{x:.4f}' if math.isfinite(x) else 'nan'; es='/'.join(fmt(r[f'energy_{e}']) for e in (1,10,50,100,200)); vs='/'.join(fmt(r[f'variance_{e}']) for e in (1,10,50,100,200)); lines.append(f"| {r['mu']} | {r['finite_epochs']} | {r['first_nonfinite_epoch']} | {es} | {vs} | {r['steady_median_sec_per_epoch']:.3f} | {r['status']} |")
(OUT/'n2_spring_n4096_mu_stability.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
