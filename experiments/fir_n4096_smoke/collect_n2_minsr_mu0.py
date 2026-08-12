#!/usr/bin/env python3
import csv,math,pathlib,statistics
ROOT=pathlib.Path('/scratch/dexuan1/runs/fir_n4096_corrected_retry1/N2')
NAME='N2eq_R2068_spring_lr0005_minsr_mu0_n4096_e200'
RUN=ROOT/'spring_mu_stability'/NAME; META=ROOT/'metadata'/NAME
OUT=pathlib.Path('/scratch/dexuan1/vmcnet/experiments/fir_n4096_smoke/results')
def read(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def n(x):
 try:return float(x)
 except:return math.nan
m=read(RUN/'training_metrics.csv') if (RUN/'training_metrics.csv').exists() else []
finite=[all(math.isfinite(n(r.get(k))) for k in ('energy','variance','accept_ratio')) for r in m]
ev=read(META/'phase_timing.csv') if (META/'phase_timing.csv').exists() else []
st={int(r['epoch']):n(r['monotonic_seconds']) for r in ev if r['event']=='epoch_end'}
secs=[st[e]-st[e-1] for e in range(11,min(200,len(m))+1) if e in st and e-1 in st]
energies=[n(r.get('energy')) for r in m]; variances=[n(r.get('variance')) for r in m]
explosive=any(statistics.fmean(variances[i:i+10])>100 for i in range(max(0,len(variances)-9)))
stable=len(m)==200 and all(finite) and all(abs(x+109.48)<2 for x in energies) and not explosive
first_nonfinite='' if len(m)==200 and all(finite) else len(m)+1
row={'method':'MinSR','mu':0.0,'learning_rate':0.0005,'kfac_start_epochs':5000,'finite_epochs':sum(finite),'first_nonfinite_epoch':first_nonfinite,'steady_median_sec_per_epoch':statistics.median(secs) if secs else math.nan,'status':'STABLE' if stable else 'NONFINITE'}
for e in (1,10,50,100,200):
 r=m[e-1] if len(m)>=e else {}; row[f'energy_{e}']=n(r.get('energy')); row[f'variance_{e}']=n(r.get('variance')); row[f'acceptance_{e}']=n(r.get('accept_ratio'))
with (OUT/'n2_minsr_mu0_n4096_summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=row);w.writeheader();w.writerow(row)
fmt=lambda x:f'{x:.6f}' if isinstance(x,float) and math.isfinite(x) else str(x)
lines=['# N2 n=4096 MinSR (SPRING mu=0)','',f"- KFAC start: 5000 epochs (`5000.npz`, internal epoch 4999)",f"- finite epochs: {row['finite_epochs']}",f"- first nonfinite: {row['first_nonfinite_epoch']}",f"- steady median: {fmt(row['steady_median_sec_per_epoch'])} s/epoch",f"- status: **{row['status']}**",'', '| epoch | energy | variance | acceptance |','|---:|---:|---:|---:|']
for e in (1,10,50,100,200):lines.append(f"| {e} | {fmt(row[f'energy_{e}'])} | {fmt(row[f'variance_{e}'])} | {fmt(row[f'acceptance_{e}'])} |")
(OUT/'n2_minsr_mu0_n4096_summary.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
