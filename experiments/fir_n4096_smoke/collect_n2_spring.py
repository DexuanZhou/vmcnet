#!/usr/bin/env python3
import csv, math, pathlib, re, statistics

PKG=pathlib.Path(__file__).resolve().parent
ROOT=pathlib.Path('/scratch/dexuan1/runs/fir_n4096_corrected_retry1/N2')
NAME='N2eq_R2068_spring_lr001_n4096_corrected_e50'
RUN=ROOT/'spring_smoke'/NAME
META=ROOT/'metadata'/NAME
RESULTS=PKG/'results'; RESULTS.mkdir(exist_ok=True)

def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))
def num(x):
    try:return float(x)
    except:return math.nan
metrics=read(RUN/'training_metrics.csv') if (RUN/'training_metrics.csv').exists() else []
events=read(META/'phase_timing.csv') if (META/'phase_timing.csv').exists() else []
stamp={int(r['epoch']):num(r['monotonic_seconds']) for r in events if r['event']=='epoch_end'}
named={r['event']:num(r['monotonic_seconds']) for r in events if r['event']!='epoch_end'}
steady=[stamp[e]-stamp[e-1] for e in range(11,51) if e in stamp and e-1 in stamp]
mean=statistics.fmean(steady) if steady else math.nan
median=statistics.median(steady) if steady else math.nan
std=statistics.stdev(steady) if len(steady)>1 else math.nan
first_compile=stamp.get(1,math.nan)-named.get('checkpoint_reload_end',math.nan)
startup=named.get('checkpoint_reload_start',math.nan)-named.get('process_start',math.nan)
reload_time=named.get('checkpoint_reload_end',math.nan)-named.get('checkpoint_reload_start',math.nan)
gpu=read(META/'gpu_memory_poll.csv') if (META/'gpu_memory_poll.csv').exists() else []
used=[num(r.get('memory_used_mib')) for r in gpu]; used=[x for x in used if math.isfinite(x)]
raw=(META/'gpu_identity.csv').read_text().strip().split(',') if (META/'gpu_identity.csv').exists() else []
model=raw[0].strip() if raw else ''; total=num(raw[1]) if len(raw)>1 else math.nan
maximum=max(used) if used else math.nan; margin=total-maximum
finite=sum(all(math.isfinite(num(r.get(k))) for k in ('energy','variance','accept_ratio')) for r in metrics)
first=metrics[0] if metrics else {}; final=metrics[-1] if metrics else {}
stderr=''.join(p.read_text(errors='replace') for p in pathlib.Path('/scratch/dexuan1/runs/logs').glob('N2-spring4096-smoke-*.err'))
oom=bool(re.search(r'out of memory|RESOURCE_EXHAUSTED',stderr,re.I))
valid=len(metrics)==50 and finite==50 and abs(num(first.get('energy'))+109.48)<2 and 0.2<num(final.get('accept_ratio'))<0.8 and not oom
status='OOM' if oom else ('NONFINITE' if metrics and finite<len(metrics) else ('SCIENTIFICALLY_VALID_RUNNABLE' if valid else 'FAILED_OTHER'))
wssr=[('rank400/warm1',0.404,43195),('rank800/warm2',0.520,43195),('rank1600/warm2',0.770,51387)]
row={'optimizer':'SPRING','learning_rate':0.001,'nchains':4096,'epochs':len(metrics),'startup_sec':startup,'checkpoint_reload_sec':reload_time,'burnin_sec':0.0,'first_epoch_compile_sec':first_compile,'steady_mean_sec_per_epoch':mean,'steady_median_sec_per_epoch':median,'steady_std_sec_per_epoch':std,'estimated_25000_hours':median*25000/3600,'estimated_100000_hours':median*100000/3600,'gpu_model':model,'gpu_total_mib':total,'max_gpu_memory_mib':maximum,'memory_margin_mib':margin,'epoch1_energy':num(first.get('energy')),'final_energy':num(final.get('energy')),'final_variance':num(final.get('variance')),'final_acceptance':num(final.get('accept_ratio')),'finite_epochs':finite,'status':status}
with (RESULTS/'n2_spring_n4096_smoke_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=row); w.writeheader(); w.writerow(row)
def f(x,n=3): return f'{x:.{n}f}' if math.isfinite(x) else 'nan'
lines=['# N2 SPRING n=4096 smoke timing','',f"- status: **{status}**",f"- steady mean/median/std: {f(mean)}/{f(median)}/{f(std)} s/epoch",f"- estimated 25k / 100k: {f(row['estimated_25000_hours'],2)} / {f(row['estimated_100000_hours'],2)} h",f"- GPU maximum / total / margin: {f(maximum,0)} / {f(total,0)} / {f(margin,0)} MiB",f"- epoch-1 / final energy: {f(row['epoch1_energy'],7)} / {f(row['final_energy'],7)} Ha",f"- final variance / acceptance: {f(row['final_variance'],6)} / {f(row['final_acceptance'],5)}",'', '| WSSR | time ratio WSSR/SPRING | memory ratio WSSR/SPRING | per-epoch comparison |','|---|---:|---:|---|']
for name,t,m in wssr:
    tr=t/median; mr=m/maximum
    lines.append(f"| {name} | {f(tr)} | {f(mr)} | {'slower' if tr>1 else 'faster'} than SPRING |")
(RESULTS/'n2_spring_n4096_smoke_summary.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
