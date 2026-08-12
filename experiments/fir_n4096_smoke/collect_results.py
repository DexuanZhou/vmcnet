#!/usr/bin/env python3
import csv
import json
import math
import pathlib
import re
import statistics
import subprocess
from datetime import datetime

HERE=pathlib.Path(__file__).resolve().parent
RUN_ROOT=pathlib.Path('/scratch/dexuan1/runs/fir_n4096_smoke')
RESULTS=HERE/'results'
RESULTS.mkdir(exist_ok=True)

def read_csv(path):
    return list(csv.DictReader(open(path))) if path.exists() else []

def event_map(path):
    rows=read_csv(path); out={}
    for r in rows:
        key=r['event'] if r['event']!='epoch_end' else f"epoch_{r['epoch']}"
        # Keep the first burn pair: eval.nepochs=0 still invokes a later eval
        # burn in this code revision, but production burn-in is the first pair.
        if key.startswith('epoch_'):
            out[key]=float(r['monotonic_seconds'])
        else:
            out.setdefault(key,float(r['monotonic_seconds']))
    return out

def sacct(job):
    if not job: return ('','','')
    p=subprocess.run(['sacct','-j',job,'-X','-n','-P','--format=Submit,Start,State'],text=True,capture_output=True)
    parts=(p.stdout.strip().splitlines() or ['||'])[0].split('|')
    return tuple((parts+['','',''])[:3])

def seconds_between(a,b):
    try: return (datetime.fromisoformat(b)-datetime.fromisoformat(a)).total_seconds()
    except: return math.nan

manifest=read_csv(HERE/'manifest.csv'); output=[]
for m in manifest:
    name=f"{m['system']}_rank{m['rank']}_warm{m['warm']}_n4096_e50"; d=RUN_ROOT/name
    train_d=d
    if not (train_d/'training_metrics.csv').exists():
        candidates=sorted(RUN_ROOT.glob(name+'_*/training_metrics.csv'))
        if candidates: train_d=candidates[-1].parent
    timing=event_map(d/'phase_timing.csv'); metrics=read_csv(train_d/'training_metrics.csv'); gpu=read_csv(d/'gpu_memory_poll.csv')
    job=(d/'slurm_job_id.txt').read_text().strip() if (d/'slurm_job_id.txt').exists() else ''
    submit,start,state=sacct(job)
    ident=next(csv.reader(open(d/'gpu_identity.csv'))) if (d/'gpu_identity.csv').exists() else ['','','']
    gpu_model=ident[0].strip() if ident else ''; total=float(ident[1]) if len(ident)>1 else math.nan
    used=[float(r['memory_used_mib']) for r in gpu if r.get('memory_used_mib','').strip()]
    max_used=max(used) if used else math.nan; margin=total-max_used if used else math.nan
    epochs=[]
    for e in range(11,51):
        if f'epoch_{e}' in timing and f'epoch_{e-1}' in timing: epochs.append(timing[f'epoch_{e}']-timing[f'epoch_{e-1}'])
    startup=timing.get('burn_start',math.nan)-timing.get('process_start',math.nan)
    burn=timing.get('burn_end',math.nan)-timing.get('burn_start',math.nan)
    first=timing.get('epoch_1',math.nan)-timing.get('burn_end',math.nan)
    finite=all(math.isfinite(float(r[k])) for r in metrics for k in ('energy','variance')) if metrics else False
    text=''
    for p in [d/'nvidia_smi_start.txt']:
        if p.exists(): text+=p.read_text(errors='replace')
    npz=len(list(d.rglob('*.npz'))) if d.exists() else 0
    if train_d != d: npz += len(list(train_d.rglob('*.npz')))
    stderr=''
    if job:
        base=pathlib.Path('/scratch/dexuan1/runs/logs'); stderr=''.join(p.read_text(errors='replace') for p in base.glob(f'wssr-n4096-smoke-*_{m["task_id"]}.err'))
    oom=bool(re.search(r'OOM|out of memory|RESOURCE_EXHAUSTED',stderr,re.I))
    if oom: status='OOM'
    elif metrics and not finite: status='NONFINITE'
    elif len(metrics)==50 and finite and npz==0:
        status='RUNNABLE_SAFE' if margin>=2048 else 'RUNNABLE_TIGHT_MEMORY'
    else: status='FAILED_OTHER'
    steady_mean=statistics.fmean(epochs) if epochs else math.nan; steady_med=statistics.median(epochs) if epochs else math.nan
    steady_std=statistics.stdev(epochs) if len(epochs)>1 else math.nan
    final=metrics[-1] if metrics else {}
    ar=[]
    if (train_d/'wssr_active_rank.txt').exists(): ar=[float(x) for x in (train_d/'wssr_active_rank.txt').read_text().split()]
    total_once=startup+burn+first
    def est(n): return steady_med*n/3600 if math.isfinite(steady_med) else math.nan
    def est_once(n): return (total_once+steady_med*n)/3600 if math.isfinite(steady_med) else math.nan
    output.append({'system':m['system'],'rank':m['rank'],'warm':m['warm'],'gpu_model':gpu_model,'gpu_total_mib':total,'max_gpu_memory_mib':max_used,'memory_margin_mib':margin,'queue_submit':submit,'job_start':start,'queue_wait_sec':seconds_between(submit,start),'slurm_state':state,'startup_sec':startup,'burnin_sec':burn,'first_epoch_compile_sec':first,'steady_mean_sec_per_epoch':steady_mean,'steady_median_sec_per_epoch':steady_med,'steady_std_sec_per_epoch':steady_std,'steady_min_sec_per_epoch':min(epochs) if epochs else math.nan,'steady_max_sec_per_epoch':max(epochs) if epochs else math.nan,'mean_optimizer_update_sec':'','estimated_25000_hours':est(25000),'estimated_100000_hours':est(100000),'estimated_25000_with_onetime_hours':est_once(25000),'estimated_100000_with_onetime_hours':est_once(100000),'final_energy':final.get('energy',''),'final_variance':final.get('variance',''),'final_active_rank':ar[-1] if ar else '','finite_epochs':len(metrics),'npz_count':npz,'status':status})

fields=list(output[0])
with open(RESULTS/'summary.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(output)
def fmt(x,n=2):
    try:return f'{float(x):.{n}f}'
    except:return str(x)
lines=['# Fir n=4096 smoke/timing results','', '| system | rank | warm | GPU MiB used/total | margin MiB | burn s | first/compile s | steady median s | 25k h | 100k h | final energy | final variance | active rank | status |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
for r in output:
    lines.append(f"| {r['system']} | {r['rank']} | {r['warm']} | {fmt(r['max_gpu_memory_mib'],0)}/{fmt(r['gpu_total_mib'],0)} | {fmt(r['memory_margin_mib'],0)} | {fmt(r['burnin_sec'])} | {fmt(r['first_epoch_compile_sec'])} | {fmt(r['steady_median_sec_per_epoch'],3)} | {fmt(r['estimated_25000_hours'])} | {fmt(r['estimated_100000_hours'])} | {fmt(r['final_energy'],7)} | {fmt(r['final_variance'],6)} | {fmt(r['final_active_rank'],0)} | {r['status']} |")
(RESULTS/'summary.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
