#!/usr/bin/env python3
import csv, json, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path('/scratch/dexuan1/vmcnet/experiments/C_exact_first_complement/results/ssi_convergence')
rows=[]
dirs=[d for d in sorted(ROOT.iterdir()) if d.is_dir()]
if (ROOT/'initial_retry1'/'results.csv').exists():
    dirs=[d for d in dirs if d.name != 'initial']
for d in dirs:
    p=d/'results.csv'
    if not p.exists(): continue
    with p.open() as f: part=list(csv.DictReader(f))
    peak=math.nan; gp=d/'gpu_memory_poll.csv'
    if gp.exists():
        with gp.open() as f: peak=max((float(r['memory_used_mib']) for r in csv.DictReader(f)),default=math.nan)
    meta=json.loads((d/'metadata.json').read_text())
    spectrum=meta['exact_singular_values_395_405']
    true_gap=abs(spectrum[5]-spectrum[6])/abs(spectrum[5])
    for r in part:
        r['job_peak_gpu_memory_mib']=peak
        # The original replay row indexed a rank-400 array at 400; JAX clamps
        # that OOB access. Recover the true 400/401 gap from saved exact values.
        r['spectral_gap_400_401']=true_gap
    rows.extend(part)
fields=list(rows[0]) if rows else []
with (ROOT/'all_results.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def threshold(rs,err,cos):
    for r in sorted(rs,key=lambda x:int(x['q'])):
        if float(r['update_relative_error'])<err and float(r['update_cosine'])>cos:return int(r['q'])
    return 'not reached'

groups={}
for r in rows: groups.setdefault((r['snapshot'],r['initialization']),[]).append(r)
summary=[]
for (snap,init),rs in groups.items():
    summary.append({'snapshot':snap,'initialization':init,'loose_q':threshold(rs,1e-2,.9999),'moderate_q':threshold(rs,5e-3,.99995),'strict_q':threshold(rs,1e-3,.99999)})
with (ROOT/'thresholds.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['snapshot','initialization','loose_q','moderate_q','strict_q']);w.writeheader();w.writerows(summary)

fig,axs=plt.subplots(1,2,figsize=(10,4))
for key,rs in groups.items():
    rs=sorted(rs,key=lambda x:int(x['q'])); q=[int(x['q']) for x in rs]; label='/'.join(key)
    axs[0].plot(q,[float(x['update_relative_error']) for x in rs],marker='o',label=label)
    axs[1].plot(q,[float(x['ssi_wall_seconds']) for x in rs],marker='o',label=label)
axs[0].set_yscale('log');axs[0].axhline(1e-2,color='0.7',ls='--');axs[0].axhline(1e-3,color='0.4',ls=':');axs[0].set(xlabel='SSI iterations q',ylabel='update relative error')
axs[1].set(xlabel='SSI iterations q',ylabel='SSI wall time (s)');axs[1].legend(fontsize=6)
fig.tight_layout();fig.savefig(ROOT/'ssi_error_runtime.svg');plt.close(fig)

lines=['# Fixed-snapshot C SSI convergence','', '| Snapshot | Initialization | Loose q | Moderate q | Strict q |','|---|---|---:|---:|---:|']
for x in summary:lines.append(f"| {x['snapshot']} | {x['initialization']} | {x['loose_q']} | {x['moderate_q']} | {x['strict_q']} |")
lines += ['', 'All comparisons use one fixed operator/force per checkpoint; MCMC and parameters are not advanced. See `all_results.csv` for the complete metrics.']
(ROOT/'report.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
