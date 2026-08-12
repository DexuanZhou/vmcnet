#!/usr/bin/env python3
"""Create the required final ablation plots from collected text/CSV artifacts."""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path('/scratch/dexuan1/vmcnet/experiments/C_wssr_six_improvements')
OUT = ROOT / 'results' / 'plots'
OUT.mkdir(parents=True, exist_ok=True)
stage1 = list(csv.DictReader(open(ROOT/'results/stage1/aggregate.csv')))
methods = json.load(open(ROOT/'results/stage1/stage2_methods.json'))
selection = json.load(open(ROOT/'results/stage1/selection.json'))


def variant(method):
    parameter = 'warm2' if method == 'baseline' else str(selection[method]['parameter'])
    return f'{method}_{parameter.replace("+", "_")}'


# Fixed-snapshot numerical fidelity.
fig, ax = plt.subplots(figsize=(10, 5))
eligible = [r for r in stage1 if r['method'] not in ('baseline','fixed_complement_reference')]
labels = [f"{r['method']}\n{r['parameter']}" for r in eligible]
colors = ['0.25' if r['eligible']=='True' else '0.75' for r in eligible]
ax.bar(np.arange(len(eligible)), [float(r['mean_full_error']) for r in eligible], color=colors)
ax.set_xticks(np.arange(len(labels)), labels, rotation=65, ha='right', fontsize=7)
ax.set_ylabel('mean relative update error vs exact full reference')
ax.set_yscale('log'); fig.tight_layout(); fig.savefig(OUT/'stage1_update_error.svg'); plt.close(fig)


# E200 raw variance and effective rank trajectories.
fig_v, ax_v = plt.subplots(figsize=(9, 5)); fig_r, ax_r = plt.subplots(figsize=(9, 5))
runtime=[]
for method in methods:
    v=variant(method); run=Path('/scratch/dexuan1/runs/C_wssr6_stage2_E200')/v
    rows=list(csv.DictReader(open(run/'training_metrics.csv')))
    raw=np.asarray([float(r['variance_noclip']) for r in rows]); epoch=np.arange(1,len(raw)+1)
    window=min(20,len(raw)); smooth=np.convolve(raw,np.ones(window)/window,mode='valid')
    ax_v.plot(epoch[window-1:],smooth,label=method)
    rank_path=run/'wssr_active_rank.txt'
    if rank_path.exists():
        rank=np.atleast_1d(np.loadtxt(rank_path));ax_r.plot(np.arange(1,len(rank)+1),rank,label=method)
    meta=Path(str(run)+'.metadata')/'phase_timing.csv'
    times=[float(r['monotonic_seconds']) for r in csv.DictReader(open(meta)) if r['event']=='epoch_end']
    runtime.append((method,float(np.median(np.diff(times)[10:]))))
ax_v.set(xlabel='epoch',ylabel='raw/unclipped local-energy variance');ax_v.legend(fontsize=7);fig_v.tight_layout();fig_v.savefig(OUT/'stage2_raw_variance_trajectory.svg');plt.close(fig_v)
ax_r.set(xlabel='epoch',ylabel='effective/active rank');ax_r.legend(fontsize=7);fig_r.tight_layout();fig_r.savefig(OUT/'stage2_effective_rank_trajectory.svg');plt.close(fig_r)
fig,ax=plt.subplots(figsize=(8,4));ax.bar([x[0] for x in runtime],[x[1] for x in runtime],color='0.35');ax.tick_params(axis='x',rotation=55);ax.set_ylabel('median steady seconds / epoch');fig.tight_layout();fig.savefig(OUT/'stage2_runtime.svg');plt.close(fig)


# Cross-seed frozen variance comparison.
stage3=list(csv.DictReader(open(ROOT/'results/stage3/cross_seed.csv')))
fig,ax=plt.subplots(figsize=(8,4));names=[r['method'] for r in stage3];means=[float(r['frozen_raw_variance_mean']) for r in stage3];std=[float(r['frozen_raw_variance_std']) for r in stage3];ax.bar(names,means,yerr=std,color='0.4');ax.tick_params(axis='x',rotation=50);ax.set_ylabel('frozen raw variance (mean ± seed SD)');fig.tight_layout();fig.savefig(OUT/'stage3_frozen_raw_variance.svg');plt.close(fig)
print(OUT)
