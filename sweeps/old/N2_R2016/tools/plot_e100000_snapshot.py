#!/usr/bin/env python3
"""Monochrome 10k-window C/N2 formal E100000 trajectory snapshots."""
import csv, math
from pathlib import Path
from plot_c_focus_10k import plot, rolling_full

HERE=Path(__file__).resolve().parent
ROOT=Path('/scratch/dexuan1/runs/e100000_followup')
C_REF=-37.84471
N2_REF=-109.5388
STYLES=[('', 'circle',3.2),('10 4','square',2.4),('3 3','triangle',2.2),('12 4 2 4','diamond',2.2)]

def metrics(path):
    rows=[]
    with (path/'training_metrics.csv').open(newline='') as f:
        for r in csv.DictReader(f):
            try: rows.append((float(r['energy']),float(r['variance'])))
            except (ValueError,KeyError): pass
    return rows

def build(specs, reference, prefix, title_system, reference_note=''):
    es=[]; vs=[]
    for (label,path),(dash,marker,width) in zip(specs,STYLES):
        data=metrics(path); energy=[x for x,_ in data]; variance=[x for _,x in data]
        er=rolling_full(energy); vr=rolling_full(variance)
        # Every source value is a full 10k-window average; render every 20th
        # source point (plus the endpoint) to keep the vector assets tractable.
        er=er[::20]+([er[-1]] if er[-1] != er[::20][-1] else [])
        vr=vr[::20]+([vr[-1]] if vr[-1] != vr[::20][-1] else [])
        # Var(E)/E0^2 = <(Delta E/E0)^2>; transform for a logarithmic plot.
        vr=[(x,math.log10(max(y/(reference*reference),1e-300))) for x,y in vr]
        es.append((f'{label} (n={len(data)})',[(x,(y-reference)*1000) for x,y in er],dash,marker,width))
        vs.append((f'{label} (n={len(data)})',vr,dash,marker,width))
    figs=HERE/'report_figs'; figs.mkdir(exist_ok=True)
    note=f' ({reference_note})' if reference_note else ''
    plot(es,f'{title_system} E100000 energy-error trajectories — snapshot{note}','10,000-iteration mean energy error (mHa)',figs/f'{prefix}_energy_window10000.svg')
    plot(vs,f'{title_system} E100000 normalized energy variance — snapshot{note}','log10[Var(E) / E₀²]',figs/f'{prefix}_variance_window10000.svg')

def main():
    c=ROOT/'C'
    c_specs=[
      ('SPRING lr=0.02',c/'C_spring_lr02_mu099_nc001_fresh_E100000_spin42'),
      ('WSSR eta=0.8 lr=0.02 SVD8/2',c/'C_wssr_eta08_lr02_tikhonov_svd8_2_fresh_E100000_spin42'),
      ('WSSR eta=0.8 lr=0.02 SVD8/5',c/'C_wssr_eta08_lr02_tikhonov_svd8_5_fresh_E100000_spin42'),
      ('WSSR eta=0.2 lr=0.002 SVD8/2',c/'C_wssr_eta02_lr0002_tikhonov_svd8_2_fresh_E100000_spin42'),
    ]
    n=ROOT/'N2eq'
    n_specs=[
      ('WSSR eta=0.2 lr=0.002 C=0.001',n/'N2eq_wssr_eta02_lr0002_tikhonov_E100000_retry1'),
      ('WSSR eta=0.3 lr=0.002 C=0.001',n/'N2eq_wssr_eta03_lr0002_tikhonov_E100000'),
      ('WSSR eta=0.3 lr=0.002 C=0.002',n/'N2eq_wssr_eta03_lr0002_tikhonov_C002_E100000'),
      ('WSSR eta=0.3 lr=0.003 C=0.001',n/'N2eq_wssr_eta03_lr0003_tikhonov_E100000'),
    ]
    build(c_specs,C_REF,'C_E100000','C spin(4,2)',f'E₀={C_REF} Ha')
    build(n_specs,N2_REF,'N2eq_R2016_E100000','N2eq R=2.016 Bohr',f'legacy E₀={N2_REF} Ha')
    print('wrote four snapshot figures')
if __name__=='__main__':main()
