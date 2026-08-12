#!/usr/bin/env python3
import csv, json, math
from pathlib import Path
import numpy as np
from vmcnet.mcmc.statistics import tau

ROOT=Path('/scratch/dexuan1/runs/wssr4096_frozen_eval/C')
OUT=Path(__file__).resolve().parent/'results'

def chunked_stats(x, chunk=128):
    """VMCNet multi-chain statistics without a full 1000x4096 complex FFT."""
    n, m=x.shape
    per_var=np.var(x,axis=0,ddof=1)
    within=float(np.mean(per_var)); between=float(np.var(np.mean(x,axis=0),ddof=1))
    variance=within*(n-1)/n+between
    ac_term=np.zeros(n,dtype=np.float64)
    norm=(n-np.arange(n))[:,None]
    for lo in range(0,m,chunk):
        y=x[:,lo:lo+chunk]-np.mean(x[:,lo:lo+chunk],axis=0,keepdims=True)
        f=np.fft.fft(y,n=2*n,axis=0)
        g=np.fft.ifft(f*np.conjugate(f),axis=0).real[:n]/norm
        g/=g[:1]
        ac_term+=np.sum(per_var[lo:lo+chunk]*g,axis=1)
    ac=1-(within-ac_term/m)/variance
    iac=float(tau(ac)); avg=float(np.mean(np.mean(x,axis=-1),axis=-1))
    return {'average':avg,'variance':variance,'std_err':math.sqrt(iac*variance/x.size),
            'integrated_autocorrelation':iac}

def summary(x):
    s=chunked_stats(x)
    s['effective_sample_size']=int(np.size(x)/s['integrated_autocorrelation'])
    s['minimum']=float(np.min(x)); s['maximum']=float(np.max(x))
    for q in (0.001,0.01,0.05,0.5,0.95,0.99,0.999): s[f'q{q:g}']=float(np.quantile(x,q))
    return s

def main():
    results={}
    for run in ('C_rank800_eta02_lr0002','C_rank800_eta08_lr02'):
        p=ROOT/run/'eval'
        raw=np.loadtxt(p/'local_energies.txt')
        finite=np.isfinite(raw)
        clean=np.where(finite,raw,np.nan)
        center=np.nanmean(clean,axis=1,keepdims=True)
        tv=np.nanmean(np.abs(clean-center),axis=1,keepdims=True)
        clipped=np.clip(clean,center-5*tv,center+5*tv)
        accept=np.loadtxt(p/'accept_ratio.txt')
        results[run]={'raw_unclipped':summary(clean),'training_style_clipped':summary(clipped),
                      'acceptance_mean':float(np.mean(accept)),'nonfinite_samples':int((~finite).sum()),
                      'raw_sample_count':int(raw.size),'shape':list(raw.shape)}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'C_frozen_eval_summary.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))
if __name__=='__main__': main()
