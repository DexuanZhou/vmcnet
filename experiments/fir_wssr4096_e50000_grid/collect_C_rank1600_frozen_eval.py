#!/usr/bin/env python3
import json,math,pathlib,numpy as np
from vmcnet.mcmc.statistics import tau
RUN=pathlib.Path('/scratch/dexuan1/runs/C_wssr_rank1600_E50000_frozen_eval_burn100k_n8192')
OUT=pathlib.Path(__file__).parent/'results/C_rank1600_frozen_eval_burn100k_n8192.json'
raw=np.loadtxt(RUN/'eval/local_energies.txt');finite=np.isfinite(raw);x=np.where(finite,raw,np.nan)
def stats(a,chunk=128):
 n,m=a.shape;means=np.nanmean(a,axis=0);pv=np.nanvar(a,axis=0,ddof=1);within=float(np.nanmean(pv));between=float(np.nanvar(means,ddof=1));var=within*(n-1)/n+between;ac=np.zeros(n);norm=(n-np.arange(n))[:,None]
 for lo in range(0,m,chunk):
  y=a[:,lo:lo+chunk]-np.nanmean(a[:,lo:lo+chunk],axis=0,keepdims=True);f=np.fft.fft(y,n=2*n,axis=0);g=np.fft.ifft(f*np.conjugate(f),axis=0).real[:n]/norm;g/=g[:1];ac+=np.nansum(pv[lo:lo+chunk]*g,axis=1)
 rho=1-(within-ac/m)/var;iac=float(tau(rho));size=int(np.isfinite(a).sum());mean=float(np.nanmean(np.nanmean(a,axis=1)));return {'mean':mean,'variance':var,'sem':math.sqrt(iac*var/size),'iac':iac,'ess':size/iac,'samples':size,'minimum':float(np.nanmin(a)),'maximum':float(np.nanmax(a)),**{f'q{q:g}':float(np.nanquantile(a,q)) for q in (.001,.01,.05,.5,.95,.99,.999)}}
center=np.nanmean(x,axis=1,keepdims=True);tv=np.nanmean(np.abs(x-center),axis=1,keepdims=True);clipped=np.clip(x,center-5*tv,center+5*tv)
accept=np.loadtxt(RUN/'eval/accept_ratio.txt')
result={'source_checkpoint':'/scratch/dexuan1/runs/wssr_C_4096_kfac1000_E50000_grid/rank1600_warm2_eta08_lr0002/checkpoints/50000.npz','reference_energy':-37.84471,'shape':list(x.shape),'nonfinite_samples':int((~finite).sum()),'acceptance_mean':float(np.mean(accept)),'unclipped':stats(x),'training_style_clipped':stats(clipped)}
OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
