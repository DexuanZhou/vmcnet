#!/usr/bin/env python3
"""Fold a reproducible Stage-3 seed into a reloaded checkpoint PRNG."""
import os,jax,csv,time,json,numpy as np
from vmcnet.train import runners
from vmcnet.train import vmc as vmc_module
from vmcnet.utils import io
seed=int(os.environ['WSSR_STAGE3_SEED'])
old=io.reload_vmc_state
out=os.environ.get('SMOKE_TIMING_DIR')
if out:
 os.makedirs(out,exist_ok=True)
 open(os.path.join(out,'seed_protocol.json'),'w').write(json.dumps({'seed_index':seed,'checkpoint_key_fold_in':0x6A09E667+seed,'reburn':True,'nburn':5000},indent=2))
def event(name,epoch=''):
 if not out:return
 os.makedirs(out,exist_ok=True);p=os.path.join(out,'phase_timing.csv');new=not os.path.exists(p)
 with open(p,'a',newline='') as f:
  w=csv.writer(f)
  if new:w.writerow(['event','epoch','wall_time_unix','monotonic_seconds'])
  w.writerow([name,epoch,time.time(),time.perf_counter()]);f.flush()
def reload(*a,**k):
 event('checkpoint_reload_start');e,d,p,o,key=old(*a,**k);jax.block_until_ready((d,p,o,key));event('checkpoint_reload_end');return e,d,p,o,jax.random.fold_in(key,0x6A09E667+seed)
io.reload_vmc_state=reload
old_append=vmc_module._append_training_metrics_csv_row
variances=[];directions=[]
def append(logdir,epoch,metrics):
 x=old_append(logdir,epoch,metrics);jax.block_until_ready(metrics);event('epoch_end',int(epoch)+1)
 variances.append(float(metrics['variance']))
 directions.append(float(metrics.get('wssr_diag_raw_direction_norm',0.0)))
 reason=None
 if len(variances)>=20:
  vb=float(np.median(variances[:20]));db=float(np.median(directions[:20]))
  if not np.isfinite(variances[-1]) or not np.isfinite(directions[-1]):reason='nonfinite_metric'
  elif vb>0 and variances[-1]>100*vb:reason='variance_gt_100x_first20_median'
  elif db>0 and directions[-1]>100*db:reason='raw_direction_gt_100x_first20_median'
 if reason:
  if out:open(os.path.join(out,'early_stop.json'),'w').write(json.dumps({'triggered':True,'reason':reason,'epoch':int(epoch)+1},indent=2))
  raise RuntimeError(reason)
 return x
vmc_module._append_training_metrics_csv_row=append
event('process_start')
try:runners.run_molecule()
finally:event('process_end')
