#!/usr/bin/env python3
"""Timing/early-stop wrapper; optional deterministic Optax count offset."""
import csv,json,os,time
import jax,jax.numpy as jnp,numpy as np
from vmcnet.train import runners,vmc as vmc_module
from vmcnet.updates import parse_optimizer_config
from vmcnet.updates import wssr
from vmcnet.utils import io
OUT=os.environ['PILOT_META']; OFFSET=int(os.environ.get('OPTAX_COUNT_OFFSET','-1'))
os.makedirs(OUT,exist_ok=True)
if os.environ.get('ADVANCE_RELOAD_EPOCH')=='1' or int(os.environ.get('EXPAND_WSSR_RANK','0'))>0:
 old_reload=io.reload_vmc_state
 def reload(*a,**k):
  e,d,p,o,key=old_reload(*a,**k);width=int(os.environ.get('EXPAND_WSSR_RANK','0'))
  if width:
   c=o.core_state;sr_o=jnp.zeros((c.sr_o.shape[0],width),c.sr_o.dtype).at[:,:c.sr_o.shape[1]].set(c.sr_o);ek=jnp.zeros((width,),c.ek.dtype).at[:c.ek.shape[0]].set(c.ek)
   c=wssr.WSSRWarmSVDCoreState(sr_o,ek,c.sr_rank0,jnp.asarray(width,c.sr_rank.dtype),c.u,c.has_u);o=o._replace(core_state=c)
  return int(e)+(1 if os.environ.get('ADVANCE_RELOAD_EPOCH')=='1' else 0),d,p,o,key
 io.reload_vmc_state=reload
if OFFSET>=0:
 old=parse_optimizer_config.initialize_optimizer
 def initialize(*a,**k):
  fn,state,key=old(*a,**k); leaves=jax.tree_util.tree_leaves(state.optax_state)
  ints=[x for x in leaves if getattr(x,'shape',None)==() and np.issubdtype(x.dtype,np.integer)]
  if len(ints)!=1: raise RuntimeError(f'expected one Optax counter, got {len(ints)}')
  state=state._replace(optax_state=jax.tree_util.tree_map(
   lambda x:np.asarray(OFFSET,dtype=x.dtype) if getattr(x,'shape',None)==() and np.issubdtype(x.dtype,np.integer) else x,
   state.optax_state))
  return fn,state,key
 parse_optimizer_config.initialize_optimizer=initialize
old_append=vmc_module._append_training_metrics_csv_row;vars=[];dirs=[]
def event(name,epoch=''):
 p=os.path.join(OUT,'phase_timing.csv');new=not os.path.exists(p)
 with open(p,'a',newline='') as f:
  w=csv.writer(f)
  if new:w.writerow(['event','epoch','wall_time_unix','monotonic_seconds'])
  w.writerow([name,epoch,time.time(),time.perf_counter()]);f.flush()
def append(logdir,epoch,metrics):
 z=old_append(logdir,epoch,metrics);jax.block_until_ready(metrics);event('epoch_end',int(epoch)+1)
 vars.append(float(metrics['variance']));dirs.append(float(metrics.get('wssr_diag_raw_direction_norm',0.)))
 if len(vars)>=20:
  vb=np.median(vars[:20]);db=np.median(dirs[:20]);reason=None
  if not np.isfinite(vars[-1]) or not np.isfinite(dirs[-1]):reason='nonfinite_metric'
  elif vb>0 and vars[-1]>100*vb:reason='variance_runaway'
  elif db>0 and dirs[-1]>100*db:reason='raw_direction_runaway'
  if reason:
   open(os.path.join(OUT,'early_stop.json'),'w').write(json.dumps({'triggered':True,'reason':reason,'epoch':int(epoch)+1},indent=2));raise RuntimeError(reason)
 return z
vmc_module._append_training_metrics_csv_row=append
event('process_start')
try:runners.run_molecule()
finally:event('process_end')
