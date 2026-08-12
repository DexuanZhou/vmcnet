"""Timing-only launcher for standard full-state checkpoint reload."""
import csv, os, time
import jax
from vmcnet.train import runners
from vmcnet.train import vmc as vmc_module

OUT=os.environ['SMOKE_TIMING_DIR']; os.makedirs(OUT,exist_ok=True)
def event(name,epoch=''):
    p=os.path.join(OUT,'phase_timing.csv'); new=not os.path.exists(p)
    with open(p,'a',newline='') as f:
        w=csv.writer(f)
        if new:w.writerow(['event','epoch','wall_time_unix','monotonic_seconds'])
        w.writerow([name,epoch,time.time(),time.perf_counter()]); f.flush()
event('process_start')
orig_reload=runners.utils.io.reload_vmc_state
def timed_reload(*a,**kw):
    event('checkpoint_reload_start'); x=orig_reload(*a,**kw); jax.block_until_ready(x); event('checkpoint_reload_end'); return x
runners.utils.io.reload_vmc_state=timed_reload
orig_append=vmc_module._append_training_metrics_csv_row
def timed_append(logdir,epoch,metrics):
    x=orig_append(logdir,epoch,metrics); jax.block_until_ready(metrics); event('epoch_end',int(epoch)+1); return x
vmc_module._append_training_metrics_csv_row=timed_append
try:runners.run_molecule()
finally:event('process_end')
