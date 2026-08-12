#!/usr/bin/env python3
import argparse,csv,json,math,os,pathlib,signal,statistics,time
p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args()
metrics=pathlib.Path(a.run)/'training_metrics.csv';diag=pathlib.Path(a.run)/'spring_diagnostics.csv'
result={'triggered':False,'reason':'none','epoch':None}
while True:
 try: os.kill(a.pid,0)
 except OSError: break
 try:
  rows=list(csv.DictReader(metrics.open())) if metrics.exists() else []
  for r in rows:
   if not all(math.isfinite(float(r[k])) for k in ('energy','variance','accept_ratio')):
    result={'triggered':True,'reason':'nonfinite_metric','epoch':int(r['epoch'])};break
  if not result['triggered'] and len(rows)>=20:
   base=statistics.median(float(r['variance']) for r in rows[:20])
   if base>0 and float(rows[-1]['variance'])>100*base:
    result={'triggered':True,'reason':'variance_gt_100x_first20_median','epoch':int(rows[-1]['epoch'])}
  drows=list(csv.DictReader(diag.open())) if diag.exists() else []
  if not result['triggered'] and len(drows)>=20:
   vals=[float(r['spring_diag_raw_solution_norm']) for r in drows]
   base=statistics.median(vals[:20])
   if base>0 and vals[-1]>100*base:
    result={'triggered':True,'reason':'raw_direction_gt_100x_first20_median','epoch':len(drows)}
  if result['triggered']:
   os.kill(a.pid,signal.SIGTERM);break
 except (OSError,ValueError,KeyError): pass
 time.sleep(2)
pathlib.Path(a.out).parent.mkdir(parents=True,exist_ok=True)
pathlib.Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
