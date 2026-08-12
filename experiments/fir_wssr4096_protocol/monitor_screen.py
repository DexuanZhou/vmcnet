#!/usr/bin/env python3
import argparse,csv,math,os,signal,time,statistics,json,pathlib
p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args()
path=pathlib.Path(a.run)/'training_metrics.csv'; result={'triggered':False,'reason':'none','epoch':None}
while True:
 try: os.kill(a.pid,0)
 except OSError: break
 if path.exists():
  try:
   rows=list(csv.DictReader(path.open()))
   for r in rows:
    vals=[float(r[k]) for k in ('energy','variance','accept_ratio')]
    if not all(math.isfinite(x) for x in vals): result={'triggered':True,'reason':'nonfinite_metric','epoch':int(r['epoch'])};break
   if not result['triggered'] and len(rows)>=20:
    baseline=statistics.median(float(r['variance']) for r in rows[:20])
    if baseline>0 and float(rows[-1]['variance'])>100*baseline: result={'triggered':True,'reason':'variance_gt_100x_first20_median','epoch':int(rows[-1]['epoch'])}
   if result['triggered']:
    os.kill(a.pid,signal.SIGTERM);break
  except (OSError,ValueError,KeyError): pass
 time.sleep(2)
pathlib.Path(a.out).parent.mkdir(parents=True,exist_ok=True);pathlib.Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
