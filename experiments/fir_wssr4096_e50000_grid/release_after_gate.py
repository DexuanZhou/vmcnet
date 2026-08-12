#!/usr/bin/env python3
import csv,json,math,pathlib,statistics,subprocess
pkg=pathlib.Path(__file__).parent; results={}; jobs={}
for sys in ('C','N2'):
 run=pathlib.Path(f'/scratch/dexuan1/runs/wssr_{sys}_4096_kfac1000_E50000_grid_gate/rank1600_warm2'); meta=pathlib.Path(str(run)+'_metadata')
 passed=False; result={'system':sys,'run':str(run)}
 try:
  rows=list(csv.DictReader(open(run/'training_metrics.csv'))); cfg=json.load(open(run/'config.json')); opt=cfg['vmc']['optimizer']['wssr_warm_svd_right']; tim=list(csv.DictReader(open(meta/'phase_timing.csv'))); ends=[(int(r['epoch']),float(r['monotonic_seconds'])) for r in tim if r['event']=='epoch_end']; dt=[ends[i][1]-ends[i-1][1] for i in range(1,len(ends)) if ends[i][0]>=2]; mem=list(csv.DictReader(open(meta/'gpu_memory_poll.csv')))
  finite=all(all(math.isfinite(float(r[k])) for k in ('energy','variance','accept_ratio')) for r in rows)
  passed=len(rows)==5 and finite and cfg['vmc']['nchains']==4096 and opt['sr_rank']==opt['sr_rank_max']==opt['sr_storage_rank']==opt['svd_working_rank']==1600 and opt['svd_maxiter_warm']==2
  result.update({'passed':passed,'epochs':len(rows),'finite':finite,'peak_gpu_mib':max(float(r['memory_used_mib']) for r in mem),'steady_median_sec_epoch':statistics.median(dt),'final_energy':float(rows[-1]['energy']),'final_variance':float(rows[-1]['variance']),'resolved_warm_key':'svd_maxiter_warm','resolved_warm':opt['svd_maxiter_warm']})
 except Exception as e: result.update({'passed':False,'reason':repr(e)})
 results[sys]=result
 array='0-4%5' if passed else '0-3%5'
 jobs[sys]=subprocess.check_output(['sbatch','--parsable',f'--array={array}',f'--export=ALL,SYSTEM={sys}',str(pkg/'scripts/grid_array.sh')],text=True).strip()
(pkg/'results/gate_results.json').write_text(json.dumps(results,indent=2)+'\n')
collector=subprocess.check_output(['sbatch','--parsable',f"--dependency=afterany:{jobs['C']}:{jobs['N2']}",str(pkg/'scripts/collector.sh')],text=True).strip()
(pkg/'results/job_ids.json').write_text(json.dumps({'gate':'49704096','arrays':jobs,'collector':collector},indent=2)+'\n')
print(json.dumps({'gate_results':results,'arrays':jobs,'collector':collector},indent=2))
