#!/usr/bin/env python3
import csv, json, math, pathlib, statistics

ROOT=pathlib.Path('/scratch/dexuan1/runs/fir_n4096_spring_mu05_precision/N2')
MIXED_RETRY_ROOT=pathlib.Path('/scratch/dexuan1/runs/fir_n4096_spring_mu05_precision_retry5/N2')
OUT=pathlib.Path(__file__).resolve().parent/'results'/'spring_mu05_precision'; OUT.mkdir(parents=True,exist_ok=True)
NAMES={'float32':'N2eq_R2068_spring_mu050_float32_n4096_e50','mixed64':'N2eq_R2068_spring_mu050_mixed64_n4096_e50'}
def n(x):
 try:return float(x)
 except:return math.nan
def read(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
runs={}; summaries=[]
for mode,name in NAMES.items():
 root=MIXED_RETRY_ROOT if mode=='mixed64' else ROOT
 run=root/name; meta=root/'metadata'/name
 tr=read(run/'training_metrics.csv'); dg=read(run/'spring_diagnostics.csv'); rows=[]
 for a,b in zip(tr,dg): q=dict(a);q.update(b);rows.append(q)
 runs[mode]=rows
 with (OUT/f'{mode}_per_epoch.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 stop=json.loads((meta/'threshold_stop.json').read_text()) if (meta/'threshold_stop.json').exists() else {}
 summaries.append({'mode':mode,'epochs':len(rows),'stop_reason':stop.get('reason','none'),'stop_epoch':stop.get('epoch',''),
  'final_energy':n(rows[-1]['energy']),'final_variance':n(rows[-1]['variance']),'final_acceptance':n(rows[-1]['accept_ratio']),
  'max_rhs_ratio':max(n(r['spring_decomp_rhs_ratio']) for r in rows),'max_true_ratio':max(n(r['spring_decomp_true_history_ratio']) for r in rows),
  'max_projection_norm':max(n(r['spring_decomp_projection_norm']) for r in rows),'max_phi_norm':max(n(r['spring_diag_raw_solution_norm']) for r in rows),
  'min_constraint_scale':min(n(r['spring_diag_norm_constraint_scale']) for r in rows)})
with (OUT/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=summaries[0]);w.writeheader();w.writerows(summaries)

selected=[]
for e in (1,4,8,12,15,17):
 if e<=len(runs['float32']):
  r=runs['float32'][e-1]
  selected.append({'epoch':e,
   'epsilon_norm':r['spring_decomp_epsilon_norm'],'projection_norm':r['spring_decomp_projection_norm'],'R_rhs':r['spring_decomp_rhs_ratio'],
   'c32_norm':r['spring_replay_current_float32_norm'],'c64_norm':r['spring_replay_current_float64_norm'],'c_relative_error':r['spring_replay_current_relative_error'],'c_cosine':r['spring_replay_current_cosine'],
   'h32_norm':r['spring_replay_history_float32_norm'],'h64_norm':r['spring_replay_history_float64_norm'],'h_relative_error':r['spring_replay_history_relative_error'],'h_cosine':r['spring_replay_history_cosine'],
   'phi32_norm':r['spring_replay_total_float32_norm'],'phi64_norm':r['spring_replay_total_float64_norm'],'phi_relative_error':r['spring_replay_total_relative_error'],'phi_cosine':r['spring_replay_total_cosine'],
   'R_true':r['spring_decomp_true_history_ratio'],'cos_c_h':r['spring_decomp_cos_current_history'],
   'operator_norm32_est':r['spring_contractivity_operator_norm_estimate'],'mu_operator_norm32_est':r['spring_contractivity_mu_operator_norm_estimate'],
   'sym_min_eig32_est':r['spring_contractivity_symmetric_min_eigenvalue'],'sym_max_eig32_est':r['spring_contractivity_symmetric_max_eigenvalue'],
   'constraint_scale':r['spring_diag_norm_constraint_scale'],'variance':r['variance'],'acceptance':r['accept_ratio']})
if selected:
 with (OUT/'selected_replay_contractivity.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=selected[0]);w.writeheader();w.writerows(selected)
print(summaries)
