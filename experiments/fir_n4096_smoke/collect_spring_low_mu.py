#!/usr/bin/env python3
"""Collect the controlled N2 n=4096 low-mu SPRING sweep."""
import csv, json, math, pathlib, statistics
import matplotlib.pyplot as plt

ROOT = pathlib.Path("/scratch/dexuan1/runs/fir_n4096_spring_low_mu/N2")
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "results" / "spring_low_mu"
OUT.mkdir(parents=True, exist_ok=True)
MUS = [(0.0,"mu000"),(0.5,"mu050"),(0.7,"mu070"),(0.8,"mu080")]

def num(x):
    try: return float(x)
    except (TypeError,ValueError): return math.nan

def read(path):
    with path.open(newline="") as f: return list(csv.DictReader(f))

def longest_true(values):
    best=cur=0
    for value in values:
        cur=cur+1 if value else 0; best=max(best,cur)
    return best

def timing(meta):
    events=read(meta/"phase_timing.csv")
    ends={int(r["epoch"]):num(r["monotonic_seconds"]) for r in events if r["event"]=="epoch_end"}
    return {e:ends[e]-ends[e-1] for e in ends if e-1 in ends}

runs={}; summary=[]
for mu,label in MUS:
    name=f"N2eq_R2068_spring_lr0005_{label}_n4096_e200_diag"
    run=ROOT/name; meta=ROOT/"metadata"/name
    if not (run/"training_metrics.csv").exists(): continue
    train=read(run/"training_metrics.csv"); diag=read(run/"spring_diagnostics.csv")
    seconds=timing(meta); n=min(len(train),len(diag)); rows=[]
    for i in range(n):
        row=dict(train[i]); row.update(diag[i]); row["seconds_per_epoch"]=seconds.get(i+1,math.nan); rows.append(row)
    runs[mu]=rows
    fields=list(rows[0]);
    with (OUT/f"{label}_per_epoch.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    ratio=[num(r["spring_diag_mu_history_over_nonhistory"]) for r in rows]
    projection=[num(r["spring_diag_history_projection_norm"]) for r in rows]
    scale=[num(r["spring_diag_norm_constraint_scale"]) for r in rows]
    history=[num(r["spring_diag_history_norm"]) for r in rows]
    variance=[num(r["variance"]) for r in rows]
    energies=[num(r["energy"]) for r in rows]
    accept=[num(r["accept_ratio"]) for r in rows]
    finite=[all(math.isfinite(x) for x in (energies[i],variance[i],accept[i],history[i])) for i in range(n)]
    first_dom=next((i+1 for i,x in enumerate(ratio) if x>1),"")
    sustained=longest_true([x>1 for x in ratio])>=5
    active=[x<0.999 for x in scale]
    growth=longest_true([history[i]>history[i-1] for i in range(1,n)])>=5
    if n>=30:
        growth=growth or statistics.median(history[-20:])>2*max(statistics.median(history[:10]),1e-30)
    stop={"triggered":False,"reason":"none","epoch":None}
    if (meta/"threshold_stop.json").exists(): stop=json.loads((meta/"threshold_stop.json").read_text())
    severe=min(scale,default=1)<0.2 or sum(active)>0.5*n
    if n<200 or not all(finite) or stop.get("triggered"): status="UNSTABLE"
    elif sustained or severe or growth: status="MARGINALLY_STABLE"
    else: status="STABLE"
    tail=rows[-50:] if n>=50 else []
    tail_e=[num(r["energy"]) for r in tail]; tail_v=[num(r["variance"]) for r in tail]
    tail_a=[num(r["accept_ratio"]) for r in tail]; tail_s=[num(r["spring_diag_norm_constraint_scale"]) for r in tail]
    tail_t=[num(r["seconds_per_epoch"]) for r in tail if math.isfinite(num(r["seconds_per_epoch"]))]
    steady_t=[num(r["seconds_per_epoch"]) for r in rows[10:] if math.isfinite(num(r["seconds_per_epoch"]))]
    rolling=[(statistics.fmean(energies[i-19:i+1]),i+1) for i in range(19,n) if all(math.isfinite(x) for x in energies[i-19:i+1])]
    best=min(rolling,default=(math.nan,""))
    summary.append({
      "mu":mu,"completed_epochs":n,"last_finite_epoch":sum(finite),
      "stop_reason":stop.get("reason","none"),"first_history_ratio_gt1":first_dom,
      "max_history_ratio":max((x for x in ratio if math.isfinite(x)),default=math.nan),
      "max_history_projection_norm":max((x for x in projection if math.isfinite(x)),default=math.nan),
      "min_constraint_scale":min((x for x in scale if math.isfinite(x)),default=math.nan),
      "constraint_active_epochs":sum(active),"constraint_active_fraction":sum(active)/n,
      "max_variance":max((x for x in variance if math.isfinite(x)),default=math.nan),
      "sustained_history_domination":sustained,"sustained_or_monotone_history_growth":growth,
      "status":status,
      "tail50_mean_energy":statistics.fmean(tail_e) if tail else math.nan,
      "tail50_energy_sem":statistics.stdev(tail_e)/math.sqrt(len(tail_e)) if len(tail_e)>1 else math.nan,
      "tail50_mean_variance":statistics.fmean(tail_v) if tail else math.nan,
      "tail50_median_acceptance":statistics.median(tail_a) if tail else math.nan,
      "tail50_median_sec_per_epoch":statistics.median(tail_t) if tail_t else math.nan,
      "available_steady_median_sec_per_epoch":statistics.median(steady_t) if steady_t else math.nan,
      "tail50_mean_constraint_scale":statistics.fmean(tail_s) if tail else math.nan,
      "tail50_median_constraint_scale":statistics.median(tail_s) if tail else math.nan,
      "best_rolling20_mean_energy":best[0],"best_rolling20_end_epoch":best[1],
    })

if summary:
    with (OUT/"summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=summary[0]); w.writeheader(); w.writerows(summary)

styles={0.0:("-","o"),0.5:("--","s"),0.7:("-.","^"),0.8:(":","D")}
def plot(filename,ylabel,key,log=False):
    fig,ax=plt.subplots(figsize=(7,4))
    for mu,rows in runs.items():
        ls,m=styles[mu]; x=[num(r["epoch"]) for r in rows]; y=[num(r[key]) for r in rows]
        ax.plot(x,y,ls=ls,marker=m,markevery=max(1,len(x)//12),ms=3,label=f"mu={mu:g}")
    if log: ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel(ylabel); ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT/filename); plt.close(fig)

if runs:
    plot("energy.svg","Energy (Ha)","energy")
    plot("variance.svg","Local-energy variance (Ha²)","variance",True)
    plot("history_ratio.svg","||mu history|| / ||non-history||","spring_diag_mu_history_over_nonhistory",True)
    plot("history_projection.svg","Sample-space history projection norm","spring_diag_history_projection_norm",True)
    plot("raw_direction.svg","Raw SPRING direction norm","spring_diag_raw_solution_norm",True)
    plot("constraint_scale.svg","Norm-constraint scale","spring_diag_norm_constraint_scale",True)
    fig,ax=plt.subplots(figsize=(7,4)); labels=[]; values=[]
    for row in summary: labels.append(f"SPRING mu={row['mu']:g}"); values.append(row["available_steady_median_sec_per_epoch"])
    for rank,value in [(400,.403778155),(800,.519870449),(1600,.769788499)]: labels.append(f"WSSR r{rank} (hist.)"); values.append(value)
    ax.barh(labels,values,color="white",edgecolor="black",hatch=["","//","xx","..","//","xx",".."][:len(labels)])
    ax.set_xlabel("Median seconds / epoch"); ax.grid(axis="x",alpha=.25); fig.tight_layout(); fig.savefig(OUT/"runtime_comparison.svg"); plt.close(fig)

print(f"Collected {len(summary)} low-mu runs into {OUT}")
for row in summary: print(row)
