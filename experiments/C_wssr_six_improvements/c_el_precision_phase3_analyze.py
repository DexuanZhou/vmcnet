"""Phase 3: NumPy/matplotlib-only analysis of matched precision results."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/scratch/dexuan1/vmcnet/experiments/C_wssr_six_improvements/results/energy_bias_audit/p2_c_fp64")


def stats(x):
    x=np.asarray(x,dtype=np.float64)
    q=np.quantile(x,[.0001,.001,.01,.5,.99,.999,.9999])
    return dict(mean=float(x.mean()),std=float(x.std(ddof=1)),min=float(x.min()),
                max=float(x.max()),q0001=float(q[0]),q001=float(q[1]),
                q01=float(q[2]),q50=float(q[3]),q99=float(q[4]),
                q999=float(q[5]),q9999=float(q[6]))


def main():
    a=np.load(OUT/"phase1_f32.npz"); b=np.load(OUT/"phase2_f64.npz")
    e32=a["energy_float32"].astype(np.float64)
    old=a["original_saved_energy"].astype(np.float64)
    e64=b["energy_float64"]; lp=b["logabspsi_float64"]; de=e64-e32
    common=(np.abs(e32)<=50)&(np.abs(e64)<=50)
    edges=np.quantile(lp,np.linspace(0,1,21)); rows=[]
    for i in range(20):
        m=(lp>=edges[i]) & ((lp<=edges[i+1]) if i==19 else (lp<edges[i+1]))
        rows.append([i,m.sum(),edges[i],edges[i+1],lp[m].mean(),de[m].mean(),
                     de[m].std(ddof=1),np.abs(de[m]).mean(),np.abs(de[m]).max()])
    rows=np.asarray(rows)
    np.savetxt(OUT/"dE_vs_logpsi_bins.csv",rows,delimiter=",",comments="",
      header="bin,count,logpsi_left,logpsi_right,logpsi_mean,dE_mean,dE_std,abs_dE_mean,abs_dE_max")
    summary={
      "sample_count":len(e32),"float32_energy":stats(e32),"float64_energy":stats(e64),
      "delta_energy_float64_minus_float32":stats(de),
      "mean_shift_Ha":float(e64.mean()-e32.mean()),
      "variance_float32":float(e32.var(ddof=1)),"variance_float64":float(e64.var(ddof=1)),
      "variance_difference":float(e64.var(ddof=1)-e32.var(ddof=1)),
      "original_saved_replay_delta":stats(e32-old),
      "original_saved_replay_allclose":bool(np.allclose(e32,old,rtol=2e-5,atol=2e-5)),
      "common_abs_energy_le_50_count":int(common.sum()),
      "common_abs_energy_le_50_fraction":float(common.mean()),
      "trimmed_float32_mean":float(e32[common].mean()),
      "trimmed_float64_mean":float(e64[common].mean()),
      "trimmed_mean_shift_Ha":float(e64[common].mean()-e32[common].mean()),
      "corr_logpsi_abs_delta_energy":float(np.corrcoef(lp,np.abs(de))[0,1]),
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    rng=np.random.default_rng(20260723); idx=rng.choice(len(de),min(20000,len(de)),False)
    fig,ax=plt.subplots(figsize=(7.2,4.8))
    ax.scatter(lp[idx],de[idx]*1000,s=3,alpha=.18,rasterized=True)
    ax.plot(rows[:,4],rows[:,5]*1000,"o-",color="black",lw=1.5,label="bin mean")
    ax.axhline(0,color=".5",lw=.8);ax.set_xlabel(r"$\\log|\\psi|$")
    ax.set_ylabel(r"$E_L^{64}-E_L^{32}$ (mHa)");ax.legend();fig.tight_layout()
    fig.savefig(OUT/"dE_vs_logpsi.svg");fig.savefig(OUT/"dE_vs_logpsi.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,4.8));lim=np.quantile(np.abs(de),.999)
    ax.hist(de[np.abs(de)<=lim]*1000,bins=160,histtype="step");ax.set_yscale("log")
    ax.set_xlabel(r"$E_L^{64}-E_L^{32}$ (mHa)");ax.set_ylabel("count");fig.tight_layout()
    fig.savefig(OUT/"dE_histogram.svg");fig.savefig(OUT/"dE_histogram.png",dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
