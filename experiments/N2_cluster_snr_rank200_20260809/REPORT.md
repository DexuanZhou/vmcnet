# N2 rank-200 spectral-cluster SNR shrinkage: no-go

## Decision

The candidate fails the pre-registered 200-step health gate decisively, so no
E1000, E5000, or E100000 accuracy run is admitted.  Slurm job `53869589`
reached 151 epochs before an undersized eight-minute wall-time limit stopped
it.  The decision uses the 50 recorded updates from epochs 101--150.  A rerun
of the remaining 49 epochs is unnecessary because the two failed quantities
are far from their gates throughout that tail window.

## Configuration

- N2 equilibrium, KFAC-pre5000 checkpoint, 1000 walkers, 10 MCMC steps/update.
- Current-batch rank 200, SSI 40/2, fixed lambda `1e-3`, learning rate `0.002`,
  norm constraint `0.001`.
- No historical S, gradient EMA, complement, error feedback, or reburn.
- Even/odd walker halves estimate coefficient reproducibility in a shared
  current-batch Rayleigh--Ritz basis.  Near-degenerate modes share one weight.

## Tail-window telemetry

| quantity | median | tail min | tail max | gate |
|---|---:|---:|---:|---:|
| spectral clusters | 29 | 24 | 34 | diagnostic |
| mean SNR weight | 0.09698 | 0.04247 | 0.20884 | diagnostic |
| weak-boundary weight | 0.02618 | 0.00000 | 0.21937 | below strong |
| strong-boundary weight | 0.31031 | 0.17520 | 0.52486 | above weak |
| cross-half force cosine | 0.12890 | -0.30479 | 0.45865 | diagnostic |
| retained update mass | **0.01139** | 0.00319 | 0.05198 | **>= 0.25** |
| cosine to unshrunk rank-200 update | **0.42268** | 0.08568 | 0.96511 | **>= 0.80** |
| finite | 1 | 1 | 1 | 1 |

## Interpretation

The estimator has real discriminatory power: weak-curvature clusters receive
far lower reproducibility weights than strong clusters.  The scientific
problem is that cross-half-batch reproducibility at 1000 walkers is too low to
serve as a multiplicative natural-gradient filter.  The median weight is only
0.097 and the full filtered update retains 1.14% of the unfiltered squared
coefficient mass.  This changes the update direction substantially rather
than merely denoising it.

Consequently, this result does **not** say that the weak retained modes are
physically useless.  It says that estimating their signal by two 500-walker
halves and then multiplying by the raw empirical-Bayes weight confounds weak
but useful signal with non-reproducibility.  Under the agreed stop rule, the
explicit parameter-space envelope/shrinkage branch is closed.

Machine-readable summary:
`/scratch/dexuan1/runs/N2_cluster_snr_rank200_20260809/health_summary.json`.
