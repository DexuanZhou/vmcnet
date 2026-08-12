# N2 spectral-cluster envelope screen (E5000)

Date: 2026-08-09

## Question

Test whether replacing single-vector history by an enclosing subspace built from
the last three rank-32 SSI bases can remove near-degenerate-mode rotation bias
while retaining a cheap low-rank update.

## Matched setup

- System: equilibrium N2, 2.016 Bohr, `(7, 7)` electrons.
- Start: common KFAC-pre5000 checkpoint.
- Training: 1000 walkers, 10 MCMC steps/update, mean-centered clipping at 5,
  `reburn=False`, 5000 updates, seed 0.
- Optimizer controls: inverse-time learning rate 0.002 with decay `1e-4`, fixed
  Tikhonov lambda `1e-3`, Euclidean norm constraint `1e-3`, SSI 40/2.
- Hard control: current-batch hard WSSR, rank 200.
- Envelope candidate: current rank `k=32`, history length `M=3`, decay 0.8,
  alpha 0.2, adaptive `gamma=bar_lambda+lambda`, no S or gradient EMA.
- Frozen evaluation: 2000 walkers, burn-in 10000, 2000 inference iterations,
  10 MCMC steps/measurement.

## Frozen results

Benchmark energy used for the error column: -109.5423 Ha.

| Epoch | Method | Frozen energy (Ha) | s.e. (Ha) | Error (mHa) | Frozen variance | Candidate / hard variance |
|---:|---|---:|---:|---:|---:|---:|
| 1000 | hard rank-200 | -109.4839973 | 0.0012748 | 58.303 | 3.11845 | -- |
| 1000 | cluster envelope | -109.4643396 | 0.0016720 | 77.960 | 6.43994 | 2.065x |
| 2500 | hard rank-200 | -109.4853414 | 0.0012968 | 56.959 | 3.49153 | -- |
| 2500 | cluster envelope | -109.4702277 | 0.0017182 | 72.072 | 6.18574 | 1.772x |
| 5000 | hard rank-200 | -109.4933222 | 0.0012263 | 48.978 | 3.03990 | -- |
| 5000 | cluster envelope | -109.4719428 | 0.0015893 | 70.357 | 5.79229 | 1.905x |

At E5000 the candidate is 21.38 mHa above the paired hard control. The gap is
far larger than the frozen standard errors. The candidate is already worse at
E1000 and does not recover by E5000.

For context only, an existing matched-start N2 SPRING mu=0.95 E1000 frozen
point has energy -109.4876621 Ha and variance 1.79339. At E1000 the envelope
candidate therefore has 3.59x its variance and is 23.32 mHa higher. This is an
external anchor rather than one of the two arms in the present paired job.

## Runtime

Steady training time was estimated from checkpoint timestamps over epochs
500--5000, excluding compilation and shared-filesystem startup.

| Method | Seconds / step | Relative to hard | Relative to known SPRING (~0.091 s/step) |
|---|---:|---:|---:|
| hard rank-200 | 0.1789 | 1.000x | 1.97x |
| cluster envelope | 0.1604 | 0.897x | 1.76x |

The rank-32 envelope is only 10.3% faster than the rank-200 hard control and is
still substantially slower than SPRING.

## Mechanism telemetry

Candidate tail-100 medians at E5000:

| Quantity | Median |
|---|---:|
| envelope numerical rank | 96 |
| history numerical rank | 64 |
| current-basis novelty fraction | 0.03740 |
| gradient capture fraction | 0.96891 |
| cluster update norm | 0.04193 |
| complement update norm | 0.002247 |
| complement / cluster update ratio | 0.05128 |
| mean cluster curvature `bar_lambda` | 35.5889 |
| norm-constraint scale | 1.0 |

The negative result is not caused by numerical collapse of the envelope: all
96 columns remain numerically active, the current basis keeps adding novelty,
and the envelope captures about 97% of the raw gradient norm. Instead, the
candidate replaces mode-wise SR/WSSR spectral weighting with one scalar mean
curvature. It consequently treats a highly anisotropic near-degenerate region
as isotropic. Increasing span coverage does not compensate for losing the
per-mode inverse-curvature information.

## Decision

This particular master update is a clear no-go. Do not extend it to 100k and do
not tune alpha, gamma, M, or k around this scalar-curvature formulation. The
cluster-invariant idea could only merit revisiting if the envelope carries a
small projected curvature matrix `Q^T S_current Q` (or an equivalent Galerkin
solve), rather than a single `bar_lambda`; that would be a materially different
method and should first pass another short screen.

## Artifacts

- Training/frozen outputs: `/scratch/dexuan1/runs/N2_cluster_envelope_E5000_20260809`
- Machine-readable summary: `/scratch/dexuan1/runs/N2_cluster_envelope_E5000_20260809/summary.json`
- Training Slurm job: `53848378`
- Frozen Slurm array: `53848379`

The frozen array records exit code 5 because the wrapper checked unsuffixed
output directories while VMCNet wrote complete results to `_1` directories.
All six `statistics.json` files exist and were used above; this is a wrapper
post-check issue, not a failed evaluation.
