# N2 matched-capacity history causal test

Date: 2026-08-09

## Question and controls

The test holds the maximum Rayleigh--Ritz capacity at 96 and changes only its
allocation between current and historical SSI directions:

| Arm | Current directions | Historical directions | EF | Complement |
|---|---:|---:|---|---|
| H96 | 96 | 0 | off | off |
| E96 | 32 | at most 64 | off | off |

Both start from the same equilibrium-N2 KFAC-pre5000 checkpoint.  The matched
protocol is 1000 walkers, 10 MCMC steps/update, clipping 5, reburn false,
learning rate 0.002 with inverse-time decay 1e-4, fixed Tikhonov lambda 1e-3,
norm constraint 1e-3, and SSI 40/2.

## E1000 frozen result

| Arm | Frozen energy (Ha) | s.e. (Ha) | Frozen variance | Variance / H96 |
|---|---:|---:|---:|---:|
| H96 | -109.480021 | 0.001096 | 4.80512 | 1.000x |
| E96 | -109.478416 | 0.001257 | 6.32302 | 1.316x |

Both arms retained numerical rank 96.  The tail-100 median regularized Ritz
condition numbers were 443.9 for H96 and 1793.3 for E96.  The completed Slurm
wall times, including compilation and checkpoint serialization, were 4:53 for
H96 and 4:16 for E96.

## Causal interpretation

History is harmful at fixed 96-dimensional capacity: replacing 64 current
directions by two rank-32 historical blocks increases frozen variance by 31.6%.
The 1.61 mHa energy difference is only comparable to the combined Monte Carlo
standard error, so the variance is the stronger endpoint here.  The existing
hard current-only rank-200 control has E1000 frozen variance 3.11845; therefore
both effects are present: indiscriminate history is harmful, and 96 total
directions are insufficient relative to 200 current directions.

This result activates the preregistered fixed-budget historical-selection
branch.  Current directions must be protected; historical novelty may enter
only by competing for explicit remaining slots using current-batch predicted
descent value.

## Jobs and artifacts

- Training array: 53857263
- Successful retry training array: 53858053
- Frozen-evaluation array: 53858582
- First submission `53857263` failed before training because new config fields
  cannot be injected directly into the paper-era reload config.
- Retry run root: `/scratch/dexuan1/runs/N2_envelope_capacity_causal_20260809_retry1`
