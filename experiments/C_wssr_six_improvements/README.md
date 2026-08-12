# C warm-SVD-right six-improvement ablation

This experiment uses all-electron C `(4,2)`, `nchains=1000`, and the common
KFAC-pre1000 checkpoint:

`/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1/checkpoints/1000.npz`

Common WSSR settings are rank 400 (except explicitly oversampled methods),
`eta=0.8`, learning rate `0.02`, Tikhonov damping `3e-4`, norm constraint
`1e-3`, and two normal warm/SSI iterations. All experimental options are
disabled by default.

## Stages

1. Four frozen snapshots compare baseline and all internal candidates against
   exact rank-400 and exact full Tikhonov references. Walkers, parameters,
   force, operator, amplitudes, and PRNG state are frozen.
2. Each eligible improvement is independently restarted from KFAC-pre1000 for
   200 epochs. Exact-full replay is measured at snapshots 1, 50, 100, and 200.
3. Baseline and the three best stable improvements are independently run for
   500 epochs with three matched seeds, followed by frozen raw evaluations.

Projected iterative complement is preserved in Stage-1 results but excluded
from training because all screened iteration counts violated the predeclared
`complement/resolved <= 0.5` eligibility rule.

## Authoritative jobs

- Stage 1: `49918208_[0-3]`
- Invariance tests: `49922369` (`15 passed`)
- Stage 2 E200: `49925325_[0-5]`
- Stage 2 exact snapshot replay: `49926055_[0-5]` (each task handles epochs
  50, 100, and 200 to avoid repeated environment startup)
- Stage 2 collector: `49926056`
- Conditional Stage 3 release: `49926057`

See `stage2_manifest.csv` for exact task mapping and output paths. No N2 job is
part of this experiment. Nothing is committed or pushed.
