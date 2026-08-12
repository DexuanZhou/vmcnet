# N2 R=2.068 paper-style optimizer controls

Both jobs independently reload the same validated 1000-walker KFAC-pre5000
checkpoint:

`/scratch/dexuan1/runs/kfac_pre/N2eq_R2068_kfac_pre5000/checkpoints/5000.npz`

They restore parameters, walkers, amplitudes, and PRNG state, create a new
optimizer state, and do not reburn the restored training walkers. The only
optimizer-level comparison is SPRING (`lr=0.002`, `mu=0.99`) versus KFAC
(`lr=0.05`). Both run 100000 training epochs and then the same frozen
evaluation with 2000 walkers, 10000 burn-in steps, and 20000 measurements.

Geometry is intentionally R=2.068 Bohr rather than the paper's R=2.016 Bohr.
