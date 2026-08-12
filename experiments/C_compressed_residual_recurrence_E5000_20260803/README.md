# C compressed residual-recurrence E5000

This experiment tests whether the only replicated positive WSSR mechanism,
full-solution residual recurrence, survives a genuine current-batch rank
compression.  With 1000 walkers the centered score matrix has rank at most
999, so the two arms use ranks 400 and 800 rather than the uncompressed
rank-1600 setting.

Both ranks use two paired seeds and start from the same KFAC-pre1000
checkpoint.  The current force is computed before projection; only the
residual correction is restricted to the retained singular subspace.  All
other settings match the positive rank-1600 recurrence experiment:
`mu=0.99`, `eta_S=eta_g=0`, fixed Tikhonov lambda `1e-3`, SSI 40/2,
learning rate `0.04`, inverse-time decay `1e-4`, norm constraint `1e-3`,
1000 walkers, ten MCMC moves per update, and `reburn=False`.

The primary scientific endpoint is light frozen variance at epoch 5000.  The
efficiency endpoint is frozen variance versus measured Slurm wall time.  A
compressed arm is useful only if it is competitive with SPRING at equal wall
time; matching an epoch count alone is insufficient.

Final status: rank400 was rejected after seed0 frozen evaluation; rank800 was
evaluated with both seeds.  See the run-root `report.md` and `summary.json`.
