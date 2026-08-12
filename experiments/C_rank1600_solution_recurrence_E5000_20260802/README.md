# C rank-1600 WSSR solution-recurrence causal test

This test isolates SPRING's current-batch projection correction inside the
matched C WSSR setup. Both arms start at the same KFAC-pre1000 checkpoint and
store the same uncompressed full-parameter history with `mu=0.99`.

- `naive`: decayed previous direction plus the standalone current-batch solve;
- `residual`: decayed previous direction plus a solve of the current-batch
  residual after subtracting the visible action of that history.

Operator/RHS history is disabled (`eta=0`). Everything else is matched:
rank 1600, SSI 40/2, fixed Tikhonov lambda `1e-3`, cutoff `3e-4`, learning rate
`0.04`, inverse-time decay `1e-4`, norm constraint `1e-3`, 1000 walkers, ten
MCMC moves per update, `reburn=False`, and isolated fp64 rank-coordinate
arithmetic. The primary endpoint is light frozen variance after 5000 updates.

The residual mechanism advances only if it beats the existing eta=.3 WSSR
control and improves at least 10% over naive recurrence in the paired seed.
