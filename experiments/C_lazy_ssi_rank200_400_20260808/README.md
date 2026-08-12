# C low-rank lazy-SSI screen

This experiment tests whether a genuinely fixed-subspace fast path can make
the full-current-batch Galerkin recurrence competitive with SPRING in wall
time.  It is deliberately distinct from the rejected 2026-08-07 delayed-Ritz
pilot: skipped steps here keep `U` fixed and do **not** run QR, an eigensolve,
or an in-subspace Ritz rotation.  They only evaluate the current
`W = O_current.T @ U`, the full residual, and the device Galerkin solve.

All arms start from the matched C KFAC-pre1000 checkpoint with 1000 walkers,
10 MCMC steps/update, `reburn=False`, fixed lambda `1e-3`, learning rate
`0.04/(1+1e-4 t)`, norm constraint `1e-3`, solution recurrence `mu=.99`, and
seed 0.  The seven 200-update timing arms are:

- matched SPRING from the same immutable current source snapshot;
- rank 200 and 400 with refresh period 1;
- rank 200 and 400 with fixed-basis refresh periods 10 and 20.

The first 20 updates are excluded from steady-state timing.  Only a lazy arm
whose measured mean time/update is no greater than SPRING and whose short
trajectory remains finite is eligible for the later E5000 frozen-variance
screen.  No 100000-step run is part of this experiment.
