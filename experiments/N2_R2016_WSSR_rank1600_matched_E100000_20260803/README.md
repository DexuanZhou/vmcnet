# N2 R=2.016 matched rank-1600 WSSR E100000

This run supplies the missing WSSR arm for the existing matched N2 comparison.
It restarts from the same KFAC-pre5000 checkpoint used by the completed KFAC,
MinSR, and SPRING runs and uses the same sampling, learning-rate schedule, norm
constraint, checkpoint cadence, and frozen-evaluation protocol.

The optimizer-specific WSSR settings are:

- right-warm WSSR, rank/storage/working rank 1600;
- SSI 40 iterations for initialization and 2 warm iterations;
- S-history weight `eta_S=0.2` and current-gradient RHS `eta_g=0`;
- learning rate `0.002`, inverse-time decay `1e-4`;
- fixed Tikhonov lambda `0.001` with relative singular-value cutoff `0.0003`;
- hard-only retained-subspace update (`complement_weight=0`);
- no solution recurrence, gradient transport, adaptive complement, or
  diagnostic reference solve.

Because the H100 queue has a 12-hour wall-time limit, training is split at
epoch 50000. Stage 2 restores the complete optimizer, WSSR warm-subspace,
walker, amplitude, and PRNG state and continues to epoch 100000 without
reburning. It then runs the matched N2 frozen evaluation (2000 walkers, 10000
burn-in steps, 20000 measurements, 10 MCMC steps per measurement).

The scripts run from a read-only source snapshot recorded at submission time so
that later edits to the development worktree cannot change stage 2.
