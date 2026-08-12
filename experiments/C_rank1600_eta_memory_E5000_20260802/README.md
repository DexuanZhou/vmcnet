# C rank-1600 long-memory E5000 causal test

This experiment is the training-stage decision gate following the early-memory
offline audit.  It starts every arm from the same C KFAC-pre1000 checkpoint and
changes only the WSSR history decay:

- control: constant `eta=0.3`;
- candidate: target `eta=0.99` with
  `eta_t=min(0.99, 1-1/t)` startup bias correction.

Both arms use rank 1600, SSI 40/2, fixed Tikhonov `lambda=1e-3`, relative
cutoff `3e-4`, learning rate `0.04`, inverse-time decay `1e-4`, Euclidean norm
constraint `1e-3`, no complement, 1000 walkers, ten MCMC moves per update,
`reburn=False`, a new WSSR optimizer state, and the same isolated fp64
rank-space solve.  Pairing is by folded checkpoint PRNG key for seeds 0 and 1.

The training endpoint is WSSR epoch 5000 (5000 WSSR updates after the shared
1000-step KFAC preliminary phase; `append=False` resets the optimizer epoch
counter).  Each checkpoint is then evaluated
with the already validated light frozen protocol: 1000 fresh walkers, 5000
burn-in moves, 2000 measurements, ten MCMC moves per measurement.

The offline residual is not a veto.  The primary decision variable is paired
frozen local-energy variance at epoch 5000.  A positive result requires both
paired seed ratios to be below one and their geometric mean to be at most 0.9.
A geometric mean below one without 2/2 seed agreement is promising but
inconclusive.  A ratio at least one is negative.

Every training step is checked for non-finite energy, variance, WSSR direction,
or optimizer finite flag.  After the first 20 updates, variance or raw direction
norm exceeding 100 times its first-20 median triggers an early stop.
