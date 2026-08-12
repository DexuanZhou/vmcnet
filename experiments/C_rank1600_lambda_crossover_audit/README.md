# C rank-1600 WSSR same-checkpoint lambda crossover

This read-only follow-up was selected after the pre-registered history-
compression audit entered its `raw_full_not_robust` reconciliation branch.

Stage 1 showed that recursive factors preserved raw operator actions to roughly
`4e-4` relative error and raw force direction above `0.99994` cosine.  It also
showed a sharp checkpoint-family difference: the legacy configuration used an
effective Tikhonov lambda near `4.8e-6`, while the fixed-lambda configuration
used `0.001`.  This audit removes checkpoint state as a confound.

## Fixed protocol

- frozen C parameters; no parameter or optimizer-state persistence;
- 1000 walkers, `eta=0.3`, rank 1600;
- actual production right-warm SSI: 40 initial, 2 warm iterations;
- depths K=2 and K=3, followed by an independent held-out batch;
- six replay replicates at each of:
  - legacy epoch 100000;
  - fixed-lambda epoch 102000;
- a single SSI decomposition and a single recurrent state are shared between
  both candidates in each step.

Only the inverse coefficient changes:

- `dynamic_lambda`: legacy rule `(damping * sigma_1)^2`;
- `fixed_lambda`: `0.001`.

The relative singular-value cutoff remains `0.0003` for both candidates.
Therefore this is a lambda crossover, not a cutoff/rank/SSI comparison.

## Primary gate

For each checkpoint and depth, fixed lambda must improve held-out residual ratio

`q = ||e_holdout - O_holdout^T d|| / ||e_holdout||`

relative to dynamic lambda with:

- median relative improvement >= 5%;
- bootstrap 95% lower bound > 0;
- win fraction >= 0.75;
- positive held-out force dot fraction >= 0.95.

The causal claim “tiny dynamic lambda amplifies otherwise small SSI/operator
errors” is supported only if the K=3 gate passes at both checkpoint states.  If
it fails, the next suspect is warm-2 inverse-direction error, not recursive
history compression.
