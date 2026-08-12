# C rank-1600 WSSR history-compression fidelity audit

This is a read-only, fixed-parameter audit.  It never updates or writes model
parameters, optimizer state, walkers, or source checkpoints.

## Question

The preceding history-refresh audit found that combining two independently
sampled batches at fixed parameters improved held-out SR residuals at all three
audited checkpoints, while refreshing the one-step-old Jacobian did not pass its
pre-registered gate.  This audit asks where that multi-batch benefit is lost, if
it is lost at all:

1. the one-shot rank-1600 cap;
2. recursive rank-1600 compression;
3. the configured cutoff used in the production history state; or
4. the production warm-SSI approximation (40 initial iterations, then 2 warm
   iterations).

No `eta` or learning-rate sweep is part of this audit.

## Fixed protocol

- C checkpoint parameters remain frozen.
- 1000 walkers per independently advanced batch.
- `eta=0.3` and the checkpoint's configured Tikhonov/cutoff controls.
- Batch depths `K=2` and `K=3` in stage 1.
- Six independent replay replicates per checkpoint family.
- Checkpoints:
  - legacy dynamic-lambda epoch 100000;
  - fixed-lambda (`lambda=0.001`) epoch 102000.
- A fourth independently advanced batch is held out from every candidate.

For depth `K`, raw batches are weighted exactly as the WSSR recurrence would
weight them:

`[eta^(K-1), (1-eta) eta^(K-2), ..., (1-eta)]`.

## Candidates

- `single_current_exact`: exact configured Tikhonov solve of batch `K` only.
- `raw_full_exact`: exact configured solve of the uncompressed weighted raw
  `K`-batch operator.
- `raw_rank1600_exact`: one-shot exact top-1600 truncation of that same raw
  operator.
- `recursive_factor_rank1600_exact`: recursively stores the unfiltered
  top-1600 factor and projected RHS, while applying the configured cutoff only
  to the candidate update.
- `recursive_production_exact`: exact spectral decomposition but the history
  factor and RHS are masked exactly as the standard production state is masked.
- `recursive_production_ssi`: the actual right-warm production path with 40
  initial and 2 warm subspace iterations.

`raw_full_exact` is not called a full physical SR reference: it is the full
sample-space solve for the finite batches and configured cutoff/Tikhonov rule.

## Pre-registered metrics and gates

Primary metric: held-out residual ratio

`q = ||e_holdout - O_holdout^T d|| / ||e_holdout||`.

Raw multi-batch evidence requires, separately at each checkpoint/depth:

- median relative `q` improvement over `single_current_exact` >= 5%;
- bootstrap 95% lower bound > 0;
- win fraction >= 0.75.

For candidate `c`, retained gain is

`R = (q_single - q_c) / (q_single - q_raw_full)`

on replicates where the raw-full denominator is positive.  A compression stage
passes only if:

- median `R >= 0.80`;
- median direction cosine with `raw_full_exact >= 0.98`;
- positive held-out force dot fraction >= 0.95.

Secondary diagnostics are projected-RHS/force error and force-relevant Gram
probe action error.  They localize a failed primary gate but do not override it.

## Decision branches

Use depth `K=3` for the main branch decision.

1. Raw-full fails: the earlier positive result was not robust to the exact
   configured solve; stop and report that contradiction.
2. Raw-full passes but one-shot rank-1600 fails: the rank cap is the bottleneck;
   audit one-shot rank 2400/3200 only.
3. One-shot passes but unfiltered recursion fails: recursive compression is the
   bottleneck; audit depth 5/8 with storage rank 1600 versus 3200.
4. Unfiltered recursion passes but production-exact fails: configured history
   masking/state encoding is the bottleneck.
5. Production-exact passes but production-SSI fails: warm-2 approximation is the
   bottleneck; audit warm iterations 4/8/40 offline.
6. Production-SSI passes: history compression is not the missing mechanism;
   do not submit another matched training run from this audit.

Only the branch selected by stage 1 may be run next.
