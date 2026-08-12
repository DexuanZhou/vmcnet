# C rank-1600 early memory-scaling audit

This is a read-only, fixed-parameter replay audit at the common C
KFAC-pre1000 checkpoint. It never updates or writes model parameters, sampler
state, optimizer state, or source checkpoints.

## Source audit resolved before this experiment

The production WSSR implementation already uses `eta` as the operator-history
EMA weight. It augments the current score matrix and residual with

`[sqrt(eta) * F_history, sqrt(1-eta) * O_current]`

and

`[sqrt(eta) * e_history, sqrt(1-eta) * e_current]`.

The score factor and residual coordinates are compressed by the same retained
SVD mask. Therefore this experiment does not add a second `rho`; it measures
whether the existing compressed recurrence benefits from longer memory.

## Fixed protocol

- C atom, spin sector `(4, 2)`.
- Common KFAC-pre1000 checkpoint, 1000 walkers.
- Parameters remain fixed.
- Rank/storage/working rank 1600.
- SSI iterations: 40 initially, 2 after warm start.
- Damping and relative cutoff: `0.0003`.
- Hard complement: complement weight zero.
- Six paired replay streams per arm.
- History depths `K = 3, 5, 8, 16`.
- Four subsequent batches form an independent future-force consensus.

The seven arms use exactly the same replay sampling keys:

1. current-batch-only WSSR;
2. recursive WSSR, constant `eta=0.3`;
3. recursive WSSR, constant `eta=0.8`;
4. recursive WSSR, constant `eta=0.95`;
5. recursive WSSR, constant `eta=0.99`;
6. recursive WSSR with `eta_t=min(0.99, 1-1/t)`;
7. exact sample-space SPRING recurrence with `mu=0.99` and damping `0.001`.

All WSSR directions use fixed `lambda=0.001`. The spectral solve and the
history/current force combination use mixed fp64 arithmetic; the network,
sampling, score matrix and stored compressed factor remain in their production
dtype. This isolates the memory-length question from the previously identified
dynamic-lambda and cancellation effects.

## Primary metric and pre-registered gate

For an update `d`, define the future consensus force as the mean force from the
four batches after depth 16. The primary metric is

`cos(d, force_future)`.

A long-memory WSSR arm is harmless and may be tested at E5000 only if, relative
to the paired `eta=0.3` short-memory control:

- median cosine change is at least `-0.01`;
- the paired bootstrap 95% lower bound is at least `-0.03`;
- at most 1/6 replay streams are worse by more than `0.02`; and
- median compressed-force cosine to the full augmented force is at least
  `0.98`, with median relative error at most `0.05`; and
- median randomized history spectral-quality retention is at least `0.80`.

Held-out SR residuals, update coherence, history/current force ratios, norm-cap
scales, and active ranks are secondary diagnostics. They do not override the
primary direction gate.

Held-out residual never vetoes a candidate. Among harmless long-memory arms,
the one with the highest K=16 future-consensus cosine is promoted to an E5000
frozen-variance test. SPRING is a
mechanism comparator and is not itself a WSSR promotion candidate.
