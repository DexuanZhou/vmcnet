# C adaptive-S indicator audit

This experiment tests whether an online statistic can decide when WSSR should
mix its historical low-rank covariance factor into the current SR operator.
It is read-only: source checkpoints, model parameters, sampler state, and
optimizer state are never modified or rewritten.

## Scientific question

For the same current gradient, compare

```text
d0 = (S_current + lambda I)^-1 g_current
dh = ((1-eta) S_current + eta S_history + lambda I)^-1 g_current.
```

Both directions are passed through the production learning-rate and Euclidean
norm constraint.  On subsequent batches at the same checkpoint, the primary
indicator is the paired predicted-descent gain

```text
G = (g_future dot dh_applied - g_future dot d0_applied)
    / abs(g_future dot d0_applied).
```

Positive `G` means historical-S mixing predicts a better descent direction on
unseen walkers.  This direction-only comparison is used because the production
norm constraint is nearly always active and previous raw-residual gates did not
predict frozen variance.

The secondary noise-versus-drift ratio is evaluated only on the update-relevant
probe space `V=[d0, dh, g_current]`:

```text
noise_sq = 0.25 * ||(S_half_A - S_half_B) V||_F^2
drift_sq = ||(S_current - S_history) V||_F^2
R = drift_sq / noise_sq.
```

`R <= 1` means the history/current discrepancy is no larger than the estimated
full-batch sampling noise; `R >> 1` indicates genuine history lag.  All operator
actions use low-rank factors or score-matrix matvecs; no dense parameter-space
matrix is formed.

## Source regimes

1. `static_pre1000`: construct 16 history batches at fixed KFAC-pre1000
   parameters.  Existing audits say averaging helps in this artificial static
   regime, so this is the positive control.
2. `early_eta095_e{1000,3000,5000}`: checkpoints from the completed S-only
   `eta_S=.95, eta_g=0` trajectory.  This trajectory is known to be worse than
   `eta_S=0`, so a useful indicator must reject mixing in most of this regime.
3. `late_eta{03,08,095}`: completed WSSR branches from the common SPRING-50k
   checkpoint.  These are secondary regime checks because their original eta
   controlled both sides of the historical equation, whereas this audit holds
   the RHS at the current gradient.

All solves use rank 1600, fixed Tikhonov `lambda=1e-3`, learning rate 0.04 (or
the globally shifted late-checkpoint value recorded by the source config), norm
constraint `1e-3`, and four independent replay replicates with four future
batches each.

## Pre-registered decision

The indicator is eligible for an online training gate only if:

- the static positive control has positive median predicted-descent gain;
- at least two of the three early bad-trajectory checkpoints have non-positive
  median gain or median `R > 1`;
- the candidate gain is not dominated by non-finite or negative-baseline-dot
  events.

If these conditions fail, no adaptive training run is launched.  If they pass,
the next stage implements a default-off, hysteretic gate and tests it with the
validated C E5000 frozen endpoint before any 100k run.
