# C delayed-subspace timing pilot

This is a 200-update timing/telemetry pilot, not an accuracy training run.  It
uses the matched C KFAC-pre1000 checkpoint and full-current-batch rank-800
residual recurrence.  Three arms refresh the SSI subspace every 1, 2, or 5
updates.  Intermediate updates keep the stored left subspace but recompute the
current-batch Ritz values, full residual, Galerkin coefficients, and norm-
constrained parameter update.

Ordinary right SSI does not provide an exact cached `O_current.T @ U` on a
refresh step: its `V*Sigma` factor is only approximate until the subspace is
invariant.  The period-1 control therefore retains the explicit Galerkin
matmul and should reproduce the pre-cache P0 timing of approximately 0.1751
s/update.  On a skipped refresh, Ritz pairs are recomputed inside the fixed
stored left subspace; there `V*Sigma == O_current.T @ U` exactly, so periods 2
and 5 reuse that factor without changing the Galerkin equation. Timing excludes
the first 20 updates. No E5000 or 100000-step run is launched from this pilot.
