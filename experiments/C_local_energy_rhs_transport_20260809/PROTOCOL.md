# C local-energy RHS transport audit (pre-registered)

This is a diagnostic-only experiment.  It does not modify an optimizer and it
must not launch a training follow-up automatically.

## Matched setup

- Source: `C_kfac_pre1000_1/checkpoints/1000.npz`
- 1000 walkers, 10 MCMC steps per parameter update, mean clipping at 5 sigma
- WSSR: rank 1600, eta 0.3, lr 0.04, inverse-time decay 1e-4,
  Tikhonov lambda 1e-3, norm constraint 1e-3, SSI 40/2
- SPRING: mu 0.99, lr 0.02, inverse-time decay 1e-4, damping 1e-3,
  norm constraint 1e-3
- two independently keyed trajectories per optimizer; 20 real updates each
- displacements use the actual post-learning-rate, post-norm-constraint
  parameter difference for k in {1, 5, 20}
- walkers are saved after MCMC at theta_(t-k), before that step's update
- local-energy JVP, finite displacement and parameter-space bridges use fp64
  and microbatches

## Quantities

With P the sample-centering projector and old walkers fixed,

```
R_J     = ||(I-P) J delta_theta|| / ||(I-P) epsilon||
R_FD    = ||(I-P) [E_L(theta_t)-E_L(theta_old)]|| / ||(I-P) epsilon||
R_mean  = ||P J delta_theta|| / ||J delta_theta||
R_param = ||O_bar^T (I-P) J delta_theta|| / ||O_bar^T epsilon_bar||
```

The report also includes the centered JVP/finite-displacement cosine and
relative linearization error, plus the same-parameter-space Fisher proxy
`||S_bar delta_theta|| / ||g||`.

For the solve-level K=2 audit, the raw and corrected systems use exactly the
same historical/current score factors and eta weights.  Only the historical
RHS changes:

```
epsilon_hist -> epsilon_hist + (I-P) J delta_theta.
```

Both candidates are evaluated on an independent current-parameter held-out
batch.  Historical O is never transported in this experiment.

## Decision rule

- update-direction cosine below 0.98 means only that transport materially
  changes the solve; it is not evidence of benefit;
- a later short training test is scientifically admissible only if held-out
  residual improves by at least 10% in both independent WSSR replicates with
  the same sign, and the first-order linearization is not invalidated;
- a large R_J with large linearization error rejects first-order transport;
- a large direction change without held-out improvement means that staleness
  enters the solve but its correction is not useful;
- a small R_J excludes local-energy RHS staleness as the leading mechanism but
  does not imply that WSSR is impossible.

The collector reports the decision and stops.  No training job is chained.
