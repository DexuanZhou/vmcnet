# C stage-wise SPRING -> WSSR switch scan

This low-cost causal scan asks when the late-basin WSSR refinement advantage
first appears.  It reuses the completed C SPRING replicate-2 trajectory and
does not retrain any SPRING control.

Two WSSR branches are trained for 5000 updates:

- SPRING epoch 5000 -> WSSR epoch 5000, compared with SPRING epoch 10000;
- SPRING epoch 20000 -> WSSR epoch 5000, compared with SPRING epoch 25000.

The already-completed epoch-50000 branch is the third switch point and is not
rerun.  WSSR uses the same configuration that won that late-checkpoint screen:
rank 1600, shared history weight 0.3, fixed Tikhonov lambda 0.001, hard-only
update, and a globally continuous `0.04 / (1 + 1e-4 epoch)` learning-rate
schedule.  Each branch starts a fresh WSSR optimizer state while retaining the
exact SPRING parameters, walkers, amplitudes, and PRNG state; `reburn=False`.

The paired light frozen evaluation uses 1000 fresh walkers, 5000 burn-in
steps, 2000 measurements, and ten MCMC steps per measurement.  Advancement is
based on both frozen variance and measured wall time.  No 100000-step run is
authorized by this experiment.
