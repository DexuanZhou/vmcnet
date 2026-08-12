# H2O N2-matched optimizer comparison

This pipeline applies the established N2 molecular protocol to equilibrium
water. The geometry, in Bohr, is

- O: `(0, 0, 0)`
- H: `(+1.43233673, 0, +1.10715266)`
- H: `(-1.43233673, 0, +1.10715266)`

This corresponds to an O--H distance of 1.81035 Bohr and an H--O--H angle of
104.594 degrees. The electronic configuration is the singlet `(5,5)`. A common KFAC
preliminary run uses 1000 walkers for 5000 steps at learning rate 0.05. Both
main runs reload its epoch-5000 parameters and walker state with a fresh
optimizer state and `reburn=False`. SPRING uses `lr=0.002`, `mu=0.95`; MinSR
is implemented by the same update with `lr=0.02`, `mu=0`. Both use damping
0.001, norm constraint 0.001, inverse-time decay 1e-4, 1000 walkers, ten MCMC
steps per update, and 100000 optimization steps. Frozen evaluation uses 2000
walkers, 10000 burn-in steps, 20000 measurements, and ten MCMC steps between
measurements.

Submit with:

```bash
bash submit.sh
```

The optimizer array is held on an `afterok` dependency on the preliminary
KFAC job, so SPRING and MinSR necessarily use the same checkpoint.

## Additional controls submitted on 2026-08-07

The formal N2 KFAC controls were checked before adding the water KFAC arm.
They reload a common KFAC-pre5000 checkpoint, create a fresh optimizer state,
set `reburn=False`, and then train KFAC for 100000 further updates at learning
rate 0.05.  `kfac_after_pre5000_100k.sh` applies that same protocol to water;
it is not a random-initialization KFAC run.

The original water MinSR trajectory terminated at epoch 28082 after detecting
NaNs.  `retry_minsr_seed1.sh` is an independent-reload replicate: it preserves
the exact pre5000 parameters and walkers, creates a fresh MinSR state, and
replaces only the checkpoint PRNG key with seed 1.  Reusing the original key
would risk deterministically reproducing the same failed trajectory and would
not test whether the instability is sample-path dependent.
