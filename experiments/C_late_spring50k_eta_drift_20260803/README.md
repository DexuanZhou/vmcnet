# C late-checkpoint WSSR memory test

This experiment tests whether the early monotone toxicity of long WSSR
operator history is caused by parameter drift.

All WSSR arms reload the exact same C SPRING replicate-2 epoch-50000
checkpoint, including its 1000 walkers and PRNG state, create a fresh WSSR
optimizer state, do not reburn, and train for 5000 updates. The arms differ
only in `eta = {0.3, 0.8, 0.95}`. The source trajectory's existing epoch-55000
checkpoint is the exact SPRING-continuation control, so it is not retrained.

The WSSR inverse-time schedule is algebraically shifted to the global epoch
50000:

```
0.04 / (1 + 1e-4 * (50000 + s))
  = (0.04 / 6) / (1 + (1e-4 / 6) * s).
```

This avoids accidentally restarting WSSR at its large epoch-zero learning
rate. All four endpoints use a paired light frozen protocol: 1000 fresh
walkers, 5000 burn-in steps, 2000 measurement epochs, and 10 MCMC moves per
measurement.
