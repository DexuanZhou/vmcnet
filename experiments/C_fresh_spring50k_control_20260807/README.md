# C fresh-SPRING state control at epoch 50000

This control separates the late learning-rate effect from optimizer-state
history.  It reloads the exact C SPRING replicate-2 epoch-50000 checkpoint,
keeps parameters, walkers, and PRNG state, but creates a fresh SPRING optimizer
state and trains for 5000 updates with `mu=0.99`.

The local schedule is algebraically shifted so that it equals the original
SPRING schedule at every global epoch:

```
(0.02 / 6) / (1 + (1e-4 / 6) * s)
  = 0.02 / (1 + 1e-4 * (50000 + s)).
```

The endpoint is evaluated with the same two frozen seeds already available for
the native SPRING-55000 and WSSR-eta0.3 endpoints.  This produces a three-way
comparison: native SPRING history, reset SPRING history, and fresh WSSR.
