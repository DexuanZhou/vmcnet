# Strict FermiNet multilevel smoke test

This is a numerical plumbing test, not an energy comparison.  It trains a
small C FermiNet for two updates, embeds it exactly into a wider/deeper model,
prolongs the active SPRING or WSSR optimizer state, and continues for ten
updates with `reburn=False` and without resetting the optimizer.

Smoke hierarchy (fixed 16 determinants):

- coarse: `((32, 8), (32,))`
- fine: `((48, 12), (48, 12), (48,))`

The checkpoint converter rejects the transition unless signs agree and the
maximum log-amplitude error on stored walkers is below the dtype tolerance.

## Gated scientific pilot (not submitted before smoke passes)

The first accuracy experiment will use a common C coarse KFAC-pre1000 state and
fixed 16 determinants throughout:

- L0: `((64, 8), (64,))`
- L1: `((128, 16), (128, 16), (128,))`
- L2: `((256, 16), (256, 16), (256, 16), (256,))`

For each optimizer, the multilevel arm runs 500 updates at L0, 500 at L1, then
1000 at L2.  The matched control embeds the same L0 KFAC checkpoint directly
into L2, initializes the same optimizer, and runs 2000 L2 updates.  Thus the
starting wavefunction, walkers, PRNG key, total optimizer updates, learning-rate
schedule, and final architecture agree; only the timing of capacity activation
differs.  SPRING and WSSR are compared within optimizer, not by conflating
architecture scheduling with different starting checkpoints.

The pilot is allowed to start only if both smoke arms show:

1. zero sign mismatches and log-amplitude error within dtype tolerance;
2. unchanged optimizer schedule count at conversion;
3. unchanged norm of every transported history object (up to serialization
   tolerance);
4. exactly ten successful fine-level updates with no reburn and no state reset.
