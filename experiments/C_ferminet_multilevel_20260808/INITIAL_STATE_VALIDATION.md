# Formal pilot initial-state validation

The optimizer-initialization array `53772789` reloaded the common L0
KFAC-pre1000 checkpoint and saved checkpoint 1 before applying the first
SPRING/WSSR update.  A leafwise comparison of those two optimizer-specific
checkpoint files found:

- stored epoch: `0` for both;
- all five sampler-data leaves: exactly equal;
- all 18 parameter leaves: exactly equal;
- PRNG key: exactly equal;
- L0 parameter count: 27,624.

Therefore the two methods, and each method's direct/multilevel fork, begin from
the identical wavefunction, walkers, cached amplitudes, and random stream.  Only
the optimizer-state payload differs by method.
