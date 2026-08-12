# N2 adaptive-complement validation

This package isolates adaptive complement on N2 R=2.068 Bohr after the validated
4096-chain KFAC-pre5000 checkpoint. No exact-first, smooth, cluster, near-tail,
or force-aware option is enabled. The fixed scalar complement is zero for hard
truncation. In adaptive mode, `complement_weight=1e-4` supplies only the nominal
ceiling passed to the beta cap; the fixed-complement update branch is not used.

Dependency order: checkpoint validation -> fixed snapshot replay -> 12 matched
E500 trajectories -> 12 frozen evaluations (4096 chains, burn 10000, 1000 eval
epochs) -> collector. No E50000 job is submitted.

Frozen evaluation uses `vmc.nepochs=0`; a lightweight Adam optimizer object is
allocated solely to avoid retaining a multi-gigabyte, unused WSSR state during
evaluation. It never updates parameters and does not affect sampling or energy.
