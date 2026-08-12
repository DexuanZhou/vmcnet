# C rank-400 exponential-S legacy-regularization control

This is a controlled follow-up to the completed rank-400 exponential-S run.
It keeps rank 400, the current-gradient RHS (`eta_g=0`), exponential
`eta_S(t)=0.95*(1-exp(-t/1000))`, 1000 walkers, SSI 40/2, and 20000 updates.

The changed settings are:

- learning rate 0.02 instead of 0.04;
- `exact_first=True` instead of false;
- legacy Tikhonov coupling: both independent overrides are set to -1, so the
  relative cutoff falls back to damping 0.0003 and
  `lambda=(0.0003*sigma_max)^2` instead of fixed lambda 0.001.

The run starts from the same C KFAC-pre1000 checkpoint with the same paired
seed and `reburn=False`. A dependent light frozen evaluation records two
million local energies at checkpoint 20000.
