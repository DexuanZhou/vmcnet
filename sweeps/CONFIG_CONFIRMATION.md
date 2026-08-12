# Pilot configuration confirmation

1. Learning-rate schedule: all three optimizer templates explicitly use `inverse_time`, i.e. `lr(t) = learning_rate / (1 + learning_decay_rate * t)`, with `learning_decay_rate=1e-4`.
2. Clipping: the repository field is `config.vmc.clip_threshold`; this pilot sets it to `5.0` and uses `clip_center=mean`. The implementation clips in units of mean absolute deviation, despite the shorthand request “std=5”.
3. MCMC work per update: `config.vmc.nsteps_per_param_update=10`.
4. Geometry units: `problem.ion_pos` is in atomic units (Bohr). C is at the origin; the target N2eq protocol uses bond length 2.068 Bohr, at z = +/-1.034 Bohr.
5. Electron convention: `problem.nelec=(n_up,n_down)`. Neutral C is `(4,2)`; singlet neutral N2 is `(7,7)`.
6. Checkpoint slimming: `config.vmc.disable_checkpointing` exists but disables all checkpoint writes; there is no native “keep last only” switch. Each template schedules a regular final checkpoint and, after successful completion, deletes `best_checkpoint.npz` and any non-final regular checkpoints. WSSR also sets `store_warm_u=False`, the current state/checkpoint slimming option that omits the left warm-start singular vectors (the repository default remains `True`).

The 500-epoch common main-stage setting applies to the 11 SPRING/WSSR runs. The two named preliminary runs intentionally use 1000 and 2000 epochs, respectively, producing `checkpoints/1000.npz` and `checkpoints/2000.npz` for the specified reload chain.
