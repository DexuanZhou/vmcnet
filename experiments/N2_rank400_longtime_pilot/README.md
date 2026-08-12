# N2 rank400 long-time pilot

This dependency-gated pilot starts only after collector 50016405 terminates.
It validates and replays the rank400 beta=0.2 epoch-30000 checkpoint, then a
release job submits only Stage-1-eligible E500 variants. A second collector
releases frozen evaluations only for scientifically stable trajectories.

A--D restore the exact rank400 warm-SVD and Optax state. E and F retain that
history and Optax schedule, zero-pad only its storage to rank 600, and perform
one exact rank-600 decomposition of the fixed first augmented operator. All
model parameters, walkers, amplitudes, and PRNG state come from the same source
checkpoint.

All new optimizer controls are default-disabled. Mode `none` retains the
existing update path.
