# C exact-first and complement validation

All runs independently reload `/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1/checkpoints/1000.npz`, including its parameters, 1000 walkers, amplitudes, and PRNG state. They create a new WSSR optimizer state and do not reburn.

Stage 1 runs four matched 10-step trajectories with expensive exact-reference diagnostics enabled. Stage 2 is intentionally not submitted until the Stage-1 comparison and complement-safety gate pass.
