# N2 rank-800 error-feedback stress test (retry 1)

This is a paired N2, R=2.016 Bohr screen starting from the same completed
KFAC-pre5000 checkpoint.  All arms restore the same 1000 training walkers,
amplitudes and PRNG state, create a fresh optimizer state, and do not reburn.
They run 1000 optimization updates with ten MCMC moves per update and save
checkpoints at E500 and E1000.

Arms A and B use rank-800 full-current-batch Galerkin residual recurrence,
solution recurrence mu 0.95, SSI 40/2, device Cholesky, lr 0.002 with inverse
time decay 1e-4, fixed Tikhonov lambda 1e-3 and norm constraint 1e-3.  Arm B
alone enables parameter-space error feedback.  Its previous feedback is
multiplied by 0.95 before reinjection, and its norm is capped at ten times the
current sample-space residual norm.  Arm A retains EF off.  The SPRING control
uses the clean paper-era commit `18b9b03` with lr 0.002, mu 0.95, damping 1e-3,
the same schedule and norm constraint.  The WSSR arms use the immutable retry-1
snapshot containing EF and the old-config reload fix.

Each of the six endpoints receives an evaluation-only frozen calculation with
2000 fresh walkers, 10000 burn-in steps, 2000 measurements and ten MCMC moves
between measurements.  Primary outcomes are frozen variance at E500/E1000,
the EF cap-trigger frequency, and the logged feedback health ratios.  Because
`||m||/||zeta||` compares parameter and sample spaces, the run also records
`||m||/||O_bar.T@zeta||` as the coordinate-consistent parameter-space ratio.
