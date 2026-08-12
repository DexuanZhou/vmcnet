# N2 fixed-state Krylov audit

This audit uses the equilibrium N2 KFAC-pre5000 checkpoint and its saved 1000
walkers. It does not advance MCMC or apply a parameter update. For
`m = 4, 8, 16`, it builds the parameter-space Krylov space
`K_m(S, g)` with twice-reorthogonalized Fisher actions, performs the exact
Galerkin solve in that space at fixed `lambda = 1e-3`, and records the
regularized normal-equation and sample-equation residuals. This is an algebraic
cost/benefit audit only; no accuracy claim is inferred from it.
