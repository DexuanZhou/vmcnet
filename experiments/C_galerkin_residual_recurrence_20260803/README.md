# C Galerkin-consistent sketched residual recurrence

This staged experiment upgrades the existing rank-coordinate residual
recurrence to a current-batch Galerkin correction.  Every arm reloads
`/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1/checkpoints/1000.npz`, folds
a paired key into the restored sampler state, sets `reburn=False`, and creates
a fresh optimizer state.

## Score-matrix convention

The paper notation is `O_bar in R^(N x N_p)`.  VMCNet's
`center_and_scale_score_matrix` stores its transpose as `o_cur = O_bar.T` with
shape `(N_p, N)`.  Therefore the implementation maps

- `O_bar @ prior` to `o_cur.T @ prior`;
- `O_bar @ U_r` to `o_cur.T @ basis`;
- `O_bar.T @ residual` to `o_cur @ residual`.

The Galerkin solve is performed in host fp64 and returned to the parameter
dtype.  `subspace_eta_S`, error feedback, dual-mode diagnostics, recurrence
telemetry, and drift monitoring all default to disabled in the library.

## Fixed matched protocol

- C (4,2), 1000 walkers, ten MCMC moves/update, clipping inherited as 5;
- learning rate `0.04/(1+1e-4*t)`, norm constraint `1e-3`;
- Tikhonov lambda `1e-3`, relative cutoff `3e-4`;
- SSI 40 initially and 2 warm iterations;
- no 100k arm.

Stages are represented by immutable TSV manifests.  A stage is submitted only
after the previous stage's decision has been collected.
