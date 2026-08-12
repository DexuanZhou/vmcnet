# C anisotropic matrix-history WSSR screen (pure suggestion 1)

This is a prepared, **not submitted** test of suggestion 1 alone.  It retains
the original WSSR augmented factor, so historical directions still enter SSI
and can alter the selected subspace.  The only algorithmic change is replacing
the scalar matrix-history weight by

```text
noise = noise_scale * lambda_1 / sqrt(N_walkers)
SNR_i = lambda_i / noise
delta_i = eta_max / (1 + SNR_i)
```

The current block uses `1 - sum(delta_i lambda_i) / sum(lambda_i)`.  This is
the spectral-mass-weighted scalar required because current sample columns do
not correspond one-to-one with historical parameter eigenmodes.  It recovers
the legacy `1-eta` scaling exactly when all `delta_i` are equal.

All arms use current RHS (`eta_g=0`) and otherwise share C KFAC-pre1000,
1000 walkers, rank 800, SSI40/2, fixed lambda `1e-3`, lr `0.04`, and the same
Euclidean norm constraint.

| arm | matrix history | eta |
|---|---|---:|
| `current_only` | none | 0 |
| `uniform_matrix` | legacy scalar augmented-factor EMA | 0.3 |
| `anisotropic_matrix` | augmented factor with per-mode delta_i | max 0.95 |

Prepared only; no job has been submitted.  If approved later, run the training
array, then the evaluation array with an `afterok` dependency, then `collect.py`.
