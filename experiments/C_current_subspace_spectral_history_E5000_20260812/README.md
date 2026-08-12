# C current-subspace spectral-history WSSR screen

This directory is a prepared, **not submitted** paired E5000 screen.  It tests
whether WSSR should use the current batch to choose the SSI subspace and use
history only to smooth the reduced metric/spectrum.

All arms start from the same C KFAC-pre1000 checkpoint and use 1000 walkers,
10 MCMC steps/update, rank 800, SSI 40/2, fixed Tikhonov lambda `1e-3`, current
gradient (`eta_g=0`), learning rate `0.04` with inverse-time decay `1e-4`, and
the same Euclidean norm constraint `1e-3`.

| arm | subspace/operator history | reduced metric history | eta_S |
|---|---|---|---:|
| `current_only` | current batch only | none | 0 |
| `full_matrix_ema` | legacy augmented-factor EMA | full matrix | 0.3 |
| `uniform_spectrum` | current batch only | diagonal Ritz spectrum | 0.3 |
| `cluster_adaptive` | current batch only | near-degenerate blocks, adaptive | max 0.95 |

The adaptive arm uses one history weight per near-degenerate cluster.  The
weight is larger in sample-noise-dominated tail modes and is suppressed when
the transported historical reduced block disagrees with the current block.
No gradient history is used by any arm.

Prepared commands (do not run until the experiment is approved):

```bash
sbatch experiments/C_current_subspace_spectral_history_E5000_20260812/train_array.sh
# After training succeeds, submit eval_array.sh with an afterok dependency.
python experiments/C_current_subspace_spectral_history_E5000_20260812/collect.py
```

Unit tests are local and do not submit jobs:

```bash
bash experiments/C_current_subspace_spectral_history_E5000_20260812/test.sh
```
