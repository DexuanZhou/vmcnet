# WSSR reproducibility bundle

This directory contains the exact KFAC initializer checkpoints used by the
matched C, N2, and H2O experiments in this repository. Each compressed
checkpoint is below GitHub's per-file limit and is stored directly in Git so a
normal clone is sufficient.

## Initializers

| System | Bundled run directory | Epoch | Original source | Source commit |
| --- | --- | ---: | --- | --- |
| C | `kfac_initializers/C_kfac_pre1000` | 1000 | `/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1` | `24093f2de61b4dd9d3bab4b1ee75b47a78799af2` |
| N2, 2.016 Bohr | `kfac_initializers/N2_R2016_kfac_pre5000` | 5000 | `/scratch/dexuan1/runs/old/N2_R2016/e100000_followup/N2eq/N2eq_kfac_pre5000` | `24093f2de61b4dd9d3bab4b1ee75b47a78799af2` |
| H2O | `kfac_initializers/H2O_kfac_pre5000` | 5000 | `/scratch/dexuan1/runs/H2O_kfac_pre5000_N2matched_20260807` | `18b9b03b68a18c0fef1256be690dd98fa3c114e7` |

Each directory deliberately contains both `config.json` and the checkpoint.
VMCNet reloads the model and molecule definition from the former and the
parameters, walkers, amplitudes, PRNG state, and KFAC state from the latter.

Verify the files with:

```bash
sha256sum -c reproducibility/kfac_initializers/SHA256SUMS
```

## Matched N2/H2O WSSR launchers

The exact Slurm pipeline requested for the N2 and H2O comparisons is in:

- `experiments/N2_H2O_WSSR_rank1600_matched_E100000_20260809/`
- `experiments/N2_H2O_WSSR_dualcap_E100000_20260810/`

Each pipeline contains `submit.sh`, a smoke gate, the 50k stage, and the
50k-to-100k resume plus frozen-evaluation stage. The historical scripts retain
their original `/scratch/dexuan1/...` paths so that the recorded jobs remain
exactly reproducible. On another checkout, point their `SOURCE` variables to
the matching bundled run directory above and set `SNAPSHOT`/`REPO` to the new
checkout.

## Recent C spectral-history tests

The two most recent, not-yet-submitted C tests are kept in:

- `experiments/C_current_subspace_spectral_history_E5000_20260812/`
- `experiments/C_anisotropic_matrix_history_E5000_20260812/`

Both use `kfac_initializers/C_kfac_pre1000` as their logical initializer. Their
launch scripts retain the original cluster path for the same reason described
above.
