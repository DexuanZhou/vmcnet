# Research code index

This index records the code and test families bundled in the August 2026 WSSR
research snapshot. Historical run outputs are intentionally not part of the
source snapshot; launchers, collectors, diagnostics, and tests are.

## Production and experimental implementation

- `vmcnet/updates/wssr.py`: WSSR solvers, warm/right SVD paths, split `eta_S`
  and `eta_g`, transport and residual-recurrence diagnostics, function-space
  constraints, adaptive complement paths, current-subspace spectral history,
  and anisotropic matrix-history weighting.
- `vmcnet/updates/wssr_experimental.py`: isolated low-rank, complement,
  envelope, and diagnostic candidates which remain opt-in.
- `vmcnet/updates/spring.py`: SPRING reference path and history diagnostics.
- `vmcnet/updates/multilevel.py` and `vmcnet/models/multilevel.py`: exact
  FermiNet embedding/prolongation and multilevel optimizer helpers.
- `vmcnet/train/burst_diagnostics.py`: opt-in short-window telemetry.
- `vmcnet/train/multilevel_checkpoint.py`: multilevel checkpoint transfer.
- `vmcnet/utils/io.py`: optimizer-PyTree leaf checkpoint format used to avoid
  a single ZIP member exceeding 4 GiB.

All new optimizer behavior is guarded by configuration flags; the reference
SPRING path and baseline WSSR behavior remain available.

## Unit tests

The main coverage is in:

- `tests/units/updates/test_wssr.py`
- `tests/units/updates/test_wssr_experimental.py`
- `tests/units/updates/test_spring_history.py`
- `tests/units/updates/test_function_space_constraint.py`
- `tests/units/updates/test_multilevel.py`
- `tests/units/models/test_multilevel.py`
- `tests/units/train/test_multilevel_checkpoint.py`
- `tests/units/train/test_burst_diagnostics.py`
- `tests/units/utils/test_io.py`

Cluster-side experiment tests live under `experiments/`; each experiment keeps
its launch, evaluation, collection, and local shell-test scripts together.

## Reproducible launch pipelines

- Matched N2/H2O rank-1600 WSSR:
  `experiments/N2_H2O_WSSR_rank1600_matched_E100000_20260809/`
- N2/H2O metric-aware dual-cap WSSR:
  `experiments/N2_H2O_WSSR_dualcap_E100000_20260810/`
- H2O KFAC-pre5000, SPRING, MinSR, and KFAC controls:
  `experiments/H2O_N2matched_KFAC5000_SPRING_MinSR_E100000_20260807/`
- C current-subspace spectral-history test:
  `experiments/C_current_subspace_spectral_history_E5000_20260812/`
- C anisotropic matrix-history test:
  `experiments/C_anisotropic_matrix_history_E5000_20260812/`

Older experimental launchers remain under `experiments/` and `sweeps/` so the
negative and positive ablations referenced by `test_mark.md` are auditable.

## Checkpoints

See `reproducibility/README.md`. The exact C, N2, and H2O KFAC initializer
checkpoints and their `config.json` files are bundled under
`reproducibility/kfac_initializers/` and tracked with Git LFS.
