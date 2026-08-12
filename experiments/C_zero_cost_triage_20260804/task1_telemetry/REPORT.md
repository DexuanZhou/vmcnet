# Task 1: P0 correctable-ratio telemetry

> Correction (2026-08-07): the quantity below is an action ratio rather than
> an orthogonal-projection fraction. It must not be read as a captured
> percentage. See `CORRECTION.md` for the dimensionally valid replacements.

No computation was launched.  This report reads the completed P0 logs under
`/scratch/dexuan1/runs/C_galerkin_residual_recurrence_20260803/`.

The logged `wssr_residual_correctable_ratio` is
`||W_t^T zeta_t|| / ||zeta_t||`.  The denominator for the second ratio is
reconstructed exactly from the logged clipped training variance because
`epsilon_bar = (E_L - mean(E_L))/sqrt(N)`, hence
`||epsilon_bar|| = sqrt(variance)`.

| seed | window | points | median `||W^T zeta||/||zeta||` | median `||zeta||/||epsilon_bar||` |
|---:|:---|---:|---:|---:|
| 0 | E0-500 | 500 | 1.00709 | 1.25616 |
| 0 | E2450-2550 | 101 | 0.794511 | 1.19748 |
| 0 | E4900-5000 | 101 | 0.559087 | 1.16228 |
| 1 | E0-500 | 500 | 1.00229 | 1.25892 |
| 1 | E2450-2550 | 101 | 0.751934 | 1.20459 |
| 1 | E4900-5000 | 101 | 0.578147 | 1.16804 |

The full-run correctable-ratio medians are 0.772152 (seed 0) and 0.776635
(seed 1).  The corresponding full-run total-conflict medians are 1.20032 and
1.20382.  The two seeds agree closely.

## Prespecified interpretation

The result changes with training stage rather than falling into one global
category.  In E0-500 the ratio is above 0.85 for both seeds, so bandwidth is
not the early-stage bottleneck by the supplied criterion.  In E4900-5000 it
falls below 0.6 for both seeds, so late in P0 a substantial conflict is
detected but cannot be corrected in rank 800; error feedback has a
mechanistically targeted late-stage role.  The middle window lies between the
two decision thresholds.  This report does not authorize or launch an EF run.

## Curves

- `seed0_p0_correctable_curve.csv`
- `seed1_p0_correctable_curve.csv`

Each CSV contains epoch, correctable ratio, residual norm, reconstructed
epsilon norm, and total-conflict ratio for all 5000 updates.
