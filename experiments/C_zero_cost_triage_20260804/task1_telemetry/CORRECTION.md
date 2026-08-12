# Corrected residual-capture diagnostics

The previously reported quantity

`||W.T @ zeta|| / ||zeta||`

is a legacy action ratio, not a projection fraction. It can exceed one and
must not be interpreted as the percentage of residual information captured by
rank 800. Consequently, the late value near 0.56 does **not** establish that
44% of the residual was discarded.

Two dimensionally valid projections answer different questions:

1. Sample-space capacity:
   `||P_col(W) zeta|| / ||zeta||`, where `W = O_bar @ U_r`.
2. Parameter-space conflict coverage:
   `||U_r U_r.T (O_bar.T @ zeta)|| / ||O_bar.T @ zeta||`.

In the implementation these are named separately:

- `RR_sample = 1 - ||zeta - W alpha|| / ||zeta||` measures the residual
  reduction actually delivered by the damped Galerkin solve;
- `Captured_param(v) = ||U_r U_r.T v|| / ||v||` measures coverage of an
  explicitly identified parameter-space target.  The audit helper accepts
  `parameter_target=v`; if omitted it uses the detected conflict
  `v=O_bar.T@zeta`.  A report must state which `v` was used.

They are not generally equal. The first is now implemented for audits using an
SVD of `W`; the second is evaluated independently. The hot Galerkin path also
records two free diagnostics based on the correction actually applied:

- `||W alpha|| / ||zeta||`;
- `1 - ||zeta - W alpha|| / ||zeta||`.

The completed P0 monitor files do not contain `alpha` or the corrected residual,
so the corrected ratios cannot be reconstructed exactly from those logs. A
read-only replay or a future short diagnostic run is required before claiming
rank-800 capacity starvation. No training was launched for this correction.

The existing EF cap is `10 * ||correction||`, not `10 * ||zeta||`, and the
stored feedback is not explicitly multiplied by `mu`.

## Validation status (2026-08-07)

- CPU job `53729943`: passed the sample/parameter separation and applied
  residual-reduction tests (`2 passed`, exit code 0).
- CPU job `53730364`: all three corrected-metric tests passed, including the
  explicit `parameter_target` case (`3 passed`, exit code 0).
- No optimizer training was launched.

## Short timing-run telemetry (not an E5000 result)

The 125-step GPU-direct timing runs provide the first dimensionally valid
short-run `RR_sample` measurements.  Medians over steady timing epochs 21--120
are:

| backend | walkers | median `RR_sample` | median `||W alpha||/||zeta||` |
|---|---:|---:|---:|
| host fp64 | 1000 | 0.669440 | 0.868283 |
| device Cholesky | 1000 | 0.667888 | 0.869216 |
| host fp64 | 4096 | 0.223231 | 0.573880 |
| device Cholesky | 4096 | 0.224289 | 0.576471 |

The agreement across solve backends validates the telemetry numerically.  The
large walker-count dependence is a rank-800/sample-dimension effect, but these
timing-only trajectories do not reconstruct the missing P0 E5000 history and
must not be used as an E5000 scientific comparison.  `Captured_param(v)` was
not recorded because neither the full target vector nor the basis was retained.
