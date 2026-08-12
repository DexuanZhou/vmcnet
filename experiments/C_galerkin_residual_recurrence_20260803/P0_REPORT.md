# P0 result: full-current Galerkin residual, rank 800

P0 used the matched C protocol from the KFAC-pre1000 checkpoint with
`mu=.99`, `eta_S=eta_g=0`, fixed Tikhonov lambda `1e-3`, learning rate `.04`,
SSI 40/2, 1000 walkers, and `reburn=False`.  Code stores
`o_cur = O_bar.T`, so the full residual and current action were evaluated as
`e_cur - o_cur.T @ prior` and `o_cur.T @ U_r`.

## 200-step prescreen

The median correction discrepancy was
`||c_full-c_rank||/||c_rank|| = 0.03310` (IQR `0.03141--0.03492`, P95
`0.03912`).  This exceeded the preregistered `1e-3` branch threshold, so the
full two-seed P0 contrast was required.

## Light frozen curve

| seed | endpoint | frozen energy | frozen variance |
|---:|---:|---:|---:|
| 0 | E1000 | -37.84182681 | 0.15738849 |
| 0 | E2500 | -37.84309267 | 0.07261760 |
| 0 | E5000 | -37.84409479 | 0.04587327 |
| 1 | E1000 | -37.84167272 | 0.21280711 |
| 1 | E2500 | -37.84334813 | 0.08241763 |
| 1 | E5000 | -37.84437147 | 0.04507317 |

## Paired P0 gate

| seed | rank-coordinate E5k | full-current E5k | improvement |
|---:|---:|---:|---:|
| 0 | 0.04922199 | 0.04587327 | 6.80% |
| 1 | 0.05219800 | 0.04507317 | 13.65% |
| geometric mean | 0.05068816 | 0.04547146 | **10.29%** |

The preregistered continuation threshold was a geometric-mean improvement of
at least 15%.  P0 therefore **failed the gate**, and P0b/P1/P2/P3 were not
submitted.  Full-current Galerkin evaluation is beneficial on both paired
seeds, but its endpoint remains 20.17% above SPRING E5k variance
`0.03784061`.

## Cost and telemetry

| seed | seconds/step | median correction difference | P95 difference | median correctable ratio | median drift beta |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.17505 | 0.01993 | 0.03426 | 0.77215 | 0.02741 |
| 1 | 0.17523 | 0.01905 | 0.03486 | 0.77663 | 0.02731 |

The old rank-coordinate recurrence was about `0.130 s/step` and SPRING about
`0.091 s/step`.  Thus P0 is approximately 35% slower than the existing rank
800 recurrence and 92% slower than SPRING per step.  The correction is not a
numerically irrelevant change, but under the registered decision rule its
accuracy gain is too small for the added cost.

Machine-readable results are in
`/scratch/dexuan1/runs/C_galerkin_residual_recurrence_20260803/p0_summary.json`.
