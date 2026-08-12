# Adaptive eta_S C E5000

This paired-seed experiment adds two adaptive-S arms to the completed split-eta
factorial experiment. Both arms use the current gradient (`eta_g=0`) and differ
only in the S-history schedule. Existing controls are reused from job 52688248:

- constant eta_S=0.95: `A_etaS095_etaG0`
- eta_S=0: `D_etaS0_etaG0`

New arms:

| task | arm | schedule | eta_max | T_warmup | tau |
| --- | --- | --- | ---: | ---: | ---: |
| 0 | linear_T1000 | linear_warmup | 0.95 | 1000 | 1000 |
| 1 | exponential_tau1000 | exponential_growth | 0.95 | 1000 | 1000 |

All arms use the same C KFAC-pre1000 checkpoint, seed/key, 1000 walkers,
rank 1600, SSI 40/2, learning rate 0.04, fixed Tikhonov lambda 0.001, norm
constraint 0.001, and no reburn.
