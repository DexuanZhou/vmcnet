# N2 rank-800 error-feedback E1000 result

All arms start from the same KFAC-pre5000 checkpoint with 1000 training walkers.
Frozen endpoints use 2000 fresh walkers, 10000 burn-in steps and 2000 measurements.

| arm | endpoint | frozen energy (Ha) | stderr | frozen variance (Ha^2) | RR_sample last100 median | m/zeta last100 median | cap frequency |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_vanilla | 500 | -109.488115632 | 0.001081 | 1.753511588 | 0.691959 | 0.000000 | -- |
| A_vanilla | 1000 | -109.493549961 | 0.001013 | 1.500764369 | 0.695572 | 0.000000 | -- |
| B_EF_mu095 | 500 | -109.489762104 | 0.001180 | 2.023211652 | 0.680898 | 0.035472 | 0.0000 |
| B_EF_mu095 | 1000 | -109.487701651 | 0.001113 | 2.326335138 | 0.676689 | 0.033761 | 0.0000 |
| SPRING_mu095 | 500 | -109.485744055 | 0.001250 | 2.296723776 | -- | -- | -- |
| SPRING_mu095 | 1000 | -109.487662107 | 0.001096 | 1.793394953 | -- | -- | -- |

## Paired variance comparisons

- E500: EF versus vanilla improvement = -15.38%; EF/SPRING variance ratio = 0.881.
- E1000: EF versus vanilla improvement = -55.01%; EF/SPRING variance ratio = 1.297.

## EF health gate

- cumulative cap frequency at E1000: 0.0000%; the requested <5% gate passes.
- `m/zeta` is reported because it defines this experiment's cap, but it compares parameter and sample spaces. The parameter-consistent `m/(O.T@zeta)` values are retained in results.csv.
- This is a one-seed short screen; it is not a 100k accuracy result.
