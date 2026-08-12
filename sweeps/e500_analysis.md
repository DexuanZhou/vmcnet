# E500 formal log-error analysis

Definition: endpoint = tail-100 mean energy minus reference; uncertainty = ±2 epoch-wise SEM. Tail slope is OLS on the final 200 epochs of the 50-epoch moving-mean log10 error. Ranking uses endpoint error, with overlapping closed ±2SEM intervals treated as ties and broken by the more negative slope.

## C

| Rank | Run | Tail-100 error ±2SEM (mHa) | Tail slope (log10 Ha / epoch) | Fit epochs |
|---:|---|---:|---:|---:|
| 1 | `wssr_center_tik` | 8.612 ± 3.231 | -2.489e-03 | 301–500 (200) |
| 2 | `wssr_eta05` | 7.909 ± 3.121 | -2.035e-03 | 301–500 (200) |
| 3 | `wssr_eta05_tik` | 6.817 ± 3.240 | -1.931e-03 | 301–500 (200) |
| 4 | `spring_lr02` | 4.589 ± 1.777 | -1.889e-03 | 301–500 (200) |
| 5 | `wssr_rank800` | 12.523 ± 3.878 | -1.939e-03 | 301–500 (200) |
| 6 | `wssr_eta02_tik` | 5.895 ± 3.177 | -6.118e-04 | 301–500 (200) |
| 7 | `wssr_center` | 11.735 ± 3.872 | -3.377e-04 | 301–500 (200) |
| 8 | `wssr_warm1` | 13.852 ± 3.056 | +8.736e-05 | 301–500 (200) |
| 9 | `wssr_eta095` | 25.801 ± 3.049 | -6.239e-04 | 301–500 (200) |
| 10 | `C_kfac_pre1000` | 48.003 ± 11.795 | -8.161e-04 | 801–1000 (200) |

## N2eq

| Rank | Run | Tail-100 error ±2SEM (mHa) | Tail slope (log10 Ha / epoch) | Fit epochs |
|---:|---|---:|---:|---:|
| 1 | `wssr_eta05_tik` | 79.708 ± 11.093 | -9.775e-04 | 301–500 (200) |
| 2 | `wssr_lr005` | 90.993 ± 10.953 | -9.725e-04 | 301–500 (200) |
| 3 | `wssr_center` | 91.136 ± 10.557 | -7.526e-04 | 301–500 (200) |
| 4 | `wssr_lr001` | 93.448 ± 12.831 | -1.639e-04 | 301–500 (200) |
| 5 | `wssr_center_tik` | 88.033 ± 9.793 | +1.818e-04 | 301–500 (200) |
| 6 | `wssr_eta095` | 118.030 ± 10.070 | -6.717e-04 | 301–500 (200) |
| 7 | `N2eq_kfac_pre2000` | 155.338 ± 16.080 | +6.098e-04 | 1801–2000 (200) |
