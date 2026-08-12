# Two-system pilot technical report

Data status: **17/19 runs complete**. The two partial runs are the original and explicitly configured N2eq SPRING attempts; they have 47 and 48 finite epochs respectively before NaN.

Endpoint comparisons use the final-100-epoch mean and epoch-wise SEM. The SEM describes temporal fluctuation, not independent-repeat uncertainty.

## 1. Configuration and complete results

All main runs use 1000 walkers, burn-in 5000, 10 MCMC steps per update, mean-centered clipping threshold 5, 500 epochs, and inverse-time decay 1e-4. WSSR uses damping 3e-4, complement weight 0, and norm constraint 1e-3.

**Spectral-regularization note:** every WSSR run without a `_tik` suffix used `hard_floor`; these runs were initially interpreted against historical tikhonov tuning and are a distinct regularization family. Every `_tik` run explicitly used `tikhonov`.

### C

| Run | Optimizer | Spectral reg | lr | eta | rank/storage/work | Status | Finite epochs | Tail-100 mean ± SEM (Ha) | Error (mHa) |
|---|---|---|---:|---:|---|---|---:|---:|---:|
| C_kfac_pre1000 | kfac_pre | — | 0.05 | — | — | complete | 1000/1000 | -37.79670654 ± 0.005897 | 48.0035 |
| spring_lr02 | spring | — | 0.02 | — | — | complete | 500/500 | -37.84012127 ± 0.0008884 | 4.58873 |
| wssr_center | wssr | hard_floor | 0.002 | 0.8 | 1600/1600/1600 | complete | 500/500 | -37.83297508 ± 0.001936 | 11.7349 |
| wssr_eta05 | wssr | hard_floor | 0.002 | 0.5 | 1600/1600/1600 | complete | 500/500 | -37.83680134 ± 0.001561 | 7.90866 |
| wssr_eta095 | wssr | hard_floor | 0.002 | 0.95 | 1600/1600/1600 | complete | 500/500 | -37.81890869 ± 0.001524 | 25.8013 |
| wssr_rank800 | wssr | hard_floor | 0.002 | 0.8 | 800/800/800 | complete | 500/500 | -37.83218681 ± 0.001939 | 12.5232 |
| wssr_warm1 | wssr | hard_floor | 0.002 | 0.8 | 1600/1600/1600 | complete | 500/500 | -37.83085835 ± 0.001528 | 13.8517 |
| wssr_center_tik | wssr | tikhonov | 0.002 | 0.8 | 1600/1600/1600 | complete | 500/500 | -37.83609756 ± 0.001616 | 8.61244 |
| wssr_eta02_tik | wssr | tikhonov | 0.002 | 0.2 | 1600/1600/1600 | complete | 500/500 | -37.83881451 ± 0.001588 | 5.89549 |
| wssr_eta05_tik | wssr | tikhonov | 0.002 | 0.5 | 1600/1600/1600 | complete | 500/500 | -37.8378928 ± 0.00162 | 6.8172 |

### N2eq

| Run | Optimizer | Spectral reg | lr | eta | rank/storage/work | Status | Finite epochs | Tail-100 mean ± SEM (Ha) | Error (mHa) |
|---|---|---|---:|---:|---|---|---:|---:|---:|
| N2eq_kfac_pre2000 | kfac_pre | — | 0.05 | — | — | complete | 2000/2000 | -109.3834617 ± 0.00804 | 155.338 |
| spring_lr002 | spring | — | 0.002 | — | — | partial | 47/500 | — ± — | — |
| spring_lr002_mu099_nc001 | spring | — | 0.002 | — | — | partial | 48/500 | — ± — | — |
| wssr_center | wssr | hard_floor | 0.002 | 0.8 | 1600/1600/1600 | complete | 500/500 | -109.4476645 ± 0.005278 | 91.1355 |
| wssr_lr001 | wssr | hard_floor | 0.001 | 0.8 | 1600/1600/1600 | complete | 500/500 | -109.4453516 ± 0.006415 | 93.4484 |
| wssr_lr005 | wssr | hard_floor | 0.005 | 0.8 | 1600/1600/1600 | complete | 500/500 | -109.4478071 ± 0.005476 | 90.9929 |
| wssr_eta095 | wssr | hard_floor | 0.002 | 0.95 | 1600/1600/1600 | complete | 500/500 | -109.4207699 ± 0.005035 | 118.03 |
| wssr_center_tik | wssr | tikhonov | 0.002 | 0.8 | 1600/1600/1600 | complete | 500/500 | -109.4507669 ± 0.004896 | 88.0331 |
| wssr_eta05_tik | wssr | tikhonov | 0.002 | 0.5 | 1600/1600/1600 | complete | 500/500 | -109.459092 ± 0.005546 | 79.708 |

## 2. Main findings

### C: center tikhonov versus hard-floor

Hard-floor gives -37.832975 Ha and tikhonov gives -37.836098 Ha: tikhonov improves by **3.122 mHa**. Their gaps behind C SPRING are 7.146 and 4.024 mHa. Therefore the regularization mismatch explains **3.122 mHa, or 43.7%**, of the hard-floor center run's lag; the remaining gap is 4.024 mHa.

### C: tikhonov eta trend

Tail-100 means are eta=0.2: -37.838815 Ha; eta=0.5: -37.837893 Ha; eta=0.8: -37.836098 Ha. Eta 0.5 improves on 0.8 by 1.795 mHa, and eta 0.2 improves on 0.5 by 0.922 mHa. The measured trend is monotonic: **eta=0.2 is best**.

### N2eq ranking and SPRING diagnosis

The original and fixed SPRING attempts become NaN after 47 and 48 finite epochs respectively, including the explicit `mu=0.99`, `constrain_norm=true`, `norm_constraint=0.001` rerun. SPRING therefore has no valid tail-100 endpoint and ranks last as failed/partial.

Among valid N2eq WSSR endpoints, lower is better:

1. `wssr_eta05_tik` (tikhonov): -109.459092 Ha; 79.71 mHa above reference
2. `wssr_center_tik` (tikhonov): -109.450767 Ha; 88.03 mHa above reference
3. `wssr_lr005` (hard_floor): -109.447807 Ha; 90.99 mHa above reference
4. `wssr_center` (hard_floor): -109.447664 Ha; 91.14 mHa above reference
5. `wssr_lr001` (hard_floor): -109.445352 Ha; 93.45 mHa above reference
6. `wssr_eta095` (hard_floor): -109.420770 Ha; 118.03 mHa above reference

## 3. Convergence

![](report_figs/convergence_C.svg)

![](report_figs/convergence_N2eq.svg)

## 4. Tikhonov eta and learning-rate sensitivity

![](report_figs/sensitivity.svg)

## 5. Interpretation limits

These are single optimization trajectories. Tail-epoch SEM is not a seed-to-seed uncertainty estimate, so paired differences require independent-repeat confirmation.
