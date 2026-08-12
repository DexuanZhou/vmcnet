# Current C and N2eq test summary

Snapshot: 2026-07-14 21:03 PDT.

## Scope and conventions

- Formal C calculations use the spin-polarized carbon sector `nelec=(4,2)` and reload the common C KFAC epoch-1000 checkpoint with a fresh optimizer state.
- Formal N2 calculations use R=2.016 Bohr and reload the new R=2.016 KFAC epoch-5000 checkpoint.
- Tail SEM is computed over epochs and is not independent-seed uncertainty.
- C energy error uses the matching reference `E0=-37.84471 Ha`.
- N2 values below labeled “legacy-reference difference” use `-109.5388 Ha`, which belongs to the earlier R=2.068 setup. They are useful as a common numerical offset but are **not a strict R=2.016 physical error**. A geometry-matched reference is still required.
- Normalized variance is `Var(E)/E0^2 = <(Delta E/E0)^2>` over the final 10,000 epochs.

## C formal E100000 results

| Run | Status | Finite epochs | Tail-500 error (mHa) | Tail-1000 error (mHa) | Tail-10000 error (mHa) | Normalized variance, tail-10000 | Acceptance | Active rank | GPU MiB | Wall time |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SPRING lr=0.02 | complete | 100000 | 0.066 | 0.040 | 0.110 | 2.463e-6 | 0.48985 | — | 3901 | 1:17:24 |
| WSSR eta=0.8, lr=0.02, SVD8/2 | complete | 100000 | -0.645 | -0.584 | -0.474 | 1.174e-5 | 0.48180 | 1000 | 34105 | 7:04:50 |
| WSSR eta=0.8, lr=0.02, SVD8/5 | running snapshot | 87331 | -0.062 | -0.364 | -0.408 | 1.338e-5 | 0.48175 | 952 | 34105 | running |
| WSSR eta=0.2, lr=0.002, SVD8/2 | complete | 100000 | 0.492 | 0.644 | 1.076 | 3.384e-5 | 0.48128 | 696 | 34105 | 7:03:25 |

### C interpretation

1. The best completed WSSR energy mean is eta=0.8/lr=0.02/SVD8/2. Its tail-10000 mean is 0.584 mHa lower than SPRING, but its normalized variance is about 4.8 times SPRING's. Negative stochastic error does not imply violation of the variational bound.
2. SVD8/2 is already competitive with the still-running SVD8/5 trajectory and is substantially cheaper. A final SVD ablation decision should wait for SVD8/5 epoch 100000.
3. The conservative eta=0.2/lr=0.002 control is worse in both tail-10000 energy and variance at E100000.
4. SPRING is much cheaper: about 1.3 hours and 3.9 GiB peak GPU memory versus about 7.1 hours and 34.1 GiB for completed WSSR.

## N2eq R=2.016 results

The new KFAC pretraining completed 5000/5000 finite epochs in 967 seconds and produced `checkpoints/5000.npz`.

| Run | Status | Finite epochs | First nonfinite | Tail-500 legacy difference (mHa) | Tail-1000 | Tail-10000 | Normalized variance, tail-10000 | Acceptance | Active rank | GPU MiB | Wall time |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WSSR eta=0.2, lr=0.002, C=0.001 | complete | 100000 | — | 10.043 | 10.736 | 12.118 | 4.688e-5 | 0.46769 | 922 | 62023 | 9:12:48 |
| WSSR eta=0.3, lr=0.002, C=0.001 | complete | 100000 | — | 10.340 | 11.873 | 12.104 | 4.833e-5 | 0.46761 | 667 | 62023 | 9:15:49 |
| WSSR eta=0.3, lr=0.002, C=0.002 | complete | 100000 | — | 10.340 | 11.873 | 12.104 | 4.833e-5 | 0.46761 | 667 | 62023 | 9:15:59 |
| WSSR eta=0.3, lr=0.003, C=0.001 | complete | 100000 | — | 13.961 | 13.695 | 12.044 | 5.264e-5 | 0.46756 | 564 | 62023 | 9:01:22 |
| SPRING lr=0.002 | numerical failure | 69 | 70 | — | — | — | — | — | — | 6331 | 2:40:58 |
| SPRING lr=0.001 rescue | infrastructure failure | 0 | — | — | — | — | — | — | — | — | no training |

### N2 interpretation

1. All four WSSR configurations are numerically stable for 100000 epochs. Tail-1000 favors eta=0.2/lr=0.002, while tail-10000 values are nearly tied and slightly favor lr=0.003. The ranking is window-sensitive; differences are small relative to trajectory fluctuations.
2. The C=0.001 and C=0.002 eta=0.3 trajectories are exactly identical, including energy, variance, acceptance, and active rank. The norm constraint evidently did not become active, so this run does not demonstrate a useful C effect.
3. Increasing lr from 0.002 to 0.003 reduces active rank (667 to 564) and slightly worsens normalized variance. It does not provide a decisive energy advantage.
4. SPRING lr=0.002 fails numerically at epoch 70 despite completing at the Slurm/program level. Only 69 epochs are finite.
5. The lr=0.001 rescue did **not** test algorithmic stability: job 48972406 failed before training because its allocated node exposed no CUDA GPU. It should be resubmitted before concluding that SPRING has no stable baseline.

## Excluded or invalid runs

- The first four C E100000 submissions incorrectly overrode carbon to `nelec=(3,3)`. They were cancelled and preserved, but are excluded from all formal comparisons.
- The first N2 eta=0.2 submission failed during Python import (`yaml.events` missing); `_retry1` is the valid completed replacement.
- Earlier N2 R=2.068 pilots and their `-109.5388 Ha` reference must not be mixed with R=2.016 formal geometry except as explicitly labeled legacy context.

## Current operational status

- Running: C WSSR eta=0.8/lr=0.02/SVD8/5, currently 87331 finite epochs with no NaN/Inf.
- Pending/running rescue: none; N2 SPRING lr=0.001 must be resubmitted after the no-GPU infrastructure failure if a rescue baseline is still required.
- Completed checkpoints are final-only by design: C/N2 E100000 save `100000.npz`; N2 KFAC saves `5000.npz`.

## Recommended analysis priorities

1. Wait for C SVD8/5 to finish, then compare SVD8/2 versus SVD8/5 using tail-10000 energy, normalized variance, wall time, and active rank.
2. Obtain a geometry-matched N2 R=2.016 reference before publishing error in mHa.
3. Resubmit N2 SPRING lr=0.001 on a healthy GPU node; do not interpret job 48972406 as a numerical failure.
4. Treat C=0.001 versus C=0.002 as an inactive-constraint null result.
5. For publication-level claims, add independent seeds; epoch-wise SEM alone cannot measure run-to-run uncertainty.

## Figures

- `report_figs/C_E100000_energy_window10000.svg`
- `report_figs/C_E100000_variance_window10000.svg`
- `report_figs/N2eq_R2016_E100000_energy_window10000.svg`
- `report_figs/N2eq_R2016_E100000_variance_window10000.svg`

The current figures are monochrome. Energy uses a linear mHa axis; normalized variance uses `log10[Var(E)/E0^2]`; every plotted source value is a full trailing average over 10000 iterations.
