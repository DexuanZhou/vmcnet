# Focused C WSSR E20000 results

All five jobs completed 20,000 finite epochs with no NaN/Inf. Energies use the C reference -37.84471 Ha. Tail statistics use epoch-wise SEM; they are not independent-seed uncertainty estimates.

| Rank | Run | Status | Finite epochs | First nonfinite | Tail-500 mean ± SEM | Tail-1000 mean ± SEM | Tail-1000 error (mHa) | Variance tail | Acceptance tail | GPU MiB | Wall time (s) | Active rank | Checkpoint |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `C_wssr_eta08_lr02_svd8_5` | complete | 20000 | — | -37.845260292 ± 3.83e-04 | -37.845759998 ± 2.67e-04 | -1.050 | 0.06419 | 0.48200 | 34105 | 8687 | 1000 | True |
| 2 | `C_wssr_control_eta02_lr0002_svd8_2` | complete | 20000 | — | -37.843963242 ± 3.81e-04 | -37.844079773 ± 2.63e-04 | 0.630 | 0.06722 | 0.48174 | 34105 | 5480 | 999 | True |
| 3 | `C_wssr_eta05_lr02_svd8_5` | complete | 20000 | — | -37.844223808 ± 2.74e-04 | -37.843637371 ± 1.90e-04 | 1.073 | 0.03239 | 0.48147 | 34105 | 8745 | 1001 | True |
| 4 | `C_wssr_eta02_lr02_svd8_5` | complete | 20000 | — | -37.843356148 ± 2.31e-04 | -37.843539532 ± 1.63e-04 | 1.170 | 0.02413 | 0.48215 | 34105 | 8806 | 852 | True |
| 5 | `C_wssr_eta09_lr02_svd8_5` | complete | 20000 | — | -37.841877785 ± 4.25e-04 | -37.841567608 ± 3.10e-04 | 3.142 | 0.09190 | 0.48207 | 34105 | 8655 | 1000 | True |

## Existing C SPRING baseline

Tail-1000: -37.844994720 ± 3.83e-05 Ha; error -0.285 mHa; variance tail 0.00135; acceptance tail 0.48168.

## Trajectories

![C energy error, 10k trailing window](report_figs/C_focus_energy_error_window10000.svg)

![C variance, 10k trailing window](report_figs/C_focus_variance_window10000.svg)

Both figures are monochrome and distinguish runs by dash pattern and marker. Every plotted point is a full trailing average over 10,000 iterations.

## Interpretation

- Eta=0.8, lr=0.02, SVD 8/5 has the lowest tail-1000 mean, 0.765 mHa below the SPRING mean. Its negative reference error must be interpreted with its much larger fluctuation: variance 0.06419 versus SPRING 0.00135.
- The control is an exact same-configuration repeat/prefix of the earlier C WSSR E40000 run, not a new SVD setting. At epoch 20,000 its tail-1000 error is 0.630 mHa.
- Raising lr from 0.002 to 0.02 is not sufficient by itself: eta=0.2 and eta=0.5 finish at 1.170 and 1.073 mHa.
- Eta=0.9 overshoots the useful high-history regime and finishes last at 3.142 mHa.
- SPRING retains much lower variance and far lower resource cost; WSSR peak GPU memory is 34,105 MiB for every focused run.

## E100000 candidates

Recommend the top 2 finite settings by tail-1000 error for E100000 + inference: `C_wssr_eta08_lr02_svd8_5`, `C_wssr_control_eta02_lr0002_svd8_2`.
