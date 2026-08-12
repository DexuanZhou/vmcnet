# C WSSR split-eta E5000 experiment

All four arms reload the same C KFAC-pre1000 checkpoint, create a fresh
optimizer state, preserve the 1000-walker sampler state without reburning, and
use the same folded PRNG key (seed 0). The matched WSSR settings are rank 1600,
SSI 40/2, learning rate 0.04 with inverse-time decay 1e-4, fixed Tikhonov
lambda 1e-3, and norm constraint 1e-3. Gradient transport and eta bias
correction are disabled.

Only the two averaging weights differ:

| array task | arm | eta_S | eta_g |
| --- | --- | ---: | ---: |
| 0 | A | 0.95 | 0.0 |
| 1 | B | 0.0 | 0.95 |
| 2 | C | 0.95 | 0.95 |
| 3 | D | 0.0 | 0.0 |

Outputs are written under
`/scratch/dexuan1/runs/C_wssr_split_eta_E5000_20260803`.
