# Active E100000 follow-up scripts

This directory contains only the active formal configurations.

## Geometry and spin policy

- Carbon uses `nelec=(4,2)` exclusively.
- N2 uses a bond length of `2.068 Bohr` exclusively in all future formal work.
- Superseded scripts, outputs, summaries, and figures are preserved under the corresponding `old/` directories and are not part of active analysis.

## Active C runs

All C scripts reload `/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1/checkpoints/1000.npz` with a new optimizer state and `reburn=True`.

| Script | Optimizer | Requested time |
|---|---|---:|
| `C_spring_lr02_mu099_nc001_fresh_E100000.sh` | SPRING | 3 h |
| `C_wssr_eta08_lr02_tikhonov_svd8_5_fresh_E100000.sh` | WSSR | 15 h |
| `C_wssr_eta02_lr0002_tikhonov_svd8_2_fresh_E100000.sh` | WSSR control | 10 h |
| `C_wssr_eta08_lr02_tikhonov_svd8_2_fresh_E100000.sh` | WSSR SVD ablation | 10 h |

Common settings are 100000 epochs, 1000 chains, burn-in 5000, 10 MCMC steps per update, inverse-time decay `1e-4`, clipping threshold 5, no training-time evaluation, and final-only checkpointing.

Run `./check_scripts.sh` for static validation and to print the active C submission commands. It does not submit jobs.
