# N2/H2O rank-1600 dual-cap WSSR

This experiment compares the same optimizer configuration on the existing
matched KFAC-pre5000 checkpoints for equilibrium N2 and H2O.  It does not
overwrite earlier WSSR runs.

The candidate keeps the existing semi-matrix-free right-warm SSI solver and
adds only two safeguards around a small tail complement:

- rank 1600, SSI 40/2, `eta_S=eta_g=0.3`;
- learning rate 0.002 with inverse-time decay 1e-4;
- Tikhonov lambda 1e-3 and relative singular-value cutoff 3e-4;
- adaptive tail weight 1e-4, capped at 10% of the retained direction in both
  Euclidean parameter norm and current-batch function norm;
- global function-space radius 8e-4, followed by the legacy Euclidean squared
  radius 1e-3 as a safety backstop;
- 1000 walkers, 10 MCMC steps/update, mean clipping at 5 sigma, no reburn;
- no gradient transport, solution recurrence, or error feedback.

Each system first runs a 20-step H100 smoke test.  Its 50k training stage is
submitted with an `afterok` dependency, and the 50k-to-100k continuation plus
20k frozen evaluation is submitted after that.

Submit with:

```bash
bash experiments/N2_H2O_WSSR_dualcap_E100000_20260810/submit.sh
```
