# C WSSR exponential-gradient transport result

Decision: **negative**. Do not enable this transport candidate by default.

## Protocol

- C from the matched KFAC-pre1000 checkpoint
- 1000 walkers, rank 1600, SSI 40/2, 5000 WSSR updates
- learning rate 0.04 with inverse-time decay 1e-4
- fixed Tikhonov lambda 1e-3 and norm constraint 1e-3
- `eta_S(t) = 0.95 * (1 - exp(-t/1000))`
- `eta_g(t) = 0.20 * (1 - exp(-t/1000))`
- two paired seeds; the only scientific configuration difference is the
  gradient-transport switch
- frozen endpoint: 1000 walkers, burn-in 5000, 2000 inference iterations

Training job: 52721137. Frozen-evaluation job: 52721146.

## Same-coordinate transport audit

The observed force change was positively aligned with `S @ delta_theta`, so
the code's plus sign is correct for its positive-gradient RHS convention. The
EMA operator action was nevertheless much too small:

| checkpoint | median cosine | predicted/observed norm | optimal scale |
|---:|---:|---:|---:|
| 1000 | 0.444 | 0.0416 | 11.14 |
| 3000 | 0.493 | 0.0205 | 19.93 |

The weak preregistered gate passed because the positive prediction had relative
error just below the zero-prediction baseline, but it did not support a
unit-scale Taylor model.

## E5000 training

| arm | seed | last-100 energy | last-100 variance |
|---|---:|---:|---:|
| vanilla | 0 | -37.838801575 | 0.112293192 |
| transported | 0 | -37.840475578 | 0.113092115 |
| vanilla | 1 | -37.840387878 | 0.115954417 |
| transported | 1 | -37.840313187 | 0.110516171 |

The last-100 transport ratio was only 0.0051--0.0052. Training statistics gave
opposite small seed effects and were not used as the primary decision.

## Frozen endpoint

| arm | seed | energy | variance | transported/vanilla |
|---|---:|---:|---:|---:|
| vanilla | 0 | -37.840820482 | 0.449616527 | |
| transported | 0 | -37.841935461 | 0.891920747 | 1.983737 |
| vanilla | 1 | -37.840862244 | 0.317725960 | |
| transported | 1 | -37.840963569 | 0.530503246 | 1.669688 |

The geometric-mean variance ratio is 1.819951: an 82.0% degradation under the
predeclared primary endpoint.

All four saved local-energy arrays contain 2,000,000 finite observations and
directly reproduce the JSON statistics. The bulk distributions are close: after
winsorizing the most extreme 0.1% in each tail, variances are 0.18376 versus
0.18325 for seed 0 and 0.17447 versus 0.17060 for seed 1. The negative primary
result is therefore caused by rarer, more extreme transported-tail events, not
by a broad deterioration of the local-energy distribution.

## Conclusion

With `eta_g_max=0.2`, unit-coefficient `S_ema @ delta_theta` transport is too
small to repair the measured gradient drift and does not improve the main
training distribution. Its accumulated trajectory change increases rare-tail
risk in both frozen seeds. Further tuning of `eta_g` is not supported by this
experiment; a future transport proposal would first need a better-calibrated
gradient Jacobian/Hessian action rather than the unscaled SR/Fisher action.
