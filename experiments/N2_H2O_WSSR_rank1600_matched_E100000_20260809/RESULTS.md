# N2 and H2O rank-1600 WSSR results

Both runs completed 100000 post-KFAC-pre5000 training updates and the matched
frozen evaluation protocol (2000 walkers, 10000 burn-in steps, 20000
measurements, and 10 MCMC moves between measurements).

The canonical run configuration is the saved `config.json` together with the
per-step `eta_S.txt` and `eta_g.txt` logs. These show `eta_S=eta_g=0.8` for all
100000 updates. The older `protocol.txt` files incorrectly retain stale
`eta_S=0.2, eta_g=0` text from the launcher template and must not be used to
label these two results.

Shared WSSR settings: rank/storage/working rank 1600, SSI 40/2, learning rate
0.002 with inverse-time decay 1e-4, fixed Tikhonov lambda 1e-3, Euclidean norm
constraint 1e-3, zero complement, and the semi-matrix-free augmented-factor
implementation.

| system/method | frozen energy (Ha) | SEM | frozen variance |
|---|---:|---:|---:|
| N2 WSSR rank1600, etaS=etaG=0.8 | -109.50513208 | 0.00028312 | 1.65392238 |
| N2 SPRING | -109.52480719 | 0.00014230 | 0.34756254 |
| N2 MinSR | -109.52037575 | 0.00019480 | 0.68339028 |
| N2 KFAC | -109.52429111 | 0.00015824 | 0.44348302 |
| H2O WSSR rank1600, etaS=etaG=0.8 | -76.42366849 | 0.00018100 | 0.87492508 |
| H2O SPRING | -76.43252873 | 0.00008186 | 0.15757659 |
| H2O KFAC | -76.43325386 | 0.00013264 | 0.23144816 |

For N2, WSSR is 19.675 mHa above SPRING and has 4.759 times its frozen
variance. It is also 15.244 mHa above MinSR. The earlier rank-1400 WSSR run
(`eta_S=0.2, eta_g=0`) reached -109.51225494 Ha with variance 0.97715459, so
the rank-1600, strongly averaged run is worse by 7.123 mHa and 1.693 times in
variance; this is not evidence that increasing rank is beneficial.

For H2O, WSSR is 8.860 mHa above SPRING and has 5.552 times its frozen
variance. It is 9.585 mHa above KFAC and has 3.780 times its variance. The H2O
MinSR trajectories failed with NaNs near 28000 updates and have no valid
100000-step frozen result.
