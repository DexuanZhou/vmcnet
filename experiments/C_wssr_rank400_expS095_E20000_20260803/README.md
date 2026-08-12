# C rank-400 exponential-S WSSR, E20000

Single matched-seed trajectory from the existing C KFAC-pre1000 checkpoint.
It uses the current-batch gradient (`eta_g=0`) and the adaptive covariance
schedule

`eta_S(t) = 0.95 * (1 - exp(-t / 1000))`.

All other scientific settings follow the recent C matched protocol: 1000
walkers, 10 MCMC steps per update, learning rate 0.04 with inverse-time decay
1e-4, fixed Tikhonov lambda 0.001, norm constraint 0.001, SSI 40/2, no
complement, and `reburn=False`. The training job produces 20000 WSSR updates;
the dependent job performs the standard light frozen evaluation using two
million local-energy samples.
