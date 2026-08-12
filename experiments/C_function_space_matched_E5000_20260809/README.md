# C matched function-space step experiment

This experiment replaces the default Euclidean update constraint by

`||O_bar_t delta_theta||_2 <= 8e-4`

for both SPRING and rank-1600 WSSR. `O_bar_t` is centered over the current
1000 walkers and divided by `sqrt(1000)`, so the bound is the RMS first-order
change in log wavefunction modulo normalization. Both arms reload the same
KFAC-pre1000 checkpoint, use paired PRNG-key folds, run to epoch 5000, and are
evaluated with the established light frozen protocol (1000 walkers, 5000
burn-in, 2000 measurements, 10 MCMC steps per measurement).

The same upper bound does not by itself prove equal realized step lengths.
The result is interpreted as a strict function-step comparison only if the
logged constraint-active fractions are near one for both arms.
