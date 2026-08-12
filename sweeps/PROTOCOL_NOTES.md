# Protocol notes

Known deviations from the SPRING-paper protocol for the two-system pilot:

1. Energy clipping uses `5 × MAD` through VMCNet's `total_variation` implementation, rather than `5 × standard deviation`. For a Gaussian distribution, `5 × MAD` is approximately equivalent to `4 × standard deviation`. SPRING and WSSR use the same clipping implementation and threshold, so their internal comparison remains fair.
2. There are no other known deviations. The inverse-time learning-rate schedule with decay rate `1e-4`, walker counts, burn-in, and MCMC steps per parameter update are aligned with the target protocol.
3. N2 geometry uses R = 2.068 Bohr (the experimental equilibrium bond length, matching the high-precision reference -109.5388(1) Ha); all internal method comparisons use the same geometry.
