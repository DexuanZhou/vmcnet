# Final conclusion for the explicit low-rank envelope branch

Two pre-registered low-cost tests answer the two remaining proposals without
an accuracy-training sweep.

1. **Spectral-cluster cross-batch SNR shrinkage is a no-go.**  It distinguishes
   weak from strong retained clusters, but at 1000 walkers it retains only
   1.14% of the unshrunk rank-200 squared update mass and rotates the update to
   cosine 0.423.  It fails both health gates, so E1000 was not submitted.
2. **Gradient-seeded Krylov is algebraically effective but too expensive.**
   `m=16` reduces the regularized normal residual by 83.27%, but costs 3.07 s
   and still leaves sample residual ratio 0.702.  It is not competitive with a
   0.1--0.2 s optimizer step, so it was not trained.

Together with the preceding capacity audit (flat d/d+1 spectral boundary and
24--28% update mass already carried by the weakest retained decile), these
results close the proposed parameter-space explicit low-rank/envelope route.
The failure is not that weak modes contain no physics.  It is that neither
hard truncation nor two-half-batch multiplicative confidence shrinkage can
separate their weak physical signal from Monte Carlo non-reproducibility at
N=1000 without either deleting most of the update or paying many full Fisher
actions.

No E1000, E5000, or E100000 was launched from these candidates.
