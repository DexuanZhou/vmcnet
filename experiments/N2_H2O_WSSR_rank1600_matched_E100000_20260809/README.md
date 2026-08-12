# N2 and H2O matched rank-1600 WSSR

This experiment adds the missing rank-1600 ordinary-WSSR arms for N2 at
2.016 Bohr and H2O.  Both start from the exact KFAC-pre5000 checkpoint used by
their existing SPRING comparison and use the matched training/evaluation
protocol.

The scientific optimizer is unchanged from the existing N2 rank-1400 arm:

- ordinary right-warm WSSR (`solution_recurrence_mode=none`);
- rank/storage/working rank 1600 and SSI 40/2;
- `eta_S=0.2`, `eta_g=0`, learning rate 0.002;
- inverse-time decay 1e-4, fixed Tikhonov lambda 1e-3;
- Euclidean norm constraint 1e-3 and no complement update.

The prior rank-1600 N2 attempt materialized the augmented score matrix and
stored a second persistent warm basis; it OOMed on an 80-GiB H100.  This run
uses the tested semi-matrix-free augmented-factor path.  The path applies the
same augmented operator and RHS blockwise, sets `store_warm_u=False`, and does
not enable recurrence or any experimental optimizer feature.  It therefore
changes memory representation, not the WSSR equation.

Each system has an independent dependency chain:

1. 20-step memory/finite-value gate that also writes and reloads a full-sized
   leaf-sharded rank-1600 optimizer checkpoint;
2. epochs 0--50000 with a leaf-sharded optimizer checkpoint;
3. restored epochs 50000--100000 followed by the matched frozen evaluation
   (2000 walkers, 10000 burn-in, 20000 measurements, 10 MCMC moves per
   measurement).

Training uses 1000 walkers, 10 MCMC moves per update, mean-centered clipping
at five standard deviations, and no reburn after KFAC-pre5000.
