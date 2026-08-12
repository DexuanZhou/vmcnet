# C/N2 WSSR rank x warm-SVD benchmark

This benchmark changes only rank and, for rank 400, `svd_maxiter_warm`.

The six combinations per system are `(200,2)`, `(400,1)`, `(400,2)`, `(400,4)`, `(800,2)`, and `(1600,2)`. The rank is applied identically to `sr_rank`, `sr_rank_max`, `sr_storage_rank`, and `svd_working_rank`.

C uses `nelec=(4,2)`, its KFAC-1000 checkpoint, eta 0.8, and learning rate 0.02. N2 uses R=2.068 Bohr, its KFAC-5000 checkpoint, eta 0.3, and learning rate 0.002. Both use fresh optimizer state, reburn, Tikhonov regularization, damping 0.0003, norm constraint 0.001, complement weight 0, initial SVD maxiter 8, 1000 chains, and 100000 epochs.

Each array permits at most two concurrent tasks. Both arrays request 12 hours per task; jobs terminate immediately when training completes. Only the final epoch-100000 checkpoint is saved.
