# N2 fixed-state Krylov audit: algebraically useful, computationally no-go

Slurm job `53873270` completed on `fc11013` in 1:49.  It loaded the equilibrium
N2 KFAC-pre5000 checkpoint, retained its saved 1000 walkers, and formed the
fixed score operator with shape `740720 x 1000`.  MCMC was not advanced and no
parameter update was applied.  The regularization was fixed at `lambda=1e-3`.

| Krylov dimension | normal residual ratio | normal residual reduction | sample residual ratio | cumulative Krylov time (s) |
|---:|---:|---:|---:|---:|
| 4 | 0.57619 | 42.381% | 0.91127 | 1.122 |
| 8 | 0.34016 | 65.984% | 0.83779 | 1.945 |
| 16 | 0.16729 | 83.271% | 0.70247 | 3.072 |

The result is not an algebraic failure: the regularized normal-equation
residual decreases monotonically and substantially.  It is a cost failure for
the intended optimizer.  Sixteen Fisher actions plus reorthogonalization cost
3.07 seconds on this fixed operator, versus roughly 0.1--0.2 seconds for an
entire measured SPRING/WSSR training step.  Even at `m=16`, 70.25% of the
unregularized sample-equation residual remains.  A 32--64 dimensional Krylov
envelope would therefore move in the wrong direction on the accuracy/runtime
Pareto frontier and is not admitted to training.

The first 20-GiB pilot (`53869590`) failed while centering the explicitly
materialized score matrix; its preserved output is
`/scratch/dexuan1/runs/N2_krylov_offline_20260809.failed_53869590`.  The
successful 40-GiB run peaked at 14.96 GB allocated but required a transient
2.76-GiB allocation that the fragmented 20-GiB pool could not satisfy.

Machine-readable results are in
`/scratch/dexuan1/runs/N2_krylov_offline_20260809/results.csv` and
`summary.json`.
