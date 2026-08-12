# Task 2: large-N timing pilot (pre-W-cache)

All accepted cells use actual fresh walker arrays at the requested N; the
leading dimensions are recorded in `timing_summary.json`.  Timing uses
epochs 21--100 after 20 warm-up updates.  All GPUs report NVIDIA H100
80GB HBM3.  Recurrence cells are the pre-cache implementation.  The
SPRING N=8192 retry freezes the learning rate at zero solely to prevent
off-distribution fresh walkers from changing parameters during a timing-only run;
the optimizer and MCMC kernels are unchanged.

| N | method | rank | status | s/step mean ± std | peak GiB | node |
|---:|:---|---:|:---|---:|---:|:---|
| 4096 | spring | -- | OK | 0.212177 ± 0.002343 | 12.70 | fc10606 |
| 4096 | recurrence | 800 | OK | 0.358828 ± 0.006604 | 53.70 | fc10606 |
| 4096 | recurrence | 1600 | OOM | OOM (request 28.73 GiB) | 33.69 | fc10606 |
| 8192 | spring | -- | INCOMPLETE | OOM (request -- GiB) | 25.22 | fc10208 |
| 8192 | recurrence | 800 | OOM | OOM (request 51.45 GiB) | 50.21 | fc10606 |
| 8192 | recurrence | 1600 | OOM | OOM (request 53.44 GiB) | 50.21 | fc10517 |
| 1000 | spring | -- | OK | 0.034415 ± 0.000578 | 3.81 | fc10606 |
| 1000 | recurrence | 800 | OK | 0.155599 ± 0.000604 | 33.31 | fc10519 |

## Low-level profiler slices

Profiler values below are duration sums divided by the five disjoint profile-tail
steps.  They are not additive wall-time partitions because CPU callbacks and GPU
kernels can overlap.  XLA identifies the explicit Galerkin `W=O U` multiply as
`jit(main)/dot_general`; the host solve is identified by its callback function.
The remaining WSSR dot kernels include Gram/SSI matmuls, while QR and eigensolve
are split out when XLA exposes their source scope.

| N | method | rank | W matmul ms | host solve ms | SSI QR ms | SSI eigh ms | other WSSR dot ms |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 4096 | spring | -- | 60.225 | 0.000 | 0.000 | 0.000 | 0.000 |
| 4096 | recurrence | 800 | 23.248 | 61.533 | 28.454 | 10.075 | 136.080 |
| 1000 | spring | -- | 4.385 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1000 | recurrence | 800 | 5.431 | 26.889 | 13.165 | 9.381 | 58.288 |

## Decision answers

There is no measured speed crossover through N=8192: at N=1000 and 4096
SPRING is faster, and at N=8192 both recurrence ranks OOM while SPRING runs.
Thus a finite crossover interval is not observed in the tested range.

At N=4096, r=1600 is `OOM`; consequently its per-step
cost ratio to SPRING is not finite/measurable.  The feasible r=800 ratio is
1.691x.
