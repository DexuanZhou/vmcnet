# Delayed-subspace 200-step timing result

| refresh period | s/step mean | std | speedup vs period1 | median residual reduction | observed refresh fraction |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.273885 | 0.074661 | 1.000x | 0.888954 | 1.000 |
| 2 | 0.375009 | 0.111957 | 0.730x | 0.883032 | 0.500 |
| 5 | 0.419316 | 0.104261 | 0.653x | 0.850276 | 0.200 |

## Read-only timing audit

All three arms ran on H100 nodes.  The first 20 updates were excluded, so the
relative slowdown is not a compilation or GPU-model artifact.  Splitting the
delayed arms by branch gives the following median update times:

| refresh period | refreshed step (s) | reused-subspace step (s) |
|---:|---:|---:|
| 2 | 0.312102 | 0.414992 |
| 5 | 0.323319 | 0.442296 |

The reused-subspace branch is itself slower.  Recomputing current-batch Ritz
pairs and the Galerkin projection inside a fixed stored left subspace costs
more in this implementation than the warm-started SSI refresh path it replaces.
The scientific trajectory also degrades within this timing-only pilot: the
epoch-200 online variances are 0.1163, 0.3450, and 0.9183 for periods 1, 2,
and 5.  These online values are not formal frozen-accuracy results, but they
are sufficient to reject delayed refresh as both a performance and stability
candidate.  No E5000 or 100000-step follow-up was submitted.

The absolute period-1 timing (0.2739 s/update) does not reproduce the earlier
0.1751 s/update P0 measurement, so it should not replace that value in a
cross-experiment Pareto table.  The within-array relative comparison remains
valid because the three arms used the same source snapshot, settings, GPU
class, and telemetry path.
