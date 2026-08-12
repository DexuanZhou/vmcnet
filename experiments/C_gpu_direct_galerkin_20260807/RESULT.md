# GPU-direct Galerkin timing result

| backend | walkers | mean s/step | std | peak GPU GiB | host Galerkin callback / 5 steps |
|---|---:|---:|---:|---:|---:|
| host_fp64 | 1000 | 0.154588 | 0.000707 | 33.306 | 134.911 ms (5 calls) |
| host_fp64 | 4096 | 0.351138 | 0.001473 | 57.696 | 308.226 ms (5 calls) |
| device_cholesky | 1000 | 0.127150 | 0.001848 | 33.558 | 0.000 ms (0 calls) |
| device_cholesky | 4096 | 0.297997 | 0.006214 | 57.950 | 0.000 ms (0 calls) |

- N=1000: device speedup = 1.216x (17.75% wall-time reduction).

- N=4096: device speedup = 1.178x (15.13% wall-time reduction).
