# GPU-direct Galerkin solve validation

This experiment changes only the full-current-batch Galerkin linear-solve
backend.  `host_fp64` is the historical NumPy `pure_callback` implementation;
`device_cholesky` forms the regularized Gram matrix and solves it with JAX
Cholesky/triangular solves on the accelerator in the training dtype. Error
feedback remains off.

The first gate is the targeted unit-test job in `unit.sh`.  Only after those
tests pass should a paired 200-step timing comparison be submitted.

The default remains `host_fp64`; runs opt in with
`galerkin_solve_backend=device_cholesky`. The paired timing array executes the
host and device backends sequentially on the same H100 for each walker count.
Epochs 1--20 warm up, epochs 21--120 are the steady timing window, and epochs
121--125 provide a disjoint JAX trace. No Error Feedback or E5000 training is
part of this experiment.
