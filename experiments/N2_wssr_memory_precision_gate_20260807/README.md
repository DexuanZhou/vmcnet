# N2 WSSR memory and precision gate

Two gated stages are used. Stage 1 verifies a rank-1600 right-warm WSSR path
that stores no persistent warm-U and applies the history/current augmented
factor blockwise, without allocating the explicit concatenated `O_aug`.
Stage 2 is permitted only after numerical regression tests and a 100-step H100
memory smoke test pass. It screens the full-current Galerkin residual
recurrence at ranks 800 and 1000 with subspace history weights 0 and 0.2 for
5000 N2 steps; no 100000-step run is launched by this gate.
