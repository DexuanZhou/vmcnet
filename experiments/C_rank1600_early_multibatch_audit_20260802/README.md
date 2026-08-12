# Early-regime fixed-lambda multibatch audit

This read-only audit is run only after the frozen gap curve selects the
checkpoint immediately before the SPRING-WSSR variance gap opens. It compares
the current-batch rank-1600 WSSR solve against the production recursive
multibatch state while holding lambda fixed at 0.001 in both candidates.

The K=3 gate requires active signal in at least 4/6 replicates, at least 10%
median held-out-residual improvement, a positive bootstrap lower bound, at
least 5/6 wins and positive held-out force dot products in at least 5/6 cases.
