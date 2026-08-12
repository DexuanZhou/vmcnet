# Early-C curvature transport audit

This is a read-only, eight-replicate test of the proposed first-order
transport of historical WSSR Fisher/SR blocks.  It uses the fixed-lambda,
`eta_S=eta_g=0`, rank-1600 C checkpoint at epoch 1000, applies one real WSSR
step in memory, and evaluates operator actions without constructing a dense
parameter-space matrix.

The finite-difference transport is

`S_transport v = 2 S(theta + delta/2) v - S(theta) v`,

where `delta` is the actual post-learning-rate and post-norm-constraint update.
The gate requires at least a 2x reduction in same-coordinate action error and
requires ordinary new-batch sampling error not to exceed the staleness error by
more than 2x.  Failure means curvature transport is too small relative to the
dominant stochastic operator variation to justify an E5000 implementation.
