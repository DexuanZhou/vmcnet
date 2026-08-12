# N2 matched-capacity envelope causal test

This experiment asks whether the history carried by the 96-dimensional
cluster envelope is itself harmful, or whether the previous envelope result
was limited simply because it used fewer current-batch directions than the
rank-200 hard control.

Both arms start from the same equilibrium-N2 KFAC-pre5000 checkpoint and use
the same optimizer, sampler, learning-rate, damping, norm constraint, SSI, and
frozen-evaluation protocol.  They differ only in how a maximum 96-dimensional
Rayleigh--Ritz space is allocated:

- `H96_current_only`: 96 current-batch SSI directions, history length 1.
- `E96_current32_history2`: 32 current-batch directions plus two stored
  rank-32 blocks, history length 3.

Both arms have complement alpha zero and rotation error feedback disabled.
Training stops at E1000.  The decision is based on frozen energy and variance,
not the E200 training-window proxy.

Interpretation is preregistered as follows:

- H96 better than E96: historical directions are harmful at fixed capacity;
- E96 comparable to H96, both below hard rank-200: capacity is the bottleneck;
- E96 better than H96: short history is useful at fixed capacity.
