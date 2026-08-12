# N2 rank-200 spectral-cluster SNR shrinkage

Default-disabled experimental screen.  The current rank-200 Rayleigh--Ritz
force is split over even and odd walkers in a shared basis. Near-degenerate
Ritz modes, separated only by relative eigengaps above 0.05, share one
cross-half-batch SNR shrinkage weight. No history, EF, complement, or S/g EMA
is active.

Stage 0 is a 200-step health screen.  It has no accuracy claim and writes no
checkpoint.  It must remain finite, retain a noncollapsed update, and show
weaker average confidence at the weak-curvature boundary than at the strong
boundary before a single E1000 frozen gate is allowed.
## Purpose

This is the single low-cost test admitted after the N2 capacity audit.  It
keeps a current-batch rank-200 Rayleigh--Ritz solve and applies one shared
cross-half-batch reproducibility weight to every near-degenerate spectral
cluster.  No historical S matrix, gradient EMA, complement, or error feedback
is active.  The feature is default-disabled as
`experimental_mode=cluster_envelope_ritz_snr`.

The score operator is stored in VMCNet as `o_cur = O_bar.T` (parameters by
walkers).  The two half-batch forces are evaluated in one common eigenbasis of
`Q.T @ O_bar.T @ O_bar @ Q`; therefore a common orthogonal rotation inside a
near-degenerate cluster leaves its signal/noise weight unchanged.

## Staged gate

1. `health.sh`: 200 steps only.  The E1000 arm is permitted only if every
   update is finite, weak and strong spectral boundaries receive distinct
   weights, at least 25% of the unshrunk update mass remains, and the candidate
   update cosine with the unshrunk rank-200 baseline is at least 0.8.
2. `train_e1000.sh` and `eval_e1000.sh`: one candidate only, and only after the
   health gate.  The scientific endpoint is the matched frozen evaluation,
   compared with hard rank-200 (variance 3.1184518135, energy
   -109.4839973286) and SPRING E1000 (variance 1.79339, energy -109.4876621).

No E5000 or E100000 is admitted by this protocol.
