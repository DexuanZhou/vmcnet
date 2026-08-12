# N2 current-only Ritz capacity timing and spectrum audit

Three 100-step arms use only the current batch and ranks 200, 384, and 512.
There is no envelope history, EF, complement, or S/g averaging.  A one-second
sidecar monitor records completed epochs and device memory, allowing steady
step time to be estimated without the one-time JAX compilation cost.

Each arm computes one extra SSI mode (working/storage rank `d+1`) while the
actual update remains strictly limited to `d`.  This exposes the boundary ratio
`sigma_d/sigma_(d+1)` without changing the optimization capacity.
The internal adaptive rank is also initialized at `d+1`; otherwise the legacy
rank mask zeros the audit-only boundary mode before it can be measured.  The
cluster/Ritz update remains capped by `cluster_envelope_rank=d`.

The audit records projected curvature extrema/condition, numerical rank,
gradient capture, the spectral boundary ratio, and the fraction of Ritz update
norm carried by the weakest 10% of projected-curvature modes.  It has no
training-accuracy objective. Historical selection gain is intentionally absent:
these are current-only arms, so that quantity is not defined and must not be
reported as zero evidence against history selection.
