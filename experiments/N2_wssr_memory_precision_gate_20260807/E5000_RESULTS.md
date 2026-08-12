# N2 R=2.016 E5000 residual-recurrence screen

All four jobs completed successfully under the matched protocol: KFAC-pre5000,
1000 walkers, ten MCMC steps per update, no reburn, `lr=0.002`, inverse-time
decay `1e-4`, `lambda=1e-3`, norm constraint `1e-3`, `mu=0.95`, and
full-current-batch residual evaluation.  Frozen evaluation used 2000 walkers,
10000 burn-in steps, 2000 measurements, and ten MCMC steps per measurement.

| rank | subspace eta_S | interpretation | frozen energy (Ha) | error (mHa) | frozen variance (Ha^2) | approx train s/step | peak GPU memory (GiB) | total job wall |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 800 | 0.0 | compressed sketched-SPRING control | -109.501489224 | 40.811 | 0.961950256 | 0.307 | 33.42 | 00:35:47 |
| 800 | 0.2 | compressed + WSSR history | -109.503022719 | 39.277 | 1.219046034 | 0.318 | 33.42 | 00:41:38 |
| 1000 | 0.0 | SPRING-equivalent regression anchor | -109.505193902 | 37.106 | 0.948080319 | 0.337 | 60.56 | 00:42:12 |
| 1000 | 0.2 | uncompressed history-subspace control | -109.498953463 | 43.347 | 1.129492053 | 0.347 | 33.42 | 00:41:59 |

Energy errors use the paper benchmark -109.5423 Ha.  Approximate training
seconds per step are elapsed times from creation of the training energy stream
to creation of the epoch-5000 checkpoint, divided by 5000; they include any
first-step compilation inside that interval.  Peak memory is the measured
per-process maximum from `nvidia-smi`.  The 60.56-GiB rank-1000/eta_S=0 value
is reproducible in that run but is path-dependent and should not be read as a
general rank-1000 requirement without a dedicated repeat.

## Interpretation

At rank 800, adding WSSR subspace history (`subspace_eta_S=0.2`) raises frozen
variance by 26.7% relative to the eta_S=0 compressed control.  Its 1.53-mHa
lower energy is only about 1.3 combined standard errors and is not persuasive
given the worse variance.  At rank 1000, history worsens both energy and
variance.  Thus fixed history-subspace averaging does not recover the
rank-truncation loss in the early N2 regime.

Rank 1000 with eta_S=0 is not a WSSR accuracy result: centering limits the
1000-walker current-batch Jacobian rank to at most 999, so the full-residual
Galerkin recurrence spans the complete current-batch row space and is
mathematically SPRING up to SSI and floating-point approximation.  It is kept
only as a regression anchor.

Matched E5000 light-frozen evaluations of the existing SPRING and MinSR
checkpoints were submitted separately.  Their first attempt (53656003) failed
at config parsing because a paper-era config lacks the `wandb` field.  Retry
job 53656744 removed that override but also failed before evaluation because
the same paper-era config lacks `vmc.disable_checkpointing`.  Thus no matched
SPRING/MinSR E5000 frozen samples were produced; no training was affected.
