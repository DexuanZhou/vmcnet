# Adaptive S-mixing indicator audit: final decision

Date: 2026-08-07

This was a read-only checkpoint replay audit.  It did not modify optimizer
logic, checkpoints, model parameters, or sampler state, and it launched no
E5000 training.  The tested decision statistics were:

```text
d0 = (S_current + lambda I)^-1 g_current
dh = ((1-eta) S_current + eta S_history + lambda I)^-1 g_current

G = (g_future . dh_applied - g_future . d0_applied)
    / abs(g_future . d0_applied)

R = ||(S_current - S_history) V||_F^2
    / (0.25 ||(S_half_A - S_half_B) V||_F^2),
V = [d0, dh, g_current].
```

Both candidate directions passed through the production learning-rate and
Euclidean norm constraint before evaluating `G`.  All solves used rank 1600,
fixed Tikhonov `lambda=1e-3`, 1000 walkers, and four independent replay
replicates with four future batches per replicate.

## Results

| regime | known role | median G | G values | median R | median cos(d0, dh) | baseline positive |
|---|---|---:|---|---:|---:|---:|
| static pre1000 | preregistered positive control | -0.1712 | -0.199, -0.143, -0.016, -0.281 | 0.866 | 0.365 | 4/4 |
| early etaS=.95, epoch1000 | known harmful training regime | +0.2624 | -0.434, +0.668, +0.004, +0.521 | 0.920 | 0.466 | 4/4 |
| early etaS=.95, epoch3000 | known harmful training regime | +0.1161 | +0.494, -0.612, -0.262, +1.434 | 1.049 | 0.449 | 4/4 |
| early etaS=.95, epoch5000 | known harmful training regime | +0.8485 | +0.809, +1.077, -0.151, +0.888 | 0.649 | 0.471 | 4/4 |
| late eta=.3 | secondary check | +0.1262 | +0.077, +0.175, -0.078, +0.644 | 0.778 | 0.944 | 4/4 |

The static positive-control requirement failed: all four `G` values were
negative.  More importantly, `G` was positive at all three checkpoints from a
trajectory for which direct E5000 frozen evaluation had already established
that large S averaging was harmful.  `R` also failed to discriminate: it would
reject only epoch3000 (`R>1`) while allowing epoch1000 and epoch5000.

The remaining late `eta=.8/.95` secondary arms could not change this decision
and were cancelled after 26 seconds under the preregistered stop rule.

## Conclusion

**FAIL: neither paired next-batch direction gain `G` nor the action-space
noise/drift ratio `R` is a valid online indicator for deciding whether to mix
historical S in this FermiNet C regime.**

This gives a stronger conclusion than the previous residual audits.  Even
after removing amplitude effects with the actual norm constraint and scoring
only future-batch direction quality, a local replay statistic can prefer S
mixing on a trajectory where direct frozen variance says that mixing is bad.
The missing information is therefore not just current-batch operator mismatch;
it is the multi-step effect of repeatedly changing the update direction and
sampler distribution during the first 5000 optimization steps.

No adaptive gate was implemented, and no E5000 run was launched.  Any future
S-mixing rule must be judged directly by paired E5000 frozen variance (or by a
new indicator first shown prospectively to predict that endpoint); the present
`G`, `R`, residual, transport-ratio, and cosine diagnostics must not be used as
stand-alone gates.

## Artifacts

- Static audit: `/scratch/dexuan1/runs/C_adaptive_S_indicator_20260807/static_pre1000/audit.json`
- Early and late-.3 audits: `/scratch/dexuan1/runs/C_adaptive_S_indicator_15m_20260807/`
- Pilot: `/scratch/dexuan1/runs/C_adaptive_S_indicator_pilot_20260807/`
- Completed jobs: `53663339_0`, `53663361_1` through `53663361_4`
- Cancelled, scientifically redundant late arms: `53664495_5`, `53664495_6`

The runner now stages legacy 8.1 GiB checkpoints to node-local SSD before
loading.  This reduced the sequential copy to 6.2 seconds and changes no
scientific state.
