# Early-C curvature-transport audit

## Outcome

The idealized first-order finite-difference transport is locally accurate, but
it corrects a parameter-staleness component that is much smaller than ordinary
walker-batch variation. The pre-registered gate therefore failed and no E5000
training is authorized from this result.

The audit used the early C fixed-lambda rank-1600 WSSR epoch-1000 checkpoint,
1000 walkers, and eight independent one-step replicates. Each `delta` was the
actual post-learning-rate and post-norm-constraint parameter update. No dense
parameter-space Fisher matrix was constructed.

| metric | median | range |
|---|---:|---:|
| actual update norm | 0.03162278 | 0.03162278--0.03162278 |
| untransported same-coordinate action error | 0.0129641 | 0.0122616--0.0147854 |
| transported same-coordinate action error | 0.00168660 | 0.00122900--0.00206893 |
| transported / untransported error | 0.128475 | 0.088264--0.150687 |
| new-batch sampling action error | 0.0626408 | 0.0379035--0.0758149 |
| sampling / parameter-staleness error | 4.66512 | 2.95837--5.54958 |
| total new-batch error before transport | 0.0640829 | 0.0392170--0.0778950 |
| total new-batch error after transport | 0.0628490 | 0.0382292--0.0747111 |
| relative total-error improvement | 0.0225166 | 0.001641--0.056428 |

Transport won the same-coordinate comparison in 8/8 replicates and reduced
that error by about 87%. Its 95% bootstrap interval for the median remaining
error fraction was `[0.09597, 0.13456]`. Nevertheless, batch variation was
about 4.67 times the uncorrected parameter-staleness error, so correcting the
latter improved the actual new-batch action error by only 2.25% at the median.

The `actual_delta` probe has a large *relative* stale-action error, but this is
an action whose absolute norm is small; earlier direct logs put `||S delta||`
at only roughly 0.5--1.2% of `||g||`. It therefore does not overturn the
aggregate magnitude-aware gate.

## Scope

The tested transport

`S_transport v = 2 S(theta + delta/2) v - S(theta) v`

is an intentionally favorable finite-difference approximation: it evaluates
the score factor at the midpoint. A production directional derivative or
geometric transport would be at least as expensive and need not be as accurate.
This audit does not prove that every possible geometric transport is useless;
it shows that one-step parameter-induced curvature staleness is not large
enough, relative to sampling noise, to justify that implementation on the
current early-C protocol.

## Infrastructure note

Slurm job `53763134` completed all eight scientific replicates and then exited
at the final `git rev-parse` provenance lookup because the immutable source
snapshot is not itself a Git worktree. The complete JSON lines were recovered
from stdout into the authoritative result below. The provenance lookup has
been fixed for future invocations; no scientific rerun was needed.

Authoritative result:
`/scratch/dexuan1/runs/C_curvature_transport_audit_20260808_retry1/early_epoch1000/audit.json`
