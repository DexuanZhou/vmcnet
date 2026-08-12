# Strict FermiNet architecture-multilevel pilot: results

The production hierarchy keeps 16 dense determinants fixed and nests the
equivariant network exactly:

| level | hidden dimensions | parameter count |
|---|---|---:|
| L0 | `((64, 8), (64,))` | 27,624 |
| L1 | `((128, 16), (128, 16), (128,))` | 134,944 |
| L2 | `((256, 16), (256, 16), (256, 16), (256,))` | 670,896 |

The L0 -> L1 and L1 -> L2 prolongations preserve every tested wavefunction
sign and log-amplitude (maximum fp32 log-amplitude discrepancy
`5.72e-6`).  The optimizer schedule count and representable history vectors
are prolonged rather than reset.  The cached walker log-amplitudes are
refreshed at the unchanged walker positions, without MCMC reburning.

## Initial 2,000-step pilot

The following numbers are training metrics, not the final independent frozen
evaluation.  `tail variance` is the mean of the last 400 recorded steps.

| optimizer | schedule | last energy | last variance | tail variance | status |
|---|---|---:|---:|---:|---|
| SPRING | direct L2 | -37.436584 | 9.468954 | 8.684592 | completed |
| SPRING | L0 -> L1 | -37.847904 | 0.075954 | 0.099569 | completed at epoch 1,000 |
| SPRING | L0 -> L1 -> L2, unscaled sidecar | -37.748828 | 2.298359 | 1.321261 | NaN at epoch 1,099 |
| WSSR | direct L2 | -37.827095 | 0.335124 | 0.321976 | completed |
| WSSR | L0 -> L1 -> L2 | -37.826164 | 0.318403 | 0.350544 | completed |

The first SPRING result shows that exact function preservation alone does not
make the newly exposed score directions benign.  The multilevel L1 segment is
substantially better, but activating a full-amplitude random L2 sidecar at
once destabilizes SPRING.  A function-preserving sidecar-scale gate is
therefore being tested for 200 steps with scales 0.1 and 0.01.  The new
orbital couplings remain exactly zero at conversion, so this changes neither
the converted wavefunction nor its old-parameter tangent; it only controls
the initial scale of the dormant fine-only features that become learnable.

## Safe-capacity activation and matched SPRING comparison

Two 200-step L1 -> L2 gates isolated the dormant-feature activation scale:

| sidecar scale | outcome | endpoint/tail training variance |
|---:|---|---:|
| 0.1 | NaN at epoch 1,065 | 0.8690 before termination |
| 0.01 | stable through epoch 1,200 | 0.05922 |

Only the 0.01 arm was continued.  A direct-L2 scale-0.01 control was then
run from the same optimizer initial state so that the benefit of staging was
not compared against the already-known unsafe full-amplitude initialization.
Both arms used 2,000 total SPRING epochs and ended in the identical L2
architecture.  The multilevel schedule used 500 L0 epochs, 500 additional L1
epochs, and 1,000 additional L2 epochs; direct used 2,000 L2 epochs.

| SPRING capacity schedule | endpoint energy | tail-400 training variance | frozen energy | frozen variance |
|---|---:|---:|---:|---:|
| direct L2, sidecar 0.01 | -37.84320450 | 0.04436706 | -37.84248592 | 0.11405280 |
| L0 -> L1 -> L2, L2 sidecar 0.01 | -37.84445953 | 0.03381278 | -37.84328868 | 0.09448237 |

The strict multilevel schedule reduced independent frozen variance by
17.16% (`multilevel/direct = 0.82841`) and lowered frozen energy by
0.000803 Ha in this one-seed pilot.  This is positive evidence for staged
capacity training, but the energy separation is only about twice the combined
reported Monte Carlo standard error, so replication is needed before making a
strong accuracy claim.

## WSSR comparison

WSSR was stable without sidecar scaling, but this short schedule did not
benefit from architecture staging:

| WSSR capacity schedule | endpoint energy | tail-400 training variance | frozen energy | frozen variance |
|---|---:|---:|---:|---:|
| direct L2 | -37.82709503 | 0.32197615 | -37.83226090 | 0.75800996 |
| L0 -> L1 -> L2 | -37.82616425 | 0.35054417 | -37.83031879 | 0.83762671 |

Thus exact optimizer-state prolongation works for WSSR, but the tested
500/1,000/2,000 schedule gives no WSSR accuracy gain.  The best strict
multilevel SPRING frozen variance is 0.12465 times the WSSR-direct value and
0.11280 times the WSSR-multilevel value in this deliberately short pilot.
These cross-optimizer ratios are descriptive only, not a matched scientific
comparison: the reported SPRING arms use the subsequently selected
`sidecar_scale=0.01`, whereas both WSSR arms use the original default
`sidecar_scale=1.0`.  They must not be used to conclude that the optimizer
alone caused the gap.

## Conclusion

Strict nesting is feasible for FermiNet width and depth growth while keeping
the determinant count fixed.  Small-model training can be continued in a
larger model with no wavefunction jump, no walker reburn, and prolonged
SPRING/WSSR state.  The newly created fine-only score directions nevertheless
need controlled activation: exact equality of function values alone is not a
stability guarantee.  A 0.01 dormant-sidecar scale produced a stable and
better SPRING multilevel path.  The WSSR arms were stable at scale 1.0 but did
not benefit from their tested capacity schedules; because the sidecar policy
was not matched, this pilot does not support a clean SPRING-versus-WSSR
accuracy conclusion.

## Job provenance

- `53772797`: original four-arm training array.  Tasks 0, 2, and 3 completed;
  task 1 exposed the unsafe unscaled SPRING L1 -> L2 transition.
- `53773613`: sidecar-scale gate.  Scale 0.1 terminated on a scientific NaN;
  scale 0.01 completed.
- `53773877`: the scale-0.01 continuation numerically completed through epoch
  2,000 and wrote its checkpoint.  Slurm reports `FAILED` only because the
  launcher pre-created the nominal output directory, VMCNet selected the `_1`
  suffix, and the final shell assertion checked the unsuffixed path.
- `53773882`: matched direct-L2 scale-0.01 control, completed.
- `53773706`, `53774023`, and `53774135`: independent frozen evaluations,
  completed.

## Five-level WSSR follow-up

To test whether the original hierarchy was too coarse, the first 1,000 WSSR
epochs were repartitioned into four 250-epoch levels while retaining 1,000
final L2 epochs and 2,000 total epochs:

`64 -> 96 -> 128 -> 192 -> 256`, with parameter counts
`27,624 -> 51,132 -> 134,944 -> 392,880 -> 670,896`.

All four conversions had zero sign mismatches.  Their maximum log-amplitude
errors were respectively `0`, `0`, `0`, and `1.43e-6`.

| WSSR capacity schedule | wall time | tail-400 training variance | frozen energy | frozen variance |
|---|---:|---:|---:|---:|
| direct L2 | 10:10 | 0.321976 | -37.83226090 | 0.75800996 |
| three levels | 8:45 | 0.350544 | -37.83031879 | 0.83762671 |
| five levels | 14:23 | 0.366281 | -37.82978044 | 0.94146464 |

The five-level frozen variance is 12.40% worse than the three-level result and
24.20% worse than direct L2.  This is a valid result for the *tested
construction*, but it is not an isolated test of level count.  Every conversion
used the then-default `sidecar_scale=1.0`: direct L2 activates one full random
sidecar, the three-level path activates two, and the five-level path activates
four.  Level count is therefore confounded with the number of full-amplitude
dormant-feature activations.  This matters because the SPRING gate in the same
pilot independently established that this activation scale can dominate
stability and selected `sidecar_scale=0.01`.

There is also a WSSR-specific transition caveat.  The converter exactly embeds
the stored coarse `sr_o` and `u` rows and initializes all new parameter rows to
zero.  This is the faithful prolongation of the state that actually existed in
the coarse optimizer, but it is not the counterfactual historical Fisher/S
state that would have been measured in the fine parameterization: the newly
opened orbital-coupling coordinates already have nonzero score derivatives.
Thus WSSR's curvature history is temporarily incomplete in new coordinates at
every transition.  Repeating this four times may penalize the five-level path.

The measured 14:23 wall time also includes four separate model
initialization/JAX-compilation/conversion passes, so it must not be interpreted
as intrinsic optimized multilevel-WSSR step cost.  The aligned trajectory was
still worse before this overhead was counted, but the defensible conclusion is
only that naive full-amplitude five-level staging with zero-extended historical
curvature did not help.  A fair level-count test must use the same controlled
sidecar policy (for example 0.01) and the same WSSR transition policy in both
three- and five-level arms, with compilation reported separately.

A post-hoc checkpoint audit verified all four five-level transitions.  In every
case the epoch, PRNG key, walker positions, optimizer schedule count, spectral
state, and active-rank metadata were preserved; old parameter, `sr_o`, and `u`
entries agreed exactly (maximum error 0), while every newly introduced `sr_o`
and `u` row was zero.  A model regression test also verifies equality of the
old-coordinate parameter scores after prolongation.  No accidental corruption
of WSSR state was found.

Jobs: five-level training `53774930`; frozen evaluation `53774931`; checkpoint
state audit `53777036`.

## Matched sidecar-scale correction

The three- and five-level WSSR schedules were rerun with
`sidecar_scale=0.01` at every transition.  All other scientific settings were
held fixed.  Two paired template seeds (`314159` and `271828`) started from
the same KFAC-pre1000/WSSR checkpoint; hence these are paired dormant-feature
initialization replicates rather than independent KFAC pretraining seeds.

| schedule | paired seed | frozen energy | frozen variance | five/three variance |
|---|---:|---:|---:|---:|
| three levels | 0 | -37.83174443 | 0.83065915 | -- |
| five levels | 0 | -37.83072985 | 0.84837960 | 1.02133 |
| three levels | 1 | -37.83083297 | 0.75074046 | -- |
| five levels | 1 | -37.83103384 | 0.78521006 | 1.04591 |

The arithmetic-mean frozen variances are `0.79069981` (three levels) and
`0.81679483` (five levels), a five/three ratio of `1.03300`.  The paired
geometric-mean ratio is `1.03355`.  Thus five levels remain about 3--3.5%
worse in this two-replicate short pilot, but the result is much weaker than
the original scale-1.0 comparison.

For the directly matched seed 0, changing the sidecar scale improves the
three-level variance only from `0.83762671` to `0.83065915` (0.83%), while it
improves the five-level variance from `0.94146464` to `0.84837960` (9.89%).
Consequently the old five/three gap shrinks from 12.40% to 2.13%.  This
confirms that repeated full-amplitude sidecar activation materially biased the
original five-level result.  The corrected conclusion is not that WSSR is
damaged by multilevel training; it is only that, after controlling activation,
adding two intermediate levels did not outperform the three-level schedule
under the fixed 2,000-step budget.

The five-level seed-0 L15-to-L2 conversion initially tripped the fixed `1e-5`
fp32 equality guard at `1.04904e-5`.  A separate audit found zero sign
mismatches, median error `4.77e-7`, and an independent 128-sample maximum of
`1.91e-6`.  The arm was resumed from its completed epoch-1000 checkpoint with
an explicit `2e-5` verification tolerance; no scientific state or training
setting changed.

Jobs: paired training `53777786`; original paired evaluation `53777787`;
conversion audit `53778945`; seed-0 resume `53779254`; final seed-0 frozen
evaluation `53780167`.

## Sidecar-scale 1e-4 follow-up

The same paired three- versus five-level matrix was repeated with
`sidecar_scale=1e-4`; all other training and evaluation settings were held
fixed relative to the matched `sidecar_scale=0.01` experiment.

| schedule | paired seed | frozen energy | frozen variance | five/three variance |
|---|---:|---:|---:|---:|
| three levels | 0 | -37.83301186 | 0.82515465 | -- |
| five levels | 0 | -37.82998092 | 1.24998079 | 1.51484 |
| three levels | 1 | -37.83191121 | 0.78278700 | -- |
| five levels | 1 | -37.83189204 | 0.71581203 | 0.91444 |

The arithmetic-mean frozen variances are `0.80397083` (three levels) and
`0.98289641` (five levels), giving a five/three ratio of `1.22255`.  The
paired geometric-mean ratio is `1.17696`.  The two five-level replicates have
opposite outcomes: seed 1 is 8.56% better than its paired three-level arm,
whereas seed 0 is 51.48% worse.  The corresponding two-seed mean frozen
energies are `-37.83246154` and `-37.83093648` Ha, respectively.

Relative to the matched `sidecar_scale=0.01` runs, the geometric-mean frozen
variance worsens by 1.77% for three levels and 15.89% for five levels.  Thus
reducing the scale from `1e-2` to `1e-4` does not yield a robust five-level
benefit under the 2,000-step budget; instead it substantially increases
replicate sensitivity.  This is consistent with newly widened channels being
activated too weakly to grow reproducibly on the available time scale, though
two paired replicates are insufficient to identify that mechanism
definitively.  Among the tested transition amplitudes, `0.01` remains the best
supported setting.

The tail-400 training variances were tightly clustered (`0.30825`, `0.31489`,
`0.31254`, and `0.31461` for three/seed0, three/seed1, five/seed0, and
five/seed1), so the seed-0 five-level degradation was visible only in the
independent frozen evaluation.  This reinforces the need to use frozen
variance rather than the in-training tail statistic for this comparison.

Jobs: paired training `53781363`; paired frozen evaluation `53781364`.

## Transition-only WSSR basis refresh

The central premise behind a proposed hybrid-basis repair was tested at the
matched three-level `L1 -> L2` transition with `sidecar_scale=0.01`.  From the
same converted checkpoint, the control used the production two warm subspace
iterations on the first L2 update (`warm2`), while the treatment used forty
iterations on that update (`refresh40`).  Both arms then returned to warm-SSI2
for another 199 epochs.  Rank remained 800 and every other optimizer, sampler,
and schedule setting was paired.

The conversion itself places exactly zero WSSR basis mass in the 535,952 new
coordinates.  Nevertheless, the ordinary warm-SSI2 update moved substantial
mass into them in one step:

| seed | mode | new-coordinate U mass after 1 step | after 200 steps | first-update mass in new coordinates |
|---:|---|---:|---:|---:|
| 0 | warm2 | 0.28498 | 0.23166 | 0.27483 |
| 0 | refresh40 | 0.29572 | 0.23230 | 0.30695 |
| 1 | warm2 | 0.28636 | 0.23113 | 0.29926 |
| 1 | refresh40 | 0.29758 | 0.23117 | 0.31540 |

Thus forty iterations add only about 1.1 percentage points of new-coordinate
basis mass on the first update.  By epoch 1,200 the treatment-control
difference is below 0.07 percentage points.  This directly falsifies the claim
that zero-padding leaves WSSR unable to see the new parameter space during
training: the current-batch operator transfers the basis into that space
immediately.

The independent frozen result is also negative:

| seed | warm2 energy | warm2 variance | refresh40 energy | refresh40 variance | refresh/warm2 |
|---:|---:|---:|---:|---:|---:|
| 0 | -37.82938875 | 0.86950313 | -37.82896760 | 1.00135554 | 1.15164 |
| 1 | -37.82852926 | 0.88948350 | -37.82935847 | 1.20340171 | 1.35292 |

The arithmetic-mean frozen variance worsens from `0.87949331` to
`1.10237863` (25.34%); the paired geometric-mean ratio is `1.24823`.  The
tail-100 in-training variances are nearly unchanged, so the treatment did not
fail by an obvious training divergence.  A more converged transition basis is
therefore not a remedy for the multilevel WSSR accuracy gap under this
protocol.  This test addresses the proposed repair's causal premise; it does
not claim that every possible coordinate-budgeted hybrid basis is identical
to the SSI40 treatment.

Jobs: training `53787055`; frozen evaluation `53787056`; basis-state audit
`53787999`.  Full state metrics are stored in
`/scratch/dexuan1/runs/C_wssr_transition_basis_refresh_20260808/transition_basis_audit.json`.
