# N2 cluster-envelope Rayleigh--Ritz and rotation-EF test

Date: 2026-08-09

## Experimental question

The first cluster-envelope prototype retained the span of the last three
rank-32 SSI bases but replaced the curvature inside the resulting 96-dimensional
space by one scalar mean. This experiment tested:

1. **A -- Rayleigh--Ritz:** solve the full current-batch projected Fisher system
   `(Q.T @ S_current @ Q + lambda I) c = Q.T @ g_current`.
2. **B -- curvature-weighted QR:** scale historical columns before QR by
   positive curvature-dependent weights.
3. **C -- rotation error feedback:** add a decayed parameter-force accumulator
   to A. Only the portion captured by the newly rotated `Q` enters the Ritz RHS;
   the uncaptured part is retained for later steps.

All features are default-disabled. The old scalar prototype remains available.

## Configuration

- System: equilibrium N2 at 2.016 Bohr, `(7, 7)` electrons.
- Start: common KFAC-pre5000 checkpoint.
- Training: 1000 walkers, 10 MCMC steps/update, mean-centered clipping 5,
  `reburn=False`, seed 0.
- Optimizer: inverse-time LR 0.002, decay `1e-4`, fixed Tikhonov lambda `1e-3`,
  norm constraint `1e-3`, SSI 40/2.
- Envelope: current rank 32, history length 3, maximum numerical rank 96.
- C feedback: reinjection alpha 0.2, state decay rho 0.95, cap `10 ||g||`.
- Frozen evaluation: 2000 walkers, burn-in 10000, 2000 inference iterations,
  10 MCMC steps/measurement.

## Unit and algebraic checks

Eight targeted tests passed on Slurm job `53854133`. They cover both scalar and
Ritz integrated updates, the legacy-checkpoint-compatible mode aliases,
envelope orthonormalization, rotation invariance, and EF state evolution.

Scheme B does not define an independent optimizer under the stated
specification. For any full-rank positive diagonal `D`,

`col([U_t, U_{t-1}, ...] D) = col([U_t, U_{t-1}, ...])`.

Consequently QR produces the same projector, and the Rayleigh--Ritz update is
basis invariant. A unit test verifies equality to numerical tolerance. B would
only change the result if it were coupled to another truncation/rank-allocation
rule, which was not part of the proposed formula; no redundant GPU run was made.

## 200-step screen

Tail-100 medians use unclipped local-energy quantities.

| Method | Energy (Ha) | Variance | Delta energy vs hard | Variance / hard |
|---|---:|---:|---:|---:|
| hard rank-200 | -109.487896 | 2.56146 | -- | 1.000x |
| scalar envelope | -109.459686 | 3.81569 | +28.21 mHa | 1.490x |
| A: Ritz-96 | -109.473106 | 3.01089 | +14.79 mHa | 1.175x |
| C: Ritz-96 + rotation EF | -109.488380 | 2.89567 | -0.48 mHa | 1.130x |

A recovers 13.42 mHa and reduces variance by 21.1% relative to the scalar
prototype, directly confirming that scalarizing the projected curvature was a
major error. C then recovers another 15.27 mHa, so it was the only candidate
advanced to E5000.

## E5000 frozen results

The hard rank-200 values are the existing matched-start, same-seed control.

| Epoch | Method | Frozen energy (Ha) | s.e. (Ha) | Frozen variance | C / hard variance |
|---:|---|---:|---:|---:|---:|
| 1000 | hard rank-200 | -109.483997 | 0.001275 | 3.11845 | -- |
| 1000 | C: Ritz-96 + EF | -109.475840 | 0.001344 | 3.44561 | 1.105x |
| 2500 | hard rank-200 | -109.485341 | 0.001297 | 3.49153 | -- |
| 2500 | C: Ritz-96 + EF | -109.483871 | 0.001363 | 3.81254 | 1.092x |
| 5000 | hard rank-200 | **-109.493322** | 0.001226 | **3.03990** | -- |
| 5000 | C: Ritz-96 + EF | -109.485583 | 0.001342 | 3.49005 | **1.148x** |

C is 8.16 mHa above hard at E1000, nearly tied at E2500, and 7.74 mHa above
hard at E5000. Its E5000 variance is 14.8% higher. Thus the favorable 200-step
energy comparison was a short-window fluctuation, not a durable reversal.

Relative to the old scalar envelope's E5000 frozen point (-109.471943 Ha,
variance 5.79229), C improves energy by 13.64 mHa and variance by 39.7%. The
proposed fixes are real improvements, but they do not close the hard-WSSR gap.

## Runtime and health

- C steady runtime from checkpoint timestamps, epochs 500--5000: 0.1647 s/step.
- Existing hard rank-200: 0.1789 s/step; C is about 8.0% faster.
- Known SPRING scale: about 0.091 s/step; C is about 1.81x slower.
- Envelope numerical rank remained 96.
- E5000 tail-100 median EF-state/gradient norm ratio: 0.286.
- EF cap triggers: 0 over all 5000 steps.
- Projected regularized curvature condition number: about 1883.
- Norm-constraint scale: 1.0; the constraint was inactive.

The failure is therefore not caused by envelope collapse, EF explosion,
clipping, or scalar curvature. It is a structural rank/subspace limitation:
the current 96-dimensional envelope does not retain enough of the useful
current-batch SR geometry, while accumulating old top-cluster directions does
not substitute for a broader current solve.

## Decision

- **A:** mathematically necessary and substantially better than scalarization,
  but insufficient alone.
- **B:** exact no-op without an additional truncation/rank-allocation rule.
- **C:** stable and beneficial at very short horizon, but fails the E5000
  frozen accuracy gate.

Do not run these configurations to 100k and do not tune alpha/rho around C.
The evidence now says that improving how the same 96-dimensional historical
envelope is weighted is unlikely to beat hard rank-200 or SPRING. Any next
candidate must change the information capacity or allocation itself (for
example a larger/current-heavy projected space with an explicit wall-time
budget), not add more history mechanics to this fixed envelope.

## Artifacts

- A short screen: `/scratch/dexuan1/runs/N2_cluster_envelope_ritz_A200_20260809_retry1`
- C short screen: `/scratch/dexuan1/runs/N2_cluster_envelope_ritz_C200_20260809`
- C E5000 and frozen points: `/scratch/dexuan1/runs/N2_cluster_envelope_ritz_C_E5000_20260809`
- A training job: `53853529`
- C short job: `53854413`
- C E5000 job: `53854655`
- C frozen jobs: `53855054`, `53855271`, `53855760`
