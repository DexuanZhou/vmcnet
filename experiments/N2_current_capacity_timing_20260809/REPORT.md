# N2 current-only Ritz capacity audit

Date: 2026-08-09

## Question and controls

This is a 100-step algebra/timing audit, not an accuracy screen.  The three
arms use current-batch Rayleigh--Ritz updates with actual ranks 200, 384, and
512.  Each SSI computes one audit-only extra mode (`d+1`) so that
`sigma_d/sigma_(d+1)` is observable; the parameter update remains strictly
limited to the first `d` modes.  There is no history, EF, complement, S/g EMA,
or checkpoint write.

The system and sampling protocol match the N2 experiments: equilibrium N2 at
2.016 Bohr, common KFAC-pre5000 checkpoint, 1000 walkers, 10 MCMC moves/update,
clipping 5, `reburn=False`, learning rate 0.002, inverse-time decay `1e-4`,
Tikhonov lambda `1e-3`, norm constraint `1e-3`, and SSI 40/2.

## Results

Tail medians are over epochs 20--100.  Time is estimated from the slope of
completed epochs in the same window, excluding startup/JAX compilation.

| d | s/step | peak GPU GiB | sigma_d/sigma_(d+1) | weakest-10% update mass | projected condition | raw-gradient capture |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 0.2184 | 13.41 | 1.0158 | 23.93% | 1,687 | 0.9946 |
| 384 | 0.3257 | 17.41 | 1.0127 | 27.27% | 7,325 | 0.9995 |
| 512 | 0.4043 | 30.79 | 1.0096 | 27.85% | 15,852 | 1.0003 |

All arms completed 100/100 finite updates.  The small capture value above one
at d=512 is a numerical orthogonality diagnostic at the 3e-4 level, not a
physical projection gain.

## Joint interpretation

1. **Throughput scaling.** From d=200 to 512, rank grows 2.56x and step time
   grows 1.85x.  The fitted local power is about d^0.66 (0.61 over 200--384 and
   0.75 over 384--512), so this range does not yet show quadratic small-matrix
   domination; fixed VMC/Jacobian costs remain large.  Nevertheless the
   absolute cost is already high: 2.40x, 3.58x, and 4.44x the established
   approximately 0.091 s/step SPRING scale.  Device memory also has a sharp
   knee at 512 (30.79 GiB on a 40-GiB MIG slice).

2. **Spectral boundary leakage.** Every tail ratio is close to one and becomes
   closer to one as d increases.  There is no natural gap at 200, 384, or 512;
   increasing d does not complete an isolated physical curvature envelope.

3. **Boundary importance.** The weakest-curvature 10% of retained Ritz modes
   carry 24--28% of the preconditioned coefficient norm, well above their 10%
   dimensional share.  Thus the flat boundary is not harmless spectral dust:
   truncation acts exactly where inverse-curvature weighting makes modes
   important.  This also explains the condition growth from 1.7e3 to 1.6e4.

4. **Historical selection gain.** It is mathematically undefined in these
   current-only arms and is intentionally not reported as zero.  The preceding
   fixed-budget experiment is the appropriate causal evidence: Selected96
   improved frozen variance by 8.23% over current-only H96 and by 30.3% over
   unconditional E96, but remained 41.4% worse than hard current-only rank200
   and ran about 1.9x slower than SPRING.  The new score-mass telemetry is kept
   for a future history-selection candidate, but the present capacity/cost
   gate supplies no reason to launch such a candidate at d=200--512.

## Decision

**No-go for dynamic expansion to d=384/512 and no E5000 launch.**  The route
does not encounter a clean spectral boundary, the modes at the boundary still
matter strongly to the update, conditioning deteriorates rapidly, and cost
and memory rise without an algebraic indication of saturation.  Among these
three choices, d=200 is the only defensible cost point, and its already-known
E5000 accuracy remains behind SPRING.  Larger historical envelopes should not
be used to consume the remaining capacity budget.

## Provenance

- Corrected Slurm array: `53867546` (all three tasks `COMPLETED 0:0`).
- Run root: `/scratch/dexuan1/runs/N2_current_capacity_timing_tailfix_20260809`
- Machine-readable table: `summary.csv` and `summary.json` in the run root.
- Plot: `capacity_audit.png` in the run root.
- The first array `53867021` used an audit-only d+1 array shape but initialized
  the legacy active-rank mask at d, which zeroed sigma_(d+1).  Its tail ratios
  are invalid and it is excluded; the corrected array initializes the internal
  SSI rank at d+1 while retaining a d-dimensional update.
