# N2 n=4096 SPRING instability mechanism

## Verdict

**Classification: A. History accumulation is the primary trigger.**

The direct history/current-correction ratio first exceeds one at epoch 3 in the
`mu=0.99` trajectory, while energy, variance, acceptance, and the centered-local-energy
input norm are still normal.  The history projection then makes the sample-space RHS
grow from 1.42 (epoch 1) to 36.7 (epoch 4), 152 (epoch 5), and 637 (epoch 6).
The current non-history correction consequently grows from 14.2 to 121 by epoch 6.
The constraint begins clipping at epoch 6 and falls below 0.2 at epoch 7; variance
crosses five times its initial-10 median at epoch 8.  This ordering places history
accumulation before sampler/local-energy deterioration.

Float32 error is a severe secondary amplifier.  The stable `mu=0` run has a
float32/float64 direction relative difference of 6--8%, cosine about 0.998, and remains
stable for 80/80 epochs.  In `mu=0.99`, by epoch 8 the float32 direction has norm 2672,
whereas the float64 replay direction has norm 68.3; relative error is 39.25 and cosine
is -0.127.  Thus float32 makes the already history-driven recurrence much worse, but
the first trajectory-specific diagnostic trigger is history domination at epoch 3,
not the float32 discrepancy (the latter threshold is already exceeded at epoch 1 in
both matched runs).

Small-eigenvalue spectral amplification is not supported as the primary trigger.
As the unstable RHS grows, the fraction of RHS energy below damping collapses from
0.288 at epoch 1 to 2.1e-5 at epoch 5 and 7.7e-8 at epoch 8.  The corresponding
low-mode correction fraction falls from 0.478 to 0.279 and then 0.00284.  Sampler
feedback is downstream: acceptance and energy remain close to baseline through the
early history/RHS growth, then depart after the correction and constrained direction
have already expanded.

Remaining ambiguity: an all-float64 counterfactual training trajectory was deliberately
not run, so these data cannot establish whether the exact same history recurrence would
eventually diverge in float64.  They do establish the ordering and mechanism in the
current production float32 trajectory.

## Matched trajectories

| quantity | mu=0 | mu=0.99 |
|---|---:|---:|
| completed finite energy/variance epochs | 80/80 | 34/80 |
| first nonfinite update attempt | none | 35 |
| final finite energy (Ha) | -109.5073 | -93.2655 |
| final finite variance (Ha^2) | 0.9618 | 704.0508 |
| final finite acceptance | 0.4679 | 0.5329 |
| first history/current ratio > 1 | never | 3 |
| first constraint scale < 0.2 | never | 7 |
| first raw norm > 5x first-10 median | never | 7 |
| first variance > 5x first-10 median | never | 8 |
| last finite epoch | 80 | 34 |

The formal onset rule is first met at epoch 1 in both runs because the replay relative
error is already greater than 1e-2.  This is a shared baseline numerical discrepancy,
not a discriminating instability onset.  The first onset unique to the unstable run is
epoch 3 (`||mu history||/||non-history|| > 1`).

Both jobs restored the exact same model parameters, 4096-walker sampler state, PRNG
state, and seed from:

`/scratch/dexuan1/runs/fir_n4096_corrected_retry1/N2/kfac_preliminary/checkpoints/5000.npz`

Resolved-config comparison found only four differences: `mu`, the run output path,
the base output path, and the replay output path.  Epoch-1 energy, variance,
acceptance, and every optimizer diagnostic are identical between the two runs.

Common settings were N2 R=2.068 Bohr (z=+-1.034), `nchains=4096`, restored walkers,
`reload.new_optimizer_state=True`, `reload.reburn=False`, SPRING learning rate
0.0005, damping 0.001, norm constraint 0.001, inverse-time decay 1e-4,
`nsteps_per_param_update=10`, clipping threshold 5, and no checkpoints.

## Mechanism sequence

| epoch | history norm | mu-history / correction | correction norm | raw norm | RHS norm | constraint scale | variance |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 14.17 | 14.17 | 1.42 | 1.000 | 2.02 |
| 3 | 17.49 | 1.209 | 14.32 | 20.29 | 10.31 | 1.000 | 1.93 |
| 4 | 20.29 | 1.264 | 15.89 | 23.01 | 36.71 | 1.000 | 1.69 |
| 5 | 23.01 | 0.700 | 32.54 | 44.38 | 152.2 | 1.000 | 1.71 |
| 6 | 44.38 | 0.363 | 121.1 | 159.6 | 637.0 | 0.397 | 3.17 |
| 7 | 159.6 | 0.318 | 497.2 | 652.8 | 2635 | 0.097 | 8.81 |
| 8 | 652.8 | 0.319 | 2029 | 2672 | 10979 | 0.0237 | 20.03 |
| 10 | 10963 | 0.312 | 34759 | 45541 | 192690 | 0.00139 | 65.11 |

The history does not remain larger than the correction: after it triggers the feedback,
the newly solved correction becomes dominant and aligns strongly with history (cosine
0.833 at epoch 6 and 0.994 at epoch 8).  This is a positive-feedback recurrence through
the history projection in the RHS, rather than a simple additive history vector staying
larger forever.

## Float32 versus float64 replay

| run/epoch | q32 norm | q64 norm | relative error | cosine | residual32 | residual64 | constraint32 | constraint64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mu=0 / 1 | 14.17 | 14.06 | 0.0663 | 0.9979 | 1.161 | 0.315 | 1.000 | 1.000 |
| mu=0 / 8 | 12.70 | 12.59 | 0.0659 | 0.9979 | 1.394 | 0.213 | 1.000 | 1.000 |
| mu=.99 / 1 | 14.17 | 14.06 | 0.0663 | 0.9979 | 1.161 | 0.315 | 1.000 | 1.000 |
| mu=.99 / 8 | 2672 | 68.32 | 39.25 | -0.127 | 4.163 | 1.06e-4 | 0.0237 | 0.926 |
| mu=.99 / 10 | 45541 | 809.95 | 56.41 | -0.175 | 4.252 | 8.93e-5 | 0.00139 | 0.0782 |
| mu=.99 / 20 | 2.23e16 | 2.23e14 | 100.1 | -0.0338 | 18.77 | 2.01e-4 | 2.85e-15 | 2.85e-13 |

The float64 constraint scale is recomputed as
`min(1, sqrt(C)/(learning_rate(epoch)*||q64||))`.  An earlier raw diagnostic field with
`float64_constraint_scale` stored effective step size rather than this dimensionless
scale; the corrected value is explicitly written in the collected per-epoch CSV.

## Commands and jobs

Validation:

```bash
pytest -q tests/units/updates/test_spring_history.py
pytest -q tests/units/train/test_vmc.py::test_training_writer_preserves_spring_diagnostics
bash -n experiments/fir_n4096_smoke/scripts/N2_spring_mechanism_array.sh
```

Final matched run and collection:

```bash
sbatch experiments/fir_n4096_smoke/scripts/N2_spring_mechanism_array.sh
# array 49610209: task 0 mu=0; task 1 mu=0.99
sbatch --dependency=afterany:49610209 experiments/fir_n4096_smoke/scripts/N2_collect_spring_mechanism.sh
# collector allocation 49610210 was cancelled while pending; collection was run locally
python experiments/fir_n4096_smoke/collect_spring_mechanism.py
```

The diagnostic array command line is preserved verbatim in
`experiments/fir_n4096_smoke/scripts/N2_spring_mechanism_array.sh`.  The first two
attempts (`49608476`, `49609441`) and their logs/directories were preserved; they exposed
diagnostic-only config/CSV logging defects and were not used for the mechanism result.

## Outputs

- Full merged per-epoch CSVs: this directory, files ending `_per_epoch.csv`
- Selected spectral summary: `selected_spectral_summary.csv`
- Selected replay comparison: `selected_float32_float64_replay.csv`
- Raw replay matrices/RHS/history projections:
  `/scratch/dexuan1/runs/fir_n4096_spring_mechanism_retry2/N2/replay/`
- Raw final trajectories:
  `/scratch/dexuan1/runs/fir_n4096_spring_mechanism_retry2/N2/`
- Plots: `energy_variance.svg`, `history_nonhistory_norm.svg`, `history_ratio.svg`,
  `raw_displacement.svg`, `constraint_scale.svg`, `low_eigen_rhs_fraction.svg`, and
  `float32_float64_replay_error.svg`
