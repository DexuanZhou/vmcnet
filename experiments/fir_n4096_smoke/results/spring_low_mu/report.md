# Controlled N2 n=4096 low-mu SPRING sweep

## Conclusion

The requested positive-history values are all **UNSTABLE**.  Only `mu=0`
(MinSR) completes 200/200 epochs.  Lowering `mu` from the previously tested
0.90--0.99 range to 0.5--0.8 does not produce a useful stable intermediate;
it only changes how quickly the same history-projection/raw-direction runaway
reaches the controlled stopping threshold.

Interpretation: **A for mu=0 (history accumulation completely suppressed), and
B for every tested positive mu (the same instability is delayed, not removed).**
There is no evidence for C or D in this sweep.

Per the requested refinement rule, because `mu=0.7` is unstable, the next
nominal value would be **mu=0.6**.  It was not submitted.  Since `mu=0.5` is
already unstable, the evidence also suggests that the actual stable boundary,
if nonzero, is below 0.5.

## Matched initialization and configuration

All four tasks restored model parameters, the 4096 equilibrated walkers, PRNG
state, and sampler state from:

`/scratch/dexuan1/runs/fir_n4096_corrected_retry1/N2/kfac_preliminary/checkpoints/5000.npz`

Resolved `config.json` files were flattened and compared.  Relative to `mu=0`,
each file differs only in:

- `vmc.optimizer.spring.mu`;
- `logdir`;
- `base_logdir`.

The source-checkpoint path files are byte-identical for all four tasks.  Common
settings include N2 R=2.068 Bohr, `nchains=4096`, restored walker/PRNG state,
`reload.new_optimizer_state=True`, `reload.reburn=False`, learning rate 0.0005,
inverse-time decay 1e-4, damping 0.001, norm constraint 0.001,
`nsteps_per_param_update=10`, mean clipping at 5 standard deviations,
`check_for_nans=True`, 200 requested epochs, and no checkpoints.

Epoch-1 energy, variance, acceptance, raw direction, and all history diagnostics
are identical across all four runs.

## Stability results

The norm constraint is counted active when its scale is below 0.999.  Sustained
history domination means ratio greater than one for at least five consecutive
epochs.  A monotone/sustained-growth flag also detects five consecutive history
norm increases or a last-20 median over twice the first-10 median.

| mu | completed / last finite | controlled stop | first ratio > 1 | max ratio | max sample-space history projection | min constraint scale | active epochs (fraction) | max variance | sustained domination | sustained growth | classification |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|
| 0.0 | 200 / 200 | none | never | 0 | 0 | 1.0000 | 0 (0%) | 2.017 | no | no | STABLE |
| 0.5 | 17 / 17 | raw direction > 100x at 17 | never | 0.634 | 6,747 | 0.0398 | 5 (29.4%) | 78.73 | no | yes | UNSTABLE |
| 0.7 | 17 / 17 | variance > 100x at 17 | never | 0.956 | 21,804 | 0.0165 | 6 (35.3%) | 192.51 | no | yes | UNSTABLE |
| 0.8 | 13 / 13 | raw direction > 100x at 13 | 4 | 1.121 | 21,182 | 0.0123 | 4 (30.8%) | 52.32 | no | yes | UNSTABLE |

The direct history/non-history ratio alone is not a sufficient alarm: at mu=0.5
and 0.7 it remains below one, even while the sample-space history projection and
newly solved non-history correction run away.  This matches the previous mechanism
diagnosis: history first perturbs the RHS, after which the current correction can
become larger than the explicit additive history term.

First strong constraint activity (`scale < 0.2`) occurs at epoch 15 for mu=0.5,
epoch 15 for mu=0.7, and epoch 11 for mu=0.8.  At their final recorded epochs:

| mu | energy (Ha) | variance | history norm | projection norm | raw norm | constraint scale |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 / epoch 17 | -113.262 | 78.73 | 753 | 6,747 | 1,592 | 0.0398 |
| 0.7 / epoch 17 | -116.056 | 192.51 | 1,047 | 21,804 | 3,843 | 0.0165 |
| 0.8 / epoch 13 | -112.513 | 52.32 | 1,531 | 21,182 | 5,129 | 0.0123 |

These low energies are accompanied by exploding variance and constrained runaway
directions; they are not optimization improvements.

## Final-50 optimization and performance statistics

Only mu=0 completed enough epochs for a final-50 estimate.

| mu | tail-50 energy mean +/- SEM (Ha) | tail-50 variance mean | median acceptance | median sec/epoch | mean / median constraint scale | best rolling-20 energy | rolling window end |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | -109.519877 +/- 0.001524 | 0.83855 | 0.46663 | 0.4781 | 1.0000 / 1.0000 | -109.529605 | 20 |
| 0.5 | unavailable (terminated at 17) | -- | -- | 0.4790 available steady | -- | -- | -- |
| 0.7 | unavailable (terminated at 17) | -- | -- | 0.4760 available steady | -- | -- | -- |
| 0.8 | unavailable (terminated at 13) | -- | -- | 0.4765 available steady | -- | -- | -- |

The failed-run timing values use available epochs after epoch 10 and are timing
descriptors only, not final-50 statistics.  The nearly identical per-epoch costs
confirm that changing mu has negligible computational cost.

## Existing-baseline comparison

| baseline | relationship to this sweep | available outcome | median sec/epoch |
|---|---|---|---:|
| MinSR mu=0, old E200 | same scientific checkpoint/settings; diagnostics protocol differs | stable 200; final E=-109.5312, variance=0.8576 | 0.4748 |
| SPRING mu=0.90, old E200 request | same checkpoint/lr; older NaN-only stop and diagnostics differ | nonfinite at epoch 38 | 0.4732 |
| SPRING mu=0.95, old E200 request | same checkpoint/lr; older NaN-only stop and diagnostics differ | nonfinite at epoch 62 | 0.4764 |
| WSSR rank 400/warm1, E50 | same checkpoint/walkers, different optimizer and only 50 epochs | final E=-109.4911, variance=1.0216 | 0.4038 |
| WSSR rank 800/warm2, E50 | same checkpoint/walkers, different optimizer and only 50 epochs | final E=-109.5057, variance=0.9477 | 0.5199 |
| WSSR rank 1600/warm2, E50 | same checkpoint/walkers, different optimizer and only 50 epochs | final E=-109.5089, variance=0.8429 | 0.7698 |

The new mu=0 run is the only directly matched 200-epoch baseline with the full
diagnostic and controlled-stop protocol.  The WSSR values are short historical
smokes and must not be interpreted as final-energy comparisons.  In cost terms,
MinSR/SPRING is slower than rank-400 WSSR but faster than rank-800 and rank-1600
WSSR on these H100 n=4096 measurements.

## Jobs and commands

```bash
# validation
bash -n experiments/fir_n4096_smoke/scripts/N2_spring_low_mu_array.sh
python -m py_compile experiments/fir_n4096_smoke/monitor_spring_thresholds.py

# execution
sbatch experiments/fir_n4096_smoke/scripts/N2_spring_low_mu_array.sh
# 49616959; tasks 0/1/2/3 = mu 0/0.5/0.7/0.8

sbatch --dependency=afterany:49616959 \
  experiments/fir_n4096_smoke/scripts/N2_collect_spring_low_mu.sh
# allocation 49616960 was cancelled while pending after the array completed;
# the identical collector command was then run locally.

python experiments/fir_n4096_smoke/collect_spring_low_mu.py
```

Slurm final states: task 0 `COMPLETED`; tasks 1, 2, and 3 ended by the requested
threshold monitor (`SIGTERM`, exit 15) and were not restarted.

## Output paths

- Raw runs: `/scratch/dexuan1/runs/fir_n4096_spring_low_mu/N2/`
- Combined summary: `summary.csv`
- Full per-run CSVs: `mu{000,050,070,080}_per_epoch.csv`
- This report: `report.md`
- Plots: `energy.svg`, `variance.svg`, `history_ratio.svg`,
  `history_projection.svg`, `raw_direction.svg`, `constraint_scale.svg`, and
  `runtime_comparison.svg`
