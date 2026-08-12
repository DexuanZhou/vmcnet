# N2 n=4096 SPRING mu=0.5: float32 contractivity versus nonlinear feedback

## Verdict

**Classification D: the previous history decomposition/diagnostics were misleading.**

The experiment rules out nonlinear parameter-sampler feedback as the primary trigger over the observed 50-epoch interval. The matched float32 trajectory runs away and is stopped at epoch 14, while the trajectory whose 4096 x 4096 sample-space matrix construction and eigensolve are performed in float64 completes 50/50 epochs with normal N2 energy, variance, acceptance, unconstrained updates, and no threshold event.

More precisely, float32 does **not** make the ideal history propagation operator non-contractive: for mu=0.5, `||mu M_k||_2 = 0.5` because the regularized projector has spectrum in `[d/(lambda_max+d), 1]`. The failure instead occurs in the numerically evaluated combined sample-space RHS `epsilon_k - p_k`. The sample-space projection `p_k` grows hundreds of times larger than `epsilon_k`; direct float32 subtraction/eigensolve then loses the cancellation between the current-data and history pieces. The separate float32 history application remains close to float64, but their directly evaluated total direction does not.

Sampler deterioration is a downstream feedback amplifier after the wrong direction appears, not the initiating mechanism in this matched test.

## Matched configuration

Both scientific runs load model parameters, all 4096 equilibrated walkers, MCMC state, and PRNG state from:

`/scratch/dexuan1/runs/fir_n4096_corrected_retry1/N2/kfac_preliminary/checkpoints/5000.npz`

They use N2 R=2.068 Bohr (z=+-1.034), `(7,7)` electrons, `nchains=4096`, `nepochs=50`, no reburn, a new SPRING optimizer state, `mu=0.5`, learning rate `0.0005`, damping `0.001`, norm constraint `0.001`, inverse-time decay `1e-4`, and the same sampler, clipping, model, checkpoint state, and seed.

A recursive resolved-config diff found only output paths, diagnostic replay epochs, and `mixed_precision_solve=False/True`. The mixed run disables redundant offline replay snapshots; that does not change its scientific update.

The diagnostic mixed path leaves the SPRING recurrence, model, VJP, parameter dtype, and sampler at production float32. A host callback casts the uncentered sample matrix and RHS inputs to NumPy float64, performs centering, symmetrization, eigendecomposition, and the three linear solves in float64, and returns the resulting directions as float32 to the normal VJP.

## Jobs and commands

| Purpose | Job ID | Result |
|---|---:|---|
| initial diagnostic regression | 49619450 | 10/10 passed |
| float32 matched trajectory (array task 0) | 49619469_0 | threshold stop after 14 finite epochs |
| local-x64 regression | 49619776 | 11/11 passed |
| final callback-based regression | 49620535 | 11/11 passed |
| valid mixed-float64 trajectory | 49620621_1 | completed 50/50 |

The valid submissions were:

```bash
sbatch experiments/fir_n4096_smoke/scripts/N2_spring_mu05_precision_array.sh
sbatch --array=1 --export=ALL,MU05_PRECISION_ROOT=/scratch/dexuan1/runs/fir_n4096_spring_mu05_precision_retry5/N2 experiments/fir_n4096_smoke/scripts/N2_spring_mu05_precision_array.sh
```

Jobs `49619510`, `49619796`, `49619935`, and `49620436` are preserved execution-layer attempts that failed before epoch 1 while localizing x64 to the solve; they contain no scientific trajectory and are excluded from all comparisons.

## Correct history decomposition

The recorded quantities are

`c_k = O_k^T A_k^-1 epsilon_k`,

`h_k = mu (I - O_k^T A_k^-1 O_k) phi_(k-1)`,

`p_k = mu O_k phi_(k-1)`, and `phi_k = c_k + h_k`.

Thus `R_rhs = ||p_k||/||epsilon_k||` measures how demanding the cancellation is in sample space, while `R_true = ||h_k||/||c_k||` measures actual parameter-space history dominance.

The old ratio was `||mu phi_(k-1)|| / ||O^T A^-1(epsilon-p)||`. Its denominator already mixes the current RHS and projected history, so it can stay below one even while `p_k` becomes enormous and the combined solve loses accuracy.

## Float32 trajectory: full corrected ratios

| epoch | ||epsilon|| | ||p|| | R_rhs | ||c|| | ||h|| | R_true | cos(c,h) | ||phi_direct|| | constraint scale | variance | acceptance |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|1|1.420|0|0|14.162|0|0|0|14.176|1.000|2.017|0.4676|
|2|1.389|1.702|1.225|13.841|4.864|0.351|0.085|15.067|1.000|1.931|0.4685|
|3|1.383|2.222|1.607|13.942|5.199|0.373|0.133|15.529|1.000|1.912|0.4643|
|4|1.263|4.869|3.854|12.525|5.525|0.441|0.122|14.356|1.000|1.596|0.4670|
|5|1.238|11.082|8.954|12.696|5.086|0.401|0.132|14.554|1.000|1.532|0.4636|
|6|1.242|22.718|18.298|12.798|5.238|0.409|0.112|15.449|1.000|1.542|0.4648|
|7|1.169|47.796|40.873|11.975|5.054|0.422|0.133|18.058|1.000|1.368|0.4635|
|8|1.305|99.563|76.319|13.102|4.913|0.375|0.122|28.786|1.000|1.702|0.4650|
|9|1.652|207.995|125.877|15.124|5.171|0.342|0.138|54.435|1.000|2.731|0.4673|
|10|2.742|434.768|158.580|22.817|5.975|0.262|0.164|109.759|0.577|7.518|0.4645|
|11|4.182|902.784|215.875|33.498|8.922|0.266|0.208|221.983|0.285|17.493|0.4651|
|12|5.832|1871.842|320.954|45.437|14.098|0.310|0.208|451.190|0.140|34.022|0.4656|
|13|7.514|3941.180|524.521|57.793|23.045|0.399|0.156|935.455|0.0677|56.472|0.4682|
|14|9.460|8796.303|929.862|75.077|41.269|0.550|0.115|2066.663|0.0306|89.510|0.4693|

`R_true` never exceeds one. In contrast, `R_rhs` first exceeds one at epoch 2 and reaches 930 by the stopping epoch. This is why the old parameter-space ratio missed the dangerous growth.

## Selected float32 versus float64 replays

| epoch | c relative error / cosine | h relative error / cosine | total phi relative error / cosine | ||phi32|| | ||phi64|| |
|---:|---:|---:|---:|---:|---:|
|1|0.0727 / 0.9974|0 / 0|0.0727 / 0.9974|14.176|14.070|
|4|0.0789 / 0.9969|0.0369 / 0.9993|0.1064 / 0.9944|14.356|14.270|
|8|0.0777 / 0.9970|0.0389 / 0.9992|1.726 / 0.4955|28.786|14.486|
|12|0.0737 / 0.9973|0.0558 / 0.9984|8.996 / 0.0765|451.190|50.038|

The first decisive error is therefore neither the current-data term nor the projected history term alone. It is the cancellation in the direct total solve. Epoch 15 is unavailable because the required immediate threshold stop occurred after epoch 14. Epoch 14 is the last finite row, but no matrix snapshot was scheduled there; its full decomposition is present in the trajectory table.

## Contractivity estimates

For the symmetric regularized operator, the parameter-space nullspace gives `||M_k||_2=1`, hence `||mu M_k||_2=0.5` in both precisions. It never exceeds one.

| epoch | min eig symmetric(M), float32 | min eig symmetric(M), float64 | max eig | ||M||2 | ||mu M||2 | h32/h64 relative error | h cosine |
|---:|---:|---:|---:|---:|---:|---:|---:|
|1|3.458e-6|3.458e-6|1|1|0.5|0|0|
|4|3.498e-6|3.498e-6|1|1|0.5|0.0369|0.9993|
|8|3.549e-6|3.549e-6|1|1|0.5|0.0389|0.9992|
|12|3.383e-6|3.383e-6|1|1|0.5|0.0558|0.9984|

This rejects the literal hypothesis that float32 makes `mu M_k` expansive. The numerical pathology is loss of accuracy in applying the algebraically equivalent combined RHS formulation when `||p|| >> ||epsilon||`.

## Mixed-float64 outcome

| metric | float32 | mixed float64 |
|---|---:|---:|
|finite epochs|14|50|
|stop reason|raw direction >100x first-10 median|none|
|final energy (Ha)|-114.0615|-109.5096|
|final variance (Ha2)|89.5097|0.9667|
|final acceptance|0.4693|0.4644|
|maximum R_rhs|929.862|1.938|
|maximum R_true|0.550|0.435|
|maximum ||p|| |8796.30|2.003|
|maximum direct ||phi|| |2066.66|15.263|
|minimum constraint scale|0.0306|1.000|

At epoch 13, before the float32 stop, mixed float64 has energy `-109.5617 Ha`, variance `1.2550`, and acceptance `0.4655`; float32 has energy `-112.499 Ha`, variance `56.472`, and acceptance `0.4682`. Acceptance remains normal in both at onset, so a sampler acceptance collapse does not precede the numerical direction failure.

## Interpretation and next experiment

The evidence supports a float32 numerical cancellation/amplification mechanism, followed by nonlinear sampler/local-energy feedback. It does not support loss of the ideal operator's contractivity, and it does not support nonlinear feedback as the primary cause under the matched 50-epoch test.

No subsequent mu sweep was launched. Now that the mixed-float64 mu=0.5 result is known, a later explicitly authorized sweep may test `mu in {0.05, 0.1, 0.2, 0.3, 0.4}`.

## Output paths

- Float32 run: `/scratch/dexuan1/runs/fir_n4096_spring_mu05_precision/N2/N2eq_R2068_spring_mu050_float32_n4096_e50`
- Valid mixed run: `/scratch/dexuan1/runs/fir_n4096_spring_mu05_precision_retry5/N2/N2eq_R2068_spring_mu050_mixed64_n4096_e50`
- Report directory: `/scratch/dexuan1/vmcnet/experiments/fir_n4096_smoke/results/spring_mu05_precision`
- Full trajectories: `float32_per_epoch.csv`, `mixed64_per_epoch.csv`
- Replay table: `selected_replay_contractivity.csv`
- Contractivity spectrum: `float64_contractivity_spectrum.csv`
- Summary: `summary.csv`

No default optimizer behavior was changed, no long run or further mu sweep was launched, and nothing was committed or pushed.
