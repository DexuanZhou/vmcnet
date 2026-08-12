# Authoritative C comparison baseline

Status: completed and retained as the default matched C comparison result.

This comparison uses the same validated C KFAC-pre1000 checkpoint for both
optimizers:

`/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1/checkpoints/1000.npz`

Common settings:

- all-electron C atom, electron counts `(4, 2)`;
- 1,000 training walkers;
- 100,000 main-optimization epochs;
- 10 MCMC steps per update;
- mean-centered clipping threshold 5;
- inverse-time learning-rate schedule with decay `1e-4`;
- `reload.new_optimizer_state=True`;
- `reload.reburn=False`;
- matched frozen evaluation with 2,000 walkers, burn-in 5,000,
  20,000 evaluation epochs, and 10 MCMC steps per measurement;
- 40,000,000 raw frozen local-energy samples per optimizer.

Optimizer-specific settings:

- SPRING: learning rate `0.02`, `mu=0.99`, damping `0.001`,
  norm constraint `0.001`;
- KFAC control: learning rate `0.02`, damping `0.001`,
  norm constraint `0.001`.

## Frozen results

| method | energy (Ha) | autocorrelation-corrected SEM (Ha) | raw variance (Ha^2) | IAT | acceptance |
|---|---:|---:|---:|---:|---:|
| SPRING | -37.8449131762 | 0.0000121338 | 0.0045313892 | 1.299629 | 0.481482 |
| KFAC | -37.8440539322 | 0.0000458361 | 0.0652560374 | 1.287821 | 0.481465 |

SPRING is lower by `0.859244 mHa`; the difference is about 18 combined
standard errors. KFAC raw variance is about 14.4 times the SPRING variance.

Source files:

- SPRING:
  `/scratch/dexuan1/runs/C_spring_official_kfac1000_E100000/`
- KFAC:
  `/scratch/dexuan1/runs/C_kfac_lr02_n1000_after_kfac1000_E100000/`

## Required interpretation

Use this pair as the primary future C comparison for SPRING versus KFAC
because the starting checkpoint, walker count, training length, and frozen
evaluation protocol are matched.

Do not merge it numerically with:

- the older 4,096-walker SPRING trajectory after KFAC-pre5000, which stopped
  after 30,288 finite epochs;
- short E500 WSSR ablations, whose training length and frozen-evaluation
  protocol differ;
- any C result with electron counts other than `(4, 2)`.

The exact Hamiltonian/reference-energy correspondence remains under audit.
The frozen SPRING value is close to `-37.8450 Ha`, but it is about
`0.203 mHa` below the older project plotting reference `-37.84471 Ha`.
Therefore this is an authoritative matched optimizer comparison, not yet a
final claim of absolute physical ground-state energy.
