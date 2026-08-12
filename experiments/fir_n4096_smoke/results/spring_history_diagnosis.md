# SPRING history-state diagnosis

## Classification

**B. Implementation matches the intended SPRING recurrence, suggesting algorithmic instability.**

The deterministic float64 tests show that the current implementation stores the
unscaled, unconstrained Kaczmarz iterate (in an opposite-sign representation),
applies `mu` exactly once, and applies learning-rate scaling and the norm
constraint only to the actual parameter displacement. This is the recurrence in
Algorithm 1 of Goldshlager, Abrahamsen, and Lin (2024).

## Recurrences

With centered, sample-normalized log-derivative matrix `O_k`, imaginary-time
right-hand side `epsilon_k = -(E_L-E_mean)/sqrt(N)`, damping `lambda`, and
unscaled history `phi_{k-1}`, the paper defines

```
phi_k = O_k^T (O_k O_k^T + lambda I + P)^-1
        (epsilon_k - mu O_k phi_{k-1}) + mu phi_{k-1}
dtheta_k = phi_k * min(eta_k, sqrt(C) / ||phi_k||)
theta_{k+1} = theta_k + dtheta_k.
```

VMCNet uses `q_k = -phi_k` because it supplies positive centered local energies
to `spring_step` and lets Optax apply its conventional negative learning-rate
sign. Its implemented recurrence is

```
q_k = O_k^T (O_k O_k^T + lambda I + P)^-1
      ((E_L-E_mean)/sqrt(N) - mu O_k q_{k-1}) + mu q_{k-1}
trace_k = q_k + 0 * trace_{k-1}
u_k = -eta_k q_k
dtheta_k = u_k * min(1, sqrt(C) / ||u_k||)
         = phi_k * min(eta_k, sqrt(C) / ||phi_k||).
```

Thus `trace_k` is the unscaled/unconstrained history, not the actual parameter
displacement. That is intentional and matches the paper.

## Conclusions

- `spring_step` applies `mu` once.
- `optax.sgd(momentum=0)` creates a trace state with decay exactly zero; it does
  not apply `mu` or any additional recurrence.
- The code-side history and raw direction have the opposite sign from the
  paper's `phi`, but Optax's negative learning-rate transform makes the actual
  displacement sign correct.
- The learning rate does not alter the internal history, as required by the
  paper.
- The norm constraint changes the actual displacement but deliberately does not
  alter the stored history. This also matches Algorithm 1.
- Saving and restoring the Optax state preserves the next update exactly.
- The stable `mu=0` and unstable `mu>0` N2 trajectories therefore do not, by
  themselves, demonstrate a history-state implementation bug.

## Non-invasive diagnostics

`config.vmc.optimizer.spring.diagnostics` is disabled by default. When enabled,
the implementation adds norms, maximum absolute entries, cosine similarities,
constraint scale, and the requested state/update ratios to training metrics.
The default mathematical update path is unchanged.

## Deterministic test result

The final synthetic float64 suite ran as Slurm job `49606791`:

```
8 passed, 73 warnings in 72.39s
```

The warnings are pre-existing `jax.tree_map` deprecation warnings. Assertions
used `rtol=atol=2e-11`, explicit norm comparisons, and cosine similarity above
`1-2e-12`. The tests cover MinSR at `mu=0`, two- and three-step history,
learning-rate independence, Optax versus a manual transform, sign reversal,
norm constraint semantics, diagnostics transparency, and optimizer-state
serialization/restoration.
