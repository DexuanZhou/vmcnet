# Strict multilevel FermiNet design

## Objective

Construct model levels with a genuine inclusion relation

\[
\mathcal F_0 \subset \mathcal F_1 \subset \cdots \subset \mathcal F_L,
\]

and a prolongation `P` such that, for every electron configuration `R`,

\[
\log |\Psi_{l+1}(P\theta_l;R)| = \log |\Psi_l(\theta_l;R)|
\]

with the same sign and position derivatives.  This is an architecture
continuation method, not a change in walker count or optimizer sketch rank.

## Supported first version

`vmcnet.models.multilevel.prolongate_ferminet_params` supports:

- widening the one-electron stream;
- widening the two-electron stream;
- appending residual blocks which are identities on the old coordinates;
- retaining randomly initialized fine-only hidden features behind an exactly
  zero orbital coupling, so the new features can enter training after the
  transition;
- a separate linear zero-extension for optimizer tangents.

The determinant count is currently required to remain fixed.  An exactly zero
new determinant is a singular input to `slogdet`; a safe determinant hierarchy
needs an explicit determinant coefficient/gate before it should be trained.

## Why generic prefix padding is wrong

The mixed one-electron terms concatenate spin means.  If a width changes from
`m` to `M`, the source coordinates

```
[spin_0: 0:m, spin_1: 0:m]
```

map to

```
[spin_0: 0:m, spin_1: M:M+m]
```

in the fine kernel.  They are not the first `2*m` target rows.  The
prolongation therefore maps `_mixed_dense` and `_dense_2e` one spin block at a
time.

## Necessary skip-connection constraint

For every common block, widening must not change whether the residual skip is
active.  A source layer `tanh(Wx+b) + x` cannot in general be represented by a
target layer `tanh(W' x+b')` with no skip.  Appended blocks must have equal
input/output widths, an enabled skip, and skip scale one, so a zero old-output
residual is an exact identity.

## Recommended initial hierarchy

Keep 16 determinants at all levels and first test the backflow hierarchy:

| level | one/two-electron widths by block | determinants |
|---|---|---:|
| L0 | `(64,8), (64,)` | 16 |
| L1 | `(128,16), (128,16), (128,)` | 16 |
| L2 | `(256,16), (256,16), (256,16), (256,)` | 16 |

Widths 64/128/256 and 8/16 avoid accidentally matching the raw input width and
changing the first-block skip behavior.  The exact condition is validated from
the initialized parameter shapes, not assumed from this table.

## Training transition

The cleanest integration is a checkpoint conversion step between ordinary VMC
runs:

1. Train level `L` and save a normal checkpoint.
2. Initialize an `L+1` parameter template with a recorded PRNG seed.
3. Prolong the checkpoint parameters into that template.
4. Convert or reset the optimizer state according to the policy below.
5. Reuse the MCMC positions and cached amplitudes.  Since the wavefunction is
   unchanged, `reburn=False` is valid.
6. Save a normal, fine-shaped checkpoint and continue with the existing runner.

This avoids adding architecture switches inside a compiled training loop.

## SPRING state

SPRING stores its historical parameter direction in the Optax trace PyTree.
The history can be preserved by the tangent prolongation

\[
\phi_{l+1}=DP\,\phi_l,
\]

which copies all old coordinates and sets all fine-only coordinates to zero.
The Optax schedule count must be copied, otherwise the inverse-time learning
rate restarts at each level.

A first scientific pilot may reset `phi` to zero at each transition, but the
schedule count must still continue.  A final optimizer comparison should use
history prolongation because history is a defining part of SPRING.

## WSSR state

For WSSR, `sr_o` and (when stored) `u` have shape
`(number_of_parameters, storage_rank)`.  Apply the same tangent embedding to
every column:

\[
O^{\mathrm{hist}}_{l+1}=DP\,O^{\mathrm{hist}}_l,\qquad
U_{l+1}=DP\,U_l.
\]

Copy `ek`, active-rank scalars, and the Optax schedule count.  Full-parameter
memories such as `solution_state`, `error_feedback_state`, transported
gradients, and complement EMAs use the same zero-extension.  A stored
`previous_theta` is replaced by the actual flattened prolonged fine
parameters, not by a zero-extended coarse vector.

This is mathematically consistent: the old score space is embedded isometrically
in the fine tangent space, while the next current batch supplies information in
the newly opened coordinates.

## Required gates before a GPU training pilot

For fixed checkpoint configurations, verify:

1. identical wavefunction sign and log amplitude;
2. identical first and second electron-position derivatives (hence identical
   local energy up to numerical tolerance);
3. nonzero gradient on the zero orbital couplings from new hidden features;
4. unchanged MCMC acceptance probabilities;
5. correct Optax step count and learning rate after transition;
6. for history-preserving runs, equality between the old stored history and
   the old-coordinate block of its prolongation at the transition point.

The *next* optimizer update need not have an identical old-coordinate block:
the fine model deliberately exposes new score directions immediately through
the zero orbital couplings, so the fine SR problem is larger even though its
starting wavefunction is identical.

Only after these gates pass should SPRING and WSSR be compared on a staged
`L0 -> L1 -> L2` run.

## Dormant-sidecar scale

Exact equality of the converted wavefunction does not constrain the scale of
the newly exposed score directions.  The fine-only hidden features are
multiplied by `sidecar_scale` before their zero orbital coupling is installed.
The zero coupling keeps the prolongation exactly function preserving for any
scale, while a scale below one reduces how abruptly those new directions enter
the first fine-level optimizer update.

In the initial C pilot, an L1 -> L2 scale of 0.1 still produced a NaN, whereas
0.01 was stable and reduced the independent frozen variance.  Consequently,
`sidecar_scale=0.01` is the current safe pilot setting for large capacity
jumps.  This is an initialization gate, not a permanent restriction: the
fine-only hidden parameters and their orbital couplings remain trainable.
