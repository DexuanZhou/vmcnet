# WSSR Low-Rank SR Implementation Spec for `vmcnet`

## 0. Purpose

This document specifies the intended implementation of two low-rank stochastic reconfiguration optimizers in `vmcnet`.

Implement only:

1. `SVDSolver`: SVD-based low-rank stochastic reconfiguration.
2. `SketchSolver`: sketch-based low-rank stochastic reconfiguration.

Do **not** implement:

- `DirectSolver`
- `MINSRSolver`
- `SPRINGSolver`
- residual-weighted SR
- inverse-residual sample weighting
- any newly invented WSSR formula

The intended algorithms are exactly the Julia `opts!` implementations for:

```julia
opts!(..., OptParams::OPTPARAMSVD, optimizer::SVDSolver, ...)
opts!(..., OptParams::OPTPARAMSKETCH, optimizer::SketchSolver, ...)
```

The first implementation should prioritize correctness and minimal integration into the existing `vmcnet` optimizer pipeline.

Recommended order:

1. Implement `wssr_svd` first.
2. Add `wssr_sketch` only after `wssr_svd` works.
3. If a faithful sketch backend is too large for the first pass, add a clearly documented placeholder for `wssr_sketch` rather than silently changing the algorithm.

---

## 1. Optimizer Names

Preferred config style:

```text
optimizer_type = "wssr_svd"
optimizer_type = "wssr_sketch"
```

Alternative acceptable style:

```text
optimizer_type = "wssr"
wssr.method = "svd" | "sketch"
```

Choose whichever is more consistent with the current `vmcnet` config system.

---

## 2. Mathematical Algorithm

At each optimization step, we are given:

- `o`: centered score/log-derivative matrix from the current batch.
- `Eloc`: centered local-energy residual vector from the current batch.
- `eta`: history mixing parameter, corresponding to Julia `η`.
- `damping`: SR damping parameter.
- `norm_constraint`: norm constraint for the update.
- `sr_o`: stored low-rank history of score directions.
- `ek`: stored projected energy residual history.
- `sr_rank0`: currently active history rank.
- `sr_rank`: current working rank.
- `sr_rank_max`: maximum allowed rank.
- `sr_scale`: rank growth factor.

The Julia implementation uses the convention

```text
o: num_params × num_samples
Eloc: num_samples
```

so that

```text
o * o'    = parameter-space SR/Fisher matrix
o' * o    = sample-space kernel/NTK matrix
```

The low-rank WSSR solvers operate primarily with the parameter-space score matrix.

---

## 3. History-Augmented Matrix and Vector

If there is no previous history, use the current batch only:

```julia
oa = o
ea = Eloc
```

Otherwise, mix the stored low-rank history and the current batch by

```julia
lmul!(sqrt(eta), OptParams.sr_o)
lmul!(sqrt(1 - eta), o)
lmul!(sqrt(eta), OptParams.ek)
lmul!(sqrt(1 - eta), Eloc)

oa = hcat(OptParams.sr_o[:, 1:sr_rank0], o)
ea = vcat(OptParams.ek[1:sr_rank0], Eloc)
```

Mathematically,

\[
O_{\mathrm{aug}}
=
\begin{bmatrix}
\sqrt{\eta} O_{\mathrm{hist}} &
\sqrt{1-\eta} O_{\mathrm{cur}}
\end{bmatrix},
\]

and

\[
E_{\mathrm{aug}}
=
\begin{bmatrix}
\sqrt{\eta} E_{\mathrm{hist}} \\
\sqrt{1-\eta} E_{\mathrm{cur}}
\end{bmatrix}.
\]

Here

```text
O_hist = sr_o[:, 1:sr_rank0]
E_hist = ek[1:sr_rank0]
```

---

## 4. SVD-Based Low-Rank SR Update

Compute a truncated or approximate SVD

\[
O_{\mathrm{aug}}
\approx
U \Sigma V^\top.
\]

Let the singular values be

\[
\sigma_1 \geq \sigma_2 \geq \cdots.
\]

The Julia code chooses the active rank by

```julia
ind = searchsortedlast(S / S[1], damping, rev = true)
```

Equivalently, retain singular values satisfying

\[
\frac{\sigma_j}{\sigma_1} > \mathrm{damping}.
\]

Let

\[
r = \mathrm{ind}.
\]

Define

\[
\sigma_0
=
\frac{1}{(\mathrm{damping}\,|\sigma_1|)^2}.
\]

Then form

\[
g
=
O_{\mathrm{aug}} E_{\mathrm{aug}}.
\]

The update direction is

\[
\Delta\theta
=
U_r
\left[
\left(
\Sigma_r^{-2}
-
\sigma_0 I
\right)
U_r^\top g
\right]
+
\sigma_0 g.
\]

In Julia this is:

```julia
mul!(OptParams.f, oa, ea)
uf = u' * OptParams.f
uf .*= ((1 ./ s).^2 .- Ref(sigma0))

mul!(OptParams.dw_tot, u, uf)
OptParams.dw_tot .+= sigma0 * OptParams.f
```

This is equivalent to applying a regularized low-rank inverse using the retained left singular subspace, with an isotropic fallback controlled by `sigma0`.

---

## 5. Norm Constraint

After computing the parameter update, apply:

```julia
OptParams.dw_tot .*= min(1, sqrt(norm_constrain) / norm(OptParams.dw_tot))
```

Mathematically,

\[
\Delta\theta
\leftarrow
\Delta\theta
\cdot
\min\left(
1,
\frac{\sqrt{\mathrm{norm\_constraint}}}
{\|\Delta\theta\|}
\right).
\]

The implementation should handle the zero-norm case safely.

---

## 6. History Update

After computing the update, overwrite the low-rank history:

```julia
fill!(OptParams.sr_o, zero(eltype(oa)))
@views mul!(OptParams.sr_o[:, 1:ind], u, Diagonal(s))

fill!(OptParams.ek, zero(eltype(oa)))
@views mul!(OptParams.ek[1:ind], v', ea)

optimizer.sr_rank0 = ind
```

Mathematically,

\[
O_{\mathrm{hist,new}}
=
U_r \Sigma_r,
\]

and

\[
E_{\mathrm{hist,new}}
=
V_r^\top E_{\mathrm{aug}}.
\]

Then set

\[
\mathrm{sr\_rank0} = r.
\]

If the active rank reaches the current working rank, grow the working rank:

```julia
if ind == sr_rank
    optimizer.sr_rank = min(Int(ceil(sr_rank * optimizer.sr_scale)), optimizer.sr_rank_max)
end
```

Mathematically,

\[
\mathrm{sr\_rank}
\leftarrow
\min\left(
\lceil \mathrm{sr\_scale}\cdot \mathrm{sr\_rank}\rceil,
\mathrm{sr\_rank\_max}
\right).
\]

---

## 7. Important Shape Convention for `vmcnet`

The Julia code uses explicit flattened parameter vectors.

In `vmcnet`, parameters are PyTrees. Therefore, the implementation must either:

1. flatten the parameter PyTree into a vector, perform the update in vector form, and unflatten it back to the original PyTree structure; or
2. use JAX VJP/JVP utilities to represent the same matrix-vector products without explicitly materializing the full parameter-space matrix.

For the first correct implementation, prefer the simplest approach that matches the Julia algorithm.

Expected shapes:

```text
num_params = flattened parameter dimension
num_samples = batch size across chains/devices
rank <= sr_rank <= sr_rank_max

O_cur:   num_params × num_samples
E_cur:   num_samples
O_hist:  num_params × sr_rank0
E_hist:  sr_rank0

O_aug:   num_params × (sr_rank0 + num_samples)
E_aug:   sr_rank0 + num_samples

U:       num_params × rank
S:       rank
V:       (sr_rank0 + num_samples) × rank

g:       num_params
dw:      num_params
```

After computing `dw`, convert it back to a parameter PyTree with the same structure as `params`.

---

## 8. Relation to Existing `vmcnet` Optimizers

Inspect these files before editing:

```text
vmcnet/updates/spring.py
vmcnet/updates/parse_optimizer_config.py
vmcnet/train/default_config.py
vmcnet/updates/optax_utils.py
vmcnet/physics/core.py
```

Reuse existing infrastructure where appropriate:

- energy/statistics function creation,
- learning-rate schedules,
- optimizer dispatch,
- norm constraint logic,
- update-function construction,
- existing PyTree utilities,
- existing pmap/pmean utilities if required.

Do not change existing behavior of:

```text
sgd
kfac
spring
```

---

## 9. Minimal Config Fields

Add config fields similar to:

```python
"wssr_svd": {
    "schedule_type": "inverse_time",
    "learning_rate": 0.05,
    "learning_decay_rate": 0.0001,
    "damping": 0.001,
    "damping_decay": 1,
    "damping_min": 0.0,
    "constrain_norm": True,
    "norm_constraint": 0.001,
    "eta": 0.99,
    "sr_rank": 10,
    "sr_rank_max": 100,
    "sr_scale": 1.1,
}
```

For `wssr_sketch`, add only the fields required for the sketch/randomized SVD backend.

Do **not** add residual-weighting fields such as:

```text
weight_mode
inverse_abs_residual
inverse_square_residual
max_weight
```

These are not part of the intended algorithm.

---

## 10. Implementation Order

### Step 1: Analysis Only

Before editing files, report:

1. which files need changes,
2. which functions can be reused,
3. how the parameter PyTree will be flattened/unflattened or handled by VJP,
4. how `O_aug`, `E_aug`, `sr_o`, and `ek` will be stored in optimizer state,
5. whether exact SVD can be implemented cleanly in JAX for the first pass,
6. how `sr_rank0`, `sr_rank`, and `sr_rank_max` will be updated.

Do not modify files in this step.

### Step 2: Implement `wssr_svd`

Implement only the exact-SVD version first.

Requirements:

1. Create the WSSR optimizer implementation.
2. Add optimizer dispatch.
3. Add default config.
4. Preserve all existing optimizers.
5. Show the git diff.
6. Run only small smoke tests.

### Step 3: Implement or Stub `wssr_sketch`

Only after `wssr_svd` works, implement the sketch backend.

If a faithful sketch implementation is too large for the first pass, create a clearly marked backend function and either:

1. implement a randomized SVD backend, or
2. raise a clear `NotImplementedError` for `wssr_sketch`.

Do not silently make `wssr_sketch` identical to `wssr_svd` unless explicitly documented as a temporary fallback.

---

## 11. Smoke Tests

Run only small tests, not long VMC training jobs.

Suggested checks:

### Import check

```bash
python -c "import vmcnet"
```

### Config parse check

Confirm that the config accepts:

```text
optimizer_type = "wssr_svd"
```

and eventually:

```text
optimizer_type = "wssr_sketch"
```

### Optimizer state construction

Construct a tiny optimizer state and verify:

```text
sr_o shape is correct
ek shape is correct
sr_rank0 <= sr_rank <= sr_rank_max
```

### Tiny update-shape test

Verify:

```text
returned update PyTree has the same structure as params
returned update contains no NaNs
state updates sr_o and ek
```

### Optional quick VMC smoke test

Use a very small `vmc-molecule` quicktest with few chains and few epochs only.

Do not run long production training.

---

## 12. Critical Constraints

- Do not invent a new WSSR algorithm.
- Do not implement residual-based weights.
- Do not modify existing SGD/KFAC/SPRING behavior.
- Do not implement `DirectSolver`, `MINSRSolver`, or `SPRINGSolver`.
- Implement only `SVDSolver` and `SketchSolver` low-rank SR.
- Start with `wssr_svd`.
- Add `wssr_sketch` only after `wssr_svd` is correct or provide a clear placeholder.
- Show the git diff after editing.
- Run only small smoke tests.

---

# Appendix: Reference Julia Implementation

This appendix contains only the relevant Julia implementation fragments for the intended low-rank SR optimizers.

## A. Solver Types

```julia
abstract type SR_method end

mutable struct SketchSolver <: SR_method
    sr_rank_max::Int64
    sr_rank::Int64
    sr_rank0::Int64
    sr_scale::Float64
end

SketchSolver(sr_rank_max::Int64) =
    SketchSolver(sr_rank_max, min(10, sr_rank_max), min(10, sr_rank_max), 1.1)

mutable struct SVDSolver <: SR_method
    sr_rank_max::Int64
    sr_rank::Int64
    sr_rank0::Int64
    sr_scale::Float64
end

SVDSolver(sr_rank_max::Int64) =
    SVDSolver(sr_rank_max, min(10, sr_rank_max), min(10, sr_rank_max), 1.1)
```

---

## B. Optimizer State for `SketchSolver`

```julia
mutable struct OPTPARAMSKETCH
    sr_o::Array{Float64, 2}
    f::Array{Float64, 1}
    dw_tot::Array{Float64, 1}
    dk::Array{Float64, 1}
    ek::Array{Float64, 1}
    uf::Array{Float64, 1}
end

function init_optim(dim_ps::Int64, nchains, sr::SketchSolver)
    @unpack sr_rank_max = sr
    sr_o = zeros(Float64, dim_ps, sr_rank_max)
    f = zeros(Float64, dim_ps)
    dw_tot = zeros(Float64, dim_ps)
    uf = zeros(Float64, sr_rank_max)
    dk = zeros(Float64, dim_ps)
    ek = zeros(Float64, sr_rank_max)
    return OPTPARAMSKETCH(sr_o, f, dw_tot, dk, ek, uf)
end
```

---

## C. Optimizer State for `SVDSolver`

```julia
mutable struct OPTPARAMSVD
    sr_o::Array{Float64, 2}
    f::Array{Float64, 1}
    dw_tot::Array{Float64, 1}
    dk::Array{Float64, 1}
    ek::Array{Float64, 1}
    uf::Array{Float64, 1}
    u::Array{Float64, 2}
end

function init_optim(dim_ps::Int64, nchains, sr::SVDSolver)
    @unpack sr_rank_max = sr
    sr_o = zeros(Float64, dim_ps, sr_rank_max)
    f = zeros(Float64, dim_ps)
    dw_tot = zeros(Float64, dim_ps)
    dk = zeros(Float64, dim_ps)
    ek = zeros(Float64, sr_rank_max)
    uf = zeros(Float64, sr_rank_max)
    u = randn(Float64, dim_ps, sr_rank_max)
    return OPTPARAMSVD(sr_o, f, dw_tot, dk, ek, uf, u)
end
```

---

## D. `SketchSolver` Update

```julia
using LowRankApprox

function opts!(
    i,
    OptParams::OPTPARAMSKETCH,
    optimizer::SketchSolver,
    Eloc,
    o::Matrix{T},
    nbatch,
    damping::Float64,
    dim_ps::Int64,
    η::Float64,
    norm_constrain,
    γ,
    m,
) where {T}

    sr_rank0 = optimizer.sr_rank0::Int64
    sr_rank = optimizer.sr_rank::Int64
    sketch_opts = LRAOptions(sketch = :srft, rank = min(dim_ps, sr_rank))

    res = norm(OptParams.f)
    lmul!(-γ, Eloc)

    if OptParams.sr_o[1] == 0.0
        oa = o::Matrix{T}
        ea = Eloc::Vector{T}
    else
        lmul!(sqrt(η), OptParams.sr_o)
        lmul!(sqrt(1 - η), o)
        lmul!(sqrt(η), OptParams.ek)
        lmul!(sqrt(1 - η), Eloc)

        oa = hcat(OptParams.sr_o[:, 1:sr_rank0], o)::Matrix{T}
        ea = vcat(OptParams.ek[1:sr_rank0], Eloc)::Vector{T}
    end

    svdo = psvdfact(oa, sketch_opts)

    ind = searchsortedlast(svdo[:S] / svdo[:S][1], damping, rev = true)
    sigma0 = 1 / (damping * abs(svdo[:S][1]))^2

    u = svdo[:U][:, 1:ind]::Matrix{T}
    s = svdo[:S][1:ind]::Vector{T}
    v = svdo[:V][:, 1:ind]::Matrix{T}

    mul!(OptParams.f, oa, ea)
    uf = u' * OptParams.f::Vector{T}
    uf .*= ((1 ./ s).^2 .- Ref(sigma0))

    mul!(OptParams.dw_tot, u, uf)
    OptParams.dw_tot .+= sigma0 * OptParams.f

    fill!(OptParams.sr_o, zero(eltype(oa)))
    @views mul!(OptParams.sr_o[:, 1:ind], u, Diagonal(s))

    fill!(OptParams.ek, zero(eltype(oa)))
    @views mul!(OptParams.ek[1:ind], v', ea)

    optimizer.sr_rank0 = ind

    if ind == sr_rank
        optimizer.sr_rank =
            min(Int(ceil(sr_rank * optimizer.sr_scale)), optimizer.sr_rank_max)
    end

    OptParams.dw_tot .*= min(1, sqrt(norm_constrain) / norm(OptParams.dw_tot))
    return OptParams.dw_tot, ind, res
end
```

---

## E. `SVDSolver` Update

```julia
function opts!(
    i,
    OptParams::OPTPARAMSVD,
    optimizer::SVDSolver,
    Eloc,
    o::Matrix{T},
    nbatch,
    damping::Float64,
    dim_ps::Int64,
    η::Float64,
    norm_constrain,
    γ,
    m,
) where {T}

    sr_rank0 = optimizer.sr_rank0::Int64
    sr_rank = optimizer.sr_rank::Int64

    res = norm(OptParams.f)
    lmul!(-γ, Eloc)

    if OptParams.sr_o[1] == 0.0
        oa = o::Matrix{T}
        ea = Eloc::Vector{T}
    else
        lmul!(sqrt(η), OptParams.sr_o)
        lmul!(sqrt(1 - η), o)
        lmul!(sqrt(η), OptParams.ek)
        lmul!(sqrt(1 - η), Eloc)

        oa = hcat(OptParams.sr_o[:, 1:sr_rank0], o)::Matrix{T}
        ea = vcat(OptParams.ek[1:sr_rank0], Eloc)::Vector{T}
    end

    r = min(min(dim_ps, nbatch), sr_rank)

    if size(OptParams.u, 2) < r
        mm = size(OptParams.u, 1)
        OptParams.u[:, size(OptParams.u, 2)+1:r] .=
            randn(mm, r - size(OptParams.u, 2))
    end

    if i == 1
        svdo = lmsvd(oa, r; maxit = 300, X = OptParams.u[:, 1:r])
    else
        svdo = ssisvd(oa, r; maxit = 50, X = OptParams.u[:, 1:r])
    end

    ind = searchsortedlast(svdo.S / svdo.S[1], damping, rev = true)
    sigma0 = 1 / (damping * abs(svdo.S[1]))^2

    OptParams.u[:, 1:length(svdo.S)] .= svdo.U

    u = svdo.U[:, 1:ind]
    s = svdo.S[1:ind]
    v = svdo.V[:, 1:ind]

    mul!(OptParams.f, oa, ea)
    uf = u' * OptParams.f::Vector{T}
    uf .*= ((1 ./ s).^2 .- Ref(sigma0))

    mul!(OptParams.dw_tot, u, uf)
    OptParams.dw_tot .+= sigma0 * OptParams.f

    fill!(OptParams.sr_o, zero(eltype(oa)))
    @views mul!(OptParams.sr_o[:, 1:ind], u, Diagonal(s))

    fill!(OptParams.ek, zero(eltype(oa)))
    @views mul!(OptParams.ek[1:ind], v', ea)

    optimizer.sr_rank0 = ind

    if ind == sr_rank
        optimizer.sr_rank =
            min(Int(ceil(sr_rank * optimizer.sr_scale)), optimizer.sr_rank_max)
    end

    OptParams.dw_tot .*= min(1, sqrt(norm_constrain) / norm(OptParams.dw_tot))
    return OptParams.dw_tot, ind, res
end
```

---

## F. Training Loop: Construction of `o`, `Eloc`, and `f`

The relevant Julia training loop centers the local energies and score matrix before calling `opts!`.

```julia
Eloc = vcat([@getfrom k _elocs for k in 1:nprocs()]...)
v̄al = median(Eloc)
val_list[i][iter], var_list[i][iter] = mean(Eloc), sqrt(var(Eloc) / length(Eloc))

for k = 1:nprocs()
    sendto(k, v̄al = v̄al)
    sendto(k, val_mean = mean(Eloc))
end

@everywhere ΔE = _elocs .- v̄al
@everywhere _a = clip * Statistics.mean(abs.(ΔE))
ā = mean([@getfrom k _a for k in 1:nprocs()])

for k = 1:nprocs()
    sendto(k, ā = ā)
end

@everywhere begin
    ΔE = clipE.(ΔE, Ref(ā))
    _elocs = v̄al .+ ΔE .- val_mean
    _o_mean = mean(_o_tot, dims = 2)
end

o_mean = sum([@getfrom k _o_mean for k in 1:nprocs()])
ldiv!(nprocs(), o_mean)

for k = 1:nprocs()
    sendto(k, o_mean = o_mean)
end

@everywhere begin
    _o_tot .-= _o_mean
    mul!(_force, _o_tot, _elocs)
end

f = sum([@getfrom k _force for k in 1:nprocs()])
OptParams.f = 2 * f / (nchains * nprocs())

o = hcat([@getfrom k _o_tot for k = 1:nprocs()]...)
Eloc = vcat([@getfrom k _elocs for k in 1:nprocs()]...)

ldiv!(sqrt(nprocs() * nchains), o)
ldiv!(sqrt(nprocs() * nchains), Eloc)

gamma = InverseLR(iter, lr, lr_dc)

OptParams.dw_tot, r, res = opts!(
    iter,
    OptParams,
    optimizer.sr_method,
    Eloc,
    o,
    nchains * nprocs(),
    damping,
    dim_ps,
    eta,
    norm_constrain,
    gamma,
    m,
)
```

Important points:

1. `Eloc` is centered after clipping.
2. `o` is centered by subtracting its mean over samples.
3. Both `o` and `Eloc` are divided by `sqrt(total_batch_size)`.
4. The learning rate `gamma` is applied by scaling `Eloc` before the low-rank solve:

```julia
lmul!(-gamma, Eloc)
```

Therefore, in `vmcnet`, either:

- include the learning-rate factor in the RHS/residual before solving, as in Julia; or
- return an unscaled update direction and let the existing Optax learning-rate wrapper apply the learning rate.

Do not apply the learning rate twice.

---

## G. Approximate SVD Reference

The Julia code uses two approximate SVD routines:

```julia
lmsvd(oa, r; maxit = 300, X = OptParams.u[:, 1:r])
ssisvd(oa, r; maxit = 50, X = OptParams.u[:, 1:r])
```

For a first JAX implementation, using exact or truncated JAX SVD is acceptable for `wssr_svd`.

For `wssr_sketch`, a randomized SVD backend can be used only if it preserves the same mathematical update formula.

Do not silently replace the low-rank SR update with a different optimizer.
