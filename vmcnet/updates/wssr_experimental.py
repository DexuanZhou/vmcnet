"""Default-disabled experimental post-SVD operations for WSSR ablations."""
from typing import Optional, Tuple
import jax.numpy as jnp
from jax import lax
from vmcnet.utils.typing import Array

MODES=("none","cluster","near_tail","adaptive_complement","residual_optimal_complement","capped_near_tail","smooth","force_aware","iterative_complement","native_proximal","cluster_envelope","cluster_envelope_ritz","cluster_envelope_ritz_ef","cluster_envelope_ritz_selected","cluster_envelope_ritz_snr","grassmann_ritz")

def validate_mode(mode:str)->None:
    if mode not in MODES: raise ValueError(f"experimental_mode must be one of {MODES}")


def procrustes_grassmann_average(
    previous_basis: Array,
    current_basis: Array,
    alpha: float,
) -> Tuple[Array, Array]:
    """Procrustes-align and average two equal-width orthonormal bases.

    This removes sign, ordering, and within-cluster gauge changes before a QR
    retraction back to the Grassmann manifold. It is intended only for a small
    top cluster because the overlap costs O(N_param * k**2).
    """
    if previous_basis.shape != current_basis.shape:
        raise ValueError("Grassmann bases must have the same shape")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Grassmann alpha must be in [0, 1]")

    # For M = U_prev.T U_cur = L diag(s) R.T, U_cur (R L.T) is the
    # representative aligned to U_prev.
    overlap = previous_basis.T @ current_basis
    left, overlap_singular_values, right_h = jnp.linalg.svd(
        overlap, full_matrices=False
    )
    aligned_current = current_basis @ (right_h.T @ left.T)
    alpha_array = jnp.asarray(alpha, dtype=current_basis.dtype)
    blended = (
        (1.0 - alpha_array) * previous_basis
        + alpha_array * aligned_current
    )
    averaged_basis, _ = jnp.linalg.qr(blended, mode="reduced")
    return averaged_basis, overlap_singular_values

def cluster_effective_rank(s:Array,target:int,threshold:float,max_rank:int)->Array:
    """First r>=target with gap(s[r-1],s[r])>threshold, else max_rank."""
    n=min(max_rank,s.shape[0]); gaps=jnp.abs(s[:-1]-s[1:])/jnp.maximum(jnp.abs(s[:-1]),1e-30)
    candidate=(jnp.arange(gaps.shape[0])>=target-1)&(jnp.arange(gaps.shape[0])<n-1)&(gaps>threshold)
    first=jnp.argmax(candidate)+1
    return jnp.where(jnp.any(candidate),first,jnp.asarray(n))

def cosine_taper(n:int,start:int,end:int,dtype)->Array:
    """Continuous nonnegative one-based mode taper."""
    i=jnp.arange(n,dtype=dtype)+1
    phase=jnp.pi*(i-start)/max(end-start,1)
    return jnp.where(i<start,1.,jnp.where(i>=end,0.,.5*(1+jnp.cos(phase))))

def adaptive_complement(
    resolved: Array,
    perp_force: Array,
    alpha_nominal: Array,
    beta: float,
    function_operator: Optional[Array] = None,
    beta_function: float = 0.0,
) -> Tuple[Array, Array]:
    """Add a complement bounded relative to the retained update.

    ``beta`` is the legacy Euclidean relative cap.  When ``beta_function`` is
    positive, the same candidate is also capped in the current-batch Fisher
    metric using ``||O_bar.T @ p||``.  A zero function-space beta preserves the
    historical implementation exactly.
    """
    eps = jnp.asarray(1e-30, dtype=resolved.dtype)
    euclidean_cap = (
        jnp.asarray(beta, dtype=resolved.dtype)
        * jnp.linalg.norm(resolved)
        / jnp.maximum(jnp.linalg.norm(perp_force), eps)
    )
    alpha = jnp.minimum(alpha_nominal, euclidean_cap)
    if function_operator is not None:
        retained_action = function_operator.T @ resolved
        complement_action = function_operator.T @ perp_force
        function_cap = (
            jnp.asarray(beta_function, dtype=resolved.dtype)
            * jnp.linalg.norm(retained_action)
            / jnp.maximum(jnp.linalg.norm(complement_action), eps)
        )
        alpha = jnp.where(
            jnp.asarray(beta_function) > 0.0,
            jnp.minimum(alpha, function_cap),
            alpha,
        )
    return alpha * perp_force, alpha

def residual_optimal_complement(
    operator: Array, force: Array, resolved: Array, perp_force: Array,
    regularizer: Array, beta: float, eps: float = 1e-30,
):
    """Minimize the regularized residual along one complement direction.

    This uses matrix-free ``A A.T`` products and then applies the requested
    nonnegative norm-ratio cap.  It never forms a parameter-space matrix.
    """
    mat = lambda x: operator @ (operator.T @ x) + regularizer * x
    residual_before = force - mat(resolved)
    action = mat(perp_force)
    alpha_star = jnp.maximum(
        0.0, jnp.vdot(action, residual_before)
        / jnp.maximum(jnp.vdot(action, action), eps)
    )
    alpha_cap = beta * jnp.linalg.norm(resolved) / jnp.maximum(
        jnp.linalg.norm(perp_force), eps
    )
    alpha = jnp.minimum(alpha_star, alpha_cap)
    complement = alpha * perp_force
    residual_after = force - mat(resolved + complement)
    return complement, alpha, alpha_star, alpha_cap, residual_before, residual_after

def cap_contribution(resolved:Array, contribution:Array,beta:float,eps:float=1e-30):
    """Scale a contribution to at most beta times the resolved norm."""
    scale=jnp.minimum(1.0,beta*jnp.linalg.norm(resolved)/jnp.maximum(jnp.linalg.norm(contribution),eps))
    return scale*contribution,scale


def project_onto_basis_envelope(
    basis: Array,
    vector: Array,
    relative_eigenvalue_cutoff: float = 1e-6,
):
    """Project ``vector`` onto the column envelope of a small basis block.

    ``basis`` need not be orthogonal and may contain repeated or zero columns.
    The projector is evaluated through the small Gram matrix, so this routine
    never materializes a parameter-space projector or performs a tall QR.  It
    is invariant to rotations inside any constituent spectral cluster.
    """
    gram = basis.T @ basis
    gram = 0.5 * (gram + gram.T)
    eigenvalues, rotation = jnp.linalg.eigh(gram)
    leading = jnp.maximum(jnp.max(eigenvalues), 0.0)
    cutoff = relative_eigenvalue_cutoff * jnp.maximum(leading, 1e-30)
    retained = eigenvalues > cutoff
    coordinates = rotation.T @ (basis.T @ vector)
    inverse = jnp.where(retained, 1.0 / jnp.maximum(eigenvalues, 1e-30), 0.0)
    scaled_coordinates = (
        inverse * coordinates
        if vector.ndim == 1
        else inverse[:, None] * coordinates
    )
    projection = basis @ (rotation @ scaled_coordinates)
    return projection, jnp.sum(retained.astype(jnp.int32)), eigenvalues


def orthonormalize_basis_envelope(
    basis: Array,
    relative_eigenvalue_cutoff: float = 1e-6,
):
    """Return a padded orthonormal basis for the envelope column span.

    A small Gram eigendecomposition avoids a tall parameter-space QR. Columns
    below the relative cutoff are represented by exact zeros, so the returned
    array keeps a static width suitable for JIT-compiled Galerkin solves.
    """
    gram = basis.T @ basis
    gram = 0.5 * (gram + gram.T)
    eigenvalues, rotation = jnp.linalg.eigh(gram)
    leading = jnp.maximum(jnp.max(eigenvalues), 0.0)
    cutoff = relative_eigenvalue_cutoff * jnp.maximum(leading, 1e-30)
    retained = eigenvalues > cutoff
    inverse_sqrt = jnp.where(
        retained,
        1.0 / jnp.sqrt(jnp.maximum(eigenvalues, 1e-30)),
        0.0,
    )
    orthonormal_basis = basis @ (rotation * inverse_sqrt[None, :])
    return (
        orthonormal_basis,
        jnp.sum(retained.astype(jnp.int32)),
        eigenvalues,
    )


def select_history_by_current_value(
    current_basis: Array,
    history_basis: Array,
    current_operator: Array,
    current_gradient: Array,
    slots: int,
    regularizer: Array,
    relative_eigenvalue_cutoff: float = 1e-6,
):
    """Select a fixed budget of useful historical novelty directions.

    Current directions are protected. Historical columns are first projected
    into their orthogonal complement and converted to an orthonormal envelope.
    The retained candidates are ranked by their current-batch predicted
    quadratic decrease ``(q.T g)^2 / (q.T S_current q + lambda)``.  Therefore
    old directions can occupy capacity only when they remain useful for the
    current equation; no parameter-space Fisher matrix is materialized.
    """
    if slots < 0 or slots > history_basis.shape[1]:
        raise ValueError("slots must be in [0, history width]")
    if slots == 0:
        empty = history_basis[:, :0]
        zero = jnp.asarray(0, dtype=jnp.int32)
        return (
            empty,
            zero,
            zero,
            jnp.zeros((0,), dtype=history_basis.dtype),
            jnp.asarray(0.0, dtype=history_basis.dtype),
        )

    history_novel = history_basis - current_basis @ (
        current_basis.T @ history_basis
    )
    history_modes, pool_rank, gram_eigenvalues = orthonormalize_basis_envelope(
        history_novel, relative_eigenvalue_cutoff
    )
    leading = jnp.maximum(jnp.max(gram_eigenvalues), 0.0)
    cutoff = relative_eigenvalue_cutoff * jnp.maximum(leading, 1e-30)
    valid = gram_eigenvalues > cutoff

    action = current_operator.T @ history_modes
    curvature = jnp.sum(jnp.square(action), axis=0)
    force = history_modes.T @ current_gradient
    predicted_decrease = jnp.square(force) / jnp.maximum(
        curvature + regularizer, 1e-30
    )
    scores = jnp.where(valid, predicted_decrease, -jnp.inf)
    selected_scores, selected_indices = lax.top_k(scores, slots)
    selected_valid = jnp.isfinite(selected_scores)
    selected = history_modes[:, selected_indices] * selected_valid[None, :]
    total_score = jnp.sum(jnp.where(valid, predicted_decrease, 0.0))
    selection_gain = jnp.sum(
        jnp.where(selected_valid, selected_scores, 0.0)
    ) / jnp.maximum(total_score, 1e-30)
    return (
        selected,
        pool_rank,
        jnp.sum(selected_valid.astype(jnp.int32)),
        selected_scores,
        selection_gain,
    )


def cluster_cross_batch_snr_weights(
    eigenvalues: Array,
    force_a: Array,
    force_b: Array,
    gap_threshold: float,
    regularizer: Array,
    eps: float = 1e-30,
):
    """Estimate rotation-invariant shrinkage weights for spectral clusters.

    Eigenvalues are ascending, as returned by ``eigh``. Adjacent modes are
    separated only at a relative eigengap larger than ``gap_threshold``.
    Within each resulting near-degenerate cluster, the two half-batch force
    estimates share one signal-to-noise weight. The cluster sums make the
    weight invariant to any common orthogonal rotation inside that cluster.
    """
    if eigenvalues.ndim != 1:
        raise ValueError("eigenvalues must be a vector")
    if force_a.shape != eigenvalues.shape or force_b.shape != eigenvalues.shape:
        raise ValueError("force estimates must match the eigenvalue vector")
    if gap_threshold < 0.0:
        raise ValueError("gap_threshold must be nonnegative")

    safe_scale = jnp.maximum(eigenvalues[1:] + regularizer, eps)
    relative_gaps = (eigenvalues[1:] - eigenvalues[:-1]) / safe_scale
    boundaries = relative_gaps > gap_threshold
    cluster_ids = jnp.concatenate(
        [jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(boundaries.astype(jnp.int32))]
    )
    n = eigenvalues.shape[0]
    membership = (
        cluster_ids[:, None] == jnp.arange(n, dtype=jnp.int32)[None, :]
    ).astype(eigenvalues.dtype)

    cross_by_cluster = membership.T @ (force_a * force_b)
    noise_by_cluster = membership.T @ (
        0.5 * jnp.square(force_a - force_b)
    )
    stable_signal = jnp.maximum(cross_by_cluster, 0.0)
    weight_by_cluster = stable_signal / jnp.maximum(
        stable_signal + noise_by_cluster, eps
    )
    weights = membership @ weight_by_cluster
    cluster_count = 1 + jnp.sum(boundaries.astype(jnp.int32))
    return weights, cluster_ids, cluster_count, cross_by_cluster, noise_by_cluster

def linear_budget(initial:float,final:float,step:Array,steps:int)->Array:
    """Inclusive linear schedule from initial at step 0 to final at steps-1."""
    fraction=jnp.clip(step,0,steps-1)/max(steps-1,1)
    return initial+fraction*(final-initial)

def native_factor_proximal_update(
    current_update:Array,u:Array,singular_values:Array,previous_update:Array,
    gamma:float,regularizer:Array,relative_cutoff:Array,eps:float=1e-12,
)->Array:
    """Apply a temporal proximal term inside the current native SSI factors.

    This is invariant to rotations within an exactly degenerate retained block
    because one common ``gamma`` is used.  The current data term keeps its
    existing ``s**2 + regularizer`` weight, so weak modes inherit more history.
    """
    leading=jnp.maximum(jnp.abs(singular_values[0]),eps)
    retained=singular_values/leading>relative_cutoff
    current_left=u.T@current_update
    previous_left=u.T@previous_update
    base=jnp.square(singular_values)+regularizer
    gamma_array=jnp.asarray(gamma,dtype=current_update.dtype)
    coefficients=(
        base*current_left+gamma_array*previous_left
    )/jnp.maximum(base+gamma_array,eps)
    return u@(coefficients*retained.astype(current_update.dtype))

def force_aware_rayleigh_ritz(a:Array,u:Array,vh:Array,force:Array,krylov_vectors:int,target_rank:int):
    """Inject A.T force and optionally one A.T A Krylov vector, then RR-compress."""
    if krylov_vectors not in (1,2): raise ValueError("krylov_vectors must be 1 or 2")
    v=vh.T;v1=a.T@force;v1=v1-v@(v.T@v1);n1=jnp.linalg.norm(v1);extras=[(v1/jnp.maximum(n1,1e-30))[:,None]]
    n2=jnp.asarray(jnp.nan,dtype=a.dtype)
    if krylov_vectors==2:
        v2=a.T@(a@extras[0][:,0]);q=jnp.concatenate([v]+extras,axis=1);v2=v2-q@(q.T@v2);n2=jnp.linalg.norm(v2);extras.append((v2/jnp.maximum(n2,1e-30))[:,None])
    q,_=jnp.linalg.qr(jnp.concatenate([v]+extras,axis=1),mode="reduced");y=a@q;g=y.T@y;ev,rot=jnp.linalg.eigh(g);idx=jnp.argsort(ev)[::-1][:target_rank];vr=q@rot[:,idx];yy=a@vr;s=jnp.sqrt(jnp.maximum(ev[idx],0));ur=yy/jnp.maximum(s,1e-30)
    return ur,s,vr.T,n1,n2

def projected_cg(a:Array,u:Array,rhs:Array,regularizer:Array,niter:int):
    """CG for P(A A.T+lambda I)P with projection at every operation."""
    proj=lambda x:x-u@(u.T@x)
    mat=lambda x:proj(a@(a.T@proj(x))+regularizer*proj(x))
    x=jnp.zeros_like(rhs);r=proj(rhs);p=r;rr=jnp.vdot(r,r)
    for _ in range(niter):
        ap=mat(p);den=jnp.vdot(p,ap);alpha=rr/jnp.maximum(den,1e-30);x=proj(x+alpha*p);rn=proj(r-alpha*ap);rrn=jnp.vdot(rn,rn);p=proj(rn+(rrn/jnp.maximum(rr,1e-30))*p);r,rr=rn,rrn
    return x,jnp.sqrt(rr)/jnp.maximum(jnp.linalg.norm(rhs),1e-30)
