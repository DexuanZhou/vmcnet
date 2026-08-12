import jax
import jax.numpy as jnp
import numpy as np
import pytest
from vmcnet.updates import wssr_experimental as x
from vmcnet.updates import wssr

def test_cluster_rank_and_taper():
    s=jnp.array([10.,9.,8.,7.99,7.98,6.])
    assert x.cluster_effective_rank(s,3,.01,6)==5
    w=x.cosine_taper(8,3,7,jnp.float64)
    assert jnp.all(w>=0);assert jnp.all(jnp.diff(w)<=0);assert w[0]==1;assert w[-1]==0

def test_adaptive_complement_enforces_ratio():
    r=jnp.array([3.,4.]);p=jnp.array([0.,10.]);c,a=x.adaptive_complement(r,p,jnp.array(100.),.1)
    assert jnp.linalg.norm(c)<=.1*jnp.linalg.norm(r)+1e-12

def test_adaptive_complement_function_cap_can_be_tighter():
    resolved=jnp.array([1.,0.]);perp=jnp.array([0.,2.])
    current_score=jnp.diag(jnp.array([1.,10.]))
    complement,alpha=x.adaptive_complement(
        resolved,perp,jnp.array(1.),.5,
        function_operator=current_score,beta_function=.1)
    # The Euclidean cap gives alpha=.25, while the function cap gives .005.
    np.testing.assert_allclose(alpha,.005,rtol=1e-6)
    np.testing.assert_allclose(complement,jnp.array([0.,.01]),rtol=1e-6)

def test_zero_function_beta_preserves_legacy_adaptive_complement():
    resolved=jnp.array([1.,0.]);perp=jnp.array([0.,2.])
    current_score=jnp.diag(jnp.array([1.,10.]))
    legacy,legacy_alpha=x.adaptive_complement(
        resolved,perp,jnp.array(1.),.5)
    candidate,candidate_alpha=x.adaptive_complement(
        resolved,perp,jnp.array(1.),.5,
        function_operator=current_score,beta_function=0.)
    np.testing.assert_allclose(candidate,legacy)
    np.testing.assert_allclose(candidate_alpha,legacy_alpha)

def test_residual_optimal_complement_minimizes_line_and_caps():
    a=jnp.diag(jnp.array([3.,2.,1.],dtype=jnp.float64))
    force=jnp.array([3.,2.,1.],dtype=jnp.float64)
    resolved=jnp.array([.2,.1,0.],dtype=jnp.float64)
    perp=jnp.array([0.,0.,1.],dtype=jnp.float64)
    comp,alpha,star,cap,rb,ra=x.residual_optimal_complement(
        a,force,resolved,perp,jnp.array(.1),.2)
    assert alpha>=0 and alpha<=cap+1e-14
    assert jnp.linalg.norm(comp)<=.2*jnp.linalg.norm(resolved)+1e-14
    assert jnp.linalg.norm(ra)<=jnp.linalg.norm(rb)+1e-14

def test_cap_contribution_exact_ratio_bound():
    r=jnp.array([3.,4.],dtype=jnp.float64);n=jnp.array([0.,20.],dtype=jnp.float64)
    got,scale=x.cap_contribution(r,n,.2)
    assert jnp.allclose(jnp.linalg.norm(got),.2*jnp.linalg.norm(r))
    assert scale<1

def test_linear_budget_has_inclusive_endpoints():
    assert x.linear_budget(.2,.05,jnp.array(0),500)==.2
    assert jnp.allclose(x.linear_budget(.2,.05,jnp.array(499),500),.05)

def test_native_factor_proximal_matches_modewise_formula():
    u=jnp.eye(3,dtype=jnp.float64)[:,:2]
    s=jnp.array([1.,2.],dtype=jnp.float64)
    current=jnp.array([2.,4.,7.],dtype=jnp.float64)
    previous=jnp.array([6.,8.,9.],dtype=jnp.float64)
    got=x.native_factor_proximal_update(
        current,u,s,previous,gamma=1.,regularizer=jnp.array(1.),
        relative_cutoff=jnp.array(0.))
    expected=jnp.array([10./3.,28./6.,0.],dtype=jnp.float64)
    assert jnp.allclose(got,expected)

def test_native_proximal_mode_is_registered():
    x.validate_mode("native_proximal")

def test_force_exact_first_overrides_existing_warm_basis_once():
    a=jnp.arange(80.,dtype=jnp.float64).reshape(10,8)/40
    state=wssr.initialize_wssr_warm_svd_core_state(10,4,4,jnp.float64,store_warm_u=False)
    state=state._replace(sr_o=jnp.eye(10,4,dtype=jnp.float64),sr_rank0=jnp.array(4))
    u,s,vh,_=wssr._wssr_right_svd_decomposition(a,state,jax.random.PRNGKey(1),4,8,2,True,1e-12,exact_first_force=True)
    ue,se,vhe=jnp.linalg.svd(a,full_matrices=False)
    got=u@(u.T@a);ref=ue[:,:4]@(ue[:,:4].T@a)
    assert jnp.linalg.norm(got-ref)/jnp.linalg.norm(ref)<1e-12

@pytest.mark.parametrize("k",[1,2])
def test_force_aware_shapes_and_finite(k):
    a=jnp.arange(48.,dtype=jnp.float64).reshape(8,6)/50
    u,s,vh=jnp.linalg.svd(a,full_matrices=False);f=jnp.arange(8.,dtype=jnp.float64)
    ur,sr,vhr,n1,n2=x.force_aware_rayleigh_ritz(a,u[:,:3],vh[:3],f,k,3)
    assert ur.shape==(8,3);assert vhr.shape==(3,6);assert jnp.all(jnp.isfinite(ur));assert n1>=0

def test_projected_cg_stays_orthogonal_and_reduces_residual():
    a=jnp.diag(jnp.array([4.,3.,2.,1.],dtype=jnp.float64));u=jnp.eye(4,dtype=jnp.float64)[:,:2];b=jnp.array([2.,3.,4.,5.]);x3,r3=x.projected_cg(a,u,b,.1,3);x5,r5=x.projected_cg(a,u,b,.1,5)
    assert jnp.linalg.norm(u.T@x3)<1e-12;assert r5<=r3+1e-12

def test_disabled_mode_is_exactly_baseline_update_and_state():
    a=jnp.arange(48.,dtype=jnp.float64).reshape(8,6)/30
    e=jnp.linspace(-1.,1.,6);u,s,vh=jnp.linalg.svd(a,full_matrices=False)
    state=wssr.initialize_wssr_core_state(8,4,6,jnp.float64)
    kw=dict(damping=.01,norm_constraint=.001,sr_rank_max=6,sr_scale=1.1,
            constrain_update_norm=False,rank_update_max=6,
            spectral_regularization='tikhonov',complement_weight=.2)
    old=wssr._wssr_update_from_svd(a,e,state,u,s,vh,**kw)
    new=wssr._wssr_update_from_svd(a,e,state,u,s,vh,
            experimental_mode='none',experimental_target_rank=3,
            cluster_gap_threshold=.5,near_tail_modes=2,
            adaptive_complement_beta=.1,smooth_transition_start=2,
            smooth_transition_end=5,iterative_complement_iterations=3,**kw)
    assert jnp.array_equal(old.grad_like_update,new.grad_like_update)
    for xold,xnew in zip(jax.tree_util.tree_leaves(old.state),jax.tree_util.tree_leaves(new.state)):
        assert jnp.array_equal(xold,xnew)

@pytest.mark.parametrize('mode', ['cluster','near_tail','adaptive_complement','residual_optimal_complement','capped_near_tail','smooth','iterative_complement'])
def test_integrated_experimental_modes_are_finite(mode):
    a=jnp.arange(80.,dtype=jnp.float64).reshape(10,8)/40;e=jnp.linspace(-1.,1.,8)
    u,s,vh=jnp.linalg.svd(a,full_matrices=False);state=wssr.initialize_wssr_core_state(10,4,8,jnp.float64)
    got=wssr._wssr_update_from_svd(a,e,state,u,s,vh,damping=.01,
      norm_constraint=.001,sr_rank_max=8,sr_scale=1.1,constrain_update_norm=False,
      rank_update_max=8,spectral_regularization='tikhonov',complement_weight=1e-4,
      experimental_mode=mode,experimental_target_rank=4,near_tail_modes=2,
      cluster_gap_threshold=.01,adaptive_complement_beta=.1,
      smooth_transition_start=3,smooth_transition_end=7,
      iterative_complement_iterations=3)
    assert jnp.all(jnp.isfinite(got.grad_like_update))


@pytest.mark.parametrize("krylov_vectors", [1, 2])
def test_integrated_force_aware_mode_is_finite(krylov_vectors):
    a = jnp.arange(80., dtype=jnp.float64).reshape(10, 8) / 40
    e = jnp.linspace(-1., 1., 8)
    state = wssr.initialize_wssr_warm_svd_core_state(
        10, 4, 4, jnp.float64, store_warm_u=True
    )
    got = wssr.wssr_warm_svd_right_core_update(
        a, e, state, jax.random.PRNGKey(3), damping=.01,
        norm_constraint=.001, sr_rank_max=4, svd_working_rank=4,
        constrain_update_norm=False, spectral_regularization='tikhonov',
        complement_weight=0., experimental_mode='force_aware',
        experimental_target_rank=4,
        force_aware_krylov_vectors=krylov_vectors,
    )
    assert jnp.all(jnp.isfinite(got.grad_like_update))


def test_cluster_envelope_projection_is_invariant_to_internal_rotation():
    u = jnp.eye(6, dtype=jnp.float64)[:, :2]
    angle = jnp.asarray(0.73, dtype=jnp.float64)
    rotation = jnp.array(
        [[jnp.cos(angle), -jnp.sin(angle)],
         [jnp.sin(angle), jnp.cos(angle)]],
        dtype=jnp.float64,
    )
    vector = jnp.arange(1.0, 7.0, dtype=jnp.float64)
    reference, reference_rank, _ = x.project_onto_basis_envelope(u, vector)
    rotated, rotated_rank, _ = x.project_onto_basis_envelope(
        jnp.concatenate([u, u @ rotation], axis=1), vector
    )
    assert reference_rank == 2
    assert rotated_rank == 2
    assert jnp.allclose(rotated, reference, atol=1e-12, rtol=1e-12)


def test_cluster_envelope_repeated_basis_does_not_inflate_rank():
    u = jnp.eye(8, dtype=jnp.float64)[:, :3]
    vector = jnp.linspace(-1.0, 1.0, 8, dtype=jnp.float64)
    projected, numerical_rank, _ = x.project_onto_basis_envelope(
        jnp.concatenate([u, 0.7 * u, 0.2 * u], axis=1), vector
    )
    assert numerical_rank == 3
    assert jnp.allclose(projected, u @ (u.T @ vector), atol=1e-12)


def test_cluster_envelope_captures_genuinely_new_orthogonal_block():
    current = jnp.eye(8, dtype=jnp.float64)[:, :2]
    history = jnp.eye(8, dtype=jnp.float64)[:, 2:4]
    vector = jnp.array([1.0, 2.0, 3.0, 4.0, 9.0, 8.0, 7.0, 6.0])
    projected, numerical_rank, _ = x.project_onto_basis_envelope(
        jnp.concatenate([current, history], axis=1), vector
    )
    assert numerical_rank == 4
    assert jnp.allclose(projected, jnp.array([1., 2., 3., 4., 0., 0., 0., 0.]))


def test_cluster_envelope_orthonormalization_matches_projection():
    basis = jnp.array(
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
        dtype=jnp.float64,
    )
    vector = jnp.array([2.0, -3.0, 5.0], dtype=jnp.float64)
    q, numerical_rank, _ = x.orthonormalize_basis_envelope(basis)
    projected, reference_rank, _ = x.project_onto_basis_envelope(basis, vector)
    assert numerical_rank == reference_rank == 2
    tolerance = 1e-10 if jax.config.x64_enabled else 1e-6
    np.testing.assert_allclose(
        q.T @ q,
        jnp.diag(jnp.array([0.0, 1.0, 1.0])),
        atol=tolerance,
    )
    np.testing.assert_allclose(
        q @ (q.T @ vector), projected, rtol=tolerance, atol=tolerance
    )


def test_positive_curvature_column_weights_do_not_change_envelope_projector():
    """Scheme-B weights cannot change a fully retained enclosing span."""
    basis = jnp.array(
        [[1.0, 0.0, 0.2], [0.0, 1.0, 0.3], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        dtype=jnp.float64,
    )
    weights = jnp.sqrt(jnp.array([9.0, 0.2, 3.0], dtype=jnp.float64))
    vector = jnp.array([0.7, -1.2, 2.0, 4.0], dtype=jnp.float64)
    q_plain, rank_plain, _ = x.orthonormalize_basis_envelope(basis, 1e-10)
    q_weighted, rank_weighted, _ = x.orthonormalize_basis_envelope(
        basis * weights[None, :], 1e-10
    )
    assert rank_plain == rank_weighted == 3
    tolerance = 1e-10 if jax.config.x64_enabled else 1e-6
    np.testing.assert_allclose(
        q_plain @ (q_plain.T @ vector),
        q_weighted @ (q_weighted.T @ vector),
        rtol=tolerance,
        atol=tolerance,
    )


def test_fixed_budget_history_selection_protects_current_and_uses_current_value():
    current = jnp.eye(5, dtype=jnp.float64)[:, :2]
    history = jnp.eye(5, dtype=jnp.float64)[:, 2:] * jnp.array(
        [2.0, 3.0, 4.0], dtype=jnp.float64
    )
    operator = jnp.eye(5, dtype=jnp.float64)
    gradient = jnp.array([0.0, 0.0, 0.1, 5.0, 0.2], dtype=jnp.float64)
    selected, pool_rank, selected_count, scores, selection_gain = (
        x.select_history_by_current_value(
            current,
            history,
            operator,
            gradient,
            slots=1,
            regularizer=jnp.asarray(0.1, dtype=jnp.float64),
        )
    )
    assert pool_rank == 3
    assert selected_count == 1
    assert selected.shape == (5, 1)
    np.testing.assert_allclose(current.T @ selected, 0.0, atol=1e-12)
    np.testing.assert_allclose(
        jnp.abs(selected[:, 0]),
        jnp.array([0.0, 0.0, 0.0, 1.0, 0.0]),
        atol=1e-12,
    )
    assert jnp.isfinite(scores[0])
    assert 0.0 < selection_gain <= 1.0


def test_cluster_cross_batch_snr_weights_identify_stable_clusters():
    eigenvalues = jnp.array([1.0, 1.01, 10.0, 10.02], dtype=jnp.float64)
    force_a = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)
    force_b = jnp.array([-1.0, -2.0, 3.0, 4.0], dtype=jnp.float64)
    weights, cluster_ids, cluster_count, _, _ = (
        x.cluster_cross_batch_snr_weights(
            eigenvalues,
            force_a,
            force_b,
            gap_threshold=0.1,
            regularizer=jnp.asarray(0.001, dtype=jnp.float64),
        )
    )
    assert cluster_count == 2
    np.testing.assert_array_equal(cluster_ids, jnp.array([0, 0, 1, 1]))
    np.testing.assert_allclose(weights[:2], 0.0, atol=1e-14)
    np.testing.assert_allclose(weights[2:], 1.0, atol=1e-14)


def test_cluster_cross_batch_snr_weights_are_rotation_invariant_within_cluster():
    eigenvalues = jnp.array([1.0, 1.0, 8.0, 8.0], dtype=jnp.float64)
    force_a = jnp.array([1.0, -2.0, 3.0, 1.0], dtype=jnp.float64)
    force_b = jnp.array([0.5, -1.0, 2.5, 0.5], dtype=jnp.float64)
    angle = jnp.asarray(0.61, dtype=jnp.float64)
    rotation = jnp.array(
        [[jnp.cos(angle), -jnp.sin(angle)],
         [jnp.sin(angle), jnp.cos(angle)]],
        dtype=jnp.float64,
    )
    rotated_a = force_a.at[:2].set(rotation.T @ force_a[:2])
    rotated_b = force_b.at[:2].set(rotation.T @ force_b[:2])
    reference = x.cluster_cross_batch_snr_weights(
        eigenvalues, force_a, force_b, 0.1, jnp.asarray(0.001)
    )[0]
    rotated = x.cluster_cross_batch_snr_weights(
        eigenvalues, rotated_a, rotated_b, 0.1, jnp.asarray(0.001)
    )[0]
    np.testing.assert_allclose(rotated, reference, rtol=1e-12, atol=1e-12)
