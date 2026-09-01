"""Channel tests for the h-dependence of compute_ept_from_clax.

GPU job 13313 attributed the stage-level AD-vs-FD h-gradient gap to the
frozen k_mpc = k_h * stop_gradient(h) resampling channel (-9.48e4 of the
stage gradient), with the frozen-pk_nw IR split (+3.27e4) as the
documented residual and the rs_h/f/h-arg channels negligible (-1.0e2).
These tests pin the fix.

CLOSURE UPDATE (job 14146, fix/ir-resummation-traced @ 322a6ab): the
frozen-pk_nw IR split is now closed too (commit 01b5162 traced
``_ir_resummation_jax``; commit 322a6ab wired it into
``compute_ept_from_clax``), on top of the pre-existing k_mpc fix. See
``test_stage_grad_h_matches_fd_per_k`` below for the measured effect.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from clax import CosmoParams
from clax.ept import compute_ept_from_clax, pk_mm_real


@pytest.fixture(scope="module")
def stage_setup(request):
    params, _prec, bg, _th, pt = request.getfixturevalue("pipeline_fast_cl_k5")
    return params, bg, pt


def _pk_of_h(bg, pt):
    base = CosmoParams()

    def f(h_val):
        p = base.replace(h=h_val)
        return pk_mm_real(compute_ept_from_clax(p, bg, pt, z=0.0))

    return f


def test_stage_grad_h_matches_fd_per_k(fast_mode, request):
    """Per-k d(pk_mm_real)/dh through the EPT stage: AD vs central FD.

    RED before the k_mpc channel is traced: FD carries the resampling
    term dP/dlnk * (1/h) which AD drops entirely, giving order-unity
    per-k relative errors at BAO scales. GREEN after the k_mpc fix
    (job 14140): residual was the frozen-pk_nw share, median 3.294e-02.

    FURTHER CLOSED (job 14146, fix/ir-resummation-traced @ 322a6ab): with
    the frozen-pk_nw IR split also traced through JAX now (commit 01b5162,
    wired in by 322a6ab), this frozen-bg/pt stage test (no full pipeline
    re-solve, so no discretization noise -- unlike the end-to-end h test in
    ``tests/test_ept_gradients.py``) measured median rel 9.825e-03
    (max 3.619e-02) over 31 modes in [0.05,0.3] -- ~3.35x lower than the
    prior 3.294e-02. The bound below (0.02) is 2x the measured 9.825e-03
    (=0.01965), rounded up to one significant figure -- never tighter than
    2x measured, per this branch's ratchet rule."""
    if fast_mode:
        pytest.skip("uses the shared full-mode pipeline fixture")
    params, bg, pt = request.getfixturevalue("stage_setup")
    f = _pk_of_h(bg, pt)
    h0 = float(params.h)

    g_ad = jax.jacfwd(f)(jnp.asarray(h0))  # fwd == rev for this stage; cheap
    eps = 1e-3
    g_fd = (f(h0 + eps) - f(h0 - eps)) / (2.0 * eps)

    k_h = np.asarray(compute_ept_from_clax(params, bg, pt, z=0.0).kh)
    sel = (k_h > 0.05) & (k_h < 0.3)
    rel = np.abs(np.asarray(g_ad - g_fd))[sel] / (
        np.abs(np.asarray(g_fd))[sel] + 1e-30)
    med = float(np.median(rel))
    print(f"\nper-k d(pk_mm)/dh AD-vs-FD: median rel {med:.3e} "
          f"(max {float(rel.max()):.3e}) over {int(sel.sum())} modes in [0.05,0.3]")
    # 0.02 = 2x the measured 9.825e-03 (job 14146), rounded up to one
    # significant figure -- never tighter than 2x measured. See docstring:
    # the traced IR-resummation splitter closed most of the frozen-pk_nw
    # share on top of the pre-existing k_mpc fix.
    assert med < 0.02, (
        f"median per-k AD-vs-FD rel err {med:.3e} >= 0.02: either the "
        f"k_mpc resampling channel (job 13313: -9.48e4) or the frozen-pk_nw "
        f"IR split (closed by commit 01b5162/322a6ab) has regressed")


def test_growth_rate_is_not_hardcoded(request, fast_mode):
    """compute_ept_from_clax must use the background growth rate, not 0.8.

    bg.Omega_m_of_z does not exist, so the hasattr fallback at
    ept.py:2061 silently yields the literal 0.8 for every cosmology and
    redshift. LCDM at z=0 has f ~ 0.52-0.53. RED until Task 4 routes
    f through bg.f_of_loga."""
    if fast_mode:
        pytest.skip("uses the shared full-mode pipeline fixture")
    params, bg, pt = request.getfixturevalue("stage_setup")
    ept = compute_ept_from_clax(params, bg, pt, z=0.0)
    f_val = float(jax.lax.stop_gradient(jnp.asarray(ept.f)))
    ref = float(bg.f_of_loga.evaluate(jnp.log(jnp.asarray(1.0))))
    assert abs(f_val - ref) < 0.01, (
        f"EPT growth rate {f_val} != background f(z=0) {ref:.4f} "
        f"(the hardcoded-0.8 fallback is still active)")
    # Physical oracle bound, independent of the f_of_loga implementation:
    # LCDM z=0 has f ~ Omega_m**0.55 = 0.315**0.55 ~ 0.53 (measured 0.5258,
    # GPU job 14140). Catches a broken f_grid/spline that the
    # self-referential check above cannot.
    assert 0.45 < f_val < 0.60, (
        f"EPT growth rate {f_val} outside the physical LCDM z=0 range")


def test_eptcomponents_pytree_roundtrip(request, fast_mode):
    """h/f/sigma2 are leaves: tree_map touches them, jit caching is safe."""
    if fast_mode:
        pytest.skip("uses the shared full-mode pipeline fixture")
    params, bg, pt = request.getfixturevalue("stage_setup")
    ept = compute_ept_from_clax(params, bg, pt, z=0.0)
    leaves, treedef = jax.tree_util.tree_flatten(ept)
    ept2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert float(ept2.f) == float(ept.f)
    assert float(ept2.h) == float(ept.h)
    n_scalar_leaves = sum(1 for l in leaves if jnp.ndim(l) == 0)
    assert n_scalar_leaves >= 4, "h/f/sigma2/delta_sigma2 must be leaves"
