"""Tests for ChebyshevInterpolant (issue #31, phase 1)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from clax.interpolation import ChebyshevInterpolant, chebyshev_lobatto_nodes


def test_nodes_ascending_with_endpoints():
    x = chebyshev_lobatto_nodes(-2.0, 3.0, 33)
    assert x.shape == (33,)
    assert np.all(np.diff(x) > 0)
    np.testing.assert_allclose([x[0], x[-1]], [-2.0, 3.0], rtol=0, atol=1e-14)


def test_reproduces_polynomials_exactly():
    # degree-(n-1) polynomials are reproduced to machine precision
    x = chebyshev_lobatto_nodes(0.0, 1.0, 8)
    coeffs = np.array([0.3, -1.2, 2.0, 0.7, -0.1, 0.05, 1.5, -0.4])
    poly = lambda t: sum(c * t**i for i, c in enumerate(coeffs))
    interp = ChebyshevInterpolant(jnp.asarray(x), jnp.asarray(poly(x)))
    t = jnp.linspace(0.0, 1.0, 101)
    np.testing.assert_allclose(np.asarray(interp.evaluate(t)),
                               np.asarray(poly(np.asarray(t))),
                               rtol=0, atol=1e-11)


def test_exact_at_nodes():
    # the barycentric formula's removable singularity must be handled
    x = chebyshev_lobatto_nodes(0.0, 2.0, 17)
    y = jnp.sin(jnp.asarray(x))
    interp = ChebyshevInterpolant(jnp.asarray(x), y)
    np.testing.assert_allclose(np.asarray(interp.evaluate(jnp.asarray(x))),
                               np.asarray(y), rtol=0, atol=1e-13)


def test_spectral_convergence_exp():
    # error drops by orders of magnitude between n=8 and n=24 for e^t
    errs = []
    for n in (8, 24):
        x = chebyshev_lobatto_nodes(-1.0, 1.0, n)
        interp = ChebyshevInterpolant(jnp.asarray(x), jnp.exp(jnp.asarray(x)))
        t = jnp.linspace(-1.0, 1.0, 501)
        errs.append(float(jnp.max(jnp.abs(interp.evaluate(t) - jnp.exp(t)))))
    assert errs[1] < errs[0] * 1e-6, f"not spectral: {errs}"


def test_clip_saturates_like_cubicspline():
    x = chebyshev_lobatto_nodes(0.0, 1.0, 9)
    y = jnp.asarray(np.asarray(x) ** 2)
    interp = ChebyshevInterpolant(jnp.asarray(x), y)
    out = interp.evaluate(jnp.asarray([-5.0, 6.0]))
    np.testing.assert_allclose(np.asarray(out), [0.0, 1.0], atol=1e-12)


def test_pytree_roundtrip_and_grad():
    x = chebyshev_lobatto_nodes(0.0, 1.0, 9)
    interp = ChebyshevInterpolant(jnp.asarray(x), jnp.sin(jnp.asarray(x)))
    leaves, treedef = jax.tree_util.tree_flatten(interp)
    interp2 = jax.tree_util.tree_unflatten(treedef, leaves)
    g = jax.grad(lambda t: interp2.evaluate(t))(0.37)
    assert jnp.isfinite(g)
    assert abs(float(g) - float(jnp.cos(0.37))) < 1e-6


def test_k_grid_chebyshev_type():
    from dataclasses import replace as dc_replace
    from clax import PrecisionParams
    from clax.perturbations import _k_grid

    prec_log = PrecisionParams.fast_cl()
    prec_ch = dc_replace(prec_log, pt_k_grid_type="chebyshev")
    k_log, k_ch = _k_grid(prec_log), _k_grid(prec_ch)
    assert k_ch.shape == k_log.shape                     # same n_k formula
    assert np.all(np.diff(np.asarray(k_ch)) > 0)
    np.testing.assert_allclose(
        [float(k_ch[0]), float(k_ch[-1])],
        [float(k_log[0]), float(k_log[-1])], rtol=1e-12)  # same endpoints
    np.testing.assert_array_equal(np.asarray(_k_grid(prec_log)),
                                  np.asarray(k_log))      # default unchanged


def test_source_interp_chebyshev_matches_spline_on_smooth():
    from clax.harmonic import _interp_sources_to_fine_k
    lk_ch = jnp.asarray(chebyshev_lobatto_nodes(np.log(1e-4), np.log(0.5), 80))
    lk_fine = jnp.linspace(lk_ch[0], lk_ch[-1], 2000)
    tau = jnp.linspace(0.0, 1.0, 7)
    # smooth BAO-like source: slow envelope + 0.02 Mpc^-1-scale ripple
    src = (jnp.exp(-0.5 * lk_ch[:, None] ** 2)
           * (1.0 + 0.05 * jnp.sin(jnp.exp(lk_ch)[:, None] / 0.02))
           * (1.0 + tau[None, :]))
    out_ch = _interp_sources_to_fine_k([src], lk_ch, lk_fine,
                                       method="chebyshev")[0]
    out_sp = _interp_sources_to_fine_k([src], lk_ch, lk_fine,
                                       method="spline")[0]
    ref = (jnp.exp(-0.5 * lk_fine[:, None] ** 2)
           * (1.0 + 0.05 * jnp.sin(jnp.exp(lk_fine)[:, None] / 0.02))
           * (1.0 + tau[None, :]))
    err_ch = float(jnp.max(jnp.abs(out_ch - ref)))
    err_sp = float(jnp.max(jnp.abs(out_sp - ref)))
    assert err_ch <= err_sp * 1.5, (err_ch, err_sp)   # at worst comparable
    assert err_ch < 1e-6                              # spectrally small


@pytest.mark.slow
def test_cls_chebyshev_path_matches_spline_path(fast_mode):
    """Full pipeline: chebyshev k-solve + barycentric interp vs the
    default log-solve + spline interp, both at fast_cl density. Two
    independent discretizations of the same integral; at matched density
    they must agree well under the benchmark's 0.5% gate."""
    if fast_mode:
        pytest.skip("two perturbation solves")
    from dataclasses import replace as dc_replace
    from clax import CosmoParams, PrecisionParams
    from clax.background import background_solve
    from clax.thermodynamics import thermodynamics_solve
    from clax.perturbations import perturbations_solve
    from clax.harmonic import compute_cls_all_fast

    params = CosmoParams()
    results = {}
    for name, prec, kw in [
        ("log", PrecisionParams.fast_cl(), {}),
        ("cheb", dc_replace(PrecisionParams.fast_cl(),
                            pt_k_grid_type="chebyshev"),
         {"k_interp_method": "chebyshev"}),
    ]:
        bg = background_solve(params, prec)
        th = thermodynamics_solve(params, prec, bg)
        pt = perturbations_solve(params, prec, bg, th)
        results[name] = compute_cls_all_fast(pt, params, bg, l_max=600, **kw)
    for l in (20, 100, 500):
        r = float(results["cheb"]["tt"][l] / results["log"]["tt"][l])
        assert abs(r - 1.0) < 0.005, f"TT l={l}: cheb/log ratio {r}"
