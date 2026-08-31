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
