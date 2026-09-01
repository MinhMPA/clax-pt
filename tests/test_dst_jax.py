"""jnp DST-II/IDST-II parity vs scipy (issue #30 pk_nw class)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.fft import dst as sdst, idst as sidst

from clax.interpolation import dst2_ortho, idst2_ortho


@pytest.mark.parametrize("n", [4, 8, 16, 64, 1024, 65536])
def test_dst2_matches_scipy(n):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n)
    np.testing.assert_allclose(np.asarray(dst2_ortho(jnp.asarray(x))),
                               sdst(x, type=2, norm="ortho"),
                               rtol=0, atol=1e-11 * max(1.0, n / 1024))


@pytest.mark.parametrize("n", [4, 8, 16, 64, 1024, 65536])
def test_idst2_matches_scipy_and_roundtrips(n):
    rng = np.random.default_rng(1)
    x = rng.standard_normal(n)
    np.testing.assert_allclose(np.asarray(idst2_ortho(jnp.asarray(x))),
                               sidst(x, type=2, norm="ortho"),
                               rtol=0, atol=1e-11 * max(1.0, n / 1024))
    np.testing.assert_allclose(
        np.asarray(idst2_ortho(dst2_ortho(jnp.asarray(x)))), x,
        rtol=0, atol=1e-11 * max(1.0, n / 1024))


def test_dst2_is_differentiable_linear():
    # linear map: jacobian-vector products must be input-independent
    x = jnp.asarray(np.random.default_rng(2).standard_normal(16))
    v = jnp.asarray(np.random.default_rng(3).standard_normal(16))
    _, jvp1 = jax.jvp(dst2_ortho, (x,), (v,))
    _, jvp2 = jax.jvp(dst2_ortho, (2.0 * x,), (v,))
    np.testing.assert_allclose(np.asarray(jvp1), np.asarray(jvp2), atol=1e-12)
