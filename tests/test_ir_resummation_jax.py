"""Parity + differentiability of the traced IR-resummation splitter."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from clax.ept import (_ir_resummation_numpy, _ir_resummation_jax,
                      ept_kgrid, EPTPrecisionParams)

REF = np.load("reference_data/lcdm_fiducial/pk.npz")


@pytest.fixture(scope="module")
def pk_setup():
    # Schema drift vs the brief: pk.npz no longer has flat "k"/"pk" keys
    # (now per-species/per-redshift). "k" is in Mpc^-1, "pk_lin_z0" in
    # Mpc^3 -- converted to h-units (h/Mpc, (Mpc/h)^3) exactly as
    # tests/test_ept_gradients.py does for the same reference file.
    h = 0.6736
    k_h = ept_kgrid(EPTPrecisionParams())
    lk, lp = np.log(REF["k"] / h), np.log(REF["pk_lin_z0"] * h ** 3)
    pk = np.exp(np.interp(np.log(k_h), lk, lp))
    return k_h, pk


def test_parity_with_numpy_splitter(pk_setup):
    k_h, pk = pk_setup
    h, rs = 0.6736, 99.05
    nw_np, w_np, s2_np, ds2_np = _ir_resummation_numpy(pk, k_h, rs_h=rs, h=h)
    nw_j, w_j, s2_j, ds2_j = _ir_resummation_jax(
        jnp.asarray(pk), k_h, jnp.asarray(rs), h)
    np.testing.assert_allclose(np.asarray(nw_j), nw_np, rtol=1e-9)
    np.testing.assert_allclose(np.asarray(w_j), w_np, rtol=0,
                               atol=1e-9 * np.max(np.abs(w_np)))
    np.testing.assert_allclose(float(s2_j), s2_np, rtol=1e-10)
    np.testing.assert_allclose(float(ds2_j), ds2_np, rtol=1e-10)


def test_pk_nw_gradient_exists_and_matches_fd(pk_setup):
    """THE red property: d(sum pk_nw)/d(amplitude) is nonzero and matches
    FD. Under the frozen NumPy splitter this derivative is 0 by
    construction -- only a traced splitter can satisfy this test."""
    k_h, pk = pk_setup
    pk_j = jnp.asarray(pk)

    def f(amp):
        nw, _, _, _ = _ir_resummation_jax(amp * pk_j, k_h,
                                          jnp.asarray(99.05), 0.6736)
        return jnp.sum(nw)

    g = float(jax.grad(f)(jnp.asarray(1.0)))
    eps = 1e-4
    fd = (float(f(jnp.asarray(1.0 + eps)))
          - float(f(jnp.asarray(1.0 - eps)))) / (2 * eps)
    assert abs(g) > 0.0
    assert abs(g - fd) / abs(fd) < 1e-6, (g, fd)


def test_sigma2_rs_gradient_matches_fd(pk_setup):
    k_h, pk = pk_setup

    def s2(rs):
        _, _, s, _ = _ir_resummation_jax(jnp.asarray(pk), k_h, rs, 0.6736)
        return s

    g = float(jax.grad(s2)(jnp.asarray(99.05)))
    eps = 1e-3
    fd = (float(s2(jnp.asarray(99.05 + eps)))
          - float(s2(jnp.asarray(99.05 - eps)))) / (2 * eps)
    assert abs(g - fd) / abs(fd) < 1e-6, (g, fd)
