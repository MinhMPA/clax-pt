"""Parity + differentiability of the traced IR-resummation splitter."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from clax import CosmoParams
from clax.ept import (_ir_resummation_numpy, _ir_resummation_jax,
                      ept_kgrid, EPTPrecisionParams,
                      compute_ept_from_clax, pk_mm_real)

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


def test_stage_grad_ln10A_s_matches_fd_through_traced_ir(fast_mode, request):
    """Stage-level red/green for Task 3's wiring: d(sum(pk_mm_real))/d(ln10A_s)
    through compute_ept_from_clax, end to end from CosmoParams with bg/pt
    frozen (mirrors tests/test_ept_h_channels.py's fixture/skip conventions).

    RED before this task: compute_ept_from_clax still calls the frozen
    _ir_resummation_numpy splitter, which drops d(pk_nw)/d(pk_lin_h)
    entirely -- the same structural residual documented at ~1.39% in
    tests/test_ept_gradients.py::test_grad_ln10A_s_end_to_end_from_cosmoparams_matches_fd
    (job 13132). GREEN after: routing through _ir_resummation_jax lets
    gradients flow through pk_nw too, tightening the bound to <5e-3.
    """
    if fast_mode:
        pytest.skip("uses the shared full-mode pipeline fixture")
    params, _prec, bg, _th, pt = request.getfixturevalue("pipeline_fast_cl_k5")

    base = CosmoParams()

    def f(ln10A_s):
        p = base.replace(ln10A_s=ln10A_s)
        return jnp.sum(pk_mm_real(compute_ept_from_clax(p, bg, pt, z=0.0)))

    x0 = float(params.ln10A_s)
    g_ad = float(jax.grad(f)(jnp.asarray(x0)))
    eps = 1e-3
    g_fd = (float(f(x0 + eps)) - float(f(x0 - eps))) / (2.0 * eps)
    rel = abs(g_ad - g_fd) / (abs(g_fd) + 1e-30)
    print(f"\nd(sum(pk_mm_real))/d(ln10A_s) stage grad: "
          f"AD={g_ad:.6e} FD={g_fd:.6e} rel={rel:.4e}")
    assert rel < 5e-3, (
        f"AD vs FD disagree for d(sum(pk_mm_real))/d(ln10A_s): "
        f"AD={g_ad:.6e}, FD={g_fd:.6e}, rel={rel:.4e} >= 5e-3 "
        f"(expected <5e-3 once compute_ept_from_clax routes through "
        f"_ir_resummation_jax)")
