"""Derivatives of the redshift-space multipoles with respect to the linear spectrum, through the
traced IR splitter (compute_ept with _ir_precomputed = _ir_resummation_jax(...), the path
compute_ept_from_clax uses).

Before Oct 10, 2026 most of d(P_ell)/d(P_lin) was missing on this path. A stop_gradient snapshot of
the IR split (the wiggle / no-wiggle decomposition of P_lin) fed both the one-loop FFTLog basis and
the no-wiggle spectrum used by the angular integration (tree, counterterms, P13ratio, IFG2
rescaling), so d(P_ell)/d(P_lin) lacked the no-wiggle tree response as well as the one-loop
response. Measured RED, AD vs FD at l=0/2/4, lcdm_fiducial (job 24203): amplitude 0.91/0.93/1.98,
stretch 1.06/1.60/1.70, wiggle 0.035/0.027/0.13; the ps_1loop validation (Task 7, jobs 24165/24167,
z=0.38) first saw ~90% (amplitude) / ~100% (stretch). The existing tests/test_ept_gradients.py holds
the IR split fixed in both arms and never saw it.

Multi-cosmology rule: the lcdm_cosmology grid (fiducial only under --fast). P_lin comes from the
stored reference tables; f and rs_h are fixed representative values (the derivative checks do not
depend on them).
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from clax.ept import (EPTPrecisionParams, _ir_resummation_jax, compute_ept, ept_kgrid,
                      pk_gg_l0, pk_gg_l2, pk_gg_l4, pk_mm_l0)
from tests.conftest import cosmology_reference_dir

BIAS = (2.0, -0.5, 0.1, -0.1)        # b1, b2, bG2, bGamma3: exercises every RSD loop family
F_GROWTH = 0.75                       # z ~ 0.4 growth rate, representative
RS_H = 99.0                           # r_d h [Mpc/h], the compute_ept default
WINDOW = (0.01, 0.3)                  # h/Mpc, the cross-validation's comparison window
FD_STEP = 1e-4                        # relative step along the tangent
AD_FD_TOL = 1e-6                      # measured worst 2.8e-9 over the 5-point grid (Oct 10, 2026);
                                      # central-FD truncation floor at step 1e-4 is ~1e-8; a partial
                                      # re-freeze reads >= 3e-2
PRIMAL_TOL = 1e-8
TANGENTS = ("amplitude", "stretch", "wiggle")


@pytest.fixture(scope="module")
def prec():
    return EPTPrecisionParams()


@pytest.fixture
def setup(lcdm_cosmology, prec):
    name, cosmo = lcdm_cosmology
    ref = cosmology_reference_dir(name)
    if ref is None:
        pytest.skip(f"no reference_data for {name}")
    d = np.load(os.path.join(ref, "pk.npz"))
    h = float(cosmo.h)
    k_h_ref, pk_h_ref = d["k"] / h, d["pk_lin_z0"] * h**3           # h/Mpc, (Mpc/h)^3
    k_ept = ept_kgrid(prec)
    # the spline extrapolates P_lin outside the reference k-range at the EPT grid ends (outside the [0.01, 0.3] window; identical in the AD and FD arms)
    spl = CubicSpline(np.log(k_h_ref), np.log(pk_h_ref), extrapolate=True)
    pk_lin = np.exp(spl(np.log(k_ept)))
    assert np.all(np.isfinite(pk_lin))
    window = (k_ept >= WINDOW[0]) & (k_ept <= WINDOW[1])
    dlnp = spl.derivative()(np.log(k_ept))
    pk_j = jnp.asarray(pk_lin)
    ir0 = _ir_resummation_jax(pk_j, k_ept, jnp.asarray(RS_H), h)
    tangents = {"amplitude": pk_j, "stretch": pk_j * jnp.asarray(dlnp), "wiggle": ir0[1]}
    return dict(name=name, h=h, k_ept=k_ept, pk=pk_j, window=window, tangents=tangents)


def rsd_multipoles_fn(k_ept, h, f, prec, window):
    """F(pk_lin) -> (P_0, P_2, P_4) on the window, IR split TRACED (compute_ept_from_clax's path)."""
    k_j = jnp.asarray(k_ept)

    def F(pk_lin):
        ir = _ir_resummation_jax(pk_lin, k_ept, jnp.asarray(RS_H), h)
        e = compute_ept(pk_lin, k_j, h=h, f=f, prec=prec, _ir_precomputed=ir, rs_h=RS_H)
        return jnp.stack([pk_gg_l0(e, *BIAS)[window], pk_gg_l2(e, *BIAS)[window], pk_gg_l4(e, *BIAS)[window]])

    return F


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


@pytest.mark.parametrize("tangent", TANGENTS)
def test_rsd_multipole_jvp_matches_finite_differences(setup, prec, tangent):
    F = rsd_multipoles_fn(setup["k_ept"], setup["h"], F_GROWTH, prec, setup["window"])
    pk, t = setup["pk"], setup["tangents"][tangent]
    _, ad = jax.jvp(F, (pk,), (t,))
    fd = (F(pk + FD_STEP * t) - F(pk - FD_STEP * t)) / (2 * FD_STEP)
    ad, fd = np.asarray(ad), np.asarray(fd)
    assert np.all(np.isfinite(ad)), f"ERROR {setup['name']} {tangent}: non-finite AD"
    assert np.all(np.isfinite(fd)), f"ERROR {setup['name']} {tangent}: non-finite FD"
    errs = [_rel(ad[i], fd[i]) for i in range(3)]
    print(f"{setup['name']} d P_ell / d P_lin along {tangent}: AD vs FD l=0,2,4 = "
          + " ".join(f"{e:.2e}" for e in errs))
    worst = float(np.max(np.asarray(errs)))               # np.max propagates a NaN; Python max can skip it
    assert worst < AD_FD_TOL, f"ERROR {setup['name']} {tangent}: AD vs FD {worst:.3e} > {AD_FD_TOL}"


def test_rsd_multipole_forward_equals_reverse(setup, prec):
    F = rsd_multipoles_fn(setup["k_ept"], setup["h"], F_GROWTH, prec, setup["window"])
    pk, t = setup["pk"], setup["tangents"]["stretch"]
    out, ad = jax.jvp(F, (pk,), (t,))
    c = 1.0 / (jnp.abs(out) + 1.0)                        # a fixed, well-scaled cotangent
    _, vjp = jax.vjp(F, pk)
    (g,) = vjp(c)
    lhs, rhs = float(jnp.vdot(c, ad)), float(jnp.vdot(g, t))
    assert np.isfinite(lhs) and np.isfinite(rhs)
    rel = abs(lhs - rhs) / abs(lhs)
    print(f"{setup['name']} c.(J t) jvp vs vjp rel diff {rel:.2e}")
    assert rel < 1e-10, f"ERROR {setup['name']}: forward/reverse mismatch {rel:.2e}"


def test_traced_path_primal_matches_numpy_path(setup, prec):
    """The traced-splitter path and the NumPy path give the same multipoles. Print the value: the
    gradient fix removes a stop_gradient only, so this number must not move."""
    F = rsd_multipoles_fn(setup["k_ept"], setup["h"], F_GROWTH, prec, setup["window"])
    traced = np.asarray(F(setup["pk"]))
    e_np = compute_ept(setup["pk"], jnp.asarray(setup["k_ept"]), h=setup["h"], f=F_GROWTH, prec=prec, rs_h=RS_H)
    w = setup["window"]
    numpy_path = np.stack([np.asarray(pk_gg_l0(e_np, *BIAS))[w], np.asarray(pk_gg_l2(e_np, *BIAS))[w],
                           np.asarray(pk_gg_l4(e_np, *BIAS))[w]])
    assert np.all(np.isfinite(traced)) and np.all(np.isfinite(numpy_path)), \
        f"ERROR {setup['name']}: non-finite primal multipoles"
    rel = float(np.max(np.asarray([_rel(traced[i], numpy_path[i]) for i in range(3)])))
    print(f"{setup['name']} traced vs numpy path max rel diff {rel:.3e}")
    assert rel <= PRIMAL_TOL, f"ERROR {setup['name']}: traced vs numpy path {rel:.3e}"


def test_matter_monopole_jvp_matches_finite_differences(setup, prec):
    """The matter-only RSD path (pk_mm_l0) through the traced splitter, amplitude direction."""
    k_j = jnp.asarray(setup["k_ept"])

    def F(pk_lin):
        ir = _ir_resummation_jax(pk_lin, setup["k_ept"], jnp.asarray(RS_H), setup["h"])
        return pk_mm_l0(compute_ept(pk_lin, k_j, h=setup["h"], f=F_GROWTH, prec=prec, _ir_precomputed=ir, rs_h=RS_H))[setup["window"]]

    pk, t = setup["pk"], setup["tangents"]["amplitude"]
    _, ad = jax.jvp(F, (pk,), (t,))
    fd = (F(pk + FD_STEP * t) - F(pk - FD_STEP * t)) / (2 * FD_STEP)
    err = _rel(np.asarray(ad), np.asarray(fd))
    print(f"{setup['name']} d pk_mm_l0 / d P_lin (amplitude): AD vs FD {err:.2e}")
    assert err < AD_FD_TOL, f"ERROR {setup['name']}: pk_mm_l0 AD vs FD {err:.3e}"
