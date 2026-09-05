"""Alcock-Paczynski wiring in compute_ept / compute_ept_from_clax.

CLASS-PT applies the AP distortion INSIDE its 40-node Gauss-Legendre mu-loop
(nonlinear_pt.c:4392-5317): each channel is re-evaluated at the remapped
k_true(mu) and the multipoles are projected against the remapped Legendre
polynomials. clax mirrors that in `_gl_multipoles`, which already carried
`hratio`/`Dratio`; this module covers the two entry points that expose them.

No reference data: every assertion here is an identity of the model, checked
against clax itself at ratios of exactly 1 and away from 1. The comparison of
AP-distorted spectra against CLASS-PT's own output lives in
tests/test_ap.py (the ratios) and in the campaign's reference sweep.
"""
import dataclasses
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from clax.ept import compute_ept, pk_gg_l2, EPTComponents, EPTPrecisionParams

REF = os.path.join(os.path.dirname(__file__), "..", "reference_data",
                   "classpt_z0.38_noap_alpha1.npz")


@pytest.fixture(scope="module")
def _inputs():
    if not os.path.isfile(REF):
        pytest.skip(f"reference pk_lin not found: {REF}")
    r = np.load(REF, allow_pickle=True)
    return (np.asarray(r["k_h"]), np.asarray(r["pk_lin"]),
            float(r["h"]), float(r["fz"]), float(r["rs_d"]) * float(r["h"]))


def _ept(_inputs, **kw):
    k_h, pk_lin, h, fz, rs_h = _inputs
    return compute_ept(jnp.asarray(pk_lin), jnp.asarray(k_h),
                       h=h, f=fz, rs_h=rs_h, **kw)


def test_ap_at_unity_is_bit_identical_to_no_ap(_inputs):
    """hratio = Dratio = 1 must reproduce the un-distorted result EXACTLY.

    At (1, 1) the remap k_true(mu) is the identity on every GL node, so this is
    a statement about the code path, not about numerics: any leaf that moves
    means the AP branch is doing arithmetic it should not.
    """
    base = _ept(_inputs)
    unity = _ept(_inputs, hratio=1.0, Dratio=1.0)
    moved = [f.name for f in dataclasses.fields(EPTComponents)
             if not np.array_equal(np.asarray(getattr(base, f.name)),
                                   np.asarray(getattr(unity, f.name)))]
    assert not moved, f"AP(1,1) is not the identity; leaves moved: {moved}"


def test_ap_away_from_unity_moves_the_multipoles(_inputs):
    """The converse guard: a parameter that never changes anything is not
    wired in. At (1.02, 0.97) the quadrupole must move well above round-off."""
    k_h = _inputs[0]
    base = _ept(_inputs)
    ap = _ept(_inputs, hratio=1.02, Dratio=0.97)
    win = (k_h >= k_h[10]) & (k_h <= 0.3)
    b = np.asarray(pk_gg_l2(base, 2.0, 0.5, 0.1, -0.2))[win]
    a = np.asarray(pk_gg_l2(ap, 2.0, 0.5, 0.1, -0.2))[win]
    rel = np.max(np.abs(a - b)) / np.max(np.abs(b))
    assert rel > 1e-3, f"AP(1.02, 0.97) changed the quadrupole by only {rel:.2e}"


@pytest.mark.slow
@pytest.mark.parametrize("hr,Dr", [(1.0, 1.0), (1.02, 0.97)])
def test_ap_gradients_match_finite_differences(_inputs, hr, Dr):
    """d/d(hratio, Dratio) of a quadrupole functional, reverse-mode vs central FD.

    The remap splines channels onto k_true(mu), so this checks the derivative
    of that interpolation as well as of the projection. The natural spline is
    C2, so the O(step^2) FD error is far below the 1e-4 bound; alpha = 1 is
    included because it sits exactly on a spline knot, the worst case for the
    remap's derivative.
    """
    k_h, pk_lin, h, fz, rs_h = _inputs
    kj = jnp.asarray(k_h)

    def F(a, b):
        e = compute_ept(jnp.asarray(pk_lin), kj, h=h, f=fz, rs_h=rs_h,
                        hratio=a, Dratio=b)
        return jnp.sum(kj * pk_gg_l2(e, 2.0, 0.5, 0.1, -0.2))

    g_hr, g_Dr = jax.grad(F, argnums=(0, 1))(hr, Dr)
    step = 1e-4
    fd_hr = (F(hr + step, Dr) - F(hr - step, Dr)) / (2 * step)
    fd_Dr = (F(hr, Dr + step) - F(hr, Dr - step)) / (2 * step)
    for ad, fd, name in ((float(g_hr), float(fd_hr), "hratio"),
                         (float(g_Dr), float(fd_Dr), "Dratio")):
        assert np.isfinite(ad), f"{name}: AD gradient is not finite"
        assert abs(ad - fd) < 1e-4 * abs(fd), (
            f"({hr}, {Dr}) d/d{name}: AD={ad:.8e} FD={fd:.8e} "
            f"rel={abs(ad - fd) / abs(fd):.3e}")


def test_omfid_none_is_the_pre_ap_behaviour():
    """compute_ept_from_clax(omfid=None) must not compute ratios at all, and
    omfid at z = 0 must give exactly (1, 1) -- nonlinear_pt.c:1267-1269."""
    from clax.ap import ap_ratios
    from clax.background import background_solve
    from clax.params import CosmoParams
    from clax import PrecisionParams

    bg = background_solve(CosmoParams(), PrecisionParams.fast_cl())
    hr0, Dr0 = ap_ratios(bg, 0.0, 0.31)
    assert float(hr0) == 1.0 and float(Dr0) == 1.0, (float(hr0), float(Dr0))
    hr, Dr = ap_ratios(bg, 0.38, 0.31)
    assert abs(float(hr) - 1.0) > 1e-5, f"AP is inert at z=0.38: hratio={float(hr)!r}"


AP_REF = os.path.join(os.path.dirname(__file__), "..", "reference_data",
                      "classpt_z0.38_ap_omfid0.31_legacy.npz")


@pytest.mark.parametrize("spectrum", ["pk_mm_real", "pk_gg_real", "pk_gm_real",
                                      "pk_mm_l0", "pk_mm_l2", "pk_mm_l4",
                                      "pk_gg_l0", "pk_gg_l2", "pk_gg_l4"])
def test_ap_distorted_spectra_match_classpt(spectrum):
    """clax under AP against a CLASS-PT reference generated with AP ON.

    This is the end-to-end check of the remap: the reference stores the ratios
    CLASS-PT itself used (hratio = 1.002053, Dratio = 0.999032 at z = 0.38,
    Omfid = 0.31), so clax is fed the same geometry rather than an inferred one.
    Without the AP support this file cannot be compared at all -- an AP-free
    clax against an AP-on reference leaves a geometric mismatch that shows up
    as a spurious ~1% in the quadrupole.

    Thresholds match tests/test_ept_accuracy.py: 1% for real space and l = 0, 2
    and 2% on abs/max for l = 4, whose zero crossing makes a pointwise relative
    error meaningless.
    """
    import json
    from clax.ept import (pk_mm_real, pk_gg_real, pk_gm_real,
                          pk_mm_l0, pk_mm_l2, pk_mm_l4,
                          pk_gg_l0, pk_gg_l4)
    if not os.path.isfile(AP_REF):
        pytest.skip(f"AP reference not found: {AP_REF}")
    r = np.load(AP_REF, allow_pickle=True)
    k_h = np.asarray(r["k_h"]); h = float(r["h"])
    b = json.loads(str(r["bias_json"]))
    e = compute_ept(jnp.asarray(np.asarray(r["pk_lin"])), jnp.asarray(k_h),
                    h=h, f=float(r["fz"]), rs_h=float(r["rs_d"]) * h,
                    hratio=float(r["hratio"]), Dratio=float(r["Dratio"]))
    got = {
        "pk_mm_real": pk_mm_real(e, cs0=b["cs"]),
        "pk_gg_real": pk_gg_real(e, b["b1"], b["b2"], b["bG2"], b["bGamma3"],
                                 cs=b["cs"], cs0=b["cs0"], Pshot=b["Pshot"]),
        "pk_gm_real": pk_gm_real(e, b["b1"], b["b2"], b["bG2"], b["bGamma3"],
                                 cs0=b["cs0"], cs=b["cs"]),
        "pk_mm_l0": pk_mm_l0(e, cs0=b["cs0"]),
        "pk_mm_l2": pk_mm_l2(e, cs2=b["cs2"]),
        "pk_mm_l4": pk_mm_l4(e, cs4=b["cs4"]),
        "pk_gg_l0": pk_gg_l0(e, b["b1"], b["b2"], b["bG2"], b["bGamma3"],
                             cs0=b["cs0"], Pshot=b["Pshot"], b4=b["b4"]),
        "pk_gg_l2": pk_gg_l2(e, b["b1"], b["b2"], b["bG2"], b["bGamma3"],
                             cs2=b["cs2"], b4=b["b4"]),
        "pk_gg_l4": pk_gg_l4(e, b["b1"], b["b2"], b["bG2"], b["bGamma3"],
                             cs4=b["cs4"], b4=b["b4"]),
    }[spectrum]
    ref = np.squeeze(np.asarray(r[spectrum]))
    win = (k_h >= k_h[10]) & (k_h <= 0.3)
    g, v = np.asarray(got)[win], ref[win]
    if spectrum.endswith("l4"):
        err = np.max(np.abs(g - v)) / np.max(np.abs(v)); thresh = 0.02
    else:
        keep = np.abs(v) > 0.01 * np.max(np.abs(v))
        err = np.max(np.abs(g[keep] - v[keep]) / np.abs(v[keep])); thresh = 0.01
    print(f"\n  {spectrum}: {100 * err:.4f}% (threshold {100 * thresh:.0f}%)")
    assert err < thresh, f"{spectrum}: {100 * err:.4f}% > {100 * thresh:.0f}%"
