# tests/test_ps1loop_gradients.py
"""Derivatives (spec §5.5): forward-mode tangents of the TOTAL P_0, P_2, P_4 at the PFS bias point
(pfs.bias_point) along f, alpha_perp, alpha_para, a k-stretch t = P dlnP/dlnk and the amplitude t = P.

What must stay true, and what breaks if it does not:
  * ps-vs-clax-closed (test_cross_code_tangents): d(ps - clax)/dt equals dD/dt, the derivative of the
    documented model difference D(x), x = (f, alpha_perp, alpha_para, P_lin) (br.predicted_total_difference:
    counterterm / fingers-of-god / a0 maps, omitted projections, ps's IR-resummation scheme; spec §4.2
    items 1-5), at the code-accuracy thresholds 1% / 1% / 2% for l = 0 / 2 / 4. Task 5's discrimination
    rule applies: where the direct ps-vs-clax derivative difference exceeds 0.1% of the clax derivative's
    scale, the closed residual must be at most half of it, so a line cannot pass merely by being small.
    If this breaks, the two codes respond differently to f, the AP distortion or the linear spectrum in a
    way no documented model difference explains -- a one-loop or AP-remapping defect in one of them, which
    would bias Fisher forecasts and HMC gradients for f sigma_8 and the BAO/AP dilations even where the
    spectra themselves agree. USER DECISION (Oct 10, 2026) and controller ruling: the closure gate applies
    to all five tangents; it reduces to the direct gate wherever dD/dt is negligible. The direct
    ps-vs-clax numbers are logged and printed but not asserted: they contain dD/dt, which along the AP
    directions is large (scratch job 24167, lcdm_fiducial: alpha_para 3.1 / 10.4 / 8.7% at l = 0 / 2 / 4).
  * ps-AD-vs-FD: ps_1loop's own jvp equals its central differences to 1% (CLAUDE.md gradient target);
    a failure is an AD defect in ps_1loop.
  * traced-IR primal: the differentiable IR split clax uses here reproduces the validated NumPy IR path
    to 1e-8, so the clax derivatives are taken at the validated spectra.

D is NumPy-valued (it reads clax's mu-loop through the br.capture_gl_inputs spy), so it cannot be
jvp'd: dD/dt is the central difference of D at x0 +/- FD_STEP * t, the step ps_1loop's own FD check uses.
D is smooth in x: its clax pieces run clax's NumPy IR path, whose central differences equal the traced
path's to ~1e-11 (scratch job 24167), and its ps piece is ps's own P_IR, whose AD equals its central
differences (test_ps1loop_ad_matches_central_differences). The truncation error of the central difference
is O(FD_STEP^2) ~ 1e-8 relative, far below the 1% gate. The counterterm map contributes to D: with IR
resummation on it is not zero (Task 5: up to ~1% of c0's and ~8% of cfog's coefficient at z = 0.38),
and the bias point carries c0 = c2 = c4 = 2 (Mpc/h)^2 and cfog = 500 (Mpc/h)^4; a0 = 0 there, so the
a0 map adds nothing.

lcdm_fiducial, massive_nu_030, w0wa_m07_m10 at z = 0.38 (AP non-trivial) in full mode; lcdm_fiducial
under --fast. Multi-cosmology rule: 3 cosmologies."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ps_1loop_jax")
import jax
import jax.numpy as jnp

from clax.interpolation import CubicSpline
from tests import ept_campaign_utils as cu
from tests import pfs_survey as pfs
from tests import ps1loop_bridge as br

GRAD_CASES = ("lcdm_fiducial", "massive_nu_030", "w0wa_m07_m10")
Z = 0.38
FD_STEP = 1e-4
FD_THRESHOLD = 0.01
DISCRIM_MIN = 1e-3     # direct differences below this (fraction of max|jvp_clax|) are held to the gate only
DISCRIM_RATIO = 0.5    # closed residual <= DISCRIM_RATIO x direct difference above DISCRIM_MIN (Task 5 rule)


def pytest_generate_tests(metafunc):
    if "case" in metafunc.fixturenames:
        fast = metafunc.config.getoption("--fast", default=False)
        metafunc.parametrize("case", ["lcdm_fiducial"] if fast else list(GRAD_CASES))


def _inputs(case):
    """(shared inputs, comparison-window mask, PFS bias point) at Z."""
    inp = br.shared_inputs(cu.require_reference(case, Z))
    assert br.has_ap(inp)
    return inp, cu.window(inp["k_h"]), pfs.bias_point(Z)


def _setup(case):
    inp, mask, theta = _inputs(case)
    k_h, kw = jnp.asarray(inp["k_h"]), jnp.asarray(inp["k_h"][mask])
    model = br.ps_model(inp["rbao"])

    def ps_fun(f, a_perp, a_para, pk):
        return br.ps_multipoles_jax(model, k_h, pk, inp["h"], f, a_perp, a_para, theta, kw,
                                    ap=True, ndens=br.NDENS_PROBE)

    def clax_fun(f, a_perp, a_para, pk, ir_traced=True):
        hratio, Dratio = 1.0 / a_para, a_perp
        e = br.clax_ept(inp, pk=pk, f=f, hratio=hratio, Dratio=Dratio, ir_traced=ir_traced)
        cb = br.to_clax(theta, f=f, ndens=br.NDENS_PROBE, Pd2d2_0=e.Pd2d2_0, hratio=hratio, Dratio=Dratio)
        return br.clax_multipoles_jax(e, cb)[:, mask]

    a_perp, a_para = br.ap_alphas(inp["hratio"], inp["Dratio"])
    pk0 = jnp.asarray(inp["pk"])
    x0 = (jnp.asarray(inp["f"]), jnp.asarray(a_perp), jnp.asarray(a_para), pk0)
    t_stretch = pk0 * CubicSpline(jnp.log(k_h), jnp.log(pk0)).derivative(jnp.log(k_h))   # P dlnP/dlnk
    one, zero, zpk = jnp.asarray(1.0), jnp.asarray(0.0), jnp.zeros_like(pk0)
    tangents = {"f": (one, zero, zero, zpk), "a_perp": (zero, one, zero, zpk),
                "a_para": (zero, zero, one, zpk), "stretch": (zero, zero, zero, t_stretch),
                "amplitude": (zero, zero, zero, pk0)}
    return ps_fun, clax_fun, x0, tangents


def _check(case, layer, pairs, thresholds):
    errs, lines = {}, []
    for name, (got, ref) in pairs.items():
        for i, l in enumerate(br.ELLS):
            e = cu.rel(np.asarray(got[i]), np.asarray(ref[i]))
            errs[f"d/d{name}:l{l}"] = {"err": e, "k": float("nan")}
            thr = thresholds[f"l{l}"]
            if not np.isfinite(e) or e > thr:
                lines.append(f"ERROR {case} {layer} d/d{name} l={l}: {100 * e:.3f}% > {100 * thr:.1f}%")
    br.log_record(layer="grad", case=case, z=Z, preset=layer, errors=errs)
    print(f"{case} [{layer}] {br.worst(errs)}")
    assert not lines, "\n".join(lines)


def test_traced_ir_primal_matches_numpy_ir(case):
    """The traced-IR clax path used for derivatives reproduces the validated numpy-IR path."""
    _, clax_fun, x0, _ = _setup(case)
    a = np.asarray(clax_fun(*x0, ir_traced=True))
    b = np.asarray(clax_fun(*x0, ir_traced=False))
    e = float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
    print(f"{case} traced-vs-numpy IR primal max rel diff {e:.3e}")
    br.log_record(layer="grad", case=case, z=Z, preset="traced-ir-primal", errors={"primal": {"err": e, "k": float("nan")}})
    assert e <= 1e-8, f"ERROR {case} traced IR primal differs from numpy IR by {e:.3e}"


def _predicted_difference(inp, theta, kw, x):
    """D(x) = documented ps - clax on the total multipoles (3, Nw) at x = (f, a_perp, a_para, P_lin):
    br.predicted_total_difference on a copy of the shared inputs carrying x (AP: hratio = 1/a_para,
    Dratio = a_perp, the convention of clax_fun)."""
    f, a_perp, a_para, pk = x
    inp_x = {**inp, "f": float(f), "hratio": 1.0 / float(a_para), "Dratio": float(a_perp),
             "pk": np.asarray(pk, dtype=float)}
    parts = br.predicted_total_difference(inp_x, theta, kw, ndens=br.NDENS_PROBE)
    return parts["ctr"] + parts["closure"], parts


def test_cross_code_tangents(case):
    """ps-vs-clax-closed gate (module docstring): rel(jvp_ps - dD/dt, jvp_clax) <= br.COEF_THRESHOLDS[l]
    per tangent and l, and <= DISCRIM_RATIO x the direct rel(jvp_ps, jvp_clax) wherever the direct value
    exceeds DISCRIM_MIN. Logged per tangent and l: direct, closed, ratio, k of the worst closed point, and
    the sizes of the counterterm-map and closure-line parts of dD/dt (fraction of max|jvp_clax|); the
    primal check (ps - D vs clax at x0) is logged and printed as a guard on how D is assembled."""
    ps_fun, clax_fun, x0, tangents = _setup(case)
    inp, mask, theta = _inputs(case)
    kw = inp["k_h"][mask]

    d0, _ = _predicted_difference(inp, theta, kw, x0)
    ps0, cl0 = np.asarray(ps_fun(*x0)), np.asarray(clax_fun(*x0))
    primal = {f"l{l}": {"direct": cu.rel(ps0[i], cl0[i]), "closed": cu.rel(ps0[i] - d0[i], cl0[i])}
              for i, l in enumerate(br.ELLS)}
    print(f"{case} [primal] ps-vs-clax direct/closed % " + " | ".join(
        f"l{l} {100 * primal[f'l{l}']['direct']:.3f}/{100 * primal[f'l{l}']['closed']:.3f}" for l in br.ELLS))

    direct_errs, closed_errs, lines = {}, {}, []
    for name, t in tangents.items():
        jps = np.asarray(jax.jvp(ps_fun, x0, t)[1])
        jcl = np.asarray(jax.jvp(clax_fun, x0, t)[1])
        dp, pp = _predicted_difference(inp, theta, kw, tuple(x + FD_STEP * dx for x, dx in zip(x0, t)))
        dm, pm = _predicted_difference(inp, theta, kw, tuple(x - FD_STEP * dx for x, dx in zip(x0, t)))
        dD = (dp - dm) / (2.0 * FD_STEP)
        d_ctr = (pp["ctr"] - pm["ctr"]) / (2.0 * FD_STEP)
        d_clo = (pp["closure"] - pm["closure"]) / (2.0 * FD_STEP)
        cells = []
        for i, l in enumerate(br.ELLS):
            key, thr = f"d/d{name}:l{l}", br.COEF_THRESHOLDS[f"l{l}"]
            scale = max(float(np.max(np.abs(jcl[i]))), 1e-300)
            direct = cu.rel(jps[i], jcl[i])
            resid = jps[i] - dD[i] - jcl[i]
            closed = cu.rel(jps[i] - dD[i], jcl[i])
            k_c = float(kw[int(np.argmax(np.abs(resid)))]) if np.all(np.isfinite(resid)) else float("nan")
            k_d = float(kw[int(np.argmax(np.abs(jps[i] - jcl[i])))])
            ratio = closed / direct if direct > 0 else float("nan")
            direct_errs[key] = {"err": direct, "k": k_d}
            closed_errs[key] = {"err": closed, "k": k_c, "direct": direct, "ratio": ratio,
                                "dD_ctr": float(np.max(np.abs(d_ctr[i])) / scale),
                                "dD_closure": float(np.max(np.abs(d_clo[i])) / scale)}
            if not np.isfinite(closed) or closed > thr:
                lines.append(f"ERROR {case} ps-vs-clax-closed d/d{name} l={l}: {100 * closed:.3f}% > "
                             f"{100 * thr:.1f}% at k={k_c:.3f} (direct {100 * direct:.3f}%)")
            if direct > DISCRIM_MIN and not closed <= DISCRIM_RATIO * direct:
                lines.append(f"ERROR {case} discrimination d/d{name} l={l}: closed {100 * closed:.3f}% > "
                             f"{DISCRIM_RATIO} x direct {100 * direct:.3f}% (ratio {ratio:.3f}) at k={k_c:.3f}")
            cells.append(f"l{l} {100 * direct:.3f}/{100 * closed:.3f}% r={ratio:.2f} k={k_c:.3f}")
        print(f"{case} [ps-vs-clax-closed] d/d{name} direct/closed: " + " | ".join(cells))

    br.log_record(layer="grad", case=case, z=Z, preset="ps-vs-clax", errors=direct_errs)
    br.log_record(layer="grad", case=case, z=Z, preset="ps-vs-clax-closed", errors=closed_errs,
                  extra={"primal": primal, "fd_step": FD_STEP})
    print(f"{case} [ps-vs-clax] (not asserted) {br.worst(direct_errs)}; [ps-vs-clax-closed] {br.worst(closed_errs)}")
    assert not lines, "\n".join(lines)


def test_ps1loop_ad_matches_central_differences(case):
    ps_fun, _, x0, tangents = _setup(case)
    pairs = {}
    for name, t in tangents.items():
        ad = jax.jvp(ps_fun, x0, t)[1]
        plus = ps_fun(*(x + FD_STEP * dx for x, dx in zip(x0, t)))
        minus = ps_fun(*(x - FD_STEP * dx for x, dx in zip(x0, t)))
        pairs[name] = (ad, (plus - minus) / (2.0 * FD_STEP))
    _check(case, "ps-AD-vs-FD", pairs, {f"l{l}": FD_THRESHOLD for l in br.ELLS})
