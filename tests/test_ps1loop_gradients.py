# tests/test_ps1loop_gradients.py
"""Derivatives (spec §5.5): forward-mode tangents of the TOTAL P_0, P_2, P_4 at the PFS bias point
(pfs.bias_point) along f, alpha_perp, alpha_para, a k-stretch t = P dlnP/dlnk and the amplitude t = P.

What must stay true, and what breaks if it does not:
  * ps-vs-clax-closed (test_cross_code_tangents, one test per tangent, pytest ids f / a_perp / a_para /
    stretch / amplitude): d(ps - clax)/dt equals dD/dt, the derivative of the
    documented model difference D(x), x = (f, alpha_perp, alpha_para, P_lin) (br.predicted_total_difference:
    counterterm / fingers-of-god / a0 maps, omitted projections, ps's IR-resummation scheme; spec §4.2
    items 1-5), at the code-accuracy thresholds 1% / 1% / 2% for l = 0 / 2 / 4. Task 5's discrimination
    rule applies: where the direct ps-vs-clax derivative difference exceeds 0.1% of the clax derivative's
    scale, the closed residual must be at most half of it, so a line cannot pass merely by being small.
    If this breaks, the two codes respond differently to f, the AP distortion or the linear spectrum in a
    way no documented model difference explains -- a one-loop or AP-remapping defect in one of them, which
    would bias Fisher forecasts and HMC gradients for f sigma_8 and the BAO/AP dilations even where the
    spectra themselves agree. USER DECISION (Oct 10, 2026) and controller ruling: the closure gate applies
    to all five tangents. Where dD/dt is negligible the closed and direct differences coincide, so such a
    line passes the discrimination rule only if its direct difference is itself below DISCRIM_MIN of the
    scale. USER RULING (Oct 12, 2026): the rule is kept on all five tangents; the stretch/amplitude lines
    fail it until the remainder is isolated -- see the SDD ledger
    (.superpowers/sdd/2026-10-05-clax-pt-vs-ps1loop-validation/progress.md) and CHANGELOG; the fast-suite
    wrapper deselects only those two parametrizations (Task 9, controller amendment E, Oct 13, 2026), and
    scripts/ps1loop_convergence_ladder.py measures the remainder per ps_1loop setting. That remainder
    is non-smooth in k, peaks at k = 0.13-0.25 h/Mpc and is 11-14x the primal remainder ps - clax - D,
    both measured over the stretch derivative's scale max|jvp_clax| (scratch job 24845). The direct
    ps-vs-clax numbers are logged and printed but not asserted: they contain dD/dt, which along the AP
    directions is large (scratch job 24167, lcdm_fiducial: alpha_para 3.1 / 10.4 / 8.7% at l = 0 / 2 / 4).
  * primal guard (test_primal_closure_guard): at x0, ps - D matches clax at the same thresholds and
    discrimination rule.
    If it breaks, D is mis-assembled (a sign, a missing monomial, a wrong bias-point value), and the
    derivative closure built on it means nothing.
  * ps-AD-vs-FD: ps_1loop's own jvp equals its central differences to FD_THRESHOLD = 1e-4 (tightened
    from the 1% CLAUDE.md gradient target by controller ruling: measured worst 1.4e-6, central-difference
    floor ~1e-8 at step 1e-4; only ever tightened); a failure is an AD defect in ps_1loop.
  * traced-IR primal: the differentiable IR split clax uses here reproduces the validated NumPy IR path
    to 1e-8, so the clax derivatives are taken at the validated spectra.

The setup, D and the closure numerics live in tests/ps1loop_bridge.py (br.tangent_setup,
br.predicted_difference_at, br.primal_closure, br.tangent_closure), shared with the convergence ladder.
D is NumPy-valued (it reads clax's mu-loop through the br.capture_gl_inputs spy), so it cannot be
jvp'd: dD/dt is the central difference of D at x0 +/- FD_STEP * t, the step ps_1loop's own FD check uses.
D is smooth in x: its clax pieces run clax's NumPy IR path, whose central differences equal the traced
path's to ~1e-11 (scratch job 24167), and its ps piece is ps's own P_IR, whose AD equals its central
differences (test_ps1loop_ad_matches_central_differences). The truncation error of the central difference
is O(FD_STEP^2) ~ 1e-8 relative, far below the 1% gate. The counterterm map contributes to D: with IR
resummation on it is not zero (Task 5 at z = 0.38, fraction of the coefficient's scale: c0 / cfog =
1.0 / 0.9% lcdm_fiducial, 1.0 / 1.8% massive_nu_030, 1.1 / 10.2% w0wa_m07_m10), and the bias point
carries c0 = c2 = c4 = 2 (Mpc/h)^2 and cfog = 500 (Mpc/h)^4; a0 = 0 there, so the a0 map adds nothing.

ps_1loop runs at nfft = PS_NFFT_GATE = 512 in every test of this file (ps model, D's counterterm part and
ps's own AD-vs-FD check alike), not at its default 256: the closure closes only once ps's FFTLog noise is
below the remainder; at ps's default 256 its own resolution spikes at k ~ 0.14 and 0.25 h/Mpc set the
maximum (finding F5; scratch jobs 24968 / 24970 / 24971).

lcdm_fiducial, massive_nu_030, w0wa_m07_m10 at z = 0.38 (AP non-trivial) in full mode; lcdm_fiducial
under --fast. Multi-cosmology rule: 3 cosmologies."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ps_1loop_jax")
import jax

from tests import ept_campaign_utils as cu
from tests import pfs_survey as pfs
from tests import ps1loop_bridge as br

GRAD_CASES = ("lcdm_fiducial", "massive_nu_030", "w0wa_m07_m10")
Z = 0.38
FD_STEP = br.FD_STEP   # 1e-4: central-difference step for dD/dt and for ps AD vs FD (bridge)
# ps AD vs its own central differences: tightened 1e-2 -> 1e-4 by controller ruling (measured worst 1.4e-6,
# central-difference floor ~1e-8 at FD_STEP = 1e-4). Never widened.
FD_THRESHOLD = 1e-4
DISCRIM_MIN, DISCRIM_RATIO = br.DISCRIM_MIN, br.DISCRIM_RATIO   # Task 5's discrimination rule (bridge)
# ps_1loop's FFTLog sampling for this gate. The closure closes only once ps's FFTLog noise is below the
# remainder; at ps's default 256 its own resolution spikes at k ~ 0.14 and 0.25 h/Mpc set the maximum
# (finding F5).
PS_NFFT_GATE = 512


def pytest_generate_tests(metafunc):
    if "case" in metafunc.fixturenames:
        fast = metafunc.config.getoption("--fast", default=False)
        metafunc.parametrize("case", ["lcdm_fiducial"] if fast else list(GRAD_CASES))


def _inputs(case):
    """(shared inputs, comparison-window mask, PFS bias point) at Z."""
    inp = br.shared_inputs(cu.require_reference(case, Z))
    assert br.has_ap(inp)
    return inp, cu.window(inp["k_h"]), pfs.bias_point(Z)


def _model(inp):
    """The ps_1loop instance of this gate: nfft = PS_NFFT_GATE, every other setting at ps's default."""
    return br.ps_model(inp["rbao"], nfft=PS_NFFT_GATE)


def _setup(case):
    """(ps_fun, clax_fun, x0, tangents) at Z with ps_1loop at nfft = PS_NFFT_GATE (br.tangent_setup)."""
    inp, mask, theta = _inputs(case)
    return br.tangent_setup(inp, theta, mask, model=_model(inp))


def _check(case, layer, pairs, thresholds, kw):
    """rel(got, ref) per tangent and l against `thresholds`; logs the k of each line's worst point and
    prints the worst line in scientific notation (the thresholds here are far below 0.1%)."""
    errs, lines = {}, []
    for name, (got, ref) in pairs.items():
        for i, l in enumerate(br.ELLS):
            g, r = np.asarray(got[i]), np.asarray(ref[i])
            e = cu.rel(g, r)
            d = np.abs(g - r)
            k = float(kw[int(np.argmax(d))]) if np.all(np.isfinite(d)) else float("nan")
            errs[f"d/d{name}:l{l}"] = {"err": e, "k": k}
            thr = thresholds[f"l{l}"]
            if not np.isfinite(e) or e > thr:
                lines.append(f"ERROR {case} {layer} d/d{name} l={l}: {e:.2e} > {thr:.0e} at k={k:.3f}")
    br.log_record(layer="grad", case=case, z=Z, preset=layer, errors=errs)
    w = max(errs, key=lambda n: np.inf if not np.isfinite(errs[n]["err"]) else errs[n]["err"])
    print(f"{case} [{layer}] worst {w} {errs[w]['err']:.2e} at k={errs[w]['k']:.3f} (threshold "
          + ", ".join(f"l{l} {thresholds[f'l{l}']:.0e}" for l in br.ELLS) + ")")
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


def test_primal_closure_guard(case):
    """Primal guard (module docstring): rel(ps - D, clax) at x0 <= br.COEF_THRESHOLDS[l], and <= DISCRIM_RATIO
    x the direct rel(ps, clax) wherever the direct value exceeds DISCRIM_MIN. It checks how D is assembled;
    the derivative closure of test_cross_code_tangents is built on the same D."""
    ps_fun, clax_fun, x0, _ = _setup(case)
    inp, mask, theta = _inputs(case)
    kw = inp["k_h"][mask]
    primal = br.primal_closure(ps_fun, clax_fun, inp, theta, kw, x0, model=_model(inp))
    print(f"{case} [primal] ps-vs-clax direct/closed % " + " | ".join(
        f"l{l} {100 * primal[f'l{l}']['direct']:.3f}/{100 * primal[f'l{l}']['closed']:.3f}" for l in br.ELLS))
    lines = []
    for l in br.ELLS:
        pr, thr = primal[f"l{l}"], br.COEF_THRESHOLDS[f"l{l}"]
        if not np.isfinite(pr["closed"]) or pr["closed"] > thr:
            lines.append(f"ERROR {case} primal ps-vs-clax-closed l={l}: {100 * pr['closed']:.3f}% > "
                         f"{100 * thr:.1f}% at k={pr['k']:.3f} (direct {100 * pr['direct']:.3f}%)")
        if pr["direct"] > DISCRIM_MIN and not pr["closed"] <= DISCRIM_RATIO * pr["direct"]:
            lines.append(f"ERROR {case} primal discrimination l={l}: closed {100 * pr['closed']:.3f}% > "
                         f"{DISCRIM_RATIO} x direct {100 * pr['direct']:.3f}% at k={pr['k']:.3f}")
    br.log_record(layer="grad", case=case, z=Z, preset="primal-guard",
                  errors={f"l{l}": {"err": primal[f"l{l}"]["closed"], "k": primal[f"l{l}"]["k"],
                                    "direct": primal[f"l{l}"]["direct"]} for l in br.ELLS})
    assert not lines, "\n".join(lines)


@pytest.mark.parametrize("tangent", br.TANGENTS)
def test_cross_code_tangents(case, tangent):
    """ps-vs-clax-closed gate (module docstring) along one tangent: rel(jvp_ps - dD/dt, jvp_clax) <=
    br.COEF_THRESHOLDS[l] per l, and <= DISCRIM_RATIO x the direct rel(jvp_ps, jvp_clax) wherever the direct
    value exceeds DISCRIM_MIN. Logged per l: direct, closed, ratio, k of the worst closed point, and the sizes
    of the counterterm-map and closure-line parts of dD/dt (fraction of max|jvp_clax|)."""
    ps_fun, clax_fun, x0, tangents = _setup(case)
    inp, mask, theta = _inputs(case)
    kw = inp["k_h"][mask]
    rec = br.tangent_closure(ps_fun, clax_fun, inp, theta, kw, x0, tangents[tangent], fd_step=FD_STEP,
                             model=_model(inp))

    direct_errs, closed_errs, lines, cells = {}, {}, [], []
    for l in br.ELLS:
        r, key, thr = rec[f"l{l}"], f"d/d{tangent}:l{l}", br.COEF_THRESHOLDS[f"l{l}"]
        direct, closed, ratio, k_c = r["direct"], r["closed"], r["ratio"], r["k"]
        direct_errs[key] = {"err": direct, "k": r["k_direct"]}
        closed_errs[key] = {"err": closed, "k": k_c, "direct": direct, "ratio": ratio,
                            "dD_ctr": r["dD_ctr"], "dD_closure": r["dD_closure"]}
        if not np.isfinite(closed) or closed > thr:
            lines.append(f"ERROR {case} ps-vs-clax-closed d/d{tangent} l={l}: {100 * closed:.3f}% > "
                         f"{100 * thr:.1f}% at k={k_c:.3f} (direct {100 * direct:.3f}%)")
        if direct > DISCRIM_MIN and not closed <= DISCRIM_RATIO * direct:
            lines.append(f"ERROR {case} discrimination d/d{tangent} l={l}: closed {100 * closed:.3f}% > "
                         f"{DISCRIM_RATIO} x direct {100 * direct:.3f}% (ratio {ratio:.3f}) at k={k_c:.3f}")
        cells.append(f"l{l} {100 * direct:.3f}/{100 * closed:.3f}% r={ratio:.2f} k={k_c:.3f}")
    print(f"{case} [ps-vs-clax-closed] d/d{tangent} direct/closed: " + " | ".join(cells))

    br.log_record(layer="grad", case=case, z=Z, preset="ps-vs-clax", errors=direct_errs)
    br.log_record(layer="grad", case=case, z=Z, preset="ps-vs-clax-closed", errors=closed_errs,
                  extra={"fd_step": FD_STEP})
    print(f"{case} [ps-vs-clax] d/d{tangent} (not asserted) {br.worst(direct_errs)}; "
          f"[ps-vs-clax-closed] {br.worst(closed_errs)}")
    assert not lines, "\n".join(lines)


def test_ps1loop_ad_matches_central_differences(case):
    ps_fun, _, x0, tangents = _setup(case)
    inp, mask, _ = _inputs(case)
    pairs = {}
    for name, t in tangents.items():
        ad = jax.jvp(ps_fun, x0, t)[1]
        plus = ps_fun(*(x + FD_STEP * dx for x, dx in zip(x0, t)))
        minus = ps_fun(*(x - FD_STEP * dx for x, dx in zip(x0, t)))
        pairs[name] = (ad, (plus - minus) / (2.0 * FD_STEP))
    _check(case, "ps-AD-vs-FD", pairs, {f"l{l}": FD_THRESHOLD for l in br.ELLS}, inp["k_h"][mask])
