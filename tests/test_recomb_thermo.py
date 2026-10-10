"""S1 (smsharma/clax#48): thermodynamics tables and their forward tangents.
Primary oracle: CLASS RECFAST recfast_Heswitch=1 (reference_data/<name>/
thermo_recfast_hesw1.npz). Bars are code-accuracy gates from the spec.
Multi-cosmology: recomb_cosmology (10 points; fiducial only under --fast)."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from clax.interpolation import CubicSpline
from tests.recomb_test_utils import (QS, THETAS, THERMO_GRIDS, as_traced, load_ref,
                                     spike_ratio, tables, tangent_tables, thermo_prec,
                                     xe_on_ref_grid, z_of)

_CACHE = {}
# R12: under --fast the 3000-point (fit_cl) grid runs at fiducial; the default
# 20000-point grid and the FD tests (a jvp compile per cosmology and theta) are slow.
GRIDS = [pytest.param(n, marks=pytest.mark.slow, id=str(n)) if n > 3000 else n
         for n in THERMO_GRIDS]


def _prim_tan(name, p, n, theta="h"):
    key = (name, n, theta)
    if key not in _CACHE:
        _CACHE[key] = tangent_tables(p, thermo_prec(n), theta)
    return _CACHE[key]


def _fd_check(recomb_cosmology, theta, zs, floor=1e-8):
    """Forward tangent vs central finite differences of the tables, at the FIXED
    sample redshifts zs (smsharma/clax#48, ruling R10).

    The jvp is the tangent of the spline-evaluated values at ln a = log(1/(1+z)):
    the thermodynamics ln a knots move with theta, so the per-knot tangent table
    (`_prim_tan`) is d/dtheta at fixed knot index, not at fixed a, and is not what
    FD measures. f1, f3 are the central FDs at relative steps 1e-3 and 3e-3; Q is
    the primal value at the sample. Each (z, Q) point is in exactly one class:
      A (resolved):  |f1/Q| > floor, |f3/Q| > floor and |f1 - f3| <= 1e-4 |f1|;
                     assert |jvp/f1 - 1| <= 1e-3.
      B (at the table's rounding floor): |f1/Q| <= floor and |f3/Q| <= floor, i.e.
                     the true derivative vanishes to rounding; assert |jvp/Q| <= 1e-6,
                     so a tangent spike where the derivative vanishes still fails.
      C: everything else, skipped and counted in the printout.
    """
    name, p = recomb_cosmology
    prec = thermo_prec(20000)
    th0 = float(getattr(p, theta))
    loga_s = jnp.log(1.0 / (1.0 + jnp.asarray(zs)))

    def sampled(x):
        tab = tables(replace(as_traced(p), **{theta: x}), prec)
        return {q: CubicSpline(tab["loga"], tab[q]).evaluate(loga_s) for q in QS}

    prim, tan = jax.jit(lambda x: jax.jvp(sampled, (x,), (jnp.ones_like(x),)))(jnp.asarray(th0, jnp.float64))
    g = jax.jit(sampled)
    fds = []
    for rel_step in (1e-3, 3e-3):
        hstep = rel_step * abs(th0)
        fp = g(jnp.asarray(th0 + hstep, jnp.float64))
        fm = g(jnp.asarray(th0 - hstep, jnp.float64))
        fds.append({q: (np.asarray(fp[q]) - np.asarray(fm[q])) / (2 * hstep) for q in QS})
    n_a = n_b = n_c = 0
    worst_a, worst_b = 0.0, 0.0
    loc_b = None
    for q in QS:
        Q, jv = np.asarray(prim[q]), np.asarray(tan[q])
        f1, f3 = fds[0][q], fds[1][q]
        # R13: a NaN tangent would fall out of every max() below, so reject it first.
        assert np.all(np.isfinite(jv)) and np.all(np.isfinite(Q)), \
            f"ERROR non-finite {q} tangent or value at the sample z ({name}, {theta})"
        small1, small3 = np.abs(f1 / Q) <= floor, np.abs(f3 / Q) <= floor
        cls_a = ~small1 & ~small3 & (np.abs(f1 - f3) <= 1e-4 * np.abs(f1))
        cls_b = small1 & small3
        n_a += int(cls_a.sum())
        n_b += int(cls_b.sum())
        n_c += int((~cls_a & ~cls_b).sum())
        if cls_a.any():
            worst_a = max(worst_a, float(np.max(np.abs(jv[cls_a] / f1[cls_a] - 1.0))))
        for i in np.flatnonzero(cls_b):
            r = abs(jv[i] / Q[i])
            if r > worst_b:
                worst_b, loc_b = float(r), (q, float(zs[i]))
    print(f"{name} d/d{theta}: A {n_a} B {n_b} C {n_c} of {len(zs) * len(QS)}, "
          f"worst A |jvp/FD-1| {worst_a:.3e}, worst B |jvp/Q| {worst_b:.3e}")
    # Minimum resolved points: a third of the sample must be in class A, except
    # theta = h. At fixed z >= 400 with omega_b and omega_cdm fixed, h enters the
    # tables only through Lambda (~1e-9 of H), so d ln Q/dh is at the rounding
    # floor, class B, and the h tangent is checked there instead. A third of the
    # sample must still be classified (A or B), so an h test cannot pass with
    # every point skipped in class C.
    n_min = len(zs) * len(QS) // 3
    min_a = 0 if theta == "h" else n_min
    assert n_a >= min_a, f"ERROR resolved FD points {n_a} < {min_a} ({name}, {theta})"
    assert n_a + n_b >= n_min, f"ERROR classified FD points {n_a + n_b} < {n_min} ({name}, {theta})"
    assert worst_a <= 1e-3, f"ERROR tangent vs FD {worst_a:.3e} > 1e-3 ({name}, {theta})"
    assert worst_b <= 1e-6, (f"ERROR tangent where the derivative vanishes: |jvp/Q| {worst_b:.3e} > 1e-6 "
                             f"at {loc_b} ({name}, {theta})")


@pytest.mark.parametrize("n", GRIDS)
def test_xe_helium_window_matches_class(recomb_cosmology, n):
    name, p = recomb_cosmology
    prim, _ = _prim_tan(name, p, n)
    ref = load_ref(name)
    sel = (ref["z"] >= 2870.0) & (ref["z"] <= 5000.0)
    rel = np.abs(xe_on_ref_grid(prim, ref["z"][sel]) / ref["x_e"][sel] - 1.0)
    i = int(np.argmax(rel))
    print(f"{name} n={n}: x_e vs CLASS z 2870-5000 max {rel[i]:.3e} at z={ref['z'][sel][i]:.1f}")
    assert rel[i] <= 1e-3, f"ERROR x_e helium window {rel[i]:.3e} > 1e-3 ({name}, n={n})"


@pytest.mark.parametrize("n", GRIDS)
@pytest.mark.parametrize("theta", THETAS)
def test_no_derivative_spikes_helium_window(recomb_cosmology, n, theta):
    name, p = recomb_cosmology
    prim, tan = _prim_tan(name, p, n, theta)
    for q in QS:      # R13: max() below would silently drop a NaN ratio
        assert np.all(np.isfinite(tan[q])) and np.all(np.isfinite(prim[q])), \
            f"ERROR non-finite {q} table or tangent ({name}, {theta}, n={n})"
    z = z_of(prim["loga"])
    worst = max(spike_ratio(tan[q] / prim[q], z, 1800.0, 6000.0) + (q,) for q in QS)
    print(f"{name} n={n} d/d{theta}: worst helium-window spike ratio {worst[0]:.3g} ({worst[2]} at z={worst[1]:.1f})")
    assert worst[0] < 10.0, f"ERROR spike ratio {worst[0]:.3g} ({name}, {theta}, n={n})"


@pytest.mark.parametrize("n", GRIDS)
def test_values_smooth_helium_window(recomb_cosmology, n):
    name, p = recomb_cosmology
    prim, _ = _prim_tan(name, p, n)
    z = z_of(prim["loga"])
    sel = (z >= 1800.0) & (z <= 6000.0)
    d2 = np.abs(np.diff(prim["xe"], 2))[sel[1:-1]]
    bar = 3e-5 * (20000.0 / n) ** 2      # second differences scale as spacing^2 (spec S1)
    print(f"{name} n={n}: max |d2 x_e| over z 1800-6000 = {d2.max():.3e} (bar {bar:.2e})")
    assert d2.max() < bar, f"ERROR x_e not smooth: {d2.max():.3e} ({name}, n={n})"


ZS_HE = np.array([5500., 4500., 3800., 3400., 3200., 3000., 2800., 2600., 2400., 2200., 2000.])


@pytest.mark.slow
@pytest.mark.parametrize("theta", THETAS)
def test_tangents_match_converged_fd_helium(recomb_cosmology, theta):
    _fd_check(recomb_cosmology, theta, ZS_HE)
