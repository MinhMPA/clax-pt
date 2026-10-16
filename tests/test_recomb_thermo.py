"""S1 (smsharma/clax#48): thermodynamics tables and their forward tangents.
Primary oracle: CLASS RECFAST recfast_Heswitch=1 (reference_data/<name>/
thermo_recfast_hesw1.npz). Bars are the code-accuracy gates set for #48.
Multi-cosmology: recomb_cosmology (10 points; fiducial only under --fast)."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from clax.background import background_solve
from clax.interpolation import CubicSpline
from clax.thermodynamics import _f_He, thermodynamics_solve
from tests.recomb_test_utils import (QS, THETAS, THERMO_GRIDS, as_traced, load_ref,
                                     spike_ratio, tables, tangent_tables, thermo_prec,
                                     xe_on_ref_grid, z_of)

_CACHE = {}
# Under --fast the 3000-point (fit_cl) grid runs at fiducial; the default 20000-point
# grid and the FD tests (a jvp compile per cosmology and theta) are marked slow.
GRIDS = [pytest.param(n, marks=pytest.mark.slow, id=str(n)) if n > 3000 else n
         for n in THERMO_GRIDS]


def _prim_tan(name, p, n, theta="h"):
    key = (name, n, theta)
    if key not in _CACHE:
        _CACHE[key] = tangent_tables(p, thermo_prec(n), theta)
    return _CACHE[key]


def _fd_check(recomb_cosmology, theta, zs, floor=1e-8):
    """Forward tangent vs central finite differences of the tables, at the FIXED
    sample redshifts zs (smsharma/clax#48).

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
        # Reject non-finite values first: Python max() drops NaN, so a NaN tangent would pass.
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
    for q in QS:      # non-finite values must fail: max() below would silently drop a NaN ratio
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
    bar = 3e-5 * (20000.0 / n) ** 2      # second differences scale as spacing^2
    print(f"{name} n={n}: max |d2 x_e| over z 1800-6000 = {d2.max():.3e} (bar {bar:.2e})")
    assert d2.max() < bar, f"ERROR x_e not smooth: {d2.max():.3e} ({name}, n={n})"


ZS_HE = np.array([5500., 4500., 3800., 3400., 3200., 3000., 2800., 2600., 2400., 2200., 2000.])


@pytest.mark.slow
@pytest.mark.parametrize("theta", THETAS)
def test_tangents_match_converged_fd_helium(recomb_cosmology, theta):
    _fd_check(recomb_cosmology, theta, ZS_HE)


# ---- Hydrogen: CLASS staging, Saha -> Peebles ODE at z = 1600 ----

@pytest.mark.parametrize("n", GRIDS)
@pytest.mark.parametrize("window", [(1590.0, 1650.0), (800.0, 1550.0), (1650.0, 2870.0)])
def test_xe_hydrogen_windows_match_class(recomb_cosmology, n, window):
    # (1650, 2870) is the helium ODE's x_e; helium staging alone could not pass it,
    # because its error sits at the z ~ 1650 hydrogen edge, which only hydrogen staging fixes.
    name, p = recomb_cosmology
    prim, _ = _prim_tan(name, p, n)
    ref = load_ref(name)
    sel = (ref["z"] >= window[0]) & (ref["z"] <= window[1])
    rel = np.abs(xe_on_ref_grid(prim, ref["z"][sel]) / ref["x_e"][sel] - 1.0)
    i = int(np.argmax(rel))
    print(f"{name} n={n}: x_e vs CLASS z {window} max {rel[i]:.3e} at z={ref['z'][sel][i]:.1f}")
    assert rel[i] <= 1e-3, f"ERROR x_e {window} {rel[i]:.3e} > 1e-3 ({name}, n={n})"


@pytest.mark.parametrize("n", GRIDS)
def test_visibility_peak_matches_class(recomb_cosmology, n):
    from tests.recomb_test_utils import visibility_peak
    name, p = recomb_cosmology
    prim, _ = _prim_tan(name, p, n)
    ref = load_ref(name)
    dz = abs(visibility_peak(z_of(prim["loga"]), prim["g"]) - visibility_peak(ref["z"], ref["g"]))
    bar = 0.01 if n == 20000 else 0.03          # per-grid bars
    print(f"{name} n={n}: visibility-peak |dz| = {dz:.4f} (bar {bar})")
    assert dz < bar, f"ERROR visibility peak shift {dz:.4f} ({name}, n={n})"


@pytest.mark.parametrize("n", GRIDS)
@pytest.mark.parametrize("theta", THETAS)
def test_no_derivative_spikes_full_range(recomb_cosmology, n, theta):
    name, p = recomb_cosmology
    prim, tan = _prim_tan(name, p, n, theta)
    for q in QS:      # non-finite values must fail: max() below would silently drop a NaN ratio
        assert np.all(np.isfinite(tan[q])) and np.all(np.isfinite(prim[q])), \
            f"ERROR non-finite {q} table or tangent ({name}, {theta}, n={n})"
    z = z_of(prim["loga"])
    worst = max(spike_ratio(tan[q] / prim[q], z, 300.0, 6000.0) + (q,) for q in QS)
    print(f"{name} n={n} d/d{theta}: worst spike ratio z 300-6000 {worst[0]:.3g} ({worst[2]} at z={worst[1]:.1f})")
    assert worst[0] < 10.0, f"ERROR spike ratio {worst[0]:.3g} ({name}, {theta}, n={n})"


@pytest.mark.parametrize("n", GRIDS)
def test_values_smooth_full_range(recomb_cosmology, n):
    name, p = recomb_cosmology
    prim, _ = _prim_tan(name, p, n)
    z = z_of(prim["loga"])
    sel = (z >= 1500.0) & (z <= 6000.0)
    d2 = np.abs(np.diff(prim["xe"], 2))[sel[1:-1]]
    bar = 3e-5 * (20000.0 / n) ** 2
    print(f"{name} n={n}: max |d2 x_e| over z 1500-6000 = {d2.max():.3e} (bar {bar:.2e})")
    assert d2.max() < bar, f"ERROR x_e not smooth: {d2.max():.3e} ({name}, n={n})"


ZS_H = np.array([1800., 1300., 1100., 800., 400.])


@pytest.mark.slow
@pytest.mark.parametrize("theta", THETAS)
def test_tangents_match_converged_fd_hydrogen(recomb_cosmology, theta):
    _fd_check(recomb_cosmology, theta, ZS_H)


# ---- Robustness checks: bounds, T_cmb, coarser grid, jit vs eager, vmap ----

@pytest.mark.parametrize("n", GRIDS)
def test_fractions_stay_physical_without_clips(recomb_cosmology, n):
    name, p = recomb_cosmology
    prim, _ = _prim_tan(name, p, n)
    xe = np.asarray(prim["xe"])
    z = z_of(prim["loga"])
    y = float(p.Y_He)
    upper = 1.0 + 2.0 * y / (3.9715 * (1.0 - y))     # >= the exactly-4 value too
    print(f"{name} n={n}: x_e in [{xe.min():.3e}, {xe.max():.6f}], upper {upper:.6f}")
    assert xe.min() >= 0.0 and xe.max() <= upper + 1e-12, f"ERROR x_e out of bounds ({name}, n={n})"
    # For 2400 <= z <= 2870 hydrogen is fully ionised (hydrogen Saha at the
    # fiducial cosmology: 1 - x_HII ~ 2e-9 at z = 2870, ~ 7e-8 at z = 2400)
    # and HeIII is negligible, so x_e <= 1 + f_He bounds the helium fraction
    # x_HeII <= 1. ThermoResult does not expose x_HeII; the clips are gone, so
    # this is the guard against the ROS2 step overshooting. Its margin is
    # 3.6e-4 to 5.2e-4 (x_HeII ~ 0.995 at its maximum in this window, not 1), so
    # it catches an x_HeII overshoot only above about 5e-3.
    he = (z >= 2400.0) & (z <= 2870.0)
    he_upper = 1.0 + y / (3.9715 * (1.0 - y)) + 1e-10
    j = int(np.argmax(np.where(he, xe, -np.inf)))
    print(f"{name} n={n}: z 2400-2870 max x_e {xe[j]:.8f} at z={z[j]:.1f}, bound {he_upper:.8f}, "
          f"margin {he_upper - xe[j]:.3e}")
    assert xe[j] <= he_upper, (f"ERROR x_e {xe[j]:.10f} at z={z[j]:.1f} exceeds the helium bound "
                               f"{he_upper:.10f}, x_HeII > 1 ({name}, n={n})")


@pytest.mark.parametrize("n", GRIDS)
def test_helium_counted_once_z2870_5000(recomb_cosmology, n):
    """x_HeII + x_HeIII <= 1 between the HeIII -> HeII hand-over (z ~ 5000) and the
    HeI staging switch (z = 2870). ThermoResult does not expose the helium
    fractions. Hydrogen is fully ionised there (1 - x_HII < 1e-8), so
    (x_e - 1)/f_He = x_HeII + 2 x_HeIII, the electrons per helium nucleus, in clax
    (its own f_He) and in CLASS (_not4_); comparing those is independent of the
    helium-to-hydrogen ratio convention. The HeI Saha root once ignored the MB95
    HeIII still present below z ~ 5000: x_HeII + x_HeIII reached 1.0007 near
    z = 4970, and this difference was 5.7e-4 to 7.5e-4 at all 10 cosmologies on
    both grids. The bar, 1e-4, fails all of those and sits 6.6x above the
    corrected worst (1.5e-5). Above z = 5000 the MB95 HeIII stage is 2.0e-4 to 2.4e-4 from
    CLASS in x_e, a separate, pre-existing offset, so the window stops there."""
    name, p = recomb_cosmology
    prim, _ = _prim_tan(name, p, n)
    ref = load_ref(name)
    sel = (ref["z"] >= 2870.0) & (ref["z"] <= 5000.0)
    z = ref["z"][sel]
    y = float(p.Y_He)
    he_clax = (xe_on_ref_grid(prim, z) - 1.0) / float(_f_He(y))
    he_class = (ref["x_e"][sel] - 1.0) / (y / (3.9715 * (1.0 - y)))
    d = np.abs(he_clax - he_class)
    j = int(np.argmax(d))
    print(f"{name} n={n}: z 2870-5000 max |electrons per He nucleus - CLASS| = {d[j]:.3e} at z={z[j]:.0f}")
    assert d[j] < 1e-4, (f"ERROR helium counted twice: electrons per He nucleus {d[j]:.3e} "
                         f"off CLASS at z={z[j]:.0f}, bar 1e-4 ({name}, n={n})")


@pytest.mark.parametrize("n", GRIDS)
def test_no_spikes_for_T_cmb(recomb_cosmology, n):
    name, p = recomb_cosmology
    prim, tan = _prim_tan(name, p, n, "T_cmb")
    for q in QS:      # non-finite values must fail: max() below would silently drop a NaN ratio
        assert np.all(np.isfinite(tan[q])) and np.all(np.isfinite(prim[q])), \
            f"ERROR non-finite {q} table or tangent ({name}, T_cmb, n={n})"
    z = z_of(prim["loga"])
    worst = max(spike_ratio(tan[q] / prim[q], z, 300.0, 6000.0) + (q,) for q in QS)
    print(f"{name} n={n} d/dT_cmb: worst spike ratio {worst[0]:.3g} ({worst[2]} at z={worst[1]:.1f})")
    assert worst[0] < 10.0, f"ERROR T_cmb spike ratio {worst[0]:.3g} ({name}, n={n})"


@pytest.mark.slow
@pytest.mark.parametrize("theta", THETAS)
def test_coarse_grid_1500_helium_hydrogen_spike_free(recomb_cosmology, theta):
    name, p = recomb_cosmology
    prim, tan = _prim_tan(name, p, 1500, theta)
    for q in QS:      # non-finite values must fail: max() below would silently drop a NaN ratio
        assert np.all(np.isfinite(tan[q])) and np.all(np.isfinite(prim[q])), \
            f"ERROR non-finite {q} table or tangent ({name}, {theta}, n=1500)"
    z = z_of(prim["loga"])
    worst = max(spike_ratio(tan[q] / prim[q], z, 800.0, 6000.0) + (q,) for q in QS)
    print(f"{name} n=1500 d/d{theta}: worst spike ratio z 800-6000 {worst[0]:.3g}")
    assert worst[0] < 10.0, f"ERROR 1500-grid spike ratio {worst[0]:.3g} ({name}, {theta})"


@pytest.mark.slow
def test_jit_and_eager_primal_agree(recomb_cosmology):
    name, p = recomb_cosmology
    prec = thermo_prec(20000)
    pt = as_traced(p)
    # thermodynamics_solve is itself @jax.jit, so tables() always runs it compiled;
    # __wrapped__ is the uncompiled function.
    # 20000 points: the jit/eager rounding-level gap of the former RK4 code showed up on this
    # grid (measured during the design study, not in the repo); 3000 points does not amplify it.
    a = np.asarray(thermodynamics_solve.__wrapped__(pt, prec, background_solve(pt, prec)).xe_of_loga.y)
    b = np.asarray(jax.jit(lambda q: tables(q, prec))(pt)["xe"])
    d = float(np.max(np.abs(a - b)))
    print(f"{name}: max |x_e(jit) - x_e(eager)| = {d:.3e}")
    assert d < 1e-12, f"ERROR jit vs eager x_e differ by {d:.3e} ({name})"


@pytest.mark.slow
def test_vmap_over_h_matches_loop(recomb_cosmology):
    name, p = recomb_cosmology
    prec = thermo_prec(3000)
    hs = float(p.h) * jnp.array([0.98, 1.0, 1.02])
    f = jax.jit(lambda h: tables(replace(as_traced(p), h=h), prec)["xe"])
    batched = np.asarray(jax.vmap(f)(hs))
    looped = np.stack([np.asarray(f(h)) for h in hs])
    d = float(np.max(np.abs(batched - looped)))
    print(f"{name}: max |vmap - loop| = {d:.3e}")
    assert d < 1e-12, f"ERROR vmap vs loop differ by {d:.3e} ({name})"
