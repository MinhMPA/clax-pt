"""Linear P(k) at z=0 through the two tau(z) lookups (smsharma/clax#42).

The bug: the C_l solve stops at 0.999*tau0 and the spline clamped, so a z=0
lookup read z~0.003 -- -0.33% in P, flat in k. It is checked as the GROWTH
RATIO P(z=0)/P(z=0.5) against CLASS's: k-dependent clax-vs-CLASS P(k) error
cancels in the ratio, leaving the growth over the last stretch -- exactly
what the cutoff removed. Predicted residual: -0.333% before, ~0 after. Bar
0.1% (with the fix applied, the measured residual is < 0.05%).

Multi-cosmology RULE: lcdm_cosmology (5 points, all with CLASS pk.npz).

Solves are memoised per cosmology (``_bg_th``, ``_cl``, ``_mpk``): a matter
solve at PK_TABLE_SOLVE_PREC is ~25 min and a C_l solve ~35 min on 12 CPU
cores, and several tests share each. The omega_cdm growth-gradient test
re-solves at every finite-difference step, so it is not part of the memoised
set.
"""
import dataclasses

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from clax.background import background_solve
from clax.ept import ept_inputs_from_clax, ept_kgrid
from clax.perturbations import perturbations_solve, perturbations_solve_mpk
from clax.thermodynamics import thermodynamics_solve
from clax.transfer import compute_linear_matter_pk_from_perturbations
from tests.conftest import cosmology_reference_dir
from tests.pk_test_utils import PK_GRAD_PARAM_STEPS, PK_TABLE_SOLVE_PREC

K_PROBE = np.geomspace(2.0e-4, 0.9, 24)            # Mpc^-1
GROWTH_BAR = 1.0e-3

_MEMO = {}


def _memo(tag, params, build):
    """Built on first use per (tag, cosmology), then reused. repr(params)
    lists every CosmoParams field, so it identifies the cosmology."""
    key = (tag, repr(params))
    if key not in _MEMO:
        _MEMO[key] = build()
    return _MEMO[key]


def _bg_th(params):
    """(background, thermodynamics) at PK_TABLE_SOLVE_PREC."""
    def build():
        bg = background_solve(params, PK_TABLE_SOLVE_PREC)
        return bg, thermodynamics_solve(params, PK_TABLE_SOLVE_PREC, bg)
    return _memo("bg_th", params, build)


def _cl(params):
    """The C_l solve (perturbations_solve): its tau grid ends at 0.999*tau0."""
    def build():
        bg, th = _bg_th(params)
        return perturbations_solve(params, PK_TABLE_SOLVE_PREC, bg, th)
    return _memo("cl", params, build)


def _mpk(params):
    """The matter solve (perturbations_solve_mpk): its tau grid reaches tau0."""
    def build():
        bg, th = _bg_th(params)
        return perturbations_solve_mpk(params, PK_TABLE_SOLVE_PREC, bg, th)
    return _memo("mpk", params, build)


def _class_growth_ratio(name, k):
    ref = np.load(f"{cosmology_reference_dir(name)}/pk.npz")

    def at(key):
        return np.exp(np.interp(np.log(k), np.log(ref["k"]), np.log(ref[key])))

    return at("pk_m_lin_z0") / at("pk_m_z0.5")


def _ratio_residual_transfer(pt, bg, params, name):
    p0 = np.asarray(compute_linear_matter_pk_from_perturbations(pt, bg, params, K_PROBE, z=0.0))
    p5 = np.asarray(compute_linear_matter_pk_from_perturbations(pt, bg, params, K_PROBE, z=0.5))
    return float(np.median((p0 / p5) / _class_growth_ratio(name, K_PROBE) - 1.0))


def _ratio_residual_ept(pt, bg, params, name):
    kh = np.asarray(ept_kgrid())
    k_mpc = kh * float(params.h)
    sel = (k_mpc >= K_PROBE[0]) & (k_mpc <= K_PROBE[-1])
    p0, _ = ept_inputs_from_clax(params, bg, pt, 0.0, field="m")
    p5, _ = ept_inputs_from_clax(params, bg, pt, 0.5, field="m")
    r_clax = (np.asarray(p0) / np.asarray(p5))[sel]
    return float(np.median(r_clax / _class_growth_ratio(name, k_mpc[sel]) - 1.0))


@pytest.mark.slow
def test_growth_to_today_matches_class_on_matter_solve(lcdm_cosmology):
    """GREEN: the tau0-complete matter solve, through both lookups."""
    name, params = lcdm_cosmology
    bg, _ = _bg_th(params)
    pt = _mpk(params)
    r_t = _ratio_residual_transfer(pt, bg, params, name)
    r_e = _ratio_residual_ept(pt, bg, params, name)
    print(f"{name}: growth residual transfer {r_t:+.4%}  EPT {r_e:+.4%}")
    assert abs(r_t) < GROWTH_BAR and abs(r_e) < GROWTH_BAR, (
        f"{name}: growth z=0.5->0 vs CLASS, median residual transfer {r_t:+.4%}, "
        f"EPT {r_e:+.4%} (bar {GROWTH_BAR:.1%}; the bug gives -0.333%)")


@pytest.mark.slow
def test_cl_result_raises_at_z0_in_both_lookups(lcdm_cosmology):
    """The C_l solve's grid ends at 0.999*tau0, so asking it for z=0 must RAISE
    (before this change it silently returned -0.33%)."""
    name, params = lcdm_cosmology
    bg, _ = _bg_th(params)
    pt = _cl(params)
    with pytest.raises(Exception, match="lies outside the perturbation tau_grid"):
        jax.block_until_ready(compute_linear_matter_pk_from_perturbations(pt, bg, params, K_PROBE, z=0.0))
    with pytest.raises(Exception, match="lies outside the perturbation tau_grid"):
        jax.block_until_ready(ept_inputs_from_clax(params, bg, pt, 0.0, field="cb")[0])


@pytest.mark.slow
def test_guard_fires_just_below_today_on_cl_result():
    """0 < z < 0.0032 is still beyond the C_l grid's end. Fiducial only: this
    probes the guard's position, not cosmology (exempt from the grid RULE)."""
    from clax import CosmoParams
    params = CosmoParams()
    bg, _ = _bg_th(params)
    pt_cl = _cl(params)
    pt_m = _mpk(params)
    with pytest.raises(Exception, match="lies outside"):
        jax.block_until_ready(compute_linear_matter_pk_from_perturbations(pt_cl, bg, params, K_PROBE, z=0.001))
    jax.block_until_ready(compute_linear_matter_pk_from_perturbations(pt_m, bg, params, K_PROBE, z=0.001))


@pytest.mark.slow
def test_cl_result_still_works_inside_its_grid(lcdm_cosmology):
    """z=0.5 lies inside the C_l grid: no raise."""
    name, params = lcdm_cosmology
    bg, _ = _bg_th(params)
    pt = _cl(params)
    p5 = compute_linear_matter_pk_from_perturbations(pt, bg, params, K_PROBE, z=0.5)
    assert np.all(np.isfinite(np.asarray(p5))), name


@pytest.mark.slow
def test_omega_cdm_growth_gradient_through_matter_solve(lcdm_cosmology):
    """AD vs central FD (< 1%) and jvp vs grad (< 1e-5) for d/d(omega_cdm) of the
    late-time growth g = ln f(z=0) - ln f(z=0.3), through the tau0-complete
    matter solve. f(z) = sum of P_cb(k_h, z) over 0.01 < k_h <= 0.3 h/Mpc on
    ept_kgrid(); both redshifts come from ONE solve per omega_cdm.

    What g measures, and why it is the right observable here: g is how fast the
    windowed P_cb grows over the last ~3.5 Gyr (z=0.3 -> 0) as omega_cdm moves,
    dg/d omega_cdm ~ 1.07 (d ln f/d omega_cdm = +0.17 (AD) at z=0 and -0.91
    (AD) at z=0.3). The ratio isolates late-time growth on the tau0-complete
    matter solve, which is what this branch changed: the z=0 lookups now read
    a solve that reaches tau0. Whatever fixes the amplitude and
    BAO shape of P(k) before z=0.3 enters both f(0) and f(0.3) and cancels in
    the ratio.

    Why not the absolute f (the absolute-f version of this test, not
    committed, failed this 1% bar): AD overstates d f/d omega_cdm by ~3.3% of
    f per unit omega_cdm at z=0 and at z=0.3, i.e. d ln P/d ln omega_cdm is
    biased by ~0.004. A diagnosis spike (jobs 21734/21735/21792/21736; probes
    21567/21568/21652/21653/21691/21697) measured, at the fiducial, AD - FD =
    +2.46e4 at z=0 (f = 7.45e5) and +1.78e4 at z=0.3 (f = 5.42e5). The
    windowed sum has a nearly cancelling log-response (d ln f/d ln omega_cdm
    = 0.016 (FD)), so the same absolute error reads +25% at the fiducial. Its
    decomposition: ~65% is the stop_gradient on the Thomson-rate splines
    (thermodynamics.py:807-840, commit d127731), which gives the scattering
    rate at fixed a zero omega_cdm derivative although x_e(a) in the
    recombination tail tracks H(a); ~9% is the ncdm fluid approximation
    (marginal: 2.2e3 +- 0.7e3, from a separate run assumed additive); ~26% is
    a massive-neutrino residual not yet identified. The error is set before
    z=0.3 (the same fraction of f at z=0 and z=0.3; the ~65% thermodynamics
    part acts at recombination, the ~26% ncdm residual is unidentified) and
    it predates this branch: upstream e894567 on its own C_l solve gives
    +1.7707e4 at z=0.3 against this branch's +1.7808e4 (job 21736). In g it
    nearly cancels: the unrounded gap/f is 3.303% at z=0 and 3.283% at z=0.3,
    a difference of ~2.0e-4 per unit omega_cdm against dg/d omega_cdm ~ 1.07,
    and the measured AD/FD - 1 is +2.08e-4 at the fiducial. So a 1% bar here
    checks what the branch changed. The absolute-P AD error is a known
    pre-existing issue, not tested: tracked upstream: smsharma/clax#30 and
    smsharma/clax#45.

    FD step PK_GRAD_PARAM_STEPS["omega_cdm"] = 2e-5, as in the absolute-f
    version. Convergence note from that version (FD of f(z=0); the 2e-5
    values are from jobs 21567/21568, the 1e-4 and 5e-4 values from jobs
    21652/21653), steps 2e-5 / 1e-4 / 5e-4: omega_cdm_low 9.0324e5 / 9.0325e5
    / 9.0306e5, fiducial 1.0007e5 / 1.0130e5 / 1.0091e5, so the absolute-f
    AD-vs-FD gap is not solver noise. The z=0 spread alone bounds the FD
    scatter at ~1.65e-3 in d ln f/d omega_cdm, ~0.15% of dg/d omega_cdm (the
    z=0.3 FD was not scanned).

    Deliberately omega_cdm, not h: the h channel carries a separate AD-vs-FD
    spread of its own, documented in the OBSERVED AD SPREAD section of
    test_grad_h_end_to_end_from_cosmoparams_matches_fd in
    tests/test_ept_gradients.py. Forward mode: jax.jvp cannot
    cross the custom_vjp adjoint that PK_TABLE_SOLVE_PREC selects
    (ode_adjoint="recursive_checkpoint"; see tests/test_pk_forward_mode.py),
    so the jvp uses ode_adjoint="direct". The replace(...) also sets
    th_grad_mode="native", but that flag is read only in
    solve_background_and_thermo, which this test does not call (it calls
    background_solve and thermodynamics_solve directly), so it is inert here.

    Cost: ~5 h CPU per cosmology on 12 cores (reverse AD ~2.7 h, FD ~0.8 h for
    two solves, jvp ~1.3 h; timings from job 21567). Slow: runs in the
    full-mode job gate on the full grid."""
    name, params = lcdm_cosmology
    prec = PK_TABLE_SOLVE_PREC
    prec_jvp = dataclasses.replace(prec, ode_adjoint="direct", th_grad_mode="native")
    kh = np.asarray(ept_kgrid())
    w = jnp.asarray((kh > 0.01) & (kh <= 0.3))

    def make_g(prec):
        def g(om):
            p = params.replace(omega_cdm=om)
            bg = background_solve(p, prec)
            th = thermodynamics_solve(p, prec, bg)
            pt = perturbations_solve_mpk(p, prec, bg, th)

            def f(z):
                pk, _ = ept_inputs_from_clax(p, bg, pt, z, field="cb")
                return jnp.sum(jnp.where(w, pk, 0.0))

            return jnp.log(f(0.0)) - jnp.log(f(0.3))
        return g

    g = make_g(prec)
    x0 = float(params.omega_cdm)
    step = PK_GRAD_PARAM_STEPS["omega_cdm"]
    g0, g_ad = jax.value_and_grad(g)(x0)
    g_ad = float(g_ad)
    g_fd = (float(g(x0 + step)) - float(g(x0 - step))) / (2 * step)
    _, g_jvp = jax.jvp(make_g(prec_jvp), (x0,), (1.0,))
    g_jvp = float(g_jvp)
    rel = g_ad / g_fd - 1.0
    rel_jvp = g_jvp / g_ad - 1.0
    print(f"{name}: g {float(g0):.6e}  AD {g_ad:.6e}  FD {g_fd:.6e}  AD/FD-1 {rel:+.3e}  "
          f"jvp {g_jvp:.6e}  jvp/AD-1 {rel_jvp:+.3e}")
    assert abs(rel) < 1e-2, f"{name}: AD {g_ad:.6e} vs FD {g_fd:.6e} (rel {rel:+.3e})"
    assert abs(rel_jvp) < 1e-5, f"{name}: jvp {g_jvp:.6e} vs grad {g_ad:.6e} (rel {rel_jvp:+.3e})"
