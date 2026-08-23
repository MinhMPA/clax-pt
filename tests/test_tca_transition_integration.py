"""Integration-level regression test for the TCA hard-switch bug.

``tests/test_tca_transition.py`` unit-tests ``_tca_blend`` in isolation;
``tests/test_divergence_guard.py`` unit-tests the divergence guard in
isolation. Neither would fail if a future edit reverted any of the 9
``_tca_blend(...)`` call sites inside ``_perturbation_rhs`` back to
``jnp.where(is_tca > 0.5, ...)`` — the actual bug this PR fixes. The only
existing coverage for that is ``tests/test_multipoint.py::TestMassiveNu::
test_pk_at_k005``, which is ``@pytest.mark.slow`` (a full ODE solve to
``z=0``) and was not touched by this PR.

This test closes that gap CHEAPLY: it calls the real ``_perturbation_rhs``
(the actual production RHS, not a solve) at a fixed, physically-realistic
probe state ``y0``, sampled densely across the exact TCA crossover for
``k=0.05, m_ncdm=0.15`` (tau ~ 111 Mpc, matching the bug report's own
"tau~111 Mpc / z~3461" almost exactly). Since ``is_tca`` depends only on
``tau`` (via background/thermo splines), not on ``y``, holding ``y`` fixed
still exercises every ``_tca_blend`` call site's is_tca-dependence through
its full [0, 1] sweep — a hard ``jnp.where`` reintroduced at any of the 9
sites shows up as an O(1) jump in the RHS output right at the crossover
tau, easily distinguished from the smooth background curvature.

Verified this test actually detects a revert: monkeypatching
``clax.perturbations._tca_blend`` back to the original
``jnp.where(is_tca > 0.5, tca_val, full_val)`` and rerunning this exact
probe gives a max jump of ~0.0396 at tau ~ 111.15 Mpc (right at the
crossover) vs. ~7e-5 with the real fixed ``_tca_blend`` (from smooth
background curvature, not a discontinuity).
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from clax import CosmoParams, PrecisionParams
from clax.background import background_solve
from clax.thermodynamics import thermodynamics_solve
from clax.perturbations import (
    _perturbation_solve_setup,
    _adiabatic_ic,
    _perturbation_rhs,
    _compute_tca_criterion,
)

# Small l_max / ncdm quadrature: this test only needs the RHS to be
# evaluable and to route every TCA switch site through its is_tca-dependent
# branch — the crossover tau (set purely by background/thermo splines via
# kappa_dot, a_prime_over_a, k) is essentially independent of hierarchy
# truncation, so we don't need production-grade l_max here.
_PREC = PrecisionParams(
    bg_n_points=200, ncdm_bg_n_points=100, bg_tol=1e-8,
    th_n_points=5000, th_z_max=5e3,
    pt_l_max_g=6, pt_l_max_pol_g=6, pt_l_max_ur=6, pt_k_max_cl=0.3,
    pt_ode_rtol=1e-3, pt_ode_atol=1e-6, ode_max_steps=8000,
    ncdm_fluid_approximation="none", ncdm_q_size=5,
)
_K = 0.05


@pytest.fixture(scope="module")
def bg_th():
    params = CosmoParams(m_ncdm=0.15)
    bg = background_solve(params, _PREC)
    th = thermodynamics_solve(params, _PREC, bg)
    return params, bg, th


@pytest.fixture(scope="module")
def probe(bg_th):
    """Real _perturbation_rhs, real bg/th, fixed adiabatic-IC y, and the
    actual TCA crossover tau located by bisection on the real is_tca."""
    params, bg, th = bg_th
    (idx, n_eq, k_grid, tau_grid, tau_ini, n_tau, tau_max,
     ncdmfa_mode_code, ncdmfa_trigger, args_ncdm,
     l_max_g, l_max_pol, l_max_ur, l_max_ncdm) = _perturbation_solve_setup(
        params, _PREC, bg, th, tau_max_factor=1.0)
    q_ncdm, w_ncdm, M_ncdm, dlnf0_ncdm = args_ncdm

    tau_ini_k = jnp.minimum(jnp.array(0.5), 0.01 / _K)
    y0 = _adiabatic_ic(_K, tau_ini_k, bg, params, idx, n_eq, args_ncdm=args_ncdm)
    ode_args = (_K, bg, th, params, idx, l_max_g, l_max_pol, l_max_ur,
                ncdmfa_mode_code, ncdmfa_trigger, q_ncdm, w_ncdm, M_ncdm, dlnf0_ncdm)

    def is_tca_at_tau(tau):
        loga = bg.loga_of_tau.evaluate(tau)
        a = jnp.exp(loga)
        aH = a * bg.H_of_loga.evaluate(loga)
        kappa_dot = th.kappa_dot_of_loga.evaluate(loga)
        is_tca_val, _ = _compute_tca_criterion(kappa_dot, aH, _K)
        return is_tca_val

    is_tca_sweep = jax.jit(jax.vmap(is_tca_at_tau))
    taus = jnp.linspace(1.0, 2000.0, 20000)
    vals = np.asarray(is_tca_sweep(taus))
    tau_cross = float(taus[np.argmin(np.abs(vals - 0.5))])

    rhs_vmap = jax.jit(jax.vmap(lambda t: _perturbation_rhs(t, y0, ode_args)))
    return rhs_vmap, tau_cross


def test_crossover_is_near_the_reported_bug_location(probe):
    """Sanity check: this probe's crossover tau must land near the bug
    report's "tau~111 Mpc" for k=0.05, m_ncdm=0.15 -- otherwise the test
    below isn't actually probing the bug site."""
    _, tau_cross = probe
    assert 90.0 < tau_cross < 140.0, (
        f"TCA crossover at tau={tau_cross:.2f} Mpc, expected near 111 Mpc "
        "(bug report location) for k=0.05, m_ncdm=0.15 -- probe setup may "
        "no longer match the bug repro"
    )


def test_rhs_has_no_hard_switch_jump_at_tca_crossover(probe):
    """THE regression test: evaluate the real _perturbation_rhs on a tight
    window straddling the real TCA crossover tau, with is_tca sweeping
    through its full [0, 1] range there. A max adjacent-sample jump above
    the tolerance means a hard jnp.where switch has been reintroduced at
    (at least) one of the 9 _tca_blend call sites -- exactly reproducing
    the discontinuity that caused the original solver stall.

    Tolerance justification: with the real fix, the measured max jump in
    this window is ~7e-5 (pure background-curvature smoothness, shrinks
    with window width); with the switch reverted to jnp.where, it is
    ~0.0396 (window-width-independent, a true step). 1e-3 sits comfortably
    between the two, with >10x margin on both sides.
    """
    rhs_vmap, tau_cross = probe
    window = jnp.linspace(tau_cross - 2.0, tau_cross + 2.0, 4001)
    outs = np.asarray(rhs_vmap(window))
    assert np.all(np.isfinite(outs)), "non-finite RHS output in TCA crossover window"
    diffs = np.abs(np.diff(outs, axis=0))
    max_jump = float(diffs.max())
    assert max_jump < 1e-3, (
        f"RHS jump of {max_jump:.4g} across the TCA crossover window at "
        f"tau~{tau_cross:.2f} Mpc — a hard jnp.where switch has likely been "
        "reintroduced at one of the 9 _tca_blend call sites in "
        "_perturbation_rhs (see clax/perturbations.py)"
    )
