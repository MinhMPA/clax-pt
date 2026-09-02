"""Multi-cosmology gradient-consistency tests (the multi-cosmology RULE's
reference implementation).

PROJECT RULE (CLAUDE.md, "Test at many parameter points"): physics-facing
value/gradient tests run at 3-5 cosmologies from tests/conftest.py's
COSMOLOGY_GRID_LCDM (or COSMOLOGY_GRID_NULCDM for massive-neutrino-related
changes), via the ``lcdm_cosmology`` / ``nulcdm_cosmology`` fixtures.

This file is the exemplar: it cross-validates reverse-mode AD (the
th_grad_mode="stable" custom-vjp path from PR #33) against forward-mode AD
(native path + direct adjoint) through background+thermodynamics at every
LCDM grid point. This is exactly the check that, run only at fiducial,
missed issue #30's cancellation for weeks.

No reference data needed (consistency, not accuracy). Under --fast the
fixture prunes to fiducial only; the full sweep runs in full mode (GPU
validation jobs / full suite).
"""
import jax
import jax.numpy as jnp
import pytest
from dataclasses import replace as dc_replace

from clax import PrecisionParams
from clax.thermodynamics import solve_background_and_thermo

# NOTE (measured, GPU job 14153): these tests MUST route through the fused
# solve_background_and_thermo entry point. A first version called
# background_solve + thermodynamics_solve directly -- which PR #33
# deliberately leaves on the NATIVE reverse path -- and reproduced the
# issue #30 catastrophic-cancellation garbage across the grid: reverse-vs-
# forward rel errors of 1.9e4 (fiducial), 2.0e11 (omega_b_high), 8.9e3
# (omega_cdm_low)... while h_high alone looked clean at 8.5e-10. That run
# is itself the best argument for this rule: a single-cosmology test at
# h_high would have certified a broken gradient path.

_PREC = PrecisionParams.fast_cl()
# Forward-mode arm: custom_vjp blocks jvp, and background's diffrax solve
# blocks jvp under the recursive-checkpoint adjoint -- use the documented
# escape hatches (cf. ADR 0001 and tests/test_pk_forward_mode.py).
_PREC_FWD = dc_replace(_PREC, th_grad_mode="native", ode_adjoint="direct")

# Consistency bound between the two AD modes, set from MEASURED values
# (>= 2x worst, GPU job 14155, V100, stable-path arm):
#   lcdm_fiducial 2.866e-03 | h_high 9.99e-12 | omega_b_high 3.38e-04
#   omega_cdm_low 8.25e-04  | ns_high 2.866e-03 (== fiducial: n_s does not
#   touch the thermo chain, an internal-consistency check in itself).
# OPEN QUESTION (characterized, not root-caused): the 1e-11..3e-3 spread is
# consistent with tangent-discretization differences between the stable
# backward's internal forward basis and this test's native+direct forward
# arm, amplified by this deliberately synthetic stress functional (raw
# table sums weight 20k grid points elementwise). The PHYSICAL pipeline
# agreement is 2.7e-6 (PR #33 delta_m validation) -- do not read this
# ceiling as pipeline gradient accuracy.
_AD_CONSISTENCY_RTOL = 6e-3


def _thermo_functional(prec):
    def f(h_val, base_params):
        p = base_params.replace(h=h_val)
        bg, th = solve_background_and_thermo(p, prec)
        # Scalar functional touching the recombination-era tables that
        # issue #30 implicated: quadratic so gradients weight the tables.
        return (jnp.sum(th.xe_of_loga.y ** 2)
                + jnp.sum(th.Tb_of_loga.y ** 2) / 1e6
                + th.z_star ** 2 / 1e6)

    return f


def test_thermo_grad_matches_jvp_across_cosmologies(lcdm_cosmology):
    """Reverse-mode (stable custom-vjp) == forward-mode (native+direct)
    for d(thermo functional)/dh at every LCDM grid point."""
    name, params = lcdm_cosmology
    h0 = jnp.asarray(float(params.h))

    f_rev = _thermo_functional(_PREC)
    f_fwd = _thermo_functional(_PREC_FWD)

    g = jax.grad(f_rev)(h0, params)
    _, tan = jax.jvp(lambda hv: f_fwd(hv, params), (h0,), (jnp.asarray(1.0),))

    assert jnp.isfinite(g), f"[{name}] reverse-mode gradient not finite: {g}"
    assert jnp.isfinite(tan), f"[{name}] forward-mode tangent not finite: {tan}"
    rel = float(jnp.abs(g - tan) / jnp.maximum(jnp.abs(tan), 1e-30))
    print(f"\n[{name}] grad={float(g):.8e} jvp={float(tan):.8e} rel={rel:.3e}")
    assert rel < _AD_CONSISTENCY_RTOL, (
        f"[{name}] reverse-vs-forward AD disagree: grad {float(g):.8e} vs "
        f"jvp {float(tan):.8e} (rel {rel:.2e} >= {_AD_CONSISTENCY_RTOL:g}). "
        f"A cosmology-dependent AD defect (issue #30 class)."
    )


def test_thermo_primal_finite_across_cosmologies(lcdm_cosmology):
    """The thermo chain produces finite tables at every LCDM grid point."""
    name, params = lcdm_cosmology
    bg, th = solve_background_and_thermo(params, _PREC)
    for leaf in jax.tree_util.tree_leaves(th):
        assert jnp.all(jnp.isfinite(leaf)), f"[{name}] non-finite thermo leaf"
