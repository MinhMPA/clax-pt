"""Tests for the public ``compute_cl_pp`` (source-Limber kernel).

The single public entry point is ``clax.lensing.compute_cl_pp(... nonlinear=...)``.
This file covers:

- Shape, sign, monotonicity contracts
- Accuracy of ``nonlinear="none"`` against an external CLASS reference
- Cross-check of ``nonlinear="none"`` against ``_compute_cl_pp_full_bessel``
  (the slow, dense-Bessel oracle kept private for tests)
- Smoke tests for ``nonlinear="halofit"`` (positivity, NL > linear at high l)
- JIT compilation and ``jax.grad`` differentiability

Accuracy of the Halofit-corrected C_l^pp against an external CLASS Halofit
reference lives in ``tests/test_clpp_halofit_ratio.py``.
"""

import os
import pytest
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from dataclasses import replace as _replace


@pytest.fixture(scope="module")
def pipeline():
    """Run pipeline once for all tests (k_max=5 so Halofit can converge)."""
    from clax import CosmoParams, PrecisionParams
    from clax.background import background_solve
    from clax.thermodynamics import thermodynamics_solve
    from clax.perturbations import perturbations_solve

    prec = _replace(PrecisionParams.fast_cl(),
                    pt_k_max_cl=5.0,
                    pt_k_chunk_size=20)
    params = CosmoParams()
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)
    pt = perturbations_solve(params, prec, bg, th)
    return params, bg, th, pt


@pytest.fixture(scope="module")
def class_reference_linear():
    """CLASS linear C_l^pp reference (matching default CosmoParams)."""
    try:
        from classy import Class
    except ImportError:
        pytest.skip("CLASS Python wrapper not available")

    cosmo = Class()
    cosmo.set({
        'A_s': 2.1e-9, 'n_s': 0.9649, 'tau_reio': 0.052,
        'omega_b': 0.02237, 'omega_cdm': 0.12, 'h': 0.6736,
        'YHe': 0.2425, 'N_ur': 2.0328, 'N_ncdm': 1, 'm_ncdm': 0.06,
        'output': 'lCl,tCl', 'lensing': 'Yes',
        'l_switch_limber': 9, 'non linear': 'none',
    })
    cosmo.compute()
    pp = cosmo.raw_cl(2500)['pp']
    cosmo.struct_cleanup()
    return pp


# -----------------------------------------------------------------------------
# Contract: shape, signature, sign
# -----------------------------------------------------------------------------

class TestContract:

    def test_import(self):
        """Public ``compute_cl_pp`` is importable from clax and clax.lensing."""
        from clax import compute_cl_pp  # noqa: F401
        from clax.lensing import compute_cl_pp  # noqa: F401

    def test_returns_correct_shape(self, pipeline):
        """Returns array of shape ``(l_max+1,)`` with l=0,1 zeroed."""
        from clax.lensing import compute_cl_pp
        params, bg, th, pt = pipeline
        cl = compute_cl_pp(pt, params, bg, th, l_max=100)
        assert cl.shape == (101,), f"expected (101,), got {cl.shape}"
        assert float(cl[0]) == 0.0
        assert float(cl[1]) == 0.0
        assert float(cl[2]) > 0.0

    def test_unknown_nonlinear_raises(self, pipeline):
        """Unknown ``nonlinear`` value is rejected with ValueError."""
        from clax.lensing import compute_cl_pp
        params, bg, th, pt = pipeline
        with pytest.raises(ValueError, match="unknown nonlinear"):
            compute_cl_pp(pt, params, bg, th, l_max=10, nonlinear="bogus")

    def test_ept_not_yet_supported(self, pipeline):
        """``nonlinear='ept'`` is not part of this PR's surface."""
        from clax.lensing import compute_cl_pp
        params, bg, th, pt = pipeline
        with pytest.raises(ValueError, match="unknown nonlinear='ept'"):
            compute_cl_pp(pt, params, bg, th, l_max=10, nonlinear="ept")


# -----------------------------------------------------------------------------
# Linear (nonlinear="none") accuracy
# -----------------------------------------------------------------------------

class TestLinearAccuracy:
    """Accuracy of the default (linear) source-Limber kernel."""

    def test_matches_class_at_low_l(self, pipeline, class_reference_linear):
        """Matches CLASS to <3% for l in {100, 200, 500}."""
        from clax.lensing import compute_cl_pp
        params, bg, th, pt = pipeline
        cl = np.array(compute_cl_pp(pt, params, bg, th, l_max=500))

        for l in [100, 200, 500]:
            ratio = cl[l] / class_reference_linear[l]
            err = abs(ratio - 1.0)
            print(f"  l={l}: clax/CLASS = {ratio:.4f} ({err:.2%})")
            assert err < 0.03, f"l={l}: {err:.2%} error exceeds 3%"

    def test_matches_class_at_medium_l(self, pipeline, class_reference_linear):
        """Matches CLASS to <5% for l = 1000."""
        from clax.lensing import compute_cl_pp
        params, bg, th, pt = pipeline
        cl = np.array(compute_cl_pp(pt, params, bg, th, l_max=1000))

        ratio = cl[1000] / class_reference_linear[1000]
        err = abs(ratio - 1.0)
        print(f"  l=1000: clax/CLASS = {ratio:.4f} ({err:.2%})")
        assert err < 0.05, f"l=1000: {err:.2%} error exceeds 5%"


class TestCrossImplAgreement:
    """Cross-check the public source-Limber path against the slow Bessel oracle.

    ``_compute_cl_pp_full_bessel`` uses ``clax.bessel.spherical_jl``, which
    relies on upward Bessel recurrence. That recurrence is reliable at low l
    (l <= ~100) but becomes unstable in the classically-forbidden region
    (x < 0.7l) at high l, so the oracle is NOT a trusted reference at
    l >= 200. The Limber approximation, in contrast, becomes more accurate
    as l increases — CLASS uses ``l_switch_limber`` between 9 and 40 for
    the lensing path because Limber is the correct answer there.

    This test therefore restricts the cross-check to l <= 100. At higher l,
    the linear-accuracy test against an external CLASS reference is the
    right verification.
    """

    def test_source_limber_vs_full_bessel_low_l(self, pipeline):
        """``compute_cl_pp(nonlinear="none")`` matches the full-Bessel oracle
        at low l where the oracle's upward Bessel recurrence is stable."""
        from clax.lensing import compute_cl_pp, _compute_cl_pp_full_bessel
        params, bg, th, pt = pipeline

        l_probe = jnp.array([10, 20, 50, 100], dtype=jnp.float64)

        cl_public = np.array(compute_cl_pp(pt, params, bg, th, l_max=100))
        cl_oracle = np.array(
            _compute_cl_pp_full_bessel(pt, params, bg, th, l_probe))

        tols = {10: 0.05, 20: 0.03, 50: 0.005, 100: 0.005}
        for i, l_val in enumerate([10, 20, 50, 100]):
            ratio = cl_public[l_val] / cl_oracle[i]
            err = abs(ratio - 1.0)
            print(f"  l={l_val}: source_limber/full_bessel = {ratio:.4f} ({err:.2%})")
            assert err < tols[l_val], (
                f"l={l_val}: {err:.2%} exceeds tolerance {tols[l_val]:.1%}")


# -----------------------------------------------------------------------------
# Halofit smoke tests
# -----------------------------------------------------------------------------

class TestHalofit:
    """Smoke tests for ``nonlinear='halofit'``.

    Quantitative cross-check against a CLASS Halofit reference lives in
    ``tests/test_clpp_halofit_ratio.py``.
    """

    @pytest.fixture(scope="class")
    def cl_pair(self, pipeline):
        from clax.lensing import compute_cl_pp
        params, bg, th, pt = pipeline
        cl_lin = np.array(compute_cl_pp(pt, params, bg, th, l_max=2000,
                                         nonlinear="none"))
        cl_nl = np.array(compute_cl_pp(pt, params, bg, th, l_max=2000,
                                        nonlinear="halofit"))
        return cl_lin, cl_nl

    def test_positive(self, cl_pair):
        """Halofit C_l^pp is positive."""
        _, cl_nl = cl_pair
        assert np.all(cl_nl[2:] > 0), "C_l^pp Halofit must be positive for l>=2"

    def test_finite(self, cl_pair):
        """Halofit C_l^pp is finite."""
        _, cl_nl = cl_pair
        assert np.all(np.isfinite(cl_nl)), "C_l^pp Halofit has non-finite entries"

    def test_nl_boost_at_high_l(self, cl_pair):
        """NL/linear ratio is >1 at l>=500 (nonlinear power exceeds linear)."""
        cl_lin, cl_nl = cl_pair
        for l_val in [500, 1000, 1500, 2000]:
            ratio = cl_nl[l_val] / cl_lin[l_val]
            print(f"  l={l_val}: NL/lin = {ratio:.4f}")
            assert ratio > 1.005, (
                f"l={l_val}: NL/lin={ratio:.4f}, expected > 1.005")

    def test_no_boost_at_low_l(self, cl_pair):
        """NL/linear ratio is ~1 at l<=50 (Halofit irrelevant on largest scales)."""
        cl_lin, cl_nl = cl_pair
        for l_val in [10, 30, 50]:
            ratio = cl_nl[l_val] / cl_lin[l_val]
            print(f"  l={l_val}: NL/lin = {ratio:.4f}")
            assert abs(ratio - 1.0) < 0.02, (
                f"l={l_val}: NL/lin={ratio:.4f}, expected ~1 at low l")


# -----------------------------------------------------------------------------
# JAX compatibility (linear path only — Halofit's vmap-over-z is heavier)
# -----------------------------------------------------------------------------

class TestJaxCompat:

    def test_jit_compatible(self, pipeline):
        """Function compiles under ``jax.jit``."""
        from clax.lensing import compute_cl_pp
        params, bg, th, pt = pipeline
        cl_jit = jax.jit(
            compute_cl_pp, static_argnums=(4,), static_argnames=("nonlinear",)
        )(pt, params, bg, th, 50)
        assert cl_jit.shape == (51,)
        assert float(cl_jit[2]) > 0

    def test_grad_wrt_ln10As(self, pipeline):
        """``jax.grad`` through ``ln10A_s`` gives a finite, nonzero gradient."""
        from clax.lensing import compute_cl_pp
        _, bg, th, pt = pipeline

        def objective(params):
            cl = compute_cl_pp(pt, params, bg, th, l_max=30)
            return jnp.sum(cl[2:])

        from clax import CosmoParams
        params = CosmoParams()
        grad = jax.grad(objective)(params)
        g_As = grad.ln10A_s
        print(f"  d(sum Cl)/d(ln10As) = {g_As:.6e}")
        assert jnp.isfinite(g_As), f"gradient is not finite: {g_As}"
        assert abs(g_As) > 0, "gradient is zero"
