"""Tests for nonlinear corrections to C_l^pp.

Contract:
- ``compute_nl_correction_halofit`` returns valid P_NL/P_lin ratios
  (positive, ~1 at low k, >1 at high k).
- ``compute_cl_pp_limber(nonlinear=True)`` shows enhancement at high l
  relative to the linear case.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from clax.lensing import (
    compute_cl_pp_limber,
    compute_nl_correction_halofit,
)


class TestNlCorrectionHalofit:
    """Tests for Halofit nl_correction ratio."""

    def test_shape(self, pipeline_fast_cl_k5):
        params, _, bg, _, pt = pipeline_fast_cl_k5
        ratio = compute_nl_correction_halofit(params, bg, pt, z_ref=0.0)
        assert ratio.shape == pt.k_grid.shape

    def test_positive(self, pipeline_fast_cl_k5):
        params, _, bg, _, pt = pipeline_fast_cl_k5
        ratio = compute_nl_correction_halofit(params, bg, pt, z_ref=0.0)
        assert jnp.all(ratio > 0), "Ratio must be positive everywhere"

    def test_linear_at_low_k(self, pipeline_fast_cl_k5):
        """At very low k (< 0.003 Mpc^-1), P_NL/P_lin should be close to 1."""
        params, _, bg, _, pt = pipeline_fast_cl_k5
        ratio = compute_nl_correction_halofit(params, bg, pt, z_ref=0.0)
        low_k = pt.k_grid < 0.003
        if jnp.any(low_k):
            low_k_ratios = ratio[low_k]
            assert jnp.all(jnp.abs(low_k_ratios - 1.0) < 0.05), (
                f"Low-k ratio should be ~1, got max deviation "
                f"{float(jnp.max(jnp.abs(low_k_ratios - 1.0))):.4f}")

    def test_enhancement_at_high_k(self, pipeline_fast_cl_k5):
        """At high k, nonlinear P(k) should exceed linear."""
        params, _, bg, _, pt = pipeline_fast_cl_k5
        ratio = compute_nl_correction_halofit(params, bg, pt, z_ref=0.0)
        high_k = pt.k_grid > 0.3
        if jnp.any(high_k):
            assert jnp.any(ratio[high_k] > 1.5), (
                "Expected significant enhancement at high k")


class TestClppLimberNonlinear:
    """Tests for compute_cl_pp_limber with nonlinear=True."""

    def test_enhancement_at_high_l(self, pipeline_fast_cl_k5):
        """Nonlinear C_l^pp should exceed linear at high l."""
        params, _, bg, th, pt = pipeline_fast_cl_k5
        l_max = 500
        cl_lin = np.array(compute_cl_pp_limber(
            pt, params, bg, th, l_max=l_max, nonlinear=False))
        cl_nl = np.array(compute_cl_pp_limber(
            pt, params, bg, th, l_max=l_max, nonlinear=True))

        # Ratio at l=500 — Halofit should enhance lensing potential
        ratio_500 = cl_nl[500] / cl_lin[500]
        assert ratio_500 > 1.005, (
            f"l=500 NL/lin ratio should be >1.005, got {ratio_500:.4f}")

    def test_all_positive(self, pipeline_fast_cl_k5):
        params, _, bg, th, pt = pipeline_fast_cl_k5
        cl_nl = np.array(compute_cl_pp_limber(
            pt, params, bg, th, l_max=200, nonlinear=True))
        assert np.all(cl_nl[2:] > 0), "Nonlinear C_l^pp must be positive"

    def test_linear_matches_nonlinear_false(self, pipeline_fast_cl_k5):
        """nonlinear=False should give same result as default."""
        params, _, bg, th, pt = pipeline_fast_cl_k5
        l_max = 100
        cl_default = np.array(compute_cl_pp_limber(
            pt, params, bg, th, l_max=l_max))
        cl_explicit = np.array(compute_cl_pp_limber(
            pt, params, bg, th, l_max=l_max, nonlinear=False))
        np.testing.assert_allclose(
            cl_default, cl_explicit, rtol=1e-12,
            err_msg="nonlinear=False must match default")
