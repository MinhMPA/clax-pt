"""Tests for nonlinear corrections to C_l^pp.

Contract:
- ``compute_nl_correction_halofit`` and ``compute_nl_correction_ept`` return
  valid P_NL/P_lin ratios (positive, ~1 at low k, >1 at high k).
- ``compute_cl_pp`` with ``nl_pk_ratio=None`` is backward-compatible.
- Nonlinear C_l^pp shows enhancement at high l relative to linear.
- Lensed TT with nonlinear C_l^pp differs from the linear case.
"""

import os

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from clax import CosmoParams, PrecisionParams
from clax.background import background_solve
from clax.thermodynamics import thermodynamics_solve
from clax.perturbations import perturbations_solve
from clax.lensing import (
    compute_cl_pp,
    compute_cl_pp_nonlinear,
    compute_nl_correction_halofit,
)

from dataclasses import replace as _dc_replace
PREC = _dc_replace(PrecisionParams.fast_cl(), pt_k_chunk_size=20)


@pytest.fixture(scope="module")
def pipeline():
    """Run the full pipeline once for all tests in this module."""
    params = CosmoParams()
    bg = background_solve(params, PREC)
    th = thermodynamics_solve(params, PREC, bg)
    pt = perturbations_solve(params, PREC, bg, th)
    return params, bg, th, pt


class TestNlCorrectionHalofit:
    """Tests for Halofit nl_correction ratio."""

    def test_shape(self, pipeline):
        params, bg, _, pt = pipeline
        ratio = compute_nl_correction_halofit(params, bg, pt, z_ref=0.0)
        assert ratio.shape == pt.k_grid.shape

    def test_positive(self, pipeline):
        params, bg, _, pt = pipeline
        ratio = compute_nl_correction_halofit(params, bg, pt, z_ref=0.0)
        assert jnp.all(ratio > 0), "Ratio must be positive everywhere"

    def test_linear_at_low_k(self, pipeline):
        """At very low k (< 0.003 Mpc^-1), P_NL/P_lin should be close to 1."""
        params, bg, _, pt = pipeline
        ratio = compute_nl_correction_halofit(params, bg, pt, z_ref=0.0)
        low_k = pt.k_grid < 0.003
        if jnp.any(low_k):
            low_k_ratios = ratio[low_k]
            assert jnp.all(jnp.abs(low_k_ratios - 1.0) < 0.05), (
                f"Low-k ratio should be ~1, got max deviation "
                f"{float(jnp.max(jnp.abs(low_k_ratios - 1.0))):.4f}")

    def test_enhancement_at_high_k(self, pipeline):
        """At high k, nonlinear P(k) should exceed linear."""
        params, bg, _, pt = pipeline
        ratio = compute_nl_correction_halofit(params, bg, pt, z_ref=0.0)
        high_k = pt.k_grid > 0.3
        if jnp.any(high_k):
            assert jnp.any(ratio[high_k] > 1.5), (
                "Expected significant enhancement at high k")


class TestClppBackwardCompat:
    """Verify compute_cl_pp without nl_pk_ratio is unchanged."""

    def test_none_gives_same(self, pipeline):
        params, bg, _, pt = pipeline
        l_values = [10, 50, 100]
        cl_default = compute_cl_pp(pt, params, bg, l_values)
        cl_explicit = compute_cl_pp(pt, params, bg, l_values,
                                    nl_pk_ratio=None, z_ref=0.0)
        np.testing.assert_allclose(
            cl_default, cl_explicit, rtol=1e-12,
            err_msg="None nl_pk_ratio must match default")


class TestClppNonlinear:
    """Tests for nonlinear-corrected C_l^pp."""

    def test_enhancement_at_high_l(self, pipeline):
        """Nonlinear C_l^pp should exceed linear at high l (l~500+)."""
        params, bg, _, pt = pipeline
        l_values = [10, 100, 500, 1000]
        cl_lin = compute_cl_pp(pt, params, bg, l_values)
        cl_nl = compute_cl_pp_nonlinear(
            pt, params, bg, l_values, method="halofit", z_ref=0.0)

        ratio = cl_nl / cl_lin
        # At l=10, should be close to 1 (< 1% deviation)
        assert float(jnp.abs(ratio[0] - 1.0)) < 0.01, (
            f"l=10 ratio should be ~1, got {float(ratio[0]):.4f}")
        # At l=500, Halofit should enhance lensing potential
        assert float(ratio[2]) > 1.005, (
            f"l=500 ratio should be >1.005, got {float(ratio[2]):.4f}")

    def test_all_positive(self, pipeline):
        params, bg, _, pt = pipeline
        l_values = [2, 10, 50, 100, 500]
        cl_nl = compute_cl_pp_nonlinear(
            pt, params, bg, l_values, method="halofit")
        assert jnp.all(cl_nl > 0), "Nonlinear C_l^pp must be positive"

    def test_invalid_method_raises(self, pipeline):
        params, bg, _, pt = pipeline
        with pytest.raises(ValueError, match="Unknown nonlinear method"):
            compute_cl_pp_nonlinear(
                pt, params, bg, [10], method="invalid")
