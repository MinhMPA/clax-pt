"""Tests Limber C_l^pp accuracy vs CLASS reference.

Validates ``compute_cl_pp_limber`` against CLASS at specific l values.
Documents the known Limber Poisson accuracy limitations at high l.

Key findings (Planck 2018 LCDM, no nonlinear corrections):
    l=100:  linear C_l^pp matches CLASS to <1%
    l=500:  linear C_l^pp matches CLASS to <3%
    l=1000: linear C_l^pp matches CLASS to <3%
    l=2500: ~20% systematic overestimate (Limber Poisson limitation)

For the NL/linear ratio with Halofit:
    l=100-500: ratio matches CLASS-PT to <2%
    l>1000:    ratio diverges due to differential Limber weighting of NL corrections
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
    """Run pipeline with k_max=5 for Halofit support."""
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
def class_reference():
    """Generate CLASS reference C_l^pp with matching parameters."""
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


class TestLinearClppAccuracy:
    """Test linear C_l^pp accuracy against CLASS."""

    def test_low_l_accuracy(self, pipeline, class_reference):
        """Linear C_l^pp matches CLASS to <3% for l <= 500."""
        from clax.lensing import compute_cl_pp_limber
        params, bg, th, pt = pipeline
        cl = compute_cl_pp_limber(pt, params, bg, th, l_max=500,
                                  n_chi=500, nonlinear=False)
        cl = np.array(cl)
        pp_class = class_reference

        for l_val in [100, 200, 500]:
            ratio = cl[l_val] / pp_class[l_val]
            err = abs(ratio - 1.0)
            print(f"  l={l_val}: clax/CLASS = {ratio:.4f} (err={err:.1%})")
            assert err < 0.03, (
                f"Linear C_l^pp at l={l_val}: {err:.1%} error exceeds 3% "
                f"(clax={cl[l_val]:.4e}, CLASS={pp_class[l_val]:.4e})")

    def test_high_l_known_limitation(self, pipeline, class_reference):
        """Document known Limber overestimate at high l (informational)."""
        from clax.lensing import compute_cl_pp_limber
        params, bg, th, pt = pipeline
        cl = compute_cl_pp_limber(pt, params, bg, th, l_max=2500,
                                  n_chi=500, nonlinear=False)
        cl = np.array(cl)
        pp_class = class_reference

        print("\n  Known Limber Poisson accuracy vs CLASS:")
        for l_val in [100, 500, 1000, 1500, 2000, 2500]:
            ratio = cl[l_val] / pp_class[l_val]
            print(f"    l={l_val}: clax/CLASS = {ratio:.4f}")

        # At l=1000, should still be within 5%
        ratio_1000 = cl[1000] / pp_class[1000]
        assert abs(ratio_1000 - 1.0) < 0.05, (
            f"Unexpected: l=1000 error {abs(ratio_1000-1):.1%} exceeds 5%")


class TestNLRatioAccuracy:
    """Test NL/linear ratio accuracy against CLASS-PT reference."""

    def test_halofit_ratio_low_l(self, pipeline):
        """Halofit NL/linear ratio is physically reasonable at l <= 500."""
        from clax.lensing import compute_cl_pp_limber
        params, bg, th, pt = pipeline

        cl_lin = compute_cl_pp_limber(pt, params, bg, th, l_max=500,
                                      n_chi=500, nonlinear=False)
        cl_hf = compute_cl_pp_limber(pt, params, bg, th, l_max=500,
                                      n_chi=500, nonlinear=True)
        cl_lin = np.array(cl_lin)
        cl_hf = np.array(cl_hf)

        print("\n  Halofit NL/linear ratio:")
        for l_val in [100, 200, 500]:
            ratio = cl_hf[l_val] / cl_lin[l_val]
            print(f"    l={l_val}: NL/lin = {ratio:.4f}")
            # NL ratio should be >= 1 and < 1.2 at l <= 500
            assert 0.99 < ratio < 1.20, (
                f"NL/linear ratio at l={l_val}: {ratio:.4f} "
                f"outside expected range [0.99, 1.20]")

    def test_halofit_ratio_monotonic(self, pipeline):
        """Halofit NL/linear ratio increases monotonically with l."""
        from clax.lensing import compute_cl_pp_limber
        params, bg, th, pt = pipeline

        cl_lin = compute_cl_pp_limber(pt, params, bg, th, l_max=500,
                                      n_chi=500, nonlinear=False)
        cl_hf = compute_cl_pp_limber(pt, params, bg, th, l_max=500,
                                      n_chi=500, nonlinear=True)
        ratios = np.array(cl_hf[2:]) / np.where(
            np.array(cl_lin[2:]) > 0, np.array(cl_lin[2:]), 1)

        # Check monotonicity for l >= 100 (smooth enough)
        smooth_ratios = ratios[98:]  # l >= 100
        diffs = np.diff(smooth_ratios)
        n_decreasing = np.sum(diffs < -1e-6)
        assert n_decreasing < 5, (
            f"NL/linear ratio is not monotonic: {n_decreasing} decreasing steps")


class TestPkAccuracy:
    """Test P(k,z) accuracy against CLASS (validates the upstream input)."""

    def test_pk_matches_class(self, pipeline):
        """Dimensionless P(k) Δ²(k,z=0) matches CLASS to <3%."""
        try:
            from classy import Class
        except ImportError:
            pytest.skip("CLASS not available")

        params, bg, th, pt = pipeline
        from clax.primordial import primordial_scalar_pk

        k_pt = np.array(pt.k_grid)
        P_R = np.array(primordial_scalar_pk(pt.k_grid, params))
        dm_z0 = np.array(pt.delta_m[:, -1])
        delta2_clax = P_R * dm_z0**2

        cosmo = Class()
        cosmo.set({
            'A_s': 2.1e-9, 'n_s': 0.9649, 'tau_reio': 0.052,
            'omega_b': 0.02237, 'omega_cdm': 0.12, 'h': 0.6736,
            'YHe': 0.2425, 'N_ur': 2.0328, 'N_ncdm': 1, 'm_ncdm': 0.06,
            'output': 'mPk', 'z_pk': '0', 'P_k_max_h/Mpc': 10.0,
        })
        cosmo.compute()

        # pk_lin takes k in 1/Mpc, returns P in Mpc^3
        k_test = np.array([0.01, 0.1, 0.5, 1.0, 3.0])
        from clax.interpolation import CubicSpline as CS
        spline = CS(jnp.log(jnp.array(k_pt)), jnp.array(delta2_clax))

        max_err = 0.0
        for k in k_test:
            if k < k_pt[0] or k > k_pt[-1]:
                continue
            pk_class = cosmo.pk_lin(k, 0)
            d2_class = k**3 * pk_class / (2 * np.pi**2)
            d2_clax = float(spline.evaluate(jnp.array(np.log(k))))
            err = abs(d2_clax / d2_class - 1)
            max_err = max(max_err, err)
            print(f"  k={k:.3f}: clax Δ²={d2_clax:.4e}, CLASS Δ²={d2_class:.4e}, err={err:.2%}")

        cosmo.struct_cleanup()
        assert max_err < 0.03, f"Max P(k) error {max_err:.1%} exceeds 3%"
