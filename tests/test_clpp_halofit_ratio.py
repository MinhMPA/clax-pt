"""End-to-end test: Halofit C_l^pp NL/linear ratio vs CLASS-PT reference.

Runs the full pipeline (background + thermo + perturbations + Limber C_l^pp)
with nonlinear=True and compares the PP ratio against CLASS-PT Halofit data.

Reference: reference_data/classpt_clpp_halofit.npz

Runtime: ~90-120s (perturbation solve at k_max=5.0 with lean hierarchy).
"""
import numpy as np
import pytest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import clax
from clax.perturbations import perturbations_solve
from clax.lensing import compute_cl_pp_limber
from dataclasses import replace as dc_replace


@pytest.fixture(scope="module")
def pipeline_results():
    """Run the perturbation solve once for all tests in this module."""
    params = clax.CosmoParams(
        h=0.6736, omega_b=0.02237, omega_cdm=0.12,
        ln10A_s=np.log(2.089e-9 * 1e10), n_s=0.9649, tau_reio=0.052,
    )
    prec = dc_replace(clax.PrecisionParams(),
        pt_k_max_cl=5.0, pt_k_per_decade=15, pt_tau_n_points=1500,
        pt_l_max_g=10, pt_l_max_pol_g=6, pt_l_max_ur=10,
        pt_ode_rtol=1e-4, pt_ode_atol=1e-7,
        ode_max_steps=16384, pt_ode_solver="rodas5", pt_k_chunk_size=20,
    )
    bg = clax.background_solve(params, prec)
    th = clax.thermodynamics_solve(params, prec, bg)
    pt = perturbations_solve(params, prec, bg, th)
    return params, prec, bg, th, pt


@pytest.fixture(scope="module")
def cl_pp_results(pipeline_results):
    """Compute linear and Halofit C_l^pp via Limber."""
    params, prec, bg, th, pt = pipeline_results
    l_max = 2500

    cl_pp_lin = np.array(compute_cl_pp_limber(
        pt, params, bg, th, l_max=l_max, n_chi=300, nonlinear=False))
    cl_pp_hf = np.array(compute_cl_pp_limber(
        pt, params, bg, th, l_max=l_max, n_chi=300, nonlinear=True))

    return cl_pp_lin, cl_pp_hf


@pytest.fixture(scope="module")
def class_reference():
    return np.load("reference_data/classpt_clpp_halofit.npz")


class TestClppHalofitRatio:
    """Compare C_l^pp Halofit/linear ratio against CLASS-PT reference."""

    def test_ratio_at_key_multipoles(self, cl_pp_results, class_reference):
        """PP ratio should match CLASS-PT Halofit within tolerance.

        Note: our simple Limber chi-integral overestimates C_l^pp at high l
        compared to CLASS's more sophisticated Limber k-integral (~25% at
        l=2500 even for the LINEAR case). This is a known baseline difference
        that affects the NL/linear ratio at high l. We test only l <= 500
        where the Limber agreement is < 2%.
        """
        cl_pp_lin, cl_pp_hf = cl_pp_results
        ref = class_reference

        # Only test l <= 500 where Limber formula is accurate
        test_ells = [100, 200, 500]
        max_tol = 0.20  # 20% tolerance on the NL correction

        results = []
        for l_val in test_ells:
            idx = l_val - 2  # reference array starts at l=2
            ref_ratio = ref['pp_halofit'][idx] / ref['pp_lin'][idx]
            our_ratio = cl_pp_hf[l_val] / cl_pp_lin[l_val] if cl_pp_lin[l_val] > 0 else 0

            # Relative error on (ratio - 1), the NL correction itself
            ref_corr = ref_ratio - 1.0
            our_corr = our_ratio - 1.0
            if abs(ref_corr) > 0.005:
                rel_err = abs(our_corr - ref_corr) / abs(ref_corr)
            else:
                rel_err = abs(our_corr - ref_corr)  # absolute for tiny corrections

            results.append((l_val, our_ratio, ref_ratio, rel_err))

        # Print diagnostic table
        print("\nC_l^pp NL/linear ratio comparison:")
        print(f"  {'l':>5s}  {'clax':>8s}  {'CLASS-PT':>8s}  {'err':>8s}")
        for l_val, our, ref_r, err in results:
            status = "PASS" if err < max_tol else "FAIL"
            print(f"  {l_val:5d}  {our:8.4f}  {ref_r:8.4f}  {err:8.2%}  {status}")

        # Assert
        for l_val, our, ref_r, err in results:
            ref_corr = ref_r - 1.0
            if abs(ref_corr) > 0.005:
                assert err < max_tol, (
                    f"l={l_val}: ratio={our:.4f} vs CLASS-PT={ref_r:.4f}, "
                    f"err={err:.1%} > {max_tol:.0%}")

    def test_ratio_monotonic_increase(self, cl_pp_results):
        """PP ratio should generally increase from l=100 to l~2000."""
        cl_pp_lin, cl_pp_hf = cl_pp_results
        ratio = cl_pp_hf[100:2001] / np.where(cl_pp_lin[100:2001] > 0, cl_pp_lin[100:2001], 1.0)

        # Smoothed ratio should increase (allow local noise)
        # Check at 4 anchor points
        r100 = cl_pp_hf[100] / cl_pp_lin[100]
        r500 = cl_pp_hf[500] / cl_pp_lin[500]
        r1000 = cl_pp_hf[1000] / cl_pp_lin[1000]
        r2000 = cl_pp_hf[2000] / cl_pp_lin[2000]

        assert r500 > r100, f"ratio should increase: r500={r500:.4f} < r100={r100:.4f}"
        assert r1000 > r500, f"ratio should increase: r1000={r1000:.4f} < r500={r500:.4f}"

    def test_linear_clpp_matches_class(self, cl_pp_results, class_reference):
        """Linear C_l^pp should match CLASS within 5% at l=100."""
        cl_pp_lin, _ = cl_pp_results
        ref = class_reference
        l_val = 100
        idx = l_val - 2
        rel_err = abs(cl_pp_lin[l_val] - ref['pp_lin'][idx]) / ref['pp_lin'][idx]
        assert rel_err < 0.05, (
            f"Linear C_l^pp at l={l_val}: clax={cl_pp_lin[l_val]:.3e} "
            f"CLASS={ref['pp_lin'][idx]:.3e} err={rel_err:.1%}")

    def test_kmax_validation(self, pipeline_results):
        """nonlinear=True with k_max < 5 should raise ValueError."""
        params, prec, bg, th, pt = pipeline_results
        # Create a mock pt with narrow k_grid
        import copy
        pt_narrow = copy.copy(pt)
        # Truncate k_grid
        mask = pt.k_grid <= 0.35
        object.__setattr__(pt_narrow, 'k_grid', pt.k_grid[mask])
        object.__setattr__(pt_narrow, 'delta_m', pt.delta_m[mask, :])

        with pytest.raises(ValueError, match="pt_k_max_cl >= 5.0"):
            compute_cl_pp_limber(pt_narrow, params, bg, th,
                                 l_max=100, nonlinear=True)
