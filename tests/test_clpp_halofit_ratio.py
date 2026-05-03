"""End-to-end test: Halofit C_l^pp NL/linear ratio vs CLASS reference.

Runs the full pipeline (background + thermo + perturbations + source-Limber
C_l^pp) with ``nonlinear="halofit"`` and compares against CLASS v3.3.4
Halofit data.

Reference: ``reference_data/classpt_clpp_halofit.npz``
  Generated with CLASS v3.3.4 (full Limber scheme + Halofit) for default
  ``CosmoParams``.

Runtime: ~90-180s (perturbation solve at k_max=5.0 with lean hierarchy +
50-point z-grid Halofit modulator).
"""
import os
import numpy as np
import pytest
import jax
jax.config.update("jax_enable_x64", True)

import clax
from clax.perturbations import perturbations_solve
from clax.lensing import compute_cl_pp
from dataclasses import replace as dc_replace


REFERENCE_FILE = os.path.join(
    os.path.dirname(__file__), "..",
    "reference_data", "classpt_clpp_halofit.npz")


@pytest.fixture(scope="module")
def pipeline_results():
    """Run the perturbation solve once for all tests in this module."""
    params = clax.CosmoParams()  # defaults match reference data exactly
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
    """Compute linear and Halofit C_l^pp via the public source-Limber path."""
    params, _, bg, th, pt = pipeline_results
    l_max = 2500

    cl_pp_lin = np.array(compute_cl_pp(
        pt, params, bg, th, l_max=l_max, nonlinear="none"))
    cl_pp_hf = np.array(compute_cl_pp(
        pt, params, bg, th, l_max=l_max, nonlinear="halofit"))

    return cl_pp_lin, cl_pp_hf


@pytest.fixture(scope="module")
def class_reference():
    if not os.path.isfile(REFERENCE_FILE):
        pytest.skip(f"reference data not found: {REFERENCE_FILE}")
    return np.load(REFERENCE_FILE)


class TestClppLinear:
    """Linear ``compute_cl_pp(nonlinear='none')`` vs CLASS v3.3.4."""

    def test_linear_clpp_all_l(self, cl_pp_results, class_reference):
        """Matches CLASS to <1% for l in {100, 500, 1000, 2000, 2500}."""
        cl_pp_lin, _ = cl_pp_results
        ref = class_reference

        print("\nLinear C_l^pp vs CLASS v3.3.4:")
        print(f"  {'l':>5s}  {'clax':>12s}  {'CLASS':>12s}  {'err':>8s}")
        for l_val in [100, 500, 1000, 2000, 2500]:
            idx = l_val - 2
            rel_err = abs(cl_pp_lin[l_val] - ref['pp_lin'][idx]) / ref['pp_lin'][idx]
            print(f"  {l_val:5d}  {cl_pp_lin[l_val]:12.4e}  {ref['pp_lin'][idx]:12.4e}  {rel_err:8.2%}")
            assert rel_err < 0.01, (
                f"Linear C_l^pp at l={l_val}: {rel_err:.2%} error exceeds 1%")


class TestClppHalofitRatio:
    """``compute_cl_pp(nonlinear='halofit')`` NL/linear ratio vs CLASS Halofit."""

    def test_ratio_at_low_l(self, cl_pp_results, class_reference):
        """NL/linear ratio matches CLASS within 1% for l <= 500.

        Source-multiplication recipe (matches CLASS) + 100-point z-grid +
        log-log k-extension to 10 Mpc^-1. Measured residuals at the
        default cosmology: 0.01% at l=100, 0.04% at l=200, 0.21% at l=500.
        """
        cl_pp_lin, cl_pp_hf = cl_pp_results
        ref = class_reference

        print("\nNL/linear ratio comparison (l <= 500):")
        print(f"  {'l':>5s}  {'clax':>8s}  {'CLASS':>8s}  {'err':>8s}")
        for l_val in [100, 200, 500]:
            idx = l_val - 2
            ref_ratio = ref['pp_halofit'][idx] / ref['pp_lin'][idx]
            our_ratio = cl_pp_hf[l_val] / cl_pp_lin[l_val]

            rel_err = abs(our_ratio / ref_ratio - 1.0)
            print(f"  {l_val:5d}  {our_ratio:8.4f}  {ref_ratio:8.4f}  {rel_err:8.2%}")
            assert rel_err < 0.01, (
                f"l={l_val}: NL ratio err={rel_err:.2%} exceeds 1%")

    def test_ratio_at_high_l(self, cl_pp_results, class_reference):
        """NL/linear ratio matches CLASS within 2% at l >= 1000.

        Measured residuals at the default cosmology: 0.63% at l=1000,
        0.79% at l=1500, 1.40% at l=2000, 0.96% at l=2500. The 2%
        threshold leaves a margin for cosmology variations.
        """
        cl_pp_lin, cl_pp_hf = cl_pp_results
        ref = class_reference

        print("\nNL/linear ratio comparison (high l):")
        print(f"  {'l':>5s}  {'clax':>8s}  {'CLASS':>8s}  {'clax/CL':>8s}")
        for l_val in [1000, 1500, 2000, 2500]:
            idx = l_val - 2
            ref_ratio = ref['pp_halofit'][idx] / ref['pp_lin'][idx]
            our_ratio = cl_pp_hf[l_val] / cl_pp_lin[l_val]
            print(f"  {l_val:5d}  {our_ratio:8.4f}  {ref_ratio:8.4f}  {our_ratio/ref_ratio:8.4f}")
            assert abs(our_ratio / ref_ratio - 1) < 0.02, (
                f"l={l_val}: ratio discrepancy exceeds 2%")

    def test_ratio_monotonic_increase(self, cl_pp_results):
        """NL/linear ratio increases with l from l=100 to l~2000."""
        cl_pp_lin, cl_pp_hf = cl_pp_results

        r100 = cl_pp_hf[100] / cl_pp_lin[100]
        r500 = cl_pp_hf[500] / cl_pp_lin[500]
        r1000 = cl_pp_hf[1000] / cl_pp_lin[1000]

        assert r500 > r100, f"ratio should increase: r500={r500:.4f} < r100={r100:.4f}"
        assert r1000 > r500, f"ratio should increase: r1000={r1000:.4f} < r500={r500:.4f}"
