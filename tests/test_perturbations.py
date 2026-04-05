"""Tests perturbation-solver forward behavior.

Contract:
- The perturbation ODE system is structurally well-posed and integrates to finite states.

Scope:
- Covers initial conditions, RHS finiteness, RHS shape, and one cheap single-mode solve.
- Excludes top-level ``P(k)`` accuracy and gradient contracts owned by dedicated files.

Notes:
- These tests intentionally avoid repeated full-pipeline value or gradient checks.
"""

import jax
jax.config.update("jax_enable_x64", True)

import diffrax
import jax.numpy as jnp
import pytest

from clax.background import background_solve
from clax.params import CosmoParams, PrecisionParams
from clax.perturbations import _adiabatic_ic, _build_indices, _perturbation_rhs
from clax.thermodynamics import thermodynamics_solve


PREC = PrecisionParams(
    bg_n_points=400,
    ncdm_bg_n_points=200,
    bg_tol=1e-8,
    th_n_points=10000,
    th_z_max=5e3,
    pt_l_max_g=17,
    pt_l_max_pol_g=17,
    pt_l_max_ur=17,
)


@pytest.fixture(scope="module")
def bg():
    """Compute the fiducial background state once for this module."""
    return background_solve(CosmoParams(), PREC)


@pytest.fixture(scope="module")
def th(bg):
    """Compute the fiducial thermodynamics state once for this module."""
    return thermodynamics_solve(CosmoParams(), PREC, bg)


def _solve_single_mode(bg, th, k=0.05):
    """Integrate one low-cost perturbation mode; expects a finite final state."""
    idx = _build_indices(6, 6, 6)
    tau_ini = min(0.5, 0.01 / k)
    y0 = _adiabatic_ic(k, jnp.array(tau_ini), bg, CosmoParams(), idx, idx["n_eq"])
    args = (k, bg, th, CosmoParams(), idx, 6, 6, 6)
    return diffrax.diffeqsolve(
        diffrax.ODETerm(_perturbation_rhs),
        solver=diffrax.Kvaerno5(),
        t0=tau_ini,
        t1=1.0,
        dt0=tau_ini * 0.1,
        y0=y0,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        max_steps=16384,
        args=args,
    )


class TestPerturbationInitialConditions:
    """Tests perturbation initial conditions."""

    def test_ic_state_is_finite(self, bg):
        """Initial conditions are finite; expects no NaN or Inf entries."""
        idx = _build_indices(6, 6, 6)
        y0 = _adiabatic_ic(0.01, jnp.array(0.5), bg, CosmoParams(), idx, idx["n_eq"])
        assert jnp.all(jnp.isfinite(y0)), "Initial conditions: found non-finite entries; expected all finite"

    def test_ic_eta_matches_curvature_normalization(self, bg):
        """Initial ``eta`` matches curvature normalization; expects <1% offset from unity."""
        idx = _build_indices(6, 6, 6)
        y0 = _adiabatic_ic(0.01, jnp.array(0.5), bg, CosmoParams(), idx, idx["n_eq"])
        eta_ini = float(y0[idx["eta"]])
        assert abs(eta_ini - 1.0) < 0.01, f"eta_ini: value {eta_ini:.6f}; expected within 1% of unity"


class TestPerturbationRhs:
    """Tests perturbation RHS evaluation."""

    def test_rhs_is_finite(self, bg, th):
        """RHS evaluated on valid ICs is finite; expects no NaN or Inf entries."""
        idx = _build_indices(6, 6, 6)
        y0 = _adiabatic_ic(0.05, jnp.array(0.5), bg, CosmoParams(), idx, idx["n_eq"])
        args = (0.05, bg, th, CosmoParams(), idx, 6, 6, 6)
        dy = _perturbation_rhs(jnp.array(100.0), y0, args)
        assert jnp.all(jnp.isfinite(dy)), "RHS: found non-finite entries; expected all finite"

    def test_rhs_shape_matches_state(self, bg, th):
        """RHS output shape matches state shape; expects identical array shapes."""
        idx = _build_indices(6, 6, 6)
        y0 = _adiabatic_ic(0.05, jnp.array(0.5), bg, CosmoParams(), idx, idx["n_eq"])
        args = (0.05, bg, th, CosmoParams(), idx, 6, 6, 6)
        dy = _perturbation_rhs(jnp.array(100.0), y0, args)
        assert dy.shape == y0.shape, f"RHS shape: got {dy.shape}; expected {y0.shape}"


class TestPerturbationIntegration:
    """Tests cheap single-mode perturbation integration."""

    def test_single_mode_final_state_is_finite(self, bg, th):
        """A very short single-mode solve reaches a finite final state; expects no NaN or Inf entries."""
        sol = _solve_single_mode(bg, th, k=0.01)
        y_final = sol.ys[-1]
        assert jnp.all(jnp.isfinite(y_final)), "Single-mode solve: found non-finite final-state entries; expected all finite"
