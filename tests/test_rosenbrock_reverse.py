"""Reverse-mode regression tests for the Rosenbrock solvers on the REAL
perturbation RHS.

The smooth-RHS reverse tests in test_rosenbrock.py pass, so the LU VJP is
sound. The NaN (GPU job 13922: grad_recursive(rodas5)=nan while
jvp_direct(rodas5)=4.029599e6) needs the real RHS + adaptive stepping.
These tests pin (a) finiteness and (b) agreement with forward mode, which
is FD-verified exact on this solver.
"""
import jax
import jax.numpy as jnp
import pytest
from dataclasses import replace as dc_replace

from clax import CosmoParams, PrecisionParams
from clax.background import background_solve
from clax.thermodynamics import thermodynamics_solve
from clax.perturbations import perturbations_solve_mpk

_PREC_RB = dc_replace(
    PrecisionParams.fast_cl(),
    pt_ode_solver="rodas5",
    pt_ode_rtol=1e-4, pt_ode_atol=1e-4,
    ode_max_steps=16384,
)


def _make_f(prec):
    base = CosmoParams()

    def f(h_val):
        p = base.replace(h=h_val)
        bg = background_solve(p, prec)
        th = thermodynamics_solve(p, prec, bg)
        pt = perturbations_solve_mpk(p, prec, bg, th)
        return jnp.sum(pt.delta_m[:, -1] ** 2)

    return f


@pytest.mark.slow
def test_grad_mpk_rodas5_batched_finite():
    """grad through the batched-Rosenbrock mPk path is finite (was NaN)."""
    g = jax.grad(_make_f(_PREC_RB))(jnp.asarray(float(CosmoParams().h)))
    assert jnp.isfinite(g), f"reverse-mode gradient is not finite: {g}"


@pytest.mark.slow
def test_grad_matches_jvp_rodas5():
    """grad == jvp on the same rodas5 graph (jvp is FD-verified exact)."""
    prec_d = dc_replace(_PREC_RB, ode_adjoint="direct")
    h0 = jnp.asarray(float(CosmoParams().h))
    g = jax.grad(_make_f(_PREC_RB))(h0)
    _, tan = jax.jvp(_make_f(prec_d), (h0,), (jnp.asarray(1.0),))
    rel = jnp.abs(g - tan) / jnp.maximum(jnp.abs(tan), 1e-30)
    assert jnp.isfinite(g), f"grad not finite: {g}"
    assert rel < 1e-4, (
        f"grad {float(g):.8e} vs jvp {float(tan):.8e}: rel {float(rel):.2e}"
    )
