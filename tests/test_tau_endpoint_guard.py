"""The tau(z) endpoint guard.

perturbations_solve stops at 0.999*tau0 while clax's CubicSpline clamps
out-of-range evaluations, so a z~0 lookup on its result silently returned
delta from z~0.003: -0.33% in P(k) at z=0 (smsharma/clax#42).
`_raise_if_tau_outside_grid` turns that clamp into a loud, AD-safe error.

These call the REAL helper on synthetic values (as
tests/test_divergence_guard.py does). Cosmology-independent -- exempt from
the multi-cosmology RULE.
"""
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from clax.perturbations import _raise_if_tau_outside_grid

GRID = jnp.linspace(100.0, 14139.0, 50)          # ends at ~0.999*tau0 for tau0 ~ 14153 Mpc
TAU0 = 14153.25


def _run(tau):
    return jax.block_until_ready(_raise_if_tau_outside_grid(jnp.asarray(tau), GRID, "t"))


def test_inside_passes_unchanged_scalar_and_vector():
    assert float(_run(5000.0)) == 5000.0
    v = jnp.array([200.0, 7000.0, 14000.0])
    assert np.array_equal(_run(v), v)


def test_endpoints_and_round_off_pass():
    """Nodes themselves, and 5e-9 relative beyond them, sit inside the 1e-8 band."""
    _run(GRID[0])
    _run(GRID[-1])
    _run(GRID[-1] * (1 + 5e-9))
    _run(GRID[0] * (1 - 5e-9))


def test_fires_at_today_on_truncated_grid():
    with pytest.raises(Exception, match="lies outside the perturbation tau_grid"):
        _run(TAU0)


def test_fires_just_beyond_tolerance_both_ends():
    with pytest.raises(Exception, match="perturbations_solve_mpk"):
        _run(GRID[-1] * (1 + 2e-8))
    with pytest.raises(Exception, match="lies outside"):
        _run(GRID[0] * (1 - 2e-8))


def test_fires_if_one_element_of_a_vector_is_out():
    """Halofit's z-grid is vmapped; one bad element must raise."""
    with pytest.raises(Exception, match="lies outside"):
        _run(jnp.array([200.0, 7000.0, TAU0]))


def test_jit_grad_jvp_compatible_on_healthy_input():
    def f(t):
        return _raise_if_tau_outside_grid(t, GRID, "t") ** 2

    assert float(jax.jit(f)(3000.0)) == 9.0e6
    assert float(jax.grad(f)(3000.0)) == 6000.0
    _, tangent = jax.jvp(f, (3000.0,), (1.0,))
    assert float(tangent) == 6000.0


def test_fires_under_jit():
    with pytest.raises(Exception, match="lies outside"):
        jax.block_until_ready(jax.jit(lambda t: _raise_if_tau_outside_grid(t, GRID, "t"))(TAU0))


class TestGuardWiring:
    """Both tau(z) -> delta lookups call the shared guard. A grep, as in
    tests/test_divergence_guard.py: the cheap way to catch a guard silently
    dropped from one site in a later refactor."""

    @pytest.mark.parametrize("modname,func", [
        ("clax.ept", "ept_inputs_from_clax"),
        ("clax.transfer", "compute_pk_from_perturbations"),
    ])
    def test_lookup_calls_guard(self, modname, func):
        import importlib
        import inspect
        src = inspect.getsource(getattr(importlib.import_module(modname), func))
        assert "_raise_if_tau_outside_grid(" in src, f"{func} no longer guards tau(z)"
