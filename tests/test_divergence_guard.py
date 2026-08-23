"""Tests for the AD-safe divergence guard on the mPk matter-power output.

``_matter_delta_m_single_k_impl`` (clax/perturbations.py) wraps its final
``delta_m`` return value in ``eqx.error_if`` to convert a diverged
perturbation solve (e.g. the TCA-transition instability at
k/kappa_dot ~ 0.01, matter-radiation equality) into a loud error instead
of a silently-reported, astronomically large "success" (observed:
P(k) ~ 1e98 while the filtered-norm step controller reports success).

Exercising the real ODE solve to force divergence is expensive on CPU, so
these tests reproduce the exact guard predicate/threshold used in
``_matter_delta_m_single_k_impl`` on synthetic ``delta_m`` values, and
confirm:
  1. Healthy (small, finite) values pass through unchanged.
  2. Diverged values (|delta_m| > 1e20, or non-finite) raise at call time.
  3. The guard is compatible with ``jax.jit`` (matching production usage:
     ``_matter_delta_m_single_k_impl`` is wrapped in plain ``jax.jit``, not
     ``eqx.filter_jit``) and with forward/reverse-mode AD (grad/jvp finite
     on the healthy path, i.e. the guard does not itself introduce NaNs
     or block differentiation).
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import pytest


def _guarded(delta_m):
    """Mirrors the exact guard in _matter_delta_m_single_k_impl (perturbations.py)."""
    return eqx.error_if(
        delta_m,
        ~jnp.isfinite(delta_m) | (jnp.abs(delta_m) > 1e20),
        "compute_pk: perturbation solve diverged (|delta_m|>1e20 or "
        "non-finite) — likely a TCA-transition instability; see project "
        "memory.",
    )


class TestDivergenceGuardHealthyPath:
    """A healthy delta_m must pass through unaltered, eager and under jit."""

    @pytest.mark.parametrize("value", [0.0, 1.0, -37.4, 1e10, -1e19, 1e19])
    def test_healthy_value_passes_through_eager(self, value):
        out = _guarded(jnp.asarray(value))
        assert float(out) == pytest.approx(value)

    @pytest.mark.parametrize("value", [0.0, 1.0, -37.4, 1e10])
    def test_healthy_value_passes_through_under_jit(self, value):
        f = jax.jit(_guarded)
        out = f(jnp.asarray(value))
        assert float(out) == pytest.approx(value)

    def test_healthy_grad_is_finite(self):
        """AD-safety: gradient through the guard must be finite, unaltered."""
        g = jax.grad(lambda x: _guarded(x) ** 2)(jnp.asarray(3.0))
        assert np.isfinite(float(g))
        assert float(g) == pytest.approx(6.0)  # d/dx x^2 at x=3

    def test_healthy_jvp_is_finite(self):
        """AD-safety (forward mode): jvp through the guard must be finite."""
        val, tan = jax.jvp(_guarded, (jnp.asarray(3.0),), (jnp.asarray(1.0),))
        assert np.isfinite(float(val)) and np.isfinite(float(tan))
        assert float(val) == pytest.approx(3.0)
        assert float(tan) == pytest.approx(1.0)


class TestDivergenceGuardFires:
    """A diverged delta_m (huge or non-finite) must raise, not pass silently."""

    @pytest.mark.parametrize("value", [1e21, -1e25, 1e98])
    def test_huge_value_raises_eager(self, value):
        with pytest.raises(Exception):
            _guarded(jnp.asarray(value))

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_nonfinite_value_raises_eager(self, value):
        with pytest.raises(Exception):
            _guarded(jnp.asarray(value))

    def test_huge_value_raises_under_jit(self):
        """Matches production: _matter_delta_m_single_k_impl is jax.jit-wrapped."""
        f = jax.jit(_guarded)
        with pytest.raises(Exception):
            f(jnp.asarray(1e30))

    def test_threshold_boundary(self):
        """1e20 itself must still pass (guard is a strict '>' on 1e20)."""
        out = _guarded(jnp.asarray(1e20))
        assert float(out) == pytest.approx(1e20)
