"""Tests for the AD-safe divergence guard on the mPk matter-power output.

``clax.perturbations._raise_if_diverged`` is the single shared guard called
at all 3 divergence-guard sites in ``clax/perturbations.py``:
  1. ``_matter_delta_m_single_k_impl`` (single-k ``compute_pk`` path)
  2. ``_solve_mpk_batched_rosenbrock`` (batched Rosenbrock path)
  3. ``_perturbations_solve_mpk_impl`` (table-assembly path behind
     ``compute_pk_table`` / ``compute_pk_interpolator``, the
     docstring-preferred production API)

It converts a diverged perturbation solve (e.g. the TCA-transition
instability at k/kappa_dot ~ 0.01, matter-radiation equality) into a loud
error instead of a silently-reported, astronomically large "success"
(observed: P(k) ~ 1e98 while the filtered-norm step controller reports
success).

Exercising the real ODE solve to force divergence is expensive on CPU, so
most tests below call the real, shared ``_raise_if_diverged`` directly on
synthetic ``delta_m``-shaped values (not a locally re-implemented mirror of
its predicate, so a change to the actual guard is caught here), confirming:
  1. Healthy (small, finite) values pass through unchanged, at both the
     scalar shape used by the single-k path and the 2D (n_k, n_tau) shape
     used by the batched table paths.
  2. Diverged values (|delta_m| > 1e20, or non-finite) raise at call time,
     including when only ONE entry in a large batched array is bad (catches
     a wrong-axis/pre-reshape reduction bug).
  3. The guard is compatible with ``jax.jit`` and with forward/reverse-mode
     AD (grad/jvp finite on the healthy path).

A separate ``TestGuardWiring`` class greps ``clax/perturbations.py`` to
confirm all 3 call sites actually invoke ``_raise_if_diverged`` — this is
the only affordable way (without a multi-minute real ODE solve) to catch a
guard silently dropped from one of the batched/table call sites in a future
refactor.
"""
import inspect

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from clax.perturbations import (
    _raise_if_diverged,
    _matter_delta_m_single_k_impl,
    _solve_mpk_batched_rosenbrock,
    _perturbations_solve_mpk_impl,
)


class TestDivergenceGuardHealthyPath:
    """A healthy delta_m must pass through unaltered, eager and under jit."""

    @pytest.mark.parametrize("value", [0.0, 1.0, -37.4, 1e10, -1e19, 1e19])
    def test_healthy_scalar_passes_through_eager(self, value):
        """Scalar shape, matching the single-k call site."""
        out = _raise_if_diverged(jnp.asarray(value), "test")
        assert float(out) == pytest.approx(value)

    @pytest.mark.parametrize("value", [0.0, 1.0, -37.4, 1e10])
    def test_healthy_scalar_passes_through_under_jit(self, value):
        f = jax.jit(lambda x: _raise_if_diverged(x, "test"))
        out = f(jnp.asarray(value))
        assert float(out) == pytest.approx(value)

    def test_healthy_2d_array_passes_through_eager(self):
        """2D (n_k, n_tau) shape, matching the batched/table call sites."""
        arr = jnp.asarray(np.random.default_rng(0).uniform(-10, 10, size=(7, 13)))
        out = _raise_if_diverged(arr, "test")
        np.testing.assert_allclose(np.asarray(out), np.asarray(arr))

    def test_healthy_grad_is_finite(self):
        g = jax.grad(lambda x: _raise_if_diverged(x, "test") ** 2)(jnp.asarray(3.0))
        assert np.isfinite(float(g))
        assert float(g) == pytest.approx(6.0)  # d/dx x^2 at x=3

    def test_healthy_jvp_is_finite(self):
        val, tan = jax.jvp(
            lambda x: _raise_if_diverged(x, "test"), (jnp.asarray(3.0),), (jnp.asarray(1.0),)
        )
        assert np.isfinite(float(val)) and np.isfinite(float(tan))
        assert float(val) == pytest.approx(3.0)
        assert float(tan) == pytest.approx(1.0)


class TestDivergenceGuardFires:
    """A diverged delta_m (huge or non-finite) must raise, not pass silently."""

    @pytest.mark.parametrize("value", [1e21, -1e25, 1e98])
    def test_huge_scalar_raises_eager(self, value):
        with pytest.raises(Exception):
            jax.block_until_ready(_raise_if_diverged(jnp.asarray(value), "test"))

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_nonfinite_scalar_raises_eager(self, value):
        with pytest.raises(Exception):
            jax.block_until_ready(_raise_if_diverged(jnp.asarray(value), "test"))

    def test_huge_scalar_raises_under_jit(self):
        """Matches production: all 3 guarded functions are plain jax.jit-wrapped.

        ``block_until_ready`` is required, not decorative: JAX dispatches
        asynchronously on GPU, so without it the guard's error is raised after
        the ``pytest.raises`` block has already exited and the test fails with
        DID NOT RAISE even though the guard fired correctly (observed on V100,
        job 13089; the same test passes on CPU, where dispatch is synchronous).
        """
        f = jax.jit(lambda x: _raise_if_diverged(x, "test"))
        with pytest.raises(Exception):
            jax.block_until_ready(f(jnp.asarray(1e30)))

    def test_threshold_boundary(self):
        """1e20 itself must still pass (guard is a strict '>' on 1e20)."""
        out = _raise_if_diverged(jnp.asarray(1e20), "test")
        assert float(out) == pytest.approx(1e20)

    def test_single_bad_entry_in_large_2d_array_raises(self):
        """One bad (k, tau) entry among many healthy ones must still raise —
        this is exactly the failure mode a wrong-axis reduction or a guard
        placed before a reshape would silently miss (see module docstring)."""
        arr = np.random.default_rng(1).uniform(-10, 10, size=(11, 23))
        arr[7, 15] = np.inf
        with pytest.raises(Exception):
            jax.block_until_ready(_raise_if_diverged(jnp.asarray(arr), "test"))

    def test_single_huge_entry_in_2d_array_raises(self):
        arr = np.random.default_rng(2).uniform(-10, 10, size=(11, 23))
        arr[3, 4] = 1e30
        with pytest.raises(Exception):
            jax.block_until_ready(_raise_if_diverged(jnp.asarray(arr), "test"))


class TestGuardWiring:
    """Static check that all 3 documented call sites actually invoke the
    real shared guard. This is the only affordable (no multi-minute ODE
    solve) way to catch the guard being silently dropped from one of the
    batched/table call sites in a future refactor -- unit-testing
    ``_raise_if_diverged`` in isolation (above) cannot detect that its
    caller stopped calling it."""

    @pytest.fixture(scope="class")
    def source(self):
        import clax.perturbations as pt
        return inspect.getsource(pt)

    @pytest.mark.parametrize("func", [
        _matter_delta_m_single_k_impl,
        _solve_mpk_batched_rosenbrock,
        _perturbations_solve_mpk_impl,
    ])
    def test_call_site_invokes_shared_guard(self, func, source):
        # functools.partial(jax.jit, ...)(fn) wraps the raw function; unwrap
        # to get back to the actual Python function whose source we inspect.
        raw = func
        while hasattr(raw, "func"):
            raw = raw.func
        try:
            body = inspect.getsource(raw)
        except (TypeError, OSError):
            pytest.skip(f"could not retrieve source for {raw}")
        assert "_raise_if_diverged(" in body, (
            f"{raw.__name__} no longer calls _raise_if_diverged — the "
            "divergence guard was silently dropped from this call site"
        )

    def test_exactly_three_call_sites_in_module(self, source):
        """Pin the call count so a 4th silent removal+addition elsewhere
        doesn't cancel out, and a definition-site self-reference is not
        miscounted as a call."""
        call_count = source.count("_raise_if_diverged(")
        def_count = source.count("def _raise_if_diverged(")
        assert call_count - def_count == 3, (
            f"expected exactly 3 call sites for _raise_if_diverged, found "
            f"{call_count - def_count} (single-k, batched-rosenbrock, "
            "table-assembly)"
        )
