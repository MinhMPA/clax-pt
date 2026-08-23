"""Tests for the NaN-safe windowed TCA<->full blend (``clax.perturbations._tca_blend``).

Background: the perturbation ODE RHS switched between the tight-coupling
approximation (TCA) and the full Boltzmann equations with a hard
``jnp.where(is_tca > 0.5, tca_expr, full_expr)``, even though ``is_tca`` is
already a smooth sigmoid (cf. ``_compute_tca_criterion``). That hard switch
injected a finite discontinuity into the RHS at the TCA-off crossover
(``tau_c/tau_k ~ 0.01``, near matter-radiation equality), which stalled the
perturbation solve at k ~ 0.05 Mpc^-1 (delta_b -> 1e49).

``_tca_blend`` fixes this by blending the two expressions continuously across
a narrow window around the crossover, while *selecting* (not blending)
outside that window so a non-finite value in the unused branch cannot poison
the result via ``0 * inf = NaN`` — preserving the NaN/Inf immunity the
original ``jnp.where`` had. These tests check exactly those properties.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from clax.perturbations import _tca_blend, _TCA_BLEND_EPS


# ---------------------------------------------------------------------------
# (a) Limits: exact recovery of each branch at the endpoints
# ---------------------------------------------------------------------------

def test_limit_is_tca_one_returns_tca_val_exactly():
    A, B = 3.14159, -2.71828
    out = _tca_blend(jnp.array(1.0), jnp.array(A), jnp.array(B))
    assert float(out) == A


def test_limit_is_tca_zero_returns_full_val_exactly():
    A, B = 3.14159, -2.71828
    out = _tca_blend(jnp.array(0.0), jnp.array(A), jnp.array(B))
    assert float(out) == B


# ---------------------------------------------------------------------------
# (b) Continuity: no jump larger than a small tolerance across the sweep,
#     and the result stays monotone between the two endpoint values.
# ---------------------------------------------------------------------------

def test_continuity_across_is_tca_sweep():
    A, B = 5.0, -3.0  # tca_val, full_val
    is_tca_grid = jnp.linspace(0.0, 1.0, 20001)
    out = jax.vmap(lambda x: _tca_blend(x, A, B))(is_tca_grid)
    out_np = np.asarray(out)

    # No jump larger than a small tolerance between adjacent samples.
    diffs = np.abs(np.diff(out_np))
    max_jump = diffs.max()
    assert max_jump < 1e-3, f"jump of {max_jump} found (discontinuity not removed)"

    # Monotone (non-decreasing) between B (at is_tca=0) and A (at is_tca=1).
    assert np.all(diffs >= -1e-9), "blend is not monotone between B and A"
    assert out_np[0] == pytest.approx(B, abs=1e-12)
    assert out_np[-1] == pytest.approx(A, abs=1e-12)


def test_continuity_at_transition_boundaries():
    # Specifically probe tightly around each window edge defined by
    # _TCA_BLEND_EPS (where the piecewise definition switches from "select"
    # to "blend"), since the coarse global sweep in the previous test may
    # step over these narrow (width ~1e-6) regions entirely.
    A, B = 1.0, -1.0
    eps = _TCA_BLEND_EPS

    def max_jump(points):
        out = jax.vmap(lambda x: _tca_blend(x, A, B))(jnp.array(points))
        return float(np.abs(np.diff(np.asarray(out))).max())

    # Around the low edge (is_tca ~ eps): a tight span of width ~10*eps must
    # not produce a jump anywhere near the full A-B range.
    low_edge = jnp.linspace(0.0, 10 * eps, 501)
    assert max_jump(low_edge) < 1e-3, "jump found near low window edge"

    # Around the high edge (is_tca ~ 1-eps): symmetric check.
    high_edge = jnp.linspace(1.0 - 10 * eps, 1.0, 501)
    assert max_jump(high_edge) < 1e-3, "jump found near high window edge"


# ---------------------------------------------------------------------------
# (c) NaN/Inf immunity — the key regression test.
# ---------------------------------------------------------------------------

def test_nan_inf_immunity_deep_tca_with_inf_full_val():
    # Deep TCA (is_tca=1.0): full_val is inf (as if kappa_dot blew up in the
    # unused full-equations branch). Primal must equal tca_val, finitely.
    tca_val = 2.5
    out = _tca_blend(jnp.array(1.0), jnp.array(tca_val), jnp.array(jnp.inf))
    assert jnp.isfinite(out)
    assert float(out) == tca_val


def test_nan_inf_immunity_deep_tca_with_inf_full_val_grad_finite():
    def f(is_tca):
        return _tca_blend(is_tca, jnp.array(2.5), jnp.array(jnp.inf))

    grad = jax.grad(f)(jnp.array(1.0))
    assert jnp.isfinite(grad), f"grad through inf-poisoned branch is not finite: {grad}"


def test_nan_inf_immunity_deep_tca_with_nan_full_val():
    tca_val = -1.75
    out = _tca_blend(jnp.array(1.0), jnp.array(tca_val), jnp.array(jnp.nan))
    assert jnp.isfinite(out)
    assert float(out) == tca_val


def test_nan_inf_immunity_deep_tca_with_nan_full_val_grad_finite():
    def f(is_tca):
        return _tca_blend(is_tca, jnp.array(-1.75), jnp.array(jnp.nan))

    grad = jax.grad(f)(jnp.array(1.0))
    assert jnp.isfinite(grad), f"grad through nan-poisoned branch is not finite: {grad}"


def test_nan_inf_immunity_mirror_case_full_streaming_with_inf_tca_val():
    # Mirror: is_tca=0.0 (fully free-streaming), tca_val is inf (as if tau_c
    # blew up in the unused TCA branch). Must return full_val finitely.
    full_val = 0.42
    out = _tca_blend(jnp.array(0.0), jnp.array(jnp.inf), jnp.array(full_val))
    assert jnp.isfinite(out)
    assert float(out) == full_val


def test_nan_inf_immunity_mirror_case_grad_finite():
    def f(is_tca):
        return _tca_blend(is_tca, jnp.array(jnp.inf), jnp.array(0.42))

    grad = jax.grad(f)(jnp.array(0.0))
    assert jnp.isfinite(grad), f"grad through inf-poisoned TCA branch is not finite: {grad}"


# ---------------------------------------------------------------------------
# (d) AD: gradients finite inside the transition window, plain and under jit.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("is_tca_val", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_grad_finite_wrt_all_args_inside_window(is_tca_val):
    def f(is_tca, tca_val, full_val):
        return _tca_blend(is_tca, tca_val, full_val)

    grads = jax.grad(f, argnums=(0, 1, 2))(
        jnp.array(is_tca_val), jnp.array(1.3), jnp.array(-0.7)
    )
    for g in grads:
        assert jnp.isfinite(g), f"non-finite grad {g} at is_tca={is_tca_val}"


def test_grad_finite_under_jit_inside_window():
    def f(is_tca, tca_val, full_val):
        return _tca_blend(is_tca, tca_val, full_val)

    jitted_grad = jax.jit(jax.grad(f, argnums=(0, 1, 2)))
    grads = jitted_grad(jnp.array(0.5), jnp.array(1.3), jnp.array(-0.7))
    for g in grads:
        assert jnp.isfinite(g), f"non-finite grad under jit: {g}"


def test_jit_matches_eager():
    is_tca = jnp.array(0.37)
    tca_val = jnp.array(2.0)
    full_val = jnp.array(-1.0)
    eager = _tca_blend(is_tca, tca_val, full_val)
    jitted = jax.jit(_tca_blend)(is_tca, tca_val, full_val)
    assert float(eager) == pytest.approx(float(jitted), abs=1e-12)
