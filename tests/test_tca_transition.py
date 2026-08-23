"""Tests for the NaN-safe continuous TCA<->full blend (``clax.perturbations._tca_blend``).

Background: the perturbation ODE RHS switched between the tight-coupling
approximation (TCA) and the full Boltzmann equations with a hard
``jnp.where(is_tca > 0.5, tca_expr, full_expr)``, even though ``is_tca`` is
already a smooth sigmoid (cf. ``_compute_tca_criterion``). That hard switch
injected a finite discontinuity into the RHS at the TCA-off crossover
(``tau_c/tau_k ~ 0.01``, near matter-radiation equality), which stalled the
perturbation solve at k ~ 0.05 Mpc^-1 (delta_b -> 1e49).

``_tca_blend`` fixes this by blending the two expressions continuously for
*every* ``is_tca`` in ``[0, 1]``. NaN/Inf immunity is restored per-operand
(via ``jnp.isfinite``), not via a narrow window around the ``is_tca``
endpoints: an earlier version of this function masked based on ``is_tca``
being within ``1e-6`` of 0 or 1, which protects only 0.0002% of the domain,
degenerating to the NaN-unsafe plain blend everywhere else — since ``is_tca``
is a pure function of background/thermo quantities, independent of how
diverged the ODE state ``y`` (and hence ``tca_val``/``full_val``) is. These
tests check the current, per-operand-finiteness design.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from clax.perturbations import _tca_blend


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
# (b) Continuity: the blend is exactly linear in is_tca (no jump anywhere)
#     whenever both operands are finite -- the overwhelmingly common case.
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


def test_blend_matches_plain_arithmetic_blend_when_both_finite():
    """When both operands are finite, _tca_blend IS the plain arithmetic
    blend (is_tca*A + (1-is_tca)*B) -- exactly, for every is_tca in [0, 1],
    not just outside some window. This is what makes it genuinely
    continuous (see test_continuity_across_is_tca_sweep) instead of merely
    "continuous except at two narrow boundaries"."""
    A, B = 1.3, -0.7
    for is_tca_val in [0.0, 1e-9, 0.001, 0.1, 0.3, 0.5, 0.7, 0.9, 0.999, 1.0 - 1e-9, 1.0]:
        is_tca = jnp.array(is_tca_val)
        got = float(_tca_blend(is_tca, jnp.array(A), jnp.array(B)))
        want = is_tca_val * A + (1.0 - is_tca_val) * B
        assert got == pytest.approx(want, abs=1e-12), f"mismatch at is_tca={is_tca_val}"


# ---------------------------------------------------------------------------
# (c) NaN/Inf immunity -- the key regression tests. Unlike the retired
#     windowed design, these now hold for is_tca ANYWHERE in [0, 1], not
#     just within 1e-6 of the endpoints -- this is the actual fix for the
#     finding that _tca_blend degenerated to the NaN-unsafe plain blend
#     across 99.9998% of the domain.
# ---------------------------------------------------------------------------

def test_nan_inf_immunity_deep_tca_with_inf_full_val():
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
    full_val = 0.42
    out = _tca_blend(jnp.array(0.0), jnp.array(jnp.inf), jnp.array(full_val))
    assert jnp.isfinite(out)
    assert float(out) == full_val


def test_nan_inf_immunity_mirror_case_grad_finite():
    def f(is_tca):
        return _tca_blend(is_tca, jnp.array(jnp.inf), jnp.array(0.42))

    grad = jax.grad(f)(jnp.array(0.0))
    assert jnp.isfinite(grad), f"grad through inf-poisoned TCA branch is not finite: {grad}"


@pytest.mark.parametrize("is_tca_val", [0.001, 0.1, 0.3, 0.5, 0.7, 0.9, 0.999])
def test_nan_inf_immunity_mid_transition_with_inf_full_val(is_tca_val):
    """THE key regression test for the finding: a non-finite operand deep
    INSIDE the transition (not just at is_tca=0/1) must not poison the
    primal or the gradient. is_tca is a pure function of background/thermo
    quantities and is independent of the ODE state y that tca_val/full_val
    depend on, so a diverged y with an ordinary mid-transition is_tca is a
    realistic HMC-pathological-proposal scenario."""
    tca_val = 2.5
    is_tca = jnp.array(is_tca_val)
    out = _tca_blend(is_tca, jnp.array(tca_val), jnp.array(jnp.inf))
    assert jnp.isfinite(out), f"primal poisoned at is_tca={is_tca_val}: {out}"
    assert float(out) == tca_val

    grad = jax.grad(lambda x: _tca_blend(x, jnp.array(tca_val), jnp.array(jnp.inf)))(is_tca)
    assert jnp.isfinite(grad), f"grad poisoned at is_tca={is_tca_val}: {grad}"


@pytest.mark.parametrize("is_tca_val", [0.001, 0.1, 0.3, 0.5, 0.7, 0.9, 0.999])
def test_nan_inf_immunity_mid_transition_with_nan_tca_val(is_tca_val):
    """Mirror of the above with the TCA operand poisoned instead."""
    full_val = -0.9
    is_tca = jnp.array(is_tca_val)
    out = _tca_blend(is_tca, jnp.array(jnp.nan), jnp.array(full_val))
    assert jnp.isfinite(out), f"primal poisoned at is_tca={is_tca_val}: {out}"
    assert float(out) == full_val

    grad = jax.grad(lambda x: _tca_blend(x, jnp.array(jnp.nan), jnp.array(full_val)))(is_tca)
    assert jnp.isfinite(grad), f"grad poisoned at is_tca={is_tca_val}: {grad}"


def test_both_operands_non_finite_falls_back_to_hard_select():
    """When BOTH operands are non-finite there is no finite value to recover
    -- this matches jnp.where's own behaviour (no worse than main)."""
    out_tca_side = _tca_blend(jnp.array(0.6), jnp.array(jnp.nan), jnp.array(jnp.inf))
    assert bool(jnp.isnan(out_tca_side))  # is_tca > 0.5 -> selects tca_val (nan)

    out_full_side = _tca_blend(jnp.array(0.3), jnp.array(jnp.nan), jnp.array(jnp.inf))
    assert bool(jnp.isinf(out_full_side))  # is_tca <= 0.5 -> selects full_val (inf)


# ---------------------------------------------------------------------------
# (d) AD: gradients finite inside the transition, plain and under jit.
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


# ---------------------------------------------------------------------------
# (e) Documented residual gap: a *finite* operand computed from an expression
#     with a locally singular derivative can still yield a non-finite
#     gradient -- exactly like a plain jnp.where on the same expressions.
#     This is parity with main (not a regression introduced by _tca_blend),
#     verified directly below so the gap is never silently unexercised.
# ---------------------------------------------------------------------------

def test_grad_can_still_be_nan_for_singular_derivative_parity_with_where():
    # full_val = 1/x is finite at x=0's neighbourhood approach but its
    # derivative blows up at x=0 itself; both a plain jnp.where select and
    # _tca_blend inherit a nan gradient here even though the primal is
    # protected (2.0, the tca branch) -- neither is a regression vs. the
    # other.
    def naive_where(is_tca, x):
        return jnp.where(is_tca > 0.5, 2.0, 1.0 / x)

    def blended(is_tca, x):
        return _tca_blend(is_tca, jnp.array(2.0), 1.0 / x)

    is_tca = jnp.array(1.0)
    x = jnp.array(0.0)

    primal_where = naive_where(is_tca, x)
    primal_blend = blended(is_tca, x)
    assert float(primal_where) == 2.0
    assert float(primal_blend) == 2.0

    grad_where = jax.grad(lambda xx: naive_where(is_tca, xx))(x)
    grad_blend = jax.grad(lambda xx: blended(is_tca, xx))(x)
    assert bool(jnp.isnan(grad_where)), "expected main's jnp.where to also have nan grad here"
    assert bool(jnp.isnan(grad_blend)), "expected _tca_blend to match jnp.where's parity gap"
