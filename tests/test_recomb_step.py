"""S0: the shared L-stable ROS2 step on a scalar stiff relaxation with a known
solution (smsharma/clax#48). Cosmology-independent numerics: exempt from the
multi-cosmology rule (CLAUDE.md), which applies to physics-facing tests.

dx/ds = -lam (x - q(s)), q(s) = alpha + beta s has the exact solution
x(s) = q(s) - beta/lam + (x0 - alpha + beta/lam) exp(-lam s).
For this ROS2 (gamma = 1 + 1/sqrt 2) the stability function is
R(z) = (1 + (1 - 2 gamma) z) / (1 - gamma z)^2 > 0 for z < 0, R -> 0: a stiff
relaxation approaches its equilibrium monotonically, with no overshoot.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from clax.thermodynamics import _ros2_advance, _ros2_scalar_step, _smoothstep

ALPHA, BETA = 0.5, 0.3


def _rate(lam):
    return lambda xe, x, s: -lam * (x - (ALPHA + BETA * s))


def _exact(lam, x0, s):
    return ALPHA + BETA * s - BETA / lam + (x0 - ALPHA + BETA / lam) * np.exp(-lam * s)


def _integrate(lam, x0, n):
    h = 1.0 / n
    x = jnp.asarray(x0)
    for j in range(n):
        x = _ros2_scalar_step(_rate(lam), lambda y: y, x, h, (j * h,), ((j + 1) * h,))
    return float(x)


def test_second_order_convergence():
    errs = [abs(_integrate(2.0, 1.0, n) - _exact(2.0, 1.0, 1.0)) for n in (20, 40, 80)]
    r1, r2 = errs[0] / errs[1], errs[1] / errs[2]
    print(f"errors {errs}, ratios {r1:.3f} {r2:.3f}")
    assert 3.5 < r1 < 4.5 and 3.5 < r2 < 4.5, \
        f"ERROR error ratios {r1:.3f}, {r2:.3f} outside the second-order window (3.5, 4.5)"


def test_l_stable_lands_on_equilibrium():
    # lam h = 1e6: one step from far away lands on q = ALPHA without overshoot
    x1 = float(_ros2_scalar_step(_rate(1e6), lambda y: y, jnp.asarray(1.0), 1.0, (0.0,), (0.0,)))
    print(f"x1 = {x1:.12f}, q = {ALPHA}")
    assert abs(x1 - ALPHA) < 1e-5 and x1 >= ALPHA, \
        f"ERROR stiff step landed at {x1:.12f}, not on or above the equilibrium {ALPHA} within 1e-5"


def test_no_excursion_past_one_near_equilibrium():
    # Peebles-like: relax toward q = 1 - 1e-4 from below, stiff (lam h = 1e3)
    lam, qeq = 1e4, 1.0 - 1e-4
    rate = lambda xe, x, s: -lam * (x - qeq)
    xs = [jnp.asarray(0.99)]
    for _ in range(20):
        xs.append(_ros2_scalar_step(rate, lambda y: y, xs[-1], 0.1, (0.0,), (0.0,)))
    xs = np.array([float(v) for v in xs])
    print(f"max x = {xs.max():.12f}")
    assert xs.max() <= qeq + 1e-15 and np.all(np.diff(xs) >= -1e-15), \
        f"ERROR relaxation passed its equilibrium or turned back: max x {xs.max():.15f} vs {qeq}"


@pytest.mark.parametrize("which", ["lam", "alpha"])
def test_step_jvp_matches_fd(which):
    def advance(theta):
        lam = theta if which == "lam" else 50.0
        alpha = theta if which == "alpha" else ALPHA
        rate = lambda xe, x, s: -lam * (x - (alpha + BETA * s))
        return _ros2_advance(rate, lambda y: y, jnp.asarray(1.0), lambda s: (s,), 1.0, 10)
    x0 = 50.0 if which == "lam" else ALPHA
    _, tan = jax.jvp(advance, (jnp.asarray(x0),), (jnp.asarray(1.0),))
    eps = 1e-6 * abs(x0)
    fd = (float(advance(x0 + eps)) - float(advance(x0 - eps))) / (2 * eps)
    print(f"{which}: jvp {float(tan):.10e} fd {fd:.10e}")
    assert np.isfinite(float(tan)) and abs(float(tan) - fd) < 1e-6 * max(abs(fd), 1e-12), \
        f"ERROR jvp {float(tan):.10e} vs finite difference {fd:.10e} ({which})"


def test_smoothstep():
    w = np.asarray(_smoothstep(jnp.linspace(-0.5, 1.5, 9)))
    assert w[0] == 0.0 and w[-1] == 1.0 and abs(float(_smoothstep(0.5)) - 0.5) < 1e-15, \
        f"ERROR smoothstep end values {w[0]}, {w[-1]} or midpoint {float(_smoothstep(0.5))}"
