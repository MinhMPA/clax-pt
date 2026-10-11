"""Unit tests for Rosenbrock-Wanner ODE solvers.

Tests Rodas5 against Kvaerno5 (reference) on stiff and non-stiff
problems. Verifies: convergence order, accuracy, L-stability, and
autodifferentiability (jax.grad through the solve).
"""

import jax
import jax.numpy as jnp
import diffrax
import pytest

jax.config.update("jax_enable_x64", True)

from clax.rosenbrock import Rodas5, Rodas5Batched


# ---------------------------------------------------------------------------
# Test problems
# ---------------------------------------------------------------------------

def _van_der_pol(mu):
    """Van der Pol oscillator, stiff for large mu."""
    def f(t, y, args):
        return jnp.array([y[1], mu * (1 - y[0]**2) * y[1] - y[0]])
    return f


def _exponential_decay(lam):
    """y' = -lam * y, exact solution y(t) = y0 * exp(-lam * t)."""
    def f(t, y, args):
        return -lam * y
    return f


def _linear_2d():
    """2D linear system with known eigenvalues.

    y' = A y,  A = [[-1, 2], [0, -1000]]
    Eigenvalues: -1 and -1000 (stiffness ratio 1000).
    Exact: y1(t) = (y1_0 + 2*y2_0/999) * exp(-t) - 2*y2_0/999 * exp(-1000t)
           y2(t) = y2_0 * exp(-1000t)
    """
    A = jnp.array([[-1.0, 2.0], [0.0, -1000.0]])
    def f(t, y, args):
        return A @ y
    def exact(t, y0):
        e1 = jnp.exp(-t)
        e2 = jnp.exp(-1000.0 * t)
        y2 = y0[1] * e2
        y1 = (y0[0] + 2.0 * y0[1] / 999.0) * e1 - 2.0 * y0[1] / 999.0 * e2
        return jnp.array([y1, y2])
    return f, exact


# ---------------------------------------------------------------------------
# Reference solver helper
# ---------------------------------------------------------------------------

def _solve(solver, f, y0, t1, rtol=1e-8, atol=1e-10, dt0=None, max_steps=50000):
    """Solve to t1 and return final state."""
    term = diffrax.ODETerm(f)
    sol = diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=t1, dt0=dt0 or t1 * 1e-3, y0=y0,
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        max_steps=max_steps,
        saveat=diffrax.SaveAt(t1=True),
    )
    return sol.ys[0]


# ---------------------------------------------------------------------------
# Rodas5 tests
# ---------------------------------------------------------------------------

class TestRodas5:
    """Tests for the 8-stage Rodas5 Rosenbrock solver."""

    def test_order(self):
        solver = Rodas5()
        term = diffrax.ODETerm(lambda t, y, args: y)
        assert solver.order(term) == 5
        assert solver.error_order(term) == 4

    def test_exponential_decay(self):
        """Exact solution for y' = -y."""
        f = _exponential_decay(1.0)
        y0 = jnp.array([1.0])
        y_final = _solve(Rodas5(), f, y0, t1=5.0)
        exact = jnp.exp(-5.0)
        assert abs(y_final[0] / exact - 1) < 1e-7, \
            f"Rodas5 exp decay: rel err {abs(y_final[0]/exact - 1):.3e}"

    def test_stiff_linear_system(self):
        """2D stiff system with stiffness ratio 1000."""
        f, exact_fn = _linear_2d()
        y0 = jnp.array([1.0, 1.0])
        y_final = _solve(Rodas5(), f, y0, t1=0.1, rtol=1e-8, atol=1e-12)
        y_exact = exact_fn(0.1, y0)
        # y2 decays to exp(-100) ~ 0; use absolute error for that component
        # y1 is O(1); use relative error
        assert abs(y_final[0] / y_exact[0] - 1) < 1e-6, \
            f"Rodas5 stiff linear y1: rel err {abs(y_final[0]/y_exact[0]-1):.3e}"
        assert abs(y_final[1]) < 1e-6, \
            f"Rodas5 stiff linear y2: |y2| = {abs(y_final[1]):.3e} (should be ~0)"

    def test_matches_kvaerno5_on_van_der_pol(self):
        """Van der Pol (mu=10) final state matches Kvaerno5 within tolerance."""
        f = _van_der_pol(10.0)
        y0 = jnp.array([2.0, 0.0])
        y_rodas = _solve(Rodas5(), f, y0, t1=20.0, rtol=1e-8, atol=1e-10)
        y_kvarn = _solve(diffrax.Kvaerno5(), f, y0, t1=20.0, rtol=1e-8, atol=1e-10)
        diff = jnp.max(jnp.abs(y_rodas - y_kvarn))
        assert diff < 1e-4, f"Rodas5 vs Kvaerno5 on VdP: diff {diff:.3e}"

    def test_l_stability(self):
        """L-stability: very stiff decay to zero (lam=1e6)."""
        f = _exponential_decay(1e6)
        y0 = jnp.array([1.0])
        y_final = _solve(Rodas5(), f, y0, t1=1.0, rtol=1e-4, atol=1e-8, dt0=1e-2)
        # Exact is exp(-1e6) ~ 0. L-stable solver must not blow up.
        assert abs(y_final[0]) < 1e-3, \
            f"Rodas5 L-stability: |y(1)| = {abs(y_final[0]):.3e} (should be ~0)"

    def test_ad_initial_condition(self):
        """jax.grad through Rodas5 w.r.t. initial condition."""
        f = _van_der_pol(1.0)
        def objective(y0_val):
            y0 = jnp.array([y0_val, 0.0])
            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(f), Rodas5(), t0=0.0, t1=5.0, dt0=0.01, y0=y0,
                stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
                max_steps=10000, saveat=diffrax.SaveAt(t1=True),
                adjoint=diffrax.RecursiveCheckpointAdjoint(),
            )
            return sol.ys[0, 0]

        grad_ad = jax.grad(objective)(2.0)
        eps = 1e-6
        grad_fd = (objective(2.0 + eps) - objective(2.0 - eps)) / (2 * eps)
        rel_err = abs(grad_ad / grad_fd - 1)
        assert rel_err < 1e-4, \
            f"Rodas5 AD IC gradient: rel err {rel_err:.3e}"

    def test_ad_parameter_gradient(self):
        """jax.grad through Rodas5 w.r.t. ODE parameter."""
        def objective(mu):
            f = _van_der_pol(mu)
            y0 = jnp.array([2.0, 0.0])
            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(f), Rodas5(), t0=0.0, t1=5.0, dt0=0.01, y0=y0,
                stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
                max_steps=10000, saveat=diffrax.SaveAt(t1=True),
                adjoint=diffrax.RecursiveCheckpointAdjoint(),
            )
            return sol.ys[0, 0]

        grad_ad = jax.grad(objective)(1.0)
        eps = 1e-5
        grad_fd = (objective(1.0 + eps) - objective(1.0 - eps)) / (2 * eps)
        rel_err = abs(grad_ad / grad_fd - 1)
        assert rel_err < 1e-3, \
            f"Rodas5 AD param gradient: rel err {rel_err:.3e}"

    def test_convergence_order(self):
        """Verify order 5 convergence on smooth non-stiff problem."""
        f = _exponential_decay(1.0)
        y0 = jnp.array([1.0])
        exact = jnp.exp(-1.0)

        errors = []
        dts = [0.2, 0.1, 0.05]
        for dt in dts:
            n_steps = int(1.0 / dt) + 2
            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(f), Rodas5(), t0=0.0, t1=1.0, dt0=dt, y0=y0,
                stepsize_controller=diffrax.ConstantStepSize(),
                max_steps=n_steps,
                saveat=diffrax.SaveAt(t1=True),
            )
            errors.append(float(abs(sol.ys[0, 0] - exact)))

        # Convergence rate: error ~ dt^p => log(e1/e2) / log(dt1/dt2) ~ p
        rate1 = jnp.log(errors[0] / errors[1]) / jnp.log(dts[0] / dts[1])
        rate2 = jnp.log(errors[1] / errors[2]) / jnp.log(dts[1] / dts[2])
        avg_rate = (rate1 + rate2) / 2
        assert avg_rate > 4.5, \
            f"Rodas5 convergence order: {avg_rate:.2f} (expected ~5)"


# ---------------------------------------------------------------------------
# SaveAt tests (intermediate times)
# ---------------------------------------------------------------------------

class TestSaveAt:
    """Test that solvers work with SaveAt(ts=...) for trajectory output."""

    def test_rodas5_saveat_ts(self):
        """Rodas5 saves states at requested times."""
        f = _exponential_decay(1.0)
        y0 = jnp.array([1.0])
        ts = jnp.linspace(0.0, 5.0, 50)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(f), Rodas5(), t0=0.0, t1=5.0, dt0=0.01, y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-10),
            max_steps=10000,
            saveat=diffrax.SaveAt(ts=ts),
        )
        exact = jnp.exp(-ts)[:, None]
        # LocalLinearInterpolation limits accuracy between steps
        rel_err = jnp.max(jnp.abs(sol.ys / exact - 1))
        assert rel_err < 0.01, \
            f"Rodas5 SaveAt trajectory: max rel err {rel_err:.3e}"

    def test_rodas5_dense(self):
        """Rodas5 with dense=True still works."""
        f = _exponential_decay(1.0)
        y0 = jnp.array([1.0])
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(f), Rodas5(), t0=0.0, t1=1.0, dt0=0.01, y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-10),
            max_steps=1000,
            saveat=diffrax.SaveAt(dense=True),
        )
        y_half = sol.evaluate(0.5)
        exact = jnp.exp(-0.5)
        # LocalLinearInterpolation adds interpolation error
        rel_err = abs(y_half[0] / exact - 1)
        assert rel_err < 0.01, \
            f"Rodas5 dense interpolation at t=0.5: rel err {rel_err:.3e}"


# ---------------------------------------------------------------------------
# Rodas5Batched tests
# ---------------------------------------------------------------------------

class TestRodas5Batched:
    """Tests for the batched Rodas5 solver (shared time-stepping across a batch)."""

    def test_order(self):
        solver = Rodas5Batched()
        term = diffrax.ODETerm(lambda t, y, args: y)
        assert solver.order(term) == 5
        assert solver.error_order(term) == 4

    def test_batch_of_exponential_decays(self):
        """Batch of 4 decays with different rates — each converges independently."""
        rates = jnp.array([0.5, 1.0, 2.0, 3.0])
        batch_size = len(rates)
        n_eq = 1

        def f_single(t, y, rate):
            return -rate * y

        def vf_batched(t, y_batch, args):
            if isinstance(args, tuple) and len(args) == 2:
                return jax.tree_util.tree_map(jnp.zeros_like, y_batch)
            return jax.vmap(f_single, in_axes=(None, 0, 0))(t, y_batch, args)

        y0 = jnp.ones((batch_size, n_eq))
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(vf_batched),
            Rodas5Batched(),
            t0=0.0, t1=1.0, dt0=0.01,
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-8),
            max_steps=5000,
            saveat=diffrax.SaveAt(t1=True),
            args=(f_single, rates),
        )
        y_final = sol.ys[0]  # (batch_size, n_eq)
        exact = jnp.exp(-rates * 1.0)
        for i in range(batch_size):
            rel_err = abs(y_final[i, 0] / exact[i] - 1)
            assert rel_err < 1e-4, (
                f"Rodas5Batched decay[{i}] (rate={rates[i]}): "
                f"rel err {rel_err:.3e}"
            )

    def test_matches_unbatched_rodas5(self):
        """Batched results match vmap of single Rodas5 on stiff 2D system."""
        A = jnp.array([[-1.0, 2.0], [0.0, -1000.0]])
        scales = jnp.array([1.0, 2.0, 0.5])
        batch_size = len(scales)
        n_eq = 2

        # Single-mode solve with unbatched Rodas5
        def f_single(t, y, scale):
            return scale * (A @ y)

        def solve_single(scale):
            f = lambda t, y, args: scale * (A @ y)
            y0 = jnp.array([1.0, 1.0])
            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(f), Rodas5(),
                t0=0.0, t1=0.05, dt0=1e-4, y0=y0,
                stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-12),
                max_steps=10000,
                saveat=diffrax.SaveAt(t1=True),
            )
            return sol.ys[0]

        y_ref = jax.vmap(solve_single)(scales)  # (batch_size, n_eq)

        # Batched solve
        def vf_batched(t, y_batch, args):
            if isinstance(args, tuple) and len(args) == 2:
                return jax.tree_util.tree_map(jnp.zeros_like, y_batch)
            return jax.vmap(f_single, in_axes=(None, 0, 0))(t, y_batch, args)

        y0_batch = jnp.tile(jnp.array([1.0, 1.0]), (batch_size, 1))
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(vf_batched),
            Rodas5Batched(),
            t0=0.0, t1=0.05, dt0=1e-4,
            y0=y0_batch,
            stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-12),
            max_steps=10000,
            saveat=diffrax.SaveAt(t1=True),
            args=(f_single, scales),
        )
        y_batched = sol.ys[0]

        diff = jnp.max(jnp.abs(y_batched - y_ref))
        assert diff < 1e-4, (
            f"Rodas5Batched vs vmap(Rodas5): max diff {diff:.3e}"
        )

    def test_saveat_ts(self):
        """Batched solver saves at intermediate times correctly."""
        rates = jnp.array([1.0, 3.0])
        batch_size = len(rates)

        def f_single(t, y, rate):
            return -rate * y

        def vf_batched(t, y_batch, args):
            if isinstance(args, tuple) and len(args) == 2:
                return jax.tree_util.tree_map(jnp.zeros_like, y_batch)
            return jax.vmap(f_single, in_axes=(None, 0, 0))(t, y_batch, args)

        ts = jnp.linspace(0.0, 1.0, 11)
        y0 = jnp.ones((batch_size, 1))
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(vf_batched),
            Rodas5Batched(),
            t0=0.0, t1=1.0, dt0=0.01,
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-8),
            max_steps=5000,
            saveat=diffrax.SaveAt(ts=ts),
            args=(f_single, rates),
        )
        # sol.ys shape: (n_ts, batch_size, 1)
        assert sol.ys.shape == (11, 2, 1)
        # Check final time for each mode
        for i in range(batch_size):
            exact = jnp.exp(-rates[i] * 1.0)
            rel_err = abs(sol.ys[-1, i, 0] / exact - 1)
            assert rel_err < 1e-5, (
                f"Rodas5Batched saveat mode {i}: rel err {rel_err:.3e}"
            )


# ---------------------------------------------------------------------------
# Dense output (state between steps) tests
# ---------------------------------------------------------------------------

def _stiff_oscillator():
    """Stiff system: an undamped oscillation driving a fast relaxing mode.

        u' = omega v,   v' = -omega u,   w' = -lam (w - u)

    Eigenvalues +-i*omega and -lam.  Exact solution for u(0)=1, v(0)=0:
        u = cos(omega t),  v = -sin(omega t),
        w = w_p(t) + (w(0) - w_p(0)) exp(-lam t),
        w_p = lam (lam cos(omega t) + omega sin(omega t)) / (lam^2 + omega^2).
    All three components have O(1) amplitude, so absolute errors are
    directly comparable.
    """
    def f(t, y, p):
        omega, lam = p[0], p[1]
        return jnp.array([omega * y[1], -omega * y[0], -lam * (y[2] - y[0])])

    def exact(t, omega, lam, w0):
        u = jnp.cos(omega * t)
        v = -jnp.sin(omega * t)
        wp = lam * (lam * jnp.cos(omega * t) + omega * jnp.sin(omega * t)) \
            / (lam**2 + omega**2)
        wp0 = lam * lam / (lam**2 + omega**2)
        w = wp + (w0 - wp0) * jnp.exp(-lam * t)
        return jnp.stack([u, v, w], axis=-1)

    def y_init(omega, lam):
        # w starts one unit off its slow manifold: a genuine stiff transient
        return jnp.array([1.0, 0.0, lam * lam / (lam**2 + omega**2) + 1.0])

    return f, exact, y_init


class TestDenseOutput:
    """The solver's local interpolation between accepted steps.

    The perturbation solve saves sources on a dense conformal-time grid
    (``SaveAt(ts=tau_grid)``), so the state at most output times comes from
    the solver's ``interpolation_cls``, not from a step endpoint.  Until
    Oct 12, 2026 Rodas5/Rodas5Batched used ``LocalLinearInterpolation``,
    which with late-time steps of hundreds of Mpc put the late-ISW source
    (hence TT at l<30) off by up to 46% relative to Kvaerno5 at rtol 1e-4 /
    atol 1e-6 (precision sweep of Oct 10-11, 2026, finding 5).  The fix is
    diffrax's ``ThirdOrderHermitePolynomialInterpolation`` from (y, dt*f) at
    both step ends, the same dense output Kvaerno5 uses.
    """

    T1 = 10.0           # ten oscillation periods at omega = 2*pi
    N_TS = 2001         # output times, ~13 per solver step
    RTOL, ATOL = 1e-4, 1e-7   # loose: 148 steps over T1, omega*h ~ 0.4

    def test_rodas5_dense_output_stiff_oscillator(self):
        """Interpolated state between large Rodas5 steps vs the exact solution.

        Measured max |y - exact| over 2001 output times (148 steps):
        5.4e-2 with LocalLinearInterpolation (before the fix),
        8.0e-4 with the cubic Hermite dense output.  Bar: 5e-3, an order of
        magnitude below the linear number.
        """
        f, exact, y_init = _stiff_oscillator()
        omega, lam = 2.0 * jnp.pi, 1000.0
        p = jnp.array([omega, lam])
        y0 = y_init(omega, lam)
        ts = jnp.linspace(0.0, self.T1, self.N_TS)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(f), Rodas5(), t0=0.0, t1=self.T1, dt0=1e-3,
            y0=y0, args=p,
            stepsize_controller=diffrax.PIDController(rtol=self.RTOL, atol=self.ATOL),
            max_steps=100000, saveat=diffrax.SaveAt(ts=ts),
        )
        err = float(jnp.max(jnp.abs(sol.ys - exact(ts, omega, lam, y0[2]))))
        n_steps = int(sol.stats["num_steps"])
        assert err < 5e-3, (
            f"Rodas5 dense output: max abs err {err:.3e} over {self.N_TS} "
            f"output times, {n_steps} steps (linear interpolation gives 5.4e-2)"
        )

    def test_rodas5_batched_dense_output_stiff_oscillator(self):
        """Same check for Rodas5Batched (two modes sharing one step size).

        Modes omega = (2pi, 3pi), lam = (1000, 300).  Measured max |y - exact|
        over 2001 output times (186 shared steps): 7.1e-2 with
        LocalLinearInterpolation (before the fix), 1.5e-3 with the cubic
        Hermite dense output.  Bar: 7e-3.
        """
        f, exact, y_init = _stiff_oscillator()
        omegas = jnp.array([2.0 * jnp.pi, 3.0 * jnp.pi])
        lams = jnp.array([1000.0, 300.0])
        params = jnp.stack([omegas, lams], axis=-1)       # (batch, 2)
        y0 = jnp.stack([y_init(omegas[i], lams[i]) for i in range(2)])

        def vf_batched(t, y_batch, args):
            if isinstance(args, tuple) and len(args) == 2:
                return jax.tree_util.tree_map(jnp.zeros_like, y_batch)
            return jax.vmap(f, in_axes=(None, 0, 0))(t, y_batch, args)

        ts = jnp.linspace(0.0, self.T1, self.N_TS)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(vf_batched), Rodas5Batched(),
            t0=0.0, t1=self.T1, dt0=1e-3, y0=y0, args=(f, params),
            stepsize_controller=diffrax.PIDController(rtol=self.RTOL, atol=self.ATOL),
            max_steps=100000, saveat=diffrax.SaveAt(ts=ts),
        )
        ex = jnp.stack([exact(ts, omegas[i], lams[i], y0[i, 2]) for i in range(2)],
                       axis=1)                                # (n_ts, batch, 3)
        err = float(jnp.max(jnp.abs(sol.ys - ex)))
        n_steps = int(sol.stats["num_steps"])
        assert err < 7e-3, (
            f"Rodas5Batched dense output: max abs err {err:.3e} over {self.N_TS} "
            f"output times, {n_steps} steps (linear interpolation gives 7.1e-2)"
        )

    def test_rodas5_ad_through_dense_output(self):
        """Reverse-mode AD through SaveAt(ts=...) vs the analytic derivative.

        Objective: sum_i u(t_i) with u = cos(omega t_i) exactly, so
        d/d omega = -sum_i t_i sin(omega t_i).  The gradient flows through the
        dense-output coefficients (and, after the fix, through the extra RHS
        evaluation per step) under RecursiveCheckpointAdjoint.  Measured rel
        err: 4.5e-3 with LocalLinearInterpolation (the interpolant's shape is
        wrong), 4.9e-6 with the cubic Hermite dense output.  Bar: 1e-3.
        """
        f, exact, y_init = _stiff_oscillator()
        lam = 1000.0
        ts = jnp.linspace(0.0, 3.0, 301)

        def objective(omega):
            p = jnp.array([omega, lam])
            sol = diffrax.diffeqsolve(
                diffrax.ODETerm(f), Rodas5(), t0=0.0, t1=3.0, dt0=1e-3,
                y0=y_init(omega, lam), args=p,
                stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
                max_steps=100000, saveat=diffrax.SaveAt(ts=ts),
                adjoint=diffrax.RecursiveCheckpointAdjoint(),
            )
            return jnp.sum(sol.ys[:, 0])

        omega = 2.0 * jnp.pi
        grad_ad = float(jax.grad(objective)(omega))
        grad_exact = float(-jnp.sum(ts * jnp.sin(omega * ts)))
        rel_err = abs(grad_ad / grad_exact - 1)
        assert rel_err < 1e-3, (
            f"Rodas5 AD through dense output: d/domega AD {grad_ad:.6f} vs "
            f"exact {grad_exact:.6f}, rel err {rel_err:.3e}"
        )
