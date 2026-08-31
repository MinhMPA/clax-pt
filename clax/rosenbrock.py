"""Rosenbrock-Wanner ODE solvers for stiff systems.

Implements the Rodas5 (8-stage, order 5/4) Rosenbrock method as a
Diffrax-compatible adaptive solver. This avoids Newton iteration: each
step requires only one Jacobian evaluation and one LU factorization,
followed by linear back-substitution per stage.

For the Einstein-Boltzmann system (~60-150 equations), this is ~3-5x
faster per step than implicit ESDIRK methods (Kvaerno5) which need
iterative Newton convergence.

Mathematical formulation (division-free transformed W-form):
    W' = I - h*gamma*J,  where J = df/dy
    For each stage i:
        W' * k_i = h*gamma*(f(t + c_i*h, y + sum_j a_{ij}*k_j) + h*d_i*dT)
                   + gamma * sum_j C_{ij}*k_j
    y_{n+1} = y_n + sum_i b_i * k_i

    This is the original W = I/(h*gamma) - J formulation with both sides of
    each stage equation multiplied through by h*gamma, so no division by h
    (the trial step size) ever appears: a rejected trial step with h -> 0
    gives W' -> I (well conditioned) instead of manufacturing inf, which
    otherwise leaks into reverse-mode cotangents via diffrax's accept/reject
    `where` (issue #30, item 5).

References:
    Rodas5: Di Marzo (1993), "RODAS5(4) - Méthodes de Rosenbrock d'ordre 5(4)"
    Hairer & Wanner (1996), "Solving ODEs II", Section IV.7
    Transformed formulation: cf. DISCO-EB (Hahn, List & Porqueres, arXiv:2311.03291)
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
from diffrax import AbstractAdaptiveSolver, AbstractTerm, ODETerm, RESULTS
from diffrax._local_interpolation import LocalLinearInterpolation


# ===========================================================================
# Rodas5 tableau (Di Marzo 1993, Type 1 — transformed W-formulation)
# ===========================================================================
_R5_GAMMA = 0.19

# Stage coupling coefficients a_{ij}: u_i = y0 + sum_j a_{ij} * k_j
_R5_A21 = 2.0
_R5_A31 = 3.040894194418781
_R5_A32 = 1.041747909077569
_R5_A41 = 2.576417536461461
_R5_A42 = 1.622083060776640
_R5_A43 = -0.9089668560264532
_R5_A51 = 2.760842080225597
_R5_A52 = 1.446624659844071
_R5_A53 = -0.3036980084553738
_R5_A54 = 0.2877498600325443
_R5_A61 = -14.09640773051259
_R5_A62 = 6.925207756232704
_R5_A63 = -41.47510893210728
_R5_A64 = 2.343771018586405
_R5_A65 = 24.13215229196062
# Stages 7, 8 accumulate: u7 = u6 + k6, u8 = u7 + k7

# Linear coupling coefficients C_{ij} (divided by dt in RHS)
_R5_C21 = -10.31323885133993
_R5_C31 = -21.04823117650003
_R5_C32 = -7.234992135176716
_R5_C41 = 32.22751541853323
_R5_C42 = -4.943732386540191
_R5_C43 = 19.44922031041879
_R5_C51 = -20.69865579590063
_R5_C52 = -8.816374604402768
_R5_C53 = 1.260436877740897
_R5_C54 = -0.7495647613787146
_R5_C61 = -46.22004352711257
_R5_C62 = -17.49534862857472
_R5_C63 = -289.6389582892057
_R5_C64 = 93.60855400400906
_R5_C65 = 318.3822534212147
_R5_C71 = 34.20013733472935
_R5_C72 = -14.15535402717690
_R5_C73 = 57.82335640988400
_R5_C74 = 25.83362985412365
_R5_C75 = 1.408950972071624
_R5_C76 = -6.551835421242162
_R5_C81 = 42.57076742291101
_R5_C82 = -13.80770672017997
_R5_C83 = 93.98938432427124
_R5_C84 = 18.77919633714503
_R5_C85 = -31.58359187223370
_R5_C86 = -6.685968952921985
_R5_C87 = -5.810979938412932

# Time node fractions: t_i = t0 + c_i * dt
_R5_C2 = 0.38
_R5_C3 = 0.3878509998321533
_R5_C4 = 0.4839718937873840
_R5_C5 = 0.4570477008819580
# c6 = c7 = c8 = 1.0 (implicit)

# Time derivative coefficients: rhs += dt * d_i * dT
_R5_D1 = _R5_GAMMA
_R5_D2 = -0.1823079225333714636
_R5_D3 = -0.319231832186874912
_R5_D4 = 0.3449828624725343
_R5_D5 = -0.377417564392089818
# d6 = d7 = d8 = 0.0 (implicit)


def _lu_solve(lu_piv, b):
    """Wrapper for LU back-substitution."""
    return jla.lu_solve(lu_piv, b)


class Rodas5(AbstractAdaptiveSolver):
    """8-stage Rosenbrock method of order 5(4) (Di Marzo 1993).

    Uses the division-free transformed W-formulation where:
        W' = I - h*gamma*J
    and the error estimate is simply k_8 (the last stage).

    This is the method used by DISCO-EB for the Einstein-Boltzmann system.
    It is L-stable and stiffly accurate, making it suitable for the stiff
    photon-baryon tight-coupling regime.

    Compared to Kvaerno5 (ESDIRK), this avoids Newton iteration entirely:
    each step needs one Jacobian + one LU factorization + 8 back-substitutions.
    """

    term_structure = AbstractTerm
    interpolation_cls = LocalLinearInterpolation

    def order(self, terms):
        return 5

    def error_order(self, terms):
        return 4

    def init(self, terms, t0, t1, y0, args):
        return None

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        dt = t1 - t0
        f = lambda t, y: terms.vf(t, y, args)

        # -- Jacobian and time derivative via forward-mode AD --
        f0 = f(t0, y0)
        n = y0.shape[0]
        J = jax.jacfwd(lambda y: f(t0, y))(y0)
        dT = jax.jacfwd(lambda t: f(t, y0))(t0)

        # -- Form W' = I - dt*gamma*J and LU factorize --
        # Algebraic identity with the original W = I/(dt*gamma) - J:
        #   W k = rhs  <=>  (I - dt*gamma*J) k = dt*gamma * rhs
        # so each C_ij/dt coupling becomes gamma*C_ij and each dt*D_i*dT
        # becomes dt^2*gamma*D_i*dT. Removes every division by dt: a
        # rejected trial step with dt ~ 0 gives W' ~ I (well conditioned)
        # instead of manufacturing inf that diffrax's accept/reject where
        # then leaks into reverse-mode cotangents (issue #30, item 5).
        dtgamma = dt * _R5_GAMMA
        W = jnp.eye(n) - dtgamma * J
        lu_piv = jla.lu_factor(W)

        # -- Stage 1 --
        k1 = _lu_solve(lu_piv, dtgamma * (f0 + dt * _R5_D1 * dT))

        # -- Stage 2 --
        u2 = y0 + _R5_A21 * k1
        k2 = _lu_solve(
            lu_piv,
            dtgamma * (f(t0 + _R5_C2 * dt, u2) + dt * _R5_D2 * dT)
            + _R5_GAMMA * _R5_C21 * k1,
        )

        # -- Stage 3 --
        u3 = y0 + _R5_A31 * k1 + _R5_A32 * k2
        k3 = _lu_solve(
            lu_piv,
            dtgamma * (f(t0 + _R5_C3 * dt, u3) + dt * _R5_D3 * dT)
            + _R5_GAMMA * (_R5_C31 * k1 + _R5_C32 * k2),
        )

        # -- Stage 4 --
        u4 = y0 + _R5_A41 * k1 + _R5_A42 * k2 + _R5_A43 * k3
        k4 = _lu_solve(
            lu_piv,
            dtgamma * (f(t0 + _R5_C4 * dt, u4) + dt * _R5_D4 * dT)
            + _R5_GAMMA * (_R5_C41 * k1 + _R5_C42 * k2 + _R5_C43 * k3),
        )

        # -- Stage 5 --
        u5 = y0 + _R5_A51 * k1 + _R5_A52 * k2 + _R5_A53 * k3 + _R5_A54 * k4
        k5 = _lu_solve(
            lu_piv,
            dtgamma * (f(t0 + _R5_C5 * dt, u5) + dt * _R5_D5 * dT)
            + _R5_GAMMA * (_R5_C51 * k1 + _R5_C52 * k2
                           + _R5_C53 * k3 + _R5_C54 * k4),
        )

        # -- Stage 6 (at t0 + dt) --
        u6 = (y0 + _R5_A61 * k1 + _R5_A62 * k2 + _R5_A63 * k3
              + _R5_A64 * k4 + _R5_A65 * k5)
        k6 = _lu_solve(
            lu_piv,
            dtgamma * f(t0 + dt, u6)
            + _R5_GAMMA * (_R5_C61 * k1 + _R5_C62 * k2 + _R5_C63 * k3
                           + _R5_C64 * k4 + _R5_C65 * k5),
        )

        # -- Stage 7 (at t0 + dt, accumulating) --
        u7 = u6 + k6
        k7 = _lu_solve(
            lu_piv,
            dtgamma * f(t0 + dt, u7)
            + _R5_GAMMA * (_R5_C71 * k1 + _R5_C72 * k2 + _R5_C73 * k3
                           + _R5_C74 * k4 + _R5_C75 * k5 + _R5_C76 * k6),
        )

        # -- Stage 8 (at t0 + dt, accumulating — error stage) --
        u8 = u7 + k7
        k8 = _lu_solve(
            lu_piv,
            dtgamma * f(t0 + dt, u8)
            + _R5_GAMMA * (_R5_C81 * k1 + _R5_C82 * k2 + _R5_C83 * k3
                           + _R5_C84 * k4 + _R5_C85 * k5 + _R5_C86 * k6
                           + _R5_C87 * k7),
        )

        # -- Solution and error --
        y1 = u8 + k8
        y_error = k8

        dense_info = dict(y0=y0, y1=y1)
        return y1, y_error, dense_info, solver_state, RESULTS.successful


class Rodas5Batched(AbstractAdaptiveSolver):
    """Batched Rodas5 for solving multiple k-modes with shared time-stepping.

    y0 has shape ``(batch_size, n_eq)``.  All modes in the batch share the
    same adaptive step size, controlled by a single scalar error norm across
    the batch.  Internally vmaps the Jacobian, LU factorisation and
    back-substitution over the batch dimension.

    Args convention for ``diffeqsolve``::

        args = (f_single, batched_per_mode_data)

    where ``f_single(t, y, per_mode_datum)`` is the single-mode RHS used
    for ``jax.jacfwd``, and ``batched_per_mode_data`` (e.g. an array of
    k-values with shape ``(batch_size,)``) is forwarded to ``terms.vf``
    for batched function evaluations.

    cf. DISCO-EB ``Rodas5Batched`` (ode_integrators_stiff.py:846-1011)
    """

    term_structure = ODETerm
    interpolation_cls = LocalLinearInterpolation

    def order(self, terms):
        return 5

    def error_order(self, terms):
        return 4

    def init(self, terms, t0, t1, y0, args):
        return None

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump

        f, _args = args

        n = y0[0].shape[0]  # state dimension from first batch element
        dt = terms.contr(t0, t1)

        # Pre-compute scaled coupling coefficients.
        # Algebraic identity with the original W = I/(dt*gamma) - J:
        #   W k = rhs  <=>  (I - dt*gamma*J) k = dt*gamma * rhs
        # so each C_ij/dt coupling becomes gamma*C_ij (dt-independent, still
        # hoisted once outside the stages) and each dt*D_i*dT term below
        # moves inside a dtgamma*(...) factor together with its stage's f
        # evaluation. Removes every division by dt: a rejected trial step
        # with dt ~ 0 gives W' ~ I (well conditioned) instead of
        # manufacturing inf that diffrax's accept/reject where then leaks
        # into reverse-mode cotangents (issue #30, item 5).
        gC21 = _R5_GAMMA * _R5_C21
        gC31 = _R5_GAMMA * _R5_C31
        gC32 = _R5_GAMMA * _R5_C32
        gC41 = _R5_GAMMA * _R5_C41
        gC42 = _R5_GAMMA * _R5_C42
        gC43 = _R5_GAMMA * _R5_C43
        gC51 = _R5_GAMMA * _R5_C51
        gC52 = _R5_GAMMA * _R5_C52
        gC53 = _R5_GAMMA * _R5_C53
        gC54 = _R5_GAMMA * _R5_C54
        gC61 = _R5_GAMMA * _R5_C61
        gC62 = _R5_GAMMA * _R5_C62
        gC63 = _R5_GAMMA * _R5_C63
        gC64 = _R5_GAMMA * _R5_C64
        gC65 = _R5_GAMMA * _R5_C65
        gC71 = _R5_GAMMA * _R5_C71
        gC72 = _R5_GAMMA * _R5_C72
        gC73 = _R5_GAMMA * _R5_C73
        gC74 = _R5_GAMMA * _R5_C74
        gC75 = _R5_GAMMA * _R5_C75
        gC76 = _R5_GAMMA * _R5_C76
        gC81 = _R5_GAMMA * _R5_C81
        gC82 = _R5_GAMMA * _R5_C82
        gC83 = _R5_GAMMA * _R5_C83
        gC84 = _R5_GAMMA * _R5_C84
        gC85 = _R5_GAMMA * _R5_C85
        gC86 = _R5_GAMMA * _R5_C86
        gC87 = _R5_GAMMA * _R5_C87

        dtd1 = dt * _R5_D1
        dtd2 = dt * _R5_D2
        dtd3 = dt * _R5_D3
        dtd4 = dt * _R5_D4
        dtd5 = dt * _R5_D5
        dtgamma = dt * _R5_GAMMA

        I = jnp.eye(n)

        # Batched Jacobian and time derivative via forward-mode AD
        dt_f = jax.jacfwd(f, 0)
        jac_f = jax.jacfwd(f, 1)

        dt_f_batched = jax.vmap(dt_f, in_axes=(None, 0, 0))
        jac_f_batched = jax.vmap(jac_f, in_axes=(None, 0, 0))

        lu_batched = jax.vmap(
            lambda a: jla.lu_factor(I - dtgamma * a))

        dT = dt_f_batched(t0, y0, _args)           # (batch, n)
        jac_blocks = jac_f_batched(t0, y0, _args)   # (batch, n, n)

        lu_and_piv = lu_batched(jac_blocks)

        lu_solve_batched = jax.vmap(jla.lu_solve, (0, 0))

        # -- 8 Rosenbrock stages (transformed W-formulation) --
        # Stage 1
        dy1 = terms.vf(t=t0, y=y0, args=_args)
        rhs = dtgamma * (dy1 + dtd1 * dT)
        k1 = lu_solve_batched(lu_and_piv, rhs)

        # Stage 2
        u = y0 + _R5_A21 * k1
        du = terms.vf(t=t0 + _R5_C2 * dt, y=u, args=_args)
        rhs = dtgamma * (du + dtd2 * dT) + gC21 * k1
        k2 = lu_solve_batched(lu_and_piv, rhs)

        # Stage 3
        u = y0 + _R5_A31 * k1 + _R5_A32 * k2
        du = terms.vf(t=t0 + _R5_C3 * dt, y=u, args=_args)
        rhs = dtgamma * (du + dtd3 * dT) + (gC31 * k1 + gC32 * k2)
        k3 = lu_solve_batched(lu_and_piv, rhs)

        # Stage 4
        u = y0 + _R5_A41 * k1 + _R5_A42 * k2 + _R5_A43 * k3
        du = terms.vf(t=t0 + _R5_C4 * dt, y=u, args=_args)
        rhs = dtgamma * (du + dtd4 * dT) + (gC41 * k1 + gC42 * k2 + gC43 * k3)
        k4 = lu_solve_batched(lu_and_piv, rhs)

        # Stage 5
        u = y0 + _R5_A51 * k1 + _R5_A52 * k2 + _R5_A53 * k3 + _R5_A54 * k4
        du = terms.vf(t=t0 + _R5_C5 * dt, y=u, args=_args)
        rhs = dtgamma * (du + dtd5 * dT) + (gC51 * k1 + gC52 * k2
                                             + gC53 * k3 + gC54 * k4)
        k5 = lu_solve_batched(lu_and_piv, rhs)

        # Stage 6 (at t0 + dt)
        u = (y0 + _R5_A61 * k1 + _R5_A62 * k2 + _R5_A63 * k3
             + _R5_A64 * k4 + _R5_A65 * k5)
        du = terms.vf(t=t0 + dt, y=u, args=_args)
        rhs = dtgamma * du + (gC61 * k1 + gC62 * k2 + gC63 * k3
                               + gC64 * k4 + gC65 * k5)
        k6 = lu_solve_batched(lu_and_piv, rhs)

        # Stage 7 (accumulating)
        u = u + k6
        du = terms.vf(t=t0 + dt, y=u, args=_args)
        rhs = dtgamma * du + (gC71 * k1 + gC72 * k2 + gC73 * k3
                               + gC74 * k4 + gC75 * k5 + gC76 * k6)
        k7 = lu_solve_batched(lu_and_piv, rhs)

        # Stage 8 (error estimate stage)
        u = u + k7
        du = terms.vf(t=t0 + dt, y=u, args=_args)
        rhs = dtgamma * du + (gC81 * k1 + gC82 * k2 + gC83 * k3
                               + gC84 * k4 + gC85 * k5 + gC86 * k6
                               + gC87 * k7)
        k8 = lu_solve_batched(lu_and_piv, rhs)

        # Solution and error estimate
        y1 = u + k8
        y_error = k8  # embedded error: (batch_size, n_eq)

        dense_info = dict(y0=y0, y1=y1)
        return y1, y_error, dense_info, None, RESULTS.successful
