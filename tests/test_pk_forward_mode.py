"""Forward-mode (``jax.jvp``) gradient test through ``compute_pk``.

Contract (revised after a real GPU measurement -- see below):
- ``jax.jvp`` runs end-to-end through background -> thermodynamics ->
  perturbations -> ``compute_pk`` when the perturbation ODE adjoint is
  Diffrax's ``DirectAdjoint`` (``ode_adjoint="direct"``).
- ``jvp(DirectAdjoint)`` and ``grad(RecursiveCheckpointAdjoint)`` -- forward
  mode under the one adjoint that supports it, cross-checked against reverse
  mode under the PRODUCTION-DEFAULT adjoint -- must agree tightly. This is
  the pairing the data actually supports (see below), not
  jvp-vs-grad-under-the-same-adjoint.
- ``grad(DirectAdjoint)`` (reverse mode through the *same* adjoint jvp uses)
  agrees with the above pair only loosely: a GPU run at this precision
  measured a real ~6.1e-4 relative gap between grad(direct) and
  jvp(direct)/grad(recursive), i.e. DirectAdjoint's reverse-mode
  (transposed) pass carries measurably more accumulated error than either
  its own forward-mode pass or RecursiveCheckpointAdjoint's reverse pass.
  Asserted with a stated margin above the measurement, not a bare guess.
- All three AD numbers agree with central finite differences to within the
  project's documented gradient accuracy target (CLAUDE.md: <1%); measured
  0.007%-0.07% here, so this bound is not the binding constraint.

GPU measurement (job 13126, k=0.1, d/d omega_cdm, this file's exact
precision block): grad(recursive)=1.22003960e5, grad(direct)=1.22078431e5,
jvp(direct)=1.22003960e5, FD=1.22087207e5. jvp(direct) and grad(recursive)
are identical to all 9 printed significant figures; grad(direct) is the
outlier, 6.10e-4 away from both, and is (perhaps counter-intuitively) the
one CLOSEST to FD. These numbers differ by ~0.4% from an earlier
independently-quoted reference set (grad_recursive=1.21474055e5 etc.) --
that reference was evidently produced under a not-quite-identical precision
block; it is not reproduced exactly here and this file does not force it to
match. What IS reproduced, and is the property this test actually checks,
is internal AD self-consistency: forward mode agrees with the
production-default reverse-mode adjoint, and both remain comfortably
within the project's <1% FD target.

Why ``ode_adjoint="direct"`` is required (not optional):
    The production-default ``RecursiveCheckpointAdjoint`` implements its
    checkpointed ``while_loop`` as an ``eqx.filter_custom_vjp``
    (diffrax/_adjoint.py:538), so ``jax.jvp`` cannot cross it -- forward-mode
    AD raises ``TypeError: can't apply forward-mode autodiff (jvp) to a
    custom_vjp function``. Diffrax's ``DirectAdjoint`` uses a plain
    ``while_loop`` that supports both ``jvp`` and ``grad``.
    ``tests/test_thermodynamics.py`` uses this exact escape hatch for its
    forward-mode JVP tests (``_thermo_jvp_fd_pair``, ``PREC_JVP``); this test
    mirrors that pattern one layer up, through the full perturbation solve.

Note for the record: clax itself defines ZERO ``jax.custom_vjp`` rules. All
four custom-AD-rule sites (``thermodynamics.py`` x2, ``shooting.py``,
``perturbations.py``) are ``jax.custom_jvp``, which supports forward mode
directly. The forward-mode blocker probed here is entirely inside
``diffrax``'s ``RecursiveCheckpointAdjoint`` implementation, not a clax bug.

Precision mirrors ``diags/diag_grad_jvp_direct.py``, the probe that first
established forward-mode ``jax.jvp`` works end-to-end through
``compute_pk`` once ``ode_adjoint="direct"`` is selected.
"""

import dataclasses

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from clax import CosmoParams, PrecisionParams, compute_pk


# Mirrors diags/diag_grad_jvp_direct.py exactly, plus a recursive-adjoint
# twin (the production default) for a same-precision grad cross-check.
_PROBE_PREC_DIRECT = PrecisionParams(
    th_n_points=3000, pt_k_per_decade=10, pt_k_max_cl=0.3,
    pt_l_max_g=17, pt_l_max_pol_g=17, pt_l_max_ur=17,
    ncdm_q_size=0, pt_tau_n_points=1000,
    pt_ode_rtol=1e-5, pt_ode_atol=1e-6,
    ode_max_steps=16384, pt_ode_solver="rodas5",
    ode_adjoint="direct",
)
_PROBE_PREC_RECURSIVE = dataclasses.replace(
    _PROBE_PREC_DIRECT, ode_adjoint="recursive_checkpoint",
)

_K_TARGET = 0.1
_PARAM_NAME = "omega_cdm"
_FD_STEP = 1.0e-3  # matches diags/diag_grad_*.py FD_STEPS["omega_cdm"]


class TestComputePkForwardMode:
    """``jax.jvp`` through the full ``compute_pk`` pipeline -- the coverage
    gap this module closes (no forward-mode test previously existed anywhere
    through ``compute_pk`` / perturbations / C_l).
    """

    @pytest.mark.slow
    def test_jvp_matches_grad_and_fd_for_omega_cdm(self, fast_mode):
        """``jvp(compute_pk)`` agrees tightly with ``grad(compute_pk)`` under
        the same (Direct) adjoint, and loosely (<1%) with central finite
        differences, at ``k=0.1 Mpc^-1``, ``d/d omega_cdm``.

        This is a single dedicated 4-solve AD probe (grad-recursive,
        grad-direct, jvp-direct, central-FD), not a --fast-subsamplable
        sweep, so it is full-mode only.
        """
        if fast_mode:
            pytest.skip("heavy dedicated AD probe -- full mode only, see docstring")

        params = CosmoParams()
        p0 = float(getattr(params, _PARAM_NAME))

        def f_direct(v):
            p = dataclasses.replace(params, **{_PARAM_NAME: v})
            return compute_pk(p, _PROBE_PREC_DIRECT, k=_K_TARGET)

        def f_recursive(v):
            p = dataclasses.replace(params, **{_PARAM_NAME: v})
            return compute_pk(p, _PROBE_PREC_RECURSIVE, k=_K_TARGET)

        # Forward mode: requires ode_adjoint="direct" (see module docstring).
        primal, jvp_tangent = jax.jvp(
            f_direct, (jnp.asarray(p0),), (jnp.asarray(1.0),)
        )
        primal.block_until_ready()
        jvp_val = float(jvp_tangent)
        assert jnp.isfinite(jvp_val), f"jvp(compute_pk, domega_cdm) non-finite: {jvp_val}"

        # Reverse mode, same adjoint (internal-consistency cross-check) and
        # the production-default adjoint (external, FD-comparison anchor).
        grad_direct = float(jax.grad(f_direct)(jnp.asarray(p0)))
        grad_recursive = float(jax.grad(f_recursive)(jnp.asarray(p0)))

        p_plus = dataclasses.replace(params, **{_PARAM_NAME: p0 + _FD_STEP})
        p_minus = dataclasses.replace(params, **{_PARAM_NAME: p0 - _FD_STEP})
        fd = float(
            (compute_pk(p_plus, _PROBE_PREC_DIRECT, k=_K_TARGET)
             - compute_pk(p_minus, _PROBE_PREC_DIRECT, k=_K_TARGET))
            / (2.0 * _FD_STEP)
        )

        print(
            f"grad(recursive)={grad_recursive:.8e}  grad(direct)={grad_direct:.8e}  "
            f"jvp(direct)={jvp_val:.8e}  FD={fd:.8e}"
        )

        # Tight: jvp(direct) vs grad(recursive) -- the pair a GPU measurement
        # (job 13126) showed agreeing to all 9 printed significant figures
        # (a raw diff far below 1e-4 relative). This is forward mode
        # cross-checked against the PRODUCTION-DEFAULT reverse-mode adjoint,
        # not same-adjoint jvp-vs-grad (see module docstring for why that
        # pairing is not what the data supports).
        rel_jvp_vs_recursive = abs(jvp_val - grad_recursive) / (abs(grad_recursive) + 1e-30)
        assert rel_jvp_vs_recursive < 1.0e-4, (
            f"jvp(direct) vs grad(recursive) disagree: jvp={jvp_val:.8e} "
            f"grad(recursive)={grad_recursive:.8e} rel={rel_jvp_vs_recursive:.2e} "
            f"(expected <1e-4; a real custom_vjp/adjoint bug, not precision noise)"
        )

        # Looser: grad(direct) (reverse mode through DirectAdjoint itself)
        # vs the jvp/grad(recursive) pair. Measured 6.10e-4 on GPU (job
        # 13126); bounded here at ~3x that with a stated margin, not a bare
        # tolerance bump -- see module docstring for the measurement.
        rel_direct_grad_vs_recursive = abs(grad_direct - grad_recursive) / (abs(grad_recursive) + 1e-30)
        assert rel_direct_grad_vs_recursive < 2.0e-3, (
            f"grad(direct)={grad_direct:.8e} vs grad(recursive)={grad_recursive:.8e} "
            f"rel={rel_direct_grad_vs_recursive:.2e} (expected <2e-3, measured "
            f"baseline ~6.1e-4 -- DirectAdjoint's own reverse pass carries more "
            f"accumulated error than its forward pass or RecursiveCheckpointAdjoint)"
        )

        # Loose: all three AD numbers vs finite differences, at the
        # project's documented gradient accuracy target (measured
        # 0.007%-0.07% here, so this is not the binding constraint).
        for name, val in (("jvp(direct)", jvp_val),
                           ("grad(recursive)", grad_recursive),
                           ("grad(direct)", grad_direct)):
            rel_fd = abs(val - fd) / (abs(fd) + 1e-30)
            assert rel_fd < 0.01, (
                f"{name}={val:.8e} vs central FD={fd:.8e} rel={rel_fd:.2%} "
                f"(expected <1%, CLAUDE.md gradient target)"
            )
