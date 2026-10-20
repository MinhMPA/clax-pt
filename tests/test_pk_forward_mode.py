"""Forward-mode (``jax.jvp``) gradient test through ``compute_pk``.

Contract (revised twice after real GPU measurements -- see below):
- ``jax.jvp`` runs end-to-end through background -> thermodynamics ->
  perturbations -> ``compute_pk`` when the perturbation ODE adjoint is
  Diffrax's ``DirectAdjoint`` (``ode_adjoint="direct"``).
- All three AD numbers (jvp under DirectAdjoint, grad under DirectAdjoint,
  grad under the production-default RecursiveCheckpointAdjoint) agree with
  EACH OTHER to within a bound derived from two independent GPU runs, and
  each agrees with central finite differences to within the project's
  documented gradient accuracy target (CLAUDE.md: <1%).

Why not a tight same-pair bound? Two GPU runs at this file's exact
precision block (k=0.1, d/d omega_cdm) gave:
    run 1 (job 13126): grad(recursive)=1.22003960e5, grad(direct)=1.22078431e5,
                        jvp(direct)=1.22003960e5,      FD=1.22087207e5
    run 2 (job 13132, post-rebase): grad(recursive)=1.21951376e5,
                        grad(direct)=1.22076600e5, jvp(direct)=1.22079427e5,
                        FD=1.22066505e5
In run 1, jvp(direct) matched grad(recursive) to all 9 printed significant
figures (rel~0) and grad(direct) was the outlier (6.10e-4 away from both).
In run 2 that FLIPPED: jvp(direct) matched grad(direct) tightly (2.32e-5)
and grad(recursive) was the outlier (~1.05e-3 away from both). There is no
run-stable "tight pair" -- an earlier version of this test asserted
jvp-vs-grad-under-the-same-adjoint tightly, and a different earlier version
asserted jvp-vs-grad(recursive) tightly; both failed on a second run.
Diffrax's adaptive step count and GPU floating-point reduction order are not
bitwise-reproducible run to run, and DirectAdjoint's forward vs reverse
passes evidently trade which one lands closer to which other estimate by an
amount comparable to that non-determinism (~1e-3 relative). This is NOT
evidence of a custom_vjp/adjoint-transposition BUG (clax has zero
custom_vjp rules to begin with -- see below); it is the AD/solver precision
floor at this precision block. So: assert mutual agreement among all three
at a bound with real margin above the observed max spread (1.05e-3, from
run 2) rather than assuming any particular pairing is the reliable one.
FD agreement is comfortably inside <1% in both runs (max 0.094%, run 2
grad(recursive) vs FD) and is not the binding constraint either time.

These numbers also differ by ~0.4% from an earlier independently-quoted
reference set (grad_recursive=1.21474055e5, jvp/grad agreement ~3e-6) --
that reference was evidently produced under a not-quite-identical precision
block or software/driver stack; it is not reproduced exactly here and this
file does not force it to match, per the task's own instruction to report
such a discrepancy rather than adjust the expected numbers to fit.

UPDATE (Oct 11, 2026, smsharma/clax#48): the two GPU runs above used the
unfixed recombination (clipped explicit helium/hydrogen steps whose "chatter",
a zigzag from grid point to grid point, made derivatives depend on step
placement). On the fixed code (Saha -> ODE staging, ADR 0003; CPU, job 24323)
the three AD numbers are grad(recursive)=1.22116400e5, grad(direct)=1.22116401e5,
jvp(direct)=1.22116400e5, equal to 1e-8, and the jvp agrees between CPU and GPU
to 1.6e-7 (jobs 24514 CPU, 24516 GPU); the three-way spread was measured on CPU
only. The 5e-3 mutual bar is unchanged.

FD=1.21310978e5, so AD sits 0.66% above FD. That offset is pre-existing and was
hidden by finite-difference noise (diagnosis jobs 24513-24517). On the unfixed
tree central FD of dP(k=0.1)/d omega_cdm scatters 1.06% across steps 1e-4 to 2e-3
(std 0.44%; jobs 24515 CPU, 24517 GPU); its mean, 1.21312e5, equals the fixed
tree's slope to 1.6e-5, and AD sits +0.64% (CPU) / +0.56% (GPU) above that mean.
On the fixed tree FD is converged (std 0.007%; jobs 24514 CPU, 24516 GPU,
agreeing to 2e-8) and AD is a steady +0.66%. The fix therefore exposed the bias by
smoothing the primal; it did not create it. The 0.008%-0.094% agreement in the
GPU runs above reflected one lucky FD step. In units of d ln P/d ln omega_cdm the
bias is 0.0092, about 2.3x the ~0.004 quoted in smsharma/clax#45, whose
measurement conditions were not recorded. The kappa_dot stop_gradient (#45) is a
candidate cause, untested, and probably not the sole one (one cosmology, one k,
one parameter, one precision block). The 1% FD bar passes with a 0.34-point
margin.

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

Note for the record: at the time this file was written, clax defined ZERO
``jax.custom_vjp`` rules — all four custom-AD-rule sites
(``thermodynamics.py`` x2, ``shooting.py``, ``perturbations.py``) were
``jax.custom_jvp``, which supports forward mode directly, so the
forward-mode blocker probed here was entirely inside ``diffrax``'s
``RecursiveCheckpointAdjoint`` implementation. Since the issue #30 fix,
clax has exactly ONE ``jax.custom_vjp``: the reverse-mode-stable fused
solve ``thermodynamics.solve_background_and_thermo`` routes through it when
``PrecisionParams.th_grad_mode == "stable"`` (the default), and that rule —
like any ``custom_vjp`` — blocks ``jax.jvp``. This file therefore sets
``th_grad_mode="native"`` in its precision block to keep exercising the
plain forward path end-to-end (the documented escape hatch for
forward-mode users; cf. tests/test_thermo_reverse_composite.py).

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
    # jax.jvp cannot cross the custom_vjp of the stable fused bg+thermo
    # solve (issue #30 fix); "native" keeps the plain forward path this
    # file exists to probe. See module docstring, "Note for the record".
    th_grad_mode="native",
)
_PROBE_PREC_RECURSIVE = dataclasses.replace(
    _PROBE_PREC_DIRECT, ode_adjoint="recursive_checkpoint",
)

_K_TARGET = 0.1
_PARAM_NAME = "omega_cdm"
_FD_STEP = 1.0e-3  # matches diags/diag_grad_*.py FD_STEPS["omega_cdm"]

# Bound for mutual AD agreement: ~4.8x the max pairwise spread observed
# across two independent GPU runs (1.05e-3, run 2 jvp-vs-grad(recursive)).
# See module docstring for both runs' full numbers.
_AD_MUTUAL_REL_TOL = 5.0e-3
# AD against central FD: 2x the worst |jvp/FD - 1| over the five LCDM grid points
# at steps 1e-3 and 5e-4 on the fixed tree (8.6e-4, lcdm_fiducial and ns_high at
# 5e-4; the FD itself moves by 6e-4 to 1e-3 between the two steps at this
# precision block), ceiled to two figures (job 27177, Oct 20, 2026). On main the
# same quantity sat +0.59% to +0.76% above FD at every grid point (job 27176):
# the opacity tables' stop_gradient cut, smsharma/clax#55, ADR 0004. The
# CLAUDE.md target is 1%.
_FD_REL_TOL = 1.8e-3


class TestComputePkForwardMode:
    """``jax.jvp`` through the full ``compute_pk`` pipeline -- the coverage
    gap this module closes (no forward-mode test previously existed anywhere
    through ``compute_pk`` / perturbations / C_l).
    """

    @pytest.mark.slow
    def test_jvp_matches_grad_and_fd_for_omega_cdm(self, fast_mode, lcdm_cosmology):
        """``jvp(compute_pk)``, ``grad(compute_pk)`` (DirectAdjoint), and
        ``grad(compute_pk)`` (RecursiveCheckpointAdjoint) mutually agree
        (<0.5%, see module docstring for why this bound and not a tighter
        same-pair one), and each agrees with central finite differences
        (``_FD_REL_TOL``, set from the measurement above), at
        ``k=0.1 Mpc^-1``, ``d/d omega_cdm``, at the five LCDM grid points
        (multi-cosmology rule; smsharma/clax#55).

        This is a dedicated 4-solve AD probe per cosmology (grad-recursive,
        grad-direct, jvp-direct, central-FD), not a --fast-subsamplable
        sweep, so it is full-mode only.
        """
        if fast_mode:
            pytest.skip("heavy dedicated AD probe -- full mode only, see docstring")

        _, params = lcdm_cosmology
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

        # Mutual AD agreement: all three pairwise, no assumption about which
        # pair is "the tight one" (see module docstring -- that assumption
        # was tested and refuted across two runs).
        ad_values = {
            "jvp(direct)": jvp_val,
            "grad(recursive)": grad_recursive,
            "grad(direct)": grad_direct,
        }
        for name_a, name_b in (
            ("jvp(direct)", "grad(recursive)"),
            ("jvp(direct)", "grad(direct)"),
            ("grad(direct)", "grad(recursive)"),
        ):
            a, b = ad_values[name_a], ad_values[name_b]
            rel = abs(a - b) / (abs(b) + 1e-30)
            assert rel < _AD_MUTUAL_REL_TOL, (
                f"{name_a}={a:.8e} vs {name_b}={b:.8e} rel={rel:.2e} "
                f"(expected <{_AD_MUTUAL_REL_TOL:.1e}; measured max spread "
                f"1.05e-3 across two GPU runs, see module docstring)"
            )

        # All three AD numbers vs finite differences (measured 0.008%-0.094%
        # across two runs, so this is not the binding constraint).
        for name, val in ad_values.items():
            rel_fd = abs(val - fd) / (abs(fd) + 1e-30)
            assert rel_fd < _FD_REL_TOL, (
                f"{name}={val:.8e} vs central FD={fd:.8e} rel={rel_fd:.2%} "
                f"(expected <{_FD_REL_TOL:.2%}, 2x the measured worst; see _FD_REL_TOL)"
            )
