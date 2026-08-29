"""Reverse-mode-stable fused background+thermodynamics gradients (issue #30).

Bug being fixed (see https://github.com/smsharma/clax/issues/30): reverse-mode
AD (``jax.grad``) through ``thermodynamics_solve`` carries a deterministic
error from catastrophic floating-point cancellation in the recombination-era
backward pass. The Peebles/RECFAST rates contain Boltzmann-exponential ratios
(``exp(B/kT) ~ e^52``), so intermediates reach ~1e13. Forward mode pairs
huge x tiny factors locally per grid point (nothing large is ever formed, so
``jax.jvp`` was verified exact against FD six independent ways); reverse mode
contracts thousands of +-1e13-scale cotangent terms into shared scalars whose
true total is ~1e-3 or exactly 0, leaving exact-ULP residue (e.g. 2^-9).
Measured upstream symptoms: end-to-end d(sum(pk_mm_real))/dh = 4.107387e6 vs
truth 4.029578e6 (+1.9%); thermo-chain reverse probe 8.66e7 vs true 1.16e5
(749x); on this file's CPU precision block, grad/jvp of sum(g^2) w.r.t. h
disagree by a factor ~52 (measured 2026-08-29, login-node probe).

The fix under test (issue #30 fix option 2): a fused entry point
``clax.thermodynamics.solve_background_and_thermo(params, prec)
-> (bg, th)`` whose only differentiable input is ``CosmoParams``, wrapped in
``jax.custom_vjp`` and gated by the static flag
``PrecisionParams.th_grad_mode``:

- ``"stable"`` (default): the backward pass computes the params cotangent for
  BOTH outputs via a batched forward-mode basis (``jax.jacfwd`` over the ~20
  traced CosmoParams leaves) -- "vjp-through-jvp". Mathematically identical to
  the native VJP (J^T ct vs <ct, J e_i>: pure re-association of the same
  arithmetic), numerically stable because the forward pass is proven exact.
- ``"native"``: plain ``background_solve`` + ``thermodynamics_solve`` calls.
  Required by forward-mode users: a ``jax.custom_vjp`` function rejects
  ``jax.jvp`` by construction.

Consistency criterion used throughout: grad (stable) must match jvp (native
forward mode, ``ode_adjoint="direct"``) -- jvp is the proven-exact oracle.
"""

import dataclasses
import inspect

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

import clax
from clax import CosmoParams, PrecisionParams
from clax.background import background_solve
from clax.thermodynamics import solve_background_and_thermo, thermodynamics_solve


# ---------------------------------------------------------------------------
# CPU-affordable precision block
# ---------------------------------------------------------------------------
# Mirrors tests/test_thermodynamics.py::PREC_JVP (the established CPU jvp
# block) with th_n_points=6000. ode_adjoint="direct" in the BASE block so the
# grad and jvp arms share the identical primal/derivative graph (DirectAdjoint
# supports both AD modes; RecursiveCheckpointAdjoint blocks jvp).
_PREC_STABLE = PrecisionParams(
    bg_n_points=400, ncdm_bg_n_points=200, bg_tol=1e-8,
    th_n_points=6000, th_z_max=5e4,  # 5e4 floor: see PrecisionParams.th_z_max
    ode_adjoint="direct",
    th_grad_mode="stable",
)
_PREC_NATIVE = dataclasses.replace(_PREC_STABLE, th_grad_mode="native")

_PARAMS = CosmoParams()

# grad(stable) vs jvp(native) agreement bound (acceptance criterion, issue #30
# handoff). Measured on the prototype (login node, CPU, 2026-08-29): worst
# case 8.1e-15 across {sum_xe2, randlin, sum_g2} x {h, omega_b}. The native
# reverse path measured 5.1e-11 (sum_xe2/h) and 5.2e+1 (sum_g2/h) on the same
# block -- the bound separates fixed from broken by >5 orders of magnitude in
# the pathological case.
_GRAD_JVP_REL_TOL = 1e-6


def _fused_functional(pname, functional, prec):
    """f(v) = functional(bg, th) through the fused solve with pname = v."""
    def f(v):
        p = dataclasses.replace(_PARAMS, **{pname: v})
        bg, th = solve_background_and_thermo(p, prec)
        return functional(bg, th)
    return f


@pytest.fixture(scope="module")
def native_primal():
    """One shared native primal solve: (bg, th, random table weights)."""
    bg = background_solve(_PARAMS, _PREC_NATIVE)
    th = thermodynamics_solve(_PARAMS, _PREC_NATIVE, bg)
    n_table = int(th.xe_of_loga.y.shape[0])
    rng = np.random.default_rng(20260829)
    weights = {
        name: jnp.asarray(rng.standard_normal(n_table))
        for name in ("xe", "Tb", "g", "kd", "cs2")
    }
    return bg, th, weights


def _make_functionals(weights):
    """Scalar functionals of the fused output.

    - sum_xe2: smooth, well-scaled derivative (O(1e5) in h).
    - randlin: fixed random-weighted linear functional across five tables plus
      two derived scalars -- a stand-in for an arbitrary downstream cotangent,
      the pattern that exposed the 749x thermo-chain reverse error upstream.
    - sum_g2: near-zero derivative in h (~5e-20); the native reverse residue
      (~2.7e-18, an exact-ULP quantum) exceeds it by ~52x on CPU. The stable
      path reproduces the jvp to ~1e-16 relative.
    """
    return {
        "sum_xe2": lambda bg, th: jnp.sum(th.xe_of_loga.y ** 2),
        "randlin": lambda bg, th: (
            jnp.sum(weights["xe"] * th.xe_of_loga.y)
            + jnp.sum(weights["Tb"] * th.Tb_of_loga.y)
            + jnp.sum(weights["g"] * th.g_of_loga.y)
            + jnp.sum(weights["kd"] * th.kappa_dot_of_loga.y)
            + jnp.sum(weights["cs2"] * th.cs2_of_loga.y)
            + th.z_star + th.rs_star
        ),
        "sum_g2": lambda bg, th: jnp.sum(th.g_of_loga.y ** 2),
    }


# ---------------------------------------------------------------------------
# Contract of the fused entry point and the th_grad_mode flag
# ---------------------------------------------------------------------------

class TestFusedSolveContract:
    def test_default_th_grad_mode_is_stable(self):
        """Reverse-mode users get the stable rule without opting in."""
        assert PrecisionParams().th_grad_mode == "stable"

    def test_invalid_th_grad_mode_raises(self):
        bad = dataclasses.replace(_PREC_STABLE, th_grad_mode="typo")
        with pytest.raises(ValueError, match="th_grad_mode"):
            solve_background_and_thermo(_PARAMS, bad)

    def test_native_mode_matches_separate_solves(self, native_primal):
        """th_grad_mode="native" is exactly the pre-existing two-call path."""
        bg_ref, th_ref, _ = native_primal
        bg, th = solve_background_and_thermo(_PARAMS, _PREC_NATIVE)
        for got, ref, name in ((bg, bg_ref, "bg"), (th, th_ref, "th")):
            got_leaves = jax.tree_util.tree_leaves(got)
            ref_leaves = jax.tree_util.tree_leaves(ref)
            assert len(got_leaves) == len(ref_leaves)
            for i, (a, b) in enumerate(zip(got_leaves, ref_leaves)):
                np.testing.assert_array_equal(
                    np.asarray(a), np.asarray(b),
                    err_msg=f"{name} leaf {i} differs between fused-native "
                            f"and separate solves",
                )

    def test_stable_primal_matches_native(self, native_primal):
        """The custom_vjp wrapper must not perturb the primal computation."""
        bg_ref, th_ref, _ = native_primal
        bg, th = solve_background_and_thermo(_PARAMS, _PREC_STABLE)
        for got, ref, name in ((bg, bg_ref, "bg"), (th, th_ref, "th")):
            for i, (a, b) in enumerate(zip(
                    jax.tree_util.tree_leaves(got),
                    jax.tree_util.tree_leaves(ref))):
                np.testing.assert_array_equal(
                    np.asarray(a), np.asarray(b),
                    err_msg=f"{name} leaf {i} differs between stable and "
                            f"native primal",
                )

    def test_stable_blocks_jvp_native_allows(self, native_primal):
        """Documents the custom_vjp caveat: jax.jvp cannot cross a custom_vjp
        function, so forward-mode users must set th_grad_mode="native"
        (tests/test_pk_forward_mode.py does exactly this).
        """
        _, _, weights = native_primal
        functional = _make_functionals(weights)["sum_xe2"]
        h0 = jnp.asarray(_PARAMS.h)
        one = jnp.asarray(1.0)

        f_native = _fused_functional("h", functional, _PREC_NATIVE)
        _, tangent = jax.jvp(f_native, (h0,), (one,))
        assert np.isfinite(float(tangent))

        f_stable = _fused_functional("h", functional, _PREC_STABLE)
        with pytest.raises(TypeError, match="custom_vjp"):
            jax.jvp(f_stable, (h0,), (one,))


# ---------------------------------------------------------------------------
# Thermo-level grad-vs-jvp consistency (acceptance criterion 1a)
# ---------------------------------------------------------------------------

class TestThermoLevelConsistency:
    """grad through the stable fused solve == jvp (the exact oracle)."""

    @pytest.mark.parametrize("pname", ["h", "omega_b"])
    @pytest.mark.parametrize("fname", ["sum_xe2", "randlin"])
    def test_stable_grad_matches_jvp(self, native_primal, fast_mode,
                                     pname, fname):
        """Well-scaled functionals, d/dh and d/domega_b: rel gap < 1e-6.

        Measured (CPU, 2026-08-29): stable 3.7e-15..8.1e-15; native reverse on
        the same block: 5.1e-11 (h) / 1.6e-10 (omega_b) -- and the same native
        pathology reaches +1.9% end-to-end on GPU (issue #30).
        """
        if fast_mode and (pname, fname) != ("h", "sum_xe2"):
            pytest.skip("--fast: run one representative (pname, functional)")
        _, _, weights = native_primal
        functional = _make_functionals(weights)[fname]
        p0 = jnp.asarray(float(getattr(_PARAMS, pname)))

        f_native = _fused_functional(pname, functional, _PREC_NATIVE)
        _, jvp_tan = jax.jvp(f_native, (p0,), (jnp.asarray(1.0),))
        f_stable = _fused_functional(pname, functional, _PREC_STABLE)
        grad_stable = jax.grad(f_stable)(p0)

        jvp_val, grad_val = float(jvp_tan), float(grad_stable)
        rel = abs(grad_val - jvp_val) / (abs(jvp_val) + 1e-30)
        assert rel < _GRAD_JVP_REL_TOL, (
            f"d({fname})/d({pname}): grad(stable)={grad_val:.10e} vs "
            f"jvp={jvp_val:.10e} rel={rel:.3e} (expected <{_GRAD_JVP_REL_TOL:.0e})"
        )

    def test_stable_grad_matches_jvp_visibility_pathology(self, native_primal):
        """f = sum(g^2), d/dh: the near-zero-derivative case where the native
        reverse residue dominates the true value (grad/jvp ratio ~52 measured
        on this exact block, CPU, 2026-08-29 -- the issue #30 ULP-quantum
        pathology in miniature). The stable rule reproduces the jvp to
        machine precision (measured 1.2e-16 relative).
        """
        _, _, weights = native_primal
        functional = _make_functionals(weights)["sum_g2"]
        h0 = jnp.asarray(_PARAMS.h)

        f_native = _fused_functional("h", functional, _PREC_NATIVE)
        _, jvp_tan = jax.jvp(f_native, (h0,), (jnp.asarray(1.0),))
        f_stable = _fused_functional("h", functional, _PREC_STABLE)
        grad_stable = jax.grad(f_stable)(h0)

        jvp_val, grad_val = float(jvp_tan), float(grad_stable)
        # Floor 1e-25 (not 1e-30): the true derivative is ~5e-20; the floor
        # stays far below it while guarding an exact-zero jvp.
        rel = abs(grad_val - jvp_val) / (abs(jvp_val) + 1e-25)
        assert rel < _GRAD_JVP_REL_TOL, (
            f"d(sum(g^2))/dh: grad(stable)={grad_val:.10e} vs "
            f"jvp={jvp_val:.10e} rel={rel:.3e} (expected <{_GRAD_JVP_REL_TOL:.0e}; "
            f"native reverse measured rel~5.2e+1 on this block)"
        )


# ---------------------------------------------------------------------------
# Pipeline routing
# ---------------------------------------------------------------------------

class TestPipelineRouting:
    """The main pipeline entry points route bg+th through the fused solve.

    Source-level wiring check (same affordable pattern as
    test_divergence_guard.py::TestGuardWiring): a future refactor that
    silently reverts a call site to the separate two-call path would
    reintroduce the issue #30 reverse-mode error without failing any
    numerical test on CPU.
    """

    @pytest.mark.parametrize("func_name", ["compute", "compute_pk_table",
                                           "compute_pk"])
    def test_entry_point_calls_fused_solve(self, func_name):
        src = inspect.getsource(getattr(clax, func_name))
        assert "solve_background_and_thermo(" in src, (
            f"clax.{func_name} no longer routes through "
            f"solve_background_and_thermo -- issue #30 reverse-mode "
            f"stability would silently regress"
        )
        assert "thermodynamics_solve(" not in src, (
            f"clax.{func_name} still calls thermodynamics_solve directly"
        )

    def test_compute_primal_equal_across_modes(self):
        """clax.compute() primal is bitwise-independent of th_grad_mode."""
        res_stable = clax.compute(_PARAMS, _PREC_STABLE)
        res_native = clax.compute(_PARAMS, _PREC_NATIVE)
        for a, b in zip(jax.tree_util.tree_leaves(res_stable),
                        jax.tree_util.tree_leaves(res_native)):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


# ---------------------------------------------------------------------------
# Pipeline-level grad-vs-jvp consistency (acceptance criterion 1b; GPU-scale)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestPipelineDeltaM:
    def test_pipeline_delta_m_grad_matches_jvp(self):
        """g(h) = sum(pt.delta_m[:, -1]^2) at fast_cl(pt_k_max_cl=5,
        pt_k_chunk_size=20): grad (stable, production RecursiveCheckpoint
        adjoint for the perturbation stage) vs jvp (native, DirectAdjoint,
        ode_max_steps=16384 -- the proven forward-mode escape hatch, cf.
        tests/test_pk_forward_mode.py) must agree to < 1e-4.

        Upstream reference numbers (issue #30, GPU jobs 13924-13946):
        jvp = -7.96748395e10, native grad = -7.93444874e10 (rel gap +4.1e-3);
        the freeze-th diagnostic achieved +6.8e-7, so cross-arm adaptive-step
        noise is ~7e-7 and the 1e-4 bound has >100x margin over it while
        sitting 40x below the native disease.

        GPU-scale: ~10-30 min on V100. Run via sbatch, not on a login node.
        """
        from clax.perturbations import perturbations_solve

        prec_grad = dataclasses.replace(
            PrecisionParams.fast_cl(),
            pt_k_max_cl=5.0, pt_k_chunk_size=20,
            th_grad_mode="stable",
        )
        prec_jvp = dataclasses.replace(
            prec_grad, th_grad_mode="native",
            ode_adjoint="direct", ode_max_steps=16384,
        )

        def g_of_h(prec):
            def g(h):
                p = dataclasses.replace(_PARAMS, h=h)
                bg, th = solve_background_and_thermo(p, prec)
                pt = perturbations_solve(p, prec, bg, th)
                return jnp.sum(pt.delta_m[:, -1] ** 2)
            return g

        h0 = jnp.asarray(_PARAMS.h)
        _, jvp_tan = jax.jvp(g_of_h(prec_jvp), (h0,), (jnp.asarray(1.0),))
        grad_stable = jax.grad(g_of_h(prec_grad))(h0)

        jvp_val, grad_val = float(jvp_tan), float(grad_stable)
        rel = abs(grad_val - jvp_val) / (abs(jvp_val) + 1e-30)
        print(f"delta_m pipeline: jvp={jvp_val:.8e} grad(stable)={grad_val:.8e} "
              f"relgap={rel:.3e}")
        assert rel < 1e-4, (
            f"pipeline d(sum(delta_m[:,-1]^2))/dh: grad(stable)={grad_val:.8e} "
            f"vs jvp={jvp_val:.8e} rel={rel:.3e} (expected <1e-4; native "
            f"reverse measured +4.1e-3, issue #30)"
        )
