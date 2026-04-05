"""Tests selected linear-matter power-spectrum gradient behavior.

Contract:
- Stable ``P(k)`` derivative contracts computed with AD agree with finite-difference estimates.

Scope:
- Covers scalar ``P(k)`` gradients for well-conditioned parameters.
- Excludes forward ``P(k)`` value-accuracy checks owned by ``test_pk_accuracy.py``.

Notes:
- These tests are intentionally isolated because reverse-mode checks are much more expensive than forward checks.
- Density-parameter ``P(k)`` gradients under the coarse test preset remain diagnostic rather than gating contracts.
"""

import jax
jax.config.update("jax_enable_x64", True)

import pytest

import clax
from clax import CosmoParams, PrecisionParams

# ---------------------------------------------------------------------------
# Precision settings
# ---------------------------------------------------------------------------

# Scalar gradient tests: single k-mode ODE via compute_pk().
# Matches test_end_to_end.py PREC_FAST which is known to work.
PREC_GRAD = PrecisionParams(
    bg_n_points=200, ncdm_bg_n_points=100, bg_tol=1e-8,
    th_n_points=5000, th_z_max=5e3,
    pt_l_max_g=17, pt_l_max_pol_g=17, pt_l_max_ur=17,
    pt_k_max_cl=0.3,
    pt_ode_rtol=1e-3, pt_ode_atol=1e-6,
    ode_max_steps=262144,
)

# ---------------------------------------------------------------------------
# Scalar gradient tests
# ---------------------------------------------------------------------------

class TestPkScalarGradients:
    """Tests stable scalar ``P(k)`` gradient contracts."""

    # Subset of params to run in --fast mode
    _FAST_PARAMS = {"ln10A_s", "n_s"}

    @pytest.mark.slow
    @pytest.mark.parametrize("param_name,fiducial,eps,k_test", [
        ("ln10A_s",   3.0445224377,    1e-4, 0.05),
        ("n_s",       0.9649,          1e-4, 0.01),  # k != k_pivot for non-trivial tilt
    ])
    def test_gradient(self, param_name, fiducial, eps, k_test, fast_mode):
        """``dP/dparam`` matches central finite differences; expects <5% relative error."""
        if fast_mode and param_name not in self._FAST_PARAMS:
            pytest.skip(f"{param_name} skipped in fast mode")

        def pk_fn(val):
            return clax.compute_pk(CosmoParams(**{param_name: val}), PREC_GRAD, k=k_test)

        grad_ad = float(jax.grad(pk_fn)(fiducial))
        grad_fd = float((pk_fn(fiducial + eps) - pk_fn(fiducial - eps)) / (2.0 * eps))
        rel_err = abs(grad_ad - grad_fd) / (abs(grad_fd) + 1e-30)

        assert rel_err < 0.05, (
            f"d(P(k={k_test}))/d({param_name}): "
            f"AD={grad_ad:.4e} FD={grad_fd:.4e} rel_err={rel_err:.2%}"
        )
