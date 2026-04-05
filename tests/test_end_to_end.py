"""Tests public API smoke behavior.

Contract:
- Top-level public entrypoints execute and return structurally sane outputs.

Scope:
- Covers lightweight smoke checks for ``clax.compute()`` and ``clax.compute_pk()``.
- Excludes reference-data accuracy and gradient contracts owned by dedicated files.

Notes:
- These tests are intentionally cheap and do not own physics-accuracy guarantees.
"""

import jax
jax.config.update("jax_enable_x64", True)

import clax
from clax import CosmoParams, PrecisionParams


# Low-res precision for speed
PREC_FAST = PrecisionParams(
    bg_n_points=200, ncdm_bg_n_points=100, bg_tol=1e-8,
    th_n_points=5000, th_z_max=5e3,
    pt_l_max_g=17, pt_l_max_pol_g=17, pt_l_max_ur=17,
    pt_k_max_cl=0.3,
    pt_ode_rtol=1e-3, pt_ode_atol=1e-6,
    ode_max_steps=262144,
)


class TestCompute:
    """Tests ``clax.compute()`` smoke behavior."""

    def test_compute_returns_result(self):
        """``compute()`` returns the expected result object; expects bg and th fields."""
        result = clax.compute(CosmoParams(), PREC_FAST)
        assert hasattr(result, 'bg')
        assert hasattr(result, 'th')
        assert float(result.bg.H0) > 0

    def test_compute_returns_finite_scalars(self):
        """``compute()`` returns finite scalar outputs; expects finite H0 and z_star."""
        result = clax.compute(CosmoParams(), PREC_FAST)
        assert jax.numpy.isfinite(result.bg.H0), "H0: found non-finite value; expected finite output"
        assert jax.numpy.isfinite(result.th.z_star), "z_star: found non-finite value; expected finite output"


class TestComputePk:
    """Tests ``clax.compute_pk()`` smoke behavior."""

    def test_pk_positive(self):
        """``compute_pk()`` returns a positive value; expects ``P(k=0.05) > 0``."""
        pk = clax.compute_pk(CosmoParams(), PREC_FAST, k=0.05)
        assert float(pk) > 0, f"P(k=0.05) = {float(pk)}"
