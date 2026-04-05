"""Tests linear-matter power-spectrum forward accuracy.

Contract:
- Forward ``P(k)`` predictions match the CLASS reference within the documented tolerance.

Scope:
- Covers fiducial ``P(k)`` value accuracy and one multi-redshift growth-rate check.
- Excludes ``P(k)`` gradient contracts owned by ``test_pk_gradients.py``.

Notes:
- These tests use one shared heavy perturbation solve per module.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from clax.background import background_solve
from clax.params import CosmoParams, PrecisionParams
from clax.perturbations import perturbations_solve
from clax.thermodynamics import thermodynamics_solve
from clax.transfer import compute_pk_from_perturbations


PREC = PrecisionParams.medium_cl()

K_FULL = np.array([0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3])
K_FAST = np.array([0.001, 0.01, 0.05, 0.1, 0.3])


def _interp_loglog(k, k_arr, pk_arr):
    """Interpolate ``P(k)`` in log-log space; expects positive inputs."""
    return np.exp(np.interp(np.log(k), np.log(k_arr), np.log(pk_arr)))


@pytest.fixture(scope="module")
def pipeline():
    """Run the fiducial forward pipeline once for this module."""
    params = CosmoParams()
    bg = background_solve(params, PREC)
    th = thermodynamics_solve(params, PREC, bg)
    pt = perturbations_solve(params, PREC, bg, th)
    return params, bg, pt


class TestPkAccuracy:
    """Tests fiducial ``P(k)`` value accuracy."""

    @pytest.mark.slow
    def test_pk_matches_class(self, pipeline, lcdm_pk_ref, fast_mode):
        """Fiducial ``P(k)`` matches CLASS; expects <1.5% max relative error on the probe grid."""
        params, bg, pt = pipeline
        k_probe = K_FAST if fast_mode else K_FULL

        k_ref = lcdm_pk_ref["k"]
        pk_ref = lcdm_pk_ref["pk_lin_z0"]
        k_grid = np.array(pt.k_grid)
        delta_m_z0 = np.array(pt.delta_m[:, -1])

        A_s = float(jnp.exp(params.ln10A_s) / 1e10)
        pk_clax = 2 * np.pi**2 / k_grid**3 * A_s * (k_grid / params.k_pivot) ** (params.n_s - 1) * delta_m_z0**2

        rel_errs = []
        valid_ks = []
        for k in k_probe:
            if not (k_grid[0] <= k <= k_grid[-1]):
                continue
            pk_us = np.exp(np.interp(np.log(k), np.log(k_grid), np.log(np.abs(pk_clax) + 1e-40)))
            pk_class = _interp_loglog(k, k_ref, pk_ref)
            rel_errs.append(abs(pk_us / pk_class - 1.0))
            valid_ks.append(k)

        rel_errs = np.array(rel_errs)
        valid_ks = np.array(valid_ks)
        max_err = float(np.max(rel_errs))
        mean_err = float(np.mean(rel_errs))
        worst_k = float(valid_ks[np.argmax(rel_errs)])

        assert max_err < 0.015, (
            f"P(k): max relative error {max_err:.2%} at k={worst_k:.3f} Mpc^-1; "
            f"expected <1.5% (mean={mean_err:.2%}, n={len(valid_ks)})"
        )


class TestPkGrowthRate:
    """Tests multi-redshift ``P(k)`` behavior."""

    @pytest.mark.slow
    def test_growth_rate_relative(self, pipeline, lcdm_pk_ref):
        """``P(k,z=0.5)/P(k,z=0)`` matches CLASS; expects <1% max relative error."""
        params, bg, pt = pipeline

        k_ref = lcdm_pk_ref["k"]
        pk_z0 = lcdm_pk_ref["pk_lin_z0"]
        pk_z05 = lcdm_pk_ref["pk_z0.5"]

        k_test = jnp.array([0.05, 0.1, 0.2])
        A_s = float(jnp.exp(params.ln10A_s) / 1e10)

        dm_z0 = np.array(compute_pk_from_perturbations(pt, bg, k_test, z=0.0))
        dm_z05 = np.array(compute_pk_from_perturbations(pt, bg, k_test, z=0.5))

        pk_clax_z0 = 2 * np.pi**2 / np.array(k_test) ** 3 * A_s * (np.array(k_test) / params.k_pivot) ** (params.n_s - 1) * dm_z0**2
        pk_clax_z05 = 2 * np.pi**2 / np.array(k_test) ** 3 * A_s * (np.array(k_test) / params.k_pivot) ** (params.n_s - 1) * dm_z05**2

        rel_errs = []
        for i, k in enumerate(np.array(k_test)):
            ratio_us = pk_clax_z05[i] / pk_clax_z0[i]
            ratio_class = _interp_loglog(k, k_ref, pk_z05) / _interp_loglog(k, k_ref, pk_z0)
            rel_errs.append(abs(ratio_us / ratio_class - 1.0))

        rel_errs = np.array(rel_errs)
        max_err = float(np.max(rel_errs))
        worst_k = float(np.array(k_test)[np.argmax(rel_errs)])

        assert max_err < 0.01, (
            f"P(z=0.5)/P(z=0): max relative error {max_err:.2%} at k={worst_k:.3f}; expected <1%"
        )
