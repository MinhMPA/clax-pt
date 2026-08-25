"""Tests scalar C_l forward accuracy at a heavier massive-neutrino point.

Contract:
- Scalar C_l (TT, EE, TE) at ``m_ncdm = 0.15`` eV are finite, sign-correct,
  and approximately consistent with CLASS -- mirroring ``test_harmonic.py``'s
  fiducial (``m_ncdm = 0.06``) contract, extended to the heavier point.

Scope:
- Covers low/mid-l TT/EE/TE forward checks at m_ncdm=0.15, the coverage gap
  this module closes (every other C_l test in the suite uses the default
  m_ncdm=0.06).
- This IS an oracle comparison: ``reference_data/massive_nu_015/cls.npz`` and
  ``cls_lensed.npz`` are real ``classy``-generated CLASS references (see
  ``scripts/generate_selected_pk_references.py`` and their use for background
  and P(k) regressions in ``tests/test_multipoint.py::TestMassiveNu``), so
  this is NOT a self-consistency-only fallback.

Notes:
- Uses the ``fast_cl`` preset (like ``test_harmonic.py``) with
  ``ncdm_fluid_approximation="none"``, matching the documented, already-
  validated choice in ``tests/test_multipoint.py`` for massive-neutrino
  robustness (the fluid-approximation switch has convergence issues for
  massive-neutrino cosmologies with the current solver at the k-scales
  ``fast_cl`` probes). Tolerances are therefore approximate (fast_cl-quality,
  not science-grade) like ``test_harmonic.py``.
- Tolerances are set EQUAL to (not looser than) ``test_harmonic.py``'s
  m_ncdm=0.06 fiducial bounds (TT: 30%/50%/50% at l=100/50/10; EE: 60%/60%
  at l=100/200), because a GPU measurement (job 13126) showed every probed
  ratio already fits inside those bounds at m_ncdm=0.15 too:
  TT(l=100)=1.1859 (30% bound, 11.4pp margin), TT(l=50)=1.0219 (50% bound),
  TT(l=10)=0.8363 (50% bound), EE(l=100)=1.5480 (60% bound, 5.2pp margin),
  EE(l=200)=1.0446 (60% bound). No measurement forced a looser bound than
  the existing fiducial contract, so none is carried here.
"""

import os
from dataclasses import replace as _dc_replace

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from clax import CosmoParams, PrecisionParams
from clax.background import background_solve
from clax.thermodynamics import thermodynamics_solve
from clax.perturbations import perturbations_solve
from clax.harmonic import compute_cl_tt, compute_cl_ee, compute_cl_te

REFERENCE_DIR = os.path.join(os.path.dirname(__file__), '..', 'reference_data')

# fast_cl + ncdm_fluid_approximation="none": the documented massive-neutrino
# -robust choice from tests/test_multipoint.py (full Boltzmann ncdm hierarchy
# throughout, no late-time fluid closure switch).
_PREC = _dc_replace(PrecisionParams.fast_cl(), ncdm_fluid_approximation="none")


@pytest.fixture(scope="module")
def pipeline_m_ncdm_015(request):
    """Background + thermo + perturbations at m_ncdm=0.15, fast_cl-quality.

    Skips under --fast: this is a second full perturbation solve (on top of
    the shared m_ncdm=0.06 ``pipeline_fast_cl`` fixture other files already
    pay for), not a --fast-subsamplable sweep, so a `pytest tests/ --fast`
    dev-loop run must not pay for it. Uses ``request.config.getoption``
    directly (not the function-scoped ``fast_mode`` fixture) because a
    module-scoped fixture cannot depend on a function-scoped one.
    """
    if request.config.getoption("--fast", default=False):
        pytest.skip("m_ncdm=0.15 full perturbation solve -- full mode only")
    params = CosmoParams(m_ncdm=0.15)
    bg = background_solve(params, _PREC)
    th = thermodynamics_solve(params, _PREC, bg)
    pt = perturbations_solve(params, _PREC, bg, th)
    return params, bg, th, pt


@pytest.fixture(scope="module")
def massive_nu_cls_ref():
    """Load the CLASS reference C_l at m_ncdm=0.15 (real oracle, not synthetic)."""
    path = os.path.join(REFERENCE_DIR, 'massive_nu_015', 'cls.npz')
    return dict(np.load(path, allow_pickle=True))


class TestClMassiveNuTT:
    """Tests scalar TT-spectrum behavior at m_ncdm=0.15."""

    def test_cl_tt_positive(self, pipeline_m_ncdm_015):
        """``C_l^TT`` is positive on the probe grid; expects positive values."""
        params, bg, th, pt = pipeline_m_ncdm_015
        cl = compute_cl_tt(pt, params, bg, [10, 50, 100])
        for i, l in enumerate([10, 50, 100]):
            assert float(cl[i]) > 0, f"C_l^TT(l={l}) = {float(cl[i]):.4e} is not positive"

    def test_cl_tt_l100_accuracy(self, pipeline_m_ncdm_015, massive_nu_cls_ref):
        """``C_l^TT`` at ``l=100`` matches CLASS (m_ncdm=0.15); expects <30% relative error (matches test_harmonic.py fiducial; measured 18.6%)."""
        params, bg, th, pt = pipeline_m_ncdm_015
        cl = compute_cl_tt(pt, params, bg, [100])
        cl_us = float(cl[0])
        cl_class = float(massive_nu_cls_ref['tt'][100])
        ratio = cl_us / cl_class
        print(f"C_l^TT(l=100, m_ncdm=0.15): clax={cl_us:.4e}, CLASS={cl_class:.4e}, ratio={ratio:.4f}")
        assert abs(ratio - 1) < 0.30, f"C_l^TT(l=100): ratio={ratio:.4f}, expected within 30%"

    def test_cl_tt_l50_accuracy(self, pipeline_m_ncdm_015, massive_nu_cls_ref):
        """``C_l^TT`` at ``l=50`` matches CLASS (m_ncdm=0.15); expects <50% relative error (matches test_harmonic.py fiducial; measured 2.2%)."""
        params, bg, th, pt = pipeline_m_ncdm_015
        cl = compute_cl_tt(pt, params, bg, [50])
        cl_us = float(cl[0])
        cl_class = float(massive_nu_cls_ref['tt'][50])
        ratio = cl_us / cl_class
        print(f"C_l^TT(l=50, m_ncdm=0.15): clax={cl_us:.4e}, CLASS={cl_class:.4e}, ratio={ratio:.4f}")
        assert abs(ratio - 1) < 0.50, f"C_l^TT(l=50): ratio={ratio:.4f}, expected within 50%"

    def test_cl_tt_l10_accuracy(self, pipeline_m_ncdm_015, massive_nu_cls_ref):
        """``C_l^TT`` at ``l=10`` matches CLASS (m_ncdm=0.15); expects <50% relative error (matches test_harmonic.py fiducial; measured 16.4%)."""
        params, bg, th, pt = pipeline_m_ncdm_015
        cl = compute_cl_tt(pt, params, bg, [10])
        cl_us = float(cl[0])
        cl_class = float(massive_nu_cls_ref['tt'][10])
        ratio = cl_us / cl_class
        print(f"C_l^TT(l=10, m_ncdm=0.15): clax={cl_us:.4e}, CLASS={cl_class:.4e}, ratio={ratio:.4f}")
        assert abs(ratio - 1) < 0.50, f"C_l^TT(l=10): ratio={ratio:.4f}, expected within 50%"


class TestClMassiveNuEE:
    """Tests scalar EE-spectrum behavior at m_ncdm=0.15."""

    def test_cl_ee_positive_and_finite(self, pipeline_m_ncdm_015):
        """``C_l^EE`` is positive and finite on the probe grid."""
        params, bg, th, pt = pipeline_m_ncdm_015
        cl = compute_cl_ee(pt, params, bg, [100, 200])
        for i, l in enumerate([100, 200]):
            val = float(cl[i])
            assert np.isfinite(val), f"C_l^EE(l={l}) is not finite"
            assert val > 0, f"C_l^EE(l={l}) = {val:.4e} is not positive"

    def test_cl_ee_l100_accuracy(self, pipeline_m_ncdm_015, massive_nu_cls_ref):
        """``C_l^EE`` at ``l=100`` matches CLASS (m_ncdm=0.15); expects <60% relative error (matches test_harmonic.py fiducial; measured 54.8%)."""
        params, bg, th, pt = pipeline_m_ncdm_015
        cl = compute_cl_ee(pt, params, bg, [100])
        cl_us = float(cl[0])
        cl_class = float(massive_nu_cls_ref['ee'][100])
        ratio = cl_us / cl_class
        print(f"C_l^EE(l=100, m_ncdm=0.15): clax={cl_us:.4e}, CLASS={cl_class:.4e}, ratio={ratio:.4f}")
        assert abs(ratio - 1) < 0.60, f"C_l^EE(l=100): ratio={ratio:.4f}"

    def test_cl_ee_l200_accuracy(self, pipeline_m_ncdm_015, massive_nu_cls_ref):
        """``C_l^EE`` at ``l=200`` matches CLASS (m_ncdm=0.15); expects <60% relative error (matches test_harmonic.py fiducial; measured 4.5%)."""
        params, bg, th, pt = pipeline_m_ncdm_015
        cl = compute_cl_ee(pt, params, bg, [200])
        cl_us = float(cl[0])
        cl_class = float(massive_nu_cls_ref['ee'][200])
        ratio = cl_us / cl_class
        print(f"C_l^EE(l=200, m_ncdm=0.15): clax={cl_us:.4e}, CLASS={cl_class:.4e}, ratio={ratio:.4f}")
        assert abs(ratio - 1) < 0.60, f"C_l^EE(l=200): ratio={ratio:.4f}"


class TestClMassiveNuTE:
    """Tests scalar TE-spectrum behavior at m_ncdm=0.15."""

    def test_cl_te_finite(self, pipeline_m_ncdm_015):
        """``C_l^TE`` is finite on the probe grid."""
        params, bg, th, pt = pipeline_m_ncdm_015
        cl = compute_cl_te(pt, params, bg, [100, 200])
        for i, l in enumerate([100, 200]):
            assert np.isfinite(float(cl[i])), f"C_l^TE(l={l}) is not finite"

    def test_cl_te_sign(self, pipeline_m_ncdm_015, massive_nu_cls_ref):
        """``C_l^TE`` sign matches CLASS (m_ncdm=0.15) on the probe grid."""
        params, bg, th, pt = pipeline_m_ncdm_015
        cl = compute_cl_te(pt, params, bg, [100, 200])
        for i, l in enumerate([100, 200]):
            cl_us = float(cl[i])
            cl_class = float(massive_nu_cls_ref['te'][l])
            sign_match = (cl_us * cl_class) > 0
            print(f"C_l^TE(l={l}, m_ncdm=0.15): clax={cl_us:.4e}, CLASS={cl_class:.4e}, sign_match={sign_match}")
            assert sign_match, f"C_l^TE(l={l}): sign mismatch"


class TestClMassiveNuVsFiducial:
    """Sanity: m_ncdm=0.15 C_l stays in a physically sensible ratio to the
    default m_ncdm=0.06 fiducial pipeline (already computed elsewhere in the
    suite via the shared ``pipeline_fast_cl`` fixture -- no extra heavy solve).
    """

    def test_cl_tt_l100_ratio_to_fiducial_is_order_unity(self, pipeline_m_ncdm_015, pipeline_fast_cl):
        """TT(l=100) at m_ncdm=0.15 stays within a factor of ~2 of the m_ncdm=0.06 fiducial."""
        params15, bg15, th15, pt15 = pipeline_m_ncdm_015
        params06, _, bg06, _, pt06 = pipeline_fast_cl
        cl15 = float(compute_cl_tt(pt15, params15, bg15, [100])[0])
        cl06 = float(compute_cl_tt(pt06, params06, bg06, [100])[0])
        ratio = cl15 / cl06
        print(f"C_l^TT(l=100): m_ncdm=0.15/{{0.06}} ratio={ratio:.4f}")
        assert 0.5 < ratio < 2.0, (
            f"C_l^TT(l=100) m_ncdm=0.15 vs 0.06 ratio={ratio:.4f}, expected order-unity (0.5-2.0)"
        )
