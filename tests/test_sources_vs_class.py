"""CMB source functions against CLASS at fixed k (regression for the RSA trigger).

Contract:
- clax's temperature source S_T0 and quadrupole source g*Pi agree with CLASS 3.3.4
  `get_sources()` to 0.5% rms of the peak amplitude over the recombination window
  at k = 0.15, 0.212, 0.30 /Mpc, i.e. the modes that build the C_l at l = 1700-4000.

Why this exists (docs/highl_residual_2026-10-12.md, branch exp/highl-investigation):
the l <= 1000 C_l tests were blind to a radiation-streaming switch that fired at
tau ~ 296 Mpc (visibility at 69% of its peak) instead of CLASS's tau_free_streaming
~ 365 Mpc (2.4%), zeroing g*Pi for k > 0.15/Mpc in the middle of recombination
(10% rms / 62% max of peak at k = 0.15). With CLASS's criterion the agreement is
0.1-0.3% rms (helium-era damping-scale offset), so the bar is 0.5%.

Reference data: reference_data/<name>/sources_k{0.15,0.2121,0.3}.npz from
scripts/generate_class_sources_reference.py (CLASS 3.3.4, RECFAST, Y_He 0.2454006,
default precision, exact k via k_output_values; conventions t0 = source_T0,
t2 = g*P = source_T2 / 8). clax solves a 3-mode log grid whose nodes are exactly
those k (PrecisionParams(pt_k_min=0.15, pt_k_max_cl=0.3, pt_k_per_decade=10)), so
no interpolation in k enters; clax's sources are linearly interpolated in tau
(0.17 Mpc spacing near recombination) onto CLASS's time sampling.

Multi-cosmology: the massless-RECFAST fiducial (the cosmology of
cls_massless_recfast.npz) plus the LCDM grid of tests/conftest.py (default 0.06 eV
neutrino) where references exist. All tests solve perturbations and are slow.
"""
import dataclasses
import glob
import os

import numpy as np
import pytest

from clax import CosmoParams, PrecisionParams
from clax.background import background_solve
from clax.thermodynamics import thermodynamics_solve
from clax.perturbations import perturbations_solve

REFERENCE_DIR = os.path.join(os.path.dirname(__file__), "..", "reference_data")
RMS_BAR = 0.005          # of the CLASS peak amplitude over the window
WINDOW = (-55.0, 170.0)  # Mpc around CLASS's visibility peak tau_rec


def _prec(**overrides):
    # Only the three reference k; the preset's other settings; a 20000-point
    # thermodynamics grid (planck_cl's 100000 changes C_l by <= 0.05%).
    return dataclasses.replace(PrecisionParams.planck_cl(), pt_k_min=0.15, pt_k_max_cl=0.3,
                               pt_k_per_decade=10, th_n_points=20000, **overrides)


def _load_refs(name):
    files = sorted(glob.glob(os.path.join(REFERENCE_DIR, name, "sources_k*.npz")))
    return [dict(np.load(f)) for f in files]


def _compare(pt, refs):
    """rms |clax - CLASS| / max |CLASS| over the window, per k, for T0 and g*Pi."""
    k_grid = np.asarray(pt.k_grid)
    tau_x = np.asarray(pt.tau_grid)
    out = {}
    for ref in refs:
        k = float(ref["k"])
        ik = int(np.argmin(np.abs(k_grid - k)))
        assert abs(k_grid[ik] / k - 1) < 1e-6, f"k={k} is not a node of the clax grid {k_grid}"
        tau_c = np.asarray(ref["tau"], dtype=float)
        lo, hi = float(ref["tau_rec"]) + WINDOW[0], float(ref["tau_rec"]) + WINDOW[1]
        m = (tau_c >= lo) & (tau_c <= hi)
        for label, clax_src, class_src in (("T0", np.asarray(pt.source_T0[ik]), np.asarray(ref["t0"], dtype=float)),
                                           ("gPi", np.asarray(pt.source_T2[ik]), 8.0 * np.asarray(ref["t2"], dtype=float))):
            sx = np.interp(tau_c[m], tau_x, clax_src)
            peak = np.max(np.abs(class_src[m]))
            out[(k, label)] = float(np.sqrt(np.mean((sx - class_src[m]) ** 2)) / peak)
    return out


def _run_and_check(name, params, prec):
    refs = _load_refs(name)
    if not refs:
        pytest.skip(f"no CLASS source reference for {name}")
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)
    pt = perturbations_solve(params, prec, bg, th)
    got = _compare(pt, refs)
    print(f"sources vs CLASS ({name}): " + ", ".join(f"k={k:.3f} {lab} {100 * v:.3f}%" for (k, lab), v in got.items()))
    bad = {kl: v for kl, v in got.items() if not v <= RMS_BAR}
    assert not bad, "rms/peak above %.1f%%: " % (100 * RMS_BAR) + ", ".join(
        f"k={k:.3f} {lab} {100 * v:.2f}%" for (k, lab), v in bad.items())


@pytest.mark.slow
class TestSourcesVsClass:
    def test_massless_recfast_fiducial(self):
        """The cosmology of cls_massless_recfast.npz: CosmoParams(m_ncdm=0.0), no ncdm block."""
        _run_and_check("massless_recfast", CosmoParams(m_ncdm=0.0), _prec(ncdm_q_size=0))

    def test_lcdm_grid(self, lcdm_cosmology):
        """LCDM grid points with the default 0.06 eV neutrino (fiducial only under --fast)."""
        name, params = lcdm_cosmology
        _run_and_check(name, params, _prec())
