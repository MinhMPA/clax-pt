"""Preset accuracy against CLASS (precision sweep, Oct 10-12, 2026).

The Oct 2026 sweep (docs/precision_sweep_2026-10-10.md) retuned planck_cl from
Kvaerno5 at l_max 50, rtol 1e-6, atol 1e-11 (4606 s per perturbation solve on
a V100) to l_max 17, rtol 1e-4, atol 1e-6 (100 s) with spectra within 0.1% of
the old preset, and fixed the fast C_l path (Bessel table 30 points per unit x,
n_k_fine 20000). Measured against the CLASS massless-RECFAST reference (sweep config P1,
Oct 12, 2026) the new preset reads TT within 0.09% (l < 30), 0.22% (l <= 500),
0.25% (l <= 1000); EE within 0.60% (l = 2-4), 0.48% (l = 5-500), 0.98% at a
single l near 526 (rms 0.28% over 500-1000); TE/sqrt(TT EE) within 0.37%.
With the default 0.06 eV neutrino against the CLASS default (HyRec) fiducial
reference: TT 0.20 / 0.23 / 0.26%, EE 0.52 / 0.49 / 0.75%, TE 0.36%. These tests pin those numbers with margin and
guard the two failure classes the sweep found: a biased Bessel table (-1% at
l < 50, +20% in EE at l = 2) and a solver whose dense output corrupts the late
ISW (rodas5: -4% to -46% at l = 2).

Multi-cosmology rule: the fiducial massless-RECFAST comparison is the tight
gate; the LCDM grid (massive 0.06 eV neutrino, CLASS default HyRec
recombination) runs with looser bars that include the RECFAST-vs-HyRec and
ncdm-fluid systematics. All tests are GPU-scale and marked slow.
"""
import dataclasses
import os

import numpy as np
import pytest

from clax import CosmoParams, PrecisionParams
from clax.background import background_solve
from clax.thermodynamics import thermodynamics_solve
from clax.perturbations import perturbations_solve
from clax.harmonic import compute_cls_all_fast

REFERENCE_DIR = os.path.join(os.path.dirname(__file__), "..", "reference_data")


def _cls(params, prec, l_max=None):
    """C_l from the preset. Always computed to the preset's hr_l_max: the
    sparse-l cubic spline's end condition contaminates the last ~50 l below
    whatever l_max is passed (2% at l~1000 when l_max=1000), so the bands
    are judged well inside the computed range."""
    l_max = prec.hr_l_max if l_max is None else l_max
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)
    pt = perturbations_solve(params, prec, bg, th)
    cls = compute_cls_all_fast(pt, params, bg, l_max=l_max, n_k_fine=prec.hr_n_k_fine)
    return {k: np.asarray(cls[k]) for k in ("tt", "ee", "te")}


def _band_max(num, ref, lo, hi):
    """max |num/ref - 1| over lo <= l <= hi; ref may extend beyond len(num)."""
    n = len(num)
    ref = np.asarray(ref)[:n]
    ell = np.arange(n)
    m = (ell >= lo) & (ell <= hi)
    return float(np.max(np.abs(num[m] / ref[m] - 1.0)))


def _te_norm_max(cls, ref, lo, hi):
    """max |(TE - TE_ref)| / sqrt(TT_ref EE_ref) over lo <= l <= hi."""
    n = len(cls["te"])
    ell = np.arange(n)
    m = (ell >= lo) & (ell <= hi)
    tt, ee, te = (np.asarray(ref[k])[:n] for k in ("tt", "ee", "te"))
    d = (cls["te"][m] - te[m]) / np.sqrt(tt[m] * ee[m])
    return float(np.max(np.abs(d)))


def _report(label, items):
    # concise: one line per spectrum/band
    print(label + ": " + ", ".join(f"{k} {100 * v:.3f}%" for k, v in items.items()))


@pytest.mark.slow
class TestPlanckClPreset:
    """planck_cl must hold the sweep's measured accuracy with margin."""

    def test_fiducial_massless_recfast(self):
        ref = dict(np.load(os.path.join(REFERENCE_DIR, "cls_massless_recfast.npz")))
        params = CosmoParams(m_ncdm=0.0)
        prec = dataclasses.replace(PrecisionParams.planck_cl(), ncdm_q_size=0)
        cls = _cls(params, prec)
        got = {
            "TT 2-30": _band_max(cls["tt"], ref["tt"], 2, 30),
            "TT 2-500": _band_max(cls["tt"], ref["tt"], 2, 500),
            "TT 2-1000": _band_max(cls["tt"], ref["tt"], 2, 1000),
            "EE 2-4": _band_max(cls["ee"], ref["ee"], 2, 4),
            "EE 5-500": _band_max(cls["ee"], ref["ee"], 5, 500),
            "EE 5-1000": _band_max(cls["ee"], ref["ee"], 5, 1000),
            "TEn 2-1000": _te_norm_max(cls, ref, 2, 1000),
        }
        _report("planck_cl vs CLASS massless RECFAST", got)
        # measured 0.091 / 0.220 / 0.254 / 0.60 / 0.48 / 0.98 / 0.37 percent (P1); bars ~1.5x
        assert got["TT 2-30"] < 0.0015
        assert got["TT 2-500"] < 0.0035
        assert got["TT 2-1000"] < 0.004
        assert got["EE 2-4"] < 0.010
        assert got["EE 5-500"] < 0.0075
        assert got["EE 5-1000"] < 0.015
        assert got["TEn 2-1000"] < 0.006

    def test_lcdm_grid_massive(self, lcdm_cosmology):
        """Grid points use the default massive neutrino and CLASS's default
        (HyRec) reference, so the bars include the recombination and ncdm
        systematics (measured ~0.3% at l <= 1000 at fiducial)."""
        name, params = lcdm_cosmology
        ref_dir = os.path.join(REFERENCE_DIR, name)
        if not os.path.isdir(ref_dir):
            pytest.skip(f"no CLASS reference for {name}")
        ref = dict(np.load(os.path.join(ref_dir, "cls.npz")))
        cls = _cls(params, PrecisionParams.planck_cl())
        got = {
            "TT 2-1000": _band_max(cls["tt"], ref["tt"], 2, 1000),
            "EE 5-1000": _band_max(cls["ee"], ref["ee"], 5, 1000),
            "TEn 2-1000": _te_norm_max(cls, ref, 2, 1000),
        }
        _report(f"planck_cl vs CLASS ({name}, massive, HyRec)", got)
        # fiducial measured 0.26 / 0.75 / 0.36 percent (P1m); other grid points unmeasured
        assert got["TT 2-1000"] < 0.008
        assert got["EE 5-1000"] < 0.015
        assert got["TEn 2-1000"] < 0.008


@pytest.mark.slow
class TestFitClPreset:
    """fit_cl is the HMC preset: coarse, but its low-l TT must not be corrupted
    by the solver's dense output (rodas5 gave -46% at l = 2 at these tolerances)."""

    def test_low_l_within_fit_budget(self):
        ref = dict(np.load(os.path.join(REFERENCE_DIR, "cls_massless_recfast.npz")))
        params = CosmoParams(m_ncdm=0.0)
        prec = PrecisionParams.fit_cl()
        cls = _cls(params, prec)
        got = {
            "TT 2-30": _band_max(cls["tt"], ref["tt"], 2, 30),
            "TT 2-500": _band_max(cls["tt"], ref["tt"], 2, 500),
            "EE 5-500": _band_max(cls["ee"], ref["ee"], 5, 500),
        }
        _report("fit_cl vs CLASS massless RECFAST", got)
        assert got["TT 2-30"] < 0.015
        assert got["TT 2-500"] < 0.02
        assert got["EE 5-500"] < 0.02
