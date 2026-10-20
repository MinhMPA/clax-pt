"""Rodas5 dense output at the C_l level: Rodas5 vs Kvaerno5 (GPU, slow).

The perturbation solve saves sources on a 5000-point conformal-time grid
(``SaveAt(ts=tau_grid)``), so between accepted solver steps the state is
reconstructed by the solver's local interpolation.  Rodas5's late-time
steps are hundreds of Mpc, and with ``LocalLinearInterpolation`` (its dense
output before Oct 12, 2026) the late-ISW source -- metric derivatives at the
output times -- was reconstructed piecewise-linearly: unlensed TT was off by
-46% at l=2 relative to Kvaerno5 at rtol 1e-4 / atol 1e-6, -4.1% at
1e-5 / 1e-10 (precision sweep of Oct 10-11, 2026, finding 5).  Kvaerno5
(diffrax; cubic Hermite dense output) is the reference: both solvers run at
identical tolerances on the same background and thermodynamics, so the
ratio isolates the solver, not the physics.

Settings: sweep config "I6"/"G6" -- planck_cl with ncdm_q_size=0 (massless
neutrinos, m_ncdm=0), CLASS-default hierarchy 12/10/17, rtol 1e-4 /
atol 1e-6, max_steps 32768, ~300 k-modes; C_l from compute_cls_all_fast at
l_max=500, n_k_fine=5000.

Multi-cosmology rule (CLAUDE.md "Test at many parameter points"): the test
is parametrized over ``COSMOLOGY_GRID_LCDM`` via ``lcdm_cosmology``.  It is
marked ``slow`` (about 5 min per cosmology on a V100; the fast suite skips
it), so for a single-point check run
``pytest tests/test_rodas5_dense_output.py -k lcdm_fiducial``.
"""

import dataclasses

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from clax import PrecisionParams
from clax.background import background_solve
from clax.thermodynamics import thermodynamics_solve
from clax.perturbations import perturbations_solve
from clax.harmonic import compute_cls_all_fast

L_MAX = 500
N_K_FINE = 5000
PREC_OVERRIDES = dict(
    ncdm_q_size=0, pt_l_max_g=12, pt_l_max_pol_g=10, pt_l_max_ur=17,
    pt_ode_rtol=1e-4, pt_ode_atol=1e-6, ode_max_steps=32768,
)
# Measured on the five LCDM grid points after the #53 recombination rewrite
# (job 26998, Oct 20, 2026): TT l=2 0.069-0.088% (the l=2 band maximum), TT
# 31-500 0.002-0.003% (at l=41-44), EE 0.030-0.033% (l=14-16). Bars: 2.3x,
# 7x and 9x those maxima; the linear dense output read 54%, 0.77% and 0.88%.
# (name, l_lo, l_hi, bar): |rodas5/kvaerno5 - 1| per band; TE is normalised
# by sqrt(TT*EE) of the reference (zero crossings make a plain ratio useless).
BARS = [
    ("tt", 2, 30, 0.002),
    ("tt", 31, L_MAX, 0.0002),
    ("ee", 2, L_MAX, 0.003),
    ("ten", 2, L_MAX, 0.003),
]


def _band_max(err, ell, lo, hi):
    m = (ell >= lo) & (ell <= hi)
    i = np.argmax(np.abs(err[m]))
    return float(np.abs(err[m])[i]), int(ell[m][i])


def _cls(params, prec, bg, th):
    pt = perturbations_solve(params, prec, bg, th)
    cls = compute_cls_all_fast(pt, params, bg, l_max=L_MAX, n_k_fine=N_K_FINE)
    out = {k: np.asarray(cls[k], dtype=float) for k in ("ell", "tt", "ee", "te")}
    jax.clear_caches()
    return out


@pytest.mark.slow
def test_rodas5_cls_match_kvaerno5(lcdm_cosmology):
    """Unlensed TT/EE/TE from Rodas5 vs Kvaerno5 at identical tolerances.

    Before the Hermite dense output this failed at l=2 with TT 46% low.
    """
    name, params = lcdm_cosmology
    params = dataclasses.replace(params, m_ncdm=0.0)
    prec_k5 = dataclasses.replace(PrecisionParams.planck_cl(),
                                  pt_ode_solver="kvaerno5", **PREC_OVERRIDES)
    prec_r5 = dataclasses.replace(prec_k5, pt_ode_solver="rodas5")

    # Background and thermodynamics do not depend on the perturbation solver:
    # computed once, shared by both solves.
    bg = background_solve(params, prec_k5)
    th = thermodynamics_solve(params, prec_k5, bg)
    ref = _cls(params, prec_k5, bg, th)
    got = _cls(params, prec_r5, bg, th)

    ell = ref["ell"]
    sel = ell >= 2
    ell = ell[sel]
    err = {
        "tt": got["tt"][sel] / ref["tt"][sel] - 1,
        "ee": got["ee"][sel] / ref["ee"][sel] - 1,
        "ten": (got["te"][sel] - ref["te"][sel])
               / np.sqrt(ref["tt"][sel] * ref["ee"][sel]),
    }
    assert all(np.isfinite(v).all() for v in err.values()), \
        f"{name}: non-finite C_l ratio"

    probe = [l for l in (2, 5, 10, 20, 30, 100, 500) if l <= L_MAX]
    print(f"{name}: rodas5/kvaerno5-1 TT at l={probe}: "
          + " ".join(f"{100 * err['tt'][l - 2]:+.3f}%" for l in probe))

    failures = []
    for spec, lo, hi, bar in BARS:
        m, l_at = _band_max(err[spec], ell, lo, hi)
        print(f"{name}: {spec.upper():>3} l={lo}-{hi}: max {100 * m:.3f}% at l={l_at} "
              f"(bar {100 * bar:.1f}%)")
        if m > bar:
            failures.append(f"{spec.upper()} l={lo}-{hi}: {100 * m:.3f}% at l={l_at} "
                            f"> {100 * bar:.1f}%")
    assert not failures, f"{name}: " + "; ".join(failures)
