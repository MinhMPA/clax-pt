"""NumPy twin of the CLASS-PT classy accessors (classy.pyx:4805-4932, ref §11).

Consumers: the reference generator asserts classy == twin on every file it
writes, and clax-side tests assemble spectra from stored `pk_mult` with any
bias set without classy.  `pm` is the (96, Nk) array from `get_pk_mult`
(CLASS units, row transforms in ref §10); `h` the Hubble parameter; `fz` the
growth rate; `kh` the k grid in h/Mpc (the patched classy convention);
`Pd2d2_0` the k->0 limit of the b2^2 term (classy.pyx:4813).

Line citations are the *live*, post-patch numbers of
/home/n2minh/CLASS-PT/python/classy.pyx at commit 09d5531a; the plan's
pre-patch citations (4783-4915) are stale by ~20 lines.

Bias keys are classy's: b1 b2 bG2 bGamma3 cs0 cs2 cs4 cs Pshot b4.
classy pk_mm_real(cs) corresponds to clax pk_mm_real(cs0=cs).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import simpson

PM_ROWS_VALID = slice(0, 48)        # rows 48-71 are fNL garbage when PNG is off
# Measured by tests/test_classpt_assembly.py::test_twin_decides_legacy_kh_convention on
# reference_data/classpt_z0.38_fullrange.npz: max rel residual over pk_gg_l0/l2/l4 is
# 2.334e-15 for kh in h/Mpc and 2.201e+01 for kh in 1/Mpc -> the legacy file already used
# the h/Mpc convention that classy_kh_units.patch now makes the default (ref §12).
LEGACY_KH_CONVENTION = "h/Mpc"


def pd2d2_0(pk_lin_h, kh) -> float:
    """classy.pyx:4813: simpson(Plin_hMpc3**2 * kh**3, x=log(kh)) / pi**2."""
    pk_lin_h = np.asarray(pk_lin_h, dtype=float)
    kh = np.asarray(kh, dtype=float)
    return float(simpson(pk_lin_h**2 * kh**3, x=np.log(kh)) / np.pi**2)


def assemble_from_pm(pm, h, fz, kh, bias, Pd2d2_0) -> dict:
    """Every classy accessor, transcribed line-for-line (a0_nbar = a2_nbar = 0)."""
    pm = np.asarray(pm, dtype=float)
    kh = np.asarray(kh, dtype=float)
    b1, b2, bG2, bG3 = bias["b1"], bias["b2"], bias["bG2"], bias["bGamma3"]
    cs0, cs2, cs4, cs = bias["cs0"], bias["cs2"], bias["cs4"], bias["cs"]
    Pshot, b4 = bias["Pshot"], bias["b4"]
    h2, h3 = h**2, h**3
    b4k = fz**2 * b4 * kh**2 * (35.0 / 8.0) * pm[13] * h     # shared b4 k^2 mu^4 factor
    out = {}
    # classy.pyx:4816-4820  pk_mm_real(cs)
    out["pk_mm_real"] = (pm[0] + pm[14] + 2.0 * cs * pm[10] / h2) * h3
    # classy.pyx:4822-4827  pk_gg_real(b1,b2,bG2,bGamma3,cs,cs0,Pshot)
    out["pk_gg_real"] = (b1**2 * pm[14] + b1**2 * pm[0]
                         + 2.0 * (cs * b1**2 + cs0 * b1) * pm[10] / h2
                         + b1 * b2 * pm[2] + 0.25 * b2**2 * pm[1]
                         + 2.0 * b1 * bG2 * pm[3] + b1 * (2.0 * bG2 + 0.8 * bG3) * pm[6]
                         + bG2**2 * pm[5] + b2 * bG2 * pm[4]) * h3 + Pshot
    # classy.pyx:4829-4834  pk_gm_real(b1,b2,bG2,bGamma3,cs,cs0)
    out["pk_gm_real"] = (b1 * pm[14] + b1 * pm[0] + (2.0 * cs * b1 + cs0) * pm[10] / h2
                         + 0.5 * b2 * pm[2] + bG2 * pm[3]
                         + (bG2 + 0.4 * bG3) * pm[6]) * h3
    # classy.pyx:4881-4897  pk_mm_l0/l2/l4(cs0/cs2/cs4)
    out["pk_mm_l0"] = (pm[15] + pm[21] + pm[16] + pm[22] + pm[17] + pm[23]
                       + 2.0 * cs0 * pm[11] / h2) * h3
    out["pk_mm_l2"] = (pm[18] + pm[24] + pm[19] + pm[25] + pm[26]
                       + 2.0 * cs2 * pm[12] / h2) * h3
    out["pk_mm_l4"] = (pm[20] + pm[27] + pm[28] + pm[29] + 2.0 * cs4 * pm[13] / h2) * h3
    # classy.pyx:4900-4912  pk_gg_l0(b1,b2,bG2,bGamma3,cs0,Pshot_nbar,a0_nbar,a2_nbar,b4)
    out["pk_gg_l0"] = ((pm[15] + pm[21] + b1 * pm[16] + b1 * pm[22]
                        + b1**2 * pm[17] + b1**2 * pm[23]
                        + 0.25 * b2**2 * pm[1] + b1 * b2 * pm[30] + b2 * pm[31]
                        + b1 * bG2 * pm[32] + bG2 * pm[33] + b2 * bG2 * pm[4] + bG2**2 * pm[5]
                        + 2.0 * cs0 * pm[11] / h2
                        + (2.0 * bG2 + 0.8 * bG3) * (b1 * pm[7] + pm[8])) * h3
                       + Pshot + 0.25 * b2**2 * Pd2d2_0
                       + b4k * (fz**2 / 9.0 + 2.0 * fz * b1 / 7.0 + b1**2 / 5.0))
    # classy.pyx:4914-4923  pk_gg_l2(b1,b2,bG2,bGamma3,cs2,a2_nbar,b4)
    out["pk_gg_l2"] = ((pm[18] + pm[24] + b1 * pm[19] + b1 * pm[25] + b1**2 * pm[26]
                        + b1 * b2 * pm[34] + b2 * pm[35] + b1 * bG2 * pm[36] + bG2 * pm[37]
                        + 2.0 * cs2 * pm[12] / h2 + (2.0 * bG2 + 0.8 * bG3) * pm[9]) * h3
                       + b4k * (70.0 * fz**2 + 165.0 * fz * b1 + 99.0 * b1**2) * 4.0 / 693.0)
    # classy.pyx:4925-4932  pk_gg_l4(b1,b2,bG2,bGamma3,cs4,b4)
    out["pk_gg_l4"] = ((pm[20] + pm[27] + b1 * pm[28] + b1**2 * pm[29]
                        + b2 * pm[38] + bG2 * pm[39] + 2.0 * cs4 * pm[13] / h2) * h3
                       + b4k * (210.0 * fz**2 + 390.0 * fz * b1 + 143.0 * b1**2) * 8.0 / 5005.0)
    return out
