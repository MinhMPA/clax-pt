# tests/pfs_survey.py
"""PFS survey inputs for the ps_1loop error-bar gate (spec §3, §5.6).

Takada et al. (2014, PASJ 66, R1; arXiv:1206.0737) Table 2, stored verbatim in
reference_data/pfs/takada2014_table2.json and converted here to (Mpc/h)^3 and (h/Mpc)^3.
Each slice is evaluated at its midpoint (the paper defines no effective redshift).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "reference_data" / "pfs" / "takada2014_table2.json"
_T = json.loads(TABLE.read_text())
AREA_DEG2: float = _T["area_deg2"]
FOV_DEG2: float = _T["fov_deg2"]

# Error-bar-gate bias point (spec §5.6).
C_REF = 2.0       # (Mpc/h)^2 -- PLACEHOLDER k^2 counterterm amplitude, c0 = c2 = c4; replace with
                  # the PFS forecast's prior centres. ~10% of the z=0.7 monopole at k = 0.2 h/Mpc.
CFOG_REF = 500.0  # (Mpc/h)^4 -- = CLASS-PT b4 of the 42-file reference set (bias_json)


@dataclass(frozen=True)
class PFSSlice:
    z_min: float
    z_max: float
    V: float              # comoving volume, (Mpc/h)^3
    nbar: float           # number density, (h/Mpc)^3
    b_g: float            # linear bias (Table 2 column)
    N_g_per_field: int    # [OII] galaxies per field

    @property
    def z_mid(self) -> float:
        return round(0.5 * (self.z_min + self.z_max), 3)


def load_slices() -> tuple[PFSSlice, ...]:
    return tuple(PFSSlice(z_min=r["z_min"], z_max=r["z_max"], V=r["V"] * 1e9,
                          nbar=r["nbar"] * 1e-4, b_g=r["b_g"], N_g_per_field=r["N_g_per_field"])
                 for r in _T["slices"])


def bias_point(z: float, *, c: float = C_REF) -> dict[str, float]:
    """ps_1loop-basis parameters at redshift z (spec §5.6): Takada b1, Lazeyras et al. (2016)
    b2(b1), coevolution bG2 and bGamma3, counterterms c, fingers-of-god CFOG_REF, Poisson
    P_shot = 1 (with ndens = nbar), a0 = a2 = 0."""
    b1 = 0.9 + 0.4 * z
    return {"b1": b1,
            "b2": 0.412 - 2.143 * b1 + 0.929 * b1**2 + 0.008 * b1**3,
            "bG2": -2.0 / 7.0 * (b1 - 1.0),
            "bGamma3": 23.0 / 42.0 * (b1 - 1.0),
            "c0": c, "c2": c, "c4": c, "cfog": CFOG_REF,
            "P_shot": 1.0, "a0": 0.0, "a2": 0.0}


# --------------------------------------------------------------------------- error-bar gate numerics (spec §5.6)
import numpy as np                                           # noqa: E402
from scipy.interpolate import CubicSpline as _CubicSpline   # noqa: E402

DK = 0.01                                                    # h/Mpc
K_EDGES = np.round(np.arange(0.01, 0.2 + 1e-9, DK), 10)      # 0.01 ... 0.20
K_CENTRES = 0.5 * (K_EDGES[1:] + K_EDGES[:-1])               # 0.015 ... 0.195 (19 bins)
_ELLS = np.array([0, 2, 4])
_GL_MU, _GL_W = np.polynomial.legendre.leggauss(40)
_L = np.stack([np.ones_like(_GL_MU), 0.5 * (3 * _GL_MU**2 - 1), (35 * _GL_MU**4 - 30 * _GL_MU**2 + 3) / 8])


def to_centres(P, kw):
    """(3, Nw) on the window -> (3, Nb) at K_CENTRES, cubic spline in ln k (same for both codes)."""
    return _CubicSpline(np.log(np.asarray(kw)), np.asarray(P), axis=1)(np.log(K_CENTRES))


def multipole_covariance(P, k, V, dk):
    """Gaussian covariance of (P0, P2, P4) per k-bin, (Nk, 3, 3):
    C_ll'(k) = (2l+1)(2l'+1)/N_k * int_{-1}^{1} L_l L_l' P(k,mu)^2 dmu, N_k = V k^2 dk/(2 pi^2).
    P(k, mu) = sum_l P_l L_l already contains the shot noise -- nothing is added (Review Focus 2)."""
    P = np.asarray(P, dtype=float)
    pkmu = P.T @ _L                                           # (Nk, 40)
    Nk = V * np.asarray(k) ** 2 * dk / (2 * np.pi**2)
    integ = np.einsum("m,am,bm,km->kab", _GL_W, _L, _L, pkmu**2)
    return np.outer(2 * _ELLS + 1, 2 * _ELLS + 1)[None] * integ / Nk[:, None, None]


def _cinv(C, v):
    return np.linalg.solve(C, np.asarray(v, dtype=float).T[..., None])[..., 0]    # (Nk, 3)


def delta_chi2(dP, C) -> float:
    return float(np.sum(np.asarray(dP).T * _cinv(C, dP)))


def counterterm_headroom(parts, c_ref) -> float:
    """Smallest c >= 0 with sum_slices (d0 + c D)^T C^-1 (d0 + c D) = 1, D = (d1 - d0)/c_ref. Exact:
    the counterterm residual is linear in c (c0 = c2 = c4 = c). parts: iterable of (d0, d1, C).
    0.0 if already >= 1 at c = 0; inf if the counterterm adds no residual."""
    a = b = e = 0.0
    for d0, d1, C in parts:
        D = (np.asarray(d1) - np.asarray(d0)) / c_ref
        a += float(np.sum(D.T * _cinv(C, D)))
        b += float(np.sum(D.T * _cinv(C, d0)))
        e += float(np.sum(np.asarray(d0).T * _cinv(C, d0)))
    if e >= 1.0:
        return 0.0
    if a <= 0.0:
        return float("inf")
    return float((-b + np.sqrt(b * b - a * (e - 1.0))) / a)
