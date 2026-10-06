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
