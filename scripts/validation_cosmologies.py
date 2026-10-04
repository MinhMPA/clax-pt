"""Canonical cosmology grid for the clax-pt vs CLASS-PT validation campaign.

Single source of truth for the 15 campaign cases (spec §4.1), redshifts,
the fixed AP fiducial, bias sets, the `--fast` subset and the reference
layout (spec §4.8).  Pure Python + NumPy so it imports in BOTH envs:
`clax` (JAX) and `classpt` (CLASS-PT oracle; no JAX, no clax).
`clax_params()` therefore imports clax lazily.

Cosmology dicts use clax `CosmoParams` names; `classpt_params_from()`
translates them to CLASS-PT `.ini` keys (spec §5.3).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_ROOT = REPO_ROOT / "reference_data" / "classpt"

# clax CosmoParams defaults (clax/params.py:46-75). ln10A_s = ln(2.1e-9 * 1e10).
FIDUCIAL = dict(
    h=0.6736, omega_b=0.02237, omega_cdm=0.1200, ln10A_s=3.0445224377,
    n_s=0.9649, tau_reio=0.0544,
    N_ncdm=1, m_ncdm=0.06, N_ur=2.0328, T_ncdm_over_T_cmb=0.71611,
    w0=-1.0, wa=0.0,
)
Y_HE_CLAX = 0.2454006   # clax/params.py:82; passed to CLASS-PT as YHe to remove the BBN seam

# name -> overrides on FIDUCIAL (spec §4.1). Values are locked to
# tests/conftest.py COSMOLOGY_GRID_LCDM / _NULCDM and
# scripts/generate_multipoint_reference.py.
CASES = {
    "lcdm_fiducial": {},
    "h_high": {"h": 0.6736 * 1.10},
    "omega_b_high": {"omega_b": 0.02237 * 1.20},
    "omega_cdm_low": {"omega_cdm": 0.1200 * 0.80},
    "ns_high": {"n_s": 0.9649 * 1.05},
    "massive_nu_006": {"m_ncdm": 0.06},          # == lcdm_fiducial (ALIASES)
    "massive_nu_015": {"m_ncdm": 0.15},
    "massive_nu_030": {"m_ncdm": 0.30},
    "massive_nu_015_h_high": {"m_ncdm": 0.15, "h": 0.6736 * 1.10},
    "massive_nu_015_omega_cdm_low": {"m_ncdm": 0.15, "omega_cdm": 0.1200 * 0.80},
    "w0wa_m09_p01": {"w0": -0.9, "wa": 0.1},
    "w0wa_m11_m01": {"w0": -1.1, "wa": -0.1},
    "w0wa_m10_p03": {"w0": -1.0, "wa": 0.3},
    "w0wa_m10_m03": {"w0": -1.0, "wa": -0.3},
    "w0wa_m07_m10": {"w0": -0.7, "wa": -1.0},    # w crosses -1 at a=0.7
}
FAMILIES = {
    "lcdm": ("lcdm_fiducial", "h_high", "omega_b_high", "omega_cdm_low", "ns_high"),
    "nulcdm": ("massive_nu_006", "massive_nu_015", "massive_nu_030",
               "massive_nu_015_h_high", "massive_nu_015_omega_cdm_low"),
    "w0wacdm": ("w0wa_m09_p01", "w0wa_m11_m01", "w0wa_m10_p03", "w0wa_m10_m03", "w0wa_m07_m10"),
}
ALIASES = {"massive_nu_006": "lcdm_fiducial"}

Z_LIST = (0.0, 0.38, 0.8)
OMFID = 0.31

# classy accessor names (classy.pyx:4795-4915). `cs` is the real-space matter
# counterterm; clax calls the same coefficient `cs0` in pk_mm_real.
BIAS = dict(b1=2.0, b2=0.0, bG2=0.0, bGamma3=0.0, cs0=0.0, cs2=0.0, cs4=0.0,
            cs=0.0, Pshot=0.0, b4=500.0)
BIAS_NONZERO = dict(b1=2.0, b2=-1.0, bG2=0.1, bGamma3=-0.1, cs0=5.0, cs2=15.0,
                    cs4=-5.0, cs=1.0, Pshot=5.0e3, b4=100.0)

FAST_CASES = ("lcdm_fiducial", "massive_nu_015", "w0wa_m07_m10")   # spec §4.6
FAST_Z = 0.38

# Exact input dict of the legacy reference_data/classpt_z0.38_fullrange.npz
# (scripts/generate_classpt_reference.py before Task A3): no N_ncdm, no N_ur,
# no YHe, no cb key -> CLASS-PT defaults N_ur=3.044, BBN YHe, cb=TRUE (which
# with N_ncdm=0 reads an unassigned delta_cb index; see Part 0 findings).
LEGACY_CLASSPT_FIDUCIAL = {
    "h": 0.6736, "omega_b": 0.02237, "omega_cdm": 0.1200,
    "A_s": 2.0989e-9, "n_s": 0.9649, "tau_reio": 0.0544,
}


def cosmo_params(name: str) -> dict:
    """clax-named parameter dict for a case (FIDUCIAL updated by its overrides)."""
    return {**FIDUCIAL, **CASES[name]}


def canonical_case(name: str) -> str:
    return ALIASES.get(name, name)


def distinct_cases() -> list[str]:
    """The 14 physically distinct cases, in CASES order (aliases dropped)."""
    return [c for c in CASES if c not in ALIASES]


def clax_params(name: str):
    """CosmoParams for a case (lazy clax import: absent in the classpt env)."""
    from clax import CosmoParams
    return CosmoParams(**cosmo_params(name))


def legacy_clax_params():
    """CosmoParams reproducing LEGACY_CLASSPT_FIDUCIAL on the clax side."""
    from clax import CosmoParams
    return CosmoParams(
        h=0.6736, omega_b=0.02237, omega_cdm=0.1200,
        ln10A_s=math.log(2.0989e-9 * 1e10), n_s=0.9649, tau_reio=0.0544,
        N_ncdm=0, N_ur=3.044,
    )


def ept_kgrid_numpy(kmin_h: float = 5e-5, kmax_h: float = 100.0, nmax: int = 256) -> np.ndarray:
    """Bit-identical twin of clax.ept.ept_kgrid (ept.py:1878-1889)."""
    return np.exp(np.linspace(np.log(kmin_h), np.log(kmax_h), nmax))


def classpt_params_from(cosmo: dict, *, z_list, ap: bool = True, omfid: float = OMFID,
                        cb: bool = True, yhe=Y_HE_CLAX, use_ppf=None, pt: bool = True) -> dict:
    """CLASS-PT input dict from a clax-named (or legacy CLASS-named) cosmology dict.

    Spec §5.3. `cosmo` may carry `A_s` directly (legacy dict) or `ln10A_s`.
    Strings are used for the CLASS-PT flags because its parser compares text
    (input.c:3952-3957 accepts N/No/NO for cb). `Omfid` is always written
    (CLASS-PT ignores it when AP=No; keeping it makes the noap files comparable).
    """
    n_ncdm = int(cosmo.get("N_ncdm", 0))
    if cb and n_ncdm == 0:
        raise ValueError("cb=True with N_ncdm=0 reads an unassigned delta_cb index in "
                         "CLASS-PT (perturbations.c:1273-1279); pass cb=False")
    prm = {"h": cosmo["h"], "omega_b": cosmo["omega_b"], "omega_cdm": cosmo["omega_cdm"],
           "n_s": cosmo["n_s"], "tau_reio": cosmo["tau_reio"]}
    prm["A_s"] = cosmo["A_s"] if "A_s" in cosmo else 1e-10 * math.exp(cosmo["ln10A_s"])
    if n_ncdm > 0:
        prm.update({"N_ncdm": n_ncdm, "m_ncdm": cosmo["m_ncdm"], "N_ur": cosmo["N_ur"],
                    "T_ncdm": cosmo["T_ncdm_over_T_cmb"]})
    elif "N_ur" in cosmo:
        prm["N_ur"] = cosmo["N_ur"]
    w0, wa = cosmo.get("w0", -1.0), cosmo.get("wa", 0.0)
    if (w0, wa) != (-1.0, 0.0):
        prm.update({"w0_fld": w0, "wa_fld": wa, "Omega_Lambda": 0.0})
    if use_ppf is not None:
        prm["use_ppf"] = "yes" if use_ppf else "no"
    if yhe is not None:
        prm["YHe"] = yhe
    if pt:
        prm.update({"output": "mPk", "non linear": "PT", "IR resummation": "Yes",
                    "Bias tracers": "Yes", "RSD": "Yes",
                    "AP": "Yes" if ap else "No", "Omfid": f"{omfid:g}",
                    "cb": "Yes" if cb else "No", "P_k_max_h/Mpc": 100.0})
    prm["z_pk"] = ",".join(f"{z:g}" for z in z_list)
    return prm


def classpt_params(name: str, **kw) -> dict:
    return classpt_params_from(cosmo_params(name), **kw)


def reference_path(case: str, z: float, *, ap: bool = True, omfid: float = OMFID,
                   cb: bool = True, bias: str = "fiducial", tag: str = "") -> Path:
    """reference_data/classpt/<canonical case>/z{z:.3f}_{ap_omfid{X}|noap}_{cb|m}[_biasnz][_tag].npz"""
    assert bias in ("fiducial", "nonzero"), bias
    stem = f"z{z:.3f}_" + (f"ap_omfid{omfid:g}" if ap else "noap") + ("_cb" if cb else "_m")
    if bias == "nonzero":
        stem += "_biasnz"
    if tag:
        stem += f"_{tag}"
    return REFERENCE_ROOT / canonical_case(case) / f"{stem}.npz"
