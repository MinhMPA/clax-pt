"""clax.ap — CLASS-PT Alcock–Paczynski ratios (nonlinear_pt.c:1245-1296).

Multi-cosmology rule: the consistency/gradient tests sweep `lcdm_cosmology`
(5 points; --fast prunes to fiducial). The oracle test runs on the single
AP-on reference that stores CLASS-PT's own ratios (Part 1a A3 legacy file);
Part 2's campaign files extend the oracle sweep to 15 cosmologies × 3 z.
"""
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from clax.ap import N_DFID, OMFID_DEFAULT, ap_ratios, ap_ratios_np
from clax.background import background_solve
from clax.params import CosmoParams

ROOT = Path(__file__).resolve().parents[1]
LEGACY_AP = ROOT / "reference_data/classpt_z0.38_ap_omfid0.31_legacy.npz"
C_KM_S = 299792.458          # km/s; H0 [1/Mpc] = 100 h / C_KM_S, cf. nonlinear_pt.c:1236 kmsMpc
OMEGA_G_H2 = 2.47282e-5      # photon density ω_γ for T_cmb = 2.7255 K (CLASS input.c)
Z_TEST = 0.38


def _omega_m(p: CosmoParams) -> float:
    """Total matter fraction today incl. the (non-relativistic) 0.06-eV neutrino."""
    return (p.omega_b + p.omega_cdm + p.m_ncdm / 93.14) / p.h**2


def _dfid_c_transcription(z, omfid, omega_g, Nz=N_DFID):
    """Literal loop of nonlinear_pt.c:1280-1284 (radiation term frozen at z)."""
    dz = z / (Nz - 1)
    rad = omega_g * (1 + z) ** 4
    E = lambda zz: np.sqrt(omfid * (1 + zz) ** 3 + (1 - omfid) + rad)
    return sum(dz * (1 / E(dz * j) + 1 / E(dz * (j - 1))) / 2.0 for j in range(1, Nz))


def test_constants_match_classpt():
    assert N_DFID == 2000          # nonlinear_pt.c:1233 `int Nz = 2000`
    assert OMFID_DEFAULT == 0.31   # CLASS-PT default `Omfid`, input.c:3879


def test_z0_is_identity():
    bg = background_solve(CosmoParams())
    h_r, d_r = ap_ratios(bg, 0.0)
    assert float(h_r) == 1.0 and float(d_r) == 1.0   # nonlinear_pt.c:1267-1269, exact


def test_twin_equals_c_transcription():
    z, omfid, omega_g = 0.8, 0.29, OMEGA_G_H2 / 0.7**2
    E_z, DM_H0 = 1.5, 0.6                      # arbitrary positive inputs
    hfid = np.sqrt(omfid * (1 + z) ** 3 + (1 - omfid) + omega_g * (1 + z) ** 4)
    h_r, d_r = ap_ratios_np(z, omfid, omega_g, E_z, DM_H0)
    assert abs(h_r - E_z / hfid) < 1e-14
    assert abs(d_r - DM_H0 / _dfid_c_transcription(z, omfid, omega_g)) < 1e-13


def test_jax_matches_twin(lcdm_cosmology):
    name, params = lcdm_cosmology
    bg = background_solve(params)
    loga = -np.log1p(Z_TEST)
    E_z = float(bg.H_of_loga.evaluate(loga) / bg.H0)
    DM_H0 = float((bg.conformal_age - bg.tau_of_loga.evaluate(loga)) * bg.H0)
    want = ap_ratios_np(Z_TEST, OMFID_DEFAULT, float(bg.Omega_g), E_z, DM_H0)
    got = ap_ratios(bg, Z_TEST)
    for label, g, w in zip(("hratio", "Dratio"), got, want):
        assert abs(float(g) - w) < 1e-10 * abs(w), f"{name} {label}: jax={float(g)!r} twin={w!r}"


def test_omfid_equal_to_omega_m_gives_unit_ratios(lcdm_cosmology):
    """With Omfid = Ω_m(cosmology) the fiducial IS the cosmology up to the
    ~1e-4 radiation/neutrino terms, so both ratios sit at 1 to < 2e-3. A
    wrong H0 factor, a Mpc/(Mpc/h) slip or a D_A vs D_M mix-up shows as ≫ 1e-2."""
    name, params = lcdm_cosmology
    bg = background_solve(params)
    h_r, d_r = ap_ratios(bg, Z_TEST, omfid=_omega_m(params))
    assert abs(float(h_r) - 1) < 2e-3, f"{name} hratio={float(h_r)!r}"
    assert abs(float(d_r) - 1) < 2e-3, f"{name} Dratio={float(d_r)!r}"


@pytest.mark.skipif(not LEGACY_AP.exists(), reason="Part 1a A3 legacy AP file absent")
def test_twin_reproduces_classpt_legacy_ratios():
    """Oracle: CLASS-PT's own hratio/Dratio (get_ap_ratios, Part 1a) from its own
    H(z), D_A(z). Only the twin is exercised — the legacy cosmology has no massive
    neutrino, which CosmoParams (N_ncdm static, default 1) does not express."""
    d = np.load(LEGACY_AP)
    z, h = float(d["z"]), float(d["h"])
    H0 = 100.0 * h / C_KM_S
    E_z = float(d["H_z"]) / H0
    DM_H0 = float(d["DA_z"]) * (1 + z) * H0
    omfid = float(d["omfid"])
    h_r, d_r = ap_ratios_np(z, omfid, OMEGA_G_H2 / h**2, E_z, DM_H0)
    assert abs(h_r - float(d["hratio"])) < 1e-6, f"hratio twin={h_r!r} classpt={float(d['hratio'])!r}"
    assert abs(d_r - float(d["Dratio"])) < 1e-6, f"Dratio twin={d_r!r} classpt={float(d['Dratio'])!r}"
    # Legacy fiducial (scripts/generate_classpt_reference.py:37-43) is Planck 2018
    # WITHOUT massive neutrinos: Ω_m = (0.02237+0.1200)/0.6736² = 0.31377 (not the
    # 0.31532 of the νΛCDM fiducial). Against Omfid = 0.31 at z = 0.38 the C
    # formula (1272/1276) gives hratio = 1.002053, i.e. ≈ +0.21 %.
    assert abs(h_r - 1.0020) < 5e-4


def _ratios(p):
    return jnp.stack(ap_ratios(background_solve(p), Z_TEST))


def _check_ad_vs_fd(name, params, fields):
    grads = jax.jacrev(_ratios)(params)
    for field, step in fields:
        x0 = getattr(params, field)
        up = _ratios(params.replace(**{field: x0 + step}))
        dn = _ratios(params.replace(**{field: x0 - step}))
        fd = np.asarray((up - dn) / (2 * step))
        ad = np.asarray(getattr(grads, field))
        rel = np.abs(ad - fd) / np.maximum(np.abs(fd), 1e-3)
        assert rel.max() < 1e-3, f"{name} d/d{field}: ad={ad} fd={fd} rel={rel.max():.2e}"


@pytest.mark.slow
def test_gradients_match_finite_differences(lcdm_cosmology):
    """d(hratio, Dratio)/d(h, omega_cdm) by reverse-mode AD vs central FD.

    w0/wa are NOT checked here: every COSMOLOGY_GRID_LCDM point sits exactly on
    clax's Lambda-vs-fluid branch seam, where their AD gradient is identically
    zero by construction (see test_w0_gradient_is_dead_on_the_lambda_branch).
    The AP w0/wa AD path is covered on the campaign's w0waCDM points instead,
    by test_w0wa_gradients_match_finite_differences.
    """
    name, params = lcdm_cosmology
    _check_ad_vs_fd(name, params, (("h", 1e-4), ("omega_cdm", 1e-4)))


W0WA_CASES = {                      # (w0, wa); the campaign's three w0waCDM points
    "w0wa_m09_p01": (-0.9, 0.1),
    "w0wa_m10_p03": (-1.0, 0.3),
    "w0wa_m07_m10": (-0.7, -1.0),
}


@pytest.mark.slow
@pytest.mark.parametrize("case", sorted(W0WA_CASES))
def test_w0wa_gradients_match_finite_differences(case):
    """d(hratio, Dratio)/d(h, omega_cdm, w0, wa) on three w0waCDM cosmologies.

    The fluid branch is active at every one of them, so all four derivatives
    are live here -- unlike exactly-LCDM, where w0/wa are structurally dead
    (see test_w0_gradient_is_dead_on_the_lambda_branch below).
    """
    w0, wa = W0WA_CASES[case]
    params = CosmoParams(w0=w0, wa=wa)
    _check_ad_vs_fd(case, params,
                    (("h", 1e-4), ("omega_cdm", 1e-4), ("w0", 1e-3), ("wa", 1e-3)))


def test_w0_gradient_is_dead_on_the_lambda_branch():
    """Characterises an UPSTREAM clax defect, not an ap.py one.

    clax/background.py:559 sets `has_fld = (w0 != -1.0) | (wa != 0.0)` and then
    picks the dark-energy sector with `jnp.where(has_fld, ...)` at :567
    (Omega_de) and :601 (rho_fld_ini). At exactly (w0, wa) = (-1, 0) the
    unselected fluid branch carries no gradient, so d/dw0 and d/dwa of ANY
    background quantity — H(z), tau(z), conformal_age, and hence both AP
    ratios — are identically 0.0 under AD, while central FD (which straddles
    the seam into the fluid branch on both sides) returns O(0.2). The value
    itself is continuous across the seam; only the derivative is lost.

    Consequence for the campaign: d(P_l)/d(w0) at a LCDM point is structurally
    zero in clax. Every campaign w0waCDM case has the fluid branch active, so
    they are unaffected.

    When clax/background.py is fixed, DELETE this test and add ("w0", 1e-3) and
    ("wa", 1e-3) back to test_gradients_match_finite_differences' field list.
    """
    params = CosmoParams()
    assert (params.w0, params.wa) == (-1.0, 0.0)   # on the seam by construction
    grads = jax.jacrev(_ratios)(params)
    for field, step in (("w0", 1e-3), ("wa", 1e-3)):
        ad = np.asarray(getattr(grads, field))
        x0 = getattr(params, field)
        fd = np.asarray((_ratios(params.replace(**{field: x0 + step}))
                         - _ratios(params.replace(**{field: x0 - step}))) / (2 * step))
        assert np.all(ad == 0.0), f"d/d{field} no longer dead: ad={ad} — see docstring"
        assert np.abs(fd).max() > 1e-2, f"d/d{field}: FD unexpectedly small, fd={fd}"
