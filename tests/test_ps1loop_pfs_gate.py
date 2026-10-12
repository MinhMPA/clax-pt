# tests/test_ps1loop_pfs_gate.py
"""Error-bar gate (spec §5.6): Delta chi^2 of ps_1loop - clax-pt (P0, P2, P4) over the seven
PFS slices of Takada et al. (2014), Gaussian l-l' covariance, 0.01 <= k <= 0.2 h/Mpc, must be
< 1 for each cosmology. vc.PFS_CASES (5) in full mode, lcdm_fiducial under --fast.

What the gate compares. Both codes get the same nominal parameters: c0 = c2 = c4 = C_REF =
2 (Mpc/h)^2 and the BOSS-like fingers-of-god amplitude cfog = CFOG_REF = 500 (Mpc/h)^4. The
assertion is on that fixed-nuisance Delta chi^2, and it fails by design: the excess is the AP
(Alcock-Paczynski) treatment of the k^4 fingers-of-god term (spec §4.2 item 2: ps_1loop weights
the full mu-dependent P_IR in true coordinates, clax-pt applies the l = 4 projection times an
isotropic factor to every multipole, in observed k^2). With the AP map switched off in both codes
Delta chi^2 <= 0.03 at cfog = 500 on all five cosmologies (scratch investigation, jobs 24867-24871:
/lustre/work/n2minh/std/clax/ps1loop/scratch-task8/ap-origin-report.md). That AP-off value sits below
the floor of ps_1loop's AP branch, which is discontinuous with its no-AP branch at zero AP offset:
for lcdm_fiducial at cfog = 500 and offsets x 0.001, Delta chi^2 = 0.128 against 0.014 with AP off
(job 24871); at the ladder's `best` setting (nfft 1024, 201-node mu grid) 0.027 against 0.025
(job 24919). The key numbers are kept in docs/ps1loop-findings.md (findings 3 and 6).

Reported, not asserted (all with the covariance frozen at clax-pt at the gate point, so every
Delta chi^2 below is an exact quadratic form in the nuisance amplitudes):
  - max |dP|/sigma over bins and multipoles;
  - c* (headroom_c): the smallest common k^2 counterterm c = c0 = c2 = c4 at which the survey
    Delta chi^2 reaches 1, cfog held at CFOG_REF. c* = 0 means Delta chi^2 >= 1 already at c = 0;
    it says nothing about which term causes the excess (here it is the fingers-of-god term);
  - cfog* (headroom_cfog): the crossing value of the fingers-of-god amplitude, c held at C_REF,
    both codes at the same cfog (the unprofiled Delta chi^2 is quadratic in cfog);
  - the marginalised Delta chi^2: ps_1loop refits its OWN (c, cfog), shared by the seven slices,
    to clax-pt held at the gate point (flat priors). This says whether PFS could tell the codes
    apart once the nuisance amplitudes float; the shifts of the best fit from (C_REF, CFOG_REF)
    say what that costs."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ps_1loop_jax")

from scripts import validation_cosmologies as vc
from tests import ept_campaign_utils as cu
from tests import pfs_survey as pfs
from tests import ps1loop_bridge as br

# Nuisance points evaluated per slice: (c = c0 = c2 = c4, cfog). "CG" is the gate point.
_POINTS = {"00": (0.0, 0.0), "C0": (pfs.C_REF, 0.0), "0G": (0.0, pfs.CFOG_REF), "CG": (pfs.C_REF, pfs.CFOG_REF)}


def pytest_generate_tests(metafunc):
    if "case" in metafunc.fixturenames:
        fast = metafunc.config.getoption("--fast", default=False)
        metafunc.parametrize("case", ["lcdm_fiducial"] if fast else list(vc.PFS_CASES))


def test_pfs_error_bar_gate(case):
    parts_c, parts_cfog, parts_marg, per_slice, sig_max = [], [], [], [], 0.0
    for sl in pfs.load_slices():
        inp = br.shared_inputs(cu.require_reference(case, sl.z_mid))
        ps, cl = {}, {}
        for key, (c, cfog) in _POINTS.items():
            p, q, kw = br.spectra_at(inp, pfs.bias_point(sl.z_mid, c=c, cfog=cfog), ndens=sl.nbar)
            ps[key], cl[key] = pfs.to_centres(p, kw), pfs.to_centres(q, kw)
        C = pfs.multipole_covariance(cl["CG"], pfs.K_CENTRES, sl.V, pfs.DK)      # frozen at the gate point
        d = {key: ps[key] - cl[key] for key in _POINTS}                           # same nuisance in both codes
        chi = pfs.delta_chi2(d["CG"], C)
        sigma = np.sqrt(np.stack([C[:, i, i] for i in range(3)]))
        sig = float(np.max(np.abs(d["CG"]) / sigma))
        per_slice.append((sl.z_mid, chi))
        sig_max = max(sig_max, sig)
        parts_c.append((d["0G"], d["CG"], C))                                     # c: 0 -> C_REF at cfog = CFOG_REF
        parts_cfog.append((d["C0"], d["CG"], C))                                  # cfog: 0 -> CFOG_REF at c = C_REF
        parts_marg.append((ps["00"] - cl["CG"], ps["C0"] - cl["CG"], ps["0G"] - cl["CG"], C))   # ps refits to clax gate
        br.log_record(layer="pfs", case=case, z=sl.z_mid, preset="slice",
                      errors={"delta_chi2": {"err": chi, "k": float("nan")},
                              "max_dP_over_sigma": {"err": sig, "k": float("nan")}})
    total = sum(chi for _, chi in per_slice)
    c_star = pfs.counterterm_headroom(parts_c, pfs.C_REF)
    cfog_cross = pfs.counterterm_headroom(parts_cfog, pfs.CFOG_REF)
    m_chi, m_c, m_cfog = pfs.marginalised_delta_chi2(parts_marg, pfs.C_REF, pfs.CFOG_REF)
    dc, dcfog = 100 * (m_c - pfs.C_REF) / pfs.C_REF, 100 * (m_cfog - pfs.CFOG_REF) / pfs.CFOG_REF
    br.log_record(layer="pfs", case=case, z=float("nan"), preset="survey",
                  errors={"delta_chi2": {"err": total, "k": float("nan")},
                          "marginalised_delta_chi2": {"err": m_chi, "k": float("nan")}},
                  extra={"per_slice": per_slice, "headroom_c": c_star, "headroom_cfog": cfog_cross,
                         "max_dP_over_sigma": sig_max, "C_REF": pfs.C_REF, "CFOG_REF": pfs.CFOG_REF,
                         "marginalised": {"c": m_c, "cfog": m_cfog, "shift_c_pct": dc, "shift_cfog_pct": dcfog}})
    slices = " ".join(f"{z:.1f}:{chi:.3f}" for z, chi in per_slice)
    summary = (f"max|dP|/sigma {sig_max:.3f}; headroom c* {c_star:.2f} (Mpc/h)^2 (c0=c2=c4 only, cfog fixed, "
               f"0 = already >= 1 at c = 0); "
               f"marginalised dchi2 {m_chi:.3f} at (c*, cfog*) = ({m_c:.2f}, {m_cfog:.0f}) "
               f"[shifts {dc:+.1f}%, {dcfog:+.1f}%]; fixed-nuisance crossing cfog* = {cfog_cross:.0f} (Mpc/h)^4")
    print(f"{case} PFS dchi2 {total:.4f} [{slices}] {summary}")
    if not total < 1.0:
        pytest.fail(f"ERROR {case}: PFS fixed-nuisance Delta chi^2 = {total:.3f} >= 1 at c = {pfs.C_REF}, "
                    f"cfog = {pfs.CFOG_REF:.0f} (per slice {slices}); {summary}", pytrace=False)
