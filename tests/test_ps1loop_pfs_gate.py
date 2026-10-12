# tests/test_ps1loop_pfs_gate.py
"""Error-bar gate (spec §5.6): Delta chi^2 of ps_1loop - clax-pt (P0, P2, P4) over the seven
PFS slices of Takada et al. (2014), Gaussian l-l' covariance, 0.01 <= k <= 0.2 h/Mpc, must be
< 1 for each cosmology. Reports per-slice Delta chi^2, max |dP|/sigma and the counterterm
headroom c*. vc.PFS_CASES (5) in full mode, lcdm_fiducial under --fast."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ps_1loop_jax")

from scripts import validation_cosmologies as vc
from tests import ept_campaign_utils as cu
from tests import pfs_survey as pfs
from tests import ps1loop_bridge as br


def pytest_generate_tests(metafunc):
    if "case" in metafunc.fixturenames:
        fast = metafunc.config.getoption("--fast", default=False)
        metafunc.parametrize("case", ["lcdm_fiducial"] if fast else list(vc.PFS_CASES))


def test_pfs_error_bar_gate(case):
    parts, per_slice, sig_max = [], [], 0.0
    for sl in pfs.load_slices():
        inp = br.shared_inputs(cu.require_reference(case, sl.z_mid))
        d = {}
        for c in (0.0, pfs.C_REF):
            ps, cl, kw = br.spectra_at(inp, pfs.bias_point(sl.z_mid, c=c), ndens=sl.nbar)
            d[c] = pfs.to_centres(ps - cl, kw)
            if c == pfs.C_REF:
                C = pfs.multipole_covariance(pfs.to_centres(cl, kw), pfs.K_CENTRES, sl.V, pfs.DK)
        chi = pfs.delta_chi2(d[pfs.C_REF], C)
        sigma = np.sqrt(np.stack([C[:, i, i] for i in range(3)]))
        sig = float(np.max(np.abs(d[pfs.C_REF]) / sigma))
        per_slice.append((sl.z_mid, chi))
        sig_max = max(sig_max, sig)
        parts.append((d[0.0], d[pfs.C_REF], C))
        br.log_record(layer="pfs", case=case, z=sl.z_mid, preset="slice",
                      errors={"delta_chi2": {"err": chi, "k": float("nan")},
                              "max_dP_over_sigma": {"err": sig, "k": float("nan")}})
    total = sum(chi for _, chi in per_slice)
    c_star = pfs.counterterm_headroom(parts, pfs.C_REF)
    br.log_record(layer="pfs", case=case, z=float("nan"), preset="survey",
                  errors={"delta_chi2": {"err": total, "k": float("nan")}},
                  extra={"per_slice": per_slice, "headroom_c": c_star, "C_REF": pfs.C_REF})
    slices = " ".join(f"{z:.1f}:{chi:.3f}" for z, chi in per_slice)
    print(f"{case} PFS dchi2 {total:.4f} [{slices}] max|dP|/sigma {sig_max:.3f} headroom c* {c_star:.2f} (Mpc/h)^2")
    assert total < 1.0, (f"ERROR {case}: PFS Delta chi^2 = {total:.3f} >= 1 (per slice {slices}); "
                         f"counterterm headroom c* = {c_star:.2f} (Mpc/h)^2 at C_REF = {pfs.C_REF}")
