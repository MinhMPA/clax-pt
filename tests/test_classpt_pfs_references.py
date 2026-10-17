# tests/test_classpt_pfs_references.py
"""The 38 new CLASS-PT reference files (spec §3): 5 cosmologies x 7 PFS slices with AP,
and 3 IR-off files at z = 0.38. Presence is asserted, never skipped -- a missing file must
not let the error-bar gate skip into a pass (Review Focus 4). Multi-cosmology rule:
5 and 3 cosmologies by construction."""
from __future__ import annotations

import json

import numpy as np
import pytest

from scripts import validation_cosmologies as vc
from tests import ept_campaign_utils as cu


def _paths():
    pfs = [vc.reference_path(c, z) for c in vc.PFS_CASES for z in vc.PFS_Z_LIST]
    irno = [vc.reference_path(c, 0.38, ap=False, tag=vc.IRNO_TAG) for c in vc.IRNO_CASES]
    return pfs + irno


def test_all_new_references_present():
    missing = [str(p.relative_to(cu.ROOT)) for p in _paths() if not p.exists()]
    assert not missing, f"ERROR {len(missing)} reference files missing: {missing[:5]} -- run slurm/classpt-refgen-pfs.sbatch"


@pytest.mark.parametrize("case", vc.PFS_CASES)
def test_pfs_reference_flags_and_growth(case):
    fz = []
    for z in vc.PFS_Z_LIST:
        ref = cu.load_reference(case, z)
        assert ref is not None, f"ERROR missing {case} z={z}"
        assert bool(ref["ap"]) and bool(ref["cb"]) and float(ref["omfid"]) == vc.OMFID
        assert abs(float(ref["z"]) - z) < 1e-12
        assert json.loads(str(ref["params_json"]))["IR resummation"] == "Yes"
        assert str(ref["kh_convention"]).startswith("h/Mpc")
        assert np.all(np.isfinite(ref["pk_mult"])) and np.all(ref["pk_lin"] > 0)
        fz.append(float(ref["fz"]))
    assert all(0.0 < a < b < 1.0 for a, b in zip(fz, fz[1:])), f"ERROR f(z) not increasing for {case}: {fz}"


@pytest.mark.parametrize("case", vc.IRNO_CASES)
def test_irno_reference_flags(case):
    ref = cu.load_reference(case, 0.38, ap=False, tag=vc.IRNO_TAG)
    assert ref is not None, f"ERROR missing IR-off file for {case}"
    assert not bool(ref["ap"]) and bool(ref["cb"])
    assert json.loads(str(ref["params_json"]))["IR resummation"] == "No"
    ap_file = cu.load_reference(case, 0.38)
    np.testing.assert_array_equal(ref["k_h"], ap_file["k_h"])
