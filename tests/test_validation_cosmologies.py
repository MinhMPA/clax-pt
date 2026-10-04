"""Locks the campaign grid (spec §4.1), CLASS-PT mapping (§5.3) and layout (§4.8).

Cosmology-independent plumbing: exempt from the multi-cosmology rule.
"""
import math

import numpy as np
import pytest

from scripts import validation_cosmologies as vc
from tests import conftest


def test_case_counts_and_families():
    assert len(vc.CASES) == 15
    assert {len(v) for v in vc.FAMILIES.values()} == {5}
    assert sorted(sum(vc.FAMILIES.values(), ())) == sorted(vc.CASES)
    assert vc.distinct_cases() == [c for c in vc.CASES if c not in vc.ALIASES]
    assert len(vc.distinct_cases()) == 14


def test_alias_is_physically_identical():
    assert vc.cosmo_params("massive_nu_006") == vc.cosmo_params("lcdm_fiducial")
    assert vc.canonical_case("massive_nu_006") == "lcdm_fiducial"
    assert vc.canonical_case("h_high") == "h_high"


def test_fiducial_matches_clax_defaults():
    pytest.importorskip("jax")
    from clax import CosmoParams
    p = CosmoParams()
    for key, val in vc.FIDUCIAL.items():
        assert getattr(p, key) == val, key
    assert vc.Y_HE_CLAX == p.Y_He


def test_conftest_grids_are_subsets_of_cases():
    assert conftest.COSMOLOGY_GRID_LCDM == {n: vc.CASES[n] for n in vc.FAMILIES["lcdm"]}
    for name, ov in conftest.COSMOLOGY_GRID_NULCDM.items():
        assert vc.CASES[name] == ov


def test_classpt_mapping_lcdm():
    prm = vc.classpt_params("lcdm_fiducial", z_list=(0.0, 0.38))
    assert math.isclose(prm["A_s"], 2.1e-9, rel_tol=1e-9)
    assert prm["N_ncdm"] == 1 and prm["m_ncdm"] == 0.06
    assert prm["N_ur"] == 2.0328 and prm["T_ncdm"] == 0.71611
    assert prm["YHe"] == vc.Y_HE_CLAX
    assert prm["z_pk"] == "0,0.38"
    assert prm["Omfid"] == "0.31" and prm["AP"] == "Yes" and prm["cb"] == "Yes"
    assert prm["non linear"] == "PT" and prm["P_k_max_h/Mpc"] == 100.0
    assert "w0_fld" not in prm and "use_ppf" not in prm


def test_classpt_mapping_w0wa_and_flags():
    prm = vc.classpt_params("w0wa_m07_m10", z_list=(0.8,), ap=False, cb=False, use_ppf=False)
    assert prm["w0_fld"] == -0.7 and prm["wa_fld"] == -1.0 and prm["Omega_Lambda"] == 0.0
    assert prm["AP"] == "No" and "Omfid" in prm and prm["cb"] == "No"
    assert prm["use_ppf"] == "no"


def test_legacy_params_refuse_cb_and_keep_exact_dict():
    with pytest.raises(ValueError):
        vc.classpt_params_from(vc.LEGACY_CLASSPT_FIDUCIAL, z_list=(0.38,), cb=True)
    prm = vc.classpt_params_from(vc.LEGACY_CLASSPT_FIDUCIAL, z_list=(0.38,), cb=False, yhe=None)
    assert prm["A_s"] == 2.0989e-9 and "N_ncdm" not in prm and "YHe" not in prm
    assert "N_ur" not in prm  # CLASS-PT default 3.044, exactly as the legacy run


def test_reference_paths():
    p = vc.reference_path("h_high", 0.38)
    assert p == vc.REFERENCE_ROOT / "h_high" / "z0.380_ap_omfid0.31_cb.npz"
    assert vc.reference_path("h_high", 0.0, ap=False, cb=False).name == "z0.000_noap_m.npz"
    assert vc.reference_path("h_high", 0.8, bias="nonzero").name == "z0.800_ap_omfid0.31_cb_biasnz.npz"
    assert vc.reference_path("w0wa_m07_m10", 0.38, tag="noppf").name == "z0.380_ap_omfid0.31_cb_noppf.npz"
    # aliases resolve to the canonical directory
    assert vc.reference_path("massive_nu_006", 0.38) == vc.reference_path("lcdm_fiducial", 0.38)


def test_fast_subset():
    assert vc.FAST_CASES == ("lcdm_fiducial", "massive_nu_015", "w0wa_m07_m10")
    assert vc.FAST_Z == 0.38 and vc.FAST_Z in vc.Z_LIST


def test_kgrid_twin_matches_clax():
    pytest.importorskip("jax")
    from clax.ept import EPTPrecisionParams, ept_kgrid
    assert np.array_equal(ept_kgrid(EPTPrecisionParams()), vc.ept_kgrid_numpy())
    assert vc.ept_kgrid_numpy().shape == (256,)
