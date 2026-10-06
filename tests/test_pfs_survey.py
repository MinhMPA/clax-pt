# tests/test_pfs_survey.py
"""PFS survey bookkeeping (spec §3, §5.6): Takada et al. (2014) Table 2 transcription and
the error-bar-gate bias point. Exempt from the multi-cosmology rule: survey numbers and
bias relations, no cosmology-dependent physics."""
from __future__ import annotations

import pytest

from scripts import validation_cosmologies as vc
from tests import pfs_survey as pfs


def test_seven_contiguous_slices_from_0p6_to_2p4():
    s = pfs.load_slices()
    assert len(s) == 7
    assert s[0].z_min == 0.6 and s[-1].z_max == 2.4
    assert all(a.z_max == b.z_min for a, b in zip(s, s[1:]))


def test_bias_column_is_takada_relation_at_midpoint():
    for sl in pfs.load_slices():
        assert abs(sl.b_g - (0.9 + 0.4 * sl.z_mid)) < 0.006, sl


def test_nbar_is_galaxies_per_field_over_volume():
    # n = N_g per field x (1464 / 1.098) fields / V; Table 2 rounds n to 2 significant figures.
    for sl in pfs.load_slices():
        nbar = sl.N_g_per_field * (pfs.AREA_DEG2 / pfs.FOV_DEG2) / sl.V
        assert abs(nbar / sl.nbar - 1.0) < 0.03, (sl.z_min, nbar, sl.nbar)


def test_units_and_volume_scale():
    sl = pfs.load_slices()[0]
    assert sl.V == pytest.approx(0.59e9)          # (Mpc/h)^3, from 0.59 (Gpc/h)^3
    assert sl.nbar == pytest.approx(1.9e-4)       # (h/Mpc)^3


def test_midpoints_are_the_reference_registry():
    assert tuple(sl.z_mid for sl in pfs.load_slices()) == vc.PFS_Z_LIST
    assert set(vc.PFS_CASES) <= set(vc.distinct_cases())
    assert set(vc.IRNO_CASES) <= set(vc.distinct_cases())


def test_bias_point_coevolution():
    z = 0.7
    th = pfs.bias_point(z)
    b1 = 0.9 + 0.4 * z
    assert th["b1"] == pytest.approx(b1)
    assert th["b2"] == pytest.approx(0.412 - 2.143 * b1 + 0.929 * b1**2 + 0.008 * b1**3)
    assert th["bG2"] == pytest.approx(-2.0 / 7.0 * (b1 - 1.0))
    assert th["bGamma3"] == pytest.approx(23.0 / 42.0 * (b1 - 1.0))
    assert th["c0"] == th["c2"] == th["c4"] == pfs.C_REF
    assert th["cfog"] == pfs.CFOG_REF and th["P_shot"] == 1.0
    assert th["a0"] == th["a2"] == 0.0
    assert pfs.bias_point(z, c=0.0)["c0"] == 0.0
