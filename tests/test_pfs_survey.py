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


import numpy as np


def test_covariance_isotropic_closed_forms():
    # Isotropic P(k, mu) = P0: C00 = 2 P0^2/N_k, C22 = 25 * (2/5) P0^2 / N_k = 10 P0^2/N_k, C02 = 0.
    # Built ONLY from the P passed in -- no extra 1/nbar (Review Focus 2).
    k = np.array([0.1, 0.2])
    P = np.zeros((3, 2))
    P[0] = [1.0e4, 5.0e3]
    C = pfs.multipole_covariance(P, k, V=1e9, dk=0.01)
    Nk = 1e9 * k**2 * 0.01 / (2 * np.pi**2)
    np.testing.assert_allclose(C[:, 0, 0], 2 * P[0] ** 2 / Nk, rtol=1e-12)
    np.testing.assert_allclose(C[:, 1, 1], 10 * P[0] ** 2 / Nk, rtol=1e-12)
    np.testing.assert_allclose(C[:, 0, 1], 0.0, atol=1e-12 * np.max(C[:, 0, 0]))


def test_delta_chi2_and_headroom():
    C = np.broadcast_to(np.eye(3), (4, 3, 3)).copy()
    d1 = np.ones((3, 4))                          # at c = 2: chi2 = 12
    assert pfs.delta_chi2(d1, C) == pytest.approx(12.0)
    # d0 = 0 -> chi2(c) = 3 c^2 -> c* = 1/sqrt(3)
    assert pfs.counterterm_headroom([(np.zeros((3, 4)), d1, C)], 2.0) == pytest.approx(1 / np.sqrt(3))
    assert pfs.counterterm_headroom([(np.full((3, 4), 2.0), d1, C)], 2.0) == 0.0     # already >= 1
    assert pfs.counterterm_headroom([(np.zeros((3, 4)), np.zeros((3, 4)), C)], 2.0) == np.inf


def test_bins_and_interpolation():
    assert pfs.K_CENTRES[0] == pytest.approx(0.015) and pfs.K_CENTRES[-1] == pytest.approx(0.195)
    assert len(pfs.K_CENTRES) == 19 and pfs.DK == 0.01
    kw = np.geomspace(1e-3, 0.3, 80)
    P = np.stack([kw**-1.0, kw**0.5, np.log(kw)])
    np.testing.assert_allclose(pfs.to_centres(P, kw), np.stack([pfs.K_CENTRES**-1.0, pfs.K_CENTRES**0.5,
                                                               np.log(pfs.K_CENTRES)]), rtol=1e-6)
