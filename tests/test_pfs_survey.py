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
    # Built ONLY from the P passed in: P(k, mu) already carries the shot noise, so no extra 1/nbar.
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


def test_covariance_anisotropic_matches_quadrature():
    # All nine C_ll' entries, P0, P2, P4 all nonzero at two k, against scipy.integrate.quad of
    # (2l+1)(2l'+1)/N_k * int L_l L_l' P(k,mu)^2 dmu (independent of the Gauss-Legendre table).
    from scipy.integrate import quad
    from scipy.special import eval_legendre
    k = np.array([0.07, 0.19])
    P = np.array([[1.3e4, 4.0e3], [-5.0e3, 2.2e3], [1.1e3, -9.0e2]])           # (3, 2)
    V, dk = 7.9e8, 0.01
    C = pfs.multipole_covariance(P, k, V, dk)
    Nk = V * k**2 * dk / (2 * np.pi**2)
    ells = (0, 2, 4)
    for ik in range(2):
        def pkmu(mu):
            return sum(P[i, ik] * eval_legendre(l, mu) for i, l in enumerate(ells))
        for i, l in enumerate(ells):
            for j, lp in enumerate(ells):
                integral, _ = quad(lambda mu: eval_legendre(l, mu) * eval_legendre(lp, mu) * pkmu(mu) ** 2,
                                   -1.0, 1.0, epsabs=0.0, epsrel=1e-12, limit=200)
                want = (2 * l + 1) * (2 * lp + 1) * integral / Nk[ik]
                np.testing.assert_allclose(C[ik, i, j], want, rtol=1e-10, err_msg=f"k={k[ik]} l={l} l'={lp}")


@pytest.mark.parametrize("seed, flip", [(0, 1.0), (1, 1.0), (2, -1.0), (3, -1.0)])
def test_headroom_root_with_cross_term(seed, flip):
    # d0 not collinear with D (b != 0) and e < 1: the root must satisfy chi2(c*) = 1 to 1e-12,
    # for b > 0 (flip = +1 aligns d0 with D) and b < 0 (flip = -1).
    rng = np.random.default_rng(seed)
    Nk = 5
    M = rng.normal(size=(Nk, 3, 3))
    C = M @ M.transpose(0, 2, 1) + 3.0 * np.eye(3)                              # symmetric positive definite
    D = rng.normal(size=(3, Nk))
    d0 = 0.1 * rng.normal(size=(3, Nk)) + flip * 0.05 * D
    c_ref = 2.0
    d1 = d0 + c_ref * D
    b = float(np.sum(D.T * pfs._cinv(C, d0)))
    e = pfs.delta_chi2(d0, C)
    assert abs(b) > 1e-3 and e < 1.0
    cstar = pfs.counterterm_headroom([(d0, d1, C)], c_ref)
    assert cstar > 0.0
    assert pfs.delta_chi2(d0 + cstar * D, C) == pytest.approx(1.0, abs=1e-12, rel=0)


def test_marginalised_fit_recovers_truth_and_matches_lstsq():
    # r(c, g) = r00 + c Dc + g Dg exactly. Put the best fit at a known (c0, g0) by making the
    # leftover r(c0, g0) = u orthogonal (in the C^-1 inner product) to Dc and Dg; then
    # chi2_min = u^T C^-1 u. Slices share (c, g). Independent check: whitened least squares.
    rng = np.random.default_rng(7)
    c_ref, g_ref, c0, g0 = 2.0, 500.0, 3.1, 420.0
    slices = []
    for Nk in (4, 6):
        M = rng.normal(size=(Nk, 3, 3))
        C = M @ M.transpose(0, 2, 1) + 3.0 * np.eye(3)
        Dc, Dg = rng.normal(size=(3, Nk)), 1e-2 * rng.normal(size=(3, Nk))
        slices.append((C, Dc, Dg, rng.normal(size=(3, Nk))))
    def ip(a, b_, C):
        return float(np.sum(a.T * pfs._cinv(C, b_)))

    # shared-nuisance projection: u = w - coef . (Dc, Dg) is C^-1-orthogonal to Dc and Dg over ALL slices
    A = np.zeros((2, 2))
    rhs = np.zeros(2)
    for C, Dc, Dg, w in slices:
        dirs = (Dc, Dg)
        for i in range(2):
            rhs[i] += ip(dirs[i], w, C)
            for j in range(2):
                A[i, j] += ip(dirs[i], dirs[j], C)
    coef = np.linalg.solve(A, rhs)                                              # u = w - coef . dirs is orthogonal
    parts, chi_u = [], 0.0
    for C, Dc, Dg, w in slices:
        u = w - coef[0] * Dc - coef[1] * Dg
        r00 = u - c0 * Dc - g0 * Dg
        parts.append((r00, r00 + c_ref * Dc, r00 + g_ref * Dg, C))
        chi_u += ip(u, u, C)
    chi, c_star, g_star = pfs.marginalised_delta_chi2(parts, c_ref, g_ref)
    assert chi == pytest.approx(chi_u, rel=1e-10)
    assert c_star == pytest.approx(c0, rel=1e-9) and g_star == pytest.approx(g0, rel=1e-9)
    # independent: stack whitened residuals and solve with lstsq
    rows, ys = [], []
    for (r00, rC0, r0G, C) in parts:
        L = np.linalg.cholesky(C)                                               # C = L L^T, whiten with L^-1
        for ik in range(C.shape[0]):
            Li = np.linalg.inv(L[ik])
            rows.append(np.stack([Li @ ((rC0 - r00)[:, ik] / c_ref), Li @ ((r0G - r00)[:, ik] / g_ref)], axis=1))
            ys.append(-Li @ r00[:, ik])
    x, res, *_ = np.linalg.lstsq(np.concatenate(rows), np.concatenate(ys), rcond=None)
    assert x[0] == pytest.approx(c_star, rel=1e-9) and x[1] == pytest.approx(g_star, rel=1e-9)
    assert chi == pytest.approx(float(res[0]), rel=1e-9)


def test_to_centres_refuses_a_short_window():
    kw = np.geomspace(0.02, 0.15, 30)                       # does not cover 0.015 ... 0.195
    with pytest.raises(ValueError, match="does not cover"):
        pfs.to_centres(np.ones((3, kw.size)), kw)


def test_bias_point_cfog_keyword():
    assert pfs.bias_point(0.9)["cfog"] == pfs.CFOG_REF
    th = pfs.bias_point(0.9, cfog=0.0)
    assert th["cfog"] == 0.0 and th["c0"] == pfs.C_REF
