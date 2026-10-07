# tests/test_ps1loop_bridge.py
"""Unit tests for tests/ps1loop_bridge.py (spec §4, §5.1).

Exempt from the multi-cosmology rule: parameter-space stencil algebra, AP and Legendre
algebra, caching and API parity. The fixture-based tests use lcdm_fiducial z = 0.38 only
to exercise ps_1loop's own API against itself, not to compare physics."""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("ps_1loop_jax")
import jax.numpy as jnp

from tests import ept_campaign_utils as cu
from tests import ps1loop_bridge as br


@pytest.fixture(scope="module")
def fid():
    return cu.require_reference("lcdm_fiducial", 0.38)


def test_stencil_recovers_every_monomial_exactly():
    monos = br.EXACT_MONOMIALS + br.MODEL_DIFF_MONOMIALS
    rng = np.random.default_rng(0)
    truth = {br.label(m): rng.normal(size=(3, 5)) for m in monos}

    def F(thetas):
        out = []
        for th in np.asarray(thetas):
            t = dict(zip(br.THETA_NAMES, th))
            val = np.zeros((3, 5))
            for m in monos:
                val = val + truth[br.label(m)] * np.prod([1.0 if v == "1" else t[v] for v in m])
            out.append(val)
        return np.stack(out)

    got = br.coefficients(F, monos)
    for lab, arr in truth.items():
        np.testing.assert_allclose(got[lab], arr, rtol=0, atol=1e-12, err_msg=lab)


def test_ap_alphas_reproduce_clax_true_coordinates():
    hratio, Dratio = 1.031, 0.974
    a_perp, a_para = br.ap_alphas(hratio, Dratio)
    mu, k = np.linspace(-1.0, 1.0, 11), 0.2
    fac = np.sqrt(1.0 + mu**2 * ((a_perp / a_para) ** 2 - 1.0))          # ps_1loop.py:332-334
    mu_ps, k_ps = mu * (a_perp / a_para) / fac, k * fac / a_perp
    ap_fac = np.sqrt(1.0 / Dratio**2 + (hratio**2 - 1.0 / Dratio**2) * mu**2)   # ept.py:1138
    mu_cl, k_cl = mu * hratio / ap_fac, k * ap_fac                            # ept.py:1139-1140
    np.testing.assert_allclose(mu_ps, mu_cl, rtol=1e-14)
    np.testing.assert_allclose(k_ps, k_cl, rtol=1e-14)
    assert br.ap_volume(hratio, Dratio) == pytest.approx(1.0 / (a_perp**2 * a_para), rel=1e-14)


def test_counterterm_map_exact_for_isotropic_spectrum():
    f, c = 0.81, {"c0": 1.3, "c2": -0.7, "c4": 2.1}
    mu = br.GL_MU
    ps = {l: br.project(-2.0 * (c["c0"] + c["c2"] * f * mu**2 + c["c4"] * f**2 * mu**4), l) for l in br.ELLS}
    cb = br.to_clax({**br.zero_theta(), **c}, f=f, ndens=1.0, Pd2d2_0=0.0, hratio=1.0, Dratio=1.0)
    pctr = {l: -br.project(f ** (l // 2) * mu**l, l) for l in br.ELLS}     # ept.py:1207-1209, k = P = 1
    for l, cs in ((0, cb["cs0"]), (2, cb["cs2"]), (4, cb["cs4"])):
        assert 2.0 * cs * pctr[l] == pytest.approx(ps[l], rel=1e-13), l


def test_cfog_is_b4_for_isotropic_spectrum():
    f, b1, cfog, k = 0.81, 1.4, 500.0, 0.2
    mu = br.GL_MU
    ps = {l: br.project(-cfog * f**4 * k**4 * mu**4 * (b1 + f * mu**2) ** 2, l) for l in br.ELLS}
    pctr4 = -br.project(k**2 * f**2 * mu**4, 4)
    poly = br.b4_poly(f, b1)
    for l in br.ELLS:
        assert f**2 * cfog * k**2 * poly[l] * (35.0 / 8.0) * pctr4 == pytest.approx(ps[l], rel=1e-12), l


def test_to_clax_stochastic_and_shot_noise():
    th = {**br.zero_theta(), "P_shot": 1.0, "a0": 0.3, "a2": -0.2, "b2": 0.5}
    hr, Dr, nd = 1.03, 0.97, 2.5e-4
    cb = br.to_clax(th, f=0.8, ndens=nd, Pd2d2_0=40.0, hratio=hr, Dratio=Dr)
    V = hr / Dr**2
    assert cb["Pshot"] == pytest.approx(V / nd - 0.25 * 0.5**2 * 40.0)
    assert cb["a2"] == pytest.approx(V * hr**2 * -0.2 / nd)
    kt2_over_k2 = 1.0 / Dr**2 + (hr**2 - 1.0 / Dr**2) * br.GL_MU**2
    assert cb["a0"] == pytest.approx(br.project(V * 0.3 / nd * kt2_over_k2, 0), rel=1e-13)


def test_ps_model_cache_keys_on_rbao_and_overrides_before_first_call():
    m1, m2, m3 = br.ps_model(99.08), br.ps_model(99.08), br.ps_model(110.0)
    assert m1 is m2 and m1 is not m3
    assert m1.rbao == 99.08 and m3.rbao == 110.0 and m1.ks == br.IR_KS
    assert br.ps_model(99.08, mu_grid=201)._mu.shape == (201,)
    assert br.ps_model(99.08)._mu.shape == (51,)


def test_rbao_is_rs_times_h(fid):
    inp = br.shared_inputs(fid)
    assert inp["rbao"] == pytest.approx(float(fid["rs_d"]) * float(fid["h"]), rel=1e-9)
    assert inp["rbao"] == pytest.approx(99.08, abs=0.05)        # 147.10 Mpc x 0.6736


def test_rbao_reaches_the_computation(fid):
    inp = br.shared_inputs(fid)
    kw = br.window_k(inp)
    th = np.array([[{**br.zero_theta(), "b1": 1.0}[n] for n in br.THETA_NAMES]])
    a = br.ps_batch(br.ps_model(inp["rbao"]), inp, kw)(th)
    b = br.ps_batch(br.ps_model(110.0), inp, kw)(th)
    # If rbao never reached Sigma^2_BAO, a == b to the last bit (a repeat evaluation of one
    # model differs by exactly 0.0). The physical effect is small: rbao 99.08 -> 110 Mpc/h
    # moves Sigma^2 by 0.9% (19.21 -> 19.39 (Mpc/h)^2), and the LO resummed tree
    # P_nw + e^{-x} P_w (1 + x) has no first-order Sigma^2 dependence, so P_l moves by
    # <= 4e-5 locally and 2.7e-6 on this global-max metric (job 22452). Bound: 270x below that.
    assert np.max(np.abs(a - b)) / np.max(np.abs(a)) > 1e-8


def test_three_in_one_projection_matches_get_pk_ell_ref(fid):
    inp = br.shared_inputs(fid)
    model, kw = br.ps_model(inp["rbao"]), br.window_k(inp)
    th = {**br.zero_theta(), "b1": 2.0, "b2": -0.4, "c0": 3.0, "cfog": 500.0, "P_shot": 1.0}
    a_perp, a_para = br.ap_alphas(inp["hratio"], inp["Dratio"])
    pk_data = {"k": jnp.asarray(inp["k_h"]), "pk": jnp.asarray(inp["pk"])}
    params = br.ps_params(th, h=inp["h"], f=inp["f"], ndens=br.NDENS_PROBE)
    ours = br.ps_batch(model, inp, kw)(np.array([[th[n] for n in br.THETA_NAMES]]))[0]
    for i, l in enumerate(br.ELLS):
        ref = np.asarray(model.get_pk_ell_ref(jnp.asarray(kw), l, a_perp, a_para, pk_data, params))
        np.testing.assert_allclose(ours[i], ref, rtol=1e-10, atol=0, err_msg=f"l={l}")


def test_coef_error_metrics():
    kw = np.linspace(0.01, 0.3, 5)
    e = br.coef_error(np.full(5, 1.01), np.ones(5), kw, scale=1.0)
    assert e["metric"] == "self" and e["err"] == pytest.approx(0.01)
    e = br.coef_error(np.full(5, 2e-3), np.zeros(5), kw, scale=1.0)
    assert e["metric"] == "monomial-scale" and e["err"] == pytest.approx(2e-3)
    bad = np.ones(5)
    bad[2] = np.nan
    e = br.coef_error(bad, np.ones(5), kw, scale=1.0)
    assert e["metric"] == "non-finite" and not np.isfinite(e["err"]) and e["k"] == kw[2]


def test_gate_flags_non_finite_and_attributes():
    kw = np.linspace(0.01, 0.3, 4)
    ps = np.ones((3, 4))
    ps[1, 2] = np.nan
    res = {"ps": {"b1": ps}, "clax": {"b1": np.ones((3, 4))}, "classpt": {"b1": np.ones((3, 4))}}
    errs, lines = br.gate(res, kw)
    assert len(lines) == 1 and lines[0].startswith("ERROR b1:l2") and "non-finite" in lines[0]
    assert "vs CLASS-PT" in lines[0]


def test_log_record_carries_both_commits(tmp_path, monkeypatch):
    monkeypatch.setattr(br, "LOG_DIR", tmp_path)
    monkeypatch.setattr(br, "ERROR_LOG", tmp_path / "errors.jsonl")
    br.log_record(layer="coef", case="lcdm_fiducial", z=0.38, preset="exact",
                  errors={"b1:l0": {"err": 1e-3, "k": 0.2, "metric": "self"}})
    rec = json.loads((tmp_path / "errors.jsonl").read_text().splitlines()[-1])
    assert rec["layer"] == "coef" and rec["ps1loop_sha"] != "" and "git_sha" in rec
