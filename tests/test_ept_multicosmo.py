# tests/test_ept_multicosmo.py
"""Stage layer of the clax-pt vs CLASS-PT campaign (spec §4.2 layer 1):
clax.ept.compute_ept on CLASS-PT's OWN linear P_cb(k), f, r_s, hratio, Dratio,
so that only the EPT stage is under test. 14 distinct cosmologies x
z in (0, 0.38, 0.8) in full mode; FAST_CASES x FAST_Z under --fast
(spec §4.6). Multi-cosmology rule: satisfied by construction.
"""
from __future__ import annotations

import json

import numpy as np
import jax.numpy as jnp

from clax.ept import EPTPrecisionParams, compute_ept
from scripts import validation_cosmologies as vc
from tests import ept_campaign_utils as cu


# ---------------------------------------------------------------------------
# utilities (cosmology-independent numerics -- exempt from the grid rule)
# ---------------------------------------------------------------------------

def test_window_and_thresholds():
    k = np.logspace(np.log10(5e-5), 2, 256)
    w = cu.window(k)
    assert not w[:cu.NSIDE].any() and w[cu.NSIDE] and k[w].max() <= 0.3 and k[~w][cu.NSIDE:].min() > 0.3
    assert set(cu.THRESHOLDS) == set(cu.SPECTRA)
    assert cu.THRESHOLDS["pk_gg_l4"] == 0.02 and cu.THRESHOLDS["pk_gg_l0"] == 0.01
    assert cu.SEAM_THRESHOLDS["pk_lin"] == 1e-3


def test_rel_and_failures():
    k = np.logspace(np.log10(5e-5), 2, 256)
    ref = {"pk_gg_l0": np.ones(256), "pk_gg_l4": np.sin(k)}     # l4 crosses zero: max-relative, not pointwise
    got = {"pk_gg_l0": np.ones(256) * 1.005, "pk_gg_l4": np.sin(k) * 1.03}
    errs = cu.compare_spectra(got, ref, k)
    assert abs(errs["pk_gg_l0"]["err"] - 0.005) < 1e-12
    assert abs(errs["pk_gg_l4"]["err"] - 0.03) < 1e-9
    lines = cu.failures(errs, cu.THRESHOLDS)
    assert lines == [f"pk_gg_l4 3.00% > 2.00% at k={errs['pk_gg_l4']['k']:.3f}"], lines


def test_log_record_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(cu, "LOG_DIR", tmp_path)
    monkeypatch.setattr(cu, "ERROR_LOG", tmp_path / "errors.jsonl")
    cu.log_record(layer="stage", case="lcdm_fiducial", z=0.38, preset="stage",
                  errors={"pk_gg_l0": {"err": 0.003, "k": 0.21}}, seams={"f": 1e-5})
    rec = json.loads((tmp_path / "errors.jsonl").read_text().splitlines()[-1])
    assert rec["layer"] == "stage" and rec["z"] == 0.38 and rec["errors"]["pk_gg_l0"]["err"] == 0.003
    assert rec["ts"].endswith("+00:00") and rec["seams"] == {"f": 1e-5}


# ---------------------------------------------------------------------------
# stage layer
# ---------------------------------------------------------------------------

def pytest_generate_tests(metafunc):
    if "case_z" in metafunc.fixturenames:
        if metafunc.config.getoption("--fast", default=False):
            grid = [(c, vc.FAST_Z) for c in vc.FAST_CASES]
        else:
            grid = [(c, z) for c in vc.distinct_cases() for z in vc.Z_LIST]
        metafunc.parametrize("case_z", grid, ids=[f"{c}-z{z:.2f}" for c, z in grid])


def run_stage(ref: dict, bias: dict):
    """compute_ept on the reference file's own inputs -> (EPTComponents, nine spectra)."""
    e = compute_ept(jnp.asarray(ref["pk_lin"]), jnp.asarray(ref["k_h"]),
                    h=float(ref["h"]), f=float(ref["fz"]), prec=EPTPrecisionParams(),
                    rs_h=float(ref["rs_d"]) * float(ref["h"]),      # r_s(z_d) h in Mpc/h, THIS cosmology
                    hratio=float(ref["hratio"]), Dratio=float(ref["Dratio"]))
    nine = {name: np.asarray(arr) for name, arr in cu.clax_nine(e, bias).items()}
    return e, nine


def _assert_flags(ref, *, ap: bool, cb: bool):
    assert bool(ref["ap"]) is ap and bool(ref["cb"]) is cb, (ref["ap"], ref["cb"])
    if ap:
        assert float(ref["omfid"]) == vc.OMFID
    assert str(ref["kh_convention"]).startswith("h/Mpc"), ref["kh_convention"]


def _check(ref, *, case, z, tag, bias=None, extra=None):
    bias = bias if bias is not None else json.loads(str(ref["bias_json"]))
    e, nine = run_stage(ref, bias)
    k_h = np.asarray(ref["k_h"])
    errs = cu.compare_spectra(nine, ref, k_h)
    rows = cu.compare_rows(cu.pm_from_leaves(e, float(ref["h"])), np.asarray(ref["pk_mult"])[:48], k_h)
    cu.log_record(layer="stage", case=case, z=z, preset=tag, errors=errs,
                  extra={"rows": rows, **(extra or {})})
    bad = cu.failures(errs, cu.THRESHOLDS)
    worst = max(errs.items(), key=lambda kv: kv[1]["err"])
    print(f"{case} z={z:.2f} [{tag}] worst {worst[0]} {100 * worst[1]['err']:.3f}% at k={worst[1]['k']:.3f}")
    assert not bad, f"{case} z={z:.2f} [{tag}]: " + "; ".join(bad)
    return nine, errs


def _windowed_rel(a, b, k_h) -> dict[str, float]:
    """{name: cu.rel(a[name], b[name])} restricted to cu.window(k_h).

    The diagnostics below size a *physical* effect (AP, the cb convention,
    the ppf seam) that lives in the campaign's comparison window (spec §4.3).
    Unwindowed cu.rel over the full 256-point EPT grid (5e-5 to 100 h/Mpc)
    picks up clax's deep-UV spline-extrapolation tail beyond k=0.3 -- exactly
    the region SEAM_THRESHOLDS['pk_lin_tail'] already flags as loose -- which
    can swamp the max-relative metric with an artifact unrelated to the
    effect under test. Verified for the AP case: windowed clax(ap-noap) on
    pk_gg_l0/l2/l4 is 6.015e-3/6.981e-3/1.876e-2, matching CLASS-PT's own
    windowed ap-vs-noap delta (6.015e-3/6.981e-3/1.876e-2) to 4 sig figs --
    the unwindowed max instead lands at k~94 h/Mpc with an 11x smaller ratio.
    """
    m = cu.window(np.asarray(k_h))
    return {n: cu.rel(np.asarray(a[n])[m], np.asarray(b[n])[m]) for n in cu.SPECTRA}


def test_stage_nine_spectra(case_z):
    case, z = case_z
    ref = cu.require_reference(case, z)
    _assert_flags(ref, ap=True, cb=True)
    _check(ref, case=case, z=z, tag="stage")


# --- diagnostics at lcdm_fiducial, z = 0.38 (skip individually when the file is absent) ---

DIAG_CASE, DIAG_Z = "lcdm_fiducial", 0.38


def test_stage_bias_nonzero():
    """Every bias/counterterm/stochastic row is live (spec §4.8): the same
    thresholds must hold with BIAS_NONZERO, otherwise a wrong row was hiding
    behind b2 = bG2 = ... = 0."""
    ref = cu.require_reference(DIAG_CASE, DIAG_Z, bias="nonzero")
    assert json.loads(str(ref["bias_json"])) == vc.BIAS_NONZERO
    _check(ref, case=DIAG_CASE, z=DIAG_Z, tag="stage-biasnz")


def test_stage_cb_vs_m():
    """cb: No file must pass on its own; record the cb-minus-m delta of the
    nine spectra (the size of the cb convention at 0.06 eV, spec §4.5)."""
    ref_m = cu.require_reference(DIAG_CASE, DIAG_Z, cb=False)
    ref_cb = cu.require_reference(DIAG_CASE, DIAG_Z)
    _assert_flags(ref_m, ap=True, cb=False)
    nine_m, _ = _check(ref_m, case=DIAG_CASE, z=DIAG_Z, tag="stage-m")
    nine_cb, _ = _check(ref_cb, case=DIAG_CASE, z=DIAG_Z, tag="stage-cb")
    delta = _windowed_rel(nine_cb, nine_m, ref_cb["k_h"])
    cu.log_record(layer="stage", case=DIAG_CASE, z=DIAG_Z, preset="stage-cb-minus-m",
                  errors={n: {"err": v, "k": float("nan")} for n, v in delta.items()})
    assert delta["pk_gg_l0"] > 1e-4, delta      # the convention is not a no-op at 0.06 eV


def test_stage_ap_off():
    """noap file: ratios must be (1, 1) and the spectra must pass without AP;
    record the AP-on minus AP-off delta (the size of the effect under test)."""
    ref_noap = cu.require_reference(DIAG_CASE, DIAG_Z, ap=False)
    ref_ap = cu.require_reference(DIAG_CASE, DIAG_Z)
    _assert_flags(ref_noap, ap=False, cb=True)
    assert (float(ref_noap["hratio"]), float(ref_noap["Dratio"])) == (1.0, 1.0)
    nine_noap, _ = _check(ref_noap, case=DIAG_CASE, z=DIAG_Z, tag="stage-noap")
    nine_ap, _ = _check(ref_ap, case=DIAG_CASE, z=DIAG_Z, tag="stage-ap")
    delta = _windowed_rel(nine_ap, nine_noap, ref_ap["k_h"])
    cu.log_record(layer="stage", case=DIAG_CASE, z=DIAG_Z, preset="stage-ap-minus-noap",
                  errors={n: {"err": v, "k": float("nan")} for n, v in delta.items()})
    assert delta["pk_gg_l2"] > 1e-3, delta      # Omega_m(fiducial) = 0.3153 vs Omfid 0.31 is a real remap


def test_stage_w0wa_ppf_seam():
    """w0wa: the canonical (use_ppf=yes) file is asserted; the noppf twin is
    compared and its delta recorded (spec §9 ppf seam)."""
    case = "w0wa_m07_m10"
    ref = cu.require_reference(case, DIAG_Z)
    assert bool(ref["use_ppf"]) is True
    ref_noppf = cu.require_reference(case, DIAG_Z, tag="noppf")
    assert bool(ref_noppf["use_ppf"]) is False
    nine_ppf, _ = _check(ref, case=case, z=DIAG_Z, tag="stage-ppf")
    e, nine_noppf = run_stage(ref_noppf, json.loads(str(ref_noppf["bias_json"])))
    delta = _windowed_rel(nine_ppf, nine_noppf, ref["k_h"])
    lin_delta = cu.rel(np.asarray(ref["pk_lin"])[cu.window(ref["k_h"])],
                       np.asarray(ref_noppf["pk_lin"])[cu.window(ref["k_h"])])
    cu.log_record(layer="stage", case=case, z=DIAG_Z, preset="stage-ppf-minus-noppf",
                  errors={n: {"err": v, "k": float("nan")} for n, v in delta.items()},
                  extra={"pk_lin_delta": lin_delta})
    print(f"w0wa ppf-vs-noppf: pk_lin {100 * lin_delta:.3f}%, pk_gg_l0 {100 * delta['pk_gg_l0']:.3f}%")
