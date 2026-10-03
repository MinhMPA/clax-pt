"""End-to-end layer of the clax-pt vs CLASS-PT campaign (spec §4.2 layer 2):
clax background -> thermodynamics -> perturbations -> compute_ept_from_clax
(omfid=0.31, field="cb") against CLASS-PT's nine spectra, with the spec §8
seams (background, f, P_cb,lin) asserted first so a failing spectrum is
already bisected. One perturbation solve per case (z by tau-interpolation).

All tests are slow (GPU); run via slurm/ptval-e2e-ap.sbatch. One cosmology
costs a full perturbation solve (~48 min on a V100 at `fast`), so the full
sweep is ~11 h serially and is normally sharded across nodes with
PTVAL_E2E_CASES. Environment:
  PTVAL_E2E_CASES  = a,b,...                     -- exact case names (the shard)
  PTVAL_E2E_PREC   = fast (default) | contract   -- clax precision preset
  PTVAL_E2E_SUBSET = fast                        -- FAST_CASES x FAST_Z only
Pass PTVAL_E2E_CASES through the SUBMIT environment (`VAR=a,b sbatch
--export=ALL ...`), never as `--export=ALL,VAR=a,b`: sbatch splits on commas.
Multi-cosmology rule: 14 cases x 3 z (full) / 3 families x 1 z (subset).
"""
from __future__ import annotations

import os
import time
from dataclasses import replace as _dc_replace

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from clax import PrecisionParams
from clax.background import background_solve, sound_horizon_drag
from clax.thermodynamics import thermodynamics_solve
from clax.perturbations import perturbations_solve
from clax.ept import compute_ept_from_clax, ept_inputs_from_clax, ept_kgrid
from clax.ap import ap_ratios
from scripts import validation_cosmologies as vc
from tests import ept_campaign_utils as cu
from tests.pk_test_utils import PK_FAST_PREC, PK_CONTRACT_PREC

pytestmark = pytest.mark.slow

# pt_k_max_cl = 5 Mpc^-1 (>= 6.7 h/Mpc on the grid): the P22/P13 FFTLog
# coefficients read pk_lin far beyond the 3 h/Mpc output cutoff, and
# ept_inputs_from_clax clamps delta beyond pt.k_grid[-1] -- see C2 notes.
E2E_PRESETS = {
    "fast": _dc_replace(PK_FAST_PREC, pt_k_max_cl=5.0, pt_k_chunk_size=0),
    "contract": _dc_replace(PK_CONTRACT_PREC, pt_k_max_cl=5.0, pt_k_chunk_size=0),
}
PRESET_NAME = os.environ.get("PTVAL_E2E_PREC", "fast")
PREC = E2E_PRESETS[PRESET_NAME]
GRAD_PREC = _dc_replace(PrecisionParams.fast_cl(), pt_k_max_cl=5.0, pt_k_chunk_size=20,
                        ncdm_q_size=5)

_PIPELINE: dict[str, tuple] = {}     # case -> (params, bg, pt, seconds)
_SEAMS: dict[tuple, dict] = {}       # (case, z) -> {seam: residual}


def _selected_cases():
    """Cases this process runs: PTVAL_E2E_CASES wins, then PTVAL_E2E_SUBSET.

    The comma-separated env var exists so the 14-cosmology sweep can be sharded
    one cosmology at a time across GPU nodes (each case costs a full
    perturbation solve, ~45 min, and the pipeline cache is per-case so a shard
    loses nothing). `-k` cannot do the split safely: "massive_nu_015" is a
    prefix of both "massive_nu_015_h_high" and "massive_nu_015_omega_cdm_low",
    so a -k shard would silently run three cases where one was meant.
    """
    known = list(vc.distinct_cases())
    raw = os.environ.get("PTVAL_E2E_CASES", "").strip()
    if raw:
        want = [c.strip() for c in raw.split(",") if c.strip()]
        unknown = [c for c in want if c not in known]
        if unknown:
            raise ValueError(
                f"PTVAL_E2E_CASES: unknown case(s) {unknown}; known: {known}")
        return want
    if os.environ.get("PTVAL_E2E_SUBSET") == "fast":
        return list(vc.FAST_CASES)
    return known


def pytest_generate_tests(metafunc):
    cases = _selected_cases()
    if "case_z" in metafunc.fixturenames:
        zs = ((vc.FAST_Z,) if os.environ.get("PTVAL_E2E_SUBSET") == "fast"
              else tuple(vc.Z_LIST))
        grid = [(c, z) for c in cases for z in zs]
        metafunc.parametrize("case_z", grid, ids=[f"{c}-z{z:.2f}" for c, z in grid])
    if "case" in metafunc.fixturenames:
        # the gradient sweep covers one case per family; intersecting with the
        # shard means a shard holding none of them collects no gradient test
        grid = [c for c in cases if c in vc.FAST_CASES]
        metafunc.parametrize("case", grid, ids=list(grid))


def pipeline(case: str):
    """One background + thermo + perturbation solve per case at PREC (cached)."""
    if case not in _PIPELINE:
        params = vc.clax_params(case)
        t0 = time.perf_counter()
        bg = background_solve(params, PREC)
        th = thermodynamics_solve(params, PREC, bg)
        pt = perturbations_solve(params, PREC, bg, th)
        jax.block_until_ready(pt.delta_cb)
        secs = time.perf_counter() - t0
        print(f"[pipeline] {case} preset={PRESET_NAME} n_k={pt.k_grid.shape[0]} {secs:.0f} s")
        _PIPELINE[case] = (params, bg, pt, secs)
    return _PIPELINE[case]


def _seam(case, z, name, value):
    _SEAMS.setdefault((case, z), {})[name] = float(value)
    return float(value)


# ---------------------------------------------------------------------------
# seams (spec §8 order). Each is its own test: a red spectrum below arrives
# with these residuals already in the log.
# ---------------------------------------------------------------------------

def test_seam_background(case_z):
    case, z = case_z
    ref = cu.require_reference(case, z)
    params, bg, _, _ = pipeline(case)
    hr, Dr = ap_ratios(bg, z, vc.OMFID)
    e_hr = _seam(case, z, "hratio", abs(float(hr) / float(ref["hratio"]) - 1.0))
    e_Dr = _seam(case, z, "Dratio", abs(float(Dr) / float(ref["Dratio"]) - 1.0))
    e_rs = _seam(case, z, "rs_d", abs(float(sound_horizon_drag(params)) / float(ref["rs_d"]) - 1.0))
    H = float(bg.H_of_loga.evaluate(jnp.log(1.0 / (1.0 + z))))
    e_H = _seam(case, z, "H_z", abs(H / float(ref["H_z"]) - 1.0))
    bad = cu.failures({k: {"err": v, "k": float("nan")} for k, v in
                       dict(hratio=e_hr, Dratio=e_Dr, rs_d=e_rs, H_z=e_H).items()}, cu.SEAM_THRESHOLDS)
    assert not bad, f"{case} z={z:.2f} background seam: " + "; ".join(bad)


def test_seam_growth_rate_f(case_z):
    case, z = case_z
    ref = cu.require_reference(case, z)
    params, bg, pt, _ = pipeline(case)
    _, f = ept_inputs_from_clax(params, bg, pt, z, field="cb")
    e_f = _seam(case, z, "f", abs(float(f) / float(ref["fz"]) - 1.0))
    assert e_f <= cu.SEAM_THRESHOLDS["f"], f"{case} z={z:.2f} f seam: {100 * e_f:.3f}% > {100 * cu.SEAM_THRESHOLDS['f']:.3f}%"


def _pk_seam(pk_got, pk_ref, k_h):
    w = cu.window(k_h)
    tail = (k_h > cu.K_MAX_COMPARE) & (k_h <= 3.0)
    ratio = np.asarray(pk_got) / np.asarray(pk_ref) - 1.0
    i = int(np.argmax(np.abs(ratio[w])))
    return float(np.max(np.abs(ratio[w]))), float(k_h[w][i]), float(np.max(np.abs(ratio[tail])))


def test_seam_pk_cb_lin(case_z):
    """clax P_cb,lin (field='cb') vs CLASS P_cb (pk_lin), pointwise on the
    window (0.1%) and on 0.3 < k <= 3 h/Mpc (3%); P_m,lin vs pk_m_lin likewise."""
    case, z = case_z
    ref = cu.require_reference(case, z)
    params, bg, pt, _ = pipeline(case)
    k_h = np.asarray(ref["k_h"])
    assert np.allclose(k_h, ept_kgrid()), "reference k_h is not the EPT grid"
    pk_cb, _ = ept_inputs_from_clax(params, bg, pt, z, field="cb")
    e_win, k_worst, e_tail = _pk_seam(pk_cb, ref["pk_lin"], k_h)
    _seam(case, z, "pk_lin", e_win); _seam(case, z, "pk_lin_k", k_worst); _seam(case, z, "pk_lin_tail", e_tail)
    if "pk_m_lin" in ref:
        pk_m, _ = ept_inputs_from_clax(params, bg, pt, z, field="m")
        e_m, _, _ = _pk_seam(pk_m, ref["pk_m_lin"], k_h)
        _seam(case, z, "pk_m_lin", e_m)
    print(f"{case} z={z:.2f} P_cb,lin seam {100 * e_win:.3f}% at k={k_worst:.3f}, tail {100 * e_tail:.2f}%")
    bad = cu.failures({"pk_lin": {"err": e_win, "k": k_worst}, "pk_lin_tail": {"err": e_tail, "k": float("nan")}},
                      cu.SEAM_THRESHOLDS)
    assert not bad, f"{case} z={z:.2f} P_lin seam ({PRESET_NAME}): " + "; ".join(bad)


# ---------------------------------------------------------------------------
# nine spectra
# ---------------------------------------------------------------------------

def test_e2e_spectra(case_z):
    case, z = case_z
    ref = cu.require_reference(case, z)
    params, bg, pt, secs = pipeline(case)
    import json
    bias = json.loads(str(ref["bias_json"]))
    e = compute_ept_from_clax(params, bg, pt, z=z, omfid=vc.OMFID, field="cb")
    nine = {n: np.asarray(a) for n, a in cu.clax_nine(e, bias).items()}
    k_h = np.asarray(ref["k_h"])
    errs = cu.compare_spectra(nine, ref, k_h)
    seams = _SEAMS.get((case, z), {})
    cu.log_record(layer="e2e", case=case, z=z, preset=PRESET_NAME, errors=errs, seams=seams,
                  extra={"solve_seconds": secs, "n_k": int(pt.k_grid.shape[0])})
    bad = cu.failures(errs, cu.THRESHOLDS)
    worst = max(errs.items(), key=lambda kv: kv[1]["err"])
    print(f"{case} z={z:.2f} [e2e/{PRESET_NAME}] worst {worst[0]} {100 * worst[1]['err']:.3f}% at k={worst[1]['k']:.3f}")
    assert not bad, (f"{case} z={z:.2f} [e2e/{PRESET_NAME}]: " + "; ".join(bad)
                     + " | seams: " + ", ".join(f"{k}={v:.2e}" for k, v in seams.items()))


# ---------------------------------------------------------------------------
# pipeline gradient sweep (B6's exemption points here)
# ---------------------------------------------------------------------------

def test_e2e_gradient_finite(case):
    """d/d(omega_cdm) of sum_window pk_gg_l0 through background -> thermo ->
    perturbations -> compute_ept_from_clax(omfid, field='cb') is finite and
    nonzero on one case per family (LCDM, nuLCDM, w0wa)."""
    base = vc.clax_params(case)
    k_h = ept_kgrid()
    w = jnp.asarray(cu.window(k_h))

    def objective(omega_cdm):
        params = base.replace(omega_cdm=omega_cdm)
        bg = background_solve(params, GRAD_PREC)
        th = thermodynamics_solve(params, GRAD_PREC, bg)
        pt = perturbations_solve(params, GRAD_PREC, bg, th)
        e = compute_ept_from_clax(params, bg, pt, z=vc.FAST_Z, omfid=vc.OMFID, field="cb")
        return jnp.sum(jnp.where(w, cu.clax_nine(e, vc.BIAS)["pk_gg_l0"], 0.0))

    t0 = time.perf_counter()
    val, g = jax.value_and_grad(objective)(jnp.asarray(base.omega_cdm))
    jax.block_until_ready(g)
    g, val = float(g), float(val)
    cu.log_record(layer="grad", case=case, z=vc.FAST_Z, preset="grad",
                  errors={}, extra={"d_sum_pk_gg_l0_d_omega_cdm": g, "value": val,
                                    "seconds": time.perf_counter() - t0})
    print(f"{case}: d(sum pk_gg_l0)/d(omega_cdm) = {g:.4e} (value {val:.4e}, {time.perf_counter() - t0:.0f} s)")
    # finiteness and non-vanishing are the contract; the SIGN is not asserted
    # (the omega_cdm derivative of the window-sum is cosmology-dependent).
    assert np.isfinite(g) and g != 0.0, (case, g)
