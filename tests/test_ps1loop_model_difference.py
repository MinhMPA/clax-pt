# tests/test_ps1loop_model_difference.py
"""Model-difference closure (spec §4.2, §5.2) and counterterm-map exactness with IR off
(spec §5.3).

Closure: for c0, c2, c4, cfog, cfog*b1, cfog*b1^2 and a0, the measured ps_1loop - clax-pt
coefficient difference must EQUAL the closed-form model difference evaluated on ps_1loop's own
P_IR.

Omitted projections (controller amendments, Oct 8, 2026; user decision Oct 9, 2026): the 13
br.CLOSURE_LINES -- six F_G2 (C1) lines, the l = 2, 4 moments of the isotropic loops b2*b2,
bG2*bG2, b2*bG2, and b1*b2:l4 -- are projections whose integrand clax-pt's mu-loop builds
(mirroring CLASS-PT) but never adds to P_l. The closure asserts these omitted projections PLUS
ps_1loop's IR-resummation scheme difference (ps: loop[P_nw] + e^{-x(mu)}(loop[P] - loop[P_nw]);
clax-pt / CLASS-PT: the channel scheme), a documented model difference with a closed form on
ps's own P_IR (spec §4.2): br.ps_scheme_predicted, built from clax's own mu-loop inputs and its
IR-off channels. The rebuild of clax's mu-loop must first reproduce every projection clax DOES
keep to <= 1e-10 (it guards the port). ps_1loop runs at its default nfft = 256 (locked
decision); the closure remainder (<= 0.56% of max|clax b1*bG2:l2|, BAO-phased) shrinks
0.52% -> 0.14% at nfft = 512, which the Task 9 convergence ladder records.

42 files in full mode (closures, reproduction) and the 14 z = 0 files (IR off); vc.FAST_CASES at
vc.FAST_Z under --fast. Multi-cosmology rule: by construction."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ps_1loop_jax")

from scripts import validation_cosmologies as vc
from tests import ept_campaign_utils as cu
from tests import ps1loop_bridge as br

REPRO_TOL = 1e-10


def pytest_generate_tests(metafunc):
    fast = metafunc.config.getoption("--fast", default=False)
    cases = list(vc.FAST_CASES) if fast else list(vc.distinct_cases())
    if "case_z" in metafunc.fixturenames:
        grid = [(c, vc.FAST_Z) for c in cases] if fast else [(c, z) for c in cases for z in vc.Z_LIST]
        metafunc.parametrize("case_z", grid, ids=[f"{c}-z{z:.2f}" for c, z in grid])
    if "case0" in metafunc.fixturenames:
        metafunc.parametrize("case0", cases)


def test_model_difference_closes(case_z):
    case, z = case_z
    inp = br.shared_inputs(cu.require_reference(case, z))
    res, kw = br.coefficient_triplet(inp, br.MODEL_DIFF_MONOMIALS)
    pred = br.coefficients(br.model_difference_factory(br.ps_model(inp["rbao"]), inp, kw),
                           br.MODEL_DIFF_MONOMIALS)
    # gate(clax + measured - predicted vs clax) == max|measured - predicted| / |clax coefficient|
    closure = {"ps": {lab: res["clax"][lab] + (res["ps"][lab] - res["clax"][lab]) - pred[lab]
                      for lab in res["clax"]},
               "clax": res["clax"], "classpt": {}}
    errs, lines = br.gate(closure, kw)
    size = {lab: float(np.max(np.abs(pred[lab])) / max(np.max(np.abs(res["clax"][lab])), 1e-300))
            for lab in pred}
    br.log_record(layer="closure", case=case, z=z, preset="model-difference", errors=errs,
                  extra={"model_difference_size": size})
    print(f"{case} z={z:.2f} [closure] {br.worst(errs)}; largest model difference "
          f"{max(size, key=size.get)} {100 * max(size.values()):.2f}%")
    assert not lines, f"{case} z={z:.2f}:\n" + "\n".join(lines)


def test_counterterm_map_exact_with_ir_off(case0):
    inp = br.shared_inputs(cu.require_reference(case0, 0.0))
    assert not br.has_ap(inp)
    res, kw = br.coefficient_triplet(inp, br.MODEL_DIFF_MONOMIALS, ir=False)
    errs, lines = br.gate(res, kw)
    br.log_record(layer="irno", case=case0, z=0.0, preset="ctr-map-exact", errors=errs)
    print(f"{case0} z=0 [ir-off map] {br.worst(errs)}")
    assert not lines, f"{case0} z=0 IR off:\n" + "\n".join(lines)


def test_omitted_projections_reproduce_clax(case_z):
    """The rebuilt mu-loop projections equal every projection clax keeps (F_G2 l=0,2, the
    isotropic loops' l=0, the b1b2 / b1bG2 rows incl. the unused l=4 ones) to <= 1e-10 of
    their integrand's largest kept row."""
    case, z = case_z
    inp = br.shared_inputs(cu.require_reference(case, z))
    mask = cu.window(inp["k_h"])
    kw = inp["k_h"][mask]
    e, cap = br.capture_gl_inputs(inp)
    proj = br.omitted_projections(inp, kw, captured=(e, cap))
    kept = {name: sign * np.asarray(getattr(e, field), dtype=float)[mask]
            for name, (field, sign, _) in br.KEPT_PROJECTIONS.items()}
    scale: dict[str, float] = {}
    for name, (_, _, fam) in br.KEPT_PROJECTIONS.items():
        scale[fam] = max(scale.get(fam, 0.0), float(np.max(np.abs(kept[name]))))
    errs, lines = {}, []
    for name, (field, sign, fam) in br.KEPT_PROJECTIONS.items():
        d = np.abs(proj[name] - kept[name])
        if not np.all(np.isfinite(d)):
            errs[name] = {"err": float("nan"), "k": float(kw[int(np.argmax(~np.isfinite(d)))]), "metric": "non-finite"}
        else:
            i = int(np.argmax(d))
            errs[name] = {"err": float(d[i] / max(scale[fam], 1e-300)), "k": float(kw[i]), "metric": "family-scale"}
        if not errs[name]["err"] <= REPRO_TOL:
            lines.append(f"ERROR rebuilt {name} vs {'-' if sign < 0 else ''}{field}: {errs[name]['err']:.2e} "
                         f"> {REPRO_TOL:.0e} at k={errs[name]['k']:.3f}")
    br.log_record(layer="closure", case=case, z=z, preset="omitted-reproduction", errors=errs)
    w = max(errs, key=lambda n: np.inf if not np.isfinite(errs[n]["err"]) else errs[n]["err"])
    print(f"{case} z={z:.2f} [omitted reproduction] worst {w} {errs[w]['err']:.2e} at k={errs[w]['k']:.3f}")
    assert not lines, f"{case} z={z:.2f}:\n" + "\n".join(lines)


def test_omitted_projections_close(case_z):
    """closure = clax + (ps - clax) - predicted on the 13 br.CLOSURE_LINES, normal gate thresholds;
    predicted = the omitted projections plus ps's IR-scheme difference (br.ps_scheme_predicted).
    Logged per line: the measured difference, the prediction, and its split into the pure omitted
    projection (br.omitted_predicted) and the scheme term (the rest)."""
    case, z = case_z
    inp = br.shared_inputs(cu.require_reference(case, z))
    res, kw = br.coefficient_triplet(inp, br.CLOSURE_MONOMIALS)
    labels = [br.label(m) for m in br.CLOSURE_MONOMIALS]
    clax = {lab: res["clax"][lab] for lab in labels}
    captured = br.capture_gl_inputs(inp)
    pred = br.ps_scheme_predicted(inp, kw, clax, captured=captured)
    omitted = br.omitted_predicted(br.omitted_projections(inp, kw, captured=captured))
    closure = {"ps": {lab: clax[lab] + (res["ps"][lab] - clax[lab]) - pred[lab] for lab in pred},
               "clax": clax, "classpt": {}}
    others = tuple(f"{lab}:{ch}" for lab in pred for ch in br.CHANNELS
                   if f"{lab}:{ch}" not in br.CLOSURE_LINES)
    errs, lines = br.gate(closure, kw, exclude=others)
    measured, _ = br.gate({"ps": {lab: res["ps"][lab] for lab in pred}, "clax": clax, "classpt": {}}, kw)
    errs = {n: errs[n] for n in br.CLOSURE_LINES}
    size, size_omitted, size_scheme = {}, {}, {}
    for n in br.CLOSURE_LINES:
        lab, ch = n.split(":")
        i = br.CHANNELS.index(ch)
        sc = float(np.max(np.abs(clax[lab])))
        size[n] = br.coef_error(clax[lab][i] + pred[lab][i], clax[lab][i], kw, scale=sc)["err"]
        size_omitted[n] = br.coef_error(clax[lab][i] + omitted[lab][i], clax[lab][i], kw, scale=sc)["err"]
        size_scheme[n] = br.coef_error(clax[lab][i] + pred[lab][i] - omitted[lab][i], clax[lab][i], kw,
                                       scale=sc)["err"]
    br.log_record(layer="closure", case=case, z=z, preset="omitted-projections+ps-scheme", errors=errs,
                  extra={"measured": {n: measured[n] for n in br.CLOSURE_LINES}, "predicted_size": size,
                         "omitted_size": size_omitted, "scheme_size": size_scheme})
    w = max(errs, key=lambda n: np.inf if not np.isfinite(errs[n]["err"]) else errs[n]["err"])
    print(f"{case} z={z:.2f} [omitted closure] {br.worst(errs)} (measured {100 * measured[w]['err']:.3f}%, "
          f"predicted {100 * size[w]:.3f}%)")
    assert not lines, f"{case} z={z:.2f}:\n" + "\n".join(lines)
