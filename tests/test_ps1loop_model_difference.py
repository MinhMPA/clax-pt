# tests/test_ps1loop_model_difference.py
"""Model-difference closure (spec §4.2, §5.2) and counterterm-map exactness with IR off
(spec §5.3).

Closure: for c0, c2, c4, cfog, cfog*b1, cfog*b1^2 and a0, the measured ps_1loop - clax-pt
coefficient difference must EQUAL the closed-form model difference evaluated on ps_1loop's own
P_IR (ps's _get_pk_irres_rsd), at the gate thresholds 1% / 1% / 2% (l = 0 / 2 / 4) on every line.
USER DECISIONS Oct 14, 2026 add two rules:
  * Discrimination, as on the 13 lines below (br.DISCRIM_MIN / br.DISCRIM_RATIO), where the closed form
    predicts a NONZERO difference: on such a line whose measured |ps - clax| exceeds 1e-3 of the gate
    denominator, the closure residual must be <= 0.5 x the measured difference, so a counterterm,
    fingers-of-god or a0 line cannot pass merely by being small. Where the prediction is identically zero
    -- the c_j monomial on its own multipole (c0:l0, c2:l2, c4:l4), where ps's -2 c_j f^{j/2} mu_t^j k_t^2 P_IR
    and clax's 2 cs_j Pctr_j are the same expression on the same P_IR, the exactly mapped a0 rows, and two
    rows that coincide only without AP (list below) -- there is no model difference to test, so the rule is
    not applied and the line stays under the threshold gate.
    "Identically zero" is a test, not a tunable: max|predicted| <= PRED_IDENTICALLY_ZERO = 1e-12 of the gate
    denominator, i.e. zero to round-off (the two expressions differ only in the order of the operations).
    The ratio is logged on every line either way.
  * ps_1loop runs at nfft = PS_NFFT_CTR_CLOSURE = 512 in this closure, the derivative gate's setting
    (tests/test_ps1loop_gradients.py PS_NFFT_GATE): at ps's default 256, c4:l2 -- a line with a real model
    difference -- closes to only 0.58-0.79 of its measured difference in 5 files (k = 0.191 h/Mpc, job
    25326); at nfft 512 to 0.155-0.182 in the same files (job 25482), at nfft 1024 with a 201-node mu grid to
    0.07-0.09 (job 25328): ps's own resolution, not the closed form, set that residual. At nfft 512 all 613
    rule-bearing line x file checks pass (worst ratio 0.199, c0:l4 omega_cdm_low z = 0.8; job 25482). Both
    decisions concern this closure only; the 13-line closure, the reproduction test and the IR-off map
    exactness below keep ps's default nfft = 256.
  The lines with an identically zero prediction are, by the test, c0:l0, c2:l2, c4:l4, a0:l0 and a0:l4 in every
  file and, without AP (z = 0), a0:l2 and cfog*b1*b1:l4 (max|predicted| <= 2.9e-14 of the denominator there,
  >= 3.0e-4 on every other line; job 25482).
Finding recorded with the exemption (docs/ps1loop-findings.md, finding 9): on the diagonal lines the measured
ps - clax is 0.10-0.18% of the coefficient's scale at k ~ 0.18-0.24 h/Mpc in 25 of 28 AP files and <= 6e-7 at
z = 0 (no AP) at ps's default resolution (job 25326). At nfft 512 it is <= 0.055% (median 0.037% on c0:l0;
job 25482); at nfft 1024 with a 201-node mu grid it shrinks a median 3.4x (13-16x near the fiducial geometry,
2-3x at large AP offsets), leaving 0.04-0.075% (job 25328): partly ps's AP-table interpolation (finding 3),
partly a piece that grows with the AP offset and is not isolated.
Logged per line: measured, predicted, residual, ratio, whether the rule applied.

Omitted projections (user decisions Oct 8 and Oct 10, 2026; spec §4.2 items 4-5): the 13
br.CLOSURE_LINES -- six F_G2 (C1) lines, the l = 2, 4 moments of the isotropic loops b2*b2,
bG2*bG2, b2*bG2, and b1*b2:l4 -- are projections whose integrand clax-pt's mu-loop builds
(mirroring CLASS-PT) but never adds to P_l. The closure asserts these omitted projections PLUS
ps_1loop's IR-resummation scheme difference (ps: loop[P_nw] + e^{-x(mu)}(loop[P] - loop[P_nw]);
clax-pt / CLASS-PT: isotropically damped loop input, p_lo(mu) on F_G2 only), a documented model
difference whose closed form, br.ps_scheme_predicted, is built from clax's own IR-off channels on
clax's P_nw, Sigma^2 and AP geometry. Three checks guard it:
  * reproduction: the rebuild of clax's mu-loop equals every projection clax keeps to <= 1e-10;
  * kept coefficient: the rebuilt full clax-scheme projection minus the omitted part equals clax's
    reported coefficient to <= 1e-10, on every multipole of the closure monomials;
  * discrimination: on each closure line whose measured |ps - clax| exceeds 1e-3 of the gate
    denominator, the closure residual must be <= 0.5 x the measured difference, so a line cannot
    pass the gate merely by being small. Lines measured at <= 1e-3 (0.1% of the coefficient's
    scale, e.g. the l = 4 isotropic loops) are physically negligible and held to the gate only.
The 13-line closure runs ps at its default nfft = 256 (locked decision). Its remainder is non-smooth in k and
on the 1% gate worst for b1*bG2:l2, 0.56% of its max|clax| at z = 0; the candidate is ps
interpolating its loops in k on its 256-point grid (not O(P_w^2), which cancels, nor AP, since that
worst case has none).
Scratch measurement (Oct 10, 2026; 3 files at z = 0.38): the b1*bG2:l2 remainder 0.52% -> 0.14% at
nfft = 512; the Task 9 convergence ladder did not re-measure this per-line remainder (docs/ps1loop-findings.md,
finding 1).

42 files in full mode (closures, reproduction) and the 14 z = 0 files (IR off); vc.FAST_CASES at
vc.FAST_Z under --fast. Multi-cosmology rule: by construction."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ps_1loop_jax")

from scripts import validation_cosmologies as vc
from tests import ept_campaign_utils as cu
from tests import ps1loop_bridge as br

REPRO_TOL = 1e-10      # rebuilt vs clax's own projections / coefficients, fraction of their scale
# ps_1loop's FFTLog sampling for the counterterm / fingers-of-god / a0 closure (USER DECISION Oct 14, 2026; module
# docstring): the derivative gate's setting. At ps's default 256 its resolution, not the closed form, sets the c4:l2
# residual (ratio 0.58-0.79 in 5 files, job 25326; 0.155-0.182 at 512, job 25482).
PS_NFFT_CTR_CLOSURE = 512
# A predicted difference is identically zero when max|predicted| <= this fraction of the gate denominator: zero to
# round-off, where the two codes' models are the same expression. A test of the algebra, not a tolerance to tune.
PRED_IDENTICALLY_ZERO = 1e-12


def pytest_generate_tests(metafunc):
    fast = metafunc.config.getoption("--fast", default=False)
    cases = list(vc.FAST_CASES) if fast else list(vc.distinct_cases())
    if "case_z" in metafunc.fixturenames:
        grid = [(c, vc.FAST_Z) for c in cases] if fast else [(c, z) for c in cases for z in vc.Z_LIST]
        metafunc.parametrize("case_z", grid, ids=[f"{c}-z{z:.2f}" for c, z in grid])
    if "case0" in metafunc.fixturenames:
        metafunc.parametrize("case0", cases)


def test_model_difference_closes(case_z):
    """Counterterm / fingers-of-god / a0 closure (module docstring): threshold gate on every line; the
    discrimination rule where the predicted difference is not identically zero; ps at PS_NFFT_CTR_CLOSURE."""
    case, z = case_z
    inp = br.shared_inputs(cu.require_reference(case, z))
    res, kw = br.coefficient_triplet(inp, br.MODEL_DIFF_MONOMIALS, nfft=PS_NFFT_CTR_CLOSURE)
    model = br.ps_model(inp["rbao"], do_irres=True, nfft=PS_NFFT_CTR_CLOSURE)   # the instance res["ps"] used
    pred = br.coefficients(br.model_difference_factory(model, inp, kw), br.MODEL_DIFF_MONOMIALS)
    # gate(clax + measured - predicted vs clax) == max|measured - predicted| / |clax coefficient|
    closure = {"ps": {lab: res["clax"][lab] + (res["ps"][lab] - res["clax"][lab]) - pred[lab]
                      for lab in res["clax"]},
               "clax": res["clax"], "classpt": {}}
    errs, lines = br.gate(closure, kw)
    measured, _ = br.gate(res, kw)                 # the direct ps - clax difference, same denominators
    size = {lab: float(np.max(np.abs(pred[lab])) / max(np.max(np.abs(res["clax"][lab])), 1e-300))
            for lab in pred}
    per_line, ratios, exempt = {}, {}, {}
    for n in errs:
        lab, ch = n.split(":")
        i = br.CHANNELS.index(ch)
        scale = float(np.max(np.abs(res["clax"][lab])))
        # max|predicted| over the gate denominator (the same metric as measured and residual)
        p = br.coef_error(res["clax"][lab][i] + pred[lab][i], res["clax"][lab][i], kw, scale=scale)["err"]
        m, r = measured[n]["err"], errs[n]["err"]
        zero = bool(p <= PRED_IDENTICALLY_ZERO)
        per_line[n] = {"measured": m, "predicted": p, "residual": r, "k_residual": errs[n]["k"],
                       "rule": "not applied: prediction identically zero" if zero else "applied"}
        if m > br.DISCRIM_MIN:
            per_line[n]["ratio"] = r / m
            if zero:
                exempt[n] = m
            else:
                ratios[n] = r / m
                if not ratios[n] <= br.DISCRIM_RATIO:
                    lines.append(f"ERROR discrimination {n}: closure residual {100 * r:.3f}% > {br.DISCRIM_RATIO} x "
                                 f"measured {100 * m:.3f}% (ratio {ratios[n]:.3f}) at k={errs[n]['k']:.3f}")
    br.log_record(layer="closure", case=case, z=z, preset="model-difference", errors=errs,
                  extra={"model_difference_size": size, "lines": per_line, "ps_nfft": PS_NFFT_CTR_CLOSURE,
                         "pred_identically_zero": PRED_IDENTICALLY_ZERO})
    wr = max(ratios, key=ratios.get) if ratios else None
    we = max(exempt, key=exempt.get) if exempt else None
    print(f"{case} z={z:.2f} [closure, ps nfft {PS_NFFT_CTR_CLOSURE}] {br.worst(errs)}; largest model difference "
          f"{max(size, key=size.get)} {100 * max(size.values()):.2f}%; worst ratio "
          + (f"{wr} {100 * per_line[wr]['measured']:.3f}->{100 * per_line[wr]['residual']:.3f}% "
             f"r={ratios[wr]:.3f} (of {len(ratios)} lines > {br.DISCRIM_MIN:.0e})" if wr else "n/a")
          + "; zero-prediction lines above 1e-3 (rule not applied): "
          + (f"{len(exempt)}, largest {we} {100 * exempt[we]:.3f}%" if we else "none"))
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
    Also asserts the kept-coefficient check and the discrimination check (module docstring). Logged
    per line: measured, predicted, residual, residual/measured, the split of the prediction into the
    pure omitted projection (br.omitted_predicted) and the scheme term, the kept-coefficient error,
    and clax vs the CLASS-PT twin (tie-breaker, not asserted)."""
    case, z = case_z
    inp = br.shared_inputs(cu.require_reference(case, z))
    res, kw = br.coefficient_triplet(inp, br.CLOSURE_MONOMIALS)
    labels = [br.label(m) for m in br.CLOSURE_MONOMIALS]
    clax = {lab: res["clax"][lab] for lab in labels}
    captured = br.capture_gl_inputs(inp)
    pred = br.ps_scheme_predicted(inp, kw, clax, captured=captured)
    omitted = br.omitted_predicted(br.omitted_projections(inp, kw, captured=captured))
    full_clax = br.clax_scheme_projections(inp, kw, captured=captured)
    closure = {"ps": {lab: clax[lab] + (res["ps"][lab] - clax[lab]) - pred[lab] for lab in labels},
               "clax": clax, "classpt": {}}
    others = tuple(f"{lab}:{ch}" for lab in labels for ch in br.CHANNELS
                   if f"{lab}:{ch}" not in br.CLOSURE_LINES)
    errs, lines = br.gate(closure, kw, exclude=others)
    measured, _ = br.gate({"ps": {lab: res["ps"][lab] for lab in labels}, "clax": clax, "classpt": {}}, kw)
    errs = {n: errs[n] for n in br.CLOSURE_LINES}

    # kept coefficient: <X_clax>_l - omitted == clax's reported coefficient, every multipole
    kept = {}
    for lab in labels:
        sc = max(float(np.max(np.abs(clax[lab]))), 1e-300)
        for i, ch in enumerate(br.CHANNELS):
            d = np.abs(full_clax[lab][i] - omitted[lab][i] - clax[lab][i])
            kept[f"{lab}:{ch}"] = float(np.max(d) / sc) if np.all(np.isfinite(d)) else float("nan")
    for n, e in kept.items():
        if not e <= REPRO_TOL:
            lines.append(f"ERROR kept-coefficient {n}: rebuilt clax-scheme minus omitted vs clax {e:.2e} > {REPRO_TOL:.0e}")

    per_line, ratios = {}, {}
    for n in br.CLOSURE_LINES:
        lab, ch = n.split(":")
        i = br.CHANNELS.index(ch)
        sc = float(np.max(np.abs(clax[lab])))
        m, r = measured[n]["err"], errs[n]["err"]
        rec = {"measured": m, "residual": r, "k_residual": errs[n]["k"],
               "predicted": br.coef_error(clax[lab][i] + pred[lab][i], clax[lab][i], kw, scale=sc)["err"],
               "omitted": br.coef_error(clax[lab][i] + omitted[lab][i], clax[lab][i], kw, scale=sc)["err"],
               "scheme": br.coef_error(clax[lab][i] + pred[lab][i] - omitted[lab][i], clax[lab][i], kw,
                                       scale=sc)["err"],
               "clax_vs_classpt": (br.coef_error(clax[lab][i], res["classpt"][lab][i], kw, scale=sc)["err"]
                                   if lab in res.get("classpt", {}) else None)}
        if m > br.DISCRIM_MIN:
            rec["ratio"] = ratios[n] = r / m
            if not ratios[n] <= br.DISCRIM_RATIO:
                lines.append(f"ERROR discrimination {n}: closure residual {100 * r:.3f}% > {br.DISCRIM_RATIO} x "
                             f"measured {100 * m:.3f}% (ratio {ratios[n]:.3f}) at k={errs[n]['k']:.3f}")
        per_line[n] = rec
    br.log_record(layer="closure", case=case, z=z, preset="omitted-projections+ps-scheme", errors=errs,
                  extra={"lines": per_line, "kept_coefficient": kept})
    wr = max(ratios, key=ratios.get) if ratios else None
    wk = max(kept, key=lambda n: np.inf if not np.isfinite(kept[n]) else kept[n])
    print(f"{case} z={z:.2f} [omitted closure] {br.worst(errs)}; worst ratio "
          + (f"{wr} {ratios[wr]:.3f} (of {len(ratios)} lines > {br.DISCRIM_MIN:.0e})" if wr else "n/a")
          + f"; kept-coefficient worst {kept[wk]:.1e}")
    assert not lines, f"{case} z={z:.2f}:\n" + "\n".join(lines)
