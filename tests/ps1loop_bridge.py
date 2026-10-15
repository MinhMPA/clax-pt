# tests/ps1loop_bridge.py
"""Bridge for the clax-pt vs ps_1loop_jax second-oracle validation.

Spec: docs/superpowers/specs/2026-10-05-clax-pt-vs-ps1loop-validation-design.md (glossary:
CONTEXT.md). Shared-input comparison: both codes receive one CLASS-PT reference file's own
P_cb(k) [(Mpc/h)^3 on k_h in h/Mpc], growth rate f, r_s(z_d) h [Mpc/h] and AP ratios, so a
residual is one-loop numerics or a documented model difference (spec §4.2).

Conventions pinned here, each with its test in tests/test_ps1loop_bridge.py:
  * AP: ps_1loop alpha_perp = Dratio, alpha_para = 1/hratio, Jacobian V = hratio/Dratio^2
    (ps_1loop.py:320-335 vs ept.py:1135-1141).
  * Basis: THETA_NAMES is ps_1loop's; `to_clax` is the basis translation (spec §4.1).
  * Coefficient spectra: exact stencils on {-1, 0, 1} per parameter. P_l has degree <= 2 in
    every parameter, so the stencils have no truncation error -- only round-off.
"""
from __future__ import annotations

import datetime as _dt
import functools
import itertools
import json
import subprocess
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from clax.ept import EPTPrecisionParams, _ir_resummation_jax, compute_ept, pk_gg_l0, pk_gg_l2, pk_gg_l4
from clax.interpolation import CubicSpline
from scripts import classpt_assembly as ca
from tests import ept_campaign_utils as cu

THETA_NAMES = ("b1", "b2", "bG2", "bGamma3", "c0", "c2", "c4", "cfog", "P_shot", "a0", "a2")
ELLS = (0, 2, 4)
CHANNELS = ("l0", "l2", "l4")
COEF_THRESHOLDS = {"l0": 0.01, "l2": 0.01, "l4": 0.02}   # = cu.THRESHOLDS (spec §5.1)
K_NL = 0.45            # h/Mpc; clax-pt hard-codes (k/0.45)^2 in a0/a2 (ept.py pk_gg_l0, classy.pyx:4910)
NDENS_PROBE = 3.0e-4   # (h/Mpc)^3; only rescales the stochastic monomials
IR_KS = 0.2            # h/Mpc; ps_1loop default k_s, = CLASS-PT's Sigma^2_BAO cut
STRUCTURAL_ZERO = 1e-12
# Discrimination rule (Task 5; reused by Task 7's derivative closure): wherever the measured ps - clax
# difference exceeds DISCRIM_MIN of the gate denominator, the closure residual (measured minus the
# documented model difference) must be at most DISCRIM_RATIO of it, so a line cannot pass the closure
# merely by being small. Below DISCRIM_MIN a line is held to the gate threshold only.
DISCRIM_MIN = 1e-3
DISCRIM_RATIO = 0.5

EXACT_MONOMIALS = (("1",), ("b1",), ("b1", "b1"), ("b2",), ("b1", "b2"), ("b2", "b2"),
                   ("bG2",), ("b1", "bG2"), ("bG2", "bG2"), ("b2", "bG2"),
                   ("bGamma3",), ("b1", "bGamma3"), ("P_shot",), ("a2",))
# The six F_G2-type projections that CLASS-PT (and clax-pt, ept.py:2483-2489, 2535-2538) omit and ps_1loop
# keeps; asserted by the closed-form closure in tests/test_ps1loop_model_difference.py (Task 5);
# see docs/ps1loop-findings.md. (b1*bG2:l4 and b1*bGamma3:l4 would pass a 2% gate only because they
# are small relative to their monopole, not because the codes agree.)
C1_LINES = ("bG2:l4", "bGamma3:l4", "b1*bG2:l2", "b1*bG2:l4", "b1*bGamma3:l2", "b1*bGamma3:l4")
# Isotropic-loop projections CLASS-PT omits (it keeps only the l=0 moment of Id2d2, IG2G2, Id2G2 in the
# mu-loop; clax/ept.py _gl_multipoles) and ps_1loop keeps; asserted by closure in Task 5.
LOOP_LINES = ("b2*b2:l2", "b2*b2:l4", "bG2*bG2:l2", "bG2*bG2:l4", "b2*bG2:l2", "b2*bG2:l4")
# AP-generated l=4 row that CLASS-PT computes but never uses: its C loop fills pm[40] P_4_b1b2 (and
# pm[41] P_4_b1bG2, whose closure rides on b1*bG2:l4 above), classy.pyx:4925-4931 never adds either to
# P_4, and clax-pt mirrors that (pk_gg_l4; the leaves stay in EPTComponents.Pk_4_b1b2 / Pk_4_b1bG2).
AP_L4_LINES = ("b1*b2:l4",)
CLOSURE_LINES = C1_LINES + LOOP_LINES + AP_L4_LINES
# The monomials that own the 13 closure lines (tests/test_ps1loop_model_difference.py).
CLOSURE_MONOMIALS = (("bG2",), ("bGamma3",), ("b1", "bG2"), ("b1", "bGamma3"),
                     ("b2", "b2"), ("bG2", "bG2"), ("b2", "bG2"), ("b1", "b2"))
MODEL_DIFF_MONOMIALS = (("c0",), ("c2",), ("c4",), ("cfog",), ("cfog", "b1"),
                        ("cfog", "b1", "b1"), ("a0",))
NO_TIEBREAK = {"a0", "a2"}   # CLASS-PT's twin has no scale-dependent stochastic rows

GL_MU, GL_W = np.polynomial.legendre.leggauss(40)

LOG_DIR = cu.ROOT / "test_logs" / "ps1loop"
ERROR_LOG = LOG_DIR / "errors.jsonl"


# --------------------------------------------------------------------------- algebra

def zero_theta() -> dict[str, float]:
    return {n: 0.0 for n in THETA_NAMES}


def label(mono: tuple[str, ...]) -> str:
    return "*".join(mono)


def legendre_np(ell: int, mu):
    mu = np.asarray(mu, dtype=float)
    return {0: np.ones_like(mu), 2: 0.5 * (3.0 * mu**2 - 1.0),
            4: (35.0 * mu**4 - 30.0 * mu**2 + 3.0) / 8.0}[ell]


def project(vals, ell: int):
    """(2l+1)/2 * int_{-1}^{1} L_l(mu) vals(mu) dmu on the 40 GL nodes (last axis)."""
    return (2 * ell + 1) / 2.0 * np.sum(GL_W * legendre_np(ell, GL_MU) * np.asarray(vals), axis=-1)


def b4_poly(f, b1) -> dict[int, float]:
    """The isotropic factors clax-pt multiplies b4 k^2 (35/8) Pk_ctr4 by (ept.py pk_gg_l0/l2/l4 P_b4)."""
    return {0: f**2 / 9.0 + 2.0 * f * b1 / 7.0 + b1**2 / 5.0,
            2: (4.0 / 693.0) * (70.0 * f**2 + 165.0 * f * b1 + 99.0 * b1**2),
            4: (8.0 / 5005.0) * (210.0 * f**2 + 390.0 * f * b1 + 143.0 * b1**2)}


# --------------------------------------------------------------------------- shared inputs

def shared_inputs(ref: dict) -> dict:
    h = float(ref["h"])
    return {"k_h": np.asarray(ref["k_h"], dtype=float), "pk": np.asarray(ref["pk_lin"], dtype=float),
            "h": h, "f": float(ref["fz"]),
            "rbao": round(float(ref["rs_d"]) * h, 9),           # r_s(z_d) h [Mpc/h], THIS cosmology
            "hratio": float(ref["hratio"]), "Dratio": float(ref["Dratio"]),
            "Pd2d2_0": float(ref["Pd2d2_0"]), "pk_mult": np.asarray(ref["pk_mult"], dtype=float)}


def window_k(inp: dict) -> np.ndarray:
    return inp["k_h"][cu.window(inp["k_h"])]


def ap_alphas(hratio, Dratio):
    """(alpha_perp, alpha_para) for ps_1loop from CLASS-PT/clax (hratio, Dratio)."""
    return Dratio, 1.0 / hratio


def ap_volume(hratio, Dratio):
    return hratio / Dratio**2                      # = 1/(alpha_perp^2 alpha_para), ept.py:1141


def has_ap(inp: dict) -> bool:
    return not (inp["hratio"] == 1.0 and inp["Dratio"] == 1.0)


# --------------------------------------------------------------------------- ps_1loop

@functools.lru_cache(maxsize=None)
def ps_model(rbao: float, *, do_irres: bool = True, nfft: int = 256, mu_grid: int = 51):
    """One PowerSpectrum1Loop per (rbao, settings). Its methods are jitted with
    static_argnames=['self'], so NOTHING may be mutated after the first call (Review Focus 1);
    the mu_grid override (spec §7 ladder) is applied here, before any call."""
    from ps_1loop_jax import PowerSpectrum1Loop
    model = PowerSpectrum1Loop(do_irres=do_irres, rbao=rbao, ks=IR_KS, nfft=nfft)
    if mu_grid != 51:
        model._mu = jnp.linspace(0.0, 1.0, mu_grid)   # TEST-ONLY: AP remap grid (ps_1loop.py:45)
    return model


def ps_params(theta: dict, *, h, f, ndens) -> dict:
    return {"h": h, "f": f,
            "bias": {n: theta[n] for n in ("b1", "b2", "bG2", "bGamma3")},
            "ctr": {n: theta[n] for n in ("c0", "c2", "c4", "cfog")},
            "stoch": {n: theta[n] for n in ("P_shot", "a0", "a2")},
            "k_nl": K_NL, "ndens": ndens}


def ps_multipoles_jax(model, k_h, pk, h, f, a_perp, a_para, theta, k, *, ap, ndens, num=256):
    """All three multipoles from ONE P(k, mu), with the numerics of
    PowerSpectrum1Loop.get_pk_ell / get_pk_ell_ref (ps_1loop.py:308-347): mu = linspace(0,1,num),
    Simpson's rule, weight (2l+1) L_l(mu). Parity: test_three_in_one_projection_matches_get_pk_ell_ref."""
    import quadax
    from ps_1loop_jax.utils_math import legendre
    pk_data = {"k": k_h, "pk": pk}
    params = ps_params(theta, h=h, f=f, ndens=ndens)
    k = jnp.atleast_1d(k)
    mu = jnp.linspace(0.0, 1.0, num)
    if ap:
        pkmu = model.get_pkmu_ref(k, mu, a_perp, a_para, pk_data, params)
    else:
        pkmu = model.get_pkmu(k, mu, pk_data, params)
    return jnp.stack([quadax.simpson(pkmu * ((2 * l + 1) * legendre(l, mu))[None, :], x=mu, axis=1)
                      for l in ELLS])


@functools.partial(jax.jit, static_argnames=("model", "ap", "num"))
def _ps_batch_jit(model, k_h, pk, h, f, a_perp, a_para, thetas, k, ndens, *, ap, num):
    def one(th):
        return ps_multipoles_jax(model, k_h, pk, h, f, a_perp, a_para,
                                 dict(zip(THETA_NAMES, th)), k, ap=ap, ndens=ndens, num=num)
    return jax.vmap(one)(thetas)


def ps_batch(model, inp, kw, *, ndens=NDENS_PROBE, num=256, ap=None):
    """F(thetas) -> (M, 3, Nw) ps_1loop multipoles on kw. ap=None takes ps's AP branch iff the inputs carry AP
    (has_ap); ap=True forces the AP branch (the convergence ladder uses it at alpha_perp = alpha_para = 1)."""
    a_perp, a_para = ap_alphas(inp["hratio"], inp["Dratio"])
    args = (jnp.asarray(inp["k_h"]), jnp.asarray(inp["pk"]), inp["h"], inp["f"], a_perp, a_para)
    branch = has_ap(inp) if ap is None else bool(ap)

    def F(thetas):
        return np.asarray(_ps_batch_jit(model, *args, jnp.asarray(thetas, dtype=float),
                                        jnp.asarray(kw), ndens, ap=branch, num=num))
    return F


# --------------------------------------------------------------------------- translation

def to_clax(theta: dict, *, f, ndens, Pd2d2_0, hratio, Dratio) -> dict:
    """Basis translation ps_1loop -> clax-pt / CLASS-PT (spec §4.1). Exact except the model
    differences of spec §4.2, which test_ps1loop_model_difference.py closes. jnp-safe."""
    V = ap_volume(hratio, Dratio)
    knl2 = (0.45 / K_NL) ** 2
    c0, c2, c4 = theta["c0"], theta["c2"], theta["c4"]
    return {"b1": theta["b1"], "b2": theta["b2"], "bG2": theta["bG2"], "bGamma3": theta["bGamma3"],
            # Legendre content of mu^2 = L0/3 + 2L2/3 and mu^4 = L0/5 + 4L2/7 + 8L4/35
            "cs0": c0 + f * c2 / 3.0 + f**2 * c4 / 5.0,
            "cs2": c2 + 6.0 * f * c4 / 7.0,
            "cs4": c4,
            "cs": 0.0,
            "b4": theta["cfog"],
            # ps_1loop: P_shot/ndens x Jacobian V, P_d2d2(k->0) subtracted (ps_1loop.py:124-135, _p22);
            # clax-pt: Pshot in observed coordinates, +0.25 b2^2 Pd2d2_0 added back (classy.pyx:4911).
            "Pshot": V * theta["P_shot"] / ndens - 0.25 * theta["b2"] ** 2 * Pd2d2_0,
            # ps_1loop.py:577-591 in true coordinates: a0 (k_t/k_nl)^2 -> l=0 moment; a2 k_t^2 mu_t^2 = k^2 mu^2 hratio^2
            "a0": V * theta["a0"] / ndens * knl2 * (2.0 / (3.0 * Dratio**2) + hratio**2 / 3.0),
            "a2": V * hratio**2 * theta["a2"] / ndens * knl2}


# --------------------------------------------------------------------------- clax-pt and CLASS-PT

def clax_ept(inp, *, ir=True, pk=None, f=None, hratio=None, Dratio=None, ir_traced=False):
    """compute_ept on the shared inputs; the overrides let Task 7 trace f, AP and P_lin.

    ir_traced=True feeds compute_ept the traced IR split (_ir_resummation_jax, as
    compute_ept_from_clax does) so a traced pk can be differentiated; the default keeps
    the numpy IR path validated in Tasks 4-6."""
    pk_in = jnp.asarray(inp["pk"]) if pk is None else pk
    k_in = jnp.asarray(inp["k_h"])
    kw = {}
    if ir and ir_traced:
        kw["_ir_precomputed"] = _ir_resummation_jax(pk_in, np.asarray(inp["k_h"]), inp["rbao"], float(inp["h"]))
    return compute_ept(pk_in, k_in,
                       h=inp["h"], f=inp["f"] if f is None else f,
                       prec=EPTPrecisionParams(ir_resummation=ir), rs_h=inp["rbao"],
                       hratio=inp["hratio"] if hratio is None else hratio,
                       Dratio=inp["Dratio"] if Dratio is None else Dratio, **kw)


def clax_multipoles_jax(e, cb: dict):
    b = (cb["b1"], cb["b2"], cb["bG2"], cb["bGamma3"])
    return jnp.stack([pk_gg_l0(e, *b, cs0=cb["cs0"], Pshot=cb["Pshot"], b4=cb["b4"], a0=cb["a0"], a2=cb["a2"]),
                      pk_gg_l2(e, *b, cs2=cb["cs2"], b4=cb["b4"], a2=cb["a2"]),
                      pk_gg_l4(e, *b, cs4=cb["cs4"], b4=cb["b4"])])


def clax_batch(e, inp, mask, *, ndens=NDENS_PROBE):
    def one(th):
        cb = to_clax(dict(zip(THETA_NAMES, th)), f=inp["f"], ndens=ndens, Pd2d2_0=e.Pd2d2_0,
                     hratio=inp["hratio"], Dratio=inp["Dratio"])
        return clax_multipoles_jax(e, cb)[:, mask]
    vm = jax.vmap(one)
    return lambda thetas: np.asarray(vm(jnp.asarray(thetas, dtype=float)))


def classpt_multipoles(inp, cb: dict) -> np.ndarray:
    """CLASS-PT at any bias point, rebuilt from its stored pk_mult rows (classy accessors,
    scripts/classpt_assembly.py; a0 = a2 = 0 there)."""
    out = ca.assemble_from_pm(inp["pk_mult"], inp["h"], inp["f"], inp["k_h"], cb, inp["Pd2d2_0"])
    return np.stack([out["pk_gg_l0"], out["pk_gg_l2"], out["pk_gg_l4"]])


def classpt_batch(inp, mask, *, ndens=NDENS_PROBE):
    def F(thetas):
        rows = []
        for th in np.asarray(thetas, dtype=float):
            cb = to_clax(dict(zip(THETA_NAMES, th)), f=inp["f"], ndens=ndens, Pd2d2_0=inp["Pd2d2_0"],
                         hratio=inp["hratio"], Dratio=inp["Dratio"])
            rows.append(classpt_multipoles(inp, {k: float(v) for k, v in cb.items()})[:, mask])
        return np.stack(rows)
    return F


# --------------------------------------------------------------------------- coefficient spectra

_FUNCTIONAL = {0: ((0, 1.0),), 1: ((-1, -0.5), (1, 0.5)), 2: ((-1, 0.5), (0, -1.0), (1, 0.5))}


def stencil(mono: tuple[str, ...]) -> list[tuple[dict, float]]:
    """[(theta, weight)] whose weighted sum of F(theta) is the coefficient of `mono`, exact for
    F of degree <= 2 in each parameter (all other parameters at zero)."""
    powers: dict[str, int] = {}
    for v in mono:
        if v != "1":
            powers[v] = powers.get(v, 0) + 1
    names = sorted(powers)
    out = []
    for combo in itertools.product(*(_FUNCTIONAL[powers[n]] for n in names)):
        th, w = zero_theta(), 1.0
        for n, (x, wx) in zip(names, combo):
            th[n] = float(x)
            w *= wx
        out.append((th, w))
    return out


def coefficients(F, monomials) -> dict[str, np.ndarray]:
    """{label: (M, Nk)} from ONE batched call of F over the union of all stencil points."""
    points: dict[tuple, int] = {}
    plan = {}
    for mono in monomials:
        terms = []
        for th, w in stencil(mono):
            key = tuple(th[n] for n in THETA_NAMES)
            terms.append((points.setdefault(key, len(points)), w))
        plan[label(mono)] = terms
    vals = np.asarray(F(np.array(list(points), dtype=float)))
    return {lab: sum(w * vals[j] for j, w in terms) for lab, terms in plan.items()}


def coefficient_triplet(inp, monomials, *, ir=True, ndens=NDENS_PROBE, nfft=256):
    """{"ps", "clax", "classpt"} coefficient spectra of `monomials` on the window, and the window kw. nfft =
    ps_1loop's FFTLog sampling (default 256, ps's own default and what the coefficient gate and the 13-line
    closure run; the counterterm / fingers-of-god / a0 closure passes 512, USER DECISION Oct 14, 2026)."""
    mask = cu.window(inp["k_h"])
    kw = inp["k_h"][mask]
    e = clax_ept(inp, ir=ir)
    res = {"ps": coefficients(ps_batch(ps_model(inp["rbao"], do_irres=ir, nfft=nfft), inp, kw, ndens=ndens),
                              monomials),
           "clax": coefficients(clax_batch(e, inp, mask, ndens=ndens), monomials),
           "classpt": {}}
    tb = [m for m in monomials if not set(m) & NO_TIEBREAK]
    if ir and tb:   # the twin is rebuilt from IR-ON pk_mult rows
        res["classpt"] = coefficients(classpt_batch(inp, mask, ndens=ndens), tb)
    return res, kw


def spectra_at(inp, theta: dict, *, ndens):
    """ps_1loop and clax-pt multipoles at one parameter point, on the window."""
    mask = cu.window(inp["k_h"])
    kw = inp["k_h"][mask]
    th = np.array([[theta[n] for n in THETA_NAMES]], dtype=float)
    ps = ps_batch(ps_model(inp["rbao"]), inp, kw, ndens=ndens)(th)[0]
    cl = clax_batch(clax_ept(inp), inp, mask, ndens=ndens)(th)[0]
    return ps, cl, kw


# --------------------------------------------------------------------------- gate

def coef_error(got, ref, kw, *, scale) -> dict:
    """Campaign metric max|got-ref|/max|ref| on the window (spec §5.1). When ref is a
    structural zero (max|ref| <= 1e-12 scale) the denominator is `scale`, the monomial's
    largest clax-pt multipole. Non-finite -> err = nan (a failure, Review Focus 5)."""
    got, ref = np.asarray(got, dtype=float), np.asarray(ref, dtype=float)
    d = np.abs(got - ref)
    if not np.all(np.isfinite(d)):
        i = int(np.argmax(~np.isfinite(d)))
        return {"err": float("nan"), "k": float(kw[i]), "metric": "non-finite"}
    i = int(np.argmax(d))
    ref_max = float(np.max(np.abs(ref)))
    zero = ref_max <= STRUCTURAL_ZERO * scale
    denom = max(scale if zero else ref_max, 1e-300)
    return {"err": float(d[i] / denom), "k": float(kw[i]), "metric": "monomial-scale" if zero else "self"}


def attribute(res, lab, i, kw, scale, thr) -> str:
    """Tie-breaker verdict (spec §6) for coefficient `lab`, channel index i."""
    ct = res.get("classpt", {}).get(lab)
    if ct is None:
        return "tie-breaker n/a"
    e_ps = coef_error(res["ps"][lab][i], ct[i], kw, scale=scale)["err"]
    e_cl = coef_error(res["clax"][lab][i], ct[i], kw, scale=scale)["err"]
    if float(np.max(np.abs(ct[i]))) <= STRUCTURAL_ZERO * scale and e_cl <= thr:
        verdict = "CLASS-PT convention (projection omitted in CLASS-PT and clax-pt)"
    elif not np.isfinite(e_ps) or e_ps > e_cl:
        verdict = "suspect ps_1loop"
    else:
        verdict = "suspect clax-pt"
    return f"vs CLASS-PT: ps {100 * e_ps:.3f}%, clax {100 * e_cl:.3f}% -> {verdict}"


def gate(res, kw, *, channels=CHANNELS, thresholds=COEF_THRESHOLDS, exclude=()):
    errs, lines = {}, []
    for lab, ref in res["clax"].items():
        scale = float(np.max(np.abs(ref)))
        for i, ch in enumerate(channels):
            e = coef_error(res["ps"][lab][i], ref[i], kw, scale=scale)
            name = f"{lab}:{ch}"
            errs[name] = e
            thr = thresholds[ch]
            if name not in exclude and (not np.isfinite(e["err"]) or e["err"] > thr):
                lines.append(f"ERROR {name} {e['metric']} {100 * e['err']:.3f}% > {100 * thr:.1f}% "
                             f"at k={e['k']:.3f} -- {attribute(res, lab, i, kw, scale, thr)}")
    return errs, lines


def worst(errs: dict) -> str:
    name, rec = max(errs.items(), key=lambda kv: np.inf if not np.isfinite(kv[1]["err"]) else kv[1]["err"])
    return f"worst {name} {100 * rec['err']:.3f}% at k={rec['k']:.3f}"


# --------------------------------------------------------------------------- log

def ps1loop_commit() -> str:
    try:
        import ps_1loop_jax
        d = Path(ps_1loop_jax.__file__).resolve().parent
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=d,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:   # installed without a git checkout -- still log
        return "unknown"


def log_record(*, layer: str, case: str, z: float, preset: str, errors: dict, extra: dict | None = None) -> None:
    """One JSON line in test_logs/ps1loop/errors.jsonl: ts, git_sha, ps1loop_sha, layer
    ('coef' | 'closure' | 'irno' | 'real' | 'grad' | 'pfs' | 'ladder'), case, z, preset, errors, extra."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rec = {"ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
           "git_sha": cu._git_sha(), "ps1loop_sha": ps1loop_commit(), "layer": layer, "case": case,
           "z": float(z), "preset": preset, "errors": errors, "extra": extra or {}}
    with ERROR_LOG.open("a") as fh:
        fh.write(json.dumps(cu._sanitize(rec), default=float, allow_nan=False) + "\n")


# --------------------------------------------------------------------------- model difference (spec §4.2)

def model_difference_factory(model, inp, kw, *, ndens=NDENS_PROBE):
    """Closed-form ps_1loop-model minus clax-pt-model for the k^2 counterterms, the k^4
    fingers-of-god term and a0, both on ps_1loop's own P_IR(k_t, mu_t). P_IR is built once
    (theta-independent); the returned F is polynomial in theta, so `coefficients` applies.

    ps_1loop model (true coordinates, Jacobian V; ps_1loop.py:497-547, 577-591, 320-335):
      V [-2 k_t^2 (c0 + c2 f mu_t^2 + c4 f^2 mu_t^4) P_IR - cfog f^4 k_t^4 mu_t^4 (b1 + f mu_t^2)^2 P_IR
         + a0 (k_t/k_nl)^2 / ndens], projected on L_l(mu) over the observed mu.
    clax-pt model (ept.py:1207-1209 and pk_gg_l0/l2/l4):
      2 cs_l Pctr_l + f^2 b4 k^2 poly_l (35/8) Pctr_4 [+ a0 (k/0.45)^2 at l=0],
      Pctr_l = -proj_l(V k_t^2 P_IR f^{l/2} mu_t^l)."""
    from ps_1loop_jax import ir_resum
    h, f, hr, Dr = inp["h"], inp["f"], inp["hratio"], inp["Dratio"]
    pk_data = {"k": jnp.asarray(inp["k_h"]), "pk": jnp.asarray(inp["pk"])}
    # same call as PowerSpectrum1Loop.get_pkmu (ps_1loop.py:284)
    pk_nw_data = ir_resum.get_pk_nw_data(pk_data, h, khmin=7e-5, khmax=7.0,
                                         kmin_interp=model._kmin, kmax_interp=model._kmax)
    a_perp, a_para = ap_alphas(hr, Dr)
    V = ap_volume(hr, Dr)
    fac = np.sqrt(1.0 + GL_MU**2 * ((a_perp / a_para) ** 2 - 1.0))
    mu_t = GL_MU * (a_perp / a_para) / fac
    kw = np.asarray(kw, dtype=float)
    k_t = kw[:, None] * fac[None, :] / a_perp                                  # (Nk, 40)
    p_ir = np.stack([np.asarray(model._get_pk_irres_rsd(jnp.asarray(k_t[:, j]), jnp.asarray([mu_t[j]]),
                                                        pk_data, pk_nw_data, f))[:, 0]
                     for j in range(GL_MU.size)], axis=1)
    pctr = {l: -project(V * k_t**2 * p_ir * f ** (l // 2) * mu_t**l, l) for l in ELLS}

    def one(theta):
        c0, c2, c4, cfog, b1, a0 = (theta[n] for n in ("c0", "c2", "c4", "cfog", "b1", "a0"))
        ps_mu = V * (-2.0 * k_t**2 * (c0 + c2 * f * mu_t**2 + c4 * f**2 * mu_t**4) * p_ir
                     - cfog * f**4 * k_t**4 * mu_t**4 * (b1 + f * mu_t**2) ** 2 * p_ir
                     + a0 * (k_t / K_NL) ** 2 / ndens)
        cb = to_clax(theta, f=f, ndens=ndens, Pd2d2_0=0.0, hratio=hr, Dratio=Dr)
        poly, cs = b4_poly(f, b1), {0: cb["cs0"], 2: cb["cs2"], 4: cb["cs4"]}
        rows = []
        for l in ELLS:
            clax_l = 2.0 * cs[l] * pctr[l] + f**2 * cb["b4"] * kw**2 * poly[l] * (35.0 / 8.0) * pctr[4]
            if l == 0:
                clax_l = clax_l + cb["a0"] * (kw / 0.45) ** 2
            rows.append(project(ps_mu, l) - clax_l)
        return np.stack(rows)

    return lambda thetas: np.stack([one(dict(zip(THETA_NAMES, th))) for th in np.asarray(thetas, dtype=float)])


# --------------------------------------------------------------------------- omitted projections (spec §4.2 item 4)
# The 13 CLOSURE_LINES are projections whose integrand clax-pt's 40-node mu-loop builds but never adds to
# P_l (mirroring CLASS-PT), while ps_1loop keeps them. Closed form (controller amendments and user decisions
# of Oct 8, 2026): rebuild each mu-loop integrand from the EXACT arguments clax's
# `_gl_multipoles` receives (spy below), with clax's own `_channels_at` and GL table, and project it on the
# omitted l with clax's AP-folded weights W_l = (2l+1)/2 w L_l(mu) hratio/Dratio^2.
# Provenance: ported from the validated scratch implementation
# /lustre/work/n2minh/std/clax/ps1loop/scratch-c1/test_c1_ifg2_dchi2.py (`_spy`, `run_ept`, `ifg2_proj`),
# which reproduced clax's kept F_G2 projections bit-exactly at 35 (case, z) points (jobs 22659/22660/22662);
# extended here to the isotropic loops and to the b1b2 / b1bG2 rows (CLASS-PT's unused pm[40] / pm[41]).

# rebuilt projection -> (EPTComponents field clax keeps, sign, family). The reproduction test asserts
# rebuilt == sign * field; `family` groups the rows of one integrand (the metric's scale).
KEPT_PROJECTIONS = {
    "IFG2:0": ("Pk_IFG2_0b1", 1.0, "IFG2"),
    "fmu2IFG2:0": ("Pk_IFG2_0", 1.0, "IFG2"),
    "fmu2IFG2:2": ("Pk_IFG2_2", 1.0, "IFG2"),
    "Pk_Id2d2:0": ("Pk_Id2d2", 1.0, "Pk_Id2d2"),
    "Pk_Id2G2:0": ("Pk_Id2G2", 1.0, "Pk_Id2G2"),
    "Pk_IG2G2:0": ("Pk_IG2G2", 1.0, "Pk_IG2G2"),
    "b1b2:0": ("Pk_0_b1b2", 1.0, "b1b2"),
    "b1b2:2": ("Pk_2_b1b2", 1.0, "b1b2"),
    "b1b2:4": ("Pk_4_b1b2", 1.0, "b1b2"),        # CLASS-PT pm[40]: computed, unused by classy
    "b1bG2:0": ("Pk_0_b1bG2", 1.0, "b1bG2"),
    "b1bG2:2": ("Pk_2_b1bG2", 1.0, "b1bG2"),
    # CLASS-PT pm[41]: computed, unused by classy. ept.py:1938 flips this one row back to CLASS-PT's
    # un-negated sign, so the projection in the galaxy convention -- the one whose l=0,2 moments are the
    # Pk_0/2_b1bG2 that pk_gg_l0/l2 add as "+ b1 bG2 Pk_l_b1bG2" -- is MINUS the stored field.
    "b1bG2:4": ("Pk_4_b1bG2", -1.0, "b1bG2"),
}


def capture_gl_inputs(inp, **overrides):
    """(EPTComponents, captured): clax_ept(inp, **overrides) with a spy on clax.ept._gl_multipoles.
    `captured` = the exact arguments the mu-loop received: chan (47 channels on the k grid), k, f,
    s2 = Sigma^2_BAO, ds2 = delta Sigma^2_BAO, hratio, Dratio. The original function is restored in
    `finally`. compute_ept is not jitted, so the spy sees concrete arrays; exactly one call is required."""
    import clax.ept as E
    orig = E._gl_multipoles
    calls = []

    def spy(chan, k, f, sigma2_bao, delta_sigma2_bao, hratio=1.0, Dratio=1.0):
        calls.append({"chan": chan, "k": k, "f": f, "s2": sigma2_bao, "ds2": delta_sigma2_bao,
                      "hratio": hratio, "Dratio": Dratio})
        return orig(chan, k, f, sigma2_bao, delta_sigma2_bao, hratio=hratio, Dratio=Dratio)

    E._gl_multipoles = spy
    try:
        e = clax_ept(inp, **overrides)
    finally:
        E._gl_multipoles = orig
    if len(calls) != 1:
        raise RuntimeError(f"ERROR spy on clax.ept._gl_multipoles called {len(calls)} times, expected 1")
    return e, calls[0]


def mu_loop_geometry(cap) -> dict:
    """clax's mu-loop geometry for one captured `_gl_multipoles` call, shared by every rebuild below
    (`mu_loop_projections`, `clax_scheme_projections`, `ps_scheme_predicted`), so the reproduction test
    covers the gated path. Returns k (knots, (Nk,)), f, mu2t = mu_t^2 (40, 1), kk = k_t (40, Nk),
    W = {l: AP-folded GL weights (2l+1)/2 w L_l(mu) hratio/Dratio^2} and Exp = e^{-k_t^2 Sigma_tot^2(mu_t)}
    (40, Nk). Line-by-line mirror of clax/ept.py _gl_multipoles 1133-1150 and 1167-1169 (CLASS-PT
    nonlinear_pt.c:4382-4394, 4470-4481)."""
    import clax.ept as E
    k, f, s2, ds2 = cap["k"], cap["f"], cap["s2"], cap["ds2"]
    mu = jnp.asarray(E._GAUSS_NODES)
    w = jnp.asarray(E._GAUSS_WEIGHTS)
    if mu.shape != (40,):
        raise RuntimeError(f"ERROR clax GL table has {mu.shape[0]} nodes, expected 40")
    hratio = jnp.asarray(cap["hratio"], dtype=k.dtype)
    Dratio = jnp.asarray(cap["Dratio"], dtype=k.dtype)
    inv_D2 = 1.0 / Dratio ** 2
    ap_fac = jnp.sqrt(inv_D2 + (hratio ** 2 - inv_D2) * mu ** 2)[:, None]
    mu_k = mu[:, None] * hratio / ap_fac
    kk = k[None, :] * ap_fac
    V = hratio / Dratio ** 2
    mu2t = mu_k ** 2
    L2 = 0.5 * (3.0 * mu ** 2 - 1.0)
    L4 = (35.0 * mu ** 4 - 30.0 * mu ** 2 + 3.0) / 8.0
    W = {0: 0.5 * w * V, 2: 2.5 * w * L2 * V, 4: 4.5 * w * L4 * V}
    Sig = (s2 * (1.0 + f * mu2t * (2.0 + f))
           + ds2 * f ** 2 * mu2t * (mu2t - 1.0))
    Exp = jnp.exp(-Sig * kk ** 2)
    return {"k": k, "f": f, "mu2t": mu2t, "kk": kk, "W": W, "Exp": Exp}


def _ifg2_in(at, Exp):
    """clax's F_G2 integrand IFG2_in = p_lo P_IFG2, p_lo = (P_nw + P_w e^{-x})/P_bin (ept.py:1175, 1214-1216)."""
    p_lin = at["pk_nw"] + at["pk_w"] * Exp
    Pbin = at["pk_disc"]
    p_lo = p_lin / jnp.where(Pbin > 1e-100, Pbin, 1.0)
    return p_lo * at["Pk_IFG2"]


def _window_mask(inp, kw, caps) -> np.ndarray:
    """The comparison-window mask, after checking kw and every captured mu-loop k grid against inp."""
    mask = cu.window(inp["k_h"])
    if not np.array_equal(np.asarray(kw, dtype=float), inp["k_h"][mask]):
        raise ValueError("ERROR kw is not the comparison window of these inputs")
    for cap in caps:
        if not np.array_equal(np.asarray(cap["k"]), inp["k_h"]):
            raise RuntimeError("ERROR clax's mu-loop k grid differs from the shared-input k grid")
    return mask


def mu_loop_projections(cap) -> dict[str, np.ndarray]:
    """{"<integrand>:<l>": (Nk,)} -- the l = 0, 2, 4 projections, on clax's whole k grid, of the mu-loop
    integrands behind the closure lines: "IFG2" = IFG2_in, "fmu2IFG2" = f mu_t^2 IFG2_in, the isotropic
    loops c(Pk_Id2d2), c(Pk_Id2G2), c(Pk_IG2G2), and "b1b2" / "b1bG2" = c(Pk_0_b) + L2(mu_t) c(Pk_2_b).
    Mirror of clax/ept.py _gl_multipoles: geometry `mu_loop_geometry`, channels at k_t 1155, IFG2_in
    1214-1216, isotropic loops 1222-1223, b1b2/b1bG2 1224-1232 (CLASS-PT nonlinear_pt.c:5273-5351)."""
    import clax.ept as E
    g = mu_loop_geometry(cap)
    f, mu2t = g["f"], g["mu2t"]
    at = E._channels_at(g["k"], cap["chan"], g["kk"])
    IFG2_in = _ifg2_in(at, g["Exp"])
    L2t = 0.5 * (3.0 * mu2t - 1.0)
    integrands = {"IFG2": IFG2_in, "fmu2IFG2": IFG2_in * f * mu2t,
                  **{n: at[n] for n in ("Pk_Id2d2", "Pk_Id2G2", "Pk_IG2G2")},
                  **{b: at[f"Pk_0_{b}"] + L2t * at[f"Pk_2_{b}"] for b in ("b1b2", "b1bG2")}}
    return {f"{name}:{l}": np.asarray(jnp.einsum("m,mk->k", g["W"][l], val))
            for name, val in integrands.items() for l in ELLS}


def omitted_projections(inp, kw, *, captured=None) -> dict[str, np.ndarray]:
    """`mu_loop_projections` of clax's own mu-loop on the shared inputs, on the comparison window kw
    (= window_k(inp)). `captured` = capture_gl_inputs(inp), to reuse one compute_ept."""
    _, cap = capture_gl_inputs(inp) if captured is None else captured
    mask = _window_mask(inp, kw, (cap,))
    return {n: v[mask] for n, v in mu_loop_projections(cap).items()}


def omitted_predicted(proj) -> dict[str, np.ndarray]:
    """{monomial label: (3, Nw)} predicted ps_1loop - clax-pt on the 13 CLOSURE_LINES; rows clax-pt keeps
    are zero. <.>_l = clax's W_l-projection (`mu_loop_projections`).
      F_G2: the galaxy P(k, mu) carries (2 bG2 + 0.8 bGamma3)(b1 + f mu_t^2) IFG2_in; clax keeps
        b1<IFG2_in>_0, <f mu_t^2 IFG2_in>_0,2, so bG2:l4 = 2<f mu_t^2 IFG2_in>_4,
        bGamma3:l4 = 0.8<f mu_t^2 IFG2_in>_4, b1*bG2:l2 = 2<IFG2_in>_2, b1*bGamma3:l2 = 0.8<IFG2_in>_2,
        b1*bGamma3:l4 = 0.8<IFG2_in>_4, b1*bG2:l4 = 2<IFG2_in>_4 + <c(Pk_0_b1bG2) + L2(mu_t) c(Pk_2_b1bG2)>_4
        (the second term = CLASS-PT's computed-but-unused pm[41], -EPTComponents.Pk_4_b1bG2).
      Isotropic loops (clax keeps l = 0 only): b2*b2:l = 0.25<c(Pk_Id2d2)>_l, bG2*bG2:l = <c(Pk_IG2G2)>_l,
        b2*bG2:l = <c(Pk_Id2G2)>_l, l = 2, 4.
      b1*b2:l4 = <c(Pk_0_b1b2) + L2(mu_t) c(Pk_2_b1b2)>_4 (CLASS-PT's unused pm[40], EPTComponents.Pk_4_b1b2)."""
    z = np.zeros_like(proj["IFG2:0"])
    rows = {
        "bG2": (z, z, 2.0 * proj["fmu2IFG2:4"]),
        "bGamma3": (z, z, 0.8 * proj["fmu2IFG2:4"]),
        "b1*bG2": (z, 2.0 * proj["IFG2:2"], 2.0 * proj["IFG2:4"] + proj["b1bG2:4"]),
        "b1*bGamma3": (z, 0.8 * proj["IFG2:2"], 0.8 * proj["IFG2:4"]),
        "b2*b2": (z, 0.25 * proj["Pk_Id2d2:2"], 0.25 * proj["Pk_Id2d2:4"]),
        "bG2*bG2": (z, proj["Pk_IG2G2:2"], proj["Pk_IG2G2:4"]),
        "b2*bG2": (z, proj["Pk_Id2G2:2"], proj["Pk_Id2G2:4"]),
        "b1*b2": (z, z, proj["b1b2:4"]),
    }
    return {lab: np.stack(r) for lab, r in rows.items()}


# --------------------------------------------------------------------------- ps IR-resummation scheme (spec §4.2 item 5)
# `omitted_predicted` is clax's OWN mu-loop integrand projected on the omitted l. ps_1loop also resums every
# loop differently: loop[P_nw] + e^{-x(mu)} (loop[P] - loop[P_nw]) (ps_1loop.py:474-494,
# get_pkmu_irres_LO_NLO), while clax-pt / CLASS-PT feed the isotropically damped P_nw + e^{-k^2 Sigma^2} P_w
# into the bias loops and rescale only F_G2 by p_lo(mu) = (P_nw + e^{-x(mu)} P_w)/P_res. For F_G2 the
# difference is f mu^2 [-P_nw J[D P_w] + e^{-x(mu)} P_nw J[P_w]] + O(P_w^2) (D = e^{-q^2 Sigma^2} on the loop
# momentum): the two pieces nearly cancel in the monopole, but the mu-dependent e^{-x(mu)} survives at
# l = 2, 4 -- exactly the omitted projections. USER DECISION (Oct 10, 2026): the closure predicts
# ps - clax = <X_ps>_l - clax_l, with each monomial's integrand X_ps rebuilt in ps's scheme from clax's own
# IR-OFF channels, on clax's P_nw, Sigma^2, dSigma^2 and AP geometry (ps 0.5.0's P_nw equals CLASS-PT's to
# 1.4e-5 of P) -- a documented model difference with a closed form.
# Provenance: ported from the scratch diagnostic
# /lustre/work/n2minh/std/clax/ps1loop/scratch-task5/test_t5_diag.py (`geometry`, `mono_integrands`, `one`),
# validated over all 42 files in job 23255 (all 13 lines close; worst 0.56% on a 1% gate, 0.71% on a 2% gate).

# The bias channels the closure monomials' integrands read (clax sign convention, ept.py:1907-1926).
PS_SCHEME_CHANNELS = ("Pk_IFG2", "Pk_Id2d2", "Pk_Id2G2", "Pk_IG2G2", "Pk_0_b1b2", "Pk_2_b1b2",
                      "Pk_0_b1bG2", "Pk_2_b1bG2", "Pk_0_bG2", "Pk_2_bG2", "Pk_4_bG2")


def closure_integrands(ch, f, mu2t, ifg2) -> dict:
    """{monomial label: (40, Nk)}: each CLOSURE_MONOMIALS monomial's coefficient in the galaxy P(k_t, mu_t)
    inside clax's mu-loop, clax sign convention. ch = the PS_SCHEME_CHANNELS at k_t; ifg2 = the F_G2 13-type
    term (clax: IFG2_in). Accessor algebra of ept.py pk_gg_l0/l2/l4 on the mu-loop integrands of
    _gl_multipoles (1214-1237): bG2 -> 22-type c0 + L2t c2 + L4t c4 plus 2 f mu_t^2 F_G2; bGamma3 -> 0.8 f mu_t^2
    F_G2; b1*bG2 -> c0 + L2t c2 plus 2 F_G2; b1*bGamma3 -> 0.8 F_G2; b2*b2 -> 0.25 Id2d2; bG2*bG2 -> IG2G2;
    b2*bG2 -> Id2G2; b1*b2 -> c0 + L2t c2."""
    L2t = 0.5 * (3.0 * mu2t - 1.0)
    L4t = (35.0 * mu2t ** 2 - 30.0 * mu2t + 3.0) / 8.0
    return {
        "bG2": ch["Pk_0_bG2"] + L2t * ch["Pk_2_bG2"] + L4t * ch["Pk_4_bG2"] + 2.0 * f * mu2t * ifg2,
        "bGamma3": 0.8 * f * mu2t * ifg2,
        "b1*bG2": ch["Pk_0_b1bG2"] + L2t * ch["Pk_2_b1bG2"] + 2.0 * ifg2,
        "b1*bGamma3": 0.8 * ifg2,
        "b2*b2": 0.25 * ch["Pk_Id2d2"],
        "bG2*bG2": ch["Pk_IG2G2"],
        "b2*bG2": ch["Pk_Id2G2"],
        "b1*b2": ch["Pk_0_b1b2"] + L2t * ch["Pk_2_b1b2"],
    }


def restrict_to_closure_lines(pred: dict) -> dict:
    """Zero every row of {label: (3, Nw)} that is not one of the 13 CLOSURE_LINES (Task 4 gates those)."""
    out = {}
    for lab, arr in pred.items():
        keep = np.array([f"{lab}:{ch}" in CLOSURE_LINES for ch in CHANNELS])
        out[lab] = np.where(keep[:, None], np.asarray(arr, dtype=float), 0.0)
    return out


def _project_monomials(integ, W, mask) -> dict:
    """{label: (3, Nw)}: l = 0, 2, 4 projections of each CLOSURE_MONOMIALS integrand, on the window."""
    return {label(m): np.stack([np.asarray(jnp.einsum("m,mk->k", W[l], integ[label(m)]))[mask] for l in ELLS])
            for m in CLOSURE_MONOMIALS}


def clax_scheme_projections(inp, kw, *, captured=None) -> dict:
    """{monomial label: (3, Nw)} = <X_clax(mono)>_l: each closure monomial's FULL integrand in clax's own
    IR-on mu-loop (`closure_integrands` on the captured channels, F_G2 = IFG2_in) projected on l = 0, 2, 4.
    Minus `omitted_predicted` it must equal clax's reported coefficient on all three multipoles -- the
    kept-coefficient check that ties the rebuild to clax's accessors (test_omitted_projections_close)."""
    import clax.ept as E
    _, cap = capture_gl_inputs(inp) if captured is None else captured
    mask = _window_mask(inp, kw, (cap,))
    g = mu_loop_geometry(cap)
    at = E._channels_at(g["k"], cap["chan"], g["kk"])
    return _project_monomials(closure_integrands(at, g["f"], g["mu2t"], _ifg2_in(at, g["Exp"])), g["W"], mask)


def ps_scheme_predicted(inp, kw, clax_coef, *, captured=None) -> dict:
    """{monomial label: (3, Nw)} predicted ps_1loop - clax-pt on the 13 CLOSURE_LINES, ps's IR scheme included:
        <X_ps(mono)>_l - clax_coef[mono]_l,   X_ps = C[P_nw](k_t) + e^{-x(k_t, mu_t)} (C[P] - C[P_nw])(k_t),
    with C[.] clax's IR-OFF bias channels computed on the shared P and on clax's own P_nw (both splined at k_t
    by clax's `_channels_at`), x = k_t^2 Sigma_tot^2(mu_t), and Sigma^2, dSigma^2, P_nw and the AP geometry
    all taken from the same IR-on mu-loop (`captured` = capture_gl_inputs(inp)) via `mu_loop_geometry`. IR
    off, the F_G2 channel carries its external spectrum (p_lo = 1), so C_FG2[X] = X I_FG2[X], as in ps's
    loop[X]. clax_coef = coefficient_triplet's "clax" entry on the same kw. Rows outside CLOSURE_LINES are
    zero. Equals omitted_predicted + <X_ps - X_clax>_l (the scheme term) to round-off."""
    import clax.ept as E
    _, cap_on = capture_gl_inputs(inp) if captured is None else captured
    _, cap_off = capture_gl_inputs(inp, ir=False)
    _, cap_nw = capture_gl_inputs(inp, ir=False, pk=jnp.asarray(cap_on["chan"]["pk_nw"]))
    mask = _window_mask(inp, kw, (cap_on, cap_off, cap_nw))
    g = mu_loop_geometry(cap_on)
    C = E._channels_at(g["k"], {n: cap_off["chan"][n] for n in PS_SCHEME_CHANNELS}, g["kk"])
    Cn = E._channels_at(g["k"], {n: cap_nw["chan"][n] for n in PS_SCHEME_CHANNELS}, g["kk"])
    X = {n: Cn[n] + g["Exp"] * (C[n] - Cn[n]) for n in PS_SCHEME_CHANNELS}
    proj = _project_monomials(closure_integrands(X, g["f"], g["mu2t"], X["Pk_IFG2"]), g["W"], mask)
    return restrict_to_closure_lines({lab: proj[lab] - np.asarray(clax_coef[lab], dtype=float) for lab in proj})


# --------------------------------------------------------------------------- ps IR scheme beyond the 13 lines (spec §4.2 items 5-6)
# What these two terms buy (USER DECISION Oct 14, 2026; Task 11): with them in D, the derivative closure of
# tests/test_ps1loop_gradients.py closes on every line -- f, alpha_perp, alpha_para, the k-stretch and the
# amplitude of P_lin, l = 0, 2, 4, three cosmologies -- once ps_1loop runs at nfft >= 512 (scratch jobs
# 24970 / 24971). Without them the stretch derivative kept a BAO-scale, non-smooth-in-k remainder that was
# 11-14x the primal one (Task 7 / Task 9), i.e. a difference in how the two codes move the BAO feature.
#
# (i) Kept bias rows (item 5 extended). ps_1loop resums its WHOLE one-loop sum as
# loop[P_nw] + e^{-x(mu)} (loop[P] - loop[P_nw]) (ps_1loop.py:474-494 get_pkmu_irres_LO_NLO; the two channels
# at 464-471). clax-pt, like CLASS-PT, feeds every bias loop the isotropically damped P_nw + e^{-k^2 Sigma^2} P_w
# (ept.py:2085 pk_resummed, the FFTLog input of _compute_bias_spectra at 2164) and rescales only the F_G2
# external leg by p_lo(mu) (ept.py:1220-1221). The scheme difference <X_ps>_l - clax_l of `ps_scheme_predicted`
# is therefore not confined to the 13 omitted projections: it sits on every row clax keeps that is built
# from the same channels, the 14 KEPT_SCHEME_ROWS. Construction: `ps_scheme_predicted`'s, unrestricted,
# extended by the linear b2 row, whose integrand c(Pk_0_b2) + L2(mu_t) c(Pk_2_b2) + L4(mu_t) c(Pk_4_b2) is
# ept.py:1239-1242 (Pb2_in), kept on all three l (pk_gg_l0/l2/l4, ept.py:2433 / 2494 / 2545).
#   Per-row decision on the six F_G2 kept rows (bG2:l0, bG2:l2, b1*bG2:l0 carry a 22-type part plus F_G2;
#   bGamma3:l0, bGamma3:l2, b1*bGamma3:l0 are F_G2 only): KEPT, all six. The F_G2 integrand is one integrand,
#   and its l = 4 / b1*bG2:l2 projections (C1_LINES) already carry this scheme term in the validated Task 5
#   closure; dropping it on l = 0, 2 would make D an l-by-l selection rather than the model difference.
#   Data at the gate setting, ps nfft 512, lcdm_fiducial z = 0.38 (max|ps - clax| -> max|ps - clax - predicted|,
#   % of the monomial's largest clax multipole; scratch job 24970): bG2:l0 0.035 -> 0.025, bG2:l2 0.091 -> 0.048,
#   b1*bG2:l0 0.047 -> 0.049, bGamma3:l0 0.026 -> 0.029, bGamma3:l2 0.048 -> 0.056, b1*bGamma3:l0 0.055 -> 0.062
#   (massive_nu_030 / w0wa_m07_m10 the same pattern, job 24971): the pure-F_G2 rows keep a 0.03-0.06%
#   remainder that is not the scheme term, three orders of magnitude below the 1% gate. On the total at the PFS
#   bias point (bG2 = -0.015, bGamma3 = +0.029) the six rows move no derivative-closure line by more than the
#   printed 0.001% (all 45 lines, nfft 256 / 512 / 1024, jobs 24970 / 24971).
#
# (ii) Matter loops (item 6). ps resums the vv, vd, dd loops EXACTLY in the wiggle spectrum P_w: with
# L[X] = L22[X, X] + X I13[X] + UV, it adds e^{-x} (L[P] - L[P_nw]). CLASS-PT and clax-pt linearise in P_w: the
# mu-loop adds e^{-x} (2 L22[P_nw, P_w] + P_w I13[P_nw] + P_nw I13[P_w]) through the "_w" channels and
# P13ratio (ept.py:1178, 1183-1193; channels from qf_split / p13_split, 1739-1768). The difference,
#     e^{-x(k_t, mu_t)} (L22[P_w, P_w] + P_w I13[P_w]),
# is second order in P_w and sits on the monomials 1 (vv), b1 (vd), b1^2 (dd) (pk_gg_l0, ept.py:2424-2426).
# It is rebuilt from ONE extra IR-off clax capture on P_w (FFTLog is linear in its input and uses the same
# bias, so its "_nw" channels are L22[P_w, P_w] and P_w I13[P_w]), after removing the P13 UV term
# sigma_v^2[P_w] k^2 P_w that clax's IR-off p13_split adds (UV coefficients ept.py:1690, 1699-1703; ps keeps one
# sigma~^2[P] for both channels, ps_1loop.py:464-468 and 411-447), and projected by clax's own
# `_gl_multipoles` (same AP geometry, Sigma_tot(mu), GL weights). Left out: the UV normalisation difference
# (sigma~^2[P] - sigma_v^2[P_res]) k^2 P_res, whose size is (sigma^2[P] - sigma^2[P_res]) / sigma^2[P] = -3.1e-6
# (job 24970). The rebuild is pinned by `matter_rebuild_checks` (P22 bilinearity, UV removal), asserted in
# tests/test_ps1loop_bridge.py.

B2_CHANNELS = ("Pk_0_b2", "Pk_2_b2", "Pk_4_b2")
SCHEME_CHANNELS = PS_SCHEME_CHANNELS + B2_CHANNELS
# The closure monomials plus the linear b2 row: every bias monomial whose clax rows are built from the
# isotropically damped channels or F_G2.
SCHEME_MONOMIALS = CLOSURE_MONOMIALS + (("b2",),)
# The 14 rows of SCHEME_MONOMIALS that clax keeps (all rows not in CLOSURE_LINES).
KEPT_SCHEME_ROWS = tuple(f"{label(m)}:{ch}" for m in SCHEME_MONOMIALS for ch in CHANNELS
                         if f"{label(m)}:{ch}" not in CLOSURE_LINES)
# Galaxy monomial -> clax matter loop (pk_gg_l0: Pk_l_vv1 + b1 Pk_l_vd1 + b1^2 Pk_l_dd1).
MATTER_LOOPS = {"1": "vv", "b1": "vd", "b1*b1": "dd"}
# P13 UV coefficient per unit sigma_v^2 of each mu-power P13 channel (ept.py:1690 UV_mu6, 1699-1703 UV_mu*;
# CLASS-PT nonlinear_pt.c:8219-8226), f-dependent.
P13_UV_PER_SIGMA2 = {"P13_mu0_dd": lambda f: -61.0 / 105.0,
                     "P13_mu2_dd": lambda f: -f * (105.0 * f - 6.0) / 105.0,
                     "P13_mu2_vd": lambda f: -f * (250.0 + 144.0 * f) / 105.0,
                     "P13_mu4_vv": lambda f: -f ** 2 * (63.0 + 48.0 * f) / 35.0,
                     "P13_mu4_vd": lambda f: -f ** 2 * (44.0 + 70.0 * f) / 35.0,
                     "P13_mu6": lambda f: -f ** 3 * (46.0 + 35.0 * f) / 35.0}


def scheme_integrands(ch, f, mu2t, ifg2) -> dict:
    """`closure_integrands` plus the linear b2 row, c(Pk_0_b2) + L2(mu_t) c(Pk_2_b2) + L4(mu_t) c(Pk_4_b2)
    (ept.py:1239-1242, Pb2_in): {label: (40, Nk)} for every SCHEME_MONOMIALS monomial."""
    out = dict(closure_integrands(ch, f, mu2t, ifg2))
    L2t = 0.5 * (3.0 * mu2t - 1.0)
    L4t = (35.0 * mu2t ** 2 - 30.0 * mu2t + 3.0) / 8.0
    out["b2"] = ch["Pk_0_b2"] + L2t * ch["Pk_2_b2"] + L4t * ch["Pk_4_b2"]
    return out


def scheme_captures(inp, *, captured=None) -> dict:
    """The four clax mu-loop captures (`capture_gl_inputs`) the extended closed form reads, on the shared
    inputs: "on" = IR on (geometry, Sigma^2, dSigma^2, P_nw, P_w, clax's own channels), "off" = IR off on P,
    "nw" = IR off on clax's P_nw, "ww" = IR off on clax's P_w (matter term). `captured` reuses an "on" capture."""
    _, cap_on = capture_gl_inputs(inp) if captured is None else captured
    _, cap_off = capture_gl_inputs(inp, ir=False)
    _, cap_nw = capture_gl_inputs(inp, ir=False, pk=jnp.asarray(cap_on["chan"]["pk_nw"]))
    _, cap_ww = capture_gl_inputs(inp, ir=False, pk=jnp.asarray(cap_on["chan"]["pk_w"]))
    return {"on": cap_on, "off": cap_off, "nw": cap_nw, "ww": cap_ww}


def _project_scheme(integ, g, mask) -> dict:
    """{label: (3, Nw)}: l = 0, 2, 4 projections of each SCHEME_MONOMIALS integrand with clax's AP-folded
    weights g["W"] (`mu_loop_geometry`), on the window."""
    return {label(m): np.stack([np.asarray(jnp.einsum("m,mk->k", g["W"][l], integ[label(m)]))[mask] for l in ELLS])
            for m in SCHEME_MONOMIALS}


def clax_scheme_coefficients(inp, kw, *, caps=None) -> dict:
    """{monomial label: (3, Nw)} clax-pt's own coefficient of every SCHEME_MONOMIALS monomial, rebuilt from its
    IR-on mu-loop: the full integrand (`scheme_integrands` on the captured channels, F_G2 = IFG2_in) projected
    on l = 0, 2, 4, minus `omitted_predicted` (the b2 row omits nothing). It equals the coefficient spectrum of
    clax's accessors pk_gg_l0/l2/l4 to round-off on all 27 rows (asserted in tests/test_ps1loop_bridge.py),
    so D subtracts what clax actually reports."""
    import clax.ept as E
    caps = scheme_captures(inp) if caps is None else caps
    cap_on = caps["on"]
    mask = _window_mask(inp, kw, tuple(caps.values()))
    g = mu_loop_geometry(cap_on)
    at = E._channels_at(g["k"], cap_on["chan"], g["kk"])
    full = _project_scheme(scheme_integrands(at, g["f"], g["mu2t"], _ifg2_in(at, g["Exp"])), g, mask)
    omitted = omitted_predicted({n: v[mask] for n, v in mu_loop_projections(cap_on).items()})
    omitted["b2"] = np.zeros_like(full["b2"])
    return {lab: full[lab] - omitted[lab] for lab in full}


def ps_scheme_all_rows(inp, kw, *, caps=None) -> dict:
    """{monomial label: (3, Nw)} predicted ps_1loop - clax-pt on EVERY row of SCHEME_MONOMIALS, ps's IR scheme
    included: <X_ps(mono)>_l - clax_coef_l, X_ps = C[P_nw] + e^{-x}(C[P] - C[P_nw]) as in `ps_scheme_predicted`
    (same captures, same geometry), clax_coef = `clax_scheme_coefficients`. Restricted to CLOSURE_LINES this is
    `ps_scheme_predicted` (asserted to round-off in tests/test_ps1loop_bridge.py); on KEPT_SCHEME_ROWS it is
    ps's scheme term on the rows clax keeps (section comment, item (i))."""
    import clax.ept as E
    caps = scheme_captures(inp) if caps is None else caps
    cap_on, cap_off, cap_nw = caps["on"], caps["off"], caps["nw"]
    mask = _window_mask(inp, kw, tuple(caps.values()))
    g = mu_loop_geometry(cap_on)
    C = E._channels_at(g["k"], {n: cap_off["chan"][n] for n in SCHEME_CHANNELS}, g["kk"])
    Cn = E._channels_at(g["k"], {n: cap_nw["chan"][n] for n in SCHEME_CHANNELS}, g["kk"])
    X = {n: Cn[n] + g["Exp"] * (C[n] - Cn[n]) for n in SCHEME_CHANNELS}
    ps_full = _project_scheme(scheme_integrands(X, g["f"], g["mu2t"], X["Pk_IFG2"]), g, mask)
    clax_coef = clax_scheme_coefficients(inp, kw, caps=caps)
    return {lab: ps_full[lab] - clax_coef[lab] for lab in ps_full}


def _matter_ww_channels(caps) -> dict:
    """{loop channel name: (Nk,)} L22[P_w, P_w] (P22 channels) and P_w I13[P_w] (P13 channels) on clax's k grid:
    the "_nw" channels of the IR-off capture on P_w, with the P13 UV term sigma_v^2[P_w] k^2 P_w e^{-(k/k_cut)^6}
    that p13_split adds (ept.py:1749-1757, sigma_v^2 1629, damping 1336) removed."""
    import clax.ept as E
    cap_on, cap_ww = caps["on"], caps["ww"]
    k, f = cap_on["k"], cap_on["f"]
    uv_damp = jnp.exp(-(k / EPTPrecisionParams().cutoff_h) ** 6)
    pw = jnp.asarray(cap_ww["chan"]["pk_disc"])            # IR off: the FFTLog input, = P_w on the grid
    sig2_w = jnp.trapezoid(pw * k, jnp.log(k)) / (6.0 * jnp.pi ** 2)
    out = {}
    for n in E._GL_CHANNELS_LOOP:
        L = jnp.asarray(cap_ww["chan"][n + "_nw"])
        if n in P13_UV_PER_SIGMA2:
            L = L - P13_UV_PER_SIGMA2[n](f) * sig2_w * k ** 2 * pw * uv_damp
        out[n] = L
    return out


def matter_scheme_term(inp, kw, *, caps=None) -> dict:
    """{"1" | "b1" | "b1*b1": (3, Nw)} = the l = 0, 2, 4 projections of e^{-x(k_t, mu_t)} (L22[P_w, P_w]
    + P_w I13[P_w]) on the vv / vd / dd loops: ps's exact-in-P_w matter-loop resummation minus clax's linearised
    one (section comment, item (ii)). Projected by clax's own `_gl_multipoles` fed only those channels as its
    wiggle ("_w") channels, every other channel zero and P_w = 0 so that P13ratio = 1."""
    import clax.ept as E
    caps = scheme_captures(inp) if caps is None else caps
    cap_on = caps["on"]
    mask = _window_mask(inp, kw, tuple(caps.values()))
    k = cap_on["k"]
    zeros = jnp.zeros_like(k)
    chan = dict(cap_on["chan"])
    for n, L in _matter_ww_channels(caps).items():
        chan[n + "_nw"], chan[n + "_w"] = zeros, L
    for n in E._GL_CHANNELS_BIAS:
        chan[n] = zeros
    chan["pk_w"] = zeros                                    # P13ratio = 1: only the e^{-x}-weighted "_w" terms
    gl = E._gl_multipoles(chan, k, cap_on["f"], cap_on["s2"], cap_on["ds2"],
                          hratio=cap_on["hratio"], Dratio=cap_on["Dratio"])
    return {lab: np.stack([np.asarray(gl[f"Pk_{l}_{c}1"])[mask] for l in ELLS]) for lab, c in MATTER_LOOPS.items()}


def matter_rebuild_checks(caps) -> dict:
    """Round-off identities that pin the matter-term rebuild (the scratch measured them at 6e-14 - 9e-14 and
    <= 8.6e-16, jobs 24970 / 24971):
      p22_bilinear  max over P22 channels of max|L22[P] - L22[P_nw] - 2 L22[P_nw, P_w] - L22[P_w, P_w]| /
                    max|L22[P]| -- the IR-off captures on P and P_nw, clax's IR-on wiggle channel (= 2 L22[P_nw, P_w],
                    qf_split) and the rebuilt L22[P_w, P_w] add up, so the rebuild is the second-order piece;
      p13_uv_removed max over P13 channels of max|P_nw (P_w I13[P_w]) - P_w (P_nw I13[P_w])| / max|P_w P_nw I13[P_w]|,
                    the second factor clax's IR-on P13 wiggle channel: zero only if the UV term was removed exactly."""
    import clax.ept as E
    cap_on, cap_off, cap_nw = caps["on"], caps["off"], caps["nw"]
    ww = _matter_ww_channels(caps)
    pnw, pw = np.asarray(cap_on["chan"]["pk_nw"]), np.asarray(cap_on["chan"]["pk_w"])
    on, off, nw = cap_on["chan"], cap_off["chan"], cap_nw["chan"]
    p22 = [n for n in E._GL_CHANNELS_LOOP if n.startswith("P22")]
    p13 = [n for n in E._GL_CHANNELS_LOOP if n.startswith("P13")]
    bil = max(float(np.max(np.abs(np.asarray(off[n + "_nw"]) - np.asarray(nw[n + "_nw"]) - np.asarray(on[n + "_w"])
                                  - np.asarray(ww[n]))) / np.max(np.abs(np.asarray(off[n + "_nw"])))) for n in p22)
    uv = max(float(np.max(np.abs(np.asarray(ww[n]) * pnw - np.asarray(on[n + "_w"]) * pw))
                   / np.max(np.abs(np.asarray(on[n + "_w"]) * pw))) for n in p13)
    return {"p22_bilinear": bil, "p13_uv_removed": uv}


# --------------------------------------------------------------------------- documented difference on the total (Task 7)

def monomial_value(lab: str, theta: dict) -> float:
    """Value at the parameter point theta of the monomial named `lab` ("1", "b1", "b1*bG2", ...)."""
    return float(np.prod([1.0 if v == "1" else theta[v] for v in lab.split("*")]))


# The four parts of D, in the order predicted_total_difference returns them.
D_PARTS = ("ctr", "closure", "kept", "matter")


def predicted_total_difference(inp, theta: dict, kw, *, ndens=NDENS_PROBE, model=None) -> dict:
    """The documented ps_1loop - clax-pt model difference (spec §4.2 items 1-6) on the TOTAL P_0, P_2, P_4 at
    the parameter point theta, on the window kw, as {"ctr", "closure", "kept", "matter"}: four (3, Nw) parts
    whose sum is D. Every input that the derivative tangents move -- P_lin, f and the AP ratios -- is read from
    `inp`, so D can be re-evaluated at a displaced point by passing a modified copy of the shared inputs.

      "ctr" (items 1-3): the k^2 counterterms, the k^4 fingers-of-god term and a0
        (`model_difference_factory` evaluated at theta itself; it is polynomial in theta, so this is
        the sum over MODEL_DIFF_MONOMIALS with their values). NOT zero with IR resummation on: ps_1loop
        multiplies the full mu_t-dependence by P_IR(k_t, mu_t), while clax-pt keeps only Legendre moments
        of the AP-projected k_t^2 P_IR (Task 5 at z = 0.38, fraction of the coefficient's scale: c0 / cfog =
        1.0 / 0.9% lcdm_fiducial, 1.0 / 1.8% massive_nu_030, 1.1 / 10.2% w0wa_m07_m10).
      "closure" (items 4-5): the 13 CLOSURE_LINES -- the projections clax-pt omits plus ps's
        IR-resummation scheme there -- per closure monomial times its value at theta. Equal to round-off to
        `ps_scheme_predicted` (the form Task 5 validates on 42 files), here read off `ps_scheme_all_rows`
        restricted to CLOSURE_LINES. The kept coefficient it subtracts is the rebuilt <X_clax>_l -
        `omitted_predicted`, which equals clax-pt's reported coefficient to <= 1e-10 (kept-coefficient check,
        test_omitted_projections_close), so no clax-pt coefficient spectra are needed.
      "kept" (item 5 on the rows clax keeps): ps's IR-scheme term on the 14 KEPT_SCHEME_ROWS (b2:l0/l2/l4,
        b1*b2:l0/l2, b2*b2:l0, bG2*bG2:l0, b2*bG2:l0 and the six F_G2 kept rows), `ps_scheme_all_rows` off
        CLOSURE_LINES, times the monomial values.
      "matter" (item 6): ps's exact-in-P_w matter-loop resummation minus clax's linearised one,
        e^{-x} (L22[P_w, P_w] + P_w I13[P_w]) on the monomials 1, b1, b1^2 (`matter_scheme_term`), times
        their values.
    Section comment above `scheme_integrands`: what the last two buy (every derivative line closes at ps
    nfft >= 512) and where each term comes from in clax/ept.py and ps_1loop.

    Not predicted: one-loop numerics (FFTLog sampling, splines, quadrature), which Tasks 4-5 gate directly
    coefficient by coefficient, and the matter loops' UV-normalisation difference (relative size 3e-6, job 24970).
    NumPy-valued -- every scheme part reads clax's mu-loop through the `capture_gl_inputs` spy -- so it cannot
    be differentiated with jax; derivatives of D are taken by central differences of this function.

    model = the ps_1loop instance whose P_IR enters "ctr" (default ps_model(inp["rbao"])); the derivative gate
    passes its nfft-512 model and the convergence ladder each rung's, so D follows ps's settings there."""
    th = np.array([[theta[n] for n in THETA_NAMES]], dtype=float)
    ps = ps_model(inp["rbao"]) if model is None else model
    ctr = model_difference_factory(ps, inp, kw, ndens=ndens)(th)[0]
    caps = scheme_captures(inp)
    rows = ps_scheme_all_rows(inp, kw, caps=caps)
    on_lines = restrict_to_closure_lines(rows)
    closure = sum(monomial_value(lab, theta) * on_lines[lab] for lab in rows)
    kept = sum(monomial_value(lab, theta) * (rows[lab] - on_lines[lab]) for lab in rows)
    matter = sum(monomial_value(lab, theta) * m for lab, m in matter_scheme_term(inp, kw, caps=caps).items())
    return {"ctr": np.asarray(ctr, dtype=float), "closure": np.asarray(closure, dtype=float),
            "kept": np.asarray(kept, dtype=float), "matter": np.asarray(matter, dtype=float)}


# --------------------------------------------------------------------------- derivative closure (Task 7; Task 9's ladder)
# Moved here from tests/test_ps1loop_gradients.py (Task 9, controller amendment B, Oct 12, 2026) so that the
# convergence ladder evaluates the same closure with each rung's ps_1loop model and Simpson num; with the
# defaults (model=None, num=256) every function below is the Task 7 code it replaced, line for line.

FD_STEP = 1e-4   # central-difference step along a tangent: dD/dt here, and ps's own AD-vs-FD check
TANGENTS = ("f", "a_perp", "a_para", "stretch", "amplitude")


def tangent_setup(inp, theta: dict, mask, *, model=None, num=256):
    """(ps_fun, clax_fun, x0, tangents) for the derivative closure at the shared inputs and the parameter
    point theta. ps_fun and clax_fun map x = (f, alpha_perp, alpha_para, P_lin) to the (3, Nw) multipoles on
    the window (AP on, ndens = NDENS_PROBE); clax_fun runs the traced IR split by default so a traced P_lin
    can be differentiated. x0 = the file's values; tangents = {name: dx} along f, alpha_perp, alpha_para,
    the k-stretch t = P dlnP/dlnk and the amplitude t = P. model / num = ps_1loop's instance and Simpson
    num (default: ps_model(rbao) and 256, what the gates run)."""
    k_h, kw = jnp.asarray(inp["k_h"]), jnp.asarray(inp["k_h"][mask])
    ps = ps_model(inp["rbao"]) if model is None else model

    def ps_fun(f, a_perp, a_para, pk):
        return ps_multipoles_jax(ps, k_h, pk, inp["h"], f, a_perp, a_para, theta, kw,
                                 ap=True, ndens=NDENS_PROBE, num=num)

    def clax_fun(f, a_perp, a_para, pk, ir_traced=True):
        hratio, Dratio = 1.0 / a_para, a_perp
        e = clax_ept(inp, pk=pk, f=f, hratio=hratio, Dratio=Dratio, ir_traced=ir_traced)
        cb = to_clax(theta, f=f, ndens=NDENS_PROBE, Pd2d2_0=e.Pd2d2_0, hratio=hratio, Dratio=Dratio)
        return clax_multipoles_jax(e, cb)[:, mask]

    a_perp, a_para = ap_alphas(inp["hratio"], inp["Dratio"])
    pk0 = jnp.asarray(inp["pk"])
    x0 = (jnp.asarray(inp["f"]), jnp.asarray(a_perp), jnp.asarray(a_para), pk0)
    t_stretch = pk0 * CubicSpline(jnp.log(k_h), jnp.log(pk0)).derivative(jnp.log(k_h))   # P dlnP/dlnk
    one, zero, zpk = jnp.asarray(1.0), jnp.asarray(0.0), jnp.zeros_like(pk0)
    tangents = {"f": (one, zero, zero, zpk), "a_perp": (zero, one, zero, zpk),
                "a_para": (zero, zero, one, zpk), "stretch": (zero, zero, zero, t_stretch),
                "amplitude": (zero, zero, zero, pk0)}
    return ps_fun, clax_fun, x0, tangents


def predicted_difference_at(inp, theta: dict, kw, x, *, model=None):
    """(D(x), parts): the documented ps - clax on the total multipoles (3, Nw) at x = (f, a_perp, a_para, P_lin),
    i.e. predicted_total_difference on a copy of the shared inputs carrying x (AP: hratio = 1/a_para,
    Dratio = a_perp, the convention of tangent_setup's clax_fun)."""
    f, a_perp, a_para, pk = x
    inp_x = {**inp, "f": float(f), "hratio": 1.0 / float(a_para), "Dratio": float(a_perp),
             "pk": np.asarray(pk, dtype=float)}
    parts = predicted_total_difference(inp_x, theta, kw, ndens=NDENS_PROBE, model=model)
    return sum(parts[n] for n in D_PARTS), parts


def primal_closure(ps_fun, clax_fun, inp, theta: dict, kw, x0, *, model=None) -> dict:
    """{"l<l>": {"direct", "closed", "k", "resid_max"}} at x0: direct = rel(ps, clax), closed = rel(ps - D, clax)
    (campaign metric, fraction of max|clax P_l| on the window), k = where |ps - D - clax| peaks, resid_max =
    max|ps - D - clax| itself [(Mpc/h)^3], so it can be put over another scale (the ladder's derivative scale)."""
    d0, _ = predicted_difference_at(inp, theta, kw, x0, model=model)
    ps0, cl0 = np.asarray(ps_fun(*x0)), np.asarray(clax_fun(*x0))
    return {f"l{l}": {"direct": cu.rel(ps0[i], cl0[i]), "closed": cu.rel(ps0[i] - d0[i], cl0[i]),
                      "k": float(kw[int(np.argmax(np.abs(ps0[i] - d0[i] - cl0[i])))]),
                      "resid_max": float(np.max(np.abs(ps0[i] - d0[i] - cl0[i])))}
            for i, l in enumerate(ELLS)}


def tangent_closure(ps_fun, clax_fun, inp, theta: dict, kw, x0, t, *, fd_step=FD_STEP, model=None) -> dict:
    """{"l<l>": {...}} along the tangent t: direct = rel(jvp_ps, jvp_clax); closed = rel(jvp_ps - dD/dt,
    jvp_clax), dD/dt the central difference of D at x0 +/- fd_step t (D is NumPy-valued, so not jvp-able);
    ratio = closed / direct; k / k_direct = where the closed / direct differences peak; dD_<part> = max|dD/dt|
    of each part of D (D_PARTS: counterterm map, the 13 closure lines, ps's scheme term on the kept bias rows,
    the matter-loop term), as fractions of max|jvp_clax|; scale = max|jvp_clax| itself (the derivative's
    scale every fraction here is taken of)."""
    jps = np.asarray(jax.jvp(ps_fun, x0, t)[1])
    jcl = np.asarray(jax.jvp(clax_fun, x0, t)[1])
    dp, pp = predicted_difference_at(inp, theta, kw, tuple(x + fd_step * dx for x, dx in zip(x0, t)), model=model)
    dm, pm = predicted_difference_at(inp, theta, kw, tuple(x - fd_step * dx for x, dx in zip(x0, t)), model=model)
    dD = (dp - dm) / (2.0 * fd_step)
    d_part = {n: (pp[n] - pm[n]) / (2.0 * fd_step) for n in D_PARTS}
    out = {}
    for i, l in enumerate(ELLS):
        scale = max(float(np.max(np.abs(jcl[i]))), 1e-300)
        direct = cu.rel(jps[i], jcl[i])
        resid = jps[i] - dD[i] - jcl[i]
        closed = cu.rel(jps[i] - dD[i], jcl[i])
        k_c = float(kw[int(np.argmax(np.abs(resid)))]) if np.all(np.isfinite(resid)) else float("nan")
        k_d = float(kw[int(np.argmax(np.abs(jps[i] - jcl[i])))])
        ratio = closed / direct if direct > 0 else float("nan")
        out[f"l{l}"] = {"direct": direct, "closed": closed, "ratio": ratio, "k": k_c, "k_direct": k_d, "scale": scale,
                        **{f"dD_{n}": float(np.max(np.abs(d_part[n][i])) / scale) for n in D_PARTS}}
    return out


# --------------------------------------------------------------------------- real space, IR off (spec §5.4)
# AP (controller ruling, Oct 7, 2026): real space has no AP and ps_1loop's get_pk_real / get_pk_gm_real
# apply none, so clax-pt is evaluated at hratio = Dratio = 1. CLASS-PT and clax-pt store Id2d2, Id2G2, IG2G2
# (pk_mult rows 1, 4, 5) as the AP-folded l = 0 mu-loop moment and reuse it in pk_gg_real (ept.py
# _gl_multipoles 1222-1223; nonlinear_pt.c 5322-5349); with the file's AP those three P_gg rows would
# differ from the AP-free real-space loops by the AP projection (0.57-0.68% at lcdm_fiducial z = 0.38) -- a
# CLASS-PT convention, not real-space numerics. The tie-breaker therefore reads only the no-AP IR-off files.

from clax.ept import pk_gg_real, pk_gm_real, pk_mm_real   # noqa: E402

REAL_CHANNELS = ("pk_mm_real", "pk_gg_real", "pk_gm_real")
REAL_THRESHOLDS = {c: 0.01 for c in REAL_CHANNELS}
# P_mm: 1, c0 (the k^2 P shape, cs0 = c0); P_gg: bias pairs + P_shot; P_gm: linear bias terms.
REAL_MONOMIALS = (("1",), ("c0",), ("b1",), ("b2",), ("bG2",), ("bGamma3",),
                  ("b1", "b1"), ("b1", "b2"), ("b2", "b2"), ("b1", "bG2"), ("bG2", "bG2"),
                  ("b2", "bG2"), ("b1", "bGamma3"), ("P_shot",))


@functools.partial(jax.jit, static_argnames=("model",))
def _ps_real_jit(model, k_h, pk, thetas, k, ndens):
    pk_data = {"k": k_h, "pk": pk}

    def one(th):
        t = dict(zip(THETA_NAMES, th))
        z0 = 0.0 * t["b1"]
        p = ps_params(t, h=1.0, f=0.0, ndens=ndens)   # h, f unused: no IR, no RSD in real space
        # P_mm = P_gg at b1 = 1, rest 0, carrying the c0 counterterm (ps_1loop.py:149-179, v0.5.0)
        mm = model.get_pk_real(k, pk_data, {**p, "bias": {"b1": 1.0 + z0, "b2": z0, "bG2": z0, "bGamma3": z0},
                                            "stoch": {"P_shot": z0, "a0": z0, "a2": z0}})
        # P_gg without counterterm: clax-pt's P_gg counterterm is parameterised differently (spec §5.4)
        gg = model.get_pk_real(k, pk_data, {**p, "ctr": {**p["ctr"], "c0": z0}})
        gm = model.get_pk_gm_real(k, pk_data, {**p, "ctr": {"c0": z0, "cs0": z0}})   # ps_1loop.py:181-258
        return jnp.stack([mm, gg, gm])
    return jax.vmap(one)(thetas)


def ps_real_batch(model, inp, kw, *, ndens=NDENS_PROBE):
    def F(thetas):
        return np.asarray(_ps_real_jit(model, jnp.asarray(inp["k_h"]), jnp.asarray(inp["pk"]),
                                       jnp.asarray(thetas, dtype=float), jnp.asarray(kw), ndens))
    return F


def clax_real_batch(e, mask, *, ndens=NDENS_PROBE):
    def one(th):
        t = dict(zip(THETA_NAMES, th))
        b = (t["b1"], t["b2"], t["bG2"], t["bGamma3"])
        return jnp.stack([pk_mm_real(e, cs0=t["c0"]),
                          pk_gg_real(e, *b, cs=0.0, cs0=0.0, Pshot=t["P_shot"] / ndens),
                          pk_gm_real(e, *b, cs0=0.0, cs=0.0)])[:, mask]
    vm = jax.vmap(one)
    return lambda thetas: np.asarray(vm(jnp.asarray(thetas, dtype=float)))


def classpt_real_batch(inp, mask, *, ndens=NDENS_PROBE):
    def F(thetas):
        rows = []
        for th in np.asarray(thetas, dtype=float):
            t = dict(zip(THETA_NAMES, th))
            # classy pk_mm_real(cs) reads `cs` (classy.pyx:4816); P_gg/P_gm counterterms off.
            cb = {"b1": t["b1"], "b2": t["b2"], "bG2": t["bG2"], "bGamma3": t["bGamma3"],
                  "cs0": 0.0, "cs2": 0.0, "cs4": 0.0, "cs": t["c0"], "Pshot": t["P_shot"] / ndens, "b4": 0.0}
            o = ca.assemble_from_pm(inp["pk_mult"], inp["h"], inp["f"], inp["k_h"], cb, inp["Pd2d2_0"])
            rows.append(np.stack([o[c] for c in REAL_CHANNELS])[:, mask])
        return np.stack(rows)
    return F


def real_coefficient_triplet(inp, *, irno_inp=None, ndens=NDENS_PROBE):
    """Real-space coefficient spectra {"ps", "clax", "classpt"} on the window, IR off in all three. clax-pt
    runs at hratio = Dratio = 1 whatever the file's AP (ruling above); irno_inp, the CLASS-PT tie-breaker,
    must be a no-AP IR-off file on the same k grid."""
    mask = cu.window(inp["k_h"])
    kw = inp["k_h"][mask]
    e = clax_ept(inp, ir=False, hratio=1.0, Dratio=1.0)
    res = {"ps": coefficients(ps_real_batch(ps_model(inp["rbao"], do_irres=False), inp, kw, ndens=ndens),
                              REAL_MONOMIALS),
           "clax": coefficients(clax_real_batch(e, mask, ndens=ndens), REAL_MONOMIALS),
           "classpt": {}}
    if irno_inp is not None:
        if has_ap(irno_inp):
            raise ValueError("ERROR real-space tie-breaker file carries AP (hratio, Dratio != 1); "
                             "CLASS-PT's AP-projected rows 1, 4, 5 must not be compared with AP-free real space")
        np.testing.assert_array_equal(irno_inp["k_h"], inp["k_h"])
        res["classpt"] = coefficients(classpt_real_batch(irno_inp, mask, ndens=ndens), REAL_MONOMIALS)
    return res, kw
