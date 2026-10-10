# tests/ps1loop_bridge.py
"""Bridge for the clax-pt vs ps_1loop_jax second-oracle validation.

Spec: docs/superpowers/specs/2026-10-05-clax-pt-vs-ps1loop-validation-design.md (glossary:
CONTEXT.md). Shared-input comparison: both codes receive one CLASS-PT reference file's own
P_cb(k) [(Mpc/h)^3 on k_h in h/Mpc], growth rate f, r_s(z_d) h [Mpc/h] and AP ratios, so a
residual is one-loop numerics or a documented model difference (spec §4.2).

Conventions pinned here, each with its test in tests/test_ps1loop_bridge.py:
  * AP: ps_1loop alpha_perp = Dratio, alpha_para = 1/hratio, Jacobian V = hratio/Dratio^2
    (ps_1loop.py:328-344 vs ept.py:1135-1141).
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

from clax.ept import EPTPrecisionParams, compute_ept, pk_gg_l0, pk_gg_l2, pk_gg_l4
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
    PowerSpectrum1Loop.get_pk_ell / get_pk_ell_ref (ps_1loop.py:316-357): mu = linspace(0,1,num),
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


def ps_batch(model, inp, kw, *, ndens=NDENS_PROBE, num=256):
    a_perp, a_para = ap_alphas(inp["hratio"], inp["Dratio"])
    args = (jnp.asarray(inp["k_h"]), jnp.asarray(inp["pk"]), inp["h"], inp["f"], a_perp, a_para)

    def F(thetas):
        return np.asarray(_ps_batch_jit(model, *args, jnp.asarray(thetas, dtype=float),
                                        jnp.asarray(kw), ndens, ap=has_ap(inp), num=num))
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
            # ps_1loop: P_shot/ndens x Jacobian V, P_d2d2(k->0) subtracted (ps_1loop.py:381-384);
            # clax-pt: Pshot in observed coordinates, +0.25 b2^2 Pd2d2_0 added back (classy.pyx:4911).
            "Pshot": V * theta["P_shot"] / ndens - 0.25 * theta["b2"] ** 2 * Pd2d2_0,
            # ps_1loop.py:576-588 in true coordinates: a0 (k_t/k_nl)^2 -> l=0 moment; a2 k_t^2 mu_t^2 = k^2 mu^2 hratio^2
            "a0": V * theta["a0"] / ndens * knl2 * (2.0 / (3.0 * Dratio**2) + hratio**2 / 3.0),
            "a2": V * hratio**2 * theta["a2"] / ndens * knl2}


# --------------------------------------------------------------------------- clax-pt and CLASS-PT

def clax_ept(inp, *, ir=True, pk=None, f=None, hratio=None, Dratio=None):
    """compute_ept on the shared inputs; the overrides let Task 7 trace f, AP and P_lin."""
    return compute_ept(jnp.asarray(inp["pk"]) if pk is None else pk, jnp.asarray(inp["k_h"]),
                       h=inp["h"], f=inp["f"] if f is None else f,
                       prec=EPTPrecisionParams(ir_resummation=ir), rs_h=inp["rbao"],
                       hratio=inp["hratio"] if hratio is None else hratio,
                       Dratio=inp["Dratio"] if Dratio is None else Dratio)


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


def coefficient_triplet(inp, monomials, *, ir=True, ndens=NDENS_PROBE):
    mask = cu.window(inp["k_h"])
    kw = inp["k_h"][mask]
    e = clax_ept(inp, ir=ir)
    res = {"ps": coefficients(ps_batch(ps_model(inp["rbao"], do_irres=ir), inp, kw, ndens=ndens), monomials),
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

    ps_1loop model (true coordinates, Jacobian V; ps_1loop.py:496-547, 576-588, 328-344):
      V [-2 k_t^2 (c0 + c2 f mu_t^2 + c4 f^2 mu_t^4) P_IR - cfog f^4 k_t^4 mu_t^4 (b1 + f mu_t^2)^2 P_IR
         + a0 (k_t/k_nl)^2 / ndens], projected on L_l(mu) over the observed mu.
    clax-pt model (ept.py:1207-1209 and pk_gg_l0/l2/l4):
      2 cs_l Pctr_l + f^2 b4 k^2 poly_l (35/8) Pctr_4 [+ a0 (k/0.45)^2 at l=0],
      Pctr_l = -proj_l(V k_t^2 P_IR f^{l/2} mu_t^l)."""
    from ps_1loop_jax import ir_resum
    h, f, hr, Dr = inp["h"], inp["f"], inp["hratio"], inp["Dratio"]
    pk_data = {"k": jnp.asarray(inp["k_h"]), "pk": jnp.asarray(inp["pk"])}
    # same call as PowerSpectrum1Loop.get_pkmu (ps_1loop.py:291-292)
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
# P_l (mirroring CLASS-PT), while ps_1loop keeps them. Closed form (task-4-diagnosis.md §2.1; controller
# amendments of Oct 8, 2026): rebuild each mu-loop integrand from the EXACT arguments clax's
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


def mu_loop_projections(cap) -> dict[str, np.ndarray]:
    """{"<integrand>:<l>": (Nk,)} -- the l = 0, 2, 4 projections, on clax's whole k grid, of the mu-loop
    integrands behind the closure lines: "IFG2" = IFG2_in, "fmu2IFG2" = f mu_t^2 IFG2_in, the isotropic
    loops c(Pk_Id2d2), c(Pk_Id2G2), c(Pk_IG2G2), and "b1b2" / "b1bG2" = c(Pk_0_b) + L2(mu_t) c(Pk_2_b).
    Line-by-line mirror of clax/ept.py _gl_multipoles: geometry and weights 1133-1150, channels at k_t
    1155, damping Sigma_tot(mu_t) 1167-1169, p_lin 1175, p_lo and IFG2_in 1214-1216, isotropic loops
    1222-1223, b1b2/b1bG2 1224-1232 (CLASS-PT nonlinear_pt.c:4382-4481, 5273-5351)."""
    import clax.ept as E
    chan, k, f, s2, ds2 = cap["chan"], cap["k"], cap["f"], cap["s2"], cap["ds2"]
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
    at = E._channels_at(k, chan, kk)
    Sig = (s2 * (1.0 + f * mu2t * (2.0 + f))
           + ds2 * f ** 2 * mu2t * (mu2t - 1.0))
    Exp = jnp.exp(-Sig * kk ** 2)
    p_lin = at["pk_nw"] + at["pk_w"] * Exp
    Pbin = at["pk_disc"]
    p_lo = p_lin / jnp.where(Pbin > 1e-100, Pbin, 1.0)
    IFG2_in = p_lo * at["Pk_IFG2"]
    L2t = 0.5 * (3.0 * mu2t - 1.0)
    integrands = {"IFG2": IFG2_in, "fmu2IFG2": IFG2_in * f * mu2t,
                  **{n: at[n] for n in ("Pk_Id2d2", "Pk_Id2G2", "Pk_IG2G2")},
                  **{b: at[f"Pk_0_{b}"] + L2t * at[f"Pk_2_{b}"] for b in ("b1b2", "b1bG2")}}
    return {f"{name}:{l}": np.asarray(jnp.einsum("m,mk->k", W[l], val))
            for name, val in integrands.items() for l in ELLS}


def omitted_projections(inp, kw, *, captured=None) -> dict[str, np.ndarray]:
    """`mu_loop_projections` of clax's own mu-loop on the shared inputs, on the comparison window kw
    (= window_k(inp)). `captured` = capture_gl_inputs(inp), to reuse one compute_ept."""
    _, cap = capture_gl_inputs(inp) if captured is None else captured
    mask = cu.window(inp["k_h"])
    if not np.array_equal(np.asarray(kw, dtype=float), inp["k_h"][mask]):
        raise ValueError("ERROR kw is not the comparison window of these inputs")
    if not np.array_equal(np.asarray(cap["k"]), inp["k_h"]):
        raise RuntimeError("ERROR clax's mu-loop k grid differs from the shared-input k grid")
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


# --------------------------------------------------------------------------- ps IR-resummation scheme (spec §4.2)
# `omitted_predicted` is clax's OWN mu-loop integrand projected on the omitted l. ps_1loop also resums every
# loop differently: loop[P_nw] + e^{-x(mu)} (loop[P] - loop[P_nw]) (ps_1loop.py get_pkmu_irres_LO_NLO), while
# clax-pt / CLASS-PT feed the isotropically damped P_nw + e^{-k^2 Sigma^2} P_w into the bias loops and rescale
# only F_G2 by p_lo(mu). The two schemes nearly cancel at l = 0 but not at l = 2, 4 -- exactly the omitted
# projections (task-4-diagnosis.md §1 E3c, §2.3). USER DECISION (Oct 9, 2026; option (a) of task-5-report.md
# §6): the closure predicts ps - clax = <X_ps>_l - clax_l, with each monomial's integrand X_ps rebuilt in ps's
# scheme from clax's own IR-OFF channels -- a documented model difference with a closed form (spec §4.2).
# Provenance: ported from the scratch diagnostic
# /lustre/work/n2minh/std/clax/ps1loop/scratch-task5/test_t5_diag.py (`geometry`, `mono_integrands`, `one`),
# validated over all 42 files in job 23255 (all 13 lines close; worst 0.56% on a 1% gate, 0.71% on a 2% gate).

# The IR-off bias channels the closure monomials' integrands read (clax sign convention, ept.py:1907-1926).
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


def ps_scheme_predicted(inp, kw, clax_coef, *, captured=None) -> dict:
    """{monomial label: (3, Nw)} predicted ps_1loop - clax-pt on the 13 CLOSURE_LINES, ps's IR scheme included:
        <X_ps(mono)>_l - clax_coef[mono]_l,   X_ps = C[P_nw](k_t) + e^{-x(k_t, mu_t)} (C[P] - C[P_nw])(k_t),
    with C[.] clax's IR-OFF bias channels computed on the shared P and on clax's own P_nw (both splined at k_t
    by clax's `_channels_at`), x = k_t^2 Sigma_tot^2(mu_t), and Sigma^2, dSigma^2, P_nw and the AP geometry
    all taken from the same IR-on mu-loop (`captured` = capture_gl_inputs(inp)). IR off, the F_G2 channel
    carries its external spectrum (p_lo = 1), so C_FG2[X] = X I_FG2[X], as in ps's loop[X]. <.>_l uses clax's
    AP-folded GL weights. clax_coef = coefficient_triplet's "clax" entry on the same kw. Rows outside
    CLOSURE_LINES are zero. Equals omitted_predicted + <X_ps - X_clax>_l (the scheme term) to round-off."""
    import clax.ept as E
    _, cap_on = capture_gl_inputs(inp) if captured is None else captured
    _, cap_off = capture_gl_inputs(inp, ir=False)
    _, cap_nw = capture_gl_inputs(inp, ir=False, pk=jnp.asarray(cap_on["chan"]["pk_nw"]))
    mask = cu.window(inp["k_h"])
    if not np.array_equal(np.asarray(kw, dtype=float), inp["k_h"][mask]):
        raise ValueError("ERROR kw is not the comparison window of these inputs")
    for cap in (cap_on, cap_off, cap_nw):
        if not np.array_equal(np.asarray(cap["k"]), inp["k_h"]):
            raise RuntimeError("ERROR clax's mu-loop k grid differs from the shared-input k grid")
    k, f = cap_on["k"], cap_on["f"]
    mu = jnp.asarray(E._GAUSS_NODES)
    w = jnp.asarray(E._GAUSS_WEIGHTS)
    hratio = jnp.asarray(cap_on["hratio"], dtype=k.dtype)
    Dratio = jnp.asarray(cap_on["Dratio"], dtype=k.dtype)
    inv_D2 = 1.0 / Dratio ** 2
    ap_fac = jnp.sqrt(inv_D2 + (hratio ** 2 - inv_D2) * mu ** 2)[:, None]       # ept.py:1137-1140
    mu2t = (mu[:, None] * hratio / ap_fac) ** 2
    kk = k[None, :] * ap_fac
    V = hratio / Dratio ** 2
    L2 = 0.5 * (3.0 * mu ** 2 - 1.0)
    L4 = (35.0 * mu ** 4 - 30.0 * mu ** 2 + 3.0) / 8.0
    W = {0: 0.5 * w * V, 2: 2.5 * w * L2 * V, 4: 4.5 * w * L4 * V}              # ept.py:1150
    Sig = (cap_on["s2"] * (1.0 + f * mu2t * (2.0 + f))
           + cap_on["ds2"] * f ** 2 * mu2t * (mu2t - 1.0))                      # ept.py:1167-1168
    Exp = jnp.exp(-Sig * kk ** 2)                                               # e^{-x(k_t, mu_t)}
    C = E._channels_at(k, {n: cap_off["chan"][n] for n in PS_SCHEME_CHANNELS}, kk)
    Cn = E._channels_at(k, {n: cap_nw["chan"][n] for n in PS_SCHEME_CHANNELS}, kk)
    X = {n: Cn[n] + Exp * (C[n] - Cn[n]) for n in PS_SCHEME_CHANNELS}
    integ = closure_integrands(X, f, mu2t, X["Pk_IFG2"])
    pred = {}
    for mono in CLOSURE_MONOMIALS:
        lab = label(mono)
        proj = np.stack([np.asarray(jnp.einsum("m,mk->k", W[l], integ[lab]))[mask] for l in ELLS])
        pred[lab] = proj - np.asarray(clax_coef[lab], dtype=float)
    return restrict_to_closure_lines(pred)
