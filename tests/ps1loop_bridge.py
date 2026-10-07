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


def gate(res, kw, *, channels=CHANNELS, thresholds=COEF_THRESHOLDS):
    errs, lines = {}, []
    for lab, ref in res["clax"].items():
        scale = float(np.max(np.abs(ref)))
        for i, ch in enumerate(channels):
            e = coef_error(res["ps"][lab][i], ref[i], kw, scale=scale)
            name = f"{lab}:{ch}"
            errs[name] = e
            thr = thresholds[ch]
            if not np.isfinite(e["err"]) or e["err"] > thr:
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
