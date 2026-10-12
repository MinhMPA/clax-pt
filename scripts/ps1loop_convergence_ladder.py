#!/usr/bin/env python
"""Convergence ladder for ps_1loop's settings (spec §7) -- a diagnostic, not a gate.

For each (case, z) point and each rung (one ps_1loop setting changed at a time; `best` changes three),
logs the ps_1loop-vs-clax-pt error per l (campaign metric, window) at the PFS bias point and, where z is
a PFS slice, the slice Delta chi^2, to test_logs/ps1loop/errors.jsonl (layer "ladder"). A gate failure
that a rung removes is a resolution effect (spec §6), filed as a recommended setting.

At REMAINDER_POINTS (lcdm_fiducial, z = 0.38; Task 9 controller amendments B and D, Oct 12-13, 2026)
every rung also logs the two open remainders, with ps_1loop at that rung's settings:
  * primal_remainder -- max|ps - clax - D| / max|clax P_l| per l and its k, D = the documented model
    difference (br.predicted_total_difference, its ps part rebuilt with the rung's model);
    primal_remainder_over_dscale -- the same max|ps - clax - D| over the stretch derivative's scale
    max|jvp_clax| (Task 7's R0_over_scale), so that stretch_amplification = stretch_closed / it compares
    the derivative remainder with the primal one over ONE scale (Task 9 fix round 1);
  * stretch_direct / stretch_closed / stretch_ratio (+ stretch_k) -- Task 7's derivative closure along
    the k-stretch tangent t = P dlnP/dlnk (br.tangent_closure, the same code that keeps
    test_cross_code_tangents red on its stretch lines): how differently the two codes respond to a shift
    of the BAO / broadband features in k, after the documented model difference is taken out;
  * ap_zero_offset -- max|P_l(ps, AP offsets x 0.001) - P_l(ps, AP off)| / max|P_l(ps, AP off)| per l
    and its k: ps_1loop's AP branch (P(k, mu) tabulated on its FFTLog k grid x mu grid, then remapped by
    interp2d) against its no-AP branch (evaluated directly) at almost zero AP offset (Task 8 scratch,
    job 24871). ap_zero_offset_clax is the same quantity for clax-pt, the size of the physical change;
  * ap_unit_alpha -- the same comparison with ps's AP branch forced at alpha_perp = alpha_para = 1: no
    physical AP response at all, so it is the interpolation error of ps's P(k, mu) table by itself.
PFS-slice floor (pfs_ap_floor; Task 9 fix round 1, item 2b): Task 8's construction (job 24871) repeated
per rung -- lcdm_fiducial, the 7 PFS slices, c = 2, cfog = 500, AP offsets x 0.001, covariance from clax
at the same offsets -- giving the survey Delta chi^2 of ps - clax, of ps's jump between its AP branch and
its no-AP branch, and of clax's (the physical change), with the largest jump in units of sigma.

Rungs (br.ps_model exposes nfft and the mu grid; br.ps_batch the Simpson num):
  default          the gates' ps_1loop: nfft 256, mu grid 51, Simpson num 256, DST cuts 120-240 (0.5.0),
                   rbao = this cosmology's r_s(z_d) h
  nfft512/1024     FFTLog sampling -- also the k grid (geomspace 1e-5..1e3 h/Mpc) on which ps tabulates
                   the one-loop P(k, mu) and, in the AP branch, the whole P(k, mu)
  simpson1024      Simpson num of the final mu projection
  mugrid201        the mu grid of the same table (one-loop interpolation and AP remap)
  best             nfft 1024 + mu grid 201 + Simpson num 1024
  dewiggle_legacy  ps_1loop 0.4.0's DST cuts 140-210 (what the legacy de-wiggling cost)
  rbao110          ps_1loop's default BAO scale instead of r_s(z_d) h (finding 1, spec §9)
Not laddered (br.ps_model does not expose them): the FFTLog range kmin_fft / kmax_fft and the IR cut ks.
Not laddered because ps_1loop 0.5.0 cannot run it: nfft 2048 -- its PT matrices come out non-finite there
(pt_matrix.get_I, a ratio of complex Gamma functions, overflows in its divide at the larger FFTLog frequencies; job 24917:
"RuntimeWarning: overflow encountered in divide", then non-finite multipoles).

Usage (igpu node through sbatch, env clax; slurm/ps1loop-ladder.sbatch):
  python scripts/ps1loop_convergence_ladder.py [--points lcdm_fiducial:2.2 ...] [--rungs default nfft512 ...]
                                               [--floor-rungs default best ...]
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import inspect
import time

import jax
import numpy as np

from tests import ept_campaign_utils as cu
from tests import pfs_survey as pfs
from tests import ps1loop_bridge as br

POINTS = (("lcdm_fiducial", 2.2), ("w0wa_m07_m10", 2.2), ("lcdm_fiducial", 0.38))
REMAINDER_POINTS = (("lcdm_fiducial", 0.38),)
RUNGS = ("default", "nfft512", "nfft1024", "simpson1024", "mugrid201", "best", "dewiggle_legacy", "rbao110")
FLOOR_CASE = "lcdm_fiducial"
FLOOR_RUNGS = ("default", "nfft1024", "mugrid201", "best")
DEFAULT_NFFT = 256         # ps_1loop's FFTLog sampling, as the gates run it
DEFAULT_NUM = 256          # Simpson num of the mu projection, as the gates run it (br.ps_batch default)
LEGACY_CUTS = (140, 210)   # ps_1loop 0.4.0's DST harmonic cuts; 0.5.0 ships CLASS-PT's 120-240
AP_ZERO_SCALE = 1e-3       # AP offsets x 0.001 (Task 8 scratch, job 24871)
REMAINDER_FIELDS = ("primal_remainder", "primal_remainder_over_dscale", "stretch_direct", "stretch_closed",
                    "stretch_ratio", "stretch_k", "stretch_amplification", "ap_zero_offset", "ap_unit_alpha",
                    "ap_zero_offset_clax")


@contextlib.contextmanager
def _wiggle_cuts(n_min: int, n_max: int):
    """Swap ps_1loop's DST mode cuts (ir_resum.remove_wiggle, 0.5.0 default 120-240) for (n_min, n_max).
    Caches are cleared on entry and exit: compiled code would keep the old cuts."""
    from ps_1loop_jax import ir_resum
    original = ir_resum.remove_wiggle
    ir_resum.remove_wiggle = functools.partial(original, n_min=n_min, n_max=n_max)
    jax.clear_caches()
    try:
        yield
    finally:
        ir_resum.remove_wiggle = original
        jax.clear_caches()


def _rung_context(rung: str):
    return _wiggle_cuts(*LEGACY_CUTS) if rung == "dewiggle_legacy" else contextlib.nullcontext()


def _dst_cuts_in_effect() -> list[int]:
    """The DST cuts ps_1loop's remove_wiggle uses right now (its signature's defaults; inside _wiggle_cuts the
    partial's), read rather than assumed."""
    from ps_1loop_jax import ir_resum
    params = inspect.signature(ir_resum.remove_wiggle).parameters
    return [int(params["n_min"].default), int(params["n_max"].default)]


def _model_and_num(rung: str, inp: dict):
    rbao = inp["rbao"]
    if rung == "default":
        return br.ps_model(rbao, nfft=DEFAULT_NFFT), DEFAULT_NUM
    if rung == "nfft512":
        return br.ps_model(rbao, nfft=512), DEFAULT_NUM
    if rung == "nfft1024":
        return br.ps_model(rbao, nfft=1024), DEFAULT_NUM
    if rung == "simpson1024":
        return br.ps_model(rbao, nfft=DEFAULT_NFFT), 1024
    if rung == "mugrid201":
        return br.ps_model(rbao, nfft=DEFAULT_NFFT, mu_grid=201), DEFAULT_NUM
    if rung == "best":
        return br.ps_model(rbao, nfft=1024, mu_grid=201), 1024
    if rung == "rbao110":
        return br.ps_model(110.0, nfft=DEFAULT_NFFT), DEFAULT_NUM   # ps_1loop's default BAO scale (finding 1)
    if rung == "dewiggle_legacy":
        from ps_1loop_jax import PowerSpectrum1Loop
        # fresh, uncached: never shares compiled code with the lru-cached `default` instance
        return PowerSpectrum1Loop(do_irres=True, rbao=rbao, ks=br.IR_KS, nfft=DEFAULT_NFFT), DEFAULT_NUM
    raise ValueError(f"unknown rung {rung!r}; known: {RUNGS}")


def _settings(model, num) -> dict:
    return {"nfft": int(model._nfft), "mu_grid": int(model._mu.size), "num": int(num),
            "rbao": float(model.rbao), "dst_cuts": _dst_cuts_in_effect()}


def _ap_scaled(inp: dict, s: float) -> dict:
    """The shared inputs with the AP offsets (hratio - 1, Dratio - 1) scaled by s; s = 0 is AP off."""
    return {**inp, "hratio": 1.0 + s * (inp["hratio"] - 1.0), "Dratio": 1.0 + s * (inp["Dratio"] - 1.0)}


def _jump(on, off, kw) -> dict:
    """{"l<l>": {"err": max|on - off| / max|off|, "k": where |on - off| peaks}} on the window."""
    return {f"l{l}": {"err": cu.rel(on[i], off[i]), "k": float(kw[int(np.argmax(np.abs(on[i] - off[i])))])}
            for i, l in enumerate(br.ELLS)}


def _ap_zero_offset_ps(model, num, inp, th, kw, ndens) -> dict:
    on = br.ps_batch(model, _ap_scaled(inp, AP_ZERO_SCALE), kw, ndens=ndens, num=num)(th)[0]   # AP branch
    off = br.ps_batch(model, _ap_scaled(inp, 0.0), kw, ndens=ndens, num=num)(th)[0]           # no-AP branch
    return _jump(on, off, kw)


def _ap_unit_alpha_ps(model, num, inp, th, kw, ndens) -> dict:
    """ps's AP branch forced at alpha_perp = alpha_para = 1 vs its no-AP branch: zero physical response."""
    unit = _ap_scaled(inp, 0.0)
    on = br.ps_batch(model, unit, kw, ndens=ndens, num=num, ap=True)(th)[0]
    off = br.ps_batch(model, unit, kw, ndens=ndens, num=num, ap=False)(th)[0]
    return _jump(on, off, kw)


def _ap_zero_offset_clax(inp, th, mask, ndens) -> dict:
    kw = inp["k_h"][mask]
    on_inp, off_inp = _ap_scaled(inp, AP_ZERO_SCALE), _ap_scaled(inp, 0.0)
    on = br.clax_batch(br.clax_ept(on_inp), on_inp, mask, ndens=ndens)(th)[0]
    off = br.clax_batch(br.clax_ept(off_inp), off_inp, mask, ndens=ndens)(th)[0]
    return _jump(on, off, kw)


def _remainders(model, num, inp, theta, mask, th) -> dict:
    """The amendment-B/D fields at one rung (module docstring), all at ndens = br.NDENS_PROBE -- the value
    br.tangent_setup fixes, as in Task 7."""
    kw = inp["k_h"][mask]
    ndens = br.NDENS_PROBE
    ps_fun, clax_fun, x0, tangents = br.tangent_setup(inp, theta, mask, model=model, num=num)
    primal = br.primal_closure(ps_fun, clax_fun, inp, theta, kw, x0, model=model)
    st = br.tangent_closure(ps_fun, clax_fun, inp, theta, kw, x0, tangents["stretch"], model=model)
    over_dscale = {l: primal[l]["resid_max"] / st[l]["scale"] for l in primal}
    return {"primal_remainder": {l: {"err": primal[l]["closed"], "k": primal[l]["k"]} for l in primal},
            "primal_remainder_over_dscale": over_dscale,
            "stretch_direct": {l: st[l]["direct"] for l in st},
            "stretch_closed": {l: st[l]["closed"] for l in st},
            "stretch_ratio": {l: st[l]["ratio"] for l in st},
            "stretch_k": {l: st[l]["k"] for l in st},
            "stretch_amplification": {l: st[l]["closed"] / over_dscale[l] for l in st},
            "ap_zero_offset": _ap_zero_offset_ps(model, num, inp, th, kw, ndens),
            "ap_unit_alpha": _ap_unit_alpha_ps(model, num, inp, th, kw, ndens)}


def run(points=POINTS, rungs=RUNGS) -> list[dict]:
    records = []
    slices = {sl.z_mid: sl for sl in pfs.load_slices()}
    for case, z in points:
        inp = br.shared_inputs(_require(case, z))
        mask = cu.window(inp["k_h"])
        kw = inp["k_h"][mask]
        sl = slices.get(z)                         # None: not a PFS slice -> no Delta chi^2
        with_rem = (case, z) in REMAINDER_POINTS
        if with_rem and sl is not None:
            raise ValueError(f"ERROR remainder point {case} z={z} is a PFS slice: its spectra line would use the "
                             f"slice nbar while the remainders use br.NDENS_PROBE")
        ndens = sl.nbar if sl is not None else br.NDENS_PROBE
        theta = pfs.bias_point(z)
        th = np.array([[theta[n] for n in br.THETA_NAMES]], dtype=float)
        cl = br.clax_batch(br.clax_ept(inp), inp, mask, ndens=ndens)(th)[0]
        C = pfs.multipole_covariance(pfs.to_centres(cl, kw), pfs.K_CENTRES, sl.V, pfs.DK) if sl is not None else None
        ap0_clax = _ap_zero_offset_clax(inp, th, mask, ndens) if with_rem else None
        for rung in rungs:
            t0 = time.perf_counter()
            with _rung_context(rung):
                model, num = _model_and_num(rung, inp)
                settings = _settings(model, num)
                ps = br.ps_batch(model, inp, kw, ndens=ndens, num=num)(th)[0]
                rem = _remainders(model, num, inp, theta, mask, th) if with_rem else {}
            if with_rem:
                rem["ap_zero_offset_clax"] = ap0_clax
            errs = {f"l{l}": {"err": cu.rel(ps[i], cl[i]), "k": float(kw[int(np.argmax(np.abs(ps[i] - cl[i])))])}
                    for i, l in enumerate(br.ELLS)}
            chi = pfs.delta_chi2(pfs.to_centres(ps - cl, kw), C) if C is not None else float("nan")
            secs = time.perf_counter() - t0
            br.log_record(layer="ladder", case=case, z=z, preset=rung, errors=errs,
                          extra={"delta_chi2": chi, "settings": settings, "seconds": secs, **rem})
            rec = {"case": case, "z": z, "rung": rung, "delta_chi2": chi, "settings": settings,
                   **{k: v["err"] for k, v in errs.items()}, **{f: rem.get(f) for f in REMAINDER_FIELDS}}
            records.append(rec)
            dchi = f"dchi2 {chi:.4f}" if np.isfinite(chi) else "dchi2 n/a (not a PFS slice)"
            print(f"{case} z={z} {rung:15s} l0 {100 * rec['l0']:.3f}% l2 {100 * rec['l2']:.3f}% "
                  f"l4 {100 * rec['l4']:.3f}% {dchi}  [{secs:.0f} s]", flush=True)
            if with_rem:
                for line in _remainder_lines(case, z, rung, rem):
                    print(line, flush=True)
    return records


def _remainder_lines(case, z, rung, rem) -> list[str]:
    ls = [f"l{l}" for l in br.ELLS]
    pr = " ".join(f"{l} {100 * rem['primal_remainder'][l]['err']:.3f}%@{rem['primal_remainder'][l]['k']:.3f}"
                  for l in ls)
    pd = " ".join(f"{l} {100 * rem['primal_remainder_over_dscale'][l]:.4f}%" for l in ls)
    st = " ".join(f"{l} {100 * rem['stretch_direct'][l]:.3f}/{100 * rem['stretch_closed'][l]:.3f}% "
                  f"r={rem['stretch_ratio'][l]:.3f}@{rem['stretch_k'][l]:.3f}" for l in ls)
    am = " ".join(f"{l} {rem['stretch_amplification'][l]:.1f}x" for l in ls)
    ap = " ".join(f"{l} {rem['ap_zero_offset'][l]['err']:.3e}@{rem['ap_zero_offset'][l]['k']:.3f}" for l in ls)
    ua = " ".join(f"{l} {rem['ap_unit_alpha'][l]['err']:.3e}@{rem['ap_unit_alpha'][l]['k']:.3f}" for l in ls)
    apc = " ".join(f"{l} {rem['ap_zero_offset_clax'][l]['err']:.3e}@{rem['ap_zero_offset_clax'][l]['k']:.3f}"
                   for l in ls)
    return [f"{case} z={z} {rung:15s} REMAINDER primal {pr} | primal/dscale {pd} | stretch direct/closed {st} | "
            f"amplification {am}",
            f"{case} z={z} {rung:15s} AP zero_offset ps {ap} | unit_alpha ps {ua} | zero_offset clax {apc}"]


def pfs_ap_floor(case=FLOOR_CASE, rungs=FLOOR_RUNGS) -> list[dict]:
    """Task 8's AP-branch floor on the PFS slices (job 24871 construction), per rung: at every slice the bias
    point (c = 2, cfog = 500, ndens = nbar), the file's AP offsets x AP_ZERO_SCALE in both codes, the slice
    covariance from clax at those offsets. Sums over the 7 slices of the Delta chi^2 of ps - clax (Task 8 at
    the default setting: 0.1283), of ps - clax with AP off in both codes, of ps's jump (AP branch at the
    tiny offsets minus no-AP branch) and of clax's jump; max_sigma = the largest |jump| / sigma(k-bin, l)."""
    out = []
    per_slice = []
    for sl in pfs.load_slices():
        inp = br.shared_inputs(_require(case, sl.z_mid))
        mask = cu.window(inp["k_h"])
        kw = inp["k_h"][mask]
        th = np.array([[pfs.bias_point(sl.z_mid)[n] for n in br.THETA_NAMES]], dtype=float)
        on_inp, off_inp = _ap_scaled(inp, AP_ZERO_SCALE), _ap_scaled(inp, 0.0)
        cl_on = br.clax_batch(br.clax_ept(on_inp), on_inp, mask, ndens=sl.nbar)(th)[0]
        cl_off = br.clax_batch(br.clax_ept(off_inp), off_inp, mask, ndens=sl.nbar)(th)[0]
        C = pfs.multipole_covariance(pfs.to_centres(cl_on, kw), pfs.K_CENTRES, sl.V, pfs.DK)
        per_slice.append((sl, kw, th, on_inp, off_inp, cl_on, cl_off, C))
    for rung in rungs:
        t0 = time.perf_counter()
        tot = {"vs_clax": 0.0, "vs_clax_ap_off": 0.0, "ps_jump": 0.0, "clax_jump": 0.0}
        sig = {"ps_jump": 0.0, "clax_jump": 0.0}
        slices_out = {}
        with _rung_context(rung):
            model, num = _model_and_num(rung, per_slice[0][4])   # rbao = r_s(z_d) h: the same at every slice
            settings = _settings(model, num)
            for sl, kw, th, on_inp, off_inp, cl_on, cl_off, C in per_slice:
                ps_on = br.ps_batch(model, on_inp, kw, ndens=sl.nbar, num=num)(th)[0]
                ps_off = br.ps_batch(model, off_inp, kw, ndens=sl.nbar, num=num)(th)[0]
                cen = {n: pfs.to_centres(v, kw) for n, v in
                       (("ps_on", ps_on), ("ps_off", ps_off), ("cl_on", cl_on), ("cl_off", cl_off))}
                sigma = np.sqrt(np.stack([C[:, i, i] for i in range(3)]))          # (3, Nb)
                d = {"vs_clax": cen["ps_on"] - cen["cl_on"], "vs_clax_ap_off": cen["ps_off"] - cen["cl_off"],
                     "ps_jump": cen["ps_on"] - cen["ps_off"], "clax_jump": cen["cl_on"] - cen["cl_off"]}
                row = {n: pfs.delta_chi2(v, C) for n, v in d.items()}
                for n in tot:
                    tot[n] += row[n]
                for n in sig:
                    row[f"{n}_max_sigma"] = float(np.max(np.abs(d[n]) / sigma))
                    sig[n] = max(sig[n], row[f"{n}_max_sigma"])
                slices_out[f"z{sl.z_mid}"] = row
        secs = time.perf_counter() - t0
        rec = {"case": case, "rung": rung, "settings": settings, **tot,
               **{f"{n}_max_sigma": v for n, v in sig.items()}, "slices": slices_out}
        br.log_record(layer="ladder", case=case, z=float("nan"), preset=f"{rung}:pfs-ap-floor",
                      errors={n: {"err": v, "k": float("nan")} for n, v in tot.items()},
                      extra={**{k: v for k, v in rec.items() if k not in ("case", "rung")},
                             "seconds": secs, "ap_scale": AP_ZERO_SCALE})
        out.append(rec)
        print(f"PFS-AP-FLOOR {case} {rung:15s} offsets x {AP_ZERO_SCALE:g}, 7 slices, cfog 500: "
              f"dchi2 ps-clax {tot['vs_clax']:.4f} (AP off {tot['vs_clax_ap_off']:.4f}) | "
              f"ps jump dchi2 {tot['ps_jump']:.3e} max {sig['ps_jump']:.4f} sigma | "
              f"clax jump dchi2 {tot['clax_jump']:.3e} max {sig['clax_jump']:.4f} sigma  [{secs:.0f} s]", flush=True)
    return out


def _require(case, z):
    ref = cu.load_reference(case, z)
    if ref is None:
        raise SystemExit(f"ERROR reference missing: {case} z={z} -- run slurm/classpt-refgen-pfs.sbatch")
    return ref


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--points", nargs="+", default=[f"{c}:{z}" for c, z in POINTS])
    p.add_argument("--rungs", nargs="+", default=list(RUNGS), choices=list(RUNGS))
    p.add_argument("--floor-rungs", nargs="*", default=list(FLOOR_RUNGS), choices=list(RUNGS),
                   help="rungs for the PFS-slice AP floor (none: skip it)")
    a = p.parse_args(argv)
    run(points=tuple((s.split(":")[0], float(s.split(":")[1])) for s in a.points), rungs=tuple(a.rungs))
    if a.floor_rungs:
        pfs_ap_floor(rungs=tuple(a.floor_rungs))


if __name__ == "__main__":
    main()
