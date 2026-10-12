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
  * stretch_direct / stretch_closed / stretch_ratio (+ stretch_k) -- Task 7's derivative closure along
    the k-stretch tangent t = P dlnP/dlnk (br.tangent_closure, the same code that keeps
    test_cross_code_tangents red on its stretch lines): how differently the two codes respond to a shift
    of the BAO / broadband features in k, after the documented model difference is taken out;
  * ap_zero_offset -- max|P_l(ps, AP offsets x 0.001) - P_l(ps, AP off)| / max|P_l(ps, AP off)| per l
    and its k: ps_1loop's AP branch (P(k, mu) tabulated on its FFTLog k grid x mu grid, then remapped by
    interp2d) against its no-AP branch (evaluated directly) at almost zero AP offset (Task 8 scratch,
    job 24871). ap_zero_offset_clax is the same quantity for clax-pt, the size of the physical change.

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

Usage (igpu node through sbatch, env clax; slurm/ps1loop-ladder.sbatch):
  python scripts/ps1loop_convergence_ladder.py [--points lcdm_fiducial:2.2 ...] [--rungs default nfft512 ...]
"""
from __future__ import annotations

import argparse
import contextlib
import functools

import jax
import numpy as np

from tests import ept_campaign_utils as cu
from tests import pfs_survey as pfs
from tests import ps1loop_bridge as br

POINTS = (("lcdm_fiducial", 2.2), ("w0wa_m07_m10", 2.2), ("lcdm_fiducial", 0.38))
REMAINDER_POINTS = (("lcdm_fiducial", 0.38),)
RUNGS = ("default", "nfft512", "nfft1024", "simpson1024", "mugrid201", "best", "dewiggle_legacy", "rbao110")
LEGACY_CUTS = (140, 210)   # ps_1loop 0.4.0's DST harmonic cuts; 0.5.0 ships CLASS-PT's 120-240
AP_ZERO_SCALE = 1e-3       # AP offsets x 0.001 (Task 8 scratch, job 24871)
REMAINDER_FIELDS = ("primal_remainder", "stretch_direct", "stretch_closed", "stretch_ratio", "stretch_k",
                    "ap_zero_offset", "ap_zero_offset_clax")


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


def _model_and_num(rung: str, inp: dict):
    rbao = inp["rbao"]
    if rung == "default":
        return br.ps_model(rbao), 256
    if rung == "nfft512":
        return br.ps_model(rbao, nfft=512), 256
    if rung == "nfft1024":
        return br.ps_model(rbao, nfft=1024), 256
    if rung == "simpson1024":
        return br.ps_model(rbao), 1024
    if rung == "mugrid201":
        return br.ps_model(rbao, mu_grid=201), 256
    if rung == "best":
        return br.ps_model(rbao, nfft=1024, mu_grid=201), 1024
    if rung == "rbao110":
        return br.ps_model(110.0), 256          # ps_1loop's default BAO scale (finding 1, spec §9)
    if rung == "dewiggle_legacy":
        from ps_1loop_jax import PowerSpectrum1Loop
        # fresh, uncached: never shares compiled code with the lru-cached `default` instance
        return PowerSpectrum1Loop(do_irres=True, rbao=rbao, ks=br.IR_KS), 256
    raise ValueError(f"unknown rung {rung!r}; known: {RUNGS}")


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


def _ap_zero_offset_clax(inp, th, mask, ndens) -> dict:
    kw = inp["k_h"][mask]
    on_inp, off_inp = _ap_scaled(inp, AP_ZERO_SCALE), _ap_scaled(inp, 0.0)
    on = br.clax_batch(br.clax_ept(on_inp), on_inp, mask, ndens=ndens)(th)[0]
    off = br.clax_batch(br.clax_ept(off_inp), off_inp, mask, ndens=ndens)(th)[0]
    return _jump(on, off, kw)


def _remainders(model, num, inp, theta, mask, th, ndens) -> dict:
    """The amendment-B/D fields at one rung (module docstring). ndens = br.NDENS_PROBE, as in Task 7."""
    kw = inp["k_h"][mask]
    ps_fun, clax_fun, x0, tangents = br.tangent_setup(inp, theta, mask, model=model, num=num)
    primal = br.primal_closure(ps_fun, clax_fun, inp, theta, kw, x0, model=model)
    st = br.tangent_closure(ps_fun, clax_fun, inp, theta, kw, x0, tangents["stretch"], model=model)
    return {"primal_remainder": {l: {"err": primal[l]["closed"], "k": primal[l]["k"]} for l in primal},
            "stretch_direct": {l: st[l]["direct"] for l in st},
            "stretch_closed": {l: st[l]["closed"] for l in st},
            "stretch_ratio": {l: st[l]["ratio"] for l in st},
            "stretch_k": {l: st[l]["k"] for l in st},
            "ap_zero_offset": _ap_zero_offset_ps(model, num, inp, th, kw, ndens)}


def run(points=POINTS, rungs=RUNGS) -> list[dict]:
    records = []
    slices = {sl.z_mid: sl for sl in pfs.load_slices()}
    for case, z in points:
        ref = cu.load_reference(case, z)
        if ref is None:
            raise SystemExit(f"ERROR reference missing: {case} z={z} -- run slurm/classpt-refgen-pfs.sbatch")
        inp = br.shared_inputs(ref)
        mask = cu.window(inp["k_h"])
        kw = inp["k_h"][mask]
        sl = slices.get(z)                         # None: not a PFS slice -> no Delta chi^2
        ndens = sl.nbar if sl is not None else br.NDENS_PROBE
        theta = pfs.bias_point(z)
        th = np.array([[theta[n] for n in br.THETA_NAMES]], dtype=float)
        cl = br.clax_batch(br.clax_ept(inp), inp, mask, ndens=ndens)(th)[0]
        C = pfs.multipole_covariance(pfs.to_centres(cl, kw), pfs.K_CENTRES, sl.V, pfs.DK) if sl is not None else None
        with_rem = (case, z) in REMAINDER_POINTS
        ap0_clax = _ap_zero_offset_clax(inp, th, mask, ndens) if with_rem else None
        for rung in rungs:
            ctx = _wiggle_cuts(*LEGACY_CUTS) if rung == "dewiggle_legacy" else contextlib.nullcontext()
            with ctx:
                model, num = _model_and_num(rung, inp)
                ps = br.ps_batch(model, inp, kw, ndens=ndens, num=num)(th)[0]
                rem = _remainders(model, num, inp, theta, mask, th, ndens) if with_rem else {}
            if with_rem:
                rem["ap_zero_offset_clax"] = ap0_clax
            errs = {f"l{l}": {"err": cu.rel(ps[i], cl[i]), "k": float(kw[int(np.argmax(np.abs(ps[i] - cl[i])))])}
                    for i, l in enumerate(br.ELLS)}
            chi = pfs.delta_chi2(pfs.to_centres(ps - cl, kw), C) if C is not None else float("nan")
            settings = {"nfft": int(model._nfft), "mu_grid": int(model._mu.size), "num": num,
                        "rbao": float(model.rbao),
                        "dst_cuts": list(LEGACY_CUTS) if rung == "dewiggle_legacy" else [120, 240]}
            br.log_record(layer="ladder", case=case, z=z, preset=rung, errors=errs,
                          extra={"delta_chi2": chi, "settings": settings, **rem})
            rec = {"case": case, "z": z, "rung": rung, "delta_chi2": chi,
                   **{k: v["err"] for k, v in errs.items()}, **{f: rem.get(f) for f in REMAINDER_FIELDS}}
            records.append(rec)
            dchi = f"dchi2 {chi:.4f}" if np.isfinite(chi) else "dchi2 n/a (not a PFS slice)"
            print(f"{case} z={z} {rung:15s} l0 {100 * rec['l0']:.3f}% l2 {100 * rec['l2']:.3f}% "
                  f"l4 {100 * rec['l4']:.3f}% {dchi}", flush=True)
            if with_rem:
                print(_remainder_line(case, z, rung, rem), flush=True)
    return records


def _remainder_line(case, z, rung, rem) -> str:
    ls = [f"l{l}" for l in br.ELLS]
    pr = " ".join(f"{l} {100 * rem['primal_remainder'][l]['err']:.3f}%@{rem['primal_remainder'][l]['k']:.3f}"
                  for l in ls)
    st = " ".join(f"{l} {100 * rem['stretch_direct'][l]:.3f}/{100 * rem['stretch_closed'][l]:.3f}% "
                  f"r={rem['stretch_ratio'][l]:.3f}@{rem['stretch_k'][l]:.3f}" for l in ls)
    ap = " ".join(f"{l} {rem['ap_zero_offset'][l]['err']:.3e}@{rem['ap_zero_offset'][l]['k']:.3f}" for l in ls)
    apc = " ".join(f"{l} {rem['ap_zero_offset_clax'][l]['err']:.3e}" for l in ls)
    return (f"{case} z={z} {rung:15s} REMAINDER primal {pr} | stretch direct/closed {st} | "
            f"ap_zero_offset ps {ap} (clax {apc})")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--points", nargs="+", default=[f"{c}:{z}" for c, z in POINTS])
    p.add_argument("--rungs", nargs="+", default=list(RUNGS), choices=list(RUNGS))
    a = p.parse_args(argv)
    run(points=tuple((s.split(":")[0], float(s.split(":")[1])) for s in a.points), rungs=tuple(a.rungs))


if __name__ == "__main__":
    main()
