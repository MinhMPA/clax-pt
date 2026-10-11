"""Rodas5 dense output: C_l vs Kvaerno5 and perturbation-stage timing (GPU).

Settings are the precision sweep's config "I6" / "G6" (Oct 10-11, 2026):
planck_cl + ncdm_q_size=0 (massless, CosmoParams(m_ncdm=0)), hierarchy
12/10/17, rtol 1e-4 / atol 1e-6, max_steps 32768, ~300 k-modes, 5000 tau
points; solver from ``--solver``.  Timing follows the sweep script: first
call at the target cosmology (compile + run; its C_l are saved), second call
at a shifted cosmology (h=0.70, omega_cdm=0.13) for cached timings.  The C_l
stage is compute_cls_all_fast at l_max 2500, n_k_fine 5000.

Usage (igpu compute node via slurm/rodas5-dense-v100.sbatch):
    python diags/diag_rodas5_dense_output.py --solver rodas5 --out r5.json
    python diags/diag_rodas5_dense_output.py --solver kvaerno5 --out k5.json
    python diags/diag_rodas5_dense_output.py --report r5.json k5.json
--report prints rodas5/kvaerno5 - 1 at selected l, band maxima, and the
timings of both runs (first JSON = candidate, second = reference).
"""
import argparse
import dataclasses
import json
import os
import socket
import subprocess
import sys
import time

sys.path.insert(0, ".")

import numpy as np

PREC_OVERRIDES = dict(
    ncdm_q_size=0, pt_l_max_g=12, pt_l_max_pol_g=10, pt_l_max_ur=17,
    pt_ode_rtol=1e-4, pt_ode_atol=1e-6, ode_max_steps=32768,
)
PROBE_L = (2, 5, 10, 20, 30, 100, 500, 1000, 1500, 2000, 2500)
BANDS = [(2, 30), (31, 500), (501, 1000), (1001, 2000), (2001, 2500)]


def _git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def run(solver, out, l_max, n_k_fine, rtol, atol, max_steps, skip_cached):
    import jax
    from clax.params import CosmoParams, PrecisionParams
    from clax.background import background_solve
    from clax.thermodynamics import thermodynamics_solve
    from clax.perturbations import perturbations_solve
    from clax.harmonic import compute_cls_all_fast

    ov = dict(PREC_OVERRIDES, pt_ode_solver=solver)
    if rtol is not None:
        ov["pt_ode_rtol"] = rtol
    if atol is not None:
        ov["pt_ode_atol"] = atol
    if max_steps is not None:
        ov["ode_max_steps"] = max_steps
    prec = dataclasses.replace(PrecisionParams.planck_cl(), **ov)
    params = CosmoParams(m_ncdm=0.0)
    params2 = CosmoParams(m_ncdm=0.0, h=0.70, omega_cdm=0.13)
    rec = dict(solver=solver, overrides=ov, l_max=l_max, n_k_fine=n_k_fine,
               host=socket.gethostname(), git=_git_head(),
               device=str(jax.devices()[0].device_kind), jax=jax.__version__,
               started=time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"=== {solver}: rtol {prec.pt_ode_rtol:.0e} atol {prec.pt_ode_atol:.0e} "
          f"max_steps {prec.ode_max_steps} @ {rec['git']} on {rec['device']} ===", flush=True)

    def stage(fn, leaf):
        t0 = time.perf_counter()
        r = fn()
        jax.block_until_ready(leaf(r))
        return r, time.perf_counter() - t0

    def one(p, label):
        bg, t_bg = stage(lambda: background_solve(p, prec), lambda r: r.conformal_age)
        th, t_th = stage(lambda: thermodynamics_solve(p, prec, bg), lambda r: r.tau_star)
        pt, t_pt = stage(lambda: perturbations_solve(p, prec, bg, th), lambda r: r.source_T0)
        cls, t_hr = stage(lambda: compute_cls_all_fast(pt, p, bg, l_max=l_max, n_k_fine=n_k_fine),
                          lambda r: r["tt"])
        n_k, n_tau = int(len(pt.k_grid)), int(len(pt.tau_grid))
        print(f"  {label}: bg {t_bg:.1f}s th {t_th:.1f}s pt {t_pt:.1f}s hr {t_hr:.1f}s "
              f"(n_k={n_k}, n_tau={n_tau})", flush=True)
        return cls, dict(bg=t_bg, th=t_th, pt=t_pt, hr=t_hr, n_k=n_k, n_tau=n_tau)

    cls, rec["first"] = one(params, "first (compile+run)")
    rec["cls"] = {k: np.asarray(cls[k], dtype=float).tolist() for k in ("ell", "tt", "ee", "te")}
    rec["finite"] = bool(all(np.isfinite(np.asarray(cls[k])).all() for k in ("tt", "ee", "te")))
    del cls
    if not skip_cached:
        cls2, rec["cached"] = one(params2, "cached (shifted cosmology)")
        rec["finite"] = rec["finite"] and bool(all(np.isfinite(np.asarray(cls2[k])).all()
                                                   for k in ("tt", "ee", "te")))
        del cls2
    rec["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(out, "w") as f:
        json.dump(rec, f)
    print(f"saved {out} (finite={rec['finite']})", flush=True)


def report(paths):
    recs = []
    for p in paths:
        with open(p) as f:
            recs.append(json.load(f))
    for r, p in zip(recs, paths):
        first, cached = r.get("first", {}), r.get("cached", {})
        print(f"{os.path.basename(p)}: {r['solver']} @ {r['git']} on {r['device']} "
              f"rtol {r['overrides']['pt_ode_rtol']:.0e} atol {r['overrides']['pt_ode_atol']:.0e}"
              f" n_k={first.get('n_k')} | first pt {first.get('pt', float('nan')):.1f}s "
              f"hr {first.get('hr', float('nan')):.1f}s | cached pt {cached.get('pt', float('nan')):.1f}s "
              f"th {cached.get('th', float('nan')):.1f}s hr {cached.get('hr', float('nan')):.1f}s"
              f" | finite={r.get('finite')}")
    if len(recs) < 2:
        return
    a, b = recs[0], recs[1]
    L = min(len(a["cls"]["tt"]), len(b["cls"]["tt"])) - 1
    ell = np.arange(2, L + 1)
    att, aee, ate = (np.array(a["cls"][k])[2:L + 1] for k in ("tt", "ee", "te"))
    btt, bee, bte = (np.array(b["cls"][k])[2:L + 1] for k in ("tt", "ee", "te"))
    err = dict(tt=att / btt - 1, ee=aee / bee - 1, ten=(ate - bte) / np.sqrt(btt * bee))
    print(f"\n{a['solver']} / {b['solver']} - 1 (TEn = dTE/sqrt(TT*EE)), in percent:")
    print("   l   " + "".join(f"{l:>9d}" for l in PROBE_L if l <= L))
    for k in ("tt", "ee", "ten"):
        print(f"  {k.upper():>4} " + "".join(f"{100 * err[k][l - 2]:>+9.3f}" for l in PROBE_L if l <= L))
    print("  band maxima:")
    for lo, hi in BANDS:
        if lo > L:
            continue
        m = (ell >= lo) & (ell <= min(hi, L))
        s = []
        for k in ("tt", "ee", "ten"):
            i = np.argmax(np.abs(err[k][m]))
            s.append(f"{k.upper()} {100 * abs(err[k][m][i]):.3f}% (l={ell[m][i]})")
        print(f"    l={lo}-{min(hi, L)}: " + ", ".join(s))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solver", default="rodas5", choices=["rodas5", "kvaerno5"])
    ap.add_argument("--out", default="")
    ap.add_argument("--l-max", type=int, default=2500)
    ap.add_argument("--n-k-fine", type=int, default=5000)
    ap.add_argument("--rtol", type=float, default=None, help="override pt_ode_rtol (default 1e-4)")
    ap.add_argument("--atol", type=float, default=None, help="override pt_ode_atol (default 1e-6)")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--skip-cached", action="store_true", help="no second (cached-timing) call")
    ap.add_argument("--report", nargs="*", default=None, help="JSONs: candidate then reference")
    args = ap.parse_args()
    if args.report is not None:
        report(args.report)
        return
    if not args.out:
        ap.error("--out is required for a run")
    run(args.solver, args.out, args.l_max, args.n_k_fine, args.rtol, args.atol,
        args.max_steps, args.skip_cached)


if __name__ == "__main__":
    main()
