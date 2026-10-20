"""Preset solver choice after the Rodas5 dense-output fix (Oct 12, 2026).

Six configurations, all massless (CosmoParams(m_ncdm=0), ncdm_q_size 0),
unlensed C_l vs CLASS ``reference_data/cls_massless_recfast.npz`` (RECFAST),
read at l <= 1000 (set 1) or l <= 500 (set 2), plus cached per-stage timings.

Set 1 (retuned planck_cl candidates): 60 k/decade, k_max 1.0, n_tau 5000,
th_n_points 100000, hierarchy 17/17/17, max_steps 8192; C_l to l_max 2500
with compute_cls_all_fast, n_k_fine 20000, read at l <= 1000.
    A  kvaerno5  rtol 1e-4 / atol 1e-6
    B  rodas5    rtol 1e-4 / atol 1e-6
    C  rodas5    rtol 1e-3 / atol 1e-4   (the old fit_cl tolerances)
Set 2 (fit_cl-like): 20 k/decade, n_tau 2000, th_n_points 3000 (k_max,
hierarchy and max_steps as set 1; fit_cl itself uses max_steps 1024); C_l to
l_max 1500, n_k_fine 5000, read at l <= 500.
    D  kvaerno5  1e-4 / 1e-6      E  rodas5  1e-4 / 1e-6      F  rodas5  1e-3 / 1e-4

C_l are computed well beyond the read range because the sparse-l spline's
end condition contaminates the last ~50 multipoles below l_max (padded on
fix/preset-retune, not on this branch).  The Bessel table on this branch is
the old 3 points per unit x, so a common ~-1% TT offset at l < 50 and ~+20%
EE at l=2 is expected in ALL runs; the differentials B-A, C-A, E-D, F-D
isolate the solver and tolerance.  TE is scored as (TE - ref)/sqrt(TT_ref
EE_ref) (zero crossings make a plain ratio useless).

Timing as in the precision sweep: first call at the target cosmology
(compile + run; its C_l are scored), second call at a shifted cosmology
(h=0.70, omega_cdm=0.13) for cached timings.  JSON is rewritten after every
config so a killed job keeps what it finished.

Usage (igpu compute node, slurm/rodas5-dense-v100.sbatch MODE=preset):
    python diags/diag_preset_solver_choice.py --configs A,B,C,D,E,F --out r.json
    python diags/diag_preset_solver_choice.py --report r.json
"""
import argparse
import dataclasses
import json
import os
import socket
import subprocess
import sys
import time
import traceback

sys.path.insert(0, ".")

import numpy as np

REF = "reference_data/cls_massless_recfast.npz"

BASE1 = dict(ncdm_q_size=0, pt_k_per_decade=60, pt_k_max_cl=1.0, pt_tau_n_points=5000,
             th_n_points=100000, pt_l_max_g=17, pt_l_max_pol_g=17, pt_l_max_ur=17,
             ode_max_steps=8192)
BASE2 = dict(BASE1, pt_k_per_decade=20, pt_tau_n_points=2000, th_n_points=3000)
HR1 = dict(l_max=2500, n_k_fine=20000)
HR2 = dict(l_max=1500, n_k_fine=5000)
BANDS1 = [("tt", 2, 30), ("tt", 2, 500), ("tt", 2, 1000),
          ("ee", 2, 4), ("ee", 5, 500), ("ee", 5, 1000), ("ten", 2, 1000)]
BANDS2 = [("tt", 2, 30), ("tt", 2, 500), ("ee", 2, 4), ("ee", 5, 500), ("ten", 2, 500)]
PROBE_L = (2, 3, 5, 10, 20, 30, 50, 100, 500, 1000)

CONFIGS = {
    "A": dict(set=1, solver="kvaerno5", rtol=1e-4, atol=1e-6, note="retuned planck_cl (kvaerno5)"),
    "B": dict(set=1, solver="rodas5", rtol=1e-4, atol=1e-6, note="planck-like, rodas5"),
    "C": dict(set=1, solver="rodas5", rtol=1e-3, atol=1e-4, note="planck-like, rodas5 at old fit_cl tolerances"),
    "D": dict(set=2, solver="kvaerno5", rtol=1e-4, atol=1e-6, note="fit_cl-like, kvaerno5"),
    "E": dict(set=2, solver="rodas5", rtol=1e-4, atol=1e-6, note="fit_cl-like, rodas5"),
    "F": dict(set=2, solver="rodas5", rtol=1e-3, atol=1e-4, note="fit_cl-like, rodas5 at old fit_cl tolerances"),
}
DIFFS = [("B", "A"), ("C", "A"), ("E", "D"), ("F", "D")]


def _git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _save(results, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f)
    os.replace(tmp, path)


def _load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _build(name):
    from clax.params import PrecisionParams
    cfg = CONFIGS[name]
    base = BASE1 if cfg["set"] == 1 else BASE2
    prec = dataclasses.replace(PrecisionParams.planck_cl(), **base, pt_ode_solver=cfg["solver"],
                               pt_ode_rtol=cfg["rtol"], pt_ode_atol=cfg["atol"])
    hr = HR1 if cfg["set"] == 1 else HR2
    return cfg, prec, hr


def run_config(name, out_path, results):
    import jax
    from clax.params import CosmoParams
    from clax.background import background_solve
    from clax.thermodynamics import thermodynamics_solve
    from clax.perturbations import perturbations_solve
    from clax.harmonic import compute_cls_all_fast

    cfg, prec, hr = _build(name)
    params = CosmoParams(m_ncdm=0.0)
    params2 = CosmoParams(m_ncdm=0.0, h=0.70, omega_cdm=0.13)
    rec = dict(name=name, **cfg, hr=hr,
               settings={k: getattr(prec, k) for k in
                         ("pt_k_per_decade", "pt_k_max_cl", "pt_tau_n_points", "th_n_points",
                          "pt_l_max_g", "pt_l_max_pol_g", "pt_l_max_ur", "ode_max_steps",
                          "pt_ode_solver", "pt_ode_rtol", "pt_ode_atol", "ncdm_q_size")},
               status="running", host=socket.gethostname(), git=_git_head(),
               device=str(jax.devices()[0].device_kind), jax=jax.__version__,
               started=time.strftime("%Y-%m-%d %H:%M:%S"))
    results[name] = rec
    _save(results, out_path)
    print(f"\n=== {name}: {cfg['note']} | {cfg['solver']} {cfg['rtol']:.0e}/{cfg['atol']:.0e} "
          f"| hr l_max {hr['l_max']} n_k_fine {hr['n_k_fine']} ===", flush=True)

    def stage(fn, leaf):
        t0 = time.perf_counter()
        r = fn()
        jax.block_until_ready(leaf(r))
        return r, time.perf_counter() - t0

    def one(p, label):
        bg, t_bg = stage(lambda: background_solve(p, prec), lambda r: r.conformal_age)
        th, t_th = stage(lambda: thermodynamics_solve(p, prec, bg), lambda r: r.tau_star)
        pt, t_pt = stage(lambda: perturbations_solve(p, prec, bg, th), lambda r: r.source_T0)
        cls, t_hr = stage(lambda: compute_cls_all_fast(pt, p, bg, l_max=hr["l_max"],
                                                       n_k_fine=hr["n_k_fine"]),
                          lambda r: r["tt"])
        n_k, n_tau = int(len(pt.k_grid)), int(len(pt.tau_grid))
        del pt
        print(f"  {label}: bg {t_bg:.1f}s th {t_th:.1f}s pt {t_pt:.1f}s hr {t_hr:.1f}s "
              f"(n_k={n_k}, n_tau={n_tau})", flush=True)
        return cls, dict(bg=t_bg, th=t_th, pt=t_pt, hr=t_hr, total=t_bg + t_th + t_pt + t_hr,
                         n_k=n_k, n_tau=n_tau)

    try:
        cls, rec["first"] = one(params, "first (compile+run)")
        rec["cls"] = {k: np.asarray(cls[k], dtype=float).tolist() for k in ("ell", "tt", "ee", "te")}
        finite = all(np.isfinite(np.asarray(cls[k])).all() for k in ("tt", "ee", "te"))
        del cls
        cls2, rec["cached"] = one(params2, "cached (shifted cosmology)")
        del cls2
        rec["status"] = "ok" if finite else "nonfinite"
    except Exception as e:  # one bad config must not kill the run
        rec["status"] = "error"
        rec["error"] = f"{type(e).__name__}: {str(e)[:800]}"
        rec["traceback"] = traceback.format_exc()[-3000:]
        print(f"  ERROR {rec['error']}", flush=True)
    rec["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _save(results, out_path)
    try:
        jax.clear_caches()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _errs(cls, ref_tt, ref_ee, ref_te, l_read):
    """Per-l errors for l = 2..l_read against a reference (CLASS or another run)."""
    ell = np.arange(2, l_read + 1)
    tt, ee, te = (np.array(cls[k])[2:l_read + 1] for k in ("tt", "ee", "te"))
    rtt, ree, rte = ref_tt[2:l_read + 1], ref_ee[2:l_read + 1], ref_te[2:l_read + 1]
    return dict(ell=ell, tt=tt / rtt - 1, ee=ee / ree - 1, ten=(te - rte) / np.sqrt(rtt * ree))


def _band(err, spec, lo, hi):
    m = (err["ell"] >= lo) & (err["ell"] <= hi)
    a = np.abs(err[spec][m])
    i = int(np.argmax(a))
    return float(a[i]), int(err["ell"][m][i])


def _pct(x):
    return f"{100 * x:.3f}"


def report(path, md_path=None):
    results = _load(path)
    ref = np.load(REF)
    lines = []

    def P(s=""):
        lines.append(s)
        print(s)

    P("# Preset solver choice after the Rodas5 dense-output fix")
    P()
    P("Errors in percent: max |C_l/ref - 1| over the band (TEn = (TE - ref)/sqrt(TT_ref EE_ref)), "
      "with the l of the maximum. Timings: cached second call, seconds. "
      "Bessel table 3/unit x on this branch: expect a common ~-1% TT offset at l<50 and ~+20% EE at l=2.")
    P()
    for s, bands, l_read in ((1, BANDS1, 1000), (2, BANDS2, 500)):
        names = [n for n in CONFIGS if CONFIGS[n]["set"] == s and n in results]
        if not names:
            continue
        hdr = "| cfg | solver | rtol/atol | n_k | bg | th | pt | hr | total | " + \
              " | ".join(f"{sp.upper()} {lo}-{hi}" for sp, lo, hi in bands) + " |"
        P(f"## Set {s} (read l <= {l_read})")
        P()
        P(hdr)
        P("|" + "---|" * (hdr.count("|") - 1))
        for n in names:
            r = results[n]
            c = r.get("cached", {})
            row = [n, r["solver"], f"{r['rtol']:.0e}/{r['atol']:.0e}", str(r.get("first", {}).get("n_k", "-"))]
            row += [f"{c[k]:.1f}" if k in c else "-" for k in ("bg", "th", "pt", "hr", "total")]
            if r.get("status") == "ok":
                e = _errs(r["cls"], ref["tt"], ref["ee"], ref["te"], l_read)
                for sp, lo, hi in bands:
                    m, l_at = _band(e, sp, lo, hi)
                    row.append(f"{_pct(m)} (l={l_at})")
            else:
                row += [r.get("status", "?")] * len(bands)
            P("| " + " | ".join(row) + " |")
        P()
        # differentials
        P(f"Differentials (max |run/base - 1| per band; TEn vs the base run's sqrt(TT EE)):")
        P()
        P("| run vs base | " + " | ".join(f"{sp.upper()} {lo}-{hi}" for sp, lo, hi in bands) + " |")
        P("|" + "---|" * (len(bands) + 1))
        for a, b in DIFFS:
            if a in names and b in names and results[a].get("status") == "ok" and results[b].get("status") == "ok":
                base = results[b]["cls"]
                e = _errs(results[a]["cls"], np.array(base["tt"]), np.array(base["ee"]), np.array(base["te"]), l_read)
                cells = []
                for sp, lo, hi in bands:
                    m, l_at = _band(e, sp, lo, hi)
                    cells.append(f"{_pct(m)} (l={l_at})")
                P(f"| {a} vs {b} | " + " | ".join(cells) + " |")
        P()
        # low-l probe
        probe = [l for l in PROBE_L if l <= l_read]
        P(f"TT and EE vs CLASS at l = {probe} (percent):")
        P()
        for n in names:
            r = results[n]
            if r.get("status") != "ok":
                continue
            e = _errs(r["cls"], ref["tt"], ref["ee"], ref["te"], l_read)
            P(f"    {n} TT: " + " ".join(f"{100 * e['tt'][l - 2]:+.2f}" for l in probe))
            P(f"    {n} EE: " + " ".join(f"{100 * e['ee'][l - 2]:+.2f}" for l in probe))
        P()
    if md_path:
        with open(md_path, "w") as f:
            f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default="A,B,C,D,E,F")
    ap.add_argument("--out", default="")
    ap.add_argument("--report", default=None, help="results JSON to summarise")
    ap.add_argument("--md", default="", help="write the report markdown here")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.report:
        report(args.report, args.md or None)
        return
    if args.list:
        for n in CONFIGS:
            cfg, prec, hr = _build(n)
            print(n, cfg["note"], prec.pt_ode_solver, prec.pt_ode_rtol, prec.pt_ode_atol,
                  prec.pt_k_per_decade, prec.pt_tau_n_points, prec.th_n_points, prec.ode_max_steps, hr)
        return
    if not args.out:
        ap.error("--out is required for a run")
    names = [n.strip() for n in args.configs.split(",") if n.strip()]
    results = _load(args.out)
    print(f"configs={names} out={args.out} git={_git_head()}", flush=True)
    for n in names:
        run_config(n, args.out, results)
    print("done", flush=True)


if __name__ == "__main__":
    main()
