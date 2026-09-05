#!/usr/bin/env python
# scripts/generate_classpt_reference.py
"""Generate CLASS-PT reference multipoles for the clax-pt validation campaign.

Runs in the `classpt` env (Task A2) only:
    micromamba run -n classpt env PYTHONPATH=<repo> python scripts/generate_classpt_reference.py \
        --cosmology lcdm_fiducial --z-list 0 0.38 0.8
Writes validation_cosmologies.reference_path(...) per z (spec §4.8) with the
raw `get_pk_mult` rows plus every classy accessor, and asserts the NumPy twin
(scripts/classpt_assembly.py) reproduces classy to 1e-10 on each file.
`--legacy` regenerates the legacy fiducial (LEGACY_CLASSPT_FIDUCIAL, cb=No,
BBN YHe) for the provenance gate (Task A4).
`--class-extra '<JSON dict>'` is merged into the classy params dict LAST (it
can override anything `classpt_params_from` set) and is recorded verbatim in
the written `params_json` (Ruling 13, Task A4 fix round 1): used to probe
CLASS-core-version drift (e.g. `N_ur`, `recombination`) against the legacy
z=0.38 reference without touching the campaign's default-input fiducial.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts import classpt_assembly as ca
from scripts import validation_cosmologies as vc

CLASSPT_DIR = Path("/home/n2minh/CLASS-PT")


def _classpt_commit() -> str:
    return subprocess.run(["git", "-C", str(CLASSPT_DIR), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def _patch_shas() -> dict:
    pdir = vc.REPO_ROOT / "scripts" / "classpt_patches"
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(pdir.glob("*.patch"))}


def _assert_close(name, a, b, rtol=1e-10):
    a, b = np.asarray(a, float), np.asarray(b, float)
    scale = max(np.max(np.abs(b)), 1e-300)
    err = np.max(np.abs(a - b)) / scale
    if not err < rtol:
        raise AssertionError(f"ERROR twin mismatch {name}: max|twin-classy|/max|classy| = {err:.3e}")


def run(case: str, cosmo: dict, z_list, *, ap: bool, omfid: float, cb: bool, bias_name: str,
        yhe, use_ppf, tag: str, outdir: Path | None, class_extra: dict | None = None) -> list[Path]:
    from classy import Class

    prm = vc.classpt_params_from(cosmo, z_list=z_list, ap=ap, omfid=omfid, cb=cb, yhe=yhe, use_ppf=use_ppf)
    if class_extra:
        prm.update(class_extra)   # merged LAST (Ruling 13): overrides anything above,
                                  # recorded verbatim via params_json below
    bias = vc.BIAS if bias_name == "fiducial" else vc.BIAS_NONZERO
    M = Class()
    M.set(prm)
    M.compute()
    if not (hasattr(M, "get_ap_ratios") and hasattr(M, "get_Pd2d2_0")):
        sys.exit("ERROR classy is unpatched: run scripts/setup_classpt_env.sh (Task A2)")
    # classy is a Cython cdef class, so its methods take NO keyword arguments
    # ("TypeError: Class.pk_mm_real() takes no keyword arguments"); every accessor
    # call below is positional, in the order of its live `def` line.  Guard the one
    # arity that silently misbinds (ref §11): the pre-A3 generator passed pk_gg_l0
    # seven positional args, which against the live 9-arg signature (classy.pyx:4900
    # `b1,b2,bG2,bGamma3,cs0,Pshot_nbar,a0_nbar,a2_nbar,b4`) would bind b4 to a0_nbar.
    try:
        M.pk_gg_l0(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)     # legacy 7-arg form
        bound_seven = True
    except TypeError:
        bound_seven = False                               # 9-arg signature, as targeted
    except Exception:
        bound_seven = True                                # bound 7 args, failed further in
    if bound_seven:
        sys.exit("ERROR classy.pk_gg_l0 accepts 7 positional args: not the 9-arg "
                 "signature this generator targets (classy.pyx:4900)")
    h = M.h()
    k_h = vc.ept_kgrid_numpy()
    k = k_h * h                                   # CLASS units for every classy call
    n_ncdm = int(prm.get("N_ncdm", 0))
    written = []
    for z in z_list:
        M.initialize_output(k, z, len(k_h))       # sets kh (h/Mpc, patched), fz, pk_mult, Pd2d2_0
        pm = np.asarray(M.get_pk_mult(k, z, len(k_h)), dtype=float)
        fz = float(M.scale_independent_growth_factor_f(z))
        hratio, Dratio, growthf = M.get_ap_ratios(z)
        Pd2d2_0 = float(M.get_Pd2d2_0())
        pk_m_lin = np.array([M.pk_lin(ki, z) for ki in k]) * h**3
        pk_cb_lin = np.array([M.pk_cb_lin(ki, z) for ki in k]) * h**3 if n_ncdm > 0 else None
        pk_lin = pk_cb_lin if cb else pk_m_lin
        b = bias
        classy_out = {
            # classy.pyx:4816  pk_mm_real(cs)
            "pk_mm_real": M.pk_mm_real(b["cs"]),
            # classy.pyx:4822  pk_gg_real(b1, b2, bG2, bGamma3, cs, cs0, Pshot)
            "pk_gg_real": M.pk_gg_real(b["b1"], b["b2"], b["bG2"], b["bGamma3"],
                                       b["cs"], b["cs0"], b["Pshot"]),
            # classy.pyx:4829  pk_gm_real(b1, b2, bG2, bGamma3, cs, cs0)
            "pk_gm_real": M.pk_gm_real(b["b1"], b["b2"], b["bG2"], b["bGamma3"],
                                       b["cs"], b["cs0"]),
            # classy.pyx:4881/4887/4893  pk_mm_l0(cs0) / pk_mm_l2(cs2) / pk_mm_l4(cs4)
            "pk_mm_l0": M.pk_mm_l0(b["cs0"]),
            "pk_mm_l2": M.pk_mm_l2(b["cs2"]),
            "pk_mm_l4": M.pk_mm_l4(b["cs4"]),
            # classy.pyx:4900  pk_gg_l0(b1, b2, bG2, bGamma3, cs0, Pshot_nbar, a0_nbar, a2_nbar, b4)
            "pk_gg_l0": M.pk_gg_l0(b["b1"], b["b2"], b["bG2"], b["bGamma3"],
                                   b["cs0"], b["Pshot"], 0.0, 0.0, b["b4"]),
            # classy.pyx:4914  pk_gg_l2(b1, b2, bG2, bGamma3, cs2, a2_nbar, b4)
            "pk_gg_l2": M.pk_gg_l2(b["b1"], b["b2"], b["bG2"], b["bGamma3"],
                                   b["cs2"], 0.0, b["b4"]),
            # classy.pyx:4925  pk_gg_l4(b1, b2, bG2, bGamma3, cs4, b4)
            "pk_gg_l4": M.pk_gg_l4(b["b1"], b["b2"], b["bG2"], b["bGamma3"],
                                   b["cs4"], b["b4"]),
        }
        classy_out = {kk: np.asarray(v, dtype=float) for kk, v in classy_out.items()}
        # --- falsify the twin against classy on this very file ---
        twin = ca.assemble_from_pm(pm, h, fz, k_h, bias, Pd2d2_0)
        for kk in classy_out:
            _assert_close(kk, twin[kk], classy_out[kk])
        _assert_close("Pd2d2_0", ca.pd2d2_0(pm[14] * h**3, k_h), Pd2d2_0, rtol=1e-8)
        # growthf is background_at_tau(tau_of_z) (nonlinear_pt.c:1262), fz is
        # background_at_z (classy.pyx:2382): the same f through two interpolation paths.
        _assert_close("growthf==fz", growthf, fz, rtol=1e-8)
        # Magnitude-only reminder that pm[14] is the IR-resummed tree, not pk_lin:
        # they differ by ~1-5% at the BAO scale (nonlinear_pt.c:2999, Bug #4).
        _assert_close("pm[14]==pk_lin(IR-resummed tree differs: expect fail)", pm[14] * h**3, pk_lin, rtol=1.0)
        if not ap:
            _assert_close("hratio", hratio, 1.0, rtol=1e-14)
            _assert_close("Dratio", Dratio, 1.0, rtol=1e-14)
        path = vc.reference_path(case, z, ap=ap, omfid=omfid, cb=cb, bias=bias_name, tag=tag)
        if outdir is not None:
            path = Path(outdir) / path.relative_to(vc.REFERENCE_ROOT)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            k_h=k_h, z=z, h=h, fz=fz, growthf=growthf,
            D_z=float(M.scale_independent_growth_factor(z)), H_z=float(M.Hubble(z)),
            DA_z=float(M.angular_distance(z)), rs_d=float(M.rs_drag()),   # Mpc; nonlinear_pt.c:2919 rbao
            hratio=hratio, Dratio=Dratio, Pd2d2_0=Pd2d2_0,
            pk_lin=pk_lin, pk_m_lin=pk_m_lin, pk_mult=pm, kh_convention="h/Mpc",
            ap=ap, omfid=omfid, cb=cb, use_ppf=str(prm.get("use_ppf", "default")),
            params_json=json.dumps(prm, sort_keys=True), bias_json=json.dumps(bias, sort_keys=True),
            classpt_commit=_classpt_commit(), patches_sha256=json.dumps(_patch_shas(), sort_keys=True),
            **classy_out,
        )
        if pk_cb_lin is not None:
            payload["pk_cb_lin"] = pk_cb_lin
        np.savez(path, **payload)
        print(f"wrote {path}  hratio={hratio:.6f} Dratio={Dratio:.6f} f={fz:.6f}")
        written.append(path)
    M.struct_cleanup()
    M.empty()
    return written


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--cosmology", choices=list(vc.CASES))
    g.add_argument("--legacy", action="store_true", help="LEGACY_CLASSPT_FIDUCIAL, cb=No, BBN YHe")
    g.add_argument("--list-distinct", action="store_true")
    p.add_argument("--z-list", type=float, nargs="+", default=list(vc.Z_LIST))
    p.add_argument("--ap", choices=["yes", "no"], default="yes")
    p.add_argument("--omfid", type=float, default=vc.OMFID)
    p.add_argument("--cb", choices=["yes", "no"], default="yes")
    p.add_argument("--bias", choices=["fiducial", "nonzero"], default="fiducial")
    p.add_argument("--yhe", default=str(vc.Y_HE_CLAX), help="float or 'none' (CLASS-PT BBN default)")
    p.add_argument("--use-ppf", choices=["default", "yes", "no"], default="default")
    p.add_argument("--tag", default="")
    p.add_argument("--outdir", default=None)
    p.add_argument("--class-extra", default=None,
                    help="JSON dict merged into the classy params LAST (overrides anything); "
                         "recorded verbatim in params_json")
    a = p.parse_args(argv)
    if a.list_distinct:
        print("\n".join(vc.distinct_cases()))
        return
    yhe = None if a.yhe.lower() == "none" else float(a.yhe)
    use_ppf = {"default": None, "yes": True, "no": False}[a.use_ppf]
    class_extra = json.loads(a.class_extra) if a.class_extra else None
    if a.legacy:
        run("legacy_fiducial", vc.LEGACY_CLASSPT_FIDUCIAL, a.z_list, ap=a.ap == "yes", omfid=a.omfid,
            cb=False, bias_name=a.bias, yhe=None, use_ppf=use_ppf, tag=a.tag, outdir=a.outdir,
            class_extra=class_extra)
    else:
        run(a.cosmology, vc.cosmo_params(a.cosmology), a.z_list, ap=a.ap == "yes", omfid=a.omfid,
            cb=a.cb == "yes", bias_name=a.bias, yhe=yhe, use_ppf=use_ppf, tag=a.tag, outdir=a.outdir,
            class_extra=class_extra)


if __name__ == "__main__":
    main()
