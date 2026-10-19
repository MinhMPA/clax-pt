"""Cross-check the committed thermo_recfast_hesw1 references against stock CLASS.

The references in reference_data/<name>/thermo_recfast_hesw1.npz (smsharma/clax#48)
were generated from a CLASS-PT build (CLASS v3.3.4 thermodynamics). This script
re-runs generate_class_reference.py's exact parameter set and z grid with whatever
`classy` is importable (meant: stock CLASS 3.3.4, e.g. the PyPI wheel) and prints
max |x_e_stock / x_e_ref - 1| and |T_b_stock / T_b_ref - 1| per cosmology, plus per
z window into an optional JSON file.

Exit status: 0 if every cosmology is within BAR in x_e and T_b; 1 if any is not;
2 if every difference is exactly zero, which means the imported classy is the build
that made the references (it reproduces them bit for bit), not an independent CLASS.
The CLASS-PT build also reports v3.3.4, so the version cannot tell them apart; the
import path is printed first.

Measured Oct 19, 2026 (stock classy 3.3.4 from PyPI, references at
tol_thermo_integration = 1e-10): x_e <= 2.9e-8 (largest at z ~ 1700-1770), T_b
<= 2.1e-10, z 2870-5000 <= 6.5e-12, at all 11 cosmologies. On the earlier
references at CLASS's default 1e-6 (Oct 16, 2026, 10 cosmologies) the same check
gave x_e <= 7.1e-6 and T_b <= 7.7e-7, and 2.3e-4 at omega_b - 20%: the two builds'
own integration error, which the tighter tolerance removes.

Usage:  python scripts/crosscheck_recfast_hesw1_stock.py [out.json]
"""
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
WINDOWS = [(200, 800), (800, 1550), (1550, 1650), (1650, 2870), (2870, 5000), (5000, 8000)]
BAR = 1e-6  # ~30x the largest measured stock-vs-reference difference (2.9e-8)


def main():
    spec = importlib.util.spec_from_file_location(
        "gen", os.path.join(ROOT, "scripts", "generate_class_reference.py"))
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)

    import classy
    from classy import Class

    res = {"classy_file": classy.__file__, "classy_version": getattr(classy, "__version__", "?")}
    print(f"classy {res['classy_version']} from {res['classy_file']}", flush=True)
    worst = []
    for name, overrides in gen.RECOMB_GRID.items():
        pars = dict(gen.RECFAST_HESW1_BASE)
        for k, v in overrides.items():
            pars["YHe" if k == "Y_He" else k] = v
        c = Class()
        c.set(pars)
        c.compute()
        th = c.get_thermodynamics()
        z_tab = np.asarray(th["z"])
        o = np.argsort(z_tab)
        ref = np.load(os.path.join(ROOT, "reference_data", name, "thermo_recfast_hesw1.npz"))
        z = ref["z"]
        xe = np.interp(np.log1p(z), np.log1p(z_tab[o]), np.asarray(th["x_e"])[o])
        tb = np.interp(np.log1p(z), np.log1p(z_tab[o]), np.asarray(th["Tb [K]"])[o])
        rx = np.abs(xe / ref["x_e"] - 1.0)
        rt = np.abs(tb / ref["Tb"] - 1.0)
        res[name] = {"xe_max": float(rx.max()), "z_at": float(z[np.argmax(rx)]),
                     "Tb_max": float(rt.max()),
                     "xe_windows": {f"{lo}-{hi}": float(rx[(z >= lo) & (z <= hi)].max())
                                    for lo, hi in WINDOWS}}
        print(f"{name:16s} max|dxe/xe| {rx.max():.2e} at z={z[np.argmax(rx)]:.0f}  "
              f"max|dTb/Tb| {rt.max():.2e}", flush=True)
        worst.append(max(rx.max(), rt.max()))
        c.struct_cleanup()
        c.empty()
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as fh:
            json.dump(res, fh, indent=1)
    if max(worst) == 0.0:
        print("ERROR every difference is exactly 0: this classy is the build that made "
              "the references, not an independent CLASS")
        sys.exit(2)
    if max(worst) > BAR:
        print(f"ERROR stock vs reference {max(worst):.2e} > {BAR:.0e}")
        sys.exit(1)
    print(f"PASS: all {len(worst)} cosmologies within {BAR:.0e} (worst {max(worst):.2e})")


if __name__ == "__main__":
    main()
