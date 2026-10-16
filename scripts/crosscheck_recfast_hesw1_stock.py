"""Cross-check the committed thermo_recfast_hesw1 references against stock CLASS.

The references in reference_data/<name>/thermo_recfast_hesw1.npz (smsharma/clax#48)
were generated from a CLASS-PT build (CLASS v3.3.4 thermodynamics). This script
re-runs generate_class_reference.py's exact parameter set and z grid with whatever
`classy` is importable (meant: stock CLASS 3.3.4, e.g. the PyPI wheel) and prints
max |x_e_stock / x_e_ref - 1| and |T_b_stock / T_b_ref - 1| per cosmology, plus per
z window into an optional JSON file.

Measured Oct 16, 2026 (stock classy 3.3.4 from PyPI): x_e <= 7.1e-6 (largest at
z 200-800), T_b <= 7.7e-7, z 2870-5000 <= 6.7e-12, at all 10 cosmologies.

Usage:  python scripts/crosscheck_recfast_hesw1_stock.py [out.json]
"""
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
WINDOWS = [(200, 800), (800, 1550), (1550, 1650), (1650, 2870), (2870, 5000), (5000, 8000)]


def main():
    spec = importlib.util.spec_from_file_location(
        "gen", os.path.join(ROOT, "scripts", "generate_class_reference.py"))
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)

    import classy
    from classy import Class

    res = {"classy_file": classy.__file__, "classy_version": getattr(classy, "__version__", "?")}
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
        c.struct_cleanup()
        c.empty()
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
