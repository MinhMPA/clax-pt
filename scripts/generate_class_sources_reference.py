#!/usr/bin/env python
"""Generate CLASS source-function references at fixed k for tests/test_sources_vs_class.py.

Requires the CLASS Python wrapper (classy 3.3.4). It is NOT installed in the
clax env; on igpu use the micromamba env `carpile`:

    /home/n2minh/micromamba/envs/carpile/bin/python scripts/generate_class_sources_reference.py

CPU only, about 5 s per cosmology. Writes reference_data/<name>/sources_k<k>.npz for
k = 0.15, 0.15*sqrt(2) = 0.212132, 0.30 /Mpc (a geometric triple, so that the three
values are exact nodes of a 3-mode log grid in clax: PrecisionParams(pt_k_min=0.15,
pt_k_max_cl=0.3, pt_k_per_decade=10)); CLASS is forced to solve at exactly these k
with `k_output_values`.

Cosmologies (tests/conftest.py COSMOLOGY_GRID_LCDM offsets on the clax fiducial):
  massless_recfast   CosmoParams(m_ncdm=0.0): N_ur 2.0328 + one massless ncdm species
                     (the cosmology of reference_data/cls_massless_recfast.npz)
  lcdm_fiducial, h_high, omega_b_high, omega_cdm_low, ns_high
                     default 0.06 eV neutrino (CosmoParams() + offset)
All runs: RECFAST (clax's recombination code), Y_He fixed to clax's 0.2454006 (the
cls.npz files in the same directories were generated with CLASS's default HyRec and
BBN Y_He; the sources here are meant for code-level comparison at matched physics),
CLASS default precision (k sampling 20 points per acoustic period, which the Oct 12
2026 precision runs showed converged to 0.005%).

Each npz holds, on CLASS's own source time sampling (about 600 points, dense around
recombination): tau [Mpc], t0 (= clax source_T0), t1 (= source_T1), t2 (= g*P with
P = Pi/8, i.e. clax source_T2 / 8), float32; scalars k, tau_rec (visibility peak),
tau_free_streaming (first tau after the peak with tau_c/tau = 1/(kappa' tau) >=
radiation_streaming_trigger_tau_c_over_tau = 5, linearly interpolated between table
points; `tau_free_streaming_table` is CLASS's own table-point value), z_rec, tau_0,
and `class_params` (JSON) / `classy_version` for provenance.
"""
import json
import os
import sys

import numpy as np

try:
    import classy
except ImportError:
    sys.exit("classy not importable; use the carpile env (see the header)")

REFERENCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reference_data")
K_VALUES = [0.15, 0.15 * np.sqrt(2.0), 0.30]
RSA_TAU_C_OVER_TAU = 5.0   # CLASS precision default radiation_streaming_trigger_tau_c_over_tau

BASE = {
    'h': 0.6736, 'omega_b': 0.02237, 'omega_cdm': 0.1200, 'A_s': 2.1e-9, 'n_s': 0.9649,
    'tau_reio': 0.0544, 'YHe': 0.2454006, 'N_ur': 2.0328, 'N_ncdm': 1, 'T_ncdm': 0.71611,
    'm_ncdm': 0.06, 'recombination': 'recfast', 'output': 'tCl pCl', 'l_max_scalars': 2500,
    'k_output_values': ",".join(repr(float(k)) for k in K_VALUES),
}
CASES = {
    'massless_recfast': {'m_ncdm': 0.0},
    'lcdm_fiducial': {},
    'h_high': {'h': 0.6736 * 1.10},
    'omega_b_high': {'omega_b': 0.02237 * 1.20},
    'omega_cdm_low': {'omega_cdm': 0.1200 * 0.80},
    'ns_high': {'n_s': 0.9649 * 1.05},
}


def tau_free_streaming(th):
    tau, kd, g = (np.asarray(th[k]) for k in ('conf. time [Mpc]', "kappa' [Mpc^-1]", 'g [Mpc^-1]'))
    o = np.argsort(tau)
    tau, kd, g = tau[o], kd[o], g[o]
    ipk = int(np.argmax(g))
    f = 1.0 / (kd * tau)
    i = int(np.argmax((np.arange(len(tau)) > ipk) & (f >= RSA_TAU_C_OVER_TAU)))
    assert i > ipk, "no tau_c/tau crossing after the visibility peak (argmax of an all-False mask is 0)"
    w = (RSA_TAU_C_OVER_TAU - f[i - 1]) / (f[i] - f[i - 1])
    return float(tau[i - 1] + w * (tau[i] - tau[i - 1])), float(tau[i]), float(tau[ipk])


def main():
    names = sys.argv[1:] or list(CASES)
    for name in names:
        p = dict(BASE)
        p.update(CASES[name])
        c = classy.Class()
        c.set(p)
        c.compute()
        src, k_arr, tau_arr = c.get_sources()
        th = c.get_thermodynamics()
        der = c.get_current_derived_parameters(['z_rec', 'tau_rec', 'conformal_age'])
        tau_fs, tau_fs_table, tau_peak = tau_free_streaming(th)
        shape = np.asarray(src['t0']).shape
        transpose = shape == (len(tau_arr), len(k_arr))
        outdir = os.path.join(REFERENCE_DIR, name)
        os.makedirs(outdir, exist_ok=True)
        for k in K_VALUES:
            ik = int(np.argmin(np.abs(k_arr - k)))
            assert abs(k_arr[ik] / k - 1) < 1e-6, (name, k, k_arr[ik])
            arrs = {}
            for key in ('t0', 't1', 't2'):
                a = np.asarray(src[key])
                arrs[key] = (a[:, ik] if transpose else a[ik, :]).astype(np.float32)
            path = os.path.join(outdir, f"sources_k{k:.4g}.npz")
            np.savez(path, k=np.float64(k_arr[ik]), tau=np.asarray(tau_arr, dtype=np.float32),
                     tau_rec=np.float64(der['tau_rec']), tau_rec_table=np.float64(tau_peak),
                     tau_free_streaming=np.float64(tau_fs), tau_free_streaming_table=np.float64(tau_fs_table),
                     z_rec=np.float64(der['z_rec']), tau_0=np.float64(der['conformal_age']),
                     rsa_trigger_tau_c_over_tau=np.float64(RSA_TAU_C_OVER_TAU),
                     class_params=json.dumps(p), classy_version=classy.__version__, **arrs)
            print(f"{name:18s} k={k_arr[ik]:.6f}: {len(tau_arr)} tau points, |t0|max {np.abs(arrs['t0']).max():.3e}, "
                  f"tau_rec {der['tau_rec']:.3f}, tau_fs {tau_fs:.3f} -> {os.path.relpath(path, REFERENCE_DIR)}", flush=True)
        c.struct_cleanup()
        c.empty()


if __name__ == '__main__':
    main()
