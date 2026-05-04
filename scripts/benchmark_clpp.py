"""Benchmark C_l^phiphi paths and primordial BB.

Times the three nonlinear settings of `compute_cl_pp` side by side
(`"none"`, `"halofit"`, `"ept"`) plus `compute_cl_bb` (primordial BB,
post-PR#19 fine-k path). Reports forward time and accuracy regression
against CLASS reference data.

Usage:
    python scripts/benchmark_clpp.py [--n-warmup 1] [--n-repeat 5]
                                     [--preset {fast,medium,planck}]
                                     [--l-max 2500]

Prerequisites:
    `clax/lensing.py` (always present); `clax/ept.py` for `nonlinear="ept"`
    (skipped automatically on branches without clax-pt).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, ".")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from clax.params import CosmoParams, PrecisionParams
from clax.background import background_solve
from clax.thermodynamics import thermodynamics_solve
from clax.perturbations import perturbations_solve, tensor_perturbations_solve
from clax.lensing import compute_cl_pp
from clax.harmonic import compute_cl_bb


def make_prec(name: str) -> PrecisionParams:
    if name == "fast":
        return PrecisionParams.fit_cl()
    if name == "medium":
        return PrecisionParams(
            th_n_points=10000,
            pt_k_per_decade=20,
            pt_k_max_cl=1.0,
            pt_l_max_g=30,
            pt_l_max_pol_g=30,
            pt_l_max_ur=30,
            ncdm_q_size=5,
            pt_tau_n_points=2000,
            pt_ode_rtol=1e-4,
            pt_ode_atol=1e-6,
            ode_max_steps=16384,
            pt_ode_solver="rodas5",
        )
    if name == "planck":
        return PrecisionParams.planck_cl()
    raise ValueError(f"unknown preset {name!r}")


def time_call(fn, *args, n_warmup=1, n_repeat=5, **kwargs):
    for _ in range(n_warmup):
        out = fn(*args, **kwargs)
        jax.block_until_ready(out)
    times = []
    for _ in range(n_repeat):
        t0 = time.time()
        out = fn(*args, **kwargs)
        jax.block_until_ready(out)
        times.append(time.time() - t0)
    return times, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-warmup", type=int, default=1)
    p.add_argument("--n-repeat", type=int, default=5)
    p.add_argument("--preset", choices=("fast", "medium", "planck"), default="fast")
    p.add_argument("--l-max", type=int, default=2500)
    args = p.parse_args()

    platform = jax.devices()[0].platform
    print(f"Platform: {platform} ({jax.devices()})")
    print(f"JAX version: {jax.__version__}")
    print(f"Preset: {args.preset}, l_max: {args.l_max}")
    print()

    params = CosmoParams()
    prec = make_prec(args.preset)

    # --- Step 1: upstream Boltzmann solve (full perturbations for source_phi_plus_psi) ---
    print("Step 1 — upstream Boltzmann (BG + TH + perturbations_solve)")
    t0 = time.time()
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)
    jax.block_until_ready(th.tau_star)
    t_bg_th = time.time() - t0
    print(f"  BG + TH: {t_bg_th:.2f}s")

    t0 = time.time()
    pt = perturbations_solve(params, prec, bg, th)
    jax.block_until_ready(pt.source_phi_plus_psi)
    t_pt = time.time() - t0
    print(f"  perturbations_solve: {t_pt:.2f}s")
    print(f"  source_phi_plus_psi shape: {tuple(pt.source_phi_plus_psi.shape)}")
    print()

    # --- Step 2: compute_cl_pp at three nonlinear settings ---
    results = {}
    for nl in ("none", "halofit", "ept"):
        try:
            print(f"Step 2.{nl} — compute_cl_pp(nonlinear={nl!r}, l_max={args.l_max})")
            times, cl_pp = time_call(
                compute_cl_pp, pt, params, bg, th, args.l_max,
                nonlinear=nl,
                n_warmup=args.n_warmup, n_repeat=args.n_repeat,
            )
            t_med = float(np.median(times))
            results[nl] = (t_med, np.array(cl_pp))
            print(f"  median: {t_med:.3f}s   range: [{min(times):.3f}, {max(times):.3f}]")
            print(f"  C_l^pp(l=100) = {float(cl_pp[100]):.4e}")
            print(f"  C_l^pp(l=1000) = {float(cl_pp[1000]):.4e}")
        except (ValueError, ImportError) as e:
            # nonlinear="ept" raises on branches without clax-pt
            print(f"  Skipped (nonlinear={nl!r}): {e}")
            results[nl] = None
        print()

    # --- Step 3: ratio of nonlinear/linear at characteristic l ---
    if results.get("none") is not None:
        cl_lin = results["none"][1]
        for nl in ("halofit", "ept"):
            if results.get(nl) is None:
                continue
            cl_nl = results[nl][1]
            print(f"Ratio  C_l^pp[{nl}] / C_l^pp[none]:")
            for l in (100, 500, 1000, 2000):
                if l < len(cl_lin):
                    r = float(cl_nl[l] / cl_lin[l])
                    print(f"  l={l:4d}: {r:.4f}")
            print()

    # --- Step 4: accuracy vs CLASS reference (linear C_l^pp only) ---
    ref_path = os.path.join("reference_data", "lcdm_fiducial", "cls.npz")
    if results.get("none") is not None and os.path.exists(ref_path):
        ref = dict(np.load(ref_path))
        if "pp" in ref:
            cl_lin = results["none"][1]
            ref_pp = np.array(ref["pp"])
            l_max_compare = min(len(cl_lin) - 1, len(ref_pp) - 1, 2500)
            probe = [int(l) for l in (10, 50, 100, 500, 1000, 1500, 2000, 2500)
                     if l <= l_max_compare]
            print("Step 4 — C_l^pp accuracy vs CLASS reference (nonlinear='none'):")
            print(f"  {'l':>5s} | {'clax':>11s} | {'CLASS':>11s} | {'rel diff':>9s}")
            print("  " + "-" * 50)
            for l in probe:
                clax_val = float(cl_lin[l])
                ref_val = float(ref_pp[l])
                if abs(ref_val) > 1e-30:
                    rel = (clax_val - ref_val) / ref_val * 100
                    print(f"  {l:>5d} | {clax_val:>11.4e} | "
                          f"{ref_val:>11.4e} | {rel:>+8.3f}%")
        else:
            print("Step 4 — reference file lacks 'pp' key; skipping accuracy check")
    print()

    # --- Step 5: tensor BB pipeline (post-PR#19 kernel + fine-k) ---
    print("Step 5 — primordial C_l^BB (compute_cl_bb, post-PR#19 path)")
    bb_prec = PrecisionParams(
        pt_l_max_g=10, pt_l_max_pol_g=10, pt_l_max_ur=10,
        pt_k_max_cl=0.1, pt_k_per_decade=10,
        pt_tau_n_points=1000,
        pt_ode_rtol=1e-3, pt_ode_atol=1e-6,
        ode_max_steps=32768,
    )
    bb_params = CosmoParams(r_t=0.1)
    bb_bg = background_solve(bb_params, bb_prec)
    bb_th = thermodynamics_solve(bb_params, bb_prec, bb_bg)
    t0 = time.time()
    tpt = tensor_perturbations_solve(bb_params, bb_prec, bb_bg, bb_th)
    jax.block_until_ready(tpt.source_p)
    t_tensor_solve = time.time() - t0
    print(f"  tensor_perturbations_solve: {t_tensor_solve:.2f}s")

    l_bb = jnp.array([2, 10, 30, 50, 80, 100, 150, 200], dtype=jnp.float64)
    times_bb, cl_bb = time_call(
        compute_cl_bb, tpt, bb_params, bb_bg, l_bb,
        n_warmup=args.n_warmup, n_repeat=args.n_repeat,
        n_k_fine=2000,
    )
    t_bb = float(np.median(times_bb))
    print(f"  compute_cl_bb (cached, n_k_fine=2000): {t_bb:.3f}s")
    bb_ref_path = os.path.join("reference_data", "tensor_r01", "cls_tensor.npz")
    if os.path.exists(bb_ref_path):
        bb_ref = dict(np.load(bb_ref_path))
        ell_ref = np.array(bb_ref["ell"])
        bb_class = np.array(bb_ref["bb"])
        print(f"  {'l':>4s} | {'clax':>11s} | {'CLASS':>11s} | {'ratio':>7s}")
        print("  " + "-" * 45)
        for i, l_val in enumerate([2, 10, 30, 50, 80, 100, 150, 200]):
            j = int(np.argmin(np.abs(ell_ref - l_val)))
            cu = float(cl_bb[i])
            cc = float(bb_class[j])
            ratio = cu / cc if abs(cc) > 1e-40 else float("nan")
            print(f"  {l_val:>4d} | {cu:>11.4e} | {cc:>11.4e} | {ratio:>7.4f}")

    # --- Summary ---
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Preset: {args.preset}")
    print(f"  Platform: {platform}")
    print(f"  Upstream solve (BG+TH+PT): {t_bg_th + t_pt:.2f}s")
    for nl in ("none", "halofit", "ept"):
        if results.get(nl) is None:
            print(f"  compute_cl_pp({nl!r}, l_max={args.l_max}): skipped")
        else:
            print(f"  compute_cl_pp({nl!r}, l_max={args.l_max}): "
                  f"{results[nl][0]:.3f}s")
    print(f"  Tensor solve + compute_cl_bb (n_k_fine=2000): "
          f"{t_tensor_solve + t_bb:.2f}s")


if __name__ == "__main__":
    main()
