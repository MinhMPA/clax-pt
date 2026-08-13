"""Benchmark clax-pt: one-loop EFTofLSS power spectrum cost.

Times the EPT computation isolated from the Boltzmann solve. Reports
forward time, multi-z scaling, and AD-gradient cost. Validates
post-condition accuracy against `reference_data/classpt_z0.38_fullrange.npz`.

Usage:
    python scripts/benchmark_ept.py [--n-warmup 1] [--n-repeat 5]
                                    [--preset {fast,medium,planck}]
                                    [--z-list 0.0,0.38,0.61,1.0]

Prerequisites:
    `clax/ept.py` must be importable (i.e. on the clax-pt branch or any
    branch that ships clax-pt). On upstream/main without clax-pt this
    script will exit gracefully.
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
from clax.perturbations import perturbations_solve_mpk


# Precision presets for the upstream Boltzmann solve. The EPT layer itself
# only depends on a P_lin(k_grid, z) input plus its own k-grid; the upstream
# precision controls how accurate the input P_lin is and how long it takes.
def make_prec(name: str) -> PrecisionParams:
    if name == "fast":
        return PrecisionParams(
            th_n_points=3000,
            pt_k_per_decade=10,
            pt_k_max_cl=0.3,
            pt_l_max_g=17,
            pt_l_max_pol_g=17,
            pt_l_max_ur=17,
            ncdm_q_size=0,
            pt_tau_n_points=1000,
            pt_ode_rtol=1e-3,
            pt_ode_atol=1e-4,
            ode_max_steps=4096,
            pt_ode_solver="rodas5",
        )
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


def time_call(fn, *args, n_warmup=1, n_repeat=5, blocking_field=None, **kwargs):
    """Run ``fn(*args, **kwargs)`` with warmup and report median cached time."""
    for _ in range(n_warmup):
        out = fn(*args, **kwargs)
        if blocking_field is None:
            jax.block_until_ready(out)
        else:
            jax.block_until_ready(getattr(out, blocking_field))
    times = []
    for _ in range(n_repeat):
        t0 = time.time()
        out = fn(*args, **kwargs)
        if blocking_field is None:
            jax.block_until_ready(out)
        else:
            jax.block_until_ready(getattr(out, blocking_field))
        times.append(time.time() - t0)
    return times, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-warmup", type=int, default=1)
    p.add_argument("--n-repeat", type=int, default=5)
    p.add_argument("--preset", choices=("fast", "medium", "planck"), default="fast")
    p.add_argument("--z-list", type=str, default="0.0,0.38,0.61,1.0",
                   help="comma-separated redshifts to evaluate EPT at")
    args = p.parse_args()

    z_list = [float(z) for z in args.z_list.split(",")]
    platform = jax.devices()[0].platform
    print(f"Platform: {platform} ({jax.devices()})")
    print(f"JAX version: {jax.__version__}")
    print(f"Preset: {args.preset}, z list: {z_list}")
    print()

    try:
        from clax.ept import (
            compute_ept_from_clax,
            EPTPrecisionParams,
            pk_mm_real,
            pk_gg_real,
        )
    except ImportError as e:
        print(f"clax.ept not importable: {e}")
        print("This branch ships without clax-pt — skipping EPT benchmark.")
        sys.exit(0)

    # --- Step 1: upstream Boltzmann solve (prerequisite for compute_ept_from_clax) ---
    params = CosmoParams()
    prec = make_prec(args.preset)

    print("Step 1 — upstream Boltzmann (BG + TH + perturbations_solve_mpk)")
    t0 = time.time()
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)
    jax.block_until_ready(th.tau_star)
    t_bg_th = time.time() - t0
    print(f"  BG + TH (compile + cached): {t_bg_th:.2f}s")

    t0 = time.time()
    pt = perturbations_solve_mpk(params, prec, bg, th)
    jax.block_until_ready(pt.delta_m)
    t_pt_compile = time.time() - t0
    print(f"  perturbations_solve_mpk (compile + cached): {t_pt_compile:.2f}s")
    print(f"  pt.delta_m shape: {tuple(pt.delta_m.shape)}")
    print()

    # --- Step 2: EPT forward at z=0.38 (BOSS-like) ---
    ept_prec = EPTPrecisionParams()  # default nmax=256
    print("Step 2 — EPT forward at z=0.38 (default EPTPrecisionParams, nmax=256)")
    times_fwd, ept = time_call(
        compute_ept_from_clax, params, bg, pt, 0.38, ept_prec,
        n_warmup=args.n_warmup, n_repeat=args.n_repeat,
        blocking_field="kh",
    )
    t_fwd = float(np.median(times_fwd))
    print(f"  median: {t_fwd:.3f}s   range: [{min(times_fwd):.3f}, {max(times_fwd):.3f}]")
    print(f"  k-grid: {ept.kh.shape[0]} points, "
          f"k in [{float(ept.kh[0]):.4f}, {float(ept.kh[-1]):.2f}] h/Mpc")
    print(f"  P_mm(k=0.1): {float(jnp.interp(0.1, ept.kh, pk_mm_real(ept))):.2f} (Mpc/h)^3")

    # --- Step 3: multi-z timing (table-style) ---
    print()
    print(f"Step 3 — multi-z scaling, z_list = {z_list}")
    multi_z_times = []
    for z in z_list:
        for _ in range(args.n_warmup):
            ept_z = compute_ept_from_clax(params, bg, pt, z, ept_prec)
            jax.block_until_ready(ept_z.kh)
        rs = []
        for _ in range(args.n_repeat):
            t0 = time.time()
            ept_z = compute_ept_from_clax(params, bg, pt, z, ept_prec)
            jax.block_until_ready(ept_z.kh)
            rs.append(time.time() - t0)
        med = float(np.median(rs))
        multi_z_times.append(med)
        print(f"  z={z:5.2f}: median {med:.3f}s")
    t_total_multi_z = sum(multi_z_times)
    print(f"  Total over {len(z_list)} z values: {t_total_multi_z:.3f}s "
          f"(per-z avg: {t_total_multi_z / len(z_list):.3f}s)")

    # --- Step 4: AD gradient cost ---
    print()
    print("Step 4 — AD gradient: d(sum P_gg) / d(omega_b)")

    def ept_scalar_omega_b(omega_b):
        p2 = params.replace(omega_b=omega_b)
        bg2 = background_solve(p2, prec)
        th2 = thermodynamics_solve(p2, prec, bg2)
        pt2 = perturbations_solve_mpk(p2, prec, bg2, th2)
        ept_obj = compute_ept_from_clax(p2, bg2, pt2, 0.38, ept_prec)
        return jnp.sum(pk_mm_real(ept_obj))

    grad_fn = jax.grad(ept_scalar_omega_b)
    t0 = time.time()
    g0 = grad_fn(jnp.float64(params.omega_b))
    jax.block_until_ready(g0)
    t_grad_compile = time.time() - t0
    print(f"  Gradient compile (first call): {t_grad_compile:.2f}s")

    grad_times = []
    for _ in range(args.n_repeat):
        omega_b_i = jnp.float64(params.omega_b + np.random.uniform(-1e-4, 1e-4))
        t0 = time.time()
        g = grad_fn(omega_b_i)
        jax.block_until_ready(g)
        grad_times.append(time.time() - t0)
    t_grad = float(np.median(grad_times))
    print(f"  median: {t_grad:.3f}s   range: [{min(grad_times):.3f}, {max(grad_times):.3f}]")
    # Forward-pass time for the same scalar (one BG + TH + PT + EPT + reduction)
    full_fwd = t_bg_th + t_pt_compile + t_fwd
    if full_fwd > 0:
        print(f"  Backward/Forward ratio (full pipeline): {t_grad/full_fwd:.2f}x")

    # --- Step 5: accuracy regression vs reference (z=0.38, b1=2, b4=500) ---
    print()
    print("Step 5 — accuracy regression vs reference_data/classpt_z0.38_fullrange.npz")
    ref_path = os.path.join("reference_data", "classpt_z0.38_fullrange.npz")
    if not os.path.exists(ref_path):
        print(f"  Reference file not found: {ref_path}. Skipping.")
    else:
        ref = dict(np.load(ref_path))
        # Recompute at z=0.38 for the comparison
        ept_z038 = compute_ept_from_clax(params, bg, pt, 0.38, ept_prec)
        # Real-space P_mm and P_gg (b1=2 is the reference's setting)
        b1 = 2.0
        pk_mm_clax = np.array(pk_mm_real(ept_z038))
        pk_gg_clax = np.array(pk_gg_real(ept_z038, b1=b1))
        kh = np.array(ept_z038.kh)

        # Reference is on its own k-grid stored as 'k_h' (h/Mpc)
        ref_kh = np.array(ref["k_h"])
        ref_pk_mm = np.array(ref.get("pk_mm_real", ref.get("pk_mm")))
        ref_pk_gg = np.array(ref.get("pk_gg_real", ref.get("pk_gg")))

        # Restrict to k in [0.005, 0.30] h/Mpc (the validated EPT range)
        kmask = (kh >= 0.005) & (kh <= 0.3)
        ref_kmask = (ref_kh >= 0.005) & (ref_kh <= 0.3)
        # Interpolate clax onto reference's k-grid for fair comparison
        pk_mm_at_ref = np.interp(ref_kh[ref_kmask], kh[kmask], pk_mm_clax[kmask])
        pk_gg_at_ref = np.interp(ref_kh[ref_kmask], kh[kmask], pk_gg_clax[kmask])
        rel_mm = np.abs(pk_mm_at_ref / ref_pk_mm[ref_kmask] - 1)
        rel_gg = np.abs(pk_gg_at_ref / ref_pk_gg[ref_kmask] - 1)
        print(f"  P_mm real: max rel err = {rel_mm.max():.3%}, "
              f"mean = {rel_mm.mean():.3%}   (pass: max < 1%)")
        print(f"  P_gg real (b1={b1}): max rel err = {rel_gg.max():.3%}, "
              f"mean = {rel_gg.mean():.3%}   (pass: max < 1%)")

    # --- Summary ---
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Preset:                   {args.preset}")
    print(f"  Platform:                 {platform}")
    print(f"  Upstream BG+TH+PT (compile + cached): "
          f"{t_bg_th + t_pt_compile:.2f}s")
    print(f"  EPT forward at z=0.38 (cached):  {t_fwd:.3f}s")
    print(f"  EPT forward (multi-z avg):       "
          f"{t_total_multi_z / max(len(z_list), 1):.3f}s")
    print(f"  EPT scalar gradient (full pipe): {t_grad:.3f}s")


if __name__ == "__main__":
    main()
