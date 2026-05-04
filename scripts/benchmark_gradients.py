"""Benchmark 2: AD gradient cost vs finite-difference gradient cost.

Compares wall-clock time for computing gradients of P(k) and C_l
via reverse-mode AD vs centered finite differences.

Usage:
    python scripts/benchmark_gradients.py [--n-warmup 1] [--n-repeat 3]
"""
import sys
import time
import argparse
import functools

sys.path.insert(0, ".")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from clax import CosmoParams, PrecisionParams, compute_pk


# Parameters to differentiate w.r.t.
GRAD_PARAMS = ["h", "omega_b", "omega_cdm", "ln10A_s", "n_s"]
FD_STEPS = {
    "h": 1e-3,
    "omega_b": 1e-5,
    "omega_cdm": 1e-3,
    "ln10A_s": 1e-3,
    "n_s": 1e-3,
}


def make_prec():
    """Low-res precision for gradient benchmarking."""
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


def pk_scalar(params, prec, k=0.1):
    """Scalar objective: P(k) at one k-value."""
    return compute_pk(params, prec, k=k)


def time_ad_gradient(params, prec, k, n_warmup=1, n_repeat=3):
    """Time reverse-mode AD gradient of P(k) w.r.t. all CosmoParams."""
    grad_fn = jax.grad(lambda p: pk_scalar(p, prec, k))

    # Warmup
    for _ in range(n_warmup):
        g = grad_fn(params)
        jax.block_until_ready(g.h)

    # Timed runs
    times = []
    for _ in range(n_repeat):
        t0 = time.time()
        g = grad_fn(params)
        jax.block_until_ready(g.h)
        times.append(time.time() - t0)

    return times, g


def time_fd_gradient(params, prec, k, n_warmup=1, n_repeat=3):
    """Time finite-difference gradient of P(k) w.r.t. selected params."""
    from dataclasses import fields

    # Warmup
    for _ in range(n_warmup):
        _ = pk_scalar(params, prec, k)

    d = len(GRAD_PARAMS)
    times = []
    grads = {}

    for _ in range(n_repeat):
        t0 = time.time()
        for pname in GRAD_PARAMS:
            eps = FD_STEPS[pname]
            val = getattr(params, pname)
            p_plus = params.replace(**{pname: val + eps})
            p_minus = params.replace(**{pname: val - eps})
            pk_plus = pk_scalar(p_plus, prec, k)
            pk_minus = pk_scalar(p_minus, prec, k)
            grads[pname] = float(pk_plus - pk_minus) / (2 * eps)
        times.append(time.time() - t0)

    return times, grads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-warmup", type=int, default=1)
    parser.add_argument("--n-repeat", type=int, default=3)
    parser.add_argument("--k", type=float, default=0.1,
                        help="k value in h/Mpc for P(k) gradient")
    args = parser.parse_args()

    platform = jax.devices()[0].platform
    print(f"Platform: {platform} ({jax.devices()})")
    print(f"JAX version: {jax.__version__}")
    print()

    params = CosmoParams()
    prec = make_prec()
    d = len(GRAD_PARAMS)

    print(f"Objective: P(k={args.k} h/Mpc)")
    print(f"Parameters: {GRAD_PARAMS} (d={d})")
    print(f"Solver: {prec.pt_ode_solver}")
    print()

    # --- Forward evaluation baseline ---
    print("Forward evaluation:")
    t0 = time.time()
    pk_val = pk_scalar(params, prec, args.k)
    jax.block_until_ready(pk_val)
    t_fwd_compile = time.time() - t0

    t0 = time.time()
    pk_val = pk_scalar(params, prec, args.k)
    jax.block_until_ready(pk_val)
    t_fwd = time.time() - t0
    print(f"  P(k) = {float(pk_val):.4e}")
    print(f"  First call: {t_fwd_compile:.2f}s (compile + run)")
    print(f"  Cached:     {t_fwd:.2f}s")
    print()

    # --- AD gradient ---
    print(f"AD gradient (reverse-mode, jax.grad):")
    ad_times, ad_grad = time_ad_gradient(
        params, prec, args.k,
        n_warmup=args.n_warmup, n_repeat=args.n_repeat,
    )
    t_ad = np.median(ad_times)
    print(f"  Median time: {t_ad:.2f}s ({args.n_repeat} repeats)")
    print(f"  Backward/forward ratio: {t_ad/t_fwd:.1f}x")
    for pname in GRAD_PARAMS:
        val = float(getattr(ad_grad, pname))
        print(f"    dP/d({pname}) = {val:.4e}")
    print()

    # --- FD gradient ---
    print(f"FD gradient (centered, 2*{d}={2*d} evaluations):")
    fd_times, fd_grads = time_fd_gradient(
        params, prec, args.k,
        n_warmup=args.n_warmup, n_repeat=args.n_repeat,
    )
    t_fd = np.median(fd_times)
    print(f"  Median time: {t_fd:.2f}s ({args.n_repeat} repeats)")
    print(f"  Per-evaluation: {t_fd/(2*d):.2f}s")
    for pname in GRAD_PARAMS:
        print(f"    dP/d({pname}) = {fd_grads[pname]:.4e}")
    print()

    # --- Comparison ---
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"  Forward:  {t_fwd:.2f}s")
    print(f"  AD grad:  {t_ad:.2f}s  (1 backward pass)")
    print(f"  FD grad:  {t_fd:.2f}s  ({2*d} forward passes)")
    print(f"  Speedup:  {t_fd/t_ad:.1f}x  (AD vs FD)")
    print(f"  Effective backward cost: {(t_ad - t_fwd)/t_fwd:.1f}x forward")
    print()

    # --- AD vs FD agreement ---
    print("AD vs FD agreement:")
    print(f"  {'Param':<12s} {'AD':>12s} {'FD':>12s} {'|AD/FD-1|':>10s}")
    for pname in GRAD_PARAMS:
        ad_val = float(getattr(ad_grad, pname))
        fd_val = fd_grads[pname]
        if abs(fd_val) > 1e-30:
            rel = abs(ad_val / fd_val - 1)
            print(f"  {pname:<12s} {ad_val:12.4e} {fd_val:12.4e} {rel:10.4%}")
        else:
            print(f"  {pname:<12s} {ad_val:12.4e} {fd_val:12.4e}  (FD~0)")

    # Scaling projection
    print()
    print("Projected scaling:")
    c_bwd = (t_ad - t_fwd) / t_fwd if t_fwd > 0 else 3.0
    for d_ext in [6, 10, 15, 20]:
        t_fd_proj = 2 * d_ext * t_fwd
        t_ad_proj = t_fwd + c_bwd * t_fwd  # constant regardless of d
        speedup = t_fd_proj / t_ad_proj if t_ad_proj > 0 else float('inf')
        print(f"  d={d_ext:2d}: FD={t_fd_proj:6.1f}s, AD={t_ad_proj:6.1f}s, "
              f"speedup={speedup:.1f}x")


if __name__ == "__main__":
    main()
