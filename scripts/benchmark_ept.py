"""Benchmark 3: clax-pt (EPT) one-loop computation time.

Times the one-loop EFT power spectrum computation separately from
the Boltzmann solver, to isolate the PT contribution.

Usage:
    python scripts/benchmark_ept.py [--n-warmup 1] [--n-repeat 5]
"""
import sys
import time
import argparse

sys.path.insert(0, ".")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from clax.params import CosmoParams, PrecisionParams
from clax.background import background_solve
from clax.thermodynamics import thermodynamics_solve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-warmup", type=int, default=1)
    parser.add_argument("--n-repeat", type=int, default=5)
    args = parser.parse_args()

    platform = jax.devices()[0].platform
    print(f"Platform: {platform} ({jax.devices()})")
    print(f"JAX version: {jax.__version__}")
    print()

    # Check if ept module exists
    try:
        from clax.ept import compute_ept, EPTComponents
    except ImportError:
        print("ERROR: clax.ept module not found. Is clax-pt installed?")
        sys.exit(1)

    params = CosmoParams()
    prec = PrecisionParams(
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

    # --- Step 1: Boltzmann solve (needed for P_lin input to EPT) ---
    print("Step 1: Boltzmann solve for P_lin(k)...")
    t0 = time.time()
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)
    jax.block_until_ready(th.tau_star)
    t_boltzmann_compile = time.time() - t0
    print(f"  BG + TH (compile): {t_boltzmann_compile:.2f}s")

    # Get linear P(k) on the EPT k-grid
    # EPT uses its own k-grid (FFTLog grid, typically 256 points)
    import clax
    t0 = time.time()
    # Use compute_pk_table for the linear P(k) that EPT needs
    pk_table = clax.compute_pk_table(params, prec, z=0.38,
                                      k_eval=np.logspace(-3, 0, 100))
    jax.block_until_ready(pk_table.pk_grid)
    t_pk = time.time() - t0
    print(f"  P(k) table (compile): {t_pk:.2f}s")
    print()

    # --- Step 2: EPT one-loop computation ---
    print("Step 2: EPT one-loop computation...")

    # Bias parameters (BOSS-like)
    z = 0.38
    b1 = 2.0
    b2 = 0.0
    bs = 0.0
    b3nl = 0.0
    alpha0 = 0.0
    alpha2 = 0.0
    alpha4 = 0.0
    ctilde = 0.0
    alphashot0 = 0.0
    alphashot2 = 0.0
    PshotP = 0.0
    bphi = 0.0
    b4 = 500.0

    # Warmup
    for i in range(args.n_warmup):
        t0 = time.time()
        ept = compute_ept(
            params, prec, bg, th,
            z=z, b1=b1, b2=b2, bs=bs, b3nl=b3nl,
            alpha0=alpha0, alpha2=alpha2, alpha4=alpha4,
            ctilde=ctilde, alphashot0=alphashot0,
            alphashot2=alphashot2, PshotP=PshotP,
            bphi=bphi, b4=b4,
        )
        # Force evaluation
        jax.block_until_ready(ept.pk_mm_real)
        t_warmup = time.time() - t0
        print(f"  Warmup {i+1}: {t_warmup:.2f}s")

    # Timed runs
    ept_times = []
    for i in range(args.n_repeat):
        # Slightly perturb to avoid result caching
        b1_i = b1 + np.random.uniform(-0.01, 0.01)
        t0 = time.time()
        ept = compute_ept(
            params, prec, bg, th,
            z=z, b1=b1_i, b2=b2, bs=bs, b3nl=b3nl,
            alpha0=alpha0, alpha2=alpha2, alpha4=alpha4,
            ctilde=ctilde, alphashot0=alphashot0,
            alphashot2=alphashot2, PshotP=PshotP,
            bphi=bphi, b4=b4,
        )
        jax.block_until_ready(ept.pk_mm_real)
        ept_times.append(time.time() - t0)

    t_ept = np.median(ept_times)
    print(f"  Median EPT time: {t_ept:.3f}s ({args.n_repeat} repeats)")
    print(f"  All times: {[f'{t:.3f}' for t in ept_times]}")
    print()

    # --- Step 3: EPT gradient ---
    print("Step 3: EPT gradient (d/db1)...")

    def ept_scalar(b1_val):
        ept = compute_ept(
            params, prec, bg, th,
            z=z, b1=b1_val, b2=b2, bs=bs, b3nl=b3nl,
            alpha0=alpha0, alpha2=alpha2, alpha4=alpha4,
            ctilde=ctilde, alphashot0=alphashot0,
            alphashot2=alphashot2, PshotP=PshotP,
            bphi=bphi, b4=b4,
        )
        return jnp.sum(ept.pk_gg_real)

    grad_fn = jax.grad(ept_scalar)

    # Warmup
    t0 = time.time()
    g = grad_fn(b1)
    jax.block_until_ready(g)
    t_grad_compile = time.time() - t0
    print(f"  Gradient compile: {t_grad_compile:.2f}s")

    # Timed
    grad_times = []
    for _ in range(args.n_repeat):
        b1_i = b1 + np.random.uniform(-0.01, 0.01)
        t0 = time.time()
        g = grad_fn(b1_i)
        jax.block_until_ready(g)
        grad_times.append(time.time() - t0)

    t_grad = np.median(grad_times)
    print(f"  Median grad time: {t_grad:.3f}s ({args.n_repeat} repeats)")
    print(f"  Backward/forward ratio: {t_grad/t_ept:.1f}x")
    print()

    # --- Step 4: Output spectra ---
    print("Output spectra (sanity check):")
    k = ept.k_h  # k in h/Mpc
    mask = (k > 0.005) & (k < 0.3)
    print(f"  k range: [{float(k[mask][0]):.4f}, {float(k[mask][-1]):.4f}] h/Mpc")
    print(f"  P_mm(k=0.1): {float(jnp.interp(0.1, k, ept.pk_mm_real)):.2f} (Mpc/h)^3")
    print(f"  P_gg(k=0.1): {float(jnp.interp(0.1, k, ept.pk_gg_real)):.2f} (Mpc/h)^3")

    # --- Summary ---
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Boltzmann (BG+TH):     {t_boltzmann_compile:.2f}s (compile)")
    print(f"  EPT forward (cached):  {t_ept:.3f}s")
    print(f"  EPT gradient (cached): {t_grad:.3f}s")
    print(f"  Backward/forward:      {t_grad/t_ept:.1f}x")
    print()
    print("  For comparison, CLASS-PT one-loop typically takes ~1-5s on CPU.")
    print(f"  EPT on {platform}: {t_ept:.3f}s")


if __name__ == "__main__":
    main()
