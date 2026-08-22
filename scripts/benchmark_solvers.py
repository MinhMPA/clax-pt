"""Benchmark 1: Compare ODE solvers for perturbation integration.

Compares Kvaerno5 vs Rodas5 (unbatched) vs Rodas5Batched on the
perturbation ODE. Measures both P(k) accuracy and wall-clock time.

Usage:
    python scripts/benchmark_solvers.py [--n-warmup 1] [--n-repeat 3]
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
from clax.perturbations import perturbations_solve_mpk


def make_prec(solver_name):
    """Create a low-res PrecisionParams with the given solver."""
    return PrecisionParams(
        bg_n_points=200,
        th_n_points=3000,
        pt_k_per_decade=20,
        pt_k_max_cl=0.3,
        pt_l_max_g=17,
        pt_l_max_pol_g=17,
        pt_l_max_ur=17,
        ncdm_q_size=0,
        pt_tau_n_points=2000,
        pt_ode_rtol=1e-4,
        pt_ode_atol=1e-4,
        ode_max_steps=4096,
        pt_ode_solver=solver_name,
    )


def time_perturbations(params, prec, bg, th, n_warmup=1, n_repeat=3):
    """Time perturbation solve with warmup and repeats."""
    # Warmup (JIT compilation)
    for _ in range(n_warmup):
        pt = perturbations_solve_mpk(params, prec, bg, th)
        jax.block_until_ready(pt.delta_m)

    # Timed runs
    times = []
    for _ in range(n_repeat):
        # Use slightly different params to avoid caching the result
        p = CosmoParams(h=0.6736 + np.random.uniform(-0.001, 0.001))
        t0 = time.time()
        pt = perturbations_solve_mpk(p, prec, bg, th)
        jax.block_until_ready(pt.delta_m)
        times.append(time.time() - t0)
    return times, pt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-warmup", type=int, default=1)
    parser.add_argument("--n-repeat", type=int, default=3)
    args = parser.parse_args()

    platform = jax.devices()[0].platform
    print(f"Platform: {platform} ({jax.devices()})")
    print(f"JAX version: {jax.__version__}")
    print()

    params = CosmoParams()

    # Shared background/thermodynamics (solver-independent)
    prec_base = make_prec("kvaerno5")
    print("Computing background + thermodynamics (shared)...")
    bg = background_solve(params, prec_base)
    th = thermodynamics_solve(params, prec_base, bg)
    print(f"  conformal age = {float(bg.conformal_age):.2f}")
    print()

    solvers = ["kvaerno5", "rodas5"]
    # Only test rosenbrock_batched if on GPU (it's designed for GPU batching)
    if platform == "gpu":
        solvers.append("rosenbrock_batched")

    results = {}
    reference_dm = None

    for solver in solvers:
        print(f"--- {solver} ---")
        prec = make_prec(solver)
        import math
        n_k = int(math.log10(prec.pt_k_max_cl / prec.pt_k_min) * prec.pt_k_per_decade)
        print(f"  n_k={n_k}, rtol={prec.pt_ode_rtol}, atol={prec.pt_ode_atol}")

        try:
            times, pt = time_perturbations(
                params, prec, bg, th,
                n_warmup=args.n_warmup,
                n_repeat=args.n_repeat,
            )
            dm = pt.delta_m[:, -1]  # delta_m at z=0

            median_t = np.median(times)
            print(f"  Time (median of {args.n_repeat}): {median_t:.2f}s")
            print(f"  delta_m shape: {dm.shape}")

            if reference_dm is None:
                reference_dm = dm
                print(f"  (reference for accuracy comparison)")
            else:
                mask = jnp.abs(reference_dm) > 1e-10
                rel_err = jnp.abs(dm[mask] / reference_dm[mask] - 1)
                print(f"  vs kvaerno5: max={float(jnp.max(rel_err)):.4%}, "
                      f"mean={float(jnp.mean(rel_err)):.4%}")

            results[solver] = {
                'times': times,
                'median': median_t,
                'dm': np.array(dm),
            }

        except Exception as e:
            print(f"  FAILED: {e}")
            results[solver] = {'error': str(e)}

        print()

    # Summary table
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  {'Solver':<25s} {'Median (s)':>10s} {'Speedup':>10s}")
    ref_time = results.get("kvaerno5", {}).get("median", None)
    for solver in solvers:
        r = results.get(solver, {})
        if "error" in r:
            print(f"  {solver:<25s} {'FAILED':>10s}")
        elif ref_time:
            speedup = ref_time / r["median"]
            print(f"  {solver:<25s} {r['median']:10.2f} {speedup:9.2f}x")


if __name__ == "__main__":
    main()
