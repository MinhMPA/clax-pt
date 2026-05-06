"""Measure jax.grad(bg.conformal_age)/dparam vs FD. If these are 6-11% off,
the residual in compute_pk AD is upstream of the perturbation solve.
"""
import sys, os, time, dataclasses
sys.path.insert(0, '.')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from clax import CosmoParams, PrecisionParams
from clax.background import background_solve

prec = PrecisionParams(
    th_n_points=3000, pt_k_per_decade=10, pt_k_max_cl=0.3,
    pt_l_max_g=17, pt_l_max_pol_g=17, pt_l_max_ur=17,
    ncdm_q_size=0, pt_tau_n_points=1000,
    pt_ode_rtol=1e-3, pt_ode_atol=1e-4,
    ode_max_steps=4096, pt_ode_solver="rodas5",
)
params = CosmoParams()
PARAM_NAMES = ["h", "omega_b", "omega_cdm", "ln10A_s", "n_s"]
FD_STEPS = {"h": 1e-3, "omega_b": 1e-5, "omega_cdm": 1e-3, "ln10A_s": 1e-3, "n_s": 1e-3}

def tau_age(p):
    return background_solve(p, prec).conformal_age

print("Computing jax.grad(bg.conformal_age) vs FD...")
print("  AD compile + run...")
t0 = time.time()
g = jax.grad(tau_age)(params)
g.h.block_until_ready()
print(f"  AD: {time.time()-t0:.1f}s")
ad = {n: float(getattr(g, n)) for n in PARAM_NAMES}

def fd_at(name, eps):
    val = getattr(params, name)
    p_plus = dataclasses.replace(params, **{name: val + eps})
    p_minus = dataclasses.replace(params, **{name: val - eps})
    return float(tau_age(p_plus) - tau_age(p_minus)) / (2 * eps)

print("\n  FD:")
fd = {}
for n in PARAM_NAMES:
    fd[n] = fd_at(n, FD_STEPS[n])

print("\n  param         AD                  FD              |AD/FD-1|")
for n in PARAM_NAMES:
    if abs(fd[n]) > 1e-30:
        r = abs(ad[n] / fd[n] - 1)
        print(f"  {n:<11s} {ad[n]:14.6e}    {fd[n]:14.6e}   {r:8.4%}")
    else:
        print(f"  {n:<11s} {ad[n]:14.6e}    {fd[n]:14.6e}   FD~0")

# Also test FD step-size sensitivity for h (decisive on FD truncation)
print("\n  FD step sensitivity for h:")
for eps in [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 1e-5]:
    fd_h = fd_at("h", eps)
    print(f"    eps={eps:.0e}: dτ/dh = {fd_h:14.6e}")
