"""Verify the in-tree fix: AD on patched compute_pk should match FD <1%."""
import sys, os, time, dataclasses
sys.path.insert(0, '.')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from clax import CosmoParams, PrecisionParams, compute_pk

prec = PrecisionParams(
    th_n_points=3000, pt_k_per_decade=10, pt_k_max_cl=0.3,
    pt_l_max_g=17, pt_l_max_pol_g=17, pt_l_max_ur=17,
    ncdm_q_size=0, pt_tau_n_points=1000,
    pt_ode_rtol=1e-3, pt_ode_atol=1e-4,
    ode_max_steps=4096, pt_ode_solver="rodas5",
)
params = CosmoParams()
K_TARGET = 0.1
PARAM_NAMES = ["h", "omega_b", "omega_cdm", "ln10A_s", "n_s"]
FD_STEPS = {"h": 1e-3, "omega_b": 1e-5, "omega_cdm": 1e-3, "ln10A_s": 1e-3, "n_s": 1e-3}

print("=== AD on patched compute_pk (in-tree exact-RHS Taylor correction) ===")
t0 = time.time()
g = jax.grad(lambda p: compute_pk(p, prec, k=K_TARGET))(params)
g.h.block_until_ready()
print(f"  AD: {time.time()-t0:.1f}s")
ad = {n: float(getattr(g, n)) for n in PARAM_NAMES}

print("\n=== FD ===")
t0 = time.time()
def fd_grad(name, eps):
    val = getattr(params, name)
    p_plus = dataclasses.replace(params, **{name: val + eps})
    p_minus = dataclasses.replace(params, **{name: val - eps})
    return float(compute_pk(p_plus, prec, k=K_TARGET) - compute_pk(p_minus, prec, k=K_TARGET)) / (2 * eps)
fd = {n: fd_grad(n, FD_STEPS[n]) for n in PARAM_NAMES}
print(f"  FD: {time.time()-t0:.1f}s")

print("\n  param         AD                FD            |AD/FD-1|")
fail = 0
for n in PARAM_NAMES:
    if abs(fd[n]) > 1e-30:
        r = abs(ad[n] / fd[n] - 1)
        flag = "" if r < 0.01 else " ← FAIL"
        if r >= 0.01:
            fail += 1
        print(f"  {n:<11s} {ad[n]:14.4e}    {fd[n]:14.4e}   {r:8.4%}{flag}")
    else:
        print(f"  {n:<11s} {ad[n]:14.4e}    {fd[n]:14.4e}   FD~0")
print(f"\nFAIL count (>1%): {fail} / {len(PARAM_NAMES)}")
