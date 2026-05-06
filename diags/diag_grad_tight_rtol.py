"""Decisive test: at pt_ode_rtol=1e-5 (vs benchmark's 1e-3), verify
  (a) FD step sweep collapses to <0.1% spread → FD is now a reliable oracle
  (b) AD+Taylor (in-tree fix) agrees with stable FD to <1% → fix is correct

If (b) is satisfied: the 6-11% residual at rtol=1e-3 was precision-floor
artifact, the Taylor fix is mathematically right, ship it.

If (b) shows ~3% residual at tight rtol: real bias, need jvp isolation
on the perturbation step.
"""
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
    pt_ode_rtol=1e-5, pt_ode_atol=1e-6,
    ode_max_steps=16384, pt_ode_solver="rodas5",
)
params = CosmoParams()
K_TARGET = 0.1
PARAM_NAMES = ["h", "omega_b", "omega_cdm", "ln10A_s", "n_s"]
FD_STEPS = {"h": 1e-3, "omega_b": 1e-5, "omega_cdm": 1e-3, "ln10A_s": 1e-3, "n_s": 1e-3}

print(f"=== Tight rtol={prec.pt_ode_rtol}, max_steps={prec.ode_max_steps} ===\n")

# ------- Test A: FD step sweep on h at tight rtol -------
print("=== Test A: FD step sensitivity on compute_pk(k=0.1) for h ===")
def pk_at_h(h_val):
    p = dataclasses.replace(params, h=h_val)
    return float(compute_pk(p, prec, k=K_TARGET))

t0 = time.time()
pk0 = pk_at_h(params.h)
print(f"  warmup pk(h0)={pk0:.6e} in {time.time()-t0:.1f}s\n")

print("  eps          FD(h)              vs eps=1e-3")
fd_h_results = {}
for eps in [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]:
    t0 = time.time()
    pk_plus = pk_at_h(params.h + eps)
    pk_minus = pk_at_h(params.h - eps)
    fd_h = (pk_plus - pk_minus) / (2 * eps)
    fd_h_results[eps] = fd_h
    print(f"  {eps:.0e}      {fd_h:14.6e}      ({time.time()-t0:.1f}s)")

ref = fd_h_results[1e-3]
print(f"\n  Sensitivity vs eps=1e-3:")
fd_h_spread = []
for eps, val in fd_h_results.items():
    rel = abs(val/ref - 1) if abs(ref) > 1e-30 else 0.0
    fd_h_spread.append(rel)
    print(f"    eps={eps:.0e}: {val:14.6e}  |delta|={rel:.4%}")
print(f"  Max spread: {max(fd_h_spread):.4%}")

# ------- Test B: full AD vs FD comparison at tight rtol -------
print("\n=== Test B: AD+Taylor vs FD at tight rtol (all 5 params) ===")
print("  AD compile+run...")
t0 = time.time()
g = jax.grad(lambda p: compute_pk(p, prec, k=K_TARGET))(params)
g.h.block_until_ready()
print(f"  AD: {time.time()-t0:.1f}s")
ad = {n: float(getattr(g, n)) for n in PARAM_NAMES}

print("  FD per param...")
t0 = time.time()
def fd_grad(name, eps):
    val = getattr(params, name)
    p_plus = dataclasses.replace(params, **{name: val + eps})
    p_minus = dataclasses.replace(params, **{name: val - eps})
    return float(compute_pk(p_plus, prec, k=K_TARGET) - compute_pk(p_minus, prec, k=K_TARGET)) / (2 * eps)
fd = {n: fd_grad(n, FD_STEPS[n]) for n in PARAM_NAMES}
print(f"  FD: {time.time()-t0:.1f}s\n")

print(f"  param         AD                FD            |AD/FD-1|")
fail = 0
for n in PARAM_NAMES:
    if abs(fd[n]) > 1e-30:
        r = abs(ad[n] / fd[n] - 1)
        flag = "" if r < 0.01 else " <- FAIL"
        if r >= 0.01:
            fail += 1
        print(f"  {n:<11s} {ad[n]:14.4e}    {fd[n]:14.4e}   {r:8.4%}{flag}")
    else:
        print(f"  {n:<11s} {ad[n]:14.4e}    {fd[n]:14.4e}   FD~0")
print(f"\nFAIL count (>1%): {fail} / {len(PARAM_NAMES)}")
