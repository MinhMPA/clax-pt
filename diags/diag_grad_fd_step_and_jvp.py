"""Two parallel discriminating tests for the residual 6-11% AD-FD gap on compute_pk:

  Test A: FD step sensitivity on compute_pk for h
    - bg.conformal_age FD was step-flat; compute_pk runs perturbation ODE at
      rtol=1e-3 so its FD oracle could itself be noisy at the 1-5% level.
    - If FD jitters across eps, FD is unreliable, not AD.

  Test B: Forward-mode jvp on compute_pk for h
    - jvp matches reverse grad -> adjoint is internally consistent (model-bug)
    - jvp differs from grad -> custom_vjp / adjoint bug
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
    pt_ode_rtol=1e-3, pt_ode_atol=1e-4,
    ode_max_steps=4096, pt_ode_solver="rodas5",
)
params = CosmoParams()
K_TARGET = 0.1

print("=== Test A: FD step sensitivity on compute_pk(k=0.1) for h ===")
print("  Each pair = 2 perturbation solves; expect ~1-2 min/eps")

def pk_at_h(h_val):
    p = dataclasses.replace(params, h=h_val)
    return float(compute_pk(p, prec, k=K_TARGET))

# Compile once with a warmup
t0 = time.time()
pk0 = pk_at_h(params.h)
print(f"  warmup pk(h0): {pk0:.6e}  in {time.time()-t0:.1f}s")

print("\n  eps          FD(h)              vs eps=1e-3")
fd_results = {}
for eps in [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]:
    t0 = time.time()
    pk_plus = pk_at_h(params.h + eps)
    pk_minus = pk_at_h(params.h - eps)
    fd_h = (pk_plus - pk_minus) / (2 * eps)
    fd_results[eps] = fd_h
    print(f"  {eps:.0e}      {fd_h:14.6e}      ({time.time()-t0:.1f}s)")

ref = fd_results[1e-3]
print(f"\n  Sensitivity vs eps=1e-3:")
for eps, val in fd_results.items():
    rel = abs(val/ref - 1) if abs(ref) > 1e-30 else 0.0
    print(f"    eps={eps:.0e}: {val:14.6e}  |delta|={rel:.4%}")

print("\n=== Test B: Forward-mode jvp on compute_pk for h ===")
print("  AD compile ~10-15 min on CPU...")

def f_of_h(h_val):
    p = dataclasses.replace(params, h=h_val)
    return compute_pk(p, prec, k=K_TARGET)

t0 = time.time()
primal, tangent = jax.jvp(f_of_h, (params.h,), (jnp.array(1.0),))
primal.block_until_ready()
print(f"  jvp: {time.time()-t0:.1f}s")
print(f"  jvp primal     = {float(primal):.6e}")
print(f"  jvp tangent(h) = {float(tangent):.6e}")
print(f"  FD(h, 1e-3)    = {fd_results[1e-3]:.6e}")
print(f"  |jvp/FD - 1|   = {abs(float(tangent)/fd_results[1e-3] - 1):.4%}")
