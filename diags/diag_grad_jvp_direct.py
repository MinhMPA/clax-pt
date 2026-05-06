"""Forward-mode jvp cross-check on compute_pk using DirectAdjoint.

  RecursiveCheckpointAdjoint blocks forward mode (its checkpointed_while_loop
  is a custom_vjp). Switching to DirectAdjoint uses a standard while_loop
  that supports both jvp and grad. PIDController's nondifferentiable factor
  remains; that's fine because t1 is still stop_gradient'd, so no tangent
  flows through the controller.

  Cross-check pass criterion: jvp(h) == grad(h) to <0.1%.
    Match → adjoint is internally consistent; the residual vs FD is precision floor.
    Mismatch → custom_vjp / adjoint bug somewhere.
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
    ode_adjoint="direct",
)
params = CosmoParams()
K_TARGET = 0.1
PARAM_NAMES = ["h", "omega_b", "omega_cdm", "ln10A_s", "n_s"]

print(f"=== ode_adjoint={prec.ode_adjoint}, rtol={prec.pt_ode_rtol} ===\n")

def f_of_param(name, val):
    p = dataclasses.replace(params, **{name: val})
    return compute_pk(p, prec, k=K_TARGET)

print("=== Forward-mode jax.jvp per param ===")
jvp_results = {}
for n in PARAM_NAMES:
    val = float(getattr(params, n))
    print(f"  {n} (compile + run)...")
    t0 = time.time()
    primal, tangent = jax.jvp(lambda v: f_of_param(n, v),
                              (jnp.asarray(val),),
                              (jnp.asarray(1.0),))
    primal.block_until_ready()
    jvp_results[n] = float(tangent)
    print(f"    primal={float(primal):.6e}  tangent={float(tangent):.6e}  ({time.time()-t0:.1f}s)")

print("\n=== Reverse-mode jax.grad ===")
t0 = time.time()
g = jax.grad(lambda p: compute_pk(p, prec, k=K_TARGET))(params)
g.h.block_until_ready()
print(f"  AD: {time.time()-t0:.1f}s")
grad_results = {n: float(getattr(g, n)) for n in PARAM_NAMES}

print("\n  param         jvp                  grad                |jvp/grad-1|")
for n in PARAM_NAMES:
    j = jvp_results[n]; r = grad_results[n]
    if abs(r) > 1e-30:
        rel = abs(j/r - 1)
        flag = "" if rel < 0.001 else " <- DISAGREE"
        print(f"  {n:<11s} {j:14.6e}      {r:14.6e}      {rel:9.6%}{flag}")
    else:
        print(f"  {n:<11s} {j:14.6e}      {r:14.6e}      grad~0")
