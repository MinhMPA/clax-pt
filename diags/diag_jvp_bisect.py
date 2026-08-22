"""Bisect which pipeline stage produces NaN in jvp(h).
  Stage 1: bg.conformal_age            (no th, no perturbation)
  Stage 2: th.z_reio                   (bg + th, no perturbation)
  Stage 3: th.kappa_dot_of_loga.evaluate(0.0)  (bg + th alt)
  Stage 4: full compute_pk             (bg + th + perturbation)

If 1 finite, 2 NaN -> _find_z_reio_jvp bug
If 1 NaN          -> bg jvp via DirectAdjoint bug
"""
import sys, os, time, dataclasses
sys.path.insert(0, '.')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from clax import CosmoParams, PrecisionParams
from clax.background import background_solve
from clax.thermodynamics import thermodynamics_solve

prec = PrecisionParams(
    th_n_points=3000, pt_k_per_decade=10, pt_k_max_cl=0.3,
    pt_l_max_g=17, pt_l_max_pol_g=17, pt_l_max_ur=17,
    ncdm_q_size=0, pt_tau_n_points=1000,
    pt_ode_rtol=1e-5, pt_ode_atol=1e-6,
    ode_max_steps=16384, pt_ode_solver="rodas5",
    ode_adjoint="direct",
)
params = CosmoParams()

def stage1(h):
    p = dataclasses.replace(params, h=h)
    bg = background_solve(p, prec)
    return bg.conformal_age

def stage2(h):
    p = dataclasses.replace(params, h=h)
    bg = background_solve(p, prec)
    th = thermodynamics_solve(p, prec, bg)
    return th.z_reio

def stage3(h):
    p = dataclasses.replace(params, h=h)
    bg = background_solve(p, prec)
    th = thermodynamics_solve(p, prec, bg)
    return th.tau_star

print("=== Stage 1: jvp(bg.conformal_age)(h, 1.0) ===")
t0 = time.time()
primal, tangent = jax.jvp(stage1, (params.h,), (jnp.asarray(1.0),))
primal.block_until_ready()
print(f"  primal={float(primal):.6e}  tangent={float(tangent):.6e}  ({time.time()-t0:.1f}s)")

print("\n=== Stage 2: jvp(th.z_reio)(h, 1.0) ===")
t0 = time.time()
primal, tangent = jax.jvp(stage2, (params.h,), (jnp.asarray(1.0),))
primal.block_until_ready()
print(f"  primal={float(primal):.6e}  tangent={float(tangent):.6e}  ({time.time()-t0:.1f}s)")

print("\n=== Stage 3: jvp(th.tau_star)(h, 1.0) ===")
t0 = time.time()
primal, tangent = jax.jvp(stage3, (params.h,), (jnp.asarray(1.0),))
primal.block_until_ready()
print(f"  primal={float(primal):.6e}  tangent={float(tangent):.6e}  ({time.time()-t0:.1f}s)")
