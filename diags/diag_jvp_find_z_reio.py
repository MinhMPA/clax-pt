"""Direct JVP test on _find_z_reio with concrete inputs/tangents.

Bypass the upstream pipeline. If JVP returns NaN here, the rule itself is buggy.
If it returns finite, the NaN comes from something upstream feeding NaN tangents in.
"""
import sys, os
sys.path.insert(0, '.')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from clax import CosmoParams, PrecisionParams
from clax.background import background_solve
from clax.thermodynamics import thermodynamics_solve, _find_z_reio, _tau_reio_for_zreio

prec = PrecisionParams(
    th_n_points=3000, pt_k_per_decade=10, pt_k_max_cl=0.3,
    pt_l_max_g=17, pt_l_max_pol_g=17, pt_l_max_ur=17,
    ncdm_q_size=0, pt_tau_n_points=1000,
    pt_ode_rtol=1e-5, pt_ode_atol=1e-6,
    ode_max_steps=16384, pt_ode_solver="rodas5",
    ode_adjoint="direct",
)
params = CosmoParams()

# Build the inputs to _find_z_reio the way thermodynamics_solve does.
# Easier: just call thermodynamics_solve once and capture the inputs.
# Simpler: rebuild a small toy version.

bg = background_solve(params, prec)
th = thermodynamics_solve(params, prec, bg)
print(f"primal th.z_reio = {float(th.z_reio):.6e}")
print(f"primal params.tau_reio = {float(params.tau_reio):.6e}\n")

# Build inputs to _find_z_reio. Need: target, xe_raw_grid, kd_prefactor, dtau_grid, z_grid, Y_He
# These live inside thermodynamics_solve's body. Reconstruct by running a pared-down version.
# Easier: probe via wrappers that run thermodynamics_solve and project to z_reio,
# then we look at the JVP of that wrapped function w.r.t. each upstream tangent.

import dataclasses
def th_z_reio(h):
    p = dataclasses.replace(params, h=h)
    bg_ = background_solve(p, prec)
    th_ = thermodynamics_solve(p, prec, bg_)
    return th_.z_reio

# Test 1: jvp w.r.t. h
print("=== Test 1: jvp(th.z_reio)(h) ===")
import time
t0 = time.time()
primal, tangent = jax.jvp(th_z_reio, (params.h,), (jnp.asarray(1.0),))
primal.block_until_ready()
print(f"  primal={float(primal):.6e}  tangent={float(tangent):.6e}  ({time.time()-t0:.1f}s)")

# Test 2: same via reverse mode
print("\n=== Test 2: grad(th.z_reio)(h) ===")
t0 = time.time()
g = jax.grad(th_z_reio)(params.h)
g.block_until_ready()
print(f"  grad={float(g):.6e}  ({time.time()-t0:.1f}s)")

# Test 3: FD for ground truth
print("\n=== Test 3: FD ===")
eps = 1e-3
fd = (th_z_reio(params.h + eps) - th_z_reio(params.h - eps)) / (2 * eps)
print(f"  FD={float(fd):.6e}")
