"""Test the hypothesis that stop_gradient(bg.conformal_age) at
clax/perturbations.py:2269 is the source of the AD-vs-FD mismatch for
h, omega_b, omega_cdm gradients of P(k).

Steps:
  1. Compute jax.grad(compute_pk)(params) for h, omega_b, omega_cdm with
     the current code (stop_gradient in place).
  2. Centered FD with the same step sizes the benchmark uses.
  3. Monkey-patch out the stop_gradient (replace with identity), recompute AD,
     compare to FD again.
  4. Also run jax.jvp (forward mode) for cross-validation.
"""
import sys, os, time, dataclasses
sys.path.insert(0, '.')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from clax import CosmoParams, PrecisionParams, compute_pk
import clax.perturbations as perturbations_mod


def make_prec():
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


prec = make_prec()
params = CosmoParams()
K_TARGET = 0.1
PARAM_NAMES = ["h", "omega_b", "omega_cdm", "ln10A_s", "n_s"]
FD_STEPS = {"h": 1e-3, "omega_b": 1e-5, "omega_cdm": 1e-3, "ln10A_s": 1e-3, "n_s": 1e-3}


def pk_fn(p):
    return compute_pk(p, prec, k=K_TARGET)


def ad_grads(label):
    print(f"\n--- {label} ---")
    t0 = time.time()
    g = jax.grad(pk_fn)(params)
    g.h.block_until_ready()
    print(f"  AD (compile + run): {time.time()-t0:.1f}s")
    return {n: float(getattr(g, n)) for n in PARAM_NAMES}


def fd_grad_one(name, eps):
    val = getattr(params, name)
    p_plus = dataclasses.replace(params, **{name: val + eps})
    p_minus = dataclasses.replace(params, **{name: val - eps})
    return float(pk_fn(p_plus) - pk_fn(p_minus)) / (2 * eps)


def fd_grads():
    print("\n--- FD gradients (centered) ---")
    t0 = time.time()
    out = {n: fd_grad_one(n, FD_STEPS[n]) for n in PARAM_NAMES}
    print(f"  FD (10 evals): {time.time()-t0:.1f}s")
    return out


def report(label, ad, fd):
    print(f"\n  {'param':<12s} {'AD':>14s} {'FD':>14s} {'|AD/FD-1|':>10s}")
    for n in PARAM_NAMES:
        if abs(fd[n]) > 1e-30:
            r = abs(ad[n] / fd[n] - 1)
            print(f"  {n:<12s} {ad[n]:14.4e} {fd[n]:14.4e} {r:10.4%}")
        else:
            print(f"  {n:<12s} {ad[n]:14.4e} {fd[n]:14.4e}  (FD~0)")


# Stage 1: current code (stop_gradient in place)
ad_before = ad_grads("AD with stop_gradient(bg.conformal_age) [current code]")
fd = fd_grads()
report("CURRENT CODE", ad_before, fd)

# Stage 2: monkey-patch jax.lax.stop_gradient inside perturbations module
print("\n=== Patching out stop_gradient in clax.perturbations ===")
orig_sg = jax.lax.stop_gradient
def identity(x):
    return x
# Patch jax.lax in the perturbations module's namespace
perturbations_mod.jax.lax.stop_gradient = identity
# Also need to clear JIT cache for _matter_delta_m_single_k_impl
jax.clear_caches()

ad_after = ad_grads("AD with stop_gradient -> identity (patched)")
report("PATCHED (no stop_gradient)", ad_after, fd)

# Restore
perturbations_mod.jax.lax.stop_gradient = orig_sg
jax.clear_caches()

# Stage 3: jvp (forward mode) for cross-check on h
print("\n=== Forward-mode jvp on h (current code) ===")
t0 = time.time()
def f_of_h(h_val):
    p = dataclasses.replace(params, h=h_val)
    return compute_pk(p, prec, k=K_TARGET)
primal, tangent = jax.jvp(f_of_h, (params.h,), (jnp.array(1.0),))
primal.block_until_ready()
print(f"  jvp: {time.time()-t0:.1f}s")
print(f"  jvp(h)         = {float(tangent):.4e}")
print(f"  grad(h) [stop] = {ad_before['h']:.4e}")
print(f"  grad(h) [patched] = {ad_after['h']:.4e}")
print(f"  FD(h)          = {fd['h']:.4e}")
