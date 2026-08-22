"""Test the Taylor-correction approach to recover the dτ_end gradient
without removing stop_gradient (which breaks RecursiveCheckpointAdjoint+
PIDController):

  delta_m(τ_traced) ≈ delta_m(τ_frozen) + RHS_eff · (τ_traced − τ_frozen)

where RHS_eff is the time-derivative of δ_m at the endpoint.

Strategy: monkey-patch _matter_delta_m_single_k_impl to compute the
extra Taylor term, then compare AD to FD.
"""
import sys, os, time, dataclasses
sys.path.insert(0, '.')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from clax import CosmoParams, PrecisionParams
import clax.perturbations as perturbations_mod
from clax.background import background_solve
from clax.thermodynamics import thermodynamics_solve
from clax.primordial import primordial_scalar_pk

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


# Build a "Taylor-corrected" compute_pk: take the existing AD pipeline (which
# stops gradient through tau_end), evaluate δ_m at τ_frozen, then add a
# linear correction (dδ_m/dτ at τ_end) · (τ_traced − τ_frozen).
#
# We need dδ_m/dτ at the endpoint. That's the time derivative of the matter
# density perturbation at z=0. In the synchronous gauge, δ_m = (Ω_b·δ_b +
# Ω_cdm·δ_cdm) / (Ω_b + Ω_cdm). Its time derivative at z=0:
#   (Ω_b·δ_b' + Ω_cdm·δ_cdm') / (Ω_b + Ω_cdm)
#
# In the matter-dominated era δ ∝ a so δ' = aH·δ. Even with dark energy
# the relation δ' ≈ f·aH·δ holds at z=0, where f = d ln D / d ln a is the
# growth rate. f≈0.527 at z=0 in Planck LCDM.
#
# So a *physically motivated* Taylor correction:
#   δ_m(τ_traced) ≈ δ_m(τ_frozen) + f·aH·δ_m · (τ_traced − τ_frozen)
#
# Both f and aH at z=0 depend on params traceably, so the correction
# carries the full d/dparam derivative through.

from clax.perturbations import _matter_delta_m_single_k_impl, _resolve_scalar_pid_config


def compute_pk_taylor(params, prec, k=0.1):
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)
    pid_config = _resolve_scalar_pid_config()
    delta_m_frozen = _matter_delta_m_single_k_impl(params, prec, bg, th, pid_config, k)

    # Taylor correction: delta_m(tau_traced) ≈ delta_m(tau_frozen) + (dδ/dτ) (τ_traced − τ_frozen)
    # In the matter+ΛCDM regime, dδ/dτ ≈ f·aH·δ at the integration endpoint.
    tau_traced = bg.conformal_age
    tau_frozen = jax.lax.stop_gradient(tau_traced)
    # f = d ln D / d ln a, evaluated at log(a)=0 (z=0)
    loga_today = 0.0
    f_today = bg.f_of_loga.evaluate(loga_today)
    a_today = jnp.exp(loga_today)
    # H(a) at a=1: H_of_loga returns H in Mpc^-1 (CLASS internal units)
    H_today = bg.H_of_loga.evaluate(loga_today)
    # aH at z=0
    aH_today = a_today * H_today
    # Correction
    delta_m = delta_m_frozen + f_today * aH_today * delta_m_frozen * (tau_traced - tau_frozen)

    primordial = primordial_scalar_pk(jnp.asarray([k]), params)[0]
    return 2.0 * jnp.pi**2 / k**3 * primordial * delta_m**2


def fd_grad(name, eps, fn):
    val = getattr(params, name)
    p_plus = dataclasses.replace(params, **{name: val + eps})
    p_minus = dataclasses.replace(params, **{name: val - eps})
    return float(fn(p_plus, prec, K_TARGET) - fn(p_minus, prec, K_TARGET)) / (2 * eps)


print("=== AD with Taylor correction (taylor compute_pk) ===")
t0 = time.time()
g = jax.grad(lambda p: compute_pk_taylor(p, prec, K_TARGET))(params)
g.h.block_until_ready()
print(f"  AD: {time.time()-t0:.1f}s")

ad = {n: float(getattr(g, n)) for n in PARAM_NAMES}

print("\n=== FD on Taylor-corrected compute_pk ===")
t0 = time.time()
fd = {n: fd_grad(n, FD_STEPS[n], compute_pk_taylor) for n in PARAM_NAMES}
print(f"  FD: {time.time()-t0:.1f}s")

print("\n  param         AD                FD            |AD/FD-1|")
for n in PARAM_NAMES:
    if abs(fd[n]) > 1e-30:
        r = abs(ad[n] / fd[n] - 1)
        print(f"  {n:<11s} {ad[n]:14.4e}    {fd[n]:14.4e}   {r:8.4%}")
    else:
        print(f"  {n:<11s} {ad[n]:14.4e}    {fd[n]:14.4e}   FD~0")
