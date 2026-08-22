"""Final decisive test at pt_ode_rtol=1e-7.

  - FD step sweep on h AND omega_b → confirm FD oracle is converged
  - AD+Taylor for all 5 params → final pass/fail verdict

Reading the result:
  - FD spread <0.5% on h and omega_b → FD oracle converged
  - AD-FD <1% on all 5 → Taylor fix is correct, ship
  - omega_b plateaus >2% while h passes → real bug, need jvp isolation
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
    pt_ode_rtol=1e-7, pt_ode_atol=1e-8,
    ode_max_steps=65536, pt_ode_solver="rodas5",
)
params = CosmoParams()
K_TARGET = 0.1
PARAM_NAMES = ["h", "omega_b", "omega_cdm", "ln10A_s", "n_s"]
FD_STEPS = {"h": 1e-3, "omega_b": 1e-5, "omega_cdm": 1e-3, "ln10A_s": 1e-3, "n_s": 1e-3}

print(f"=== rtol={prec.pt_ode_rtol}, atol={prec.pt_ode_atol}, max_steps={prec.ode_max_steps} ===\n")

def pk_at(name, val):
    p = dataclasses.replace(params, **{name: val})
    return float(compute_pk(p, prec, k=K_TARGET))

# -------- Test A: FD step sweep for h --------
print("=== Test A: FD step sensitivity on compute_pk(k=0.1) for h ===")
t0 = time.time()
pk0 = pk_at("h", params.h)
print(f"  warmup pk(h0)={pk0:.6e} in {time.time()-t0:.1f}s\n")

print("  eps          FD(h)              vs eps=1e-3")
fd_h_results = {}
for eps in [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]:
    t0 = time.time()
    pk_plus = pk_at("h", params.h + eps)
    pk_minus = pk_at("h", params.h - eps)
    fd_h = (pk_plus - pk_minus) / (2 * eps)
    fd_h_results[eps] = fd_h
    print(f"  {eps:.0e}      {fd_h:14.6e}      ({time.time()-t0:.1f}s)")

ref = fd_h_results[1e-3]
spread_h = []
print(f"\n  Sensitivity vs eps=1e-3:")
for eps, val in fd_h_results.items():
    rel = abs(val/ref - 1) if abs(ref) > 1e-30 else 0.0
    spread_h.append(rel)
    print(f"    eps={eps:.0e}: {val:14.6e}  |delta|={rel:.4%}")
print(f"  Max spread (h): {max(spread_h):.4%}")
fd_h_median = float(np.median(list(fd_h_results.values())))
print(f"  Median FD(h):    {fd_h_median:.6e}")

# -------- Test B: FD step sweep for omega_b --------
print("\n=== Test B: FD step sensitivity for omega_b ===")
print("  eps          FD(omega_b)        vs eps=1e-5")
fd_ob_results = {}
for eps in [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6]:
    t0 = time.time()
    pk_plus = pk_at("omega_b", params.omega_b + eps)
    pk_minus = pk_at("omega_b", params.omega_b - eps)
    fd_ob = (pk_plus - pk_minus) / (2 * eps)
    fd_ob_results[eps] = fd_ob
    print(f"  {eps:.0e}      {fd_ob:14.6e}      ({time.time()-t0:.1f}s)")

ref = fd_ob_results[1e-5]
spread_ob = []
print(f"\n  Sensitivity vs eps=1e-5:")
for eps, val in fd_ob_results.items():
    rel = abs(val/ref - 1) if abs(ref) > 1e-30 else 0.0
    spread_ob.append(rel)
    print(f"    eps={eps:.0e}: {val:14.6e}  |delta|={rel:.4%}")
print(f"  Max spread (omega_b): {max(spread_ob):.4%}")
fd_ob_median = float(np.median(list(fd_ob_results.values())))
print(f"  Median FD(omega_b): {fd_ob_median:.6e}")

# -------- Test C: full AD + FD comparison --------
print("\n=== Test C: AD+Taylor vs FD at rtol=1e-7 ===")
print("  AD compile+run...")
t0 = time.time()
g = jax.grad(lambda p: compute_pk(p, prec, k=K_TARGET))(params)
g.h.block_until_ready()
print(f"  AD: {time.time()-t0:.1f}s")
ad = {n: float(getattr(g, n)) for n in PARAM_NAMES}

print("  FD per param (single eps for non-h/non-ob)...")
t0 = time.time()
def fd_grad(name, eps):
    val = getattr(params, name)
    p_plus = dataclasses.replace(params, **{name: val + eps})
    p_minus = dataclasses.replace(params, **{name: val - eps})
    return float(compute_pk(p_plus, prec, k=K_TARGET) - compute_pk(p_minus, prec, k=K_TARGET)) / (2 * eps)
fd = {n: fd_grad(n, FD_STEPS[n]) for n in PARAM_NAMES}
print(f"  FD: {time.time()-t0:.1f}s\n")

# Use median FD where we have a sweep
fd_compare = dict(fd)
fd_compare["h"] = fd_h_median
fd_compare["omega_b"] = fd_ob_median

print(f"  param         AD                FD (median)        |AD/FD-1|")
fail = 0
for n in PARAM_NAMES:
    if abs(fd_compare[n]) > 1e-30:
        r = abs(ad[n] / fd_compare[n] - 1)
        flag = "" if r < 0.01 else " <- FAIL"
        if r >= 0.01:
            fail += 1
        print(f"  {n:<11s} {ad[n]:14.4e}    {fd_compare[n]:14.4e}    {r:8.4%}{flag}")
    else:
        print(f"  {n:<11s} {ad[n]:14.4e}    {fd_compare[n]:14.4e}    FD~0")
print(f"\nFAIL count (>1%): {fail} / {len(PARAM_NAMES)}")
print(f"FD step-spread: h={max(spread_h):.4%}, omega_b={max(spread_ob):.4%}")
