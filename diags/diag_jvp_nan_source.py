"""Pinpoint which input tangent into thermodynamics_solve has NaN, and at what index."""
import jax, jax.numpy as jnp, dataclasses, sys
sys.path.insert(0, '/home/n2minh/clax')

from clax.params import CosmoParams, PrecisionParams
from clax.background import background_solve
from clax.thermodynamics import thermodynamics_solve

PREC = PrecisionParams(
    bg_n_points=400, ncdm_bg_n_points=200, bg_tol=1e-8,
    th_n_points=10000, th_z_max=5e4,  # 5e4 floor: see PrecisionParams.th_z_max
    ode_adjoint="direct",
)
params = CosmoParams()

h0 = jnp.asarray(params.h)

print("=== Background JVP NaN check ===")
def bg_only(h):
    p = dataclasses.replace(params, h=h)
    return background_solve(p, PREC)

bg_primal, bg_tangent = jax.jvp(bg_only, (h0,), (jnp.asarray(1.0),))
bg_flat, _ = jax.tree_util.tree_flatten(bg_tangent)
for i, leaf in enumerate(bg_flat):
    if hasattr(leaf, 'shape') and leaf.size > 1:
        nans = ~jnp.isfinite(leaf)
        if jnp.any(nans):
            first_nan = int(jnp.argmax(nans))
            print(f"  BG leaf {i}: FIRST NaN at idx={first_nan}, shape={leaf.shape}")
        else:
            print(f"  BG leaf {i}: OK shape={leaf.shape}")
    elif hasattr(leaf, 'shape'):
        ok = "OK" if jnp.isfinite(leaf) else "NaN"
        print(f"  BG leaf {i}: {ok} scalar={float(leaf):.4e}")

print("\n=== Full pipeline JVP NaN check ===")
def full_thermo(h):
    p = dataclasses.replace(params, h=h)
    bg_ = background_solve(p, PREC)
    return thermodynamics_solve(p, PREC, bg_)

thermo_primal, thermo_tangent = jax.jvp(full_thermo, (h0,), (jnp.asarray(1.0),))
thermo_flat, _ = jax.tree_util.tree_flatten(thermo_tangent)
primal_flat, _ = jax.tree_util.tree_flatten(thermo_primal)

for i, (leaf, plf) in enumerate(zip(thermo_flat, primal_flat)):
    if hasattr(leaf, 'shape') and leaf.size > 1:
        nans = ~jnp.isfinite(leaf)
        if jnp.any(nans):
            first_nan = int(jnp.argmax(nans))
            print(f"  Thermo leaf {i}: FIRST NaN at idx={first_nan}/{leaf.size}, shape={leaf.shape}")
            # Show a few values around the first NaN
            lo, hi = max(0, first_nan - 2), min(leaf.size, first_nan + 3)
            print(f"    tangent[{lo}:{hi}] = {[float(leaf.flat[j]) for j in range(lo, hi)]}")
            print(f"    primal[{lo}:{hi}]  = {[float(plf.flat[j]) for j in range(lo, hi)]}")
        else:
            print(f"  Thermo leaf {i}: OK shape={leaf.shape}")
    elif hasattr(leaf, 'shape'):
        ok = "OK" if jnp.isfinite(leaf) else "NaN"
        print(f"  Thermo leaf {i}: {ok} scalar={float(leaf):.4e}")
