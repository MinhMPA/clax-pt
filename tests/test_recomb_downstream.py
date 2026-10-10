"""S2 / S3 (smsharma/clax#48): d/dh of f = sum_k delta_m(k, tau0)^2 through the
matter solve is independent of solver step placement, agrees between forward
and reverse mode (native and production custom_vjp), and is smooth along h.
Multi-cosmology: recomb_downstream_cosmology (3 points)."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.recomb_test_utils import PREC_RB, dm_functional, dm_functional_production

pytestmark = pytest.mark.slow
FWD = replace(PREC_RB, ode_adjoint="direct")


def _fwd(base, prec, jit):
    f = dm_functional(prec, base)
    g = lambda h: jax.jvp(f, (h,), (jnp.asarray(1.0),))[1]
    return float((jax.jit(g) if jit else g)(jnp.asarray(float(base.h))))


def test_dfdh_independent_of_step_placement(recomb_downstream_cosmology):
    name, base = recomb_downstream_cosmology
    vals = {(c, j): _fwd(base, replace(FWD, pt_k_chunk_size=c), j)
            for c in (8, 62) for j in (True, False)}
    v = np.array(list(vals.values()))
    spread = (v.max() - v.min()) / abs(np.median(v))
    print(f"{name}: df/dh over chunk x jit {vals} spread {spread:.3e}")
    assert spread < 3e-4, f"ERROR step-placement spread {spread:.3e} ({name})"


def test_native_reverse_matches_forward(recomb_downstream_cosmology):
    name, base = recomb_downstream_cosmology
    fwd = _fwd(base, FWD, True)
    rev = float(jax.grad(dm_functional(PREC_RB, base))(jnp.asarray(float(base.h))))
    rel = abs(rev / fwd - 1.0)
    print(f"{name}: forward {fwd:.8e} native reverse {rev:.8e} rel {rel:.3e}")
    assert rel < 3e-4, f"ERROR native reverse vs forward {rel:.3e} ({name})"


def test_production_reverse_matches_forward(recomb_downstream_cosmology):
    name, base = recomb_downstream_cosmology
    fwd = _fwd(base, FWD, True)
    rev = float(jax.grad(dm_functional_production(PREC_RB, base))(jnp.asarray(float(base.h))))
    rel = abs(rev / fwd - 1.0)
    print(f"{name}: forward {fwd:.8e} production reverse {rev:.8e} rel {rel:.3e}")
    assert rel < 3e-4, f"ERROR production reverse vs forward {rel:.3e} ({name})"


def test_dfdh_smooth_along_h(recomb_downstream_cosmology):
    name, base = recomb_downstream_cosmology
    hs = np.linspace(0.62, 0.72, 11)
    f = dm_functional(FWD, base)
    g = jax.jit(lambda h: jax.jvp(f, (h,), (jnp.asarray(1.0),))[1])
    d = np.array([float(g(jnp.asarray(h))) for h in hs])
    resid = d - np.polyval(np.polyfit(hs, d, 3), hs)
    rough = float(np.max(np.abs(resid)) / abs(np.median(d)))
    print(f"{name}: df/dh along h {d} cubic-fit residual {rough:.3e}")
    assert rough < 3e-4, f"ERROR df/dh not smooth along h: {rough:.3e} ({name})"
