"""Shared helpers for the recombination-staging tests (smsharma/clax#48).
Presets and metrics copy the variant probe (/lustre/work/n2minh/clax-probes/
helium/scripts/thermo_probe.py) so the in-repo numbers are comparable."""
import os
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from clax import PrecisionParams
from clax.background import background_solve
from clax.thermodynamics import thermodynamics_solve

REF_DIR = os.path.join(os.path.dirname(__file__), "..", "reference_data")
PREC_RB = replace(PrecisionParams.fast_cl(), pt_ode_solver="rodas5",
                  pt_ode_rtol=1e-4, pt_ode_atol=1e-4, ode_max_steps=16384)   # spec D8
THERMO_GRIDS = (20000, 3000)          # default grid; fit_cl's thermodynamics grid
THETAS = ("h", "omega_b", "omega_cdm", "Y_He", "N_ur")
QS = ("xe", "cs2", "tb")


def thermo_prec(n_points):
    """Thermodynamics preset on an n_points grid; jax.jvp through
    background_solve needs the direct adjoint."""
    return replace(PREC_RB, th_n_points=n_points, ode_adjoint="direct")


def as_traced(p):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float64), p)


def tables(p, prec):
    bg = background_solve(p, prec)
    th = thermodynamics_solve(p, prec, bg)
    return {"loga": th.xe_of_loga.x, "xe": th.xe_of_loga.y, "cs2": th.cs2_of_loga.y,
            "tb": th.Tb_of_loga.y, "g": th.g_of_loga.y}


def tangent_tables(p, prec, theta):
    """(primal tables, forward tangents d/d theta) as numpy arrays."""
    p = as_traced(p)
    t = jax.tree_util.tree_map(jnp.zeros_like, p).replace(**{theta: jnp.asarray(1.0)})
    prim, tan = jax.jit(lambda pp, tt: jax.jvp(lambda q: tables(q, prec), (pp,), (tt,)))(p, t)
    return (jax.tree_util.tree_map(np.asarray, prim), jax.tree_util.tree_map(np.asarray, tan))


def z_of(loga):
    return 1.0 / np.exp(np.asarray(loga)) - 1.0


def spike_ratio(dln, z, zlo, zhi, half=25, floor=1e-8):
    """max over grid points in [zlo, zhi] of |dln| / median(|dln| over +-half points)."""
    a = np.abs(np.asarray(dln))
    pad = np.pad(a, half, mode="edge")
    med = np.median(np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1), axis=1)
    sel = (z >= zlo) & (z <= zhi)
    r = np.where(sel, a / np.maximum(med, floor), 0.0)
    i = int(np.argmax(r))
    return float(r[i]), float(z[i])


def visibility_peak(z, g, zlo=800.0, zhi=1400.0):
    s = (z >= zlo) & (z <= zhi)
    zz, gg = np.asarray(z)[s], np.asarray(g)[s]
    i = int(np.argmax(gg))
    c = np.polyfit(zz[i - 1:i + 2], gg[i - 1:i + 2], 2)
    return float(-c[1] / (2.0 * c[0]))


def load_ref(name):
    return np.load(os.path.join(REF_DIR, name, "thermo_recfast_hesw1.npz"))


def xe_on_ref_grid(prim, ref_z):
    """clax x_e (CubicSpline in log a) evaluated at the reference redshifts."""
    from clax.interpolation import CubicSpline
    sp = CubicSpline(jnp.asarray(prim["loga"]), jnp.asarray(prim["xe"]))
    return np.asarray(sp.evaluate(jnp.log(1.0 / (1.0 + jnp.asarray(ref_z)))))
