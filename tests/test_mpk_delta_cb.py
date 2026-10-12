"""delta_cb on the matter-power solve (perturbations_solve_mpk).

The matter solve integrates to tau0 (tau_max_factor=1.0); the C_l solve stops
at 0.999*tau0. EPT needs delta_cb at z=0, so the matter solve must carry it.
"""
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from clax.background import background_solve, tau_of_z
from clax.interpolation import CubicSpline as CS
from clax.perturbations import (
    MatterPerturbationResult,
    _pt_saved_output_count,
    _raise_if_diverged,
    perturbations_solve,
    perturbations_solve_mpk,
)
from clax.thermodynamics import thermodynamics_solve
from clax.transfer import compute_linear_matter_pk_from_perturbations
from tests.conftest import cosmology_reference_dir
from tests.pk_test_utils import PK_TABLE_SOLVE_PREC

K_PROBE = np.geomspace(2.0e-4, 0.9, 24)            # Mpc^-1, inside the table preset's grid


def test_pytree_roundtrip_keeps_delta_cb_last():
    """Cosmology-independent pytree plumbing -- exempt from the grid RULE.
    tree_unflatten is positional, so delta_cb must stay LAST."""
    r = MatterPerturbationResult(k_grid=jnp.arange(3.0), tau_grid=jnp.arange(4.0),
                                 delta_m=jnp.ones((3, 4)), delta_cb=2 * jnp.ones((3, 4)))
    leaves, treedef = jax.tree_util.tree_flatten(r)
    assert float(leaves[-1][0, 0]) == 2.0
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert np.array_equal(back.delta_cb, r.delta_cb) and np.array_equal(back.delta_m, r.delta_m)


def test_mpk_output_count_is_two():
    """Batch sizing must count the second stored array. Exempt (no physics)."""
    assert _pt_saved_output_count(solve_kind="mpk") == 2


def test_joint_guard_fires_on_bad_delta_cb_alone():
    """The matter solve guards (delta_m, delta_cb) in ONE _raise_if_diverged
    call (tests/test_divergence_guard.py pins the call count at 3). A
    non-finite delta_cb with a healthy delta_m must still raise. Exempt."""
    good = jnp.ones((4, 5))
    bad = good.at[2, 3].set(jnp.nan)
    out = _raise_if_diverged((good, good), "ok")
    assert np.array_equal(out[0], good) and np.array_equal(out[1], good)
    with pytest.raises(Exception, match="diverged"):
        jax.block_until_ready(_raise_if_diverged((good, bad), "pair"))


_SOLVES = {}


def _solves(params):
    """(bg, C_l solve, matter solve) for one cosmology, memoised by parameter
    values: each solve costs tens of CPU-minutes at PK_TABLE_SOLVE_PREC, and
    lcdm_fiducial and massive_nu_006 are the same cosmology (default
    m_ncdm = 0.06 eV)."""
    key = repr(params)
    if key not in _SOLVES:
        prec = PK_TABLE_SOLVE_PREC
        bg = background_solve(params, prec)
        th = thermodynamics_solve(params, prec, bg)
        _SOLVES[key] = (bg, perturbations_solve(params, prec, bg, th),
                        perturbations_solve_mpk(params, prec, bg, th))
    return _SOLVES[key]


@pytest.mark.slow
def test_mpk_delta_cb_matches_cl_solve_inside_both_grids(nulcdm_cosmology):
    """At z=0.5 both grids contain tau, so delta_cb must agree. Bar 1e-4 on
    delta. An earlier baseline run measured the same solve swap for delta_m at
    <1e-4 in P, i.e. <5e-5 in delta, at fast_cl's rtol 1e-4 / atol 1e-8. This
    test also sweeps the full k-grid, at PK_TABLE_SOLVE_PREC (rtol 1e-5 /
    atol 1e-10, 10x tighter in rtol), and reports the worst mode's k."""
    name, params = nulcdm_cosmology
    bg, pt_cl, pt_m = _solves(params)
    tau = tau_of_z(bg, 0.5)

    def at(pt):
        return np.asarray(jax.vmap(lambda d: CS(pt.tau_grid, d).evaluate(tau))(pt.delta_cb))

    dev = np.abs(at(pt_m) / at(pt_cl) - 1.0)
    i = int(np.argmax(dev))
    rel, k_worst = float(dev[i]), float(np.asarray(pt_m.k_grid)[i])
    print(f"\nDELTA_CB_SWAP {name}: max rel diff {rel:.3e} at k = {k_worst:.3e} Mpc^-1", flush=True)
    assert rel < 1e-4, (f"{name}: mpk vs C_l delta_cb at z=0.5 differ by {rel:.3e} "
                        f"(worst mode k = {k_worst:.3e} Mpc^-1)")


@pytest.mark.slow
def test_mpk_cb_growth_to_today_matches_class(nulcdm_cosmology):
    """delta_cb's growth from z=0.5 to TODAY vs CLASS, as the ratio
    P_cb(z=0)/P_cb(z=0.5): what EPT's default field="cb" needs at z=0, and
    exactly the stretch the C_l solve cannot reach. k-dependent clax-vs-CLASS
    error cancels in the ratio. Bar 1e-3 on the median over k, as for the
    matter growth test in tests/test_pk_tau_endpoint.py; the bug this branch
    fixes is a flat -0.33%.

    Deliberately not P_cb/P_m: through P_m that ratio tests delta_ncdm, where
    clax's ncdm fluid approximation (on in PK_TABLE_SOLVE_PREC) departs from
    the CLASS reference by up to ~0.5% near k ~ 4e-3 Mpc^-1 at 0.15 eV, and
    the reference's P_cb carries a flat ~8e-4 offset from using CLASS's d_tot
    (which includes radiation) as delta_m. Neither is delta_cb's.
    CLASS data exists only for lcdm_fiducial and massive_nu_015 (conftest)."""
    name, params = nulcdm_cosmology
    d = cosmology_reference_dir(name)
    if d is None:
        pytest.skip(f"no CLASS reference for {name} (consistency covered above)")
    ref = np.load(f"{d}/pk.npz")

    def interp(key):
        return np.exp(np.interp(np.log(K_PROBE), np.log(ref["k"]), np.log(ref[key])))

    r_class = interp("pk_cb_lin_z0") / interp("pk_cb_z0.5")
    bg, _, pt_m = _solves(params)
    as_cb = MatterPerturbationResult(k_grid=pt_m.k_grid, tau_grid=pt_m.tau_grid,
                                     delta_m=pt_m.delta_cb, delta_cb=pt_m.delta_cb)
    p0 = np.asarray(compute_linear_matter_pk_from_perturbations(as_cb, bg, params, K_PROBE, z=0.0))
    p5 = np.asarray(compute_linear_matter_pk_from_perturbations(as_cb, bg, params, K_PROBE, z=0.5))
    r = (p0 / p5) / r_class - 1.0
    i = int(np.argmax(np.abs(r)))
    print(f"{name}: P_cb growth z=0.5->0 vs CLASS: median {np.median(r):+.4%}  "
          f"max|.| {abs(r[i]):.3e} at k={K_PROBE[i]:.2e} Mpc^-1")
    assert abs(np.median(r)) < 1e-3, (
        f"{name}: P_cb growth z=0.5->0 vs CLASS, median residual {np.median(r):+.4%} (bar 0.1%)")
