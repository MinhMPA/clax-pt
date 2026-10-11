"""Resolution of the fast-path Bessel lookup table (precision sweep, Oct 2026).

``compute_cls_all_fast`` tabulates j_l(x) on a uniform x grid and interpolates
linearly. Linear interpolation of an oscillation of period 2*pi at spacing h
biases the amplitude by about -h^2/12 on average: at the historical 3 points
per unit x (h ~ 0.33 for k_max = 1/Mpc) that is -0.9% in Delta_l, which the
Oct 2026 sweep measured as a -1% TT offset at l < 50, +20% in EE at l = 2 and a
k_max-dependent C_l. At 30 points per unit x the bias is below 1e-4 and the
fast path matches the exact-j_l path to 0.01% for l <= 500. This test pins the
module default and the bias law. Cosmology-independent numerics: exempt from the
multi-cosmology rule.
"""
import numpy as np
import pytest

import clax.harmonic as harmonic
from clax.bessel import build_jl_table, sparse_l_grid, spherical_jl


def _table_bias(x_per_unit, l_values, x_max=15000.0):
    """Mean and max relative error of linear interpolation of the j_l table
    against the exact j_l over x in [2 l, 3000], for a production-size table
    (x_max = 15000 as for k_max = 1/Mpc; the 30000-point floor is inactive).
    Returns (mean_rel, max_rel, h)."""
    n_x = max(30000, int(x_max * x_per_unit))
    x_table, jl_table, _ = build_jl_table(int(max(l_values)), n_x=n_x, x_max=x_max)
    x_table = np.asarray(x_table)
    jl_table = np.asarray(jl_table)
    h = float(x_table[1] - x_table[0])
    l_sparse = sparse_l_grid(int(max(l_values)))
    rng = np.random.default_rng(0)
    mean_rel, max_rel = [], []
    for l in l_values:
        row = jl_table[int(np.where(l_sparse == l)[0][0])]
        x = rng.uniform(2.0 * l, 3000.0, 20000)
        exact = np.asarray(spherical_jl(l, x))
        interp = np.interp(x, x_table, row)
        env = np.sqrt(np.mean(exact**2))          # rms envelope of j_l on the window
        keep = np.abs(exact) > 0.5 * env          # avoid zero crossings
        rel = (interp[keep] - exact[keep]) / exact[keep]
        mean_rel.append(rel.mean())
        max_rel.append(np.abs(rel).max())
    return np.array(mean_rel), np.array(max_rel), h


L_VALUES = (2, 10, 50, 200)


def test_default_table_resolution_is_fine():
    """The module default must be the measured-adequate 30 points per unit x."""
    assert harmonic.JL_TABLE_X_PER_UNIT >= 30.0


def test_linear_interpolation_bias_follows_h2_over_12():
    """Mean relative error of the linearly interpolated table is -h^2/12 to
    within 50% at both resolutions (h = 0.33: -0.9%; h = 0.033: -9e-5), and
    the fine table's pointwise error is below 5e-4."""
    mean3, max3, h3 = _table_bias(3.0, L_VALUES)
    mean30, max30, h30 = _table_bias(30.0, L_VALUES)
    for mean, h in ((mean3, h3), (mean30, h30)):
        law = -h**2 / 12
        assert np.all(np.abs(mean - law) < 0.5 * abs(law)), f"h={h:.3f}: bias {mean} vs law {law:.2e}"
    assert np.all(max30 < 5e-4), f"fine-table max error: {max30}"
    assert np.all(max3 > 20 * max30)


def test_fast_path_default_uses_fine_table():
    """compute_cls_all_fast must default to the module constant."""
    import inspect
    sig = inspect.signature(harmonic.compute_cls_all_fast)
    assert sig.parameters["x_per_unit"].default == harmonic.JL_TABLE_X_PER_UNIT
