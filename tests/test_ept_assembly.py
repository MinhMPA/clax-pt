"""Assembly function sanity tests for clax.ept.

No reference data required -- these test algebraic consistency of the
bias combination functions (pk_gg_real, pk_mm_real, etc.).

Tests:
  1. pk_gg_real(b1=1, b2=0, ...) ~ pk_mm_real(cs0) (galaxy reduces to matter)
  2. Counterterm sign: pk_mm_real(cs0=10) < pk_mm_real(cs0=0) (subtracts power)
  3. Shot noise additive: pk_gg_real(..., Pshot=X) - pk_gg_real(..., Pshot=0) ~ X

Usage:
    pytest tests/test_ept_assembly.py -v
    pytest tests/test_ept_assembly.py -v --fast
"""

# Force CPU backend BEFORE importing JAX
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"

import pytest
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp

from clax.ept import (
    compute_ept, EPTPrecisionParams,
    pk_mm_real, pk_gg_real,
)


# ---------------------------------------------------------------------------
# Fixture: compute EPT from fiducial pk_lin
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def assembly_setup():
    """Load pk_lin from CLASS-PT reference, run compute_ept, return components."""
    ref_path = os.path.join(
        os.path.dirname(__file__), "..", "reference_data",
        "classpt_z0.38_fullrange.npz"
    )
    if not os.path.isfile(ref_path):
        pytest.skip(f"Reference data not found: {ref_path}")

    ref = np.load(ref_path, allow_pickle=True)
    k_ept = ref["k_h"]
    pk_lin_ept_np = ref["pk_lin"]
    h = float(ref["h"])
    fz = float(ref["fz"])

    pk_lin_ept = jnp.array(pk_lin_ept_np)
    k_ept_jax = jnp.array(k_ept)

    prec = EPTPrecisionParams()
    ept_out = compute_ept(
        pk_lin_ept, k_ept_jax, h=h, f=fz, prec=prec,
    )

    return {
        "ept_out": ept_out,
        "k_ept": k_ept,
    }


# ---------------------------------------------------------------------------
# Test 1: pk_gg_real(b1=1, b2=0, bG2=0, bGamma3=0, Pshot=0) ~ pk_mm_real
# ---------------------------------------------------------------------------

def test_pk_gg_real_reduces_to_pk_mm(assembly_setup):
    """pk_gg_real with b1=1 and all other biases zero should equal pk_mm_real.

    When b1=1, b2=bG2=bGamma3=0, cs=0, Pshot=0:
      pk_gg_real = 1^2*(Pk_tree+Pk_loop) + 2*(0+cs0)*Pk_ctr = pk_mm_real(cs0)
    """
    ept_out = assembly_setup["ept_out"]
    k_ept = assembly_setup["k_ept"]
    cs0 = 5.0

    p_mm = np.array(pk_mm_real(ept_out, cs0=cs0))
    p_gg = np.array(pk_gg_real(
        ept_out, b1=1.0, b2=0.0, bG2=0.0, bGamma3=0.0,
        cs=0.0, cs0=cs0, Pshot=0.0,
    ))

    # Compare on k < 0.5 h/Mpc where both are well-defined
    mask = k_ept < 0.5
    abs_ref = np.abs(p_mm[mask])
    valid = abs_ref > 1e-3 * abs_ref.max()

    if valid.sum() < 5:
        pytest.skip("Too few valid points for comparison")

    rel_err = np.abs(p_gg[mask][valid] - p_mm[mask][valid]) / abs_ref[valid]
    max_err = float(rel_err.max())
    mean_err = float(rel_err.mean())

    print(f"\npk_gg(b1=1,b2=0,...) vs pk_mm: max_err={max_err:.6e}, "
          f"mean_err={mean_err:.6e}")

    assert max_err < 1e-10, (
        f"pk_gg_real(b1=1, others=0) should exactly equal pk_mm_real, "
        f"but max rel err = {max_err:.2e}"
    )


# ---------------------------------------------------------------------------
# Test 2: Counterterm sign -- cs0 > 0 subtracts power
# ---------------------------------------------------------------------------

def test_counterterm_sign(assembly_setup):
    """pk_mm_real(cs0=10) - pk_mm_real(cs0=0) should be negative.

    The counterterm is 2*cs0*Pk_ctr where Pk_ctr = -k^2 * P_lin.
    Since cs0 > 0 and Pk_ctr < 0, the correction is negative (subtracts power).
    """
    ept_out = assembly_setup["ept_out"]
    k_ept = assembly_setup["k_ept"]

    p_cs0 = np.array(pk_mm_real(ept_out, cs0=0.0))
    p_cs10 = np.array(pk_mm_real(ept_out, cs0=10.0))

    diff = p_cs10 - p_cs0

    # Check on k > 0.01 where the counterterm is non-negligible
    mask = k_ept > 0.01
    diff_masked = diff[mask]

    n_negative = int((diff_masked < 0).sum())
    n_total = len(diff_masked)
    frac_negative = n_negative / n_total

    print(f"\nCounterterm sign: {n_negative}/{n_total} = {frac_negative:.1%} "
          f"of modes have P(cs0=10) < P(cs0=0)")
    print(f"  diff range: [{diff_masked.min():.3e}, {diff_masked.max():.3e}]")

    assert frac_negative > 0.95, (
        f"Expected >95% of modes to have negative counterterm correction, "
        f"but only {frac_negative:.1%}. Counterterm sign may be wrong."
    )


# ---------------------------------------------------------------------------
# Test 3: Shot noise is additive
# ---------------------------------------------------------------------------

def test_pshot_additive(assembly_setup):
    """pk_gg_real(..., Pshot=X) - pk_gg_real(..., Pshot=0) should be X everywhere.

    Shot noise enters as a constant additive term in pk_gg_real.
    """
    ept_out = assembly_setup["ept_out"]
    Pshot_val = 1000.0

    p_no_shot = np.array(pk_gg_real(
        ept_out, b1=2.0, b2=0.5, bG2=-0.1, bGamma3=0.0,
        cs=0.0, cs0=0.0, Pshot=0.0,
    ))
    p_with_shot = np.array(pk_gg_real(
        ept_out, b1=2.0, b2=0.5, bG2=-0.1, bGamma3=0.0,
        cs=0.0, cs0=0.0, Pshot=Pshot_val,
    ))

    diff = p_with_shot - p_no_shot
    abs_err = np.abs(diff - Pshot_val)
    max_abs_err = float(abs_err.max())

    print(f"\nPshot additive: max |diff - {Pshot_val}| = {max_abs_err:.6e}")

    assert max_abs_err < 1e-8, (
        f"Shot noise should be additive, but max deviation = {max_abs_err:.2e}. "
        f"Expected P(Pshot={Pshot_val}) - P(Pshot=0) = {Pshot_val} everywhere."
    )


# ---------------------------------------------------------------------------
# Row-map guards (ported individually from the clax-pt validation campaign
# branch; the rest of that branch's copy of this file duplicated tests 1-3
# above).  These close the triangle between the two independent transcriptions
# of classy's accessors (classy.pyx:4816-4932 @ 09d5531a): the NumPy twin
# `scripts/classpt_assembly.py` (asserted against classy itself on every
# generated reference file) and clax's own nine accessors.  A pk_mult array is
# rebuilt from clax's leaves via `tests.ept_campaign_utils.pm_from_leaves` (the
# inverse of the row map) and compared both to the twin and to CLASS-PT's own
# stored alpha = 1 array.  A mismatch is a transcription error or a sign flip,
# not an accuracy statement: both sides see the SAME leaves.
#
# Multi-cosmology RULE (CLAUDE.md): `twin_setup` sweeps three exact-alpha = 1
# CLASS-PT references, one per cosmology family (LCDM z = 0.38, nuLCDM z = 0,
# w0waCDM z = 0), and prunes to the LCDM point under ``--fast``.
# ---------------------------------------------------------------------------

from clax.ept import _pd2d2_0                                        # noqa: E402
from tests.ept_campaign_utils import (                               # noqa: E402
    BIAS_KEYS, pm_from_leaves as _pm_from_leaves, clax_nine as _clax_nine,
    rel as _rel, _pm_rows_h3, _PM_ROWS_H,
)

try:
    from scripts import classpt_assembly as ca   # repo root on sys.path (PYTHONPATH / rootdir)
except ImportError:                              # pragma: no cover
    ca = None
needs_twin = pytest.mark.skipif(
    ca is None, reason="scripts/classpt_assembly.py not importable (repo root not on sys.path)")

_ROOT = os.path.join(os.path.dirname(__file__), "..")
CLASSPT_DIR = os.path.join(_ROOT, "reference_data", "classpt")
# One exact-alpha=1 reference per cosmology family.  hratio == Dratio == 1 is
# asserted in the fixture, not assumed: compute_ept's AP defaults are then the
# identity and the stored pk_mult is the alpha = 1 array the row map inverts.
TWIN_CASES = {
    "lcdm_fiducial_z0.38": os.path.join(CLASSPT_DIR, "lcdm_fiducial", "z0.380_noap_cb.npz"),
    "massive_nu_015_z0": os.path.join(CLASSPT_DIR, "massive_nu_015", "z0.000_ap_omfid0.31_cb.npz"),
    "w0wa_m07_m10_z0": os.path.join(CLASSPT_DIR, "w0wa_m07_m10", "z0.000_ap_omfid0.31_cb.npz"),
}
FAST_TWIN_CASE = "lcdm_fiducial_z0.38"
K_WINDOW = (0.01, 0.4)          # the window the CLASS-PT row comparisons use

# All ten biases nonzero: a zero hides exactly the sign and factor bugs these
# tests exist to catch -- b2 gates the b2^2 P_Id2d2 / Pd2d2_0 pair, cs vs cs0
# the pk_gm_real counterterm, b4 the k^2 mu^4 tail.
BIAS_BASE = (2.0, 0.3, -0.2, 0.1, 5.0, 15.0, -3.0, 2.0, 300.0, 100.0)

_PM_ROWS_H3 = _pm_rows_h3()
# Rows 42-47 (Id2d2_2 ... IG2G2_4) are filled by nonlinear_pt.c and read by no
# classy accessor, so pm_from_leaves leaves them zero.
_PM_ROWS_UNUSED = tuple(range(42, 48))


@pytest.fixture(scope="module", params=list(TWIN_CASES), ids=list(TWIN_CASES))
def twin_setup(request):
    """compute_ept + the stored alpha=1 pk_mult at one CLASS-PT reference."""
    name = request.param
    if request.config.getoption("--fast") and name != FAST_TWIN_CASE:
        pytest.skip(f"--fast runs {FAST_TWIN_CASE} only (skipping {name})")
    path = TWIN_CASES[name]
    if not os.path.isfile(path):
        pytest.skip(f"reference missing: {path}")
    d = np.load(path, allow_pickle=False)
    hr, dr = float(d["hratio"]), float(d["Dratio"])
    assert hr == 1.0 and dr == 1.0, \
        f"{name}: not an exact alpha=1 reference (hratio={hr!r}, Dratio={dr!r})"
    h, fz = float(d["h"]), float(d["fz"])
    ept_out = compute_ept(jnp.array(d["pk_lin"]), jnp.array(d["k_h"]),
                          h=h, f=fz, prec=EPTPrecisionParams())
    return {"name": name, "ept_out": ept_out, "h": h, "fz": fz,
            "pm_stored": np.asarray(d["pk_mult"])[:48]}


def test_pm_row_map_is_complete_and_unique():
    """Structural guard on the row map: rows 0-41 each appear exactly once and
    nothing writes into 42-47 (which no classy accessor reads).  A duplicated
    or dropped row here would silently zero a term in every comparison below.

    Cosmology-independent (RULE exemption): it inspects the map, not any output.
    """
    rows = [r for r, _, _ in _PM_ROWS_H3] + [r for r, _ in _PM_ROWS_H]
    assert sorted(rows) == list(range(42)), \
        f"row map is not a permutation of 0-41: dup {sorted({r for r in rows if rows.count(r) > 1})}, " \
        f"missing {sorted(set(range(42)) - set(rows))}, stray {sorted(r for r in rows if r >= 42)}"
    assert set(_PM_ROWS_UNUSED) == set(range(42, 48))


def test_pm_row_map_signs_are_uniformly_plus_one():
    """Against the RELEASED clax every leaf is stored in the sign
    `classy.pyx`'s `get_pk_mult` hands out, so the 48-row map carries no -1.

    This is the documentary half of the claim; the empirical half is
    `test_pm_from_leaves_row_signs_at_alpha1` below, which measures each row
    against CLASS-PT's own alpha = 1 `pk_mult` (a flip reads ~2.0 there).  If
    this assertion ever has to be relaxed, the sign convention has moved --
    report it, do not restore a -1.

    Cosmology-independent (RULE exemption): it inspects the map, not any output.
    """
    flipped = sorted(r for r, s, _ in _PM_ROWS_H3 if s != +1)
    assert not flipped, f"rows still carrying a -1: {flipped}"


@needs_twin
def test_pd2d2_0_matches_twin(twin_setup):
    """clax `_pd2d2_0` and the twin's `pd2d2_0` are the same transcription of
    classy.pyx:4813 (simpson(P^2 kh^3, x=ln kh)/pi^2), fed the same Ptree."""
    e = twin_setup["ept_out"]
    want = ca.pd2d2_0(np.asarray(e.Pk_tree), np.asarray(e.kh))
    got = float(_pd2d2_0(e.Pk_tree, e.kh))
    assert abs(got - want) < 1e-12 * abs(want), (twin_setup["name"], got, want)


@needs_twin
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_accessors_match_classy_twin(twin_setup, seed):
    """All nine spectra: the clax accessors vs `assemble_from_pm` on a pm array
    built from the SAME EPTComponents, every bias nonzero.

    Both sides are double-precision transcriptions of classy.pyx:4816-4932 over
    identical leaves, so 1e-12 is the round-off floor of the /h**3-then-*h**3
    round trip, not an accuracy tolerance -- a failure is a transcription error
    on one of the two sides (or a mislabelled row), never a CLASS-PT residual.
    """
    e, h, fz = twin_setup["ept_out"], twin_setup["h"], twin_setup["fz"]
    rng = np.random.default_rng(seed)
    bias = {k: v * (1.0 + 0.3 * rng.uniform(-1, 1))       # all nonzero, seed-varied
            for k, v in zip(BIAS_KEYS, BIAS_BASE)}
    pm = _pm_from_leaves(e, h)
    kh = np.asarray(e.kh)
    Pd2d2_0 = float(_pd2d2_0(e.Pk_tree, e.kh))
    twin = ca.assemble_from_pm(pm, h, fz, kh, bias, Pd2d2_0)
    ours = _clax_nine(e, bias)
    resid = {n: _rel(ours[n], twin[n]) for n in ours}
    bad = {n: f"{r:.3e}" for n, r in resid.items() if r > 1e-12}
    assert not bad, (f"{twin_setup['name']} seed {seed}: clax accessors vs classy "
                     f"twin: {bad}")


def test_pm_from_leaves_row_signs_at_alpha1(twin_setup):
    """The inverse row map against CLASS-PT's own alpha=1 `pk_mult`.

    This is a SIGN and normalisation guard on `pm_from_leaves`, not an accuracy
    claim: a flipped sign reads 2.0 on this metric while the residual against
    CLASS-PT tops out near 1.6e-2 (the RSD one-loop rows 21-29), so the 5e-2
    threshold has 3x margin below and 40x above.  NEVER widen it to accommodate
    a file.

    Rows 40/41 are excluded from the ratio and asserted ~0 on both sides
    instead: they are l=4 reprojections of the b1b2 / b1bG2 kernels, which
    vanish by orthogonality at alpha = 1 (max|stored| ~ 1e-13 of max|pm[14]|).
    """
    e, h = twin_setup["ept_out"], twin_setup["h"]
    stored = twin_setup["pm_stored"]
    kh = np.asarray(e.kh)
    sel = (kh > K_WINDOW[0]) & (kh <= K_WINDOW[1])
    pm = _pm_from_leaves(e, h)
    rows = [r for r, _, _ in _PM_ROWS_H3 if r not in (40, 41)] + [r for r, _ in _PM_ROWS_H]
    rel = {r: _rel(pm[r][sel], stored[r][sel]) for r in sorted(rows)}
    bad = [(r, f"{rel[r]:.3e}") for r in sorted(rel) if rel[r] > 5e-2]
    worst = max((v, r) for r, v in rel.items())
    assert not bad, (f"{twin_setup['name']}: inverse row map disagrees with the alpha=1 "
                     f"pk_mult on {bad} (sign flip reads ~2.0); worst overall "
                     f"row {worst[1]} {worst[0]:.3e}")
    print(f"\n{twin_setup['name']}: row-map worst row {worst[1]} {worst[0]:.3e}")
    # rows 40/41: near-zero on both sides at alpha = 1
    scale = float(np.max(np.abs(stored[14][sel])))
    for r in (40, 41):
        for label, arr in (("clax", pm[r]), ("CLASS-PT", stored[r])):
            m = float(np.max(np.abs(np.asarray(arr)[sel])))
            assert m < 1e-6 * scale, (f"{twin_setup['name']}: {label} pm[{r}] is not ~0 at "
                                      f"alpha=1 (max {m:.3e} vs 1e-6*max|pm[14]| "
                                      f"{1e-6 * scale:.3e})")
