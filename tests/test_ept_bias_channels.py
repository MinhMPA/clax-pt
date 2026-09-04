"""Regression tests for the EPT quadratic-bias channels (b2, bG2) and the
zero-lag P_d2d2(k->0) constant.

No reference data is needed for the physics here: every assertion is an
algebraic invariant of the model itself.  That matters, because the shipped
CLASS-PT reference (`reference_data/classpt_z0.38_fullrange.npz`) was generated
with b2 = bG2 = bGamma3 = cs* = Pshot = 0, so `scripts/accuracy_classpt.py`
multiplies every quadratic-bias channel by zero and cannot see them at all.

Each test below fails on a bug that was live before this module was added:

  1. f=0 reduction  -- the RSD bias kernels were evaluated with nu1/nu2 rebound
     to the b=-0.3 MATTER FFTLog basis instead of the b=-1.6 bias basis.
  2. bG2 sign       -- CLASS-PT negates pk_mult[32,33,36,37,39] on unpacking
     (classy.pyx:4721-4731); clax did not, so bG2 entered with the wrong sign.
  3. Id2d2 sign     -- CLASS-PT's fabs() there belongs to its log-spline
     machinery, not the spectrum; P_Id2d2 is genuinely negative.
  4. Pd2d2_0        -- CLASS-PT adds 0.25*b2^2*Pd2d2_0 in pk_gg_l0
     (classy.pyx:4911); clax omitted it.

Usage:
    pytest tests/test_ept_bias_channels.py -v
"""

# Force CPU backend BEFORE importing JAX
import dataclasses
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp

from clax.ept import compute_ept, pk_gg_l0, pk_gm_real, EPTPrecisionParams


REF = os.path.join(os.path.dirname(__file__), "..", "reference_data",
                   "classpt_z0.38_fullrange.npz")


def _load_pk_lin():
    if not os.path.isfile(REF):
        pytest.skip(f"reference pk_lin not found: {REF}")
    r = np.load(REF, allow_pickle=True)
    return r["k_h"], r["pk_lin"], float(r["h"]), float(r["fz"])


@pytest.fixture(scope="module")
def ept_f0():
    """EPT components at f = 0: redshift space degenerates to real space."""
    k_h, pk_lin, h, _ = _load_pk_lin()
    return compute_ept(jnp.asarray(pk_lin), jnp.asarray(k_h), h=h, f=0.0)


@pytest.fixture(scope="module")
def ept_fz():
    """EPT components at the reference growth rate."""
    k_h, pk_lin, h, fz = _load_pk_lin()
    return compute_ept(jnp.asarray(pk_lin), jnp.asarray(k_h), h=h, f=fz)


def _band(ept):
    kh = np.asarray(ept.kh)
    return (kh > 0.01) & (kh < 0.5)


# ---------------------------------------------------------------------------
# 1. f -> 0 reduces the RSD bias channels to their real-space counterparts
# ---------------------------------------------------------------------------

def test_b1b2_monopole_reduces_to_Id2_at_f0(ept_f0):
    """At f=0, pk_gg_l0's b1*b2 channel IS the real-space b1*b2 channel.

    Catches the nu-basis bug: Pk_Id2 is built before nu1/nu2 are rebound, while
    Pk_0_b1b2 is built after, so a basis mismatch shows up as a ratio != 1.
    """
    m = _band(ept_f0)
    ratio = np.asarray(ept_f0.Pk_0_b1b2)[m] / np.asarray(ept_f0.Pk_Id2)[m]
    assert np.allclose(ratio, 1.0, rtol=1e-6), \
        f"Pk_0_b1b2(f=0)/Pk_Id2 ranges {ratio.min():.6f}..{ratio.max():.6f}, expected 1"


def test_b1bG2_monopole_reduces_to_IG2_at_f0(ept_f0):
    """At f=0, the b1*bG2 monopole channel is 2 x the real-space IG2 channel.

    The factor 2 is CLASS-PT's convention: pk_gg_real carries `2.*b1*bG2*pk_mult[3]`
    (classy.pyx:4827) while pk_gg_l0 carries `b1*bG2*pk_mult[32]` (:4908).

    Catches BOTH the nu-basis bug and the missing bG2 negation -- the latter
    flips the sign, giving -2 instead of +2.
    """
    m = _band(ept_f0)
    ratio = np.asarray(ept_f0.Pk_0_b1bG2)[m] / np.asarray(ept_f0.Pk_IG2)[m]
    assert np.allclose(ratio, 2.0, rtol=1e-6), \
        f"Pk_0_b1bG2(f=0)/Pk_IG2 ranges {ratio.min():.6f}..{ratio.max():.6f}, expected +2"


@pytest.mark.parametrize("channel", ["Pk_0_b2", "Pk_0_bG2", "Pk_2_b2", "Pk_2_bG2",
                                     "Pk_4_b2", "Pk_4_bG2", "Pk_2_b1b2", "Pk_2_b1bG2"])
def test_rsd_only_channels_vanish_at_f0(ept_f0, channel):
    """Every kernel whose numerator carries an explicit factor of f must vanish
    identically at f=0 (see nonlinear_pt.c:5065-5095)."""
    v = np.asarray(getattr(ept_f0, channel))[_band(ept_f0)]
    assert np.max(np.abs(v)) == 0.0, f"{channel}(f=0) is not identically zero"


# ---------------------------------------------------------------------------
# 2. Sign of P_Id2d2
# ---------------------------------------------------------------------------

def test_Id2d2_is_negative(ept_fz):
    """P_d2d2(k) - P_d2d2(0) < 0: the zero-lag value is the maximum.

    CLASS-PT applies fabs() at this point only as part of its
    add-large-constant / log-spline / subtract machinery; the spectrum it
    returns is negative.  Taking abs() permanently flipped the sign of the
    0.25*b2^2 term in every galaxy spectrum.
    """
    v = np.asarray(ept_fz.Pk_Id2d2)[_band(ept_fz)]
    assert np.all(v < 0), \
        f"Pk_Id2d2 should be negative; max value is {v.max():.6g}"


# ---------------------------------------------------------------------------
# 3. Zero-lag constant Pd2d2_0
# ---------------------------------------------------------------------------

def test_Pd2d2_0_matches_analytic_integral():
    """Pd2d2_0 = (1/pi^2) * int P_lin(q)^2 q^3 dln q, the k->0 limit of P_d2d2.

    IR resummation is switched off so the integrand is exactly the supplied
    linear P(k); with it on, clax integrates the RESUMMED spectrum (matching
    CLASS-PT's pk_mult[14]), which shifts the integral by ~0.05%.

    The quadrature is Simpson, not trapezoid: classy.pyx:4813 evaluates this
    with scipy's `simpson`. The two rules differ by ~7e-6 relative here, above
    this test's 1e-6 bound, so the rule has to match to compare against
    CLASS-PT at all.
    """
    from scipy.integrate import simpson
    k_h, pk_lin, h, fz = _load_pk_lin()
    prec = EPTPrecisionParams(ir_resummation=False)
    ept = compute_ept(jnp.asarray(pk_lin), jnp.asarray(k_h), h=h, f=fz, prec=prec)
    expect = simpson(pk_lin ** 2 * k_h ** 3, x=np.log(k_h)) / np.pi ** 2
    got = float(ept.Pd2d2_0)
    assert got == pytest.approx(expect, rel=1e-6), \
        f"Pd2d2_0 = {got:.4f}, analytic integral = {expect:.4f}"


def test_Pd2d2_0_uses_the_resummed_spectrum_by_default(ept_fz):
    """With IR resummation on (the default) the integral is over the resummed
    P(k), so it sits close to -- but not exactly at -- the raw-input value."""
    k_h, pk_lin, _, _ = _load_pk_lin()
    raw = np.trapezoid(pk_lin ** 2 * k_h ** 3, np.log(k_h)) / np.pi ** 2
    assert float(ept_fz.Pd2d2_0) == pytest.approx(raw, rel=5e-3)


def test_Pd2d2_0_is_positive_and_order_of_magnitude(ept_fz):
    """Sanity guard: it is a variance-like quantity, so strictly positive."""
    assert float(ept_fz.Pd2d2_0) > 0


def test_pk_gg_l0_carries_the_b2_squared_zero_lag_term(ept_fz):
    """pk_gg_l0 must contain 0.25*b2^2*Pd2d2_0 (CLASS-PT classy.pyx:4911).

    Isolate it: the b2-dependence of pk_gg_l0 at fixed everything else is
        0.25*b2^2*(Pk_Id2d2 + Pd2d2_0) + b1*b2*Pk_0_b1b2 + b2*Pk_0_b2
    so the second difference in b2 isolates the quadratic piece.
    """
    b1, B = 2.0, 0.7
    kw = dict(bG2=0.0, bGamma3=0.0, cs0=0.0, Pshot=0.0, b4=0.0)
    p_plus = np.asarray(pk_gg_l0(ept_fz, b1, B, **kw))
    p_zero = np.asarray(pk_gg_l0(ept_fz, b1, 0.0, **kw))
    p_minus = np.asarray(pk_gg_l0(ept_fz, b1, -B, **kw))
    quad = 0.5 * (p_plus + p_minus) - p_zero          # = 0.25*B^2*(Id2d2 + Pd2d2_0)
    expect = 0.25 * B ** 2 * (np.asarray(ept_fz.Pk_Id2d2) + float(ept_fz.Pd2d2_0))
    assert np.allclose(quad, expect, rtol=1e-8, atol=1e-8)
    # and it must be non-trivial: dropping Pd2d2_0 would change the result
    without = 0.25 * B ** 2 * np.asarray(ept_fz.Pk_Id2d2)
    assert not np.allclose(quad, without, rtol=1e-3)


# ---------------------------------------------------------------------------
# 4. Pytree plumbing
# ---------------------------------------------------------------------------

def test_pytree_roundtrip_preserves_Pd2d2_0(ept_fz):
    leaves, treedef = jax.tree_util.tree_flatten(ept_fz)
    rt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert float(rt.Pd2d2_0) == float(ept_fz.Pd2d2_0)
    assert float(rt.h) == float(ept_fz.h)
    assert float(rt.f) == float(ept_fz.f)
    assert np.allclose(np.asarray(rt.kh), np.asarray(ept_fz.kh))

# ---------------------------------------------------------------------------
# 6. The FFTLog resolution knob actually works
# ---------------------------------------------------------------------------
# EPTPrecisionParams.nmax advertises 128/256/512. CLASS-PT stores every one of
# those matrices in the same LAPACK 'L' column-major packed order -- it picks
# the filename with Nstr_M22 = "N128"/"N512"/"N256_packed" (nonlinear_pt.c:847-864)
# and feeds all three through one CONVERT_REAL_TO_COMPLEX_M22 macro (:895) and
# one P22 consumer. Reading a non-256 file with any other ordering yields a
# silently wrong M22, which showed up as a ~300% error and a NEGATIVE monopole.
# No reference data: this is clax against itself at three resolutions.


@pytest.fixture(scope="module")
def _nmax_baseline():
    from clax.ept import ept_kgrid
    k_h, pk_lin, h, fz = _load_pk_lin()
    prec = dataclasses.replace(EPTPrecisionParams(), nmax=256, ir_resummation=False)
    kg = np.asarray(ept_kgrid(prec))
    pk = np.exp(np.interp(np.log(kg), np.log(np.asarray(k_h)), np.log(np.asarray(pk_lin))))
    e = compute_ept(jnp.asarray(pk), jnp.asarray(kg), h=h, f=fz, prec=prec)
    return kg, np.asarray(pk_gg_l0(e, 1.0, 0.0, 0.0, 0.0)), h, fz, k_h, pk_lin


@pytest.mark.parametrize("nmax", [128, 512])
def test_nmax_is_a_working_resolution_knob(_nmax_baseline, nmax):
    """P_gg^(0) at nmax=128/512 must stay positive and track nmax=256.

    The FFTLog grid genuinely changes with nmax, so exact agreement is not
    expected; 3% bounds the honest discretisation difference while catching
    the wrong-matrix failure, which is ~300% with a sign flip.
    """
    from clax.ept import ept_kgrid
    kg0, P0, h, fz, k_h, pk_lin = _nmax_baseline
    prec = dataclasses.replace(EPTPrecisionParams(), nmax=nmax, ir_resummation=False)
    kg = np.asarray(ept_kgrid(prec))
    pk = np.exp(np.interp(np.log(kg), np.log(np.asarray(k_h)), np.log(np.asarray(pk_lin))))
    e = compute_ept(jnp.asarray(pk), jnp.asarray(kg), h=h, f=fz, prec=prec)
    P = np.asarray(pk_gg_l0(e, 1.0, 0.0, 0.0, 0.0))
    assert np.all(P > 0), (
        f"nmax={nmax}: monopole goes negative (min {P.min():.4g}) -- "
        "M22/M22b were read in the wrong triangular order")
    kc = np.geomspace(0.02, 0.3, 60)
    dev = np.max(np.abs(np.interp(kc, kg, P) / np.interp(kc, kg0, P0) - 1.0))
    assert dev < 0.03, f"nmax={nmax}: max deviation from nmax=256 is {100*dev:.3f}%"


# ---------------------------------------------------------------------------
# 7. The galaxy-matter counterterm coefficient
# ---------------------------------------------------------------------------


def test_gm_counterterm_carries_the_factor_two(ept_fz):
    """P_gm's cs term is 2*cs*b1*P_ctr, per classy.pyx:4834.

    Differencing in cs isolates that one term exactly, so this pins the
    coefficient without any reference spectrum. With the pre-fix (cs*b1 + cs0)
    the measured slope is half the required one.
    """
    b1, cs = 1.9, 3.0
    d = np.asarray(pk_gm_real(ept_fz, b1, 0.0, 0.0, 0.0, cs0=0.0, cs=cs)) - \
        np.asarray(pk_gm_real(ept_fz, b1, 0.0, 0.0, 0.0, cs0=0.0, cs=0.0))
    want = 2.0 * cs * b1 * np.asarray(ept_fz.Pk_ctr)
    m = _band(ept_fz)
    rel = np.max(np.abs(d[m] - want[m])) / np.max(np.abs(want[m]))
    assert rel < 1e-12, f"cs slope is off by {rel:.3e} of the expected 2*cs*b1*Pk_ctr"


# ---------------------------------------------------------------------------
# 8. Pk_tree is CLASS-PT's Ptree, and Pd2d2_0 is built from it
# ---------------------------------------------------------------------------


def test_tree_is_the_ir_resummed_tree(ept_fz):
    """Pk_tree must be pm[14]: Pnw + Pw*exp(-S k^2)*(1 + S k^2).

    nonlinear_pt.c:2999. clax previously used the raw linear spectrum, which is
    the S -> 0 limit, so it differed from CLASS-PT wherever the BAO wiggle and
    the damping are both non-negligible.
    """
    kh = np.asarray(ept_fz.kh)
    S = float(ept_fz.sigma2_bao)
    damp = np.exp(-S * kh ** 2)
    want = np.asarray(ept_fz.pk_nw) + np.asarray(ept_fz.pk_w) * damp * (1.0 + S * kh ** 2)
    got = np.asarray(ept_fz.Pk_tree)
    m = _band(ept_fz)
    assert np.max(np.abs(got[m] - want[m])) / np.max(np.abs(want[m])) < 1e-12
    # and it is genuinely not the raw linear spectrum
    lin = np.asarray(ept_fz.pk_nw) + np.asarray(ept_fz.pk_w)
    assert np.max(np.abs(got[m] - lin[m])) / np.max(np.abs(lin[m])) > 1e-4


def test_Pd2d2_0_uses_simpson_on_the_tree(ept_fz):
    """Pd2d2_0 = simpson(Ptree^2 kh^3, ln kh)/pi^2 -- classy.pyx:4805-4813.

    Trapezoid on the FFTLog-discretised spectrum, which clax used before, is a
    different integrand under a different rule.
    """
    from scipy.integrate import simpson
    kh = np.asarray(ept_fz.kh)
    want = simpson(np.asarray(ept_fz.Pk_tree) ** 2 * kh ** 3, x=np.log(kh)) / np.pi ** 2
    got = float(ept_fz.Pd2d2_0)
    assert abs(got - want) / want < 1e-10, f"Pd2d2_0 {got:.6f} vs simpson {want:.6f}"
