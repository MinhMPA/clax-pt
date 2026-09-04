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

from clax.ept import (
    compute_ept, pk_gg_l0, pk_gg_l2, pk_gg_l4, pk_gm_real, EPTPrecisionParams,
    _GAUSS_NODES, _GAUSS_WEIGHTS,
)


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
                                     "Pk_4_b2", "Pk_4_bG2"])
def test_rsd_only_channels_vanish_at_f0(ept_f0, channel):
    """Every kernel whose numerator carries an explicit factor of f must vanish
    identically at f=0 (see nonlinear_pt.c:5065-5095).

    These six are the ones the GL mu-loop builds from l=0,2,4 inputs that ALL
    carry an f (nonlinear_pt.c:5327-5328), so every term in the projection is
    exactly 0.0 and the sum is bit-zero.  The b1b2 / b1bG2 quadrupoles are
    built from a NON-zero l=0 input and are covered separately below.
    """
    v = np.asarray(getattr(ept_f0, channel))[_band(ept_f0)]
    assert np.max(np.abs(v)) == 0.0, f"{channel}(f=0) is not identically zero"


@pytest.mark.parametrize("channel,parent", [("Pk_2_b1b2", "Pk_0_b1b2"),
                                            ("Pk_2_b1bG2", "Pk_0_b1bG2")])
def test_b1b2_quadrupoles_vanish_at_f0_to_quadrature_roundoff(ept_f0, channel, parent):
    """P_2_b1b2 / P_2_b1bG2 at f=0 are zero only to the GL rule's own accuracy.

    Their M22 kernels carry an explicit f, so the l=2 INPUT is bit-zero; but
    nonlinear_pt.c:5326 rebuilds the mu-dependence as
    P_0_b1bG2 + LegendreP2true * P_2_b1bG2 and projects that on L2 at :5351, so
    what survives at f=0 is the non-zero l=0 channel times the quadrature's own
    <L2> residual.  That residual is sum(2.5 w L2) over CLASS-PT's 40-node
    gauss_tab.dat, ~4e-14, and it is CLASS-PT's residual too -- not a clax
    approximation.  The bound below is tied to the measured residual rather
    than to a hand-picked number.

    Before the mu-loop port these leaves were the bare kernel contraction, so
    they were bit-zero here; the change is a consequence of following CLASS-PT's
    construction, not a loss of accuracy.
    """
    mu = np.asarray(_GAUSS_NODES)
    w = np.asarray(_GAUSS_WEIGHTS)
    L2_residual = abs(np.sum(2.5 * w * 0.5 * (3.0 * mu ** 2 - 1.0)))
    assert L2_residual < 1e-12, "gauss_tab.dat is far worse than expected"

    m = _band(ept_f0)
    v = np.max(np.abs(np.asarray(getattr(ept_f0, channel))[m]))
    p = np.max(np.abs(np.asarray(getattr(ept_f0, parent))[m]))
    assert v <= 5.0 * L2_residual * p, (
        f"{channel}(f=0) is {v:.3e}, above 5x the quadrature <L2> residual "
        f"({5.0 * L2_residual * p:.3e}); that is a real f-independent leak, "
        "not round-off")


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


# ---------------------------------------------------------------------------
# 9. The mu-loop: counterterms, IFG2 and the galaxy multipole assembly
# ---------------------------------------------------------------------------
# CLASS-PT computes every mu-dependent leaf inside a 40-node Gauss-Legendre
# loop over an ANISOTROPIC IR damping Sigmatot(mu) (nonlinear_pt.c:4480).  clax
# used closed-form isotropic Legendre moments for the counterterms and for the
# IFG2 multipoles, which are exact only while the damping is mu-independent.
# Everything below is clax against its own leaves -- no reference data.
#
# Multi-cosmology (CLAUDE.md RULE): compute_ept's cosmology IS its (pk_lin, h,
# f) input, so the three points below vary the linear amplitude and the growth
# rate together, rather than only the fiducial pair.

_MU_POINTS = [
    ("fiducial", 1.00, 1.00),
    ("lowamp",   0.70, 0.93),
    ("highamp",  1.45, 1.07),
]

_mu_cache = {}


def _mu_ept(label, pk_scale, f_scale):
    """compute_ept at one (amplitude, growth-rate) point, cached per-module."""
    if label not in _mu_cache:
        k_h, pk_lin, h, fz = _load_pk_lin()
        _mu_cache[label] = compute_ept(
            jnp.asarray(np.asarray(pk_lin) * pk_scale), jnp.asarray(k_h),
            h=h, f=fz * f_scale)
    return _mu_cache[label]


def _gl_pieces(ept):
    """(mu, w, Sigmatot(mu), Exp(mu), p_lin(mu)) at alpha = 1, from the leaves.

    Mirrors nonlinear_pt.c:4480-4481 and the p_lin bracket of :4498-4500.
    Axis 0 is the mu node, axis 1 is k.
    """
    mu = np.asarray(_GAUSS_NODES)[:, None]
    w = np.asarray(_GAUSS_WEIGHTS)
    k = np.asarray(ept.kh)[None, :]
    f = float(ept.f)
    S, dS = float(ept.sigma2_bao), float(ept.delta_sigma2_bao)
    mu2 = mu ** 2
    Sig = S * (1.0 + f * mu2 * (2.0 + f)) + dS * f ** 2 * mu2 * (mu2 - 1.0)
    Exp = np.exp(-Sig * k ** 2)
    p_lin = np.asarray(ept.pk_nw)[None, :] + np.asarray(ept.pk_w)[None, :] * Exp
    return mu, w, Sig, Exp, p_lin


def _proj(w, mu, ell, integrand):
    """LEGENDRE_PROJECT, nonlinear_pt.c:2565-2568, at alpha = 1."""
    L = {0: np.ones_like(mu), 2: 0.5 * (3.0 * mu ** 2 - 1.0),
         4: (35.0 * mu ** 4 - 30.0 * mu ** 2 + 3.0) / 8.0}[ell]
    return np.einsum("m,mk->k", w * (2.0 * ell + 1.0) / 2.0, L * integrand)


@pytest.mark.parametrize("label,pk_scale,f_scale", _MU_POINTS)
def test_counterterms_are_gl_projections_not_closed_forms(label, pk_scale, f_scale):
    """Pk_ctr0/2/4 must be the 40-node GL projections of k^2 (Pnw + Pw Exp(mu)).

    nonlinear_pt.c:4498-4500 builds Pctr0/2/4 = ktrue^2 (Pnw + Pw Exp) times
    {1, f mu2t, f^2 mu4t} INSIDE the mu loop, against the anisotropic Sigmatot
    of :4480, and projects at :4550-4552.  The closed forms clax used --
    -k^2 Pbin, -k^2 Pbin f 2/3, -k^2 Pbin f^2 8/35 -- are the isotropic
    Legendre moments, exact only if Exp were mu-independent.  Since every b4
    term in all three galaxy multipoles is built on Pk_ctr4 (classy.pyx:4912,
    4922, 4932), the error propagates into b4 as well.
    """
    ept = _mu_ept(label, pk_scale, f_scale)
    mu, w, _Sig, Exp, p_lin = _gl_pieces(ept)
    k = np.asarray(ept.kh)
    f = float(ept.f)
    m = _band(ept)

    want = {0: -_proj(w, mu, 0, k[None, :] ** 2 * p_lin),
            2: -_proj(w, mu, 2, k[None, :] ** 2 * p_lin * f * mu ** 2),
            4: -_proj(w, mu, 4, k[None, :] ** 2 * p_lin * f ** 2 * mu ** 4)}
    # the isotropic closed forms that used to be stored instead
    pk_res = np.asarray(ept.pk_nw) + np.asarray(ept.pk_w) * np.exp(
        -float(ept.sigma2_bao) * k ** 2)
    closed = {0: -k ** 2 * pk_res,
              2: -k ** 2 * pk_res * f * (2.0 / 3.0),
              4: -k ** 2 * pk_res * (8.0 / 35.0) * f ** 2}

    for ell in (0, 2, 4):
        got = np.asarray(getattr(ept, f"Pk_ctr{ell}"))
        scale = np.max(np.abs(want[ell][m]))
        rel = np.max(np.abs(got[m] - want[ell][m])) / scale
        assert rel < 1e-12, (
            f"[{label}] Pk_ctr{ell} is not the GL projection: {rel:.3e}")
        # and the two are genuinely different, so the assertion above has teeth
        gap = np.median(np.abs(want[ell][m] - closed[ell][m])) / scale
        assert gap > 1e-4, (
            f"[{label}] GL and closed-form Pk_ctr{ell} agree to {gap:.3e}; "
            "the test cannot distinguish them at this cosmology")


@pytest.mark.parametrize("label,pk_scale,f_scale", _MU_POINTS)
def test_ifg2_multipoles_carry_the_p_lo_rescaling(label, pk_scale, f_scale):
    """P_IFG2_0b1/0/2 must be the GL projections of (Pnw + Pw Exp)/Pbin * P_IFG2.

    nonlinear_pt.c:5318-5320 forms p_lo = (Pnw + Pw Exp)/Pbin and multiplies
    P_IFG2 (which carries a factor Pbin, :4987) by it before projecting at
    :5356-5358.  clax used P_IFG2 * {1, f/3, 2f/3}, i.e. p_lo == 1.  The factor
    multiplies (2 bG2 + 0.8 bGamma3), so the error propagates into bGamma3.
    """
    ept = _mu_ept(label, pk_scale, f_scale)
    mu, w, _Sig, _Exp, p_lin = _gl_pieces(ept)
    k = np.asarray(ept.kh)
    f = float(ept.f)
    m = _band(ept)

    Pbin = np.asarray(ept.pk_nw) + np.asarray(ept.pk_w) * np.exp(
        -float(ept.sigma2_bao) * k ** 2)                       # CLASS-PT Pbin
    IFG2_in = (p_lin / Pbin[None, :]) * np.asarray(ept.Pk_IFG2)[None, :]
    want = {"Pk_IFG2_0b1": _proj(w, mu, 0, IFG2_in),
            "Pk_IFG2_0":   _proj(w, mu, 0, IFG2_in * f * mu ** 2),
            "Pk_IFG2_2":   _proj(w, mu, 2, IFG2_in * f * mu ** 2)}
    closed = {"Pk_IFG2_0b1": np.asarray(ept.Pk_IFG2),
              "Pk_IFG2_0":   np.asarray(ept.Pk_IFG2) * f / 3.0,
              "Pk_IFG2_2":   np.asarray(ept.Pk_IFG2) * 2.0 * f / 3.0}

    for name in want:
        got = np.asarray(getattr(ept, name))
        scale = np.max(np.abs(want[name][m]))
        rel = np.max(np.abs(got[m] - want[name][m])) / scale
        assert rel < 1e-12, f"[{label}] {name} is not the GL projection: {rel:.3e}"
        gap = np.median(np.abs(want[name][m] - closed[name][m])) / scale
        assert gap > 1e-4, (
            f"[{label}] rescaled and un-rescaled {name} agree to {gap:.3e}; "
            "the test cannot distinguish them at this cosmology")


@pytest.mark.parametrize("label,pk_scale,f_scale", _MU_POINTS)
def test_pk_gg_l2_tree_carries_b1_squared_dd(label, pk_scale, f_scale):
    """The l=2 tree is Pk_2_vv + b1 Pk_2_vd + b1^2 Pk_2_dd.

    classy.pyx:4921 is pm[18] + pm[24] + b1 pm[19] + b1 pm[25] + b1^2 pm[26],
    and pm[26] is P1loopdd_ap_ir, which nonlinear_pt.c:4529 OPENS with p_tree --
    so the dd TREE rides on pm[26] with a b1^2 in front of it.  clax dropped it,
    on the claim that pm[26] is "a 1-loop contribution, not tree".  It is not
    numerically inert: with IR resummation on, Sigmatot(mu) is mu-dependent
    through f, so Pk_2_dd is non-zero even at hratio = Dratio = 1.
    """
    ept = _mu_ept(label, pk_scale, f_scale)
    b1, b2, bG2, bG3, cs2, b4 = 1.9, -0.8, 0.35, -0.2, 7.0, 60.0
    m = _band(ept)
    kh = np.asarray(ept.kh)
    f = float(ept.f)

    got = np.asarray(pk_gg_l2(ept, b1, b2, bG2, bG3, cs2=cs2, b4=b4))
    want = (np.asarray(ept.Pk_2_vv) + b1 * np.asarray(ept.Pk_2_vd)
            + b1 ** 2 * np.asarray(ept.Pk_2_dd)
            + np.asarray(ept.Pk_2_vv1) + b1 * np.asarray(ept.Pk_2_vd1)
            + b1 ** 2 * np.asarray(ept.Pk_2_dd1)
            + b1 * b2 * np.asarray(ept.Pk_2_b1b2) + b2 * np.asarray(ept.Pk_2_b2)
            + b1 * bG2 * np.asarray(ept.Pk_2_b1bG2) + bG2 * np.asarray(ept.Pk_2_bG2)
            + (2.0 * bG2 + 0.8 * bG3) * np.asarray(ept.Pk_IFG2_2)
            + 2.0 * cs2 * np.asarray(ept.Pk_ctr2)
            + f ** 2 * b4 * kh ** 2
            * (f ** 2 * 70.0 + 165.0 * f * b1 + 99.0 * b1 ** 2)
            * (4.0 / 693.0) * (35.0 / 8.0) * np.asarray(ept.Pk_ctr4))
    rel = np.max(np.abs(got[m] - want[m])) / np.max(np.abs(want[m]))
    assert rel < 1e-12, f"[{label}] pk_gg_l2 is not classy.pyx:4921: {rel:.3e}"

    # Pk_2_dd is not zero, so dropping it really does change the answer.
    dd = b1 ** 2 * np.asarray(ept.Pk_2_dd)
    gap = np.max(np.abs(dd[m])) / np.max(np.abs(want[m]))
    assert gap > 1e-4, (
        f"[{label}] b1^2 Pk_2_dd is only {gap:.3e} of the quadrupole; "
        "the test cannot see the missing tree term")


@pytest.mark.parametrize("label,pk_scale,f_scale", _MU_POINTS)
def test_pk_gg_l4_tree_carries_its_b1_powers(label, pk_scale, f_scale):
    """The l=4 tree is Pk_4_vv + b1 Pk_4_vd + b1^2 Pk_4_dd.

    classy.pyx:4931 is pm[20] + pm[27] + b1 pm[28] + b1^2 pm[29]; pm[28]/pm[29]
    are P1loopvd_ap_ir / P1loopdd_ap_ir, which open with the vd/dd tree
    (nonlinear_pt.c:4529-4539), so the tree carries the same b1 powers as the
    loop.  clax summed the three trees with no b1 at all.
    """
    ept = _mu_ept(label, pk_scale, f_scale)
    b1, b2, bG2, bG3, cs4, b4 = 1.9, -0.8, 0.35, -0.2, -3.0, 60.0
    m = _band(ept)
    kh = np.asarray(ept.kh)
    f = float(ept.f)

    got = np.asarray(pk_gg_l4(ept, b1, b2, bG2, bG3, cs4=cs4, b4=b4))
    want = (np.asarray(ept.Pk_4_vv) + b1 * np.asarray(ept.Pk_4_vd)
            + b1 ** 2 * np.asarray(ept.Pk_4_dd)
            + np.asarray(ept.Pk_4_vv1) + b1 * np.asarray(ept.Pk_4_vd1)
            + b1 ** 2 * np.asarray(ept.Pk_4_dd1)
            + b2 * np.asarray(ept.Pk_4_b2) + bG2 * np.asarray(ept.Pk_4_bG2)
            + 2.0 * cs4 * np.asarray(ept.Pk_ctr4)
            + f ** 2 * b4 * kh ** 2
            * (f ** 2 * 210.0 + 390.0 * f * b1 + 143.0 * b1 ** 2)
            * (8.0 / 5005.0) * (35.0 / 8.0) * np.asarray(ept.Pk_ctr4))
    rel = np.max(np.abs(got[m] - want[m])) / np.max(np.abs(want[m]))
    assert rel < 1e-12, f"[{label}] pk_gg_l4 is not classy.pyx:4931: {rel:.3e}"

    # the b1 powers are not cosmetic: the unweighted sum is a different spectrum
    flat = (np.asarray(ept.Pk_4_vv) + np.asarray(ept.Pk_4_vd)
            + np.asarray(ept.Pk_4_dd))
    weighted = (np.asarray(ept.Pk_4_vv) + b1 * np.asarray(ept.Pk_4_vd)
                + b1 ** 2 * np.asarray(ept.Pk_4_dd))
    gap = np.max(np.abs(flat[m] - weighted[m])) / np.max(np.abs(want[m]))
    assert gap > 1e-4, (
        f"[{label}] weighted and unweighted l=4 trees differ by only {gap:.3e}")


@pytest.mark.parametrize("label,pk_scale,f_scale", _MU_POINTS)
def test_pk_gg_l4_omits_the_two_rows_classy_omits(label, pk_scale, f_scale):
    """pk_gg_l4 must not carry b1 b2 P_4_b1b2 + b1 bG2 P_4_b1bG2.

    The C loop fills rows 40 and 41 (nonlinear_pt.c:5350-5351) and classy
    unpacks them (classy.pyx:4734-4735), but classy.pyx:4925-4931 never uses
    them.  Both rows vanish here by <L2 L4> = 0 -- they are only populated once
    mutrue != mu, i.e. under an Alcock-Paczynski remap, which this branch does
    not expose -- so the leaves are OVERWRITTEN with a large synthetic signal
    and the accessor must not respond to it.  That keeps the test sharp without
    needing the AP path, and it is exactly why a no-AP comparison against
    CLASS-PT could never have caught the two extra terms.
    """
    ept = _mu_ept(label, pk_scale, f_scale)
    b1, b2, bG2, bG3 = 1.9, -0.8, 0.35, -0.2
    m = _band(ept)

    base = np.asarray(pk_gg_l4(ept, b1, b2, bG2, bG3))
    scale = np.max(np.abs(base[m]))
    spike = jnp.asarray(np.full_like(np.asarray(ept.kh), scale))
    poked = dataclasses.replace(ept, Pk_4_b1b2=spike, Pk_4_b1bG2=spike)
    # a live pm[40]/pm[41] term would shift the spectrum by
    # (b1*b2 + b1*bG2) * scale = 0.86 x its own peak, so this cannot hide
    assert abs(b1 * b2 + b1 * bG2) > 0.5
    got = np.asarray(pk_gg_l4(poked, b1, b2, bG2, bG3))
    assert np.array_equal(got, base), (
        f"[{label}] pk_gg_l4 still reads Pk_4_b1b2 / Pk_4_b1bG2; classy.pyx:"
        f"4925-4931 has no pm[40]/pm[41] term (max shift "
        f"{np.max(np.abs(got[m] - base[m])) / scale:.3e} of the hexadecapole)")
