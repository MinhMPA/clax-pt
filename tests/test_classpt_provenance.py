"""Spec §4.2 / Ruling 13 + Ruling 14 + Ruling 15 (Task A4 fix rounds 1-3):
drift-bounded, band-resolved, explained provenance gate.

Root cause (Ruling 13): the legacy npz (reference_data/classpt_z0.38_fullrange
.npz) predates ~/CLASS-PT (the re-port onto class_public 3.3.4, cloned
2026-05-06) and was produced by the classic CLASS-PT on a CLASS v2.6.3 base,
which no longer exists to re-run. Pinning the CLASS 2.6.x-era defaults the
legacy run used (N_ur=3.046, recombination=recfast, via --class-extra) into
reference_data/classpt/legacy_fiducial/z0.380_ap_omfid0.31_m_legacyinputs.npz
cuts the drift from CLASS-3.3.4-default levels (fz 3.74e-08, pk_lin up to
4.8e-04) to fz=1.11e-10, pk_lin<=8.71e-05. That residual is drift, not a bug:
1e-6 bit-for-bit reproduction is unattainable across the 2.6.3->3.3.4 gap.

RETRACTION (Ruling 14), refined by Ruling 15: fix round 1 reported the
full-range worst cases as a "PT-stage amplification" (pk_mult rows /
real-space spectra drifting 11-239x more than pk_lin). That CONCLUSION does
not survive a check of *where* the drift sits in k and stays retracted here
and from every docstring below. But the full-range worst cases do not all
share one cause, and round 2's blanket "P22/P13 cancel physically" was
wrong for one of them (Ruling 15(a)): pk_mm_real/pk_gg_real/pk_gm_real at
k=3.90 h/Mpc genuinely sit in the P22/P13-cancellation regime (a real
one-loop residual against a much larger tree term); pk_gg_l0/l2/l4 at
k=100 h/Mpc show the same high-k CLASS-core drift pk_lin itself has there;
but pk_mult row 1 at k=53.5 h/Mpc is neither -- it is a pure 22-type
integral with no P13 partner, sitting at 85% of its own peak (no
cancellation), and its large residual is a large-constant-offset
subtraction (`large_for_logs_big`) amplifying a small stored-value drift by
85x -- see test_pk_mult_rows_match for the arithmetic. None of these three
mechanisms is an in-window PT-stage amplification of physical signal:
restricted to the comparison window the spec actually validates against,
k <= 0.3 h/Mpc, real-space spectra and the physically dominant pk_mult rows
track pk_lin's in-window drift to within an order of magnitude (see
test_real_space_spectra_match, test_pk_mult_rows_match), not 11-239x.

Every quantity below therefore gets a PRIMARY assertion inside the window
and a SECONDARY assertion over the full range, each with its own tolerance
= ceil_to_1sf(2 x measured worst) and its own named cause, quoted in the
owning test's docstring. Nothing is absorbed silently. The PRIMARY metric
follows Ruling 13 exactly (Ruling 15(b)): relative for a spectrum that is
positive-definite over the window, max|delta|/max|ref| -- normalised over
that SAME window, not the full range -- only for one that changes sign
inside it (pk_gg_l4).

Exempt from the multi-cosmology rule: this is a single-file provenance check.
"""
import json

import numpy as np
import pytest

from scripts import classpt_assembly as ca
from scripts import validation_cosmologies as vc

LEGACY = vc.REPO_ROOT / "reference_data" / "classpt_z0.38_fullrange.npz"
LEGACYINPUTS = vc.REFERENCE_ROOT / "legacy_fiducial" / "z0.380_ap_omfid0.31_m_legacyinputs.npz"
DEFAULT_INPUTS = vc.REFERENCE_ROOT / "legacy_fiducial" / "z0.380_ap_omfid0.31_m.npz"

# Ruling 14 item 1: spec §4.7 "over k ≤ 0.3 h/Mpc" (design doc line 160) and
# §5.3 "the comparison window k ≤ 0.3 h/Mpc" (line 232) -- the physically
# meaningful one-loop PT regime this gate's PRIMARY assertions are scoped to.
KH_WINDOW_MAX = 0.3

# Ruling 14 item 3: split pk_mult's 48 valid rows by each row's own max|ref|
# (against the legacy file) relative to the largest row's max|ref| (row 17,
# 5.4576e4); rows below 1e-2x that (5.4576e2) are grouped separately so a
# near-zero row's spline noise cannot set the tolerance for the physically
# dominant rows. Measured once against the legacy reference file.
PM_TINY_ROWS = frozenset({10, 11, 12, 13, 21, 29, 31, 33, 38, 39, 40, 41,
                          42, 43, 44, 45, 46, 47})


@pytest.fixture(scope="module")
def pair():
    # A provenance gate that silently skips when its inputs are missing is a
    # silent pass (campaign rule): both files must exist, or this is an error.
    for p in (LEGACY, LEGACYINPUTS):
        if not p.exists():
            pytest.fail(f"ERROR missing {p}")
    return np.load(LEGACY), np.load(LEGACYINPUTS)


@pytest.fixture(scope="module")
def default_inputs():
    if not DEFAULT_INPUTS.exists():
        pytest.fail(f"ERROR missing {DEFAULT_INPUTS}")
    return np.load(DEFAULT_INPUTS)


def _window(kh):
    return kh <= KH_WINDOW_MAX


def _rel(a, b, mask=None):
    """Max relative residual, optionally restricted to `mask`. The
    max(|b|, 1e-300) floor only prevents a literal 0/0 division; it is not
    a "matching" special case (Ruling 15 minor 6 -- Ruling 12's
    exactly-zero-in-both guard applies to `pk_mult`, which is compared with
    `_mdmr` below, not `_rel`). An entry zero in one file but not the other
    yields a large finite ratio here and is not masked -- that is a real
    mismatch and must fail loudly, not be swallowed."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    r = np.abs(a - b) / np.maximum(np.abs(b), 1e-300)
    return np.max(r[mask]) if mask is not None else np.max(r)


def _mdmr(a, b, mask=None, ref_mask=None):
    """max|delta| / max|ref|, for arrays that cross zero or involve a
    large-constant-subtraction cancellation. `mask` restricts where the
    max|delta| search looks. `ref_mask` independently restricts where
    max|ref| is computed; it defaults to the FULL array (the pk_mult-row
    convention, Ruling 14, so in-window and full-range values of the same
    row stay directly comparable). Pass `ref_mask=mask` for a fully
    window-scoped metric (Ruling 15(b)): pk_gg_l4 changes sign inside the
    comparison window, so a window-scoped max|ref| -- not the full-range
    one, which for pk_gg_l4 sits at k=100, ~4360x larger -- is the only
    version of this metric that is physically meaningful there."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.abs(a - b)
    ref_b = b[ref_mask] if ref_mask is not None else b
    ref = max(np.max(np.abs(ref_b)), 1e-300)
    if mask is not None:
        d = d[mask]
    return np.max(d) / ref


def test_inputs_are_the_legacy_inputs(pair):
    _, li = pair
    prm = json.loads(str(li["params_json"]))
    assert prm["A_s"] == 2.0989e-9 and "N_ncdm" not in prm and "YHe" not in prm
    assert prm["cb"] == "No" and prm["AP"] == "Yes" and prm["Omfid"] == "0.31"
    assert str(li["kh_convention"]) == "h/Mpc"
    # Ruling 13: this file additionally pins the CLASS-core (2.6.x-era)
    # defaults the legacy run used, via --class-extra (root-cause fix).
    assert prm["N_ur"] == 3.046 and prm["recombination"] == "recfast"


def test_grid_and_background_match(pair):
    """k_h, h, fz are not k-dependent spectra, so Ruling 14's windowing does
    not apply to them (they are scalars/grids, not functions of k)."""
    old, li = pair
    # Ruling 12: k_h is NOT bit-identical between the legacy and classpt envs
    # (measured max|old/new - 1| = 2.0e-15) -- grids agree to ULP, not bit-exactly:
    # numpy versions differ between the legacy and classpt envs.
    assert np.max(np.abs(old["k_h"] / li["k_h"] - 1.0)) < 1e-13
    # h is a literal echoed CLASS input, not computed: expect bit-exact (measured 0.0).
    assert abs(float(old["h"]) - float(li["h"])) < 1e-12
    # Ruling 13: fz tolerance fixed at 1e-9 (measured 1.1082e-10 here, vs
    # 3.7409e-08 for A3's CLASS-3.3.4-default file -- see
    # test_default_inputs_drift_more_than_legacy_inputs).
    assert abs(float(old["fz"]) - float(li["fz"])) / abs(float(old["fz"])) < 1e-9


def test_pk_lin_matches(pair):
    """Linear power spectrum: CLASS-core-only, cb/PT-independent quantity.
    PRIMARY in-window (k<=0.3 h/Mpc): tolerance 9e-5 = ceil_to_1sf(2x),
    measured 4.3123e-05 at k_h=0.1285.
    SECONDARY full-range: tolerance 2e-4, measured 8.7103e-05 at k_h=100 --
    ordinary high-k CLASS-core (2.6.3->3.3.4) drift, unrelated to PT."""
    old, li = pair
    win = _window(li["k_h"])
    assert _rel(li["pk_lin"], old["pk_lin"], win) < 9e-5
    assert _rel(li["pk_lin"], old["pk_lin"]) < 2e-4


def test_pk_mult_rows_match(pair):
    """Ruling 14 retracts round 1's "PT-stage amplification" CONCLUSION.
    Ruling 15(a) corrects round 2's replacement cause for row 1, which was
    itself wrong: "P22/P13 cancel physically" does not apply here. `P_Id2d2`
    is a PURE 22-type integral with no P13 partner to cancel against
    (nonlinear_pt.c:4853 `P_Id2d2[j] = fabs(k^3*f22_Id2d2[j]
    - k0^3*2*f22_Id2d2_real[0] + epsilon_for_logs)` -- only f22_* appears),
    and nothing is cancelling at k=53.5 h/Mpc: the row sits at 85% of its
    own peak there (pm[1]=-1.1901e+04 vs. row max|ref|=1.3948e+04 at
    k=47.73), not near a zero.

    The actual mechanism is arithmetic, not physical: `pm[1]` is recovered
    as `large_for_logs_big - raw` (nonlinear_pt.c:3364
    `large_for_logs_big = 1000000.`; classy.pyx:4675-76
    `pk_mult[1] = -raw_pk[1] + large_b`). At k=53.5, the stored
    (offset-shifted) quantity is old=1.011901e+06 vs. li=1.011611e+06 -- a
    2.8692e-04 relative drift of the STORED value, only ~3x pk_lin's own
    high-k drift (8.7103e-05 @ k=100; the exact FFTLog-settings origin of
    the residual 2.9e-4 is not recoverable without the classic,
    pre-2026-05, CLASS-PT build) -- but recovering `pm[1]` (~-1.19e4) via
    that 1e6-offset subtraction amplifies the drift by
    |1e6-pm[1]|/|pm[1]| = 85.0x: 85.0 x 2.8692e-04 = 0.024395, matching the
    measured local relative residual there exactly. This is
    out-of-window only: in-window, row 1's mdmr is 3.77e-06.

      * big rows (physically dominant): worst is row 1 (pk_Id2d2, the b2^2
        loop term) at 0.020815 @ k=53.5 h/Mpc (large_for_logs_big
        amplification above -- not a cancellation).
      * tiny rows (max|ref| < 1e-2x the largest row's, see PM_TINY_ROWS):
        worst is row 45 at 9.682e-04 @ k=0.401, whose own max|ref| is only
        1.389e-03 -- ordinary spline noise on a near-zero row, not physics.
    Inside k <= 0.3 h/Mpc both groups are two orders of magnitude tighter.

    PRIMARY in-window, ceil_to_1sf(2x):
      big rows:  8e-4 (measured 3.6889e-04, row 25 @ k=0.181)
      tiny rows: 2e-3 (measured 9.6155e-04, row 47 @ k=0.082, max|ref|=6.39e-04)
    SECONDARY full-range, ceil_to_1sf(2x):
      big rows:  5e-2 (measured 0.020815, row 1 @ k=53.5)
      tiny rows: 2e-3 (measured 9.6819e-04, row 45 @ k=0.401, max|ref|=1.39e-03)
    """
    old, li = pair
    rows = range(*ca.PM_ROWS_VALID.indices(96))
    win = _window(li["k_h"])
    tols = {("big", "in"): 8e-4, ("big", "full"): 5e-2,
            ("tiny", "in"): 2e-3, ("tiny", "full"): 2e-3}
    bad = {}
    for r in rows:
        group = "tiny" if r in PM_TINY_ROWS else "big"
        v_in = _mdmr(li["pk_mult"][r], old["pk_mult"][r], win)
        v_full = _mdmr(li["pk_mult"][r], old["pk_mult"][r])
        if v_in >= tols[(group, "in")]:
            bad[(r, "in-window")] = v_in
        if v_full >= tols[(group, "full")]:
            bad[(r, "full-range")] = v_full
    assert not bad, f"ERROR pk_mult rows beyond tolerance: {bad}"


def test_real_space_spectra_match(pair):
    """pk_mm_real, pk_gg_real, pk_gm_real: relative metric (positive-definite
    here -- fiducial bias b1=2, all other biases 0, so pk_gg_real =
    4*pk_mm_real and pk_gm_real = 2*pk_mm_real exactly and all three share
    the same measured relative drift).

    Ruling 14 (retracts round 1's finding): the full-range worst case sits
    at k=3.90 h/Mpc, in the P22/P13-cancellation regime -- not a PT-stage
    amplification. Inside k <= 0.3 h/Mpc the drift tracks pk_lin's
    in-window drift (test_pk_lin_matches, 4.3123e-05) almost exactly.

    PRIMARY in-window: 8e-5 = ceil_to_1sf(2x), measured 3.6779e-05 @ k=0.1285.
    SECONDARY full-range: 7e-3 = ceil_to_1sf(2x), measured 3.3019e-03 @ k=3.90.
    """
    old, li = pair
    win = _window(li["k_h"])
    for key, oldkey in [("pk_mm_real", "pk_mm_real"), ("pk_gg_real", "pk_gg_real"),
                        ("pk_gm_real", "pk_mg_real")]:
        assert _rel(li[key], old[oldkey], win) < 8e-5, f"{key} in-window"
        assert _rel(li[key], old[oldkey]) < 7e-3, f"{key} full-range"


def test_multipoles_match(pair):
    """RSD multipoles. Ruling 15(b) (fix round 3) corrects round 2's PRIMARY
    metric: `_mdmr` normalised by max|ref| over the FULL range even for the
    in-window assertion, which for pk_gg_l* made the in-window numbers
    meaningless -- pk_gg_l0/l2/l4's full-range max|ref| sits at k=100
    (counterterm-dominated), 61x/431x/4360x larger than their in-window
    max|ref|. Ruling 13's actual rule: relative metric for a spectrum that
    is positive-definite over the window, max|delta|/max|ref| (window-
    scoped, not full-range) only for one that changes sign inside it.

    In-window positivity (both files, k<=0.3 h/Mpc): pk_mm_l0/l2/l4 and
    pk_gg_l0/l2 are strictly positive there (min values 17-736); pk_gg_l4 is
    NOT (crosses zero, min approx -225) -- so it alone gets the window-
    scoped max|delta|/max|ref| metric instead of relative.

    PRIMARY in-window, ceil_to_1sf(2x measured):
      pk_mm_l0 relative   8e-5 (3.6277e-05 @ k=0.285)
      pk_mm_l2 relative   3e-4 (1.2428e-04 @ k=0.181)
      pk_mm_l4 relative   5e-4 (2.4989e-04 @ k=0.181)
      pk_gg_l0 relative   7e-5 (3.0338e-05 @ k=0.285)
      pk_gg_l2 relative   4e-4 (1.5647e-04 @ k=0.181)
      pk_gg_l4 window-mdmr 3e-4 (1.0536e-04 @ k=0.181)
    SECONDARY full-range, max|delta|/max|ref| (unchanged from round 2 --
    all six cross zero somewhere over the full range, so a plain relative
    metric is not meaningful there):
      pk_mm_l0/l2/l4: worst case already lies inside the window, so
      full-range == in-window mdmr exactly (2e-5/3e-5/2e-4, from 8.9901e-06
      / 1.4359e-05 / 5.1835e-05) -- strictly stronger than any window bound
      since the divisor is band-independent.
      pk_gg_l0/l2/l4: worst sits at k=100 h/Mpc (8.7298e-05, all three --
      same high-k CLASS-core drift as pk_lin's full-range case, not a
      PT-stage effect), tol 2e-4 for all three.
    """
    old, li = pair
    win = _window(li["k_h"])
    rel_tols = {"pk_mm_l0": 8e-5, "pk_mm_l2": 3e-4, "pk_mm_l4": 5e-4,
                "pk_gg_l0": 7e-5, "pk_gg_l2": 4e-4}
    for key, tol in rel_tols.items():
        assert _rel(li[key], old[key], win) < tol, f"{key} in-window"
    assert _mdmr(li["pk_gg_l4"], old["pk_gg_l4"], win, win) < 3e-4, "pk_gg_l4 in-window"

    mm_full_tols = {"pk_mm_l0": 2e-5, "pk_mm_l2": 3e-5, "pk_mm_l4": 2e-4}
    for key, tol in mm_full_tols.items():
        assert _mdmr(li[key], old[key]) < tol, f"{key} full-range"
    for key in ("pk_gg_l0", "pk_gg_l2", "pk_gg_l4"):
        assert _mdmr(li[key], old[key]) < 2e-4, f"{key} full-range"


def test_galaxy_multipoles_match_in_legacy_kh_convention(pair):
    """New files store pk_gg_* with kh in h/Mpc; the legacy file used
    LEGACY_KH_CONVENTION. Re-assemble the pk_gg_l* from the legacyinputs
    file's own raw pk_mult (via the classpt_assembly twin) and compare --
    this is the only place the two conventions meet, and it end-to-end
    exercises the twin + LEGACY_KH_CONVENTION on top of test_multipoles_
    match's direct comparison of classy's stored accessor outputs. Same
    corrected metrics as test_multipoles_match's pk_gg_l* (Ruling 15(b)):
    relative in-window for pk_gg_l0/l2 (positive-definite over the window),
    window-scoped max|delta|/max|ref| in-window for pk_gg_l4 (changes sign
    in-window), full-range max|delta|/max|ref| (2e-4) for all three as
    SECONDARY. The twin's self-consistency error is ~1e-10
    (tests/test_classpt_assembly.py), negligible next to these -- measured
    values here are identical to test_multipoles_match's to 6 significant
    figures."""
    old, li = pair
    if ca.LEGACY_KH_CONVENTION is None:
        pytest.xfail("legacy pk_gg_* non-reproducible (see test_classpt_assembly)")
    h, fz = float(li["h"]), float(li["fz"])
    kh = li["k_h"] if ca.LEGACY_KH_CONVENTION == "h/Mpc" else li["k_h"] * h
    win = _window(kh)
    bias = json.loads(str(li["bias_json"]))
    out = ca.assemble_from_pm(li["pk_mult"], h, fz, kh, bias, ca.pd2d2_0(li["pk_mult"][14] * h**3, kh))
    assert _rel(out["pk_gg_l0"], old["pk_gg_l0"], win) < 7e-5, "pk_gg_l0 in-window"
    assert _rel(out["pk_gg_l2"], old["pk_gg_l2"], win) < 4e-4, "pk_gg_l2 in-window"
    assert _mdmr(out["pk_gg_l4"], old["pk_gg_l4"], win, win) < 3e-4, "pk_gg_l4 in-window"
    for key in ("pk_gg_l0", "pk_gg_l2", "pk_gg_l4"):
        assert _mdmr(out[key], old[key]) < 2e-4, f"{key} full-range"


def test_default_inputs_drift_more_than_legacy_inputs(pair, default_inputs):
    """Ruling 13 item 4 (band-resolved per Ruling 14): proves the fz/pk_lin
    drift is attributable to CLASS-core defaults (N_ur, recombination), not
    build noise, and catches an accidental same-input regeneration of the
    legacyinputs file. A3's default-input file (CLASS-3.3.4 defaults:
    N_ur=3.044, hyrec) must drift MORE from the legacy (CLASS-2.6.3-based)
    run than the legacyinputs file (N_ur=3.046, recombination=recfast) does.

    fz (k-independent): li=1.108e-10 vs df=3.741e-08 (337.6x).
    pk_lin in-window (PRIMARY): li=4.312e-05 vs df=3.846e-04 (8.9x).
    pk_lin full-range (SECONDARY): li=8.710e-05 vs df=4.758e-04 (5.5x)."""
    old, li = pair
    df = default_inputs
    win = _window(li["k_h"])
    fz_li = abs(float(li["fz"]) - float(old["fz"])) / abs(float(old["fz"]))
    fz_df = abs(float(df["fz"]) - float(old["fz"])) / abs(float(old["fz"]))
    assert fz_df > fz_li, f"fz: df={fz_df:.3e} not > li={fz_li:.3e}"
    r_li_in = _rel(li["pk_lin"], old["pk_lin"], win)
    r_df_in = _rel(df["pk_lin"], old["pk_lin"], win)
    assert r_df_in > r_li_in, f"pk_lin in-window: df={r_df_in:.3e} not > li={r_li_in:.3e}"
    r_li_full = _rel(li["pk_lin"], old["pk_lin"])
    r_df_full = _rel(df["pk_lin"], old["pk_lin"])
    assert r_df_full > r_li_full, f"pk_lin full-range: df={r_df_full:.3e} not > li={r_li_full:.3e}"
