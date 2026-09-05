"""Shared helpers for the clax-pt vs CLASS-PT validation campaign
(Part 2, Tasks C1-C3): spectra names, comparison window, thresholds, the
error metric, reference-file loading and the JSONL error log that
scripts/summarize_ept_validation.py renders into the report.

Thresholds live HERE and nowhere else (spec §4.7; Part 2 Global
Constraints: they only tighten -- C4 ratchets them to >= 2x the measured
worst case).
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from scripts import validation_cosmologies as vc

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "test_logs" / "ptval"
ERROR_LOG = LOG_DIR / "errors.jsonl"

# Report order. Real-space first (the AP/RSD-free sanity row), then
# matter multipoles, then galaxy multipoles.
SPECTRA = ("pk_mm_real", "pk_gg_real", "pk_gm_real",
           "pk_mm_l0", "pk_mm_l2", "pk_mm_l4",
           "pk_gg_l0", "pk_gg_l2", "pk_gg_l4")

K_MAX_COMPARE = 0.3   # h/Mpc, spec §4.3
NSIDE = 10            # CLASS-PT's Nside: the first 10 grid points are FFTLog-edge garbage (ref §7)


def window(k_h: np.ndarray) -> np.ndarray:
    """Comparison mask: grid points [NSIDE:] with k <= K_MAX_COMPARE (spec §4.3)."""
    k_h = np.asarray(k_h)
    mask = np.zeros(k_h.shape, dtype=bool)
    mask[NSIDE:] = k_h[NSIDE:] <= K_MAX_COMPARE
    return mask


# Spec §4.7: 1% for l=0, l=2 and real space; 2% for l=4 (small, noisy).
THRESHOLDS = {name: 0.01 for name in SPECTRA}
THRESHOLDS["pk_mm_l4"] = 0.02
THRESHOLDS["pk_gg_l4"] = 0.02

# Spec §7 Phase 4 / §9 seams (e2e layer). pk_lin is pointwise on the window;
# pk_lin_tail is pointwise on 0.3 < k <= 3 h/Mpc (the P22/P13 UV region --
# loose because clax's spline clamps beyond pt_k_max_cl, see C0 docstring).
SEAM_THRESHOLDS = {"hratio": 1e-4, "Dratio": 1e-4, "H_z": 1e-4, "rs_d": 1e-3,
                   "f": 1e-3, "pk_lin": 1e-3, "pk_lin_tail": 3e-2}


def rel(a, b) -> float:
    """max|a - b| / max|b| -- the campaign metric (spec §4.7). Scale-relative
    so that zero crossings of l=2/l=4 do not blow up a pointwise ratio.
    Floors the denominator at 1e-300 (fix round 1, F1) so an all-zero `b`
    (e.g. an unmeasured pk_mult row) returns a large-but-finite number
    instead of NaN -- B7's original `_rel` verbatim."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


def _err_and_k(a, b, k_h, mask):
    a = np.asarray(a, dtype=float)[mask]
    b = np.asarray(b, dtype=float)[mask]
    k = np.asarray(k_h, dtype=float)[mask]
    diff = np.abs(a - b)
    i = int(np.argmax(diff))
    return {"err": float(diff[i] / np.max(np.abs(b))), "k": float(k[i])}


def compare_spectra(got: dict, ref: dict, k_h) -> dict[str, dict]:
    """{name: {"err", "k"}} over SPECTRA present in both dicts, on window(k_h)."""
    mask = window(k_h)
    return {name: _err_and_k(got[name], ref[name], k_h, mask)
            for name in SPECTRA if name in got and name in ref}


def compare_rows(pm_got, pm_ref, k_h) -> list[dict]:
    """Per-row diagnostics for the 48 pk_mult rows: [{"row", "err", "k"}]."""
    mask = window(k_h)
    out = []
    for i in range(min(len(pm_got), len(pm_ref))):
        if np.max(np.abs(np.asarray(pm_ref[i])[mask])) == 0.0:
            out.append({"row": i, "err": 0.0, "k": float("nan")})
            continue
        out.append({"row": i, **_err_and_k(pm_got[i], pm_ref[i], k_h, mask)})
    return out


def failures(errs: dict[str, dict], thresholds: dict[str, float]) -> list[str]:
    """One greppable line per violated threshold: 'pk_gg_l4 2.31% > 2.00% at k=0.297'."""
    out = []
    for name, rec in errs.items():
        thr = thresholds.get(name)
        if thr is not None and rec["err"] > thr:
            out.append(f"{name} {100 * rec['err']:.2f}% > {100 * thr:.2f}% at k={rec['k']:.3f}")
    return out


def load_reference(case: str, z: float, **kw) -> dict | None:
    path = vc.reference_path(case, z, **kw)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as npz:
        return {key: (npz[key].item() if npz[key].shape == () else np.asarray(npz[key]))
                for key in npz.files}


def require_reference(case: str, z: float, **kw) -> dict:
    import pytest
    ref = load_reference(case, z, **kw)
    if ref is None:
        pytest.skip(f"reference missing: {vc.reference_path(case, z, **kw).name} "
                    "-- run slurm/classpt-refgen.sbatch (Part 1a, A5)")
    return ref


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # detached tarball, no git -- the record still gets written
        return "unknown"


def _sanitize(obj):
    """Recursively replace non-finite floats (NaN/+-Inf) with None.

    compare_rows and the delta diagnostics use `float("nan")` as a "no k"
    placeholder (an all-zero row, or a scalar delta with no associated k);
    bare NaN is not standard JSON, so log_record sanitises before
    json.dumps(..., allow_nan=False) rather than emitting a token a strict
    parser (C3's summariser) would reject."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def log_record(*, layer: str, case: str, z: float, preset: str, errors: dict,
               seams: dict | None = None, extra: dict | None = None) -> None:
    """Append one JSON line to ERROR_LOG. Keys: ts (ISO-8601 UTC), git_sha,
    layer ('stage' | 'e2e' | 'grad'), case, z, preset, errors
    ({spectrum: {"err", "k"}}), seams ({name: residual}), extra (free).
    Non-finite floats (e.g. "k": float("nan") for an all-zero row or a
    k-less delta diagnostic) are sanitised to JSON null via _sanitize before
    json.dumps(..., allow_nan=False), so every line is standard JSON for
    C3's summariser -- not Python-only bare NaN."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rec = {"ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
           "git_sha": _git_sha(), "layer": layer, "case": case, "z": float(z),
           "preset": preset, "errors": errors, "seams": seams or {}, "extra": extra or {}}
    rec = _sanitize(rec)
    with ERROR_LOG.open("a") as fh:
        fh.write(json.dumps(rec, default=float, allow_nan=False) + "\n")


# ---------------------------------------------------------------------------
# B7 helpers (moved verbatim from tests/test_ept_assembly.py; that file now
# imports them). The 48-row layout mirrors CLASS-PT's pk_mult (ref §9).
# ---------------------------------------------------------------------------

BIAS_KEYS = ("b1", "b2", "bG2", "bGamma3", "cs0", "cs2", "cs4", "cs", "Pshot", "b4")

# The ten RSD bias rows 30-39 of classy's `pk_mult`: leaf -> (row, sign).
# On the campaign branch this map lived in tests/test_ept_ap.py and carried -1
# on rows 32/33/36/37/39, because that older clax stored those five leaves with
# nonlinear_pt.c's RAW storage sign.  The released clax negates them at the
# source (clax/ept.py:1909-1916), i.e. it stores the sign `get_pk_mult` hands
# out, so against this code every entry is +1.  See the sign note below.
BIAS_ROWS = {"Pk_0_b1b2": (30, +1), "Pk_0_b2": (31, +1),
             "Pk_0_b1bG2": (32, +1), "Pk_0_bG2": (33, +1),
             "Pk_2_b1b2": (34, +1), "Pk_2_b2": (35, +1),
             "Pk_2_b1bG2": (36, +1), "Pk_2_bG2": (37, +1),
             "Pk_4_b2": (38, +1), "Pk_4_bG2": (39, +1)}

# (row, sign, leaf names to sum) for every (Mpc/h)^3 pk_mult row a classy
# accessor reads: leaf_sum == sign * pm[row] * h**3.  `sign` is -1 exactly when
# get_pk_mult hands the row out negated *relative to the array clax stores* --
# which is NOT the same as "get_pk_mult negates raw_pk", because nonlinear_pt.c
# stores fabs() for several kernels where clax keeps the signed one, so the two
# negations cancel.
#
# Against the RELEASED clax the map is UNIFORMLY +1: every leaf is stored in
# the sign classy.pyx's `get_pk_mult` returns, so no row needs a flip.  A row
# that turns out to need -1 means the convention is not what is written here --
# report it, do not add the sign back.
#
#   row 1        clax drops nonlinear_pt.c:4853's fabs(P_Id2d2): that |.|
#                belongs to CLASS-PT's add-constant / log-spline machinery, not
#                to the spectrum (P_d2d2(k) - P_d2d2(0) <= 0), so clax stores
#                the signed kernel classy.pyx:4676 returns after negating
#                (clax/ept.py:1400)  ->  +1.  The campaign branch stored |.|
#                here and compensated with -1.
#   rows 3,4,6,7-9  classy.pyx:4680/4682/4686/4688/4690/4692 negate raw_pk, but
#                the C stores fabs (nonlinear_pt.c:4930 P_IG2, :4961 P_Id2G2,
#                :4987 P_IFG2, and :5344-5346 project that same fabs'd P_IFG2
#                into rows 7-9) while clax keeps the signed kernel
#                ->  the leaf already holds classy's sign, +1.
#   rows 32/33/36/37/39  clax negates the five bG2 channels at the source
#                (clax/ept.py:1909-1916, `Pk_* = -qf2(M_*)`) to match
#                classy.pyx:4721/4722/4727/4728/4732  ->  +1.
#   row 41       clax flips Pk_4_b1bG2 back (clax/ept.py:1938): the C loop
#                builds it from the raw P_0_b1bG2 / P_2_b1bG2 and
#                classy.pyx:4735 does NOT negate row 41, while clax feeds the
#                mu-loop the two already-negated parents  ->  +1.
#   rows 40/41   filled by the C loop, read by no classy accessor
#                (classy.pyx:4925-4932 has no pm[40]/pm[41]); ~0 at alpha = 1.
#
# Rows 10-13 are Mpc/h, not (Mpc/h)^3: classy reads them as pm[10..13]/h**2 *
# h**3 = pm * h.  Sign +1: nonlinear_pt.c:3540 / 3888 / 3895 / 3902 store
# P_CTR(_l) = +k^2 P (accumulated positive at :4498-4500, :4550-4552) and
# classy.pyx:4694-4697 negate them, while clax already stores the negated form
# (clax/ept.py Pk_ctr = -k^2 P_resummed, Pk_ctr0 = -proj(...)).
#
# Rows 42-47 (Id2d2_2 ... IG2G2_4) are filled by nonlinear_pt.c and read by no
# classy accessor, so pm_from_leaves leaves them zero.
#
# Every sign is re-measured against the stored alpha=1 pk_mult by
# tests/test_ept_assembly.py::test_pm_from_leaves_row_signs_at_alpha1 -- a flip
# reads ~2.0 on that metric, so the guard is empirical, not documentary.
def _pm_rows_h3(bias_rows=BIAS_ROWS):
    return (
        (0, +1, ("Pk_loop",)),
        (1, +1, ("Pk_Id2d2",)),
        (2, +1, ("Pk_Id2",)),
        (3, +1, ("Pk_IG2",)),
        (4, +1, ("Pk_Id2G2",)),
        (5, +1, ("Pk_IG2G2",)),
        (6, +1, ("Pk_IFG2",)),
        (7, +1, ("Pk_IFG2_0b1",)),
        (8, +1, ("Pk_IFG2_0",)),
        (9, +1, ("Pk_IFG2_2",)),
        (14, +1, ("Pk_tree",)),
        (15, +1, ("Pk_0_vv",)), (16, +1, ("Pk_0_vd",)), (17, +1, ("Pk_0_dd",)),
        (18, +1, ("Pk_2_vv",)), (19, +1, ("Pk_2_vd",)), (20, +1, ("Pk_4_vv",)),
        (21, +1, ("Pk_0_vv1",)), (22, +1, ("Pk_0_vd1",)), (23, +1, ("Pk_0_dd1",)),
        (24, +1, ("Pk_2_vv1",)), (25, +1, ("Pk_2_vd1",)), (27, +1, ("Pk_4_vv1",)),
        # rows 26/28/29 are classy's l=2 dd / l=4 vd / l=4 dd rows, which
        # nonlinear_pt.c fills as tree + 1-loop (P1loop*_ap_ir), so the pair sums.
        (26, +1, ("Pk_2_dd", "Pk_2_dd1")),
        (28, +1, ("Pk_4_vd", "Pk_4_vd1")),
        (29, +1, ("Pk_4_dd", "Pk_4_dd1")),
        (40, +1, ("Pk_4_b1b2",)),
        (41, +1, ("Pk_4_b1bG2",)),
    ) + tuple((row, sgn, (leaf,))
              for leaf, (row, sgn) in sorted(bias_rows.items(), key=lambda kv: kv[1][0]))


_PM_ROWS_H = ((10, "Pk_ctr"), (11, "Pk_ctr0"), (12, "Pk_ctr2"), (13, "Pk_ctr4"))


def pm_from_leaves(e, h):
    """(48, Nk) classy-convention `pk_mult` rebuilt from EPTComponents.

    The inverse of the ref §10 row map; signs, units and their CLASS-PT
    provenance are documented on the row-map comment above. Rows 42-47 are
    read by no accessor and stay zero.
    """
    def L(*names):
        return sum(np.asarray(getattr(e, n), dtype=float) for n in names)

    h3 = h ** 3
    pm = np.zeros((48, np.asarray(e.kh).shape[0]))
    for row, sgn, names in _pm_rows_h3():
        pm[row] = sgn * L(*names) / h3
    for row, name in _PM_ROWS_H:
        pm[row] = L(name) / h
    return pm


def clax_nine(e, bias):
    """The nine clax accessors at one bias set, keyed like assemble_from_pm.

    classy's `pk_mm_real(cs)` is clax's `pk_mm_real(cs0=cs)` (classy.pyx:4816,
    return at 4820); `pk_gm_real` takes cs0 before cs, the opposite of
    `pk_gg_real` (B4 Bug #5).
    """
    from clax.ept import (
        pk_mm_real, pk_gg_real, pk_gm_real,
        pk_mm_l0, pk_mm_l2, pk_mm_l4, pk_gg_l0, pk_gg_l2, pk_gg_l4,
    )
    b1, b2, bG2, bG3 = (bias[k] for k in ("b1", "b2", "bG2", "bGamma3"))
    return {
        "pk_mm_real": pk_mm_real(e, cs0=bias["cs"]),
        "pk_gg_real": pk_gg_real(e, b1, b2, bG2, bG3,
                                 cs=bias["cs"], cs0=bias["cs0"], Pshot=bias["Pshot"]),
        "pk_gm_real": pk_gm_real(e, b1, b2, bG2, bG3, cs0=bias["cs0"], cs=bias["cs"]),
        "pk_mm_l0": pk_mm_l0(e, cs0=bias["cs0"]),
        "pk_mm_l2": pk_mm_l2(e, cs2=bias["cs2"]),
        "pk_mm_l4": pk_mm_l4(e, cs4=bias["cs4"]),
        "pk_gg_l0": pk_gg_l0(e, b1, b2, bG2, bG3,
                             cs0=bias["cs0"], Pshot=bias["Pshot"], b4=bias["b4"]),
        "pk_gg_l2": pk_gg_l2(e, b1, b2, bG2, bG3, cs2=bias["cs2"], b4=bias["b4"]),
        "pk_gg_l4": pk_gg_l4(e, b1, b2, bG2, bG3, cs4=bias["cs4"], b4=bias["b4"]),
    }
