"""Reference accuracy tests: clax.ept vs CLASS-PT at z=0.38.

Loads pre-generated CLASS-PT reference spectra and compares all 9 output
power spectra (real-space + RSD multipoles) against clax.ept.

Accuracy targets (k < 0.3 h/Mpc):
  - Real-space and l=0,2:  relative error < 1%
  - l=4 (hexadecapole):    abs/max(ref) < 2% (robust to zero-crossings)

Usage:
    pytest tests/test_ept_accuracy.py -v
    pytest tests/test_ept_accuracy.py -v --fast   # 3 representative spectra only
"""

# Force CPU backend BEFORE importing JAX (Metal does not support float64)
import json
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
    pk_mm_real, pk_gg_real, pk_gm_real,
    pk_mm_l0, pk_mm_l2, pk_mm_l4,
    pk_gg_l0, pk_gg_l2, pk_gg_l4,
    _ir_resummation_numpy,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

K_MAX_COMPARE = 0.3     # h/Mpc -- perturbative regime cutoff
THRESH_L02 = 0.01       # 1% relative error for real-space, l=0, l=2
THRESH_L4 = 0.02        # 2% abs/max(ref) for l=4 (zero-crossing robust)
VALID_L02 = 0.01        # skip |ref| < 1% of max for l=0,l=2

L4_NAMES = {"pk_mm_l4", "pk_gg_l4"}

# Full set and fast subset
ALL_SPECTRA = [
    "pk_mm_real", "pk_gg_real", "pk_gm_real",
    "pk_mm_l0", "pk_mm_l2", "pk_mm_l4",
    "pk_gg_l0", "pk_gg_l2", "pk_gg_l4",
]
FAST_SPECTRA = ["pk_mm_real", "pk_mm_l0", "pk_gg_l2"]

# Reference NPZ key mapping: clax name -> NPZ key
# The legacy reference names the cross spectrum `pk_mg_real`; the alpha = 1
# reference names it `pk_gm_real`. _ref_key() picks whichever the file has.
REF_KEY_MAP = {
    "pk_mm_real": "pk_mm_real",
    "pk_gg_real": "pk_gg_real",
    "pk_gm_real": "pk_mg_real",
    "pk_mm_l0": "pk_mm_l0",
    "pk_mm_l2": "pk_mm_l2",
    "pk_mm_l4": "pk_mm_l4",
    "pk_gg_l0": "pk_gg_l0",
    "pk_gg_l2": "pk_gg_l2",
    "pk_gg_l4": "pk_gg_l4",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _ref_key(ref, name):
    """CLASS-PT key for a clax spectrum, tolerating the two naming schemes."""
    mapped = REF_KEY_MAP[name]
    return mapped if mapped in ref.files else name


def _ref_bias(ref):
    """Bias dict from either the flat `bias_*` keys or a `bias_json` blob."""
    if "bias_json" in ref.files:
        return json.loads(str(ref["bias_json"]))
    return {k: float(ref[f"bias_{k}"]) for k in
            ("b1", "b2", "bG2", "bGamma3", "cs0", "cs2", "cs4", "Pshot", "b4", "cs")}


@pytest.fixture(scope="module")
def classpt_ref():
    """CLASS-PT reference at alpha = 1, i.e. with Alcock-Paczynski OFF.

    `classpt_z0.38_noap_alpha1.npz` carries `ap = False` and
    `hratio = Dratio = 1.0`, so it is directly comparable to this module's
    AP-free `compute_ept`. It also records its own provenance -- the CLASS-PT
    commit and the sha256 of every patch applied to that build -- so the
    comparison is reproducible without external metadata.

    The older `classpt_z0.38_fullrange.npz` was generated with AP ON, against a
    CLASS-PT build that no longer exists. Comparing an AP-free clax against it
    leaves a geometric mismatch that happened to cancel part of the BAO
    no-wiggle error; once that error was fixed the mismatch surfaced as a
    spurious 1.15% in pk_gg_l2. It is kept in the tree as a drift bound, not as
    a correctness oracle, and the AP-aware comparison against it returns with
    the Alcock-Paczynski support.
    """
    ref_path = os.path.join(
        os.path.dirname(__file__), "..", "reference_data",
        "classpt_z0.38_noap_alpha1.npz"
    )
    if not os.path.isfile(ref_path):
        pytest.skip(f"Reference data not found: {ref_path}")
    return np.load(ref_path, allow_pickle=True)


@pytest.fixture(scope="module")
def ept_result(classpt_ref):
    """Run compute_ept on reference pk_lin and return (ept_out, bias_dict, k_h)."""
    ref = classpt_ref
    k_h = ref["k_h"]
    pk_lin = ref["pk_lin"]
    h = float(ref["h"])
    fz = float(ref["fz"])

    bias = _ref_bias(ref)

    prec = EPTPrecisionParams()
    # Feed the cosmology's own sound horizon rather than compute_ept's fiducial
    # default, so the BAO splitter sees what CLASS-PT saw.
    kw = {"rs_h": float(ref["rs_d"]) * h} if "rs_d" in ref.files else {}
    ept_out = compute_ept(
        jnp.array(pk_lin), jnp.array(k_h), h=h, f=fz, prec=prec, **kw
    )

    return ept_out, bias, k_h


@pytest.fixture(scope="module")
def clax_spectra(ept_result):
    """Compute all 9 clax spectra from EPT output."""
    ept_out, b, k_h = ept_result
    spectra = {
        "pk_mm_real": np.array(pk_mm_real(ept_out, cs0=b["cs0"])),
        "pk_gg_real": np.array(pk_gg_real(
            ept_out, b["b1"], b["b2"], b["bG2"], b["bGamma3"],
            cs=b["cs"], cs0=b["cs0"], Pshot=b["Pshot"])),
        "pk_gm_real": np.array(pk_gm_real(
            ept_out, b["b1"], b["b2"], b["bG2"], b["bGamma3"],
            cs0=b["cs0"], cs=b["cs"])),
        "pk_mm_l0": np.array(pk_mm_l0(ept_out, cs0=b["cs0"])),
        "pk_mm_l2": np.array(pk_mm_l2(ept_out, cs2=b["cs2"])),
        "pk_mm_l4": np.array(pk_mm_l4(ept_out, cs4=b["cs4"])),
        "pk_gg_l0": np.array(pk_gg_l0(
            ept_out, b["b1"], b["b2"], b["bG2"], b["bGamma3"],
            cs0=b["cs0"], Pshot=b["Pshot"], b4=b["b4"])),
        "pk_gg_l2": np.array(pk_gg_l2(
            ept_out, b["b1"], b["b2"], b["bG2"], b["bGamma3"],
            cs2=b["cs2"], b4=b["b4"])),
        "pk_gg_l4": np.array(pk_gg_l4(
            ept_out, b["b1"], b["b2"], b["bG2"], b["bGamma3"],
            cs4=b["cs4"], b4=b["b4"])),
    }
    return spectra


# ---------------------------------------------------------------------------
# Parametrized accuracy test
# ---------------------------------------------------------------------------

def _get_spectra_list(request):
    """Return spectrum list depending on --fast mode."""
    fast = request.config.getoption("--fast", default=False)
    return FAST_SPECTRA if fast else ALL_SPECTRA


@pytest.mark.parametrize("spectrum_name", ALL_SPECTRA)
def test_ept_accuracy(spectrum_name, classpt_ref, clax_spectra, ept_result, request):
    """Compare clax.ept spectrum against CLASS-PT reference at k < 0.3 h/Mpc."""
    fast = request.config.getoption("--fast", default=False)
    if fast and spectrum_name not in FAST_SPECTRA:
        pytest.skip("--fast mode: skipping non-representative spectrum")

    _, _, k_h = ept_result
    mask = k_h < K_MAX_COMPARE
    k_compare = k_h[mask]
    n_modes = int(mask.sum())

    clax_vals = clax_spectra[spectrum_name][mask]
    ref_key = _ref_key(classpt_ref, spectrum_name)
    ref_vals = np.squeeze(classpt_ref[ref_key])[mask]

    is_l4 = spectrum_name in L4_NAMES

    if is_l4:
        # Absolute-normalised error: robust to zero-crossings
        ref_scale = np.abs(ref_vals).max()
        abs_err = np.abs(clax_vals - ref_vals)
        norm_err = abs_err / ref_scale
        max_err = float(norm_err.max())
        mean_err = float(norm_err.mean())
        k_at_max = float(k_compare[norm_err.argmax()])
        threshold = THRESH_L4
        metric = "abs/max(ref)"
    else:
        abs_ref = np.abs(ref_vals)
        valid = abs_ref > VALID_L02 * abs_ref.max()
        n_valid = int(valid.sum())
        if n_valid < 5:
            pytest.skip(f"{spectrum_name}: too few valid points ({n_valid})")

        rel_err = np.abs(clax_vals[valid] - ref_vals[valid]) / abs_ref[valid]
        max_err = float(rel_err.max())
        mean_err = float(rel_err.mean())
        k_at_max = float(k_compare[valid][rel_err.argmax()])
        threshold = THRESH_L02
        metric = "rel"

    passed = max_err < threshold
    print(f"\n  {spectrum_name}: max={max_err:.4%} mean={mean_err:.4%} "
          f"k@max={k_at_max:.4f} ({metric}, {n_modes} modes) "
          f"{'PASS' if passed else 'FAIL'}")

    assert passed, (
        f"{spectrum_name}: max {metric} error {max_err:.4%} > {threshold:.0%} "
        f"at k={k_at_max:.4f} h/Mpc (mean={mean_err:.4%})"
    )
