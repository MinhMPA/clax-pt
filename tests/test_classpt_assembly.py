"""NumPy twin of classy's CLASS-PT accessors, checked against the legacy npz.

Cosmology-independent algebra (exempt from the multi-cosmology rule): the
twin is asserted against classy on every generated file by the generator.
"""
import numpy as np
import pytest

from scripts import classpt_assembly as ca
from scripts import validation_cosmologies as vc

LEGACY = vc.REPO_ROOT / "reference_data" / "classpt_z0.38_fullrange.npz"


@pytest.fixture(scope="module")
def legacy():
    if not LEGACY.exists():
        pytest.skip(f"{LEGACY} missing")
    d = np.load(LEGACY)
    bias = {k[len("bias_"):]: float(d[k]) for k in d if k.startswith("bias_")}
    return d, bias


def _rel(a, b):
    return np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))


def test_pd2d2_0_power_law():
    kh = np.exp(np.linspace(np.log(1e-3), np.log(10.0), 401))
    pk = kh ** -1.5                       # P^2 k^3 == 1  ->  integral = ln(kmax/kmin)
    assert abs(ca.pd2d2_0(pk, kh) - np.log(1e4) / np.pi**2) < 1e-10


def test_twin_reproduces_legacy_matter_and_real_space(legacy):
    d, bias = legacy
    out = ca.assemble_from_pm(d["pk_mult"], float(d["h"]), float(d["fz"]), d["k_h"], bias, 0.0)
    for key, legacy_key in [("pk_mm_real", "pk_mm_real"), ("pk_gg_real", "pk_gg_real"),
                            ("pk_gm_real", "pk_mg_real"), ("pk_mm_l0", "pk_mm_l0"),
                            ("pk_mm_l2", "pk_mm_l2"), ("pk_mm_l4", "pk_mm_l4")]:
        assert _rel(out[key], d[legacy_key]) < 1e-10, key


def test_twin_decides_legacy_kh_convention(legacy):
    """The legacy pk_gg_* carry b4=500 terms in kh^2: exactly one unit reproduces them."""
    d, bias = legacy
    h, fz = float(d["h"]), float(d["fz"])
    hits = {}
    for label, kh in [("h/Mpc", d["k_h"]), ("1/Mpc", d["k_h"] * h)]:
        out = ca.assemble_from_pm(d["pk_mult"], h, fz, kh, bias, ca.pd2d2_0(d["pk_mult"][14] * h**3, kh))
        hits[label] = max(_rel(out[k], d[k]) for k in ("pk_gg_l0", "pk_gg_l2", "pk_gg_l4"))
    winners = [k for k, v in hits.items() if v < 1e-8]
    assert len(winners) == 1, f"kh convention undecidable: {hits}"
    assert winners[0] == ca.LEGACY_KH_CONVENTION, hits
