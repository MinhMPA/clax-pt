# clax-pt Validation — Part 1a: CLASS-PT Oracle (Track A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Read Part 0 first**: `docs/superpowers/plans/2026-09-03-clax-pt-validation-part0-common.md` — Global Constraints, oracle findings, run recipes (sbatch templates, commit recipe), reviewer briefs. Nothing from Part 0 is repeated here.

**Goal:** A reproducible, patched CLASS-PT oracle (`classpt` env) and a reference generator that writes one `.npz` per (case, z, flags) for all 14 distinct campaign cosmologies × 3 redshifts, gated by reproducing the legacy `reference_data/classpt_z0.38_fullrange.npz`.

**Architecture:** One pure-Python module (`scripts/validation_cosmologies.py`) is the single source of truth for cases, bias sets, redshifts, and file paths — importable in both the `clax` and `classpt` envs. A NumPy twin of classy's accessor algebra (`scripts/classpt_assembly.py`) is asserted against classy on every generated file and reused clax-side. The generator is a thin driver over classy + these two modules. All CLASS-PT runs go through sbatch (CPU, no GPU).

**Tech Stack:** Python 3.10 (`classpt` env: numpy<2, cython<3, scipy, openblas), CLASS-PT `09d5531a` + two patches, micromamba, SLURM.

**Spec:** `docs/superpowers/specs/2026-09-03-clax-pt-validation-design.md` (§4.1 cases, §4.8 layout/envs, §5.3 mapping, §4.2 provenance gate).
**Companion reference:** `docs/superpowers/plans/2026-09-03-clax-pt-validation-classpt-inloop-reference.md` (cited as "ref §N").

**Task order:** A1 ∥ A2 → A3 → A4 → A5. Track B (Part 1b) only needs A1 (for `ept_case`) and A3 (for `classpt_assembly`, Task B7).

---

## File structure (Track A)

| Path | Responsibility | Task |
|---|---|---|
| `scripts/validation_cosmologies.py` | cases, families, aliases, bias sets, z list, fast subset, k-grid twin, CLASS-PT param mapping, reference paths | A1 |
| `tests/test_validation_cosmologies.py` | locks the module's invariants | A1 |
| `tests/conftest.py` | `ept_case` fixture (append only) | A1 |
| `scripts/classpt_patches/classy_ap_ratios.patch`, `classy_kh_units.patch` | classy accessor + kh-unit patches (ref §13) | A2 |
| `scripts/setup_classpt_env.sh` | idempotent env build + patch + compile + verify | A2 |
| `docs/classpt-build-notes.md` | what the build needed (only if a retry was needed) | A2 |
| `scripts/classpt_assembly.py` | NumPy twin of classy accessors (ref §11) | A3 |
| `tests/test_classpt_assembly.py` | twin vs legacy npz; kh-convention decision | A3 |
| `scripts/generate_classpt_reference.py` | rewritten generator (keyword accessors, provenance fields) | A3 |
| `tests/test_classpt_provenance.py` | legacy-reproduction gate | A4 |
| `slurm/classpt-refgen.sbatch`, `scripts/write_classpt_manifest.py`, `reference_data/classpt/MANIFEST.md` | full reference generation + manifest | A5 |

---

### Task A1: `validation_cosmologies.py` — the campaign's single source of truth

**Files:**
- Create: `scripts/validation_cosmologies.py`
- Create: `tests/test_validation_cosmologies.py`
- Modify: `tests/conftest.py` (append the `ept_case` fixture after `nulcdm_cosmology`, line ~175; leave the grid literals untouched)

**Interfaces:**
- Consumes: nothing from clax at import time (lazy import inside `clax_params`).
- Produces (used by A3, A5, B1, B7, C0–C3):
  - `FIDUCIAL: dict`, `CASES: dict[str, dict]` (15 names → clax-named overrides), `FAMILIES: dict[str, tuple[str, ...]]`, `ALIASES = {"massive_nu_006": "lcdm_fiducial"}`, `Z_LIST = (0.0, 0.38, 0.8)`, `OMFID = 0.31`, `BIAS`, `BIAS_NONZERO`, `FAST_CASES`, `FAST_Z = 0.38`, `LEGACY_CLASSPT_FIDUCIAL`, `Y_HE_CLAX = 0.2454006`
  - `cosmo_params(name) -> dict`, `canonical_case(name) -> str`, `distinct_cases() -> list[str]` (14), `clax_params(name) -> CosmoParams`, `legacy_clax_params() -> CosmoParams`
  - `ept_kgrid_numpy(kmin_h=5e-5, kmax_h=100.0, nmax=256) -> np.ndarray` (bit-identical to `clax.ept.ept_kgrid()`)
  - `classpt_params_from(cosmo, *, z_list, ap=True, omfid=OMFID, cb=True, yhe=Y_HE_CLAX, use_ppf=None, pt=True) -> dict`
  - `classpt_params(name, **kw) -> dict`
  - `reference_path(case, z, *, ap=True, omfid=OMFID, cb=True, bias="fiducial", tag="") -> Path`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validation_cosmologies.py
"""Locks the campaign grid (spec §4.1), CLASS-PT mapping (§5.3) and layout (§4.8).

Cosmology-independent plumbing: exempt from the multi-cosmology rule.
"""
import math

import numpy as np
import pytest

from scripts import validation_cosmologies as vc
from tests import conftest


def test_case_counts_and_families():
    assert len(vc.CASES) == 15
    assert {len(v) for v in vc.FAMILIES.values()} == {5}
    assert sorted(sum(vc.FAMILIES.values(), ())) == sorted(vc.CASES)
    assert vc.distinct_cases() == [c for c in vc.CASES if c not in vc.ALIASES]
    assert len(vc.distinct_cases()) == 14


def test_alias_is_physically_identical():
    assert vc.cosmo_params("massive_nu_006") == vc.cosmo_params("lcdm_fiducial")
    assert vc.canonical_case("massive_nu_006") == "lcdm_fiducial"
    assert vc.canonical_case("h_high") == "h_high"


def test_fiducial_matches_clax_defaults():
    pytest.importorskip("jax")
    from clax import CosmoParams
    p = CosmoParams()
    for key, val in vc.FIDUCIAL.items():
        assert getattr(p, key) == val, key
    assert vc.Y_HE_CLAX == p.Y_He


def test_conftest_grids_are_subsets_of_cases():
    assert conftest.COSMOLOGY_GRID_LCDM == {n: vc.CASES[n] for n in vc.FAMILIES["lcdm"]}
    for name, ov in conftest.COSMOLOGY_GRID_NULCDM.items():
        assert vc.CASES[name] == ov


def test_classpt_mapping_lcdm():
    prm = vc.classpt_params("lcdm_fiducial", z_list=(0.0, 0.38))
    assert math.isclose(prm["A_s"], 2.1e-9, rel_tol=1e-9)
    assert prm["N_ncdm"] == 1 and prm["m_ncdm"] == 0.06
    assert prm["N_ur"] == 2.0328 and prm["T_ncdm"] == 0.71611
    assert prm["YHe"] == vc.Y_HE_CLAX
    assert prm["z_pk"] == "0,0.38"
    assert prm["Omfid"] == "0.31" and prm["AP"] == "Yes" and prm["cb"] == "Yes"
    assert prm["non linear"] == "PT" and prm["P_k_max_h/Mpc"] == 100.0
    assert "w0_fld" not in prm and "use_ppf" not in prm


def test_classpt_mapping_w0wa_and_flags():
    prm = vc.classpt_params("w0wa_m07_m10", z_list=(0.8,), ap=False, cb=False, use_ppf=False)
    assert prm["w0_fld"] == -0.7 and prm["wa_fld"] == -1.0 and prm["Omega_Lambda"] == 0.0
    assert prm["AP"] == "No" and "Omfid" in prm and prm["cb"] == "No"
    assert prm["use_ppf"] == "no"


def test_legacy_params_refuse_cb_and_keep_exact_dict():
    with pytest.raises(ValueError):
        vc.classpt_params_from(vc.LEGACY_CLASSPT_FIDUCIAL, z_list=(0.38,), cb=True)
    prm = vc.classpt_params_from(vc.LEGACY_CLASSPT_FIDUCIAL, z_list=(0.38,), cb=False, yhe=None)
    assert prm["A_s"] == 2.0989e-9 and "N_ncdm" not in prm and "YHe" not in prm
    assert "N_ur" not in prm  # CLASS-PT default 3.044, exactly as the legacy run


def test_reference_paths():
    p = vc.reference_path("h_high", 0.38)
    assert p == vc.REFERENCE_ROOT / "h_high" / "z0.380_ap_omfid0.31_cb.npz"
    assert vc.reference_path("h_high", 0.0, ap=False, cb=False).name == "z0.000_noap_m.npz"
    assert vc.reference_path("h_high", 0.8, bias="nonzero").name == "z0.800_ap_omfid0.31_cb_biasnz.npz"
    assert vc.reference_path("w0wa_m07_m10", 0.38, tag="noppf").name == "z0.380_ap_omfid0.31_cb_noppf.npz"
    # aliases resolve to the canonical directory
    assert vc.reference_path("massive_nu_006", 0.38) == vc.reference_path("lcdm_fiducial", 0.38)


def test_fast_subset():
    assert vc.FAST_CASES == ("lcdm_fiducial", "massive_nu_015", "w0wa_m07_m10")
    assert vc.FAST_Z == 0.38 and vc.FAST_Z in vc.Z_LIST


def test_kgrid_twin_matches_clax():
    pytest.importorskip("jax")
    from clax.ept import EPTPrecisionParams, ept_kgrid
    assert np.array_equal(ept_kgrid(EPTPrecisionParams()), vc.ept_kgrid_numpy())
    assert vc.ept_kgrid_numpy().shape == (256,)
```

- [ ] **Step 2: Run it to verify it fails**

Run (login node, CPU probe env from Part 0):
`PYTHONPATH=/home/n2minh/clax-ptval JAX_PLATFORMS=cpu python -m pytest tests/test_validation_cosmologies.py -q`
Expected: `ModuleNotFoundError: No module named 'scripts.validation_cosmologies'`.

- [ ] **Step 3: Write the module**

```python
# scripts/validation_cosmologies.py
"""Canonical cosmology grid for the clax-pt vs CLASS-PT validation campaign.

Single source of truth for the 15 campaign cases (spec §4.1), redshifts,
the fixed AP fiducial, bias sets, the `--fast` subset and the reference
layout (spec §4.8).  Pure Python + NumPy so it imports in BOTH envs:
`clax` (JAX) and `classpt` (CLASS-PT oracle; no JAX, no clax).
`clax_params()` therefore imports clax lazily.

Cosmology dicts use clax `CosmoParams` names; `classpt_params_from()`
translates them to CLASS-PT `.ini` keys (spec §5.3).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_ROOT = REPO_ROOT / "reference_data" / "classpt"

# clax CosmoParams defaults (clax/params.py:46-75). ln10A_s = ln(2.1e-9 * 1e10).
FIDUCIAL = dict(
    h=0.6736, omega_b=0.02237, omega_cdm=0.1200, ln10A_s=3.0445224377,
    n_s=0.9649, tau_reio=0.0544,
    N_ncdm=1, m_ncdm=0.06, N_ur=2.0328, T_ncdm_over_T_cmb=0.71611,
    w0=-1.0, wa=0.0,
)
Y_HE_CLAX = 0.2454006   # clax/params.py:82; passed to CLASS-PT as YHe to remove the BBN seam

# name -> overrides on FIDUCIAL (spec §4.1). Values are locked to
# tests/conftest.py COSMOLOGY_GRID_LCDM / _NULCDM and
# scripts/generate_multipoint_reference.py.
CASES = {
    "lcdm_fiducial": {},
    "h_high": {"h": 0.6736 * 1.10},
    "omega_b_high": {"omega_b": 0.02237 * 1.20},
    "omega_cdm_low": {"omega_cdm": 0.1200 * 0.80},
    "ns_high": {"n_s": 0.9649 * 1.05},
    "massive_nu_006": {"m_ncdm": 0.06},          # == lcdm_fiducial (ALIASES)
    "massive_nu_015": {"m_ncdm": 0.15},
    "massive_nu_030": {"m_ncdm": 0.30},
    "massive_nu_015_h_high": {"m_ncdm": 0.15, "h": 0.6736 * 1.10},
    "massive_nu_015_omega_cdm_low": {"m_ncdm": 0.15, "omega_cdm": 0.1200 * 0.80},
    "w0wa_m09_p01": {"w0": -0.9, "wa": 0.1},
    "w0wa_m11_m01": {"w0": -1.1, "wa": -0.1},
    "w0wa_m10_p03": {"w0": -1.0, "wa": 0.3},
    "w0wa_m10_m03": {"w0": -1.0, "wa": -0.3},
    "w0wa_m07_m10": {"w0": -0.7, "wa": -1.0},    # w crosses -1 at a=0.7
}
FAMILIES = {
    "lcdm": ("lcdm_fiducial", "h_high", "omega_b_high", "omega_cdm_low", "ns_high"),
    "nulcdm": ("massive_nu_006", "massive_nu_015", "massive_nu_030",
               "massive_nu_015_h_high", "massive_nu_015_omega_cdm_low"),
    "w0wacdm": ("w0wa_m09_p01", "w0wa_m11_m01", "w0wa_m10_p03", "w0wa_m10_m03", "w0wa_m07_m10"),
}
ALIASES = {"massive_nu_006": "lcdm_fiducial"}

Z_LIST = (0.0, 0.38, 0.8)
OMFID = 0.31

# classy accessor names (classy.pyx:4795-4915). `cs` is the real-space matter
# counterterm; clax calls the same coefficient `cs0` in pk_mm_real.
BIAS = dict(b1=2.0, b2=0.0, bG2=0.0, bGamma3=0.0, cs0=0.0, cs2=0.0, cs4=0.0,
            cs=0.0, Pshot=0.0, b4=500.0)
BIAS_NONZERO = dict(b1=2.0, b2=-1.0, bG2=0.1, bGamma3=-0.1, cs0=5.0, cs2=15.0,
                    cs4=-5.0, cs=1.0, Pshot=5.0e3, b4=100.0)

FAST_CASES = ("lcdm_fiducial", "massive_nu_015", "w0wa_m07_m10")   # spec §4.6
FAST_Z = 0.38

# Exact input dict of the legacy reference_data/classpt_z0.38_fullrange.npz
# (scripts/generate_classpt_reference.py before Task A3): no N_ncdm, no N_ur,
# no YHe, no cb key -> CLASS-PT defaults N_ur=3.044, BBN YHe, cb=TRUE (which
# with N_ncdm=0 reads an unassigned delta_cb index; see Part 0 findings).
LEGACY_CLASSPT_FIDUCIAL = {
    "h": 0.6736, "omega_b": 0.02237, "omega_cdm": 0.1200,
    "A_s": 2.0989e-9, "n_s": 0.9649, "tau_reio": 0.0544,
}


def cosmo_params(name: str) -> dict:
    """clax-named parameter dict for a case (FIDUCIAL updated by its overrides)."""
    return {**FIDUCIAL, **CASES[name]}


def canonical_case(name: str) -> str:
    return ALIASES.get(name, name)


def distinct_cases() -> list[str]:
    """The 14 physically distinct cases, in CASES order (aliases dropped)."""
    return [c for c in CASES if c not in ALIASES]


def clax_params(name: str):
    """CosmoParams for a case (lazy clax import: absent in the classpt env)."""
    from clax import CosmoParams
    return CosmoParams(**cosmo_params(name))


def legacy_clax_params():
    """CosmoParams reproducing LEGACY_CLASSPT_FIDUCIAL on the clax side."""
    from clax import CosmoParams
    return CosmoParams(
        h=0.6736, omega_b=0.02237, omega_cdm=0.1200,
        ln10A_s=math.log(2.0989e-9 * 1e10), n_s=0.9649, tau_reio=0.0544,
        N_ncdm=0, N_ur=3.044,
    )


def ept_kgrid_numpy(kmin_h: float = 5e-5, kmax_h: float = 100.0, nmax: int = 256) -> np.ndarray:
    """Bit-identical twin of clax.ept.ept_kgrid (ept.py:1878-1889)."""
    return np.exp(np.linspace(np.log(kmin_h), np.log(kmax_h), nmax))


def classpt_params_from(cosmo: dict, *, z_list, ap: bool = True, omfid: float = OMFID,
                        cb: bool = True, yhe=Y_HE_CLAX, use_ppf=None, pt: bool = True) -> dict:
    """CLASS-PT input dict from a clax-named (or legacy CLASS-named) cosmology dict.

    Spec §5.3. `cosmo` may carry `A_s` directly (legacy dict) or `ln10A_s`.
    Strings are used for the CLASS-PT flags because its parser compares text
    (input.c:3952-3957 accepts N/No/NO for cb). `Omfid` is always written
    (CLASS-PT ignores it when AP=No; keeping it makes the noap files comparable).
    """
    n_ncdm = int(cosmo.get("N_ncdm", 0))
    if cb and n_ncdm == 0:
        raise ValueError("cb=True with N_ncdm=0 reads an unassigned delta_cb index in "
                         "CLASS-PT (perturbations.c:1273-1279); pass cb=False")
    prm = {"h": cosmo["h"], "omega_b": cosmo["omega_b"], "omega_cdm": cosmo["omega_cdm"],
           "n_s": cosmo["n_s"], "tau_reio": cosmo["tau_reio"]}
    prm["A_s"] = cosmo["A_s"] if "A_s" in cosmo else 1e-10 * math.exp(cosmo["ln10A_s"])
    if n_ncdm > 0:
        prm.update({"N_ncdm": n_ncdm, "m_ncdm": cosmo["m_ncdm"], "N_ur": cosmo["N_ur"],
                    "T_ncdm": cosmo["T_ncdm_over_T_cmb"]})
    elif "N_ur" in cosmo:
        prm["N_ur"] = cosmo["N_ur"]
    w0, wa = cosmo.get("w0", -1.0), cosmo.get("wa", 0.0)
    if (w0, wa) != (-1.0, 0.0):
        prm.update({"w0_fld": w0, "wa_fld": wa, "Omega_Lambda": 0.0})
    if use_ppf is not None:
        prm["use_ppf"] = "yes" if use_ppf else "no"
    if yhe is not None:
        prm["YHe"] = yhe
    if pt:
        prm.update({"output": "mPk", "non linear": "PT", "IR resummation": "Yes",
                    "Bias tracers": "Yes", "RSD": "Yes",
                    "AP": "Yes" if ap else "No", "Omfid": f"{omfid:g}",
                    "cb": "Yes" if cb else "No", "P_k_max_h/Mpc": 100.0})
    prm["z_pk"] = ",".join(f"{z:g}" for z in z_list)
    return prm


def classpt_params(name: str, **kw) -> dict:
    return classpt_params_from(cosmo_params(name), **kw)


def reference_path(case: str, z: float, *, ap: bool = True, omfid: float = OMFID,
                   cb: bool = True, bias: str = "fiducial", tag: str = "") -> Path:
    """reference_data/classpt/<canonical case>/z{z:.3f}_{ap_omfid{X}|noap}_{cb|m}[_biasnz][_tag].npz"""
    assert bias in ("fiducial", "nonzero"), bias
    stem = f"z{z:.3f}_" + (f"ap_omfid{omfid:g}" if ap else "noap") + ("_cb" if cb else "_m")
    if bias == "nonzero":
        stem += "_biasnz"
    if tag:
        stem += f"_{tag}"
    return REFERENCE_ROOT / canonical_case(case) / f"{stem}.npz"
```

- [ ] **Step 4: Add the `ept_case` fixture to `tests/conftest.py`** (append after `nulcdm_cosmology = ...`, keep everything above unchanged):

```python
# --- clax-pt validation campaign (docs/superpowers/specs/2026-09-03-clax-pt-validation-design.md) ---
from scripts import validation_cosmologies as _vc


@pytest.fixture(params=_vc.distinct_cases(), ids=_vc.distinct_cases())
def ept_case(request):
    """(name, CosmoParams) over the 14 distinct campaign cases; under --fast
    only validation_cosmologies.FAST_CASES run (spec §4.6)."""
    name = request.param
    if request.config.getoption("--fast") and name not in _vc.FAST_CASES:
        pytest.skip(f"--fast runs {_vc.FAST_CASES} only (skipping {name})")
    return name, _vc.clax_params(name)
```

- [ ] **Step 5: Run the tests**

Run: `PYTHONPATH=/home/n2minh/clax-ptval JAX_PLATFORMS=cpu python -m pytest tests/test_validation_cosmologies.py tests/test_fast_flag_selection.py -q`
Expected: all pass (the second file guards that conftest still imports and `--fast` pruning still works). If `from scripts import validation_cosmologies` fails inside conftest, `scripts/` is not on the path: `pyproject.toml` has `testpaths=["tests"]` and pytest inserts the rootdir — confirm with `python -c "import scripts.validation_cosmologies"` from the worktree root and, if needed, add `pythonpath = ["."]` under `[tool.pytest.ini_options]` (say so in the commit).

- [ ] **Step 6: Commit** (Part 0 commit recipe)

`feat(ptval): validation_cosmologies — campaign grid, CLASS-PT mapping, reference layout (A1)`

---

### Task A2: `classpt` env, CLASS-PT patches, build script

**Files:**
- Create: `scripts/classpt_patches/classy_ap_ratios.patch`, `scripts/classpt_patches/classy_kh_units.patch`
- Create: `scripts/setup_classpt_env.sh`
- Create (only if a retry was needed): `docs/classpt-build-notes.md`

**Interfaces:**
- Produces: env `classpt` with an importable `classy` exposing `Class.get_ap_ratios(z) -> (hratio, Dratio, f)` and `Class.get_Pd2d2_0() -> float`, and `initialize_output` storing `self.kh` in **h/Mpc**. Consumed by A3/A5.

**Facts to hold** (verified at `09d5531a`): `include/nonlinear_pt.h:588-590` declares `double * growthf; double * hratio_array; double * Dratio_array;` and `nonlinear_pt.c:1222-1224` allocates them for every `z_pk` whenever `non linear = PT` (AP=No fills 1.0 at `1268-1269`/`1294-1295`; RSD=No fills f=1 at `1307`). `python/cclassy.pxd:424-425` already declares `int z_pk_num` / `double z_pk[100]` inside `cdef struct nonlinear_pt:`; the enum above it declares `nlpt_none, nlpt_spt`. `python/classy.pyx:119-120` has `cdef double fz` / `cdef double Pd2d2_0` (private); `4607` `def get_pk_mult(...)`; `4783-4785` `def initialize_output(...)` / `self.kh = k`. `Makefile:50` hard-codes an OpenBLAS path; `python/setup.py` reads `OPENBLAS_PATH`.

- [ ] **Step 1: Make the edits in `/home/n2minh/CLASS-PT` and export the patches**

Edit 1 — `python/cclassy.pxd`, inside `cdef struct nonlinear_pt:` directly after `double z_pk[100]` (line 425), add (8-space indent, matching neighbours):
```
        double * growthf
        double * hratio_array
        double * Dratio_array
```
Edit 2 — `python/classy.pyx`, insert immediately BEFORE `def get_pk_mult(` (line 4607), class-method indentation:
```cython
    def get_ap_ratios(self, double z):
        """(hratio, Dratio, f) as used in-loop by nonlinear_pt.c:1245-1296 for z_pk == z.
        AP=No returns (1, 1, f); RSD=No returns (1, 1, 1). Raises if z is not a requested z_pk."""
        cdef int i
        if self.nlpt.method != nlpt_spt:
            raise CosmoSevereError("get_ap_ratios: requires 'non linear': 'PT'")
        for i in range(self.nlpt.z_pk_num):
            if abs(self.nlpt.z_pk[i] - z) < 1e-10:
                return (self.nlpt.hratio_array[i], self.nlpt.Dratio_array[i], self.nlpt.growthf[i])
        raise CosmoSevereError("get_ap_ratios: z=%g is not in z_pk" % z)

    def get_Pd2d2_0(self):
        """Value computed by the last initialize_output() call (classy.pyx:4791)."""
        return self.Pd2d2_0

```
Then: `git -C /home/n2minh/CLASS-PT diff > /home/n2minh/clax-ptval/scripts/classpt_patches/classy_ap_ratios.patch`, then `git -C /home/n2minh/CLASS-PT checkout -- python/`.

Edit 3 — `python/classy.pyx:4785`: `self.kh = k` → `self.kh = k / self.ba.h` (rationale ref §12: callers pass k in 1/Mpc; `Pd2d2_0`, the `b4 kh²` and `a_n (kh/0.45)²` terms are written for h/Mpc). Export as `classy_kh_units.patch`, then `git checkout -- python/` again so the tree is clean.

Verify: `git -C /home/n2minh/CLASS-PT status --short` prints nothing; `git -C /home/n2minh/CLASS-PT apply --check <patch>` succeeds for each; `grep -c "^+[^+]" scripts/classpt_patches/classy_kh_units.patch` is 1.

- [ ] **Step 2: Write the build script**

```bash
#!/bin/bash -l
# scripts/setup_classpt_env.sh — build the CLASS-PT oracle in a dedicated env.
# Idempotent: re-running skips env creation and already-applied patches.
# NEVER installs into any other env (carpile/cosmopower/cosmodesi/fli-mf-nuts).
set -euo pipefail

ENV_NAME=classpt
CLASSPT_DIR=${CLASSPT_DIR:-/home/n2minh/CLASS-PT}
CLASSPT_COMMIT=09d5531a
REPO=$(cd "$(dirname "$0")/.." && pwd)
PATCHES="$REPO/scripts/classpt_patches"

eval "$(micromamba shell hook --shell bash)"
if [ ! -d "$HOME/micromamba/envs/$ENV_NAME" ]; then
  micromamba create -y -n "$ENV_NAME" -c conda-forge python=3.10 "numpy<2" "cython<3" \
    scipy "setuptools<60" gcc gxx make pip openblas
fi
micromamba activate "$ENV_NAME"

cd "$CLASSPT_DIR"
git rev-parse --short HEAD | grep -q "^$CLASSPT_COMMIT" || { echo "ERROR CLASS-PT HEAD is not $CLASSPT_COMMIT"; exit 1; }
for p in classy_ap_ratios classy_kh_units; do
  if git apply --reverse --check "$PATCHES/$p.patch" 2>/dev/null; then
    echo "patch $p already applied"
  else
    git apply "$PATCHES/$p.patch" && echo "applied $p"
  fi
done

export OPENBLAS_PATH="$CONDA_PREFIX/lib" CC=gcc
make clean >/dev/null 2>&1 || true
make OPENBLAS="-L$CONDA_PREFIX/lib -lopenblas" CC=gcc PYTHON=python class libclass.a
make OPENBLAS="-L$CONDA_PREFIX/lib -lopenblas" CC=gcc PYTHON=python classy

cd "$REPO"
python - <<'EOF'
from classy import Class
M = Class()
assert hasattr(M, "get_ap_ratios") and hasattr(M, "get_Pd2d2_0"), "patched accessors missing"
print("classy OK:", __import__("classy").__file__)
EOF
```

Run it: `bash scripts/setup_classpt_env.sh 2>&1 | tail -30` (login node is fine: gcc + Cython, no JAX). If `make` fails with "multiple definition of ..." (gcc ≥10 `-fno-common` default), rerun both `make` lines with `OPTFLAG="-O4 -ffast-math -fcommon"` appended and record the exact error and the fix in `docs/classpt-build-notes.md` (5–10 lines). Any other failure: read the first error line, fix the cause in the script, keep the script idempotent.

- [ ] **Step 3: Verify the oracle end to end with the legacy fiducial (CPU, seconds)**

```bash
micromamba run -n classpt env PYTHONPATH=/home/n2minh/clax-ptval python - <<'EOF'
import numpy as np
from classy import Class
from scripts import validation_cosmologies as vc
prm = vc.classpt_params_from(vc.LEGACY_CLASSPT_FIDUCIAL, z_list=(0.38,), cb=False, yhe=None)
M = Class(); M.set(prm); M.compute()
h = M.h(); k_h = vc.ept_kgrid_numpy(); k = k_h * h
M.initialize_output(k, 0.38, len(k_h))
print("ap", M.get_ap_ratios(0.38), "f", M.scale_independent_growth_factor_f(0.38), "Pd2d2_0", M.get_Pd2d2_0())
try: M.get_ap_ratios(0.5)
except Exception as e: print("expected error:", type(e).__name__)
EOF
```
Expected: `hratio`/`Dratio` both within (0.9, 1.1) and not exactly 1 (AP=Yes, Ωm≠0.31); `f` ≈ 0.71; `Pd2d2_0 > 0`; the second call raises. Paste this output in the commit message.

- [ ] **Step 4: Commit** — `build(ptval): classpt env, classy accessor + kh-unit patches, setup script (A2)`. Only `scripts/classpt_patches/*.patch`, `scripts/setup_classpt_env.sh`, and (if present) `docs/classpt-build-notes.md` go in; never any file under `/home/n2minh/CLASS-PT`.

---

### Task A3: NumPy accessor twin + generator rewrite

**Files:**
- Create: `scripts/classpt_assembly.py`
- Create: `tests/test_classpt_assembly.py`
- Rewrite: `scripts/generate_classpt_reference.py` (full replacement; the legacy positional 7-arg `pk_gg_l0` call cannot run against the current 9-arg classy)

**Interfaces:**
- Consumes: A1 (`validation_cosmologies`), A2 (`classpt` env with patched classy).
- Produces:
  - `classpt_assembly.pd2d2_0(pk_lin_h, kh) -> float` — `simpson(P² kh³, x=ln kh)/π²` (classy.pyx:4791)
  - `classpt_assembly.assemble_from_pm(pm, h, fz, kh, bias, Pd2d2_0) -> dict` with keys `pk_mm_real, pk_gg_real, pk_gm_real, pk_mm_l0, pk_mm_l2, pk_mm_l4, pk_gg_l0, pk_gg_l2, pk_gg_l4` (all (Mpc/h)³; `a0_nbar = a2_nbar = 0`)
  - `classpt_assembly.LEGACY_KH_CONVENTION: str` — `"h/Mpc"` or `"1/Mpc"`, set from the test in Step 2 (the legacy file's `pk_gg_*` were assembled with `self.kh` in that unit)
  - `classpt_assembly.PM_ROWS_VALID = slice(0, 48)` — rows compared anywhere (48–71 are fNL garbage, ref §10)
  - `generate_classpt_reference.py` CLI: `--cosmology NAME | --legacy`, `--z-list Z [Z ...]`, `--ap {yes,no}`, `--omfid F`, `--cb {yes,no}`, `--bias {fiducial,nonzero}`, `--yhe F|none`, `--use-ppf {default,yes,no}`, `--tag STR`, `--outdir PATH`; one npz per z at `validation_cosmologies.reference_path(...)`.
  - npz keys (all arrays float64; scalars 0-d): `k_h (256,)`, `z, h, fz, growthf, D_z, H_z, DA_z, rs_d, hratio, Dratio, Pd2d2_0` (`rs_d` = classy `rs_drag()` in Mpc — CLASS-PT's BAO damping scale `rbao = pth->rs_d`, `nonlinear_pt.c:2919`; Part 2 C1 injects `rs_h = rs_d * h` into `compute_ept`), `pk_lin` (the spectrum CLASS-PT looped over: cb if `cb` else m; (Mpc/h)³), `pk_m_lin`, `pk_cb_lin` (only when N_ncdm>0), `pk_mult (96,256)` raw `get_pk_mult` rows (CLASS units, ref §10), `pk_mm_real, pk_gg_real, pk_gm_real, pk_mm_l0, pk_mm_l2, pk_mm_l4, pk_gg_l0, pk_gg_l2, pk_gg_l4` from classy, `kh_convention="h/Mpc"`, `ap (bool), omfid, cb (bool), use_ppf (str)`, `params_json`, `bias_json`, `classpt_commit`, `patches_sha256` (JSON `{name: sha256}`).

- [ ] **Step 1: Write the twin's failing tests**

```python
# tests/test_classpt_assembly.py
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
```

- [ ] **Step 2: Write the twin, run the tests, set `LEGACY_KH_CONVENTION`**

```python
# scripts/classpt_assembly.py
"""NumPy twin of the CLASS-PT classy accessors (classy.pyx:4795-4915, ref §11).

Consumers: the reference generator asserts classy == twin on every file it
writes, and clax-side tests assemble spectra from stored `pk_mult` with any
bias set without classy.  `pm` is the (96, Nk) array from `get_pk_mult`
(CLASS units, row transforms in ref §10); `h` the Hubble parameter; `fz` the
growth rate; `kh` the k grid in h/Mpc (the patched classy convention);
`Pd2d2_0` the k->0 limit of the b2^2 term (classy.pyx:4791).

Bias keys are classy's: b1 b2 bG2 bGamma3 cs0 cs2 cs4 cs Pshot b4.
classy pk_mm_real(cs) corresponds to clax pk_mm_real(cs0=cs).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import simpson

PM_ROWS_VALID = slice(0, 48)        # rows 48-71 are fNL garbage when PNG is off
LEGACY_KH_CONVENTION = None         # set by tests/test_classpt_assembly.py: "h/Mpc" or "1/Mpc"


def pd2d2_0(pk_lin_h, kh) -> float:
    """classy.pyx:4791: simpson(Plin_hMpc3**2 * kh**3, x=log(kh)) / pi**2."""
    pk_lin_h = np.asarray(pk_lin_h, dtype=float)
    kh = np.asarray(kh, dtype=float)
    return float(simpson(pk_lin_h**2 * kh**3, x=np.log(kh)) / np.pi**2)


def assemble_from_pm(pm, h, fz, kh, bias, Pd2d2_0) -> dict:
    """Every classy accessor, transcribed line-for-line (a0_nbar = a2_nbar = 0)."""
    pm = np.asarray(pm, dtype=float)
    kh = np.asarray(kh, dtype=float)
    b1, b2, bG2, bG3 = bias["b1"], bias["b2"], bias["bG2"], bias["bGamma3"]
    cs0, cs2, cs4, cs = bias["cs0"], bias["cs2"], bias["cs4"], bias["cs"]
    Pshot, b4 = bias["Pshot"], bias["b4"]
    h2, h3 = h**2, h**3
    b4k = fz**2 * b4 * kh**2 * (35.0 / 8.0) * pm[13] * h     # shared b4 k^2 mu^4 factor
    out = {}
    # classy.pyx:4795-4800  pk_mm_real(cs)
    out["pk_mm_real"] = (pm[0] + pm[14] + 2.0 * cs * pm[10] / h2) * h3
    # classy.pyx:4803-4814  pk_gg_real(b1,b2,bG2,bGamma3,cs,cs0,Pshot)
    out["pk_gg_real"] = (b1**2 * pm[14] + b1**2 * pm[0]
                         + 2.0 * (cs * b1**2 + cs0 * b1) * pm[10] / h2
                         + b1 * b2 * pm[2] + 0.25 * b2**2 * pm[1]
                         + 2.0 * b1 * bG2 * pm[3] + b1 * (2.0 * bG2 + 0.8 * bG3) * pm[6]
                         + bG2**2 * pm[5] + b2 * bG2 * pm[4]) * h3 + Pshot
    # classy.pyx:4817-4826  pk_gm_real(b1,b2,bG2,bGamma3,cs,cs0)
    out["pk_gm_real"] = (b1 * pm[14] + b1 * pm[0] + (2.0 * cs * b1 + cs0) * pm[10] / h2
                         + 0.5 * b2 * pm[2] + bG2 * pm[3]
                         + (bG2 + 0.4 * bG3) * pm[6]) * h3
    # classy.pyx:4829-4851  pk_mm_l0/l2/l4(cs0/cs2/cs4)
    out["pk_mm_l0"] = (pm[15] + pm[21] + pm[16] + pm[22] + pm[17] + pm[23]
                       + 2.0 * cs0 * pm[11] / h2) * h3
    out["pk_mm_l2"] = (pm[18] + pm[24] + pm[19] + pm[25] + pm[26]
                       + 2.0 * cs2 * pm[12] / h2) * h3
    out["pk_mm_l4"] = (pm[20] + pm[27] + pm[28] + pm[29] + 2.0 * cs4 * pm[13] / h2) * h3
    # classy.pyx:4854-4880  pk_gg_l0(b1,b2,bG2,bGamma3,cs0,Pshot_nbar,a0_nbar,a2_nbar,b4)
    out["pk_gg_l0"] = ((pm[15] + pm[21] + b1 * pm[16] + b1 * pm[22]
                        + b1**2 * pm[17] + b1**2 * pm[23]
                        + 0.25 * b2**2 * pm[1] + b1 * b2 * pm[30] + b2 * pm[31]
                        + b1 * bG2 * pm[32] + bG2 * pm[33] + b2 * bG2 * pm[4] + bG2**2 * pm[5]
                        + 2.0 * cs0 * pm[11] / h2
                        + (2.0 * bG2 + 0.8 * bG3) * (b1 * pm[7] + pm[8])) * h3
                       + Pshot + 0.25 * b2**2 * Pd2d2_0
                       + b4k * (fz**2 / 9.0 + 2.0 * fz * b1 / 7.0 + b1**2 / 5.0))
    # classy.pyx:4883-4899  pk_gg_l2(b1,b2,bG2,bGamma3,cs2,a2_nbar,b4)
    out["pk_gg_l2"] = ((pm[18] + pm[24] + b1 * pm[19] + b1 * pm[25] + b1**2 * pm[26]
                        + b1 * b2 * pm[34] + b2 * pm[35] + b1 * bG2 * pm[36] + bG2 * pm[37]
                        + 2.0 * cs2 * pm[12] / h2 + (2.0 * bG2 + 0.8 * bG3) * pm[9]) * h3
                       + b4k * (70.0 * fz**2 + 165.0 * fz * b1 + 99.0 * b1**2) * 4.0 / 693.0)
    # classy.pyx:4902-4915  pk_gg_l4(b1,b2,bG2,bGamma3,cs4,b4)
    out["pk_gg_l4"] = ((pm[20] + pm[27] + b1 * pm[28] + b1**2 * pm[29]
                        + b2 * pm[38] + bG2 * pm[39] + 2.0 * cs4 * pm[13] / h2) * h3
                       + b4k * (210.0 * fz**2 + 390.0 * fz * b1 + 143.0 * b1**2) * 8.0 / 5005.0)
    return out
```

Before trusting the transcription, open `/home/n2minh/CLASS-PT/python/classy.pyx` lines 4795–4915 and diff every term against the code above (ref §11 was written from the same lines; a second pair of eyes here is the whole point of the twin). Then run:

`PYTHONPATH=/home/n2minh/clax-ptval python -m pytest tests/test_classpt_assembly.py -q`

Expected: `test_pd2d2_0_power_law` and `test_twin_reproduces_legacy_matter_and_real_space` pass; `test_twin_decides_legacy_kh_convention` fails on the last assert with `hits` showing one label < 1e-8. Set `LEGACY_KH_CONVENTION` to that label with a one-line comment stating the measured residuals of both candidates, rerun, all pass. If neither label is < 1e-8 (legacy classy differed in more than kh units), leave `LEGACY_KH_CONVENTION = None`, mark that test `xfail(strict=True, reason=...)` with the two residuals in the reason, and record in the commit message that legacy `pk_gg_*` are non-reproducible — Task A4 then gates on `pk_mult` rows and `pk_mm_*` only.

- [ ] **Step 3: Rewrite the generator**

```python
#!/usr/bin/env python
# scripts/generate_classpt_reference.py
"""Generate CLASS-PT reference multipoles for the clax-pt validation campaign.

Runs in the `classpt` env (Task A2) only:
    micromamba run -n classpt env PYTHONPATH=<repo> python scripts/generate_classpt_reference.py \
        --cosmology lcdm_fiducial --z-list 0 0.38 0.8
Writes validation_cosmologies.reference_path(...) per z (spec §4.8) with the
raw `get_pk_mult` rows plus every classy accessor, and asserts the NumPy twin
(scripts/classpt_assembly.py) reproduces classy to 1e-10 on each file.
`--legacy` regenerates the legacy fiducial (LEGACY_CLASSPT_FIDUCIAL, cb=No,
BBN YHe) for the provenance gate (Task A4).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts import classpt_assembly as ca
from scripts import validation_cosmologies as vc

CLASSPT_DIR = Path("/home/n2minh/CLASS-PT")


def _classpt_commit() -> str:
    return subprocess.run(["git", "-C", str(CLASSPT_DIR), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def _patch_shas() -> dict:
    pdir = vc.REPO_ROOT / "scripts" / "classpt_patches"
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(pdir.glob("*.patch"))}


def _assert_close(name, a, b, rtol=1e-10):
    a, b = np.asarray(a, float), np.asarray(b, float)
    scale = max(np.max(np.abs(b)), 1e-300)
    err = np.max(np.abs(a - b)) / scale
    if not err < rtol:
        raise AssertionError(f"ERROR twin mismatch {name}: max|twin-classy|/max|classy| = {err:.3e}")


def run(case: str, cosmo: dict, z_list, *, ap: bool, omfid: float, cb: bool, bias_name: str,
        yhe, use_ppf, tag: str, outdir: Path | None) -> list[Path]:
    from classy import Class

    prm = vc.classpt_params_from(cosmo, z_list=z_list, ap=ap, omfid=omfid, cb=cb, yhe=yhe, use_ppf=use_ppf)
    bias = vc.BIAS if bias_name == "fiducial" else vc.BIAS_NONZERO
    M = Class()
    M.set(prm)
    M.compute()
    if not (hasattr(M, "get_ap_ratios") and hasattr(M, "get_Pd2d2_0")):
        sys.exit("ERROR classy is unpatched: run scripts/setup_classpt_env.sh (Task A2)")
    h = M.h()
    k_h = vc.ept_kgrid_numpy()
    k = k_h * h                                   # CLASS units for every classy call
    n_ncdm = int(prm.get("N_ncdm", 0))
    written = []
    for z in z_list:
        M.initialize_output(k, z, len(k_h))       # sets kh (h/Mpc, patched), fz, pk_mult, Pd2d2_0
        pm = np.asarray(M.get_pk_mult(k, z, len(k_h)), dtype=float)
        fz = float(M.scale_independent_growth_factor_f(z))
        hratio, Dratio, growthf = M.get_ap_ratios(z)
        Pd2d2_0 = float(M.get_Pd2d2_0())
        pk_m_lin = np.array([M.pk_lin(ki, z) for ki in k]) * h**3
        pk_cb_lin = np.array([M.pk_cb_lin(ki, z) for ki in k]) * h**3 if n_ncdm > 0 else None
        pk_lin = pk_cb_lin if cb else pk_m_lin
        b = bias
        classy_out = {
            "pk_mm_real": M.pk_mm_real(cs=b["cs"]),
            "pk_gg_real": M.pk_gg_real(b1=b["b1"], b2=b["b2"], bG2=b["bG2"], bGamma3=b["bGamma3"],
                                       cs=b["cs"], cs0=b["cs0"], Pshot=b["Pshot"]),
            "pk_gm_real": M.pk_gm_real(b1=b["b1"], b2=b["b2"], bG2=b["bG2"], bGamma3=b["bGamma3"],
                                       cs=b["cs"], cs0=b["cs0"]),
            "pk_mm_l0": M.pk_mm_l0(cs0=b["cs0"]),
            "pk_mm_l2": M.pk_mm_l2(cs2=b["cs2"]),
            "pk_mm_l4": M.pk_mm_l4(cs4=b["cs4"]),
            "pk_gg_l0": M.pk_gg_l0(b1=b["b1"], b2=b["b2"], bG2=b["bG2"], bGamma3=b["bGamma3"],
                                   cs0=b["cs0"], Pshot_nbar=b["Pshot"], a0_nbar=0.0, a2_nbar=0.0,
                                   b4=b["b4"]),
            "pk_gg_l2": M.pk_gg_l2(b1=b["b1"], b2=b["b2"], bG2=b["bG2"], bGamma3=b["bGamma3"],
                                   cs2=b["cs2"], a2_nbar=0.0, b4=b["b4"]),
            "pk_gg_l4": M.pk_gg_l4(b1=b["b1"], b2=b["b2"], bG2=b["bG2"], bGamma3=b["bGamma3"],
                                   cs4=b["cs4"], b4=b["b4"]),
        }
        classy_out = {kk: np.asarray(v, dtype=float) for kk, v in classy_out.items()}
        # --- falsify the twin against classy on this very file ---
        twin = ca.assemble_from_pm(pm, h, fz, k_h, bias, Pd2d2_0)
        for kk in classy_out:
            _assert_close(kk, twin[kk], classy_out[kk])
        _assert_close("Pd2d2_0", ca.pd2d2_0(pm[14] * h**3, k_h), Pd2d2_0, rtol=1e-8)
        _assert_close("growthf==fz", growthf, fz, rtol=1e-8)
        _assert_close("pm[14]==pk_lin(IR-resummed tree differs: expect fail)", pm[14] * h**3, pk_lin, rtol=1.0)  # sanity: same order of magnitude only
        if not ap:
            _assert_close("hratio", hratio, 1.0, rtol=1e-14)
            _assert_close("Dratio", Dratio, 1.0, rtol=1e-14)
        path = vc.reference_path(case, z, ap=ap, omfid=omfid, cb=cb, bias=bias_name, tag=tag)
        if outdir is not None:
            path = Path(outdir) / path.relative_to(vc.REFERENCE_ROOT)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            k_h=k_h, z=z, h=h, fz=fz, growthf=growthf,
            D_z=float(M.scale_independent_growth_factor(z)), H_z=float(M.Hubble(z)),
            DA_z=float(M.angular_distance(z)), rs_d=float(M.rs_drag()),   # Mpc; nonlinear_pt.c:2919 rbao
            hratio=hratio, Dratio=Dratio, Pd2d2_0=Pd2d2_0,
            pk_lin=pk_lin, pk_m_lin=pk_m_lin, pk_mult=pm, kh_convention="h/Mpc",
            ap=ap, omfid=omfid, cb=cb, use_ppf=str(prm.get("use_ppf", "default")),
            params_json=json.dumps(prm, sort_keys=True), bias_json=json.dumps(bias, sort_keys=True),
            classpt_commit=_classpt_commit(), patches_sha256=json.dumps(_patch_shas(), sort_keys=True),
            **classy_out,
        )
        if pk_cb_lin is not None:
            payload["pk_cb_lin"] = pk_cb_lin
        np.savez(path, **payload)
        print(f"wrote {path}  hratio={hratio:.6f} Dratio={Dratio:.6f} f={fz:.6f}")
        written.append(path)
    M.struct_cleanup()
    M.empty()
    return written


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--cosmology", choices=list(vc.CASES))
    g.add_argument("--legacy", action="store_true", help="LEGACY_CLASSPT_FIDUCIAL, cb=No, BBN YHe")
    g.add_argument("--list-distinct", action="store_true")
    p.add_argument("--z-list", type=float, nargs="+", default=list(vc.Z_LIST))
    p.add_argument("--ap", choices=["yes", "no"], default="yes")
    p.add_argument("--omfid", type=float, default=vc.OMFID)
    p.add_argument("--cb", choices=["yes", "no"], default="yes")
    p.add_argument("--bias", choices=["fiducial", "nonzero"], default="fiducial")
    p.add_argument("--yhe", default=str(vc.Y_HE_CLAX), help="float or 'none' (CLASS-PT BBN default)")
    p.add_argument("--use-ppf", choices=["default", "yes", "no"], default="default")
    p.add_argument("--tag", default="")
    p.add_argument("--outdir", default=None)
    a = p.parse_args(argv)
    if a.list_distinct:
        print("\n".join(vc.distinct_cases()))
        return
    yhe = None if a.yhe.lower() == "none" else float(a.yhe)
    use_ppf = {"default": None, "yes": True, "no": False}[a.use_ppf]
    if a.legacy:
        run("legacy_fiducial", vc.LEGACY_CLASSPT_FIDUCIAL, a.z_list, ap=a.ap == "yes", omfid=a.omfid,
            cb=False, bias_name=a.bias, yhe=None, use_ppf=use_ppf, tag=a.tag, outdir=a.outdir)
    else:
        run(a.cosmology, vc.cosmo_params(a.cosmology), a.z_list, ap=a.ap == "yes", omfid=a.omfid,
            cb=a.cb == "yes", bias_name=a.bias, yhe=yhe, use_ppf=use_ppf, tag=a.tag, outdir=a.outdir)


if __name__ == "__main__":
    main()
```

Remove the line `_assert_close("pm[14]==pk_lin(...)", ..., rtol=1.0)` before committing if it fires: it exists only to remind you that `pm[14]` is the IR-resummed tree, not `pk_lin` — they differ at the BAO scale by ~1–5%, so a `rtol=1.0` check is a magnitude sanity check, nothing more. `reference_path("legacy_fiducial", ...)` works because `canonical_case` passes unknown names through.

- [ ] **Step 4: Smoke-run the generator on the legacy fiducial (login node OK — pure CLASS-PT, ~1 min)**

```bash
mkdir -p /lustre/work/n2minh/std/clax/ptval
micromamba run -n classpt env PYTHONPATH=/home/n2minh/clax-ptval python \
  scripts/generate_classpt_reference.py --legacy --z-list 0.38 2>&1 | tail -5
```
Expected: `wrote .../reference_data/classpt/legacy_fiducial/z0.380_ap_omfid0.31_m.npz  hratio=... Dratio=... f=0.716647...`. If a twin assertion fires, the twin or ref §11 is wrong for that accessor — fix the twin from `classy.pyx`, never loosen `rtol`.

Then the same cosmology with AP off — `--legacy --z-list 0.38 --ap no` → `legacy_fiducial/z0.380_noap_m.npz` (expected `hratio=1 Dratio=1` exactly). Track B needs it: the legacy fiducial has Ωm = 0.14237/0.6736² = 0.3138 ≠ Omfid, so the AP file carries hratio, Dratio ≈ 1 ± 2e-3 and is *not* an α=1 oracle; the noap file is (Part 1b, B3 Step 1 and B5).

Then a single campaign case to confirm cb + ncdm + the `pk_cb_lin` path: `--cosmology massive_nu_015 --z-list 0.38` (expected: `pk_cb_lin` present, `hratio≠1`). Delete that file afterwards (`git status` must show only `legacy_fiducial/`), A5 regenerates everything.

- [ ] **Step 5: Commit** — `feat(ptval): NumPy classy twin + rewritten CLASS-PT reference generator (A3)`. Include both `reference_data/classpt/legacy_fiducial/z0.380_ap_omfid0.31_m.npz` (the A4 gate target) and `reference_data/classpt/legacy_fiducial/z0.380_noap_m.npz` (Track B's α=1 oracle), ~250 KB each.

---

### Task A4: Provenance gate — the new oracle reproduces the legacy file

**Files:**
- Create: `tests/test_classpt_provenance.py`

**Interfaces:**
- Consumes: `reference_data/classpt_z0.38_fullrange.npz` (legacy), `reference_data/classpt/legacy_fiducial/z0.380_ap_omfid0.31_m.npz` (A3 Step 4), `classpt_assembly.LEGACY_KH_CONVENTION`, `PM_ROWS_VALID`.

- [ ] **Step 1: Write the gate**

```python
# tests/test_classpt_provenance.py
"""Spec §4.2: the rebuilt, patched CLASS-PT must reproduce the legacy z=0.38
reference before any new reference is trusted.  Tolerances are 1e-6 on the
raw pk_mult rows (same code, same inputs, different compiler/BLAS) — do not
loosen them; a larger discrepancy means the inputs differ (Part 0 findings:
N_ur, YHe, cb defaults) and must be explained, not absorbed.

Exempt from the multi-cosmology rule: this is a single-file provenance check.
"""
import json

import numpy as np
import pytest

from scripts import classpt_assembly as ca
from scripts import validation_cosmologies as vc

LEGACY = vc.REPO_ROOT / "reference_data" / "classpt_z0.38_fullrange.npz"
NEW = vc.REFERENCE_ROOT / "legacy_fiducial" / "z0.380_ap_omfid0.31_m.npz"


@pytest.fixture(scope="module")
def pair():
    for p in (LEGACY, NEW):
        if not p.exists():
            pytest.skip(f"{p} missing")
    return np.load(LEGACY), np.load(NEW)


def _rel(a, b):
    return np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))


def test_inputs_are_the_legacy_inputs(pair):
    _, new = pair
    prm = json.loads(str(new["params_json"]))
    assert prm["A_s"] == 2.0989e-9 and "N_ncdm" not in prm and "YHe" not in prm
    assert prm["cb"] == "No" and prm["AP"] == "Yes" and prm["Omfid"] == "0.31"
    assert str(new["kh_convention"]) == "h/Mpc"


def test_grid_and_background_match(pair):
    old, new = pair
    assert np.array_equal(old["k_h"], new["k_h"])
    assert abs(float(old["h"]) - float(new["h"])) < 1e-12
    assert abs(float(old["fz"]) - float(new["fz"])) < 1e-10 * float(old["fz"])


def test_pk_mult_rows_match(pair):
    old, new = pair
    rows = range(*ca.PM_ROWS_VALID.indices(96))
    bad = {r: _rel(new["pk_mult"][r], old["pk_mult"][r]) for r in rows
           if _rel(new["pk_mult"][r], old["pk_mult"][r]) >= 1e-6}
    assert not bad, f"ERROR pk_mult rows beyond 1e-6: {bad}"


def test_matter_multipoles_match(pair):
    old, new = pair
    for key in ("pk_mm_real", "pk_mm_l0", "pk_mm_l2", "pk_mm_l4", "pk_gg_real"):
        assert _rel(new[key], old[key]) < 1e-6, key
    assert _rel(new["pk_gm_real"], old["pk_mg_real"]) < 1e-6


def test_galaxy_multipoles_match_in_legacy_kh_convention(pair):
    """New files store pk_gg_* with kh in h/Mpc; the legacy file used
    LEGACY_KH_CONVENTION.  Re-assemble the new pm rows in the legacy
    convention and compare — this is the only place the two conventions meet."""
    old, new = pair
    if ca.LEGACY_KH_CONVENTION is None:
        pytest.xfail("legacy pk_gg_* non-reproducible (see test_classpt_assembly)")
    h, fz = float(new["h"]), float(new["fz"])
    kh = new["k_h"] if ca.LEGACY_KH_CONVENTION == "h/Mpc" else new["k_h"] * h
    bias = json.loads(str(new["bias_json"]))
    out = ca.assemble_from_pm(new["pk_mult"], h, fz, kh, bias, ca.pd2d2_0(new["pk_mult"][14] * h**3, kh))
    for key in ("pk_gg_l0", "pk_gg_l2", "pk_gg_l4"):
        assert _rel(out[key], old[key]) < 1e-6, key
```

- [ ] **Step 2: Run it**

`PYTHONPATH=/home/n2minh/clax-ptval python -m pytest tests/test_classpt_provenance.py tests/test_classpt_assembly.py -q`

Expected: all pass. If `test_pk_mult_rows_match` fails, print the offending rows' max residual and k index (5 lines, not arrays) and check, in order: (1) `N_ur` — legacy had none → 3.044; (2) `YHe` — legacy had none → BBN; (3) cb — the legacy run had `cb=TRUE` reading `index_tp_delta_cb`; with `output=mPk` only, `index_tp_delta_m` is index 0 and the unassigned `delta_cb` index is also 0, so `cb=No` must reproduce it. If (3) is the discrepancy, the 0-index coincidence does not hold in this build: regenerate with `--cb yes` on an `N_ncdm=0` run only for this gate (document that it is UB-by-construction in the test docstring) and keep `cb=No` everywhere else. Report which it was in the commit message.

- [ ] **Step 3: Commit** — `test(ptval): CLASS-PT provenance gate reproduces legacy z=0.38 reference (A4)`.

---

### Task A5: Full reference generation job + MANIFEST

**Files:**
- Create: `slurm/classpt-refgen.sbatch`
- Create: `scripts/write_classpt_manifest.py`
- Create (generated): `reference_data/classpt/<case>/*.npz` (14 × 3 campaign files + diagnostics), `reference_data/classpt/MANIFEST.md`

**Interfaces:**
- Consumes: A3 generator CLI, A1 `distinct_cases()`, Part 0 classpt CPU sbatch template.
- Produces: the campaign reference set consumed by Track C (C1–C3) and B7:
  - campaign: every distinct case × `Z_LIST`, `ap=yes cb=yes omfid=0.31 bias=fiducial`
  - diagnostics at `lcdm_fiducial` z=0.38: `--cb no` (`..._m.npz`), `--ap no` (`z0.380_noap_cb.npz`), `--bias nonzero` (`..._biasnz.npz`); at `w0wa_m07_m10` z=0.38: `--use-ppf no --tag noppf` (if CLASS-PT rejects `use_ppf=no` for a w=−1 crossing, record the error text in MANIFEST under "Skipped" and move on)
  - `MANIFEST.md`: one row per npz — path, sha256, `classpt_commit`, `patches_sha256`, `hratio`, `Dratio`, `fz`, `params_json` hash — plus a "Skipped" section.

- [ ] **Step 1: Write the manifest writer**

```python
#!/usr/bin/env python
# scripts/write_classpt_manifest.py
"""Scan reference_data/classpt/**/*.npz and write MANIFEST.md (spec §4.8).

Runs in either env (numpy only).  Idempotent; rerun after any regeneration.
Skipped runs are appended from --skipped "path: reason" arguments.
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np

from scripts import validation_cosmologies as vc


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skipped", nargs="*", default=[], help='"relative/path.npz: reason"')
    a = p.parse_args(argv)
    rows = []
    for path in sorted(vc.REFERENCE_ROOT.rglob("*.npz")):
        d = np.load(path)
        rel = path.relative_to(vc.REFERENCE_ROOT)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        psha = hashlib.sha256(str(d["params_json"]).encode()).hexdigest()[:12]
        rows.append(f"| `{rel}` | `{sha}` | `{d['classpt_commit']}` | {float(d['hratio']):.6f} | "
                    f"{float(d['Dratio']):.6f} | {float(d['fz']):.6f} | `{psha}` |")
    patches = json.loads(str(d["patches_sha256"])) if rows else {}
    lines = ["# CLASS-PT reference manifest", "",
             "Generated by `scripts/write_classpt_manifest.py`; files by "
             "`scripts/generate_classpt_reference.py` in the `classpt` env "
             "(`scripts/setup_classpt_env.sh`). Layout: spec §4.8.", "",
             "Patches: " + ", ".join(f"`{k}` `{v[:16]}`" for k, v in patches.items()), "",
             "| file | sha256[:16] | CLASS-PT | hratio | Dratio | f | params sha[:12] |",
             "|---|---|---|---|---|---|---|", *rows, ""]
    if a.skipped:
        lines += ["## Skipped", "", *[f"- {s}" for s in a.skipped], ""]
    (vc.REFERENCE_ROOT / "MANIFEST.md").write_text("\n".join(lines))
    print(f"MANIFEST.md: {len(rows)} files, {len(a.skipped)} skipped")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the sbatch** (Part 0 classpt CPU template; body below)

```bash
#!/bin/bash -l
#SBATCH --job-name=classpt-refgen
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/lustre/work/n2minh/std/clax/ptval/%x-%j.out
#SBATCH --error=/lustre/work/n2minh/std/clax/ptval/%x-%j.err
set -uo pipefail
eval "$(micromamba shell hook --shell bash)"
micromamba activate classpt
export OMP_NUM_THREADS=8 PYTHONPATH=/home/n2minh/clax-ptval
cd /home/n2minh/clax-ptval
GEN="python scripts/generate_classpt_reference.py"
FAILED=()
for c in $($GEN --list-distinct); do
  $GEN --cosmology "$c" --z-list 0 0.38 0.8 || FAILED+=("$c")
done
$GEN --cosmology lcdm_fiducial --z-list 0.38 --cb no        || FAILED+=("lcdm_fiducial:cb-no")
$GEN --cosmology lcdm_fiducial --z-list 0.38 --ap no        || FAILED+=("lcdm_fiducial:ap-no")
$GEN --cosmology lcdm_fiducial --z-list 0.38 --bias nonzero || FAILED+=("lcdm_fiducial:biasnz")
SKIPPED=()
$GEN --cosmology w0wa_m07_m10 --z-list 0.38 --use-ppf no --tag noppf \
  || SKIPPED+=("w0wa_m07_m10/z0.380_ap_omfid0.31_cb_noppf.npz: use_ppf=no rejected or failed (see job log)")
python scripts/write_classpt_manifest.py --skipped "${SKIPPED[@]}"
echo "FAILED: ${FAILED[*]:-none}"
[ ${#FAILED[@]} -eq 0 ]
```
`--cb no` at `lcdm_fiducial` is legal (N_ncdm=1). The job does no git operations.

- [ ] **Step 3: Submit, poll, inspect**

`mkdir -p /lustre/work/n2minh/std/clax/ptval && sbatch slurm/classpt-refgen.sbatch`; poll with `squeue -u n2minh --name=classpt-refgen -h` (expected wall time 20–60 min: 15 CLASS-PT runs × 3 z, each ≈30–90 s). Then: `grep -c "^wrote" <out>` = 46 (42 campaign + 3 diagnostics + 1 noppf, or 45 if noppf was skipped); `grep ERROR <err>` empty; `tail -1 <out>` shows `FAILED: none`.

Spot-check physics (login node, numpy only): `hratio, Dratio` = 1 exactly for the `noap` file; for `h_high` they differ from `lcdm_fiducial`'s; `massive_nu_030` `pk_cb_lin/pk_m_lin` at k=0.1 h/Mpc ≈ 1.02–1.03 (the cb spectrum sits above total matter by ≈ 2 f_ν); `w0wa_m07_m10` `D_z` at z=0.8 differs from `lcdm_fiducial`'s. Put the four numbers in the commit message.

- [ ] **Step 4: Commit** — `data(ptval): CLASS-PT reference set, 14 cases × 3 z + diagnostics, MANIFEST (A5)`. Total size ≈ 12 MB (46 × ~250 KB); if `git` complains about size, that is unexpected — stop and report rather than adding LFS.

---

## Track A self-check (before handing to Track C)

- `reference_data/classpt/` has 14 case dirs + `legacy_fiducial/` + `MANIFEST.md`; every campaign path returned by `validation_cosmologies.reference_path(case, z)` for `case in distinct_cases()`, `z in Z_LIST` exists.
- `pytest tests/test_validation_cosmologies.py tests/test_classpt_assembly.py tests/test_classpt_provenance.py -q` green in the `clax` env (CPU).
- `LEGACY_KH_CONVENTION` is set (or its xfail is documented) and the commit messages for A3/A4 state which convention the legacy file used and which of N_ur/YHe/cb explained any residual.
