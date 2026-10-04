# clax-pt Validation — Part 2: `compute_ept_from_clax` wiring, multi-cosmology campaign, report (Track C)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Read first:** `2026-09-03-clax-pt-validation-part0-common.md` (constraints, oracle findings, run recipes, commit recipe, reviewer briefs) and `2026-09-03-clax-pt-validation-classpt-inloop-reference.md` (ref §; every `ref §n` below points there). Track C starts only when A5 (reference files + MANIFEST), B6 (`compute_ept(..., hratio, Dratio)` with gradients) and P1 (`PerturbationResult.delta_cb`) are on `campaign/clax-pt-validation`.

**Goal:** Wire `compute_ept_from_clax(..., omfid, field)` on top of Parts 1a/1b, then run the two-layer validation of clax-pt against CLASS-PT — stage layer (clax.ept on CLASS-PT's own inputs) and end-to-end layer (clax background → perturbations → clax.ept) — over 14 distinct cosmologies × 3 redshifts for the nine spectra (`pk_mm_real, pk_gg_real, pk_gm_real, pk_mm_l0/l2/l4, pk_gg_l0/l2/l4`), and write the report.

**Architecture:** C0 factors the P_lin/f extraction out of `compute_ept_from_clax` into `ept_inputs_from_clax(field=)` and adds the `omfid` → `ap_ratios` wiring. C1 builds the shared campaign utilities (`tests/ept_campaign_utils.py`: spectra names, window, thresholds, comparison, JSONL error log; B7's helpers move here) and the stage-layer test over every reference file. C2 adds the end-to-end layer with the spec §8 seams as separate tests (background → f → P_cb,lin → EPT), one perturbation solve per case, and the pipeline gradient sweep. C3 is the V100 campaign job plus the summarizer that renders `test_logs/ptval/errors.jsonl` into `docs/validation/2026-09-clax-pt-multipoles.md`. C4 ratchets thresholds from the measured worst cases, writes the CHANGELOG entry and opens the draft PR.

**Tech Stack:** Python 3.14 / JAX 0.9.2 (`clax` env), pytest, NumPy, SLURM (V100 `igpu` nodes). No CLASS-PT runs in this part — every oracle number is read from `reference_data/classpt/<case>/*.npz` written by Part 1a A5.

**Spec:** `docs/superpowers/specs/2026-09-03-clax-pt-validation-design.md` — this part implements §4.2 (two layers), §4.3–4.5 (window, AP, cb, f), §4.6 (fast subset), §4.7 (thresholds), §4.9 (report), §5.3 (parameter seams), §6.2 (`compute_ept_from_clax`), §6.6 (job + summarizer), §7 Phases 3–5, §8 (failure policy), §9 (seams), §12 (success criteria).

## Global Constraints

Part 0's **Global Constraints**, **Oracle findings** and **Run recipes** apply verbatim (environments, compute placement, physics rules, the two-tier commit gate, sbatch templates, commit recipe). Track C adds:

- **Thresholds only tighten.** `THRESHOLDS`/`SEAM_THRESHOLDS` in `tests/ept_campaign_utils.py` start at the spec §4.7/§7 numbers; C4 may lower them to ≥ 2× the measured worst case; nobody raises them. A failing threshold is a finding to bisect (spec §8), never a number to move. A precision-preset change on the clax side (C2 `PTVAL_E2E_PREC=contract`) is a legitimate lever; a multiplicative factor is not (CLAUDE.md "Never add fudge factors").
- **One perturbation solve per (case, preset).** C2 caches `(params, prec, bg, th, pt)` per case at module scope; every z reuses the solve (τ-interpolation inside `ept_inputs_from_clax`). Never re-solve inside a parametrized test body.
- **Everything with a `perturbations_solve` is `slow`** and runs only through sbatch (`slurm/ptval-track-c.sbatch`, created in C0, or the campaign job `slurm/ept-multicosmo-e2e.sbatch`, C3). `pytest.mark.slow` is deselected by `--fast` (conftest), so the login-node gate never touches them.
- **Concise output** (CLAUDE.md rule 2): a failing comparison prints one line per spectrum — `pk_gg_l4 2.31% > 2.00% at k=0.297` — and the full per-k tables go to `test_logs/ptval/`.
- **Every comparison writes a JSON record** to `test_logs/ptval/errors.jsonl` via `ept_campaign_utils.log_record` (git sha, layer, case, z, preset, per-spectrum error and k, seams). C3's summarizer reads only this file and the junit XML; a comparison that is not logged does not exist for the report.
- `PYTHONPATH=/home/n2minh/clax-ptval` in every command; the tests import `from scripts import validation_cosmologies as vc` (A1) and `from tests import ept_campaign_utils as cu` (`tests/__init__.py` exists).

---

## File structure

```
clax/ept.py                                  C0  ept_inputs_from_clax(params, bg, pt, z, prec, *, field) ; compute_ept_from_clax(..., *, omfid=None, field="cb")
clax/lensing.py                              C0  :293 passes field="m" (the lensing ratio is a total-matter quantity)
tests/test_ept_from_clax.py                  C0  wiring tests (real background, synthetic tables) + slow delta_cb physics (nulcdm grid)
slurm/ptval-track-c.sbatch                   C0  V100 job running $PTVAL_PYTEST_ARGS (reused by C1/C2 smokes)
docs/superpowers/specs/...-design.md         C0  Phase 3 sign fix ("below 1" → "above 1")
tests/ept_campaign_utils.py                  C1  SPECTRA, window, THRESHOLDS, SEAM_THRESHOLDS, rel, compare_spectra, failures, load/require_reference, log_record, pm_from_leaves, clax_nine (moved from test_ept_assembly)
tests/test_ept_assembly.py                   C1  imports the moved helpers; test bodies unchanged
tests/test_ept_multicosmo.py                 C1  stage layer: 14 cases × 3 z + 4 fiducial diagnostics
tests/test_ept_e2e_multicosmo.py             C2  e2e layer (slow): seams, nine spectra, gradient sweep
slurm/ept-multicosmo-e2e.sbatch              C3  V100 campaign job (both layers, junit, summarizer)
scripts/summarize_ept_validation.py          C3  errors.jsonl + junit → markdown report
docs/validation/2026-09-clax-pt-multipoles.md C3 generated report (committed)
CHANGELOG.md                                 C4  `### Sep D, 2026:` entry; thresholds ratchet commit
```

## Task order

C0 → C1 → C2 → C3 → C4, strictly sequential (each consumes the previous task's files). C0 needs B5/B6/P1; C1 additionally needs A5's reference files on disk (`reference_data/classpt/MANIFEST.md` exists) — without them every stage test skips with the reason `reference missing`, which is the signal to wait, not to proceed.

## Interfaces consumed (exact, from Parts 1a/1b)

- `clax.ept.compute_ept(pk_lin_h, k_h, h, f, prec=EPTPrecisionParams(), _ir_precomputed=None, rs_h=99.0, hratio=1.0, Dratio=1.0) -> EPTComponents` (B5). `rs_h` is r_s(z_d)·h in Mpc/h and is read only on the numpy IR path (`_ir_precomputed=None`); the default 99.0 is the fiducial value and is wrong for any other cosmology — the stage layer passes `rs_d * h` from the reference file.
- `clax.ap.ap_ratios(bg, z: float, omfid: float = OMFID_DEFAULT) -> (hratio, Dratio)` 0-d arrays, `z`/`omfid` static Python floats, `(1, 1)` at z = 0 (B2).
- `PerturbationResult.delta_cb: Float[Array, "Nk Ntau"]`, same τ-grid and normalization as `delta_m`, bit-equal to `delta_m` when `ncdm_q_size = 0` (P1). `MatterPerturbationResult` (from `perturbations_solve_mpk`) has no `delta_cb`.
- `scripts.validation_cosmologies` (A1): `CASES`, `FAMILIES`, `Z_LIST = (0.0, 0.38, 0.8)`, `OMFID = 0.31`, `BIAS` (b1=2, b4=500, rest 0), `BIAS_NONZERO`, `FAST_CASES = ("lcdm_fiducial", "massive_nu_015", "w0wa_m07_m10")`, `FAST_Z = 0.38`, `distinct_cases() -> list[str]` (14), `clax_params(name) -> CosmoParams`, `reference_path(case, z, *, ap=True, omfid=OMFID, cb=True, bias="fiducial", tag="") -> Path`, `canonical_case(name)`.
- Reference npz keys (A3): `k_h (256,), z, h, fz, growthf, D_z, H_z, DA_z, rs_d, hratio, Dratio, Pd2d2_0, pk_lin, pk_m_lin, pk_cb_lin (N_ncdm>0 only), pk_mult (96, 256), pk_mm_real, pk_gg_real, pk_gm_real, pk_mm_l0, pk_mm_l2, pk_mm_l4, pk_gg_l0, pk_gg_l2, pk_gg_l4, kh_convention, ap, omfid, cb, use_ppf, params_json, bias_json, classpt_commit, patches_sha256`. `pk_lin` is the cb spectrum in cb files, in (Mpc/h)³ on `k_h`; `H_z` in 1/Mpc (classy `Hubble`); `rs_d` in Mpc.
- Files on disk after A5: for every `case in distinct_cases()`, `z in Z_LIST`: `reference_path(case, z)`; diagnostics at `lcdm_fiducial`, z=0.38: `reference_path(..., cb=False)`, `reference_path(..., ap=False)`, `reference_path(..., bias="nonzero")`; `reference_path("w0wa_m07_m10", 0.38, tag="noppf")`.
- B7 helpers in `tests/test_ept_assembly.py`: `BIAS_KEYS`, `_pm_from_leaves(e, h) -> (48, Nk)`, `_clax_nine(e, bias) -> dict`, `_rel(a, b) -> float`.
- `tests/pk_test_utils.py`: `PK_FAST_PREC` (40 k/decade, l_max 35, rtol 1e-5, `pt_k_max_cl=1.0`, `ncdm_fluid_approximation="none"`, `pt_k_chunk_size=1`), `PK_CONTRACT_PREC` (60 k/decade, l_max 50, rtol 1e-6).
- `tests/conftest.py`: `lcdm_cosmology`, `nulcdm_cosmology` fixtures (`(name, CosmoParams)`; `--fast` → fiducial only), `fast_mode`.

---

### Task C0: `ept_inputs_from_clax(field)` and `compute_ept_from_clax(omfid, field)`

**Files:**
- Modify: `clax/ept.py:2225-2344` (`compute_ept_from_clax`)
- Modify: `clax/lensing.py:293`
- Modify: `docs/superpowers/specs/2026-09-03-clax-pt-validation-design.md:370-371`
- Create: `tests/test_ept_from_clax.py`
- Create: `slurm/ptval-track-c.sbatch`

**Interfaces:**
- Consumes: `compute_ept(..., hratio, Dratio)` (B5), `clax.ap.ap_ratios` (B2), `PerturbationResult.delta_cb` (P1), `ept_kgrid`, `_ir_resummation_jax`, `sound_horizon_drag`.
- Produces:
  - `clax.ept.ept_inputs_from_clax(params, bg, pt, z: float = 0.0, prec: EPTPrecisionParams = EPTPrecisionParams(), *, field: str = "cb") -> tuple[Float[Array, "Nk"], Float[Array, ""]]` — `(pk_h, f)`: linear P(k, z) of the chosen field on `ept_kgrid(prec)` in (Mpc/h)³, and f(z) from `bg.f_of_loga`. `field="cb"` reads `pt.delta_cb` (`ValueError` if the object has none), `field="m"` reads `pt.delta_m`, anything else `ValueError`. C2's seam tests call this directly.
  - `clax.ept.compute_ept_from_clax(params, bg, pt, z: float = 0.0, prec: EPTPrecisionParams = EPTPrecisionParams(), *, omfid: float | None = None, field: str = "cb") -> EPTComponents` — `omfid=None` → `hratio = Dratio = 1.0`; else `ap_ratios(bg, z, omfid)` (`z` must then be a static Python float). Everything else (traced h, traced IR splitter, `rs_h`) unchanged from bf8ac18.
  - `slurm/ptval-track-c.sbatch`: runs `python -m pytest $PTVAL_PYTEST_ARGS` on a V100; submit with `sbatch --export=ALL,PTVAL_PYTEST_ARGS="<args>" slurm/ptval-track-c.sbatch`.

**What changes for existing callers.** Default `field="cb"` alters numbers only for solves with an active ncdm hierarchy (`ncdm_q_size > 0`), because `delta_cb == delta_m` bit-for-bit otherwise (P1). `tests/test_ept_gradients.py:529, 752` use `PrecisionParams.fast_cl()` (`ncdm_q_size = 0`) and are unaffected; `clax/lensing.py:293` is switched to `field="m"` explicitly (the lensing nonlinear ratio `P_NL/P_lin` is a total-matter quantity, cf. the `compute_linear_matter_pk_from_perturbations` call right below it).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ept_from_clax.py
"""Wiring tests for clax.ept.ept_inputs_from_clax / compute_ept_from_clax(omfid, field)
(clax-pt validation Part 2, Task C0; spec §6.2, §7 Phase 3).

Multi-cosmology rule: the wiring tests sweep `lcdm_cosmology` with a real
background_solve and a synthetic perturbation table (no ODE solve, login-node
cheap); the delta_cb physics tests sweep `nulcdm_cosmology` and are `slow`
(one perturbation solve per neutrino mass, cached at module scope).
"""
import inspect
import types
from dataclasses import replace as _dc_replace

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from clax import CosmoParams, PrecisionParams
from clax.background import background_solve, sound_horizon_drag
from clax.thermodynamics import thermodynamics_solve
from clax.perturbations import perturbations_solve
import clax.ept as ept_mod
from clax.ept import (
    EPTPrecisionParams, ept_kgrid, ept_inputs_from_clax, compute_ept_from_clax,
)
from clax.ap import ap_ratios

Z = 0.38
OMFID = 0.31
BG_PREC = PrecisionParams.fast_cl()          # background only: ~5 s on 2 CPU threads
NMAX = EPTPrecisionParams().nmax


def _synthetic_pt(bg, cb_over_m=1.05, n_k=64, n_tau=48):
    """Stand-in for PerturbationResult: ept_inputs_from_clax reads only
    k_grid, tau_grid, delta_m, delta_cb. delta_m = (k/k0)^-1 (tau/tau0)^2 is
    smooth, positive and tau-dependent (the tau interpolation is exercised);
    delta_cb = cb_over_m * delta_m so the field ratio is known exactly."""
    k = np.logspace(-4, np.log10(3.0), n_k)                     # Mpc^-1
    tau0 = float(bg.tau_of_loga.evaluate(0.0))
    tau = np.linspace(0.05 * tau0, tau0, n_tau)
    dm = (k[:, None] / k[0]) ** -1.0 * (tau[None, :] / tau0) ** 2
    return types.SimpleNamespace(k_grid=jnp.asarray(k), tau_grid=jnp.asarray(tau),
                                 delta_m=jnp.asarray(dm), delta_cb=jnp.asarray(cb_over_m * dm))


def _record_compute_ept(monkeypatch):
    """Replace clax.ept.compute_ept with a recorder so the wiring is tested
    without the 20 s loop; returns the list of recorded kwargs."""
    calls = []

    def fake(pk_lin_h, k_h, h, f, prec=None, _ir_precomputed=None, rs_h=99.0,
             hratio=1.0, Dratio=1.0):
        calls.append(dict(pk_lin_h=pk_lin_h, k_h=k_h, h=h, f=f, rs_h=rs_h,
                          hratio=hratio, Dratio=Dratio, ir=_ir_precomputed))
        return "sentinel"

    monkeypatch.setattr(ept_mod, "compute_ept", fake)
    return calls


# ---------------------------------------------------------------------------
# field selection
# ---------------------------------------------------------------------------

def test_field_selects_delta_cb_or_delta_m(lcdm_cosmology):
    name, params = lcdm_cosmology
    bg = background_solve(params, BG_PREC)
    pt = _synthetic_pt(bg, cb_over_m=1.05)
    pk_cb, f_cb = ept_inputs_from_clax(params, bg, pt, Z, field="cb")
    pk_m, f_m = ept_inputs_from_clax(params, bg, pt, Z, field="m")
    assert pk_cb.shape == (NMAX,) and pk_m.shape == (NMAX,)
    assert np.all(np.isfinite(np.asarray(pk_cb))) and np.all(np.asarray(pk_cb) > 0), name
    np.testing.assert_allclose(np.asarray(pk_cb / pk_m), 1.05 ** 2, rtol=1e-12)
    f_want = float(bg.f_of_loga.evaluate(jnp.log(1.0 / (1.0 + Z))))
    assert float(f_cb) == f_want and float(f_m) == f_want, (name, float(f_cb), f_want)


def test_field_validation():
    params = CosmoParams()
    bg = background_solve(params, BG_PREC)
    pt = _synthetic_pt(bg)
    with pytest.raises(ValueError, match="field"):
        ept_inputs_from_clax(params, bg, pt, Z, field="matter")
    # MatterPerturbationResult-like object (perturbations_solve_mpk): no delta_cb
    no_cb = types.SimpleNamespace(k_grid=pt.k_grid, tau_grid=pt.tau_grid, delta_m=pt.delta_m)
    with pytest.raises(ValueError, match="delta_cb"):
        ept_inputs_from_clax(params, bg, no_cb, Z, field="cb")
    pk_m, _ = ept_inputs_from_clax(params, bg, no_cb, Z, field="m")
    assert pk_m.shape == (NMAX,)


def test_inputs_match_bf8ac18_extraction(lcdm_cosmology):
    """field="m" reproduces the pre-C0 extraction (delta_m spline, primordial
    normalisation, h^3) to round-off: the refactor must not move numbers."""
    name, params = lcdm_cosmology
    bg = background_solve(params, BG_PREC)
    pt = _synthetic_pt(bg)
    from clax.primordial import primordial_scalar_pk
    from clax.interpolation import CubicSpline as CS
    h = params.h
    k_h = ept_kgrid()
    k_mpc = jnp.asarray(k_h) * h
    tau_z = bg.tau_of_loga.evaluate(jnp.log(1.0 / (1.0 + Z)))
    dm_z = jax.vmap(lambda dm_k: CS(pt.tau_grid, dm_k).evaluate(tau_z))(pt.delta_m)
    dm_ept = CS(jnp.log(pt.k_grid), dm_z).evaluate(jnp.log(k_mpc))
    want = 2.0 * jnp.pi ** 2 / k_mpc ** 3 * primordial_scalar_pk(k_mpc, params) * dm_ept ** 2 * h ** 3
    got, _ = ept_inputs_from_clax(params, bg, pt, Z, field="m")
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-12, err_msg=name)


# ---------------------------------------------------------------------------
# omfid wiring
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("z", [0.0, Z])
def test_omfid_wiring(lcdm_cosmology, monkeypatch, z):
    name, params = lcdm_cosmology
    bg = background_solve(params, BG_PREC)
    pt = _synthetic_pt(bg)
    calls = _record_compute_ept(monkeypatch)
    assert compute_ept_from_clax(params, bg, pt, z=z) == "sentinel"
    compute_ept_from_clax(params, bg, pt, z=z, omfid=OMFID)
    none_call, ap_call = calls
    assert (float(none_call["hratio"]), float(none_call["Dratio"])) == (1.0, 1.0)
    hr, Dr = ap_ratios(bg, z, OMFID)
    assert float(ap_call["hratio"]) == float(hr) and float(ap_call["Dratio"]) == float(Dr), name
    if z == 0.0:
        assert (float(hr), float(Dr)) == (1.0, 1.0)          # B2: AP is the identity at z = 0
    else:
        assert abs(float(hr) - 1.0) > 1e-4, (name, float(hr))  # Omega_m != 0.31 on every grid point
    # the inputs reaching compute_ept are ept_inputs_from_clax's, cb by default
    pk_h, f = ept_inputs_from_clax(params, bg, pt, z, field="cb")
    for call in calls:
        np.testing.assert_allclose(np.asarray(call["pk_lin_h"]), np.asarray(pk_h), rtol=0, atol=0)
        assert float(call["f"]) == float(f)
        assert np.isclose(float(call["rs_h"]), float(sound_horizon_drag(params)) * float(params.h))
        assert call["ir"] is not None                          # traced IR splitter still in use


def test_defaults_and_lensing_field():
    sig = inspect.signature(compute_ept_from_clax)
    assert sig.parameters["field"].default == "cb"
    assert sig.parameters["omfid"].default is None
    assert sig.parameters["field"].kind is inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["omfid"].kind is inspect.Parameter.KEYWORD_ONLY
    import clax.lensing
    src = inspect.getsource(clax.lensing)
    assert 'compute_ept_from_clax(params, bg, pt, z=0.0, field="m")' in src, (
        "clax.lensing must request the total-matter field explicitly: "
        "the CMB-lensing nonlinear ratio is P_mm,NL / P_mm,lin")


# ---------------------------------------------------------------------------
# delta_cb physics (spec §7 Phase 3; sign corrected in C0: delta_cb > delta_m
# above the free-streaming scale because delta_nu < delta_cb there)
# ---------------------------------------------------------------------------

DELTA_PREC = _dc_replace(PrecisionParams.fast_cl(), ncdm_q_size=5, pt_k_max_cl=0.3,
                         pt_k_chunk_size=20)
K_LOW_H, K_HIGH_H = 1e-3, 0.3          # h/Mpc
NU_MASSES = (0.06, 0.15, 0.30)         # COSMOLOGY_GRID_NULCDM
_SOLVES: dict[float, tuple] = {}       # m_ncdm -> (params, bg, pt): one solve per mass


def _solve(params):
    key = float(params.m_ncdm)
    if key not in _SOLVES:
        bg = background_solve(params, DELTA_PREC)
        th = thermodynamics_solve(params, DELTA_PREC, bg)
        pt = perturbations_solve(params, DELTA_PREC, bg, th)
        _SOLVES[key] = (params, bg, pt)
    return _SOLVES[key]


def _cb_over_m(params, bg, pt, z):
    pk_cb, _ = ept_inputs_from_clax(params, bg, pt, z, field="cb")
    pk_m, _ = ept_inputs_from_clax(params, bg, pt, z, field="m")
    return np.sqrt(np.asarray(pk_cb) / np.asarray(pk_m))       # = delta_cb / delta_m


def _f_nu(params):
    omega_nu = float(params.m_ncdm) / 93.14
    return omega_nu / (float(params.omega_b) + float(params.omega_cdm) + omega_nu)


@pytest.mark.slow
def test_delta_cb_over_delta_m_physics(nulcdm_cosmology):
    """delta_cb/delta_m -> 1 as k -> 0 (k <= 1e-3 h/Mpc, below every free-
    streaming scale on the grid) and sits in (1, 1/(1 - f_nu)] at 0.3 h/Mpc
    (delta_m = (1 - f_nu) delta_cb + f_nu delta_nu with 0 <= delta_nu < delta_cb)."""
    name, params = nulcdm_cosmology
    params, bg, pt = _solve(params)
    k_h = ept_kgrid()
    r = _cb_over_m(params, bg, pt, 0.0)
    low = k_h <= K_LOW_H
    assert np.max(np.abs(r[low] - 1.0)) < 1e-3, (name, float(np.max(np.abs(r[low] - 1.0))))
    i = int(np.argmin(np.abs(k_h - K_HIGH_H)))
    f_nu = _f_nu(params)
    assert 1e-4 < r[i] - 1.0 <= f_nu / (1.0 - f_nu) + 1e-6, (name, float(r[i] - 1.0), f_nu)
    print(f"{name}: delta_cb/delta_m - 1 = {r[i] - 1.0:.3e} at k = {k_h[i]:.3f} h/Mpc (f_nu = {f_nu:.4f})")


@pytest.mark.slow
def test_delta_cb_suppression_grows_with_m_ncdm(fast_mode):
    """|delta_cb/delta_m - 1| at 0.3 h/Mpc increases monotonically over
    m_ncdm = 0.06, 0.15, 0.30 eV (free-streaming suppression grows with f_nu)."""
    if fast_mode:
        pytest.skip("needs three neutrino masses (full mode)")
    k_h = ept_kgrid()
    i = int(np.argmin(np.abs(k_h - K_HIGH_H)))
    vals = []
    for m in NU_MASSES:
        params, bg, pt = _solve(CosmoParams(m_ncdm=m))
        vals.append(float(_cb_over_m(params, bg, pt, 0.0)[i] - 1.0))
    assert vals == sorted(vals) and vals[0] < vals[-1], dict(zip(NU_MASSES, vals))
```

- [ ] **Step 2: Run to verify the failure**

```bash
cd /home/n2minh/clax-ptval && JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2" PYTHONPATH=/home/n2minh/clax-ptval timeout 120 /home/n2minh/micromamba/envs/clax/bin/python -m pytest tests/test_ept_from_clax.py --fast -x -q -p no:cacheprovider 2>&1 | tail -n 5
```
Expected: collection error `ImportError: cannot import name 'ept_inputs_from_clax' from 'clax.ept'`.

- [ ] **Step 3: Implement `ept_inputs_from_clax` and the new `compute_ept_from_clax`**

Replace `clax/ept.py:2225-2344` (the whole `compute_ept_from_clax`, from `def compute_ept_from_clax(` to the closing `_ir_precomputed=ir_pre)`) with the two functions below. The bodies are the bf8ac18 code with `delta` substituted for `pt.delta_m`; keep the existing comments (stop_gradient rationale, h-resampling channel, f-of-loga note) where they are shown.

```python
def ept_inputs_from_clax(
    params,           # CosmoParams
    bg,               # BackgroundResult
    pt,               # PerturbationResult (field="cb" reads .delta_cb)
    z: float = 0.0,
    prec: EPTPrecisionParams = EPTPrecisionParams(),
    *,
    field: str = "cb",
) -> tuple[Float[Array, "Nk"], Float[Array, ""]]:
    """Linear P(k, z) on the EPT k-grid in (Mpc/h)^3 and the growth rate f(z).

    field="cb" (default) samples ``pt.delta_cb`` -- CLASS-PT's ``cb: Yes``
    (input.c:3952 switches the PT input from delta_m to delta_cb; spec §4.5).
    field="m" samples ``pt.delta_m`` (``cb: No``); clax.lensing uses it
    because the CMB-lensing nonlinear ratio is a total-matter quantity.

    Args:
        params: CosmoParams (h, primordial spectrum); h is traced throughout
        bg:     BackgroundResult (tau(z), f(z))
        pt:     PerturbationResult; a MatterPerturbationResult (no delta_cb)
                is accepted only with field="m"
        z:      target redshift (may be traced)
        prec:   EPT precision (sets the k-grid)
        field:  "cb" or "m"

    Returns:
        (pk_h, f): pk_h shape (prec.nmax,) in (Mpc/h)^3 on ept_kgrid(prec);
        f = bg.f_of_loga at z (0-d).
    """
    from clax.primordial import primordial_scalar_pk
    from clax.interpolation import CubicSpline as CS

    if field == "cb":
        delta = getattr(pt, "delta_cb", None)
        if delta is None:
            raise ValueError(
                "field='cb' needs PerturbationResult.delta_cb (perturbations_solve); "
                f"{type(pt).__name__} has none -- use field='m' or the full solver")
    elif field == "m":
        delta = pt.delta_m
    else:
        raise ValueError(f"field must be 'cb' or 'm', got {field!r}")

    h = params.h  # traced: carries d(pk_h)/dh AND the k-resampling channel

    # EPT k-grid in h/Mpc (static shape source) -> Mpc^-1, TRACED in h so
    # the delta/primordial sampling points move with h under AD exactly
    # as they do under finite differences (issue #30 item 4).
    k_h = ept_kgrid(prec)              # static numpy array
    k_mpc = jnp.asarray(k_h) * h       # traced jnp array

    lnk_pt  = jnp.log(pt.k_grid)
    lnk_out = jnp.log(jnp.array(k_mpc))

    # Interpolate delta to tau(z) along the tau axis (vmap-safe; no Python
    # branch on z), then spline along log-k onto the EPT k-grid. Beyond
    # pt.k_grid[-1] the spline clamps to the last value (constant delta):
    # solve with pt_k_max_cl >= 3 Mpc^-1 (the P22/P13 UV cutoff CUTOFF = 3
    # h/Mpc) when the loop integrals matter, cf. tests/test_ept_e2e_multicosmo.py.
    loga_z = jnp.log(1.0 / (1.0 + z))
    tau_z = bg.tau_of_loga.evaluate(loga_z)
    delta_at_z = jax.vmap(
        lambda d_k: CS(pt.tau_grid, d_k).evaluate(tau_z))(delta)
    delta_ept = CS(lnk_pt, delta_at_z).evaluate(lnk_out)

    # Linear P(k) in Mpc^3: P(k) = 2 pi^2 / k^3 * P_R(k) * delta^2(k)
    # (matches clax/transfer.py::compute_linear_matter_pk_from_perturbations)
    k_arr = jnp.array(k_mpc)
    prim = primordial_scalar_pk(k_arr, params)  # dimensionless P_R(k)
    pk_mpc3 = 2.0 * jnp.pi**2 / k_arr ** 3 * prim * delta_ept ** 2

    # Convert to h-units: P_h = P * h^3, k_h = k / h. This h**3 cancels
    # exactly against 1/k_arr**3 (k_arr = k_h * h) and contributes nothing
    # to d(pk_h)/dh; the true h-derivative is the traced k-resampling above.
    pk_h = pk_mpc3 * h ** 3  # (Mpc/h)^3

    # Growth rate from the background solve (f = dlnD/dlna spline),
    # z-consistent and differentiable. cf. background.py:681 (f_of_loga).
    f = bg.f_of_loga.evaluate(loga_z)
    return pk_h, f


def compute_ept_from_clax(
    params,           # CosmoParams
    bg,               # BackgroundResult
    pt,               # PerturbationResult
    z: float = 0.0,
    prec: EPTPrecisionParams = EPTPrecisionParams(),
    *,
    omfid: Optional[float] = None,
    field: str = "cb",
) -> EPTComponents:
    """Compute EPT components from a full clax perturbation run.

    Converts clax's linear P(k) of ``field`` (in Mpc^3) to h-units on the
    EPT k-grid (ept_inputs_from_clax) and runs compute_ept(), with the
    Alcock-Paczynski ratios of CLASS-PT (nonlinear_pt.c:1245-1296) when a
    fiducial Omega_m is given.

    Args:
        params: CosmoParams (for h, primordial spectrum)
        bg:     BackgroundResult (for growth factor, distances)
        pt:     PerturbationResult (for delta_cb / delta_m (k, tau))
        z:      target redshift; a static Python float when omfid is given
        prec:   EPT precision
        omfid:  fiducial Omega_m for the AP remap (CLASS-PT ``Omfid``);
                None -> hratio = Dratio = 1 (no AP), the pre-C0 behaviour
        field:  "cb" (default, CLASS-PT ``cb: Yes``) or "m"

    Returns:
        EPTComponents in h-units

    Note (stop_gradient rationale): `h` is traced throughout, including the
    k_mpc = k_h * h resampling of delta/the primordial spectrum onto the
    EPT k-grid, so d(pk_h)/dh carries the k-resampling channel exactly as
    finite differences do (issue #30 item 4; previously frozen at -9.48e4
    of the stage h-gradient, GPU job 13313). The IR resummation itself now
    runs through the traced `_ir_resummation_jax` splitter below, so
    d(pk_nw)/d(params) and d(Sigma^2)/d(params) both flow -- this closes
    the two channels that used to be stop_gradient'd here: the 1.39%-class
    pk_nw/ln10A_s residual (job 13132) and the rs_h (sound-horizon) channel
    (previously bounded below -1.0e2 of the stage h-gradient, GPU job
    13313). Two channels remain deliberately frozen:
      - the DST grid endpoints inside `_ir_resummation_jax` itself, built
        from a concrete h_conc = stop_gradient(h) (np.linspace(7e-5/h,
        7/h, ...) needs a fixed shape, not a differentiability choice --
        see that function's docstring);
      - the RSD FFTLog basis (`_pk_nw_np_rsd`/`_pk_w_np_rsd` at
        compute_ept's `_ir_precomputed` RSD consumption): PHASE-2 FREEZE,
        deferred by this PR -- see the inline comment there and
        tests/test_ept_gradients.py docstrings for the residual this
        leaves. pk_mm_real is unaffected -- it never reads those FFTLog
        bases.
    """
    from clax.background import sound_horizon_drag

    pk_h, f = ept_inputs_from_clax(params, bg, pt, z, prec, field=field)

    h = params.h
    # Concrete copy EXISTS ONLY for the DST grid endpoints inside
    # _ir_resummation_jax below (7e-5/h .. 7/h feeding np.linspace, which
    # must stay concrete -- see that function's docstring). Do NOT reuse
    # h_conc anywhere else: reusing it for k_mpc was the frozen resampling
    # channel measured at -9.48e4 of the stage h-gradient (GPU job 13313,
    # issue #30 item 4).
    h_conc = float(jax.lax.stop_gradient(h))
    k_h = ept_kgrid(prec)

    # AP ratios (spec §4.4): Python branch on the *static* omfid, never on a
    # traced value. ap_ratios is differentiable in bg (B2/B6).
    if omfid is None:
        hratio, Dratio = 1.0, 1.0
    else:
        from clax.ap import ap_ratios
        hratio, Dratio = ap_ratios(bg, z, omfid)

    # Traced IR resummation: d(pk_nw)/d(params) and d(Sigma^2)/d(params)
    # now flow (closes the 1.39% ln10A_s / 1.19% h structural residual,
    # issue #30). Only the DST grid endpoints stay pinned (see
    # _ir_resummation_jax) and the RSD FFTLog basis (phase 2, below).
    rs_h_traced = sound_horizon_drag(params) * params.h
    ir_pre = _ir_resummation_jax(pk_h, k_h, rs_h_traced, h_conc)
    return compute_ept(pk_h, jnp.array(k_h), h=h, f=f, prec=prec,
                        # rs_h is a dead argument on the _ir_precomputed path
                        # (audit 2026-08-31: only reaches the untaken numpy
                        # branch); stop_gradient is a no-op here but keeps
                        # this call site's own gradient graph minimal.
                        rs_h=jax.lax.stop_gradient(rs_h_traced),
                        _ir_precomputed=ir_pre,
                        hratio=hratio, Dratio=Dratio)
```

`Optional` is already imported at `clax/ept.py:34`. Check the file still imports: `PYTHONPATH=/home/n2minh/clax-ptval JAX_PLATFORMS=cpu /home/n2minh/micromamba/envs/clax/bin/python -c "import clax.ept as e; print(e.ept_inputs_from_clax.__name__)"`.

- [ ] **Step 4: Point `clax/lensing.py` at the matter field, fix the spec's sign**

`clax/lensing.py:293`: `ept = compute_ept_from_clax(params, bg, pt, z=0.0)` → `ept = compute_ept_from_clax(params, bg, pt, z=0.0, field="m")` and, on the line above it, the comment `# total-matter field: the ratio below divides by the matter P_lin (C0)`.

Spec lines 370-371 (`docs/superpowers/specs/2026-09-03-clax-pt-validation-design.md`): replace `is below 1 at\nk = 0.3 h/Mpc` with `is above 1 at\nk = 0.3 h/Mpc (δ_ν < δ_cb above the free-streaming scale, so δ_m < δ_cb)`.

- [ ] **Step 5: Run the cheap tests, then the slow ones on a V100**

```bash
cd /home/n2minh/clax-ptval && JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2" PYTHONPATH=/home/n2minh/clax-ptval timeout 120 /home/n2minh/micromamba/envs/clax/bin/python -m pytest tests/test_ept_from_clax.py --fast -x -q -p no:cacheprovider 2>&1 | tail -n 5
```
Expected: `6 passed, 2 skipped` (slow deselected; `lcdm_cosmology` pruned to fiducial) in < 60 s. Then the same command without `--fast` and with `-m "not slow"`: `14 passed` (5 cosmologies × 2 sweeps + 2 × `z` + 2 singles), still < 2 min — if it exceeds 2 min, stop and move that run to the sbatch below instead of waiting.

Create `slurm/ptval-track-c.sbatch` from Part 0's V100 template (job-name `ptval-track-c`, time `02:00:00`), body:

```bash
ARGS=${PTVAL_PYTEST_ARGS:-"tests/test_ept_from_clax.py -q"}
echo "pytest args: $ARGS"
python -m pytest $ARGS -p no:cacheprovider -rs 2>&1 | tail -n 40
echo "PASS"
```

(`set -euo pipefail` + `pipefail` makes a pytest failure abort before `PASS`.) Submit:

```bash
cd /home/n2minh/clax-ptval && sbatch --export=ALL,PTVAL_PYTEST_ARGS="tests/test_ept_from_clax.py -q" slurm/ptval-track-c.sbatch
```
Expected log tail: `16 passed` then `PASS` (4 slow items: 4 `nulcdm_cosmology` points — `lcdm_fiducial` and `massive_nu_006` share the 0.06 eV solve — plus the monotonicity test; 3 perturbation solves in total). Paste the printed `delta_cb/delta_m - 1` lines into the commit body. If `test_delta_cb_over_delta_m_physics` fails on the lower bound `1e-4` at 0.06 eV only, the number itself is the finding — report it with the measured value; the threshold stays.

- [ ] **Step 6: Gates and commit**

Cluster gate: `sbatch slurm/ptval-fast-suite.sbatch` → `PASS` (this is where `tests/test_ept_gradients.py` and `tests/test_lensing*.py` prove the default change and the lensing `field="m"` switch moved no numbers: both run at `ncdm_q_size = 0`). Commit (`commit-C0.txt`):

```
feat(ept): compute_ept_from_clax(omfid, field) + ept_inputs_from_clax

ept_inputs_from_clax(params, bg, pt, z, prec, *, field) returns (pk_h, f)
for field "cb" (PerturbationResult.delta_cb, CLASS-PT cb: Yes) or "m";
compute_ept_from_clax gains omfid (None -> ratios (1, 1); else
clax.ap.ap_ratios(bg, z, omfid)) and field="cb" by default. clax.lensing
passes field="m" (total-matter ratio). Spec §7 Phase 3 sign corrected:
delta_cb/delta_m > 1 above the free-streaming scale.
Measured (V100 job <id>): delta_cb/delta_m - 1 at 0.3 h/Mpc = <v006>,
<v015>, <v030> for 0.06/0.15/0.30 eV; ptval-fast-suite <id> PASS.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YBYncmhA8JrJQVNw4pVzmb
```

```bash
cd /home/n2minh/clax-ptval && git add clax/ept.py clax/lensing.py tests/test_ept_from_clax.py slurm/ptval-track-c.sbatch docs/superpowers/specs/2026-09-03-clax-pt-validation-design.md && git commit -F /tmp/claude-17163/-home-n2minh-clax/85880c7b-9aa1-44ed-b5d2-49320fe59b2a/scratchpad/commit-C0.txt
```

---

### Task C1: campaign utilities + stage-layer test (`clax.ept` on CLASS-PT inputs)

**Files:**
- Create: `tests/ept_campaign_utils.py`
- Modify: `tests/test_ept_assembly.py` (B7: helpers move out, tests stay)
- Create: `tests/test_ept_multicosmo.py`

**Interfaces:**
- Consumes: `compute_ept(pk_lin_h, k_h, h, f, prec, rs_h=, hratio=, Dratio=)` (B5/B6), A1's `validation_cosmologies`, A3/A5 reference files, B7's `BIAS_KEYS`, `_pm_from_leaves`, `_clax_nine`, `_rel`.
- Produces (`tests/ept_campaign_utils.py`, imported by C1–C3 as `cu`):
  - `SPECTRA: tuple[str, ...]` — the nine spectrum names in report order.
  - `K_MAX_COMPARE = 0.3`, `NSIDE = 10`, `window(k_h: np.ndarray) -> np.ndarray[bool]` — comparison mask (spec §4.3: k_h[10] ≤ k ≤ 0.3 h/Mpc).
  - `THRESHOLDS: dict[str, float]`, `SEAM_THRESHOLDS: dict[str, float]` — the only place a threshold lives.
  - `rel(a, b) -> float`, `compare_spectra(got: dict, ref: dict, k_h) -> dict[str, dict]` (`{name: {"err": float, "k": float}}` with `err = max|got-ref|/max|ref|` over the window, `k` = where it occurs), `compare_rows(pm_got, pm_ref, k_h) -> list[dict]` (per-row, same metric, for the 48 `pk_mult` rows), `failures(errs, thresholds) -> list[str]`.
  - `load_reference(case, z, **kw) -> dict | None` (npz → dict of arrays/scalars; `**kw` forwarded to `vc.reference_path`), `require_reference(case, z, **kw) -> dict` (pytest.skip when missing).
  - `log_record(*, layer, case, z, preset, errors, seams=None, extra=None) -> None` — appends one JSON line to `ERROR_LOG`.
  - `pm_from_leaves(e, h)`, `clax_nine(e, bias)`, `BIAS_KEYS` — B7's, unchanged bodies.
- Produces (`tests/test_ept_multicosmo.py`): `run_stage(ref: dict, bias: dict) -> tuple[EPTComponents, dict]` used by C2's `--fast` smoke comparisons.

**Design notes for the implementer.**
- `rs_h = rs_d * h`: `compute_ept` reads `rs_h` only on its numpy IR path, which is the path the stage layer takes (`_ir_precomputed=None`). The stage layer must feed the cosmology's own sound horizon, not the fiducial default 99.0 — a 10% h shift moves r_s·h by ~5%, and the wiggle/no-wiggle split (ref §5) is sensitive to it at the percent level in the ℓ=2,4 BAO region.
- Row diagnostics (`pk_mult[:48]` vs `pm_from_leaves`) are **logged, not asserted**, except that the nine assembled spectra are asserted. Row 28 (`Pk_4_vd1`) is the known 17% outlier (Part 0 finding, hypothesised AP leakage); the log keeps it visible without blocking the campaign. If the assembled `pk_gg_l4` fails while rows 30–41 pass, the bisection starts at row 28.
- The stage test at `--fast` runs 3 (case) × 1 (z) comparisons ≈ 3 × 20 s on 2 CPU threads — inside the login-node budget. Full mode (14 × 3 + 4 diagnostics = 46 EPT calls, ~15 min CPU) runs only on the cluster (`ptval-track-c.sbatch`).

- [ ] **Step 1: Write the utilities module**

```python
# tests/ept_campaign_utils.py
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
    so that zero crossings of l=2/l=4 do not blow up a pointwise ratio."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


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


def log_record(*, layer: str, case: str, z: float, preset: str, errors: dict,
               seams: dict | None = None, extra: dict | None = None) -> None:
    """Append one JSON line to ERROR_LOG. Keys: ts (ISO-8601 UTC), git_sha,
    layer ('stage' | 'e2e' | 'grad'), case, z, preset, errors
    ({spectrum: {"err", "k"}}), seams ({name: residual}), extra (free)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rec = {"ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
           "git_sha": _git_sha(), "layer": layer, "case": case, "z": float(z),
           "preset": preset, "errors": errors, "seams": seams or {}, "extra": extra or {}}
    with ERROR_LOG.open("a") as fh:
        fh.write(json.dumps(rec, default=float) + "\n")


# ---------------------------------------------------------------------------
# B7 helpers (moved verbatim from tests/test_ept_assembly.py; that file now
# imports them). The 48-row layout mirrors CLASS-PT's pk_mult (ref §9).
# ---------------------------------------------------------------------------

BIAS_KEYS = ("b1", "b2", "bG2", "bGamma3", "cs0", "cs2", "cs4", "cs", "Pshot", "b4")


def pm_from_leaves(e, h):
    # <-- paste B7's _pm_from_leaves body here unchanged (rows 0..47, /h^3 and /h units)
    ...


def clax_nine(e, bias):
    # <-- paste B7's _clax_nine body here unchanged (the nine pk_* calls with bias kwargs)
    ...
```

The two `...` bodies are B7's functions moved **verbatim** — cut them from `tests/test_ept_assembly.py` (they are the only definitions of `_pm_from_leaves`/`_clax_nine`/`_rel`/`BIAS_KEYS` there), drop the leading underscore, paste. Then at the top of `tests/test_ept_assembly.py` add

```python
from tests.ept_campaign_utils import (
    BIAS_KEYS, pm_from_leaves as _pm_from_leaves, clax_nine as _clax_nine, rel as _rel,
)
```

so every B7 test body compiles unchanged. Guard: B7's tests are the regression test for the move.

```bash
cd /home/n2minh/clax-ptval && JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2" PYTHONPATH=/home/n2minh/clax-ptval timeout 120 /home/n2minh/micromamba/envs/clax/bin/python -m pytest tests/test_ept_assembly.py --fast -x -q -p no:cacheprovider 2>&1 | tail -n 3
```
Expected: the same pass/skip count B7 recorded in its commit message (no new failures; `needs_twin` skips unchanged).

- [ ] **Step 2: Unit-test the utilities (cheap, no JAX)**

Append to `tests/test_ept_multicosmo.py` (created in this step; the stage tests follow in Step 3):

```python
# tests/test_ept_multicosmo.py
"""Stage layer of the clax-pt vs CLASS-PT campaign (spec §4.2 layer 1):
clax.ept.compute_ept on CLASS-PT's OWN linear P_cb(k), f, r_s, hratio, Dratio,
so that only the EPT stage is under test. 14 distinct cosmologies x
z in (0, 0.38, 0.8) in full mode; FAST_CASES x FAST_Z under --fast
(spec §4.6). Multi-cosmology rule: satisfied by construction.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import jax.numpy as jnp

from clax.ept import EPTPrecisionParams, compute_ept
from scripts import validation_cosmologies as vc
from tests import ept_campaign_utils as cu


# ---------------------------------------------------------------------------
# utilities (cosmology-independent numerics -- exempt from the grid rule)
# ---------------------------------------------------------------------------

def test_window_and_thresholds():
    k = np.logspace(np.log10(5e-5), 2, 256)
    w = cu.window(k)
    assert not w[:cu.NSIDE].any() and w[cu.NSIDE] and k[w].max() <= 0.3 and k[~w][cu.NSIDE:].min() > 0.3
    assert set(cu.THRESHOLDS) == set(cu.SPECTRA)
    assert cu.THRESHOLDS["pk_gg_l4"] == 0.02 and cu.THRESHOLDS["pk_gg_l0"] == 0.01
    assert cu.SEAM_THRESHOLDS["pk_lin"] == 1e-3


def test_rel_and_failures():
    k = np.logspace(np.log10(5e-5), 2, 256)
    ref = {"pk_gg_l0": np.ones(256), "pk_gg_l4": np.sin(k)}     # l4 crosses zero: max-relative, not pointwise
    got = {"pk_gg_l0": np.ones(256) * 1.005, "pk_gg_l4": np.sin(k) * 1.03}
    errs = cu.compare_spectra(got, ref, k)
    assert abs(errs["pk_gg_l0"]["err"] - 0.005) < 1e-12
    assert abs(errs["pk_gg_l4"]["err"] - 0.03) < 1e-9
    lines = cu.failures(errs, cu.THRESHOLDS)
    assert lines == [f"pk_gg_l4 3.00% > 2.00% at k={errs['pk_gg_l4']['k']:.3f}"], lines


def test_log_record_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(cu, "LOG_DIR", tmp_path)
    monkeypatch.setattr(cu, "ERROR_LOG", tmp_path / "errors.jsonl")
    cu.log_record(layer="stage", case="lcdm_fiducial", z=0.38, preset="stage",
                  errors={"pk_gg_l0": {"err": 0.003, "k": 0.21}}, seams={"f": 1e-5})
    rec = json.loads((tmp_path / "errors.jsonl").read_text().splitlines()[-1])
    assert rec["layer"] == "stage" and rec["z"] == 0.38 and rec["errors"]["pk_gg_l0"]["err"] == 0.003
    assert rec["ts"].endswith("+00:00") and rec["seams"] == {"f": 1e-5}
```

Run: same login-node command on `tests/test_ept_multicosmo.py --fast -x -q` → `3 passed` (Step 3's tests are not written yet).

- [ ] **Step 3: Write the stage-layer tests**

Append to `tests/test_ept_multicosmo.py`:

```python
# ---------------------------------------------------------------------------
# stage layer
# ---------------------------------------------------------------------------

def pytest_generate_tests(metafunc):
    if "case_z" in metafunc.fixturenames:
        if metafunc.config.getoption("--fast", default=False):
            grid = [(c, vc.FAST_Z) for c in vc.FAST_CASES]
        else:
            grid = [(c, z) for c in vc.distinct_cases() for z in vc.Z_LIST]
        metafunc.parametrize("case_z", grid, ids=[f"{c}-z{z:.2f}" for c, z in grid])


def run_stage(ref: dict, bias: dict):
    """compute_ept on the reference file's own inputs -> (EPTComponents, nine spectra)."""
    e = compute_ept(jnp.asarray(ref["pk_lin"]), jnp.asarray(ref["k_h"]),
                    h=float(ref["h"]), f=float(ref["fz"]), prec=EPTPrecisionParams(),
                    rs_h=float(ref["rs_d"]) * float(ref["h"]),      # r_s(z_d) h in Mpc/h, THIS cosmology
                    hratio=float(ref["hratio"]), Dratio=float(ref["Dratio"]))
    nine = {name: np.asarray(arr) for name, arr in cu.clax_nine(e, bias).items()}
    return e, nine


def _assert_flags(ref, *, ap: bool, cb: bool):
    assert bool(ref["ap"]) is ap and bool(ref["cb"]) is cb, (ref["ap"], ref["cb"])
    if ap:
        assert float(ref["omfid"]) == vc.OMFID
    assert str(ref["kh_convention"]).startswith("h/Mpc"), ref["kh_convention"]


def _check(ref, *, case, z, tag, bias=None, extra=None):
    bias = bias if bias is not None else json.loads(str(ref["bias_json"]))
    e, nine = run_stage(ref, bias)
    k_h = np.asarray(ref["k_h"])
    errs = cu.compare_spectra(nine, ref, k_h)
    rows = cu.compare_rows(cu.pm_from_leaves(e, float(ref["h"])), np.asarray(ref["pk_mult"])[:48], k_h)
    cu.log_record(layer="stage", case=case, z=z, preset=tag, errors=errs,
                  extra={"rows": rows, **(extra or {})})
    bad = cu.failures(errs, cu.THRESHOLDS)
    worst = max(errs.items(), key=lambda kv: kv[1]["err"])
    print(f"{case} z={z:.2f} [{tag}] worst {worst[0]} {100 * worst[1]['err']:.3f}% at k={worst[1]['k']:.3f}")
    assert not bad, f"{case} z={z:.2f} [{tag}]: " + "; ".join(bad)
    return nine, errs


def test_stage_nine_spectra(case_z):
    case, z = case_z
    ref = cu.require_reference(case, z)
    _assert_flags(ref, ap=True, cb=True)
    _check(ref, case=case, z=z, tag="stage")


# --- diagnostics at lcdm_fiducial, z = 0.38 (skip individually when the file is absent) ---

DIAG_CASE, DIAG_Z = "lcdm_fiducial", 0.38


def test_stage_bias_nonzero():
    """Every bias/counterterm/stochastic row is live (spec §4.8): the same
    thresholds must hold with BIAS_NONZERO, otherwise a wrong row was hiding
    behind b2 = bG2 = ... = 0."""
    ref = cu.require_reference(DIAG_CASE, DIAG_Z, bias="nonzero")
    assert json.loads(str(ref["bias_json"])) == vc.BIAS_NONZERO
    _check(ref, case=DIAG_CASE, z=DIAG_Z, tag="stage-biasnz")


def test_stage_cb_vs_m():
    """cb: No file must pass on its own; record the cb-minus-m delta of the
    nine spectra (the size of the cb convention at 0.06 eV, spec §4.5)."""
    ref_m = cu.require_reference(DIAG_CASE, DIAG_Z, cb=False)
    ref_cb = cu.require_reference(DIAG_CASE, DIAG_Z)
    _assert_flags(ref_m, ap=True, cb=False)
    nine_m, _ = _check(ref_m, case=DIAG_CASE, z=DIAG_Z, tag="stage-m")
    nine_cb, _ = _check(ref_cb, case=DIAG_CASE, z=DIAG_Z, tag="stage-cb")
    delta = {n: cu.rel(nine_cb[n], nine_m[n]) for n in cu.SPECTRA}
    cu.log_record(layer="stage", case=DIAG_CASE, z=DIAG_Z, preset="stage-cb-minus-m",
                  errors={n: {"err": v, "k": float("nan")} for n, v in delta.items()})
    assert delta["pk_gg_l0"] > 1e-4, delta      # the convention is not a no-op at 0.06 eV


def test_stage_ap_off():
    """noap file: ratios must be (1, 1) and the spectra must pass without AP;
    record the AP-on minus AP-off delta (the size of the effect under test)."""
    ref_noap = cu.require_reference(DIAG_CASE, DIAG_Z, ap=False)
    ref_ap = cu.require_reference(DIAG_CASE, DIAG_Z)
    _assert_flags(ref_noap, ap=False, cb=True)
    assert (float(ref_noap["hratio"]), float(ref_noap["Dratio"])) == (1.0, 1.0)
    nine_noap, _ = _check(ref_noap, case=DIAG_CASE, z=DIAG_Z, tag="stage-noap")
    nine_ap, _ = _check(ref_ap, case=DIAG_CASE, z=DIAG_Z, tag="stage-ap")
    delta = {n: cu.rel(nine_ap[n], nine_noap[n]) for n in cu.SPECTRA}
    cu.log_record(layer="stage", case=DIAG_CASE, z=DIAG_Z, preset="stage-ap-minus-noap",
                  errors={n: {"err": v, "k": float("nan")} for n, v in delta.items()})
    assert delta["pk_gg_l2"] > 1e-3, delta      # Omega_m(fiducial) = 0.3153 vs Omfid 0.31 is a real remap


def test_stage_w0wa_ppf_seam():
    """w0wa: the canonical (use_ppf=yes) file is asserted; the noppf twin is
    compared and its delta recorded (spec §9 ppf seam)."""
    case = "w0wa_m07_m10"
    ref = cu.require_reference(case, DIAG_Z)
    assert bool(ref["use_ppf"]) is True
    ref_noppf = cu.require_reference(case, DIAG_Z, tag="noppf")
    assert bool(ref_noppf["use_ppf"]) is False
    nine_ppf, _ = _check(ref, case=case, z=DIAG_Z, tag="stage-ppf")
    e, nine_noppf = run_stage(ref_noppf, json.loads(str(ref_noppf["bias_json"])))
    delta = {n: cu.rel(nine_ppf[n], nine_noppf[n]) for n in cu.SPECTRA}
    lin_delta = cu.rel(np.asarray(ref["pk_lin"])[cu.window(ref["k_h"])],
                       np.asarray(ref_noppf["pk_lin"])[cu.window(ref["k_h"])])
    cu.log_record(layer="stage", case=case, z=DIAG_Z, preset="stage-ppf-minus-noppf",
                  errors={n: {"err": v, "k": float("nan")} for n, v in delta.items()},
                  extra={"pk_lin_delta": lin_delta})
    print(f"w0wa ppf-vs-noppf: pk_lin {100 * lin_delta:.3f}%, pk_gg_l0 {100 * delta['pk_gg_l0']:.3f}%")
```

- [ ] **Step 4: Run on the login node (`--fast`), then full on the cluster**

```bash
cd /home/n2minh/clax-ptval && JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2" PYTHONPATH=/home/n2minh/clax-ptval timeout 120 /home/n2minh/micromamba/envs/clax/bin/python -m pytest tests/test_ept_multicosmo.py --fast -q -p no:cacheprovider -rs -s 2>&1 | grep -E "worst|passed|failed|skipped|ERROR" | head -n 20
```
Expected: three `worst ...` lines (one per FAST_CASE at z=0.38), then `10 passed` when the four diagnostic files exist, else `6 passed, 4 skipped` with `reference missing: ...` reasons. Any `failed` line: read the greppable `pk_xx N% > M% at k=` message, look at the `rows` list for that record in `test_logs/ptval/errors.jsonl` (`python -c "import json; r=[json.loads(l) for l in open('test_logs/ptval/errors.jsonl')][-1]; print(sorted(r['extra']['rows'], key=lambda d: -d['err'])[:5])"`), and report which row is off — that is the finding. The `stage` layer has no clax-side precision lever: a stage failure is an EPT-kernel or assembly bug (or a wrong reference-file convention); it goes to the campaign log and to the user, and the threshold stays.

Full sweep: `sbatch --export=ALL,PTVAL_PYTEST_ARGS="tests/test_ept_multicosmo.py -q -rs -s" slurm/ptval-track-c.sbatch` → expected `49 passed` (42 grid points + 3 utility tests + 4 diagnostics), `PASS`. Put the three worst-case lines per family in the commit body.

- [ ] **Step 5: Gates and commit**

Local gate: `tests/test_ept_assembly.py tests/test_ept_multicosmo.py --fast -q`. Cluster gate: `sbatch slurm/ptval-fast-suite.sbatch` → `PASS`. Commit (`commit-C1.txt`):

```
test(ept): stage-layer campaign vs CLASS-PT + shared campaign utils

tests/ept_campaign_utils.py: SPECTRA, window (k_h[10] <= k <= 0.3),
THRESHOLDS (1% / 2% for l=4), SEAM_THRESHOLDS, max|dP|/max|P| metric,
JSONL error log; B7's pm_from_leaves/clax_nine move here.
tests/test_ept_multicosmo.py: compute_ept on each reference file's own
P_cb, f, r_s h, hratio, Dratio; 14 cases x 3 z + biasnz / cb-vs-m /
AP-off / w0wa ppf diagnostics at lcdm_fiducial z=0.38.
Worst (V100 job <id>): LCDM <spec> <x>% (<case>, z=<z>); nuLCDM ...;
w0wa ...; row 28 (Pk_4_vd1) logged at <y>% as known.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YBYncmhA8JrJQVNw4pVzmb
```

```bash
cd /home/n2minh/clax-ptval && git add tests/ept_campaign_utils.py tests/test_ept_assembly.py tests/test_ept_multicosmo.py && git commit -F /tmp/claude-17163/-home-n2minh-clax/85880c7b-9aa1-44ed-b5d2-49320fe59b2a/scratchpad/commit-C1.txt
```

---

### Task C2: end-to-end layer — seams, nine spectra, gradient sweep (slow, V100)

**Files:**
- Create: `tests/test_ept_e2e_multicosmo.py`

**Interfaces:**
- Consumes: `compute_ept_from_clax(params, bg, pt, z, prec, *, omfid, field)` and `ept_inputs_from_clax` (C0), `ap_ratios` (B2), `sound_horizon_drag`, `background_solve`/`thermodynamics_solve`/`perturbations_solve`, `PK_FAST_PREC`/`PK_CONTRACT_PREC` (`tests/pk_test_utils.py`), `cu.*` (C1), `vc.*` (A1).
- Produces: nothing imported elsewhere. Environment knobs read by C3's job: `PTVAL_E2E_PREC` (`fast` default | `contract`), `PTVAL_E2E_SUBSET` (`fast` → FAST_CASES × FAST_Z, unset → full grid).

**Design notes for the implementer.**
- *Why `pt_k_max_cl = 5.0` Mpc⁻¹:* `ept_inputs_from_clax` clamps δ beyond `pt.k_grid[-1]` (constant extrapolation), which makes P_lin fall as k^{n_s−4} instead of the true k^{n_s−8}ln²k. `compute_ept`'s P22/P13 FFTLog coefficients see `pk_lin` on the whole EPT grid (up to 100 h/Mpc; the `exp(−(k/3)⁶)` damping is applied to the *output* k only), so the tail must be physical up to well past 3 h/Mpc. 5.0 Mpc⁻¹ = 7.4 h/Mpc at h = 0.6736 (lowest h on the grid is 0.6736 too; `h_high` gives 6.7 h/Mpc). This mirrors the existing e2e gradient test (`tests/test_ept_gradients.py`, `pt_k_max_cl=5.0`).
- *Presets:* `fast` = `PK_FAST_PREC` (40 k/decade, l_max 35, rtol 1e-5, ncdm_q_size 5, fluid approximation off) with `pt_k_max_cl=5.0` → 5.7 decades × 40 ≈ 228 k-modes; `contract` = `PK_CONTRACT_PREC` (60/decade, l_max 50, rtol 1e-6) ≈ 342 modes. `pt_k_chunk_size=0` (auto-batched) for GPU throughput. Time one solve per preset in the smoke job and write it into the commit body — the campaign job's wall-time (C3) is sized from it.
- *Seams before spectra:* the spec §8 bisection order is background (hratio/Dratio/r_s/H) → f → P_cb,lin → EPT. Each seam is its own test so a red spectrum comes with its seam residuals already printed. The P_cb,lin seam is the one expected to bind: clax's P(k) sits at ~1% vs CLASS today while the spec asks 0.1%. When it fails, log it, keep the threshold, and run the `contract` preset — that is the legitimate lever. A seam failure with passing spectra is still a failure (the spectra passed by cancellation).
- *cb by default:* the reference `pk_lin` is P_cb; the e2e comparison uses `field="cb"`. `field="m"` is compared to `pk_m_lin` as a second seam (both fields must agree with CLASS).
- *Gradient sweep:* B6's exemption statement names "Part 2 C2's `test_e2e_gradient_finite`" as the pipeline-level gradient sweep over the real cosmology grid, so it is parametrized over `vc.FAST_CASES` (LCDM, νΛCDM, w0wa) — three distinct families, satisfying the multi-cosmology rule. It uses `PrecisionParams.fast_cl()`-based `GRAD_PREC` (cheaper than the value presets; the gradient only needs to be finite and nonzero, not converged).

- [ ] **Step 1: Write the test module**

```python
# tests/test_ept_e2e_multicosmo.py
"""End-to-end layer of the clax-pt vs CLASS-PT campaign (spec §4.2 layer 2):
clax background -> thermodynamics -> perturbations -> compute_ept_from_clax
(omfid=0.31, field="cb") against CLASS-PT's nine spectra, with the spec §8
seams (background, f, P_cb,lin) asserted first so a failing spectrum is
already bisected. One perturbation solve per case (z by tau-interpolation).

All tests are slow (GPU); run via slurm/ptval-track-c.sbatch or the
campaign job slurm/ept-multicosmo-e2e.sbatch. Environment:
  PTVAL_E2E_PREC   = fast (default) | contract   -- clax precision preset
  PTVAL_E2E_SUBSET = fast                        -- FAST_CASES x FAST_Z only
Multi-cosmology rule: 14 cases x 3 z (full) / 3 families x 1 z (subset).
"""
from __future__ import annotations

import os
import time
from dataclasses import replace as _dc_replace

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from clax import PrecisionParams
from clax.background import background_solve, sound_horizon_drag
from clax.thermodynamics import thermodynamics_solve
from clax.perturbations import perturbations_solve
from clax.ept import compute_ept_from_clax, ept_inputs_from_clax, ept_kgrid
from clax.ap import ap_ratios
from scripts import validation_cosmologies as vc
from tests import ept_campaign_utils as cu
from tests.pk_test_utils import PK_FAST_PREC, PK_CONTRACT_PREC

pytestmark = pytest.mark.slow

# pt_k_max_cl = 5 Mpc^-1 (>= 6.7 h/Mpc on the grid): the P22/P13 FFTLog
# coefficients read pk_lin far beyond the 3 h/Mpc output cutoff, and
# ept_inputs_from_clax clamps delta beyond pt.k_grid[-1] -- see C2 notes.
E2E_PRESETS = {
    "fast": _dc_replace(PK_FAST_PREC, pt_k_max_cl=5.0, pt_k_chunk_size=0),
    "contract": _dc_replace(PK_CONTRACT_PREC, pt_k_max_cl=5.0, pt_k_chunk_size=0),
}
PRESET_NAME = os.environ.get("PTVAL_E2E_PREC", "fast")
PREC = E2E_PRESETS[PRESET_NAME]
GRAD_PREC = _dc_replace(PrecisionParams.fast_cl(), pt_k_max_cl=5.0, pt_k_chunk_size=20,
                        ncdm_q_size=5)

_PIPELINE: dict[str, tuple] = {}     # case -> (params, bg, pt, seconds)
_SEAMS: dict[tuple, dict] = {}       # (case, z) -> {seam: residual}


def pytest_generate_tests(metafunc):
    if "case_z" in metafunc.fixturenames:
        if os.environ.get("PTVAL_E2E_SUBSET") == "fast":
            grid = [(c, vc.FAST_Z) for c in vc.FAST_CASES]
        else:
            grid = [(c, z) for c in vc.distinct_cases() for z in vc.Z_LIST]
        metafunc.parametrize("case_z", grid, ids=[f"{c}-z{z:.2f}" for c, z in grid])


def pipeline(case: str):
    """One background + thermo + perturbation solve per case at PREC (cached)."""
    if case not in _PIPELINE:
        params = vc.clax_params(case)
        t0 = time.perf_counter()
        bg = background_solve(params, PREC)
        th = thermodynamics_solve(params, PREC, bg)
        pt = perturbations_solve(params, PREC, bg, th)
        jax.block_until_ready(pt.delta_cb)
        secs = time.perf_counter() - t0
        print(f"[pipeline] {case} preset={PRESET_NAME} n_k={pt.k_grid.shape[0]} {secs:.0f} s")
        _PIPELINE[case] = (params, bg, pt, secs)
    return _PIPELINE[case]


def _seam(case, z, name, value):
    _SEAMS.setdefault((case, z), {})[name] = float(value)
    return float(value)


# ---------------------------------------------------------------------------
# seams (spec §8 order). Each is its own test: a red spectrum below arrives
# with these residuals already in the log.
# ---------------------------------------------------------------------------

def test_seam_background(case_z):
    case, z = case_z
    ref = cu.require_reference(case, z)
    params, bg, _, _ = pipeline(case)
    hr, Dr = ap_ratios(bg, z, vc.OMFID)
    e_hr = _seam(case, z, "hratio", abs(float(hr) / float(ref["hratio"]) - 1.0))
    e_Dr = _seam(case, z, "Dratio", abs(float(Dr) / float(ref["Dratio"]) - 1.0))
    e_rs = _seam(case, z, "rs_d", abs(float(sound_horizon_drag(params)) / float(ref["rs_d"]) - 1.0))
    H = float(bg.H_of_loga.evaluate(jnp.log(1.0 / (1.0 + z))))
    e_H = _seam(case, z, "H_z", abs(H / float(ref["H_z"]) - 1.0))
    bad = cu.failures({k: {"err": v, "k": float("nan")} for k, v in
                       dict(hratio=e_hr, Dratio=e_Dr, rs_d=e_rs, H_z=e_H).items()}, cu.SEAM_THRESHOLDS)
    assert not bad, f"{case} z={z:.2f} background seam: " + "; ".join(bad)


def test_seam_growth_rate_f(case_z):
    case, z = case_z
    ref = cu.require_reference(case, z)
    params, bg, pt, _ = pipeline(case)
    _, f = ept_inputs_from_clax(params, bg, pt, z, field="cb")
    e_f = _seam(case, z, "f", abs(float(f) / float(ref["fz"]) - 1.0))
    assert e_f <= cu.SEAM_THRESHOLDS["f"], f"{case} z={z:.2f} f seam: {100 * e_f:.3f}% > {100 * cu.SEAM_THRESHOLDS['f']:.3f}%"


def _pk_seam(pk_got, pk_ref, k_h):
    w = cu.window(k_h)
    tail = (k_h > cu.K_MAX_COMPARE) & (k_h <= 3.0)
    ratio = np.asarray(pk_got) / np.asarray(pk_ref) - 1.0
    i = int(np.argmax(np.abs(ratio[w])))
    return float(np.max(np.abs(ratio[w]))), float(k_h[w][i]), float(np.max(np.abs(ratio[tail])))


def test_seam_pk_cb_lin(case_z):
    """clax P_cb,lin (field='cb') vs CLASS P_cb (pk_lin), pointwise on the
    window (0.1%) and on 0.3 < k <= 3 h/Mpc (3%); P_m,lin vs pk_m_lin likewise."""
    case, z = case_z
    ref = cu.require_reference(case, z)
    params, bg, pt, _ = pipeline(case)
    k_h = np.asarray(ref["k_h"])
    assert np.allclose(k_h, ept_kgrid()), "reference k_h is not the EPT grid"
    pk_cb, _ = ept_inputs_from_clax(params, bg, pt, z, field="cb")
    e_win, k_worst, e_tail = _pk_seam(pk_cb, ref["pk_lin"], k_h)
    _seam(case, z, "pk_lin", e_win); _seam(case, z, "pk_lin_k", k_worst); _seam(case, z, "pk_lin_tail", e_tail)
    if "pk_m_lin" in ref:
        pk_m, _ = ept_inputs_from_clax(params, bg, pt, z, field="m")
        e_m, _, _ = _pk_seam(pk_m, ref["pk_m_lin"], k_h)
        _seam(case, z, "pk_m_lin", e_m)
    print(f"{case} z={z:.2f} P_cb,lin seam {100 * e_win:.3f}% at k={k_worst:.3f}, tail {100 * e_tail:.2f}%")
    bad = cu.failures({"pk_lin": {"err": e_win, "k": k_worst}, "pk_lin_tail": {"err": e_tail, "k": float("nan")}},
                      cu.SEAM_THRESHOLDS)
    assert not bad, f"{case} z={z:.2f} P_lin seam ({PRESET_NAME}): " + "; ".join(bad)


# ---------------------------------------------------------------------------
# nine spectra
# ---------------------------------------------------------------------------

def test_e2e_spectra(case_z):
    case, z = case_z
    ref = cu.require_reference(case, z)
    params, bg, pt, secs = pipeline(case)
    import json
    bias = json.loads(str(ref["bias_json"]))
    e = compute_ept_from_clax(params, bg, pt, z=z, omfid=vc.OMFID, field="cb")
    nine = {n: np.asarray(a) for n, a in cu.clax_nine(e, bias).items()}
    k_h = np.asarray(ref["k_h"])
    errs = cu.compare_spectra(nine, ref, k_h)
    seams = _SEAMS.get((case, z), {})
    cu.log_record(layer="e2e", case=case, z=z, preset=PRESET_NAME, errors=errs, seams=seams,
                  extra={"solve_seconds": secs, "n_k": int(pt.k_grid.shape[0])})
    bad = cu.failures(errs, cu.THRESHOLDS)
    worst = max(errs.items(), key=lambda kv: kv[1]["err"])
    print(f"{case} z={z:.2f} [e2e/{PRESET_NAME}] worst {worst[0]} {100 * worst[1]['err']:.3f}% at k={worst[1]['k']:.3f}")
    assert not bad, (f"{case} z={z:.2f} [e2e/{PRESET_NAME}]: " + "; ".join(bad)
                     + " | seams: " + ", ".join(f"{k}={v:.2e}" for k, v in seams.items()))


# ---------------------------------------------------------------------------
# pipeline gradient sweep (B6's exemption points here)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", vc.FAST_CASES)
def test_e2e_gradient_finite(case):
    """d/d(omega_cdm) of sum_window pk_gg_l0 through background -> thermo ->
    perturbations -> compute_ept_from_clax(omfid, field='cb') is finite and
    nonzero on one case per family (LCDM, nuLCDM, w0wa)."""
    base = vc.clax_params(case)
    k_h = ept_kgrid()
    w = jnp.asarray(cu.window(k_h))

    def objective(omega_cdm):
        params = base.replace(omega_cdm=omega_cdm)
        bg = background_solve(params, GRAD_PREC)
        th = thermodynamics_solve(params, GRAD_PREC, bg)
        pt = perturbations_solve(params, GRAD_PREC, bg, th)
        e = compute_ept_from_clax(params, bg, pt, z=vc.FAST_Z, omfid=vc.OMFID, field="cb")
        return jnp.sum(jnp.where(w, cu.clax_nine(e, vc.BIAS)["pk_gg_l0"], 0.0))

    t0 = time.perf_counter()
    val, g = jax.value_and_grad(objective)(jnp.asarray(base.omega_cdm))
    jax.block_until_ready(g)
    g, val = float(g), float(val)
    cu.log_record(layer="grad", case=case, z=vc.FAST_Z, preset="grad",
                  errors={}, extra={"d_sum_pk_gg_l0_d_omega_cdm": g, "value": val,
                                    "seconds": time.perf_counter() - t0})
    print(f"{case}: d(sum pk_gg_l0)/d(omega_cdm) = {g:.4e} (value {val:.4e}, {time.perf_counter() - t0:.0f} s)")
    # finiteness and non-vanishing are the contract; the SIGN is not asserted
    # (the omega_cdm derivative of the window-sum is cosmology-dependent).
    assert np.isfinite(g) and g != 0.0, (case, g)
```

- [ ] **Step 2: Smoke on a V100 (subset, both presets)**

```bash
cd /home/n2minh/clax-ptval && sbatch --export=ALL,PTVAL_E2E_SUBSET=fast,PTVAL_PYTEST_ARGS="tests/test_ept_e2e_multicosmo.py -q -rs -s -m slow" slurm/ptval-track-c.sbatch
cd /home/n2minh/clax-ptval && sbatch --export=ALL,PTVAL_E2E_SUBSET=fast,PTVAL_E2E_PREC=contract,PTVAL_PYTEST_ARGS="tests/test_ept_e2e_multicosmo.py -q -rs -s -m slow -k 'seam or spectra'" slurm/ptval-track-c.sbatch
```
Expected (first job): three `[pipeline] ... s` lines, three `P_cb,lin seam ...` lines, three `worst ...` lines, three gradient lines, `15 passed` and `PASS` (3 cases × 4 tests + 3 gradients). If `test_seam_pk_cb_lin` fails at `fast` and passes at `contract`, the campaign runs at `contract` (C3 sets `PTVAL_E2E_PREC=contract`) and the fast-preset residual is recorded in the report as a finding about clax's P(k) accuracy — not silenced. If it fails at both, stop and report: the seam residual and its k are the finding; do not proceed to C3 with a red seam.

Record from the logs: solve seconds per case per preset (sizes C3's wall-time: 14 cases × solve + 42 EPT calls), and the gradient values.

- [ ] **Step 3: Gates and commit**

Local gate: `tests/test_ept_from_clax.py tests/test_ept_multicosmo.py --fast -q` (collect-only for the new module: `pytest tests/test_ept_e2e_multicosmo.py --collect-only -q | tail -n 2` → `N tests collected`). Cluster gate: `sbatch slurm/ptval-fast-suite.sbatch` → `PASS`. Commit (`commit-C2.txt`):

```
test(ept): end-to-end multi-cosmology layer with spec §8 seams

tests/test_ept_e2e_multicosmo.py (slow): per case one clax solve
(PK_FAST_PREC / PK_CONTRACT_PREC + pt_k_max_cl=5 Mpc^-1, PTVAL_E2E_PREC),
seams hratio/Dratio/rs_d/H (1e-4/1e-3), f (1e-3), P_cb,lin (0.1% window,
3% tail), then the nine spectra via compute_ept_from_clax(omfid=0.31,
field="cb"), and d(sum pk_gg_l0)/d(omega_cdm) finite on FAST_CASES.
Smoke (V100 jobs <id>, <id>): solve <s> s (fast) / <s> s (contract);
P_cb,lin seam <x>% (<case>); worst spectrum <spec> <y>%.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YBYncmhA8JrJQVNw4pVzmb
```

```bash
cd /home/n2minh/clax-ptval && git add tests/test_ept_e2e_multicosmo.py && git commit -F /tmp/claude-17163/-home-n2minh-clax/85880c7b-9aa1-44ed-b5d2-49320fe59b2a/scratchpad/commit-C2.txt
```

---

### Task C3: campaign job + report summarizer

**Files:**
- Create: `slurm/ept-multicosmo-e2e.sbatch`
- Create: `scripts/summarize_ept_validation.py`
- Create: `docs/validation/2026-09-clax-pt-multipoles.md` (generated; committed)

**Interfaces:**
- Consumes: `test_logs/ptval/errors.jsonl` (C1's `log_record` schema), `test_logs/ptval/junit-campaign.xml` (pytest `--junitxml`), `vc.FAMILIES` / `vc.CASES` (A1), `cu.SPECTRA` / `cu.THRESHOLDS` / `cu.SEAM_THRESHOLDS` (C1).
- Produces: `python scripts/summarize_ept_validation.py --errors <jsonl> --junit <xml> --out <md> [--preset contract]` → the markdown report (spec §4.9): provenance, per-family `case × z × spectrum` tables of max|ΔP|/max|P| (with k), the seam table, the four diagnostics (biasnz, cb−m, AP−noAP, ppf−noppf), the gradient table, the pass/fail roll-up from junit, and the measured worst case per spectrum (C4's ratchet input).

**Design notes.** The summarizer is pure Python (`json`, `xml.etree`, no JAX) so it runs on the login node in a second; C4 re-runs it after the ratchet. Records are keyed by `(layer, case, z, preset)`; when a key repeats (a re-run appended to the same jsonl) the **last** record wins — the job deletes the log before starting, so a single campaign never has duplicates, but a manual re-run of one test may.

- [ ] **Step 1: Write the summarizer with its unit test**

```python
# tests/test_summarize_ept_validation.py  (cosmology-independent: exempt from the grid rule)
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" tests="3" failures="1" skipped="1">
<testcase classname="tests.test_ept_multicosmo" name="test_stage_nine_spectra[lcdm_fiducial-z0.38]" time="20.1"/>
<testcase classname="tests.test_ept_e2e_multicosmo" name="test_e2e_spectra[h_high-z0.80]" time="30.2">
<failure message="h_high z=0.80 [e2e/fast]: pk_gg_l4 2.31% &gt; 2.00% at k=0.297 | seams: pk_lin=1.2e-03">x</failure></testcase>
<testcase classname="tests.test_ept_e2e_multicosmo" name="test_e2e_spectra[w0wa_m07_m10-z0.00]" time="0.1">
<skipped message="reference missing: z0.000_ap_omfid0.31_cb.npz -- run slurm/classpt-refgen.sbatch (Part 1a, A5)"/></testcase>
</testsuite></testsuites>"""


def test_summarizer_renders(tmp_path):
    errs = tmp_path / "errors.jsonl"
    recs = [
        {"ts": "2026-09-04T10:22:31+00:00", "git_sha": "abc1234", "layer": "stage", "case": "lcdm_fiducial",
         "z": 0.38, "preset": "stage", "errors": {"pk_gg_l0": {"err": 0.0031, "k": 0.212},
                                                  "pk_gg_l4": {"err": 0.012, "k": 0.290}}, "seams": {},
         "extra": {"rows": [{"row": 28, "err": 0.17, "k": 0.25}]}},
        {"ts": "2026-09-04T10:40:00+00:00", "git_sha": "abc1234", "layer": "e2e", "case": "h_high",
         "z": 0.8, "preset": "fast", "errors": {"pk_gg_l4": {"err": 0.0231, "k": 0.297}},
         "seams": {"pk_lin": 1.2e-3, "f": 2e-5, "hratio": 1e-6}, "extra": {"solve_seconds": 41.0}},
        {"ts": "2026-09-04T10:41:00+00:00", "git_sha": "abc1234", "layer": "grad", "case": "lcdm_fiducial",
         "z": 0.38, "preset": "grad", "errors": {}, "seams": {},
         "extra": {"d_sum_pk_gg_l0_d_omega_cdm": -1.23e7, "value": 4.5e6, "seconds": 300.0}},
    ]
    errs.write_text("".join(json.dumps(r) + "\n" for r in recs))
    junit = tmp_path / "junit.xml"
    junit.write_text(JUNIT)
    out = tmp_path / "report.md"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "summarize_ept_validation.py"),
                    "--errors", str(errs), "--junit", str(junit), "--out", str(out), "--job-id", "12345"],
                   check=True, cwd=ROOT)
    text = out.read_text()
    for needle in ("abc1234", "12345", "## LCDM", "lcdm_fiducial", "0.31%", "1.20%",   # stage table
                   "## End-to-end", "h_high", "2.31%", "## Seams", "1.2e-03",           # e2e + seams
                   "## Gradients", "-1.23e+07", "## Worst case per spectrum", "pk_gg_l4",
                   "1 failed", "1 skipped", "reference missing", "row 28"):
        assert needle in text, needle
```

```python
# scripts/summarize_ept_validation.py
"""Render the clax-pt validation campaign log into a markdown report
(Part 2, Task C3; spec §4.9).

    python scripts/summarize_ept_validation.py --errors test_logs/ptval/errors.jsonl \
        --junit test_logs/ptval/junit-campaign.xml --out docs/validation/2026-09-clax-pt-multipoles.md \
        [--job-id N] [--preset fast|contract]

Reads only the JSONL written by tests.ept_campaign_utils.log_record and
pytest's junit XML; pure Python (no JAX), runs on the login node.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from scripts import validation_cosmologies as vc            # noqa: E402
from tests.ept_campaign_utils import SPECTRA, THRESHOLDS, SEAM_THRESHOLDS   # noqa: E402

FAMILY_TITLES = OrderedDict([("lcdm", "LCDM"), ("nulcdm", "nuLCDM"), ("w0wacdm", "w0waCDM")])   # keys = vc.FAMILIES


def load_records(path: Path) -> dict[tuple, dict]:
    """Last record per (layer, case, z, preset) wins."""
    out: dict[tuple, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[(r["layer"], r["case"], round(float(r["z"]), 3), r["preset"])] = r
    return out


def load_junit(path: Path | None):
    if path is None or not path.exists():
        return {"total": 0, "failed": 0, "skipped": 0, "failures": [], "skips": []}
    root = ET.parse(path).getroot()
    res = {"total": 0, "failed": 0, "skipped": 0, "failures": [], "skips": []}
    for tc in root.iter("testcase"):
        res["total"] += 1
        name = f"{tc.get('classname')}::{tc.get('name')}"
        f = tc.find("failure") or tc.find("error")
        if f is not None:
            res["failed"] += 1
            res["failures"].append((name, f.get("message", "")))
        s = tc.find("skipped")
        if s is not None:
            res["skipped"] += 1
            res["skips"].append((name, s.get("message", "")))
    return res


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def family_of(case: str) -> str:
    for fam, names in vc.FAMILIES.items():
        if case in names:
            return fam
    return "other"


def spectra_table(records: dict, layer: str, preset_filter) -> list[str]:
    lines = []
    for fam, title in FAMILY_TITLES.items():
        rows = [(k, r) for k, r in records.items()
                if k[0] == layer and preset_filter(k[3]) and family_of(k[1]) == fam]
        if not rows:
            continue
        lines += [f"### {title}", "", "| case | z | " + " | ".join(SPECTRA) + " |",
                  "|" + "---|" * (2 + len(SPECTRA))]
        for (_, case, z, _), r in sorted(rows, key=lambda kr: (vc.FAMILIES[fam].index(kr[0][1]), kr[0][2])):
            cells = []
            for s in SPECTRA:
                e = r["errors"].get(s)
                if e is None:
                    cells.append("–")
                    continue
                flag = " **!**" if e["err"] > THRESHOLDS[s] else ""
                cells.append(f"{pct(e['err'])} @{e['k']:.3f}{flag}")
            lines.append(f"| {case} | {z:.2f} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def seams_table(records: dict, preset_filter) -> list[str]:
    keys = ["hratio", "Dratio", "H_z", "rs_d", "f", "pk_lin", "pk_lin_tail", "pk_m_lin"]
    lines = ["| case | z | " + " | ".join(keys) + " |", "|" + "---|" * (2 + len(keys))]
    for (layer, case, z, preset), r in sorted(records.items(), key=lambda kr: (family_of(kr[0][1]), kr[0][1], kr[0][2])):
        if layer != "e2e" or not preset_filter(preset) or not r.get("seams"):
            continue
        cells = []
        for k in keys:
            v = r["seams"].get(k)
            if v is None:
                cells.append("–")
            else:
                flag = " **!**" if k in SEAM_THRESHOLDS and v > SEAM_THRESHOLDS[k] else ""
                cells.append(f"{v:.1e}{flag}")
        lines.append(f"| {case} | {z:.2f} | " + " | ".join(cells) + " |")
    return lines + [""]


def diagnostics(records: dict) -> list[str]:
    lines = []
    for preset, title in [("stage-biasnz", "BIAS_NONZERO (all rows live)"),
                          ("stage-cb-minus-m", "cb minus m (0.06 eV, z=0.38)"),
                          ("stage-ap-minus-noap", "AP on minus AP off (z=0.38)"),
                          ("stage-ppf-minus-noppf", "w0wa: ppf minus noppf (z=0.38)")]:
        rs = [r for k, r in records.items() if k[3] == preset]
        if not rs:
            continue
        r = rs[-1]
        lines += [f"- **{title}** ({r['case']}): " + ", ".join(
            f"{s} {pct(r['errors'][s]['err'])}" for s in SPECTRA if s in r["errors"])]
        if "pk_lin_delta" in r.get("extra", {}):
            lines[-1] += f"; P_lin {pct(r['extra']['pk_lin_delta'])}"
    rows = [(k, r) for k, r in records.items() if k[0] == "stage" and r.get("extra", {}).get("rows")]
    if rows:
        worst_rows = {}
        for _, r in rows:
            for d in r["extra"]["rows"]:
                if d["err"] > worst_rows.get(d["row"], (0.0, None))[0]:
                    worst_rows[d["row"]] = (d["err"], r["case"])
        top = sorted(worst_rows.items(), key=lambda kv: -kv[1][0])[:5]
        lines.append("- **pk_mult rows, worst over the stage sweep:** " + ", ".join(
            f"row {i} {pct(e)} ({c})" for i, (e, c) in top))
    return lines + [""]


def gradients(records: dict) -> list[str]:
    rs = [(k, r) for k, r in records.items() if k[0] == "grad"]
    if not rs:
        return ["(no gradient records)", ""]
    lines = ["| case | d(sum pk_gg_l0)/d(omega_cdm) | value | seconds |", "|---|---|---|---|"]
    for (_, case, _, _), r in rs:
        x = r["extra"]
        lines.append(f"| {case} | {x['d_sum_pk_gg_l0_d_omega_cdm']:.2e} | {x['value']:.2e} | {x['seconds']:.0f} |")
    return lines + [""]


def worst_per_spectrum(records: dict, layer: str, preset_filter) -> list[str]:
    lines = ["| spectrum | threshold | worst | case | z | k |", "|---|---|---|---|---|---|"]
    for s in SPECTRA:
        best = None
        for (l, case, z, preset), r in records.items():
            if l != layer or not preset_filter(preset) or s not in r["errors"]:
                continue
            e = r["errors"][s]
            if best is None or e["err"] > best[0]:
                best = (e["err"], case, z, e["k"])
        if best:
            lines.append(f"| {s} | {pct(THRESHOLDS[s])} | {pct(best[0])} | {best[1]} | {best[2]:.2f} | {best[3]:.3f} |")
    return lines + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--errors", type=Path, required=True)
    ap.add_argument("--junit", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--job-id", default="n/a")
    ap.add_argument("--preset", default=None, help="e2e preset to tabulate (default: all)")
    a = ap.parse_args()

    records = load_records(a.errors)
    junit = load_junit(a.junit)
    shas = sorted({r["git_sha"] for r in records.values()})
    presets = sorted({k[3] for k in records if k[0] == "e2e"})
    pf = (lambda p: p == a.preset) if a.preset else (lambda p: True)

    L = [f"# clax-pt vs CLASS-PT: power spectrum multipoles (generated {dt.date.today():%b %-d, %Y})", "",
         "Two-layer validation of `clax.ept` against CLASS-PT (spec "
         "`docs/superpowers/specs/2026-09-03-clax-pt-validation-design.md`). Metric: "
         "max|P_clax − P_ref| / max|P_ref| on k_h[10] ≤ k ≤ 0.3 h/Mpc; thresholds "
         + ", ".join(f"{s} {pct(t)}" for s, t in THRESHOLDS.items()) + ". `**!**` marks a violation.", "",
         "## Provenance", "",
         f"- clax git sha: {', '.join(shas) or 'n/a'}; SLURM job: {a.job_id}",
         f"- e2e presets present: {', '.join(presets) or 'none'}" + (f" (tabulated: {a.preset})" if a.preset else ""),
         f"- pytest: {junit['total']} collected, {junit['failed']} failed, {junit['skipped']} skipped",
         f"- records: {len(records)} (last-wins per layer/case/z/preset)", ""]
    if junit["failures"]:
        L += ["### Failures", ""] + [f"- `{n}` — {m}" for n, m in junit["failures"]] + [""]
    if junit["skips"]:
        L += ["### Skips", ""] + [f"- `{n}` — {m}" for n, m in junit["skips"]] + [""]
    L += ["## Stage layer (clax.ept on CLASS-PT inputs)", ""]
    L += spectra_table(records, "stage", lambda p: p == "stage")
    L += ["## End-to-end layer (clax pipeline → clax.ept)", ""]
    L += spectra_table(records, "e2e", pf)
    L += ["## Seams (e2e; |clax/ref − 1|)", ""] + seams_table(records, pf)
    L += ["## Diagnostics (lcdm_fiducial, z=0.38)", ""] + diagnostics(records)
    L += ["## Gradients", ""] + gradients(records)
    L += ["## Worst case per spectrum", "", "Stage:", ""] + worst_per_spectrum(records, "stage", lambda p: p == "stage")
    L += ["End-to-end:", ""] + worst_per_spectrum(records, "e2e", pf)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L) + "\n")
    print(f"wrote {a.out} ({len(records)} records, {junit['failed']} failed, {junit['skipped']} skipped)")


if __name__ == "__main__":
    main()
```

Run: `cd /home/n2minh/clax-ptval && PYTHONPATH=/home/n2minh/clax-ptval timeout 60 /home/n2minh/micromamba/envs/clax/bin/python -m pytest tests/test_summarize_ept_validation.py -q -p no:cacheprovider 2>&1 | tail -n 3` → `1 passed`. (`vc.FAMILIES` maps `"lcdm" | "nulcdm" | "w0wacdm"` → a 5-tuple of case names (A1); `massive_nu_006` is an alias of `lcdm_fiducial` and has no record of its own, so the nuLCDM table shows four rows — say so in the report's provenance if a reader could miss it.)

- [ ] **Step 2: Write the campaign job**

`slurm/ept-multicosmo-e2e.sbatch` from Part 0's V100 template (job-name `ept-multicosmo-e2e`, time `06:00:00`; raise to the measured `14 × solve_seconds(contract) + 45 min` if C2's smoke says so), body:

```bash
PRESET=${PTVAL_E2E_PREC:-fast}
export PTVAL_E2E_PREC="$PRESET"
mkdir -p test_logs/ptval
rm -f test_logs/ptval/errors.jsonl test_logs/ptval/junit-campaign.xml
echo "campaign: sha=$(git rev-parse --short HEAD) preset=$PRESET job=$SLURM_JOB_ID"
status=0
python -m pytest tests/test_ept_from_clax.py tests/test_ept_multicosmo.py tests/test_ept_e2e_multicosmo.py \
    -q -p no:cacheprovider -rs -s -m "slow or not slow" \
    --junitxml=test_logs/ptval/junit-campaign.xml 2>&1 | grep -vE "^\s*$" | tail -n 120 || status=$?
python scripts/summarize_ept_validation.py --errors test_logs/ptval/errors.jsonl \
    --junit test_logs/ptval/junit-campaign.xml --job-id "$SLURM_JOB_ID" --preset "$PRESET" \
    --out docs/validation/2026-09-clax-pt-multipoles.md
echo "pytest status: $status"
[ "$status" -eq 0 ] && echo "PASS"
exit $status
```

(`git rev-parse` is read-only — Part 0's "no git ops inside sbatch" forbids writes. `-m "slow or not slow"` selects everything including `slow`; pytest's `-x` is deliberately absent so one red case does not hide the others.) Submit with the preset C2 settled on:

```bash
cd /home/n2minh/clax-ptval && sbatch --export=ALL,PTVAL_E2E_PREC=<fast|contract> slurm/ept-multicosmo-e2e.sbatch
```
Expected: `pytest status: 0`, `PASS`, and `docs/validation/2026-09-clax-pt-multipoles.md` written. Read the report's "Worst case per spectrum" tables — they are C4's input. A non-zero status is not a plan failure: the report still renders (with a `### Failures` section) and the failing (case, z, spectrum, seam) tuples go to the user as findings.

- [ ] **Step 3: Gates and commit**

Local gate: `tests/test_summarize_ept_validation.py tests/test_ept_multicosmo.py --fast -q`. Cluster gate: campaign job status above (it ran the whole Track C suite) plus `sbatch slurm/ptval-fast-suite.sbatch` → `PASS`. Commit (`commit-C3.txt`):

```
feat(ept): multi-cosmology campaign job + validation report

slurm/ept-multicosmo-e2e.sbatch runs C0-C2 on a V100 with junit output
and renders docs/validation/2026-09-clax-pt-multipoles.md through
scripts/summarize_ept_validation.py (per-family case x z x spectrum
tables, seams, diagnostics, gradients, worst case per spectrum).
Campaign job <id> (preset <p>): <n> passed, <f> failed, <s> skipped;
worst stage <spec> <x>% (<case> z=<z>); worst e2e <spec> <y>% (<case> z=<z>).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YBYncmhA8JrJQVNw4pVzmb
```

```bash
cd /home/n2minh/clax-ptval && git add slurm/ept-multicosmo-e2e.sbatch scripts/summarize_ept_validation.py tests/test_summarize_ept_validation.py docs/validation/2026-09-clax-pt-multipoles.md && git commit -F /tmp/claude-17163/-home-n2minh-clax/85880c7b-9aa1-44ed-b5d2-49320fe59b2a/scratchpad/commit-C3.txt
```

---

### Task C4: threshold ratchet, CHANGELOG, draft PR

**Files:**
- Modify: `tests/ept_campaign_utils.py` (`THRESHOLDS`, `SEAM_THRESHOLDS` only)
- Modify: `CHANGELOG.md`
- Modify: `docs/validation/2026-09-clax-pt-multipoles.md` (re-rendered after the ratchet)

**Interfaces:** consumes the report's "Worst case per spectrum" tables (C3); produces the ratcheted thresholds every later run is held to, and the draft PR.

- [ ] **Step 1: Ratchet (tighten only, ≥ 2× worst, separate commit)**

For each spectrum, the new threshold is `min(current, ceil_to_1sf(2 × worst_over_both_layers))` — e.g. worst `pk_gg_l0` 0.31% → 2× = 0.62% → `0.007`; worst `pk_gg_l4` 1.6% → `0.04` → stays `0.02` (never raised). Same for the seams from the e2e seam table (`pk_lin` worst 0.05% → `0.001` stays; `f` worst 2e-5 → `5e-5`). Edit only the dict values in `tests/ept_campaign_utils.py`; add a comment line per changed value: `# ratcheted Sep D, 2026 from worst 0.31% (lcdm_fiducial z=0.38, job <id>)`.

Verify the ratchet holds without a re-solve: the stage layer is cheap —

```bash
cd /home/n2minh/clax-ptval && sbatch --export=ALL,PTVAL_PYTEST_ARGS="tests/test_ept_multicosmo.py -q -rs" slurm/ptval-track-c.sbatch
```
→ `49 passed`, `PASS`. For the e2e layer, re-run the summarizer over the campaign's `errors.jsonl` (the thresholds are read from the module at render time, so `**!**` marks appear iff the ratchet is violated):

```bash
cd /home/n2minh/clax-ptval && PYTHONPATH=/home/n2minh/clax-ptval /home/n2minh/micromamba/envs/clax/bin/python scripts/summarize_ept_validation.py --errors test_logs/ptval/errors.jsonl --junit test_logs/ptval/junit-campaign.xml --job-id <campaign id> --preset <p> --out docs/validation/2026-09-clax-pt-multipoles.md && grep -c '\*\*!\*\*' docs/validation/2026-09-clax-pt-multipoles.md
```
→ `0`. Commit (`commit-C4a.txt`): `test(ept): ratchet campaign thresholds to >= 2x measured worst case` with the before/after table in the body and the trailers.

- [ ] **Step 2: Stale-comment check and CHANGELOG**

```bash
cd /home/n2minh/clax-ptval && grep -n "populated via AP integration" clax/ept.py; grep -n "rs_h=99.0" clax/ept.py | head
```
The first grep must be empty (B5 removed the stale `Pk_4_b1b2` note; if it is not empty, B5's step was skipped — fix it here and say so in the commit). The second shows only `compute_ept`'s signature — confirm its docstring says the default is the fiducial value and that `compute_ept_from_clax`/the stage layer pass the cosmology's own `rs_h`.

CHANGELOG entry at the top of the file (heading format `### Mon D, YYYY:`):

```markdown
### Sep D, 2026: clax-pt validated against CLASS-PT (ℓ=0,2,4; 14 cosmologies × 3 z)

- Campaign `campaign/clax-pt-validation` (draft PR #<n>): stage layer (clax.ept on
  CLASS-PT's P_cb, f, r_s, AP ratios) and end-to-end layer (clax pipeline →
  compute_ept_from_clax(omfid=0.31, field="cb")) vs CLASS-PT <commit> for
  pk_mm/gg/gm real and ℓ=0,2,4 on k_h[10] ≤ k ≤ 0.3 h/Mpc. Worst: stage <spec>
  <x>% (<case>, z); e2e <spec> <y>% (<case>, z). Report:
  `docs/validation/2026-09-clax-pt-multipoles.md`.
- Bugs fixed on the way (Part 1b): #1 <one line>, #2 ..., #5 ... (B4).
- New: `PerturbationResult.delta_cb` (P1); `clax.ap.ap_ratios` (B2);
  `compute_ept(..., hratio, Dratio)` true AP integration (B3/B5);
  `compute_ept_from_clax(omfid, field)` + `ept_inputs_from_clax` (C0); campaign
  utilities/tests (C1–C3); thresholds ratcheted to <list> (C4).
- Findings kept open: <P_cb,lin seam residual at fast preset>, <row 28 Pk_4_vd1
  n%>, <anything red in the report>. Failed approaches: <from the Track A/B logs>.
- Runs: `sbatch slurm/ept-multicosmo-e2e.sbatch` (V100, ~<t> h at preset <p>);
  login-node smoke `pytest tests/test_ept_multicosmo.py --fast -q`.
```

Fill every `<...>` from the report and the Track A/B commit bodies; an unfilled bracket is a plan failure.

- [ ] **Step 3: Gates, commit, draft PR**

Cluster gate: `sbatch slurm/ptval-fast-suite.sbatch` → `PASS`. Commit (`commit-C4b.txt`): `docs: CHANGELOG entry for the clax-pt validation campaign` + trailers. Then:

```bash
cd /home/n2minh/clax-ptval && git push -u origin campaign/clax-pt-validation && gh pr create --draft --repo MinhMPA/clax-pt --base main --head campaign/clax-pt-validation --title "clax-pt validation vs CLASS-PT: multipoles ℓ=0,2,4 over 14 cosmologies" --body-file /tmp/claude-17163/-home-n2minh-clax/85880c7b-9aa1-44ed-b5d2-49320fe59b2a/scratchpad/pr-body-ptval.md
```

`pr-body-ptval.md`: the CHANGELOG entry verbatim, the "Worst case per spectrum" tables copied from the report, the list of open findings, and the trailer block

```
🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01YBYncmhA8JrJQVNw4pVzmb
```

Draft only — no merge, no force-push, no branch deletion (standing constraint). Report the PR URL.

---

## Track C self-check (run by the orchestrator after C4)

1. `grep -n "TBD\|TODO\|fill in\|implement later" docs/superpowers/plans/2026-09-03-clax-pt-validation-part2-campaign.md` → empty.
2. Signature consistency: `grep -n "def ept_inputs_from_clax\|def compute_ept_from_clax\|def compute_ept(" clax/ept.py` shows `field`/`omfid` keyword-only and `hratio=1.0, Dratio=1.0` on `compute_ept`; `grep -rn "compute_ept_from_clax(" clax tests | grep -v "def "` shows `field="m"` in `clax/lensing.py` and `omfid=vc.OMFID, field="cb"` in the e2e test.
3. `grep -c "log_record(" tests/test_ept_multicosmo.py tests/test_ept_e2e_multicosmo.py` ≥ 7 and ≥ 2 — every comparison logs.
4. Thresholds appear in exactly one module: `grep -rn "0\.02\b.*l4\|THRESHOLDS\s*=" tests scripts` → only `tests/ept_campaign_utils.py`.
5. The report exists, has zero `**!**` marks after the ratchet, and its provenance sha equals `git rev-parse --short HEAD~2` (the campaign ran before the two C4 commits).
6. Spec §12 success criteria, each with evidence: (a) nine spectra within threshold on all 42 (case, z) points at both layers — junit `0 failed` in the campaign job; (b) seams within `SEAM_THRESHOLDS` — seam table clean; (c) gradients finite on three families — gradient table; (d) diagnostics recorded — diagnostics section non-empty; (e) draft PR open with the report linked.
