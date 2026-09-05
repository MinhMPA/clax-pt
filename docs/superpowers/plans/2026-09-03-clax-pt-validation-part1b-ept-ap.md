# clax-pt Validation — Part 1b: EPT in-loop AP, bug fixes, `delta_cb` (Tracks B, P)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Read first:** `2026-09-03-clax-pt-validation-part0-common.md` (Global Constraints, sbatch templates, commit recipe, reviewer briefs — they apply to every task here verbatim) and the CLASS-PT reference `2026-09-03-clax-pt-validation-classpt-inloop-reference.md` (cited below as *ref §N*).

**Goal:** Make `clax.ept` reproduce CLASS-PT's in-loop Alcock–Paczynski treatment (`nonlinear_pt.c:4386-4562`, `5225-5366`) for the ℓ=0,2,4 multipoles at `(hratio, Dratio) ≠ (1, 1)`, fix the five clax-pt defects found in review (ref §15 Bugs #1–#5), and expose the baryon+CDM contrast `delta_cb` from `clax.perturbations` — all guarded by an α=1 regression baseline so every step is checked against the code it replaces.

**Architecture:** The 40-node Gauss–Legendre μ-loop in `_compute_bias_spectra` (`clax/ept.py:1441-1497`) becomes one vectorized module-level function `_gl_multipoles` that evaluates all 40 nodes at once, remaps every channel to `ktrue = k·ap_fac(μ)` with `clax.interpolation.CubicSpline` (the same natural cubic spline CLASS-PT uses, ref §3), and projects with fiducial-μ Legendre weights (ref §4–§6). `(hratio, Dratio)` are traced scalars computed by a new `clax/ap.py::ap_ratios(bg, z, omfid)` mirroring `nonlinear_pt.c:1245-1296`. At `(1, 1)` the spline returns knot values bit-for-bit, so an α=1 baseline (`reference_data/ept_alpha1_baseline.npz`) detects any unintended change at 1e-10.

**Tech Stack:** JAX 0.9.2 (`jax.numpy`, `jax.vmap`, `jnp.einsum`), NumPy 2, scipy 1.17.1 (`scipy.integrate.simpson`, tests only), pytest. Python 3.14 in the `clax` micromamba env.

**Spec:** `docs/superpowers/specs/2026-09-03-clax-pt-validation-design.md` §4.2–4.6, §5, §6.2–6.5, §7 (Phase 2), §8.

## Global Constraints

Part 0 §Global Constraints applies verbatim (worktree `/home/n2minh/clax-ptval`, branch `campaign/clax-pt-validation`, login-node CPU flags, two-tier commit gate, trailers). Part 1b adds:

- **α=1 baseline policy.** `tests/test_ept_ap.py::test_alpha1_matches_frozen_baseline` compares every `EPTComponents` leaf against `reference_data/ept_alpha1_baseline.npz` at 1e-10 (relative to `max|baseline leaf|`). A task that legitimately changes a leaf must (1) prove the new value against CLASS-PT (a `pm[...]` row of the legacy file, or an identity stated in the step), (2) list the leaf in the test's `EXEMPT` dict with the task id and reason while the task is in review, and (3) refreeze with `scripts/freeze_ept_alpha1_baseline.py --reason "<task>: <what changed and why>"` in the task's last commit, emptying `EXEMPT`. B5 (spline remap) refreezes nothing: it must reproduce B4's baseline with `EXEMPT = {}`.
- **Login-node budget.** `compute_ept` on the 256-point legacy grid runs in ~20 s on CPU; every test in this plan except the two `slow` P1 solves is designed for the login node under the Part 0 CPU flags. The `slow` tests run only through `slurm/ptval-fast-suite.sbatch` (created in B1).
- **Citations.** The local CLASS-PT (`/home/n2minh/CLASS-PT`, commit `09d5531a`, 6069 lines) is refactored; clax's existing `nonlinear_pt.c:12871`-style citations refer to the original 13k-line file. New and touched comments cite the local anchors listed in Part 0 finding 19 and ref §1–§9.
- **Units.** `k_h` in h/Mpc, every `Pk_*` leaf in (Mpc/h)³ (ref §10). `Pd2d2_0` uses `kh` in h/Mpc (ref §12; the campaign applies `classy_kh_units.patch`).

---

## File structure

| File | Task | Responsibility |
|---|---|---|
| `clax/perturbations.py` | P1 | `PerturbationResult.delta_cb` (b+c contrast, all `N_ncdm`) |
| `tests/test_perturbations.py` | P1 | pytree + physics tests for `delta_cb` (appended class) |
| `scripts/freeze_ept_alpha1_baseline.py` | B1 | writes `reference_data/ept_alpha1_baseline.npz` from the legacy inputs |
| `reference_data/ept_alpha1_baseline.npz` | B1, B3, B4 | frozen α=1 `EPTComponents` leaves + provenance |
| `tests/test_ept_ap.py` | B1, B3, B4, B5, B6 | GL-node guard, α=1 baseline test, loop-transcription tests, Bug #1–#5 tests, α≠1 oracle tests, AP gradient tests |
| `slurm/ptval-fast-suite.sbatch` | B1 | cluster gate: `pytest tests/ --fast -x -q` on a V100 |
| `clax/ap.py` | B2 | `ap_ratios(bg, z, omfid)` (JAX) and `ap_ratios_np(...)` (NumPy twin) |
| `tests/test_ap.py` | B2 | z=0 identity, Omfid=Ωm bound, twin agreement, legacy oracle, FD gradients |
| `slurm/ptval-track-b-full.sbatch` | B2 | cluster gate part 2: Track B's own test files in full mode (slow + all grid points) |
| `clax/ept.py` | B3, B4, B5 | `_gl_multipoles`; Bug #1–#5 fixes; `_simpson`/`_pd2d2_0`; `_channels_at` + `hratio`/`Dratio` plumbing |
| `tests/test_ept_accuracy.py` | B5 | legacy comparison switched to the legacy file's true `(hratio, Dratio)` |
| `tests/test_ept_assembly.py` | B7 | (extend, do not clobber) `_pm_from_leaves`; `assemble_from_pm` ↔ clax accessors cross-check |

## Task order

```
P1 ∥ B1 ∥ B2   →   B3 (loop refactor, α=1)   →   B4 (bug fixes)   →   B5 (in-loop AP)   →   B6 (AP gradients)
B7 (assembly cross-check) after B4 and Part 1a A3.
Cluster gate = slurm/ptval-fast-suite.sbatch (B1) + slurm/ptval-track-b-full.sbatch (B2): end of P1 (fast-suite only), B2 (track-b-full only), B3, B5, B7.
```

Interfaces consumed from Part 1a: `scripts/classpt_assembly.py::pd2d2_0(pk_lin_h, kh)`, `assemble_from_pm(pm, h, fz, kh, bias, Pd2d2_0)`, `reference_path(case, z, *, ap=True, omfid=0.31, cb=True, bias="fiducial", tag="")`, the regenerated legacy files `reference_data/classpt/legacy_fiducial/z0.380_ap_omfid0.31_m.npz` (keys include `hratio`, `Dratio`) and `.../z0.380_noap_m.npz`. Interfaces produced for Part 2: `compute_ept(..., hratio=1.0, Dratio=1.0)`, `clax.ap.ap_ratios`, `PerturbationResult.delta_cb`.

---

### Task P1: `PerturbationResult.delta_cb`

**Files:**
- Modify: `clax/perturbations.py:516` (field), `:530-538` (`tree_flatten`), `:1768-1777` (`_extract_sources` return), `:1977-1978` (`_pt_saved_output_count`), `:2183-2199` (constructor)
- Test: `tests/test_perturbations.py` (append a class; extend the import block at `:33-48`)

**Interfaces:**
- Consumes: nothing new.
- Produces: `PerturbationResult.delta_cb: Float[Array, "Nk Ntau"]` — `(ρ_b δ_b + ρ_cdm δ_cdm)/(ρ_b + ρ_cdm)` for every `N_ncdm`; identical to `delta_m` when the ncdm hierarchy is off (`ncdm_q_size = 0` or `N_ncdm = 0`). Part 2 C0 selects it with `field="cb"`.

Why: CLASS-PT's `cb: Yes` runs the loop on `P_cb` (spec §4.3); clax's `delta_m` includes ncdm whenever the hierarchy is active, so the campaign needs the cb contrast as a first-class output. The 14-count also repairs `_pt_saved_output_count("full")`, which returned 12 for a 13-array return (Part 0 finding: a memory-estimate miscount, not a shape bug).

- [ ] **Step 1: Write the failing tests**

Add `PerturbationResult` and `perturbations_solve` to the `from clax.perturbations import (...)` block at `tests/test_perturbations.py:33-48`, then append at the end of the file:

```python
class TestDeltaCb:
    """`PerturbationResult.delta_cb` = (ρ_b δ_b + ρ_cdm δ_cdm)/(ρ_b + ρ_cdm) (Part 1b, P1).

    Multi-cosmology rule: the two pytree tests are cosmology-independent plumbing
    (exempt). The physics tests use the session fixture (fiducial, m_ncdm=0.06,
    ncdm_q_size=5: delta_cb != delta_m) and a fit_cl solve (ncdm_q_size=0:
    delta_cb == delta_m bit-for-bit). Both solves run through the sbatch gate.
    """

    def test_full_solve_saves_fourteen_outputs(self):
        # 13 arrays returned by _extract_sources before P1, plus delta_cb.
        assert _pt_saved_output_count(solve_kind="full") == 14

    def test_pytree_roundtrip_carries_delta_cb(self):
        arrays = [jnp.full((2, 3), float(i)) for i in range(14)]
        pt = PerturbationResult(
            k_grid=jnp.array([0.1, 0.2]), tau_grid=jnp.array([1.0, 2.0, 3.0]),
            source_T0=arrays[0], source_T1=arrays[1], source_T2=arrays[2],
            source_E=arrays[3], source_lens=arrays[4], delta_m=arrays[5],
            source_SW=arrays[6], source_ISW_vis=arrays[7], source_ISW_fs=arrays[8],
            source_Doppler=arrays[9], source_Doppler_nonIBP=arrays[10],
            source_T0_noDopp=arrays[11], source_phi_plus_psi=arrays[12],
            delta_cb=arrays[13],
        )
        leaves, aux = pt.tree_flatten()
        assert len(leaves) == 16, f"expected 16 leaves (2 grids + 14 arrays), got {len(leaves)}"
        back = PerturbationResult.tree_unflatten(aux, leaves)
        assert back.delta_cb is arrays[13]
        assert back.delta_m is arrays[5]

    @pytest.mark.slow
    def test_delta_cb_below_delta_m_with_active_ncdm_hierarchy(self, pipeline_fast_cl):
        # fast_cl keeps ncdm_q_size=5, so delta_m carries the 0.06 eV neutrinos
        # (f_nu ~ 0.0045). Above the free-streaming scale delta_nu < delta_cb, hence
        # delta_m/delta_cb = 1 - f_nu (1 - delta_nu/delta_cb) sits in (1 - f_nu, 1).
        _, _, _, _, pt = pipeline_fast_cl
        assert pt.delta_cb.shape == pt.delta_m.shape
        today = -1
        mask = np.asarray(pt.k_grid) > 0.05          # 1/Mpc, well above k_fs(z=0)
        ratio = np.asarray(pt.delta_m[mask, today] / pt.delta_cb[mask, today])
        assert np.all(np.isfinite(ratio)), "non-finite delta_m/delta_cb"
        assert np.all(ratio < 1.0 - 1e-4), f"no nu suppression: max ratio {ratio.max():.6f}"
        assert np.all(ratio > 0.99), f"suppression too large: min ratio {ratio.min():.6f}"

    @pytest.mark.slow
    def test_delta_cb_equals_delta_m_without_ncdm_hierarchy(self):
        # fit_cl sets ncdm_q_size=0 -> _extract_sources takes the n_q == 0 branch,
        # where delta_m is already the b+c combination: identical arrays expected.
        params = CosmoParams()
        prec = PrecisionParams.fit_cl()
        bg = background_solve(params, prec)
        th = thermodynamics_solve(params, prec, bg)
        pt = perturbations_solve(params, prec, bg, th)
        assert np.array_equal(np.asarray(pt.delta_cb), np.asarray(pt.delta_m))
```

- [ ] **Step 2: Run the cheap tests to verify they fail**

```bash
cd /home/n2minh/clax-ptval && JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2" PYTHONPATH=/home/n2minh/clax-ptval /home/n2minh/micromamba/envs/clax/bin/python -m pytest tests/test_perturbations.py -q -k "TestDeltaCb and not slow" -p no:cacheprovider 2>&1 | tail -5
```
Expected: 2 failed — `assert 12 == 14` and `TypeError: ... unexpected keyword argument 'delta_cb'`.

- [ ] **Step 3: Add the field and the leaf**

`clax/perturbations.py:529-530` — append after `source_phi_plus_psi`:

```python
    # Weyl potential (phi+psi) in synchronous gauge (eta + alpha_prime)
    source_phi_plus_psi: Float[Array, "Nk Ntau"]

    # Baryon+CDM density contrast for galaxy clustering (CLASS-PT `cb: Yes`):
    # the delta_m combination without the ncdm term, for every N_ncdm.
    delta_cb: Float[Array, "Nk Ntau"]
```

`tree_flatten` (`:530-538`): add `self.delta_cb,` as the last entry, after `self.source_phi_plus_psi,`. `tree_unflatten` is positional (`cls(*fields)`), so the dataclass order and the flatten order must both end with `delta_cb`.

- [ ] **Step 4: Compute it in `_extract_sources` and thread it through**

Replace `clax/perturbations.py:1768-1777` with:

```python
    # Baryon+CDM density contrast (galaxy-clustering field, CLASS-PT `cb: Yes`)
    delta_cb = (rho_b * delta_b + rho_cdm * delta_cdm) / (rho_b + rho_cdm)

    # Matter density contrast (for P(k))
    # Include ncdm when hierarchy is active (n_q > 0) to match CLASS P_m(k)
    if n_q > 0 and q_ncdm is not None:
        delta_m = (rho_b * delta_b + rho_cdm * delta_cdm + rho_ncdm * delta_ncdm_src) / (rho_b + rho_cdm + rho_ncdm)
    else:
        delta_m = delta_cb

    return (source_T0, source_T1, source_T2, source_E, source_lens, delta_m,
            source_SW, source_ISW_vis, source_ISW_fs, source_Doppler,
            source_Doppler_nonIBP, source_T0_noDopp, source_phi_plus_psi,
            delta_cb)
```

`_pt_saved_output_count` (`:1977-1978`): `return 14` with the comment `# 13 source arrays + delta_cb (see _extract_sources)`.

Constructor (`:2183-2199`): add `delta_cb=all_sources[13],` after `source_phi_plus_psi=all_sources[12],`.

- [ ] **Step 5: Run the cheap tests to verify they pass, and the existing perturbation unit tests**

```bash
cd /home/n2minh/clax-ptval && JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2" PYTHONPATH=/home/n2minh/clax-ptval /home/n2minh/micromamba/envs/clax/bin/python -m pytest tests/test_perturbations.py -q -k "TestDeltaCb and not slow or batch_size or saved_output" -p no:cacheprovider 2>&1 | tail -5
```
Expected: all selected pass (the batch-size tests exercise the new count).

- [ ] **Step 6: Cluster gate**

Requires B1's `slurm/ptval-fast-suite.sbatch` (if B1 has not landed yet, create the file exactly as B1 Step 6 specifies — the two tasks produce identical content). Submit and wait per Part 0 §Submitting and waiting:

```bash
cd /home/n2minh/clax-ptval && sbatch slurm/ptval-fast-suite.sbatch
```
Expected: the log ends with `PASS`. `--fast` skips `slow`, so also run the two P1 solves once in full mode on the node: copy the template to `slurm/ptval-p1-deltacb.sbatch` with body `python -m pytest tests/test_perturbations.py -q -k TestDeltaCb -p no:cacheprovider && echo PASS` (time `01:00:00`), submit, expect `4 passed` then `PASS`. Record both job ids in the commit message body.

- [ ] **Step 7: Commit**

Write the message to `/tmp/claude-17163/-home-n2minh-clax/85880c7b-9aa1-44ed-b5d2-49320fe59b2a/scratchpad/commit-P1.txt` (Part 0 §Commit recipe):

```
feat(perturbations): add PerturbationResult.delta_cb (b+c contrast)

delta_cb = (rho_b delta_b + rho_cdm delta_cdm)/(rho_b + rho_cdm) for every
N_ncdm; equals delta_m when the ncdm hierarchy is off. Needed for the
CLASS-PT `cb: Yes` comparison (spec 2026-09-03 §4.3). Also corrects
_pt_saved_output_count("full") from 12 to 14 (13 arrays + delta_cb).

Gate: ptval-fast-suite job <id> PASS; ptval-p1-deltacb job <id> 4 passed.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YBYncmhA8JrJQVNw4pVzmb
```

```bash
cd /home/n2minh/clax-ptval && git add clax/perturbations.py tests/test_perturbations.py slurm/ptval-p1-deltacb.sbatch && git commit -F /tmp/claude-17163/-home-n2minh-clax/85880c7b-9aa1-44ed-b5d2-49320fe59b2a/scratchpad/commit-P1.txt
```

---

### Task B1: α=1 baseline, GL-node guard, cluster-gate job

**Files:**
- Create: `scripts/freeze_ept_alpha1_baseline.py`, `reference_data/ept_alpha1_baseline.npz`, `tests/test_ept_ap.py`, `slurm/ptval-fast-suite.sbatch`

**Interfaces:**
- Consumes: `reference_data/classpt_z0.38_fullrange.npz` (legacy: `k_h (256,)`, `pk_lin`, `h=0.6736`, `fz=0.71665`), `clax.ept.compute_ept`, `clax.ept.EPTComponents` (dataclass fields = leaf names), `clax.ept._GAUSS_NODES/_GAUSS_WEIGHTS`.
- Produces: `reference_data/ept_alpha1_baseline.npz` with one array per `EPTComponents` field plus `_git_sha`, `_reason`, `_jax_version`, `_leaf_names`; `tests/test_ept_ap.py` with `EXEMPT: dict[str, tuple[str, str]]`, fixtures `legacy` and `alpha1`, tests `test_gauss_table_loaded`, `test_alpha1_matches_frozen_baseline`; the sbatch job every cluster gate submits.

- [ ] **Step 1: Write the failing test file**

Create `tests/test_ept_ap.py`:

```python
"""α=1 regression guard, GL-node guard and (from B5) the in-loop AP tests for clax.ept.

The α=1 baseline `reference_data/ept_alpha1_baseline.npz` freezes every
EPTComponents leaf computed from the legacy z=0.38 inputs
(`reference_data/classpt_z0.38_fullrange.npz`). Part 1b tasks that legitimately
change a leaf list it in EXEMPT while in review and refreeze with a reason
(`scripts/freeze_ept_alpha1_baseline.py --reason ...`) in their last commit.

Multi-cosmology rule: this file is cosmology-independent numerics on one fixed
input (exempt); the multi-cosmology coverage of the AP path is Part 2 (C1/C2).
"""

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"

import dataclasses

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import clax.ept as ept_mod
from clax.ept import EPTComponents, EPTPrecisionParams, compute_ept

ROOT = os.path.join(os.path.dirname(__file__), "..")
LEGACY = os.path.join(ROOT, "reference_data", "classpt_z0.38_fullrange.npz")
BASELINE = os.path.join(ROOT, "reference_data", "ept_alpha1_baseline.npz")
RTOL_IDENTITY = 1e-10

# Leaves whose α=1 value legitimately changed since the last freeze:
# name -> (task id, reason). Must be empty at every commit that refreezes.
EXEMPT: dict = {}


def _leaf_names():
    return [f.name for f in dataclasses.fields(EPTComponents)]


def test_gauss_table_loaded():
    """40-node GL table present (ref §7); the silent 10-point leggauss fallback
    (`ept.py:72-77`) would change every multipole."""
    assert len(ept_mod._GAUSS_NODES) == 40, \
        "gauss_tab.dat not found under _CLASSPT_DIR: 10-point leggauss fallback active"
    assert abs(float(np.sum(ept_mod._GAUSS_WEIGHTS)) - 2.0) < 1e-12


@pytest.fixture(scope="module")
def legacy():
    if not os.path.isfile(LEGACY):
        pytest.skip(f"legacy reference missing: {LEGACY}")
    return np.load(LEGACY)


@pytest.fixture(scope="module")
def alpha1(legacy):
    """compute_ept at the legacy inputs, (hratio, Dratio) = (1, 1)."""
    return compute_ept(jnp.asarray(legacy["pk_lin"]), jnp.asarray(legacy["k_h"]),
                       h=float(legacy["h"]), f=float(legacy["fz"]),
                       prec=EPTPrecisionParams())


def test_alpha1_matches_frozen_baseline(alpha1):
    if not os.path.isfile(BASELINE):
        pytest.skip("run scripts/freeze_ept_alpha1_baseline.py --reason '...'")
    base = np.load(BASELINE)
    names = _leaf_names()
    missing = [n for n in names if n not in base.files]
    assert not missing, f"baseline lacks leaves {missing}: refreeze with a reason"
    worst = []
    for n in names:
        if n in EXEMPT:
            continue
        new = np.asarray(getattr(alpha1, n), dtype=float)
        old = np.asarray(base[n], dtype=float)
        assert new.shape == old.shape, f"{n}: shape {new.shape} != baseline {old.shape}"
        scale = max(float(np.max(np.abs(old))), 1e-300)   # all-zero leaves must stay zero
        worst.append((float(np.max(np.abs(new - old))) / scale, n))
    worst.sort(reverse=True)
    bad = [(n, e) for e, n in worst if e > RTOL_IDENTITY]
    assert not bad, (f"alpha=1 drift in {len(bad)} leaves vs baseline @ {base['_git_sha']}; "
                     f"worst {worst[0][1]}: {worst[0][0]:.3e} (rel. to max|baseline|); "
                     f"baseline reason: {base['_reason']}")
```

- [ ] **Step 2: Run it to verify the state**

```bash
cd /home/n2minh/clax-ptval && JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2" PYTHONPATH=/home/n2minh/clax-ptval /home/n2minh/micromamba/envs/clax/bin/python -m pytest tests/test_ept_ap.py -q -p no:cacheprovider 2>&1 | tail -5
```
Expected: `1 passed, 1 skipped` (the baseline file does not exist yet). If `test_gauss_table_loaded` fails, `/home/n2minh/CLASS-PT/pt_matrices/gauss_tab.dat` is not where `ept.py:60-64` looks — stop and report; do not weaken the assertion.

- [ ] **Step 3: Write the freeze script**

Create `scripts/freeze_ept_alpha1_baseline.py`:

```python
"""Freeze the α=1 EPTComponents baseline used by tests/test_ept_ap.py.

Input : reference_data/classpt_z0.38_fullrange.npz (legacy k_h, pk_lin, h, fz)
Output: reference_data/ept_alpha1_baseline.npz — one array per EPTComponents
        dataclass field, plus _git_sha, _reason, _jax_version, _leaf_names.

Refreeze policy (Part 1b plan, Global Constraints): only a task that legitimately
changes a leaf refreezes, and --reason must name the task and the CLASS-PT check
that justified the change. Run on the login node with the Part 0 CPU flags.
"""

import argparse
import dataclasses
import os
import subprocess
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from clax.ept import EPTComponents, EPTPrecisionParams, compute_ept

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY = os.path.join(ROOT, "reference_data", "classpt_z0.38_fullrange.npz")
OUT = os.path.join(ROOT, "reference_data", "ept_alpha1_baseline.npz")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reason", required=True,
                    help="'<task>: <what changed and why>' — recorded in the file")
    args = ap.parse_args(argv)

    ref = np.load(LEGACY)
    ept = compute_ept(jnp.asarray(ref["pk_lin"]), jnp.asarray(ref["k_h"]),
                      h=float(ref["h"]), f=float(ref["fz"]), prec=EPTPrecisionParams())
    leaves = {f.name: np.asarray(getattr(ept, f.name)) for f in dataclasses.fields(EPTComponents)}
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.strip()
    np.savez(OUT, **leaves, _git_sha=sha, _reason=args.reason,
             _jax_version=jax.__version__, _leaf_names=np.array(sorted(leaves)))
    nonfinite = [n for n, v in leaves.items() if not np.all(np.isfinite(v))]
    print(f"wrote {OUT}: {len(leaves)} leaves @ {sha[:9]}; non-finite leaves: {nonfinite or 'none'}")
    return 1 if nonfinite else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Freeze and verify the test passes**

```bash
cd /home/n2minh/clax-ptval && JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2" PYTHONPATH=/home/n2minh/clax-ptval /home/n2minh/micromamba/envs/clax/bin/python scripts/freeze_ept_alpha1_baseline.py --reason "B1: initial freeze at bf8ac18 behaviour (pre-refactor)"
cd /home/n2minh/clax-ptval && JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2" PYTHONPATH=/home/n2minh/clax-ptval /home/n2minh/micromamba/envs/clax/bin/python -m pytest tests/test_ept_ap.py -q -p no:cacheprovider 2>&1 | tail -3
```
Expected: `wrote ...: 56 leaves @ ...; non-finite leaves: none` then `2 passed`. (56 = 52 array leaves + `h, f, sigma2_bao, delta_sigma2_bao`; if the count differs, `dataclasses.fields(EPTComponents)` disagrees with ref §15 — report the actual field list in the commit body, do not edit the class.)

- [ ] **Step 5: Determinism check**

Run the freeze a second time and compare the two files:

```bash
cd /home/n2minh/clax-ptval && cp reference_data/ept_alpha1_baseline.npz /tmp/claude-17163/-home-n2minh-clax/85880c7b-9aa1-44ed-b5d2-49320fe59b2a/scratchpad/baseline_run1.npz && JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2" PYTHONPATH=/home/n2minh/clax-ptval /home/n2minh/micromamba/envs/clax/bin/python scripts/freeze_ept_alpha1_baseline.py --reason "B1: initial freeze at bf8ac18 behaviour (pre-refactor)" && JAX_PLATFORMS=cpu PYTHONPATH=/home/n2minh/clax-ptval /home/n2minh/micromamba/envs/clax/bin/python -c "
import numpy as np
a=np.load('/tmp/claude-17163/-home-n2minh-clax/85880c7b-9aa1-44ed-b5d2-49320fe59b2a/scratchpad/baseline_run1.npz'); b=np.load('reference_data/ept_alpha1_baseline.npz')
d=max(float(np.max(np.abs(a[n]-b[n]))/max(float(np.max(np.abs(a[n]))),1e-300)) for n in a['_leaf_names'])
print('max rel diff between two freezes:', d)"
```
Expected: `max rel diff between two freezes: 0.0`. A non-zero value means `compute_ept` is non-deterministic on CPU — report it; the 1e-10 tolerance was chosen assuming bit-reproducibility.

- [ ] **Step 6: Create the cluster-gate job**

Create `slurm/ptval-fast-suite.sbatch` from the Part 0 V100 template (header verbatim, `LD_LIBRARY_PATH` loop copied from `slurm/bench-v100-igpu.sbatch` if they differ) with `--time=03:00:00` and body:

```bash
python -m pytest tests/ --fast -x -q -p no:cacheprovider 2>&1 | tail -n 40
test "${PIPESTATUS[0]}" -eq 0 && echo PASS
```

Submit once to prove the job runs end to end (this is B1's own gate and the first run of the suite on the branch):

```bash
cd /home/n2minh/clax-ptval && sbatch slurm/ptval-fast-suite.sbatch
```
Expected: log ends with `PASS`. Any failure in the existing suite at this point is pre-existing on `origin/main` — record the failing test ids in the commit body and report them; do not fix them in B1.

- [ ] **Step 7: Commit**

Message file `.../scratchpad/commit-B1.txt`:

```
test(ept): freeze alpha=1 EPTComponents baseline + GL-node guard + cluster gate job

- scripts/freeze_ept_alpha1_baseline.py writes reference_data/ept_alpha1_baseline.npz
  (56 leaves from the legacy z=0.38 inputs, with git sha and reason).
- tests/test_ept_ap.py: 40-node gauss_tab guard (ref §7) and a 1e-10 per-leaf
  regression test with an EXEMPT list for in-review changes.
- slurm/ptval-fast-suite.sbatch: `pytest tests/ --fast -x -q` on a V100 (job <id>: PASS).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YBYncmhA8JrJQVNw4pVzmb
```

```bash
cd /home/n2minh/clax-ptval && git add scripts/freeze_ept_alpha1_baseline.py reference_data/ept_alpha1_baseline.npz tests/test_ept_ap.py slurm/ptval-fast-suite.sbatch && git commit -F /tmp/claude-17163/-home-n2minh-clax/85880c7b-9aa1-44ed-b5d2-49320fe59b2a/scratchpad/commit-B1.txt
```

---

### Task B2: `clax/ap.py` — CLASS-PT AP ratios `(hratio, Dratio)`

Independent of B1/B3 (touches no EPT code). Mirrors ref §1 (reference doc lines 26–60 quote `nonlinear_pt.c:1245-1296` verbatim — read them before Step 3).

**Files:**
- Create: `clax/ap.py`
- Create: `tests/test_ap.py`
- Create: `slurm/ptval-track-b-full.sbatch` (full-mode run of Track B's own test files; from B3 on "cluster gate" = `ptval-fast-suite.sbatch` AND this job)
- Read: `clax/background.py:49-110` (`BackgroundResult` fields: `tau_of_loga`, `H_of_loga` [Mpc⁻¹], `conformal_age` [Mpc], `H0` [Mpc⁻¹], `Omega_g`), `tests/conftest.py:131-175` (cosmology fixtures)

**Interfaces:**
- Consumes: `BackgroundResult` (above); Part 1a A3 npz keys `z, h, H_z` (classy `M.Hubble(z)`, Mpc⁻¹), `DA_z` (classy `M.angular_distance(z)`, Mpc), `hratio, Dratio`.
- Produces (B5 and Part 2 C0 call these — names and order are fixed):
  - `clax.ap.ap_ratios(bg: BackgroundResult, z: float, omfid: float = OMFID_DEFAULT) -> tuple[Array, Array]` — 0-d `(hratio, Dratio)`; `z`, `omfid` static Python floats, `bg` traced.
  - `clax.ap.ap_ratios_np(z: float, omfid: float, Omega_g: float, E_z: float, DM_H0: float) -> tuple[float, float]` — NumPy twin fed with CLASS-independent scalars.
  - `clax.ap.OMFID_DEFAULT = 0.31`, `clax.ap.N_DFID = 2000`.

- [ ] **Step 1: Write the failing tests**

`tests/test_ap.py`:

```python
"""clax.ap — CLASS-PT Alcock–Paczynski ratios (nonlinear_pt.c:1245-1296).

Multi-cosmology rule: the consistency/gradient tests sweep `lcdm_cosmology`
(5 points; --fast prunes to fiducial). The oracle test runs on the single
AP-on reference that stores CLASS-PT's own ratios (Part 1a A3 legacy file);
Part 2's campaign files extend the oracle sweep to 15 cosmologies × 3 z.
"""
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from clax.ap import N_DFID, OMFID_DEFAULT, ap_ratios, ap_ratios_np
from clax.background import background_solve
from clax.params import CosmoParams

ROOT = Path(__file__).resolve().parents[1]
LEGACY_AP = ROOT / "reference_data/classpt/legacy_fiducial/z0.380_ap_omfid0.31_m.npz"
C_KM_S = 299792.458          # km/s; H0 [1/Mpc] = 100 h / C_KM_S, cf. nonlinear_pt.c:1276 kmsMpc
OMEGA_G_H2 = 2.47282e-5      # photon density ω_γ for T_cmb = 2.7255 K (CLASS input.c)
Z_TEST = 0.38


def _omega_m(p: CosmoParams) -> float:
    """Total matter fraction today incl. the (non-relativistic) 0.06-eV neutrino."""
    return (p.omega_b + p.omega_cdm + p.m_ncdm / 93.14) / p.h**2


def _dfid_c_transcription(z, omfid, omega_g, Nz=N_DFID):
    """Literal loop of nonlinear_pt.c:1280-1284 (radiation term frozen at z)."""
    dz = z / (Nz - 1)
    rad = omega_g * (1 + z) ** 4
    E = lambda zz: np.sqrt(omfid * (1 + zz) ** 3 + (1 - omfid) + rad)
    return sum(dz * (1 / E(dz * j) + 1 / E(dz * (j - 1))) / 2.0 for j in range(1, Nz))


def test_constants_match_classpt():
    assert N_DFID == 2000          # nonlinear_pt.c:1280 `int Nz = 2000`
    assert OMFID_DEFAULT == 0.31   # CLASS-PT default `Omfid`


def test_z0_is_identity():
    bg = background_solve(CosmoParams())
    h_r, d_r = ap_ratios(bg, 0.0)
    assert float(h_r) == 1.0 and float(d_r) == 1.0   # nonlinear_pt.c:1267-1269, exact


def test_twin_equals_c_transcription():
    z, omfid, omega_g = 0.8, 0.29, OMEGA_G_H2 / 0.7**2
    E_z, DM_H0 = 1.5, 0.6                      # arbitrary positive inputs
    hfid = np.sqrt(omfid * (1 + z) ** 3 + (1 - omfid) + omega_g * (1 + z) ** 4)
    h_r, d_r = ap_ratios_np(z, omfid, omega_g, E_z, DM_H0)
    assert abs(h_r - E_z / hfid) < 1e-14
    assert abs(d_r - DM_H0 / _dfid_c_transcription(z, omfid, omega_g)) < 1e-13


def test_jax_matches_twin(lcdm_cosmology):
    name, params = lcdm_cosmology
    bg = background_solve(params)
    loga = -np.log1p(Z_TEST)
    E_z = float(bg.H_of_loga.evaluate(loga) / bg.H0)
    DM_H0 = float((bg.conformal_age - bg.tau_of_loga.evaluate(loga)) * bg.H0)
    want = ap_ratios_np(Z_TEST, OMFID_DEFAULT, float(bg.Omega_g), E_z, DM_H0)
    got = ap_ratios(bg, Z_TEST)
    for label, g, w in zip(("hratio", "Dratio"), got, want):
        assert abs(float(g) - w) < 1e-10 * abs(w), f"{name} {label}: jax={float(g)!r} twin={w!r}"


def test_omfid_equal_to_omega_m_gives_unit_ratios(lcdm_cosmology):
    """With Omfid = Ω_m(cosmology) the fiducial IS the cosmology up to the
    ~1e-4 radiation/neutrino terms, so both ratios sit at 1 to < 2e-3. A
    wrong H0 factor, a Mpc/(Mpc/h) slip or a D_A vs D_M mix-up shows as ≫ 1e-2."""
    name, params = lcdm_cosmology
    bg = background_solve(params)
    h_r, d_r = ap_ratios(bg, Z_TEST, omfid=_omega_m(params))
    assert abs(float(h_r) - 1) < 2e-3, f"{name} hratio={float(h_r)!r}"
    assert abs(float(d_r) - 1) < 2e-3, f"{name} Dratio={float(d_r)!r}"


@pytest.mark.skipif(not LEGACY_AP.exists(), reason="Part 1a A3 legacy AP file absent")
def test_twin_reproduces_classpt_legacy_ratios():
    """Oracle: CLASS-PT's own hratio/Dratio (get_ap_ratios, Part 1a) from its own
    H(z), D_A(z). Only the twin is exercised — the legacy cosmology has no massive
    neutrino, which CosmoParams (N_ncdm static, default 1) does not express."""
    d = np.load(LEGACY_AP)
    z, h = float(d["z"]), float(d["h"])
    H0 = 100.0 * h / C_KM_S
    E_z = float(d["H_z"]) / H0
    DM_H0 = float(d["DA_z"]) * (1 + z) * H0
    omfid = float(d["omfid"])
    h_r, d_r = ap_ratios_np(z, omfid, OMEGA_G_H2 / h**2, E_z, DM_H0)
    assert abs(h_r - float(d["hratio"])) < 1e-6, f"hratio twin={h_r!r} classpt={float(d['hratio'])!r}"
    assert abs(d_r - float(d["Dratio"])) < 1e-6, f"Dratio twin={d_r!r} classpt={float(d['Dratio'])!r}"
    assert abs(h_r - 1.0020) < 5e-4   # legacy: Ω_m=0.3153 vs Omfid=0.31 → ≈ +0.2 %


@pytest.mark.slow
def test_gradients_match_finite_differences(lcdm_cosmology):
    """d(hratio, Dratio)/d(h, omega_cdm, w0) by reverse-mode AD vs central FD."""
    name, params = lcdm_cosmology

    def ratios(p):
        return jnp.stack(ap_ratios(background_solve(p), Z_TEST))

    grads = jax.jacrev(ratios)(params)
    for field, step in (("h", 1e-4), ("omega_cdm", 1e-4), ("w0", 1e-3)):
        x0 = getattr(params, field)
        up = ratios(params.replace(**{field: x0 + step}))
        dn = ratios(params.replace(**{field: x0 - step}))
        fd = np.asarray((up - dn) / (2 * step))
        ad = np.asarray(getattr(grads, field))
        rel = np.abs(ad - fd) / np.maximum(np.abs(fd), 1e-3)
        assert rel.max() < 1e-3, f"{name} d/d{field}: ad={ad} fd={fd} rel={rel.max():.2e}"
```

`params.replace(...)` — check `CosmoParams` has a `replace` method (`grep -n "def replace" clax/params.py`); if not, use `dataclasses.replace(params, **{field: ...})`.

- [ ] **Step 2: Run to verify failure**

```bash
cd /home/n2minh/clax-ptval && JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2" PYTHONPATH=/home/n2minh/clax-ptval timeout 120 /home/n2minh/micromamba/envs/clax/bin/python -m pytest tests/test_ap.py -q --fast -p no:cacheprovider 2>&1 | tail -n 5
```
Expected: collection error `ModuleNotFoundError: No module named 'clax.ap'`.

- [ ] **Step 3: Write `clax/ap.py`**

```python
"""Alcock–Paczynski ratios for the CLASS-PT in-loop AP treatment.

Mirrors nonlinear_pt.c:1245-1296 (local CLASS-PT 09d5531a; ref §1). For an
output redshift z > 0 with AP = Yes and fiducial flat-LCDM matter fraction Omfid:

    hfid   = sqrt(Omfid (1+z)^3 + (1 - Omfid) + Omega0_g (1+z)^4)             # 1272
    hnew   = H(z) / (kmsMpc·100·h) = E(z)                                       # 1276
    hratio = hnew / hfid                                                        # 1278
    Dfid   = trapezoid_{j=1..Nz-1} dz / sqrt(Omfid (1+z')^3 + (1-Omfid)
                                          + Omega0_g (1+z)^4),  Nz = 2000       # 1280-1284
    Da     = D_A(z) · kmsMpc·100·h · (1+z) = D_M(z) H0                          # 1288
    Dratio = Da / Dfid                                                          # 1291

z = 0 and AP = No give (1, 1) (1267-1269, 1293-1296). Both ratios are h-independent
(E and D_M·H0 are dimensionless). Quirk reproduced on purpose: the radiation term
inside the Dfid integrand is frozen at the OUTPUT z, and Omega0_g is photons only.
The fiducial has no massive-neutrino or dark-energy freedom.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from clax.background import BackgroundResult

N_DFID = 2000            # nonlinear_pt.c:1280  `int Nz = 2000` (1999 trapezoid panels)
OMFID_DEFAULT = 0.31     # CLASS-PT default `Omfid` (input.c / explanatory.ini)


def _hfid(zz, z_out, omfid, omega_g, xp):
    """sqrt(Omfid(1+zz)³ + (1−Omfid) + Ω_γ(1+z_out)⁴) — nonlinear_pt.c:1272 / 1283."""
    return xp.sqrt(omfid * (1.0 + zz) ** 3 + (1.0 - omfid) + omega_g * (1.0 + z_out) ** 4)


def _dfid(z, omfid, omega_g, xp):
    """Trapezoid of nonlinear_pt.c:1280-1284 on linspace(0, z, N_DFID)."""
    zz = xp.linspace(0.0, z, N_DFID)
    return xp.trapezoid(1.0 / _hfid(zz, z, omfid, omega_g, xp), zz)


def ap_ratios_np(z: float, omfid: float, Omega_g: float, E_z: float, DM_H0: float
                 ) -> tuple[float, float]:
    """NumPy twin of `ap_ratios` from CLASS-independent scalars E(z), D_M(z)·H0."""
    if z <= 0.0:
        return 1.0, 1.0
    hratio = E_z / _hfid(z, z, omfid, Omega_g, np)
    Dratio = DM_H0 / _dfid(z, omfid, Omega_g, np)
    return float(hratio), float(Dratio)


def ap_ratios(bg: BackgroundResult, z: float, omfid: float = OMFID_DEFAULT
              ) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """(hratio, Dratio) at static redshift z for fiducial Omfid; bg is traced."""
    z = float(z)
    if z <= 0.0:                                                  # 1267-1269
        return jnp.ones(()), jnp.ones(())
    loga = -jnp.log1p(z)
    E_z = bg.H_of_loga.evaluate(loga) / bg.H0                     # 1276  hnew = H/H0
    DM_H0 = (bg.conformal_age - bg.tau_of_loga.evaluate(loga)) * bg.H0   # 1288  Da = D_M·H0 (flat)
    hratio = E_z / _hfid(z, z, omfid, bg.Omega_g, jnp)            # 1278
    Dratio = DM_H0 / _dfid(z, omfid, bg.Omega_g, jnp)             # 1291
    return hratio, Dratio
```

Facts to hold: `bg.H0` is in Mpc⁻¹ (`background.py:84`), `tau_of_loga` in Mpc, so `DM_H0` is dimensionless like CLASS-PT's `Da`. `conformal_age − τ(z)` is the comoving distance only for a flat universe — clax is flat-only, as is CLASS-PT's `Dfid`.

- [ ] **Step 4: Run the non-slow tests (login node, --fast prunes to fiducial)**

Same command as Step 2. Expected: `5 passed, 1 skipped` (the slow test) — or `4 passed, 2 skipped` if the A3 legacy AP file does not exist yet (A3 is Track A; re-run this file after A3 lands and record the oracle result in the B2 commit body or, if A3 is later, in the B5 commit body).

Then full mode without `--fast` but still excluding slow (5 background solves on CPU, ≈1 min):

```bash
cd /home/n2minh/clax-ptval && JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2" PYTHONPATH=/home/n2minh/clax-ptval timeout 120 /home/n2minh/micromamba/envs/clax/bin/python -m pytest tests/test_ap.py -q -m "not slow" -p no:cacheprovider 2>&1 | tail -n 5
```
Expected: all 5 cosmologies pass `test_jax_matches_twin` and `test_omfid_equal_to_omega_m_gives_unit_ratios`. If it exceeds the 2-minute budget, stop and move the full-mode run to Step 5's job.

- [ ] **Step 5: Track-B full-mode cluster job**

`slurm/ptval-track-b-full.sbatch` — copy `slurm/ptval-fast-suite.sbatch` (B1) and change only `--job-name=ptval-track-b-full`, the log name, and the body:

```bash
python -m pytest tests/test_ap.py tests/test_ept_ap.py tests/test_ept.py tests/test_ept_assembly.py tests/test_ept_accuracy.py -q -p no:cacheprovider 2>&1 | tail -n 40
test "${PIPESTATUS[0]}" -eq 0 && echo PASS
```

Submit: `sbatch slurm/ptval-track-b-full.sbatch`, then poll `squeue -u $USER -n ptval-track-b-full` / `sacct -j <id> -o State` and read the log. Expected: `PASS`, with `test_gradients_match_finite_differences` run at all 5 grid points (`-k gradients` shows `5 passed`). A gradient failure at only some grid points is a real finding (fiducial-only cancellation, CLAUDE.md multi-cosmology rule) — report it, do not loosen the tolerance.

- [ ] **Step 6: Commit**

Message file `.../scratchpad/commit-B2.txt`:

```
feat(ap): CLASS-PT Alcock-Paczynski ratios hratio/Dratio (nonlinear_pt.c:1245-1296)

clax/ap.py mirrors CLASS-PT's per-z AP ratios including the Dfid quirk
(radiation term frozen at the output z, photons only). NumPy twin
ap_ratios_np() fed with CLASS-PT's own H(z), D_A(z) reproduces the stored
legacy hratio/Dratio to <1e-6 (hratio-1 = <value>). AD gradients wrt
h, omega_cdm, w0 match central FD to <1e-3 at 5 LCDM grid points
(slurm job <id>: PASS).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YBYncmhA8JrJQVNw4pVzmb
```

```bash
cd /home/n2minh/clax-ptval && git add clax/ap.py tests/test_ap.py slurm/ptval-track-b-full.sbatch && git commit -F /tmp/claude-17163/-home-n2minh-clax/85880c7b-9aa1-44ed-b5d2-49320fe59b2a/scratchpad/commit-B2.txt
```

---

### Task B3: vectorize the GL μ-loop into `_gl_multipoles` (α=1, no AP yet)

Requires B1 (baseline + `tests/test_ept_ap.py`). Pure restructuring of `_compute_bias_spectra` at `(hratio, Dratio) = (1, 1)`: the 40-iteration Python loop at `clax/ept.py:1441-1497` becomes one module-level function that also absorbs the analytic counterterm / IFG2 multipoles (`:1035-1052`) and the bias reprojection CLASS-PT does in-loop (ref §5). Eight leaves change value at α=1 (listed in Step 6); every other leaf must match the B1 baseline at 1e-10. Line numbers below are for `bf8ac18`; re-anchor with `grep -n` if the file moved.

**Files:**
- Modify: `clax/ept.py:1030-1076` (analytic IFG2/ctr/tree zeros), `:1401-1497` (split + loop), `:1620-1645` (zeros + return); add `_gl_multipoles` + channel-name tuples right before `def _compute_bias_spectra` (`:920`)
- Modify: `tests/test_ept_ap.py` (append unit tests; edit `EXEMPT` twice)
- Modify: `reference_data/ept_alpha1_baseline.npz` (refreeze, Step 8)
- Read: ref §4 (lines 126–182) and §5 (184–225) — the C code the brackets below transcribe; `clax/ept.py:1206-1228` (`use_ir_rsd`, `x_nw/x_w/x_w2`, `Exp`, `pk_nw_arr`).

**Interfaces:**
- Consumes: `_GAUSS_NODES`, `_GAUSS_WEIGHTS` (`ept.py:67-77`); `qf_split(M) -> (nw, w)`, `p13_split(M13_kernel, UV) -> (nw, w)` (`:1381-1399`); `pk_nw_arr`, `_pk_w_for_ratio`, `_sig2_bao`, `_delta_sig2` (`:1221/1228`, `:1436-1439`).
- Produces (B5 extends the signature; Part 2 never calls it directly):
  - `_gl_multipoles(chan: dict[str, Array], k: Array, f, sigma2_bao, delta_sigma2_bao) -> dict[str, Array]` returning exactly the 39 keys `Pk_{0,2,4}_{vv,vd,dd}`, `Pk_{0,2,4}_{vv,vd,dd}1`, `Pk_ctr{0,2,4}`, `Pk_IFG2_0b1`, `Pk_IFG2_0`, `Pk_IFG2_2`, `Pk_Id2d2`, `Pk_Id2G2`, `Pk_IG2G2`, `Pk_{0,2,4}_{b1b2,b1bG2,b2,bG2}`.
  - `_GL_CHANNELS_LOOP` (15 names, each present in `chan` as `<name>_nw` and `<name>_w`), `_GL_CHANNELS_BIAS` (14 names), `_GL_CHANNELS_LIN = ("pk_nw", "pk_w", "pk_disc")` — 30 + 14 + 3 = 47 channel arrays, the set B5 remaps to `ktrue`.
  - `_compute_bias_spectra` returns the same 46 keys as before (7 pass-through + the 39 above).

- [ ] **Step 1: Write the failing unit tests (append to `tests/test_ept_ap.py`)**

Add to the import block: `from clax.ept import _GAUSS_NODES, _GAUSS_WEIGHTS, _gl_multipoles, _GL_CHANNELS_LOOP, _GL_CHANNELS_BIAS`. Append:

```python
# ---------------------------------------------------------------------------
# B3: _gl_multipoles on synthetic channels. Cosmology-independent numerics
# (multi-cosmology rule exempt): the inputs are random arrays, the oracle is a
# NumPy transcription of the pre-B3 scalar loop plus closed-form identities.
# ---------------------------------------------------------------------------
GL_KEYS_TREE = [f"Pk_{l}_{c}" for l in (0, 2, 4) for c in ("vv", "vd", "dd")]
GL_KEYS_LOOP = [k + "1" for k in GL_KEYS_TREE]
GL_KEYS_ALL = (GL_KEYS_TREE + GL_KEYS_LOOP
               + ["Pk_ctr0", "Pk_ctr2", "Pk_ctr4", "Pk_IFG2_0b1", "Pk_IFG2_0", "Pk_IFG2_2",
                  "Pk_Id2d2", "Pk_Id2G2", "Pk_IG2G2"]
               + [f"Pk_{l}_{b}" for l in (0, 2, 4) for b in ("b1b2", "b1bG2", "b2", "bG2")])


def _synthetic_channels(seed, nk=64, ir=True):
    rng = np.random.default_rng(seed)
    k = np.geomspace(1e-3, 1.0, nk)
    pk_nw = 1e4 * k / (1 + (k / 0.02) ** 2) ** 1.5              # smooth, positive
    pk_w = 0.05 * pk_nw * np.sin(100 * k) if ir else np.zeros(nk)
    chan = {"pk_nw": pk_nw, "pk_w": pk_w, "pk_disc": pk_nw + 0.5 * pk_w}
    for name in _GL_CHANNELS_LOOP:
        chan[name + "_nw"] = rng.normal(size=nk) * pk_nw
        chan[name + "_w"] = 0.1 * rng.normal(size=nk) * pk_nw if ir else np.zeros(nk)
    for name in _GL_CHANNELS_BIAS:
        chan[name] = rng.normal(size=nk) * pk_nw
    return jnp.asarray(k), {n: jnp.asarray(v) for n, v in chan.items()}


def _old_scalar_loop(chan, k, f, sig2, dsig2):
    """NumPy transcription of clax/ept.py:1441-1497 @ bf8ac18 (tree + 1-loop only)."""
    c = {n: np.asarray(v) for n, v in chan.items()}
    k = np.asarray(k)
    out = {key: np.zeros_like(k) for key in GL_KEYS_TREE + GL_KEYS_LOOP}
    pk_nw, pk_w = c["pk_nw"], c["pk_w"]
    pk_nw_safe = np.where(pk_nw > 1e-100, pk_nw, 1.0)
    for mu, w in zip(_GAUSS_NODES, _GAUSS_WEIGHTS):
        mu2 = float(mu) ** 2
        Sig = sig2 * (1 + f * mu2 * (2 + f)) + dsig2 * f ** 2 * mu2 * (mu2 - 1)
        Eg = np.exp(-Sig * k ** 2)
        r13 = np.where(pk_nw > 1e-100, 1.0 + (pk_w / pk_nw_safe) * Eg, 1.0)
        Pvv = ((c["P13_mu4_vv_nw"] * r13 + c["P22_mu4_vv_nw"] + (c["P22_mu4_vv_w"] + c["P13_mu4_vv_w"]) * Eg) * mu2 ** 2
               + (c["P13_mu6_nw"] * r13 + c["P22_mu6_vv_nw"] + (c["P22_mu6_vv_w"] + c["P13_mu6_w"]) * Eg) * mu2 ** 3
               + (c["P22_mu8_nw"] + c["P22_mu8_w"] * Eg) * mu2 ** 4)
        Pdd = ((c["P22_mu0_dd_nw"] + c["P13_mu0_dd_nw"] * r13 + (c["P13_mu0_dd_w"] + c["P22_mu0_dd_w"]) * Eg)
               + (c["P22_mu2_dd_nw"] + c["P13_mu2_dd_nw"] * r13 + (c["P22_mu2_dd_w"] + c["P13_mu2_dd_w"]) * Eg) * mu2
               + (c["P22_mu4_dd_nw"] + c["P22_mu4_dd_w"] * Eg) * mu2 ** 2)
        Pvd = ((c["P13_mu2_vd_nw"] * r13 + c["P22_mu2_vd_nw"] + (c["P22_mu2_vd_w"] + c["P13_mu2_vd_w"]) * Eg) * mu2
               + (c["P13_mu4_vd_nw"] * r13 + c["P22_mu4_vd_nw"] + (c["P22_mu4_vd_w"] + c["P13_mu4_vd_w"]) * Eg) * mu2 ** 2
               + (c["P22_mu6_vd_nw"] + c["P22_mu6_vd_w"] * Eg) * mu2 ** 3)
        W = {0: w * 0.5, 2: w * 2.5 * 0.5 * (3 * mu2 - 1), 4: w * 4.5 * (35 * mu2 ** 2 - 30 * mu2 + 3) / 8.0}
        p_tree = pk_nw + pk_w * Eg * (1.0 + Sig * k ** 2)
        tree = {"vv": f ** 2 * mu2 ** 2 * p_tree, "vd": 2.0 * f * mu2 * p_tree, "dd": p_tree}
        loop = {"vv": Pvv, "vd": Pvd, "dd": Pdd}
        for ell, Wl in W.items():
            for ch in ("vv", "vd", "dd"):
                out[f"Pk_{ell}_{ch}"] += Wl * tree[ch]
                out[f"Pk_{ell}_{ch}1"] += Wl * loop[ch]
    return out


def _rel(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / max(float(np.max(np.abs(b))), 1e-300))


def test_gl_multipoles_output_keys():
    k, chan = _synthetic_channels(0)
    out = _gl_multipoles(chan, k, 0.7, 30.0, 10.0)
    assert sorted(out) == sorted(GL_KEYS_ALL), sorted(set(out) ^ set(GL_KEYS_ALL))
    assert all(v.shape == k.shape for v in out.values())


@pytest.mark.parametrize("seed", [1, 2])
def test_gl_multipoles_reproduces_scalar_loop(seed):
    """Tree and 1-loop channels: the vectorized form is the old loop to round-off."""
    k, chan = _synthetic_channels(seed, ir=True)
    f, sig2, dsig2 = 0.78, 30.0, 10.0
    new = _gl_multipoles(chan, k, f, sig2, dsig2)
    old = _old_scalar_loop(chan, k, f, sig2, dsig2)
    worst = max((_rel(new[key], old[key]), key) for key in old)
    assert worst[0] < 1e-12, f"vectorized loop drifts from scalar loop: {worst[1]} rel {worst[0]:.2e}"


def test_gl_multipoles_no_ir_identities():
    """With pk_w = 0 and Σ = 0 every projection is a Legendre moment in closed form
    (∫μ²L₂ = 4/15, ∫μ⁴L₄ = 16/315): the ctr / IFG2 multipoles equal the analytic
    forms they replace (ept.py:1035-1052 @ bf8ac18) and the bias reprojection is
    the identity (Pk_4_b1b2 → 0 by orthogonality)."""
    k, chan = _synthetic_channels(3, ir=False)
    f = 0.6
    out = _gl_multipoles(chan, k, f, 0.0, 0.0)
    pk = chan["pk_nw"]
    checks = {
        "Pk_0_dd": pk, "Pk_2_vd": 2 * f * pk * (2 / 3), "Pk_4_vv": f ** 2 * pk * (8 / 35),
        "Pk_ctr0": -k ** 2 * pk, "Pk_ctr2": -k ** 2 * pk * f * (2 / 3), "Pk_ctr4": -k ** 2 * pk * f ** 2 * (8 / 35),
        "Pk_IFG2_0b1": chan["Pk_IFG2"], "Pk_IFG2_0": chan["Pk_IFG2"] * f / 3, "Pk_IFG2_2": chan["Pk_IFG2"] * 2 * f / 3,
        "Pk_Id2d2": chan["Pk_Id2d2"], "Pk_Id2G2": chan["Pk_Id2G2"], "Pk_IG2G2": chan["Pk_IG2G2"],
        "Pk_0_b1b2": chan["Pk_0_b1b2"], "Pk_2_b1b2": chan["Pk_2_b1b2"],
        "Pk_0_b1bG2": chan["Pk_0_b1bG2"], "Pk_2_b1bG2": chan["Pk_2_b1bG2"],
        "Pk_0_b2": chan["Pk_0_b2"], "Pk_2_b2": chan["Pk_2_b2"], "Pk_4_b2": chan["Pk_4_b2"],
        "Pk_0_bG2": chan["Pk_0_bG2"], "Pk_2_bG2": chan["Pk_2_bG2"], "Pk_4_bG2": chan["Pk_4_bG2"],
    }
    bad = [(n, _rel(out[n], want)) for n, want in checks.items() if _rel(out[n], want) > 1e-12]
    assert not bad, f"closed-form identities violated: {bad}"
    for n in ("Pk_4_b1b2", "Pk_4_b1bG2"):
        assert float(np.max(np.abs(out[n]))) < 1e-12 * float(np.max(np.abs(chan["Pk_0_b1b2"]))), n
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /home/n2minh/clax-ptval && JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2" PYTHONPATH=/home/n2minh/clax-ptval timeout 120 /home/n2minh/micromamba/envs/clax/bin/python -m pytest tests/test_ept_ap.py -q -p no:cacheprovider -k gl_multipoles 2>&1 | tail -n 5
```
Expected: `ImportError: cannot import name '_gl_multipoles' from 'clax.ept'`.

- [ ] **Step 3: Add `_gl_multipoles` to `clax/ept.py`**

Insert immediately before `def _compute_bias_spectra(` (`:920`):

```python
# ---------------------------------------------------------------------------
# Vectorized Gauss–Legendre μ-integration (the CLASS-PT in-loop block)
# ---------------------------------------------------------------------------
# Channel names consumed by _gl_multipoles. Each LOOP name is present in `chan`
# twice, as "<name>_nw" and "<name>_w" (no-wiggle / wiggle FFTLog parts).
_GL_CHANNELS_LOOP = (
    "P22_mu0_dd", "P13_mu0_dd",
    "P22_mu2_vd", "P22_mu2_dd", "P13_mu2_vd", "P13_mu2_dd",
    "P22_mu4_vv", "P22_mu4_vd", "P22_mu4_dd", "P13_mu4_vv", "P13_mu4_vd",
    "P22_mu6_vv", "P22_mu6_vd", "P13_mu6", "P22_mu8",
)
_GL_CHANNELS_BIAS = (
    "Pk_0_b1b2", "Pk_2_b1b2", "Pk_0_b1bG2", "Pk_2_b1bG2",
    "Pk_0_b2", "Pk_2_b2", "Pk_4_b2", "Pk_0_bG2", "Pk_2_bG2", "Pk_4_bG2",
    "Pk_Id2d2", "Pk_Id2G2", "Pk_IG2G2", "Pk_IFG2",
)
_GL_CHANNELS_LIN = ("pk_nw", "pk_w", "pk_disc")


def _gl_multipoles(chan: dict, k, f, sigma2_bao, delta_sigma2_bao) -> dict:
    """40-node Gauss–Legendre μ-integration of every RSD channel, vectorized.

    Mirrors the per-node body of CLASS-PT's AP/IR loop (local 09d5531a):
    tree, 1-loop and counterterms `nonlinear_pt.c:4386-4562` (ref §4), bias
    and IFG2 `5225-5366` (ref §5), at (hratio, Dratio) = (1, 1): axis 0 is
    the μ node (40), axis 1 is k. Projections use the FIDUCIAL-μ Legendre
    weights W_ℓ = (2ℓ+1)/2 · w · L_ℓ(μ) (4470-4471, 4534-4558); the IR
    damping uses Σ_tot(μ) (4480) evaluated at the "true" μ, which here equals
    the fiducial one.

    chan: 47 arrays on the k grid — `<name>_nw`/`<name>_w` for every
          _GL_CHANNELS_LOOP name, the _GL_CHANNELS_BIAS spectra, and
          pk_nw, pk_w (= 0 without IR), pk_disc (the FFTLog input spectrum).
    Returns 39 multipole arrays (tree ×9, loop ×9, ctr ×3, IFG2 ×3, Id2d2/Id2G2/IG2G2, bias ×12).
    """
    mu = jnp.asarray(_GAUSS_NODES)                       # (40,)
    w = jnp.asarray(_GAUSS_WEIGHTS)
    kk = k[None, :]                                      # (1, Nk)   B5: ktrue (40, Nk)
    mu_k = mu[:, None]                                   # (40, 1)   B5: mutrue
    mu2t = mu_k ** 2                                     # 4474-4477
    mu4t, mu6t, mu8t = mu2t ** 2, mu2t ** 3, mu2t ** 4
    L2 = 0.5 * (3.0 * mu ** 2 - 1.0)                     # 4470-4471, fiducial μ
    L4 = (35.0 * mu ** 4 - 30.0 * mu ** 2 + 3.0) / 8.0
    W0, W2, W4 = 0.5 * w, 2.5 * w * L2, 4.5 * w * L4     # 4534-4558 projection weights

    def proj(val, W):
        return jnp.einsum("m,mk->k", W, val)

    def c(name):                                         # B5: spline at ktrue
        return chan[name][None, :]

    def nw(name):
        return c(name + "_nw")

    def wg(name):
        return c(name + "_w")

    Sig = (sigma2_bao * (1.0 + f * mu2t * (2.0 + f))
           + delta_sigma2_bao * f ** 2 * mu2t * (mu2t - 1.0))        # 4480 Sigmatot
    Exp = jnp.exp(-Sig * kk ** 2)                                     # 4481
    Pnw, Pw = c("pk_nw"), c("pk_w")
    Pnw_safe = jnp.where(Pnw > 1e-100, Pnw, 1.0)
    P13ratio = jnp.where(Pnw > 1e-100, 1.0 + (Pw / Pnw_safe) * Exp, 1.0)   # 4485
    p_tree = Pnw + (1.0 + Sig * kk ** 2) * Pw * Exp                   # 4483
    p_lin = Pnw + Pw * Exp                                            # 4498-4500 bracket

    # 4503 P1loopvv
    Pvv = ((nw("P13_mu4_vv") * P13ratio + nw("P22_mu4_vv") + (wg("P22_mu4_vv") + wg("P13_mu4_vv")) * Exp) * mu4t
           + (nw("P13_mu6") * P13ratio + nw("P22_mu6_vv") + (wg("P22_mu6_vv") + wg("P13_mu6")) * Exp) * mu6t
           + (nw("P22_mu8") + wg("P22_mu8") * Exp) * mu8t)
    # 4512 P1loopdd
    Pdd = ((nw("P22_mu0_dd") + nw("P13_mu0_dd") * P13ratio + (wg("P13_mu0_dd") + wg("P22_mu0_dd")) * Exp)
           + (nw("P22_mu2_dd") + nw("P13_mu2_dd") * P13ratio + (wg("P22_mu2_dd") + wg("P13_mu2_dd")) * Exp) * mu2t
           + (nw("P22_mu4_dd") + wg("P22_mu4_dd") * Exp) * mu4t)
    # 4521 P1loopvd
    Pvd = ((nw("P13_mu2_vd") * P13ratio + nw("P22_mu2_vd") + (wg("P22_mu2_vd") + wg("P13_mu2_vd")) * Exp) * mu2t
           + (nw("P13_mu4_vd") * P13ratio + nw("P22_mu4_vd") + (wg("P22_mu4_vd") + wg("P13_mu4_vd")) * Exp) * mu4t
           + (nw("P22_mu6_vd") + wg("P22_mu6_vd") * Exp) * mu6t)
    # 4494-4496 tree (CLASS-PT folds it into pm rows 26/28/29; clax keeps it in its own leaves)
    tree = {"vv": f ** 2 * mu4t * p_tree, "vd": 2.0 * f * mu2t * p_tree, "dd": p_tree}
    loop = {"vv": Pvv, "vd": Pvd, "dd": Pdd}

    out = {}
    for ell, W in ((0, W0), (2, W2), (4, W4)):
        for ch in ("vv", "vd", "dd"):
            out[f"Pk_{ell}_{ch}"] = proj(tree[ch], W)
            out[f"Pk_{ell}_{ch}1"] = proj(loop[ch], W)

    # 4498-4500, 4551-4553 counterterms; clax stores -P_CTR_ℓ (pm[11..13] = -P_CTR_ℓ, ref §10)
    out["Pk_ctr0"] = -proj(kk ** 2 * p_lin, W0)
    out["Pk_ctr2"] = -proj(kk ** 2 * p_lin * f * mu2t, W2)
    out["Pk_ctr4"] = -proj(kk ** 2 * p_lin * f ** 2 * mu4t, W4)

    # 5225-5366 bias block (ref §5). p_lo rescales P_IFG2 ∝ P_lin from the
    # isotropic FFTLog input to the anisotropic resummed linear spectrum.
    Pbin = c("pk_disc")
    p_lo = p_lin / jnp.where(Pbin > 1e-100, Pbin, 1.0)
    IFG2_in = p_lo * c("Pk_IFG2")
    out["Pk_IFG2_0b1"] = proj(IFG2_in, W0)                            # 5318-5320, 5344-5346
    out["Pk_IFG2_0"] = proj(IFG2_in * f * mu2t, W0)
    out["Pk_IFG2_2"] = proj(IFG2_in * f * mu2t, W2)
    for name in ("Pk_Id2d2", "Pk_Id2G2", "Pk_IG2G2"):                # ℓ=0 only (rows 42-47 unused by clax)
        out[name] = proj(c(name), W0)
    L2t = 0.5 * (3.0 * mu2t - 1.0)                                    # L2true/L4true from the true μ
    L4t = (35.0 * mu4t - 30.0 * mu2t + 3.0) / 8.0
    for b in ("b1b2", "b1bG2"):                                       # 5325-5327, 5350-5352
        val = c(f"Pk_0_{b}") + L2t * c(f"Pk_2_{b}")
        for ell, W in ((0, W0), (2, W2), (4, W4)):
            out[f"Pk_{ell}_{b}"] = proj(val, W)
    for b in ("b2", "bG2"):
        val = c(f"Pk_0_{b}") + L2t * c(f"Pk_2_{b}") + L4t * c(f"Pk_4_{b}")
        for ell, W in ((0, W0), (2, W2), (4, W4)):
            out[f"Pk_{ell}_{b}"] = proj(val, W)
    return out
```

`V = hratio/Dratio²` (4386-4562: a factor on every term) is absent on purpose — it is 1 here and B5 folds it into `W0/W2/W4`. Line anchors inside the bias block were verified against the local refactored file on Sep 3, 2026 (`grep -n "IFG2_in\|Pb1b2_in\|Pb2_in\|LEGENDRE_PROJECT"`); re-verify with `grep -n "P_IFG2_0b1\|Pb1b2_in\|Pb2_in" /home/n2minh/CLASS-PT/source/nonlinear_pt.c` before committing and correct them in the comments if they differ.

- [ ] **Step 4: Run the unit tests**

Same command as Step 2. Expected: `4 passed` (keys, two seeds of the loop transcription, identities). A failure in `reproduces_scalar_loop` is a transcription error in one bracket — diff the failing key's bracket in `_gl_multipoles` against the transcription and ref §4 (4503/4512/4521) term by term; do not touch the test's transcription unless it disagrees with `clax/ept.py:1449-1466 @ bf8ac18` (`git show bf8ac18:clax/ept.py | sed -n 1441,1497p`).

- [ ] **Step 5: Route `_compute_bias_spectra` through `_gl_multipoles`**

Edits, top to bottom (each removes or replaces the quoted block):

1. `:1035-1042` (comment + `Pk_IFG2_0b1 = …`, `Pk_IFG2_0 = …`, `Pk_IFG2_2 = …`): delete. Replace with the single comment `# IFG2 RSD multipoles and counterterms: in-loop, see _gl_multipoles (ref §4-§5).`
2. `:1044-1052` (COUNTERTERM block): delete.
3. `:1054-1076` (RSD TREE-LEVEL comment + nine `Pk_ℓ_xx = jnp.zeros_like(k)`): delete.
4. `:1401-1416` (fifteen `…_nw, …_w = …` lines): replace with

```python
    # nw/w split of every μ-power channel, keyed for _gl_multipoles
    split = {}
    for name, M in (("P22_mu0_dd", M22), ("P22_mu2_vd", M22_mu2_vd_bare), ("P22_mu2_dd", M22_mu2_dd_bare),
                    ("P22_mu4_vv", M22_mu4_vv_bare), ("P22_mu4_vd", M22_mu4_vd_bare), ("P22_mu4_dd", M22_mu4_dd_bare),
                    ("P22_mu6_vv", M22_mu6_vv_mat), ("P22_mu6_vd", M22_mu6_vd_mat), ("P22_mu8", M22_mu8_mat)):
        split[name + "_nw"], split[name + "_w"] = qf_split(M)
    for name, M, UV in (("P13_mu0_dd", M13, UV_mu0_dd), ("P13_mu2_vd", M13_mu2_vd_bare, UV_mu2_vd),
                        ("P13_mu2_dd", M13_mu2_dd_bare, UV_mu2_dd), ("P13_mu4_vv", M13_mu4_vv_bare, UV_mu4_vv),
                        ("P13_mu4_vd", M13_mu4_vd_bare, UV_mu4_vd), ("P13_mu6", M13_mu6_mat, UV_mu6)):
        split[name + "_nw"], split[name + "_w"] = p13_split(M, UV)
```

5. `:1418-1422`: keep the comment, rewrite the four lines to read from `split` (`P22_mu6_vv = split["P22_mu6_vv_nw"] + split["P22_mu6_vv_w"] * Exp`, likewise `P22_mu6_vd`, `P22_mu8`, `P13_mu6`). These four leaves keep the ISOTROPIC `Exp` (`:1220`) — they are consumed by `_pk_mm_tree_mu68_at_mu`, not by the loop.
6. `:1424-1497` (loop init, `_pk_nw_safe`, and the whole `for _mu_g, _w_g` loop): replace with

```python
    _pk_w_for_ratio = (jnp.asarray(pk_w) if pk_w is not None else jnp.zeros_like(k)) if use_ir_rsd else jnp.zeros_like(k)
    _delta_sig2 = delta_sigma2_bao if delta_sigma2_bao is not None else 0.0
    _sig2_bao = sigma2_bao if sigma2_bao is not None else 0.0
```

7. `:1620-1623` (`Pk_4_b1b2 = zeros`, `Pk_4_b1bG2 = zeros` + their comment): delete.
8. `:1625-1645` (return dict): replace with

```python
    chan = {
        **split,
        "pk_nw": pk_nw_arr, "pk_w": _pk_w_for_ratio, "pk_disc": pk_disc,
        "Pk_IFG2": Pk_IFG2, "Pk_Id2d2": Pk_Id2d2, "Pk_Id2G2": Pk_Id2G2, "Pk_IG2G2": Pk_IG2G2,
        "Pk_0_b1b2": Pk_0_b1b2, "Pk_2_b1b2": Pk_2_b1b2, "Pk_0_b1bG2": Pk_0_b1bG2, "Pk_2_b1bG2": Pk_2_b1bG2,
        "Pk_0_b2": Pk_0_b2, "Pk_2_b2": Pk_2_b2, "Pk_4_b2": Pk_4_b2,
        "Pk_0_bG2": Pk_0_bG2, "Pk_2_bG2": Pk_2_bG2, "Pk_4_bG2": Pk_4_bG2,
    }
    gl = _gl_multipoles(chan, k, f, _sig2_bao, _delta_sig2)
    return {
        "Pk_Id2": Pk_Id2, "Pk_IG2": Pk_IG2, "Pk_IFG2": Pk_IFG2,
        "P22_mu6_vv": P22_mu6_vv, "P22_mu6_vd": P22_mu6_vd,
        "P22_mu8": P22_mu8, "P13_mu6": P13_mu6,
        **gl,
    }
```

Sanity before running: `grep -n "Pk_ctr0\|Pk_IFG2_0b1\|Pk_0_vv1\|Pk_4_b1b2" clax/ept.py` must show no remaining assignment inside `_compute_bias_spectra` (only `_gl_multipoles`, `EPTComponents` fields and the accessors), and `python -c "import clax.ept"` must import. The no-IR path (`use_ir_rsd` False) feeds `pk_nw = pk_disc`, `pk_w = 0`, Σ = 0, so `_gl_multipoles` reduces to the closed forms tested in Step 1 — no separate branch.

- [ ] **Step 6: α=1 baseline — exempt exactly the eight leaves that legitimately change, verify them against CLASS-PT**

Run the full file:

```bash
cd /home/n2minh/clax-ptval && JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2" PYTHONPATH=/home/n2minh/clax-ptval timeout 120 /home/n2minh/micromamba/envs/clax/bin/python -m pytest tests/test_ept_ap.py -q -p no:cacheprovider 2>&1 | tail -n 8
```
Expected: `test_alpha1_matches_frozen_baseline` FAILS naming `Pk_ctr0/2/4`, `Pk_IFG2_0b1/0/2`, `Pk_4_b1b2`, `Pk_4_b1bG2` — and only those. Any other leaf in the failure list is a transcription bug: fix it in Step 5's edits (the tree/loop leaves are covered by Step 1's transcription test, so a drift there means the `chan` wiring — e.g. a swapped `_nw`/`_w` key). To see the full list temporarily raise the assert into a print, or run:

```bash
cd /home/n2minh/clax-ptval && JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2" PYTHONPATH=/home/n2minh/clax-ptval timeout 120 /home/n2minh/micromamba/envs/clax/bin/python - <<'EOF'
import numpy as np, jax.numpy as jnp, dataclasses
from clax.ept import compute_ept, EPTPrecisionParams, EPTComponents
L = np.load("reference_data/classpt_z0.38_fullrange.npz"); B = np.load("reference_data/ept_alpha1_baseline.npz")
e = compute_ept(jnp.asarray(L["pk_lin"]), jnp.asarray(L["k_h"]), h=float(L["h"]), f=float(L["fz"]), prec=EPTPrecisionParams())
for fld in dataclasses.fields(EPTComponents):
    n = fld.name; new = np.asarray(getattr(e, n), float); old = np.asarray(B[n], float)
    r = np.max(np.abs(new - old)) / max(np.max(np.abs(old)), 1e-300)
    if r > 1e-10: print(f"DRIFT {n} {r:.3e}")
EOF
```

Then set, in `tests/test_ept_ap.py`:

```python
EXEMPT: dict = {
    "Pk_ctr0": ("B3", "in-loop ctr: anisotropic Exp(mu) per GL node, nonlinear_pt.c:4498-4500"),
    "Pk_ctr2": ("B3", "idem"), "Pk_ctr4": ("B3", "idem"),
    "Pk_IFG2_0b1": ("B3", "in-loop IFG2 with p_lo = (Pnw+Pw Exp(mu))/Pbin, nonlinear_pt.c:5318-5320, 5344-5346"),
    "Pk_IFG2_0": ("B3", "idem"), "Pk_IFG2_2": ("B3", "idem"),
    "Pk_4_b1b2": ("B3", "generated by the bias reprojection (5325, 5350); round-off at alpha=1"),
    "Pk_4_b1bG2": ("B3", "idem"),
}
```

and prove the new values (login node, ≈30 s). The IR path evaluates `Exp` per node, so the six ctr/IFG2 leaves move at the sub-percent level; the legacy `pm` rows are CLASS-PT's own in-loop values (AP on, α−1 ≈ 2e-3), so agreement must be ≤ 1 % everywhere on k ≤ 0.4 h/Mpc, and ≤ 0.5 % against the α=1 `noap` file if Part 1a A3 has produced it:

```bash
cd /home/n2minh/clax-ptval && JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2" PYTHONPATH=/home/n2minh/clax-ptval timeout 120 /home/n2minh/micromamba/envs/clax/bin/python - <<'EOF'
import numpy as np, jax.numpy as jnp, os
from clax.ept import compute_ept, EPTPrecisionParams
L = np.load("reference_data/classpt_z0.38_fullrange.npz"); h = float(L["h"])
e = compute_ept(jnp.asarray(L["pk_lin"]), jnp.asarray(L["k_h"]), h=h, f=float(L["fz"]), prec=EPTPrecisionParams())
files = {"legacy(AP on, alpha-1~2e-3)": L}
noap = "reference_data/classpt/legacy_fiducial/z0.380_noap_m.npz"
if os.path.isfile(noap): files["noap(alpha=1)"] = np.load(noap)
rows = {"Pk_ctr0": (11, h), "Pk_ctr2": (12, h), "Pk_ctr4": (13, h), "Pk_IFG2_0b1": (7, h**3), "Pk_IFG2_0": (8, h**3), "Pk_IFG2_2": (9, h**3)}
for tag, F in files.items():
    pm = F["pk_mult"]; kh = F["k_h"]; sel = kh <= 0.4
    for leaf, (row, unit) in rows.items():
        ref = pm[row] * unit; new = np.asarray(getattr(e, leaf))
        r = np.abs(new - ref)[sel] / np.maximum(np.abs(ref)[sel], 1e-300)
        print(f"{tag:28s} {leaf:12s} max rel {r.max():.3e} at k={kh[sel][r.argmax()]:.3f}")
for leaf in ("Pk_4_b1b2", "Pk_4_b1bG2"):
    print(leaf, "max|.|/max|Pk_0_b1b2| =", np.max(np.abs(np.asarray(getattr(e, leaf)))) / np.max(np.abs(np.asarray(e.Pk_0_b1b2))))
EOF
```

Expected: every `max rel` ≤ 1e-2 (legacy) / ≤ 5e-3 (noap); the two `Pk_4_*` ratios < 1e-10. If a ctr row disagrees by ≈ a constant factor, the culprit is the pm-row unit (`·h`, ref §10) — re-read ref §10 rather than adjusting the code; if only the legacy file disagrees while `noap` agrees, that is AP leakage in the legacy file (expected). Put the printed numbers in the commit body. Re-run the pytest command: expected `all passed` (baseline test now passes with the exemptions).

- [ ] **Step 7: `Pk_4_vd1` 17 % diagnostic (Part 0 finding, ref §15)**

`tests/test_ept_accuracy.py` tolerates a 17 % discrepancy in the hexadecapole vd 1-loop against the legacy file. With the `noap` file present, test the two hypotheses (AP leakage in the legacy file; pm28 includes the tree):

```bash
cd /home/n2minh/clax-ptval && JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2" PYTHONPATH=/home/n2minh/clax-ptval timeout 120 /home/n2minh/micromamba/envs/clax/bin/python - <<'EOF'
import numpy as np, jax.numpy as jnp, os
from clax.ept import compute_ept, EPTPrecisionParams
L = np.load("reference_data/classpt_z0.38_fullrange.npz"); h = float(L["h"])
e = compute_ept(jnp.asarray(L["pk_lin"]), jnp.asarray(L["k_h"]), h=h, f=float(L["fz"]), prec=EPTPrecisionParams())
new = np.asarray(e.Pk_4_vd + e.Pk_4_vd1)
for tag, path in (("legacy", "reference_data/classpt_z0.38_fullrange.npz"), ("noap", "reference_data/classpt/legacy_fiducial/z0.380_noap_m.npz")):
    if not os.path.isfile(path): print(tag, "absent"); continue
    F = np.load(path); ref = F["pk_mult"][28] * h**3; sel = (F["k_h"] > 0.01) & (F["k_h"] <= 0.3)
    r = np.abs(new - ref)[sel] / np.abs(ref)[sel]
    print(f"{tag}: Pk_4_vd+Pk_4_vd1 vs pm28*h^3  max rel {r.max():.3e}  median {np.median(r):.3e}")
EOF
```

Record the two numbers in the commit body. Outcome A (`noap` ≪ `legacy`): AP leakage — B5's α≠1 test will close it. Outcome B (both large): the 17 % is a real clax defect in the μ⁴/μ⁶ vd channels — open a finding for B4 with the numbers (do not fix it in B3, which is a pure refactor); the review team decides whether a B4b task is added.

- [ ] **Step 8: Refreeze, local gate, cluster gate**

```bash
cd /home/n2minh/clax-ptval && JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2" PYTHONPATH=/home/n2minh/clax-ptval timeout 120 /home/n2minh/micromamba/envs/clax/bin/python scripts/freeze_ept_alpha1_baseline.py --reason "B3: in-loop ctr/IFG2 (anisotropic Exp at each GL node, nonlinear_pt.c:4498-4500, 5318-5320, 5344-5346) and generated Pk_4_b1b2/Pk_4_b1bG2 (5325, 5350); verified vs legacy pm rows 7-9, 11-13"
```
Set `EXEMPT: dict = {}` again. Local gate (all four cheap EPT files, ≈2 min — split into two commands if the 2-minute budget is exceeded):

```bash
cd /home/n2minh/clax-ptval && JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2" PYTHONPATH=/home/n2minh/clax-ptval timeout 120 /home/n2minh/micromamba/envs/clax/bin/python -m pytest tests/test_ept_ap.py tests/test_ept.py -q -p no:cacheprovider 2>&1 | tail -n 5
cd /home/n2minh/clax-ptval && JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2" PYTHONPATH=/home/n2minh/clax-ptval timeout 120 /home/n2minh/micromamba/envs/clax/bin/python -m pytest tests/test_ept_assembly.py tests/test_ept_accuracy.py -q -p no:cacheprovider 2>&1 | tail -n 5
```
Expected: all pass. `test_ept_accuracy.py` tolerances are unchanged in B3; if it fails, the failing leaf and its new error go into the report — B3 does not edit tolerances.

Cluster gate (both jobs; this task is the largest diff in `ept.py`, and `tests/test_ept_gradients.py` / `tests/test_ept_h_channels.py` need a GPU): `sbatch slurm/ptval-fast-suite.sbatch && sbatch slurm/ptval-track-b-full.sbatch`; wait for both logs to end in `PASS`.

- [ ] **Step 9: Commit**

Message file `.../scratchpad/commit-B3.txt`:

```
refactor(ept): vectorize the GL mu-loop into _gl_multipoles (alpha=1)

The 40-iteration Python loop in _compute_bias_spectra becomes one
(40, Nk) evaluation projected with fiducial-mu Legendre weights, keyed on
47 named channels so the in-loop AP remap (B5) is a spline on the same
dict. Absorbs into the loop, as CLASS-PT does: the counterterm multipoles
(nonlinear_pt.c:4498-4500; previously analytic with the isotropic Exp),
the IFG2 multipoles with p_lo (5318-5320, 5344-5346), and the bias reprojection that
generates Pk_4_b1b2 / Pk_4_b1bG2 (5325-5327, 5350-5352; previously hard-coded 0).

alpha=1 baseline: 48 leaves unchanged at 1e-10; the 8 leaves above
refrozen after checking against legacy pm rows 7-9/11-13:
  <paste Step 6 table>
Pk_4_vd+Pk_4_vd1 vs pm28: legacy <x>, noap <y> (Step 7).
Cluster gate: ptval-fast-suite <id> PASS, ptval-track-b-full <id> PASS.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YBYncmhA8JrJQVNw4pVzmb
```

```bash
cd /home/n2minh/clax-ptval && git add clax/ept.py tests/test_ept_ap.py reference_data/ept_alpha1_baseline.npz && git commit -F /tmp/claude-17163/-home-n2minh-clax/85880c7b-9aa1-44ed-b5d2-49320fe59b2a/scratchpad/commit-B3.txt
```

---

### Task B4: fix Bugs #1–#5 (bias-kernel basis, Id2d2 sign + `Pd2d2_0`, ℓ=2/4 b1 weighting, IR-resummed `Pk_tree`, `pk_gm_real` counterterm)

Requires B3. Five independent defects, each proven against CLASS-PT by its own test before the fix (ref §9, §11, §12, §14.4, §15). All five are invisible to the existing suite because the legacy reference was generated with `b2 = bG2 = cs = 0` and `pk_gg_l2/l4` were only ever compared with `b1` alone — say so in the commit. Anchor edits by `grep -n` (B3 moved the file); bf8ac18 line numbers are orientation only. One commit per bug (Steps 4–7 each end in a commit; the baseline refreeze rides with the last).

**Files:**
- Modify: `clax/ept.py` — the `RSD BIAS CROSS-TERMS` banner (`grep -n "RSD BIAS CROSS-TERMS"`, bf8ac18 `:1499-1504`); the three `Pk_tree = pk_lin_h` lines (`grep -n "Pk_tree = pk_lin_h"`, `:1774/1787/1790`); accessors `pk_gg_real` (`:1966-1985`), `pk_gg_l0` (`:2105-2115`), `pk_gg_l2` (`:2144-2150`), `pk_gg_l4` (`:2193-2209`); new `_simpson`, `_pd2d2_0` before `def pk_mm_real` (`grep -n "^def pk_mm_real"`)
- Modify: `tests/test_ept_ap.py` (append; `EXEMPT` set then cleared), `reference_data/ept_alpha1_baseline.npz` (refreeze)
- Read: ref §9 (Id2d2 sign), §11 (accessor algebra), §12 (`Pd2d2_0`), §14.4 (`Ptree`); `/home/n2minh/CLASS-PT/python/classy.pyx:4783-4792, 4800-4806, 4878-4907`; `/home/n2minh/CLASS-PT/source/nonlinear_pt.c:2625-2640, 2999, 4529-4558, 5052-5091`

**Interfaces:**
- Consumes: `compute_ept`, `EPTComponents` (`Pk_tree, kh, f, Pk_Id2d2, Pk_Id2, Pk_{ℓ}_{vv,vd,dd}, Pk_{ℓ}_{vv,vd,dd}1, Pk_{ℓ}_{b1b2,b2,b1bG2,bG2}, Pk_ctr{0,2,4}, Pk_IFG2_{0b1,0,2}`), the accessor signatures `pk_gg_real(ept, b1, b2, bG2, bGamma3, cs=0.0, cs0=0.0, Pshot=0.0)`, `pk_gg_l0(ept, b1, b2, bG2, bGamma3, cs0=0.0, Pshot=0.0, b4=0.0)`, `pk_gg_l2(ept, b1, b2, bG2, bGamma3, cs2=0.0, b4=0.0)`, `pk_gg_l4(ept, b1, b2, bG2, bGamma3, cs4=0.0, b4=0.0)` (all unchanged).
- Produces: `clax.ept._simpson(y: Float[Array, "N"], x: Float[Array, "N"]) -> Float[Array, ""]` (scipy ≥ 1.11 `simpson` algorithm incl. the even-N end correction; N static, ≥ 3), `clax.ept._pd2d2_0(pk_tree: Float[Array, "Nk"], kh: Float[Array, "Nk"]) -> Float[Array, ""]` (classy.pyx:4791), both jit/grad-safe. B7 cross-checks `_pd2d2_0` against `scripts.classpt_assembly.pd2d2_0`; Part 2's accessor comparisons need all four fixes.

CLASS-PT facts each fix transcribes (verified Sep 3, 2026 against the local files, commit 09d5531a):
- **#1** `nonlinear_pt.c:2625-2640` `FILL_M22_BIAS`: `nu1 = -0.5*etam2[i]`, `nu2 = -0.5*etam2[l]` — the **bias** basis (`b = −1.6`) for all ten RSD bias kernels filled at 5052–5091. clax defines exactly these at bf8ac18 `:978-979`, then the matter block rebinds `nu1 = nu_i = -0.5*etam[None,:]`, `nu2 = nu_l` at `:1088-1091`, and the kernel block at `:1506-1607` reads the rebound values while multiplying the bias-basis `M22b`. The stale banner comment "nu1 = -0.5*etam2[i] … (b=-1.6 bias basis)" describes what the code no longer does.
- **#2** `classy.pyx:4654` `pk_mult[1] = -raw_pk[1] + large_b`, and `nonlinear_pt.c` stores `P_Id2d2 + large_b` (ref §9) ⇒ `pm[1] = −P_Id2d2` with `P_Id2d2 = |P_d2d2(k) − P_d2d2(k_min)| + ε ≥ 0`. `pk_gg_real` (4805) has `0.25 b2² pm[1]`, a **negative** term; `pk_gg_l0` (4886–4889) adds back the constant `0.25 b2² Pd2d2_0`, `Pd2d2_0 = simpson((pm[14]·h³)² kh³, x=ln kh)/π²` (4788–4791): the integrand is the **IR-resummed tree** `pm[14]` on the output `kh` grid. `pk_gg_l2/l4` have no b2² term.
- **#3** `classy.pyx:4900, 4907` (ref §11): ℓ=2 galaxy = `(pm18 + pm24 + b1(pm19 + pm25) + b1²·pm26 + …)h³`; ℓ=4 = `(pm20 + pm27 + b1·pm28 + b1²·pm29 + b2 pm38 + bG2 pm39 + 2cs4 pm13/h²)h³`. `nonlinear_pt.c:4529-4541, 4555-4558`: rows 26/28/29 are projected from `P1loopdd_ap_ir`/`P1loopvd_ap_ir`, which carry `p_tree` in the μ⁰/μ² slots, so the galaxy tree is `Pk_2_vv + b1 Pk_2_vd + b1² Pk_2_dd` and `Pk_4_vv + b1 Pk_4_vd + b1² Pk_4_dd`. The ℓ=4 accessor has **no** `pm40/pm41` (`P_4_b1b2/P_4_b1bG2`) terms even though the C loop fills them.
- **#4** `nonlinear_pt.c:2999` `Ptree = Pnw + Pw·e^{−ΣBAO k²}(1 + ΣBAO k²)` is `pm[14]`, used by `pk_mm_real`, `pk_gg_real`, `pk_gm_real` and `Pd2d2_0`. clax sets `Pk_tree = pk_lin_h` on both IR branches.
- **#5** `classy.pyx:4821` `pk_gm_real`: counterterm `(2·cs·b1 + cs0)·pm[10]/h²`. clax `pk_gm_real` (`grep -n "(cs \* b1 + cs0)" clax/ept.py`, bf8ac18 `:1949`) has `(cs·b1 + cs0)·Pk_ctr` — the factor 2 on the matter counterterm is missing. It is also inconsistent with clax's own `pk_gg_real` `2(cs·b1² + cs0·b1)·Pk_ctr` (`:1978`, = classy 4805): with the galaxy counterterm `c_g = b1·cs + cs0` and the matter one `cs`, `P_gg ⊃ 2·b1·c_g` and `P_gm ⊃ c_g + b1·cs = 2·cs·b1 + cs0`. `pk_mm_real` (`2·cs0·Pk_ctr`, = classy 4798 `2·cs`) is right. Invisible so far because every reference had `cs = 0`.

- [ ] **Step 1: Write the failing tests (append to `tests/test_ept_ap.py`)**

Add to the imports: `from scipy.integrate import simpson as _scipy_simpson` and `from clax.ept import _simpson, _pd2d2_0, pk_gg_real, pk_gm_real, pk_gg_l0, pk_gg_l2, pk_gg_l4`. Move `BIAS_ROWS` (below) above `EXEMPT` at the top of the file — Step 7 uses it there. Append:

```python
# ---------------------------------------------------------------------------
# B4: Bugs #1-#4.
# #1 and #4 are checked against legacy pm rows — fiducial only, because the
# legacy file is the only CLASS-PT output with those rows until Part 2 (C1
# re-checks them at 15 cosmologies). #2/#3 are accessor algebra checked against
# classy.pyx (ref §11): cosmology-independent, exempt from the multi-cosmology rule.
# ---------------------------------------------------------------------------
BIAS_ROWS = {"Pk_0_b1b2": 30, "Pk_0_b2": 31, "Pk_0_b1bG2": 32, "Pk_0_bG2": 33,
             "Pk_2_b1b2": 34, "Pk_2_b2": 35, "Pk_2_b1bG2": 36, "Pk_2_bG2": 37,
             "Pk_4_b2": 38, "Pk_4_bG2": 39}


@pytest.mark.parametrize("n", [7, 8, 255, 256])
def test_simpson_matches_scipy(n):
    """scipy.integrate.simpson (odd N: composite; even N: + Cartwright end
    correction) reproduced in JAX on a log-spaced grid, the grid Pd2d2_0 uses."""
    rng = np.random.default_rng(n)
    x = np.geomspace(1e-3, 1.0, n)
    y = rng.normal(size=n) * x
    got = float(_simpson(jnp.asarray(y), jnp.asarray(x)))
    want = float(_scipy_simpson(y, x=x))
    assert abs(got - want) < 1e-12 * max(abs(want), 1e-300), (got, want)


def test_pd2d2_0_matches_classy_formula(legacy):
    kh = np.asarray(legacy["k_h"]); pk = np.asarray(legacy["pk_lin"])
    want = _scipy_simpson(pk ** 2 * kh ** 3, x=np.log(kh)) / np.pi ** 2      # classy.pyx:4791
    got = float(_pd2d2_0(jnp.asarray(pk), jnp.asarray(kh)))
    assert abs(got - want) < 1e-12 * abs(want)


def test_bias_kernels_use_bias_basis(legacy):
    """Bug #1. At f = 0 the b1b2 monopole kernel equals the Id2 kernel algebraically
    ((-3+2nu12)(-12+21nu12)/(42 nu1 nu2) in both FILL_M22_BIAS 5052 and the Id2
    fill), so Pk_0_b1b2(f=0) == Pk_Id2 to round-off once both use etam2."""
    e0 = compute_ept(jnp.asarray(legacy["pk_lin"]), jnp.asarray(legacy["k_h"]),
                     h=float(legacy["h"]), f=0.0, prec=EPTPrecisionParams())
    assert _rel(e0.Pk_0_b1b2, e0.Pk_Id2) < 1e-10


def test_bias_multipoles_match_legacy_rows(alpha1, legacy):
    """Bug #1, quantitatively: the ten RSD bias multipoles vs legacy pm[30..39]·h³
    on k <= 0.4 h/Mpc. Legacy has AP on with |alpha-1| ~ 2e-3, hence 1 %."""
    h = float(legacy["h"]); kh = np.asarray(legacy["k_h"]); sel = kh <= 0.4
    bad = []
    for leaf, row in BIAS_ROWS.items():
        ref = np.asarray(legacy["pk_mult"][row]) * h ** 3
        got = np.asarray(getattr(alpha1, leaf))
        r = float(np.max(np.abs(got - ref)[sel]) / np.max(np.abs(ref)[sel]))
        if r > 1e-2:
            bad.append((leaf, round(r, 4)))
    assert not bad, f"bias multipoles off vs legacy pm rows (max rel over k<=0.4): {bad}"


def test_pk_tree_is_ir_resummed(alpha1):
    """Bug #4. Pk_tree == Pnw + Pw e^{-Σk²}(1 + Σk²)  (nonlinear_pt.c:2999, pm[14])."""
    k = np.asarray(alpha1.kh); s2 = float(alpha1.sigma2_bao)
    want = np.asarray(alpha1.pk_nw) + np.asarray(alpha1.pk_w) * np.exp(-s2 * k ** 2) * (1 + s2 * k ** 2)
    assert _rel(alpha1.Pk_tree, want) < 1e-12
    raw = np.asarray(alpha1.pk_nw) + np.asarray(alpha1.pk_w)
    assert _rel(alpha1.Pk_tree, raw) > 1e-4, "Pk_tree is still the raw linear spectrum"


def test_pk_tree_matches_legacy_pm14(alpha1, legacy):
    h = float(legacy["h"]); kh = np.asarray(legacy["k_h"]); sel = kh <= 0.4
    ref = np.asarray(legacy["pk_mult"][14]) * h ** 3
    r = float(np.max(np.abs(np.asarray(alpha1.Pk_tree) - ref)[sel] / ref[sel]))
    assert r < 5e-3, f"Pk_tree vs legacy pm[14]·h³: max rel {r:.3e} on k<=0.4"


def test_accessor_algebra_matches_classy(alpha1):
    """Bugs #2, #3 and #5: pk_gg_real/pk_gm_real/l0/l2/l4 reproduce classy.pyx:4805,
    4821, 4886-4907 (ref §11) term by term, with every bias nonzero so each term
    is exercised."""
    e = alpha1
    b1, b2, bG2, bG3 = 1.9, 0.7, -0.4, 0.3
    cs, cs0, cs2, cs4, Pshot, b4 = 5.0, 10.0, 20.0, 30.0, 3000.0, 400.0
    L = lambda a: np.asarray(a)
    f = float(e.f); kh = L(e.kh); Pd2d2_0 = float(_pd2d2_0(e.Pk_tree, e.kh))
    want = {
        "real": (b1 ** 2 * (L(e.Pk_tree) + L(e.Pk_loop)) + 2 * (cs * b1 ** 2 + cs0 * b1) * L(e.Pk_ctr)
                 + b1 * b2 * L(e.Pk_Id2) - 0.25 * b2 ** 2 * L(e.Pk_Id2d2) + 2 * b1 * bG2 * L(e.Pk_IG2)
                 + b1 * (2 * bG2 + 0.8 * bG3) * L(e.Pk_IFG2) + bG2 ** 2 * L(e.Pk_IG2G2)
                 + b2 * bG2 * L(e.Pk_Id2G2) + Pshot),
        "gm": (b1 * (L(e.Pk_tree) + L(e.Pk_loop)) + (2 * cs * b1 + cs0) * L(e.Pk_ctr)      # classy.pyx:4821
               + 0.5 * b2 * L(e.Pk_Id2) + bG2 * L(e.Pk_IG2) + (bG2 + 0.4 * bG3) * L(e.Pk_IFG2)),
        "l0": (L(e.Pk_0_vv) + L(e.Pk_0_vv1) + b1 * (L(e.Pk_0_vd) + L(e.Pk_0_vd1))
               + b1 ** 2 * (L(e.Pk_0_dd) + L(e.Pk_0_dd1))
               + 0.25 * b2 ** 2 * (Pd2d2_0 - L(e.Pk_Id2d2))
               + b1 * b2 * L(e.Pk_0_b1b2) + b2 * L(e.Pk_0_b2) + b1 * bG2 * L(e.Pk_0_b1bG2) + bG2 * L(e.Pk_0_bG2)
               + b2 * bG2 * L(e.Pk_Id2G2) + bG2 ** 2 * L(e.Pk_IG2G2) + 2 * cs0 * L(e.Pk_ctr0)
               + (2 * bG2 + 0.8 * bG3) * (b1 * L(e.Pk_IFG2_0b1) + L(e.Pk_IFG2_0)) + Pshot
               + f ** 2 * b4 * kh ** 2 * (f ** 2 / 9 + 2 * f * b1 / 7 + b1 ** 2 / 5) * (35 / 8) * L(e.Pk_ctr4)),
        "l2": (L(e.Pk_2_vv) + L(e.Pk_2_vv1) + b1 * (L(e.Pk_2_vd) + L(e.Pk_2_vd1))
               + b1 ** 2 * (L(e.Pk_2_dd) + L(e.Pk_2_dd1))
               + b1 * b2 * L(e.Pk_2_b1b2) + b2 * L(e.Pk_2_b2) + b1 * bG2 * L(e.Pk_2_b1bG2) + bG2 * L(e.Pk_2_bG2)
               + 2 * cs2 * L(e.Pk_ctr2) + (2 * bG2 + 0.8 * bG3) * L(e.Pk_IFG2_2)
               + f ** 2 * b4 * kh ** 2 * (70 * f ** 2 + 165 * f * b1 + 99 * b1 ** 2) * (4 / 693) * (35 / 8) * L(e.Pk_ctr4)),
        "l4": (L(e.Pk_4_vv) + L(e.Pk_4_vv1) + b1 * (L(e.Pk_4_vd) + L(e.Pk_4_vd1))
               + b1 ** 2 * (L(e.Pk_4_dd) + L(e.Pk_4_dd1))
               + b2 * L(e.Pk_4_b2) + bG2 * L(e.Pk_4_bG2) + 2 * cs4 * L(e.Pk_ctr4)
               + f ** 2 * b4 * kh ** 2 * (210 * f ** 2 + 390 * f * b1 + 143 * b1 ** 2) * (8 / 5005) * (35 / 8) * L(e.Pk_ctr4)),
    }
    got = {
        "real": pk_gg_real(e, b1, b2, bG2, bG3, cs=cs, cs0=cs0, Pshot=Pshot),
        "gm": pk_gm_real(e, b1, b2, bG2, bG3, cs0=cs0, cs=cs),
        "l0": pk_gg_l0(e, b1, b2, bG2, bG3, cs0=cs0, Pshot=Pshot, b4=b4),
        "l2": pk_gg_l2(e, b1, b2, bG2, bG3, cs2=cs2, b4=b4),
        "l4": pk_gg_l4(e, b1, b2, bG2, bG3, cs4=cs4, b4=b4),
    }
    bad = [(n, _rel(got[n], want[n])) for n in got if _rel(got[n], want[n]) > 1e-12]
    assert not bad, f"accessor algebra differs from classy.pyx (ref §11): {bad}"


def test_galaxy_accessors_reduce_to_matter(alpha1):
    """b1 = 1, all other biases 0 must give the matter multipoles (a consequence of
    the classy formulas: pm26/28/29 carry the tree, ref §11)."""
    for name, gg, mm in (("l0", pk_gg_l0, pk_mm_l0), ("l2", pk_gg_l2, pk_mm_l2), ("l4", pk_gg_l4, pk_mm_l4)):
        assert _rel(gg(alpha1, 1.0, 0.0, 0.0, 0.0), mm(alpha1)) < 1e-12, name
```
(`pk_mm_l0/l2/l4` are in the B2 import line; add them if not.)

- [ ] **Step 2: Run to verify failure**

```bash
cd /home/n2minh/clax-ptval && JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2" PYTHONPATH=/home/n2minh/clax-ptval timeout 120 /home/n2minh/micromamba/envs/clax/bin/python -m pytest tests/test_ept_ap.py -q -p no:cacheprovider -k "simpson or pd2d2 or bias_kernels or bias_multipoles or pk_tree or accessor_algebra or reduce_to_matter" 2>&1 | tail -n 14
```
Expected: collection error `ImportError: cannot import name '_simpson'`. Do Step 3, re-run: `test_simpson_matches_scipy[*]` and `test_pd2d2_0_matches_classy_formula` pass; `test_bias_kernels_use_bias_basis`, `test_bias_multipoles_match_legacy_rows`, `test_pk_tree_is_ir_resummed`, `test_pk_tree_matches_legacy_pm14`, `test_accessor_algebra_matches_classy` (on `real`, `gm`, `l0`, `l2`, `l4`) FAIL; `test_galaxy_accessors_reduce_to_matter` fails on `l2` and `l4` (the ℓ=0 tree is already right). Copy the two `bad` lists and the `Pk_tree` max-rel into the scratchpad: they are the before-fix numbers for the commit bodies.

- [ ] **Step 3: `_simpson` and `_pd2d2_0` (no commit yet — they land with Bug #2)**

Insert before `def pk_mm_real(`:

```python
def _simpson(y: Float[Array, "N"], x: Float[Array, "N"]) -> Float[Array, ""]:
    """scipy.integrate.simpson (>= 1.11) for one 1-D sample set, in JAX.

    Odd N: composite Simpson over (N-1)/2 parabolic segments. Even N: the same
    over the first N-2 intervals plus Cartwright's (2017, eq. 8) correction for
    the last interval, the branch scipy takes, so the two agree to round-off
    (tests/test_ept_ap.py::test_simpson_matches_scipy). N is static; N >= 3.
    """
    n = y.shape[0]
    if n < 3:
        raise ValueError("_simpson needs at least 3 samples")
    h = jnp.diff(x)

    def basic(stop):
        h0, h1 = h[0:stop:2], h[1:stop + 1:2]
        y0, y1, y2 = y[0:stop:2], y[1:stop + 1:2], y[2:stop + 2:2]
        hsum = h0 + h1
        return jnp.sum(hsum / 6.0 * (y0 * (2.0 - h1 / h0) + y1 * hsum ** 2 / (h0 * h1) + y2 * (2.0 - h0 / h1)))

    if n % 2 == 1:
        return basic(n - 2)
    h0, h1 = h[-2], h[-1]
    alpha = (2.0 * h1 ** 2 + 3.0 * h0 * h1) / (6.0 * (h0 + h1))
    beta = (h1 ** 2 + 3.0 * h0 * h1) / (6.0 * h0)
    eta = h1 ** 3 / (6.0 * h0 * (h0 + h1))
    return basic(n - 3) + alpha * y[-1] + beta * y[-2] - eta * y[-3]


def _pd2d2_0(pk_tree: Float[Array, "Nk"], kh: Float[Array, "Nk"]) -> Float[Array, ""]:
    """Constant P_{δ²δ²} offset added back in the galaxy monopole.

    classy.pyx:4788-4791: Pd2d2_0 = simpson(Ptree² kh³, x=ln kh)/π², Ptree = pm[14]·h³
    (the IR-resummed tree) on the output k grid. Units (Mpc/h)³.
    """
    return _simpson(pk_tree ** 2 * kh ** 3, jnp.log(kh)) / jnp.pi ** 2
```

- [ ] **Step 4: Bug #1 — rebind the bias-basis ν before the RSD bias kernels; commit**

Directly after the `RSD BIAS CROSS-TERMS` banner (the `# ===` block whose last line is `# Exact kernels from nonlinear_pt.c lines 12871–13339.`) insert

```python
    # Bias basis nu = -0.5*etam2 (FILL_M22_BIAS, nonlinear_pt.c:2625-2640): the
    # matter block above rebound nu1/nu2 to -0.5*etam; every M22b × kernel product
    # below must be in the same basis as x2.
    nu1 = -0.5 * eta_i
    nu2 = -0.5 * eta_l
```

Rewrite the banner: `nu1 = -0.5*etam2[i], nu2 = -0.5*etam2[l] (b=-1.6 bias basis)` stays (it is now true); replace `Exact kernels from nonlinear_pt.c lines 12871–13339.` with `Kernels: FILL_M22_BIAS calls at nonlinear_pt.c:5052-5091 (local 09d5531a; the 128xx-133xx numbers were the original CLASS-PT).` and change each per-kernel `# Ref: nonlinear_pt.c line NNNNN` comment in the block to its `FILL_M22_BIAS` line: `0_b1b2 5052, 0_b2 5054, 0_b1bG2 5056, 0_bG2 5058, 2_b1b2 5076, 2_b2 5078, 2_b1bG2 5080, 2_bG2 5083, 4_b2 5089, 4_bG2 5091` (`grep -n "FILL_M22_BIAS(pnlpt->M22_" /home/n2minh/CLASS-PT/source/nonlinear_pt.c` to confirm).

Run Step 2's command: `test_bias_kernels_use_bias_basis` and `test_bias_multipoles_match_legacy_rows` pass; the others' status unchanged. Then run the whole file: `test_alpha1_matches_frozen_baseline` fails naming exactly the ten `BIAS_ROWS` leaves plus `Pk_4_b1b2`, `Pk_4_b1bG2` (reprojected from the b1b2/b1bG2 channels in `_gl_multipoles`). Anything else named is a regression — stop and find it. Set

```python
EXEMPT: dict = {
    **{leaf: ("B4", "Bug #1: bias-basis nu in the RSD bias kernels, FILL_M22_BIAS 2625-2640") for leaf in BIAS_ROWS},
    "Pk_4_b1b2": ("B4", "Bug #1, via the reprojection of the b1b2 channels"),
    "Pk_4_b1bG2": ("B4", "Bug #1, via the reprojection of the b1bG2 channels"),
}
```
Whole file passes except the four Bug #2/#3/#4 tests. Commit (message file `.../scratchpad/commit-B4a.txt`):

```
fix(ept): use the b=-1.6 bias basis nu in the RSD bias kernels (Bug #1)

Pk_{0,2,4}_{b1b2,b2,b1bG2,bG2} multiplied the bias-basis FFTLog coefficients
x2 by kernels evaluated with the matter-basis nu (-0.5*etam), left over from
the block above. CLASS-PT FILL_M22_BIAS (nonlinear_pt.c:2625-2640) uses
nu = -0.5*etam2. Now Pk_0_b1b2(f=0) == Pk_Id2 to 1e-10 and the ten
multipoles agree with legacy pm[30..39] to <1% on k<=0.4 (before: <paste>).
Invisible until now: the legacy comparison uses b2 = bG2 = 0.
Baseline: 12 leaves EXEMPT pending the B4 refreeze.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YBYncmhA8JrJQVNw4pVzmb
```

```bash
cd /home/n2minh/clax-ptval && git add clax/ept.py tests/test_ept_ap.py && git commit -F /tmp/claude-17163/-home-n2minh-clax/85880c7b-9aa1-44ed-b5d2-49320fe59b2a/scratchpad/commit-B4a.txt
```

- [ ] **Step 5: Bug #4 — IR-resummed `Pk_tree`; commit**

In `compute_ept`, `_ir_precomputed` branch: replace the comment lines + `Pk_tree = pk_lin_h` (bf8ac18 `:1768-1774`) with

```python
        # Tree-level spectrum = CLASS-PT Ptree (nonlinear_pt.c:2999, pm[14]):
        # IR-resummed with the isotropic damping, including the (1 + Σ k²) term.
        Pk_tree = pk_nw + pk_w * damp * (1.0 + sigma2_bao * k_h ** 2)
```

Default `elif prec.ir_resummation` branch: replace `pk_resummed = pk_nw + pk_w * jnp.exp(-sigma2_bao * k_h ** 2)` and the comment lines + `Pk_tree = pk_lin_h` (`:1784-1787`) with

```python
        damp = jnp.exp(-sigma2_bao * k_h ** 2)
        pk_resummed = pk_nw + pk_w * damp
        # Tree-level spectrum = CLASS-PT Ptree (nonlinear_pt.c:2999, pm[14]).
        Pk_tree = pk_nw + pk_w * damp * (1.0 + sigma2_bao * k_h ** 2)
```

The `else` branch keeps `Pk_tree = pk_lin_h` (with IR off CLASS-PT has `Pw = 0`, `Ptree = Pnw`). Check `grep -n "Pk_tree = pk_lin_h" clax/ept.py` shows one hit. Run Step 2's command: `test_pk_tree_*` pass. Whole file: baseline drift names exactly `Pk_tree` — add `"Pk_tree": ("B4", "Bug #4: IR-resummed Ptree, nonlinear_pt.c:2999")` to `EXEMPT`. Also run `tests/test_ept_accuracy.py` (`pk_mm_real`/`pk_gg_real` use `Pk_tree`): they should tighten (record the before/after max-rel from the assertion messages or by printing them); if one now fails, the error went UP — report the numbers, do not touch the tolerance. Commit (`commit-B4b.txt`):

```
fix(ept): Pk_tree is the IR-resummed tree Ptree (Bug #4)

compute_ept returned the raw linear spectrum as Pk_tree on both IR
branches; CLASS-PT's Ptree = Pnw + Pw e^{-Σk²}(1 + Σk²)
(nonlinear_pt.c:2999) is what pm[14] and the real-space accessors use.
Pk_tree vs legacy pm[14]·h³ on k<=0.4: <before> -> <after>.
test_ept_accuracy pk_mm_real/pk_gg_real: <before> -> <after>.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YBYncmhA8JrJQVNw4pVzmb
```

```bash
cd /home/n2minh/clax-ptval && git add clax/ept.py tests/test_ept_ap.py && git commit -F /tmp/claude-17163/-home-n2minh-clax/85880c7b-9aa1-44ed-b5d2-49320fe59b2a/scratchpad/commit-B4b.txt
```

- [ ] **Step 6: Bug #3 — b1 weights in `pk_gg_l2/l4`; commit**

`pk_gg_l2`: replace the comment block above `new_l2_tree` (the lines from `# Tree-level (isotropic IR, analytical)` through the `mu^0 … cancel` remark) and the assignment with

```python
    # Tree: (b1 + f mu²)² p_tree projected on L2 — classy.pyx:4900 pm18 + b1·pm19 + b1²·pm26,
    # where pm26 carries the dd tree (P1loopdd_ap_ir, nonlinear_pt.c:4529, 4537).
    new_l2_tree = ept.Pk_2_vv + b1 * ept.Pk_2_vd + b1 ** 2 * ept.Pk_2_dd
```

`pk_gg_l4`: replace the comment above `new_l4_tree` and the assignment with

```python
    # Tree: classy.pyx:4907 pm20 + b1·pm28 + b1²·pm29 — pm28/pm29 carry the vd/dd tree
    # (P1loopvd_ap_ir / P1loopdd_ap_ir, nonlinear_pt.c:4529-4541).
    new_l4_tree = ept.Pk_4_vv + b1 * ept.Pk_4_vd + b1 ** 2 * ept.Pk_4_dd
```

and in `P_bias_l4` delete `+ b1 * b2 * ept.Pk_4_b1b2` and `+ b1 * bG2 * ept.Pk_4_b1bG2`, leaving the comment `# classy.pyx:4907 has no pm[40]/pm[41] (P_4_b1b2/P_4_b1bG2) term although the C loop fills them; mirror the accessor. The leaves stay in EPTComponents.` Run Step 2's command: `test_galaxy_accessors_reduce_to_matter` passes; `test_accessor_algebra_matches_classy` still fails, now only on `real`, `gm` and `l0` (print `bad` to confirm). Baseline unchanged (accessors are not leaves). `tests/test_ept_accuracy.py`: the `pk_gg_l2/l4` comparisons at `b1 = 2` move (ℓ=2 gains `b1²·Pk_2_dd`; ℓ=4 gains `(b1−1)·Pk_4_vd + (b1²−1)·Pk_4_dd`) and should move TOWARD legacy — record before/after. Commit (`commit-B4c.txt`):

```
fix(ept): b1-weighted tree in pk_gg_l2/pk_gg_l4 (Bug #3)

pk_gg_l2 omitted b1²·Pk_2_dd; pk_gg_l4 used the unweighted matter tree.
classy.pyx:4900/4907 read pm26 = loop_2_dd + tree_2_dd (×b1²),
pm28 = loop_4_vd + tree_4_vd (×b1), pm29 = loop_4_dd + tree_4_dd (×b1²)
— nonlinear_pt.c:4529-4541 fold the tree into P1loop{dd,vd}_ap_ir.
pk_gg_l4 also carried b1 b2 Pk_4_b1b2 + b1 bG2 Pk_4_b1bG2, which classy's
accessor does not include (pm40/pm41 are filled but unused); dropped.
test_ept_accuracy pk_gg_l2/l4 vs legacy (b1=2): <before> -> <after>.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YBYncmhA8JrJQVNw4pVzmb
```

```bash
cd /home/n2minh/clax-ptval && git add clax/ept.py tests/test_ept_ap.py && git commit -F /tmp/claude-17163/-home-n2minh-clax/85880c7b-9aa1-44ed-b5d2-49320fe59b2a/scratchpad/commit-B4c.txt
```

- [ ] **Step 6b: Bug #5 — `pk_gm_real` counterterm; commit**

In `pk_gm_real` (`grep -n "(cs \* b1 + cs0)" clax/ept.py`) change

```python
        + (cs * b1 + cs0) * ept.Pk_ctr
```
to
```python
        + (2.0 * cs * b1 + cs0) * ept.Pk_ctr        # classy.pyx:4821: (2 cs b1 + cs0) pm[10]/h²
```
and the docstring line `(cs*b1 + cs0)*P_CTR/h²` to `(2 cs b1 + cs0) P_CTR`. Run Step 2's command: `test_accessor_algebra_matches_classy` now fails only on `real` and `l0`. `tests/test_ept_accuracy.py::pk_gm_real` is unaffected (legacy `cs = 0`). Commit (`commit-B4e.txt`):

```
fix(ept): factor 2 on the matter counterterm in pk_gm_real (Bug #5)

classy.pyx:4821: P_gm ⊃ (2 cs b1 + cs0) pm[10]/h². clax had (cs b1 + cs0),
inconsistent with its own pk_gg_real 2(cs b1² + cs0 b1) (galaxy counterterm
c_g = b1 cs + cs0 ⇒ P_gm ⊃ c_g + b1 cs). Invisible to every reference so far
(cs = 0); exercised by test_accessor_algebra_matches_classy (cs = 5).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YBYncmhA8JrJQVNw4pVzmb
```

```bash
cd /home/n2minh/clax-ptval && git add clax/ept.py tests/test_ept_ap.py && git commit -F /tmp/claude-17163/-home-n2minh-clax/85880c7b-9aa1-44ed-b5d2-49320fe59b2a/scratchpad/commit-B4e.txt
```

- [ ] **Step 7: Bug #2 — Id2d2 sign and `Pd2d2_0`; refreeze; gates; commit**

`pk_gg_real`: `+ 0.25 * b2 ** 2 * ept.Pk_Id2d2` → `- 0.25 * b2 ** 2 * ept.Pk_Id2d2`; docstring `+ b1 b2 P_Id2 + b2²/4 P_Id2d2` → `+ b1 b2 P_Id2 − b2²/4 P_Id2d2` and add the line `(classy.pyx:4805 with pm[1] = −P_Id2d2; no constant in real space.)`.

`pk_gg_l0`: `0.25 * b2 ** 2 * ept.Pk_Id2d2` → `0.25 * b2 ** 2 * (_pd2d2_0(ept.Pk_tree, ept.kh) - ept.Pk_Id2d2)` with the comment `# classy.pyx:4886-4889: 0.25 b2² pm[1]·h³ + 0.25 b2² Pd2d2_0, pm[1] = −P_Id2d2`.

Run Step 2's command: all pass. Whole file: `test_alpha1_matches_frozen_baseline` passes with the 13-entry `EXEMPT`. Refreeze:

```bash
cd /home/n2minh/clax-ptval && JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2" PYTHONPATH=/home/n2minh/clax-ptval timeout 120 /home/n2minh/micromamba/envs/clax/bin/python scripts/freeze_ept_alpha1_baseline.py --reason "B4: Bug #1 bias-basis nu (10 bias leaves + Pk_4_b1b2/b1bG2, verified vs legacy pm[30..39]); Bug #4 IR-resummed Pk_tree (nonlinear_pt.c:2999, verified vs pm[14])"
```

Set `EXEMPT: dict = {}`. Local gate = the two B3 Step 8 pytest commands. Cluster gate: `sbatch slurm/ptval-fast-suite.sbatch && sbatch slurm/ptval-track-b-full.sbatch`, both logs end in `PASS`. Commit (`commit-B4d.txt`):

```
fix(ept): -0.25 b2² P_Id2d2 in pk_gg_real, + 0.25 b2² Pd2d2_0 in pk_gg_l0 (Bug #2)

classy stores pm[1] = -P_Id2d2 (classy.pyx:4654), so pk_gg_real's b2² term
is negative and pk_gg_l0 adds back the constant
Pd2d2_0 = simpson(Ptree² k³, x=ln k)/π² (classy.pyx:4788-4791, 4889).
Adds _simpson (JAX twin of scipy.integrate.simpson, incl. the even-N end
correction, 1e-12 vs scipy) and _pd2d2_0. Invisible until now: the legacy
comparison uses b2 = 0.
alpha=1 baseline refrozen for Pk_tree and the 12 bias leaves (B4 Steps 4-5).
Cluster gate: ptval-fast-suite <id> PASS, ptval-track-b-full <id> PASS.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YBYncmhA8JrJQVNw4pVzmb
```

```bash
cd /home/n2minh/clax-ptval && git add clax/ept.py tests/test_ept_ap.py reference_data/ept_alpha1_baseline.npz && git commit -F /tmp/claude-17163/-home-n2minh-clax/85880c7b-9aa1-44ed-b5d2-49320fe59b2a/scratchpad/commit-B4d.txt
```

---

### Task B5: in-loop Alcock–Paczynski remap — `(hratio, Dratio)` from `compute_ept` down to `_gl_multipoles`

**Files:**
- Modify: `clax/ept.py` — `_gl_multipoles` (B3; `grep -n "^def _gl_multipoles"`), `_compute_bias_spectra` signature and its `gl = _gl_multipoles(` call, `compute_ept` signature and its `bias = _compute_bias_spectra(` call, the import block (`:36-39`). Create the helper `_channels_at` directly above `_gl_multipoles`.
- Modify: `tests/test_ept_ap.py` (append the B5 block), `tests/test_ept_accuracy.py` (`grep -n "def ept_result"`).
- Test: `tests/test_ept_ap.py`

**Interfaces:**
- Consumes: B3's `_gl_multipoles(chan, k, f, sigma2_bao, delta_sigma2_bao)` and the 47-key `chan` dict; `clax.interpolation._compute_natural_spline_coeffs(x, y) -> d2y` (`clax/interpolation.py:139-197`, Thomas algorithm in a `fori_loop`, vmappable); B4's `_pd2d2_0`; test helpers `_synthetic_channels`, `_old_scalar_loop`, `_rel`, `GL_KEYS_ALL`, fixtures `legacy`, `alpha1`, constants `ROOT`, `EXEMPT`; Part 1a A3 files `reference_data/classpt/legacy_fiducial/z0.380_ap_omfid0.31_m.npz` (keys `k_h, h, fz, hratio, Dratio, pk_lin, pk_mult, bias_json, pk_mm_real, pk_gg_real, pk_gm_real, pk_mm_l0/l2/l4, pk_gg_l0/l2/l4`) and `z0.380_noap_m.npz` (same keys, `hratio = Dratio = 1`).
- Produces: `compute_ept(pk_lin_h, k_h, h, f, prec=EPTPrecisionParams(), _ir_precomputed=None, rs_h=99.0, hratio=1.0, Dratio=1.0) -> EPTComponents` (both new kwargs accept Python floats or traced 0-d arrays); `_compute_bias_spectra(..., pk_w=None, hratio=1.0, Dratio=1.0)`; `_gl_multipoles(chan, k, f, sigma2_bao, delta_sigma2_bao, hratio=1.0, Dratio=1.0)`; `_channels_at(k, chan, kq) -> dict`. Part 2 C0 feeds `clax.ap.ap_ratios(bg, z, omfid)` into the two kwargs; `EPTComponents` gains no field (Part 2 records `hratio`/`Dratio` in its run files).

**CLASS-PT facts this task mirrors** (`/home/n2minh/CLASS-PT/source/nonlinear_pt.c` @ 09d5531a; re-`grep` each before citing):

1. `4382-4384`: `ap_inv_Dr2_ = 1/Dratio²`, `ap_hr2_minus_inv_Dr2_ = hratio² − 1/Dratio²`, `ap_w_hr_Dr2_ = hratio/Dratio²` (= V, the volume factor multiplying every accumulated term: `4494-4500`, the three `P1loop*` lines `4503/4512/4521`, and the bias block `5320-5328`).
2. `4392-4394` per node: `ap_fac = sqrt(ap_inv_Dr2_ + ap_hr2_minus_inv_Dr2_·μ²)`, `mutrue = μ·hratio/ap_fac`, `ktrue = kdisc[j]·ap_fac`. The α=1 branch `4396-4397` (`mutrue = μ`, `ktrue = kdisc[j]`) is what these formulas give at (1, 1) — `ap_fac ≡ 1` exactly in floating point, so B5 leaves every α=1 leaf bit-identical.
3. Splines: `AP_SPLINE_SETUP` `2355-2362` builds a NATURAL cubic spline in LINEAR k over the whole `kdisc` for each channel; `AP_BSEARCH_SETUP` `2383-2394` bisects `ktrue` into an interval clamped to `[0, N−2]` but computes `a = (k[sup] − ktrue)/h`, `b = 1 − a` from the UNCLAMPED `ktrue` — outside the grid the end interval's cubic is continued, not clamped; `AP_INTERP_FAST` `2372-2374` is `a·y[inf] + b·y[sup] + ((a³−a)·dd[inf] + (b³−b)·dd[sup])·h²/6`. clax's `CubicSpline.evaluate` CLAMPS `x_eval` (`clax/interpolation.py`, constant extrapolation) — hence the dedicated helper below.
4. Everything the loop reads is interpolated at `ktrue`: `Pnw`, `Pw` and the 32 loop channels at `4403-4437`, the 14 bias channels and `Pbin` at `5300-5314`. That is exactly the 47 keys of `chan` — no exceptions.
5. Legendre projection weights use the FIDUCIAL μ (`4470-4471`, `4534-4558`); `mu2t..mu8t` (`4474-4477`), `Sigmatot` (`4480`), `L2true/L4true` (bias block) use `mutrue`; `Exp` (`4481`) and the counterterm `k²` (`4498-4500`) use `ktrue`. `P1b1`/`P1` (`4491-4492`) feed only `P10b1/P10/P12`, which are freed unused at `5507` — do not port them.
6. The loop runs `index_j = Nside … Nmax − Nside` (interior of `kdisc`); clax evaluates all knots. The two differ only where `ktrue` leaves `[k[0], k[-1]]`, i.e. within a factor `max(hratio, 1/Dratio)` of the grid ends — outside `0.01 ≤ k ≤ 0.3 h/Mpc` for every grid in this campaign (`KMIN_H`/`KMAX_H` in `ept.py`).
7. classy computes `hratio`, `Dratio` itself from `Omfid` (`nonlinear_pt.c:1245-1296`, mirrored by B2's `clax.ap.ap_ratios`); the Python layer never sees them. The A3 AP file stores the values CLASS-PT actually used.

- [ ] **Step 1: Failing tests — spline helper, α=1 identity, independent AP-loop transcription, AP oracle files**

Append to the import block of `tests/test_ept_ap.py`:

```python
import json
import warnings

from scipy.interpolate import CubicSpline as _ScipyCubicSpline

from clax.ept import _channels_at, pk_mm_real, pk_gg_real, pk_gm_real
```

(`pk_gg_real`, `pk_gg_l0/l2/l4`, `pk_mm_l0/l2/l4` are already imported by B4's block; add only the names missing.) Append at the end of the file:

```python
# ---------------------------------------------------------------------------
# B5: in-loop AP remap (nonlinear_pt.c:4380-4558, 5300-5352 @ 09d5531a)
# ---------------------------------------------------------------------------
LEGACY_AP = os.path.join(ROOT, "reference_data", "classpt", "legacy_fiducial", "z0.380_ap_omfid0.31_m.npz")
LEGACY_NOAP = os.path.join(ROOT, "reference_data", "classpt", "legacy_fiducial", "z0.380_noap_m.npz")
AP_PAIRS = [(1.02, 0.97), (0.985, 1.03)]      # (hratio, Dratio): both sides of alpha = 1
SPECTRA_L02 = ["pk_mm_real", "pk_gg_real", "pk_gm_real", "pk_mm_l0", "pk_mm_l2", "pk_gg_l0", "pk_gg_l2"]
SPECTRA_L4 = ["pk_mm_l4", "pk_gg_l4"]
THRESH_L02, THRESH_L4 = 1e-2, 2e-2            # same numbers as tests/test_ept_accuracy.py; C4 ratchets


def test_channels_at_is_natural_spline_with_end_cubic_extrapolation():
    """_channels_at == scipy natural cubic spline in linear k, INCLUDING points
    outside the grid (scipy continues the end polynomial, as AP_BSEARCH_SETUP
    2383-2394 does); at the knots it returns the channel values exactly."""
    k, chan = _synthetic_channels(11, nk=48)
    kn = np.asarray(k)
    kq = np.stack([kn * 1.03, kn * 0.97, kn * 1.0])            # above, below, on the knots
    got = _channels_at(k, chan, jnp.asarray(kq))
    assert set(got) == set(chan) and all(v.shape == kq.shape for v in got.values())
    bad = []
    for name, y in chan.items():
        want = _ScipyCubicSpline(kn, np.asarray(y), bc_type="natural", extrapolate=True)(kq)
        r = _rel(got[name], want)
        if r > 1e-12:
            bad.append((name, r))
        assert np.array_equal(np.asarray(got[name][2]), np.asarray(y)), f"{name}: knot values not exact"
    assert not bad, f"spline mismatch vs scipy natural spline: {bad[:5]}"


def _classpt_ap_loop(chan, k, f, sig2, dsig2, hratio, Dratio):
    """Independent NumPy transcription of the CLASS-PT AP/IR loop, one Gauss
    node at a time: nonlinear_pt.c:4380-4558 (tree, 1-loop, ctr) and 5300-5352
    (bias block) @ 09d5531a. Written from the C code, not from clax/ept.py:
    reviewers check it against those lines. Splines: natural cubic in linear k
    over the whole grid (2355-2362), end cubic continued outside (2383-2394)."""
    k = np.asarray(k)
    spl = {n: _ScipyCubicSpline(k, np.asarray(v), bc_type="natural", extrapolate=True) for n, v in chan.items()}
    inv_Dr2 = 1.0 / Dratio ** 2                       # 4382 ap_inv_Dr2_
    hr2_minus_inv_Dr2 = hratio ** 2 - inv_Dr2         # 4383
    V = hratio / Dratio ** 2                          # 4384 ap_w_hr_Dr2_
    out = {key: np.zeros_like(k) for key in GL_KEYS_ALL}
    for mu, w in zip(_GAUSS_NODES, _GAUSS_WEIGHTS):
        mu, w = float(mu), float(w)
        ap_fac = np.sqrt(inv_Dr2 + hr2_minus_inv_Dr2 * mu ** 2)    # 4392
        mutrue = mu * hratio / ap_fac                              # 4393
        ktrue = k * ap_fac                                         # 4394
        s = {n: sp(ktrue) for n, sp in spl.items()}                # 4403-4437, 5300-5314
        L2 = 0.5 * (3 * mu ** 2 - 1)                               # 4470-4471: fiducial mu
        L4 = (35 * mu ** 4 - 30 * mu ** 2 + 3) / 8.0
        mu2t = mutrue ** 2                                         # 4474-4477
        mu4t, mu6t, mu8t = mu2t ** 2, mu2t ** 3, mu2t ** 4
        Sig = sig2 * (1 + f * mu2t * (2 + f)) + dsig2 * f ** 2 * mu2t * (mu2t - 1)   # 4480
        Exp = np.exp(-Sig * ktrue ** 2)                            # 4481
        Pnw, Pw = s["pk_nw"], s["pk_w"]
        p_tree = Pnw + (1 + Sig * ktrue ** 2) * Pw * Exp            # 4483
        P13ratio = 1 + (Pw / Pnw) * Exp                            # 4485
        W = {0: 0.5 * w * V, 2: 2.5 * w * L2 * V, 4: 4.5 * w * L4 * V}   # w*V at 4494-4500, (2l+1)/2 L_l at 4534-4558
        Pvv = ((s["P13_mu4_vv_nw"] * P13ratio + s["P22_mu4_vv_nw"] + (s["P22_mu4_vv_w"] + s["P13_mu4_vv_w"]) * Exp) * mu4t
               + (s["P13_mu6_nw"] * P13ratio + s["P22_mu6_vv_nw"] + (s["P22_mu6_vv_w"] + s["P13_mu6_w"]) * Exp) * mu6t
               + (s["P22_mu8_nw"] + s["P22_mu8_w"] * Exp) * mu8t)                                            # 4503
        Pdd = ((s["P22_mu0_dd_nw"] + s["P13_mu0_dd_nw"] * P13ratio + (s["P13_mu0_dd_w"] + s["P22_mu0_dd_w"]) * Exp)
               + (s["P22_mu2_dd_nw"] + s["P13_mu2_dd_nw"] * P13ratio + (s["P22_mu2_dd_w"] + s["P13_mu2_dd_w"]) * Exp) * mu2t
               + (s["P22_mu4_dd_nw"] + s["P22_mu4_dd_w"] * Exp) * mu4t)                                      # 4512
        Pvd = ((s["P13_mu2_vd_nw"] * P13ratio + s["P22_mu2_vd_nw"] + (s["P22_mu2_vd_w"] + s["P13_mu2_vd_w"]) * Exp) * mu2t
               + (s["P13_mu4_vd_nw"] * P13ratio + s["P22_mu4_vd_nw"] + (s["P22_mu4_vd_w"] + s["P13_mu4_vd_w"]) * Exp) * mu4t
               + (s["P22_mu6_vd_nw"] + s["P22_mu6_vd_w"] * Exp) * mu6t)                                      # 4521
        tree = {"vv": f ** 2 * mu4t * p_tree, "vd": 2.0 * f * mu2t * p_tree, "dd": p_tree}                    # 4494-4496
        loop = {"vv": Pvv, "vd": Pvd, "dd": Pdd}
        p_lin = Pnw + Pw * Exp
        ctr = {0: ktrue ** 2 * p_lin, 2: ktrue ** 2 * p_lin * f * mu2t, 4: ktrue ** 2 * p_lin * f ** 2 * mu4t}   # 4498-4500
        for ell, Wl in W.items():
            for ch in ("vv", "vd", "dd"):
                out[f"Pk_{ell}_{ch}"] += Wl * tree[ch]
                out[f"Pk_{ell}_{ch}1"] += Wl * loop[ch]
            out[f"Pk_ctr{ell}"] -= Wl * ctr[ell]                   # clax stores -P_CTR (ref §10)
        p_lo = p_lin / s["pk_disc"]                                # 5318
        IFG2 = p_lo * s["Pk_IFG2"]                                 # 5320
        out["Pk_IFG2_0b1"] += W[0] * IFG2                          # 5344-5346
        out["Pk_IFG2_0"] += W[0] * IFG2 * f * mu2t
        out["Pk_IFG2_2"] += W[2] * IFG2 * f * mu2t
        for n in ("Pk_Id2d2", "Pk_Id2G2", "Pk_IG2G2"):             # 5322-5324 (l=0 rows only)
            out[n] += W[0] * s[n]
        L2t = 0.5 * (3 * mu2t - 1)                                 # L2true / L4true
        L4t = (35 * mu4t - 30 * mu2t + 3) / 8.0
        for b in ("b1b2", "b1bG2"):                                # 5325-5326, 5350-5352
            val = s[f"Pk_0_{b}"] + L2t * s[f"Pk_2_{b}"]
            for ell, Wl in W.items():
                out[f"Pk_{ell}_{b}"] += Wl * val
        for b in ("b2", "bG2"):                                    # 5327-5328
            val = s[f"Pk_0_{b}"] + L2t * s[f"Pk_2_{b}"] + L4t * s[f"Pk_4_{b}"]
            for ell, Wl in W.items():
                out[f"Pk_{ell}_{b}"] += Wl * val
    return out


def test_classpt_ap_loop_transcription_reduces_to_alpha1_loop():
    """Self-check of the transcription: at (1, 1) it must equal B3's alpha=1
    transcription on the tree/loop keys (splines evaluated at the knots)."""
    k, chan = _synthetic_channels(5)
    f, sig2, dsig2 = 0.78, 30.0, 10.0
    ap = _classpt_ap_loop(chan, k, f, sig2, dsig2, 1.0, 1.0)
    old = _old_scalar_loop(chan, k, f, sig2, dsig2)
    worst = max((_rel(ap[key], old[key]), key) for key in old)
    assert worst[0] < 1e-12, f"AP transcription != alpha=1 transcription: {worst[1]} rel {worst[0]:.2e}"


def test_gl_multipoles_alpha1_is_bit_identical():
    """(hratio, Dratio) = (1, 1) must leave every key bit-identical to the
    B3 call: ap_fac == 1 exactly, so the spline is evaluated on its knots."""
    k, chan = _synthetic_channels(6)
    a = _gl_multipoles(chan, k, 0.7, 30.0, 10.0)
    b = _gl_multipoles(chan, k, 0.7, 30.0, 10.0, hratio=1.0, Dratio=1.0)
    notsame = [key for key in a if not np.array_equal(np.asarray(a[key]), np.asarray(b[key]))]
    assert not notsame, f"alpha=1 path not bit-identical for {notsame}"


@pytest.mark.parametrize("hratio,Dratio", AP_PAIRS)
@pytest.mark.parametrize("seed", [1, 2])
def test_gl_multipoles_reproduces_classpt_ap_loop(seed, hratio, Dratio):
    """All 39 keys vs the independent transcription at alpha != 1 (synthetic channels)."""
    k, chan = _synthetic_channels(seed, ir=True)
    f, sig2, dsig2 = 0.78, 30.0, 10.0
    new = _gl_multipoles(chan, k, f, sig2, dsig2, hratio=hratio, Dratio=Dratio)
    ref = _classpt_ap_loop(chan, k, f, sig2, dsig2, hratio, Dratio)
    worst = sorted(((_rel(new[key], ref[key]), key) for key in ref), reverse=True)
    assert worst[0][0] < 1e-10, f"AP loop drifts from CLASS-PT transcription: {worst[:3]}"


def test_gl_multipoles_ap_moves_the_multipoles():
    """Guard against a no-op wiring: at alpha != 1 the quadrupole tree changes at O(alpha-1)."""
    k, chan = _synthetic_channels(7)
    a = _gl_multipoles(chan, k, 0.7, 30.0, 10.0)
    b = _gl_multipoles(chan, k, 0.7, 30.0, 10.0, hratio=1.02, Dratio=0.97)
    assert _rel(b["Pk_2_dd"], a["Pk_2_dd"]) > 1e-3


def _nine_spectra(ept, bias):
    return {
        # classy pk_mm_real(cs) == clax pk_mm_real(cs0=cs)  (Part 1a assembly note)
        "pk_mm_real": pk_mm_real(ept, cs0=bias["cs"]),
        "pk_mm_l0": pk_mm_l0(ept, cs0=bias["cs0"]), "pk_mm_l2": pk_mm_l2(ept, cs2=bias["cs2"]), "pk_mm_l4": pk_mm_l4(ept, cs4=bias["cs4"]),
        "pk_gg_real": pk_gg_real(ept, bias["b1"], bias["b2"], bias["bG2"], bias["bGamma3"], cs=bias["cs"], cs0=bias["cs0"], Pshot=bias["Pshot"]),
        "pk_gm_real": pk_gm_real(ept, bias["b1"], bias["b2"], bias["bG2"], bias["bGamma3"], cs0=bias["cs0"], cs=bias["cs"]),
        "pk_gg_l0": pk_gg_l0(ept, bias["b1"], bias["b2"], bias["bG2"], bias["bGamma3"], cs0=bias["cs0"], Pshot=bias["Pshot"], b4=bias["b4"]),
        "pk_gg_l2": pk_gg_l2(ept, bias["b1"], bias["b2"], bias["bG2"], bias["bGamma3"], cs2=bias["cs2"], b4=bias["b4"]),
        "pk_gg_l4": pk_gg_l4(ept, bias["b1"], bias["b2"], bias["bG2"], bias["bGamma3"], cs4=bias["cs4"], b4=bias["b4"]),
    }


def _oracle_check(path, use_stored_ap):
    """compute_ept at the file's inputs (+ its hratio/Dratio when use_stored_ap)
    vs the nine CLASS-PT spectra stored by Part 1a A3, on 0.01 <= k <= 0.3."""
    d = np.load(path)
    kh = np.asarray(d["k_h"]); h, fz = float(d["h"]), float(d["fz"])
    hr, Dr = (float(d["hratio"]), float(d["Dratio"])) if use_stored_ap else (1.0, 1.0)
    bias = {"b1": 1.0, "b2": 0.0, "bG2": 0.0, "bGamma3": 0.0, "cs": 0.0, "cs0": 0.0, "cs2": 0.0, "cs4": 0.0, "Pshot": 0.0, "b4": 0.0}
    bias.update(json.loads(str(d["bias_json"])))
    e = compute_ept(jnp.asarray(d["pk_lin"]), jnp.asarray(kh), h=h, f=fz, prec=EPTPrecisionParams(), hratio=hr, Dratio=Dr)
    got = _nine_spectra(e, bias)
    sel = (kh >= 0.01) & (kh <= 0.3)
    rows = {}
    for name in SPECTRA_L02 + SPECTRA_L4:
        ref = np.asarray(d[name])[sel]
        r = np.abs(np.asarray(got[name])[sel] - ref) / np.maximum(np.abs(ref), 1e-300)
        rows[name] = (float(r.max()), float(kh[sel][r.argmax()]))
    # AP-sensitive leaf sums straight from classy.pyx:4900/4907 (rows 18+24, 28, 29)
    pm = d["pk_mult"]
    for leaf, ref in (("Pk_2_vv+Pk_2_vv1", (pm[18] + pm[24]) * h ** 3), ("Pk_4_vd+Pk_4_vd1", pm[28] * h ** 3), ("Pk_4_dd+Pk_4_dd1", pm[29] * h ** 3)):
        a, b = leaf.split("+")
        val = np.asarray(getattr(e, a) + getattr(e, b))[sel]
        r = np.abs(val - ref[sel]) / np.maximum(np.abs(ref[sel]), 1e-300)
        rows[leaf] = (float(r.max()), float(kh[sel][r.argmax()]))
    return rows, (hr, Dr)


@pytest.mark.skipif(not os.path.isfile(LEGACY_NOAP), reason="Part 1a A3 noap file absent")
def test_compute_ept_alpha1_matches_classpt_noap_file():
    """True alpha=1 oracle (AP_effect=No in CLASS-PT): all nine spectra + three leaf sums."""
    rows, _ = _oracle_check(LEGACY_NOAP, use_stored_ap=False)
    bad = [(n, f"{r:.3e}@k={kk:.3f}") for n, (r, kk) in rows.items() if r > (THRESH_L4 if (n in SPECTRA_L4 or n.startswith("Pk_4")) else THRESH_L02)]
    assert not bad, f"noap oracle: {bad}"
    print("noap max-rel:", {n: f"{r:.2e}" for n, (r, _) in rows.items()})


@pytest.mark.skipif(not os.path.isfile(LEGACY_AP), reason="Part 1a A3 AP file absent")
def test_compute_ept_with_ap_matches_classpt_ap_file():
    """AP oracle: CLASS-PT ran with AP_effect=Yes, Omfid=0.31 and stored the
    (hratio, Dratio) it used; compute_ept with the same pair must match all
    nine spectra. If this fails while the noap test passes, first rerun with
    (1/hratio, 1/Dratio): a pass there means B2's ratio convention is inverted
    relative to nonlinear_pt.c:1245-1296 — fix clax/ap.py, not this file."""
    rows, (hr, Dr) = _oracle_check(LEGACY_AP, use_stored_ap=True)
    assert abs(hr - 1.0) > 1e-4 or abs(Dr - 1.0) > 1e-4, "AP file has alpha == 1: A3 wrote the wrong file"
    bad = [(n, f"{r:.3e}@k={kk:.3f}") for n, (r, kk) in rows.items() if r > (THRESH_L4 if (n in SPECTRA_L4 or n.startswith("Pk_4")) else THRESH_L02)]
    assert not bad, f"AP oracle at (hratio, Dratio)=({hr:.5f}, {Dr:.5f}): {bad}"
    print(f"AP ({hr:.5f}, {Dr:.5f}) max-rel:", {n: f"{r:.2e}" for n, (r, _) in rows.items()})


@pytest.mark.skipif(not os.path.isfile(LEGACY_AP), reason="Part 1a A3 AP file absent")
def test_ap_off_against_ap_file_is_worse_than_ap_on():
    """The remap must explain the AP-on reference: alpha=1 against the AP file
    is worse than alpha=stored on at least the quadrupole leaf sums."""
    on, _ = _oracle_check(LEGACY_AP, use_stored_ap=True)
    off, _ = _oracle_check(LEGACY_AP, use_stored_ap=False)
    assert on["Pk_2_vv+Pk_2_vv1"][0] < off["Pk_2_vv+Pk_2_vv1"][0]
    assert on["pk_mm_l2"][0] < off["pk_mm_l2"][0]
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /home/n2minh/clax-ptval && JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2" PYTHONPATH=/home/n2minh/clax-ptval timeout 120 /home/n2minh/micromamba/envs/clax/bin/python -m pytest tests/test_ept_ap.py -q -p no:cacheprovider -k "channels_at or classpt_ap_loop or alpha1_is_bit or reproduces_classpt_ap or ap_moves or noap_file or ap_file or ap_off" 2>&1 | tail -n 5
```
Expected: `ImportError: cannot import name '_channels_at' from 'clax.ept'`. Temporarily comment that import out and rerun: `transcription_reduces_to_alpha1_loop` PASSES (it needs no new code — if it fails, the transcription is wrong; fix it against the C lines before anything else), the `_gl_multipoles(..., hratio=…)` tests fail with `TypeError: unexpected keyword argument 'hratio'`, `compute_ept(..., hratio=…)` likewise. Restore the import.

- [ ] **Step 3: `_channels_at` — natural spline of every channel at `ktrue`, end cubic continued**

Add to the import block of `clax/ept.py` (after `import jax.numpy as jnp`): `from clax.interpolation import _compute_natural_spline_coeffs`. Insert immediately above `def _gl_multipoles(`:

```python
def _channels_at(k, chan: dict, kq) -> dict:
    """Every channel in `chan` (knots `k`, shape (Nk,)) evaluated at `kq`
    (any shape, here (40, Nk) = ktrue) with a NATURAL cubic spline in LINEAR
    k over the whole grid — CLASS-PT's AP_SPLINE_SETUP
    (nonlinear_pt.c:2355-2362, `array_spline_table_columns(kdisc, …,
    _SPLINE_NATURAL_)`). Outside [k[0], k[-1]] the end interval's cubic is
    continued, as AP_BSEARCH_SETUP/AP_INTERP_FAST do (2372-2394: the interval
    is clamped, the weights a, b are not). At kq == k the knot values are
    returned exactly (a = 1, b = 0), which keeps alpha = 1 bit-identical.
    Differentiable in kq (and through it in hratio, Dratio)."""
    names = tuple(chan)
    Y = jnp.stack([chan[n] for n in names])                                   # (C, Nk)
    d2 = jax.vmap(_compute_natural_spline_coeffs, in_axes=(None, 0))(k, Y)   # (C, Nk)
    n = k.shape[0]
    idx = jnp.clip(jnp.searchsorted(k, kq, side="right") - 1, 0, n - 2)      # 2384-2388 bisection
    hh = k[idx + 1] - k[idx]
    b = (kq - k[idx]) / hh                                                    # 2390-2391, unclamped
    a = 1.0 - b
    a3a, b3b, h2_6 = a ** 3 - a, b ** 3 - b, hh ** 2 / 6.0                    # 2392-2394

    def one(y, dd):                                                           # 2373-2374 AP_INTERP_FAST
        return a * y[idx] + b * y[idx + 1] + (a3a * dd[idx] + b3b * dd[idx + 1]) * h2_6

    vals = jax.vmap(one)(Y, d2)                                               # (C, *kq.shape)
    return dict(zip(names, vals))
```

Run Step 2's command: `test_channels_at_is_natural_spline_with_end_cubic_extrapolation` passes. If the extrapolated rows fail while the knot row passes, the bisection is clamping `b` — compare with scipy's `extrapolate=True` and re-read 2390. If everything fails at ~1e-8, `_compute_natural_spline_coeffs` returned float32: confirm `jax_enable_x64` is on in the test process (it is set at the top of the file) and that `k` is float64.

- [ ] **Step 4: thread `hratio`, `Dratio` into `_gl_multipoles`**

Edits to `_gl_multipoles` (B3 Step 3 code; anchor each on the quoted line):

1. Signature: `def _gl_multipoles(chan: dict, k, f, sigma2_bao, delta_sigma2_bao, hratio=1.0, Dratio=1.0) -> dict:`.
2. Docstring: replace `at (hratio, Dratio) = (1, 1): axis 0 is` with `with the AP remap of 4382-4394 (hratio = H_true/H_fid, Dratio = D_A,true/D_A,fid, ref §3; both 1 ⇒ CLASS-PT's alpha=1 branch 4396-4397 bit-for-bit): axis 0 is`, and `evaluated at the "true" μ, which here equals the fiducial one.` with `evaluated at the "true" μ; every channel is splined at the "true" k (_channels_at, 4403-4437, 5300-5314); the volume factor V = hratio/Dratio² (4384) is folded into W0/W2/W4.`
3. Replace the four lines from `kk = k[None, :]` through `mu4t, mu6t, mu8t = …` with

```python
    hratio = jnp.asarray(hratio, dtype=k.dtype)
    Dratio = jnp.asarray(Dratio, dtype=k.dtype)
    inv_D2 = 1.0 / Dratio ** 2                                        # 4382 ap_inv_Dr2_
    ap_fac = jnp.sqrt(inv_D2 + (hratio ** 2 - inv_D2) * mu ** 2)[:, None]   # 4392  (40, 1)
    mu_k = mu[:, None] * hratio / ap_fac                              # 4393 mutrue  (40, 1)
    kk = k[None, :] * ap_fac                                          # 4394 ktrue   (40, Nk)
    V = hratio / Dratio ** 2                                          # 4384 ap_w_hr_Dr2_
    mu2t = mu_k ** 2                                                  # 4474-4477
    mu4t, mu6t, mu8t = mu2t ** 2, mu2t ** 3, mu2t ** 4
```
4. `W0, W2, W4 = 0.5 * w, 2.5 * w * L2, 4.5 * w * L4` → `W0, W2, W4 = 0.5 * w * V, 2.5 * w * L2 * V, 4.5 * w * L4 * V` (comment: `# 4534-4558 projection weights × V (4494-4500, 5320-5328)`).
5. Replace `def c(name): return chan[name][None, :]` (and its `# B5:` comment) with

```python
    at = _channels_at(k, chan, kk)                       # 4403-4437, 5300-5314: all 47 channels at ktrue

    def c(name):
        return at[name]                                  # (40, Nk)
```
6. Delete the two `# B5:` trailing comments that remain (`kk`, `mu_k` lines were replaced; check with `grep -n "B5:" clax/ept.py` → no output).

Nothing else in the body changes: `Sig`, `Exp`, `P13ratio`, `p_tree`, `p_lin`, the three loop brackets, the counterterms (`kk ** 2` is now `ktrue²`, 4498-4500), `p_lo`, `L2t/L4t` all already read `kk`/`mu_k`/`c(...)`.

- [ ] **Step 5: thread the kwargs through `_compute_bias_spectra` and `compute_ept`**

1. `_compute_bias_spectra` signature (`grep -n "^def _compute_bias_spectra"`): append `hratio=1.0, Dratio=1.0` after `pk_w=None`. Its docstring gets one line: `hratio, Dratio: AP ratios (ref §3), passed to _gl_multipoles; 1, 1 = no remap.`
2. `gl = _gl_multipoles(chan, k, f, _sig2_bao, _delta_sig2)` → `gl = _gl_multipoles(chan, k, f, _sig2_bao, _delta_sig2, hratio=hratio, Dratio=Dratio)`.
3. `compute_ept` signature (`grep -n "^def compute_ept"`): append `hratio=1.0, Dratio=1.0` after `rs_h: float = 99.0`, typed `hratio: float | Float[Array, ""] = 1.0`. Docstring `Args:` block gets

```
        hratio, Dratio: Alcock–Paczynski ratios H_true(z)/H_fid(z) and
            D_A,true(z)/D_A,fid(z) (CLASS-PT nonlinear_pt.c:1245-1296; see
            clax.ap.ap_ratios). Default (1, 1) = no AP remap. Applied inside the
            Gauss–Legendre μ-loop exactly as CLASS-PT does (4382-4394); the
            real-space leaves (Pk_tree, Pk_loop, Pk_ctr, Pk_Id2, …) are not
            remapped, matching classy.
```
4. The `bias = _compute_bias_spectra(...)` call (`grep -n "bias = _compute_bias_spectra("`): append `hratio=hratio, Dratio=Dratio,` after the `pk_w=...` argument.
5. `compute_ept_from_clax` (`grep -n "^def compute_ept_from_clax"`) is left alone — Part 2 C0 wires `clax.ap.ap_ratios` through it.

Run the full file:

```bash
cd /home/n2minh/clax-ptval && JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2" PYTHONPATH=/home/n2minh/clax-ptval timeout 120 /home/n2minh/micromamba/envs/clax/bin/python -m pytest tests/test_ept_ap.py -q -p no:cacheprovider 2>&1 | tail -n 8
```
Expected: everything passes, `EXEMPT` still `{}` and `test_alpha1_matches_frozen_baseline` untouched — the α=1 path is bit-identical, so **no refreeze in B5**. A baseline drift here means `ap_fac ≠ 1` at (1, 1) (dtype promotion: `hratio` arrived as float32 — the `jnp.asarray(..., dtype=k.dtype)` lines handle it) or a knot lookup off by one in `_channels_at` (the knot test in Step 3 would have caught it). A failure of `reproduces_classpt_ap_loop` with the α=1 tests passing is localized: print the worst three keys — tree/loop keys → `ap_fac`/`mutrue`/`ktrue` (4392-4394) or `V`; ctr keys → `kk ** 2` must be `ktrue²`; bias keys → `L2t/L4t` from `mutrue`, `p_lo` from `p_lin/Pbin` both at `ktrue`. The two oracle tests skip until A3 has produced the files; if they are present and `..._ap_file` fails while `..._noap_file` passes, follow the inversion check in its docstring before touching anything else.

- [ ] **Step 6: `tests/test_ept_accuracy.py` runs at the reference's AP ratios**

The legacy reference (`classpt_z0.38_fullrange.npz`) was generated with AP on (Ωm ≠ Omfid, α − 1 ≈ 2e-3, Part 0 findings); `ept_result` has always compared an α=1 clax against it. Above the fixtures add

```python
# Part 1a A3 regenerated the legacy point with AP on and stored the ratios CLASS-PT used.
LEGACY_AP_PATH = os.path.join(
    os.path.dirname(__file__), "..", "reference_data", "classpt", "legacy_fiducial",
    "z0.380_ap_omfid0.31_m.npz"
)


def _legacy_ap_ratios(h):
    """(hratio, Dratio) CLASS-PT used for the legacy point; (1, 1) with a warning if A3 has not run."""
    if not os.path.isfile(LEGACY_AP_PATH):
        warnings.warn(f"{LEGACY_AP_PATH} absent: comparing an alpha=1 clax against the AP-on legacy reference")
        return 1.0, 1.0
    ap = np.load(LEGACY_AP_PATH)
    assert abs(float(ap["h"]) - h) < 1e-12, "legacy AP file is not the legacy cosmology"
    return float(ap["hratio"]), float(ap["Dratio"])
```
(`import warnings` at the top.) In `ept_result`, before the `compute_ept(` call: `hratio, Dratio = _legacy_ap_ratios(h)`; add `hratio=hratio, Dratio=Dratio` to the call. Run the before/after table:

```bash
cd /home/n2minh/clax-ptval && JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2" PYTHONPATH=/home/n2minh/clax-ptval timeout 120 /home/n2minh/micromamba/envs/clax/bin/python - <<'EOF'
import numpy as np, jax.numpy as jnp, os, json
from clax.ept import compute_ept, EPTPrecisionParams, pk_mm_real, pk_mm_l0, pk_mm_l2, pk_mm_l4, pk_gg_real, pk_gm_real, pk_gg_l0, pk_gg_l2, pk_gg_l4
L = np.load("reference_data/classpt_z0.38_fullrange.npz", allow_pickle=True); h = float(L["h"]); kh = L["k_h"]
A = np.load("reference_data/classpt/legacy_fiducial/z0.380_ap_omfid0.31_m.npz"); hr, Dr = float(A["hratio"]), float(A["Dratio"])
bias = {k[5:]: float(L[k]) for k in L.files if k.startswith("bias_")}
b1, b2, bG2, bG3 = bias["b1"], bias["b2"], bias["bG2"], bias["bGamma3"]
sel = (kh >= 0.01) & (kh <= 0.3)
for tag, (a, b) in (("alpha=1", (1.0, 1.0)), (f"AP({hr:.5f},{Dr:.5f})", (hr, Dr))):
    e = compute_ept(jnp.asarray(L["pk_lin"]), jnp.asarray(kh), h=h, f=float(L["fz"]), prec=EPTPrecisionParams(), hratio=a, Dratio=b)
    got = {"pk_mm_real": pk_mm_real(e), "pk_mm_l0": pk_mm_l0(e), "pk_mm_l2": pk_mm_l2(e), "pk_mm_l4": pk_mm_l4(e),
           "pk_gg_real": pk_gg_real(e, b1, b2, bG2, bG3), "pk_mg_real": pk_gm_real(e, b1, b2, bG2, bG3),
           "pk_gg_l0": pk_gg_l0(e, b1, b2, bG2, bG3), "pk_gg_l2": pk_gg_l2(e, b1, b2, bG2, bG3), "pk_gg_l4": pk_gg_l4(e, b1, b2, bG2, bG3)}
    print(tag, {n: f"{np.max(np.abs(np.asarray(v)[sel] - L[n][sel]) / np.abs(L[n][sel])):.2e}" for n, v in got.items()})
EOF
```
(`bias_*` keys and the `cs*` conventions are whatever `ept_result` already uses — copy its bias extraction verbatim if it differs from the four-line version above.) Expected: the AP row is ≤ the α=1 row on every multipole, most visibly `pk_mm_l2/l4`, `pk_gg_l2/l4`; the real-space rows are unchanged (no remap). Put both rows in the commit body; `THRESH_*` in `test_ept_accuracy.py` stay as they are — C4 ratchets from the Part 2 sweep.

- [ ] **Step 7: local gate, cluster gate**

```bash
cd /home/n2minh/clax-ptval && JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2" PYTHONPATH=/home/n2minh/clax-ptval timeout 120 /home/n2minh/micromamba/envs/clax/bin/python -m pytest tests/test_ept_ap.py tests/test_ept.py -q -p no:cacheprovider 2>&1 | tail -n 5
cd /home/n2minh/clax-ptval && JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2" PYTHONPATH=/home/n2minh/clax-ptval timeout 120 /home/n2minh/micromamba/envs/clax/bin/python -m pytest tests/test_ept_assembly.py tests/test_ept_accuracy.py -q -p no:cacheprovider 2>&1 | tail -n 5
```
Expected: all pass (the `warnings.warn` line appears only if A3's file is absent — say which in the commit). Cluster gate: `sbatch slurm/ptval-fast-suite.sbatch && sbatch slurm/ptval-track-b-full.sbatch`; both logs end in `PASS` (`tests/test_ept_gradients.py` on the GPU now differentiates through `_channels_at` at α=1; a NaN there means a `0/0` in `b = (kq − k[idx])/hh` on a degenerate grid — `hh > 0` on every geomspace grid, so report rather than patch).

- [ ] **Step 8: Commit**

Message file `.../scratchpad/commit-B5.txt`:

```
feat(ept): in-loop Alcock-Paczynski remap (hratio, Dratio) in _gl_multipoles

compute_ept(..., hratio=1.0, Dratio=1.0) -> _compute_bias_spectra ->
_gl_multipoles: per Gauss node ap_fac = sqrt(1/Dr² + (hr² − 1/Dr²) μ²),
mutrue = μ hr/ap_fac, ktrue = k ap_fac (nonlinear_pt.c:4382-4394); all 47
channels splined at ktrue with a natural cubic in linear k, end cubic
continued outside the grid (2355-2394, _channels_at); Legendre weights at
the fiducial μ, Σ_tot/L2true/L4true at mutrue, Exp and k² at ktrue, volume
factor hr/Dr² on every term (4384). (1, 1) is bit-identical to the alpha=1
path: baseline unchanged, no refreeze.

Verified: _gl_multipoles vs an independent NumPy transcription of
4380-4558 + 5300-5352 at (1.02, 0.97) and (0.985, 1.03), 39 keys < 1e-10;
_channels_at vs scipy natural CubicSpline incl. extrapolation < 1e-12.
Oracle (Part 1a A3, z=0.38): noap file <max-rel table>; AP file at
(hratio, Dratio) = (<hr>, <Dr>): <max-rel table>.
test_ept_accuracy vs legacy, alpha=1 -> AP: <two rows from Step 6>.
Cluster gate: ptval-fast-suite <id> PASS, ptval-track-b-full <id> PASS.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YBYncmhA8JrJQVNw4pVzmb
```

```bash
cd /home/n2minh/clax-ptval && git add clax/ept.py tests/test_ept_ap.py tests/test_ept_accuracy.py && git commit -F /tmp/claude-17163/-home-n2minh-clax/85880c7b-9aa1-44ed-b5d2-49320fe59b2a/scratchpad/commit-B5.txt
```

---

### Task B6: gradients through the AP remap — AD vs finite differences, grad vs jvp, jit

Requires B5. The AP ratios are the quantities a full-shape fit differentiates through (H(z), D_A(z) enter only via `hratio`, `Dratio` once Part 2 C0 wires `clax.ap.ap_ratios`), so their gradients get the same bottom-up treatment CLAUDE.md prescribes: a cheap functional at the μ-loop level, then the whole `compute_ept`. Nothing in `clax/ept.py` should need to change; if a test fails, the failure is a finding (see Step 3), not a tolerance to loosen.

**Files:**
- Modify: `tests/test_ept_ap.py` (append)

**Interfaces:**
- Consumes: B5's `_gl_multipoles(chan, k, f, sigma2_bao, delta_sigma2_bao, hratio=1.0, Dratio=1.0)` and `compute_ept(..., hratio=1.0, Dratio=1.0)`; B3's `_synthetic_channels`, `_rel`; B1's `legacy` fixture; `pk_mm_l2`, `pk_gg_l2` (already imported).
- Produces: nothing new in `clax/`. Test names below are referenced by the Track B self-check.

**Multi-cosmology rule — exemption statement (put it in the module docstring of the appended block, verbatim):** *"B6 tests are cosmology-independent numerics of the μ-loop and its spline remap: they need no reference data, and the cosmology enters only through the arrays `(pk_lin, f, h)` and the scalars `(hratio, Dratio)`. Three input variants (legacy `pk_lin`, a tilted copy, a rescaled copy with different `f`, `hratio`, `Dratio`) stand in for a grid; the pipeline-level gradient sweep over the real cosmology grid is Part 2 C2's `test_e2e_gradient_finite`."*

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ept_ap.py`:

```python
# ---------------------------------------------------------------------------
# B6: gradients through (hratio, Dratio)
# ---------------------------------------------------------------------------
# B6 tests are cosmology-independent numerics of the μ-loop and its spline
# remap: they need no reference data, and the cosmology enters only through the
# arrays (pk_lin, f, h) and the scalars (hratio, Dratio). Three input variants
# (legacy pk_lin, a tilted copy, a rescaled copy with different f, hratio,
# Dratio) stand in for a grid; the pipeline-level gradient sweep over the real
# cosmology grid is Part 2 C2's test_e2e_gradient_finite.

AP_GRAD_POINTS = [(1.0, 1.0), (1.02, 0.97), (0.985, 1.03)]     # alpha = 1 sits on every spline knot


def _l2_functional(chan, k, f, sig2, dsig2):
    """Scalar F(hratio, Dratio) = Σ_k k · P_2^{mm}(k) over the loop-level quadrupole
    (tree + loop, vv + vd + dd) — every AP-dependent piece of _gl_multipoles feeds it."""
    keys = ("Pk_2_vv", "Pk_2_vv1", "Pk_2_vd", "Pk_2_vd1", "Pk_2_dd", "Pk_2_dd1")

    def F(hr, Dr):
        gl = _gl_multipoles(chan, k, f, sig2, dsig2, hratio=hr, Dratio=Dr)
        return jnp.sum(k * sum(gl[key] for key in keys))

    return F


def _central_fd(F, hr, Dr, step):
    d_hr = (F(hr + step, Dr) - F(hr - step, Dr)) / (2 * step)
    d_Dr = (F(hr, Dr + step) - F(hr, Dr - step)) / (2 * step)
    return float(d_hr), float(d_Dr)


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("hratio,Dratio", AP_GRAD_POINTS)
def test_gl_multipoles_ap_grad_matches_fd(seed, hratio, Dratio):
    """Reverse-mode d F/d(hratio, Dratio) against central differences. The
    natural spline is C², so F is C¹ across knots and the O(step²) FD error
    is ~1e-8 relative; 1e-5 leaves room for round-off in the 40×Nk sums."""
    k, chan = _synthetic_channels(seed, ir=True)
    F = _l2_functional(chan, k, 0.78, 30.0, 10.0)
    g_hr, g_Dr = jax.grad(F, argnums=(0, 1))(hratio, Dratio)
    fd_hr, fd_Dr = _central_fd(F, hratio, Dratio, 1e-4)
    assert np.isfinite([g_hr, g_Dr]).all(), (g_hr, g_Dr)
    assert abs(float(g_hr) - fd_hr) < 1e-5 * abs(fd_hr), ("hratio", float(g_hr), fd_hr)
    assert abs(float(g_Dr) - fd_Dr) < 1e-5 * abs(fd_Dr), ("Dratio", float(g_Dr), fd_Dr)


@pytest.mark.parametrize("hratio,Dratio", AP_GRAD_POINTS)
def test_gl_multipoles_ap_grad_matches_jvp(hratio, Dratio):
    """Forward mode (jvp) and reverse mode (grad) agree to round-off; a
    disagreement pins a custom_vjp/where-gradient bug in the remap."""
    k, chan = _synthetic_channels(4, ir=True)
    F = _l2_functional(chan, k, 0.78, 30.0, 10.0)
    g = jax.grad(F, argnums=(0, 1))(hratio, Dratio)
    for i, e in enumerate(((1.0, 0.0), (0.0, 1.0))):
        _, t = jax.jvp(F, (hratio, Dratio), e)
        assert abs(float(t) - float(g[i])) < 1e-10 * max(abs(float(g[i])), 1e-30), (i, float(t), float(g[i]))


def test_gl_multipoles_ap_is_jittable_with_traced_ratios():
    """(hratio, Dratio) traced under jit reproduces the eager result to round-off
    (XLA fusion may reorder the k-sum, so 1e-12 rather than bit-equality); the
    searchsorted/clip path must not depend on concrete values."""
    k, chan = _synthetic_channels(5, ir=True)
    F = _l2_functional(chan, k, 0.78, 30.0, 10.0)
    eager = float(F(1.02, 0.97))
    jitted = float(jax.jit(F)(jnp.float64(1.02), jnp.float64(0.97)))
    assert abs(eager - jitted) < 1e-12 * abs(eager), (eager, jitted)


_PK_VARIANTS = {
    # name: (pk_lin transform, f, hratio, Dratio)
    "legacy":   (lambda kh, pk: pk,                         None, 1.02,  0.97),
    "tilted":   (lambda kh, pk: pk * (kh / 0.1) ** 0.05,    0.65, 0.985, 1.03),
    "rescaled": (lambda kh, pk: 0.8 * pk,                   0.90, 1.0,   1.0),
}


@pytest.mark.slow
@pytest.mark.parametrize("variant", list(_PK_VARIANTS))
def test_compute_ept_ap_grad_matches_fd(legacy, variant):
    """Whole pipeline: d/d(hratio, Dratio) of Σ_k k·P_2^{gg}(k) (b2, bG2, cs2, b4
    all nonzero) by jax.grad vs central FD with step 1e-3 — rel < 1e-3
    (CLAUDE.md gradient target 1%). Also asserts the jit of the traced-ratio
    pipeline compiles and matches eager to 1e-12."""
    transform, f_override, hr0, Dr0 = _PK_VARIANTS[variant]
    kh = jnp.asarray(legacy["k_h"]); pk = transform(kh, jnp.asarray(legacy["pk_lin"]))
    h = float(legacy["h"]); f = float(legacy["fz"]) if f_override is None else f_override
    b1, b2, bG2, bG3, cs2, b4 = 1.9, 0.7, -0.4, 0.3, 20.0, 400.0

    def F(hr, Dr):
        e = compute_ept(pk, kh, h=h, f=f, prec=EPTPrecisionParams(), hratio=hr, Dratio=Dr)
        return jnp.sum(kh * pk_gg_l2(e, b1, b2, bG2, bG3, cs2=cs2, b4=b4))

    g_hr, g_Dr = jax.grad(F, argnums=(0, 1))(hr0, Dr0)
    fd_hr, fd_Dr = _central_fd(F, hr0, Dr0, 1e-3)
    assert np.isfinite([g_hr, g_Dr]).all(), (variant, g_hr, g_Dr)
    assert abs(float(g_hr) - fd_hr) < 1e-3 * abs(fd_hr), (variant, "hratio", float(g_hr), fd_hr)
    assert abs(float(g_Dr) - fd_Dr) < 1e-3 * abs(fd_Dr), (variant, "Dratio", float(g_Dr), fd_Dr)
    eager = float(F(hr0, Dr0)); jitted = float(jax.jit(F)(jnp.float64(hr0), jnp.float64(Dr0)))
    assert abs(eager - jitted) < 1e-10 * abs(eager), (variant, eager, jitted)
```

- [ ] **Step 2: Run to verify the state**

```bash
cd /home/n2minh/clax-ptval && JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2" PYTHONPATH=/home/n2minh/clax-ptval timeout 120 /home/n2minh/micromamba/envs/clax/bin/python -m pytest tests/test_ept_ap.py -q -p no:cacheprovider --fast -k "ap_grad or traced_ratios" 2>&1 | tail -n 12
```
These are tests of code that already exists (B5), so "fail first" is not available; what is available is proof that each test can fail. Expected: the three μ-loop tests pass (`3×3 + 3 + 1 = 13 passed`), `test_compute_ept_ap_grad_matches_fd` is skipped by `--fast`. Then make each one fail once, on purpose, and revert:

1. In `_gl_multipoles` temporarily replace `ap_fac = jnp.sqrt(...)` with `ap_fac = jax.lax.stop_gradient(jnp.sqrt(...))` — `..._matches_fd` must fail at both α≠1 points and at α=1 (the gradient at α=1 is nonzero: `∂ktrue/∂hratio = k μ² ≠ 0`), `..._matches_jvp` passes (both modes see the same stop). Revert.
2. Temporarily replace `b = (kq - k[idx]) / hh` with `b = jnp.clip((kq - k[idx]) / hh, 0.0, 1.0)` in `_channels_at` — `..._matches_fd` fails at the α≠1 points on `seed`s whose extrapolated nodes carry weight, passes at α=1; `test_channels_at_is_natural_spline_with_end_cubic_extrapolation` (B5) fails. Revert.

`git diff --stat clax/ept.py` must be empty after the reverts.

- [ ] **Step 3: Full-pipeline test on the cluster; interpret a failure**

`sbatch slurm/ptval-track-b-full.sbatch` runs `tests/test_ept_ap.py` without `--fast`, so the three `test_compute_ept_ap_grad_matches_fd[...]` cases run on the V100 (each is 1 AD + 4 FD + 1 jit `compute_ept` calls; ≈1 min total). Expected log tail: `PASS`. If a case fails:

- NaN gradient with finite values → a `jnp.where` whose untaken branch divides by zero under the remap (`P13ratio`'s `pk_w/pk_nw` at a `pk_nw = 0` knot, or `p_lo = p_lin/Pbin` at `Pbin = 0`): the fix is the standard `jnp.where(safe, x/jnp.where(safe, y, 1.0), 1.0)` double-where at that line, cited to the CLASS-PT line the guard mirrors — and the fix goes in a commit of its own, with the NaN reproduced in a new test first.
- Finite but > 1e-3 off FD on `tilted`/`rescaled` only → the FD step straddles a knot where the extrapolated cubic changes curvature fast (only possible for `ktrue` beyond `k[-1]` at the highest μ nodes): rerun the FD at step 3e-4 by hand; if the gap closes, lower the step in the test to 3e-4 with a comment; if it does not, it is a real AD/FD gap — report it, do not touch the tolerance.

- [ ] **Step 4: Local gate, commit**

Local gate = B3 Step 8's two pytest commands (`--fast` is not part of them; the slow test is deselected there by adding `-m "not slow"` to the first command only for this task). Commit (`commit-B6.txt`):

```
test(ept): gradients through the AP remap — AD vs FD, grad vs jvp, jit

d/d(hratio, Dratio) of Σ_k k P_2(k): reverse-mode vs central FD < 1e-5
on the μ-loop (3 synthetic channel sets × {alpha=1, (1.02, 0.97),
(0.985, 1.03)}), grad == jvp to 1e-10, traced ratios under jit bit-equal
to eager; whole compute_ept -> pk_gg_l2 (all biases nonzero) < 1e-3 vs
FD on legacy/tilted/rescaled pk_lin (slow; V100 job <id> PASS).
Each test shown to fail under a deliberate stop_gradient / clamped-b
patch, then reverted (clax/ept.py unchanged).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YBYncmhA8JrJQVNw4pVzmb
```

```bash
cd /home/n2minh/clax-ptval && git add tests/test_ept_ap.py && git commit -F /tmp/claude-17163/-home-n2minh-clax/85880c7b-9aa1-44ed-b5d2-49320fe59b2a/scratchpad/commit-B6.txt
```

---

### Task B7: clax accessors ↔ Part 1a's classy twin (`scripts/classpt_assembly.py`)

Requires B4 and Part 1a A3. Two independent transcriptions of `classy.pyx:4795-4915` now exist: the NumPy twin (A3, asserted against classy itself on every generated file) and clax's accessors (B4, asserted against the ref §11 algebra). B7 closes the triangle: build a classy-convention `pm` array from clax's own leaves (the inverse of ref §10), assemble it with the twin, and demand agreement with the accessors to 1e-12 at biases where every term is live. A mismatch is a transcription error in one of the two — the commit that fixes it must say which, with the classy line.

**Files:**
- Modify: `tests/test_ept_assembly.py` — extend the `assembly_setup` fixture's return dict (add `"h"`, `"fz"`) and the import block (`:30-33`); append the tests. Keep the three existing tests as they are.

**Interfaces:**
- Consumes: `scripts.classpt_assembly.assemble_from_pm(pm, h, fz, kh, bias, Pd2d2_0) -> dict` (keys `pk_mm_real, pk_gg_real, pk_gm_real, pk_mm_l0, pk_mm_l2, pk_mm_l4, pk_gg_l0, pk_gg_l2, pk_gg_l4`), `scripts.classpt_assembly.pd2d2_0(pk_lin_h, kh) -> float`; clax `_pd2d2_0`, the nine accessors; `EPTComponents` leaves (ref §15).
- Produces: `tests/test_ept_assembly.py::_pm_from_leaves(e, h) -> np.ndarray (48, Nk)` — Part 2 C1 reuses it to compare stored `pk_mult` rows against clax leaves row-by-row.

Facts the transcription depends on (ref §10, §15):
1. `pm[row] = leaf / h³` for every Mpc³ row, except `pm[1] = −Pk_Id2d2/h³` (classy stores Id2d2 negated; B4 Bug #2) and the counterterm rows `pm[10..13] = Pk_ctr, Pk_ctr0, Pk_ctr2, Pk_ctr4` divided by `h` (Mpc, not Mpc³).
2. Rows 26, 28, 29 are `tree + loop` (`Pk_2_dd + Pk_2_dd1`, `Pk_4_vd + Pk_4_vd1`, `Pk_4_dd + Pk_4_dd1`); rows 15–20 are the six tree leaves, 21–25 and 27 the matching loop leaves.
3. Rows 40, 41 (`Pk_4_b1b2`, `Pk_4_b1bG2`) and 42–47 (`Id2d2_2 … IG2G2_4`) are never read by any classy accessor; fill 40/41 from the leaves and leave 42–47 zero.
4. classy `pk_mm_real(cs)` ↔ clax `pk_mm_real(cs0=cs)`; `pk_gg_real` takes `cs` and `cs0`; `pk_gm_real` takes `cs0` and `cs` (B4 Bug #5); `kh` passed to the twin is in h/Mpc (`e.kh`).
5. The twin's `Pd2d2_0` argument is `pd2d2_0(pm[14]·h³, kh)` — the IR-resummed tree (classy.pyx:4789-4791), which is clax `Pk_tree` after Bug #4 and what `pk_gg_l0` integrates internally via `_pd2d2_0(e.Pk_tree, e.kh)`.

- [ ] **Step 1: Write the failing tests**

Change the fixture's return to `return {"ept_out": ept_out, "k_ept": k_ept, "h": h, "fz": fz}` and extend the import block to

```python
from clax.ept import (
    compute_ept, EPTPrecisionParams, _pd2d2_0,
    pk_mm_real, pk_gg_real, pk_gm_real,
    pk_mm_l0, pk_mm_l2, pk_mm_l4, pk_gg_l0, pk_gg_l2, pk_gg_l4,
)
```

Append:

```python
# ---------------------------------------------------------------------------
# B7: clax accessors == scripts.classpt_assembly.assemble_from_pm on a pm array
# built from clax's own leaves (inverse of the ref §10 row map).
# Requires Part 1a A3 (scripts/classpt_assembly.py); skips until it exists.
# ---------------------------------------------------------------------------

try:
    from scripts import classpt_assembly as ca          # repo root on sys.path (PYTHONPATH / rootdir)
except ImportError:                                     # A3 not on this branch yet
    ca = None
needs_twin = pytest.mark.skipif(ca is None, reason="Part 1a A3 (scripts/classpt_assembly.py) not on this branch")

BIAS_KEYS = ("b1", "b2", "bG2", "bGamma3", "cs0", "cs2", "cs4", "cs", "Pshot", "b4")


def _pm_from_leaves(e, h):
    """(48, Nk) classy-convention pk_mult from EPTComponents (ref §10 inverted).
    Only pm[1] (Id2d2 stored negated) and pm[10..13] (Mpc, not Mpc³) differ
    from leaf/h³; rows 26/28/29 carry tree + loop; rows 42-47 are unused."""
    L = lambda a: np.asarray(a, dtype=float)
    h3 = h ** 3
    pm = np.zeros((48, L(e.kh).shape[0]))
    pm[0] = L(e.Pk_loop) / h3
    pm[1] = -L(e.Pk_Id2d2) / h3                                        # classy.pyx:4654
    for row, leaf in ((2, e.Pk_Id2), (3, e.Pk_IG2), (4, e.Pk_Id2G2), (5, e.Pk_IG2G2), (6, e.Pk_IFG2),
                      (7, e.Pk_IFG2_0b1), (8, e.Pk_IFG2_0), (9, e.Pk_IFG2_2), (14, e.Pk_tree),
                      (15, e.Pk_0_vv), (16, e.Pk_0_vd), (17, e.Pk_0_dd), (18, e.Pk_2_vv), (19, e.Pk_2_vd), (20, e.Pk_4_vv),
                      (21, e.Pk_0_vv1), (22, e.Pk_0_vd1), (23, e.Pk_0_dd1), (24, e.Pk_2_vv1), (25, e.Pk_2_vd1), (27, e.Pk_4_vv1),
                      (30, e.Pk_0_b1b2), (31, e.Pk_0_b2), (32, e.Pk_0_b1bG2), (33, e.Pk_0_bG2),
                      (34, e.Pk_2_b1b2), (35, e.Pk_2_b2), (36, e.Pk_2_b1bG2), (37, e.Pk_2_bG2),
                      (38, e.Pk_4_b2), (39, e.Pk_4_bG2), (40, e.Pk_4_b1b2), (41, e.Pk_4_b1bG2)):
        pm[row] = L(leaf) / h3
    pm[26] = (L(e.Pk_2_dd) + L(e.Pk_2_dd1)) / h3                       # ref §15: rows 26/28/29 = tree + loop
    pm[28] = (L(e.Pk_4_vd) + L(e.Pk_4_vd1)) / h3
    pm[29] = (L(e.Pk_4_dd) + L(e.Pk_4_dd1)) / h3
    for row, leaf in ((10, e.Pk_ctr), (11, e.Pk_ctr0), (12, e.Pk_ctr2), (13, e.Pk_ctr4)):
        pm[row] = L(leaf) / h                                          # classy.pyx: pm[10..13]/h² × h³
    return pm


def _clax_nine(e, bias):
    b1, b2, bG2, bG3 = (bias[k] for k in ("b1", "b2", "bG2", "bGamma3"))
    return {
        "pk_mm_real": pk_mm_real(e, cs0=bias["cs"]),                   # classy pk_mm_real(cs)
        "pk_gg_real": pk_gg_real(e, b1, b2, bG2, bG3, cs=bias["cs"], cs0=bias["cs0"], Pshot=bias["Pshot"]),
        "pk_gm_real": pk_gm_real(e, b1, b2, bG2, bG3, cs0=bias["cs0"], cs=bias["cs"]),
        "pk_mm_l0": pk_mm_l0(e, cs0=bias["cs0"]),
        "pk_mm_l2": pk_mm_l2(e, cs2=bias["cs2"]),
        "pk_mm_l4": pk_mm_l4(e, cs4=bias["cs4"]),
        "pk_gg_l0": pk_gg_l0(e, b1, b2, bG2, bG3, cs0=bias["cs0"], Pshot=bias["Pshot"], b4=bias["b4"]),
        "pk_gg_l2": pk_gg_l2(e, b1, b2, bG2, bG3, cs2=bias["cs2"], b4=bias["b4"]),
        "pk_gg_l4": pk_gg_l4(e, b1, b2, bG2, bG3, cs4=bias["cs4"], b4=bias["b4"]),
    }


def _rel(a, b):
    """max|a-b| / max|b| — norm-relative, so a zero crossing of a multipole
    cannot inflate a round-off difference (same metric as tests/test_ept_ap.py)."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


@needs_twin
def test_pd2d2_0_matches_twin(assembly_setup):
    e, h = assembly_setup["ept_out"], assembly_setup["h"]
    want = ca.pd2d2_0(np.asarray(e.Pk_tree), np.asarray(e.kh))
    got = float(_pd2d2_0(e.Pk_tree, e.kh))
    assert abs(got - want) < 1e-12 * abs(want), (got, want)


@needs_twin
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_accessors_match_classy_twin(assembly_setup, seed):
    """All nine spectra: clax accessors vs assemble_from_pm on pm built from the
    same EPTComponents, every bias nonzero (b2, cs, b4 included so Bugs #2, #3,
    #5 would show). 1e-12: both sides are the same double-precision algebra."""
    e, h, fz = assembly_setup["ept_out"], assembly_setup["h"], assembly_setup["fz"]
    rng = np.random.default_rng(seed)
    bias = dict(zip(BIAS_KEYS, [2.0, -1.0, 0.1, -0.1, 5.0, 15.0, -5.0, 1.0, 5.0e3, 100.0]))
    bias = {k: v * (1.0 + 0.3 * rng.uniform(-1, 1)) for k, v in bias.items()}   # all nonzero, seed-varied
    pm = _pm_from_leaves(e, h)
    kh = np.asarray(e.kh)
    Pd2d2_0 = ca.pd2d2_0(pm[14] * h ** 3, kh)
    twin = ca.assemble_from_pm(pm, h, fz, kh, bias, Pd2d2_0)
    ours = _clax_nine(e, bias)
    bad = {n: _rel(ours[n], twin[n]) for n in ours if _rel(ours[n], twin[n]) > 1e-12}
    assert not bad, f"clax accessors vs classy twin (ref §11): {bad}"


# rows whose agreement with the legacy file Track B already established (B4:
# bias rows 30-39 ~0.5%, tree rows 15-20 and 0/14 ~1% with legacy AP on)
_ROWS_ESTABLISHED = (0, 14, 15, 16, 17, 18, 19, 20) + tuple(range(30, 40))


def test_pm_from_leaves_roundtrips_legacy_rows(assembly_setup):
    """Sanity of the inverse map itself, not a new accuracy claim: on the legacy
    file the rows B4 already matched must still match through _pm_from_leaves
    (a mislabelled row would show as a gross mismatch). Every row 0-47 is
    written to test_logs/ept_assembly_row_roundtrip.txt for Part 2 C1."""
    ref = np.load(os.path.join(os.path.dirname(__file__), "..", "reference_data",
                               "classpt_z0.38_fullrange.npz"), allow_pickle=True)
    e, h = assembly_setup["ept_out"], assembly_setup["h"]
    pm = _pm_from_leaves(e, h)
    stored = np.asarray(ref["pk_mult"])[:48]
    kh = np.asarray(e.kh); sel = kh <= 0.4
    rel = {r: _rel(pm[r][sel], stored[r][sel]) for r in range(48)}
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "test_logs"), exist_ok=True)
    with open(os.path.join(os.path.dirname(__file__), "..", "test_logs", "ept_assembly_row_roundtrip.txt"), "w") as fh:
        fh.write("row  max|clax-legacy|/max|legacy| on k<=0.4 (legacy has AP on, |alpha-1|~2e-3)\n")
        fh.writelines(f"{r:3d}  {rel[r]:.3e}\n" for r in range(48))
    bad = [(r, round(rel[r], 4)) for r in _ROWS_ESTABLISHED if rel[r] > 2e-2]
    assert not bad, f"inverse row map disagrees with B4 on established rows (row, rel): {bad}"
```

- [ ] **Step 2: Run to verify failure / skip**

```bash
cd /home/n2minh/clax-ptval && JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2" PYTHONPATH=/home/n2minh/clax-ptval timeout 120 /home/n2minh/micromamba/envs/clax/bin/python -m pytest tests/test_ept_assembly.py -q -p no:cacheprovider 2>&1 | tail -n 12
```
Expected without A3 on the branch: the three original tests and `test_pm_from_leaves_roundtrips_legacy_rows` pass, the two twin tests are skipped by `needs_twin` (`4 passed, 4 skipped`). With A3 present: `test_pd2d2_0_matches_twin` passes (both are scipy-Simpson transcriptions of classy.pyx:4791); `test_accessors_match_classy_twin` passes for every seed if B4 and A3 both transcribed classy correctly. If it fails, the `bad` dict names the spectrum: open `/home/n2minh/CLASS-PT/python/classy.pyx` at the accessor's lines (ref §11) and settle which side deviates, term by term, before changing either — a fix on the clax side is a B4-style bug commit (test first, classy line cited); a fix on the twin side is a Track A commit (`scripts/classpt_assembly.py` + `tests/test_classpt_assembly.py`) and must keep A3's `classy == twin` assertion green, which is the arbiter. `test_pm_from_leaves_roundtrips_legacy_rows` passes; an established row > 2% off with the accessor test green means the inverse map has that row mislabelled — check it against ref §10 (the accessor test cannot see a row that classy multiplies by a bias the row map also mislabels consistently). Rows outside `_ROWS_ESTABLISHED` are only logged; a large number there (row 28, the `Pk_4_vd1` deficit, is the known one) is Part 2 C1's business.

- [ ] **Step 3: Gates, commit**

Local gate = B3 Step 8's two pytest commands. Cluster gate: `sbatch slurm/ptval-fast-suite.sbatch && sbatch slurm/ptval-track-b-full.sbatch`, both logs end in `PASS`. Commit (`commit-B7.txt`):

```
test(ept): clax accessors cross-checked against the classy NumPy twin

tests/test_ept_assembly.py: pm (48, Nk) rebuilt from EPTComponents (ref §10
inverted: pm[1] = -Pk_Id2d2/h³, pm[10..13] = ctr/h, rows 26/28/29 tree+loop),
assembled with scripts.classpt_assembly.assemble_from_pm and compared with the
nine clax accessors at three all-nonzero bias sets: max rel <table>. _pd2d2_0
== classpt_assembly.pd2d2_0 to 1e-12. Inverse row map vs legacy pk_mult on
0.01-0.3: worst row <r> at <x>%.
Cluster gate: ptval-fast-suite <id> PASS, ptval-track-b-full <id> PASS.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YBYncmhA8JrJQVNw4pVzmb
```

```bash
cd /home/n2minh/clax-ptval && git add tests/test_ept_assembly.py && git commit -F /tmp/claude-17163/-home-n2minh-clax/85880c7b-9aa1-44ed-b5d2-49320fe59b2a/scratchpad/commit-B7.txt
```

---

## Track B self-check (run by the last Track B implementer; paste the answers in the B7 commit body or a follow-up note)

Every line is a command or a yes/no with evidence; "looks fine" is not an answer.

1. `grep -n "B5:\|TODO\|XXX" clax/ept.py` → no output.
2. `grep -n "^EXEMPT" tests/test_ept_ap.py` → `EXEMPT: dict = {}`; `python -c "import numpy as np; d=np.load('reference_data/ept_alpha1_baseline.npz'); print(d['_git_sha'], d['_reason'])"` → the B4 Bug #2 commit's sha and reason (B5–B7 refroze nothing).
3. `git log --oneline bf8ac18..HEAD -- clax/ept.py` → exactly B3, B4 (five bug commits: `commit-B4a..e`), B5. Any other commit touching `ept.py` is explained in its own message with a test.
4. The Bug #1–#5 tests each failed before their fix: the five B4 commit bodies quote the before-fix numbers (`bad` lists, `Pk_tree` max-rel, `gm` mismatch).
5. `pytest tests/test_ept_ap.py -q -k "reproduces_classpt_ap_loop"` → `4 passed` (2 seeds × 2 AP pairs) — the transcription in `_classpt_ap_loop` was written from `nonlinear_pt.c` lines, not from `_gl_multipoles` (diff the two by eye: they share no helper).
6. If A3's files exist: `pytest tests/test_ept_ap.py -q -k "noap_file or ap_file or ap_off"` → `3 passed`, and the max-rel tables in the B5 commit show the AP-on comparison ≤ the α=1 comparison on every multipole. If they do not exist yet: the B5 commit body says so and Part 2 C1 is where the oracle numbers land.
7. `tests/test_ept_accuracy.py` before/after table in the B5 commit: `pk_gg_l2`, `pk_gg_l4` tightened; `Pk_4_vd1` deficit (B3 finding) explained — either closed by B4/B5 (say which commit, with the number) or still open and recorded in CHANGELOG as a Part 2 C1 item.
8. Both cluster jobs (`ptval-fast-suite`, `ptval-track-b-full`) `PASS` on the final Track B HEAD, job ids in the B7 commit.
9. `git status --porcelain` → empty; `git diff main --stat -- reference_data/` lists only `ept_alpha1_baseline.npz` and the `gauss_legendre` table (B1) — no legacy `.npz` was rewritten by Track B.
10. Findings for the user, one line each, with the classy/nonlinear_pt.c line: anything discovered in B3–B7 that the plan did not predict (a sixth bug, a spline-boundary effect, a tolerance that had to move).
