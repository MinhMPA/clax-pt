# clax-pt RSD-basis gradient fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make clax-pt's derivatives of the redshift-space multipoles with respect to the linear spectrum complete, by removing the `stop_gradient` snapshot that feeds the one-loop FFTLog basis, and prove it with AD-vs-finite-difference tests at several cosmologies.

**Architecture:** One behaviour change in `compute_ept` (`clax/ept.py`): on the `_ir_precomputed` path the RSD basis decomposition reads the traced `pk_nw`/`pk_w` the function already holds. Values are bit-identical; only the gradient graph grows. A new test file differentiates P_0, P_2, P_4 through the traced IR splitter and compares with central finite differences; existing suites pin the NumPy path and the real-space gradients.

**Tech Stack:** Python 3.14, JAX 0.9.2 (x64), numpy/scipy, pytest; SLURM igpu nodes, env `clax`.

**Spec:** `docs/superpowers/specs/2026-10-10-ept-rsd-basis-gradient-design.md` (read it first).

## Global Constraints

- Branch `fix/ept-rsd-basis-gradient` from clax fork `main` at `6ba5176`, worktree `/home/n2minh/clax-fix-rsdbasis`. Never edit the main checkout `/home/n2minh/clax` (it is on another branch with uncommitted work) or the validation worktree `/home/n2minh/clax-ps1loop`.
- All pytest/JAX runs go through `.superpowers/igpu/run-pytest.sh` (igpu SLURM, env `clax`, `PYTHONPATH=<worktree>`). No heavy JAX on the login node.
- The NumPy path of `compute_ept` (`_ir_precomputed=None`) is byte-identical before and after; `tests/test_ept_accuracy.py` must stay 9/9.
- Thresholds (spec §4): AD vs FD 1e-2; forward = reverse 1e-10; primal traced vs NumPy ≤ 1e-8 and unchanged by the fix. Never widen a threshold to pass; a failure is investigated and reported.
- Multi-cosmology rule (CLAUDE.md): the new value tests run on the `lcdm_cosmology` fixture grid (5 points; fiducial only under `--fast`).
- Commit messages: conventional prefix, ending with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Never push.
- CHANGELOG date headings use `Mon D, YYYY`.

## Review Focus

1. **A user calls `compute_ept` without `_ir_precomputed` (the NumPy path).** They expect nothing to change. Pinned by `tests/test_ept_accuracy.py` (Task 0 baseline and Task 2 regression) and by the primal test's printed traced-vs-NumPy value being identical before and after the fix.
2. **A user differentiates P_ℓ with respect to f, hratio or Dratio.** They expect these gradients untouched. Pinned by the existing `tests/test_ept_gradients.py` RSD tests (`pk_mm_l0`, `pk_gg_l2`) staying green, and by the AP-ratio FD test if present on main (Task 0 lists it).
3. **A user takes reverse-mode gradients (HMC) through the new traced basis.** They expect finite values equal to forward mode. Pinned by the forward = reverse and finiteness tests (Task 1).
4. **A user runs at a cosmology far from fiducial (e.g. h_high, omega_cdm_low).** They expect the derivative accuracy to hold. Pinned by the `lcdm_cosmology` grid (Task 1, full mode).
5. **A user reads the code comments to decide whether the freeze still applies.** They expect them to be true. Pinned by Task 2's documentation sweep (every listed line) and the reviewer's check.

---

### Task 0: Worktree, igpu runner, baseline

**Files:**
- Create (local, git-excluded): `.superpowers/igpu/clax-pytest.sbatch`, `.superpowers/igpu/run-pytest.sh`, `.superpowers/igpu/run-fast-suite.sh`
- Copy in and commit: this plan and its spec

**Interfaces:**
- Produces: `.superpowers/igpu/run-pytest.sh <pytest args...>` (blocks; exits with pytest's status; full log `/lustre/work/n2minh/std/clax/fixrsd/clax-fix-tests.pytest.<jobid>.log`); `.superpowers/igpu/run-fast-suite.sh` (the clax fast suite with the campaign deselect).

- [ ] **Step 1: Create the branch and worktree**

```bash
cd /home/n2minh/clax
git fetch -q fork && git worktree add -b fix/ept-rsd-basis-gradient /home/n2minh/clax-fix-rsdbasis fork/main
cd /home/n2minh/clax-fix-rsdbasis && git log --oneline -1   # expect 6ba5176
grep -q "^.superpowers/" "$(git rev-parse --git-common-dir)/info/exclude" || echo ".superpowers/" >> "$(git rev-parse --git-common-dir)/info/exclude"
mkdir -p .superpowers/igpu docs/superpowers/specs docs/superpowers/plans
cp /home/n2minh/clax/docs/superpowers/specs/2026-10-10-ept-rsd-basis-gradient-design.md docs/superpowers/specs/
cp /home/n2minh/clax/docs/superpowers/plans/2026-10-10-ept-rsd-basis-gradient.md docs/superpowers/plans/
git status --porcelain   # expect only the two docs
```

- [ ] **Step 2: Write the runner**

Copy the validation worktree's runner and adapt it: `cp /home/n2minh/clax-ps1loop/slurm/ps1loop-tests.sbatch .superpowers/igpu/clax-pytest.sbatch` and `cp /home/n2minh/clax-ps1loop/slurm/run-ps1loop-pytest.sh .superpowers/igpu/run-pytest.sh`, then edit:
- in the sbatch: `--job-name=clax-fix-tests`; output/error paths and `OUT` to `/lustre/work/n2minh/std/clax/fixrsd`; `REPO=${CLAX_REPO:-/home/n2minh/clax-fix-rsdbasis}`; the echo line prints only `HEAD`; the import check becomes `python -c "import jax, clax; print('devices', jax.devices(), clax.__file__)"` and the job must abort (`exit 3`) if `clax.__file__` is not under `$REPO`; log name `clax-fix-tests.pytest.<jobid>.log`.
- in the wrapper: `OUT=/lustre/work/n2minh/std/clax/fixrsd`, `mkdir -p`, the sbatch path `"$(dirname "$0")/clax-pytest.sbatch"`, the `cd` to the worktree root (`"$(dirname "$0")/../.."`), tail `clax-fix-tests.out.$jid`, grep `clax-fix-tests.pytest.$jid.log`.

Create `.superpowers/igpu/run-fast-suite.sh`:

```bash
#!/bin/bash
# The clax fast suite as the validation runs it (tests/ --fast -x, with the campaign's standing deselect).
exec "$(dirname "$0")/run-pytest.sh" tests/ --fast -x -q -p no:cacheprovider \
  --deselect tests/test_solver_selection.py::TestRosenbrockPk::test_pk_rosenbrock_vs_kvaerno5
```

`chmod +x .superpowers/igpu/run-pytest.sh .superpowers/igpu/run-fast-suite.sh`.

- [ ] **Step 3: Baseline**

Run: `.superpowers/igpu/run-pytest.sh tests/test_ept_gradients.py tests/test_ir_resummation_jax.py tests/test_ept.py tests/test_ept_assembly.py tests/test_ept_accuracy.py --fast -q -p no:cacheprovider`
Expected: `PASS`; the out file shows `clax.__file__` under `/home/n2minh/clax-fix-rsdbasis`. Record the job id and counts. Also run `ls tests/ | grep -E "ept_ap|test_ap"` and record whether an AP-ratio FD test exists on main (Review Focus 2). Pre-existing failures are recorded, not fixed.

- [ ] **Step 4: Commit the docs**

```bash
git add docs/superpowers/specs/2026-10-10-ept-rsd-basis-gradient-design.md docs/superpowers/plans/2026-10-10-ept-rsd-basis-gradient.md
git commit -m "docs(ept): spec and plan for the RSD-basis gradient fix

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 1: The failing gradient tests

**Files:**
- Create: `tests/test_ept_rsd_basis_gradients.py`

**Interfaces:**
- Consumes: `clax.ept.compute_ept(pk_lin_h, k_h, h=, f=, prec=, _ir_precomputed=, rs_h=)`, `clax.ept._ir_resummation_jax(pk_lin_h, k_h_static, rs_h, h_conc) -> (pk_nw, pk_w, sigma2, dsigma2)`, `clax.ept.ept_kgrid(prec)`, `clax.ept.EPTPrecisionParams`, `clax.ept.pk_gg_l0/l2/l4(ept, b1, b2, bG2, bGamma3)`, `clax.ept.pk_mm_l0(ept)`; `tests/conftest.py` fixtures `lcdm_cosmology -> (name, CosmoParams)`, `cosmology_reference_dir(name)`; `reference_data/<name>/pk.npz` (`k` [1/Mpc], `pk_lin_z0` [Mpc^3]).
- Produces: the module-level helper `rsd_multipoles_fn(k_ept, h, f, prec, window) -> F(pk_lin) -> (3, Nw)`, used by Task 2's documentation only.

- [ ] **Step 1: Write the tests**

Create `tests/test_ept_rsd_basis_gradients.py`:

```python
"""Derivatives of the redshift-space multipoles with respect to the linear spectrum, through the
traced IR splitter (compute_ept with _ir_precomputed = _ir_resummation_jax(...), the path
compute_ept_from_clax uses).

Before Oct 10, 2026 the one-loop FFTLog basis on this path was a stop_gradient snapshot of the IR
split, so d(P_ell)/d(P_lin) omitted the one-loop response: AD disagreed with finite differences by
~90-100% along amplitude and stretch directions (ps_1loop validation, Task 7). The existing
tests/test_ept_gradients.py holds the IR split fixed in both arms and never saw it.

Multi-cosmology rule: the lcdm_cosmology grid (fiducial only under --fast). P_lin comes from the
stored reference tables; f and rs_h are fixed representative values (the derivative checks do not
depend on them).
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from clax.ept import (EPTPrecisionParams, _ir_resummation_jax, compute_ept, ept_kgrid,
                      pk_gg_l0, pk_gg_l2, pk_gg_l4, pk_mm_l0)
from tests.conftest import cosmology_reference_dir

BIAS = (2.0, -0.5, 0.1, -0.1)        # b1, b2, bG2, bGamma3: exercises every RSD loop family
F_GROWTH = 0.75                       # z ~ 0.4 growth rate, representative
RS_H = 99.0                           # r_d h [Mpc/h], the compute_ept default
WINDOW = (0.01, 0.3)                  # h/Mpc, the cross-validation's comparison window
FD_STEP = 1e-4                        # relative step along the tangent
AD_FD_TOL = 1e-2                      # CLAUDE.md gradient target
PRIMAL_TOL = 1e-8
TANGENTS = ("amplitude", "stretch", "wiggle")


@pytest.fixture(scope="module")
def prec():
    return EPTPrecisionParams()


@pytest.fixture
def setup(lcdm_cosmology, prec):
    name, cosmo = lcdm_cosmology
    ref = cosmology_reference_dir(name)
    if ref is None:
        pytest.skip(f"no reference_data for {name}")
    d = np.load(os.path.join(ref, "pk.npz"))
    h = float(cosmo.h)
    k_h_ref, pk_h_ref = d["k"] / h, d["pk_lin_z0"] * h**3           # h/Mpc, (Mpc/h)^3
    k_ept = ept_kgrid(prec)
    spl = CubicSpline(np.log(k_h_ref), np.log(pk_h_ref), extrapolate=True)
    pk_lin = np.exp(spl(np.log(k_ept)))
    assert np.all(np.isfinite(pk_lin))
    window = (k_ept >= WINDOW[0]) & (k_ept <= WINDOW[1])
    dlnp = spl.derivative()(np.log(k_ept))
    pk_j = jnp.asarray(pk_lin)
    ir0 = _ir_resummation_jax(pk_j, k_ept, jnp.asarray(RS_H), h)
    tangents = {"amplitude": pk_j, "stretch": pk_j * jnp.asarray(dlnp), "wiggle": ir0[1]}
    return dict(name=name, h=h, k_ept=k_ept, pk=pk_j, window=window, tangents=tangents)


def rsd_multipoles_fn(k_ept, h, f, prec, window):
    """F(pk_lin) -> (P_0, P_2, P_4) on the window, IR split TRACED (compute_ept_from_clax's path)."""
    k_j = jnp.asarray(k_ept)

    def F(pk_lin):
        ir = _ir_resummation_jax(pk_lin, k_ept, jnp.asarray(RS_H), h)
        e = compute_ept(pk_lin, k_j, h=h, f=f, prec=prec, _ir_precomputed=ir, rs_h=RS_H)
        return jnp.stack([pk_gg_l0(e, *BIAS)[window], pk_gg_l2(e, *BIAS)[window], pk_gg_l4(e, *BIAS)[window]])

    return F


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


@pytest.mark.parametrize("tangent", TANGENTS)
def test_rsd_multipole_jvp_matches_finite_differences(setup, prec, tangent):
    F = rsd_multipoles_fn(setup["k_ept"], setup["h"], F_GROWTH, prec, setup["window"])
    pk, t = setup["pk"], setup["tangents"][tangent]
    _, ad = jax.jvp(F, (pk,), (t,))
    fd = (F(pk + FD_STEP * t) - F(pk - FD_STEP * t)) / (2 * FD_STEP)
    ad, fd = np.asarray(ad), np.asarray(fd)
    assert np.all(np.isfinite(ad)), f"{setup['name']} {tangent}: non-finite AD"
    errs = [_rel(ad[i], fd[i]) for i in range(3)]
    print(f"{setup['name']} d P_ell / d P_lin along {tangent}: AD vs FD l=0,2,4 = "
          + " ".join(f"{e:.2e}" for e in errs))
    assert max(errs) < AD_FD_TOL, f"ERROR {setup['name']} {tangent}: AD vs FD {max(errs):.3e} > {AD_FD_TOL}"


def test_rsd_multipole_forward_equals_reverse(setup, prec):
    F = rsd_multipoles_fn(setup["k_ept"], setup["h"], F_GROWTH, prec, setup["window"])
    pk, t = setup["pk"], setup["tangents"]["stretch"]
    out, ad = jax.jvp(F, (pk,), (t,))
    c = 1.0 / (jnp.abs(out) + 1.0)                        # a fixed, well-scaled cotangent
    _, vjp = jax.vjp(F, pk)
    (g,) = vjp(c)
    lhs, rhs = float(jnp.vdot(c, ad)), float(jnp.vdot(g, t))
    assert np.isfinite(lhs) and np.isfinite(rhs)
    rel = abs(lhs - rhs) / abs(lhs)
    print(f"{setup['name']} c.(J t) jvp vs vjp rel diff {rel:.2e}")
    assert rel < 1e-10, f"ERROR {setup['name']}: forward/reverse mismatch {rel:.2e}"


def test_traced_path_primal_matches_numpy_path(setup, prec):
    """The traced-splitter path and the NumPy path give the same multipoles. Print the value: the
    gradient fix removes a stop_gradient only, so this number must not move."""
    F = rsd_multipoles_fn(setup["k_ept"], setup["h"], F_GROWTH, prec, setup["window"])
    traced = np.asarray(F(setup["pk"]))
    e_np = compute_ept(setup["pk"], jnp.asarray(setup["k_ept"]), h=setup["h"], f=F_GROWTH, prec=prec, rs_h=RS_H)
    w = setup["window"]
    numpy_path = np.stack([np.asarray(pk_gg_l0(e_np, *BIAS))[w], np.asarray(pk_gg_l2(e_np, *BIAS))[w],
                           np.asarray(pk_gg_l4(e_np, *BIAS))[w]])
    rel = max(_rel(traced[i], numpy_path[i]) for i in range(3))
    print(f"{setup['name']} traced vs numpy path max rel diff {rel:.3e}")
    assert rel <= PRIMAL_TOL, f"ERROR {setup['name']}: traced vs numpy path {rel:.3e}"


def test_matter_monopole_jvp_matches_finite_differences(setup, prec):
    """The matter-only RSD path (pk_mm_l0) through the traced splitter, amplitude direction."""
    k_j = jnp.asarray(setup["k_ept"])

    def F(pk_lin):
        ir = _ir_resummation_jax(pk_lin, setup["k_ept"], jnp.asarray(RS_H), setup["h"])
        return pk_mm_l0(compute_ept(pk_lin, k_j, h=setup["h"], f=F_GROWTH, prec=prec, _ir_precomputed=ir, rs_h=RS_H))[setup["window"]]

    pk, t = setup["pk"], setup["tangents"]["amplitude"]
    _, ad = jax.jvp(F, (pk,), (t,))
    fd = (F(pk + FD_STEP * t) - F(pk - FD_STEP * t)) / (2 * FD_STEP)
    err = _rel(np.asarray(ad), np.asarray(fd))
    print(f"{setup['name']} d pk_mm_l0 / d P_lin (amplitude): AD vs FD {err:.2e}")
    assert err < AD_FD_TOL, f"ERROR {setup['name']}: pk_mm_l0 AD vs FD {err:.3e}"
```

- [ ] **Step 2: Run them to verify the RED**

Run: `.superpowers/igpu/run-pytest.sh tests/test_ept_rsd_basis_gradients.py --fast -v -s -p no:cacheprovider`
Expected: the three `jvp_matches_finite_differences` cases and `matter_monopole` FAIL with AD vs FD ≈ 0.9–1.0 (record each number); `forward_equals_reverse` PASSES (both modes see the same frozen graph); `traced_path_primal_matches_numpy_path` PASSES — record its printed value to ~3 digits (Task 2 must reproduce it). If the primal test fails (> 1e-8), STOP and report: the two splitters differ more than expected, a separate issue.

Do not commit the tests alone; Task 2 commits tests and fix together so the branch never holds a known-red test.

---

### Task 2: The fix and the documentation sweep

**Files:**
- Modify: `clax/ept.py` (`compute_ept`, the `_ir_precomputed` RSD-basis block ~2106–2131; docstrings/comments at ~820, 831, 862, 2001–2009, 2686, 2725)
- Modify: `tests/test_ept_gradients.py` (docstring wording at ~567, 687–701, 772–783)
- Test: `tests/test_ept_rsd_basis_gradients.py` (from Task 1)

- [ ] **Step 1: Implement**

In `compute_ept`, replace the RSD-basis block

```python
    if prec.ir_resummation:
        if _ir_precomputed is not None:
            # PHASE-2 FREEZE: ...
            _pk_nw_np_rsd = jax.lax.stop_gradient(jnp.asarray(_ir_precomputed[0]))
            _pk_w_np_rsd  = jax.lax.stop_gradient(jnp.asarray(_ir_precomputed[1]))
        else:
            _pk_nw_np_rsd = np.array(pk_nw)
            _pk_w_np_rsd  = np.array(pk_w)
        _cmsym_nw_np, _ = _fftlog_decompose(_pk_nw_np_rsd, kmin, kmax, nmax, b)
        _cmsym_w_np, _  = _fftlog_decompose(_pk_w_np_rsd,  kmin, kmax, nmax, b)
        cmsym_nw_jnp = jnp.array(_cmsym_nw_np)
        cmsym_w_jnp  = jnp.array(_cmsym_w_np)
        pk_nw_jnp    = jnp.array(_pk_nw_np_rsd)
```

with

```python
    if prec.ir_resummation:
        # The RSD one-loop basis is decomposed from the SAME pk_nw / pk_w (= pk_lin_h - pk_nw)
        # used for the tree and the real-space loops above. On the _ir_precomputed path both are
        # traced, so d(P_ell)/d(pk_lin_h) carries the one-loop basis response (the earlier
        # stop_gradient snapshot -- "PHASE-2 FREEZE", a leftover of the NumPy-only splitter --
        # dropped it: AD vs FD was ~90-100% off along P_lin directions, ps_1loop validation
        # Task 7, Oct 10, 2026; fixed Oct 10, 2026). On the NumPy path (no _ir_precomputed)
        # pk_nw / pk_w are concrete and nothing is traced, exactly as before. Values are
        # bit-identical on both paths: _ir_resummation_jax's pk_w is pk_lin_h - pk_nw too.
        # cf. nonlinear_pt.c lines 8215, 8246, 8562, 8586 (x_nw / x_w bases).
        cmsym_nw_jnp, _ = _fftlog_decompose(pk_nw, kmin, kmax, nmax, b)
        cmsym_w_jnp, _  = _fftlog_decompose(pk_w,  kmin, kmax, nmax, b)
        pk_nw_jnp = pk_nw
```

(Keep the surrounding comment block's CLASS-PT line citations; delete the paragraph that says the RSD path "deliberately reads the PINNED snapshot".) Check that `_fftlog_decompose` returns a jnp array for jnp input (it is `jnp.fft`-based per its definition at `ept.py:366`), so `cmsym_nw_jnp`/`cmsym_w_jnp` need no `jnp.array(...)` wrapping; if it returns NumPy for NumPy input on the NumPy path, wrap with `jnp.asarray` — values unchanged either way.

Documentation sweep (each must describe the new state truthfully):
- `ept.py` ~820–831 (`_ir_resummation_jax` docstring): item (2) "the RSD FFTLog basis (phase 2, …)" → now traced; only the DST grid endpoints stay pinned.
- `ept.py` ~862: the "concrete phase-2 item" sentence refers to the DST endpoints, keep; remove any claim that the RSD basis is frozen.
- `ept.py` ~2001–2009 (`compute_ept` docstring, "Physically-correct-freeze justification"): rewrite to say no freeze is applied on either input of the RSD basis.
- `ept.py` ~2686 and ~2725 (`compute_ept_from_clax` "stop_gradient rationale" note): the RSD basis is traced since Oct 10, 2026; the remaining pinned item is the DST grid endpoints.
- `tests/test_ept_gradients.py` ~567, 687–701, 772–783: change "still deferred elsewhere" / "deferred" to "removed Oct 10, 2026 (tests/test_ept_rsd_basis_gradients.py)"; the statements that those real-space functionals never read the RSD basis stay true.

- [ ] **Step 2: Run the new tests to verify GREEN**

Run: `.superpowers/igpu/run-pytest.sh tests/test_ept_rsd_basis_gradients.py --fast -v -s -p no:cacheprovider`
Expected: all pass; AD vs FD ≲ 1e-4 on every tangent (record); the primal test prints the SAME value as in Task 1 (compare digit by digit); forward = reverse ≤ 1e-10.
If AD vs FD lands between 1e-3 and 1e-2 on any tangent, it passes but report the value: the likely cause is the pinned DST-grid endpoints (spec §2, not changed), and the number goes to the CHANGELOG.

Then full mode (all 5 grid points): `.superpowers/igpu/run-pytest.sh tests/test_ept_rsd_basis_gradients.py -v -s -p no:cacheprovider` → all pass; record the worst AD-vs-FD per tangent over the grid.

- [ ] **Step 3: Regression suites, fast suite, commit**

Run: `.superpowers/igpu/run-pytest.sh tests/test_ept_gradients.py tests/test_ir_resummation_jax.py tests/test_ept.py tests/test_ept_assembly.py tests/test_ept_accuracy.py --fast -q -p no:cacheprovider` → `PASS` (test_ept_accuracy 9/9). Then `.superpowers/igpu/run-fast-suite.sh` → `PASS`.

```bash
git add clax/ept.py tests/test_ept_gradients.py tests/test_ept_rsd_basis_gradients.py
git commit -m "fix(ept): differentiate the RSD one-loop FFTLog basis through the traced IR split

The _ir_precomputed path fed a stop_gradient snapshot of pk_nw/pk_w into the RSD basis
(PHASE-2 FREEZE, a leftover of the NumPy-only splitter), so d(P_ell)/d(P_lin) omitted the
one-loop response: AD vs FD ~90-100% off along amplitude/stretch (ps_1loop validation Task 7).
Values are bit-identical; only the gradient graph changes. New AD-vs-FD, forward=reverse and
primal tests on the lcdm_cosmology grid.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: CHANGELOG and hand-off

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: CHANGELOG entry**

Add at the top of the progress section, in the file's `Mon D, YYYY` style:

```markdown
### Oct 10, 2026: RSD one-loop basis now differentiable through the traced IR split (clax-pt)

- `compute_ept`'s `_ir_precomputed` path built the redshift-space FFTLog basis from a
  `stop_gradient` snapshot of the IR split ("PHASE-2 FREEZE", explicit since 322a6ab, a leftover
  of the NumPy-only splitter). d(P_0,2,4)/d(P_lin) therefore omitted the one-loop response:
  forward-mode AD vs central finite differences was ~90% (amplitude) / ~100% (stretch) off at
  lcdm z=0.38 (found by the clax-pt vs ps_1loop_jax cross-validation, Task 7). f / AP-ratio
  derivatives were unaffected. No earlier clax result depended on this channel: the existing
  gradient tests hold the IR split fixed in both arms or use real-space functionals, and the e2e
  campaign's gradient tests assert finiteness / AP ratios only.
- Fix: the basis is decomposed from the traced pk_nw / pk_w already in hand; values bit-identical
  (NumPy path untouched, test_ept_accuracy 9/9). New `tests/test_ept_rsd_basis_gradients.py`:
  AD vs FD along amplitude, stretch and wiggle tangents on the lcdm_cosmology grid (worst <value>),
  forward = reverse (<value>), traced-vs-NumPy primal (<value>, unchanged by the fix).
- Still pinned (documented): the DST grid endpoints inside `_ir_resummation_jax` (h-derivative
  boundary term).
```

Fill `<value>` from Task 2's full-mode run.

- [ ] **Step 2: Fast suite, commit**

Run `.superpowers/igpu/run-fast-suite.sh` → `PASS` (or cite Task 2's run if no code changed since; the CHANGELOG is docs only).

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): RSD-basis gradient fix

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 3: Hand-off (controller)**

Do not push. Report `git log --oneline 6ba5176..HEAD`, the worst AD-vs-FD per tangent over the grid, and the primal value. The controller asks the user: (1) push `fix/ept-rsd-basis-gradient` and open a PR on the fork (`MinhMPA/clax`, the PR #8 precedent); (2) after merge, merge `main` into `feat/ps1loop-cross-validation` (3 commits, no `compute_ept` loop change) and resume the validation's Task 7 against the fixed clax-pt, restoring the saved WIP (`task7-wip.patch`).
