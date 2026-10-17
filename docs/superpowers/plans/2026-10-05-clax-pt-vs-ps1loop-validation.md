# clax-pt vs ps_1loop_jax Second-Oracle Validation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate clax-pt's one-loop EFT galaxy power spectrum against ps_1loop_jax on identical inputs, coefficient by coefficient under the code-accuracy gate, and in full under a PFS Δχ² < 1 error-bar gate, with CLASS-PT as tie-breaker.

**Architecture:** One test-support bridge (`tests/ps1loop_bridge.py`) feeds a CLASS-PT reference file's own P_cb(k), f, r_s·h and AP ratios to both codes, translates ps_1loop's parameters into clax-pt's, and extracts coefficient spectra with exact parameter-space stencils. Test files then gate: exact coefficients (42 files), model-difference closure, real space with IR off, cross-code derivatives, and the PFS error-bar gate (35 new CLASS-PT files). A diagnostic ladder attributes ps_1loop resolution effects.

**Tech Stack:** Python 3.14, JAX 0.9.2 (x64), clax (`clax.ept`), ps_1loop_jax 0.4.0 (quadax, interpax), CLASS-PT (env `classpt`) via `scripts/generate_classpt_reference.py`, pytest, SLURM on igpu V100 nodes.

**Spec:** `docs/superpowers/specs/2026-10-05-clax-pt-vs-ps1loop-validation-design.md` (glossary: `CONTEXT.md`). Read both before starting.

## Global Constraints

- Code-accuracy thresholds: 1% for ℓ = 0, 2 and real space; 2% for ℓ = 4 — `cu.THRESHOLDS` values. Thresholds only tighten (ceil_to_1sf(2× worst)), never below 1e-3 relative in-window (Ruling 20).
- Window: CLASS-PT grid points `[NSIDE:]` (NSIDE = 10) with k ≤ 0.3 h/Mpc — `cu.window`.
- Metric: `max|a − b| / max|b|` on the window; structural-zero branch per spec §5.1 (denominator = the monomial's largest clax-pt multipole when max|clax_ℓ| ≤ 1e-12 × that).
- Error-bar gate: Σ over 7 PFS slices of ΔPᵀC⁻¹ΔP < 1 per cosmology; bins Δk = 0.01 h/Mpc, 0.01 ≤ k ≤ 0.2 h/Mpc; Gaussian ℓ-ℓ′ covariance from clax-pt P(k,μ) which already contains shot noise.
- PFS numbers: Takada et al. (2014) Table 2 only (spec §3). b_g = 0.9 + 0.4z. C_REF = 2.0 (Mpc/h)² is a labelled placeholder. CFOG_REF = 500.0 (Mpc/h)⁴.
- No fudge factors. A failing gate is investigated and attributed (spec §6), never loosened.
- Units: k in h/Mpc, P in (Mpc/h)³, r_s·h in Mpc/h, V in (Mpc/h)³, n̄ in (h/Mpc)³.
- Multi-cosmology rule (CLAUDE.md): physics tests run at ≥ 3 cosmologies in full mode; `--fast` prunes to `vc.FAST_CASES` × z = 0.38 or to `lcdm_fiducial`. Exemptions are stated in the test-module docstring.
- All pytest/JAX runs go through SLURM on igpu nodes (`slurm/run-ps1loop-pytest.sh`); never heavy JAX on the login node. CLASS-PT runs use env `classpt`; JAX runs env `clax` (`micromamba activate clax`).
- No git network operations inside sbatch scripts.
- Before every commit: the fast suite passes (`slurm/run-ps1loop-fast-suite.sh`, CLAUDE.md rule).
- Every new test prints ≤ 10 lines on success; failure lines start with `ERROR` (greppable).
- CHANGELOG dates use `Mon D, YYYY`.
- Comments cite source lines for every translated formula (ps_1loop.py:… / ept.py:… / classy.pyx:…).
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **ps_1loop methods are jitted with `static_argnames=['self']`.** Changing `rbao`, `ks` or `_mu` on an instance after its first call is silently ignored, because the compiled code is reused. The expected behaviour is that each cosmology's r_s·h actually reaches Σ²_BAO. Pinned by `test_ps_model_cache_keys_on_rbao_and_overrides_before_first_call` and `test_rbao_reaches_the_computation` (Task 3), and by the ladder's undo check (Task 9).
2. **Shot noise counted twice in the covariance.** The clax-pt P_ℓ already contain Pshot = V/n̄. Adding 1/n̄ again would inflate σ and pass the gate too easily. Pinned by `test_covariance_isotropic_closed_forms` (Task 8), which builds C only from the P passed in.
3. **Mixed units: h/Mpc vs 1/Mpc, (Gpc/h)³ vs (Mpc/h)³.** A missed factor of h in `rbao`, or 10⁹ in V, gives a wrong Σ² or a wrong N_k without any crash. Pinned by `test_rbao_is_rs_times_h` (Task 3) and `test_units_and_volume_scale` (Task 1).
4. **A missing reference file silently skips the gate.** `cu.require_reference` skips when a file is absent, so an unrun refgen job would make the PFS gate "pass". Pinned by `test_all_new_references_present` (Task 2), which fails (not skips) on any missing file.
5. **NaN read as a pass.** `err > thr` is False for NaN. Pinned by `test_coef_error_metrics` and `test_gate_flags_non_finite_and_attributes` (Task 3).

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `slurm/ps1loop-tests.sbatch` | create | one pytest job on an igpu node, env `clax`, args from `PYTEST_ARGS` |
| `slurm/run-ps1loop-pytest.sh` | create | login-node wrapper: submit, wait, print the tail and the ERROR lines |
| `slurm/run-ps1loop-fast-suite.sh` | create | the CLAUDE.md pre-commit fast suite (with the campaign's deselect) |
| `reference_data/pfs/takada2014_table2.json` | create | Takada et al. 2014 Table 2, verbatim |
| `tests/pfs_survey.py` | create | PFS slices, bias point, covariance, Δχ², headroom, bin interpolation |
| `tests/test_pfs_survey.py` | create | transcription and numerics checks |
| `scripts/validation_cosmologies.py` | modify | `PFS_Z_LIST`, `PFS_CASES`, `IRNO_CASES`, `IRNO_TAG` |
| `slurm/classpt-refgen-pfs.sbatch` | create | 35 PFS-slice + 3 IR-off CLASS-PT runs + manifest |
| `reference_data/classpt/<case>/z*.npz`, `MANIFEST.md` | add | the 38 new reference files |
| `tests/test_classpt_pfs_references.py` | create | presence and flag gate for the 38 files |
| `tests/ps1loop_bridge.py` | create | shared inputs, ps_1loop/clax/CLASS-PT evaluators, translation, stencils, gate, attribution, log |
| `tests/test_ps1loop_bridge.py` | create | bridge unit tests |
| `tests/test_ps1loop_coefficients.py` | create | exact-monomial coefficient gate, 42 files |
| `tests/test_ps1loop_model_difference.py` | create | closure + IR-off exactness |
| `tests/test_ps1loop_real_space.py` | create | real space with IR off + clax IR-off vs CLASS-PT IR-off |
| `tests/test_ps1loop_gradients.py` | create | cross-code tangents + ps_1loop AD vs FD |
| `tests/test_ps1loop_pfs_gate.py` | create | the error-bar gate |
| `scripts/ps1loop_convergence_ladder.py` | create | diagnostic ladder |
| `tests/test_ps1loop_ladder.py` | create | ladder smoke + undo check |
| `docs/ps1loop-findings.md` | create | drafted ps_1loop issues (not posted) |
| `CHANGELOG.md` | modify | dated entry with measured numbers |

---

### Task 0: Workspace, dependency, job runner

**Files:**
- Create: `slurm/ps1loop-tests.sbatch`, `slurm/run-ps1loop-pytest.sh`, `slurm/run-ps1loop-fast-suite.sh`
- Copy in: `CONTEXT.md`, the spec and this plan (from `/home/n2minh/clax`, where they are untracked)

**Interfaces:**
- Produces: `slurm/run-ps1loop-pytest.sh <pytest args>` (blocks; exit code = pytest's), `slurm/run-ps1loop-fast-suite.sh` (no args). Worktree `/home/n2minh/clax-ps1loop` on branch `feat/ps1loop-cross-validation`.

- [ ] **Step 1: Create the branch and worktree from fork `main` over HTTPS**

This node has no SSH key. Use `gh`'s token for this one fetch only (no global config change):

```bash
cd /home/n2minh/clax
git -c credential.helper='!gh auth git-credential' fetch https://github.com/MinhMPA/clax-pt.git main:refs/remotes/fork-https/main
git worktree add -b feat/ps1loop-cross-validation /home/n2minh/clax-ps1loop refs/remotes/fork-https/main
cd /home/n2minh/clax-ps1loop
test -f tests/ept_campaign_utils.py && ls reference_data/classpt/lcdm_fiducial/ | head
```
Expected: the `ls` lists `z0.000_ap_omfid0.31_cb.npz`, `z0.380_ap_omfid0.31_cb.npz`, `z0.800_ap_omfid0.31_cb.npz` (among others). If `tests/ept_campaign_utils.py` is missing, stop: fork `main` lacks PR #8, so ask the user.

- [ ] **Step 2: Copy in the glossary, spec and plan**

```bash
cp /home/n2minh/clax/CONTEXT.md /home/n2minh/clax-ps1loop/CONTEXT.md
mkdir -p /home/n2minh/clax-ps1loop/docs/superpowers/specs /home/n2minh/clax-ps1loop/docs/superpowers/plans
cp /home/n2minh/clax/docs/superpowers/specs/2026-10-05-clax-pt-vs-ps1loop-validation-design.md /home/n2minh/clax-ps1loop/docs/superpowers/specs/
cp /home/n2minh/clax/docs/superpowers/plans/2026-10-05-clax-pt-vs-ps1loop-validation.md /home/n2minh/clax-ps1loop/docs/superpowers/plans/
```

- [ ] **Step 3: Install ps_1loop into env `clax` without touching JAX**

```bash
eval "$(micromamba shell hook --shell bash)"; micromamba activate clax
JAX_BEFORE=$(pip freeze | grep -iE '^(jax|jaxlib)==')
pip install --no-deps -e /home/n2minh/ps_1loop_jax-for-pfs
JAX_PLATFORMS=cpu python -c "import quadax, interpax, sympy" || pip install --no-deps quadax interpax sympy mpmath
[ "$(pip freeze | grep -iE '^(jax|jaxlib)==')" = "$JAX_BEFORE" ] && echo "JAX UNCHANGED"
JAX_PLATFORMS=cpu python -c "import ps_1loop_jax; print('ps_1loop_jax', ps_1loop_jax.__file__)"
git -C /home/n2minh/ps_1loop_jax-for-pfs rev-parse --short HEAD
```
Expected: `JAX UNCHANGED`; the import prints a path under `/home/n2minh/ps_1loop_jax-for-pfs/src/`; the commit is `270b2b5`, unless the fork has moved on, in which case record the new commit in the CHANGELOG entry (Task 10).

- [ ] **Step 4: Write the job script**

Create `slurm/ps1loop-tests.sbatch`:

```bash
#!/bin/bash -l
#SBATCH --job-name=ps1loop-tests
#SBATCH --output=/lustre/work/n2minh/std/clax/ps1loop/%x.out.%j
#SBATCH --error=/lustre/work/n2minh/std/clax/ps1loop/%x.err.%j
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=125GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=igpu01,igpu02,igpu03,igpu04,igpu05,igpu06,igpu07,igpu08
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nhat.minh.nguyen@ipmu.jp
# clax-pt vs ps_1loop_jax validation (docs/superpowers/specs/2026-10-05-clax-pt-vs-ps1loop-validation-design.md).
# One pytest invocation; arguments come from $PYTEST_ARGS. No git network operations here
# (compute nodes have no SSH key). CPU by default, as in slurm/ptval-fast-suite.sbatch
# (Ruling 50: small problems are latency-bound on the V100); set JAX_PLATFORMS=cuda to override.
set -euo pipefail
eval "$(micromamba shell hook --shell bash)"
micromamba activate clax
NVIDIA_SP=$(python -c "import site; print(site.getsitepackages()[0])")/nvidia
for d in "${NVIDIA_SP}"/*/lib; do
    export LD_LIBRARY_PATH="${d}:${LD_LIBRARY_PATH:-}"
done
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
REPO=${CLAX_REPO:-/home/n2minh/clax-ps1loop}
cd "$REPO"
export PYTHONPATH="$REPO"
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
echo "HEAD: $(git -C "$REPO" rev-parse --short HEAD)  ps_1loop: $(git -C /home/n2minh/ps_1loop_jax-for-pfs rev-parse --short HEAD)"
git -C "$REPO" status --porcelain | head -20
python -c "import jax, ps_1loop_jax; print('devices', jax.devices())"
python -m pytest ${PYTEST_ARGS:?set PYTEST_ARGS} 2>&1 | tail -n 60
test "${PIPESTATUS[0]}" -eq 0 && echo PASS
```

- [ ] **Step 5: Write the login-node wrappers**

Create `slurm/run-ps1loop-pytest.sh`:

```bash
#!/bin/bash
# Usage (login node): slurm/run-ps1loop-pytest.sh <pytest args...>
# Submits slurm/ps1loop-tests.sbatch for THIS worktree, waits, prints the tail and ERROR lines,
# and exits with the job's status.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=/lustre/work/n2minh/std/clax/ps1loop
mkdir -p "$OUT"
jid=$(sbatch --parsable --wait --export=ALL,CLAX_REPO="$PWD",PYTEST_ARGS="$*" slurm/ps1loop-tests.sbatch)
rc=$?
jid=${jid%%;*}
echo "job $jid exit $rc"
tail -n 25 "$OUT/ps1loop-tests.out.$jid"
grep -h "ERROR" "$OUT/ps1loop-tests.out.$jid" | head -n 20
exit $rc
```

Create `slurm/run-ps1loop-fast-suite.sh`:

```bash
#!/bin/bash
# The CLAUDE.md pre-commit gate: pytest tests/ --fast -x -q, with the campaign's documented
# deselect (pre-existing max-steps failure, CHANGELOG Aug 29 and Sep 1, 2026; issue #30 item 5).
exec "$(dirname "$0")/run-ps1loop-pytest.sh" tests/ --fast -x -q -p no:cacheprovider \
  --deselect tests/test_solver_selection.py::TestRosenbrockPk::test_pk_rosenbrock_vs_kvaerno5
```

```bash
chmod +x slurm/run-ps1loop-pytest.sh slurm/run-ps1loop-fast-suite.sh
```

- [ ] **Step 6: Smoke-run the runner**

Run: `slurm/run-ps1loop-pytest.sh tests/test_ept_multicosmo.py::test_window_and_thresholds -q`
Expected: `1 passed`, `PASS`, exit 0, and the header line shows a `ps_1loop:` commit.

- [ ] **Step 7: Fast suite, then commit**

Run: `slurm/run-ps1loop-fast-suite.sh`
Expected: `PASS`.

```bash
git add CONTEXT.md docs/superpowers/specs/2026-10-05-clax-pt-vs-ps1loop-validation-design.md \
  docs/superpowers/plans/2026-10-05-clax-pt-vs-ps1loop-validation.md \
  slurm/ps1loop-tests.sbatch slurm/run-ps1loop-pytest.sh slurm/run-ps1loop-fast-suite.sh
git commit -m "chore(ps1loop): spec, plan, glossary and igpu job runner for the ps_1loop validation

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 1: PFS survey table, bias point, reference registry

**Files:**
- Create: `reference_data/pfs/takada2014_table2.json`, `tests/pfs_survey.py`
- Modify: `scripts/validation_cosmologies.py` (after the `FAST_Z = 0.38` line)
- Test: `tests/test_pfs_survey.py`

**Interfaces:**
- Produces (`tests/pfs_survey.py`): `PFSSlice(z_min, z_max, V, nbar, b_g, N_g_per_field)` with property `z_mid`; `load_slices() -> tuple[PFSSlice, ...]`; `AREA_DEG2`, `FOV_DEG2`, `C_REF = 2.0`, `CFOG_REF = 500.0`; `bias_point(z, *, c=C_REF) -> dict` keyed by `ps1loop_bridge.THETA_NAMES`. Task 8 adds the covariance functions.
- Produces (`scripts/validation_cosmologies.py`): `PFS_Z_LIST`, `PFS_CASES`, `IRNO_CASES`, `IRNO_TAG`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pfs_survey.py`:

```python
# tests/test_pfs_survey.py
"""PFS survey bookkeeping (spec §3, §5.6): Takada et al. (2014) Table 2 transcription and
the error-bar-gate bias point. Exempt from the multi-cosmology rule: survey numbers and
bias relations, no cosmology-dependent physics."""
from __future__ import annotations

import pytest

from scripts import validation_cosmologies as vc
from tests import pfs_survey as pfs


def test_seven_contiguous_slices_from_0p6_to_2p4():
    s = pfs.load_slices()
    assert len(s) == 7
    assert s[0].z_min == 0.6 and s[-1].z_max == 2.4
    assert all(a.z_max == b.z_min for a, b in zip(s, s[1:]))


def test_bias_column_is_takada_relation_at_midpoint():
    for sl in pfs.load_slices():
        assert abs(sl.b_g - (0.9 + 0.4 * sl.z_mid)) < 0.006, sl


def test_nbar_is_galaxies_per_field_over_volume():
    # n = N_g per field x (1464 / 1.098) fields / V; Table 2 rounds n to 2 significant figures.
    for sl in pfs.load_slices():
        nbar = sl.N_g_per_field * (pfs.AREA_DEG2 / pfs.FOV_DEG2) / sl.V
        assert abs(nbar / sl.nbar - 1.0) < 0.03, (sl.z_min, nbar, sl.nbar)


def test_units_and_volume_scale():
    sl = pfs.load_slices()[0]
    assert sl.V == pytest.approx(0.59e9)          # (Mpc/h)^3, from 0.59 (Gpc/h)^3
    assert sl.nbar == pytest.approx(1.9e-4)       # (h/Mpc)^3


def test_midpoints_are_the_reference_registry():
    assert tuple(sl.z_mid for sl in pfs.load_slices()) == vc.PFS_Z_LIST
    assert set(vc.PFS_CASES) <= set(vc.distinct_cases())
    assert set(vc.IRNO_CASES) <= set(vc.distinct_cases())


def test_bias_point_coevolution():
    z = 0.7
    th = pfs.bias_point(z)
    b1 = 0.9 + 0.4 * z
    assert th["b1"] == pytest.approx(b1)
    assert th["b2"] == pytest.approx(0.412 - 2.143 * b1 + 0.929 * b1**2 + 0.008 * b1**3)
    assert th["bG2"] == pytest.approx(-2.0 / 7.0 * (b1 - 1.0))
    assert th["bGamma3"] == pytest.approx(23.0 / 42.0 * (b1 - 1.0))
    assert th["c0"] == th["c2"] == th["c4"] == pfs.C_REF
    assert th["cfog"] == pfs.CFOG_REF and th["P_shot"] == 1.0
    assert th["a0"] == th["a2"] == 0.0
    assert pfs.bias_point(z, c=0.0)["c0"] == 0.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `slurm/run-ps1loop-pytest.sh tests/test_pfs_survey.py -q`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'tests.pfs_survey'`.

- [ ] **Step 3: Write the table, the module and the registry constants**

Create `reference_data/pfs/takada2014_table2.json`:

```json
{
  "source": "Takada et al. 2014, PASJ 66, R1 (arXiv:1206.0737), Table 2 'PFS cosmology survey parameters'",
  "area_deg2": 1464.0,
  "fov_deg2": 1.098,
  "bias_relation": "b_g(z) = 0.9 + 0.4 z (Orsi et al. 2010 fit, sigma_8 = 0.9 normalisation)",
  "units": {"V": "(Gpc/h)^3", "nbar": "1e-4 (h/Mpc)^3"},
  "slices": [
    {"z_min": 0.6, "z_max": 0.8, "V": 0.59, "N_g_per_field": 85,  "nbar": 1.9, "b_g": 1.18},
    {"z_min": 0.8, "z_max": 1.0, "V": 0.79, "N_g_per_field": 358, "nbar": 6.0, "b_g": 1.26},
    {"z_min": 1.0, "z_max": 1.2, "V": 0.96, "N_g_per_field": 420, "nbar": 5.8, "b_g": 1.34},
    {"z_min": 1.2, "z_max": 1.4, "V": 1.09, "N_g_per_field": 640, "nbar": 7.8, "b_g": 1.42},
    {"z_min": 1.4, "z_max": 1.6, "V": 1.19, "N_g_per_field": 491, "nbar": 5.5, "b_g": 1.50},
    {"z_min": 1.6, "z_max": 2.0, "V": 2.58, "N_g_per_field": 598, "nbar": 3.1, "b_g": 1.62},
    {"z_min": 2.0, "z_max": 2.4, "V": 2.71, "N_g_per_field": 539, "nbar": 2.7, "b_g": 1.78}
  ]
}
```

Create `tests/pfs_survey.py`:

```python
# tests/pfs_survey.py
"""PFS survey inputs for the ps_1loop error-bar gate (spec §3, §5.6).

Takada et al. (2014, PASJ 66, R1; arXiv:1206.0737) Table 2, stored verbatim in
reference_data/pfs/takada2014_table2.json and converted here to (Mpc/h)^3 and (h/Mpc)^3.
Each slice is evaluated at its midpoint (the paper defines no effective redshift).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "reference_data" / "pfs" / "takada2014_table2.json"
_T = json.loads(TABLE.read_text())
AREA_DEG2: float = _T["area_deg2"]
FOV_DEG2: float = _T["fov_deg2"]

# Error-bar-gate bias point (spec §5.6).
C_REF = 2.0       # (Mpc/h)^2 -- PLACEHOLDER k^2 counterterm amplitude, c0 = c2 = c4; replace with
                  # the PFS forecast's prior centres. ~10% of the z=0.7 monopole at k = 0.2 h/Mpc.
CFOG_REF = 500.0  # (Mpc/h)^4 -- = CLASS-PT b4 of the 42-file reference set (bias_json)


@dataclass(frozen=True)
class PFSSlice:
    z_min: float
    z_max: float
    V: float              # comoving volume, (Mpc/h)^3
    nbar: float           # number density, (h/Mpc)^3
    b_g: float            # linear bias (Table 2 column)
    N_g_per_field: int    # [OII] galaxies per field

    @property
    def z_mid(self) -> float:
        return round(0.5 * (self.z_min + self.z_max), 3)


def load_slices() -> tuple[PFSSlice, ...]:
    return tuple(PFSSlice(z_min=r["z_min"], z_max=r["z_max"], V=r["V"] * 1e9,
                          nbar=r["nbar"] * 1e-4, b_g=r["b_g"], N_g_per_field=r["N_g_per_field"])
                 for r in _T["slices"])


def bias_point(z: float, *, c: float = C_REF) -> dict[str, float]:
    """ps_1loop-basis parameters at redshift z (spec §5.6): Takada b1, Lazeyras et al. (2016)
    b2(b1), coevolution bG2 and bGamma3, counterterms c, fingers-of-god CFOG_REF, Poisson
    P_shot = 1 (with ndens = nbar), a0 = a2 = 0."""
    b1 = 0.9 + 0.4 * z
    return {"b1": b1,
            "b2": 0.412 - 2.143 * b1 + 0.929 * b1**2 + 0.008 * b1**3,
            "bG2": -2.0 / 7.0 * (b1 - 1.0),
            "bGamma3": 23.0 / 42.0 * (b1 - 1.0),
            "c0": c, "c2": c, "c4": c, "cfog": CFOG_REF,
            "P_shot": 1.0, "a0": 0.0, "a2": 0.0}
```

In `scripts/validation_cosmologies.py`, directly after the line `FAST_Z = 0.38`, add:

```python
# ps_1loop second-oracle validation (docs/superpowers/specs/2026-10-05-clax-pt-vs-ps1loop-validation-design.md §3):
# PFS slices = midpoints of Takada et al. (2014) Table 2; IR-off CLASS-PT tie-breakers at z = 0.38.
PFS_Z_LIST = (0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.2)
PFS_CASES = ("lcdm_fiducial", "omega_cdm_low", "massive_nu_015", "massive_nu_030", "w0wa_m07_m10")
IRNO_CASES = ("lcdm_fiducial", "massive_nu_030", "w0wa_m07_m10")
IRNO_TAG = "irno"
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `slurm/run-ps1loop-pytest.sh tests/test_pfs_survey.py -q`
Expected: `6 passed`.

- [ ] **Step 5: Fast suite, then commit**

Run: `slurm/run-ps1loop-fast-suite.sh` → `PASS`.

```bash
git add reference_data/pfs/takada2014_table2.json tests/pfs_survey.py tests/test_pfs_survey.py scripts/validation_cosmologies.py
git commit -m "test(pfs): Takada 2014 Table 2, error-bar-gate bias point, PFS/IR-off reference registry

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: CLASS-PT references at the PFS slices and with IR off

**Files:**
- Create: `slurm/classpt-refgen-pfs.sbatch`, `tests/test_classpt_pfs_references.py`
- Add: 38 files under `reference_data/classpt/`, regenerated `reference_data/classpt/MANIFEST.md`

**Interfaces:**
- Consumes: `vc.PFS_CASES`, `vc.PFS_Z_LIST`, `vc.IRNO_CASES`, `vc.IRNO_TAG`, `vc.reference_path(case, z, *, ap, tag)`.
- Produces: files `z{z:.3f}_ap_omfid0.31_cb.npz` (PFS) and `z0.380_noap_cb_irno.npz` (IR off), with keys `k_h, pk_lin, fz, rs_d, h, hratio, Dratio, Pd2d2_0, pk_mult, params_json, bias_json, ap, cb, omfid` plus the nine spectra.

- [ ] **Step 1: Write the failing test**

Create `tests/test_classpt_pfs_references.py`:

```python
# tests/test_classpt_pfs_references.py
"""The 38 new CLASS-PT reference files (spec §3): 5 cosmologies x 7 PFS slices with AP,
and 3 IR-off files at z = 0.38. Presence is asserted, never skipped -- a missing file must
not let the error-bar gate skip into a pass (Review Focus 4). Multi-cosmology rule:
5 and 3 cosmologies by construction."""
from __future__ import annotations

import json

import numpy as np
import pytest

from scripts import validation_cosmologies as vc
from tests import ept_campaign_utils as cu


def _paths():
    pfs = [vc.reference_path(c, z) for c in vc.PFS_CASES for z in vc.PFS_Z_LIST]
    irno = [vc.reference_path(c, 0.38, ap=False, tag=vc.IRNO_TAG) for c in vc.IRNO_CASES]
    return pfs + irno


def test_all_new_references_present():
    missing = [str(p.relative_to(cu.ROOT)) for p in _paths() if not p.exists()]
    assert not missing, f"ERROR {len(missing)} reference files missing: {missing[:5]} -- run slurm/classpt-refgen-pfs.sbatch"


@pytest.mark.parametrize("case", vc.PFS_CASES)
def test_pfs_reference_flags_and_growth(case):
    fz = []
    for z in vc.PFS_Z_LIST:
        ref = cu.load_reference(case, z)
        assert ref is not None, f"ERROR missing {case} z={z}"
        assert bool(ref["ap"]) and bool(ref["cb"]) and float(ref["omfid"]) == vc.OMFID
        assert abs(float(ref["z"]) - z) < 1e-12
        assert json.loads(str(ref["params_json"]))["IR resummation"] == "Yes"
        assert str(ref["kh_convention"]).startswith("h/Mpc")
        assert np.all(np.isfinite(ref["pk_mult"])) and np.all(ref["pk_lin"] > 0)
        fz.append(float(ref["fz"]))
    assert all(0.0 < a < b < 1.0 for a, b in zip(fz, fz[1:])), f"ERROR f(z) not increasing for {case}: {fz}"


@pytest.mark.parametrize("case", vc.IRNO_CASES)
def test_irno_reference_flags(case):
    ref = cu.load_reference(case, 0.38, ap=False, tag=vc.IRNO_TAG)
    assert ref is not None, f"ERROR missing IR-off file for {case}"
    assert not bool(ref["ap"]) and bool(ref["cb"])
    assert json.loads(str(ref["params_json"]))["IR resummation"] == "No"
    ap_file = cu.load_reference(case, 0.38)
    np.testing.assert_array_equal(ref["k_h"], ap_file["k_h"])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `slurm/run-ps1loop-pytest.sh tests/test_classpt_pfs_references.py -q`
Expected: FAIL — `ERROR 38 reference files missing: ...` (and the per-case tests fail with `ERROR missing`).

- [ ] **Step 3: Write the generation job**

Create `slurm/classpt-refgen-pfs.sbatch` (same CPU template as `slurm/classpt-refgen.sbatch`):

```bash
#!/bin/bash -l
#SBATCH --job-name=classpt-refgen-pfs
#SBATCH --output=/lustre/work/n2minh/std/clax/ps1loop/%x.out.%j
#SBATCH --error=/lustre/work/n2minh/std/clax/ps1loop/%x.err.%j
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nhat.minh.nguyen@ipmu.jp
# ps_1loop validation references (spec §3): vc.PFS_CASES x vc.PFS_Z_LIST at the campaign
# default (ap=yes, cb=yes, omfid=0.31, bias=fiducial), one z per call (Ruling 19), plus
# vc.IRNO_CASES at z=0.38 with IR resummation off and AP off. CPU-only (env classpt).
# `main` is hidden from plain sinfo but schedules (Ruling 5). No git network operations.
set -uo pipefail
eval "$(micromamba shell hook --shell bash)"
micromamba activate classpt
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
REPO=${CLAX_REPO:-/home/n2minh/clax-ps1loop}
cd "$REPO"
export PYTHONPATH="$REPO"
echo "Node: $(hostname)  Commit: $(git rev-parse --short HEAD)"
python -c "import classy, numpy; print('classy', classy.__file__, 'numpy', numpy.__version__)"
GEN="python scripts/generate_classpt_reference.py"
VC="from scripts import validation_cosmologies as vc"
FAILED=()
for c in $(python -c "$VC; print(' '.join(vc.PFS_CASES))"); do
  for z in $(python -c "$VC; print(' '.join(str(z) for z in vc.PFS_Z_LIST))"); do
    $GEN --cosmology "$c" --z-list "$z" || FAILED+=("$c:z$z")
  done
done
for c in $(python -c "$VC; print(' '.join(vc.IRNO_CASES))"); do
  $GEN --cosmology "$c" --z-list 0.38 --ap no --tag irno \
       --class-extra '{"IR resummation": "No"}' || FAILED+=("$c:irno")
done
python scripts/write_classpt_manifest.py || FAILED+=("manifest")
echo "FAILED: ${FAILED[*]:-none}"
[ ${#FAILED[@]} -eq 0 ]
```

- [ ] **Step 4: Run the job and the test**

```bash
mkdir -p /lustre/work/n2minh/std/clax/ps1loop
sbatch --wait --export=ALL,CLAX_REPO=/home/n2minh/clax-ps1loop slurm/classpt-refgen-pfs.sbatch
grep -h "FAILED:" /lustre/work/n2minh/std/clax/ps1loop/classpt-refgen-pfs.out.* | tail -n 1
git status --porcelain reference_data/classpt | wc -l
```
Expected: `FAILED: none`; 39 changed paths (38 `.npz` + `MANIFEST.md`). If a z ≥ 1.8 run fails, read the job's error file. CLASS-PT's own `z_max_pk` limit is the first suspect; a fix goes through `--class-extra` and is recorded in `MANIFEST.md`, never by dropping the slice.

Run: `slurm/run-ps1loop-pytest.sh tests/test_classpt_pfs_references.py -q`
Expected: `9 passed`.

- [ ] **Step 5: Fast suite, then commit**

Run: `slurm/run-ps1loop-fast-suite.sh` → `PASS`.

```bash
git add slurm/classpt-refgen-pfs.sbatch tests/test_classpt_pfs_references.py reference_data/classpt
git commit -m "test(classpt): references at the 7 PFS slices x 5 cosmologies and 3 IR-off files

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: The bridge — shared inputs, evaluators, translation, stencils, gate

**Files:**
- Create: `tests/ps1loop_bridge.py`
- Test: `tests/test_ps1loop_bridge.py`

**Interfaces:**
- Consumes: `clax.ept.compute_ept(pk_lin_h, k_h, h, f, prec, rs_h=, hratio=, Dratio=)`, `EPTPrecisionParams(ir_resummation=)`, `pk_gg_l0(e, b1, b2, bG2, bGamma3, cs0=, Pshot=, b4=, a0=, a2=)`, `pk_gg_l2(..., cs2=, b4=, a2=)`, `pk_gg_l4(..., cs4=, b4=)`; `scripts.classpt_assembly.assemble_from_pm(pm, h, fz, kh, bias, Pd2d2_0)`; `tests.ept_campaign_utils` (`window`, `ROOT`, `_git_sha`, `_sanitize`, `require_reference`); ps_1loop `PowerSpectrum1Loop(do_irres, rbao, ks, nfft)` with `get_pkmu`, `get_pkmu_ref`, `get_pk_ell_ref`.
- Produces (used by Tasks 4–9):
  - constants `THETA_NAMES`, `ELLS = (0, 2, 4)`, `CHANNELS = ("l0", "l2", "l4")`, `COEF_THRESHOLDS`, `K_NL = 0.45`, `NDENS_PROBE = 3e-4`, `IR_KS = 0.2`, `STRUCTURAL_ZERO = 1e-12`, `EXACT_MONOMIALS`, `MODEL_DIFF_MONOMIALS`, `NO_TIEBREAK`, `GL_MU`, `GL_W`, `LOG_DIR`, `ERROR_LOG`
  - `zero_theta() -> dict`, `label(mono) -> str`, `legendre_np(ell, mu)`, `project(vals, ell)` (40-node GL, last axis), `b4_poly(f, b1) -> dict`
  - `shared_inputs(ref) -> dict` (keys `k_h, pk, h, f, rbao, hratio, Dratio, Pd2d2_0, pk_mult`), `window_k(inp) -> np.ndarray`, `ap_alphas(hratio, Dratio) -> (a_perp, a_para)`, `ap_volume(hratio, Dratio) -> float`, `has_ap(inp) -> bool`
  - `ps_model(rbao, *, do_irres=True, nfft=256, mu_grid=51)` (cached per arguments)
  - `ps_params(theta, *, h, f, ndens) -> dict`, `ps_multipoles_jax(model, k_h, pk, h, f, a_perp, a_para, theta, k, *, ap, ndens, num=256) -> jnp (3, Nk)`
  - `to_clax(theta, *, f, ndens, Pd2d2_0, hratio, Dratio) -> dict` (keys `b1 b2 bG2 bGamma3 cs0 cs2 cs4 cs b4 Pshot a0 a2`)
  - `clax_ept(inp, *, ir=True, pk=None, f=None, hratio=None, Dratio=None)`, `clax_multipoles_jax(e, cb) -> jnp (3, Nk_full)`, `classpt_multipoles(inp, cb) -> np (3, Nk_full)`
  - batch evaluators `ps_batch(model, inp, kw, *, ndens=NDENS_PROBE, num=256)`, `clax_batch(e, inp, mask, *, ndens=NDENS_PROBE)`, `classpt_batch(inp, mask, *, ndens=NDENS_PROBE)` — each returns `F(thetas: (B, 11)) -> np (B, 3, Nw)`
  - `stencil(mono) -> list[(theta, weight)]`, `coefficients(F, monomials) -> {label: np (M, Nk)}`
  - `coefficient_triplet(inp, monomials, *, ir=True, ndens=NDENS_PROBE) -> (res, kw)` with `res = {"ps", "clax", "classpt"}`
  - `spectra_at(inp, theta, *, ndens) -> (ps (3, Nw), clax (3, Nw), kw)`
  - `coef_error(got, ref, kw, *, scale) -> {"err", "k", "metric"}`, `gate(res, kw, *, channels=CHANNELS, thresholds=COEF_THRESHOLDS) -> (errs, lines)`, `attribute(res, lab, i, kw, scale, thr) -> str`, `worst(errs) -> str`
  - `ps1loop_commit() -> str`, `log_record(*, layer, case, z, preset, errors, extra=None)`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ps1loop_bridge.py`:

```python
# tests/test_ps1loop_bridge.py
"""Unit tests for tests/ps1loop_bridge.py (spec §4, §5.1).

Exempt from the multi-cosmology rule: parameter-space stencil algebra, AP and Legendre
algebra, caching and API parity. The fixture-based tests use lcdm_fiducial z = 0.38 only
to exercise ps_1loop's own API against itself, not to compare physics."""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("ps_1loop_jax")
import jax.numpy as jnp

from tests import ept_campaign_utils as cu
from tests import ps1loop_bridge as br


@pytest.fixture(scope="module")
def fid():
    return cu.require_reference("lcdm_fiducial", 0.38)


def test_stencil_recovers_every_monomial_exactly():
    monos = br.EXACT_MONOMIALS + br.MODEL_DIFF_MONOMIALS
    rng = np.random.default_rng(0)
    truth = {br.label(m): rng.normal(size=(3, 5)) for m in monos}

    def F(thetas):
        out = []
        for th in np.asarray(thetas):
            t = dict(zip(br.THETA_NAMES, th))
            val = np.zeros((3, 5))
            for m in monos:
                val = val + truth[br.label(m)] * np.prod([1.0 if v == "1" else t[v] for v in m])
            out.append(val)
        return np.stack(out)

    got = br.coefficients(F, monos)
    for lab, arr in truth.items():
        np.testing.assert_allclose(got[lab], arr, rtol=0, atol=1e-12, err_msg=lab)


def test_ap_alphas_reproduce_clax_true_coordinates():
    hratio, Dratio = 1.031, 0.974
    a_perp, a_para = br.ap_alphas(hratio, Dratio)
    mu, k = np.linspace(-1.0, 1.0, 11), 0.2
    fac = np.sqrt(1.0 + mu**2 * ((a_perp / a_para) ** 2 - 1.0))          # ps_1loop.py:332-334
    mu_ps, k_ps = mu * (a_perp / a_para) / fac, k * fac / a_perp
    ap_fac = np.sqrt(1.0 / Dratio**2 + (hratio**2 - 1.0 / Dratio**2) * mu**2)   # ept.py:1138
    mu_cl, k_cl = mu * hratio / ap_fac, k * ap_fac                            # ept.py:1139-1140
    np.testing.assert_allclose(mu_ps, mu_cl, rtol=1e-14)
    np.testing.assert_allclose(k_ps, k_cl, rtol=1e-14)
    assert br.ap_volume(hratio, Dratio) == pytest.approx(1.0 / (a_perp**2 * a_para), rel=1e-14)


def test_counterterm_map_exact_for_isotropic_spectrum():
    f, c = 0.81, {"c0": 1.3, "c2": -0.7, "c4": 2.1}
    mu = br.GL_MU
    ps = {l: br.project(-2.0 * (c["c0"] + c["c2"] * f * mu**2 + c["c4"] * f**2 * mu**4), l) for l in br.ELLS}
    cb = br.to_clax({**br.zero_theta(), **c}, f=f, ndens=1.0, Pd2d2_0=0.0, hratio=1.0, Dratio=1.0)
    pctr = {l: -br.project(f ** (l // 2) * mu**l, l) for l in br.ELLS}     # ept.py:1207-1209, k = P = 1
    for l, cs in ((0, cb["cs0"]), (2, cb["cs2"]), (4, cb["cs4"])):
        assert 2.0 * cs * pctr[l] == pytest.approx(ps[l], rel=1e-13), l


def test_cfog_is_b4_for_isotropic_spectrum():
    f, b1, cfog, k = 0.81, 1.4, 500.0, 0.2
    mu = br.GL_MU
    ps = {l: br.project(-cfog * f**4 * k**4 * mu**4 * (b1 + f * mu**2) ** 2, l) for l in br.ELLS}
    pctr4 = -br.project(k**2 * f**2 * mu**4, 4)
    poly = br.b4_poly(f, b1)
    for l in br.ELLS:
        assert f**2 * cfog * k**2 * poly[l] * (35.0 / 8.0) * pctr4 == pytest.approx(ps[l], rel=1e-12), l


def test_to_clax_stochastic_and_shot_noise():
    th = {**br.zero_theta(), "P_shot": 1.0, "a0": 0.3, "a2": -0.2, "b2": 0.5}
    hr, Dr, nd = 1.03, 0.97, 2.5e-4
    cb = br.to_clax(th, f=0.8, ndens=nd, Pd2d2_0=40.0, hratio=hr, Dratio=Dr)
    V = hr / Dr**2
    assert cb["Pshot"] == pytest.approx(V / nd - 0.25 * 0.5**2 * 40.0)
    assert cb["a2"] == pytest.approx(V * hr**2 * -0.2 / nd)
    kt2_over_k2 = 1.0 / Dr**2 + (hr**2 - 1.0 / Dr**2) * br.GL_MU**2
    assert cb["a0"] == pytest.approx(br.project(V * 0.3 / nd * kt2_over_k2, 0), rel=1e-13)


def test_ps_model_cache_keys_on_rbao_and_overrides_before_first_call():
    m1, m2, m3 = br.ps_model(99.08), br.ps_model(99.08), br.ps_model(110.0)
    assert m1 is m2 and m1 is not m3
    assert m1.rbao == 99.08 and m3.rbao == 110.0 and m1.ks == br.IR_KS
    assert br.ps_model(99.08, mu_grid=201)._mu.shape == (201,)
    assert br.ps_model(99.08)._mu.shape == (51,)


def test_rbao_is_rs_times_h(fid):
    inp = br.shared_inputs(fid)
    assert inp["rbao"] == pytest.approx(float(fid["rs_d"]) * float(fid["h"]), rel=1e-9)
    assert inp["rbao"] == pytest.approx(99.08, abs=0.05)        # 147.10 Mpc x 0.6736


def test_rbao_reaches_the_computation(fid):
    inp = br.shared_inputs(fid)
    kw = br.window_k(inp)
    th = np.array([[{**br.zero_theta(), "b1": 1.0}[n] for n in br.THETA_NAMES]])
    a = br.ps_batch(br.ps_model(inp["rbao"]), inp, kw)(th)
    b = br.ps_batch(br.ps_model(110.0), inp, kw)(th)
    assert np.max(np.abs(a - b)) / np.max(np.abs(a)) > 1e-4


def test_three_in_one_projection_matches_get_pk_ell_ref(fid):
    inp = br.shared_inputs(fid)
    model, kw = br.ps_model(inp["rbao"]), br.window_k(inp)
    th = {**br.zero_theta(), "b1": 2.0, "b2": -0.4, "c0": 3.0, "cfog": 500.0, "P_shot": 1.0}
    a_perp, a_para = br.ap_alphas(inp["hratio"], inp["Dratio"])
    pk_data = {"k": jnp.asarray(inp["k_h"]), "pk": jnp.asarray(inp["pk"])}
    params = br.ps_params(th, h=inp["h"], f=inp["f"], ndens=br.NDENS_PROBE)
    ours = br.ps_batch(model, inp, kw)(np.array([[th[n] for n in br.THETA_NAMES]]))[0]
    for i, l in enumerate(br.ELLS):
        ref = np.asarray(model.get_pk_ell_ref(jnp.asarray(kw), l, a_perp, a_para, pk_data, params))
        np.testing.assert_allclose(ours[i], ref, rtol=1e-10, atol=0, err_msg=f"l={l}")


def test_coef_error_metrics():
    kw = np.linspace(0.01, 0.3, 5)
    e = br.coef_error(np.full(5, 1.01), np.ones(5), kw, scale=1.0)
    assert e["metric"] == "self" and e["err"] == pytest.approx(0.01)
    e = br.coef_error(np.full(5, 2e-3), np.zeros(5), kw, scale=1.0)
    assert e["metric"] == "monomial-scale" and e["err"] == pytest.approx(2e-3)
    bad = np.ones(5)
    bad[2] = np.nan
    e = br.coef_error(bad, np.ones(5), kw, scale=1.0)
    assert e["metric"] == "non-finite" and not np.isfinite(e["err"]) and e["k"] == kw[2]


def test_gate_flags_non_finite_and_attributes():
    kw = np.linspace(0.01, 0.3, 4)
    ps = np.ones((3, 4))
    ps[1, 2] = np.nan
    res = {"ps": {"b1": ps}, "clax": {"b1": np.ones((3, 4))}, "classpt": {"b1": np.ones((3, 4))}}
    errs, lines = br.gate(res, kw)
    assert len(lines) == 1 and lines[0].startswith("ERROR b1:l2") and "non-finite" in lines[0]
    assert "vs CLASS-PT" in lines[0]


def test_log_record_carries_both_commits(tmp_path, monkeypatch):
    monkeypatch.setattr(br, "LOG_DIR", tmp_path)
    monkeypatch.setattr(br, "ERROR_LOG", tmp_path / "errors.jsonl")
    br.log_record(layer="coef", case="lcdm_fiducial", z=0.38, preset="exact",
                  errors={"b1:l0": {"err": 1e-3, "k": 0.2, "metric": "self"}})
    rec = json.loads((tmp_path / "errors.jsonl").read_text().splitlines()[-1])
    assert rec["layer"] == "coef" and rec["ps1loop_sha"] != "" and "git_sha" in rec
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `slurm/run-ps1loop-pytest.sh tests/test_ps1loop_bridge.py -q`
Expected: FAIL at collection with `ImportError: cannot import name 'ps1loop_bridge' from 'tests'`.

- [ ] **Step 3: Write the bridge**

Create `tests/ps1loop_bridge.py`:

```python
# tests/ps1loop_bridge.py
"""Bridge for the clax-pt vs ps_1loop_jax second-oracle validation.

Spec: docs/superpowers/specs/2026-10-05-clax-pt-vs-ps1loop-validation-design.md (glossary:
CONTEXT.md). Shared-input comparison: both codes receive one CLASS-PT reference file's own
P_cb(k) [(Mpc/h)^3 on k_h in h/Mpc], growth rate f, r_s(z_d) h [Mpc/h] and AP ratios, so a
residual is one-loop numerics or a documented model difference (spec §4.2).

Conventions pinned here, each with its test in tests/test_ps1loop_bridge.py:
  * AP: ps_1loop alpha_perp = Dratio, alpha_para = 1/hratio, Jacobian V = hratio/Dratio^2
    (ps_1loop.py:328-344 vs ept.py:1135-1141).
  * Basis: THETA_NAMES is ps_1loop's; `to_clax` is the basis translation (spec §4.1).
  * Coefficient spectra: exact stencils on {-1, 0, 1} per parameter. P_l has degree <= 2 in
    every parameter, so the stencils have no truncation error -- only round-off.
"""
from __future__ import annotations

import datetime as _dt
import functools
import itertools
import json
import subprocess
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from clax.ept import EPTPrecisionParams, compute_ept, pk_gg_l0, pk_gg_l2, pk_gg_l4
from scripts import classpt_assembly as ca
from tests import ept_campaign_utils as cu

THETA_NAMES = ("b1", "b2", "bG2", "bGamma3", "c0", "c2", "c4", "cfog", "P_shot", "a0", "a2")
ELLS = (0, 2, 4)
CHANNELS = ("l0", "l2", "l4")
COEF_THRESHOLDS = {"l0": 0.01, "l2": 0.01, "l4": 0.02}   # = cu.THRESHOLDS (spec §5.1)
K_NL = 0.45            # h/Mpc; clax-pt hard-codes (k/0.45)^2 in a0/a2 (ept.py pk_gg_l0, classy.pyx:4910)
NDENS_PROBE = 3.0e-4   # (h/Mpc)^3; only rescales the stochastic monomials
IR_KS = 0.2            # h/Mpc; ps_1loop default k_s, = CLASS-PT's Sigma^2_BAO cut
STRUCTURAL_ZERO = 1e-12

EXACT_MONOMIALS = (("1",), ("b1",), ("b1", "b1"), ("b2",), ("b1", "b2"), ("b2", "b2"),
                   ("bG2",), ("b1", "bG2"), ("bG2", "bG2"), ("b2", "bG2"),
                   ("bGamma3",), ("b1", "bGamma3"), ("P_shot",), ("a2",))
MODEL_DIFF_MONOMIALS = (("c0",), ("c2",), ("c4",), ("cfog",), ("cfog", "b1"),
                        ("cfog", "b1", "b1"), ("a0",))
NO_TIEBREAK = {"a0", "a2"}   # CLASS-PT's twin has no scale-dependent stochastic rows

GL_MU, GL_W = np.polynomial.legendre.leggauss(40)

LOG_DIR = cu.ROOT / "test_logs" / "ps1loop"
ERROR_LOG = LOG_DIR / "errors.jsonl"


# --------------------------------------------------------------------------- algebra

def zero_theta() -> dict[str, float]:
    return {n: 0.0 for n in THETA_NAMES}


def label(mono: tuple[str, ...]) -> str:
    return "*".join(mono)


def legendre_np(ell: int, mu):
    mu = np.asarray(mu, dtype=float)
    return {0: np.ones_like(mu), 2: 0.5 * (3.0 * mu**2 - 1.0),
            4: (35.0 * mu**4 - 30.0 * mu**2 + 3.0) / 8.0}[ell]


def project(vals, ell: int):
    """(2l+1)/2 * int_{-1}^{1} L_l(mu) vals(mu) dmu on the 40 GL nodes (last axis)."""
    return (2 * ell + 1) / 2.0 * np.sum(GL_W * legendre_np(ell, GL_MU) * np.asarray(vals), axis=-1)


def b4_poly(f, b1) -> dict[int, float]:
    """The isotropic factors clax-pt multiplies b4 k^2 (35/8) Pk_ctr4 by (ept.py pk_gg_l0/l2/l4 P_b4)."""
    return {0: f**2 / 9.0 + 2.0 * f * b1 / 7.0 + b1**2 / 5.0,
            2: (4.0 / 693.0) * (70.0 * f**2 + 165.0 * f * b1 + 99.0 * b1**2),
            4: (8.0 / 5005.0) * (210.0 * f**2 + 390.0 * f * b1 + 143.0 * b1**2)}


# --------------------------------------------------------------------------- shared inputs

def shared_inputs(ref: dict) -> dict:
    h = float(ref["h"])
    return {"k_h": np.asarray(ref["k_h"], dtype=float), "pk": np.asarray(ref["pk_lin"], dtype=float),
            "h": h, "f": float(ref["fz"]),
            "rbao": round(float(ref["rs_d"]) * h, 9),           # r_s(z_d) h [Mpc/h], THIS cosmology
            "hratio": float(ref["hratio"]), "Dratio": float(ref["Dratio"]),
            "Pd2d2_0": float(ref["Pd2d2_0"]), "pk_mult": np.asarray(ref["pk_mult"], dtype=float)}


def window_k(inp: dict) -> np.ndarray:
    return inp["k_h"][cu.window(inp["k_h"])]


def ap_alphas(hratio, Dratio):
    """(alpha_perp, alpha_para) for ps_1loop from CLASS-PT/clax (hratio, Dratio)."""
    return Dratio, 1.0 / hratio


def ap_volume(hratio, Dratio):
    return hratio / Dratio**2                      # = 1/(alpha_perp^2 alpha_para), ept.py:1141


def has_ap(inp: dict) -> bool:
    return not (inp["hratio"] == 1.0 and inp["Dratio"] == 1.0)


# --------------------------------------------------------------------------- ps_1loop

@functools.lru_cache(maxsize=None)
def ps_model(rbao: float, *, do_irres: bool = True, nfft: int = 256, mu_grid: int = 51):
    """One PowerSpectrum1Loop per (rbao, settings). Its methods are jitted with
    static_argnames=['self'], so NOTHING may be mutated after the first call (Review Focus 1);
    the mu_grid override (spec §7 ladder) is applied here, before any call."""
    from ps_1loop_jax import PowerSpectrum1Loop
    model = PowerSpectrum1Loop(do_irres=do_irres, rbao=rbao, ks=IR_KS, nfft=nfft)
    if mu_grid != 51:
        model._mu = jnp.linspace(0.0, 1.0, mu_grid)   # TEST-ONLY: AP remap grid (ps_1loop.py:46)
    return model


def ps_params(theta: dict, *, h, f, ndens) -> dict:
    return {"h": h, "f": f,
            "bias": {n: theta[n] for n in ("b1", "b2", "bG2", "bGamma3")},
            "ctr": {n: theta[n] for n in ("c0", "c2", "c4", "cfog")},
            "stoch": {n: theta[n] for n in ("P_shot", "a0", "a2")},
            "k_nl": K_NL, "ndens": ndens}


def ps_multipoles_jax(model, k_h, pk, h, f, a_perp, a_para, theta, k, *, ap, ndens, num=256):
    """All three multipoles from ONE P(k, mu), with the numerics of
    PowerSpectrum1Loop.get_pk_ell / get_pk_ell_ref (ps_1loop.py:316-357): mu = linspace(0,1,num),
    Simpson's rule, weight (2l+1) L_l(mu). Parity: test_three_in_one_projection_matches_get_pk_ell_ref."""
    import quadax
    from ps_1loop_jax.utils_math import legendre
    pk_data = {"k": k_h, "pk": pk}
    params = ps_params(theta, h=h, f=f, ndens=ndens)
    k = jnp.atleast_1d(k)
    mu = jnp.linspace(0.0, 1.0, num)
    if ap:
        pkmu = model.get_pkmu_ref(k, mu, a_perp, a_para, pk_data, params)
    else:
        pkmu = model.get_pkmu(k, mu, pk_data, params)
    return jnp.stack([quadax.simpson(pkmu * ((2 * l + 1) * legendre(l, mu))[None, :], x=mu, axis=1)
                      for l in ELLS])


@functools.partial(jax.jit, static_argnames=("model", "ap", "num"))
def _ps_batch_jit(model, k_h, pk, h, f, a_perp, a_para, thetas, k, ndens, *, ap, num):
    def one(th):
        return ps_multipoles_jax(model, k_h, pk, h, f, a_perp, a_para,
                                 dict(zip(THETA_NAMES, th)), k, ap=ap, ndens=ndens, num=num)
    return jax.vmap(one)(thetas)


def ps_batch(model, inp, kw, *, ndens=NDENS_PROBE, num=256):
    a_perp, a_para = ap_alphas(inp["hratio"], inp["Dratio"])
    args = (jnp.asarray(inp["k_h"]), jnp.asarray(inp["pk"]), inp["h"], inp["f"], a_perp, a_para)

    def F(thetas):
        return np.asarray(_ps_batch_jit(model, *args, jnp.asarray(thetas, dtype=float),
                                        jnp.asarray(kw), ndens, ap=has_ap(inp), num=num))
    return F


# --------------------------------------------------------------------------- translation

def to_clax(theta: dict, *, f, ndens, Pd2d2_0, hratio, Dratio) -> dict:
    """Basis translation ps_1loop -> clax-pt / CLASS-PT (spec §4.1). Exact except the model
    differences of spec §4.2, which test_ps1loop_model_difference.py closes. jnp-safe."""
    V = ap_volume(hratio, Dratio)
    knl2 = (0.45 / K_NL) ** 2
    c0, c2, c4 = theta["c0"], theta["c2"], theta["c4"]
    return {"b1": theta["b1"], "b2": theta["b2"], "bG2": theta["bG2"], "bGamma3": theta["bGamma3"],
            # Legendre content of mu^2 = L0/3 + 2L2/3 and mu^4 = L0/5 + 4L2/7 + 8L4/35
            "cs0": c0 + f * c2 / 3.0 + f**2 * c4 / 5.0,
            "cs2": c2 + 6.0 * f * c4 / 7.0,
            "cs4": c4,
            "cs": 0.0,
            "b4": theta["cfog"],
            # ps_1loop: P_shot/ndens x Jacobian V, P_d2d2(k->0) subtracted (ps_1loop.py:381-384);
            # clax-pt: Pshot in observed coordinates, +0.25 b2^2 Pd2d2_0 added back (classy.pyx:4911).
            "Pshot": V * theta["P_shot"] / ndens - 0.25 * theta["b2"] ** 2 * Pd2d2_0,
            # ps_1loop.py:576-588 in true coordinates: a0 (k_t/k_nl)^2 -> l=0 moment; a2 k_t^2 mu_t^2 = k^2 mu^2 hratio^2
            "a0": V * theta["a0"] / ndens * knl2 * (2.0 / (3.0 * Dratio**2) + hratio**2 / 3.0),
            "a2": V * hratio**2 * theta["a2"] / ndens * knl2}


# --------------------------------------------------------------------------- clax-pt and CLASS-PT

def clax_ept(inp, *, ir=True, pk=None, f=None, hratio=None, Dratio=None):
    """compute_ept on the shared inputs; the overrides let Task 7 trace f, AP and P_lin."""
    return compute_ept(jnp.asarray(inp["pk"]) if pk is None else pk, jnp.asarray(inp["k_h"]),
                       h=inp["h"], f=inp["f"] if f is None else f,
                       prec=EPTPrecisionParams(ir_resummation=ir), rs_h=inp["rbao"],
                       hratio=inp["hratio"] if hratio is None else hratio,
                       Dratio=inp["Dratio"] if Dratio is None else Dratio)


def clax_multipoles_jax(e, cb: dict):
    b = (cb["b1"], cb["b2"], cb["bG2"], cb["bGamma3"])
    return jnp.stack([pk_gg_l0(e, *b, cs0=cb["cs0"], Pshot=cb["Pshot"], b4=cb["b4"], a0=cb["a0"], a2=cb["a2"]),
                      pk_gg_l2(e, *b, cs2=cb["cs2"], b4=cb["b4"], a2=cb["a2"]),
                      pk_gg_l4(e, *b, cs4=cb["cs4"], b4=cb["b4"])])


def clax_batch(e, inp, mask, *, ndens=NDENS_PROBE):
    def one(th):
        cb = to_clax(dict(zip(THETA_NAMES, th)), f=inp["f"], ndens=ndens, Pd2d2_0=e.Pd2d2_0,
                     hratio=inp["hratio"], Dratio=inp["Dratio"])
        return clax_multipoles_jax(e, cb)[:, mask]
    vm = jax.vmap(one)
    return lambda thetas: np.asarray(vm(jnp.asarray(thetas, dtype=float)))


def classpt_multipoles(inp, cb: dict) -> np.ndarray:
    """CLASS-PT at any bias point, rebuilt from its stored pk_mult rows (classy accessors,
    scripts/classpt_assembly.py; a0 = a2 = 0 there)."""
    out = ca.assemble_from_pm(inp["pk_mult"], inp["h"], inp["f"], inp["k_h"], cb, inp["Pd2d2_0"])
    return np.stack([out["pk_gg_l0"], out["pk_gg_l2"], out["pk_gg_l4"]])


def classpt_batch(inp, mask, *, ndens=NDENS_PROBE):
    def F(thetas):
        rows = []
        for th in np.asarray(thetas, dtype=float):
            cb = to_clax(dict(zip(THETA_NAMES, th)), f=inp["f"], ndens=ndens, Pd2d2_0=inp["Pd2d2_0"],
                         hratio=inp["hratio"], Dratio=inp["Dratio"])
            rows.append(classpt_multipoles(inp, {k: float(v) for k, v in cb.items()})[:, mask])
        return np.stack(rows)
    return F


# --------------------------------------------------------------------------- coefficient spectra

_FUNCTIONAL = {0: ((0, 1.0),), 1: ((-1, -0.5), (1, 0.5)), 2: ((-1, 0.5), (0, -1.0), (1, 0.5))}


def stencil(mono: tuple[str, ...]) -> list[tuple[dict, float]]:
    """[(theta, weight)] whose weighted sum of F(theta) is the coefficient of `mono`, exact for
    F of degree <= 2 in each parameter (all other parameters at zero)."""
    powers: dict[str, int] = {}
    for v in mono:
        if v != "1":
            powers[v] = powers.get(v, 0) + 1
    names = sorted(powers)
    out = []
    for combo in itertools.product(*(_FUNCTIONAL[powers[n]] for n in names)):
        th, w = zero_theta(), 1.0
        for n, (x, wx) in zip(names, combo):
            th[n] = float(x)
            w *= wx
        out.append((th, w))
    return out


def coefficients(F, monomials) -> dict[str, np.ndarray]:
    """{label: (M, Nk)} from ONE batched call of F over the union of all stencil points."""
    points: dict[tuple, int] = {}
    plan = {}
    for mono in monomials:
        terms = []
        for th, w in stencil(mono):
            key = tuple(th[n] for n in THETA_NAMES)
            terms.append((points.setdefault(key, len(points)), w))
        plan[label(mono)] = terms
    vals = np.asarray(F(np.array(list(points), dtype=float)))
    return {lab: sum(w * vals[j] for j, w in terms) for lab, terms in plan.items()}


def coefficient_triplet(inp, monomials, *, ir=True, ndens=NDENS_PROBE):
    mask = cu.window(inp["k_h"])
    kw = inp["k_h"][mask]
    e = clax_ept(inp, ir=ir)
    res = {"ps": coefficients(ps_batch(ps_model(inp["rbao"], do_irres=ir), inp, kw, ndens=ndens), monomials),
           "clax": coefficients(clax_batch(e, inp, mask, ndens=ndens), monomials),
           "classpt": {}}
    tb = [m for m in monomials if not set(m) & NO_TIEBREAK]
    if ir and tb:   # the twin is rebuilt from IR-ON pk_mult rows
        res["classpt"] = coefficients(classpt_batch(inp, mask, ndens=ndens), tb)
    return res, kw


def spectra_at(inp, theta: dict, *, ndens):
    """ps_1loop and clax-pt multipoles at one parameter point, on the window."""
    mask = cu.window(inp["k_h"])
    kw = inp["k_h"][mask]
    th = np.array([[theta[n] for n in THETA_NAMES]], dtype=float)
    ps = ps_batch(ps_model(inp["rbao"]), inp, kw, ndens=ndens)(th)[0]
    cl = clax_batch(clax_ept(inp), inp, mask, ndens=ndens)(th)[0]
    return ps, cl, kw


# --------------------------------------------------------------------------- gate

def coef_error(got, ref, kw, *, scale) -> dict:
    """Campaign metric max|got-ref|/max|ref| on the window (spec §5.1). When ref is a
    structural zero (max|ref| <= 1e-12 scale) the denominator is `scale`, the monomial's
    largest clax-pt multipole. Non-finite -> err = nan (a failure, Review Focus 5)."""
    got, ref = np.asarray(got, dtype=float), np.asarray(ref, dtype=float)
    d = np.abs(got - ref)
    if not np.all(np.isfinite(d)):
        i = int(np.argmax(~np.isfinite(d)))
        return {"err": float("nan"), "k": float(kw[i]), "metric": "non-finite"}
    i = int(np.argmax(d))
    ref_max = float(np.max(np.abs(ref)))
    zero = ref_max <= STRUCTURAL_ZERO * scale
    denom = max(scale if zero else ref_max, 1e-300)
    return {"err": float(d[i] / denom), "k": float(kw[i]), "metric": "monomial-scale" if zero else "self"}


def attribute(res, lab, i, kw, scale, thr) -> str:
    """Tie-breaker verdict (spec §6) for coefficient `lab`, channel index i."""
    ct = res.get("classpt", {}).get(lab)
    if ct is None:
        return "tie-breaker n/a"
    e_ps = coef_error(res["ps"][lab][i], ct[i], kw, scale=scale)["err"]
    e_cl = coef_error(res["clax"][lab][i], ct[i], kw, scale=scale)["err"]
    if float(np.max(np.abs(ct[i]))) <= STRUCTURAL_ZERO * scale and e_cl <= thr:
        verdict = "CLASS-PT convention (projection omitted in CLASS-PT and clax-pt)"
    elif not np.isfinite(e_ps) or e_ps > e_cl:
        verdict = "suspect ps_1loop"
    else:
        verdict = "suspect clax-pt"
    return f"vs CLASS-PT: ps {100 * e_ps:.3f}%, clax {100 * e_cl:.3f}% -> {verdict}"


def gate(res, kw, *, channels=CHANNELS, thresholds=COEF_THRESHOLDS):
    errs, lines = {}, []
    for lab, ref in res["clax"].items():
        scale = float(np.max(np.abs(ref)))
        for i, ch in enumerate(channels):
            e = coef_error(res["ps"][lab][i], ref[i], kw, scale=scale)
            name = f"{lab}:{ch}"
            errs[name] = e
            thr = thresholds[ch]
            if not np.isfinite(e["err"]) or e["err"] > thr:
                lines.append(f"ERROR {name} {e['metric']} {100 * e['err']:.3f}% > {100 * thr:.1f}% "
                             f"at k={e['k']:.3f} -- {attribute(res, lab, i, kw, scale, thr)}")
    return errs, lines


def worst(errs: dict) -> str:
    name, rec = max(errs.items(), key=lambda kv: np.inf if not np.isfinite(kv[1]["err"]) else kv[1]["err"])
    return f"worst {name} {100 * rec['err']:.3f}% at k={rec['k']:.3f}"


# --------------------------------------------------------------------------- log

def ps1loop_commit() -> str:
    try:
        import ps_1loop_jax
        d = Path(ps_1loop_jax.__file__).resolve().parent
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=d,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:   # installed without a git checkout -- still log
        return "unknown"


def log_record(*, layer: str, case: str, z: float, preset: str, errors: dict, extra: dict | None = None) -> None:
    """One JSON line in test_logs/ps1loop/errors.jsonl: ts, git_sha, ps1loop_sha, layer
    ('coef' | 'closure' | 'irno' | 'real' | 'grad' | 'pfs' | 'ladder'), case, z, preset, errors, extra."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rec = {"ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
           "git_sha": cu._git_sha(), "ps1loop_sha": ps1loop_commit(), "layer": layer, "case": case,
           "z": float(z), "preset": preset, "errors": errors, "extra": extra or {}}
    with ERROR_LOG.open("a") as fh:
        fh.write(json.dumps(cu._sanitize(rec), default=float, allow_nan=False) + "\n")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `slurm/run-ps1loop-pytest.sh tests/test_ps1loop_bridge.py -q`
Expected: `12 passed` (13 once Task 5 adds `test_omitted_predicted_covers_exactly_the_closure_lines`). If `test_three_in_one_projection_matches_get_pk_ell_ref` fails, `ps_multipoles_jax` has drifted from `get_pk_ell_ref`. Re-read ps_1loop.py:316-357 and fix the bridge. Never widen the tolerance.

- [ ] **Step 5: Fast suite, then commit**

Run: `slurm/run-ps1loop-fast-suite.sh` → `PASS`.

```bash
git add tests/ps1loop_bridge.py tests/test_ps1loop_bridge.py
git commit -m "test(ps1loop): bridge -- shared inputs, basis translation, exact coefficient stencils, gate

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Coefficient gate for the exactly-translating monomials (42 files)

**Files:**
- Test: `tests/test_ps1loop_coefficients.py`

**Interfaces:**
- Consumes: `br.shared_inputs`, `br.coefficient_triplet`, `br.gate`, `br.worst`, `br.log_record`, `br.EXACT_MONOMIALS`; `vc.distinct_cases()`, `vc.Z_LIST`, `vc.FAST_CASES`, `vc.FAST_Z`; `cu.require_reference`.

- [ ] **Step 1: Write the test**

Create `tests/test_ps1loop_coefficients.py`:

```python
# tests/test_ps1loop_coefficients.py
"""Code-accuracy gate on coefficient spectra (spec §5.1): ps_1loop vs clax-pt on CLASS-PT's
own inputs, for every monomial whose translation is exact (spec §4.1). 14 cosmologies x
z in (0, 0.38, 0.8) in full mode; vc.FAST_CASES x vc.FAST_Z under --fast. Multi-cosmology
rule: satisfied by construction. Failures carry the CLASS-PT tie-breaker verdict (spec §6)."""
from __future__ import annotations

import pytest

pytest.importorskip("ps_1loop_jax")

from scripts import validation_cosmologies as vc
from tests import ept_campaign_utils as cu
from tests import ps1loop_bridge as br


def pytest_generate_tests(metafunc):
    if "case_z" in metafunc.fixturenames:
        if metafunc.config.getoption("--fast", default=False):
            grid = [(c, vc.FAST_Z) for c in vc.FAST_CASES]
        else:
            grid = [(c, z) for c in vc.distinct_cases() for z in vc.Z_LIST]
        metafunc.parametrize("case_z", grid, ids=[f"{c}-z{z:.2f}" for c, z in grid])


def test_exact_coefficients(case_z):
    case, z = case_z
    inp = br.shared_inputs(cu.require_reference(case, z))
    res, kw = br.coefficient_triplet(inp, br.EXACT_MONOMIALS)
    errs, lines = br.gate(res, kw)
    br.log_record(layer="coef", case=case, z=z, preset="exact", errors=errs)
    print(f"{case} z={z:.2f} [coef] {br.worst(errs)}")
    assert not lines, f"{case} z={z:.2f}:\n" + "\n".join(lines)
```

- [ ] **Step 2: Run it in fast mode**

Run: `slurm/run-ps1loop-pytest.sh tests/test_ps1loop_coefficients.py --fast -q`
Expected: `3 passed`, each printing a `worst` line.

If a case FAILS, this is a finding, not a test bug. Read the `ERROR` line's verdict:
- **suspect ps_1loop**: re-check the translation row in `to_clax` for that monomial against spec §4.1 and the cited source lines. If the translation is right, record the residual (monomial, ℓ, k, size, verdict) in `docs/ps1loop-findings.md` (Task 10) and mark the test `xfail(strict=True, reason="<finding id>")` for exactly that `(case, monomial)` only. Ask the user before adding any xfail.
- **suspect clax-pt**: stop and report to the user with the line. A clax-pt defect is fixed on its own branch, never inside this plan.
- **CLASS-PT convention**: record it in `docs/ps1loop-findings.md` as a model difference and ask the user whether it moves to the closure set (Task 5).

- [ ] **Step 3: Run it in full mode**

Run: `slurm/run-ps1loop-pytest.sh tests/test_ps1loop_coefficients.py -q`
Expected: `42 passed`. Then:

```bash
python - <<'EOF'
import json
recs = [json.loads(l) for l in open("test_logs/ps1loop/errors.jsonl") if '"layer": "coef"' in l]
w = max(((r["case"], r["z"], n, e["err"]) for r in recs for n, e in r["errors"].items() if e["err"] is not None), key=lambda t: t[3])
print("worst over 42 files:", w)
EOF
```
Record the printed worst value for the CHANGELOG (Task 10).

- [ ] **Step 4: Fast suite, then commit**

Run: `slurm/run-ps1loop-fast-suite.sh` → `PASS`.

```bash
git add tests/test_ps1loop_coefficients.py
git commit -m "test(ps1loop): coefficient-spectrum gate vs ps_1loop on CLASS-PT inputs, 42 files

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Model-difference closure and IR-off exactness

**Files:**
- Modify: `tests/ps1loop_bridge.py` (append `model_difference_factory`)
- Test: `tests/test_ps1loop_model_difference.py`

**Interfaces:**
- Consumes: `br.ps_model`, `br.to_clax`, `br.project`, `br.b4_poly`, `br.ap_alphas`, `br.ap_volume`, `br.coefficients`, `br.coefficient_triplet`, `br.gate`, `br.MODEL_DIFF_MONOMIALS`; ps_1loop `ir_resum.get_pk_nw_data(pk_data, h, khmin, khmax, kmin_interp, kmax_interp)`, `PowerSpectrum1Loop._get_pk_irres_rsd(k, mu, pk_data, pk_nw_data, f) -> (Nk, Nmu)`.
- Produces: `model_difference_factory(model, inp, kw, *, ndens=NDENS_PROBE) -> F(thetas) -> np (B, 3, Nw)`, the predicted ps_1loop − clax-pt difference.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ps1loop_model_difference.py`:

```python
# tests/test_ps1loop_model_difference.py
"""Model-difference closure (spec §4.2, §5.2) and counterterm-map exactness with IR off
(spec §5.3). Closure: for c0, c2, c4, cfog, cfog*b1, cfog*b1^2 and a0, the measured
ps_1loop - clax-pt coefficient difference must EQUAL the closed-form model difference
evaluated on ps_1loop's own P_IR. 42 files in full mode (closure) and the 14 z = 0 files
(IR off); vc.FAST_CASES under --fast. Multi-cosmology rule: by construction."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ps_1loop_jax")

from scripts import validation_cosmologies as vc
from tests import ept_campaign_utils as cu
from tests import ps1loop_bridge as br


def pytest_generate_tests(metafunc):
    fast = metafunc.config.getoption("--fast", default=False)
    cases = list(vc.FAST_CASES) if fast else list(vc.distinct_cases())
    if "case_z" in metafunc.fixturenames:
        grid = [(c, vc.FAST_Z) for c in cases] if fast else [(c, z) for c in cases for z in vc.Z_LIST]
        metafunc.parametrize("case_z", grid, ids=[f"{c}-z{z:.2f}" for c, z in grid])
    if "case0" in metafunc.fixturenames:
        metafunc.parametrize("case0", cases)


def test_model_difference_closes(case_z):
    case, z = case_z
    inp = br.shared_inputs(cu.require_reference(case, z))
    res, kw = br.coefficient_triplet(inp, br.MODEL_DIFF_MONOMIALS)
    pred = br.coefficients(br.model_difference_factory(br.ps_model(inp["rbao"]), inp, kw),
                           br.MODEL_DIFF_MONOMIALS)
    # gate(clax + measured - predicted vs clax) == max|measured - predicted| / |clax coefficient|
    closure = {"ps": {lab: res["clax"][lab] + (res["ps"][lab] - res["clax"][lab]) - pred[lab]
                      for lab in res["clax"]},
               "clax": res["clax"], "classpt": {}}
    errs, lines = br.gate(closure, kw)
    size = {lab: float(np.max(np.abs(pred[lab])) / max(np.max(np.abs(res["clax"][lab])), 1e-300))
            for lab in pred}
    br.log_record(layer="closure", case=case, z=z, preset="model-difference", errors=errs,
                  extra={"model_difference_size": size})
    print(f"{case} z={z:.2f} [closure] {br.worst(errs)}; largest model difference "
          f"{max(size, key=size.get)} {100 * max(size.values()):.2f}%")
    assert not lines, f"{case} z={z:.2f}:\n" + "\n".join(lines)


def test_counterterm_map_exact_with_ir_off(case0):
    inp = br.shared_inputs(cu.require_reference(case0, 0.0))
    assert not br.has_ap(inp)
    res, kw = br.coefficient_triplet(inp, br.MODEL_DIFF_MONOMIALS, ir=False)
    errs, lines = br.gate(res, kw)
    br.log_record(layer="irno", case=case0, z=0.0, preset="ctr-map-exact", errors=errs)
    print(f"{case0} z=0 [ir-off map] {br.worst(errs)}")
    assert not lines, f"{case0} z=0 IR off:\n" + "\n".join(lines)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `slurm/run-ps1loop-pytest.sh tests/test_ps1loop_model_difference.py --fast -q`
Expected: the closure tests FAIL with `AttributeError: module 'tests.ps1loop_bridge' has no attribute 'model_difference_factory'`. The IR-off tests may already pass. They exercise only Task 3 code, and a pass confirms the map is exact when P is isotropic.

- [ ] **Step 3: Append the closed form to the bridge**

Append to `tests/ps1loop_bridge.py`:

```python
# --------------------------------------------------------------------------- model difference (spec §4.2)

def model_difference_factory(model, inp, kw, *, ndens=NDENS_PROBE):
    """Closed-form ps_1loop-model minus clax-pt-model for the k^2 counterterms, the k^4
    fingers-of-god term and a0, both on ps_1loop's own P_IR(k_t, mu_t). P_IR is built once
    (theta-independent); the returned F is polynomial in theta, so `coefficients` applies.

    ps_1loop model (true coordinates, Jacobian V; ps_1loop.py:496-547, 576-588, 328-344):
      V [-2 k_t^2 (c0 + c2 f mu_t^2 + c4 f^2 mu_t^4) P_IR - cfog f^4 k_t^4 mu_t^4 (b1 + f mu_t^2)^2 P_IR
         + a0 (k_t/k_nl)^2 / ndens], projected on L_l(mu) over the observed mu.
    clax-pt model (ept.py:1207-1209 and pk_gg_l0/l2/l4):
      2 cs_l Pctr_l + f^2 b4 k^2 poly_l (35/8) Pctr_4 [+ a0 (k/0.45)^2 at l=0],
      Pctr_l = -proj_l(V k_t^2 P_IR f^{l/2} mu_t^l)."""
    from ps_1loop_jax import ir_resum
    h, f, hr, Dr = inp["h"], inp["f"], inp["hratio"], inp["Dratio"]
    pk_data = {"k": jnp.asarray(inp["k_h"]), "pk": jnp.asarray(inp["pk"])}
    # same call as PowerSpectrum1Loop.get_pkmu (ps_1loop.py:291-292)
    pk_nw_data = ir_resum.get_pk_nw_data(pk_data, h, khmin=7e-5, khmax=7.0,
                                         kmin_interp=model._kmin, kmax_interp=model._kmax)
    a_perp, a_para = ap_alphas(hr, Dr)
    V = ap_volume(hr, Dr)
    fac = np.sqrt(1.0 + GL_MU**2 * ((a_perp / a_para) ** 2 - 1.0))
    mu_t = GL_MU * (a_perp / a_para) / fac
    kw = np.asarray(kw, dtype=float)
    k_t = kw[:, None] * fac[None, :] / a_perp                                  # (Nk, 40)
    p_ir = np.stack([np.asarray(model._get_pk_irres_rsd(jnp.asarray(k_t[:, j]), jnp.asarray([mu_t[j]]),
                                                        pk_data, pk_nw_data, f))[:, 0]
                     for j in range(GL_MU.size)], axis=1)
    pctr = {l: -project(V * k_t**2 * p_ir * f ** (l // 2) * mu_t**l, l) for l in ELLS}

    def one(theta):
        c0, c2, c4, cfog, b1, a0 = (theta[n] for n in ("c0", "c2", "c4", "cfog", "b1", "a0"))
        ps_mu = V * (-2.0 * k_t**2 * (c0 + c2 * f * mu_t**2 + c4 * f**2 * mu_t**4) * p_ir
                     - cfog * f**4 * k_t**4 * mu_t**4 * (b1 + f * mu_t**2) ** 2 * p_ir
                     + a0 * (k_t / K_NL) ** 2 / ndens)
        cb = to_clax(theta, f=f, ndens=ndens, Pd2d2_0=0.0, hratio=hr, Dratio=Dr)
        poly, cs = b4_poly(f, b1), {0: cb["cs0"], 2: cb["cs2"], 4: cb["cs4"]}
        rows = []
        for l in ELLS:
            clax_l = 2.0 * cs[l] * pctr[l] + f**2 * cb["b4"] * kw**2 * poly[l] * (35.0 / 8.0) * pctr[4]
            if l == 0:
                clax_l = clax_l + cb["a0"] * (kw / 0.45) ** 2
            rows.append(project(ps_mu, l) - clax_l)
        return np.stack(rows)

    return lambda thetas: np.stack([one(dict(zip(THETA_NAMES, th))) for th in np.asarray(thetas, dtype=float)])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `slurm/run-ps1loop-pytest.sh tests/test_ps1loop_model_difference.py --fast -q`
Expected: `12 passed` (3 closure + 3 IR-off + 3 omitted-projection reproduction + 3 omitted-projection closure; Task 5 amendments), with closure lines naming the largest model difference.

If the closure fails while the IR-off test passes, the remaining gap is not the documented model difference. Treat it as a finding, following the Task 4 Step 2 protocol. One case to check first: if the residual's shape is k²·P_IR, record it as a UV-convention difference (spec §6) and ask the user. Do not re-fit anything.

Run: `slurm/run-ps1loop-pytest.sh tests/test_ps1loop_model_difference.py -q`
Expected: `140 passed` (42 closure + 14 IR-off + 42 omitted-projection reproduction + 42 omitted-projection closure; Task 5 amendments).

- [ ] **Step 5: Fast suite, then commit**

Run: `slurm/run-ps1loop-fast-suite.sh` → `PASS`.

```bash
git add tests/ps1loop_bridge.py tests/test_ps1loop_model_difference.py
git commit -m "test(ps1loop): closed-form counterterm/FoG/a0 model difference closes; map exact with IR off

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Real space with IR off

**Files:**
- Modify: `tests/ps1loop_bridge.py` (append real-space evaluators)
- Test: `tests/test_ps1loop_real_space.py`

**Interfaces:**
- Consumes: ps_1loop `get_pk_real(k, pk_data, params)`, `get_pk_gm_real(k, pk_data, params)` (ps_1loop.py:158-266); clax `pk_mm_real(e, cs0=)`, `pk_gg_real(e, b1, b2, bG2, bGamma3, cs=, cs0=, Pshot=)`, `pk_gm_real(e, b1, b2, bG2, bGamma3, cs0=, cs=)`; `ca.assemble_from_pm` keys `pk_mm_real`, `pk_gg_real`, `pk_gm_real` (classy `pk_mm_real(cs)` uses key `cs`); `cu.clax_nine`, `cu.compare_spectra`, `cu.failures`, `cu.THRESHOLDS`.
- Produces: `REAL_CHANNELS`, `REAL_THRESHOLDS`, `REAL_MONOMIALS`, `real_coefficient_triplet(inp, *, irno_inp=None, ndens=NDENS_PROBE) -> (res, kw)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ps1loop_real_space.py`:

```python
# tests/test_ps1loop_real_space.py
"""Real-space P_mm, P_gg, P_gm with IR resummation OFF in both codes (spec §5.4): ps_1loop's
real-space path has no IR resummation, so only IR-off is a numerics comparison. 42 files in
full mode, vc.FAST_CASES x vc.FAST_Z under --fast; tie-breaker from the IR-off CLASS-PT files
where they exist. Plus: clax-pt's IR-off path on its own vs CLASS-PT IR-off (3 cosmologies).
Multi-cosmology rule: by construction."""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("ps_1loop_jax")

from scripts import validation_cosmologies as vc
from tests import ept_campaign_utils as cu
from tests import ps1loop_bridge as br


def pytest_generate_tests(metafunc):
    if "case_z" in metafunc.fixturenames:
        if metafunc.config.getoption("--fast", default=False):
            grid = [(c, vc.FAST_Z) for c in vc.FAST_CASES]
        else:
            grid = [(c, z) for c in vc.distinct_cases() for z in vc.Z_LIST]
        metafunc.parametrize("case_z", grid, ids=[f"{c}-z{z:.2f}" for c, z in grid])


def test_real_space_coefficients_ir_off(case_z):
    case, z = case_z
    inp = br.shared_inputs(cu.require_reference(case, z))
    irno = cu.load_reference(case, z, ap=False, tag=vc.IRNO_TAG)
    res, kw = br.real_coefficient_triplet(inp, irno_inp=br.shared_inputs(irno) if irno is not None else None)
    errs, lines = br.gate(res, kw, channels=br.REAL_CHANNELS, thresholds=br.REAL_THRESHOLDS)
    br.log_record(layer="real", case=case, z=z, preset="ir-off", errors=errs)
    print(f"{case} z={z:.2f} [real, IR off] {br.worst(errs)}")
    assert not lines, f"{case} z={z:.2f}:\n" + "\n".join(lines)


@pytest.mark.parametrize("case", vc.IRNO_CASES)
def test_clax_ir_off_matches_classpt_ir_off(case):
    ref = cu.require_reference(case, 0.38, ap=False, tag=vc.IRNO_TAG)
    e = br.clax_ept(br.shared_inputs(ref), ir=False)
    nine = {n: np.asarray(a) for n, a in cu.clax_nine(e, json.loads(str(ref["bias_json"]))).items()}
    errs = cu.compare_spectra({n: nine[n] for n in br.REAL_CHANNELS}, ref, ref["k_h"])
    br.log_record(layer="real", case=case, z=0.38, preset="clax-vs-classpt-irno", errors=errs)
    bad = cu.failures(errs, cu.THRESHOLDS)
    print(f"{case} [clax IR off vs CLASS-PT IR off] {br.worst(errs)}")
    assert not bad, f"ERROR {case}: " + "; ".join(bad)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `slurm/run-ps1loop-pytest.sh tests/test_ps1loop_real_space.py --fast -q`
Expected: the coefficient tests FAIL with `AttributeError: ... no attribute 'real_coefficient_triplet'`. The 3 `clax_ir_off` tests run on Task 2's files; if one fails, that is a clax-pt IR-off finding — stop and report.

- [ ] **Step 3: Append the real-space evaluators to the bridge**

Append to `tests/ps1loop_bridge.py`:

```python
# --------------------------------------------------------------------------- real space, IR off (spec §5.4)

from clax.ept import pk_gg_real, pk_gm_real, pk_mm_real   # noqa: E402

REAL_CHANNELS = ("pk_mm_real", "pk_gg_real", "pk_gm_real")
REAL_THRESHOLDS = {c: 0.01 for c in REAL_CHANNELS}
# P_mm: 1, c0 (the k^2 P shape, cs0 = c0); P_gg: bias pairs + P_shot; P_gm: linear bias terms.
REAL_MONOMIALS = (("1",), ("c0",), ("b1",), ("b2",), ("bG2",), ("bGamma3",),
                  ("b1", "b1"), ("b1", "b2"), ("b2", "b2"), ("b1", "bG2"), ("bG2", "bG2"),
                  ("b2", "bG2"), ("b1", "bGamma3"), ("P_shot",))


@functools.partial(jax.jit, static_argnames=("model",))
def _ps_real_jit(model, k_h, pk, thetas, k, ndens):
    pk_data = {"k": k_h, "pk": pk}

    def one(th):
        t = dict(zip(THETA_NAMES, th))
        z0 = 0.0 * t["b1"]
        p = ps_params(t, h=1.0, f=0.0, ndens=ndens)   # h, f unused: no IR, no RSD in real space
        # P_mm = P_gg at b1 = 1, rest 0, carrying the c0 counterterm (ps_1loop.py:158-188)
        mm = model.get_pk_real(k, pk_data, {**p, "bias": {"b1": 1.0 + z0, "b2": z0, "bG2": z0, "bGamma3": z0},
                                            "stoch": {"P_shot": z0, "a0": z0, "a2": z0}})
        # P_gg without counterterm: clax-pt's P_gg counterterm is parameterised differently (spec §5.4)
        gg = model.get_pk_real(k, pk_data, {**p, "ctr": {**p["ctr"], "c0": z0}})
        gm = model.get_pk_gm_real(k, pk_data, {**p, "ctr": {"c0": z0, "cs0": z0}})
        return jnp.stack([mm, gg, gm])
    return jax.vmap(one)(thetas)


def ps_real_batch(model, inp, kw, *, ndens=NDENS_PROBE):
    def F(thetas):
        return np.asarray(_ps_real_jit(model, jnp.asarray(inp["k_h"]), jnp.asarray(inp["pk"]),
                                       jnp.asarray(thetas, dtype=float), jnp.asarray(kw), ndens))
    return F


def clax_real_batch(e, mask, *, ndens=NDENS_PROBE):
    def one(th):
        t = dict(zip(THETA_NAMES, th))
        b = (t["b1"], t["b2"], t["bG2"], t["bGamma3"])
        return jnp.stack([pk_mm_real(e, cs0=t["c0"]),
                          pk_gg_real(e, *b, cs=0.0, cs0=0.0, Pshot=t["P_shot"] / ndens),
                          pk_gm_real(e, *b, cs0=0.0, cs=0.0)])[:, mask]
    vm = jax.vmap(one)
    return lambda thetas: np.asarray(vm(jnp.asarray(thetas, dtype=float)))


def classpt_real_batch(inp, mask, *, ndens=NDENS_PROBE):
    def F(thetas):
        rows = []
        for th in np.asarray(thetas, dtype=float):
            t = dict(zip(THETA_NAMES, th))
            # classy pk_mm_real(cs) reads `cs` (classy.pyx:4816); P_gg/P_gm counterterms off.
            cb = {"b1": t["b1"], "b2": t["b2"], "bG2": t["bG2"], "bGamma3": t["bGamma3"],
                  "cs0": 0.0, "cs2": 0.0, "cs4": 0.0, "cs": t["c0"], "Pshot": t["P_shot"] / ndens, "b4": 0.0}
            o = ca.assemble_from_pm(inp["pk_mult"], inp["h"], inp["f"], inp["k_h"], cb, inp["Pd2d2_0"])
            rows.append(np.stack([o[c] for c in REAL_CHANNELS])[:, mask])
        return np.stack(rows)
    return F


def real_coefficient_triplet(inp, *, irno_inp=None, ndens=NDENS_PROBE):
    mask = cu.window(inp["k_h"])
    kw = inp["k_h"][mask]
    e = clax_ept(inp, ir=False)
    res = {"ps": coefficients(ps_real_batch(ps_model(inp["rbao"], do_irres=False), inp, kw, ndens=ndens),
                              REAL_MONOMIALS),
           "clax": coefficients(clax_real_batch(e, mask, ndens=ndens), REAL_MONOMIALS),
           "classpt": {}}
    if irno_inp is not None:
        np.testing.assert_array_equal(irno_inp["k_h"], inp["k_h"])
        res["classpt"] = coefficients(classpt_real_batch(irno_inp, mask, ndens=ndens), REAL_MONOMIALS)
    return res, kw
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `slurm/run-ps1loop-pytest.sh tests/test_ps1loop_real_space.py --fast -q`
Expected: `6 passed` (3 coefficient + 3 clax-vs-CLASS-PT IR-off).

Run: `slurm/run-ps1loop-pytest.sh tests/test_ps1loop_real_space.py -q`
Expected: `45 passed`. Failures follow the Task 4 Step 2 protocol.

- [ ] **Step 5: Fast suite, then commit**

Run: `slurm/run-ps1loop-fast-suite.sh` → `PASS`.

```bash
git add tests/ps1loop_bridge.py tests/test_ps1loop_real_space.py
git commit -m "test(ps1loop): real-space coefficients with IR off; clax-pt IR-off vs CLASS-PT IR-off

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Cross-code derivatives and ps_1loop AD vs finite differences

**Files:**
- Test: `tests/test_ps1loop_gradients.py`

**Interfaces:**
- Consumes: `br.shared_inputs`, `br.ps_model`, `br.ps_multipoles_jax`, `br.clax_ept(inp, pk=, f=, hratio=, Dratio=)`, `br.to_clax`, `br.clax_multipoles_jax`, `br.ap_alphas`, `br.has_ap`, `br.COEF_THRESHOLDS`, `br.ELLS`, `br.NDENS_PROBE`, `br.log_record`, `br.worst`; `pfs.bias_point`; `clax.interpolation.CubicSpline(x, y).derivative(xq)`; `cu.rel`, `cu.window`.

- [ ] **Step 1: Write the test**

Create `tests/test_ps1loop_gradients.py`:

```python
# tests/test_ps1loop_gradients.py
"""Derivatives (spec §5.5): forward-mode tangents of P_l along f, alpha_perp, alpha_para,
a k-stretch t = P dlnP/dlnk and the amplitude t = P -- ps_1loop vs clax-pt (code-accuracy
thresholds) and ps_1loop's AD vs its own central differences (1%, CLAUDE.md gradient target).
lcdm_fiducial, massive_nu_030, w0wa_m07_m10 at z = 0.38 (AP non-trivial) in full mode;
lcdm_fiducial under --fast. Multi-cosmology rule: 3 cosmologies."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ps_1loop_jax")
import jax
import jax.numpy as jnp

from clax.interpolation import CubicSpline
from tests import ept_campaign_utils as cu
from tests import pfs_survey as pfs
from tests import ps1loop_bridge as br

GRAD_CASES = ("lcdm_fiducial", "massive_nu_030", "w0wa_m07_m10")
Z = 0.38
FD_STEP = 1e-4
FD_THRESHOLD = 0.01


def pytest_generate_tests(metafunc):
    if "case" in metafunc.fixturenames:
        fast = metafunc.config.getoption("--fast", default=False)
        metafunc.parametrize("case", ["lcdm_fiducial"] if fast else list(GRAD_CASES))


def _setup(case):
    inp = br.shared_inputs(cu.require_reference(case, Z))
    assert br.has_ap(inp)
    mask = cu.window(inp["k_h"])
    k_h, kw = jnp.asarray(inp["k_h"]), jnp.asarray(inp["k_h"][mask])
    theta = pfs.bias_point(Z)
    model = br.ps_model(inp["rbao"])

    def ps_fun(f, a_perp, a_para, pk):
        return br.ps_multipoles_jax(model, k_h, pk, inp["h"], f, a_perp, a_para, theta, kw,
                                    ap=True, ndens=br.NDENS_PROBE)

    def clax_fun(f, a_perp, a_para, pk):
        hratio, Dratio = 1.0 / a_para, a_perp
        e = br.clax_ept(inp, pk=pk, f=f, hratio=hratio, Dratio=Dratio)
        cb = br.to_clax(theta, f=f, ndens=br.NDENS_PROBE, Pd2d2_0=e.Pd2d2_0, hratio=hratio, Dratio=Dratio)
        return br.clax_multipoles_jax(e, cb)[:, mask]

    a_perp, a_para = br.ap_alphas(inp["hratio"], inp["Dratio"])
    pk0 = jnp.asarray(inp["pk"])
    x0 = (jnp.asarray(inp["f"]), jnp.asarray(a_perp), jnp.asarray(a_para), pk0)
    t_stretch = pk0 * CubicSpline(jnp.log(k_h), jnp.log(pk0)).derivative(jnp.log(k_h))   # P dlnP/dlnk
    one, zero, zpk = jnp.asarray(1.0), jnp.asarray(0.0), jnp.zeros_like(pk0)
    tangents = {"f": (one, zero, zero, zpk), "a_perp": (zero, one, zero, zpk),
                "a_para": (zero, zero, one, zpk), "stretch": (zero, zero, zero, t_stretch),
                "amplitude": (zero, zero, zero, pk0)}
    return ps_fun, clax_fun, x0, tangents


def _check(case, layer, pairs, thresholds):
    errs, lines = {}, []
    for name, (got, ref) in pairs.items():
        for i, l in enumerate(br.ELLS):
            e = cu.rel(np.asarray(got[i]), np.asarray(ref[i]))
            errs[f"d/d{name}:l{l}"] = {"err": e, "k": float("nan")}
            thr = thresholds[f"l{l}"]
            if not np.isfinite(e) or e > thr:
                lines.append(f"ERROR {case} {layer} d/d{name} l={l}: {100 * e:.3f}% > {100 * thr:.1f}%")
    br.log_record(layer="grad", case=case, z=Z, preset=layer, errors=errs)
    print(f"{case} [{layer}] {br.worst(errs)}")
    assert not lines, "\n".join(lines)


def test_cross_code_tangents(case):
    ps_fun, clax_fun, x0, tangents = _setup(case)
    pairs = {name: (jax.jvp(ps_fun, x0, t)[1], jax.jvp(clax_fun, x0, t)[1]) for name, t in tangents.items()}
    _check(case, "ps-vs-clax", pairs, br.COEF_THRESHOLDS)


def test_ps1loop_ad_matches_central_differences(case):
    ps_fun, _, x0, tangents = _setup(case)
    pairs = {}
    for name, t in tangents.items():
        ad = jax.jvp(ps_fun, x0, t)[1]
        plus = ps_fun(*(x + FD_STEP * dx for x, dx in zip(x0, t)))
        minus = ps_fun(*(x - FD_STEP * dx for x, dx in zip(x0, t)))
        pairs[name] = (ad, (plus - minus) / (2.0 * FD_STEP))
    _check(case, "ps-AD-vs-FD", pairs, {f"l{l}": FD_THRESHOLD for l in br.ELLS})
```

- [ ] **Step 2: Run it in fast mode**

Run: `slurm/run-ps1loop-pytest.sh tests/test_ps1loop_gradients.py --fast -q`
Expected: `2 passed`.

If `ps-AD-vs-FD` fails only for `a_perp`/`a_para`, suspect differentiation through ps_1loop's 2D cubic interpolation (`interpax.interp2d`). Re-run that direction with FD_STEP 1e-3 and 1e-5 in a scratch script. FD that converges to AD means step noise; then report to the user before changing FD_STEP. FD that doesn't converge to AD is a ps_1loop finding (Task 10). If `ps-vs-clax` fails but both codes' AD-vs-FD pass, the codes disagree on the derivative, not on AD. Follow the Task 4 Step 2 protocol.

- [ ] **Step 3: Run it in full mode**

Run: `slurm/run-ps1loop-pytest.sh tests/test_ps1loop_gradients.py -q`
Expected: `6 passed`.

- [ ] **Step 4: Fast suite, then commit**

Run: `slurm/run-ps1loop-fast-suite.sh` → `PASS`.

```bash
git add tests/test_ps1loop_gradients.py
git commit -m "test(ps1loop): cross-code tangents along f, AP, k-stretch, amplitude; ps_1loop AD vs FD

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: The PFS error-bar gate

**Files:**
- Modify: `tests/pfs_survey.py` (append covariance, Δχ², headroom, bin interpolation)
- Modify: `tests/test_pfs_survey.py` (append numerics tests)
- Test: `tests/test_ps1loop_pfs_gate.py`

**Interfaces:**
- Consumes: `br.spectra_at(inp, theta, *, ndens)`, `br.shared_inputs`, `br.log_record`; `vc.PFS_CASES`; `pfs.load_slices`, `pfs.bias_point`, `pfs.C_REF`.
- Produces: `pfs.K_EDGES`, `pfs.K_CENTRES`, `pfs.DK`, `pfs.to_centres(P, kw) -> (3, Nb)`, `pfs.multipole_covariance(P, k, V, dk) -> (Nk, 3, 3)`, `pfs.delta_chi2(dP, C) -> float`, `pfs.counterterm_headroom(parts, c_ref) -> float`.

- [ ] **Step 1: Write the failing numerics tests**

Append to `tests/test_pfs_survey.py`:

```python
import numpy as np


def test_covariance_isotropic_closed_forms():
    # Isotropic P(k, mu) = P0: C00 = 2 P0^2/N_k, C22 = 25 * (2/5) P0^2 / N_k = 10 P0^2/N_k, C02 = 0.
    # Built ONLY from the P passed in -- no extra 1/nbar (Review Focus 2).
    k = np.array([0.1, 0.2])
    P = np.zeros((3, 2))
    P[0] = [1.0e4, 5.0e3]
    C = pfs.multipole_covariance(P, k, V=1e9, dk=0.01)
    Nk = 1e9 * k**2 * 0.01 / (2 * np.pi**2)
    np.testing.assert_allclose(C[:, 0, 0], 2 * P[0] ** 2 / Nk, rtol=1e-12)
    np.testing.assert_allclose(C[:, 1, 1], 10 * P[0] ** 2 / Nk, rtol=1e-12)
    np.testing.assert_allclose(C[:, 0, 1], 0.0, atol=1e-12 * np.max(C[:, 0, 0]))


def test_delta_chi2_and_headroom():
    C = np.broadcast_to(np.eye(3), (4, 3, 3)).copy()
    d1 = np.ones((3, 4))                          # at c = 2: chi2 = 12
    assert pfs.delta_chi2(d1, C) == pytest.approx(12.0)
    # d0 = 0 -> chi2(c) = 3 c^2 -> c* = 1/sqrt(3)
    assert pfs.counterterm_headroom([(np.zeros((3, 4)), d1, C)], 2.0) == pytest.approx(1 / np.sqrt(3))
    assert pfs.counterterm_headroom([(np.full((3, 4), 2.0), d1, C)], 2.0) == 0.0     # already >= 1
    assert pfs.counterterm_headroom([(np.zeros((3, 4)), np.zeros((3, 4)), C)], 2.0) == np.inf


def test_bins_and_interpolation():
    assert pfs.K_CENTRES[0] == pytest.approx(0.015) and pfs.K_CENTRES[-1] == pytest.approx(0.195)
    assert len(pfs.K_CENTRES) == 19 and pfs.DK == 0.01
    kw = np.geomspace(1e-3, 0.3, 80)
    P = np.stack([kw**-1.0, kw**0.5, np.log(kw)])
    np.testing.assert_allclose(pfs.to_centres(P, kw), np.stack([pfs.K_CENTRES**-1.0, pfs.K_CENTRES**0.5,
                                                               np.log(pfs.K_CENTRES)]), rtol=1e-6)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `slurm/run-ps1loop-pytest.sh tests/test_pfs_survey.py -q`
Expected: 3 FAIL with `AttributeError: module 'tests.pfs_survey' has no attribute 'multipole_covariance'` (or `K_CENTRES`); 6 pass.

- [ ] **Step 3: Append the numerics to `tests/pfs_survey.py`**

```python
# --------------------------------------------------------------------------- error-bar gate numerics (spec §5.6)
import numpy as np                                           # noqa: E402
from scipy.interpolate import CubicSpline as _CubicSpline   # noqa: E402

DK = 0.01                                                    # h/Mpc
K_EDGES = np.round(np.arange(0.01, 0.2 + 1e-9, DK), 10)      # 0.01 ... 0.20
K_CENTRES = 0.5 * (K_EDGES[1:] + K_EDGES[:-1])               # 0.015 ... 0.195 (19 bins)
_ELLS = np.array([0, 2, 4])
_GL_MU, _GL_W = np.polynomial.legendre.leggauss(40)
_L = np.stack([np.ones_like(_GL_MU), 0.5 * (3 * _GL_MU**2 - 1), (35 * _GL_MU**4 - 30 * _GL_MU**2 + 3) / 8])


def to_centres(P, kw):
    """(3, Nw) on the window -> (3, Nb) at K_CENTRES, cubic spline in ln k (same for both codes)."""
    return _CubicSpline(np.log(np.asarray(kw)), np.asarray(P), axis=1)(np.log(K_CENTRES))


def multipole_covariance(P, k, V, dk):
    """Gaussian covariance of (P0, P2, P4) per k-bin, (Nk, 3, 3):
    C_ll'(k) = (2l+1)(2l'+1)/N_k * int_{-1}^{1} L_l L_l' P(k,mu)^2 dmu, N_k = V k^2 dk/(2 pi^2).
    P(k, mu) = sum_l P_l L_l already contains the shot noise -- nothing is added (Review Focus 2)."""
    P = np.asarray(P, dtype=float)
    pkmu = P.T @ _L                                           # (Nk, 40)
    Nk = V * np.asarray(k) ** 2 * dk / (2 * np.pi**2)
    integ = np.einsum("m,am,bm,km->kab", _GL_W, _L, _L, pkmu**2)
    return np.outer(2 * _ELLS + 1, 2 * _ELLS + 1)[None] * integ / Nk[:, None, None]


def _cinv(C, v):
    return np.linalg.solve(C, np.asarray(v, dtype=float).T[..., None])[..., 0]    # (Nk, 3)


def delta_chi2(dP, C) -> float:
    return float(np.sum(np.asarray(dP).T * _cinv(C, dP)))


def counterterm_headroom(parts, c_ref) -> float:
    """Smallest c >= 0 with sum_slices (d0 + c D)^T C^-1 (d0 + c D) = 1, D = (d1 - d0)/c_ref. Exact:
    the counterterm residual is linear in c (c0 = c2 = c4 = c). parts: iterable of (d0, d1, C).
    0.0 if already >= 1 at c = 0; inf if the counterterm adds no residual."""
    a = b = e = 0.0
    for d0, d1, C in parts:
        D = (np.asarray(d1) - np.asarray(d0)) / c_ref
        a += float(np.sum(D.T * _cinv(C, D)))
        b += float(np.sum(D.T * _cinv(C, d0)))
        e += float(np.sum(np.asarray(d0).T * _cinv(C, d0)))
    if e >= 1.0:
        return 0.0
    if a <= 0.0:
        return float("inf")
    return float((-b + np.sqrt(b * b - a * (e - 1.0))) / a)
```

- [ ] **Step 4: Run the numerics tests to verify they pass**

Run: `slurm/run-ps1loop-pytest.sh tests/test_pfs_survey.py -q`
Expected: `9 passed`.

- [ ] **Step 5: Write the gate test**

Create `tests/test_ps1loop_pfs_gate.py`:

```python
# tests/test_ps1loop_pfs_gate.py
"""Error-bar gate (spec §5.6): Delta chi^2 of ps_1loop - clax-pt (P0, P2, P4) over the seven
PFS slices of Takada et al. (2014), Gaussian l-l' covariance, 0.01 <= k <= 0.2 h/Mpc, must be
< 1 for each cosmology. Reports per-slice Delta chi^2, max |dP|/sigma and the counterterm
headroom c*. vc.PFS_CASES (5) in full mode, lcdm_fiducial under --fast."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ps_1loop_jax")

from scripts import validation_cosmologies as vc
from tests import ept_campaign_utils as cu
from tests import pfs_survey as pfs
from tests import ps1loop_bridge as br


def pytest_generate_tests(metafunc):
    if "case" in metafunc.fixturenames:
        fast = metafunc.config.getoption("--fast", default=False)
        metafunc.parametrize("case", ["lcdm_fiducial"] if fast else list(vc.PFS_CASES))


def test_pfs_error_bar_gate(case):
    parts, per_slice, sig_max = [], [], 0.0
    for sl in pfs.load_slices():
        inp = br.shared_inputs(cu.require_reference(case, sl.z_mid))
        d = {}
        for c in (0.0, pfs.C_REF):
            ps, cl, kw = br.spectra_at(inp, pfs.bias_point(sl.z_mid, c=c), ndens=sl.nbar)
            d[c] = pfs.to_centres(ps - cl, kw)
            if c == pfs.C_REF:
                C = pfs.multipole_covariance(pfs.to_centres(cl, kw), pfs.K_CENTRES, sl.V, pfs.DK)
        chi = pfs.delta_chi2(d[pfs.C_REF], C)
        sigma = np.sqrt(np.stack([C[:, i, i] for i in range(3)]))
        sig = float(np.max(np.abs(d[pfs.C_REF]) / sigma))
        per_slice.append((sl.z_mid, chi))
        sig_max = max(sig_max, sig)
        parts.append((d[0.0], d[pfs.C_REF], C))
        br.log_record(layer="pfs", case=case, z=sl.z_mid, preset="slice",
                      errors={"delta_chi2": {"err": chi, "k": float("nan")},
                              "max_dP_over_sigma": {"err": sig, "k": float("nan")}})
    total = sum(chi for _, chi in per_slice)
    c_star = pfs.counterterm_headroom(parts, pfs.C_REF)
    br.log_record(layer="pfs", case=case, z=float("nan"), preset="survey",
                  errors={"delta_chi2": {"err": total, "k": float("nan")}},
                  extra={"per_slice": per_slice, "headroom_c": c_star, "C_REF": pfs.C_REF})
    slices = " ".join(f"{z:.1f}:{chi:.3f}" for z, chi in per_slice)
    print(f"{case} PFS dchi2 {total:.4f} [{slices}] max|dP|/sigma {sig_max:.3f} headroom c* {c_star:.2f} (Mpc/h)^2")
    assert total < 1.0, (f"ERROR {case}: PFS Delta chi^2 = {total:.3f} >= 1 (per slice {slices}); "
                         f"counterterm headroom c* = {c_star:.2f} (Mpc/h)^2 at C_REF = {pfs.C_REF}")
```

- [ ] **Step 6: Run the gate**

Run: `slurm/run-ps1loop-pytest.sh tests/test_ps1loop_pfs_gate.py --fast -q`
Expected: `1 passed`, with one line giving Δχ², the per-slice split and c*.

Run: `slurm/run-ps1loop-pytest.sh tests/test_ps1loop_pfs_gate.py -q`
Expected: `5 passed`. A failure is reported to the user with the per-slice split and c*. A c* below C_REF means the counterterm model difference alone exceeds the PFS errors at the placeholder amplitude. That is a scientific result for the user to weigh, not a threshold to adjust.

- [ ] **Step 7: Fast suite, then commit**

Run: `slurm/run-ps1loop-fast-suite.sh` → `PASS`.

```bash
git add tests/pfs_survey.py tests/test_pfs_survey.py tests/test_ps1loop_pfs_gate.py
git commit -m "test(ps1loop): PFS error-bar gate -- survey Delta chi^2 < 1, Gaussian l-l' covariance, headroom

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Convergence ladder (diagnostic)

**Files:**
- Create: `scripts/ps1loop_convergence_ladder.py`
- Test: `tests/test_ps1loop_ladder.py`

**Interfaces:**
- Consumes: `br.ps_model(rbao, nfft=, mu_grid=)`, `br.ps_batch(model, inp, kw, ndens=, num=)`, `br.clax_batch`, `br.clax_ept`, `br.shared_inputs`, `br.log_record`, `br.IR_KS`, `br.THETA_NAMES`, `br.ELLS`; `pfs.load_slices`, `pfs.bias_point`, `pfs.to_centres`, `pfs.multipole_covariance`, `pfs.delta_chi2`, `pfs.K_CENTRES`, `pfs.DK`; `ps_1loop_jax.ir_resum.remove_wiggle(kh, pk, n_min=140, n_max=210)`; `cu.load_reference`, `cu.window`, `cu.rel`.
- Produces: `run(points=POINTS, rungs=RUNGS) -> list[dict]` (keys `case z rung delta_chi2 l0 l2 l4`), CLI `python scripts/ps1loop_convergence_ladder.py`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ps1loop_ladder.py`:

```python
# tests/test_ps1loop_ladder.py
"""Smoke test of the convergence ladder (spec §7). Exempt from the multi-cosmology rule:
a diagnostic that gates nothing; this checks its plumbing at one point -- in particular that
the wiggle-cut patch takes effect AND is fully undone (jit caches; Review Focus 1)."""
from __future__ import annotations

import pytest

pytest.importorskip("ps_1loop_jax")

from tests import ept_campaign_utils as cu
from tests import ps1loop_bridge as br


def test_ladder_patch_takes_effect_and_is_undone(tmp_path, monkeypatch):
    if cu.load_reference("lcdm_fiducial", 2.2) is None:
        pytest.skip("reference missing: lcdm_fiducial z=2.2 -- run slurm/classpt-refgen-pfs.sbatch")
    monkeypatch.setattr(br, "LOG_DIR", tmp_path)
    monkeypatch.setattr(br, "ERROR_LOG", tmp_path / "errors.jsonl")
    from scripts import ps1loop_convergence_ladder as lad
    recs = lad.run(points=(("lcdm_fiducial", 2.2),), rungs=("default", "dewiggle_classpt", "default"))
    assert [r["rung"] for r in recs] == ["default", "dewiggle_classpt", "default"]
    assert recs[2]["delta_chi2"] == pytest.approx(recs[0]["delta_chi2"], rel=1e-12)
    assert abs(recs[1]["delta_chi2"] - recs[0]["delta_chi2"]) > 1e-9 * max(recs[0]["delta_chi2"], 1e-30)
    assert len((tmp_path / "errors.jsonl").read_text().splitlines()) == 3
```

- [ ] **Step 2: Run it to verify it fails**

Run: `slurm/run-ps1loop-pytest.sh tests/test_ps1loop_ladder.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.ps1loop_convergence_ladder'`.

- [ ] **Step 3: Write the ladder**

Create `scripts/ps1loop_convergence_ladder.py`:

```python
#!/usr/bin/env python
"""Convergence ladder for ps_1loop's settings (spec §7) -- a diagnostic, not a gate.

For each (case, z) point and each rung (one setting changed at a time), logs the
ps_1loop-vs-clax-pt error per l (campaign metric, window) and the slice Delta chi^2 at the
PFS bias point to test_logs/ps1loop/errors.jsonl (layer "ladder"). A failure of the gates
that a rung removes is a resolution effect (spec §6), filed as a recommended setting.

Usage (igpu node, env clax):
  python scripts/ps1loop_convergence_ladder.py [--points lcdm_fiducial:2.2 ...] [--rungs default nfft512 ...]
"""
from __future__ import annotations

import argparse
import contextlib
import functools

import jax
import numpy as np

from tests import ept_campaign_utils as cu
from tests import pfs_survey as pfs
from tests import ps1loop_bridge as br

POINTS = (("lcdm_fiducial", 2.2), ("w0wa_m07_m10", 2.2))
RUNGS = ("default", "nfft512", "simpson1024", "mugrid201", "dewiggle_classpt", "rbao110")


@contextlib.contextmanager
def _wiggle_cuts(n_min: int, n_max: int):
    """Swap ps_1loop's DST mode cuts (ir_resum.remove_wiggle, default 140-210) for CLASS-PT's
    120-240. Caches are cleared on entry and exit: compiled code would keep the old cuts."""
    from ps_1loop_jax import ir_resum
    original = ir_resum.remove_wiggle
    ir_resum.remove_wiggle = functools.partial(original, n_min=n_min, n_max=n_max)
    jax.clear_caches()
    try:
        yield
    finally:
        ir_resum.remove_wiggle = original
        jax.clear_caches()


def _model_and_num(rung: str, inp: dict):
    if rung == "nfft512":
        return br.ps_model(inp["rbao"], nfft=512), 256
    if rung == "simpson1024":
        return br.ps_model(inp["rbao"]), 1024
    if rung == "mugrid201":
        return br.ps_model(inp["rbao"], mu_grid=201), 256
    if rung == "rbao110":
        return br.ps_model(110.0), 256          # ps_1loop's default BAO scale (finding 1, spec §9)
    if rung == "dewiggle_classpt":
        from ps_1loop_jax import PowerSpectrum1Loop
        return PowerSpectrum1Loop(do_irres=True, rbao=inp["rbao"], ks=br.IR_KS), 256   # fresh, uncached
    if rung == "default":
        return br.ps_model(inp["rbao"]), 256
    raise ValueError(f"unknown rung {rung!r}; known: {RUNGS}")


def run(points=POINTS, rungs=RUNGS) -> list[dict]:
    records = []
    slices = {sl.z_mid: sl for sl in pfs.load_slices()}
    for case, z in points:
        ref = cu.load_reference(case, z)
        if ref is None:
            raise SystemExit(f"ERROR reference missing: {case} z={z} -- run slurm/classpt-refgen-pfs.sbatch")
        sl, inp = slices[z], br.shared_inputs(ref)
        mask = cu.window(inp["k_h"])
        kw = inp["k_h"][mask]
        th = np.array([[pfs.bias_point(z)[n] for n in br.THETA_NAMES]], dtype=float)
        cl = br.clax_batch(br.clax_ept(inp), inp, mask, ndens=sl.nbar)(th)[0]
        C = pfs.multipole_covariance(pfs.to_centres(cl, kw), pfs.K_CENTRES, sl.V, pfs.DK)
        for rung in rungs:
            ctx = _wiggle_cuts(120, 240) if rung == "dewiggle_classpt" else contextlib.nullcontext()
            with ctx:
                model, num = _model_and_num(rung, inp)
                ps = br.ps_batch(model, inp, kw, ndens=sl.nbar, num=num)(th)[0]
            errs = {f"l{l}": {"err": cu.rel(ps[i], cl[i]), "k": float("nan")} for i, l in enumerate(br.ELLS)}
            chi = pfs.delta_chi2(pfs.to_centres(ps - cl, kw), C)
            br.log_record(layer="ladder", case=case, z=z, preset=rung, errors=errs, extra={"delta_chi2": chi})
            rec = {"case": case, "z": z, "rung": rung, "delta_chi2": chi, **{k: v["err"] for k, v in errs.items()}}
            records.append(rec)
            print(f"{case} z={z} {rung:17s} l0 {100 * rec['l0']:.3f}% l2 {100 * rec['l2']:.3f}% "
                  f"l4 {100 * rec['l4']:.3f}% dchi2 {chi:.4f}")
    return records


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--points", nargs="+", default=[f"{c}:{z}" for c, z in POINTS])
    p.add_argument("--rungs", nargs="+", default=list(RUNGS), choices=list(RUNGS))
    a = p.parse_args(argv)
    run(points=tuple((s.split(":")[0], float(s.split(":")[1])) for s in a.points), rungs=tuple(a.rungs))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test, then the full ladder**

Run: `slurm/run-ps1loop-pytest.sh tests/test_ps1loop_ladder.py -q`
Expected: `1 passed`.

Run the full ladder on an igpu node (it appends to `test_logs/ps1loop/errors.jsonl` in the worktree):

```bash
srun --partition=main --gres=gpu:1 --time=02:00:00 \
  bash -lc 'eval "$(micromamba shell hook --shell bash)"; micromamba activate clax; \
            cd /home/n2minh/clax-ps1loop; export PYTHONPATH=$PWD JAX_PLATFORMS=cpu; \
            python scripts/ps1loop_convergence_ladder.py'
```
Expected: 12 lines (2 points × 6 rungs). Record them for the CHANGELOG and findings (Task 10).

- [ ] **Step 5: Fast suite, then commit**

Run: `slurm/run-ps1loop-fast-suite.sh` → `PASS`.

```bash
git add scripts/ps1loop_convergence_ladder.py tests/test_ps1loop_ladder.py
git commit -m "feat(ps1loop): convergence ladder for ps_1loop settings (nfft, Simpson, AP grid, DST cuts, rbao)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Findings, CHANGELOG, final verification

**Files:**
- Create: `docs/ps1loop-findings.md`
- Modify: `CHANGELOG.md` (new dated entry at the top of the log section)

**Interfaces:**
- Consumes: `test_logs/ps1loop/errors.jsonl` from Tasks 4–9.

- [ ] **Step 1: Summarise the log**

```bash
python - <<'EOF'
import json, math
recs = [json.loads(l) for l in open("test_logs/ps1loop/errors.jsonl")]
for layer in ("coef", "closure", "irno", "real", "grad"):
    rs = [(r["case"], r["z"], n, e["err"]) for r in recs if r["layer"] == layer for n, e in r["errors"].items()
          if e["err"] is not None and math.isfinite(e["err"])]
    if rs:
        w = max(rs, key=lambda t: t[3])
        print(f"{layer:8s} n={len({(c, z) for c, z, *_ in rs})} worst {w[2]} {100*w[3]:.3f}% ({w[0]} z={w[1]})")
for r in recs:
    if r["layer"] == "pfs" and r["preset"] == "survey":
        print(f"pfs      {r['case']:22s} dchi2 {r['errors']['delta_chi2']['err']:.4f} headroom c* {r['extra']['headroom_c']:.2f}")
for r in recs:
    if r["layer"] == "ladder":
        print(f"ladder   {r['case']} z={r['z']} {r['preset']:17s} dchi2 {r['extra']['delta_chi2']:.4f}")
EOF
```
Expected: one `worst` line per layer, one `pfs` line per cosmology, 12 `ladder` lines. These numbers fill Steps 2 and 3.

- [ ] **Step 2: Draft the ps_1loop findings (not posted)**

Create `docs/ps1loop-findings.md` with one section per finding. Each section has: title; what happens (with file:line in ps_1loop); consequence for a PFS fit (quantified where Step 1 measured it); suggested fix. Start from spec §9 items 1–5:

1. `rbao` default 110 Mpc/h vs r_s·h ≈ 99 Mpc/h. Quantify with the `rbao110` ladder Δχ² at z = 2.2.
2. `utils_math.legendre(n, x)` returns L₄ for any n ∉ {0, 2} (utils_math.py:7-14).
3. `get_pk_real` / `get_pk_gm_real` have no IR resummation (ps_1loop.py:158).
4. `clax_adapter` feeds total-matter P_m, `fast_cl`, k_max ≈ 1 h/Mpc (clax_adapter.py:97-148). The bias is ≈ 2f_ν in P: 0.9% at 0.06 eV, ≈ 4% at 0.3 eV.
5. `jit` with `static_argnames=['self']`: attribute changes after the first call are ignored (ps_1loop.py:286 and every jitted method).

Then add every gate finding attributed "suspect ps_1loop", and every ladder rung that moves Δχ² by more than 10% (a "resolution" verdict), with its numbers.

Then ask the user whether to open these as issues on MinhMPA/ps_1loop_jax-for-pfs. Do not post without an explicit yes.

- [ ] **Step 3: Write the CHANGELOG entry**

Add at the top of the dated log in `CHANGELOG.md` (date in `Mon D, YYYY` format, the day of writing). Fill every bracketed field with the Step 1 numbers:

```markdown
### Oct D, 2026: clax-pt vs ps_1loop_jax second-oracle validation (feat/ps1loop-cross-validation)

Spec `docs/superpowers/specs/2026-10-05-clax-pt-vs-ps1loop-validation-design.md`; ps_1loop `[sha from Task 0 Step 3]`.
Shared-input comparison on CLASS-PT's own P_cb, f, r_s h, AP; CLASS-PT tie-breaker.
- Coefficient gate (14 exact monomials x l=0,2,4): [n]/42 files pass, worst [value].
- Model-difference closure (c0, c2, c4, cfog, cfog*b1, cfog*b1^2, a0): [n]/42 close, worst [value]; largest model difference [monomial, %].
- Counterterm map exact with IR off (z=0): [n]/14.
- Real space, IR off: [n]/42; clax-pt IR-off vs CLASS-PT IR-off: [n]/3.
- Gradients (f, alpha_perp, alpha_para, k-stretch, amplitude): cross-code [n]/3, ps_1loop AD-vs-FD [n]/3.
- PFS error-bar gate (Takada 2014, 7 slices): Delta chi^2 per cosmology [5 values]; headroom c* [5 values] (Mpc/h)^2 at placeholder C_REF = 2.
- Ladder (z=2.2): [one line per rung that moved Delta chi^2 by >10%].
- New references: 35 PFS-slice + 3 IR-off CLASS-PT files (slurm/classpt-refgen-pfs.sbatch).
- ps_1loop findings drafted in docs/ps1loop-findings.md ([n] items; posted: yes/no).
- Failed approaches / dead ends: [list, or "none"].
```

- [ ] **Step 4: Final verification**

Run: `slurm/run-ps1loop-pytest.sh tests/test_pfs_survey.py tests/test_classpt_pfs_references.py tests/test_ps1loop_bridge.py tests/test_ps1loop_coefficients.py tests/test_ps1loop_model_difference.py tests/test_ps1loop_real_space.py tests/test_ps1loop_gradients.py tests/test_ps1loop_pfs_gate.py tests/test_ps1loop_ladder.py -q`
Expected: every test passes, or is an `xfail(strict=True)` that the user approved in an earlier task. Total: 9 + 9 + 12 + 42 + 56 + 45 + 6 + 5 + 1 = 185.

Run: `slurm/run-ps1loop-fast-suite.sh` → `PASS`.
Run: `git diff CHANGELOG.md | grep -n '\[' | head` — no bracketed placeholder may remain in the new entry (Markdown links excepted).

- [ ] **Step 5: Commit**

```bash
git add docs/ps1loop-findings.md CHANGELOG.md
git commit -m "docs(ps1loop): validation results and drafted ps_1loop findings

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Then ask the user whether to push `feat/ps1loop-cross-validation` and open a PR on MinhMPA/clax-pt. Pushing is outward-facing; do not push without approval.

### Task 11 (added Oct 14, 2026 by user decision)

Added after the final whole-branch review, by the user's decisions of Oct 14, 2026. Requirements, commit sequence,
runs and evidence: `.superpowers/sdd/2026-10-05-clax-pt-vs-ps1loop-validation/task-11-brief.md` (with
`final-fix-brief.md` §A; SDD ledger, not committed; the decisions are restated in CHANGELOG Oct 14, 2026). This
plan stays the document of record; in short:

- Commit 1 (RED): the derivative gate runs ps at nfft 512 (`PS_NFFT_GATE`) with the restricted closed form; the
  stretch discrimination lines fail (job 25318).
- Commit 2 (GREEN): the closed form D carries ps's IR scheme on every bias row clax keeps and the matter loops'
  second-order-in-P_w scheme term (spec §4.2 items 5-6); all 45 derivative lines pass in 3 cosmologies (job 25321).
- Commit 3: the ladder's remainder smoke test checks the nfft512 rung against the derivative gate itself
  (rel 1e-10) instead of golden numbers; full ladder rerun (job 25324).
- Commit 4: the PFS error-bar gate is carried as `xfail(strict=True)` with the decision text; the fast-suite wrapper
  keeps only the Rosenbrock deselect; the counterterm / fingers-of-god / a0 closure gets the discrimination rule
  where its prediction is not identically zero, and runs ps at nfft 512 (user decision of Oct 14, 2026, after the
  read-only diagnostic job 25328).
- Commit 5: spec, findings (finding 2 attributed; new finding 9), CHANGELOG `### Oct 14, 2026` entry, this section,
  and the final-review fix items; final runs: the nine campaign files in full and the fast suite through the wrapper.
