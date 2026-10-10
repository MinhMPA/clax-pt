# Recombination staging (smsharma/clax#48) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Gradients through `thermodynamics_solve` become precise, smooth and independent of solver step placement, platform and cosmology, while x_e stays within 0.1% of CLASS. Helium and hydrogen follow CLASS's Saha → ODE staging with one L-stable step and no clips.

**Architecture:** One shared L-stable two-stage linearly implicit (ROS2) scalar step advances both Peebles equations inside the existing recombination `lax.scan`. Helium stays on Saha above z = 2870, hydrogen above z = 1600, and CLASS's `f2` smoothstep blends each handoff. Tests attach at five seams: S0 (the step on a scalar ODE), S1 (thermodynamics tables and their tangents), S2 (δ_m(today) through the matter solve), S3 (the production `custom_vjp` entry) and S4 (C_ℓ at `fit_cl`).

**Tech Stack:** JAX 0.9 (float64), diffrax, pytest; CLASS-PT 3.3.4 `classy` (the `classpt` env) for reference data only; SLURM on igpu (CPU by default, V100 for GPU tasks).

**Spec:** `docs/superpowers/specs/2026-10-09-recombination-staging-design.md` (read it; this plan argues from it).

## Global Constraints

- CLASS is the **primary oracle**. Every new equation carries a CLASS file:line comment. Citations use the CLASS-PT tree at `/home/n2minh/CLASS-PT` (`_VERSION_` v3.3.4); no stock `class_public-3.3.4` exists on this machine.
- Bars are **code-accuracy gates**: never loosened, no fudge factors. If a bar fails, report BLOCKED with numbers.
- Multi-cosmology: S1–S4 run at 3–10 cosmologies (fiducial only under `--fast`). S0 is cosmology-independent numerics and says so in its docstring.
- **Red first:** each new test is shown failing on the unfixed code (upstream `4d084b5`) on at least one grid or cosmology before its fix lands. The red evidence (job ID and numbers) goes in the commit message.
- **Every commit passes the gate:** `tests/ --fast` (CPU job), plus full mode for the files this branch touches and `tests/test_thermodynamics.py`.
  - `tests/test_solver_selection.py::TestRosenbrockPk::test_pk_rosenbrock_vs_kvaerno5` fails identically on `4d084b5` (max_steps). It is pre-existing and must be reported as such, not counted against this branch.
- **Any deviation from the probe's VE formulas is a design change:** stop and report.
- **D10 stop rule:** if total `fit_cl` wall-clock grows by more than ~5%, stop and report.
- Commits:
  - explicit paths only;
  - message via `git commit -F <file>`, written to `/lustre/work/n2minh/clax-probes/recomb/msg-<task>.txt`;
  - trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`;
  - `git commit` runs as its own command, because a hook blocks commands where `git commit` is followed by ` -n`.
- CHANGELOG headings use `Mon D, YYYY` format.
- **Compute:**
  - Heavy JAX runs only in SLURM jobs. The login node is limited to collection, `py_compile` and the S0 tests (each wrapped in `timeout 300`).
  - Pass variables through the submit environment (`VAR=... sbatch --export=ALL ...`), never `--export=ALL,VAR=...`.
  - Poll with `sacct -j <id> -X`.
- Never touch jobs you did not submit, or any conda env. CLASS-PT runs only in the `classpt` env.
- A PreToolUse "Fact-Forcing Gate" hook may require stated facts before Bash/Edit/Write. State them, then retry.
- No upstream PR without the user's go-ahead. The spec and plan stay on the branch and are left out of the PR.

## Review Focus

1. **Bounds without clips:** with the clips removed, x_e must stay in [0, 1 + 2f_He] at every cosmology and on both grids. Test in Task 4.
2. **T_cmb variation** (a parameter that shifts every recombination redshift but is not in the spec's θ list): spike ratio < 10 for θ = T_cmb. Test in Task 4.
3. **A coarser grid than `fit_cl` (1500 points):** helium and hydrogen windows still spike-free (the probe measured 5.1 and 1.78). Test in Task 4. The T_b Taylor switch near z ≈ 624 is out of scope, so z < 800 is excluded.
4. **jit vs eager primal:** the same x_e table to 10⁻¹² (the probe measured 8.7×10⁻¹⁵ for VE, against 2.7×10⁻⁵ upstream). Test in Task 4.
5. **vmap over parameters** (ensembles, emulator training): `vmap` over h gives the same tables as a Python loop, to 10⁻¹². Test in Task 4.

---

## File structure

| File | Responsibility |
|---|---|
| `clax/thermodynamics.py` (modify) | Constants for the stage boundaries; `_smoothstep`, `_ros2_scalar_step`, `_ros2_advance` (internal); helium staging; hydrogen staging; `_f_He` helper (D4) |
| `scripts/generate_class_reference.py` (modify) | New `generate_recfast_hesw1_thermo` plus the `--recfast-hesw1-thermo` CLI mode |
| `reference_data/<name>/thermo_recfast_hesw1.npz` and `.json` (create, ×10) | CLASS RECFAST `Heswitch=1` thermodynamics and its provenance |
| `tests/conftest.py` (modify) | `COSMOLOGY_GRID_RECOMB` and fixtures `recomb_cosmology`, `recomb_downstream_cosmology` |
| `tests/recomb_test_utils.py` (create) | Shared presets, table and tangent helpers, spike ratio, visibility peak, the S2/S3/S4 functionals |
| `tests/test_recomb_reference.py` (create) | Sanity of the committed reference data |
| `tests/test_recomb_step.py` (create) | S0 |
| `tests/test_recomb_thermo.py` (create) | S1 plus Review Focus 1–5, and the D4 test |
| `tests/test_recomb_downstream.py` (create) | S2, S3 |
| `tests/test_recomb_cl.py` (create) | S4 |
| `tests/test_recomb_gpu.py` (create) | S1 GPU-only check. It must **not** pin JAX to CPU at import. |
| `docs/adr/0003-recombination-staging.md` (create), `docs/adr/0001-...md` (modify) | Decision record; a note on ADR 0001 |
| `CHANGELOG.md` (modify) | Entry |

Job tooling, not committed: `/lustre/work/n2minh/clax-probes/recomb/` (`pytest.sbatch`, `run.sbatch`, `class.sbatch`, red snapshots, logs).

---

### Task 1: CLASS reference data, recombination cosmology grid, job tooling

**Files:**
- Modify: `scripts/generate_class_reference.py` (add the function, constants, `import sys` and the CLI mode; keep `main()`'s default behaviour)
- Modify: `tests/conftest.py` (after `nulcdm_cosmology = ...`)
- Create: `tests/test_recomb_reference.py`
- Create: `reference_data/{lcdm_fiducial,h_high,omega_b_high,omega_cdm_low,ns_high,YHe_022,YHe_028,Neff_2,Neff_4,massive_nu_015}/thermo_recfast_hesw1.{npz,json}`

**Interfaces:**
- Produces:
  - `COSMOLOGY_GRID_RECOMB: dict[str, dict]`;
  - fixtures `recomb_cosmology -> (name, CosmoParams)` (10 points) and `recomb_downstream_cosmology -> (name, CosmoParams)`, the latter over `lcdm_fiducial`, `omega_b_high` and `YHe_028`;
  - the file `reference_data/<name>/thermo_recfast_hesw1.npz`, with keys `z` (ascending, 4000 points over z ∈ [200, 8000]), `x_e`, `Tb`, `g`, `kappa_dot`.

- [ ] **Step 1: Job tooling.** Create `/lustre/work/n2minh/clax-probes/recomb/pytest.sbatch`:

```bash
#!/bin/bash -l
#SBATCH --job-name=recomb-pytest
#SBATCH --output=/lustre/work/n2minh/clax-probes/recomb/%x.out.%j
#SBATCH --partition=main
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=12 --mem=96GB
#SBATCH --time=24:00:00
#SBATCH --nodelist=igpu01,igpu02,igpu03,igpu04,igpu05,igpu06,igpu07,igpu08
set -euo pipefail
eval "$(micromamba shell hook --shell bash)"; micromamba activate clax
if [ "${JAX_PLATFORMS:-cpu}" = "cuda" ]; then
  NVIDIA_SP=$(python -c "import site; print(site.getsitepackages()[0])")/nvidia
  for d in "${NVIDIA_SP}"/*/lib; do export LD_LIBRARY_PATH="${d}:${LD_LIBRARY_PATH:-}"; done
fi
SRC=${SRC:-/home/n2minh/clax-recomb}
export JAX_PLATFORMS=${JAX_PLATFORMS:-cpu} OMP_NUM_THREADS=12 PYTHONUNBUFFERED=1 PYTHONPATH=$SRC
cd "$SRC"
python -c "import clax; print('clax from', clax.__file__)"
echo "src $SRC host $(hostname) JAX_PLATFORMS=$JAX_PLATFORMS args: ${PYTEST_ARGS}"
set +e; eval python -u -m pytest ${PYTEST_ARGS} -p no:cacheprovider -rs --verbosity=2 -s; rc=$?; set -e
test "$rc" -eq 0 && echo PASS || { echo "FAIL rc=$rc"; exit "$rc"; }
```

Create `run.sbatch`: identical except job name `recomb-run` and, in place of the pytest line, `python -u "${SCRIPT}" ${SCRIPT_ARGS:-}`.

Create `class.sbatch`: job name `recomb-class`, `--cpus-per-task=4 --mem=16GB --time=2:00:00`, `micromamba activate classpt`, no JAX exports, `cd /home/n2minh/clax-recomb`, then `python -u scripts/generate_class_reference.py --recfast-hesw1-thermo`.

GPU runs reuse `pytest.sbatch` or `run.sbatch` with `JAX_PLATFORMS=cuda` in the submit env and `--gres=gpu:1 --time=6:00:00`.

**Red snapshot recipe** (used by later tasks). The unfixed code plus the new tests and references:

```bash
R=/lustre/work/n2minh/clax-probes/recomb/red-<task>; mkdir -p $R
git -C /home/n2minh/clax-recomb archive 4d084b5 | tar -x -C $R
cp /home/n2minh/clax-recomb/tests/{conftest.py,recomb_test_utils.py,test_recomb_*.py} $R/tests/
for d in /home/n2minh/clax-recomb/reference_data/*/; do n=$(basename $d); [ -f $d/thermo_recfast_hesw1.npz ] && mkdir -p $R/reference_data/$n && cp $d/thermo_recfast_hesw1.* $R/reference_data/$n/; done
```

It is submitted as `SRC=$R PYTEST_ARGS='...' sbatch --job-name=recomb-<task>-red --export=ALL /lustre/work/n2minh/clax-probes/recomb/pytest.sbatch`. Its log line `clax from $R/clax/__init__.py` proves the snapshot ran.

- [ ] **Step 2: Add the cosmology grid.** In `tests/conftest.py`, after `nulcdm_cosmology = _make_cosmology_fixture(COSMOLOGY_GRID_NULCDM)`:

```python
# Recombination staging (smsharma/clax#48): the 5 LCDM points plus helium,
# N_eff and massive-neutrino extremes. Names match reference_data/<name>/
# thermo_recfast_hesw1.npz (CLASS RECFAST, recfast_Heswitch=1).
NEFF_NCDM = (0.71611 / (4.0 / 11.0) ** (1.0 / 3.0)) ** 4  # 1.0132: one ncdm species' early N_eff
COSMOLOGY_GRID_RECOMB = {
    **COSMOLOGY_GRID_LCDM,
    "YHe_022": {"Y_He": 0.22},
    "YHe_028": {"Y_He": 0.28},
    "Neff_2": {"N_ur": 2.0 - NEFF_NCDM},
    "Neff_4": {"N_ur": 4.0 - NEFF_NCDM},
    "massive_nu_015": {"m_ncdm": 0.15},
}
recomb_cosmology = _make_cosmology_fixture(COSMOLOGY_GRID_RECOMB)
recomb_downstream_cosmology = _make_cosmology_fixture(
    {k: COSMOLOGY_GRID_RECOMB[k] for k in ("lcdm_fiducial", "omega_b_high", "YHe_028")})
```

- [ ] **Step 3: Write the failing reference test** `tests/test_recomb_reference.py`:

```python
"""The committed CLASS RECFAST (recfast_Heswitch=1) thermodynamics references
used by the recombination-staging tests (smsharma/clax#48) exist and are
physically sane. Generated by scripts/generate_class_reference.py
--recfast-hesw1-thermo in the classpt env (CLASS-PT build, CLASS v3.3.4)."""
import json
import os

import numpy as np
import pytest

from tests.conftest import COSMOLOGY_GRID_RECOMB

REF = os.path.join(os.path.dirname(__file__), "..", "reference_data")
NOT4 = 3.9715  # CLASS thermodynamics.h:707


@pytest.mark.parametrize("name", list(COSMOLOGY_GRID_RECOMB))
def test_recfast_hesw1_reference_is_sane(name):
    path = os.path.join(REF, name, "thermo_recfast_hesw1.npz")
    assert os.path.isfile(path), f"ERROR missing {path}"
    d = np.load(path)
    z, xe, g = d["z"], d["x_e"], d["g"]
    assert z.size == 4000 and np.all(np.diff(z) > 0)
    assert np.all(np.isfinite(xe)) and np.all(xe > 0)
    with open(path.replace(".npz", ".json")) as fh:
        prov = json.load(fh)
    yhe = prov["class_params"]["YHe"]
    f_he = yhe / (NOT4 * (1.0 - yhe))
    assert abs(xe[-1] - (1.0 + 2.0 * f_he)) < 1e-2, (name, xe[-1], 1 + 2 * f_he)   # z = 8000: H, He fully ionised
    assert xe[np.argmin(np.abs(z - 500.0))] < 0.05, name                           # recombined by z = 500
    zp = z[np.argmax(g)]
    assert 1040.0 < zp < 1120.0, (name, zp)                                         # visibility peak
    print(f"{name}: x_e(8000)={xe[-1]:.5f} peak z={zp:.1f} YHe={yhe}")
```

- [ ] **Step 4: Confirm it fails.** On the login node: `timeout 300 env JAX_PLATFORMS=cpu python -m pytest tests/test_recomb_reference.py -q -x`. Expected: FAIL with "ERROR missing".

- [ ] **Step 5: Add the generator** to `scripts/generate_class_reference.py`. Put `import sys` with the other imports at the top, and these definitions directly above `def main()`:

```python
# Recombination staging (smsharma/clax#48): CLASS RECFAST with
# recfast_Heswitch=1 (Sobolev singlet only), the helium physics clax implements.
# Cosmology names/overrides mirror tests/conftest.py COSMOLOGY_GRID_RECOMB.
NEFF_NCDM = (0.71611 / (4.0 / 11.0) ** (1.0 / 3.0)) ** 4
RECOMB_GRID = {
    "lcdm_fiducial": {},
    "h_high": {"h": 0.6736 * 1.10},
    "omega_b_high": {"omega_b": 0.02237 * 1.20},
    "omega_cdm_low": {"omega_cdm": 0.1200 * 0.80},
    "ns_high": {"n_s": 0.9649 * 1.05},
    "YHe_022": {"Y_He": 0.22},
    "YHe_028": {"Y_He": 0.28},
    "Neff_2": {"N_ur": 2.0 - NEFF_NCDM},
    "Neff_4": {"N_ur": 4.0 - NEFF_NCDM},
    "massive_nu_015": {"m_ncdm": 0.15},
}
RECFAST_HESW1_BASE = {
    'h': 0.6736, 'omega_b': 0.02237, 'omega_cdm': 0.1200, 'A_s': 2.1e-9,
    'n_s': 0.9649, 'tau_reio': 0.0544, 'N_ur': 2.0328, 'N_ncdm': 1,
    'm_ncdm': 0.06, 'T_ncdm': 0.71611, 'YHe': 0.2454006,  # = clax CosmoParams default, never BBN
    'recombination': 'RECFAST', 'recfast_Heswitch': 1,
    'tol_background_integration': 1e-12,
}


def generate_recfast_hesw1_thermo(name, overrides, outdir_base):
    """CLASS RECFAST (recfast_Heswitch=1) thermodynamics on a 4000-point
    log(1+z) grid over z in [200, 8000], for tests/test_recomb_*.py.

    Interpolated linearly in log(1+z) from CLASS's own table (~28k rows).
    Writes thermo_recfast_hesw1.npz (z, x_e, Tb, g, kappa_dot) and a .json
    provenance file next to it.
    """
    import classy
    from classy import Class

    pars = dict(RECFAST_HESW1_BASE)
    for k, v in overrides.items():
        pars['YHe' if k == 'Y_He' else k] = v
    c = Class()
    c.set(pars)
    c.compute()
    th = c.get_thermodynamics()
    z_tab = np.asarray(th['z'])
    order = np.argsort(z_tab)
    z_out = np.expm1(np.linspace(np.log1p(200.0), np.log1p(8000.0), 4000))

    def on_grid(key):
        return np.interp(np.log1p(z_out), np.log1p(z_tab[order]), np.asarray(th[key])[order])

    outdir = os.path.join(outdir_base, name)
    os.makedirs(outdir, exist_ok=True)
    np.savez(os.path.join(outdir, 'thermo_recfast_hesw1.npz'),
             z=z_out, x_e=on_grid('x_e'), Tb=on_grid('Tb [K]'),
             g=on_grid('g [Mpc^-1]'), kappa_dot=on_grid("kappa' [Mpc^-1]"))
    with open(os.path.join(outdir, 'thermo_recfast_hesw1.json'), 'w') as fh:
        json.dump({"class_params": pars,
                   "classy": getattr(classy, '__version__', 'unknown'),
                   "build": "CLASS-PT tree (CLASS v3.3.4 thermodynamics), classpt env",
                   "grid": "4000 points, uniform in log(1+z), z in [200, 8000]"}, fh, indent=1)
    c.struct_cleanup()
    c.empty()
    print(f"  {name}: x_e(z=2500)={np.interp(2500.0, z_out, on_grid('x_e')):.6f}")
```

Change `main()` to:

```python
def main():
    outdir = os.path.join(os.path.dirname(__file__), '..', 'reference_data')
    if '--recfast-hesw1-thermo' in sys.argv:
        for name, overrides in RECOMB_GRID.items():
            generate_recfast_hesw1_thermo(name, overrides, outdir)
        return
    generate_model(FIDUCIAL_PARAMS, 'lcdm_fiducial', outdir)
```

- [ ] **Step 6: Generate the references.**
  - Submit `sbatch --export=ALL /lustre/work/n2minh/clax-probes/recomb/class.sbatch` and poll until it completes.
  - Cross-check against the probe's references by interpolating each `/lustre/work/n2minh/clax-probes/helium/out/class_ref_<probe name>_hesw1.npz` (keys `z`, `x_e`) onto the new z grid.
  - The probe names map as `fiducial`→`lcdm_fiducial`, `mnu_015`→`massive_nu_015`, and are otherwise identical. `ns_high` has no probe counterpart.
  - Expected: max |Δx_e/x_e| < 10⁻⁵ for every shared cosmology. Write the numbers into the commit message.

- [ ] **Step 7: Run the reference test.** `timeout 300 env JAX_PLATFORMS=cpu python -m pytest tests/test_recomb_reference.py -q`. Expected: 10 passed.

- [ ] **Step 8: Commit.**

```bash
git add scripts/generate_class_reference.py tests/conftest.py tests/test_recomb_reference.py reference_data/*/thermo_recfast_hesw1.npz reference_data/*/thermo_recfast_hesw1.json
git commit -F /lustre/work/n2minh/clax-probes/recomb/msg-task1.txt
```

Message subject: `test(thermo): CLASS RECFAST Heswitch=1 references at 10 cosmologies (#48)`. The body records the provenance and the Step 6 cross-check numbers.

---

### Task 2: The shared stiff step (D1) and its direct tests (S0)

**Files:**
- Modify: `clax/thermodynamics.py`. Add constants and helpers after `_recfast_dxHe_dlna`; behaviour is unchanged, since nothing calls them yet.
- Create: `tests/test_recomb_step.py`

**Interfaces:**
- Produces, in `clax.thermodynamics`:
  - `_smoothstep(s)`: 3s²−2s³, with s clipped to [0, 1];
  - `_ros2_scalar_step(rate, xe_of, x, h, env0, env1) -> x_next`;
  - `_ros2_advance(rate, xe_of, x0, env_of_s, h_total, n_sub) -> x_end`.
- `rate(xe, x, *env) -> dx/dlna` and `xe_of(x) -> x_e`. `env_of_s(s)` returns the env tuple at fraction s ∈ [0, 1] of the step.

- [ ] **Step 1: Write the failing S0 tests** `tests/test_recomb_step.py`:

```python
"""S0: the shared L-stable ROS2 step on a scalar stiff relaxation with a known
solution (smsharma/clax#48). Cosmology-independent numerics: exempt from the
multi-cosmology rule (CLAUDE.md), which applies to physics-facing tests.

dx/ds = -lam (x - q(s)), q(s) = alpha + beta s has the exact solution
x(s) = q(s) - beta/lam + (x0 - alpha + beta/lam) exp(-lam s).
For this ROS2 (gamma = 1 + 1/sqrt 2) the stability function is
R(z) = (1 + (1 - 2 gamma) z) / (1 - gamma z)^2 > 0 for z < 0, R -> 0: a stiff
relaxation approaches its equilibrium monotonically, with no overshoot.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from clax.thermodynamics import _ros2_advance, _ros2_scalar_step, _smoothstep

ALPHA, BETA = 0.5, 0.3


def _rate(lam):
    return lambda xe, x, s: -lam * (x - (ALPHA + BETA * s))


def _exact(lam, x0, s):
    return ALPHA + BETA * s - BETA / lam + (x0 - ALPHA + BETA / lam) * np.exp(-lam * s)


def _integrate(lam, x0, n):
    h = 1.0 / n
    x = jnp.asarray(x0)
    for j in range(n):
        x = _ros2_scalar_step(_rate(lam), lambda y: y, x, h, (j * h,), ((j + 1) * h,))
    return float(x)


def test_second_order_convergence():
    errs = [abs(_integrate(2.0, 1.0, n) - _exact(2.0, 1.0, 1.0)) for n in (20, 40, 80)]
    r1, r2 = errs[0] / errs[1], errs[1] / errs[2]
    print(f"errors {errs}, ratios {r1:.3f} {r2:.3f}")
    assert 3.5 < r1 < 4.5 and 3.5 < r2 < 4.5


def test_l_stable_lands_on_equilibrium():
    # lam h = 1e6: one step from far away lands on q = ALPHA without overshoot
    x1 = float(_ros2_scalar_step(_rate(1e6), lambda y: y, jnp.asarray(1.0), 1.0, (0.0,), (0.0,)))
    print(f"x1 = {x1:.12f}, q = {ALPHA}")
    assert abs(x1 - ALPHA) < 1e-5 and x1 >= ALPHA


def test_no_excursion_past_one_near_equilibrium():
    # Peebles-like: relax toward q = 1 - 1e-4 from below, stiff (lam h = 1e3)
    lam, qeq = 1e4, 1.0 - 1e-4
    rate = lambda xe, x, s: -lam * (x - qeq)
    xs = [jnp.asarray(0.99)]
    for _ in range(20):
        xs.append(_ros2_scalar_step(rate, lambda y: y, xs[-1], 0.1, (0.0,), (0.0,)))
    xs = np.array([float(v) for v in xs])
    print(f"max x = {xs.max():.12f}")
    assert xs.max() <= qeq + 1e-15 and np.all(np.diff(xs) >= -1e-15)


@pytest.mark.parametrize("which", ["lam", "alpha"])
def test_step_jvp_matches_fd(which):
    def advance(theta):
        lam = theta if which == "lam" else 50.0
        alpha = theta if which == "alpha" else ALPHA
        rate = lambda xe, x, s: -lam * (x - (alpha + BETA * s))
        return _ros2_advance(rate, lambda y: y, jnp.asarray(1.0), lambda s: (s,), 1.0, 10)
    x0 = 50.0 if which == "lam" else ALPHA
    _, tan = jax.jvp(advance, (jnp.asarray(x0),), (jnp.asarray(1.0),))
    eps = 1e-6 * abs(x0)
    fd = (float(advance(x0 + eps)) - float(advance(x0 - eps))) / (2 * eps)
    print(f"{which}: jvp {float(tan):.10e} fd {fd:.10e}")
    assert np.isfinite(float(tan)) and abs(float(tan) - fd) < 1e-6 * max(abs(fd), 1e-12)


def test_smoothstep():
    w = np.asarray(_smoothstep(jnp.linspace(-0.5, 1.5, 9)))
    assert w[0] == 0.0 and w[-1] == 1.0 and abs(float(_smoothstep(0.5)) - 0.5) < 1e-15
```

- [ ] **Step 2: Confirm they fail.** `timeout 300 env JAX_PLATFORMS=cpu python -m pytest tests/test_recomb_step.py -q`. Expected: ImportError (`_ros2_scalar_step` not defined).

- [ ] **Step 3: Implement.** In `clax/thermodynamics.py`, directly after the function `_recfast_dxHe_dlna`:

```python
# ---------------------------------------------------------------------------
# Shared L-stable stiff step for the recombination Peebles equations (#48)
# ---------------------------------------------------------------------------

_ROS2_GAMMA = 1.0 + 1.0 / math.sqrt(2.0)   # ROS2, Verwer et al. 1999 (L-stable)


def _smoothstep(s):
    """CLASS's stage-transition weight f2(s) = 3 s^2 - 2 s^3 (thermodynamics.h:49),
    with s clipped to [0, 1] so it is exactly 0 or 1 outside the window."""
    s = jnp.clip(s, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def _ros2_scalar_step(rate, xe_of, x, h, env0, env1):
    """One two-stage linearly implicit (ROS2) step of dx/dlna = rate(x_e, x, *env).

    W = 1 - gamma h J,  k1 = h f(s_j, x_j)/W,  k2 = (h f(s_j+1, x_j + k1) - 2 k1)/W,
    x_j+1 = x_j + 3/2 k1 + 1/2 k2, with gamma = 1 + 1/sqrt(2). J = d rate/d x at
    the sub-step start with x_e held fixed (ROS2 stays second order for any J).
    L-stable: a stiff relaxation lands on its equilibrium without overshoot, so
    no clip is needed. env0/env1: environment tuples at the sub-step's two ends.
    """
    xe_j = xe_of(x)
    f0, J = jax.jvp(lambda y: rate(xe_j, y, *env0), (x,), (jnp.ones_like(x),))
    W = 1.0 - _ROS2_GAMMA * h * J
    k1 = h * f0 / W
    x2 = x + k1
    f1 = rate(xe_of(x2), x2, *env1)
    k2 = (h * f1 - 2.0 * k1) / W
    return x + 1.5 * k1 + 0.5 * k2


def _ros2_advance(rate, xe_of, x0, env_of_s, h_total, n_sub):
    """Advance x over one grid step of length h_total (in ln a) with n_sub ROS2
    sub-steps; env_of_s(s) gives the environment at fraction s in [0, 1]."""
    h = h_total / n_sub

    def body(j, x):
        s0 = jnp.asarray(j, dtype=jnp.result_type(float)) / n_sub
        return _ros2_scalar_step(rate, xe_of, x, h, env_of_s(s0), env_of_s(s0 + 1.0 / n_sub))

    return jax.lax.fori_loop(0, n_sub, body, x0)
```

- [ ] **Step 4: Run S0.** `timeout 300 env JAX_PLATFORMS=cpu python -m pytest tests/test_recomb_step.py -q -s`. Expected: 6 passed. Paste the printed error ratios into the task report.

- [ ] **Step 5: Gate.** `PYTEST_ARGS='tests/ --fast -q' sbatch --job-name=recomb-t2-fast --export=ALL /lustre/work/n2minh/clax-probes/recomb/pytest.sbatch` must PASS, apart from the pre-existing `test_pk_rosenbrock_vs_kvaerno5`. Behaviour is unchanged, since the helpers are unused.

- [ ] **Step 6: Commit** `clax/thermodynamics.py tests/test_recomb_step.py`, with subject `feat(thermo): shared L-stable ROS2 step for the recombination equations (#48)`. The body carries the S0 ratios.

---

### Task 3: Helium follows CLASS staging (D2) — S1 helium-window tests

**Files:**
- Create: `tests/recomb_test_utils.py`, `tests/test_recomb_thermo.py`
- Modify: `clax/thermodynamics.py` (constants; shared sub-step environment; helium block; escape probability; C_He switch; carry)

**Interfaces:**
- Consumes: `_ros2_advance` and `_smoothstep` (Task 2); `recomb_cosmology` (Task 1).
- Produces:
  - in `tests/recomb_test_utils.py`: `PREC_RB`, `THERMO_GRIDS`, `THETAS`, `QS`, `thermo_prec(n_points)`, `as_traced(p)`, `tables(p, prec) -> dict[loga, xe, cs2, tb, g]`, `tangent_tables(p, prec, theta) -> (primal, tangent)` (numpy), `z_of(loga)`, `spike_ratio(dln, z, zlo, zhi) -> (ratio, z_at)`, `visibility_peak(z, g)`, `load_ref(name)`, `xe_on_ref_grid(prim, ref_z)`;
  - in `tests/test_recomb_thermo.py`: `_prim_tan(name, p, n, theta)` and `_fd_check(recomb_cosmology, theta, zs)`;
  - in `clax/thermodynamics.py`: the constants `_CB1_He1`, `_Z_EARLY_H_RECOMB`, `_DZ_EARLY_H_RECOMB`, `_Z_FULL_H_RECOMB`, `_DZ_FULL_H_RECOMB`; the scan-local names `sub_env`, `dlna_main`, `f_He_xe`; and a scan-carry field `xHeII_ode`.

- [ ] **Step 1: Write the shared test helpers** `tests/recomb_test_utils.py`:

```python
"""Shared helpers for the recombination-staging tests (smsharma/clax#48).
Presets and metrics copy the variant probe (/lustre/work/n2minh/clax-probes/
helium/scripts/thermo_probe.py) so the in-repo numbers are comparable."""
import os
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from clax import PrecisionParams
from clax.background import background_solve
from clax.thermodynamics import thermodynamics_solve

REF_DIR = os.path.join(os.path.dirname(__file__), "..", "reference_data")
PREC_RB = replace(PrecisionParams.fast_cl(), pt_ode_solver="rodas5",
                  pt_ode_rtol=1e-4, pt_ode_atol=1e-4, ode_max_steps=16384)   # spec D8
THERMO_GRIDS = (20000, 3000)          # default grid; fit_cl's thermodynamics grid
THETAS = ("h", "omega_b", "omega_cdm", "Y_He", "N_ur")
QS = ("xe", "cs2", "tb")


def thermo_prec(n_points):
    """Thermodynamics preset on an n_points grid; jax.jvp through
    background_solve needs the direct adjoint."""
    return replace(PREC_RB, th_n_points=n_points, ode_adjoint="direct")


def as_traced(p):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x, jnp.float64), p)


def tables(p, prec):
    bg = background_solve(p, prec)
    th = thermodynamics_solve(p, prec, bg)
    return {"loga": th.xe_of_loga.x, "xe": th.xe_of_loga.y, "cs2": th.cs2_of_loga.y,
            "tb": th.Tb_of_loga.y, "g": th.g_of_loga.y}


def tangent_tables(p, prec, theta):
    """(primal tables, forward tangents d/d theta) as numpy arrays."""
    p = as_traced(p)
    t = jax.tree_util.tree_map(jnp.zeros_like, p).replace(**{theta: jnp.asarray(1.0)})
    prim, tan = jax.jit(lambda pp, tt: jax.jvp(lambda q: tables(q, prec), (pp,), (tt,)))(p, t)
    return (jax.tree_util.tree_map(np.asarray, prim), jax.tree_util.tree_map(np.asarray, tan))


def z_of(loga):
    return 1.0 / np.exp(np.asarray(loga)) - 1.0


def spike_ratio(dln, z, zlo, zhi, half=25, floor=1e-8):
    """max over grid points in [zlo, zhi] of |dln| / median(|dln| over +-half points)."""
    a = np.abs(np.asarray(dln))
    pad = np.pad(a, half, mode="edge")
    med = np.median(np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1), axis=1)
    sel = (z >= zlo) & (z <= zhi)
    r = np.where(sel, a / np.maximum(med, floor), 0.0)
    i = int(np.argmax(r))
    return float(r[i]), float(z[i])


def visibility_peak(z, g, zlo=800.0, zhi=1400.0):
    s = (z >= zlo) & (z <= zhi)
    zz, gg = np.asarray(z)[s], np.asarray(g)[s]
    i = int(np.argmax(gg))
    c = np.polyfit(zz[i - 1:i + 2], gg[i - 1:i + 2], 2)
    return float(-c[1] / (2.0 * c[0]))


def load_ref(name):
    return np.load(os.path.join(REF_DIR, name, "thermo_recfast_hesw1.npz"))


def xe_on_ref_grid(prim, ref_z):
    """clax x_e (CubicSpline in log a) evaluated at the reference redshifts."""
    from clax.interpolation import CubicSpline
    sp = CubicSpline(jnp.asarray(prim["loga"]), jnp.asarray(prim["xe"]))
    return np.asarray(sp.evaluate(jnp.log(1.0 / (1.0 + jnp.asarray(ref_z)))))
```

- [ ] **Step 2: Write the failing helium-window S1 tests** `tests/test_recomb_thermo.py`:

```python
"""S1 (smsharma/clax#48): thermodynamics tables and their forward tangents.
Primary oracle: CLASS RECFAST recfast_Heswitch=1 (reference_data/<name>/
thermo_recfast_hesw1.npz). Bars are code-accuracy gates from the spec.
Multi-cosmology: recomb_cosmology (10 points; fiducial only under --fast)."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from clax.interpolation import CubicSpline
from tests.recomb_test_utils import (QS, THETAS, THERMO_GRIDS, as_traced, load_ref,
                                     spike_ratio, tables, tangent_tables, thermo_prec,
                                     xe_on_ref_grid, z_of)

pytestmark = pytest.mark.slow
_CACHE = {}


def _prim_tan(name, p, n, theta="h"):
    key = (name, n, theta)
    if key not in _CACHE:
        _CACHE[key] = tangent_tables(p, thermo_prec(n), theta)
    return _CACHE[key]


def _fd_check(recomb_cosmology, theta, zs):
    """jvp vs central FD at sampled z, only where FD at relative steps 1e-3 and
    3e-3 agree to < 1e-4 (spec S1, footnote). Bar 1e-3."""
    name, p = recomb_cosmology
    prec = thermo_prec(20000)
    prim, tan = _prim_tan(name, p, 20000, theta)
    th0 = float(getattr(p, theta))
    loga_s = jnp.log(1.0 / (1.0 + jnp.asarray(zs)))
    ev = jax.jit(lambda x, y: CubicSpline(x, y).evaluate(loga_s))
    g = jax.jit(lambda q: tables(q, prec))
    fds = []
    for rel_step in (1e-3, 3e-3):
        hstep = rel_step * abs(th0)
        tp = g(as_traced(replace(p, **{theta: th0 + hstep})))
        tm = g(as_traced(replace(p, **{theta: th0 - hstep})))
        fds.append({q: (np.asarray(ev(tp["loga"], tp[q])) - np.asarray(ev(tm["loga"], tm[q])))
                    / (2 * hstep) for q in QS})
    n_cmp, worst = 0, 0.0
    for q in QS:
        jv = np.asarray(ev(jnp.asarray(prim["loga"]), jnp.asarray(tan[q])))
        f1, f3 = fds[0][q], fds[1][q]
        ok = np.abs(f1 - f3) <= 1e-4 * np.maximum(np.abs(f1), 1e-30)
        n_cmp += int(ok.sum())
        if ok.any():
            worst = max(worst, float(np.max(np.abs(jv[ok] - f1[ok]) / np.maximum(np.abs(f1[ok]), 1e-30))))
    print(f"{name} d/d{theta}: compared {n_cmp}/{len(zs) * len(QS)} points, worst |jvp/FD-1| {worst:.3e}")
    assert n_cmp >= len(zs) * len(QS) // 2, f"ERROR FD unconverged at too many points ({n_cmp}) ({name}, {theta})"
    assert worst <= 1e-3, f"ERROR tangent vs FD {worst:.3e} > 1e-3 ({name}, {theta})"


@pytest.mark.parametrize("n", THERMO_GRIDS)
def test_xe_helium_window_matches_class(recomb_cosmology, n):
    name, p = recomb_cosmology
    prim, _ = _prim_tan(name, p, n)
    ref = load_ref(name)
    sel = (ref["z"] >= 2870.0) & (ref["z"] <= 5000.0)
    rel = np.abs(xe_on_ref_grid(prim, ref["z"][sel]) / ref["x_e"][sel] - 1.0)
    i = int(np.argmax(rel))
    print(f"{name} n={n}: x_e vs CLASS z 2870-5000 max {rel[i]:.3e} at z={ref['z'][sel][i]:.1f}")
    assert rel[i] <= 1e-3, f"ERROR x_e helium window {rel[i]:.3e} > 1e-3 ({name}, n={n})"


@pytest.mark.parametrize("n", THERMO_GRIDS)
@pytest.mark.parametrize("theta", THETAS)
def test_no_derivative_spikes_helium_window(recomb_cosmology, n, theta):
    name, p = recomb_cosmology
    prim, tan = _prim_tan(name, p, n, theta)
    z = z_of(prim["loga"])
    worst = max(spike_ratio(tan[q] / prim[q], z, 1800.0, 6000.0) + (q,) for q in QS)
    print(f"{name} n={n} d/d{theta}: worst helium-window spike ratio {worst[0]:.3g} ({worst[2]} at z={worst[1]:.1f})")
    assert worst[0] < 10.0, f"ERROR spike ratio {worst[0]:.3g} ({name}, {theta}, n={n})"


@pytest.mark.parametrize("n", THERMO_GRIDS)
def test_values_smooth_helium_window(recomb_cosmology, n):
    name, p = recomb_cosmology
    prim, _ = _prim_tan(name, p, n)
    z = z_of(prim["loga"])
    sel = (z >= 1800.0) & (z <= 6000.0)
    d2 = np.abs(np.diff(prim["xe"], 2))[sel[1:-1]]
    bar = 3e-5 * (20000.0 / n) ** 2      # second differences scale as spacing^2 (spec S1)
    print(f"{name} n={n}: max |d2 x_e| over z 1800-6000 = {d2.max():.3e} (bar {bar:.2e})")
    assert d2.max() < bar, f"ERROR x_e not smooth: {d2.max():.3e} ({name}, n={n})"


ZS_HE = np.array([5500., 4500., 3800., 3400., 3200., 3000., 2800., 2600., 2400., 2200., 2000.])


@pytest.mark.parametrize("theta", THETAS)
def test_tangents_match_converged_fd_helium(recomb_cosmology, theta):
    _fd_check(recomb_cosmology, theta, ZS_HE)
```

- [ ] **Step 3: Show RED on the unfixed code.**
  - Make a red snapshot `red-t3` (recipe in Task 1).
  - Submit `SRC=$R PYTEST_ARGS='tests/test_recomb_thermo.py -m slow' sbatch --job-name=recomb-t3-red --export=ALL .../pytest.sbatch`.
  - Expected: the helium x_e test FAILS (probe: 6.3×10⁻³), the spike test FAILS (probe: up to 9.5×10¹⁴), the smoothness test FAILS, and the FD test FAILS or reports unconverged FD.
  - Paste the failing lines into the task report.
  - While the red job runs, implement Steps 4–6. The worktree is safe to edit because the job imports from the snapshot.

- [ ] **Step 4: Constants.** In `clax/thermodynamics.py`, after the line `_T_1_He = 10.0**5.114`:

```python
# HeI ionization temperature, cf. wrap_recfast.c:78 and thermodynamics.c:823
# (const_Tion_HeI = h c L_He1_ion / k_B), ~285,335 K.
_CB1_He1 = _hP_SI * _c_SI * _L_He1_ion / _kB_SI
# CLASS approximation-stage boundaries (smsharma/clax#48):
# start of regime "H" (x_He evolved below it): precisions.h:201-202,
# thermodynamics.c:868 (ap_z_limits) and :894 (smoothing delta);
# start of regime "frec" (x_H evolved below it): precisions.h:204-205,
# thermodynamics.c:870 and :895.
_Z_EARLY_H_RECOMB = 2870.0
_DZ_EARLY_H_RECOMB = 50.0
_Z_FULL_H_RECOMB = 1600.0
_DZ_FULL_H_RECOMB = 50.0
```

- [ ] **Step 5: The shared sub-step environment.** In `thermodynamics_solve`'s `scan_step`, directly after the line `TR_new = T_cmb / new_a`, which precedes the hydrogen RK4 block, insert:

```python
        # Sub-step environment for the recombination ROS2 steps (#48): at
        # fraction s of this grid step, ln a is linear and ln H, T_b are linear
        # in ln a. Returns the rate arguments (n_H, H, z, T_b, T_r) in CGS.
        lna0 = jnp.log(a)
        dlna_main = new_loga - lna0
        lnH0 = jnp.log(adot / a**2)          # a' = a^2 H -> H at the step start
        lnH1 = jnp.log(new_H)
        f_He_xe = 0.25 * Y_He / (1.0 - Y_He)  # x_e bookkeeping (made CLASS's in D4)

        def sub_env(s):
            a_s = jnp.exp(lna0 + s * dlna_main)
            return (n_H_0_cgs / a_s**3,
                    jnp.exp(lnH0 + s * (lnH1 - lnH0)) * _c_over_Mpc,
                    1.0 / a_s - 1.0,
                    tb + s * (new_tb - tb),
                    T_cmb / a_s)
```

- [ ] **Step 6: Replace the helium step, the escape probability and the C_He switch.** In `scan_step`, replace exactly this text:

```python
        # HeII Peebles ODE with sub-stepped Heun (10 sub-steps for stability)
        # The helium Peebles rate can be ~200/efold, requiring dlna << 0.005.
        # Main steps have dlna~0.02, so we sub-step to avoid overshoot.
        n_He_sub = 10
        dlna_He_sub = dlna_step / n_He_sub
        def he_sub_step(i, xHe_i):
            xe_i = new_xHII + 0.25 * Y_He / (1.0 - Y_He) * (xHe_i + 2.0 * new_xHeIII)
            dxHe_1 = _recfast_dxHe_dlna(
                xe_i, xHe_i, n_H_cgs, H_cgs, z_half, tbhalf, TR_half, fHe)
            xHe_pred = jnp.clip(xHe_i + dlna_He_sub * dxHe_1, 0.0, 1.0)
            xe_pred = new_xHII + 0.25 * Y_He / (1.0 - Y_He) * (xHe_pred + 2.0 * new_xHeIII)
            dxHe_2 = _recfast_dxHe_dlna(
                xe_pred, xHe_pred, nH_new, H_new, z_new, new_tb, TR_new, fHe)
            return jnp.clip(xHe_i + 0.5 * dlna_He_sub * (dxHe_1 + dxHe_2), 0.0, 1.0)
        new_xHeII_peebles = jax.lax.fori_loop(0, n_He_sub, he_sub_step, xHeII)
```

with:

```python
        # HeII -> HeI (smsharma/clax#48): CLASS staging. Regime "He2" (z > 2870)
        # is Saha; below, the Peebles equation (wrap_recfast.c:346-349, dx_He/dz
        # -> d/dlna) advanced by the shared L-stable ROS2 step, 10 sub-steps,
        # no clip.
        new_xHeII_peebles = _ros2_advance(
            lambda xe_, x_, *env: _recfast_dxHe_dlna(xe_, x_, *env, fHe),
            lambda x_: new_xHII + f_He_xe * (x_ + 2.0 * new_xHeIII),
            carry['xHeII_ode'], sub_env, dlna_main, 10)
        # Saha HeII <-> HeI with H fully ionized, thermodynamics.c:4053-4067:
        #   rhs = 4 (C_R T_b)^{3/2} exp(-T_HeI/T_b) / n_H(z)          (:4057)
        #   x_e = 0.5 (sqrt((rhs-1)^2 + 4(1+fHe) rhs) - (rhs-1)),  x_He = (x_e-1)/fHe
        # taken in the cancellation-free root of y = fHe x_He:
        #   y^2 + (1+rhs) y - fHe rhs = 0.
        rhs_He = 4.0 * (_CR_CGS * new_tb)**1.5 * jnp.exp(
            jnp.maximum(-_CB1_He1 / new_tb, -500.0)) / nH_new
        xHeII_saha_he2 = 2.0 * rhs_He / (
            (1.0 + rhs_He) + jnp.sqrt((1.0 + rhs_He)**2 + 4.0 * fHe * rhs_He))
        # ODE state: the Saha value while z > 2870 (initial condition on entering
        # regime "H", thermodynamics.c:2059-2065), the Peebles step below.
        new_xHeII_ode = jnp.where(z_new > _Z_EARLY_H_RECOMB, xHeII_saha_he2,
                                  new_xHeII_peebles)
        # Output smoothing after the switch, thermodynamics.c:2973-2987:
        # f2((z_lim - z)/(2 Dz)), exactly 0 or 1 outside z in [2770, 2870].
        w_He = _smoothstep((_Z_EARLY_H_RECOMB - z_new) / (2.0 * _DZ_EARLY_H_RECOMB))
        xHeII_staged = w_He * new_xHeII_ode + (1.0 - w_He) * xHeII_saha_he2
```

Then replace `new_xHeII = w_saha_He * new_xHeII_saha + (1.0 - w_saha_He) * new_xHeII_peebles` with `new_xHeII = w_saha_He * new_xHeII_saha + (1.0 - w_saha_He) * xHeII_staged`.

In `_recfast_dxHe_dlna`, replace

```python
    tauHe_s_safe = jnp.maximum(tauHe_s, 1e-30)
    pHe_s = (1.0 - jnp.exp(-tauHe_s_safe)) / tauHe_s_safe
```

with

```python
    tauHe_s_safe = jnp.maximum(tauHe_s, 1e-30)
    # p_s = (1 - e^{-tau})/tau, wrap_recfast.c:260, cancellation-free (expm1),
    # series below tau = 1e-4 (truncation tau^4/120 < 1e-18); double-where
    # keeps both branches finite for AD.
    _small = tauHe_s_safe < 1e-4
    _t_big = jnp.where(_small, 1.0, tauHe_s_safe)
    pHe_s = jnp.where(_small,
                      1.0 - tauHe_s_safe / 2.0 + tauHe_s_safe**2 / 6.0 - tauHe_s_safe**3 / 24.0,
                      -jnp.expm1(-_t_big) / _t_big)
```

and replace `C_He = jnp.where((xHe < 1e-15) | (xHe > 0.999), 1.0, C_He)` with:

```python
    # CLASS has no C_He switch near x_He = 1: its x_He switches are the K_He form
    # (Heflag, wrap_recfast.c:243-244) and dx_He/dz = 0 for x_He < 1e-15
    # (wrap_recfast.c:328-329); C_He is always wrap_recfast.c:346.
    C_He = jnp.where(xHe < 1e-15, 1.0, C_He)
```

Carry: in `init` add `'xHeII_ode': xHeII0,`; in `new_carry` add `'xHeII_ode': new_xHeII_ode,`. Leave the output `keys` tuple unchanged.

- [ ] **Step 7: Run green on the worktree.** Submit `PYTEST_ARGS='tests/test_recomb_thermo.py tests/test_recomb_step.py tests/test_thermodynamics.py -m "slow or not slow"' sbatch --job-name=recomb-t3-green ...`. Expected: all pass. Compare against the probe's VB/VC values: helium x_e ≈ 6.4×10⁻⁴; spike ratio ≤ 2.55.
  - If a test fails, compare against `/lustre/work/n2minh/clax-probes/helium/VC/clax/thermodynamics.py` (the probe's helium-only variant) **before** changing anything. Any formula difference from VC is a bug in this task.
  - A bar miss on the fixed code is BLOCKED; report the numbers.

- [ ] **Step 8: Gate.** `tests/ --fast` CPU job. The pre-existing `test_pk_rosenbrock_vs_kvaerno5` failure is reported as such; no other failure is allowed.

- [ ] **Step 9: Commit** `clax/thermodynamics.py tests/recomb_test_utils.py tests/test_recomb_thermo.py`, with subject `fix(thermo): helium follows CLASS's Saha -> ODE staging with an L-stable step (#48)`. The body carries the red job ID and numbers, then green.

---

### Task 4: Hydrogen follows CLASS staging (D3) — hydrogen-band and full-range S1, Review Focus

**Files:**
- Modify: `clax/thermodynamics.py` (hydrogen block; carry)
- Modify: `tests/test_recomb_thermo.py` (append tests)

**Interfaces:**
- Consumes: `sub_env`, `dlna_main` and `f_He_xe` (Task 3, inside `scan_step`, defined before the hydrogen block); `_ros2_advance` and `_smoothstep`; `_prim_tan` and `_fd_check` (Task 3).
- Produces: the scan-carry field `xHII_ode`.

- [ ] **Step 1: Append the failing tests** to `tests/test_recomb_thermo.py`:

```python
@pytest.mark.parametrize("n", THERMO_GRIDS)
@pytest.mark.parametrize("window", [(1590.0, 1650.0), (800.0, 1550.0)])
def test_xe_hydrogen_windows_match_class(recomb_cosmology, n, window):
    name, p = recomb_cosmology
    prim, _ = _prim_tan(name, p, n)
    ref = load_ref(name)
    sel = (ref["z"] >= window[0]) & (ref["z"] <= window[1])
    rel = np.abs(xe_on_ref_grid(prim, ref["z"][sel]) / ref["x_e"][sel] - 1.0)
    i = int(np.argmax(rel))
    print(f"{name} n={n}: x_e vs CLASS z {window} max {rel[i]:.3e} at z={ref['z'][sel][i]:.1f}")
    assert rel[i] <= 1e-3, f"ERROR x_e {window} {rel[i]:.3e} > 1e-3 ({name}, n={n})"


@pytest.mark.parametrize("n", THERMO_GRIDS)
def test_visibility_peak_matches_class(recomb_cosmology, n):
    from tests.recomb_test_utils import visibility_peak
    name, p = recomb_cosmology
    prim, _ = _prim_tan(name, p, n)
    ref = load_ref(name)
    dz = abs(visibility_peak(z_of(prim["loga"]), prim["g"]) - visibility_peak(ref["z"], ref["g"]))
    bar = 0.01 if n == 20000 else 0.03          # spec S1 (per-grid bars)
    print(f"{name} n={n}: visibility-peak |dz| = {dz:.4f} (bar {bar})")
    assert dz < bar, f"ERROR visibility peak shift {dz:.4f} ({name}, n={n})"


@pytest.mark.parametrize("n", THERMO_GRIDS)
@pytest.mark.parametrize("theta", THETAS)
def test_no_derivative_spikes_full_range(recomb_cosmology, n, theta):
    name, p = recomb_cosmology
    prim, tan = _prim_tan(name, p, n, theta)
    z = z_of(prim["loga"])
    worst = max(spike_ratio(tan[q] / prim[q], z, 300.0, 6000.0) + (q,) for q in QS)
    print(f"{name} n={n} d/d{theta}: worst spike ratio z 300-6000 {worst[0]:.3g} ({worst[2]} at z={worst[1]:.1f})")
    assert worst[0] < 10.0, f"ERROR spike ratio {worst[0]:.3g} ({name}, {theta}, n={n})"


@pytest.mark.parametrize("n", THERMO_GRIDS)
def test_values_smooth_full_range(recomb_cosmology, n):
    name, p = recomb_cosmology
    prim, _ = _prim_tan(name, p, n)
    z = z_of(prim["loga"])
    sel = (z >= 1500.0) & (z <= 6000.0)
    d2 = np.abs(np.diff(prim["xe"], 2))[sel[1:-1]]
    bar = 3e-5 * (20000.0 / n) ** 2
    print(f"{name} n={n}: max |d2 x_e| over z 1500-6000 = {d2.max():.3e} (bar {bar:.2e})")
    assert d2.max() < bar, f"ERROR x_e not smooth: {d2.max():.3e} ({name}, n={n})"


ZS_H = np.array([1800., 1300., 1100., 800., 400.])


@pytest.mark.parametrize("theta", THETAS)
def test_tangents_match_converged_fd_hydrogen(recomb_cosmology, theta):
    _fd_check(recomb_cosmology, theta, ZS_H)


# ---- Review Focus (plan): bounds, T_cmb, coarser grid, jit vs eager, vmap ----

@pytest.mark.parametrize("n", THERMO_GRIDS)
def test_fractions_stay_physical_without_clips(recomb_cosmology, n):
    name, p = recomb_cosmology
    prim, _ = _prim_tan(name, p, n)
    xe = np.asarray(prim["xe"])
    y = float(p.Y_He)
    upper = 1.0 + 2.0 * y / (3.9715 * (1.0 - y))     # >= the exactly-4 value too
    print(f"{name} n={n}: x_e in [{xe.min():.3e}, {xe.max():.6f}], upper {upper:.6f}")
    assert xe.min() >= 0.0 and xe.max() <= upper + 1e-12, f"ERROR x_e out of bounds ({name}, n={n})"


@pytest.mark.parametrize("n", THERMO_GRIDS)
def test_no_spikes_for_T_cmb(recomb_cosmology, n):
    name, p = recomb_cosmology
    prim, tan = _prim_tan(name, p, n, "T_cmb")
    z = z_of(prim["loga"])
    worst = max(spike_ratio(tan[q] / prim[q], z, 300.0, 6000.0) + (q,) for q in QS)
    print(f"{name} n={n} d/dT_cmb: worst spike ratio {worst[0]:.3g} ({worst[2]} at z={worst[1]:.1f})")
    assert worst[0] < 10.0, f"ERROR T_cmb spike ratio {worst[0]:.3g} ({name}, n={n})"


@pytest.mark.parametrize("theta", THETAS)
def test_coarse_grid_1500_helium_hydrogen_spike_free(recomb_cosmology, theta):
    name, p = recomb_cosmology
    prim, tan = _prim_tan(name, p, 1500, theta)
    z = z_of(prim["loga"])
    worst = max(spike_ratio(tan[q] / prim[q], z, 800.0, 6000.0) + (q,) for q in QS)
    print(f"{name} n=1500 d/d{theta}: worst spike ratio z 800-6000 {worst[0]:.3g}")
    assert worst[0] < 10.0, f"ERROR 1500-grid spike ratio {worst[0]:.3g} ({name}, {theta})"


def test_jit_and_eager_primal_agree(recomb_cosmology):
    name, p = recomb_cosmology
    prec = thermo_prec(3000)
    a = np.asarray(tables(as_traced(p), prec)["xe"])
    b = np.asarray(jax.jit(lambda q: tables(q, prec))(as_traced(p))["xe"])
    d = float(np.max(np.abs(a - b)))
    print(f"{name}: max |x_e(jit) - x_e(eager)| = {d:.3e}")
    assert d < 1e-12, f"ERROR jit vs eager x_e differ by {d:.3e} ({name})"


def test_vmap_over_h_matches_loop(recomb_cosmology):
    name, p = recomb_cosmology
    prec = thermo_prec(3000)
    hs = float(p.h) * jnp.array([0.98, 1.0, 1.02])
    f = jax.jit(lambda h: tables(replace(as_traced(p), h=h), prec)["xe"])
    batched = np.asarray(jax.vmap(f)(hs))
    looped = np.stack([np.asarray(f(h)) for h in hs])
    d = float(np.max(np.abs(batched - looped)))
    print(f"{name}: max |vmap - loop| = {d:.3e}")
    assert d < 1e-12, f"ERROR vmap vs loop differ by {d:.3e} ({name})"
```

If the vmap test fails, first compare `bg.H_of_loga.y` from vmap and from the loop. That tells whether the difference starts in the background solve or in recombination; report it, without loosening the bar.

`tangent_tables` accepts `theta="T_cmb"` (a traced `CosmoParams` field) and `n = 1500` unchanged.

- [ ] **Step 2: Show RED.**
  - Make a red snapshot `red-t4` from `4d084b5`; it has neither fix. Submit `-k "hydrogen or visibility or full_range or physical or T_cmb or coarse or jit_and_eager or vmap"`.
  - Expected: the hydrogen x_e window FAILS on the 3000-point grid (0.97); the full-range spike test FAILS; the visibility peak FAILS on 3000 points (0.256); jit vs eager FAILS (probe: 2.7×10⁻⁵).
  - Also run the same selection on the **Task 3 commit**, the worktree HEAD before Step 3. The hydrogen-band spike tests must still fail there (probe B/C: up to 6.9×10³), proving the hydrogen tests see the hydrogen bug. Report both.

- [ ] **Step 3: Implement.** In `scan_step`, replace exactly the RK4 block:

```python
        k1 = _recfast_dxHII_dlna(xe, xHII, n_H_cgs, H_cgs, z_half, tbhalf, TR_half)
        xHII_2 = jnp.clip(xHII + 0.5 * dlna_step * k1, 0.0, 1.0)
        xe_2 = xHII_2 + 0.25 * Y_He / (1.0 - Y_He) * (xHeII + 2.0 * xHeIII)
        k2 = _recfast_dxHII_dlna(xe_2, xHII_2, n_H_cgs, H_cgs, z_half, tbhalf, TR_half)
        xHII_3 = jnp.clip(xHII + 0.5 * dlna_step * k2, 0.0, 1.0)
        xe_3 = xHII_3 + 0.25 * Y_He / (1.0 - Y_He) * (xHeII + 2.0 * xHeIII)
        k3 = _recfast_dxHII_dlna(xe_3, xHII_3, n_H_cgs, H_cgs, z_half, tbhalf, TR_half)
        xHII_4 = jnp.clip(xHII + dlna_step * k3, 0.0, 1.0)
        xe_4 = xHII_4 + 0.25 * Y_He / (1.0 - Y_He) * (xHeII + 2.0 * xHeIII)
        k4 = _recfast_dxHII_dlna(xe_4, xHII_4, nH_new, H_new, z_new, new_tb, TR_new)
        new_xHII_recfast = jnp.clip(
            xHII + dlna_step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 0.0, 1.0)
```

with:

```python
        # Hydrogen (smsharma/clax#48): the Peebles equation (wrap_recfast.c:110-174,
        # dx_H/dz -> d/dlna) advanced from the ODE state by the shared L-stable
        # ROS2 step, 10 sub-steps, no clip. It replaces an explicit RK4 whose
        # h*J = -3.2 at z = 1600 lies outside RK4's stability interval [-2.79, 0].
        new_xHII_peebles = _ros2_advance(
            _recfast_dxHII_dlna,
            lambda x_: x_ + f_He_xe * (xHeII + 2.0 * xHeIII),
            carry['xHII_ode'], sub_env, dlna_main, 10)
```

Then replace:

```python
        w_saha = jax.nn.sigmoid(0.1 * (z_half - 1600.0))
        new_xHII = w_saha * xHII_saha + (1.0 - w_saha) * new_xHII_recfast
```

with:

```python
        # CLASS staging for hydrogen: regime "H" (z > 1600) is Saha with the x_HeII
        # correction (thermodynamics.c:4070-4087, the quadratic above). The ODE
        # state is the Saha value while z > 1600 (initial condition of regime
        # "frec", thermodynamics.c:2093), the Peebles step below; the output is
        # smoothed by f2 over z in [1500, 1600] (thermodynamics.c:2973-2987).
        new_xHII_ode = jnp.where(z_new > _Z_FULL_H_RECOMB, xHII_saha, new_xHII_peebles)
        w_H = _smoothstep((_Z_FULL_H_RECOMB - z_new) / (2.0 * _DZ_FULL_H_RECOMB))
        new_xHII = w_H * new_xHII_ode + (1.0 - w_H) * xHII_saha
```

Carry: in `init` add `'xHII_ode': xHII0,`; in `new_carry` add `'xHII_ode': new_xHII_ode,`. Then run `grep -n "dlna_step\|n_H_cgs\|z_half\|tbhalf\|TR_half" clax/thermodynamics.py` and delete only locals that nothing reads any more, keeping those still used, e.g. by the hydrogen Saha or the T_b update.

- [ ] **Step 4: Green.** Run `tests/test_recomb_thermo.py tests/test_recomb_step.py tests/test_thermodynamics.py -m "slow or not slow"` as CPU jobs, sharded by `-k` on cosmology names into 2–3 jobs if one would exceed ~8 h. All must pass. Compare with the probe's VE values: hydrogen bands ≤ 2.8×10⁻⁴ and 1.2×10⁻⁴ (4.4×10⁻⁴ on 3000 points); spike ≤ 1.19 / 2.55; peak 0.001 / 0.013.
  - If a value differs from VE by more than 2×, diff against `/lustre/work/n2minh/clax-probes/helium/VE/clax/thermodynamics.py`, ignoring its `[HE-SPIKE debug]` lines, before changing anything.
  - The 3000-point smoothness bar (1.33×10⁻³) was not measured by the probe. If the fix misses it, STOP and report.

- [ ] **Step 5: Gate.** Run `tests/ --fast` (CPU job) and full mode of `tests/test_thermodynamics.py`.

- [ ] **Step 6: Commit** `clax/thermodynamics.py tests/test_recomb_thermo.py`, with subject `fix(thermo): hydrogen follows CLASS's Saha -> ODE staging; the fit_cl x_e collapse at z~1559 is gone (#48)`. The body carries red on `4d084b5` and on the Task 3 commit, then green.

---

### Task 5: Downstream seams — S2 (δ_m functional) and S3 (production `custom_vjp`)

**Files:**
- Modify: `tests/recomb_test_utils.py` (append the functionals)
- Create: `tests/test_recomb_downstream.py`

**Interfaces:**
- Consumes: `PREC_RB` and `as_traced`; `recomb_downstream_cosmology` (Task 1).
- Produces: `dm_functional(prec, base) -> Callable[[h], scalar]` and `dm_functional_production(prec, base) -> Callable[[h], scalar]`.

- [ ] **Step 1: Append the functionals** to `tests/recomb_test_utils.py`:

```python
def dm_functional(prec, base):
    """f(h) = sum_k delta_m(k, tau0)^2 through the matter solve (spec D8)."""
    from clax.perturbations import perturbations_solve_mpk

    def f(h_val):
        p = replace(as_traced(base), h=h_val)
        bg = background_solve(p, prec)
        th = thermodynamics_solve(p, prec, bg)
        pt = perturbations_solve_mpk(p, prec, bg, th)
        return jnp.sum(pt.delta_m[:, -1] ** 2)
    return f


def dm_functional_production(prec, base):
    """Same functional through solve_background_and_thermo (ADR 0001's stable
    custom_vjp), the path clax.compute and NUTS use. Reverse mode only."""
    from clax.perturbations import perturbations_solve_mpk
    from clax.thermodynamics import solve_background_and_thermo

    def f(h_val):
        p = replace(as_traced(base), h=h_val)
        bg, th = solve_background_and_thermo(p, replace(prec, th_grad_mode="stable"))
        pt = perturbations_solve_mpk(p, prec, bg, th)
        return jnp.sum(pt.delta_m[:, -1] ** 2)
    return f
```

- [ ] **Step 2: Write the tests** `tests/test_recomb_downstream.py`:

```python
"""S2 / S3 (smsharma/clax#48): d/dh of f = sum_k delta_m(k, tau0)^2 through the
matter solve is independent of solver step placement, agrees between forward
and reverse mode (native and production custom_vjp), and is smooth along h.
Multi-cosmology: recomb_downstream_cosmology (3 points)."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.recomb_test_utils import PREC_RB, dm_functional, dm_functional_production

pytestmark = pytest.mark.slow
FWD = replace(PREC_RB, ode_adjoint="direct")


def _fwd(base, prec, jit):
    f = dm_functional(prec, base)
    g = lambda h: jax.jvp(f, (h,), (jnp.asarray(1.0),))[1]
    return float((jax.jit(g) if jit else g)(jnp.asarray(float(base.h))))


def test_dfdh_independent_of_step_placement(recomb_downstream_cosmology):
    name, base = recomb_downstream_cosmology
    vals = {(c, j): _fwd(base, replace(FWD, pt_k_chunk_size=c), j)
            for c in (8, 62) for j in (True, False)}
    v = np.array(list(vals.values()))
    spread = (v.max() - v.min()) / abs(np.median(v))
    print(f"{name}: df/dh over chunk x jit {vals} spread {spread:.3e}")
    assert spread < 3e-4, f"ERROR step-placement spread {spread:.3e} ({name})"


def test_native_reverse_matches_forward(recomb_downstream_cosmology):
    name, base = recomb_downstream_cosmology
    fwd = _fwd(base, FWD, True)
    rev = float(jax.grad(dm_functional(PREC_RB, base))(jnp.asarray(float(base.h))))
    rel = abs(rev / fwd - 1.0)
    print(f"{name}: forward {fwd:.8e} native reverse {rev:.8e} rel {rel:.3e}")
    assert rel < 3e-4, f"ERROR native reverse vs forward {rel:.3e} ({name})"


def test_production_reverse_matches_forward(recomb_downstream_cosmology):
    name, base = recomb_downstream_cosmology
    fwd = _fwd(base, FWD, True)
    rev = float(jax.grad(dm_functional_production(PREC_RB, base))(jnp.asarray(float(base.h))))
    rel = abs(rev / fwd - 1.0)
    print(f"{name}: forward {fwd:.8e} production reverse {rev:.8e} rel {rel:.3e}")
    assert rel < 3e-4, f"ERROR production reverse vs forward {rel:.3e} ({name})"


def test_dfdh_smooth_along_h(recomb_downstream_cosmology):
    name, base = recomb_downstream_cosmology
    hs = np.linspace(0.62, 0.72, 11)
    f = dm_functional(FWD, base)
    g = jax.jit(lambda h: jax.jvp(f, (h,), (jnp.asarray(1.0),))[1])
    d = np.array([float(g(jnp.asarray(h))) for h in hs])
    resid = d - np.polyval(np.polyfit(hs, d, 3), hs)
    rough = float(np.max(np.abs(resid)) / abs(np.median(d)))
    print(f"{name}: df/dh along h {d} cubic-fit residual {rough:.3e}")
    assert rough < 3e-4, f"ERROR df/dh not smooth along h: {rough:.3e} ({name})"
```

- [ ] **Step 3: Show RED.** Make snapshot `red-t5` (from `4d084b5` plus the new tests). Submit `PYTEST_ARGS='tests/test_recomb_downstream.py'` as **three jobs**, one per cosmology (`-k lcdm_fiducial`, `-k omega_b_high`, `-k YHe_028`).
  - Expected at fiducial: step-placement spread ≈ 0.17, native reverse ≈ 6×10⁻², production reverse up to ~1.1 on CPU, and h-scan roughness ≈ 6×10⁻². The probe found `omega_b_high` and `YHe_028` can pass by luck; red on fiducial is enough.

- [ ] **Step 4: Green.** Same three jobs on the worktree. All pass. Probe VE values: spread 6.3×10⁻⁵, reverse ≤ 7.7×10⁻⁵, h-scan 1.0×10⁻⁴.
  - S3 (production path) was **not** measured for VE by the probe. If it misses 3×10⁻⁴, stop and report it as a design question; do not widen the bar.

- [ ] **Step 5: Commit** `tests/recomb_test_utils.py tests/test_recomb_downstream.py`, with subject `test(thermo): delta_m gradients are step-placement-invariant, mode-consistent and smooth (#48)`.

---

### Task 6: C_ℓ seam — S4 at the `fit_cl` preset (bars provisional until measured)

**Files:**
- Modify: `tests/recomb_test_utils.py` (append)
- Create: `tests/test_recomb_cl.py`
- Create (not committed): `/lustre/work/n2minh/clax-probes/recomb/s4_measure.py`

**Interfaces:**
- Consumes: `as_traced`.
- Produces: `CL_LMAX`, `cl_index(prec, base) -> np.ndarray` (the static positions of 30 ≤ ℓ ≤ 1200), and `cl_functional(prec, base, theta, idx) -> Callable[[x], scalar]`.

- [ ] **Step 1: Append** to `tests/recomb_test_utils.py`:

```python
CL_LMAX = 1200


def cl_index(prec, base):
    """Static positions of 30 <= ell <= 1200 in compute_cls_all_fast's output."""
    from clax.harmonic import compute_cls_all_fast
    from clax.perturbations import perturbations_solve
    from clax.thermodynamics import solve_background_and_thermo
    p = as_traced(base)
    pr = replace(prec, th_grad_mode="native")
    bg, th = solve_background_and_thermo(p, pr)
    cls = compute_cls_all_fast(perturbations_solve(p, pr, bg, th), p, bg, l_max=CL_LMAX)
    ell = np.asarray(cls["ell"])
    return np.nonzero((ell >= 30) & (ell <= CL_LMAX))[0]


def cl_functional(prec, base, theta, idx):
    """F(theta) = sum_{ell=30}^{1200} ln C_ell^TT at prec (spec D8, S4)."""
    from clax.harmonic import compute_cls_all_fast
    from clax.perturbations import perturbations_solve
    from clax.thermodynamics import solve_background_and_thermo

    def F(x):
        p = replace(as_traced(base), **{theta: x})
        bg, th = solve_background_and_thermo(p, prec)
        cls = compute_cls_all_fast(perturbations_solve(p, prec, bg, th), p, bg, l_max=CL_LMAX)
        return jnp.sum(jnp.log(jnp.take(cls["tt"], jnp.asarray(idx))))
    return F
```

- [ ] **Step 2: Measure first** (GPU job). Write `s4_measure.py`, which for θ ∈ {h, omega_b} at `lcdm_fiducial` prints:
  - forward dF/dθ at `pt_k_chunk_size` ∈ {8, 64} × jit/no-jit, under `replace(PrecisionParams.fit_cl(), th_grad_mode="native", ode_adjoint="direct")`;
  - the production reverse `jax.grad` under `PrecisionParams.fit_cl()`;
  - forward dF/dh at 7 points h ∈ [0.64, 0.70].

  It imports `cl_functional` and `cl_index` from `tests.recomb_test_utils`. Run it on the red snapshot (`4d084b5`) and on the worktree:

  `SRC=<dir> SCRIPT=/lustre/work/n2minh/clax-probes/recomb/s4_measure.py JAX_PLATFORMS=cuda sbatch --gres=gpu:1 --time=6:00:00 --job-name=recomb-t6-measure --export=ALL /lustre/work/n2minh/clax-probes/recomb/run.sbatch`

  **Decision rule:**
  - If the worktree meets 3×10⁻⁴ on all three quantities (step-placement spread, production reverse vs forward, cubic-fit residual along h), keep the provisional bars.
  - If the worktree misses any of them, STOP and report both tables to the controller (a design question). Do not write the test with a looser bar.

- [ ] **Step 3: Write the tests** `tests/test_recomb_cl.py`:

```python
"""S4 (smsharma/clax#48): C_ell-level gradients at the fit_cl preset.
F = sum_{ell=30}^{1200} ln C_ell^TT. Bars 3e-4, provisional in the spec and
confirmed by measurement (plan Task 6 Step 2). GPU-preferred.
Multi-cosmology: recomb_downstream_cosmology (3 points)."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from clax import PrecisionParams
from tests.recomb_test_utils import cl_functional, cl_index

pytestmark = pytest.mark.slow
PROD = PrecisionParams.fit_cl()
FWD = replace(PROD, th_grad_mode="native", ode_adjoint="direct")


def _fwd(base, prec, theta, idx, jit=True):
    F = cl_functional(prec, base, theta, idx)
    g = lambda x: jax.jvp(F, (x,), (jnp.asarray(1.0),))[1]
    return float((jax.jit(g) if jit else g)(jnp.asarray(float(getattr(base, theta)))))


@pytest.mark.parametrize("theta", ["h", "omega_b"])
def test_cl_gradient_independent_of_step_placement(recomb_downstream_cosmology, theta):
    name, base = recomb_downstream_cosmology
    idx = cl_index(FWD, base)
    v = np.array([_fwd(base, replace(FWD, pt_k_chunk_size=c), theta, idx, j)
                  for c in (8, 64) for j in (True, False)])
    spread = (v.max() - v.min()) / abs(np.median(v))
    print(f"{name} dF/d{theta}: {v} spread {spread:.3e}")
    assert spread < 3e-4, f"ERROR C_l step-placement spread {spread:.3e} ({name}, {theta})"


@pytest.mark.parametrize("theta", ["h", "omega_b"])
def test_cl_production_reverse_matches_forward(recomb_downstream_cosmology, theta):
    name, base = recomb_downstream_cosmology
    idx = cl_index(FWD, base)
    fwd = _fwd(base, FWD, theta, idx)
    rev = float(jax.grad(cl_functional(PROD, base, theta, idx))(jnp.asarray(float(getattr(base, theta)))))
    rel = abs(rev / fwd - 1.0)
    print(f"{name} dF/d{theta}: forward {fwd:.8e} production reverse {rev:.8e} rel {rel:.3e}")
    assert rel < 3e-4, f"ERROR C_l production reverse vs forward {rel:.3e} ({name}, {theta})"


def test_cl_gradient_smooth_along_h(recomb_downstream_cosmology):
    name, base = recomb_downstream_cosmology
    idx = cl_index(FWD, base)
    hs = np.linspace(0.64, 0.70, 7)
    d = np.array([_fwd(replace(base, h=float(h)), FWD, "h", idx) for h in hs])
    resid = d - np.polyval(np.polyfit(hs, d, 3), hs)
    rough = float(np.max(np.abs(resid)) / abs(np.median(d)))
    print(f"{name}: dF/dh along h {d} cubic-fit residual {rough:.3e}")
    assert rough < 3e-4, f"ERROR C_l gradient not smooth along h: {rough:.3e} ({name})"
```

- [ ] **Step 4: Red and green.** Run `tests/test_recomb_cl.py` on the red snapshot and on the worktree, each as GPU jobs (`JAX_PLATFORMS=cuda`, `--gres=gpu:1`), one per cosmology. Red must fail on at least one cosmology and θ. Green must pass everywhere.

- [ ] **Step 5: Commit** `tests/recomb_test_utils.py tests/test_recomb_cl.py`, with subject `test(thermo): C_l-level gradients at fit_cl are step-placement-invariant and smooth (#48)`. The body carries the Step 2 tables.

---

### Task 7: GPU-only check — CPU vs GPU tangents (S1)

**Files:**
- Create: `tests/test_recomb_gpu.py`. It must **not** set `JAX_PLATFORMS` or `jax_platform_name`.

**Interfaces:**
- Consumes: `tangent_tables`, `thermo_prec` and `QS` (Task 3); `recomb_downstream_cosmology` (Task 1).

- [ ] **Step 1: Write the test.**

```python
"""S1 GPU-only (smsharma/clax#48): thermodynamics tangents agree between CPU and
GPU to < 1e-10. Must run in its OWN pytest session on a GPU node: several other
test modules pin JAX to CPU at import, which pins the whole session. Skipped
without a GPU. Multi-cosmology: recomb_downstream_cosmology (3 points)."""
import jax
import numpy as np
import pytest

from tests.recomb_test_utils import QS, tangent_tables, thermo_prec

pytestmark = pytest.mark.slow


def _has_gpu():
    try:
        return len(jax.devices("gpu")) > 0
    except RuntimeError:
        return False


@pytest.mark.skipif(not _has_gpu(), reason="needs a GPU")
@pytest.mark.parametrize("theta", ["h", "omega_b", "Y_He"])
def test_cpu_gpu_tangents_agree(recomb_downstream_cosmology, theta):
    name, p = recomb_downstream_cosmology
    prec = thermo_prec(20000)
    with jax.default_device(jax.devices("cpu")[0]):
        _, t_cpu = tangent_tables(p, prec, theta)
    with jax.default_device(jax.devices("gpu")[0]):
        _, t_gpu = tangent_tables(p, prec, theta)
    worst = 0.0
    for q in QS:
        a, b = np.asarray(t_cpu[q]), np.asarray(t_gpu[q])
        worst = max(worst, float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-300))))
    print(f"{name} d/d{theta}: max CPU-GPU tangent rel diff {worst:.3e}")
    assert worst < 1e-10, f"ERROR CPU vs GPU tangents {worst:.3e} ({name}, {theta})"
```

- [ ] **Step 2: Red and green on GPU.**
  - Run `PYTEST_ARGS='tests/test_recomb_gpu.py'` alone, with `JAX_PLATFORMS=cuda`, `--gres=gpu:1`, on the red snapshot and on the worktree.
  - Expected red: a factor of ~21 (probe V0). Expected green: ≲ 3×10⁻¹³ (probe VE).
  - The log must show `JAX_PLATFORMS=cuda` and no skip.

- [ ] **Step 3: Commit** `tests/test_recomb_gpu.py`, with subject `test(thermo): CPU and GPU thermodynamics tangents agree to round-off (#48)`.

---

### Task 8: Helium-to-hydrogen number ratio uses CLASS's `_NOT4` (D4)

**Files:**
- Modify: `clax/thermodynamics.py`
- Modify: `tests/test_recomb_thermo.py` (append one test)

**Interfaces:**
- Produces: `_f_He(Y_He) -> Y_He / (_NOT4 (1 − Y_He))` in `clax.thermodynamics`.

- [ ] **Step 1: Write the failing test.** Append to `tests/test_recomb_thermo.py`:

```python
def test_fully_ionised_xe_uses_class_helium_mass_ratio(recomb_cosmology):
    """x_e at z = 2e4 (H, He fully ionised) = 1 + 2 f_He with f_He = Y/(3.9715 (1-Y)),
    CLASS thermodynamics.h:707 (_not4_). Fails while any site uses exactly 4."""
    from clax.background import background_solve
    from clax.thermodynamics import thermodynamics_solve
    name, p = recomb_cosmology
    prec = thermo_prec(20000)
    q = as_traced(p)
    th = thermodynamics_solve(q, prec, background_solve(q, prec))
    xe = float(th.xe_of_loga.evaluate(jnp.log(1.0 / 2.0001e4)))
    y = float(p.Y_He)
    expect = 1.0 + 2.0 * y / (3.9715 * (1.0 - y))
    print(f"{name}: x_e(z=2e4) = {xe:.8f}, expected {expect:.8f}")
    assert abs(xe - expect) < 1e-6, f"ERROR fully ionised x_e {xe:.8f} vs {expect:.8f} ({name})"
```

- [ ] **Step 2: RED.** Run it on the worktree **before** Step 3 (CPU job, `-k fully_ionised`). Expected: FAIL. The offset is 2 f_He (4/3.9715 − 1) ≈ 1.2×10⁻³ at fiducial.

- [ ] **Step 3: Implement.** Add after `_NOT4 = 3.9715  # He-4/H mass ratio, cf. CLASS thermodynamics.h:707`:

```python
def _f_He(Y_He):
    """Helium-to-hydrogen number ratio n_He/n_H = Y/(_not4_ (1-Y)), CLASS
    thermodynamics.h:707. The one definition used by every x_e sum and helium
    rate in this module (smsharma/clax#48)."""
    return Y_He / (_NOT4 * (1.0 - Y_He))
```

Run `grep -nE "0\.25 \* Y_He / \(1\.0 - Y_He\)|Y_He / \(4\.0 \* \(1\.0 - Y_He\)\)|Y_He / \(_NOT4 \* \(1\.0 - Y_He\)\)" clax/thermodynamics.py`. Replace every match:
- `0.25 * Y_He / (1.0 - Y_He)` → `_f_He(Y_He)`;
- `Y_He / (4.0 * (1.0 - Y_He))` → `_f_He(Y_He)`;
- `Y_He / (_NOT4 * (1.0 - Y_He))` → `_f_He(Y_He)`.

This includes the `_ionHe` constant `c`, `xe0`, `xHeII_contrib`, `fHe`, `f_He_xe` and every `new_xe` sum. `_f_He` is defined at module level after `_NOT4`; `_ionHe` and the reionisation helper sit above or below it, so verify by import that no name is used before definition at call time. Do not touch expressions where Y_He is a **mass** fraction (e.g. n_H from ρ_b), and list any you leave in the task report. Re-run the grep: expected zero matches outside `_f_He`.

- [ ] **Step 4: Green.** Run all of `tests/test_recomb_thermo.py` (CPU jobs, sharded as in Task 4). Every bar must still pass. The probe's VE still used exactly 4, so this step goes beyond what the probe measured.
  - Expected, as an estimate rather than a gate: the helium-window x_e vs CLASS (6.4×10⁻⁴ for VE) shrinks. The exactly-4 offset there is about f_He x_HeII (4/3.9715 − 1)/x_e ≈ 5×10⁻⁴.
  - Record before and after. If any bar that passed in Task 4 now fails, STOP and report.

- [ ] **Step 5: Gate.** Run `tests/ --fast` and full mode of `tests/test_thermodynamics.py`.

- [ ] **Step 6: Commit** `clax/thermodynamics.py tests/test_recomb_thermo.py`, with subject `fix(thermo): one CLASS helium-to-hydrogen ratio (_not4_) everywhere (#48)`. The body carries the before/after x_e numbers.

---

### Task 9: Regression gate, benchmark, C_ℓ vs CLASS before and after

**Files:**
- Create (not committed): `/lustre/work/n2minh/clax-probes/recomb/bench.py`, `/lustre/work/n2minh/clax-probes/recomb/cl_vs_class.py`

- [ ] **Step 1: Full regression gate.** Run three CPU jobs on the worktree, sharded:
  - (a) `tests/ --fast`;
  - (b) full mode (`-m "slow or not slow"`) of `tests/test_thermodynamics.py tests/test_recomb_reference.py tests/test_recomb_step.py tests/test_recomb_thermo.py tests/test_recomb_downstream.py`;
  - (c) full mode of the C_ℓ and P(k) value suites: list them with `ls tests/test_harmonic*.py tests/test_lensing*.py tests/test_pk*.py tests/test_ept_gradients.py tests/test_multicosmo_consistency.py` and pass that list.

  Every failure is compared against the same test on the red snapshot. A test failing only on the worktree is a regression: STOP and report it. A failure identical on `4d084b5` is pre-existing; list it.

- [ ] **Step 2: Benchmark (D10).**
  - `bench.py` times, on a V100 (`JAX_PLATFORMS=cuda`):
    - `thermodynamics_solve` at `PrecisionParams.fit_cl()`: jitted, the first call excluded, median of 20;
    - the full `fit_cl` pipeline (`solve_background_and_thermo` → `perturbations_solve` → `compute_cls_all_fast(l_max=1500)`): jitted, median of 5.
  - Run it on the red snapshot and on the worktree, in **one GPU job each, on the same node** (`--nodelist=<same node>`).
  - Report both medians and the ratio. If total `fit_cl` grows by more than 5%, STOP and report.

- [ ] **Step 3: C_ℓ vs CLASS before and after.**
  - `cl_vs_class.py` computes C_ℓ TT/EE/TE at `fit_cl` and `planck_cl` for `lcdm_fiducial`, and prints max |ΔC_ℓ/C_ℓ| in ℓ ∈ [2, 500], [500, 1500] and [1500, 2500] (planck only) against `reference_data/lcdm_fiducial/cls_recfast.npz` and `cls.npz`.
  - Run it on the red snapshot and on the worktree (GPU jobs).
  - Report the table. These are numbers for the PR, not gates. Existing tests already gate C_ℓ in Step 1.

- [ ] **Step 4: No commit;** results go into the task report and into Task 10's CHANGELOG.

---

### Task 10: ADRs, docstrings, CHANGELOG

**Files:**
- Create: `docs/adr/0003-recombination-staging.md`
- Modify: `docs/adr/0001-thermo-reverse-mode-vjp-through-jvp.md` (append a dated note)
- Modify: `clax/thermodynamics.py` (module docstring: recombination method paragraph)
- Modify: `CHANGELOG.md`
- Modify: any existing test docstring that quotes a number moved by Tasks 3–8. Find candidates with `grep -rnE "x_e|xe\(" tests/*.py | grep -E "[0-9]\.[0-9]+e-|%"` and compare against the Task 9 gate logs.

- [ ] **Step 1: ADR 0003.** Same structure as ADR 0001: Context, Decision, Consequences, Status.
  - Context: #48, with the probe table (V0, A, B, C, VE) and its numbers from the spec.
  - Decision: CLASS staging plus ROS2 plus `_NOT4`.
  - Consequences: speed (the Task 9 numbers) and C_ℓ shifts (Task 9).
  - Status: Accepted. Date: today in `YYYY-MM-DD`, as ADR 0001's header does.

- [ ] **Step 2: ADR 0001 note.** Append:

```markdown
## Note (Oct 10, 2026, smsharma/clax#48)

The premise "forward mode is exact" holds only when ODE solver stages miss the
recombination chatter that clipped explicit helium/hydrogen steps produced; with
them, both modes carried step-placement-dependent spikes (#48). ADR 0003 removes
the chatter; this rule's re-association remains valid and is unchanged.
```

- [ ] **Step 3: Module docstring.** Add one paragraph to `clax/thermodynamics.py`'s module docstring:
  - recombination follows CLASS's approximation stages (Saha for He above z = 2870 and for H above z = 1600; `f2` smoothing), with one L-stable ROS2 step and no clips;
  - why: #48;
  - where: ADR 0003.

- [ ] **Step 4: CHANGELOG.** Add an entry above the first `### ` heading, headed `### <Mon D, YYYY>: Recombination follows CLASS's Saha -> ODE staging; thermodynamics gradients are smooth (#48)`, with:
  - **Bug:** what was wrong, using the numbers from the spec's Problem Statement.
  - **Fix:** D1–D4.
  - **Measured (before -> after):** every S0–S4 number from the task reports; Task 9's benchmark and C_ℓ table.
  - **Not done:** the #45 probe; the items in the spec's Out of Scope.

- [ ] **Step 5: Gate.** Run `tests/ --fast`, since docstrings changed.

- [ ] **Step 6: Commit** the explicitly listed files, with subject `docs: ADR 0003, ADR 0001 note, CHANGELOG for recombination staging (#48)`.

---

## After the plan

- Hand back to the user with the branch state and every measured number. Do **not** open a PR without approval.
- The #45 follow-on (spec, Further Notes) is a separate plan.
