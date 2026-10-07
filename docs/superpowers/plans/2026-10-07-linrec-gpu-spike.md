# Linear-Recurrence (Parallel-in-Time) Perturbation Solver — GPU Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, validate and benchmark on Bridges-2 GPUs a parallel-in-time
integrator for clax's scalar perturbation system that forms every Rodas5
step as a matrix and composes them with `jax.lax.associative_scan`, and
decide from measurements whether it beats the current sequential adaptive
loop at equal accuracy.

**Architecture:** A new experimental module `clax/linrec.py` reuses the
existing RHS (`_perturbation_rhs`) and Rodas5 tableau (`clax/rosenbrock.py`)
unchanged. It (1) assembles `A(tau,k)` by `jacfwd` of the linear RHS,
(2) forms the Rodas5 step map `(M_n, E_n)` per `(step, k)` in parallel,
(3) composes prefix products with `associative_scan` inside a blocked
`lax.scan` (memory bound), (4) replays recorded adaptive meshes (gradsolve
record-and-replay) so results are checkable against the sequential solver to
rounding, and (5) exposes an a-posteriori error monitor and a drop-in
`PerturbationResult` builder so C_l can be compared end to end.

**Tech Stack:** JAX 0.6.2 (`jax.lax.associative_scan`, `jax.jacfwd`,
`jax.scipy.linalg.lu_factor/lu_solve`), diffrax 0.7.0 (`Rodas5` reference
step, `SaveAt(steps=True)` mesh recording, `ForwardMode`), numpy, pytest.
GPUs: Bridges-2 V100-32GB (primary), H100 (if SU allow). Float64 only.

**Spec:** `docs/stiff_ode_speed_plan_2026-10-07.md` (sections 2.4, 4.0, 5
Phase 0 items 5-6). Branch: `exp/linrec-gpu-spike` (created from
`origin/main` @ `e3f04ac`).

## Global Constraints

- `jax.config.update("jax_enable_x64", True)` everywhere; fp64 GPUs only
  (V100/A100/H100). Do not benchmark on L40S (fp64 rate 1/64).
- `clax/perturbations.py` and `clax/rosenbrock.py` are **read-only** for
  this spike: the module imports their private helpers; it never edits them.
- No new dependencies. No `gradsolve`, no Pallas, no float32.
- Test output: <=10 lines on success (`pytest -q`); verbose diagnostics go
  to `test_logs/`. All tests in `tests/test_linrec.py` must pass with
  `pytest tests/test_linrec.py -q --fast` on CPU in <=5 min; `slow`-marked
  tests run only in full mode (GPU job).
- Multi-cosmology RULE (CLAUDE.md): the physics-facing tests (Task 8 JVP,
  Task 9 C_l) use the `lcdm_cosmology` fixture from `tests/conftest.py`
  (5 points; `--fast` prunes to fiducial). Numerics-only tests state their
  exemption in the docstring.
- CHANGELOG date headings use `Mon D, YYYY` (e.g. `### Oct 7, 2026: ...`).
- Commit after every task; never commit with failing tests.
- Environment note: on the macOS-27 laptop every conda env's `scipy` fails
  to dlopen `_propack`, which crashes jaxlib's LAPACK init (`lu_factor`).
  Run `pip install -U scipy` in the env before local CPU runs. Bridges-2
  is unaffected.

## Review Focus

1. **Padded (zero-length) steps** must be exact identities and must not
   inject NaN into forward or JVP values (`1/dt`). Pinned by Task 3
   `test_zero_step_is_identity_and_jvp_finite`.
2. **Merged meshes** (recorded nodes ∪ `tau_grid`) must contain every
   `tau_grid` value bit-exactly so `searchsorted` gathers the right state.
   Pinned by Task 4 `test_merged_mesh_contains_tau_grid_exactly`.
3. **`N % block_size != 0`** must raise, never silently drop steps.
   Pinned by Task 6 `test_bad_block_size_raises`.
4. **A too-coarse mesh must be detectable**: the error monitor must exceed
   1 when the recorded mesh is coarsened 2x. Pinned by Task 7
   `test_error_ratio_flags_coarse_mesh`.
5. **GPU memory at production size** (nk=300, d=61, N~1000) must fit in
   32 GB for at least one block size. Pinned by Task 11 gate G4 (benchmark
   reports `peak_bytes_in_use` per block size).

---

## Part A — Problem, boundary, solution map

### A.1 Problem (measured, 2026-10-07)

| Quantity | CLASS 3.3.4 (M4 Max) | clax today |
|---|---|---|
| Lensed TT/TE/EE l<=2500, massless nu | 1.54 s (1 thr) / 0.25 s (8 thr); **12.4 s with TCA/RSA/UFA off** | `planck_cl` 487 s H100; `fit_cl` 34 s V100, **5.1 s on the same Mac's CPU** |
| Per-step cost of the ODE solver | n/a | 0.08 ms (Rodas5) / 0.16 ms (Kvaerno5) per mode-step on CPU; ~100 ms per lockstep iteration on GPU (two-point fit, both state sizes) |
| Lockstep length (max over k) | ~10^3 (ndf15 with approximations) | 255 (`fit_cl`, rtol 1e-3), 1366 (rtol 1e-6, l_max 17), 7106 (rtol 1e-6, RSA damping off) |

Cost model `T_pert = N_chunks x max_k(steps) x t_iteration`. The GPU
iteration cost is independent of state size (61 vs ~250): it is launch
latency and host syncs from diffrax's adaptive while loop, inner
Newton/save loops, `lax.map` over chunks and the checkpointed adjoint.
The CPU runs the same arithmetic ~600x closer to its floor.

### A.2 Boundary

- Differentiability is **not** the constraint: forward-mode AD passes
  through adaptive `lax.while_loop` (ABCMB `jacfwd`; clax main already
  supports `jax.jvp` through the pipeline); approximation switches are
  measure-zero jumps already sigmoid-blended in clax.
- The constraint is **state-dependent control flow** (accept/reject,
  Newton, save loops) which serializes the GPU, plus the **number of
  sequential steps** set by approximations/tolerance.
- The scalar RHS is **linear and homogeneous** in `y`: every gate
  (`is_tca`, `is_rsa`, ncdm blend) depends only on `(tau, k, background)`.
  Task 1 asserts this before anything else is built.

### A.3 Solution map

| Approach | Attacks | Status in this plan |
|---|---|---|
| CLASS-matched approximations (l_max 12/10/17, exact RSA/UFA, k_max rule) | step count | not in scope (Phase 1/2) |
| One `vmap` instead of `lax.map` chunks; `ForwardMode`+`jacfwd` | chunking, adjoint overhead | not in scope (Phase 1) |
| Fixed-length scan with recorded/heuristic mesh, no inner loops | per-iteration latency | **Task 5 (sequential control arm)** |
| **Linear recurrence: step = matrix, `associative_scan`** | sequential depth N -> log N | **Tasks 2-3, 6 (parallel arm)** |
| A-posteriori error monitor (embedded estimate) | replaces in-loop adaptivity | **Task 7** |
| Forward sensitivities through the scan | gradient cost | **Task 8** |
| Fused Pallas/CUDA step kernel | per-iteration latency | out of scope (Phase 3) |
| Magnus/exponential step for free streaming | step count without RSA | out of scope (R&D) |

### A.4 Supporting literature

- CLASS II (Blas, Lesgourgues, Tram 2011): ndf15, TCA/RSA/UFA — the 8x.
- SymBoltz.jl (Sletmoen 2026, arXiv:2509.24740): Rodas5P + sparse analytic
  Jacobian, approximation-free CPU parity with CLASS.
- DISCO-EB (Hahn, List, Porqueres 2024, arXiv:2311.03291): own batched
  Rodas5 in JAX; 3.5 s A100 for P(k) at rtol 1e-3.
- ABCMB (Zhou, Giovanetti, Liu 2026, arXiv:2602.15104): diffrax Kvaerno5,
  `ForwardMode` + `jacfwd`; 6.7 s H100.
- gradsolve (Spurio Mancini 2026, arXiv:2609.02876, 2609.28458):
  record-and-replay discrete adjoint; fused per-thread kernels (small d).
- Parallel prefix for linear recurrences: Blelloch 1990; Martin & Cundy
  2018 (arXiv:1709.04057); S4/S5/Mamba; Särkkä & García-Fernández 2021;
  Bosch et al. 2024 (JMLR, arXiv:2310.01145) — parallel-in-time ODE
  solvers via associative scan, linear -> logarithmic in N.
- gCAMB (2025, arXiv:2509.25110): GPU gangs over k, time stepping on CPU —
  no parallel-in-time in any Boltzmann solver to date.

---

## Part B — File structure

| File | Responsibility |
|---|---|
| `clax/linrec.py` (create) | `assemble_A`, `assemble_dA_dtau`, `rodas5_step_mat`, `record_mesh`, `pad_meshes`, `solve_sequential`, `solve_parallel`, `error_ratio`, `linrec_perturbations_solve`, jit helpers |
| `tests/test_linrec.py` (create) | unit + validation tests (Tasks 1-9) |
| `scripts/benchmark_linrec.py` (create) | GPU/CPU benchmark harness, appends a markdown table |
| `scripts/slurm/linrec_bench.sbatch` (create) | Bridges-2 job |
| `docs/linrec_results_2026-10.md` (create) | results + go/no-go gates |
| `CHANGELOG.md` (modify, top) | spike entry |

Interfaces shared across tasks (exact signatures; all arrays fp64):

```python
assemble_A(tau: float, args: tuple, n_eq: int) -> Array[n_eq, n_eq]
assemble_dA_dtau(tau: float, args: tuple, n_eq: int) -> Array[n_eq, n_eq]
rodas5_step_mat(A_fn, dA_fn, t0, t1) -> (M: Array[d, d], E: Array[d, d])
record_mesh(k, y0, args, *, prec, idx, tau_ini, tau_max, max_steps=16384) -> (ts: np.ndarray[N+1], ys: np.ndarray[N, d])
pad_meshes(meshes: list[np.ndarray], block_size: int) -> Array[nk, N+1]
solve_sequential(make_args, k_grid: Array[nk], y0: Array[nk, d], mesh: Array[nk, N+1]) -> Array[N, nk, d]
solve_parallel(make_args, k_grid, y0, mesh, n_eq: int, block_size: int) -> (ys: Array[N, nk, d], errs: Array[N, nk, d])
error_ratio(errs, ys, y0, idx, k_grid, rtol, atol) -> Array[N, nk]
jit_solve_sequential(make_args) -> callable(k_grid, y0, mesh)
jit_solve_parallel(make_args, n_eq, block_size) -> callable(k_grid, y0, mesh)
linrec_perturbations_solve(params, prec, bg, th, *, block_size=32, max_steps=16384) -> PerturbationResult
```

`make_args(k)` returns the ODE args tuple consumed by `_perturbation_rhs`:
`(k, bg, th, params, idx, l_max_g, l_max_pol, l_max_ur, ncdmfa_mode_code,
ncdmfa_trigger, q_ncdm, w_ncdm, M_ncdm, dlnf0_ncdm)` — the same tuple
`_perturbations_solve_impl` builds (`clax/perturbations.py`, `ode_args`).

---

## Part C — Tasks

### Task 0: Branch bookkeeping and spec commit

**Files:**
- Commit: `docs/stiff_ode_speed_plan_2026-10-07.md`, `docs/superpowers/plans/2026-10-07-linrec-gpu-spike.md`

- [ ] **Step 1: Confirm the branch**

Run: `git branch --show-current && git log --oneline -1`
Expected: `exp/linrec-gpu-spike` and `e3f04ac Merge pull request #8 ...`

- [ ] **Step 2: Confirm the baseline suite is green before touching code**

Run: `pytest tests/ -q --fast -x 2>&1 | tail -3`
Expected: last line like `N passed, M skipped in ...s` (no failures).

- [ ] **Step 3: Commit the spec and this plan**

```bash
git add docs/stiff_ode_speed_plan_2026-10-07.md docs/superpowers/plans/2026-10-07-linrec-gpu-spike.md
git commit -m "docs: stiff-ODE speed memo and linear-recurrence GPU spike plan"
```

---

### Task 1: Linearity check and `A(tau,k)` assembly

**Files:**
- Create: `clax/linrec.py`
- Create: `tests/test_linrec.py`

**Interfaces:**
- Consumes: `clax.perturbations._perturbation_rhs(tau, y, args)`,
  `_perturbation_solve_setup`, `background_solve`, `thermodynamics_solve`.
- Produces: `assemble_A`, `assemble_dA_dtau`, and the test-module fixture
  `setup` (dict with keys `params, prec, bg, th, idx, n_eq, k_grid,
  tau_grid, tau_ini, tau_max, make_args, args_ncdm`) used by every later
  test.

- [ ] **Step 1: Write the failing test file**

```python
"""Tests for clax/linrec.py — linear-recurrence (parallel-in-time) solver spike.

Numerics-only tests (Tasks 1-7) are exempt from the multi-cosmology rule
(cosmology-independent solver parity); the JVP and C_l tests use the
``lcdm_cosmology`` fixture.
"""
import dataclasses

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from clax import CosmoParams, PrecisionParams, background_solve, thermodynamics_solve
from clax import rosenbrock as R
import clax.perturbations as P
import clax.linrec as L

# Small, CPU-friendly preset: 12 k-modes to k=0.15 Mpc^-1, l_max 17 (n_eq=61),
# Rodas5 recorded meshes at rtol 1e-4.
SMALL = dataclasses.replace(
    PrecisionParams.fit_cl(),
    pt_k_max_cl=0.15, pt_k_per_decade=3, pt_tau_n_points=300,
    pt_ode_rtol=1e-4, pt_ode_atol=1e-8, ode_max_steps=4096,
    hr_n_k_fine=2000, hr_l_max=400,
)


def _build(params, prec):
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)
    (idx, n_eq, k_grid, tau_grid, tau_ini, n_tau, tau_max,
     ncdmfa_mode_code, ncdmfa_trigger, args_ncdm,
     l_max_g, l_max_pol, l_max_ur, l_max_ncdm) = P._perturbation_solve_setup(params, prec, bg, th)
    q_ncdm, w_ncdm, M_ncdm, dlnf0_ncdm = args_ncdm

    def make_args(k):
        return (k, bg, th, params, idx, l_max_g, l_max_pol, l_max_ur,
                ncdmfa_mode_code, ncdmfa_trigger, q_ncdm, w_ncdm, M_ncdm, dlnf0_ncdm)

    def ic(k):
        return P._adiabatic_ic(k, jnp.asarray(tau_ini), bg, params, idx, n_eq, args_ncdm=args_ncdm)

    return dict(params=params, prec=prec, bg=bg, th=th, idx=idx, n_eq=n_eq,
                k_grid=k_grid, tau_grid=tau_grid, tau_ini=float(tau_ini),
                tau_max=float(tau_max), make_args=make_args, args_ncdm=args_ncdm, ic=ic)


@pytest.fixture(scope="module")
def setup():
    return _build(CosmoParams(), SMALL)


def test_rhs_is_linear_homogeneous_and_A_matches(setup):
    """Cosmology-independent numerics check (exempt from multi-cosmology rule):
    f(0)=0, f(a y1 + b y2) = a f(y1) + b f(y2), and assemble_A(tau) @ y == f(y)."""
    s = setup
    d = s["n_eq"]
    args = s["make_args"](s["k_grid"][3])
    rng = np.random.default_rng(0)
    y1 = jnp.asarray(rng.standard_normal(d))
    y2 = jnp.asarray(rng.standard_normal(d))
    worst = 0.0
    for tau in (3.0 * s["tau_ini"], 50.0, 280.0, 5000.0):
        f = lambda y: P._perturbation_rhs(tau, y, args)
        assert float(jnp.max(jnp.abs(f(jnp.zeros(d))))) == 0.0, f"f(0) != 0 at tau={tau}"
        lhs = f(2.0 * y1 - 0.5 * y2)
        rhs = 2.0 * f(y1) - 0.5 * f(y2)
        scale = float(jnp.max(jnp.abs(rhs)))
        worst = max(worst, float(jnp.max(jnp.abs(lhs - rhs))) / scale)
        A = L.assemble_A(tau, args, d)
        assert A.shape == (d, d)
        worst = max(worst, float(jnp.max(jnp.abs(A @ y1 - f(y1)))) / float(jnp.max(jnp.abs(f(y1)))))
    print(f"linearity worst rel err {worst:.2e}")
    assert worst < 1e-10


def test_dA_dtau_matches_finite_difference(setup):
    """Cosmology-independent numerics check: jacfwd(A) in tau vs central FD."""
    s = setup
    d = s["n_eq"]
    args = s["make_args"](s["k_grid"][5])
    tau = 5000.0            # matter era: A varies on the Hubble scale, FD error ~(h/tau)^2
    h = 1e-3 * tau
    dA = L.assemble_dA_dtau(tau, args, d)
    fd = (L.assemble_A(tau + h, args, d) - L.assemble_A(tau - h, args, d)) / (2 * h)
    scale = float(jnp.max(jnp.abs(fd)))
    err = float(jnp.max(jnp.abs(dA - fd))) / scale
    print(f"dA/dtau vs FD rel err {err:.2e}")
    assert err < 1e-5
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_linrec.py -q --fast 2>&1 | tail -3`
Expected: `ModuleNotFoundError: No module named 'clax.linrec'`

- [ ] **Step 3: Create the module with the assembly functions**

```python
"""Linear-recurrence (parallel-in-time) integrator for the scalar
Einstein-Boltzmann system — experimental spike.

The scalar perturbation RHS ``_perturbation_rhs`` is linear and homogeneous
in the state vector, ``y' = A(tau, k) y`` (every TCA/RSA/ncdm-fluid gate
depends on tau, k and background only).  Hence one Rodas5 step (Di Marzo
1993 tableau, ``clax.rosenbrock``) is a matrix map ``y_{n+1} = M_n y_n``
with ``M_n = M(tau_n, h_n, k, theta)`` independent of ``y``.  All ``M_n``
are formed in parallel over (step, k) and composed with
``jax.lax.associative_scan`` (prefix products), so N sequential iterations
become O(log N) parallel rounds per block.
See ``docs/stiff_ode_speed_plan_2026-10-07.md`` section 4.0.

Stage-by-stage mirror of ``clax.rosenbrock.Rodas5.step``: with
f(t, y) = A(t) y the vector stage
    k_i = W^{-1} [ f(t_i, u_i) + dt d_i f_t(t0, y0) + sum_j (c_ij/dt) k_j ],
    u_i = y0 + sum_j a_ij k_j,   W = I/(dt gamma) - A(t0),
becomes, with u_i = U_i y0 and k_i = K_i y0,
    K_i = W^{-1} [ A(t_i) U_i + dt d_i A'(t0) + sum_j (c_ij/dt) K_j ],
    U_i = I + sum_j a_ij K_j,
and the step map is M = U_8 + K_8, the error map E = K_8.

Limitations (spike): fixed mesh (recorded from an adaptive run, or
supplied), a-posteriori error control only (``error_ratio``), dense d x d
step matrices (no sparsity), fp64 only.
"""
from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import numpy as np

from clax import rosenbrock as R
from clax.perturbations import (
    PerturbationPIDControllerConfig,
    PerturbationResult,
    _adiabatic_ic,
    _extract_sources,
    _make_scalar_pid_controller,
    _perturbation_rhs,
    _perturbation_solve_setup,
    _scalar_pid_filtered_variable_indices,
    _scalar_pid_filtered_variable_weights,
)


# ---------------------------------------------------------------------------
# A(tau, k) assembly
# ---------------------------------------------------------------------------

def assemble_A(tau, args, n_eq: int):
    """Return A(tau) such that ``_perturbation_rhs(tau, y, args) == A @ y``.

    The RHS is linear and homogeneous in ``y``, so its Jacobian at any point
    (here y = 0) is the full coefficient matrix.  ``args`` is the ODE args
    tuple of ``_perturbation_rhs`` (k is ``args[0]``).
    """
    return jax.jacfwd(lambda y: _perturbation_rhs(tau, y, args))(jnp.zeros(n_eq))


def assemble_dA_dtau(tau, args, n_eq: int):
    """dA/dtau at ``tau`` by forward-mode AD (Rodas5 needs f_t = A'(t0) y0)."""
    return jax.jacfwd(lambda t: assemble_A(t, args, n_eq))(jnp.asarray(tau, dtype=jnp.float64))
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_linrec.py -q --fast 2>&1 | tail -3`
Expected: `2 passed` (first run compiles bg/thermo; allow ~1 min).

- [ ] **Step 5: Commit**

```bash
git add clax/linrec.py tests/test_linrec.py
git commit -m "feat(linrec): A(tau,k) assembly by jacfwd + linearity tests of the scalar RHS"
```

---

### Task 2: Rodas5 step as a matrix map

**Files:**
- Modify: `clax/linrec.py` (append)
- Modify: `tests/test_linrec.py` (append)

**Interfaces:**
- Consumes: `clax.rosenbrock.Rodas5().step(terms, t0, t1, y0, args, None, False) -> (y1, y_error, ...)`, tableau constants `R._R5_*`.
- Produces: `rodas5_step_mat(A_fn, dA_fn, t0, t1) -> (M, E)`.

- [ ] **Step 1: Write the failing test**

```python
def test_matrix_step_matches_rodas5_vector_step(setup):
    """Cosmology-independent numerics check: M @ y0 == Rodas5.step(...)[0] and
    E @ y0 == its error estimate, for three k-modes and one step from tau_ini."""
    s = setup
    d = s["n_eq"]
    terms = diffrax.ODETerm(P._perturbation_rhs)
    solver = R.Rodas5()
    t0 = s["tau_ini"]
    t1 = 1.1 * t0
    worst = 0.0
    for ik in (0, 6, 11):
        k = s["k_grid"][ik]
        args = s["make_args"](k)
        y0 = s["ic"](k)
        y1_vec, err_vec = solver.step(terms, t0, t1, y0, args, None, False)[:2]
        A_fn = lambda t: L.assemble_A(t, args, d)
        dA_fn = lambda t: L.assemble_dA_dtau(t, args, d)
        M, E = L.rodas5_step_mat(A_fn, dA_fn, t0, t1)
        scale = float(jnp.max(jnp.abs(y1_vec)))
        worst = max(worst, float(jnp.max(jnp.abs(M @ y0 - y1_vec))) / scale)
        worst = max(worst, float(jnp.max(jnp.abs(E @ y0 - err_vec))) / scale)
    print(f"matrix vs vector Rodas5 step worst rel err {worst:.2e}")
    assert worst < 1e-9
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_linrec.py -q --fast -k matrix_step 2>&1 | tail -3`
Expected: `AttributeError: module 'clax.linrec' has no attribute 'rodas5_step_mat'`

- [ ] **Step 3: Implement the matrix step (append to `clax/linrec.py`)**

```python
# ---------------------------------------------------------------------------
# Rodas5 step as a matrix map
# ---------------------------------------------------------------------------

def rodas5_step_mat(A_fn, dA_fn, t0, t1):
    """One Rodas5 step as matrices: ``y1 = M @ y0``, ``y_error = E @ y0``.

    Mirrors ``clax.rosenbrock.Rodas5.step`` stage by stage (same tableau,
    same W = I/(dt gamma) - A(t0), same LU reused by all 8 stages).  A
    zero-length step (t1 <= t0) returns (I, 0) so padded meshes are
    exact no-ops, and the 1/dt terms never see dt = 0.
    """
    dt_raw = t1 - t0
    live = dt_raw > 0
    dt = jnp.where(live, dt_raw, 1.0)
    inv_dt = 1.0 / dt
    A0 = A_fn(t0)
    dA0 = dA_fn(t0)
    n = A0.shape[0]
    I = jnp.eye(n, dtype=A0.dtype)
    W = I / (dt * R._R5_GAMMA) - A0
    lu_piv = jla.lu_factor(W)

    def solve(B):
        return jla.lu_solve(lu_piv, B)

    A1 = A_fn(t0 + dt)

    K1 = solve(A0 + dt * R._R5_D1 * dA0)
    U2 = I + R._R5_A21 * K1
    K2 = solve(A_fn(t0 + R._R5_C2 * dt) @ U2 + dt * R._R5_D2 * dA0
               + inv_dt * (R._R5_C21 * K1))
    U3 = I + R._R5_A31 * K1 + R._R5_A32 * K2
    K3 = solve(A_fn(t0 + R._R5_C3 * dt) @ U3 + dt * R._R5_D3 * dA0
               + inv_dt * (R._R5_C31 * K1 + R._R5_C32 * K2))
    U4 = I + R._R5_A41 * K1 + R._R5_A42 * K2 + R._R5_A43 * K3
    K4 = solve(A_fn(t0 + R._R5_C4 * dt) @ U4 + dt * R._R5_D4 * dA0
               + inv_dt * (R._R5_C41 * K1 + R._R5_C42 * K2 + R._R5_C43 * K3))
    U5 = I + R._R5_A51 * K1 + R._R5_A52 * K2 + R._R5_A53 * K3 + R._R5_A54 * K4
    K5 = solve(A_fn(t0 + R._R5_C5 * dt) @ U5 + dt * R._R5_D5 * dA0
               + inv_dt * (R._R5_C51 * K1 + R._R5_C52 * K2
                           + R._R5_C53 * K3 + R._R5_C54 * K4))
    U6 = (I + R._R5_A61 * K1 + R._R5_A62 * K2 + R._R5_A63 * K3
          + R._R5_A64 * K4 + R._R5_A65 * K5)
    K6 = solve(A1 @ U6
               + inv_dt * (R._R5_C61 * K1 + R._R5_C62 * K2 + R._R5_C63 * K3
                           + R._R5_C64 * K4 + R._R5_C65 * K5))
    U7 = U6 + K6
    K7 = solve(A1 @ U7
               + inv_dt * (R._R5_C71 * K1 + R._R5_C72 * K2 + R._R5_C73 * K3
                           + R._R5_C74 * K4 + R._R5_C75 * K5 + R._R5_C76 * K6))
    U8 = U7 + K7
    K8 = solve(A1 @ U8
               + inv_dt * (R._R5_C81 * K1 + R._R5_C82 * K2 + R._R5_C83 * K3
                           + R._R5_C84 * K4 + R._R5_C85 * K5 + R._R5_C86 * K6
                           + R._R5_C87 * K7))
    M = U8 + K8
    E = K8
    M = jnp.where(live, M, I)
    E = jnp.where(live, E, jnp.zeros_like(E))
    return M, E
```

- [ ] **Step 4: Run the test**

Run: `pytest tests/test_linrec.py -q --fast -k matrix_step 2>&1 | tail -3`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add clax/linrec.py tests/test_linrec.py
git commit -m "feat(linrec): Rodas5 step as a (M, E) matrix map, verified against Rodas5.step"
```

---

### Task 3: Zero-length steps are identities (forward and JVP)

**Files:**
- Modify: `tests/test_linrec.py` (append)

**Interfaces:**
- Consumes: `rodas5_step_mat`.
- Produces: nothing new; pins Review Focus item 1.

- [ ] **Step 1: Write the test**

```python
def test_zero_step_is_identity_and_jvp_finite(setup):
    """Review focus 1: padded steps (t1 == t0) return (I, 0) with no NaN in
    value or in the JVP w.r.t. t1 (the 1/dt terms must be guarded)."""
    s = setup
    d = s["n_eq"]
    args = s["make_args"](s["k_grid"][4])
    A_fn = lambda t: L.assemble_A(t, args, d)
    dA_fn = lambda t: L.assemble_dA_dtau(t, args, d)
    t0 = 100.0
    M, E = L.rodas5_step_mat(A_fn, dA_fn, t0, t0)
    assert np.array_equal(np.asarray(M), np.eye(d))
    assert np.array_equal(np.asarray(E), np.zeros((d, d)))
    tangent = jax.jvp(lambda t1: L.rodas5_step_mat(A_fn, dA_fn, t0, t1)[0], (t0,), (1.0,))[1]
    assert np.all(np.isfinite(np.asarray(tangent)))
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_linrec.py -q --fast -k zero_step 2>&1 | tail -3`
Expected: `1 passed` (the guard is already in Task 2's implementation; if
it fails with NaN, the `jnp.where(live, ...)` guards are missing).

- [ ] **Step 3: Commit**

```bash
git add tests/test_linrec.py
git commit -m "test(linrec): zero-length steps are exact identities with finite JVP"
```

---

### Task 4: Mesh recording and padding

**Files:**
- Modify: `clax/linrec.py` (append)
- Modify: `tests/test_linrec.py` (append)

**Interfaces:**
- Consumes: `diffrax.diffeqsolve` with `SaveAt(t0=True, steps=True)`,
  `_make_scalar_pid_controller(prec=, k=, idx=, config=)`,
  `PerturbationPIDControllerConfig()`.
- Produces: `record_mesh(k, y0, args, *, prec, idx, tau_ini, tau_max, max_steps=16384) -> (ts, ys)` (numpy; `ts[0] == tau_ini`, `ys[n]` is the accepted state at `ts[n+1]`), `pad_meshes(meshes, block_size) -> Array[nk, N+1]`.

- [ ] **Step 1: Write the failing tests**

```python
def _record(s, ik, max_steps=4096):
    k = s["k_grid"][ik]
    ts, ys = L.record_mesh(k, s["ic"](k), s["make_args"](k), prec=s["prec"], idx=s["idx"],
                           tau_ini=s["tau_ini"], tau_max=s["tau_max"], max_steps=max_steps)
    return k, ts, ys


def test_record_mesh_nodes_are_valid(setup):
    """Cosmology-independent plumbing check: recorded nodes start at tau_ini,
    end at tau_max, are strictly increasing, and ys has one row per step."""
    s = setup
    k, ts, ys = _record(s, 7)
    assert ts[0] == s["tau_ini"]
    assert np.isclose(ts[-1], s["tau_max"], rtol=0, atol=1e-9 * s["tau_max"])
    assert np.all(np.diff(ts) > 0)
    assert ys.shape == (len(ts) - 1, s["n_eq"])
    assert 50 < len(ts) < 4096
    print(f"k={float(k):.3g}: {len(ts) - 1} accepted steps")


def test_pad_meshes_shape_and_tail(setup):
    """Cosmology-independent plumbing check: (nk, N+1) with N % B == 0 and
    the padded tail repeating the last node."""
    s = setup
    meshes = [_record(s, ik)[1] for ik in (2, 9)]
    mesh = L.pad_meshes(meshes, block_size=16)
    nk, Np1 = mesh.shape
    assert nk == 2 and (Np1 - 1) % 16 == 0
    assert Np1 - 1 >= max(len(m) - 1 for m in meshes)
    for i, m in enumerate(meshes):
        assert np.array_equal(np.asarray(mesh[i, :len(m)]), m)
        assert np.all(np.asarray(mesh[i, len(m):]) == m[-1])


def test_merged_mesh_contains_tau_grid_exactly(setup):
    """Review focus 2: np.unique(recorded ∪ tau_grid) contains every tau_grid
    value bit-exactly, so searchsorted finds the right node."""
    s = setup
    _, ts, _ = _record(s, 5)
    tau_np = np.asarray(s["tau_grid"])
    merged = np.unique(np.concatenate([ts, tau_np]))
    pos = np.searchsorted(merged, tau_np)
    assert np.array_equal(merged[pos], tau_np)
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_linrec.py -q --fast -k "record_mesh or pad_meshes or merged_mesh" 2>&1 | tail -3`
Expected: `AttributeError: module 'clax.linrec' has no attribute 'record_mesh'`

- [ ] **Step 3: Implement (append to `clax/linrec.py`)**

```python
# ---------------------------------------------------------------------------
# Mesh recording (gradsolve-style record-and-replay) and padding
# ---------------------------------------------------------------------------

def record_mesh(k, y0, args, *, prec, idx, tau_ini, tau_max, max_steps: int = 16384):
    """Run clax's adaptive Rodas5 solve for one k and return its accepted mesh.

    Returns ``(ts, ys)`` as numpy arrays: ``ts`` has the step nodes with
    ``ts[0] == tau_ini`` and ``ts[-1] == tau_max``; ``ys[n]`` is the accepted
    state at ``ts[n + 1]``.  Tolerances come from ``prec.pt_ode_rtol`` /
    ``prec.pt_ode_atol`` through the same filtered PID controller as the
    production path.  Raises if the solve does not reach ``tau_max``.
    """
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(_perturbation_rhs),
        solver=R.Rodas5(),
        t0=tau_ini, t1=tau_max, dt0=tau_ini * 0.1, y0=y0, args=args,
        saveat=diffrax.SaveAt(t0=True, steps=True),
        stepsize_controller=_make_scalar_pid_controller(
            prec=prec, k=k, idx=idx, config=PerturbationPIDControllerConfig()),
        adjoint=diffrax.ForwardMode(),
        max_steps=max_steps,
        throw=False,
    )
    if not bool(sol.result == diffrax.RESULTS.successful):
        raise RuntimeError(f"record_mesh: adaptive solve failed for k={float(k):.4g} "
                           f"({sol.result}); raise max_steps or loosen tolerances")
    ts = np.asarray(sol.ts)
    ys = np.asarray(sol.ys)
    keep = np.isfinite(ts)
    ts = ts[keep]
    ys = ys[keep][1:]          # drop the t0 row: ys[n] belongs to ts[n+1]
    return ts, ys


def pad_meshes(meshes, block_size: int):
    """Stack per-k node arrays into ``(nk, N+1)`` with ``N`` a multiple of
    ``block_size``; padding repeats the last node (zero-length steps are
    identity maps in ``rodas5_step_mat``)."""
    n_steps = max(len(m) - 1 for m in meshes)
    N = -(-n_steps // block_size) * block_size
    out = np.empty((len(meshes), N + 1), dtype=np.float64)
    for i, m in enumerate(meshes):
        out[i, :len(m)] = m
        out[i, len(m):] = m[-1]
    return jnp.asarray(out)
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_linrec.py -q --fast -k "record_mesh or pad_meshes or merged_mesh" 2>&1 | tail -3`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add clax/linrec.py tests/test_linrec.py
git commit -m "feat(linrec): record adaptive Rodas5 meshes and pad them to a common block-aligned length"
```

---

### Task 5: Sequential fixed-mesh control arm

**Files:**
- Modify: `clax/linrec.py` (append)
- Modify: `tests/test_linrec.py` (append)

**Interfaces:**
- Consumes: `R.Rodas5().step`, `record_mesh`, `pad_meshes`.
- Produces: `solve_sequential(make_args, k_grid, y0, mesh) -> Array[N, nk, d]`, `jit_solve_sequential(make_args)`.

- [ ] **Step 1: Write the failing test**

```python
def test_sequential_replay_reproduces_adaptive_states(setup):
    """Cosmology-independent solver parity: replaying the recorded mesh with
    Rodas5.step in a lax.scan reproduces the adaptive run's accepted states
    (same arithmetic, same steps) to rounding."""
    s = setup
    iks = (1, 6, 11)
    rec = [_record(s, ik) for ik in iks]
    k_grid = jnp.asarray([float(r[0]) for r in rec])
    y0 = jnp.stack([s["ic"](k) for k in k_grid])
    mesh = L.pad_meshes([r[1] for r in rec], block_size=16)
    ys = L.jit_solve_sequential(s["make_args"])(k_grid, y0, mesh)   # (N, nk, d)
    worst = 0.0
    for i, (_, ts, ys_ref) in enumerate(rec):
        n = len(ts) - 1
        got = np.asarray(ys[:n, i])
        scale = np.max(np.abs(ys_ref), axis=0) + 1e-300
        worst = max(worst, float(np.max(np.abs(got - ys_ref) / scale)))
        # padded tail must hold the final state
        assert np.array_equal(np.asarray(ys[n:, i]), np.repeat(np.asarray(ys[n - 1:n, i]), ys.shape[0] - n, axis=0))
    print(f"sequential replay vs adaptive worst rel err {worst:.2e}")
    assert worst < 1e-8
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_linrec.py -q --fast -k sequential_replay 2>&1 | tail -3`
Expected: `AttributeError: module 'clax.linrec' has no attribute 'jit_solve_sequential'`

- [ ] **Step 3: Implement (append to `clax/linrec.py`)**

```python
# ---------------------------------------------------------------------------
# Sequential control arm: fixed mesh, vector Rodas5.step in a lax.scan
# ---------------------------------------------------------------------------

def solve_sequential(make_args, k_grid, y0, mesh):
    """Replay a fixed mesh with ``clax.rosenbrock.Rodas5.step`` (vector form)
    in a ``lax.scan`` over steps, ``vmap`` over k.  Returns ``ys`` of shape
    ``(N, nk, d)``: ``ys[n]`` is the state at node ``n + 1``.  Zero-length
    (padded) steps leave the state unchanged."""
    terms = diffrax.ODETerm(_perturbation_rhs)
    solver = R.Rodas5()

    def one_k(k, y0_k, nodes):
        args = make_args(k)

        def body(y, t01):
            t0, t1 = t01
            live = t1 > t0
            t1_safe = jnp.where(live, t1, t0 + 1.0)
            y1 = solver.step(terms, t0, t1_safe, y, args, None, False)[0]
            y1 = jnp.where(live, y1, y)
            return y1, y1

        _, ys = jax.lax.scan(body, y0_k, (nodes[:-1], nodes[1:]))
        return ys

    ys = jax.vmap(one_k)(k_grid, y0, mesh)        # (nk, N, d)
    return jnp.swapaxes(ys, 0, 1)


def jit_solve_sequential(make_args):
    """``jax.jit`` wrapper with ``make_args`` closed over (functions are not
    valid traced arguments)."""
    return jax.jit(lambda k_grid, y0, mesh: solve_sequential(make_args, k_grid, y0, mesh))
```

- [ ] **Step 4: Run the test**

Run: `pytest tests/test_linrec.py -q --fast -k sequential_replay 2>&1 | tail -3`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add clax/linrec.py tests/test_linrec.py
git commit -m "feat(linrec): sequential fixed-mesh Rodas5 replay (control arm) matches adaptive states"
```

---

### Task 6: Parallel-in-time solve (matrix steps + blocked associative scan)

**Files:**
- Modify: `clax/linrec.py` (append)
- Modify: `tests/test_linrec.py` (append)

**Interfaces:**
- Consumes: `assemble_A`, `assemble_dA_dtau`, `rodas5_step_mat`, `solve_sequential`.
- Produces: `solve_parallel(make_args, k_grid, y0, mesh, n_eq, block_size) -> (ys, errs)` each `(N, nk, d)`; `errs[n] = E_n @ y_{n-1}` (embedded 4th-order error of step n); `jit_solve_parallel(make_args, n_eq, block_size)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_parallel_matches_sequential(setup):
    """Cosmology-independent solver parity: associative-scan prefix products
    reproduce the sequential replay to rounding, across >= 2 blocks and with
    padded modes (different N per k)."""
    s = setup
    iks = (1, 6, 11)
    rec = [_record(s, ik) for ik in iks]
    k_grid = jnp.asarray([float(r[0]) for r in rec])
    y0 = jnp.stack([s["ic"](k) for k in k_grid])
    mesh = L.pad_meshes([r[1] for r in rec], block_size=16)
    assert mesh.shape[1] - 1 >= 32, "need >= 2 blocks for this test"
    ys_seq = L.jit_solve_sequential(s["make_args"])(k_grid, y0, mesh)
    ys_par, errs = L.jit_solve_parallel(s["make_args"], s["n_eq"], 16)(k_grid, y0, mesh)
    assert ys_par.shape == ys_seq.shape == errs.shape
    scale = jnp.max(jnp.abs(ys_seq), axis=0, keepdims=True) + 1e-300
    worst = float(jnp.max(jnp.abs(ys_par - ys_seq) / scale))
    print(f"parallel vs sequential worst rel err {worst:.2e}")
    assert worst < 1e-6
    assert bool(jnp.all(jnp.isfinite(errs)))


def test_bad_block_size_raises(setup):
    """Review focus 3: N not a multiple of block_size must raise."""
    s = setup
    _, ts, _ = _record(s, 3)
    k_grid = jnp.asarray([float(s["k_grid"][3])])
    y0 = s["ic"](k_grid[0])[None]
    mesh = L.pad_meshes([ts], block_size=16)
    with pytest.raises(ValueError):
        L.solve_parallel(s["make_args"], k_grid, y0, mesh, s["n_eq"], block_size=7)
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_linrec.py -q --fast -k "parallel_matches or bad_block" 2>&1 | tail -3`
Expected: `AttributeError: module 'clax.linrec' has no attribute 'jit_solve_parallel'`

- [ ] **Step 3: Implement (append to `clax/linrec.py`)**

```python
# ---------------------------------------------------------------------------
# Parallel-in-time arm: matrix steps formed in parallel, composed by a
# blocked associative scan
# ---------------------------------------------------------------------------

def solve_parallel(make_args, k_grid, y0, mesh, n_eq: int, block_size: int):
    """Linear-recurrence solve on a fixed mesh.

    ``mesh`` is ``(nk, N+1)`` with ``N % block_size == 0``.  Per block of
    ``B = block_size`` steps: form ``(M_n, E_n)`` for all (k, n) in parallel,
    compute prefix products ``P_n = M_n ... M_1`` with
    ``jax.lax.associative_scan`` (depth O(log B)), apply them to the block's
    start state, and carry the end state to the next block (``lax.scan``
    over blocks bounds memory to O(B nk d^2)).

    Returns ``(ys, errs)``, both ``(N, nk, d)``: ``ys[n]`` is the state at
    node ``n+1``; ``errs[n] = E_n @ y_n_prev`` is the embedded error of step
    ``n`` (feed to ``error_ratio``).
    """
    nk, Np1 = mesh.shape
    N = Np1 - 1
    if N % block_size != 0:
        raise ValueError(f"mesh has N={N} steps, not a multiple of block_size={block_size}; "
                         "use pad_meshes(meshes, block_size)")
    nb = N // block_size
    node_idx = (jnp.arange(nb) * block_size)[:, None] + jnp.arange(block_size + 1)[None, :]
    nodes = jnp.swapaxes(mesh[:, node_idx], 0, 1)          # (nb, nk, B+1)

    def step_mats_one_k(k, nodes_k):                        # nodes_k: (B+1,)
        args = make_args(k)
        A_fn = lambda t: assemble_A(t, args, n_eq)
        dA_fn = lambda t: assemble_dA_dtau(t, args, n_eq)
        return jax.vmap(lambda t0, t1: rodas5_step_mat(A_fn, dA_fn, t0, t1))(nodes_k[:-1], nodes_k[1:])

    def compose(a, b):                                      # apply a first, then b
        return jnp.matmul(b, a)

    def block(y_start, nodes_b):                            # y_start (nk, d); nodes_b (nk, B+1)
        M, E = jax.vmap(step_mats_one_k)(k_grid, nodes_b)   # (nk, B, d, d) each
        M = jnp.swapaxes(M, 0, 1)                           # (B, nk, d, d)
        P = jax.lax.associative_scan(compose, M, axis=0)    # P[n] = M[n] ... M[0]
        ys = jnp.einsum("bkij,kj->bki", P, y_start)         # (B, nk, d)
        y_prev = jnp.concatenate([y_start[None], ys[:-1]], axis=0)
        errs = jnp.einsum("kbij,bkj->bki", E, y_prev)       # E_n applied to the pre-step state
        return ys[-1], (ys, errs)

    _, (ys, errs) = jax.lax.scan(block, y0, nodes)          # (nb, B, nk, d)
    return ys.reshape(N, nk, n_eq), errs.reshape(N, nk, n_eq)


def jit_solve_parallel(make_args, n_eq: int, block_size: int):
    """``jax.jit`` wrapper with the static arguments closed over."""
    return jax.jit(lambda k_grid, y0, mesh: solve_parallel(make_args, k_grid, y0, mesh, n_eq, block_size))
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_linrec.py -q --fast -k "parallel_matches or bad_block" 2>&1 | tail -3`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add clax/linrec.py tests/test_linrec.py
git commit -m "feat(linrec): parallel-in-time solve via matrix steps + blocked associative_scan"
```

---

### Task 7: A-posteriori error monitor

**Files:**
- Modify: `clax/linrec.py` (append)
- Modify: `tests/test_linrec.py` (append)

**Interfaces:**
- Consumes: `_scalar_pid_filtered_variable_indices(idx)`, `_scalar_pid_filtered_variable_weights(k)`, `solve_parallel` outputs.
- Produces: `error_ratio(errs, ys, y0, idx, k_grid, rtol, atol) -> Array[N, nk]` (<= 1 means "would have been accepted").

- [ ] **Step 1: Write the failing test**

```python
def test_error_ratio_flags_coarse_mesh(setup):
    """Review focus 4: on the recorded mesh the filtered error ratio stays
    O(1) (steps were accepted); on a 2x-coarsened mesh it exceeds 1."""
    s = setup
    iks = (4, 10)
    rec = [_record(s, ik) for ik in iks]
    k_grid = jnp.asarray([float(r[0]) for r in rec])
    y0 = jnp.stack([s["ic"](k) for k in k_grid])
    mesh = L.pad_meshes([r[1] for r in rec], block_size=32)
    rtol, atol = s["prec"].pt_ode_rtol, s["prec"].pt_ode_atol
    ys, errs = L.jit_solve_parallel(s["make_args"], s["n_eq"], 16)(k_grid, y0, mesh)
    fine = L.error_ratio(errs, ys, y0, s["idx"], k_grid, rtol, atol)
    coarse_mesh = mesh[:, ::2]                                  # N/2 steps, still % 16 == 0
    ys_c, errs_c = L.jit_solve_parallel(s["make_args"], s["n_eq"], 16)(k_grid, y0, coarse_mesh)
    coarse = L.error_ratio(errs_c, ys_c, y0, s["idx"], k_grid, rtol, atol)
    print(f"max error ratio: recorded mesh {float(fine.max()):.3f}, 2x coarser {float(coarse.max()):.3f}")
    assert fine.shape == (mesh.shape[1] - 1, 2)
    assert float(fine.max()) < 1.5
    assert float(coarse.max()) > 1.0
    assert float(coarse.max()) > float(fine.max())
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_linrec.py -q --fast -k error_ratio 2>&1 | tail -3`
Expected: `AttributeError: module 'clax.linrec' has no attribute 'error_ratio'`

- [ ] **Step 3: Implement (append to `clax/linrec.py`)**

```python
# ---------------------------------------------------------------------------
# A-posteriori error monitor (same measure as the adaptive filtered PID)
# ---------------------------------------------------------------------------

def error_ratio(errs, ys, y0, idx, k_grid, rtol, atol):
    """Filtered RMS of ``err / (atol + rtol |y_prev|)`` per (step, k).

    Uses the same six filtered variables and k-dependent weights as the
    production PID controller (``_scalar_pid_filtered_variable_*``), so a
    ratio <= 1 means the fixed mesh would have been accepted step by step.
    Returns ``(N, nk)``.
    """
    y_prev = jnp.concatenate([y0[None], ys[:-1]], axis=0)        # (N, nk, d)
    scaled = errs / (atol + rtol * jnp.abs(y_prev))
    fidx = _scalar_pid_filtered_variable_indices(idx)             # (6,)
    w = jax.vmap(_scalar_pid_filtered_variable_weights)(k_grid)   # (nk, 6)
    e = scaled[:, :, fidx] * w[None]                              # (N, nk, 6)
    return jnp.sqrt(jnp.mean(e * e, axis=-1))
```

- [ ] **Step 4: Run the test**

Run: `pytest tests/test_linrec.py -q --fast -k error_ratio 2>&1 | tail -3`
Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add clax/linrec.py tests/test_linrec.py
git commit -m "feat(linrec): a-posteriori filtered error ratio for fixed meshes"
```

---

### Task 8: Forward-mode AD through assembly and scan

**Files:**
- Modify: `tests/test_linrec.py` (append)

**Interfaces:**
- Consumes: `jit_solve_parallel`, `lcdm_cosmology` fixture (`tests/conftest.py`), `_build` helper from Task 1.
- Produces: nothing new; proves `jax.jvp` flows through `assemble_A` (jacfwd-of-jacfwd), `lu_factor/lu_solve`, `associative_scan`.

- [ ] **Step 1: Write the test**

```python
def test_jvp_wrt_k_matches_finite_difference(lcdm_cosmology):
    """Physics-facing gradient test (multi-cosmology rule): d(final state)/dk
    through the parallel solve on a fixed mesh vs central finite differences,
    on the six filtered variables."""
    name, params = lcdm_cosmology
    s = _build(params, SMALL)
    k, ts, _ = _record(s, 6)
    k = float(k)
    y0 = s["ic"](k)[None]
    mesh = L.pad_meshes([ts], block_size=16)
    fidx = np.asarray(P._scalar_pid_filtered_variable_indices(s["idx"]))
    solve = L.jit_solve_parallel(s["make_args"], s["n_eq"], 16)

    def final_state(kk):
        return solve(jnp.asarray([kk]), y0, mesh)[0][-1, 0][fidx]

    jvp = np.asarray(jax.jvp(final_state, (k,), (1.0,))[1])
    h = 1e-3 * k
    fd = np.asarray((final_state(k + h) - final_state(k - h)) / (2 * h))
    denom = np.maximum(np.abs(fd), 1e-3 * np.max(np.abs(fd)))
    err = float(np.max(np.abs(jvp - fd) / denom))
    print(f"[{name}] jvp vs FD (dk) worst rel err {err:.2e}")
    assert np.all(np.isfinite(jvp))
    assert err < 1e-4
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_linrec.py -q --fast -k jvp 2>&1 | tail -3`
Expected: `1 passed, 4 skipped` (non-fiducial grid points skip under `--fast`).

- [ ] **Step 3: Commit**

```bash
git add tests/test_linrec.py
git commit -m "test(linrec): forward-mode JVP through assembly + associative scan matches finite differences"
```

---

### Task 9: Drop-in `PerturbationResult` and end-to-end C_l parity

**Files:**
- Modify: `clax/linrec.py` (append)
- Modify: `tests/test_linrec.py` (append)

**Interfaces:**
- Consumes: `_perturbation_solve_setup`, `_adiabatic_ic`, `_extract_sources(y, k, tau, bg, th, idx, params, q_ncdm=, w_ncdm=, M_ncdm=, ncdmfa_mode_code=, ncdmfa_trigger=)` (returns a 14-tuple), `PerturbationResult` (14 source fields incl. `delta_cb`), `clax.harmonic.compute_cls_all_fast(pt, params, bg, l_max=, n_k_fine=) -> {'ell','tt','ee','te'}`, `clax.perturbations.perturbations_solve`.
- Produces: `linrec_perturbations_solve(params, prec, bg, th, *, block_size=32, max_steps=16384) -> PerturbationResult`.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.slow
def test_cl_parity_with_standard_pipeline(lcdm_cosmology):
    """Physics-facing value test (multi-cosmology rule, slow): sources on
    tau_grid from the parallel solve vs the standard adaptive path give the
    same C_l^TT/EE to 0.2% at l in {20, 100, 300} and the same delta_m(z=0)."""
    from clax.harmonic import compute_cls_all_fast
    name, params = lcdm_cosmology
    s = _build(params, SMALL)
    pt_std = P.perturbations_solve(params, SMALL, s["bg"], s["th"])
    pt_lin = L.linrec_perturbations_solve(params, SMALL, s["bg"], s["th"], block_size=16, max_steps=4096)
    assert pt_lin.source_T0.shape == pt_std.source_T0.shape
    dm = float(np.max(np.abs(pt_lin.delta_m[:, -1] / pt_std.delta_m[:, -1] - 1.0)))
    cl_std = compute_cls_all_fast(pt_std, params, s["bg"], l_max=400, n_k_fine=2000)
    cl_lin = compute_cls_all_fast(pt_lin, params, s["bg"], l_max=400, n_k_fine=2000)
    ells = np.asarray(cl_std["ell"])
    worst = 0.0
    for l in (20, 100, 300):
        i = int(np.argmin(np.abs(ells - l)))
        for key in ("tt", "ee"):
            worst = max(worst, abs(float(cl_lin[key][i] / cl_std[key][i]) - 1.0))
    print(f"[{name}] delta_m(z=0) worst rel {dm:.2e}; C_l worst rel {worst:.2e}")
    assert dm < 1e-3
    assert worst < 2e-3
```

- [ ] **Step 2: Run to verify it fails (full mode, not --fast)**

Run: `pytest tests/test_linrec.py -q -k cl_parity -x 2>&1 | tail -3`
Expected: `AttributeError: module 'clax.linrec' has no attribute 'linrec_perturbations_solve'`

- [ ] **Step 3: Implement (append to `clax/linrec.py`)**

```python
# ---------------------------------------------------------------------------
# Drop-in PerturbationResult builder (sources exactly at tau_grid nodes)
# ---------------------------------------------------------------------------

def linrec_perturbations_solve(params, prec, bg, th, *, block_size: int = 32, max_steps: int = 16384):
    """Experimental replacement for ``perturbations_solve``.

    Records a per-k adaptive Rodas5 mesh at ``prec.pt_ode_rtol/atol``,
    merges it with ``tau_grid`` (so every output time is a mesh node and no
    dense output is needed), runs ``solve_parallel`` and extracts the 14
    source functions with ``_extract_sources``.  Mesh recording is a Python
    loop over k (one-off cost; amortised across an MCMC in production use).
    """
    (idx, n_eq, k_grid, tau_grid, tau_ini, n_tau, tau_max,
     ncdmfa_mode_code, ncdmfa_trigger, args_ncdm,
     l_max_g, l_max_pol, l_max_ur, l_max_ncdm) = _perturbation_solve_setup(params, prec, bg, th)
    q_ncdm, w_ncdm, M_ncdm, dlnf0_ncdm = args_ncdm
    nk = int(k_grid.shape[0])

    def make_args(k):
        return (k, bg, th, params, idx, l_max_g, l_max_pol, l_max_ur,
                ncdmfa_mode_code, ncdmfa_trigger, q_ncdm, w_ncdm, M_ncdm, dlnf0_ncdm)

    ic = jax.jit(lambda k: _adiabatic_ic(k, jnp.asarray(tau_ini), bg, params, idx, n_eq, args_ncdm=args_ncdm))
    y0 = jax.vmap(ic)(k_grid)                                   # (nk, d)

    tau_np = np.asarray(tau_grid)
    meshes = []
    for i in range(nk):
        ts, _ = record_mesh(k_grid[i], y0[i], make_args(k_grid[i]), prec=prec, idx=idx,
                            tau_ini=float(tau_ini), tau_max=float(tau_max), max_steps=max_steps)
        meshes.append(np.unique(np.concatenate([ts, tau_np])))
    mesh = pad_meshes(meshes, block_size)                       # (nk, N+1)

    ys, _ = jit_solve_parallel(make_args, n_eq, block_size)(k_grid, y0, mesh)   # (N, nk, d)
    states = jnp.concatenate([y0[None], ys], axis=0)            # (N+1, nk, d): node n -> states[n]
    mesh_np = np.asarray(mesh)
    pos = np.stack([np.searchsorted(mesh_np[i], tau_np) for i in range(nk)])   # (nk, n_tau)
    y_at = states[jnp.asarray(pos).T, jnp.arange(nk)[None, :]]  # (n_tau, nk, d)

    def extract(y, k, tau):
        return jnp.stack(_extract_sources(
            y, k, tau, bg, th, idx, params, q_ncdm=q_ncdm, w_ncdm=w_ncdm, M_ncdm=M_ncdm,
            ncdmfa_mode_code=ncdmfa_mode_code, ncdmfa_trigger=ncdmfa_trigger))

    src = jax.vmap(jax.vmap(extract, in_axes=(0, None, 0)), in_axes=(1, 0, None))(y_at, k_grid, tau_grid)
    src = jnp.moveaxis(src, -1, 0)                              # (14, nk, n_tau)
    return PerturbationResult(
        k_grid=k_grid, tau_grid=tau_grid,
        source_T0=src[0], source_T1=src[1], source_T2=src[2], source_E=src[3],
        source_lens=src[4], delta_m=src[5], source_SW=src[6], source_ISW_vis=src[7],
        source_ISW_fs=src[8], source_Doppler=src[9], source_Doppler_nonIBP=src[10],
        source_T0_noDopp=src[11], source_phi_plus_psi=src[12], delta_cb=src[13],
    )
```

- [ ] **Step 4: Run the test (full mode; ~5-10 min on CPU for 5 cosmologies)**

Run: `pytest tests/test_linrec.py -q -k cl_parity 2>&1 | tail -3`
Expected: `5 passed`

- [ ] **Step 5: Run the whole file in fast mode and the baseline suite**

Run: `pytest tests/test_linrec.py -q --fast 2>&1 | tail -3 && pytest tests/ -q --fast -x 2>&1 | tail -2`
Expected: linrec `10 passed, 5 skipped`-style line (slow + non-fiducial skipped); baseline unchanged.

- [ ] **Step 6: Commit**

```bash
git add clax/linrec.py tests/test_linrec.py
git commit -m "feat(linrec): drop-in PerturbationResult from the parallel solve; C_l parity with the standard path"
```

---

### Task 10: Benchmark harness and Bridges-2 job

**Files:**
- Create: `scripts/benchmark_linrec.py`
- Create: `scripts/slurm/linrec_bench.sbatch`

**Interfaces:**
- Consumes: `clax.perturbations.perturbations_solve` (current production path), `record_mesh`, `pad_meshes`, `jit_solve_sequential`, `jit_solve_parallel`, `error_ratio`.
- Produces: a markdown table appended to `--out` with one row per block size; columns `GPU | preset | nk | d | N | B | t_current | t_seq | t_par | t_mats(B) | t_scan(B) | t_jvp | peak_GB | max_err_ratio`.

- [ ] **Step 1: Write the benchmark script**

```python
"""Benchmark the linear-recurrence (parallel-in-time) perturbation solver.

Usage (GPU node):
    python scripts/benchmark_linrec.py --preset fit_cl --k-per-decade 20 --k-max 1.0 \
        --rtol 1e-4 --atol 1e-8 --block-sizes 16,32,64 --out docs/linrec_results_2026-10.md

Arms (all cached, second call timed):
  current : clax.perturbations.perturbations_solve (adaptive, production path)
  seq     : fixed recorded mesh, vector Rodas5.step in lax.scan   (control)
  par     : matrix steps + blocked associative_scan               (treatment)
  mats/scan: per-block phase timings of `par` (step-matrix formation / scan)
  jvp     : jax.jvp of `par` w.r.t. k_grid (forward sensitivities)
Peak device memory from jax.devices()[0].memory_stats() (GPU only).
"""
import argparse
import dataclasses
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from clax import CosmoParams, PrecisionParams, background_solve, thermodynamics_solve
import clax.perturbations as P
import clax.linrec as L


def timed(fn, *args):
    """Warm up once (compile), then time one cached call."""
    jax.block_until_ready(fn(*args))
    t0 = time.perf_counter()
    out = jax.block_until_ready(fn(*args))
    return time.perf_counter() - t0, out


def peak_gb():
    stats = jax.devices()[0].memory_stats()
    return (stats or {}).get("peak_bytes_in_use", 0) / 1024**3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="fit_cl", choices=["fit_cl", "planck_cl"])
    ap.add_argument("--k-per-decade", type=int, default=20)
    ap.add_argument("--k-max", type=float, default=1.0)
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--atol", type=float, default=1e-8)
    ap.add_argument("--l-max", type=int, default=17, help="photon/pol/ur hierarchy l_max")
    ap.add_argument("--block-sizes", default="16,32,64")
    ap.add_argument("--max-steps", type=int, default=16384)
    ap.add_argument("--out", default="docs/linrec_results_2026-10.md")
    a = ap.parse_args()

    device = jax.devices()[0]
    print(f"device: {device.platform} {device.device_kind}")
    base = getattr(PrecisionParams, a.preset)()
    prec = dataclasses.replace(
        base, pt_k_per_decade=a.k_per_decade, pt_k_max_cl=a.k_max,
        pt_ode_rtol=a.rtol, pt_ode_atol=a.atol, ode_max_steps=a.max_steps,
        pt_l_max_g=a.l_max, pt_l_max_pol_g=a.l_max, pt_l_max_ur=a.l_max,
        pt_ode_solver="rodas5")
    params = CosmoParams()
    bg = background_solve(params, prec)
    th = thermodynamics_solve(params, prec, bg)

    (idx, n_eq, k_grid, tau_grid, tau_ini, n_tau, tau_max,
     mode, trig, args_ncdm, lg, lp, lu, ln) = P._perturbation_solve_setup(params, prec, bg, th)
    q, w, M, dlnf0 = args_ncdm

    def make_args(k):
        return (k, bg, th, params, idx, lg, lp, lu, mode, trig, q, w, M, dlnf0)

    ic = jax.jit(lambda k: P._adiabatic_ic(k, jnp.asarray(tau_ini), bg, params, idx, n_eq, args_ncdm=args_ncdm))
    y0 = jax.vmap(ic)(k_grid)
    nk = int(k_grid.shape[0])

    t0 = time.perf_counter()
    meshes = [L.record_mesh(k_grid[i], y0[i], make_args(k_grid[i]), prec=prec, idx=idx,
                            tau_ini=float(tau_ini), tau_max=float(tau_max), max_steps=a.max_steps)[0]
              for i in range(nk)]
    t_record = time.perf_counter() - t0
    steps = np.array([len(m) - 1 for m in meshes])
    print(f"nk={nk} d={n_eq} recorded steps: min {steps.min()} median {int(np.median(steps))} max {steps.max()} "
          f"(recording {t_record:.1f}s incl. compile)")

    # current production path
    t_current, _ = timed(lambda: P.perturbations_solve(params, prec, bg, th))
    print(f"current perturbations_solve (adaptive {prec.pt_ode_solver}): {t_current:.2f}s")

    rows = []
    for B in [int(b) for b in a.block_sizes.split(",")]:
        mesh = L.pad_meshes(meshes, B)
        N = mesh.shape[1] - 1
        t_seq, _ = timed(L.jit_solve_sequential(make_args), k_grid, y0, mesh)
        par = L.jit_solve_parallel(make_args, n_eq, B)
        t_par, (ys, errs) = timed(par, k_grid, y0, mesh)
        ratio = float(L.error_ratio(errs, ys, y0, idx, k_grid, a.rtol, a.atol).max())
        mem = peak_gb()

        # phase timings on one block of B steps
        nodes_b = mesh[:, :B + 1]
        mats = jax.jit(lambda kk, nb: jax.vmap(lambda k, n: jax.vmap(
            lambda t0_, t1_: L.rodas5_step_mat(
                lambda t: L.assemble_A(t, make_args(k), n_eq),
                lambda t: L.assemble_dA_dtau(t, make_args(k), n_eq), t0_, t1_))(n[:-1], n[1:]))(kk, nb))
        t_mats, (Mb, _) = timed(mats, k_grid, nodes_b)
        Mb = jnp.swapaxes(Mb, 0, 1)
        scan = jax.jit(lambda m: jax.lax.associative_scan(lambda x, y: jnp.matmul(y, x), m, axis=0))
        t_scan, _ = timed(scan, Mb)

        jvp_fn = jax.jit(lambda kk: jax.jvp(lambda kkk: par(kkk, y0, mesh)[0], (kk,), (jnp.ones_like(kk),))[1])
        t_jvp, _ = timed(jvp_fn, k_grid)

        print(f"B={B:4d} N={N:5d}  seq {t_seq:7.2f}s  par {t_par:7.2f}s  "
              f"mats/block {t_mats:6.3f}s  scan/block {t_scan:6.3f}s  jvp {t_jvp:7.2f}s  "
              f"peak {mem:5.1f} GB  max err ratio {ratio:.2f}")
        rows.append(f"| {device.device_kind} | {a.preset} | {nk} | {n_eq} | {N} | {B} | {t_current:.2f} | "
                    f"{t_seq:.2f} | {t_par:.2f} | {t_mats:.3f} | {t_scan:.3f} | {t_jvp:.2f} | {mem:.1f} | {ratio:.2f} |")

    header = ("| GPU | preset | nk | d | N | B | t_current (s) | t_seq (s) | t_par (s) | t_mats/block (s) | "
              "t_scan/block (s) | t_jvp (s) | peak GB | max err ratio |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    with open(a.out, "a") as f:
        f.write(f"\n<!-- benchmark_linrec {time.strftime('%Y-%m-%d %H:%M')} rtol={a.rtol} atol={a.atol} l_max={a.l_max} -->\n")
        f.write(header + "\n".join(rows) + "\n")
    print(f"appended {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test on CPU with a tiny configuration**

Run: `python scripts/benchmark_linrec.py --k-per-decade 3 --k-max 0.15 --block-sizes 16 --max-steps 4096 --out /tmp/linrec_smoke.md 2>&1 | tail -4 && tail -3 /tmp/linrec_smoke.md`
Expected: lines `device: cpu ...`, `nk=12 d=61 recorded steps: ...`, `B=  16 N=...`, `appended 1 rows`, and the markdown row (CPU `peak` prints 0.0).

- [ ] **Step 3: Write the Bridges-2 job script**

```bash
#!/bin/bash
#SBATCH -J linrec_bench
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -o linrec_bench_%j.out
# Usage: CLAX_PYTHON=/path/to/env/bin/python sbatch scripts/slurm/linrec_bench.sbatch
# (CLAX_PYTHON defaults to the `python` on PATH; set it to the env that runs
#  scripts/gpu_planck_test.py on Bridges-2.)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
PY="${CLAX_PYTHON:-python}"
export JAX_ENABLE_X64=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
OUT=docs/linrec_results_2026-10.md
# fit_cl-like (d=61), production k-range, two tolerances
$PY scripts/benchmark_linrec.py --preset fit_cl --k-per-decade 20 --k-max 1.0 --rtol 1e-4 --atol 1e-8 --block-sizes 16,32,64,128 --out $OUT
$PY scripts/benchmark_linrec.py --preset fit_cl --k-per-decade 20 --k-max 1.0 --rtol 1e-6 --atol 1e-11 --block-sizes 16,32,64 --out $OUT
# production mode count (~300 modes) at the CLASS k_max rule (0.32 Mpc^-1)
$PY scripts/benchmark_linrec.py --preset fit_cl --k-per-decade 65 --k-max 0.32 --rtol 1e-4 --atol 1e-8 --block-sizes 16,32,64 --out $OUT
# l_max 50 (d=160): memory/flop scaling
$PY scripts/benchmark_linrec.py --preset fit_cl --k-per-decade 20 --k-max 1.0 --rtol 1e-4 --atol 1e-8 --l-max 50 --block-sizes 8,16,32 --out $OUT
# validation suite in full mode (5 cosmologies, slow tests on)
$PY -m pytest tests/test_linrec.py -q 2>&1 | tail -5
```

- [ ] **Step 4: Commit**

```bash
chmod +x scripts/slurm/linrec_bench.sbatch
git add scripts/benchmark_linrec.py scripts/slurm/linrec_bench.sbatch
git commit -m "bench(linrec): GPU benchmark harness + Bridges-2 job for the parallel-in-time spike"
```

---

### Task 11: Run on Bridges-2, record results, decide

**Files:**
- Create: `docs/linrec_results_2026-10.md`
- Modify: `CHANGELOG.md` (insert after the `## Status:` paragraph at the top)

**Interfaces:**
- Consumes: benchmark rows from Task 10, `pytest tests/test_linrec.py -q` output from the job.
- Produces: the results document with the five gates filled from measurements.

- [ ] **Step 1: Create the results document skeleton (before the job, so the benchmark appends into it)**

```markdown
# Linear-recurrence (parallel-in-time) spike — GPU results

Spec: `docs/stiff_ode_speed_plan_2026-10-07.md` §4.0. Plan:
`docs/superpowers/plans/2026-10-07-linrec-gpu-spike.md`. Branch
`exp/linrec-gpu-spike`.

## Gates (fill from the tables below)

| Gate | Criterion | Measured | Pass? |
|---|---|---|---|
| G1 correctness | `pytest tests/test_linrec.py -q` (full mode, 5 cosmologies) all pass on the GPU node | | |
| G2 parallel vs sequential | `t_par <= 0.25 x t_seq` at nk=100, d=61, rtol 1e-4, best B | | |
| G3 vs production path | `t_par <= 0.2 x t_current` at the same rtol/atol | | |
| G4 memory | some B gives `peak <= 24 GB` at nk~300, d=61 | | |
| G5 gradient | `t_jvp <= 3 x t_par` | | |

Decision rule: G1 and G2 and G4 pass -> adopt the linear-recurrence
integrator as the Phase 2 core (spec §5); G2 fails but G3 passes -> keep
the fixed-mesh *sequential* scan as Phase 2 core and drop the parallel
scan; G3 fails -> the GPU per-iteration latency is not where the spike
assumed; re-profile the production path first (spec §5 Phase 0 items 1-4).

## Hardware

(nvidia-smi line from the job output)

## Tables (appended by scripts/benchmark_linrec.py)
```

- [ ] **Step 2: Submit the job and wait**

Run (on Bridges-2 login node, repo at `/ocean/projects/phy230064p/smishrasharma/jaxclass/`, branch checked out and up to date):
```bash
CLAX_PYTHON=$(which python) sbatch scripts/slurm/linrec_bench.sbatch
squeue -u $USER
```
Expected: job id printed; when finished, `linrec_bench_<id>.out` ends with the pytest summary line and `docs/linrec_results_2026-10.md` has four appended tables.

- [ ] **Step 3: Fill the gates table from the measurements**

Copy the `nvidia-smi` line into `## Hardware`, and for each gate write the
measured ratio (e.g. `t_par/t_seq = 0.18 at B=32`) and `yes`/`no`. If G1
fails, paste the failing test names and stop: do not fill G2-G5.

- [ ] **Step 4: Add the CHANGELOG entry (top of file, after the Status paragraph)**

```markdown
### Oct 7, 2026: Linear-recurrence (parallel-in-time) perturbation solver spike

**Experimental `clax/linrec.py`: the scalar RHS is linear in the state, so
each Rodas5 step is a matrix `M_n`; all `M_n` are formed in parallel over
(step, k) and composed with `jax.lax.associative_scan` inside a blocked
`lax.scan`.** Meshes are recorded from the adaptive Rodas5 solve
(gradsolve-style record-and-replay); an a-posteriori filtered error ratio
replaces in-loop adaptivity. `linrec_perturbations_solve` is a drop-in
`PerturbationResult` builder.

- Tests: `tests/test_linrec.py` — RHS linearity, matrix step == `Rodas5.step`,
  sequential replay == adaptive states, parallel == sequential (rounding),
  error monitor flags a 2x coarser mesh, JVP vs FD, C_l parity (slow).
- Benchmark: `scripts/benchmark_linrec.py`, job `scripts/slurm/linrec_bench.sbatch`.
- Results and go/no-go gates: `docs/linrec_results_2026-10.md`.
- Spec and literature: `docs/stiff_ode_speed_plan_2026-10-07.md` (§4.0).
```

- [ ] **Step 5: Commit**

```bash
git add docs/linrec_results_2026-10.md CHANGELOG.md
git commit -m "docs(linrec): Bridges-2 benchmark results, gates, and CHANGELOG entry"
```

---

## Self-review checklist (done by the plan author)

1. Spec coverage: §2.4 linearity (Task 1); §4.0 matrix step (Task 2), parallel
   scan (Task 6), mesh-as-data (Tasks 4-5), error monitor (Task 7), forward
   sensitivities (Task 8), C_l end to end (Task 9); §5 Phase 0 items 5-6
   benchmark (Tasks 10-11). Phase 1/2 approximation and preset work is
   explicitly out of scope.
2. Placeholder scan: none (`CLAX_PYTHON` is an environment variable with a
   default, not a placeholder).
3. Type consistency: `make_args(k)` tuple order matches `ode_args` in
   `_perturbations_solve_impl`; `(ys, errs)` shapes `(N, nk, d)` used
   identically in Tasks 6, 7, 9, 10; `record_mesh` returns `(ts, ys)` in
   Tasks 4, 5, 9, 10.
4. Review Focus items 1-4 are pinned to tests in Tasks 3, 4, 6, 7; item 5 to
   gate G4 in Task 11.
```
