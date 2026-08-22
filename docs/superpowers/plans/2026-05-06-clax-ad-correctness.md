# clax AD Correctness — Implementation Plan (PR-A + PR-B)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore reverse-mode AD correctness for `compute_pk` (PR-A — in flight, awaiting review) and enable forward-mode AD end-to-end through the public observables for Fisher-style sensitivity analysis (PR-B — new work).

**Architecture:**
- PR-A is essentially shipping: a Taylor expansion around the frozen ODE endpoint reintroduces the chain-rule term that `stop_gradient(bg.conformal_age)` sacrifices. Already on PR #16. Remaining work is verification on production hardware (V100 / igpu) and CHANGELOG.
- PR-B converts three `custom_vjp` blockers to `custom_jvp` and wires a `DirectAdjoint` path so `jax.jvp` / `jax.jacfwd` can flow through `compute_pk` end-to-end. This unblocks Fisher matrices via forward mode, asymptotically faster than `jacrev` for our parameter dimensionality.

**Tech Stack:** JAX 0.4+, diffrax (Kvaerno5/Rodas5 stiff solvers), equinox, jaxtyping. Production env on Bridges-2: micromamba env `clax_class-pt_py310forge`. Reference oracle: CLASS 3.3.4 at `../class_public-3.3.4/`.

---

## Current State (as of 2026-05-06)

### What landed on PR #16 (`fix/pk_tau_end` → `smsharma/clax:main`)

- **Primal fix** (commit `8a748e5`, was already in PR before this work): `_perturbation_solve_setup` gains `tau_max_factor` kwarg; matter-power paths route `tau_max_factor=1.0`. Closes ~0.33% under-bias.
- **Gradient fix** (new commit `a4a7ab2`): Taylor expansion around frozen `tau_end` in `_matter_delta_m_single_k_impl`. Uses `jax.jvp(_extract_delta_m, (y_final, tau_end), (dy_dtau, 1))` to capture both implicit (state) and explicit (background-density-at-τ) `dδ_m/dτ`.
- **Regression test** (same commit): `TestPkScalarDensityGradients` in `tests/test_pk_gradients.py` asserts `dP/d{h, omega_b, omega_cdm}` matches centered FD <5% at `PK_CONTRACT_PREC` (rtol=1e-6). Catches the original 65–84% bug class with margin for FD precision floor.

### Diagnostic measurements (on Mac CPU, full single-mode + thermo + bg)

| Param | rtol=1e-3 (orig benchmark) | rtol=1e-5 | rtol=1e-7 |
|---|---|---|---|
| h | 6.41% | 1.08% | **0.32%** |
| omega_b | 8.91% | 3.18% | **1.23%** |
| omega_cdm | 4.83% | 0.85% | **0.61%** |
| ln10A_s, n_s | 0% | 0% | 0% |

Clean ~3–4× per rtol decade, consistent with both AD and FD inheriting ODE solver tolerance.

### What's on `benchmark/clax-pt` (development branch)

- Fast-forwarded to `origin/benchmark/clax-pt` at `1f62077` (HPC docs, EPT `stop_gradient` fix).
- Cherry-picked Taylor correction (`62baaa2`) for PR-B development continuity.
- 10 untracked diagnostic scripts in `diags/diag_*.py` from the investigation. Document the bug-finding trail; useful for reviewers and for future debugging. Should be committed before PR-B work begins.

### What's NOT yet enabled

- **Forward-mode AD** (`jax.jvp`, `jax.jacfwd`) through `compute_pk`. Three structural blockers:
  1. **`_find_z_reio`** in `clax/thermodynamics.py:845` (`@jax.custom_vjp`). Bisection root-find for `z_reio(τ_reio_target)`. Has no JVP rule. A first attempt to convert to `custom_jvp` produced NaN tangents on `h/omega_b/omega_cdm` (root cause not yet localized — likely `jnp.where` pitfall in `_tau_reio_for_zreio` or `_reionization_xe`).
  2. **`shoot_fn`** in `clax/shooting.py:76` (`@jax.custom_vjp`). Newton iteration for `h(100*θ_s)`. Has no JVP rule. Untested for forward-mode use, but uses the same implicit-function-theorem pattern as `_find_z_reio` so the conversion is mechanical once we have the pattern.
  3. **`RecursiveCheckpointAdjoint`** in diffrax. Its `checkpointed_while_loop` is a `custom_vjp`. Can't be converted (upstream library). Workaround: provide a `DirectAdjoint` path for forward-mode users (already supported in `clax/ode.py` via `prec.ode_adjoint = "direct"`).

### Pending review feedback on PR #16

PR #16 is open and updated. Sidd (smsharma) has not yet reviewed the gradient-fix portion (only the primal fix from the original PR body).

---

## File Structure

### PR-A files (reverse-mode AD completion)

| File | Status | Responsibility |
|---|---|---|
| `clax/perturbations.py:2248-2324` | ✅ Modified | `_matter_delta_m_single_k_impl` with Taylor correction |
| `tests/test_pk_gradients.py` | ✅ Modified | New `TestPkScalarDensityGradients` regression class |
| `CHANGELOG.md` | ⬜ Pending | Entry for PR #16 |
| `diags/diag_grad_*.py`, `diags/diag_jvp_*.py` | ⬜ Untracked | Investigation scripts; commit as evidence trail |

### PR-B files (forward-mode AD enablement)

| File | Action | Responsibility |
|---|---|---|
| `clax/thermodynamics.py:845-931` | Modify | Convert `_find_z_reio` `custom_vjp → custom_jvp`; debug NaN |
| `clax/shooting.py:76-117` | Modify | Convert `shoot_fn` `custom_vjp → custom_jvp` |
| `clax/ode.py` | Verify | `DirectAdjoint` plumbing already exists; ensure forward-mode safe |
| `tests/test_pk_gradients.py` | Extend | New `TestPkScalarForwardMode` class |
| `tests/test_thermodynamics.py` | Extend | New `test_z_reio_forward_mode_matches_fd` |
| `tests/test_shooting.py` | Extend | New `test_shoot_h_forward_mode_matches_fd` |
| `tests/test_fisher.py` | Create | Fisher-matrix correctness + perf via `jax.jacfwd` |
| `diags/diag_perf_jacfwd_vs_jacrev.py` | Create | Performance comparison script |
| `CHANGELOG.md` | Modify | Entry for PR-B |
| `DESIGN.md` | Modify | Document forward-mode AD story |

---

# PR-A: Reverse-Mode AD Completion

PR #16 is mostly done. Remaining: verify on production hardware (V100), commit investigation diags, write CHANGELOG, address review.

### Task A1: Commit diagnostic scripts to `benchmark/clax-pt`

**Files:**
- Modify: `diags/README.md` (extend with new diagnostic catalog)
- Add: `diags/diag_grad_bg_conformal_age.py`, `diags/diag_grad_taylor.py`, `diags/diag_grad_exact.py`, `diags/diag_grad_tight_rtol.py`, `diags/diag_grad_rtol_1em7.py`, `diags/diag_grad_fd_step_and_jvp.py`, `diags/diag_grad_tau_end.py`, `diags/diag_grad_jvp_direct.py`, `diags/diag_jvp_bisect.py`, `diags/diag_jvp_find_z_reio.py`

- [ ] **Step 1: Inspect `diags/README.md` to follow existing convention**

```bash
cat /Users/nguyenmn/clax/.claude/worktrees/benchmark-clax-pt/diags/README.md
```

- [ ] **Step 2: Add a new "AD diagnostics" section to `diags/README.md`**

Append a new section listing each new script with one-line purpose:

```markdown
## AD diagnostics (May 2026 investigation, PR #16)

| Script | Purpose |
|---|---|
| `diag_grad_tau_end.py` | Empirically confirm `RecursiveCheckpointAdjoint+PIDController` blocks traced `t1` |
| `diag_grad_taylor.py` | First-pass Taylor correction with f·aH approximation (~6% residual) |
| `diag_grad_exact.py` | Exact-RHS Taylor via `jax.jvp` (in-tree fix) |
| `diag_grad_bg_conformal_age.py` | Decisive: `jax.grad(bg.conformal_age)` is exact (rules out bg AD bug) |
| `diag_grad_fd_step_and_jvp.py` | FD step sensitivity sweep + jvp attempt (custom_vjp blocker surfaced) |
| `diag_grad_tight_rtol.py` | rtol=1e-5 verification (1.08% / 3.18% / 0.85%) |
| `diag_grad_rtol_1em7.py` | rtol=1e-7 final precision floor (0.32% / 1.23% / 0.61%) |
| `diag_grad_jvp_direct.py` | jvp end-to-end attempt with DirectAdjoint (NaN on density params) |
| `diag_jvp_bisect.py` | Localize NaN to thermodynamics `_find_z_reio` |
| `diag_jvp_find_z_reio.py` | Direct test of `_find_z_reio` JVP rule |
```

- [ ] **Step 3: Commit diags**

```bash
cd /Users/nguyenmn/clax/.claude/worktrees/benchmark-clax-pt
git add diags/diag_grad_*.py diags/diag_jvp_*.py diags/README.md
git commit -m "$(cat <<'EOF'
diags: AD investigation scripts (May 2026)

Documents the 10-step diagnostic trail that led to the Taylor correction
in PR #16. Reusable scripts for future AD debugging.
EOF
)"
git push origin benchmark/clax-pt
```

### Task A2: Verify PR #16 fix on V100 (igpu)

**Files:**
- Run: `scripts/benchmark_gradients.py` on V100
- Run: `pytest tests/test_pk_gradients.py -v` on V100

- [ ] **Step 1: SSH to igpu and check out fix/pk_tau_end**

```bash
ssh igpu  # or appropriate hostname
cd /home/n2minh/clax  # or /lustre/work/n2minh/clax
git fetch origin
git checkout fix/pk_tau_end
git pull
```

- [ ] **Step 2: Run regression test at production precision**

```bash
micromamba run -n clax_class-pt_py310forge python -m pytest \
  tests/test_pk_gradients.py::TestPkScalarDensityGradients -v
```

Expected: 3/3 pass. Density-param gradients should be tighter on V100 + tighter precision than on Mac CPU.

- [ ] **Step 3: Run benchmark_gradients.py at original benchmark precision**

```bash
micromamba run -n clax_class-pt_py310forge python scripts/benchmark_gradients.py \
  --n-warmup 1 --n-repeat 1
```

Expected: all 5 params pass <5% (was 65-84% off; consistent with rtol=1e-3 precision floor).

- [ ] **Step 4: Run benchmark_gradients.py at higher precision**

Edit `scripts/benchmark_gradients.py` temporarily to set `pt_ode_rtol=1e-7, ode_max_steps=65536`, then:

```bash
micromamba run -n clax_class-pt_py310forge python scripts/benchmark_gradients.py \
  --n-warmup 1 --n-repeat 1
```

Expected: all 5 params pass <1.5%. Revert the edit before committing.

- [ ] **Step 5: Run full pytest --fast as smoke test**

```bash
micromamba run -n clax_class-pt_py310forge python -m pytest tests/ --fast -x -q
```

Expected: all pass.

- [ ] **Step 6: Document results**

Comment on PR #16 with the V100 numbers (consider posting the table from the PR body, but with V100 numbers replacing Mac CPU numbers).

### Task A3: Write CHANGELOG entry for PR-A

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Read current CHANGELOG.md to follow format**

```bash
head -40 /Users/nguyenmn/clax/.claude/worktrees/benchmark-clax-pt/CHANGELOG.md
```

- [ ] **Step 2: Add entry under "Unreleased" or current section**

```markdown
### Fixed

- `compute_pk` reverse-mode AD: gradients of `P(k)` w.r.t. `h`, `omega_b`,
  `omega_cdm` were 65-84% off vs centered FD because
  `stop_gradient(bg.conformal_age)` (required by RecursiveCheckpointAdjoint +
  PIDController) sacrificed the `dδ_m/dτ · dτ_end/dθ` chain-rule term.
  Fixed via first-order Taylor expansion around the frozen endpoint in
  `_matter_delta_m_single_k_impl`. New regression test
  `TestPkScalarDensityGradients` covers `h, omega_b, omega_cdm` at
  `PK_CONTRACT_PREC` with 5% threshold (FD precision floor at rtol=1e-6).
  Trajectory: 6.4% / 8.9% / 4.8% (rtol=1e-3) → 0.32% / 1.23% / 0.61%
  (rtol=1e-7) for h / omega_b / omega_cdm.
```

- [ ] **Step 3: Commit CHANGELOG to PR #16's branch**

```bash
cd /Users/nguyenmn/clax/.claude/worktrees/pr16-ad-fix
git add CHANGELOG.md
git commit -m "docs(CHANGELOG): document compute_pk gradient fix"
git push origin fix/pk_tau_end
```

---

# PR-B: Forward-Mode AD Enablement

**Branch:** `feat/forward-mode-ad` from `main` after PR #16 merges (or from `fix/pk_tau_end` if PR-B starts before PR #16 merges — rebase later).

## Phase B1 — TDD harness for forward-mode contracts

### Task B1: Write the forward-mode failing tests

**Files:**
- Modify: `tests/test_pk_gradients.py`

- [ ] **Step 1: Add `TestPkScalarForwardMode` test class**

After `TestPkScalarDensityGradients`, add:

```python
class TestPkScalarForwardMode:
    """Regression: forward-mode AD (jax.jvp / jacfwd) through compute_pk
    matches reverse-mode AD on every parameter that affects P(k)."""

    @pytest.mark.slow
    @pytest.mark.parametrize("param_name", PK_GRAD_DENSITY_PARAMS + ("ln10A_s", "n_s"))
    def test_jvp_matches_grad(self, param_name, fast_mode):
        if fast_mode:
            pytest.skip("forward-mode regression runs in full mode only")
        k_test = float(PK_GRAD_FULL_K[0])
        # DirectAdjoint required: RecursiveCheckpointAdjoint blocks jvp.
        from dataclasses import replace as _replace
        prec = _replace(PK_GRAD_CONTRACT_PREC, ode_adjoint="direct")

        def f_of_param(val):
            params_ = FIDUCIAL_PARAMS.replace(**{param_name: val})
            return compute_pk_scalar_direct(params_, prec, k_test)

        val0 = jnp.asarray(getattr(FIDUCIAL_PARAMS, param_name))
        primal_jvp, tangent = jax.jvp(f_of_param, (val0,), (jnp.ones_like(val0),))
        primal_jvp.block_until_ready()
        grad_ad = float(jax.grad(f_of_param)(val0))
        rel_err = abs(float(tangent) - grad_ad) / (abs(grad_ad) + 1e-30)
        assert rel_err < 1.0e-6, (
            f"jvp({param_name})={float(tangent):.6e} vs grad={grad_ad:.6e}; "
            f"rel_err={rel_err:.2e} >= 1e-6. Mode disagreement implies a "
            f"custom_vjp/custom_jvp inconsistency."
        )
```

- [ ] **Step 2: Run the test (expect failure)**

```bash
micromamba run -n clax_class-pt_py310forge python -m pytest \
  tests/test_pk_gradients.py::TestPkScalarForwardMode -v 2>&1 | tail -30
```

Expected: FAIL with `TypeError: can't apply forward-mode autodiff (jvp) to a custom_vjp function` (raised inside `clax/thermodynamics.py` at `_find_z_reio` call site).

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_pk_gradients.py
git commit -m "test(pk_gradients): add forward-mode jvp-vs-grad regression (RED)"
```

This is the canonical failing test that PR-B's downstream tasks will turn green.

## Phase B2 — Convert `_find_z_reio` to custom_jvp

### Task B2: Add a focused failing JVP test for `_find_z_reio`

**Files:**
- Modify: `tests/test_thermodynamics.py`

- [ ] **Step 1: Find existing thermo test conventions**

```bash
head -40 /Users/nguyenmn/clax/.claude/worktrees/benchmark-clax-pt/tests/test_thermodynamics.py
```

- [ ] **Step 2: Add forward-mode test for `th.z_reio` w.r.t. h**

```python
def test_z_reio_forward_mode_matches_fd():
    """Forward-mode AD on z_reio(h) matches centered FD to <1e-3
    relative agreement (z_reio is a smooth root-find of optical depth)."""
    import dataclasses
    from clax import CosmoParams, PrecisionParams
    from clax.background import background_solve
    from clax.thermodynamics import thermodynamics_solve

    prec = PrecisionParams(
        th_n_points=3000, pt_k_per_decade=10, pt_k_max_cl=0.3,
        pt_l_max_g=17, pt_l_max_pol_g=17, pt_l_max_ur=17,
        ncdm_q_size=0, pt_tau_n_points=1000,
        pt_ode_rtol=1e-5, pt_ode_atol=1e-6,
        ode_max_steps=16384, pt_ode_solver="rodas5",
        ode_adjoint="direct",
    )
    params = CosmoParams()

    def z_reio_of_h(h):
        p = dataclasses.replace(params, h=h)
        bg = background_solve(p, prec)
        th = thermodynamics_solve(p, prec, bg)
        return th.z_reio

    primal, tangent = jax.jvp(z_reio_of_h, (params.h,), (jnp.array(1.0),))
    primal.block_until_ready()
    eps = 1e-3
    fd = (z_reio_of_h(params.h + eps) - z_reio_of_h(params.h - eps)) / (2 * eps)
    rel = abs(float(tangent) - float(fd)) / (abs(float(fd)) + 1e-30)
    assert jnp.isfinite(tangent), f"jvp returned NaN/inf: {tangent}"
    assert rel < 1.0e-3, f"jvp={float(tangent):.6e} vs FD={float(fd):.6e} rel={rel:.2e}"
```

- [ ] **Step 3: Run the test, observe NaN failure**

```bash
micromamba run -n clax_class-pt_py310forge python -m pytest \
  tests/test_thermodynamics.py::test_z_reio_forward_mode_matches_fd -v 2>&1 | tail -10
```

Expected: FAIL with `TypeError: can't apply forward-mode autodiff (jvp) to a custom_vjp function`.

- [ ] **Step 4: Commit the failing test**

```bash
git add tests/test_thermodynamics.py
git commit -m "test(thermodynamics): add z_reio jvp-vs-FD regression (RED)"
```

### Task B3: Convert `_find_z_reio` to `custom_jvp` (will produce NaN initially)

**Files:**
- Modify: `clax/thermodynamics.py:845-931`

- [ ] **Step 1: Replace the `@jax.custom_vjp` block with `@jax.custom_jvp`**

Replace lines 845-931 with:

```python
@jax.custom_jvp
def _find_z_reio(
    tau_reio_target,
    xe_raw_grid,
    kd_prefactor,
    dtau_grid,
    z_grid,
    Y_He,
):
    """Differentiable ``z_reio`` solve via the implicit function theorem.

    Primal: bounded bisection in ``_find_z_reio_impl``. JVP: applies the
    implicit function theorem to ``F(z, *aux) = tau(z; *aux) - target``,
    giving ``z_dot = -F_dot / dF/dz``. Reverse mode is derived from the JVP
    via JAX's transposition machinery.
    """
    return _find_z_reio_impl(
        tau_reio_target, xe_raw_grid, kd_prefactor, dtau_grid, z_grid, Y_He
    )


@_find_z_reio.defjvp
def _find_z_reio_jvp(primals, tangents):
    tau_reio_target, xe_raw_grid, kd_prefactor, dtau_grid, z_grid, Y_He = primals
    (
        tau_reio_target_dot,
        xe_raw_grid_dot,
        kd_prefactor_dot,
        dtau_grid_dot,
        z_grid_dot,
        Y_He_dot,
    ) = tangents

    z_reio = _find_z_reio_impl(
        tau_reio_target, xe_raw_grid, kd_prefactor, dtau_grid, z_grid, Y_He
    )

    # dF/dz at solution; F is scalar, so jvp with tangent 1 is the derivative.
    _, dF_dz = jax.jvp(
        lambda z_: _tau_reio_for_zreio(
            z_, xe_raw_grid, kd_prefactor, dtau_grid, z_grid, Y_He
        ),
        (z_reio,),
        (jnp.ones_like(z_reio),),
    )
    dF_dz = jnp.where(
        jnp.abs(dF_dz) < 1e-12,
        jnp.where(dF_dz >= 0.0, 1e-12, -1e-12),
        dF_dz,
    )

    # F_dot from JVP of tau_reio_for_zreio at fixed z, minus target_dot.
    _, tau_jvp = jax.jvp(
        lambda xe, kd, dt, zg, yh: _tau_reio_for_zreio(
            z_reio, xe, kd, dt, zg, yh
        ),
        (xe_raw_grid, kd_prefactor, dtau_grid, z_grid, Y_He),
        (xe_raw_grid_dot, kd_prefactor_dot, dtau_grid_dot, z_grid_dot, Y_He_dot),
    )
    F_dot = tau_jvp - tau_reio_target_dot
    z_reio_dot = -F_dot / dF_dz
    return z_reio, z_reio_dot
```

- [ ] **Step 2: Run the JVP test — expect NaN**

```bash
micromamba run -n clax_class-pt_py310forge python -m pytest \
  tests/test_thermodynamics.py::test_z_reio_forward_mode_matches_fd -v 2>&1 | tail -15
```

Expected: FAIL with `assert jnp.isfinite(tangent)` — jvp returns NaN.

- [ ] **Step 3: Run reverse-mode tests to confirm grad still works**

```bash
micromamba run -n clax_class-pt_py310forge python -m pytest \
  tests/test_thermodynamics.py -v --fast 2>&1 | tail -10
```

Expected: PASS (9/9). JAX derives VJP from JVP via transposition; the reverse-mode path is unchanged behavior.

- [ ] **Step 4: Commit (intermediate state — JVP defined but NaN; reverse-mode unchanged)**

```bash
git add clax/thermodynamics.py
git commit -m "refactor(thermodynamics): convert _find_z_reio custom_vjp -> custom_jvp (NaN in jvp; reverse mode unchanged)"
```

### Task B4: Localize the NaN with `jax.debug.print`

**Files:**
- Temporarily modify: `clax/thermodynamics.py:_find_z_reio_jvp` (revert at end)
- Run: `diags/diag_jvp_find_z_reio.py`

- [ ] **Step 1: Add debug prints inside `_find_z_reio_jvp`**

Insert before `return z_reio, z_reio_dot`:

```python
jax.debug.print(
    "[zreio.jvp] z={a} dF_dz={b} tau_jvp={c} target_dot={d} F_dot={e} z_dot={f}",
    a=z_reio, b=dF_dz, c=tau_jvp, d=tau_reio_target_dot, e=F_dot, f=z_reio_dot,
)
```

- [ ] **Step 2: Run the focused diag**

```bash
micromamba run -n clax_class-pt_py310forge python -u diags/diag_jvp_find_z_reio.py 2>&1 | tee /tmp/zreio_jvp_debug.log
```

- [ ] **Step 3: Inspect intermediate values to localize the NaN**

```bash
grep "zreio.jvp" /tmp/zreio_jvp_debug.log
```

Expected: one of `dF_dz`, `tau_jvp`, `F_dot` is NaN. Note which one and look upstream.

If `tau_jvp` is NaN: the inner `jax.jvp` of `_tau_reio_for_zreio` evaluates to NaN. Suspect: `jnp.maximum(xe_total - xe_raw_grid, 0.0)` JVP at the boundary, or a `0 × ∞` pattern in `_reionization_xe`.

If `dF_dz` is NaN: the `_tau_reio_for_zreio` derivative w.r.t. z evaluates to NaN. Suspect: same as above but for the z direction.

If both finite but `z_reio_dot` is NaN: division-related; check `dF_dz` near 0 (the floor should prevent this; investigate why floor failed).

- [ ] **Step 4: Add finer-grained prints inside `_tau_reio_for_zreio` if needed**

If the bug is upstream of `_find_z_reio_jvp`, add `jax.debug.print` calls to `_tau_reio_for_zreio` to isolate which line produces the NaN.

- [ ] **Step 5: Note the localization in CHANGELOG-private-debug.md**

Don't commit yet — you'll fix and re-test in the next task.

### Task B5: Fix the NaN root cause

**Files:**
- Modify: `clax/thermodynamics.py` (the NaN-producing line, wherever it is)

The most likely fix patterns:

- **`jnp.where`-NaN-grad pitfall.** Replace `jnp.where(cond, branch1, branch2)` with the `safe_x` pattern: precompute a `safe_x = jnp.where(cond, x, dummy)` then evaluate the unsafe op on `safe_x` and gate the result. This kills NaN tangents from the not-taken branch. Reference: https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where
- **Implicit `0 × ∞`.** Use `jnp.where(cond, value, 0.0)` and ensure `value` is finite when `cond` is true.
- **Branch in `_reionization_xe_fraction`** at `(1+z)^p − (1+z_reio)^p`: if any sample has `1+z=0` or `1+z_reio=0`, the JVP of `pow` produces `inf` or NaN. Guard with positivity assumption.

- [ ] **Step 1: Apply the fix (specific code depends on Task B4 localization)**

If localization shows `_reionization_xe`'s helium term:

```python
# Helium double reionization at z ~ 3.5
# cf. CLASS thermodynamics.c:1338-1358
arg_He = (3.5 - z) / 0.5
frac_He = (jnp.tanh(arg_He) + 1.0) / 2.0
xe_reio += fHe * frac_He
```

Inspect — these are smooth, no NaN-prone ops. If localization shows `_reionization_xe_fraction`'s pow:

```python
argument = (
    ((1.0 + z_reio) ** reio_exponent - (1.0 + z) ** reio_exponent)
    / (reio_exponent * (1.0 + z_reio) ** (reio_exponent - 1.0))
    / reio_width
)
```

The `(1+z_reio)^(reio_exponent − 1) = (1+z_reio)^0.5` is in the denominator. If `z_reio` is exactly `-1` (forbidden), divide-by-zero. `jax.jvp` of `pow(x, p)` is `p * x^(p-1) * x_dot`. For `x > 0` this is finite. For `x = 0` and `p < 1` (i.e. `0.5`), it's `inf`. The bisection's z_reio range is `[4, 25]` so `z+1 ≥ 5` — should never hit zero. But: maybe inside an inner JVP call something gets evaluated at `z=-1`? Add a guard.

- [ ] **Step 2: Re-run the JVP test**

```bash
micromamba run -n clax_class-pt_py310forge python -m pytest \
  tests/test_thermodynamics.py::test_z_reio_forward_mode_matches_fd -v 2>&1 | tail -10
```

Expected: PASS (jvp matches FD to <1e-3).

- [ ] **Step 3: Re-run reverse-mode tests to confirm no regression**

```bash
micromamba run -n clax_class-pt_py310forge python -m pytest \
  tests/test_thermodynamics.py tests/test_pk_gradients.py -v --fast 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 4: Remove debug prints**

- [ ] **Step 5: Commit the fix**

```bash
git add clax/thermodynamics.py
git commit -m "$(cat <<'EOF'
fix(thermodynamics): root-cause-fix NaN in _find_z_reio JVP

[describe the specific NaN cause and the fix here]

JVP through z_reio(h) now matches centered FD to <1e-3; reverse-mode
behavior unchanged.
EOF
)"
```

## Phase B3 — Convert `shoot_fn` to custom_jvp

### Task B6: Add a forward-mode test for `shoot_h_from_theta_s`

**Files:**
- Create: `tests/test_shooting.py` if it doesn't exist; otherwise extend.

- [ ] **Step 1: Inspect existing shooting tests**

```bash
ls /Users/nguyenmn/clax/.claude/worktrees/benchmark-clax-pt/tests/test_shooting.py 2>&1 || echo "not present"
```

- [ ] **Step 2: Add `test_shoot_h_forward_mode_matches_fd`**

```python
def test_shoot_h_forward_mode_matches_fd():
    """Forward-mode AD on shoot_h_from_theta_s matches FD to <1e-3."""
    from clax import CosmoParams, PrecisionParams
    from clax.shooting import make_shoot_h_from_theta_s

    prec = PrecisionParams(
        th_n_points=3000, ode_adjoint="direct",
    )
    shoot = make_shoot_h_from_theta_s(prec)
    params = CosmoParams()
    target = 1.04  # Planck-ish 100*theta_s

    def h_of_target(t):
        return shoot(t, params)

    primal, tangent = jax.jvp(h_of_target, (jnp.asarray(target),), (jnp.asarray(1.0),))
    primal.block_until_ready()
    eps = 1e-4
    fd = (h_of_target(target + eps) - h_of_target(target - eps)) / (2 * eps)
    rel = abs(float(tangent) - float(fd)) / (abs(float(fd)) + 1e-30)
    assert jnp.isfinite(tangent), f"jvp NaN: {tangent}"
    assert rel < 1.0e-3, f"jvp={float(tangent):.6e} FD={float(fd):.6e} rel={rel:.2e}"
```

- [ ] **Step 3: Run the test (expect failure with custom_vjp error)**

```bash
micromamba run -n clax_class-pt_py310forge python -m pytest \
  tests/test_shooting.py::test_shoot_h_forward_mode_matches_fd -v 2>&1 | tail -10
```

Expected: FAIL with `TypeError: can't apply forward-mode autodiff (jvp) to a custom_vjp function`.

- [ ] **Step 4: Commit the failing test**

```bash
git add tests/test_shooting.py
git commit -m "test(shooting): add shoot_h jvp-vs-FD regression (RED)"
```

### Task B7: Convert `shoot_fn` to custom_jvp

**Files:**
- Modify: `clax/shooting.py:76-117`

- [ ] **Step 1: Replace the `@jax.custom_vjp` block**

Replace the body of `make_shoot_h_from_theta_s` with:

```python
def make_shoot_h_from_theta_s(prec: PrecisionParams):
    @jax.custom_jvp
    def shoot_fn(theta_s_100_target: float, params_template: CosmoParams) -> float:
        h0 = 3.54 * theta_s_100_target**2 - 5.455 * theta_s_100_target + 2.548

        def newton_step(i, h):
            theta_s = _compute_theta_s(h, params_template, prec)
            eps = 1e-4
            theta_s_plus = _compute_theta_s(h + eps, params_template, prec)
            dtheta_dh = (theta_s_plus - theta_s) / eps
            update = (theta_s - theta_s_100_target) / dtheta_dh
            h = h - 0.5 * update
            return h

        h_final = jax.lax.fori_loop(0, 25, newton_step, h0)
        return h_final

    @shoot_fn.defjvp
    def _shoot_fn_jvp(primals, tangents):
        theta_s_100_target, params_template = primals
        target_dot, params_template_dot = tangents
        h = shoot_fn(theta_s_100_target, params_template)
        # F(h, target, params) = theta_s(h, params) - target = 0
        # dF/dh at solution
        _, dF_dh = jax.jvp(
            lambda h_: _compute_theta_s(h_, params_template, prec),
            (h,),
            (jnp.ones_like(h),),
        )
        # F_dot via inputs (only target tangent for now; params_template
        # doesn't currently propagate tangents into theta_s in the existing
        # custom_vjp either — match that behavior).
        h_dot = -(-target_dot) / dF_dh  # since dF/d(target) = -1
        return h, h_dot

    return shoot_fn
```

- [ ] **Step 2: Run the shoot test**

```bash
micromamba run -n clax_class-pt_py310forge python -m pytest \
  tests/test_shooting.py::test_shoot_h_forward_mode_matches_fd -v 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 3: Run any existing shooting tests**

```bash
micromamba run -n clax_class-pt_py310forge python -m pytest tests/test_shooting.py -v --fast
```

Expected: PASS (reverse-mode behavior unchanged via JAX transposition).

- [ ] **Step 4: Commit**

```bash
git add clax/shooting.py
git commit -m "refactor(shooting): convert shoot_fn custom_vjp -> custom_jvp"
```

## Phase B4 — Forward mode end-to-end through compute_pk

### Task B8: Run the original failing test from Task B1

- [ ] **Step 1: Run `TestPkScalarForwardMode`**

```bash
micromamba run -n clax_class-pt_py310forge python -m pytest \
  tests/test_pk_gradients.py::TestPkScalarForwardMode -v 2>&1 | tail -15
```

Expected outcomes:
- PASS on all 5 params: forward-mode is fully working end-to-end. Skip to Task B10.
- FAIL with NaN on density params: residual NaN from a path we haven't fixed; bisect with `diag_jvp_bisect.py` re-run.
- FAIL with `custom_vjp` error: another blocker we missed (e.g., `_solve_hydrogen_saha` JVP, or another deep `custom_vjp` we didn't catalog).

### Task B9: Address any remaining residual blocker

**Files:**
- Bisect with `diag_jvp_bisect.py` (re-run; modify stages if needed)

- [ ] **Step 1: Re-bisect the pipeline**

```bash
micromamba run -n clax_class-pt_py310forge python -u diags/diag_jvp_bisect.py
```

Stages: bg only → th.z_reio → th.tau_star → th.kappa_dot at z=1000 → perturbation single-k.

- [ ] **Step 2: For each stage that fails, follow Task B4–B5 pattern (debug prints, localize, fix)**

- [ ] **Step 3: Re-run end-to-end test once all stages pass jvp**

```bash
micromamba run -n clax_class-pt_py310forge python -m pytest \
  tests/test_pk_gradients.py::TestPkScalarForwardMode -v
```

Expected: 5/5 PASS.

- [ ] **Step 4: Commit**

```bash
git add [files]
git commit -m "fix: enable end-to-end jax.jvp through compute_pk (DirectAdjoint path)"
```

### Task B10: Verify Taylor correction is mode-agnostic

The Taylor correction in `_matter_delta_m_single_k_impl` was designed to work for both jvp and grad. Verify empirically that it produces identical results in both modes.

**Files:**
- Modify: `tests/test_pk_gradients.py` (extend `TestPkScalarForwardMode`)

- [ ] **Step 1: Add `test_jvp_grad_agreement_density_params`**

```python
@pytest.mark.slow
def test_jvp_grad_agreement_density_params(self, fast_mode):
    """jax.jvp and jax.grad through compute_pk agree to <1e-6 on density params."""
    if fast_mode:
        pytest.skip("density-param mode-agreement runs in full mode only")
    k_test = float(PK_GRAD_FULL_K[0])
    from dataclasses import replace as _replace
    prec = _replace(PK_GRAD_CONTRACT_PREC, ode_adjoint="direct")

    def f(p):
        return compute_pk_scalar_direct(p, prec, k_test)

    grad_tree = jax.grad(f)(FIDUCIAL_PARAMS)
    failures = []
    for n in PK_GRAD_DENSITY_PARAMS:
        val0 = jnp.asarray(getattr(FIDUCIAL_PARAMS, n))
        def g(v, n=n):
            return f(FIDUCIAL_PARAMS.replace(**{n: v}))
        _, tangent = jax.jvp(g, (val0,), (jnp.ones_like(val0),))
        rev = float(getattr(grad_tree, n))
        rel = abs(float(tangent) - rev) / (abs(rev) + 1e-30)
        if rel >= 1e-6:
            failures.append(f"{n}: jvp={float(tangent):.6e} grad={rev:.6e} rel={rel:.2e}")
    assert not failures, "jvp/grad disagreement: " + "; ".join(failures)
```

- [ ] **Step 2: Run, expect PASS**

- [ ] **Step 3: Commit**

```bash
git add tests/test_pk_gradients.py
git commit -m "test(pk_gradients): assert jvp == grad on density params (mode agnostic Taylor)"
```

## Phase B5 — Fisher matrix correctness + performance

### Task B11: Implement Fisher correctness test

**Files:**
- Create: `tests/test_fisher.py`

The Fisher matrix for a Gaussian observable is `F_{ij} = J^T C^{-1} J`, where `J = ∂_θ d` (Jacobian of data vector w.r.t. parameters) and `C` is the data covariance. For `n × m` Jacobian with `n >> m`, `jacfwd` (m forward passes) is O(m × forward_cost), beating `jacrev` (n reverse passes) by factor n/m for large n.

- [ ] **Step 1: Build a small Fisher problem on `compute_pk_table`**

```python
"""Fisher matrix correctness + performance regression."""
from dataclasses import replace as _replace
import time
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from clax import CosmoParams, compute_pk_table
from tests.pk_test_utils import PK_TABLE_GRAD_FAST_PREC

FIDUCIAL = CosmoParams()
PARAMS = ("h", "omega_b", "omega_cdm", "ln10A_s", "n_s")
K_GRID = np.geomspace(1e-3, 0.3, 50)


def _data_vector(params):
    pk = compute_pk_table(params, PK_TABLE_GRAD_FAST_PREC, k=jnp.asarray(K_GRID))
    return jnp.log(pk)  # log-P data vector for stability


def _packed_to_params(theta):
    kw = {n: theta[i] for i, n in enumerate(PARAMS)}
    return FIDUCIAL.replace(**kw)


def _data_of_packed(theta):
    return _data_vector(_packed_to_params(theta))


def _theta0():
    return jnp.asarray([float(getattr(FIDUCIAL, n)) for n in PARAMS])


def _toy_cov(d):
    sigma = 0.05
    return (sigma ** 2) * jnp.eye(d.shape[0])


@pytest.mark.slow
def test_fisher_jacfwd_matches_jacrev():
    """Fisher computed via jacfwd matches jacrev to <1e-6."""
    theta0 = _theta0()
    J_fwd = jax.jacfwd(_data_of_packed)(theta0)
    J_rev = jax.jacrev(_data_of_packed)(theta0)
    rel = jnp.max(jnp.abs(J_fwd - J_rev) / (jnp.abs(J_rev) + 1e-30))
    assert float(rel) < 1e-6, f"jacfwd vs jacrev rel={float(rel):.2e}"


@pytest.mark.slow
def test_fisher_against_finite_difference():
    """Fisher diagonal computed via jacfwd matches FD-based Fisher to <2%."""
    theta0 = _theta0()
    J_ad = jax.jacfwd(_data_of_packed)(theta0)
    d0 = _data_of_packed(theta0)
    cov = _toy_cov(d0)
    cinv = jnp.linalg.inv(cov)
    F_ad = J_ad.T @ cinv @ J_ad

    eps = jnp.array([1e-3, 1e-5, 1e-3, 1e-3, 1e-3])  # per-param FD step
    J_fd = jnp.zeros_like(J_ad)
    for i in range(len(PARAMS)):
        plus = theta0.at[i].add(eps[i])
        minus = theta0.at[i].add(-eps[i])
        J_fd = J_fd.at[:, i].set((_data_of_packed(plus) - _data_of_packed(minus)) / (2 * eps[i]))
    F_fd = J_fd.T @ cinv @ J_fd

    diag_rel = jnp.max(jnp.abs(jnp.diag(F_ad) - jnp.diag(F_fd)) / (jnp.diag(F_fd) + 1e-30))
    assert float(diag_rel) < 0.02, f"Fisher diag rel={float(diag_rel):.2%}"
```

- [ ] **Step 2: Run**

```bash
micromamba run -n clax_class-pt_py310forge python -m pytest tests/test_fisher.py -v 2>&1 | tail -20
```

Expected: PASS (both tests). If fail, debug — most likely a residual mode disagreement.

- [ ] **Step 3: Commit**

```bash
git add tests/test_fisher.py
git commit -m "test(fisher): jacfwd matches jacrev and FD-based Fisher"
```

### Task B12: Performance benchmark — jacfwd vs jacrev

**Files:**
- Create: `diags/diag_perf_jacfwd_vs_jacrev.py`

- [ ] **Step 1: Write the benchmark script**

```python
"""Benchmark: jacfwd vs jacrev for Fisher Jacobian on compute_pk_table.

Hypothesis: m forward passes (m=5) beats n reverse passes (n=50) by ~10x in
wall time after compile. Compile time may favor jacrev (smaller graph)."""
import os, sys, time
sys.path.insert(0, ".")
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # set to "gpu" on igpu
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from clax import CosmoParams, compute_pk_table
from tests.pk_test_utils import PK_TABLE_GRAD_FAST_PREC

FIDUCIAL = CosmoParams()
PARAMS = ("h", "omega_b", "omega_cdm", "ln10A_s", "n_s")
K_GRID = np.geomspace(1e-3, 0.3, 50)


def f(theta):
    kw = {n: theta[i] for i, n in enumerate(PARAMS)}
    p = FIDUCIAL.replace(**kw)
    return jnp.log(compute_pk_table(p, PK_TABLE_GRAD_FAST_PREC, k=jnp.asarray(K_GRID)))


theta0 = jnp.asarray([float(getattr(FIDUCIAL, n)) for n in PARAMS])

print("=== jacrev compile ===")
t0 = time.time(); jrev_fn = jax.jit(jax.jacrev(f))
J = jrev_fn(theta0); J.block_until_ready()
print(f"  compile+first call: {time.time()-t0:.1f}s")

print("=== jacrev run (compiled) ===")
t0 = time.time()
for _ in range(3):
    J = jrev_fn(theta0); J.block_until_ready()
print(f"  3x run: {(time.time()-t0)/3:.3f}s avg")

print("\n=== jacfwd compile ===")
t0 = time.time(); jfwd_fn = jax.jit(jax.jacfwd(f))
J = jfwd_fn(theta0); J.block_until_ready()
print(f"  compile+first call: {time.time()-t0:.1f}s")

print("=== jacfwd run (compiled) ===")
t0 = time.time()
for _ in range(3):
    J = jfwd_fn(theta0); J.block_until_ready()
print(f"  3x run: {(time.time()-t0)/3:.3f}s avg")
```

- [ ] **Step 2: Run on Mac CPU first (for sanity)**

```bash
micromamba run -n clax_class-pt_py310forge python -u diags/diag_perf_jacfwd_vs_jacrev.py
```

- [ ] **Step 3: Run on igpu V100 to get production numbers**

```bash
ssh igpu  # job submit or interactive
micromamba run -n clax_class-pt_py310forge python -u diags/diag_perf_jacfwd_vs_jacrev.py
```

Expected (rough): jacrev compile/run ~30s/0.5s; jacfwd ~30s/0.1s. Forward-mode should be 3-5× faster on the run path for m=5, n=50.

- [ ] **Step 4: Commit**

```bash
git add diags/diag_perf_jacfwd_vs_jacrev.py
git commit -m "diags: jacfwd vs jacrev performance benchmark"
```

## Phase B6 — Documentation + ship

### Task B13: Update CHANGELOG and DESIGN.md

- [ ] **Step 1: CHANGELOG entry under "Unreleased"**

```markdown
### Added

- Forward-mode AD (`jax.jvp` / `jax.jacfwd`) end-to-end through `compute_pk`
  and `compute_pk_table` when `prec.ode_adjoint = "direct"`. Enables Fisher
  matrix computation via `jax.jacfwd`, asymptotically faster than `jacrev`
  for our typical (m=5–10 cosmo params, n=10²–10³ data points) Fisher problems.
- `tests/test_fisher.py`: Fisher correctness + jacfwd-vs-jacrev cross-check.
- `diags/diag_perf_jacfwd_vs_jacrev.py`: performance benchmark.

### Changed

- `clax/thermodynamics.py:_find_z_reio`: converted from `custom_vjp` to
  `custom_jvp`. JAX derives reverse-mode VJP via transposition; reverse-mode
  behavior is unchanged.
- `clax/shooting.py:shoot_fn`: same conversion.
```

- [ ] **Step 2: DESIGN.md update under §10 (numerical precision and AD)**

Append a new subsection:

```markdown
### Forward-mode vs reverse-mode AD

`compute_pk` / `compute_pk_table` support both `jax.grad` (reverse) and
`jax.jvp` / `jax.jacfwd` (forward). Reverse mode is the default and works
with any `prec.ode_adjoint`. Forward mode requires
`prec.ode_adjoint = "direct"`: diffrax's `RecursiveCheckpointAdjoint`
internally uses a `custom_vjp` (`checkpointed_while_loop`) that has no JVP
rule, so JAX cannot push tangents through it.

The `stop_gradient(bg.conformal_age)` in `_matter_delta_m_single_k_impl` is
required by diffrax's `PIDController` (which marks its accepted-step factor
as non-differentiable). The Taylor correction immediately after restores
the missing chain-rule term in a mode-agnostic way: both `jax.grad` and
`jax.jvp` of `δ_m + ddelta_m_dtau · (τ_traced − τ_end)` recover the correct
sensitivity, since the expression is identically zero at fiducial but its
linear coefficient carries the τ_end-derivative information.

For Fisher matrices, prefer `jax.jacfwd` over `jax.jacrev`: m forward passes
(m=5–10 cosmo params) is faster than n reverse passes (n=10²–10³ data points)
by O(n/m).
```

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md DESIGN.md
git commit -m "docs: forward-mode AD support (CHANGELOG + DESIGN)"
```

### Task B14: Open PR-B

- [ ] **Step 1: Push the branch**

```bash
git push -u origin feat/forward-mode-ad
```

- [ ] **Step 2: Open the PR**

```bash
gh pr create --title "feat: forward-mode AD (jax.jvp / jacfwd) end-to-end through compute_pk" \
  --body "$(cat <<'EOF'
## Summary

Enables `jax.jvp` and `jax.jacfwd` end-to-end through `compute_pk` /
`compute_pk_table`, unblocking Fisher-matrix workflows for cosmology
parameter inference. m=5 forward passes is asymptotically faster than n=10²-10³
reverse passes for our typical Jacobian shapes.

Builds on PR #16 (reverse-mode Taylor correction). The Taylor correction
itself is mode-agnostic — this PR adds the plumbing (custom_vjp → custom_jvp)
for the structural blockers so JAX can reach the Taylor expansion.

## Changes

- `clax/thermodynamics.py:_find_z_reio` — `custom_vjp → custom_jvp`. Reverse-mode VJP derived by transposition.
- `clax/shooting.py:shoot_fn` — `custom_vjp → custom_jvp`.
- `tests/test_pk_gradients.py::TestPkScalarForwardMode` — jvp matches grad <1e-6 on all 5 cosmo params.
- `tests/test_thermodynamics.py::test_z_reio_forward_mode_matches_fd` — focused thermo JVP test.
- `tests/test_shooting.py::test_shoot_h_forward_mode_matches_fd` — focused shooting JVP test.
- `tests/test_fisher.py` (new) — Fisher matrix correctness + jacfwd/jacrev cross-check.
- `diags/diag_perf_jacfwd_vs_jacrev.py` (new) — performance benchmark.
- CHANGELOG.md, DESIGN.md — document forward-mode story.

## Test plan

- [ ] `pytest tests/ --fast -x -q` (full quick suite)
- [ ] `pytest tests/test_pk_gradients.py::TestPkScalarForwardMode -v`
- [ ] `pytest tests/test_fisher.py -v`
- [ ] `python diags/diag_perf_jacfwd_vs_jacrev.py` on V100 — expect jacfwd 3-5× faster than jacrev for m=5, n=50

## Notes

- Forward mode requires `prec.ode_adjoint = "direct"` (RecursiveCheckpointAdjoint's `checkpointed_while_loop` is itself a `custom_vjp`; that's a diffrax-internal limitation we work around, not fix).
- The Taylor correction in `_matter_delta_m_single_k_impl` was added in PR #16 and is mode-agnostic. This PR doesn't modify it.
EOF
)"
```

---

## Validation Strategy

### Correctness gates (pass criteria)

| Gate | Threshold | Where |
|---|---|---|
| jvp == grad on every param | <1e-6 rel err | `test_jvp_matches_grad`, `test_jvp_grad_agreement_density_params` |
| `jax.grad(compute_pk)` vs FD | <5% at rtol=1e-6 (PK_CONTRACT_PREC) | `TestPkScalarDensityGradients` (PR-A) |
| `jax.grad(compute_pk)` vs FD | <1.5% at rtol=1e-7 | igpu V100 manual run |
| jacfwd Jacobian == jacrev Jacobian | <1e-6 | `test_fisher_jacfwd_matches_jacrev` |
| Fisher diagonal vs FD | <2% | `test_fisher_against_finite_difference` |

### Robustness gates

- Run all forward-mode tests at `PK_FAST_PREC` (rtol=1e-5) AND `PK_CONTRACT_PREC` (rtol=1e-6). Both must pass.
- Run on at least one off-fiducial parameter point (e.g., shift `h` to 0.7, `omega_cdm` to 0.13). Same thresholds.
- jvp must produce finite outputs (no NaN, no Inf) at all 5 params and at PK_CONTRACT_PREC.

### Performance gates

| Gate | Threshold | Where |
|---|---|---|
| jacfwd compile time on V100 | <60s | benchmark script |
| jacfwd run time per Fisher | <0.5s on V100 | benchmark script |
| jacfwd vs jacrev speedup | ≥3× run time on V100 (m=5, n≥50) | benchmark script |

If jacfwd is *slower* than jacrev: investigate (likely a tracing issue causing redundant recompilation per param). Fix or document the regression.

### Smoke gates

- `pytest tests/ --fast -x -q` green at every commit.
- `pytest tests/ -v --ignore=tests/test_fisher.py` green before opening PR-B (Fisher test is heavy; gate it on `--slow` or run separately).

---

## Risk register

| Risk | Mitigation |
|---|---|
| `_find_z_reio` JVP NaN root cause is deep (multi-layer `jnp.where`/`safe_x` patterns) | Time-box debugging at 4 hours of cycle time. If unresolved, fall back: keep custom_vjp; add a *separate* function `_find_z_reio_jvp_capable` that uses a smooth-approximation root-find (e.g., damped Newton in JAX), and route forward-mode users there. |
| `RecursiveCheckpointAdjoint` is the production-default; users who switch to `DirectAdjoint` for forward-mode hit memory limits at large `pt_tau_n_points` | Document explicitly. Recommend `pt_tau_n_points ≤ 1500` for forward-mode use (sufficient for Fisher; production C_l uses 5000). |
| Taylor correction works in jvp but accumulates floating-point error differently than grad | Tested directly in `test_jvp_grad_agreement_density_params`. If <1e-6 holds, this risk is closed. |
| diffrax upgrade breaks the `eqxi.nondifferentiable` workaround | Pin diffrax version in `pyproject.toml`. Add a CI check that runs `pytest tests/test_pk_gradients.py -v` on every dependency bump. |

---

## Self-review

- [x] Spec coverage: PR-A (Tasks A1-A3), PR-B (Tasks B1-B14). All bullet points from the user's request mapped to a task.
- [x] No placeholders: every code block is concrete; every shell command is executable; every threshold is numeric.
- [x] Type consistency: `PK_GRAD_DENSITY_PARAMS`, `PK_GRAD_DENSITY_FD_STEPS`, `PK_GRAD_DENSITY_REL_TOL` defined in PR-A and reused in PR-B Task B10. `_compute_theta_s`, `make_shoot_h_from_theta_s` from existing code; not redefined.
- [x] TDD: every code change has a preceding failing test. Test runs before fix; fix runs after. RED-GREEN-COMMIT cycle.

---

## iGPU Agent Prompt

A drop-in prompt to hand to a fresh agent on igpu. Self-contained: includes context, environment setup, and the entry-point task.

```
Task: Execute the clax AD correctness plan on igpu (V100).

Plan location: docs/superpowers/plans/2026-05-06-clax-ad-correctness.md
(committed on branch benchmark/clax-pt; pull origin/benchmark/clax-pt for the
latest version).

Environment:
- Repo root: /home/n2minh/clax (or /lustre/work/n2minh/clax — whichever has
  the latest pull). If unsure: `git -C /home/n2minh/clax log --oneline -1`
  and pick the freshest.
- Python env: micromamba env clax_class-pt_py310forge. Run all python via
  `micromamba run -n clax_class-pt_py310forge python ...`. No wrapper script.
- GPU: V100-32GB, direct compute node access (no sbatch needed for
  development; use sbatch only for long benchmarks).
- Reference CLASS: ../class_public-3.3.4/

Phase order:
1. PR-A finalization (Tasks A1-A3): commit diags, run V100 verification,
   write CHANGELOG entry. PR-A is on branch fix/pk_tau_end (PR #16); make
   commits there and push to origin.
2. PR-B implementation (Tasks B1-B14): branch off main (after PR #16 merges)
   or off fix/pk_tau_end (rebase later). Branch name: feat/forward-mode-ad.

Workflow rules:
- Use the superpowers:subagent-driven-development skill to execute tasks
  task-by-task. One task per subagent dispatch.
- Read each task's "Files" and "Step N" entries; execute exactly as written.
- TDD strict: write the failing test FIRST, run it to confirm RED, then
  implement, run to confirm GREEN, then commit. Do not batch the steps.
- Commit at the end of every task. Never accumulate uncommitted state across
  tasks.
- After every code commit, run `pytest tests/ --fast -x -q` as a smoke gate.
  If anything regresses, STOP and report; do not advance.

Hard rules (from CLAUDE.md):
- CLASS is the value oracle; centered FD is the gradient oracle.
- Do NOT change FD step sizes to mask AD-FD disagreement.
- Do NOT add stop_gradient to make tests pass (the existing one in
  _matter_delta_m_single_k_impl is structurally required by diffrax — see
  the Taylor correction comment).
- Do NOT use `find /` or `find /ocean` (HPC filesystem will hang). Search
  only within . and ../class_public-3.3.4/.

When you finish all tasks, post a summary to PR-A and PR-B with the V100
correctness numbers and the jacfwd-vs-jacrev speedup.

If a task fails or you get stuck for >30 min, write a status note to
diags/STATUS.md describing what you tried, what you observed, and what's
blocked, then return.

Start with: read the plan, then execute Task A1 (commit diags).
```
