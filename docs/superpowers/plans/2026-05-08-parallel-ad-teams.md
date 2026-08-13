# Parallel AD Teams — PR-A Finish + PR-B Start

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two independent agent teams work simultaneously: Team A finalises PR-A (reverse-mode AD verification + CHANGELOG on `fix/pk_tau_end`); Team B begins PR-B (forward-mode AD enablement on a new `feat/forward-mode-ad` branch).

**Architecture:**
- Teams work on separate branches — no file conflicts. Team A touches only `CHANGELOG.md` and runs tests (read-only); Team B creates a new branch and modifies `clax/thermodynamics.py`, `clax/shooting.py`, and test files.
- Team B uses TDD strictly: write the failing test, observe RED, implement, observe GREEN, commit. No code changes before a failing test exists.
- Both teams run on igpu (direct GPU access, no sbatch needed for development).

**Tech Stack:** JAX 0.9.2, diffrax, micromamba env `clax` (`micromamba activate clax`), igpu V100-32GB. Run python/pytest directly — no wrapper scripts.

---

## Environment

Both teams run from `/home/n2minh/clax`. The micromamba env is `clax`:

```bash
micromamba activate clax
```

Direct GPU access on igpu: run `python` / `pytest` without any `gpu-run.sh` wrapper (CLAUDE.md).

---

## File Structure

### Team A (PR-A)

| File | Action |
|---|---|
| `fix/pk_tau_end` branch | Run tests, write CHANGELOG, push |
| `CHANGELOG.md` | Add entry for the Taylor-correction gradient fix |

### Team B (PR-B)

| File | Action |
|---|---|
| `clax/thermodynamics.py:845–931` | Replace `@jax.custom_vjp` block + fwd/bwd helpers + `defvjp` call with `@jax.custom_jvp` + `defjvp` |
| `clax/shooting.py:76–118` | Replace `@jax.custom_vjp` block + `shoot_fwd` + `shoot_bwd` + `defvjp` with `@jax.custom_jvp` + `defjvp` |
| `tests/test_thermodynamics.py` | Add `test_find_z_reio_forward_mode_matches_fd` at module level |
| `tests/test_shooting.py` | Add `TestShootingForwardMode` class with `test_shoot_fn_forward_mode_matches_fd` |

---

# TEAM A — PR-A Completion

Team A's branch is `fix/pk_tau_end`. All commits go there.

## Task A1: Checkout and verify the gradient regression tests on V100

**Files:** (read-only; run tests only)

- [ ] **Step 1: Switch to fix/pk_tau_end**

```bash
cd /home/n2minh/clax
git fetch origin
git checkout fix/pk_tau_end
git pull
git log --oneline -5
```

Expected: top commit is `a4a7ab2 fix(perturbations): recover dtau_end gradient via Taylor correction`. Confirm `TestPkScalarDensityGradients` class exists:

```bash
grep -n "class TestPkScalarDensityGradients" tests/test_pk_gradients.py
```

Expected: one match around line 159.

- [ ] **Step 2: Run the density-gradient regression test**

```bash
python -m pytest tests/test_pk_gradients.py::TestPkScalarDensityGradients -v \
  2>&1 | tee /tmp/a1_density_grad.log
tail -20 /tmp/a1_density_grad.log
```

Expected: `3 passed` (h, omega_b, omega_cdm all below the 5% relative-error threshold at `PK_GRAD_CONTRACT_PREC`).

- [ ] **Step 3: Run smoke test**

```bash
python -m pytest tests/ --fast -x -q 2>&1 | tail -15
```

Expected: all pass (or same baseline failures as the branch baseline — no new failures).

---

## Task A2: Write CHANGELOG entry

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Read the top of CHANGELOG.md to follow its format**

```bash
head -60 CHANGELOG.md
```

- [ ] **Step 2: Add the PR-A entry near the top (immediately before the first existing dated entry)**

Insert the following block:

```markdown
### May 8, 2026: compute_pk reverse-mode AD — recover dδ_m/dτ·dτ_end/dθ via Taylor correction

**Background:** `RecursiveCheckpointAdjoint` + `PIDController` require
`stop_gradient(bg.conformal_age)` when passing `tau_max` to `diffeqsolve`.
This froze the ODE endpoint from AD's perspective, sacrificing the
`dδ_m/dτ · dτ_end/dθ` chain-rule term and producing 65–84% errors in
`dP(k)/d{h, omega_b, omega_cdm}` vs centred FD.

**Fix (commit `a4a7ab2` on `fix/pk_tau_end`, cherry-picked as `62baaa2` on
`benchmark/clax-pt`):** First-order Taylor expansion around the frozen
endpoint in `_matter_delta_m_single_k_impl`:

    δ_m ≈ δ_m_frozen + (dδ_m/dτ) · (τ_traced − τ_end)

The correction is identically zero at the primal (τ_traced = τ_end), so
the forward pass is unchanged. The tangent carries the missing `dτ_end/dθ`
chain-rule term.

**Gradient errors after fix (rtol=1e-3 / V100):**

| Param      | Before (rtol=1e-3) | After (rtol=1e-3) | After (rtol=1e-7) |
|------------|--------------------|-------------------|-------------------|
| h          | ~65%               | <5%               | <1%               |
| omega_b    | ~84%               | <5%               | <1.5%             |
| omega_cdm  | ~75%               | <5%               | <1%               |
| ln10A_s, n_s | 0%               | 0%                | 0%                |

**Regression test:** `tests/test_pk_gradients.py::TestPkScalarDensityGradients`
asserts `dP/d{h, omega_b, omega_cdm}` matches centred FD at `<5%` with
`PK_GRAD_CONTRACT_PREC` (rtol=1e-6). Catches the 65–84% bug class with
margin for the FD precision floor.
```

- [ ] **Step 3: Commit CHANGELOG to fix/pk_tau_end**

```bash
git add CHANGELOG.md
git commit -m "$(cat <<'EOF'
docs(CHANGELOG): document compute_pk gradient fix (PR-A)

Taylor correction in _matter_delta_m_single_k_impl recovers the
dδ_m/dτ·dτ_end/dθ chain-rule term suppressed by stop_gradient.
Errors drop from 65-84% to <5% at rtol=1e-3 vs centred FD.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: Push to origin**

```bash
git push origin fix/pk_tau_end
```

---

## Task A3: Confirm PR #16 is up to date

- [ ] **Step 1: Check PR status**

```bash
gh pr view 16 --json title,state,headRefName 2>&1
```

Expected: `state: OPEN`, `headRefName: fix/pk_tau_end`.

- [ ] **Step 2: Check for unaddressed review comments**

```bash
gh pr view 16 --comments 2>&1 | tail -40
```

If `smsharma` left review comments that are not yet addressed, respond or fix them in a follow-up commit to `fix/pk_tau_end` and push. Otherwise Team A is done.

---

# TEAM B — PR-B Start

Team B creates and works on branch `feat/forward-mode-ad`, branched from `fix/pk_tau_end` (so it inherits the Taylor correction as baseline). All commits go to `feat/forward-mode-ad`.

## Setup

- [ ] **Step 0: Create the branch**

```bash
cd /home/n2minh/clax
git fetch origin
git checkout fix/pk_tau_end
git pull
git checkout -b feat/forward-mode-ad
git log --oneline -3
```

Expected: top commit is the `a4a7ab2` Taylor correction commit (or later if Team A has pushed since).

---

## Task B1: Write failing test for `_find_z_reio` JVP

This test verifies `jax.jvp` through the thermodynamics stack's `z_reio` output. It should **fail (RED)** before the conversion in Task B2 with `TypeError: can't apply forward-mode autodiff (jvp) to a custom_vjp function`.

**Files:**
- Modify: `tests/test_thermodynamics.py`

- [ ] **Step 1: Check the end of test_thermodynamics.py to find the insertion point**

```bash
tail -20 tests/test_thermodynamics.py
```

- [ ] **Step 2: Append the test at the end of tests/test_thermodynamics.py**

```python


def test_find_z_reio_forward_mode_matches_fd():
    """jax.jvp through z_reio(h) is finite and matches centred FD to <1%.

    RED before converting _find_z_reio from custom_vjp to custom_jvp
    (raises TypeError: can't apply forward-mode autodiff to a custom_vjp function).
    GREEN after conversion.
    """
    import dataclasses

    PREC_JVP = PrecisionParams(
        bg_n_points=400, ncdm_bg_n_points=200, bg_tol=1e-8,
        th_n_points=10000, th_z_max=5e3,
        ode_adjoint="direct",
    )
    params = CosmoParams()

    def z_reio_of_h(h):
        p = dataclasses.replace(params, h=h)
        bg_ = background_solve(p, PREC_JVP)
        th_ = thermodynamics_solve(p, PREC_JVP, bg_)
        return th_.z_reio

    # Forward-mode AD
    primal, tangent = jax.jvp(z_reio_of_h, (params.h,), (jnp.asarray(1.0),))
    primal.block_until_ready()

    assert jnp.isfinite(tangent), f"jvp returned non-finite tangent: {tangent}"

    # Centred FD for ground truth
    eps = 1e-3
    fd = (z_reio_of_h(params.h + eps) - z_reio_of_h(params.h - eps)) / (2 * eps)
    rel = abs(float(tangent) - float(fd)) / (abs(float(fd)) + 1e-30)
    assert rel < 0.01, (
        f"jvp(z_reio, h)={float(tangent):.6e}  FD={float(fd):.6e}  rel={rel:.2%}"
    )
```

- [ ] **Step 3: Run the test — confirm RED**

```bash
python -m pytest tests/test_thermodynamics.py::test_find_z_reio_forward_mode_matches_fd -v \
  2>&1 | tail -15
```

Expected failure: `TypeError: can't apply forward-mode autodiff (jvp) to a custom_vjp function`.
If PASS here, `_find_z_reio` was already converted — skip Task B2's conversion step.

- [ ] **Step 4: Commit the RED test**

```bash
git add tests/test_thermodynamics.py
git commit -m "$(cat <<'EOF'
test(thermodynamics): add z_reio jvp-vs-FD regression (RED)

Fails until _find_z_reio is converted from custom_vjp to custom_jvp.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task B2: Convert `_find_z_reio` from `custom_vjp` to `custom_jvp`

Replaces the three-part `custom_vjp` pattern (decorator + fwd fn + bwd fn + `defvjp` call, lines 845–931) with `@jax.custom_jvp` + `@_find_z_reio.defjvp`. JAX derives the VJP from the JVP via transposition; reverse-mode behaviour is preserved.

**Files:**
- Modify: `clax/thermodynamics.py:845–931`

- [ ] **Step 1: Confirm the exact line range to replace**

```bash
grep -n "^@jax.custom_vjp\|^def _find_z_reio_fwd\|^def _find_z_reio_bwd\|^_find_z_reio.defvjp" \
  clax/thermodynamics.py
```

Expected (approximate):
```
845:@jax.custom_vjp
870:def _find_z_reio_fwd(
892:def _find_z_reio_bwd(res, g):
931:_find_z_reio.defvjp(_find_z_reio_fwd, _find_z_reio_bwd)
```

- [ ] **Step 2: Delete lines 845–931 and replace with the custom_jvp implementation**

Delete everything from `@jax.custom_vjp` (line 845) through `_find_z_reio.defvjp(...)` (line 931, inclusive) and replace with:

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
    """Differentiable z_reio solve: primal uses bounded bisection in
    _find_z_reio_impl; JVP applies the implicit function theorem to
    F(z, inputs) = tau_reio_model(z, inputs) - tau_reio_target.
    JAX derives VJP via transposition, preserving reverse-mode behaviour."""
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

    # dF/dz at the solution: F(z) = tau_reio_model(z) - target
    _, dF_dz = jax.jvp(
        lambda z_: _tau_reio_for_zreio(
            z_, xe_raw_grid, kd_prefactor, dtau_grid, z_grid, Y_He
        ),
        (z_reio,),
        (jnp.ones_like(z_reio),),
    )
    # Guard against near-zero denominator
    dF_dz = jnp.where(
        jnp.abs(dF_dz) < 1e-12,
        jnp.where(dF_dz >= 0.0, 1e-12, -1e-12),
        dF_dz,
    )

    # Tangent of F w.r.t. inputs at fixed z_reio (implicit function theorem)
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

- [ ] **Step 3: Run existing reverse-mode thermodynamics tests — confirm no regression**

```bash
python -m pytest tests/test_thermodynamics.py -v --fast 2>&1 | tail -20
```

Expected: all pass. JAX derives VJP from JVP via transposition.

- [ ] **Step 4: Run the forward-mode test from Task B1**

```bash
python -m pytest tests/test_thermodynamics.py::test_find_z_reio_forward_mode_matches_fd -v \
  2>&1 | tail -15
```

Two outcomes:
- **PASS** → skip Task B3, go directly to Task B4 (commit).
- **FAIL with NaN** → proceed to Task B3 (debug + fix NaN).

---

## Task B3: Debug and fix NaN in `_find_z_reio_jvp` (only if Task B2 Step 4 fails with NaN)

Skip entirely if Task B2 Step 4 passed.

**Files:**
- Temporarily modify: `clax/thermodynamics.py` (remove all debug prints before the commit)

- [ ] **Step 1: Run the existing pipeline bisect diag to confirm the NaN stage**

```bash
python -u diags/diag_jvp_bisect.py 2>&1 | head -30
```

Expected: Stage 1 (bg.conformal_age) is finite; Stage 2 (th.z_reio) is NaN. This confirms the NaN is inside `_find_z_reio_jvp`, not upstream.

- [ ] **Step 2: Add debug prints inside `_find_z_reio_jvp` to identify the NaN variable**

Temporarily insert immediately before `return z_reio, z_reio_dot` in `_find_z_reio_jvp`:

```python
        jax.debug.print(
            "[zreio.jvp] z={a}  dF_dz={b}  tau_jvp={c}  target_dot={d}  F_dot={e}  z_dot={f}",
            a=z_reio, b=dF_dz, c=tau_jvp, d=tau_reio_target_dot, e=F_dot, f=z_reio_dot,
        )
```

Re-run the test:

```bash
python -m pytest tests/test_thermodynamics.py::test_find_z_reio_forward_mode_matches_fd -v \
  2>&1 | grep "zreio.jvp"
```

Note which variable (`dF_dz`, `tau_jvp`, `F_dot`, or `z_dot`) is first NaN.

- [ ] **Step 3A: If `tau_jvp` or `dF_dz` is NaN — apply the safe-mask fix in `_tau_reio_for_zreio`**

The likely cause: `jnp.maximum(xe_total - xe_raw_grid, 0.0)` at line ~806 can produce NaN tangents when the `jnp.maximum` boundary is hit. Apply the JAX safe-mask pattern.

Find the line:

```bash
grep -n "jnp.maximum.*xe_raw_grid" clax/thermodynamics.py
```

Replace the `jnp.maximum` call (should read `xe_extra = jnp.maximum(xe_total - xe_raw_grid, 0.0)`) with:

```python
    diff = xe_total - xe_raw_grid
    # safe-mask: avoids NaN tangents from jnp.maximum at the boundary
    xe_extra = jnp.where(diff > 0.0, diff, jnp.zeros_like(diff))
```

Re-run:

```bash
python -m pytest tests/test_thermodynamics.py::test_find_z_reio_forward_mode_matches_fd -v \
  2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 3B: If `dF_dz` is NaN but `tau_jvp` is finite — add finer-grained prints**

Insert inside `_tau_reio_for_zreio` (immediately after each assignment):

```python
    jax.debug.print("[tau_reio] xe_total[:3]={x}", x=xe_total[:3])
    jax.debug.print("[tau_reio] xe_extra[:3]={x}", x=xe_extra[:3])
    jax.debug.print("[tau_reio] kappa_integ[:3]={x}", x=kappa_integ[:3])
```

Run the diag:

```bash
python -u diags/diag_jvp_find_z_reio.py 2>&1 | grep "tau_reio"
```

Identify the first NaN and apply the analogous safe fix (replace the offending op with a `jnp.where`-guarded equivalent).

- [ ] **Step 4: Remove all debug prints added in Steps 2 and 3B**

```bash
grep -n "zreio.jvp\|tau_reio.*debug\|debug.print" clax/thermodynamics.py
```

Expected: no matches. Delete any that appear.

- [ ] **Step 5: Re-run both suites to confirm GREEN + no regression**

```bash
python -m pytest tests/test_thermodynamics.py -v --fast 2>&1 | tail -15
python -m pytest tests/test_thermodynamics.py::test_find_z_reio_forward_mode_matches_fd -v \
  2>&1 | tail -10
```

Expected: all pass.

---

## Task B4: Commit `_find_z_reio` conversion

Run this task regardless of whether Task B3 was needed.

```bash
git add clax/thermodynamics.py
git commit -m "$(cat <<'EOF'
refactor(thermodynamics): convert _find_z_reio custom_vjp -> custom_jvp

IFT-based JVP rule: dz/d(inputs) = -dF/d(inputs) / dF/dz at solution.
JAX derives VJP via transposition; reverse-mode behaviour unchanged.
Forward-mode jvp(z_reio, h) now matches centred FD to <1%.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task B5: Write failing test for `shoot_fn` JVP

**Files:**
- Modify: `tests/test_shooting.py`

- [ ] **Step 1: Check the end of tests/test_shooting.py**

```bash
tail -20 tests/test_shooting.py
```

- [ ] **Step 2: Append the forward-mode test class at the end of tests/test_shooting.py**

```python


class TestShootingForwardMode:
    """Forward-mode AD through the shooting map theta_s_100 -> h."""

    def test_shoot_fn_forward_mode_matches_fd(self):
        """jax.jvp(shoot_fn, theta_s, tangent=1.0) matches centred FD to <1%.

        RED before converting shoot_fn from custom_vjp to custom_jvp.
        GREEN after conversion.
        """
        params = CosmoParams()
        shoot_fn = make_shoot_h_from_theta_s(PREC)
        theta_s_fid = float(_compute_theta_s(params.h, params, PREC))

        # Forward-mode AD
        primal, tangent = jax.jvp(
            lambda ts: shoot_fn(ts, params),
            (jnp.asarray(theta_s_fid),),
            (jnp.asarray(1.0),),
        )
        primal.block_until_ready()

        assert jnp.isfinite(tangent), f"jvp returned non-finite: {tangent}"

        # Centred FD for ground truth
        eps = 1e-5
        h_plus = float(shoot_fn(theta_s_fid + eps, params))
        h_minus = float(shoot_fn(theta_s_fid - eps, params))
        fd = (h_plus - h_minus) / (2 * eps)
        rel = abs(float(tangent) - fd) / (abs(fd) + 1e-30)
        assert rel < 0.01, (
            f"jvp(shoot_fn, theta_s)={float(tangent):.6e}  FD={fd:.6e}  rel={rel:.2%}"
        )
```

- [ ] **Step 3: Run the test — confirm RED**

```bash
python -m pytest \
  "tests/test_shooting.py::TestShootingForwardMode::test_shoot_fn_forward_mode_matches_fd" \
  -v 2>&1 | tail -10
```

Expected failure: `TypeError: can't apply forward-mode autodiff (jvp) to a custom_vjp function`.

- [ ] **Step 4: Commit the RED test**

```bash
git add tests/test_shooting.py
git commit -m "$(cat <<'EOF'
test(shooting): add shoot_fn jvp-vs-FD regression (RED)

Fails until shoot_fn is converted from custom_vjp to custom_jvp.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task B6: Convert `shoot_fn` from `custom_vjp` to `custom_jvp`

**Files:**
- Modify: `clax/shooting.py:76–118`

- [ ] **Step 1: Confirm the exact lines to replace**

```bash
grep -n "@jax.custom_vjp\|def shoot_fwd\|def shoot_bwd\|shoot_fn.defvjp" clax/shooting.py
```

Expected:
```
76:    @jax.custom_vjp
102:    def shoot_fwd(theta_s_100_target, params_template):
106:    def shoot_bwd(res, g):
117:    shoot_fn.defvjp(shoot_fwd, shoot_bwd)
```

The block to replace is from `    @jax.custom_vjp` (line 76) through `    return shoot_fn` (line 118).

- [ ] **Step 2: Delete lines 76–118 and replace with the custom_jvp implementation**

Delete from `    @jax.custom_vjp` through `    return shoot_fn` and replace with:

```python
    @jax.custom_jvp
    def shoot_fn(theta_s_100_target: float, params_template: CosmoParams) -> float:
        """Find h such that 100*theta_s(h) = theta_s_100_target.

        Uses Newton's method with a fixed number of iterations.
        Initial guess from CLASS input.c:1190:
            h_guess = 3.54*theta_s^2 - 5.455*theta_s + 2.548
        """
        # CLASS's initial guess formula (input.c:1190)
        h0 = 3.54 * theta_s_100_target**2 - 5.455 * theta_s_100_target + 2.548

        def newton_step(i, h):
            theta_s = _compute_theta_s(h, params_template, prec)
            # Finite difference derivative for the forward Newton solve
            eps = 1e-4
            theta_s_plus = _compute_theta_s(h + eps, params_template, prec)
            dtheta_dh = (theta_s_plus - theta_s) / eps
            # Damped Newton update to prevent oscillation
            update = (theta_s - theta_s_100_target) / dtheta_dh
            h = h - 0.5 * update
            return h

        h_final = jax.lax.fori_loop(0, 25, newton_step, h0)
        return h_final

    @shoot_fn.defjvp
    def _shoot_fn_jvp(primals, tangents):
        theta_s_100_target, params_template = primals
        target_dot, _ = tangents  # params_template tangents not propagated (None)
        h = shoot_fn(theta_s_100_target, params_template)
        # Implicit function theorem: F(h, target) = theta_s(h) - target = 0
        # dF/dh = dtheta_s/dh,  dF/d(target) = -1
        # => h_dot = target_dot / (dtheta_s/dh)
        _, dtheta_dh = jax.jvp(
            lambda h_: _compute_theta_s(h_, params_template, prec),
            (h,),
            (jnp.ones_like(h),),
        )
        h_dot = target_dot / dtheta_dh
        return h, h_dot

    return shoot_fn
```

- [ ] **Step 3: Run existing shooting tests — confirm no regression**

```bash
python -m pytest tests/test_shooting.py -v --fast 2>&1 | tail -20
```

Expected: all existing tests pass (`TestComputeThetaS`, `TestShootingRoundTrip`, `TestShootingGradient`). JAX derives VJP from JVP via transposition; reverse-mode unchanged.

- [ ] **Step 4: Run the forward-mode test from Task B5 — confirm GREEN**

```bash
python -m pytest \
  "tests/test_shooting.py::TestShootingForwardMode::test_shoot_fn_forward_mode_matches_fd" \
  -v 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 5: Run full smoke test**

```bash
python -m pytest tests/ --fast -x -q 2>&1 | tail -15
```

Expected: all pass. If anything regresses, fix before proceeding.

- [ ] **Step 6: Commit**

```bash
git add clax/shooting.py
git commit -m "$(cat <<'EOF'
refactor(shooting): convert shoot_fn custom_vjp -> custom_jvp

IFT-based JVP rule: dh/d(target) = 1 / (dtheta_s/dh).
JAX derives VJP via transposition; reverse-mode behaviour unchanged.
Forward-mode jvp(shoot_fn, theta_s) now matches centred FD to <1%.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task B7: Smoke-check forward-mode pipeline through thermodynamics

Verify that both blockers are gone by running the existing pipeline bisect diag.

**Files:** (read-only)

- [ ] **Step 1: Run the bisect diag**

```bash
python -u diags/diag_jvp_bisect.py 2>&1
```

Expected: Stage 1 (bg.conformal_age), Stage 2 (th.z_reio), Stage 3 (th.tau_star) all return **finite** tangents.
Stage 4 (full compute_pk) will still fail with `RecursiveCheckpointAdjoint` / `custom_vjp` error — that is expected and is addressed in later PR-B tasks (out of scope for this plan).

- [ ] **Step 2: Push the branch**

```bash
git push -u origin feat/forward-mode-ad
```

---

## Validation Summary

| Gate | Owner | Command | Pass criterion |
|---|---|---|---|
| `TestPkScalarDensityGradients` (3 params) | Team A | `pytest tests/test_pk_gradients.py::TestPkScalarDensityGradients -v` | 3/3 PASS |
| Full smoke (fix/pk_tau_end) | Team A | `pytest tests/ --fast -x -q` | All pass |
| `test_find_z_reio_forward_mode_matches_fd` | Team B | `pytest tests/test_thermodynamics.py::test_find_z_reio_forward_mode_matches_fd -v` | PASS; tangent finite; <1% vs FD |
| Reverse-mode thermo (no regression) | Team B | `pytest tests/test_thermodynamics.py --fast -v` | All pass |
| `TestShootingForwardMode` | Team B | `pytest tests/test_shooting.py::TestShootingForwardMode -v` | PASS |
| Reverse-mode shooting (no regression) | Team B | `pytest tests/test_shooting.py --fast -v` | All pass |
| Full smoke (feat/forward-mode-ad) | Team B | `pytest tests/ --fast -x -q` | All pass |

---

## Self-review

**Spec coverage:**
- PR-A: gradient verification (A1) ✓, CHANGELOG (A2) ✓, PR check (A3) ✓
- PR-B: `_find_z_reio` blocker removed (B1–B4) ✓, `shoot_fn` blocker removed (B5–B6) ✓, smoke-check (B7) ✓
- PR-B end-to-end forward pass through `compute_pk` (Tasks B8–B14 in the original `2026-05-06-clax-ad-correctness.md` plan) — out of scope here; pick up after this plan's `feat/forward-mode-ad` branch is reviewed.

**Placeholder scan:** No TBD/TODO/placeholder. All code blocks are complete and executable.

**Type consistency:** `make_shoot_h_from_theta_s`, `_compute_theta_s`, `_find_z_reio_impl`, `_tau_reio_for_zreio` all exist in the current codebase; none redefined. Test helpers (`PREC`, `CosmoParams`, `background_solve`, `thermodynamics_solve`) follow conventions already established in each test file.

**NaN handling:** Task B3 provides two explicit diagnostic paths (3A: safe-mask fix for `jnp.maximum`; 3B: deeper print-based localisation) so Team B is not blocked if NaN appears after the initial conversion.
