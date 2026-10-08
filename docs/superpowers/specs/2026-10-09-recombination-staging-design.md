# Smooth recombination stepping: helium and hydrogen follow CLASS's Saha → ODE staging

**Date:** Oct 9, 2026
**Issue:** smsharma/clax#48 (helium-recombination derivative spikes); related #30, #45
**Branch:** `fix/recombination-staging`, from upstream main `4d084b5`
**Status:** design approved section by section; written spec awaiting review

## 1. Problem (physics first)

Every AD gradient that passes through `thermodynamics_solve` with respect to h, ω_b, ω_cdm, Y_He, N_eff or T_cmb carries a random error. It ranges from per-mille to order unity, and depends on where the perturbation ODE solver happens to place its steps.

**Cause.** clax integrates the stiff HeII → HeI and hydrogen Peebles equations with explicit steps clipped to [0, 1], through the very redshift ranges where CLASS does not integrate at all (it uses Saha there).

**Helium, z ≈ 3000–3500.**
- Explicit Heun sub-steps clipped to [0, 1] (`thermodynamics.py:655-669`), plus a hard `xHe > 0.999 → C_He = 1` switch (`:270`).
- x_e wiggles by 0.35% between neighbouring grid points.
- Its derivative spikes to |d ln x_e/dh| ≈ 5×10¹², against smooth values ≈ 10⁻².

**Hydrogen, z ≈ 1550–1750.**
- Explicit RK4, clipped (`:615-627`), sigmoid-blended with Saha at z = 1600 (`:645-646`).
- It runs past RK4's stability limit: h·J = −3.2 at z = 1600, against the limit of −2.79.
- On the `fit_cl` grid (3000 points, the NUTS preset) x_e at fiducial drops to **0.035 at z ≈ 1559**, where the truth is ≈ 1.0. Derivative spikes there reach 2×10⁴.

**How the spikes reach the observables.**
- They enter the perturbations through c_s² k² δ_b in θ_b′.
- Whether a solver stage lands on a spike depends on the step sequence: CPU vs GPU batching, solver, stage-matrix form, jit vs no jit.
- Consequences, measured: forward and reverse AD disagree by up to 113% on the production gradient path; CPU and GPU derivatives differ by a factor of 21; df/dh jumps by 6% along a line in h.

**Why it matters.**
- For NUTS this costs efficiency (energy errors, smaller steps, divergences), not bias: accept/reject uses values, and the values are nearly unaffected except for the hydrogen dip on `fit_cl`.
- Fisher forecasts, MAP/Laplace and derivative-trained emulators are biased.

**Evidence.**
- Diagnosis: `/lustre/work/n2minh/clax-probes/rodas/diag/report.md`.
- Variant probe: `/lustre/work/n2minh/clax-probes/helium/`, jobs 22933–23060.

## 2. Goal and success criteria

The goal is gradients through thermodynamics that are **precise, smooth and robust**, without losing value precision. The four criteria, as the user set them:

1. **Output precision:** x_e, T_b and the visibility agree with CLASS within the project's 0.1% target.
2. **Derivative precision:** AD agrees with converged finite differences, and forward with reverse.
3. **Gradient smoothness:** no spikes, and derivatives independent of where solver steps fall, of platform and of compilation.
4. **Robustness across cosmology:** all of the above at every test cosmology, including extremes in Y_He, N_eff and m_ν.

## 3. How CLASS does it (the oracle)

CLASS v3.3.4 splits recombination into approximation stages (`source/thermodynamics.c:836-900`, `include/precisions.h:194-214`):
- From z = 3500 down to `recfast_z_early_H_recombination` = 2870, **x_He is analytic** (Saha, regime "He2", `thermodynamics.c:4053-4067`).
- Below 2870 it is evolved by the implicit `ndf15` solver (`:1964-1975`).
- Hydrogen follows Saha until `recfast_z_full_H_recombination` = 1600 (regime "H", `:4070-4087`).
- Stage transitions are blended with the smoothstep `f2` (`thermodynamics.h:49`; applied at `thermodynamics.c:2973-2987`; Δz = 50, `precisions.h:202, 205`).

CLASS's x_He > 0.995 switch (`wrap_recfast.c:243`) changes the form of K_He, **not** C_He. clax's `0.999 → C_He = 1` switch has no CLASS counterpart.

**Version note:** no stock `class_public-3.3.4` tree exists on this machine. All CLASS citations and references come from the CLASS-PT tree at `/home/n2minh/CLASS-PT` (`_VERSION_` v3.3.4). Its RECFAST run reproduces `reference_data/lcdm_fiducial/cls_recfast_thermo.npz` to 8.4×10⁻⁶ in x_e.

## 4. Design

### 4.1 What changes in `clax/thermodynamics.py`

The recombination scan keeps its structure: one `lax.scan` over the thermodynamics grid. Within each grid step, the two stiff species change how they are advanced. Every new block carries CLASS file:line comments.

**(a) One shared stiff step, `_ros2_scalar_step`.** An L-stable two-stage linearly implicit (Rosenbrock, ROS2) step for one fraction x, with γ = 1 + 1/√2:

```
W  = 1 − γ h J
k1 = h f(s_j, x_j) / W
k2 = [h f(s_{j+1}, x_j + k1) − 2 k1] / W
x_{j+1} = x_j + (3/2) k1 + (1/2) k2
```

- J = ∂f/∂x comes from a scalar `jax.jvp` with the environment held fixed: n_H, H, T_b, T_r, z, and x_e as an input, exactly as probed.
- ROS2 stays second order for any J.
- 10 sub-steps per grid step. The environment s_j is interpolated per sub-step: ln a linear; ln H and T_b linear in ln a.
- **No clip.**

**(b) Helium, HeII → HeI.**
1. *Saha stage, z > 2870* (CLASS "He2", `thermodynamics.c:4053-4067`):
   - rhs = 4 (C_R T_b)^{3/2} exp(−T_HeI / T_b) / n_H(z), with C_R = 2π m_e k_B / h² and T_HeI = h c L_He1,ion / k_B ≈ 285,335 K (`wrap_recfast.c:78`, `thermodynamics.c:823`);
   - x_HeII^Saha = 2 rhs / [(1 + rhs) + √((1 + rhs)² + 4 f_He rhs)].

   This is CLASS's root x_e = ½[√((rhs−1)² + 4(1+f_He) rhs) − (rhs−1)], x_He = (x_e − 1)/f_He, rearranged to avoid cancellation.
2. *ODE state.* While z > 2870 it is set to the Saha value. Below, the Peebles equation advances it from the previous state with `_ros2_scalar_step`. Its initial condition is the analytic value, as at `thermodynamics.c:2059-2065`.
3. *Output:*
   - x_HeII = w · x_ode + (1 − w) · x_Saha, with w = 3s² − 2s³ and s = clip((2870 − z)/100, 0, 1).
   - This is CLASS's `f2` with Δz = 50, and it is exactly 0 or 1 outside z ∈ [2770, 2870], so neither branch leaks.
4. *Peebles right-hand side* (`wrap_recfast.c:349`, converted to d/d ln a):
   - f = −[x_e x n_H R_down − R_up (1 − x) e^{−T_L/T}] C_He / H.
   - C_He = (1/A + Λ)/(1/A + Λ + R_up), with A = K_He n_He (1 − x) e^{ΔE_2s2p/kT} (`wrap_recfast.c:346`).
   - **The 0.999 switch is deleted.** The probe confirmed C_He is continuous as x → 1: it rises smoothly from 0.0019 at 1 − x = 10⁻² to 0.9992. The x < 10⁻¹⁵ guard stays (CLASS `x_He_trigger_small`).
   - Escape probability p_s = −expm1(−τ)/τ, with 1 − τ/2 + τ²/6 − τ³/24 for τ < 10⁻⁴ (`:259-260`).
5. The HeIII → HeII Saha sigmoid at z ≈ 5000 stays; the probe measured it smooth, with spike ratio 1.19 at z ≈ 4946.

**(c) Hydrogen, the same pattern at z = 1600.**
- Saha via the existing `_solve_hydrogen_saha`, with the x_HeII correction (CLASS "H", `thermodynamics.c:4070-4087`), for z > 1600. Initial condition as at `thermodynamics.c:2093`.
- The Peebles equation with `_ros2_scalar_step` below 1600, **replacing the clipped RK4**.
- The smoothstep blend over z = 1600 → 1500, replacing the sigmoid at `:645`.
- RECFAST's C = 1 for x_H ≥ 0.995 (CLASS `wrap_recfast.c:163-171`; clax `:215`) stays; it is in CLASS.

**(d) Helium-to-hydrogen number ratio (separate commit).**
- Every helium-per-hydrogen ratio uses f_He = Y/(`_NOT4` (1 − Y)), with `_NOT4` = 3.9715 (CLASS `thermodynamics.h:707`), through one helper.
- Today `:417` already does this, while `:367`, `:524`, `:640`, `:650` and every x_e sum use exactly 4 (`0.25 * Y_He / (1 - Y_He)`).
- Measured effect: a 6.4×10⁻⁴ offset in x_e at z = 2870–5000, and 1.2×10⁻³ above z = 5000 at Y_He = 0.28.

**Unchanged:** the background, the κ̇/visibility construction (including the `stop_gradient` at `:807-840`, until §6), reionisation, the T_b update, the `custom_vjp` paths, and the HeIII stage.

### 4.2 Tests and acceptance

**Reference data (new).**
- CLASS in RECFAST mode with `recfast_Heswitch = 1`, the helium physics clax implements.
- x_e and T_b on a ~2000-point z grid, at 10 cosmologies: the 5 of `COSMOLOGY_GRID_LCDM` plus extremes Y_He = 0.22 / 0.28, N_eff = 2 / 4 (via N_ur) and m_ν = 0.15 eV.
- Y_He is passed explicitly, never via BBN.
- Generated by a new function in `scripts/generate_class_reference.py`, run in the `classpt` env. The provenance file records the CLASS-PT 3.3.4 build and its settings.

**Thermodynamics-only tests** (CPU, seconds each; fiducial under `--fast`, all 10 cosmologies in full mode; each on the default 20000-point grid and `fit_cl`'s 3000-point grid):

| Test | Bar | Today (V0) | Fix (probe VE) |
|---|---|---|---|
| x_e vs CLASS, z 2870–5000 | ≤ 1×10⁻³ | 6.3×10⁻³ | 6.4×10⁻⁴ |
| x_e vs CLASS, z 1590–1650 | ≤ 1×10⁻³ | 0.18 | 2.8×10⁻⁴ |
| x_e vs CLASS, z 800–1550 | ≤ 1×10⁻³ | 5.5×10⁻⁴ (0.97 on `fit_cl`) | 1.2×10⁻⁴ (4.4×10⁻⁴ on `fit_cl`) |
| Visibility-peak shift vs CLASS | < 0.01 in z | 0.040 (0.256 on `fit_cl`) | 0.001 (0.013 on `fit_cl`) |
| Values smooth: max \|Δ² x_e\|, z 1500–6000 | < 3×10⁻⁵ | 0.36 | 8.4×10⁻⁶ |
| Derivative spike ratio†, Q ∈ {x_e, c_s², T_b}, θ ∈ {h, ω_b, ω_cdm, Y_He, N_eff}, z 300–6000 | < 10 | 9.5×10¹⁴ | ≤ 1.19 (20000); ≤ 2.55 (3000) |
| Thermodynamics derivatives vs converged FD at sampled z‡ | ≤ 1×10⁻³ | up to 7.4 | ≤ 1.1×10⁻³ (FD-limited) |

† max over grid points of |d ln Q/dθ| ÷ its median over ±25 neighbouring points, with the median floored at 10⁻⁸.
‡ only at sampled redshifts where FD's own two steps (relative steps 10⁻³ and 3×10⁻³) agree to better than 10⁻⁴; other redshifts are skipped and counted in the printout. In the probe, VE's 1.1×10⁻³ in the z ≤ 2000 band came from points where FD agreed only to 10⁻³, so this rule excludes them as FD-limited, not as a failure of the fix. The bar stays 1×10⁻³.

**Downstream tests** (slow; the δ_m functional f = Σ_k δ_m(k, τ₀)² at the rodas5 preset `_PREC_RB` of the probe; 3 cosmologies: fiducial, ω_b +20%, Y_He 0.28):

| Test | Bar | Today | Fix |
|---|---|---|---|
| df/dh independent of step placement (k-chunk 8 vs 62, jit vs no-jit) | spread < 3×10⁻⁴ | 0.17 | 6.3×10⁻⁵ |
| Reverse (native and production `custom_vjp`) vs forward | < 3×10⁻⁴ | up to 1.13 | ≤ 7.7×10⁻⁵ |
| df/dh along h = 0.62–0.72 (11 points) | cubic-fit residual < 3×10⁻⁴ | 6% jump | 1.0×10⁻⁴ |

**GPU-only check** (in the GPU validation job; skipped without a GPU): CPU vs GPU thermodynamics derivatives agree to < 10⁻¹⁰. Today: factor 21; the fix: 2.7×10⁻¹³.

**Regression.**
- Every existing test passes at its current bar: thermodynamics, C_ℓ TT/TE/EE vs CLASS, P(k), EPT and all gradient tests.
- Values will shift: by up to ~6×10⁻³ in x_e in the helium window, and much more around z ≈ 1560 on `fit_cl`.
- A failing existing bar is investigated, never loosened. Docstrings that quote a number this change moves get the new measured value.

**Red first.** Each new test is shown failing on the unfixed code, on at least one grid or cosmology, before its fix lands. For example, x_e at z 800–1550 passes today on the 20000-point grid (5.5×10⁻⁴) but fails on the 3000-point `fit_cl` grid (0.97). The red evidence (job IDs and numbers) goes in the commit message.

**Cost.**
- Benchmark on a V100, before and after: the thermodynamics solve alone, and the full `fit_cl` C_ℓ wall-clock.
- If total `fit_cl` time grows by more than ~5%, stop and report before continuing.

## 5. Delivery

**Branch:** `fix/recombination-staging` from `4d084b5`, worktree `/home/n2minh/clax-recomb`. **Every commit passes the full suite**, so the branch can go upstream without squashing and stays bisectable.

1. **Reference data:** the generator function and its outputs at the 10 cosmologies.
2. **Shared step + helium:** `_ros2_scalar_step`, the Saha → ODE staging at z = 2870, deletion of the 0.999 switch, `expm1`. Includes the tests helium alone can satisfy: helium-window values and derivatives, spike ratio over z 1800–6000, values smooth over z 1800–6000.
3. **Hydrogen:** staging at z = 1600 with `_ros2_scalar_step` replacing RK4, plus the hydrogen-band tests and the full-range and downstream tests.
4. **`_NOT4`:** one helper for f_He, with before/after x_e numbers in the message.
5. **Documentation:** existing docstrings whose quoted numbers moved, and the CHANGELOG entry (`Mon D, YYYY` heading).

Commit messages go via `-F` files with explicit paths. Trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

**The spec and plan stay on the development branch but are left out of the upstream PR**, as recommended for #47.

## 6. Follow-on: the #45 probe (separate PR)

With helium and hydrogen smooth, test whether the κ̇ `stop_gradient` (`thermodynamics.py:807-840`, added in d127731 to stop a "~10⁸× spurious gradient from the recombination scan") is still needed. That blow-up may have been this same chatter.

- **Kept** only if, with the `stop_gradient` removed, gradients stay finite and AD matches FD for ω_b and ω_cdm at 3–5 cosmologies, within the bars above. It would then fix the ~65% thermodynamics part of #45.
- **Otherwise** the result is recorded and the `stop_gradient` stays.
- It is a separate PR in either case, so it cannot hold up §5.

The probe already saw a 2.4×10⁻⁴ jvp-vs-FD residual for the fix at fiducial. The κ̇ cut is a candidate cause; that is a hypothesis, untested.

## 7. Risks

- **Value shifts move C_ℓ.** The `fit_cl` hydrogen dip disappears and the helium window moves by up to ~6×10⁻³ in x_e. C_ℓ TT/EE/TE vs CLASS will be measured before and after, on `fit_cl` and `planck_cl`. The effect is not measured yet. At z ≈ 1560 the photon–baryon fluid is expected to stay tightly coupled even with x_e 30× too low, but that is untested.
- **Speed.** One scalar `jvp` and two RHS evaluations per sub-step, against Heun's two or RK4's four RHS evaluations. The probe saw ~35% more thermodynamics wall-clock at job level, which is not a per-solve benchmark. Handled by the benchmark and the 5% stop rule (§4.2).
- **Fixed switch redshifts** (2870, 1600) do not depend on parameters, as in CLASS, so no derivative passes through a switch location. Extremes in Y_He, N_eff and m_ν are in the test set.
- **Reference provenance:** CLASS-PT 3.3.4 stands in for stock CLASS 3.3.4 (§3).
- **The plan must reproduce the probe in-repo.** All numbers above come from throwaway copies (`/lustre/work/n2minh/clax-probes/helium/VE`). The implementation must re-measure them; any deviation from VE's formulas is a design change and comes back for approval.

## 8. Out of scope (filed separately)

- **The `Heswitch=1` vs `6` model difference.** clax implements RECFAST's Sobolev-singlet helium only. Against CLASS's default `Heswitch=6`, x_e differs by 1.5% at z = 2000 and 4.2% at z ≈ 1771; the visibility peak moves by ≤ 0.04 in z.
- **The hard Taylor switch in the T_b update** at x_e < 10⁻³ (`thermodynamics.py:589`): spike ratio 20 at z ≈ 624, on the 1500-point grid only.
- **c_s² vs CLASS c_b²:** 1.3% apart at z ≈ 632.

## 9. References

- Diagnosis: `/lustre/work/n2minh/clax-probes/rodas/diag/report.md`, jobs 22768–22820.
- Variant probe: `/lustre/work/n2minh/clax-probes/helium/`:
  - variants V0 and VA–VE, built by `scripts/make_variants.py`;
  - results in `out/`.
  - Key jobs:
    - 22933: CLASS references;
    - 22934–22937, 22952: thermodynamics probes at 20000 points;
    - 22948/22949/22953/22959/23034: step placement and h-scan;
    - 22973/22976/22977/23060: downstream reverse vs forward;
    - 23000: GPU;
    - 23055: 1500-point grid;
    - 22972: VE at 3000 points.
- Issues: smsharma/clax#48 (this fix), #30, #45 (§6), #46.
