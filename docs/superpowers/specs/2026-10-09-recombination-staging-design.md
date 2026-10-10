# Smooth recombination stepping: helium and hydrogen follow CLASS's Saha → ODE staging

**Date:** Oct 9, 2026 (reviewed Oct 10, 2026)
**Issue:** smsharma/clax#48 (helium-recombination derivative spikes); related #30, #45
**Branch:** `fix/recombination-staging`, from upstream main `4d084b5`
**Status:** design approved section by section; spec reviewed against the interview; test seams confirmed (S0–S4)
**Vocabulary:**
- **primary oracle:** CLASS, the code clax mirrors line by line.
- **code-accuracy gate:** a pass/fail tolerance between two codes, set by numerical accuracy.

## Problem Statement

A cosmologist running NUTS (or a Fisher forecast, a MAP fit, or a derivative-trained emulator) on clax needs gradients that are precise and smooth. Today any gradient that passes through the thermodynamics module carries a random error, with respect to h, ω_b, ω_cdm, Y_He, N_eff or T_cmb. The error ranges from per-mille to order unity, and the user cannot see it: the values look fine.

**Measured, on unmodified upstream:**
- forward and reverse AD disagree by up to **113%** on the production gradient path;
- CPU and GPU derivatives differ by a **factor of 21**;
- df/dh **jumps by 6%** along a smooth line in h;
- the same derivative changes by **17%** when the perturbation solver groups k-modes differently.

On `fit_cl`, the NUTS preset, the values themselves are also wrong: x_e drops to **0.035 at z ≈ 1559**, where the truth is ≈ 1.0.

**Cause.**
- clax integrates the stiff HeII → HeI and hydrogen Peebles equations with **explicit steps clipped to [0, 1]**, through exactly the redshift ranges where the primary oracle does not integrate at all: CLASS uses Saha there.
- **Helium (z ≈ 3000–3500):** clipped explicit Heun sub-steps, plus a hard `xHe > 0.999 → C_He = 1` switch that has no CLASS counterpart. x_e wiggles by only 0.35%, but its derivative spikes to |d ln x_e/dh| ≈ 5×10¹² against smooth values ≈ 10⁻².
- **Hydrogen (z ≈ 1550–1750):** a clipped explicit RK4, sigmoid-blended with Saha at z = 1600. It runs past RK4's stability limit (h·J = −3.2 at z = 1600 against −2.79), so x_e zigzags (by 18% at z ≈ 1613 on the default grid), and on the coarse `fit_cl` grid it collapses.
- **How the spikes reach observables:** through c_s² k² δ_b in the baryon velocity equation. Whether a solver stage lands on a spike depends on step placement (platform batching, solver, compilation), so the error looks random.

**Consequences.**
- For NUTS: lost efficiency (energy errors, smaller steps, divergences), not bias, since accept/reject uses values. The exception is `fit_cl`'s x_e collapse, which is a value error.
- For Fisher, MAP/Laplace and derivative-trained emulators: bias.

## Solution

**Recombination follows the primary oracle's staging.**
- Helium stays on Saha down to z = 2870; hydrogen stays on Saha down to z = 1600.
- Only below those redshifts are the Peebles equations integrated.
- The handoff is CLASS's smoothstep blend.
- The integration uses one L-stable, second-order linearly implicit step with no clip.

The helium-per-hydrogen number ratio is made consistent with CLASS.

From the user's point of view:
- values agree with CLASS to within the 0.1% code-accuracy gate;
- derivatives match converged finite differences;
- forward and reverse agree;
- CPU and GPU agree to round-off;
- derivatives no longer depend on how the solver places its steps;
- all of this holds across cosmologies, including extremes.

This is the variant ("VE") that met all four of the user's criteria at 12 cosmologies, in a probe of six variants. A linearly implicit step **without** staging ("A") fails outright: it overshoots to x_e = −988.

## User Stories

1. As a cosmologist running NUTS on `fit_cl`, I want gradients w.r.t. h, ω_b, ω_cdm, Y_He and N_eff that do not depend on where the ODE solver places its steps, so that my chains do not lose efficiency to random force kicks.
2. As a NUTS user, I want my CPU and GPU runs to give the same gradients to round-off, so that I can debug on CPU and sample on GPU.
3. As a NUTS user, I want forward- and reverse-mode gradients to agree, so that I can pick whichever mode is cheaper for my parameter count without changing the answer.
4. As a NUTS user, I want df/dθ to vary smoothly as θ moves, so that the leapfrog integrator's energy error stays small and my step size stays large.
5. As a user of the `fit_cl` preset, I want x_e to be correct near z ≈ 1600, so that my C_ℓ are not computed from a recombination history that briefly collapses to x_e ≈ 0.03.
6. As a Fisher-forecast user, I want derivatives that match converged finite differences, so that my forecast errors are not biased.
7. As someone training a derivative-informed emulator on clax, I want derivatives that are smooth across parameter space, so that I do not train on noise.
8. As a user varying Y_He or N_eff, I want the gradients to be as clean as at fiducial, so that extended models are as reliable as ΛCDM.
9. As a user with massive neutrinos, I want the recombination fix to hold at m_ν = 0.15 eV, so that νΛCDM inference is not a special case.
10. As a user comparing against CLASS, I want x_e, T_b and the visibility to agree with the primary oracle (same helium physics) within 0.1%, so that values are not traded for smoothness.
11. As a user, I want the visibility-peak redshift to agree with CLASS to better than 0.01, so that the sound horizon and the acoustic scale are unaffected.
12. As a user, I want no hard switches or clips left in the recombination step, so that no derivative passes through a discontinuity.
13. As a user, I want the stage-switch redshifts fixed (parameter-independent, as in CLASS), so that no derivative passes through a switch location.
14. As a user of `planck_cl`, I want my C_ℓ accuracy against CLASS to be unchanged or better, so that the fix costs me nothing in science-grade runs.
15. As a user of `fit_cl`, I want the thermodynamics solve to stay fast (total `fit_cl` time within ~5%), so that the fix does not slow my chains.
16. As a user of the production gradient entry `solve_background_and_thermo`, I want its reverse-mode gradients to agree with forward mode, so that ADR 0001's stability rule delivers what it promises.
17. As a user taking derivatives of C_ℓ directly, I want C_ℓ-level gradients to be independent of step placement and smooth along parameter lines, so that the fix reaches the observable I actually sample.
18. As a maintainer, I want every equation in the new recombination code traced to a CLASS file:line, so that I can review it term by term against the primary oracle.
19. As a maintainer, I want the stiff step tested directly on a scalar ODE with a known answer (second order, L-stable, no overshoot), so that a regression in the integrator is caught at the lowest level.
20. As a maintainer, I want tests that fail on today's code and pass on the fix, each on at least one grid or cosmology, so that I know the tests can see the bug.
21. As a maintainer, I want the CLASS reference data and its provenance (build, settings, Y_He) committed, so that the tests need no CLASS installation at runtime.
22. As a maintainer, I want every commit on the branch to pass the whole suite, so that the history stays bisectable when it goes upstream.
23. As a maintainer, I want the decision recorded in an ADR, with ADR 0001's "forward mode is exact" premise annotated, so that the next person knows why the stepping changed and what #48 refined.
24. As a reviewer, I want each existing test's documented numbers updated where the fix moves them, so that docstrings do not quote stale values.
25. As a reviewer, I want the before/after benchmark and the C_ℓ shift vs CLASS reported in the PR, so that I can weigh accuracy against cost.
26. As the person following up on #45, I want this fix to land first and the κ̇ `stop_gradient` question tested separately afterwards, so that each change is judged on its own evidence.

## Implementation Decisions

**Module.** Only `clax/thermodynamics.py` changes in behaviour. The recombination `lax.scan` keeps its structure; within each grid step, the two stiff species change how they are advanced. Every new block carries CLASS file:line comments. clax line numbers are not cited here because they go stale.

**D1. One shared stiff step, `_ros2_scalar_step` (internal).**
- An L-stable two-stage linearly implicit Rosenbrock (ROS2) step for one fraction x, γ = 1 + 1/√2. From the prototype, decision-rich:
  ```
  W  = 1 − γ h J
  k1 = h f(s_j, x_j) / W
  k2 = [h f(s_{j+1}, x_j + k1) − 2 k1] / W
  x_{j+1} = x_j + (3/2) k1 + (1/2) k2
  ```
- J = ∂f/∂x comes from a scalar `jax.jvp` with the environment held fixed: n_H, H, T_b, T_r, z, and x_e as an input, as probed. ROS2 is second order for any J.
- 10 sub-steps per grid step. The environment is interpolated per sub-step: ln a linear; ln H and T_b linear in ln a.
- **No clip.**

**D2. Helium, HeII → HeI (CLASS regime "He2" then ODE).**
- **Saha for z > 2870** (`thermodynamics.c:4053-4067`): rhs = 4 (C_R T_b)^{3/2} exp(−T_HeI/T_b)/n_H, with C_R = 2π m_e k_B/h² and T_HeI = h c L_He1,ion/k_B ≈ 285,335 K (`wrap_recfast.c:78`, `thermodynamics.c:823`).
  - From the prototype: x_HeII^Saha = 2 rhs / [(1 + rhs) + √((1 + rhs)² + 4 f_He rhs)].
  - This is CLASS's root, rearranged to avoid cancellation.
- **The ODE state** is set to the Saha value while z > 2870. Below, the Peebles equation (`wrap_recfast.c:349`, in d/d ln a) advances it with D1. The initial condition is the analytic value, as at `thermodynamics.c:2059-2065`.
- **Output:** x_HeII = w · x_ode + (1 − w) · x_Saha, with w = 3s² − 2s³ and s = clip((2870 − z)/100, 0, 1).
  - This is CLASS's `f2` (`thermodynamics.h:49`, applied at `thermodynamics.c:2973-2987`, Δz = 50 at `precisions.h:202`).
  - w is exactly 0 or 1 outside z ∈ [2770, 2870].
- **Rate function:**
  - delete the `xHe > 0.999 → C_He = 1` switch; C_He = (1/A + Λ)/(1/A + Λ + R_up) is continuous as x → 1, measured;
  - keep the x < 10⁻¹⁵ guard (CLASS `x_He_trigger_small`);
  - escape probability p_s = −expm1(−τ)/τ, with 1 − τ/2 + τ²/6 − τ³/24 for τ < 10⁻⁴.
- **The HeIII → HeII Saha blend at z ≈ 5000 is unchanged;** it was measured smooth.

**D3. Hydrogen (CLASS regime "H" then ODE), the same pattern at z = 1600.**
- Saha via the existing hydrogen Saha solver with the x_HeII correction (`thermodynamics.c:4070-4087`) for z > 1600. Initial condition as at `thermodynamics.c:2093`.
- The Peebles equation advanced with D1 below 1600, **replacing the clipped RK4**.
- The smoothstep blend over z = 1600 → 1500 (`precisions.h:204-205`) replaces the sigmoid.
- RECFAST's C = 1 for x_H ≥ 0.995 (`wrap_recfast.c:163-171`) stays; it is in CLASS.

**D4. Helium-to-hydrogen number ratio (its own commit).**
- One helper gives f_He = Y/(`_NOT4` (1 − Y)), with `_NOT4` = 3.9715 (CLASS `thermodynamics.h:707`).
- Every helium-per-hydrogen ratio in the module uses it: the HeIII Saha solver, the initial x_e, the x_HeII contribution to hydrogen Saha, the helium rate's f_He, and every x_e sum.
- Today one site already uses `_NOT4` while the others use exactly 4.
- Measured effect: a 6.4×10⁻⁴ offset in x_e at z = 2870–5000 (1.2×10⁻³ above z = 5000 at Y_He = 0.28).

**D5. Unchanged:** the background; the κ̇/visibility construction, including its `stop_gradient` until the follow-on in Further Notes; reionisation; the T_b update; the `custom_vjp` entry `solve_background_and_thermo` and its forward-mode basis (ADR 0001); the HeIII stage.

**D6. Decision record.**
- A new **ADR 0003**, "Recombination follows CLASS's Saha → ODE staging with an L-stable step", records the probe evidence: why staging, why ROS2, and why not "A".
- **ADR 0001** gets a dated note. Its premise "forward mode is exact" holds only when solver stages miss the recombination chatter (#48); this fix removes the chatter.

**D7. Reference data and provenance.**
- CLASS in RECFAST mode with `recfast_Heswitch = 1`, the helium physics clax implements. x_e and T_b on a ~2000-point z grid.
- 10 cosmologies: the 5 of `COSMOLOGY_GRID_LCDM`, plus Y_He = 0.22 / 0.28, N_eff = 2 / 4 (via N_ur) and m_ν = 0.15 eV. Y_He is passed explicitly, never via BBN.
- Generated by a new function in the CLASS-reference generator script, run in the `classpt` env.
- No stock `class_public-3.3.4` tree exists on this machine. The CLASS-PT tree (`_VERSION_` v3.3.4) stands in; it reproduces the existing fiducial RECFAST reference to 8.4×10⁻⁶ in x_e. The provenance file records the build and settings.

**D8. The downstream functional is defined in the spec, not imported from a parked branch.**
- f = Σ_k δ_m(k, τ₀)², from `perturbations_solve_mpk`.
- Preset: `fast_cl` (whose `pt_k_max_cl` is 0.15 Mpc⁻¹) with `pt_ode_solver="rodas5"`, `pt_ode_rtol = pt_ode_atol = 1e-4` and `ode_max_steps = 16384`, exactly the probe's preset.
- For the C_ℓ seam (S4): F = Σ_{ℓ=30}^{1200} ln C_ℓ^TT at the `fit_cl` preset.

**D9. Delivery.** Branch `fix/recombination-staging` from `4d084b5`; **every commit passes the full suite**. Commits:
1. reference data and generator;
2. D1 + D2, with the S0 tests and the helium-window S1 tests;
3. D3, with the hydrogen-band and full-range S1 tests, and S2–S4;
4. D4;
5. D6 (ADRs), docstrings whose quoted numbers moved, and the CHANGELOG (`Mon D, YYYY` heading).

Commit messages via `-F` files with explicit paths; trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. The spec and plan stay on the branch but are left out of the upstream PR.

**D10. Cost stop rule.**
- Benchmark on a V100, before and after: the thermodynamics solve alone, and the full `fit_cl` C_ℓ wall-clock.
- If total `fit_cl` time grows by more than ~5%, stop and report before continuing.

## Testing Decisions

**What makes a good test here.**
- Test observable behaviour at the highest seam that shows it: values against the primary oracle, derivatives against converged finite differences, invariance under step placement. Do not test the scan's internals.
- The one exception, S0, is deliberate and was requested: the stiff step is tested directly on a problem with a known answer, so an integrator regression is caught at the lowest level.

**Rules that apply to every test.**
- **Red first:** each new test is shown failing on the unfixed code, on at least one grid or cosmology, before its fix lands. Example: x_e at z 800–1550 passes on the 20000-point grid today but fails on `fit_cl`'s 3000-point grid. The red evidence (job IDs, numbers) goes in the commit message.
- **Multi-cosmology:** S1–S4 run at 3–10 cosmologies, fiducial only under `--fast`. S0 is cosmology-independent numerics, and says so in its docstring.
- **Bars are code-accuracy gates**, never loosened. Each existing docstring that quotes a number this change moves gets the newly measured value.

**Seams and tests.**

**S0: the stiff step (new, lowest seam).** A scalar stiff relaxation, dx/ds = −λ(x − x_eq(s)), with a known solution; CPU, fast. Tests:
- second-order convergence: the error falls ~4× when h halves, for λh ≪ 1;
- L-stability: at λh = 10⁶ the step lands on x_eq without overshoot or oscillation;
- monotone approach with no excursion past 1 for a Peebles-like rate near equilibrium;
- the step's `jax.jvp` w.r.t. λ and x_eq is finite and matches FD.

Prior art: the toy-ODE tests in `tests/test_rosenbrock.py`.

**S1: `thermodynamics_solve` output tables and their forward tangents.** CPU, seconds each; all 10 cosmologies in full mode; each test on the default 20000-point grid and `fit_cl`'s 3000-point grid. Prior art: `tests/test_thermodynamics.py` (`TestThermoGradients` and its jvp-vs-FD helper).

| Test | Bar | Today | Fix (probe VE) |
|---|---|---|---|
| x_e vs CLASS, z 2870–5000 | ≤ 1×10⁻³ | 6.3×10⁻³ | 6.4×10⁻⁴ |
| x_e vs CLASS, z 1590–1650 | ≤ 1×10⁻³ | 0.18 | 2.8×10⁻⁴ |
| x_e vs CLASS, z 800–1550 | ≤ 1×10⁻³ | 5.5×10⁻⁴ (0.97 on `fit_cl`) | 1.2×10⁻⁴ (4.4×10⁻⁴ on `fit_cl`) |
| Visibility-peak shift vs CLASS | < 0.01 in z (20000 points); < 0.03 (3000 points) | 0.040 (0.256 on `fit_cl`) | 0.001 (0.013 on `fit_cl`) |
| Values smooth: max \|Δ² x_e\|, z 1500–6000 | < 3×10⁻⁵ (20000 points); < 1.33×10⁻³ (3000 points) | 0.36 | 8.4×10⁻⁶ (20000 points) |
| Spike ratio†, Q ∈ {x_e, c_s², T_b}, θ ∈ {h, ω_b, ω_cdm, Y_He, N_eff}, z 300–6000 | < 10 | 9.5×10¹⁴ | ≤ 1.19 (20000); ≤ 2.55 (3000) |
| Tangents vs converged FD at sampled z‡ | ≤ 1×10⁻³ | up to 7.4 | ≤ 1×10⁻³ |
| CPU vs GPU tangents (GPU-only§) | < 10⁻¹⁰ | factor 21 | 2.7×10⁻¹³ |

The two per-grid bars were corrected while planning (Oct 10, 2026):
- **Visibility peak:** grid spacing limits peak location on the coarse grid, where the fix itself measures 0.013. The 3000-point bar is 2× that, rounded up.
- **Values smooth:** second differences scale as the square of the grid spacing, so the 3000-point bar is 3×10⁻⁵ × (20000/3000)² = 1.33×10⁻³. The probe did not measure it; the plan measures it first, and if the fix misses, it is reported as a design question.

† max over grid points of |d ln Q/dθ| ÷ its median over ±25 neighbours, median floored at 10⁻⁸.
‡ only at redshifts where FD's own two steps (relative 10⁻³, 3×10⁻³) agree to better than 10⁻⁴. Skipped points are counted in the printout. The probe's 1.1×10⁻³ for z ≤ 2000 came from FD-limited points.
§ in a module that does **not** pin JAX to CPU at import, run in its own pytest session in the GPU validation job, and skipped without a GPU. Several existing modules pin CPU at import, which pins a whole session.

**S2: δ_m(today) functional through the matter solve (D8).** Slow; CPU; 3 cosmologies (fiducial, ω_b +20%, Y_He 0.28). Prior art: `tests/test_pk_tau_endpoint.py` (the ω_cdm growth gradient).

| Test | Bar | Today | Fix |
|---|---|---|---|
| df/dh independent of step placement (k-chunk 8 vs 62, jit vs no-jit) | spread < 3×10⁻⁴ | 0.17 | 6.3×10⁻⁵ |
| Native reverse vs forward | < 3×10⁻⁴ | up to 0.18 | ≤ 7.7×10⁻⁵ |
| df/dh along h = 0.62–0.72 (11 points) | cubic-fit residual < 3×10⁻⁴ | 6% jump | 1.0×10⁻⁴ |

**S3: the production gradient entry `solve_background_and_thermo` (ADR 0001's `custom_vjp`).** Slow; same functional and cosmologies as S2. The test: production reverse vs forward, bar < 3×10⁻⁴. Today up to 1.13 (CPU). For the fix this path was **not** measured separately in the variant probe; the diagnosis's helium-cut diagnostic copy gave 1.2×10⁻⁵, which is indicative only, and the plan measures it. Prior art: the ADR 0001 tests in `tests/test_thermodynamics.py`.

**S4: C_ℓ at the `fit_cl` preset (D8).** Slow; GPU-preferred (`fit_cl` takes ~34 s on a V100); 3 cosmologies; θ ∈ {h, ω_b}.
- **Tests:**
  - forward dF/dθ independent of step placement (k-chunk settings, jit vs no-jit);
  - production reverse vs forward;
  - dF/dh smooth along h (7 points).
- **Bars are provisional at 3×10⁻⁴.** The C_ℓ-level behaviour was **not** measured by the probe. The plan's first S4 task measures today's code and the fix. If the fix misses 3×10⁻⁴, stop and report it as a design question; the bar is not widened to fit.
- Prior art: the C_ℓ gradient tests in the EPT and lensing suites.

**Regression.**
- Every existing test passes at its current bar: thermodynamics, C_ℓ TT/TE/EE vs CLASS (`fit_cl` and `planck_cl`), P(k), EPT and all gradient tests.
- C_ℓ vs CLASS is measured before and after on both presets and reported.
- A failing existing bar is investigated, never loosened.

## Out of Scope

- **Porting CLASS's default helium physics (`Heswitch = 6`, Doppler and triplet terms).** clax implements `Heswitch = 1`. Against CLASS's default, x_e differs by 1.5% at z = 2000 and 4.2% at z ≈ 1771, with the visibility peak within 0.04 in z. This is a model difference, filed separately.
- **The hard Taylor switch in the T_b update at x_e < 10⁻³:** spike ratio 20 at z ≈ 624, on a 1500-point grid only. Filed separately.
- **c_s² vs CLASS c_b²,** 1.3% apart at z ≈ 632. Filed separately.
- **Removing the κ̇ `stop_gradient` (#45):** a follow-on, see Further Notes.
- **The parked `rodas5` W′ branch (`fix/rodas5-division-free`):** not the fix for #48. Its step-budget change for `TestRosenbrockPk` may be offered separately.
- **Publishing this spec to an issue tracker:** the project's tracker vocabulary is not set up. #48 is the issue, and this file is the spec.

## Further Notes

**#45 follow-on (separate PR).**
- With recombination smooth, test whether the κ̇ `stop_gradient` (added in d127731 against a "~10⁸× spurious gradient from the recombination scan") is still needed; that blow-up may have been this chatter.
- **Keep the removal** only if gradients stay finite and AD matches FD for ω_b and ω_cdm at 3–5 cosmologies within the bars above.
- **Otherwise** record the result and leave the `stop_gradient`.
- The probe saw a 2.4×10⁻⁴ jvp-vs-FD residual for the fix at fiducial. The κ̇ cut is a candidate cause; that is a hypothesis, untested.

**Risks.**
- **C_ℓ shifts:** values move by up to ~6×10⁻³ in x_e in the helium window, and much more around z ≈ 1560 on `fit_cl`. The effect on C_ℓ is unmeasured. Tight coupling is expected to hold at z ≈ 1560, but that is untested.
- **Speed:** one scalar `jvp` and two RHS evaluations per sub-step. The probe saw ~35% more thermodynamics wall-clock at job level, which is not a per-solve number. D10 governs.
- **The plan must reproduce the probe in-repo.** All numbers above come from throwaway copies. Any deviation from VE's formulas is a design change and comes back for approval.

**Evidence.**
- Diagnosis: `/lustre/work/n2minh/clax-probes/rodas/diag/report.md` (jobs 22768–22820).
- Variant probe: `/lustre/work/n2minh/clax-probes/helium/` (variants V0 and VA–VE; results in `out/`). Key jobs:
  - 22933: CLASS references;
  - 22934–22937, 22952: thermodynamics probes at 20000 points;
  - 22948/22949/22953/22959/23034: step placement and h-scan;
  - 22973/22976/22977/23060: reverse vs forward;
  - 23000: GPU;
  - 23055: 1500 points;
  - 22972: VE at 3000 points.

**Related:** smsharma/clax#48 (this fix), #30, #45, #46; ADR 0001.
