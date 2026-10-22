# Claude Code task: Critical investigation + first pilot of a fast differentiable GPU matter-transfer solver

You are acting as a computational cosmologist, numerical-analysis researcher, JAX/CUDA engineer, and skeptical independent reviewer. Your job is to **investigate first, benchmark second, implement only a bounded pilot third**. Do not assume that the proposed architecture is good merely because it appears promising.

**Companion brief:** `matter_transfer_gpu_pilot_spec.html`. Read it first. Treat its numbers as *proposed targets*, not observations.

## Goal

Develop and compare two candidate routes to a fast, differentiable **linear matter-transfer / matter-power solver** for cosmological inference. Start from the physics already implemented in Clax and try to remove perturbation-integration overhead **without changing physical approximations** in the first experiment.

For fixed background cosmology and Fourier mode, the linearized Einstein–Boltzmann system typically has the form

\[
\frac{d\mathbf y_k}{dx}=A(k,x;\theta)\mathbf y_k + s(k,x;\theta),\quad x=\ln a.
\]

Its Jacobian is potentially sparse/structured (metric–fluid core; streaming multipole chains). Verify the actual Clax system—including the tight-coupling (TCA) and radiation-streaming (RSA) branches—rather than assuming the entire RHS is a constant-shape homogeneous linear system.

The eventual target is an accurate JAX-callable \(T_c(k,z),T_b(k,z),T_{cb}(k,z)\) and \(P_{cb}^{\rm lin}(k,z)\) with practical gradients in \(\omega_b,\omega_c,h\). Maintain a consistent synchronous-gauge normalization, primordial curvature convention and Mpc\(^{-1}\) wavenumbers. Amplitude and tilt derivatives can initially be analytic.

## Repositories to inspect (pin commits)

1. **Clax:** <https://github.com/smsharma/clax> — inspect `clax/perturbations.py`, `clax/rosenbrock.py`, the `compute_pk_table` pathway, initial conditions, background, thermodynamics, tests, benchmark scripts, developer documentation and open issues.
2. **modax:** <https://github.com/lawrenceberry/modax> — read <https://lawrenceberry.github.io/modax/usage/sparse/> and <https://lawrenceberry.github.io/modax/usage/gradients/>; inspect requirements of Numba-CUDA-MLIR RHS functions, GPU sparse Rodas5P, no-pivot factorization, sensitivity support and what variables the JAX FFI genuinely differentiates.
3. **DISCO-EB:** <https://github.com/ohahn/DISCO-EB> — inspect its perturbation state ordering, sparsity and benchmarks as an independent reference; **observe GPL-3.0 compatibility** before reusing any code.
4. **CLASS:** <https://github.com/lesgourg/class_public> — use as the high-accuracy physics reference where available.

Use real source inspection; do not invent APIs, file paths, Jacobian structure, benchmarks or successful gradients. Cite code locations and record repository SHAs. Read project-specific `AGENTS.md` / `CLAUDE.md` and test/format conventions before editing. Do not modify upstream originals; work in a separate experiment branch or sandbox.

## Scientific scope and restrictions

**In the first pilot:** flat ΛCDM with massless neutrinos (but *retain massless relativistic neutrino perturbations*); existing photon polarization, TCA/RSA, baryon pressure, background/thermodynamic treatment and adiabatic initial conditions; \(z=0,1,2,3\); initially 128–256 logarithmically spaced modes over \(10^{-4}\lesssim k\lesssim1\ \mathrm{Mpc}^{-1}\), plus smaller/larger mode-count tests.

**Do not initially implement:** CMB \(C_\ell\)'s, line-of-sight projection, massive-neutrino momentum quadrature, modified gravity, nonlinear \(P(k)\), new recombination prescriptions, new tight-coupling approximations, emulators, or full HMC. Existing controlled approximations in the reference physics are permissible; silently *changing* them is not.

Keep separately measured timings for (a) the perturbation stage with fixed/pregenerated coefficient tables and (b) the complete background → thermodynamics → transfer → \(P(k,z)\) pipeline. A kernel-only speedup is not an end-to-end result.

## Two competing pilots

### Option A — GPU-native Clax → modax

Test whether compiling the **same RHS physics** into modax's Numba-CUDA-MLIR interface and using per-mode adaptive Rodas5P with sparse LU improves realistic \(k\)-ensemble throughput.

1. Determine whether Clax's operations and regime switches can be compiled faithfully into a modax device-callable RHS; identify substitutions, if any, and prove numerical equivalence.
2. Identify the exact state vector, Jacobian pattern, symbolic fill, condition numbers and any risk of modax's non-pivoted sparse LU; the sparsity mask **must include the union of all actual nonzero entries across regimes and tested parameters**.
3. Start with a single representative mode and *frozen* background/thermodynamic tables, then a 32/128/256/1024-mode benchmark with matched error tolerance.
4. Critically analyze shared coefficient interpolation: memory layout, parameter passing, host/device copies, temporal interpolation error, differentiability and compilation overhead.
5. Verify JAX gradients end-to-end. Modax may differentiate explicit `params` and `y0` but **do not assume** that differentiation through upstream Clax arrays, coefficient-table construction, closure state, time grids, or FFI custom calls works automatically. Provide a mathematically and programmatically correct chain rule, or state precisely why the option currently fails.
6. Audit any TCA/RSA switching surfaces for value and derivative continuity; do not disguise nonsmooth gradients with finite-difference tuning.

**Core risk:** costly cross-toolchain port, derivative discontinuities or missing coefficient sensitivities, poor occupancy for a few hundred modes, and a sparse factorization whose speed gain disappears at realistic sizes.

### Option B — Structure-aware pure-JAX Clax

Test the lower-risk alternative before developing a completely new integrator:

1. Profile Clax's `Rodas5Batched`: RHS evaluation, `jax.jacfwd` of time/state, batched dense LU and triangular solves, interpolation, accepted/rejected steps, shared step-size control and GPU kernel dispatch.
2. Construct an analytic/explicit linear Jacobian (or optimized Jacobian-vector representation) **from the exact existing RHS**; verify it numerically against JAX AD over representative times, modes and TCA/RSA states. First keep the existing dense LU to isolate Jacobian-assembly improvements.
3. Characterize sparsity and factorization fill. Only then test block-banded / block-sparse / Schur-complement solves if they are implementable efficiently in JAX. Optimized dense batched GPU LU may beat generic JAX sparse routines: **benchmark instead of presuming**.
4. Investigate whether Clax's shared adaptive step size across a batch wastes work when \(k\) modes have heterogeneous stiffness, and whether grouping modes or changing the step controller pays off.
5. Keep native JAX differentiation and verify gradient smoothness; quantify compile times and memory.

**Core risk:** JAX GPU execution and tiny matrix solves remain bottlenecked by launches/loop structure; numerical sparsity does not guarantee faster wall time.

## Required investigation questions (answer explicitly)

- Is the state equation genuinely linear in `y` in each active regime? Are there inhomogeneous sources or nonlinear stabilization terms? Is the Jacobian effectively time dependent but state independent?
- What are the state dimensions under each precision preset? Does the `P(k)` pathway avoid CMB outputs but still carry photon/polarization and neutrino hierarchies?
- What fraction of time is spent on thermodynamics versus perturbations versus transfer/interpolation? How much is due to dense LU, Jacobian construction, step rejection, and shared stepping?
- Is the Jacobian pattern robust to changes of regime, parameters, neutrino treatment and the number of multipoles? What is the actual sparse LU fill count, and is non-pivoted LU safe?
- Does modax compile the needed RHS and interpolation primitives? Exactly how would upstream \(\partial A/\partial\theta\), initial-condition derivatives and dependence of output-time mappings enter its sensitivity equations?
- Would a pure-JAX analytic Jacobian, better state ordering or mode grouping yield most of the benefit with far less engineering?
- Are the two routes mutually exclusive, or could Option B's Jacobian work be reused in Option A?
- What is the simplest falsifiable experiment that decides which route merits further investment?

## Benchmark and validation contract

Make a standalone, reproducible harness with pinned versions, hardware identification, and a fixed cosmology suite. Establish the baseline **before** altering equations or integrators. If suitable hardware or dependencies are unavailable, do static analysis plus executable CPU-side math tests, and explicitly label GPU benchmarking as not performed; do not fabricate measurements.

1. **Fiducial and parameter sweep:** fiducial flat ΛCDM plus ≥30 declared samples in a physically sensible parameter box; test 32, 128, 256 and 1024 modes.
2. **Physics outputs:** \(T_c,T_b,T_{cb}\), BAO wiggles, transfer phases, \(P_{cb}^{\rm lin}(k,z)\), and optionally growth rate; distinguish raw transfer discrepancies from interpolation errors. Compare against unchanged Clax and a high-accuracy CLASS reference.
3. **Numerical convergence:** vary tolerances, multipole truncations, time-table spacing, k-grid density and stage-switch thresholds **only as diagnostic sweeps**; never report a speedup from loosened accuracy as equivalent performance.
4. **Proposed science gate:** aim for ≤0.2% error in \(P_{cb}\) across the agreed domain (except appropriately flagged near-zero transfer-function regions); make the pilot at least as accurate as its Clax baseline. Near zeros, compare absolute/scaled error.
5. **AD gate:** compare \(\partial P/\partial\omega_b\), \(\partial P/\partial\omega_c\), \(\partial P/\partial h\) to independent high-accuracy finite-difference sweeps and existing Clax JAX derivatives. Propose dimensionally normalized derivative-error metrics; test gradient noise under small parameter perturbations.
6. **Performance metrics:** JIT warm-up separated; `block_until_ready()` on relevant outputs; forward and `jax.value_and_grad` timings; wall-time distribution, compile cost, max GPU memory, k-mode scaling, step counts, rejected steps, LU/Jacobian costs, coefficient table staging and GPU utilization.
7. **Go/no-go:** prefer a route if it preserves physics and full derivatives and achieves a reproducible ≥2× **perturbation-stage** speedup at matched accuracy on representative mode counts. State end-to-end gain separately; if neither hits it, identify the next real bottleneck and recommend whether to abandon, revise, or combine the routes.

The aspirational \(~0.2\!\text{–}\!0.5\) s forward and \(~0.5\!\text{–}\!1\) s value+gradient targets are **not** baseline claims and **not** sufficient justification to weaken accuracy.

## Working procedure

**Phase 0 — Review before edits.** Inspect sources, pin SHAs, verify actual existing APIs, make a call graph from Clax's public `P(k)` entry point to RHS/Jacobian/solver, identify profiling instrumentation, and write `docs/feasibility_review.md`. Make an explicit recommendation about whether to attempt A, B, both as micro-spikes, or a different controlled alternative. Challenge this brief's assumptions and correct mistakes.

**Phase 1 — Baseline and minimum reproducible tests.** Add minimal reproducible scripts and unit tests without refactoring physics. Include a single-mode RHS/Jacobian consistency diagnostic, Clax `P(k)` and CLASS comparison where available, a GPU/CPU availability diagnostic and synchronized timings. Record raw results, commands and environment.

**Phase 2 — Bounded implementation spikes.** If feasible, first isolate B's analytic Jacobian change using Clax's existing stepper (the easiest apples-to-apples comparison). Independently test A's smallest Numba-compatible RHS and sparse one-mode / small ensemble solve. *Do not* undertake a full package rewrite without convincing correctness and performance results. For difficult dependencies, make a small reproducer and document blockers.

**Phase 3 — Compare and decide.** Put evidence, timings, errors, failed attempts and architectural recommendation in `docs/pilot_decision.md`. Distinguish observed results from estimates and speculation. Give a prioritized set of follow-up changes: recombination/thermodynamics, asymptotic TCA reduction, late-time growth integration, or whole-pipeline AD only if justified by profiling.

## Expected deliverables

- `docs/feasibility_review.md`: code-grounded architecture, correct mathematical formulation, precise feasibility risks, comparison A vs B.
- `docs/benchmark_plan.md`: reproducible experiments, metrics, tolerances, parameter set and test matrix.
- `experiments/` (or an appropriate repo-local folder): minimal benchmark harness and exact commands; tests validating RHS/Jacobian/output equivalence; *bounded* working prototype if executable.
- `docs/pilot_decision.md`: empirical table of physics errors, gradient errors, perturbation and end-to-end timings, GPU resources and a defensible go/no-go recommendation.
- A concise final report: what was inspected, changed, proven, benchmarked, blocked and recommended next. **Never claim tests ran when they did not.**

**Operating principle:** optimize the numerical representation and execution of *the same Einstein–Boltzmann dynamics* before changing the physical model. Preserve provenance and distinguish a fast perturbation kernel from a fast differentiable matter-power pipeline.
