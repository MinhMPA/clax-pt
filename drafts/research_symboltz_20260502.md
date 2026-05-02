# SymBoltz.jl Deep Research: What Can clax Learn?

**Date**: 2026-05-02
**Confidence**: High (peer-reviewed A&A paper + source code inspection)

---

## Executive Summary

SymBoltz.jl's "approximation-free" claim is **genuine and verified**: it solves the full Boltzmann hierarchy at all times without TCA, RSA, or UFA switching, using Rodas5P (a Rosenbrock method) as its default implicit solver. The paper demonstrates this is **faster than CLASS's implicit solver** (>10× with Rodas5P vs CLASS's ndf15), though still slower than CLASS with its full approximation suite + explicit solver for C_l. The key insight for clax: **Rosenbrock methods are the right choice for approximation-free Boltzmann solving**, and clax has already adopted this with Rodas5/Rodas5Batched.

---

## Claim Verification

### "Approximation-free" — TRUE

SymBoltz.jl solves the full photon, neutrino, and polarization hierarchies at all times. No TCA/RSA/UFA switching occurs during integration. The initial conditions encode tight-coupling results (standard practice) but the equations are integrated in full from the start.

Evidence: `photons.jl` shows no switching logic — just the raw hierarchy equations with Thomson scattering terms. The paper states explicitly: "Full equations are solved at all times without tight-coupling, ultrarelativistic fluid and radiation-streaming approximations."

clax status: **Same approach.** clax also integrates the full hierarchy without hard TCA/RSA/UFA switching. clax uses smooth RSA relaxation damping (a soft switch) post-recombination, which is technically not "approximation-free" in the SymBoltz sense, but is physically motivated and preserves AD stability.

### "Adaptive implicit ODE solvers" — TRUE

Default solver: **Rodas5P** (5th-order Rosenbrock, same family as clax's Rodas5). The paper benchmarks several solvers:
- Rodas5P: "most efficient and stable on both background and perturbations"
- KenCarp4 (ESDIRK, similar to Kvaerno5): "perform well with much fewer time steps"
- BDF methods: "slowest"

This matches clax's experience: Rosenbrock > ESDIRK for this problem.

### "Differentiable" — TRUE, with caveats

Forward-mode AD works for Fisher forecasts (derivatives of spectra w.r.t. parameters). However: **"Differentiable computations are not yet fast enough for MCMCs with perturbation-derived spectra."** Only background-level MCMCs are currently feasible.

clax status: **Ahead here.** clax has reverse-mode AD working through the full pipeline (RecursiveCheckpointAdjoint + custom VJPs), and the 34s/step fit_cl preset is HMC-ready. SymBoltz.jl acknowledges "more work is needed for fast reverse-mode automatic differentiation."

---

## Performance Comparison

| Metric | SymBoltz.jl | clax | CLASS |
|--------|------------|------|-------|
| P(k) solve time | ~0.5s (Rodas5P, CPU) | ~30s (Kvaerno5 V100) / TBD (Rodas5Batched) | ~3-5s (ndf15 implicit) |
| C_l solve time | "slower than CLASS" (not quantified) | 34s total pipeline (V100) | ~10s (CPU, with approx) |
| Accuracy vs CLASS | ~0.1% | <0.2% lensed C_l | — |
| Reverse-mode AD | Not yet fast enough for MCMC | Working, HMC-ready | N/A |
| GPU support | No | Yes | No |
| Approximation-free | Yes (strict) | Mostly (smooth RSA damping) | No (TCA+RSA+UFA) |
| State vector (default) | ~127 eqs (lmax=10, 4 nu momenta) | 59 eqs (fit_cl, lmax=17) | varies |

---

## What clax Can Learn

### 1. Rodas5P vs Rodas5 — clax already has this

SymBoltz.jl uses Rodas5P (the "P" variant from Steinebach 2020, improved stability). clax uses the original Rodas5 (Di Marzo 1993). The difference is minor — Rodas5P has slightly better error estimation for DAE systems. Not worth switching unless stability issues arise.

### 2. Sparse Jacobians — clax should investigate

SymBoltz.jl uses **sparse Jacobians with KLU factorization** for the perturbation system. The Einstein-Boltzmann Jacobian is approximately block-diagonal (photon hierarchy couples mainly to itself + metric). Exploiting this sparsity could reduce the O(n²) Jacobian and O(n³) LU costs substantially.

- SymBoltz: uses `KLUFactorization` for sparse systems
- clax: uses dense `jax.jacfwd` + dense `lu_factor`
- Potential speedup: For n=59 (fit_cl), sparse may not help much. For n=152 (planck_cl with ncdm), sparse could give 2-3× on the LU step.

**Verdict**: Worth investigating for planck_cl preset but probably not impactful for fit_cl.

### 3. Symbolic Jacobian generation — not applicable to JAX

SymBoltz.jl generates **symbolic Jacobians** via ModelingToolkit, which are then compiled to fast code. This is faster than AD-based Jacobians because the symbolic simplification eliminates redundant terms.

clax uses `jax.jacfwd` (forward-mode AD) for the Jacobian, which is exact but doesn't benefit from symbolic simplification. JAX doesn't have an equivalent to ModelingToolkit's symbolic pipeline.

**Verdict**: Not actionable for clax. The JAX ecosystem doesn't support this pattern.

### 4. SymBoltz's P(k) at 0.5s on CPU is striking

SymBoltz.jl computes P(k) in ~0.5s on CPU with Rodas5P. clax's perturbation solve takes ~30s on V100 (Kvaerno5). Even accounting for GPU compilation overhead, this is a large gap.

Possible explanations:
- SymBoltz may use fewer k-modes (not clear from the paper)
- Julia's compilation model (ahead-of-time specialization) vs JAX's JIT may differ
- SymBoltz's symbolic Jacobian avoids the per-step AD cost
- SymBoltz's sparse LU is faster for large state vectors

The key question is whether Rodas5Batched on GPU closes this gap. If clax's perturbation solve drops to ~2-5s with Rodas5Batched on GPU (matching DISCO-EB's architecture), the comparison becomes GPU-favorable.

### 5. C_l is NOT SymBoltz's strength

The paper admits: "SymBoltz currently computes C_l slower than CAMB and CLASS." The C_l computation (line-of-sight integration, Bessel functions) is where SymBoltz is less optimized. clax has invested heavily here (table-based Bessel, source interpolation, harmonic in 2.4s).

**Verdict**: clax is likely ahead on C_l performance.

### 6. Reverse-mode AD is clax's differentiator

SymBoltz.jl's paper says: "More work is needed for fast reverse-mode automatic differentiation of scalar loss functions." This means gradient-based MCMC (HMC, NUTS) is not yet practical with SymBoltz.jl for perturbation-derived spectra.

clax has this working: RecursiveCheckpointAdjoint through the full pipeline, custom VJPs for shooting and Saha, verified to 0.15% vs FD. This is arguably clax's strongest unique feature.

---

## Summary: What's Left to Do?

| Optimization | SymBoltz has it? | clax has it? | Impact | Effort |
|---|---|---|---|---|
| Rosenbrock solver | ✓ (Rodas5P) | ✓ (Rodas5 + Rodas5Batched) | Already done | — |
| Sparse Jacobian | ✓ (KLU) | ✗ (dense jacfwd) | Medium for planck_cl | Medium |
| Symbolic Jacobian | ✓ (ModelingToolkit) | ✗ (not possible in JAX) | N/A | N/A |
| GPU batching | ✗ | ✓ (Rodas5Batched) | clax advantage | Already done |
| Reverse-mode AD | ✗ (forward only) | ✓ (RecursiveCheckpoint) | clax advantage | Already done |
| Table-based Bessel | N/A | ✓ | clax advantage | Already done |

**Bottom line**: clax has already adopted the most impactful optimization from the approximation-free approach (Rosenbrock solvers) and has features SymBoltz lacks (GPU, reverse-mode AD, batched solver). The remaining gap is sparse Jacobians, which would help the high-accuracy planck_cl preset but not the HMC-ready fit_cl preset.
