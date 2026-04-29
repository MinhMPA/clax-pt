# Paper 1: AI for Physics Workshop (ICML 2026)

## Title

**clax: A Differentiable Boltzmann Solver and One-Loop Perturbation Theory Pipeline in JAX**

## Framing

Technical methods paper. The contribution is the tool itself and what it enables: gradient-based cosmological inference from first-principles theory predictions. Not a story about AI collaboration — that goes to Paper 2.

---

## Abstract sketch (150 words)

We present clax, a fully differentiable reimplementation of the CLASS cosmological Boltzmann solver in JAX, extended with one-loop EFT perturbation theory (clax-pt) for galaxy clustering. The pipeline computes CMB angular power spectra (C_l^TT/EE/TE/BB, lensed), matter power spectra P(k), and redshift-space galaxy power spectrum multipoles P_gg(k, ell=0,2,4) — all end-to-end differentiable via automatic differentiation. Against CLASS v3.3.4, clax achieves sub-0.2% C_l accuracy at l=20-1200. Against CLASS-PT, clax-pt achieves sub-percent accuracy on all monopole and quadrupole spectra at k<0.3 h/Mpc. The full pipeline from cosmological parameters to P_gg(k) runs in 34 seconds on a V100 GPU with HMC-ready accuracy, and exact gradients are verified to 0.03% against finite differences. We discuss the design choices that make differentiability compatible with the numerical stiffness of the Einstein-Boltzmann system, and demonstrate gradient-based parameter inference as a downstream application.

---

## Section outline

### 1. Introduction (1 page)

- Cosmological inference requires repeated evaluation of theory predictions (Boltzmann solver + perturbation theory) inside MCMC or other samplers. Current tools (CLASS, CAMB) are not differentiable — gradients must be estimated by finite differences, which scales poorly with dimension.
- Differentiable simulators have transformed other fields (fluid dynamics, molecular dynamics, climate). Cosmology is ripe for the same treatment: the Boltzmann equations are a well-defined ODE system, and the one-loop integrals are closed-form.
- Prior work: Bolt.jl (Julia, Li+2023), CosmoPower (emulator, Spurio Mancini+2022), DISCO-EB (Hahn+2023). clax differs from emulators (no training data, exact theory) and from Bolt.jl (JAX ecosystem, GPU-native, one-loop PT extension).
- Contributions: (1) full Boltzmann pipeline in JAX with sub-0.2% C_l accuracy; (2) one-loop EFT galaxy power spectrum with sub-percent accuracy vs CLASS-PT; (3) end-to-end AD gradients verified to 0.03%; (4) 34s GPU wall-clock for HMC-ready accuracy.

### 2. Method (3 pages)

#### 2.1 Architecture overview
- Sequential pipeline of pure functions: background → thermodynamics → perturbations → primordial → transfer → harmonic → lensing.
- CosmoParams (JAX-traced) vs PrecisionParams (static). No branching on traced values.
- Frozen dataclass result types registered as JAX pytrees.

#### 2.2 Einstein-Boltzmann system
- Full scalar Boltzmann hierarchy in synchronous gauge (Ma & Bertschinger 1995).
- TCA (tight-coupling approximation) with smooth jnp.where switching (no hard if/else).
- RSA (radiation streaming approximation) for post-recombination efficiency.
- Dark energy fluid perturbations (CPL w0-wa) in the Einstein constraint equations.
- Stiff ODE solver: Kvaerno5 (ESDIRK, default) or Rodas5 (Rosenbrock, batched for GPU).
- DISCO-EB-style filtered PID step control with k-dependent weighting.

#### 2.3 One-loop perturbation theory (clax-pt)
- FFTLog decomposition of P22 and P13 integrals (McEwen+2016, Fang+2017).
- Precomputed M22/M13 kernel matrices (from CLASS-PT).
- IR resummation via DST with odd/even spline mode removal.
- EFT bias expansion: b1, b2, bs2, b3nl, counterterms, stochastic contributions.
- RSD multipoles via Gauss-Legendre quadrature over mu with anisotropic BAO damping.
- End-to-end differentiable: jax.grad through FFTLog + IR resummation + bias assembly.

#### 2.4 Differentiability strategy
- Inner ODE: analytical Jacobian for the stiff solver; RecursiveCheckpointAdjoint for reverse-mode AD.
- custom_vjp for the shooting method (theta_s → H0 via implicit differentiation).
- Key challenge: ODE solver must not branch on traced cosmological parameters.

### 3. Validation (1.5 pages)

#### 3.1 Boltzmann solver accuracy
- Table: C_l^TT/EE accuracy vs CLASS at select multipoles (the README table).
- Multi-cosmology validation: 10 LCDM parameter points, all sub-0.5% TT.
- P(k) accuracy: sub-1% at k=0.01-0.2, including w0-wa dark energy.

#### 3.2 One-loop PT accuracy
- Table: all 9 spectra vs CLASS-PT (the accuracy table from CHANGELOG).
- Gradient accuracy: AD vs finite differences, 0.03% agreement.

#### 3.3 Performance
- Table: fit_cl (34s V100) vs planck_cl (487s H100) presets.
- Breakdown by module: background, thermodynamics, perturbations, harmonic.
- Rodas5Batched: 1.37x speedup on CPU via shared-timestep batching.

### 4. Application: gradient-based inference (1 page)

- Demonstrate Fisher forecast or simple HMC posterior using exact gradients from clax-pt.
- Compare wall-clock per effective sample against finite-difference MCMC.
- Show scaling with number of parameters (AD: O(1) gradient cost vs O(d) for FD).
- Or: demonstrate sensitivity analysis / Fisher matrix computation from exact Jacobian.

### 5. Discussion and outlook (0.5 pages)

- Current limitations: TT accuracy at l>1200, float64 requirement, thermodynamics (RECFAST vs HyRec).
- Future: float32 mixed-precision, bispectrum, field-level inference integration.
- Broader impact: differentiable theory codes as a new paradigm for cosmological analysis.

---

## Figures

1. Pipeline architecture diagram (params → modules → outputs, with AD arrows)
2. C_l accuracy plot: clax vs CLASS residuals across multipoles
3. P_gg(k) accuracy plot: clax-pt vs CLASS-PT for monopole, quadrupole, hexadecapole
4. Gradient accuracy: AD vs FD scatter plot
5. Performance scaling or inference demonstration figure

---

## Key selling points for this workshop

- **Topic 2 (High-fidelity surrogate simulators)**: clax is not an emulator — it is an exact differentiable simulator. This is a stronger statement than neural surrogates: no training data, no domain shift, exact gradients.
- **Topic 3 (Inverse problems and differentiable inference)**: The entire motivation is gradient-based inference. The 34s wall-clock makes HMC feasible for full-shape analysis.
- **Distinguished from emulators**: CosmoPower, cosmopower-jax, etc. approximate the mapping. clax computes it from first principles. The accuracy is bounded by numerical tolerances, not training set coverage.
