# clax speed plan: stiff ODEs in JAX (2026-10-07)

Goal: make clax as fast and as accurate as CLASS, or better.
Status: research memo + proposed plan. Nothing implemented. Numbers marked
(est.) are estimates to be replaced by Phase 0 measurements.

## 0. Executive summary

- The problem is not primarily "stiff ODEs in JAX". On this Mac, CLASS v3.3.4
  computes lensed TT/TE/EE to l=2500 in **1.54 s** (1 thread) and in
  **12.4 s** with TCA/RSA/UFA switched off. The 8x is the cost of *not
  approximating*, inside a good implicit C solver. clax pays that 8x and then
  another ~40x on top from GPU loop structure.
- Cost model: `T_pert ~ N_chunks x max_k(steps) x t_step`. clax loses on all
  three factors: sequential chunks (`lax.map`), ~1400 lockstep steps at
  rtol 1e-6 with RSA damping and 5000-7000 without it (measured, Appendix
  A), and a GPU per-iteration cost of order 100 ms. Two-point fit of the
  cost model to the two GPU timings: `fit_cl` = 1 chunk x ~255 steps ->
  ~120 ms/iteration; `planck_cl` = 3 chunks (300 modes / cap 128) x ~1300
  steps -> ~100 ms/iteration. Same cost for state 61 vs ~250 and batch 100
  vs 128: the GPU iteration is pure latency, and `planck_cl`'s 13x over
  `fit_cl` is chunking (3x) times steps (5x). (Hypothesis: both GPU
  timings predate the April solver/norm changes.) Like for like, the same
  `fit_cl` pipeline runs **14x faster on this Mac's CPU** (perturbations
  2.2 s vs 30 s on V100; 5.1 s total cached); the solver's arithmetic floor
  is 0.16 ms per step per mode, so the GPU iteration sits ~600x above
  arithmetic.
- The best JAX codes today (ABCMB 6.7 s H100, DISCO-EB 3.5 s A100 for P(k)
  at rtol 1e-3) are still 2-5x slower than single-thread CLASS at the
  default Planck setting (l<=2500, 1.5 s). ABCMB does match single-core
  CLASS only at l_max=4000 with a massive neutrino, where CLASS itself
  takes 13 s. SymBoltz.jl (Julia, Rodas5P + analytic sparse
  Jacobian, approximation-free) reaches CLASS parity on CPU.
- Proposed design: CLASS-matched approximations (TCA/RSA/UFA/ncdm-fluid,
  l_max 12/10/17, CLASS k-grid), a batched Rosenbrock integrator on the
  *linear* system `y' = A(tau,k) y` (Jacobian = A, no `jacfwd`), run as a
  fixed-length `lax.scan` over a per-k mesh with no inner loops, forward-mode
  sensitivities sharing the LU, thermodynamics on CPU. Target (est.):
  **1-2 s forward, 3-5 s gradient on H100 at <=0.2%**. Stretch: fused
  Pallas/CUDA step kernel for another ~5-10x.
- Keep the approximation-free mode as a validation instrument: it is how
  clax can be *more* accurate than CLASS (quantify CLASS's own approximation
  error), not how it runs in production.

## 1. Success criteria

| Quantity | CLASS (this Mac, M4 Max) | clax today | Target |
|---|---|---|---|
| Lensed TT/TE/EE, l<=2500, massless nu, forward | 1.54 s (1 thr), 0.25 s (8 thr) | 487 s H100 (`planck_cl`, Feb 2026); `fit_cl` (1.5% accuracy): 34 s V100, **5.1 s this Mac CPU** (2026-10-07) | <=2 s GPU (Phase 2); <=0.3 s (Phase 3 stretch) |
| Same + 1 massive nu (0.06 eV) | 3.17 s / 0.47 s | not benchmarked | <=4 s |
| Gradient (6-8 cosmo params) | n/a (FD: 7-9x forward) | not benchmarked | <=3x forward (forward-mode) |
| P(k) linear, k<=1 h/Mpc | 0.20 s / 0.05 s | 0.77 s CPU, 15 modes, `Rodas5Batched` (Apr 2026) | <=0.3 s |
| Accuracy vs CLASS at fiducial | - | <0.2% TT/EE (planck_cl) | <=0.1% l=2..2500; EE l<30 after PR #18 |
| Accuracy across 10-cosmology suite | - | <0.5% TT (medium_cl) | <=0.3% |

CLASS timings measured with `classy` 3.3.4, `OMP_NUM_THREADS` = 1 and 8.
Note: `evolver=rk` (explicit) with approximations is 2.38 s; CLASS's own
implicit ndf15 is only 1.5x faster than explicit *because* TCA removes the
stiffness.

## 2. Problem space and boundary

### 2.1 Where clax's time goes (planck_cl, H100, Feb-Mar 2026; pre-Rodas5Batched)

| Stage | Time | Mechanism |
|---|---|---|
| Background | 1 s | sequential scan; fine |
| Thermodynamics | 53 s | 100000-point `lax.scan`, Heun/MB95 semi-implicit; latency-bound (~0.5 ms/step on GPU) |
| Perturbations | 401 s | 300 k-modes, Kvaerno5, rtol 1e-6, l_max 50 (state ~250 with 5 ncdm q-bins), max_steps 131072 |
| Harmonic | 33 s | `lax.scan` over 83 l; table Bessel brings it to 2.5 s |

`fit_cl` (V100): 0.5 + 1.5 + 30 + 2.4 = 34 s at 1.5% accuracy (l<=500);
100 k-modes in one chunk, l_max 17 (state 61), rtol 1e-3. The probe in
Appendix A shows the lockstep length is only ~255 steps (max over k), so
the V100 spends **~100 ms per lockstep iteration**, while one mode costs
**0.16 ms per step on this Mac's CPU** (Kvaerno5, `ForwardMode`,
`SaveAt(t1)`), and the whole `fit_cl` pipeline runs in 5.1 s cached on the
CPU (perturbations 2.2 s with 25 sequential chunks of 4 modes). The GPU
path is therefore latency-bound by two to three orders of magnitude, not
compute-bound. ABCMB's Kvaerno5 does ~2000 steps for 600 modes in 6.7 s
(~3 ms per iteration) on H100, so ~30x of that gap is clax-specific loop
structure, to be located in Phase 0.

### 2.2 Why a step costs 65 ms (code facts, `clax/perturbations.py`, `clax/ode.py`)

1. Per-k `diffrax.diffeqsolve` under `jax.vmap`, chunked with `lax.map`
   (<=128 modes/chunk on GPU, 4 on CPU) -> chunks run **sequentially**.
2. `Kvaerno5` (ESDIRK): Jacobian via `jacfwd`, then per stage a chord/Newton
   iteration with its own `while_loop` -> several host-device syncs and
   O(100s) of kernel launches per step.
3. `RecursiveCheckpointAdjoint` -> checkpointed bounded while loop
   (`max_steps` 131072 sets tree depth); `DirectAdjoint` was 1.7x slower.
4. `SaveAt(ts=tau_grid, fn=_extract_sources)` with 2000-5000 save points ->
   diffrax 0.7's `save_ts_impl` runs an `inner_while_loop` inside every
   step (`_integrate.py:438-462`), evaluating `_extract_sources` (a large
   function) at each saved time: ~8 host syncs and O(100s) of launches per
   step for 2000 points over ~255 steps.
5. `Rodas5Batched` (one LU + 8 back-substitutions, no inner loops) exists
   but is used only on the mPk path, never on the C_l path.

### 2.3 Why there are so many steps (est.; see Appendix A for the probe)

- Early times: baryon-photon tight coupling, kappa_dot ~ 10^3-10^4 Mpc^-1.
  clax has a sigmoid-blended TCA (CLASS `compromise_CLASS`), so this regime
  costs O(100) implicit steps. Not the problem.
- Late times: free-streaming photon/neutrino multipoles oscillate at
  frequency k. CLASS switches the hierarchy off (RSA at k*tau>45,
  optically thin; UFA for ur). clax instead *relaxes* the hierarchy toward
  the RSA targets at rate k and uses a filtered error norm (6 variables,
  DISCO-EB weights). Measured (Appendix A, l_max 17, Kvaerno5): at rtol
  1e-6 the step count is 412-1366 per mode with RSA damping and 2000-7100
  without it at k>=0.05; at rtol 1e-3 it is 108-255. So RSA damping is
  worth ~5x at Planck tolerance and the lockstep length at l_max 17 is
  ~1400, not 10^4. l_max 50 leaves the step count unchanged (1270) and
  instead multiplies the CPU per-step cost by 3.6x; `planck_cl`'s
  `max_steps=131072` is ~100x oversized (it sets checkpoint-tree depth and
  compile time, not the number of runtime iterations).
- 8-27% of attempted steps are rejected (PID with pcoeff 0.25, icoeff 0.8).
  A fixed mesh removes rejections entirely.
- l_max=50 (vs CLASS 12/10/17) was chosen to suppress truncation
  reflections *because* UFA/RSA were not CLASS-exact. It takes the state
  from 61 to 160 (250 with 5 ncdm q-bins), multiplies the LU cost by ~18x,
  and makes the hierarchy ring longer.
- `pt_k_max_cl = 1.0` Mpc^-1 vs CLASS `k_max_tau0_over_l_max = 1.8`
  -> k_max = 1.8*2500/14000 = 0.32 Mpc^-1 for l_max 2500. The highest k sets
  the lockstep step count; this alone is ~3x too many steps.

### 2.4 Boundary statement

Stiffness proper (the TC regime) is solved: L-stable Rosenbrock/ESDIRK with
TCA. The remaining cost is (i) the *number of sequential steps*, set by
approximations, k_max, l_max and tolerance; (ii) *GPU per-step latency*, set
by loop structure (inner loops, host syncs, chunking, checkpointing); (iii)
thermodynamics being run as a 10^5-step sequential scan on a GPU.
Any plan that does not attack (i) and (ii) together cannot reach CLASS.

Other code facts that constrain the design:
- The RHS is **linear in y** (all `where`/`maximum` gates depend only on
  background, tau, k). The Jacobian *is* the RHS matrix; `jacfwd` (d JVPs per
  step) recomputes it needlessly.
- Two `custom_vjp` sites block forward-mode AD: `_find_z_reio`
  (`thermodynamics.py:843`) and `shoot_fn` (`shooting.py:76`).
  `jax.jvp` through `custom_vjp` raises. `_solve_hydrogen_saha` is already
  `custom_jvp`.
- `Rodas5`/`Rodas5Batched` use `LocalLinearInterpolation` for dense output
  (DISCO-EB has the same TODO). With post-RSA steps larger than the source
  grid spacing, linear interpolation is a Planck-precision risk.
- `ncdm_fluid_approximation` is unstable (tests force `"none"`), so massive
  nu currently pays the full q-bin hierarchy (5x18 = 90 extra variables).
- TCA uses hard `jnp.where(is_tca > 0.5, ...)` switches in several terms
  (`perturbations.py` ~1065-1165): the switch time depends on parameters
  but AD ignores that dependence (same convention as CLASS).

## 3. Solution space: how other codes get their speed

| Code | Language / backend | Perturbation solver | Approximations | Batching | Reported cost | Notes |
|---|---|---|---|---|---|---|
| CLASS 3.3.4 | C, OpenMP | ndf15 (variable-order NDF/BDF), own sparse LU, Jacobian reuse across steps | TCA, RSA, UFA, ncdm fluid; per-k start time; l_max 12/10/17 | threads over k | 1.54 s / 0.25 s (1/8 thr, this Mac); 12.4 s without approximations | the oracle |
| CAMB | Fortran | explicit RK (dverk) | TCA, truncation, l_max cuts | threads over k | ~1-2 s | shows post-TCA system is explicit-friendly at h ~ 0.3/k |
| SymBoltz.jl (Sletmoen 2026, A&A 707 A128) | Julia, CPU | Rodas5P, symbolic analytic sparse Jacobian, KLU | **none** (l_max 16) | threads over k | P(k) 0.3 s, C_l 3.1 s (114 k, laptop) | 10x faster than CLASS-ndf15-without-approximations; "not yet fast enough for MCMC with perturbation-derived spectra" |
| DISCO-EB (Hahn, List, Porqueres 2024) | JAX | own GRKT4 / Rodas5 / Rodas5Batched (jacfwd + batched LU) | TCA-free; l_max 11, nqmax 3 | shared-mesh batches | 3.5 s A100, 4.2 s RTX3090 for P(k) to k=100 h/Mpc, 256 k, **rtol 1e-3** | not C_l; loose tolerance |
| ABCMB (Zhou, Giovanetti, Liu 2026) | JAX | diffrax Kvaerno5 in ln a, rtol 1e-5 (k<0.01) / 1e-4, max_steps 2048, `ForwardMode` adjoint + `jacfwd` | none (TCA deliberately dropped); l_max 12/10/17; 600 k | `vmap` over k (or `lax.scan`) | 6.7 s H100 (12.8 s with massive nu); grad 15 s / 32 s; 110 s on 1 CPU core | ~1 permille vs CLASS; no inference run |
| Bolt.jl | Julia | OrdinaryDiffEq, forward-mode AD | partial | threads | - | limited physics |
| gradsolve (Spurio Mancini 2026) | JAX + fused CUDA/Warp | per-thread fused RK/Rosenbrock23; record-and-replay adjoint | n/a | one trajectory per thread | 11-15x vs diffrax on d=3-9 non-stiff ensembles; parity on stiff at tight tol | state limited by 255 registers; not for d~60 |
| DiffEqGPU.jl (Utkarsh+ 2024) | Julia/CUDA | fused per-thread kernels | n/a | per thread | array-style (vmap) solvers 20-100x slower than fused kernels | same lesson |
| torchode | PyTorch | per-problem step control in a batch | n/a | masked lockstep | - | per-op dispatch bound |
| Pallas (JAX) | GPU kernels | fuse a scan body into one kernel | n/a | - | 5-10 us per kernel launch; tens of launches per iteration is the floor for XLA loops | path to Phase 3 |
| Magnus / exponential integrators | - | step limited by variation of A, not by ||A|| | n/a | - | "unexplored for Einstein-Boltzmann" (Agocs+ 2020) | research option for the free-streaming regime |
| Emulators (CosmoPower-JAX etc.) | - | - | - | - | ms per call | out of scope: not a solver |

Lessons:
1. **Approximations are worth 8x in CLASS and are the only way any code
   reaches ~1 s.** ABCMB gets 1 permille without RSA/UFA only by letting an
   L-stable solver numerically damp the late hierarchy at rtol 1e-4; it
   still needs ~2000 steps/mode and 6.7 s.
2. **Rosenbrock beats ESDIRK/BDF for this system** (SymBoltz, DISCO-EB,
   clax's own CPU benchmark): one LU, no Newton loop, L-stable.
3. **The GPU floor is launches per step, not flops.** d=60 dense batched LU
   is ~70 kflop per mode; 600 modes x 2000 steps is ~10^11 flop (<1 s even
   at low efficiency). What costs is 100-1000 launches/step and host syncs.
4. **Forward-mode AD is the right gradient for <=10 cosmological
   parameters** (ABCMB: 2.2x forward). Reverse mode through adaptive loops
   needs checkpointing and is the slow path.
5. **Per-thread fused kernels are a 10x beyond XLA**, but only for small
   states; for d~60 they need shared memory and a custom kernel.

## 4. Synthesis: proposed design

### 4.0 Key insight (added after review discussion)

Neither adaptivity nor approximations block differentiability:
- `jax.lax.while_loop` is forward-differentiable; diffrax's `ForwardMode`
  adjoint + `jacfwd` differentiates an *adaptive* Kvaerno5 solve (ABCMB,
  Table 4). clax already uses adaptive steps (PID) today.
- The derivative with the accepted mesh held fixed is the standard
  discrete adjoint (diffrax, gradsolve, DISCO-EB); the dropped mesh-motion
  term is O(tolerance). HMC stays exact with inexact gradients as long as
  the accept step uses the exact energy; gradient error only costs
  acceptance.
- Approximation switches are measure-zero, approximation-accuracy-sized
  jumps; clax already sigmoid-blends TCA and ncdm-fluid and sigmoid-gates
  RSA. The cost of approximations is accuracy control, not AD.

What *does* block the GPU is control flow that depends on the state:
accept/reject, Newton convergence, save loops. The Einstein-Boltzmann
system is **linear and homogeneous** in the perturbations, `y' = A(tau,k) y`,
so for any one-step method the step is a matrix `y_{n+1} = M_n y_n` with
`M_n` depending only on `(tau_n, h_n, k, theta)`, never on `y`. Therefore:
1. all `M_n` can be formed in parallel (batched over k *and* n);
2. the solve is a linear recurrence -> `jax.lax.associative_scan` gives all
   prefix products `P_n = M_n ... M_1` in ~log2(N) parallel rounds (the
   S4/S5/Mamba parallel-scan trick applied to a Boltzmann hierarchy);
3. the mesh is *data*: a smooth function of background timescales, so the
   model is smooth in theta with no frozen-mesh convention; approximations
   are masks on a fixed state layout; `jacfwd` flows through everything.
Cost (est., d=61, 600 modes, N=1400): forming `M_n` is 8 x 2d^3 ~ 3.5 Mflop
per mode-step (60x the vector method) -> ~3e12 flop total, 1-3 s on V100,
<1 s on H100; memory for all `M_n` is 17 GB, so block the scan (sequential
over ~10 blocks of ~140 steps: ~80 sequential rounds instead of 1400).
Trades flops for latency: a win on GPU, a loss on CPU. A 2-day spike on
the real `A(tau,k)` decides it (Phase 0 item 6).

Further R&D on the same structure: an exponential/Magnus one-step map
`M_n = exp(h A(tau_{n+1/2}))` is exact for frozen `A`, so in the
free-streaming regime (A ~ k M_0 + slow terms) the step is limited by the
variation of `A`, not by `k` -- an approximation-free alternative to RSA
within the truncated hierarchy (truncation ringing at l_max remains).


Cost model and what each piece attacks:

```
T_pert = N_chunks * max_k(N_steps) * t_step
         ^ (a)      ^ (b)             ^ (c)
```

### 4.1 Perturbations

(a) **One batch, sorted k, few chunks.** With no checkpoint memory
(forward-mode), run all modes in one `vmap`, or 2-4 chunks of similar k so
each chunk's lockstep length matches its members' needs.

(b) **CLASS-matched step budget.**
- l_max 12/10/17 (photon T / pol / ur), ncdm 17 with a *stable* fluid
  approximation; CLASS k-grid (`k_step_sub/super`, k_max = 1.8 l_max/tau_0)
  -> ~600 modes to k ~0.32 Mpc^-1 for l<=2500.
- RSA as CLASS does it (hierarchy frozen/replaced by algebraic values at
  k*tau>45 and kappa_dot/aH<5), UFA for ur, ncdm fluid at tau*k>31. Smooth
  the switches over a short window for AD; validate the window's effect.
- Per-k start time (`start_small_k_at_tau_c_over_tau_h`,
  `start_large_k_at_tau_h_over_tau_k`), as CLASS and ABCMB do, to stop
  low-k modes paying for the stiff epoch.
- Expected (est.): max_k N_steps ~ 1500-3000 at Planck precision.

(c) **Per-step cost: batched Rosenbrock on the linear system, no inner
loops.**
- Assemble `A(tau,k)` (d x d, batched over k) directly from the existing RHS
  (the RHS is linear in y; a `jax.jacfwd` once per step is the fallback).
  Each Rodas5 stage is then `W k_i = A(t_i) v_i + h gamma_i dA/dt y + ...`,
  i.e. one batched LU of `W = I/(gamma h) - A` and 8 batched triangular
  solves + matvecs -> O(20-50) kernels per step (est. 0.3-1 ms on H100).
- **Fixed-length `lax.scan` with a per-k mesh `h_i(k)`** instead of an
  adaptive while loop: no accept/reject, no host sync, no padding, outputs
  at mesh points (no `SaveAt` loop, no interpolation). Two mesh sources,
  both to be tried:
  1. *Heuristic mesh*: CLASS's own sampling rule, `dtau = eps * min(tau_h,
     tau_k [before RSA], tau_c-based)`, with eps calibrated once against
     the adaptive solver.
  2. *Recorded mesh* (gradsolve's record-and-replay): run the adaptive
     `Rodas5Batched` once at the fiducial, store accepted steps, replay for
     nearby cosmologies. Reuse across an MCMC with periodic re-recording.
  In both cases keep the embedded 4th-order error estimate in the replay as
  a *monitor* (max over steps/modes), assert it against rtol, and trigger
  re-record / mesh refinement when violated. Dense output: implement
  Rodas5's continuous extension if any output falls between mesh points.
- Rosenbrock order reduction in the TC regime and truncation ringing are
  the two numerical risks; both are measurable against the adaptive solve.

(d) **Gradients.** `jax.jacfwd` over the cosmological parameters (<=10)
through the scan; forward sensitivities reuse the same LU per step (one
factorization, 1+p solves per stage). Prerequisite: convert `_find_z_reio`
and `shoot_fn` from `custom_vjp` to `custom_jvp` (implicit-function
theorem, same math). Reverse mode stays available via `jax.checkpoint` on
the scan body for many-parameter use cases.

(e) **Precision policy.** Two presets: `class_cl` (production, CLASS-matched
approximations and grids) and `exact_cl` (approximation-free, tight
tolerance, l_max 50) used only to measure the approximation error. Float64
throughout; avoid L40S (fp64 1/64 rate); V100/A100/H100.

### 4.2 Thermodynamics

Run on the CPU device (tiny, sequential; 20 ms est.) via a separate jit
whose outputs are moved to the GPU, or replace the 10^5-point Heun/MB95
scan with a ~2000-point Rodas scan of the RECFAST ODE (3 stiff variables,
closed-form 3x3 solve). Later: HyRec-2 port (ABCMB's HyRex: 50 ms CPU) to
remove the RECFAST EE -0.15% systematic.

### 4.3 Harmonic and lensing

Replace the `lax.scan` over 83 l-values with batched contractions over
(l, k_fine, tau) using the tabulated j_l; est. <=0.2 s. Fix TT l>1200
k-sampling (hybrid linear/log grid) at the same time.

### 4.4 What this buys (est.)

| Stage | Today (H100) | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|---|
| Thermo | 53 s | 0.5 s (fewer points) / 0.05 s (CPU) | 0.05 s | 0.05 s |
| Perturbations | 401 s | 5-10 s | 1-2 s | 0.1-0.3 s |
| Harmonic + lensing | 33 s | 2.5 s | 0.3 s | 0.3 s |
| Forward total | 487 s | ~10 s | ~2 s | ~0.5 s |
| Gradient (8 params) | - | ~25 s | ~5 s | ~1.5 s |

NUTS on CMB (2-3e4 gradients/chain): ~1-2 GPU-days per chain at Phase 2,
~10 GPU-hours at Phase 3.

## 5. Plan with gates

**Phase 0 - measure (1 week, Bridges-2 V100/H100).** Re-baseline; the 487 s
predates `Rodas5Batched`.
1. Step histogram per k for `fit_cl`/`planck_cl` (Appendix A gives the CPU
   version).
2. Per-step wall time and kernel/host-sync counts (nsys or XLA profiler)
   for Kvaerno5 vs Rodas5 vs Rodas5Batched; `SaveAt(ts, fn)` vs
   `SaveAt(steps=True)`; `RecursiveCheckpointAdjoint` vs `DirectAdjoint` vs
   `ForwardMode`.
3. Chunk policy: `lax.map` chunks vs one `vmap`.
4. Thermo on CPU vs GPU.
5. A toy fixed-mesh Rodas5 scan on the real RHS (no approximations changed)
   to measure the achievable t_step on GPU.
6. Matrix-step + `associative_scan` spike on the real `A(tau,k)` (section
   4.0): time and memory vs the sequential scan, `fit_cl` case.
Gate: a table that assigns the 401 s to (a), (b), (c) with measurements.

**Phase 1 - XLA-level fixes, low risk (2-3 weeks).**
1. Convert the two `custom_vjp` sites to `custom_jvp` first (`jax.jacfwd`
   of the full pipeline fails today), then `ForwardMode` adjoint + `jacfwd`
   gradients.
2. One `vmap` (or 1-4 sorted-k chunks) instead of `lax.map` over <=128-mode
   chunks: worth ~3x on `planck_cl` by the two-point fit alone.
   `Rodas5Batched` on the C_l path only if Phase 0 shows it wins at matched
   achieved accuracy (it takes 2.5x more steps than Kvaerno5 at rtol 1e-6).
3. `class_cl` preset: l_max 12/10/17, CLASS k-grid and k_max rule, rtol
   split 1e-5/1e-4, per-k start time.
4. Thermo: fewer points + higher-order stepper, or CPU device.
5. Harmonic as batched contractions.
Gate: Planck-accuracy (<=0.2%) forward <=10 s on H100; gradient <=3x.

**Phase 2 - integrator redesign (4-8 weeks).**
1. `A(tau,k)` assembly + batched linear one-step method (Rodas5 tableau
   from `rosenbrock.py`, or an ESDIRK stage loop with a fixed iteration
   count if Phase 0 favours Kvaerno5's step count).
2. Fixed-length scan with heuristic and recorded meshes; embedded error
   monitor; Rodas5 dense output.
3. CLASS-exact RSA/UFA and a stable ncdm fluid approximation, smoothed.
4. Forward sensitivities sharing the LU.
Gate: <=2 s forward, <=5 s gradient on H100; <=0.1% at fiducial and <=0.3%
across the 10-cosmology suite; `exact_cl` - `class_cl` difference documented.

**Phase 3 - stretch (only if Phase 2 is still latency-bound).**
1. Pallas (Mosaic GPU) or CUDA `jax.ffi` fused per-k Rosenbrock step
   (d<=64 LU in shared memory; 28 KB/mode).
2. CPU parity path: `shard_map` over cores + analytic sparse Jacobian.
3. HyRec-2 port.
4. R&D: exponential/Magnus integrator for the free-streaming regime
   (steps limited by dA/dt, not k).

## 6. Risks

| Risk | Mitigation |
|---|---|
| Fixed mesh under-resolves some cosmology in the NUTS prior volume | embedded error monitor + automatic re-record; conservative eps; test at +-5 sigma and massive nu/w0wa |
| Smoothed switches (TCA/RSA/UFA) shift results at the 0.1% level | measure window dependence; CLASS's hard switches are the limit |
| Rosenbrock order reduction / ringing with l_max 12 | compare to `exact_cl`; keep l_max as a preset knob |
| Linear interpolation of dense output | mesh hits output times; implement Rodas5 continuous extension |
| `custom_vjp`->`custom_jvp` conversions change gradients | finite-difference tests already exist for both sites |
| cuBLAS batched LU for (600, 60, 60) is launch-heavy in XLA | profile; fall back to `jnp.linalg.solve` with stacked RHS or a Pallas LU |
| Massive nu doubles cost (as in CLASS) | stable ncdm fluid approximation; nqmax 3-5 |

## 7. Non-goals

Emulators; float32; `gradsolve` as a dependency (state too large for its
kernels); explicit solvers without TCA; custom kernels before the XLA-level
fixes are exhausted; reproducing CLASS's ndf15.

## 8. Decisions for the user

1. Production mode = CLASS-matched approximations (`class_cl`), with the
   approximation-free mode kept as the accuracy instrument. Agree?
2. GPU-first (Phase 2 target 2 s on H100) vs CPU parity (Phase 3 item 2).
3. Is massive nu in the first target, or after the ncdm fluid fix?
4. Phase 0 runs on Bridges-2 (SU budget ~1925); this Mac has no GPU.

## Appendix A - CPU step-count probe (this Mac, M4 Max, 2026-10-07)

Single k-mode `diffeqsolve` on clax's `_perturbation_rhs` with the `fit_cl`
preset modified as listed (`ncdm_q_size=0`, `pt_k_max_cl=1`,
`max_steps=131072`), `ForwardMode` adjoint, `SaveAt(t1)`, filtered PID
norm. Columns: attempted steps (accepted+rejected). Time is one mode on one
CPU; per-step cost is time/steps.

| Config (solver, rtol, l_max, RSA damping) | n_eq | steps at k = 1e-4 / 1e-3 / 1e-2 / 0.05 / 0.1 / 0.3 / 0.5 / 1 Mpc^-1 | max | ms/step | time at k=1 |
|---|---|---|---|---|---|
| Kvaerno5, 1e-3, 17, on (= `fit_cl`) | 61 | 108 / 246 / 255 / 142 / 135 / 156 / 183 / 204 | 255 | 0.16 | 0.03 s |
| Kvaerno5, 1e-6, 17, on | 61 | 412 / 963 / 974 / 781 / 881 / 1019 / 1084 / 1366 | 1366 | 0.16 | 0.21 s |
| Kvaerno5, 1e-6, 17, **off** | 61 | 415 / 968 / 1092 / 1952 / 3514 / **7106** / 5006 / 2543 | 7106 | 0.15 | 0.40 s |
| Kvaerno5, 1e-6, **50**, on | 160 | 385 / 952 / 992 / 774 / 889 / 1010 / 1099 / 1270 | 1270 | 0.57 | 0.73 s |
| Rodas5, 1e-6, 17, on | 61 | 483 / 1049 / 1110 / 1201 / 1580 / 2279 / 2736 / 3474 | 3474 | 0.08 | 0.27 s |
| Rodas5, **1e-4**, 17, on | 61 | 238 / 499 / 503 / 367 / 382 / 465 / 516 / 566 | 566 | 0.08 | 0.05 s |

All runs `ok=True` (no `max_steps` hit). Rejections are 8-27% of attempts
(Kvaerno5 ~20%, Rodas5 at rtol 1e-6 ~10%). Compile ~8 s per config.

Readings:
- RSA damping is worth up to 5x in steps at k>=0.1 and tight tolerance
  (7106 -> 1019 at k=0.3). Without it the k=0.3 mode, not k=1, is the worst.
- l_max 50 does not change the step count (1270 vs 1366); on CPU it
  multiplies the per-step cost by 3.6x (state 160 vs 61). On the GPU the
  two-point fit in section 0 says the iteration cost is state-independent,
  so l_max 50 costs nothing there today and will only start to matter once
  the latency is gone.
- Rodas5 takes 2.5x more steps than Kvaerno5 at rtol 1e-6 (cause not
  identified; both have 4th-order embedded estimates) but each step is 2x
  cheaper, so the CPU time is similar. On a latency-bound GPU 2.5x more
  iterations is a real risk: Rodas5 only wins if its per-iteration latency
  (no inner Newton loop) is >=2.5x lower. Phase 0 must compare the two at
  matched *achieved* accuracy vs CLASS, not matched rtol. At rtol 1e-4
  (ABCMB's large-k setting) Rodas5's lockstep length is 566.
- Scale check: `planck_cl`-like settings cost 0.73 s per mode on one CPU
  core here, so 300 modes serial is ~220 s on **one core** -- the same
  order as the 401 s measured on an H100. The GPU currently delivers one
  CPU core's worth of throughput.
- CPU parity estimate for a CLASS-matched preset (~600 modes, Rodas5,
  rtol 1e-4, l_max 17): 0.05 s per mode -> ~30 s serial on one core, ~2-4 s
  with `shard_map` over this Mac's 16 cores (est.).

## References

- CLASS II (Blas, Lesgourgues, Tram 2011) - ndf15, TCA/RSA/UFA.
- SymBoltz.jl: arXiv:2509.24740, A&A 707, A128 (2026).
- DISCO-EB: arXiv:2311.03291; repo `ohahn/DISCO-EB` (BENCHMARKS.md,
  `ode_integrators_stiff.py`).
- ABCMB: arXiv:2602.15104; repo `TonyZhou729/ABCMB` (`perturbations.py`).
- gradsolve: arXiv:2609.02876 (library), arXiv:2609.28458 (applications).
- DiffEqGPU.jl: Utkarsh et al. 2024, CMAME.
- torchode: Lienen & Guennemann 2022.
- Agocs et al. 2020, arXiv:1907.11638 ("Beyond RKWKB") - Magnus unexplored
  for Einstein-Boltzmann.
- candl: arXiv:2401.13433 (differentiable CMB likelihoods; NUTS via emulators).
