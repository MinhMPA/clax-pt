# Report for the linrec agent: modax, gradsolve and the TAM formulation (2026-10-10)

Audience: the agent executing `docs/superpowers/plans/2026-10-07-linrec-gpu-spike.md` on branch `exp/linrec-gpu-spike`. Spec: `docs/stiff_ode_speed_plan_2026-10-07.md`.

Basis: a full read of the modax paper (Berry, Handley, Hahn, Schöneberg, arXiv:2609.35661), the two gradsolve papers (Spurio Mancini, arXiv:2609.02876 "lib" and arXiv:2609.28458 "astro"), the TAM derivation paper (Kamionkowski & Caldwell, arXiv:2609.36000), and both codebases as checked out on this machine: modax 0.0.4 at `/Users/nguyenmn/modax`, gradsolve 0.2.1 at `/Users/nguyenmn/gradsolve`. File:line references below point into those checkouts.

## 1. Verdict in four lines

1. Neither library becomes a dependency of linrec. The plan's global constraint ("No new dependencies. No gradsolve.") stands; the same applies to modax.
2. modax is the competing design for the same latency problem linrec attacks, not a component of it: a fused per-k adaptive Rodas5P kernel with a compiled sparse LU, already run on DISCO-EB's 50-variable Einstein-Boltzmann system. Treat it as the external bar for gate G3, and as a possible kernel for the matrix-forming stage if Phase 3 is ever reached.
3. gradsolve contributes one idea (record-and-replay) that Tasks 4 and 5 already implement, plus a handful of copyable code patterns listed in section 4. Its engines are unusable here (dimension limits, lockstep record, dense `jacfwd` Jacobian).
4. The TAM formulation removes the hierarchy and with it the Markov structure the associative scan relies on. If the project moves to TAM, keep linrec's parallel assembly, mesh-as-data and a-posteriori monitor; replace the scan by a blocked lower-triangular solve. Section 5.

## 2. What the two libraries are, in the terms the plan uses

| | modax 0.0.4 | gradsolve 0.2.1 |
|---|---|---|
| Execution model | One CUDA thread per trajectory, own adaptive steps, whole solve in one kernel launched through an XLA FFI custom call (`modax/_jax_numba_custom_call.py`) | Fused NVIDIA Warp or hand-CUDA kernels for registered right-hand sides; a pure-JAX record-and-replay path for everything else; diffrax as forward fallback (`gradsolve/dispatch.py:178-265`) |
| Stiff method | Rodas5P (Steinebach 2023), df/dt term, PID controller, per-component error weights, fp32 or fp64 LU (`modax/rodas5P.py:40-113, 340-347, 903-928`) | Fused: Rosenbrock23, order 2, autonomous only, refuses cuda above dim 12 (`gradsolve/warp/warp_rosenbrock.py:67, 324-340`). JAX path: Rodas5P with a dense `jax.jacfwd` Jacobian and `jnp.linalg.solve` per stage (`gradsolve/solvers/rodas5p_step.py:119-146`) |
| Jacobian | Enzyme forward sweeps, one per colour of a user-supplied sparsity pattern, written straight into a compiled sparse LU buffer (`modax/rodas5P.py:251-303, 445-500`) | Dense, by `jacfwd` of the user RHS at every step |
| Sparsity | AMD ordering (cvxopt), symbolic fill, CSR, unrolled straight-line factorise and solve (`modax/_sparse_direct.py`) | None |
| Dimension range | Tested to 128 state variables; DISCO-EB at 50 to 96 | Fused kernels: dim ≤ 64 nonstiff, ≤ 12 stiff on a GPU; above that the JAX path, with `remat` forced at dim ≥ 16 and OOM at 64 without it (`dispatch.py:124-129`) |
| Gradients | Continuous forward sensitivities integrated jointly, `custom_jvp`, cost about 0.7 (1 + n_inputs) of a solve; no reverse mode (`modax/_sensitivity.py`) | Reverse-mode discrete adjoint of a replayed frozen mesh via `lax.scan`; forward mode through the same replay also works (lib p.19) |
| JAX composability | `jit`, `vmap` (lowered to one launch), `scan`, `jvp`, `grad` as a primitive | `solve()` returns NumPy; `grad_closure` records on the host at concrete inputs; only the returned replay closure is traceable (`gradsolve/api.py:291-400, 792-1030`) |
| Platform | Linux x86_64 + NVIDIA only; 237 MB vendored LLVM 15 + Enzyme wheel (`modax/wheels/README.md`) | CPU or NVIDIA; pip-only |

## 3. How each maps onto the linrec tasks

| Plan item | modax | gradsolve |
|---|---|---|
| Task 1, assert linearity and `assemble_A` | Nothing | Nothing |
| Task 2, Rodas5 step as `(M_n, E_n)` | Same step structure in kernel form (one LU, eight solves, `modax/rodas5P.py:770-901`), but Rodas5P (γ = 0.21194) not clax's Rodas5 (Di Marzo, γ = 0.19, `clax/rosenbrock.py:36`). Do not mix tableaux inside the spike | Pure-JAX Rodas5P stage loop (`rodas5p_step.py:119-146`) is a readable cross-check of the stage recursion if you switch tableau later |
| Task 3, zero-length padded steps | Not applicable | Directly relevant. gradsolve hit the same bug class: a padded `dt = 0` row is an algebraic identity but `0 * inf = NaN` whenever `J` or `f_t` is non-finite at the padded node; they mask `J` and `f_t` with `jnp.where(dt != 0, ...)` before forming W (`rodas5p_step.py:126-134`). Also their safe-denominator form for `theta = (ts - t)/dt` on padded rows, which has the right value but a NaN derivative if written naively (`gradsolve/solvers/dense.py:197-210`). Pin both in `test_zero_step_is_identity_and_jvp_finite` |
| Task 4, record and pad meshes | A modax solve returns the history and step counts only, never the accepted step sizes: it cannot record a mesh | Their device recorder is a vmapped `lax.while_loop` with a doubling buffer cap and four status codes (`gradsolve/solvers/record_jax.py:74-213`). clax already gets the same mesh from diffrax `SaveAt(steps=True)`; the status-code discipline (reached, exhausted, underflow, buffer full) is worth copying into `record_mesh` so an incomplete mesh can never be replayed silently (`record_jax.py:200-211`) |
| Task 5, sequential control arm | A stronger control arm exists outside XLA: the modax kernel. Measured by its authors on DISCO-EB's 50-variable system at 128 k-modes: 398 ms per solve on an RTX 4070 SUPER with the sparse direct LU, 509 ms with the hand-written Schur solver it replaced (`modax/README.md` "Sparse systems"; tolerance not stated there). Do not compare that number with a V100 or H100 timing; it only says the fused sequential design reaches sub-second on a consumer card at this state size | `rodas5p_replay` is a sequential fixed-mesh scan, but with `jacfwd` per step, lockstep padding to the longest trajectory and the Rodas5P tableau. The plan's own arm, using `A` directly, is strictly better. Do not substitute |
| Task 6, blocked `associative_scan` | The composition stays in XLA. If forming `M_n` by batched dense `lu_factor` turns out launch-heavy (memo §6 risk table), `modax._sparse_direct` is importable on its own: `sparse_direct_solver(sparsity, n_vars, ordering="amd")` returns `factorize_local(lu, ipiv)` and `solve_local(lu, ipiv, rhs)` as numba-cuda-mlir device functions plus a `CompressedJacobian` with `.slot(row, col)`, `.diagonal` and `.size` (`modax/_sparse_direct.py:393-452`, `modax/_sparsity.py:42-141`). One thread per (n, k) could form `M_n` column by column with d solves per stage against one factorisation. This needs the numba-cuda-mlir toolchain and is Phase 3 material, not spike material | Nothing |
| Task 7, a-posteriori error monitor | Nothing | gradsolve has no monitor; its replay trusts the recorded mesh |
| Task 8, forward sensitivities through the scan | modax's sensitivities cannot see θ-dependence through closed-over background tables (Enzyme treats them as constants); linrec's native `jvp` through `assemble_A` has no such hole. This is the structural reason linrec, not modax, is the gradient path for clax | Their measured drift of a frozen mesh across a 400-step Adam fit stays within 0.05° of gradient direction and 2.1e-6 in final state (lib p.21-22, Table 4), supporting the memo's "record at fiducial, reuse, monitor" policy. Their user-facing re-record pattern is an outer loop around inner gradient steps (`gradsolve/examples/07_saveat_timeseries_fit.py:85-101`) |
| Task 9, C_l parity | Nothing | Nothing |
| Tasks 10-11, benchmark and gates | External bar for G3 (`t_par ≤ 0.2 t_current`): see Task 5 row. If a like-for-like number is wanted on Bridges-2, see section 6 | The lib paper's §5 controls (p.24-26) are the right template for attributing a speedup: uneven step counts contribute only 1.33x; the fixed-length scan against a checkpointed `while_loop` on the GPU is the whole 8 to 11x; on CPU the scan shows no advantage. Expect the same: CPU runs of the spike (Tasks 0-10) will not show the effect; only Task 11 can |

## 4. Patterns worth copying (code, not dependencies)

1. Padded-step NaN masking and the double-`where` theta: `gradsolve/solvers/rodas5p_step.py:126-134`, `gradsolve/solvers/dense.py:197-210`. Use in Task 3 and in any dense-output lane.
2. Recorder status codes and the rule "never replay an incomplete mesh": `gradsolve/solvers/record_jax.py:60-72, 200-211`.
3. Bracket-and-re-step dense output and its host-side validator that every requested time maps to `theta` in [0, 1] of exactly one bracketing step: `gradsolve/solvers/dense.py:106-163, 230-316`. Relevant to Review focus 2 (merged meshes must contain `tau_grid` bit-exactly).
4. Rodas5P with a published order-4 continuous extension, for after the spike. clax's Rodas5 uses `LocalLinearInterpolation`, flagged as a Planck-precision risk in memo §2.4 and §6. Both libraries carry Steinebach's dense-output weights and agree with each other: `gradsolve/solvers/rodas5p_step.py:87-116, 171-218` (stored H form, Horner evaluation, `b_i(1) == m_i` bitwise) and `modax/rodas5P.py:933-982` (same numbers inside the kernel). Switching tableau changes Task 2's reference (`Rodas5.step`), so it is a follow-up, not part of the spike.
5. A jaxpr-to-scalar-source translator, if Phase 3 ever generates a fused kernel from `_perturbation_rhs`: `gradsolve/warp/jax_field.py:98-354`. It walks `jax.make_jaxpr`, folds constants, and emits one scalar assignment per output component; it covers arithmetic, comparisons, `exp/log/sin/cos/tan/tanh/sqrt`, `integer_pow`, static reshapes, `concatenate`, `stack`, `iota` and inlines `pjit`. Missing for clax: `select_n` (emit a branchless `a + (b - a) * cond`), gathers into background tables (emit indexed reads of a device array), and closed-over array constants (it refuses them, `jax_field.py:417-421`).
6. The modax constant-memory pitfall if anything is written in numba-cuda-mlir: an `int32` read from a `cuda.const` array promotes as unsigned in index arithmetic, so a negative table entry addresses garbage silently (`modax/_sparse_direct.py:625-639`, `modax/AGENTS.md` "A negative value in a constant index table is fatal").

## 5. The TAM formulation and what it does to linrec

Reference: Kamionkowski & Caldwell, arXiv:2609.36000 (read in full). It is the derivation behind the integral-equation codes CLASSIER (Ji et al. 2022; Lee et al. 2025, 2026a) and CLASSIER-DDM.

What changes numerically:

- The photon hierarchy is replaced by Volterra integral equations for the monopole, dipole and quadrupole. Eq. 24 (p.6) gives the temperature moments as integrals of the metric and baryon sources against `j_l(w)`, `R_l^{L,L}(w)` and `j_l'(w)` with `w = k(τ - τ')`; Eq. 30 (p.8) closes the quadrupole: `Π_k(τ) = -Δ^T_{2,k}(τ) + 9 ∫ dτ' e^{-κ(τ,τ')} κ̇(τ') Π_k(τ') j_2(w)/w^2`. Massless neutrinos are the `κ̇ → 0` limit; massive ones follow Ji et al. 2022.
- The transfer functions are line-of-sight quadratures of those sources: Eq. 25 (temperature), Eq. 29 (E), Eqs. 40-42 and 48 (tensors), with the radial functions of Appendix A (p.12-13). Appendix C recovers the usual hierarchy by differentiating the integral equations, which is the consistency check to keep.
- Birefringence enters only through a `cos 2β` factor in the source `Π_k` (Eq. 36, p.9) and a rotation of the E-mode LOS integral (Eq. 35); the solver structure is unchanged.
- Footnote 7 (p.8): the monopole and dipole may be taken from the coupling ODEs instead of the integral equations where that is more stable. The split between ODE variables and integral unknowns is a design choice to measure.

Consequences for linrec:

| linrec ingredient | Under TAM |
|---|---|
| Linearity in the perturbations | Preserved. Everything is linear and homogeneous; the whole per-k evolution on a τ grid is one causal linear system `X = S + K X` with `K` strictly lower block-triangular |
| Mesh as data, no state-dependent control flow | Preserved and strengthened: there is no adaptive stepping at all; the τ grid is a quadrature grid |
| Parallel assembly over (k, n) | Preserved: all kernel entries `K(τ_n, τ_m; k)` form in parallel over (k, n, m) |
| Rodas5 step matrices `M_n` | Gone. The free-streaming hierarchy that made the per-step matrix d×d with d = 47 to 250 no longer exists; the ODE remainder is the metric plus matter, of order 5 to 10 variables |
| `associative_scan` prefix products | Gone. The memory kernel makes the recurrence non-Markov (`X_n` depends on all `X_m`, `m < n`), so the parallel-in-time primitive is a blocked lower-triangular solve: dense solve inside a τ block, matmul updates to later blocks, level-3 BLAS throughout |
| A-posteriori error monitor | Replaced by a residual of the discretised integral equation and a grid-refinement check |
| Forward sensitivities | Preserved: `jvp` through assembly and the blocked solve |

Two facts to exploit when designing the TAM kernel: `e^{-κ(τ,τ')} = e^{-κ(τ)} e^{+κ(τ')}` is separable, and the remaining dependence is on `w = k(τ - τ')` only, so on a uniform τ grid `K` is a diagonally scaled convolution and applying it costs `O(N log N)` per k. A dense `K` (N ≈ 2000 to 5000 nodes, 3 to 6 integral unknowns per species, 600 k) does not fit in GPU memory without that structure or blocking.

Neither modax nor gradsolve can host the TAM evolution: both solve `y' = f(t, y, p)` and give the right-hand side no access to the history. modax's `save_hook` can accumulate the LOS quadrature of Eq. 25 inside a launch for a forward-only run, but a hooked solve supports neither `vmap` nor differentiation (`modax/rodas5P.py:1204-1211`). The only route back to an ODE library would be a sum-of-exponentials fit of `j_2(w)/w^2` turning the memory into auxiliary variables, which is research, not a library feature.

## 6. If a like-for-like modax number is wanted (optional, after G1-G5)

Install: `pip install modax-solvers` on a Linux x86_64 node with an NVIDIA GPU; it pulls `numba-cuda-mlir` (cu12 or cu13 to match the JAX build) and the 237 MB `numba-enzyme-cuda` wheel. Do not install upstream `numba-enzyme` alongside it.

Work needed before the first solve: a callback in modax's form, a Python function `(y, t, p) -> tuple` using `math` only and constant tuple indices (`modax/docs/guide/callbacks.md`); background and thermodynamics tables as `cuda.to_device` arrays closed over by the callback (device arrays are fine, `modax/tests/test_save_hook.py:43`; host arrays would land in the 64 KiB constant window); the sparsity pattern of `A` passed as `sparsity=`; `lu_precision="fp64"`; per-k start handled by `tf_index` for the end and by shifting `t_span` or rescaling time for the start. The modax authors generated DISCO-EB's callback from its looped right-hand side (`modax/AGENTS.md` "Writing ODE callbacks"), so a generator from `assemble_A`'s structure is the realistic path, and item 5 of section 4 is the pattern for it.

Report it as a sequential-fused reference point at matched achieved accuracy against `exact_cl`, on the same card as the linrec runs, with its own compile time stated separately (49 s cold at 96 variables, about 2 min for denser sparse patterns, per `modax/AGENTS.md`).

## 7. Do not

- Add gradsolve or modax as dependencies of `clax/linrec.py`.
- Use `gradsolve.solvers.rodas5p_replay` as the sequential control arm.
- Quote modax milliseconds (RTX 4070 SUPER) next to clax milliseconds (V100, H100) without rerunning both on one card.
- Switch the spike's tableau from Rodas5 to Rodas5P before Task 9 passes; it changes the Task 2 reference.
