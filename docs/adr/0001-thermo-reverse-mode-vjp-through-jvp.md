# ADR 0001: Reverse-mode thermodynamics gradients via a forward-mode basis ("vjp-through-jvp")

Date: 2026-08-29
Status: Accepted
Issue: https://github.com/smsharma/clax/issues/30

## Context

Reverse-mode AD (`jax.grad`) through `thermodynamics_solve` carries a ~2%
error on h-like parameters, while forward mode (`jax.jvp`) through the same
code is exact (verified six independent ways against step-converged finite
differences, on two ODE solvers and two adjoints). Root cause — established by
input-channel bisection, stage splitting, and per-table/per-region probes
(GPU jobs 13313–13946): **catastrophic floating-point cancellation in the
recombination-era backward pass**. The Peebles/RECFAST recombination rates
contain Boltzmann-exponential ratios (`exp(B/kT) ~ e^52`,
cf. `_recfast_dxHII_dlna`, CLASS `wrap_recfast.c:133-134`), so AD
intermediates reach ~1e13. Forward mode pairs huge × tiny factors *locally
per grid point* — nothing large is ever formed. Reverse mode must contract
thousands of ±1e13-scale cotangent terms through shared scalars whose true
total is ~1e-3 (or exactly 0); float64 keeps a deterministic exact-ULP
residue (measured: the xe.y full-table reverse "derivative" is 2⁻⁹ exactly —
one float64 ULP at magnitude ~1e13; the bg→thermo chain's reverse returned
8.66e7 where the true value is 1.16e5, 749×; end-to-end
`d(sum(pk_mm_real))/dh` = 4.107e6 vs truth 4.0296e6, +1.9%).

This is not a wrong derivative graph: forward and reverse compute the same
mathematical object in different association orders, and only the reverse
order is numerically pathological. The error is deterministic per compiled
program but shifts under XLA fusion changes, and is independent of solver
tolerance, adjoint implementation, and root-finder choice.

## Decision

Give the pipeline a **fused entry point**
`clax.thermodynamics.solve_background_and_thermo(params, prec) -> (bg, th)`
whose **only differentiable input is `CosmoParams`**, wrapped in
`jax.custom_vjp` (issue #30 fix option 2). Its backward pass computes the
params cotangent for **both** outputs as

```
(J^T ct)_i = <ct, J e_i>,      J = d(bg, th)/d(params),
```

evaluating the Jacobian columns `J e_i` with **one batched forward-mode
basis** (`jax.jacfwd` over the ~20 traced `CosmoParams` leaves) and
contracting each column with the incoming cotangent. This is *identical
arithmetic content* to the native VJP — pure re-association of the chain-rule
contraction — so it is a numerical-stability fix, not an approximation and
not a fudge factor. Cost: one extra primal plus ~20 fused tangent passes in
the backward (a few times one bg+thermo solve; negligible next to any
perturbation solve; memory trivial).

Fusing bg and th into one rule (rather than wrapping `thermodynamics_solve`
alone) means the bg-mediated reverse channel — whose health was never
independently established — is covered by the same proven-exact forward
basis, and the rule has a single differentiable input, which keeps the
custom_vjp signature trivial.

Wiring:

- New static field `PrecisionParams.th_grad_mode: str = "stable"`
  (values `"stable" | "native"`), mirroring the `ode_adjoint` precedent.
  `"stable"` routes through the custom_vjp; `"native"` is bitwise the old
  two-call path in both primal and AD.
- Inside the backward basis, `background_solve` runs with
  `ode_adjoint="direct"`: diffrax's `RecursiveCheckpointAdjoint` is an
  `eqx.filter_custom_vjp` (diffrax/_adjoint.py:538) and blocks `jax.jvp`;
  `DirectAdjoint` supports both modes. This replacement is internal to the
  backward rule and does not change caller-visible prec semantics (the
  primal the caller sees is produced under the caller's own prec; the
  adjoint selects derivative propagation, not the solution).
  `thermodynamics_solve` itself is a semi-implicit `lax.scan` with no
  diffrax solve and is jvp-transparent under any prec.
- Routed call sites: `clax.compute`, `clax.compute_pk_table` (and
  `compute_pk_interpolator` through it), `clax.compute_pk`. Public APIs
  `background_solve` / `thermodynamics_solve` are unchanged. Intentionally
  left native: `clax.shooting` (wraps its solves in its own custom_jvp
  implicit-differentiation rule; not an issue #30 acceptance path) and any
  direct user calls of the two separate solvers.

## Alternatives considered

1. **Log-space reformulation of the recombination rates** (issue #30 option
   3): eliminate the ~1e13 intermediates so native reverse mode is well
   conditioned. Rejected for now: invasive — every RECFAST/Peebles rate
   expression must be rewritten and re-validated term-by-term against CLASS
   (`wrap_recfast.c`), touching physics code that currently matches the
   oracle to <0.1%, for the same end result the custom rule achieves without
   touching any physics expression.
2. **Forward-only policy** (issue #30 option 1): document the ~2% `jax.grad`
   ceiling and require `jvp`/`jacfwd` + `ode_adjoint="direct"` for
   gradient-critical work. Rejected as the *fix*: HMC and the existing test
   infrastructure are built on reverse mode; a documentation-only mitigation
   leaves a silent correctness ceiling in the default path. (It survives as
   the `"native"` escape hatch.)
3. **`jax.custom_transpose`** — a `custom_jvp` whose linear tangent map has a
   custom transpose — would preserve forward *and* reverse mode through one
   wrapper with no flag. Held as a stretch alternative; not attempted since
   the flag approach met all acceptance criteria.
4. **Wrapping `thermodynamics_solve` alone** (issue #30 fix option 1 /
   "hybrid"): backward = forward basis for the params argument + native VJP
   restricted to the bg argument. Not chosen here: it leaves the bg-channel
   reverse path (untested health) in the gradient, and needs a two-input
   custom rule. The fused shape was the ultimately preferred design.

## Consequences

- `jax.grad` through the routed pipeline entry points is now consistent with
  the proven-exact forward mode: measured on CPU, grad(stable) vs jvp agrees
  to 1.2e-16–8.1e-15 relative across {sum(xe²), random-linear, sum(g²)} ×
  {h, omega_b}, where the native reverse showed 5.1e-11 up to 5.2e+1
  relative error on the same precision block. GPU pipeline-level numbers are
  recorded in the PR for branch `fix/thermo-stable-reverse-composite`.
- **Caveat — custom_vjp blocks jvp**: `jax.jvp` through the fused solve with
  `th_grad_mode="stable"` raises `TypeError`. Forward-mode users (and
  `tests/test_pk_forward_mode.py`) must set `th_grad_mode="native"`, exactly
  as they already set `ode_adjoint="direct"`. `jax.hessian`-style
  forward-over-reverse composition through the stable path is likewise
  unavailable.
- The backward pass costs ~20 forward tangent passes through bg+thermo
  (batched; seconds) instead of one native reverse sweep — irrelevant next
  to the perturbation solve that dominates every pipeline gradient.
- A future k-dependent or table-shaped output added to `BackgroundResult` /
  `ThermoResult` is covered automatically (the rule contracts whole pytrees);
  a future *integer* leaf would need a float0-cotangent guard in
  `_solve_bg_th_stable_bwd` (all current leaves are float).
- `clax` now contains exactly one `jax.custom_vjp`; the "zero custom_vjp
  rules" invariant documented in `tests/test_pk_forward_mode.py` is retired
  (note updated in place).
