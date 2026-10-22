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

## Note (Oct 20, 2026, smsharma/clax#55)

The opacity tables (kappa_dot, exp(-kappa), g, dkappa_dot/dlna, g') carried a
`stop_gradient` with an n_H_0 rescale from before this ADR (d127731), so the
forward basis this rule contracts had no x_e response in them, along any
parameter. ADR 0004 removes that cut; the rule is unchanged and its stable
reverse agrees with native reverse and with forward mode to 6e-9 on the
full tangents (probe of Oct 20, 2026, five cosmologies, two grids).

## Note (Oct 11, 2026, smsharma/clax#48)

The premise "forward mode is exact" holds only when ODE solver stages miss the
recombination chatter that clipped explicit helium/hydrogen steps produced; with
them, both modes carried step-placement-dependent spikes (#48). ADR 0003 removes
the chatter; this rule's re-association remains valid and is unchanged.
(Re-association here means this ADR's rule: compute the thermodynamics
reverse-mode product through a batch of forward-mode jvp columns instead of one
native reverse sweep.)

After ADR 0003, on CPU: reverse (this rule's stable path) and forward mode through
background + thermodynamics agree to 2.4e-16-4.6e-15 at the five LCDM grid points
of `tests/test_multicosmo_consistency.py` (job 24321), and through the full matter
solve (f = sum_k delta_m(k, tau_0)^2, `fast_cl`) to <= 3.1e-5 at three
cosmologies on the production gradient entry, i.e. the path of this rule (jobs
24227/24229/24231; the first version of the #48 fix, `fast_cl`'s Kvaerno5 at
rtol 1e-3 on the 800-point background). On the final version of that fix,
measured with Rodas5 at rtol = atol = 1e-4 on a V100, the same disagreement is
<= 1.2e-10 (ADR 0003, Consequences). On unmodified upstream the same
production-path disagreement was 1.16 at fiducial (job 24226) and 1.13 in the
diagnosis run.

On the smooth tables, native reverse mode and the production `custom_vjp` path
coincide at fiducial: both give -2.75564438e9 against forward -2.75572953e9 (job
24227). At all three cosmologies their relative disagreements with forward mode
agree to about 1e-3 relative (omega_b_high 8.081e-7 vs 8.073e-7; jobs
24227/24229/24231). So the re-association is still valid, and for this functional
at k <= 0.15 Mpc^-1 the two reverse paths agree.

The pipeline-level acceptance test of this ADR, `TestPipelineDeltaM`
(`tests/test_thermo_reverse_composite.py`), reaches much higher k: g(h) =
sum_k delta_m(k, tau_0)^2 at `fast_cl` with `pt_k_max_cl = 5`, production reverse
mode against a forward-mode jvp, bar 1e-4. Run in full mode on the fixed tree, it
passes on both platforms: on 12 CPU cores jvp = -7.96742981e10, grad(stable) =
-7.96743037e10, relative gap 6.955e-08 (job 24528, 2 h); on a V100 jvp =
-7.96743395e10, grad(stable) = -7.96740772e10, relative gap 3.293e-06 (job 24538,
1 h 41 min). The two platforms' jvp values differ by 5.2e-7, of the order of the
adaptive-step noise the test docstring quotes (~7e-7).
