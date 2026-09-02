# ADR 0002: Chebyshev k-sampling for C_l sources (phase 1, opt-in)

Date: 2026-09-02
Status: Accepted (opt-in; no preset default changed)
Issue: https://github.com/smsharma/clax/issues/31

## Context

The C_l k-integration has an open accuracy debt at ℓ>1200: `CHANGELOG.md`
(~line 1312) lists "l>1200: n_k_fine=10000 under-resolved (need 20000+ or
hybrid k-grid)" among the remaining blockers, and `BENCHMARK.md:340` still
tracks a "Hybrid k-grid for TT ℓ>1200" item as an untracked planned PR. The
README's "Known limitations" section (CHANGELOG entry, ~line 607) carries the
same TT ℓ>1200 / TE ℓ=1000 item, attributed to "k-grid under-resolution at
high ℓ".

Two prior-art constraints already fixed the shape of any fix before this
branch existed:

1. **Never interpolate Δ_ℓ(k) in k.** `CHANGELOG.md:2192-2199` (the
   "Confirmed correct / failed approaches" register) records: "CubicSpline
   interpolation of T_l(k): T_l oscillates faster than k-grid. CubicSpline
   introduces aliasing. Must interpolate SOURCE functions (smooth) instead,
   then compute T_l on the fine grid," and "Intermediate k-density (30-120
   k/decade): Non-monotonic convergence for raw trapezoidal C_l integration.
   Either use very dense (200+) or source interpolation." clax's
   `_cl_k_integral`/`_cl_k_integral_cross` path (with `k_interp_factor>1`)
   already violates this by splining `T_l(k)` directly and is flagged in its
   own docstring as risky; the source-interpolated path
   (`compute_cl_*_interp`, `compute_cls_all_fast`) is the sanctioned pattern
   and the only one this ADR touches.
2. **arXiv:2608.24682 (Sletmoen 2026, method note committed at 605d8c8,
   `docs/superpowers/plans/notes-2608.24682-method.md`) confirms and
   generalizes constraint 1**: the paper solves perturbation ODEs explicitly
   at only Chebyshev k-nodes, Chebyshev-interpolates the smooth source
   `S(τ,k)`, and always computes `Δ_ℓ(k)` by explicit LOS quadrature with the
   exact Bessel function on the fine grid — never interpolating `Δ_ℓ(k)`
   itself. k-alone node counts converge to the paper's noise floor at N=80
   (Figs. 2-4); the headline "50-80 nodes, 2.5x-4x speedup" is the *combined*
   k+ℓ result (§7), not a k-alone claim. The paper uses raw (linear) k, not
   log k — clax's k-domain spans several decades, so log k Lobatto nodes are
   a clax-specific, independently justified deviation, not taken from the
   paper.

## Decision

Add an **opt-in** phase-1 path: perturbation sources solved at
Chebyshev-Lobatto nodes in log k, evaluated onto the fine k-grid by
barycentric Chebyshev interpolation, as an alternative to the existing
log-uniform-grid + cubic-spline path. Nothing in the default pipeline
changes.

- `clax/interpolation.py`: `chebyshev_lobatto_nodes(a, b, n)` (static numpy
  grid constructor) and `ChebyshevInterpolant` (barycentric evaluation,
  registered as a JAX pytree, clip-saturating boundary policy matching
  `CubicSpline`). Contract: `x` must be Lobatto-spaced — barycentric weights
  are derived from index parity, not from `x` itself.
- `clax/params.py`: new static `PrecisionParams.pt_k_grid_type: str = "log"`
  (`"log" | "chebyshev"`). Default `"log"` is the historical `jnp.logspace`
  path, bit-identical.
- `clax/perturbations.py`: `_k_grid()` honors the knob. `"chebyshev"` builds
  Lobatto nodes in `log10(k)` with the same count and endpoints as the log
  path, then exponentiates. Every downstream consumer treats the k-grid as
  an opaque ascending array, so the knob is safe by construction.
- `clax/harmonic.py`: `_interp_sources_to_fine_k(..., method="spline")` gains
  a `"chebyshev"` method — one dense barycentric evaluation matrix `B`
  (`_barycentric_matrix`), applied to every source as a single matmul in
  place of one Thomas-solve `CubicSpline` per (source, τ). Threaded as a
  `k_interp_method="spline"` kwarg through `compute_cl_tt_interp`,
  `compute_cl_ee_interp`, `compute_cl_te_interp`, `compute_cls_all_interp`,
  and `compute_cls_all_fast`. Default `"spline"` everywhere — bit-identical
  until a caller opts in. `compute_cl_bb` is untouched (native inline
  spline, out of scope for phase 1). The four duplicate fine-grid
  constructions were also consolidated into one `_fine_log_k_grid` helper
  (pure refactor, no behavior change) — the fine grid **stays log-uniform +
  trapezoid** regardless of `pt_k_grid_type`; only the *coarse solve grid*
  and the *coarse→fine interpolation method* are affected by the new knobs.
  `Δ_ℓ(k)` itself is still always computed by explicit quadrature on the
  fine grid, per constraint 1/2 above — unchanged.
- **Explicit precondition, cross-checked empirically (see Consequences)**:
  `k_interp_method="chebyshev"` requires the perturbation solve to have used
  `pt_k_grid_type="chebyshev"`; the converse also holds — a Lobatto solve
  grid should be paired with barycentric interpolation, not cubic spline.

## Alternatives considered

1. **Clenshaw–Curtis quadrature directly on the coarse Chebyshev nodes**
   (skip the fine grid, integrate C_l from ~150-300 node values). Rejected:
   `T_l(k)` (≡ `Δ_ℓ(k)`) is oscillatory in k at the node spacing achievable
   at this `pt_k_per_decade`, which is exactly the failure mode constraint 1
   already identified for direct `T_l(k)` interpolation/quadrature — a
   coarse-node quadrature rule over an oscillatory integrand is not safe
   without either much higher node density or an explicit fine-grid
   quadrature step, which is what this ADR's design already does.
2. **Hybrid linear/log fine grid** (the `BENCHMARK.md:340` / CHANGELOG-listed
   ℓ>1200 remedy). Deferred, not rejected — complementary: it targets the
   *fine*-grid trapezoid integration density/spacing, while this ADR targets
   the *coarse* solve-grid placement and the coarse→fine interpolation
   method. Both could combine in a later phase.
3. **Interpolating `Δ_ℓ(k)` itself** (Chebyshev or otherwise) in place of
   source interpolation. Rejected per `CHANGELOG.md:2192-2199` and per the
   method note's finding (b): the paper never interpolates `Δ_ℓ(k)`, only
   the smooth source `S(τ,k)`; `Δ_ℓ(k)` is oscillatory in k and must always
   be computed by explicit quadrature with the exact Bessel function on the
   fine grid. clax's existing `_cl_k_integral(..., k_interp_factor>1)` path
   already does this and is called out in the method note as the
   anti-pattern the paper avoids — left untouched, phase-2 backlog.

## Consequences

**Two grid types (and two interpolation methods) to maintain and keep in
sync**, with a validated but narrow compatibility contract: chebyshev grid +
spline interp is the worst combination measured (see below); chebyshev
interp requires a chebyshev solve grid.

**Measured GPU A/B (V100, planck_cl base + `pt_k_max_cl=1.0`, ℓ_max=2000,
unlensed C_l vs `reference_data/lcdm_fiducial/cls.npz`, pct =
(clax−CLASS)/|CLASS|·100 at exact ℓ; jobs 14138+14141, per-stage seconds,
`t_pt` includes per-shape JIT compile in a fresh process per job):**

| grid | kpd | interp | n_k | t_pt(s) | t_cl(s) | TT20 | TT100 | TT500 | TT1000 | TT1500 | TT2000 | EE20 | EE100 | EE500 | EE1000 | EE1500 | EE2000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| log | 30 | spline | 150 | 5360 | 11.25 | -1.00 | -0.823 | -0.85 | -1.51 | -5.16 | -5.78 | -1.05 | -0.592 | -0.406 | 0.127 | -3.00 | -0.864 |
| log | 60 | spline | 300 | 6397 | 3.48 | -0.999 | -0.82 | -0.778 | -0.907 | -3.21 | 0.47 | -0.947 | -0.592 | -0.386 | 0.286 | -1.70 | 0.997 |
| chebyshev | 30 | spline | 150 | 5028 | 3.26 | -1.01 | -0.827 | -0.984 | -2.67 | -7.02 | -10.1 | -1.83 | -0.589 | -0.496 | -0.0832 | -5.37 | -2.47 |
| chebyshev | 30 | chebyshev | 150 | 5117 | 4.23 | -1.04 | -0.84 | -0.727 | -0.83 | -3.09 | 2.01 | -0.538 | 0.589 | -1.16 | 0.285 | -0.587 | 2.68 |
| chebyshev | 60 | spline | 300 | 5860 | 3.22 | -1.00 | -0.821 | -0.785 | -0.951 | -3.32 | 0.529 | -0.972 | -0.592 | -0.39 | 0.262 | -1.72 | 0.353 |
| chebyshev | 60 | chebyshev | 300 | 5967 | 12.84 | -0.999 | -0.82 | -0.773 | -0.878 | -3.13 | 0.938 | -0.986 | -0.539 | -0.426 | 0.334 | -1.56 | 0.277 |

**Density-convergence diagnostic** (job 14142; cheb/log C_l ratio−1, fast_cl
base, both arms opt-in vs default):

| kpd | n_k | TT20 | TT100 | TT500 | EE20 | EE100 | EE500 |
|---|---|---|---|---|---|---|---|
| 15 | 62 | +0.0220% | +0.0538% | +1.7003% | +2.3767% | -0.0039% | +0.2397% |
| 30 | 125 | -0.0125% | +0.0026% | +0.0831% | -0.0422% | +0.0076% | +0.0212% |
| 60 | 250 | +0.0003% | +0.0002% | +0.0049% | +0.0040% | -0.0000% | +0.0011% |

**Verdicts (measured facts, stated as-is):**

1. No-regression clause (a), strict: FAILS at exactly one probed point — TT
   ℓ=2000 at matched n_k=300 (cheb 0.938% vs log 0.47%; the only point where
   the log arm is inside the 0.5% benchmark gate and the chebyshev arm is
   not). Everywhere ℓ≤1500 the matched-density differentials are ≤0.16pp
   (chebyshev equal or slightly better); EE ℓ=2000 IMPROVES (0.277% vs
   0.997% — there the log arm fails the gate and chebyshev passes).
2. Payoff: within the probed set {150,300}, the smallest chebyshev n_k
   matching the log path's full-density (n_k=300) accuracy across all
   sampled ℓ is 300 — no node-count reduction demonstrated at planck scale.
   However chebyshev degrades far more gracefully: at n_k=150,
   cheb+barycentric gives TT2000=2.01% vs log's −5.78% and TT1500 −3.09%
   vs −5.16%.
3. ℓ>1200: NOT materially improved at matched density (TT1500 −3.13 vs
   −3.21; the residual high-ℓ errors are common to both arms, i.e. not
   coarse-k-node limited — they live elsewhere: fine-grid integration/other
   stages). Improvement appears only in the graceful-degradation sense at
   reduced n_k.
4. Chebyshev grid + cubic spline is the WORST combination (TT2000 −10.1% at
   n_k=150): Lobatto nodes are interior-sparse (~π/2× coarser mid-domain);
   only barycentric evaluation recovers the accuracy. The docstring
   precondition (chebyshev interp requires Lobatto solve grid) is
   empirically validated, and the converse holds too: a Lobatto solve grid
   should use barycentric interp.
5. The in-repo GATE test `test_cls_chebyshev_path_matches_spline_path`
   FAILED on GPU (job 14138): TT ℓ=500 cheb/log ratio 1.0170 > 1.005
   tolerance at fast_cl density (n_k=62). The convergence table shows this
   is coarse-density discretization difference (1.70% at n_k=62 → 0.083% at
   125 → 0.005% at 250), not an implementation bug. Per STOP-and-report the
   test is LEFT AS COMMITTED (it fails when run without `--fast`); the
   recommended fix — pending maintainer ruling, NOT applied — is to run the
   gate at `pt_k_per_decade=30` (measured margin 6x under the 0.5% gate,
   ~10 min/arm on V100).
6. Caveat: all six cells share a −0.8…−1.0% absolute offset at ℓ≤1000 (both
   arms, including the pure-default log cells) — a configuration-level
   offset of this probe setup (planck_cl + `pt_k_max_cl=1.0` + ℓ_max=2000 +
   `compute_cls_all_fast` defaults) vs the reference; it cancels in every
   cheb-vs-log differential and predates this branch. BENCHMARK.md's
   planck_cl <0.2% claim comes from a different accuracy configuration and
   was not reproduced here; not investigated further in this PR.
7. Wall-clock: `t_pt` is compile-dominated in fresh processes; chebyshev
   cells are not slower (5967 vs 6397 at n_k=300). Barycentric `t_cl` ≈
   spline `t_cl` (first call in a process includes jit compile: 12.84s vs
   steady ~3.2-4.2s).
8. GPU fastsuite (`pytest tests/ --fast -q`) in job 14138 hit its 3600s
   timeout with zero failures observed up to the kill; per-file CPU runs
   during Tasks 1-4 were all green (interpolation 11/11, chebyshev file
   8/8, perturbations `--fast`, harmonic `--fast` 11/11, harmonic+high_l
   17/17).

**Follow-on consequences:**

- **Preset flips require the A/B table above — and per the measured A/B, no
  preset flip is justified yet.** No `PrecisionParams` preset
  (`fast_cl`/`planck_cl`/etc.) sets `pt_k_grid_type` or passes
  `k_interp_method`; both stay opt-in kwargs.
- The failing gate-test density question is open: whether to recalibrate the
  gate to `pt_k_per_decade=30` (or another density), keep it `xfail`-ed at
  `fast_cl` density, or restructure it as a convergence test, is left to
  maintainer ruling.
- The ℓ>1200 debt in `CHANGELOG.md`/`BENCHMARK.md` is not closed by this ADR
  — verdict 3 shows the residual high-ℓ error is common to both k-grid arms,
  so it is not primarily a coarse-k-node placement problem. The hybrid
  linear/log fine grid item (alternative 2) remains the more promising lead
  for that specific debt.
- `compute_cl_bb` keeps its own native inline spline and is unaffected by
  either knob.
