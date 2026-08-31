# Method note: arXiv:2608.24682 — Chebyshev interpolation in Einstein–Boltzmann codes

Sletmoen, H. (2026), "Chebyshev interpolation in Einstein–Boltzmann codes",
arXiv:2608.24682 (submitted to A&A). Implemented/benchmarked in SymBoltz.jl.

**Fetch status: paper read in full (verified, not fallback).** `WebFetch` on
the abs page confirmed title/author/abstract. `WebFetch` on the PDF URL could
not parse the encoded PDF streams into text, but it saved the raw PDF bytes
locally; the `Read` tool then rendered and read all 8 pages (body + refs +
Appendix A) directly as images. All claims below are cited to specific
sections/figures/equations of the paper, not to the GitHub issue summary.

## Findings

**(a) k-direction transform: raw (linear) k, not log k.** §5: "Chebyshev
k-interpolation consists of evolving the perturbation ODEs for the Chebyshev
k-nodes (6) of a wavenumber interval `[k_min, k_max]`," using the paper's
generic affine map x(y)=(2y−a−b)/(b−a) (§3) applied directly to k — no log
transform. Figs. 1–2 plot sources against linear `k/(H0/c)`, confirming this.
The paper never tests log k. **Deviation for clax**: clax's k-domain spans
several decades, so log k Chebyshev–Lobatto nodes are a clax-specific choice,
independently justified, not taken from the paper — flagged in the table so
phase 2 doesn't misattribute "log k" to the paper.

**(b) Δ_ℓ(k) is never itself interpolated in k; only the smooth source S(τ,k)
is.** §5: solve perturbation ODEs explicitly at only Chebyshev k-nodes
(10²–10³ k), Chebyshev-interpolate the resulting *source* `S(τ,k)` (smooth in
k) up to 10³–10⁴ k-values, and only then run the LOS integral
`Δ_ℓ(k)=∫dτ S(τ,k) j_ℓ(kχ)` (eq. 2) *explicitly* at each fine k with the exact
Bessel function. §7 reconfirms this order of operations. The Bessel-oscillation
problem is sidestepped by construction: Δ_ℓ(k) is oscillatory in k, so it is
always computed by quadrature on the fine grid, never interpolated. Maps onto
clax's `harmonic.py:compute_cl_*_interp` (interpolates
`pt.source_T0/T1/T2/E`, then calls `_exact_transfer_tt/_exact_transfer_ee`) —
already the paper's pattern. clax's other path,
`_cl_k_integral(..., k_interp_factor>1)`, instead CubicSpline-interpolates
`T_l(k)` (≡Δ_ℓ(k)) directly — the pattern the paper avoids; its docstring
already flags this as risky.

**(c) k-alone node counts / error (§5, Figs. 2–4):** N≥40 beats cubic
splines; N=80 fully converges to the numerical noise floor, indistinguishable
from solving every k-mode explicitly (splines still ~4 orders of magnitude
worse). N=100 (Fig. 3): ~4 orders of magnitude more accurate than splines
near recombination (a≈10⁻³) for T/E; more accurate and more uniform over k
for matter/lensing. N≥48 (Fig. 4, RMS over full (τ,k)): lowest error of every
method, every source. At N=400 splines still haven't converged. **Caveat**:
the abstract's "50–80 points, 10⁻⁴–10⁻⁵ error, 2.5×–4× speedup" is the
*combined* k+ℓ result (§7, Fig. 7), not k-alone.

**(d) Integer-rounding trick for ℓ (§6):** standard Chebyshev ℓ-nodes are
generically non-integer, but C_ℓ is only needed at integer ℓ. Two fixes: (i)
generalize the LOS integral and j_ℓ(x) to non-integer ℓ (well-posed, but
needs non-integer-order Bessel evaluation), or (ii) **round the Chebyshev
ℓ-nodes to the nearest integer** and rebuild barycentric weights numerically
(eq. 9) instead of the closed form (eq. 10). This stays robust because node
density near the rounded points still resembles the Chebyshev density, and
discreteness is negligible against 1000s of multipoles. SymBoltz implements
this as `ChebyshevIntegerInterpolator`. Fig. 6/7: integer-rounded and
standard Chebyshev ℓ-interpolation perform equally well; the author
recommends the integer-rounded form since it generalizes to curved
geometries (hyperspherical Bessel functions).

## Paper method → clax seam → phase

| Paper method (§) | clax seam | Phase |
|---|---|---|
| Solve perturbation ODEs only at Chebyshev nodes over a k-interval; barycentric-interpolate smooth S(τ,k) to any k (§5, eqs. 6/8–10) | `perturbations.py:_k_grid` (currently `jnp.logspace`-uniform, line ~1862) — replace with Chebyshev–Lobatto nodes in **log k** | **Phase 1** |
| Fine-k source interpolation feeding the LOS integral (§5, §7) | `harmonic.py:_interp_single_source` / `_interp_sources_to_fine_k` (currently `CubicSpline` in log k, lines ~412–441) — replace with barycentric Chebyshev evaluation | **Phase 1** |
| Δ_ℓ(k) always computed by explicit LOS quadrature on the fine k-grid with exact j_ℓ; never interpolated in k (§5 core trick) | `harmonic.py:_exact_transfer_tt/_exact_transfer_ee` — no change needed, already matches when reached via the `*_interp` path | **Phase 1** (consumer, unchanged) |
| — (anti-pattern the paper avoids) | `harmonic.py:_cl_k_integral`/`_cl_k_integral_cross` with `k_interp_factor>1`: splines `T_l(k)` (≡Δ_ℓ(k)) directly | **Phase 2 backlog** — this *is* Δ_ℓ(k)-in-k interpolation, explicitly out of phase-1 scope; paper shows the source-interp path is the correct pattern, so the fix is likely "prefer `*_interp` path" not "Chebyshev-ify this function" — needs its own design pass |
| ℓ-interpolation with integer-rounded Chebyshev nodes to avoid non-integer-order Bessel functions (§6, eqs. 9–11) | `harmonic.py:compute_cl_*_interp_l` + `bessel.py:sparse_l_grid`/`build_jl_table` (currently sparse-ℓ + `CubicSpline` in ℓ) | **Phase 2 backlog** — ℓ-direction, explicitly excluded from phase 1 by task scope |
| Author deliberately does *not* use Chebyshev in τ; recommends ODE dense-output/Hermite instead (§4) | clax's existing `CubicSpline(tau_grid, ...)` for τ-direction (e.g. `transfer.py:_interp_delta_m`) | **N/A** — confirms current clax choice (spline/Hermite-style in τ) already matches the paper's recommendation; no backlog item |
| Chebyshev nodes placed on **raw k** over one bounded interval, not log k (§5, finding (a) above) | clax plan's use of **log k** Chebyshev–Lobatto nodes | **Phase 1** (deviation, flagged) — log k is a clax design choice for a multi-decade k-domain, independently justified (spectral interpolation of smooth functions + fewer ODE solves per task brief), *not* copied from the paper; phase 2 must not assume the paper validates log-k node spacing specifically |

## Phase-1 scope (unchanged, per task ruling)

Phase 1 = perturbation sources solved at Chebyshev–Lobatto nodes in log k,
with barycentric Chebyshev evaluation replacing the coarse→fine cubic-spline
step for those sources in the C_l pipeline. This is independently justified
regardless of the above findings. Everything touching Δ_ℓ(k)-in-k
interpolation, the ℓ-direction, or non-integer-ℓ Bessel evaluation is
phase-2 backlog per the rationale in the table above.
