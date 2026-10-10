# clax-pt: gradient of the redshift-space one-loop basis (design)

**Date:** Oct 10, 2026 · **Status:** approved by the user (Oct 10, 2026: "fix clax first")
**Origin:** Task 7 of the clax-pt ↔ ps_1loop_jax cross-validation (cross-code derivatives). The key numbers are
copied below so this spec stands alone.
**Base:** clax fork `main` at `6ba5176`.

## 1. Defect

`compute_ept` (`clax/ept.py`) builds the redshift-space one-loop terms from an FFTLog power-law decomposition of the
no-wiggle and wiggle parts of the linear spectrum, `cmsym_nw` and `cmsym_w`. On the differentiable path
(`_ir_precomputed` given, as `compute_ept_from_clax` does) the two inputs of that decomposition are a
`jax.lax.stop_gradient` snapshot of the IR split (`ept.py:2116-2124`, "PHASE-2 FREEZE"). The snapshot was made
explicit in commit `322a6ab` (Sep 1, 2026) as the leftover of an earlier NumPy-only splitter; it has no physical
motivation, and the code comment next to it says the decomposition itself is `jnp.fft` and differentiates fine.

Consequence: the derivative of every redshift-space multipole P_0, P_2, P_4 with respect to the linear spectrum
omits the response of the one-loop terms. Only the tree level and the resummed input spectrum respond.

**Evidence** (ps validation, jobs 24165 and 24167, lcdm_fiducial z = 0.38, bias point of the PFS error-bar gate):
clax's forward-mode derivative of P_ℓ along two directions in P_lin — a uniform amplitude change and a stretch in k
(t = P dlnP/dlnk) — differs from clax's own central finite differences by about 90% (amplitude) and 100% (stretch) of
the finite-difference derivative's scale, for all three multipoles. ps_1loop's derivative matches clax's finite
differences to 0.03–2%. Derivatives along f, α⊥ and α∥ are unaffected (clax AD = clax FD to 1e-8).

**Why no existing test saw it.** `tests/test_ept_gradients.py` holds the IR split fixed in both the AD and the FD arm
(`ept_setup` passes one precomputed tuple), so the frozen basis moves in neither. Its end-to-end tests differentiate
real-space functionals (`pk_mm_real`), which never read the RSD basis. The e2e AP campaign's gradient tests assert
finiteness (`test_e2e_gradient_finite`) or differentiate the AP ratios with P_lin fixed. No earlier test asserts,
and no analysis uses, a value of this channel; `test_e2e_gradient_finite` (tests/test_ept_e2e_multicosmo.py on
branches feat/classpt-e2e-ap-validation and feat/e2e-ap-on-tau0-fix, not on `main`) logs three
d(sum pk_gg_l0)/d(omega_cdm) values computed through the frozen basis (only finiteness was asserted), and they are
biased. No gradient-driven multipole likelihood exists on `main`.

**Who is affected.** Any gradient of P_ℓ(RSD) taken through `compute_ept_from_clax` with respect to parameters that
move P_lin (A_s, n_s, ω_b, ω_cdm, h, m_ν, …). HMC stays exact (the accept step uses exact energies) but samples less
efficiently; Fisher matrices, optimisers and gradient diagnostics are biased.

## 2. Fix

In `compute_ept`'s `_ir_precomputed` branch, build the RSD basis from the traced `pk_nw` and `pk_w = pk_lin_h − pk_nw`
that the function already computes a few lines earlier, instead of the `stop_gradient` snapshot. The NumPy path
(`_ir_precomputed` None) is untouched. Values do not change: `_ir_resummation_jax` returns `pk_w = pk_lin_h − pk_nw`
(`ept.py:967`), the same arithmetic as the traced `pk_w`, so the decomposition sees bit-identical inputs.

Documentation that describes the freeze is updated to describe the new state: `ept.py` lines ~820, 831, 862,
2001–2009, 2106–2124, 2686, 2725, and the `tests/test_ept_gradients.py` docstrings at 567, 687–701, 772–783 (which
correctly rule the freeze out for their real-space functionals; only the "still deferred" wording changes).

Not changed: the pinned DST-grid endpoints inside `_ir_resummation_jax` (a boundary term in the h-derivative,
documented separately); the IR-resummation scheme; the model differences recorded in the ps validation.

## 3. Consequences

- All primal outputs are bit-identical on both paths. `tests/test_ept_accuracy.py` (NumPy path vs CLASS-PT, 9 tests)
  and the ps validation's Tasks 4–6 numbers are unaffected.
- The traced graph gains one more differentiable FFT pair; compile time and memory change negligibly.
- `grad`/`jvp` of P_ℓ(RSD) through the traced splitter now carry the one-loop response. The documented ~1.4% end-to-end
  h-gradient residual (real-space functional) is unrelated and stays as documented.

## 4. Tests (acceptance)

New file `tests/test_ept_rsd_basis_gradients.py`, P_lin from `reference_data/<cosmology>/pk.npz` on the EPT grid,
3–5 cosmologies through the `lcdm_cosmology` fixture (fiducial only under `--fast`), galaxy bias point
(b1, b2, bG2, bΓ3) = (2.0, −0.5, 0.1, −0.1), window k ∈ [0.01, 0.3] h/Mpc, IR split traced
(`_ir_precomputed = _ir_resummation_jax(pk_lin, k, rs_h, h)`):

1. **AD vs FD (the fix):** `jax.jvp` of (P_0, P_2, P_4) along three tangents in P_lin — amplitude (t = P), stretch
   (t = P dlnP/dlnk) and wiggle (t = P_w) — matches central finite differences (step 1e-4 along t) to
   max|AD − FD| / max|FD| < 1e-2 per multipole (CLAUDE.md gradient target). Before the fix: ≈ 1 (RED).
   Expected after: ≲ 1e-4.
2. **Forward = reverse:** for a fixed cotangent c, c·(J t) from `jvp` equals (Jᵀc)·t from `vjp`, to 1e-10 relative.
3. **Primal unchanged:** the traced-path multipoles vs the NumPy-path multipoles (`compute_ept` without
   `_ir_precomputed`, same `rs_h`) differ by ≤ 1e-8 relative; the printed value must be the same before and after
   the fix (the fix removes a `stop_gradient` only).
4. **Finiteness:** every AD output in 1–2 is finite.
5. **Regression:** `tests/test_ept_gradients.py`, `tests/test_ir_resummation_jax.py`, `tests/test_ept.py`,
   `tests/test_ept_assembly.py`, `tests/test_ept_accuracy.py` pass; the fast suite passes.

## 5. Out of scope

The ps validation's Task 7 itself (resumes against the fixed clax after merge); the DST-endpoint freeze; any change
to the IR-resummation scheme or to the RSD loop assembly.
