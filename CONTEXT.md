# clax

A differentiable JAX reimplementation of CLASS, extended by clax-pt to one-loop
EFT galaxy power spectra. This file fixes the words used when validating it.

## Language

### Validation roles

**Oracle**:
An independent code whose output a clax result is compared against.
_Avoid_: ground truth, reference code (when the role is meant)

**Primary oracle**:
CLASS (and CLASS-PT for one-loop spectra) — the code clax mirrors line by line.

**Second oracle**:
An independent implementation of the same theory (here ps_1loop_jax) used to
cross-check clax-pt; agreement is required, but it is not assumed correct.
_Avoid_: target, reference implementation

**Tie-breaker**:
The primary oracle's role when clax-pt and a second oracle disagree: whichever
code also disagrees with CLASS-PT carries the bug.

**Code-accuracy gate**:
A pass/fail tolerance on the difference between two codes, set by numerical
accuracy, independent of any survey.

**Error-bar gate**:
A pass/fail bound on the difference between two codes measured in a survey's
statistical errors: Δχ² = Σ_slices Σ_k ΔPᵀC⁻¹ΔP over (P₀,P₂,P₄), Gaussian
covariance with ℓ-ℓ′ blocks, Δk = 0.01 on 0.01 ≤ k ≤ 0.2 h/Mpc; passes at
Δχ² < 1 for the full seven-**PFS slice** survey.

**Counterterm headroom**:
The k² counterterm amplitude c at which the **counterterm model difference**
alone would bring the **error-bar gate** to Δχ² = 1.

**Shared-input comparison**:
A comparison in which both codes receive the identical linear P_cb(k,z) table,
growth rate f, r_s·h and AP ratios, so residuals are one-loop numerics only.
_Avoid_: end-to-end comparison (that runs each code from cosmological parameters)

### Spectrum decomposition

**Coefficient spectrum**:
The k-dependent spectrum multiplying one bias/EFT monomial (b1², b1·b2, cfog, …)
in a multipole P_ℓ; P_ℓ is a quadratic polynomial in the parameters with these
as coefficients.
_Avoid_: template (overloaded with emulator templates), component (overloaded with `pk_mult` rows)

**Basis translation**:
The map from ps_1loop's parameters (b1, b2, bG2, bGamma3, c0, c2, c4, cfog,
P_shot, a0, a2) to clax-pt's (b1, b2, bG2, bGamma3, cs0, cs2, cs4, b4, Pshot,
a0, a2), including the AP volume factor V = hratio/Dratio² on the stochastic
terms and the 0.25·b2²·P_δ²δ²(k→0) shift in Pshot (spec §4.1).

**Model difference**:
A difference between two codes that no parameter translation removes because
the codes implement different functions, not different numerics; it is
predicted in closed form and tested for equality, never absorbed in a tolerance.
_Avoid_: discrepancy, error (when the model differs)

**Structural zero**:
A coefficient spectrum that a code omits by construction (e.g. CLASS-PT's ℓ=4
F_G2 term); compared on the scale of the monomial's largest multipole.

**Counterterm map**:
The constant, f-dependent triangular map from ps_1loop's μ-polynomial k²
counterterms to clax-pt's per-multipole ones: cs0 = c0 + (f/3)c2 + (f²/5)c4,
cs2 = c2 + (6f/7)c4, cs4 = c4. Exact when the spectrum they multiply is isotropic.

**Counterterm model difference**:
The residual the **counterterm map** cannot remove: clax-pt projects each cs_ℓ
term onto its own ℓ only, ps_1loop projects the whole polynomial, so the
μ-dependent BAO damping (and, at z>0, AP) leaks c_j into other multipoles in
ps_1loop only. A model difference, computable in closed form — not a numerical error.
_Avoid_: "no exact mapping" (true only for the wiggle part)

### Survey

**PFS slice**:
One of the seven redshift slices of Takada et al. (2014, PASJ 66, R1) Table 2
(0.6–2.4), with its volume V, number density n̄ and linear bias b_g = 0.9 + 0.4z;
evaluated at the slice midpoint.
_Avoid_: PFS bin (ambiguous with k-bins), z-bin

## Relationships

- A validation compares **clax-pt** against one **second oracle** under a
  **code-accuracy gate**, and reports the residual under an **error-bar gate**.
- A **code-accuracy gate** failure is assigned to a code by the **tie-breaker**.

## Example dialogue

> **Dev:** "ps_1loop and clax-pt differ by 0.4% in P_2 at k=0.2 h/Mpc — clax-pt bug?"
> **Physicist:** "Ask the **tie-breaker**: if clax-pt matches CLASS-PT there, the
> **second oracle** carries it. Either way it's far inside the **error-bar gate**."

## Flagged ambiguities

- "reference" was used for both the oracle code and the stored `.npz` data —
  resolved: the code is an **oracle**; `reference_data/` holds its outputs.
