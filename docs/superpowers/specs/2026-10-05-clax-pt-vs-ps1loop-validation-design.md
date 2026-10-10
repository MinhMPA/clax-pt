# clax-pt vs ps_1loop_jax — second-oracle validation (design)

**Date:** Oct 5, 2026 · **Status:** approved in grilling session (Oct 3–5, 2026), Q1–Q12 below
**Glossary:** `CONTEXT.md` (oracle, second oracle, tie-breaker, code-accuracy gate,
error-bar gate, shared-input comparison, coefficient spectrum, basis translation,
counterterm map, counterterm model difference, counterterm headroom, PFS slice)
**Second oracle:** `~/ps_1loop_jax-for-pfs` (fork MinhMPA/ps_1loop_jax-for-pfs, v0.4.0, `270b2b5`)

## 1. Purpose

Cross-check clax-pt's one-loop EFT galaxy power spectrum against ps_1loop_jax, an
independent JAX implementation of the same theory (Chudaykin et al. 2020,
arXiv:2004.10607), with CLASS-PT as **tie-breaker**. Two gates:

- **Code-accuracy gate** — the difference between the two codes must sit inside the
  same tolerances clax-pt already meets against CLASS-PT.
- **Error-bar gate** — the remaining difference must shift a PFS full-survey fit by
  less than Δχ² = 1.

A failure is assigned to a code (or to a convention) by the tie-breaker protocol (§6).

## 2. Scope

In scope:
- **Shared-input comparison only** (Q2). Both codes receive the CLASS-PT reference
  file's own linear P_cb(k) [(Mpc/h)³ on `k_h` in h/Mpc], growth rate f (`fz`),
  r_s(z_d)·h (`rs_d·h`, Mpc/h) and AP ratios (`hratio`, `Dratio`). No Boltzmann solve.
- Redshift-space galaxy multipoles ℓ = 0, 2, 4. Matter multipoles are covered through
  the coefficient spectra multiplying b1⁰, b1¹, b1² (Q4).
- Real-space P_mm, P_gg, P_gm with **IR resummation off in both codes** (Q12) —
  ps_1loop's real-space path has no IR resummation, so with IR on the two differ by
  the whole BAO-damping effect, which is a model difference, not numerics.
- Forward-mode derivatives along f, α⊥, α∥, a k-stretch of P_lin and the P_lin
  amplitude (Q9).

Out of scope: end-to-end runs from cosmological parameters, ps_1loop's
`clax_adapter.py` (it feeds total-matter P_m, not P_cb — filed as a finding, §9),
bispectrum, primordial non-Gaussianity (`ps_1loop_lpng`).

## 3. Inputs

| Set | Cosmologies | Redshifts | Source | Used by |
|---|---|---|---|---|
| Campaign set (existing, 42 files) | 14 distinct (`vc.distinct_cases()`) | 0, 0.38, 0.8 | `reference_data/classpt/<case>/z{z:.3f}_ap_omfid0.31_cb.npz` | coefficient gate, closure, real space (§5.1–5.4) |
| PFS-slice set (new, 35 files) | `lcdm_fiducial`, `omega_cdm_low`, `massive_nu_015`, `massive_nu_030`, `w0wa_m07_m10` | 0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.2 | same layout, generated one z per call (Ruling 19) | error-bar gate (§5.6), ladder (§7) |
| IR-off set (new, 3 files) | `lcdm_fiducial`, `massive_nu_030`, `w0wa_m07_m10` | 0.38 | `z0.380_noap_cb_irno.npz`, `--class-extra '{"IR resummation": "No"}'` | real-space tie-breaker (§5.4) |

At z = 0 every file has `hratio = Dratio = 1`, so z = 0 is the no-AP baseline at all
14 cosmologies (Q3); ps_1loop is then run without its AP remap (`get_pk_ell`), and
with it (`get_pk_ell_ref`) at z > 0. Ωm,fid = 0.31 throughout.

**PFS slices** — Takada et al. (2014, PASJ 66, R1; arXiv:1206.0737), Table 2, 1464 deg²,
b_g(z) = 0.9 + 0.4z, evaluated at the slice midpoint:

| z | V [(Gpc/h)³] | N_g per field | n̄ [10⁻⁴ (h/Mpc)³] | b_g |
|---|---|---|---|---|
| 0.6–0.8 | 0.59 | 85 | 1.9 | 1.18 |
| 0.8–1.0 | 0.79 | 358 | 6.0 | 1.26 |
| 1.0–1.2 | 0.96 | 420 | 5.8 | 1.34 |
| 1.2–1.4 | 1.09 | 640 | 7.8 | 1.42 |
| 1.4–1.6 | 1.19 | 491 | 5.5 | 1.50 |
| 1.6–2.0 | 2.58 | 598 | 3.1 | 1.62 |
| 2.0–2.4 | 2.71 | 539 | 2.7 | 1.78 |

Transcription checks: n̄ = N_g · (1464/1.098 fields) / V to 3% (rounding), and
b_g = 0.9 + 0.4 z_mid to 0.006.

## 4. Parameter translation

### 4.1 Exact map (ps_1loop θ → clax-pt / CLASS-PT)

θ = (b1, b2, bG2, bGamma3, c0, c2, c4, cfog, P_shot, a0, a2), with ps_1loop's
`ndens` and `k_nl` (= 0.45 h/Mpc, the value clax-pt hard-codes). Write
α⊥ = `Dratio`, α∥ = 1/`hratio`, V = 1/(α⊥²α∥) = `hratio`/`Dratio`²
(ps_1loop.py:328-344 against ept.py:1135-1141).

| clax-pt | from ps_1loop | why |
|---|---|---|
| b1, b2, bG2, bGamma3 | identical | same Chudaykin et al. basis (ps_1loop.py:166-175 vs ept.py pk_gg_real) |
| cs0 | c0 + (f/3)c2 + (f²/5)c4 | Legendre content of μ², μ⁴ |
| cs2 | c2 + (6f/7)c4 | " |
| cs4 | c4 | " |
| b4 | cfog | μ⁴(b1+fμ²)² projections equal clax's (35/8)·poly_ℓ factors |
| Pshot | V·P_shot/ndens − 0.25·b2²·Pd2d2_0 | ps_1loop subtracts P_δ²δ²(k→0) from every 22 term and applies AP to stochastic terms; clax-pt adds 0.25 b2² Pd2d2_0 back (classy.pyx:4911) and adds Pshot in observed coordinates |
| a2 | V·hratio²·a2/ndens·(0.45/k_nl)² | k_t²μ_t² = k²μ²/α∥², pure μ² |
| a0 | V·a0/ndens·(0.45/k_nl)²·[(2/3)/Dratio² + (1/3)hratio²] | matches ℓ = 0 exactly |
| cs (real space) | 0 | see §5.4 |

Pd2d2_0 is each code's own (clax-pt: `EPTComponents.Pd2d2_0`; CLASS-PT: the file's).

### 4.2 Model differences (what the map cannot remove)

These are differences of *model*, not numerics. Each has a closed form, and §5.2 tests that the
measured difference equals it. Items 1-3 are evaluated on ps_1loop's own P_IR(k,μ). Items 4-5 use
clax's μ-loop inputs and clax's P_nw, which ps_1loop 0.5.0's P_nw matches to 1.4e-5 of P.

1. **k² counterterms** — clax-pt projects each cs_ℓ term onto its own ℓ only
   (ept.py:1207-1209, nonlinear_pt.c:4498-4552); ps_1loop projects the whole
   polynomial. The μ-dependent BAO damping e^{−k²Σ²_tot(μ)} (wiggle part) and, at
   z > 0, the AP mapping k_t(μ) leak c_j into other multipoles in ps_1loop only.
2. **k⁴ fingers-of-god** — clax-pt uses the ℓ = 4 projection `Pk_ctr4` times the
   isotropic factor for every ℓ, in observed k² (ept.py pk_gg_l0/l2/l4 `P_b4`);
   ps_1loop multiplies the full μ-dependent P_IR in true coordinates.
3. **a0 under AP** — ps_1loop's k_t² carries μ² dependence; clax-pt adds a0 to ℓ = 0
   in observed k. Leak: Δ₂ = (2/3)·V·a0/ndens·(hratio² − 1/Dratio²)·(k/k_nl)².
4. **Omitted projections** (13 lines; user decisions Oct 8, 2026). CLASS-PT, and clax-pt,
   which mirrors it, build these projections' integrands inside the 40-node μ-loop but never
   add them to P_ℓ. ps_1loop keeps them; the μ-dependent BAO damping and AP generate them.
   - F_G2 ℓ2/ℓ4: bG2:ℓ4, bGamma3:ℓ4, b1·bG2:ℓ2/ℓ4, b1·bGamma3:ℓ2/ℓ4.
   - Isotropic loops ℓ2/ℓ4: b2², bG2², b2·bG2.
   - b1·b2:ℓ4: CLASS-PT's computed-but-unused pm[40]; pm[41] P_4_b1bG2 enters b1·bG2:ℓ4.
   They are excluded from the §5.1 gate and asserted by closure (§5.2). The closed form
   rebuilds each integrand from clax's own μ-loop inputs, after a reproduction check
   (≤ 1e-10) of every projection clax keeps.
5. **IR-resummation scheme** (user decision Oct 10, 2026). The two codes resum the loops differently:
   - ps_1loop (and its successor elf): loop[P_nw] + e^{−x(μ)}(loop[P] − loop[P_nw]).
   - CLASS-PT and clax-pt: an isotropically damped input P_nw + e^{−k²Σ²}P_w for the bias loops, plus
     p_lo(μ) = (P_nw + e^{−x(μ)}P_w)/P_res on F_G2 only.

   The difference nearly cancels at ℓ = 0 and survives at ℓ = 2, 4, i.e. on the item-4 lines.
   Its closed form, X_ps = C[P_nw] + e^{−x(μ)}(C[P] − C[P_nw]), is built from clax's IR-off
   channels C[·] on the shared P and on clax's P_nw, with clax's Σ², δΣ² and AP geometry. The
   item-4 closure predicts ps − clax = ⟨X_ps⟩_ℓ − clax_ℓ.

## 5. Gates

### 5.1 Coefficient gate (code-accuracy)

P_ℓ is a polynomial in θ; each **coefficient spectrum** is extracted by an exact
finite stencil on {−1, 0, 1} per parameter (no truncation error; P_ℓ has degree ≤ 2
per parameter). Monomials: 1, b1, b1², b2, b1b2, b2², bG2, b1bG2, bG2², b2bG2,
bGamma3, b1bGamma3, P_shot, a2 (exact set); c0, c2, c4, cfog, cfog·b1, cfog·b1², a0
(model-difference set, §5.2).

- Window: CLASS-PT grid points `[NSIDE:]` with k ≤ 0.3 h/Mpc (`cu.window`).
- Metric: `max|ps − clax| / max|clax|` on the window (campaign metric). When clax-pt's
  coefficient is a **structural zero** (max|clax_ℓ| ≤ 1e-12 × the monomial's largest
  multipole), the denominator is that largest multipole instead.
- Thresholds: 1% for ℓ = 0, 2; 2% for ℓ = 4 (`cu.THRESHOLDS`). Only ever tightened, by
  ceil_to_1sf(2× worst) (C4 ratchet), never below 1e-3 (Ruling 20).
- A NaN/inf anywhere is a failure.

### 5.2 Model-difference closure

For each model-difference monomial: measured = coef_ps − coef_clax(translated);
predicted = closed form of §4.2 on ps_1loop's P_IR. Gate: `max|measured − predicted|`
normalised as in §5.1, same thresholds. The size of each model difference is logged.

### 5.3 Counterterm-map exactness with IR off

At z = 0 (no AP), ps_1loop `do_irres=False` and clax-pt `ir_resummation=False`: the
model-difference monomials must pass the plain coefficient gate — the map is exact
when P is isotropic.

### 5.4 Real space, IR off (42 files)

Coefficients of P_mm (1, c0), P_gg (b1², b1b2, b2², b1bG2, bG2², b2bG2, b1bGamma3,
P_shot) and P_gm (b1, b2, bG2, bGamma3), threshold 1%. Counterterms differ in
parameterisation (ps P_gg: −2c0k²P; clax: 2(cs·b1² + cs0·b1)·Pk_ctr), so only the
k²P shape is compared, once, through P_mm (cs0 = c0). Tie-breaker: the IR-off CLASS-PT
files (§3); clax-pt's IR-off path is first checked against them on its own.

### 5.5 Gradients

At `lcdm_fiducial`, `massive_nu_030`, `w0wa_m07_m10`, z = 0.38, at the PFS bias point
for z = 0.38: forward-mode tangents of P_ℓ along f, α⊥, α∥, a k-stretch
t = P·dlnP/dlnk and the amplitude t = P. Cross-code gate: §5.1 thresholds on the
tangent spectra. ps_1loop's own AD vs central differences (step 1e-4): 1% (CLAUDE.md
gradient target). clax-pt's own AD-vs-FD tests already exist.

### 5.6 Error-bar gate

- Bias point per slice: b1 = 0.9 + 0.4z; b2 = 0.412 − 2.143b1 + 0.929b1² + 0.008b1³
  (Lazeyras et al. 2016); bG2 = −(2/7)(b1−1); bGamma3 = (23/42)(b1−1) (coevolution);
  c0 = c2 = c4 = C_REF = 2 (Mpc/h)² (**placeholder** — replace with the forecast's
  prior centres); cfog = b4 = 500 (Mpc/h)⁴; P_shot = 1 with ndens = n̄ (Poisson);
  a0 = a2 = 0.
- Residual ΔP = (ΔP₀, ΔP₂, ΔP₄) = ps_1loop − clax-pt, at k-bin centres 0.015 … 0.195
  (Δk = 0.01, 0.01 ≤ k ≤ 0.2 h/Mpc), both codes interpolated from the window with
  the same cubic spline in ln k.
- Gaussian covariance per bin, from clax-pt's P(k,μ) = Σ P_ℓ L_ℓ (which already
  contains the shot noise — no extra 1/n̄):
  C_ℓℓ′(k) = (2ℓ+1)(2ℓ′+1)/N_k · ∫_{−1}^{1} L_ℓ L_ℓ′ P(k,μ)² dμ, N_k = V k² Δk/(2π²).
- **Pass:** Σ_slices Σ_k ΔPᵀC⁻¹ΔP < 1, for each of the 5 cosmologies.
- Reported, not gated: Δχ² per slice; max |ΔP_ℓ|/σ_ℓ per bin; **counterterm
  headroom** c* (all three c_j = c), from Δ(c) = Δ(0) + c·D (exact: linear in c).

## 6. Tie-breaker protocol

For every failing (monomial, ℓ), CLASS-PT's coefficient is rebuilt at the same
translated θ from the file's `pk_mult` rows (`scripts/classpt_assembly.py::
assemble_from_pm`), and the failure message carries ps-vs-CLASS-PT and
clax-vs-CLASS-PT errors with a verdict:

- **suspect ps_1loop** / **suspect clax-pt** — whichever departs further from CLASS-PT;
- **CLASS-PT convention** — CLASS-PT's coefficient is a structural zero and clax-pt
  agrees with it: the projection is omitted by construction (§4.2 item 4);
- **resolution** — the convergence ladder (§7) moves ps_1loop towards clax-pt and
  CLASS-PT;
- **tie-breaker n/a** — a0, a2 (CLASS-PT's twin has no scale-dependent stochastic rows).

A residual whose shape is k²·P_IR is degenerate with the counterterms (a UV
renormalisation convention); it is recorded as such, never absorbed by re-fitting.
No fudge factors; thresholds are never loosened to pass.

The tie-breaker says which code departs from CLASS-PT, not whether CLASS-PT's own
convention is right. Known oracle caveat: classy's `pk_gg_l4` omits `pk_mult` rows
40/41 (campaign oracle finding); the twin inherits it.

## 7. Convergence ladder (diagnostic, Q10)

Gates run at ps_1loop's defaults (what PFS users run). At `lcdm_fiducial` and
`w0wa_m07_m10`, z = 2.2, one setting changed at a time:
`nfft` 256→512; Simpson `num` 256→1024; AP μ grid 51→201 (test-only override of
`self._mu` before the first jitted call); wiggle-removal cuts 140–210 → CLASS-PT's
120–240; `rbao` = 110 (ps_1loop's default) instead of r_s·h. Each rung logs the
ps-vs-clax error per ℓ and the slice Δχ². Not a gate.

## 8. Infrastructure (Q11)

- Branch `feat/ps1loop-cross-validation` from fork `main` (MinhMPA/clax-pt, has PR #8's
  apparatus), worktree `/home/n2minh/clax-ps1loop`. Fetch over HTTPS through `gh`
  (no SSH key on this node).
- ps_1loop installed editable into env `clax` (`pip install --no-deps -e`); tests
  `pytest.importorskip("ps_1loop_jax")`; every log record carries ps_1loop's commit.
- Logs: `test_logs/ps1loop/errors.jsonl`. All JAX runs via sbatch on igpu nodes
  (`slurm/ps1loop-tests.sbatch`); CLASS-PT runs in env `classpt`
  (`slurm/classpt-refgen-pfs.sbatch`).
- `--fast`: `vc.FAST_CASES` × z = 0.38 for the 42-file tests; `lcdm_fiducial` for the
  PFS and gradient tests.

## 9. ps_1loop findings to file (drafted, posted only with approval)

1. `rbao=110` default vs r_s·h ≈ 99 Mpc/h (Planck 2018): Σ²_BAO uses the wrong BAO scale
   unless set; ladder rung quantifies it.
2. `utils_math.legendre(n, x)` returns L₄ for every n ∉ {0, 2}.
3. `get_pk_real` / `get_pk_gm_real` have no IR resummation (inconsistent with the
   IR-resummed multipoles).
4. `clax_adapter` uses total-matter P_m (not P_cb), `fast_cl` precision, k_max ~ 1 h/Mpc.
5. Methods are jitted with `static_argnames=['self']`: mutating `rbao`, `ks`, `_mu`
   after the first call is silently ignored.

## 10. Decision log

| Q | Decision |
|---|---|
| 1 | (C) code-accuracy gate + error-bar gate; CLASS-PT tie-breaker |
| 2 | (B) shared-input comparison only |
| 3 | (A) the 42 existing files; z = 0 is the no-AP baseline |
| 4 | (A) coefficient by coefficient, explicit translation; refined: counterterm map + closure |
| 5 | (B) new CLASS-PT runs at the PFS slices; Takada et al. 2014 Table 2 |
| 6 | (B) 5 cosmologies × 7 slices = 35 runs |
| 7 | (A) coevolution bias point, non-zero counterterms through the map; C_REF = 2 placeholder; headroom reported |
| 8 | (A) Δχ² < 1 over the survey, 0.01 ≤ k ≤ 0.2, Δk = 0.01 |
| 9 | (A) cross-code JVPs + ps_1loop AD-vs-FD, 3 cosmologies |
| 10 | (A) gates at ps_1loop defaults + convergence ladder |
| 11 | (A) new branch/worktree from fork main; tests in clax; ps_1loop editable install |
| 12 | (B) real space with IR off in both codes, 42 files |
