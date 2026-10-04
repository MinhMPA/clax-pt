# clax-pt validation campaign — design

**Date:** Sep 3, 2026
**Branch:** `campaign/clax-pt-validation` (worktree `/home/n2minh/clax-ptval`),
cut from `origin/main` = `MinhMPA/clax-pt` @ `9198580`.
**Status:** design approved in chat; this document is the spec that the
implementation plan is written from.

## 1. Goal

User request (verbatim): *"I want to review and test the fork clax-pt. In
particular, validate clax-pt against CLASS-PT for the power spectrum
multipoles ℓ=0,2,4 in 5 LCDM and 5 nuLCDM and 5 w0waCDM cosmologies."*

Deliverable: a committed, reproducible validation of clax-pt's redshift-space
multipoles P₀/P₂/P₄ (galaxy and matter) plus the three real-space spectra
against CLASS-PT over 15 cosmologies × 3 redshifts, with the one missing
physics piece — the Alcock–Paczynski (AP) distortion — implemented first so
the two codes compute the same observable. CLASS-PT is the oracle; every
residual above numerical noise is a bug to be located, never absorbed.

## 2. Verified current state (audited Sep 2–3, 2026)

- `origin/main` (clax-pt) = `fork/main` = `9198580`; contains `upstream/main`
  (smsharma/clax, `09062c1`) plus 3 fork-only commits. Local `main` was
  fast-forwarded to `9198580` on Sep 3. PRs #33–#38 are merged: stable
  reverse thermodynamics, EPT traced h-channels with the real background
  growth rate (`bg.f_of_loga`), traced IR resummation, Chebyshev k-sampling,
  multi-cosmology rule + fixtures, changelog date format.
- `clax/ept.py` implements CLASS-PT's *AP code path at α = 1*: the full
  anisotropic P(k, μ) with μ-dependent IR damping Σtot(μ), the 40-node
  Gauss–Legendre μ-quadrature from `pt_matrices/gauss_tab.dat` (`ept.py:67-73`),
  and Legendre projection for tree and 1-loop terms (`ept.py:~1225-1300`).
  `compute_ept` (`ept.py:1652`) and `compute_ept_from_clax` (`ept.py:2225`)
  take **no** `hratio`/`Dratio` inputs, so the AP *distortion* is absent.
  The comment near `Pk_4_b1b2` (`ept.py:~1409-1415`, "our reference was
  generated without AP") is wrong and must be corrected.
- `reference_data/classpt_z0.38_fullrange.npz` **was** generated with
  `'AP': 'Yes', 'Omfid': '0.31'` (generator since commit `291013b`;
  `docs/clax-pt.md` says so). Its fiducial has **no massive neutrino**
  (CLASS-PT defaults) and `A_s = 2.0989e-9`, whereas clax's default
  `CosmoParams` carries `N_ncdm=1, m_ncdm=0.06, N_ur=2.0328` and
  `ln10A_s = 3.0445224377` (`A_s = 2.1e-9`). Ω_m(true) ≈ 0.3138 vs
  Omfid 0.31 gives hratio − 1 ≈ +0.2 % at z = 0.38 — a sub-percent
  systematic hidden inside the current 1 %/2 % thresholds. Offset
  cosmologies (h+10 % → Ω_m ≈ 0.26) would fail spuriously under a fixed
  Omfid with α = 1 code, which is why AP comes first.
- CLASS-PT (`~/CLASS-PT`, base CLASS v3.3.4, upstream `09d5531a`) is present
  but **not built**; `classy` is not importable. Its `cb` flag
  (`input.c:3952`) switches the PT input from `delta_m` to `delta_cb`
  (`nonlinear_pt.c:1081, 1833-1843`); `classy.pyx` exposes `pk_cb`,
  `pk_cb_lin`, `sigma8_cb`. The growth rate used by the PT module is the
  background `index_bg_f` (`nonlinear_pt.c:1262`) — the same definition as
  clax's `bg.f_of_loga`.
- Multi-cosmology fixtures exist (`tests/conftest.py:131-153`):
  `COSMOLOGY_GRID_LCDM` (5 cases) and `COSMOLOGY_GRID_NULCDM` (4 cases),
  plus `cosmology_reference_dir(name)`. clax already has CPL dark-energy
  background (`background.py:392`) and fluid perturbations
  (`perturbations.py:341-345, 715-718`); the only existing w0wa CLASS
  reference is `reference_data/w0wa_m09_01/` built with
  `w0_fld=-0.9, wa_fld=0.1, Omega_Lambda=0.0`
  (`scripts/generate_selected_pk_references.py:81`).

## 3. Relationship to the 2026-09-02 sibling plans

Three plans written on Sep 2, 2026 overlap with this campaign. This spec is
the single design for the overlapping parts; the plan derived from it
replaces the overlapping tasks and leaves the rest untouched.

| Sibling plan | Decision |
|---|---|
| `2026-09-02-classpt-oracle-multiz.md` (oracle build, generator CLI, multi-z references) | **Adopted and extended.** Dedicated micromamba env `classpt` (py3.10, numpy<2, old Cython; never the user's `carpile`/`cosmopower`/`cosmodesi` envs); generator CLI `--z-list --cosmology --ap {yes,no} --omfid`; layout `reference_data/classpt/<cosmology>/z{z:.3f}_{ap\|noap}[_omfid{X}].npz` (this campaign appends `_{cb\|m}`, §6.4); `MANIFEST.md`. This campaign adds `--cb {yes,no}`, the 15-case table, stored AP ratios, and the provenance gate. Its DESI z-bins stay the likelihood track's concern; the same CLI produces them later. Its Task 5 (point `test_ept_accuracy.py` at a no-AP reference) is superseded by §7.2 (pass the true ratios instead). |
| `2026-09-02-ept-alcock-paczynski.md` (multipole-level AP, `clax/ap.py`, `pk_gg_multipoles_obs`) | **Superseded.** User decision Sep 3: "Mirror CLASS-PT in-loop". Its q-factor geometry is absorbed into `clax/ap.py::ap_ratios` (§5.1); its multipole-level remap and `pk_gg_multipoles_obs` are not built. Its `h_fid` argument is dropped: in h/Mpc units the AP ratios depend only on E(z) = H/H₀ (§5.1), so CLASS-PT's Omfid-only convention is exact. |
| `2026-09-02-fullshape-mock-likelihood.md` | **Unaffected in scope**; its dependency lines will be updated at plan-writing time to call `compute_ept_from_clax(..., omfid=...)` followed by the existing `pk_gg_l*` accessors. |

## 4. Locked decisions

### 4.1 Cosmology grid (15 cases)

All cases are clax `CosmoParams` overrides on top of clax's defaults
(h 0.6736, ω_b 0.02237, ω_cdm 0.1200, ln10A_s 3.0445224377, n_s 0.9649,
τ_reio 0.0544, N_ncdm 1, m_ncdm 0.06, N_ur 2.0328, w0 −1, wa 0). The table
lives in **one** pure-Python module, `scripts/validation_cosmologies.py`
(no JAX import, so the `classpt` env can import it), from which
`tests/conftest.py` re-exports `COSMOLOGY_GRID_LCDM`/`COSMOLOGY_GRID_NULCDM`
so the existing multi-cosmology fixtures keep working.

| Family | Case name | Overrides |
|---|---|---|
| ΛCDM | `lcdm_fiducial` | — |
| ΛCDM | `h_high` | h × 1.10 |
| ΛCDM | `omega_b_high` | ω_b × 1.20 |
| ΛCDM | `omega_cdm_low` | ω_cdm × 0.80 |
| ΛCDM | `ns_high` | n_s × 1.05 |
| νΛCDM | `massive_nu_006` | m_ncdm 0.06 (identical to fiducial; kept as the family's anchor) |
| νΛCDM | `massive_nu_015` | m_ncdm 0.15 |
| νΛCDM | `massive_nu_030` | m_ncdm 0.30 |
| νΛCDM | `massive_nu_015_h_high` | m_ncdm 0.15, h × 1.10 |
| νΛCDM | `massive_nu_015_omega_cdm_low` | m_ncdm 0.15, ω_cdm × 0.80 |
| w0waCDM | `w0wa_m09_p01` | w0 −0.9, wa +0.1 (same cosmology as the existing `reference_data/w0wa_m09_01/`) |
| w0waCDM | `w0wa_m11_m01` | w0 −1.1, wa −0.1 |
| w0waCDM | `w0wa_m10_p03` | w0 −1.0, wa +0.3 |
| w0waCDM | `w0wa_m10_m03` | w0 −1.0, wa −0.3 |
| w0waCDM | `w0wa_m07_m10` | w0 −0.7, wa −1.0 |

`lcdm_fiducial` and `massive_nu_006` are the same cosmology; the generator
writes the file once and the manifest records the alias. Net distinct
cosmologies: 14. The same module also carries `LEGACY_CLASSPT_FIDUCIAL`, a
raw CLASS-PT parameter dict (no massive neutrino, `A_s = 2.0989e-9`,
exactly the historical generator's `FIDUCIAL_PARAMS`) used only by the
Phase-0 provenance gate and by `tests/test_ept_accuracy.py`; it is not one
of the 15 cases and is never run through clax end-to-end.

### 4.2 Redshifts

z ∈ {0.0, 0.38, 0.8}. At z = 0 CLASS-PT sets hratio = Dratio = 1 by
construction (`nonlinear_pt.c:1234-1295`), so z = 0 is the AP-free control;
z = 0.38 connects to the legacy reference; z = 0.8 exercises the largest
AP/growth lever.

### 4.3 Species convention: cb everywhere

Every case has `N_ncdm = 1`, so the loop input is the **cb** (baryon+CDM)
spectrum on both sides: CLASS-PT with `cb: Yes`, clax with the new
`PerturbationResult.delta_cb` (§6.3). For `lcdm_fiducial` at z = 0.38 the
generator also writes a `cb=No` variant (total-matter P_m) so the report
quantifies the cb-vs-m difference once; that file is a diagnostic, not a
test target.

### 4.4 AP: in-loop mirror of CLASS-PT, fixed Omfid = 0.31

CLASS-PT is run with `AP: Yes, Omfid: 0.31` for every case (its default
fiducial). clax computes (hratio, Dratio) from its own background with the
same Omfid and applies them **inside the μ-loop** exactly as
`nonlinear_pt.c:4392-5317` does (§5.2). AP ratios are also stored in each
reference file so the ratio computation and the remap can be tested as
separate seams.

### 4.5 Growth rate

Background growth rate on both sides: CLASS-PT `index_bg_f`, clax
`bg.f_of_loga.evaluate(log(1/(1+z)))`. No "f_cb" variant is introduced.

### 4.6 Test structure: hybrid, reference-first

Two test layers per case × z:

1. **Stage-level** (`tests/test_ept_multicosmo.py`): inject CLASS-PT's own
   `k_h`, `pk_lin`, `fz`, `hratio`, `Dratio` into `compute_ept`; compare the
   9 spectra. Isolates the EPT layer (loop integrals, IR resummation, bias,
   counterterms, AP remap, projection).
2. **End-to-end** (`tests/test_ept_e2e_multicosmo.py`, `slow`, GPU): run
   clax background → thermodynamics → perturbations → `compute_ept_from_clax`
   at the same parameters; assert the seams first (f, P_cb,lin, hratio,
   Dratio against the stored CLASS-PT values), then the spectra.

### 4.7 Thresholds

Provisional gates are the current suite's: |Δ|/max|P| ≤ 1 % for ℓ = 0, 2 and
real-space spectra, ≤ 2 % for ℓ = 4, over k ≤ 0.3 h/Mpc. After the campaign
run, gates are ratcheted per spectrum to ≥ 2 × the measured worst case
across all 45 (case, z) pairs, recorded in the report. No fudge factors;
a case that cannot meet the provisional gate is bisected through the seams
(§8) until the first divergent quantity is found.

### 4.8 Environment and HPC rules

- CLASS-PT build and every `classy` run: env `classpt`, CPU sbatch.
- Every clax solve and every pytest that touches JAX: env `clax`, V100 igpu
  sbatch. Nothing heavy on the shared login node.
- Never touch jobs named `fnl_closure_resumable` or `p2_d*_real`/
  `p2_d*_fourier`, nor any job not submitted by this campaign.
- Search only within `.` and `../class_public-3.3.4/` (plus `~/CLASS-PT`
  for the oracle source); never `find /`.

### 4.9 Repository conventions

One draft PR on `MinhMPA/clax-pt` from `campaign/clax-pt-validation`; no
merges by the agent. Commits are one logical change each, all tests passing,
trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`, commit
messages via `-F` file. CHANGELOG entries use the `Mon D, YYYY` heading.

## 5. Conventions the oracle dictates

### 5.1 AP ratios (`nonlinear_pt.c:1234-1295`)

For z_pk = 0: hratio = Dratio = 1. Otherwise, with Omfid = Ω_fid:

- E_fid(z) = sqrt(Ω_fid (1+z)³ + (1 − Ω_fid) + Ω_γ (1+z_pk)⁴), where Ω_γ is
  the **true** cosmology's photon density (`pba->Omega0_g`) — a quirk to
  mirror, not a model choice.
- hratio = E_true(z) / E_fid(z), with E_true = H(z)/(100 h km/s/Mpc)
  (`nonlinear_pt.c:1276`).
- D_fid = ∫₀^z dz'/E_fid(z') by a trapezoid over CLASS-PT's Nz steps, with
  the photon term **frozen at (1+z_pk)⁴** inside the integrand — a second
  quirk to mirror.
- D_true = D_A(z)(1+z) H₀/c in the same dimensionless units
  (`nonlinear_pt.c:1288`); Dratio = D_true / D_fid.

Because clax-pt works in h/Mpc, both ratios depend only on E(z); a fiducial
h cancels identically. Standard-convention mapping: α∥ = 1/hratio,
α⊥ = Dratio.

`clax/ap.py::ap_ratios(bg, z, omfid) -> (hratio, Dratio)` implements this
from clax's differentiable background (H and D_A splines), including both
quirks, with the Nz and trapezoid rule matching CLASS-PT's constants. A
pure-NumPy twin of CLASS-PT's arithmetic, fed the same H and D_A values,
must agree with `ap_ratios` to 1e-10 (mirror test); agreement with the
CLASS-PT-stored ratios then measures only clax's background accuracy.

### 5.2 In-loop remap (`nonlinear_pt.c:4382-4500, 5222-5330`)

Per k-bin j and GL node (μ_i, w_i):

- ap_fac = sqrt(1/Dratio² + (hratio² − 1/Dratio²) μ_i²)
- k_true = k_j · ap_fac; μ_true = μ_i · hratio / ap_fac
- volume V = hratio / Dratio²
- Σtot = Σ_BAO (1 + f μ_true² (2 + f)) + f² μ_true² (μ_true² − 1) δΣ_BAO;
  damping exp(−Σtot k_true²) (`nonlinear_pt.c:4470-4481`)
- every k-function is evaluated at k_true by cubic-spline interpolation on
  the EPT k-grid (CLASS-PT: 87 arrays, `*_ap_out`): P_nw, P_w, the
  P22/P13/P12 μ-power components (dd/vd/vv, wiggle and no-wiggle), the
  counterterm kernels, IFG2, and the b2/bG2 cross-terms — the latter
  reconstructed at (k_true, μ_true) from their own ℓ ≤ 4 multipoles
  (`P_{0,2,4}_b2`, `P_{0,2,4}_bG2`, `P_{0,2}_b1b2`, `P_{0,2}_b1bG2`) inside
  the same loop, which is how CLASS-PT itself treats them
- multipoles accumulate as P_ℓ(k_j) += w_i (2ℓ+1)/2 L_ℓ(μ_i) V P(k_true, μ_true)

At hratio = Dratio = 1 this reduces to interpolation at the grid nodes
(exact for a cubic spline), so the α = 1 outputs must equal the current
code's to round-off. The loop is restricted to CLASS-PT's interior bins
(`index_j = Nside..Nmax−Nside`); the comparison window k ≤ 0.3 h/Mpc lies
inside it.

### 5.3 CLASS-PT parameter mapping

Single source of truth is the clax-named table (§4.1); the generator
converts: `A_s = 1e-10·exp(ln10A_s)`; `N_ncdm, m_ncdm, N_ur, T_ncdm=0.71611`
passed through (matching `scripts/generate_class_reference.py:26-29`);
w0wa as `w0_fld, wa_fld, Omega_Lambda=0.0` (CLASS defaults `use_ppf=yes`,
`cs2_fld=1`, matching `generate_selected_pk_references.py:81`); PT block
`'output':'mPk', 'non linear':'PT', 'IR resummation':'Yes',
'Bias tracers':'Yes', 'RSD':'Yes', 'AP':'Yes', 'Omfid':'0.31',
'cb':'Yes'`; z via `z_pk`; bias values as today (b1 2.0, b4 500.0, others
0). The output k grid is the existing generator's `k_h` array
(`initialize_output(k_1Mpc, z, len(k_h))`).

## 6. Architecture

```
scripts/validation_cosmologies.py ──► scripts/generate_classpt_reference.py (env classpt, CPU sbatch)
        │                                        │
        │                                        ▼
        │                    reference_data/classpt/<case>/z{z}_{ap|noap}[_omfid0.31][_cb|_m].npz
        │                                        │
        ▼                                        ▼
tests/conftest.py ──► tests/test_ept_multicosmo.py   (stage: inject k_h, pk_lin, fz, hratio, Dratio)
        │             tests/test_ept_e2e_multicosmo.py (slow/GPU: full clax pipeline, seam asserts)
        │             tests/test_ap.py, tests/test_ept_ap.py
        ▼
clax/ap.py::ap_ratios ──► clax/ept.py::compute_ept(..., hratio, Dratio) ◄── clax/perturbations.py::delta_cb
                          clax/ept.py::compute_ept_from_clax(..., omfid=None, field="cb")
        │
        ▼
slurm/*.sbatch ──► scripts/summarize_ept_validation.py ──► docs/validation/2026-09-clax-pt-multipoles.md
```

### 6.1 `clax/ap.py` (new, small)

`ap_ratios(bg, z, omfid)` per §5.1, pure JAX, differentiable through `bg`.
Depends only on `BackgroundResult` splines. Unit-tested in isolation.

### 6.2 `clax/ept.py` (modified)

- `compute_ept(pk_lin_h, k_h, h, f, prec, _ir_precomputed, rs_h,
  hratio=1.0, Dratio=1.0)` — traced scalars; §5.2 inside the existing GL
  loop. IFG2 and the counterterms move into the loop (their CLASS-PT home);
  the b2/bG2 cross-terms are reconstructed from their multipoles at the
  remapped node inside the loop (§5.2).
  Where a μ-power component is currently held only as pre-projected
  multipoles, the refactor exposes the μ-power decomposition CLASS-PT uses
  (P22_mu{0,2,4,6,8}, P13_mu{0,2,4,6}, with wiggle/no-wiggle splits) so it
  can be interpolated at k_true.
- `compute_ept_from_clax(params, bg, pt, z, prec, *, omfid=None,
  field="cb")` — `omfid=None` → ratios (1, 1); otherwise `ap_ratios`.
  `field` ∈ {"cb", "m"} selects `pt.delta_cb` / `pt.delta_m`.
- `pk_mm_l*`, `pk_gg_l*`, `pk_gm_real`, `pk_gg_real`, `pk_mm_real`
  unchanged: they read already-distorted components.
- Correct the stale provenance comment near `Pk_4_b1b2`.

### 6.3 `clax/perturbations.py` (modified)

`PerturbationResult.delta_cb: Float[Array, "n_k n_tau"]` extracted next to
`delta_m` using the existing baryon+CDM combination
(`perturbations.py:1773`), for every `N_ncdm`. Same τ-grid, same
normalization as `delta_m`, so `compute_ept_from_clax` treats the two
fields identically.

### 6.4 `scripts/generate_classpt_reference.py` (modified)

CLI `--z-list --cosmology --ap {yes,no} --omfid --cb {yes,no}`; case table
from `scripts/validation_cosmologies.py`; output file
`reference_data/classpt/<case>/z{z:.3f}_{ap|noap}[_omfid{X}]_{cb|m}.npz`.
Saved keys: existing `k_h, pk_lin, fz, pk_mm_real, pk_gg_real, pk_mg_real,
pk_mm_l0/l2/l4, pk_gg_l0/l2/l4, pk_mult`, plus `D_z, H_z, DA_z, hratio,
Dratio, omfid, ap, cb, params_json, bias_json, classpt_commit`.
`hratio`/`Dratio` are CLASS-PT's **own** values: the code stores them in
`pnlpt->hratio_array[i_z]`/`Dratio_array[i_z]` (`include/nonlinear_pt.h:589-590`)
but `classy.pyx` does not expose them, so the build recipe carries a
read-only accessor patch (`scripts/classpt_patches/classy_ap_ratios.patch`,
adding `get_ap_ratios(z)` to `classy.pyx`; no physics touched). The
generator asserts the accessor's values against the §5.1 NumPy twin fed
with `classy` background outputs, so a transcription error in the twin is
caught at generation time. `reference_data/classpt/MANIFEST.md` maps
file → exact invocation and CLASS-PT commit + patch hash.

### 6.5 Tests (new)

- `tests/test_ap.py`: z = 0 identity; Omfid = Ω_m(true) gives
  |ratio − 1| below a measured bound; NumPy-twin mirror at 1e-10;
  finite-difference gradients of both ratios w.r.t. h, ω_cdm, w0, wa.
- `tests/test_ept_ap.py`: α = 1 regression — all `EPTComponents` arrays
  from the new loop vs. a baseline npz captured from the pre-refactor code
  at fiducial z = 0.38 (`reference_data/ept_alpha1_baseline.npz`, committed
  before the first loop edit), equal to round-off (≤ 1e-10 relative; the
  only permitted differences are summation-order effects, since spline
  evaluation at the nodes and 40-node GL projection of ≤ degree-12
  μ-polynomials are both exact); α ≠ 1 stage-level comparison against the
  regenerated fiducial z = 0.38 reference; finite-difference gradient of
  P₂ w.r.t. hratio and Dratio.
- `tests/test_ept_multicosmo.py`: §4.6 layer 1, parametrized over the 15
  cases × 3 z; skips (with reason) when a reference file is absent.
- `tests/test_ept_e2e_multicosmo.py`: §4.6 layer 2, marked `slow`; `--fast`
  runs `lcdm_fiducial`, `massive_nu_015`, `w0wa_m07_m10` at z = 0.38 only.

### 6.6 Campaign runner

`slurm/classpt-refgen.sbatch` (CPU, env `classpt`: builds nothing, only
generates), `slurm/ept-multicosmo-e2e.sbatch` (V100, env `clax`: runs both
test layers with `--junitxml` and a per-spectrum error table to
`test_logs/`), and `scripts/summarize_ept_validation.py` producing
`docs/validation/2026-09-clax-pt-multipoles.md`: one table per family
(case × z × spectrum → max |Δ|/max|P|, k of the maximum), the seam table,
the cb-vs-m diagnostic, the α = 1 vs AP delta at fiducial, and the
ratcheted thresholds.

## 7. Phases and testing strategy

### Phase 0 — Oracle build and provenance gate
Build CLASS-PT in env `classpt`; regenerate
`classpt_z0.38_fullrange.npz` with its **original** parameters through the
new CLI; gate: all 9 spectra agree with the committed legacy file to 1e-6
relative. Proves the fresh build reproduces the historical oracle before any
new reference is trusted.

### Phase 1 — References
Generate 14 distinct cosmologies × 3 z with `--ap yes --omfid 0.31 --cb
yes`, plus `lcdm_fiducial` z = 0.38 `--cb no` and `--ap no` diagnostics.
Validate every file loads with all keys; `fz` increases with z; `pk_lin` > 0;
hratio = Dratio = 1 at z = 0. Commit npz files and manifest.

### Phase 2 — AP in clax-pt (test-first)
`clax/ap.py` + `tests/test_ap.py`, then the §5.2 loop change guarded by the
α = 1 regression test, then `tests/test_ept_ap.py` against the regenerated
fiducial reference. `test_ept_accuracy.py` switches to the regenerated
legacy file and passes the true ratios — it becomes the first α ≠ 1
regression and its residual is expected to drop below today's 0.5–1.4 %.

### Phase 3 — `delta_cb` and `compute_ept_from_clax` plumbing
Test-first: at z = 0 the ratio δ_cb/δ_m → 1 as k → 0 and is above 1 at
k = 0.3 h/Mpc (δ_ν < δ_cb above the free-streaming scale, so δ_m < δ_cb),
with |δ_cb/δ_m − 1| increasing monotonically across
m_ncdm ∈ {0.06, 0.15, 0.30} (free-streaming suppression grows with f_ν);
the linear P_cb it produces matches CLASS-PT's stored `pk_lin` (cb) at the
§7 seam threshold; `field="cb"` and `omfid` wiring in
`compute_ept_from_clax`.

### Phase 4 — Multi-cosmology tests
Both layers of §6.5; `--fast` subsets; concise output (≤ 10 lines on pass,
≤ 20 on failure, greppable `ERROR` lines; arrays only in `test_logs/`).

### Phase 5 — Campaign run, ratchet, report, PR
sbatch both layers; summarize; ratchet thresholds; fix the stale comment;
CHANGELOG entry; draft PR. Threshold ratcheting is a separate commit from
any bug fix it motivated.

Seam thresholds (provisional, ratcheted like the spectra): f within 1e-3
relative; P_cb,lin within 0.1 % for k ≤ 0.3 h/Mpc (CLAUDE.md P(k) target);
hratio, Dratio within 1e-4 of the stored CLASS-PT values.

## 8. Failure policy

A failing (case, z, spectrum) is bisected in this order, each step a
separate assertion so the first divergent seam is named in the failure
message: background (H, D_A → hratio, Dratio) → growth rate f → linear
P_cb → EPT at injected inputs (stage layer) → EPT through the pipeline.
A bug found upstream of `ept.py` is fixed in its own commit with its own
test; a bug in CLASS-PT itself (if the bisection proves it) is documented in
the report, not worked around in clax. The AP path has its own bisection:
α = 1 regression → ratio mirror → single-node remap arithmetic → full
multipoles.

## 9. Deliverables

1. `clax/ap.py`, `clax/ept.py` (in-loop AP, `field`, `omfid`),
   `clax/perturbations.py` (`delta_cb`), with tests.
2. `scripts/validation_cosmologies.py`, generalized generator, 14 × 3
   reference files + diagnostics + `MANIFEST.md`, CLASS-PT build notes
   (`docs/classpt-build-notes.md`, env recipe).
3. `tests/test_ap.py`, `tests/test_ept_ap.py`, `tests/test_ept_multicosmo.py`,
   `tests/test_ept_e2e_multicosmo.py`; `tests/test_ept_accuracy.py` moved to
   true ratios.
4. `slurm/classpt-refgen.sbatch`, `slurm/ept-multicosmo-e2e.sbatch`,
   `scripts/summarize_ept_validation.py`,
   `docs/validation/2026-09-clax-pt-multipoles.md`.
5. CHANGELOG entry; one draft PR on `MinhMPA/clax-pt`.

Planning note: this is one implementation plan. If it exceeds ~15 tasks,
split it at the Phase 2/Phase 3 boundary (oracle + references + AP first;
`delta_cb` + multi-cosmology tests + campaign second) — the second plan
consumes the first's committed references and AP API unchanged.

## 10. Out of scope

Multipole-level AP; a fiducial h or fiducial r_d; per-cosmology Omfid; DESI
z-bins and the mock likelihood; bispectrum; ℓ > 4; changing CLASS-PT's
physics or numerics (the only CLASS-PT modification is the read-only
`get_ap_ratios` accessor patch of §6.4); any clax accuracy work not exposed
by a seam failure in this campaign.

## 11. Risks

- **ept.py refactor breadth.** Moving IFG2/counterterms into the loop and
  exposing μ-power components touches the assembly of most spectra. Guard:
  the α = 1 bit-for-bit regression test is written before the first
  loop edit and stays green at every commit.
- **Interpolation flavour.** CLASS-PT uses a natural cubic spline in linear
  k on its own log grid; clax-pt's grid is Chebyshev-sampled. Residual
  differences at k_true are below the 1 % gate but may set the ratchet
  floor; the report states the measured floor.
- **CLASS-PT build fragility** (Cython/numpy pins). Isolated in env
  `classpt`; recipe committed; the provenance gate catches silent
  numerical drift.
- **Fiducial mismatch with the legacy reference** (no ν, A_s 2.0989e-9).
  The legacy file is used only by the provenance gate and the regenerated
  stage-level test, where `pk_lin` and `fz` are injected, so the mismatch
  cannot leak into the campaign.
- **w0wa perturbations.** clax's fluid treatment vs CLASS's PPF may differ
  at the 0.1 % level in P_cb,lin; the seam threshold isolates this from the
  EPT layer, and any excess is a clax finding recorded, not absorbed.
- **GPU time.** 45 e2e solves at fit-grade precision ≈ 45 × (30–60 s) on a
  V100 — one sbatch job; the `--fast` subset keeps the development loop
  under two minutes.

## 12. Success criteria

- Provenance gate passes (legacy reference reproduced to 1e-6).
- α = 1 regression: new loop equals the pre-refactor baseline to round-off
  (≤ 1e-10 relative).
- `ap_ratios` mirrors CLASS-PT's arithmetic to 1e-10 and agrees with
  CLASS-PT's stored ratios at the seam threshold; gradients pass
  finite-difference checks at < 1 %.
- All 15 cases × 3 z (45 test parametrizations over 42 distinct reference
  files) pass both test layers at the provisional gates, then at the
  ratcheted gates, with the seam assertions green.
- `test_ept_accuracy.py` residuals shrink once true ratios are passed
  (expected; recorded before/after).
- The report table, the manifest, and the reproducibility recipe are
  committed; `pytest tests/ --fast -x -q` is green on the branch.
