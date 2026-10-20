# The l > 1000 residual against CLASS: diagnosis (Oct 12, 2026)

Status: COMPLETE (Oct 12, 2026, 03:00 JST); igpu V100 jobs 24610 (legacy trigger)
and 24620 (CLASS trigger), each 60/120/240 k per decade. Branch
`exp/highl-investigation` (from `exp/precision-sweep` cdda459), worktree
`/lustre/work/n2minh/clax-wt-highl`, results under `/lustre/work/n2minh/std/clax/highl/`.
Regenerate the C_l tables with
`python diags/diag_highl_report.py /lustre/work/n2minh/std/clax/highl/results-*.json`.

## Question

After the Oct 10-11 precision sweep (`docs/precision_sweep_2026-10-10.md`, findings 6
and 8) the unlensed C_l of the massless-neutrino RECFAST set agree with
`reference_data/cls_massless_recfast.npz` to 0.07% (l < 30), 0.24% (l <= 500), 0.27%
(l <= 1000) in TT and 0.3% in EE, with either the exact-Bessel path or the fast path at
30 table points per unit x and n_k_fine 20000. Above l ~ 1000 both paths share a
residual: TT rms 1.7% (l = 1000-2000) and 2.0% (2000-2500), EE 3.0% and 8.2%, with
TT at -3.6%/-3.8% at l = 1900/2000 and EE at -9%/-15%/+11% at l = 1800/2150/2300.
Which stage is it, and what fixes it?

## Answer

The bulk is the second trigger of the radiation streaming approximation (RSA) in
`clax/perturbations.py`. CLASS switches photons to the streaming solution once
k tau > 45 **and** tau > tau_free_streaming, the time at which the Thomson mean free
path exceeds five conformal times, tau_c/tau = 1/(kappa' tau) > 5
(`thermodynamics.c`; precision parameter `radiation_streaming_trigger_tau_c_over_tau
= 5`). clax implements the second condition as kappa'/(aH) < 5. In matter domination
aH = 2/tau, so this is kappa' tau < 10, fifty times earlier in kappa' tau: at the
fiducial cosmology it fires at tau = 296 Mpc (z = 1013), where the visibility function
is still 69% of its peak and 36% of its integral is still to come, instead of CLASS's
tau = 365 Mpc (z = 760, visibility 2.4% of its peak, 8% of the integral to come). For
every k > 0.152/Mpc the photon quadrupole/polarization source g*Pi is set to zero and
delta_g replaced by its streaming value in the middle of recombination; for 0.123 < k
< 0.152/Mpc the k tau > 45 condition places the switch at tau = 45/k, still inside the
visibility tail. The onset in l is k chi_* ~ 0.12 x 13893 ~ 1700, where the residual
appears.

Measured against CLASS's own source functions (`get_sources()`, same cosmology),
clax's g*Pi at k = 0.15/Mpc drops from 45% of its peak amplitude to 1e-25 between
tau = 292 and 303 Mpc and the T0 source loses 25-45% of its local value after the
switch, while at k <= 0.12/Mpc all four sources agree with CLASS to 0.05-0.2% rms.
With CLASS's criterion (module knob `_RSA_TRIGGER = "class"`, diagnostic only) the
sources agree to 0.07-0.28% rms at every node up to k = 0.3/Mpc, and the C_l residual
collapses to a smooth +0.3% to +0.5% offset:

| l band | TT rms before | TT rms after | EE rms before | EE rms after |
|---|---|---|---|---|
| 2-1000 | 0.15% | 0.17% | 0.23% | 0.23% |
| 1000-2000 | 1.73% (max 4.10) | 0.38% (max 0.49) | 2.96% (max 8.94) | 0.39% (max 0.69) |
| 2000-2500 | 2.04% (max 3.66) | 0.49% (max 0.99) | 8.19% (max 15.6) | 0.52% (max 0.71) |

Residual clax/CLASS - 1 at l = 1000, 1100, ..., 2500 (percent):

```
legacy  TT: +0.15 +0.10 -0.00 -0.26 -0.29 -0.56 -1.57 -1.88 -1.92 -3.45 -3.66 -1.73 -1.22 -3.17 -0.29 -0.59
legacy  EE: +0.27 +0.22 +0.05 +0.12 -0.37 -1.66 -0.98 -2.35 -8.93 -2.98 +2.12 -7.26 -6.20 +11.24 +2.27 +1.07
class   TT: +0.29 +0.28 +0.29 +0.34 +0.37 +0.37 +0.39 +0.43 +0.47 +0.43 +0.39 +0.45 +0.49 +0.46 +0.43 +0.45
class   EE: +0.29 +0.31 +0.30 +0.35 +0.36 +0.25 +0.35 +0.37 +0.38 +0.41 +0.49 +0.46 +0.47 +0.51 +0.52 +0.44
```

(Job 24610 legacy, job 24620 CLASS trigger; both G6 settings of the sweep: l_max
12/10/17, Kvaerno5, rtol 1e-4, atol 1e-6, 60 k per decade, n_tau 5000, th_n_points
100000, fast path with 30 table points per unit x and n_k_fine 20000, l_max 2500; same
perturbation cost, 173 vs 174 s on the V100.)

Two smaller terms remain. The perturbation k density: 60 per decade is 7.5 points per
acoustic period at k = 0.15/Mpc and biases C_l low by +0.08% (l = 1500), +0.24-0.31%
(2000), +0.60% (2500) in TT and +0.06, +0.17, +0.48% in EE relative to 120 per decade
(same numbers with either trigger; 240 per decade equals 120 within 0.04% TT, 0.14%
rms EE at 2000-2500). With the CLASS trigger and 120 per decade the residual is a
smooth +0.32% (l = 1000), +0.45% (1500), +0.63% (2000), +1.06% (2500) in TT and
+0.30, +0.42, +0.65, +0.92% in EE (band rms 0.50% / 0.89% TT, 0.46% / 0.82% EE).
Subtracting the +0.22% already present at l = 500-1000, that is +0.10, +0.23, +0.41,
+0.83% at l = 1000, 1500, 2000, 2500, against +0.13, +0.30, +0.53, +0.83% predicted
from the helium-era opacity: clax's x_e is 1.3-3.5% above CLASS RECFAST at z = 1300-3500
(HeI recombination; `_recfast_dxHe_dlna` is the Heflag = 0 form while CLASS runs
Heswitch = 6), which raises kappa' there, and 31% of the Silk damping integral
k_D^-2 accumulates at z > 1300. The result is k_D(tau_*) = 0.138307 vs 0.138137 /Mpc
(+0.12%), i.e. clax is less damped by 2 (k/k_D)^2 dln k_D in the sources: predicted
+0.30% at k = 0.15, +0.51% at 0.2, +1.2% at 0.3/Mpc; measured source amplitude at
the visibility peak, clax/CLASS, with the CLASS trigger: +0.27%, +0.50%, +0.83%
(T0) and +0.32%, +0.61%, +0.92% (g*Pi). In C_l that is 4 (k/k_D)^2 dln k_D ~ +0.5%
at l = 2000, the observed offset. The +0.2-0.3% already present at l = 500-1000 is
the low-k end of the same effect plus the sweep's documented residual.

## Ranked causes

| rank | cause | evidence | share of the l = 1000-2500 residual |
|---|---|---|---|
| 1 | RSA second trigger kappa'/(aH) < 5 instead of CLASS's kappa' tau < 0.2 (tau_c/tau > 5) | sources vs CLASS at matched (k, tau): g*Pi zeroed at tau ~ 296-300 Mpc for k > 0.15; C_l with the CLASS trigger: TT rms 1.73 -> 0.38% and 2.04 -> 0.49%, EE 2.96 -> 0.39% and 8.19 -> 0.52% (60/decade); max 4.1 -> 0.5% TT, 15.6 -> 0.7% EE | all of the oscillating part: 78% of the TT rms (1000-2000), 76% (2000-2500), 87% and 94% of the EE rms; the rest is smooth |
| 2 | Helium-era kappa' (x_e 1.3-3.5% off CLASS RECFAST at z = 1300-3500) | k_D +0.12%; source amplitude excess growing as 2 (k/k_D)^2 dln k_D; after the trigger fix and at 120/decade the residual above the l <= 1000 baseline is +0.10 / +0.23 / +0.41 / +0.83% at l = 1000 / 1500 / 2000 / 2500 vs +0.13 / +0.30 / +0.53 / +0.83% predicted; present on the #48 branch too | the remaining smooth 0.4-0.8% |
| 3 | Reference precision | CLASS default vs cl_ref.pre-like: TT max 0.07% (1000-2000), 0.09% (2000-2500), EE 0.14% and 0.38% (single-l maxima at EE troughs); CAMB 1.6.5 at AccuracyBoost 3 agrees with the reference to 0.05-0.09% TT, 0.13-0.37% EE | <= 0.1% TT, <= 0.4% EE at single l; rms 0.04-0.05% |
| 4 | Perturbation k density (60/decade = 7.5 points per acoustic period at k = 0.15, 5.6 at 0.2) | C_l 120 vs 60 per decade: +0.08% (l = 1500), +0.24-0.31% (2000), +0.60% (2500) TT, +0.06 / +0.17 / +0.48% EE, with either trigger; 240 = 120 within 0.04% TT (0.14% rms EE at 2000-2500); log-k spline error on CLASS's 6499-mode sources 0.03-0.09% rms of peak at k = 0.15-0.2 (rms over the whole tau window, which under-weights the visibility peak by ~2x) | a bias low of 0.2-0.6% at l = 2000-2500; cured by 120/decade (+56 s) |
| 5 | Fine k grid, tau grid, Bessel path | n_k_fine 40000 vs 20000 (tau grid thinned to 2500): 0.000%; tau grid 2500 vs 5000: 0.04-0.06% TT, 0.05% (1000-2000) and 0.25% (2000-2500) EE; fast vs exact-Bessel at l = 1650-2300: 0.001-0.005% | none |
| 6 | Sparse-l interpolation | residual rms at the l = 50n nodes 1.86% (TT) / 5.25% (EE) vs 1.88 / 5.30 at the midpoints (l = 1000-2500) | none |

Hypothesis 1 of the brief (recombination history / visibility) is excluded as the bulk:
x_e, kappa' and g agree with CLASS RECFAST to 0.01-0.03% at z = 800-1150, the
visibility peak (280.691 Mpc), mean (291.45) and rms width (26.85 Mpc) agree to
1e-5, and the k_D offset has the wrong sign for a deficit; the x_e oracle injection
was therefore not needed. Hypothesis 2 (the reference) is excluded by regeneration at
higher precision and by CAMB. Hypotheses 3 and 5 are at or below 0.1%.

## What was done

### Reference provenance and precision (login node; classy 3.3.4 in env carpile, CAMB 1.6.5 in env cosmodesi)

`cls_massless_recfast.npz` had no generating script in the tree (it came in with
commit 1cd05be, Feb 11, 2026, from a since-deleted `scripts/gen_massless_ref.py`).
`diags/diag_class_massless_ref.py` reproduces it to 5e-6 in TT and EE at every
l <= 2500 with CLASS 3.3.4 at **default precision** and `N_ur = 2.0328, N_ncdm = 1,
m_ncdm = 0, T_ncdm = 0.71611` (N_eff = 3.046, YHe = 0.245301, 511 k modes, 604 source
times, 20 k points per acoustic period above k = 0.07/Mpc); `N_ur = 3.046` alone
differs by 0.04-0.08% in TT. Against this default run, cl_ref.pre-like settings move
the spectra by (max |ratio - 1|, l = 1000-2000 / 2000-2500):

| CLASS settings | TT | EE |
|---|---|---|
| k sampling only (`k_step_sub` 0.015, `k_step_super` 1e-4: 6499 modes) | 0.004% / 0.047% | 0.006% / 0.003% |
| ODE, hierarchy, approximation triggers only (`tol_perturbations_integration` 1e-6, l_max 50/25/50, rsa trigger 240) | 0.023% / 0.008% | 0.105% / 0.055% |
| transfer sampling only (`hyper_sampling_flat` 12, `q_linstep` 0.2, `l_linstep` 25) | 0.046% / 0.096% | 0.105% / 0.361% |
| all of the above | 0.065% / 0.089% | 0.142% / 0.378% |

(`k_step_sub` 0.025 or 0.0075 change nothing beyond 0.004%: CLASS's 20 points per
acoustic period with cubic-spline source interpolation are converged.) CAMB (RECFAST,
`AccuracyBoost` 3, `lAccuracyBoost` 3, `lSampleBoost` 2) agrees with the CLASS
reference to 0.048% max (1000-2000) and 0.086% (2000-2500) in TT, 0.13% and 0.37% in
EE, after an overall 0.04% normalisation. The reference is converged at the 0.1% (TT)
/ 0.4% (EE, single-l at troughs) level over l = 1000-2500.

### Thermodynamics (CPU, `diags/diag_thermo_vs_class.py`)

clax's thermodynamics splines evaluated at the redshifts of CLASS's RECFAST table
(same cosmology; th_n_points 100000), max |clax/CLASS - 1|:

| z | x_e | kappa' | exp(-kappa) | g |
|---|---|---|---|---|
| 800-950 | 0.004% | 0.015% | 0.002% | 0.015% |
| 950-1050 | 0.005% | 0.016% | 0.007% | 0.014% |
| 1050-1150 | 0.009% | 0.019% | 0.030% | 0.011% |
| 1150-1300 | 0.016% | 0.027% | 0.16% | 0.13% |
| 1300-1600 | 1.63% | 1.62% | 12.5% | 11.1% |
| 1600-2000 | 3.43% | 3.42% | 69% | 69% |
| 2000-3500 | 1.30% | 1.29% | 77% | 77% |
| 3500-10000 | 0.15% | 0.16% | 75% | 75% |

(c_s^2 / CLASS c_b^2 = 0.9995-0.9998 at z = 800-1300; T_b agrees to 1e-6 above
z = 12.) Visibility peak at tau = 280.691 Mpc in both, mean 291.45 and rms width
26.85 Mpc in both (500 < z < 2000), integral 1.000000 in both; tau(z) and H(z) agree
with CLASS to 1e-6 at all z. The z > 1300 differences are where exp(-kappa) < 1e-4, so
they do not change the sources directly, but kappa' enters the Silk damping integral
k_D^-2 = int dtau (R^2 + 16(1+R)/15) / (6 kappa' (1+R)^2): 31% of it accumulates at
z > 1300 and 16% at z > 1600. Result: k_D(tau_*) = 0.138307 (clax) vs 0.138137 /Mpc
(CLASS). The same differences are present with the `fix/recombination-staging` (#48)
code (x_e 1.65% / 3.47% / 1.35% in the three bands), so they are a separate
helium-physics item, not the hydrogen staging.

### Source functions against CLASS (`diags/diag_sources_vs_class.py`)

CLASS's `get_sources()` (t0, t1, t2, p on 511 k x 604 tau) against the clax dumps
(300 k x 5000 tau), cubic-spline interpolated to the same (k, tau). Conventions: t0 =
source_T0, t1 = source_T1, t2 = source_T2/8 (clax stores g*Pi, CLASS g*P with
P = Pi/8), p = source_E * 2 sqrt(6)/3. At clax's own k nodes (CLASS interpolated from
its 20-points-per-period grid onto the node, so clax's sparse-grid interpolation is
not in the test); rms and max difference over tau = 230-450 Mpc in percent of the
CLASS peak amplitude:

| k node /Mpc | legacy T0 rms / max | legacy g*Pi rms / max | tau of max | CLASS trigger T0 rms / max | CLASS trigger g*Pi rms / max |
|---|---|---|---|---|---|
| 0.0992 | 0.063 / 0.166 | 0.060 / 0.191 | 281-288 | 0.063 / 0.165 | 0.060 / 0.191 |
| 0.1203 | 0.081 / 0.194 | 0.136 / 0.477 | 306-323 | 0.065 / 0.170 | 0.053 / 0.190 |
| 0.1516 | 0.83 / 4.19 | 10.0 / 61.8 | 297.5 | 0.072 / 0.274 | 0.098 / 0.317 |
| 0.1768 | 1.44 / 8.11 | 7.75 / 64.3 | 296-299 | 0.102 / 0.392 | 0.139 / 0.421 |
| 0.1985 | 1.02 / 6.84 | 4.80 / 44.4 | 296.5 | 0.134 / 0.539 | 0.189 / 0.616 |
| 0.2500 | 0.36 / 1.91 | 2.46 / 18.3 | 296-301 | 0.197 / 0.759 | 0.175 / 0.646 |
| 0.3031 | 0.38 / 1.20 | 0.82 / 5.07 | 295-299 | 0.283 / 0.926 | 0.222 / 0.942 |

With the legacy trigger the maximum differences all sit at tau = 296-300 Mpc. The
tau profile at k = 0.1516/Mpc: g*Pi is -5.27e-4 (CLASS) vs -5.26e-4 (clax) at tau =
292.2, then -4.11e-4 vs -3.7e-25 at 302.9, -6.19e-5 vs -2.9e-62 at 313.6, +3.77e-5 vs
-2.4e-99 at 324.5; CLASS's own RSA zeroes it only between tau = 350 and 366 Mpc,
where it is below 1% of the peak. The T0 source after the switch: +1.29e-3 vs
+9.94e-4 at 302.9 (-2.7% of peak), +5.62e-4 vs +3.06e-4 at 313.6. With the CLASS
trigger the same profile agrees to 0.06% of peak at tau = 302.9 and 0.03% at 313.6,
and the residual differences are the smooth amplitude excess at the peak quoted
above (helium-era damping). The late-time (tau > 1000 Mpc) levels agree in both
codes with either trigger: the streaming solution itself is right, only the time it
was imposed was wrong.

### The trigger

`clax/perturbations.py`, three sites (Einstein substitution in `_perturbation_rhs`,
the relaxation term in `photon_hierarchy_step`, and `_extract_sources`):

```python
tau_k = tau * k
kd_over_aH = kappa_dot / jnp.maximum(a_prime_over_a, 1e-30)
is_rsa = (tau_k > 45.0) & (kd_over_aH < 5.0)
```

CLASS (`thermodynamics.c`: `tau_free_streaming`; `perturbations.c`,
`perturbations_approximations`): rsa_on when `tau/tau_k > 45` and `tau >
tau_free_streaming`, the first time at which `1/(kappa' tau) >
radiation_streaming_trigger_tau_c_over_tau = 5`. From CLASS's table at the fiducial
cosmology (`diag_thermo_vs_class.py`):

| criterion | first true | z | g/g_peak | visibility integral after | k with 45/k earlier |
|---|---|---|---|---|---|
| CLASS kappa' tau < 0.2 | tau = 364.8 Mpc | 760 | 0.024 | 0.084 | k > 0.123/Mpc |
| legacy kappa'/(aH) < 5 | tau = 296.2 Mpc | 1013 | 0.689 | 0.357 | k > 0.152/Mpc |

The diagnostic knob on this branch (`_RSA_TRIGGER = "legacy" | "class"`, default
unchanged) keeps the first condition and the 45, and replaces the second by
`kappa_dot * tau < 0.2` (hard) and `sigmoid(2 (tau_c/tau - 5))` (relaxation term).
This also explains why nothing in the sweep moved the residual: the hierarchy size,
the tolerances, the tau and k grids and the Bessel path all act before or after the
switch, and l_max 50 and 12/10/17 are zeroed by the same mask.

### Perturbation k density, fine k grid, tau grid, Bessel path (GPU, jobs 24610 and 24620)

- Fast path vs exact-Bessel path at l = 1650, 1800, 1950, 2100, 2150, 2300 (same
  sources, n_k_fine 20000): -0.005 to -0.001% TT, -0.003 to -0.001% EE.
- n_k_fine 40000 vs 20000 (tau grid thinned to every other point so that 40000 fits
  the V100): 0.000% at all l. Thinned (2500) vs full (5000) tau grid: 0.038% /
  0.059% TT and 0.047% / 0.245% EE (max, 1000-2000 / 2000-2500).
- Perturbation k density, C_l band rms vs CLASS (TT 1000-2000 / 2000-2500, EE
  1000-2000 / 2000-2500) and perturbation stage cost on the V100:

| k per decade | modes | legacy trigger | CLASS trigger | pt (s) |
|---|---|---|---|---|
| 60 | 300 | 1.73 / 2.04, 2.96 / 8.19 | 0.38 / 0.49, 0.39 / 0.52 | 173 |
| 120 | 600 | 1.61 / 1.74, 2.88 / 8.09 | 0.50 / 0.89, 0.46 / 0.82 | 229 |
| 240 | 1200 | 1.60 / 1.72, 2.86 / 8.14 | 0.50 / 0.91, 0.46 / 0.84 | 336 |

  120/60 - 1: TT +0.08% (l = 1500), +0.24-0.31% (2000), +0.60% (2500); EE +0.06,
  +0.17, +0.48% (rms 0.17-0.18% over 1500-2000, 0.32-0.44% over 2000-2500), the same
  with either trigger; 240/120 - 1: TT <= 0.04%, EE rms 0.14% (max 0.49% at an EE
  trough) over 2000-2500. The sparse grid biases C_l low, so under the legacy trigger
  it partly masked the damping-scale excess; with the CLASS trigger the denser grid
  exposes it (table above).

## Recommended fix

1. Replace the second RSA criterion by CLASS's: compute `tau_free_streaming` once in
   the thermodynamics module exactly as CLASS does (first tau at which
   1/(kappa' tau) > 5) and switch on `tau > tau_free_streaming` (one scalar; the
   relaxation term can keep a sigmoid in tau around it), or minimally use
   `kappa_dot * tau < 0.2` at the three sites as the knob does. Expected gain: TT and
   EE rms over l = 1000-2500 from 1.7-8% to 0.4-0.5% at 60/decade (0.5-0.9% at
   120/decade, where the bias of item 3 no longer hides item 2); the planck_cl
   accuracy bar (0.2%) is then set by item 2. Cost: none measurable (174 vs 173 s
   perturbation stage; the hierarchy relaxes 70 Mpc later). Add a regression test on the source
   functions against CLASS `get_sources()` at k = 0.15-0.3/Mpc (the C_l tests at
   l <= 1000 were blind to this) and keep the C_l test to l = 2500.
2. Helium recombination: implement CLASS's Heswitch = 6 corrections in
   `_recfast_dxHe_dlna` so that x_e matches at z = 1300-3500 (today 1.3-3.5% off);
   expected gain: the remaining +0.3 to +0.5% at l = 1000-2500 (and ~+0.2% at
   l = 500-1000); cost: thermodynamics only.
3. Perturbation k density: 120 per decade (600 modes) for l > 1500; it removes a
   0.2-0.6% low bias at l = 2000-2500 for +56 s (229 vs 173 s on the V100); 240 adds
   nothing (<= 0.04% TT). Alternatively a CLASS-like linear step of 20 points per
   acoustic period above k ~ 0.07/Mpc (CLASS's 511 modes) at the same cost as 60/decade.
4. Nothing is needed for the reference, the l interpolation, the tau grid, the C_l
   k grid or the Bessel path.

Expected end state after 1-3: TT and EE within ~0.2% of CLASS to l = 2500 (the
reference itself is good to 0.1% TT / 0.4% EE at single l), at 229 + 72 + 20 s per
call on the V100 with the sweep's G6 settings.

## Commands

- Worktree: `git -C /home/n2minh/clax worktree add /lustre/work/n2minh/clax-wt-highl -b exp/highl-investigation exp/precision-sweep`
- CLASS: `OMP_NUM_THREADS=4 /home/n2minh/micromamba/envs/carpile/bin/python diags/diag_class_massless_ref.py --out /lustre/work/n2minh/std/clax/highl/class_massless.npz --configs ncdm0,ur3046,ur3044` and `--out .../class_massless_hp.npz --configs hpk_ncdm0,hpt_ncdm0,hpp_ncdm0,hpkt_ncdm0,hp_ncdm0,hpk2_ncdm0,hpk3_ncdm0` (3 s per default run, 133 s for the full high-precision run)
- CAMB: `/home/n2minh/micromamba/envs/cosmodesi/bin/python diags/diag_camb_massless_ref.py --out /lustre/work/n2minh/std/clax/highl/camb_massless.npz`
- Thermo: `JAX_PLATFORMS=cpu python diags/diag_thermo_vs_class.py --class-npz /lustre/work/n2minh/std/clax/highl/class_massless.npz --label ncdm0` (8 s)
- Sources: `python diags/diag_sources_vs_class.py --clax /lustre/work/n2minh/std/clax/highl/clax_sources_k60{,_rsaclass}.npz --class-npz /lustre/work/n2minh/std/clax/highl/class_massless.npz`
- GPU job 1 (legacy trigger, job 24610): `SCRIPT=diags/diag_highl_kdensity.py ARGS="--kpd 60,240,120 --exact-ls 1650,1800,1950,2100,2150,2300 --dump-dir /lustre/work/n2minh/std/clax/highl --thin-test" TAG=kdensity sbatch --export=ALL --job-name=clax-highl-kdensity slurm/highl-v100.sbatch`
- GPU job 2 (CLASS trigger, job 24620): same with `ARGS="--kpd 60,120,240 --rsa-trigger class --tag _rsaclass --exact-ls 1650,1800,1950,2100,2150,2300 --dump-dir /lustre/work/n2minh/std/clax/highl"`
- Tables: `python diags/diag_highl_report.py /lustre/work/n2minh/std/clax/highl/results-*.json`

## Not tested

- The x_e oracle injection (`diag_class_xe_oracle.py`): unnecessary once the
  thermodynamics tables were compared directly (0.01-0.03% at z = 800-1150) and the
  remaining offset was reproduced from k_D.
- The massive-neutrino set and other cosmologies: the trigger is a kappa' tau
  threshold, so the switch time scales with recombination in the same way for all of
  them; the `--rsa-trigger class` knob with the sweep's P1m config would confirm.
- Lensed spectra: not in scope; the unlensed fix propagates.
