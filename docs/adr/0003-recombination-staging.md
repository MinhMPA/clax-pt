# ADR 0003: Recombination follows CLASS's Saha -> ODE staging, advanced by an L-stable step

Date: 2026-10-11
Status: Accepted
Issue: https://github.com/smsharma/clax/issues/48 (related: #30, #45, #46; ADR 0001)

## Context

Any gradient taken through clax's thermodynamics carried a random error of
per-mille to order unity, with respect to h, omega_b, omega_cdm, Y_He, N_eff or
T_cmb, while the values (x_e, the visibility function, C_l) looked fine. On
unmodified upstream (4d084b5) the symptoms were:

- forward and reverse AD disagreed by up to **113%** on the production gradient
  path (`solve_background_and_thermo`, the ADR 0001 entry point);
- CPU and GPU derivatives differed by a factor of **21**;
- df/dh jumped by **6%** along a smooth line in h, with
  f = sum_k delta_m(k, tau_0)^2;
- the same derivative changed by **17%** when the perturbation solver grouped
  k-modes differently;
- on `fit_cl` (3000-point thermodynamics grid) the **values** were wrong too:
  x_e dropped to **0.035 at z ~ 1559**, where it is ~1.0.

**Cause.** clax integrated the stiff HeII -> HeI and hydrogen Peebles equations
with explicit steps clipped to [0, 1], through exactly the redshift ranges where
CLASS does not integrate at all (it uses Saha equilibrium there).

- *Helium, z ~ 3000-3500.* Heun with 10 sub-steps, clipped, plus a hard switch
  `x_He > 0.999 -> C_He = 1` that has no CLASS counterpart. x_e wiggles by only
  0.35% (1.0772 ... 1.0811 against the smooth 1.0812), but d ln x_e/dh spikes to
  ~5e12 at about ten isolated grid points, each ~4e-4 wide in ln a (0.1-0.3 Mpc
  in conformal time), against smooth values of 0.01-0.15.
- *Hydrogen, z ~ 1550-1750.* A clipped explicit RK4, sigmoid-blended with Saha at
  z = 1600. At z = 1600 the step has h*J = -3.2, outside RK4's stability
  interval [-2.79, 0], so x_e zigzags (by 18% at z ~ 1613 on the default
  20000-point grid) and on the coarse `fit_cl` grid it collapses.
- *How the spikes reach observables.* The baryon sound speed enters the baryon
  velocity equation through c_s^2 k^2 delta_b, so each spike in the derivative is
  a near-delta-function source in the tangent equation. Whether one of the
  adaptive solver's stage evaluations lands on a 0.2 Mpc spike is decided by step
  placement: how many k-modes share a step (8 on CPU, 62 on GPU), the solver, even
  jit against no-jit. The step-size controller watches only the primal, so it
  never sees the kick. This is why the error looks random, and why Kvaerno5 and
  Rodas5 both show it.
- *Reverse mode inherits the same spikes.* ADR 0001 removes the arithmetic
  cancellation in the reverse pass but not the spike sampling: upstream the
  production path gave -5.75e9 against a forward-mode -2.70e9 (113% off) on CPU.

**What it costs inference.** NUTS loses efficiency (energy errors, smaller steps,
divergences) but not correctness, because accept/reject uses values; the
exception is the `fit_cl` x_e collapse, which is a value error. Fisher forecasts,
MAP/Laplace fits and derivative-trained emulators are biased.

**Probe.** Six throwaway variants of `clax/thermodynamics.py` (V0 = upstream,
VA-VE), run on 12 cosmologies against CLASS RECFAST with `recfast_Heswitch = 1`
(the singlet-only helium physics clax implements). Fiducial cosmology unless
noted. "Spike ratio" is the maximum over grid points of |d ln Q/d theta| divided
by its median over +-25 neighbours, for Q in {x_e, c_s^2, T_b} and theta in
{h, omega_b, omega_cdm, Y_He, N_eff}, z 1500-6000. Source: the probe's results
table `final_table.txt` (cluster scratch, not in the repo:
`/lustre/work/n2minh/clax-probes/helium/out/`), jobs 22933-22937, 22948, 22949,
22952, 22953, 22959, 22972, 22973, 22976, 22977, 23000, 23034, 23055, 23060.

| variant | what changes | x_e vs CLASS, z 2000-5000 | x_e vs CLASS, z 800-1600 | spike ratio | k-chunk spread of df/dh (worst of 3 cosmologies) | CPU vs GPU tangents (normwise) | verdict |
|---|---|---|---|---|---|---|---|
| V0 | unmodified upstream | 6.3e-3 | 0.15 | 9.5e14 | 0.17 | 21 | fails |
| VA | L-stable step for helium, no staging | 1.3e3 (x_e overshoots to -988) | 0.15 | 6.9e3 | not measured | 3.1e-9 | fails outright |
| VB | CLASS helium staging, existing clipped Heun below it | 6.4e-4 | 0.15 | 2.3e4 | 5.7e-4 | 4.8e-10 | helium fixed, hydrogen still chatters |
| VC | VB staging + L-stable step for helium | 6.4e-4 | 0.15 | 6.9e3 | 1.3e-4 | 4.8e-10 | helium fixed, hydrogen still chatters |
| VD | VC + CLASS hydrogen staging at z = 1600, clipped RK4 below | 6.4e-4 | 4.4e-3 | 10.9 | 1.1e-4 | not measured | misses the x_e bar and the spike bar (10) |
| **VE** | VD with RK4 replaced by the same L-stable step | **6.4e-4** | **3.5e-4** | **1.19** | **6.3e-5** | **2.7e-13** | **meets every criterion** |

Bars used in this repository's tests (spec seams S1-S3: the thermodynamics tables
and their tangents, the delta_m gradient through the matter solve, and the
production gradient entry of ADR 0001): x_e <= 1e-3, spike ratio < 10, spread
< 3e-4, CPU vs GPU < 1e-10. "Chatters" in the verdict column means the explicit
step zigzags from one grid point to the next, which puts spikes in the
derivative.

## Decision

Recombination follows CLASS's approximation staging, advanced by one L-stable
linearly implicit step, with **no clip anywhere in the recombination step**.
This is probe variant VE, with one deliberate difference added afterwards (D4
below). Before the helium-ratio change (D4), the in-repo tables agreed with the
stored VE tables to 2.05e-12 in x_e at every grid knot (job 24182). The
CLASS-ratio change then moves x_e by design: VE kept the helium-to-hydrogen
ratio at exactly 4, which puts a 1.2e-3 offset in the fully ionised x_e and
5.4e-4 in z 2870-5000, so the shipped tables no longer equal VE's. The review of
PR #53 (Oct 16, 2026) found two defects that VE shared, now fixed and noted under
D2 and D3 (the HeI Saha root ignoring HeIII; the hydrogen Saha half-step lag), and
questioned the fixed-x_e Jacobian, which measurement kept (D1).

**D1. One shared stiff step, `_ros2_scalar_step` (internal).** The two-stage,
L-stable, second-order linearly implicit Rosenbrock step ROS2
(Verwer et al. 1999), gamma = 1 + 1/sqrt(2), for dx/d ln a = f(x):

```
W  = 1 - gamma h J
k1 = h f(s_j, x_j) / W
k2 = [h f(s_{j+1}, x_j + k1) - 2 k1] / W
x_{j+1} = x_j + (3/2) k1 + (1/2) k2
```

J = df/dx comes from a scalar forward-mode `jax.jvp` with the environment held
fixed (n_H, H, T_b, T_r, z, and x_e as an input); ROS2 stays second order for any
J. 10 sub-steps per grid step; ln a is linear within a step and ln H, T_b are
linear in ln a. "L-stable" means a stiff relaxation (rate times step >> 1) lands
on its equilibrium without overshoot or oscillation, which is what removes the
need for a clip; that holds for the exact Jacobian, in which x_e moves with the
species fraction. Holding x_e fixed would halve J for a rate dominated by
recombination (proportional to x_e x_p) and turn the stiff limit into a damped
alternating approach (amplification -0.66 per sub-step). Measured (review of
PR #53, Oct 16, 2026), that is not this system: wherever the step is stiff
(|h J| > 0.5), photoionisation dominates the hydrogen rate and the fixed-x_e J is
0.988-0.995 of the exact one, and the largest |h J| reached is 2.35 (hydrogen at
z ~ 1604, 3000-point grid, worst of 10 cosmologies; 0.31 on 20000 points; 4.0 on
the 1500-point test grid; helium <= 1.1). ROS2's one-step factor for a decaying
mode then stays within 0.006 of the exact-J one (+0.261 against +0.266 at
h J = -2.35), positive, so no alternation. The exact J was tried and is the less
accurate choice at these step sizes: against the code's own 100000-point
solution, x_e over z 800-1650 is 2.3e-4 with it against 1.3e-4 with the
fixed-x_e J on the 3000-point grid (visibility peak 0.031 against 0.025 in z;
20000 points 1.2e-5 against 1.0e-5), because ROS2's error constant depends on J
where the step is not stiff. The fixed-x_e J is kept.

**D2. Helium, HeII -> HeI.** CLASS's staging: Saha equilibrium above z = 2870
(`thermodynamics.c:4053-4067`, HeI ionisation temperature 285,335 K,
`wrap_recfast.c:78`), taken in the cancellation-free root of the quadratic. The
ODE state equals the Saha value while z > 2870 (`thermodynamics.c:2059-2065`);
below, the RECFAST helium Peebles equation (`wrap_recfast.c:346-349`) advances it
with D1. The output blends the two with CLASS's smoothstep
f2(s) = 3 s^2 - 2 s^3, s = clip((2870 - z)/100, 0, 1) (`thermodynamics.h:49`,
applied at `thermodynamics.c:2973-2987`, Delta z = 50 at `precisions.h:202`),
so the weight is exactly 0 or 1 outside z in [2770, 2870]. The hard
`x_He > 0.999` switch is deleted (C_He is continuous as x_He -> 1, measured); the
`x_He < 1e-15` guard stays; the Sobolev escape probability uses
`-expm1(-tau)/tau` with a series below tau = 1e-4. The HeIII -> HeII Saha blend
at z ~ 5000 is unchanged (measured smooth). The HeI Saha root assumes no HeIII,
but below z ~ 5000 the MB95 HeIII fraction is still nonzero, so the root is
multiplied by (1 - x_HeIII): HeI Saha splits only the helium not in HeIII.
Without that share (the first version of this PR) x_HeII + x_HeIII reached
1.0007 near z = 4970 on both grids, and x_e sat 4.3e-5 to 6.6e-5 off CLASS over
z 2870-5000 at all 10 cosmologies; with it the sum stays <= 1 and that window
is 1.4e-6.

**D3. Hydrogen.** The same pattern at z = 1600: Saha with the x_HeII correction
above (`thermodynamics.c:4070-4087`), initial condition at `thermodynamics.c:2093`,
the Peebles equation (`wrap_recfast.c:110-174`) advanced with D1 below, f2 blend
over z = 1600 -> 1500 (`precisions.h:204-205`) replacing the sigmoid. RECFAST's
C = 1 for x_H >= 0.995 stays (it is in CLASS). The hydrogen Saha value is
evaluated at the step end (T_b and n_H at the new grid point), like the regime
switch, the blend weight and the helium Saha; the first version of this PR took
it at the step midpoint, so the Peebles initial condition at z = 1600 and the
Saha half of the blend lagged half a grid step.

**Cost.** Below its switch each species needs the ROS2 advance; above it the
advance would be discarded by the regime switch, so `jax.lax.cond` skips it
(identical output). The grid is uniform in ln tau, so the index of z = 1600 or
2870 moves with the cosmology and the scan cannot be split at a fixed index;
under `vmap` with a batched predicate both branches run, as before. The 11
sub-step boundary environments are built once per grid step and shared by both
species, and one `_ionHe` solve supplies both helium fractions.

**D4. One helium-to-hydrogen number ratio.**
`_f_He(Y_He) = Y_He / (3.9715 (1 - Y_He))`, CLASS's `_not4_`
(`thermodynamics.h:707`, `pth->fHe` at `thermodynamics.c:352`), at every
number-ratio site: the HeIII Saha solver, the initial x_e, the x_HeII term in
hydrogen Saha, the helium rate's f_He, and every x_e sum. Before this, one site
(`_reionization_xe`) used 3.9715 and the rest used exactly 4, so clax was
internally inconsistent; the difference is a 5.4e-4 offset in x_e at
z = 2870-5000 and 1.2e-3 when fully ionised (fiducial).

**What D4 does not cover.** The four `1 - 0.75 Y_He` mean-molecular-weight sites
(the c_s^2 prefactor at three places and the Compton-coupling fraction in the
T_b update at one) keep exactly 4. They belong to the out-of-scope
c_s^2-versus-CLASS-c_b^2 follow-up, where CLASS's form `1 + (1/_not4_ - 1) Y`
(`thermodynamics.c:128, 3003, 3007`) should be applied together with the c_b^2
comparison. Until then c_s^2 sits 2.6e-4 (x_e = 1.16) to 5.4e-4 (x_e = 0) from
CLASS's form in relative terms. The subject line of the D4 commit says
"everywhere"; it means the number-ratio sites only.

The stage-switch redshifts (2870, 1600) are fixed numbers, independent of the
parameters, as in CLASS, so no derivative passes through a switch location.

**Unchanged:** the background; the kappa_dot / visibility construction including
its `stop_gradient` (see #45 under "Not done"); reionisation; the T_b update; the
`custom_vjp` entry `solve_background_and_thermo` and its forward-mode basis
(ADR 0001); the HeIII stage.

**Reference data.** CLASS RECFAST with `recfast_Heswitch = 1` at 10 cosmologies
(the 5 of `COSMOLOGY_GRID_LCDM`, Y_He = 0.22 / 0.28, N_eff = 2 / 4 via N_ur, and
m_nu = 0.15 eV; Y_He passed explicitly, never via BBN), x_e and T_b on 4000
points uniform in ln(1+z) over z in [200, 8000], committed under
`reference_data/<name>/thermo_recfast_hesw1.{npz,json}` with their provenance.
They were generated from the CLASS-PT tree (CLASS v3.3.4 thermodynamics), since no
stock `class_public-3.3.4` source tree exists on the development machine. Stock
CLASS 3.3.4 (the PyPI `classy` wheel) reproduces all 10 to <= 7.1e-6 in x_e
(largest at z 200-800; <= 6.7e-12 over z 2870-5000) and <= 7.7e-7 in T_b
(`scripts/crosscheck_recfast_hesw1_stock.py`), 140 times below the 1e-3 x_e
gates. `tests/test_recomb_reference.py::test_reference_grid_matches_test_grid`
fails if the generator's cosmology grid and `tests/conftest.py`'s drift apart.

## Alternatives considered

1. **Staging with the old clipped explicit steps** (VB, VD). Rejected: the clipped
   Heun and RK4 still chatter where they are used (spike ratio 2.3e4 with
   hydrogen untouched, 10.9 with hydrogen staged but still clipped RK4), and VD
   misses the x_e bar at z 800-1600 (4.4e-3).
2. **L-stable step without staging** (VA). Rejected: x_e overshoots to -988 and
   the visibility peak moves by 294 in z. The probe did not isolate why; staging
   removes the stiff Saha-dominated range from the integration altogether, which
   is what CLASS does.
3. **Fix the ODE solver instead** (the parked `fix/rodas5-division-free` branch,
   which hardens the Rodas5 stage matrix by writing it division-free, the "W'"
   form I - dt gamma J instead of I/(dt gamma) - J).
   Rejected as the fix for #48: the diagnosis (jobs 22768-22820) found Kvaerno5
   shows the same corruption (one mode, k = 0.0226 Mpc^-1, kicked to
   d ln delta_m/dh = +1.32 at tau = 114-131 Mpc). The solver only decides how
   much of the thermodynamics error reaches delta_m. That branch's step-budget
   change for `TestRosenbrockPk` may be offered separately.
4. **Port CLASS's default helium physics** (`Heswitch = 6`, Doppler and triplet
   terms). Out of scope: clax implements `Heswitch = 1`. Against CLASS's default,
   x_e differs by 1.5% at z = 2000 and 4.2% at z ~ 1771, with the visibility
   peak within 0.04 in z. This is a model difference, filed separately.

## Consequences

**Values.** x_e agrees with CLASS RECFAST (`Heswitch = 1`) to within the 1e-3 gate
in every window, worst over the 10 cosmologies (default 20000-point grid /
`fit_cl` 3000-point grid; CPU, after the review fixes): z 2870-5000 1.4e-6 /
1.4e-6; z 1650-2870 8.0e-7 / 1.3e-6; z 1590-1650 1.0e-6 / 2.5e-6; z 1500-1600
5.6e-6 / 5.6e-5; z 800-1550 2.0e-5 / 1.1e-4. Above z = 5000 the MB95 HeIII stage,
which this change leaves alone, sits 2.0e-4 to 2.4e-4 from CLASS. The `fit_cl`
x_e collapse at z ~ 1559 is gone. The visibility peak
sits within 1.8e-3 (20000 points) and 2.3e-2 (3000 points) in z of CLASS (bars
0.01 / 0.03); the shifts of the sound horizon r_s and the acoustic scale theta_s
were not computed, so they are untested. The first version of this ADR read the
3000-point hydrogen residual (z 1590-1650: 4.2e-4; 800-1550: 5.6e-4) and its
+4.1e-4 gap to the 20000-point grid as discretisation error. The review of
PR #53 traced it to the hydrogen Saha value lagging half a grid step (D3); with
that fixed, z 1590-1650 is 2.5e-6 on the 3000-point grid.

**Derivatives** (after the review fixes, Oct 16, 2026, unless noted). The spike
ratio falls from 9.5e14 (20000 points) and 5.4e8 (3000 points) to <= 2.53;
tangents match converged finite differences to 1.1e-4 (helium window) and 3.9e-4
(hydrogen window) against a 1e-3 bar (CPU, job 26314); CPU and GPU tangents
agree to 2.2e-13 of the largest tangent entry (V100, job 26355; unfixed 1.0 /
17.2 / 4.1e-7 at three cosmologies, job 24275). Downstream, for
f = sum_k delta_m(k, tau_0)^2 through the full matter solve (`fast_cl`, Rodas5,
rtol = atol = 1e-4) at fiducial, omega_b + 20% and Y_He = 0.28, on a V100 (job
26274): df/dh no longer depends on how k-modes are grouped (spread <= 9.6e-5
against 3e-4; unfixed 0.19 at fiducial on CPU), and native reverse mode, the
production reverse path of ADR 0001 and forward mode agree to <= 1.2e-10
(unfixed 1.16 on the production path at fiducial). Smoothness of df/dh along
h = 0.62-0.72 (cubic-fit residual, bar 3e-4; unfixed 5.5e-2): on CPU 5.5e-5 at
fiducial (job 26403). On a V100 the test fails at fiducial (3.22e-4) and at
Y_He = 0.28 (3.55e-4), passing at omega_b + 20% (3.8e-5). The fiducial failure is
identical on the first version of this PR (3.224e-4, job 26402), whose CPU value
was 7.8e-6 and V100 values 6.0e-6 / 2.84e-4 at omega_b + 20% / Y_He = 0.28 (jobs
26401, 26414). The residual is not a kink, and it does not come from
thermodynamics: with omega_b and omega_cdm fixed, the thermodynamics tables carry
<= 1e-11 of df/dh (V100, job 26548). It comes from the background tables, splines
on `bg_n_points` = 800 points in ln a. h moves ln a(tau) at late times, so the
times at which the matter solve crosses the splines' knots move with h, and the
solver's long steps near today integrate that part of the tangent with an error
that jumps whenever the step pattern changes. On fine h scans (41 points over
0.665-0.675) df/dh is smooth between changes of the step count of the k batch that
dominates f, and jumps by 2-4e-5 at each change (jobs 26558 CPU, 26560 V100). The
part of df/dh carried by H and the densities at fixed ln a is smooth to 3.6e-6;
the part carried by tau_0 and ln a(tau) is not (job 26548). Forcing 60-80 extra
steps over z = 0.01-0.3 or 0.05-1 removes the roughness (5.5e-6 to 5.9e-6), over
z = 0.3-1 it does not (2.1e-4; jobs 26568-26575). The residual does fall with the
matter-solve tolerance, but only slowly between 1e-4 and 1e-5: V100 fiducial
3.2e-4, 1.4e-4, 6.1e-5, 6.9e-6 and 6.1e-6 at rtol = atol = 1e-4, 3e-5, 1e-5, 1e-6
and 1e-7 (CPU 6.5e-6 at 1e-6; jobs 26535-26537, 26553-26556). An earlier revision
of this test ran at 3e-5 and read that slow fall as a tolerance-independent floor;
there is none. About 6e-6 is the cubic fit's own error: the residual is then the
smooth shape of a missing quartic term, and a quartic fit leaves 1.1e-6. The test
now tabulates the background on 8000 points and keeps rtol = atol = 1e-4:
6.1e-6 / 5.5e-6 / 5.7e-6 on the V100 and 5.6e-6 / 5.5e-6 / 5.5e-6 on CPU (fiducial /
omega_b + 20% / Y_He = 0.28; jobs 26586, 26587), for +17% per evaluation (29.3 s
against 24.8 s on the V100, job 26578). The unfixed code still fails it: 1.4e2 at
fiducial on CPU, 5.0e-4 / 2.0 / 3.9e-4 on the V100 (jobs 26590, 26588). The
800-point noise is pre-existing and reaches any
production derivative with respect to a parameter that moves the late-time
expansion at fixed omega; it is outside this PR. The forward df/dh at fiducial itself
moves by 3.5% between the unfixed (-2.659e9) and the fixed (-2.756e9) code, and
the unfixed value also changed with step placement. Finite differences of
P(k) also become trustworthy: the spread of a central-difference
dP(k = 0.1 Mpc^-1)/d omega_cdm over steps 1e-4 to 2e-3 falls from 1.06% to 0.02%
(CPU jobs 24515 -> 24514; GPU 1.02% -> 0.02%, jobs 24517 -> 24516).

**Speed** (after the review fixes). On one V100 (job 26226, four processes in
the order unfixed / fixed / unfixed / fixed) the whole `fit_cl` chain takes
8.19 s and 7.75 s unfixed against 8.04 s and 8.04 s fixed: no change within the
unfixed code's own 5% scatter. The first version of this PR measured 0.917 in the
same benchmark (job 24327); the perturbation stage is 3.41-3.49 s with the fix
against 3.47-3.86 s without (overlapping), and `fit_cl`'s thermodynamics alone is now
12-18% slower than unfixed there (1967-2010 ms against 1665-1797 ms). Measured
separately (thermodynamics only, best of 5-7 after compile, primal / jvp in d/dh;
jobs 26220, 26244 and CPU): unfixed 0.63 / 1.79 s on CPU at 20000 points, 2.09 /
2.27 s on a V100 at 3000, 11.0 / 11.6 s at 20000, 47.9 / 54.6 s at 100000; fixed
0.20 / 2.19 s, 1.89 / 2.55 s, 9.8 / 13.6 s and 46.4 / 65.2 s. The first version of
this PR had the jvp at 3.94 s (CPU) and 2.96 / 16.4 / 79.8 s (V100); the
`lax.cond` skipping of discarded advances recovers 10-16% of the jvp on the GPU
and about 40% on CPU. The jvp stays 1.12-1.22x the unfixed code's. Unrolling the 10
sub-steps (`fori_loop(..., unroll=True)`, identical output) is fastest on the GPU
(1.76 / 2.50 s at 3000 points) but makes the CPU jvp 4x slower, so it is not used.
The 8 s total is not the 34 s quoted in CLAUDE.md; that figure was not reproduced
here.

**C_l.** These C_l numbers were measured on the first version of this PR. The
review fixes move x_e relative to it by at most 5.4e-5 on `planck_cl`'s
100000-point grid (at z ~ 4970), 7.1e-5 on the 20000-point grid and 4.6e-4 on
`fit_cl`'s 3000-point grid (both at z ~ 1560), against changes of up to 0.66
from the fix itself; C_l was not re-run. Lead with the direct effect of the fix.
At `planck_cl` the unlensed
spectra move by at most (TT / EE) 0.0051% / 0.0037% for l <= 500, 0.0188% /
0.0255% for 500 < l <= 1500 and 0.0518% / 0.0627% for 1500 < l <= 2500,
identical for the table-Bessel and the per-l harmonic path (job pairs
24330/24331 and 24360/24361). The maximum error against CLASS in any TT or EE
band does not move by more than 0.04 percentage points.

At `fit_cl` the shifts are larger (TT up to 10.3% at l = 4, 1.7% at l = 30,
0.46% over 100-500; jobs 24328/24329). The direction of the error against CLASS
(max |C_l/C_CLASS - 1|, unfixed -> fixed) differs by band: TT l 2-30
84% -> 94% and 30-100 4.5% -> 4.7% (worse), TT 100-500 1.85% -> 1.55% and
500-1500 15.8% -> 14.9% (better); EE 2-30 16.8% -> 16.8% (unchanged), EE 30-100
3.6% -> 0.8%, 100-500 1.30% -> 1.11% and 500-1500 19.5% -> 17.7% (better). So five
bands improve, two (low-l TT) worsen and one does not move. The unfixed `fit_cl`
grid had a real value error: x_e off CLASS by 0.46 at fiducial in z 1590-1650 and
0.54 in z 800-1550 (worst over the 10 cosmologies 0.66 and 0.78; jobs
24169-24171), against 4.2e-4 and 5.6e-4 fixed. **The cause of the `fit_cl` C_l
shift is unresolved.** Two candidates, neither tested: (i) the genuine x_e
correction at z ~ 1560-1650 on the 3000-point grid; (ii) noise at `fit_cl`'s
20 k/decade grid and rtol = 1e-3. The `planck_cl` shift of 3e-6 at l <= 30 does
not discriminate between them. A test would be to feed the fixed and unfixed
tables through `fit_cl`'s perturbation solve with a finer k grid and a tighter
tolerance.

Absolute errors against CLASS are a **pre-existing baseline at the preset
defaults, not an effect of this change**. `planck_cl` with its default
`hr_n_k_fine = 5000` is k-unconverged above l ~ 500: TT errors against CLASS
RECFAST of 1.1% (l <= 500), 3.3% (500-1500) and 9.3% (1500-2500) on the
table-Bessel path, 0.6% / 2.9% / 9.0% on the per-l path, identically on the
unfixed and the fixed code. The unfixed run gives TT = -0.78, -0.92, -3.2, +0.4%
at l = 500, 1000, 1500, 2000, in line with the -0.78, -0.91, -3.21, +0.47%
recorded for the closest configuration (log grid, 60 k/decade, 300 modes, spline
interpolation) in the Sep 2, 2026 Chebyshev k-sampling entry of `CHANGELOG.md`.
This ADR does not claim that any existing test gates clax's own C_l at the
sub-percent level (`tests/test_harmonic.py` bars are 30-60% at `fast_cl`;
`tests/test_lensing.py` lenses CLASS's own spectra).

**Tests.** The gates are in `tests/test_recomb_step.py` (S0, the stiff step on a
scalar ODE with a known answer), `tests/test_recomb_thermo.py` (S1, tables and
forward tangents at 10 cosmologies on both grids; a `--fast` subset runs the
3000-point grid at fiducial), `tests/test_recomb_downstream.py` (S2 and S3, the
delta_m functional and the production gradient entry),
`tests/test_recomb_gpu.py` (CPU vs GPU; run it with `JAX_PLATFORMS=cuda,cpu`, it
skips when either backend is missing) and
`tests/test_recomb_reference.py` (the reference data). Each new test family was
shown failing on the unfixed code before the fix and passing after, with two
kinds of exception. The red failures of S0 and of the reference-data tests were
an ImportError (the step function does not exist on 4d084b5) and a missing file,
not an assertion. The two bounds tests (x_e in [0, 1 + 2 f_He], and x_HeII <= 1
for z in [2400, 2870]) cannot be red: the old clipped code satisfied them by
construction. The
x_HeII <= 1 guard has a 3.6e-4 to 5.2e-4 margin, so it catches an overshoot in
x_HeII only above about 5e-3. Above z = 2870, `test_helium_counted_once_z2870_5000`
holds the electrons per helium nucleus, (x_e - 1)/f_He = x_HeII + 2 x_HeIII
(hydrogen is fully ionised there), to CLASS over z 2870-5000 with a 1e-4 bar;
each code divides by its own f_He, so the test does not depend on the
helium-ratio convention. It fails on the first version of this PR at all 10
cosmologies on both grids (5.7e-4 to 7.5e-4, the double-counted HeIII) and passes
after the fix (<= 1.5e-5).

**D4 guards.** The fully ionised x_e test sees only the final x_e sum. A second
guard, `test_helium_ratio_guard_xe_windows`, holds x_e to CLASS RECFAST in five
z windows (5000-8000, 2870-5000, 2870-3500, 2000-2400, 1650-2870) with bars of
2 x the worst error over the 10 cosmologies, rounded up to 2 significant figures:
4.8e-4, 2.8e-6, 2.4e-8 / 4.5e-8, 3.7e-7 / 8.9e-7 and 1.7e-6 / 2.7e-6
(20000 / 3000 points; one value where the grids agree), 2 to 4e4 times tighter
than the 1e-3 gates. The bars were first derived on the PR head (jobs
26063-26072) and re-derived on Oct 16, 2026 after the review fixes lowered the
error at z 1650-5000. The reverted-site check was then repeated on the fixed code
(CPU, one scratch copy per site reverting exactly that site to exactly 4,
fiducial and Y_He = 0.28 on both grids, the five guard windows plus z 1500-1650 and
800-1550). Mutant error over its bar, best window: the guard fails on the HeII
Saha quadratic (11, z 2870-3500), the helium Peebles rate (210) and the helium
Peebles x_e sum (20, both z 2000-2400), and the final x_e sum (every window, 2.4
to 2.6e4). It does not see the HeIII Saha constant in `_ionHe` (0.53), the f_He
in the hydrogen Peebles x_e sum (0.68) or in the hydrogen Saha correction (0.49):
those reversions move x_e by less than the code's own error against CLASS there
(the HeIII constant shifts x_e by 1.9e-5 at z 5500-6500, against a 2.0e-4 offset
at fiducial), so no bar passes the current code with a 2x margin and fails them.
It cannot see the initial x_e of the scan (it sets the table only above z = 5e4)
or `_reionization_xe` (z <~ 8), which move x_e by up to 1.0e-3 there, because the
references stop at z in [200, 8000]. Those five sites stay unguarded.

One existing test changed: `test_pk_table_auto_batch_matches_full_vmap`
(`tests/test_end_to_end.py`) runs both solves at rtol = 1e-3 instead of 1e-2, bar
1e-3 unchanged. With smooth tables the shared-step Rosenbrock solver shows a
steady -1.09e-3 mismatch between auto-batched and full-vmap solves at
rtol = 1e-2 (about rtol/10), which scatter of 7.2e-4 to 2.06e-3 had hidden; at
rtol = 1e-3 it is 1.8e-4. This is a property of the shared step, not a loosening
of the bar.

**ADR 0001.** Its premise "forward mode is exact" is refined, not overturned; see
the dated note appended there.

**Maintenance.** Reverse mode through the stiff step needs a static Python
integer for the sub-step count (the loop lowers to a scan); the count is the
length of the shared boundary-environment array (11 points, 10 sub-steps), a
static shape. The ADR 0001 caveat (forward mode needs
`th_grad_mode="native"`) is unchanged.

## Not done

- **S4, a C_l-level gradient test at `fit_cl`, is parked pending a decision.**
  `compute_cls_all_fast` cannot be differentiated with respect to any parameter
  that moves the conformal age tau_0 (h, omega_b, ...), in forward or reverse
  mode: `clax/harmonic.py:836-837` converts tau_0 to a Python float and int to
  size the Bessel table. Its reverse mode would also need ~46 GB on a 32 GB V100
  (5.7 GiB per stacked per-l residual: 77 l-values x 5000 k x 2000 tau x 8 B).
  This is a pre-existing defect, unrelated to #48 (identical on 4d084b5): a
  C_l-based NUTS fit through `compute_cls_all_fast` cannot run today. S2 and S3
  already show the fix reaching the matter field. Options: a separate issue and
  PR for harmonic differentiability, where S4 lands; or a harmonic task on this
  branch, after which the C_l tables above must be re-run.
- **#45** (does the kappa_dot `stop_gradient` still need to stay?) is a separate
  follow-on: with recombination smooth, test removing it and keep the removal
  only if gradients stay finite and AD matches FD for omega_b and omega_cdm at
  3-5 cosmologies. Related, pre-existing, both newly measured:
  - AD sits a steady +0.66% above converged FD for dP(k = 0.1 Mpc^-1)/d omega_cdm
    on the fixed tree (`tests/test_pk_forward_mode.py`; jobs 24514 CPU and 24516
    GPU, FD std 0.007%, platforms agreeing to 2e-8). On the unfixed tree the same
    offset (+0.64% CPU, +0.56% GPU against the mean FD slope, which equals the
    fixed slope to 1.6e-5) was hidden because FD itself scattered by 1.06% across
    steps (std 0.44%; jobs 24515, 24517). The fix exposed the bias by smoothing
    the primal; it did not create it. In units of d ln P/d ln omega_cdm the bias
    is 0.0092, about 2.3 times the ~0.004 quoted in #45, whose measurement
    conditions were not recorded. The kappa_dot `stop_gradient` is a candidate
    cause, untested, and probably not the sole one; one cosmology, one k, one
    parameter, one precision block. The test's 1% bar passes with a 0.34-point
    margin.
  - `tests/test_pk_gradients.py::TestPkPublicTableGradients::test_public_table_partials_match_fd_full`
    fails on both trees (AD against FD for dP(k = 0.05 Mpc^-1)/d omega_b is off by
    4.02% fixed, 4.08% unfixed; jobs 24324, 24429); the `stop_gradient` is a
    candidate, untested. #45 as filed is about omega_cdm, so this awaits a
    decision whether to extend #45 or file a new issue.
- The out-of-scope items: `Heswitch = 6` helium physics; the hard Taylor switch in
  the T_b update at x_e < 1e-3 (spike ratio 20 at z ~ 624, on a 1500-point grid
  only); c_s^2 against CLASS's c_b^2 (1.3% apart at z ~ 632), together with the
  four `1 - 0.75 Y_He` sites of D4.
