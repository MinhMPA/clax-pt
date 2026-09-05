# CLASS-PT in-loop reference (companion to the clax-pt validation plans)

**Purpose:** the *oracle text* that Tracks A and B of the validation plans mirror.
Every formula below is a line-faithful transcription of CLASS-PT at
`/home/n2minh/CLASS-PT` commit `09d5531a4ec61187d84f506e9fdaf7fdcc8c7718`, followed by
the clax equivalent. When a plan step says "reference §N", it means this file.
When this file and the C source disagree, the C source wins — re-read the cited
lines and fix this file in the same commit.

**Verification rule for implementers and reviewers:** quote the C line number next
to every mirrored term in the clax code (CLAUDE.md "Trace every equation to CLASS
source code"). A reviewer rejects a mirror that cites no line.

Sections:
§1 AP ratios · §2 grid/window/extrapolation · §3 AP spline macros ·
§4 GL μ-loop, RSD block · §5 GL μ-loop, bias block · §6 LEGENDRE_PROJECT ·
§7 GL nodes · §8 dead accumulators · §9 Id2d2 conventions ·
§10 pm rows: `get_pk_mult` transforms, units, index map · §11 classy accessor
formulas · §12 `initialize_output`, `Pd2d2_0`, kh units · §13 patches ·
§14 quirks to mirror, not fix · §15 clax-side channel inventory.

---

## §1 AP ratios — `nonlinear_pt.c:1245-1296`

Computed once per `z_pk[i_z]` before any loop, for **every** run (AP=No writes 1.0):

```c
int Nz = 2000; double Omfid = pnlpt->OmfidAP; double kmsMpc = 3.33564095198145e-6;
double Da = 0, Dfid = 0, hnew = 0, hfid = 0, dz;
background_at_tau(pba, tau_req[i_z], long_info, inter_normal, &last_indexf, pvecbackf);
pnlpt->growthf[i_z] = pvecbackf[pba->index_bg_f];            // 1261 (unless replace_background)
if (pnlpt->AP_effect == AP_effect_yes) {
  if (pnlpt->z_pk[i_z] == 0.) { hratio_array[i_z] = 1.; Dratio_array[i_z] = 1.; }   // 1267-1269
  else {
    hfid = pow(Omfid*pow(1+z,3) + (1-Omfid) + pba->Omega0_g*pow(1+z,4), 0.5);        // 1272
    hnew = pvecbackf[pba->index_bg_H] / kmsMpc / 100.0 / pba->h;                      // 1276  == E(z)
    hratio_array[i_z] = hnew / hfid;                                                   // 1278
    dz = z / (double)(Nz - 1);                                                         // 1280
    for (j = 1; j < Nz; j++)                                                           // 1282-1284 trapezoid
      Dfid += dz * ( 1/sqrt(Omfid*pow(1+dz*j,3)     + (1-Omfid) + Omega0_g*pow(1+z,4))
                   + 1/sqrt(Omfid*pow(1+dz*(j-1),3) + (1-Omfid) + Omega0_g*pow(1+z,4)) ) / 2.;
    Da = pvecbackf[pba->index_bg_ang_distance] * kmsMpc * 100 * pba->h * (1+z);        // 1288  == D_M·H0
    Dratio_array[i_z] = Da / Dfid;                                                     // 1291
  }
} else { hratio_array[i_z] = 1.; Dratio_array[i_z] = 1.; }                            // 1293-1296
```

Quirk (mirror it): inside the `Dfid` integrand the radiation term is frozen at the
*target* redshift, `Omega0_g·(1+z_pk)^4`, not at the integration variable.
`Omega0_g` is photons only. `kmsMpc·100·h` is H0 in 1/Mpc, so `hnew = H/H0 = E(z)`
and `Da = D_M(z)·H0` (dimensionless, units of c/H0); both are h-independent.

clax mapping (`clax/ap.py::ap_ratios(bg, z, omfid)`):

| CLASS-PT | clax |
|---|---|
| `pvecbackf[index_bg_H]/(kmsMpc·100·h)` | `bg.H_of_loga.evaluate(loga_z) / bg.H0`, `loga_z = jnp.log(1/(1+z))` |
| `D_A·H0·(1+z)` (flat) | `(bg.conformal_age - bg.tau_of_loga.evaluate(loga_z)) * bg.H0` |
| `pba->Omega0_g` | `bg.Omega_g` |
| `Nz = 2000` trapezoid | `jnp.trapezoid` over `zz = jnp.linspace(0, z, 2000)` with the frozen radiation term (identical sum) |
| `z == 0` branch | `jnp.where(z == 0, 1.0, ...)`; z is a Python float in the campaign, so a Python `if` is also acceptable |

## §2 Grid, window, extrapolation

`nonlinear_pt.c:3185-3189`: `Nmax = ppr->nmax_nlpt` (=256), `kmin = 0.00005*pba->h`,
`kmax = 100.*pba->h` (1/Mpc). `kdisc` is log-spaced between them, i.e. exactly
`clax.ept.ept_kgrid()` in h/Mpc (`ept.py:1878-1889`).

`3255-3261`:
```c
if (pnlpt->AP_effect == AP_effect_yes) Nside = 10; else Nside = 1;
kmaxnew = kdisc[Nmax - 1 - Nside];  kminnew = kdisc[Nside];
```
All in-loop accumulations run `for (index_j = Nside; index_j < Nmax - Nside; index_j++)`.
Outside `[kminnew, kmaxnew]` the outputs are filled by `SPLINE_INTERP_OUTPUT`
(`2468-2497`) with per-channel extrapolation expressions (`4615-4626`, `4685-4700`), e.g.

* loop terms: `-exp(lnpk_l + 2*lnk_l) * sigmav * coef + large_for_logs_big` with
  `coef`: `0_vv = f²(441+566f+175f²)/1225`, `0_vd = 2f(625+558f+315f²)/1575`,
  `0_dd = (61−2f+35f²)/105`;
* ctr terms: `exp(lnpk_l + 2*lnk_l) * {1, 2f/3, 8f²/35}` with `(out<=0 ? 1e-16 : out)`;
* tree terms: `exp(lnpk_l) * f²/5 + large_for_logs_big` etc.
* `sigmav = Σ_i (lnk[i+1]−lnk[i])·(k_{i+1}P_{i+1} + k_iP_i)/2 / (6π²)` (`3265-3267`).

**Campaign decision:** clax computes the remap on all 256 bins (its `CubicSpline`
clamps `ktrue` to the grid ends, §3); every new comparison test restricts to
`k_h[10] <= k <= 0.3 h/Mpc` (`k_h[10] ≈ 9.0e-5`), so the extrapolation window is
never compared. State this in each test docstring.

## §3 AP spline macros — `nonlinear_pt.c:2372-2395`

```c
#define AP_BSEARCH_SETUP()                       /* binary search kdisc for ktrue */ \
  ap_inf_ = 0; ap_sup_ = Nmax - 1;                                                    \
  while (ap_sup_ - ap_inf_ > 1) { ap_mid_ = (ap_sup_ + ap_inf_) >> 1;                \
    if (ktrue < kdisc[ap_mid_]) ap_sup_ = ap_mid_; else ap_inf_ = ap_mid_; }          \
  ap_h_ = kdisc[ap_sup_] - kdisc[ap_inf_];  ap_b_ = (ktrue - kdisc[ap_inf_]) / ap_h_; \
  ap_a_ = 1. - ap_b_;  ap_a3a_ = ap_a_*ap_a_*ap_a_ - ap_a_;                           \
  ap_b3b_ = ap_b_*ap_b_*ap_b_ - ap_b_;  ap_h2_6_ = ap_h_*ap_h_/6.;
#define AP_INTERP_FAST(name)                                                          \
  name##_ap_out = ap_a_*name[ap_inf_] + ap_b_*name[ap_sup_]                           \
                + (ap_a3a_*dd_##name[ap_inf_] + ap_b3b_*dd_##name[ap_sup_])*ap_h2_6_;
```
`dd_<name>` are natural-cubic-spline second derivatives in **linear k**
(`array_spline_table_columns(kdisc, Nmax, name, 1, dd_name, _SPLINE_NATURAL_, ...)`).
The bias block uses `AP_INTERP_EX_FAST(name)` — same arithmetic on the `_new` copies.

Extrapolation: the bisection clamps the *interval* to `[0, Nmax−2]` but `ap_b_` is
computed from the unclamped `ktrue`, so outside `[kdisc[0], kdisc[Nmax−1]]` the end
interval's **cubic is continued**. (Reachable only for the last ~Nside points at
`hratio > 1`; CLASS-PT's own loop runs over the interior `Nside … Nmax−Nside`, §4.)

clax: `clax.interpolation.CubicSpline(k_h, y).evaluate(ktrue)` has the same natural
boundary and A/B/(A³−A)/(B³−B)·h²/6 formula (`interpolation.py:52-75`) but **clamps
`ktrue`** to `[k_h[0], k_h[-1]]` (constant extrapolation) — a different function
outside the grid. Part 1b B5 therefore builds `_channels_at(k, chan, kq)` directly
on `_compute_natural_spline_coeffs` (`interpolation.py:139-197`) with the unclamped
`b`, one vmapped coefficient solve for all 47 channels, evaluated at the (40, Nk)
`ktrue` array. At α=1, `ktrue == k_h[j]` exactly → `A=1, B=0` → the knot value is
returned bit-for-bit, which is why the α=1 baseline holds to 1e-10.

## §4 GL μ-loop, RSD block — `nonlinear_pt.c:4386-4562`

Runs when `irindex == 1` (IR resummation on) **regardless of AP**; with AP=No the
branch at `4395-4398` sets `mutrue = mu; ktrue = kdisc[index_j]` (and Nside=1). The
no-IR path is the analytic assembly at `3886-3907` (not mirrored: clax always IR-resums).

Per `index_j` in the window, per GL node `(mu, w) = (gauss_x[i], gauss_w[i])`, `i < 40`:

```c
ap_fac = sqrt(1./Dratio/Dratio + (hratio*hratio - 1./Dratio/Dratio)*mu*mu);   // 4392
mutrue = mu*hratio/ap_fac;  ktrue = kdisc[index_j]*ap_fac;                     // 4393-4394
AP_BSEARCH_SETUP(); AP_INTERP_FAST(Pnw); AP_INTERP_FAST(Pw);                   // 4401-4404
AP_INTERP_FAST( P22_mu4_vv, P13_mu4_vv, P22_mu4_vv_w, P13_mu4_vv_w,            // 4411-4439
                P13_mu6, P22_mu6_vv, P22_mu6_vv_w, P13_mu6_w, P22_mu8, P22_mu8_w,
                P22_mu0_dd, P13_mu0_dd, P13_mu0_dd_w, P22_mu0_dd_w,
                P22_mu2_dd, P13_mu2_dd, P22_mu2_dd_w, P13_mu2_dd_w, P22_mu4_dd, P22_mu4_dd_w,
                P13_mu2_vd, P22_mu2_vd, P22_mu2_vd_w, P13_mu2_vd_w,
                P13_mu4_vd, P22_mu4_vd, P22_mu4_vd_w, P13_mu4_vd_w, P22_mu6_vd, P22_mu6_vd_w );
LegendreP2 = (3mu²−1)/2;  LegendreP4 = (35mu⁴−30mu²+3)/8;      // 4470-4471  FIDUCIAL mu
mu2t = mutrue²; mu4t = mu2t²; mu6t = mu4t·mu2t; mu8t = mu4t²;  // 4474-4477  TRUE mu
Sigmatot = SigmaBAO*(1 + f*mu2t*(2+f)) + f*f*mu2t*(mu2t−1)*deltaSigmaBAO;   // 4480
Exp = exp(−Sigmatot*ktrue*ktrue);                                            // 4481
p_tree = Pnw_ap + (1 + Sigmatot*ktrue²)*Pw_ap*Exp;                           // 4483
P13ratio = 1 + (Pw_ap/Pnw_ap)*Exp;                                           // 4485
V = hratio/Dratio/Dratio;                     // written inline on every term below
p_tree_vv = p_tree*f*f*mu4t*w*V;  p_tree_vd = p_tree*2*f*mu2t*w*V;  p_tree_dd = p_tree*w*V;   // 4494-4496
Pctr0 = ktrue²*(Pnw_ap + Pw_ap*Exp)*w*V;                                     // 4498
Pctr2 = (Pnw_ap + Pw_ap*Exp)*w*f*mu2t*ktrue²*V;                              // 4499
Pctr4 = ktrue²*(Pnw_ap + Pw_ap*Exp)*w*f*f*mu4t*V;                            // 4500
P1loopvv = ( (P13_mu4_vv_ap*P13ratio + P22_mu4_vv_ap + (P22_mu4_vv_w_ap + P13_mu4_vv_w_ap)*Exp)*mu4t
           + (P13_mu6_ap*P13ratio    + P22_mu6_vv_ap + (P22_mu6_vv_w_ap + P13_mu6_w_ap)*Exp)*mu6t
           + (P22_mu8_ap + P22_mu8_w_ap*Exp)*mu8t ) * w*V;                   // 4503
P1loopdd = ( (P22_mu0_dd_ap + P13_mu0_dd_ap*P13ratio + (P13_mu0_dd_w_ap + P22_mu0_dd_w_ap)*Exp)
           + (P22_mu2_dd_ap + P13_mu2_dd_ap*P13ratio + (P22_mu2_dd_w_ap + P13_mu2_dd_w_ap)*Exp)*mu2t
           + (P22_mu4_dd_ap + P22_mu4_dd_w_ap*Exp)*mu4t ) * w*V;             // 4512
P1loopvd = ( (P13_mu2_vd_ap*P13ratio + P22_mu2_vd_ap + (P22_mu2_vd_w_ap + P13_mu2_vd_w_ap)*Exp)*mu2t
           + (P13_mu4_vd_ap*P13ratio + P22_mu4_vd_ap + (P22_mu4_vd_w_ap + P13_mu4_vd_w_ap)*Exp)*mu4t
           + (P22_mu6_vd_ap + P22_mu6_vd_w_ap*Exp)*mu6t ) * w*V;             // 4521
P1loopdd_ap_ir = P1loopdd + p_tree*w*V;            // 4530: p_tree added inside the mu^0 bracket
P1loopvd_ap_ir = P1loopvd + p_tree*2*f*mu2t*w*V;   // 4532: p_tree*2f added inside the mu^2 bracket
LEGENDRE_PROJECT(P1loopvv, P1loop_0_vv, P1loop_2_vv, P1loop_4_vv);           // 4534
P1loop_0_dd += P1loopdd*L0/2;  P1loop_2_dd += P1loopdd_ap_ir*L2*2.5;  P1loop_4_dd += P1loopdd_ap_ir*L4*4.5;  // 4535-4537
P1loop_0_vd += P1loopvd*L0/2;  P1loop_2_vd += P1loopvd*L2*2.5;        P1loop_4_vd += P1loopvd_ap_ir*L4*4.5;  // 4538-4540
P_CTR_0 += Pctr0*L0/2;  P_CTR_2 += Pctr2*L2*2.5;  P_CTR_4 += Pctr4*L4*4.5;   // 4551-4553
LEGENDRE_PROJECT(p_tree_vv, Ptree_0_vv, Ptree_2_vv, Ptree_4_vv);             // 4555
Ptree_0_vd += p_tree_vd*L0/2;  Ptree_0_dd += p_tree_dd*L0/2;  Ptree_2_vd += p_tree_vd*L2*2.5;   // 4556-4558
```

Consequences the plan relies on:

* pm rows `2_dd` (26), `4_vd` (28), `4_dd` (29) **contain the tree contribution**
  (`_ap_ir`), while `0_vv/0_vd/0_dd`, `2_vv/2_vd`, `4_vv` do not (tree lives in
  `Tree_0_vv/0_vd/0_dd`, `Tree_2_vv/2_vd`, `Tree_4_vv` = pm 15-20). There is no
  `Tree_2_dd`, `Tree_4_vd`, `Tree_4_dd` row.
* Legendre weights use the **fiducial** μ; every μ-power in the kernels uses **mutrue**.
* The counterterms are in-loop at `ktrue` with the anisotropic `Exp` — an analytic
  `−k²P·{1, 2f/3, 8f²/35}` matches only at α=1 without IR damping differences.
* `V = hratio/Dratio²` multiplies every in-loop term; nothing outside the loop
  (pm 2, 3, 6, 10, 14 and the fNL rows) carries it.

## §5 GL μ-loop, bias block — `nonlinear_pt.c:5225-5366`

Before the loop (`5225-5250`) a `memset(0)` on: `P_IFG2_0b1_x, P_IFG2_0, P_IFG2_2,
P_Id2d2_2, P_Id2d2_4, P_Id2G2_2, P_Id2G2_4, P_IG2G2_2, P_IG2G2_4, P_4_b1b2, P_4_b1bG2,
P_Id2d2, P_Id2G2, P_IG2G2, P_0_b1b2, P_2_b1b2, P_0_b1bG2, P_2_b1bG2, P_0_b2, P_2_b2,
P_4_b2, P_0_bG2, P_2_bG2, P_4_bG2` (+fNL). So under the GL loop the real-space
monopoles `Id2d2`, `Id2G2`, `IG2G2` (pm 1, 4, 5) **are re-accumulated in-loop**;
`IFG2` (pm 6), `Id2` (pm 2), `IG2` (pm 3), `nl` (pm 0), `Tree` (pm 14) stay un-remapped.

Per node (same `mu, w, ap_fac, mutrue, ktrue` as §4):
```c
LegendreP2 = (3mu²−1)/2;  LegendreP4 = (35mu⁴−30mu²+3)/8;
mu2t = mutrue²; mu4t = mu2t²;
LegendreP2true = (3mu2t−1)/2;  LegendreP4true = (35mu4t−30mu2t+3)/8;
AP_BSEARCH_SETUP(); AP_INTERP_FAST(Pnw); AP_INTERP_FAST(Pw); AP_INTERP_FAST(Pbin);
AP_INTERP_EX_FAST( P_IFG2, P_Id2d2, P_Id2G2, P_IG2G2, P_0_b1b2, P_2_b1b2, P_0_b1bG2, P_2_b1bG2,
                   P_0_b2, P_2_b2, P_4_b2, P_0_bG2, P_2_bG2, P_4_bG2 );         // from the _new copies
Sigmatot = SigmaBAO*(1 + f*mu2t*(2+f)) + f*f*mu2t*(mu2t−1)*deltaSigmaBAO;
Exp = exp(−Sigmatot*ktrue²);
p_lo = (Pnw_ap + Pw_ap*Exp) / Pbin_ap;                                          // IR ratio on the linear P
IFG2_in  = p_lo*P_IFG2_ap*w*V;
Pd2d2_in = P_Id2d2_ap*w*V;   Pd2G2_in = P_Id2G2_ap*w*V;   PG2G2_in = P_IG2G2_ap*w*V;
Pb1b2_in  = (P_0_b1b2_ap  + LegendreP2true*P_2_b1b2_ap )*w*V;
Pb1bG2_in = (P_0_b1bG2_ap + LegendreP2true*P_2_b1bG2_ap)*w*V;
Pb2_in    = (P_0_b2_ap  + LegendreP2true*P_2_b2_ap  + LegendreP4true*P_4_b2_ap )*w*V;
PbG2_in   = (P_0_bG2_ap + LegendreP2true*P_2_bG2_ap + LegendreP4true*P_4_bG2_ap)*w*V;
P_IFG2_0b1_x += IFG2_in*L0/2;  P_IFG2_0 += IFG2_in*f*mu2t*L0/2;  P_IFG2_2 += IFG2_in*f*mu2t*L2*2.5;
LEGENDRE_PROJECT(Pd2d2_in, P_Id2d2, P_Id2d2_2, P_Id2d2_4);
LEGENDRE_PROJECT(Pd2G2_in, P_Id2G2, P_Id2G2_2, P_Id2G2_4);
LEGENDRE_PROJECT(PG2G2_in, P_IG2G2, P_IG2G2_2, P_IG2G2_4);
LEGENDRE_PROJECT(Pb1b2_in,  P_0_b1b2,  P_2_b1b2,  P_4_b1b2);
LEGENDRE_PROJECT(Pb1bG2_in, P_0_b1bG2, P_2_b1bG2, P_4_b1bG2);
LEGENDRE_PROJECT(Pb2_in,  P_0_b2,  P_2_b2,  P_4_b2);
LEGENDRE_PROJECT(PbG2_in, P_0_bG2, P_2_bG2, P_4_bG2);
```
Note `Pb1b2_in`/`Pb1bG2_in` reconstruct the μ-dependence from the *pre-loop* ℓ=0,2
multipoles (there is no ℓ=4 input for b1b2/b1bG2 — `P_4_b1b2` and `P_4_b1bG2` are
**generated** by the projection of the L2true term, not zero). `Pb2_in`/`PbG2_in`
use ℓ=0,2,4 inputs. `p_lo` applies IR damping to `IFG2` only.

## §6 LEGENDRE_PROJECT — `nonlinear_pt.c:2565-2568`

```c
#define LEGENDRE_PROJECT(val, P0, P2, P4) \
  P0[index_j] += (val)*LegendreP0/2.;  P2[index_j] += (val)*LegendreP2*2.5;  P4[index_j] += (val)*LegendreP4*4.5;
```
Weights `(2ℓ+1)/2` with the fiducial-μ Legendre polynomials; GL nodes span μ∈[−1,1].

## §7 GL nodes — `nonlinear_pt.c:986-992`

`pnlpt->gauss` is read from `__CLASSDIR__/pt_matrices/gauss_tab.dat`: 80 numbers,
`gauss_x = gauss[0:40]`, `gauss_w = gauss[40:80]`. clax loads the **same file**
(`clax/ept.py:55-77`) via `_CLASSPT_DIR = <repo>/../CLASS-PT`, i.e.
`/home/n2minh/CLASS-PT/pt_matrices/gauss_tab.dat`, and silently falls back to a
10-point `numpy.polynomial.legendre.leggauss` (`ept.py:72-77`) when the file is
missing. **Every sbatch script asserts `len(clax.ept._GAUSS_NODES) == 40`.**

## §8 Dead accumulators

`P10b1`, `P10`, `P12` (`4559-4561`) are accumulated, freed at `5507-5508`, and never
written to an output row. Mirror nothing for them.

## §9 Id2d2 conventions

* `4852-4853`: `f22_Id2d2[j] = 2·f22_Id2d2_real[j]` and
  `P_Id2d2[j] = fabs(k³·f22_Id2d2[j] − k0³·2·f22_Id2d2_real[0] + epsilon_for_logs)`,
  `epsilon_for_logs = 1e-6` (`5486-5492`). Since `2·f22_real[0] = f22_Id2d2[0]`, this is
  simply `|P(k) − P(k0) + ε|` with `P = k³·f22_Id2d2` — the same array clax builds as
  `raw_Id2d2 = 2·k³·Σ x2·(x2@M22b)` (`ept.py:1019-1021`). No cutoff factor is applied
  here (clax multiplies by `uv_damp = exp(−(k/3)⁶)`, < 1e-6 below k = 0.3 h/Mpc).
* Under the GL loop the monopole is re-accumulated from that array (§5), so pm[1]
  is `∫dμ/2 · V · P_Id2d2(ktrue)`.
* Output: `pk_Id2d2 = interp + large_for_logs_big` (`5455-5461`), and `get_pk_mult`
  returns `pm[1] = −raw[1] + large_b` → **`pm[1] = −P_Id2d2 ≤ 0`** (§10).
* classy's `pk_gg_l0` adds `+0.25·b2²·pm1·h³` **and** `+0.25·b2²·Pd2d2_0` (§11-12), so
  the physical b2² monopole is `0.25 b2² (Pd2d2_0 − P_Id2d2(k))`. `pk_gg_real` adds
  `0.25 b2² pm1 h³` only (no `Pd2d2_0`).
* clax today: the array `Pk_Id2d2 = (|raw − raw[0]| + 1e-6)·uv_damp` (`ept.py:1020-1021`)
  equals `P_Id2d2` (both ≥ 0); the accessors are wrong: `pk_gg_l0` adds
  `+0.25 b2² Pk_Id2d2` where classy adds `0.25 b2² (Pd2d2_0 − P_Id2d2)`, and
  `pk_gg_real` adds `+0.25 b2² Pk_Id2d2` where classy adds `−0.25 b2² P_Id2d2`
  (**Bug #2**, Part 1b Task B4: accessor sign flip + `Pd2d2_0`; the array stays).

## §10 pm rows — `classy.pyx:4607-4686` and `nonlinear_pt.c:5560-5625`

`raw` = 96×k_size array from `nonlinear_pt_pk_mult_at_kvec_and_z` (CLASS units,
Mpc³; ctr rows Mpc). `large_m = 50000, large_b = 1e6, large_s = 100`. Transforms:

```
pm[0]  = raw[0]  − large_m          nl  (1-loop matter, real space)
pm[1]  = −raw[1] + large_b          Id2d2      → −P_Id2d2  (≤ 0)
pm[2]  = raw[2]  − large_s          Id2
pm[3]  = −raw[3] + large_s          IG2        → −P_IG2
pm[4]  = −raw[4] + large_b          Id2G2      → −P_Id2G2
pm[5]  = raw[5]  − large_b          IG2G2
pm[6]  = −raw[6] + large_s          IFG2       → −P_IFG2
pm[7..9]  = −raw + large_b          IFG2_0b1, IFG2_0, IFG2_2   → negated
pm[10..13] = −raw                   CTR, CTR_0, CTR_2, CTR_4   → negated, no offset
pm[14] = raw[14]                    Tree (linear, IR-resummed)
pm[15..29] = raw − large_b          Tree_0_vv(15) Tree_0_vd(16) Tree_0_dd(17) Tree_2_vv(18)
                                    Tree_2_vd(19) Tree_4_vv(20) 0_vv(21) 0_vd(22) 0_dd(23)
                                    2_vv(24) 2_vd(25) 2_dd(26) 4_vv(27) 4_vd(28) 4_dd(29)
pm[30..47] (bias, analogous offsets): 0_b1b2(30) 0_b2(31) 0_b1bG2(32) 0_bG2(33)
                                    2_b1b2(34) 2_b2(35) 2_b1bG2(36) 2_bG2(37) 4_b2(38)
                                    4_bG2(39) 4_b1b2(40) 4_b1bG2(41) Id2d2_2(42) Id2G2_2(43)
                                    IG2G2_2(44) Id2d2_4(45) Id2G2_4(46) IG2G2_4(47)
pm[48..71] fNL rows (NaN/garbage when PNG is off — never compare)
```

Units for a clax-side comparison (`h = ref["h"]`):

| clax `EPTComponents` field | pm row · factor |
|---|---|
| `Pk_tree` | `pm[14]·h³` |
| `Pk_loop` | `pm[0]·h³` |
| `Pk_ctr` | `pm[10]·h` |
| `Pk_ctr0/2/4` | `pm[11/12/13]·h` |
| `Pk_Id2`, `Pk_IG2`, `Pk_IFG2` | `pm[2/3/6]·h³` (sign per row transform above) |
| `Pk_Id2d2`, `Pk_Id2G2`, `Pk_IG2G2` | `pm[1/4/5]·h³` |
| `Pk_IFG2_0b1`, `Pk_IFG2_0`, `Pk_IFG2_2` | `pm[7/8/9]·h³` |
| `Pk_0_vv + Pk_0_vv1` etc. (tree+loop split, see §15) | `pm[15]+pm[21]` etc. `·h³` |
| `Pk_0_b1b2, Pk_0_b2, Pk_0_b1bG2, Pk_0_bG2` | `pm[30/31/32/33]·h³` |
| `Pk_2_b1b2, Pk_2_b2, Pk_2_b1bG2, Pk_2_bG2` | `pm[34/35/36/37]·h³` |
| `Pk_4_b2, Pk_4_bG2, Pk_4_b1b2, Pk_4_b1bG2` | `pm[38/39/40/41]·h³` |

## §11 classy accessor formulas — `classy.pyx:4795-4915` (verbatim algebra)

`h = self.ba.h`, `fz = self.fz`, `kh = self.kh` (§12), `pm = self.pk_mult`.
```
pk_mm_real(cs)                     = (pm0 + pm14 + 2 cs pm10/h²)·h³
pk_gg_real(b1,b2,bG2,bGamma3,cs,cs0,Pshot)
   = (b1² pm14 + b1² pm0 + 2(cs b1² + cs0 b1) pm10/h² + b1 b2 pm2 + 0.25 b2² pm1
      + 2 b1 bG2 pm3 + b1(2 bG2 + 0.8 bGamma3) pm6 + bG2² pm5 + b2 bG2 pm4)·h³ + Pshot
pk_gm_real(b1,b2,bG2,bGamma3,cs,cs0)
   = (b1 pm14 + b1 pm0 + (2 cs b1 + cs0) pm10/h² + (b2/2) pm2 + bG2 pm3 + (bG2 + 0.4 bGamma3) pm6)·h³
pk_mm_l0(cs0) = (pm15 + pm21 + pm16 + pm22 + pm17 + pm23 + 2 cs0 pm11/h²)·h³
pk_mm_l2(cs2) = (pm18 + pm24 + pm19 + pm25 + pm26 + 2 cs2 pm12/h²)·h³
pk_mm_l4(cs4) = (pm20 + pm27 + pm28 + pm29 + 2 cs4 pm13/h²)·h³
pk_gg_l0(b1,b2,bG2,bGamma3,cs0,Pshot_nbar,a0_nbar,a2_nbar,b4)
   = (pm15 + pm21 + b1 pm16 + b1 pm22 + b1² pm17 + b1² pm23 + 0.25 b2² pm1 + b1 b2 pm30 + b2 pm31
      + b1 bG2 pm32 + bG2 pm33 + b2 bG2 pm4 + bG2² pm5 + 2 cs0 pm11/h² + (2 bG2 + 0.8 bGamma3)(b1 pm7 + pm8))·h³
     + Pshot_nbar + a0_nbar (kh/0.45)² + a2_nbar (1/3)(kh/0.45)² + 0.25 b2² Pd2d2_0
     + fz² b4 kh² (fz²/9 + 2 fz b1/7 + b1²/5)(35/8) pm13·h
pk_gg_l2(b1,b2,bG2,bGamma3,cs2,a2_nbar,b4)
   = (pm18 + pm24 + b1 pm19 + b1 pm25 + b1² pm26 + b1 b2 pm34 + b2 pm35 + b1 bG2 pm36 + bG2 pm37
      + 2 cs2 pm12/h² + (2 bG2 + 0.8 bGamma3) pm9)·h³
     + a2_nbar (2/3)(kh/0.45)² + fz² b4 kh² ((70 fz² + 165 fz b1 + 99 b1²)·4/693)(35/8) pm13·h
pk_gg_l4(b1,b2,bG2,bGamma3,cs4,b4)
   = (pm20 + pm27 + b1 pm28 + b1² pm29 + b2 pm38 + bG2 pm39 + 2 cs4 pm13/h²)·h³
     + fz² b4 kh² ((210 fz² + 390 fz b1 + 143 b1²)·8/5005)(35/8) pm13·h
```
Read-off for clax (**Bug #3**): ℓ=2 galaxy tree+loop is `pm18+pm24 + b1(pm19+pm25) + b1²·pm26`
(pm26 includes tree dd); ℓ=4 is `pm20+pm27 + b1·pm28 + b1²·pm29` (pm28, pm29 include
tree). clax `pk_gg_l2` (`ept.py:2150`) omits `b1²·Pk_2_dd`; `pk_gg_l4` (`2196`) uses
`Pk_4_vv + Pk_4_vd + Pk_4_dd` without b1 weights.

**Bug #5** (`pk_gm_real`, `classy.pyx:4821`): the counterterm is `(2 cs b1 + cs0) pm10/h²`;
clax `pk_gm_real` has `(cs b1 + cs0) Pk_ctr` — factor 2 missing on `cs b1`. Invisible
at the legacy `cs = 0`.

Quirk (mirror, do not fix): `pk_gg_l4` reads `pm38` (`4_b2`) and `pm39` (`4_bG2`) but never
`pm40` (`4_b1b2`) or `pm41` (`4_b1bG2`), although the bias loop generates both (§5,
`5344-5352`). clax keeps `Pk_4_b1b2`, `Pk_4_b1bG2` as leaves and its `pk_gg_l4` ignores
them, exactly like classy.

Signature note: the legacy generator calls `pk_gg_l0(b1,b2,bG2,bGamma3,cs0,Pshot,b4)`
**positionally** (7 args) — with the current 9-arg signature that binds `b4` to
`a0_nbar`. The refactored generator (Part 1, A3) must use keyword arguments.

## §12 `initialize_output`, `Pd2d2_0`, kh units — `classy.pyx:4783-4792`

```python
def initialize_output(self, k, z, k_size):
    self.kh = k                                     # 4785  k as passed by the caller
    self.fz = self.scale_independent_growth_factor_f(z)
    self.pk_mult = self.get_pk_mult(k, z, k_size)   # needs k in 1/Mpc (CLASS units)
    Plin_hMpc3 = self.pk_mult[14]*self.ba.h**3
    self.Pd2d2_0 = simpson(Plin_hMpc3**2. * self.kh**3., x=np.log(self.kh)) / (np.pi**2.)
```
`get_pk_mult` interpolates at `k` in **1/Mpc**, so callers pass `k_h*h` and `self.kh`
becomes 1/Mpc — while `Pd2d2_0` (units (Mpc/h)³ from `Plin_hMpc3²·kh³`) and the
`b4 kh²` / `a_n (kh/0.45)²` terms are written for `kh` in h/Mpc. Either the caller
must pass h/Mpc (then `get_pk_mult` is evaluated at the wrong k) or `self.kh` must be
divided by h. `cdef double Pd2d2_0` and `cdef double fz` (`classy.pyx:119-120`) are not
`public` — Python cannot read them; the patch (§13) adds a getter.

Legacy provenance question (Part 1a A3/A4): which convention produced
`reference_data/classpt_z0.38_fullrange.npz`? Decidable offline: the file stores
`pk_mult` and `pk_gg_l0/l2/l4` with b4=500 — assemble §11 with `kh = k_h` and with
`kh = k_h·h` and see which reproduces the stored spectra to 1e-8.

## §13 Patches (`scripts/classpt_patches/`)

Produced by editing `/home/n2minh/CLASS-PT` and running
`git -C /home/n2minh/CLASS-PT diff > scripts/classpt_patches/<name>.patch`; sha256 in
`reference_data/classpt/MANIFEST.md`. Applied by `scripts/setup_classpt_env.sh`
(idempotent: `git apply --reverse --check` first).

**`classy_ap_ratios.patch`** — `python/cclassy.pxd`, inside `cdef struct nonlinear_pt:`
(line 413+, which already declares `int z_pk_num`, `double z_pk[100]`, `int k_size`):
```cython
        double * growthf
        double * hratio_array
        double * Dratio_array
```
and `python/classy.pyx`, a new method next to `initialize_output` (4783):
```cython
    def get_ap_ratios(self, double z):
        """(hratio, Dratio, f) as used in-loop by nonlinear_pt.c:1266-1296 for z_pk == z.
        AP=No returns (1, 1, f). Raises if z is not one of the requested z_pk values."""
        cdef int i
        for i in range(self.nlpt.z_pk_num):
            if abs(self.nlpt.z_pk[i] - z) < 1e-10:
                return (self.nlpt.hratio_array[i], self.nlpt.Dratio_array[i], self.nlpt.growthf[i])
        raise CosmoSevereError("get_ap_ratios: z=%g is not in z_pk" % z)

    def get_Pd2d2_0(self):
        """Value computed by the last initialize_output() call (classy.pyx:4791)."""
        return self.Pd2d2_0
```
Partial `cdef extern` struct declarations are legal Cython: the C compiler resolves
the fields from `nonlinear_pt.h:588-590`.

**`classy_kh_units.patch`** — `python/classy.pyx:4785`: `self.kh = k / self.ba.h`.
Applied for the campaign (rationale §12); the A4 provenance gate records which
convention the legacy file used.

## §14 Quirks to mirror, not fix

1. `Dfid` radiation term frozen at target z (§1).
2. Fiducial-μ Legendre weights with true-μ kernels (§4, §5).
3. `_ap_ir` folding: tree included in `2_dd`, `4_vd`, `4_dd` rows only (§4).
4. `p_lo` IR ratio applied to `IFG2` only; `Id2d2/Id2G2/IG2G2` monopoles remapped
   without IR factors (§5).
5. `P_Id2d2 = |P(k) − P(k0) + ε|` (no cutoff factor), then negated in `get_pk_mult`; `pk_gg_l0` adds `Pd2d2_0` back, `pk_gg_real` does not (§9).
6. `Nside = 10` window under AP (§2) — compare in-window only.
7. Analytic counterterm/IFG2 multipoles in clax (`Pk_ctr0/2/4`, `Pk_IFG2_0/2`) must
   move in-loop to match `Pctr*`, `IFG2_in` at ktrue (Part 1b, B3).
8. `pk_gg_l0` in classy adds `0.25 b2² Pd2d2_0`; `pk_gg_real` does not (§9).

## §15 clax-side channel inventory (`clax/ept.py` at `bf8ac18`)

`EPTComponents` pytree leaves (positional order, `ept.py:1652` region):
`kh(0) Pk_tree(1) Pk_loop(2) Pk_ctr(3) Pk_Id2d2(4) Pk_Id2(5) Pk_IG2(6) Pk_Id2G2(7)
Pk_IG2G2(8) Pk_IFG2(9) Pk_IFG2_0b1(10) Pk_IFG2_0(11) Pk_IFG2_2(12) Pk_ctr0/2/4(13-15)
Pk_0_vv/vd/dd(16-18) Pk_2_vv/vd(19-20) Pk_4_vv(21) Pk_0_vv1/vd1/dd1(22-24)
Pk_2_vv1/vd1/dd1(25-27) Pk_4_vv1/vd1/dd1(28-30) Pk_0_b1b2/b2/b1bG2/bG2(31-34)
Pk_2_b1b2/b2/b1bG2/bG2(35-38) Pk_4_b2/bG2/b1b2/b1bG2(39-42) pk_nw(43) pk_w(44)
P22_mu6_vv(45) P22_mu6_vd(46) P22_mu8(47) P13_mu6(48) Pk_2_dd(49) Pk_4_vd(50) Pk_4_dd(51)`
then scalars `h, f, sigma2_bao, delta_sigma2_bao`.

Split convention: `Pk_<l>_<ab>` = tree part, `Pk_<l>_<ab>1` = loop part
(pm row = `Pk_l_ab + Pk_l_ab1`). `Pk_2_dd`, `Pk_4_vd`, `Pk_4_dd` exist as separate
tree leaves (49-51) — under the `_ap_ir` folding the reference rows 26/28/29 equal
`tree + loop` for those three.

GL loop today (`ept.py:1441-1497`): `for _mu_g, _w_g in zip(_GAUSS_NODES, _GAUSS_WEIGHTS)`
with `_Sig`, `_Eg = exp(−_Sig k²)`, `_r13 = where(pk_nw>1e-100, 1 + (pk_w/pk_nw)·_Eg, 1)`,
`_Pvv/_Pdd/_Pvd` brackets on the `*_nw/*_w` channels, `_L0/_L2/_L4`, `_p_tree`,
`_tree_vv = f²μ⁴p_tree`, `_tree_vd = 2fμ²p_tree`, `_tree_dd = p_tree`, accumulating
`Pk_{0,2,4}_{vv,vd,dd}` (tree) and `Pk_{0,2,4}_{vv1,dd1,vd1}` (loop) with weights
`w·{0.5, 2.5, 4.5}·L`. Everything is evaluated at the grid `k` (α=1 form).

Channel arrays that B5 remaps at `ktrue` (natural cubic spline in linear `k_h`, end cubic continued — §3):
`pk_nw, pk_w, pk_disc` (=`Pbin`), the 30 P13/P22 μ-power channels listed in §4
(nw and w each), `Pk_IFG2, Pk_Id2d2, Pk_Id2G2, Pk_IG2G2, Pk_0_b1b2, Pk_2_b1b2,
Pk_0_b1bG2, Pk_2_b1bG2, Pk_0_b2, Pk_2_b2, Pk_4_b2, Pk_0_bG2, Pk_2_bG2, Pk_4_bG2`.

Known clax-pt defects at `bf8ac18` (all fixed in Part 1b B4):
* **Bug #1** `ept.py:1088-1092` rebinds `nu1 = nu_i, nu2 = nu_l` (matter basis,
  b=−0.3) and the RSD bias kernels from `1506` to `~1620` use them although the
  comment at `1502` claims the bias basis (`b = B_BASIC = −1.6`). Rebinding
  `nu1 = -0.5*eta_i; nu2 = -0.5*eta_l` before 1506 makes `Pk_0_b1b2(f=0) == Pk_Id2`
  exactly and brings all ten `Pk_{0,2,4}_{b1b2,b2,b1bG2,bG2}` within ~0.5% of legacy
  `pm[30..39]·h³`. Invisible to existing tests because b2 = bG2 = 0 there.
* **Bug #2** Id2d2 sign / missing `Pd2d2_0` (§9).
* **Bug #3** `pk_gg_l2` / `pk_gg_l4` b1 weighting (§11).
* **Bug #4** `Pk_tree = pk_lin_h` (raw linear) on every IR branch (`ept.py:1774,
  1787, 1790`); CLASS-PT `pm[14]` is the IR-resummed
  `Ptree = Pnw + Pw·e^{−Σk²}(1 + Σk²)` (`nonlinear_pt.c:2999`, §14.4). Affects the
  three real-space accessors and the `Pd2d2_0` integrand (§12).
* **Bug #5** `pk_gm_real` counterterm `(cs b1 + cs0)` should be `(2 cs b1 + cs0)` (§11).
* `Pk_4_b1b2 = Pk_4_b1bG2 = zeros` with a stale comment (`1621-1623`); CLASS-PT
  generates them in-loop (§5) — B3 generates them too; classy's `pk_gg_l4` never
  reads them (§11 quirk).
* `CubicSpline.evaluate` clamps outside the grid; CLASS-PT continues the end cubic
  (§3) — B5 uses its own `_channels_at`.
* `Pk_4_vd1` vs `pm[28]·h³ − Pk_4_vd` has a median ratio ≈ 0.83 at legacy fiducial
  (17% deficit) — unexplained; B3 investigates before B4 changes the loop.
