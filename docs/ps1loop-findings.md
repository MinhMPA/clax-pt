# clax-pt vs ps_1loop_jax: findings for both codes (draft, Oct 13, 2026)

**Status: draft for the user. Nothing here has been posted.** Findings 1-4b concern ps_1loop_jax and are
candidates for issues on MinhMPA/ps_1loop_jax-for-pfs. Findings 5-8 concern CLASS-PT and clax-pt; some may
become upstream issues, like smsharma/clax#42-44.

They come from the second-oracle validation on branch `feat/ps1loop-cross-validation`:
- spec `docs/superpowers/specs/2026-10-05-clax-pt-vs-ps1loop-validation-design.md`;
- plan `docs/superpowers/plans/2026-10-05-clax-pt-vs-ps1loop-validation.md`;
- the record of decisions, `.superpowers/sdd/2026-10-05-clax-pt-vs-ps1loop-validation/progress.md` (the SDD
  ledger; not committed).

Versions: ps_1loop_jax 0.5.0 (fork main 576ec09) unless stated. clax-pt is this branch, which includes the
RSD-basis gradient fix (merged as 74222a6).

Every measured number below carries the igpu SLURM job that produced it. Logs are under
`/lustre/work/n2minh/std/clax/ps1loop/`: `ps1loop-tests.pytest.<job>.log` for tests and
`ps1loop-ladder.out.<job>` for the convergence ladder.

## How the codes were compared

- **Shared input.** Both codes get CLASS-PT's own linear P_cb, growth rate f, BAO scale r_s(z_d) h and AP ratios
  from the CLASS-PT reference files. The AP ratios are hratio = H_true/H_fid and Dratio = D_A,true/D_A,fid, with
  alpha_perp = Dratio and alpha_para = 1/hratio. So a difference is always the one-loop model or its numerics,
  never the linear theory. CLASS-PT breaks ties when the two codes disagree.
- **Grids.** The coefficient-level tests run on 14 cosmologies x z = 0, 0.38, 0.8 (42 files). The PFS error-bar
  gate runs on 5 cosmologies at the 7 PFS slices of Takada et al. (2014), z = 0.7-2.2 (35 files).
- **PFS bias point** (each slice):
  - bias parameters from the coevolution relations;
  - k^2 counterterms c0 = c2 = c4 = 2 (Mpc/h)^2;
  - fingers-of-god amplitude cfog = 500 (Mpc/h)^4. This is CLASS-PT's b4 in the reference set, a BOSS-like value;
  - shot noise at the slice's mean density.
- **PFS Delta chi^2.** Summed over 7 slices x 19 k bins (centres 0.015-0.195 h/Mpc, width 0.01 h/Mpc), with a
  Gaussian (P_0, P_2, P_4) covariance computed from clax-pt.
- **Two normalisations appear below. Each number says which one it uses; they are never mixed:**
  - **"% of max|clax P_l|"** is the campaign metric. Take the largest |ps - clax| over the comparison window
    (the reference grid without its first 10 points, k <= 0.3 h/Mpc). Divide by the window's largest |clax| for
    the same multipole or bias coefficient.
  - **"% of the derivative's scale"** applies to a derivative along an input direction t. Divide by the window's
    largest |derivative of clax-pt along t| (max|jvp_clax|).

## Summary

| # | finding | code | cost to a PFS analysis if ignored | recommended setting or fix |
|---|---|---|---|---|
| 1 | default nfft = 256 under-resolves the BAO wiggles in the one-loop terms | ps_1loop | 0.1-0.6% of single bias-coefficient tables; a slice Delta chi^2 moves by <= 15% | nfft = 512 (nfft >= 1024 with AP, finding 3) |
| 2 | derivatives along a k-stretch of P_lin differ by 0.25-1.5% of the derivative's scale beyond the documented model difference | open, not attributed | Fisher or Laplace derivatives along P_lin directions differ between codes by up to 1.5% at l = 4 | open; nfft >= 512 removes up to 41% of it at l = 4 |
| 3 | AP branch discontinuous with the no-AP branch at zero AP offset | ps_1loop | a 0.15 sigma step at alpha = 1, or the same error as an offset inside the AP branch | nfft >= 1024 with a 201-node mu grid (reduces 13-25x, does not remove) |
| 4 | IR-resummation scheme differs from CLASS-PT's | model difference | none once documented; it is part of the Delta chi^2 0.07-0.13 left with c = cfog = 0 in both codes | document; use the closed form when comparing codes |
| 4b | nfft = 2048 gives non-finite PT matrices | ps_1loop | the FFTLog plateau beyond 1024 cannot be checked | evaluate the Gamma ratio in log space (untested) |
| 5 | 13 projections computed but never added | CLASS-PT, clax-pt | six F_G2 lines: survey Delta chi^2 <= 2.2e-4 at the coevolution bias point; the other seven bounded at coefficient level only | keep the CLASS-PT convention (user decision); optional flag in clax-pt |
| 6 | AP treatment of the k^4 fingers-of-god term | CLASS-PT, clax-pt vs ps_1loop | fixed-nuisance Delta chi^2 1.55-40.6 at cfog = 500; absorbed (0.12-0.78) by refitting cfog, which then shifts by up to +20% | prefer the exact treatment for PFS; optional exact-AP FoG term in clax-pt |
| 7 | RSD-basis gradient freeze | clax-pt | AD gradients of P_0,2,4 along P_lin wrong at order unity | fixed (MinhMPA/clax-pt#9) |
| 8 | real-space bias loops carry AP; items filed as smsharma/clax#42-44 | CLASS-PT, clax-pt | real-space P_gg at an AP setting off by up to 12.4% in the b2^2, b2 bG2, bG2^2 terms | evaluate real space at hratio = Dratio = 1 |

---

## Part A: ps_1loop_jax

### 1. The default FFTLog sampling (nfft = 256) under-resolves the BAO wiggles in the one-loop terms

**Consequence.** At the default nfft = 256, ps_1loop's one-loop terms carry errors of 0.1-0.6% of individual
bias-coefficient scales at k ~ 0.15-0.25 h/Mpc. They are largest in cosmologies with a high baryon fraction.
Doubling to nfft = 512 removes most of the error. On the total PFS multipoles the effect is small: it moves a
z = 2.2 slice's Delta chi^2 by <= 15%. But it is the largest numerical error in ps_1loop's per-coefficient
tables, and it feeds the derivative remainder of finding 2.

**Mechanism.** nfft is the number of log-spaced k points, geomspace(1e-5, 1e3 h/Mpc, nfft)
(`ps_1loop.py:29-31, 44`). On this grid ps decomposes P_lin into power laws (FFTLog) and tabulates every one-loop
term, then splines the result to the output k (interpax cubic). At 256 points Delta ln k = 0.072, which is about
1.8 rad of BAO phase per grid step near k = 0.25 h/Mpc.

**Evidence.**
- **Real space, IR resummation off** (42 files, job 24147).
  - The c0 line (the k^2 P counterterm shape) is off by up to 0.617% of its scale (massive_nu_015_omega_cdm_low),
    against 0.33% at fiducial.
  - Rebuilding ps's 256-point spline of -2 k^2 P reproduces the residual to 7e-14 of the scale (scratch job 24155).
  - At nfft = 512 (job 24155): c0 0.057% (fiducial) and 0.125% (worst); F_G2 lines 0.39% -> 0.11%; I_d2
    0.13% -> 0.019%; bG2^2 0.105% -> 0.007%; b2 bG2 0.084% -> 0.004%.
  - clax-pt vs CLASS-PT on the same lines: <= 0.0114%, so the error is on the ps side (job 24155).
- **Redshift space, omitted-projection closure** (scratch elf comparison, jobs 24076-24078, z = 0.38).
  - The b1 bG2 l = 2 closure without ps's IR scheme (finding 4) misses by 1.015% of the coefficient's scale at
    nfft = 256 (omega_cdm_low; job 23244, 1% gate) and by 0.80% at nfft = 512.
  - With the IR scheme included, the remainder falls 0.52% -> 0.145% at nfft = 512 (omega_cdm_low) and
    0.33% -> 0.052% (lcdm_fiducial). On bG2 l = 4 it falls 0.46% -> 0.19% (omega_cdm_low).
  - The Task 9 ladder did not re-measure these per-coefficient lines.
- **Convergence ladder** (job 24919, lcdm_fiducial z = 0.38). The total-spectrum remainder max|ps - clax - D|
  (D = documented model difference, findings 4-6), as % of max|clax P_l| at l = 0 / 2 / 4:

  | nfft | l = 0 | l = 2 | l = 4 |
  |---|---|---|---|
  | 256 | 0.014 | 0.024 | 0.048 |
  | 512 | 0.010 | 0.021 | 0.047 |
  | 1024 | 0.010 | 0.022 | 0.047 |

  512 already gives the full improvement at this point.
- **PFS slice z = 2.2** (jobs 24899, 24919). Delta chi^2(ps - clax) for nfft 256 -> 512 -> 1024:
  - lcdm_fiducial 0.350 -> 0.305 -> 0.302;
  - w0wa_m07_m10 0.656 -> 0.694 -> 0.697.

**Cost if ignored.**
- Fits to the total P_0,2,4: one slice's Delta chi^2 moves by 0.04-0.05, which is negligible.
- Per-coefficient tables (emulator training, per-bias-parameter Fisher terms): 0.1-0.6% of a coefficient's scale.
- Derivatives along a k-stretch of P_lin: nfft 256 -> 512 shrinks finding 2's l = 4 remainder by 41%
  (1.493 -> 0.881% of the derivative's scale).

**Recommended setting.**
- nfft = 512 for one-loop accuracy. The ps fix branch kept the default at 256 to limit HMC cost, which grows as
  nfft^2 for the 22-type PT matrices.
- nfft >= 1024 whenever the AP branch is used (finding 3).
- Convergence beyond 1024 cannot be checked (finding 4b).

### 2. Open: along a k-stretch of the linear spectrum, the two codes' derivatives differ by 0.25-1.5% of the derivative's scale beyond the documented model difference

**Consequence.** Consider the direction that stretches P_lin in k, t = P dlnP/dlnk. Every parameter that moves
the BAO scale or reshapes the broadband (omega_b, omega_cdm, h, n_s) acts on P_lin partly along it. Along t,
d(P_0, P_2, P_4)/dt differs between ps_1loop and clax-pt by 0.25-1.5% of the derivative's scale. That is after
the derivative of the documented model difference D (findings 4-6) has been removed. Which code is closer to the
truth is not established.

The spectra themselves agree after D to <= 0.064% of max|clax P_l| (primal check, three cosmologies at
z = 0.38, job 24864). So a likelihood built from either code is unaffected; only the derivatives differ.

**Evidence** (Task 7, jobs 24844 and 24864, identical to the printed digit; z = 0.38).

- **Closed residual**, d(ps - clax)/dt - dD/dt, as % of the derivative's scale:

  | cosmology | l = 0 | l = 2 | l = 4 |
  |---|---|---|---|
  | lcdm_fiducial | 0.283 | 0.458 | 1.493 |
  | massive_nu_030 | 0.245 | 0.441 | 1.266 |
  | w0wa_m07_m10 | 0.307 | 0.425 | 1.320 |

- **D explains little of it.** The closed residual is up to 0.93 of the direct difference. The test therefore
  applies the discrimination rule: where the direct difference exceeds 0.1% of the scale, the closed residual must
  be <= 0.5 x the direct one. It fails on:
  - stretch l = 0 in 3 of 3 cosmologies;
  - stretch l = 2 and l = 4 in 2 of 3;
  - amplitude (t = P) l = 4 in lcdm_fiducial (closed 0.098% vs direct 0.185%).
- **The same closure works along the other inputs.** Along f, alpha_perp and alpha_para the closed residual is
  <= 0.44%, from direct differences of up to 10.4%.
- **Shape.** The residual is non-smooth in k: 97 / 99 / 97% of its rms survives a quartic fit in k. That metric
  cannot tell an oscillation from a hump. The residual peaks at k = 0.13-0.25 h/Mpc (scratch job 24845).
- **Size relative to the spectra, like for like** (both over the derivative's scale; job 24919, lcdm_fiducial).
  - The primal remainder max|ps - clax - D| is 0.025 / 0.043 / 0.106% of the derivative's scale. As a fraction of
    max|clax P_l| the same remainder is 0.014 / 0.024 / 0.048%.
  - The stretch residual is therefore 11.3 / 10.6 / 14.1x the primal remainder at nfft 256, and 8.4-11.2x on the
    nfft 512, nfft 1024 and nfft 1024 + 201-node mu grid ("best") settings.
  - On the rungs with nfft >= 512 it peaks at k ~ 0.129 h/Mpc (l = 0, 2) and 0.097 h/Mpc (l = 4), i.e.
    k r_s ~ 13 / 10 with r_s h = 99.08 Mpc/h (at nfft 256 the peaks are at 0.144 / 0.129 / 0.254 h/Mpc). That is the
    amplification expected if a small primal leftover that oscillates with the BAO phase is differentiated along
    the stretch. This is an interpretation, not a measurement.
- **Not the wiggle/no-wiggle split** (scratch job 24845). Along the stretch, the two codes' responses agree to
  8.7e-5 for P_nw and 2.4e-4 for P_w. Sigma^2_BAO and dSigma^2_BAO agree to 4e-6.
- **No setting ps exposes closes it** (job 24919). Closed residual, % of the derivative's scale:

  | l | nfft 256 | nfft 512 | nfft 1024 |
  |---|---|---|---|
  | 0 | 0.283 | 0.158 | 0.196 |
  | 2 | 0.458 | 0.422 | 0.437 |
  | 4 | 1.493 | 0.881 | 0.869 |

  - nfft moves l = 0 and l = 4; l = 2 is insensitive.
  - At nfft 1024 the l = 4 ratio closed/direct falls to 0.478 (0.474 at "best"). That is because the direct
    difference rises (1.546 -> 1.817%) while the closed residual stays flat (0.881 -> 0.869%), not because the
    closure improves.
  - The number of mu points in ps's final Simpson-rule projection onto Legendre multipoles ("Simpson num"), the
    mu grid and rbao change the closed residual by <= 0.017 percentage points.
- **ps 0.4.0's de-wiggling cuts** (DST band 140-210) multiply the residual 6-17x (l = 4: 24.8%, ratio ~ 1;
  job 24919). For derivatives, the 0.5.0 de-wiggling fix matters more than any sampling setting.

**Cost if ignored.**
- Fisher forecasts or Laplace approximations built from one code's derivatives differ from the other code's by up
  to 1.5% of the l = 4 derivative (0.3-0.5% at l = 0, 2) along P_lin directions.
- HMC or NUTS sampling stays exact with either code, because the Metropolis step uses the likelihood value; the
  gradient accuracy only changes the acceptance rate.

**Remaining candidates** (the ladder does not vary them):
- ps's spline of the input P_lin (`get_pk` on the extrapolated 1e-5..1e3 h/Mpc grid) against clax-pt's spline of
  its loop channels at the true k (`_channels_at`);
- O(P_w^2) terms that D's closed form for the IR-resummation scheme leaves out;
- ps's FFTLog range kmin_fft / kmax_fft (`ps_1loop.py:29-30`), which the test bridge does not expose;
- clax-pt numerics: its loop k grid, FFTLog and 40 Gauss-Legendre mu nodes.

**Recommended next step.**
- Ladder kmin_fft / kmax_fft and clax-pt's mu nodes and k grid.
- Add the O(P_w^2) terms to D.
- Until then: run ps at nfft >= 512, and do not mix derivatives from the two codes in one analysis.
- `tests/test_ps1loop_gradients.py::test_cross_code_tangents[<case>-stretch]` and `[<case>-amplitude]` stay red,
  by the user's decision of Oct 12, 2026.

### 3. The AP branch is discontinuous with the no-AP branch at zero AP offset

**Consequence.** As the AP parameters approach the fiducial (alpha_perp, alpha_para -> 1), ps_1loop's multipoles
do not approach its own no-AP multipoles: they jump by 1-3e-4 of max|P_l|. Over the 7 PFS slices this is a
0.15 sigma step (lcdm_fiducial, job 24871). clax-pt's physical change across the same step is 0.002 sigma.
- A pipeline that switches branches at alpha = 1 (as this campaign's bridge does) sees a 0.15 sigma step exactly
  where the posterior peaks for a correct fiducial.
- A pipeline that always uses the AP branch carries the same error, the table's interpolation error, as an
  offset. At alpha = 1 it equals the jump to <= 2.5% (job 24919); it was not measured at other alpha.

It also sets the floor of any AP-on comparison against another code: Delta chi^2 ~ 0.13 for lcdm_fiducial at
cfog = 500, not the 0.014 seen with AP off.

**Mechanism.**
- In the AP branch (`get_pk_ell_ref` -> `get_pkmu_ref`, `ps_1loop.py:320-338`), ps tabulates the full P(k, mu) on
  its FFTLog k grid (nfft points, `:44`) x a 51-node mu grid (`:45`), then remaps it to true coordinates with
  `interpax.interp2d` (`:332`).
- The no-AP branch (`get_pk_ell`, `:308`) evaluates directly.
- The jump is the k-interpolation error of that table. Forcing the AP branch at alpha_perp = alpha_para = 1, where
  the physical response is exactly zero, reproduces it: 1.43e-4 / 1.12e-4 / 2.83e-4, against a zero-offset jump of
  1.43e-4 / 1.13e-4 / 2.82e-4 (job 24919).

**Evidence** (lcdm_fiducial).
- **At AP offsets x 0.001, z = 0.38** (job 24919). ps's jump is 1.43e-4 / 1.13e-4 / 2.82e-4 of max|P_l|, peaking
  at k = 0.191 / 0.108 / 0.191 h/Mpc. clax-pt's physical change is 6.1e-6 / 6.7e-6 / 9.7e-6, peaking at
  k ~ 0.01-0.03 h/Mpc.
- **PFS slices** (Task 8 construction, cfog = 500):
  - Delta chi^2(ps - clax) is 0.128 at offsets x 0.001, against 0.014 with AP off (job 24871).
  - ps's own jump is 0.15 sigma at most (job 24871; 0.151 sigma, Delta chi^2 0.150, in job 24919).
  - The jump does not involve the fingers-of-god term: the Delta chi^2 of the shift is 0.147 at cfog = 500 and
    0.169 at cfog = 0 (job 24871).
- **Ladder** (job 24919).
  - "best" (nfft 1024 + 201-node mu grid) reduces the two measures by different factors:
    - the zero-offset jump 13.7 / 12.6 / 25.4x, from 1.43e-4 / 1.13e-4 / 2.82e-4 to 1.04e-5 / 8.9e-6 / 1.11e-5
      (k = 0.121 / 0.121 / 0.181 h/Mpc);
    - the unit-alpha interpolation error 15.2 / 12.8 / 30.4x, from 1.43e-4 / 1.12e-4 / 2.83e-4 to 9.4e-6 /
      8.8e-6 / 9.3e-6 (same k).
  - The residual at "best" is still interpolation error. Compared with clax-pt's physical change: the
    unit-alpha error is 1.0-1.5x clax-pt's maximum, which sits at k ~ 0.01-0.03 h/Mpc, and on the PFS slices the
    remaining ps jump is 0.006 sigma against clax-pt's own 0.002 sigma (about 3x).
  - The 201-node mu grid alone does nothing. nfft 1024 alone brings l = 0 and l = 2 to the "best" level (still
    ~1e-5). The low-k l = 4 piece (k ~ 0.015 h/Mpc) also needs the mu grid.
  - PFS floor: ps's jump goes from Delta chi^2 0.150 (0.151 sigma) to 4.7e-4 (0.006 sigma) at "best", and to
    4.9e-4 (0.0059 sigma) with nfft 1024 alone. clax-pt's own jump is 7.6e-5 (0.002 sigma).

**Cost if ignored.** A 0.15 sigma step, or offset, in the PFS multipoles at the default settings (lcdm_fiducial,
Delta chi^2 0.15 over 7 slices).

**Recommended setting.** nfft >= 1024 with a 201-node mu grid whenever the AP branch is used. This reduces the
error but does not remove it: about 1e-5 of P_l remains at k ~ 0.12-0.18 h/Mpc (0.006 sigma over PFS).

**Fix candidates (untested).** Evaluate P(k_true, mu_true) directly at the true coordinates, as clax-pt and
CLASS-PT do at their Gauss-Legendre nodes, or refine the table in k.

### 4. The IR-resummation scheme differs from CLASS-PT's (a documented model difference, not a defect)

**Consequence.** ps_1loop and CLASS-PT/clax-pt damp the BAO wiggle inside the one-loop terms differently. The
difference nearly cancels in the monopole and survives at l = 2, 4. On the b1 bG2 l = 2 coefficient it is about
0.8% of the coefficient's scale (omega_cdm_low z = 0.38). It has an exact closed form, so the codes can be compared
term by term, and either scheme can be chosen knowingly.

**Mechanism.**
- ps resums every loop as loop[P_nw] + e^{-x(mu)}(loop[P] - loop[P_nw]), each loop with its own mu-dependent
  damping (`_get_pk_irres_rsd`, `ps_1loop.py:550`).
- CLASS-PT and clax-pt feed the isotropically damped P_nw + e^{-k^2 Sigma^2} P_w into the bias loops, and apply
  the mu-dependent damping p_lo(mu) only to the F_G2 term.
- The closed form ("option (a)", user decision of Oct 10, 2026, `tests/ps1loop_bridge.py`
  `ps_scheme_predicted`):
  - predicted ps - clax = <X_ps>_l - clax_l, with X_ps = C[P_nw] + e^{-x(k_t, mu_t)}(C[P] - C[P_nw]), where
    k_t, mu_t are the true-coordinate wavenumber and angle cosine that the AP map assigns to the observed (k, mu);
  - C are clax-pt's own loop channels with IR resummation off, evaluated with clax-pt's P_nw, Sigma^2 and AP
    geometry;
  - nothing is fitted.

**Evidence.**
- **b1 bG2 l = 2, omega_cdm_low z = 0.38.** ps - clax is measured at 8.49% and predicted at 8.42%. The prediction
  is the omitted projection (finding 5), 8.76%, plus a scheme term of 0.77% of opposite sign. Without the scheme
  term, the residual (1.015%, job 23244) is 89% scheme-shaped (template fit, scratch job 23253).
- **All 13 omitted-projection lines close at all 42 files with the closed form** (job 24110).
  - Worst: b1 bG2 l = 2 at 0.560% (1% gate) and bG2 l = 4 at 0.705% (2% gate).
  - On every line whose measured difference exceeds 0.5%, the residual is <= 0.069 of it.
- **elf** (17d131d, the successor of ps_1loop) uses ps's scheme. The closed form closes all 13 lines for elf fed
  clax-pt's P_nw to <= 0.16% (<= 0.011% at fiducial; job 24077). It is the right closed form for the ps/elf family,
  not a ps-only patch.

**Cost if ignored.** Mistaking a scheme difference for a code bug: the campaign's first closure attempt failed for
that reason (job 23244). Its PFS weight was not isolated. With c = 0 and cfog = 0 in both codes, the total
remaining difference over the 7 slices is Delta chi^2 0.07-0.13 (job 24860), and that includes the scheme term.
This is not a bound: pieces of a quadratic form can cancel.

**Recommended.**
- Document the scheme in ps_1loop.
- Pick one scheme for the PFS analysis.
- Compare the codes with the closed form above.

### 4b. ps_1loop 0.5.0 cannot run nfft = 2048

**Consequence.** ps_1loop's FFTLog convergence cannot be checked beyond nfft = 1024. That matters because the
direct ps - clax differences are not monotonic in nfft at any l. For example, the stretch-derivative direct
difference at l = 4 is 1.979 / 1.546 / 1.817% of the derivative's scale for nfft 256 / 512 / 1024 (job 24919).
The plateau is therefore judged from 512 vs 1024 only.

**Mechanism.** `pt_matrix.get_I` (`pt_matrix.py:6-7`) evaluates a ratio of complex Gamma functions with
`scipy.special.gamma`. At the larger FFTLog frequencies of nfft = 2048 the divide overflows and the PT matrices
go non-finite. ps_1loop itself does not raise: it returns non-finite multipoles, with only numpy RuntimeWarnings
(job 24917, lcdm_fiducial z = 2.2):
- `pt_matrix.py:7: RuntimeWarning: overflow encountered in divide`;
- `pt_matrix.py:7: RuntimeWarning: invalid value encountered in divide`;
- `pt_matrix.py:22: RuntimeWarning: invalid value encountered in multiply`.

The exception in that job, ``ValueError: `y` must contain only finite values.``, came from the campaign's own
harness: a scipy CubicSpline on ps - clax in `tests/pfs_survey.py:82` (`to_centres`).

**Cost if ignored.** None at nfft <= 1024. A user who sets nfft = 2048 gets NaN or inf multipoles with only
warnings, which then propagate silently into a likelihood unless something downstream checks for finite values.

**Fix (suggested, untested).**
- Raise when the PT matrices are non-finite.
- Evaluate the Gamma ratio in log space (`scipy.special.loggamma`) and exponentiate once.

### Other ps_1loop items (spec §9, from reading the code)

- **`rbao` default 110 Mpc/h** (`ps_1loop.py:25`), against r_s(z_d) h = 99.08 Mpc/h at fiducial. Measured
  negligible:
  - Sigma^2_BAO changes 19.2119 -> 19.3901 (Mpc/h)^2 (+0.93%), and P_l moves <= 4.2e-5 locally (job 22452);
  - the z = 2.2 slice Delta chi^2 goes 0.3500 -> 0.3500 (lcdm_fiducial) and 0.6556 -> 0.6545 (w0wa_m07_m10)
    (job 24919);
  - the finding-2 residual changes by <= 0.017 percentage points.

  Passing rbao = r_s(z_d) h is still the consistent choice.
- **`utils_math.legendre(n, x)` returns L_4 for every n not in {0, 2}** (`utils_math.py:6-14`). Today's callers
  pass l in {0, 2, 4} (`ps_1loop.py:314, 344`; `bs_tree.py:493, 524, 669, 710`), so the bug is latent: l = 1, 3
  or 6 would get a wrong answer silently. Fix: raise for unsupported n, or use the general recursion.
- **`get_pk_real` / `get_pk_gm_real` have no IR resummation** (`ps_1loop.py:149-179, 181-258`; comment "no IR
  resummation" at `:150`). The real-space spectra are therefore inconsistent with the IR-resummed multipoles. The
  campaign compared real space with IR off in both codes: <= 0.617% (c0) and <= 0.388% (F_G2 lines), job 24147.
- **`clax_adapter` feeds total-matter P_m, not P_cb** (`clax_adapter.py:118-142`), at `PrecisionParams.fast_cl`
  precision (`:88`), with a default k_max of 1 h/Mpc (`:58-65`). CLASS-PT and clax-pt feed P_cb to the one-loop
  terms. The campaign bypassed the adapter (both codes got CLASS-PT's P_cb), so the size was not measured here.
- **Methods are jitted with `static_argnames=['self']`** (`ps_1loop.py:106` and every jitted method). Changing
  `rbao`, `ks` or `_mu` after the first call is silently ignored. The campaign's bridge builds a fresh model per
  setting and sets `_mu` before the first call.
- **`ps_1loop_lpng.py` (primordial non-Gaussianity) keeps the pre-0.5.0 structure.** It uses one nu = -0.7
  decomposition (`:31`) and the analytic UV add-back (`:268`, `:410`), so defects F1-F4 below still apply there.
  It also looks up `kernel_to_decomp_dict['22_gg']` / `['13_gg']` (`:112, :134`), keys that do not exist, so its
  RSD path cannot run. It was out of scope for the 0.5.0 fix branch by ruling.

### Already fixed in ps_1loop 0.5.0 (MinhMPA/ps_1loop_jax-for-pfs#6), for the record

Found by this campaign's first coefficient run against 0.4.0 (diagnosis jobs 22611, 22618, 22621, 22623):
- **F1.** The RSD 22-type bias kernels were decomposed at nu = -0.7, which adds a spurious constant x k^2 term
  (b2 l = 0: 4.83% -> 0.32% in the diagnosis rung with nu = -1.6; 0.5.0 ships nu = -1, see F7).
- **F2.** The F_G2-type 13 kernels were also at nu = -0.7, plus an analytic UV add-back, 2.1% off ps's own
  real-space F_G2.
- **F3.** The UV add-back was resummed with sigma^2 computed separately from P and from P_nw (b1 l = 4:
  2.6-2.9%).
- **F4.** The RSD matter kernels were at nu = -0.7.
- **F6.** The de-wiggled P_nw kept ~40% of the BAO wiggle. CLASS-PT's de-wiggling was ported.
- **F7.** nu = -1 for the 22-type bias kernels (user decision).

After the fixes, ps's P_nw equals CLASS-PT's to 1.4e-5 (ledger entry of Oct 8, 2026).

---

## Part B: CLASS-PT and clax-pt

### 5. CLASS-PT, and clax-pt with it, omit 13 projections that their own mu-loop computes

**Consequence.** At the PFS bias point the six omitted F_G2 projections, the only ones measured at PFS level,
are negligible. The other seven lines are bounded at coefficient level only (<= 0.024% after closure).
- The six F_G2 projections cost a survey Delta chi^2 of 0.8-2.2e-4 over the 7 slices at k <= 0.2 h/Mpc. That is
  <= 2.3e-5 of the l = 2 scale and <= 1.3e-4 of the l = 4 scale, and <= 0.4% of sigma in any bin (jobs 22659,
  22660, 22662).
- They are small because the F_G2 amplitude A = 2 bG2 + 0.8 bGamma3 = -(2/15)(b1 - 1) is only -0.02 to -0.10 under
  coevolution. Since Delta chi^2 scales as A^2, A would have to reach 68-111x its coevolution value for the survey
  Delta chi^2 to reach 1.
- As individual coefficients they are large: before the closure, bG2 l = 4 differs between the codes by 28.85% of
  its scale (omega_cdm_low z = 0.8, job 24091).

**Mechanism.** CLASS-PT's mu-loop builds the full mu-dependent integrand, but classy adds only some projections
to P_l. Never added:
- six F_G2 lines (bG2 l=4, bGamma3 l=4, b1 bG2 l=2, 4, b1 bGamma3 l=2, 4). These are the l = 2, 4 projections of
  b1 x IFG2 and the l = 4 projection of f mu^2 x IFG2, where IFG2 is the IR-resummed, AP-mapped F_G2 spectrum;
- the l = 2, 4 moments of the isotropic loops b2^2, bG2^2 and b2 bG2. CLASS-PT computes them as rows 42-47
  (`nonlinear_pt.c:5322-5349`) but classy does not use them;
- b1 b2 l = 4: pm[40] P_4_b1b2 is computed but never added to P_4 (`classy.pyx:4925-4931`). pm[41] P_4_b1bG2
  is likewise unused; it is part of the omitted b1 bG2 l = 4 line above.

clax-pt mirrors CLASS-PT; ps_1loop includes these terms. They arise only from the mu-dependent BAO damping and the
AP mapping: with both switched off, the omitted pieces are <= 2.6e-12 (Mpc/h)^3 (same jobs).

**Evidence.**
- Task 5 closes all 13 lines at all 42 files (job 24110).
- The PFS Delta chi^2 of the seven lines outside F_G2 was not measured. Before the closure their ps - clax was
  <= 1.07% of the coefficient's scale (jobs 23193, 24091); after it, <= 0.024% (job 24110).

**Cost if ignored.** For the six F_G2 lines, none for PFS at the coevolution bias point; a fit with bG2 and
bGamma3 far from coevolution would see more. For the other seven lines, the PFS cost was not measured; at
coefficient level they close to <= 0.024% (job 24110).

**Decision and follow-up.** User decision of Oct 7, 2026: keep the CLASS-PT convention in clax-pt. Follow-up, not
urgent: an optional clax-pt flag that projects the full mu-dependent integrand on every l, off by default for
CLASS-PT parity.

### 6. The AP treatment of the k^4 fingers-of-god term is detectable by PFS at BOSS-like amplitudes

**Consequence.** Both codes run at cfog = 500 (Mpc/h)^4 with fixed nuisance parameters. Their multipoles then
differ by Delta chi^2 = 1.55 / 40.6 / 2.53 / 4.45 / 3.15 over the 7 PFS slices (lcdm_fiducial / omega_cdm_low /
massive_nu_015 / massive_nu_030 / w0wa_m07_m10; job 24884). The excess above the 0.05-0.13 floor comes from the AP
mapping of the k^4 term. That floor includes ps's 0.128 AP-branch floor at default resolution (finding 3).

If ps_1loop refits its own (c, cfog), shared by the 7 slices, the difference falls to Delta chi^2 = 0.120 /
0.689 / 0.126 / 0.137 / 0.782, all below 1. The price is a shift in the fitted cfog of -2.1% / +19.9% / -2.9% /
-4.1% / +3.0% (job 24884). So PFS cannot tell the codes apart once cfog floats, but the inferred cfog depends on
the prescription: by up to +20% in a cosmology whose geometry is several percent off the fiducial.

**Mechanism.**
- clax-pt (`clax/ept.py:2441-2446, 2500-2505, 2549-2554`) and CLASS-PT take the l = 4 projection `Pk_ctr4` of
  k^2 P_lin f^2 mu^4 (`ept.py:1214`), multiply it by an isotropic factor, and add it to every multipole in observed
  k^2.
- ps_1loop multiplies the full mu-dependent IR-resummed P(k, mu) by -cfog f^4 mu^4 (b1 + f mu^2)^2 k^4 in true
  coordinates (`get_pkmu_ctr_k4`, `ps_1loop.py:520-541`). This is the exact treatment.

**Evidence.**
- **Per slice** (job 24884), lcdm_fiducial: 0.012 / 0.093 / 0.156 / 0.293 / 0.281 / 0.368 / 0.350 for
  z = 0.7 ... 2.2. The difference grows with z. An interpretation: the growth follows the signal-to-noise on the
  k^4 term. That is not established, because in lcdm_fiducial the AP offsets also grow with z. The w0wa_m07_m10
  z = 0.7 slice, which has large offsets and a small Delta chi^2, has the smallest anisotropic offset of the
  w0wa slices.
- **Which term** (job 24860, at c = 0). With c = 0 and cfog = 0 in both codes, Delta chi^2 is 0.07-0.13 in all
  five cosmologies; at the gate's c = 2 with cfog = 0 it is 0.05-0.09 (scratch, jobs 24867-24868). b2, bG2 and
  bGamma3 change it by < 5%.
- **AP origin** (scratch jobs 24867-24871).
  - With the AP map off in both codes: 0.0137 / 0.0293 / 0.0124 / 0.0105 / 0.0143 at cfog = 500.
  - Swapping AP offsets between lcdm_fiducial and omega_cdm_low moves the excess with them: 1.55 -> 68.3 and
    40.6 -> 1.23.
  - omega_cdm_low's offsets are about 10x lcdm_fiducial's.
  - The isotropic dilation and the anisotropic warping each contribute (lcdm_fiducial 0.47 and 0.65 of 1.55).
- **Crossing cfog**, where the fixed-nuisance Delta chi^2 reaches 1:
  - 384 / 75 / 291 / 213 / 277 (Mpc/h)^4 with the covariance frozen at the gate point (job 24884);
  - 397-401 / 82 / 310 / 234 / 294-297 with the covariance recomputed at each cfog (scratch, jobs 24867-24868).
  - CFOG_REF = 500 is above the crossing in every cosmology.
- **Floor caveat.** The AP-off numbers sit below ps's own AP-branch floor (finding 3): 0.128 for lcdm_fiducial at
  offsets x 0.001 (job 24871).

**Cost if ignored.** With cfog fixed or tightly bounded, a Delta chi^2 of 1.6 (fiducial) to 41 (omega_cdm 20% low)
between two codes that are otherwise equivalent. With cfog free, a cfog posterior shifted by up to 20%.

**Recommended.**
- For PFS, prefer the exact treatment, or give cfog a prior wide enough to absorb the difference.
- For clax-pt, candidate upstream item: an optional exact-AP fingers-of-god term.
- `tests/test_ps1loop_pfs_gate.py::test_pfs_error_bar_gate` stays red by the user's decision of Oct 12, 2026. The
  marginalised Delta chi^2 and the crossing cfog are reported, not asserted.

### 7. clax-pt's RSD one-loop basis was not differentiable along P_lin (fixed)

**Consequence (before the fix).** clax-pt's AD derivatives of P_0,2,4 with respect to the linear spectrum were
wrong at order unity: they differed from clax-pt's own finite differences by ~100% along the stretch and ~90%
along the amplitude (Task 7, Oct 10, 2026; jobs 24165, 24167).

**Mechanism.** clax-pt's `compute_ept` read a frozen (`stop_gradient`) snapshot of the no-wiggle / wiggle split,
so the derivatives lacked the no-wiggle tree response and the one-loop response.

Fixed in MinhMPA/clax-pt#9 and merged into this branch as 74222a6. After the fix, AD vs finite differences agrees
to 1.5e-10 (amplitude) and 2.8e-9 (stretch) over the 5-cosmology grid (job 24208).

Cost if ignored (before the fix): every Fisher matrix, optimiser step or HMC mass matrix built from
`compute_ept_from_clax` multipole gradients was wrong at order unity along P_lin directions. Details are in
CHANGELOG.md, "Oct 10, 2026".

### 8. CLASS-PT's real-space bias loops carry AP, and the items already filed upstream

- **Real-space rows with AP.**
  - Consequence: a real-space P_gg taken from classy or clax-pt at an AP setting is not the real-space spectrum.
    Its b2^2, b2 bG2 and bG2^2 terms are off by up to 12.4%.
  - Mechanism: CLASS-PT stores the isotropic loops Id2d2, Id2G2 and IG2G2 (pk_mult rows 1, 4, 5) as the
    AP-folded l = 0 moment of its mu-loop, and classy's `pk_gg_real` reuses them.
  - Size: 0.680 / 0.575 / 0.584% at lcdm_fiducial z = 0.38 (numpy on the stored rows), up to 12.4% at
    w0wa_m07_m10 z = 0.38 (scratch job 24143). Above 1% at 25 of 28 AP files.
  - clax-pt mirrors this.
  - Recommended: evaluate real space at hratio = Dratio = 1, as this campaign does (Task 6).
- **Already filed upstream** (cited, not restated): smsharma/clax#42 (P_cb and the z = 0 endpoint; part 2 is the
  oscillation at k > 0.2), #43 (PPF dark energy, which CLASS-PT lacks; the w0wa reference that needs it is
  skipped), #44 (the r_s fitting formula).

---

## Test status on the branch (two red tests, by the user's decision)

| test | status | reason | condition for removing the fast-suite deselect |
|---|---|---|---|
| `tests/test_ps1loop_gradients.py::test_cross_code_tangents[<case>-stretch]`, `[<case>-amplitude]` | red: discrimination rule fails (finding 2). Stretch is red in all 3 cases; amplitude only for lcdm_fiducial (l = 4: 0.098% closed vs 0.185% direct). massive_nu_030 and w0wa_m07_m10 pass amplitude (job 24898). The wrapper deselects all six parametrizations | user decision of Oct 12, 2026: keep the rule on all five directions until the remainder is isolated | the stretch/amplitude remainder is attributed and closed (closed <= 0.5 x direct on every line), or the user revises the rule |
| `tests/test_ps1loop_pfs_gate.py::test_pfs_error_bar_gate` (5 cosmologies) | red: fixed-nuisance Delta chi^2 1.55-40.6 >= 1 (finding 6) | user decision of Oct 12, 2026: keep the spec's fixed-nuisance gate red as the finding; report the marginalised Delta chi^2 and the crossing cfog | the fingers-of-god AP treatments are reconciled in one of the codes, or the user changes the gate's nuisance treatment (for example a PFS-motivated cfog or the marginalised Delta chi^2) |

The fast-suite wrapper `slurm/run-ps1loop-fast-suite.sh` deselects exactly these, plus the older
Rosenbrock max-steps failure (CHANGELOG Oct 7, 2026).
