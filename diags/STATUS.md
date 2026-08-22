# Task Status Log

## PR-B (feat/forward-mode-ad)

Task B1: completed — 10f24a6 (RED test: test_find_z_reio_forward_mode_matches_fd)
Task B2: completed — 379fa81 (custom_jvp for _find_z_reio + C_He NaN fix + dkd/dloga fix)
Task B3: completed — 379fa81 (C_He reformulation: inv_A+Lambda/inv_A+Lambda+Rup avoids inf-inf tangent at z~15)
Task B4: completed — 379fa81 (all debug prints removed, changes committed)
Task B5: completed — 41824ab (RED test: test_shoot_fn_forward_mode_matches_fd)
Task B6: completed — cbc727c (custom_jvp for shoot_fn with IFT rule)
Task B7: completed — branch pushed to origin/feat/forward-mode-ad
  BLOCKER: gh CLI not available on this node; PR must be opened manually.
  PR URL: https://github.com/MinhMPA/clax-pt/pull/new/feat/forward-mode-ad
  PR title: "fix: forward-mode AD for z_reio and shoot_fn"

## PR-C (fix/thermo-remaining-gradients)

Task C4: completed — 9c6af5f (RED tests: kappa_dot, exp_m_kappa, g gradient regressions)
Task C5: completed — 0a29a6d (_kd_safe rescaling for kappa_dot_of_loga + g_prime_grid)
Task C6: completed — 0a29a6d (_kappa_safe rescaling for exp_m_kappa_of_loga and g_of_loga)
Task C7: completed — branch pushed to origin/fix/thermo-remaining-gradients
  Gate background-test failures were transient JAX CUDA graph OOM (75-min run);
  15/15 background tests confirmed passing in clean isolated run after push.
  PR must be opened manually (gh CLI unavailable on compute node):
  MinhMPA:fix/thermo-remaining-gradients → smsharma/clax:fix/kd-dloga-gradient
  PR title: "fix(thermodynamics): AD-safe splines for kappa_dot, exp_m_kappa, g"

---

## PR bodies for manual creation

### PR-B (feat/forward-mode-ad → main)

Converts two custom_vjp functions to custom_jvp so jax.jvp works through the full pipeline:

- _find_z_reio (thermodynamics.py): IFT JVP rule. Fixes inf-inf NaN in He C_He
  Boltzmann tangent at low T (z~15) by reformulating C_He = (inv_A+L)/(inv_A+L+R).
- shoot_fn (shooting.py): IFT JVP rule dh/dtheta_s = 1/(dtheta_s/dh).
  custom_jvp also provides VJP via transposition — existing gradient tests unchanged.

Also includes dkappa_dot_dloga_of_loga stable-gradient fix from PR-A.

Tests: test_thermodynamics.py 10/10, test_shooting.py 7/7.

### PR-C (fix/thermo-remaining-gradients → fix/kd-dloga-gradient)

Builds on PR-A. Applies n_H_0 rescaling to three remaining splines that carry
the accumulated Friedmann-scan gradient (~10^8x FD blowup):

- kappa_dot_of_loga: exact gradient at fixed x_e
- exp_m_kappa_of_loga: correct kappa-path gradient
- g_of_loga: product — gives d(g)/d(omega_b) = g*(1-kappa)/omega_b

Accuracy: exact where x_e~const (loga<-8). Near recombination (loga~-7),
10-30% residual from d(xe)/d(omega_b) — still finite vs prior 10^8x blowup.

Tests: 3 new gradient regression tests GREEN; 10/10 thermodynamics tests pass.


## PR-D (fix/ad-correctness-clax-pt → benchmark/clax-pt)

Port of PR-A + PR-B + PR-C AD correctness fixes to the benchmark/clax-pt branch.

### Changes applied

**clax/thermodynamics.py**
- Fix 1 (C_He stable form): replaced inf-inf cancellation with inv_A form
  `_inv_A_He = 1/(K_He * n_1s_He * B_He); C_He = (inv_A+L)/(inv_A+L+R)`
  Prevents NaN in JVP tangent at z~15 where B_He → ∞.
- Fix 2 (n_H_0 rescaling): `_kd_safe = sg(kappa_dot) * (n_H_0 / sg(n_H_0))`
  Stops the accumulated Friedmann-scan eigenvalue (~10^12x blowup for kappa_dot).
  Applied to kappa_dot_of_loga spline AND dkappa_dot_dloga_of_loga spline.
- Fix 3 (_find_z_reio custom_vjp → custom_jvp): IFT JVP rule enables forward-mode AD.
  dz_reio = -dF/dinputs / dF/dz via jax.jvp on the residual.

**clax/shooting.py**
- Fix 4 (shoot_fn custom_vjp → custom_jvp): IFT JVP rule.
  dh/dtheta_s = 1 / (dtheta_s/dh) via jax.grad on _compute_theta_s.

**tests/test_thermodynamics.py**
- Added TestThermoGradients::test_kappa_dot_gradient_matches_fd_for_omega_b
- Added TestThermoGradients::test_exp_m_kappa_gradient_matches_fd_for_omega_b (loga=-7.0)
- Added TestThermoGradients::test_g_gradient_matches_fd_for_omega_b (loga=-7.0)
- Added _PREC_JVP (ode_adjoint="direct") and _thermo_jvp_fd_pair helper
- Added test_find_z_reio_forward_mode_matches_fd
- Added TestThermoForwardModeAD (kappa_dot, exp_m_kappa, g JVP tests)
  Note: clax-pt's background.py wires prec.ode_adjoint, so NO xfail needed.

**tests/test_shooting.py**
- Added TestShootingForwardModeAD::test_shoot_fn_forward_mode_matches_fd
  Uses default PREC (no ode_adjoint="direct" needed — custom_jvp boundary
  intercepts forward-mode; inner jax.grad call is standalone reverse-mode).

### Test results (2026-05-10)

Task D-test-1: completed — b7yit46ro — 6/6 targeted JVP+kappa_dot tests GREEN on GPU 3
  Tested: test_find_z_reio_forward_mode_matches_fd, TestThermoForwardModeAD (3),
          test_kappa_dot_gradient_matches_fd_for_omega_b,
          test_opacity_logderivative_gradient_matches_fd_for_omega_b

Task D-test-2: completed — bpgma4gft — 8/8 test_shooting.py GREEN on GPU 3 (2:19:51)

Task D-test-3: completed — bsu7m7ji0 — 24/24 (thermodynamics 16/16 + shooting 8/8) GREEN on GPU 2 (2:12:08)

Task D-test-4: completed — bswg59lin — 1/1 TestShootingForwardModeAD GREEN on GPU 2 (0:47:02)

Task D-test-5: completed — blq88d375 — 7 passed / 1 failed (expected: ran without _PREC_JVP, old test snapshot)
  The failure (test_find_z_reio_forward_mode_matches_fd) is expected; confirmed GREEN in b7yit46ro.

Task D-commit: completed — fa3f878 — all 5 files committed, branch pushed to origin/fix/ad-correctness-clax-pt
  Branch URL: https://github.com/MinhMPA/clax-pt/tree/fix/ad-correctness-clax-pt
  PR must be opened manually (gh CLI not authenticated on compute node):
    URL: https://github.com/MinhMPA/clax-pt/pull/new/fix/ad-correctness-clax-pt
    Base: benchmark/clax-pt  Head: fix/ad-correctness-clax-pt
    Title: "fix(ad): port PR-A/B/C AD correctness fixes to benchmark/clax-pt"

### Key differences from main clax

- clax-pt background.py DOES wire prec.ode_adjoint (line 628); ode_adjoint="direct"
  enables full forward-mode AD through the background ODE.
- clax-pt has _solve_hydrogen_saha with custom_jvp (already forward-mode compatible).
- clax-pt _first_derivative_table uses finite differences (not CubicSpline.derivative).
  The n_H_0 rescaling is applied by passing _kd_safe to _first_derivative_table.
