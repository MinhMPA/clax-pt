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
