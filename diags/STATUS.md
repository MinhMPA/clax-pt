# Task Status Log — benchmark/clax-pt

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

### Key differences from main clax

- clax-pt background.py DOES wire prec.ode_adjoint (line 628); ode_adjoint="direct"
  enables full forward-mode AD through the background ODE.
- clax-pt has _solve_hydrogen_saha with custom_jvp (already forward-mode compatible).
- clax-pt _first_derivative_table uses finite differences (not CubicSpline.derivative).
  The n_H_0 rescaling is applied by passing _kd_safe to _first_derivative_table.
