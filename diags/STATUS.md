# Task Status Log

## PR-B (feat/forward-mode-ad)

Task B1: completed — 10f24a6 (RED test: test_find_z_reio_forward_mode_matches_fd)
Task B2: completed — 379fa81 (custom_jvp for _find_z_reio + C_He NaN fix + dkd/dloga fix)
Task B3: completed — 379fa81 (C_He reformulation: inv_A+Lambda/inv_A+Lambda+Rup avoids inf-inf tangent at z~15)
Task B4: completed — 379fa81 (all debug prints removed, changes committed)
Task B5: completed — 41824ab (RED test: test_shoot_fn_forward_mode_matches_fd)
Task B6: completed — cbc727c (custom_jvp for shoot_fn with IFT rule)
Task B7: in progress — push and open PR pending

## PR-C (fix/thermo-remaining-gradients)

Task C4: completed — 9c6af5f (RED tests: kappa_dot, exp_m_kappa, g gradient regressions)
Task C5: completed — 0a29a6d (_kd_safe rescaling for kappa_dot_of_loga + g_prime_grid)
Task C6: completed — 0a29a6d (_kappa_safe rescaling for exp_m_kappa_of_loga and g_of_loga)
Task C7: in progress — pre-push regression gate running
