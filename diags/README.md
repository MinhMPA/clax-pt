# GPU Diagnostic Scripts

Reusable diagnostic scripts originally created during Bridges-2 GPU debugging sessions.
These are **not** part of the test suite — they're standalone scripts for targeted
investigation of accuracy bottlenecks.

Run on a GPU node:
```bash
python diags/diag_cl_comprehensive.py
```

## Scripts

| Script | Purpose |
|--------|---------|
| `diag_cl_comprehensive.py` | Dense l-sampling + n_k_fine convergence test |
| `diag_cl_fast_v2.py` | Apples-to-apples C_l comparison (massless ncdm + RECFAST) |
| `diag_class_xe_oracle.py` | Inject CLASS-exact x_e to isolate error sources |
| `diag_pert_vars.py` | Compare raw perturbation variables against CLASS at specific k |
| `diag_source_decomp_v2.py` | Decompose TT error into SW+Doppler vs ISW contributions |
| `timing_test.py` | JIT compilation and execution timing benchmarks |

## AD diagnostics (May 2026 investigation, PR #16)

| Script | Purpose |
|---|---|
| `diag_grad_tau_end.py` | Empirically confirm `RecursiveCheckpointAdjoint+PIDController` blocks traced `t1` |
| `diag_grad_taylor.py` | First-pass Taylor correction with f·aH approximation (~6% residual) |
| `diag_grad_exact.py` | Exact-RHS Taylor via `jax.jvp` (in-tree fix) |
| `diag_grad_bg_conformal_age.py` | Decisive: `jax.grad(bg.conformal_age)` is exact (rules out bg AD bug) |
| `diag_grad_fd_step_and_jvp.py` | FD step sensitivity sweep + jvp attempt (custom_vjp blocker surfaced) |
| `diag_grad_tight_rtol.py` | rtol=1e-5 verification (1.08% / 3.18% / 0.85%) |
| `diag_grad_rtol_1em7.py` | rtol=1e-7 final precision floor (0.32% / 1.23% / 0.61%) |
| `diag_grad_jvp_direct.py` | jvp end-to-end attempt with DirectAdjoint (NaN on density params) |
| `diag_jvp_bisect.py` | Localize NaN to thermodynamics `_find_z_reio` |
| `diag_jvp_find_z_reio.py` | Direct test of `_find_z_reio` JVP rule |
