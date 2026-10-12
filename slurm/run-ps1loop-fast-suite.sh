#!/bin/bash
# The CLAUDE.md pre-commit gate: pytest tests/ --fast -x -q, with the campaign's documented
# deselect (pre-existing max-steps failure, CHANGELOG Aug 29 and Sep 1, 2026; issue #30 item 5).
# test_cross_code_tangents: RED by user decision Oct 12, 2026 (SDD ledger .superpowers/sdd/2026-10-05-clax-pt-vs-ps1loop-validation/progress.md): remove when the stretch-derivative remainder is isolated (Task 9)
# test_pfs_error_bar_gate: RED at CFOG_REF = 500 (Mpc/h)^4, under the user's review Oct 12, 2026 (same SDD ledger); hides only the lcdm_fiducial case under --fast
exec "$(dirname "$0")/run-ps1loop-pytest.sh" tests/ --fast -x -q -p no:cacheprovider \
  --deselect tests/test_solver_selection.py::TestRosenbrockPk::test_pk_rosenbrock_vs_kvaerno5 \
  --deselect tests/test_ps1loop_gradients.py::test_cross_code_tangents \
  --deselect tests/test_ps1loop_pfs_gate.py::test_pfs_error_bar_gate
