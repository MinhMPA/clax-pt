#!/bin/bash
# The CLAUDE.md pre-commit gate: pytest tests/ --fast -x -q, with the campaign's documented
# deselect (pre-existing max-steps failure, CHANGELOG Aug 29 and Sep 1, 2026; issue #30 item 5).
# test_cross_code_tangents[<case>-stretch] and [<case>-amplitude]: RED by user decision Oct 12, 2026 (SDD ledger .superpowers/sdd/2026-10-05-clax-pt-vs-ps1loop-validation/progress.md); remove when the stretch-derivative remainder is isolated (Task 9 ladder).
#   Narrowed Oct 13, 2026 (Task 9, controller amendment E): the test is parametrized by tangent, so only the two red tangents are hidden;
#   its f / a_perp / a_para lines and test_primal_closure_guard run. All three cases are listed; under --fast only lcdm_fiducial exists.
# test_pfs_error_bar_gate: RED at CFOG_REF = 500 (Mpc/h)^4, under the user's review Oct 12, 2026 (same SDD ledger); hides only the lcdm_fiducial case under --fast
exec "$(dirname "$0")/run-ps1loop-pytest.sh" tests/ --fast -x -q -p no:cacheprovider \
  --deselect tests/test_solver_selection.py::TestRosenbrockPk::test_pk_rosenbrock_vs_kvaerno5 \
  --deselect 'tests/test_ps1loop_gradients.py::test_cross_code_tangents[lcdm_fiducial-stretch]' \
  --deselect 'tests/test_ps1loop_gradients.py::test_cross_code_tangents[lcdm_fiducial-amplitude]' \
  --deselect 'tests/test_ps1loop_gradients.py::test_cross_code_tangents[massive_nu_030-stretch]' \
  --deselect 'tests/test_ps1loop_gradients.py::test_cross_code_tangents[massive_nu_030-amplitude]' \
  --deselect 'tests/test_ps1loop_gradients.py::test_cross_code_tangents[w0wa_m07_m10-stretch]' \
  --deselect 'tests/test_ps1loop_gradients.py::test_cross_code_tangents[w0wa_m07_m10-amplitude]' \
  --deselect tests/test_ps1loop_pfs_gate.py::test_pfs_error_bar_gate
