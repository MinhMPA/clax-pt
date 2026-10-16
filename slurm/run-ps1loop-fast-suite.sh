#!/bin/bash
# The CLAUDE.md pre-commit gate: pytest tests/ --fast -x -q, with the campaign's one documented
# deselect: the pre-existing max-steps failure (CHANGELOG Aug 29 and Sep 1, 2026; issue #30 item 5).
# No campaign test is deselected (Oct 14, 2026): the derivative closure passes on every tangent since the
# extended closed form and ps nfft 512, and the PFS error-bar gate is red by the user decision of
# Oct 12, 2026 (fixed-nuisance gate kept red as the finding) and carried as xfail(strict=True) in
# tests/test_ps1loop_pfs_gate.py -- remove that mark when one code's fingers-of-god AP treatment is changed
# to match the other's or the user changes the gate's nuisance treatment (CHANGELOG Oct 13 and Oct 14, 2026;
# docs/ps1loop-findings.md "Test status").
# NB: with -x, a strict XPASS of that gate (the fixed-nuisance Delta chi^2 dropping below 1) aborts this
# suite at that point -- that is the intended alarm, not a harness fault; revisit the xfail mark then.
exec "$(dirname "$0")/run-ps1loop-pytest.sh" tests/ --fast -x -q -p no:cacheprovider \
  --deselect tests/test_solver_selection.py::TestRosenbrockPk::test_pk_rosenbrock_vs_kvaerno5
